"""Ciclo v8: resolução antes das alternativas. Cobre o construtor dos dados e a gramática."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import construir_v8 as c  # noqa: E402
import schema_utils as su  # noqa: E402


def questao(res, alts=None, gab="C"):
    return {"enunciado": "Quanto é 15 + 20?",
            "alternativas": alts or {"A": "30", "B": "25", "C": "35", "D": "45", "E": "Nenhuma das alternativas anteriores"},
            "resolucao_passo_a_passo": res, "resposta_correta": gab, "difficulty": "EASY"}


def exemplo(qs, prompt="Gere 1 questão(ões) de matemática. Ano: 2º ano. Habilidade: H06. Dificuldade: Fácil."):
    return {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": prompt},
                         {"role": "assistant", "content": json.dumps({"questoes": qs}, ensure_ascii=False)}],
            "meta": {"codigo_item": "INJ-2-H06-F-00001"}}


class TestOrdem(unittest.TestCase):
    def test_ordem_das_chaves(self):
        q = c.ordenar_v8(questao("15 + 20 = 35"))
        self.assertEqual(list(q), list(c.ORDEM_V8))
        self.assertLess(list(q).index("resolucao_passo_a_passo"), list(q).index("alternativas"))

    def test_chave_fora_do_contrato_e_recusada(self):
        q = questao("15 + 20 = 35")
        q["extra"] = 1
        with self.assertRaises(KeyError):
            c.ordenar_v8(q)
        q2 = questao("x")
        del q2["difficulty"]
        with self.assertRaises(KeyError):
            c.ordenar_v8(q2)


class TestLimpeza(unittest.TestCase):
    def test_remove_frase_final_que_cita_a_alternativa(self):
        nova, rem = c.limpar_resolucao("Somando, 15 + 20 = 35. Portanto, a alternativa correta é a C.")
        self.assertEqual(nova, "Somando, 15 + 20 = 35.")
        self.assertEqual(len(rem), 1)

    def test_nao_remove_opcoes_de_problema_de_contagem(self):
        res = "Há 3 opções de camisa e 2 de calça. Logo, 3 × 2 = 6 opções de combinação."
        nova, rem = c.limpar_resolucao(res)
        self.assertEqual(nova, res)
        self.assertEqual(rem, [])

    def test_remove_lista_de_marcadores(self):
        res = "7 + 8 = 15. Comparando: A) 14 não; B) 15 sim; C) 16 não. Portanto, 15."
        nova, _ = c.limpar_resolucao(res)
        self.assertNotIn("A)", nova)
        self.assertIn("7 + 8 = 15.", nova)
        self.assertTrue(nova.endswith("Portanto, 15."))

    def test_algebra_com_letra_maiuscula_e_parenteses_fica(self):
        res = "Seja (x + B) o fator. Então 2 × 3 = 6."
        self.assertEqual(c.limpar_resolucao(res)[1], [])

    def test_so_finais_nao_mexe_no_meio(self):
        res = "Entre as opções apresentadas, 1 ÷ 2 = 0,5. Depois 0,5 × 4 = 2. Portanto, a letra C."
        nova, rem = c.limpar_resolucao(res, so_finais=True)
        self.assertEqual(nova, "Entre as opções apresentadas, 1 ÷ 2 = 0,5. Depois 0,5 × 4 = 2.")
        self.assertEqual(len(rem), 1)


class TestTransformacao(unittest.TestCase):
    def test_veredito_do_verificador_nao_muda(self):
        q = questao("Somando as duas quantidades do enunciado: 15 + 20 = 35. Portanto, a alternativa correta é a C.")
        antes = su.check_consistency_detalhado(q)[0]
        nova, info = c.transformar_questao(q)
        self.assertIsNotNone(nova)
        self.assertEqual(su.check_consistency_detalhado(nova)[0], antes)
        self.assertEqual(info["modo"], "completa")
        for k in ("enunciado", "alternativas", "resposta_correta", "difficulty"):
            self.assertEqual(nova[k], q[k])

    def test_resolucao_que_ficaria_curta_demais_e_descartada(self):
        nova, info = c.transformar_questao(questao("15 + 20 = 35. Portanto, a alternativa correta é a C."))
        self.assertIsNone(nova)
        self.assertEqual(info["modo"], "reprovada_na_trava")

    def test_descarta_quando_a_limpeza_mudaria_o_veredito(self):
        # a única conta que sustenta o gabarito está na frase que cita as alternativas
        q = questao("Verificando as alternativas: 15 + 20 = 35.")
        nova, info = c.transformar_questao(q)
        self.assertIsNone(nova)
        self.assertIn("descartada", info)

    def test_item_real_e_mantido_em_vez_de_removido(self):
        q = questao("Verificando as alternativas: 15 + 20 = 35.")
        nova, info = c.transformar_questao(q, preservar=True)
        self.assertIsNotNone(nova)
        self.assertEqual(info["modo"], "mantida_sem_limpeza")
        self.assertEqual(nova["resolucao_passo_a_passo"], q["resolucao_passo_a_passo"])

    def test_multi_questao_vira_um_exemplo_por_questao(self):
        qs = [questao("15 + 20 = 35."), questao("15 + 20 = 35."), questao("15 + 20 = 35.")]
        ex = exemplo(qs, prompt="Gere 3 questão(ões) de matemática. Ano: 2º ano.")
        saida, infos = c.transformar_exemplo(ex)
        self.assertEqual(len(saida), 3)
        self.assertEqual(len(infos), 3)
        for i, s in enumerate(saida, 1):
            self.assertTrue(s["messages"][1]["content"].startswith("Gere 1 questão(ões)"))
            self.assertEqual(len(json.loads(s["messages"][2]["content"])["questoes"]), 1)
            self.assertTrue(s["meta"]["codigo_item"].endswith(f"-q{i}"))

    def test_serializacao_mantem_acentos_e_ordem(self):
        s = c.serializar([c.ordenar_v8(questao("15 + 20 = 35."))])
        self.assertIn("Nenhuma das alternativas anteriores", s)
        self.assertLess(s.index('"resolucao_passo_a_passo"'), s.index('"alternativas"'))

    def test_recusa_escrever_em_arquivo_protegido(self):
        with tempfile.TemporaryDirectory() as d:
            for nome in ("train_curado.jsonl", "train.jsonl", "val_frozen_v1.jsonl", "val_x.jsonl"):
                with self.subTest(nome=nome), self.assertRaises(SystemExit):
                    c.construir(ROOT / "data" / "val_frozen_v1.jsonl", Path(d) / nome,
                                ROOT / "data" / "val_frozen_v1.jsonl", Path(d) / "h.jsonl", Path(d) / "r.json")


class TestGramatica(unittest.TestCase):
    def regra(self, nome):
        txt = (ROOT / "grammars" / nome).read_text(encoding="utf-8")
        linha = next(l for l in txt.splitlines() if l.startswith("questao ::="))
        return re.findall(r'\\"(\w+)\\"', linha)

    def test_ordem_nova_so_troca_resolucao_e_alternativas(self):
        antiga, nova = self.regra("questao.gbnf"), self.regra("questao_v8.gbnf")
        self.assertEqual(antiga, ["enunciado", "alternativas", "resolucao_passo_a_passo", "resposta_correta", "difficulty"])
        self.assertEqual(nova, list(c.ORDEM_V8))
        self.assertEqual(sorted(antiga), sorted(nova))

    def test_resto_da_gramatica_e_identico(self):
        def corpo(nome):
            txt = (ROOT / "grammars" / nome).read_text(encoding="utf-8")
            return txt[txt.index("alts ::="):]
        self.assertEqual(corpo("questao.gbnf"), corpo("questao_v8.gbnf"))


class TestCorpusGerado(unittest.TestCase):
    def test_corpus_v8_se_existir(self):
        p = ROOT / "data" / "train_curado_v8.jsonl"
        if not p.exists():
            self.skipTest("data/train_curado_v8.jsonl ausente neste checkout")
        val = {json.loads(l)["meta"]["codigo_item"]
               for l in (ROOT / "data" / "val_frozen_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        n = 0
        for l in p.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            qs = json.loads(r["messages"][2]["content"])["questoes"]
            self.assertEqual(len(qs), 1)
            self.assertEqual(list(qs[0]), list(c.ORDEM_V8))
            self.assertEqual(list(qs[0]["alternativas"]), list("ABCDE"))
            self.assertNotIn(r["meta"]["codigo_item"], val)
            n += 1
        self.assertGreater(n, 2800)


if __name__ == "__main__":
    unittest.main()
