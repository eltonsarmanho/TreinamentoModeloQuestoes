"""Padronização de CAIXA ALTA: invariante de segurança, fallback determinístico, limites."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import padronizar_enunciado as pe  # noqa: E402


class FakeCliente:
    def __init__(self, resposta):
        self.resposta, self.n = resposta, 0

    def chat_completion(self, messages, max_tokens, temperature):
        self.n += 1
        txt = self.resposta(messages[-1]["content"]) if callable(self.resposta) else self.resposta
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=txt))])


Q = {"enunciado": "JOÃO TEM 12 PECAS DE MADEIRA. QUANTAS PECAS HÁ NO TRIÂNGULO ABC?",
     "alternativas": {"A": "12 PECAS.", "B": "20 PECAS.", "C": "5 PECAS.", "D": "7 PECAS.",
                      "E": "Nenhuma das alternativas anteriores"},
     "resolucao_passo_a_passo": "Correto - 12 + 8 = 20", "resposta_correta": "B", "difficulty": "EASY"}


class TestInvariante(unittest.TestCase):
    def test_so_caixa_e_acento_podem_mudar(self):
        self.assertTrue(pe.respeita_invariante("COMUNITARIA  DE 5 CM", "Comunitária de 5 cm"))
        self.assertFalse(pe.respeita_invariante("TEM 12 PECAS", "Tem 13 peças"))
        self.assertFalse(pe.respeita_invariante("TEM 12 PECAS", "Tem 12 peças a mais"))
        self.assertFalse(pe.respeita_invariante("12 + 8", "12 - 8"))

    def test_deteccao(self):
        self.assertTrue(pe.eh_caixa_alta("QUANTOS ANOS MARIA TEM A MAIS QUE JOÃO?"))
        self.assertFalse(pe.eh_caixa_alta("Maria tem 45 anos e João tem 37 anos."))
        self.assertFalse(pe.eh_caixa_alta("A B C"))  # poucas letras


class TestDeterministico(unittest.TestCase):
    def test_caixa_siglas_unidades_e_nomes(self):
        t = pe.normalizar_deterministico("O TRIÂNGULO ABC TEM LADOS DE 3 CM. JOÃO COMPROU 2 L E 500 ML.",
                                         frozenset({"joão"}))
        self.assertEqual(t, "O triângulo ABC tem lados de 3 cm. João comprou 2 L e 500 mL.")

    def test_nao_maiusculiza_palavra_comum_depois_de_lado(self):
        t = pe.normalizar_deterministico("O LADO DE UM QUADRADO. NO LADO MAIOR. PONTOS A, B E C.")
        self.assertIn("lado de um", t)
        self.assertIn("lado maior", t)
        self.assertTrue(t.endswith("Pontos A, B e C."))


class TestQuestao(unittest.TestCase):
    def test_sem_llm_so_toca_o_que_esta_em_caixa_alta(self):
        novo, log = pe.Padronizador().questao(Q)
        self.assertEqual(log["metodo"], "deterministico")
        self.assertEqual(novo["resposta_correta"], "B")
        self.assertEqual(novo["alternativas"]["E"], Q["alternativas"]["E"])
        self.assertTrue(pe.respeita_invariante(Q["enunciado"], novo["enunciado"]))
        self.assertFalse(pe.eh_caixa_alta(novo["enunciado"]))

    def test_questao_em_caixa_normal_nao_muda(self):
        ok = dict(Q, enunciado="Maria tem 12 peças de madeira.",
                  alternativas={"A": "12 peças", "B": "20 peças", "C": "5", "D": "7", "E": "Nenhuma das alternativas anteriores"})
        novo, log = pe.Padronizador().questao(ok)
        self.assertIsNone(log)
        self.assertEqual(novo, ok)

    def test_llm_valido_e_aceito(self):
        resp = json.dumps({"enunciado": "João tem 12 peças de madeira. Quantas peças há no triângulo ABC?",
                           "alternativas": {"A": "12 peças.", "B": "20 peças.", "C": "5 peças.", "D": "7 peças.",
                                            "E": "Nenhuma das alternativas anteriores"}}, ensure_ascii=False)
        p = pe.Padronizador("sabia-4", FakeCliente(resp), None, max_chamadas=5)
        novo, log = p.questao(Q)
        self.assertEqual(log["metodo"], "sabia-4")
        self.assertIn("peças", novo["enunciado"])

    def test_llm_que_muda_numero_cai_no_deterministico(self):
        resp = json.dumps({"enunciado": "João tem 13 peças de madeira. Quantas peças há no triângulo ABC?",
                           "alternativas": Q["alternativas"]}, ensure_ascii=False)
        p = pe.Padronizador("sabia-4", FakeCliente(resp), None, max_chamadas=5)
        novo, log = p.questao(Q)
        self.assertEqual(log["metodo"], "deterministico")
        self.assertEqual(p.contagem["llm_violou_invariante"], 1)
        self.assertIn("12", novo["enunciado"])

    def test_teto_duro_de_chamadas(self):
        c = FakeCliente("{}")
        p = pe.Padronizador("sabia-4", c, None, max_chamadas=0)
        p.questao(Q)
        self.assertEqual(c.n, 0)
        self.assertEqual(p.contagem["teto_de_chamadas"], 1)


class TestArquivo(unittest.TestCase):
    def _linha(self, q):
        return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "Gere 1 questão(ões)"},
                             {"role": "assistant", "content": json.dumps({"questoes": [q]}, ensure_ascii=False)}],
                "meta": {"codigo_item": "X"}}

    def test_processa_e_preserva_prompt_meta_e_gabarito(self):
        with tempfile.TemporaryDirectory() as d:
            e, s = Path(d) / "in.jsonl", Path(d) / "out.jsonl"
            e.write_text(json.dumps(self._linha(Q), ensure_ascii=False) + "\n", encoding="utf-8")
            r = pe.processar(e, s, pe.Padronizador())
            self.assertEqual(r["questoes_alteradas"], 1)
            out = json.loads(s.read_text(encoding="utf-8"))
            self.assertEqual(out["messages"][1]["content"], "Gere 1 questão(ões)")
            self.assertEqual(out["meta"], {"codigo_item": "X"})
            self.assertEqual(json.loads(out["messages"][2]["content"])["questoes"][0]["resposta_correta"], "B")

    def test_recusa_sobrescrever_arquivos_protegidos(self):
        with tempfile.TemporaryDirectory() as d:
            e = Path(d) / "in.jsonl"
            e.write_text(json.dumps(self._linha(Q), ensure_ascii=False) + "\n", encoding="utf-8")
            for nome in ("train_curado.jsonl", "train.jsonl", "val_frozen_v1.jsonl"):
                with self.assertRaises(SystemExit):
                    pe.processar(e, Path(d) / nome, pe.Padronizador())
            with self.assertRaises(SystemExit):
                pe.processar(e, e, pe.Padronizador())


if __name__ == "__main__":
    unittest.main()
