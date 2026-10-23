"""Tarefa F, item 3: conjunto rotulado."""
import json
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import f_rotulados as fr  # noqa: E402
from f_comum import ROTULOS, assinatura, sn, status_verificador, valida_nucleo  # noqa: E402

F = RAIZ / "outputs" / "melhoria_v8" / "F"


class TestLogica(unittest.TestCase):
    def test_sn(self):
        self.assertIs(sn("s"), True)
        self.assertIs(sn(" Sim "), True)
        self.assertIs(sn("n"), False)
        self.assertIs(sn("não"), False)
        self.assertIsNone(sn(""))
        self.assertIsNone(sn("talvez"))

    def test_valida_nucleo(self):
        ok = {"correcao_matematica": True, "resposta_unica": True, "dados_suficientes": True}
        self.assertTrue(valida_nucleo(ok))
        self.assertFalse(valida_nucleo({**ok, "resposta_unica": False}))
        self.assertIsNone(valida_nucleo({**ok, "dados_suficientes": None}))

    def test_assinatura_ignora_caixa_acento_espaco(self):
        a = {"enunciado": "Quanto É 2 + 2?", "alternativas": {"A": "4", "B": "5"}}
        b = {"enunciado": "quanto e  2 + 2?", "alternativas": {"A": "4", "B": "5"}}
        c = {"enunciado": "Quanto é 2 + 3?", "alternativas": {"A": "4", "B": "5"}}
        self.assertEqual(assinatura(a), assinatura(b))
        self.assertNotEqual(assinatura(a), assinatura(c))

    def test_status_verificador_classes(self):
        q = {"enunciado": "x", "alternativas": {"A": "5", "B": "6", "C": "7", "D": "8", "E": "9"},
             "resolucao_passo_a_passo": "Somando, 2 + 3 = 5. A resposta é 5.", "resposta_correta": "A"}
        self.assertEqual(status_verificador(q), "ok")
        q2 = {**q, "resolucao_passo_a_passo": "Pense bem no problema."}
        self.assertEqual(status_verificador(q2), "nao_verificavel")
        self.assertTrue(status_verificador({**q, "resposta_correta": "B"}).startswith("inconsistente"))

    def test_marcar_duplicados(self):
        it = [{"uid": "a:1", "sig": "x", "valida_nucleo": True},
              {"uid": "b:7", "sig": "y", "valida_nucleo": True},
              {"uid": "c:2", "sig": "x", "valida_nucleo": False}]
        fr.marcar_duplicados(it)
        self.assertIsNone(it[0]["dup_de"])
        self.assertEqual(it[2]["dup_de"], "a:1")
        self.assertTrue(it[2]["conflito_rotulo"])
        self.assertFalse(it[1]["conflito_rotulo"])


@unittest.skipUnless((F / "rotulados.jsonl").exists(), "rode src/f_rotulados.py")
class TestSaida(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.itens = [json.loads(l) for l in (F / "rotulados.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.rel = json.loads((F / "rotulados_relatorio.json").read_text(encoding="utf-8"))

    def test_campos_e_uid_unico(self):
        uids = [i["uid"] for i in self.itens]
        self.assertEqual(len(uids), len(set(uids)))
        for i in self.itens:
            for k in ("enunciado", "alternativas", "resolucao", "gabarito_modelo", "ano", "habilidade",
                      "dificuldade_pedida", "braco", "familia", "amostra", "status_verificador", *ROTULOS):
                self.assertIn(k, i)
            self.assertEqual(set(i["alternativas"]), set("ABCDE"))
            self.assertEqual(i["valida_nucleo"], i["correcao_matematica"] and i["resposta_unica"] and i["dados_suficientes"])

    def test_cada_amostra_reproduz_a_apuracao_do_professor(self):
        for f in self.rel["fontes"]:
            self.assertEqual(f["status"], "ok", f)
            self.assertIn(f.get("reproduz_apuracao_do_professor"), (True, None), f)

    def test_nada_foi_descartado_em_silencio(self):
        total = sum(f["n_linhas_folha"] for f in self.rel["fontes"] if f["status"] == "ok")
        self.assertEqual(total, len(self.itens) + len(self.rel["nao_juntados"]))


if __name__ == "__main__":
    unittest.main()
