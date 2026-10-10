"""Auditoria de gabaritos em camadas e resolução cega dos fracos (sem rede)."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import auditar_gabaritos_base as ag  # noqa: E402
import reverificar_gabaritos as rg  # noqa: E402


def q(enun, alts, gab, resol):
    return {"enunciado": enun, "alternativas": dict(zip("ABCDE", alts)), "resposta_correta": gab,
            "resolucao_passo_a_passo": resol, "difficulty": "EASY"}


class TestClassificar(unittest.TestCase):
    def test_conta_certa_e_confirmada(self):
        x = q("Pedro tinha 25 figurinhas e comprou 17. Quantas tem?", ["38", "42", "43", "44", "45"], "B", "25 + 17 = 42.")
        self.assertEqual(ag.classificar(x, "DIST-H05-Fácil-00001", {}, {})[0], "CONFIRMADO")

    def test_resultado_fora_das_alternativas_e_refutado(self):
        x = q("Variação de 120 para 80 camisetas?", ["100%", "20%", "25%", "50%", "120%"], "C", "A variação foi de 40 camisetas. 40/120=1/3≈33.3%, ou 33.3% ≈ 33%. Portanto, a resposta é 33%.")
        self.assertEqual(ag.classificar(x, "DIST-H03-Moderado-00002", {}, {})[0], "REFUTADO")

    def test_gabarito_diferente_do_banco_e_refutado(self):
        x = q("Qual a cor?", ["a", "b", "c", "d", "e"], "A", "")
        self.assertEqual(ag.classificar(x, "MT1", {"MT1": ("B", "B")}, {})[0], "REFUTADO")

    def test_gabarito_comparado_por_conteudo_depois_de_permutar(self):
        # banco: B = "4" é a correta. No treino a mesma alternativa foi movida para a letra D.
        x = q("Quanto é 2 + 2?", ["3", "5", "6", "4", "7"], "D", "")
        db = ("B", "B", {"A": "3", "B": "4", "C": "5", "D": "6"})
        self.assertNotEqual(ag.classificar(x, "MT1", {"MT1": db}, {})[0], "REFUTADO")
        errado = q("Quanto é 2 + 2?", ["3", "5", "6", "4", "7"], "A", "")
        self.assertEqual(ag.classificar(errado, "MT1", {"MT1": db}, {})[0], "REFUTADO")

    def test_justificativa_oficial_confirma(self):
        x = q("Qual a cor?", ["a", "b", "c", "d", "e"], "B", "")
        self.assertEqual(ag.classificar(x, "MT1", {"MT1": ("B", "B")}, {})[0], "CONFIRMADO")

    def test_juizes_e_fraco(self):
        x = q("Qual a cor?", ["a", "b", "c", "d", "e"], "B", "")
        self.assertEqual(ag.classificar(x, "DIST-X", {}, {"DIST-X": {"confianca": "alta"}})[0], "JUIZES")
        self.assertEqual(ag.classificar(x, "DIST-X", {}, {"DIST-X": {"confianca": "baixa"}})[0], "FRACO")
        self.assertEqual(ag.classificar(x, "DIST-Y", {}, {})[0], "FRACO")
        self.assertEqual(ag.classificar(x, "INJ-1-H01-F-1", {}, {})[0], "JUIZES")

    def test_resolucao_cega_confirma_o_que_os_juizes_deixaram_fraco(self):
        x = q("Qual a cor?", ["a", "b", "c", "d", "e"], "B", "")
        self.assertEqual(ag.classificar(x, "DIST-X", {}, {"DIST-X": {"confianca": "baixa"}}, {"DIST-X"})[0], "CEGO")
        self.assertEqual(ag.classificar(x, "DIST-X", {}, {"DIST-X": {"confianca": "baixa"}}, set())[0], "FRACO")

    def test_origem(self):
        self.assertEqual([ag.origem(c) for c in ("DIST-a", "INJ-a", "SINT-a", "EF01MA02-046-L2-2026-09", "MT9042MH09MT")],
                         ["destilado", "injetado", "sintetico", "real_L2", "real_banco"])


class FakeCliente:
    def __init__(self, f):
        self.f, self.n = f, 0

    def chat_completion(self, messages, max_tokens, temperature):
        self.n += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.f(messages[-1]["content"])))])


Q = q("Quanto é 2 + 2?", ["3", "4", "5", "6", "7"], "B", "2 + 2 = 4")


def resp_correta(pergunta):
    """Resolve de verdade: acha a linha cujo texto é '4' e devolve a letra exibida."""
    for linha in pergunta.splitlines():
        if linha[3:].strip() == "4":
            return json.dumps({"resposta": linha[0], "motivo": "2+2=4"})
    return json.dumps({"resposta": "X", "motivo": ""})


class TestResolucaoCega(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.p = mock.patch.object(rg, "CACHE", Path(self.tmp.name) / "cache.jsonl")
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def test_permutacoes_diferentes_e_gabarito_nao_aparece_na_pergunta(self):
        p1, p2 = rg.permutacoes(Q, 0)
        self.assertNotEqual(p1, p2)
        txt = rg.montar_pergunta(Q, p2)
        self.assertNotIn("resposta_correta", txt)
        self.assertNotIn("2 + 2 = 4", txt)

    def test_confirmado_proposta_e_inconclusivo(self):
        self.assertEqual(rg.Resolvedor("m", FakeCliente(resp_correta), 10).resolver(Q, 0)[0], "confirmado_cego")
        errado = dict(Q, resposta_correta="A")
        self.assertEqual(rg.Resolvedor("m", FakeCliente(resp_correta), 10).resolver(errado, 0)[0], "proposta_trocar")
        self.assertEqual(rg.Resolvedor("outro-modelo", FakeCliente(lambda p: '{"resposta": "X"}'), 10).resolver(Q, 0)[0], "inconclusivo")

    def test_teto_duro_de_chamadas(self):
        c = FakeCliente(resp_correta)
        classe, _ = rg.Resolvedor("m", c, 1).resolver(Q, 0)
        self.assertEqual(c.n, 1)
        self.assertEqual(classe, "inconclusivo")

    def test_cache_evita_chamada_repetida(self):
        c = FakeCliente(resp_correta)
        rg.Resolvedor("m", c, 10).resolver(Q, 0)
        n1 = c.n
        rg.Resolvedor("m", c, 10).resolver(Q, 0)
        self.assertEqual(c.n, n1)


if __name__ == "__main__":
    unittest.main()
