"""Tarefa F, item 4: lógica pura do experimento Jev (sem rede, sem chave)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import f_jev as fj  # noqa: E402

ITEM = {"uid": "a:1", "sig": "s1", "familia": "qwen_4b_ft", "enunciado": "Ana tem 12 bolas e ganhou 5.",
        "alternativas": {"A": "7", "B": "17", "C": "12", "D": "60", "E": "Nenhuma"},
        "resolucao": "12 + 5 = 17.", "gabarito_modelo": "B", "valida_nucleo": True}


class TestCorpo(unittest.TestCase):
    def test_estado_completo_tem_campos_nomeados_e_texto_do_gabarito(self):
        e = fj.estado_do_item(ITEM, "completo")
        self.assertEqual(set(e), {"enunciado", "alternativas", "resolucao", "resposta_marcada"})
        self.assertEqual(e["resposta_marcada"], {"letra": "B", "texto": "17"})

    def test_estado_enunciado_nao_vaza_resolucao_nem_gabarito(self):
        e = fj.estado_do_item(ITEM, "enunciado")
        self.assertEqual(set(e), {"enunciado", "alternativas"})

    def test_corpo_tem_uma_pergunta_noul_por_defeito_com_criterios(self):
        c = fj.corpo_requisicao(ITEM, "completo")
        self.assertEqual(c["model"], "typesafe/jev-1.13")
        self.assertEqual(len(c["questions"]), 6)
        for q in c["questions"].values():
            self.assertEqual(q["type"], "noul")
            self.assertEqual(set(q["criteria"]), {"true", "false"})

    def test_variante_enunciado_so_tem_perguntas_que_nao_precisam_da_resolucao(self):
        c = fj.corpo_requisicao(ITEM, "enunciado")
        self.assertEqual(set(c["questions"]), {"dados_suficientes", "resposta_unica",
                                               "enunciado_contraditorio", "referencia_a_figura_ausente"})

    def test_instrucoes_sem_dupla_negacao_simples(self):
        for p in fj.PERGUNTAS.values():
            self.assertNotIn(" not not ", p["instructions"].lower())
            self.assertTrue(p["instructions"].rstrip().endswith("?"))


class TestPontuacao(unittest.TestCase):
    def test_inverte_perguntas_de_defeito(self):
        p = {"dados_suficientes": 0.9, "enunciado_contraditorio": 0.8}
        self.assertAlmostEqual(fj.pontuacao_de_validade(p, "dados_suficientes"), 0.9)
        self.assertAlmostEqual(fj.pontuacao_de_validade(p, "enunciado_contraditorio"), 0.2)

    def test_combinar_min(self):
        p = {"dados_suficientes": 0.9, "resposta_unica": 0.4, "enunciado_contraditorio": 0.1}
        self.assertAlmostEqual(fj.combinar_min(p, ["dados_suficientes", "resposta_unica"]), 0.4)
        self.assertAlmostEqual(fj.combinar_min(p, ["dados_suficientes", "enunciado_contraditorio"]), 0.9)
        self.assertIsNone(fj.combinar_min(p, ["gabarito_coerente_com_resolucao"]))


class TestContador(unittest.TestCase):
    def test_aborta_antes_de_passar_do_teto_e_persiste(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fj.Contador(Path(tmp) / "c.json", teto=3, teto_usd=1.0)
            for _ in range(3):
                c.reservar()
            with self.assertRaises(fj.TetoExcedido):
                c.reservar()
            self.assertEqual(json.loads((Path(tmp) / "c.json").read_text())["requisicoes"], 3)
            # reabrir continua de onde parou
            c2 = fj.Contador(Path(tmp) / "c.json", teto=3, teto_usd=1.0)
            with self.assertRaises(fj.TetoExcedido):
                c2.reservar()

    def test_teto_de_gasto(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fj.Contador(Path(tmp) / "c.json", teto=1000, teto_usd=fj.TARIFA_CONSERVADORA * 2.5)
            c.reservar()
            c.reservar()
            with self.assertRaises(fj.TetoExcedido):
                c.reservar()

    def test_liquidar_troca_a_reserva_pelo_custo_real(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fj.Contador(Path(tmp) / "c.json", teto=10, teto_usd=1.0)
            c.reservar()
            c.liquidar(0.00002)
            self.assertAlmostEqual(c.gasto, 0.00002)

    def test_cada_tentativa_http_conta_inclusive_as_que_falham(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fj.Contador(Path(tmp) / "c.json", teto=2, teto_usd=1.0)
            erro = fj.urllib.error.HTTPError("u", 503, "x", {}, None)
            with mock.patch.object(fj.urllib.request, "urlopen", side_effect=erro), \
                    mock.patch.object(fj.time, "sleep"):
                with self.assertRaises(fj.TetoExcedido):
                    fj.chamar({"state": {}}, "CHAVE-FALSA", c, tentativas=5)
            self.assertEqual(c.requisicoes, 2)

    def test_erro_nao_vaza_a_chave(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fj.Contador(Path(tmp) / "c.json", teto=5, teto_usd=1.0)
            with mock.patch.object(fj.urllib.request, "urlopen", side_effect=RuntimeError("CHAVE-FALSA")), \
                    mock.patch.object(fj.time, "sleep"):
                r = fj.chamar({}, "CHAVE-FALSA", c, tentativas=1)
            self.assertNotIn("CHAVE-FALSA", json.dumps(r))


class TestPlanoEPrefiltro(unittest.TestCase):
    def test_plano_pula_o_que_ja_foi_respondido_e_so_gera_enunciado_para_gerados(self):
        a = {**ITEM, "uid": "a:1", "sig": "s1", "familia": "qwen_4b_ft"}
        b = {**ITEM, "uid": "b:1", "sig": "s2", "familia": "base_corpus"}
        p = fj.plano([a, b], {("s1", "completo")})
        self.assertEqual([(i["sig"], v) for i, v in p], [("s2", "completo"), ("s1", "enunciado")])

    def test_tabela_prefiltro(self):
        scores = [0.0, 0.05, 0.9, 0.95, 0.1]
        y = [False, False, True, True, True]  # 2 inválidos, 3 válidos
        t = fj.tabela_prefiltro(scores, y, limiares=(0.08, 0.2))
        self.assertEqual(t[0]["n_reprovados"], 2)
        self.assertEqual(t[0]["precisao_invalido"], 1.0)
        self.assertEqual(t[0]["recall_invalido"], 1.0)
        self.assertEqual(t[1]["n_reprovados"], 3)
        self.assertAlmostEqual(t[1]["precisao_invalido"], 0.667, places=3)
        self.assertAlmostEqual(t[1]["validos_perdidos_pct"], 33.3, places=1)


if __name__ == "__main__":
    unittest.main()
