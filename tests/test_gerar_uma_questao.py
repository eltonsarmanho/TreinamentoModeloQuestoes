import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import gerar_uma_questao as g


class TestHistoricoUmaQuestao(unittest.TestCase):
    def test_filtra_por_ano_e_habilidade(self):
        h = [{"ano": "5º", "habilidade": "H08", "subtema": "a"}, {"ano": "5º", "habilidade": "H09", "subtema": "b"},
             {"ano": "9º", "habilidade": "H08", "subtema": "c"}]
        self.assertEqual([x["subtema"] for x in g.do_par(h, "5º", "H08")], ["a"])

    def test_registrar_acrescenta_e_apara_so_o_mesmo_par(self):
        h = [{"ano": "5º", "habilidade": "H09", "subtema": "x"}]
        for i in range(g.MAX_HISTORICO + 5):
            h = g.registrar(h, "5º", "H08", {"subtema": f"s{i}", "contexto": "feira"})
        mesmo = g.do_par(h, "5º", "H08")
        self.assertEqual(len(mesmo), g.MAX_HISTORICO)
        self.assertEqual(mesmo[-1]["subtema"], f"s{g.MAX_HISTORICO + 4}")  # fica o mais recente
        self.assertEqual(len(g.do_par(h, "5º", "H09")), 1)

    def test_historico_ausente_ou_corrompido_vira_vazio(self):
        self.assertEqual(g.ler_historico("/nao/existe.json"), [])


if __name__ == "__main__":
    unittest.main()
