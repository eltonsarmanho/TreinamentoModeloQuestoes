import json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import juiz_maritaca as jm  # noqa: E402


def resp(status, premissa=True, visual=False, eq=None, calc="C"):
    return json.dumps({"dados": [], "premissa_possivel": premissa, "pergunta": "x", "resolucao": "r",
                       "alternativas": dict(zip("ABCDE", status)), "motivos": {L: "m" for L in "ABCDE"},
                       "equivalentes": eq or [], "dependencia_visual": visual, "resposta_calculada": calc,
                       "problemas": [], "confianca": 0.9})


class T(unittest.TestCase):
    def test_aceita_quando_unica_e_igual_ao_gabarito(self):
        self.assertTrue(jm.decidir(resp("FFVFF"), "C"))

    def test_reprova_gabarito_diferente_multiplas_e_premissa(self):
        self.assertFalse(jm.decidir(resp("FFVFF"), "B"))
        self.assertFalse(jm.decidir(resp("FVVFF", calc=None), "C"))
        self.assertFalse(jm.decidir(resp("FFVFF", premissa=False), "C"))
        self.assertFalse(jm.decidir(resp("FFVFF", visual=True), "C"))
        self.assertFalse(jm.decidir(resp("FFVFF", eq=[["A", "B"]]), "C"))

    def test_malformado_e_abstencao(self):
        self.assertIsNone(jm.decidir("isto nao e json", "C"))

    def test_estatistica(self):
        self.assertEqual(jm.mcnemar_exato(0, 0), 1.0)
        self.assertAlmostEqual(jm.kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
