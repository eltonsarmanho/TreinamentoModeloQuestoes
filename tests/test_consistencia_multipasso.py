"""check_consistency em resoluções de vários passos (casos reais do sabia-4-thinking)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from schema_utils import check_consistency, fix_gabarito  # noqa: E402


def _q(res, alts, gab):
    return {"enunciado": "x", "alternativas": dict(zip("ABCDE", alts)),
            "resolucao_passo_a_passo": res, "resposta_correta": gab, "difficulty": "EASY"}


class TestMultiPasso(unittest.TestCase):
    def test_resposta_na_ultima_conta(self):
        q = _q("Primeiro, 6 + 7 = 13. Depois, 13 + 5 = 18.", ["12", "13", "18", "19", "20"], "C")
        self.assertEqual(check_consistency(q), (True, None))
        self.assertEqual(fix_gabarito(q)[1], "ok")  # não troca C pelo passo intermediário

    def test_erro_real_ainda_detectado_com_sugestao_final(self):
        q = _q("Primeiro, 6 + 7 = 13. Depois, 13 + 5 = 18.", ["12", "14", "18", "19", "20"], "A")
        self.assertEqual(check_consistency(q), (False, "C"))

    def test_fracao_equivalente_continua_ok(self):
        q = _q("2 x 2 = 4 e 3 x 2 = 6, logo 4/6.", ["4/6", "2/6", "4/3", "6/4", "2/3"], "A")
        self.assertEqual(check_consistency(q)[0], True)

    def test_sem_conta_nao_verificavel(self):
        self.assertEqual(check_consistency(_q("É um triângulo obtusângulo.", list("abcde"), "A")),
                         (None, None))


if __name__ == "__main__":
    unittest.main()
