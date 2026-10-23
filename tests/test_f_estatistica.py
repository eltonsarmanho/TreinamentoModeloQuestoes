"""Tarefa F: estatística própria (AUC por postos, kappa, limiar, Wilson, bootstrap OOB)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import f_estatistica as fe  # noqa: E402


class TestAuc(unittest.TestCase):
    def test_separacao_perfeita(self):
        self.assertEqual(fe.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)

    def test_inversa(self):
        self.assertEqual(fe.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)

    def test_empate_total_vale_meio(self):
        self.assertEqual(fe.auc([0.5] * 6, [0, 1, 0, 1, 0, 1]), 0.5)

    def test_confere_com_definicao_de_pares(self):
        s = [0.1, 0.4, 0.4, 0.35, 0.8, 0.7, 0.2]
        y = [0, 0, 1, 1, 1, 0, 1]
        pos = [a for a, b in zip(s, y) if b]
        neg = [a for a, b in zip(s, y) if not b]
        esperado = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
        self.assertAlmostEqual(fe.auc(s, y), esperado)

    def test_uma_classe_so_devolve_none(self):
        self.assertIsNone(fe.auc([0.1, 0.9], [1, 1]))


class TestKappa(unittest.TestCase):
    def test_concordancia_perfeita(self):
        self.assertEqual(fe.kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)

    def test_acaso_vale_zero(self):
        # predição independente do rótulo, mesma marginal
        self.assertAlmostEqual(fe.kappa([1, 1, 0, 0], [1, 0, 1, 0]), 0.0)

    def test_valor_conhecido(self):
        # tp=20 fp=5 fn=10 tn=15 -> po=0.7, pe=(25*30+25*20)/2500=0.5 -> kappa=0.4
        pred = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
        real = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
        self.assertAlmostEqual(fe.kappa(pred, real), 0.4)

    def test_indefinido_quando_so_uma_classe_em_tudo(self):
        self.assertIsNone(fe.kappa([1, 1], [1, 1]))


class TestLimiar(unittest.TestCase):
    def test_acha_o_corte_que_separa(self):
        s = [0.05, 0.1, 0.2, 0.7, 0.8, 0.95]
        y = [0, 0, 0, 1, 1, 1]
        t, v = fe.melhor_limiar(s, y, "acuracia")
        self.assertEqual(v, 1.0)
        self.assertTrue(0.2 < t <= 0.7)

    def test_bootstrap_oob_eh_determinista_e_ordenado(self):
        s = [i / 40 for i in range(40)]
        y = [1 if x > 0.5 else 0 for x in s]
        a = fe.bootstrap_oob(s, y, "acuracia", reps=200, seed=3)
        b = fe.bootstrap_oob(s, y, "acuracia", reps=200, seed=3)
        self.assertEqual(a, b)
        self.assertLessEqual(a["ic95"][0], a["media"])
        self.assertLessEqual(a["media"], a["ic95"][1])
        self.assertGreater(a["media"], 0.9)


class TestWilson(unittest.TestCase):
    def test_bate_com_a_funcao_do_gate(self):
        from promover_checkpoint import wilson
        lo, hi = fe.wilson_ic(33, 60)
        glo, ghi = wilson(33, 60)
        self.assertAlmostEqual(100 * lo, glo, places=1)
        self.assertAlmostEqual(100 * hi, ghi, places=1)

    def test_tabela_decresce_com_n(self):
        t = fe.tabela_tamanho_amostra([20, 40, 80, 160, 320])
        larg = [x["meia_largura_pp"] for x in t]
        self.assertEqual(larg, sorted(larg, reverse=True))
        self.assertAlmostEqual(t[0]["meia_largura_pp"], 19.3, delta=0.3)  # n=20, 65%: Wilson ~ +-19 pp


if __name__ == "__main__":
    unittest.main()
