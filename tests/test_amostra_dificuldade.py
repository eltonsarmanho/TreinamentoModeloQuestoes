import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import amostra_dificuldade as ad  # noqa: E402


class TestKappa(unittest.TestCase):
    def test_concordancia_total(self):
        pares = [(x, x) for x in ad.NIVEIS for _ in range(3)]
        self.assertAlmostEqual(ad.kappa_ponderado(pares), 1.0)

    def test_inversao_total_e_negativa(self):
        pares = [("Fácil", "Difícil")] * 3 + [("Difícil", "Fácil")] * 3 + [("Moderado", "Moderado")] * 3
        self.assertLess(ad.kappa_ponderado(pares), 0)

    def test_vazio(self):
        self.assertIsNone(ad.kappa_ponderado([]))


class TestApurar(unittest.TestCase):
    def test_folha_simulada(self):
        with tempfile.TemporaryDirectory() as d:
            chave = {"itens": {f"D{i}": {"rotulo": r, "ano": "5º", "habilidade": "H1", "origem": "banco"}
                               for i, r in enumerate(["Difícil", "Difícil", "Fácil", "Moderado"])}}
            json.dump(chave, open(f"{d}/c.json", "w"))
            with open(f"{d}/f.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "dificuldade_professor", "comentario"])
                w.writerows([["D0", "F", ""], ["D1", "difícil", ""], ["D2", "f", ""], ["D3", "x", ""]])
            class A:  # noqa: D401
                folha, chave_, saida = f"{d}/f.csv", None, f"{d}/o.json"
            a = A()
            a.chave = f"{d}/c.json"
            ad.cmd_apurar(a)
            r = json.load(open(a.saida))
            self.assertEqual((r["n"], r["iguais"], r["inflado"], r["baixo"]), (3, 2, 1, 0))
            self.assertEqual(len(r["problemas"]), 1)  # "x" inválido


class TestFolhaCega(unittest.TestCase):
    def test_folha_nao_revela_rotulo(self):
        self.assertNotIn("dificuldade_pedida", ad.COLUNAS)
        self.assertNotIn("rotulo", ad.COLUNAS)
