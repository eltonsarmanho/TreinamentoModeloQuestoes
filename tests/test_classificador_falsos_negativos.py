"""Falsos negativos reais do classificador (outputs/diversidade_atual.json, 2026-09).

Cada enunciado abaixo foi gerado pelo GGUF, é aderente à habilidade e era
classificado como "outros" — derrubando a aderência medida sem culpa do modelo.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import diversidade as dv  # noqa: E402

CASOS = [
    ("9º", "H16", "condicao_existencia",
     "Um casal vai montar uma estrutura com 3 cordas de diferentes comprimentos. "
     "Para que o triângulo possa ser construído, qual das medidas serve?"),
    ("5º", "H16", "retangulo_quadrado",
     "Um jardim tem 12 metros de comprimento e 10 metros de largura. Qual é a área?"),
    ("5º", "H20", "chance_maior_menor",
     "A mesada tem 5 bolas de gude azuis e 3 vermelhas. Qual é a chance de retirar uma bola vermelha?"),
    ("9º", "H08", "fracao",
     "Uma planta foi dividida em 4 partes iguais. Se 1 das partes foi plantada, quantas partes faltam?"),
]


class TestFalsosNegativos(unittest.TestCase):
    def test_nao_cai_em_outros(self):
        for ano, hab, esperado, texto in CASOS:
            with self.subTest(ano=ano, hab=hab):
                c = dv.classificar_questao({"enunciado": texto}, ano, hab)
                self.assertNotEqual(c["subtema"], "outros")
                if esperado in {s["id"] for s in dv.obter_habilidade(ano, hab)["subtemas"]}:
                    self.assertEqual(c["subtema"], esperado)

    def test_contexto_nao_vira_subtema(self):
        c = dv.classificar_questao({"enunciado": "Em uma festa junina, quanto é 320 - 280?"}, "9º", "H22")
        self.assertEqual(c["subtema"], "outros")


if __name__ == "__main__":
    unittest.main()
