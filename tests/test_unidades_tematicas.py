"""Mapa (ano, habilidade) -> unidade temática (src/unidades_tematicas.py)."""
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import unidades_tematicas as ut  # noqa: E402

CORPUS = RAIZ / "data" / "train_curado_v3.jsonl"
TAXONOMIA = RAIZ / "data" / "taxonomia_subtemas.json"


class TestUnidades(unittest.TestCase):
    def test_as_79_habilidades_do_corpus_estao_mapeadas(self):
        if not CORPUS.exists():
            self.skipTest("corpus ausente")
        pares = set()
        for linha in CORPUS.read_text(encoding="utf-8").splitlines():
            m = json.loads(linha)["meta"]
            pares.add((m["ano"], m["habilidade"]))
        self.assertEqual(len(pares), 79)
        sem = sorted(p for p in pares if ut.unidade_de(*p) not in ut.UNIDADES)
        self.assertEqual(sem, [])

    def test_toda_habilidade_da_taxonomia_esta_mapeada(self):
        tax = json.loads(TAXONOMIA.read_text(encoding="utf-8"))["habilidades"]
        sem = [k for k in tax if ut.unidade_de(*k.split("|")) is None]
        self.assertEqual(sem, [])

    def test_tamanho_de_cada_unidade_por_ano(self):
        cont = Counter((a, u) for (a, _), u in ut.MAPA.items())
        esperado = {"2º": (8, 2, 3, 5, 3), "5º": (6, 3, 5, 5, 3), "9º": (9, 4, 5, 4, 4)}
        for ano, qtds in esperado.items():
            self.assertEqual(tuple(cont[(ano, u)] for u in ut.UNIDADES), qtds, ano)
        self.assertEqual(len(ut.MAPA), 21 + 22 + 26)

    def test_pontos_nao_obvios(self):
        casos = {("2º", "H09"): ut.ALGEBRA, ("5º", "H09"): ut.ALGEBRA,
                 ("2º", "H13"): ut.GEOMETRIA, ("5º", "H11"): ut.GEOMETRIA,
                 ("5º", "H19"): ut.GRANDEZAS,   # moedas
                 ("9º", "H20"): ut.GRANDEZAS,   # perímetro
                 ("9º", "H21"): ut.GRANDEZAS,   # área
                 ("9º", "H09"): ut.NUMEROS,     # fração/decimal/%
                 ("5º", "H20"): ut.ESTATISTICA,  # chances de ocorrência
                 ("9º", "H26"): ut.ESTATISTICA}
        for (ano, hab), unidade in casos.items():
            self.assertEqual(ut.unidade_de(ano, hab), unidade, (ano, hab))

    def test_bncc_e_numeros_e_formatos_de_ano(self):
        self.assertEqual(ut.unidade_de("3º", "EF03MA07"), ut.NUMEROS)
        self.assertEqual(ut.unidade_de("5º ano", "H12"), ut.GEOMETRIA)
        self.assertEqual(ut.unidade_de(5, "h12"), ut.GEOMETRIA)

    def test_desconhecida_devolve_none(self):
        self.assertIsNone(ut.unidade_de("9º", "H99"))
        self.assertIsNone(ut.unidade_de("3º", "H01"))


if __name__ == "__main__":
    unittest.main()
