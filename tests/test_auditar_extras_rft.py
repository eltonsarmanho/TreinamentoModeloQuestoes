import csv
import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import auditar_extras_rft as ae


class TestAuditoriaExtras(unittest.TestCase):
    def test_folha_cega_balanceada_e_determinista(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            ae.montar(12, 4, 7, Path(t1))
            ae.montar(12, 4, 7, Path(t2))
            a = (Path(t1) / "folha_professor.csv").read_text(encoding="utf-8-sig")
            self.assertEqual(a, (Path(t2) / "folha_professor.csv").read_text(encoding="utf-8-sig"))
            for palavra in ("rft", "controle", "injecao"):
                self.assertNotIn(palavra, a.lower())
            chave = json.loads((Path(t1) / "chave_oculta.json").read_text(encoding="utf-8"))["itens"]
            self.assertEqual(sum(v["origem"] == "controle_base" for v in chave.values()), 4)
            linhas = list(csv.DictReader(open(Path(t1) / "folha_professor.csv", encoding="utf-8-sig")))
            self.assertEqual({l["id"] for l in linhas}, set(chave))
            self.assertTrue(all("unidade" in v for v in chave.values()))


if __name__ == "__main__":
    unittest.main()
