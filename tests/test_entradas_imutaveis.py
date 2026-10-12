"""Ciclo v6: a v5, a v3, o treino bruto, as validações congeladas e o banco não mudam.

Os sha256 ficam em ciclo_v6/ENTRADAS_IMUTAVEIS.json (versionado). data/ e DB/ estão no
.gitignore: num checkout sem os dados o teste é pulado, nunca aprovado às cegas.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from congelar_baseline import sha256_arquivo  # noqa: E402

TRAVA = ROOT / "ciclo_v6" / "ENTRADAS_IMUTAVEIS.json"


class TestEntradasImutaveis(unittest.TestCase):
    def test_cada_entrada_confere_com_o_sha_travado(self):
        arquivos = json.loads(TRAVA.read_text(encoding="utf-8"))["arquivos"]
        presentes = {rel: sha for rel, sha in arquivos.items() if (ROOT / rel).exists()}
        if not presentes:
            self.skipTest("dados ausentes neste checkout")
        for rel, sha in presentes.items():
            with self.subTest(arquivo=rel):
                self.assertEqual(sha256_arquivo(ROOT / rel), sha, f"{rel} mudou")

    def test_a_trava_cobre_a_v5_e_as_validacoes_congeladas(self):
        arquivos = json.loads(TRAVA.read_text(encoding="utf-8"))["arquivos"]
        for rel in ("data/train_curado_v5.jsonl", "data/val.jsonl", "data/val_frozen_v1.jsonl", "DB/questoes.db"):
            self.assertIn(rel, arquivos)


if __name__ == "__main__":
    unittest.main()
