"""Manifesto por execução de treino (Fase 1.1, 2026-10-14). Sem GPU, sem unsloth."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import manifesto_treino as mt  # noqa: E402


class TestManifestoTreino(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.treino = self.dir / "t.jsonl"
        self.val = self.dir / "v.jsonl"
        self.treino.write_text('{"a":1}\n{"a":2}\n\n', encoding="utf-8")
        self.val.write_text('{"a":3}\n', encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _manifesto(self):
        cli = {"train": self.treino, "val": self.val, "out": self.dir / "lora",
               "checkpoints": self.dir / "ck"}
        return mt.criar(RAIZ, cli, {"seed": 43, "learning_rate": 2e-4}, "GPU X",
                        mt.contagem_truncadas([10, 2000, 1024, 1025], 1024))

    def test_hash_e_linhas_dos_dados(self):
        m = self._manifesto()
        self.assertEqual(m["dados"]["treino"]["linhas"], 2)  # linha vazia não conta
        self.assertEqual(m["dados"]["treino"]["sha256"], mt.sha256_arquivo(self.treino))
        self.assertNotEqual(m["dados"]["treino"]["sha256"], m["dados"]["validacao"]["sha256"])

    def test_arquivo_inexistente_e_registrado_nao_ignorado(self):
        self.assertFalse(mt.descreve_arquivo(self.dir / "nao_existe.jsonl")["existe"])

    def test_truncagem_conta_so_acima_do_limite(self):
        t = mt.contagem_truncadas([10, 2000, 1024, 1025], 1024)
        self.assertEqual((t["acima_do_limite"], t["maior"], t["exemplos"]), (2, 2000, 4))

    def test_hiperparametros_e_ambiente_presentes(self):
        m = self._manifesto()
        self.assertEqual(m["hiperparametros"]["seed"], 43)
        self.assertEqual(m["ambiente"]["gpu"], "GPU X")
        self.assertIn("torch", m["ambiente"]["pacotes"])
        self.assertIsNone(m["resultado"])

    def test_gravar_e_resultado_roundtrip(self):
        m = self._manifesto()
        m["resultado"] = mt.resumo_resultado(
            [{"epoch": 1.0, "eval_loss": 0.68}, {"epoch": 1.0, "loss": 0.5}, {"epoch": 2.0, "eval_loss": 0.67}],
            0.45, "ck/checkpoint-356", 0.67)
        destino = mt.gravar(m, self.dir / "lora")
        self.assertEqual(destino.name, "MANIFESTO_TREINO.json")
        lido = json.loads(destino.read_text(encoding="utf-8"))
        self.assertEqual([e["eval_loss"] for e in lido["resultado"]["eval_por_epoca"]], [0.68, 0.67])
        self.assertEqual(lido["resultado"]["melhor_checkpoint"], "ck/checkpoint-356")

    def test_train_py_exige_train_e_val(self):
        """Sem execução (unsloth é pesado): o código-fonte não pode ter padrão para os dois."""
        src = (RAIZ / "src" / "train.py").read_text(encoding="utf-8")
        self.assertRegex(src, r'add_argument\("--train",\s*required=True')
        self.assertRegex(src, r'add_argument\("--val",\s*required=True')


if __name__ == "__main__":
    unittest.main()
