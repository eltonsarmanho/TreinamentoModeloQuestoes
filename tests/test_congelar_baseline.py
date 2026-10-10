"""Manifesto de baseline congelado (P0-1): hashes, verificação de deriva, sem segredos."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import congelar_baseline as cb  # noqa: E402


class TestHashes(unittest.TestCase):
    def test_sha256_arquivo_igual_ao_do_texto(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"
            p.write_text("olá", encoding="utf-8")
            self.assertEqual(cb.sha256_arquivo(p), cb.sha256_texto("olá"))

    def test_entrada_conta_linhas_de_jsonl_e_marca_ausente(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "x.jsonl").write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
            self.assertEqual(cb._entrada(d, "x.jsonl")["linhas"], 2)
            self.assertFalse(cb._entrada(d, "nao_existe.gguf")["presente"])


class TestVerificar(unittest.TestCase):
    def _manifesto_real(self):
        return cb.montar_manifesto(ROOT, "teste")

    def test_disco_atual_confere_com_o_manifesto_recem_montado(self):
        m = self._manifesto_real()
        self.assertEqual(cb.verificar(m, ROOT), [])

    def test_detecta_arquivo_alterado_e_arquivo_sumido(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.bin").write_bytes(b"1")
            m = {"artefatos": {"a": cb._entrada(d, "a.bin")}, "datasets": {}, "relatorios": {},
                 "prompt": self._manifesto_real()["prompt"],
                 "inferencia": self._manifesto_real()["inferencia"]}
            self.assertEqual(cb.verificar(m, d), [])
            (Path(d) / "a.bin").write_bytes(b"2")
            self.assertTrue(any("sha256 mudou" in p for p in cb.verificar(m, d)))
            (Path(d) / "a.bin").unlink()
            self.assertTrue(any("sumiu" in p for p in cb.verificar(m, d)))

    def test_detecta_mudanca_de_inferencia(self):
        m = self._manifesto_real()
        m["inferencia"]["n_ctx"] = 4096
        self.assertTrue(any("n_ctx" in p for p in cb.verificar(m, ROOT)))


class TestConteudo(unittest.TestCase):
    def test_sem_segredos_e_com_o_essencial(self):
        m = cb.montar_manifesto(ROOT, "teste")
        txt = json.dumps(m, ensure_ascii=False)
        for proibido in ("MARITALK", "GOOGLE_API", "HF_TOKEN", "hf_", ".env"):
            self.assertNotIn(proibido, txt)
        self.assertEqual(m["treino"]["LORA_RANK"], "16")
        self.assertEqual(m["inferencia"]["n_ctx"], 2048)
        self.assertIn("sha256", m["artefatos"]["gguf"])
        self.assertEqual(m["datasets"]["val_congelado"]["linhas"], 30)


if __name__ == "__main__":
    unittest.main()
