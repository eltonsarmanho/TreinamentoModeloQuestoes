import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_model as tm


class _Resp:
    def read(self):
        return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestAmostragemOpcional(unittest.TestCase):
    def setUp(self):
        self.p = [mock.patch.object(tm, "MOTOR", "server"),
                  mock.patch.object(tm._Servidor, "garantir", return_value="http://x"),
                  mock.patch.object(tm._Servidor, "parar")]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()

    def _enviado(self):
        with mock.patch.object(tm.urllib.request, "urlopen", return_value=_Resp()) as u:
            tm.generate("cli", "m.gguf", "Gere 1", 8, 512, seed=1)
        return json.loads(u.call_args[0][0].data)

    def test_padrao_nao_envia_top_k_nem_min_p(self):
        corpo = self._enviado()
        self.assertNotIn("top_k", corpo)
        self.assertNotIn("min_p", corpo)
        self.assertEqual((corpo["temperature"], corpo["top_p"]), (0.7, 0.8))
        self.assertEqual(tm.amostragem_extra(), {})

    def test_card_do_qwen3_chega_ao_servidor_e_ao_relatorio(self):
        with mock.patch.object(tm, "TOP_K", 20), mock.patch.object(tm, "MIN_P", 0.0):
            corpo = self._enviado()
            self.assertEqual((corpo["top_k"], corpo["min_p"]), (20, 0.0))  # min_p 0 não pode sumir
            self.assertEqual(tm.amostragem_extra(), {"top_k": 20, "min_p": 0.0})

    def test_temperatura_so_entra_no_relatorio_se_mudar(self):
        with mock.patch.object(tm, "TEMPERATURE", 0.4):
            self.assertEqual(tm.amostragem_extra(), {"temperatura": 0.4})
            self.assertEqual(self._enviado()["temperature"], 0.4)


if __name__ == "__main__":
    unittest.main()
