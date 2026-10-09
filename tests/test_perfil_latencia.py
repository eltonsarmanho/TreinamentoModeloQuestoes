"""Perfil de latência do llama-cli (--perfil): parser do stderr e agregação."""
import json
import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_model as tm  # noqa: E402

# Trecho real de `llama-cli -lv 3 --log-timestamps` (build b10194).
STDERR_REAL = """\
0.02.904.191 I srv  llama_server: model loaded
0.08.778.703 I slot print_timing: id  0 | task 0 | n_decoded =    100, tg =  28.38 t/s, tg_3s =  28.38 t/s
0.12.229.245 I slot print_timing: id  0 | task 0 | prompt eval time =    2232.46 ms /   329 tokens (    6.79 ms per token,   147.37 tokens per second)
0.12.229.248 I slot print_timing: id  0 | task 0 |        eval time =    6973.92 ms /   198 tokens (   35.22 ms per token,    28.39 tokens per second)
0.12.229.248 I slot print_timing: id  0 | task 0 |       total time =    9206.38 ms /   527 tokens
"""


class TestParsePerfil(unittest.TestCase):
    def test_stderr_real(self):
        p = tm.parse_perfil(STDERR_REAL, 12.54)
        self.assertAlmostEqual(p["carga_s"], 2.904, places=3)
        self.assertEqual((p["prefill_tokens"], p["decode_tokens"]), (329, 198))
        self.assertAlmostEqual(p["prefill_s"], 2.232, places=3)
        self.assertAlmostEqual(p["decode_s"], 6.974, places=3)
        self.assertAlmostEqual(p["outros_s"], 12.54 - 2.904 - 2.232 - 6.974, places=2)

    def test_minutos_no_timestamp(self):
        p = tm.parse_perfil("1.05.500.000 I srv  llama_server: model loaded\n", 70.0)
        self.assertAlmostEqual(p["carga_s"], 65.5)

    def test_timeout_sem_stderr(self):
        p = tm.parse_perfil("", 300.0)
        self.assertIsNone(p["carga_s"])
        self.assertIsNone(p["outros_s"])
        self.assertEqual(p["wall_s"], 300.0)

    def test_resumo_ignora_none_e_soma_fracoes(self):
        chamadas = [tm.parse_perfil(STDERR_REAL, 12.54), tm.parse_perfil("", 300.0)]
        r = tm.resumo_perfil(chamadas)
        self.assertEqual(r["chamadas"], 2)
        self.assertEqual(r["decode_tokens"]["mediana"], 198)
        self.assertAlmostEqual(r["wall_s"]["soma"], 312.54)
        self.assertLess(sum(v for v in r["fracao_do_wall_pct"].values()), 100)


class TestGenerateNaoMudaSemPerfil(unittest.TestCase):
    def test_desligado_por_padrao(self):
        self.assertIsNone(tm.PERFIL_CHAMADAS)


class _Resp:
    def __init__(self, corpo):
        self._b = json.dumps(corpo).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestMotorServer(unittest.TestCase):
    def setUp(self):
        self.p = [mock.patch.object(tm, "MOTOR", "server"),
                  mock.patch.object(tm._Servidor, "garantir", return_value="http://x"),
                  mock.patch.object(tm._Servidor, "parar")]
        for x in self.p:
            x.start()

    def tearDown(self):
        for x in self.p:
            x.stop()
        tm.PERFIL_CHAMADAS = None

    def test_resposta_e_perfil(self):
        corpo = {"choices": [{"message": {"content": '{"questoes": []}'}}],
                 "timings": {"prompt_n": 50, "cache_n": 307, "prompt_ms": 400.0,
                             "predicted_n": 200, "predicted_ms": 5000.0,
                             "prompt_per_second": 125.0, "predicted_per_second": 40.0}}
        tm.PERFIL_CHAMADAS = []
        with mock.patch.object(tm.urllib.request, "urlopen", return_value=_Resp(corpo)) as u:
            texto, ptps, gtps, _ = tm.generate("cli", "m.gguf", "Gere 1", 8, 512, seed=7)
        enviado = json.loads(u.call_args[0][0].data)
        self.assertEqual((enviado["seed"], enviado["cache_prompt"]), (7, True))
        self.assertEqual((texto, ptps, gtps), ('{"questoes": []}', 125.0, 40.0))
        c = tm.PERFIL_CHAMADAS[0]
        self.assertEqual((c["prefill_tokens"], c["prefill_cache_tokens"], c["decode_tokens"]), (50, 307, 200))
        self.assertEqual(c["carga_s"], 0.0)

    def test_erro_de_rede_vira_geracao_invalida(self):
        with mock.patch.object(tm.urllib.request, "urlopen", side_effect=TimeoutError()):
            texto, _, gtps, elapsed = tm.generate("cli", "m.gguf", "Gere 1", 8, 512, seed=1)
        self.assertEqual((texto, gtps), ("", None))
        tm._Servidor.parar.assert_called()


if __name__ == "__main__":
    unittest.main()
