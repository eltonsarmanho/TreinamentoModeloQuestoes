import json
import random
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import cliente_llama_server as c
import extract_data

TAX = json.loads((RAIZ / "data" / "taxonomia_subtemas.json").read_text(encoding="utf-8"))


class TestClienteLlamaServer(unittest.TestCase):
    def test_system_prompt_identico_ao_do_treino(self):
        self.assertEqual(c.SYSTEM_PROMPT, extract_data.SYSTEM_PROMPT)

    def test_prompt_bate_com_os_vetores_do_guia(self):
        vetores = json.loads((RAIZ / "Doc" / "vetores_sufixo_app.json").read_text(encoding="utf-8"))["vetores"]
        for v in vetores:
            s = v["slot"]
            slot = {"subtema": s.get("subtema"), "tipo": s.get("tipo_raciocinio"), "contexto": s["contexto"]}
            self.assertEqual(c.montar_prompt(TAX, v["ano"], v["habilidade"], v["descricao"], v["dificuldade"], slot),
                             v["prompt_esperado"], v["habilidade"])

    def test_sorteio_cobre_subtemas_distintos_sem_repetir(self):
        rng, hist, vistos = random.Random(1), [], []
        k = len(TAX["habilidades"]["5º|H01"]["subtemas"])
        for _ in range(k):
            s = c.sortear_slot(TAX, "5º", "H01", hist, rng)
            hist.append(s)
            vistos.append(s["subtema"])
        self.assertEqual(len(set(vistos)), k)

    def test_estrutura(self):
        q = {"enunciado": "x", "alternativas": dict(zip("ABCDE", "12345")), "resposta_correta": "B"}
        self.assertTrue(c.estrutura_ok({"questoes": [q]}))
        self.assertFalse(c.estrutura_ok({"questoes": [{**q, "resposta_correta": "F"}]}))
        self.assertFalse(c.estrutura_ok({"questoes": []}))


if __name__ == "__main__":
    unittest.main()
