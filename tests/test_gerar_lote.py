"""
Testes do modo planejado de inferência (src/gerar_lote.py) SEM llama-cli:
test_model.generate é substituído por um mock via o parâmetro gen_fn.
    venv/bin/python -m unittest tests/test_gerar_lote.py -v
"""
import json
import math
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import diversidade as dv  # noqa: E402
import gerar_lote as gl  # noqa: E402
from schema_utils import check_structure  # noqa: E402

TAX = dv.carregar_taxonomia()
RX_SUBTEMA = re.compile(r"Subtema: ([^.]+)\.")


def _palavra(n):
    """Palavra só de consoantes, única por n: dígitos seriam mascarados como
    '#', e vogais poderiam casar por acaso com as regex de contexto."""
    alfa = "bcdfghjklmnpqrstvwxz"
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, len(alfa))
        s = alfa[r] + s
    return "x" + s


def _questao(texto, n):
    # vocabulário exclusivo por chamada: nenhuma duas questões são near-duplicatas
    extra = " ".join(_palavra(n * 20 + i) for i in range(20))
    return {"enunciado": f"{texto} {extra}?",
            "alternativas": {L: f"opcao {_palavra(n * 7 + j)}" for j, L in enumerate("ABCDE")},
            "resolucao_passo_a_passo": "Pela definição, a alternativa A é a correta.",
            "resposta_correta": "A", "difficulty": "MEDIUM"}


class MockGen:
    """Substituto de test_model.generate. `modo`:
    - 'obedece': gera sobre o subtema pedido no prompt (rótulo do slot);
    - 'teimoso': sempre fala do texto fixo `fixo`, ignorando o prompt;
    - 'invalido_depois': 1ª chamada válida e fixa, demais JSON quebrado."""

    def __init__(self, modo="obedece", fixo="triângulos"):
        self.modo, self.fixo, self.chamadas, self.prompts, self.seeds = modo, fixo, 0, [], []

    def __call__(self, llama_cli, gguf_path, user_prompt, threads, max_new_tokens,
                 seed=None, grammar=None):
        self.prompts.append(user_prompt)
        self.seeds.append(seed)
        n = self.chamadas
        self.chamadas += 1
        if self.modo == "invalido_depois" and n > 0:
            return "{quebrado", None, 30.0, 0.01
        if self.modo == "obedece":
            m = RX_SUBTEMA.search(user_prompt)
            tema = m.group(1) if m else "geral"
        else:
            tema = self.fixo
        q = _questao(f"Questão sobre {tema}", n)
        return json.dumps({"questoes": [q]}, ensure_ascii=False), None, 30.0, 0.01


def _gerar(ano, hab, n, gen, **kw):
    return gl.gerar_lote_planejado("llama-cli", "m.gguf", ano, hab, "desc", "Moderado", n, 4,
                                   base_seed=kw.pop("base_seed", 7), gen_fn=gen,
                                   taxonomia=TAX, **kw)


class TestLotePlanejado(unittest.TestCase):
    def test_tamanhos_de_lote(self):
        # 9º H21 fica fora da contagem exata de chamadas: o rótulo "retângulos e
        # quadrados" casa a regex de contexto "quadra" (esporte) em diversidade.py.
        for ano, hab in (("9º", "H17"), ("9º", "H16"), ("5º", "H18"), ("2º", "H17")):
            k = len(dv.obter_habilidade(ano, hab, TAX)["subtemas"])
            for n in (1, 3, 5, 10, 20):
                with self.subTest(ano=ano, hab=hab, n=n):
                    gen = MockGen()
                    r = _gerar(ano, hab, n, gen)
                    self.assertEqual(len(r["questoes"]), n)
                    self.assertTrue(r["flags"]["quantidade_correta"])
                    self.assertTrue(all(check_structure({"questoes": [q]}, 1)["schema_completo"]
                                        for q in r["questoes"]))
                    self.assertEqual(set(r["obj"]), {"questoes"})  # schema inalterado
                    self.assertEqual(len(r["plano"]), n)
                    # uma chamada por questão quando o modelo obedece o plano
                    self.assertEqual(gen.chamadas, n)
                    self.assertEqual(r["regeneracoes_diversidade"], 0)
                    subt = {s["subtema"] for s in r["plano"]}
                    self.assertEqual(len(subt), min(n, k))
                    self.assertIsNotNone(r["metricas"])
                    self.assertIn("diversity_score", r["metricas"])
                    self.assertEqual(r["metricas"]["duplicate_rate"], 0.0)
                    self.assertGreaterEqual(r["tempo_s"], 0)

    def test_prompt_tem_sufixo_e_quantidade_1(self):
        gen = MockGen()
        _gerar("9º", "H17", 4, gen)
        for p in gen.prompts:
            self.assertIn("Subtema:", p)
            self.assertIn("Tipo de raciocínio:", p)
            self.assertNotIn("Restrição", p)
        rotulos = [RX_SUBTEMA.search(p).group(1) for p in gen.prompts]
        self.assertEqual(sorted(rotulos), ["quadriláteros", "quadriláteros", "triângulos", "triângulos"])

    def test_9_h17_cobre_triangulos_e_quadrilateros(self):
        r = _gerar("9º", "H17", 10, MockGen())
        self.assertEqual(r["metricas"]["subtema_distribution"], {"triangulo": 5, "quadrilatero": 5})
        self.assertEqual(r["metricas"]["coverage_score"], 1.0)

    def test_regeneracao_respeita_limite(self):
        for max_t in (0, 1, 2, 3):
            with self.subTest(max_t=max_t):
                gen = MockGen("teimoso", "triângulos")
                n = 4
                r = _gerar("9º", "H17", n, gen, max_tentativas_diversidade=max_t)
                self.assertEqual(len(r["questoes"]), n)  # nunca descarta slot
                self.assertLessEqual(gen.chamadas, n * (max_t + 1))
                for d in r["detalhes"]:
                    self.assertLessEqual(d["tentativas_diversidade"], max_t + 1)
                slots_quad = [d for d in r["detalhes"] if d["subtema_planejado"] == "quadrilatero"]
                for d in slots_quad:
                    self.assertEqual(d["tentativas_diversidade"], max_t + 1)
                # violação remanescente fica registrada
                self.assertTrue(any(v["violacoes"] for v in r["violacoes"]))
                if max_t:
                    self.assertTrue(any("Restrição:" in p for p in gen.prompts))
                    self.assertGreater(r["regeneracoes_diversidade"], 0)

    def test_regeneracao_usa_seed_diferente(self):
        gen = MockGen("teimoso", "triângulos")
        _gerar("9º", "H17", 2, gen, max_tentativas_diversidade=2)
        self.assertEqual(len(gen.seeds), len(set(gen.seeds)))

    def test_qualidade_primeiro(self):
        """Candidato válido não diverso não é trocado por inválido."""
        gen = MockGen("invalido_depois", "quadriláteros")
        r = gl.gerar_lote_planejado("x", "y", "9º", "H17", "d", "Moderado", 1, 4,
                                    base_seed=3, gen_fn=gen, taxonomia=TAX, retries=0,
                                    max_tentativas_diversidade=2)
        plano = r["plano"][0]["subtema"]
        self.assertEqual(len(r["questoes"]), 1)
        self.assertEqual(r["detalhes"][0]["status"], "nao_verificavel")
        self.assertEqual(r["detalhes"][0]["tentativa_escolhida"], 0)
        if plano != "quadrilatero":
            self.assertGreater(gen.chamadas, 1)

    def test_determinismo(self):
        a = _gerar("9º", "H21", 5, MockGen(), base_seed=11)
        b = _gerar("9º", "H21", 5, MockGen(), base_seed=11)
        self.assertEqual(a["plano"], b["plano"])
        self.assertEqual(a["questoes"], b["questoes"])

    def test_habilidade_fora_da_taxonomia(self):
        r = _gerar("9º", "H99", 3, MockGen())
        self.assertEqual(len(r["questoes"]), 3)
        self.assertFalse(r["planejado"])
        self.assertIsNone(r["metricas"])

    def test_run_one_modo_antigo_preservado(self):
        import test_model
        self.assertTrue(hasattr(test_model, "run_planejado"))
        import inspect
        self.assertIn("planejado", inspect.signature(test_model.run_one).parameters)
        self.assertIn("gen_fn", inspect.signature(test_model.generate_validated).parameters)



class TestQualidadeAntesDeDiversidade(unittest.TestCase):
    def test_falha_estrutural_sem_violacao_retenta(self):
        # 1ª resposta com alternativas repetidas (sem violação de diversidade):
        # não pode ser entregue sem usar as tentativas restantes.
        estado = {"n": 0}

        def gen(llama_cli, gguf_path, user_prompt, threads, max_new_tokens, seed=None, grammar=None):
            estado["n"] += 1
            q = _questao("Questão sobre triângulos", estado["n"])
            if estado["n"] <= 2:  # 2 = retries internos do generate_validated
                q["alternativas"]["E"] = q["alternativas"]["D"]
            return json.dumps({"questoes": [q]}, ensure_ascii=False), None, 30.0, 0.01

        r = _gerar("9º", "H17", 1, gen, retries=1, max_tentativas_diversidade=2)
        self.assertTrue(r["flags"]["alternativas_distintas"])
        self.assertNotEqual(r["detalhes"][0]["status"], "falha")
        self.assertEqual(r["detalhes"][0]["tentativa_escolhida"], 1)

if __name__ == "__main__":
    unittest.main()
