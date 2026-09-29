"""Testes do avaliador de diversidade e do gate G11 (relatórios sintéticos).

Rodar: venv/bin/python -m unittest tests/test_gate_diversidade.py
"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import avaliar_diversidade as av  # noqa: E402
import promover_checkpoint as pc  # noqa: E402


def _rel(**ag):
    base = {"json_valido_pct": 100.0, "schema_pct": 100.0, "consistencia_inconsistente_pct": 5.0,
            "aderencia_pct": 80.0, "diversity_score": 0.40}
    base.update(ag)
    return {"prompt_ids": ["P01", "P02"], "modos": {"atual": {"agregado": dict(base)},
                                                    "ajustado": {"agregado": dict(base)}}}


class TestGateDiversidade(unittest.TestCase):
    def test_passa_quando_diversidade_sobe_sem_perda(self):
        g = pc.gate_diversidade(_rel(), _rel(diversity_score=0.7))
        self.assertTrue(g.passou, g.detalhe)
        self.assertTrue(g.bloqueante)

    def test_bloqueia_diversidade_maior_com_json_pior(self):
        g = pc.gate_diversidade(_rel(), _rel(diversity_score=0.9, json_valido_pct=96.0))
        self.assertFalse(g.passou)
        self.assertIn("json_valido_pct", g.detalhe)

    def test_bloqueia_schema_consistencia_aderencia(self):
        for campo, valor in (("schema_pct", 90.0), ("consistencia_inconsistente_pct", 12.0),
                             ("aderencia_pct", 60.0)):
            g = pc.gate_diversidade(_rel(), _rel(diversity_score=0.9, **{campo: valor}))
            self.assertFalse(g.passou, campo)
            self.assertIn(campo, g.detalhe)

    def test_tolerancia(self):
        g = pc.gate_diversidade(_rel(), _rel(diversity_score=0.9, aderencia_pct=78.0), tolerancia_pp=5)
        self.assertTrue(g.passou, g.detalhe)

    def test_bloqueia_se_diversidade_piora(self):
        g = pc.gate_diversidade(_rel(), _rel(diversity_score=0.39))
        self.assertFalse(g.passou)
        self.assertIn("diversity_score piorou", g.detalhe)

    def test_metrica_ausente_reprova(self):
        c = _rel()
        del c["modos"]["ajustado"]["agregado"]["schema_pct"]
        self.assertFalse(pc.gate_diversidade(_rel(), c).passou)
        self.assertFalse(pc.gate_diversidade(_rel(), {"modos": {}}).passou)

    def test_prompts_diferentes_reprova(self):
        c = _rel(diversity_score=0.9)
        c["prompt_ids"] = ["P01"]
        self.assertFalse(pc.gate_diversidade(_rel(), c).passou)

    def test_modos_distintos_no_mesmo_relatorio(self):
        r = _rel()
        r["modos"]["ajustado"]["agregado"]["diversity_score"] = 0.8
        self.assertTrue(pc.gate_diversidade(r, r, "atual", "ajustado").passou)
        self.assertFalse(pc.gate_diversidade(r, r, "ajustado", "atual").passou)

    def test_ganho_diversidade(self):
        self.assertEqual(pc.ganho_diversidade(_rel(), _rel()), [])
        self.assertEqual(len(pc.ganho_diversidade(_rel(), _rel(diversity_score=0.5))), 1)

    def test_gates_antigos_intactos(self):
        # avalia() continua devolvendo G1..G9, sem G11 implícito.
        ids = [g.id for g in pc.avalia({}, {})]
        self.assertEqual(ids, ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"])


class TestAvaliador(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts = av.carregar_prompts()

    def test_conjunto_fixo(self):
        p0 = self.prompts[0]
        self.assertEqual((p0["ano"], p0["habilidade"], p0["quantidade"]), ("9º", "H17", 10))
        self.assertEqual(av.carregar_prompts(num_prompts=1)[0]["id"], p0["id"])
        chaves = {(p["ano"], p["habilidade"]) for p in self.prompts}
        for ano in ("9º", "5º"):
            for h in ("H16", "H17", "H18", "H20", "H21", "H22"):
                self.assertIn((ano, h), chaves)
        self.assertTrue(any(p["k_subtemas"] == 1 for p in self.prompts))
        self.assertTrue({p["quantidade"] for p in self.prompts} >= {1, 3, 5, 10})
        self.assertEqual(len({p["seed"] for p in self.prompts}), len(self.prompts))

    def test_metricas_lote(self):
        q = av.GeradorSimulado()._q("Classifique o triângulo com lados 3, 4 e 5 quanto aos lados.", "Fácil")
        ruim = dict(q, difficulty="HARD", resposta_correta="Z")
        m = av.metricas_lote([q, ruim], self.prompts[0], 1, 1, 2.0)
        self.assertEqual(m["quantidade_gerada"], 2)
        self.assertEqual(m["schema_ok"], 1)
        self.assertEqual(m["difficulty_correta"], 1)
        self.assertEqual(m["aderentes"], 2)  # "triângulo" -> subtema de 9º H17
        self.assertEqual(m["tempo_por_questao_s"], 1.0)
        self.assertEqual(sum(m["consistencia"].values()), 2)

    def test_dry_run_ajustado_mais_diverso_e_mesma_qualidade(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(av.main(["--rotulo", "t", "--dry-run", "--num-prompts", "4",
                                      "--out-dir", d]), 0)
            rel = json.loads((Path(d) / "diversidade_t.json").read_text(encoding="utf-8"))
            md = (Path(d) / "diversidade_t.md").read_text(encoding="utf-8")
        a, j = rel["modos"]["atual"]["agregado"], rel["modos"]["ajustado"]["agregado"]
        self.assertGreater(j["diversity_score"], a["diversity_score"])
        self.assertLess(j["duplicate_rate"], a["duplicate_rate"])
        self.assertEqual(j["json_valido_pct"], a["json_valido_pct"])
        for rotulo in ("JSON válido", "Correção matemática", "Aderência", "Cobertura",
                       "Repetição semântica", "Diversidade estrutural", "Diversidade contextual"):
            self.assertIn(rotulo, md)
        # O relatório alimenta o gate diretamente (atual x ajustado).
        self.assertTrue(pc.gate_diversidade(rel, rel, "atual", "ajustado").passou)

    def test_modo_independente_usa_n_chamadas(self):
        m = av.rodar_atual(av.GeradorSimulado(), self.prompts[0], "independente")
        self.assertEqual(m["chamadas"], 10)
        self.assertEqual(m["quantidade_gerada"], 10)

    def test_agregar_ignora_lotes_de_1_na_diversidade(self):
        g = av.GeradorSimulado()
        p1 = next(p for p in self.prompts if p["quantidade"] == 1)
        l1 = av.rodar_atual(g, p1, "unico")
        l10 = av.rodar_atual(g, self.prompts[0], "unico")
        ag = av.agregar([l1, l10])
        self.assertEqual(ag["lotes_diversidade_n"], 1)
        self.assertAlmostEqual(ag["diversity_score"], round(l10["diversity_score"], 4))

    def test_ajustado_indisponivel_nao_inventa_numeros(self):
        class SemLote(av.GeradorSimulado):
            def ajustado(self, *a, **k):
                raise ImportError("gerar_lote ausente")
        r = av.avaliar(SemLote(), self.prompts[:1], log=lambda *_: None)
        self.assertIn("erro", r["ajustado"])
        self.assertIn("agregado", r["atual"])


if __name__ == "__main__":
    unittest.main()
