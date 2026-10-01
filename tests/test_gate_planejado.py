"""Perfil 'planejado' do promover_checkpoint: gates no modo que o app usa."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import promover_checkpoint as pc  # noqa: E402
from avaliar_diversidade import agregar  # noqa: E402


def _lote(pid, ano="9º", gerada=5, aderentes=5, inconsist=0, t=50.0):
    return {"id": pid, "ano": ano, "habilidade": "H17", "quantidade_pedida": 5,
            "quantidade_gerada": gerada, "k_subtemas": 3, "aderencia_mensuravel": True,
            "chamadas": 5, "json_validos": 5, "schema_ok": gerada, "alternativas_distintas": gerada,
            "difficulty_correta": gerada, "aderentes": aderentes, "depende_de_visual": 0,
            "consistencia": {"ok": gerada - inconsist, "nao_verificavel": 0, "inconsistente": inconsist},
            "tempo_s": t, "coverage_score": 1.0, "duplicate_rate": 0.0, "diversity_score": 0.8,
            "semantic_similarity": {"cosseno_media": 0.2, "jaccard_max": 0.1}}


def _rel(sha, lotes, seed=0):
    return {"artefato_sha256": sha, "seed_offset": seed, "prompt_ids": [l["id"] for l in lotes],
            "max_tentativas_diversidade": 1,
            "modos": {"ajustado": {"lotes": lotes, "agregado": agregar(lotes)}}}


class TestPerfilPlanejado(unittest.TestCase):
    def _gravar(self, d, nome, rel):
        p = Path(d) / nome
        p.write_text(json.dumps(rel), encoding="utf-8")
        return str(p)

    def test_junta_seeds_e_pareia(self):
        with tempfile.TemporaryDirectory() as d:
            ps = [self._gravar(d, f"b_s{s}.json", _rel("a", [_lote("P1")], s)) for s in (0, 1)]
            rel, chave = pc.junta_planejado(ps)
            self.assertEqual(chave, [(0, "P1"), (1, "P1")])
            self.assertEqual(rel["modos"]["ajustado"]["agregado"]["num_lotes"], 2)

    def test_entrega_incompleta_reprova(self):
        base = _rel("a", [_lote("P1")])
        cand = _rel("b", [_lote("P1", gerada=4, aderentes=4)])
        g = {x.id: x for x in pc.avalia_planejado(base, cand)}
        self.assertFalse(g["P2"].passou)

    def test_aderencia_dentro_da_margem_passa(self):
        base = _rel("a", [_lote(f"P{i}") for i in range(10)])
        cand = _rel("b", [_lote("P0", aderentes=4)] + [_lote(f"P{i}") for i in range(1, 10)])
        g = {x.id: x for x in pc.avalia_planejado(base, cand)}
        self.assertTrue(g["P4"].passou)  # 100 -> 98: dentro dos 2pp

    def test_tempo_e_informativo(self):
        base = _rel("a", [_lote("P1", t=10.0)])
        cand = _rel("b", [_lote("P1", t=100.0)])
        p8 = [x for x in pc.avalia_planejado(base, cand) if x.id == "P8"][0]
        self.assertFalse(p8.passou)
        self.assertFalse(p8.bloqueante)

    def test_mistura_de_artefatos_aborta(self):
        with tempfile.TemporaryDirectory() as d:
            ps = [self._gravar(d, "x_s0.json", _rel("a", [_lote("P1")])),
                  self._gravar(d, "x_s1.json", _rel("z", [_lote("P1")], 1))]
            with self.assertRaises(SystemExit):
                pc.junta_planejado(ps)


class TestTetoAbsolutoP3(unittest.TestCase):
    """Pré-registro 2026-09-30: P3 tem teto absoluto de 2% de gabaritos
    inconsistentes, que não é relaxado por tolerância nem por significância."""

    def _rel(self, sha, inconsist_por_lote, n_lotes=20):
        return _rel(sha, [_lote(f"P{i}", inconsist=inconsist_por_lote) for i in range(n_lotes)])

    def test_acima_do_teto_reprova_mesmo_sem_piorar(self):
        # 3 em 5 por lote = 60% nos dois lados: sem piora relativa, mas acima de 2%
        base, cand = self._rel("a", 3), self._rel("b", 3)
        p3 = {g.id: g for g in pc.avalia_planejado(base, cand)}["P3"]
        self.assertFalse(p3.passou)
        self.assertIn("limite absoluto", p3.detalhe)

    def test_abaixo_do_teto_e_sem_piora_passa(self):
        base, cand = self._rel("a", 0), self._rel("b", 0)
        p3 = {g.id: g for g in pc.avalia_planejado(base, cand)}["P3"]
        self.assertTrue(p3.passou)

    def test_piso_continua_sendo_piso_para_metricas_positivas(self):
        base = _rel("a", [_lote("P1")]); cand = _rel("b", [_lote("P1")])
        self.assertTrue({g.id: g for g in pc.avalia_planejado(base, cand)}["P1"].passou)


if __name__ == "__main__":
    unittest.main()
