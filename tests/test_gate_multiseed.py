"""Perfil multiseed dos gates (revisão de 2026-10-05).

Cobre: G2/G3 agregados por McNemar sobre (rodada, item), guarda absoluta de
falhas, demais gates exigidos em todas as rodadas, aborto sem pareamento e o
caso real que motivou a revisão (VEREDITO_v3: um único item reprovando G2 e G3).
"""
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import promover_checkpoint as pc  # noqa: E402

N = 30


def _rel(rodada, sha, falhas=(), inconsistentes=(), tps=30.0, json_pct=100.0):
    """Relatório test_model.py mínimo: itens 0..N-1, os 10 primeiros verificáveis."""
    det = []
    for i in range(N):
        verif = i < 10
        consistente = None if not verif else i not in inconsistentes
        status = "falha" if i in falhas else ("ok" if verif else "nao_verificavel")
        det.append({"codigo_item_ref": f"IT{i:02d}", "status": status,
                    "consistencia_resposta_correta": consistente,
                    "difficulty_aderente": True})
    ok = sum(1 for x in det if x["consistencia_resposta_correta"])
    estrutura = {f: 100.0 for f in pc.FLAGS_ESTRUTURAIS}
    estrutura.update({
        "json_valido_pct": json_pct,
        "consistencia_resposta_correta_pct": round(100 * ok / 10, 1),
        "consistencia_verificavel_n": 10,
        "depende_de_visual_ausente_pct": 0.0,
        "difficulty_aderente_pct": 100.0,
        "gabarito_letra_mais_frequente_pct": 30.0,
        "distribuicao_respostas_corretas": {"A": 9, "B": 6, "C": 5, "D": 5, "E": 5},
    })
    return {"artefato": sha, "artefato_sha256": sha, "seed_rodada": rodada, "retries": 1,
            "conjunto_avaliacao": "data/val_frozen_v1.jsonl", "num_amostras": N,
            "estrutura": estrutura,
            "por_ano": {"5º": {"estrutura_ok_pct": 100.0}, "9º": {"estrutura_ok_pct": 100.0}},
            "pos_processamento": {"falha": len(falhas), "regeneracoes_total": 5},
            "velocidade_cpu_real": {"tokens_por_segundo_geracao": tps},
            "detalhes": det}


def _gate(gates, ident):
    return next(g for g in gates if g.id == ident)


class TestMultiseed(unittest.TestCase):
    def test_caso_v3_um_item_nao_reprova(self):
        """1 falha/inconsistência em 90 pares não é distinguível de ruído."""
        bases = [_rel(r, "base") for r in range(3)]
        cands = [_rel(0, "cand", falhas={5}, inconsistentes={5}),
                 _rel(1, "cand"), _rel(2, "cand")]
        gates, _, _, ac = pc.avalia_multiseed(bases, cands)
        self.assertTrue(_gate(gates, "G2*").passou)
        self.assertTrue(_gate(gates, "G3*").passou)
        self.assertEqual(ac["n"], 90)
        self.assertNotIn("G2", [g.id for g in gates])
        self.assertNotIn("G3", [g.id for g in gates])

    def test_piora_significativa_reprova_g3(self):
        bases = [_rel(r, "base") for r in range(3)]
        cands = [_rel(r, "cand", inconsistentes={0, 1, 2}) for r in range(3)]
        gates, _, _, _ = pc.avalia_multiseed(bases, cands)
        self.assertFalse(_gate(gates, "G3*").passou)  # 9 piorados, 0 melhorados

    def test_teto_absoluto_de_falhas(self):
        """Empatar com um baseline ruim não basta: > 5% de falha reprova."""
        ruins = {0, 1}
        bases = [_rel(r, "base", falhas=ruins) for r in range(3)]
        cands = [_rel(r, "cand", falhas=ruins) for r in range(3)]
        gates, _, _, _ = pc.avalia_multiseed(bases, cands)
        self.assertFalse(_gate(gates, "G2*").passou)  # 6/90 = 6,7%

    def test_outro_gate_precisa_passar_em_todas_as_rodadas(self):
        bases = [_rel(r, "base") for r in range(3)]
        cands = [_rel(0, "cand"), _rel(1, "cand", tps=10.0), _rel(2, "cand")]
        gates, _, _, _ = pc.avalia_multiseed(bases, cands)
        g8 = _gate(gates, "G8")
        self.assertFalse(g8.passou)
        self.assertIn("rodada 1", g8.detalhe)

    def test_rodadas_nao_pareadas_abortam(self):
        with self.assertRaises(SystemExit):
            pc.avalia_multiseed([_rel(r, "b") for r in (0, 1, 2)],
                                [_rel(r, "c") for r in (0, 1, 3)])

    def test_menos_de_tres_rodadas_aborta(self):
        """Com 1 rodada seria a régua frouxa sobre os mesmos 30 itens."""
        with self.assertRaises(SystemExit):
            pc.avalia_multiseed([_rel(0, "b")], [_rel(0, "c", falhas={5})])

    def test_mesmo_artefato_aborta(self):
        with self.assertRaises(SystemExit):
            pc.avalia_multiseed([_rel(r, "x") for r in range(3)], [_rel(r, "x") for r in range(3)])

    def test_relatorio_antigo_sem_campo_vale_rodada_zero(self):
        b = _rel(0, "base")
        del b["seed_rodada"]
        gates, _, _, _ = pc.avalia_multiseed([b, _rel(1, "base"), _rel(2, "base")],
                                             [_rel(r, "cand") for r in range(3)])
        self.assertTrue(_gate(gates, "G2*").passou)

    def test_ganho_agregado(self):
        bases = [_rel(r, "base", inconsistentes={0}) for r in range(3)]
        cands = [_rel(r, "cand") for r in range(3)]
        _, ganhos, _, _ = pc.avalia_multiseed(bases, cands)
        self.assertTrue(any(g.startswith("consistência") for g in ganhos))


class TestCompararInferencia(unittest.TestCase):
    """Mesmo .gguf, configuração de inferência diferente (motor/threads/cache)."""

    def _par(self, lat_b=10.0, lat_c=5.0, inf_b="cli", inf_c="server", sha_c="m"):
        def rel(r, sha, inf, lat):
            x = _rel(r, sha)
            x["inferencia"] = {"motor": inf}
            x["velocidade_cpu_real"]["latencia_media_total_s"] = lat
            return x
        return ([rel(r, "m", inf_b, lat_b) for r in range(3)],
                [rel(r, sha_c, inf_c, lat_c) for r in range(3)])

    def test_mesmo_sha_aceito_e_latencia_conta_como_ganho(self):
        gates, ganhos, _, _ = pc.avalia_multiseed(*self._par(), comparar_inferencia=True)
        self.assertTrue(_gate(gates, "G2*").passou)
        self.assertTrue(any(g.startswith("latência") for g in ganhos))

    def test_sem_a_opcao_mesmo_sha_continua_abortando(self):
        with self.assertRaises(SystemExit):
            pc.avalia_multiseed(*self._par())

    def test_mesma_inferencia_aborta(self):
        with self.assertRaises(SystemExit):
            pc.avalia_multiseed(*self._par(inf_c="cli"), comparar_inferencia=True)

    def test_artefatos_diferentes_abortam(self):
        with self.assertRaises(SystemExit):
            pc.avalia_multiseed(*self._par(sha_c="outro"), comparar_inferencia=True)

    def test_latencia_pior_nao_e_ganho(self):
        _, ganhos, _, _ = pc.avalia_multiseed(*self._par(lat_c=12.0), comparar_inferencia=True)
        self.assertFalse(any(g.startswith("latência") for g in ganhos))


class TestG6Agregado(unittest.TestCase):
    """Revisão de 2026-10-09: G6 decide sobre a contagem de letras somada."""

    def _com_letras(self, rodada, sha, dist):
        x = _rel(rodada, sha)
        x["estrutura"]["distribuicao_respostas_corretas"] = dist
        x["estrutura"]["gabarito_letra_mais_frequente_pct"] = round(
            100 * max(dist.values()) / sum(dist.values()), 1)
        return x

    def test_caso_real_ruido_numa_rodada_passa(self):
        bases = [self._com_letras(0, "b", {"A": 7, "B": 11, "C": 5, "D": 4, "E": 3}),
                 self._com_letras(1, "b", {"A": 10, "B": 5, "C": 7, "D": 4, "E": 4}),
                 self._com_letras(2, "b", {"A": 10, "B": 10, "C": 6, "D": 2, "E": 2})]
        cands = [self._com_letras(0, "c", {"A": 8, "B": 13, "C": 5, "D": 2, "E": 2}),
                 self._com_letras(1, "c", {"A": 8, "B": 6, "C": 8, "D": 4, "E": 4}),
                 self._com_letras(2, "c", {"A": 11, "B": 9, "C": 4, "D": 4, "E": 2})]
        gates, _, _, _ = pc.avalia_multiseed(bases, cands)
        self.assertTrue(_gate(gates, "G6*").passou)
        self.assertNotIn("G6", [g.id for g in gates])

    def test_vies_agregado_acima_da_tolerancia_reprova(self):
        bal = {"A": 6, "B": 6, "C": 6, "D": 6, "E": 6}
        bases = [self._com_letras(r, "b", bal) for r in range(3)]
        cands = [self._com_letras(r, "c", {"A": 12, "B": 6, "C": 4, "D": 4, "E": 4}) for r in range(3)]
        gates, _, _, _ = pc.avalia_multiseed(bases, cands)
        self.assertFalse(_gate(gates, "G6*").passou)  # 20% -> 40%


if __name__ == "__main__":
    unittest.main()
