"""Significância estatística nos gates de promoção (Fases 0.3 e 0.5).

Cobre: intervalo de Wilson, McNemar pareado no formato do perfil planejado
(lotes com listas de questões, não `detalhes`), a unificação da régua entre P3 e
G11 e o caso real que motivou a revisão de 2026-09-30.
"""
import json
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import promover_checkpoint as pc  # noqa: E402
from avaliar_diversidade import agregar  # noqa: E402

ALTS = {"A": "10", "B": "12", "C": "14", "D": "16", "E": "18"}


def _q(consistente=True):
    """5 + 7 = 12: gabarito B bate a conta; gabarito A não bate nenhuma
    alternativa e cai no ramo numérico de check_consistency."""
    return {"enunciado": "Some 5 e 7 e diga o total.",
            "alternativas": dict(ALTS),
            "resolucao_passo_a_passo": "Somando: 5 + 7 = 12.",
            "resposta_correta": "B" if consistente else "A",
            "difficulty": "MEDIUM"}


def _lote(pid, n, inconsistentes=0, gerada=None, aderentes=None, ano="9º", hab="H17"):
    """Lote no formato de avaliar_diversidade.metricas_lote, com as questões."""
    gerada = n if gerada is None else gerada
    qs = [_q(i >= inconsistentes) for i in range(gerada)]
    return {"id": pid, "ano": ano, "habilidade": hab, "dificuldade": "Moderado",
            "quantidade_pedida": n, "quantidade_gerada": gerada, "k_subtemas": 3,
            "aderencia_mensuravel": True, "chamadas": n, "json_validos": n,
            "schema_ok": gerada, "alternativas_distintas": gerada,
            "difficulty_correta": gerada,
            "consistencia": {"ok": gerada - inconsistentes, "nao_verificavel": 0,
                             "inconsistente": inconsistentes},
            "aderentes": gerada if aderentes is None else aderentes,
            "depende_de_visual": 0, "tempo_s": 10.0, "tempo_por_questao_s": 1.0,
            "coverage_score": 1.0, "duplicate_rate": 0.0, "diversity_score": 0.8,
            "structural_diversity": 0.8, "context_diversity": 0.8,
            "raciocinio_diversity": 0.8,
            "semantic_similarity": {"cosseno_media": 0.2, "jaccard_max": 0.1},
            "questoes": qs}


def _rel(sha, lotes, modo="ajustado"):
    return {"artefato_sha256": sha, "seed_offset": 0,
            "prompt_ids": [l["id"] for l in lotes],
            "max_tentativas_diversidade": 0,
            "modos": {modo: {"lotes": lotes, "agregado": agregar(lotes)}}}


def _gates(base, cand):
    return {g.id: g for g in pc.avalia_planejado(base, cand)}


class TestWilson(unittest.TestCase):
    def test_valores_conhecidos(self):
        # tabela clássica do intervalo de Wilson 95% (z = 1,96)
        self.assertEqual(pc.wilson(0, 10), (0.0, 27.75))
        self.assertEqual(pc.wilson(5, 10), (23.66, 76.34))
        self.assertEqual(pc.wilson(10, 10), (72.25, 100.0))

    def test_extremos_nao_degeneram(self):
        """O ponto do Wilson: 0/n e n/n não viram intervalo de largura zero,
        como aconteceria com o intervalo normal (Wald)."""
        lo, hi = pc.wilson(0, 30)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 10.0)  # 0/30 ainda admite até ~11%
        lo, hi = pc.wilson(30, 30)
        self.assertEqual(hi, 100.0)
        self.assertLess(lo, 90.0)

    def test_n_maior_estreita_o_intervalo(self):
        a = pc.wilson(1, 30)
        b = pc.wilson(10, 300)
        self.assertGreater(a[1] - a[0], b[1] - b[0])

    def test_n_zero(self):
        self.assertIsNone(pc.wilson(0, 0))

    def test_limites_no_intervalo_valido(self):
        for k, n in ((0, 1), (1, 1), (3, 7), (228, 228), (1, 228)):
            lo, hi = pc.wilson(k, n)
            self.assertGreaterEqual(lo, 0.0)
            self.assertLessEqual(hi, 100.0)
            self.assertLessEqual(lo, 100.0 * k / n)
            self.assertGreaterEqual(hi, 100.0 * k / n)


class TestContagem(unittest.TestCase):
    def test_deriva_numerador_e_denominador_dos_lotes(self):
        rel = _rel("a", [_lote("P1", 10, inconsistentes=2), _lote("P2", 10)])
        self.assertEqual(pc.contagem(rel, "ajustado", "consistencia_inconsistente_pct"), (2, 20))
        self.assertEqual(pc.contagem(rel, "ajustado", "aderencia_pct"), (20, 20))
        self.assertEqual(pc.contagem(rel, "ajustado", "json_valido_pct"), (20, 20))

    def test_aderencia_ignora_lote_nao_mensuravel(self):
        l = _lote("P2", 10)
        l["aderencia_mensuravel"] = False
        rel = _rel("a", [_lote("P1", 10), l])
        self.assertEqual(pc.contagem(rel, "ajustado", "aderencia_pct"), (10, 10))

    def test_sem_lotes_devolve_none(self):
        self.assertIsNone(pc.contagem({"modos": {"ajustado": {}}}, "ajustado", "json_valido_pct"))

    def test_ic_aparece_no_detalhe_do_gate(self):
        base, cand = _rel("a", [_lote("P1", 10)]), _rel("b", [_lote("P1", 10)])
        self.assertIn("IC95", _gates(base, cand)["P3"].detalhe)


class TestMcNemarPlanejado(unittest.TestCase):
    def test_conta_apenas_pares_discordantes(self):
        base = _rel("a", [_lote("P1", 10, inconsistentes=0)])
        cand = _rel("b", [_lote("P1", 10, inconsistentes=3)])
        piorou, melhorou, p, n, fora = pc.mcnemar_planejado(
            base, cand, "ajustado", "consistencia_inconsistente_pct")
        self.assertEqual((piorou, melhorou, n, fora), (3, 0, 10, 0))
        self.assertAlmostEqual(p, 2 * 1 / 2 ** 3)  # binomial exata, 3 discordantes

    def test_sem_discordancia_p_igual_a_um(self):
        base = _rel("a", [_lote("P1", 10, inconsistentes=2)])
        cand = _rel("b", [_lote("P1", 10, inconsistentes=2)])
        piorou, melhorou, p, n, _ = pc.mcnemar_planejado(
            base, cand, "ajustado", "consistencia_inconsistente_pct")
        self.assertEqual((piorou, melhorou, p, n), (0, 0, 1.0, 10))

    def test_melhora_tambem_e_detectada(self):
        base = _rel("a", [_lote("P1", 10, inconsistentes=6)])
        cand = _rel("b", [_lote("P1", 10, inconsistentes=0)])
        piorou, melhorou, p, _, _ = pc.mcnemar_planejado(
            base, cand, "ajustado", "consistencia_inconsistente_pct")
        self.assertEqual((piorou, melhorou), (0, 6))
        self.assertLess(p, 0.05)

    def test_lote_com_entrega_incompleta_fica_fora_do_pareamento(self):
        """Índice só é comparável se nenhum slot foi descartado: um descarte
        desloca todos os índices seguintes (ver gerar_lote.py:155)."""
        base = _rel("a", [_lote("P1", 10), _lote("P2", 10)])
        cand = _rel("b", [_lote("P1", 10, gerada=8), _lote("P2", 10)])
        piorou, melhorou, p, n, fora = pc.mcnemar_planejado(
            base, cand, "ajustado", "consistencia_inconsistente_pct")
        self.assertEqual((fora, n), (1, 10))  # só P2 entra no pareado

    def test_pareia_por_seed_e_prompt_id(self):
        """Dois relatórios de seeds diferentes com o MESMO prompt_id não podem
        colidir: junta_planejado marca seed_pareamento em cada lote."""
        b0, b1 = _lote("P1", 4, inconsistentes=0), _lote("P1", 4, inconsistentes=0)
        c0, c1 = _lote("P1", 4, inconsistentes=1), _lote("P1", 4, inconsistentes=2)
        b0["seed_pareamento"] = c0["seed_pareamento"] = 0
        b1["seed_pareamento"] = c1["seed_pareamento"] = 1
        base, cand = _rel("a", [b0, b1]), _rel("b", [c0, c1])
        piorou, melhorou, _, n, fora = pc.mcnemar_planejado(
            base, cand, "ajustado", "consistencia_inconsistente_pct")
        self.assertEqual((piorou, melhorou, n, fora), (3, 0, 8, 0))

    def test_chave_ambigua_nao_pareia(self):
        """Sem seed_pareamento, dois lotes com o mesmo id são indistinguíveis —
        melhor devolver None do que parear errado."""
        base = _rel("a", [_lote("P1", 4), _lote("P1", 4)])
        cand = _rel("b", [_lote("P1", 4), _lote("P1", 4)])
        self.assertIsNone(pc.mcnemar_planejado(
            base, cand, "ajustado", "consistencia_inconsistente_pct"))

    def test_metrica_sem_veredito_por_questao_nao_e_pareavel(self):
        base, cand = _rel("a", [_lote("P1", 10)]), _rel("b", [_lote("P1", 10)])
        self.assertIsNone(pc.mcnemar_planejado(base, cand, "ajustado", "json_valido_pct"))

    def test_relatorio_sem_questoes_nao_e_pareavel(self):
        base, cand = _rel("a", [_lote("P1", 10)]), _rel("b", [_lote("P1", 10)])
        for r in (base, cand):
            for l in r["modos"]["ajustado"]["lotes"]:
                l.pop("questoes")
        self.assertIsNone(pc.mcnemar_planejado(
            base, cand, "ajustado", "consistencia_inconsistente_pct"))

    def test_aderencia_tem_predicado_proprio(self):
        base = _rel("a", [_lote("P1", 10)])
        cand = _rel("b", [_lote("P1", 10)])
        r = pc.mcnemar_planejado(base, cand, "ajustado", "aderencia_pct")
        self.assertIsNotNone(r)
        self.assertEqual(r[3], 10)  # 10 pares (lote mensurável)


class TestNotaDeRegua(unittest.TestCase):
    """Os relatórios de outputs/ guardam a contagem feita pela versão de
    check_consistency vigente na hora da geração. Se a régua mudou depois, o
    veredito tem de dizer isso em vez de fingir que o número é atual."""

    def test_silencia_quando_relatorio_e_codigo_concordam(self):
        rel = _rel("a", [_lote("P1", 10, inconsistentes=3)])
        self.assertEqual(pc.recontagem(rel, "ajustado", "consistencia_inconsistente_pct"), (3, 10))
        self.assertEqual(pc.nota_regua(rel, rel, "ajustado",
                                       "consistencia_inconsistente_pct", 30.0, 30.0), "")

    def test_avisa_quando_a_contagem_gravada_ficou_velha(self):
        base = _rel("a", [_lote("P1", 10, inconsistentes=0)])
        cand = _rel("b", [_lote("P1", 10, inconsistentes=0)])
        # simula um relatório gravado com uma régua mais acusadora
        for r in (base, cand):
            r["modos"]["ajustado"]["agregado"]["consistencia_inconsistente_pct"] = 50.0
        nota = pc.nota_regua(base, cand, "ajustado",
                             "consistencia_inconsistente_pct", 50.0, 50.0)
        self.assertIn("régua ATUAL", nota)
        self.assertIn("0.0 -> 0.0", nota)

    def test_aderencia_conta_quem_adere(self):
        rel = _rel("a", [_lote("P1", 10)])
        k, n = pc.recontagem(rel, "ajustado", "aderencia_pct")
        self.assertEqual(n, 10)
        self.assertLessEqual(k, n)


class TestP3ComSignificancia(unittest.TestCase):
    def test_piora_grande_e_significativa_reprova(self):
        base = _rel("a", [_lote("P1", 100, inconsistentes=0)])
        cand = _rel("b", [_lote("P1", 100, inconsistentes=20)])
        g = _gates(base, cand)["P3"]
        self.assertFalse(g.passou)
        self.assertIn("NÃO demonstram", g.detalhe)

    def test_piora_acima_da_tolerancia_com_poucos_discordantes_reprova(self):
        """REESCRITO após a revisão adversarial de 2026-09-30.

        A versão anterior deste teste exigia que 1,0% -> 4,0% (4x o baseline,
        3pp acima de uma tolerância de 1pp) PASSASSE, porque McNemar dava
        p=0,25. Isso fixava o defeito como contrato: o p mínimo do McNemar exato
        bicaudal é 2/2^d, então d<=5 nunca atinge 0,05 e a regra "reprova só se
        p<ALFA" tornava P3 incapaz de reprovar qualquer deterioração pequena —
        que é o único regime em que ele opera.

        O contrato correto é o inverso: com pouquíssimos pares discordantes os
        dados NÃO demonstram que a piora cabe na tolerância, e um gate
        BLOQUEANTE reprova quando não tem prova de não-inferioridade.
        """
        base = _rel("a", [_lote("P1", 100, inconsistentes=1)])
        cand = _rel("b", [_lote("P1", 100, inconsistentes=4)])
        ag_b = base["modos"]["ajustado"]["agregado"]
        ag_c = cand["modos"]["ajustado"]["agregado"]
        self.assertEqual((ag_b["consistencia_inconsistente_pct"],
                          ag_c["consistencia_inconsistente_pct"]), (1.0, 4.0))
        g = _gates(base, cand)["P3"]
        self.assertFalse(g.passou)
        self.assertIn("NÃO demonstram", g.detalhe)

    def test_nao_inferioridade_demonstrada_absolve(self):
        """A absolvição estatística continua existindo — mas tem de ser
        DEMONSTRADA, e isso exige amostra. Com n grande e discordância
        equilibrada o limite superior da piora cabe na tolerância e o gate
        absolve, mesmo com o percentual gravado acima dela."""
        sup = pc.limite_superior_piora(piorou=12, melhorou=10, n=2000)
        self.assertLess(sup, 1.0)
        self.assertGreater(pc.limite_superior_piora(piorou=5, melhorou=0, n=228), 1.0)

    def test_limite_superior_negativo_quando_ha_melhora(self):
        """Corrige o segundo achado: o p bicaudal não enxerga o SENTIDO e podia
        'confirmar piora' sobre uma discordância favorável ao candidato."""
        self.assertLess(pc.limite_superior_piora(piorou=15, melhorou=33, n=201), 0.0)
        reprova, txt, estourou = pc._piorou_de_verdade(
            _rel("a", [_lote("P1", 100)]), _rel("b", [_lote("P1", 100)]),
            "ajustado", "aderencia_pct", vb=90.0, vc=80.0, sentido=1, tol=2.0)
        self.assertTrue(estourou)
        self.assertFalse(reprova)

    def test_sem_dados_pareados_vale_so_a_tolerancia(self):
        base = _rel("a", [_lote("P1", 100, inconsistentes=1)])
        cand = _rel("b", [_lote("P1", 100, inconsistentes=4)])
        for r in (base, cand):
            for l in r["modos"]["ajustado"]["lotes"]:
                l.pop("questoes")
        g = _gates(base, cand)["P3"]
        self.assertFalse(g.passou)
        self.assertIn("sem dados pareados", g.detalhe)

    def test_dentro_da_tolerancia_passa_e_mostra_o_teste(self):
        base = _rel("a", [_lote("P1", 100, inconsistentes=0)])
        cand = _rel("b", [_lote("P1", 100, inconsistentes=1)])
        g = _gates(base, cand)["P3"]
        self.assertTrue(g.passou)
        self.assertIn("informativo", g.detalhe)

    def test_significancia_nao_relaxa_piso_absoluto(self):
        """P1/P2 têm piso de contrato; não passam por McNemar."""
        base = _rel("a", [_lote("P1", 10)])
        cand = _rel("b", [_lote("P1", 10, gerada=8)])
        g = _gates(base, cand)
        self.assertFalse(g["P2"].passou)


class TestG11MesmaReguaQueP3(unittest.TestCase):
    """Fase 0.3: G11 e P3 mediam consistencia_inconsistente_pct com réguas
    diferentes (0pp x 1pp) e se contradiziam sobre a MESMA medição."""

    def test_tolerancia_de_g11_vem_de_p3(self):
        self.assertEqual(pc.tolerancia_diversidade("consistencia_inconsistente_pct"),
                         pc.MARGENS_PLANEJADO["P3"][2])
        self.assertEqual(pc.tolerancia_diversidade("aderencia_pct"),
                         pc.MARGENS_PLANEJADO["P4"][2])

    def test_metrica_sem_gate_p_mantem_o_padrao_de_0pp(self):
        self.assertEqual(pc.tolerancia_diversidade("schema_pct"), 0.0)
        base = _rel("a", [_lote("P1", 100)])
        cand = _rel("b", [_lote("P1", 100)])
        cand["modos"]["ajustado"]["agregado"]["schema_pct"] = 99.0
        g = pc.gate_diversidade(base, cand)
        self.assertFalse(g.passou)
        self.assertIn("schema_pct piorou", g.detalhe)

    def test_p3_e_g11_concordam_na_faixa_do_conflito(self):
        """1pp de piora: antes P3 PASSAVA e G11 FALHAVA sobre o mesmo número."""
        base = _rel("a", [_lote("P1", 100, inconsistentes=0)])
        cand = _rel("b", [_lote("P1", 100, inconsistentes=1)])
        p3 = _gates(base, cand)["P3"]
        g11 = pc.gate_diversidade(base, cand)
        self.assertTrue(p3.passou)
        self.assertTrue(g11.passou)

    def test_p3_e_g11_concordam_tambem_quando_reprovam(self):
        base = _rel("a", [_lote("P1", 100, inconsistentes=0)])
        cand = _rel("b", [_lote("P1", 100, inconsistentes=20)])
        self.assertFalse(_gates(base, cand)["P3"].passou)
        self.assertFalse(pc.gate_diversidade(base, cand).passou)

    def test_g11_continua_bloqueante_e_cobra_diversity_score(self):
        base = _rel("a", [_lote("P1", 100)])
        cand = _rel("b", [_lote("P1", 100)])
        cand["modos"]["ajustado"]["agregado"]["diversity_score"] = 0.79
        g = pc.gate_diversidade(base, cand)
        self.assertTrue(g.bloqueante)
        self.assertFalse(g.passou)
        self.assertIn("diversity_score piorou", g.detalhe)

    def test_modos_diferentes_nao_usam_teste_pareado(self):
        """atual x ajustado tem os mesmos prompts, mas pipelines diferentes:
        comparar questão a questão por índice não é pareamento."""
        base = _rel("a", [_lote("P1", 100, inconsistentes=1)], modo="atual")
        cand = _rel("b", [_lote("P1", 100, inconsistentes=4)])
        g = pc.gate_diversidade(base, cand, "atual", "ajustado")
        self.assertFalse(g.passou)
        self.assertIn("sem dados pareados", g.detalhe)


REAIS = [RAIZ / "outputs" / f"diversidade_{n}.json" for n in
         ("base_k0_s0", "base_k0_s1", "base_k0_s2", "exp_C_s0", "exp_C_s1", "exp_C_s2")]


@unittest.skipUnless(all(p.exists() for p in REAIS), "relatórios reais ausentes")
class TestCasoRealQueMotivouAMudanca(unittest.TestCase):
    """base_k0_s{0,1,2} x exp_C_s{0,1,2}, 2026-09-30: inconsistentes 1,32% ->
    2,19%. P3 passava (tolerância 1pp) e G11 reprovava (0pp) — o veredito
    dependia de qual gate se olhava."""

    @classmethod
    def setUpClass(cls):
        cls.base, _ = pc.junta_planejado([str(p) for p in REAIS[:3]])
        cls.cand, _ = pc.junta_planejado([str(p) for p in REAIS[3:]])

    def test_a_medicao_e_a_que_motivou_a_revisao(self):
        b = pc.agregado_diversidade(self.base)["consistencia_inconsistente_pct"]
        c = pc.agregado_diversidade(self.cand)["consistencia_inconsistente_pct"]
        self.assertEqual((b, c), (1.32, 2.19))

    def test_junta_planejado_marca_a_seed_em_cada_lote(self):
        seeds = {l["seed_pareamento"] for l in self.base["modos"]["ajustado"]["lotes"]}
        self.assertEqual(seeds, {0, 1, 2})

    def test_p3_e_g11_concordam_na_piora_relativa(self):
        """A régua RELATIVA (tolerância + não-inferioridade) é a mesma nos dois.
        O que pode separá-los é só o teto absoluto de 2% (pré-registro 2026-09-30),
        que pertence a P3 e é um motivo distinto e nomeado no detalhe. Este teste
        NÃO fixa veredito algum: o resultado sobre estes relatórios é 2,19% > 2%,
        logo P3 reprova pelo teto — e o detalhe tem de dizer isso."""
        p3 = _gates(self.base, self.cand)["P3"]
        g11 = pc.gate_diversidade(self.base, self.cand)
        piora_relativa_reprova = "NÃO demonstram" in p3.detalhe
        self.assertEqual(piora_relativa_reprova, not g11.passou)
        self.assertIn("limite absoluto", p3.detalhe)
        self.assertFalse(p3.passou)

    def test_o_teste_pareado_roda_sobre_os_228_pares(self):
        r = pc.mcnemar_planejado(self.base, self.cand, "ajustado",
                                 "consistencia_inconsistente_pct")
        piorou, melhorou, p, n, fora = r
        self.assertEqual((n, fora), (228, 0))
        self.assertGreaterEqual(p, 0.05)  # a piora observada é ruído

    def test_aderencia_melhorou_de_forma_significativa(self):
        piorou, melhorou, p, n, _ = pc.mcnemar_planejado(
            self.base, self.cand, "ajustado", "aderencia_pct")
        # 'piorou' = aderia no baseline e deixou de aderir no candidato
        self.assertEqual((piorou, melhorou, n), (15, 33, 201))
        self.assertLess(p, 0.05)

    def test_veredito_nao_e_fixado_por_este_teste(self):
        """Só verifica que cada gate reprovado nomeia o motivo no detalhe —
        nunca que o veredito é PROMOVIDO/NÃO PROMOVIDO (isso é decisão do gate
        sobre os dados, não algo que a suíte deva defender)."""
        gates = pc.avalia_planejado(self.base, self.cand)
        gates.append(pc.gate_diversidade(self.base, self.cand))
        for g in gates:
            if g.bloqueante and not g.passou:
                self.assertTrue(g.detalhe.strip(), f"{g.id} reprovou sem motivo")


if __name__ == "__main__":
    unittest.main()
