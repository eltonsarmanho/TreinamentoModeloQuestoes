"""Fase 1 — GUARDAS DE INFERÊNCIA (src/test_model.py e src/gerar_lote.py).

Cobre:
  1.1 rejeitar resposta fora das alternativas e re-amostrar;
  1.2 rejeitar alternativas degeneradas (duplicata disfarçada);
  1.3 permutação determinística da letra do gabarito.

Os casos marcados REAL são cópias literais de questões de
outputs/diversidade_*.json (relatórios de avaliação do modelo). Ficam embutidos
porque outputs/ é gitignored — o teste de PROPRIEDADE sobre >=200 questões reais
(TestPermutacaoCorpusReal) lê os relatórios quando existem e é pulado quando não.

    venv/bin/python -m pytest tests/test_guardas_inferencia.py -q
"""
import glob
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# test_model.py está neste mesmo diretório (tests/); sem isto, "import test_model"
# dependeria do modo de import do pytest em vez de ser explícito.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gerar_lote as gl  # noqa: E402
import test_model as tm  # noqa: E402
from schema_utils import (  # noqa: E402
    MOTIVO_FORA_DAS_ALTERNATIVAS,
    check_consistency_detalhado,
    check_structure,
    fix_gabarito,
)

LETRAS = "ABCDE"
FLAGS_OK = {"json_valido": True, "wrapper_valido": True, "quantidade_correta": True,
            "schema_completo": True, "resposta_valida": True,
            "alternativas_distintas": True, "difficulty_valida": True}


def _q(**kw):
    base = {
        "enunciado": "Quanto sobra?",
        "alternativas": {L: t for L, t in zip(LETRAS, ["60", "100", "110", "120", "130"])},
        "resolucao_passo_a_passo": "Subtraindo, 150 - 80 = 70.",
        "resposta_correta": "C",
        "difficulty": "MEDIUM",
    }
    base.update(kw)
    return base


# --- REAL: diversidade_exp_C_s2 / P12-5º-H21-N5 q#4 -------------------------
# Alternativas 60/100/110/120/130, gabarito C=110, resolução conclui 70.
# Nenhuma alternativa contém 70: a questão é IRRESPONDÍVEL.
REAL_FORA = _q()

# --- REAL: diversidade_atual / lote "ajustado" ------------------------------
# D e E são a MESMA opção; só o ponto final as separa, então a flag literal
# `alternativas_distintas` de check_structure aprova.
REAL_DEGENERADA = {
    "enunciado": "Qual coleção ocupa a maior área?",
    "alternativas": {
        "A": "Coleção 1 tem maior área.",
        "B": "Coleção 2 tem maior área.",
        "C": "As duas coleções têm a mesma área.",
        "D": "Nenhuma das alternativas anteriores.",
        "E": "Nenhuma das alternativas anteriores",
    },
    "resolucao_passo_a_passo": "Cada coleção ocupa 24 quadradinhos, logo empatam.",
    "resposta_correta": "C",
    "difficulty": "MEDIUM",
}

# --- REAL: diversidade_exp_C_s2 — o gabarito C tem um clone em D ------------
REAL_DEGENERADA_GABARITO = {
    "enunciado": "Que horas o ônibus chega?",
    "alternativas": {"A": "12h45", "B": "012h45", "C": "12:45",
                     "D": "12:45.", "E": "12h45e"},
    "resolucao_passo_a_passo": "O ônibus sai 12:00 e leva 45 minutos.",
    "resposta_correta": "C",
    "difficulty": "EASY",
}

# Questão limpa e permutável: sem âncora citada, sem letra no texto.
QUESTAO_SEGURA = {
    "enunciado": "Um livro tem 30 páginas e Ana leu 12. Quantas faltam?",
    "alternativas": {"A": "18 páginas", "B": "42 páginas", "C": "12 páginas",
                     "D": "20 páginas", "E": "Nenhuma das alternativas anteriores"},
    "resolucao_passo_a_passo": "Basta subtrair: 30 - 12 = 18.",
    "resposta_correta": "A",
    "difficulty": "EASY",
}


# ===========================================================================
# 1.1 — resposta fora das alternativas
# ===========================================================================
class TestGuardaForaDasAlternativas(unittest.TestCase):
    def test_caso_real_e_detectado_como_irrecuperavel(self):
        ok, sugestao, motivo = check_consistency_detalhado(REAL_FORA)
        self.assertIs(ok, False)
        self.assertIsNone(sugestao, "não há letra a sugerir: nada a corrigir")
        self.assertEqual(motivo, MOTIVO_FORA_DAS_ALTERNATIVAS)
        self.assertEqual(fix_gabarito(REAL_FORA)[1], "fora_das_alternativas")
        # e a estrutura está PERFEITA — por isso escapava do filtro estrutural
        self.assertTrue(all(check_structure({"questoes": [REAL_FORA]}, 1).values()))

    def test_escala_de_score(self):
        s = lambda **kw: tm._score_candidato(FLAGS_OK, **kw)
        consistente = s(consistente=True)
        nao_verif = s(consistente=None)
        corrigivel = s(consistente=False)
        fora = s(consistente=False, fora_das_alternativas=True)
        self.assertEqual((consistente, nao_verif, corrigivel, fora), (8, 6, 4, 2))
        # o que a Fase 1 exige: irrespondível vale MENOS que não verificável...
        self.assertLess(fora, nao_verif)
        # ...e menos que um gabarito meramente trocado (que fix_gabarito conserta)
        self.assertLess(fora, corrigivel)
        # ...e mais que estrutura quebrada, que nem chega a ser uma questão
        self.assertGreater(fora, tm._score_candidato(
            {**FLAGS_OK, "schema_completo": False}, consistente=True))

    def test_penalidade_visual_preservada_em_todas_as_faixas(self):
        visual = "Observe a figura abaixo e responda."
        for kw in ({"consistente": True}, {"consistente": None},
                   {"consistente": False},
                   {"consistente": False, "fora_das_alternativas": True}):
            with self.subTest(**kw):
                self.assertEqual(tm._score_candidato(FLAGS_OK, texto=visual, **kw),
                                 tm._score_candidato(FLAGS_OK, **kw) - 1)

    def test_retorno_antecipado_so_nos_aprovados_e_resolviveis(self):
        self.assertEqual(tm.SCORES_APROVADOS, (6, 8))
        self.assertEqual(tm._score_candidato(FLAGS_OK, consistente=True), 8)
        self.assertEqual(tm._score_candidato(FLAGS_OK, consistente=None), 6)
        # os scores com visual ausente NÃO param a busca
        for score in tm.SCORES_COM_VISUAL_AUSENTE:
            self.assertNotIn(score, tm.SCORES_APROVADOS)

    def test_generate_validated_prefere_reamostrar(self):
        """Candidato 0 irrespondível, candidato 1 bom: entrega o bom."""
        boa = _q(alternativas={L: t for L, t in zip(LETRAS, ["60", "70", "110", "120", "130"])},
                 resposta_correta="B")
        saidas = [REAL_FORA, boa]

        def gen(*a, seed=None, grammar=None, **kw):
            q = saidas[min(len(saidas) - 1, gen.n)]
            gen.n += 1
            return json.dumps({"questoes": [q]}, ensure_ascii=False), None, 30.0, 0.01
        gen.n = 0

        r = tm.generate_validated("cli", "m.gguf", "p", 4, 512, retries=1, gen_fn=gen)
        self.assertEqual(gen.n, 2, "tem de gastar a 2ª amostra em vez de entregar a 1ª")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["obj"]["resposta_correta"], "B")

    def test_corrigivel_vence_irreparavel_mesmo_chegando_depois(self):
        """Antes as duas valiam score 2 e `>` estrito fazia a 1ª vencer."""
        corrigivel = _q(resposta_correta="A")  # 70 não está; mas veja abaixo
        corrigivel["alternativas"] = {L: t for L, t in
                                      zip(LETRAS, ["60", "70", "110", "120", "130"])}
        # gabarito A=60, conta dá 70 -> fix_gabarito sugere B
        saidas = [REAL_FORA, corrigivel]

        def gen(*a, seed=None, grammar=None, **kw):
            q = saidas[min(len(saidas) - 1, gen.n)]
            gen.n += 1
            return json.dumps({"questoes": [q]}, ensure_ascii=False), None, 30.0, 0.01
        gen.n = 0

        r = tm.generate_validated("cli", "m.gguf", "p", 4, 512, retries=1, gen_fn=gen)
        self.assertEqual(r["status"], "corrigido")
        self.assertEqual(r["obj"]["resposta_correta"], "B")

    def test_status_continua_falha_para_nao_afrouxar_o_gate_g2(self):
        def gen(*a, seed=None, grammar=None, **kw):
            return json.dumps({"questoes": [REAL_FORA]}, ensure_ascii=False), None, 30.0, 0.01

        r = tm.generate_validated("cli", "m.gguf", "p", 4, 512, retries=1, gen_fn=gen)
        self.assertEqual(r["status"], "falha")  # promover_checkpoint.G2 exige falha==0
        self.assertEqual(r["motivo_consistencia"], MOTIVO_FORA_DAS_ALTERNATIVAS)


# ===========================================================================
# 1.2 — alternativas degeneradas
# ===========================================================================
class TestGuardaAlternativasDegeneradas(unittest.TestCase):
    def test_check_structure_pega_duplicata_literal(self):
        """A validação existente NÃO foi removida: duplicata exata segue reprovada."""
        q = _q()
        q["alternativas"]["E"] = q["alternativas"]["D"]
        self.assertFalse(check_structure({"questoes": [q]}, 1)["alternativas_distintas"])
        self.assertEqual(tm._score_candidato(
            check_structure({"questoes": [q]}, 1), consistente=None, questao=q), 0)

    def test_caso_real_escapava_por_um_ponto_final(self):
        # exatamente por isso a questão saiu no relatório: a flag literal aprova
        self.assertTrue(check_structure(
            {"questoes": [REAL_DEGENERADA]}, 1)["alternativas_distintas"])
        self.assertTrue(tm.alternativas_degeneradas(REAL_DEGENERADA))
        self.assertEqual(tm._score_candidato(FLAGS_OK, consistente=None,
                                             questao=REAL_DEGENERADA), 0)

    def test_caso_real_com_clone_do_proprio_gabarito(self):
        self.assertTrue(tm.alternativas_degeneradas(REAL_DEGENERADA_GABARITO))

    def test_nao_confunde_alternativas_legitimas(self):
        for q in (_q(), QUESTAO_SEGURA, REAL_FORA):
            with self.subTest(q=q["enunciado"][:30]):
                self.assertFalse(tm.alternativas_degeneradas(q))
        # separador decimal diferente NÃO é tratado como duplicata
        q = _q(alternativas={"A": "2,5", "B": "2.5", "C": "3", "D": "4", "E": "5"})
        self.assertFalse(tm.alternativas_degeneradas(q))

    def test_alternativa_vazia_e_degenerada(self):
        self.assertTrue(tm.alternativas_degeneradas(
            _q(alternativas={"A": "1", "B": "  ", "C": "3", "D": "4", "E": "5"})))


# ===========================================================================
# 1.1/1.2 no LOTE — orçamento extra de qualidade em gerar_lote
# ===========================================================================
def _gen_fixo(questoes):
    """gen_fn que devolve `questoes[i]` na i-ésima chamada (repete a última)."""
    def gen(llama_cli, gguf, prompt, threads, max_new_tokens, seed=None, grammar=None):
        q = questoes[min(len(questoes) - 1, gen.n)]
        gen.n += 1
        return json.dumps({"questoes": [q]}, ensure_ascii=False), None, 30.0, 0.01
    gen.n = 0
    return gen


class TestOrcamentoDeQualidade(unittest.TestCase):
    def _lote(self, gen, **kw):
        return gl.gerar_lote_planejado("cli", "m.gguf", "9º", "H99", "desc", "Moderado",
                                       1, 4, base_seed=5, gen_fn=gen, retries=0,
                                       max_tentativas_diversidade=0, **kw)

    def test_k0_ainda_reamostra_questao_irrespondivel(self):
        """Todos os relatórios rodaram com max_tentativas_diversidade=0."""
        boa = dict(QUESTAO_SEGURA)
        gen = _gen_fixo([REAL_FORA, boa])
        r = self._lote(gen)
        self.assertEqual(gen.n, 2, "sem TENTATIVAS_QUALIDADE isto era 1")
        self.assertNotEqual(r["detalhes"][0]["status"], "falha")

    def test_orcamento_de_qualidade_tem_teto(self):
        gen = _gen_fixo([REAL_FORA])  # nunca melhora
        r = self._lote(gen)
        self.assertEqual(gen.n, gl.TENTATIVAS_QUALIDADE + 1)
        self.assertEqual(r["detalhes"][0]["status"], "falha")
        self.assertEqual(len(r["questoes"]), 1, "nunca descarta o slot")

    def test_questao_boa_continua_custando_uma_chamada(self):
        gen = _gen_fixo([dict(QUESTAO_SEGURA)])
        self._lote(gen)
        self.assertEqual(gen.n, 1, "o orçamento extra não pode custar nada no caso comum")

    def test_orcamento_de_qualidade_nao_infla_regeneracoes_diversidade(self):
        """G11 e a comparação pareada leem regeneracoes_diversidade."""
        gen = _gen_fixo([REAL_FORA])
        r = self._lote(gen)
        self.assertEqual(gen.n, gl.TENTATIVAS_QUALIDADE + 1, "houve re-amostragem")
        self.assertEqual(r["regeneracoes_diversidade"], 0,
                         "re-amostragem por QUALIDADE não conta como de diversidade")

    def test_degenerada_tambem_dispara_reamostragem(self):
        gen = _gen_fixo([REAL_DEGENERADA, dict(QUESTAO_SEGURA)])
        r = self._lote(gen)
        self.assertEqual(gen.n, 2)
        self.assertNotEqual(r["detalhes"][0]["status"], "falha")


# ===========================================================================
# 1.3 — permutação
# ===========================================================================
class TestVetosPermutacao(unittest.TestCase):
    def test_questao_segura_nao_tem_veto(self):
        self.assertEqual(gl.vetos_permutacao(QUESTAO_SEGURA), [])

    def test_vetos(self):
        casos = {
            "resolucao_cita_letra": {"resolucao_passo_a_passo":
                                     "30 - 12 = 18. A resposta correta é A."},
            "enunciado_cita_letra": {"enunciado": "Qual? (A) 18 (B) 42 ..."},
            "enunciado_pede_ordem": {"enunciado": "Qual é a ordem crescente?"},
            "alternativa_referencia_outras": {"alternativas": {
                "A": "A e C", "B": "18 páginas", "C": "42 páginas",
                "D": "12 páginas", "E": "Nenhuma das alternativas anteriores"}},
            "alternativa_e_so_uma_letra": {"alternativas": {
                "A": "A", "B": "18 páginas", "C": "42 páginas",
                "D": "12 páginas", "E": "Nenhuma das alternativas anteriores"}},
            "alternativas_duplicadas": {"alternativas": {
                "A": "18 páginas", "B": "18 páginas", "C": "42 páginas",
                "D": "12 páginas", "E": "Nenhuma das alternativas anteriores"}},
            "sem_slots_livres": {"alternativas": {
                "A": "Nenhuma das alternativas anteriores",
                "B": "Todas as alternativas", "C": "N.D.A.",
                "D": "Nenhuma das opções", "E": "18 páginas"}},
        }
        for esperado, patch in casos.items():
            with self.subTest(esperado):
                q = {**QUESTAO_SEGURA, **patch}
                self.assertIn(esperado, gl.vetos_permutacao(q))
                self.assertIs(gl.permutar_alternativas(q, seed=1), q,
                              "questão vetada tem de voltar INTACTA")

    def test_enunciado_que_embute_a_lista(self):
        q = {**QUESTAO_SEGURA,
             "enunciado": "O painel é: A. Retângulo B. Quadrado C. Triângulo"}
        self.assertTrue(gl.vetos_permutacao(q))

    def test_schema_invalido(self):
        self.assertEqual(gl.vetos_permutacao({"alternativas": {}}), ["schema"])
        self.assertEqual(gl.vetos_permutacao(None), ["nao_e_dict"])
        self.assertEqual(gl.vetos_permutacao({**QUESTAO_SEGURA,
                                              "resposta_correta": "Z"}), ["schema"])


class TestPermutarAlternativas(unittest.TestCase):
    def test_invariantes(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                nova = gl.permutar_alternativas(QUESTAO_SEGURA, seed)
                orig = QUESTAO_SEGURA
                self.assertEqual(sorted(nova["alternativas"].values()),
                                 sorted(orig["alternativas"].values()))
                self.assertEqual(nova["alternativas"][nova["resposta_correta"]],
                                 orig["alternativas"][orig["resposta_correta"]])
                self.assertEqual(set(nova["alternativas"]), set(LETRAS))
                self.assertEqual(set(nova), set(orig), "schema do contrato intacto")
                self.assertEqual(nova["enunciado"], orig["enunciado"])
                self.assertEqual(nova["resolucao_passo_a_passo"],
                                 orig["resolucao_passo_a_passo"])
                self.assertEqual(nova["difficulty"], orig["difficulty"])

    def test_ancora_fica_congelada(self):
        for seed in range(12):
            nova = gl.permutar_alternativas(QUESTAO_SEGURA, seed)
            self.assertEqual(nova["alternativas"]["E"],
                             "Nenhuma das alternativas anteriores")
            self.assertNotEqual(nova["resposta_correta"], "E")

    def test_gabarito_na_ancora_e_no_op(self):
        q = {**QUESTAO_SEGURA, "resposta_correta": "E"}
        self.assertIs(gl.permutar_alternativas(q, 3), q)

    def test_determinismo(self):
        a = gl.permutar_alternativas(QUESTAO_SEGURA, 42)
        b = gl.permutar_alternativas(QUESTAO_SEGURA, 42)
        self.assertEqual(a, b)
        self.assertEqual(a["resposta_correta"],
                         gl.permutar_alternativas(dict(QUESTAO_SEGURA), 42)["resposta_correta"])

    def test_seeds_diferentes_movem_a_letra(self):
        letras = {gl.permutar_alternativas(QUESTAO_SEGURA, s)["resposta_correta"]
                  for s in range(50)}
        self.assertGreater(len(letras), 1)
        self.assertNotIn("E", letras)

    def test_alvo_explicito(self):
        nova = gl.permutar_alternativas(QUESTAO_SEGURA, 0, alvo="D")
        self.assertEqual(nova["resposta_correta"], "D")
        self.assertEqual(nova["alternativas"]["D"], "18 páginas")

    def test_consistencia_nunca_piora(self):
        ordem = {True: 2, None: 1, False: 0}
        for seed in range(30):
            nova = gl.permutar_alternativas(QUESTAO_SEGURA, seed)
            self.assertGreaterEqual(
                ordem[check_consistency_detalhado(nova)[0]],
                ordem[check_consistency_detalhado(QUESTAO_SEGURA)[0]])


class TestPermutarLote(unittest.TestCase):
    def _lote(self, n):
        return [{**QUESTAO_SEGURA,
                 "enunciado": f"Um livro tem {30 + i} páginas e Ana leu 12. Quantas faltam?",
                 "resolucao_passo_a_passo": f"Basta subtrair: {30 + i} - 12 = {18 + i}.",
                 "alternativas": {**QUESTAO_SEGURA["alternativas"], "A": f"{18 + i} páginas"},
                 "resposta_correta": "A"} for i in range(n)]

    def test_balanceamento(self):
        lote = self._lote(20)
        self.assertEqual(Counter(q["resposta_correta"] for q in lote), {"A": 20})
        novo = gl.permutar_lote(lote, seed=7)
        c = Counter(q["resposta_correta"] for q in novo)
        self.assertEqual(sorted(c.values()), [5, 5, 5, 5], "4 slots livres, 20 questões")
        self.assertNotIn("E", c)

    def test_determinismo_do_lote(self):
        lote = self._lote(9)
        self.assertEqual(gl.permutar_lote(lote, 3), gl.permutar_lote(lote, 3))
        self.assertNotEqual([q["resposta_correta"] for q in gl.permutar_lote(lote, 3)],
                            [q["resposta_correta"] for q in gl.permutar_lote(lote, 4)])

    def test_vetadas_passam_intactas_mas_contam_no_balanceamento(self):
        vetada = {**QUESTAO_SEGURA, "resposta_correta": "B",
                  "resolucao_passo_a_passo": "A resposta correta é B."}
        lote = [vetada] + self._lote(3)
        novo = gl.permutar_lote(lote, seed=1)
        self.assertIs(novo[0], vetada)
        # o slot B já estava usado, então as 3 seguintes evitam B primeiro
        self.assertNotIn("B", [q["resposta_correta"] for q in novo[1:]])

    def test_tolera_lixo_na_lista(self):
        self.assertEqual(gl.permutar_lote([None, "x"], 1), [None, "x"])


class TestPermutacaoNoPipeline(unittest.TestCase):
    """A permutação roda no FIM de gerar_lote_planejado, sobre o que é entregue."""

    def _lote(self, n, **kw):
        qs = [{**QUESTAO_SEGURA,
               "enunciado": f"Um livro tem {30 + i} páginas e Ana leu 12. Quantas faltam?",
               "resolucao_passo_a_passo": f"Basta subtrair: {30 + i} - 12 = {18 + i}.",
               "alternativas": {**QUESTAO_SEGURA["alternativas"], "A": f"{18 + i} páginas"},
               "resposta_correta": "A"} for i in range(n)]
        gen = _gen_fixo(qs)
        return gl.gerar_lote_planejado("cli", "m.gguf", "9º", "H99", "d", "Moderado",
                                       n, 4, base_seed=13, gen_fn=gen, retries=0,
                                       max_tentativas_diversidade=0, **kw)

    def test_lote_entregue_vem_balanceado(self):
        r = self._lote(8)
        c = Counter(q["resposta_correta"] for q in r["questoes"])
        self.assertEqual(max(c.values()), 2)
        self.assertEqual(r["permutacoes_gabarito"], 6)
        self.assertTrue(r["flags"]["quantidade_correta"])
        self.assertTrue(r["flags"]["alternativas_distintas"])
        self.assertEqual(set(r["obj"]), {"questoes"}, "schema do wrapper intacto")

    def test_pode_ser_desligada(self):
        r = self._lote(8, permutar=False)
        self.assertEqual({q["resposta_correta"] for q in r["questoes"]}, {"A"})
        self.assertEqual(r["permutacoes_gabarito"], 0)

    def test_determinismo_do_pipeline(self):
        self.assertEqual([q["resposta_correta"] for q in self._lote(6)["questoes"]],
                         [q["resposta_correta"] for q in self._lote(6)["questoes"]])


# ===========================================================================
# 1.3 — PROPRIEDADE sobre o corpus REAL (>=200 questões)
# ===========================================================================
def _questoes_reais():
    lotes = []
    for f in sorted(glob.glob(str(ROOT / "outputs" / "diversidade_*.json"))):
        if "dryrun" in f:  # GeradorSimulado, não é saída de modelo
            continue
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for mv in (d.get("modos") or {}).values():
            for lote in mv.get("lotes") or []:
                qs = [q for q in (lote.get("questoes") or []) if isinstance(q, dict)]
                if qs:
                    lotes.append(qs)
    return lotes


class TestPermutacaoCorpusReal(unittest.TestCase):
    """Permutar nunca pode alterar o veredito de check_consistency nem quebrar
    check_structure. Medido em 2026-09: 1.051 questões permutadas, 0 regressões."""

    @classmethod
    def setUpClass(cls):
        cls.lotes = _questoes_reais()
        cls.n = sum(len(l) for l in cls.lotes)
        if cls.n < 200:
            raise unittest.SkipTest(
                f"precisa de >=200 questões reais em outputs/diversidade_*.json "
                f"(encontradas: {cls.n}); outputs/ é gitignored")

    def test_propriedade_sobre_o_corpus(self):
        ordem = {True: 2, None: 1, False: 0}
        permutadas = 0
        for lote in self.lotes:
            novos = gl.permutar_lote(lote, seed=0)
            self.assertEqual(len(novos), len(lote))
            for antiga, nova in zip(lote, novos):
                if nova is antiga:
                    continue
                permutadas += 1
                with self.subTest(enunciado=str(antiga.get("enunciado"))[:60]):
                    self.assertEqual(sorted(nova["alternativas"].values()),
                                     sorted(antiga["alternativas"].values()))
                    self.assertEqual(nova["alternativas"][nova["resposta_correta"]],
                                     antiga["alternativas"][antiga["resposta_correta"]])
                    self.assertEqual(set(nova), set(antiga))
                    self.assertGreaterEqual(
                        ordem[check_consistency_detalhado(nova)[0]],
                        ordem[check_consistency_detalhado(antiga)[0]],
                        "check_consistency regrediu")
                    self.assertEqual(check_structure({"questoes": [nova]}, 1),
                                     check_structure({"questoes": [antiga]}, 1),
                                     "check_structure mudou")
        self.assertGreaterEqual(permutadas, 200,
                                "a amostra efetivamente permutada é pequena demais")

    def test_reduz_o_vies_de_letra(self):
        antes, depois = Counter(), Counter()
        for lote in self.lotes:
            for q in lote:
                antes[q.get("resposta_correta")] += 1
            for q in gl.permutar_lote(lote, seed=0):
                depois[q.get("resposta_correta")] += 1
        top_antes = 100 * max(antes.values()) / self.n
        top_depois = 100 * max(depois.values()) / self.n
        self.assertGreater(top_antes, 35.0, "o viés medido no corpus era de 38,5%")
        # teto absoluto folgado: a medição dá 23,1%, e o G6 reprova acima de 40%
        self.assertLess(top_depois, 30.0)
        self.assertLess(top_depois, top_antes - 10)

    def test_determinismo_sobre_o_corpus(self):
        for lote in self.lotes[:20]:
            self.assertEqual(gl.permutar_lote(lote, 5), gl.permutar_lote(lote, 5))


if __name__ == "__main__":
    unittest.main()
