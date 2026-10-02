"""Integração do verificador de geometria ao pipeline de PRODUÇÃO.

Caso que motivou (teste real do usuário, 2026-10-01): 20 questões de 9º H17
geradas pelo modelo promovido em modo planejado; a auditoria humana
(outputs/testes_locais/Log.txt) aprovou só 5. O best-of-N não regenerou
NENHUMA: o verificador aritmético dava "sem_conta" nas 20, então todas saíam
como "nao_verificavel" (score 6, aprovado) na 1ª amostra.

Cobre, sem llama-cli (gen_fn mockado, padrão de tests/test_gerar_lote.py):
  * test_model._score_candidato / generate_validated: reprovação de geometria
    vale menos que "não verificável" e re-amostra; "ok" conta como verificada;
    a letra NUNCA é trocada com base na geometria; modo "sombra" = antigo;
  * gerar_lote: 1º candidato premissa_impossivel, 2º ok -> entrega o 2º;
    orçamento de qualidade; guarda de dados insuficientes do plano;
    permutação re-verificada pela geometria (alternativas "barraca A/B/C");
  * avaliar_diversidade: contagem de vereditos SEM mudar métricas existentes;
  * test_model.batch: seção "geometria" no relatório.

As questões auditadas vêm de tests/test_verificador_geometria.AUDITADAS
(cópias literais de outputs/testes_locais/teste_20261001_*.json).

    venv/bin/python -m pytest tests/test_integracao_geometria.py -q
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import avaliar_diversidade as av  # noqa: E402
import diversidade as dv  # noqa: E402
import gerar_lote as gl  # noqa: E402
import test_model as tm  # noqa: E402
import verificador_geometria as vg  # noqa: E402
from test_verificador_geometria import APROVAR, AUDITADAS  # noqa: E402

TAX = dv.carregar_taxonomia()
FLAGS_OK = {"json_valido": True, "wrapper_valido": True, "quantidade_correta": True,
            "schema_completo": True, "resposta_valida": True,
            "alternativas_distintas": True, "difficulty_valida": True}

PREMISSA = AUDITADAS["R1-Q5"]   # triângulo com dois ângulos retos
CORRETA = AUDITADAS["R1-Q7"]    # lados 5, 5, 8 -> isósceles (auditoria: correta)
CONTA_ERRADA = AUDITADAS["R1-Q3"]  # 5, 7, 9 -> "Acutângulo"; o certo é C (25+49<81)
BARRACAS = AUDITADAS["R1-Q1"]   # alternativas compostas "A barraca A é ..., a barraca B é ..."

# REAL: corpus gerado G-9H17-0023 (outputs/diversidade_*.json). Nenhuma medida
# nem propriedade no enunciado; o verificador de geometria fica em
# nao_aplicavel (não entende "um lado maior"), mas a guarda do plano pega.
SEM_DADOS = {
    "enunciado": "Triângulo com um lado maior que os outros dois: classifique pelo ângulo interno.",
    "alternativas": {"A": "Retângulo", "B": "Triângulo escaleno", "C": "Triângulo obtusângulo",
                     "D": "Triângulo acutângulo", "E": "Nenhuma das alternativas anteriores"},
    "resolucao_passo_a_passo": "Triângulo obtusângulo tem um ângulo obtuso (maior que 90°).",
    "resposta_correta": "C", "difficulty": "EASY",
}


def _gen_fixo(questoes):
    """gen_fn que devolve `questoes[i]` na i-ésima chamada (repete a última)."""
    def gen(llama_cli, gguf, prompt, threads, max_new_tokens, seed=None, grammar=None):
        gen.prompts.append(prompt)
        q = questoes[min(len(questoes) - 1, gen.n)]
        gen.n += 1
        return json.dumps({"questoes": [q]}, ensure_ascii=False), None, 30.0, 0.01
    gen.n, gen.prompts = 0, []
    return gen


def _gv(gen, **kw):
    # Estes testes exercitam o modo "ativo" EXPLICITAMENTE: desde a revisão de
    # 2026-10-01 o padrão de produção é "sombra" (ver TestCorrecoesRevisao).
    kw.setdefault("retries", 1)
    kw.setdefault("modo_geometria", "ativo")
    return tm.generate_validated("cli", "m.gguf", "p", 4, 512, gen_fn=gen, **kw)


# ===========================================================================
# Escala de score
# ===========================================================================
class TestEscalaComGeometria(unittest.TestCase):
    def test_reprovacao_vale_menos_que_nao_verificavel(self):
        nao_verif = tm._score_candidato(FLAGS_OK, None)
        for v in sorted(vg.GEO_REJEITA):
            with self.subTest(veredito=v):
                s = tm._score_candidato(FLAGS_OK, None, geometria=v)
                # mesma faixa da resposta fora das alternativas
                self.assertEqual(s, tm._score_candidato(FLAGS_OK, False,
                                                        fora_das_alternativas=True))
                self.assertLess(s, nao_verif)
                self.assertNotIn(s, tm.SCORES_APROVADOS)
                # abaixo também do "corrigível": geometria não se corrige por letra
                self.assertLess(s, tm._score_candidato(FLAGS_OK, False))
                # e acima de estrutura quebrada
                self.assertGreater(s, tm.SCORE_ESTRUTURA_QUEBRADA)

    def test_reprovacao_prevalece_sobre_conta_que_bate(self):
        for v in sorted(vg.GEO_REJEITA):
            with self.subTest(veredito=v):
                self.assertEqual(tm._score_candidato(FLAGS_OK, True, geometria=v), 2)

    def test_ok_conta_como_verificada(self):
        self.assertEqual(tm._score_candidato(FLAGS_OK, None, geometria="ok"), 8)
        self.assertEqual(tm._score_candidato(FLAGS_OK, True, geometria="ok"), 8)

    def test_nao_aplicavel_e_none_mantem_a_escala_antiga(self):
        for c, esperado in ((True, 8), (None, 6), (False, 4)):
            for g in (None, "nao_aplicavel"):
                with self.subTest(consistente=c, geometria=g):
                    self.assertEqual(tm._score_candidato(FLAGS_OK, c, geometria=g), esperado)

    def test_contradicao_ok_x_conta_regenera(self):
        # fix_gabarito trocaria para uma letra que a geometria provou errada
        self.assertTrue(tm.geometria_bloqueia("ok", False))
        self.assertEqual(tm._score_candidato(FLAGS_OK, False, geometria="ok"), 2)

    def test_penalidade_visual_continua_dentro_da_faixa(self):
        visual = "Observe a figura abaixo e responda."
        for g in ("ok", "premissa_impossivel"):
            with self.subTest(geometria=g):
                self.assertEqual(tm._score_candidato(FLAGS_OK, None, visual, geometria=g),
                                 tm._score_candidato(FLAGS_OK, None, geometria=g) - 1)

    def test_vinte_auditadas(self):
        """15 erradas viram regeneração; as 5 corretas viram 'ok' (score 8)."""
        for id_, q in AUDITADAS.items():
            with self.subTest(id_=id_):
                gen = _gen_fixo([q])
                r = _gv(gen, retries=0)
                if id_ in APROVAR:
                    self.assertEqual(r["status"], "ok")
                else:
                    self.assertEqual(r["status"], "falha")
                    self.assertIn(r["geometria"]["veredito"], vg.GEO_REJEITA)


# ===========================================================================
# generate_validated
# ===========================================================================
class TestGenerateValidated(unittest.TestCase):
    def test_premissa_impossivel_depois_ok_entrega_o_segundo(self):
        gen = _gen_fixo([PREMISSA, CORRETA])
        r = _gv(gen)
        self.assertEqual(gen.n, 2, "tem de gastar a 2ª amostra em vez de entregar a 1ª")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["obj"]["enunciado"], CORRETA["enunciado"])
        self.assertEqual(r["geometria"]["veredito"], "ok")
        self.assertEqual(r["regeneracoes"], 1)
        self.assertEqual(r["reprovacoes_geometria"], 1)

    def test_ok_na_primeira_custa_uma_chamada(self):
        gen = _gen_fixo([CORRETA])
        r = _gv(gen, retries=3)
        self.assertEqual(gen.n, 1)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["reprovacoes_geometria"], 0)

    def test_gabarito_errado_nunca_e_corrigido_pela_letra(self):
        """R1-Q3: a geometria sugere C, mas a resolução diz 'acutângulo'.
        Trocar só a letra entregaria C com uma resolução que defende B."""
        v, det = vg.verificar_geometria(CONTA_ERRADA)
        self.assertEqual((v, det["sugestao"]), ("gabarito_errado", "C"))
        gen = _gen_fixo([CONTA_ERRADA])
        r = _gv(gen, retries=2)
        self.assertEqual(gen.n, 3, "re-amostra até esgotar")
        self.assertEqual(r["status"], "falha")  # G2 continua enxergando
        self.assertEqual(r["obj"]["resposta_correta"], "B", "fix_gabarito não pode rodar")
        self.assertEqual(r["obj"]["resolucao_passo_a_passo"],
                         CONTA_ERRADA["resolucao_passo_a_passo"])

    def test_reprovada_com_visual_nao_vira_depende_de_visual(self):
        """depende_de_visual é rank 1 (utilizável) em gerar_lote: uma questão
        errada pela geometria não pode escapar por essa porta."""
        q = dict(PREMISSA, enunciado="Observe a figura abaixo. " + PREMISSA["enunciado"])
        r = _gv(_gen_fixo([q]), retries=0)
        self.assertEqual(r["status"], "falha")

    def test_modo_sombra_reproduz_o_comportamento_antigo(self):
        gen = _gen_fixo([PREMISSA, CORRETA])
        r = _gv(gen, modo_geometria="sombra")
        self.assertEqual(gen.n, 1, "em sombra a 1ª amostra é aceita, como antes")
        self.assertEqual(r["status"], "nao_verificavel")
        self.assertEqual(r["geometria"]["veredito"], "premissa_impossivel")
        self.assertEqual(r["geometria"]["modo"], "sombra")
        self.assertEqual(r["reprovacoes_geometria"], 1)

    def test_modo_invalido(self):
        with self.assertRaises(ValueError):
            _gv(_gen_fixo([CORRETA]), modo_geometria="desligado")

    def test_modo_explicito_ativo(self):
        self.assertEqual(_gv(_gen_fixo([CORRETA]))["geometria"]["modo"], "ativo")

    def test_questao_fora_de_geometria_nao_muda(self):
        """Fora de H17 o verificador é nao_aplicavel: mesmo status e custo."""
        q = {"enunciado": "Um livro tem 30 páginas e Ana leu 12. Quantas faltam?",
             "alternativas": {"A": "18 páginas", "B": "42 páginas", "C": "12 páginas",
                              "D": "20 páginas", "E": "Nenhuma das alternativas anteriores"},
             "resolucao_passo_a_passo": "Basta subtrair: 30 - 12 = 18.",
             "resposta_correta": "A", "difficulty": "EASY"}
        for modo in ("ativo", "sombra"):
            with self.subTest(modo=modo):
                gen = _gen_fixo([q])
                r = _gv(gen, modo_geometria=modo)
                self.assertEqual((gen.n, r["status"], r["geometria"]["veredito"]),
                                 (1, "ok", "nao_aplicavel"))


# ===========================================================================
# gerar_lote
# ===========================================================================
def _seed_com_subtema(subtema):
    for s in range(1, 50):
        if dv.planejar_lote("9º", "H17", 1, "Fácil", seed=s, taxonomia=TAX)[0]["subtema"] == subtema:
            return s
    raise AssertionError("nenhuma seed")


class TestGerarLote(unittest.TestCase):
    def _lote(self, gen, hab="H17", **kw):
        kw.setdefault("base_seed", _seed_com_subtema("triangulo"))
        kw.setdefault("retries", 0)
        kw.setdefault("max_tentativas_diversidade", 0)
        kw.setdefault("modo_geometria", "ativo")
        return gl.gerar_lote_planejado("cli", "m.gguf", "9º", hab, "desc", "Fácil", 1, 4,
                                       gen_fn=gen, taxonomia=TAX, **kw)

    def test_premissa_impossivel_depois_ok_entrega_o_segundo(self):
        gen = _gen_fixo([PREMISSA, CORRETA])
        r = self._lote(gen)
        self.assertEqual(gen.n, 2)
        self.assertEqual(r["questoes"][0]["enunciado"], CORRETA["enunciado"])
        d = r["detalhes"][0]
        self.assertEqual((d["status"], d["geometria"], d["reprovacoes_geometria"]),
                         ("ok", "ok", 1))
        self.assertEqual(r["regeneracoes_diversidade"], 0, "qualidade não infla G11")
        self.assertEqual(set(r["obj"]), {"questoes"})  # schema inalterado

    def test_com_retries_internos_tambem(self):
        gen = _gen_fixo([PREMISSA, CORRETA])
        r = self._lote(gen, retries=1)
        self.assertEqual(gen.n, 2, "a 2ª amostra sai do best-of-N interno")
        self.assertEqual(r["detalhes"][0]["status"], "ok")

    def test_orcamento_de_qualidade_tem_teto_e_nao_descarta_slot(self):
        gen = _gen_fixo([PREMISSA])
        r = self._lote(gen, retries=1)
        self.assertEqual(gen.n, (gl.TENTATIVAS_QUALIDADE + 1) * 2)
        self.assertEqual(r["detalhes"][0]["status"], "falha")
        self.assertEqual(len(r["questoes"]), 1)

    def test_sombra_custa_uma_chamada(self):
        gen = _gen_fixo([PREMISSA, CORRETA])
        r = self._lote(gen, modo_geometria="sombra")
        self.assertEqual(gen.n, 1)
        self.assertEqual(r["detalhes"][0]["geometria"], "premissa_impossivel")


class TestGuardaDadosInsuficientes(unittest.TestCase):
    def setUp(self):
        self.slot = dv.planejar_lote("9º", "H17", 1, "Fácil",
                                     seed=_seed_com_subtema("triangulo"), taxonomia=TAX)[0]

    def test_peso_igual_a_dados_ausentes(self):
        self.assertEqual(gl.PESO_VIOLACAO["dados_insuficientes"],
                         gl.PESO_VIOLACAO["dados_ausentes"])

    def test_casos_auditados(self):
        # R1-Q2 (ônibus) e R2-Q7 (garrafas PET): rotulados dados_ausentes
        for id_ in ("R1-Q2", "R2-Q7"):
            with self.subTest(id_=id_):
                tipos = [v["tipo"] for v in gl.violacoes_slot([], AUDITADAS[id_], self.slot,
                                                              "9º", "H17", 1, TAX)]
                self.assertIn("dados_insuficientes", tipos)
        for id_ in sorted(APROVAR):
            with self.subTest(id_=id_):
                tipos = [v["tipo"] for v in gl.violacoes_slot([], AUDITADAS[id_], self.slot,
                                                              "9º", "H17", 1, TAX)]
                self.assertNotIn("dados_insuficientes", tipos)

    def test_nao_acusa_pergunta_que_nao_e_de_classificar(self):
        q = {"enunciado": "Quantos lados tem um triângulo?",
             "alternativas": {"A": "2", "B": "3", "C": "4", "D": "5", "E": "6"},
             "resolucao_passo_a_passo": "Todo triângulo tem 3 lados.",
             "resposta_correta": "B", "difficulty": "EASY"}
        self.assertTrue(dv.dados_insuficientes_classificacao(q, self.slot))  # bruto acusaria
        tipos = [v["tipo"] for v in gl.violacoes_slot([], q, self.slot, "9º", "H17", 1, TAX)]
        self.assertNotIn("dados_insuficientes", tipos)

    def test_regenera_com_restricao_e_entrega_a_com_dados(self):
        """G-9H17-0023 passa no best-of-N (geometria nao_aplicavel) e só a
        guarda do plano a pega: a 2ª tentativa recebe a restrição com o eixo."""
        self.assertEqual(vg.verificar_geometria(SEM_DADOS)[0], "nao_aplicavel")
        gen = _gen_fixo([SEM_DADOS, CORRETA])
        r = gl.gerar_lote_planejado("cli", "m.gguf", "9º", "H17", "desc", "Fácil", 1, 4,
                                    base_seed=_seed_com_subtema("triangulo"), gen_fn=gen,
                                    taxonomia=TAX, retries=0, max_tentativas_diversidade=2)
        self.assertEqual(gen.n, 2)
        self.assertNotIn("Restrição", gen.prompts[0])
        self.assertIn("Restrição:", gen.prompts[1])
        self.assertIn("não dá nenhuma medida", gen.prompts[1])
        self.assertIn("Eixo:", gen.prompts[1])
        self.assertEqual(r["questoes"][0]["enunciado"], CORRETA["enunciado"])
        self.assertEqual(r["violacoes"][0]["violacoes"], [])

    def test_chave_prefere_quem_tem_dados(self):
        base = {"obj": {}, "status": "nao_verificavel"}
        com = dict(base, violacoes=[{"tipo": "dados_insuficientes"}])
        sem = dict(base, violacoes=[{"tipo": "contexto_repetido"}])
        self.assertGreater(gl._chave(sem), gl._chave(com))


# ===========================================================================
# Permutação re-verificada
# ===========================================================================
class TestPermutacaoGeometria(unittest.TestCase):
    def test_barracas_permuta_sem_quebrar_alternativas_compostas(self):
        self.assertEqual(gl.vetos_permutacao(BARRACAS), [])
        gab_texto = BARRACAS["alternativas"][BARRACAS["resposta_correta"]]
        for alvo in "ABCE":
            with self.subTest(alvo=alvo):
                p = gl.permutar_alternativas(BARRACAS, 7, alvo=alvo)
                self.assertEqual(p["resposta_correta"], alvo)
                self.assertEqual(p["alternativas"][alvo], gab_texto)
                # os textos viajam INTEIROS ("a barraca A é ..." não é reescrito)
                self.assertEqual(sorted(p["alternativas"].values()),
                                 sorted(BARRACAS["alternativas"].values()))
                self.assertEqual(vg.verificar_geometria(p)[0], "ok")

    def test_veredito_invariante_nas_vinte_auditadas(self):
        for id_, q in AUDITADAS.items():
            v0 = vg.verificar_geometria(q)[0]
            for seed in range(3):
                for alvo in "ABCDE":
                    with self.subTest(id_=id_, seed=seed, alvo=alvo):
                        p = gl.permutar_alternativas(q, seed, alvo=alvo)
                        self.assertEqual(vg.verificar_geometria(p)[0], v0)

    def test_veredito_que_muda_desfaz_a_permutacao(self):
        """Mecanismo: se a leitura do verificador mudar com a troca de posição,
        a questão sai intacta (o verificador real é invariante; aqui ele é
        forçado a não ser)."""
        real = vg.verificar_geometria

        def instavel(q):
            v, d = real(q)
            if q.get("resposta_correta") != BARRACAS["resposta_correta"]:
                return "nao_unica", dict(d, motivo="forcado")
            return v, d

        with mock.patch.object(gl.vg, "verificar_geometria", side_effect=instavel):
            self.assertIs(gl.permutar_alternativas(BARRACAS, 7, alvo="A"), BARRACAS)
            lote = gl.permutar_lote([BARRACAS], 7)
        self.assertEqual(lote[0]["resposta_correta"], BARRACAS["resposta_correta"])

    def test_sugestao_acompanha_o_texto(self):
        """R1-Q3 (gabarito_errado, sugere C='Obtusângulo'): depois de permutar,
        a sugestão aponta para o MESMO texto."""
        p = gl.permutar_alternativas(CONTA_ERRADA, 3, alvo="A")
        if p is CONTA_ERRADA:
            self.skipTest("vetada por outro motivo")
        sug = vg.verificar_geometria(p)[1]["sugestao"]
        self.assertEqual(p["alternativas"][sug], "Obtusângulo")


# ===========================================================================
# Relatórios (só medição)
# ===========================================================================
PROMPT = {"id": "P01-9º-H17-N3", "ano": "9º", "habilidade": "H17", "dificuldade": "Fácil",
          "quantidade": 3, "seed": 1, "descricao": "Classificar"}


class TestRelatorioDiversidade(unittest.TestCase):
    def test_metricas_existentes_nao_mudam(self):
        qs = [PREMISSA, CORRETA, AUDITADAS["R2-Q3"]]
        m = av.metricas_lote(qs, PROMPT, 3, 3, 1.0, TAX)
        self.assertEqual(m["geometria"]["ok"], 1)
        self.assertEqual(m["geometria"]["premissa_impossivel"], 1)
        self.assertEqual(m["geometria"]["nao_unica"], 1)
        sem = {k: v for k, v in m.items() if k != "geometria"}
        ag_com = av.agregar([m])
        ag_sem = av.agregar([dict(sem, geometria={v: 0 for v in vg.VEREDITOS})])
        novas = {"geometria_ok", "geometria_reprovada", "geometria_reprovada_total",
                 "geometria_nao_aplicavel"}
        self.assertTrue(novas <= set(ag_com))
        self.assertEqual({k: v for k, v in ag_com.items() if k not in novas},
                         {k: v for k, v in ag_sem.items() if k not in novas})
        self.assertEqual(ag_com["geometria_ok"], 1)
        self.assertEqual(ag_com["geometria_reprovada"],
                         {"dados_insuficientes": 0, "gabarito_errado": 0, "nao_unica": 1,
                          "premissa_impossivel": 1})
        self.assertEqual(ag_com["geometria_reprovada_total"], 2)

    def test_relatorio_antigo_sem_o_campo_e_recalculado(self):
        m = av.metricas_lote([PREMISSA, CORRETA], PROMPT, 2, 2, 1.0, TAX)
        antigo = {k: v for k, v in m.items() if k != "geometria"}
        self.assertEqual(av.agregar([antigo])["geometria_ok"], 1)
        self.assertEqual(av.agregar([antigo])["geometria_reprovada_total"], 1)

    def test_markdown_mostra_a_linha(self):
        m = av.metricas_lote([PREMISSA, CORRETA], PROMPT, 2, 2, 1.0, TAX)
        rel = {"rotulo": "t", "num_prompts": 1, "modo_atual": "unico", "dry_run": True,
               "modos": {"ajustado": {"lotes": [m], "agregado": av.agregar([m])}}}
        md = av.tabela_markdown(rel)
        self.assertIn("Verificador de geometria", md)
        self.assertIn("ok=1", md)


class TestBatch(unittest.TestCase):
    def test_secao_geometria_no_relatorio(self):
        respostas = [PREMISSA, CORRETA]
        estado = {"n": 0}

        def falso_generate(llama_cli, gguf, prompt, threads, mnt, seed=None, grammar=None):
            q = respostas[min(estado["n"], len(respostas) - 1)]
            estado["n"] += 1
            return json.dumps({"questoes": [q]}, ensure_ascii=False), None, 30.0, 0.01

        ex = {"messages": [{"role": "system", "content": "s"},
                           {"role": "user", "content": "Gere 1 questão 9º H17"},
                           {"role": "assistant", "content": "{}"}],
              "meta": {"codigo_item": "X", "ano": "9º", "dificuldade": "Fácil"}}
        with tempfile.TemporaryDirectory() as d:
            val = Path(d) / "val.jsonl"
            val.write_text(json.dumps(ex, ensure_ascii=False) + "\n", encoding="utf-8")
            gguf = Path(d) / "m.gguf"
            gguf.write_bytes(b"x")
            out = Path(d) / "rel.json"
            with mock.patch.object(tm, "generate", side_effect=falso_generate), \
                    mock.patch("builtins.print"):
                tm.batch("cli", gguf, 4, None, val_path=val, report_path=out,
                         modo_geometria="ativo")
            rel = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(rel["pos_processamento"]["ok"], 1)
        self.assertEqual(rel["pos_processamento"]["falha"], 0)
        self.assertEqual(rel["pos_processamento"]["regeneracoes_total"], 1)
        self.assertEqual(rel["geometria"]["modo"], "ativo")
        self.assertEqual(rel["geometria"]["ok"], 1)
        self.assertEqual(rel["geometria"]["reprovacoes_nas_amostras"], 1)
        self.assertEqual(rel["detalhes"][0]["geometria"], "ok")
        # G3 continua aritmético: classificação não tem conta
        self.assertEqual(rel["estrutura"]["consistencia_verificavel_n"], 0)


# ===========================================================================
# Correções da revisão adversarial de 2026-10-01
# ===========================================================================
class TestCorrecoesRevisao(unittest.TestCase):
    # --- uma constante só; padrão "sombra" até medir o custo com o modelo real
    def test_modo_padrao_unico_e_sombra(self):
        """Havia vg.MODO_GEOMETRIA='sombra' (que nada lia) e
        tm.MODO_GEOMETRIA='ativo' (a efetiva). O custo medido do ativo em 9º
        H17 com o modelo atual (75% de rejeição) é ~3,3 chamadas/questão e
        17-45% dos slots esgotam o orçamento: liga-se só depois de medir."""
        self.assertEqual(vg.MODO_GEOMETRIA, "sombra")
        self.assertIs(tm.MODO_GEOMETRIA, vg.MODO_GEOMETRIA)
        self.assertIs(tm.MODOS_GEOMETRIA, vg.MODOS_GEOMETRIA)
        gen = _gen_fixo([PREMISSA, CORRETA])
        r = tm.generate_validated("cli", "m.gguf", "p", 4, 512, gen_fn=gen, retries=1)
        self.assertEqual((r["geometria"]["modo"], gen.n), ("sombra", 1))

    def test_gerador_real_repassa_o_modo(self):
        g = av.GeradorReal("cli", "m.gguf", 4, None, 0, modo_geometria="ativo")
        with mock.patch.object(tm, "generate_validated",
                               return_value={"text": "{}", "obj": None, "elapsed": 0.1}) as m:
            g.chamada({"ano": "9º", "habilidade": "H17", "descricao": "d",
                       "dificuldade": "Fácil"}, 1, 0)
        self.assertEqual(m.call_args.kwargs["modo_geometria"], "ativo")
        with mock.patch.object(gl, "gerar_lote_planejado",
                               return_value={"questoes": [], "obj": None}) as m:
            g.ajustado({"ano": "9º", "habilidade": "H17", "descricao": "d",
                        "dificuldade": "Fácil", "quantidade": 1}, 0, 0)
        self.assertEqual(m.call_args.kwargs["modo_geometria"], "ativo")

    def _relatorio(self, d, rotulo, *extra):
        av.main(["--rotulo", rotulo, "--dry-run", "--num-prompts", "3", "--modos", "ajustado",
                 "--out-dir", d, *extra])
        return json.loads((Path(d) / f"diversidade_{rotulo}.json").read_text(encoding="utf-8"))

    def test_relatorio_de_diversidade_registra_a_regua(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("builtins.print"):
            rel = self._relatorio(d, "a")
            rel_ativo = self._relatorio(d, "b", "--geometria", "ativo")
        self.assertEqual(rel["modo_geometria"], "sombra")
        self.assertEqual(rel_ativo["modo_geometria"], "ativo")
        self.assertEqual(rel["regua_sha256"], dv.versao_regua())
        self.assertRegex(rel["regua_sha256"], r"^[0-9a-f]{64}$")

    def test_g11_recusa_regua_diferente(self):
        """A régua efetiva de 9º H17 mudou com a taxonomia (n_tipos 2 -> 3):
        baseline antigo x candidato novo reprovava G11 sem o modelo mudar."""
        import promover_checkpoint as pc
        with tempfile.TemporaryDirectory() as d, mock.patch("builtins.print"):
            base = self._relatorio(d, "a")
            ativo = self._relatorio(d, "b", "--geometria", "ativo")
        self.assertTrue(pc.gate_diversidade(base, json.loads(json.dumps(base))).passou)
        cand = json.loads(json.dumps(base))
        cand["regua_sha256"] = "0" * 64
        g = pc.gate_diversidade(base, cand)
        self.assertFalse(g.passou)
        self.assertIn("régua", g.detalhe)
        antigo = {k: v for k, v in base.items() if k not in ("regua_sha256", "modo_geometria")}
        self.assertFalse(pc.gate_diversidade(antigo, base).passou)
        g = pc.gate_diversidade(base, ativo)
        self.assertFalse(g.passou)
        self.assertIn("modo_geometria", g.detalhe)
        # relatórios antigos dos DOIS lados (testes e baselines já gravados): sem checagem
        self.assertTrue(pc.gate_diversidade(antigo, json.loads(json.dumps(antigo))).passou)

    def test_houve_ganho_nao_conta_regeneracoes_entre_modos_diferentes(self):
        import promover_checkpoint as pc
        base = {"pos_processamento": {"regeneracoes_total": 5}, "geometria": {"modo": "ativo"}}
        cand = {"pos_processamento": {"regeneracoes_total": 1}, "geometria": {"modo": "sombra"}}
        self.assertEqual(pc.houve_ganho(base, cand), [])
        antigo = {"pos_processamento": {"regeneracoes_total": 5}}  # anterior à guarda = sombra
        cand_ativo = {"pos_processamento": {"regeneracoes_total": 1}, "geometria": {"modo": "ativo"}}
        self.assertEqual(pc.houve_ganho(antigo, cand_ativo), [])
        self.assertEqual(pc.houve_ganho(antigo, cand), ["regenerações: 5 -> 1"])
        cand_ativo_base = dict(base, pos_processamento={"regeneracoes_total": 1})
        self.assertEqual(pc.houve_ganho(base, cand_ativo_base), ["regenerações: 5 -> 1"])

    # --- convenção única de classes entre o plano e o verificador
    def test_instrucao_do_plano_segue_a_convencao_do_verificador(self):
        """O verificador aprova o gabarito MAIS ESPECÍFICO com a superclasse
        como distrator (R2-Q5, aprovada pela auditoria; MT9050/MT9081/MT9084
        do banco). O plano dizia o contrário ("sem outra alternativa que também
        sirva"): duas regras incompatíveis para o porte TypeScript."""
        tipos = {t["id"]: t for t in dv.TIPOS_RACIOCINIO}
        for tid in ("classificacao_lados", "classificacao_propriedades"):
            ins = tipos[tid]["instrucao"]
            with self.subTest(tid):
                self.assertIn("mais específic", ins)
                self.assertNotRegex(ins, r"sem outra alternativa|só uma alternativa")
        nda = "Nenhuma das alternativas anteriores"
        q_quad = {"enunciado": "Um quadrilátero tem quatro lados com a mesma medida e os quatro "
                               "ângulos retos. Qual é o nome mais específico dele?",
                  "alternativas": dict(zip("ABCDE", ["Quadrado", "Retângulo", "Losango",
                                                     "Trapézio", "Paralelogramo"])),
                  "resolucao_passo_a_passo": "x", "resposta_correta": "A", "difficulty": "EASY"}
        q_tri = {"enunciado": "Um triângulo tem lados de 5 cm, 5 cm e 5 cm. Quanto aos lados, "
                              "ele é:",
                 "alternativas": dict(zip("ABCDE", ["Equilátero", "Isósceles", "Escaleno",
                                                    "Retângulo", nda])),
                 "resolucao_passo_a_passo": "x", "resposta_correta": "A", "difficulty": "EASY"}
        for qq in (q_quad, q_tri):
            self.assertEqual(vg.verificar_geometria(qq)[0], "ok")
        self.assertIn(vg.CONVENCAO_GABARITO, dv.__doc__ + "".join(
            t.get("instrucao", "") for t in dv.TIPOS_RACIOCINIO) + _comentarios(dv))

    # --- (?i:...) trocado por classes explícitas, com o MESMO comportamento
    def test_regex_de_letra_sem_flag_local_equivalente(self):
        import re as _re
        antigo_subst = r"(?i:alternativas?|letras?|op[çc](?:[ãa]o|[õo]es)|itens|item)"
        antigos = {
            "_REF_CRUZADA": _re.compile(
                r"\b[A-E]\s*(?:e|,|ou)\s*[A-E]\b|" + antigo_subst + r"\s+[A-E]\b"
                r"|\b(?:apenas|somente)\s+[IVX]+\b|\b[IVX]+\s*(?:e|,)\s*[IVX]+\b"),
            "_CITA_LETRA": _re.compile(antigo_subst + r"\s+[A-E]\b|[A-E]\s*\)\s*\S"),
            "_RESOL_CITA_LETRA": _re.compile(
                antigo_subst + r"\s+[A-E]\b"
                r"|(?i:respostas?|afirma[çc](?:[ãa]o|[õo]es)|assertivas?)\s+"
                r"(?:correta\s+)?(?:[ée]\s+)?(?:a\s+)?[A-E]\b"
                r"|(?:" + antigo_subst + r"|(?i:respostas?|afirma[çc](?:[ãa]o|[õo]es)"
                r"|assertivas?))[^.\n]{0,24}?\b[A-E]\b"
                r"|[A-E]\s*\)\s*\S|\"[A-E]\"|'[A-E]'|[A-E]\s*[:\-]\s"
                r"|(?:primeir|segund|terceir|quart|quint|[úu]ltim)\w*\s+"
                r"(?:alternativa|op[çc][ãa]o|item|resposta)"),
        }
        textos = ["Alternativa D", "alternativa d", "ALTERNATIVAS B e C", "Letra B", "LETRA b",
                  "opção C", "OPÇÃO C", "Opções A", "OPÇÕES E", "Itens B", "ITEM A", "a resposta "
                  "correta é C", "RESPOSTA CORRETA É A B", "Afirmação correta é a D",
                  "AFIRMAÇÕES A", "Assertiva E", "a alternativa correta é B", "e o a", "Opcao B",
                  "a afirmação correta é a A", "Resposta: 12", "A e C", "apenas II"]
        for nome, rx in antigos.items():
            novo = getattr(gl, nome)
            for t in textos:
                with self.subTest(nome, t=t):
                    self.assertEqual(bool(novo.search(t)), bool(rx.search(t)))
                    self.assertEqual([m.span() for m in novo.finditer(t)],
                                     [m.span() for m in rx.finditer(t)])


def _comentarios(mod):
    import inspect
    return "\n".join(l for l in inspect.getsource(mod).splitlines() if l.lstrip().startswith("#"))


if __name__ == "__main__":
    unittest.main()
