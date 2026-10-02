"""Testes das decisões de 2026-10-01 (passo 1 da base v3): H2 (duas casas
decimais), H3 (decisões humanas versionáveis), H4 (árbitro Gemini antes da
remoção pela D2), P1 (pré-filtro D5 de geometria), P2 (taxonomia de
multiplicação x divisão), P3 (revisor sem âncora de dificuldade e sem subtema
que vaza a resposta) e P4 (resposta truncada fora do cache).

Tudo com cliente SIMULADO/falso: nenhuma chamada de rede. O cliente Gemini é
testado com um `http` falso que grava as requisições. Arquivos só em diretório
temporário; os protegidos (train/val) são conferidos por sha1 no
test_agentes_questoes.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import agentes_questoes as aq  # noqa: E402
import arbitro_gemini as ag  # noqa: E402
import auditar_base as ab  # noqa: E402
import build_taxonomia as bt  # noqa: E402
import diversidade as dv  # noqa: E402
import distill_teacher as dt  # noqa: E402
import injetar_questoes as iq  # noqa: E402
import test_agentes_questoes as T  # noqa: E402  (só helpers; nenhuma classe de teste é reimportada)
from schema_utils import check_consistency  # noqa: E402

LETRAS = "ABCDE"


def q_fracao(enunciado, alts, gab="A"):
    return {"enunciado": enunciado, "alternativas": alts, "resolucao_passo_a_passo": "conta.",
            "resposta_correta": gab, "difficulty": "MEDIUM"}


EN_1348 = ("Em uma biblioteca escolar, 5 de cada 11 livros emprestados no mês são de ficção científica. "
           "Qual é a representação decimal correspondente a essa fração?")
ALTS_1348 = {"A": "0,45", "B": "0,55", "C": "0,511", "D": "0,227", "E": "0,50"}


# ===========================================================================
class TestH2Arredondamento(unittest.TestCase):
    def test_duas_casas_vale_para_dizima(self):
        r = aq.verificar_aritmetica(q_fracao(EN_1348, ALTS_1348, "A"))
        self.assertEqual((r["status"], r["verdadeiras"]), ("confirma", ["A"]))
        self.assertIn("duas casas: 0,45", r["esperado"])

    def test_truncado_e_arredondado_valem_e_os_dois_juntos_tornam_nao_unica(self):
        en = "Qual é a representação decimal da fração 2/3?"
        base = {"A": "0,67", "B": "0,5", "C": "1,5", "D": "0,23", "E": "0,32"}
        self.assertEqual(aq.verificar_aritmetica(q_fracao(en, base, "A"))["status"], "confirma")      # arredondado
        self.assertEqual(aq.verificar_aritmetica(q_fracao(en, dict(base, A="0,66"), "A"))["status"],
                         "confirma")                                                                   # truncado
        r = aq.verificar_aritmetica(q_fracao(en, dict(base, B="0,66"), "A"))                          # os dois
        self.assertEqual((r["status"], r["verdadeiras"]), ("contradiz", ["A", "B"]))
        # a dízima com reticências e as duas casas juntas: também não única
        self.assertEqual(aq.verificar_aritmetica(q_fracao(EN_1348, dict(ALTS_1348, B="0,4545..."), "A"))["status"],
                         "contradiz")

    def test_outra_precisao_ou_decimal_exato_o_verificador_se_abstem(self):
        # "0,5" é 5/11 com UMA casa: a regra do usuário não cobre -> sem veredito
        self.assertEqual(aq.verificar_aritmetica(q_fracao(EN_1348, dict(ALTS_1348, A="0,5", E="0,62"), "A"))["status"],
                         "sem_veredito")
        # 3/8 = 0,375 é decimal EXATO: "0,38" sem pedido de aproximação não é decidido
        en = "Qual é o número decimal equivalente à fração 3/8?"
        alts = {"A": "0,38", "B": "0,75", "C": "0,5", "D": "0,25", "E": "0,125"}
        self.assertEqual(aq.verificar_aritmetica(q_fracao(en, alts))["status"], "sem_veredito")
        # precisão pedida diferente de duas casas: abstém-se
        en1 = "Qual é o número decimal, com uma casa decimal, aproximadamente igual a 5/11?"
        self.assertEqual(aq.verificar_aritmetica(q_fracao(en1, {"A": "0,4", "B": "0,7", "C": "0,9", "D": "0,2",
                                                                "E": "0,1"}))["status"], "sem_veredito")
        # pedido explícito de duas casas: vale mesmo com o número 2 no texto
        en2 = "Escreva 15/33 como número decimal com 2 casas decimais."
        self.assertEqual(aq.verificar_aritmetica(q_fracao(en2, {"A": "0,45", "B": "0,55", "C": "0,33", "D": "0,15",
                                                                "E": "2,2"}))["status"], "confirma")

    def test_check_consistency_nao_acusa_arredondamento(self):
        # schema_utils fica como está (contrato congelado): a conta aproximada
        # não é "a op b = r" exata, então o item é não verificável, nunca acusado
        q = dict(q_fracao(EN_1348, ALTS_1348), resolucao_passo_a_passo=(
            "Divida 5 por 11. O resultado é 0,454545..., escrito como 0,45."))
        self.assertEqual(check_consistency(q), (None, None))
        q2 = dict(q_fracao("Dividiu 10 litros entre 3 garrafas. Quanto, aproximadamente, em cada uma?",
                           {"A": "3,33", "B": "3,5", "C": "30", "D": "7", "E": "13"}),
                  resolucao_passo_a_passo="10 ÷ 3 = 3,33. Aproximadamente 3,33 litros.")
        self.assertEqual(check_consistency(q2), (None, None))

    def test_prompts_trazem_a_regra(self):
        self.assertIn("12. Arredondamento", aq.GERADOR_ADDENDUM)
        self.assertIn("aproximadamente", aq.GERADOR_ADDENDUM)
        for p in (aq.VALIDADOR_SISTEMA, aq.REVISOR_SISTEMA):
            self.assertIn("DUAS casas decimais", p)
            self.assertIn("0,45", p)

    def test_refiltrar_absolve_o_1348(self):
        # registro gravado com o filtro ANTIGO (f3): verificador "contradiz" e nível baixa
        ex = T.exemplo("9º", "H09", q_fracao(EN_1348, ALTS_1348, "A"), "DIST-H09-Difícil-00569",
                       dificuldade="Difícil", destilado=True, professor="p")
        reg = T._registro(ex, 1348, T._juiz(True), T._juiz(True), filtros=["verificador_aritmetico"])
        reg["versao_filtros"] = "f3-aritmetica"
        reg["filtros"]["aritmetica"] = {"status": "contradiz", "detalhe": "5/11 != 0,45"}
        self.assertTrue(ab.erro_matematico_confirmado(reg)[0])
        novo = ab.refiltrar(reg, ex)
        self.assertEqual(novo["confianca"], "alta")
        self.assertEqual(novo["filtros"]["reprovado"], [])
        self.assertNotIn("filtro:verificador_aritmetico", novo["problemas"])
        self.assertEqual(novo["refiltrado"]["filtros_absolvidos"], ["verificador_aritmetico"])
        self.assertFalse(ab.erro_matematico_confirmado(novo)[0])
        self.assertEqual(ab.decidir("destilado", novo), ("manter", "alta"))


# ===========================================================================
def _decisoes(path, decisoes):
    Path(path).write_text(json.dumps({"versao": 1, "decisoes": decisoes}, ensure_ascii=False), encoding="utf-8")


class TestH3DecisoesHumanas(unittest.TestCase):
    def setUp(self):
        self.amb = T.Ambiente()
        d = self.amb.d
        self.vazia = dict(T.questao_boa(12, 3, "B", dif="MEDIUM", texto="Uma caixa tem 12 lápis e Ana pôs mais 3; "
                                                                        "ela fez 12 + 3. Quantos lápis há?"),
                          resolucao_passo_a_passo="")
        self.ruim = dict(T.questao_boa(30, 9, "D", texto="Um ônibus levava 30 pessoas e subiram 9; o motorista "
                                                          "fez 30 + 9. Quantas pessoas há?"), resposta_correta="E")
        self.exs = [T.exemplo("9º", "H04", self.vazia, "MT9018MH04MT", dificuldade="Moderado"),
                    T.exemplo("9º", "H99", T.questao_boa(), "MT9999"),
                    T.exemplo("9º", "H99", self.ruim, "DIST-H99-Fácil-00002", destilado=True, professor="p")]
        self.amb.train.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in self.exs),
                                  encoding="utf-8")
        # auditoria: o 2 tem erro matemático confirmado pela D2 e o árbitro concorda
        regs = [T._registro(self.exs[2], 2, T._juiz(False, ["gabarito_errado"]),
                            T._juiz(False, ["gabarito_errado"]))]
        (d / "auditoria.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs),
                                           encoding="utf-8")
        self.res = "Somando 12 + 3 obtemos 15 lápis na caixa."

    def tearDown(self):
        self.amb.fechar()

    def dec(self, **kw):
        base = {"data": "2026-10-01", "autor": "usuário", "motivo": "teste"}
        return dict(base, **kw)

    def montar(self):
        return T.silencioso(ab.montar, self.amb.args_auditoria("--montar"))

    def test_manter_editar_resolucao_e_dificuldade_nos_tres_lugares(self):
        _decisoes(self.amb.d / "decisoes_humanas.json", [self.dec(
            codigo_item="MT9018MH04MT", idx=0, hash_questao=ab.hash_exemplo(self.exs[0]), acao="manter_editar",
            edicoes={"resolucao_passo_a_passo": self.res, "dificuldade": "Fácil"},
            pre_condicoes={"resolucao_passo_a_passo": "", "dificuldade": "Moderado"})])
        rel = self.montar()
        v3 = {e["meta"]["codigo_item"]: e for e in self.amb.linhas("v3.jsonl")}
        e = v3["MT9018MH04MT"]
        q = json.loads(e["messages"][2]["content"])["questoes"][0]
        self.assertEqual(q["resolucao_passo_a_passo"], self.res)
        self.assertEqual(q["difficulty"], "EASY")
        self.assertEqual(e["meta"]["dificuldade"], "Fácil")
        self.assertIn("Dificuldade: Fácil.", e["messages"][1]["content"])
        self.assertNotIn("Moderado", e["messages"][1]["content"])
        # nada além disso mudou na questão
        self.assertEqual({k: v for k, v in q.items() if k not in ("resolucao_passo_a_passo", "difficulty")},
                         {k: v for k, v in self.vazia.items() if k not in ("resolucao_passo_a_passo", "difficulty")})
        self.assertEqual(e["meta"]["rotulo_corrigido"]["fonte"], "decisao_humana")
        self.assertEqual(e["meta"]["decisao_humana"]["autor"], "usuário")
        # deixou de estar na lista de resolução vazia e entrou nos rótulos corrigidos
        self.assertNotIn("MT9018MH04MT", [r["codigo_item"] for r in self.amb.linhas("resolucao_vazia_v3.jsonl")])
        self.assertEqual([(r["codigo_item"], r["fonte"], r["de"], r["para"])
                          for r in self.amb.linhas("rotulos_corrigidos_v3.jsonl")],
                         [("MT9018MH04MT", "decisao_humana", "Moderado", "Fácil")])
        self.assertEqual(rel["n_decisoes_humanas_aplicadas"], 1)
        self.assertEqual(aq.defeitos_resolucao(q), [])

    def test_remover_e_manter_tem_precedencia_sobre_as_regras_automaticas(self):
        # o árbitro CONFIRMA o erro do item 2 — mesmo assim a decisão "manter" vale
        arb = {"idx": 2, "hash_questao": ab.hash_exemplo(self.exs[2]), "modelo": "sim",
               "resultados": [{"avaliado": True, "resposta_calculada": "D",
                               "problemas": [{"codigo": "gabarito_errado", "detalhe": "x"}]}]}
        (self.amb.d / "arbitragem_d2.jsonl").write_text(json.dumps(arb) + "\n", encoding="utf-8")
        _decisoes(self.amb.d / "decisoes_humanas.json", [
            self.dec(codigo_item="DIST-H99-Fácil-00002", idx=2, acao="manter"),
            self.dec(codigo_item="MT9999", idx=1, hash_questao=ab.hash_exemplo(self.exs[1]), acao="remover",
                     motivo="habilidade desalinhada")])
        rel = self.montar()
        cods = [e["meta"]["codigo_item"] for e in self.amb.linhas("v3.jsonl")]
        self.assertIn("DIST-H99-Fácil-00002", cods)
        self.assertNotIn("MT9999", cods)
        rem = self.amb.linhas("removidos_v3.jsonl")
        self.assertEqual([(r["codigo_item"], r["motivo"], r["evidencia"]["autor"], r["evidencia"]["motivo"])
                          for r in rem], [("MT9999", "decisao_humana", "usuário", "habilidade desalinhada")])
        self.assertEqual(rel["n_removidos_decisao_humana"], 1)
        # decidido pelo humano não volta para a lista de revisão humana
        self.assertEqual(self.amb.linhas("humana.jsonl"), [])
        # o mantido fica EXATAMENTE como estava (sem D2/D3)
        self.assertEqual(next(e for e in self.amb.linhas("v3.jsonl")
                              if e["meta"]["codigo_item"] == "DIST-H99-Fácil-00002"), self.exs[2])

    def test_conteudo_mudou_ou_pre_condicao_falha_nao_aplica(self):
        _decisoes(self.amb.d / "decisoes_humanas.json", [
            self.dec(codigo_item="MT9999", idx=1, hash_questao="0" * 40, acao="remover"),
            self.dec(codigo_item="MT9018MH04MT", idx=0, acao="manter_editar",
                     edicoes={"resolucao_passo_a_passo": self.res},
                     pre_condicoes={"resolucao_passo_a_passo": "outra coisa"})])
        rel = self.montar()
        self.assertIn("MT9999", [e["meta"]["codigo_item"] for e in self.amb.linhas("v3.jsonl")])
        motivos = {d["codigo_item"]: d["motivo"] for d in rel["decisoes_nao_aplicadas"]}
        self.assertTrue(motivos["MT9999"].startswith("conteudo_mudou"))
        self.assertEqual(motivos["MT9018MH04MT"], "pre_condicao_resolucao_passo_a_passo")
        # sem a edição, o MT9018 segue como estava (resolução vazia listada)
        self.assertIn("MT9018MH04MT", [r["codigo_item"] for r in self.amb.linhas("resolucao_vazia_v3.jsonl")])

    def test_arquivo_malformado_levanta(self):
        p = self.amb.d / "dec.json"
        for ruim in ([{"codigo_item": "X", "acao": "remover", "data": "d", "motivo": "m"}],          # sem autor
                     [self.dec(codigo_item="X", acao="apagar")],                                     # ação inválida
                     [self.dec(codigo_item="X", acao="remover"), self.dec(codigo_item="X", acao="manter")],
                     [self.dec(codigo_item="X", acao="manter_editar", edicoes={"enunciado": "y"})]):
            _decisoes(p, ruim)
            with self.subTest(ruim=ruim), self.assertRaises(ValueError):
                ab.carregar_decisoes(p)
        self.assertEqual(ab.carregar_decisoes(self.amb.d / "nao_existe.json"), {})

    def test_arquivo_real_versionado(self):
        """Doc/decisoes_humanas.json: as decisões do usuário de 2026-10-01."""
        decs = ab.carregar_decisoes(ab.DECISOES_PADRAO)
        acoes = {c: d["acao"] for c, d in decs.items()}
        self.assertEqual(acoes.pop("MT9018MH04MT"), "manter_editar")
        self.assertEqual(acoes.pop("DIST-H09-Difícil-00569"), "manter")  # 1348, H2
        # 2ª rodada de decisões (2026-10-01): 15 restaurações D6 (manter_editar
        # de alternativas) + 2 D6 ambíguos + 6 remoções da revisão humana.
        restaurar = {c for c, a in acoes.items() if a == "manter_editar"}
        self.assertEqual(len(restaurar), 15)
        self.assertTrue(all(set(decs[c]["edicoes"]) == {"alternativas"} for c in restaurar))
        remover = {c for c, a in acoes.items() if a == "remover"}
        self.assertTrue({"MT9094DH19MT", "MT9061FH13MT", "MT2032MH07MT", "MT9013FH05TD", "MT9049DH10MT",
                         "SINT-H08-Moderado-0032", "DIST-H18-Moderado-00102", "DIST-H03-Difícil-00273",
                         "DIST-H17-Difícil-00348", "DIST-H01-Difícil-00408", "DIST-H25-Moderado-00650",
                         "MT9020MfH07TD", "MT90125DH25MT"} <= remover)
        self.assertEqual(len(remover), 11 + 2 + 6)
        self.assertEqual(set(acoes.values()), {"remover", "manter_editar"})
        self.assertTrue(all(d["autor"] == "usuário" and d["data"] == "2026-10-01" and d["motivo"]
                            for d in decs.values()))
        d = decs["MT9018MH04MT"]
        self.assertEqual(d["edicoes"]["dificuldade"], "Fácil")
        self.assertGreater(len(d["edicoes"]["resolucao_passo_a_passo"]), 40)
        # contra a base real (só leitura): a decisão casa com o item e a edição se aplica
        base = ROOT / "data" / "train_curado.jsonl"
        if not base.exists():
            self.skipTest("data/train_curado.jsonl ausente")
        ex = json.loads(base.read_text(encoding="utf-8").splitlines()[264])
        self.assertEqual(ab.decisao_aplicavel(d, 264, ex), (True, None))
        novo, falha = ab.aplicar_edicoes_humanas(ex, d)
        self.assertIsNone(falha)
        q = json.loads(novo["messages"][2]["content"])["questoes"][0]
        self.assertEqual((q["difficulty"], q["resposta_correta"]), ("EASY", "B"))
        self.assertIn("2,45 < 2,5 < 2,75", q["resolucao_passo_a_passo"])


# ===========================================================================
class HttpFalso:
    """requests falso: grava as requisições e devolve respostas fixas."""

    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.reqs = []

    def _resp(self):
        st, corpo = self.respostas.pop(0)
        return SimpleNamespace(status_code=st, json=lambda: corpo)

    def post(self, url, headers=None, json=None, timeout=None):
        self.reqs.append(("POST", url, headers, json))
        return self._resp()

    def get(self, url, headers=None, params=None, timeout=None):
        self.reqs.append(("GET", url, headers, params))
        return self._resp()


def resp_gemini(texto, fim="STOP", pensamento="rascunho"):
    return {"candidates": [{"content": {"parts": [{"text": pensamento, "thought": True}, {"text": texto}]},
                            "finishReason": fim}],
            "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 40, "thoughtsTokenCount": 60}}


class TestH4ArbitroGemini(unittest.TestCase):
    CHAVE = "AIzaSy-CHAVE-FALSA-0123456789"

    def test_cliente_monta_a_requisicao_e_le_a_resposta_sem_expor_a_chave(self):
        http = HttpFalso([(200, resp_gemini('{"ok": 1}'))])
        cli = ag.ClienteGemini("gemini-x", api_key=self.CHAVE, http=http)
        r = cli.chat_completion([{"role": "system", "content": "SIS"}, {"role": "user", "content": "USR"}], 512, 0.0)
        metodo, url, cab, corpo = http.reqs[0]
        self.assertEqual(url, f"{ag.GEMINI_URL}/models/gemini-x:generateContent")
        self.assertNotIn(self.CHAVE, url)                       # chave só no cabeçalho
        self.assertEqual(cab["x-goog-api-key"], self.CHAVE)
        self.assertEqual(corpo["systemInstruction"]["parts"][0]["text"], "SIS")
        self.assertEqual(corpo["contents"], [{"role": "user", "parts": [{"text": "USR"}]}])
        self.assertEqual(corpo["generationConfig"], {"temperature": 0.0, "maxOutputTokens": 512})
        self.assertEqual(r.choices[0].message.content, '{"ok": 1}')  # sem a parte de pensamento
        self.assertEqual((r.usage.prompt_tokens, r.usage.completion_tokens), (100, 100))
        self.assertEqual(r.choices[0].finish_reason, "STOP")
        self.assertNotIn(self.CHAVE, repr(cli))

    def test_erro_http_vira_erro_com_status_e_sem_chave(self):
        http = HttpFalso([(404, {"error": {"message": "models/gemini-x is not found"}})])
        cli = ag.ClienteGemini("gemini-x", api_key=self.CHAVE, http=http)
        with self.assertRaises(ag.ErroGemini) as c:
            cli.chat_completion([{"role": "user", "content": "u"}], 10, 0.0)
        self.assertEqual(c.exception.response.status_code, 404)
        self.assertFalse(aq._retentavel(c.exception))           # 404 não se repete (custa)
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": self.CHAVE}):
            self.assertNotIn(self.CHAVE, aq._sanitizar(f"falha com chave {self.CHAVE}"))

    def test_escolha_do_modelo_nao_troca_em_silencio(self):
        def m(nome, thinking=True):
            return {"name": f"models/{nome}", "thinking": thinking, "supportedGenerationMethods": ["generateContent"]}
        lista = [m("gemini-2.5-flash"), m("gemini-2.5-flash-lite"), m("gemini-3.8-flash"), m("gemini-2.5-pro"),
                 m("gemini-flash-latest"), m("gemini-3-flash-preview"), m("gemini-2.5-flash-preview-tts", None)]
        self.assertEqual(ag.escolher_modelo(lista + [m(ag.MODELO_PEDIDO, None)]),
                         (ag.MODELO_PEDIDO, "pedido_disponivel"))
        nome, motivo = ag.escolher_modelo(lista)
        self.assertEqual(nome, "gemini-2.5-flash")               # flash + thinking, estável, versão mais próxima
        self.assertIn("pedido_indisponivel", motivo)
        self.assertEqual(ag.escolher_modelo([m("gemini-2.5-pro")]), (None, "nenhum_flash_com_thinking"))
        # a troca está registrada como configuração
        self.assertEqual(ag.MODELO_PEDIDO, "gemini-2.0-flash-thinking-exp-01-21")
        self.assertEqual(ag.MODELO_ARBITRO, "gemini-2.5-flash")
        self.assertIn("gemini-2.5-flash", ag.MOTIVO_TROCA_MODELO)

    def test_arbitro_resolve_as_cegas_com_orcamento_proprio(self):
        q = dict(T.questao_boa(gab="B"), resolucao_passo_a_passo="RESOLUCAO-MARCADOR 35 + 18 = 53.")
        perm = aq.permutacao(q, sal="arbitro")
        texto = T.fase1({**q, "alternativas": q["alternativas"]}, T.so_v("C"))  # resolve C (erro do gabarito B?)
        # fase1() aplica a permutação do VALIDADOR; o árbitro usa sal "arbitro":
        st = {L: T.so_v("C")[perm[L]] for L in LETRAS}
        obj = json.loads(texto)
        obj["alternativas"] = {L: st[L] for L in LETRAS}
        obj["resposta_calculada"] = aq.resposta_unica(st)
        http = HttpFalso([(200, resp_gemini(json.dumps(obj, ensure_ascii=False)))])
        arb = ag.Arbitro(ag.ClienteGemini("g", api_key="k", http=http), "g", aq.Orcamento(1), log_uso=None,
                         cache_path=None)
        r = arb.resolver_cego(q)
        enviado = json.dumps(http.reqs[0][3], ensure_ascii=False)
        self.assertNotIn("RESOLUCAO-MARCADOR", enviado)          # sem a resolução do autor
        self.assertNotIn("resposta_correta", enviado)
        self.assertTrue(r["avaliado"])
        self.assertEqual(r["resposta_calculada"], "C")           # letra ORIGINAL
        self.assertIn("gabarito_errado", [p["codigo"] for p in r["problemas"]])
        with self.assertRaises(aq.OrcamentoEsgotado):            # teto próprio, duro
            arb.resolver_cego(q)

    def test_truncada_nao_avalia_e_nao_entra_no_cache(self):
        q = T.questao_boa(gab="B")
        with tempfile.TemporaryDirectory() as t:
            cache = Path(t) / "arb.jsonl"
            bom = T.fase1(q, T.so_v("B"))
            http = HttpFalso([(200, resp_gemini(bom, fim="MAX_TOKENS")), (200, resp_gemini(bom))])
            arb = ag.Arbitro(ag.ClienteGemini("g", api_key="k", http=http), "g", aq.Orcamento(5), log_uso=None,
                             cache_path=cache)
            self.assertFalse(arb.resolver_cego(q)["avaliado"])
            self.assertFalse(cache.exists())
            arb.resolver_cego(q)                                  # refaz de verdade
            self.assertEqual(len(http.reqs), 2)
            self.assertTrue(cache.exists())
            arb2 = ag.Arbitro(ag.ClienteGemini("g", api_key="k", http=HttpFalso([])), "g", aq.Orcamento(0),
                              log_uso=None, cache_path=cache)
            arb2.resolver_cego(q)                                 # do cache, 0 chamadas
            self.assertEqual(arb2.hits, 1)

    def test_decisao_d2_com_arbitro(self):
        ex = T.exemplo("9º", "H99", T.questao_boa(), "DIST-H99-Fácil-00009", destilado=True, professor="p")
        reg = T._registro(ex, 9, T._juiz(False, ["gabarito_errado"]), T._juiz(False, ["gabarito_errado"]))
        self.assertEqual(ab.decidir("destilado", reg), ("remover", "erro_matematico_confirmado"))
        self.assertEqual(ab.decidir_com_arbitro("destilado", reg, None),
                         ("manter", "d2_aguardando_arbitro_revisao_humana"))

        def arb(*codigos, avaliado=True, resp="A"):
            return {"resultados": [{"avaliado": avaliado, "resposta_calculada": resp,
                                    "problemas": [{"codigo": c, "detalhe": ""} for c in codigos]}]}
        self.assertEqual(ab.decidir_com_arbitro("destilado", reg, arb("gabarito_errado")),
                         ("remover", "erro_matematico_confirmado_arbitro"))
        self.assertEqual(ab.decidir_com_arbitro("destilado", reg, arb()),
                         ("manter", "d2_arbitro_discorda_revisao_humana"))
        self.assertEqual(ab.decidir_com_arbitro("destilado", reg, arb("gabarito_errado", avaliado=False)),
                         ("manter", "d2_aguardando_arbitro_revisao_humana"))
        # incompleto não é errado (fica); mas com a resolução supondo o dado (1368), é.
        # Revisão do passo 2: o rótulo diz que o árbitro VIU o defeito (1080), não "discorda".
        self.assertEqual(ab.decidir_com_arbitro("destilado", reg, arb("dados_insuficientes",
                                                                      "gabarito_sem_resposta_unica")),
                         ("manter", "d2_arbitro_ve_item_incompleto_revisao_humana"))
        reg_supoe = T._registro(ex, 9, T._juiz(False, ["dados_insuficientes", "gabarito_sem_resposta_unica"]),
                                T._juiz(True), filtros=["resolucao_supoe_dado"])
        self.assertEqual(ab.decidir_com_arbitro("destilado", reg_supoe, arb("dados_insuficientes",
                                                                            "gabarito_sem_resposta_unica"))[1],
                         "erro_matematico_confirmado_arbitro")

    def test_arbitrar_so_os_candidatos_da_d2_e_retoma_sem_custo(self):
        amb = T.Ambiente()
        try:
            T.silencioso(ab.auditar, amb.args_auditoria(), agentes=T.agentes_com(T.Roteiro(), 100))
            cli = T.Roteiro()
            res = T.silencioso(ab.arbitrar, amb.args_auditoria("--arbitrar"), arbitro=T.arbitro_com(cli, 5))
            self.assertEqual(res["arbitrados"], 1)                # só o DIST-H99-Fácil-00002
            self.assertEqual(cli.chamadas["validador"], 1)        # o árbitro usa o prompt do validador
            self.assertEqual(sum(cli.chamadas.values()), 1)
            reg = amb.linhas("arbitragem_d2.jsonl")[0]
            self.assertEqual((reg["codigo_item"], reg["arbitro_confirma_erro"]), ("DIST-H99-Fácil-00002", True))
            cli2 = T.Roteiro()
            res2 = T.silencioso(ab.arbitrar, amb.args_auditoria("--arbitrar"), arbitro=T.arbitro_com(cli2, 5))
            self.assertEqual((res2["arbitrados"], cli2.total()), (0, 0))
            # decisão humana tira o item da arbitragem
            (amb.d / "arbitragem_d2.jsonl").unlink()
            _decisoes(amb.d / "decisoes_humanas.json", [{"codigo_item": "DIST-H99-Fácil-00002", "acao": "remover",
                                                         "data": "d", "autor": "usuário", "motivo": "m"}])
            self.assertEqual(ab.candidatos_d2(amb.args_auditoria()), [])
            # fora do dry-run, sem teto explícito, não gasta
            with self.assertRaises(SystemExit):
                ab.arbitrar(amb.args_auditoria("--arbitrar"))
        finally:
            amb.fechar()


# ===========================================================================
class TestP1PreFiltroGeometria(unittest.TestCase):
    TRI = ("Um triângulo tem lados de {a} cm, {a} cm e {c} cm. Quanto aos lados, ele é:")

    def q(self, a=6, c=7, alts=("Equilátero", "Isósceles", "Escaleno", "Acutângulo", "Obtusângulo")):
        return {"enunciado": self.TRI.format(a=a, c=c), "alternativas": dict(zip(LETRAS, alts)),
                "resolucao_passo_a_passo": "Dois lados iguais: isósceles.", "resposta_correta": "B",
                "difficulty": "EASY"}

    def test_filtro_injecao_barra_distrator_verdadeiro_antes_dos_juizes(self):
        obj = {"questoes": [self.q()]}
        motivo, det = aq.filtro_injecao(obj, json.dumps(obj, ensure_ascii=False), ano="9º", habilidade="H17",
                                        dificuldade="Fácil")
        self.assertEqual(motivo, "geometria_d5")
        self.assertEqual(det["geometria_d5"]["outro_eixo"], ["D"])
        # controle: 5-5-6 com "Retângulo" (falso) passa
        ok = {"questoes": [self.q(5, 6, ("Equilátero", "Isósceles", "Escaleno", "Retângulo", "Obtusângulo"))]}
        self.assertIsNone(aq.filtro_injecao(ok, json.dumps(ok, ensure_ascii=False), ano="9º", habilidade="H17",
                                            dificuldade="Fácil")[0])

    def test_na_injecao_nao_gasta_juizes(self):
        q = self.q()

        class Gera(T.Roteiro):
            def _gerar(self, user):
                self.n += 1
                return json.dumps({"questoes": [q]}, ensure_ascii=False)
        amb = T.Ambiente(falta_facil=1)
        try:
            cli = Gera()
            res = T.silencioso(iq.injetar, amb.args_injecao("--tentativas-por-slot", "2"),
                               agentes=T.agentes_com(cli, 100))
            self.assertEqual(res["funil"]["aceitos"], 0)
            self.assertEqual({r["motivo"] for r in amb.linhas("rejeitadas.jsonl")}, {"geometria_d5"})
            self.assertEqual(cli.chamadas["validador"] + cli.chamadas["revisor"], 0)
            self.assertEqual(cli.chamadas["gerador"], 2)
        finally:
            amb.fechar()

    def test_auditoria_so_informa_e_montagem_exclui_injetada(self):
        f = aq.filtros_auditoria(self.q())
        self.assertNotIn("geometria_d5", f["reprovado"])         # D2 não remove por ele sozinho
        self.assertEqual(f["geometria_d5"]["outro_eixo"], ["D"])
        ex = T.exemplo("9º", "H17", self.q(), "INJ-9-H17-F-00099", versao_prompts=aq.VERSAO_PROMPTS)
        self.assertEqual(ab.injetada_valida(ex), (False, "geometria_d5"))
        self.assertEqual(iq.filtros_rejulgamento(self.q(), "9º", "H17"), "geometria_d5")


# ===========================================================================
def _tax_lexico(chave="5º|H04", ids=("multiplicacao", "divisao")):
    """Taxonomia mínima com as palavras-chave ATUAIS do LEXICO (sem depender
    de data/taxonomia_subtemas.json)."""
    lex = {e[0]: e for e in bt.LEXICO}
    return {"habilidades": {chave: {"subtemas": [{"id": i, "rotulo": lex[i][1], "palavras_chave": [lex[i][3]]}
                                                 for i in ids], "tipos_raciocinio": ["calculo"]}}}


class TestP2MultiplicacaoDivisao(unittest.TestCase):
    # as 4 candidatas de multiplicação do 5º H04 rejeitadas no piloto 1 (36% de falsas)
    MULT = [
        "Em uma campanha de reciclagem, 248 alunos da escola juntaram 125 latinhas de alumínio cada um. Quantas "
        "latinhas foram arrecadadas ao todo?",
        "Em uma campanha de reciclagem, 12 turmas participaram. Cada turma foi dividida em 8 grupos com 1 aluno em "
        "cada grupo, e cada aluno recolheu 15 kg de papel. Quantos quilogramas de papel foram recolhidos ao todo?",
        "No mercado do bairro, o gerente comprou 300 caixas de iogurte, cada uma com 200 unidades. Se cada iogurte "
        "for vendido por R$ 15, qual será o valor total arrecadado com a venda de todos os iogurtes?",
        "Em uma feira do bairro, Dona Lúcia vendeu 145 caixas de laranja, cada uma por R$ 638. Qual foi o valor "
        "total arrecadado por ela com a venda dessas caixas?",
    ]
    DIV = [
        "240 bombons são igualmente divididos entre 8 crianças. Quantos bombons cada criança recebe?",
        "Ana tem 48 balas e vai dar a mesma quantidade a 6 amigos. Quantas balas cada um vai receber?",
        "Em uma horta há 18.480 mudas para 24 canteiros. Quantas mudas serão plantadas em cada canteiro?",
    ]

    def test_multiplicacao_nao_e_mais_rejeitada_como_divisao(self):
        tax = _tax_lexico()
        slot = {"subtema": "multiplicacao"}
        for en in self.MULT:
            q = {"enunciado": en, "alternativas": {L: str(i) for i, L in enumerate(LETRAS)}}
            with self.subTest(en=en[:50]):
                self.assertFalse(dt.subtema_divergente(q, "5º", "H04", slot, tax))
                self.assertEqual(dv.classificar_questao(q, "5º", "H04", tax)["subtema"], "multiplicacao")

    def test_divisao_continua_divisao(self):
        tax = _tax_lexico()
        for en in self.DIV:
            q = {"enunciado": en, "alternativas": {L: str(i) for i, L in enumerate(LETRAS)}}
            with self.subTest(en=en[:50]):
                self.assertEqual(dv.classificar_questao(q, "5º", "H04", tax)["subtema"], "divisao")
                self.assertTrue(dt.subtema_divergente(q, "5º", "H04", {"subtema": "multiplicacao"}, tax))

    def test_palavras_chave_portaveis(self):
        lex = {e[0]: e for e in bt.LEXICO}
        for i in ("multiplicacao", "divisao"):
            for proibido in ("(?<=", "(?<!", "(?P<", "(?i"):
                self.assertNotIn(proibido, lex[i][3])
        self.assertNotIn("cada um", lex["divisao"][3].split("|"))  # a palavra solta que puxava para divisão

    def test_taxonomia_regenerada(self):
        p = ROOT / "data" / "taxonomia_subtemas.json"
        if not p.exists():
            self.skipTest("taxonomia ausente")
        tax = json.loads(p.read_text(encoding="utf-8"))
        lex = {e[0]: e for e in bt.LEXICO}
        for k, h in tax["habilidades"].items():
            for s in h["subtemas"]:
                if s["id"] in ("multiplicacao", "divisao"):
                    self.assertEqual(s["palavras_chave"], [lex[s["id"]][3]], k)


# ===========================================================================
class TestP3RevisorSemAncora(unittest.TestCase):
    def test_mensagem_sem_dificuldade_pedida_e_rubrica_no_sistema(self):
        cli = T.Grava()
        T.agentes_com(cli).revisar(T.questao_boa(), "5º", "H04", "d", "adição", "Difícil")
        user = [m for p, m in cli.msgs if p == "revisor"][0][1]["content"]
        self.assertNotIn("Dificuldade", user)
        self.assertNotIn("Difícil", user)
        for trecho in ("RUBRICA ABSOLUTA", "etapas", "1º ano: até 100", "5º ano: centenas de milhar",
                       "Fácil = uma etapa", "C3 é sempre true"):
            self.assertIn(trecho, aq.REVISOR_SISTEMA)
        self.assertNotIn("compare com a pedida", aq.REVISOR_SISTEMA)

    def test_subtema_que_vaza_a_resposta_nao_vai(self):
        q = {"enunciado": "Que sólido tem duas bases circulares e uma superfície lateral curva?",
             "alternativas": {"A": "Cilindro", "B": "Cone", "C": "Esfera", "D": "Cubo", "E": "Pirâmide"},
             "resolucao_passo_a_passo": "Duas bases circulares: cilindro.", "resposta_correta": "A",
             "difficulty": "EASY"}
        cli = T.Grava()
        T.agentes_com(cli).revisar(q, "5º", "H12", "d", "cilindros")
        user = [m for p, m in cli.msgs if p == "revisor"][0][1]["content"]
        self.assertIn("Subtema pedido: não especificado", user)
        self.assertNotIn("cilindros", user)
        self.assertTrue(aq.subtema_vaza_resposta("raio e diâmetro", {"A": "Raio", "B": "Diâmetro", "C": "Corda",
                                                                     "D": "Arco", "E": "Centro"}))
        self.assertTrue(aq.subtema_vaza_resposta("retangulo_quadrado", {"A": "RETÂNGULO.", "B": "x"}))
        # subtema que não coincide com alternativa continua indo
        self.assertFalse(aq.subtema_vaza_resposta("adição", {L: f"{i} figurinhas" for i, L in enumerate(LETRAS)}))
        self.assertFalse(aq.subtema_vaza_resposta("quadriláteros", {"A": "Losango", "B": "Trapézio", "C": "Quadrado",
                                                                    "D": "Pipa", "E": "Retângulo"}))
        self.assertEqual(aq.subtema_para_revisor("adição", {"A": "53"}), "adição")

    def test_d3_so_com_a_rubrica_sem_ancora(self):
        ex = T.exemplo("9º", "H99", T.questao_boa(dif="MEDIUM"), "DIST-H99-Moderado-00001", dificuldade="Moderado",
                       destilado=True, professor="p")
        reg = T._registro(ex, 2, T._juiz(True), T._juiz(True, dif="Fácil"))
        self.assertTrue(ab.deve_rebaixar_para_facil(reg, ex["meta"], "revisor"))
        antigo = {k: v for k, v in reg.items() if k != "rubrica_dificuldade"}  # piloto 2: com âncora
        self.assertFalse(ab.deve_rebaixar_para_facil(antigo, ex["meta"], "revisor"))
        # passo 2: a regra fixada é "suspensa" (nenhum rebaixamento automático)
        self.assertEqual(ab.REGRA_D3, "suspensa")
        self.assertFalse(ab.deve_rebaixar_para_facil(reg, ex["meta"]))

    def test_auditoria_grava_a_marca_da_rubrica(self):
        amb = T.Ambiente()
        try:
            T.silencioso(ab.auditar, amb.args_auditoria(), agentes=T.agentes_com(T.Roteiro(), 100))
            self.assertTrue(all(r["rubrica_dificuldade"] == aq.RUBRICA_DIFICULDADE
                                for r in amb.linhas("auditoria.jsonl")))
        finally:
            amb.fechar()


# ===========================================================================
class TestP4CacheSemTruncada(unittest.TestCase):
    class Cortada(T.Roteiro):
        """Responde certo, mas com finish_reason "length" na 1ª chamada do validador."""

        def __init__(self):
            super().__init__()
            self.cortar = True

        def chat_completion(self, messages, max_tokens, temperature):
            r = super().chat_completion(messages, max_tokens, temperature)
            if T.papel_de(messages) == "validador" and self.cortar:
                self.cortar = False
                r.choices[0].finish_reason = "length"
            return r

    def test_truncada_nao_vai_para_o_cache_e_e_refeita(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "cache.jsonl"
            q = T.questao_boa()
            cli = self.Cortada()
            ag1 = T.agentes_com(cli, 10)
            ag1.cache_juizes = aq.CacheRespostas([path])
            ag1.validar(q)
            self.assertEqual(ag1.cache_juizes.resumo()["truncadas_nao_gravadas"], 1)
            self.assertEqual(ag1.cache_juizes.resumo()["respostas_novas_gravadas"], 1)  # só a fase 2
            cli2 = T.Roteiro()
            ag2 = T.agentes_com(cli2, 10)
            ag2.cache_juizes = aq.CacheRespostas([path])
            ag2.validar(q)
            self.assertEqual(cli2.chamadas["validador"], 1)       # refeita de verdade
            self.assertEqual(cli2.chamadas["validador_fase2"], 0)  # a fase 2 veio do cache

    def test_cache_da_calibracao_tambem(self):
        import calibrar_agentes as ca
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cli = self.Cortada()
            ag1 = ca.AgentesComCache({p: cli for p in aq.PAPEIS}, orcamento=aq.Orcamento(10), log_uso=None,
                                     dormir=lambda s: None, cache_path=d / "c.jsonl",
                                     ledger=ca.Ledger(d / "l.json", 10), rodada="t")
            ag1.validar(T.questao_boa())
            papeis = [json.loads(x)["papel"] for x in (d / "c.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(papeis, ["validador_fase2"])

    def test_cliente_maritaca_repassa_o_finish_reason(self):
        dados = {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 2}}
        with mock.patch("requests.post", return_value=SimpleNamespace(raise_for_status=lambda: None,
                                                                      json=lambda: dados)):
            cli = aq.ClienteMaritacaMedido.__new__(aq.ClienteMaritacaMedido)
            cli.api_key, cli.model, cli.timeout = "k", "m", 1
            r = cli.chat_completion([{"role": "user", "content": "u"}], 10, 0)
        self.assertEqual(r.choices[0].finish_reason, "length")
        self.assertTrue(aq.resposta_truncada({"finish_reason": "length"}))
        self.assertTrue(aq.resposta_truncada({"finish_reason": "MAX_TOKENS"}))
        self.assertFalse(aq.resposta_truncada({"finish_reason": "stop"}))
        self.assertFalse(aq.resposta_truncada({}))


if __name__ == "__main__":
    unittest.main()


class TestEdicaoAlternativasD6(unittest.TestCase):
    """Restauração humana das frações que a planilha virou data (D6),
    decisão do usuário em 2026-10-01: só troca texto de letras existentes, com
    pré-condição por letra; gabarito e demais campos intactos."""

    def _ex(self, alt_a="2025-06-03 00:00:00"):
        q = {"enunciado": "Dado de 6 faces: probabilidade de sair par?",
             "alternativas": {"A": alt_a, "B": "2/6", "C": "4/6", "D": "1/6",
                              "E": "Nenhuma das alternativas anteriores"},
             "resolucao_passo_a_passo": "Pares: 2, 4, 6 -> 3/6", "resposta_correta": "A",
             "difficulty": "MEDIUM"}
        return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                             {"role": "assistant", "content": json.dumps({"questoes": [q]}, ensure_ascii=False)}],
                "meta": {"codigo_item": "MT90123MH25MT", "dificuldade": "Moderado"}}

    def _dec(self, pre="2025-06-03 00:00:00"):
        return {"codigo_item": "MT90123MH25MT", "acao": "manter_editar", "data": "2026-10-01", "autor": "usuário",
                "motivo": "D6", "edicoes": {"alternativas": {"A": "3/6"}},
                "pre_condicoes": {"alternativas": {"A": pre}, "resposta_correta": "A"}}

    def test_restaura_so_a_letra(self):
        novo, porque = ab.aplicar_edicoes_humanas(self._ex(), self._dec())
        self.assertIsNone(porque)
        q = json.loads(novo["messages"][2]["content"])["questoes"][0]
        self.assertEqual(q["alternativas"]["A"], "3/6")
        self.assertEqual(q["alternativas"]["B"], "2/6")
        self.assertEqual(q["resposta_correta"], "A")
        self.assertIsNone(ab.corrompido_planilha(novo))

    def test_pre_condicao_diferente_nao_aplica(self):
        novo, porque = ab.aplicar_edicoes_humanas(self._ex(alt_a="3/6"), self._dec())
        self.assertIsNone(novo)
        self.assertEqual(porque, "pre_condicao_alternativa_A")

    def test_sem_pre_condicao_nao_aplica(self):
        dec = self._dec(); dec["pre_condicoes"] = {}
        novo, porque = ab.aplicar_edicoes_humanas(self._ex(), dec)
        self.assertEqual(porque, "alternativas_sem_pre_condicao")
