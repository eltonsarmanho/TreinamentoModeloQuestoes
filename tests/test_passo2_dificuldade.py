"""Testes do passo 2 da recalibração (2026-10-01): referência de dificuldade
com itens REAIS do banco, concordância ordinal (kappa ponderado), árbitro
Gemini julgando a dificuldade pela mesma rubrica do revisor e a escolha da
regra D3 a partir dos números.

Sem rede e sem custo: HTTP falso, linhas de calibração montadas à mão.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import agentes_questoes as aq  # noqa: E402
import arbitro_gemini as ag  # noqa: E402
import calibrar_agentes as ca  # noqa: E402


class FakeRow(dict):
    def keys(self):
        return list(super().keys())


def _row(cod, grau, ano="5º", hab="H01", enunciado="Quanto é 12 + 7?"):
    r = {"disciplina": "Matemática", "imagem": None, "enunciado_item": enunciado, "texto_auxiliar": "",
         "gabarito": "B", "grau_resolucao": grau, "codigo_item": cod, "ano": ano, "habilidade": hab,
         "descricao_item": "Resolver problemas", "lote": None, "justificativa_geral": "",
         "depende_de_imagem": 0, "descricao_imagem": None, "imagem_path": None}
    for k, L in enumerate("abcd"):
        r[f"alternativa_{L}"] = ("18", "19", "20", "21")[k]
        r[f"alternativa_{L}_imagem"] = None
        r[f"justificativa_alternativa_{L}"] = "Conta: 12 + 7 = 19, logo B." if L == "b" else ""
    return FakeRow(r)


def _linha(id_, banco, revisor, gemini=None, ano="5º", veredito=True):
    linha = {"id": id_, "conjunto": "dif45", "boa": True, "ano": ano, "dificuldade_rotulo": banco,
             "validador": None, "par": None,
             "revisor": {"veredito": veredito, "avaliado": True, "dificuldade_real": revisor, "problemas": []}}
    if gemini is not None:
        linha["gemini"] = {"avaliado": True, "dificuldade_real": gemini}
    return linha


class HttpFalso:
    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.reqs = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.reqs.append((url, headers, json))
        st, corpo = self.respostas.pop(0)
        return SimpleNamespace(status_code=st, json=lambda: corpo)


def resp_gemini(texto, fim="STOP"):
    return {"candidates": [{"content": {"parts": [{"text": "pensando", "thought": True}, {"text": texto}]},
                            "finishReason": fim}],
            "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 20, "thoughtsTokenCount": 30}}


class TestReferenciaDificuldade(unittest.TestCase):
    def test_so_codigo_com_letra_que_bate_com_o_grau(self):
        rows = [_row("MT5001FH01MT", "Fácil"), _row("MT5002MH01MT", "Moderado"),
                _row("MT5003DH01MT", "Difícil"),
                _row("MT2003DH01TD", "Fácil"),                    # letra D, grau Fácil: ambíguo, fora
                _row("EF01MA02-001-L2-2026-09", "Fácil"),         # sem letra no código, fora
                _row("MT9018MH04MT", "Moderado")]                 # decisão humana (reclassificado), fora
        itens, info = ca.carregar_ref_dificuldade(n_por_nivel=5, rows=rows)
        self.assertEqual(sorted(it["codigo_item"] for it in itens), ["MT5001FH01MT", "MT5002MH01MT", "MT5003DH01MT"])
        self.assertEqual({it["codigo_item"]: it["dificuldade"] for it in itens},
                         {"MT5001FH01MT": "Fácil", "MT5002MH01MT": "Moderado", "MT5003DH01MT": "Difícil"})
        self.assertEqual(info["excluidos"]["letra_diverge_do_grau"], 1)
        self.assertEqual(info["excluidos"]["codigo_sem_dificuldade"], 1)
        self.assertEqual(info["excluidos"]["excluido_lista"], 1)
        self.assertTrue(all(it["conjunto"] == "dif45" for it in itens))

    def test_estratifica_por_nivel_e_ano(self):
        rows = [_row(f"MT{a}{i:03d}{L}H0{i % 3 + 1}MT", g, ano=f"{a}º", enunciado=f"Quanto é {i} + 7?")
                for a in (2, 5, 9) for i, (L, g) in enumerate([("F", "Fácil"), ("M", "Moderado"), ("D", "Difícil")] * 3)]
        itens, _ = ca.carregar_ref_dificuldade(n_por_nivel=3, rows=rows)
        par = sorted((it["dificuldade"], it["ano"]) for it in itens)
        self.assertEqual(par, sorted((g, a) for g in ("Fácil", "Moderado", "Difícil") for a in ("2º", "5º", "9º")))
        again, _ = ca.carregar_ref_dificuldade(n_por_nivel=3, rows=rows)
        self.assertEqual([x["id"] for x in itens], [x["id"] for x in again])   # semente fixa


class TestConcordancia(unittest.TestCase):
    def test_kappa_ponderado(self):
        perfeito = [("Fácil", "Fácil"), ("Moderado", "Moderado"), ("Difícil", "Difícil")]
        self.assertEqual(ca.kappa_ponderado(perfeito), 1.0)
        # errar por 2 níveis pesa mais que por 1 (quadrático)
        um = perfeito + [("Difícil", "Moderado")]
        dois = perfeito + [("Difícil", "Fácil")]
        self.assertGreater(ca.kappa_ponderado(um), ca.kappa_ponderado(dois))
        # valor conhecido: tudo "Fácil" do outro lado = sem informação
        self.assertEqual(ca.kappa_ponderado([("Fácil", "Fácil"), ("Moderado", "Fácil")]), 0.0)
        self.assertIsNone(ca.kappa_ponderado([]))
        self.assertEqual(ca.kappa_ponderado(perfeito, "nominal"), 1.0)

    def test_rebaixaria_e_matriz(self):
        linhas = [_linha("a", "Fácil", "Fácil"), _linha("b", "Moderado", "Fácil"),
                  _linha("c", "Moderado", "Moderado"), _linha("d", "Difícil", "Fácil"),
                  _linha("e", "Difícil", "Difícil", veredito=False)]
        m = ca.metricas_dificuldade(linhas)
        r = m["revisor"]
        self.assertEqual(r["rebaixaria_para_facil"]["n"], 2)
        self.assertEqual(r["rebaixaria_para_facil"]["de"], 4)
        self.assertEqual(r["matriz_banco_x_julgado"]["Difícil"], {"Fácil": 1, "Moderado": 0, "Difícil": 1})
        self.assertEqual(r["rebaixaria_ids"], ["b", "d"])
        self.assertEqual(m["revisor_falsa_rejeicao_reais"]["ids"], ["e"])


class TestRegraD3(unittest.TestCase):
    def _md(self, rev_reb, gem=None):
        """10 M/D + 5 F; rev_reb dos M/D o revisor diz Fácil; gem = quantos
        desses o Gemini também diz Fácil."""
        linhas = [_linha(f"f{i}", "Fácil", "Fácil", "Fácil" if gem is not None else None) for i in range(5)]
        for i in range(10):
            banco = "Moderado" if i % 2 else "Difícil"
            rev = "Fácil" if i < rev_reb else banco
            g = None
            if gem is not None:
                g = "Fácil" if i < gem else banco
            linhas.append(_linha(f"m{i}", banco, rev, g))
        return ca.metricas_dificuldade(linhas)

    def test_revisor_sozinho_quando_concorda_bem(self):
        m = self._md(rev_reb=1)
        self.assertEqual(m["regra_d3"]["regra"], "revisor")

    def test_dois_de_dois_quando_o_revisor_rebaixa_demais(self):
        m = self._md(rev_reb=5, gem=1)
        self.assertEqual(m["regra_2_de_2"]["rebaixaria"], 1)
        self.assertEqual(m["regra_d3"]["regra"], "revisor_e_arbitro")

    def test_suspensa_quando_nem_o_2_de_2_basta(self):
        self.assertEqual(self._md(rev_reb=5, gem=4)["regra_d3"]["regra"], "suspensa")
        self.assertEqual(self._md(rev_reb=5)["regra_d3"]["regra"], "suspensa")   # sem Gemini

    def test_alvo_do_gemini_todos_md_e_amostra_de_f(self):
        linhas = [_linha(f"f{i}", "Fácil", "Fácil", ano=("2º", "5º", "9º")[i % 3]) for i in range(9)]
        linhas += [_linha(f"m{i}", "Moderado", "Fácil") for i in range(4)]
        alvo = ca.alvo_gemini(linhas, n_facil=3)
        self.assertEqual(sum(l["dificuldade_rotulo"] != "Fácil" for l in alvo), 4)
        self.assertEqual(sorted(l["ano"] for l in alvo if l["dificuldade_rotulo"] == "Fácil"), ["2º", "5º", "9º"])


class TestGeminiDificuldade(unittest.TestCase):
    Q = {"enunciado": "Ana tinha 12 figurinhas e ganhou 7. Com quantas ficou?",
         "alternativas": {"A": "18", "B": "19", "C": "20", "D": "21", "E": "Nenhuma das alternativas anteriores"},
         "resposta_correta": "B", "resolucao_passo_a_passo": "MARCADOR-RESOLUCAO 12 + 7 = 19", "difficulty": "HARD"}

    def test_rubrica_e_a_do_revisor_e_nada_do_autor_vai(self):
        self.assertIn("Fácil = uma etapa", ag.ARBITRO_DIFICULDADE_SISTEMA)
        self.assertIn(ag._rubrica_c3(), aq.REVISOR_SISTEMA)
        http = HttpFalso([(200, resp_gemini('{"resolucao": "12 + 7 = 19", "etapas": 1, "exigencias": [], '
                                            '"dificuldade_real": "Fácil"}'))])
        arb = ag.Arbitro(ag.ClienteGemini("g", api_key="k", http=http), "g", aq.Orcamento(1), log_uso=None,
                         cache_path=None)
        r = arb.julgar_dificuldade(self.Q, "1º", "EF01MA08", "Resolver problemas de adição")
        enviado = json.dumps(http.reqs[0][2], ensure_ascii=False)
        # sem resolução do autor, sem gabarito e sem a dificuldade pedida (a âncora que o P3 tirou)
        for proibido in ("MARCADOR-RESOLUCAO", "resposta_correta", "HARD", "Dificuldade:", "Dificuldade pedida"):
            self.assertNotIn(proibido, enviado)
        self.assertTrue(r["avaliado"])
        self.assertEqual((r["dificuldade_real"], r["etapas"]), ("Fácil", 1))
        with self.assertRaises(aq.OrcamentoEsgotado):
            arb.julgar_dificuldade(self.Q, "1º", "EF01MA08")

    def test_malformada_ou_truncada_nao_avalia_e_nao_entra_no_cache(self):
        with tempfile.TemporaryDirectory() as t:
            cache = Path(t) / "c.jsonl"
            bom = '{"resolucao": "x", "etapas": 2, "exigencias": ["leitura"], "dificuldade_real": "Moderado"}'
            http = HttpFalso([(200, resp_gemini('{"resolucao": "x", "dificuldade_real": "Média-alta"}')),
                              (200, resp_gemini(bom, fim="MAX_TOKENS")), (200, resp_gemini(bom))])
            arb = ag.Arbitro(ag.ClienteGemini("g", api_key="k", http=http), "g", aq.Orcamento(5), log_uso=None,
                             cache_path=cache)
            self.assertFalse(arb.julgar_dificuldade(self.Q, "5º", "H01")["avaliado"])   # nível inválido
            self.assertFalse(arb.julgar_dificuldade(self.Q, "5º", "H01")["avaliado"])   # truncada
            self.assertFalse(cache.exists())
            self.assertEqual(arb.julgar_dificuldade(self.Q, "5º", "H01")["dificuldade_real"], "Moderado")
            arb2 = ag.Arbitro(ag.ClienteGemini("g", api_key="k", http=HttpFalso([])), "g", aq.Orcamento(0),
                              log_uso=None, cache_path=cache)
            self.assertEqual(arb2.julgar_dificuldade(self.Q, "5º", "H01")["dificuldade_real"], "Moderado")
            self.assertEqual(arb2.hits, 1)
            # o cache de dificuldade não serve de resposta ao árbitro da D2 (outro papel, outra chave)
            linhas = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([x["papel"] for x in linhas], ["arbitro_dificuldade"])


class TestVerificadorExtenso(unittest.TestCase):
    """idx 1052 (DIST-H03-Difícil-00273): os dois juízes novos aprovaram juntos na
    regressão r7 um item com distrator verdadeiro ("Três centenas e sete unidades"
    também é 307). O verificador exato do número por extenso o pega."""

    def q(self, n, alts, gab="A", en=None):
        return {"enunciado": en or f"Foram arrecadadas {n} garrafas. Qual alternativa escreve esse número por extenso?",
                "alternativas": dict(zip("ABCDE", alts)), "resposta_correta": gab}

    def test_grafia_canonica(self):
        self.assertEqual(aq.numero_por_extenso(307), ("trezentos e sete", None))
        self.assertEqual(aq.numero_por_extenso(100)[0], "cem")
        self.assertEqual(aq.numero_por_extenso(1200)[0], "mil e duzentos")
        self.assertEqual(aq.numero_por_extenso(2405)[0], "dois mil quatrocentos e cinco")
        self.assertEqual(aq.numero_por_extenso(21300)[0], "vinte e um mil e trezentos")

    def test_1052_contradiz_por_decomposicao_verdadeira(self):
        q = self.q(307, ["Trezentos e sete", "Trezentos e setenta", "Três centenas e sete unidades",
                         "Trezentos sete", "Trinta e sete"])
        r = aq.verificar_aritmetica(q)
        self.assertEqual((r["status"], r["verdadeiras"]), ("contradiz", ["A", "C"]))
        # a mesma questão com outro distrator: só A, confirma (grafia sem "e" é falsa)
        q["alternativas"]["C"] = "Três centenas e oito unidades"
        self.assertEqual(aq.verificar_aritmetica(q)["status"], "confirma")

    def test_gabarito_errado_e_numerais(self):
        q = self.q(345, ["TREZENTOS E CINQUENTA E QUATRO.", "TREZENTOS E QUARENTA E CINCO.", "345 + 0",
                         "QUATROCENTOS E TRINTA E CINCO.", "Nenhuma das alternativas anteriores"], gab="A",
                   en="LAURA JÁ TEM 345 FIGURINHAS. COMO SE ESCREVE ESSE NÚMERO POR EXTENSO?")
        r = aq.verificar_aritmetica(q)
        self.assertEqual((r["status"], r["verdadeiras"]), ("contradiz", ["B", "C"]))

    def test_abstem_no_conector_do_mil_e_em_texto_desconhecido(self):
        # 10.408: "Dez mil e quatrocentos e oito" é a variante discutível -> sem veredito (idx 1187 é correto)
        q = self.q("10.408", ["Dez mil quatrocentos e oito.", "Mil quatrocentos e oito.",
                              "Dez mil e quatrocentos e oito.", "Dez mil quatrocentos e oitenta.",
                              "Um mil quatrocentos e oito."])
        self.assertEqual(aq.verificar_aritmetica(q)["status"], "sem_veredito")
        q = self.q(307, ["Trezentos e sete", "Um número ímpar", "Trinta e sete", "Setenta", "Três"])
        self.assertEqual(aq.verificar_aritmetica(q)["status"], "sem_veredito")
        # sem pedido de escrita por extenso: não se aplica
        q = self.q(307, ["Trezentos e sete", "Três centenas e sete unidades", "a", "b", "c"],
                   en="Foram arrecadadas 307 garrafas. Qual alternativa é verdadeira?")
        self.assertEqual(aq.verificar_aritmetica(q)["status"], "sem_veredito")

    def test_pipeline_conta_o_filtro_antes_do_par(self):
        linhas = [{"id": "r", "boa": False, "par": True, "filtro_deterministico": ["verificador_aritmetico"],
                   "conjunto": "defeitos"},
                  {"id": "b", "boa": True, "par": True, "filtro_deterministico": [], "conjunto": "defeitos"}]
        m = ca.metricas_pipeline(linhas)
        self.assertEqual((m["fp"], m["vn"], m["vp"]), (0, 1, 1))
        self.assertEqual(m["barradas_pelo_filtro"], ["r"])


if __name__ == "__main__":
    unittest.main()
