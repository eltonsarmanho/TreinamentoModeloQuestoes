"""Ciclo v6: profiling, distratores, triagem/reescrita de resoluções, chamada medida e plano de injeção.

Critérios de aceite cobertos (ciclo_v6/SPEC.md): falha de API/truncada/JSON inválido nunca aprova;
o teto de chamadas e o de R$ nunca são ultrapassados; o dry-run não usa rede nem custa; a questão
original (enunciado, alternativas, gabarito) nunca é alterada; a proveniência fica no registro.
"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import auditar_distratores as ad  # noqa: E402
import melhorar_resolucoes as mr  # noqa: E402
import perfil_base as pb  # noqa: E402
import planejar_injecao_v6 as pi  # noqa: E402
import sabia_v6 as sv  # noqa: E402
import triar_resolucoes as tr  # noqa: E402

PRECOS = {"sabia-4": {"entrada": 5.0, "saida": 20.0}, "sabia-4-thinking": {"entrada": 5.0, "saida": 40.0}}


def q_soma(res="Pedro tinha 35 figurinhas e ganhou 18. Somamos as duas quantidades: 35 + 18 = 53. "
                "Logo, Pedro ficou com 53 figurinhas no total.", gab="C"):
    alts = {"A": "17 figurinhas", "B": "43 figurinhas", "C": "53 figurinhas", "D": "54 figurinhas", "E": "63 figurinhas"}
    return {"enunciado": "Pedro tinha 35 figurinhas e ganhou 18 do primo. Quantas figurinhas tem agora?",
            "alternativas": alts, "resolucao_passo_a_passo": res, "resposta_correta": gab, "difficulty": "EASY"}


def resp(texto, fim="stop", usage=(100, 50)):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=texto), finish_reason=fim)],
                           usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]))


class Fila:
    def __init__(self, respostas):
        self.respostas, self.n = list(respostas), 0

    def chat_completion(self, messages, max_tokens, temperature):
        self.n += 1
        r = self.respostas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def chamador(clientes, max_chamadas=10, teto=1.0, simulado=False, **kw):
    papeis = {"professor_resolucao": {"modelo": "sabia-4", "max_tokens": 900, "temperatura": 0.2, "versao_prompt": "p1"},
              "avaliador_resolucao": {"modelo": "sabia-4", "max_tokens": 300, "temperatura": 0.0, "versao_prompt": "a1"}}
    if not isinstance(clientes, dict):
        clientes = {p: clientes for p in papeis}
    return sv.ChamadorSabia(papeis, clientes, max_chamadas, teto, PRECOS, tentativas=1, dormir=lambda s: None,
                            simulado=simulado, **kw)


PROF_OK = json.dumps({"discorda": False, "motivo": "", "resolucao":
                      "A pergunta pede o total de figurinhas. Pedro tinha 35 e ganhou mais 18, então juntamos as "
                      "quantidades: 35 + 18 = 53. Pedro ficou com 53 figurinhas."}, ensure_ascii=False)
AVAL_OK = json.dumps({"resolucao_correta": True, "usa_dado_ausente": False, "justificativa_falsa": False,
                      "chega_na_alternativa": True, "problemas": [], "confianca": 0.9})
ITEM = {"codigo_item": "X-1", "linha": 0, "questao": 0, "ano": "2º", "habilidade": "H01", "origem": "real_L2",
        "classe": "correta_pouco_explicativa"}


# ---------------------------------------------------------------- perfil
class TestPerfil(unittest.TestCase):
    def test_entropia_usa_o_universo_e_nao_so_as_categorias_presentes(self):
        self.assertAlmostEqual(pb.entropia_normalizada(["a", "b"] * 5, 2), 1.0)
        self.assertLess(pb.entropia_normalizada(["a", "b"] * 5, 5), 0.5)  # 2 de 5 subtemas não é diverso
        self.assertIsNone(pb.entropia_normalizada(["a"] * 4, 1))  # nada a diversificar

    def test_molde_ignora_so_os_numeros(self):
        a, b = q_soma(), q_soma()
        b = dict(b, enunciado=b["enunciado"].replace("35", "40"))
        self.assertEqual(pb.molde(a), pb.molde(b))
        self.assertNotEqual(pb.hash_questao(a), pb.hash_questao(b))

    def test_perfil_conta_metadado_ausente_como_desconhecido(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "b.jsonl"
            ex = {"messages": [{"role": "system", "content": "s"},
                               {"role": "user", "content": "Gere 1 questão(ões) de matemática. Ano: 2º ano. "
                                                           "Habilidade: H01 — d. Dificuldade: Fácil."},
                               {"role": "assistant", "content": json.dumps({"questoes": [q_soma()]})}],
                  "meta": {"codigo_item": "DIST-H01-Fácil-1", "ano": "2º", "habilidade": "H01", "dificuldade": "Fácil"}}
            base.write_text(json.dumps(ex, ensure_ascii=False) + "\n", encoding="utf-8")
            p, itens, pares = pb.perfilar(base)
        self.assertEqual(p["questoes"], 1)
        self.assertEqual(p["metadados_taxa_classificacao_pct"]["subtema_meta"], 0.0)
        self.assertEqual(itens[0]["meta_subtema"], pb.DESCONHECIDO)
        self.assertEqual(p["classe_auditoria"], {pb.DESCONHECIDO: 1})


# ---------------------------------------------------------------- distratores
class TestDistratores(unittest.TestCase):
    def test_mesmo_numero_com_objeto_diferente_nao_e_valor_repetido(self):
        q = {"enunciado": "Quantos ovos?", "alternativas": {"A": "9 ovos.", "B": "8 ovos.", "C": "8 galinhas.",
                                                           "D": "7 ovos.", "E": "Nenhuma das alternativas anteriores"},
             "resposta_correta": "B"}
        self.assertFalse(ad.avaliar_questao(q)["gabarito_com_valor_repetido"])

    def test_milhar_pt_br_nao_vira_o_mesmo_numero(self):
        q = {"enunciado": "Quanto?", "alternativas": {"A": "5", "B": "500", "C": "5.000", "D": "50.000", "E": "500.000"},
             "resposta_correta": "C"}
        self.assertFalse(ad.avaliar_questao(q)["valor_repetido"])

    def test_mesmo_valor_em_duas_notacoes_vai_para_revisao(self):
        q = {"enunciado": "Escreva quatro vírgula zero cinco.", "alternativas": {"A": "4,05", "B": "4,5", "C": "4,50",
                                                                                "D": "4.05", "E": "405"},
             "resposta_correta": "A"}
        ind = ad.avaliar_questao(q)
        self.assertTrue(ind["gabarito_com_valor_repetido"])
        self.assertTrue(ind["revisao"])

    def test_forma_irredutivel_pedida_isenta_fracao_equivalente(self):
        q = {"enunciado": "Qual fração é equivalente a 7/14 e está na forma irredutível?",
             "alternativas": {"A": "35/70", "B": "1/2", "C": "2/4", "D": "14/7", "E": "Nenhuma das alternativas anteriores"},
             "resposta_correta": "B"}
        ind = ad.avaliar_questao(q, real=True)
        self.assertTrue(ind["fracoes_equivalentes_ao_gabarito"])
        self.assertFalse(ind["revisao"])
        self.assertTrue(ind["nda_artificial"])

    def test_inequacoes_com_sinais_diferentes_nao_sao_repetidas(self):
        q = {"enunciado": "x", "alternativas": {"A": "4 + 0,5x ≥ 10", "B": "4 + 0,5x ≤ 10", "C": "a", "D": "b", "E": "c"},
             "resposta_correta": "B"}
        self.assertFalse(ad.avaliar_questao(q)["alternativas_repetidas"])

    def test_distrator_de_operacao_trocada_e_explicavel(self):
        ind = ad.avaliar_questao(q_soma())  # 17 = 35-18, 43 = 53-10, 54 = 53+1, 63 = 53+10
        self.assertEqual(ind["distratores_valor"], 4)
        self.assertEqual(ind["distratores_explicados"], 4)


# ---------------------------------------------------------------- triagem
class TestTriagem(unittest.TestCase):
    def test_conta_curta_que_confere_e_correta_pouco_explicativa(self):
        self.assertEqual(tr.triar(q_soma(res="Correto - 35 + 18 = 53"))[0], "correta_pouco_explicativa")

    def test_resolucao_completa_e_adequada(self):
        self.assertEqual(tr.triar(q_soma())[0], "adequada")

    def test_conta_que_aponta_outra_alternativa_e_problematica(self):
        # a conta confere (53) mas o gabarito marcado é B (43)
        self.assertEqual(tr.triar(q_soma(gab="B"))[0], "matematicamente_problematica")

    def test_vazia_nao_e_avaliavel(self):
        self.assertEqual(tr.triar(q_soma(res=""))[0], "nao_avaliavel")

    def test_texto_que_nao_chega_a_resposta_e_incompleta(self):
        q = q_soma(res="Precisamos juntar as figurinhas que Pedro já tinha com as que ele ganhou do primo dele.")
        self.assertEqual(tr.triar(q)[0], "incompleta")


# ---------------------------------------------------------------- chamada medida
class TestChamadorSabia(unittest.TestCase):
    def test_teto_de_chamadas_bloqueia_antes_de_enviar(self):
        f = Fila([resp(PROF_OK)])
        ch = chamador(f, max_chamadas=1)
        ch.chamar("professor_resolucao", [{"role": "user", "content": "a"}], hash_exemplo="h1")
        with self.assertRaises(sv.OrcamentoEsgotado):
            ch.chamar("professor_resolucao", [{"role": "user", "content": "b"}], hash_exemplo="h2")
        self.assertEqual(f.n, 1)

    def test_teto_em_reais_reserva_o_pior_caso(self):
        f = Fila([resp(PROF_OK)])
        # pior caso de uma chamada: ~1 token de entrada + 900 de saída a R$20/M = ~R$0,018
        ch = chamador(f, teto=0.01)
        with self.assertRaises(sv.OrcamentoEsgotado):
            ch.chamar("professor_resolucao", [{"role": "user", "content": "a"}], hash_exemplo="h")
        self.assertEqual(f.n, 0)
        self.assertLessEqual(ch.gasto_brl, 0.01)

    def test_modo_real_exige_teto(self):
        with self.assertRaises(ValueError):
            chamador(Fila([]), teto=None)
        with self.assertRaises(ValueError):
            chamador(Fila([]), max_chamadas=0)

    def test_falha_truncada_e_ininterpretavel_sao_abstencao(self):
        for r in (RuntimeError("timeout"), resp(PROF_OK, fim="length"), resp("não é json")):
            ch = chamador(Fila([r]))
            txt = ch.chamar("professor_resolucao", [{"role": "user", "content": "a"}], hash_exemplo="h",
                            interpretavel=lambda t: mr.parse_professor(t) is not None)
            self.assertIsNone(txt)
            self.assertIn("h", ch.abstencoes)

    def test_cache_nao_paga_de_novo_e_log_tem_os_campos(self):
        with tempfile.TemporaryDirectory() as d:
            cache, log = Path(d) / "c.jsonl", Path(d) / "u.jsonl"
            f = Fila([resp(PROF_OK)])
            m = [{"role": "user", "content": "a"}]
            chamador(f, cache=cache, log_uso=log).chamar("professor_resolucao", m, hash_exemplo="h", codigo_item="X")
            f2 = Fila([])
            self.assertEqual(chamador(f2, cache=cache).chamar("professor_resolucao", m, hash_exemplo="h"), PROF_OK)
            self.assertEqual(f2.n, 0)
            reg = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
            for campo in ("modelo", "papel", "versao_prompt", "hash_exemplo", "codigo_item", "prompt_tokens",
                          "completion_tokens", "latencia_s", "custo_brl", "ok"):
                self.assertIn(campo, reg)
            self.assertAlmostEqual(reg["custo_brl"], (100 * 5 + 50 * 20) / 1e6)

    def test_resposta_truncada_nao_entra_no_cache(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "c.jsonl"
            chamador(Fila([resp(PROF_OK, fim="length")]), cache=cache).chamar(
                "professor_resolucao", [{"role": "user", "content": "a"}], hash_exemplo="h")
            self.assertFalse(cache.exists())

    def test_estimativa_soma_os_papeis(self):
        papeis = {"x": {"modelo": "sabia-4", "chamadas_por_item": 1}}
        e = sv.estimar_lote(100, papeis, PRECOS, {"x": (1000, 100, 200)}, retentativas=1.0)
        self.assertEqual(e["chamadas"], 100)
        self.assertAlmostEqual(e["custo_mediano_brl"], 100 * (1000 * 5 + 100 * 20) / 1e6, places=2)


# ---------------------------------------------------------------- reescrita
class TestReescrita(unittest.TestCase):
    def test_validar_nova_rejeita_letra_dado_inventado_e_aceita_a_boa(self):
        q = q_soma(res="Correto - 35 + 18 = 53")
        ok, _m = mr.validar_nova(q, json.loads(PROF_OK)["resolucao"])
        self.assertTrue(ok, _m)
        ok, m = mr.validar_nova(q, json.loads(PROF_OK)["resolucao"] + " Portanto, a alternativa C.")
        self.assertIn("cita_letra", m)
        inventa = ("Pedro tinha 35 e ganhou 18, e depois comprou 9 no mercado: 35 + 18 = 53 e 53 + 9 = 62, "
                   "mas devolveu 9 ao amigo, então 62 - 9 = 53. Pedro ficou com 53 figurinhas.")
        self.assertIn("dado_inventado", mr.validar_nova(q, inventa)[1])

    def test_perder_a_verificabilidade_rejeita(self):
        q = q_soma(res="Correto - 35 + 18 = 53")
        nova = "Juntando o que Pedro tinha com o que ganhou do primo, ele passa a ter 53 figurinhas ao todo agora."
        self.assertIn("perdeu_verificabilidade", mr.validar_nova(q, nova)[1])

    def test_fluxo_aprova_e_preserva_a_questao_original(self):
        q = q_soma(res="Correto - 35 + 18 = 53")
        antes = copy.deepcopy(q)
        reg = mr.processar_item(ITEM, q, chamador(Fila([resp(PROF_OK), resp(AVAL_OK)])))
        self.assertEqual(reg["decisao"], "aprovada")
        self.assertEqual(q, antes)
        self.assertEqual(reg["resolucao_original"], "Correto - 35 + 18 = 53")
        self.assertEqual(reg["gabarito"], "C")
        self.assertEqual(reg["hash_questao"], pb.hash_questao(q))

    def test_professor_que_discorda_vai_para_humano_sem_trocar_gabarito(self):
        q = q_soma(res="Correto - 35 + 18 = 53")
        r = json.dumps({"discorda": True, "motivo": "acho que é D", "resolucao": ""})
        reg = mr.processar_item(ITEM, q, chamador(Fila([resp(r)])))
        self.assertEqual(reg["decisao"], "revisao_humana")
        self.assertEqual(q["resposta_correta"], "C")
        self.assertIsNone(reg["resolucao_nova"])

    def test_falha_de_api_nunca_aprova(self):
        q = q_soma(res="Correto - 35 + 18 = 53")
        for fila in ([RuntimeError("503")], [resp(PROF_OK), RuntimeError("timeout")],
                     [resp(PROF_OK), resp(AVAL_OK, fim="length")], [resp(PROF_OK), resp("{lixo")]):
            reg = mr.processar_item(ITEM, q, chamador(Fila(fila)))
            self.assertEqual(reg["decisao"], "abstencao", fila)

    def test_avaliador_que_reprova_rejeita(self):
        q = q_soma(res="Correto - 35 + 18 = 53")
        ruim = json.dumps({"resolucao_correta": False, "usa_dado_ausente": False, "justificativa_falsa": False,
                           "chega_na_alternativa": True, "problemas": [{"codigo": "resolucao_errada", "detalhe": "x"}],
                           "confianca": 0.8})
        reg = mr.processar_item(ITEM, q, chamador(Fila([resp(PROF_OK), resp(ruim)])))
        self.assertEqual(reg["decisao"], "rejeitada_avaliador")

    def test_dry_run_nao_gasta(self):
        q = q_soma(res="Correto - 35 + 18 = 53")
        ch = chamador(mr.ClienteSimuladoResolucao({}), teto=None, simulado=True)
        mr.processar_item(ITEM, q, ch)
        self.assertEqual(ch.gasto_brl, 0.0)


# ---------------------------------------------------------------- plano de injeção
class TestPlanoInjecao(unittest.TestCase):
    def test_classificador_em_outros_nao_vira_lacuna(self):
        lac = {"deficit": 6, "meta": 30}
        hp = {"subtemas_taxonomia": 2, "subtemas_vazios": ["a", "b"], "subtema_dominante_pct": 100,
              "diversidade": {"entropia_raciocinio": 0.7, "pct_outros": 100.0}}
        score, comp = pi.prioridade(lac, hp)
        self.assertEqual(comp["subtemas_vazios"], 0.0)
        self.assertTrue(comp["verificar_classificador"])
        hp2 = dict(hp, diversidade={"entropia_raciocinio": 0.7, "pct_outros": 0.0})
        self.assertGreater(pi.prioridade(lac, hp2)[0], score)

    def test_nivel_de_evidencia_mais_alto_para_conceitual_e_geometria(self):
        n = pi.nivel_evidencia("9º", "H17", {"subtema": "triangulo", "dificuldade": "Difícil"}, {"conceitual": True})
        for exigido in ("validador_cego", "revisor_cego", "segundo_revisor_gemini", "verificador_geometria_ok"):
            self.assertIn(exigido, n)
        # D3 suspensa (2026-10-01): a dificuldade julgada pelo revisor não decide a entrada
        self.assertNotIn("dificuldade_real_do_revisor_igual_a_pedida", n)


# ---------------------------------------------------------------- prompt do gerador v6
class TestGeradorV6(unittest.TestCase):
    def test_sem_representacao_nem_restricoes_o_prompt_nao_muda(self):
        import agentes_questoes as aq
        self.assertEqual(aq.addendum_v6(None), "")
        self.assertEqual(aq.addendum_v6({"subtema": "x"}), "")

    def test_addendum_leva_representacao_e_restricoes_so_ao_professor(self):
        import agentes_questoes as aq
        import distill_teacher as dt
        import diversidade as dv
        base = dv.planejar_lote("5º", "H01", 1, "Fácil", seed=1)[0]
        slot = dict(base, representacao="verbal", restricoes=["r1", "r2"])
        t = aq.addendum_v6(slot)
        self.assertIn("em palavras", t)
        self.assertIn("r1; r2", t)
        # o prompt de TREINO (contrato do app) não vê nada disso
        self.assertNotIn("Representação pedida", dt.prompt_usuario("5º", "H01", "d", "Fácil", slot))

    def test_versao_propria_e_versao_dos_juizes_intacta(self):
        import agentes_questoes as aq
        self.assertEqual(aq.VERSAO_PROMPTS, "3358bf312d22")
        self.assertEqual(len(aq.VERSAO_GERADOR_V6), 12)

    def test_enriquecer_slots_nao_muda_subtema_tipo_contexto(self):
        import diversidade as dv
        import injetar_questoes as iq
        tax = dv.carregar_taxonomia()
        lac = {"ano": "5º", "habilidade": "H01", "descricao": "d", "permitidas": ["Fácil"], "falta_dif": {"Fácil": 2},
               "falta_sub": {}}
        slots = iq.planejar_slots(lac, 2, {}, [], tax, seed=1)
        novos = iq.enriquecer_slots_v6(lac, slots, [], tax)
        for (d0, s0), (d1, s1) in zip(slots, novos):
            self.assertEqual(d0, d1)
            for k in ("subtema", "tipo_raciocinio", "contexto", "estrutura"):
                self.assertEqual(s0[k], s1[k])
            self.assertTrue(s1["restricoes"] and s1["representacao"])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------- candidata v6 (usa os arquivos gerados; pula se ausentes)
class TestCandidataV6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import montar_candidata_v6 as mc
        cls.mc = mc
        if not (mc.V6.exists() and mc.V5.exists() and mc.PROVENIENCIA.exists()):
            raise unittest.SkipTest("candidata v6 não gerada neste checkout")

    def test_nao_escreve_em_arquivo_protegido(self):
        for nome in ("train_curado_v5.jsonl", "train.jsonl", "val.jsonl"):
            with self.assertRaises(SystemExit):
                self.mc.montar(saida=ROOT / "data" / nome, sem_gravar=True)

    def test_toda_questao_real_da_v5_esta_na_v6_com_mesmo_gabarito_e_enunciado(self):
        mc = self.mc
        v6 = {pb.hash_questao(q): q for e in mc._linhas(mc.V6) for q in mc._qs(e)}
        for e in mc._linhas(mc.V5):
            if pb.origem_de(e["meta"]).startswith("real"):
                for q in mc._qs(e):
                    self.assertIn(pb.hash_questao(q), v6, e["meta"]["codigo_item"])
                    q6 = v6[pb.hash_questao(q)]
                    self.assertEqual(q6["alternativas"][q6["resposta_correta"]], q["alternativas"][q["resposta_correta"]])

    def test_so_a_resolucao_muda_nas_aprimoradas_e_a_original_esta_na_proveniencia(self):
        mc = self.mc
        prov = [r for r in mc._linhas(mc.PROVENIENCIA) if r["operacao"] == "resolucao_aprimorada"]
        self.assertTrue(prov)
        v6 = {pb.hash_questao(q): q for e in mc._linhas(mc.V6) for q in mc._qs(e)}
        for r in prov[:50]:
            self.assertTrue(r["resolucao_original"] and r["resolucao_nova"])
            self.assertEqual(v6[r["hash"]]["resolucao_passo_a_passo"], r["resolucao_nova"])

    def test_conceitual_sem_segundo_revisor_nao_entra(self):
        mc = self.mc
        for e in mc._linhas(mc.V6):
            if e["meta"].get("origem_v6") == "injecao_v6" and (e["meta"]["ano"], e["meta"]["habilidade"]) in mc.CONCEITUAIS:
                self.assertIs(e["meta"]["vereditos"]["revisor2"], True)
