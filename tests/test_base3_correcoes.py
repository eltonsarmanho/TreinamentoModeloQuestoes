"""Testes das correções da revisão do passo 2 (2026-10-01).

1. D6 — alternativas corrompidas pela planilha: 17 itens reais da base (e 3 de
   cada val) têm frações que o Excel converteu em data ("2025-06-03 00:00:00"
   no lugar de 3/6). O gabarito é uma data; nenhum filtro via isso.
2. Referência de dificuldade só com itens bons: os 5 que o revisor reprovou
   na r7 eram defeitos de verdade (3 com datas, MT9036DH12TD e MT9009DH03TD
   com gabarito errado), e a métrica os contava como "falsa rejeição".
3. H2 — precisão pedida no singular ("uma casa decimal"), valor EXATO pedido
   e aproximação fora da regra com o valor exato entre as alternativas.

Sem rede: tudo com dados montados à mão (e a base real só para leitura).
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import agentes_questoes as aq  # noqa: E402
import auditar_base as ab  # noqa: E402
import calibrar_agentes as ca  # noqa: E402
import test_agentes_questoes as T  # noqa: E402  (só helpers)
from test_passo2_dificuldade import _linha, _row  # noqa: E402

# MT90123MH25MT, literal da base (idx 259): probabilidade de par no dado, 3/6
Q_DATAS = {"enunciado": "Um dado comum de 6 faces será lançado uma vez. Qual é a probabilidade de sair um número par?",
           "alternativas": {"A": "2025-06-03 00:00:00", "B": "2025-06-02 00:00:00", "C": "2025-06-04 00:00:00",
                            "D": "2025-06-01 00:00:00", "E": "Nenhuma das alternativas anteriores"},
           "resolucao_passo_a_passo": "Números pares no dado: 2, 4, 6 → 3 resultados possíveis. Total: 6 → 3/6",
           "resposta_correta": "A", "difficulty": "MEDIUM"}


def q_dec(enunciado, alts, gab="A"):
    return {"enunciado": enunciado, "alternativas": dict(zip("ABCDE", alts)), "resolucao_passo_a_passo": "conta.",
            "resposta_correta": gab, "difficulty": "MEDIUM"}


class TestD6AlternativaCorrompidaPlanilha(unittest.TestCase):
    def test_detecta_data_e_propoe_a_fracao(self):
        c = aq.alternativas_corrompidas_planilha(Q_DATAS)
        self.assertEqual(sorted(c), list("ABCD"))
        # dia/mês: "3/6" virou 3 de junho -> 2025-06-03
        self.assertEqual(c["A"], "3/6")
        self.assertEqual(c["D"], "1/6")
        # alternativas normais (inclusive datas brasileiras e frações) não disparam
        normal = q_dec("Qual fração?", ["3/6", "12/03/2024", "2025", "R$ 2.025,00", "Nenhuma das alternativas"])
        self.assertEqual(aq.alternativas_corrompidas_planilha(normal), {})

    def test_filtro_da_auditoria_reprova(self):
        f = aq.filtros_auditoria(Q_DATAS)
        self.assertIn("alternativa_corrompida_planilha", f["reprovado"])
        self.assertEqual(sorted(f["planilha"]), list("ABCD"))
        self.assertNotIn("alternativa_corrompida_planilha", aq.FILTROS_MATEMATICOS)

    def test_injecao_e_injetada_barram(self):
        ex = T.exemplo("9º", "H25", Q_DATAS, "INJ-9-H25-M-00001", dificuldade="Moderado",
                       versao_prompts=aq.VERSAO_PROMPTS)
        self.assertEqual(ab.injetada_valida(ex), (False, "alternativa_corrompida_planilha"))

    def test_montar_remove_com_proposta_e_decisao_humana_vence(self):
        amb = T.Ambiente()
        try:
            exs = [T.exemplo("9º", "H25", Q_DATAS, "MT90123MH25MT", dificuldade="Moderado"),
                   T.exemplo("9º", "H99", T.questao_boa(), "MT9999")]
            amb.train.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in exs), encoding="utf-8")
            rel = T.silencioso(ab.montar, amb.args_auditoria("--montar"))
            cods = [e["meta"]["codigo_item"] for e in amb.linhas("v3.jsonl")]
            self.assertEqual(cods, ["MT9999"])
            rem = amb.linhas("removidos_v3.jsonl")
            self.assertEqual([(r["codigo_item"], r["motivo"]) for r in rem],
                             [("MT90123MH25MT", "alternativa_corrompida_planilha")])
            ev = rem[0]["evidencia"]
            self.assertEqual(ev["alternativas"]["A"], {"atual": "2025-06-03 00:00:00", "proposta": "3/6"})
            self.assertEqual(rel["n_removidos_planilha"], 1)
            # o item corrompido não precisa de árbitro: não é candidato da D2
            self.assertEqual(ab.candidatos_d2(amb.args_auditoria("--arbitrar")), [])
            # decisão humana tem precedência (ex.: o usuário restaura e manda manter)
            Path(amb.d / "decisoes_humanas.json").write_text(json.dumps({"versao": 1, "decisoes": [{
                "codigo_item": "MT90123MH25MT", "idx": 0, "acao": "manter", "data": "2026-10-01",
                "autor": "usuário", "motivo": "teste"}]}), encoding="utf-8")
            T.silencioso(ab.montar, amb.args_auditoria("--montar"))
            self.assertIn("MT90123MH25MT", [e["meta"]["codigo_item"] for e in amb.linhas("v3.jsonl")])
        finally:
            amb.fechar()

    def test_base_real_tem_exatamente_os_17(self):
        base = ROOT / "data" / "train_curado.jsonl"
        if not base.exists():
            self.skipTest("data/train_curado.jsonl ausente")
        achados = []
        for i, linha in enumerate(base.read_text(encoding="utf-8").splitlines()):
            ex = json.loads(linha)
            if any(aq.alternativas_corrompidas_planilha(q) for q in ab.questoes_do_exemplo(ex) or []):
                achados.append(ex["meta"]["codigo_item"])
        self.assertEqual(sorted(achados), sorted([
            "MT9031FH07MT", "MT9020MfH07TD", "MT90122MH25MT", "MT9023MH08TD", "MT9043MH09MT", "MT9040DH08MT",
            "MT9073FH25TD", "MT90124DH25MT", "MT9075DH25TD", "MT90123MH25MT", "MT90121FH25MT", "MT9035DH07MT",
            "MT9038MH08MT", "MT90125DH25MT", "MT9026MH09TD", "MT9033MH07MT", "MT9034DH07MT"]))


class TestReferenciaSoComItensBons(unittest.TestCase):
    def _rows(self):
        rows = [_row(f"MT5{i:03d}{L}H0{i % 3 + 1}MT", g, enunciado=f"Quanto é {i} + 7?")
                for i, (L, g) in enumerate([("F", "Fácil"), ("M", "Moderado"), ("D", "Difícil")] * 4)]
        return rows

    def test_defeito_sai_no_sorteio_sem_mudar_os_outros(self):
        rows = self._rows()
        antes, _ = ca.carregar_ref_dificuldade(n_por_nivel=2, rows=rows)
        alvo = next(it for it in antes if it["dificuldade"] == "Moderado")["codigo_item"]
        depois, info = ca.carregar_ref_dificuldade(n_por_nivel=2, rows=rows, defeitos={alvo: "gabarito errado"})
        ids_a, ids_d = [x["id"] for x in antes], [x["id"] for x in depois]
        self.assertNotIn(f"DIF-{alvo}", ids_d)
        self.assertEqual(len(ids_d), 6)
        # só o defeituoso foi trocado; o resto do sorteio é o mesmo
        self.assertEqual(set(ids_a) - set(ids_d), {f"DIF-{alvo}"})
        self.assertEqual(len(set(ids_d) - set(ids_a)), 1)
        self.assertEqual(info["descartados_defeito"], {alvo: "gabarito errado"})

    def test_data_da_planilha_e_descartada_automaticamente(self):
        rows = self._rows()
        antes, _ = ca.carregar_ref_dificuldade(n_por_nivel=2, rows=rows)
        alvo = next(it for it in antes if it["dificuldade"] == "Difícil")["codigo_item"]
        for r in rows:
            if r["codigo_item"] == alvo:
                r["alternativa_a"] = "2025-06-03 00:00:00"
        depois, info = ca.carregar_ref_dificuldade(n_por_nivel=2, rows=rows, defeitos={})
        self.assertNotIn(f"DIF-{alvo}", [x["id"] for x in depois])
        self.assertIn("planilha", info["descartados_defeito"][alvo])

    def test_defeitos_reais_da_r7_estao_na_lista(self):
        for cod in ("MT9036DH12TD", "MT9009DH03TD"):
            self.assertIn(cod, ca.DEFEITOS_REAIS_CONFIRMADOS)

    def test_metrica_so_com_a_selecao_e_rejeicao_de_defeito_e_acerto(self):
        linhas = [_linha("DIF-a", "Fácil", "Fácil"), _linha("DIF-b", "Moderado", "Fácil"),
                  _linha("DIF-c", "Difícil", "Fácil", veredito=False),        # defeito: fora da seleção
                  _linha("DIF-d", "Difícil", "Difícil", veredito=False)]
        sel = {"ids": ["DIF-a", "DIF-b", "DIF-d"], "descartados_defeito": {"c": "gabarito errado"}}
        for l in linhas:
            l["codigo_item"] = l["id"][4:]
        m = ca.metricas_dificuldade(linhas, selecao=sel)
        self.assertEqual(m["revisor"]["n"], 3)
        self.assertEqual(m["revisor"]["rebaixaria_para_facil"]["de"], 2)
        self.assertEqual(m["revisor_falsa_rejeicao_reais"]["ids"], ["DIF-d"])
        self.assertEqual(m["revisor_rejeicao_correta_defeitos"]["ids"], ["DIF-c"])
        self.assertEqual(m["fora_da_selecao"], ["DIF-c"])


class TestH2PrecisaoPedida(unittest.TestCase):
    ALTS_511 = ["0,45", "0,55", "0,11", "0,51", "Nenhuma das alternativas anteriores"]

    def test_uma_casa_decimal_no_singular_nao_aplica_a_regra_das_duas(self):
        for en in ("Qual é a representação decimal de 5/11, arredondada para uma casa decimal?",
                   "Representação decimal aproximada de 5/11 com uma casa decimal:",
                   "Qual é a representação decimal de 5/11 com 1 casa decimal?"):
            with self.subTest(en=en):
                self.assertNotEqual(aq.verificar_aritmetica(q_dec(en, self.ALTS_511))["status"], "confirma")

    def test_duas_casas_no_singular_de_casa_continua_valendo(self):
        en = "Qual é a representação decimal de 5/11 com duas casas decimais?"
        self.assertEqual(aq.verificar_aritmetica(q_dec(en, self.ALTS_511))["status"], "confirma")

    def test_valor_exato_pedido(self):
        en = "Qual é a representação decimal exata de 5/11?"
        # 0,45 não é o valor exato
        self.assertEqual(aq.verificar_aritmetica(q_dec(en, self.ALTS_511))["status"], "contradiz")
        # com a dízima entre as alternativas, ela é a única certa
        alts = ["0,4545...", "0,45", "0,55", "0,11", "Nenhuma das alternativas anteriores"]
        self.assertEqual(aq.verificar_aritmetica(q_dec(en, alts))["status"], "confirma")
        self.assertEqual(aq.verificar_aritmetica(q_dec(en, alts, gab="B"))["status"], "contradiz")
        # sem "exata", a regra H2 vale e as duas casam: não única (como antes)
        self.assertEqual(aq.verificar_aritmetica(q_dec("Qual é a representação decimal de 5/11?", alts))["status"],
                         "contradiz")

    def test_aproximacao_fora_da_regra_com_o_exato_presente(self):
        en38 = "Qual é a representação decimal de 3/8?"
        r = aq.verificar_aritmetica(q_dec(en38, ["0,38", "0,375", "0,83", "3,8", "Nenhuma das alternativas"]))
        self.assertEqual(r["status"], "contradiz")
        r = aq.verificar_aritmetica(q_dec(en38, ["0,38", "0,375", "0,83", "3,8", "Nenhuma das alternativas"],
                                          gab="B"))
        self.assertEqual(r["status"], "confirma")
        en72 = "Qual é a representação decimal de 7/2?"
        r = aq.verificar_aritmetica(q_dec(en72, ["4", "3,5", "2,7", "7,2", "Nenhuma das alternativas"]))
        self.assertEqual(r["status"], "contradiz")
        # sem o valor exato entre as alternativas, continua se abstendo
        en511 = "Qual é a representação decimal de 5/11?"
        r = aq.verificar_aritmetica(q_dec(en511, ["0,5", "0,45", "0,11", "0,51", "Nenhuma das alternativas"]))
        self.assertEqual(r["status"], "sem_veredito")

    def test_1348_continua_certo(self):
        base = ROOT / "data" / "train_curado.jsonl"
        if not base.exists():
            self.skipTest("data/train_curado.jsonl ausente")
        ex = json.loads(base.read_text(encoding="utf-8").splitlines()[1348])
        q = ab.questoes_do_exemplo(ex)[0]
        self.assertEqual(aq.verificar_aritmetica(q)["status"], "confirma")


class TestAuditarIdx(unittest.TestCase):
    def test_audita_so_os_idx_pedidos(self):
        amb = T.Ambiente()
        try:
            T.silencioso(ab.auditar, amb.args_auditoria("--idx", "2,0"), agentes=T.agentes_com(T.Roteiro(), 100))
            self.assertEqual(sorted(r["idx"] for r in amb.linhas("auditoria.jsonl")), [0, 2])
            with self.assertRaises(SystemExit):
                T.silencioso(ab.auditar, amb.args_auditoria("--idx", "99"), agentes=T.agentes_com(T.Roteiro(), 100))
        finally:
            amb.fechar()


class TestArbitroIncompleto(unittest.TestCase):
    def test_arbitro_que_ve_resposta_nao_unica_por_falta_de_dado_nao_e_discordancia(self):
        """1080: o Gemini também viu que não há resposta única, mas disse
        dados_insuficientes; isso não é 'discordar' do defeito."""
        ex = T.exemplo("9º", "H99", T.questao_boa(), "DIST-H99-Fácil-00001", destilado=True, professor="p")
        reg = T._registro(ex, 0, T._juiz(False, ["gabarito_sem_resposta_unica"]),
                          T._juiz(False, ["gabarito_errado"]))
        arb = {"resultados": [{"avaliado": True, "problemas": [
            {"codigo": "resposta_nao_unica", "detalhe": "x"}, {"codigo": "dados_insuficientes", "detalhe": "y"}]}]}
        acao, motivo = ab.decidir_com_arbitro("destilado", reg, arb)
        self.assertEqual(acao, "manter")
        self.assertEqual(motivo, "d2_arbitro_ve_item_incompleto_revisao_humana")


if __name__ == "__main__":
    unittest.main()
