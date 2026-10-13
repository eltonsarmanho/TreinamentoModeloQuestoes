"""Curadoria item a item (src/curar_base_v7.py): decisões, política de itens reais, sem rede."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import curar_base_v7 as cb  # noqa: E402
import sabia_v6 as sv  # noqa: E402

LETRAS = "ABCDE"


def q_base(gab="B", resolucao="Correto - 35 + 18 = 53"):
    return {"enunciado": "Pedro tinha 35 figurinhas e ganhou 18. Quantas tem agora?",
            "alternativas": {"A": "17", "B": "53", "C": "43", "D": "413", "E": "52"},
            "resolucao_passo_a_passo": resolucao, "resposta_correta": gab, "difficulty": "EASY"}


def item(origem="injetado"):
    return {"linha": 0, "questao": 0, "codigo_item": "T-1", "ano": "5º", "habilidade": "H05", "origem": origem,
            "dificuldade": "Fácil"}


def resp_curador(**kw):
    base = {"dados": [], "premissa_possivel": True, "resolucao_propria": "x",
            "alternativas": {"A": "F", "B": "V", "C": "F", "D": "F", "E": "F"}, "resposta_calculada": "B",
            "problemas": [], "nivel": "Fácil", "veredito": "aprovado", "resolucao_aluno": None, "item_corrigido": None,
            "confianca": 0.9}
    base.update(kw)
    return json.dumps(base, ensure_ascii=False)


def resp_validador(letra="B", status=None, problemas=None, eq=None):
    st = status or {L: ("V" if L == letra else "F") for L in LETRAS}
    return json.dumps({"dados": [], "premissa_possivel": True, "pergunta": "", "resolucao": "x", "alternativas": st,
                       "motivos": {L: "" for L in LETRAS}, "equivalentes": eq or [], "dependencia_visual": False,
                       "resposta_calculada": letra, "problemas": problemas or [], "confianca": 0.9})


class Roteiro:
    """Cliente que devolve respostas prontas, na ordem."""

    def __init__(self, *textos):
        self.textos = list(textos)

    def chat_completion(self, messages, max_tokens, temperature):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.textos.pop(0)),
                                                        finish_reason="stop")], usage=None)


def chamador(cur, val=None):
    papeis = cb.papeis_padrao("sabia-4", "sabia-4")
    clientes = {"curador": Roteiro(*cur), "validador_correcao": Roteiro(*(val or []))}
    return sv.ChamadorSabia(papeis, clientes, 10, None, sv.carregar_precos(), log_uso=None, cache=None, simulado=True)


class TestParse(unittest.TestCase):
    def test_aprovado_valido(self):
        self.assertEqual(cb.parse_curador(resp_curador())["veredito"], "aprovado")

    def test_malformado_e_none(self):
        self.assertIsNone(cb.parse_curador("lixo"))
        self.assertIsNone(cb.parse_curador(resp_curador(nivel="Impossível")))
        self.assertIsNone(cb.parse_curador(resp_curador(veredito="talvez")))

    def test_corrigivel_sem_item_corrigido_e_none(self):
        self.assertIsNone(cb.parse_curador(resp_curador(veredito="corrigivel", item_corrigido=None)))

    def test_alternativas_repetidas_na_correcao_sao_none(self):
        corr = {"enunciado": "e", "alternativas": {L: "1" for L in LETRAS}, "resposta_correta": "A",
                "resolucao_passo_a_passo": "r", "motivo": "m"}
        self.assertIsNone(cb.parse_curador(resp_curador(veredito="corrigivel", item_corrigido=corr)))


class TestDecisoes(unittest.TestCase):
    def test_confirmada_quando_concorda_e_unica(self):
        reg = cb.processar_item(item(), q_base(), chamador([resp_curador()]))
        self.assertEqual(reg["decisao"], "confirmada")
        self.assertEqual(reg["nivel_estimado"], "Fácil")

    def test_discorda_do_gabarito_nunca_troca(self):
        c = resp_curador(alternativas={"A": "F", "B": "F", "C": "V", "D": "F", "E": "F"}, resposta_calculada="C")
        reg = cb.processar_item(item(), q_base(gab="B"), chamador([c]))
        self.assertEqual(reg["decisao"], "gabarito_discorda")
        self.assertEqual(reg["gabarito"], "B")

    def test_aprovado_com_duas_verdadeiras_nao_e_confirmada(self):
        c = resp_curador(alternativas={"A": "V", "B": "V", "C": "F", "D": "F", "E": "F"}, resposta_calculada=None)
        reg = cb.processar_item(item(), q_base(), chamador([c]))
        self.assertEqual(reg["decisao"], "gabarito_discorda")

    def test_aprovado_com_premissa_impossivel_e_discorda_com_motivo_claro(self):
        c = resp_curador(premissa_possivel=False)  # aprovado + letra certa, mas a premissa não existe
        reg = cb.processar_item(item(), q_base(), chamador([c]))
        self.assertEqual(reg["decisao"], "gabarito_discorda")
        self.assertEqual(reg["motivo"], "premissa_impossivel_apesar_de_aprovado")

    def test_inviavel(self):
        c = resp_curador(veredito="inviavel", resposta_calculada=None, problemas=[{"codigo": "premissa_impossivel", "detalhe": "x"}])
        reg = cb.processar_item(item(), q_base(), chamador([c]))
        self.assertEqual(reg["decisao"], "inviavel")
        self.assertIn("premissa_impossivel", reg["motivo"])

    def test_resposta_truncada_ou_invalida_e_abstencao(self):
        reg = cb.processar_item(item(), q_base(), chamador(["isto não é json"]))
        self.assertEqual(reg["decisao"], "abstencao")

    def test_corrigida_exige_validador_cego_confirmar(self):
        corr = {"enunciado": "Pedro tinha 35 figurinhas e ganhou 18. Quantas figurinhas ele tem agora?",
                "alternativas": {"A": "17", "B": "53", "C": "43", "D": "413", "E": "62"},
                "resposta_correta": "B", "resolucao_passo_a_passo": "Pedro tinha 35 e ganhou 18. Somando: 35 + 18 = 53. Pedro tem 53 figurinhas.",
                "motivo": "E igual a outra"}
        c = resp_curador(veredito="corrigivel", item_corrigido=corr, resposta_calculada=None,
                         problemas=[{"codigo": "distrator_verdadeiro", "detalhe": "E"}])
        ok = cb.processar_item(item(), q_base(), chamador([c], [resp_validador("B")]))
        self.assertEqual(ok["decisao"], "corrigida")
        self.assertEqual(ok["mudanca"], "enunciado")
        ruim = cb.processar_item(item(), q_base(), chamador([c], [resp_validador("C")]))
        self.assertEqual(ruim["decisao"], "correcao_rejeitada")

    def test_correcao_que_cita_letra_e_rejeitada(self):
        corr = {"enunciado": "Pedro tinha 35 figurinhas e ganhou 18. Quantas figurinhas ele tem agora?",
                "alternativas": {"A": "17", "B": "53", "C": "43", "D": "413", "E": "62"}, "resposta_correta": "B",
                "resolucao_passo_a_passo": "Somando 35 + 18 = 53, a alternativa B está correta.", "motivo": "m"}
        c = resp_curador(veredito="corrigivel", item_corrigido=corr, resposta_calculada=None)
        reg = cb.processar_item(item(), q_base(), chamador([c], [resp_validador("B")]))
        self.assertEqual(reg["decisao"], "correcao_rejeitada")
        self.assertIn("resolucao_cita_letra", reg["motivo"])

    def test_classe_mudanca(self):
        q = q_base()
        mesma = {"enunciado": q["enunciado"], "alternativas": dict(q["alternativas"], E="62"), "resposta_correta": "B"}
        self.assertEqual(cb.classe_mudanca(q, mesma), "so_distratores")
        outro = {"enunciado": q["enunciado"], "alternativas": dict(q["alternativas"], B="54"), "resposta_correta": "B"}
        self.assertEqual(cb.classe_mudanca(q, outro), "gabarito_alterado")


def exemplo(q, codigo, dif="Fácil"):
    return {"meta": {"codigo_item": codigo, "ano": "5º", "habilidade": "H05", "dificuldade": dif},
            "messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": f"Gere 1 questão(ões). Ano: 5º ano. Dificuldade: {dif}."},
                         {"role": "assistant", "content": json.dumps({"questoes": [q]}, ensure_ascii=False)}]}


class TestMontagem(unittest.TestCase):
    def _montar(self, codigo, decisao, extra=None, aplicar_reais=False, aplicar_dificuldade=False):
        q = q_base()
        h = cb.hash_questao(q)
        reg = {"hash_questao": h, "decisao": decisao, "origem": "x", "modelos": {}, "versoes_prompt": {},
               "nivel_estimado": "Difícil", "motivo": "m", "problemas": []}
        reg.update(extra or {})
        with tempfile.TemporaryDirectory() as d:
            base, cur, out = Path(d) / "b.jsonl", Path(d) / "c.jsonl", Path(d) / "v7_teste.jsonl"
            base.write_text(json.dumps(exemplo(q, codigo)) + "\n", encoding="utf-8")
            cur.write_text(json.dumps(reg) + "\n", encoding="utf-8")
            old = cb.SAIDA_DIR, cb.PROPOSTAS_REAIS
            cb.SAIDA_DIR, cb.PROPOSTAS_REAIS = Path(d), Path(d) / "p.jsonl"
            try:
                resumo = cb.montar(base, cur, out, aplicar_reais, aplicar_dificuldade)
                saida = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
                props = cb.PROPOSTAS_REAIS.read_text(encoding="utf-8").splitlines()
            finally:
                cb.SAIDA_DIR, cb.PROPOSTAS_REAIS = old
        return resumo, saida, props

    def test_inviavel_nao_real_e_removido(self):
        resumo, saida, _ = self._montar("INJ-1-H05-F-1", "inviavel")
        self.assertEqual(saida, [])
        self.assertEqual(resumo["operacoes"], {"quarentena": 1})

    def test_inviavel_real_vira_proposta_e_item_fica(self):
        resumo, saida, props = self._montar("MT5001FH05MT", "inviavel")
        self.assertEqual(len(saida), 1)
        self.assertEqual(len(props), 1)
        self.assertEqual(resumo["operacoes"], {"proposta_real_pendente": 1})

    def test_real_com_aplicar_reais_e_removido(self):
        _r, saida, _p = self._montar("MT5001FH05MT", "inviavel", aplicar_reais=True)
        self.assertEqual(saida, [])

    def test_corrigida_nao_real_troca_e_mantem_o_gabarito_value(self):
        corr = {"enunciado": "Pedro tinha 35 figurinhas e ganhou 18. Quantas figurinhas tem?",
                "alternativas": {"A": "17", "B": "53", "C": "43", "D": "413", "E": "62"}, "resposta_correta": "B",
                "resolucao_passo_a_passo": "x", "motivo": "m"}
        _r, saida, _p = self._montar("INJ-1-H05-F-1", "corrigida", {"item_corrigido": corr, "mudanca": "enunciado"})
        q = json.loads(saida[0]["messages"][2]["content"])["questoes"][0]
        self.assertEqual(q["alternativas"]["E"], "62")
        self.assertEqual(q["alternativas"][q["resposta_correta"]], "53")

    def test_nivel_so_muda_com_a_flag(self):
        _r, saida, _p = self._montar("INJ-1-H05-F-1", "confirmada")
        self.assertEqual(saida[0]["meta"]["dificuldade"], "Fácil")
        _r, saida, _p = self._montar("INJ-1-H05-F-1", "confirmada", aplicar_dificuldade=True)
        self.assertEqual(saida[0]["meta"]["dificuldade"], "Difícil")
        self.assertIn("Dificuldade: Difícil.", saida[0]["messages"][1]["content"])
        self.assertEqual(json.loads(saida[0]["messages"][2]["content"])["questoes"][0]["difficulty"], "HARD")

    def test_nao_escreve_em_arquivo_protegido(self):
        with self.assertRaises(SystemExit):
            cb.montar(ROOT / "data" / "train_curado_v6.jsonl", cb.CURADORIA, ROOT / "data" / "train_curado_v5.jsonl")


if __name__ == "__main__":
    unittest.main()
