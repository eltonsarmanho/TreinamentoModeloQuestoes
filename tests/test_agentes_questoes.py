"""Testes do pipeline da base de conhecimento (agentes, injeção, auditoria).

Tudo com CLIENTE SIMULADO/ROTEIRIZADO — nenhuma chamada de rede, nenhum custo.
Os arquivos de entrada (relatório, base, val, taxonomia) são mínimos e ficam
em diretório temporário: a taxonomia real está em edição pelo outro workflow
e os dados reais não podem ser escritos. Cobrem as regras que decidem o que
ENTRA na base: veredito duplo, filtros, fail-closed no parse, orçamento duro,
retomada, arquivos protegidos e chave da API fora de qualquer log.
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import agentes_questoes as aq  # noqa: E402
import arbitro_gemini as ag  # noqa: E402
import auditar_base as ab  # noqa: E402
import injetar_questoes as iq  # noqa: E402

LETRAS = "ABCDE"
PROTEGIDOS = [ROOT / "data" / n for n in ("train.jsonl", "train_curado.jsonl", "val.jsonl",
                                          "val_frozen_v1.jsonl", "val_novos_v1.jsonl")]


def _sha(p):
    return hashlib.sha1(p.read_bytes()).hexdigest() if p.exists() else None


SHA_INICIAL = {p: _sha(p) for p in PROTEGIDOS}


def questao_boa(a=35, b=18, gab="B", dif="EASY", texto=None):
    certo = a + b
    vals = {"A": a - b, "B": certo + 10, "C": certo - 1, "D": certo + 1, "E": certo + 2}
    vals[gab], vals["B" if gab != "B" else "A"] = certo, vals[gab]
    return {"enunciado": texto or f"Pedro tinha {a} figurinhas e ganhou {b} do primo na escola. "
                                   f"Para achar o total, fez {a} + {b}. Quantas figurinhas tem agora?",
            "alternativas": {L: f"{vals[L]} figurinhas" for L in LETRAS},
            "resolucao_passo_a_passo": f"{a} + {b} = {certo}.", "resposta_correta": gab, "difficulty": dif}


def papel_de(messages):
    s, u = messages[0]["content"], messages[-1]["content"]
    if "Regras de qualidade do SAEB/INEP" in s:
        return "gerador"
    if "Você NÃO conhece o gabarito" in s:
        return "validador"
    if "Sua análise concluiu" in u:
        return "validador_fase2"
    if "especialista em avaliação educacional" in s:
        return "revisor"
    return "?"


class Roteiro(aq.ClienteSimulado):
    """ClienteSimulado (resolve 'a + b') com sabotagens por papel e contagem."""

    def __init__(self, revisor_aprova=True, validador="ok", fase2="ok", excecao=None):
        super().__init__(erro_a_cada=0)
        self.revisor_aprova = revisor_aprova
        self.validador = validador
        self.fase2 = fase2
        self.excecao = excecao
        self.chamadas = Counter()

    def chat_completion(self, messages, max_tokens, temperature):
        papel = papel_de(messages)
        self.chamadas[papel] += 1
        if self.excecao is not None:
            raise self.excecao
        if papel == "validador" and self.validador == "malformado":
            return self._resp('{"alternativas": {"A": {"status": "V"', messages)
        if papel == "validador_fase2" and self.fase2 == "justificativa_falsa":
            return self._resp(json.dumps({"resolucao_correta": True, "usa_dado_ausente": False,
                                          "justificativa_falsa": True, "chega_na_alternativa": True,
                                          "problemas": [], "confianca": 0.8}), messages)
        resp = super().chat_completion(messages, max_tokens, temperature)
        if papel == "revisor" and not self.revisor_aprova:
            o = aq.extrair_json(resp.choices[0].message.content, ("criterios",), estrito=True)
            o["criterios"]["C7"] = {"ok": False, "nota": "contexto forçado"}
            o["problemas"] = [{"codigo": "contexto_forcado", "detalhe": "x"}]
            o["veredito"] = False
            return self._resp(json.dumps(o, ensure_ascii=False), messages)
        return resp

    def total(self):
        return sum(self.chamadas.values())


class Fila:
    """Cliente que devolve textos fixos em ordem (para o parse/decisão)."""

    def __init__(self, textos):
        self.textos = list(textos)
        self.n = 0

    def chat_completion(self, messages, max_tokens, temperature):
        self.n += 1
        if not self.textos:
            raise AssertionError("chamada inesperada")
        t = self.textos.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=t))])


def agentes_com(cliente, max_chamadas=None, log_uso=None, tentativas=3):
    return aq.Agentes({p: cliente for p in aq.PAPEIS}, orcamento=aq.Orcamento(max_chamadas),
                      log_uso=log_uso, tentativas=tentativas, dormir=lambda s: None)


def arbitro_com(cliente, max_chamadas=10):
    """Árbitro (H4) com cliente simulado: sem rede, sem log e sem cache reais."""
    return ag.Arbitro(cliente, "simulado:gemini", aq.Orcamento(max_chamadas), log_uso=None, cache_path=None,
                      dormir=lambda s: None)


def fase1(questao, status_por_original, **extra):
    """JSON da fase 1 nas letras EXIBIDAS (aplica a mesma permutação do código)."""
    perm = aq.permutacao(questao)
    st = {L: status_por_original[perm[L]] for L in LETRAS}
    unica = aq.resposta_unica(st)
    o = {"dados": [], "premissa_possivel": True, "pergunta": "p", "resolucao": "r",
         "alternativas": {L: {"status": st[L], "motivo": "m"} for L in LETRAS}, "equivalentes": [],
         "dependencia_visual": False, "resposta_calculada": unica, "problemas": [], "confianca": 0.9}
    o.update(extra)
    return json.dumps(o, ensure_ascii=False)


FASE2_OK = json.dumps({"resolucao_correta": True, "usa_dado_ausente": False, "justificativa_falsa": False,
                       "chega_na_alternativa": True, "problemas": [], "confianca": 0.8})


def revisor_json(status, veredito=True, falha=None, problemas=(), questao=None, dificuldade_real="Moderado"):
    """JSON do revisor CEGO. `status` vem nas letras ORIGINAIS; com `questao`,
    é convertido para as letras EXIBIDAS (permutação própria do revisor)."""
    if questao is not None:
        perm = aq.permutacao(questao, sal="revisor")
        status = {L: status[perm[L]] for L in LETRAS}
    crit = {f"C{i}": {"ok": True, "nota": "ok"} for i in range(1, 10)}
    if falha:
        crit[falha] = {"ok": False, "nota": "x"}
    return json.dumps({"resolucao_propria": "r", "alternativas": status,
                       "resposta_calculada": aq.resposta_unica(status), "dificuldade_real": dificuldade_real,
                       "criterios": crit, "problemas": list(problemas), "sugestoes": [], "veredito": veredito,
                       "confianca": 0.7}, ensure_ascii=False)


def so_v(letra):
    return {L: ("V" if L == letra else "F") for L in LETRAS}


# ---------------------------------------------------------------------------
# Ambiente temporário de injeção/auditoria
# ---------------------------------------------------------------------------

def exemplo(ano, hab, q, codigo, dificuldade="Fácil", **meta):
    from extract_data import SYSTEM_PROMPT
    import distill_teacher as dt
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": dt.prompt_usuario(ano, hab, "Resolver problemas de adição.",
                                                                     dificuldade, None)},
                         {"role": "assistant", "content": json.dumps({"questoes": [q]}, ensure_ascii=False)}],
            "meta": dict({"codigo_item": codigo, "ano": ano, "habilidade": hab, "dificuldade": dificuldade}, **meta)}


class Ambiente:
    def __init__(self, falta_facil=2):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.d = d
        self.relatorio = d / "relatorio.json"
        self.relatorio.write_text(json.dumps({"habilidades": [{
            "ano": "9º", "habilidade": "H99", "descricao": "Resolver problemas de adição.", "meta": 30,
            "n_efetivas": 30 - falta_facil, "falta_por_dificuldade": {"Fácil": falta_facil, "Moderado": 0,
                                                                      "Difícil": 0},
            "falta_por_subtema": {}, "banco": {"graus": ["Fácil", "Moderado"]}}]}), encoding="utf-8")
        q_real = questao_boa(10, 5, "C", texto="Uma turma tem 10 meninas e 5 meninos. Para contar, "
                                               "a professora fez 10 + 5. Quantos alunos há na turma?")
        q_dist = questao_boa(20, 7, "A", texto="A padaria assou 20 pães de manhã e 7 à tarde; o padeiro "
                                               "fez 20 + 7. Quantos pães assou no dia?")
        q_dist_ruim = dict(questao_boa(30, 9, "D", texto="Um ônibus levava 30 pessoas e subiram 9; o "
                                                         "motorista fez 30 + 9. Quantas pessoas há?"),
                           resposta_correta="E")
        q_sint = questao_boa(11, 4, "B", texto="Calcule 11 + 4 usando a reta numérica da sala. Qual é o "
                                               "resultado da adição?")
        self.train = d / "train.jsonl"
        linhas = [exemplo("9º", "H99", q_real, "MT9999"),
                  exemplo("9º", "H99", q_dist, "DIST-H99-Fácil-00001", destilado=True, professor="p"),
                  exemplo("9º", "H99", q_dist_ruim, "DIST-H99-Fácil-00002", destilado=True, professor="p"),
                  exemplo("9º", "H99", q_sint, "SINT-H99-Fácil-0001", sintetico=True)]
        self.train.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in linhas), encoding="utf-8")
        self.val = d / "val_tmp.jsonl"
        self.val.write_text(json.dumps(exemplo("9º", "H99", questao_boa(40, 2, "A", texto=(
            "Na biblioteca havia 40 livros e chegaram 2 doações; a bibliotecária fez 40 + 2. "
            "Quantos livros há?")), "MT9998"), ensure_ascii=False) + "\n", encoding="utf-8")
        self.tax = d / "tax.json"
        self.tax.write_text(json.dumps({"habilidades": {}}), encoding="utf-8")

    def args_injecao(self, *extra):
        d = self.d
        return iq.construir_parser().parse_args([
            "--relatorio", str(self.relatorio), "--train", str(self.train), "--val", str(self.val),
            "--taxonomia", str(self.tax), "--saida", str(d / "injecao_saeb.jsonl"),
            "--rejeitadas", str(d / "rejeitadas.jsonl"), "--resumo", str(d / "resumo.json"),
            "--log-uso", str(d / "uso.jsonl"), "--max-chamadas", "100", *extra])

    def args_auditoria(self, *extra):
        d = self.d
        return ab.construir_parser().parse_args([
            "--train", str(self.train), "--val", str(self.val), "--auditoria", str(d / "auditoria.jsonl"),
            "--injecao", str(d / "injecao_saeb.jsonl"), "--saida-v3", str(d / "v3.jsonl"),
            "--revisao-humana", str(d / "humana.jsonl"), "--montagem", str(d / "montagem.json"),
            "--resumo-json", str(d / "aud_resumo.json"), "--calibracao-json", str(d / "calib.json"),
            "--log-uso", str(d / "uso.jsonl"), "--removidos", str(d / "removidos_v3.jsonl"),
            "--rotulos", str(d / "rotulos_corrigidos_v3.jsonl"),
            "--resolucao-vazia", str(d / "resolucao_vazia_v3.jsonl"), "--max-chamadas", "100",
            # decisões humanas e arbitragem do TESTE: nunca os arquivos reais
            "--decisoes-humanas", str(d / "decisoes_humanas.json"), "--arbitragem", str(d / "arbitragem_d2.jsonl"),
            "--log-uso-arbitro", str(d / "uso_arbitro.jsonl"), *extra])

    def linhas(self, nome):
        p = self.d / nome
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []

    def fechar(self):
        self.tmp.cleanup()


def silencioso(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ===========================================================================
class TestVereditoDuplo(unittest.TestCase):
    def setUp(self):
        self.amb = Ambiente(falta_facil=2)

    def tearDown(self):
        self.amb.fechar()

    def _rodar(self, cliente, *extra):
        ag = agentes_com(cliente, max_chamadas=100, log_uso=self.amb.d / "uso.jsonl")
        return silencioso(iq.injetar, self.amb.args_injecao(*extra), agentes=ag), ag

    def test_ambos_true_entra(self):
        cli = Roteiro()
        res, _ = self._rodar(cli)
        aceitas = self.amb.linhas("injecao_saeb.jsonl")
        self.assertEqual(len(aceitas), 2)
        m = aceitas[0]["meta"]
        self.assertEqual(m["origem"], "injecao_saeb")
        self.assertEqual(m["vereditos"], {"validador": True, "revisor": True})
        self.assertEqual(m["confianca"]["nivel"], "alta")
        self.assertTrue(m["destilado"])
        self.assertEqual(m["versao_prompts"], aq.VERSAO_PROMPTS)
        self.assertTrue(m["codigo_item"].startswith("INJ-9-H99-F-"))
        self.assertIn("data", m)
        self.assertEqual(res["funil"]["aceitos"], 2)

    def test_prompt_de_treino_igual_ao_formato_de_inferencia(self):
        import distill_teacher as dt
        from extract_data import SYSTEM_PROMPT
        self._rodar(Roteiro())
        ex = self.amb.linhas("injecao_saeb.jsonl")[0]
        self.assertEqual(ex["messages"][0]["content"], SYSTEM_PROMPT)
        esperado = dt.prompt_usuario("9º", "H99", "Resolver problemas de adição.", "Fácil", None)
        self.assertEqual(ex["messages"][1]["content"], esperado)
        # nada do que é só do professor vaza para o treino
        self.assertNotIn("letra", ex["messages"][1]["content"])
        self.assertNotIn("EXATAMENTE a habilidade", ex["messages"][1]["content"])

    def test_revisor_reprova_nao_entra(self):
        cli = Roteiro(revisor_aprova=False)
        res, _ = self._rodar(cli, "--tentativas-por-slot", "1")
        self.assertEqual(self.amb.linhas("injecao_saeb.jsonl"), [])
        rej = self.amb.linhas("rejeitadas.jsonl")
        self.assertTrue(rej and all(r["etapa"] == "revisor" for r in rej))
        self.assertEqual(rej[0]["motivo"], "contexto_forcado")
        self.assertTrue(rej[0]["validador"]["veredito"])

    def test_validador_reprova_revisor_nem_e_chamado(self):
        cli = Roteiro(fase2="justificativa_falsa")
        self._rodar(cli, "--tentativas-por-slot", "1")
        self.assertEqual(self.amb.linhas("injecao_saeb.jsonl"), [])
        self.assertEqual(cli.chamadas["revisor"], 0)
        rej = self.amb.linhas("rejeitadas.jsonl")
        self.assertTrue(all(r["etapa"] == "validador" for r in rej))
        self.assertEqual(rej[0]["motivo"], "justificativa_falsa")

    def test_julgar_curto_circuito(self):
        q = questao_boa()
        ag = agentes_com(Fila([fase1(q, so_v("C"))]))  # cego diz C, gabarito B
        j = ag.julgar(q, "9º", "H99", "d")
        self.assertFalse(j["ambos"])
        self.assertIsNone(j["revisor"])
        self.assertEqual(ag.orcamento.chamadas, 1)


# ===========================================================================
class TestValidadorRevisor(unittest.TestCase):
    def test_aprova_com_permutacao_e_fase2(self):
        q = questao_boa(gab="D")
        ag = agentes_com(Fila([fase1(q, so_v("D")), FASE2_OK]))
        v = ag.validar(q)
        self.assertTrue(v["veredito"])
        self.assertEqual(v["resposta_calculada"], "D")  # letra ORIGINAL, não a exibida
        self.assertEqual(v["fases"], 2)

    def test_cego_diferente_nao_roda_fase2(self):
        q = questao_boa(gab="B")
        cli = Fila([fase1(q, so_v("A"))])
        v = agentes_com(cli).validar(q)
        self.assertFalse(v["veredito"])
        self.assertEqual(cli.n, 1)
        self.assertIn("gabarito_errado", [p["codigo"] for p in v["problemas"]])

    def test_hierarquia_resposta_nao_unica(self):
        # quadrado (gabarito) e retângulo V, ou um I: nunca é resposta única
        q = questao_boa(gab="B")
        for st in ({"A": "V", "B": "V", "C": "F", "D": "F", "E": "F"},
                   {"A": "I", "B": "V", "C": "F", "D": "F", "E": "F"}):
            v = agentes_com(Fila([fase1(q, st)])).validar(q)
            self.assertFalse(v["veredito"])
            self.assertIsNone(v["resposta_calculada"])
            self.assertIn("gabarito_sem_resposta_unica", [p["codigo"] for p in v["problemas"]])

    def test_incoerencia_interna_reprova(self):
        q = questao_boa(gab="B")
        txt = fase1(q, so_v("B"))
        o = json.loads(txt)
        o["resposta_calculada"] = "E" if o["resposta_calculada"] != "E" else "A"
        v = agentes_com(Fila([json.dumps(o)])).validar(q)
        self.assertFalse(v["veredito"])
        self.assertIn("incoerencia_interna", [p["codigo"] for p in v["problemas"]])

    def test_equivalentes_premissa_visual_reprovam(self):
        q = questao_boa(gab="B")
        for extra in ({"equivalentes": [["A", "C"]]}, {"premissa_possivel": False},
                      {"dependencia_visual": True},
                      {"problemas": [{"codigo": "pergunta_ambigua", "detalhe": "x"}]}):
            cli = Fila([fase1(q, so_v("B"), **extra), FASE2_OK])
            v = agentes_com(cli).validar(q)
            self.assertFalse(v["veredito"], extra)
            self.assertEqual(cli.n, 1, "problema na fase 1 não deve gastar a fase 2")

    def test_revisor_regras(self):
        q = questao_boa(gab="B")
        self.assertTrue(agentes_com(Fila([revisor_json(so_v("B"), questao=q)])).revisar(q, "9º", "H99", "d")["veredito"])
        # um critério bloqueante falho reprova mesmo com veredito true do LLM
        r = agentes_com(Fila([revisor_json(so_v("B"), falha="C9", questao=q)])).revisar(q, "9º", "H99", "d")
        self.assertFalse(r["veredito"])
        self.assertEqual(r["criterios_falhos"], ["C9"])
        # C3 (dificuldade) é aviso: registrado, não veta (calibração r3)
        r = agentes_com(Fila([revisor_json(so_v("B"), falha="C3", questao=q)])).revisar(q, "9º", "H99", "d")
        self.assertTrue(r["veredito"])
        self.assertEqual(r["criterios_falhos"], ["C3"])
        self.assertIn("C3", r["avisos"])
        # ... nem quando o LLM dá veredito false só por causa dele
        obj = json.loads(revisor_json(so_v("B"), falha="C3", questao=q))
        obj.update(veredito=False, problemas=[{"codigo": "dificuldade_incoerente", "detalhe": "é EASY"}])
        self.assertTrue(agentes_com(Fila([json.dumps(obj)])).revisar(q, "9º", "H99", "d")["veredito"])
        # aviso + problema bloqueante continua reprovando
        obj["problemas"].append({"codigo": "contexto_forcado", "detalhe": "x"})
        self.assertFalse(agentes_com(Fila([json.dumps(obj)])).revisar(q, "9º", "H99", "d")["veredito"])
        # veredito false sem nenhuma queixa é incoerente: fail-closed
        obj = json.loads(revisor_json(so_v("B"), questao=q))
        obj["veredito"] = False
        self.assertFalse(agentes_com(Fila([json.dumps(obj)])).revisar(q, "9º", "H99", "d")["veredito"])
        # veredito true do LLM, mas a matemática dele aponta outra letra
        self.assertFalse(agentes_com(Fila([revisor_json(so_v("C"), questao=q)])).revisar(q, "9º", "H99", "d")["veredito"])
        # veredito true com problema listado = incoerente => reprova
        r = agentes_com(Fila([revisor_json(so_v("B"), problemas=[{"codigo": "portugues", "detalhe": "x"}], questao=q)])
                        ).revisar(q, "9º", "H99", "d")
        self.assertFalse(r["veredito"])

    def test_permutacao(self):
        q = questao_boa()
        p1, p2 = aq.permutacao(q), aq.permutacao(q)
        self.assertEqual(p1, p2)  # determinística
        self.assertEqual(sorted(p1.values()), list(LETRAS))
        q_nda = dict(q, alternativas=dict(q["alternativas"], E="Nenhuma das alternativas anteriores"))
        self.assertEqual(aq.permutacao(q_nda)["E"], "E")  # NDA fica no lugar
        q_ref = dict(q, alternativas=dict(q["alternativas"], E="A e B"))
        self.assertEqual(aq.permutacao(q_ref), {L: L for L in LETRAS})
        # enunciado que repete as alternativas rotuladas: permutar trocaria as letras
        q_rot = dict(q, enunciado=q["enunciado"] + " A) 12 B) 13 C) 14 D) 15 E) 16")
        self.assertEqual(aq.permutacao(q_rot), {L: L for L in LETRAS})
        # uma letra solta com parêntese não basta (ex.: "item A) do problema")
        q_um = dict(q, enunciado=q["enunciado"] + " (ver item A) acima")
        self.assertEqual(aq.permutacao(q_um), aq.permutacao(dict(q, enunciado=q["enunciado"] + " (ver item A) acima")))


# ===========================================================================
class TestParseMalformado(unittest.TestCase):
    """Veredito malformado => rejeita; NUNCA aceita por falha de parse."""

    def test_variantes_fase1(self):
        q = questao_boa(gab="B")
        bom = json.loads(fase1(q, so_v("B")))
        sem_e = json.loads(json.dumps(bom))
        del sem_e["alternativas"]["E"]
        talvez = json.loads(json.dumps(bom))
        talvez["alternativas"]["A"]["status"] = "talvez"
        premissa_str = dict(bom, premissa_possivel="sim")
        sem_dep = {k: v for k, v in bom.items() if k != "dependencia_visual"}
        casos = ["", "não sei", '{"alternativas": {"A": {"status": "V"', "<think>raciocinando sem fim",
                 json.dumps(sem_e), json.dumps(talvez), json.dumps(premissa_str), json.dumps(sem_dep),
                 json.dumps({"veredito": True})]
        for texto in casos:
            v = agentes_com(Fila([texto, FASE2_OK])).validar(q)
            self.assertFalse(v["veredito"], texto[:40])
            self.assertFalse(v["avaliado"], texto[:40])
            self.assertEqual(v["erro"], "veredito_malformado")

    def test_juizes_nao_usam_reparo_de_sintaxe(self):
        """T5: os deslizes que o reparo consertava (chave esquecida depois de
        "E", aspas escapadas fora de string) agora REPROVAM no juiz
        (malformado = reprovação, sempre). O gerador continua tolerante."""
        q = questao_boa(gab="B")
        bom = fase1(q, so_v("B"))
        o = json.loads(bom)
        alts = dict(o.pop("alternativas"))
        alts.update(o)
        self.assertIsNone(aq.interpretar_fase1(json.dumps({"alternativas": alts})[:-1]))
        # mesmo como JSON válido, campos caídos dentro de "alternativas" reprovam
        self.assertIsNone(aq.interpretar_fase1(json.dumps({"alternativas": alts})))
        escapado = bom.replace('"motivo": "m"', '"motivo": \\"m\\"', 1)
        self.assertIsNone(aq.interpretar_fase1(escapado))
        v = agentes_com(Fila([escapado, FASE2_OK])).validar(q)
        self.assertFalse(v["veredito"])
        self.assertEqual(v["erro"], "veredito_malformado")
        # fora do modo estrito (saída do gerador) o reparo continua
        self.assertEqual(aq.extrair_json('{"a": \\"x\\", "alternativas": {}}', ("alternativas",))["a"], "x")
        self.assertIsNone(aq.extrair_json('{"a": \\"x\\", "alternativas": {}}', ("alternativas",), estrito=True))

    def test_status_geral(self):
        """G (classe mais geral do mesmo critério) não anula a unicidade, mas
        nunca cria uma resposta: sem V, nada é aprovado."""
        self.assertEqual(aq.resposta_unica(dict(zip(LETRAS, "VGFFF"))), "A")
        self.assertIsNone(aq.resposta_unica(dict(zip(LETRAS, "GGFFF"))))
        self.assertIsNone(aq.resposta_unica(dict(zip(LETRAS, "VGIFF"))))
        self.assertEqual(aq._status_valido({"status": "G"}), "G")
        self.assertEqual(aq._status_valido("geral"), "G")
        self.assertIsNone(aq._status_valido("garantida"))  # primeira letra não basta para G

    def test_fase2_e_revisor_malformados(self):
        q = questao_boa(gab="B")
        v = agentes_com(Fila([fase1(q, so_v("B")), '{"resolucao_correta": true}'])).validar(q)
        self.assertFalse(v["veredito"])
        self.assertFalse(v["avaliado"])  # _falha não pode ser revertida pelos campos extras
        sem_c9 = json.loads(revisor_json(so_v("B"), questao=q))
        del sem_c9["criterios"]["C9"]
        sem_dif = json.loads(revisor_json(so_v("B"), questao=q))
        del sem_dif["dificuldade_real"]
        dif_ruim = json.loads(revisor_json(so_v("B"), questao=q, dificuldade_real="talvez"))
        for texto in (json.dumps(sem_c9), json.dumps(sem_dif), json.dumps(dif_ruim),
                      revisor_json(so_v("B"), questao=q)[:-30], '{"veredito": true}'):
            r = agentes_com(Fila([texto])).revisar(q, "9º", "H99", "d")
            self.assertFalse(r["veredito"])
            self.assertFalse(r["avaliado"])

    def test_extrair_json_tolerante(self):
        o = {"alternativas": {"A": "V"}, "x": 1}
        texto = "<think>pensa {nada} aqui</think>\nResposta: ```json\n" + json.dumps(o) + "\n```"
        self.assertEqual(aq.extrair_json(texto, ("alternativas",)), o)
        self.assertIsNone(aq.extrair_json("<think>{\"alternativas\": 1}", ("alternativas",)))

    def test_erro_de_api_e_fail_closed(self):
        q = questao_boa()
        cli = Roteiro(excecao=RuntimeError("timeout"))
        ag = agentes_com(cli, tentativas=2)
        v = ag.validar(q)
        self.assertFalse(v["veredito"])
        self.assertEqual(v["erro"], "erro_api")
        self.assertEqual(cli.total(), 2)  # retentou uma vez

    def test_injecao_com_validador_malformado_nao_aceita(self):
        amb = Ambiente()
        try:
            cli = Roteiro(validador="malformado")
            silencioso(iq.injetar, amb.args_injecao("--tentativas-por-slot", "1"),
                       agentes=agentes_com(cli, max_chamadas=100))
            self.assertEqual(amb.linhas("injecao_saeb.jsonl"), [])
            self.assertEqual(cli.chamadas["revisor"], 0)
            self.assertTrue(all(r["motivo"] == "veredito_malformado" for r in amb.linhas("rejeitadas.jsonl")))
        finally:
            amb.fechar()


# ===========================================================================
class TestFiltros(unittest.TestCase):
    TAX = {"habilidades": {}}

    def _f(self, q, **kw):
        obj = None if q is None else {"questoes": [q]}
        base = dict(ano="9º", habilidade="H99", dificuldade="Fácil", taxonomia=self.TAX)
        base.update(kw)
        return aq.filtro_injecao(obj, json.dumps(obj or {}), **base)[0]

    def test_aprova_e_nao_muta_estado(self):
        vistos = {"outra"}
        self.assertIsNone(self._f(questao_boa(), vistos=vistos))
        self.assertEqual(vistos, {"outra"})

    def test_cada_filtro_rejeita(self):
        import distill_teacher as dt
        q = questao_boa()
        sem_dif = {k: v for k, v in q.items() if k != "difficulty"}
        casos = {
            "json_invalido": (None, {}),
            "estrutura": (sem_dif, {}),
            "menciona_figura": (dict(q, enunciado="Observe a figura e responda: " + q["enunciado"]), {}),
            "dependencia_visual": (dict(q, enunciado="Conforme a tabela abaixo, " + q["enunciado"]), {}),
            "muito_longa": (dict(q, enunciado=q["enunciado"] + " Detalhe." * 120), {}),
            "resposta_inconsistente": (dict(q, resposta_correta="A"), {}),
            "duplicata": (q, {"vistos": {dt.normalizar(q["enunciado"])}}),
            "difficulty_divergente": (q, {"dificuldade": "Difícil"}),
            "near_duplicata": (q, {"comparar_com": [questao_boa(36, 19)]}),
            "contamina_val": (q, {"val_questoes": [questao_boa(70, 1)]}),
            "dados_ausentes": (dict(q, enunciado="A tabela mostra os votos da turma. Qual cor venceu?"),
                               {"slot": {"subtema": "tabela_simples", "subtema_rotulo": "tabela"}}),
        }
        for esperado, (questao, kw) in casos.items():
            self.assertEqual(self._f(questao, **kw), esperado, esperado)

    def test_geometria_opcional(self):
        q = questao_boa()
        with mock.patch.object(aq, "_vg", None):
            self.assertEqual(aq.veredito_geometria(q)["status"], "indisponivel")
        reprova = SimpleNamespace(verificar_questao=lambda questao: {"ok": False, "motivo": "quadrado ⊂ retângulo"})
        with mock.patch.object(aq, "_vg", reprova):
            self.assertEqual(self._f(q), "geometria")
        def explode(questao):
            raise ValueError("api nova")
        with mock.patch.object(aq, "_vg", SimpleNamespace(verificar=explode)):
            self.assertIsNone(self._f(q))  # falha do verificador nunca reprova
        with mock.patch.object(aq, "_vg", SimpleNamespace(verificar=lambda q, ano=None: ("aprovada", []))):
            self.assertEqual(aq.veredito_geometria(q, "9º")["status"], "aprovada")

    def test_geometria_api_verificar_geometria(self):
        """API publicada pelo outro workflow: (veredito_str, detalhe). Os de
        GEO_REJEITA reprovam; "ok" aprova; "nao_aplicavel" se abstém."""
        q = questao_boa()
        def vg(veredito):
            return SimpleNamespace(GEO_REJEITA=frozenset({"gabarito_errado", "nao_unica", "premissa_impossivel",
                                                          "dados_insuficientes"}),
                                   verificar_geometria=lambda questao: (veredito, {"motivo": "m", "explicacao": "e",
                                                                                   "fatos": ["grande"] * 50}))
        for v in ("gabarito_errado", "nao_unica", "premissa_impossivel", "dados_insuficientes"):
            with mock.patch.object(aq, "_vg", vg(v)):
                self.assertEqual(aq.veredito_geometria(q)["status"], "reprovada", v)
                self.assertEqual(self._f(q), "geometria")
                self.assertEqual(aq.filtros_auditoria(q)["reprovado"], ["geometria"])
        with mock.patch.object(aq, "_vg", vg("ok")):
            self.assertEqual(aq.veredito_geometria(q)["status"], "aprovada")
            self.assertIsNone(self._f(q))
        with mock.patch.object(aq, "_vg", vg("nao_aplicavel")):
            self.assertEqual(aq.veredito_geometria(q)["status"], "sem_veredito")
            self.assertIsNone(self._f(q))
            self.assertEqual(aq.filtros_auditoria(q)["reprovado"], [])


# ===========================================================================
class TestOrcamento(unittest.TestCase):
    def test_teto_duro(self):
        o = aq.Orcamento(2)
        o.reservar("x")
        o.reservar("x")
        with self.assertRaises(aq.OrcamentoEsgotado):
            o.reservar("x")
        self.assertEqual(o.chamadas, 2)

    def test_retries_contam_e_param_no_teto(self):
        cli = Roteiro(excecao=RuntimeError("503"))
        ag = agentes_com(cli, max_chamadas=2, tentativas=5)
        with self.assertRaises(aq.OrcamentoEsgotado):
            ag.validar(questao_boa())
        self.assertEqual(cli.total(), 2)

    def test_erro_nao_retentavel(self):
        exc = RuntimeError("401")
        exc.response = SimpleNamespace(status_code=401)
        cli = Roteiro(excecao=exc)
        agentes_com(cli, tentativas=5).validar(questao_boa())
        self.assertEqual(cli.total(), 1)

    def test_injecao_respeita_max_chamadas(self):
        amb = Ambiente(falta_facil=20)
        try:
            for teto in (0, 3, 6, 11):
                cli = Roteiro(revisor_aprova=False)
                args = amb.args_injecao("--max-chamadas", str(teto), "--saida", str(amb.d / f"s{teto}.jsonl"),
                                        "--rejeitadas", str(amb.d / f"r{teto}.jsonl"))
                res = silencioso(iq.injetar, args, agentes=agentes_com(cli, max_chamadas=teto))
                self.assertLessEqual(cli.total(), teto)
                self.assertLessEqual(res["uso"]["chamadas"], teto)
                self.assertIn("orcamento", res["parada"])
        finally:
            amb.fechar()

    def test_sigint_grava_resumo(self):
        """Interromper a injeção (SIGINT) no meio de um candidato não perde o
        resumo nem grava o candidato pela metade."""
        class Interrompe(Roteiro):
            def chat_completion(self, messages, max_tokens, temperature):
                if papel_de(messages) == "revisor" and sum(self.chamadas.values()) >= 6:
                    raise KeyboardInterrupt
                return super().chat_completion(messages, max_tokens, temperature)
        amb = Ambiente(falta_facil=5)
        try:
            cli = Interrompe()
            res = silencioso(iq.injetar, amb.args_injecao(), agentes=agentes_com(cli, max_chamadas=100))
            self.assertIn("interrompido", res["parada"])
            self.assertTrue((amb.d / "resumo.json").exists())
            self.assertEqual(len(amb.linhas("injecao_saeb.jsonl")), res["funil"]["aceitos"])
            self.assertEqual(res["funil"]["candidatos"], res["funil"]["aceitos"] + len(amb.linhas("rejeitadas.jsonl")))
        finally:
            amb.fechar()

    def test_tokens_contabilizados_e_logados(self):
        amb = Ambiente()
        try:
            ag = agentes_com(Roteiro(), max_chamadas=100, log_uso=amb.d / "uso.jsonl")
            res = silencioso(iq.injetar, amb.args_injecao(), agentes=ag)
            uso = amb.linhas("uso.jsonl")
            self.assertEqual(len(uso), res["uso"]["chamadas"])
            self.assertEqual(sum(u["prompt_tokens"] for u in uso), res["uso"]["prompt_tokens"])
            self.assertTrue(all({"agente", "fase", "modelo", "latencia_s", "ok"} <= set(u) for u in uso))
            self.assertIsNotNone(res["custo_por_aceita"]["chamadas"])
        finally:
            amb.fechar()

    def test_cli_exige_max_chamadas_fora_do_dry_run(self):
        amb = Ambiente()
        try:
            args = amb.args_injecao()
            args.max_chamadas = None
            with self.assertRaises(SystemExit):
                silencioso(iq.injetar, args)
        finally:
            amb.fechar()


# ===========================================================================
class TestRetomada(unittest.TestCase):
    def test_injecao_idempotente(self):
        amb = Ambiente(falta_facil=2)
        try:
            silencioso(iq.injetar, amb.args_injecao(), agentes=agentes_com(Roteiro(), 100))
            self.assertEqual(len(amb.linhas("injecao_saeb.jsonl")), 2)
            cli = Roteiro()
            res = silencioso(iq.injetar, amb.args_injecao(), agentes=agentes_com(cli, 100))
            self.assertEqual(cli.total(), 0)
            self.assertEqual(res["funil"]["candidatos"], 0)
            self.assertEqual(len(amb.linhas("injecao_saeb.jsonl")), 2)
            # linha final truncada (interrupção no meio da escrita) não quebra a retomada
            with open(amb.d / "injecao_saeb.jsonl", "a", encoding="utf-8") as f:
                f.write('{"messages": [')
            silencioso(iq.injetar, amb.args_injecao(), agentes=agentes_com(Roteiro(), 100))
        finally:
            amb.fechar()

    def test_tentativas_anteriores_contam_no_teto(self):
        amb = Ambiente(falta_facil=1)
        try:
            cli = Roteiro(revisor_aprova=False)
            silencioso(iq.injetar, amb.args_injecao("--tentativas-por-slot", "2"), agentes=agentes_com(cli, 100))
            self.assertEqual(len(amb.linhas("rejeitadas.jsonl")), 2)
            cli2 = Roteiro(revisor_aprova=False)
            silencioso(iq.injetar, amb.args_injecao("--tentativas-por-slot", "2"), agentes=agentes_com(cli2, 100))
            self.assertEqual(cli2.total(), 0)  # teto 1 x 2 tentativas já gasto
        finally:
            amb.fechar()

    def test_auditoria_retoma_e_refaz_nao_avaliado(self):
        amb = Ambiente()
        try:
            # 1ª: API cai em tudo => nao_avaliado (nunca aprovação)
            cli = Roteiro(excecao=RuntimeError("503"))
            silencioso(ab.auditar, amb.args_auditoria("--tentativas-api", "1"), agentes=agentes_com(cli, 100, tentativas=1))
            regs = amb.linhas("auditoria.jsonl")
            # quem passou no filtro determinístico e não foi julgado é nao_avaliado;
            # quem o filtro reprovou é "baixa" sem gastar chamada
            self.assertTrue(all(r["confianca"] == "nao_avaliado" for r in regs if not r["filtros"]["reprovado"]))
            self.assertTrue(all(r["confianca"] == "baixa" for r in regs if r["filtros"]["reprovado"]))
            self.assertEqual(sum(r["confianca"] == "nao_avaliado" for r in regs), 3)
            # 2ª: refaz só os nao_avaliado
            cli2 = Roteiro()
            silencioso(ab.auditar, amb.args_auditoria(), agentes=agentes_com(cli2, 100))
            self.assertGreater(cli2.total(), 0)
            # 3ª: tudo em cache, 0 chamadas
            cli3 = Roteiro()
            silencioso(ab.auditar, amb.args_auditoria(), agentes=agentes_com(cli3, 100))
            self.assertEqual(cli3.total(), 0)
        finally:
            amb.fechar()


# ===========================================================================
class TestAuditoriaEMontagem(unittest.TestCase):
    def test_niveis(self):
        ok = {"veredito": True, "avaliado": True}
        no = {"veredito": False, "avaliado": True}
        na = {"veredito": False, "avaliado": False}
        self.assertEqual(ab.nivel_confianca([], ok, ok), "alta")
        self.assertEqual(ab.nivel_confianca([], ok, no), "media")
        self.assertEqual(ab.nivel_confianca([], no, ok), "media")
        self.assertEqual(ab.nivel_confianca([], no, no), "baixa")
        self.assertEqual(ab.nivel_confianca(["visual"], ok, ok), "baixa")
        self.assertEqual(ab.nivel_confianca([], na, ok), "nao_avaliado")
        self.assertEqual(ab.nivel_confianca([], ok, na), "nao_avaliado")

    def test_auditoria_e_v3(self):
        amb = Ambiente()
        try:
            silencioso(iq.injetar, amb.args_injecao(), agentes=agentes_com(Roteiro(), 100))
            silencioso(ab.auditar, amb.args_auditoria(), agentes=agentes_com(Roteiro(), 100))
            regs = {r["codigo_item"]: r for r in amb.linhas("auditoria.jsonl")}
            self.assertEqual(regs["MT9999"]["confianca"], "alta")
            self.assertEqual(regs["MT9999"]["origem"], "real")
            ruim = regs["DIST-H99-Fácil-00002"]  # gabarito E, conta dá D
            self.assertEqual(ruim["confianca"], "baixa")
            self.assertTrue(ruim["problemas_matematicos"])
            for campo in ("idx", "ano", "habilidade", "vereditos", "problemas", "hash_questao", "versao_prompts",
                          "filtros", "uso"):
                self.assertIn(campo, ruim)
            # H4: sem o árbitro de outra família, a D2 não remove — fica e vai
            # para revisão humana
            rel = silencioso(ab.montar, amb.args_auditoria("--montar"))
            self.assertIn("DIST-H99-Fácil-00002", [e["meta"]["codigo_item"] for e in amb.linhas("v3.jsonl")])
            self.assertEqual(rel["d2"], {"d2_aguardando_arbitro_revisao_humana": 1})
            # com o árbitro concordando (resolve 30 + 9 = 39 = D, gabarito E), sai
            silencioso(ab.arbitrar, amb.args_auditoria("--arbitrar"), arbitro=arbitro_com(Roteiro()))
            rel = silencioso(ab.montar, amb.args_auditoria("--montar"))
            v3 = amb.linhas("v3.jsonl")
            codigos = [e["meta"]["codigo_item"] for e in v3]
            self.assertNotIn("DIST-H99-Fácil-00002", codigos)
            self.assertIn("MT9999", codigos)
            self.assertEqual(sum(c.startswith("INJ-") for c in codigos), 2)
            self.assertEqual(rel["n_removidos"], 1)
            self.assertEqual(rel["removidos"][0]["motivo"], "erro_matematico_confirmado_arbitro")
        finally:
            amb.fechar()

    def test_reais_e_sinteticos_nunca_saem_automaticamente(self):
        baixa = {"confianca": "baixa", "problemas_matematicos": ["gabarito_errado"]}
        media = {"confianca": "media", "problemas_matematicos": ["resposta_nao_unica"]}
        self.assertEqual(ab.decidir("real", baixa)[0], "manter")
        self.assertEqual(ab.decidir("sintetico", baixa)[0], "manter")
        self.assertEqual(ab.decidir("real", baixa, remover_reais_baixa=True)[0], "remover")
        # Revisão do piloto 2: destilado reprovado SEM a confirmação da D2 não sai
        # mais (a regra antiga removeu o idx 1187, correto, por um juiz só); fica
        # e vai para a revisão humana.
        self.assertEqual(ab.decidir("destilado", baixa), ("manter", "baixa_sem_confirmacao_revisao_humana"))
        self.assertEqual(ab.decidir("destilado", media), ("manter", "media_sem_confirmacao_revisao_humana"))
        self.assertEqual(ab.decidir("destilado", media, manter_media=True)[0], "manter")
        self.assertEqual(ab.decidir("destilado", {"confianca": "media", "problemas_matematicos": []})[0], "manter")
        self.assertEqual(ab.decidir("destilado", None), ("manter", "nao_auditado"))
        self.assertEqual(ab.decidir("destilado", {"confianca": "nao_avaliado"})[0], "manter")

    def test_amostra_estratificada(self):
        exs = [{"meta": {"ano": "9º", "habilidade": f"H{i % 5}"}} for i in range(50)]
        idx = ab.amostra_estratificada(exs, 10, seed=1)
        self.assertEqual(len(idx), 10)
        self.assertEqual(len({exs[i]["meta"]["habilidade"] for i in idx}), 5)
        self.assertEqual(idx, ab.amostra_estratificada(exs, 10, seed=1))

    def test_amostra_cobre_todos_os_anos(self):
        # 3 anos com 1 habilidade cada e 1 ano com 20: amostra de 5 < 23
        # habilidades ainda pega todos os anos (anos intercalados).
        exs = [{"meta": {"ano": a, "habilidade": "EF"}} for a in ("1º", "3º", "4º") for _ in range(30)]
        exs += [{"meta": {"ano": "9º", "habilidade": f"H{i:02d}"}} for i in range(20) for _ in range(3)]
        for seed in range(5):
            idx = ab.amostra_estratificada(exs, 5, seed=seed)
            self.assertEqual({exs[i]["meta"]["ano"] for i in idx}, {"1º", "3º", "4º", "9º"})

    def test_calibracao_gate(self):
        boa = questao_boa(gab="C")
        ruim = dict(questao_boa(gab="A"), resposta_correta="E")
        itens = [{"id": "X-Q1", "ano": "9º", "habilidade": "H99", "descricao": "d", "dificuldade": "Fácil",
                  "questao": boa, "boa": True},
                 {"id": "X-Q2", "ano": "9º", "habilidade": "H99", "descricao": "d", "dificuldade": "Fácil",
                  "questao": ruim, "boa": False}]
        amb = Ambiente()
        try:
            res = silencioso(ab.calibrar, amb.args_auditoria("--calibrar"), agentes=agentes_com(Roteiro(), 80),
                             itens=itens)
            self.assertEqual(res["gate"], "passa")
            self.assertEqual(res["par"]["reprova_ruins"], "1/1")
        finally:
            amb.fechar()
        self.assertEqual(ab.wilson(15, 15)[0], 0.796)
        self.assertEqual(ab.kappa([(True, True), (False, False)]), 1.0)
        self.assertEqual(len(ab.carregar_calibracao()), 20)
        # D5: R2-Q5 ("Isósceles" verdadeiro para o equilátero) passa a ruim;
        # o rótulo do Log.txt fica registrado
        cal = {i["id"]: i for i in ab.carregar_calibracao()}
        self.assertEqual(sum(i["boa"] for i in cal.values()), 4)
        self.assertFalse(cal["R2-Q5"]["boa"])
        self.assertTrue(cal["R2-Q5"]["rotulo_log"])


# ===========================================================================
class TestPlano(unittest.TestCase):
    def test_lacunas_respeita_graus_do_banco(self):
        # 2º H21: o banco só tem Difícil; Fácil/Moderado continuam fora por padrão
        rel = {"habilidades": [{"ano": "2º", "habilidade": "H21", "meta": 30, "n_efetivas": 30,
                                "falta_por_dificuldade": {"Fácil": 10, "Moderado": 10, "Difícil": 0},
                                "falta_por_subtema": {}, "banco": {"graus": ["Difícil"]}}]}
        lista, ign = iq.lacunas(rel)
        self.assertEqual(lista, [])
        self.assertEqual(ign[0]["motivo"], "dificuldade_ausente_no_banco")
        lista, _ = iq.lacunas(rel, incluir_dif_ausente=True)
        self.assertEqual(lista[0]["alvo"], 20)

    def test_d4_dificil_liberado_no_1o_e_3o_ano(self):
        """D4: o banco do 1º/3º ano não tem Difícil, mas a decisão do usuário
        libera — por padrão, só nesses anos."""
        def rel(ano):
            return {"habilidades": [{"ano": ano, "habilidade": "EFX", "meta": 30, "n_efetivas": 47,
                                     "falta_por_dificuldade": {"Fácil": 0, "Moderado": 0, "Difícil": 10},
                                     "falta_por_subtema": {}, "banco": {"graus": ["Fácil", "Moderado"]}}]}
        for ano in ("1º", "3º"):
            lista, ign = iq.lacunas(rel(ano))
            self.assertEqual(ign, [], ano)
            self.assertEqual(lista[0]["alvo"], 10)
            self.assertIn("Difícil", lista[0]["permitidas"])
        lista, ign = iq.lacunas(rel("2º"))
        self.assertEqual(lista, [])
        self.assertEqual(ign[0]["falta_ignorada"], {"Difícil": 10})
        lista, _ = iq.lacunas(rel("1º"), liberar_dificil=())
        self.assertEqual(lista, [])

    def test_d4_prompt_do_professor_no_dificil_anos_iniciais(self):
        msgs = []

        class Grava(aq.ClienteSimulado):
            def chat_completion(self, messages, max_tokens, temperature):
                msgs.append(messages)
                return super().chat_completion(messages, max_tokens, temperature)
        ag = agentes_com(Grava())
        ag.gerar("1º", "EF01MA08", "Resolver problemas de adição.", "Difícil")
        ag.gerar("1º", "EF01MA08", "Resolver problemas de adição.", "Fácil")
        ag.gerar("9º", "H02", "Resolver problemas.", "Difícil")
        self.assertIn("SEM sair do conteúdo do 1º ano", msgs[0][1]["content"])
        self.assertNotIn("SEM sair do conteúdo", msgs[1][1]["content"])
        self.assertNotIn("SEM sair do conteúdo", msgs[2][1]["content"])

    def test_dificuldades_intercaladas(self):
        lac = {"ano": "9º", "habilidade": "H99", "permitidas": ["Fácil", "Moderado", "Difícil"],
               "falta_dif": {"Fácil": 2, "Moderado": 2, "Difícil": 1}}
        self.assertEqual(Counter(iq.dificuldades_para(lac, 5, Counter())), Counter({"Fácil": 2, "Moderado": 2,
                                                                                  "Difícil": 1}))
        # Fácil já coberto pelo injetado: só Moderado e Difícil, maior falta primeiro
        out = iq.dificuldades_para(lac, 3, Counter({"Fácil": 2}))
        self.assertEqual(out[0], "Moderado")
        self.assertEqual(Counter(out), Counter({"Moderado": 2, "Difícil": 1}))
        # determinístico
        self.assertEqual(iq.dificuldades_para(lac, 5, Counter()), iq.dificuldades_para(lac, 5, Counter()))

    def test_t6_empate_nao_favorece_facil(self):
        """T6: com faltas iguais, a 1ª dificuldade não é sempre Fácil."""
        primeiras = Counter()
        for i in range(60):
            lac = {"ano": "5º", "habilidade": f"H{i:02d}", "permitidas": ["Fácil", "Moderado", "Difícil"],
                   "falta_dif": {"Fácil": 3, "Moderado": 3, "Difícil": 3}}
            out = iq.dificuldades_para(lac, 3, Counter())
            self.assertEqual(sorted(out), sorted(["Fácil", "Moderado", "Difícil"]))
            primeiras[out[0]] += 1
        self.assertEqual(set(primeiras), {"Fácil", "Moderado", "Difícil"})
        self.assertLess(primeiras["Fácil"], 35, primeiras)
        # sem falta nenhuma: o giro também não começa sempre em Fácil
        inicios = Counter(iq.dificuldades_para({"ano": "5º", "habilidade": f"H{i}", "permitidas": list(iq.DIFICULDADES),
                                                "falta_dif": {}}, 1, Counter())[0] for i in range(60))
        self.assertEqual(set(inicios), set(iq.DIFICULDADES))

    def test_letra_alvo_equilibra(self):
        import random
        self.assertEqual(iq.escolher_letra(Counter({"A": 3, "B": 3, "C": 3, "D": 3}), random.Random(0)), "E")

    def test_parse_habilidades(self):
        self.assertEqual(iq.parse_habilidades("H17, 9º:H17,2 h07"), {(None, "H17"), ("9º", "H17"), ("2º", "H07")})


# ===========================================================================
class TestSeguranca(unittest.TestCase):
    CHAVE = "sk-SEGREDO-0123456789abcdef"

    def test_arquivos_protegidos(self):
        for p in PROTEGIDOS + [ROOT / "data" / "val_qualquer.jsonl"]:
            with self.assertRaises(PermissionError):
                aq._garantir_gravavel(p)
        aq._garantir_gravavel(ROOT / "data" / "injecao_saeb.jsonl")  # permitido
        amb = Ambiente()
        try:
            args = amb.args_injecao("--saida", str(ROOT / "data" / "train_curado.jsonl"))
            with self.assertRaises(PermissionError):
                silencioso(iq.injetar, args, agentes=agentes_com(Roteiro(), 10))
            args = amb.args_auditoria("--montar", "--saida-v3", str(ROOT / "data" / "train_curado.jsonl"))
            with self.assertRaises(PermissionError):
                silencioso(ab.montar, args)
        finally:
            amb.fechar()

    def test_chave_nunca_gravada(self):
        amb = Ambiente()
        try:
            with mock.patch.dict(os.environ, {"MARITALK_API_KEY": self.CHAVE}):
                exc = RuntimeError(f"401 Client Error: headers={{'Authorization': 'Bearer {self.CHAVE}'}} "
                                   f"api_key={self.CHAVE}")
                cli = Roteiro(excecao=exc)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    iq.injetar(amb.args_injecao("--max-erros-api", "2"),
                               agentes=agentes_com(cli, 100, log_uso=amb.d / "uso.jsonl", tentativas=2))
                    ab.auditar(amb.args_auditoria(), agentes=agentes_com(cli, 100, log_uso=amb.d / "uso.jsonl",
                                                                          tentativas=1))
                self.assertGreater(cli.total(), 0)
                self.assertNotIn(self.CHAVE, out.getvalue())
                for p in amb.d.rglob("*"):
                    if p.is_file():
                        self.assertNotIn(self.CHAVE, p.read_text(encoding="utf-8", errors="ignore"), p.name)
                self.assertTrue(any("***" in l.get("erro", "") for l in amb.linhas("uso.jsonl")))
                c = aq.ClienteMaritacaMedido("sabia-4", api_key=self.CHAVE)
                self.assertNotIn(self.CHAVE, repr(c))
                self.assertNotIn(self.CHAVE, aq._sanitizar(f"x {self.CHAVE} y"))
        finally:
            amb.fechar()

    def test_nenhum_arquivo_protegido_mudou(self):
        # roda por último na ordem alfabética da classe; confere o estado após a suíte
        for p, sha in SHA_INICIAL.items():
            self.assertEqual(_sha(p), sha, p.name)


# ===========================================================================
# Correções da revisão adversarial do piloto (etapa 3)
# ===========================================================================

def questao_hexagono(enunciado_extra=""):
    """idx 1368 do train_curado (DIST-H15-Fácil-00589): o enunciado não diz que
    o hexágono é regular e a resolução SUPÕE que é. Validador (fase 2) e
    revisor aprovaram; a auditoria humana confirmou o erro."""
    return {"enunciado": ("Larissa resolveu guardar sua mesada em um cofrinho de base hexagonal"
                          + enunciado_extra + ". Para decorá-lo, ela quer colar adesivos em cada lado "
                          "fazendo um ângulo igual ao ângulo interno do hexágono. Qual é a medida desse ângulo interno?"),
            "alternativas": {"A": "108º", "B": "120º", "C": "135º", "D": "150º", "E": "144º"},
            "resolucao_passo_a_passo": ("A soma dos ângulos internos de um polígono é S = (n - 2) x 180º. Para um "
                                        "hexágono (n = 6), S = (6 - 2) x 180º = 4 x 180º = 720º. Como o hexágono é "
                                        "regular, cada ângulo interno mede 720º ÷ 6 = 120º."),
            "resposta_correta": "B", "difficulty": "EASY"}


class TestCorrecoesRevisao(unittest.TestCase):
    # --- (1) fail-open: JSON truncado logo depois de resposta_calculada --------
    def test_fase1_truncada_antes_de_problemas_reprova(self):
        q = questao_boa(gab="B")
        completo = fase1(q, so_v("B"))
        corte = completo.index(', "problemas"')
        truncado = completo[:corte]  # o reparo fecharia a chave e "aprovaria"
        sem_conf = completo[:completo.index(', "confianca"')]
        for texto in (truncado, sem_conf):
            self.assertIsNone(aq.interpretar_fase1(texto))
            v = agentes_com(Fila([texto, FASE2_OK])).validar(q)
            self.assertFalse(v["veredito"])
            self.assertEqual(v["erro"], "veredito_malformado")
        self.assertIsNotNone(aq.interpretar_fase1(completo))

    def test_fase2_e_revisor_truncados_reprovam(self):
        f2 = json.loads(FASE2_OK)
        sem_prob = json.dumps({k: v for k, v in f2.items() if k not in ("problemas", "confianca")})
        self.assertIsNone(aq.interpretar_fase2(sem_prob))
        self.assertIsNotNone(aq.interpretar_fase2(FASE2_OK))
        rev = json.loads(revisor_json(so_v("B")))
        del rev["confianca"]
        self.assertIsNotNone(aq.interpretar_revisor(revisor_json(so_v("B"))))
        self.assertIsNone(aq.interpretar_revisor(json.dumps(rev)))

    # --- (2) resolução que supõe dado ausente / resolução vazia -------------
    def test_resolucao_supoe_dado_reprova_no_filtro(self):
        q = questao_hexagono()
        self.assertEqual(aq.defeitos_resolucao(q), ["resolucao_supoe_dado"])
        f = aq.filtros_auditoria(q)
        self.assertIn("resolucao_supoe_dado", f["reprovado"])
        # o mesmo item COM "regular" no enunciado é legítimo
        ok = questao_hexagono(" regular")
        self.assertEqual(aq.defeitos_resolucao(ok), [])
        self.assertNotIn("resolucao_supoe_dado", aq.filtros_auditoria(ok)["reprovado"])
        # outras formas do mesmo erro (não só o hexágono)
        penta = dict(questao_boa(), resolucao_passo_a_passo="Supondo que todas as caixas tenham 12 ovos, 3 x 12 = 36.")
        self.assertEqual(aq.defeitos_resolucao(penta), ["resolucao_supoe_dado"])
        octo = dict(questao_hexagono(), resolucao_passo_a_passo="Como o octógono é regular, 1080 ÷ 8 = 135.")
        self.assertEqual(aq.defeitos_resolucao(octo), ["resolucao_supoe_dado"])
        # conta comum não dispara
        self.assertEqual(aq.defeitos_resolucao(questao_boa()), [])

    def test_resolucao_vazia_reprova(self):
        q = dict(questao_boa(), resolucao_passo_a_passo="")
        self.assertEqual(aq.defeitos_resolucao(q), ["resolucao_vazia"])
        self.assertIn("resolucao_vazia", aq.filtros_auditoria(q)["reprovado"])
        self.assertEqual(aq.defeitos_resolucao(dict(questao_boa(), resolucao_passo_a_passo="48 ÷ 8 = 6.")), [])

    def test_filtro_injecao_barra_dado_suposto(self):
        q = dict(questao_boa(), resolucao_passo_a_passo="Supondo que cada caixa tem 10, 35 + 18 = 53.")
        obj = {"questoes": [q]}
        motivo, _ = aq.filtro_injecao(obj, json.dumps(obj, ensure_ascii=False), ano="9º", habilidade="H99",
                                      dificuldade="Fácil")
        self.assertEqual(motivo, "resolucao_supoe_dado")

    def test_auditoria_reaplica_filtro_novo_ao_cache_sem_chamadas(self):
        """Registro auditado com filtros ANTIGOS (como os 59 do piloto) é
        refiltrado com 0 chamadas: o rótulo 'alta' do hexágono vira 'baixa'."""
        amb = Ambiente()
        try:
            with open(amb.train, "a", encoding="utf-8") as f:
                f.write(json.dumps(exemplo("9º", "H99", questao_hexagono(), "DIST-H15-Fácil-00589",
                                           destilado=True, professor="p"), ensure_ascii=False) + "\n")
            # 1ª auditoria com o filtro antigo (sem defeitos_resolucao)
            with mock.patch.object(aq, "defeitos_resolucao", lambda q: []), \
                    mock.patch.object(aq, "VERSAO_FILTROS", "antiga"):
                silencioso(ab.auditar, amb.args_auditoria(), agentes=agentes_com(_HexagonoAprova(), 100))
            antes = {r["codigo_item"]: r for r in amb.linhas("auditoria.jsonl")}
            self.assertEqual(antes["DIST-H15-Fácil-00589"]["confianca"], "alta")
            cli = Roteiro()
            silencioso(ab.auditar, amb.args_auditoria(), agentes=agentes_com(cli, 100))
            self.assertEqual(cli.total(), 0)
            depois = ab.ultimos_registros(amb.d / "auditoria.jsonl")
            hexa = next(r for r in depois.values() if r["codigo_item"] == "DIST-H15-Fácil-00589")
            self.assertEqual(hexa["confianca"], "baixa")
            self.assertIn("filtro:resolucao_supoe_dado", hexa["problemas_matematicos"])
            self.assertEqual(hexa["versao_filtros"], aq.VERSAO_FILTROS)
            self.assertTrue(hexa["validador"]["veredito"])  # o que os juízes disseram fica registrado
            outros = {r["codigo_item"]: r["confianca"] for r in depois.values()}
            for cod, r in antes.items():
                if cod != "DIST-H15-Fácil-00589":
                    self.assertEqual(outros[cod], r["confianca"], cod)
            self.assertTrue(all(r["versao_filtros"] == aq.VERSAO_FILTROS for r in depois.values()))
            # 3ª: nada a refazer nem a refiltrar
            n = len(amb.linhas("auditoria.jsonl"))
            silencioso(ab.auditar, amb.args_auditoria(), agentes=agentes_com(Roteiro(), 100))
            self.assertEqual(len(amb.linhas("auditoria.jsonl")), n)
            # Revisão do piloto 2: com os DOIS juízes aprovando, o filtro (heurística
            # de texto) sozinho não confirma erro matemático (D2): o destilado fica
            # e vai para a revisão humana com a evidência — antes saía pela regra
            # antiga "baixa", sem confirmação. (Com o validador reprovando por
            # dados_insuficientes, como no idx 1368 real, sai: ver TestCriterioD2.)
            silencioso(ab.montar, amb.args_auditoria("--montar"))
            self.assertIn("DIST-H15-Fácil-00589", [e["meta"]["codigo_item"] for e in amb.linhas("v3.jsonl")])
            humanos = {r["codigo_item"]: r for r in amb.linhas("humana.jsonl")}
            self.assertEqual(humanos["DIST-H15-Fácil-00589"]["motivo_montagem"],
                             "baixa_sem_confirmacao_revisao_humana")
        finally:
            amb.fechar()

    # --- (3) retomada: linha truncada + códigos INJ removidos ---------------
    def test_append_apos_linha_truncada_e_codigo_sem_colisao(self):
        amb = Ambiente(falta_facil=3)
        try:
            silencioso(iq.injetar, amb.args_injecao("--meta-por-habilidade", "2"),
                       agentes=agentes_com(Roteiro(), 100))
            linhas = amb.linhas("injecao_saeb.jsonl")
            self.assertEqual(len(linhas), 2)
            # a 1ª aceita sai (auditoria independente) e a escrita seguinte foi interrompida
            saida = amb.d / "injecao_saeb.jsonl"
            saida.write_text(json.dumps(linhas[1], ensure_ascii=False) + "\n" + '{"messages": [{"ro',
                             encoding="utf-8")
            with open(amb.d / "rejeitadas.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"ano": "9º", "habilidade": "H99", "etapa": "auditoria_independente",
                                    "motivo": "auditoria_independente",
                                    "codigo_item": linhas[0]["meta"]["codigo_item"]}, ensure_ascii=False) + "\n")
            silencioso(iq.injetar, amb.args_injecao("--meta-por-habilidade", "2", "--tentativas-por-slot", "3"),
                       agentes=agentes_com(Roteiro(), 100))
            brutas = saida.read_text(encoding="utf-8").splitlines()
            validas, invalidas = [], []
            for l in brutas:
                try:
                    validas.append(json.loads(l))
                except json.JSONDecodeError:
                    invalidas.append(l)
            # T3: o fragmento sai da saída (quarentena) e a nova aceita não é colada nele
            self.assertEqual(invalidas, [])
            quarentena = amb.linhas("injecao_saeb_truncadas.jsonl")
            self.assertEqual([r["fragmento"] for r in quarentena], ['{"messages": [{"ro'])
            self.assertEqual(len(validas), 2)
            codigos = [e["meta"]["codigo_item"] for e in validas] + [linhas[0]["meta"]["codigo_item"]]
            self.assertEqual(len(codigos), len(set(codigos)), codigos)
            sufixos = sorted(int(c.rsplit("-", 1)[1]) for c in codigos)
            self.assertEqual(sufixos, [1, 2, 3])
        finally:
            amb.fechar()


# ===========================================================================
# Decisões do usuário de 2026-10-01 (D1-D5) e correções técnicas (T1-T6)
# ===========================================================================

class Grava(aq.ClienteSimulado):
    """ClienteSimulado que guarda TODAS as mensagens enviadas, por papel."""

    def __init__(self, **kw):
        super().__init__(erro_a_cada=0, **kw)
        self.msgs = []

    def chat_completion(self, messages, max_tokens, temperature):
        self.msgs.append((papel_de(messages), messages))
        return super().chat_completion(messages, max_tokens, temperature)


class TestD1RevisorCego(unittest.TestCase):
    def test_revisor_nao_recebe_gabarito_nem_resolucao(self):
        q = dict(questao_boa(gab="D"), resolucao_passo_a_passo="Somando 35 + 18 obtemos 53 (RESOLUCAO-MARCADOR).")
        cli = Grava()
        ag = agentes_com(cli)
        rev = ag.revisar(q, "9º", "H99", "Resolver problemas de adição.", "adição", "Moderado")
        enviadas = [m for papel, m in cli.msgs if papel == "revisor"]
        self.assertEqual(len(enviadas), 1)
        texto = "\n".join(x["content"] for x in enviadas[0])
        self.assertNotIn("RESOLUCAO-MARCADOR", texto)
        self.assertNotIn("Somando 35 + 18", texto)
        user = enviadas[0][1]["content"]
        self.assertNotIn("abarito", user)
        self.assertNotIn("esolução", user)
        self.assertNotIn("Resolva o item antes de ler", user)
        # nenhuma linha "X) ..." com a letra do gabarito destacada nem o texto da resolução
        self.assertNotIn("difficulty", user)
        self.assertNotIn("MEDIUM", enviadas[0][1]["content"])
        self.assertNotIn("EASY", enviadas[0][1]["content"])
        # o que ele precisa para julgar a pedagogia está lá
        for trecho in ("Ano: 9º ano", "Habilidade: H99", "Subtema pedido: adição", q["enunciado"]):
            self.assertIn(trecho, enviadas[0][1]["content"])
        # P3 (2026-10-01): e a dificuldade pedida NÃO (ancorava a dificuldade_real)
        self.assertNotIn("Dificuldade pedida", user)
        self.assertNotIn("Moderado", user)
        # e mesmo assim o código compara com o gabarito (resolve 35 + 18 = 53 -> D)
        self.assertTrue(rev["veredito"], rev["problemas"])
        self.assertEqual(rev["resposta_calculada"], "D")
        self.assertEqual(rev["dificuldade_real"] in aq.DIFICULDADES_VALIDAS, True)

    def test_revisor_ve_outra_permutacao_que_o_validador(self):
        q = questao_boa(gab="B")
        self.assertEqual(sorted(aq.permutacao(q, sal="revisor").values()), list(LETRAS))
        # em algum item a ordem difere (não é a mesma semente)
        difere = any(aq.permutacao(questao_boa(a, b)) != aq.permutacao(questao_boa(a, b), sal="revisor")
                     for a, b in ((35, 18), (40, 12), (51, 7), (22, 9)))
        self.assertTrue(difere)
        cli = Grava()
        agentes_com(cli).julgar(q, "9º", "H99", "d", curto_circuito=False)
        alts_val = [m for p, m in cli.msgs if p == "validador"][0][1]["content"].split("\nA) ")[1]
        alts_rev = [m for p, m in cli.msgs if p == "revisor"][0][1]["content"].split("\nA) ")[1]
        self.assertTrue(alts_val and alts_rev)

    def test_revisor_errado_contra_gabarito_reprova(self):
        q = questao_boa(gab="B")
        r = agentes_com(Fila([revisor_json(so_v("C"), questao=q)])).revisar(q, "9º", "H99", "d")
        self.assertFalse(r["veredito"])
        self.assertEqual(r["resposta_calculada"], "C")  # letra ORIGINAL
        self.assertIn("gabarito_errado", [p["codigo"] for p in r["problemas"]])

    def test_modelo_padrao_do_revisor_e_thinking(self):
        self.assertEqual(aq.MODELOS_PADRAO["revisor"], "sabia-4-thinking")
        args = iq.construir_parser().parse_args(["--dry-run"])
        self.assertEqual(args.modelo_revisor, "sabia-4-thinking")
        sh = (ROOT / "run_base_conhecimento.sh").read_text(encoding="utf-8")
        self.assertIn('MODELO_REVISOR:-sabia-4-thinking', sh)


class TestD5DistratorVerdadeiro(unittest.TestCase):
    def test_alternativa_verdadeira_por_outro_eixo_reprova_no_validador(self):
        """'Quanto aos lados' de um 6-6-7 (acutângulo): 'Acutângulo' é V."""
        q = {"enunciado": "Um triângulo tem lados de 6 cm, 6 cm e 7 cm. Quanto aos lados, ele é:",
             "alternativas": {"A": "Equilátero", "B": "Isósceles", "C": "Escaleno", "D": "Acutângulo",
                              "E": "Obtusângulo"},
             "resolucao_passo_a_passo": "Dois lados iguais: isósceles.", "resposta_correta": "B",
             "difficulty": "MEDIUM"}
        st = {"A": "F", "B": "V", "C": "F", "D": "V", "E": "F"}
        cli = Fila([fase1(q, st, problemas=[{"codigo": "distrator_verdadeiro", "detalhe": "acutângulo"}]),
                    FASE2_OK])
        v = agentes_com(cli).validar(q)
        self.assertFalse(v["veredito"])
        self.assertEqual(cli.n, 1)
        self.assertIsNone(v["resposta_calculada"])
        cods = [p["codigo"] for p in v["problemas"]]
        self.assertIn("gabarito_sem_resposta_unica", cods)
        self.assertIn("distrator_verdadeiro", cods)
        # o revisor pelo mesmo motivo
        r = agentes_com(Fila([revisor_json(st, questao=q)])).revisar(q, "9º", "H17", "d")
        self.assertFalse(r["veredito"])

    def test_classe_mais_geral_e_distrator_verdadeiro_no_revisor(self):
        q = {"enunciado": "Um triângulo tem os três lados iguais. Como se chama esse triângulo?",
             "alternativas": {"A": "Escaleno", "B": "Isósceles", "C": "Retângulo", "D": "Equilátero",
                              "E": "Obtusângulo"},
             "resolucao_passo_a_passo": "Três lados iguais: equilátero.", "resposta_correta": "D",
             "difficulty": "EASY"}
        st = {"A": "F", "B": "G", "C": "F", "D": "V", "E": "F"}
        r = agentes_com(Fila([revisor_json(st, questao=q)])).revisar(q, "5º", "H14", "d")
        self.assertFalse(r["veredito"])
        self.assertEqual(r["resposta_calculada"], "D")
        self.assertIn("distrator_verdadeiro", [p["codigo"] for p in r["problemas"]])

    def test_prompts_trazem_a_regra(self):
        self.assertIn("Distratores ESTRITAMENTE FALSOS", aq.GERADOR_ADDENDUM)
        self.assertIn("Acutângulo", aq.GERADOR_ADDENDUM)
        self.assertIn("por QUALQUER critério", aq.VALIDADOR_SISTEMA)
        self.assertIn("[N9]", aq.VALIDADOR_SISTEMA)
        self.assertIn("distrator_verdadeiro", aq.REVISOR_SISTEMA)
        self.assertIn("C5 CLAREZA", aq.REVISOR_SISTEMA)


class TestT2Contexto(unittest.TestCase):
    def test_contexto_forcado_veta_e_reais_em_centavos_e_filtro(self):
        q = questao_boa(gab="B")
        obj = json.loads(revisor_json(so_v("B"), questao=q, falha="C7"))
        obj["problemas"] = [{"codigo": "contexto_forcado", "detalhe": "234 álbuns de 6 figurinhas"}]
        obj["veredito"] = False
        self.assertFalse(agentes_com(Fila([json.dumps(obj)])).revisar(q, "9º", "H99", "d")["veredito"])
        # C7 falho sozinho (sem código) também veta
        r = agentes_com(Fila([revisor_json(so_v("B"), questao=q, falha="C7", veredito=False)])).revisar(
            q, "9º", "H99", "d")
        self.assertFalse(r["veredito"])
        self.assertIn("C7", r["criterios_bloqueantes_falhos"])
        self.assertIn("C7 CONTEXTO (veta)", aq.REVISOR_SISTEMA)
        ruim = dict(q, enunciado="Ana tinha R$ 4.500 centavos e gastou 35 + 18. Quanto sobrou?")
        obj = {"questoes": [ruim]}
        self.assertEqual(aq.filtro_injecao(obj, json.dumps(obj), ano="9º", habilidade="H99",
                                           dificuldade="Fácil")[0], "contexto_inverossimil")
        self.assertFalse(aq.contexto_inverossimil(questao_boa()))


class TestT1DuplicataSemantica(unittest.TestCase):
    def _q(self, enunciado, resp="Retângulo", outras=("Acutângulo", "Obtusângulo", "Isósceles", "Equilátero")):
        alts = dict(zip("ABCDE", (resp,) + tuple(outras)))
        return {"enunciado": enunciado, "alternativas": alts, "resolucao_passo_a_passo": "Um ângulo de 90°.",
                "resposta_correta": "A", "difficulty": "EASY"}

    def test_mesmo_30_60_90_em_contextos_diferentes(self):
        a = self._q("A rampa da escola forma um triângulo com ângulos de 30°, 60° e 90°. Quanto aos ângulos, ele é:")
        b = self._q("Numa pipa triangular, os ângulos medem 90°, 30° e 60°. Classifique-a quanto aos ângulos.")
        self.assertFalse(__import__("diversidade").e_near_duplicata(a, b))  # o Jaccard não vê
        self.assertTrue(aq.e_duplicata_semantica(a, b))
        c = self._q("Num triângulo, os ângulos medem 40°, 50° e 90°. Quanto aos ângulos, ele é:")
        self.assertFalse(aq.e_duplicata_semantica(a, c))  # outros dados
        d = self._q(a["enunciado"], resp="Escaleno")
        self.assertFalse(aq.e_duplicata_semantica(a, d))  # outra resposta

    def test_numeros_canonicos_e_unidade_ignorada(self):
        a = questao_boa(35, 18, "B", texto="Pedro tinha 35 figurinhas e ganhou 18. Quantas tem?")
        b = dict(questao_boa(35, 18, "C", texto="Uma loja tinha 18 bolas e chegaram 35. Quantas bolas há?"),
                 alternativas={L: v.replace("figurinhas", "bolas") for L, v in
                               questao_boa(35, 18, "C")["alternativas"].items()})
        self.assertTrue(aq.e_duplicata_semantica(a, b))
        self.assertEqual(aq.numeros("R$ 4.500,50 e 2,5 kg"), ("2.5", "4500.5"))
        self.assertIsNone(aq.assinatura_dados({"enunciado": "Quanto é o dobro de 7?", "resposta_correta": "A",
                                               "alternativas": {"A": "14"}}))  # 1 número: não compara

    def test_filtro_contra_base_aceitas_e_val(self):
        a = self._q("A rampa da escola forma um triângulo com ângulos de 30°, 60° e 90°. Quanto aos ângulos, ele é:")
        b = self._q("Numa pipa triangular, os ângulos medem 90°, 30° e 60°. Classifique-a quanto aos ângulos.")
        obj = {"questoes": [b]}
        kw = dict(ano="9º", habilidade="H17", dificuldade="Fácil", taxonomia={"habilidades": {}})
        self.assertEqual(aq.filtro_injecao(obj, json.dumps(obj), comparar_com=[a], **kw)[0], "duplicata_semantica")
        self.assertEqual(aq.filtro_injecao(obj, json.dumps(obj), assinaturas_base={aq.assinatura_dados(a): "MT1"},
                                           **kw)[0], "duplicata_semantica")
        self.assertEqual(aq.filtro_injecao(obj, json.dumps(obj), val_questoes=[a], **kw)[0], "contamina_val")
        self.assertIsNone(aq.filtro_injecao(obj, json.dumps(obj), **kw)[0])

    def test_injecao_barra_parafrase_de_item_da_base(self):
        """Gerador que só parafraseia o item real da base (mesmos 10 + 5)."""
        class Parafraseia(Roteiro):
            def _gerar(self, user):
                q = questao_boa(10, 5, "C", texto="Na excursão foram 10 meninas e 5 meninos; a monitora fez "
                                                  "10 + 5 para contar. Quantas crianças foram?")
                return json.dumps({"questoes": [q]}, ensure_ascii=False)
        amb = Ambiente(falta_facil=1)
        try:
            cli = Parafraseia()
            silencioso(iq.injetar, amb.args_injecao("--tentativas-por-slot", "2"), agentes=agentes_com(cli, 100))
            self.assertEqual(amb.linhas("injecao_saeb.jsonl"), [])
            self.assertEqual({r["motivo"] for r in amb.linhas("rejeitadas.jsonl")}, {"duplicata_semantica"})
            self.assertEqual(cli.chamadas["validador"], 0)  # barrado antes de gastar com os juízes
        finally:
            amb.fechar()


class TestT4RespostaConcentrada(unittest.TestCase):
    def test_limite_e_aviso_ao_professor(self):
        class SoRetangulo(Roteiro):
            """Gera sempre 'Retângulo' como resposta (dados diferentes a cada vez)."""
            CENAS = ["A rampa do pátio da escola", "Uma bandeirola da festa junina", "O telhado da casinha do cachorro",
                     "Uma fatia de pizza cortada pelo padeiro", "A vela do barco de brinquedo",
                     "Uma placa de trânsito na avenida", "O suporte de madeira da estante",
                     "Um pedaço de cartolina recortado", "A pipa que Davi montou"]

            def _gerar(self, user):
                self.n += 1
                a, b, c = 20 + self.n, 70 - self.n, 90
                q = {"enunciado": f"{self.CENAS[self.n % len(self.CENAS)]} tem forma triangular, com ângulos de "
                                  f"{a}°, {b}° e {c}°. Quanto aos ângulos, ele é:",
                     # "Isósceles" e não "Escaleno": com três ângulos diferentes o
                     # triângulo É escaleno, distrator verdadeiro pela D5 — o
                     # pré-filtro P1 de geometria barraria todas as candidatas.
                     "alternativas": {"A": "Retângulo", "B": "Acutângulo", "C": "Obtusângulo", "D": "Isósceles",
                                      "E": "Equilátero"},
                     "resolucao_passo_a_passo": f"Há um ângulo de {c}°, então é retângulo.",
                     "resposta_correta": "A", "difficulty": "EASY"}
                return json.dumps({"questoes": [q]}, ensure_ascii=False)

            def _status(self, user, sal=""):
                alts = self._alternativas(user)
                return {L: ("V" if alts[L] == "Retângulo" else "F") for L in LETRAS}
        amb = Ambiente(falta_facil=7)
        try:
            cli = SoRetangulo()
            cli_msgs = []
            orig = cli.chat_completion

            def grava(messages, max_tokens, temperature):
                cli_msgs.append(messages)
                return orig(messages, max_tokens, temperature)
            cli.chat_completion = grava
            silencioso(iq.injetar, amb.args_injecao("--tentativas-por-slot", "1"), agentes=agentes_com(cli, 200))
            aceitas = amb.linhas("injecao_saeb.jsonl")
            self.assertEqual(len(aceitas), 2)  # limite = max(2, ceil(0,25 x 7)) = 2
            rej = amb.linhas("rejeitadas.jsonl")
            self.assertIn("resposta_concentrada", {r["motivo"] for r in rej})
            ger = [m[1]["content"] for m in cli_msgs if papel_de(m) == "gerador"]
            self.assertNotIn("Retângulo", ger[0])  # 1ª geração: nada a evitar
            self.assertTrue(any('resposta correta é "Retângulo"' in g for g in ger[1:]))
            self.assertNotIn("Retângulo", aceitas[0]["messages"][1]["content"])  # aviso não vai para o treino
        finally:
            amb.fechar()


class TestT3Retomada(unittest.TestCase):
    def test_diario_recupera_aceita_cortada_sem_duplicar_codigo(self):
        amb = Ambiente(falta_facil=3)
        try:
            silencioso(iq.injetar, amb.args_injecao("--meta-por-habilidade", "2"),
                       agentes=agentes_com(Roteiro(), 100))
            saida = amb.d / "injecao_saeb.jsonl"
            linhas = amb.linhas("injecao_saeb.jsonl")
            self.assertEqual(len(linhas), 2)
            self.assertFalse(iq.caminho_pendente(saida).exists())  # diário apagado depois do fsync
            # simula a queda NO MEIO do append da 2ª aceita: o diário ficou inteiro
            linha2 = json.dumps(linhas[1], ensure_ascii=False)
            saida.write_text(json.dumps(linhas[0], ensure_ascii=False) + "\n" + linha2[:57], encoding="utf-8")
            iq.caminho_pendente(saida).write_text(linha2, encoding="utf-8")
            cli = Roteiro()
            silencioso(iq.injetar, amb.args_injecao("--meta-por-habilidade", "2"), agentes=agentes_com(cli, 100))
            depois = amb.linhas("injecao_saeb.jsonl")
            self.assertEqual(depois, linhas)  # a questão paga voltou, inteira e única
            self.assertEqual(cli.total(), 0)  # e não foi gerada de novo
            self.assertFalse(iq.caminho_pendente(saida).exists())
            self.assertEqual(len(amb.linhas("injecao_saeb_truncadas.jsonl")), 1)
            # todas as linhas da saída são JSON válido e terminam em \n
            self.assertTrue(saida.read_text(encoding="utf-8").endswith("\n"))
        finally:
            amb.fechar()

    def test_fragmento_com_codigo_reserva_o_codigo(self):
        with tempfile.TemporaryDirectory() as d:
            saida = Path(d) / "inj.jsonl"
            saida.write_text('{"meta": {"codigo_item": "INJ-9-H99-F-00007", "ano": "9º"', encoding="utf-8")
            rel = iq.reparar_saida(saida)
            self.assertEqual(rel["fragmentos"], 1)
            self.assertEqual(rel["codigos_reservados"], ["INJ-9-H99-F-00007"])
            self.assertEqual(saida.read_text(encoding="utf-8"), "")
            _, _, _, ultimo = iq.estado_previo(saida, Path(d) / "rej.jsonl")
            self.assertEqual(ultimo, 7)
            # diário com uma aceita que JÁ está na saída: só é apagado (sem duplicar)
            ex = {"meta": {"codigo_item": "INJ-9-H99-F-00008"}, "messages": []}
            saida.write_text(json.dumps(ex) + "\n", encoding="utf-8")
            iq.caminho_pendente(saida).write_text(json.dumps(ex), encoding="utf-8")
            rel = iq.reparar_saida(saida)
            self.assertIsNone(rel["recuperada"])
            self.assertEqual(saida.read_text(encoding="utf-8").count("INJ-9-H99-F-00008"), 1)
            self.assertFalse(iq.caminho_pendente(saida).exists())


class TestT5JsonEstrito(unittest.TestCase):
    def test_truncado_logo_apos_resposta_calculada_reprova(self):
        q = questao_boa(gab="B")
        completo = fase1(q, so_v("B"))
        truncado = completo[:completo.index(', "problemas"')]
        for texto in (truncado, truncado + "}", truncado + ', "problemas": [], "confianca": 0.9'):
            self.assertIsNone(aq.interpretar_fase1(texto), texto[-40:])
            v = agentes_com(Fila([texto, FASE2_OK])).validar(q)
            self.assertFalse(v["veredito"])
            self.assertEqual(v["erro"], "veredito_malformado")

    def test_rascunho_completo_mais_resposta_final_truncada_reprova(self):
        """Raciocínio fora das tags com um JSON-rascunho que APROVA, seguido da
        resposta final truncada: antes valia o rascunho."""
        q = questao_boa(gab="B")
        rascunho = fase1(q, so_v("B"))
        final_truncado = fase1(q, so_v("B"), problemas=[{"codigo": "pergunta_ambigua", "detalhe": "x"}])[:200]
        texto = "Rascunho: " + rascunho + "\nResposta final: " + final_truncado
        self.assertIsNone(aq.interpretar_fase1(texto))
        v = agentes_com(Fila([texto, FASE2_OK])).validar(q)
        self.assertFalse(v["veredito"])
        # rascunho + resposta final COMPLETA: vale a final
        final = fase1(q, so_v("B"), problemas=[{"codigo": "pergunta_ambigua", "detalhe": "x"}])
        f1 = aq.interpretar_fase1("Rascunho: " + rascunho + "\nResposta final: " + final)
        self.assertEqual([p["codigo"] for p in f1["problemas"]], ["pergunta_ambigua"])
        # revisor e fase 2 também estritos
        rev = revisor_json(so_v("B"), questao=q)
        self.assertIsNone(aq.interpretar_revisor(rev + " " + rev[:80]))
        self.assertIsNone(aq.interpretar_fase2(FASE2_OK[:-1]))
        self.assertIsNotNone(aq.interpretar_revisor("<think>pensa</think>```json\n" + rev + "\n```"))


# ---------------------------------------------------------------------------
# D2 e D3 na montagem da v3
# ---------------------------------------------------------------------------

def _juiz(veredito, codigos=(), dif=None):
    j = {"veredito": veredito, "avaliado": True, "problemas": [{"codigo": c, "detalhe": "x"} for c in codigos]}
    if dif:
        j["dificuldade_real"] = dif
    return j


def _registro(ex, idx, val, rev, filtros=(), confianca=None):
    m = ex["meta"]
    if confianca is None:
        confianca = ("baixa" if filtros else "alta" if val["veredito"] and rev["veredito"]
                     else "media" if val["veredito"] != rev["veredito"] else "baixa")
    cods = sorted({p["codigo"] for j in (val, rev) for p in j["problemas"]} | {f"filtro:{f}" for f in filtros})
    return {"idx": idx, "codigo_item": m["codigo_item"], "ano": m["ano"], "habilidade": m["habilidade"],
            "origem": ab.origem(m), "hash_questao": ab.hash_exemplo(ex), "versao_prompts": aq.VERSAO_PROMPTS,
            "versao_filtros": aq.VERSAO_FILTROS, "filtros": {"reprovado": list(filtros)},
            "validador": val, "revisor": rev, "vereditos": {"validador": val["veredito"], "revisor": rev["veredito"]},
            "confianca": confianca, "problemas": cods,
            "problemas_matematicos": [c for c in cods if c in aq.CODIGOS_MATEMATICOS or c.startswith("filtro:")],
            "dificuldade_real": [rev.get("dificuldade_real")], "revisao_humana": ab.origem(m) in ab.PROTEGIDAS,
            "rubrica_dificuldade": aq.RUBRICA_DIFICULDADE}


class TestD2D3Montagem(unittest.TestCase):
    def setUp(self):
        self.amb = Ambiente()
        d = self.amb.d
        mt5023 = questao_boa(1728, 722, "A", texto="O mercado recebeu 2.450 caixas de leite e vendeu 1.728. "
                                                     "Quantas caixas restaram?")
        mt5023["alternativas"] = {"A": "732", "B": "722", "C": "4.178", "D": "1.722", "E": "Nenhuma das alternativas anteriores"}
        mt5023["resolucao_passo_a_passo"] = "2.450 - 1.728 = 732."
        vazia = dict(questao_boa(12, 3, "B", texto="Uma caixa tem 12 lápis e Ana pôs mais 3; ela fez 12 + 3. "
                                                    "Quantos lápis há?"), resolucao_passo_a_passo="")
        mod_facil = questao_boa(14, 6, "C", dif="MEDIUM", texto="Bia tinha 14 selos e ganhou 6; fez 14 + 6. "
                                                                  "Com quantos selos ficou?")
        dif_mod = questao_boa(25, 4, "D", dif="HARD", texto="Caio leu 25 páginas e depois mais 4; fez 25 + 4. "
                                                              "Quantas páginas leu?")
        dist_mod = questao_boa(31, 2, "E", dif="MEDIUM", texto="A horta tem 31 pés de alface e foram plantados mais 2; "
                                                                "a turma fez 31 + 2. Quantos pés há?")
        self.exs = [
            exemplo("5º", "H05", mt5023, "MT5023MH05MT", dificuldade="Moderado"),       # 0 real com gabarito errado
            exemplo("9º", "H06", vazia, "MT9049DH10MT", dificuldade="Difícil"),          # 1 real com resolução vazia
            # 2 destilado fácil -> rebaixa só com --regra-d3 revisor (passo 2: real nunca é rebaixado
            # automaticamente e a regra padrão é "suspensa")
            exemplo("9º", "H99", mod_facil, "DIST-H99-Moderado-00001", dificuldade="Moderado", destilado=True,
                    professor="p"),
            exemplo("9º", "H99", dif_mod, "MT9002DH99MT", dificuldade="Difícil"),        # 3 revisor: Moderado -> nada
            exemplo("9º", "H99", dist_mod, "DIST-H99-Moderado-00003", dificuldade="Moderado", destilado=True,
                    professor="p"),                                                       # 4 val ok, rev reprova
            exemplo("9º", "H99", questao_boa(9, 9, "A", texto="Uma fila tem 9 + 9 pessoas. Quantas são?"),
                    "MT9003FH99MT", dificuldade="Fácil"),                               # 5 já é Fácil
            exemplo("5º", "H05", questao_boa(50, 8, "B", texto="Dona Rosa vendeu 50 bolos e 8 tortas; fez 50 + 8. "
                                                                "Quantos doces vendeu?"), "MT5099DH05MT",
                    dificuldade="Difícil"),                                              # 6 val reprova só por dados
        ]
        self.amb.train.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in self.exs),
                                  encoding="utf-8")
        regs = [
            _registro(self.exs[0], 0, _juiz(False, ["gabarito_errado"]), _juiz(False, ["gabarito_errado"], "Fácil")),
            _registro(self.exs[1], 1, _juiz(False, ["resolucao_errada", "resolucao_nao_conclui"]),
                      _juiz(True, dif="Fácil"), filtros=["resolucao_vazia"]),
            _registro(self.exs[2], 2, _juiz(True), _juiz(True, dif="Fácil")),
            _registro(self.exs[3], 3, _juiz(True), _juiz(True, dif="Moderado")),
            _registro(self.exs[4], 4, _juiz(True), _juiz(False, ["distrator_implausivel"], "Fácil")),
            _registro(self.exs[5], 5, _juiz(True), _juiz(True, dif="Fácil")),
            _registro(self.exs[6], 6, _juiz(False, ["gabarito_sem_resposta_unica", "dados_insuficientes"]),
                      _juiz(False, ["dados_insuficientes"], "Difícil")),
        ]
        (d / "auditoria.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in regs),
                                           encoding="utf-8")
        # H4: o árbitro de outra família também resolveu o MT5023 e achou 722 (B)
        arb = {"idx": 0, "codigo_item": "MT5023MH05MT", "hash_questao": ab.hash_exemplo(self.exs[0]),
               "modelo": "simulado", "resultados": [{"avaliado": True, "veredito": False, "resposta_calculada": "B",
                                                     "problemas": [{"codigo": "gabarito_errado", "detalhe": "x"}]}]}
        (d / "arbitragem_d2.jsonl").write_text(json.dumps(arb, ensure_ascii=False) + "\n", encoding="utf-8")
        # o MECANISMO da D3 é testado com a regra explícita "revisor"; a regra
        # padrão ("suspensa", passo 2) tem teste próprio abaixo
        self.args = self.amb.args_auditoria("--montar", "--removidos", str(d / "removidos.jsonl"),
                                            "--rotulos", str(d / "rotulos.jsonl"),
                                            "--resolucao-vazia", str(d / "vazia.jsonl"), "--regra-d3", "revisor")
        self.rel = silencioso(ab.montar, self.args)
        self.v3 = {e["meta"]["codigo_item"]: e for e in self.amb.linhas("v3.jsonl")}

    def tearDown(self):
        self.amb.fechar()

    def test_d2_remove_erro_confirmado_inclusive_real_e_grava_lista(self):
        self.assertNotIn("MT5023MH05MT", self.v3)
        rem = {r["codigo_item"]: r for r in self.amb.linhas("removidos.jsonl")}
        self.assertEqual(set(rem), {"MT5023MH05MT"})
        self.assertEqual(rem["MT5023MH05MT"]["motivo"], "erro_matematico_confirmado_arbitro")
        self.assertEqual(rem["MT5023MH05MT"]["evidencia"]["arbitro"]["codigos"], ["gabarito_errado"])
        self.assertEqual(rem["MT5023MH05MT"]["evidencia"]["validador"], ["gabarito_errado"])
        self.assertEqual(rem["MT5023MH05MT"]["origem"], "real")
        self.assertEqual(self.rel["n_removidos_erro_matematico"], 1)

    def test_d2_resolucao_vazia_nao_remove_e_e_listada(self):
        self.assertIn("MT9049DH10MT", self.v3)
        vaz = self.amb.linhas("vazia.jsonl")
        self.assertEqual([r["codigo_item"] for r in vaz], ["MT9049DH10MT"])
        self.assertEqual(vaz[0]["acao"], "manter")
        # mesmo com o revisor também reprovando pela resolução, não é erro matemático
        reg = _registro(self.exs[1], 1, _juiz(False, ["resolucao_errada"]), _juiz(False, ["resolucao_errada"]),
                        filtros=["resolucao_vazia"])
        self.assertFalse(ab.erro_matematico_confirmado(reg)[0])
        self.assertEqual(ab.decidir("real", reg, remover_reais_baixa=True)[0], "manter")

    def test_d2_exige_os_dois_sinais(self):
        so_val = _registro(self.exs[0], 0, _juiz(False, ["gabarito_errado"]), _juiz(True))
        self.assertFalse(ab.erro_matematico_confirmado(so_val)[0])
        self.assertEqual(ab.decidir("real", so_val)[0], "manter")
        com_filtro = _registro(self.exs[0], 0, _juiz(False, ["gabarito_errado"]), _juiz(True),
                               filtros=["consistencia"])
        self.assertTrue(ab.erro_matematico_confirmado(com_filtro)[0])
        so_rev = _registro(self.exs[0], 0, _juiz(True), _juiz(False, ["gabarito_errado"]))
        self.assertFalse(ab.erro_matematico_confirmado(so_rev)[0])
        # incompleto (dados ausentes) não é errado: fica, vai para revisão humana
        self.assertIn("MT5099DH05MT", self.v3)
        pedag = _registro(self.exs[0], 0, _juiz(False, ["gabarito_errado"]), _juiz(False, ["contexto_forcado"]))
        self.assertFalse(ab.erro_matematico_confirmado(pedag)[0])

    def test_d3_rebaixa_nos_tres_lugares_e_nada_mais(self):
        antes = self.exs[2]
        depois = self.v3["DIST-H99-Moderado-00001"]
        self.assertEqual(depois["meta"]["dificuldade"], "Fácil")
        q = json.loads(depois["messages"][2]["content"])["questoes"][0]
        self.assertEqual(q["difficulty"], "EASY")
        self.assertIn("Dificuldade: Fácil.", depois["messages"][1]["content"])
        self.assertNotIn("Moderado", depois["messages"][1]["content"])
        # nada além dos três lugares (+ o registro rotulo_corrigido na meta)
        self.assertEqual(depois["messages"][0], antes["messages"][0])
        self.assertEqual(depois["messages"][1]["content"],
                         antes["messages"][1]["content"].replace("Dificuldade: Moderado.", "Dificuldade: Fácil."))
        q_antes = json.loads(antes["messages"][2]["content"])["questoes"][0]
        self.assertEqual(q, dict(q_antes, difficulty="EASY"))
        self.assertEqual({k: v for k, v in depois["meta"].items() if k not in ("dificuldade", "rotulo_corrigido")},
                         {k: v for k, v in antes["meta"].items() if k != "dificuldade"})
        self.assertEqual(depois["meta"]["rotulo_corrigido"]["de"], "Moderado")
        rot = {r["codigo_item"]: r for r in self.amb.linhas("rotulos.jsonl")}
        self.assertEqual(set(rot), {"DIST-H99-Moderado-00001"})
        self.assertEqual((rot["DIST-H99-Moderado-00001"]["de"], rot["DIST-H99-Moderado-00001"]["para"]),
                         ("Moderado", "Fácil"))

    def test_d3_padrao_suspensa_e_real_nunca_rebaixa(self):
        # Passo 2 (2026-10-01): sem --regra-d3 nada é rebaixado (o revisor sem âncora
        # disse Fácil para 83% dos M/D reais do banco)
        d = self.amb.d
        args = self.amb.args_auditoria("--montar", "--removidos", str(d / "rem2.jsonl"), "--rotulos",
                                       str(d / "rot2.jsonl"), "--resolucao-vazia", str(d / "vaz2.jsonl"),
                                       "--saida-v3", str(d / "v3b.jsonl"))
        rel = silencioso(ab.montar, args)
        self.assertEqual(rel["n_rotulos_corrigidos"], 0)
        self.assertIn("suspensa", rel["regras"]["D3"])
        v3 = {e["meta"]["codigo_item"]: e for e in self.amb.linhas("v3b.jsonl")}
        self.assertEqual(v3["DIST-H99-Moderado-00001"], self.exs[2])
        # com a regra "revisor", o mesmo registro num item REAL não rebaixa (rótulo do banco é a referência)
        real = dict(self.exs[2], meta=dict(self.exs[2]["meta"], codigo_item="MT9001MH99MT", destilado=False))
        reg = _registro(real, 2, _juiz(True), _juiz(True, dif="Fácil"))
        self.assertFalse(ab.deve_rebaixar_para_facil(reg, real["meta"], "revisor"))
        reg_dist = _registro(self.exs[2], 2, _juiz(True), _juiz(True, dif="Fácil"))
        self.assertTrue(ab.deve_rebaixar_para_facil(reg_dist, self.exs[2]["meta"], "revisor"))
        self.assertFalse(ab.deve_rebaixar_para_facil(reg_dist, self.exs[2]["meta"]))
        with self.assertRaises(ValueError):
            ab.deve_rebaixar_para_facil(reg_dist, self.exs[2]["meta"], "revisor_e_arbitro")

    def test_d3_so_rebaixa_para_facil(self):
        # Difícil julgado Moderado: nada muda (não reclassifica em outra direção)
        self.assertEqual(self.v3["MT9002DH99MT"]["meta"]["dificuldade"], "Difícil")
        self.assertEqual(self.v3["MT9002DH99MT"], self.exs[3])
        # Fácil continua Fácil e intacto
        self.assertEqual(self.v3["MT9003FH99MT"], self.exs[5])
        # revisor reprovou (pedagógico): matemática não confirmada pelos dois -> não rebaixa
        self.assertEqual(self.v3["DIST-H99-Moderado-00003"]["meta"]["dificuldade"], "Moderado")
        # resolução vazia com validador reprovando: não rebaixa
        self.assertEqual(self.v3["MT9049DH10MT"]["meta"]["dificuldade"], "Difícil")
        # registro antigo sem dificuldade_real nunca rebaixa
        reg = _registro(self.exs[2], 2, _juiz(True), _juiz(True))
        self.assertFalse(ab.deve_rebaixar_para_facil(reg, self.exs[2]["meta"], "revisor"))
        # prompt sem "Dificuldade: X." único: não aplica (rótulo inconsistente é pior)
        ex = json.loads(json.dumps(self.exs[2]))
        ex["messages"][1]["content"] = ex["messages"][1]["content"].replace("Dificuldade: Moderado.", "")
        self.assertEqual(ab.rebaixar_para_facil(ex), (None, "prompt_sem_dificuldade_unica"))
        self.assertEqual(ab.rebaixar_para_facil(self.exs[5]), (None, "so_rebaixa_moderado_ou_dificil"))

    def test_v3_nao_toca_a_base(self):
        self.assertEqual(self.rel["n_base"], 7)
        self.assertEqual(self.amb.train.read_text(encoding="utf-8").count("Dificuldade: Moderado."), 3)


class TestD3Injecao(unittest.TestCase):
    """Pedida Moderado, julgada Fácil pelos dois juízes aprovando: entra como
    Fácil (três lugares) ocupando um slot Fácil; o slot Moderado continua."""

    def _ambiente(self, falta):
        amb = Ambiente()
        rel = json.loads(amb.relatorio.read_text(encoding="utf-8"))
        h = rel["habilidades"][0]
        h["falta_por_dificuldade"] = falta
        h["n_efetivas"] = 30 - sum(falta.values())
        amb.relatorio.write_text(json.dumps(rel), encoding="utf-8")
        return amb

    class FacilSempre(Roteiro):
        def chat_completion(self, messages, max_tokens, temperature):
            resp = super().chat_completion(messages, max_tokens, temperature)
            if papel_de(messages) == "revisor":
                o = aq.extrair_json(resp.choices[0].message.content, ("criterios",), estrito=True)
                o["dificuldade_real"] = "Fácil"
                return self._resp(json.dumps(o, ensure_ascii=False), messages)
            return resp

    def test_d3_suspensa_entra_com_a_dificuldade_pedida(self):
        # Passo 2 (2026-10-01): regra padrão "suspensa" — o revisor dizer Fácil não
        # rebaixa nem rejeita; a dificuldade_real fica só registrada na meta.
        amb = self._ambiente({"Fácil": 1, "Moderado": 2, "Difícil": 0})
        try:
            res = silencioso(iq.injetar, amb.args_injecao(), agentes=agentes_com(self.FacilSempre(), 100))
            aceitas = amb.linhas("injecao_saeb.jsonl")
            self.assertEqual(sorted(e["meta"]["dificuldade"] for e in aceitas), ["Fácil", "Moderado", "Moderado"])
            for e in aceitas:
                self.assertEqual(e["meta"]["dificuldade"], e["meta"]["dificuldade_pedida"])
                self.assertEqual(e["meta"]["dificuldade_real"], "Fácil")
                self.assertNotIn("rotulo_corrigido", e["meta"])
            self.assertEqual(res["funil"]["aceitos_rebaixados_para_facil"], 0)
        finally:
            amb.fechar()

    def test_rebaixada_entra_como_facil_consistente(self):
        import distill_teacher as dt
        amb = self._ambiente({"Fácil": 1, "Moderado": 2, "Difícil": 0})
        try:
            res = silencioso(iq.injetar, amb.args_injecao("--regra-d3", "revisor"),
                             agentes=agentes_com(self.FacilSempre(), 100))
            aceitas = amb.linhas("injecao_saeb.jsonl")
            self.assertEqual(len(aceitas), 1, res["motivos"])
            ex = aceitas[0]
            m = ex["meta"]
            self.assertEqual(m["dificuldade"], "Fácil")
            self.assertEqual(m["dificuldade_pedida"], "Moderado")
            self.assertEqual(m["rotulo_corrigido"]["para"], "Fácil")
            self.assertTrue(m["codigo_item"].startswith("INJ-9-H99-F-"))
            q = json.loads(ex["messages"][2]["content"])["questoes"][0]
            self.assertEqual(q["difficulty"], "EASY")
            self.assertEqual(ex["messages"][1]["content"],
                             dt.prompt_usuario("9º", "H99", "Resolver problemas de adição.", "Fácil", None))
            self.assertEqual(res["funil"]["aceitos_rebaixados_para_facil"], 1)
            # o slot Moderado continuou aberto: as tentativas seguintes pediram Moderado
            # e, sem slot Fácil livre, foram rejeitadas (nunca entram rotuladas Moderado)
            rej = amb.linhas("rejeitadas.jsonl")
            self.assertTrue(rej)
            self.assertTrue(all(r["dificuldade"] == "Moderado" for r in rej))
            self.assertEqual({r["motivo"] for r in rej}, {"dificuldade_real_facil_sem_lacuna"})
            # e no v3 a injetada aparece na lista de rótulos corrigidos
            silencioso(ab.montar, amb.args_auditoria("--montar", "--rotulos", str(amb.d / "rot.jsonl"),
                                                     "--removidos", str(amb.d / "rem.jsonl"),
                                                     "--resolucao-vazia", str(amb.d / "vaz.jsonl")))
            rot = amb.linhas("rot.jsonl")
            self.assertEqual([(r["fonte"], r["de"], r["para"]) for r in rot], [("injecao", "Moderado", "Fácil")])
        finally:
            amb.fechar()

    def test_construir_exemplo_nao_reclassifica_para_cima(self):
        lac = {"ano": "9º", "habilidade": "H99"}
        juizo = {"validador": {"confianca": 0.9}, "revisor": {"confianca": 0.9, "dificuldade_real": "Difícil"}}
        with self.assertRaises(ValueError):
            iq.construir_exemplo(lac, "d", "Difícil", questao_boa(), None, 1, juizo, {"gerador": "g"},
                                 dificuldade_pedida="Moderado")


class _HexagonoAprova(Roteiro):
    """Juízes que aprovam o hexágono (o que aconteceu no piloto real)."""

    def chat_completion(self, messages, max_tokens, temperature):
        u = messages[-1]["content"]
        if "hexagonal" not in u:
            return super().chat_completion(messages, max_tokens, temperature)
        papel = papel_de(messages)
        self.chamadas[papel] += 1
        q = questao_hexagono()
        if papel == "validador":
            return self._resp(fase1(q, so_v("B")), messages)
        if papel == "validador_fase2":
            return self._resp(FASE2_OK, messages)
        return self._resp(revisor_json(so_v("B"), questao=q), messages)



# ===========================================================================
# Piloto 2 (2026-10-01): reauditoria da mesma amostra, cache dos juízes,
# prévia da montagem e meta relativa na injeção.
# ===========================================================================
class TestPiloto2Opcoes(unittest.TestCase):
    def test_testes_nao_escrevem_listas_reais(self):
        # A montagem dos testes grava as listas no diretório temporário (antes
        # gravava em outputs/agentes/removidos_v3.jsonl de verdade).
        amb = Ambiente()
        try:
            args = amb.args_auditoria("--montar")
            p = ab.caminhos(args)
            for k in ("removidos", "rotulos", "resolucao_vazia"):
                self.assertEqual(p[k].parent, amb.d)
        finally:
            amb.fechar()

    def test_cache_dos_juizes_nao_gasta_e_gerador_nunca_usa(self):
        with tempfile.TemporaryDirectory() as t:
            cache_path = Path(t) / "cache.jsonl"
            q = questao_boa()
            cli = Roteiro()
            ag = agentes_com(cli, 10)
            ag.cache_juizes = aq.CacheRespostas([cache_path])
            j1 = ag.julgar(q, "9º", "H99", "d", None, "Fácil", curto_circuito=False)
            n1 = cli.total()
            self.assertEqual(ag.orcamento.chamadas, n1)
            self.assertEqual(ag.cache_juizes.resumo()["respostas_novas_gravadas"], n1)
            # 2ª vez, cache relido do disco: 0 chamadas, mesmo veredito
            cli2 = Roteiro()
            ag2 = agentes_com(cli2, 0)  # orçamento ZERO: qualquer envio levantaria
            ag2.cache_juizes = aq.CacheRespostas([cache_path])
            j2 = ag2.julgar(q, "9º", "H99", "d", None, "Fácil", curto_circuito=False)
            self.assertEqual(cli2.total(), 0)
            self.assertEqual(j1["ambos"], j2["ambos"])
            self.assertEqual(ag2.cache_juizes.resumo()["acertos"], n1)
            # outro contexto do revisor (outra habilidade) = outra mensagem = chamada nova
            ag3 = agentes_com(Roteiro(), 5)
            ag3.cache_juizes = aq.CacheRespostas([cache_path])
            ag3.julgar(q, "9º", "H98", "d", None, "Fácil", curto_circuito=False)
            self.assertEqual(ag3.cache_juizes.resumo()["acertos_por_papel"].get("revisor", 0), 0)
            self.assertGreater(ag3.cache_juizes.resumo()["acertos_por_papel"].get("validador", 0), 0)
            # o gerador nunca passa pelo cache
            cli4 = Roteiro()
            ag4 = agentes_com(cli4, 10)
            ag4.cache_juizes = aq.CacheRespostas([cache_path])
            ag4.gerar("9º", "H99", "d", "Fácil")
            ag4.gerar("9º", "H99", "d", "Fácil")
            self.assertEqual(cli4.chamadas["gerador"], 2)

    def test_chave_igual_a_da_calibracao(self):
        import calibrar_agentes as ca
        msgs = [{"role": "user", "content": "x"}]
        self.assertEqual(ca._chave_cache("validador", "m", msgs), aq.chave_cache("validador", "m", msgs))

    def test_reauditoria_dos_idx_registrados_com_cache(self):
        amb = Ambiente()
        try:
            # 1ª auditoria só do idx 1 e 2 (registros "antigos")
            silencioso(ab.auditar, amb.args_auditoria(), agentes=agentes_com(Roteiro(), 100))
            regs = amb.linhas("auditoria.jsonl")
            antigos = [dict(r, versao_prompts="antiga") for r in regs if r["idx"] in (1, 2)]
            (amb.d / "auditoria.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                           for r in antigos), encoding="utf-8")
            cache = amb.d / "juizes_cache.jsonl"
            cli = Roteiro()
            res = silencioso(ab.auditar, amb.args_auditoria("--idx-registrados", "--cache-juizes", str(cache)),
                             agentes=agentes_com(cli, 100))
            novos = [r for r in amb.linhas("auditoria.jsonl") if r["versao_prompts"] == aq.VERSAO_PROMPTS]
            self.assertEqual(sorted(r["idx"] for r in novos), [1, 2])
            self.assertGreater(cli.total(), 0)
            self.assertIn("cache_juizes", res)
            # de novo com o registro apagado: tudo vem do cache, 0 chamadas
            (amb.d / "auditoria.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                           for r in antigos), encoding="utf-8")
            cli2 = Roteiro()
            silencioso(ab.auditar, amb.args_auditoria("--idx-registrados", "--cache-juizes", str(cache)),
                       agentes=agentes_com(cli2, 100))
            self.assertEqual(cli2.total(), 0)
        finally:
            amb.fechar()

    def test_previa_nao_escreve_v3(self):
        amb = Ambiente()
        try:
            silencioso(ab.auditar, amb.args_auditoria(), agentes=agentes_com(Roteiro(), 100))
            silencioso(ab.arbitrar, amb.args_auditoria("--arbitrar"), arbitro=arbitro_com(Roteiro()))
            args = amb.args_auditoria("--montar", "--previa")
            args.montagem = None
            args.revisao_humana = None
            rel = silencioso(ab.montar, args)
            self.assertFalse((amb.d / "v3.jsonl").exists())
            self.assertTrue(rel["previa"])
            self.assertIsNone(rel["saida"])
            self.assertTrue((amb.d / "montagem_v3_previa.json").exists())
            self.assertTrue((amb.d / "revisao_humana_previa.jsonl").exists())
            rem = amb.linhas("removidos_v3.jsonl")
            self.assertEqual([r["codigo_item"] for r in rem], ["DIST-H99-Fácil-00002"])
        finally:
            amb.fechar()

    def test_meta_relativa_soma_as_aceitas_previas(self):
        amb = Ambiente(falta_facil=6)
        try:
            silencioso(iq.injetar, amb.args_injecao("--meta-por-habilidade", "2"), agentes=agentes_com(Roteiro(), 100))
            self.assertEqual(len(amb.linhas("injecao_saeb.jsonl")), 2)
            # sem --meta-relativa a meta 2 já está cumprida: 0 chamadas
            cli = Roteiro()
            silencioso(iq.injetar, amb.args_injecao("--meta-por-habilidade", "2"), agentes=agentes_com(cli, 100))
            self.assertEqual(cli.total(), 0)
            # com --meta-relativa: +2 a partir das 2 que já existem
            res = silencioso(iq.injetar, amb.args_injecao("--meta-por-habilidade", "2", "--meta-relativa"),
                             agentes=agentes_com(Roteiro(), 100))
            self.assertEqual(len(amb.linhas("injecao_saeb.jsonl")), 4)
            ph = res["por_habilidade"]["9º H99"]
            self.assertEqual(ph["aceitas_agora"], 2)
            self.assertEqual(ph["aceitas_por_dificuldade_pedida"], {"Fácil": 2})
            # o simulador repete questões: parte dos candidatos cai na duplicata
            self.assertEqual(set(ph["candidatos_por_dificuldade_pedida"]), {"Fácil"})
            self.assertGreaterEqual(ph["candidatos_por_dificuldade_pedida"]["Fácil"], 2)
            self.assertEqual(res["funil"]["candidatos"], ph["candidatos_por_dificuldade_pedida"]["Fácil"])
            codigos = [e["meta"]["codigo_item"] for e in amb.linhas("injecao_saeb.jsonl")]
            self.assertEqual(len(set(codigos)), 4)
        finally:
            amb.fechar()


class TestMotivoRevisor(unittest.TestCase):
    def test_motivo_e_o_primeiro_problema_que_veta(self):
        rev = {"problemas": [{"codigo": "dificuldade_incoerente"}, {"codigo": "incoerencia_interna"},
                             {"codigo": "resposta_nao_unica"}], "criterios_bloqueantes_falhos": []}
        self.assertEqual(iq.motivo_revisor(rev), "resposta_nao_unica")
        self.assertEqual(iq.motivo_revisor({"problemas": [], "criterios_bloqueantes_falhos": ["C7"]}), "criterio_C7")
        self.assertEqual(iq.motivo_revisor({"problemas": [{"codigo": "incoerencia_interna"}]}), "incoerencia_interna")
        self.assertEqual(iq.motivo_revisor({}), "reprovado")


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# Revisão do piloto 2 (auditoria independente, 2026-10-01): achados altos.
# Cada teste falha no código anterior (cópia em /tmp/claude-1000/base2/
# backup_fix/) e passa no atual.
# ===========================================================================
def q_dias(gab="A", texto=None):
    return {"enunciado": texto or ("A escola de Lucas organizou uma gincana que começou em uma terça-feira e "
                                   "durou exatamente 10 dias seguidos, sem interrupções. Em qual dia da "
                                   "semana terminou a gincana?"),
            "alternativas": {"A": "Quinta-feira", "B": "Quarta-feira", "C": "Sexta-feira", "D": "Sábado",
                             "E": "Domingo"},
            "resolucao_passo_a_passo": "Dia 1 terça ... dia 10 quinta.", "resposta_correta": gab,
            "difficulty": "MEDIUM"}


def q_dizima(gab="A", alts=None):
    return {"enunciado": "Em uma biblioteca escolar, 5 de cada 11 livros emprestados no mês são de ficção "
                         "científica. Qual é a representação decimal correspondente a essa fração?",
            "alternativas": alts or {"A": "0,45", "B": "0,55", "C": "0,511", "D": "0,227", "E": "0,50"},
            "resolucao_passo_a_passo": "5 ÷ 11 = 0,4545...", "resposta_correta": gab, "difficulty": "HARD"}


def q_decomp(gab="C"):
    return {"enunciado": "LUCAS TEM 245 BOLAS DE GUDE. QUAL É OUTRA FORMA DE REPRESENTAR A QUANTIDADE DE "
                         "BOLAS DE GUDE QUE LUCAS POSSUI?",
            "alternativas": {"A": "150 + 95", "B": "180 + 75", "C": "100 + 145", "D": "250 + 45",
                             "E": "Nenhuma das alternativas anteriores"},
            "resolucao_passo_a_passo": "100 + 145 = 245.", "resposta_correta": gab, "difficulty": "MEDIUM"}


def _com_aritmetica(reg, q):
    reg = json.loads(json.dumps(reg))
    reg["filtros"]["aritmetica"] = aq.verificar_aritmetica(q)
    if reg["filtros"]["aritmetica"]["status"] == "contradiz":
        reg["filtros"]["reprovado"] = reg["filtros"]["reprovado"] + ["verificador_aritmetico"]
        reg["confianca"] = "baixa"
    return reg


class TestRevisaoPiloto2(unittest.TestCase):
    # --- validador: G reprova (D5) -------------------------------------------
    def test_validador_com_status_g_reprova(self):
        q = questao_boa()
        st = dict(so_v("B"), A="G")
        cli = Fila([fase1(q, st), FASE2_OK])
        v = agentes_com(cli).validar(q)
        self.assertFalse(v["veredito"])
        self.assertIn("distrator_verdadeiro", [p["codigo"] for p in v["problemas"]])
        self.assertEqual(cli.n, 1)  # reprovou sem gastar a fase 2
        # e a D2 passa a poder confirmar esse erro
        reg = _registro(exemplo("9º", "H17", q, "X"), 0, {**v, "avaliado": True},
                        _juiz(False, ["distrator_verdadeiro"]))
        self.assertTrue(ab.erro_matematico_confirmado(reg)[0])

    # --- cache: malformado não entra nem sai ---------------------------------
    def test_cache_nao_guarda_nem_serve_malformado(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "cache.jsonl"
            msgs = [{"role": "user", "content": "x"}]
            ruim = '{"veredito": true, "resposta_calculada": "12"'
            c = aq.CacheRespostas([path])
            c.guardar("validador", "m", msgs, ruim, {})
            self.assertEqual(c.resumo()["malformadas_nao_gravadas"], 1)
            self.assertIsNone(aq.CacheRespostas([path]).obter("validador", "m", msgs))
            # resposta malformada gravada ANTES desta regra (o cache da calibração tem algumas)
            path.write_text(json.dumps({"chave": aq.chave_cache("validador", "m", msgs), "papel": "validador",
                                        "modelo": "m", "texto": ruim}) + "\n", encoding="utf-8")
            c2 = aq.CacheRespostas([path])
            self.assertIsNone(c2.obter("validador", "m", msgs))
            self.assertEqual(c2.resumo()["malformadas_ignoradas"], 1)

    def test_malformado_em_cache_e_refeito_na_rodada_seguinte(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "cache.jsonl"
            q = questao_boa()
            ag = agentes_com(Roteiro(validador="malformado"), 10)
            ag.cache_juizes = aq.CacheRespostas([path])
            self.assertEqual(ag.validar(q)["erro"], "veredito_malformado")
            cli = Roteiro()
            ag2 = agentes_com(cli, 10)
            ag2.cache_juizes = aq.CacheRespostas([path])
            v = ag2.validar(q)
            self.assertEqual(cli.chamadas["validador"], 1)  # refez de verdade
            self.assertTrue(v["avaliado"])

    def test_cache_da_calibracao_ignora_malformado(self):
        import calibrar_agentes as ca
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            q = questao_boa()
            perm_user = aq.VALIDADOR_USUARIO.format(enunciado=q["enunciado"],
                                                    **{L: q["alternativas"][aq.permutacao(q)[L]] for L in LETRAS})
            msgs = [{"role": "system", "content": aq.VALIDADOR_SISTEMA}, {"role": "user", "content": perm_user}]
            modelo = aq.MODELOS_PADRAO["validador"]
            (d / "cache.jsonl").write_text(json.dumps({"chave": aq.chave_cache("validador", modelo, msgs),
                                                       "papel": "validador", "modelo": modelo,
                                                       "texto": '{"alternativas": {"A": "V"'}) + "\n",
                                           encoding="utf-8")
            cli = Roteiro()
            ag = ca.AgentesComCache({p: cli for p in aq.PAPEIS}, orcamento=aq.Orcamento(10), log_uso=None,
                                    dormir=lambda s: None, cache_path=d / "cache.jsonl",
                                    ledger=ca.Ledger(d / "ledger.json", 10), rodada="t")
            self.assertTrue(ag.validar(q)["avaliado"])
            self.assertEqual(cli.chamadas["validador"], 1)

    # --- verificador aritmético exato -----------------------------------------
    def test_verificador_aritmetico(self):
        v = aq.verificar_aritmetica
        self.assertEqual(v(q_dias("A"))["status"], "confirma")      # terça + 10 dias = quinta (idx 881)
        self.assertEqual(v(q_dias("C"))["status"], "contradiz")     # "sexta" dos dois juízes
        self.assertEqual(v(q_dias("A", "A colheita foi numa quarta-feira. O plantio é sempre 4 dias depois. "
                                       "Em que dia da semana será o plantio?"))["esperado"], "domingo")
        self.assertEqual(v(q_dias("A", "Ela preparou a massa numa segunda-feira e precisa descansar por 3 "
                                       "dias. Em que dia vai assar?"))["status"], "sem_veredito")  # ambíguo
        # H2 (decisão do usuário): 5/11 = 0,4545... e "0,45" (duas casas) é
        # correto no ensino básico (idx 1348)
        self.assertEqual(v(q_dizima("A"))["status"], "confirma")
        self.assertEqual(v(q_dizima("B"))["status"], "contradiz")   # "0,55" não é 5/11
        self.assertEqual(v(q_dizima("A", {"A": "0,4545...", "B": "0,55", "C": "0,511", "D": "0,227",
                                          "E": "0,50"}))["status"], "confirma")
        tres_quartos = dict(q_dizima("C"), enunciado="Qual é o número decimal equivalente à fração 3/4?",
                            alternativas={"A": "0.25", "B": "0.5", "C": "0.75", "D": "1.25",
                                          "E": "Nenhuma das alternativas anteriores"})
        self.assertEqual(v(tres_quartos)["status"], "confirma")
        r = v(q_decomp("C"))                                         # A e C dão 245 (idx 561)
        self.assertEqual((r["status"], r["verdadeiras"]), ("contradiz", ["A", "C"]))
        cdu = dict(q_decomp("E"), enunciado="Um jogo custa 247 reais. Qual opção mostra uma decomposição em "
                                            "centenas, dezenas e unidades?",
                   alternativas={"A": "20 + 40 + 7", "B": "200 + 4 + 7", "C": "240 + 7", "D": "200 + 47",
                                 "E": "200 + 40 + 7"})
        self.assertEqual(v(cdu)["status"], "sem_veredito")          # a forma restringe: não decide
        self.assertEqual(v(questao_boa())["status"], "sem_veredito")
        obj = {"questoes": [q_dias("C")]}
        motivo, det = aq.filtro_injecao(obj, json.dumps(obj, ensure_ascii=False), ano="2º", habilidade="H18",
                                        dificuldade="Moderado")
        self.assertEqual(motivo, "verificador_aritmetico")
        self.assertIn("verificador_aritmetico", aq.filtros_auditoria(q_dizima("B"))["reprovado"])
        self.assertNotIn("verificador_aritmetico", aq.filtros_auditoria(q_dizima("A"))["reprovado"])

    # --- D2: critério ----------------------------------------------------------
    def test_d2_verificador_que_confirma_impede_remocao_por_erro_conjunto(self):
        ex = exemplo("2º", "H18", q_dias("A"), "DIST-H18-Moderado-00102", dificuldade="Moderado",
                     destilado=True, professor="p")
        reg = _com_aritmetica(_registro(ex, 881, _juiz(False, ["gabarito_errado"]),
                                        _juiz(False, ["gabarito_errado"])), q_dias("A"))
        self.assertFalse(ab.erro_matematico_confirmado(reg)[0])
        self.assertEqual(ab.decidir("destilado", reg), ("manter", "verificador_confirma_gabarito_revisao_humana"))

    def test_d2_verificador_que_contradiz_confirma_sozinho(self):
        # gabarito "0,55" para 5/11 (o 1348 real, "0,45", é correto pela H2)
        ex = exemplo("9º", "H09", q_dizima("B"), "DIST-H09-Difícil-00569", dificuldade="Difícil",
                     destilado=True, professor="p")
        reg = _com_aritmetica(_registro(ex, 1348, _juiz(True), _juiz(True)), q_dizima("B"))
        conf, evid = ab.erro_matematico_confirmado(reg)
        self.assertTrue(conf)
        self.assertIn("5/11", evid["verificador_aritmetico"])
        self.assertEqual(ab.decidir("destilado", reg)[0], "remover")
        self.assertFalse(ab.deve_rebaixar_para_facil(reg, ex["meta"]))

    def test_d2_ambiguidade_de_redacao_do_validador_nao_e_erro_matematico(self):
        # idx 599 (MT9013): o "ou" do enunciado; gabarito 4 certo na leitura usual
        q = questao_boa()
        ex = exemplo("9º", "H05", q, "MT9013FH05TD")
        reg = _registro(ex, 599, _juiz(False, ["pergunta_ambigua", "gabarito_sem_resposta_unica"]),
                        _juiz(False, ["enunciado_ambiguo", "distrator_verdadeiro", "resposta_nao_unica"]))
        self.assertFalse(ab.erro_matematico_confirmado(reg)[0])
        self.assertEqual(ab.decidir("real", reg), ("manter", "baixa_protegido_revisao_humana"))
        # o revisor marcar enunciado_ambiguo junto com um erro real não salva o item (idx 561)
        reg561 = _registro(ex, 561, _juiz(False, ["resposta_nao_unica", "gabarito_sem_resposta_unica"]),
                           _juiz(False, ["resposta_nao_unica", "enunciado_ambiguo"]))
        self.assertTrue(ab.erro_matematico_confirmado(reg561)[0])

    def test_d2_resolucao_que_supoe_o_dado_ausente_e_erro(self):
        # idx 1368: validador "dados_insuficientes" + filtro resolucao_supoe_dado
        ex = exemplo("9º", "H15", questao_hexagono(), "DIST-H15-Fácil-00589", destilado=True, professor="p")
        reg = _registro(ex, 1368, _juiz(False, ["dados_insuficientes", "gabarito_sem_resposta_unica"]),
                        _juiz(True), filtros=["resolucao_supoe_dado"])
        conf, evid = ab.erro_matematico_confirmado(reg)
        self.assertTrue(conf)
        self.assertEqual(evid["validador"], ["resolucao_usa_dado_ausente"])
        # sem o filtro (figura perdida na transcrição), continua "incompleto": fica
        reg2 = _registro(ex, 1368, _juiz(False, ["dados_insuficientes", "gabarito_sem_resposta_unica"]),
                         _juiz(False, ["dados_insuficientes"]))
        self.assertFalse(ab.erro_matematico_confirmado(reg2)[0])

    def test_destilado_reprovado_por_um_juiz_so_fica_e_vai_para_revisao_humana(self):
        # idx 1187 ("dez mil quatrocentos e oito", correto): só o revisor reprovou
        ex = exemplo("5º", "H01", questao_boa(), "DIST-H01-Difícil-00408", dificuldade="Difícil",
                     destilado=True, professor="p")
        reg = _registro(ex, 1187, _juiz(True), _juiz(False, ["distrator_verdadeiro", "dificuldade_incoerente"]))
        self.assertEqual(reg["confianca"], "media")
        self.assertEqual(ab.decidir("destilado", reg), ("manter", "media_sem_confirmacao_revisao_humana"))

    # --- montagem: injetadas com prompts antigos --------------------------------
    def _amb_injetado(self):
        amb = Ambiente(falta_facil=3)
        silencioso(iq.injetar, amb.args_injecao(), agentes=agentes_com(Roteiro(), 100))
        aceitas = amb.linhas("injecao_saeb.jsonl")
        self.assertEqual(len(aceitas), 3)
        return amb, aceitas

    def _reescrever_saida(self, amb, exs):
        (amb.d / "injecao_saeb.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in exs),
                                                  encoding="utf-8")

    def test_montar_exclui_injetada_julgada_com_prompts_antigos(self):
        amb, aceitas = self._amb_injetado()
        try:
            aceitas[0]["meta"]["versao_prompts"] = "7570dee3dd46"
            self._reescrever_saida(amb, aceitas)
            rel = silencioso(ab.montar, amb.args_auditoria("--montar"))
            v3 = {e["meta"]["codigo_item"] for e in amb.linhas("v3.jsonl")}
            antiga = aceitas[0]["meta"]["codigo_item"]
            self.assertNotIn(antiga, v3)
            self.assertEqual(rel["n_injetados"], 2)
            excl = amb.linhas("injetadas_excluidas_v3.jsonl")
            self.assertEqual([(r["codigo_item"], r["motivo"]) for r in excl],
                             [(antiga, "julgada_com_prompts_antigos_sem_rejulgamento")])
            # rejulgada com os prompts atuais: entra
            aceitas[0]["meta"]["rejulgamento"] = {"versao_prompts": aq.VERSAO_PROMPTS}
            self._reescrever_saida(amb, aceitas)
            silencioso(ab.montar, amb.args_auditoria("--montar"))
            self.assertIn(antiga, {e["meta"]["codigo_item"] for e in amb.linhas("v3.jsonl")})
        finally:
            amb.fechar()

    def test_rejulgar_aprova_marca_e_tira_a_reprovada(self):
        amb, aceitas = self._amb_injetado()
        try:
            for ex in aceitas[:2]:
                ex["meta"]["versao_prompts"] = "7570dee3dd46"
            # a 2ª tem gabarito errado: o validador cego reprova no rejulgamento
            q = json.loads(aceitas[1]["messages"][2]["content"])["questoes"][0]
            q["resposta_correta"] = "A" if q["resposta_correta"] != "A" else "B"
            aceitas[1]["messages"][2]["content"] = json.dumps({"questoes": [q]}, ensure_ascii=False)
            self._reescrever_saida(amb, aceitas)
            cods = [e["meta"]["codigo_item"] for e in aceitas]
            n_rej = len(amb.linhas("rejeitadas.jsonl"))
            cli = Roteiro()
            args = amb.args_injecao("--rejulgar")
            res = silencioso(iq.rejulgar, args, agentes=agentes_com(cli, 20))
            self.assertEqual(res["aprovadas"], [cods[0]])
            self.assertEqual(list(res["saem"]), [cods[1]])
            self.assertEqual(res["ja_atuais"], 1)
            self.assertEqual(cli.chamadas["gerador"], 0)
            depois = {e["meta"]["codigo_item"]: e for e in amb.linhas("injecao_saeb.jsonl")}
            self.assertEqual(set(depois), {cods[0], cods[2]})
            self.assertEqual(depois[cods[0]]["meta"]["rejulgamento"]["versao_prompts"], aq.VERSAO_PROMPTS)
            rej = amb.linhas("rejeitadas.jsonl")[n_rej:]
            self.assertEqual([(r["codigo_item"], r["etapa"]) for r in rej], [(cods[1], "rejulgamento")])
            # o código que saiu fica reservado
            _, _, _, ultimo = iq.estado_previo(amb.d / "injecao_saeb.jsonl", amb.d / "rejeitadas.jsonl")
            self.assertGreaterEqual(ultimo, int(cods[1][-5:]))
            # 2ª rodada: nada a rejulgar, 0 chamadas
            cli2 = Roteiro()
            silencioso(iq.rejulgar, args, agentes=agentes_com(cli2, 20))
            self.assertEqual(cli2.total(), 0)
            # e a montagem aceita a rejulgada
            silencioso(ab.montar, amb.args_auditoria("--montar"))
            self.assertEqual({e["meta"]["codigo_item"] for e in amb.linhas("v3.jsonl")} & set(cods),
                             {cods[0], cods[2]})
        finally:
            amb.fechar()

    def test_mover_por_auditoria_independente_e_rebaixar(self):
        amb, aceitas = self._amb_injetado()
        try:
            cods = [e["meta"]["codigo_item"] for e in aceitas]
            args = amb.args_injecao()
            r = iq.mover_para_rejeitadas(args, {cods[0]: "viola a D5", "INJ-NAO-EXISTE": "x"})
            self.assertEqual((r["movidas"], r["nao_encontradas"]), ([cods[0]], ["INJ-NAO-EXISTE"]))
            rej = amb.linhas("rejeitadas.jsonl")[-1]
            self.assertEqual((rej["codigo_item"], rej["motivo"], rej["detalhe"]),
                             (cods[0], "auditoria_independente", "viola a D5"))
            self.assertEqual([e["meta"]["codigo_item"] for e in amb.linhas("injecao_saeb.jsonl")], cods[1:])
            # nova injeção não reutiliza o código que saiu
            silencioso(iq.injetar, amb.args_injecao("--meta-por-habilidade", "3", "--meta-relativa"),
                       agentes=agentes_com(Roteiro(), 100))
            novos = [e["meta"]["codigo_item"] for e in amb.linhas("injecao_saeb.jsonl")]
            self.assertNotIn(cods[0], novos)
            self.assertEqual(len(set(novos)), len(novos))
            # rebaixar (D3 por decisão humana): só Moderado/Difícil -> Fácil
            ex = amb.linhas("injecao_saeb.jsonl")[0]
            self.assertEqual(iq.rebaixar_aceitas(args, {ex["meta"]["codigo_item"]: "n"})["falhas"],
                             {ex["meta"]["codigo_item"]: "so_rebaixa_moderado_ou_dificil"})
        finally:
            amb.fechar()

    def test_rebaixar_aceita_dificil_nos_tres_lugares(self):
        amb, aceitas = self._amb_injetado()
        try:
            ex = json.loads(json.dumps(aceitas[0]))
            ex["meta"]["dificuldade"] = "Difícil"
            ex["messages"][1]["content"] = ex["messages"][1]["content"].replace("Dificuldade: Fácil.",
                                                                                "Dificuldade: Difícil.")
            q = json.loads(ex["messages"][2]["content"])["questoes"][0]
            ex["messages"][2]["content"] = json.dumps({"questoes": [dict(q, difficulty="HARD")]}, ensure_ascii=False)
            self._reescrever_saida(amb, [ex] + aceitas[1:])
            cod = ex["meta"]["codigo_item"]
            r = iq.rebaixar_aceitas(amb.args_injecao(), {cod: "13-14-15 quanto aos lados: um passo"})
            self.assertEqual(r["rebaixadas"], [cod])
            novo = amb.linhas("injecao_saeb.jsonl")[0]
            self.assertEqual(novo["meta"]["dificuldade"], "Fácil")
            self.assertIn("Dificuldade: Fácil.", novo["messages"][1]["content"])
            self.assertEqual(json.loads(novo["messages"][2]["content"])["questoes"][0]["difficulty"], "EASY")
            self.assertEqual(novo["meta"]["rotulo_corrigido"]["fonte"], "auditoria_independente")
        finally:
            amb.fechar()

    def test_dois_processos_na_mesma_saida_param_antes_de_gastar(self):
        amb = Ambiente()
        try:
            args = amb.args_injecao()
            cli = Roteiro()
            with iq.trava_saida(amb.d / "injecao_saeb.jsonl"):
                with self.assertRaises(iq.SaidaOcupada):
                    iq.injetar(args, agentes=agentes_com(cli, 100))
            self.assertEqual(cli.total(), 0)
        finally:
            amb.fechar()

    # --- auditoria: decisão nova sobre respostas já pagas -----------------------
    def test_auditoria_refaz_versao_de_juizes_antiga_pelo_cache_sem_custo(self):
        amb = Ambiente()
        try:
            cache = amb.d / "juizes_cache.jsonl"
            silencioso(ab.auditar, amb.args_auditoria("--cache-juizes", str(cache)),
                       agentes=agentes_com(Roteiro(), 100))
            regs = amb.linhas("auditoria.jsonl")
            self.assertTrue(all(r["versao_juizes"] == aq.VERSAO_JUIZES for r in regs))
            self.assertTrue(all("status" in r["validador"] for r in regs if r.get("validador")))
            antigos = [{k: v for k, v in r.items() if k != "versao_juizes"} for r in regs]
            (amb.d / "auditoria.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                                           for r in antigos), encoding="utf-8")
            cli = Roteiro()
            silencioso(ab.auditar, amb.args_auditoria("--cache-juizes", str(cache)), agentes=agentes_com(cli, 0))
            self.assertEqual(cli.total(), 0)
            depois = ab.ultimos_registros(amb.d / "auditoria.jsonl")
            self.assertTrue(all(r["versao_juizes"] == aq.VERSAO_JUIZES for r in depois.values()))
        finally:
            amb.fechar()
