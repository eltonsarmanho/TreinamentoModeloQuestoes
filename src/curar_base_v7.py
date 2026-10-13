"""Curadoria item a item da base (ciclo v7): um professor de Matemática do SAEB revisa CADA exemplo.

Padrão: DRY-RUN (sem rede, custo 0). Modo real exige --real --max-chamadas N --teto-brl R.
Nunca escreve em data/train_curado_v5.jsonl, v6, train.jsonl, train_curado.jsonl nem val*.

FLUXO (uma chamada paga por exemplo; a 2ª só quando o curador propõe uma correção)
  1. curador (sabia, papel "curador"): recebe SÓ ano, habilidade, enunciado e as 5 alternativas.
     Não recebe gabarito, resolução nem a dificuldade pedida (resolve às cegas, como o validador
     de agentes_questoes; a comparação com o gabarito é feita aqui no código). Devolve: resolução
     própria, V/F/I/G de cada alternativa, problemas (códigos do SAEB), nível (Fácil/Moderado/
     Difícil relativo ao ano), veredito (aprovado | corrigivel | inviavel) e, se corrigível, o
     item corrigido completo + resolução para o aluno.
  2. código: compara a resposta do curador com o gabarito e decide:
       confirmada       curador e gabarito concordam, resposta única      -> item fica (resolução só é
                        trocada se a original for defeituosa e a nova passar em validar_nova)
       corrigida        item corrigido passou nos filtros determinísticos E na validação cega (2ª chamada,
                        o validador calibrado de agentes_questoes) -> item corrigido entra
       gabarito_discorda curador resolve e chega em outra letra (ou em nenhuma)  -> revisão humana
       inviavel         não há correção honesta (figura ausente, premissa impossível...) -> quarentena
       abstencao        API falhou / truncou / JSON inválido  -> nada é decidido; volta na próxima rodada
  3. `montar` (sem API) aplica a política por origem e grava data/train_curado_v7.jsonl + proveniência.

POLÍTICA POR ORIGEM (invariante do projeto: item real do banco SAEB não muda em silêncio)
  - injetado/sintético/destilado: confirmada, corrigida e inviavel (remoção documentada) são aplicadas.
  - real_banco / real_L2: só "confirmada" (+ resolução) é aplicada sozinha; correção e remoção viram
    PROPOSTA em outputs/melhoria_v7/propostas_reais.jsonl e só entram com --aplicar-reais (decisão humana).
  - gabarito_discorda nunca troca o gabarito de ninguém: o item sai da base (não real) ou vira proposta (real).
  - Dificuldade: o nível estimado é GRAVADO; só vira rótulo com --aplicar-dificuldade (decisão D5 do v6:
    o revisor automático tende a chamar de "Fácil" o que o SAEB chama de Moderado/Difícil).

    venv/bin/python src/curar_base_v7.py curar --estimar --base data/train_curado_v6.jsonl
    venv/bin/python src/curar_base_v7.py curar --base data/train_curado_v6.jsonl            # dry-run
    venv/bin/python src/curar_base_v7.py curar --real --max-chamadas 40 --teto-brl 5 --limite 30   # piloto
    venv/bin/python src/curar_base_v7.py montar --base data/train_curado_v6.jsonl            # sem API
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agentes_questoes as aq  # noqa: E402
import sabia_v6 as sv  # noqa: E402
import schema_utils as su  # noqa: E402
from perfil_base import carregar, hash_questao, origem_de  # noqa: E402

LETRAS = "ABCDE"
SAIDA_DIR = ROOT / "outputs" / "melhoria_v7"
CURADORIA = SAIDA_DIR / "curadoria_v7.jsonl"
CACHE = ROOT / "outputs" / "agentes" / "v7_curadoria_cache.jsonl"
LOG_USO = SAIDA_DIR / "uso_api_curadoria.jsonl"
V6 = ROOT / "data" / "train_curado_v6.jsonl"
V7 = ROOT / "data" / "train_curado_v7.jsonl"
PROPOSTAS_REAIS = SAIDA_DIR / "propostas_reais.jsonl"
PROTEGIDOS = ("train_curado.jsonl", "train.jsonl", "train_curado_v3.jsonl", "train_curado_v5.jsonl", "train_curado_v6.jsonl")
NIVEIS = ("Fácil", "Moderado", "Difícil")
DIFF_EN = {"Fácil": "EASY", "Moderado": "MEDIUM", "Difícil": "HARD"}
# problemas que tornam o item inviável SEM correção honesta no app só-texto
CODIGOS = ("premissa_impossivel", "dados_insuficientes", "pergunta_ambigua", "resposta_nao_unica", "distrator_verdadeiro",
           "nenhuma_correta", "alternativas_equivalentes", "nda_indevida", "dependencia_visual", "conta_impossivel",
           "contexto_implausivel", "enunciado_ilegivel")

CURADOR_SISTEMA = """Você é professor de Matemática do Ensino Fundamental, com anos de sala de aula, e elaborador/revisor de itens do SAEB (INEP, Matriz de Referência de Matemática). Você revisa UM item de múltipla escolha (5 alternativas) de um banco que treina um gerador de questões para alunos. Você NÃO conhece o gabarito do autor, a resolução dele nem a dificuldade pedida: resolva o item do zero, como faria numa prova, e só então julgue. Use as definições inclusivas da BNCC. O app é SOMENTE TEXTO: se o item depende de figura, malha, gráfico, tabela, calendário ou imagem cujos dados não estão escritos no enunciado, ele só é corrigível se for possível escrever esses dados no texto sem mudar a habilidade; senão é inviável.

Siga esta ordem:
1. DADOS: liste o que o enunciado realmente informa. Não complete com o que o autor "provavelmente" quis dizer. "Pentágono" não diz que a figura é regular.
2. PREMISSA: a situação pode existir? Impossíveis: triângulo com dois ângulos retos ou dois obtusos; lados que violam a desigualdade triangular (4, 4 e 8 não formam triângulo); quadrilátero com quatro ângulos agudos ou quatro obtusos; parte maior que o todo; número fracionário de pessoas; uma moeda com face 4; retângulo com diâmetro.
3. RESOLUÇÃO: resolva passo a passo e confira por outro caminho (operação inversa, estimativa, substituição). Em contagem de dias ou termos, decida se os extremos entram. Se o enunciado diz que algo é dado "para CADA", multiplique.
4. ALTERNATIVAS: julgue A a E separadamente: "V" verdadeira garantida pelos dados; "F" falsa; "I" indeterminada (falta dado); "G" verdadeira mas classe mais geral do mesmo critério que outra "V" (só em pergunta pelo nome da figura). Arredondamento do ensino básico: valor arredondado OU truncado em duas casas de uma dízima é "V". Dois valores iguais em formas diferentes (1/2 e 0,5) são ambos "V". "Nenhuma das alternativas anteriores" é "V" só se as outras quatro forem "F".
5. PROBLEMAS (use só estes códigos): premissa_impossivel, dados_insuficientes, pergunta_ambigua, resposta_nao_unica, distrator_verdadeiro, nenhuma_correta, alternativas_equivalentes, nda_indevida, dependencia_visual, conta_impossivel, contexto_implausivel, enunciado_ilegivel.
6. NÍVEL, em relação ao ANO do aluno: "Fácil" = um passo ou aplicação direta de um conceito, dados prontos; "Moderado" = dois passos, ou interpretação/conversão de unidade/escolha da operação; "Difícil" = três ou mais passos, conceitos combinados ou raciocínio não rotineiro. Não infle: um passo só nunca é Difícil.
7. VEREDITO: "aprovado" = resposta única, dados suficientes, contexto plausível, enunciado claro; "corrigivel" = há defeito que um professor consertaria mexendo o mínimo; "inviavel" = não há conserto honesto.

Padrão SAEB para o item e para o conserto: enunciado claro e na linguagem do ano; uma ÚNICA resposta correta; distratores plausíveis que refletem erros típicos de aluno (operação trocada, etapa esquecida, erro por um, ordem de grandeza, conceito vizinho), nenhum verdadeiro, nenhum equivalente a outro, nenhum absurdo; contexto do cotidiano e possível. Ao corrigir: mantenha ano, habilidade, ideia e números sempre que der; mude só o necessário (trocar distrator verdadeiro ou repetido; escrever o dado que faltava; trocar contexto absurdo; reescrever enunciado ilegível; ajustar números para a premissa existir). Mantenha a alternativa "Nenhuma das alternativas anteriores" na posição E se já existia, e não a torne a correta se não era. Devolva TODAS as alternativas do item corrigido, a letra correta e a resolução para o aluno: até 6 frases, contas explícitas no formato "a + b = c", sem citar letras, terminando com a resposta escrita com o mesmo valor e unidade da alternativa correta. Não ensine nada que não esteja no ano.

Responda APENAS com um JSON numa linha, nesta ordem de campos:
{"dados": [str], "premissa_possivel": bool, "resolucao_propria": str, "alternativas": {"A": "V|F|I|G", "B": "V|F|I|G", "C": "V|F|I|G", "D": "V|F|I|G", "E": "V|F|I|G"}, "resposta_calculada": "A|B|C|D|E" ou null, "problemas": [{"codigo": str, "detalhe": str}], "nivel": "Fácil|Moderado|Difícil", "veredito": "aprovado|corrigivel|inviavel", "resolucao_aluno": str ou null, "item_corrigido": null ou {"enunciado": str, "alternativas": {"A": str, "B": str, "C": str, "D": str, "E": str}, "resposta_correta": "A|B|C|D|E", "resolucao_passo_a_passo": str, "motivo": str}, "confianca": número de 0 a 1}
"resolucao_aluno" só quando veredito = "aprovado" (resolução para o aluno do item como está). "item_corrigido" só quando veredito = "corrigivel". Seja breve: "resolucao_propria" até 6 frases."""

CURADOR_USUARIO = """Ano: {ano}. Habilidade: {habilidade}.

Enunciado: {enunciado}

A) {A}
B) {B}
C) {C}
D) {D}
E) {E}"""


def papeis_padrao(modelo_curador="sabia-4-thinking", modelo_validador="sabia-4-thinking"):
    return {
        "curador": {"modelo": modelo_curador, "temperatura": 0.0,
                    "max_tokens": 7000 if "thinking" in modelo_curador else 2500,
                    "versao_prompt": sv.versao_prompt(CURADOR_SISTEMA, CURADOR_USUARIO)},
        "validador_correcao": {"modelo": modelo_validador, "temperatura": 0.0,
                               "max_tokens": aq.max_tokens_para("validador", modelo_validador),
                               "versao_prompt": sv.versao_prompt(aq.VALIDADOR_SISTEMA, aq.VALIDADOR_USUARIO)},
    }


# ------------------------------------------------------------------ mensagens e parse
def mensagens_curador(q, ano, habilidade):
    u = CURADOR_USUARIO.format(ano=ano, habilidade=habilidade, enunciado=q["enunciado"],
                               **{L: q["alternativas"].get(L, "") for L in LETRAS})
    return [{"role": "system", "content": CURADOR_SISTEMA}, {"role": "user", "content": u}]


def mensagens_validador(q):
    u = aq.VALIDADOR_USUARIO.format(enunciado=q["enunciado"], **{L: q["alternativas"].get(L, "") for L in LETRAS})
    return [{"role": "system", "content": aq.VALIDADOR_SISTEMA}, {"role": "user", "content": u}]


def _alts_ok(a):
    return (isinstance(a, dict) and set(a) == set(LETRAS) and all(isinstance(a[L], str) and a[L].strip() for L in LETRAS)
            and len({re.sub(r"\s+", " ", a[L].strip().lower()) for L in LETRAS}) == 5)


def parse_curador(texto):
    """dict normalizado ou None (malformado = abstenção, nunca aprovação)."""
    o = aq.extrair_json(texto, ("alternativas", "veredito"), estrito=True)
    if not isinstance(o, dict):
        return None
    st = o.get("alternativas")
    if not isinstance(st, dict) or set(st) != set(LETRAS) or any(st[L] not in ("V", "F", "I", "G") for L in LETRAS):
        return None
    if o.get("veredito") not in ("aprovado", "corrigivel", "inviavel") or o.get("nivel") not in NIVEIS:
        return None
    if not isinstance(o.get("premissa_possivel"), bool):
        return None
    calc = o.get("resposta_calculada")
    if calc is not None and calc not in tuple(LETRAS):
        return None
    probs = [{"codigo": str(p.get("codigo", ""))[:40], "detalhe": str(p.get("detalhe", ""))[:300]}
             for p in (o.get("problemas") or []) if isinstance(p, dict)]
    corr = o.get("item_corrigido")
    if o["veredito"] == "corrigivel":
        if not (isinstance(corr, dict) and isinstance(corr.get("enunciado"), str) and corr["enunciado"].strip()
                and _alts_ok(corr.get("alternativas")) and corr.get("resposta_correta") in tuple(LETRAS)
                and isinstance(corr.get("resolucao_passo_a_passo"), str) and corr["resolucao_passo_a_passo"].strip()):
            return None
        corr = {"enunciado": corr["enunciado"].strip(), "alternativas": {L: corr["alternativas"][L].strip() for L in LETRAS},
                "resposta_correta": corr["resposta_correta"], "resolucao_passo_a_passo": corr["resolucao_passo_a_passo"].strip(),
                "motivo": str(corr.get("motivo", ""))[:300]}
    else:
        corr = None
    ra = o.get("resolucao_aluno")
    return {"status": dict(st), "premissa_possivel": o["premissa_possivel"], "resposta_calculada": calc, "problemas": probs,
            "nivel": o["nivel"], "veredito": o["veredito"], "item_corrigido": corr,
            "resolucao_aluno": ra.strip() if isinstance(ra, str) and ra.strip() else None,
            "resolucao_propria": str(o.get("resolucao_propria", ""))[:900], "confianca": aq._confianca(o.get("confianca"))}


# ------------------------------------------------------------------ verificações determinísticas
def unica(status, letra):
    """Exatamente a alternativa `letra` é V e as demais são F ou G."""
    return status.get(letra) == "V" and all(status[L] in ("F", "G") for L in LETRAS if L != letra)


def filtros_item(q):
    """Motivos pelos quais um item (original ou corrigido) não pode entrar. [] = passou."""
    motivos = []
    flags = su.check_structure({"questoes": [dict(q, difficulty=q.get("difficulty", "EASY"))]}, quantidade_esperada=1)
    if not all(flags.values()):
        motivos.append("schema")
    ok, _s, mot = su.check_consistency_detalhado(q)
    if ok is False:
        motivos.append(f"verificador_reprova({mot})")
    if su.resposta_fora_das_alternativas(q):
        motivos.append("resultado_fora_das_alternativas")
    if su.depende_de_visual_ausente(q.get("enunciado", "")):
        motivos.append("dependencia_visual")
    try:
        from verificador_geometria import GEO_REJEITA, verificar_geometria
        if verificar_geometria(q)[0] in GEO_REJEITA:
            motivos.append("geometria_reprova")
    except Exception:  # noqa: BLE001 — o verificador abstém; na dúvida não acusa
        pass
    return motivos


def resolucao_defeituosa(q):
    """A resolução original merece troca? (verificador reprova, fora das alternativas, ou triagem não 'adequada')."""
    from triar_resolucoes import triar
    classe, _s = triar(q)
    return classe != "adequada", classe


def classe_mudanca(q, corr):
    g0, g1 = q["alternativas"][q["resposta_correta"]], corr["alternativas"][corr["resposta_correta"]]
    if corr["enunciado"] == q["enunciado"] and g0 == g1:
        return "so_distratores"
    return "enunciado" if g0 == g1 else "gabarito_alterado"


# ------------------------------------------------------------------ fluxo por item
def processar_item(item, q, ch, verificar_correcoes=True):
    """Registro de curadoria do item. Nunca altera q."""
    h = hash_questao(q)
    reg = {"codigo_item": item["codigo_item"], "linha": item["linha"], "questao": item["questao"], "hash_questao": h,
           "ano": item["ano"], "habilidade": item["habilidade"], "origem": item["origem"],
           "gabarito": q["resposta_correta"], "dificuldade_declarada": item.get("dificuldade"),
           "modelos": {p: c["modelo"] for p, c in ch.papeis.items()},
           "versoes_prompt": {p: c["versao_prompt"] for p, c in ch.papeis.items()}}
    txt = ch.chamar("curador", mensagens_curador(q, item["ano"], item["habilidade"]), hash_exemplo=h,
                    codigo_item=item["codigo_item"], interpretavel=lambda t: parse_curador(t) is not None)
    if txt is None:
        reg.update(decisao="abstencao", motivo=ch.abstencoes.get(h, "sem_resposta"))
        return reg
    p = parse_curador(txt)
    reg.update(nivel_estimado=p["nivel"], veredito_curador=p["veredito"], resposta_curador=p["resposta_calculada"],
               status_alternativas=p["status"], problemas=p["problemas"], confianca=p["confianca"],
               premissa_possivel=p["premissa_possivel"])
    gab = q["resposta_correta"]

    if p["veredito"] == "inviavel":
        reg.update(decisao="inviavel", motivo=",".join(x["codigo"] for x in p["problemas"]) or "inviavel")
    elif p["veredito"] == "aprovado":
        if p["resposta_calculada"] == gab and unica(p["status"], gab) and p["premissa_possivel"]:
            reg.update(decisao="confirmada", motivo="")
            defeituosa, classe = resolucao_defeituosa(q)
            reg["classe_resolucao_original"] = classe
            nova = p["resolucao_aluno"]
            if defeituosa and nova:
                from melhorar_resolucoes import validar_nova
                ok, motivos = validar_nova(q, nova)
                reg["resolucao_nova"] = nova if ok else None
                reg["resolucao_nova_rejeitada"] = None if ok else motivos
        else:
            if not p["premissa_possivel"]:
                motivo = "premissa_impossivel_apesar_de_aprovado"
            elif not unica(p["status"], gab):
                motivo = f"resposta_nao_unica:status={p['status']}"
            else:
                motivo = f"curador={p['resposta_calculada']} gabarito={gab}"
            reg.update(decisao="gabarito_discorda", motivo=motivo)
    else:  # corrigivel
        corr = p["item_corrigido"]
        novo = {"enunciado": corr["enunciado"], "alternativas": corr["alternativas"],
                "resposta_correta": corr["resposta_correta"], "resolucao_passo_a_passo": corr["resolucao_passo_a_passo"],
                "difficulty": DIFF_EN[p["nivel"]]}
        reg["item_corrigido"] = {k: corr[k] for k in ("enunciado", "alternativas", "resposta_correta", "resolucao_passo_a_passo", "motivo")}
        reg["mudanca"] = classe_mudanca(q, corr)
        motivos = filtros_item(novo)
        from triar_resolucoes import REF_LETRA
        if REF_LETRA.search(novo["resolucao_passo_a_passo"]):
            motivos.append("resolucao_cita_letra")
        if hash_questao(novo) == h:
            motivos.append("correcao_identica")
        if motivos:
            reg.update(decisao="correcao_rejeitada", motivo=",".join(motivos))
        elif not verificar_correcoes:
            reg.update(decisao="corrigida_nao_verificada", motivo="")
        else:
            hv = hash_questao(novo)
            t2 = ch.chamar("validador_correcao", mensagens_validador(novo), hash_exemplo=hv, codigo_item=item["codigo_item"],
                           interpretavel=lambda t: aq.interpretar_fase1(t) is not None)
            if t2 is None:
                reg.update(decisao="abstencao", motivo=ch.abstencoes.get(hv, "sem_resposta_validador"))
                return reg
            f1 = aq.interpretar_fase1(t2)
            reg["validacao_correcao"] = {"resposta": f1["resposta_llm"], "status": f1["status"],
                                         "problemas": [x["codigo"] for x in f1["problemas"]]}
            ok = (f1["resposta_llm"] == novo["resposta_correta"] and unica(f1["status"], novo["resposta_correta"])
                  and f1["premissa_possivel"] and not f1["dependencia_visual"] and not f1["problemas"] and not f1["equivalentes"])
            reg.update(decisao="corrigida" if ok else "correcao_rejeitada",
                       motivo="" if ok else "validador_cego_nao_confirma:" + ",".join(x["codigo"] for x in f1["problemas"]))
    ch.registrar_resultado("curador", h, item["codigo_item"], reg["decisao"], reg.get("motivo", ""))
    return reg


# ------------------------------------------------------------------ dry-run
class ClienteSimuladoCurador:
    """Sem rede. Conhece o gabarito só para simular (o curador real é cego). Aprova todo item."""

    def __init__(self, gabarito_por_enunciado):
        self.g = gabarito_por_enunciado

    def chat_completion(self, messages, max_tokens, temperature):
        u = messages[-1]["content"]
        enun = u.split("Enunciado: ", 1)[1].split("\n\nA) ", 1)[0] if "Enunciado: " in u else u.split("Enunciado: ")[-1]
        letra = self.g.get(enun.strip(), "A")
        st = {L: ("V" if L == letra else "F") for L in LETRAS}
        if messages[0]["content"] == CURADOR_SISTEMA:
            t = json.dumps({"dados": [], "premissa_possivel": True, "resolucao_propria": "(simulado)", "alternativas": st,
                            "resposta_calculada": letra, "problemas": [], "nivel": "Moderado", "veredito": "aprovado",
                            "resolucao_aluno": None, "item_corrigido": None, "confianca": 0.9}, ensure_ascii=False)
        else:
            t = json.dumps({"dados": [], "premissa_possivel": True, "pergunta": "", "resolucao": "(simulado)", "alternativas": st,
                            "motivos": {L: "" for L in LETRAS}, "equivalentes": [], "dependencia_visual": False,
                            "resposta_calculada": letra, "problemas": [], "confianca": 0.9})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=t), finish_reason="stop")], usage=None)


# ------------------------------------------------------------------ estimativa
# Premissas ainda NÃO medidas: rode o piloto (--limite 30) e substitua pelos logs reais.
TOKENS_MEDIDOS = {
    "curador": {"sabia-4": (1500, 900, 1400), "sabia-4-thinking": (1500, 3000, 5000), "sabiazinho-4": (1500, 900, 1400)},
    "validador_correcao": {"sabia-4": (4300, 700, 900), "sabia-4-thinking": (4300, 2500, 4000), "sabiazinho-4": (4300, 700, 900)},
}
FRACAO_CORRIGIVEL = 0.25  # premissa: a amostra humana v6 achou ~41% de defeito duro; só parte vira correção


def ja_decididos(saida=CURADORIA):
    out = {}
    if Path(saida).exists():
        for l in open(saida, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("decisao") != "abstencao":
                out[r["hash_questao"]] = r["decisao"]
    return out


def itens_da_base(base):
    itens = []
    for i, j, m, q, _p in carregar(base):
        itens.append(({"linha": i, "questao": j, "codigo_item": m["codigo_item"], "ano": m["ano"], "habilidade": m["habilidade"],
                       "origem": origem_de(m), "dificuldade": m.get("dificuldade")}, q))
    return itens


def cmd_curar(args):
    pares = itens_da_base(args.base)
    feitos = ja_decididos(args.saida)
    alvo = [(x, q) for x, q in pares if hash_questao(q) not in feitos]
    if args.origem:
        alvo = [(x, q) for x, q in alvo if x["origem"].startswith(args.origem)]
    if args.limite:
        alvo = alvo[:args.limite]
    papeis = papeis_padrao(args.modelo_curador, args.modelo_validador)
    precos = sv.carregar_precos()
    cfg = {"curador": dict(papeis["curador"], chamadas_por_item=1),
           "validador_correcao": dict(papeis["validador_correcao"], chamadas_por_item=FRACAO_CORRIGIVEL if args.verificar else 0)}
    est = sv.estimar_lote(len(alvo), cfg, precos, {p: TOKENS_MEDIDOS[p][papeis[p]["modelo"]] for p in papeis})
    print(f"fila: {len(alvo)} exemplos (já decididos fora: {len(feitos)})")
    print("estimativa (premissas de tokens NÃO medidas; rode o piloto):", json.dumps(est, ensure_ascii=False))
    if args.estimar:
        return
    saida = Path(args.saida)
    if args.real:
        if not args.max_chamadas or not args.teto_brl:
            sys.exit("ABORTADO: --real exige --max-chamadas N e --teto-brl R (tetos duros).")
        clientes = {p: aq.criar_cliente_real("maritaca", c["modelo"]) for p, c in papeis.items()}
        ch = sv.ChamadorSabia(papeis, clientes, args.max_chamadas, args.teto_brl, precos, log_uso=LOG_USO, cache=CACHE)
    else:
        sim = ClienteSimuladoCurador({q["enunciado"].strip(): q["resposta_correta"] for _x, q in pares})
        ch = sv.ChamadorSabia(papeis, {p: sim for p in papeis}, max(1, 2 * len(alvo)), None, precos,
                              log_uso=SAIDA_DIR / "dryrun" / "uso_api_curadoria.jsonl", cache=None, simulado=True)
        saida = SAIDA_DIR / "dryrun" / "curadoria_v7.jsonl"
        print("DRY-RUN: sem rede, custo 0; saída em", saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    cont = Counter()
    for x, q in alvo:
        try:
            reg = processar_item(x, q, ch, verificar_correcoes=args.verificar)
        except sv.OrcamentoEsgotado as e:
            print(f"PARADO: {e}. Nada foi decidido por falta de orçamento; rode de novo (o cache evita pagar o que já foi feito).")
            break
        with aq.abrir_para_gravar(saida, "a") as f:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        cont[reg["decisao"]] += 1
    print("decisões:", dict(cont), "| chamador:", ch.resumo())


# ------------------------------------------------------------------ montagem (sem API)
def _qs(ex):
    return json.loads(ex["messages"][2]["content"])["questoes"]


def _set_qs(ex, qs):
    ex["messages"][2]["content"] = json.dumps({"questoes": qs}, ensure_ascii=False)


def aplicar_nivel(ex, nivel):
    """Troca o rótulo de dificuldade do exemplo inteiro (meta, prompt do usuário e campo difficulty)."""
    antigo = ex["meta"].get("dificuldade")
    ex["meta"]["dificuldade"] = nivel
    ex["messages"][1]["content"] = re.sub(r"Dificuldade: [^.]+\.", f"Dificuldade: {nivel}.", ex["messages"][1]["content"])
    qs = _qs(ex)
    for q in qs:
        q["difficulty"] = DIFF_EN[nivel]
    _set_qs(ex, qs)
    return antigo


def montar(base, curadoria, saida, aplicar_reais=False, aplicar_dificuldade=False, sem_gravar=False):
    saida = Path(saida)
    if saida.name in PROTEGIDOS or saida.name.startswith("val") or saida.resolve() == Path(base).resolve():
        raise SystemExit(f"ABORTADO: não escrevo em {saida.name}.")
    dec = {}
    for l in open(curadoria, encoding="utf-8"):
        r = json.loads(l)
        if r.get("decisao") != "abstencao":
            dec[r["hash_questao"]] = r
    out, prov, propostas = [], [], []
    for l in open(base, encoding="utf-8"):
        if not l.strip():
            continue
        ex = json.loads(l)
        cod, real = ex["meta"]["codigo_item"], origem_de(ex["meta"]).startswith("real")
        qs = _qs(ex)
        if len(qs) != 1:
            out.append(ex)
            prov.append({"codigo_item": cod, "operacao": "mantida", "motivo": "exemplo_com_varias_questoes"})
            continue
        q = qs[0]
        h = hash_questao(q)
        r = dec.get(h)
        if r is None:
            out.append(ex)
            prov.append({"codigo_item": cod, "hash": h, "operacao": "sem_curadoria", "motivo": "ainda_nao_decidido"})
            continue
        d = r["decisao"]
        novo = json.loads(json.dumps(ex))
        reg = {"codigo_item": cod, "hash": h, "decisao_curador": d, "origem": r["origem"], "nivel_estimado": r.get("nivel_estimado"),
               "modelos": r["modelos"], "versoes_prompt": r["versoes_prompt"]}
        precisa_humano = real and not aplicar_reais
        if d == "confirmada":
            nova = r.get("resolucao_nova")
            if nova:
                nq = dict(q, resolucao_passo_a_passo=nova)
                _set_qs(novo, [nq])
                reg.update(operacao="resolucao_corrigida", resolucao_original=q.get("resolucao_passo_a_passo"), resolucao_nova=nova)
            else:
                reg["operacao"] = "mantida"
        elif d in ("corrigida", "corrigida_nao_verificada"):
            if precisa_humano:
                propostas.append({"codigo_item": cod, "hash": h, "tipo": "correcao", "mudanca": r.get("mudanca"), "item_corrigido": r["item_corrigido"], "problemas": r.get("problemas")})
                out.append(ex)
                prov.append(dict(reg, operacao="proposta_real_pendente", motivo="correcao_de_item_real_exige_decisao_humana"))
                continue
            c = r["item_corrigido"]
            nq = {"enunciado": c["enunciado"], "alternativas": c["alternativas"], "resolucao_passo_a_passo": c["resolucao_passo_a_passo"],
                  "resposta_correta": c["resposta_correta"], "difficulty": q.get("difficulty", "MEDIUM")}
            _set_qs(novo, [nq])
            reg.update(operacao="corrigida", mudanca=r.get("mudanca"), antes={k: q.get(k) for k in ("enunciado", "alternativas", "resposta_correta", "resolucao_passo_a_passo")},
                       motivo=c.get("motivo"), validacao_correcao=r.get("validacao_correcao"))
        else:  # inviavel, gabarito_discorda, correcao_rejeitada: o item não pode ficar como está
            if precisa_humano:
                propostas.append({"codigo_item": cod, "hash": h, "tipo": "remocao", "decisao_curador": d, "motivo": r.get("motivo"), "problemas": r.get("problemas")})
                out.append(ex)
                prov.append(dict(reg, operacao="proposta_real_pendente", motivo=f"{d}:{r.get('motivo')}"))
                continue
            prov.append(dict(reg, operacao="quarentena", motivo=f"{d}:{r.get('motivo')}"))
            continue
        if aplicar_dificuldade and r.get("nivel_estimado") in NIVEIS and r["nivel_estimado"] != ex["meta"].get("dificuldade"):
            antes = aplicar_nivel(novo, r["nivel_estimado"])
            reg["dificuldade_relabel"] = {"antes": antes, "depois": r["nivel_estimado"]}
        out.append(novo)
        prov.append(reg)
    letras = Counter(_qs(e)[0]["resposta_correta"] for e in out if len(_qs(e)) == 1)
    n = sum(letras.values())
    resumo = {"exemplos_base": sum(1 for l in open(base, encoding="utf-8") if l.strip()), "exemplos_v7": len(out),
              "operacoes": dict(Counter(p["operacao"] for p in prov)),
              "decisoes_curador": dict(Counter(r["decisao"] for r in dec.values())),
              "letras_pct": {L: round(100 * letras.get(L, 0) / n, 1) for L in LETRAS} if n else {},
              "propostas_reais": len(propostas)}
    if not sem_gravar:
        saida.parent.mkdir(parents=True, exist_ok=True)
        saida.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in out), encoding="utf-8")
        (SAIDA_DIR / "proveniencia_v7.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prov), encoding="utf-8")
        PROPOSTAS_REAIS.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in propostas), encoding="utf-8")
    return resumo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curar", help="chamadas pagas (dry-run por padrão)")
    c.add_argument("--base", default=str(V6))
    c.add_argument("--real", action="store_true")
    c.add_argument("--estimar", action="store_true")
    c.add_argument("--max-chamadas", type=int, default=None)
    c.add_argument("--teto-brl", type=float, default=None)
    c.add_argument("--limite", type=int, default=None)
    c.add_argument("--origem", default=None, help="prefixo da origem (ex.: real, injetado, v5-sintetico)")
    c.add_argument("--modelo-curador", default="sabia-4-thinking")
    c.add_argument("--modelo-validador", default="sabia-4-thinking")
    c.add_argument("--sem-verificar", dest="verificar", action="store_false", help="NÃO faz a 2ª chamada cega nas correções (não recomendado)")
    c.add_argument("--saida", default=str(CURADORIA))
    m = sub.add_parser("montar", help="aplica a curadoria e grava a v7 (sem API)")
    m.add_argument("--base", default=str(V6))
    m.add_argument("--curadoria", default=str(CURADORIA))
    m.add_argument("--saida", default=str(V7))
    m.add_argument("--aplicar-reais", action="store_true", help="aplica correção/remoção também em item REAL (decisão humana)")
    m.add_argument("--aplicar-dificuldade", action="store_true")
    m.add_argument("--simular", action="store_true")
    args = ap.parse_args()
    if args.cmd == "curar":
        cmd_curar(args)
    else:
        print(json.dumps(montar(args.base, args.curadoria, args.saida, args.aplicar_reais, args.aplicar_dificuldade, args.simular),
                         ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
