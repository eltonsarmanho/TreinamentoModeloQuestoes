"""Reescrita de resoluções pouco explicativas (ciclo v6, etapa 4). Padrão: DRY-RUN, sem rede.

Entrada: a fila de reescrita de src/triar_resolucoes.py (classes correta_pouco_explicativa e
incompleta). Para cada item:
  1. enunciado, alternativas e gabarito ficam CONGELADOS (o professor recebe o gabarito);
  2. professor (papel "professor_resolucao") escreve só a explicação nova; se discordar do
     gabarito, responde discorda=true -> o item vai para REVISÃO HUMANA, nada é trocado;
  3. verificações determinísticas da resolução nova (validar_nova):
       >= 80 caracteres e >= 6 palavras fora das contas; não cita letra; não é CAIXA ALTA;
       verificador aritmético não reprova e o resultado não fica fora das alternativas;
       se a original era verificável (ok), a nova também tem de ser;
       cita o valor/conteúdo do gabarito;
       não introduz número que não esteja no enunciado, nas alternativas nem na resolução
       original e que o verificador marque como dado_inventado;
  4. avaliador (papel "avaliador_resolucao" = o juiz de fase 2 já calibrado de agentes_questoes,
     VALIDADOR_FASE2, mesmo prompt): resolução correta, sem dado ausente, sem justificativa falsa,
     chega na alternativa do gabarito. Qualquer "não" rejeita a resolução nova (o original fica).
  Falha de API / truncada / JSON inválido = ABSTENÇÃO: o item fica com a resolução original e
  volta na próxima rodada (cache). Nunca é aprovação.

Saída append-only: outputs/melhoria_v6/resolucoes_novas.jsonl — um registro por item com o
original, a nova, os hashes, as verificações e a decisão. A base não é tocada; a montagem da
candidata (etapa 9) é que lê este arquivo.

    venv/bin/python src/melhorar_resolucoes.py                       # dry-run (sem rede, custo 0)
    venv/bin/python src/melhorar_resolucoes.py --estimar             # só a estimativa de custo
    venv/bin/python src/melhorar_resolucoes.py --real --max-chamadas N --teto-brl R  [--limite K]
"""
import argparse
import json
import random
import re
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import schema_utils as su  # noqa: E402
import sabia_v6 as sv  # noqa: E402
from perfil_base import SAIDA_DIR, carregar, hash_questao  # noqa: E402
from triar_resolucoes import REF_LETRA, MIN_EXPLICATIVA, MIN_PALAVRAS, cita_o_gabarito, palavras_fora_das_contas, triar  # noqa: E402

LETRAS = "ABCDE"
SAIDA = SAIDA_DIR / "resolucoes_novas.jsonl"
CACHE = ROOT / "outputs" / "agentes" / "v6_resolucoes_cache.jsonl"
LOG_USO = SAIDA_DIR / "uso_api_resolucoes.jsonl"

PROFESSOR_SISTEMA = (
    "Você é professor de matemática do ensino fundamental no Brasil e escreve resoluções para alunos. "
    "Você recebe uma questão de múltipla escolha e a alternativa correta, já conferida. Escreva a resolução "
    "passo a passo que um aluno do ano indicado entenda: diga o que a pergunta pede, quais dados do enunciado "
    "usar, faça as contas explícitas no formato 'a + b = c' (sinais + - x ÷) e termine com a resposta escrita "
    "por extenso, com o mesmo valor e a mesma unidade da alternativa correta. Regras: use SOMENTE dados do "
    "enunciado; não mude nenhum número; não cite letras de alternativas (cite valores); não comente as "
    "alternativas erradas; no máximo 6 frases. Se você achar que a alternativa indicada NÃO é a correta, ou que "
    "há mais de uma correta, não escreva a resolução: responda discorda=true e explique em uma frase. "
    'Responda APENAS com JSON numa linha: {"discorda": false, "motivo": "", "resolucao": "<passo a passo>"}')

PROFESSOR_USUARIO = """Ano: {ano}. Habilidade: {habilidade}.
Questão: {enunciado}
{alternativas}
Alternativa correta: {letra}) {correta}"""


def papeis_padrao(modelo_professor="sabia-4", modelo_avaliador="sabia-4"):
    import agentes_questoes as aq
    return {
        "professor_resolucao": {"modelo": modelo_professor, "temperatura": 0.2,
                                "max_tokens": 4096 if "thinking" in modelo_professor else 900,
                                "versao_prompt": sv.versao_prompt(PROFESSOR_SISTEMA, PROFESSOR_USUARIO)},
        "avaliador_resolucao": {"modelo": modelo_avaliador, "temperatura": 0.0,
                                "max_tokens": 2048 if "thinking" in modelo_avaliador else 300,
                                "versao_prompt": sv.versao_prompt(aq.VALIDADOR_FASE2_SISTEMA, aq.VALIDADOR_USUARIO,
                                                                  aq.VALIDADOR_FASE2)},
    }


# ------------------------------------------------------------------ mensagens e parse
def mensagens_professor(q, ano, habilidade):
    alts = "\n".join(f"{L}) {q['alternativas'][L]}" for L in LETRAS if L in q["alternativas"])
    u = PROFESSOR_USUARIO.format(ano=ano, habilidade=habilidade, enunciado=q["enunciado"], alternativas=alts,
                                 letra=q["resposta_correta"], correta=q["alternativas"][q["resposta_correta"]])
    return [{"role": "system", "content": PROFESSOR_SISTEMA}, {"role": "user", "content": u}]


def mensagens_avaliador(q, resolucao):
    import agentes_questoes as aq
    u = (aq.VALIDADOR_USUARIO.format(enunciado=q["enunciado"], **{L: q["alternativas"].get(L, "") for L in LETRAS})
         + "\n\n" + aq.VALIDADOR_FASE2.format(letra=q["resposta_correta"], resolucao=resolucao))
    return [{"role": "system", "content": aq.VALIDADOR_FASE2_SISTEMA}, {"role": "user", "content": u}]


def parse_professor(texto):
    import agentes_questoes as aq
    o = aq.extrair_json(texto, ("resolucao", "discorda"), estrito=True)
    if not isinstance(o, dict) or not isinstance(o.get("discorda"), bool):
        return None
    if o["discorda"]:
        return {"discorda": True, "motivo": str(o.get("motivo", ""))[:300], "resolucao": None}
    r = o.get("resolucao")
    return {"discorda": False, "motivo": "", "resolucao": r.strip()} if isinstance(r, str) and r.strip() else None


# ------------------------------------------------------------------ verificação determinística
_RESULTADO = re.compile(r"[=≈]\s*(?:R\$\s*)?(-?\d[\d.,]*)")
# constantes didáticas e de conversão (dezena, centena, minutos, horas, dias, meses, %)
CONSTANTES = {Fraction(x) for x in (0, 1, 2, 10, 100, 1000, 60, 24, 7, 12, 30, 365)}


def _numeros(texto):
    return {x for x in su._numeros_do_enunciado_exatos(texto or "") if x is not None}


def validar_nova(q, nova):
    """(ok, motivos:list). q = questão ORIGINAL (com a resolução original)."""
    motivos = []
    if len(nova) < MIN_EXPLICATIVA or len(palavras_fora_das_contas(nova)) < MIN_PALAVRAS:
        motivos.append("pouco_explicativa")
    if REF_LETRA.search(nova):
        motivos.append("cita_letra")
    from padronizar_enunciado import eh_caixa_alta
    if eh_caixa_alta(nova):
        motivos.append("caixa_alta")
    t = dict(q, resolucao_passo_a_passo=nova)
    ok_novo, _s, motivo = su.check_consistency_detalhado(t)
    ok_orig, _s2, _m2 = su.check_consistency_detalhado(q)
    if ok_novo is False:
        motivos.append(f"verificador_reprova({motivo})")
    if su.resposta_fora_das_alternativas(t):
        motivos.append("resultado_fora_das_alternativas")
    if ok_orig is True and ok_novo is not True:
        motivos.append("perdeu_verificabilidade")
    if not cita_o_gabarito(t):
        motivos.append("nao_chega_ao_gabarito")
    conhecidos = (_numeros(q["enunciado"]) | _numeros(" ".join(q["alternativas"].values()))
                  | _numeros(q.get("resolucao_passo_a_passo", "")))
    novos = [Fraction(str(a["valor"])) for a in su.avisos_consistencia(t) if a.get("tipo") == "dado_inventado"]
    # O aviso do verificador perdoa número que a própria resolução escreveu antes em prosa ("comprou 9
    # ... 53 + 9"); aqui não: todo número da nova tem de vir do enunciado, das alternativas, da
    # resolução original, de um resultado "= r" da própria nova, ou ser uma constante de conversão.
    resultados = _numeros(" ".join(_RESULTADO.findall(nova)))
    soltos = _numeros(nova) - conhecidos - resultados - CONSTANTES
    if any(v not in conhecidos for v in novos) or soltos:
        motivos.append("dado_inventado")
    return not motivos, motivos


# ------------------------------------------------------------------ fluxo por item
def processar_item(item, q, ch):
    """Registro de proveniência do item (sempre devolve um, nunca altera q)."""
    h = hash_questao(q)
    reg = {"codigo_item": item["codigo_item"], "linha": item["linha"], "questao": item["questao"], "hash_questao": h,
           "ano": item["ano"], "habilidade": item["habilidade"], "origem": item["origem"],
           "classe_triagem": item["classe"], "gabarito": q["resposta_correta"],
           "resolucao_original": q.get("resolucao_passo_a_passo", ""), "resolucao_nova": None,
           "modelos": {p: c["modelo"] for p, c in ch.papeis.items()},
           "versoes_prompt": {p: c["versao_prompt"] for p, c in ch.papeis.items()}}
    import agentes_questoes as aq
    txt = ch.chamar("professor_resolucao", mensagens_professor(q, item["ano"], item["habilidade"]),
                    hash_exemplo=h, codigo_item=item["codigo_item"],
                    interpretavel=lambda t: parse_professor(t) is not None)
    if txt is None:
        reg.update(decisao="abstencao", motivo=ch.abstencoes.get(h, "sem_resposta"))
        return reg
    p = parse_professor(txt)
    if p["discorda"]:
        reg.update(decisao="revisao_humana", motivo="professor_discorda_do_gabarito", detalhe=p["motivo"])
        ch.registrar_resultado("professor_resolucao", h, item["codigo_item"], "discorda", p["motivo"])
        return reg
    reg["resolucao_nova"] = p["resolucao"]
    ok, motivos = validar_nova(q, p["resolucao"])
    reg["verificacoes"] = motivos
    if not ok:
        reg.update(decisao="rejeitada_verificador", motivo=",".join(motivos))
        ch.registrar_resultado("professor_resolucao", h, item["codigo_item"], "rejeitada", reg["motivo"])
        return reg
    txt2 = ch.chamar("avaliador_resolucao", mensagens_avaliador(q, p["resolucao"]), hash_exemplo=h,
                     codigo_item=item["codigo_item"], interpretavel=lambda t: aq.interpretar_fase2(t) is not None)
    if txt2 is None:
        reg.update(decisao="abstencao", motivo=ch.abstencoes.get(h, "sem_resposta_avaliador"))
        return reg
    f2 = aq.interpretar_fase2(txt2)
    reg["avaliador"] = {k: f2[k] for k in ("resolucao_correta", "usa_dado_ausente", "justificativa_falsa",
                                           "chega_na_alternativa")} | {"confianca": f2["confianca"]}
    aprovada = (f2["resolucao_correta"] and not f2["usa_dado_ausente"] and not f2["justificativa_falsa"]
                and f2["chega_na_alternativa"])
    reg.update(decisao="aprovada" if aprovada else "rejeitada_avaliador",
               motivo="" if aprovada else ",".join(c["codigo"] for c in f2["problemas"]) or "avaliador_reprova")
    ch.registrar_resultado("avaliador_resolucao", h, item["codigo_item"], reg["decisao"], reg["motivo"])
    return reg


# ------------------------------------------------------------------ dry-run
class ClienteSimuladoResolucao:
    """Sem rede. Professor: devolve uma explicação montada da resolução original (ou discorda se
    a questão tiver 'DISCORDA' no enunciado, para teste). Avaliador: aprova."""

    def __init__(self, base_por_hash):
        self.base = base_por_hash

    def chat_completion(self, messages, max_tokens, temperature):
        s, u = messages[0]["content"], messages[-1]["content"]
        if "escreve resoluções para alunos" in s:
            if "DISCORDA" in u:
                t = json.dumps({"discorda": True, "motivo": "simulado", "resolucao": ""})
            else:
                corr = u.rsplit("Alternativa correta: ", 1)[1].split(") ", 1)[1].strip()
                t = json.dumps({"discorda": False, "motivo": "",
                                "resolucao": f"(simulado) A pergunta pede o valor indicado no enunciado. "
                                             f"Usando os dados dados no texto, chegamos ao resultado. "
                                             f"Resposta: {corr}"}, ensure_ascii=False)
        else:
            t = json.dumps({"resolucao_correta": True, "usa_dado_ausente": False, "justificativa_falsa": False,
                            "chega_na_alternativa": True, "problemas": [], "confianca": 0.9})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=t), finish_reason="stop")],
                               usage=None)


# ------------------------------------------------------------------ CLI
TOKENS_MEDIDOS = {
    # medianas/p90 dos logs reais (outputs/agentes/escala/uso_api_*.jsonl, juizes_cache.jsonl), 2026-10-10:
    # avaliador = validador_fase2 (sabia-4): entrada ~470, saída 54/55. Professor sabia-4: entrada ~ o prompt
    # deste módulo (~650) e saída limitada a ~6 frases (~350; p90 600). Professor thinking: saída do gerador
    # thinking (920; p90 1500).
    "professor_resolucao": {"sabia-4": (650, 350, 600), "sabia-4-thinking": (650, 920, 1500), "sabiazinho-4": (650, 350, 600)},
    "avaliador_resolucao": {"sabia-4": (470, 54, 55), "sabia-4-thinking": (3180, 680, 900), "sabiazinho-4": (470, 54, 55)},
}


def fila(itens_triagem, classes=("correta_pouco_explicativa", "incompleta")):
    return [x for x in itens_triagem if x["classe"] in classes]


def ja_decididos(saida=SAIDA):
    """hash -> decisão final já gravada (abstenção não conta: volta na próxima rodada)."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(ROOT / "data" / "train_curado_v5.jsonl"))
    ap.add_argument("--real", action="store_true", help="chamadas pagas (exige --max-chamadas e --teto-brl)")
    ap.add_argument("--estimar", action="store_true", help="só imprime a estimativa de custo e sai")
    ap.add_argument("--max-chamadas", type=int, default=None)
    ap.add_argument("--teto-brl", type=float, default=None)
    ap.add_argument("--limite", type=int, default=None, help="processa só os K primeiros da fila (amostra/piloto)")
    ap.add_argument("--amostra-fixa", type=int, default=None,
                    help="K itens sorteados com seed fixa (protocolo de comparação de modelos)")
    ap.add_argument("--modelo-professor", default="sabia-4")
    ap.add_argument("--modelo-avaliador", default="sabia-4")
    ap.add_argument("--saida", default=str(SAIDA))
    args = ap.parse_args()

    base = {(i, j): q for i, j, _m, q, _p in carregar(args.base)}
    itens = []
    for i, j, m, q, _p in carregar(args.base):
        classe, _s = triar(q)
        from perfil_base import origem_de
        itens.append({"linha": i, "questao": j, "codigo_item": m["codigo_item"], "ano": m["ano"],
                      "habilidade": m["habilidade"], "origem": origem_de(m), "classe": classe})
    alvo = fila(itens)
    if args.amostra_fixa:
        alvo = random.Random(20261010).sample(alvo, min(args.amostra_fixa, len(alvo)))
    if args.limite:
        alvo = alvo[:args.limite]
    feitos = ja_decididos(args.saida)
    alvo = [x for x in alvo if hash_questao(base[(x["linha"], x["questao"])]) not in feitos]

    papeis = papeis_padrao(args.modelo_professor, args.modelo_avaliador)
    precos = sv.carregar_precos()
    est = sv.estimar_lote(len(alvo), {p: dict(c, chamadas_por_item=1) for p, c in papeis.items()}, precos,
                          {p: TOKENS_MEDIDOS[p][papeis[p]["modelo"]] for p in papeis})
    print(f"fila: {len(alvo)} itens (já decididos fora: {len(feitos)})")
    print("estimativa:", json.dumps(est, ensure_ascii=False))
    if args.estimar:
        return
    if args.real:
        if not args.max_chamadas or not args.teto_brl:
            sys.exit("ABORTADO: --real exige --max-chamadas N e --teto-brl R (tetos duros).")
        import agentes_questoes as aq
        clientes = {p: aq.criar_cliente_real("maritaca", c["modelo"]) for p, c in papeis.items()}
        ch = sv.ChamadorSabia(papeis, clientes, args.max_chamadas, args.teto_brl, precos, log_uso=LOG_USO, cache=CACHE)
        saida = Path(args.saida)
    else:
        sim = ClienteSimuladoResolucao({})
        ch = sv.ChamadorSabia(papeis, {p: sim for p in papeis}, max(1, 2 * len(alvo)), None, precos,
                              log_uso=SAIDA_DIR / "dryrun" / "uso_api_resolucoes.jsonl", cache=None, simulado=True)
        saida = SAIDA_DIR / "dryrun" / "resolucoes_novas.jsonl"
        print("DRY-RUN: sem rede, custo 0; saída em", saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    cont = {}
    for x in alvo:
        try:
            reg = processar_item(x, base[(x["linha"], x["questao"])], ch)
        except sv.OrcamentoEsgotado as e:
            print(f"PARADO: {e}. Nada foi decidido por falta de orçamento; rode de novo com teto maior (o cache "
                  f"evita pagar o que já foi feito).")
            break
        import agentes_questoes as aq
        with aq.abrir_para_gravar(saida, "a") as f:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        cont[reg["decisao"]] = cont.get(reg["decisao"], 0) + 1
    print("decisões:", cont, "| chamador:", ch.resumo())


if __name__ == "__main__":
    main()
