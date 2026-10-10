"""OBSERVAÇÕES informativas do gate de promoção (P0-4, 2026-10-10).

Nada aqui é critério: promover_checkpoint.py imprime e grava estas medidas
numa seção `observacoes` do veredito, mas elas NUNCA entram em `reprovados`,
em `ganhos` nem no PROMOVIDO / NÃO PROMOVIDO. Pela regra do projeto, critério
bloqueante só muda DEPOIS de um veredito reprovado e documentado no código;
estas medidas existem para que, quando esse dia chegar, já haja histórico
medido (baseline vs candidato) para fixar um limite com base em dados.

Três observações, todas sobre relatórios de `tests/test_model.py --batch`:

  1. cobertura_verificacao — % de questões entregues como `ok` (a conta da
     resolução foi checada por schema_utils.check_consistency) contra
     `nao_verificavel` (nenhuma conta reconhecível: o gate G3 não olha nada
     dessas), por rodada e agregado. Vem de `pos_processamento`, que todo
     relatório tem.
  2. variabilidade_entre_rodadas — para o MESMO item (codigo_item_ref) gerado
     em rodadas de seed diferentes, quão parecidos são os enunciados
     (Jaccard de trigramas de diversidade.jaccard; near-dup pelo mesmo
     critério de diversidade.e_near_duplicata). Mede se o modelo só repete a
     mesma questão trocando a seed. Precisa de >= 2 rodadas e do texto
     (`detalhes[].obj`, gravado a partir de 2026-10-10).
  3. vicios_de_texto — taxa de alternativa E "Nenhuma das alternativas
     anteriores" e de enunciado em CAIXA ALTA (herança do corpus SAEB).
     Precisa do texto.

Sem texto (relatórios antigos) as observações 2 e 3 valem "n/d", sem erro.
`calcular_seguro` nunca levanta: qualquer falha vira {"erro": ...} e o
veredito segue intacto.
"""
import re
from collections import defaultdict
from itertools import combinations

import diversidade as dv

STATUS = ("ok", "nao_verificavel", "corrigido", "depende_de_visual", "falha")
ND = "n/d"
LIMIAR_CAIXA_ALTA = 0.8   # fração de letras maiúsculas do enunciado
MIN_LETRAS_CAIXA_ALTA = 10  # enunciados mais curtos não contam (siglas, "R$ 5")

_NENHUMA = re.compile(
    r"\bnenhum[ao]?s?\b[^.]{0,25}\b(anteriores|anterior|acima|alternativas|opcoes|respostas)\b")


def _pct(a, b):
    return round(100 * a / b, 1) if b else None


# --------------------------------------------------------------------------
# Extração do texto (relatório novo: detalhes[].obj; avaliar_diversidade:
# modos[*].lotes[].questoes). Relatório antigo: nada -> lista vazia.
# --------------------------------------------------------------------------
def itens_com_texto(rel):
    """[(chave_do_item, questao_dict)] do relatório; [] se não houver texto."""
    out = []
    for d in (rel or {}).get("detalhes") or []:
        q = d.get("obj") if isinstance(d, dict) else None
        if isinstance(q, dict) and q.get("enunciado"):
            out.append((d.get("codigo_item_ref"), q))
    if out:
        return out
    modos = (rel or {}).get("modos")
    if isinstance(modos, dict) and modos:
        modo = rel.get("modo_atual") if rel.get("modo_atual") in modos else next(iter(modos))
        for lote in (modos[modo] or {}).get("lotes") or []:
            for i, q in enumerate(lote.get("questoes") or []):
                if isinstance(q, dict) and q.get("enunciado"):
                    out.append((f"{lote.get('id')}#{i}", q))
    return out


# --------------------------------------------------------------------------
# 1. cobertura de verificação
# --------------------------------------------------------------------------
def _linha_cobertura(contagem, n):
    return {"n": n,
            **{k: contagem.get(k, 0) for k in STATUS},
            "ok_pct": _pct(contagem.get("ok", 0), n),
            "nao_verificavel_pct": _pct(contagem.get("nao_verificavel", 0), n)}


def cobertura_verificacao(rels):
    por_rodada, total, n_total = [], defaultdict(int), 0
    for r in rels:
        pp = r.get("pos_processamento") or {}
        cont = {k: pp.get(k, 0) or 0 for k in STATUS}
        n = r.get("num_amostras") or sum(cont.values())
        if not n:
            continue
        por_rodada.append({"rodada": r.get("seed_rodada", 0), **_linha_cobertura(cont, n)})
        for k, v in cont.items():
            total[k] += v
        n_total += n
    if not por_rodada:
        return ND
    return {"por_rodada": por_rodada, "agregado": _linha_cobertura(total, n_total)}


# --------------------------------------------------------------------------
# 2. variabilidade entre rodadas do mesmo item
# --------------------------------------------------------------------------
def variabilidade_entre_rodadas(rels):
    if len(rels) < 2:
        return ND
    por_item = defaultdict(list)
    for r in rels:
        for chave, q in itens_com_texto(r):
            por_item[chave].append(q)
    jac, dup, itens = [], 0, 0
    for qs in por_item.values():
        if len(qs) < 2:
            continue
        itens += 1
        for a, b in combinations(qs, 2):
            sa = dv.shingles(dv.texto_questao(a, com_alternativas=False))
            sb = dv.shingles(dv.texto_questao(b, com_alternativas=False))
            jac.append(dv.jaccard(sa, sb))
            dup += int(dv.e_near_duplicata(a, b))
    if not jac:
        return ND
    sufixo = any((r.get("inferencia") or {}).get("sufixo") for r in rels)
    out = {"itens_comparados": itens, "pares": len(jac),
           "jaccard_medio": round(sum(jac) / len(jac), 3),
           "near_dup_pct": _pct(dup, len(jac)),
           "limiar_near_dup": dv.LIMIAR_NEAR_DUP,
           "prompt_identico_entre_rodadas": not sufixo}
    if sufixo:
        out["nota"] = ("relatório gerado com --sufixo: o sufixo de diversidade muda por "
                       "rodada, então os prompts NÃO são idênticos entre rodadas")
    return out


# --------------------------------------------------------------------------
# 3. vícios de texto
# --------------------------------------------------------------------------
def e_nenhuma_das_anteriores(q):
    alts = q.get("alternativas") if isinstance(q, dict) else None
    if not isinstance(alts, dict) or "E" not in alts:
        return False
    return bool(_NENHUMA.search(dv.normalizar_texto(alts["E"])))


def e_caixa_alta(q):
    letras = [c for c in str((q or {}).get("enunciado", "")) if c.isalpha()]
    if len(letras) < MIN_LETRAS_CAIXA_ALTA:
        return False
    return sum(c.isupper() for c in letras) / len(letras) >= LIMIAR_CAIXA_ALTA


def vicios_de_texto(rels):
    qs = [q for r in rels for _, q in itens_com_texto(r)]
    if not qs:
        return ND
    nen = [q for q in qs if e_nenhuma_das_anteriores(q)]
    alta = [q for q in qs if e_caixa_alta(q)]
    return {"questoes_com_texto": len(qs),
            "nenhuma_das_anteriores_n": len(nen),
            "nenhuma_das_anteriores_pct": _pct(len(nen), len(qs)),
            "nenhuma_e_o_gabarito_n": sum(q.get("resposta_correta") == "E" for q in nen),
            "caixa_alta_n": len(alta),
            "caixa_alta_pct": _pct(len(alta), len(qs))}


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
def _lado(rels):
    return {"cobertura_verificacao": cobertura_verificacao(rels),
            "variabilidade_entre_rodadas": variabilidade_entre_rodadas(rels),
            "vicios_de_texto": vicios_de_texto(rels)}


def observacoes(bases, cands):
    """bases/cands: listas de relatórios (1 por rodada; n1 = lista de 1)."""
    return {"nota": ("OBSERVAÇÃO (não bloqueante): não entra no veredito. "
                     "n/d = relatório sem o dado (sem texto ou sem >= 2 rodadas)."),
            "baseline": _lado(bases), "candidato": _lado(cands)}


def calcular_seguro(bases, cands):
    """Como observacoes(), mas jamais levanta: o gate não pode cair por causa
    de uma medida informativa."""
    try:
        return observacoes(bases, cands)
    except Exception as e:  # noqa: BLE001 — proposital, ver docstring
        return {"erro": f"{type(e).__name__}: {e}"}


def _fmt(v):
    return ND if v is None else str(v)


def formatar(obs):
    """Linhas para imprimir ao final do veredito."""
    rot = "OBSERVAÇÃO (não bloqueante)"
    if "erro" in obs:
        return [f"{rot}: não calculada ({obs['erro']})"]
    L = []
    for lado in ("baseline", "candidato"):
        c = obs[lado]["cobertura_verificacao"]
        if c == ND:
            L.append(f"{rot} cobertura de verificação [{lado}]: {ND}")
            continue
        ag = c["agregado"]
        rod = ", ".join(f"r{x['rodada']}: ok {x['ok_pct']}% / nv {x['nao_verificavel_pct']}%"
                        for x in c["por_rodada"])
        L.append(f"{rot} cobertura de verificação [{lado}]: agregado ok {ag['ok_pct']}% / "
                 f"nao_verificavel {ag['nao_verificavel_pct']}% (n={ag['n']})  | {rod}")
    for lado in ("baseline", "candidato"):
        v = obs[lado]["variabilidade_entre_rodadas"]
        if v == ND:
            L.append(f"{rot} variabilidade entre rodadas [{lado}]: {ND}")
        else:
            L.append(f"{rot} variabilidade entre rodadas [{lado}]: Jaccard médio "
                     f"{v['jaccard_medio']}, near-dup {v['near_dup_pct']}% "
                     f"({v['pares']} pares, {v['itens_comparados']} itens)"
                     + (f"  [{v['nota']}]" if v.get("nota") else ""))
    for lado in ("baseline", "candidato"):
        t = obs[lado]["vicios_de_texto"]
        if t == ND:
            L.append(f"{rot} vícios de texto [{lado}]: {ND}")
        else:
            L.append(f"{rot} vícios de texto [{lado}]: E='nenhuma das anteriores' "
                     f"{_fmt(t['nenhuma_das_anteriores_pct'])}% ({t['nenhuma_das_anteriores_n']}/"
                     f"{t['questoes_com_texto']}), enunciado em CAIXA ALTA "
                     f"{_fmt(t['caixa_alta_pct'])}% ({t['caixa_alta_n']}/{t['questoes_com_texto']})")
    return L
