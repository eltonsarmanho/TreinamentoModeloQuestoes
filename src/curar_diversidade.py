"""
Curadoria de diversidade do conjunto de treino.

Etapas (todas determinísticas, só biblioteca padrão + src/diversidade.py):

  1. Auditoria: classifica cada questão de data/train.jsonl com
     diversidade.classificar_questao, usando (ano, habilidade) do meta — ou do
     prompt do usuário quando o meta não tiver. A chave é SEMPRE (ano,
     habilidade): o mesmo código H muda de sentido conforme o ano.
  2. Relatório por (ano, habilidade): distribuição de subtema, tipo de
     raciocínio, contexto, estrutura, dificuldade, origem e duplicate_rate;
     sinaliza concentração (subtema dominante > LIMIAR_CONCENTRACAO com K>=2).
     Saída: outputs/curadoria_diversidade.json e .md.
  3. Plano de rebalanceamento: (a) quais exemplos SINTÉTICOS/DESTILADOS são
     near-duplicatas redundantes (reais do banco nunca são removidos) e
     (b) quantos exemplos novos pedir ao professor por subtema
     sub-representado. Saída: data/plano_destilacao.json.
  4. --apply: grava data/train_curado.jsonl sem as near-dups redundantes e,
     com --sufixo, anexa diversidade.sufixo_prompt ao prompt do usuário dos
     exemplos de 1 questão com subtema conhecido (ensina o condicionamento).
     NUNCA sobrescreve train.jsonl nem toca val*.jsonl. Exemplos minoritários
     não são duplicados — o déficit vai para o plano do professor.

Uso:
    venv/bin/python src/curar_diversidade.py              # auditoria + plano
    venv/bin/python src/curar_diversidade.py --apply --sufixo
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversidade as dv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "train.jsonl"
CURADO_PATH = ROOT / "data" / "train_curado.jsonl"
PLANO_PATH = ROOT / "data" / "plano_destilacao.json"
RELATORIO_JSON = ROOT / "outputs" / "curadoria_diversidade.json"
RELATORIO_MD = ROOT / "outputs" / "curadoria_diversidade.md"

# Subtema dominante acima disso (com K>=2) é concentração excessiva.
LIMIAR_CONCENTRACAO = 0.60
# Abaixo disso a proporção é ruído (1 item muda >30 p.p.): não sinaliza.
MIN_N_CONCENTRACAO = 3
# Piso de exemplos por subtema: com menos que isso o modelo mal vê o subtema.
PISO_POR_SUBTEMA = 4
# Teto de pedidos ao professor por (ano, hab, subtema) numa rodada — evita que
# uma habilidade com 1 item real receba dezenas de destilados de uma vez.
TETO_PEDIDO = 12

_RX_USER = re.compile(r"Ano:\s*(\S+)\s*ano\.\s*Habilidade:\s*([^\s—]+)")


# --------------------------------------------------------------------------
# Leitura
# --------------------------------------------------------------------------

def origem(meta):
    """'destilado' | 'sintetico' | 'real'. Só as duas primeiras são removíveis."""
    if meta.get("destilado"):
        return "destilado"
    if meta.get("sintetico"):
        return "sintetico"
    return "real"


def _msg(ex, role):
    for m in ex.get("messages", []):
        if m.get("role") == role:
            return m.get("content", "")
    return ""


def ano_habilidade(ex):
    """(ano, habilidade) do meta; se faltar, extrai do prompt do usuário."""
    meta = ex.get("meta") or {}
    ano, hab = meta.get("ano"), meta.get("habilidade")
    if not (ano and hab):
        m = _RX_USER.search(_msg(ex, "user"))
        if m:
            ano, hab = ano or m.group(1), hab or m.group(2)
    return ano, hab


def questoes_do_exemplo(ex):
    """Lista de questões do JSON do assistant ([] se não parsear)."""
    try:
        obj = json.loads(_msg(ex, "assistant"))
    except (json.JSONDecodeError, TypeError):
        return []
    qs = obj.get("questoes") if isinstance(obj, dict) else None
    return [q for q in qs if isinstance(q, dict)] if isinstance(qs, list) else []


def carregar_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# --------------------------------------------------------------------------
# Auditoria
# --------------------------------------------------------------------------

def auditar(exemplos, taxonomia=None):
    """Classifica cada questão. Retorna lista de registros por questão:
    {ex_idx, q_idx, ano, habilidade, origem, dificuldade, questao, **classificacao}."""
    tax = taxonomia or dv.carregar_taxonomia()
    regs = []
    for i, ex in enumerate(exemplos):
        ano, hab = ano_habilidade(ex)
        meta = ex.get("meta") or {}
        for j, q in enumerate(questoes_do_exemplo(ex)):
            c = dv.classificar_questao(q, ano, hab, tax)
            regs.append({"ex_idx": i, "q_idx": j, "ano": ano, "habilidade": hab,
                         "origem": origem(meta), "dificuldade": meta.get("dificuldade") or q.get("difficulty"),
                         "questao": q, **c})
    return regs


def _dist(regs, campo):
    c = Counter(r[campo] for r in regs)
    n = sum(c.values())
    return {k: {"n": v, "pct": round(100 * v / n, 1)} for k, v in c.most_common()}


def near_dups_redundantes(regs_grupo, limiar=dv.LIMIAR_NEAR_DUP):
    """Índices (ex_idx) de exemplos removíveis dentro de um grupo (ano, hab).

    Ordem de preferência: reais entram primeiro no conjunto 'mantido' (nunca
    saem); depois sintéticos/destilados em ordem de arquivo. Um exemplo
    removível sai só se TODAS as suas questões forem near-dup de algo já
    mantido — um exemplo com lote misto tem ao menos uma questão nova e fica.
    """
    por_ex = defaultdict(list)
    for r in regs_grupo:
        por_ex[r["ex_idx"]].append(r)
    ordem = sorted(por_ex, key=lambda e: (por_ex[e][0]["origem"] != "real", e))
    mantidas, remover = [], set()
    for e in ordem:
        qs = [r["questao"] for r in por_ex[e]]
        if por_ex[e][0]["origem"] != "real" and mantidas and all(
                any(dv.e_near_duplicata(q, m, limiar) for m in mantidas) for q in qs):
            remover.add(e)
            continue
        mantidas.extend(qs)
    return remover


def relatorio_grupo(ano, hab, regs, tax, limiar=dv.LIMIAR_NEAR_DUP):
    """Métricas de um (ano, habilidade) + plano de rebalanceamento do grupo."""
    h = dv.obter_habilidade(ano, hab, tax)
    subtemas = h["subtemas"] if h else []
    k = len(subtemas)
    qs = [r["questao"] for r in regs]
    dist_sub = Counter(r["subtema"] for r in regs)
    classificadas = sum(v for s, v in dist_sub.items() if s != "outros")
    dominante, n_dom = (None, 0)
    if classificadas:
        dominante, n_dom = max(((s, v) for s, v in dist_sub.items() if s != "outros"),
                               key=lambda x: (x[1], x[0]))
    share = n_dom / classificadas if classificadas else 0.0
    concentrado = k >= 2 and classificadas >= MIN_N_CONCENTRACAO and share > LIMIAR_CONCENTRACAO

    remover = near_dups_redundantes(regs, limiar)
    # contagem pós-remoção por subtema: é sobre ela que calculamos o déficit
    pos = Counter(r["subtema"] for r in regs if r["ex_idx"] not in remover)
    n_pos = sum(pos.values())
    alvo = max(PISO_POR_SUBTEMA, math.ceil(n_pos / k)) if k else 0
    pedidos = []
    for s in subtemas:
        falta = max(0, alvo - pos.get(s["id"], 0))
        if falta:
            pedidos.append({"subtema": s["id"], "subtema_rotulo": s["rotulo"],
                            "atual": pos.get(s["id"], 0), "alvo": alvo,
                            "quantidade": min(falta, TETO_PEDIDO),
                            "tipos_raciocinio": s.get("tipos_raciocinio") or h["tipos_raciocinio"]})
    rem_por_sub = Counter(r["subtema"] for r in regs if r["ex_idx"] in remover)
    return {
        "ano": ano, "habilidade": hab, "k_subtemas": k,
        "descricao": (h["descricoes"][0] if h and h.get("descricoes") else ""),
        "n_questoes": len(regs), "n_exemplos": len({r["ex_idx"] for r in regs}),
        "origem": dict(Counter(r["origem"] for r in regs)),
        "subtema": _dist(regs, "subtema"),
        "tipo_raciocinio": _dist(regs, "tipo_raciocinio"),
        "contexto": _dist(regs, "contexto"),
        "estrutura": _dist(regs, "estrutura"),
        "dificuldade": _dist(regs, "dificuldade"),
        "duplicate_rate": round(dv.duplicate_rate(qs, limiar), 3),
        "coverage_score": round(dv.coverage_score([r["subtema"] for r in regs], k), 3) if k else 0.0,
        "subtemas_ausentes": [s["id"] for s in subtemas if not dist_sub.get(s["id"])],
        "dominante": dominante, "dominante_pct": round(100 * share, 1),
        "concentracao_excessiva": concentrado,
        "remover_ex_idx": sorted(remover),
        "remover_questoes_por_subtema": dict(rem_por_sub),
        "pedidos_professor": pedidos,
    }


def curar(exemplos, taxonomia=None, limiar=dv.LIMIAR_NEAR_DUP, incluir_ausentes=False):
    """Auditoria completa. Retorna (relatorio, registros). `incluir_ausentes`
    adiciona ao plano os grupos da taxonomia sem nenhum exemplo no treino."""
    tax = taxonomia or dv.carregar_taxonomia()
    regs = auditar(exemplos, tax)
    grupos = defaultdict(list)
    for r in regs:
        grupos[(r["ano"], r["habilidade"])].append(r)
    # Grupos da taxonomia SEM nenhum exemplo no treino também precisam de plano:
    # sem isto, uma habilidade ausente do treino (ex.: 5º H21/H22, tabelas e
    # gráficos) nunca é pedida ao professor e o modelo nunca a aprende.
    for chave in (tax.get("habilidades", {}) if incluir_ausentes else ()):
        a, h = chave.split("|", 1)
        grupos.setdefault((a, h), [])
    por_grupo = [relatorio_grupo(a, h, grupos[(a, h)], tax, limiar) for a, h in sorted(grupos, key=str)]
    remover = sorted(e for g in por_grupo for e in g["remover_ex_idx"])
    rel = {
        "n_exemplos": len(exemplos), "n_questoes": len(regs),
        "origem_exemplos": dict(Counter(origem(ex.get("meta") or {}) for ex in exemplos)),
        "limiar_near_dup": limiar, "limiar_concentracao": LIMIAR_CONCENTRACAO,
        "n_grupos": len(por_grupo),
        "grupos_concentrados": [f"{g['ano']}|{g['habilidade']}" for g in por_grupo if g["concentracao_excessiva"]],
        "n_remover": len(remover), "remover_ex_idx": remover,
        "n_pedidos_professor": sum(p["quantidade"] for g in por_grupo for p in g["pedidos_professor"]),
        "grupos": por_grupo,
    }
    return rel, regs


# --------------------------------------------------------------------------
# Plano / apply
# --------------------------------------------------------------------------

def plano_destilacao(rel):
    """Plano consumível por distill_teacher: uma entrada por (ano, hab, subtema)
    sub-representado, com a descrição da habilidade e os tipos de raciocínio
    compatíveis (o professor gira entre eles)."""
    itens = []
    for g in rel["grupos"]:
        for p in g["pedidos_professor"]:
            itens.append({"ano": g["ano"], "habilidade": g["habilidade"], "descricao": g["descricao"], **p})
    return {"versao": 1, "fonte": "src/curar_diversidade.py",
            "nota": "quantidade = exemplos novos a pedir ao professor por subtema; "
                    "a chave é sempre (ano, habilidade, subtema).",
            "piso_por_subtema": PISO_POR_SUBTEMA, "teto_pedido": TETO_PEDIDO,
            "total": sum(i["quantidade"] for i in itens), "itens": itens}


def slot_de_classificacao(c, ano, hab, taxonomia=None):
    """Monta um 'slot' (formato de planejar_lote) a partir da classificação de
    uma questão existente, para gerar o mesmo sufixo usado na inferência.
    None se o subtema for desconhecido (não ensinamos condicionamento falso)."""
    h = dv.obter_habilidade(ano, hab, taxonomia)
    if not h or c["subtema"] == "outros":
        return None
    rot = {s["id"]: s["rotulo"] for s in h["subtemas"]}
    if c["subtema"] not in rot:
        return None
    return {"subtema": c["subtema"], "subtema_rotulo": rot[c["subtema"]],
            "tipo_raciocinio": c["tipo_raciocinio"],
            "tipo_raciocinio_rotulo": (dv._rotulo(dv.TIPOS_RACIOCINIO, c["tipo_raciocinio"])
                                       if c["tipo_raciocinio"] != "indefinido" else "livre"),
            "contexto": c["contexto"],
            "contexto_rotulo": (dv._rotulo(dv.CONTEXTOS, c["contexto"])
                                if c["contexto"] != dv.SEM_CONTEXTO else "sem contexto narrativo")}


def aplicar(exemplos, rel, regs, sufixo=False, taxonomia=None):
    """Nova lista de exemplos: sem os removíveis e (opcional) com sufixo de
    subtema no user. Não altera os dicts de entrada e nunca duplica exemplos."""
    remover = set(rel["remover_ex_idx"])
    por_ex = defaultdict(list)
    for r in regs:
        por_ex[r["ex_idx"]].append(r)
    saida, n_sufixo = [], 0
    for i, ex in enumerate(exemplos):
        if i in remover:
            continue
        novo = json.loads(json.dumps(ex, ensure_ascii=False))
        rs = por_ex.get(i, [])
        # sufixo descreve UM slot: só faz sentido em exemplo de 1 questão
        if sufixo and len(rs) == 1:
            slot = slot_de_classificacao(rs[0], rs[0]["ano"], rs[0]["habilidade"], taxonomia)
            if slot:
                for m in novo["messages"]:
                    if m["role"] == "user":
                        m["content"] = m["content"] + dv.sufixo_prompt(slot)
                novo.setdefault("meta", {}).update(
                    {"subtema": slot["subtema"], "tipo_raciocinio": slot["tipo_raciocinio"],
                     "contexto": slot["contexto"]})
                n_sufixo += 1
        saida.append(novo)
    return saida, n_sufixo


# --------------------------------------------------------------------------
# Relatório markdown
# --------------------------------------------------------------------------

def _fmt(d, n=4):
    return ", ".join(f"{k} {v['pct']}%" for k, v in list(d.items())[:n])


def relatorio_md(rel):
    L = ["# Curadoria de diversidade — data/train.jsonl", "",
         f"- Exemplos: {rel['n_exemplos']} ({rel['origem_exemplos']}); questões: {rel['n_questoes']}",
         f"- Grupos (ano, habilidade): {rel['n_grupos']}; com concentração > "
         f"{int(100 * rel['limiar_concentracao'])}% (K>=2): {len(rel['grupos_concentrados'])}",
         f"- Near-dups redundantes (só sintético/destilado) a remover: {rel['n_remover']} exemplos",
         f"- Pedidos ao professor (plano_destilacao.json): {rel['n_pedidos_professor']} exemplos", "",
         "| ano | hab | n | K | subtemas | dominante | dup_rate | ausentes | remover | pedir |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for g in rel["grupos"]:
        flag = " **!**" if g["concentracao_excessiva"] else ""
        L.append(f"| {g['ano']} | {g['habilidade']} | {g['n_questoes']} | {g['k_subtemas']} | "
                 f"{_fmt(g['subtema'])} | {g['dominante']} {g['dominante_pct']}%{flag} | "
                 f"{g['duplicate_rate']} | {', '.join(g['subtemas_ausentes'])} | "
                 f"{len(g['remover_ex_idx'])} | {sum(p['quantidade'] for p in g['pedidos_professor'])} |")
    L += ["", "Dominante % é calculado sobre questões com subtema reconhecido ('outros' excluído).",
          "Classificação por léxico/regex: ruidosa em grupos pequenos (n<=8)."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(TRAIN_PATH))
    ap.add_argument("--apply", action="store_true", help="grava data/train_curado.jsonl")
    ap.add_argument("--sufixo", action="store_true", help="com --apply: anexa sufixo de subtema ao user")
    ap.add_argument("--saida", default=str(CURADO_PATH))
    ap.add_argument("--plano", default=str(PLANO_PATH))
    ap.add_argument("--relatorio", default=str(RELATORIO_JSON))
    ap.add_argument("--so-presentes", action="store_true",
                    help="plano só para grupos que já têm exemplo no treino (padrão: inclui os ausentes)")
    ap.add_argument("--limiar", type=float, default=dv.LIMIAR_NEAR_DUP)
    a = ap.parse_args(argv)

    saida = Path(a.saida).resolve()
    # trava de segurança: nunca escrever sobre o treino original nem sobre val*
    if saida == Path(a.train).resolve() or saida.name.startswith("val") or saida == TRAIN_PATH.resolve():
        ap.error(f"saída proibida: {saida}")

    exemplos = carregar_jsonl(a.train)
    rel, regs = curar(exemplos, limiar=a.limiar, incluir_ausentes=not a.so_presentes)
    rj = Path(a.relatorio)
    rj.parent.mkdir(parents=True, exist_ok=True)
    rj.write_text(json.dumps(rel, ensure_ascii=False, indent=1), encoding="utf-8")
    rj.with_suffix(".md").write_text(relatorio_md(rel), encoding="utf-8")
    Path(a.plano).write_text(json.dumps(plano_destilacao(rel), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"relatório: {rj} (+ .md); plano: {a.plano}")
    print(f"{rel['n_exemplos']} exemplos, {rel['n_questoes']} questões, "
          f"{len(rel['grupos_concentrados'])} grupos concentrados, "
          f"{rel['n_remover']} removíveis, {rel['n_pedidos_professor']} pedidos ao professor")
    if a.apply:
        novos, n_suf = aplicar(exemplos, rel, regs, sufixo=a.sufixo)
        with open(saida, "w", encoding="utf-8") as f:
            for ex in novos:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"curado: {saida} ({len(novos)} exemplos, {n_suf} com sufixo)")


if __name__ == "__main__":
    main()
