"""
Relatório sumário da base de conhecimento (data/train_curado.jsonl).

Por que existe: antes de injetar questões novas (padrão SAEB) é preciso saber
ONDE a base é fraca — por ano e por (ano, habilidade) — e com que critério se
diz "fraca". Este script só LÊ: nunca altera train*.jsonl, val*.jsonl, a
taxonomia nem o banco. É reexecutável: rode de novo depois de cada injeção
para ver o déficit cair.

O que mede, por (ano, habilidade) — a chave é SEMPRE o par, porque o mesmo
código H muda de sentido conforme o ano:
  - nº de exemplos/questões e ORIGEM:
      real_banco  -> codigo_item existe em DB/questoes.db (itens MT####..., da
                     matriz SAEB, e EFxxMAxx-NNN-L2-..., lote de professor);
      sintetico   -> meta.sintetico / codigo_item "SINT-..." (templates);
      destilado   -> meta.destilado / codigo_item "DIST-..." (modelo professor);
      real_fora_do_banco -> nenhum dos anteriores (não deveria ocorrer; alerta).
  - dificuldade (Fácil/Moderado/Difícil, do meta);
  - cobertura de subtemas da taxonomia via diversidade.classificar_questao;
  - near-duplicatas (diversidade.e_near_duplicata) dentro da habilidade;
  - verificabilidade do gabarito (schema_utils.check_consistency_detalhado);
  - itens reais disponíveis no banco (âncoras para a injeção);
  - META proposta e DÉFICIT (ver calcular_meta).

Saídas: outputs/relatorio_base.json (completo) e Doc/RELATORIO_BASE.md.

Uso:
    venv/bin/python src/relatorio_base.py
    venv/bin/python src/relatorio_base.py --train data/train_curado_v3.jsonl \
        --saida-json outputs/relatorio_base_v3.json --saida-md Doc/RELATORIO_BASE_v3.md
"""
import argparse
import inspect
import json
import math
import sqlite3
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diversidade as dv  # noqa: E402  (somente leitura)
import schema_utils as su  # noqa: E402  (somente leitura)

# O verificador de geometria está sendo escrito por OUTRO workflow em paralelo.
# Import opcional: se não existir (ou quebrar no import), o relatório segue sem
# ele e registra "indisponível" — nunca deve derrubar o relatório.
try:  # pragma: no cover - depende do estado do outro workflow
    import verificador_geometria as vg  # noqa: E402
except Exception:  # noqa: BLE001
    vg = None

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "train_curado.jsonl"
VAL_PATHS = [ROOT / "data" / n for n in ("val.jsonl", "val_frozen_v1.jsonl", "val_novos_v1.jsonl")]
DB_PATH = ROOT / "DB" / "questoes.db"
SAIDA_JSON = ROOT / "outputs" / "relatorio_base.json"
SAIDA_MD = ROOT / "Doc" / "RELATORIO_BASE.md"

DIFICULDADES = ("Fácil", "Moderado", "Difícil")
ORDEM_ANOS = ("1º", "2º", "3º", "4º", "5º", "9º")

# ---------------------------------------------------------------------------
# Critério da META (explícito e parametrizável pela CLI)
# ---------------------------------------------------------------------------
# 30 por habilidade: com 3 dificuldades dá 10 por (habilidade, dificuldade) —
# o prompt de treino é condicionado nesse par, e abaixo de ~10 o modelo tende a
# decorar o exemplo em vez de aprender o padrão. É também ~4x o que o SAEB real
# oferece por descritor no banco (8), o que mantém a âncora real relevante.
META_MIN_HABILIDADE = 30
# 3 por subtema: com 1 exemplo o modelo copia; com 2 ainda pode interpolar entre
# dois moldes; 3 é o mínimo para haver variação de contexto/estrutura dentro do
# subtema. Habilidade com muitos subtemas (K) sobe a meta para 3*K.
MIN_POR_SUBTEMA = 3
# Equilíbrio de dificuldade: meta dividida igualmente entre as 3 (arredondada
# para múltiplo de 3). Desequilíbrio ensina o modelo a ignorar a dificuldade pedida.

# ---------------------------------------------------------------------------
# Limiares de ALERTA
# ---------------------------------------------------------------------------
# Participação de uma dificuldade fora de [15%, 60%] (com n>=6) é desequilíbrio:
# no equilíbrio perfeito cada uma teria 33%.
ALERTA_DIF_MIN, ALERTA_DIF_MAX, ALERTA_DIF_N = 0.15, 0.60, 6
# >=10% de near-duplicatas: a habilidade tem molde repetido com números trocados.
ALERTA_DUP = 0.10
# >=30% das questões caem em "outros": a taxonomia não cobre o que a base tem
# (ou o classificador é cego para a forma de escrever da habilidade).
ALERTA_OUTROS = 0.30
# >=80% de gabarito "não verificável" (n>=3): check_consistency é cego para a
# habilidade (classificação, leitura, geometria conceitual). É onde o erro do
# caso 9º H17 auditado ("tem uma propriedade de X => é X") passa sem ser visto,
# e onde a validação por agente/verificador conceitual é obrigatória.
ALERTA_CEGO, ALERTA_CEGO_N = 0.80, 3
# Letra correta fora de [10%, 30%] no global: o modelo aprende o viés de posição
# (itens reais têm 4 alternativas e o "E" é sempre "Nenhuma das anteriores").
ALERTA_LETRA_MIN, ALERTA_LETRA_MAX = 0.10, 0.30


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------

def carregar_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(linha) for linha in f if linha.strip()]


def _msg(ex, role):
    for m in ex.get("messages", []):
        if m.get("role") == role:
            return m.get("content", "")
    return ""


def questoes_do_exemplo(ex):
    """Questões do JSON do assistant; None se o JSON não parsear (conta como
    defeito estrutural, diferente de lista vazia)."""
    try:
        obj = json.loads(_msg(ex, "assistant"))
    except (json.JSONDecodeError, TypeError):
        return None
    qs = obj.get(su.QUESTOES_KEY) if isinstance(obj, dict) else None
    return [q for q in qs if isinstance(q, dict)] if isinstance(qs, list) else None


def origem_exemplo(meta, codigos_db):
    """real_banco | sintetico | destilado | real_fora_do_banco.

    A flag do meta tem precedência; o prefixo do codigo_item cobre exemplos
    antigos em que a flag não foi gravada. "Real" só é afirmado se o código
    existe no banco — é isso que permite dizer que há âncora humana.
    """
    codigo = str(meta.get("codigo_item") or "")
    if meta.get("destilado") or codigo.startswith("DIST-"):
        return "destilado"
    if meta.get("sintetico") or codigo.startswith("SINT-"):
        return "sintetico"
    if codigo in codigos_db:
        return "real_banco"
    return "real_fora_do_banco"


def carregar_banco(db_path):
    """Itens do banco por (ano, habilidade) + descrição mais frequente.

    'textual' = sem imagem e sem flag depende_de_imagem: só esses servem de
    âncora para um gerador que exige questão autocontida (sem figura).
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        linhas = con.execute(
            "SELECT codigo_item, ano, habilidade, descricao_item, "
            "imagem IS NOT NULL, COALESCE(depende_de_imagem, 0), grau_resolucao, lote "
            "FROM itens").fetchall()
    finally:
        con.close()
    itens = defaultdict(list)
    descr = defaultdict(Counter)
    sem_hab = 0
    for cod, ano, hab, desc, tem_img, dep_img, grau, lote in linhas:
        if not ano or not hab or str(hab).lower() == "nan" or str(ano).lower() == "nan":
            sem_hab += 1
            continue
        itens[(ano, hab)].append({"codigo_item": cod, "textual": not tem_img and not dep_img,
                                  "grau": grau, "lote": lote})
        if desc and str(desc).strip() and str(desc).lower() != "nan":
            descr[(ano, hab)][str(desc).strip()] += 1
    descricoes = {k: c.most_common(1)[0][0] for k, c in descr.items()}
    codigos = {it["codigo_item"] for lst in itens.values() for it in lst}
    return {"itens": itens, "descricoes": descricoes, "codigos": codigos,
            "total": len(linhas), "sem_habilidade": sem_hab}


# ---------------------------------------------------------------------------
# Verificador de geometria (opcional)
# ---------------------------------------------------------------------------

def _veredito_geometria(questao, ano, habilidade):
    """Chama o verificador do outro workflow se houver uma função reconhecível.

    A API dele ainda não é conhecida aqui; por isso só se passa os parâmetros
    que a assinatura declara e o retorno é normalizado com cautela. Qualquer
    falha vira None (= "não opinou"), nunca reprovação.
    """
    if vg is None:
        return None
    fn = next((getattr(vg, n) for n in ("verificar_questao", "verificar", "avaliar_questao")
               if callable(getattr(vg, n, None))), None)
    if fn is None:
        return None
    try:
        params = inspect.signature(fn).parameters
        kw = {k: v for k, v in (("ano", ano), ("habilidade", habilidade)) if k in params}
        r = fn(questao, **kw)
    except Exception:  # noqa: BLE001 - módulo alheio em construção
        return None
    if isinstance(r, tuple) and r:
        r = r[0]
    if isinstance(r, dict):
        for chave in ("ok", "valida", "aprovada", "veredito"):
            if chave in r:
                r = r[chave]
                break
    if isinstance(r, str):
        r = {"ok": True, "aprovada": True, "valida": True, "verdadeiro": True,
             "reprovada": False, "invalida": False, "falso": False}.get(r.lower())
    return r if isinstance(r, bool) else None


# ---------------------------------------------------------------------------
# Meta e déficit
# ---------------------------------------------------------------------------

def calcular_meta(k_subtemas, meta_min=META_MIN_HABILIDADE, min_sub=MIN_POR_SUBTEMA):
    """Meta por habilidade = max(meta_min, min_sub*K), arredondada para cima
    para múltiplo de 3 (divisão exata entre as 3 dificuldades)."""
    bruta = max(meta_min, min_sub * max(k_subtemas, 1))
    return int(math.ceil(bruta / 3) * 3)


def calcular_deficit(meta, n_efetivo, por_dif, por_sub, subtemas_ids, min_sub=MIN_POR_SUBTEMA):
    """Déficit mínimo de questões NOVAS para cumprir os três critérios.

    Cada questão nova tem UMA dificuldade e UM subtema, livremente escolhidos;
    logo o mínimo que satisfaz tudo ao mesmo tempo é o MAIOR dos três déficits
    (total, soma por dificuldade, soma por subtema) — não a soma deles.
    """
    alvo_dif = meta // 3
    falta_dif = {d: max(0, alvo_dif - por_dif.get(d, 0)) for d in DIFICULDADES}
    falta_sub = {s: max(0, min_sub - por_sub.get(s, 0)) for s in subtemas_ids}
    total = max(meta - n_efetivo, sum(falta_dif.values()), sum(falta_sub.values()), 0)
    return {"alvo_por_dificuldade": alvo_dif, "falta_por_dificuldade": falta_dif,
            "falta_por_subtema": {s: v for s, v in falta_sub.items() if v},
            "deficit": total}


# ---------------------------------------------------------------------------
# Análise
# ---------------------------------------------------------------------------

_PRIORIDADE_ORIGEM = {"real_banco": 0, "real_fora_do_banco": 1, "sintetico": 2, "destilado": 3}


def analisar(exemplos, banco, taxonomia, val_codigos, meta_min, min_sub):
    regs = []          # um registro por QUESTÃO
    json_invalido = []
    for i, ex in enumerate(exemplos):
        meta = ex.get("meta") or {}
        ano, hab = meta.get("ano"), meta.get("habilidade")
        org = origem_exemplo(meta, banco["codigos"])
        qs = questoes_do_exemplo(ex)
        if not qs:
            json_invalido.append(meta.get("codigo_item") or f"linha {i + 1}")
            continue
        for j, q in enumerate(qs):
            ok, sug, motivo = su.check_consistency_detalhado(q)
            cls = dv.classificar_questao(q, ano, hab, taxonomia)
            dif_meta = meta.get("dificuldade")
            regs.append({
                "linha": i + 1, "q_idx": j, "codigo_item": meta.get("codigo_item"),
                "ano": ano, "habilidade": hab, "origem": org, "dificuldade": dif_meta,
                "difficulty_campo": q.get("difficulty"),
                "difficulty_divergente": (dif_meta in su.DIFFICULTY_MAP
                                          and q.get("difficulty") != su.DIFFICULTY_MAP[dif_meta]),
                "subtema": cls["subtema"], "subtema_meta": meta.get("subtema"),
                "tipo_raciocinio": cls["tipo_raciocinio"],
                "gabarito": "ok" if ok is True else ("inconsistente" if ok is False else "nao_verificavel"),
                "motivo_gabarito": motivo, "sugestao_gabarito": sug,
                "letra": q.get("resposta_correta"),
                "geometria": _veredito_geometria(q, ano, hab),
                "_q": q,
            })

    grupos = defaultdict(list)
    for r in regs:
        grupos[(r["ano"], r["habilidade"])].append(r)

    habs = []
    chaves = set(grupos) | {tuple(k.split("|", 1)) for k in taxonomia["habilidades"]}
    for ano, hab in sorted(chaves, key=lambda k: (ORDEM_ANOS.index(k[0]) if k[0] in ORDEM_ANOS else 99, k[1])):
        g = grupos.get((ano, hab), [])
        # Reais primeiro: quando há near-dup, quem é marcado como redundante é
        # o sintético/destilado, nunca a âncora humana.
        g.sort(key=lambda r: (_PRIORIDADE_ORIGEM[r["origem"]], r["linha"], r["q_idx"]))
        mantidas, pares = [], []
        for r in g:
            par = next((m for m in mantidas if dv.e_near_duplicata(r["_q"], m["_q"])), None)
            r["near_dup"] = par is not None
            if par is None:
                mantidas.append(r)
            elif len(pares) < 8:
                pares.append({"redundante": r["codigo_item"], "igual_a": par["codigo_item"]})
        habs.append(_resumo_habilidade(ano, hab, g, pares, banco, taxonomia, val_codigos, meta_min, min_sub))

    return regs, habs, json_invalido


def _resumo_habilidade(ano, hab, g, pares, banco, taxonomia, val_codigos, meta_min, min_sub):
    ent = dv.obter_habilidade(ano, hab, taxonomia) or {}
    subtemas = [s["id"] for s in ent.get("subtemas", [])]
    exemplos_ids = {r["linha"] for r in g}
    n_q = len(g)

    # Questão EFETIVA = não-inconsistente e não-near-dup. Só ela conta para a
    # meta: questão com gabarito errado ensina errado e near-dup não ensina nada novo.
    efetivas = [r for r in g if r["gabarito"] != "inconsistente" and not r["near_dup"]]
    por_dif_ef = Counter(r["dificuldade"] for r in efetivas)
    por_sub_ef = Counter(r["subtema"] for r in efetivas)

    meta = calcular_meta(len(subtemas), meta_min, min_sub)
    deficit = calcular_deficit(meta, len(efetivas), por_dif_ef, por_sub_ef, subtemas, min_sub)

    por_ex_org = Counter()
    por_ex_dif = Counter()
    vistos = set()
    for r in g:
        if r["linha"] not in vistos:
            vistos.add(r["linha"])
            por_ex_org[r["origem"]] += 1
            por_ex_dif[r["dificuldade"]] += 1

    sub_todas = Counter(r["subtema"] for r in g)
    com_meta = [r for r in g if r["subtema_meta"]]
    concord = sum(1 for r in com_meta if r["subtema_meta"] == r["subtema"])
    gab = Counter(r["gabarito"] for r in g)
    geo = Counter({True: "aprovada", False: "reprovada"}.get(r["geometria"], "sem_veredito") for r in g)

    itens_db = banco["itens"].get((ano, hab), [])
    usados = {r["codigo_item"] for r in g if r["origem"] == "real_banco"}
    textuais = [it for it in itens_db if it["textual"]]
    livres = [it for it in textuais if it["codigo_item"] not in usados and it["codigo_item"] not in val_codigos]

    return {
        "ano": ano, "habilidade": hab,
        "matriz": "SAEB" if hab.startswith("H") else "BNCC",
        "descricao": banco["descricoes"].get((ano, hab)) or (ent.get("descricoes") or [""])[0],
        "n_exemplos": len(exemplos_ids), "n_questoes": n_q, "n_efetivas": len(efetivas),
        "origem": dict(por_ex_org),
        "dificuldade": {d: por_ex_dif.get(d, 0) for d in DIFICULDADES},
        "dificuldade_efetiva": {d: por_dif_ef.get(d, 0) for d in DIFICULDADES},
        "difficulty_divergente": sum(1 for r in g if r["difficulty_divergente"]),
        "subtemas": {
            "taxonomia": subtemas, "k": len(subtemas),
            "contagem": {s: sub_todas.get(s, 0) for s in subtemas},
            "outros": sub_todas.get("outros", 0),
            "cobertos": sum(1 for s in subtemas if sub_todas.get(s, 0) > 0),
            "vazios": [s for s in subtemas if sub_todas.get(s, 0) == 0],
            "concordancia_meta_classificador": (round(concord / len(com_meta), 3) if com_meta else None),
            "n_com_subtema_meta": len(com_meta),
        },
        "near_dup": {"n": sum(1 for r in g if r["near_dup"]),
                     "taxa": round(sum(1 for r in g if r["near_dup"]) / n_q, 3) if n_q else 0.0,
                     "exemplos": pares},
        "gabarito": {"ok": gab.get("ok", 0), "inconsistente": gab.get("inconsistente", 0),
                     "nao_verificavel": gab.get("nao_verificavel", 0),
                     "motivos": dict(Counter(r["motivo_gabarito"] for r in g)),
                     "inconsistentes": [{"codigo_item": r["codigo_item"], "q_idx": r["q_idx"],
                                         "motivo": r["motivo_gabarito"], "letra": r["letra"],
                                         "sugestao": r["sugestao_gabarito"]}
                                        for r in g if r["gabarito"] == "inconsistente"]},
        "geometria": dict(geo),
        "banco": {"total": len(itens_db), "textuais": len(textuais), "usados_no_treino": len(usados),
                  "graus": sorted({it["grau"] for it in itens_db if it["grau"] in DIFICULDADES},
                                  key=DIFICULDADES.index),
                  "em_validacao": sum(1 for it in itens_db if it["codigo_item"] in val_codigos),
                  "textuais_livres": len(livres),
                  "codigos_livres": [it["codigo_item"] for it in livres]},
        "meta": meta, **deficit,
    }


# ---------------------------------------------------------------------------
# Alertas e agregados
# ---------------------------------------------------------------------------

def gerar_alertas(habs):
    al = defaultdict(list)
    descr_por_ano = defaultdict(list)
    for h in habs:
        rot = f"{h['ano']} {h['habilidade']}"
        n = h["n_questoes"]
        dif = h["dificuldade"]
        tot = sum(dif.values())
        if tot >= ALERTA_DIF_N:
            fora = {d: round(v / tot, 2) for d, v in dif.items()
                    if not (ALERTA_DIF_MIN <= v / tot <= ALERTA_DIF_MAX)}
            if fora:
                al["desequilibrio_dificuldade"].append({"habilidade": rot, "dificuldade": dif, "fora_da_faixa": fora})
        elif tot and any(v == 0 for v in dif.values()):
            al["desequilibrio_dificuldade"].append({"habilidade": rot, "dificuldade": dif,
                                                    "fora_da_faixa": {"n_pequeno": tot}})
        if h["subtemas"]["k"] >= 2 and h["subtemas"]["vazios"]:
            al["subtemas_vazios"].append({"habilidade": rot, "vazios": h["subtemas"]["vazios"],
                                          "k": h["subtemas"]["k"]})
        if n and h["near_dup"]["taxa"] >= ALERTA_DUP:
            al["duplicatas_altas"].append({"habilidade": rot, "taxa": h["near_dup"]["taxa"],
                                           "n": h["near_dup"]["n"]})
        if h["gabarito"]["inconsistente"]:
            al["gabarito_inconsistente"].append({"habilidade": rot, "n": h["gabarito"]["inconsistente"],
                                                 "itens": h["gabarito"]["inconsistentes"]})
        if n and h["subtemas"]["outros"] / n >= ALERTA_OUTROS:
            al["classificador_outros"].append({"habilidade": rot, "outros": h["subtemas"]["outros"], "n": n})
        if n >= ALERTA_CEGO_N and h["gabarito"]["nao_verificavel"] / n >= ALERTA_CEGO:
            al["gabarito_cego"].append({"habilidade": rot, "nao_verificavel": h["gabarito"]["nao_verificavel"],
                                        "n": n})
        if h["difficulty_divergente"]:
            al["difficulty_divergente_do_meta"].append({"habilidade": rot, "n": h["difficulty_divergente"]})
        if h["origem"].get("real_banco", 0) == 0 and h["banco"]["textuais_livres"] == 0:
            al["sem_ancora_real"].append({"habilidade": rot, "itens_banco": h["banco"]["total"],
                                          "textuais": h["banco"]["textuais"]})
        if h["banco"]["graus"] and set(DIFICULDADES) - set(h["banco"]["graus"]):
            # A meta pede as 3 dificuldades porque a interface aceita qualquer uma
            # para qualquer habilidade; mas se o banco real nunca usou "Difícil"
            # nesse ano, a injeção dessa faixa é decisão pedagógica a confirmar.
            al["dificuldade_ausente_no_banco"].append({
                "habilidade": rot, "graus_banco": h["banco"]["graus"],
                "falta_meta": {d: h["falta_por_dificuldade"][d]
                               for d in DIFICULDADES if d not in h["banco"]["graus"]}})
        if h["origem"].get("real_fora_do_banco"):
            al["origem_desconhecida"].append({"habilidade": rot, "n": h["origem"]["real_fora_do_banco"]})
        if h["descricao"]:
            descr_por_ano[(h["ano"], h["descricao"])].append(h["habilidade"])
    for (ano, d), hs in descr_por_ano.items():
        if len(hs) > 1:
            al["descricao_repetida"].append({"ano": ano, "habilidades": hs, "descricao": d})
    return dict(al)


def _piores(habs, n=15):
    """Piores = maior fração da meta que falta; desempate por menos efetivas."""
    return sorted(habs, key=lambda h: (-h["deficit"] / h["meta"], h["n_efetivas"]))[:n]


def agregar(regs, habs, json_invalido, banco, meta_min, min_sub):
    gab = Counter(r["gabarito"] for r in regs)
    letras = Counter(r["letra"] for r in regs)
    org = Counter()
    dif = Counter()
    for h in habs:
        org.update(h["origem"])
        dif.update(h["dificuldade"])
    k_total = sum(h["subtemas"]["k"] for h in habs)
    return {
        "n_exemplos": sum(h["n_exemplos"] for h in habs) + len(json_invalido),
        "n_questoes": len(regs), "n_efetivas": sum(h["n_efetivas"] for h in habs),
        "n_habilidades": len(habs), "json_invalido": json_invalido,
        "origem": dict(org), "dificuldade": {d: dif.get(d, 0) for d in DIFICULDADES},
        "gabarito": {k: gab.get(k, 0) for k in ("ok", "inconsistente", "nao_verificavel")},
        "letra_correta": {L: letras.get(L, 0) for L in "ABCDE"},
        "near_dup": sum(h["near_dup"]["n"] for h in habs),
        "subtemas_k": k_total, "subtemas_cobertos": sum(h["subtemas"]["cobertos"] for h in habs),
        "banco": {"total_itens": banco["total"], "sem_habilidade": banco["sem_habilidade"],
                  "textuais_livres": sum(h["banco"]["textuais_livres"] for h in habs)},
        "meta_total": sum(h["meta"] for h in habs),
        "deficit_total": sum(h["deficit"] for h in habs),
        "deficit_por_dificuldade": {d: sum(h["falta_por_dificuldade"][d] for h in habs) for d in DIFICULDADES},
        "habilidades_abaixo_da_meta": sum(1 for h in habs if h["deficit"] > 0),
        "habilidades_acima_2x_meta": [f"{h['ano']} {h['habilidade']}" for h in habs if h["n_efetivas"] >= 2 * h["meta"]],
        "geometria_verificador": "disponivel" if vg is not None else "indisponivel",
        "criterio_meta": {
            "meta_min_habilidade": meta_min, "min_por_subtema": min_sub,
            "formula": "meta = ceil(max(meta_min, min_por_subtema*K)/3)*3; alvo por dificuldade = meta/3; "
                       "deficit = max(meta - efetivas, soma faltas por dificuldade, soma faltas por subtema)",
            "efetiva": "questão com gabarito não-inconsistente e que não é near-dup de outra da mesma habilidade",
        },
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "—"


def _dif_str(d):
    return "/".join(str(d.get(x, 0)) for x in DIFICULDADES)


def _org_str(o):
    return "/".join(str(o.get(x, 0)) for x in ("real_banco", "sintetico", "destilado"))


def gerar_md(rel):
    g, habs, al = rel["global"], rel["habilidades"], rel["alertas"]
    L = []
    w = L.append
    w("# Relatório da Base de Conhecimento (treino)")
    w("")
    w(f"Gerado por `src/relatorio_base.py` em {rel['gerado_em']} a partir de `{rel['fontes']['train']}`, "
      f"`{rel['fontes']['db']}` e `{rel['fontes']['taxonomia']}`. Reexecute o script para atualizar: "
      "este arquivo é gerado e não deve ser editado à mão.")
    w("")
    w("## 1. Resumo executivo")
    w("")
    w(f"- **{g['n_exemplos']} exemplos** de treino, **{g['n_questoes']} questões**, "
      f"**{g['n_habilidades']} habilidades** (ano × código).")
    o = g["origem"]
    w(f"- Origem (exemplos): real do banco **{o.get('real_banco', 0)}** ({_pct(o.get('real_banco', 0), g['n_exemplos'])}), "
      f"sintético **{o.get('sintetico', 0)}**, destilado **{o.get('destilado', 0)}** "
      f"({_pct(o.get('destilado', 0), g['n_exemplos'])})"
      + (f", origem desconhecida **{o['real_fora_do_banco']}**" if o.get("real_fora_do_banco") else "") + ".")
    d = g["dificuldade"]
    w(f"- Dificuldade (exemplos): Fácil {d['Fácil']} · Moderado {d['Moderado']} · Difícil {d['Difícil']}.")
    gb = g["gabarito"]
    w(f"- Gabarito (`check_consistency`): ok **{gb['ok']}** ({_pct(gb['ok'], g['n_questoes'])}), "
      f"inconsistente **{gb['inconsistente']}**, não verificável **{gb['nao_verificavel']}** "
      f"({_pct(gb['nao_verificavel'], g['n_questoes'])}).")
    w(f"- Near-duplicatas dentro da habilidade: **{g['near_dup']}** questões "
      f"({_pct(g['near_dup'], g['n_questoes'])}).")
    w(f"- Subtemas da taxonomia com ≥1 exemplo: **{g['subtemas_cobertos']}/{g['subtemas_k']}**.")
    w(f"- Questões **efetivas** (sem gabarito inconsistente e sem near-dup): **{g['n_efetivas']}**.")
    w(f"- **Meta total: {g['meta_total']} questões efetivas. Déficit total a injetar: {g['deficit_total']}** "
      f"(Fácil {g['deficit_por_dificuldade']['Fácil']} · Moderado {g['deficit_por_dificuldade']['Moderado']} · "
      f"Difícil {g['deficit_por_dificuldade']['Difícil']} só para equilibrar dificuldade). "
      f"{g['habilidades_abaixo_da_meta']}/{g['n_habilidades']} habilidades estão abaixo da meta.")
    w(f"- Itens reais **textuais livres** no banco (sem imagem, fora do treino e da validação): "
      f"**{g['banco']['textuais_livres']}** — âncoras possíveis para a injeção.")
    lt = g["letra_correta"]
    w(f"- Letra da resposta correta: " + " · ".join(f"{k} {v} ({_pct(v, g['n_questoes'])})" for k, v in lt.items())
      + ". (Equilíbrio seria 20% cada.)")
    w(f"- Verificador de geometria (`src/verificador_geometria.py`): {g['geometria_verificador']}.")
    w("")
    w("### Distribuição por ano")
    w("")
    w("| Ano | Matriz | Hab. | Exemplos | Questões | Efetivas | Real/Sint/Dest | F/M/D | Subtemas cobertos | Near-dup | Gab. incons. | Livres no banco | Meta | Déficit |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ano, a in rel["por_ano"].items():
        w(f"| {ano} | {'+'.join(a['matriz'])} | {a['habilidades']} | {a['n_exemplos']} | {a['n_questoes']} | "
          f"{a['n_efetivas']} | {_org_str(a['origem'])} | {_dif_str(a['dificuldade'])} | "
          f"{a['subtemas_cobertos']}/{a['subtemas_k']} | {a['near_dup']} | {a['gabarito_inconsistente']} | "
          f"{a['banco_textuais_livres']} | {a['meta']} | **{a['deficit']}** |")
    w("")
    w("### As 15 piores habilidades (maior fração da meta faltando)")
    w("")
    w("| # | Ano | Hab. | Descrição | Efetivas | Meta | Déficit | F/M/D | Subtemas | Livres no banco |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for i, h in enumerate(_piores(habs), 1):
        w(f"| {i} | {h['ano']} | {h['habilidade']} | {_curta(h['descricao'])} | {h['n_efetivas']} | {h['meta']} | "
          f"**{h['deficit']}** | {_dif_str(h['dificuldade_efetiva'])} | "
          f"{h['subtemas']['cobertos']}/{h['subtemas']['k']} | {h['banco']['textuais_livres']} |")
    w("")
    w("## 2. Critério da meta (como o déficit é calculado)")
    w("")
    c = g["criterio_meta"]
    w(f"1. **Mínimo de {c['meta_min_habilidade']} questões efetivas por habilidade.** O prompt de treino é "
      "condicionado em (ano, habilidade, dificuldade); 30 dá 10 por dificuldade, abaixo disso o modelo tende a "
      "decorar exemplos em vez de aprender o padrão. O banco SAEB real tem só 8 itens por descritor, então 30 "
      "mantém a âncora real relevante (~25%) sem ser engolida pelos sintéticos.")
    w(f"2. **Pelo menos {c['min_por_subtema']} por subtema da taxonomia.** Com 1 exemplo o modelo copia o molde; "
      "3 é o mínimo para variar contexto e estrutura dentro do subtema. Habilidade com K subtemas tem meta "
      f"≥ {c['min_por_subtema']}·K.")
    w("3. **Equilíbrio entre as 3 dificuldades:** a meta é arredondada para múltiplo de 3 e o alvo por dificuldade "
      "é meta/3. Desequilíbrio ensina o modelo a ignorar a dificuldade pedida no prompt.")
    w(f"4. Só contam questões **efetivas**: {c['efetiva']}. Gabarito errado ensina errado; near-dup não ensina nada novo.")
    w("5. **Déficit = max(meta − efetivas, Σ faltas por dificuldade, Σ faltas por subtema).** Cada questão nova "
      "tem uma dificuldade e um subtema escolhidos livremente; por isso o mínimo que cumpre os três critérios "
      "ao mesmo tempo é o maior dos três, não a soma.")
    w("")
    w("Limites conhecidos: o subtema vem do classificador léxico de `diversidade.py` (pode errar em enunciados "
      "atípicos — veja a coluna 'outros'); `check_consistency` só verifica contas explícitas e **não detecta erro "
      "conceitual** (ex.: o erro central do caso 9º H17 auditado, \"tem uma propriedade de X ⇒ é X\"). "
      "\"ok\" aqui é necessário, não suficiente.")
    w("")
    w("## 3. Alertas")
    w("")
    titulos = {
        "gabarito_inconsistente": "Gabarito inconsistente (a conta da resolução aponta outra alternativa ou nenhuma)",
        "desequilibrio_dificuldade": f"Desequilíbrio de dificuldade (alguma fora de {int(ALERTA_DIF_MIN*100)}–{int(ALERTA_DIF_MAX*100)}%, n≥{ALERTA_DIF_N}; ou dificuldade ausente)",
        "subtemas_vazios": "Subtemas da taxonomia sem nenhum exemplo (K≥2)",
        "duplicatas_altas": f"Near-duplicatas altas (≥{int(ALERTA_DUP*100)}% das questões)",
        "gabarito_cego": f"Gabarito sem verificação automática (≥{int(ALERTA_CEGO*100)}% não verificável): exige validação conceitual",
        "classificador_outros": f"Classificador não reconhece o subtema (≥{int(ALERTA_OUTROS*100)}% em 'outros')",
        "difficulty_divergente_do_meta": "Campo `difficulty` do JSON diverge da dificuldade do meta",
        "sem_ancora_real": "Sem âncora real (nenhum item real no treino e nenhum textual livre no banco)",
        "descricao_repetida": "Descrição idêntica para códigos diferentes no mesmo ano",
        "dificuldade_ausente_no_banco": "Dificuldade exigida pela meta que o banco real nunca usa nessa habilidade (confirmar com pedagogia)",
        "letra_correta_desbalanceada": f"Letra da resposta correta desbalanceada (fora de {int(ALERTA_LETRA_MIN*100)}–{int(ALERTA_LETRA_MAX*100)}% no global)",
        "origem_desconhecida": "Origem desconhecida (codigo_item fora do banco, sem flag de sintético/destilado)",
    }
    for chave, titulo in titulos.items():
        itens = al.get(chave, [])
        w(f"### {titulo} — {len(itens)}")
        w("")
        if not itens:
            w("Nenhum.")
            w("")
            continue
        for it in itens:
            w("- " + _fmt_alerta(chave, it))
        w("")
    w("## 4. Detalhe por habilidade")
    w("")
    w("Colunas: Ex = exemplos; Q = questões; Ef = efetivas; R/S/D = real do banco/sintético/destilado; "
      "F/M/D = dificuldade (exemplos); Sub = subtemas cobertos/K; Dup = taxa de near-dup; "
      "Gab ok/inc/nv = gabarito ok/inconsistente/não verificável; Banco = itens totais/textuais/livres.")
    for ano in rel["por_ano"]:
        w("")
        w(f"### {ano} ano")
        w("")
        w("| Hab. | Descrição | Ex | Q | Ef | R/S/D | F/M/D | Sub | Vazios | Dup | Gab ok/inc/nv | Banco | Meta | Déficit |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for h in (x for x in habs if x["ano"] == ano):
            s, gb, b = h["subtemas"], h["gabarito"], h["banco"]
            w(f"| {h['habilidade']} | {_curta(h['descricao'], 70)} | {h['n_exemplos']} | {h['n_questoes']} | "
              f"{h['n_efetivas']} | {_org_str(h['origem'])} | {_dif_str(h['dificuldade'])} | "
              f"{s['cobertos']}/{s['k']} | {', '.join(s['vazios']) or '—'} | {h['near_dup']['taxa']:.0%} | "
              f"{gb['ok']}/{gb['inconsistente']}/{gb['nao_verificavel']} | "
              f"{b['total']}/{b['textuais']}/{b['textuais_livres']} | {h['meta']} | **{h['deficit']}** |")
    w("")
    w("Detalhes completos (contagem por subtema, falta por dificuldade/subtema, pares de near-dup, códigos "
      "livres no banco, itens inconsistentes) estão em `outputs/relatorio_base.json`.")
    w("")
    return "\n".join(L)


def _curta(t, n=60):
    t = " ".join(str(t or "").split()).replace("|", "/")
    return t if len(t) <= n else t[: n - 1] + "…"


def _fmt_alerta(chave, it):
    if chave == "gabarito_inconsistente":
        itens = ", ".join(f"`{x['codigo_item']}` ({x['motivo']}" + (f", sugere {x['sugestao']}" if x["sugestao"] else "") + ")"
                          for x in it["itens"])
        return f"**{it['habilidade']}**: {it['n']} — {itens}"
    if chave == "desequilibrio_dificuldade":
        return f"**{it['habilidade']}**: F/M/D = {_dif_str(it['dificuldade'])} (fora: {it['fora_da_faixa']})"
    if chave == "subtemas_vazios":
        return f"**{it['habilidade']}**: {len(it['vazios'])}/{it['k']} vazios — {', '.join(it['vazios'])}"
    if chave == "duplicatas_altas":
        return f"**{it['habilidade']}**: {it['taxa']:.0%} ({it['n']} questões)"
    if chave == "gabarito_cego":
        return f"**{it['habilidade']}**: {it['nao_verificavel']}/{it['n']} não verificáveis"
    if chave == "classificador_outros":
        return f"**{it['habilidade']}**: {it['outros']}/{it['n']} em 'outros'"
    if chave == "difficulty_divergente_do_meta":
        return f"**{it['habilidade']}**: {it['n']} questões"
    if chave == "sem_ancora_real":
        return f"**{it['habilidade']}**: banco tem {it['itens_banco']} itens, {it['textuais']} textuais"
    if chave == "dificuldade_ausente_no_banco":
        return (f"**{it['habilidade']}**: banco só tem {', '.join(it['graus_banco'])}; "
                f"a meta pede {it['falta_meta']} a mais nas faixas ausentes")
    if chave == "letra_correta_desbalanceada":
        return (f"Global {it['global']}; fora da faixa: {it['fora_da_faixa']}. Na injeção, sortear a posição "
                "da correta de modo uniforme e não usar 'Nenhuma das anteriores' como preenchimento fixo do E.")
    if chave == "descricao_repetida":
        return f"**{it['ano']}** {', '.join(it['habilidades'])}: \"{_curta(it['descricao'], 90)}\""
    return f"**{it.get('habilidade')}**: {it}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def gerar_relatorio(train=TRAIN_PATH, db=DB_PATH, taxonomia_path=None, val_paths=VAL_PATHS,
                    meta_min=META_MIN_HABILIDADE, min_sub=MIN_POR_SUBTEMA):
    exemplos = carregar_jsonl(train)
    banco = carregar_banco(db)
    taxonomia = dv.carregar_taxonomia(taxonomia_path)
    val_codigos = set()
    for p in val_paths:
        if Path(p).exists():
            val_codigos |= {(e.get("meta") or {}).get("codigo_item") for e in carregar_jsonl(p)}
    regs, habs, json_invalido = analisar(exemplos, banco, taxonomia, val_codigos, meta_min, min_sub)
    alertas = gerar_alertas(habs)
    g = agregar(regs, habs, json_invalido, banco, meta_min, min_sub)
    nq = g["n_questoes"] or 1
    fora = {L: round(v / nq, 3) for L, v in g["letra_correta"].items()
            if not (ALERTA_LETRA_MIN <= v / nq <= ALERTA_LETRA_MAX)}
    if fora:
        alertas["letra_correta_desbalanceada"] = [{"global": g["letra_correta"], "fora_da_faixa": fora}]
    por_ano = _por_ano(habs)

    def _rel(p):
        try:
            return str(Path(p).resolve().relative_to(ROOT))
        except ValueError:
            return str(p)

    return {
        "gerado_em": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "fontes": {"train": _rel(train), "db": _rel(db),
                   "taxonomia": _rel(taxonomia_path or dv.TAXONOMIA_PATH),
                   "validacao": [_rel(p) for p in val_paths if Path(p).exists()]},
        "global": g, "por_ano": por_ano,
        "piores_15": [f"{h['ano']} {h['habilidade']}" for h in _piores(habs)],
        "alertas": alertas, "habilidades": habs,
    }


def _por_ano(habs):
    out = {}
    for ano in ORDEM_ANOS:
        hs = [h for h in habs if h["ano"] == ano]
        if not hs:
            continue
        org, dif = Counter(), Counter()
        for h in hs:
            org.update(h["origem"])
            dif.update(h["dificuldade"])
        out[ano] = {
            "habilidades": len(hs), "matriz": sorted({h["matriz"] for h in hs}),
            "n_exemplos": sum(h["n_exemplos"] for h in hs), "n_questoes": sum(h["n_questoes"] for h in hs),
            "n_efetivas": sum(h["n_efetivas"] for h in hs),
            "origem": dict(org), "dificuldade": {d: dif.get(d, 0) for d in DIFICULDADES},
            "subtemas_k": sum(h["subtemas"]["k"] for h in hs),
            "subtemas_cobertos": sum(h["subtemas"]["cobertos"] for h in hs),
            "near_dup": sum(h["near_dup"]["n"] for h in hs),
            "gabarito_inconsistente": sum(h["gabarito"]["inconsistente"] for h in hs),
            "banco_textuais_livres": sum(h["banco"]["textuais_livres"] for h in hs),
            "meta": sum(h["meta"] for h in hs), "deficit": sum(h["deficit"] for h in hs),
            "habilidades_abaixo_da_meta": sum(1 for h in hs if h["deficit"] > 0),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--train", default=str(TRAIN_PATH))
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--taxonomia", default=None, help="padrão: data/taxonomia_subtemas.json")
    ap.add_argument("--saida-json", default=str(SAIDA_JSON))
    ap.add_argument("--saida-md", default=str(SAIDA_MD))
    ap.add_argument("--meta-min", type=int, default=META_MIN_HABILIDADE)
    ap.add_argument("--min-por-subtema", type=int, default=MIN_POR_SUBTEMA)
    a = ap.parse_args(argv)

    rel = gerar_relatorio(a.train, a.db, a.taxonomia, VAL_PATHS, a.meta_min, a.min_por_subtema)
    Path(a.saida_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida_md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida_json).write_text(json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    md = gerar_md(rel)
    # Seções escritas à mão por outras etapas (ex.: "Calibração dos agentes",
    # gerada a partir de outputs/calibracao_agentes.json) ficam entre
    # marcadores e sobrevivem à regeneração: sem isso, rodar o relatório de
    # novo apagaria o registro da calibração.
    anterior = Path(a.saida_md).read_text(encoding="utf-8") if Path(a.saida_md).exists() else ""
    blocos = re.findall(r"<!-- secao-preservada:inicio -->.*?<!-- secao-preservada:fim -->", anterior, flags=re.S)
    if blocos:
        md = md.rstrip("\n") + "\n\n" + "\n\n".join(blocos) + "\n"
    Path(a.saida_md).write_text(md, encoding="utf-8")

    g = rel["global"]
    print(f"exemplos={g['n_exemplos']} questoes={g['n_questoes']} efetivas={g['n_efetivas']} "
          f"habilidades={g['n_habilidades']}")
    print(f"origem={g['origem']} dificuldade={g['dificuldade']}")
    print(f"gabarito={g['gabarito']} near_dup={g['near_dup']} "
          f"subtemas={g['subtemas_cobertos']}/{g['subtemas_k']}")
    print(f"meta_total={g['meta_total']} deficit_total={g['deficit_total']} "
          f"abaixo_da_meta={g['habilidades_abaixo_da_meta']}/{g['n_habilidades']}")
    for ano, x in rel["por_ano"].items():
        print(f"  {ano}: hab={x['habilidades']} ex={x['n_exemplos']} ef={x['n_efetivas']} "
              f"meta={x['meta']} deficit={x['deficit']}")
    print("alertas: " + ", ".join(f"{k}={len(v)}" for k, v in rel["alertas"].items()))
    print(f"-> {a.saida_json}\n-> {a.saida_md}")


if __name__ == "__main__":
    main()
