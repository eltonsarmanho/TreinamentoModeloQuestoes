"""Profiling reprodutível de uma base de treino (ciclo v6, etapa 2). Sem API, sem escrever na base.

Complementa src/relatorio_base.py (que já dá cobertura/meta/déficit por habilidade e
subtema) com as dimensões que ele não mede: comprimento e estilo das resoluções,
alternativa "Nenhuma das alternativas anteriores" (NDA), letra do gabarito, rótulo de
dificuldade x dificuldade julgada, variação só numérica, near-duplicatas auditáveis
(com os trigramas coincidentes), diversidade por dimensão (subtema, raciocínio,
estrutura, representação, contexto, valores numéricos) para a base e por (ano,
habilidade), dependência visual, verificabilidade e classe da auditoria de gabarito.

Metadado ausente é contado como "desconhecido" e a taxa de classificação é informada;
nada é inventado. O subtema/tipo/contexto vêm de duas fontes que são reportadas em
separado: o declarado em meta (o pedido no prompt de geração) e o classificador
léxico de diversidade.py (o que o texto parece ser).

    venv/bin/python src/perfil_base.py --base data/train_curado_v5.jsonl --rotulo v5
    -> outputs/melhoria_v6/perfil_<rotulo>.json, .md, near_dups_<rotulo>.jsonl
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import diversidade as dv  # noqa: E402
import schema_utils as su  # noqa: E402
from auditar_gabaritos_base import origem as origem_codigo  # noqa: E402
from unidades_tematicas import unidade_de  # noqa: E402

SAIDA_DIR = ROOT / "outputs" / "melhoria_v6"
DESCONHECIDO = "desconhecido"
FAIXAS_RESOLUCAO = ((0, 1, "vazia"), (1, 40, "<40"), (40, 100, "40-99"), (100, 200, "100-199"), (200, 10 ** 9, ">=200"))
TERMO_VISUAL = re.compile(  # mesmo padrão de consolidar_base.TERMO_VISUAL (sem importar o módulo pago)
    r"\b(figura|imagem|desenho|ilustra\w*|esquema|mapa|malha|croqui|planta|gr[áa]fico|tabela|quadro|"
    r"reta num[ée]rica|abaixo|acima|ao lado|a seguir|veja|observe)\b", re.I)


def sha256_arquivo(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_questao(q):
    """Identidade estável de uma questão (enunciado + alternativas em ordem de letra)."""
    bruto = json.dumps([q.get("enunciado", ""), [q.get("alternativas", {}).get(L, "") for L in "ABCDE"]],
                       ensure_ascii=False)
    return hashlib.sha1(bruto.encode("utf-8")).hexdigest()[:16]


def eh_nda(texto):
    return su.classe_alternativa(texto or "") == "nda"


def estilo_resolucao(texto):
    t = (texto or "").strip()
    if not t:
        return "vazia"
    if re.match(r"(?i)corret[oa]\b", t):
        return "correto_curto" if len(t) < 60 else "correto_explicado"
    return "texto"


def carregar(base):
    """Uma entrada por QUESTÃO: (linha, indice, meta, questao, prompt_usuario)."""
    out = []
    for i, l in enumerate(open(base, encoding="utf-8")):
        if not l.strip():
            continue
        r = json.loads(l)
        for j, q in enumerate(json.loads(r["messages"][2]["content"])["questoes"]):
            out.append((i, j, r["meta"], q, r["messages"][1]["content"]))
    return out


def origem_de(meta):
    """Origem do exemplo. 'destilado' em meta sobrepõe o prefixo do código (injeção e destilação)."""
    o = origem_codigo(meta["codigo_item"])
    if o.startswith("real"):
        return o
    if meta.get("origem") == "injecao_saeb":
        return "injetado"
    if meta.get("sintetico"):
        return "sintetico"
    if meta.get("destilado"):
        return "destilado"
    return o


def entropia_normalizada(valores, universo):
    """Entropia de Shannon / log(min(n, universo)). 1 = uniforme sobre TODAS as categorias que a
    dimensão admite (não só as presentes): 30 questões em 2 de 5 subtemas não dão 1,0.
    Universo 1 (habilidade com um único subtema) não tem o que diversificar: devolve None."""
    n = len(valores)
    teto = min(n, universo)
    if teto <= 1:
        return None
    c = Counter(valores)
    h = -sum((v / n) * math.log(v / n) for v in c.values())
    return h / math.log(teto)


def _r(x):
    return None if x is None else round(abs(x), 3)


def assinatura_numerica(q):
    """Conjunto ordenado dos números do enunciado (diversidade de valores)."""
    return tuple(sorted(str(x) for x in su._numeros_do_enunciado_exatos(q.get("enunciado", ""))))


def molde(q):
    """Enunciado sem números (números mascarados): igual = variação SÓ numérica."""
    return re.sub(r"[^a-z#]", "", dv.mascarar_numeros(q.get("enunciado", "")))


def near_dups_auditaveis(itens, limiar=dv.LIMIAR_NEAR_DUP):
    """Pares near-dup dentro do mesmo (ano, habilidade), com evidência: Jaccard de trigramas,
    trigramas coincidentes e se a diferença é só numérica. Nada é removido aqui."""
    grupos = defaultdict(list)
    for x in itens:
        grupos[(x["ano"], x["habilidade"])].append(x)
    pares = []
    for (ano, hab), g in grupos.items():
        sh = [dv.shingles(x["q"].get("enunciado", "")) for x in g]
        md = [molde(x["q"]) for x in g]
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                so_num = bool(md[a]) and md[a] == md[b]
                jac = dv.jaccard(sh[a], sh[b])
                if not so_num and jac < limiar:
                    continue
                comuns = sorted(sh[a] & sh[b], key=len, reverse=True)[:6]
                mesma_resp = dv.normalizar_texto(g[a]["q"]["alternativas"].get(g[a]["q"]["resposta_correta"], "")) == \
                    dv.normalizar_texto(g[b]["q"]["alternativas"].get(g[b]["q"]["resposta_correta"], ""))
                pares.append({
                    "ano": ano, "habilidade": hab,
                    "a": {"codigo_item": g[a]["codigo_item"], "origem": g[a]["origem"], "hash": g[a]["hash"]},
                    "b": {"codigo_item": g[b]["codigo_item"], "origem": g[b]["origem"], "hash": g[b]["hash"]},
                    "jaccard_trigramas": round(jac, 3), "so_numeros_diferem": so_num,
                    "mesma_resposta": mesma_resp, "trigramas_comuns": comuns,
                    "classe": ("variacao_so_numerica" if so_num and not mesma_resp else
                               "duplicata_provavel" if so_num or jac >= 0.85 else "semelhante_revisar"),
                    "envolve_real": g[a]["origem"].startswith("real") or g[b]["origem"].startswith("real"),
                })
    return pares


def _diversidade_grupo(g, k_subtemas, k_tipos, subtema_composto=False):
    """k_subtemas/k_tipos: tamanho do universo (da habilidade, ou da taxonomia inteira para a base).
    subtema_composto: na base inteira o subtema é (ano, habilidade, subtema) — 'geral' de duas
    habilidades não é o mesmo subtema."""
    cls = [x["cls"] for x in g]
    subt = [f"{x['ano']}|{x['habilidade']}|{x['cls']['subtema']}" if subtema_composto else x["cls"]["subtema"] for x in g]
    nums = [x["assinatura_numerica"] for x in g]
    return {
        "n": len(g),
        "subtemas_distintos": len({s for s, c in zip(subt, cls) if c["subtema"] != "outros"}),
        # +1: 'outros'/'indefinido' (não classificado) também é uma categoria observável
        "entropia_subtema": None if k_subtemas <= 1 else _r(entropia_normalizada(subt, k_subtemas + 1)),
        "entropia_raciocinio": _r(entropia_normalizada([c["tipo_raciocinio"] for c in cls], k_tipos + 1)),
        "entropia_estrutura": _r(entropia_normalizada([c["estrutura"] for c in cls], len(dv.ESTRUTURAS))),
        "entropia_representacao": _r(entropia_normalizada([c["representacao"] for c in cls], 4)),
        "entropia_contexto": _r(entropia_normalizada([c["contexto"] for c in cls], len(dv.CONTEXTOS) + 1)),
        "valores_numericos_distintos": round(len(set(nums)) / len(nums), 3) if nums else 0.0,
        "moldes_distintos": round(len({x["molde"] for x in g}) / len(g), 3),
        "pct_outros": round(100 * sum(c["subtema"] == "outros" for c in cls) / len(cls), 1),
    }


def perfilar(base, auditoria=None, taxonomia=None):
    """Retorna (perfil:dict, itens:list, pares_near_dup:list). Determinístico."""
    tax = taxonomia or dv.carregar_taxonomia()
    classe_aud = {}
    if auditoria is not None:
        classe_aud = {(x["linha"], x["questao"]): x["classe"] for x in auditoria}
    itens = []
    for i, j, m, q, prompt in carregar(base):
        ok, _s, motivo = su.check_consistency_detalhado(q)
        res = q.get("resolucao_passo_a_passo", "") or ""
        alts = q.get("alternativas", {})
        itens.append({
            "linha": i, "questao": j, "codigo_item": m["codigo_item"], "ano": m["ano"],
            "habilidade": m["habilidade"], "unidade": unidade_de(m["ano"], m["habilidade"]),
            "origem": origem_de(m), "dificuldade": m.get("dificuldade") or DESCONHECIDO,
            "dificuldade_real": m.get("dificuldade_real") or DESCONHECIDO,
            "meta_subtema": m.get("subtema") or DESCONHECIDO,
            "meta_tipo": m.get("tipo_raciocinio") or DESCONHECIDO,
            "meta_contexto": m.get("contexto") or DESCONHECIDO,
            "tem_sufixo": " Subtema:" in prompt,
            "cls": dv.classificar_questao(q, m["ano"], m["habilidade"], tax),
            "q": q, "hash": hash_questao(q), "molde": molde(q), "assinatura_numerica": assinatura_numerica(q),
            "len_resolucao": len(res.strip()), "estilo_resolucao": estilo_resolucao(res),
            "nda_letra": next((L for L, v in alts.items() if eh_nda(v)), None),
            "gabarito": q.get("resposta_correta"),
            "verificador": {True: "ok", False: "inconsistente", None: "nao_verificavel"}[ok],
            "motivo_verificador": motivo,
            "visual_ausente": bool(su.depende_de_visual_ausente(q.get("enunciado", ""))),
            "termo_visual_sem_dados": bool(TERMO_VISUAL.search(q.get("enunciado", "")) and not dv.tem_dados_textuais(q)),
            "dados_ausentes": bool(dv.dados_ausentes(q)),
            "classe_auditoria": classe_aud.get((i, j), DESCONHECIDO),
        })
    n = len(itens)
    pares = near_dups_auditaveis(itens)
    exatos = Counter(x["hash"] for x in itens)
    enunc = Counter(dv.normalizar_texto(x["q"].get("enunciado", "")) for x in itens)
    moldes = Counter(x["molde"] for x in itens if x["molde"])
    lens = [x["len_resolucao"] for x in itens]

    def dist(chave, fonte=itens):
        return dict(Counter(x[chave] if not callable(chave) else chave(x) for x in fonte).most_common())

    def taxa_conhecida(chave):
        return round(100 * sum(x[chave] != DESCONHECIDO for x in itens) / n, 1)

    por_hab = defaultdict(list)
    for x in itens:
        por_hab[(x["ano"], x["habilidade"])].append(x)
    habilidades = {}
    for (ano, hab), g in sorted(por_hab.items()):
        h = dv.obter_habilidade(ano, hab, tax)
        subt_tax = [s["id"] for s in h["subtemas"]] if h else []
        presentes = Counter(x["cls"]["subtema"] for x in g)
        habilidades[f"{ano}|{hab}"] = {
            "ano": ano, "habilidade": hab, "unidade": g[0]["unidade"], "n": len(g),
            "origem": dict(Counter(x["origem"] for x in g)),
            "dificuldade": dict(Counter(x["dificuldade"] for x in g)),
            "subtemas_taxonomia": len(subt_tax),
            "subtemas_vazios": [s for s in subt_tax if presentes.get(s, 0) == 0],
            "subtema_dominante_pct": round(100 * max(presentes.values()) / len(g), 1),
            "nda_pct": round(100 * sum(x["nda_letra"] is not None for x in g) / len(g), 1),
            "resolucao_lt40_pct": round(100 * sum(x["len_resolucao"] < 40 for x in g) / len(g), 1),
            "verificavel_pct": round(100 * sum(x["verificador"] == "ok" for x in g) / len(g), 1),
            "near_dup_pares": sum(1 for p in pares if (p["ano"], p["habilidade"]) == (ano, hab)),
            "diversidade": _diversidade_grupo(g, len(subt_tax) or 1,
                                              len(h["tipos_raciocinio"]) if h else len(dv.TIPOS_RACIOCINIO)),
        }

    def cruz(a, b):
        t = defaultdict(Counter)
        for x in itens:
            t[x[a]][x[b]] += 1
        return {k: dict(v) for k, v in sorted(t.items(), key=lambda kv: str(kv[0]))}

    perfil = {
        "base": str(Path(base).relative_to(ROOT)) if Path(base).is_absolute() and ROOT in Path(base).parents else str(base),
        "sha256": sha256_arquivo(base),
        "regua_diversidade_sha256": dv.versao_regua(tax),
        "exemplos": len({x["linha"] for x in itens}), "questoes": n,
        "habilidades": len(por_hab),
        "por_ano": dist("ano"), "por_unidade": dist("unidade"), "por_origem": dist("origem"),
        "dificuldade_rotulo": dist("dificuldade"),
        "dificuldade_rotulo_x_julgada": cruz("dificuldade", "dificuldade_real"),
        "metadados_taxa_classificacao_pct": {
            "subtema_meta": taxa_conhecida("meta_subtema"), "tipo_meta": taxa_conhecida("meta_tipo"),
            "contexto_meta": taxa_conhecida("meta_contexto"), "dificuldade_julgada": taxa_conhecida("dificuldade_real"),
            "subtema_classificador": round(100 * sum(x["cls"]["subtema"] != "outros" for x in itens) / n, 1),
            "tipo_classificador": round(100 * sum(x["cls"]["tipo_raciocinio"] != "indefinido" for x in itens) / n, 1),
        },
        "concordancia_subtema_meta_x_classificador_pct": round(
            100 * sum(x["meta_subtema"] == x["cls"]["subtema"] for x in itens if x["meta_subtema"] != DESCONHECIDO)
            / max(1, sum(x["meta_subtema"] != DESCONHECIDO for x in itens)), 1),
        "prompt_com_sufixo": sum(x["tem_sufixo"] for x in itens),
        "tipo_raciocinio_classificador": dist(lambda x: x["cls"]["tipo_raciocinio"]),
        "tipo_raciocinio_meta": dist("meta_tipo"),
        "estrutura": dist(lambda x: x["cls"]["estrutura"]),
        "representacao": dist(lambda x: x["cls"]["representacao"]),
        "contexto_classificador": dist(lambda x: x["cls"]["contexto"]),
        "contexto_meta": dist("meta_contexto"),
        "resolucao": {
            "mediana_chars": statistics.median(lens) if lens else 0,
            "faixas": {rot: sum(lo <= v < hi for v in lens) for lo, hi, rot in FAIXAS_RESOLUCAO},
            "estilo": dist("estilo_resolucao"),
            "lt40_por_origem": dict(Counter(x["origem"] for x in itens if x["len_resolucao"] < 40)),
        },
        "nda": {
            "questoes_com_nda": sum(x["nda_letra"] is not None for x in itens),
            "posicao": dict(Counter(x["nda_letra"] for x in itens if x["nda_letra"])),
            "nda_e_o_gabarito": sum(x["nda_letra"] is not None and x["nda_letra"] == x["gabarito"] for x in itens),
            "por_origem": dict(Counter(x["origem"] for x in itens if x["nda_letra"])),
            "por_ano": dict(Counter(x["ano"] for x in itens if x["nda_letra"])),
        },
        "letra_gabarito": dist("gabarito"),
        "letra_gabarito_entre_itens_com_nda": dict(Counter(x["gabarito"] for x in itens if x["nda_letra"])),
        "duplicatas": {
            "questao_exata_extras": sum(v - 1 for v in exatos.values() if v > 1),
            "enunciado_exato_extras": sum(v - 1 for v in enunc.values() if v > 1),
            "moldes_repetidos_extras": sum(v - 1 for v in moldes.values() if v > 1),
            "near_dup_pares": len(pares),
            "near_dup_por_classe": dict(Counter(p["classe"] for p in pares)),
            "near_dup_envolvendo_real": sum(p["envolve_real"] for p in pares),
            "limiar_jaccard": dv.LIMIAR_NEAR_DUP,
        },
        "diversidade_base": _diversidade_grupo(itens, sum(len(h["subtemas"]) for h in tax["habilidades"].values()),
                                               len(dv.TIPOS_RACIOCINIO), subtema_composto=True),
        "visual": {
            "visual_ausente": sum(x["visual_ausente"] for x in itens),
            "termo_visual_sem_dados": sum(x["termo_visual_sem_dados"] for x in itens),
            "dados_ausentes": sum(x["dados_ausentes"] for x in itens),
        },
        "verificador": dist("verificador"),
        "classe_auditoria": dist("classe_auditoria"),
        "habilidades_detalhe": habilidades,
    }
    return perfil, itens, pares


def _tabela(d, cab=("valor", "n")):
    linhas = [f"| {cab[0]} | {cab[1]} |", "|---|---|"]
    linhas += [f"| {k} | {v} |" for k, v in d.items()]
    return linhas


def resumo_md(p):
    n = p["questoes"]
    pct = lambda v: f"{v} ({100 * v / n:.1f}%)"  # noqa: E731
    L = [f"# Perfil da base — {p['base']}", "",
         f"sha256 `{p['sha256']}` · régua de diversidade `{p['regua_diversidade_sha256'][:12]}`. "
         "Gerado por `src/perfil_base.py`, sem API; não edite à mão.", "",
         f"- **{p['exemplos']} exemplos, {n} questões, {p['habilidades']} habilidades (ano × código).**",
         f"- Prompts com sufixo (Subtema/Tipo/Contexto): {pct(p['prompt_com_sufixo'])}.",
         f"- Taxa de metadado conhecido (%): {p['metadados_taxa_classificacao_pct']}.",
         f"- Concordância subtema declarado × classificador: {p['concordancia_subtema_meta_x_classificador_pct']}%.",
         "", "## Ano, unidade, origem, dificuldade", ""]
    for t, k in (("Ano", "por_ano"), ("Unidade temática", "por_unidade"), ("Origem", "por_origem"),
                 ("Dificuldade (rótulo do prompt)", "dificuldade_rotulo")):
        L += [f"### {t}", ""] + _tabela(p[k]) + [""]
    L += ["### Rótulo de dificuldade × dificuldade julgada pelos juízes", "",
          "| rótulo | " + " | ".join(["Fácil", "Moderado", "Difícil", DESCONHECIDO]) + " |", "|---|---|---|---|---|"]
    for k, v in p["dificuldade_rotulo_x_julgada"].items():
        L.append(f"| {k} | " + " | ".join(str(v.get(c, 0)) for c in ["Fácil", "Moderado", "Difícil", DESCONHECIDO]) + " |")
    r = p["resolucao"]
    L += ["", "## Resoluções", "", f"Mediana: {r['mediana_chars']} caracteres.", ""] + _tabela(r["faixas"], ("faixa", "n"))
    L += ["", "Estilo:", ""] + _tabela(r["estilo"]) + ["", "Abaixo de 40 caracteres, por origem:", ""] + _tabela(r["lt40_por_origem"])
    a = p["nda"]
    L += ["", "## \"Nenhuma das alternativas anteriores\"", "",
          f"- Questões com NDA: {pct(a['questoes_com_nda'])}; posição {a['posicao']}; NDA é o gabarito em {a['nda_e_o_gabarito']}.",
          f"- Por origem: {a['por_origem']}. Por ano: {a['por_ano']}.",
          f"- Letra do gabarito nas questões COM NDA: {p['letra_gabarito_entre_itens_com_nda']}.",
          "", "## Letra do gabarito", ""] + _tabela(p["letra_gabarito"])
    d = p["duplicatas"]
    L += ["", "## Duplicatas", "",
          f"- Questão idêntica (enunciado + alternativas): {d['questao_exata_extras']} extras. Enunciado idêntico: {d['enunciado_exato_extras']}.",
          f"- Enunciado igual a menos dos números (variação só numérica, base inteira): {d['moldes_repetidos_extras']} extras.",
          f"- Pares near-dup na mesma habilidade (Jaccard de trigramas ≥ {d['limiar_jaccard']} ou só números diferem): "
          f"{d['near_dup_pares']} — {d['near_dup_por_classe']}; envolvendo item real: {d['near_dup_envolvendo_real']}.",
          "  Cada par, com os trigramas coincidentes, está em `near_dups_<rotulo>.jsonl`. Nada foi removido.",
          "", "## Diversidade (base inteira)", "",
          "Entropia normalizada pelo universo da dimensão (252 subtemas (ano, hab, subtema), tipos, 6 estruturas, "
          "4 representações, contextos+sem_contexto): 1 = uniforme sobre tudo que a taxonomia admite.", ""]
    L += _tabela(p["diversidade_base"], ("métrica", "valor"))
    for t, k in (("Tipo de raciocínio (classificador)", "tipo_raciocinio_classificador"), ("Estrutura", "estrutura"),
                 ("Representação", "representacao"), ("Contexto (classificador)", "contexto_classificador")):
        L += ["", f"### {t}", ""] + _tabela(p[k])
    v = p["visual"]
    L += ["", "## Dependência visual, verificabilidade e auditoria", "",
          f"- Referência a visual ausente (`depende_de_visual_ausente`): {v['visual_ausente']}; termo visual sem dados em texto: "
          f"{v['termo_visual_sem_dados']}; tabela/gráfico citado sem dados: {v['dados_ausentes']}.",
          f"- Verificador aritmético: {p['verificador']}.",
          f"- Classe na auditoria de gabarito: {p['classe_auditoria']}.",
          "", "## Por habilidade (piores primeiro em diversidade de subtema)", "",
          "| ano|hab | n | subtemas vazios | entropia subtema | subtema dominante % | entropia raciocínio | moldes distintos | NDA % | res<40 % | verificável % | near-dup |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    hs = sorted(p["habilidades_detalhe"].items(),
                key=lambda kv: (kv[1]["diversidade"]["entropia_subtema"] is None, kv[1]["diversidade"]["entropia_subtema"] or 0, kv[1]["n"]))
    for k, h in hs:
        dvh = h["diversidade"]
        na = lambda v: "n/a" if v is None else v  # noqa: E731
        L.append(f"| {k} | {h['n']} | {len(h['subtemas_vazios'])}/{h['subtemas_taxonomia']} | {na(dvh['entropia_subtema'])} | "
                 f"{h['subtema_dominante_pct']} | "
                 f"{na(dvh['entropia_raciocinio'])} | {dvh['moldes_distintos']} | {h['nda_pct']} | {h['resolucao_lt40_pct']} | "
                 f"{h['verificavel_pct']} | {h['near_dup_pares']} |")
    L += ["", "## Limites", "",
          "- Subtema, tipo, estrutura, representação e contexto do classificador são léxicos (regex): erram em enunciados atípicos; "
          "'outros'/'indefinido' = não classificado.",
          "- O metadado declarado (meta) é o que foi PEDIDO na geração, não o que o texto contém.",
          "- Near-dup é lexical (trigramas com números mascarados): pega reescrita literal, não paráfrase com outras palavras.",
          "- Verificável = conta explícita confere; não certifica o enunciado nem a matemática inteira.",
          "- `dificuldade_real` só existe para itens injetados com juízes; o resto é desconhecido."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(ROOT / "data" / "train_curado_v5.jsonl"))
    ap.add_argument("--rotulo", default="v5")
    ap.add_argument("--auditoria", default=None,
                    help="AUDITORIA_GABARITOS_<rotulo>.jsonl; padrão: outputs/relatorios/AUDITORIA_GABARITOS_<rotulo>.jsonl se existir")
    ap.add_argument("--saida-dir", default=str(SAIDA_DIR))
    args = ap.parse_args()
    aud_path = Path(args.auditoria or ROOT / "outputs" / "relatorios" / f"AUDITORIA_GABARITOS_{args.rotulo}.jsonl")
    auditoria = [json.loads(l) for l in open(aud_path, encoding="utf-8") if l.strip()] if aud_path.exists() else None
    if auditoria is None:
        print(f"aviso: {aud_path} ausente; classe de auditoria = desconhecido")
    perfil, _itens, pares = perfilar(args.base, auditoria)
    out = Path(args.saida_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"perfil_{args.rotulo}.json").write_text(json.dumps(perfil, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / f"perfil_{args.rotulo}.md").write_text(resumo_md(perfil), encoding="utf-8")
    with open(out / f"near_dups_{args.rotulo}.jsonl", "w", encoding="utf-8") as f:
        for p in pares:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{perfil['questoes']} questões; near-dup pares={len(pares)}; -> {out}/perfil_{args.rotulo}.{{json,md}}")


if __name__ == "__main__":
    main()
