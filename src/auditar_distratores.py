"""Auditoria determinística dos distratores (ciclo v6, etapa 5). Sem API, não altera a base.

Indicadores por questão (todos de triagem; nenhum remove nem corrige nada):
  alternativas_repetidas  — duas alternativas com o mesmo texto normalizado;
  valor_repetido          — duas alternativas de valor com o mesmo número ("0,5" e "1/2" não
                            são unificados: só número igual); se uma delas é o gabarito, há mais
                            de uma resposta defensável -> revisão, NUNCA troca de gabarito;
  espaco_heterogeneo      — alternativas não-NDA de classes diferentes (número x texto);
  unidade_inconsistente   — o gabarito tem unidade e algum distrator numérico não tem a mesma;
  correta_mais_longa      — pista de comprimento: o gabarito é a única alternativa mais longa e
                            >= 1,5x a segunda (só alternativas textuais);
  nda / nda_artificial    — "Nenhuma das alternativas anteriores"; artificial = acrescentada pelo
                            extract_data aos itens reais de 4 alternativas (posição E, nunca o
                            gabarito); é um distrator "gratuito": reduz a questão a 4 opções;
  explicabilidade         — para questões de valor: fração de distratores numéricos que um erro
                            comum explica (operação trocada entre dois números do enunciado,
                            número copiado do enunciado, ±1, ±10, x10, ÷10, dígitos invertidos,
                            metade/dobro do gabarito). Baixa = distratores arbitrários.

O que NÃO é verificável aqui (fica para o juiz pago ou para o professor): se cada distrator é
de fato incorreto em questões textuais, pistas gramaticais (gênero/número), e se a questão se
resolve sem dominar a habilidade.

    venv/bin/python src/auditar_distratores.py --base data/train_curado_v5.jsonl --rotulo v5
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import diversidade as dv  # noqa: E402
import schema_utils as su  # noqa: E402
from perfil_base import SAIDA_DIR, carregar, hash_questao, origem_de  # noqa: E402

NDA_ARTIFICIAL = "nenhuma das alternativas anteriores"
_PEDE_FORMA = re.compile(r"irredut|forma (de |da )?(uma )?fra[c]a|forma fracionaria|forma decimal|representacao (decimal|fracionaria)")
LIMIAR_COMPRIMENTO = 1.5


_NUM_INICIAL = re.compile(r"^(r\$\s*)?(-?\d{1,3}(?:\.\d{3})+(?:,\d+)?|-?\d+(?:[.,]\d+)?)(?:\s*/\s*(\d+))?(.*)$")


def _ler(texto):
    """(valor: Fraction, resto normalizado) da alternativa de valor, ou None.

    Leitura própria (só para esta auditoria): "5.000" é cinco mil (milhar pt-BR) — o
    schema_utils lê 5, o que juntaria "5" e "5.000" como valor repetido. O resto ("ovos",
    "galinhas", "ºa") distingue alternativas com o mesmo número e objeto diferente."""
    t = dv.normalizar_texto(texto).rstrip(".").strip()
    m = _NUM_INICIAL.match(t)
    if not m:
        return None
    num = m.group(2)
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+(?:,\d+)?", num):
        num = num.replace(".", "").replace(",", ".")
    else:
        num = num.replace(",", ".")
    v = Fraction(num)
    if m.group(3):
        if int(m.group(3)) == 0:
            return None
        v = v / int(m.group(3))
    resto = ("r$ " if m.group(1) else "") + m.group(4).strip()
    return v, resto


def _valor(texto):
    lido = _ler(texto)
    return lido[0] if lido else None


def _unidade(texto):
    """Primeira palavra/sinal depois do número ('12 cm' -> 'cm', 'R$ 5' -> 'r$', '25%' -> '%')."""
    lido = _ler(texto)
    if not lido:
        return None
    resto = lido[1]
    if resto.startswith("r$"):
        return "r$"
    m = re.match(r"(%|[a-z²³°]+)", resto)
    if not m:
        return ""
    u = m.group(1)
    return u[:-1] if len(u) > 3 and u.endswith("s") else u  # "corrida" = "corridas"; "cm"/"kg" intactos


def erros_explicativos(enunciado, correto):
    """Valores que um erro comum produziria: operação trocada entre números do enunciado,
    número copiado, e deslizes sobre o próprio resultado. Conjunto de Fractions."""
    nums = [n for n in su._numeros_do_enunciado_exatos(enunciado) if n is not None][:12]
    cand = set(nums)
    for a, b in permutations(nums, 2):
        cand |= {a + b, a - b, a * b}
        if b != 0:
            cand.add(a / b)
    if correto is not None:
        cand |= {correto + 1, correto - 1, correto + 10, correto - 10, correto * 10, correto / 10,
                 correto * 2, correto / 2}
        s = str(correto)
        if s.isdigit() and len(s) > 1:
            cand.add(Fraction(int(s[::-1])))
    return cand


def avaliar_questao(q, real=False):
    alts = q.get("alternativas", {})
    gab = q.get("resposta_correta")
    classes = {L: su.classe_alternativa(v) for L, v in alts.items()}
    nda = [L for L, c in classes.items() if c == "nda"]
    nao_nda = [L for L in alts if L not in nda]
    # sem dobrar para ASCII: normalizar_texto apaga "≤"/"≥" e juntaria "x ≤ 10" com "x ≥ 10"
    norm = Counter(re.sub(r"\s+", " ", str(v).strip().lower()).rstrip(".").strip() for v in alts.values())
    valores = {L: _valor(alts[L]) for L in nao_nda if classes[L] in ("numero", "fracao")}
    lidos = {L: _ler(alts[L]) for L in valores}
    # mesma chave = mesmo valor E mesmo resto ("8 ovos" x "8 galinhas" são alternativas distintas)
    chaves = Counter(lidos[L] for L in lidos if lidos[L] is not None)
    classes_presentes = {("valor" if classes[L] in ("numero", "fracao") else "texto") for L in nao_nda}
    ind = {
        "alternativas_repetidas": any(v > 1 for v in norm.values()),
        "valor_repetido": any(v > 1 for v in chaves.values()),
        "gabarito_com_valor_repetido": gab in lidos and lidos[gab] is not None and chaves[lidos[gab]] > 1,
        # 1/2 e 2/4 (e 35/70): equivalentes. Só é defeito se a questão não pede a forma irredutível.
        "fracoes_equivalentes_ao_gabarito": classes.get(gab) == "fracao" and valores.get(gab) is not None and sum(
            1 for L in valores if L != gab and classes[L] == "fracao" and valores[L] == valores[gab]) > 0,
        "espaco_heterogeneo": len(classes_presentes) > 1,
        "nda": bool(nda), "nda_letra": nda[0] if nda else None,
        "nda_e_gabarito": gab in nda,
        "nda_artificial": bool(nda) and real and dv.normalizar_texto(alts.get(nda[0], "")).rstrip(".") == NDA_ARTIFICIAL
        and nda[0] == "E" and gab != "E",
        "tipo_espaco": "valor" if classes_presentes == {"valor"} else "texto" if classes_presentes == {"texto"} else "misto",
    }
    # unidade
    ind["unidade_inconsistente"] = False
    if gab in valores:
        ug = _unidade(alts[gab])
        if ug:
            ind["unidade_inconsistente"] = any(_unidade(alts[L]) != ug for L in valores if L != gab)
    # pista de comprimento (só textuais)
    ind["correta_mais_longa"] = False
    if ind["tipo_espaco"] == "texto" and gab in nao_nda and len(nao_nda) >= 3:
        lens = sorted(((len(alts[L].strip()), L) for L in nao_nda), reverse=True)
        ind["correta_mais_longa"] = lens[0][1] == gab and lens[0][0] >= LIMIAR_COMPRIMENTO * max(1, lens[1][0])
    # explicabilidade dos distratores de valor
    ind["distratores_valor"] = 0
    ind["distratores_explicados"] = 0
    if ind["tipo_espaco"] == "valor" and gab in valores and valores[gab] is not None:
        cand = erros_explicativos(q.get("enunciado", ""), valores[gab])
        dis = [valores[L] for L in valores if L != gab and valores[L] is not None]
        ind["distratores_valor"] = len(dis)
        ind["distratores_explicados"] = sum(d in cand for d in dis)
    # "na forma irredutível", "em forma de fração": o mesmo valor em OUTRA forma é o distrator pedido
    ind["pede_forma_especifica"] = bool(_PEDE_FORMA.search(dv.normalizar_texto(q.get("enunciado", ""))))
    ind["revisao"] = bool(ind["alternativas_repetidas"] or (
        (ind["gabarito_com_valor_repetido"] or ind["fracoes_equivalentes_ao_gabarito"]) and not ind["pede_forma_especifica"]))
    return ind


def auditar(base):
    out = []
    for i, j, m, q, _prompt in carregar(base):
        org = origem_de(m)
        ind = avaliar_questao(q, real=org.startswith("real"))
        out.append({"linha": i, "questao": j, "codigo_item": m["codigo_item"], "hash": hash_questao(q),
                    "ano": m["ano"], "habilidade": m["habilidade"], "origem": org, **ind})
    return out


INDICADORES = ("alternativas_repetidas", "valor_repetido", "gabarito_com_valor_repetido",
               "fracoes_equivalentes_ao_gabarito", "espaco_heterogeneo",
               "unidade_inconsistente", "correta_mais_longa", "nda", "nda_artificial", "nda_e_gabarito", "revisao")


def agregar(itens, chave):
    g = defaultdict(list)
    for x in itens:
        g[x[chave] if isinstance(chave, str) else chave(x)].append(x)
    tab = {}
    for k, xs in sorted(g.items(), key=lambda kv: str(kv[0])):
        dv_ = sum(x["distratores_valor"] for x in xs)
        tab[str(k)] = {"n": len(xs), **{ind: round(100 * sum(x[ind] for x in xs) / len(xs), 1) for ind in INDICADORES},
                       "explicabilidade_pct": round(100 * sum(x["distratores_explicados"] for x in xs) / dv_, 1) if dv_ else None,
                       "espaco_valor_pct": round(100 * sum(x["tipo_espaco"] == "valor" for x in xs) / len(xs), 1)}
    return tab


def relatorio(itens):
    return {"n": len(itens), "geral": agregar(itens, lambda x: "base")["base"],
            "por_origem": agregar(itens, "origem"), "por_ano": agregar(itens, "ano"),
            "por_habilidade": agregar(itens, lambda x: f"{x['ano']}|{x['habilidade']}"),
            "revisao": [{k: x[k] for k in ("codigo_item", "hash", "ano", "habilidade", "origem")}
                        | {"motivos": [ind for ind in ("alternativas_repetidas", "gabarito_com_valor_repetido",
                                                     "fracoes_equivalentes_ao_gabarito") if x[ind]]}
                        for x in itens if x["revisao"]]}


def resumo_md(r, base):
    cols = ("n", "explicabilidade_pct", "espaco_valor_pct", "nda", "nda_artificial", "correta_mais_longa",
            "unidade_inconsistente", "espaco_heterogeneo", "valor_repetido", "revisao")
    def tabela(t):
        L = ["| grupo | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        L += [f"| {k} | " + " | ".join("n/d" if v.get(c) is None else str(v.get(c)) for c in cols) + " |" for k, v in t.items()]
        return L
    L = [f"# Distratores — {Path(base).name}", "", "Gerado por `src/auditar_distratores.py` (sem API). Percentuais por questão; "
         "`explicabilidade_pct` = % dos distratores de valor que um erro comum explica.", "",
         "## Base", ""] + tabela({"base": r["geral"]})
    L += ["", "## Por origem", ""] + tabela(r["por_origem"]) + ["", "## Por ano", ""] + tabela(r["por_ano"])
    piores = dict(sorted(((k, v) for k, v in r["por_habilidade"].items() if v["explicabilidade_pct"] is not None),
                         key=lambda kv: kv[1]["explicabilidade_pct"])[:15])
    L += ["", "## 15 habilidades com distratores de valor menos explicáveis", ""] + tabela(piores)
    L += ["", f"## Para revisão humana ({len(r['revisao'])})", "",
          "Mais de uma alternativa com o mesmo texto, o gabarito com o mesmo valor e unidade em outra alternativa, ou "
          "frações equivalentes ao gabarito. "
          "Nunca se corrige por troca de gabarito.", ""]
    L += [f"- {x['codigo_item']} ({x['ano']} {x['habilidade']}, {x['origem']}): {', '.join(x['motivos'])}" for x in r["revisao"]]
    L += ["", "## Limites", "",
          "- Explicabilidade só cobre distratores numéricos e erros de uma operação; um distrator plausível por erro "
          "conceitual (ex.: perímetro no lugar de área com fórmula) pode aparecer como não explicado.",
          "- Distrator verdadeiro em questão textual, pista gramatical e questão resolvível sem a habilidade exigem juiz.",
          "- `nda_artificial` identifica a E acrescentada aos itens reais de 4 alternativas; não diz que ela é ruim em "
          "todos os casos, só que nunca é o gabarito."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(ROOT / "data" / "train_curado_v5.jsonl"))
    ap.add_argument("--rotulo", default="v5")
    ap.add_argument("--saida-dir", default=str(SAIDA_DIR))
    args = ap.parse_args()
    itens = auditar(args.base)
    r = relatorio(itens)
    out = Path(args.saida_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"distratores_{args.rotulo}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / f"distratores_{args.rotulo}.md").write_text(resumo_md(r, args.base), encoding="utf-8")
    with open(out / f"distratores_{args.rotulo}_itens.jsonl", "w", encoding="utf-8") as f:
        for x in itens:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{len(itens)} questões; revisão={len(r['revisao'])}; geral={r['geral']}")


if __name__ == "__main__":
    main()
