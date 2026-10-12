"""Triagem determinística das resoluções passo a passo (ciclo v6, etapa 4). Sem API, não altera a base.

O comprimento é só um sinal entre vários. Classes (a primeira que se aplica):

  matematicamente_problematica — o verificador aritmético reprova (conta errada ou resultado em
                                 outra alternativa) ou o resultado final não está em nenhuma
                                 alternativa. Vai para REVISÃO CEGA, nunca para reescrita.
  nao_avaliavel                — vazia, ou sem conta verificável E sem citar o conteúdo do gabarito
                                 E curta: nada aqui permite dizer se está certa.
  incompleta                   — não chega à resposta: não cita o valor/conteúdo do gabarito.
  correta_pouco_explicativa    — chega à resposta (conta confere ou cita o gabarito), mas explica
                                 pouco: < 80 caracteres, ou é só "Correto - a op b = c", ou tem
                                 menos de 6 palavras fora das contas.
  adequada                     — o resto.

Sinais extras (não mudam a classe): cita letra de alternativa, CAIXA ALTA, `dado_inventado`.

Filas geradas (para a etapa paga, que só roda com autorização):
  fila_reescrita  = correta_pouco_explicativa + incompleta (o enunciado, as alternativas e o
                    gabarito ficam congelados; só a explicação é pedida de novo);
  fila_revisao    = matematicamente_problematica + nao_avaliavel (juiz cego decide; se ele
                    discordar do gabarito, vai para revisão humana — nunca troca automática).

    venv/bin/python src/triar_resolucoes.py --base data/train_curado_v5.jsonl --rotulo v5
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import schema_utils as su  # noqa: E402
from padronizar_enunciado import eh_caixa_alta  # noqa: E402
from perfil_base import SAIDA_DIR, carregar, hash_questao, origem_de  # noqa: E402

# Mesmo padrão de consolidar_base.REF_LETRA (sem importar o módulo pago).
REF_LETRA = re.compile(r"\b(alternativa|letra|op[cç][aã]o|item)\s+\(?[A-E]\)?(?![\wà-ÿ])|^\s*\(?[A-E]\)\s", re.I | re.M)
CLASSES = ("adequada", "correta_pouco_explicativa", "incompleta", "matematicamente_problematica", "nao_avaliavel")
MIN_EXPLICATIVA = 80
MIN_PALAVRAS = 6


def palavras_fora_das_contas(texto):
    """Palavras (>= 2 letras) depois de tirar números, operadores e o prefixo 'Correto -'."""
    t = re.sub(r"(?i)^\s*corret[oa]\s*[-–—:]?\s*", "", texto or "")
    return [w for w in re.findall(r"[A-Za-zÀ-ÿ]{2,}", t)]


def cita_o_gabarito(q):
    """A resolução cita o valor (alternativa numérica) ou o conteúdo (textual) do gabarito?"""
    res = q.get("resolucao_passo_a_passo", "") or ""
    alt = q.get("alternativas", {}).get(q.get("resposta_correta"), "")
    valores = su._valores_alternativa(alt)
    if valores:
        return su._valor_no_texto(res, valores)
    return any(su._corresponde(alt, f) for f in su._frases(res)) if res.strip() else False


def triar(q):
    """(classe, sinais:dict). Determinístico."""
    res = (q.get("resolucao_passo_a_passo", "") or "").strip()
    ok, _s, motivo = su.check_consistency_detalhado(q)
    fora = su.resposta_fora_das_alternativas(q)
    cita = cita_o_gabarito(q) if res else False
    pal = palavras_fora_das_contas(res)
    sinais = {
        "verificador": {True: "ok", False: "inconsistente", None: "nao_verificavel"}[ok], "motivo": motivo,
        "fora_das_alternativas": bool(fora), "cita_gabarito": cita, "chars": len(res), "palavras": len(pal),
        "estilo_correto": bool(re.match(r"(?i)corret[oa]\b", res)),
        "cita_letra": bool(REF_LETRA.search(res)), "caixa_alta": bool(res) and eh_caixa_alta(res),
        "dado_inventado": len(su.avisos_consistencia(q)),
    }
    if ok is False or fora:
        return "matematicamente_problematica", sinais
    if not res or (ok is None and not cita and len(res) < 40):
        return "nao_avaliavel", sinais
    if ok is not True and not cita:
        return "incompleta", sinais
    if len(res) < MIN_EXPLICATIVA or len(pal) < MIN_PALAVRAS or (sinais["estilo_correto"] and len(res) < 120):
        return "correta_pouco_explicativa", sinais
    return "adequada", sinais


def triar_base(base):
    out = []
    for i, j, m, q, _p in carregar(base):
        classe, sinais = triar(q)
        out.append({"linha": i, "questao": j, "codigo_item": m["codigo_item"], "hash": hash_questao(q),
                    "ano": m["ano"], "habilidade": m["habilidade"], "origem": origem_de(m),
                    "classe": classe, **sinais})
    return out


def resumo(itens):
    por_origem = defaultdict(Counter)
    for x in itens:
        por_origem[x["origem"]][x["classe"]] += 1
    lt40 = [x for x in itens if x["chars"] < 40]
    return {
        "n": len(itens), "classes": {c: sum(x["classe"] == c for x in itens) for c in CLASSES},
        "por_origem": {k: dict(v) for k, v in sorted(por_origem.items())},
        "menos_de_40_chars": {"n": len(lt40), "classes": dict(Counter(x["classe"] for x in lt40))},
        "sinais": {"cita_letra": sum(x["cita_letra"] for x in itens), "caixa_alta": sum(x["caixa_alta"] for x in itens),
                   "com_dado_inventado": sum(x["dado_inventado"] > 0 for x in itens)},
        "fila_reescrita": sum(x["classe"] in ("correta_pouco_explicativa", "incompleta") for x in itens),
        "fila_revisao": sum(x["classe"] in ("matematicamente_problematica", "nao_avaliavel") for x in itens),
    }


def resumo_md(r, base):
    L = [f"# Triagem das resoluções — {Path(base).name}", "",
         "Gerado por `src/triar_resolucoes.py`, sem API. Classes determinísticas (pré-classificação): "
         "a fila de reescrita ainda passa pelo professor e pelos verificadores.", "",
         "| classe | questões |", "|---|---|"] + [f"| {c} | {n} |" for c, n in r["classes"].items()]
    L += ["", "## Por origem", "", "| origem | " + " | ".join(CLASSES) + " |", "|---" * (len(CLASSES) + 1) + "|"]
    L += [f"| {o} | " + " | ".join(str(v.get(c, 0)) for c in CLASSES) + " |" for o, v in r["por_origem"].items()]
    L += ["", f"## As {r['menos_de_40_chars']['n']} resoluções com menos de 40 caracteres", "",
          f"Classes: {r['menos_de_40_chars']['classes']}. Comprimento curto não é defeito por si: "
          "uma conta curta que confere e chega ao gabarito é 'correta, porém pouco explicativa'.", "",
          f"Sinais: {r['sinais']}.", "",
          f"- Fila de reescrita (só explicação; enunciado, alternativas e gabarito congelados): **{r['fila_reescrita']}**",
          f"- Fila de revisão cega (problemática ou não avaliável): **{r['fila_revisao']}**"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(ROOT / "data" / "train_curado_v5.jsonl"))
    ap.add_argument("--rotulo", default="v5")
    ap.add_argument("--saida-dir", default=str(SAIDA_DIR))
    args = ap.parse_args()
    itens = triar_base(args.base)
    r = resumo(itens)
    out = Path(args.saida_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"resolucoes_{args.rotulo}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / f"resolucoes_{args.rotulo}.md").write_text(resumo_md(r, args.base), encoding="utf-8")
    with open(out / f"resolucoes_{args.rotulo}_itens.jsonl", "w", encoding="utf-8") as f:
        for x in itens:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
