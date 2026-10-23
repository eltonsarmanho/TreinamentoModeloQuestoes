"""Tarefa F: validação FORA DA AMOSTRA do juiz Jev na folha cega dos 222 itens do 4B v7b.

Regra e limiar congelados ANTES de ver estes itens (outputs/melhoria_v8/F/calibracao_jev.md):
reprovar o item se a pontuação de `resposta_unica` < 0,9. Também avalia a combinação
min(dados_suficientes, resposta_unica) no mesmo limiar. Alvo: valida_nucleo do professor.

    venv/bin/python src/f_jev_validar.py [--folha ...] [--max-requisicoes 300]

Reutiliza o cliente, o contador com teto e as perguntas de src/f_jev.py. A chave nunca é impressa.
"""
import argparse
import csv
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f_jev as fj  # noqa: E402
from f_comum import SAIDA  # noqa: E402

FOLHA = SAIDA / "folha_lote_01_preenchida.csv"
RESP = SAIDA / "jev_validacao_respostas.jsonl"
CONT = SAIDA / "jev_validacao_contador.json"
LIMIAR = 0.9
NUCLEO = ("correcao_matematica", "resposta_unica", "dados_suficientes")


def ler_folha(p):
    out = []
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            alts = {l: r[l] for l in "ABCDE"}
            out.append({"id": r["id"], "enunciado": r["enunciado"], "alternativas": alts,
                        "resolucao": r["resolucao"], "gabarito_modelo": r["gabarito_modelo"],
                        "nucleo": all(r[k].strip() == "s" for k in NUCLEO)})
    return out


def kappa(pred, y):
    n = len(y)
    po = sum(a == b for a, b in zip(pred, y)) / n
    pe = sum((sum(pred) / n) * (sum(y) / n) + (1 - sum(pred) / n) * (1 - sum(y) / n) for _ in [0])
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def auc(scores, y):
    pos = [s for s, v in zip(scores, y) if v]
    neg = [s for s, v in zip(scores, y) if not v]
    g = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return g / (len(pos) * len(neg))


def boot(fun, scores, y, reps=2000, seed=0):
    rnd, n, vals = random.Random(seed), len(y), []
    for _ in range(reps):
        idx = [rnd.randrange(n) for _ in range(n)]
        ys = [y[i] for i in idx]
        if len(set(ys)) < 2:
            continue
        v = fun([scores[i] for i in idx], ys)
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    return round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals))], 3)


def coletar(itens, teto):
    feitos = set()
    if RESP.exists():
        feitos = {json.loads(l)["id"] for l in RESP.read_text(encoding="utf-8").splitlines() if l.strip()}
    pend = [i for i in itens if i["id"] not in feitos]
    chave, cont = fj.ler_chave(), fj.Contador(CONT, teto=teto, teto_usd=0.15)

    def um(i):
        r = fj.chamar(fj.corpo_requisicao(i, "completo"), chave, cont)
        return {"id": i["id"], "resposta": r}

    with ThreadPoolExecutor(4) as ex, RESP.open("a", encoding="utf-8") as f:
        for reg in ex.map(um, pend):
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
            f.flush()
    return len(pend)


def analisar(itens):
    resp = {}
    for l in RESP.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        a = (r["resposta"] or {}).get("answers")
        if a:
            resp[r["id"]] = {k: v["noul"] for k, v in a.items()}
    base = [i for i in itens if i["id"] in resp]
    y = [i["nucleo"] for i in base]
    res = {"n_itens": len(itens), "n_com_resposta": len(base), "n_validos_nucleo": sum(y)}
    por = {}
    for nome in fj.PERGUNTAS:
        sc = [fj.pontuacao_de_validade(resp[i["id"]], nome) for i in base]
        por[nome] = {"auc": round(auc(sc, y), 3), "auc_ic95": boot(auc, sc, y)}
    res["auc_por_pergunta"] = por
    regras = {"resposta_unica": [fj.pontuacao_de_validade(resp[i["id"]], "resposta_unica") for i in base],
              "min(dados_suficientes,resposta_unica)": [fj.combinar_min(resp[i["id"]], ["dados_suficientes", "resposta_unica"]) for i in base]}
    saida = {}
    for nome, sc in regras.items():
        pred = [s >= LIMIAR for s in sc]
        rej = [not p for p in pred]
        inval = [not v for v in y]
        tp = sum(r and iv for r, iv in zip(rej, inval))
        saida[nome] = {
            "limiar_congelado": LIMIAR, "n_reprovados": sum(rej),
            "precisao_invalido": round(tp / sum(rej), 3) if sum(rej) else None,
            "recall_invalido": round(tp / sum(inval), 3),
            "validos_perdidos": sum(r and v for r, v in zip(rej, y)),
            "validos_perdidos_pct": round(100 * sum(r and v for r, v in zip(rej, y)) / sum(y), 1),
            "kappa": round(kappa(pred, y), 3),
            "kappa_ic95": boot(lambda s, yy: kappa([x >= LIMIAR for x in s], yy), sc, y),
            "acuracia": round(sum(a == b for a, b in zip(pred, y)) / len(y), 3),
            "taxa_valida_entre_aceitos": round(sum(v for p, v in zip(pred, y) if p) / sum(pred), 3),
            "auc": round(auc(sc, y), 3), "auc_ic95": boot(auc, sc, y)}
    res["regras_congeladas"] = saida
    res["taxa_valida_sem_filtro"] = round(sum(y) / len(y), 3)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folha", default=str(FOLHA))
    ap.add_argument("--max-requisicoes", type=int, default=300)
    ap.add_argument("--so-analisar", action="store_true")
    a = ap.parse_args()
    itens = ler_folha(a.folha)
    if not a.so_analisar:
        print("requisições novas:", coletar(itens, a.max_requisicoes))
    res = analisar(itens)
    (SAIDA / "validacao_jev_fora_da_amostra.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
