"""Avaliação barata de um braço (modelo) com o juiz Jev, pareada por prompt e seed.

Entrada: JSONL de src/rft_gerar.py gerado com os MESMOS --prompts/--amostras/--seed em cada braço
(prompt pid e seed k iguais => comparação pareada). Use uma seed DIFERENTE da usada para treinar o RFT
(avaliação em prompts novos).

Métrica por amostra: entregue (status != falha e questão parseada) E Jev `resposta_unica` >= 0,9
("aceita"). Taxa de aceitação sobre TODAS as amostras (falha conta como não aceita). Proxy validado contra
o professor (kappa 0,80); não substitui a folha cega, que decide a promoção.

    venv/bin/python src/avaliar_jev.py --braco v7b=data/aval_v7b.jsonl --braco v8rft=data/aval_v8rft.jsonl \
        --max-requisicoes 2500            # gasta centavos; sem --real só conta e estima
Cache por braço em outputs/melhoria_v8/avaliar_jev_<braco>.jsonl (retomável).
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import rft_filtrar as rf  # noqa: E402

LIM = rf.LIMIAR


def ler(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def julgar_braco(nome, amostras, real, teto):
    import f_jev as fj
    cache_p = ROOT / "outputs" / "melhoria_v8" / f"avaliar_jev_{nome}.jsonl"
    cache = {}
    if cache_p.exists():
        cache = {r["uid"]: r["p"] for r in map(json.loads, cache_p.read_text().splitlines()) if r}
    ok1, _ = rf.camada1_sem_dedupe(amostras)
    pend = [s for s in ok1 if f"{s['pid']}:{s['k']}" not in cache]
    print(f"[{nome}] amostras {len(amostras)} | entregues {len(ok1)} | sem cache {len(pend)} | ~US$ {len(pend)*0.00012:.2f}")
    if real and pend:
        chave = fj.ler_chave()
        cont = fj.Contador(ROOT / "outputs" / "melhoria_v8" / f"avaliar_jev_{nome}_contador.json", teto=teto, teto_usd=1.0)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_p, "a", encoding="utf-8") as f:
            for s in pend:
                r = fj.chamar(fj.corpo_requisicao(rf.item_jev(s), "completo"), chave, cont)
                if (r or {}).get("erro") in ("HTTP 402", "HTTP 401", "HTTP 403"):
                    raise SystemExit(f"API recusou ({r['erro']}): créditos esgotados ou chave inválida no OpenRouter. "
                                     "Recarregue/corrija e rode de novo; o cache retoma.")
                a = (r or {}).get("answers")
                if not a:
                    continue
                uid = f"{s['pid']}:{s['k']}"
                cache[uid] = a["resposta_unica"]["noul"]
                f.write(json.dumps({"uid": uid, "p": cache[uid]}) + "\n"); f.flush()
    return {f"{s['pid']}:{s['k']}": cache.get(f"{s['pid']}:{s['k']}") for s in ok1}, ok1


def wilson(s, n, z=1.96):
    p = s / n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return round(100*(c-h), 1), round(100*(c+h), 1)


def mcnemar_exato(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2*sum(math.comb(n, i) for i in range(k+1)) / 2**n)


def resumo(amostras, scores):
    uids = {f"{s['pid']}:{s['k']}" for s in amostras}
    acc = {u for u, p in scores.items() if p is not None and p >= LIM}
    julg = sum(p is not None for p in scores.values())
    por_ano = defaultdict(lambda: [0, 0])
    for s in amostras:
        u = f"{s['pid']}:{s['k']}"; por_ano[s["ano"]][1] += 1; por_ano[s["ano"]][0] += u in acc
    return acc, uids, julg, por_ano


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--braco", action="append", required=True, help="nome=arquivo.jsonl (o primeiro é a referência)")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--max-requisicoes", type=int, default=0)
    a = ap.parse_args()
    if a.real and a.max_requisicoes <= 0:
        raise SystemExit("--real exige --max-requisicoes")
    res = {}
    for par in a.braco:
        nome, arq = par.split("=", 1)
        am = ler(arq)
        sc, _ = julgar_braco(nome, am, a.real, a.max_requisicoes)
        res[nome] = (am, sc, *resumo(am, sc))
    nomes = list(res)
    print("\nbraço          n   entregues  julgadas  aceitas (taxa sobre n)  IC95        por ano")
    for n in nomes:
        am, sc, acc, uids, julg, pa = res[n]
        print(f"{n:12s} {len(am):5d} {len(sc):9d} {julg:9d} {len(acc):7d} ({100*len(acc)/len(am):4.1f}%)  "
              f"{wilson(len(acc), len(am))}  " + " ".join(f"{k}:{100*v[0]/v[1]:.0f}%" for k, v in sorted(pa.items())))
    ref = nomes[0]
    for n in nomes[1:]:
        ua, ub = res[ref][2], res[n][2]
        comuns = {f"{s['pid']}:{s['k']}" for s in res[ref][0]} & {f"{s['pid']}:{s['k']}" for s in res[n][0]}
        b = len((ua - ub) & comuns); c = len((ub - ua) & comuns)
        print(f"\n{n} x {ref} (pareado, {len(comuns)} pares): só {ref} aceita {b} | só {n} aceita {c} | McNemar exato p={mcnemar_exato(b, c):.3f}")
    print("\nAbstenção (API) não é aprovação: itens sem julgamento contam como não aceitos.")


if __name__ == "__main__":
    main()
