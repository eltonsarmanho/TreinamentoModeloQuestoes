"""Tarefa F: estatística sem dependências (numpy/sklearn não são necessários).

wilson_ic        IC de Wilson (aceita k fracionário, usado na tabela de tamanho de amostra)
auc              AUC por postos (Mann-Whitney, empates com posto médio)
matriz / kappa   concordância binária e Cohen kappa
melhor_limiar    limiar (score >= t => 1) que maximiza a métrica escolhida
bootstrap_oob    IC bootstrap em que o limiar é reajustado em cada reamostra e avaliado nos
                 itens fora da reamostra (out-of-bag): não premia o ajuste ao próprio dado
"""
import math
import random


def wilson_ic(k, n, z=1.959964):
    """(lo, hi) em proporção [0,1]. k pode ser fracionário. n == 0 -> (nan, nan)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    den = 1 + z2 / n
    centro = (p + z2 / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return (max(0.0, centro - meia), min(1.0, centro + meia))


def meia_largura_wilson(p, n):
    lo, hi = wilson_ic(p * n, n)
    return (hi - lo) / 2


def tabela_tamanho_amostra(ns, p=0.65):
    """[{n, ic_lo_pct, ic_hi_pct, meia_largura_pp}] — IC de Wilson 95% quando a taxa observada é p."""
    out = []
    for n in ns:
        lo, hi = wilson_ic(p * n, n)
        out.append({"n": n, "ic_lo_pct": round(100 * lo, 1), "ic_hi_pct": round(100 * hi, 1),
                    "meia_largura_pp": round(100 * (hi - lo) / 2, 1)})
    return out


def auc(scores, rotulos):
    """AUC = P(score do positivo > score do negativo) + 0,5 P(empate). None se faltar uma classe.
    `rotulos` em {0,1}/{False,True}; positivo = 1."""
    pares = sorted(zip(scores, rotulos), key=lambda x: x[0])
    n = len(pares)
    n_pos = sum(1 for _, y in pares if y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # postos médios para empates
    soma_pos = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pares[j + 1][0] == pares[i][0]:
            j += 1
        posto = (i + j) / 2 + 1  # 1-based
        soma_pos += posto * sum(1 for t in range(i, j + 1) if pares[t][1])
        i = j + 1
    return (soma_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def matriz(pred, real):
    tp = sum(1 for p, y in zip(pred, real) if p and y)
    fp = sum(1 for p, y in zip(pred, real) if p and not y)
    fn = sum(1 for p, y in zip(pred, real) if not p and y)
    tn = sum(1 for p, y in zip(pred, real) if not p and not y)
    return tp, fp, fn, tn


def kappa(pred, real):
    """Cohen kappa binário. None se indefinido (acordo esperado = 1)."""
    tp, fp, fn, tn = matriz(pred, real)
    n = tp + fp + fn + tn
    if n == 0:
        return None
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    if pe >= 1:
        return None
    return (po - pe) / (1 - pe)


def metricas_binarias(pred, real):
    tp, fp, fn, tn = matriz(pred, real)
    n = tp + fp + fn + tn

    def div(a, b):
        return a / b if b else None
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "acuracia": div(tp + tn, n), "precisao": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "especificidade": div(tn, tn + fp), "kappa": kappa(pred, real)}


def _valor_metrica(pred, real, metrica):
    if metrica == "acuracia":
        n = len(real)
        return sum(1 for p, y in zip(pred, real) if bool(p) == bool(y)) / n if n else None
    if metrica == "kappa":
        k = kappa(pred, real)
        return k
    raise ValueError(metrica)


def melhor_limiar(scores, real, metrica="kappa"):
    """Limiar t (predição = score >= t) que maximiza `metrica`. Candidatos: pontos médios entre
    scores distintos adjacentes + extremos. Desempate: o MAIOR t (mais conservador).
    Devolve (t, valor). valor None se a métrica for indefinida para todos."""
    vals = sorted(set(scores))
    if not vals:
        return None, None
    cands = [vals[0] - 1e-9] + [(a + b) / 2 for a, b in zip(vals, vals[1:])] + [vals[-1] + 1e-9]
    melhor, melhor_t = None, None
    for t in cands:
        v = _valor_metrica([s >= t for s in scores], real, metrica)
        if v is None:
            continue
        if melhor is None or v >= melhor:
            melhor, melhor_t = v, t
    return melhor_t, melhor


def percentil(valores, q):
    v = sorted(x for x in valores if x is not None)
    if not v:
        return None
    pos = q * (len(v) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def bootstrap_ic(fn, n, reps=1000, seed=0):
    """IC bootstrap percentil 95% de fn(indices) sobre reamostragens com reposição."""
    rng = random.Random(seed)
    vs = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        vs.append(fn(idx))
    return percentil(vs, 0.025), percentil(vs, 0.975)


def bootstrap_oob(scores, real, metrica="kappa", reps=1000, seed=0):
    """Reajusta o limiar em cada reamostra e mede a métrica nos itens fora dela.
    Devolve {"media", "ic95": (lo, hi), "n_validas", "limiar_mediano"}."""
    rng = random.Random(seed)
    n = len(real)
    oob_vals, limiares = [], []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        dentro = set(idx)
        fora = [i for i in range(n) if i not in dentro]
        if len(fora) < 5:
            continue
        t, _ = melhor_limiar([scores[i] for i in idx], [real[i] for i in idx], metrica)
        if t is None:
            continue
        v = _valor_metrica([scores[i] >= t for i in fora], [real[i] for i in fora], metrica)
        if v is None:
            continue
        oob_vals.append(v)
        limiares.append(t)
    if not oob_vals:
        return {"media": None, "ic95": (None, None), "n_validas": 0, "limiar_mediano": None}
    return {"media": sum(oob_vals) / len(oob_vals),
            "ic95": (percentil(oob_vals, 0.025), percentil(oob_vals, 0.975)),
            "n_validas": len(oob_vals), "limiar_mediano": percentil(limiares, 0.5)}


def bootstrap_auc(scores, real, reps=1000, seed=0):
    n = len(real)
    return bootstrap_ic(lambda idx: auc([scores[i] for i in idx], [real[i] for i in idx]),
                        n, reps=reps, seed=seed)
