"""Opção B, etapa 2: filtrar as amostras e montar o corpus RFT e os pares de preferência.

Filtro em duas camadas, ambas já validadas contra o professor:
  1. determinístico: questão parseada, status do verificador ok ou nao_verificavel, 5 alternativas
     distintas, sem duplicata de enunciado;
  2. juiz Jev (online, ~US$ 0,00005/item): aceita se `resposta_unica` >= 0,9. Limiar congelado antes de ver a
     folha dos 222 itens e confirmado fora da amostra (kappa 0,80; ciclo_v8/SPEC.md).

Padrão: DRY-RUN, sem rede (só a camada 1 e a estimativa de custo). `--real` chama o Jev e exige
`--max-requisicoes`. Respostas do Jev ficam em cache (retomável). Nunca escreve em data/train_curado*.jsonl
nem val*.

Saídas (quando há aceitos):
  data/rft_v8_extras.jsonl         só os exemplos RFT, na ORDEM v8 (resolve antes das alternativas)
  data/train_curado_v8_rft.jsonl   data/train_curado_v8.jsonl + extras, teto de 30% do corpus base,
                                   rodízio priorizando as unidades fracas
  data/rft_v8_pares.jsonl          {prompt, chosen, rejected} para ORPO/DPO: mesmo prompt, aceito x rejeitado

    venv/bin/python src/rft_filtrar.py                       # dry-run
    venv/bin/python src/rft_filtrar.py --real --max-requisicoes 3000
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import construir_v8 as c8  # noqa: E402
import unidades_tematicas as ut  # noqa: E402

LIMIAR = 0.9
TETO_FRACAO = 0.30
STATUS_OK = ("ok", "nao_verificavel")
FRACAS = ["Geometria", "Álgebra", "Probabilidade e estatística", "Números", "Grandezas e medidas"]  # ordem de prioridade
CACHE = ROOT / "outputs" / "melhoria_v8" / "rft_jev_cache.jsonl"
SEED_PERMUTACAO = 20261022


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    return re.sub(r"[^a-z0-9]+", " ", "".join(ch for ch in t if not unicodedata.combining(ch))).strip()


def camada1(amostras):
    """(aceitas, rejeitadas_com_motivo). Dedupe por enunciado normalizado."""
    vistos, ok, rej = set(), [], []
    for s in amostras:
        q = s.get("obj")
        # generate_validated devolve a questão direta; aceita também o envelope {"questoes": [q]}
        if isinstance(q, dict) and "questoes" in q:
            qs = q["questoes"]
            q = qs[0] if isinstance(qs, list) and len(qs) == 1 else None
        if not isinstance(q, dict) or "enunciado" not in q:
            rej.append((s, "sem_questao")); continue
        if s.get("status") not in STATUS_OK:
            rej.append((s, f"status_{s.get('status')}")); continue
        alts = [norm(str(q.get("alternativas", {}).get(l, ""))) for l in "ABCDE"]
        if len(set(alts)) < 5 or not all(alts):
            rej.append((s, "alternativas_repetidas")); continue
        chave = norm(q.get("enunciado", ""))
        if chave in vistos:
            rej.append((s, "duplicata")); continue
        vistos.add(chave)
        ok.append({**s, "q": q})
    return ok, rej


def camada1_sem_dedupe(amostras):
    """Como camada1, sem remover duplicatas (avaliação mede cada amostra, não a diversidade)."""
    ok, rej = [], []
    for i, s in enumerate(amostras):
        o, r = camada1([dict(s, obj=s.get("obj"))])
        if o:
            ok.append({**o[0]})
        else:
            rej.extend(r)
    return ok, rej


def item_jev(s):
    q = s["q"]
    return {"enunciado": q["enunciado"], "alternativas": q["alternativas"],
            "resolucao": q["resolucao_passo_a_passo"], "gabarito_modelo": q["resposta_correta"]}


def julgar(itens, real, teto):
    import f_jev as fj
    cache = {}
    if CACHE.exists():
        for l in CACHE.read_text(encoding="utf-8").splitlines():
            r = json.loads(l); cache[r["uid"]] = r["p"]
    pend = [s for s in itens if f"{s['pid']}:{s['k']}" not in cache]
    print(f"Jev: {len(itens)} itens, {len(pend)} sem cache; custo estimado ~US$ {len(pend) * 0.00006:.2f}")
    if real and pend:
        chave, cont = fj.ler_chave(), fj.Contador(ROOT / "outputs" / "melhoria_v8" / "rft_jev_contador.json",
                                                  teto=teto, teto_usd=1.0)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "a", encoding="utf-8") as f:
            for s in pend:
                r = fj.chamar(fj.corpo_requisicao(item_jev(s), "completo"), chave, cont)
                if (r or {}).get("erro") in ("HTTP 402", "HTTP 401", "HTTP 403"):
                    raise SystemExit(f"API recusou ({r['erro']}): créditos esgotados ou chave inválida no OpenRouter. "
                                     "Recarregue/corrija e rode de novo; o cache retoma.")
                a = (r or {}).get("answers")
                if not a:
                    continue  # falha de API = abstenção, nunca aprovação
                uid = f"{s['pid']}:{s['k']}"
                cache[uid] = a["resposta_unica"]["noul"]
                f.write(json.dumps({"uid": uid, "p": cache[uid]}) + "\n"); f.flush()
    return cache


def montar(aceitas, rejeitadas_jev, base, extras_p, treino_p, pares_p):
    from extract_data import SYSTEM_PROMPT
    base_ex = [json.loads(l) for l in Path(base).read_text(encoding="utf-8").splitlines() if l.strip()]
    teto = int(TETO_FRACAO * len(base_ex))
    por_un = defaultdict(list)
    for s in aceitas:
        por_un[ut.unidade_de(s["ano"], s["habilidade"])].append(s)
    escolhidas, i = [], 0
    while len(escolhidas) < teto and any(por_un.values()):
        for u in FRACAS:
            if por_un[u] and len(escolhidas) < teto:
                escolhidas.append(por_un[u].pop(0))
        i += 1
    # Balanceia a letra do gabarito ANTES de serializar (2026-10-22). Sem isto a 1ª rodada herdou o viés do
    # gerador (A 34,5%, E 7,7% nos extras) e o modelo treinado saiu com 44% de gabaritos em A (teto G6: 40%).
    # Mesmo mecanismo de consolidar_base: um único lote para todos os extras, seed fixa, determinístico.
    from gerar_lote import permutar_lote
    permutadas = permutar_lote([s["q"] for s in escolhidas], seed=SEED_PERMUTACAO)
    extras = []
    for s, qp in zip(escolhidas, permutadas):
        nova, info = c8.transformar_questao(qp)
        if nova is None:
            continue
        s = dict(s, q=qp)
        extras.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": s["prompt"]},
                                    {"role": "assistant", "content": c8.serializar([nova])}],
                       "meta": {"codigo_item": f"RFT-{s['pid']}-{s['k']}", "ano": s["ano"], "habilidade": s["habilidade"],
                                "dificuldade": s["dificuldade"], "origem": "rft_v8", "status_verificador": s["status"]}})
    pares = []
    rej = {r["pid"]: r for r in rejeitadas_jev}
    for s in escolhidas:
        r = rej.get(s["pid"])
        if r:
            pares.append({"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": s["prompt"]}],
                          "chosen": c8.serializar([c8.ordenar_v8(s["q"])]),
                          "rejected": c8.serializar([c8.ordenar_v8(r["q"])])})
    for p, linhas in ((extras_p, extras), (treino_p, base_ex + extras)):
        c8.escrever(p, linhas)
    Path(pares_p).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pares), encoding="utf-8")
    return len(extras), len(pares), teto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amostras", default=str(ROOT / "data" / "rft_amostras.jsonl"))
    ap.add_argument("--base", default=str(ROOT / "data" / "train_curado_v8.jsonl"))
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--max-requisicoes", type=int, default=0)
    ap.add_argument("--sufixo-saida", default="", help="ex.: _bal -> data/train_curado_v8_rft_bal.jsonl (não sobrescreve o corpus já treinado)")
    a = ap.parse_args()
    for n in (a.amostras,):
        if not Path(n).exists():
            raise SystemExit(f"{n} não existe: rode src/rft_gerar.py antes")
    am = [json.loads(l) for l in Path(a.amostras).read_text(encoding="utf-8").splitlines() if l.strip()]
    ok1, rej1 = camada1(am)
    print(f"amostras {len(am)} | camada 1 aceita {len(ok1)} | rejeitadas {len(rej1)}")
    if a.real and a.max_requisicoes <= 0:
        raise SystemExit("--real exige --max-requisicoes")
    cache = julgar(ok1, a.real, a.max_requisicoes)
    julgadas = [s for s in ok1 if f"{s['pid']}:{s['k']}" in cache]
    aceitas = [s for s in julgadas if cache[f"{s['pid']}:{s['k']}"] >= LIMIAR]
    rejeitadas = [s for s in julgadas if cache[f"{s['pid']}:{s['k']}"] < 0.5]
    print(f"Jev: julgadas {len(julgadas)} | aceitas {len(aceitas)} | rejeitadas firmes (<0,5) {len(rejeitadas)}")
    if not aceitas:
        print("nada a montar (dry-run ou sem cache)."); return
    sx = a.sufixo_saida
    ne, npar, teto = montar(aceitas, rejeitadas, a.base, ROOT / "data" / f"rft_v8_extras{sx}.jsonl",
                            ROOT / "data" / f"train_curado_v8_rft{sx}.jsonl", ROOT / "data" / f"rft_v8_pares{sx}.jsonl")
    print(f"extras {ne} (teto {teto}) | pares {npar}")


if __name__ == "__main__":
    main()
