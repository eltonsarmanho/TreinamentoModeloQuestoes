"""Juiz Maritaca (substitui o Jev) para o ciclo v8. PAGO: dry-run por padrão; `--real` exige tetos.

Usa o VALIDADOR cego do projeto (agentes_questoes): o modelo resolve o item SEM ver gabarito nem resolução e
julga cada alternativa (V/F/I/G). Regra de aceitação (fixada antes de medir):
  aceita se a fase 1 é interpretável, a premissa é possível, não há dependência visual, não há alternativas
  equivalentes, existe EXATAMENTE uma alternativa V (outras F/G) E essa letra é a que o modelo gerador marcou.
Falha de API, truncagem, JSON ruim ou teto = abstenção (None); nunca aprovação.

  calibrar  confronta a regra com valida_nucleo do professor nos 222 itens (kappa, precisão, recall).
            FAÇA ISSO ANTES de usar o juiz em qualquer decisão.
  avaliar   taxa de aceitação por braço, pareada por prompt/seed (arquivos de src/rft_gerar.py).

    venv/bin/python src/juiz_maritaca.py calibrar                                  # estimativa, sem rede
    venv/bin/python src/juiz_maritaca.py calibrar --real --max-chamadas 300 --teto-brl 15
    venv/bin/python src/juiz_maritaca.py avaliar --braco v7b=data/aval_v7b.jsonl --braco v8rft=data/aval_v8rft.jsonl \
        --por-prompt 2 --real --max-chamadas 1200 --teto-brl 60
Cache (retomável, só respostas completas): outputs/agentes/juiz_maritaca_cache.jsonl
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import agentes_questoes as aq  # noqa: E402
import sabia_v6 as sv  # noqa: E402

MODELO = "sabia-4"
FOLHA = ROOT / "outputs" / "melhoria_v8" / "F" / "folha_lote_01_preenchida.csv"
CACHE = ROOT / "outputs" / "agentes" / "juiz_maritaca_cache.jsonl"
LOG = ROOT / "outputs" / "melhoria_v8" / "juiz_maritaca_uso.jsonl"
SAIDA = ROOT / "outputs" / "melhoria_v8"
NUCLEO = ("correcao_matematica", "resposta_unica", "dados_suficientes")
SAIDA_MEDIANA = 700  # tokens de saída esperados do validador (estimativa; a medição real fica no log)


def papeis():
    return {"validador": {"modelo": MODELO, "max_tokens": aq.max_tokens_para("validador", MODELO), "temperatura": 0.0,
                          "versao_prompt": sv.versao_prompt(aq.VALIDADOR_SISTEMA, aq.VALIDADOR_USUARIO)}}


def mensagens(item):
    a = item["alternativas"]
    return [{"role": "system", "content": aq.VALIDADOR_SISTEMA},
            {"role": "user", "content": aq.VALIDADOR_USUARIO.format(enunciado=item["enunciado"], **{L: a.get(L, "") for L in "ABCDE"})}]


def decidir(texto, gabarito):
    """True/False pela regra congelada; None se a resposta não é interpretável."""
    r = aq.interpretar_fase1(texto)
    if r is None:
        return None
    unica = aq.resposta_unica(r["status"])
    ok = (r["premissa_possivel"] and not r["dependencia_visual"] and not r["equivalentes"]
          and unica is not None and unica == gabarito and r.get("resposta_llm") in (None, unica))
    return bool(ok)


def estimar(itens, precos):
    ent = sum(sv.estimar_tokens(m["content"]) for it in itens[:1] for m in mensagens(it))
    custo = len(itens) * sv.custo_brl(precos, MODELO, ent, SAIDA_MEDIANA)
    return {"itens": len(itens), "tokens_entrada_por_item": ent, "custo_estimado_brl": round(custo, 2),
            "custo_p90_brl": round(custo * 1.4, 2)}


def julgar(itens, real, max_chamadas, teto_brl):
    """{uid: True|False|None}. itens: dicts com uid, enunciado, alternativas, gabarito."""
    precos = sv.carregar_precos()
    est = estimar(itens, precos)
    print("estimativa:", json.dumps(est, ensure_ascii=False))
    if not real:
        print("DRY-RUN: nada enviado. Para gastar: --real --max-chamadas N --teto-brl R"); return None
    if not max_chamadas or not teto_brl:
        sys.exit("ABORTADO: --real exige --max-chamadas N e --teto-brl R (tetos duros).")
    p = papeis()
    ch = sv.ChamadorSabia(p, {"validador": aq.criar_cliente_real("maritaca", MODELO)}, max_chamadas, teto_brl, precos,
                          log_uso=LOG, cache=CACHE)
    out = {}
    try:
        for it in itens:
            h = aq.hash_questao({"enunciado": it["enunciado"], "alternativas": it["alternativas"]})
            txt = ch.chamar("validador", mensagens(it), hash_exemplo=h, codigo_item=it["uid"],
                            interpretavel=lambda t: aq.interpretar_fase1(t) is not None)
            out[it["uid"]] = None if txt is None else decidir(txt, it["gabarito"])
    except sv.OrcamentoEsgotado as e:
        print("TETO ATINGIDO:", e, "- o que já foi pago está no cache; rode de novo para retomar.")
    print("uso:", json.dumps(ch.resumo(), ensure_ascii=False))
    return out


# ---------------------------------------------------------------------------- estatística
def kappa(pred, y):
    n = len(y); po = sum(a == b for a, b in zip(pred, y)) / n
    pe = (sum(pred)/n)*(sum(y)/n) + (1-sum(pred)/n)*(1-sum(y)/n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def wilson(s, n, z=1.96):
    p = s/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return round(100*(c-h), 1), round(100*(c+h), 1)


def mcnemar_exato(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2*sum(math.comb(n, i) for i in range(min(b, c)+1)) / 2**n)


# ---------------------------------------------------------------------------- comandos
def cmd_calibrar(a):
    itens, y = [], {}
    with open(a.folha, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            itens.append({"uid": r["id"], "enunciado": r["enunciado"], "alternativas": {L: r[L] for L in "ABCDE"},
                          "gabarito": r["gabarito_modelo"]})
            y[r["id"]] = all(r[k].strip() == "s" for k in NUCLEO)
    res = julgar(itens, a.real, a.max_chamadas, a.teto_brl)
    if not res:
        return
    julg = [(res[u], y[u]) for u in res if res[u] is not None]
    pred, ys = [p for p, _ in julg], [v for _, v in julg]
    rej = [not p for p in pred]; inval = [not v for v in ys]
    tp = sum(r and i for r, i in zip(rej, inval))
    out = {"n": len(itens), "julgados": len(julg), "abstencoes": len(itens) - len(julg),
           "kappa": round(kappa(pred, ys), 3), "aceitos": sum(pred),
           "valido_entre_aceitos": round(sum(v for p, v in julg if p) / max(1, sum(pred)), 3),
           "precisao_reprovado_invalido": round(tp / max(1, sum(rej)), 3), "recall_invalidos": round(tp / max(1, sum(inval)), 3),
           "validos_perdidos_pct": round(100 * sum(r and v for r, v in zip(rej, ys)) / max(1, sum(ys)), 1),
           "taxa_valida_sem_filtro": round(sum(ys)/len(ys), 3)}
    (SAIDA / "calibracao_maritaca.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("Meta: kappa >= 0,6. Abaixo disso o juiz NÃO decide nada; o professor continua sendo o único árbitro.")


def cmd_avaliar(a):
    sys.path.insert(0, str(ROOT / "src"))
    import rft_filtrar as rf
    brs = {}
    for par in a.braco:
        nome, arq = par.split("=", 1)
        am = [json.loads(l) for l in Path(arq).read_text(encoding="utf-8").splitlines() if l.strip()]
        ok, _ = rf.camada1_sem_dedupe(am)
        if a.por_prompt:
            ok = [s for s in ok if s["k"] < a.por_prompt]
        brs[nome] = (am if not a.por_prompt else [s for s in am if s["k"] < a.por_prompt], ok)
    itens, vistos = [], set()
    for nome, (am, ok) in brs.items():
        for s in ok:
            uid = f"{nome}:{s['pid']}:{s['k']}"
            if uid not in vistos:
                vistos.add(uid)
                q = s["q"]
                itens.append({"uid": uid, "enunciado": q["enunciado"], "alternativas": q["alternativas"], "gabarito": q["resposta_correta"]})
    res = julgar(itens, a.real, a.max_chamadas, a.teto_brl)
    if not res:
        return
    acc = {}
    print("\nbraço     n   entregues  julgadas  aceitas  taxa/n   IC95")
    for nome, (am, ok) in brs.items():
        ac = {f"{s['pid']}:{s['k']}" for s in ok if res.get(f"{nome}:{s['pid']}:{s['k']}") is True}
        acc[nome] = ac
        jl = sum(res.get(f"{nome}:{s['pid']}:{s['k']}") is not None for s in ok)
        print(f"{nome:8s} {len(am):4d} {len(ok):9d} {jl:9d} {len(ac):8d} {100*len(ac)/len(am):5.1f}%  {wilson(len(ac), len(am))}")
    nomes = list(brs); ref = nomes[0]
    for n in nomes[1:]:
        comuns = {f"{s['pid']}:{s['k']}" for s in brs[ref][0]} & {f"{s['pid']}:{s['k']}" for s in brs[n][0]}
        b = len((acc[ref] - acc[n]) & comuns); c = len((acc[n] - acc[ref]) & comuns)
        print(f"{n} x {ref} ({len(comuns)} pares): só {ref} {b} | só {n} {c} | McNemar p={mcnemar_exato(b, c):.3f}")
    print("Abstenção conta como não aceito. Itens falha/sem questão contam como não aceitos.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nome in ("calibrar", "avaliar"):
        s = sub.add_parser(nome)
        s.add_argument("--real", action="store_true")
        s.add_argument("--max-chamadas", type=int, default=0)
        s.add_argument("--teto-brl", type=float, default=0)
    sub.choices["calibrar"].add_argument("--folha", default=str(FOLHA))
    sub.choices["avaliar"].add_argument("--braco", action="append", required=True)
    sub.choices["avaliar"].add_argument("--por-prompt", type=int, default=0, help="usa só as K primeiras amostras de cada prompt")
    a = ap.parse_args()
    {"calibrar": cmd_calibrar, "avaliar": cmd_avaliar}[a.cmd](a)


if __name__ == "__main__":
    main()
