"""Verifica a candidata v7 contra a v6 (curadoria item a item do ciclo v7). Sem API.

Mesmas ferramentas de medição de src/verificar_candidata_v6.py (perfil_base,
auditar_gabaritos_base, triar_resolucoes, auditar_distratores), mas os critérios são os do
v7: curadoria não injeta itens novos, só confirma, corrige ou remove — então a régua é
"não piora em silêncio" nas mesmas dimensões, sem o critério de explicabilidade de
distratores de itens novos (V6-12i), que não se aplica aqui.

Sai com código 1 se algum critério falhar. Escreve ciclo_v7/ANTES_DEPOIS_v7.{json,md}.

    venv/bin/python src/verificar_v7.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import auditar_distratores as ad  # noqa: E402
import auditar_gabaritos_base as agb  # noqa: E402
import diversidade as dv  # noqa: E402
import perfil_base as pb  # noqa: E402
import triar_resolucoes as tr  # noqa: E402

V6 = ROOT / "data" / "train_curado_v6.jsonl"
V7 = ROOT / "data" / "train_curado_v7.jsonl"
SAIDA = ROOT / "ciclo_v7"


def medir(base):
    perfil, itens, pares = pb.perfilar(base, auditoria=None)
    aud = agb.auditar(base)
    triagem = tr.resumo(tr.triar_base(base))
    dist = ad.relatorio(ad.auditar(base))
    return {"perfil": perfil, "itens": itens, "pares": pares, "aud": aud, "triagem": triagem, "dist": dist}


def val_enunciados():
    vs = []
    for nome in ("val.jsonl", "val_frozen_v1.jsonl", "val_novos_v1.jsonl"):
        p = ROOT / "data" / nome
        if p.exists():
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                for q in json.loads(r["messages"][2]["content"])["questoes"]:
                    vs.append(dv.normalizar_texto(q["enunciado"]))
    return set(vs)


def verificar(v6=V6, v7=V7, excecoes=()):
    res = {}
    linhas7 = [json.loads(l) for l in open(v7, encoding="utf-8")]
    qs7 = [q for e in linhas7 for q in json.loads(e["messages"][2]["content"])["questoes"]]

    schema_ok, sem_schema = True, 0
    for e in linhas7:
        try:
            if [m["role"] for m in e["messages"]] != ["system", "user", "assistant"]:
                schema_ok = False
            for q in json.loads(e["messages"][2]["content"])["questoes"]:
                if set(q) != {"enunciado", "alternativas", "resolucao_passo_a_passo", "resposta_correta", "difficulty"}:
                    schema_ok = False
                if list(q["alternativas"]) != list("ABCDE") or q["resposta_correta"] not in "ABCDE":
                    schema_ok = False
        except Exception:  # noqa: BLE001
            schema_ok = False
    res["V7-01 schema igual ao da v6"] = schema_ok

    enun = Counter(dv.normalizar_texto(q["enunciado"]) for q in qs7)
    val = val_enunciados()
    res["V7-02 sem enunciado duplicado e sem contaminação com val*"] = all(v == 1 for v in enun.values()) and not (set(enun) & val)

    a, b = medir(v6), medir(v7)
    cls = Counter(x["classe"] for x in b["aud"])
    res["V7-03a REFUTADO = 0 e FRACO = 0"] = cls["REFUTADO"] == 0 and cls["FRACO"] == 0
    res["V7-03b matematicamente_problematica = 0"] = b["triagem"]["classes"]["matematicamente_problematica"] == 0
    ver = lambda m: 100 * sum(x["verificador"] == "ok" for x in m["itens"]) / len(m["itens"])  # noqa: E731
    res["V7-03c % verificável >= v6"] = ver(b) >= ver(a) - 1e-9
    res["V7-03d pouco_explicativas <= v6"] = (b["triagem"]["classes"]["correta_pouco_explicativa"]
                                              <= a["triagem"]["classes"]["correta_pouco_explicativa"])
    res["V7-03e near-dup pares <= v6"] = len(b["pares"]) <= len(a["pares"])
    pct = b["perfil"]["letra_gabarito"]
    n7 = sum(pct.values())
    res["V7-03f letras entre 18% e 22%"] = all(18 <= 100 * pct.get(L, 0) / n7 <= 22 for L in "ABCDE")
    dv_, expl = 0, 0
    for q in qs7:
        i = ad.avaliar_questao(q)
        dv_ += i["distratores_valor"]
        expl += i["distratores_explicados"]
    expl7 = 100 * expl / dv_ if dv_ else None
    dv6, expl6 = 0, 0
    for e in (json.loads(l) for l in open(v6, encoding="utf-8")):
        for q in json.loads(e["messages"][2]["content"])["questoes"]:
            i = ad.avaliar_questao(q)
            dv6 += i["distratores_valor"]
            expl6 += i["distratores_explicados"]
    expl6pct = 100 * expl6 / dv6 if dv6 else None
    res["V7-03g explicabilidade dos distratores >= v6"] = (expl7 is not None and expl6pct is not None
                                                           and expl7 >= expl6pct - 1e-9)

    def resumo(m, n_exemplos):
        p = m["perfil"]
        return {"exemplos": n_exemplos, "questoes": p["questoes"], "auditoria_gabarito": dict(Counter(x["classe"] for x in m["aud"])),
                "resolucoes": m["triagem"]["classes"], "verificavel_pct": round(ver(m), 1), "near_dup_pares": len(m["pares"]),
                "letras": p["letra_gabarito"], "diversidade": p["diversidade_base"],
                "distratores": {k: m["dist"]["geral"][k] for k in ("explicabilidade_pct", "nda", "revisao")}}
    n6 = sum(1 for _ in open(v6, encoding="utf-8"))
    exc = {k: v for k, v in res.items() if not v and any(k.startswith(e) for e in excecoes)}
    out = {"criterios": res, "excecoes_registradas": sorted(exc), "aprovada": all(v or k in exc for k, v in res.items()),
           "antes_v6": resumo(a, n6), "depois_v7": resumo(b, len(linhas7)),
           "distratores_explicabilidade": {"v6": expl6pct, "v7": expl7}}
    return out


def md(o):
    a, b = o["antes_v6"], o["depois_v7"]
    L = ["# Candidata v7 × v6: antes e depois", "", "Gerado por `src/verificar_v7.py` (sem API).", "",
         "## Critérios de aprovação", "", "| critério | resultado |", "|---|---|"]
    L += [f"| {k} | {'PASSA' if v else ('**EXCEÇÃO REGISTRADA**' if k in o['excecoes_registradas'] else '**FALHA**')} |"
          for k, v in o["criterios"].items()]
    L += ["", f"**Candidata {'APROVADA' if o['aprovada'] else 'NÃO APROVADA'} pelos critérios automáticos.**", "",
          "## Antes × depois", "", "| métrica | v6 | v7 |", "|---|---|---|"]
    for k in ("exemplos", "questoes", "verificavel_pct", "near_dup_pares"):
        L.append(f"| {k} | {a[k]} | {b[k]} |")
    for k in ("auditoria_gabarito", "resolucoes", "letras"):
        L.append(f"| {k} | {a[k]} | {b[k]} |")
    for k in a["diversidade"]:
        L.append(f"| diversidade.{k} | {a['diversidade'][k]} | {b['diversidade'][k]} |")
    for k in a["distratores"]:
        L.append(f"| distratores.{k} | {a['distratores'][k]} | {b['distratores'][k]} |")
    return "\n".join(L) + "\n"


def main():
    excecoes = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--excecao=")]
    o = verificar(excecoes=excecoes)
    SAIDA.mkdir(exist_ok=True)
    (SAIDA / "ANTES_DEPOIS_v7.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    (SAIDA / "ANTES_DEPOIS_v7.md").write_text(md(o), encoding="utf-8")
    for k, v in o["criterios"].items():
        print(("PASSA " if v else ("EXCEÇÃO " if k in o["excecoes_registradas"] else "FALHA ")) + k)
    print("APROVADA" if o["aprovada"] else "NÃO APROVADA")
    sys.exit(0 if o["aprovada"] else 1)


if __name__ == "__main__":
    main()
