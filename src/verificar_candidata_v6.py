"""Verifica a candidata v6 contra a v5 (critérios V6-01..V6-12 do ciclo_v6/SPEC.md). Sem API.

Sai com código 1 se algum critério bloqueante falhar. Escreve ciclo_v6/ANTES_DEPOIS_v6.{json,md}.

    venv/bin/python src/verificar_candidata_v6.py
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
import montar_candidata_v6 as mc  # noqa: E402
import perfil_base as pb  # noqa: E402
import triar_resolucoes as tr  # noqa: E402

SAIDA = ROOT / "ciclo_v6"


def canon(t):
    return agb._canon(t)


def medir(base, rotulo):
    perfil, itens, pares = pb.perfilar(base, auditoria=None)
    aud = agb.auditar(base)
    triagem = tr.resumo(tr.triar_base(base))
    dist = ad.relatorio(ad.auditar(base))
    return {"perfil": perfil, "itens": itens, "pares": pares, "aud": aud, "triagem": triagem, "dist": dist}


def verificar(excecoes=()):
    res = {}
    v5, v6 = mc.V5, mc.V6
    ex5, ex6 = mc._linhas(v5), mc._linhas(v6)
    imut = json.loads((ROOT / "ciclo_v6" / "ENTRADAS_IMUTAVEIS.json").read_text(encoding="utf-8"))["arquivos"]
    res["V6-01 entradas imutáveis intactas"] = all(
        pb.sha256_arquivo(ROOT / r) == s for r, s in imut.items() if (ROOT / r).exists())

    # V6-05/09/11: reais presentes, gabarito por conteúdo, schema
    idx6 = {}
    for e in ex6:
        for q in mc._qs(e):
            idx6[pb.hash_questao(q)] = (e, q)
    prov = mc._linhas(mc.PROVENIENCIA)
    removidos = {r["hash"] for r in prov if r["operacao"] in ("quarentena", "removida_duplicata") and r.get("hash")}
    perdidos_reais, gab_mudou, schema_ok, sem_prov = [], 0, True, 0
    prov_hash = {r["hash"] for r in prov if r.get("hash")} | {r["hash_v6"] for r in prov if r.get("hash_v6")}
    for e in ex5:
        for q in mc._qs(e):
            h = pb.hash_questao(q)
            real = pb.origem_de(e["meta"]).startswith("real")
            if h in idx6:
                q6 = idx6[h][1]
                if canon(q6["alternativas"][q6["resposta_correta"]]) != canon(q["alternativas"][q["resposta_correta"]]):
                    gab_mudou += 1
            elif real:
                perdidos_reais.append(e["meta"]["codigo_item"])
    for e in ex6:
        try:
            if [m["role"] for m in e["messages"]] != ["system", "user", "assistant"]:
                schema_ok = False
            for q in mc._qs(e):
                if set(q) != {"enunciado", "alternativas", "resolucao_passo_a_passo", "resposta_correta", "difficulty"}:
                    schema_ok = False
                if list(q["alternativas"]) != list("ABCDE") or q["resposta_correta"] not in "ABCDE":
                    schema_ok = False
                if pb.hash_questao(q) not in prov_hash:
                    sem_prov += 1
        except Exception:  # noqa: BLE001
            schema_ok = False
    res["V6-09 nenhuma questão real perdida"] = not perdidos_reais
    res["V6-05 gabarito (por conteúdo) igual ao da v5 em todo item mantido"] = gab_mudou == 0
    res["V6-11 schema igual ao da v5"] = schema_ok
    res["V6-08 toda questão da candidata tem proveniência (hash)"] = sem_prov == 0

    # V6-10: duplicata exata e contaminação
    enun = Counter(dv.normalizar_texto(q["enunciado"]) for e in ex6 for q in mc._qs(e))
    val = mc.val_enunciados()
    res["V6-10 sem enunciado duplicado e sem contaminação com val*"] = (
        all(v == 1 for v in enun.values()) and not (set(enun) & val))

    a, b = medir(v5, "v5"), medir(v6, "v6")
    cls = Counter(x["classe"] for x in b["aud"])
    res["V6-12a REFUTADO = 0 e FRACO = 0"] = cls["REFUTADO"] == 0 and cls["FRACO"] == 0
    res["V6-12b matematicamente_problematica = 0"] = b["triagem"]["classes"]["matematicamente_problematica"] == 0
    ver = lambda m: 100 * sum(x["verificador"] == "ok" for x in m["itens"]) / len(m["itens"])  # noqa: E731
    res["V6-12c % verificável >= v5"] = ver(b) >= ver(a) - 1e-9
    res["V6-12d pouco_explicativas < v5"] = (b["triagem"]["classes"]["correta_pouco_explicativa"]
                                             < a["triagem"]["classes"]["correta_pouco_explicativa"])
    res["V6-12e near-dup pares <= v5"] = len(b["pares"]) <= len(a["pares"])
    vaz = lambda m: sum(len(h["subtemas_vazios"]) for h in m["perfil"]["habilidades_detalhe"].values())  # noqa: E731
    res["V6-12f subtemas vazios < v5"] = vaz(b) < vaz(a)
    res["V6-12g entropia de raciocínio >= v5"] = (b["perfil"]["diversidade_base"]["entropia_raciocinio"]
                                                  >= a["perfil"]["diversidade_base"]["entropia_raciocinio"])
    pct = b["perfil"]["letra_gabarito"]
    n6 = sum(pct.values())
    res["V6-12h letras entre 18% e 22%"] = all(18 <= 100 * pct.get(L, 0) / n6 <= 22 for L in "ABCDE")
    novos = [mc._qs(e)[0] for e in ex6 if e["meta"].get("origem_v6") == "injecao_v6"]
    dv_, expl = 0, 0
    for q in novos:
        i = ad.avaliar_questao(q)
        dv_ += i["distratores_valor"]
        expl += i["distratores_explicados"]
    expl_novos = 100 * expl / dv_ if dv_ else None
    res["V6-12i explicabilidade dos distratores nos itens novos >= 60,3% (v5)"] = expl_novos is not None and expl_novos >= 60.3
    nda_novos = sum(1 for q in novos if ad.avaliar_questao(q)["nda"])

    def resumo(m):
        p = m["perfil"]
        return {"questoes": p["questoes"], "exemplos": p["exemplos"], "auditoria_gabarito": dict(Counter(x["classe"] for x in m["aud"])),
                "resolucoes": m["triagem"]["classes"], "res_lt40": m["triagem"]["menos_de_40_chars"]["n"],
                "verificavel_pct": round(ver(m), 1), "near_dup_pares": len(m["pares"]), "subtemas_vazios": vaz(m),
                "habilidades_lt30": sum(h["n"] < 30 for h in p["habilidades_detalhe"].values()),
                "nda": p["nda"]["questoes_com_nda"], "letras": p["letra_gabarito"], "origem": p["por_origem"],
                "dificuldade": p["dificuldade_rotulo"], "diversidade": p["diversidade_base"],
                "distratores": {k: m["dist"]["geral"][k] for k in ("explicabilidade_pct", "nda", "revisao")}}
    exc = {k: v for k, v in res.items() if not v and any(k.startswith(e) for e in excecoes)}
    out = {"criterios": res, "excecoes_registradas": sorted(exc),
           "aprovada": all(v or k in exc for k, v in res.items()), "antes_v5": resumo(a), "depois_v6": resumo(b),
           "novos": {"n": len(novos), "explicabilidade_pct": None if expl_novos is None else round(expl_novos, 1),
                     "com_nda": nda_novos}, "reais_perdidos": perdidos_reais}
    return out


def md(o):
    a, b = o["antes_v5"], o["depois_v6"]
    L = ["# Candidata v6 × v5: antes e depois", "", "Gerado por `src/verificar_candidata_v6.py` (sem API).", "",
         "## Critérios de aprovação (SPEC V6-xx)", "", "| critério | resultado |", "|---|---|"]
    L += [f"| {k} | {'PASSA' if v else ('**EXCEÇÃO REGISTRADA**' if k in o['excecoes_registradas'] else '**FALHA**')} |"
          for k, v in o["criterios"].items()]
    L += ["", f"**Candidata {'APROVADA' if o['aprovada'] else 'NÃO APROVADA'} pelos critérios automáticos.** A amostra humana cega "
          "de 40 itens continua pendente.", "", "## Antes × depois", "", "| métrica | v5 | v6 |", "|---|---|---|"]
    for k in ("exemplos", "questoes", "res_lt40", "verificavel_pct", "near_dup_pares", "subtemas_vazios", "habilidades_lt30", "nda"):
        L.append(f"| {k} | {a[k]} | {b[k]} |")
    for k in ("auditoria_gabarito", "resolucoes", "origem", "dificuldade", "letras"):
        L.append(f"| {k} | {a[k]} | {b[k]} |")
    for k in a["diversidade"]:
        L.append(f"| diversidade.{k} | {a['diversidade'][k]} | {b['diversidade'][k]} |")
    for k in a["distratores"]:
        L.append(f"| distratores.{k} | {a['distratores'][k]} | {b['distratores'][k]} |")
    L += ["", f"Itens novos: {o['novos']['n']}; explicabilidade dos distratores {o['novos']['explicabilidade_pct']}%; "
          f"com NDA: {o['novos']['com_nda']}."]
    return "\n".join(L) + "\n"


def main():
    excecoes = [a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--excecao=")]
    o = verificar(excecoes)
    (SAIDA / "ANTES_DEPOIS_v6.json").write_text(json.dumps(o, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    (SAIDA / "ANTES_DEPOIS_v6.md").write_text(md(o), encoding="utf-8")
    for k, v in o["criterios"].items():
        print(("PASSA " if v else ("EXCEÇÃO " if k in o["excecoes_registradas"] else "FALHA ")) + k)
    print("APROVADA" if o["aprovada"] else "NÃO APROVADA")
    sys.exit(0 if o["aprovada"] else 1)


if __name__ == "__main__":
    main()
