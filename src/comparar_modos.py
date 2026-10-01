"""
Consolida o experimento de modos de geração (run_experimento_modos.sh) e aplica
a regra de decisão: QUALIDADE PRIMEIRO, depois tempo.

Lê outputs/diversidade_exp_<braco>_s<k>.json (um modo por arquivo), junta os
lotes de todas as seeds de cada braço e reagrega.

Regra: um braço é ELEGÍVEL se não for pior que o melhor braço, em cada métrica
de qualidade, além da margem de não-inferioridade (MARGENS). Entre os
elegíveis, vence o de menor tempo por questão pedida.

    python comparar_modos.py            # -> outputs/experimento_modos.md/.json
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from avaliar_diversidade import agregar

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
_RX = re.compile(r"diversidade_exp_(?P<braco>[A-Z]\w*)_s(?P<seed>\d+)\.json$")

DESCRICAO = {
    "A": "1 chamada 'Gere N' (app hoje)",
    "B": "N chamadas sem plano",
    "C": "N planejadas, 0 regenerações",
    "D": "N planejadas, 1 regeneração",
    "E": "N planejadas, 2 regenerações",
}

# métrica: (sentido, margem de não-inferioridade)
MARGENS = {
    "quantidade_entregue_pct": ("maior", 1.0),
    "schema_pct": ("maior", 1.0),
    "consistencia_inconsistente_pct": ("menor", 1.0),
    "aderencia_pct": ("maior", 2.0),
    "depende_de_visual_pct": ("menor", 1.0),
    "duplicate_rate": ("menor", 0.03),
    "coverage_score": ("maior", 0.05),
}


def carregar(out=None):
    out = out or OUT
    lotes = defaultdict(list)
    seeds = defaultdict(set)
    for p in sorted(out.glob("diversidade_exp_*_s*.json")):
        m = _RX.search(p.name)
        if not m:
            continue
        rel = json.loads(p.read_text(encoding="utf-8"))
        for nome, modo in rel["modos"].items():
            if "lotes" in modo:
                unica = nome == "atual" and rel.get("modo_atual") == "unico"
                lotes[m["braco"]] += [{**l, "_unica_chamada": unica} for l in modo["lotes"]]
                seeds[m["braco"]].add(int(m["seed"]))
    return lotes, seeds


def resumir(lotes):
    ag = agregar(lotes)
    ped = sum(l["quantidade_pedida"] for l in lotes)
    ag["tempo_por_questao_pedida_s"] = round(ag["tempo_total_s"] / ped, 2) if ped else None
    # espera até a 1ª questão: 1 chamada = o lote inteiro; N chamadas ~ tempo/questões.
    esperas = []
    for l in lotes:
        if l.get("_unica_chamada"):
            esperas.append(l["tempo_s"])
        elif l["quantidade_gerada"]:
            esperas.append(l["tempo_s"] / l["quantidade_gerada"])
    ag["espera_1a_questao_s"] = round(sum(esperas) / len(esperas), 2) if esperas else None
    return ag


def decidir(res):
    melhor = {}
    for met, (sentido, _) in MARGENS.items():
        vals = [r[met] for r in res.values() if r.get(met) is not None]
        if vals:
            melhor[met] = max(vals) if sentido == "maior" else min(vals)
    motivos = {}
    for b, r in res.items():
        falhas = []
        for met, (sentido, margem) in MARGENS.items():
            v = r.get(met)
            if v is None or met not in melhor:
                continue
            pior = (melhor[met] - v) if sentido == "maior" else (v - melhor[met])
            if pior > margem + 1e-9:
                falhas.append(f"{met} {v} vs melhor {melhor[met]}")
        motivos[b] = falhas
    elegiveis = [b for b in res if not motivos[b]]
    vencedor = min(elegiveis, key=lambda b: res[b]["tempo_por_questao_pedida_s"] or 1e9) if elegiveis else None
    return vencedor, elegiveis, motivos


def main():
    lotes, seeds = carregar()
    if not lotes:
        raise SystemExit("nenhum outputs/diversidade_exp_*_s*.json — rode run_experimento_modos.sh")
    res = {b: resumir(ls) for b, ls in sorted(lotes.items())}
    vencedor, elegiveis, motivos = decidir(res)

    linhas = [
        ("Questões entregues (%)", "quantidade_entregue_pct"), ("Schema completo (%)", "schema_pct"),
        ("Gabarito inconsistente (%)", "consistencia_inconsistente_pct"),
        ("Verificáveis (n)", "consistencia_verificavel_n"), ("Aderência (%)", "aderencia_pct"),
        ("Visual ausente (%)", "depende_de_visual_pct"), ("Duplicatas", "duplicate_rate"),
        ("Cobertura", "coverage_score"), ("diversity_score", "diversity_score"),
        ("Tempo por questão pedida (s)", "tempo_por_questao_pedida_s"),
        ("Espera até 1ª questão (s)", "espera_1a_questao_s"),
    ]
    bs = list(res)
    md = ["# Experimento de modos de geração", "",
          "| Braço | Descrição | seeds | lotes |", "|---|---|---|---|"]
    md += [f"| {b} | {DESCRICAO.get(b, b)} | {sorted(seeds[b])} | {res[b]['num_lotes']} |" for b in bs]
    md += ["", "| Métrica | " + " | ".join(bs) + " |", "|---|" + "---|" * len(bs)]
    for nome, k in linhas:
        md.append(f"| {nome} | " + " | ".join(str(res[b].get(k)) for b in bs) + " |")
    md += ["", f"**Elegíveis (qualidade não-inferior ao melhor):** {', '.join(elegiveis) or 'nenhum'}",
           f"**Vencedor (mais rápido entre elegíveis):** {vencedor} — {DESCRICAO.get(vencedor, '')}", ""]
    for b in bs:
        if motivos[b]:
            md.append(f"- {b} excluído: " + "; ".join(motivos[b]))
    texto = "\n".join(md) + "\n"
    (OUT / "experimento_modos.md").write_text(texto, encoding="utf-8")
    (OUT / "experimento_modos.json").write_text(json.dumps(
        {"resultados": res, "elegiveis": elegiveis, "vencedor": vencedor, "motivos": motivos,
         "margens": MARGENS}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(texto)
    print(f"Relatório: {OUT / 'experimento_modos.md'}")


if __name__ == "__main__":
    main()
