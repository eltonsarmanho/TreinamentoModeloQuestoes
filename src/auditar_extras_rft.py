"""Auditoria humana cega dos extras do RFT (ciclo v8, 2026-10-23).

Pergunta: dos 875 itens `rft_v8` que o Jev aprovou e entraram no treino do v8rftb, quantos o professor
aprova no critério núcleo? Para saber se a régua do professor é a mesma nos dois lados, a folha mistura,
às cegas, itens de CONTROLE da base curada (`injecao_saeb`, já medida em ~98% de núcleo).

  gerar   -> folha_professor.csv (ids opacos, mesma ordem de colunas das outras folhas), chave_oculta.json
             (com `origem` e `unidade`, para `amostra_humana.py apurar`) e INSTRUCOES_PROFESSOR.md.
  Sem API, sem GPU. Lê o corpus; não escreve em data/.

Uso:
  venv/bin/python src/auditar_extras_rft.py --saida-dir outputs/melhoria_v8/auditoria_extras_rft
  venv/bin/python src/amostra_humana.py apurar --folha <preenchida.csv> --chave <saida-dir>/chave_oculta.json
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import gerar_folha_professor as gf  # noqa: E402
from unidades_tematicas import unidade_de  # noqa: E402

CORPUS = SRC.parent / "data" / "train_curado_v8_rft_bal.jsonl"
HAB = re.compile(r"Ano: (?P<ano>[^.]+?) ano\. Habilidade: (?P<hab>\S+) — (?P<desc>.*?)\. Dificuldade: (?P<dif>[^.]+)\.")


def ler_itens(corpus=CORPUS):
    itens = {"rft_v8": [], "injecao_saeb": []}
    for linha in corpus.read_text(encoding="utf-8").splitlines():
        r = json.loads(linha)
        meta = r.get("meta") or {}
        if meta.get("origem") not in itens:
            continue
        user, asst = r["messages"][-2]["content"], r["messages"][-1]["content"]
        m = HAB.search(user)
        qs = json.loads(asst).get("questoes") or []
        if not m or len(qs) != 1:
            continue
        itens[meta["origem"]].append({
            "questao": qs[0], "ano": m["ano"].strip() + " ano" if "º" not in m["ano"] else m["ano"].strip(),
            "habilidade": m["hab"], "descricao": m["desc"], "dificuldade": m["dif"],
            "status": meta.get("status_verificador") or "base", "origem": meta["origem"],
            "codigo_item": meta.get("codigo_item")})
    return itens


def amostrar(itens, n, rng):
    """Estratificado por ano (proporcional, mínimo 1 por ano presente), determinístico."""
    por_ano = defaultdict(list)
    for it in itens:
        por_ano[it["ano"]].append(it)
    total = len(itens)
    sel = []
    for ano, lst in sorted(por_ano.items()):
        k = max(1, round(n * len(lst) / total))
        sel += rng.sample(lst, min(k, len(lst)))
    rng.shuffle(sel)
    return sel[:n] if len(sel) > n else sel


def montar(n_extras, n_controle, seed, saida):
    rng = random.Random(seed)
    itens = ler_itens()
    ext = amostrar(itens["rft_v8"], n_extras, rng)
    ctl = amostrar(itens["injecao_saeb"], n_controle, rng)
    regs = []
    for it in ext + ctl:
        regs.append({**it, "seed": seed})
    gf.montar_folha(regs, "auditoria_rft", saida, seed)
    # montar_folha embaralha com a mesma seed: reconstrói a correspondência id -> origem sem vazar na folha
    ordem = list(range(len(regs)))
    random.Random(seed).shuffle(ordem)
    chave = json.loads((saida / "chave_oculta.json").read_text(encoding="utf-8"))
    for k, idx in enumerate(ordem, start=1):
        reg = regs[idx]
        c = chave["itens"][f"Q{1000 + k}"]
        c["origem"] = "rft_extra" if reg["origem"] == "rft_v8" else "controle_base"
        c["unidade"] = unidade_de(reg["ano"], reg["habilidade"])
        c["codigo_item"] = reg["codigo_item"]
        c["status"] = reg["status"]
    (saida / "chave_oculta.json").write_text(json.dumps(chave, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(ext), len(ctl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida-dir", required=True)
    ap.add_argument("--extras", type=int, default=60)
    ap.add_argument("--controles", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261123)
    a = ap.parse_args()
    ne, nc = montar(a.extras, a.controles, a.seed, Path(a.saida_dir))
    print(f"folha: {ne} extras do RFT + {nc} controles da base em {a.saida_dir}")


if __name__ == "__main__":
    main()
