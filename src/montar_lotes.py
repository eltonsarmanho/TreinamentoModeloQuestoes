"""
Monta exemplos de treino com N>1 questões por resposta ("Gere N questão(ões)"),
o modo usado pelo app mobile (uma única chamada), a partir das questões N=1 já
curadas/destiladas. Sem custo de professor e determinístico (seed).

Cada lote é escolhido de forma gulosa dentro de (ano, habilidade, dificuldade):
prioriza subtema ainda não usado no lote, depois contexto não usado, e descarta
near-duplicatas (dv.e_near_duplicata). O prompt do usuário é o do app, SEM o
sufixo de condicionamento — o modelo precisa diversificar sozinho.

Uso:
    python montar_lotes.py                      # data/train_curado.jsonl -> data/train_multi.jsonl
    python montar_lotes.py --tamanhos 2,3,5,10 --lotes-por-grupo 3 --max-tokens 1900
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import diversidade as dv
import curar_diversidade as cd

ROOT = Path(__file__).resolve().parent.parent
ENTRADA = ROOT / "data" / "train_curado.jsonl"
SAIDA = ROOT / "data" / "train_multi.jsonl"

_RX_SUFIXO = re.compile(r"\s*Subtema:.*$", re.S)
_RX_QTD = re.compile(r"^Gere \d+ questão\(ões\)")


def prompt_app(user, n):
    """Prompt do app: remove o sufixo de condicionamento e ajusta a quantidade."""
    return _RX_QTD.sub(f"Gere {n} questão(ões)", _RX_SUFIXO.sub("", user).rstrip())


def estimar_tokens(texto):
    # ~3,2 caracteres/token no tokenizer Qwen para PT-BR com JSON (conservador).
    return int(len(texto) / 3.2) + 1


def escolher_lote(cands, n, rng, ano, hab, tax):
    """Seleção gulosa: subtema novo > contexto novo > aleatório; sem near-dup."""
    pool = cands[:]
    rng.shuffle(pool)
    lote, subs, ctxs = [], set(), set()
    while pool and len(lote) < n:
        pool.sort(key=lambda c: (c["cls"]["subtema"] in subs, c["cls"]["contexto"] in ctxs))
        c = pool.pop(0)
        if any(dv.e_near_duplicata(c["q"], x["q"]) for x in lote):
            continue
        lote.append(c)
        subs.add(c["cls"]["subtema"])
        ctxs.add(c["cls"]["contexto"])
    return lote if len(lote) == n else None


def montar(exemplos, tamanhos=(2, 3, 5, 10), lotes_por_grupo=3, max_tokens=1900, seed=0,
           taxonomia=None):
    tax = taxonomia or dv.carregar_taxonomia()
    rng = random.Random(seed)
    grupos = defaultdict(list)
    for ex in exemplos:
        qs = cd.questoes_do_exemplo(ex)
        if len(qs) != 1:
            continue
        m = ex.get("meta", {})
        ano, hab, dif = m.get("ano"), m.get("habilidade"), m.get("dificuldade")
        if not (ano and hab and dif):
            continue
        cls = dv.classificar_questao(qs[0], ano, hab, tax)
        grupos[(ano, hab, dif)].append({"ex": ex, "q": qs[0], "cls": cls})

    novos = []
    for (ano, hab, dif), cands in sorted(grupos.items()):
        for n in tamanhos:
            if len(cands) < n:
                continue
            for i in range(lotes_por_grupo):
                lote = escolher_lote(cands, n, rng, ano, hab, tax)
                if not lote:
                    break
                base = lote[0]["ex"]
                user = prompt_app(cd._msg(base, "user"), n)
                resp = json.dumps({"questoes": [c["q"] for c in lote]}, ensure_ascii=False)
                sistema = cd._msg(base, "system")
                if estimar_tokens(sistema + user + resp) > max_tokens:
                    continue
                novos.append({
                    "messages": [{"role": "system", "content": sistema},
                                 {"role": "user", "content": user},
                                 {"role": "assistant", "content": resp}],
                    "meta": {"codigo_item": f"LOTE-{hab}-{dif}-N{n}-{i}", "ano": ano,
                             "habilidade": hab, "dificuldade": dif, "lote_montado": True,
                             "subtemas": sorted({c["cls"]["subtema"] for c in lote})},
                })
    return novos


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default=str(ENTRADA))
    ap.add_argument("--saida", default=str(SAIDA))
    ap.add_argument("--tamanhos", default="2,3,5,10")
    ap.add_argument("--lotes-por-grupo", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1900,
                    help="descarta exemplos acima disso (deve caber em MAX_SEQ_LENGTH do train.py)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    exemplos = cd.carregar_jsonl(args.entrada)
    tamanhos = tuple(int(x) for x in args.tamanhos.split(","))
    novos = montar(exemplos, tamanhos, args.lotes_por_grupo, args.max_tokens, args.seed)
    with open(args.saida, "w", encoding="utf-8") as f:
        for ex in list(exemplos) + novos:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    por_n = defaultdict(int)
    for ex in novos:
        por_n[len(cd.questoes_do_exemplo(ex))] += 1
    print(f"{len(exemplos)} originais + {len(novos)} lotes montados -> {args.saida}")
    print("lotes por N:", dict(sorted(por_n.items())))


if __name__ == "__main__":
    main()
