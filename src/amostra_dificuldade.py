"""Validação humana do RÓTULO DE DIFICULDADE da base de treino (2026-10-14).

Motivo: na folha cega da base (`outputs/amostra_base_v7b`), 15 das 22 reprovações do professor
foram `dificuldade_adequada = n` (itens rotulados "Difícil" que são uma adição simples). Ele só disse
"adequada ou não"; faltava o rótulo CERTO. Aqui o professor classifica cada item em Fácil / Moderado /
Difícil SEM ver o rótulo atual. Com isso medimos o erro do rótulo por ano e por origem e passamos a
ter referência humana para validar qualquer método automático (regra ou Jev) antes de aplicá-lo.

  gerar   sorteia n itens estratificados por (ano, rótulo atual) e escreve a folha + chave oculta
  apurar  compara a classificação do professor com o rótulo atual (matriz, kappa ponderado, por ano)

Nunca escreve em data/. Itens REFORCO-/SINT- ficam de fora (rótulo vem do gerador, não é julgamento).
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amostra_humana as ah  # noqa: E402

NIVEIS = ("Fácil", "Moderado", "Difícil")
ABREV = {"f": "Fácil", "facil": "Fácil", "fácil": "Fácil", "m": "Moderado", "moderado": "Moderado",
         "d": "Difícil", "dificil": "Difícil", "difícil": "Difícil"}
COLUNAS = (["id", "ano", "habilidade", "descricao_habilidade", "enunciado"] + list(ah.LETRAS)
           + ["resolucao", "gabarito_indicado", "dificuldade_professor", "comentario"])
CORPUS = "data/train_curado_v7b.jsonl"


def candidatos(corpus):
    out = []
    for n, linha in enumerate(Path(corpus).read_text(encoding="utf-8").splitlines(), 1):
        if not linha.strip():
            continue
        d = json.loads(linha)
        meta = d.get("meta") or {}
        cod = str(meta.get("codigo_item", ""))
        if cod.startswith(("REFORCO-", "SINT")) or meta.get("dificuldade") not in NIVEIS:
            continue
        try:
            qs = json.loads(d["messages"][-1]["content"]).get("questoes") or []
        except (KeyError, ValueError):
            continue
        for k, q in enumerate(qs):
            origem = ("injecao" if meta.get("origem") == "injecao_saeb" else
                      "destilado" if cod.startswith("DIST-") else
                      "L2" if meta.get("lote") == "L2-2026-09" else "banco")
            out.append({"ref": f"L{n}.{k}", "item": cod, "q": q, "ano": meta.get("ano"),
                        "habilidade": meta.get("habilidade"), "rotulo": meta["dificuldade"],
                        "origem": origem, "sig": ah._assinatura(q)})
    return out


def cmd_gerar(a):
    cands = candidatos(a.corpus)
    if a.origens:
        cands = [c for c in cands if c["origem"] in a.origens]
    rng = random.Random(a.seed)
    celulas = defaultdict(list)
    for c in cands:
        celulas[(c["ano"], c["rotulo"])].append(c)
    for v in celulas.values():
        rng.shuffle(v)
    por_celula = max(1, a.n // len(celulas))
    escolhidas, vistas = [], set()
    for chave in sorted(celulas, key=str):
        pegas = 0
        for c in celulas[chave]:
            if pegas >= por_celula:
                break
            if c["sig"] in vistas:
                continue
            vistas.add(c["sig"])
            escolhidas.append(c)
            pegas += 1
    rng.shuffle(escolhidas)
    for c, i in zip(escolhidas, rng.sample(range(1000, 10000), len(escolhidas))):
        c["id"] = f"D{i}"
    saida = Path(a.saida_dir)
    saida.mkdir(parents=True, exist_ok=True)
    with open(saida / "folha_professor.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        for c in escolhidas:
            q = c["q"]
            alts = q.get("alternativas") or {}
            w.writerow({"id": c["id"], "ano": c["ano"], "habilidade": c["habilidade"],
                        "descricao_habilidade": ah._descricao(c["ano"], c["habilidade"]),
                        "enunciado": q.get("enunciado", ""), **{l: alts.get(l, "") for l in ah.LETRAS},
                        "resolucao": q.get("resolucao_passo_a_passo", ""),
                        "gabarito_indicado": q.get("resposta_correta", ""),
                        "dificuldade_professor": "", "comentario": ""})
    chave = {"versao": 1, "aviso": "NÃO ENTREGAR AO PROFESSOR — traz o rótulo atual.", "seed": a.seed,
             "corpus": a.corpus, "n": len(escolhidas), "por_celula": por_celula,
             "pool_por_celula": {f"{k[0]}|{k[1]}": len(v) for k, v in celulas.items()},
             "itens": {c["id"]: {"rotulo": c["rotulo"], "ano": c["ano"], "habilidade": c["habilidade"],
                                 "origem": c["origem"], "item": c["item"], "ref": c["ref"]}
                       for c in escolhidas}}
    (saida / "chave_oculta.json").write_text(json.dumps(chave, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"{len(escolhidas)} itens ({por_celula} por célula ano x rótulo; {len(celulas)} células)")
    print("  por rótulo atual:", dict(Counter(c["rotulo"] for c in escolhidas)))
    print("  por ano:", dict(sorted(Counter(str(c["ano"]) for c in escolhidas).items())))
    print(f"  {saida / 'folha_professor.csv'} (entregar) | {saida / 'chave_oculta.json'} (NÃO entregar)")


def kappa_ponderado(pares):
    """Kappa com pesos lineares (a distância Fácil<->Difícil pesa o dobro de Fácil<->Moderado)."""
    n = len(pares)
    if not n:
        return None
    idx = {x: i for i, x in enumerate(NIVEIS)}
    obs = Counter(pares)
    a = Counter(p for p, _ in pares)
    b = Counter(q for _, q in pares)
    peso = lambda x, y: abs(idx[x] - idx[y]) / 2  # noqa: E731
    do = sum(peso(x, y) * c for (x, y), c in obs.items()) / n
    de = sum(peso(x, y) * a[x] * b[y] for x in NIVEIS for y in NIVEIS) / (n * n)
    return None if de == 0 else 1 - do / de


def cmd_apurar(a):
    chave = json.loads(Path(a.chave).read_text(encoding="utf-8"))["itens"]
    pares, problemas, linhas = [], [], []
    with open(a.folha, encoding="utf-8-sig", newline="") as f:
        for i, l in enumerate(csv.DictReader(f), 2):
            qid = (l.get("id") or "").strip()
            if qid not in chave:
                problemas.append(f"linha {i}: id {qid!r} fora da chave")
                continue
            v = ABREV.get((l.get("dificuldade_professor") or "").strip().lower())
            if v is None:
                problemas.append(f"{qid} (linha {i}): dificuldade_professor inválida ou vazia")
                continue
            linhas.append((qid, chave[qid], v, l.get("comentario", "")))
            pares.append((chave[qid]["rotulo"], v))
    for p in problemas:
        print("PROBLEMA:", p)
    n = len(pares)
    if not n:
        raise SystemExit("ERRO: nenhuma linha válida")
    igual = sum(1 for r, p in pares if r == p)
    acima = sum(1 for r, p in pares if NIVEIS.index(r) > NIVEIS.index(p))
    abaixo = sum(1 for r, p in pares if NIVEIS.index(r) < NIVEIS.index(p))
    ic = ah.wilson(igual, n)
    print(f"n={n} | rótulo atual = professor: {igual}/{n} ({100*igual/n:.1f}%)"
          + (f" IC95 [{ic[0]:.1f}; {ic[1]:.1f}]" if ic else ""))
    print(f"  rótulo MAIS ALTO que o do professor (inflado): {acima} ({100*acima/n:.1f}%)")
    print(f"  rótulo MAIS BAIXO que o do professor:           {abaixo} ({100*abaixo/n:.1f}%)")
    k = kappa_ponderado(pares)
    print(f"  kappa ponderado: {k:.2f}" if k is not None else "  kappa ponderado: n/d")
    print("\nMatriz (linhas = rótulo atual, colunas = professor)")
    print(f"{'':10}" + "".join(f"{x:>10}" for x in NIVEIS))
    for r in NIVEIS:
        print(f"{r:10}" + "".join(f"{sum(1 for x, y in pares if x == r and y == p):>10}" for p in NIVEIS))
    for titulo, campo in (("ano", "ano"), ("origem", "origem")):
        print(f"\nPor {titulo}: inflado / igual / baixo")
        g = defaultdict(lambda: [0, 0, 0])
        for _, meta, v, _c in linhas:
            d = NIVEIS.index(meta["rotulo"]) - NIVEIS.index(v)
            g[str(meta[campo])][0 if d > 0 else (1 if d == 0 else 2)] += 1
        for k2 in sorted(g):
            print(f"  {k2:10} {g[k2][0]:>3} / {g[k2][1]:>3} / {g[k2][2]:>3}")
    Path(a.saida).write_text(json.dumps(
        {"n": n, "iguais": igual, "inflado": acima, "baixo": abaixo, "kappa_ponderado": k,
         "pares": [{"id": q, "rotulo": m["rotulo"], "professor": v, "ano": m["ano"],
                    "habilidade": m["habilidade"], "origem": m["origem"], "comentario": c}
                   for q, m, v, c in linhas], "problemas": problemas},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nApuração gravada em {a.saida}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    s = p.add_subparsers(dest="cmd", required=True)
    g = s.add_parser("gerar")
    g.add_argument("--corpus", default=CORPUS)
    g.add_argument("--n", type=int, default=108)
    g.add_argument("--seed", type=int, default=20261015)
    g.add_argument("--origens", nargs="+", choices=["banco", "L2", "destilado", "injecao"],
                   help="restringe a amostra a estas origens (ex.: banco = itens reais, teste-âncora)")
    g.add_argument("--saida-dir", default="outputs/amostra_dificuldade")
    g.set_defaults(f=cmd_gerar)
    r = s.add_parser("apurar")
    r.add_argument("--folha", required=True)
    r.add_argument("--chave", required=True)
    r.add_argument("--saida", default="outputs/amostra_dificuldade/apuracao.json")
    r.set_defaults(f=cmd_apurar)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
