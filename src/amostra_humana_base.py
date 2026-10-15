"""Folha cega para medir o TETO DA BASE de treino (Fase 3, 2026-10-14).

Pergunta: que fração dos itens de `train_curado_v7b.jsonl` o professor aprovaria com a mesma
rubrica da amostra humana dos modelos? Se a base já está abaixo de ~70%, nenhuma curadoria fina
leva o modelo acima disso. Reusa `amostra_humana` (mesmas colunas, mesma rubrica, mesma
`apurar`), só muda a origem das questões: o corpus, não um relatório de modelo.

Origens (rótulos da chave, não vistos pelo professor):
  banco      itens reais do banco SAEB (código MT...)
  L2         itens reais do lote L2-2026-09
  injecao    itens gerados por LLM e aceitos pelo pipeline de injeção (`origem: injecao_saeb`)
  destilado  itens gerados pelo modelo professor (código DIST-...)
Ficam de fora os `REFORCO-` e os `SINT-` (gabarito e distratores calculados em Python, corretos
por construção: entrariam só para inflar a taxa). A amostra é igual por origem (n/4 cada) e
balanceada por unidade dentro da origem; a taxa da BASE inteira se obtém ponderando as origens
pelo tamanho do pool (campo `pool_por_origem` da chave).

Uso: python src/amostra_humana_base.py --n 48 --saida-dir outputs/amostra_base_v7b
Depois: python src/amostra_humana.py apurar --folha <preenchida> --chave <dir>/chave_oculta.json
Nunca escreve em data/.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amostra_humana as ah  # noqa: E402
from unidades_tematicas import unidade_de  # noqa: E402

CORPUS = "data/train_curado_v7b.jsonl"


def rotulo_origem(meta):
    cod = str(meta.get("codigo_item", ""))
    if cod.startswith(("REFORCO-", "SINT-")):
        return None
    if cod.startswith("DIST-"):
        return "destilado"
    if meta.get("origem") == "injecao_saeb":
        return "injecao"
    if meta.get("lote") == "L2-2026-09" or "-L2-" in cod:
        return "L2"
    return "banco"


def candidatos(corpus):
    cands = []
    for n, linha in enumerate(Path(corpus).read_text(encoding="utf-8").splitlines(), 1):
        if not linha.strip():
            continue
        d = json.loads(linha)
        meta = d.get("meta") or {}
        origem = rotulo_origem(meta)
        if origem is None:
            continue
        try:
            qs = json.loads(d["messages"][-1]["content"]).get("questoes") or []
        except (KeyError, ValueError):
            continue
        ano, hab = meta.get("ano"), meta.get("habilidade")
        for k, q in enumerate(qs):
            cands.append({
                "origem": origem, "relatorio": f"{corpus}#L{n}.{k}", "item": meta.get("codigo_item"),
                "q": q, "ano": ano, "habilidade": hab,
                "unidade": unidade_de(ano, hab) if ano and hab else None,
                "dificuldade_pedida": meta.get("dificuldade"),
                "status": None, "geometria": None, "seed_rodada": 0,
                "artefato_sha256": None, "sig": ah._assinatura(q),
            })
    return cands


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", default=CORPUS)
    p.add_argument("--n", type=int, default=48, help="total; dividido igualmente entre as origens")
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--min-por-unidade", type=int, default=2)
    p.add_argument("--saida-dir", default="outputs/amostra_base_v7b")
    a = p.parse_args(argv)

    cands = candidatos(a.corpus)
    pool = Counter(c["origem"] for c in cands)
    origens = sorted(pool)
    por_origem = a.n // len(origens)
    escolhidas, cota = [], {}
    for o in origens:
        esc, cot = ah.sortear([c for c in cands if c["origem"] == o], por_origem, a.seed,
                              a.min_por_unidade)
        escolhidas += esc
        cota[o] = cot
    import random
    rng = random.Random(a.seed)
    rng.shuffle(escolhidas)  # a ordem da folha não revela a origem
    for c, i in zip(escolhidas, rng.sample(range(1000, 10000), len(escolhidas))):
        c["id"] = f"Q{i}"
    saida = Path(a.saida_dir)
    saida.mkdir(parents=True, exist_ok=True)
    with open(saida / "folha_professor.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ah.COLUNAS_FOLHA)
        w.writeheader()
        for c in escolhidas:
            w.writerow(ah.linha_folha(c))
    chave = {
        "versao": 1, "aviso": "NÃO ENTREGAR AO PROFESSOR — quebra o cegamento.",
        "seed": a.seed, "n_pedido": a.n, "n": len(escolhidas), "corpus": a.corpus,
        "pool_por_origem": dict(pool), "cota_por_origem_e_unidade": cota,
        "itens": {c["id"]: {"origem": c["origem"], "relatorio": c["relatorio"], "item": c["item"],
                            "unidade": c["unidade"] or ah.SEM_UNIDADE, "ano": c["ano"],
                            "habilidade": c["habilidade"], "status": None, "geometria": None,
                            "seed_rodada": 0} for c in escolhidas},
    }
    (saida / "chave_oculta.json").write_text(json.dumps(chave, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"Amostra: {len(escolhidas)} questões (pool {sum(pool.values())}: {dict(pool)})")
    print("  por origem: ", dict(Counter(c["origem"] for c in escolhidas)))
    print("  por unidade:", dict(Counter(chave["itens"][c["id"]]["unidade"] for c in escolhidas)))
    print(f"  {saida / 'folha_professor.csv'}  (entregar)")
    print(f"  {saida / 'chave_oculta.json'}    (NÃO entregar)")


if __name__ == "__main__":
    main()
