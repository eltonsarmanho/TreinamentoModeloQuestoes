"""Tarefa F: apuração de um ou vários lotes (possivelmente parciais) da folha cega do 4B.

Diferença para `amostra_humana.py apurar`: junta vários arquivos de lote, ignora SILENCIOSAMENTE as
linhas ainda em branco (lote não terminado), recusa linhas parcialmente preenchidas ou com valor
fora de s/n (lista os ids) e informa quantos itens de cada unidade temática entraram, para o
leitor ver se a amostra apurada segue balanceada. A apuração em si (Wilson 95%, por unidade, por
origem, núcleo) é a MESMA função de `amostra_humana`.

Uso:
  venv/bin/python src/f_apurar_lotes.py \\
      --lotes outputs/melhoria_v8/F/folha_lote_01_preenchida.csv [outros lotes...] \\
      --chave outputs/melhoria_v8/F/chave/chave_oculta.json [--saida apuracao_parcial.json]
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amostra_humana as ah  # noqa: E402


def ler_lotes(caminhos, chave):
    """(validas, em_branco, problemas, duplicadas). Mesma regra de validade de ah.ler_folha,
    mas linha totalmente em branco = pendente (não é problema)."""
    validas, em_branco, problemas, vistos = [], [], [], set()
    for caminho in caminhos:
        with open(caminho, encoding="utf-8-sig", newline="") as f:
            leitor = csv.DictReader(f)
            faltam = [c for c in ("id", *ah.OBRIGATORIAS) if c not in (leitor.fieldnames or [])]
            if faltam:
                raise SystemExit(f"ERRO: {caminho} sem as colunas {faltam}")
            linhas = list(leitor)
        for i, l in enumerate(linhas, 2):
            qid = (l.get("id") or "").strip()
            if qid not in chave["itens"]:
                problemas.append(f"{Path(caminho).name} linha {i}: id {qid!r} não está na chave")
                continue
            if qid in vistos:
                problemas.append(f"{Path(caminho).name} linha {i}: id {qid} repetido")
                continue
            vistos.add(qid)
            vals = {k: ah._sn(l.get(k)) for k in ah.OBRIGATORIAS}
            vazios = [k for k, v in vals.items() if v is None]
            ruins = [k for k, v in vals.items() if v == "invalido"]
            if len(vazios) == len(ah.OBRIGATORIAS):
                em_branco.append(qid)
                continue
            if vazios or ruins:
                problemas.append(f"{qid} ({Path(caminho).name} linha {i}): "
                                 + (f"em branco: {', '.join(vazios)}; " if vazios else "")
                                 + (f"fora de s/n: {', '.join(ruins)}" if ruins else ""))
                continue
            validas.append({"id": qid, **vals, "comentario": l.get("comentario", "")})
    return validas, em_branco, problemas


def balanco_por_unidade(validas, chave):
    c = Counter(chave["itens"][l["id"]]["unidade"] for l in validas)
    return dict(c), (max(c.values()) - min(c.values()) if c else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lotes", nargs="+", required=True)
    ap.add_argument("--chave", required=True)
    ap.add_argument("--saida")
    args = ap.parse_args(argv)
    chave = json.loads(Path(args.chave).read_text(encoding="utf-8"))
    validas, em_branco, problemas = ler_lotes(args.lotes, chave)
    if problemas:
        print(f"LINHAS PARCIAIS/INVÁLIDAS descartadas ({len(problemas)}):", file=sys.stderr)
        for p in problemas:
            print("  -", p, file=sys.stderr)
    if not validas:
        print("ERRO: nenhuma linha completa para apurar.", file=sys.stderr)
        return 2
    por_u, dif = balanco_por_unidade(validas, chave)
    print(f"Linhas completas: {len(validas)}; em branco (ignoradas): {len(em_branco)}; "
          f"descartadas por preenchimento parcial: {len(problemas)}")
    print(f"Itens apurados por unidade: {por_u}  ({'balanceado' if dif <= 1 else 'DESBALANCEADO, diferença ' + str(dif)})")
    res = ah.apurar_linhas(validas, chave)
    res.update({"problemas_de_preenchimento": problemas, "n_em_branco_ignoradas": len(em_branco),
                "n_na_chave": len(chave["itens"]), "itens_por_unidade": por_u})
    ah.imprimir(res)
    if args.saida:
        Path(args.saida).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nApuração gravada em: {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
