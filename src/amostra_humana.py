"""Amostra humana CEGA: o professor julga questões sem saber de onde vêm.

As métricas automáticas dos gates (estrutura, aritmética, aderência ao rótulo
de dificuldade) não medem "a questão está certa e é boa". Este script monta
uma amostra para um(a) responsável pedagógico(a) avaliar com uma rubrica
fixa, e depois apura a validade com intervalo de confiança, por unidade
temática e por origem (ex.: baseline x candidato).

  gerar   relatórios de `tests/test_model.py --batch` COM texto (campo
          detalhes[].obj, gravado desde 2026-10-10) -> amostra estratificada por
          unidade temática (src/unidades_tematicas.py), ordem EMBARALHADA, id
          opaco. Saídas em --saida-dir:
            folha_professor.csv  o que o professor recebe. Sem NENHUM indício de
                                 origem (relatório, modelo, sufixo, status do
                                 verificador, rodada).
            chave_oculta.json    id opaco -> origem, item, unidade, status.
                                 NÃO entregar ao professor.
  apurar  folha preenchida + chave -> validade (valida_geral=s) por unidade e
          por origem com IC de Wilson 95%, e a taxa de cada dimensão da rubrica.

Exemplos:
  python src/amostra_humana.py gerar \\
      --relatorios baseline=outputs/relatorios/b_s0.json baseline=outputs/relatorios/b_s1.json \\
                   candidato=outputs/relatorios/c_s0.json candidato=outputs/relatorios/c_s1.json \\
      --n 40 --seed 20261010 --min-por-unidade 4 --saida-dir outputs/amostra_humana
  python src/amostra_humana.py apurar --folha folha_preenchida.csv \\
      --chave outputs/amostra_humana/chave_oculta.json --saida apuracao.json

Procedimento para o responsável pedagógico: Doc/PROCEDIMENTO_AMOSTRA_HUMANA.md.
Não chama modelo nem API: só lê relatórios e escreve arquivos.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import diversidade as dv  # noqa: E402
from observacoes_gate import itens_com_texto  # noqa: E402
from promover_checkpoint import wilson  # noqa: E402
from unidades_tematicas import UNIDADES, unidade_de  # noqa: E402

SEM_UNIDADE = "sem_unidade"
LETRAS = ("A", "B", "C", "D", "E")
DIMENSOES = ("correcao_matematica", "resposta_unica", "dados_suficientes",
             "aderencia_ano_habilidade", "contexto_plausivel", "dificuldade_adequada")
OBRIGATORIAS = DIMENSOES + ("valida_geral",)
COLUNAS_FOLHA = (["id", "ano", "habilidade", "descricao_habilidade", "dificuldade_pedida",
                  "enunciado"] + list(LETRAS) + ["resolucao", "gabarito_modelo"]
                 + list(OBRIGATORIAS) + ["comentario"])
SEED_PADRAO = 20261010


# --------------------------------------------------------------------------
# gerar
# --------------------------------------------------------------------------
def _assinatura(q):
    """Mesma questão (enunciado + alternativas, ignorando caixa/espaço/acento)."""
    texto = dv.normalizar_texto(dv.texto_questao(q, com_alternativas=True))
    return hashlib.sha1(texto.encode()).hexdigest()


def _parse_relatorios(specs):
    """['rotulo=caminho' | 'caminho'] -> [(rotulo, caminho)]."""
    out = []
    for s in specs:
        if "=" in s:
            rot, _, cam = s.partition("=")
        else:
            rot, cam = Path(s).stem, s
        out.append((rot.strip(), cam.strip()))
    return out


def candidatos(relatorios):
    """Todas as questões com texto: [{origem, relatorio, item, q, ...}].
    Relatórios sem texto são avisados e ignorados."""
    cands, avisos = [], []
    for rotulo, caminho in relatorios:
        rel = json.loads(Path(caminho).read_text(encoding="utf-8"))
        detalhes = {d.get("codigo_item_ref"): d for d in rel.get("detalhes") or []}
        itens = itens_com_texto(rel)
        if not itens:
            avisos.append(f"{caminho}: sem texto das questões (detalhes[].obj) — ignorado")
            continue
        for item, q in itens:
            d = detalhes.get(item, {})
            ano, hab = d.get("ano"), d.get("habilidade")
            cands.append({
                "origem": rotulo, "relatorio": str(caminho), "item": item, "q": q,
                "ano": ano, "habilidade": hab,
                "unidade": unidade_de(ano, hab) if ano and hab else None,
                "dificuldade_pedida": d.get("difficulty_pedida"),
                "status": d.get("status"), "geometria": d.get("geometria"),
                "seed_rodada": rel.get("seed_rodada", 0),
                "artefato_sha256": rel.get("artefato_sha256"),
                "sig": _assinatura(q),
            })
    return cands, avisos


def alocar(tamanhos, n, minimo):
    """Quantas questões sortear de cada unidade. `tamanhos` = {unidade: pool}.
    Primeiro o mínimo (limitado ao pool), depois o resto em rodízio entre as
    unidades que ainda têm questões — igualdade entre unidades, não
    proporcionalidade, porque o objetivo é medir validade POR unidade."""
    unidades = [u for u in (*UNIDADES, SEM_UNIDADE) if tamanhos.get(u)]
    cota = {u: min(minimo, tamanhos[u]) for u in unidades}
    if sum(cota.values()) > n:
        raise SystemExit(f"ERRO: mínimo por unidade ({minimo}) x {len(unidades)} unidades "
                         f"= {sum(cota.values())} > n={n}")
    restante = n - sum(cota.values())
    while restante > 0:
        livres = [u for u in unidades if cota[u] < tamanhos[u]]
        if not livres:
            break
        for u in livres:
            if restante == 0:
                break
            cota[u] += 1
            restante -= 1
    return cota


def sortear(cands, n, seed, minimo):
    """Amostra estratificada por unidade (e, dentro dela, balanceada entre as
    origens), sem repetir a mesma questão. Determinística em (cands, seed)."""
    rng = random.Random(seed)
    ordenados = sorted(cands, key=lambda c: (c["origem"], c["relatorio"], str(c["item"])))
    por_unidade = defaultdict(lambda: defaultdict(list))
    for c in ordenados:
        por_unidade[c["unidade"] or SEM_UNIDADE][c["origem"]].append(c)
    for u in por_unidade.values():
        for lista in u.values():
            rng.shuffle(lista)
    tamanhos = {u: len({c["sig"] for lista in og.values() for c in lista})
                for u, og in por_unidade.items()}
    cota = alocar(tamanhos, n, minimo)
    escolhidas, vistas = [], set()
    for u in (*UNIDADES, SEM_UNIDADE):
        if not cota.get(u):
            continue
        origens = sorted(por_unidade[u])
        fila = {o: list(por_unidade[u][o]) for o in origens}
        pegas = 0
        while pegas < cota[u] and any(fila.values()):
            for o in origens:
                while fila[o] and fila[o][0]["sig"] in vistas:
                    fila[o].pop(0)
                if fila[o] and pegas < cota[u]:
                    c = fila[o].pop(0)
                    vistas.add(c["sig"])
                    escolhidas.append(c)
                    pegas += 1
    rng.shuffle(escolhidas)  # a ordem da folha não revela unidade nem origem
    ids = rng.sample(range(1000, 10000), len(escolhidas))
    for c, i in zip(escolhidas, ids):
        c["id"] = f"Q{i}"
    return escolhidas, cota


def _descricao(ano, hab):
    try:
        h = dv.obter_habilidade(ano, hab)
        return (h or {}).get("descricoes", [""])[0] if h else ""
    except Exception:  # noqa: BLE001 — a descrição é só conveniência
        return ""


def linha_folha(c):
    q = c["q"]
    alts = q.get("alternativas") or {}
    linha = {"id": c["id"], "ano": c["ano"], "habilidade": c["habilidade"],
             "descricao_habilidade": _descricao(c["ano"], c["habilidade"]),
             "dificuldade_pedida": c["dificuldade_pedida"],
             "enunciado": q.get("enunciado", ""),
             **{l: alts.get(l, "") for l in LETRAS},
             "resolucao": q.get("resolucao_passo_a_passo", ""),
             "gabarito_modelo": q.get("resposta_correta", ""),
             "comentario": ""}
    linha.update({k: "" for k in OBRIGATORIAS})
    return linha


def cmd_gerar(args):
    cands, avisos = candidatos(_parse_relatorios(args.relatorios))
    for a in avisos:
        print("AVISO:", a, file=sys.stderr)
    if not cands:
        raise SystemExit("ERRO: nenhum relatório com texto das questões. Gere-os de novo com "
                         "tests/test_model.py --batch (grava detalhes[].obj desde 2026-10-10).")
    sem = sum(c["unidade"] is None for c in cands)
    if sem:
        print(f"AVISO: {sem} questões sem unidade temática mapeada (grupo '{SEM_UNIDADE}')",
              file=sys.stderr)
    escolhidas, cota = sortear(cands, args.n, args.seed, args.min_por_unidade)
    if len(escolhidas) < args.n:
        print(f"AVISO: só há {len(escolhidas)} questões distintas para n={args.n}", file=sys.stderr)

    saida = Path(args.saida_dir)
    saida.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: o Excel abre acentos corretamente sem passar pelo assistente de importação.
    with open(saida / "folha_professor.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_FOLHA)
        w.writeheader()
        for c in escolhidas:
            w.writerow(linha_folha(c))
    relatorios = {}
    for c in cands:
        relatorios.setdefault((c["origem"], c["relatorio"]), {
            "rotulo": c["origem"], "arquivo": c["relatorio"], "seed_rodada": c["seed_rodada"],
            "artefato_sha256": c["artefato_sha256"]})
    chave = {
        "versao": 1, "aviso": "NÃO ENTREGAR AO PROFESSOR — quebra o cegamento.",
        "seed": args.seed, "n_pedido": args.n, "n": len(escolhidas),
        "min_por_unidade": args.min_por_unidade, "cota_por_unidade": cota,
        "relatorios": list(relatorios.values()),
        "itens": {c["id"]: {"origem": c["origem"], "relatorio": c["relatorio"],
                            "item": c["item"], "unidade": c["unidade"] or SEM_UNIDADE,
                            "ano": c["ano"], "habilidade": c["habilidade"],
                            "status": c["status"], "geometria": c["geometria"],
                            "seed_rodada": c["seed_rodada"]} for c in escolhidas},
    }
    (saida / "chave_oculta.json").write_text(
        json.dumps(chave, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Amostra: {len(escolhidas)} questões (seed {args.seed}, pool {len(cands)})")
    print("  por unidade:", dict(Counter(chave["itens"][c["id"]]["unidade"] for c in escolhidas)))
    print("  por origem: ", dict(Counter(c["origem"] for c in escolhidas)))
    print(f"  {saida / 'folha_professor.csv'}  (entregar)")
    print(f"  {saida / 'chave_oculta.json'}    (NÃO entregar)")
    return 0


# --------------------------------------------------------------------------
# apurar
# --------------------------------------------------------------------------
def _sn(valor):
    """'s'/'n' normalizado; None se vazio; 'invalido' se outra coisa."""
    v = unicodedata.normalize("NFKD", str(valor or "")).encode("ascii", "ignore").decode()
    v = v.strip().lower()
    if not v:
        return None
    if v in ("s", "sim"):
        return "s"
    if v in ("n", "nao"):
        return "n"
    return "invalido"


def ler_folha(caminho, chave):
    """(linhas válidas, problemas, avisos). Linha válida = id da chave + as 7
    colunas obrigatórias em s/n."""
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        leitor = csv.DictReader(f)
        faltam = [c for c in ("id", *OBRIGATORIAS) if c not in (leitor.fieldnames or [])]
        if faltam:
            raise SystemExit(f"ERRO: folha sem as colunas {faltam}")
        linhas = list(leitor)
    problemas, avisos, validas, vistos = [], [], [], set()
    for i, l in enumerate(linhas, 2):  # 2 = primeira linha de dados (cabeçalho é a 1)
        qid = (l.get("id") or "").strip()
        if qid not in chave["itens"]:
            problemas.append(f"linha {i}: id {qid!r} não está na chave")
            continue
        if qid in vistos:
            problemas.append(f"linha {i}: id {qid} repetido")
            continue
        vistos.add(qid)
        vals = {k: _sn(l.get(k)) for k in OBRIGATORIAS}
        vazios = [k for k, v in vals.items() if v is None]
        ruins = [f"{k}={l.get(k)!r}" for k, v in vals.items() if v == "invalido"]
        if len(vazios) == len(OBRIGATORIAS):
            problemas.append(f"{qid} (linha {i}): linha vazia (não avaliada)")
        elif vazios:
            problemas.append(f"{qid} (linha {i}): campos em branco: {', '.join(vazios)}")
        if ruins:
            problemas.append(f"{qid} (linha {i}): valor fora de s/n: {'; '.join(ruins)}")
        if vazios or ruins:
            continue
        if vals["valida_geral"] == "s" and "n" in (vals[d] for d in DIMENSOES):
            avisos.append(f"{qid}: valida_geral=s com dimensão marcada n "
                          f"({', '.join(d for d in DIMENSOES if vals[d] == 'n')})")
        validas.append({"id": qid, **vals, "comentario": l.get("comentario", "")})
    ausentes = sorted(set(chave["itens"]) - vistos)
    if ausentes:
        problemas.append(f"{len(ausentes)} ids da chave não aparecem na folha: "
                         + ", ".join(ausentes[:10]) + ("..." if len(ausentes) > 10 else ""))
    return validas, problemas, avisos


def _taxa(k, n):
    ic = wilson(k, n)
    return {"n": n, "s": k, "pct": round(100 * k / n, 1) if n else None,
            "ic95_pct": list(ic) if ic else None}


def apurar_linhas(validas, chave):
    def grupo(chave_fn):
        g = defaultdict(list)
        for l in validas:
            g[chave_fn(chave["itens"][l["id"]])].append(l)
        return g

    def bloco(linhas):
        return {"validade": _taxa(sum(l["valida_geral"] == "s" for l in linhas), len(linhas)),
                "dimensoes": {d: _taxa(sum(l[d] == "s" for l in linhas), len(linhas))
                              for d in DIMENSOES}}
    por_u, por_o = grupo(lambda i: i["unidade"]), grupo(lambda i: i["origem"])
    cruz = grupo(lambda i: (i["unidade"], i["origem"]))
    return {
        "n_avaliadas": len(validas),
        "geral": bloco(validas),
        "por_unidade": {u: bloco(l) for u, l in sorted(por_u.items())},
        "por_origem": {o: bloco(l) for o, l in sorted(por_o.items())},
        "por_unidade_e_origem": {f"{u} | {o}": _taxa(sum(x["valida_geral"] == "s" for x in l), len(l))
                                 for (u, o), l in sorted(cruz.items())},
    }


def _linha(rotulo, t):
    ic = f"[{t['ic95_pct'][0]:.1f}; {t['ic95_pct'][1]:.1f}]" if t["ic95_pct"] else "n/d"
    pct = f"{t['pct']:.1f}%" if t["pct"] is not None else "n/d"
    return f"  {rotulo:<34} {t['s']:>3}/{t['n']:<3} {pct:>7}  IC95 {ic}"


def imprimir(res):
    print("=" * 74 + "\nAMOSTRA HUMANA — validade (valida_geral = s), IC de Wilson 95%\n" + "=" * 74)
    print(_linha("GERAL", res["geral"]["validade"]))
    print("\nPor unidade temática")
    for u, b in res["por_unidade"].items():
        print(_linha(u, b["validade"]))
    print("\nPor origem")
    for o, b in res["por_origem"].items():
        print(_linha(o, b["validade"]))
    print("\nPor unidade x origem")
    for k, t in res["por_unidade_e_origem"].items():
        print(_linha(k, t))
    print("\nDimensões da rubrica (% de 's')")
    cab = ["geral", *res["por_origem"]]
    print(f"  {'dimensão':<28}" + "".join(f"{c:>14}" for c in cab))
    for d in DIMENSOES:
        celulas = [res["geral"]["dimensoes"][d]] + [b["dimensoes"][d] for b in res["por_origem"].values()]
        print(f"  {d:<28}" + "".join(f"{(str(c['pct']) + '%') if c['pct'] is not None else 'n/d':>14}"
                                     for c in celulas))
    print("\nObservação: com n pequeno por célula os intervalos são largos; compare origens "
          "só quando os ICs não se sobrepõem.")


def cmd_apurar(args):
    chave = json.loads(Path(args.chave).read_text(encoding="utf-8"))
    validas, problemas, avisos = ler_folha(args.folha, chave)
    for a in avisos:
        print("AVISO:", a, file=sys.stderr)
    if problemas:
        print(f"PREENCHIMENTO INCOMPLETO/INVÁLIDO ({len(problemas)} problema(s)):", file=sys.stderr)
        for p in problemas:
            print("  -", p, file=sys.stderr)
        if not args.ignorar_incompletas:
            print("Corrija a folha ou use --ignorar-incompletas (apura só as linhas completas).",
                  file=sys.stderr)
            return 2
    if not validas:
        print("ERRO: nenhuma linha completa para apurar.", file=sys.stderr)
        return 2
    res = apurar_linhas(validas, chave)
    res["problemas_de_preenchimento"] = problemas
    res["avisos"] = avisos
    res["n_na_chave"] = len(chave["itens"])
    imprimir(res)
    if args.saida:
        Path(args.saida).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nApuração gravada em: {args.saida}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gerar", help="sorteia a amostra e escreve folha + chave")
    g.add_argument("--relatorios", nargs="+", required=True,
                   help="relatórios de test_model.py --batch com texto; 'rotulo=caminho' agrupa "
                        "origens (ex.: baseline=a.json baseline=b.json candidato=c.json)")
    g.add_argument("--n", type=int, default=40)
    g.add_argument("--seed", type=int, default=SEED_PADRAO)
    g.add_argument("--min-por-unidade", type=int, default=4)
    g.add_argument("--saida-dir", default="outputs/amostra_humana")
    a = sub.add_parser("apurar", help="calcula validade e dimensões da folha preenchida")
    a.add_argument("--folha", required=True)
    a.add_argument("--chave", required=True)
    a.add_argument("--saida")
    a.add_argument("--ignorar-incompletas", action="store_true")
    args = p.parse_args(argv)
    return cmd_gerar(args) if args.cmd == "gerar" else cmd_apurar(args)


if __name__ == "__main__":
    raise SystemExit(main())
