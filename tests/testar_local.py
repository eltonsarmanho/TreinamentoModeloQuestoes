"""
Simula localmente o fluxo do app: escolher ANO, escolher habilidade(s), quantidade
de questões por habilidade, dificuldade OPCIONAL, apertar "Gerar".

Por padrão usa o modelo PROMOVIDO (outputs/gguf_diversidade_gguf), não o que
test_model.find_gguf() pegaria sozinho (ordena alfabeticamente e cairia em
gguf_candidato_gguf — o modelo ANTIGO). Sempre usa o modo planejado quando
quantidade > 1 (gerar_lote.gerar_lote_planejado), que é o modo decidido para o
app (ver Doc/PLANO_MELHORIA_GATES.md): N chamadas de 1 questão, cada uma com o
sufixo de subtema/contexto, guardas de inferência e permutação de gabarito.

Uso interativo (pede ano, lista habilidades do banco, deixa escolher várias):
    python tests/testar_local.py

Uso direto (não interativo, bom para repetir testes):
    python tests/testar_local.py --ano 5º --habilidades H17,H21 --quantidade 5
    python tests/testar_local.py --ano 9º --habilidades H17 --quantidade 10 --dificuldade Difícil
    python tests/testar_local.py --ano 2º --habilidades todas --quantidade 3

Sem --dificuldade, o app deixaria em aberto; aqui replicamos isso rodando as 3
dificuldades (Fácil/Moderado/Difícil) para cada habilidade, como uma seleção
"qualquer" faria no treino. Para fixar uma só, passe --dificuldade.

Toda rodada grava um JSON consolidado em outputs/testes_locais/ para revisão.
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# test_model.py está neste mesmo diretório (tests/); gerar_lote.py está em
# src/ — só este segundo precisa ser adicionado ao caminho de import.
sys.path.insert(0, str(ROOT / "src"))

import test_model as tm  # noqa: E402
from gerar_lote import gerar_lote_planejado  # noqa: E402

MODELO_PROMOVIDO = ROOT / "outputs" / "gguf_diversidade_gguf" / "qwen3-1.7b.Q4_K_M.gguf"
OUT_DIR = ROOT / "outputs" / "testes_locais"
DIFICULDADES = ("Fácil", "Moderado", "Difícil")


def carregar_habilidades(ano=None):
    """(ano, habilidade, descricao) distintos do banco, como o menu interativo usa."""
    if not tm.DB_PATH.exists():
        return []
    con = sqlite3.connect(tm.DB_PATH)
    q = ("SELECT DISTINCT ano, habilidade, descricao_item FROM itens "
         "WHERE disciplina='Matemática' AND habilidade IS NOT NULL AND habilidade != ''")
    params = ()
    if ano:
        q += " AND ano = ?"
        params = (ano,)
    q += " ORDER BY ano, habilidade"
    rows = [r for r in con.execute(q, params).fetchall() if r[0] and r[0].lower() != "nan"]
    con.close()
    return rows


def escolher_interativo():
    todas = carregar_habilidades()
    if not todas:
        raise SystemExit(f"{tm.DB_PATH} não encontrado — rode com --ano/--habilidades/--descricao.")
    anos = sorted({o[0] for o in todas})
    print("Anos disponíveis:", ", ".join(anos))
    ano = input("Ano: ").strip()
    do_ano = [o for o in todas if o[0] == ano]
    if not do_ano:
        raise SystemExit("Ano inválido.")

    print("\nHabilidades disponíveis:")
    for idx, (_, hab, desc) in enumerate(do_ano):
        print(f"  [{idx}] {hab} — {desc[:90]}")
    escolha = input("Números das habilidades (ex: 0,2,5) ou 'todas': ").strip()
    if escolha.lower() == "todas":
        selecionadas = do_ano
    else:
        idxs = [int(x) for x in escolha.split(",") if x.strip().isdigit()]
        selecionadas = [do_ano[i] for i in idxs if 0 <= i < len(do_ano)]
    if not selecionadas:
        raise SystemExit("Nenhuma habilidade escolhida.")

    quantidade = input("Quantas questões por habilidade? [5]: ").strip()
    quantidade = int(quantidade) if quantidade.isdigit() and int(quantidade) > 0 else 5

    dif = input("Dificuldade (Fácil/Moderado/Difícil) [deixe em branco = as 3]: ").strip()
    dificuldades = (dif,) if dif else DIFICULDADES

    return ano, selecionadas, quantidade, dificuldades


def resolver_selecao(ano, habilidades_arg, descricao_arg):
    """Modo não interativo: resolve a lista de (ano, habilidade, descricao)."""
    todas = carregar_habilidades(ano)
    por_hab = {h: d for _, h, d in todas}
    if habilidades_arg.strip().lower() == "todas":
        if not todas:
            raise SystemExit(f"Nenhuma habilidade encontrada para o ano {ano} em {tm.DB_PATH}.")
        return [(ano, h, d) for _, h, d in todas]
    selecionadas = []
    for hab in habilidades_arg.split(","):
        hab = hab.strip()
        desc = por_hab.get(hab, descricao_arg)
        if not desc:
            print(f"Aviso: descrição de {hab} não encontrada no banco; usando string vazia "
                  f"(--descricao para fixar manualmente).")
        selecionadas.append((ano, hab, desc or ""))
    return selecionadas


def gerar_para_habilidade(llama_cli, gguf_path, ano, habilidade, descricao, dificuldade,
                          quantidade, threads, grammar, retries, max_tentativas_diversidade):
    inicio = time.perf_counter()
    r = gerar_lote_planejado(
        llama_cli, gguf_path, ano, habilidade, descricao, dificuldade, quantidade,
        threads, grammar=grammar, retries=retries,
        max_tentativas_diversidade=max_tentativas_diversidade, verbose=False,
    )
    elapsed = time.perf_counter() - inicio

    print(f"\n### {ano} | {habilidade} | {dificuldade} | pedidas={quantidade} "
          f"entregues={len(r['questoes'])} tempo={elapsed:.1f}s "
          f"({elapsed / max(len(r['questoes']), 1):.1f}s/questão)")
    for i, q in enumerate(r["questoes"], 1):
        tm.print_question(q, "")
        print()
    m = r.get("metricas") or {}
    if m:
        print(f"  diversity_score={m['diversity_score']:.3f} coverage={m['coverage_score']:.3f} "
              f"duplicate_rate={m['duplicate_rate']:.3f}")
    if r.get("regeneracoes_diversidade"):
        print(f"  regenerações por diversidade: {r['regeneracoes_diversidade']}")

    return {
        "ano": ano, "habilidade": habilidade, "descricao": descricao,
        "dificuldade": dificuldade, "quantidade_pedida": quantidade,
        "quantidade_entregue": len(r["questoes"]), "tempo_s": round(elapsed, 2),
        "tempo_por_questao_s": round(elapsed / max(len(r["questoes"]), 1), 2),
        "metricas": m, "regeneracoes_diversidade": r.get("regeneracoes_diversidade"),
        "questoes": r["questoes"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help=f"padrão: modelo promovido ({MODELO_PROMOVIDO.name})")
    ap.add_argument("--llama-cli", default=None)
    ap.add_argument("--threads", type=int, default=tm.DEFAULT_THREADS)
    ap.add_argument("--ano", help="ex: 5º — com --habilidades, roda sem menu interativo")
    ap.add_argument("--habilidades", help="ex: H17,H21 ou 'todas'")
    ap.add_argument("--descricao", default="", help="só usada se a habilidade não estiver no banco")
    ap.add_argument("--dificuldade", help="Fácil|Moderado|Difícil — se omitido, roda as 3")
    ap.add_argument("--quantidade", type=int, default=5, help="questões por habilidade")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--max-tentativas-diversidade", type=int, default=2)
    ap.add_argument("--no-grammar", action="store_true")
    args = ap.parse_args()

    gguf_path = Path(args.model) if args.model else (
        MODELO_PROMOVIDO if MODELO_PROMOVIDO.exists() else tm.find_gguf())
    if not gguf_path.exists():
        raise SystemExit(f"Modelo não encontrado: {gguf_path}")
    llama_cli = tm.find_llama_cli(args.llama_cli)
    grammar = None if args.no_grammar else (tm.GRAMMAR_PATH if tm.GRAMMAR_PATH.exists() else None)

    print(f"llama-cli: {llama_cli}")
    print(f"modelo:    {gguf_path}")
    print(f"grammar:   {'sim' if grammar else 'não'}\n")

    if args.ano and args.habilidades:
        selecionadas = resolver_selecao(args.ano, args.habilidades, args.descricao)
        quantidade = args.quantidade
        dificuldades = (args.dificuldade,) if args.dificuldade else DIFICULDADES
    else:
        ano, selecionadas, quantidade, dificuldades = escolher_interativo()

    resultados = []
    for ano, habilidade, descricao in selecionadas:
        for dificuldade in dificuldades:
            resultados.append(gerar_para_habilidade(
                llama_cli, gguf_path, ano, habilidade, descricao, dificuldade,
                quantidade, args.threads, grammar, args.retries,
                args.max_tentativas_diversidade))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saida = OUT_DIR / f"teste_{time.strftime('%Y%m%d_%H%M%S')}.json"
    saida.write_text(json.dumps({
        "modelo": str(gguf_path), "modelo_sha256_prefixo": None,
        "grammar": bool(grammar), "resultados": resultados,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    total_pedidas = sum(r["quantidade_pedida"] for r in resultados)
    total_entregues = sum(r["quantidade_entregue"] for r in resultados)
    print(f"\n{'=' * 70}\nResumo: {total_entregues}/{total_pedidas} questões entregues "
          f"em {len(resultados)} lote(s).\nSalvo em: {saida}")


if __name__ == "__main__":
    main()
