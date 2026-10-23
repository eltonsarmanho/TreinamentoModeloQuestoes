"""Gera N questões com um modelo (pipeline de produção) e monta a folha cega para o professor.

Cada questão é uma chamada de `generate_validated` (grammar GBNF + verificador + best-of-N),
com o prompt de UMA questão e o sufixo de diversidade do app. Só entram na folha questões que o
pipeline ENTREGA (status diferente de `falha`), porque é isso que o app mostraria ao usuário.

Saídas (em --saida-dir):
  folha_professor.csv  entregar ao professor (ids opacos, embaralhada, colunas de avaliação vazias)
  chave_oculta.json    NÃO entregar: id -> modelo, ano, habilidade, dificuldade, status, seed
  INSTRUCOES_PROFESSOR.md (cópia das instruções padrão, com o número de questões)

Uso (4B fine-tunado):
    venv/bin/python src/gerar_folha_professor.py \
        --gguf outputs/qwen3_4b/Qwen3-4B-Q4_K_M.gguf \
        --lora outputs/qwen3_4b/lora_4b_v7b_s44.f16.gguf \
        --rotulo ft4b --n 40 --seed 20261015 --saida-dir outputs/folha_professor_ft4b_40

Uso (1.7B v3, o publicado):
    venv/bin/python src/gerar_folha_professor.py \
        --gguf outputs/gguf_v3_gguf/qwen3-1.7b.Q4_K_M.gguf --rotulo v3 --n 40 \
        --saida-dir outputs/folha_professor_v3_40
"""
import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import diversidade as dv  # noqa: E402
import unidades_tematicas as ut  # noqa: E402
import test_model as tm  # noqa: E402
from gerar_lote import prompt_com_sufixo  # noqa: E402

INSTRUCOES_PADRAO = ROOT / "outputs" / "amostra_3bracos" / "INSTRUCOES_PROFESSOR.md"
DIFICULDADES = ["Fácil", "Moderado", "Difícil"]
ANOS = ["2º", "5º", "9º"]
COLUNAS_BASE = ["id", "ano", "habilidade", "descricao_habilidade", "dificuldade_pedida",
                "enunciado", "A", "B", "C", "D", "E", "resolucao", "gabarito_modelo"]
COLUNAS_AVALIACAO = ["correcao_matematica", "resposta_unica", "dados_suficientes",
                     "aderencia_ano_habilidade", "contexto_plausivel", "dificuldade_adequada",
                     "valida_geral", "comentario"]


# Unidades temáticas oficiais (BNCC), mesma fonte usada na apuração da amostra humana
# (src/unidades_tematicas.py): Números, Álgebra, Geometria, Grandezas e medidas,
# Probabilidade e estatística. Substitui a classificação por palavras-chave usada antes
# (2026-10-18): esta é a fonte única do projeto, testada em tests/test_unidades_tematicas.py.
CONTEUDOS = list(ut.UNIDADES)


def sortear_combinacoes(n, seed, conteudo=None):
    """Pares (ano, habilidade) da taxonomia, sorteados com seed; dificuldade em rodízio.
    Com `conteudo`, só habilidades daquela unidade temática (ut.unidade_de)."""
    tax = dv.carregar_taxonomia()
    pares = sorted(tax["habilidades"].keys())
    rng = random.Random(seed)
    rng.shuffle(pares)
    combos = []
    for i, chave in enumerate(pares):
        ano, hab = chave.split("|")
        desc = (tax["habilidades"][chave].get("descricoes") or [""])[0]
        if not desc:
            continue
        if conteudo and ut.unidade_de(ano, hab) != conteudo:
            continue
        combos.append((ano, hab, desc, DIFICULDADES[i % len(DIFICULDADES)]))
    return combos


def gerar(n, seed, gguf, lora, conteudo=None, max_tentativas=3):
    """Gera até `n` questões ENTREGUES (não falhas). Retorna lista de registros."""
    tm.LORA_PATH = Path(lora) if lora else None
    llama_cli = tm.find_llama_cli(None)
    grammar = tm.GRAMMAR_PATH if tm.GRAMMAR_PATH.exists() else None
    combos = sortear_combinacoes(n * max_tentativas, seed, conteudo)
    if not combos:
        raise SystemExit(f"nenhuma habilidade na categoria {conteudo!r}")
    entregues, tentativas = [], 0
    for i, (ano, hab, desc, dif) in enumerate(combos):
        if len(entregues) >= n:
            break
        tentativas += 1
        prompt, slot = prompt_com_sufixo(ano, hab, desc, dif, seed=i)
        r = tm.generate_validated(llama_cli, gguf, prompt, tm.DEFAULT_THREADS, tm.MAX_NEW_TOKENS,
                                  grammar=grammar, retries=1, quantidade=1, base_seed=seed + i,
                                  modo_geometria=tm.MODO_GEOMETRIA)
        obj, status = r.get("obj"), r.get("status")
        if status == "falha" or not obj:
            print(f"  [{i + 1}] {ano} {hab} {dif}: falha (não entregue)")
            continue
        entregues.append({"ano": ano, "habilidade": hab, "descricao": desc, "dificuldade": dif,
                          "status": status, "seed": seed + i, "questao": obj})
        print(f"  [{len(entregues)}/{n}] {ano} {hab} {dif}: {status}")
    print(f"entregues {len(entregues)}/{n} em {tentativas} sorteios")
    return entregues


def montar_folha(registros, rotulo, saida, seed):
    rng = random.Random(seed)
    ordem = list(range(len(registros)))
    rng.shuffle(ordem)
    linhas, chave = [], {"modelo": rotulo, "itens": {}}
    for k, idx in enumerate(ordem, start=1):
        reg = registros[idx]
        q = reg["questao"]
        qid = f"Q{1000 + k}"
        alt = q["alternativas"]
        linhas.append({
            "id": qid, "ano": reg["ano"], "habilidade": reg["habilidade"],
            "descricao_habilidade": reg["descricao"], "dificuldade_pedida": reg["dificuldade"],
            "enunciado": q["enunciado"], **{L: alt.get(L, "") for L in "ABCDE"},
            "resolucao": q["resolucao_passo_a_passo"], "gabarito_modelo": q["resposta_correta"],
            **{c: "" for c in COLUNAS_AVALIACAO},
        })
        chave["itens"][qid] = {"modelo": rotulo, "ano": reg["ano"], "habilidade": reg["habilidade"],
                               "dificuldade": reg["dificuldade"], "status": reg["status"],
                               "seed": reg["seed"]}
    saida.mkdir(parents=True, exist_ok=True)
    with open(saida / "folha_professor.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_BASE + COLUNAS_AVALIACAO)
        w.writeheader()
        w.writerows(linhas)
    (saida / "chave_oculta.json").write_text(json.dumps(chave, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    if INSTRUCOES_PADRAO.exists():
        txt = INSTRUCOES_PADRAO.read_text(encoding="utf-8").replace("60 questões", f"{len(linhas)} questões")
        (saida / "INSTRUCOES_PROFESSOR.md").write_text(txt, encoding="utf-8")
    return len(linhas)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gguf", required=True, help="GGUF base (ex.: Qwen3-4B Q4_K_M ou o 1.7B v3)")
    p.add_argument("--lora", default=None, help="adaptador LoRA em GGUF (opcional)")
    p.add_argument("--gramatica", default=None,
                   help="GBNF a usar (padrão grammars/questao.gbnf; modelos v8 exigem grammars/questao_v8.gbnf)")
    p.add_argument("--rotulo", required=True, help="nome do modelo na chave (ex.: ft4b, v3)")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--saida-dir", required=True)
    p.add_argument("--conteudo", choices=CONTEUDOS, default=None,
                   help="só habilidades desta unidade temática BNCC (ver src/unidades_tematicas.py)")
    p.add_argument("--flash-attn", action="store_true",
                   help="opt-in: passa -fa on ao llama-server (mais rápido no prefill; não é o padrão)")
    a = p.parse_args()
    if a.gramatica:
        tm.GRAMMAR_PATH = Path(a.gramatica).resolve()
    tm.FLASH_ATTN = a.flash_attn
    registros = gerar(a.n, a.seed, a.gguf, a.lora, a.conteudo)
    total = montar_folha(registros, a.rotulo, Path(a.saida_dir), a.seed)
    print(f"folha: {a.saida_dir}/folha_professor.csv ({total} questões)")
    print("chave: NÃO entregar ao professor.")


if __name__ == "__main__":
    main()
