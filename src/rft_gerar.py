"""Opção B, etapa 1: gerar K amostras por prompt com o 4B (inferência, NÃO é treino). Retomável.

Prompts: pares (ano, habilidade) da taxonomia, dificuldade em rodízio e o sufixo de diversidade do app
(os mesmos de gerar_folha_professor). Cada amostra usa seed própria e retries=0: queremos a distribuição
bruta do modelo, boas e ruins, porque o filtro vem depois (src/rft_filtrar.py). Cada linha do JSONL é uma
amostra com o status do verificador determinístico.

    venv/bin/python src/rft_gerar.py --dry-run                                  # só o plano e a estimativa
    venv/bin/python src/rft_gerar.py --gguf outputs/qwen3_4b/Qwen3-4B-Q4_K_M.gguf \
        --lora outputs/qwen3_4b/lora_4b_v7b_s44.f16.gguf --prompts 400 --amostras 6
Para um modelo v8 acrescente --gramatica grammars/questao_v8.gbnf (a gramática tem de ser a do treino).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

SAIDA = ROOT / "data" / "rft_amostras.jsonl"
SEG_POR_AMOSTRA = 11.0  # medido no 4B: ~21-26 s por questão com retries=1; sem retry ~11 s


def plano(n_prompts, seed):
    import gerar_folha_professor as gf
    from gerar_lote import prompt_com_sufixo
    base = gf.sortear_combinacoes(n_prompts, seed)
    if not base:
        raise SystemExit("taxonomia vazia")
    out = []
    for i in range(n_prompts):
        ano, hab, desc, dif = base[i % len(base)]
        dif = gf.DIFICULDADES[(i // len(base) + i) % 3]
        prompt, _slot = prompt_com_sufixo(ano, hab, desc, dif, seed=seed + i)
        out.append({"pid": i, "ano": ano, "habilidade": hab, "dificuldade": dif, "prompt": prompt})
    return out


def feitos(p):
    if not Path(p).exists():
        return set()
    return {(r["pid"], r["k"]) for r in map(json.loads, Path(p).read_text(encoding="utf-8").splitlines()) if r}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf"); ap.add_argument("--lora"); ap.add_argument("--gramatica")
    ap.add_argument("--prompts", type=int, default=400)
    ap.add_argument("--amostras", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--saida", default=str(SAIDA))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pl = plano(a.prompts, a.seed)
    total = a.prompts * a.amostras
    ja = feitos(a.saida)
    print(f"{a.prompts} prompts x {a.amostras} amostras = {total}; já feitas {len(ja)}; "
          f"estimativa restante ~{(total - len(ja)) * SEG_POR_AMOSTRA / 3600:.1f} h de GPU")
    if a.dry_run:
        print("exemplo de prompt:", pl[0]["prompt"][:200]); return
    if not a.gguf:
        raise SystemExit("--gguf obrigatório fora do --dry-run")
    import test_model as tm
    tm.LORA_PATH = Path(a.lora) if a.lora else None
    if a.gramatica:
        tm.GRAMMAR_PATH = Path(a.gramatica).resolve()
    grammar = tm.GRAMMAR_PATH if tm.GRAMMAR_PATH.exists() else None
    cli = tm.find_llama_cli(None)
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    with open(a.saida, "a", encoding="utf-8") as f:
        for p in pl:
            for k in range(a.amostras):
                if (p["pid"], k) in ja:
                    continue
                seed = a.seed * 1000 + p["pid"] * a.amostras + k
                r = tm.generate_validated(cli, a.gguf, p["prompt"], tm.DEFAULT_THREADS, tm.MAX_NEW_TOKENS,
                                          grammar=grammar, retries=0, quantidade=1, base_seed=seed,
                                          modo_geometria=tm.MODO_GEOMETRIA)
                f.write(json.dumps({**p, "k": k, "seed": seed, "status": r.get("status"),
                                    "obj": r.get("obj"), "grammar": grammar.name if grammar else None},
                                   ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
