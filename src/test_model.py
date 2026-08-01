"""Testa o modelo REAL — o mesmo binário e arquivo .gguf que rodam offline no
app mobile — usando llama.cpp via subprocess. Não depende de torch/unsloth.

Diferente de evaluate.py (que mede o modelo em 4-bit via bitsandbytes/HF na
GPU, um caminho só de desenvolvimento), aqui a geração passa pelo artefato de
produção de fato: o .gguf quantizado Q4_K_M rodando em CPU pelo llama-cli,
igual ao que acontece no celular.

Modos:
  Interativo (padrão): escolha ano/habilidade/dificuldade num menu (as opções
  vêm do próprio banco DB/questoes.db) e veja a questão gerada, formatada.

      python src/test_model.py

  Direto, uma pergunta específica:

      python src/test_model.py --ano "5º" --habilidade H08 --descricao "Resolver problemas de adição ou subtração." --dificuldade Fácil

  Gerar N variações do mesmo pedido (ver estabilidade/variabilidade):

      python src/test_model.py --ano "9º" --habilidade H17 --n 5

  Lote real contra o conjunto de validação (sanity check fim a fim do
  artefato exportado, mesmas métricas estruturais do evaluate.py):

      python src/test_model.py --batch
      python src/test_model.py --batch --num-samples 10   # mais rápido
"""

import argparse
import json
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from extract_data import SYSTEM_PROMPT, USER_TEMPLATE
from schema_utils import IMAGE_PATTERN, check_consistency, check_structure, parse_json

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "DB" / "questoes.db"
VAL_PATH = ROOT / "data" / "val.jsonl"
REPORT_PATH = ROOT / "outputs" / "eval_report_gguf.json"

DEFAULT_LLAMA_CLI = Path.home() / ".unsloth" / "llama.cpp" / "llama-cli"
# Q4_K_M: melhor tokens/s medido no benchmark (ver README) com poucas threads —
# a geração é limitada por banda de memória, não por núcleos de CPU.
DEFAULT_THREADS = 4
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.8

SPEED_PATTERN = re.compile(
    r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s"
)


def find_llama_cli(explicit=None):
    if explicit:
        return Path(explicit)
    if DEFAULT_LLAMA_CLI.exists():
        return DEFAULT_LLAMA_CLI
    found = shutil.which("llama-cli")
    if found:
        return Path(found)
    raise SystemExit(
        "llama-cli não encontrado. Rode antes: python src/export_gguf.py "
        "(ele baixa e compila o llama.cpp), ou informe o caminho com --llama-cli."
    )


def find_gguf(explicit=None):
    if explicit:
        return Path(explicit)
    candidates = sorted((ROOT / "outputs").glob("**/*.gguf"))
    if not candidates:
        raise SystemExit(
            "Nenhum .gguf em outputs/ — rode antes: python src/export_gguf.py"
        )
    return candidates[0]


def generate(llama_cli, gguf_path, user_prompt, threads, max_new_tokens, seed=None):
    """Chama llama-cli em modo single-turn e retorna (texto, prompt_tps, gen_tps, latencia_total_s)."""
    cmd = [
        str(llama_cli),
        "-m", str(gguf_path),
        "--jinja",
        "-sys", SYSTEM_PROMPT,
        "-p", user_prompt,
        "-st", "-no-cnv",
        "-rea", "off",
        "--temp", str(TEMPERATURE),
        "--top-p", str(TOP_P),
        "-n", str(max_new_tokens),
        "--no-display-prompt", "--simple-io",
        "-t", str(threads),
        "--log-disable",
    ]
    if seed is not None:
        cmd += ["-s", str(seed)]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.perf_counter() - start

    stdout = result.stdout
    match = SPEED_PATTERN.search(stdout)
    prompt_tps = float(match.group(1)) if match else None
    gen_tps = float(match.group(2)) if match else None

    # Remove o eco do prompt ("> ...") e a linha de timing antes de parsear.
    answer = stdout
    marker = f"> {user_prompt}"
    if marker in answer:
        answer = answer.split(marker, 1)[1]
    answer = SPEED_PATTERN.sub("", answer).strip()

    return answer, prompt_tps, gen_tps, elapsed


def print_question(obj, raw_text):
    if obj is None:
        print("  [FALHA AO PARSEAR JSON] Saída bruta do modelo:")
        print(f"  {raw_text[:600]}")
        return
    print(f"  Enunciado: {obj.get('enunciado', '?')}")
    if obj.get("comando"):
        print(f"  Comando:   {obj['comando']}")
    alts = obj.get("alternativas", {})
    gabarito = obj.get("gabarito", "?")
    for letra in "ABCD":
        marca = " <- gabarito" if letra == gabarito else ""
        print(f"    {letra}) {alts.get(letra, '?')}{marca}")
    just = obj.get("justificativas")
    if just:
        print("  Justificativas:")
        for letra, texto in just.items():
            print(f"    {letra}: {texto}")


def run_one(llama_cli, gguf_path, ano, habilidade, descricao, dificuldade, threads, n):
    user_prompt = USER_TEMPLATE.format(
        ano=ano, habilidade=habilidade, descricao=descricao, dificuldade=dificuldade
    )
    print(f"\nPrompt: {user_prompt}\n")

    for i in range(n):
        if n > 1:
            print(f"--- Variação {i + 1}/{n} ---")
        text, prompt_tps, gen_tps, elapsed = generate(
            llama_cli, gguf_path, user_prompt, threads, MAX_NEW_TOKENS
        )
        obj = parse_json(text)
        flags = check_structure(obj)
        print_question(obj, text)
        figura = " | menciona figura!" if IMAGE_PATTERN.search(text) else ""
        print(
            f"  [json_valido={flags['json_valido']} "
            f"schema_completo={flags['schema_completo']} "
            f"gabarito_valido={flags['gabarito_valido']} "
            f"alternativas_distintas={flags['alternativas_distintas']}{figura}]"
        )
        consistente, sugestao = check_consistency(obj)
        if consistente is False:
            aviso = f" (conta bate com {sugestao})" if sugestao else ""
            print(f"  [!] gabarito INCONSISTENTE com a conta da justificativa{aviso}")
        elif consistente is True:
            print("  [ok] gabarito consistente com a conta da justificativa")
        gen_str = f"{gen_tps:.1f} tok/s" if gen_tps else "n/d"
        print(f"  Tempo total: {elapsed:.1f}s (inclui carregar o modelo) | Geração: {gen_str}\n")


def load_habilidades():
    """Carrega (ano, habilidade, descricao) distintos direto do banco, para o menu interativo."""
    if not DB_PATH.exists():
        return []
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT DISTINCT ano, habilidade, descricao_item FROM itens "
        "WHERE disciplina='Matemática' AND habilidade IS NOT NULL "
        "AND habilidade != '' ORDER BY ano, habilidade"
    ).fetchall()
    con.close()
    return [r for r in rows if r[0] and r[0].lower() != "nan"]


def interactive(llama_cli, gguf_path, threads):
    opcoes = load_habilidades()
    print(f"llama-cli: {llama_cli}")
    print(f"modelo:    {gguf_path}\n")

    if not opcoes:
        print("DB/questoes.db não encontrado — informe os campos manualmente.")
        while True:
            ano = input("Ano escolar (ex: 5º) ou 'sair': ").strip()
            if ano.lower() in ("sair", "exit", "q"):
                return
            habilidade = input("Habilidade (ex: H08): ").strip()
            descricao = input("Descrição da habilidade: ").strip()
            dificuldade = input("Dificuldade (Fácil/Moderado/Difícil): ").strip()
            run_one(llama_cli, gguf_path, ano, habilidade, descricao, dificuldade, threads, 1)
        return

    anos = sorted({o[0] for o in opcoes})
    while True:
        print("Anos disponíveis:", ", ".join(anos))
        ano = input("Escolha o ano (ou 'sair'): ").strip()
        if ano.lower() in ("sair", "exit", "q"):
            return
        do_ano = [o for o in opcoes if o[0] == ano]
        if not do_ano:
            print("Ano inválido.\n")
            continue

        print("\nHabilidades disponíveis:")
        for idx, (_, hab, desc) in enumerate(do_ano):
            print(f"  [{idx}] {hab} — {desc[:90]}")
        escolha = input("Escolha o número da habilidade: ").strip()
        if not escolha.isdigit() or not (0 <= int(escolha) < len(do_ano)):
            print("Escolha inválida.\n")
            continue
        _, habilidade, descricao = do_ano[int(escolha)]

        dificuldade = input("Dificuldade (Fácil/Moderado/Difícil) [Moderado]: ").strip() or "Moderado"
        n = input("Quantas variações gerar? [1]: ").strip()
        n = int(n) if n.isdigit() and int(n) > 0 else 1

        run_one(llama_cli, gguf_path, ano, habilidade, descricao, dificuldade, threads, n)
        print()


def batch(llama_cli, gguf_path, threads, num_samples):
    examples = [json.loads(line) for line in open(VAL_PATH, encoding="utf-8")]
    if num_samples:
        examples = examples[:num_samples]

    results, gen_tps_list, latencies, gabaritos = [], [], [], []
    image_mentions = 0
    consistencia_ok, consistencia_verificavel = 0, 0
    for i, ex in enumerate(examples, 1):
        user_msg = ex["messages"][1]["content"]
        print(f"[{i}/{len(examples)}] {user_msg[:80]}...")
        text, _, gen_tps, elapsed = generate(
            llama_cli, gguf_path, user_msg, threads, MAX_NEW_TOKENS
        )
        obj = parse_json(text)
        flags = check_structure(obj)
        if obj:
            gabaritos.append(str(obj.get("gabarito")))
        if IMAGE_PATTERN.search(text):
            image_mentions += 1
        if gen_tps:
            gen_tps_list.append(gen_tps)
        latencies.append(elapsed)

        consistente, sugestao = check_consistency(obj)
        if consistente is not None:
            consistencia_verificavel += 1
            consistencia_ok += int(consistente)

        results.append({
            "codigo_item_ref": ex["meta"]["codigo_item"],
            **flags,
            "consistencia_gabarito": consistente,
            "sugestao_gabarito": sugestao,
        })

    n = len(results)
    pct = lambda key: round(100 * sum(r[key] for r in results) / n, 1)
    report = {
        "artefato": str(gguf_path),
        "motor": "llama.cpp (llama-cli)",
        "num_amostras": n,
        "estrutura": {
            "json_valido_pct": pct("json_valido"),
            "schema_completo_pct": pct("schema_completo"),
            "gabarito_valido_pct": pct("gabarito_valido"),
            "alternativas_distintas_pct": pct("alternativas_distintas"),
            "distribuicao_gabaritos": dict(Counter(gabaritos)),
            "mencoes_figura_pct": round(100 * image_mentions / n, 1),
            "consistencia_gabarito_pct": (
                round(100 * consistencia_ok / consistencia_verificavel, 1)
                if consistencia_verificavel else None
            ),
            "consistencia_verificavel_n": consistencia_verificavel,
        },
        "velocidade_cpu_real": {
            "tokens_por_segundo_geracao": round(statistics.mean(gen_tps_list), 1) if gen_tps_list else None,
            "latencia_media_total_s": round(statistics.mean(latencies), 2),
            "nota": (
                "latencia_media_total_s inclui o carregamento do modelo a cada "
                "chamada (processo novo por questão); tokens_por_segundo_geracao "
                "vem do próprio llama.cpp e reflete só a fase de geração."
            ),
        },
        "detalhes": results,
    }

    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n===== Teste real (GGUF via llama.cpp) — {n} amostras =====")
    for section in ("estrutura", "velocidade_cpu_real"):
        print(f"[{section}]")
        for k, v in report[section].items():
            if k != "nota":
                print(f"  {k}: {v}")
    print(f"\nRelatório completo: {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--llama-cli", help="caminho do binário llama-cli")
    parser.add_argument("--model", help="caminho do .gguf (padrão: acha em outputs/)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)

    parser.add_argument("--ano", help="ex: 5º — pula o modo interativo")
    parser.add_argument("--habilidade", help="ex: H08")
    parser.add_argument("--descricao", default="", help="descrição da habilidade")
    parser.add_argument("--dificuldade", default="Moderado")
    parser.add_argument("--n", type=int, default=1, help="variações a gerar")

    parser.add_argument("--batch", action="store_true", help="roda sobre data/val.jsonl")
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    llama_cli = find_llama_cli(args.llama_cli)
    gguf_path = find_gguf(args.model)

    if args.batch:
        batch(llama_cli, gguf_path, args.threads, args.num_samples)
    elif args.ano and args.habilidade:
        run_one(
            llama_cli, gguf_path, args.ano, args.habilidade,
            args.descricao, args.dificuldade, args.threads, args.n,
        )
    else:
        interactive(llama_cli, gguf_path, args.threads)


if __name__ == "__main__":
    main()
