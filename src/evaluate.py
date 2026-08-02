"""Avalia o modelo (base ou fine-tuned) no conjunto de validação.

Métricas:
  Estruturais (sobre gerações reais):
    - % de saídas com JSON válido
    - % com schema completo (enunciado, comando, alternativas A-D, gabarito)
    - % com gabarito em {A,B,C,D} e 4 alternativas distintas
    - distribuição dos gabaritos gerados (detecta viés)
    - % de saídas citando figura/imagem/gráfico (indesejado: modelo é só texto)
  Linguagem:
    - perplexity da resposta de referência (loss só nos tokens do assistant)
  Velocidade:
    - latência média/p95 por questão, tokens/s de geração, tokens de saída

Uso:
    python src/evaluate.py                     # avalia outputs/lora (fine-tuned)
    python src/evaluate.py --baseline          # avalia o Qwen3-1.7B base
    python src/evaluate.py --num-samples 10    # avaliação rápida
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

# Carrega HF_TOKEN do .env antes de qualquer acesso ao HF Hub.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from unsloth import FastLanguageModel  # deve ser o primeiro import (patches)

import torch
from tqdm import tqdm

from schema_utils import IMAGE_PATTERN, check_consistency, check_structure, parse_json

ROOT = Path(__file__).resolve().parent.parent
VAL_PATH = ROOT / "data" / "val.jsonl"
LORA_DIR = ROOT / "outputs" / "lora"
REPORT_PATH = ROOT / "outputs" / "eval_report.json"

BASE_MODEL = "unsloth/Qwen3-1.7B"
MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 512
# Amostragem recomendada pela Qwen para o modo non-thinking.
TEMPERATURE = 0.7
TOP_P = 0.8


def reference_perplexity(model, tokenizer, examples):
    """Perplexity média das respostas de referência (loss só no assistant)."""
    nlls, n_tokens = [], 0
    for ex in examples:
        messages = ex["messages"]
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(model.device)
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
            return_tensors="pt",
        ).to(model.device)

        labels = full_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100
        with torch.no_grad():
            loss = model(input_ids=full_ids, labels=labels).loss
        answer_tokens = int((labels != -100).sum())
        nlls.append(loss.item() * answer_tokens)
        n_tokens += answer_tokens
    return math.exp(sum(nlls) / n_tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", action="store_true",
        help="avalia o modelo base (sem fine-tuning) para comparação A/B",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    model_path = BASE_MODEL if args.baseline else str(LORA_DIR)
    if not args.baseline and not LORA_DIR.exists():
        raise SystemExit(
            f"{LORA_DIR} não existe — rode primeiro: python src/train.py "
            "(ou use --baseline para avaliar o modelo base)"
        )

    print(f"Carregando modelo: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    examples = [json.loads(line) for line in open(VAL_PATH, encoding="utf-8")]
    if args.num_samples:
        examples = examples[: args.num_samples]

    results, latencies, tokens_per_sec, output_tokens = [], [], [], []
    gabaritos, image_mentions = [], 0
    consistencia_ok, consistencia_verificavel = 0, 0

    for ex in tqdm(examples, desc="Gerando questões"):
        prompt_ids = tokenizer.apply_chat_template(
            ex["messages"][:-1],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(model.device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        out = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        new_tokens = out.shape[1] - prompt_ids.shape[1]
        text = tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)

        obj = parse_json(text)
        flags = check_structure(obj)
        if obj:
            gabaritos.append(str(obj.get("gabarito")))
            blob = json.dumps(obj, ensure_ascii=False)
        else:
            blob = text
        if IMAGE_PATTERN.search(blob):
            image_mentions += 1

        consistente, sugestao = check_consistency(obj)
        if consistente is not None:
            consistencia_verificavel += 1
            consistencia_ok += int(consistente)

        latencies.append(elapsed)
        tokens_per_sec.append(new_tokens / elapsed)
        output_tokens.append(new_tokens)
        results.append({
            "codigo_item_ref": ex["meta"]["codigo_item"],
            **flags,
            "consistencia_gabarito": consistente,
            "sugestao_gabarito": sugestao,
        })

    n = len(results)
    pct = lambda key: 100 * sum(r[key] for r in results) / n
    ppl = reference_perplexity(model, tokenizer, examples)

    from collections import Counter
    report = {
        "modelo": model_path,
        "num_amostras": n,
        "estrutura": {
            "json_valido_pct": round(pct("json_valido"), 1),
            "schema_completo_pct": round(pct("schema_completo"), 1),
            "gabarito_valido_pct": round(pct("gabarito_valido"), 1),
            "alternativas_distintas_pct": round(pct("alternativas_distintas"), 1),
            "justificativas_distintas_pct": round(pct("justificativas_distintas"), 1),
            "distribuicao_gabaritos": dict(Counter(gabaritos)),
            "mencoes_figura_pct": round(100 * image_mentions / n, 1),
            "consistencia_gabarito_pct": (
                round(100 * consistencia_ok / consistencia_verificavel, 1)
                if consistencia_verificavel else None
            ),
            "consistencia_verificavel_n": consistencia_verificavel,
            "consistencia_nota": (
                "% das respostas em que o gabarito bate com a conta resolvida na "
                "justificativa correspondente; medido só sobre as N amostras com "
                "uma expressão aritmética 'a op b = r' reconhecível no texto."
            ),
        },
        "linguagem": {"perplexity_referencia": round(ppl, 3)},
        "velocidade_gpu": {
            "latencia_media_s": round(statistics.mean(latencies), 2),
            "latencia_p95_s": round(sorted(latencies)[int(0.95 * (n - 1))], 2),
            "tokens_por_segundo": round(statistics.mean(tokens_per_sec), 1),
            "tokens_saida_media": round(statistics.mean(output_tokens), 1),
            "nota": (
                "Medido na GPU local (RTX 3060). O tempo de resposta real no "
                "mobile deve ser medido com o GGUF via llama-bench/llama-cli."
            ),
        },
        "detalhes": results,
    }

    REPORT_PATH.parent.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n===== Avaliação: {model_path} ({n} amostras) =====")
    for section in ("estrutura", "linguagem", "velocidade_gpu"):
        print(f"[{section}]")
        for k, v in report[section].items():
            if k not in ("nota", "consistencia_nota"):
                print(f"  {k}: {v}")
    print(f"\nRelatório completo: {REPORT_PATH}")


if __name__ == "__main__":
    main()
