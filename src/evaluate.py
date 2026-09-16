"""Avalia o modelo (base ou fine-tuned) no conjunto de validação.

Métricas:
  Estruturais (sobre gerações reais):
    - % de saídas com JSON válido e wrapper {"questoes": [...]} válido
    - % com schema completo (enunciado, alternativas A-E, resposta_correta, difficulty)
    - % com resposta_correta em {A,B,C,D,E} e 5 alternativas distintas
    - distribuição das respostas corretas geradas (detecta viés)
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

from schema_utils import (
    DIFFICULTY_MAP,
    IMAGE_PATTERN,
    check_consistency,
    check_structure,
    extract_questao,
    parse_json,
)

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
    parser.add_argument("--model-path", default=None,
                        help="caminho explícito do modelo/adaptador a avaliar")
    parser.add_argument("--val", default=str(VAL_PATH),
                        help="conjunto de avaliação (padrão: data/val.jsonl)")
    parser.add_argument("--report", default=str(REPORT_PATH),
                        help="arquivo de saída do relatório")
    args = parser.parse_args()

    model_path = args.model_path or (BASE_MODEL if args.baseline else str(LORA_DIR))
    if not args.baseline and not args.model_path and not LORA_DIR.exists():
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

    examples = [json.loads(line) for line in open(args.val, encoding="utf-8")]
    if args.num_samples:
        examples = examples[: args.num_samples]

    results, latencies, tokens_per_sec, output_tokens = [], [], [], []
    respostas_corretas, image_mentions = [], 0
    consistencia_ok, consistencia_verificavel = 0, 0
    dif_aderente, dif_avaliavel = 0, 0
    from collections import defaultdict
    por_ano = defaultdict(lambda: {"n": 0, "estrutura_ok": 0, "dif_ok": 0,
                                   "consist_ok": 0, "consist_verif": 0})

    for idx, ex in enumerate(tqdm(examples, desc="Gerando questões"), 1):
        # Seed por amostra: dois modelos diferentes veem exatamente a mesma
        # sequência de amostragem, tornando a comparação pareada (ver nota
        # equivalente em test_model.generate_validated).
        torch.manual_seed(1000 + idx)
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
        flags = check_structure(obj, quantidade_esperada=1)
        questao = extract_questao(obj, 0)
        if questao:
            respostas_corretas.append(str(questao.get("resposta_correta")))
            blob = json.dumps(obj, ensure_ascii=False)
        else:
            blob = text
        if IMAGE_PATTERN.search(blob):
            image_mentions += 1

        consistente, sugestao = check_consistency(questao)
        if consistente is not None:
            consistencia_verificavel += 1
            consistencia_ok += int(consistente)

        latencies.append(elapsed)
        tokens_per_sec.append(new_tokens / elapsed)
        output_tokens.append(new_tokens)
        esperado = DIFFICULTY_MAP.get(ex["meta"].get("dificuldade"))
        aderente = None
        if esperado and questao:
            dif_avaliavel += 1
            aderente = questao.get("difficulty") == esperado
            dif_aderente += int(aderente)

        ano = ex["meta"].get("ano", "?")
        estrutura_ok = all(flags[k] for k in (
            "json_valido", "wrapper_valido", "schema_completo",
            "resposta_valida", "alternativas_distintas", "difficulty_valida"))
        por_ano[ano]["n"] += 1
        por_ano[ano]["estrutura_ok"] += int(estrutura_ok)
        por_ano[ano]["dif_ok"] += int(bool(aderente))
        if consistente is not None:
            por_ano[ano]["consist_verif"] += 1
            por_ano[ano]["consist_ok"] += int(consistente)

        results.append({
            "codigo_item_ref": ex["meta"]["codigo_item"],
            "ano": ano,
            **flags,
            "consistencia_resposta_correta": consistente,
            "sugestao_resposta_correta": sugestao,
            "difficulty_pedida": ex["meta"].get("dificuldade"),
            "difficulty_emitida": questao.get("difficulty") if questao else None,
            "difficulty_aderente": aderente,
        })

    n = len(results)
    pct = lambda key: 100 * sum(r[key] for r in results) / n
    ppl = reference_perplexity(model, tokenizer, examples)

    from collections import Counter
    report = {
        "modelo": model_path,
        "conjunto_avaliacao": str(args.val),
        "num_amostras": n,
        "estrutura": {
            "json_valido_pct": round(pct("json_valido"), 1),
            "wrapper_valido_pct": round(pct("wrapper_valido"), 1),
            "schema_completo_pct": round(pct("schema_completo"), 1),
            "resposta_valida_pct": round(pct("resposta_valida"), 1),
            "alternativas_distintas_pct": round(pct("alternativas_distintas"), 1),
            "difficulty_valida_pct": round(pct("difficulty_valida"), 1),
            "distribuicao_respostas_corretas": dict(Counter(respostas_corretas)),
            "mencoes_figura_pct": round(100 * image_mentions / n, 1),
            "consistencia_resposta_correta_pct": (
                round(100 * consistencia_ok / consistencia_verificavel, 1)
                if consistencia_verificavel else None
            ),
            "consistencia_verificavel_n": consistencia_verificavel,
            "consistencia_nota": (
                "% das respostas em que resposta_correta bate com a conta resolvida "
                "em resolucao_passo_a_passo; medido só sobre as N amostras com uma "
                "expressão aritmética 'a op b = r' reconhecível no texto."
            ),
            "difficulty_aderente_pct": (
                round(100 * dif_aderente / dif_avaliavel, 1) if dif_avaliavel else None
            ),
            "difficulty_avaliavel_n": dif_avaliavel,
            "gabarito_letra_mais_frequente_pct": (
                round(100 * max(Counter(respostas_corretas).values()) / len(respostas_corretas), 1)
                if respostas_corretas else None
            ),
        },
        "por_ano": {
            ano: {
                "n": v["n"],
                "estrutura_ok_pct": round(100 * v["estrutura_ok"] / v["n"], 1),
                "difficulty_aderente_pct": round(100 * v["dif_ok"] / v["n"], 1),
                "consistencia_pct": (
                    round(100 * v["consist_ok"] / v["consist_verif"], 1)
                    if v["consist_verif"] else None
                ),
                "consistencia_verificavel_n": v["consist_verif"],
            }
            for ano, v in sorted(por_ano.items())
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

    report_path = Path(args.report)
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n===== Avaliação: {model_path} ({n} amostras) =====")
    for section in ("estrutura", "por_ano", "linguagem", "velocidade_gpu"):
        print(f"[{section}]")
        for k, v in report[section].items():
            if k not in ("nota", "consistencia_nota"):
                print(f"  {k}: {v}")
    print(f"\nRelatório completo: {report_path}")


if __name__ == "__main__":
    main()
