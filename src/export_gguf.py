"""Exporta o modelo fine-tuned para GGUF (llama.cpp) para uso offline no mobile.

Faz o merge dos adaptadores LoRA no modelo base em fp16 e quantiza para
Q4_K_M (~1.1GB — melhor equilíbrio qualidade/velocidade para mobile).
O Unsloth baixa e compila o llama.cpp automaticamente na primeira execução.

Uso:
    python src/export_gguf.py                # Q4_K_M (padrão para o app)
    python src/export_gguf.py --also-q8      # gera também Q8_0 para comparação

Benchmark local (proxy do desempenho mobile, roda em CPU):
    llama-bench -m outputs/gguf/*Q4_K_M.gguf -p 256 -n 256

Teste de geração:
    llama-cli -m outputs/gguf/*Q4_K_M.gguf --temp 0.7 --top-p 0.8 \
        -p "Gere uma questão de matemática. Ano: 5º ano. ..."

Dica para o app: use grammar GBNF (json.gbnf do llama.cpp) na inferência para
GARANTIR JSON válido na saída, além de --no-think/template non-thinking.
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

# Carrega HF_TOKEN do .env antes de qualquer acesso ao HF Hub.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from unsloth import FastLanguageModel  # deve ser o primeiro import (patches)

ROOT = Path(__file__).resolve().parent.parent
LORA_DIR = ROOT / "outputs" / "lora"
GGUF_DIR = ROOT / "outputs" / "gguf"

MAX_SEQ_LENGTH = 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--also-q8", action="store_true",
        help="gera também Q8_0 (~1.8GB) para comparação de qualidade",
    )
    args = parser.parse_args()

    if not LORA_DIR.exists():
        raise SystemExit(f"{LORA_DIR} não existe — rode primeiro: python src/train.py")

    print(f"Carregando adaptadores LoRA de {LORA_DIR} (merge em fp16)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(LORA_DIR),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,  # merge precisa dos pesos em fp16
    )

    methods = ["q4_k_m"] + (["q8_0"] if args.also_q8 else [])
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    for method in methods:
        print(f"\nExportando GGUF {method.upper()}...")
        model.save_pretrained_gguf(
            str(GGUF_DIR), tokenizer, quantization_method=method
        )

    print("\n===== Exportação concluída =====")
    for f in sorted(GGUF_DIR.glob("*.gguf")):
        print(f"  {f.name}: {f.stat().st_size / 1e9:.2f} GB")
    print(
        "\nIntegração mobile: carregue o .gguf com llama.cpp (Android NDK/iOS), "
        "LLMFarm, ChatterUI ou binding nativo. Use o chat template do Qwen3 em "
        "modo non-thinking e grammar GBNF para JSON garantido."
    )


if __name__ == "__main__":
    main()
