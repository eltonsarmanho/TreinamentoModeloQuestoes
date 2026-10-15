"""Fine-tuning do Qwen3-1.7B com QLoRA 4-bit (Unsloth + TRL SFTTrainer).

Calibrado para RTX 3060 Laptop 6GB (Ampere/bf16): adaptadores LoRA r=16 (alpha=32,
regra alpha=2r) sobre o modelo base congelado em NF4 4-bit, loss apenas nos tokens
do assistant e early stopping por eval_loss. Hiperparâmetros conferidos contra a
literatura de referência (LoRA Without Regret, Thinking Machines/Schulman 2025-09):
r/alpha, LR 2e-4 (padrão QLoRA), batch efetivo 16 (<32), target_modules em todas as
projeções lineares (attention + MLP) já seguem a receita; nada disso mudou em 2026-10-13.

MAX_SEQ_LENGTH = 1024. Testado 2048 em 2026-10-13 (treino v7c, mesma base v7b): o gate
multiseed reprovou (G2* 1→11 falhas, p=0,006; G3* 98,5%→88,3%, p=0,021; G7). Manter 1024.
17 exemplos (0,6%) passam de 1024 tokens e são truncados; o efeito disso não foi medido.

Se ocorrer OOM: reduzir BATCH_SIZE para 1 (compensando em GRAD_ACCUMULATION).

Uso (2026-10-14: --train e --val passaram a ser OBRIGATÓRIOS):
    python src/train.py --train data/train_curado_v7b.jsonl --val data/val_frozen_v1.jsonl \\
        --out outputs/lora_<v> --checkpoints outputs/checkpoints_<v> [--seed 43]

Motivo: sem argumentos o script treinava data/train.jsonl (963 exemplos), enquanto o ciclo
v7 usava train_curado_v7b.jsonl (2.846). Um comando que treina o corpus errado sem avisar
não pode ser o padrão. --out e --checkpoints continuam com padrão (outputs/lora,
outputs/checkpoints); use nomes versionados para não sobrescrever.

Cada treino grava MANIFESTO_TREINO.json (hash dos dados, versões, GPU, seed, truncagens,
melhor checkpoint) em --checkpoints (ao iniciar) e ao lado do LoRA (ao terminar).
Ver src/manifesto_treino.py.
"""

import argparse
from pathlib import Path


from dotenv import load_dotenv

# Carrega HF_TOKEN do .env antes de qualquer acesso ao HF Hub.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from unsloth import FastLanguageModel  # deve ser o primeiro import (patches)
from unsloth.chat_templates import train_on_responses_only

from datasets import load_dataset
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

import manifesto_treino  # stdlib apenas; ver src/manifesto_treino.py

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

BASE_MODEL = "unsloth/Qwen3-1.7B"
MAX_SEQ_LENGTH = 1024

# LoRA
LORA_RANK = 16
LORA_ALPHA = 32

# Treino (batch efetivo = BATCH_SIZE * GRAD_ACCUMULATION = 16)
BATCH_SIZE = 2
GRAD_ACCUMULATION = 8
LEARNING_RATE = 2e-4
EPOCHS = 3
SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True,
                        help="JSONL de treino (obrigatório: sem padrão, para não treinar o corpus errado)")
    parser.add_argument("--val", required=True,
                        help="JSONL de validação para eval_loss (obrigatório)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"seed de treino (padrão {SEED}); varie para medir a variância entre treinos")
    parser.add_argument("--out", default=str(OUTPUT_DIR / "lora"),
                        help="onde salvar os adaptadores LoRA")
    parser.add_argument("--checkpoints", default=str(OUTPUT_DIR / "checkpoints"))
    # 2026-10-14 (Fase 3.1): o Qwen3-4B zero-shot superou o 1.7B fine-tunado na amostra humana
    # (núcleo 70% x 50%, ciclo_v7/SPEC.md). Estes argumentos permitem treinar o 4B com a mesma
    # receita; os padrões reproduzem exatamente o treino do 1.7B.
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help=f"modelo base (padrão {BASE_MODEL}; ex.: unsloth/Qwen3-4B-unsloth-bnb-4bit)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="batch por passo; reduza no 4B (6 GB de VRAM) e compense em --grad-accum")
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUMULATION)
    args = parser.parse_args()
    seed = args.seed
    base_model, batch_size, grad_accum = args.base_model, args.batch_size, args.grad_accum

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    dataset = load_dataset(
        "json",
        data_files={"train": args.train, "val": args.val},
    )

    def format_chat(batch):
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            for messages in batch["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(
        format_chat, batched=True, remove_columns=["messages", "meta"]
    )

    # Manifesto de entrada (gravado ANTES do treino: existe mesmo se o treino cair).
    comprimentos = [len(tokenizer(t, add_special_tokens=False)["input_ids"])
                    for t in dataset["train"]["text"]]
    truncagem = manifesto_treino.contagem_truncadas(comprimentos, MAX_SEQ_LENGTH)
    print(f"Exemplos de treino acima de {MAX_SEQ_LENGTH} tokens (truncados): "
          f"{truncagem['acima_do_limite']}/{truncagem['exemplos']} (maior: {truncagem['maior']})")
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    manifesto = manifesto_treino.criar(
        ROOT,
        {"train": args.train, "val": args.val, "out": args.out, "checkpoints": args.checkpoints},
        {"base_model": base_model, "max_seq_length": MAX_SEQ_LENGTH, "lora_rank": LORA_RANK,
         "lora_alpha": LORA_ALPHA, "batch_size": batch_size, "grad_accumulation": grad_accum,
         "learning_rate": LEARNING_RATE, "epochs": EPOCHS, "seed": seed,
         "scheduler": "cosine", "warmup_ratio": 0.05, "weight_decay": 0.01,
         "optim": "adamw_8bit", "bf16": True, "early_stopping_patience": 2,
         "full_determinism": False},
        gpu, truncagem)
    manifesto_treino.gravar(manifesto, args.checkpoints)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        args=SFTConfig(
            output_dir=args.checkpoints,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            optim="adamw_8bit",
            bf16=True,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=seed,
            report_to="none",
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Loss apenas nos tokens de resposta do assistant (marcadores do Qwen3).
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    result = trainer.train()

    lora_dir = Path(args.out)
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    manifesto["resultado"] = manifesto_treino.resumo_resultado(
        trainer.state.log_history, result.metrics.get("train_loss"),
        trainer.state.best_model_checkpoint, trainer.state.best_metric)
    manifesto_treino.gravar(manifesto, args.checkpoints)
    destino = manifesto_treino.gravar(manifesto, lora_dir)
    print(f"Manifesto do treino: {destino}")

    print("\n===== Treino concluído =====")
    print(f"train_loss final: {result.metrics.get('train_loss', float('nan')):.4f}")
    for log in trainer.state.log_history:
        if "eval_loss" in log:
            print(f"época {log.get('epoch', 0):.1f}: eval_loss={log['eval_loss']:.4f}")
    print(f"Adaptadores LoRA salvos em: {lora_dir}")
    print("Próximos passos: python src/evaluate.py && python src/export_gguf.py")


if __name__ == "__main__":
    main()
