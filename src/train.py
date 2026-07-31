"""Fine-tuning do Qwen3-1.7B com QLoRA 4-bit (Unsloth + TRL SFTTrainer).

Calibrado para RTX 3060 Laptop 6GB (Ampere/bf16) e dataset pequeno (~300
exemplos): adaptadores LoRA r=16 sobre o modelo base congelado em NF4 4-bit,
loss apenas nos tokens do assistant e early stopping por eval_loss.

Se ocorrer OOM: reduzir MAX_SEQ_LENGTH para 768 ou BATCH_SIZE para 1
(compensando em GRAD_ACCUMULATION).

Uso:
    python src/train.py
"""

from pathlib import Path

from dotenv import load_dotenv

# Carrega HF_TOKEN do .env antes de qualquer acesso ao HF Hub.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from unsloth import FastLanguageModel  # deve ser o primeiro import (patches)
from unsloth.chat_templates import train_on_responses_only

from datasets import load_dataset
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

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
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
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
        random_state=SEED,
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "val": str(DATA_DIR / "val.jsonl"),
        },
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

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION,
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
            seed=SEED,
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

    lora_dir = OUTPUT_DIR / "lora"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    print("\n===== Treino concluído =====")
    print(f"train_loss final: {result.metrics.get('train_loss', float('nan')):.4f}")
    for log in trainer.state.log_history:
        if "eval_loss" in log:
            print(f"época {log.get('epoch', 0):.1f}: eval_loss={log['eval_loss']:.4f}")
    print(f"Adaptadores LoRA salvos em: {lora_dir}")
    print("Próximos passos: python src/evaluate.py && python src/export_gguf.py")


if __name__ == "__main__":
    main()
