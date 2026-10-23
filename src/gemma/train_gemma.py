"""Fine-tuning do Gemma 3 1B-it com QLoRA 4-bit (Unsloth FastModel + TRL SFTTrainer).

Variante Gemma do src/train.py (que continua sendo o treino da linha Qwen). Mesma base de
dados, mesmo val congelado, mesmo manifesto; o modelo Gemma só se compara com Gemma no gate
(ver Doc/PIPELINE_GEMMA3.md e CLAUDE.md).

Por que o 1B e não o Gemma 4 E2B: o checkpoint do E2B é `Gemma4ForConditionalGeneration`, e o
kernel de cross-entropy eficiente do Unsloth só se aplica a classes `*ForCausalLM` — o E2B ficava
de fora e estourava os 5,67 GB utilizáveis da GPU mesmo em QLoRA 4-bit. O Gemma 3 1B-it é
`Gemma3ForCausalLM` (puro texto, sem torres de visão/áudio) e pega o caminho otimizado.

Receita (fontes e justificativa em Doc/PIPELINE_GEMMA3.md, seção "Hiperparâmetros"):
  - QLoRA NF4 4-bit: cabe com folga em 6 GB (1B parâmetros).
  - Só camadas de linguagem, attention + MLP (LoRA só em attention rende menos, LoRA Without
    Regret 2025).
  - r=16, alpha=32, dropout 0; LR 2e-4 (~10x o LR de full FT), batch efetivo 16 (=1x16; LoRA
    perde mais com batch grande), cosine, warmup 5%, max_grad_norm 0,3 (QLoRA, Dettmers 2023).
  - 2 épocas (não 3): nas 8 runs Qwen em v7b a melhor época foi sempre a 1 ou a 2 e a 3 sempre
    piorou o eval_loss (ciclo_v7/SPEC.md, Fase 2). Melhor checkpoint por eval_loss.
  - bf16 (a GPU é Ampere, suporta).
  - Loss só nos tokens do modelo (<start_of_turn>model\\n); texto sem <bos> (o SFTTrainer põe um).
  - O Gemma 3 não tem turno de `system` próprio: o template funde o conteúdo do `system` como
    prefixo do primeiro turno de `user` (sem erro, diferente do Gemma 2, que rejeita `system`).

Uso:
    python src/gemma/train_gemma.py --checar-formato --train data/train_curado_v7b_fix1.jsonl \\
        --val data/val_frozen_v1.jsonl                     # só renderiza e valida, sem GPU
    python src/gemma/train_gemma.py --train data/train_curado_v7b_fix1.jsonl \\
        --val data/val_frozen_v1.jsonl --out outputs/gemma3/lora_s42 \\
        --checkpoints outputs/gemma3/checkpoints_s42 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "gemma")]

import gemma_formato  # noqa: E402  (stdlib)
import manifesto_treino  # noqa: E402  (stdlib)

FAMILIA = "gemma3"
BASE_MODEL = "unsloth/gemma-3-1b-it"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16
LORA_ALPHA = 32
BATCH_SIZE = 1
GRAD_ACCUMULATION = 16
LEARNING_RATE = 2e-4
EPOCHS = 2
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 0.3
SEED = 42


def _ler_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(linha) for linha in f if linha.strip()]


def _tokenizer_texto(tok):
    """O Gemma 3 1B é puro texto: FastModel devolve o tokenizer direto (sem processor)."""
    return getattr(tok, "tokenizer", tok)


def checar_formato(tokenizer, exemplos, max_seq, limite_erros=5):
    """Renderiza todos os exemplos e devolve (textos, comprimentos, erros)."""
    tok = _tokenizer_texto(tokenizer)
    renderizador = tok if getattr(tok, "chat_template", None) else tokenizer
    bos = tok.bos_token or gemma_formato.BOS
    textos, comprimentos, erros = [], [], []
    for i, ex in enumerate(exemplos):
        texto, prompt = gemma_formato.renderizar(renderizador, ex["messages"], bos)
        resposta = ex["messages"][-1]["content"]
        for e in gemma_formato.problemas_formato(texto, prompt, resposta):
            erros.append(f"exemplo {i} ({ex.get('meta', {}).get('codigo_item')}): {e}")
        textos.append(texto)
        comprimentos.append(len(tok(texto, add_special_tokens=True)["input_ids"]))
    if erros:
        print(f"{len(erros)} defeito(s) de formato; primeiros:")
        for e in erros[:limite_erros]:
            print("  -", e)
    trunc = manifesto_treino.contagem_truncadas(comprimentos, max_seq)
    print(f"Exemplos acima de {max_seq} tokens (truncados): {trunc['acima_do_limite']}/"
          f"{trunc['exemplos']} (maior: {trunc['maior']})")
    return textos, comprimentos, erros


def _hiperparametros(args):
    return {"familia": FAMILIA, "base_model": args.base_model, "loader": "unsloth.FastModel",
            "max_seq_length": args.max_seq, "lora_rank": args.rank, "lora_alpha": args.alpha,
            "lora_dropout": 0.0, "camadas": "linguagem (attention + MLP)",
            "batch_size": args.batch_size, "grad_accumulation": args.grad_accum,
            "learning_rate": args.lr, "epochs": args.epochs, "seed": args.seed,
            "scheduler": "cosine", "warmup_ratio": WARMUP_RATIO, "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM, "optim": "adamw_8bit", "bf16": True,
            "template": args.template, "load_in_4bit": True,
            "early_stopping_patience": 1, "full_determinism": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "gemma3" / "lora"))
    parser.add_argument("--checkpoints", default=str(ROOT / "outputs" / "gemma3" / "checkpoints"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--max-seq", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--rank", type=int, default=LORA_RANK)
    parser.add_argument("--alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUMULATION)
    parser.add_argument("--template", choices=("nativo", "unsloth"), default="nativo",
                        help="nativo = chat template do próprio checkpoint (o mesmo que vai para o "
                             ".gguf do baseline zero-shot); unsloth = get_chat_template('gemma-3')")
    parser.add_argument("--checar-formato", action="store_true",
                        help="só carrega o tokenizer, renderiza e valida os exemplos; não treina")
    args = parser.parse_args()

    treino, val = _ler_jsonl(args.train), _ler_jsonl(args.val)

    if args.checar_formato:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if args.template == "unsloth":
            from unsloth.chat_templates import get_chat_template
            tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        _, _, erros = checar_formato(tokenizer, treino + val, args.max_seq)
        texto, _ = gemma_formato.renderizar(tokenizer, treino[0]["messages"],
                                            tokenizer.bos_token or gemma_formato.BOS)
        print("\n----- exemplo 0 renderizado (sem <bos>) -----\n" + texto)
        raise SystemExit(1 if erros else 0)

    from unsloth import FastModel  # primeiro import pesado (patches)
    from unsloth.chat_templates import train_on_responses_only
    import torch
    from datasets import Dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base_model, max_seq_length=args.max_seq,
        load_in_4bit=True, full_finetuning=False, dtype=None)
    if args.template == "unsloth":
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=args.seed)

    textos_treino, comprimentos, erros = checar_formato(tokenizer, treino, args.max_seq)
    textos_val, _, erros_val = checar_formato(tokenizer, val, args.max_seq)
    if erros or erros_val:
        raise SystemExit("ABORTADO: defeitos de formato (rode --checar-formato para detalhes)")
    dataset_treino = Dataset.from_dict({"text": textos_treino})
    dataset_val = Dataset.from_dict({"text": textos_val})

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    manifesto = manifesto_treino.criar(
        ROOT, {"train": args.train, "val": args.val, "out": args.out, "checkpoints": args.checkpoints},
        _hiperparametros(args), gpu,
        manifesto_treino.contagem_truncadas(comprimentos, args.max_seq))
    manifesto_treino.gravar(manifesto, args.checkpoints)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset_treino,
        eval_dataset=dataset_val,
        args=SFTConfig(
            output_dir=args.checkpoints,
            dataset_text_field="text",
            max_seq_length=args.max_seq,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=MAX_GRAD_NORM,
            optim="adamw_8bit",
            bf16=True,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=args.seed,
            report_to="none",
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=gemma_formato.MARCADOR_USUARIO,
        response_part=gemma_formato.MARCADOR_MODELO,
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

    print("\n===== Treino Gemma 3 concluído =====")
    print(f"train_loss final: {result.metrics.get('train_loss', float('nan')):.4f}")
    for log in trainer.state.log_history:
        if "eval_loss" in log:
            print(f"época {log.get('epoch', 0):.1f}: eval_loss={log['eval_loss']:.4f}")
    print(f"LoRA: {lora_dir}   manifesto: {destino}")
    print(f"Próximo: python src/gemma/export_gguf_gemma.py --lora {lora_dir} --out outputs/gemma3/gguf_<rotulo>")


if __name__ == "__main__":
    main()
