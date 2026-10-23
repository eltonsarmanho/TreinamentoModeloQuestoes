"""Exporta o Gemma 3 1B para GGUF Q4_K_M (variante Gemma do export_gguf.py).

O Gemma 3 1B é `Gemma3ForCausalLM` (puro texto), então o `save_pretrained_gguf` do Unsloth
funciona direto, igual ao `src/export_gguf.py` do Qwen — sem o contorno de merge manual +
`convert_hf_to_gguf.py` que o Gemma 4 (`ForConditionalGeneration`, só Q8_0/BF16/F16) exigia.

`--base-only` exporta o modelo base SEM LoRA (16-bit, sem adaptador): é o baseline zero-shot
do gate Gemma. Mesmo conversor e quantização do candidato, para os dois diferirem só no adaptador.

Uso:
    python src/gemma/export_gguf_gemma.py --base-only --out outputs/gemma3/gguf_zeroshot
    python src/gemma/export_gguf_gemma.py --lora outputs/gemma3/lora_s42 --out outputs/gemma3/gguf_s42

Grava <out>/PROCEDENCIA.json (origem, sha256, arquitetura lida do .gguf).
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "gemma")]

import gguf_meta  # noqa: E402
import manifesto_treino  # noqa: E402

BASE_MODEL = "unsloth/gemma-3-1b-it"
FAMILIA = "gemma3"
DISCO_MIN_GB = 8  # modelo base 16-bit (~2 GB) + GGUF (quant + folga)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    origem = parser.add_mutually_exclusive_group(required=True)
    origem.add_argument("--lora", help="pasta do adaptador (saída de train_gemma.py)")
    origem.add_argument("--base-only", action="store_true",
                        help="exporta o base sem LoRA (baseline zero-shot do gate Gemma)")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quant", default="q4_k_m")
    parser.add_argument("--also-q8", action="store_true")
    parser.add_argument("--max-seq", type=int, default=1024)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    livre_gb = shutil.disk_usage(out).free / 1e9
    if livre_gb < DISCO_MIN_GB:
        raise SystemExit(f"ABORTADO: {livre_gb:.1f} GB livres, export pede >= {DISCO_MIN_GB} GB")

    from unsloth import FastModel  # primeiro import pesado (patches)

    if args.base_only:
        model_name, papel, lora_dir = args.base_model, "baseline zero-shot", None
    else:
        lora_dir = Path(args.lora)
        if not (lora_dir / "adapter_config.json").exists():
            raise SystemExit(f"{lora_dir} não tem adapter_config.json")
        model_name, papel = str(lora_dir), "candidato fine-tunado"

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name, max_seq_length=args.max_seq, load_in_4bit=False)

    methods = [args.quant] + (["q8_0"] if args.also_q8 else [])
    model.save_pretrained_gguf(str(out), tokenizer, quantization_method=methods)

    artefatos = sorted(out.glob("*.gguf")) + sorted((out.parent / f"{out.name}_gguf").glob("*.gguf"))
    if not artefatos:
        raise SystemExit("ABORTADO: nenhum .gguf foi gerado")
    for a in artefatos:
        arq = gguf_meta.arquitetura(a)
        if arq != FAMILIA:
            raise SystemExit(f"ABORTADO: {a} saiu com arquitetura {arq!r}, esperado {FAMILIA!r}")

    procedencia = {
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "familia_modelo": FAMILIA,
        "base_model": args.base_model,
        "lora": str(lora_dir) if lora_dir else None,
        "lora_manifesto": (manifesto_treino.descreve_arquivo(lora_dir / manifesto_treino.NOME_ARQUIVO)
                           if lora_dir else None),
        "papel": papel,
        "caminho": "unsloth.FastModel.save_pretrained_gguf",
        "git": manifesto_treino.git_info(ROOT),
        "artefatos": [{"arquivo": str(a), "bytes": a.stat().st_size,
                       "sha256": hashlib.sha256(a.read_bytes()).hexdigest(),
                       "arquitetura": gguf_meta.arquitetura(a)} for a in artefatos],
    }
    (out / "PROCEDENCIA.json").write_text(json.dumps(procedencia, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print("\n===== Exportação Gemma 3 concluída =====")
    for a in procedencia["artefatos"]:
        print(f"  {a['arquivo']}: {a['bytes'] / 1e9:.2f} GB  sha256={a['sha256'][:16]}  ({a['arquitetura']})")
    print(f"Procedência: {out / 'PROCEDENCIA.json'}")
    print("Próximo: python src/gemma/verificar_template_gemma.py --gguf <arquivo .gguf acima>")


if __name__ == "__main__":
    main()
