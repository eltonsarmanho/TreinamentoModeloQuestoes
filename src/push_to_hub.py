"""Publica o modelo treinado num repositório privado do Hugging Face Hub.

Sobe os seguintes artefatos, cada um sob um subdiretório (ou raiz) do mesmo
repo de modelo:
  - README.md (raiz)  : model card completo — dataset, hiperparâmetros, métricas,
                         benchmarks e instruções de uso (LoRA e GGUF).
  - eval_report.json (raiz): relatório bruto de outputs/eval_report.json.
  - lora/             : adaptadores LoRA (outputs/lora, ~78MB) — para re-mesclar/
                         re-quantizar depois.
  - gguf/             : o .gguf Q4_K_M pronto para uso no app mobile via llama.cpp.
  - data/             : train.jsonl e val.jsonl (dataset usado no fine-tuning),
                         para reprodutibilidade e transparência com a comunidade.

Requer um HF_TOKEN com permissão de ESCRITA no .env (o token de leitura padrão
não consegue criar repositório nem subir arquivos). Gere um em:
  https://huggingface.co/settings/tokens -> "New token" -> role "Write"
  (ou fine-grained com "Write access to contents/settings of repos you own").

Uso:
    python src/push_to_hub.py --repo-id eltonsarmanho/qwen3-1.7b-questoes-matematica
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from huggingface_hub import HfApi  # noqa: E402  (após load_dotenv)

LORA_DIR = ROOT / "outputs" / "lora"
GGUF_DIR = ROOT / "outputs" / "gguf_gguf"
DATA_DIR = ROOT / "data"
MODEL_CARD = ROOT / "outputs" / "HF_MODEL_CARD.md"
EVAL_REPORT = ROOT / "outputs" / "eval_report.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id", required=True,
        help="ex: eltonsarmanho/qwen3-1.7b-questoes-matematica",
    )
    parser.add_argument(
        "--skip-lora", action="store_true", help="não subir os adaptadores LoRA"
    )
    parser.add_argument(
        "--skip-gguf", action="store_true", help="não subir o .gguf"
    )
    parser.add_argument(
        "--skip-data", action="store_true", help="não subir train.jsonl/val.jsonl"
    )
    parser.add_argument("--lora-dir", default=str(LORA_DIR),
                        help="pasta dos adaptadores LoRA (padrão: outputs/lora)")
    parser.add_argument("--gguf-dir", default=str(GGUF_DIR),
                        help="pasta do .gguf (padrão: outputs/gguf_gguf)")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="pasta a subir como data/ (padrão: data/ inteira; "
                             "para um ciclo específico, aponte para uma pasta com "
                             "só os jsonl relevantes)")
    parser.add_argument("--model-card", default=str(MODEL_CARD),
                        help="model card a subir como README.md")
    parser.add_argument("--eval-report", default=str(EVAL_REPORT),
                        help="relatório de avaliação a subir como eval_report.json")
    args = parser.parse_args()
    lora_dir, gguf_dir, data_dir = Path(args.lora_dir), Path(args.gguf_dir), Path(args.data_dir)
    model_card, eval_report = Path(args.model_card), Path(args.eval_report)

    if not gguf_dir.exists() and not args.skip_gguf:
        raise SystemExit(
            f"{gguf_dir} não existe — rode primeiro: python src/export_gguf.py"
        )
    if not lora_dir.exists() and not args.skip_lora:
        raise SystemExit(
            f"{lora_dir} não existe — rode primeiro: python src/train.py"
        )

    api = HfApi()
    print(f"Criando/verificando repositório privado: {args.repo_id}")
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=True)

    if model_card.exists():
        print("Enviando model card -> README.md")
        api.upload_file(
            repo_id=args.repo_id,
            path_or_fileobj=str(model_card),
            path_in_repo="README.md",
            commit_message="Adiciona model card com dataset, hiperparâmetros e métricas",
        )

    if eval_report.exists():
        print("Enviando eval_report.json")
        api.upload_file(
            repo_id=args.repo_id,
            path_or_fileobj=str(eval_report),
            path_in_repo="eval_report.json",
            commit_message="Adiciona relatório de avaliação",
        )

    if not args.skip_data and data_dir.exists():
        print(f"Enviando dataset de {data_dir} -> data/")
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(data_dir),
            path_in_repo="data",
            commit_message="Adiciona dataset de fine-tuning (train/val jsonl)",
        )

    if not args.skip_lora:
        print(f"Enviando adaptadores LoRA de {lora_dir} -> lora/")
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(lora_dir),
            path_in_repo="lora",
            commit_message="Adiciona adaptadores LoRA (fine-tuning Qwen3-1.7B)",
        )

    if not args.skip_gguf:
        print(f"Enviando GGUF de {gguf_dir} -> gguf/")
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(gguf_dir),
            path_in_repo="gguf",
            commit_message="Adiciona modelo quantizado GGUF Q4_K_M para uso mobile",
        )

    print(f"\nConcluído: https://huggingface.co/{args.repo_id}")
    print(
        "A equipe pode baixar apenas o .gguf com:\n"
        f"  huggingface-cli download {args.repo_id} --include 'gguf/*' "
        "--local-dir modelo_mobile --token <TOKEN_DE_LEITURA>"
    )


if __name__ == "__main__":
    main()
