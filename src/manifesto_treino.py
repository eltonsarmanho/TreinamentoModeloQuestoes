"""Manifesto por execução de treino (Fase 1.1, 2026-10-14).

Motivo: v7b e v7d foram "o mesmo treino" e deram vereditos opostos, e o comando
documentado (`python src/train.py`) treinava `data/train.jsonl` (963 exemplos) enquanto
o ciclo usava `train_curado_v7b.jsonl` (2.846). Sem registro do que entrou em cada LoRA,
não dá para ligar um checkpoint aos dados e ao ambiente que o produziram.

Este módulo NÃO importa unsloth/torch no topo (os testes rodam sem GPU). Quem chama
(`train.py`) passa o que só ele sabe: nome da GPU, truncagens, resultado do treino.
O manifesto é gravado ao lado do LoRA como `MANIFESTO_TREINO.json` (o nome
`MANIFESTO.json` já é do baseline congelado, ver congelar_baseline.py).
"""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

NOME_ARQUIVO = "MANIFESTO_TREINO.json"
PACOTES = ("unsloth", "unsloth_zoo", "trl", "transformers", "peft", "torch",
           "bitsandbytes", "datasets", "accelerate", "xformers", "triton")


def sha256_arquivo(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _linhas(path):
    with open(path, "rb") as f:
        return sum(1 for linha in f if linha.strip())


def descreve_arquivo(path):
    p = Path(path)
    if not p.is_file():
        return {"caminho": str(path), "existe": False}
    return {"caminho": str(path), "existe": True, "sha256": sha256_arquivo(p),
            "linhas": _linhas(p), "bytes": p.stat().st_size}


def versoes_pacotes():
    out = {}
    for nome in PACOTES:
        try:
            out[nome] = metadata.version(nome)
        except metadata.PackageNotFoundError:
            out[nome] = None
    return out


def git_info(raiz):
    def _git(*args):
        try:
            return subprocess.run(["git", "-C", str(raiz), *args], capture_output=True,
                                  text=True, timeout=20, check=True).stdout.strip()
        except Exception:
            return None
    commit = _git("rev-parse", "HEAD")
    sujo = _git("status", "--porcelain", "--", "src", "tests")
    return {"commit": commit, "alteracoes_nao_commitadas_em_src_tests": bool(sujo) if sujo is not None else None}


def contagem_truncadas(comprimentos, max_seq_length):
    """comprimentos: tokens de cada exemplo já formatado com o chat template."""
    acima = [c for c in comprimentos if c > max_seq_length]
    return {"max_seq_length": max_seq_length, "exemplos": len(comprimentos),
            "acima_do_limite": len(acima), "maior": max(comprimentos) if comprimentos else 0}


def criar(raiz, args_cli, hiperparametros, gpu, truncagem):
    """Manifesto de ENTRADA (gravado antes do treino, para existir mesmo se o treino cair)."""
    raiz = Path(raiz)
    return {
        "versao_manifesto": 1,
        "iniciado_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(raiz),
        "ambiente": {"python": platform.python_version(), "so": platform.platform(),
                     "gpu": gpu, "pacotes": versoes_pacotes()},
        "dados": {"treino": descreve_arquivo(args_cli["train"]),
                  "validacao": descreve_arquivo(args_cli["val"]),
                  "banco_sqlite": descreve_arquivo(raiz / "DB" / "questoes.db")},
        "hiperparametros": hiperparametros,
        "truncagem": truncagem,
        "saidas": {"lora": str(args_cli["out"]), "checkpoints": str(args_cli["checkpoints"])},
        "resultado": None,
    }


def resumo_resultado(log_history, train_loss, melhor_checkpoint, melhor_eval_loss):
    evals = [{"epoca": round(x.get("epoch", 0), 2), "eval_loss": x["eval_loss"]}
             for x in log_history if "eval_loss" in x]
    return {"train_loss_final": train_loss, "eval_por_epoca": evals,
            "melhor_checkpoint": melhor_checkpoint, "melhor_eval_loss": melhor_eval_loss}


def gravar(manifesto, pasta):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / NOME_ARQUIVO
    destino.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino
