"""Confere que o prompt que o llama-server monta a partir do .gguf é o mesmo que o treino viu.

O treino (train_gemma.py) renderiza com o chat template do tokenizer HF; a avaliação e o app
(tests/test_model.py) mandam system+user para o llama-server com `--jinja -rea off`, que usa o
template EMBUTIDO no .gguf. Se os dois divergirem, o modelo é avaliado num formato que não
treinou e o gate mede outra coisa. Rode antes do gate, nos DOIS .gguf (baseline zero-shot e
candidato).

Uso:
    python src/gemma/verificar_template_gemma.py --gguf outputs/gemma3/gguf_s42/gemma-3-1b-it-ft.Q4_K_M.gguf
    python src/gemma/verificar_template_gemma.py --gguf ... --tokenizer outputs/gemma3/lora_s42

Saída 0 = idêntico nos N prompts; 1 = divergência (imprime a primeira diferença).
"""

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "gemma")]

import gemma_formato  # noqa: E402
import gguf_meta  # noqa: E402

LLAMA_SERVER = Path.home() / ".unsloth" / "llama.cpp" / "llama-server"
VAL = ROOT / "data" / "val_frozen_v1.jsonl"


def _servidor(binario, gguf, n_ctx):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        porta = s.getsockname()[1]
    # Mesmas flags de template do tests/test_model.py (_Servidor.garantir).
    proc = subprocess.Popen([str(binario), "-m", str(gguf), "--jinja", "-rea", "off",
                             "-c", str(n_ctx), "-np", "1", "--host", "127.0.0.1",
                             "--port", str(porta), "--log-disable"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{porta}"
    inicio = time.time()
    while time.time() - inicio < 180:
        if proc.poll() is not None:
            raise SystemExit(f"llama-server encerrou ao iniciar (código {proc.returncode})")
        try:
            urllib.request.urlopen(url + "/health", timeout=1).read()
            return proc, url
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.2)
    proc.terminate()
    raise SystemExit("llama-server não respondeu em 180s")


def _prompt_servidor(url, messages):
    req = urllib.request.Request(url + "/apply-template", data=json.dumps({"messages": messages}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))["prompt"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--tokenizer", default="unsloth/gemma-3-1b-it",
                        help="tokenizer do treino (id HF ou a pasta do LoRA, que salva o tokenizer)")
    parser.add_argument("--val", default=str(VAL))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--llama-server", default=str(LLAMA_SERVER))
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    arq = gguf_meta.arquitetura(args.gguf)
    if arq != "gemma3":
        raise SystemExit(f"ABORTADO: {args.gguf} tem arquitetura {arq!r}, não gemma3")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    bos = tok.bos_token or gemma_formato.BOS
    exemplos = [json.loads(linha) for linha in open(args.val, encoding="utf-8") if linha.strip()][:args.n]

    proc, url = _servidor(args.llama_server, args.gguf, args.n_ctx)
    try:
        for i, ex in enumerate(exemplos):
            msgs = ex["messages"][:-1]
            esperado = gemma_formato.remover_bos(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True), bos)
            obtido = gemma_formato.remover_bos(_prompt_servidor(url, msgs), bos)
            dif = gemma_formato.diferenca(esperado, obtido)
            if dif:
                print(f"DIVERGÊNCIA no prompt {i} ({ex['meta'].get('codigo_item')}), posição {dif['posicao']}")
                print(f"  treino (tokenizer): {dif['a']!r}")
                print(f"  .gguf (servidor):   {dif['b']!r}")
                return 1
    finally:
        proc.terminate()
    print(f"OK: {len(exemplos)} prompts idênticos entre o tokenizer do treino e o template do .gguf "
          f"({Path(args.gguf).name}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
