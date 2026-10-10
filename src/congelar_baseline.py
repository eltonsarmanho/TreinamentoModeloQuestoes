"""Congela o baseline de produção num manifesto com SHA-256 (P0-1).

Por que existe: toda melhoria seguinte (verificador, sufixo, base saneada, treino
v4) só pode ser julgada contra um ponto de referência que não muda sem aviso. O
promover_checkpoint.py já trava o sha256 do GGUF dentro de um relatório, mas
nada registrava, num lugar só, o GGUF + LoRA + datasets + banco + prompt +
gramática + hiperparâmetros + configuração de inferência + métricas de
referência que formam a "release" em produção.

Duas operações:
    python src/congelar_baseline.py --nome baseline_v3        # grava baseline_v3/MANIFESTO.json
    python src/congelar_baseline.py --verificar baseline_v3   # compara o disco com o manifesto

O manifesto só guarda hashes, tamanhos, contagens e números de relatórios — nunca
o conteúdo dos arquivos e nunca o .env. Fica FORA de outputs/ e data/ (ambos no
.gitignore) para poder ser commitado. `--verificar` sai com código 1 se qualquer
artefato congelado mudou (ou sumiu), e é o que um gate futuro deve chamar antes
de aceitar uma comparação.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

# Arquivos que definem a release. Caminhos relativos à raiz do repositório.
ARTEFATOS = {
    "gguf": "outputs/gguf_v3_gguf/qwen3-1.7b.Q4_K_M.gguf",
    "lora_adapter": "outputs/lora_v3/adapter_model.safetensors",
    "lora_config": "outputs/lora_v3/adapter_config.json",
    "gramatica": "grammars/questao.gbnf",
    "banco_saeb": "DB/questoes.db",
}
DATASETS = {
    "treino": "data/train_curado_v3.jsonl",
    "val_congelado": "data/val_frozen_v1.jsonl",
    "val": "data/val.jsonl",
    "val_novos": "data/val_novos_v1.jsonl",
    "taxonomia": "data/taxonomia_subtemas.json",
}
# Relatórios que sustentam a promoção (métricas de referência).
RELATORIOS = {
    "veredito_modelo_v3": "outputs/relatorios/VEREDITO_v3_multiseed.json",
    "veredito_inferencia_server": "outputs/relatorios/VEREDITO_inferencia_server_g6agregado.json",
    "eval_v3_s0": "outputs/relatorios/eval_v3_gguf.json",
    "eval_v3_s1": "outputs/relatorios/eval_v3_s1_gguf.json",
    "eval_v3_s2": "outputs/relatorios/eval_v3_s2_gguf.json",
}


def sha256_arquivo(caminho, bloco=1 << 20):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for b in iter(lambda: f.read(bloco), b""):
            h.update(b)
    return h.hexdigest()


def sha256_texto(texto):
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _entrada(raiz, rel):
    p = Path(raiz) / rel
    if not p.exists():
        return {"caminho": rel, "presente": False}
    e = {"caminho": rel, "presente": True, "sha256": sha256_arquivo(p), "bytes": p.stat().st_size}
    if p.suffix == ".jsonl":
        with open(p, "rb") as f:
            e["linhas"] = sum(1 for _ in f)
    return e


def _git(raiz, *args):
    try:
        return subprocess.run(["git", "-C", str(raiz), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 — sem git não impede o congelamento
        return None


def _constantes_treino(raiz):
    """Hiperparâmetros lidos do código-fonte de train.py (sem importar unsloth)."""
    src = (Path(raiz) / "src" / "train.py").read_text(encoding="utf-8")
    out = {}
    for nome in ("BASE_MODEL", "MAX_SEQ_LENGTH", "LORA_RANK", "LORA_ALPHA", "BATCH_SIZE",
                 "GRAD_ACCUMULATION", "LEARNING_RATE", "EPOCHS", "SEED"):
        m = re.search(rf"^{nome}\s*=\s*(.+?)\s*(?:#.*)?$", src, re.M)
        if m:
            out[nome] = m.group(1).strip().strip('"')
    return out


def _metricas_referencia(raiz):
    """Números do veredito multiseed que promoveu o modelo (se o arquivo existir)."""
    p = Path(raiz) / RELATORIOS["veredito_modelo_v3"]
    if not p.exists():
        return None
    v = json.loads(p.read_text(encoding="utf-8"))
    return {"decisao": v.get("decisao"), "perfil": v.get("perfil"),
            "agregado_candidato": v.get("agregado_candidato"),
            "agregado_baseline": v.get("agregado_baseline")}


def montar_manifesto(raiz=ROOT, nome="baseline_v3"):
    """Dict do manifesto. Não escreve nada em disco."""
    raiz = Path(raiz)
    import test_model as tm
    from extract_data import SYSTEM_PROMPT, USER_TEMPLATE
    import diversidade as dv

    sujo = _git(raiz, "status", "--porcelain", "--untracked-files=no")
    return {
        "nome": nome,
        "congelado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        "codigo": {"commit": _git(raiz, "rev-parse", "HEAD"),
                   "arvore_com_alteracoes": bool(sujo),
                   "arquivos_alterados": sujo.splitlines() if sujo else []},
        "artefatos": {k: _entrada(raiz, v) for k, v in ARTEFATOS.items()},
        "datasets": {k: _entrada(raiz, v) for k, v in DATASETS.items()},
        "prompt": {
            "system_prompt_sha256": sha256_texto(SYSTEM_PROMPT),
            "user_template_sha256": sha256_texto(USER_TEMPLATE),
            "user_template": USER_TEMPLATE,
            "regua_diversidade_sha256": dv.versao_regua(),
        },
        "treino": _constantes_treino(raiz),
        "inferencia": {"motor": tm.MOTOR, "n_ctx": tm.N_CTX, "threads_padrao": tm.DEFAULT_THREADS,
                       "temperatura": tm.TEMPERATURE, "top_p": tm.TOP_P,
                       "max_tokens": tm.MAX_NEW_TOKENS, "np": 1, "cache_prompt": True},
        "relatorios": {k: _entrada(raiz, v) for k, v in RELATORIOS.items()},
        "metricas_referencia": _metricas_referencia(raiz),
    }


def _folhas(manifesto):
    """(secao, chave, entrada) de tudo que tem sha256 e portanto pode ser verificado."""
    for secao in ("artefatos", "datasets", "relatorios"):
        for chave, e in manifesto.get(secao, {}).items():
            yield secao, chave, e


def verificar(manifesto, raiz=ROOT):
    """Compara o disco com o manifesto. Retorna lista de divergências (vazia = íntegro)."""
    raiz = Path(raiz)
    problemas = []
    for secao, chave, e in _folhas(manifesto):
        if not e.get("presente"):
            continue  # não estava no congelamento; nada a comparar
        p = raiz / e["caminho"]
        if not p.exists():
            problemas.append(f"{secao}.{chave}: arquivo sumiu ({e['caminho']})")
        elif sha256_arquivo(p) != e["sha256"]:
            problemas.append(f"{secao}.{chave}: sha256 mudou ({e['caminho']})")
    import test_model as tm
    from extract_data import SYSTEM_PROMPT, USER_TEMPLATE
    pr = manifesto.get("prompt", {})
    if sha256_texto(SYSTEM_PROMPT) != pr.get("system_prompt_sha256"):
        problemas.append("prompt.system_prompt: mudou")
    if sha256_texto(USER_TEMPLATE) != pr.get("user_template_sha256"):
        problemas.append("prompt.user_template: mudou")
    inf = manifesto.get("inferencia", {})
    for campo, atual in (("n_ctx", tm.N_CTX), ("temperatura", tm.TEMPERATURE),
                         ("top_p", tm.TOP_P), ("max_tokens", tm.MAX_NEW_TOKENS)):
        if inf.get(campo) != atual:
            problemas.append(f"inferencia.{campo}: congelado={inf.get(campo)} atual={atual}")
    return problemas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nome", default="baseline_v3", help="pasta (na raiz) onde gravar MANIFESTO.json")
    ap.add_argument("--verificar", metavar="PASTA", help="verifica o disco contra PASTA/MANIFESTO.json")
    ap.add_argument("--sobrescrever", action="store_true",
                    help="permite regravar um manifesto existente (o baseline congelado não deve mudar)")
    args = ap.parse_args()

    if args.verificar:
        alvo = ROOT / args.verificar / "MANIFESTO.json"
        problemas = verificar(json.loads(alvo.read_text(encoding="utf-8")))
        if problemas:
            print(f"DIVERGE de {alvo.relative_to(ROOT)}:")
            for p in problemas:
                print("  -", p)
            sys.exit(1)
        print(f"OK: o disco confere com {alvo.relative_to(ROOT)}")
        return

    pasta = ROOT / args.nome
    destino = pasta / "MANIFESTO.json"
    if destino.exists() and not args.sobrescrever:
        sys.exit(f"ABORTADO: {destino.relative_to(ROOT)} já existe. Um baseline congelado não é "
                 f"regravado; use outro --nome ou --sobrescrever de propósito.")
    pasta.mkdir(exist_ok=True)
    m = montar_manifesto(ROOT, args.nome)
    destino.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ausentes = [f"{s}.{k}" for s, k, e in _folhas(m) if not e.get("presente")]
    print(f"Gravado {destino.relative_to(ROOT)}")
    print(f"  gguf sha256: {m['artefatos']['gguf'].get('sha256', 'AUSENTE')[:16]}…")
    print(f"  commit: {m['codigo']['commit'][:10] if m['codigo']['commit'] else '?'}"
          f" (alterações não commitadas: {m['codigo']['arvore_com_alteracoes']})")
    if ausentes:
        print("  AUSENTES:", ", ".join(ausentes))


if __name__ == "__main__":
    main()
