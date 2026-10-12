"""Manifesto de proveniência do ciclo v6 (etapa 9/10). Só lê; grava ciclo_v6/MANIFESTO_<fase>.json.

Registra: commit e arquivos alterados; sha256 das entradas imutáveis (e confere com
ciclo_v6/ENTRADAS_IMUTAVEIS.json); versões dos verificadores e prompts (sha256 do código de
schema_utils, diversidade, verificador_geometria; VERSAO_PROMPTS/JUIZES/FILTROS de
agentes_questoes; régua de diversidade); sha256 e contagens de todos os relatórios gerados em
outputs/melhoria_v6/; e, se existir, da candidata (data/train_curado_v6.jsonl).

    venv/bin/python src/manifesto_v6.py --fase auditoria
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from congelar_baseline import _entrada, _git, sha256_arquivo  # noqa: E402

TRAVA = ROOT / "ciclo_v6" / "ENTRADAS_IMUTAVEIS.json"
CODIGO_VERIFICADORES = ("src/schema_utils.py", "src/diversidade.py", "src/verificador_geometria.py",
                        "src/agentes_questoes.py", "src/perfil_base.py", "src/auditar_distratores.py",
                        "src/triar_resolucoes.py", "src/melhorar_resolucoes.py", "src/sabia_v6.py",
                        "src/planejar_injecao_v6.py", "src/auditar_gabaritos_base.py", "grammars/questao.gbnf",
                        "data/taxonomia_subtemas.json", "ciclo_v6/precos_maritaca.json")


def montar(fase):
    import agentes_questoes as aq
    import diversidade as dv
    trava = json.loads(TRAVA.read_text(encoding="utf-8"))["arquivos"]
    entradas = {}
    for rel, sha in trava.items():
        p = ROOT / rel
        atual = sha256_arquivo(p) if p.exists() else None
        entradas[rel] = {"sha256_travado": sha, "sha256_atual": atual, "intacto": atual == sha}
    sujo = _git(ROOT, "status", "--porcelain")
    relatorios = {}
    for p in sorted((ROOT / "outputs" / "melhoria_v6").glob("*")):
        if p.is_file():
            relatorios[p.name] = _entrada(ROOT, str(p.relative_to(ROOT)))
    cand = ROOT / "data" / "train_curado_v6.jsonl"
    return {
        "fase": fase, "gerado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        "codigo": {"commit": _git(ROOT, "rev-parse", "HEAD"), "alteracoes_nao_commitadas": sujo.splitlines() if sujo else []},
        "entradas_imutaveis": entradas, "todas_intactas": all(e["intacto"] for e in entradas.values()),
        "versoes": {"VERSAO_PROMPTS": aq.VERSAO_PROMPTS, "VERSAO_JUIZES": aq.VERSAO_JUIZES,
                    "VERSAO_FILTROS": aq.VERSAO_FILTROS, "regua_diversidade": dv.versao_regua(),
                    "codigo_sha256": {rel: sha256_arquivo(ROOT / rel) for rel in CODIGO_VERIFICADORES if (ROOT / rel).exists()}},
        "relatorios": relatorios,
        "candidata": _entrada(ROOT, str(cand.relative_to(ROOT))) if cand.exists() else {"presente": False},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fase", default="auditoria")
    args = ap.parse_args()
    m = montar(args.fase)
    destino = ROOT / "ciclo_v6" / f"MANIFESTO_{args.fase}.json"
    destino.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{destino.relative_to(ROOT)}: entradas intactas={m['todas_intactas']}, relatórios={len(m['relatorios'])}, "
          f"candidata={'sim' if m['candidata'].get('presente') else 'não'}")
    if not m["todas_intactas"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
