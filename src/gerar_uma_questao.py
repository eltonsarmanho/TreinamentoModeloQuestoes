"""Gera UMA questão por execução, com diversidade entre execuções (PC; o app faz o mesmo no celular).

Por que uma por execução: cada chamada recebe um slot (subtema, tipo de raciocínio, contexto) e o modelo
segue o slot pedido (ver Doc/GUIA_SUFIXO_APP.md). Pedir N questões numa chamada deixa a variedade por conta
da amostragem. Para a variedade valer ENTRE execuções, o script guarda os slots já usados num arquivo de
histórico e o planejador escolhe o subtema e o contexto MENOS usados para (ano, habilidade).

Passa pelo mesmo pipeline de produção (gramática GBNF, verificador aritmético, best-of-N, geometria) de
`tests/test_model.py`. Sobe um llama-server persistente só durante a execução.

Uso (na raiz do repositório):
  venv/bin/python src/gerar_uma_questao.py --ano 5º --habilidade H08 --dificuldade Moderado \\
      --gguf Qwen3-4B-Q4_K_M.gguf --lora lora_4b_v7b_s44.f16.gguf
  ... --saida questao.json            # grava o JSON da questão
  ... --historico meu_historico.json  # padrão: ~/.cache/questoes_historico.json
"""
import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tests"))

HISTORICO_PADRAO = Path.home() / ".cache" / "questoes_historico.json"
MAX_HISTORICO = 200  # por (ano, habilidade); o planejador só precisa dos usos recentes


def ler_historico(caminho):
    try:
        dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
        return dados if isinstance(dados, list) else []
    except (OSError, ValueError):
        return []


def do_par(historico, ano, habilidade):
    """O planejador conta TODO o histórico que recebe; filtra aqui para não misturar habilidades."""
    return [h for h in historico if h.get("ano") == ano and h.get("habilidade") == habilidade]


def registrar(historico, ano, habilidade, classificacao):
    """Acrescenta o slot realmente gerado e apara os mais antigos do mesmo par."""
    entrada = {"ano": ano, "habilidade": habilidade, "subtema": classificacao["subtema"],
               "contexto": classificacao.get("contexto")}
    novo = historico + [entrada]
    mesmo = [h for h in novo if h.get("ano") == ano and h.get("habilidade") == habilidade]
    excesso = len(mesmo) - MAX_HISTORICO
    if excesso > 0:
        descartar = {id(h) for h in mesmo[:excesso]}
        novo = [h for h in novo if id(h) not in descartar]
    return novo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ano", required=True, help="ex: 5º")
    ap.add_argument("--habilidade", required=True, help="ex: H08")
    ap.add_argument("--dificuldade", default="Moderado", choices=["Fácil", "Moderado", "Difícil"])
    ap.add_argument("--descricao", default="", help="padrão: descrição da habilidade na taxonomia")
    ap.add_argument("--gguf", required=True, help="Qwen3-4B-Q4_K_M.gguf")
    ap.add_argument("--lora", help="adaptador lora_4b_v7b_s44.f16.gguf")
    ap.add_argument("--threads", type=int, help="núcleos físicos / de alto desempenho (padrão: metade dos núcleos, até 8)")
    ap.add_argument("--sem-flash-attn", action="store_true")
    ap.add_argument("--historico", default=str(HISTORICO_PADRAO))
    ap.add_argument("--seed", type=int, help="fixa a seed (reprodutível); padrão: sorteada a cada execução")
    ap.add_argument("--saida", help="grava o JSON da questão neste arquivo")
    a = ap.parse_args()

    import test_model as tm
    import diversidade as dv
    from gerar_lote import gerar_lote_planejado

    tm.LORA_PATH = Path(a.lora).resolve() if a.lora else None
    tm.FLASH_ATTN = not a.sem_flash_attn
    descricao = a.descricao
    if not descricao:
        hab = dv.obter_habilidade(a.ano, a.habilidade)
        descricao = (hab or {}).get("descricoes", [""])[0]
    historico = ler_historico(a.historico)
    # o llama-server sobe na primeira chamada e é encerrado por atexit (tests/test_model.py)
    r = gerar_lote_planejado(
        tm.find_llama_cli(None), Path(a.gguf), a.ano, a.habilidade, descricao, a.dificuldade, 1,
        a.threads or tm.DEFAULT_THREADS, grammar=tm.GRAMMAR_PATH, retries=1, base_seed=a.seed,
        historico=do_par(historico, a.ano, a.habilidade))
    if not r["questoes"]:
        sys.exit("nenhuma questão válida nesta execução; rode de novo (outra seed é sorteada)")
    d = r["detalhes"][0]
    print(json.dumps(r["questoes"][0], ensure_ascii=False, indent=2))
    print(f"\n[status={d['status']} | {r['tempo_s']} s | seed={r['base_seed']} | "
          f"subtema={d['classificacao']['subtema'] if d['classificacao'] else 'n/d'}]", file=sys.stderr)
    if d["classificacao"]:
        destino = Path(a.historico)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(registrar(historico, a.ano, a.habilidade, d["classificacao"]),
                                      ensure_ascii=False), encoding="utf-8")
    if a.saida:
        Path(a.saida).write_text(json.dumps(r["questoes"][0], ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
