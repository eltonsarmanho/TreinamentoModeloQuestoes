"""Avaliação de diversidade + qualidade de lotes de questões (FASE 7).

Roda um conjunto FIXO de prompts (data/prompts_diversidade.json) contra um
.gguf em dois modos, com as MESMAS seeds por prompt:

  atual     como o app gera hoje. Submodo configurável (--modo-atual):
              "unico"         uma chamada pedindo N questões (USER_TEMPLATE com
                              quantidade=N, como o app manda);
              "independente"  N chamadas de 1 questão, sem plano nem memória
                              entre elas (equivale ao laço --n do test_model).
  ajustado  gerar_lote.gerar_lote_planejado: plano de subtemas + regeneração
            guiada por violação de diversidade.

Por que medir qualidade JUNTO com diversidade: forçar subtema/contexto pode
aumentar a variedade às custas de JSON inválido, gabarito errado ou questões
fora da habilidade. Diversidade só vale se a qualidade não cair — é isso que o
gate de promoção (promover_checkpoint.gate_diversidade) cobra a partir deste
relatório.

Métricas por lote: json_valido, schema (todas as flags de check_structure por
questão), alternativas distintas, consistência (check_consistency ->
ok/nao_verificavel/inconsistente), difficulty correta, aderência à habilidade
(classificador léxico: subtema ∈ taxonomia da (ano, habilidade), não "outros"),
dependência de visual ausente, coverage_score, duplicate_rate,
semantic_similarity, structural_diversity, context_diversity,
diversity_score e tempo por questão.

Agregação: métricas de qualidade são contadas por QUESTÃO (somando lotes);
métricas de diversidade são a média por LOTE, só sobre lotes com N>=2 (um lote
de 1 questão não tem o que repetir e inflaria a média com 1.0).

Uso:
    python src/avaliar_diversidade.py --rotulo candidato --model outputs/gguf_gguf/qwen3-1.7b.Q4_K_M.gguf
    python src/avaliar_diversidade.py --rotulo teste --num-prompts 2
    python src/avaliar_diversidade.py --rotulo seco --dry-run   # sem llama.cpp
Saída: outputs/diversidade_<rotulo>.json e outputs/diversidade_<rotulo>.md
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# test_model.py mora em tests/ (é a CLI de inferência de produção, não um
# teste pytest), importada mais abaixo por GeradorReal e main().
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import diversidade as dv  # noqa: E402
from extract_data import USER_TEMPLATE  # noqa: E402
from schema_utils import (  # noqa: E402
    check_consistency, check_structure, depende_de_visual_ausente, extract_questoes, parse_json,
)

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = ROOT / "data" / "prompts_diversidade.json"
OUT_DIR = ROOT / "outputs"
MODOS_ATUAL = ("unico", "independente")
DIFFICULTY_MAP = {"Fácil": "EASY", "Moderado": "MEDIUM", "Difícil": "HARD"}
MAX_NEW_TOKENS_POR_QUESTAO = 512


# --------------------------------------------------------------------------
# Conjunto de prompts
# --------------------------------------------------------------------------

def carregar_prompts(path=PROMPTS_PATH, num_prompts=None):
    """Lista fixa de prompts. `num_prompts` corta do início — o 1º (9º H17 N=10)
    é obrigatório e por isso está sempre na rodada reduzida."""
    dados = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = dados["prompts"] if isinstance(dados, dict) else dados
    return prompts[:num_prompts] if num_prompts else prompts


# --------------------------------------------------------------------------
# Métricas de um lote
# --------------------------------------------------------------------------

def _flags_questao(q):
    """check_structure sobre UMA questão (envolta no wrapper), para contar por questão."""
    return check_structure({"questoes": [q]}, quantidade_esperada=1)


def metricas_lote(questoes, prompt, json_validos, chamadas, tempo_s, taxonomia=None, extra=None):
    """Mede um lote gerado. `json_validos`/`chamadas`: quantas saídas brutas do
    modelo parsearam como JSON (no modo único, 1 chamada; nos outros, N)."""
    ano, hab = prompt["ano"], prompt["habilidade"]
    n_ped = int(prompt["quantidade"])
    alvo_dif = DIFFICULTY_MAP.get(prompt.get("dificuldade"))
    entrada = dv.obter_habilidade(ano, hab, taxonomia)
    ids_sub = {s["id"] for s in entrada["subtemas"]} if entrada else set()
    k = len(ids_sub)

    qs = [q for q in questoes if isinstance(q, dict)]
    schema = alts = dif_ok = aderente = visual = 0
    cons = {"ok": 0, "nao_verificavel": 0, "inconsistente": 0}
    for q in qs:
        f = _flags_questao(q)
        if all(f[x] for x in ("schema_completo", "resposta_valida", "alternativas_distintas",
                              "difficulty_valida")):
            schema += 1
        alts += bool(f["alternativas_distintas"])
        dif_ok += bool(alvo_dif and q.get("difficulty") == alvo_dif)
        ok, _ = check_consistency(q)
        cons["ok" if ok is True else ("nao_verificavel" if ok is None else "inconsistente")] += 1
        sub = dv.classificar_questao(q, ano, hab, taxonomia)["subtema"]
        aderente += bool(sub != "outros" and sub in ids_sub)
        visual += bool(depende_de_visual_ausente(dv.texto_questao(q)))

    div = dv.diversity_score(qs, ano, hab, taxonomia=taxonomia) if qs else None
    m = {
        "id": prompt["id"], "ano": ano, "habilidade": hab, "dificuldade": prompt.get("dificuldade"),
        "quantidade_pedida": n_ped, "quantidade_gerada": len(qs), "k_subtemas": k,
        # K=1: toda questão cai no único subtema; aderência não discrimina.
        "aderencia_mensuravel": k >= 2,
        "chamadas": chamadas, "json_validos": json_validos,
        "schema_ok": schema, "alternativas_distintas": alts, "difficulty_correta": dif_ok,
        "consistencia": cons, "aderentes": aderente, "depende_de_visual": visual,
        "tempo_s": round(tempo_s, 3),
        "tempo_por_questao_s": round(tempo_s / max(len(qs), 1), 3),
    }
    if div:
        m.update({x: div[x] for x in ("coverage_score", "duplicate_rate", "semantic_similarity",
                                      "structural_diversity", "context_diversity",
                                      "raciocinio_diversity", "diversity_score",
                                      "subtema_distribution")})
    m["questoes"] = qs
    if extra:
        m.update(extra)
    return m


def _pct(a, b):
    return round(100.0 * a / b, 2) if b else None


def _media(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def agregar(lotes):
    """Agrega lotes de UM modo. Qualidade por questão; diversidade por lote (N>=2)."""
    nq = sum(l["quantidade_gerada"] for l in lotes)
    ped = sum(l["quantidade_pedida"] for l in lotes)
    cham = sum(l["chamadas"] for l in lotes)
    cons = {c: sum(l["consistencia"][c] for l in lotes) for c in ("ok", "nao_verificavel", "inconsistente")}
    verif = cons["ok"] + cons["inconsistente"]
    ader = [l for l in lotes if l["aderencia_mensuravel"]]
    multi = [l for l in lotes if l["quantidade_pedida"] >= 2 and l["quantidade_gerada"] >= 1]
    tempo = sum(l["tempo_s"] for l in lotes)
    return {
        "num_lotes": len(lotes), "questoes_pedidas": ped, "questoes_geradas": nq,
        "json_valido_pct": _pct(sum(l["json_validos"] for l in lotes), cham),
        "quantidade_entregue_pct": _pct(min(nq, ped), ped),
        "schema_pct": _pct(sum(l["schema_ok"] for l in lotes), nq),
        "alternativas_distintas_pct": _pct(sum(l["alternativas_distintas"] for l in lotes), nq),
        "difficulty_correta_pct": _pct(sum(l["difficulty_correta"] for l in lotes), nq),
        "consistencia": cons,
        "consistencia_ok_pct": _pct(cons["ok"], verif),  # entre as verificáveis
        "consistencia_inconsistente_pct": _pct(cons["inconsistente"], nq),  # sobre todas
        "consistencia_verificavel_n": verif,
        "aderencia_pct": _pct(sum(l["aderentes"] for l in ader),
                              sum(l["quantidade_gerada"] for l in ader)),
        "depende_de_visual_pct": _pct(sum(l["depende_de_visual"] for l in lotes), nq),
        "coverage_score": _media([l.get("coverage_score") for l in multi]),
        "duplicate_rate": _media([l.get("duplicate_rate") for l in multi]),
        "cosseno_media": _media([l["semantic_similarity"]["cosseno_media"] for l in multi
                                 if l.get("semantic_similarity")]),
        "jaccard_max": _media([l["semantic_similarity"]["jaccard_max"] for l in multi
                               if l.get("semantic_similarity")]),
        "structural_diversity": _media([l.get("structural_diversity") for l in multi]),
        "context_diversity": _media([l.get("context_diversity") for l in multi]),
        "raciocinio_diversity": _media([l.get("raciocinio_diversity") for l in multi]),
        "diversity_score": _media([l.get("diversity_score") for l in multi]),
        "lotes_diversidade_n": len(multi),
        "tempo_total_s": round(tempo, 2),
        "tempo_por_questao_s": round(tempo / nq, 3) if nq else None,
    }


# --------------------------------------------------------------------------
# Geradores (real e simulado)
# --------------------------------------------------------------------------

def _user_prompt(prompt, quantidade):
    return USER_TEMPLATE.format(quantidade=quantidade, ano=prompt["ano"],
                                habilidade=prompt["habilidade"], descricao=prompt["descricao"],
                                dificuldade=prompt["dificuldade"])


class GeradorReal:
    """Envolve test_model.generate_validated (best-of-N + validações de produção)."""

    def __init__(self, llama_cli, gguf, threads, grammar, retries):
        import test_model  # importação tardia: --dry-run não precisa do llama.cpp
        self.tm = test_model
        self.llama_cli, self.gguf, self.threads = llama_cli, gguf, threads
        self.grammar, self.retries = grammar, retries

    def chamada(self, prompt, quantidade, base_seed):
        """Retorna (questoes, json_valido, tempo_s)."""
        r = self.tm.generate_validated(
            self.llama_cli, self.gguf, _user_prompt(prompt, quantidade), self.threads,
            MAX_NEW_TOKENS_POR_QUESTAO * quantidade, grammar=self.grammar,
            retries=self.retries, quantidade=quantidade, base_seed=base_seed)
        top = parse_json(r["text"])
        qs = extract_questoes(top)
        # generate_validated já aplicou fix_gabarito na 1ª questão (r["obj"]):
        # usa a versão pós-processada, que é a que o app entregaria.
        if qs and isinstance(r.get("obj"), dict):
            qs = [r["obj"]] + list(qs[1:])
        return qs, top is not None, r["elapsed"]

    def ajustado(self, prompt, base_seed, max_tentativas):
        import gerar_lote
        r = gerar_lote.gerar_lote_planejado(
            self.llama_cli, self.gguf, prompt["ano"], prompt["habilidade"], prompt["descricao"],
            prompt["dificuldade"], int(prompt["quantidade"]), self.threads, grammar=self.grammar,
            retries=self.retries, base_seed=base_seed, max_tentativas_diversidade=max_tentativas)
        qs = r.get("questoes") or extract_questoes(r.get("obj"))
        extra = {"regeneracoes_diversidade": r.get("regeneracoes_diversidade"),
                 "violacoes_finais": r.get("violacoes"), "plano": r.get("plano")}
        return qs, r.get("obj") is not None, float(r.get("tempo_s") or 0.0), extra


class GeradorSimulado:
    """Gerador determinístico para --dry-run e testes: sem llama.cpp.

    Imita o comportamento observado na auditoria: sem plano, o modelo repete o
    mesmo esqueleto trocando números (o que NÃO é diversidade); com plano, cada
    slot recebe o subtema/contexto pedido no sufixo. Serve para exercitar o
    pipeline de métricas, não para medir modelo algum.
    """

    def _q(self, enunciado, dif):
        return {"enunciado": enunciado,
                "alternativas": {"A": "10", "B": "12", "C": "14", "D": "16",
                                 "E": "Nenhuma das alternativas anteriores"},
                "resolucao_passo_a_passo": "Somando: 5 + 7 = 12.", "resposta_correta": "B",
                "difficulty": DIFFICULTY_MAP.get(dif, "MEDIUM")}

    def _base(self, prompt):
        ent = dv.obter_habilidade(prompt["ano"], prompt["habilidade"])
        return ent["subtemas"][0]["rotulo"] if ent else prompt["descricao"]

    def chamada(self, prompt, quantidade, base_seed):
        s = base_seed or 0
        qs = [self._q(f"Considere {self._base(prompt)} com medida {s + i} cm. Qual é a resposta?",
                      prompt["dificuldade"]) for i in range(quantidade)]
        return qs, True, 0.01 * quantidade

    def ajustado(self, prompt, base_seed, max_tentativas):
        def gerar_fn(slot, restricao, tentativa):
            return self._q(f"Em uma situação de {slot['contexto_rotulo']}, uma questão sobre "
                           f"{slot['subtema_rotulo']}: {slot['estrutura_rotulo']}, raciocínio "
                           f"{slot['tipo_raciocinio_rotulo']}. Qual é a resposta?",
                           prompt["dificuldade"])
        qs, rel = dv.gerar_lote_diverso(gerar_fn, prompt["ano"], prompt["habilidade"],
                                        int(prompt["quantidade"]), prompt["dificuldade"],
                                        seed=str(base_seed), max_tentativas=max_tentativas)
        return qs, True, 0.01 * len(qs), {"regeneracoes_diversidade": sum(r["tentativas"] - 1 for r in rel)}


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------

def rodar_atual(gerador, prompt, modo_atual, taxonomia=None):
    n, seed = int(prompt["quantidade"]), int(prompt["seed"])
    if modo_atual == "unico":
        qs, ok, t = gerador.chamada(prompt, n, seed)
        return metricas_lote(qs, prompt, int(ok), 1, t, taxonomia)
    qs, oks, t = [], 0, 0.0
    for i in range(n):
        # seeds distintas por chamada, mas idênticas entre modelos (pareamento).
        q, ok, ti = gerador.chamada(prompt, 1, seed * 1000 + i)
        qs += q[:1]
        oks += int(ok)
        t += ti
    return metricas_lote(qs, prompt, oks, n, t, taxonomia)


def rodar_ajustado(gerador, prompt, max_tentativas, taxonomia=None):
    qs, ok, t, extra = gerador.ajustado(prompt, int(prompt["seed"]), max_tentativas)
    return metricas_lote(qs, prompt, int(ok), 1, t, taxonomia, extra)


def avaliar(gerador, prompts, modos=("atual", "ajustado"), modo_atual="unico",
            max_tentativas=dv.MAX_TENTATIVAS_DIVERSIDADE, taxonomia=None, log=print):
    """Roda todos os prompts em cada modo e retorna {modo: {lotes, agregado}}."""
    saida = {}
    for modo in modos:
        lotes = []
        for p in prompts:
            t0 = time.perf_counter()
            try:
                if modo == "atual":
                    lote = rodar_atual(gerador, p, modo_atual, taxonomia)
                else:
                    lote = rodar_ajustado(gerador, p, max_tentativas, taxonomia)
            except ImportError as e:
                # gerar_lote.py pode ainda não existir: registra e segue, sem
                # inventar números para o modo ajustado.
                log(f"  [{modo}] {p['id']}: indisponível ({e})")
                saida[modo] = {"erro": str(e)}
                break
            lotes.append(lote)
            log(f"  [{modo}] {p['id']}: {lote['quantidade_gerada']}/{lote['quantidade_pedida']} "
                f"div={lote.get('diversity_score', 0):.3f} ({time.perf_counter() - t0:.1f}s)")
        else:
            saida[modo] = {"lotes": lotes, "agregado": agregar(lotes)}
    return saida


# --------------------------------------------------------------------------
# Relatório
# --------------------------------------------------------------------------

def _fmt(v, suf=""):
    if v is None:
        return "—"
    return f"{v:.3f}{suf}" if isinstance(v, float) and not suf else f"{v}{suf}"


def tabela_markdown(relatorio):
    """Tabela comparativa exigida (atual x ajustado) + detalhamento."""
    modos = [m for m in ("atual", "ajustado") if "agregado" in relatorio["modos"].get(m, {})]
    ag = {m: relatorio["modos"][m]["agregado"] for m in modos}
    linhas = [
        ("JSON válido (%)", "json_valido_pct", "%", "maior"),
        ("Correção matemática — inconsistentes (%)", "consistencia_inconsistente_pct", "%", "menor"),
        ("Correção matemática — ok entre verificáveis (%)", "consistencia_ok_pct", "%", "maior"),
        ("Aderência à habilidade (%)", "aderencia_pct", "%", "maior"),
        ("Cobertura de subtemas (coverage_score)", "coverage_score", "", "maior"),
        ("Repetição semântica (duplicate_rate)", "duplicate_rate", "", "menor"),
        ("Repetição semântica (cosseno médio)", "cosseno_media", "", "menor"),
        ("Diversidade estrutural", "structural_diversity", "", "maior"),
        ("Diversidade contextual", "context_diversity", "", "maior"),
        ("diversity_score", "diversity_score", "", "maior"),
        ("Schema completo (%)", "schema_pct", "%", "maior"),
        ("Difficulty correta (%)", "difficulty_correta_pct", "%", "maior"),
        ("Depende de visual ausente (%)", "depende_de_visual_pct", "%", "menor"),
        ("Tempo por questão (s)", "tempo_por_questao_s", "", "menor"),
    ]
    out = [f"# Diversidade — {relatorio['rotulo']}", "",
           f"- artefato: `{relatorio.get('artefato')}`  sha256: `{(relatorio.get('artefato_sha256') or '—')[:12]}`",
           f"- prompts: {relatorio['num_prompts']}  modo atual: `{relatorio['modo_atual']}`"
           f"  dry-run: {relatorio['dry_run']}", "",
           "| Métrica | " + " | ".join(modos) + " | melhor |",
           "|---|" + "---|" * len(modos) + "---|"]
    for nome, chave, suf, sentido in linhas:
        out.append(f"| {nome} | " + " | ".join(_fmt(ag[m].get(chave), suf) for m in modos)
                   + f" | {sentido} |")
    for m, v in relatorio["modos"].items():
        if "erro" in v:
            out.append(f"\n> modo `{m}` indisponível: {v['erro']}")
    out += ["", "## Por lote (diversity_score / coverage / duplicate_rate)", "",
            "| prompt | K | " + " | ".join(modos) + " |", "|---|---|" + "---|" * len(modos)]
    ids = [l["id"] for l in relatorio["modos"][modos[0]]["lotes"]] if modos else []
    por = {m: {l["id"]: l for l in relatorio["modos"][m]["lotes"]} for m in modos}
    for i in ids:
        cel = []
        for m in modos:
            l = por[m].get(i, {})
            cel.append(f"{_fmt(l.get('diversity_score'))} / {_fmt(l.get('coverage_score'))} / "
                       f"{_fmt(l.get('duplicate_rate'))}")
        out.append(f"| {i} | {por[modos[0]][i]['k_subtemas']} | " + " | ".join(cel) + " |")
    return "\n".join(out) + "\n"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rotulo", required=True, help="nome do relatório (outputs/diversidade_<rotulo>.json)")
    ap.add_argument("--model", help=".gguf a avaliar (padrão: test_model.find_gguf)")
    ap.add_argument("--llama-cli")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--no-grammar", action="store_true")
    ap.add_argument("--prompts", default=str(PROMPTS_PATH))
    ap.add_argument("--num-prompts", type=int, help="rodada reduzida: os primeiros N prompts")
    ap.add_argument("--modos", default="atual,ajustado")
    ap.add_argument("--modo-atual", choices=MODOS_ATUAL, default="unico",
                    help="unico = 1 chamada com N questões; independente = N chamadas sem plano")
    ap.add_argument("--max-tentativas-diversidade", type=int, default=dv.MAX_TENTATIVAS_DIVERSIDADE)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="réplica: soma 7919*offset às seeds dos prompts")
    ap.add_argument("--dry-run", action="store_true", help="gerador simulado; não chama o modelo")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    prompts = carregar_prompts(args.prompts, args.num_prompts)
    if args.seed_offset:
        # réplicas com outras seeds; mesmo offset => mesmas seeds em todos os braços (pareado).
        prompts = [{**p, "seed": int(p["seed"]) + 7919 * args.seed_offset} for p in prompts]
    modos = tuple(m.strip() for m in args.modos.split(",") if m.strip())
    if not re.fullmatch(r"[\w.\-]+", args.rotulo):
        raise SystemExit("--rotulo deve conter só letras, números, '.', '_' ou '-'")

    artefato = sha = None
    if args.dry_run:
        gerador = GeradorSimulado()
    else:
        import test_model
        llama = test_model.find_llama_cli(args.llama_cli)
        gguf = test_model.find_gguf(args.model)
        grammar = None if args.no_grammar else (test_model.GRAMMAR_PATH
                                                if test_model.GRAMMAR_PATH.exists() else None)
        gerador = GeradorReal(llama, gguf, args.threads, grammar, args.retries)
        artefato, sha = str(gguf), _sha256(gguf)

    print(f"Avaliando {len(prompts)} prompts, modos={modos}, atual={args.modo_atual}"
          f"{' (DRY-RUN)' if args.dry_run else ''}")
    resultado = avaliar(gerador, prompts, modos, args.modo_atual, args.max_tentativas_diversidade)
    relatorio = {
        "rotulo": args.rotulo, "artefato": artefato, "artefato_sha256": sha,
        "dry_run": args.dry_run, "modo_atual": args.modo_atual,
        "seed_offset": args.seed_offset,
        "max_tentativas_diversidade": args.max_tentativas_diversidade,
        "conjunto_prompts": str(args.prompts), "num_prompts": len(prompts),
        "prompt_ids": [p["id"] for p in prompts],
        "pesos_diversidade": dv.PESOS_DIVERSIDADE, "modos": resultado,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pj, pm = out / f"diversidade_{args.rotulo}.json", out / f"diversidade_{args.rotulo}.md"
    pj.write_text(json.dumps(relatorio, ensure_ascii=False, indent=1), encoding="utf-8")
    md = tabela_markdown(relatorio)
    pm.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Relatório: {pj}\nTabela:    {pm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
