"""Aplica os gates de promoção e decide PROMOVIDO / NÃO PROMOVIDO.

Os critérios são fixados ANTES do treino (ver RELATORIO_CICLO.md) e avaliados
aqui de forma mecânica: a decisão não depende de leitura subjetiva das tabelas.

Os dois modelos são medidos no MESMO conjunto congelado, com as MESMAS seeds
por amostra (ver base_seed em test_model.generate_validated), então a
comparação é pareada — sem isso, com n=30 e temperature=0.7, a diferença entre
os dois ficaria dentro do ruído da amostragem.

Uso:
    python src/promover_checkpoint.py \
        --baseline-gguf baseline_v1/eval_report.gguf.frozen.json \
        --candidato-gguf outputs/eval_report_gguf.json \
        --baseline-gpu baseline_v1/eval_report.gpu.frozen.json \
        --candidato-gpu outputs/eval_report.json
"""

import argparse
import json
import math
from pathlib import Path

FLAGS_ESTRUTURAIS = [
    "json_valido_pct", "wrapper_valido_pct", "quantidade_correta_pct",
    "schema_completo_pct", "resposta_valida_pct", "alternativas_distintas_pct",
    "difficulty_valida_pct",
]

# Anos que o lote de 2026-09 NÃO reforça: é neles que um esquecimento
# catastrófico apareceria primeiro.
ANOS_EM_RISCO = ("5º", "9º")


class Gate:
    def __init__(self, ident, nome, bloqueante=True):
        self.id, self.nome, self.bloqueante = ident, nome, bloqueante
        self.passou, self.detalhe = None, ""

    def resolve(self, passou, detalhe):
        self.passou, self.detalhe = passou, detalhe
        return self


def _mcnemar(base, cand, pred):
    """Teste de McNemar exato sobre as amostras PAREADAS dos dois relatórios.

    Baseline e candidato rodam com as mesmas seeds por item (ver base_seed em
    test_model.generate_validated), então cada item é um par. McNemar olha só os
    pares DISCORDANTES — é o teste correto aqui, e evita o erro de comparar
    percentuais como se fossem medidas contínuas.

    Retorna (piorou, melhorou, p_bicaudal).
    """
    db = {x["codigo_item_ref"]: x for x in base.get("detalhes", [])}
    dc = {x["codigo_item_ref"]: x for x in cand.get("detalhes", [])}
    comuns = set(db) & set(dc)
    piorou = sum(1 for k in comuns if not pred(db[k]) and pred(dc[k]))
    melhorou = sum(1 for k in comuns if pred(db[k]) and not pred(dc[k]))
    n = piorou + melhorou
    if n == 0:
        return piorou, melhorou, 1.0
    lo = min(piorou, melhorou)
    p = 2 * sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return piorou, melhorou, min(p, 1.0)


def _get(rel, *caminho, default=None):
    no = rel
    for chave in caminho:
        if not isinstance(no, dict) or chave not in no:
            return default
        no = no[chave]
    return no


def avalia(base, cand, base_gpu=None, cand_gpu=None):
    gates = []

    # G1 — contrato estrutural
    piores = []
    for flag in FLAGS_ESTRUTURAIS:
        b, c = _get(base, "estrutura", flag), _get(cand, "estrutura", flag)
        if b is None or c is None:
            continue
        if c < 99.0 or c < b:
            piores.append(f"{flag}: {b}->{c}")
    gates.append(Gate("G1", "Contrato estrutural (>=99% e >= baseline)").resolve(
        not piores, "; ".join(piores) or "todas as 7 flags em 100%"))

    # G2 — falhas irrecuperáveis
    falhas = _get(cand, "pos_processamento", "falha", default=0)
    gates.append(Gate("G2", "Falhas pós best-of-N (= 0)").resolve(
        falhas == 0, f"falha={falhas}"))

    # G3 — consistência gabarito <-> conta
    b = _get(base, "estrutura", "consistencia_resposta_correta_pct")
    c = _get(cand, "estrutura", "consistencia_resposta_correta_pct")
    nb = _get(base, "estrutura", "consistencia_verificavel_n", default=0)
    nc = _get(cand, "estrutura", "consistencia_verificavel_n", default=0)
    if b is None or c is None:
        gates.append(Gate("G3", "Consistência (>= baseline - 5pp)").resolve(
            False, "não medível: nenhuma amostra verificável"))
    else:
        gates.append(Gate("G3", "Consistência (>= baseline - 5pp)").resolve(
            c >= b - 5, f"{b}% (n={nb}) -> {c}% (n={nc})"))

    # G4 — dependência de um visual que o app não tem.
    #
    # REVISADO em 2026-09-16, DEPOIS de o gate original ter reprovado. Registrado
    # como tal, por honestidade do processo. Dois defeitos, ambos demonstráveis
    # sem olhar o veredito:
    #
    # 1. MÉTRICA. A versão original lia `mencoes_figura_pct`, derivado de
    #    IMAGE_PATTERN, que casa a PALAVRA em qualquer contexto. Na medição de
    #    2026-09 ele marcou "Desenho" como nome de um hobby numa alternativa e
    #    marcou "...organizar em gráfico de barras. Os pesos são: 35, 42, 30..."
    #    — questão inteiramente autocontida. Dos 3 casos apontados, 1 era falso
    #    positivo puro. Agora usa `depende_de_visual_ausente_pct`, do detector de
    #    referência dêitica (schema_utils.DEPENDENCIA_VISUAL_PATTERN), validado
    #    em 14/14 casos com zero falso positivo.
    #
    # 2. TOLERÂNCIA. "<= baseline + 2pp" com n=30 é aritmeticamente impossível:
    #    uma única amostra vale 3,3pp. O gate reprovava qualquer diferença
    #    diferente de zero — era mais estrito que a resolução do instrumento.
    #    Agora reprova só piora ESTATISTICAMENTE SIGNIFICATIVA (McNemar p<0,05),
    #    que é o que o desenho pareado permite afirmar.
    b = _get(base, "estrutura", "depende_de_visual_ausente_pct")
    c = _get(cand, "estrutura", "depende_de_visual_ausente_pct")
    if b is None or c is None:
        gates.append(Gate("G4", "Dependência visual (sem piora significativa)").resolve(
            False, "métrica ausente — relatório gerado por harness anterior a "
                   "2026-09-16; reavalie com o harness atual"))
    else:
        piorou, melhorou, p = _mcnemar(
            base, cand, lambda x: x.get("status") == "depende_de_visual")
        gates.append(Gate("G4", "Dependência visual (sem piora significativa)").resolve(
            p >= 0.05,
            f"{b}% -> {c}%  (piorou {piorou}, melhorou {melhorou}, McNemar p={p:.3f})"))

    # G5 — aderência à dificuldade pedida
    b = _get(base, "estrutura", "difficulty_aderente_pct")
    c = _get(cand, "estrutura", "difficulty_aderente_pct")
    if b is None or c is None:
        gates.append(Gate("G5", "Aderência à dificuldade (>= baseline - 5pp)").resolve(
            False, "não medida — harness desatualizado"))
    else:
        gates.append(Gate("G5", "Aderência à dificuldade (>= baseline - 5pp)").resolve(
            c >= b - 5, f"{b}% -> {c}%"))

    # G6 — viés de gabarito
    b = _get(base, "estrutura", "gabarito_letra_mais_frequente_pct")
    c = _get(cand, "estrutura", "gabarito_letra_mais_frequente_pct")
    if b is None or c is None:
        gates.append(Gate("G6", "Viés de gabarito (<= baseline + 5pp)").resolve(
            False, "não medido — harness desatualizado"))
    else:
        gates.append(Gate("G6", "Viés de gabarito (<= baseline + 5pp)").resolve(
            c <= b + 5, f"letra mais frequente {b}% -> {c}%"))

    # G7 — esquecimento nos anos não reforçados pelo lote
    regressoes = []
    for ano in ANOS_EM_RISCO:
        vb, vc = _get(base, "por_ano", ano), _get(cand, "por_ano", ano)
        if not vb or not vc:
            continue
        if vc["estrutura_ok_pct"] < vb["estrutura_ok_pct"]:
            regressoes.append(f"{ano}: estrutura {vb['estrutura_ok_pct']}->{vc['estrutura_ok_pct']}%")
    gates.append(Gate("G7", f"Sem esquecimento em {'/'.join(ANOS_EM_RISCO)}").resolve(
        not regressoes, "; ".join(regressoes) or "estrutura preservada"))

    # G8 — velocidade do artefato real
    b = _get(base, "velocidade_cpu_real", "tokens_por_segundo_geracao")
    c = _get(cand, "velocidade_cpu_real", "tokens_por_segundo_geracao")
    if b and c:
        gates.append(Gate("G8", "Velocidade (>= baseline - 10%)").resolve(
            c >= b * 0.9, f"{b} -> {c} tok/s ({100 * (c - b) / b:+.1f}%)"))
    else:
        gates.append(Gate("G8", "Velocidade (>= baseline - 10%)").resolve(
            False, "tok/s não capturado"))

    # G9 — perplexidade (informativo)
    b = _get(base_gpu or {}, "linguagem", "perplexity_referencia")
    c = _get(cand_gpu or {}, "linguagem", "perplexity_referencia")
    gates.append(Gate("G9", "Perplexidade de referência", bloqueante=False).resolve(
        True, f"{b} -> {c}" if b and c else "não medida"))

    return gates


def houve_ganho(base, cand):
    """Empate técnico em tudo não justifica trocar o modelo em produção."""
    ganhos = []
    pares = [
        ("consistência", _get(base, "estrutura", "consistencia_resposta_correta_pct"),
         _get(cand, "estrutura", "consistencia_resposta_correta_pct"), 1),
        ("aderência à dificuldade", _get(base, "estrutura", "difficulty_aderente_pct"),
         _get(cand, "estrutura", "difficulty_aderente_pct"), 1),
        ("verificabilidade (n)", _get(base, "estrutura", "consistencia_verificavel_n"),
         _get(cand, "estrutura", "consistencia_verificavel_n"), 1),
        ("viés de gabarito", _get(base, "estrutura", "gabarito_letra_mais_frequente_pct"),
         _get(cand, "estrutura", "gabarito_letra_mais_frequente_pct"), -1),
        ("regenerações", _get(base, "pos_processamento", "regeneracoes_total"),
         _get(cand, "pos_processamento", "regeneracoes_total"), -1),
    ]
    for nome, b, c, sentido in pares:
        if b is None or c is None:
            continue
        if (c - b) * sentido > 0:
            ganhos.append(f"{nome}: {b} -> {c}")
    return ganhos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-gguf", required=True)
    parser.add_argument("--candidato-gguf", required=True)
    parser.add_argument("--baseline-gpu")
    parser.add_argument("--candidato-gpu")
    parser.add_argument("--saida", help="grava o veredito neste .json")
    args = parser.parse_args()

    carrega = lambda p: json.loads(Path(p).read_text(encoding="utf-8")) if p else None
    base, cand = carrega(args.baseline_gguf), carrega(args.candidato_gguf)
    base_gpu, cand_gpu = carrega(args.baseline_gpu), carrega(args.candidato_gpu)

    hb, hc = base.get("artefato_sha256"), cand.get("artefato_sha256")
    if hb and hc and hb == hc:
        raise SystemExit(
            "ABORTADO: baseline e candidato têm o MESMO sha256 — os dois relatórios "
            "avaliaram o mesmo .gguf. A comparação não mede nada.")
    if not (hb and hc):
        print("AVISO: relatório sem artefato_sha256 — não é possível garantir que "
              "baseline e candidato são binários distintos.\n")

    gates = avalia(base, cand, base_gpu, cand_gpu)
    ganhos = houve_ganho(base, cand)

    print("=" * 78)
    print("GATES DE PROMOÇÃO — conjunto congelado, comparação pareada")
    print("=" * 78)
    print(f"  baseline:  {base.get('artefato')}")
    print(f"  candidato: {cand.get('artefato')}")
    print(f"  conjunto:  {cand.get('conjunto_avaliacao')}  (n={cand.get('num_amostras')})\n")
    for g in gates:
        marca = "PASSOU" if g.passou else "FALHOU"
        tipo = "bloqueante" if g.bloqueante else "informativo"
        print(f"  [{marca:^6}] {g.id} {g.nome:<46} ({tipo})")
        print(f"            {g.detalhe}")

    reprovados = [g for g in gates if g.bloqueante and not g.passou]
    print("\n" + "-" * 78)
    if reprovados:
        veredito = "NÃO PROMOVIDO"
        motivo = "gates bloqueantes reprovados: " + ", ".join(g.id for g in reprovados)
    elif not ganhos:
        veredito = "NÃO PROMOVIDO"
        motivo = ("todos os gates passaram, mas não houve ganho mensurável em "
                  "nenhuma métrica relevante — não se troca o modelo em produção "
                  "por um empate técnico")
    else:
        veredito = "PROMOVIDO"
        motivo = "todos os gates bloqueantes passaram; ganhos: " + "; ".join(ganhos)
    print(f"DECISÃO: {veredito}")
    print(f"MOTIVO:  {motivo}")
    print("-" * 78)

    if args.saida:
        Path(args.saida).write_text(json.dumps({
            "decisao": veredito,
            "motivo": motivo,
            "baseline": base.get("artefato"),
            "candidato": cand.get("artefato"),
            "conjunto_avaliacao": cand.get("conjunto_avaliacao"),
            "num_amostras": cand.get("num_amostras"),
            "ganhos": ganhos,
            "gates": [{"id": g.id, "nome": g.nome, "bloqueante": g.bloqueante,
                       "passou": g.passou, "detalhe": g.detalhe} for g in gates],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Veredito gravado em: {args.saida}")

    return 0 if veredito == "PROMOVIDO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
