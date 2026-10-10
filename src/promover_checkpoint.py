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
import statistics
import math
from collections import Counter
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


# --------------------------------------------------------------------------
# Significância estatística nas métricas de proporção (revisão de 2026-09-30)
# --------------------------------------------------------------------------
# Os gates comparavam percentuais como se fossem medidas exatas. Não são: são
# proporções estimadas em amostras pequenas, e o percentual esconde o tamanho da
# amostra. O caso mais grave é P3: `consistencia_inconsistente_pct` tem
# denominador de 228 questões, mas o NUMERADOR só pode sair das ~31 que
# check_consistency consegue verificar (13,6% do corpus medido em 2026-09). Na
# medição que motivou esta revisão o gate decidiu sobre a diferença entre 3 e 5
# questões — 1,32% contra 2,19%. É a mesma crítica já feita ao G4 em 2026-09-16,
# resolvida lá com McNemar. Aqui: (a) todo gate de proporção passa a exibir o
# intervalo de Wilson 95%, para que o leitor veja a resolução do instrumento
# (1,32% é [0,45; 3,80] e 2,19% é [0,94; 5,03] — intervalos que se sobrepõem em
# quase toda a extensão); (b) P3, P4 e o G11 só reprovam se a piora, além de
# estourar a tolerância, for distinguível do ruído amostral.
Z_95 = 1.959963985
ALFA = 0.05


def wilson(k, n, z=Z_95):
    """Intervalo de confiança de Wilson para uma proporção, em pontos percentuais.

    Preferido ao intervalo normal (Wald) porque não degenera em [0,0] quando
    k=0 nem estoura de [0,100] — justamente o regime em que estes gates operam
    (json_valido 100%, inconsistentes 1-2%).
    """
    if not n:
        return None
    p = k / n
    d = 1.0 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100.0 * max(0.0, centro - meia), 2),
            round(100.0 * min(1.0, centro + meia), 2))


# Numerador/denominador de cada métrica do agregado, a partir dos lotes. Sem
# isso só existe o percentual já arredondado, e IC de Wilson precisa de n.
_CONTAGEM_LOTE = {
    "json_valido_pct": (lambda l: l["json_validos"], lambda l: l["chamadas"]),
    "quantidade_entregue_pct": (lambda l: min(l["quantidade_gerada"], l["quantidade_pedida"]),
                                lambda l: l["quantidade_pedida"]),
    "schema_pct": (lambda l: l["schema_ok"], lambda l: l["quantidade_gerada"]),
    "consistencia_inconsistente_pct": (lambda l: l["consistencia"]["inconsistente"],
                                       lambda l: l["quantidade_gerada"]),
    "aderencia_pct": (lambda l: l["aderentes"] if l.get("aderencia_mensuravel") else 0,
                      lambda l: l["quantidade_gerada"] if l.get("aderencia_mensuravel") else 0),
    "depende_de_visual_pct": (lambda l: l["depende_de_visual"], lambda l: l["quantidade_gerada"]),
    "difficulty_correta_pct": (lambda l: l["difficulty_correta"], lambda l: l["quantidade_gerada"]),
}


def contagem(rel, modo, met):
    """(k, n) da métrica `met` somando os lotes do modo, ou None se não der."""
    lotes = _get(rel or {}, "modos", modo, "lotes")
    f = _CONTAGEM_LOTE.get(met)
    if not lotes or not f:
        return None
    try:
        return sum(f[0](l) for l in lotes), sum(f[1](l) for l in lotes)
    except (KeyError, TypeError):
        return None


def _ic_str(base_rel, cand_rel, modo, met, modo_base=None):
    """Sufixo com o IC95 de Wilson dos dois lados, ou '' se n não for derivável."""
    cb = contagem(base_rel, modo_base or modo, met)
    cc = contagem(cand_rel, modo, met)
    if not cb or not cc or not cb[1] or not cc[1]:
        return ""
    ib, ic = wilson(*cb), wilson(*cc)
    return (f"  [IC95 base {ib[0]}–{ib[1]}% (n={cb[1]}), "
            f"cand {ic[0]}–{ic[1]}% (n={cc[1]})]")


# --------------------------------------------------------------------------
# Pareamento no perfil planejado
# --------------------------------------------------------------------------
# PAREAMENTO POR (seed, prompt_id, índice da questão no lote).
#
# É válido — e essa é a parte que precisa ficar escrita, não suposta: o plano de
# slots vem de diversidade.planejar_lote(ano, habilidade, N, dificuldade,
# seed=base_seed) (ver gerar_lote.gerar_lote_planejado). O plano é função só do
# prompt e da seed, NÃO do modelo. O slot i recebe o mesmo subtema, o mesmo tipo
# de raciocínio, o mesmo contexto e a mesma seed (base_seed*1000 + i*10 + tent)
# nos dois lados. Entre baseline e candidato varia apenas o peso do modelo — que
# é exatamente o efeito que se quer isolar. É o mesmo argumento que sustenta o
# pareamento por codigo_item_ref no G4, aplicado ao formato do perfil planejado,
# onde os relatórios trazem `lotes[].questoes[]` e não `detalhes[]`.
#
# ONDE O PAREAMENTO POR ÍNDICE QUEBRA: gerar_lote só acrescenta o slot cujo
# melhor candidato não é None (gerar_lote.py:155). Se um slot é descartado,
# todos os índices seguintes deslizam e o par (i, i) passa a comparar slots
# diferentes. Por isso um lote com quantidade_gerada != quantidade_pedida em
# QUALQUER um dos lados é excluído do teste pareado e contado à parte, em vez de
# ser pareado errado. Nos relatórios reais de 2026-09 isso nunca ocorre
# (quantidade_entregue_pct = 100% nos dois lados).

def _lotes_por_chave(rel, modo):
    """{(seed, prompt_id): lote}, ou None se houver chave repetida (sem pareamento)."""
    out = {}
    for l in _get(rel or {}, "modos", modo, "lotes", default=[]) or []:
        k = (l.get("seed_pareamento", 0), l.get("id"))
        if k in out:
            return None
        out[k] = l
    return out or None


def _ruim_consistencia(lote, q):
    from schema_utils import check_consistency
    return check_consistency(q)[0] is False


def _ruim_aderencia(lote, q):
    import diversidade as dv
    ent = dv.obter_habilidade(lote["ano"], lote["habilidade"])
    ids = {s["id"] for s in ent["subtemas"]} if ent else set()
    sub = dv.classificar_questao(q, lote["ano"], lote["habilidade"])["subtema"]
    return not (sub != "outros" and sub in ids)


# métrica -> (predicado "esta questão está no estado RUIM", filtro de lote).
# Só existem para as métricas cujo estado é decidível questão a questão; as
# demais (json válido, tempo) são de lote e não entram no teste pareado.
PREDICADO_PAREADO = {
    "consistencia_inconsistente_pct": (_ruim_consistencia, lambda l: True),
    "aderencia_pct": (_ruim_aderencia, lambda l: bool(l.get("aderencia_mensuravel"))),
}


def mcnemar_planejado(base_rel, cand_rel, modo, met):
    """McNemar exato sobre os pares (seed, prompt_id, índice) do perfil planejado.

    Retorna (piorou, melhorou, p, n_pares, lotes_excluidos) ou None se a métrica
    não for pareável ou os relatórios não trouxerem as questões.

    ATENÇÃO À RÉGUA: o veredito por questão é RECALCULADO aqui com o
    schema_utils/diversidade ATUAIS, porque os relatórios guardam só a contagem
    por lote, não o veredito por questão. Os dois lados são recalculados com a
    MESMA régua, então o teste pareado é interno e coerente. O percentual que a
    tolerância compara continua vindo do agregado gravado no relatório — ver a
    nota em `_piorou_de_verdade`.
    """
    par = PREDICADO_PAREADO.get(met)
    if not par:
        return None
    pred, filtro = par
    lb, lc = _lotes_por_chave(base_rel, modo), _lotes_por_chave(cand_rel, modo)
    if not lb or not lc:
        return None
    piorou = melhorou = n = fora = 0
    for k in sorted(set(lb) & set(lc)):
        a, b = lb[k], lc[k]
        if not (filtro(a) and filtro(b)):
            continue
        qa, qb = a.get("questoes"), b.get("questoes")
        if qa is None or qb is None:
            return None
        # índice só é comparável se nenhum slot foi descartado dos dois lados
        if (len(qa) != a.get("quantidade_pedida") or len(qb) != b.get("quantidade_pedida")
                or len(qa) != len(qb)):
            fora += 1
            continue
        for x, y in zip(qa, qb):
            n += 1
            ra, rb = pred(a, x), pred(b, y)
            piorou += (not ra) and rb
            melhorou += ra and (not rb)
    if not n:
        return None
    d = piorou + melhorou
    if d == 0:
        return piorou, melhorou, 1.0, n, fora
    lo = min(piorou, melhorou)
    p = min(2 * sum(math.comb(d, i) for i in range(lo + 1)) / (2 ** d), 1.0)
    return piorou, melhorou, p, n, fora


def limite_superior_piora(piorou, melhorou, n, z=Z_95):
    """Limite superior 95% da PIORA verdadeira, em pontos percentuais.

    Intervalo de McNemar para a diferença de proporções pareadas: a estimativa
    é (piorou - melhorou)/n e a variância é a dos pares discordantes. Negativo
    significa que até o pior caso compatível com os dados é uma MELHORA.

    Por que este número e não o p-valor, que é o que estava aqui antes
    (revisão adversarial de 2026-09-30):

    O teste de McNemar exato bicaudal tem p mínimo 2/2^d, com d = número de
    pares discordantes. Logo d<=5 NUNCA atinge p<0,05. A regra anterior era
    "reprova = estourou a tolerância E p<ALFA", e com ela um candidato que
    quebrasse até 5 gabaritos em 228 passava por P3 E por G11 — MEDIDO: partindo
    de base_k0_s{0,1,2} e quebrando 5 gabaritos hoje consistentes, a métrica ia
    de 1,32% para 3,51% (2,7x o baseline, 2,2pp acima da tolerância de 1pp),
    McNemar dava p=0,0625 e os DOIS gates aprovavam. Só a partir de 6 quebras
    (p=0,031) reprovavam. Isso inverte o ônus da prova de um gate BLOQUEANTE:
    "o teste não detectou piora" virava "não houve piora", quando a causa era
    falta de poder, não ausência de efeito.

    O limite superior não tem esse defeito porque falta de poder ALARGA o
    intervalo em vez de estreitar o p: quando os dados não sustentam a
    absolvição, o limite estoura a tolerância e o gate reprova. É a direção
    conservadora que um gate bloqueante exige, e resolve de quebra o outro
    achado da revisão — o p bicaudal ignorava o SENTIDO do efeito e podia
    "confirmar piora" sobre uma discordância favorável ao candidato (o caso real
    de P4: piorou 15, melhorou 33 e p=0,013).

    Continua sendo uma ABSOLVIÇÃO, nunca uma acusação: só é consultado depois de
    a tolerância ter sido estourada no percentual gravado no relatório.
    """
    if not n:
        return None
    d = piorou + melhorou
    dif = (piorou - melhorou) / n
    var = (d - (piorou - melhorou) ** 2 / n) / (n * n)
    return 100.0 * (dif + z * math.sqrt(max(var, 0.0)))


def recontagem(rel, modo, met):
    """(k, n) da métrica recalculando o veredito questão a questão com a régua
    ATUAL do código, ou None. Existe para tornar visível a diferença entre o
    número gravado no relatório e o que o verificador de hoje diria: os
    relatórios de outputs/ foram gravados antes da revisão de check_consistency
    de 2026-09-30, e comparar um candidato medido com a régua nova contra um
    baseline gravado com a antiga daria leitura falsa."""
    par = PREDICADO_PAREADO.get(met)
    lotes = _get(rel or {}, "modos", modo, "lotes")
    if not par or not lotes:
        return None
    pred, filtro = par
    # sentido do numerador: para aderencia_pct conta quem ADERE (predicado ruim
    # negado); para consistencia_inconsistente_pct conta quem é inconsistente.
    conta_ruim = met != "aderencia_pct"
    k = n = 0
    for l in lotes:
        if not filtro(l):
            continue
        qs = l.get("questoes")
        if qs is None:
            return None
        for q in qs:
            n += 1
            k += pred(l, q) == conta_ruim
    return (k, n) if n else None


def nota_regua(base_rel, cand_rel, modo, met, vb, vc):
    """Aviso quando o relatório e o código atual discordam sobre a métrica."""
    cb, cc = recontagem(base_rel, modo, met), recontagem(cand_rel, modo, met)
    if not cb or not cc:
        return ""
    rb, rc = round(100.0 * cb[0] / cb[1], 2), round(100.0 * cc[0] / cc[1], 2)
    if abs(rb - vb) < 0.01 and abs(rc - vc) < 0.01:
        return ""
    return (f"AVISO: o relatório traz {vb} -> {vc}, mas a régua ATUAL do código "
            f"dá {rb} -> {rc} ({cb[0]}/{cb[1]} -> {cc[0]}/{cc[1]}). O relatório "
            f"é anterior à revisão do verificador; regrave-o antes de usar este "
            f"número como baseline de outro ciclo.")


def _piorou_de_verdade(base_rel, cand_rel, modo, met, vb, vc, sentido, tol):
    """Régua ÚNICA de reprovação por piora, usada por P3, P4 e G11.

    Reprova se a piora estoura a tolerância no percentual gravado no relatório
    E os dados pareados NÃO demonstram que a piora verdadeira cabe dentro da
    tolerância (não-inferioridade: limite superior 95% da piora <= tolerância).

    As duas etapas usam fontes diferentes de propósito: a tolerância tem de
    olhar o número que o relatório publica, que é o que o histórico comparou; o
    limite superior precisa do veredito por questão, que só existe recalculando.
    A conjunção só pode ABSOLVER por causa disso — nunca acusar.

    O ÔNUS DA PROVA É DO CANDIDATO, e essa é a diferença para a versão anterior
    desta função. Antes, a absolvição vinha de "p >= 0,05", que um n pequeno
    concede de graça; agora vem de um limite superior que um n pequeno NEGA.
    Ver limite_superior_piora para a medição que motivou a troca.

    Retorna (reprova, texto_do_teste, estourou_a_tolerancia). O teste pareado é
    SEMPRE calculado, mesmo quando a tolerância não foi estourada: quem lê o
    veredito precisa ver o tamanho do efeito medido, não só o carimbo.
    """
    estourou = (vc - vb) * sentido < -tol - 1e-9
    mc = mcnemar_planejado(base_rel, cand_rel, modo, met)
    if mc is None:
        return estourou, ("sem dados pareados — vale só a tolerância" if estourou else ""), estourou
    piorou, melhorou, p, n, fora = mc
    sup = limite_superior_piora(piorou, melhorou, n)
    txt = (f"McNemar pareado (régua atual): piorou {piorou}, melhorou {melhorou}, "
           f"p={p:.3f}, piora <= {sup:+.2f}pp (IC95 sup.), {n} pares"
           + (f", {fora} lote(s) sem pareamento por índice" if fora else ""))
    if not estourou:
        # O SENTIDO vem da contagem de discordantes, não do p bicaudal: o p não
        # distingue "piorou muito" de "melhorou muito".
        sentido_txt = ("melhora" if melhorou > piorou else "piora") + (
            " significativa" if p < ALFA else " dentro do ruído")
        return False, f"{txt}  [informativo, dentro da tolerância: {sentido_txt}]", False
    if sup is not None and sup <= tol + 1e-9:
        return False, txt + (f"  — não-inferioridade demonstrada (piora verdadeira "
                             f"<= {sup:+.2f}pp <= {tol}pp), NÃO reprova"), True
    return True, txt + ("  — os dados pareados NÃO demonstram que a piora cabe na "
                        f"tolerância de {tol}pp"), True


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


def _modo_geometria(rel):
    # Relatório anterior à guarda de geometria não tem a seção: equivale a "sombra".
    return _get(rel or {}, "geometria", "modo") or "sombra"


def houve_ganho(base, cand):
    """Empate técnico em tudo não justifica trocar o modelo em produção.

    "regenerações" só vale entre relatórios com o MESMO modo de geometria: no
    modo ativo a guarda regenera as questões de 9º H17 reprovadas, então o
    número mede o pipeline, não o modelo (baseline em ativo x candidato em
    sombra dava ganho espúrio; o inverso escondia ganho real — revisão
    adversarial de 2026-10-01)."""
    ganhos = []
    mesmo_modo = _modo_geometria(base) == _modo_geometria(cand)
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
        if nome == "regenerações" and not mesmo_modo:
            continue
        if (c - b) * sentido > 0:
            ganhos.append(f"{nome}: {b} -> {c}")
    return ganhos


# --------------------------------------------------------------------------
# G11 — diversidade sem regressão de qualidade (OPCIONAL)
# --------------------------------------------------------------------------
# Só roda se relatórios de src/avaliar_diversidade.py forem fornecidos. Mede o
# modo de geração escolhido (padrão "ajustado", o pipeline com plano de
# subtemas) nos MESMOS prompts/seeds do conjunto fixo data/prompts_diversidade.json.
#
# Por que é bloqueante quando presente: forçar subtema/contexto pode aumentar a
# variedade às custas do que o aluno de fato recebe (JSON quebrado, gabarito
# errado, questão fora da habilidade). Diversidade só conta como ganho se a
# qualidade ficar >= à do baseline (tolerância explícita, 0pp por padrão).

# (métrica, sentido): +1 = maior é melhor; -1 = menor é melhor.
QUALIDADE_DIVERSIDADE = [
    ("json_valido_pct", 1),
    ("schema_pct", 1),
    ("consistencia_inconsistente_pct", -1),
    ("aderencia_pct", 1),
]
TOLERANCIA_QUALIDADE_DIVERSIDADE_PP = 0.0

# REVISÃO DE 2026-09-30 — G11 passa a usar a MESMA régua que P3.
#
# Registrada como revisão feita DEPOIS de um veredito reprovado, como já foi
# feito com o G4 em 2026-09-16, para que a mudança fique exposta ao escrutínio
# em vez de escondida no histórico do git.
#
# O DEFEITO: P3 (perfil planejado) e G11 mediam a MESMA métrica,
# `consistencia_inconsistente_pct`, com tolerâncias diferentes — 1pp em P3 e 0pp
# em G11. Na medição de 2026-09-30 (base_k0_s{0,1,2} x exp_C_s{0,1,2}) a métrica
# foi de 1,32% para 2,19%: P3 PASSOU e G11 FALHOU sobre o mesmo número. Um
# veredito em que dois gates se contradizem não mede o modelo, mede o erro de
# projeto de quem escreveu os gates. Não há justificativa de desenho para as
# duas réguas: o G11 nasceu antes do perfil planejado e herdou um 0pp genérico
# que nunca foi pensado para esta métrica.
#
# POR QUE A REVISÃO É LEGÍTIMA, e não um afrouxamento para deixar passar:
# 1. Ela é demonstrável SEM olhar o veredito: basta ver que a mesma métrica tem
#    duas tolerâncias no mesmo arquivo.
# 2. PROCEDÊNCIA DAS DUAS RÉGUAS — CORRIGIDO EM 2026-09-30 APÓS REVISÃO
#    ADVERSARIAL. A versão anterior deste comentário afirmava que a régua de P3
#    era a consolidada e a de G11 o "default herdado". É o CONTRÁRIO, e o git
#    prova: `git show HEAD:src/promover_checkpoint.py` não contém a string
#    "MARGENS_PLANEJADO" nem "P3" (nenhum casamento), enquanto
#    TOLERANCIA_QUALIDADE_DIVERSIDADE_PP = 0.0 está lá, commitado. O 1pp de P3 é
#    trabalho NÃO COMMITADO do mesmo dia e do mesmo conjunto de mudanças que
#    precisava dele para promover; o 0pp de G11 é a regra estabelecida.
#    Portanto esta unificação NÃO se justifica por "adotar a régua mais
#    consolidada" — ela se justifica só pelo argumento 4 (0pp é mais estrito que
#    a resolução do instrumento), e quem revisar deve pesá-la por esse argumento
#    sozinho, sabendo que o valor adotado é o mais novo e o mais frouxo dos dois.
# 3. Ela não desarma o gate: G11 continua BLOQUEANTE, continua exigindo que
#    diversity_score não piore nem um décimo, e continua com 0pp em
#    json_valido_pct e schema_pct. Só as duas métricas que têm gate P
#    equivalente mudam de tolerância, e mudam para a do gate equivalente
#    (aderencia_pct passa a usar a de P4, pelo mesmo motivo).
# 4. Tolerância de 0pp com n=228 questões reprova por UMA questão: o passo
#    mínimo da métrica é 0,44pp, então "0pp" não significa "sem piora", significa
#    "nenhuma questão a mais pode falhar em 228". Era o mesmo defeito do G4
#    original — gate mais estrito que a resolução do instrumento.
#
# Vale a partir do ciclo de 2026-09-30. Vereditos anteriores a esta data foram
# decididos com a régua antiga e NÃO devem ser reinterpretados com esta.
#
# A unificação é nos DOIS eixos (tolerância e significância). Unificar só a
# tolerância recriaria o mesmo conflito um nível acima: P3 absolveria uma piora
# não significativa e G11 a reprovaria, sobre a mesma medição.
TOLERANCIA_POR_METRICA_DIVERSIDADE = {
    "consistencia_inconsistente_pct": "P3",  # tolerância de MARGENS_PLANEJADO["P3"]
    "aderencia_pct": "P4",
}


def tolerancia_diversidade(metrica, padrao=TOLERANCIA_QUALIDADE_DIVERSIDADE_PP):
    """Tolerância do G11 para uma métrica: a mesma do gate P-equivalente, quando
    existe; senão o padrão (0pp)."""
    gid = TOLERANCIA_POR_METRICA_DIVERSIDADE.get(metrica)
    return MARGENS_PLANEJADO[gid][2] if gid else padrao


def agregado_diversidade(rel, modo="ajustado"):
    """Agregado de um modo de um relatório de avaliar_diversidade.py, ou None."""
    return _get(rel or {}, "modos", modo, "agregado")


def gate_diversidade(base_rel, cand_rel, modo_base="ajustado", modo_cand="ajustado",
                     tolerancia_pp=TOLERANCIA_QUALIDADE_DIVERSIDADE_PP):
    """G11: diversity_score não pode piorar e nenhuma métrica de qualidade pode
    piorar além da tolerância da métrica. Métrica ausente reprova (mesma regra
    dos outros gates). Também reprova se os conjuntos de prompts diferirem — sem
    pareamento a comparação não mede nada.

    `tolerancia_pp` é o PADRÃO, usado nas métricas que não têm gate P
    equivalente; consistencia_inconsistente_pct e aderencia_pct usam a régua de
    P3/P4 (ver TOLERANCIA_POR_METRICA_DIVERSIDADE)."""
    g = Gate("G11", "Diversidade sem perda de qualidade")
    b, c = agregado_diversidade(base_rel, modo_base), agregado_diversidade(cand_rel, modo_cand)
    if b is None or c is None:
        return g.resolve(False, f"agregado ausente (baseline modo={modo_base}: {b is not None}, "
                                f"candidato modo={modo_cand}: {c is not None})")
    ib, ic = base_rel.get("prompt_ids"), cand_rel.get("prompt_ids")
    if ib is not None and ic is not None and ib != ic:
        return g.resolve(False, "conjuntos de prompts diferentes — comparação não pareada")
    # Mesma RÉGUA e mesmo pipeline. Revisão adversarial de 2026-10-01: com a
    # taxonomia nova, as MESMAS questões gravadas perderam até 0,033 de
    # diversity_score por lote (agregado -0,0014 a -0,0056, dez vezes o ganho
    # medido no dry-run); com tolerância zero, um baseline antigo reprovaria o
    # G11 sem o modelo ter mudado. Relatório sem os campos (anterior a esta
    # data) só compara com outro também sem eles: a falta de um lado reprova.
    rb, rc = base_rel.get("regua_sha256"), cand_rel.get("regua_sha256")
    if (rb or rc) and rb != rc:
        return g.resolve(False, f"régua de diversidade diferente (baseline {str(rb)[:12]}, "
                                f"candidato {str(rc)[:12]}): regere o baseline com o pipeline atual")
    mb, mc = base_rel.get("modo_geometria"), cand_rel.get("modo_geometria")
    if (mb or mc) and mb != mc:
        return g.resolve(False, f"modo_geometria diferente (baseline {mb}, candidato {mc}): "
                                "rode os dois braços com o mesmo --geometria")
    problemas, detalhes = [], []
    for chave, sentido in QUALIDADE_DIVERSIDADE:
        vb, vc = b.get(chave), c.get(chave)
        if vb is None or vc is None:
            # aderência pode faltar legitimamente se só houver habilidades K=1
            # nos dois lados; nos demais casos é métrica ausente.
            if vb is None and vc is None and chave == "aderencia_pct":
                continue
            problemas.append(f"{chave} ausente")
            continue
        tol = tolerancia_diversidade(chave, tolerancia_pp)
        detalhes.append(f"{chave} {vb}->{vc}"
                        + _ic_str(base_rel, cand_rel, modo_cand, chave, modo_base))
        # Mesma régua de P3/P4: só reprova piora acima da tolerância E
        # significativa. Modos diferentes nos dois lados não são pareáveis
        # (prompts iguais, pipelines diferentes), então aí só vale a tolerância.
        reprova, teste, estourou = _piorou_de_verdade(
            base_rel if modo_base == modo_cand else None,
            cand_rel, modo_cand, chave, vb, vc, sentido, tol)
        if teste and estourou:
            detalhes.append(f"{chave}: {teste}")
        if reprova:
            problemas.append(f"{chave} piorou {vb}->{vc} (tolerância {tol}pp"
                             + (f"; {teste}" if teste else "") + ")")
    db, dc = b.get("diversity_score"), c.get("diversity_score")
    if db is None or dc is None:
        problemas.append("diversity_score ausente")
    else:
        detalhes.insert(0, f"diversity_score {db}->{dc}")
        if dc < db - 1e-9:
            problemas.append(f"diversity_score piorou {db}->{dc}")
    return g.resolve(not problemas, "; ".join(problemas) if problemas else "; ".join(detalhes))


def ganho_diversidade(base_rel, cand_rel, modo_base="ajustado", modo_cand="ajustado"):
    """Lista (0 ou 1 item) de ganho estrito de diversity_score, para houve_ganho.
    Sem isso, um candidato que só melhora diversidade daria 'NÃO PROMOVIDO'."""
    b = agregado_diversidade(base_rel, modo_base) or {}
    c = agregado_diversidade(cand_rel, modo_cand) or {}
    vb, vc = b.get("diversity_score"), c.get("diversity_score")
    if vb is not None and vc is not None and vc > vb:
        return [f"diversity_score: {vb} -> {vc}"]
    return []


# --------------------------------------------------------------------------
# Perfil "planejado" — gates no modo que o app usa (decidido em 2026-09-30)
# --------------------------------------------------------------------------
# O app pede N questões e as gera em N chamadas de 1 questão, cada uma com o
# sufixo do plano (Subtema/Tipo de raciocínio/Contexto) — ver gerar_lote.py.
# O prompt "Gere 1 questão" SEM sufixo, medido por G1–G8 em val_frozen_v1,
# nunca é usado em produção; nesse perfil G1–G8 viram informativos e os gates
# bloqueantes P1–P7 + G11 são calculados sobre relatórios de
# avaliar_diversidade.py (modo "ajustado"), com várias seeds, pareados.

MARGENS_PLANEJADO = {
    # (métrica do agregado, sentido, tolerância vs baseline, limite absoluto)
    # O limite absoluto segue o sentido da métrica: com sentido +1 é um PISO
    # (vc >= limite), com sentido -1 é um TETO (vc <= limite).
    "P1": ("json_valido_pct", 1, 0.0, 99.0),
    "P2": ("quantidade_entregue_pct", 1, 0.0, 100.0),
    # TETO de 2% em P3, decidido em 2026-09-30 ANTES da rodada de 32 prompts que
    # o aplica pela primeira vez (pré-registro em Doc/PLANO_MELHORIA_GATES.md §6).
    # Motivo: a tolerância relativa (+1pp) protege contra piorar em relação ao
    # baseline, mas não contra os DOIS modelos serem ruins ao mesmo tempo. Para
    # o piloto com alunos o que importa é a taxa entregue: 2% = no máximo 1
    # gabarito errado em 50 questões, seja qual for o baseline. Simulação com
    # taxa real de 1–1,3% e n=456: tolerância 0pp reprova o MESMO modelo por
    # acaso em ~43% das rodadas (passo mínimo da métrica = 1 questão); 1pp +
    # teto 2% reprova por acaso em 7–12% e detecta uma piora real para o dobro.
    "P3": ("consistencia_inconsistente_pct", -1, 1.0, 2.0),
    "P4": ("aderencia_pct", 1, 2.0, None),
    "P5": ("depende_de_visual_pct", -1, 1.0, None),
    "P6": ("difficulty_correta_pct", 1, 5.0, None),
}
NOMES_PLANEJADO = {
    "P1": "JSON válido (>=99% e >= baseline)",
    "P2": "Entrega todas as N questões",
    "P3": "Gabarito inconsistente (<= baseline + 1pp e <= 2%)",
    "P4": "Aderência à habilidade (>= baseline - 2pp)",
    "P5": "Visual ausente (<= baseline + 1pp)",
    "P6": "Dificuldade correta (>= baseline - 5pp)",
}
TOLERANCIA_TEMPO_PLANEJADO = 0.20  # +20% no tempo por questão (informativo)


def _seed_rel(rel, path):
    import re
    if rel.get("seed_offset") is not None:
        return int(rel["seed_offset"])
    m = re.search(r"_s(\d+)\.json$", str(path))
    return int(m.group(1)) if m else 0


def junta_planejado(paths, modo="ajustado"):
    """Junta os lotes de vários relatórios (seeds) de um mesmo modelo/modo.
    Retorna (relatorio_sintetico, chave_de_pareamento)."""
    from avaliar_diversidade import agregar
    lotes, chave, sha = [], [], set()
    for p in paths:
        rel = json.loads(Path(p).read_text(encoding="utf-8"))
        m = _get(rel, "modos", modo)
        if not m or "lotes" not in m:
            raise SystemExit(f"{p}: modo '{modo}' ausente")
        seed = _seed_rel(rel, p)
        # marca a seed em cada lote: é a 1ª metade da chave de pareamento
        # (seed, prompt_id, índice) usada pelo teste de McNemar do perfil.
        for l in m["lotes"]:
            l["seed_pareamento"] = seed
        lotes += m["lotes"]
        chave += [(seed, i) for i in rel.get("prompt_ids", [])]
        sha.add(rel.get("artefato_sha256"))
        tent = rel.get("max_tentativas_diversidade")
    if len(sha) != 1:
        raise SystemExit(f"relatórios de artefatos diferentes misturados: {sha}")
    return ({"artefato_sha256": sha.pop(), "prompt_ids": sorted(chave),
             "max_tentativas_diversidade": tent,
             "modos": {modo: {"lotes": lotes, "agregado": agregar(lotes)}}}, sorted(chave))


def _contagem_ano(lotes, ano):
    ls = [l for l in lotes if l["ano"] == ano]
    return sum(l["schema_ok"] for l in ls), sum(l["quantidade_gerada"] for l in ls)


def _pct_ano(lotes, ano):
    k, n = _contagem_ano(lotes, ano)
    return round(100.0 * k / n, 2) if n else None


# Gates cuja reprovação por PIORA exige significância estatística além da
# tolerância. São os dois cujo estado é decidível questão a questão e cujo n é
# pequeno o bastante para uma única questão mover o percentual mais que a
# tolerância (ver PREDICADO_PAREADO e o comentário da revisão de 2026-09-30).
# P1/P2 não entram: são pisos absolutos de contrato, não comparações com o
# baseline. P5/P6 não entram por falta de veredito por questão nos relatórios.
GATES_COM_SIGNIFICANCIA = ("P3", "P4")


def avalia_planejado(base_rel, cand_rel, modo="ajustado"):
    gates = []
    b, c = agregado_diversidade(base_rel, modo), agregado_diversidade(cand_rel, modo)
    for gid, (met, sentido, tol, piso) in MARGENS_PLANEJADO.items():
        vb, vc = b.get(met), c.get(met)
        g = Gate(gid, NOMES_PLANEJADO[gid])
        if vb is None or vc is None:
            gates.append(g.resolve(False, f"{met} ausente"))
            continue
        detalhe = f"{met} {vb} -> {vc}" + _ic_str(base_rel, cand_rel, modo, met)
        if gid in GATES_COM_SIGNIFICANCIA:
            piorou, teste, _ = _piorou_de_verdade(base_rel, cand_rel, modo, met,
                                                  vb, vc, sentido, tol)
            for linha in (teste, nota_regua(base_rel, cand_rel, modo, met, vb, vc)):
                if linha:
                    detalhe += f"\n            {linha}"
        else:
            piorou = (vc - vb) * sentido < -tol - 1e-9
        # O limite absoluto NÃO é relaxado por significância: é contrato, não
        # comparação. Só a piora relativa ao baseline passa pelo teste.
        dentro_limite = piso is None or (vc >= piso if sentido > 0 else vc <= piso)
        if not dentro_limite:
            detalhe += f"\n            fora do limite absoluto ({'>=' if sentido > 0 else '<='} {piso}%)"
        ok = (not piorou) and dentro_limite
        gates.append(g.resolve(ok, detalhe))
    # P7 — esquecimento nos anos em risco, no modo real
    lb, lc = base_rel["modos"][modo]["lotes"], cand_rel["modos"][modo]["lotes"]
    regr = []
    for ano in ANOS_EM_RISCO:
        vb, vc = _pct_ano(lb, ano), _pct_ano(lc, ano)
        if vb is not None and vc is not None and vc < vb - 1e-9:
            ib, ic = wilson(*_contagem_ano(lb, ano)), wilson(*_contagem_ano(lc, ano))
            regr.append(f"{ano}: schema {vb}->{vc}%  [IC95 {ib[0]}–{ib[1]}% -> {ic[0]}–{ic[1]}%]")
    gates.append(Gate("P7", f"Sem esquecimento em {'/'.join(ANOS_EM_RISCO)}").resolve(
        not regr, "; ".join(regr) or "schema preservado"))
    # P8 — tempo (informativo: qualidade tem prioridade sobre tempo)
    tb, tc = b.get("tempo_por_questao_s"), c.get("tempo_por_questao_s")
    if tb and tc:
        gates.append(Gate("P8", "Tempo por questão (<= baseline + 20%)", bloqueante=False).resolve(
            tc <= tb * (1 + TOLERANCIA_TEMPO_PLANEJADO), f"{tb} -> {tc} s ({100 * (tc - tb) / tb:+.1f}%)"))
    return gates


def houve_ganho_planejado(base_rel, cand_rel, modo="ajustado"):
    b, c = agregado_diversidade(base_rel, modo), agregado_diversidade(cand_rel, modo)
    ganhos = []
    for nome, met, sentido in (("aderência", "aderencia_pct", 1),
                               ("inconsistentes", "consistencia_inconsistente_pct", -1),
                               ("cobertura", "coverage_score", 1),
                               ("duplicatas", "duplicate_rate", -1)):
        vb, vc = b.get(met), c.get(met)
        if vb is not None and vc is not None and (vc - vb) * sentido > 0:
            ganhos.append(f"{nome}: {vb} -> {vc}")
    return ganhos + ganho_diversidade(base_rel, cand_rel, modo, modo)


# --------------------------------------------------------------------------
# Perfil MULTISEED (G2/G3 agregados sobre várias rodadas de seeds)
# --------------------------------------------------------------------------
# REVISÃO DE 2026-10-05, registrada como feita DEPOIS de um veredito reprovado
# (VEREDITO_v3.json: G2 e G3 reprovados), como já foi feito com o G4 em
# 2026-09-16 e com o G11 em 2026-09-30.
#
# O DEFEITO: com n=30 e uma única rodada de seeds, G2 ("falhas = 0") e G3
# ("consistência >= baseline - 5pp", sobre ~10 itens verificáveis) decidem
# sobre UMA amostra. No veredito da v3 os dois gates reprovaram pelo MESMO item
# (9º H06, MT9027MH06MT): 1 falha = G2 reprovado; 1 inconsistência em 10
# verificáveis = 10pp = G3 reprovado. Regerar esse prompt 4 vezes com a v3 deu
# 4/4 aprovadas — o instrumento não distingue esse resultado de ruído.
#
# REGRA (fixada ANTES de rodar as seeds novas):
#   * baseline e candidato rodam as MESMAS rodadas de seed (test_model.py
#     --seed-rodada r), pareadas por (rodada, codigo_item_ref);
#   * G2* reprova se a piora de falhas for significativa (McNemar exato,
#     p < 0,05, com mais pares piorados que melhorados) OU se a taxa agregada
#     de falha do candidato passar de TETO_FALHA_MULTISEED (guarda absoluta:
#     não basta empatar com um baseline ruim);
#   * G3* reprova se a piora de inconsistência gabarito<->conta for
#     significativa (mesmo teste);
#   * todos os OUTROS gates bloqueantes (G1, G4–G8) continuam com a régua
#     original e precisam passar em TODAS as rodadas — a mudança não afrouxa
#     nada além de G2/G3.
#
# REVISÃO DE 2026-10-09 (G6*), registrada como feita DEPOIS de um veredito
# reprovado (VEREDITO_inferencia_server.json, --comparar-inferencia: llama-cli
# 4 threads x llama-server + cache de prompt + 8 threads, mesmo .gguf v3).
# O DEFEITO é o mesmo de G2/G3: G6 ("letra mais frequente <= baseline + 5pp")
# decidia em CADA rodada de 30 itens, onde 2 respostas = 6,7pp. Reprovou só na
# rodada 0 (36,7% -> 43,3%: B 11 -> 13); na rodada 1 o candidato MELHOROU
# 6,6pp, e somadas as 90 amostras foi 30,0% -> 31,1%.
# REGRA: G6* aplica a MESMA tolerância (+5pp) ao viés agregado das rodadas
# (_agregado()["vies_pct"], contagem de letras somada) em vez de exigi-la por
# rodada. G1, G4, G5, G7 e G8 continuam por rodada.
TETO_FALHA_MULTISEED = 0.05
# Com 1 rodada o perfil seria só a régua frouxa sobre os MESMOS 30 itens; o
# ganho de resolução vem das rodadas novas. Mínimo fixado junto com a regra.
MIN_RODADAS_MULTISEED = 3


def _rodada(rel):
    return rel.get("seed_rodada", 0)


def _juntar_detalhes(rels):
    """Relatório-pseudo com detalhes de todas as rodadas, chave (rodada, item)."""
    return {"detalhes": [{**x, "codigo_item_ref": f"{_rodada(r)}:{x['codigo_item_ref']}"}
                         for r in rels for x in r.get("detalhes", [])]}


def _agregado(rels):
    det = [x for r in rels for x in r.get("detalhes", [])]
    verif = [x for x in det if x.get("consistencia_resposta_correta") is not None]
    ader = [x for x in det if x.get("difficulty_aderente") is not None]
    letras = Counter()
    for r in rels:
        letras.update(_get(r, "estrutura", "distribuicao_respostas_corretas", default={}) or {})
    return {
        "n": len(det),
        "falhas": sum(x.get("status") == "falha" for x in det),
        "verificaveis": len(verif),
        "consistencia_pct": round(100 * sum(bool(x["consistencia_resposta_correta"]) for x in verif)
                                  / len(verif), 1) if verif else None,
        "aderencia_pct": round(100 * sum(bool(x["difficulty_aderente"]) for x in ader)
                               / len(ader), 1) if ader else None,
        "regeneracoes": sum(_get(r, "pos_processamento", "regeneracoes_total", default=0) or 0
                            for r in rels),
        "vies_pct": round(100 * max(letras.values()) / sum(letras.values()), 1) if letras else None,
    }


def avalia_multiseed(bases, cands, comparar_inferencia=False):
    """Retorna (gates, ganhos, agregado_base, agregado_cand). Ver bloco acima.

    comparar_inferencia (2026-10-09): compara CONFIGURAÇÕES de inferência do
    MESMO .gguf (motor, threads, cache de prompt), que mudam o texto gerado por
    arredondamento. Inverte a trava de sha256 (tem de ser o mesmo artefato e
    report["inferencia"] tem de diferir) e conta latência como ganho. Os gates
    bloqueantes não mudam."""
    rb = sorted(_rodada(r) for r in bases)
    rc = sorted(_rodada(r) for r in cands)
    if len(rc) < MIN_RODADAS_MULTISEED:
        raise SystemExit(f"ABORTADO: perfil multiseed exige >= {MIN_RODADAS_MULTISEED} rodadas "
                         f"de seed (recebeu {len(rc)})")
    if rb != rc or len(set(rb)) != len(rb):
        raise SystemExit(f"ABORTADO: rodadas de seed não pareadas (baseline {rb}, candidato {rc})")
    pares = sorted(zip(sorted(bases, key=_rodada), sorted(cands, key=_rodada)),
                   key=lambda p: _rodada(p[0]))
    for b, c in pares:
        hb, hc = b.get("artefato_sha256"), c.get("artefato_sha256")
        if comparar_inferencia:
            if not (hb and hb == hc):
                raise SystemExit(f"ABORTADO: --comparar-inferencia exige o MESMO sha256 "
                                 f"(rodada {_rodada(b)}: {hb} x {hc}).")
            if not (b.get("inferencia") and c.get("inferencia")) or b["inferencia"] == c["inferencia"]:
                raise SystemExit(f"ABORTADO: rodada {_rodada(b)} sem 'inferencia' diferente "
                                 f"nos dois lados ({b.get('inferencia')} x {c.get('inferencia')}).")
        elif hb and hc and hb == hc:
            raise SystemExit(f"ABORTADO: rodada {_rodada(b)} com o MESMO sha256 nos dois lados.")
        if Path(str(b.get("conjunto_avaliacao"))).name != Path(str(c.get("conjunto_avaliacao"))).name:
            raise SystemExit(f"ABORTADO: rodada {_rodada(b)} com conjuntos de avaliação diferentes.")
        if b.get("retries", 1) != c.get("retries", 1):
            raise SystemExit(f"ABORTADO: rodada {_rodada(b)} com regenerações diferentes.")

    por_rodada = [(_rodada(b), avalia(b, c)) for b, c in pares]
    gates = []
    for i, modelo in enumerate(por_rodada[0][1]):
        if modelo.id in ("G2", "G3", "G6"):
            continue
        falhas = [f"rodada {r}: {gs[i].detalhe}" for r, gs in por_rodada if not gs[i].passou]
        gates.append(Gate(modelo.id, modelo.nome + " [todas as rodadas]", modelo.bloqueante).resolve(
            not falhas, "; ".join(falhas) or f"passou nas {len(pares)} rodadas"))

    jb, jc = _juntar_detalhes(bases), _juntar_detalhes(cands)
    ab, ac = _agregado(bases), _agregado(cands)

    piorou, melhorou, p = _mcnemar(jb, jc, lambda x: x.get("status") == "falha")
    significativa = piorou > melhorou and p < 0.05
    taxa = ac["falhas"] / ac["n"] if ac["n"] else 1.0
    gates.insert(1, Gate("G2*", "Falhas pós best-of-N (sem piora signif., <= 5%)").resolve(
        not significativa and taxa <= TETO_FALHA_MULTISEED,
        f"{ab['falhas']}/{ab['n']} -> {ac['falhas']}/{ac['n']} ({100 * taxa:.1f}%)  "
        f"(piorou {piorou}, melhorou {melhorou}, McNemar p={p:.3f})"))

    piorou, melhorou, p = _mcnemar(jb, jc, lambda x: x.get("consistencia_resposta_correta") is False)
    gates.insert(2, Gate("G3*", "Consistência (sem piora significativa)").resolve(
        not (piorou > melhorou and p < 0.05),
        f"{ab['consistencia_pct']}% (n={ab['verificaveis']}) -> {ac['consistencia_pct']}% "
        f"(n={ac['verificaveis']})  (piorou {piorou}, melhorou {melhorou}, McNemar p={p:.3f})"))

    vb, vc = ab["vies_pct"], ac["vies_pct"]
    gates.insert(3, Gate("G6*", "Viés de gabarito agregado (<= baseline + 5pp)").resolve(
        vb is not None and vc is not None and vc <= vb + 5,
        f"letra mais frequente {vb}% -> {vc}% (n={ac['n']})"))

    ganhos = []
    for nome, chave, sentido in (("consistência", "consistencia_pct", 1),
                                 ("aderência à dificuldade", "aderencia_pct", 1),
                                 ("verificabilidade (n)", "verificaveis", 1),
                                 ("viés de gabarito", "vies_pct", -1),
                                 ("falhas", "falhas", -1),
                                 ("regenerações", "regeneracoes", -1)):
        vb, vc = ab.get(chave), ac.get(chave)
        if nome == "regenerações" and any(
                _modo_geometria(b) != _modo_geometria(c) for b, c in pares):
            continue
        if vb is not None and vc is not None and (vc - vb) * sentido > 0:
            ganhos.append(f"{nome}: {vb} -> {vc}")
    if comparar_inferencia:
        lat = lambda rels: statistics.mean(
            _get(r, "velocidade_cpu_real", "latencia_media_total_s") for r in rels)
        lb, lc = lat(bases), lat(cands)
        if lc < lb:
            ganhos.append(f"latência média por item: {lb:.2f}s -> {lc:.2f}s "
                          f"({100 * (lc - lb) / lb:+.1f}%)")
    return gates, ganhos, ab, ac


def _observacoes(bases, cands):
    """OBSERVAÇÕES informativas (P0-4; src/observacoes_gate.py): impressas e
    gravadas em `observacoes`, mas FORA do veredito — nunca alimentam
    `reprovados`, `ganhos` nem o PROMOVIDO / NÃO PROMOVIDO. Para virarem
    critério é preciso um veredito reprovado e a mudança documentada aqui."""
    import observacoes_gate as og  # import tardio, como diversidade (src/ no path)
    obs = og.calcular_seguro(bases, cands)
    print()
    for linha in og.formatar(obs):
        print(linha)
    return obs


def main_multiseed(args, carrega):
    if len(args.baseline_gguf_seeds) != len(args.candidato_gguf_seeds):
        raise SystemExit("perfil multiseed exige o MESMO número de relatórios nos dois lados")
    bases = [carrega(p) for p in args.baseline_gguf_seeds]
    cands = [carrega(p) for p in args.candidato_gguf_seeds]
    gates, ganhos, ab, ac = avalia_multiseed(bases, cands, args.comparar_inferencia)

    print("=" * 78)
    print("GATES DE PROMOÇÃO — MULTISEED, conjunto congelado, comparação pareada")
    print("=" * 78)
    print(f"  baseline:  {bases[0].get('artefato')}")
    print(f"  candidato: {cands[0].get('artefato')}")
    print(f"  rodadas:   {sorted(_rodada(r) for r in cands)}  "
          f"(amostras pareadas: {ac['n']})\n")
    for g in gates:
        marca = "PASSOU" if g.passou else "FALHOU"
        tipo = "bloqueante" if g.bloqueante else "informativo"
        print(f"  [{marca:^6}] {g.id} {g.nome:<46} ({tipo})")
        print(f"            {g.detalhe}")
    reprovados = [g for g in gates if g.bloqueante and not g.passou]
    if reprovados:
        veredito, motivo = "NÃO PROMOVIDO", "gates bloqueantes reprovados: " + ", ".join(g.id for g in reprovados)
    elif not ganhos:
        veredito, motivo = "NÃO PROMOVIDO", "sem ganho mensurável — empate técnico"
    else:
        veredito, motivo = "PROMOVIDO", "todos os gates bloqueantes passaram; ganhos: " + "; ".join(ganhos)
    print("\n" + "-" * 78 + f"\nDECISÃO: {veredito}\nMOTIVO:  {motivo}\n" + "-" * 78)
    obs = _observacoes(bases, cands)
    if args.saida:
        Path(args.saida).write_text(json.dumps({
            "decisao": veredito, "motivo": motivo, "perfil": "multiseed",
            "observacoes": obs,  # informativo (P0-4): não entra no veredito
            "baseline": bases[0].get("artefato"), "candidato": cands[0].get("artefato"),
            "conjunto_avaliacao": cands[0].get("conjunto_avaliacao"),
            "rodadas": sorted(_rodada(r) for r in cands),
            "inferencia_baseline": bases[0].get("inferencia"),
            "inferencia_candidato": cands[0].get("inferencia"),
            "agregado_baseline": ab, "agregado_candidato": ac, "ganhos": ganhos,
            "gates": [{"id": g.id, "nome": g.nome, "bloqueante": g.bloqueante,
                       "passou": g.passou, "detalhe": g.detalhe} for g in gates],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Veredito gravado em: {args.saida}")
    return 0 if veredito == "PROMOVIDO" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perfil", choices=("n1", "planejado", "multiseed"), default="n1",
                        help="n1 = gates G1–G8 bloqueantes (histórico); planejado = modo do app; "
                             "multiseed = G1–G8 com G2/G3 agregados sobre várias rodadas de seed")
    parser.add_argument("--baseline-gguf-seeds", nargs="+",
                        help="multiseed: relatórios test_model.py do baseline (1 por --seed-rodada)")
    parser.add_argument("--candidato-gguf-seeds", nargs="+")
    parser.add_argument("--comparar-inferencia", action="store_true",
                        help="multiseed: mesmo .gguf, configurações de inferência diferentes "
                             "(report['inferencia']); latência conta como ganho")
    parser.add_argument("--planejado-baseline", nargs="+",
                        help="relatórios avaliar_diversidade.py do baseline (1 por seed)")
    parser.add_argument("--planejado-candidato", nargs="+")
    parser.add_argument("--baseline-gguf")
    parser.add_argument("--candidato-gguf")
    parser.add_argument("--baseline-gpu")
    parser.add_argument("--candidato-gpu")
    parser.add_argument("--saida", help="grava o veredito neste .json")
    # G11 (opcional): relatórios de src/avaliar_diversidade.py.
    parser.add_argument("--diversidade-baseline")
    parser.add_argument("--diversidade-candidato")
    parser.add_argument("--modo-diversidade-baseline", default="ajustado",
                        help="modo do relatório baseline (atual|ajustado)")
    parser.add_argument("--modo-diversidade-candidato", default="ajustado")
    parser.add_argument("--tolerancia-diversidade-pp", type=float,
                        default=TOLERANCIA_QUALIDADE_DIVERSIDADE_PP)
    args = parser.parse_args()

    carrega = lambda p: json.loads(Path(p).read_text(encoding="utf-8")) if p else None
    if args.perfil == "planejado":
        return main_planejado(args, carrega)
    if args.perfil == "multiseed":
        if not (args.baseline_gguf_seeds and args.candidato_gguf_seeds):
            parser.error("perfil multiseed exige --baseline-gguf-seeds e --candidato-gguf-seeds")
        return main_multiseed(args, carrega)
    if not (args.baseline_gguf and args.candidato_gguf):
        parser.error("perfil n1 exige --baseline-gguf e --candidato-gguf")
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
    div_b, div_c = carrega(args.diversidade_baseline), carrega(args.diversidade_candidato)
    if (div_b is None) != (div_c is None):
        raise SystemExit("G11 exige os DOIS relatórios: --diversidade-baseline e --diversidade-candidato")
    if div_b is not None:
        md_b, md_c = args.modo_diversidade_baseline, args.modo_diversidade_candidato
        gates.append(gate_diversidade(div_b, div_c, md_b, md_c, args.tolerancia_diversidade_pp))
        ganhos += ganho_diversidade(div_b, div_c, md_b, md_c)

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
    obs = _observacoes([base], [cand])

    if args.saida:
        Path(args.saida).write_text(json.dumps({
            "decisao": veredito,
            "motivo": motivo,
            "observacoes": obs,  # informativo (P0-4): não entra no veredito
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


def main_planejado(args, carrega):
    if not (args.planejado_baseline and args.planejado_candidato):
        raise SystemExit("perfil planejado exige --planejado-baseline e --planejado-candidato")
    modo = args.modo_diversidade_candidato
    base, kb = junta_planejado(args.planejado_baseline, modo)
    cand, kc = junta_planejado(args.planejado_candidato, modo)
    if base["artefato_sha256"] == cand["artefato_sha256"]:
        raise SystemExit("ABORTADO: baseline e candidato têm o MESMO sha256.")
    if kb != kc:
        raise SystemExit("ABORTADO: prompts/seeds diferentes entre baseline e candidato — não pareado.")
    if base.get("max_tentativas_diversidade") != cand.get("max_tentativas_diversidade"):
        print(f"AVISO: regenerações diferentes (baseline {base.get('max_tentativas_diversidade')}, "
              f"candidato {cand.get('max_tentativas_diversidade')})\n")

    gates = avalia_planejado(base, cand, modo)
    gates.append(gate_diversidade(base, cand, modo, modo, args.tolerancia_diversidade_pp))
    ganhos = houve_ganho_planejado(base, cand, modo)
    # G1–G8 (N=1 sem sufixo) só como informação, se os relatórios vierem.
    if args.baseline_gguf and args.candidato_gguf:
        for g in avalia(carrega(args.baseline_gguf), carrega(args.candidato_gguf)):
            g.bloqueante = False
            g.nome += " [N=1 sem sufixo]"
            gates.append(g)

    print("=" * 78)
    print("GATES DE PROMOÇÃO — perfil PLANEJADO (N chamadas com plano), pareado")
    print("=" * 78)
    print(f"  baseline:  sha {base['artefato_sha256']}")
    print(f"  candidato: sha {cand['artefato_sha256']}")
    print(f"  pares prompt×seed: {len(kc)}\n")
    for g in gates:
        marca = "PASSOU" if g.passou else "FALHOU"
        tipo = "bloqueante" if g.bloqueante else "informativo"
        print(f"  [{marca:^6}] {g.id} {g.nome:<46} ({tipo})")
        print(f"            {g.detalhe}")
    reprovados = [g for g in gates if g.bloqueante and not g.passou]
    if reprovados:
        veredito, motivo = "NÃO PROMOVIDO", "gates bloqueantes reprovados: " + ", ".join(g.id for g in reprovados)
    elif not ganhos:
        veredito, motivo = "NÃO PROMOVIDO", "sem ganho mensurável — empate técnico"
    else:
        veredito, motivo = "PROMOVIDO", "todos os gates bloqueantes passaram; ganhos: " + "; ".join(ganhos)
    print("\n" + "-" * 78 + f"\nDECISÃO: {veredito}\nMOTIVO:  {motivo}\n" + "-" * 78)
    if args.saida:
        Path(args.saida).write_text(json.dumps({
            "decisao": veredito, "motivo": motivo, "perfil": "planejado",
            "baseline_sha256": base["artefato_sha256"], "candidato_sha256": cand["artefato_sha256"],
            "max_tentativas_diversidade": cand.get("max_tentativas_diversidade"),
            "pares": len(kc), "ganhos": ganhos,
            "gates": [{"id": g.id, "nome": g.nome, "bloqueante": g.bloqueante,
                       "passou": g.passou, "detalhe": g.detalhe} for g in gates],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Veredito gravado em: {args.saida}")
    return 0 if veredito == "PROMOVIDO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
