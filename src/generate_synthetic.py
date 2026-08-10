"""Gera questões sintéticas de operações matemáticas básicas para aumentar o
dataset de treino com gabarito 100% garantido por computação em Python — não
por um LLM. As ~300 questões reais de DB/questoes.db ensinam formato/estilo,
mas são poucas para o modelo generalizar cálculo; aqui cada resposta correta é
calculada antes de montar a questão, então serve de reforço "limpo" de
aritmética, complementando (não substituindo) o dataset real.

Cada questão tem 5 alternativas (A-E), no contrato exigido pelo app mobile: o
distrator não é um número aleatório, é o resultado de um ERRO PEDAGÓGICO
específico (operação trocada, erro de vai-um, erro de vírgula, confundir
desconto com valor final...) — só a justificativa da alternativa CORRETA vira
`resolucao_passo_a_passo` (o contrato não tem mais um campo de justificativa
por alternativa).

As tuplas (ano, habilidade, descricao_item, dificuldade) usadas são as mesmas
que existem de verdade em DB/questoes.db, para o prompt de treino continuar
idêntico ao que o app manda em produção (USER_TEMPLATE). Uma fração dos
exemplos agrupa várias questões num único `{"questoes": [...]}` (ver
TAMANHOS_LOTE), ensinando o modelo a devolver lotes quando o usuário pedir N
questões de uma vez.

Escopo: adição, subtração, multiplicação, divisão (exata), porcentagem,
potenciação e frações (representação pictórica em texto, equivalência e
conversão fração→porcentagem — H07/H08/H09 do 9º ano).

Uso:
    python src/generate_synthetic.py                     # gera e mescla em data/train.jsonl
    python src/generate_synthetic.py --num-per-skill 20   # mais exemplos por habilidade/dificuldade
    python src/generate_synthetic.py --dry-run            # só mostra estatísticas, não escreve nada

Reexecutar é seguro: exemplos sintéticos anteriores (meta.sintetico=true) são
removidos do train.jsonl antes de gerar e escrever o novo lote.
"""

import argparse
import json
import math
import random
from pathlib import Path

from extract_data import ALTERNATIVA_E_PADRAO, SYSTEM_PROMPT, USER_TEMPLATE
from schema_utils import (
    ALTERNATIVE_LETTERS,
    DIFFICULTY_MAP,
    _UNICODE_MATH,
    check_consistency,
    check_structure,
    extract_questoes,
)

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "train.jsonl"
DEFAULT_NUM_PER_SKILL = 12
# Distribuição de tamanho de lote por exemplo de treino: a maioria pede 1
# questão (uso mais comum), mas uma fração pede várias — ensina o modelo a
# preencher "questoes" com N itens quando o usuário pedir um lote.
TAMANHOS_LOTE = [1, 1, 1, 1, 1, 1, 2, 3, 5]

NOMES = ["Ana", "Bruno", "Carla", "Diego", "Elisa", "Felipe",
         "Gabriela", "Hugo", "Isabela", "João"]
CONTEXTOS = ["bolinhas de gude", "figurinhas", "reais", "livros", "doces",
             "laranjas", "lápis", "cadernos", "bonecos", "adesivos"]


def fmt(n):
    """Formata número no padrão brasileiro (vírgula decimal)."""
    if isinstance(n, int):
        return str(n)
    if float(n).is_integer():
        return str(int(n))
    return f"{n:.1f}".replace(".", ",")


def _norm(v, is_integer):
    return int(round(v)) if is_integer else round(v, 1)


def _montar_questao(enunciado, comando, alternativas, gabarito, resolucao, dificuldade):
    """Monta o dict de questão no contrato do app (enunciado+comando fundidos,
    5 alternativas, resposta_correta = LETRA, resolucao_passo_a_passo = texto
    único, difficulty em inglês)."""
    enunciado_completo = f"{enunciado} {comando}".strip() if comando else enunciado
    return {
        "enunciado": enunciado_completo,
        "alternativas": alternativas,
        "resolucao_passo_a_passo": resolucao,
        "resposta_correta": gabarito,
        "difficulty": DIFFICULTY_MAP.get(dificuldade, "MEDIUM"),
    }


def montar(rng, enunciado, comando, itens, dificuldade, is_integer=True):
    """Monta a questão a partir de `itens`: lista de (valor, justificativa),
    onde o PRIMEIRO item é o correto e os demais são distratores pedagógicos.
    A justificativa do correto vira `resolucao_passo_a_passo`; as dos
    distratores só servem para gerar valores plausíveis e distintos (o
    contrato não tem mais um campo de justificativa por alternativa).

    Deduplica valores, completa até 5 alternativas (A-E) se algum distrator
    colidir, embaralha as posições.
    """
    correto, justificativa_correta = itens[0]
    correto = _norm(correto, is_integer)

    finais = [(correto, justificativa_correta)]
    vistos = {correto}
    for valor, justificativa in itens[1:]:
        valor = _norm(valor, is_integer)
        if valor not in vistos and valor >= 0:
            finais.append((valor, justificativa))
            vistos.add(valor)
        if len(finais) == len(ALTERNATIVE_LETTERS):
            break

    # Completa com erros de estimativa caso algum distrator tenha colidido.
    step, i = (3 if is_integer else 0.5), 1
    while len(finais) < len(ALTERNATIVE_LETTERS):
        for candidato in (correto + i * step, correto - i * step):
            candidato = _norm(candidato, is_integer)
            if candidato not in vistos and candidato >= 0 and len(finais) < len(ALTERNATIVE_LETTERS):
                finais.append((
                    candidato,
                    f"Erro de estimativa: {fmt(candidato)} não é o resultado "
                    f"correto da operação, que é {fmt(correto)}.",
                ))
                vistos.add(candidato)
        i += 1

    ordem = list(range(len(ALTERNATIVE_LETTERS)))
    rng.shuffle(ordem)
    alternativas = {ALTERNATIVE_LETTERS[pos]: fmt(finais[idx][0]) for pos, idx in enumerate(ordem)}
    gabarito = ALTERNATIVE_LETTERS[ordem.index(0)]

    return _montar_questao(enunciado, comando, alternativas, gabarito, justificativa_correta, dificuldade)


def montar_texto(rng, enunciado, comando, itens, dificuldade):
    """Como `montar()`, mas para alternativas que já são texto pronto (frações
    como "3/8", porcentagens como "6%") — sem a normalização/estimativa
    numérica de `montar()`, que assume valores escalares.

    `itens`: lista de (texto, justificativa) com o item [0] sempre correto.
    Cada gerador que usa esta função é responsável por produzir pelo menos 4
    distratores textualmente distintos do correto e entre si (5 alternativas
    no total).
    """
    correto, justificativa_correta = itens[0]
    finais, vistos = [(correto, justificativa_correta)], {correto}
    for texto, justificativa in itens[1:]:
        if texto not in vistos:
            finais.append((texto, justificativa))
            vistos.add(texto)
        if len(finais) == len(ALTERNATIVE_LETTERS):
            break
    if len(finais) < len(ALTERNATIVE_LETTERS):
        raise ValueError(
            f"gerador não produziu {len(ALTERNATIVE_LETTERS)} alternativas "
            f"textuais distintas: {[f[0] for f in finais]}"
        )

    ordem = list(range(len(ALTERNATIVE_LETTERS)))
    rng.shuffle(ordem)
    alternativas = {ALTERNATIVE_LETTERS[pos]: finais[idx][0] for pos, idx in enumerate(ordem)}
    gabarito = ALTERNATIVE_LETTERS[ordem.index(0)]

    return _montar_questao(enunciado, comando, alternativas, gabarito, justificativa_correta, dificuldade)


# ---- Geradores de questão ----------------------------------------------------
# Cada gerador devolve o dict da questão. O primeiro item da lista é sempre o
# correto; os demais são distratores derivados de erros pedagógicos reais.

def gen_calc_soma_subtracao(rng, faixa, dificuldade):
    a, b = rng.randint(*faixa), rng.randint(*faixa)
    op = rng.choice(["+", "-"])
    if op == "-" and b > a:
        a, b = b, a
    if op == "+":
        correto, trocada = a + b, a - b
        nome_op, nome_trocada = "adição", "subtração"
    else:
        correto, trocada = a - b, a + b
        nome_op, nome_trocada = "subtração", "adição"
    vai_um = correto + rng.choice([-10, 10])
    itens = [
        (correto, f"{a} {op} {b} = {correto}."),
        (trocada, f"Erro de operação: faz a {nome_trocada} em vez da {nome_op} "
                  f"e obtém {fmt(abs(trocada))}."),
        (vai_um, f"Erro de reagrupamento (vai-um): obtém {fmt(vai_um)} em vez de {correto}."),
        (correto + rng.choice([-1, 1]), "Erro de contagem de uma unidade."),
    ]
    return montar(
        rng,
        f"Calcule o resultado da operação: {a} {op} {b}.",
        "Qual é o resultado?",
        itens,
        dificuldade,
    )


def gen_problema_soma_subtracao(rng, faixa, dificuldade):
    a, b = rng.randint(*faixa), rng.randint(*faixa)
    nome, contexto = rng.choice(NOMES), rng.choice(CONTEXTOS)
    if rng.random() < 0.5:
        correto, trocada = a + b, abs(a - b)
        enunciado = f"{nome} tinha {a} {contexto}. Ganhou mais {b} {contexto}."
        justificativa = f"Juntando: {a} + {b} = {correto} {contexto}."
        erro_op = (f"Erro: subtrai em vez de somar ({max(a, b)} - {min(a, b)} = "
                   f"{abs(a - b)}), mas {nome} ganhou, não perdeu.")
    else:
        if b > a:
            a, b = b, a
        correto, trocada = a - b, a + b
        enunciado = f"{nome} tinha {a} {contexto}. Perdeu {b} {contexto}."
        justificativa = f"Retirando: {a} - {b} = {correto} {contexto}."
        erro_op = (f"Erro: soma em vez de subtrair ({a} + {b} = {a + b}), "
                   f"mas {nome} perdeu, não ganhou.")
    itens = [
        (correto, justificativa),
        (trocada, erro_op),
        (correto + rng.choice([-10, 10]), "Erro de reagrupamento (vai-um) no cálculo."),
        (correto + rng.choice([-1, 1]), "Erro de contagem de uma unidade."),
    ]
    return montar(rng, enunciado, f"Quantos {contexto} {nome} tem agora?", itens, dificuldade)


def gen_calc_mult_div(rng, faixas, dificuldade):
    faixa_fator1, faixa_fator2 = faixas
    if rng.random() < 0.5:
        a, b = rng.randint(*faixa_fator1), rng.randint(*faixa_fator2)
        correto = a * b
        itens = [
            (correto, f"{a} x {b} = {correto}."),
            (a + b, f"Erro de operação: soma em vez de multiplicar ({a} + {b} = {a + b})."),
            (a * (b + 1), f"Erro de tabuada: multiplica por {b + 1} em vez de {b}."),
            (correto - a, f"Erro: conta um grupo de {a} a menos."),
        ]
        enunciado = f"Calcule o resultado da operação: {a} x {b}."
    else:
        quociente = rng.randint(*faixa_fator1)
        b = max(2, rng.randint(*faixa_fator2))
        a = b * quociente
        correto = quociente
        itens = [
            (correto, f"{a} ÷ {b} = {correto}."),
            (correto + 1, f"Erro na divisão: obtém {correto + 1}, mas {b} x {correto + 1} = "
                          f"{b * (correto + 1)}, que passa de {a}."),
            (correto - 1, f"Erro na divisão: obtém {correto - 1}, mas {b} x {correto - 1} = "
                          f"{b * (correto - 1)}, que não chega a {a}."),
            (a - b, f"Erro de operação: subtrai em vez de dividir ({a} - {b} = {a - b})."),
        ]
        enunciado = f"Calcule o resultado da operação: {a} ÷ {b}."
    return montar(rng, enunciado, "Qual é o resultado?", itens, dificuldade)


def gen_problema_mult_div(rng, faixas, dificuldade):
    faixa_grupos, faixa_itens = faixas
    nome, contexto = rng.choice(NOMES), rng.choice(CONTEXTOS)
    grupos, itens_por_grupo = rng.randint(*faixa_grupos), rng.randint(*faixa_itens)
    if rng.random() < 0.5:
        correto = grupos * itens_por_grupo
        enunciado = (f"{nome} organizou {contexto} em {grupos} grupos com "
                     f"{itens_por_grupo} {contexto} cada um.")
        comando = f"Quantos {contexto} há ao todo?"
        itens = [
            (correto, f"{grupos} grupos de {itens_por_grupo}: {grupos} x {itens_por_grupo} = {correto}."),
            (grupos + itens_por_grupo, f"Erro: soma grupos com itens ({grupos} + "
                                       f"{itens_por_grupo} = {grupos + itens_por_grupo}) em vez de multiplicar."),
            (correto - itens_por_grupo, "Erro: esquece um dos grupos na contagem."),
            (correto + itens_por_grupo, "Erro: conta um grupo a mais."),
        ]
    else:
        total = grupos * itens_por_grupo
        correto = itens_por_grupo
        enunciado = f"{nome} tem {total} {contexto} e quer separar em {grupos} grupos iguais."
        comando = f"Quantos {contexto} ficam em cada grupo?"
        itens = [
            (correto, f"Repartição igual: {total} ÷ {grupos} = {correto}."),
            (total - grupos, f"Erro: subtrai em vez de dividir ({total} - {grupos} = {total - grupos})."),
            (correto + 1, f"Erro na divisão: com {correto + 1} em cada grupo seriam "
                          f"{grupos * (correto + 1)} {contexto}, mais do que {total}."),
            (correto - 1, f"Erro na divisão: com {correto - 1} em cada grupo sobrariam {contexto} sem grupo."),
        ]
    return montar(rng, enunciado, comando, itens, dificuldade)


def gen_porcentagem(rng, faixas, dificuldade):
    base_mult_faixa, percentuais = faixas
    base = rng.randint(*base_mult_faixa) * 20  # múltiplo de 20 -> % múltiplo de 5 é exato
    percentual = rng.choice(percentuais)
    correto = base * percentual // 100
    itens = [
        (correto, f"O desconto é {percentual}% de {base}: {base} x {percentual} ÷ 100 = {correto}."),
        (base - correto, f"Erro: confunde o VALOR DO DESCONTO com o valor final a pagar "
                         f"({base} - {correto} = {base - correto})."),
        (base * percentual // 10, f"Erro de casa decimal: divide por 10 em vez de 100 e "
                                  f"obtém {base * percentual // 10}."),
        (percentual, f"Erro: responde a própria taxa ({percentual}) como se fosse o valor em reais."),
    ]
    return montar(
        rng,
        f"Uma loja vende um produto de R$ {base} com {percentual}% de desconto.",
        "Qual é o valor, em reais, do desconto?",
        itens,
        dificuldade,
    )


def gen_potenciacao(rng, faixas, dificuldade):
    faixa_base, faixa_exp = faixas
    base, exp = rng.randint(*faixa_base), rng.randint(*faixa_exp)
    correto = base ** exp
    fatores = " x ".join([str(base)] * exp)
    itens = [
        (correto, f"{base}^{exp} = {fatores} = {correto}."),
        (base * exp, f"Erro clássico: multiplica a base pelo expoente ({base} x {exp} = {base * exp}) "
                     f"em vez de multiplicar a base por ela mesma {exp} vezes."),
        (base ** (exp - 1) if exp > 1 else base + exp,
         f"Erro: aplica o expoente uma vez a menos ({base}^{exp - 1} = {base ** (exp - 1)})."
         if exp > 1 else f"Erro: soma base e expoente ({base} + {exp} = {base + exp})."),
        (base + exp, f"Erro: soma base e expoente ({base} + {exp} = {base + exp})."),
    ]
    return montar(
        rng,
        f"Calcule o resultado da potência {base}^{exp} ({base} elevado ao expoente {exp}).",
        "Qual é o resultado?",
        itens,
        dificuldade,
    )


def gen_calc_real_decimal(rng, faixa, dificuldade):
    a, b = round(rng.uniform(*faixa), 1), round(rng.uniform(*faixa), 1)
    op = rng.choice(["+", "-", "x"])
    if op == "-" and b > a:
        a, b = b, a
    if op == "+":
        correto, trocada = round(a + b, 1), round(abs(a - b), 1)
        nome_trocada = "subtrai"
    elif op == "-":
        correto, trocada = round(a - b, 1), round(a + b, 1)
        nome_trocada = "soma"
    else:
        correto, trocada = round(a * b, 1), round(a + b, 1)
        nome_trocada = "soma"
    virgula = round(correto * 10, 1) if rng.random() < 0.5 else round(correto / 10, 1)
    itens = [
        (correto, f"{fmt(a)} {op} {fmt(b)} = {fmt(correto)}."),
        (trocada, f"Erro de operação: {nome_trocada} em vez de aplicar '{op}' e obtém {fmt(trocada)}."),
        (virgula, f"Erro ao posicionar a vírgula: obtém {fmt(virgula)} em vez de {fmt(correto)}."),
        (round(correto + 1, 1), "Erro de cálculo de uma unidade."),
    ]
    return montar(
        rng,
        f"Calcule o resultado da operação: {fmt(a)} {op} {fmt(b)}.",
        "Qual é o resultado?",
        itens,
        dificuldade,
        is_integer=False,
    )


CONTEXTOS_INTEIROS = [("Uma pizza", "dividida"), ("Uma barra de chocolate", "dividida"),
                      ("Um bolo", "dividido"), ("Uma folha de papel", "dividida"),
                      ("Um tablete de chocolate", "dividido")]
ACOES_FRACAO = [("comida", "comeu"), ("pintada", "pintou"), ("usada", "usou")]


def gen_fracao_pictorica(rng, faixa, dificuldade):
    """H07: representação de frações — texto no lugar da figura (um todo
    dividido em N partes iguais, M delas destacadas; sem imagem real, só a
    descrição verbal da mesma informação que uma figura mostraria)."""
    den = rng.randint(*faixa)
    num = rng.randint(1, den - 1)
    contexto, particio = rng.choice(CONTEXTOS_INTEIROS)
    acao, verbo = rng.choice(ACOES_FRACAO)
    nome = rng.choice(NOMES)
    correto = f"{num}/{den}"

    candidatos = []

    def add(n2, d2, justificativa):
        texto = f"{n2}/{d2}"
        if texto != correto and n2 >= 0 and d2 >= 1 and texto not in {c[0] for c in candidatos}:
            candidatos.append((texto, justificativa))

    add(den, num, "Erro: inverte numerador e denominador.")
    add(num, den + 1, f"Erro ao contar o total de partes: usa {den + 1} em vez de {den}.")
    add(min(num + 1, den - 1), den, "Erro de contagem: conta uma parte a mais.")
    add(max(num - 1, 0), den, "Erro de contagem: conta uma parte a menos.")
    k = 2
    while len(candidatos) < 4:
        add(num, den + k, f"Erro ao contar o total de partes: usa {den + k} em vez de {den}.")
        k += 1

    itens = [
        (correto, f"{num} partes de um total de {den} partes iguais correspondem à fração {num}/{den}."),
        *candidatos[:4],
    ]
    return montar_texto(
        rng,
        f"{contexto} foi {particio} em {den} partes iguais. {nome} {verbo} {num} dessas partes.",
        f"Qual fração representa a parte {acao}?",
        itens,
        dificuldade,
    )


def gen_fracoes_equivalentes(rng, faixas, dificuldade):
    """H08: identificar frações equivalentes (multiplicar numerador e
    denominador pelo mesmo fator vs. distratores que quebram a proporção)."""
    den_faixa, k_faixa = faixas
    den = rng.randint(*den_faixa)
    num = rng.randint(1, den - 1)
    g = math.gcd(num, den)
    num, den = num // g, den // g  # fração base simplificada, mais natural no enunciado
    k = rng.randint(*k_faixa)
    correto = f"{num * k}/{den * k}"

    candidatos = []

    def add(n2, d2, justificativa):
        texto = f"{n2}/{d2}"
        if texto != correto and n2 >= 1 and d2 >= 1 and texto not in {c[0] for c in candidatos}:
            candidatos.append((texto, justificativa))

    add(num * k + 1, den * k, "Erro: soma 1 ao numerador em vez de manter a proporção.")
    add(num * k, den * k + 1, "Erro: soma 1 ao denominador em vez de manter a proporção.")
    add(num + k, den + k, f"Erro: soma {k} ao numerador e ao denominador em vez de multiplicar por {k}.")
    j = k + 1
    while len(candidatos) < 4 and j <= k + 20:
        add(num * j, den * j + 1,
            f"Erro: {num * j}/{den * j + 1} não é equivalente a {num}/{den}, a proporção não se mantém.")
        j += 1
    if len(candidatos) < 4:
        raise ValueError(f"gen_fracoes_equivalentes: não achou 4 distratores para {correto}")

    itens = [
        (correto, f"Multiplicando numerador e denominador de {num}/{den} por {k}: "
                  f"{num} x {k} = {num * k} e {den} x {k} = {den * k}, logo {correto}."),
        *candidatos[:4],
    ]
    return montar_texto(
        rng,
        f"Considere a fração {num}/{den}.",
        "Qual das alternativas é uma fração equivalente a ela?",
        itens,
        dificuldade,
    )


def gen_fracao_porcentagem(rng, denominadores, dificuldade):
    """H09: converter fração -> porcentagem. Denominadores escolhidos de
    forma que 100 é sempre divisível por eles, garantindo conversão exata."""
    den = rng.choice(denominadores)
    num = rng.randint(1, den - 1)
    fator = 100 // den
    correto_valor = num * fator
    correto = f"{correto_valor}%"

    candidatos = []

    def add(p, justificativa):
        texto = f"{p}%"
        if texto != correto and 0 <= p <= 100 and texto not in {c[0] for c in candidatos}:
            candidatos.append((texto, justificativa))

    v1 = min(correto_valor + 10, 100)
    add(v1, f"Erro de cálculo: obtém {v1}% em vez de {correto_valor}% ao converter a fração.")
    v2 = max(correto_valor - 10, 0)
    add(v2, f"Erro de cálculo: obtém {v2}% em vez de {correto_valor}% ao converter a fração.")
    v3 = min(num * 10, 100)
    add(v3, f"Erro: multiplica {num} por 10 (dá {v3}%) em vez de converter corretamente.")
    # Padding bidirecional (para cima E para baixo) — perto de 0% ou 100% um
    # dos dois lados sempre satura e para de gerar valores novos; sem o outro
    # lado o laço giraria para sempre.
    m = 1
    while len(candidatos) < 4 and m <= 20:
        for delta in (15 * m, -15 * m):
            vm = correto_valor + delta
            if 0 <= vm <= 100:
                add(vm, f"Erro de cálculo: obtém {vm}% em vez de {correto_valor}% ao converter a fração.")
            if len(candidatos) == 4:
                break
        m += 1
    if len(candidatos) < 4:
        raise ValueError(f"gen_fracao_porcentagem: não achou 4 distratores para {correto}")

    itens = [
        (correto, f"{num} x {fator} = {correto_valor}%. (Cada {den}-ésimo equivale a {fator}%, "
                  f"pois 100 ÷ {den} = {fator}.)"),
        *candidatos[:4],
    ]
    return montar_texto(
        rng,
        f"Em uma turma, {num} de cada {den} alunos gostam de matemática.",
        "Isso representa quantos por cento dos alunos?",
        itens,
        dificuldade,
    )


# ---- Habilidades reais do banco (mesmos descritores usados em produção) ----

SKILLS = [
    {
        "ano": "2º", "habilidade": "H06",
        "descricao": "Calcular o resultado de adições ou subtrações, envolvendo números naturais de até 3 ordens.",
        "gerador": gen_calc_soma_subtracao,
        "faixas": {"Fácil": (1, 20), "Moderado": (1, 99), "Difícil": (100, 999)},
    },
    {
        "ano": "2º", "habilidade": "H08",
        "descricao": (
            "Resolver problemas de adição ou de subtração, envolvendo números naturais "
            "de até 3 ordens, com os significados de juntar, acrescentar, separar ou retirar."
        ),
        "gerador": gen_problema_soma_subtracao,
        "faixas": {"Fácil": (2, 20), "Moderado": (10, 99), "Difícil": (50, 300)},
    },
    {
        "ano": "5º", "habilidade": "H03",
        "descricao": "Calcular o resultado de adições ou subtrações envolvendo números naturais de até 6 ordens.",
        "gerador": gen_calc_soma_subtracao,
        "faixas": {"Fácil": (100, 999), "Moderado": (1000, 9999), "Difícil": (10000, 99999)},
    },
    {
        "ano": "5º", "habilidade": "H04",
        "descricao": "Calcular o resultado de multiplicações ou divisões envolvendo números naturais de até 6 ordens.",
        "gerador": gen_calc_mult_div,
        "faixas": {
            "Fácil": ((2, 9), (2, 9)),
            "Moderado": ((2, 20), (10, 50)),
            "Difícil": ((10, 50), (10, 99)),
        },
    },
    {
        "ano": "5º", "habilidade": "H06",
        "descricao": (
            "Resolver problemas de multiplicação ou de divisão, envolvendo números naturais de "
            "até 6 ordens, com os significados de formação de grupos iguais (incluindo repartição "
            "equitativa e medida), proporcionalidade ou disposição retangular."
        ),
        "gerador": gen_problema_mult_div,
        "faixas": {
            "Fácil": ((2, 6), (2, 9)),
            "Moderado": ((3, 10), (5, 20)),
            "Difícil": ((5, 15), (10, 40)),
        },
    },
    {
        "ano": "9º", "habilidade": "H03",
        "descricao": "Porcentagem, acréscimos, decréscimos e taxas sucessivas.",
        "gerador": gen_porcentagem,
        "faixas": {
            "Fácil": ((1, 5), (10, 25, 50)),
            "Moderado": ((5, 15), (15, 30, 40, 60)),
            "Difícil": ((15, 40), (35, 45, 65, 75, 85)),
        },
    },
    {
        "ano": "9º", "habilidade": "H02",
        "descricao": (
            "Resolver problemas de adição, subtração, multiplicação, divisão, potenciação ou "
            "radiciação envolvendo números reais, inclusive notação científica."
        ),
        "gerador": gen_potenciacao,
        "faixas": {"Fácil": ((2, 5), (2, 2)), "Moderado": ((2, 9), (2, 3)), "Difícil": ((2, 12), (2, 4))},
    },
    {
        "ano": "9º", "habilidade": "H06",
        "descricao": "Cálculo com números reais: adição, subtração, multiplicação e divisão.",
        "gerador": gen_calc_real_decimal,
        "faixas": {"Fácil": (1, 20), "Moderado": (1, 99), "Difícil": (1, 500)},
    },
    {
        "ano": "9º", "habilidade": "H07",
        "descricao": "Representar ou associar frações a representações pictóricas.",
        "gerador": gen_fracao_pictorica,
        "faixas": {"Fácil": (4, 8), "Moderado": (6, 12), "Difícil": (8, 20)},
    },
    {
        "ano": "9º", "habilidade": "H08",
        "descricao": "Identificar frações equivalentes.",
        "gerador": gen_fracoes_equivalentes,
        "faixas": {
            "Fácil": ((2, 6), (2, 3)),
            "Moderado": ((3, 9), (2, 4)),
            "Difícil": ((4, 12), (3, 5)),
        },
    },
    {
        "ano": "9º", "habilidade": "H09",
        "descricao": "Converter entre representações de números racionais positivos (frações, decimais, porcentagens)",
        "gerador": gen_fracao_porcentagem,
        "faixas": {
            "Fácil": (2, 4, 5, 10),
            "Moderado": (5, 10, 20, 25),
            "Difícil": (10, 20, 25, 50),
        },
    },
]


def build_example(ano, habilidade, descricao, dificuldade, questoes, idx):
    """`questoes`: lista de 1+ dicts de questão — vira o wrapper {"questoes": [...]}
    exigido pelo contrato do app. quantidade no prompt bate com len(questoes),
    ensinando o modelo a devolver lotes quando o usuário pede N questões."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    quantidade=len(questoes), ano=ano, habilidade=habilidade,
                    descricao=descricao, dificuldade=dificuldade,
                ),
            },
            {"role": "assistant", "content": json.dumps({"questoes": questoes}, ensure_ascii=False)},
        ],
        "meta": {
            "codigo_item": f"SINT-{habilidade}-{dificuldade}-{idx:04d}",
            "ano": ano,
            "habilidade": habilidade,
            "dificuldade": dificuldade,
            "sintetico": True,
        },
    }


def _agrupar(rng, pool):
    """Particiona `pool` em lotes de tamanho aleatório (ver TAMANHOS_LOTE)."""
    grupos = []
    i = 0
    while i < len(pool):
        tamanho = min(rng.choice(TAMANHOS_LOTE), len(pool) - i)
        grupos.append(pool[i:i + tamanho])
        i += tamanho
    return grupos


def gerar_todos(num_per_skill, seed):
    rng = random.Random(seed)
    exemplos, idx = [], 0
    for skill in SKILLS:
        for dificuldade, faixa in skill["faixas"].items():
            pool = [skill["gerador"](rng, faixa, dificuldade) for _ in range(num_per_skill)]
            for grupo in _agrupar(rng, pool):
                idx += 1
                exemplos.append(
                    build_example(skill["ano"], skill["habilidade"], skill["descricao"], dificuldade, grupo, idx)
                )
    return exemplos


def validar(exemplos):
    """Confere schema (5 alternativas, resposta_correta, difficulty) +
    consistência resposta_correta<->resolucao_passo_a_passo + ausência de
    operadores Unicode.

    O modelo tende a imitar o que vê no treino: se um operador tipográfico
    (− × –) entrar nos dados, ele passa a emiti-lo na inferência e quebra
    qualquer verificação baseada em texto. Esta checagem trava isso na origem.
    """
    problemas, verificaveis, ok, unicode_ops = 0, 0, 0, 0
    for ex in exemplos:
        blob = ex["messages"][2]["content"]
        if any(ch in _UNICODE_MATH and not ch.isspace() for ch in blob):
            unicode_ops += 1
        obj = json.loads(blob)
        flags = check_structure(obj)
        if not (flags["schema_completo"] and flags["resposta_valida"]
                and flags["alternativas_distintas"] and flags["difficulty_valida"]):
            problemas += 1
        for questao in extract_questoes(obj):
            consistente, _ = check_consistency(questao)
            if consistente is not None:
                verificaveis += 1
                ok += int(consistente)
    return problemas, verificaveis, ok, unicode_ops


def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-per-skill", type=int, default=DEFAULT_NUM_PER_SKILL,
                         help="exemplos gerados por (habilidade, dificuldade)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="só mostra estatísticas, não escreve")
    args = parser.parse_args()

    sinteticos = gerar_todos(args.num_per_skill, args.seed)
    problemas, verificaveis, ok, unicode_ops = validar(sinteticos)

    total_questoes = sum(len(json.loads(ex["messages"][2]["content"])["questoes"]) for ex in sinteticos)
    print(f"Gerados: {len(sinteticos)} exemplos de treino / {total_questoes} questões "
          f"({len(SKILLS)} habilidades x 3 dificuldades x {args.num_per_skill})")
    print("Cada questão tem 5 alternativas (A-E); resolucao_passo_a_passo mostra a "
          "conta que leva à resposta correta.")
    print(f"Falhas de schema/resposta_correta/alternativas/difficulty: {problemas} (esperado: 0)")
    print(f"Exemplos com operador Unicode (−, ×, –): {unicode_ops} (esperado: 0)")
    print(f"Consistência resposta_correta<->resolucao_passo_a_passo: {ok}/{verificaveis} "
          f"verificáveis ({100 * verificaveis // max(total_questoes, 1)}% de cobertura — "
          "sem um campo dedicado ao VALOR da resposta, a verificação depende de regex "
          "sobre a equação do texto, cobertura menor do que com o campo `resposta`)")

    if args.dry_run:
        print("\n--dry-run: nada escrito. Exemplo:")
        print(json.dumps(sinteticos[0], ensure_ascii=False, indent=2))
        return

    if problemas:
        raise SystemExit(f"Abortando: {problemas} exemplo(s) sintético(s) com schema inválido.")
    if unicode_ops:
        raise SystemExit(
            f"Abortando: {unicode_ops} exemplo(s) com operador matemático Unicode. "
            "Use - + x ÷ ASCII nos geradores (ver docstring de validar())."
        )

    real = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")] if TRAIN_PATH.exists() else []
    real_sem_sintetico = [ex for ex in real if not ex.get("meta", {}).get("sintetico")]
    combinado = real_sem_sintetico + sinteticos
    write_jsonl(TRAIN_PATH, combinado)

    print(f"\nTreino real:      {len(real_sem_sintetico)}")
    print(f"Treino sintético: {len(sinteticos)}")
    print(f"Total escrito em {TRAIN_PATH}: {len(combinado)}")
    print(
        "\ndata/val.jsonl não foi tocado — a validação continua só com questões reais do "
        "SAEB, para medir o modelo no que ele de fato vai enfrentar em produção."
    )


if __name__ == "__main__":
    main()
