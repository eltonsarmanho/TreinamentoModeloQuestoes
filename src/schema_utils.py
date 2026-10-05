"""Validação do schema JSON de questão — sem dependências pesadas de ML.

Compartilhado por evaluate.py (avalia o modelo em 4-bit via HF/bitsandbytes,
usado em desenvolvimento) e test_model.py (testa o .gguf real via llama.cpp,
o mesmo artefato que roda no app mobile), garantindo que os dois caminhos
julguem "resposta válida" da mesma forma.

Schema (contrato fixo definido pelos envolvidos, sem exceção):
    {"questoes": [
        {"enunciado": str,
         "alternativas": {"A": str, "B": str, "C": str, "D": str, "E": str},
         "resolucao_passo_a_passo": str,
         "resposta_correta": "A|B|C|D|E",
         "difficulty": "EASY|MEDIUM|HARD"},
        ...
    ]}

Nota sobre a ordem das chaves dentro de cada questão: o modelo é TREINADO a
emitir `resolucao_passo_a_passo` ANTES de `resposta_correta` (mostra o
trabalho antes de se comprometer com a letra), embora o exemplo do contrato
liste `resposta_correta` primeiro. JSON não garante ordem de chaves para
quem consome por nome — todo consumidor padrão (`json.loads`, `JSON.parse`)
lê por chave, não por posição — então o contrato (mesmas chaves, mesma
estrutura) é respeitado. A ordem de emissão evita reintroduzir o problema
descrito no README ("Gabarito inconsistente com a justificativa", causa 1):
comprometer a letra antes de mostrar o raciocínio.
"""

import json
import re
import unicodedata

IMAGE_PATTERN = re.compile(r"\b(figura|imagem|gráfico|desenho|ilustração)\b", re.I)

# IMAGE_PATTERN casa a palavra solta. Para FILTRAR DADOS DE TREINO isso é o
# desejado (conservador: na dúvida, não treine no item). Para o pipeline de
# PRODUÇÃO é ruim demais: medido em 2026-09, das 3 gerações que ele marcou,
# uma era "Desenho" como nome de um hobby numa alternativa, e outra era
# "...organizar em gráfico de barras. Os pesos são: 35, 42, 30, ..." — questão
# inteiramente autocontida. Descartar essas seria jogar fora questão boa.
#
# O que de fato quebra a questão é a referência DÊITICA: o enunciado aponta
# para um artefato visual que não existe ("Observe a imagem abaixo", "Gráfico
# mostra quantidade de alunos..."), deixando o aluno sem o dado. É isso que
# DEPENDENCIA_VISUAL_PATTERN detecta.
# "imagem" (singular) termina em M; "imagens?" casaria só "imagen"/"imagens".
_ARTEFATO_VISUAL = (
    r"figuras?|image(?:m|ns)|gr[áa]ficos?|desenhos?|"
    r"ilustra[çc](?:[ãa]o|[õo]es)|tabelas?|quadros?"
)
DEPENDENCIA_VISUAL_PATTERN = re.compile(
    # 1) "observe a figura", "conforme o gráfico", "com base na tabela"
    r"(?:observe|veja|analise|conforme|segundo|de acordo com|com base n|"
    r"a partir d|utilizando (?:o|a)|considere (?:o|a))\s+(?:[oa]s?\s+)?"
    rf"(?:{_ARTEFATO_VISUAL})"
    # 2) "figura abaixo", "gráfico a seguir", "imagem apresentada"
    rf"|(?:{_ARTEFATO_VISUAL})\s+(?:abaixo|acima|ao lado|a seguir|seguinte|"
    r"apresentad[oa]s?|mostrad[oa]s?)"
    # 3) "Gráfico mostra ...", "A tabela indica ..." — o artefato é SUJEITO de um
    #    verbo de apresentação, logo o dado está nele. Os lookbehinds excluem
    #    "QUAL gráfico representa essas notas?", em que o artefato é a RESPOSTA
    #    (as alternativas são tipos de gráfico) e nada falta ao enunciado.
    r"|(?<!qual )(?<!quais )(?<!que )"
    rf"\b(?:{_ARTEFATO_VISUAL})\s+(?:mostra|apresenta|indica|representa|"
    r"exibe|traz|cont[ée]m)\b",
    re.I,
)


def depende_de_visual_ausente(texto):
    """True se o texto aponta para um artefato visual que o app não tem.

    Usado no best-of-N (test_model._score_candidato) para preferir um candidato
    resolvível a um que manda o aluno olhar uma imagem inexistente.
    """
    return bool(DEPENDENCIA_VISUAL_PATTERN.search(str(texto or "")))

QUESTOES_KEY = "questoes"
ALTERNATIVE_LETTERS = "ABCDE"


def _letra_valida(v):
    """True só para uma letra A-E. `None in "ABCDE"` levanta TypeError (derrubou a
    injeção do 2º ano às 21h05 de 01/10, com resposta_correta nula) e `"" in
    "ABCDE"`/`"AB" in "ABCDE"` davam True: a checagem por substring aprovava
    resposta vazia ou com duas letras."""
    return isinstance(v, str) and v in tuple(ALTERNATIVE_LETTERS)
REQUIRED_KEYS = {
    "enunciado",
    "alternativas",
    "resolucao_passo_a_passo",
    "resposta_correta",
    "difficulty",
}
DIFFICULTY_VALUES = {"EASY", "MEDIUM", "HARD"}
DIFFICULTY_MAP = {"Fácil": "EASY", "Moderado": "MEDIUM", "Difícil": "HARD"}


def parse_json(text):
    """Extrai o primeiro objeto JSON do texto gerado (tolera texto/tags ao redor)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    start = text.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def extract_questoes(obj):
    """Retorna a lista de questões do wrapper {"questoes": [...]}, ou [] se inválido."""
    if not isinstance(obj, dict):
        return []
    questoes = obj.get(QUESTOES_KEY)
    return questoes if isinstance(questoes, list) else []


def extract_questao(obj, index=0):
    """Atalho para uma questão específica do wrapper, ou None se não existir."""
    questoes = extract_questoes(obj)
    if index < len(questoes) and isinstance(questoes[index], dict):
        return questoes[index]
    return None


def check_structure(obj, quantidade_esperada=None):
    """Retorna dict de flags estruturais para o wrapper `{"questoes": [...]}` já parseado.

    As flags de schema (`schema_completo`, `resposta_valida`,
    `alternativas_distintas`, `difficulty_valida`) exigem que TODAS as
    questões da lista as satisfaçam — uma só quebrada já reprova o lote.
    """
    flags = {
        "json_valido": obj is not None,
        "wrapper_valido": False,
        "quantidade_correta": False,
        "schema_completo": False,
        "resposta_valida": False,
        "alternativas_distintas": False,
        "difficulty_valida": False,
    }
    questoes = extract_questoes(obj)
    flags["wrapper_valido"] = len(questoes) > 0
    if not flags["wrapper_valido"]:
        return flags
    flags["quantidade_correta"] = (
        quantidade_esperada is None or len(questoes) == quantidade_esperada
    )

    flags["schema_completo"] = all(
        isinstance(q, dict) and REQUIRED_KEYS.issubset(q.keys()) for q in questoes
    )
    flags["resposta_valida"] = all(
        isinstance(q, dict) and _letra_valida(q.get("resposta_correta"))
        for q in questoes
    )
    flags["difficulty_valida"] = all(
        isinstance(q, dict) and q.get("difficulty") in DIFFICULTY_VALUES
        for q in questoes
    )

    def _alts_ok(q):
        alts = q.get("alternativas") if isinstance(q, dict) else None
        if not isinstance(alts, dict) or set(alts.keys()) < set(ALTERNATIVE_LETTERS):
            return False
        values = [str(alts[k]).strip() for k in ALTERNATIVE_LETTERS]
        return len(set(values)) == len(ALTERNATIVE_LETTERS) and all(values)

    flags["alternativas_distintas"] = all(_alts_ok(q) for q in questoes)
    return flags


_NUM_PATTERN = re.compile(r"-?\d+(?:[.,]\d+)?")

# O modelo emite operadores tipográficos Unicode (−, ×, –) em vez dos ASCII
# usados no dataset de treino. Sem normalizar, o regex abaixo não casa e o
# verificador devolve "não verificável" — silenciosamente deixando passar
# gabaritos errados.
_UNICODE_MATH = {
    "−": "-",  # MINUS SIGN
    "–": "-",  # EN DASH
    "—": "-",  # EM DASH
    "×": "x",  # MULTIPLICATION SIGN
    "⋅": "x",  # DOT OPERATOR
    "∙": "x",  # BULLET OPERATOR
    "∕": "/",  # DIVISION SLASH
    "＝": "=",  # FULLWIDTH EQUALS
    " ": " ",  # NO-BREAK SPACE
    " ": " ",  # NARROW NO-BREAK SPACE
}
_UNICODE_TABLE = str.maketrans(_UNICODE_MATH)


def normalize_math(text):
    """Converte operadores matemáticos Unicode para os equivalentes ASCII."""
    return str(text or "").translate(_UNICODE_TABLE)


# Casa expressões simples "a op b = r" dentro do texto de resolução, ex.:
# "35 - 20 = 15", "3x4=12", "10/2 = 5" (após normalize_math).
_EXPR_PATTERN = re.compile(
    r"(-?\d+(?:[.,]\d+)?)\s*([-+xX*÷/])\s*(-?\d+(?:[.,]\d+)?)\s*=\s*(-?\d+(?:[.,]\d+)?)"
)
_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "x": lambda a, b: a * b,
    "X": lambda a, b: a * b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b if b else None,
    "÷": lambda a, b: a / b if b else None,
}


def _to_number(text):
    try:
        value = float(text.replace(",", "."))
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def _computed_result(text):
    """Extrai o resultado de uma conta 'a op b = r' no texto, se a conta bater."""
    match = _EXPR_PATTERN.search(normalize_math(text))
    if not match:
        return None
    a_str, op, b_str, r_str = match.groups()
    a, b, r = _to_number(a_str), _to_number(b_str), _to_number(r_str)
    if a is None or b is None or r is None:
        return None
    computed = _OPS[op](a, b)
    if computed is None:
        return None
    return r if abs(computed - r) < 1e-6 else None


def _computed_results(text):
    """Resultados de TODAS as contas 'a op b = r' corretas do texto, em ordem.

    Resoluções de vários passos ("6 + 7 = 13. Depois, 13 + 5 = 18") têm a
    resposta na ÚLTIMA conta; olhar só a primeira (13) reprovava questões
    corretas e fazia fix_gabarito trocar o gabarito certo por um errado."""
    out = []
    for match in _EXPR_PATTERN.finditer(normalize_math(text)):
        a_str, op, b_str, r_str = match.groups()
        a, b, r = _to_number(a_str), _to_number(b_str), _to_number(r_str)
        if a is None or b is None or r is None:
            continue
        computed = _OPS[op](a, b)
        if computed is not None and abs(computed - r) < 1e-6:
            out.append(r)
    return out


def _leading_number(text):
    match = _NUM_PATTERN.search(normalize_math(text))
    return _to_number(match.group()) if match else None


# Separador de MILHAR pt-BR: "125.438 + 234.729 = 360.167". _to_number lê o
# ponto como decimal (360.167), a alternativa escreve "360167", e a questão —
# perfeitamente correta — era acusada de inconsistente. Medido em
# data/train_curado.jsonl: 4 das 11 acusações restantes vinham só daqui
# (train:33, train:100, train:472, train:631).
#
# O token exige grupos de EXATAMENTE 3 dígitos e parte inteira começando em
# 1-9, então "0.875", "8.5" e "1.25" (decimais ASCII legítimos, que aparecem
# nas alternativas do corpus) NÃO casam. O lookahead `(?![\d,])` recusa
# "1.234,50" (ali o ponto é milhar mas a vírgula decimal quebra o parse de
# qualquer jeito) e aceita o ponto final de frase em "= 360.167.".
_MILHAR_PT_BR = re.compile(r"(?<![\d.,])([1-9]\d{0,2}(?:\.\d{3})+)(?![\d,])")


def _fim_da_ultima_conta(texto_normalizado):
    """Posição onde termina a ÚLTIMA conta ARITMETICAMENTE VÁLIDA do texto (0 se
    não houver). Usa a conta válida, não o último casamento da regex: em
    "9 - 12 + 6 = 3" a regex casa "12 + 6 = 3", que não fecha, e tomar o fim
    desse casamento esconderia justamente a cauda que interessa."""
    fim = 0
    for match in _EXPR_PATTERN.finditer(texto_normalizado):
        a_str, op, b_str, r_str = match.groups()
        a, b, r = _to_number(a_str), _to_number(b_str), _to_number(r_str)
        if a is None or b is None or r is None:
            continue
        computed = _OPS[op](a, b)
        if computed is not None and abs(computed - r) < 1e-6:
            fim = match.end()
    return fim


def _valor_na_cauda(resolucao, valores):
    """True se algum de `valores` aparece como número DEPOIS da última conta válida.

    REGRA E — guarda de CAUDA. `_EXPR_PATTERN` só enxerga contas da forma
    "a op b = r"; o que a resolução escreve DEPOIS da última conta que fecha
    fica invisível para ela, e é exatamente aí que mora a resposta final em três
    formatos frequentes no corpus curado do projeto:
      * arredondamento  — "31 ÷ 4 = 7,75 → arredondado = 7,8" (train_curado:430)
      * truncamento     — "7 ÷ 8 = 0,875 → 0,87" (:458), "86,4º → 86º" (:982)
      * cadeia de 3+ operandos, que a regex não casa — "3² - 4x3 + 6 = 9 - 12 +
        6 = 3" (:492) e "3x2 + 2x3 = 6 + 6 = 12" (:712)
      * unidade/moeda entre o "=" e o número — "803,25 ÷ 4 = R$ 212,25" (val:12)
    Nos cinco casos o gabarito está CERTO e a resolução escreve o valor dele com
    todas as letras; o verificador é que parava de ler cedo demais.

    Usada SÓ PARA ABSOLVER, como `_sem_separador_milhar`: nunca gera acusação
    nova, só impede uma acusação de `resposta_fora_das_alternativas` quando a
    resolução, na sua cauda, afirma o valor do gabarito. A questão degrada para
    "não verificável" em vez de virar rejeição dura — e isso importa porque a
    Fase 1 (test_model._score_candidato, gerar_lote.RANK_STATUS) trata
    `resposta_fora_das_alternativas` como irrecuperável.

    MEDIDO antes/depois em 2026-09-30 (ver o relatório da tarefa): das 7
    acusações de data/train_curado.jsonl esta guarda absolve as 6 que são
    matematicamente corretas e mantém a única errada (:270); nos 14 relatórios
    de outputs/ derruba 12 instâncias (10 únicas) para 10 instâncias (8 únicas),
    silenciando só os 2 casos de arredondamento discutível ("42÷5=8,4 → 8,5
    (aproximado)" e "10÷4=2,5, portanto 2 dezenas") e preservando os 8 erros
    reais confirmados à mão.
    """
    alvos = {v for v in valores if v is not None}
    if not alvos:
        return False
    texto = normalize_math(resolucao)
    fim = _fim_da_ultima_conta(texto)
    return any(match.start() >= fim and _to_number(match.group()) in alvos
               for match in _NUM_PATTERN.finditer(texto))


def _sem_separador_milhar(text):
    """Remove o ponto de milhar pt-BR, devolvendo uma LEITURA ALTERNATIVA do texto.

    Usada só para ABSOLVER: o resultado desta leitura nunca gera acusação nova,
    apenas impede uma acusação que a leitura decimal produziria. Assim a
    ambiguidade "ponto é milhar ou decimal?" degrada para "não verificável"
    em vez de virar "gabarito errado".
    """
    return _MILHAR_PT_BR.sub(
        lambda m: m.group(1).replace(".", ""), normalize_math(text)
    )


# ---------------------------------------------------------------------------
# Classificação da ALTERNATIVA (e não do enunciado) — corrige o falso positivo
# das perguntas de comparação.
#
# Medido em 2026-09 sobre as 916 questões únicas dos relatórios
# outputs/diversidade_*.json: das 16 marcadas "inconsistente" pela regra
# anterior, 4 eram FALSO POSITIVO (25%) e todas pelo MESMO motivo — a
# alternativa não carrega o VALOR da resposta, então _leading_number lia um
# número irrelevante (ou nenhum):
#   * "Qual jardim tem maior área?" com alternativas "Jardim com 12 x 10 m²" /
#     "Jardim com 14 x 8 m²"; a resolução calcula 120 e 112 e o gabarito A está
#     CERTO, mas _leading_number("Jardim com 12 x 10 m²") devolve 12.
#   * "Área A"/"Área B"/"Área C" (10, 24 e 36 quadradinhos), gabarito C CERTO.
#   * "A caixa retangular."/"A caixa quadrada." (6x4=24 vs 5x5=25), gabarito B CERTO.
#   * alternativas literalmente "A"/"B"/"Ambas têm a mesma área." (3x6 vs 5x2),
#     gabarito A CERTO.
#
# O discriminante NÃO é o enunciado ("qual é maior?"): a questão
# "diferença entre o maior e o menor valor" também é comparativa e tem resposta
# numérica legítima (150-80=70, gabarito C=110 ERRADO — tem de continuar
# reprovando). O discriminante é a CLASSE DA ALTERNATIVA. Por isso a detecção
# por lista de palavras de comparação foi medida e descartada: cobre menos e
# falha mais.
# ---------------------------------------------------------------------------

# "Nenhuma das alternativas anteriores" e variantes. Não é um valor: aparece em
# 70,7% das questões geradas, então tratá-la como caso de borda seria errado.
_NDA_PATTERN = re.compile(
    r"nenhuma\s+das\s+(?:alternativas|op[çc][õo]es|anteriores|respostas)"
    r"|todas\s+as\s+(?:alternativas|op[çc][õo]es|anteriores)"
    r"|^n\.?\s?d\.?\s?a\.?$"
    r"|n[ãa]o\s+[ée]\s+poss[íi]vel\s+(?:determinar|comparar|calcular)",
    re.I,
)

# Um ÚNICO número, NO INÍCIO, opcionalmente seguido de unidade curta:
# "14 m", "36 balões.", "8.5", "R$ 12,50", "25%", "12 unidades de área".
# O teste de "número no início" foi preferido ao de "valor puro com unidade
# ASCII curta" porque este último quebrava 14 questões hoje corretas
# ("450π", "20 PÁGINAS.", "12 unidades de área").
_NUM_ISOLADO = re.compile(
    r"^(?:r\$\s*)?(-?\d+(?:[.,]\d+)?)\s*"
    r"(?:%|[a-zà-ÿ²³ºª°/]{0,12}(?:\s+[a-zà-ÿ²³ºª°]{1,12}){0,2})$",
    re.I,
)
_FRACAO_ISOLADA = re.compile(r"^(-?\d+)\s*/\s*(-?\d+)\s*[a-zà-ÿ²³ºª°%]{0,12}$", re.I)


def _texto_alternativa(texto):
    """Normaliza a alternativa para classificação: Unicode math + ponto final."""
    t = normalize_math(texto).strip()
    return t[:-1].strip() if t.endswith(".") else t


def classe_alternativa(texto):
    """Classifica UMA alternativa: "nda" | "numero" | "fracao" | "textual".

    É este classificador — e não o enunciado — que separa a questão cujo
    gabarito é um VALOR (o resultado da conta tem de estar entre as
    alternativas) da questão cujo gabarito é um RÓTULO ("Área C", "A caixa
    quadrada"), em que o valor literal é irrelevante.
    """
    t = _texto_alternativa(texto)
    if not t:
        return "textual"
    if _NDA_PATTERN.search(t):
        return "nda"
    if _FRACAO_ISOLADA.match(t):
        return "fracao"
    if _NUM_ISOLADO.match(t):
        return "numero"
    return "textual"


def _valores_alternativa(texto):
    """Valores comparáveis de uma alternativa de valor isolado (lista, pode ser vazia).

    Para fração "4/12" devolve [4, 0.333...]: o numerador cobre a resolução que
    só calcula os numeradores ("2 x 4 = 8"), e o quociente cobre a que calcula
    o valor decimal ("3 ÷ 6 = 0,5" com alternativa "1/2").
    """
    t = _texto_alternativa(texto)
    match = _FRACAO_ISOLADA.match(t)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        return [a] + ([a / b] if b else [])
    match = _NUM_ISOLADO.match(t)
    if match:
        valor = _to_number(match.group(1))
        return [] if valor is None else [valor]
    return []


def _chave_valor(texto):
    """Chave de identidade de uma alternativa de valor, para detectar repetição.

    Serve à guarda do ramo numérico: alternativas descritivas do tipo
    "4 lados iguais" / "4 ângulos retos" / "4 vértices" são classificadas como
    "numero" (o número está no início), mas repetem o mesmo valor. Exigir
    valores DISTINTOS bloqueia essas 12 questões do corpus sem custar nenhuma
    das detecções reais.
    """
    t = _texto_alternativa(texto)
    match = _FRACAO_ISOLADA.match(t)
    if match:
        return ("f", match.group(1), match.group(2))
    match = _NUM_ISOLADO.match(t)
    if match:
        return ("n", _to_number(match.group(1)))
    return None


# ---------------------------------------------------------------------------
# Correspondência entre a ÚLTIMA frase da resolução e a alternativa textual.
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    """a o as os um uma uns umas de do da dos das em no na nos nas por para com
    e ou que ao aos pelo pela pelos pelas sao tem foi era com mais menos""".split()
)


def _sem_acento(texto):
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def _frases(texto):
    """Divide a resolução em frases, protegendo o ponto decimal.

    Cuidado medido: o divisor ingênuo r'(?<!\\d)[.;!?\\n](?!\\d)' NÃO separa
    "3 x 6 = 18. Área do time B..." (há dígito ANTES do ponto), e isso fazia a
    "última frase" englobar a resolução inteira, perdendo o caso dos times A/B.
    A seta "→" entra como separador porque o modelo a usa como passo.
    """
    t = normalize_math(texto)
    t = re.sub(r"(?<=\d)\.(?=\d)", "\x00", t)  # protege "8.4"
    partes = re.split(r"[.;!?\n→]+", t)
    return [p.replace("\x00", ".").strip() for p in partes if p.strip()]


def _tokens_correspondencia(texto):
    """(numeros, palavras, letras_maiusculas_isoladas) usados na correspondência.

    O artigo inicial é removido tanto da alternativa quanto da frase. Isso é a
    mitigação do risco mais delicado da regra: em "A área do time A é maior", o
    primeiro "A" é ARTIGO capitalizado, não rótulo de alternativa. Sem remover,
    uma frase iniciada por "A" casaria a alternativa cujo texto é só "A" e o
    verificador sugeriria trocar um gabarito certo.
    """
    t = normalize_math(texto).strip()
    t = re.sub(r"^(?:[AaOo]s?|[Aa]o)\s+", "", t)
    numeros = [_to_number(x) for x in re.findall(r"-?\d+(?:[.,]\d+)?", t)]
    letras = set(re.findall(r"(?<![\wÀ-ÿ])([A-E])(?![\wÀ-ÿ])", t))
    palavras = {
        w for w in re.findall(r"[A-Za-zÀ-ÿ]{3,}", _sem_acento(t).lower())
        if w not in _STOPWORDS
    }
    return [n for n in numeros if n is not None], palavras, letras


def _corresponde(alternativa, frase):
    """A frase final da resolução aponta para ESTA alternativa?

    Exige que TODOS os tokens da alternativa (números com multiplicidade,
    palavras de conteúdo e letras maiúsculas isoladas) apareçam na frase.
    Conservador de propósito: se duas alternativas casarem, o resultado é
    "ambíguo" e a questão vira não verificável — nunca uma acusação falsa.
    """
    nums_a, pal_a, let_a = _tokens_correspondencia(alternativa)
    nums_f, pal_f, let_f = _tokens_correspondencia(frase)
    if not (nums_a or pal_a or let_a):
        return False
    restantes = list(nums_f)
    for n in nums_a:
        if n not in restantes:
            return False
        restantes.remove(n)
    return pal_a.issubset(pal_f) and let_a.issubset(let_f)


# Motivos devolvidos por check_consistency_detalhado(). São documentação
# executável: quem chama pode distinguir "gabarito trocado" (recuperável por
# fix_gabarito) de "resposta fora das alternativas" (irrecuperável).
MOTIVO_FORA_DAS_ALTERNATIVAS = "resposta_fora_das_alternativas"


def check_consistency_detalhado(questao):
    """Como check_consistency, mas devolve (ok, sugestao, motivo).

    `motivo` é uma string curta que explica QUAL regra decidiu. Existe porque
    "inconsistente" mistura dois casos muito diferentes:
      * "gabarito_errado" — a conta bate com OUTRA alternativa; fix_gabarito
        conserta trocando a letra;
      * MOTIVO_FORA_DAS_ALTERNATIVAS — o resultado final não está em alternativa
        nenhuma; não há letra para sugerir e a única saída é regenerar.
    """
    if not isinstance(questao, dict):
        return None, None, "sem_questao"
    gabarito = questao.get("resposta_correta")
    alternativas = questao.get("alternativas")
    if not isinstance(alternativas, dict) or gabarito not in alternativas:
        return None, None, "sem_gabarito"

    resolucao = questao.get("resolucao_passo_a_passo", "")
    resultados = _computed_results(resolucao)
    if not resultados:
        return None, None, "sem_conta"

    # (1) e (2): comportamento ANTERIOR, preservado byte a byte. Consistente se
    # o valor do gabarito é resultado de ALGUMA conta correta da resolução
    # (cobre passos intermediários). É o que garante zero regressão nas
    # questões hoje classificadas "ok".
    if _leading_number(alternativas.get(gabarito)) in resultados:
        return True, None, "valor_bate"
    if any(v in resultados for v in _valores_alternativa(alternativas.get(gabarito))):
        return True, None, "valor_bate_fracao"
    # Absolvição pela leitura de milhar pt-BR (ver _sem_separador_milhar).
    resultados_milhar = _computed_results(_sem_separador_milhar(resolucao))
    if resultados_milhar != resultados:
        if _leading_number(alternativas.get(gabarito)) in resultados_milhar:
            return True, None, "valor_bate_milhar"
        if any(v in resultados_milhar
               for v in _valores_alternativa(alternativas.get(gabarito))):
            return True, None, "valor_bate_milhar"

    classes = {letra: classe_alternativa(txt) for letra, txt in alternativas.items()}
    de_valor = [L for L, c in classes.items() if c in ("numero", "fracao")]
    textuais = [L for L, c in classes.items() if c == "textual"]
    chaves = [_chave_valor(alternativas[L]) for L in de_valor]
    final = resultados[-1]

    # (3) RAMO NUMÉRICO — todas as alternativas são valor isolado (NDA à parte).
    # Só aqui faz sentido exigir que o resultado da conta esteja entre elas.
    # Guardas: no mínimo 3 alternativas de valor e valores DISTINTOS entre si,
    # para não arrastar para cá listas descritivas ("4 lados iguais",
    # "4 ângulos retos"), em que o número é adjetivo e não resposta.
    homogeneo_numerico = (
        not textuais and len(de_valor) >= 3 and len(set(chaves)) == len(chaves)
    )
    if homogeneo_numerico:
        if classes[gabarito] == "nda":
            # Gabarito "Nenhuma das anteriores" e o resultado não está nas
            # demais: é exatamente o que "nenhuma" afirma. Mas a conta da
            # própria resolução pode estar errada (malha 3x3 que calcula
            # "3 x 1 = 3"), então não dá para aprovar nem reprovar.
            return None, None, "gabarito_nda"
        for letra in de_valor:
            if final in _valores_alternativa(alternativas[letra]):
                return False, letra, "gabarito_errado"
        # Nenhuma alternativa contém o resultado final: questão IRRECUPERÁVEL.
        # Ex. real (P12-5º-H21-N5): alternativas 60/100/110/120/130, gabarito
        # C=110, resolução "150 - 80 = 70". Não existe letra para sugerir —
        # deliberadamente NÃO se sugere a letra NDA, porque em 2 dos 5 casos
        # reais com NDA a própria resolução está errada e fix_gabarito gravaria
        # um gabarito errado de outro jeito.
        #
        # Última absolvição: se a leitura de milhar pt-BR produz um resultado
        # que ESTÁ entre as alternativas, o "." era separador e não decimal —
        # a acusação seria artefato de parsing. Degrada para não verificável.
        if resultados_milhar and any(
            resultados_milhar[-1] in _valores_alternativa(alternativas[L])
            for L in de_valor
        ):
            return None, None, "milhar_ambiguo"
        # REGRA E — guarda de cauda. A resolução pode escrever a resposta final
        # DEPOIS da última conta que _EXPR_PATTERN fecha (arredondamento,
        # cadeia de 3+ operandos, unidade entre "=" e o número). Se o valor do
        # gabarito está nessa cauda, o verificador parou de ler cedo demais e a
        # acusação é artefato — degrada para não verificável. Ver _valor_na_cauda.
        if _valor_na_cauda(resolucao,
                           [_leading_number(alternativas.get(gabarito))]
                           + list(_valores_alternativa(alternativas.get(gabarito)))):
            return None, None, "resultado_na_cauda"
        return False, None, MOTIVO_FORA_DAS_ALTERNATIVAS

    # (4) RAMO TEXTUAL/REFERENCIAL — o gabarito é um rótulo, não um valor.
    # A verificação correta é por CORRESPONDÊNCIA entre a alternativa e a
    # conclusão da resolução (sua última frase).
    partes = _frases(resolucao)
    if not partes:
        return None, None, "sem_frase"
    ultima = partes[-1]
    casam = [
        L for L in alternativas
        if classes[L] != "nda" and _corresponde(alternativas[L], ultima)
    ]
    if len(casam) == 1:
        if casam[0] == gabarito:
            return True, None, "corresponde"
        return False, casam[0], "corresponde_outra"
    # Mais de uma casou (ambíguo) ou nenhuma: conservador, não acusa.
    # NÃO se usa a letra citada na resolução ("a alternativa correta é a C")
    # como confirmação: é circular (30/918 disparos, todos concordando com o
    # gabarito, ganho informativo zero) e o padrão frouxo gera falso positivo.
    return None, None, "ambiguo" if casam else "sem_correspondencia"


def resposta_fora_das_alternativas(questao):
    """True quando o resultado final da resolução não está em alternativa nenhuma.

    Caso IRRECUPERÁVEL: não existe letra para fix_gabarito sugerir, logo trocar
    o gabarito não resolve e a única saída é regenerar a questão. Exposto como
    função pública para que o best-of-N (test_model._score_candidato) e o
    ranking de lote (gerar_lote.RANK_STATUS) consigam distinguir esse caso de
    um gabarito meramente trocado.
    """
    return check_consistency_detalhado(questao)[2] == MOTIVO_FORA_DAS_ALTERNATIVAS


def check_consistency(questao):
    """Confere se `resposta_correta` (a LETRA) aponta para a alternativa cujo
    valor bate com a conta extraída de `resolucao_passo_a_passo`.

    Único caminho de verificação possível neste schema: não existe mais um
    campo dedicado ao VALOR da resposta (o antigo campo `resposta`, removido
    para seguir o contrato exigido pelos envolvidos). Por isso a cobertura
    volta a depender de regex sobre texto livre — cobre bem contas simples
    ("a op b = r"), mas não pega raciocínio verbal sem equação nem frações/
    porcentagens textuais. Ver seção "Limitações" da documentação científica.

    Retorna (ok, sugestao):
      ok=True         resposta_correta aponta para a alternativa certa
      ok=False        não aponta — `sugestao` traz a letra correta, se identificável
      ok=None         não deu para verificar (sem equação reconhecível no texto)

    ASSINATURA E CONTRATO CONGELADOS: test_model.py, gerar_lote.py,
    avaliar_diversidade.py, evaluate.py, distill_teacher.py, curar_diversidade.py,
    generate_synthetic.py e validar_entrega.py dependem desta tupla de 2. Quem
    precisa saber POR QUE o veredito foi esse usa check_consistency_detalhado().
    """
    ok, sugestao, _motivo = check_consistency_detalhado(questao)
    return ok, sugestao


def fix_gabarito(questao):
    """Correção determinística pós-geração (para o app e para os scripts de teste).

    Se `resposta_correta` não aponta para a alternativa que bate com a conta
    de `resolucao_passo_a_passo`, troca a letra pela correta. Retorna
    (questao, status):
      "ok"              resposta_correta já aponta para a alternativa certa
      "corrigido"       letra trocada para a alternativa certa
      "inconsistente"   a conta não corresponde a nenhuma alternativa —
                        questão malformada, regenerar é o único remédio
      "fora_das_alternativas"  (NOVO, aditivo) subcaso de "inconsistente" em que
                        TODAS as alternativas são valor numérico puro e o
                        resultado final da resolução não está em nenhuma delas.
                        É IRRECUPERÁVEL por troca de letra; só regenerar resolve.
      "nao_verificavel" sem conta reconhecível em resolucao_passo_a_passo

    O status novo é ADITIVO: o único chamador que compara status exato é
    test_model.generate_validated (`if fix_status == "corrigido"`), e
    "fora_das_alternativas" cai no mesmo ramo em que "inconsistente" já caía —
    comportamento idêntico ao de antes, mas agora distinguível por quem quiser.

    Nota: só a letra de `resposta_correta` muda; `resolucao_passo_a_passo`
    permanece como está.
    """
    consistente, sugestao, motivo = check_consistency_detalhado(questao)
    if consistente is True:
        return questao, "ok"
    if consistente is None:
        return questao, "nao_verificavel"
    if sugestao:
        corrigido = dict(questao)
        corrigido["resposta_correta"] = sugestao
        return corrigido, "corrigido"
    if motivo == MOTIVO_FORA_DAS_ALTERNATIVAS:
        return questao, "fora_das_alternativas"
    return questao, "inconsistente"
