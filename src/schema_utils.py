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
from fractions import Fraction

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
    if "," in text and "." in text:  # "1.250,50": ponto de milhar + vírgula decimal
        text = text.replace(".", "")
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
    """Resultados de TODAS as contas corretas do texto, em ordem de posição.

    Resoluções de vários passos ("6 + 7 = 13. Depois, 13 + 5 = 18") têm a
    resposta na ÚLTIMA conta; olhar só a primeira (13) reprovava questões
    corretas e fazia fix_gabarito trocar o gabarito certo por um errado.
    Desde 2026-10-10 inclui as contas n-árias/encadeadas/com unidade do motor
    estendido (ver _analisar_contas), unidas às da regex binária antiga."""
    return [_numero(r[1]) for r in _contas_ricas(text)]


# ---------------------------------------------------------------------------
# MOTOR ARITMÉTICO ESTENDIDO (2026-10-10).
#
# Por que existe. A auditoria de 2026-10-10 mediu que _EXPR_PATTERN (conta
# BINÁRIA "a op b = r" com números puros) deixava 52,5% do corpus de treino
# (data/train_curado_v3.jsonl, 1449 de 2759) e 61 de 90 gerações do val como
# "não verificável" (Geometria 92%, Grandezas 75%, Prob/Estat 71%). Dali saíam
# sem reprovação 13 de 16 gabaritos errados cujo resultado calculado na
# resolução estava FORA das alternativas. Casos reais que passavam:
#   "8 + 6 + 8 + 6 = 28 cm", "180° - 150° = 30°", "2 x (8 + 6) = 28"  (-> [])
#   "40/120 = 1/3 ≈ 33,3%" com alternativas 100%/20%/25%/50%/120%, gabarito 25%.
#
# O que lê agora: somas/produtos n-ários, parênteses, potência (²,³,^), unidades
# coladas ou separadas (° cm m m² km kg g L mL h min R$ %), vírgula decimal e
# ponto de milhar pt-BR ("1.250,50"), sinais - + x × * ÷ / (":" só entre
# espaços), "p% de N" e "a/b de N", relógio "10h30" (em minutos) e igualdades
# ENCADEADAS ("a + b = c = d", "a op b = c; c ÷ 2 = d"). "≈" aceita arredondamento
# (|dif| < 10^-casas; sem casas, < 1). Divisão por zero, expoente > 12 e
# módulo > 1e15 invalidam o termo (nunca viram resultado). Aritmética em
# Fraction: sem erro de ponto flutuante.
#
# Contrato: só ACRESCENTA resultados a _computed_results (a regex antiga continua
# sendo unida a eles), logo tudo que era "ok" continua "ok". Uma conta interna
# ERRADA não vira resultado (como antes); só deixa de contar.
#
# MEDIDO em 2026-10-10 (sem rodar o modelo): train_curado_v3 (2759) ok 1310 ->
# 1624 (47,5% -> 58,9%), nao_verificavel 1449 -> 1135; por unidade ok antes ->
# depois: Números 67,6 -> 73,6%, Álgebra 38,8 -> 45,1%, Geometria 8,4 -> 21,9%,
# Grandezas 25,5 -> 55,8%, Prob/Estat 28,5 -> 42,3%. Reprovadas novas no treino:
# 0 (a regra crua dava 42 falsos positivos, todos absolvidos por
# _valor_na_cauda/_valor_mencionado; ver os testes de TestGuardasContraFalsoPositivo).
# Nas 90 gerações do val: 14 nv -> ok e 9 nv -> reprovada, as 9 lidas à mão são
# erros reais. Em 1623 questões únicas dos relatórios: 49 reprovadas novas
# (36 únicas), lidas à mão. O que continua CEGO: pergunta errada com resolução
# coerente ("12 m² -> 28 m², quantos m² agora?", gabarito 16), gabarito que bate
# com a conta errada da própria resolução, raciocínio verbal sem conta, e o ramo
# textual (alternativas-rótulo), que só roda para quem já tinha conta binária.
#
# Portabilidade (módulo roda no app via TypeScript): sem grupo nomeado,
# lookbehind, flag embutida, \Z ou classe \p; a varredura é manual.
# ---------------------------------------------------------------------------

_UNIDADES_CANON = {
    "mm": "mm", "cm": "cm", "dm": "dm", "m": "m", "km": "km", "metro": "m", "metros": "m",
    "mm²": "mm²", "cm²": "cm²", "m²": "m²", "km²": "km²", "cm³": "cm³", "m³": "m³",
    "mg": "mg", "g": "g", "kg": "kg", "t": "t", "grama": "g", "gramas": "g",
    "quilo": "kg", "quilos": "kg", "quilograma": "kg", "quilogramas": "kg",
    "l": "l", "ml": "ml", "litro": "l", "litros": "l",
    "s": "s", "seg": "s", "segundo": "s", "segundos": "s",
    "min": "min", "minuto": "min", "minutos": "min",
    "h": "h", "hora": "h", "horas": "h",
    "real": "rs", "reais": "rs", "centavo": "cent", "centavos": "cent",
    "grau": "°", "graus": "°",
}
# (de, para) -> fator tal que valor_em_de * fator = valor_em_para
_FATORES_UNIDADE = {
    ("m", "cm"): 100, ("m", "mm"): 1000, ("cm", "mm"): 10, ("km", "m"): 1000,
    ("km", "cm"): 100000, ("dm", "cm"): 10, ("m", "dm"): 10,
    ("kg", "g"): 1000, ("g", "mg"): 1000, ("kg", "mg"): 10 ** 6, ("t", "kg"): 1000,
    ("l", "ml"): 1000, ("h", "min"): 60, ("min", "s"): 60, ("h", "s"): 3600,
    ("rs", "cent"): 100, ("m²", "cm²"): 10 ** 4, ("km²", "m²"): 10 ** 6,
    ("m³", "l"): 1000, ("m³", "cm³"): 10 ** 6, ("cm³", "ml"): 1,
}
_LIMITE_MODULO = 10 ** 15
_LIMITE_EXPOENTE = 12
_NUM_MILHAR_RE = re.compile(r"[1-9]\d{0,2}(?:\.\d{3})+(?:,\d+)?(?!\d)")
_NUM_PLANO_RE = re.compile(r"\d+(?:[.,]\d+)?")
_RELOGIO_RE = re.compile(r"(\d{1,2})h(\d{2})(?!\d)")
_RELOGIO_ALT = re.compile(r"^(\d{1,2})h(\d{2})(?:\s*min)?$", re.I)


class _Invalido(Exception):
    """Termo sem valor aritmético (sintaxe, divisão por zero, overflow)."""


class _Tok(object):
    __slots__ = ("k", "v", "dec", "un", "pct", "relogio", "fim", "colado", "op", "ini")

    def __init__(self, k, fim, op=None, v=None, dec=0, un=None, pct=False,
                 relogio=False, colado=False, ini=0):
        self.k, self.fim, self.op, self.v, self.dec = k, fim, op, v, dec
        self.ini = ini
        self.un, self.pct, self.relogio, self.colado = un, pct, relogio, colado


def _fator_unidade(de, para):
    if de == para:
        return 1
    if (de, para) in _FATORES_UNIDADE:
        return Fraction(_FATORES_UNIDADE[(de, para)])
    if (para, de) in _FATORES_UNIDADE:
        return Fraction(1, _FATORES_UNIDADE[(para, de)])
    return None


def _ler_numero(s, i):
    """(valor Fraction, casas decimais, fim, relogio) do número que começa em s[i]."""
    m = _RELOGIO_RE.match(s, i)
    if m:
        minutos = int(m.group(1)) * 60 + int(m.group(2))
        if int(m.group(2)) < 60:
            return Fraction(minutos), 0, m.end(), True
    m = _NUM_MILHAR_RE.match(s, i)
    if m and "," in m.group(0) and m.end() - i <= 18:
        bruto = m.group(0).replace(".", "").replace(",", ".")
        return Fraction(bruto), len(bruto.split(".")[1]), m.end(), False
    m = _NUM_PLANO_RE.match(s, i)
    if m.end() - i > 18:  # gigante: nem converte (Fraction de 5000 dígitos estoura)
        return None, 0, m.end(), False
    bruto = m.group(0).replace(",", ".")
    casas = len(bruto.split(".")[1]) if "." in bruto else 0
    return Fraction(bruto), casas, m.end(), False


def _tokenizar(s):
    """Lista de _Tok: NUM, OP (+ - * / ^), LP, RP, EQ, AEQ e W (palavra/outro =
    quebra a expressão). Unidade, % e R$ ficam DENTRO do NUM."""
    toks, n, i = [], len(s), 0
    moeda = False
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c.isdigit() and c.isascii():
            v, casas, j, rel = _ler_numero(s, i)
            if v is None or j - i > 18:  # número gigante: não é conta de questão
                toks.append(_Tok("W", j))
                i = j
                continue
            un, pct = ("rs" if moeda else None), False
            moeda = False
            if not rel:
                k = j + 1 if j < n and s[j] == " " else j
                if k < n and s[k] == "%":
                    pct, j = True, k + 1
                elif k < n and s[k] in "°º":
                    un, j = "°", k + 1
                elif k < n and s[k].isalpha() and s[k] not in "xX":
                    f = k
                    while f < n and s[f].isalpha():
                        f += 1
                    palavra = s[k:f].lower()
                    f2 = f
                    while f2 < n and s[f2] in "²³":
                        f2 += 1
                    sup = s[f:f2]
                    proximo = f2
                    while proximo < n and s[proximo] == " ":
                        proximo += 1
                    canon = _UNIDADES_CANON.get(palavra + sup)
                    solta = proximo < n and s[proximo] in "=+-*/÷)≈xX^:·" and palavra != "de"
                    if canon or solta:
                        un = canon or un
                        j = f2
                        # unidade composta "km/h": consome "/palavra" colada
                        if j + 1 < n and s[j] == "/" and s[j + 1].isalpha():
                            g = j + 1
                            while g < n and s[g].isalpha():
                                g += 1
                            j, un = g, None
            toks.append(_Tok("NUM", j, v=v, dec=casas, un=un, pct=pct, relogio=rel, ini=i))
            if j < n and s[j] in "²³" and not rel:
                toks.append(_Tok("OP", j + 1, op="^"))
                toks.append(_Tok("NUM", j + 1, v=Fraction(2 if s[j] == "²" else 3)))
                j += 1
            i = j
        elif c == "R" and s[i:i + 2] == "R$":
            moeda, i = True, i + 2
        elif c in "+*÷·":
            toks.append(_Tok("OP", i + 1, op={"+": "+", "*": "*", "÷": "/", "·": "*"}[c]))
            i += 1
        elif c == "-":
            colado = i + 1 < n and s[i + 1].isdigit()
            toks.append(_Tok("OP", i + 1, op="-", colado=colado))
            i += 1
        elif c == "/":
            toks.append(_Tok("OP", i + 1, op="/"))
            i += 1
        elif c == "^":
            toks.append(_Tok("OP", i + 1, op="^"))
            i += 1
        elif c == ":" and 0 < i < n - 1 and s[i - 1] == " " and s[i + 1] == " ":
            toks.append(_Tok("OP", i + 1, op="/"))
            i += 1
        elif c in "([":
            toks.append(_Tok("LP", i + 1))
            i += 1
        elif c in ")]":
            toks.append(_Tok("RP", i + 1))
            i += 1
        elif c == "=":
            toks.append(_Tok("EQ", i + 1))
            i += 1
        elif c in "≈≅":
            toks.append(_Tok("AEQ", i + 1))
            i += 1
        elif c in "xX" and not (i > 0 and s[i - 1].isalpha()) and not (
                i + 1 < n and s[i + 1].isalpha()):
            k = i + 1
            while k < n and s[k] == " ":
                k += 1
            ant = toks[-1].k if toks else None
            if ant in ("NUM", "RP") and k < n and (s[k].isdigit() or s[k] in "(-R"):
                toks.append(_Tok("OP", i + 1, op="*"))
            else:
                toks.append(_Tok("W", i + 1))
            i += 1
        elif c.isalpha():
            f = i
            while f < n and s[f].isalpha():
                f += 1
            while f < n and s[f].isdigit() and s[f].isascii():
                f += 1
            palavra = s[i:f].lower()
            ant = toks[-1] if toks else None
            if palavra == "de" and ant is not None and ant.k == "NUM" and (
                    ant.pct or (len(toks) >= 3 and toks[-2].k == "OP" and toks[-2].op == "/"
                                and toks[-3].k == "NUM")):
                if ant.pct:
                    ant.v, ant.pct = ant.v / 100, False
                toks.append(_Tok("OP", f, op="*"))
            else:
                toks.append(_Tok("W", f))
            i = f
        else:
            toks.append(_Tok("W", i + 1))
            i += 1
    return toks


def _avaliar_termo(toks):
    """(valor, tem_conta, unidade, pct, casas, relogio) de uma lista de _Tok, ou None."""
    toks = list(toks)
    if sum(1 for t in toks if t.k == "LP") > 20:
        return None  # parênteses demais: não é conta de questão
    # traço de lista ("- 5 + 3 = 8"): separado do número, no começo do termo
    while toks and toks[0].k == "OP" and not (toks[0].op == "-" and toks[0].colado):
        toks = toks[1:]
    if not toks:
        return None
    pos, estado = [0], {"conta": False}

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def soma():
        v = produto()
        while peek() is not None and peek().k == "OP" and peek().op in "+-":
            op = toks[pos[0]].op
            pos[0] += 1
            w = produto()
            v = v + w if op == "+" else v - w
            estado["conta"] = True
        return v

    def produto():
        v = potencia()
        while peek() is not None and peek().k == "OP" and peek().op in "*/":
            op = toks[pos[0]].op
            pos[0] += 1
            w = potencia()
            if op == "*":
                v = v * w
            else:
                if w == 0:
                    raise _Invalido("divisao por zero")
                v = v / w
            estado["conta"] = True
            if abs(v) > _LIMITE_MODULO:
                raise _Invalido("overflow")
        return v

    def potencia():
        v = unario()
        if peek() is not None and peek().k == "OP" and peek().op == "^":
            pos[0] += 1
            e = unario()
            if e.denominator != 1 or e < 0 or e > _LIMITE_EXPOENTE:
                raise _Invalido("expoente")
            v = v ** int(e)
            estado["conta"] = True
            if abs(v) > _LIMITE_MODULO:
                raise _Invalido("overflow")
        return v

    def unario():
        t = peek()
        if t is not None and t.k == "OP" and t.op == "-":
            pos[0] += 1
            estado["conta"] = True
            return -unario()
        return atomo()

    def atomo():
        t = peek()
        if t is None:
            raise _Invalido("fim")
        if t.k == "NUM":
            pos[0] += 1
            return t.v
        if t.k == "LP":
            pos[0] += 1
            v = soma()
            f = peek()
            if f is None or f.k != "RP":
                raise _Invalido("parentese")
            pos[0] += 1
            estado["conta"] = True
            return v
        raise _Invalido("atomo")

    try:
        valor = soma()
    except (_Invalido, ArithmeticError, ValueError, RecursionError):
        return None
    if pos[0] != len(toks) or abs(valor) > _LIMITE_MODULO:
        return None
    nums = [t for t in toks if t.k == "NUM"]
    unidades = {t.un for t in nums if t.un}
    # termo de UM número guarda a unidade/%/casas dele; com conta, só se todas
    # as unidades coincidem (duas unidades misturadas => sem unidade).
    unidade = next(iter(unidades)) if len(unidades) == 1 else None
    if any(t.relogio for t in nums) and ("h" in unidades):
        return None  # relógio + duração em horas: ambíguo, não verifica
    return (valor, estado["conta"], unidade, nums[0].pct if len(nums) == 1 else False,
            nums[0].dec if len(nums) == 1 else None, any(t.relogio for t in nums))


def _iguais(esq, dir_, aprox):
    """Os termos avaliados `esq` e `dir_` valem o mesmo?  "=" é igualdade EXATA
    (depois de converter unidade pela tabela e "0,25 = 25%"). Só "≈" aceita
    arredondamento/truncamento: |dif| < 10^-k com k>=1 casas escritas no
    resultado, e |dif| < 1 sem casas."""
    lv, _c, lu, lp, _d, _r = esq
    rv, rconta, ru, rp, rdec, _rr = dir_
    cands = [lv]
    if rp and not lp:
        cands.append(lv * 100)
    if lp and not rp:
        cands.append(lv / 100)
    if lu and ru and lu != ru:
        f = _fator_unidade(lu, ru)
        if f is not None:
            cands.append(lv * f)
    for c in cands:
        if c == rv:
            return True
    # Tolerância de arredondamento SÓ para "≈". Com "=", a conta aproximada
    # ("10 ÷ 3 = 3,33") continua sem valor de verificação — decisão H2 de
    # tests/test_base3_decisoes.py (contrato congelado: aproximado nunca vira
    # "verificado" nem acusação).
    if not aprox or rconta or rdec is None:
        return False
    if rdec >= 1:
        return any(abs(c - rv) < Fraction(1, 10 ** rdec) for c in cands)
    return any(abs(c - rv) < 1 for c in cands)


def _analisar_contas(texto):
    """(registros, cadeias, spans) das contas CORRETAS do texto.

    registros: [(fim, valor Fraction, unidade, pct, casas)] em ordem de posição;
    cadeias:   [{"fim": int, "operandos": [Fraction...]}] com os operandos do
               PRIMEIRO termo de cada igualdade verificada (alimenta o aviso de
               "dado inventado");
    spans:     [(ini, fim)] dos números que participam de alguma igualdade
               verificada (para separar "usado em conta" de "mencionado")."""
    s = normalize_math(texto)[:20000]
    toks = _tokenizar(s)
    registros, cadeias, spans = [], [], []
    i, n = 0, len(toks)
    while i < n:
        if toks[i].k not in ("NUM", "OP", "LP", "RP", "EQ", "AEQ"):
            i += 1
            continue
        j = i
        while j < n and toks[j].k in ("NUM", "OP", "LP", "RP", "EQ", "AEQ"):
            j += 1
        seg = toks[i:j]
        i = j
        termos, sinais, atual = [], [], []
        for t in seg:
            if atual and atual[-1].k in ("NUM", "RP") and t.k in ("NUM", "LP"):
                # dois operandos sem operador ("15 3 x 2 = 6", "5 (…"): são
                # expressões distintas; o que veio antes fecha numa cadeia própria.
                termos.append(atual)
                sinais.append(None)
                atual = []
            if t.k in ("EQ", "AEQ"):
                termos.append(atual)
                sinais.append(t.k == "AEQ")
                atual = []
            else:
                atual.append(t)
        termos.append(atual)
        if len(termos) < 2:
            continue
        vals = [_avaliar_termo(t) if t else None for t in termos]
        estabelecido = False
        primeira = True
        for a in range(len(termos) - 1):
            esq, dir_ = vals[a], vals[a + 1]
            if sinais[a] is None:  # quebra entre expressões, não é igualdade
                estabelecido = False
                primeira = True
                continue
            if esq is None or dir_ is None:
                estabelecido = False
                continue
            if not (esq[1] or estabelecido):
                estabelecido = False
                continue
            if _iguais(esq, dir_, sinais[a]):
                estabelecido = True
                t_fim = termos[a + 1][-1]
                registros.append((t_fim.fim, dir_[0], dir_[2], dir_[3], dir_[4]))
                spans.extend((t.ini, t.fim) for t in termos[a] + termos[a + 1]
                             if t.k == "NUM")
                if primeira and esq[1]:
                    cadeias.append({"fim": t_fim.fim, "operandos": [
                        t.v for t in termos[a] if t.k == "NUM"],
                        "ini": min(t.ini for t in termos[a] if t.k == "NUM")})
                primeira = False
            else:
                estabelecido = False
    return registros, cadeias, spans


def _contas_ricas(texto):
    """Registros de _analisar_contas UNIDOS aos da regex antiga (superconjunto:
    nada que era resultado deixa de ser), ordenados por posição."""
    registros, _cadeias, _spans = _analisar_contas(texto)
    por_fim = {r[0]: r for r in registros}
    for match in _EXPR_PATTERN.finditer(normalize_math(texto)):
        a_str, op, b_str, r_str = match.groups()
        a, b, r = _to_number(a_str), _to_number(b_str), _to_number(r_str)
        if a is None or b is None or r is None:
            continue
        computed = _OPS[op](a, b)
        if computed is not None and abs(computed - r) < 1e-6 and match.end() not in por_fim:
            por_fim[match.end()] = (match.end(), Fraction(str(r)), None, False, None)
    return [por_fim[k] for k in sorted(por_fim)]


def _numero(valor):
    """Fraction -> int/float, como _to_number devolvia (para comparar com _leading_number)."""
    f = float(valor)
    return int(f) if f.is_integer() else f


def _resultados_binarios(text):
    """Só as contas 'a op b = r' da regex ANTIGA. Existe para que o ramo
    textual (4) de check_consistency_detalhado continue sendo alcançado
    exatamente pelas mesmas questões de antes de 2026-10-10: o motor estendido
    descobre mais contas, mas o ramo textual tem perfil de falso positivo
    próprio (ex.: 'corresponde_outra' em notação científica) e não foi o alvo."""
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
    # motor estendido (2026-10-10): contas n-árias/encadeadas/com unidade
    for registro in _contas_ricas(texto_normalizado):
        fim = max(fim, registro[0])
    return fim


def _valor_no_texto(texto, valores):
    """True se algum de `valores` aparece como número em QUALQUER posição do texto."""
    alvos = {v for v in valores if v is not None}
    return bool(alvos) and any(
        t.k == "NUM" and not t.relogio and _numero(t.v) in alvos
        for t in _tokenizar(normalize_math(texto)))


def _valor_mencionado(resolucao, valores):
    """True se algum de `valores` aparece na resolução como número FORA de uma
    igualdade verificada (ex.: "4x = 60 → x = 15 cm" antes de "15 + 30 + 27 = 72").

    Guarda de FALSO POSITIVO medida em 2026-10-10: a resolução acha a resposta
    por álgebra/prosa e depois ESCREVE UMA CONFERÊNCIA aritmética; a conferência
    vira o "último resultado" e a questão certa parecia ter a resposta fora das
    alternativas (train_curado_v3 linha 1336). Um valor que só aparece como
    OPERANDO de conta ("12 + 8 = 20" com gabarito 12) não conta como menção."""
    alvos = {v for v in valores if v is not None}
    if not alvos:
        return False
    texto = normalize_math(resolucao)
    _r, _c, spans = _analisar_contas(texto)
    for t in _tokenizar(texto):
        if t.k == "NUM" and not t.relogio and _numero(t.v) in alvos:
            if not any(ini <= t.ini and t.fim <= fim for ini, fim in spans):
                return True
    return False


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
    # 2026-10-10: aceita também "1.250,00" (milhar + vírgula decimal pt-BR),
    # que antes caía em "textual" e tirava a questão do ramo numérico.
    r"^(?:r\$\s*)?(-?(?:[1-9]\d{0,2}(?:\.\d{3})+,\d+|\d+(?:[.,]\d+)?))\s*"
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


# ---------------------------------------------------------------------------
# Comparação TOLERANTE entre o resultado da resolução e uma alternativa
# (2026-10-10). Só serve a ABSOLVER (gabarito que bate "aproximadamente") e a
# decidir "fora das alternativas"; a troca de letra por fix_gabarito continua
# exigindo igualdade exata, salvo letra ÚNICA por tolerância (ver abaixo).
#
# Tolerância explícita: formatação (1.250 = 1250 = "1250 reais" = "R$ 1.250,00";
# 0,25 = 25%; 2,5 m = 250 cm pela tabela de unidades) é igualdade EXATA depois
# de normalizar. Arredondamento só vale quando a alternativa tem MENOS casas
# decimais que o resultado: "33,3" ~ "33%" (|dif| < 10^-casas_da_alternativa);
# um resultado inteiro nunca "arredonda" para outra alternativa ("120" não casa
# com "121"; 0,5 não "arredonda" para 0). Arredondamento: meio para cima na
# casa da alternativa (7,75 ~ 7,8; 7,75 ~ 8); truncamento (0,875 -> 0,87) só quando a alternativa tem casas.
# Alternativa-fração aceita o decimal que a resolução escreveu arredondado
# ("0,33" ~ "1/3").
# ---------------------------------------------------------------------------

def _casas(x):
    """Casas decimais de uma Fraction (99 se a dízima não termina)."""
    d, k = x.denominator, 0
    if d == 1:
        return 0
    for fator in (2, 5):
        e = 0
        while d % fator == 0:
            d //= fator
            e += 1
        k = max(k, e)
    return k if d == 1 else 99


def _arredonda(x, k):
    """x arredondado a k casas, meio para cima (7,75 -> 7,8; 0,5 -> 1)."""
    n = abs(x) * 10 ** k + Fraction(1, 2)
    r = Fraction(n.numerator // n.denominator, 10 ** k)
    return -r if x < 0 else r


def _trunca(x, k):
    n = abs(x) * 10 ** k
    r = Fraction(n.numerator // n.denominator, 10 ** k)
    return -r if x < 0 else r


def _info_alternativa(texto):
    """Lista de {"v": Fraction, "dec": int|None, "un", "pct", "fracao"} com as
    leituras de uma alternativa de VALOR (numero/fracao); [] para as demais.
    "1.250" tem duas leituras (1,25 e 1250); "1.250,00" tem uma (1250)."""
    classe = classe_alternativa(texto)
    t = _texto_alternativa(texto)
    rel = _RELOGIO_ALT.match(t) if classe == "textual" else None
    if rel and int(rel.group(2)) < 60:  # horário "11h00" = 660 min do dia
        return [{"v": Fraction(int(rel.group(1)) * 60 + int(rel.group(2))), "dec": 0,
                 "un": None, "pct": False, "fracao": False}]
    if classe == "fracao":
        m = _FRACAO_ISOLADA.match(t)
        a, b = int(m.group(1)), int(m.group(2))
        out = [{"v": Fraction(a), "dec": 0, "un": None, "pct": False, "fracao": True}]
        if b:
            out.append({"v": Fraction(a, b), "dec": None, "un": None, "pct": False,
                        "fracao": True})
        return out
    if classe != "numero":
        return []
    toks = [tk for tk in _tokenizar(t) if tk.k == "NUM"]
    if not toks:
        return []
    tk = toks[0]
    out = [{"v": tk.v, "dec": tk.dec, "un": tk.un, "pct": tk.pct, "fracao": False}]
    m = _NUM_MILHAR_RE.search(t)
    if m and "," not in m.group(0):
        out.append({"v": Fraction(m.group(0).replace(".", "")), "dec": 0, "un": tk.un,
                    "pct": tk.pct, "fracao": False})
    return out


def _casa(registro, alt):
    """O resultado `registro` (de _contas_ricas) corresponde à leitura `alt`?"""
    _fim, v, un, pct, dec = registro
    cands = [v]
    if alt["pct"] and not pct:
        cands.append(v * 100)
    if pct and not alt["pct"]:
        cands.append(v / 100)
    if un and alt["un"] and un != alt["un"]:
        f = _fator_unidade(un, alt["un"])
        if f is not None:
            cands.append(v * f)
    a = alt["v"]
    for x in cands:
        if x == a:
            return True
        if alt["fracao"]:
            if dec is not None and dec >= 1 and abs(x - a) < Fraction(1, 10 ** dec):
                return True
        elif alt["dec"] is not None and _casas(x) > alt["dec"]:
            if _arredonda(x, alt["dec"]) == a or (
                    alt["dec"] >= 1 and _trunca(x, alt["dec"]) == a):
                return True  # arredondamento; ou truncamento (só com casas: 0,875 -> 0,87)
    return False


def _letras_que_casam(registro, infos):
    """Letras cuja alternativa casa (tolerante) com o registro. infos: {letra: [alt]}"""
    return [L for L, leituras in infos.items() if any(_casa(registro, a) for a in leituras)]


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
    registros = _contas_ricas(resolucao)
    resultados = [_numero(r[1]) for r in registros]
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
    rec_milhar = _contas_ricas(_sem_separador_milhar(resolucao))
    resultados_milhar = [_numero(r[1]) for r in rec_milhar]
    if resultados_milhar != resultados:
        if _leading_number(alternativas.get(gabarito)) in resultados_milhar:
            return True, None, "valor_bate_milhar"
        if any(v in resultados_milhar
               for v in _valores_alternativa(alternativas.get(gabarito))):
            return True, None, "valor_bate_milhar"

    # (2b) EXTENSÃO 2026-10-10: o gabarito bate com algum resultado só pela
    # tolerância documentada acima (arredondamento, %, unidade, milhar pt-BR).
    # Só absolve; nunca acusa.
    info_gab = _info_alternativa(alternativas.get(gabarito))
    if info_gab:
        if any(_casa(r, a) for r in registros + rec_milhar for a in info_gab):
            return True, None, "valor_bate_aprox"

    classes = {letra: classe_alternativa(txt) for letra, txt in alternativas.items()}
    # Horário "11h00" só ABSOLVE (2b acima, via _info_alternativa); NÃO entra no
    # ramo numérico: reclassificá-lo fazia "ponteiro no 8 = 40 min" (resposta
    # "9h40") virar resultado_na_cauda e regredir 2 questões que o ramo textual
    # aprovava (train_curado_v3 linhas 1731 e 1765).
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
        # Guarda de cauda TAMBÉM antes de trocar a letra (2026-10-10): se depois
        # da última conta verificada a resolução escreve o valor do gabarito
        # ("... 23,333...%, arredondada para 23,8%"), ela defende o gabarito e o
        # "final" que bateu com outra alternativa era só um passo (o total 30
        # de "12 + 8 + 7 + 3 = 30"); trocar a letra gravaria um gabarito errado.
        valores_gab = ([_leading_number(alternativas.get(gabarito))]
                       + list(_valores_alternativa(alternativas.get(gabarito))))
        for letra in de_valor:
            if final in _valores_alternativa(alternativas[letra]):
                if _valor_na_cauda(resolucao, valores_gab) or _valor_mencionado(resolucao, valores_gab):
                    return None, None, "resultado_na_cauda"
                return False, letra, "gabarito_errado"
        # EXTENSÃO 2026-10-10: o resultado final cai numa alternativa só pela
        # tolerância (33,3 ~ 33%; 1.250 = 1250; 0,25 = 25%). Letra ÚNICA => é o
        # gabarito certo e o marcado está errado; várias => ambíguo, não acusa.
        info_alts = {L: _info_alternativa(alternativas[L]) for L in de_valor}
        aprox = _letras_que_casam(registros[-1], info_alts)
        if len(aprox) == 1:
            if _valor_na_cauda(resolucao, valores_gab) or _valor_mencionado(resolucao, valores_gab):
                return None, None, "resultado_na_cauda"
            return False, aprox[0], "gabarito_errado_aprox"
        if len(aprox) > 1:
            return None, None, "ambiguo_aprox"
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
        if resultados_milhar and (any(
            resultados_milhar[-1] in _valores_alternativa(alternativas[L])
            for L in de_valor
        ) or _letras_que_casam(rec_milhar[-1], info_alts)):
            return None, None, "milhar_ambiguo"
        # REGRA E — guarda de cauda. A resolução pode escrever a resposta final
        # DEPOIS da última conta que _EXPR_PATTERN fecha (arredondamento,
        # cadeia de 3+ operandos, unidade entre "=" e o número). Se o valor do
        # gabarito está nessa cauda, o verificador parou de ler cedo demais e a
        # acusação é artefato — degrada para não verificável. Ver _valor_na_cauda.
        if _valor_na_cauda(resolucao, valores_gab) or _valor_mencionado(resolucao, valores_gab):
            return None, None, "resultado_na_cauda"
        # CONFERÊNCIA de dado do enunciado (2026-10-10, achado no val): "3x + 5 =
        # 26 → 3x7 + 5 = 21 + 5 = 26" (gabarito 7, certo). A última conta só
        # reproduz o 26 que o ENUNCIADO já dava ao substituir a resposta; o
        # "final" não é a resposta. Se o resultado final é um número do
        # enunciado, não dá para afirmar que a resposta falta.
        # Só vale se o valor do gabarito aparece na resolução (a substituição "3x7"):
        # sem isso, "10h20min = 10 x 60 + 20 = 620" com gabarito 1020 deixaria de
        # ser pego só porque 620 min é o horário do enunciado.
        if (registros[-1][1] in _numeros_do_enunciado_exatos(questao.get("enunciado"))
                and _valor_no_texto(resolucao, valores_gab)):
            return None, None, "resultado_e_dado_do_enunciado"
        # Unidade INCOMPATÍVEL (2026-10-10): o "final" é "120°" e as alternativas
        # são "12 cm" — o último cálculo foi um passo intermediário (ângulo
        # central), não a resposta. Não dá para afirmar que a resposta falta.
        unid_final = registros[-1][2]
        unid_alts = {a["un"] for L in de_valor for a in info_alts[L] if a["un"]}
        if unid_final and unid_alts and all(
                _fator_unidade(unid_final, u) is None for u in unid_alts):
            return None, None, "unidade_incompativel"
        return False, None, MOTIVO_FORA_DAS_ALTERNATIVAS

    # (4) RAMO TEXTUAL/REFERENCIAL — o gabarito é um rótulo, não um valor.
    # A verificação correta é por CORRESPONDÊNCIA entre a alternativa e a
    # conclusão da resolução (sua última frase).
    if not _resultados_binarios(resolucao):
        return None, None, "sem_conta_binaria"
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


# ---------------------------------------------------------------------------
# REGRA SUAVE (2026-10-10): "dado inventado". SÓ AVISA; nunca muda status.
#
# Um número usado como OPERANDO de uma conta da resolução que não está no
# enunciado, nem é resultado de passo anterior, nem é constante conhecida, nem
# número por extenso do enunciado, foi trazido do nada. Caso real (5º H17):
# enunciado "bolo de 45 minutos ... 10h30", resolução "10h30 + 30 minutos =
# 11h00" — o 30 não existe no enunciado (o certo seria 45; 11h15). A conta é
# internamente correta e 11h00 ESTÁ nas alternativas, então nenhuma regra de
# aritmética pega: só este aviso. Só olha os operandos do primeiro termo de cada
# igualdade verificada; o "10" e o "30" de um horário "10h30" NÃO são números
# soltos do enunciado (o horário vale 630 min).
# ---------------------------------------------------------------------------

_CONSTANTES_CONHECIDAS = frozenset(Fraction(x) for x in (
    "0", "1", "2", "3", "4", "7", "10", "12", "24", "60", "90", "100", "180", "360",
    "1000", "1/2", "157/50", "3927/1250"))  # ..., 0,5, 3,14, 3,1416
_NUMEROS_POR_EXTENSO = {
    "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5,
    "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10, "onze": 11, "doze": 12,
    "treze": 13, "catorze": 14, "quatorze": 14, "quinze": 15, "dezesseis": 16,
    "dezessete": 17, "dezoito": 18, "dezenove": 19, "vinte": 20, "trinta": 30,
    "quarenta": 40, "cinquenta": 50, "sessenta": 60, "setenta": 70, "oitenta": 80,
    "noventa": 90, "cem": 100, "cento": 100, "mil": 1000, "duzentos": 200,
    "trezentos": 300, "quatrocentos": 400, "quinhentos": 500, "seiscentos": 600,
    "setecentos": 700, "oitocentos": 800, "novecentos": 900, "dobro": 2, "triplo": 3, "quadruplo": 4,
    "metade": 2, "terco": 3, "quarto": 4, "duzia": 12, "meia": 2, "semana": 7,
    "dia": 24, "hora": 60, "bimestre": 2, "trimestre": 3, "semestre": 6, "ano": 12,
}


def _numeros_do_enunciado_exatos(enunciado):
    """Números escritos no enunciado (nas duas leituras do ponto), sem variações."""
    texto = normalize_math(enunciado)
    return {t.v for t in _tokenizar(texto) + _tokenizar(_sem_separador_milhar(texto))
            if t.k == "NUM" and t.v is not None}


def _numeros_do_enunciado(enunciado):
    """Conjunto de Fractions que o enunciado fornece: números escritos (nas duas
    leituras do ponto), números por extenso e as versões /100 e x100 (percentual)."""
    texto = normalize_math(enunciado)
    valores = set()
    for t in _tokenizar(texto) + _tokenizar(_sem_separador_milhar(texto)):
        if t.k == "NUM":
            valores.add(t.v)
            valores.add(t.v / 100)
            valores.add(t.v * 100)
            if not t.relogio and t.v.denominator == 1 and t.v > 9:
                # algoritmo posicional ("7 + 5 = 12, escreve 2 e vai 1"; "100 + 31"):
                # dígitos e valores posicionais do número dado não são dado novo
                digs = str(int(t.v))
                for k, d in enumerate(reversed(digs)):
                    valores.add(Fraction(int(d)))
                    valores.add(Fraction(int(d) * 10 ** k))
    for palavra in re.findall(r"[a-z]+", _sem_acento(texto.lower())):
        if palavra in _NUMEROS_POR_EXTENSO:
            valores.add(Fraction(_NUMEROS_POR_EXTENSO[palavra]))
    return valores


def avisos_consistencia(questao):
    """Lista de avisos NÃO BLOQUEANTES sobre a resolução. Hoje: "dado_inventado".
    Cada aviso: {"tipo": "dado_inventado", "valor": número, "posicao": int}."""
    if not isinstance(questao, dict):
        return []
    resolucao = questao.get("resolucao_passo_a_passo") or ""
    registros, cadeias, _spans = _analisar_contas(resolucao)
    tokens_res = [t for t in _tokenizar(normalize_math(resolucao)) if t.k == "NUM"]
    permitidos = _numeros_do_enunciado(questao.get("enunciado") or "") | _CONSTANTES_CONHECIDAS
    avisos, vistos = [], set()
    for cad in cadeias:
        anteriores = set()
        for reg in registros:
            if reg[0] <= cad["fim"]:
                anteriores.add(reg[1])
                anteriores.add(reg[1] / 100)
                anteriores.add(reg[1] * 100)
        # número já escrito na prosa ANTES desta conta ("A variação foi de 40
        # camisetas. 40/120 = ...") foi derivado pelo autor: não é dado do nada.
        anteriores.update(t.v for t in tokens_res if t.fim <= cad["ini"])
        for v in cad["operandos"]:
            if v in permitidos or v in anteriores or v in vistos:
                continue
            vistos.add(v)
            avisos.append({"tipo": "dado_inventado", "valor": _numero(v), "posicao": cad["fim"]})
    return avisos


def detalhe_consistencia(questao):
    """check_consistency_detalhado + avisos, num dict (campo de detalhe/relatório):
    {"consistente", "sugestao", "motivo", "avisos"}. Os avisos não alteram o veredito."""
    ok, sugestao, motivo = check_consistency_detalhado(questao)
    return {"consistente": ok, "sugestao": sugestao, "motivo": motivo,
            "avisos": avisos_consistencia(questao)}


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
