"""Validação do schema JSON de questão — sem dependências pesadas de ML.

Compartilhado por evaluate.py (avalia o modelo em 4-bit via HF/bitsandbytes,
usado em desenvolvimento) e test_model.py (testa o .gguf real via llama.cpp,
o mesmo artefato que roda no app mobile), garantindo que os dois caminhos
julguem "resposta válida" da mesma forma.
"""

import json
import re

# `resposta` (o VALOR da resposta correta, ex.: "5") vem antes de `gabarito`
# (a LETRA) no schema. Isso transforma a verificação de "extrair a conta do
# texto livre da justificativa com regex" — que falhava em 70-77% dos casos,
# seja por operadores Unicode, seja por raciocínio verbal sem equação — em
# uma comparação exata `alternativas[gabarito] == resposta`.
REQUIRED_KEYS = {"enunciado", "comando", "alternativas", "resposta", "gabarito"}
IMAGE_PATTERN = re.compile(r"\b(figura|imagem|gráfico|desenho|ilustração)\b", re.I)


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


def check_structure(obj):
    """Retorna dict de flags estruturais para um JSON parseado (ou None)."""
    flags = {
        "json_valido": obj is not None,
        "schema_completo": False,
        "gabarito_valido": False,
        "alternativas_distintas": False,
        "justificativas_distintas": False,
    }
    if obj is None:
        return flags
    flags["schema_completo"] = REQUIRED_KEYS.issubset(obj.keys())
    flags["gabarito_valido"] = obj.get("gabarito") in {"A", "B", "C", "D"}
    alts = obj.get("alternativas")
    if isinstance(alts, dict) and set(alts.keys()) >= {"A", "B", "C", "D"}:
        values = [str(alts[k]).strip() for k in "ABCD"]
        flags["alternativas_distintas"] = len(set(values)) == 4 and all(values)
    just = obj.get("justificativas")
    if isinstance(just, dict) and set(just.keys()) >= {"A", "B", "C", "D"}:
        values = [str(just[k]).strip() for k in "ABCD"]
        flags["justificativas_distintas"] = len(set(values)) == 4 and all(values)
    return flags


_NUM_PATTERN = re.compile(r"-?\d+(?:[.,]\d+)?")

# O modelo emite operadores tipográficos Unicode (−, ×, –) em vez dos ASCII
# usados no dataset de treino. Sem normalizar, o regex abaixo não casa e o
# verificador devolve "não verificável" — silenciosamente deixando passar
# gabaritos errados. Medido: 70-77% das saídas caíam nesse buraco.
_UNICODE_MATH = {
    "−": "-",  # MINUS SIGN
    "–": "-",  # EN DASH
    "—": "-",  # EM DASH
    "×": "x",  # MULTIPLICATION SIGN
    "⋅": "x",  # DOT OPERATOR
    "∙": "x",  # BULLET OPERATOR
    "∕": "/",  # DIVISION SLASH
    "＝": "=",  # FULLWIDTH EQUALS
    " ": " ",  # NO-BREAK SPACE
    " ": " ",  # NARROW NO-BREAK SPACE
}
_UNICODE_TABLE = str.maketrans(_UNICODE_MATH)


def normalize_math(text):
    """Converte operadores matemáticos Unicode para os equivalentes ASCII."""
    return str(text or "").translate(_UNICODE_TABLE)


# Casa expressões simples "a op b = r" dentro do texto da justificativa,
# ex.: "35 - 20 = 15", "3x4=12", "10/2 = 5" (após normalize_math).
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


def _leading_number(text):
    match = _NUM_PATTERN.search(normalize_math(text))
    return _to_number(match.group()) if match else None


def _canon(text):
    """Forma canônica para comparar alternativa com `resposta`: minúsculas, sem
    espaços redundantes, operadores normalizados. Mantém unidades ("5 bolas"),
    que o casamento textual exato usa e o fallback numérico ignora."""
    return re.sub(r"\s+", " ", normalize_math(text).strip().lower())


def _match_letra(alternativas, valor):
    """Acha a letra cuja alternativa corresponde a `valor`.

    Duas passadas, da mais estrita para a mais tolerante:
      1. igualdade textual canônica ("3/4" == "3/4", "20%" == "20%")
      2. igualdade do primeiro número ("5" ~ "5 bonecos")
    A segunda só decide se casar com UMA única alternativa — se duas
    alternativas começam com o mesmo número, é ambíguo e devolvemos None em
    vez de escolher errado.
    """
    alvo = _canon(valor)
    exatas = [l for l, t in alternativas.items() if _canon(t) == alvo]
    if len(exatas) == 1:
        return exatas[0]

    num = _leading_number(valor)
    if num is None:
        return None
    numericas = [l for l, t in alternativas.items() if _leading_number(t) == num]
    return numericas[0] if len(numericas) == 1 else None


def check_consistency(obj):
    """Confere se o `gabarito` (a LETRA) aponta para a alternativa que contém a
    `resposta` (o VALOR).

    Estratégia em duas vias:
      1. **Exata** — se o JSON traz o campo `resposta`, basta comparar
         `alternativas[gabarito]` com ele. Não depende de regex, de operador
         Unicode nem de como a justificativa foi redigida: cobre qualquer tipo
         de questão (fração, porcentagem, texto).
      2. **Fallback por regex** — para saídas de modelos antigos, sem o campo
         `resposta`, extrai uma conta "a op b = r" da justificativa do gabarito.
         Cobria só ~25% dos casos; é a razão de existir a via 1.

    Retorna (ok, sugestao):
      ok=True         gabarito aponta para a alternativa correta
      ok=False        não aponta — `sugestao` traz a letra correta, se identificável
      ok=None         não deu para verificar por nenhuma das duas vias
    """
    if not obj:
        return None, None
    gabarito = obj.get("gabarito")
    alternativas = obj.get("alternativas")
    if not isinstance(alternativas, dict) or gabarito not in alternativas:
        return None, None

    # Via 1: casamento exato contra o campo `resposta`.
    resposta = obj.get("resposta")
    if resposta not in (None, ""):
        letra = _match_letra(alternativas, resposta)
        if letra is not None:
            return (True, None) if letra == gabarito else (False, letra)
        # `resposta` não casou com nenhuma alternativa. Dois cenários bem
        # diferentes, e confundi-los gera falso negativo:
        #   (a) modelo NÃO retreinado ecoa a LETRA ("B") no campo, porque a
        #       grammar o obriga a preencher algo que ele nunca aprendeu —
        #       não é erro da questão; caímos para a via 2.
        #   (b) modelo retreinado declara um VALOR que não existe entre as
        #       alternativas — aí a questão está de fato malformada.
        if _canon(resposta) not in {"a", "b", "c", "d"}:
            return False, None

    # Via 2 (legado / modelo não retreinado): extrai a conta da justificativa.
    justificativas = obj.get("justificativas")
    if not isinstance(justificativas, dict):
        return None, None
    result = _computed_result(justificativas.get(gabarito, ""))
    if result is None:
        return None, None
    if _leading_number(alternativas.get(gabarito)) == result:
        return True, None
    for letra, texto in alternativas.items():
        if _leading_number(texto) == result:
            return False, letra
    return False, None


def fix_gabarito(obj):
    """Correção determinística pós-geração (para o app e para os scripts de teste).

    Se o gabarito não aponta para a alternativa que contém a `resposta` (ou,
    no fallback legado, o resultado da conta), troca a letra do gabarito para a
    correta. Retorna (obj, status):
      "ok"              gabarito já aponta para a alternativa certa
      "corrigido"       letra do gabarito trocada para a alternativa certa
      "inconsistente"   a resposta/conta não corresponde a nenhuma alternativa —
                        questão malformada, regenerar é o único remédio
      "nao_verificavel" sem `resposta` nem conta reconhecível — segue como está

    Nota: ao corrigir, apenas a letra do gabarito muda; as justificativas
    permanecem onde estão. A letra corrigida é a pedagogicamente correta
    (a alternativa cujo valor é de fato a resposta).
    """
    consistente, sugestao = check_consistency(obj)
    if consistente is True:
        return obj, "ok"
    if consistente is None:
        return obj, "nao_verificavel"
    if sugestao:
        corrigido = dict(obj)
        corrigido["gabarito"] = sugestao
        return corrigido, "corrigido"
    return obj, "inconsistente"
