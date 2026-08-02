"""Validação do schema JSON de questão — sem dependências pesadas de ML.

Compartilhado por evaluate.py (avalia o modelo em 4-bit via HF/bitsandbytes,
usado em desenvolvimento) e test_model.py (testa o .gguf real via llama.cpp,
o mesmo artefato que roda no app mobile), garantindo que os dois caminhos
julguem "resposta válida" da mesma forma.
"""

import json
import re

REQUIRED_KEYS = {"enunciado", "comando", "alternativas", "gabarito"}
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
# Casa expressões simples "a op b = r" dentro do texto da justificativa,
# ex.: "35 - 20 = 15", "3x4=12", "10/2 = 5".
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
    match = _EXPR_PATTERN.search(text or "")
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
    match = _NUM_PATTERN.search(str(text or ""))
    return _to_number(match.group()) if match else None


def check_consistency(obj):
    """Confere se o `gabarito` é a alternativa cujo valor bate com a conta
    resolvida na justificativa correspondente.

    Retorna (ok, sugestao):
      ok=True         conta da justificativa do gabarito bate com o valor da alternativa
      ok=False        não bate — `sugestao` traz a letra da alternativa correta, se achada
      ok=None         não deu pra verificar (sem uma conta "a op b = r" clara no texto)

    Heurística best-effort para questões de aritmética simples; não substitui
    a correção humana/curadoria dos dados de treino.
    """
    if not obj:
        return None, None
    gabarito = obj.get("gabarito")
    justificativas = obj.get("justificativas")
    alternativas = obj.get("alternativas")
    if not isinstance(justificativas, dict) or not isinstance(alternativas, dict):
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

    Se a conta resolvida na justificativa do gabarito bate com o valor de OUTRA
    alternativa, troca o gabarito para essa letra. Retorna (obj, status):
      "ok"              gabarito consistente com a conta — nada a fazer
      "corrigido"       gabarito trocado para a letra cujo valor bate com a conta
      "inconsistente"   conta não bate com nenhuma alternativa — regenerar é o
                        único remédio
      "nao_verificavel" sem conta "a op b = r" reconhecível — segue como está

    Nota: ao corrigir, apenas a letra do gabarito muda; a justificativa que
    contém a conta continua na letra original. A letra corrigida é a
    pedagogicamente correta (valor == resultado da conta).
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
