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

IMAGE_PATTERN = re.compile(r"\b(figura|imagem|gráfico|desenho|ilustração)\b", re.I)

QUESTOES_KEY = "questoes"
ALTERNATIVE_LETTERS = "ABCDE"
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
        isinstance(q, dict) and q.get("resposta_correta") in ALTERNATIVE_LETTERS
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


def _leading_number(text):
    match = _NUM_PATTERN.search(normalize_math(text))
    return _to_number(match.group()) if match else None


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
    """
    if not isinstance(questao, dict):
        return None, None
    gabarito = questao.get("resposta_correta")
    alternativas = questao.get("alternativas")
    if not isinstance(alternativas, dict) or gabarito not in alternativas:
        return None, None

    resultado = _computed_result(questao.get("resolucao_passo_a_passo", ""))
    if resultado is None:
        return None, None
    if _leading_number(alternativas.get(gabarito)) == resultado:
        return True, None
    for letra, texto in alternativas.items():
        if _leading_number(texto) == resultado:
            return False, letra
    return False, None


def fix_gabarito(questao):
    """Correção determinística pós-geração (para o app e para os scripts de teste).

    Se `resposta_correta` não aponta para a alternativa que bate com a conta
    de `resolucao_passo_a_passo`, troca a letra pela correta. Retorna
    (questao, status):
      "ok"              resposta_correta já aponta para a alternativa certa
      "corrigido"       letra trocada para a alternativa certa
      "inconsistente"   a conta não corresponde a nenhuma alternativa —
                        questão malformada, regenerar é o único remédio
      "nao_verificavel" sem conta reconhecível em resolucao_passo_a_passo

    Nota: só a letra de `resposta_correta` muda; `resolucao_passo_a_passo`
    permanece como está.
    """
    consistente, sugestao = check_consistency(questao)
    if consistente is True:
        return questao, "ok"
    if consistente is None:
        return questao, "nao_verificavel"
    if sugestao:
        corrigido = dict(questao)
        corrigido["resposta_correta"] = sugestao
        return corrigido, "corrigido"
    return questao, "inconsistente"
