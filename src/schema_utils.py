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
    }
    if obj is None:
        return flags
    flags["schema_completo"] = REQUIRED_KEYS.issubset(obj.keys())
    flags["gabarito_valido"] = obj.get("gabarito") in {"A", "B", "C", "D"}
    alts = obj.get("alternativas")
    if isinstance(alts, dict) and set(alts.keys()) >= {"A", "B", "C", "D"}:
        values = [str(alts[k]).strip() for k in "ABCD"]
        flags["alternativas_distintas"] = len(set(values)) == 4 and all(values)
    return flags
