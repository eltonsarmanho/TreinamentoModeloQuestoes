"""Tarefa F (medição precisa do 4B + calibração de juiz): utilidades compartilhadas.

Sem chamada de API, sem GPU. Só lê arquivos e calcula.
"""
import csv
import hashlib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import diversidade as dv  # noqa: E402
import schema_utils as su  # noqa: E402
from unidades_tematicas import UNIDADES, unidade_de  # noqa: E402

LETRAS = ("A", "B", "C", "D", "E")
DIMENSOES = ("correcao_matematica", "resposta_unica", "dados_suficientes",
             "aderencia_ano_habilidade", "contexto_plausivel", "dificuldade_adequada")
ROTULOS = DIMENSOES + ("valida_geral",)
NUCLEO = ("correcao_matematica", "resposta_unica", "dados_suficientes")
SAIDA = RAIZ / "outputs" / "melhoria_v8" / "F"


def ler_csv(caminho):
    """Linhas de uma folha (utf-8-sig, como o Excel salva)."""
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def questao_de_linha(linha):
    """Linha da folha -> dict no formato de questão do projeto."""
    return {
        "enunciado": linha.get("enunciado", ""),
        "alternativas": {l: linha.get(l, "") for l in LETRAS},
        "resolucao_passo_a_passo": linha.get("resolucao", ""),
        "resposta_correta": linha.get("gabarito_modelo", ""),
    }


def assinatura(q):
    """Mesma questão = mesmo enunciado + alternativas (sem caixa/acento/espaço).
    Idêntica à de amostra_humana._assinatura, para ser comparável com as folhas antigas."""
    texto = dv.normalizar_texto(dv.texto_questao(q, com_alternativas=True))
    return hashlib.sha1(texto.encode()).hexdigest()


def status_verificador(q):
    """'ok' | 'nao_verificavel' | 'inconsistente:<motivo>' pelo verificador atual."""
    try:
        ok, _sug, motivo = su.check_consistency_detalhado(q)
    except Exception as e:  # noqa: BLE001 — o verificador nunca pode derrubar a coleta
        return f"erro:{type(e).__name__}"
    if ok is True:
        return "ok"
    if ok is None:
        return "nao_verificavel"
    return f"inconsistente:{motivo}"


def sn(valor):
    """'s'/'n' -> True/False; qualquer outra coisa -> None."""
    v = str(valor or "").strip().lower()
    if v in ("s", "sim"):
        return True
    if v in ("n", "nao", "não"):
        return False
    return None


def valida_nucleo(rotulos):
    """correcao_matematica ∧ resposta_unica ∧ dados_suficientes (s/n já em bool).
    None se alguma das três faltar."""
    vals = [rotulos.get(k) for k in NUCLEO]
    if any(v is None for v in vals):
        return None
    return all(vals)


def unidade_da_linha(ano, habilidade):
    return unidade_de(ano, habilidade) if ano and habilidade else None


__all__ = ["RAIZ", "SRC", "SAIDA", "LETRAS", "DIMENSOES", "ROTULOS", "NUCLEO", "UNIDADES",
           "ler_csv", "questao_de_linha", "assinatura", "status_verificador", "sn",
           "valida_nucleo", "unidade_da_linha"]
