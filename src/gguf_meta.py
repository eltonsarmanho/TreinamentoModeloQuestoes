"""Lê `general.architecture` do cabeçalho de um .gguf (só stdlib, sem o pacote gguf).

Usado por tests/test_model.py para gravar `familia_modelo` no relatório e por
promover_checkpoint.py para recusar comparar famílias diferentes (Gemma x Qwen).
"""
import struct
from pathlib import Path

_MAGIC = b"GGUF"
_TAM_ESCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY = 8, 9


def _ler(f, n):
    dados = f.read(n)
    if len(dados) != n:
        raise ValueError("gguf truncado")
    return dados


def _string(f):
    (n,) = struct.unpack("<Q", _ler(f, 8))
    return _ler(f, n).decode("utf-8", errors="replace")


def _pular(f, tipo):
    if tipo in _TAM_ESCALAR:
        f.seek(_TAM_ESCALAR[tipo], 1)
    elif tipo == _STRING:
        (n,) = struct.unpack("<Q", _ler(f, 8))
        f.seek(n, 1)
    elif tipo == _ARRAY:
        sub, n = struct.unpack("<IQ", _ler(f, 12))
        if sub in _TAM_ESCALAR:
            f.seek(_TAM_ESCALAR[sub] * n, 1)
        else:
            for _ in range(n):
                _pular(f, sub)
    else:
        raise ValueError(f"tipo gguf desconhecido: {tipo}")


def metadados(path, chaves=("general.architecture", "general.name")):
    """Devolve {chave: valor} para as chaves string pedidas que existirem no cabeçalho."""
    achadas = {}
    with open(path, "rb") as f:
        if _ler(f, 4) != _MAGIC:
            raise ValueError(f"{path} não é um arquivo gguf")
        versao, _, n_kv = struct.unpack("<IQQ", _ler(f, 20))
        if versao < 2:
            raise ValueError(f"gguf versão {versao} não suportada")
        for _ in range(n_kv):
            chave = _string(f)
            (tipo,) = struct.unpack("<I", _ler(f, 4))
            if chave in chaves and tipo == _STRING:
                achadas[chave] = _string(f)
                if len(achadas) == len(chaves):
                    break
            else:
                _pular(f, tipo)
    return achadas


def arquitetura(path):
    """'gemma3', 'qwen3', ... ou None se o arquivo não puder ser lido."""
    try:
        return metadados(Path(path), ("general.architecture",)).get("general.architecture")
    except (OSError, ValueError, struct.error):
        return None
