"""Padroniza maiúsculas/minúsculas de enunciados e alternativas (CAIXA ALTA -> caixa normal).

Por que existe: 478 dos 2759 enunciados de data/train_curado_v3.jsonl (e 646 dos
1020 do DB/questoes.db, quase todos do lote L2 e de itens antigos) estão em CAIXA
ALTA. O modelo de 1,7B aprendeu o estilo e o reproduz: 7 de 90 gerações do val sem
sufixo saíram em CAIXA ALTA (auditoria de 2026-10-10). Texto em caixa alta também
perde acentos e nomes próprios ("COMUNITARIA", "PECAS"), então a padronização
restaura caixa e acentos.

INVARIANTE DE SEGURANÇA (vale para o método determinístico e para o LLM): o texto novo
tem de ser IGUAL ao original depois de tirar acentos e ignorar maiúsculas e espaços
repetidos. Isto proíbe trocar palavra, número, sinal ou unidade; só caixa e acento
podem mudar. Resposta do LLM que viola o invariante é descartada e o item cai no
método determinístico (marcado "deterministico" no log, para revisão humana).

O que NUNCA é tocado: `resposta_correta`, `difficulty`, a ordem/letras das
alternativas, o prompt do usuário (role=user), `meta`, e qualquer item que não esteja
em caixa alta. Não escreve em data/train_curado.jsonl, data/train.jsonl nem val*.jsonl
(a saída é sempre um arquivo novo e versionado).

Uso:
    python src/padronizar_enunciado.py --entrada data/train_curado_v3.jsonl \\
        --saida data/train_curado_v4.jsonl                        # só determinístico, 0 chamadas
    python src/padronizar_enunciado.py ... --real --modelo sabia-4 --max-chamadas 600
        # Maritaca (MARITALK_API_KEY do .env, nunca impresso); teto DURO de chamadas;
        # resposta em cache (outputs/agentes/padronizacao_cache.jsonl): rodar de novo é grátis.
"""
import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CACHE_PADRAO = ROOT / "outputs" / "agentes" / "padronizacao_cache.jsonl"
PROIBIDOS_SAIDA = ("train_curado.jsonl", "train.jsonl")  # nunca sobrescrever
LIMIAR_CAIXA_ALTA = 0.80
MIN_LETRAS = 10

# --------------------------------------------------------------------------- detecção


def _letras(t):
    return [c for c in t if c.isalpha()]


def frac_maiuscula(texto):
    L = _letras(texto)
    return sum(c.isupper() for c in L) / len(L) if L else 0.0


def _tem_palavra(texto):
    """Alguma sequência de 3+ letras. '2N, 1L, 1S' (Norte/Leste/Sul) são códigos, não texto em caixa alta."""
    return re.search(r"[A-Za-zÀ-ÿ]{3,}", texto) is not None


def eh_caixa_alta(texto, min_letras=MIN_LETRAS):
    """Mesmo critério de observacoes_gate: >= 80% das letras maiúsculas e >= 10 letras (e ao menos uma palavra)."""
    L = _letras(texto)
    return len(L) >= min_letras and frac_maiuscula(texto) >= LIMIAR_CAIXA_ALTA and _tem_palavra(texto)


def _alternativa_em_caixa_alta(texto):
    """Alternativa curta ('10 PONTOS.') não alcança 10 letras: usa 3 e exige TODA maiúscula."""
    L = _letras(texto)
    return len(L) >= 3 and frac_maiuscula(texto) >= 0.95 and _tem_palavra(texto)


# --------------------------------------------------------------------------- invariante


def _sem_acento(t):
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def forma_canonica(t):
    """Texto sem acento, minúsculo, com espaços colapsados: o que o invariante compara."""
    return re.sub(r"\s+", " ", _sem_acento(t).casefold()).strip()


def respeita_invariante(original, novo):
    return isinstance(novo, str) and forma_canonica(original) == forma_canonica(novo)


# --------------------------------------------------------------------------- determinístico

# Unidades que a minúscula destrói e como restaurá-las (depois de número).
_UNIDADES = [(re.compile(r"(?<=\d)(\s*)ml\b", re.I), r"\1mL"), (re.compile(r"(?<=\d)(\s*)l\b(?!\w)", re.I), r"\1L")]
_SUBSTANTIVO_GEOM = re.compile(
    r"\b(ponto|pontos|vértice|vértices|vertice|vertices|ângulo|angulo|triângulo|triangulo|"
    r"quadrilátero|quadrilatero|segmento|reta|retas|lado|lados)\s+([a-z]{1,4}(?:\s*(?:,|e)\s*[a-z]{1,4})*)(?![\wà-ÿ])")
# Palavras curtas que NUNCA são rótulo de figura (o rótulo vem depois de ponto/lado/...).
_NAO_ROTULO = frozenset("de da do das dos em na no nas nos que com por ao aos os as se um uma tem são sao ser foi era e ou "
                        "mas mais para sem sob tal cada dois duas três tres mede iguais reto reta maior menor".split())


def _eh_rotulo(tok):
    """Rótulo de figura: 1 letra, ou 2-4 letras DISTINTAS em ordem alfabética (AB, ABC, ABCD)."""
    if tok in _NAO_ROTULO:
        return False
    if len(tok) == 1:
        return True
    return list(tok) == sorted(set(tok))


def _maiuscula_rotulos(m):
    resto = re.sub(r"[a-z]{1,4}", lambda x: x.group(0).upper() if _eh_rotulo(x.group(0)) else x.group(0), m.group(2))
    return m.group(1) + " " + resto


_FIM_FRASE = re.compile(r"([.!?]\s+)([a-zà-ÿ])")


def nomes_proprios_do_corpus(textos):
    """Palavras que aparecem capitalizadas no MEIO de frase em texto de caixa mista, nunca
    em minúscula, e ao menos 2 vezes (João, Maria, Brasil). Dicionário do fallback."""
    cap, minus = Counter(), Counter()
    for t in textos:
        if eh_caixa_alta(t):
            continue
        for i, m in enumerate(re.finditer(r"[A-Za-zÀ-ÿ]+", t)):
            w = m.group(0)
            if len(w) < 3:
                continue
            ini_frase = m.start() == 0 or re.search(r"[.!?:]\s*$", t[:m.start()])
            if w[0].isupper() and w[1:].islower() and not ini_frase:
                cap[w.lower()] += 1
            elif w.islower():
                minus[w] += 1
    return {w for w, n in cap.items() if n >= 2 and minus[w] == 0}


def normalizar_deterministico(texto, nomes=frozenset()):
    """Minúsculas + inicial de frase + nomes próprios do corpus + unidades + rótulos
    geométricos em contexto. Não restaura acentos (sem dicionário); é o fallback."""
    t = texto.lower()
    # rótulos de figura primeiro (casam em minúscula); a inicial de frase vem por último
    t = _SUBSTANTIVO_GEOM.sub(_maiuscula_rotulos, t)
    for rx, sub in _UNIDADES:
        t = rx.sub(sub, t)
    if nomes:
        t = re.sub(r"[a-zà-ÿ]+", lambda m: m.group(0).capitalize() if m.group(0) in nomes else m.group(0), t)
    t = t[:1].upper() + t[1:] if t else t
    t = _FIM_FRASE.sub(lambda m: m.group(1) + m.group(2).upper(), t)
    assert respeita_invariante(texto, t), "bug: o método determinístico violou o invariante"
    return t


# --------------------------------------------------------------------------- LLM (Maritaca)

SISTEMA_PADRONIZAR = (
    "Você revisa a grafia de questões de matemática do ensino básico brasileiro. O texto veio "
    "em CAIXA ALTA e, às vezes, sem acentos. Devolva o MESMO conteúdo com maiúsculas e minúsculas "
    "normais do português (inicial de frase, nomes próprios como João e Brasil, siglas e rótulos "
    "de figuras como ABC, A, B em maiúscula; unidades como cm, kg, mL, L) e com os acentos e cedilhas "
    "restaurados. PROIBIDO: trocar, acrescentar ou remover palavras, números, sinais, unidades ou "
    "pontuação; corrigir o conteúdo matemático; reordenar alternativas. Responda APENAS com JSON "
    'no mesmo formato recebido: {"enunciado": str, "alternativas": {"A": str, ...}}.'
)


def _hash(modelo, payload):
    return hashlib.sha256((modelo + "\x1e" + SISTEMA_PADRONIZAR + "\x1e" + payload).encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, caminho):
        self.caminho = Path(caminho)
        self.d = {}
        if self.caminho.exists():
            for linha in self.caminho.read_text(encoding="utf-8").splitlines():
                if linha.strip():
                    r = json.loads(linha)
                    self.d[r["chave"]] = r["texto"]

    def get(self, chave):
        return self.d.get(chave)

    def put(self, chave, texto, modelo):
        self.d[chave] = texto
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        with open(self.caminho, "a", encoding="utf-8") as f:
            f.write(json.dumps({"chave": chave, "modelo": modelo, "texto": texto,
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False) + "\n")


def _extrair_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class Padronizador:
    def __init__(self, modelo=None, cliente=None, cache=None, max_chamadas=0, nomes=frozenset()):
        self.modelo, self.cliente, self.cache = modelo, cliente, cache
        self.max_chamadas, self.chamadas, self.nomes = max_chamadas, 0, nomes
        self.contagem = Counter()

    def _chamar(self, payload):
        chave = _hash(self.modelo, payload)
        if self.cache is not None and self.cache.get(chave) is not None:
            self.contagem["cache"] += 1
            return self.cache.get(chave)
        if self.chamadas >= self.max_chamadas:
            self.contagem["teto_de_chamadas"] += 1
            return None
        self.chamadas += 1
        for tentativa in range(3):
            try:
                r = self.cliente.chat_completion(
                    [{"role": "system", "content": SISTEMA_PADRONIZAR}, {"role": "user", "content": payload}],
                    max_tokens=1200, temperature=0.0)
                txt = r.choices[0].message.content
                if self.cache is not None:
                    self.cache.put(chave, txt, self.modelo)
                return txt
            except Exception as e:  # noqa: BLE001 — erro de rede/cota: tenta de novo, nunca imprime a chave
                self.contagem["erro_api"] += 1
                time.sleep(2 * (tentativa + 1))
                if tentativa == 2:
                    self.erro = type(e).__name__
        return None

    def questao(self, q):
        """Retorna (questao_nova, log). Só mexe nos campos em caixa alta."""
        campos = {}
        if eh_caixa_alta(q.get("enunciado", "")):
            campos["enunciado"] = q["enunciado"]
        em_alta = {L: v for L, v in q.get("alternativas", {}).items() if _alternativa_em_caixa_alta(v)}
        if campos or em_alta:
            # alternativas só são normalizadas quando em caixa alta; as demais ficam como estão
            campos["alternativas"] = em_alta
        if not campos:
            return q, None
        novo = json.loads(json.dumps(q))
        metodo = "deterministico"
        llm = None
        if self.cliente is not None and self.modelo:
            payload = json.dumps({"enunciado": q["enunciado"], "alternativas": q["alternativas"]}, ensure_ascii=False)
            resp = self._chamar(payload)
            j = _extrair_json(resp) if resp else None
            if j and isinstance(j.get("alternativas"), dict) and set(j["alternativas"]) == set(q["alternativas"]) \
                    and respeita_invariante(q["enunciado"], j.get("enunciado")) \
                    and all(respeita_invariante(q["alternativas"][L], j["alternativas"][L]) for L in q["alternativas"]):
                llm = j
                metodo = self.modelo
            elif resp:
                self.contagem["llm_violou_invariante"] += 1
        if llm:
            if "enunciado" in campos:
                novo["enunciado"] = llm["enunciado"]
            for L in em_alta:
                novo["alternativas"][L] = llm["alternativas"][L]
        else:
            if "enunciado" in campos:
                novo["enunciado"] = normalizar_deterministico(q["enunciado"], self.nomes)
            for L, v in em_alta.items():
                novo["alternativas"][L] = normalizar_deterministico(v, self.nomes)
        if "resolucao_passo_a_passo" in q and eh_caixa_alta(q["resolucao_passo_a_passo"]):
            novo["resolucao_passo_a_passo"] = normalizar_deterministico(q["resolucao_passo_a_passo"], self.nomes)
        self.contagem[metodo] += 1
        return novo, {"metodo": metodo,
                      "antes": {"enunciado": q.get("enunciado"), "alternativas": q.get("alternativas")},
                      "depois": {"enunciado": novo["enunciado"], "alternativas": novo["alternativas"]}}


# --------------------------------------------------------------------------- arquivo


def processar(entrada, saida, padronizador, log_path=None):
    linhas = [json.loads(l) for l in open(entrada, encoding="utf-8") if l.strip()]
    textos = []
    for r in linhas:
        for q in json.loads(r["messages"][2]["content"])["questoes"]:
            textos.append(q["enunciado"])
            textos += list(q["alternativas"].values())
    padronizador.nomes = nomes_proprios_do_corpus(textos)
    log, saidas, alteradas = [], [], 0
    for i, r in enumerate(linhas):
        obj = json.loads(r["messages"][2]["content"])
        mudou = False
        for j, q in enumerate(obj["questoes"]):
            novo, info = padronizador.questao(q)
            if info:
                obj["questoes"][j] = novo
                mudou = True
                log.append({"linha": i, "codigo_item": r.get("meta", {}).get("codigo_item"), "questao": j, **info})
        if mudou:
            alteradas += 1
            r = json.loads(json.dumps(r))
            r["messages"][2]["content"] = json.dumps(obj, ensure_ascii=False)
        saidas.append(r)
    saida = Path(saida)
    if saida.name in PROIBIDOS_SAIDA or saida.name.startswith("val") or saida.resolve() == Path(entrada).resolve():
        raise SystemExit(f"ABORTADO: não escrevo em {saida.name} (arquivo protegido ou igual à entrada).")
    saida.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in saidas), encoding="utf-8")
    if log_path:
        Path(log_path).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in log), encoding="utf-8")
    return {"exemplos": len(linhas), "exemplos_alterados": alteradas, "questoes_alteradas": len(log),
            "metodos": dict(padronizador.contagem), "chamadas_api": padronizador.chamadas}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default=str(ROOT / "data" / "train_curado_v3.jsonl"))
    ap.add_argument("--saida", default=str(ROOT / "data" / "train_curado_v4.jsonl"))
    ap.add_argument("--log", default=str(ROOT / "outputs" / "relatorios" / "padronizacao_log.jsonl"))
    ap.add_argument("--real", action="store_true", help="usa a Maritaca (padrão: só determinístico, 0 chamadas)")
    ap.add_argument("--modelo", default="sabia-4")
    ap.add_argument("--max-chamadas", type=int, default=0, help="teto DURO de chamadas (obrigatório com --real)")
    args = ap.parse_args()
    cliente = cache = None
    if args.real:
        if args.max_chamadas <= 0:
            sys.exit("ABORTADO: --real exige --max-chamadas N > 0 (teto duro).")
        import distill_teacher as dt  # carrega o .env; a chave nunca é impressa
        cliente, cache = dt.MaritacaClient(args.modelo), Cache(CACHE_PADRAO)
    p = Padronizador(args.modelo if args.real else None, cliente, cache, args.max_chamadas)
    print(json.dumps(processar(args.entrada, args.saida, p, args.log), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
