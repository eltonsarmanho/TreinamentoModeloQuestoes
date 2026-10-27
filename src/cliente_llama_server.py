"""Cliente mínimo (só biblioteca padrão do Python) para gerar UMA questão por execução num llama-server.

Serve no PC e no celular (Termux), sem instalar nada além do Python. Faz o que o app faz
(Doc/GUIA_SUFIXO_APP.md): sorteia o slot (subtema, tipo de raciocínio, contexto) MENOS usado no histórico,
monta o prompt de 1 questão com o sufixo de diversidade, manda com a gramática e guarda o slot no histórico.
Isso dá diversidade ENTRE execuções. Não substitui a verificação aritmética do repositório de código
(`schema_utils.check_consistency`): aqui só se confere a estrutura.

  python cliente_llama_server.py --ano 5º --habilidade H08 --dificuldade Moderado
Arquivos esperados ao lado do script: questao.gbnf e taxonomia_subtemas.json. O servidor deve estar no ar
(porta 8080 por padrão); veja o model card para o comando do llama-server.
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

AQUI = Path(__file__).resolve().parent

# Cópia de SYSTEM_PROMPT (src/extract_data.py). Um teste do repositório garante que são idênticos:
# mudar uma vírgula aqui muda o prompt de prefixo e quebra o cache e a distribuição de treino.
SYSTEM_PROMPT = (
    "Você é um gerador de questões de matemática no padrão SAEB para a educação "
    "básica brasileira. Gere EXATAMENTE a quantidade de questões de múltipla "
    "escolha pedida, cada uma com 5 alternativas (A a E), conforme o ano "
    "escolar, a habilidade e a dificuldade pedidos. Responda APENAS com JSON "
    "válido, sem texto extra, no schema: "
    '{"questoes": [{"enunciado": str, "alternativas": {"A": str, "B": str, '
    '"C": str, "D": str, "E": str}, "resolucao_passo_a_passo": str, '
    '"resposta_correta": "A|B|C|D|E", "difficulty": "EASY|MEDIUM|HARD"}, ...]}. '
    "Resolva passo a passo antes de decidir a resposta. O campo "
    "\"resposta_correta\" deve ser a LETRA da alternativa correta. O campo "
    "\"difficulty\" reflete a dificuldade pedida (Fácil=EASY, Moderado=MEDIUM, "
    "Difícil=HARD). Use os sinais - + x ÷ nas contas. A questão deve ser "
    "autocontida, sem depender de figuras, imagens ou gráficos."
)
INSTRUCAO_DADOS = (" Dados: escreva no enunciado todos os dados da tabela ou do gráfico em texto, "
                   "um item por linha no formato 'rótulo: valor'.")
HISTORICO_PADRAO = Path.home() / ".cache" / "questoes_historico.json"


def _rotulo(lista, id_):
    return next((x["rotulo"] for x in lista if x["id"] == id_), id_)


def menos_usado(ids, uso, rng):
    """Itens de menor uso no histórico; empate sorteado."""
    minimo = min(uso(i) for i in ids)
    return rng.choice([i for i in ids if uso(i) == minimo])


def sortear_slot(tax, ano, hab, hist, rng=random):
    h = tax["habilidades"].get(f"{ano}|{hab}")
    ctx_pool = (h or {}).get("contextos") or [c["id"] for c in tax["contextos"]]
    contexto = menos_usado(ctx_pool, lambda i: sum(s.get("contexto") == i for s in hist), rng)
    if not h:
        return {"subtema": None, "tipo": None, "contexto": contexto}
    subs = {s["id"]: s for s in h["subtemas"]}
    sub = menos_usado(list(subs), lambda i: sum(s.get("subtema") == i for s in hist), rng)
    tipos = subs[sub].get("tipos_raciocinio") or h["tipos_raciocinio"]
    return {"subtema": sub, "tipo": rng.choice(tipos), "contexto": contexto}


def montar_prompt(tax, ano, hab, descricao, dificuldade, slot):
    p = f"Gere 1 questão(ões) de matemática. Ano: {ano} ano. Habilidade: {hab} — {descricao}. Dificuldade: {dificuldade}."
    ctx = _rotulo(tax["contextos"], slot["contexto"])
    h = tax["habilidades"].get(f"{ano}|{hab}")
    if h is None or slot["subtema"] is None:  # habilidade fora da taxonomia: só o contexto
        return p + f" Contexto: {ctx}."
    sub = next(s for s in h["subtemas"] if s["id"] == slot["subtema"])
    tipo = next(t for t in tax["tipos_raciocinio"] if t["id"] == slot["tipo"])
    p += f" Subtema: {sub['rotulo']}. Tipo de raciocínio: {tipo['rotulo']}. Contexto: {ctx}."
    if tipo.get("instrucao") and re.search(tipo.get("aplica_a") or "", slot["subtema"]):
        p += tipo["instrucao"]
    if slot["subtema"].startswith(("tabela_", "grafico_", "histograma", "listas")):
        p += INSTRUCAO_DADOS
    return p


def estrutura_ok(obj):
    try:
        q = obj["questoes"]
        return (len(q) == 1 and set(q[0]["alternativas"]) == set("ABCDE")
                and q[0]["resposta_correta"] in "ABCDE" and bool(q[0]["enunciado"].strip()))
    except (KeyError, TypeError, AttributeError):
        return False


def pedir(url, prompt, gramatica, seed):
    corpo = {"messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
             "temperature": 0.7, "top_p": 0.8, "max_tokens": 512, "seed": seed, "cache_prompt": True,
             "grammar": gramatica}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(corpo).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(json.loads(r.read().decode())["choices"][0]["message"]["content"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ano", required=True)
    ap.add_argument("--habilidade", required=True)
    ap.add_argument("--dificuldade", default="Moderado", choices=["Fácil", "Moderado", "Difícil"])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--historico", default=str(HISTORICO_PADRAO))
    ap.add_argument("--tentativas", type=int, default=3, help="mesmo prompt, seed nova, se a estrutura falhar")
    a = ap.parse_args()
    tax = json.loads((AQUI / "taxonomia_subtemas.json").read_text(encoding="utf-8"))
    gramatica = (AQUI / "questao.gbnf").read_text(encoding="utf-8")
    h = tax["habilidades"].get(f"{a.ano}|{a.habilidade}")
    descricao = h["descricoes"][0] if h else ""
    try:
        hist_todo = json.loads(Path(a.historico).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        hist_todo = []
    hist = [s for s in hist_todo if s.get("ano") == a.ano and s.get("habilidade") == a.habilidade]
    slot = sortear_slot(tax, a.ano, a.habilidade, hist)
    prompt = montar_prompt(tax, a.ano, a.habilidade, descricao, a.dificuldade, slot)
    base = random.randrange(1, 10**6)
    for t in range(a.tentativas):
        inicio = time.perf_counter()
        obj = pedir(a.url, prompt, gramatica, base * 100 + t)
        if estrutura_ok(obj):
            print(json.dumps(obj["questoes"][0], ensure_ascii=False, indent=2))
            print(f"\n[{time.perf_counter() - inicio:.1f} s | slot={slot}]", file=sys.stderr)
            Path(a.historico).parent.mkdir(parents=True, exist_ok=True)
            Path(a.historico).write_text(json.dumps(
                hist_todo[-500:] + [{"ano": a.ano, "habilidade": a.habilidade, **slot}], ensure_ascii=False),
                encoding="utf-8")
            return
    sys.exit("estrutura inválida em todas as tentativas")


if __name__ == "__main__":
    main()
