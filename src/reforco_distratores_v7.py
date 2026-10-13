"""Reforço sintético de distratores para as 3 habilidades onde o ciclo v7 regrediu (ver
ciclo_v7/SPEC.md e outputs/relatorios/VEREDITO_v7_multiseed_n150.json): 5º H03 (soma/subtração até
6 ordens), 5º H01 (escrever número por extenso) e 9º H02 (operações com números reais/grandes).

Diagnóstico medido na amostra n=150: a curadoria v7 derrubou a taxa de "Nenhuma das alternativas
anteriores" no candidato de 58,0% para 23,3% — o modelo perdeu o atalho fácil e agora precisa
gerar 4 distratores plausíveis de verdade. Nessas 3 habilidades (aritmética de números grandes e
escrita por extenso) ele erra a própria conta: "432+187=619" com alternativas {117,439,620,519,518}
(619 não está em nenhuma), ou gera alternativas duplicadas no H01. A quantidade de exemplos reais
dessas habilidades não caiu na curadoria (v6: 32/39/32 -> v7: 31/38/32) — o problema é falta de
exemplos de treino que ENSINEM o padrão certo de distrator nessas habilidades, não falta de volume
geral.

Reaproveita os geradores determinísticos de src/generate_synthetic.py (gabarito calculado em
Python, nunca por um LLM): gen_calc_soma_subtracao (5º H03 e, com faixa maior, 9º H02 — a
descrição oficial de H02 cobre soma/sub/mult/div de números reais, não só potenciação, que é o
único gerador que já existia para ela) e um gerador NOVO, gen_escrita_por_extenso (5º H01), com
distratores de erro pedagógico: trocar "mil" por "milhão"/"cem", esquecer o "e", trocar um dígito.

NÃO escreve em data/train.jsonl, train_curado.jsonl nem train_curado_v6/v7.jsonl (protegidos).
Gera data/train_curado_v7b.jsonl = v7 + reforço (arquivo novo, v7 original intocado).

    venv/bin/python src/reforco_distratores_v7.py --dry-run
    venv/bin/python src/reforco_distratores_v7.py --num-per-skill 15
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate_synthetic import (  # noqa: E402
    _agrupar,
    _montar_questao,
    fmt,
    montar,
    montar_texto,
)
import diversidade as dv  # noqa: E402
from generate_synthetic import CONTEXTOS as CONTEXTOS_ITENS, NOMES  # noqa: E402
from perfil_base import carregar as carregar_base, hash_questao  # noqa: E402
from schema_utils import check_consistency, check_structure, extract_questoes  # noqa: E402

V7 = ROOT / "data" / "train_curado_v7.jsonl"
V7B = ROOT / "data" / "train_curado_v7b.jsonl"
PROTEGIDOS = ("train_curado.jsonl", "train.jsonl", "train_curado_v3.jsonl", "train_curado_v5.jsonl",
             "train_curado_v6.jsonl", "train_curado_v7.jsonl")

DEFAULT_NUM_PER_SKILL = 15

MAGNITUDE = ("", "mil", "milhão", "bilhão")


def _num2words(n):
    from num2words import num2words
    return num2words(int(n), lang="pt_BR")


def _trocar_magnitude(texto):
    """Erro de ordem de grandeza: troca 'mil' <-> 'milhão'/'cem' no texto por extenso."""
    if " milhão" in texto or " milhões" in texto:
        return texto.replace(" milhões", " mil").replace(" milhão", " mil")
    if " mil" in texto:
        return texto.replace(" mil", " milhão", 1)
    return "cem " + texto


def _sem_e(texto):
    """Erro de omitir o 'e' que liga as dezenas/unidades (comum em erro de ditado)."""
    partes = texto.split(" e ", 1)
    return " ".join(partes) if len(partes) == 2 else texto + " e zero"


def _digito_trocado(n, rng):
    """Erro de dígito: muda um dígito do número (exceto o primeiro) antes de escrever por extenso."""
    s = list(str(n))
    if len(s) < 2:
        return n + rng.choice([1, 2, 3])
    i = rng.randrange(1, len(s))
    novo = str(rng.choice([d for d in range(10) if str(d) != s[i]]))
    s[i] = novo
    return int("".join(s))


CONTEXTOS_EXTENSO = [
    ("Em uma campanha de arrecadação, a escola reuniu {n} itens doados.", "itens doados"),
    ("Uma fábrica produziu {n} peças no último mês.", "peças produzidas"),
    ("O estádio registrou um público de {n} pessoas na partida.", "pessoas no público"),
    ("A prefeitura distribuiu {n} cestas básicas no bairro.", "cestas distribuídas"),
    ("Uma rede de livrarias vendeu {n} livros na campanha de leitura.", "livros vendidos"),
]


def gen_escrita_por_extenso(rng, faixa, dificuldade):
    """5º H01: escrever um número natural (até 6 ordens) por extenso. Gabarito e os 4
    distratores (ordem de grandeza trocada, 'e' omitido, dígito trocado, combinação) são
    computados em Python via num2words — nunca pelo LLM. Contexto sorteado entre 5 cenários
    (além da pergunta seca) para não colapsar no detector de near-dup, que mascara números."""
    n = rng.randint(*faixa)
    correto = _num2words(n)
    candidatos = [
        (_trocar_magnitude(correto), "Erro de ordem de grandeza: troca a magnitude (mil/milhão/cem)."),
        (_sem_e(correto), "Erro de ditado: omite o 'e' que liga as ordens do número."),
        (_sem_e(_trocar_magnitude(correto)), "Erro combinado: magnitude trocada e 'e' omitido."),
    ]
    for _ in range(6):
        nd = _digito_trocado(n, rng)
        candidatos.append((_num2words(nd), f"Erro de dígito: escreve o número {nd} em vez de {n}."))
    itens = [(correto, f"O número {n} escrito por extenso é '{correto}'.")]
    vistos = {correto}
    for texto, justificativa in candidatos:
        if texto not in vistos:
            itens.append((texto, justificativa))
            vistos.add(texto)
        if len(itens) == 5:
            break
    contexto, rotulo = rng.choice(CONTEXTOS_EXTENSO)
    nome = rng.choice(NOMES)
    enunciado = (f"{nome} registrou o seguinte dado numa pesquisa sobre {rng.choice(CONTEXTOS_ITENS)}: "
                + contexto.format(n=n) + f" Como esse número de {rotulo} é escrito por extenso?")
    return montar_texto(rng, enunciado, "", itens, dificuldade)


CONTEXTOS_CALCULO = [
    "Calcule o resultado da operação: {a} {op} {b}.",
    "Uma planilha registra dois valores, {a} e {b}.",
    "Determine o valor de {a} {op} {b}.",
    "Em um relatório financeiro, aparecem os números {a} e {b}.",
    "Uma balança registrou as medidas {a} e {b}, em quilogramas.",
]


def gen_calc_variado(rng, faixa, dificuldade):
    """5º H03 / 9º H02: soma ou subtração de números naturais, com enunciado sorteado entre 5
    frames para não colapsar no detector de near-dup (que mascara números antes de comparar).
    Gabarito e distratores pedagógicos (operação trocada, vai-um, erro de 1 unidade) computados
    em Python, igual a generate_synthetic.gen_calc_soma_subtracao."""
    a, b = rng.randint(*faixa), rng.randint(*faixa)
    op = rng.choice(["+", "-"])
    if op == "-" and b > a:
        a, b = b, a
    if op == "+":
        correto, trocada = a + b, a - b
        nome_op, nome_trocada = "adição", "subtração"
    else:
        correto, trocada = a - b, a + b
        nome_op, nome_trocada = "subtração", "adição"
    vai_um = correto + rng.choice([-10, 10])
    itens = [
        (correto, f"{a} {op} {b} = {correto}."),
        (trocada, f"Erro de operação: faz a {nome_trocada} em vez da {nome_op} e obtém {fmt(abs(trocada))}."),
        (vai_um, f"Erro de reagrupamento (vai-um): obtém {fmt(vai_um)} em vez de {correto}."),
        (correto + rng.choice([-1, 1]), "Erro de contagem de uma unidade."),
    ]
    base_enun = rng.choice(CONTEXTOS_CALCULO).format(a=a, op=op, b=b)
    nome, contexto = rng.choice(NOMES), rng.choice(CONTEXTOS_ITENS)
    base_enun = f"{nome} está organizando uma planilha sobre {contexto}. " + base_enun
    return montar(rng, base_enun, f"Qual é o resultado de {a} {op} {b}?", itens, dificuldade)


# (ano, habilidade, descricao oficial, gerador, {dificuldade: faixa})
REFORCOS = [
    {
        "ano": "5º", "habilidade": "H03",
        "descricao": "Calcular o resultado de adições ou subtrações envolvendo números naturais de até 6 ordens.",
        "gerador": gen_calc_variado,
        "faixas": {"Fácil": (100, 999), "Moderado": (1000, 9999), "Difícil": (10000, 99999)},
    },
    {
        "ano": "5º", "habilidade": "H01",
        "descricao": ("Escrever números racionais (naturais de até 6 ordens, representação fracionária ou decimal "
                     "finita até a ordem dos milésimos) em sua representação por algarismos ou em língua materna "
                     "OU Associar o registro numérico ao registro em língua materna"),
        "gerador": gen_escrita_por_extenso,
        "faixas": {"Fácil": (10, 999), "Moderado": (1000, 99999), "Difícil": (100000, 999999)},
    },
    {
        "ano": "9º", "habilidade": "H02",
        "descricao": ("Resolver problemas de adição, subtração, multiplicação, divisão, potenciação ou radiciação "
                     "envolvendo números reais, inclusive notação científica."),
        "gerador": gen_calc_variado,
        # faixa de 6-7 dígitos: reproduz o padrão observado na falha real ("1.200.000 + 250.000 + 150.000")
        "faixas": {"Fácil": (1000, 9999), "Moderado": (10000, 999999), "Difícil": (100000, 9999999)},
    },
]


def build_example(ano, habilidade, descricao, dificuldade, questoes, idx):
    from extract_data import SYSTEM_PROMPT, USER_TEMPLATE
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                quantidade=len(questoes), ano=ano, habilidade=habilidade, descricao=descricao, dificuldade=dificuldade)},
            {"role": "assistant", "content": json.dumps({"questoes": questoes}, ensure_ascii=False)},
        ],
        "meta": {"codigo_item": f"REFORCO-V7-{habilidade}-{ano}-{dificuldade}-{idx:04d}", "ano": ano,
                "habilidade": habilidade, "dificuldade": dificuldade, "sintetico": True,
                "origem_v7": "reforco_distratores"},
    }


def _questoes_existentes(base, ano, habilidade):
    if base is None:
        return []
    return [q for _i, _j, m, q, _p in carregar_base(base) if m["ano"] == ano and m["habilidade"] == habilidade]


def gerar_todos(num_per_skill, seed, base=None, tentativas_por_item=8):
    """base: caminho do train_curado_v7.jsonl (ou None) — contra quem checar near-dup, além
    do que já foi aceito nesta mesma geração. Descarta (sem substituir por outro slot: a
    tentativa seguinte do mesmo slot já conta como nova amostra) e tenta de novo até
    `tentativas_por_item` vezes; se não conseguir, mantém o último gerado (log via contagem)."""
    rng = random.Random(seed)
    exemplos, idx, descartados = [], 0, 0
    for skill in REFORCOS:
        aceitas = list(_questoes_existentes(base, skill["ano"], skill["habilidade"]))
        for dificuldade, faixa in skill["faixas"].items():
            pool = []
            for _ in range(num_per_skill):
                for tentativa in range(tentativas_por_item):
                    q = skill["gerador"](rng, faixa, dificuldade)
                    if not any(dv.e_near_duplicata(q, outra) for outra in aceitas):
                        break
                    descartados += 1
                pool.append(q)
                aceitas.append(q)
            for grupo in _agrupar(rng, pool):
                idx += 1
                exemplos.append(build_example(skill["ano"], skill["habilidade"], skill["descricao"], dificuldade, grupo, idx))
    if descartados:
        print(f"(near-dup descartados e regenerados durante a criação: {descartados})")
    return exemplos


def validar(exemplos):
    problemas, verificaveis, ok = 0, 0, 0
    for ex in exemplos:
        obj = json.loads(ex["messages"][2]["content"])
        flags = check_structure(obj)
        if not (flags["schema_completo"] and flags["resposta_valida"] and flags["alternativas_distintas"]
                and flags["difficulty_valida"]):
            problemas += 1
        for q in extract_questoes(obj):
            consistente, _s = check_consistency(q)
            if consistente is not None:
                verificaveis += 1
                ok += int(consistente)
    return problemas, verificaveis, ok


def val_enunciados():
    vs = set()
    for nome in ("val.jsonl", "val_frozen_v1.jsonl", "val_novos_v1.jsonl"):
        p = ROOT / "data" / nome
        if p.exists():
            for l in open(p, encoding="utf-8"):
                for q in json.loads(json.loads(l)["messages"][2]["content"])["questoes"]:
                    vs.add(q["enunciado"].strip().lower())
    return vs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-per-skill", type=int, default=DEFAULT_NUM_PER_SKILL)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--base", default=str(V7))
    ap.add_argument("--saida", default=str(V7B))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    saida = Path(args.saida)
    if saida.name in PROTEGIDOS:
        raise SystemExit(f"ABORTADO: não escrevo em {saida.name}.")

    exemplos = gerar_todos(args.num_per_skill, args.seed, base=args.base)
    problemas, verificaveis, ok = validar(exemplos)
    total_q = sum(len(json.loads(e["messages"][2]["content"])["questoes"]) for e in exemplos)
    print(f"Gerados: {len(exemplos)} exemplos / {total_q} questões ({len(REFORCOS)} habilidades x 3 dificuldades "
          f"x {args.num_per_skill})")
    print(f"Falhas de schema: {problemas} (esperado: 0)")
    print(f"Consistência verificável: {ok}/{verificaveis}")
    contaminadas = sum(1 for e in exemplos for q in json.loads(e["messages"][2]["content"])["questoes"]
                       if q["enunciado"].strip().lower() in val_enunciados())
    print(f"Contaminação com val*: {contaminadas} (esperado: 0)")

    if args.dry_run:
        print("\n--dry-run: nada escrito. Exemplo de cada habilidade:")
        vistos = set()
        for e in exemplos:
            if e["meta"]["habilidade"] not in vistos:
                vistos.add(e["meta"]["habilidade"])
                print(json.dumps(e, ensure_ascii=False, indent=1))
        return

    if problemas or contaminadas:
        raise SystemExit(f"ABORTADO: {problemas} falha(s) de schema, {contaminadas} contaminação(ões) com val*.")

    base = [json.loads(l) for l in open(args.base, encoding="utf-8")]
    combinado = base + exemplos
    saida.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in combinado), encoding="utf-8")
    print(f"\nBase ({args.base}): {len(base)} exemplos")
    print(f"Reforço sintético:  {len(exemplos)} exemplos")
    print(f"Total escrito em {saida}: {len(combinado)}")


if __name__ == "__main__":
    main()
