"""Gera questões sintéticas de operações matemáticas básicas para aumentar o
dataset de treino com gabarito 100% garantido por computação em Python — não
por um LLM. As ~300 questões reais de DB/questoes.db ensinam formato/estilo,
mas são poucas para o modelo generalizar cálculo; aqui cada resposta correta é
calculada antes de montar a questão, então serve de reforço "limpo" de
aritmética, complementando (não substituindo) o dataset real.

As tuplas (ano, habilidade, descricao_item, dificuldade) usadas são as mesmas
que existem de verdade em DB/questoes.db, para o prompt de treino continuar
idêntico ao que o app manda em produção (USER_TEMPLATE).

Escopo: adição, subtração, multiplicação, divisão (exata), porcentagem e
potenciação. Frações (H07-H09 do 9º ano) ficam de fora por enquanto — exigem
lógica de equivalência mais cuidadosa para gerar distratores corretos.

Uso:
    python src/generate_synthetic.py                     # gera e mescla em data/train.jsonl
    python src/generate_synthetic.py --num-per-skill 20   # mais exemplos por habilidade/dificuldade
    python src/generate_synthetic.py --dry-run            # só mostra estatísticas, não escreve nada

Reexecutar é seguro: exemplos sintéticos anteriores (meta.sintetico=true) são
removidos do train.jsonl antes de gerar e escrever o novo lote.
"""

import argparse
import json
import random
from pathlib import Path

from extract_data import SYSTEM_PROMPT, USER_TEMPLATE
from schema_utils import check_consistency, check_structure

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "train.jsonl"
DEFAULT_NUM_PER_SKILL = 12

NOMES = ["Ana", "Bruno", "Carla", "Diego", "Elisa", "Felipe",
         "Gabriela", "Hugo", "Isabela", "João"]
CONTEXTOS = ["bolinhas de gude", "figurinhas", "reais", "livros", "doces",
             "laranjas", "lápis", "cadernos", "bonecos", "adesivos"]


def fmt(n):
    """Formata número no padrão brasileiro (vírgula decimal)."""
    if isinstance(n, int):
        return str(n)
    return f"{n:.1f}".replace(".", ",")


def build_alternativas(rng, correto, wrong_op_result=None, is_integer=True):
    """Monta 4 alternativas distintas (1 certa + 3 distratores plausíveis) e
    retorna (dict letra->valor, letra do gabarito)."""
    candidatos = []

    def add(v):
        v = int(v) if is_integer else round(v, 1)
        if v != correto and v not in candidatos and (is_integer is False or v >= 0):
            candidatos.append(v)

    if wrong_op_result is not None:
        add(wrong_op_result)

    deltas = ([-10, -5, -3, -2, -1, 1, 2, 3, 5, 10] if is_integer
              else [-5, -2, -1, -0.5, -0.2, 0.2, 0.5, 1, 2, 5])
    rng.shuffle(deltas)
    for d in deltas:
        if len(candidatos) >= 3:
            break
        add(correto + d)

    step, i = (3 if is_integer else 0.3), 1
    while len(candidatos) < 3:
        add(correto + i * step)
        if len(candidatos) < 3:
            add(correto - i * step)
        i += 1

    valores = candidatos[:3] + [correto]
    rng.shuffle(valores)
    letras = "ABCD"
    posicoes = {letras[i]: valores[i] for i in range(4)}
    gabarito = next(l for l, v in posicoes.items() if v == correto)
    return posicoes, gabarito


def montar(enunciado, comando, posicoes, gabarito, justificativa):
    return {
        "enunciado": enunciado,
        "comando": comando,
        "alternativas": {l: fmt(v) for l, v in posicoes.items()},
        "justificativas": {gabarito: justificativa},
        "gabarito": gabarito,
    }


# ---- Geradores de questão (cada um recebe rng + faixa e devolve o dict acima) ----

def gen_calc_soma_subtracao(rng, faixa):
    a, b = rng.randint(*faixa), rng.randint(*faixa)
    op = rng.choice(["+", "-"])
    if op == "-" and b > a:
        a, b = b, a
    correto = a + b if op == "+" else a - b
    errado_op = a - b if op == "+" else a + b
    posicoes, gabarito = build_alternativas(rng, correto, errado_op)
    return montar(
        f"Calcule o resultado da operação: {a} {op} {b}.",
        "Qual é o resultado?",
        posicoes, gabarito, f"{a} {op} {b} = {correto}",
    )


def gen_problema_soma_subtracao(rng, faixa):
    a, b = rng.randint(*faixa), rng.randint(*faixa)
    nome, contexto = rng.choice(NOMES), rng.choice(CONTEXTOS)
    if rng.random() < 0.5:
        correto, errado_op = a + b, abs(a - b)
        enunciado = f"{nome} tinha {a} {contexto}. Ganhou mais {b} {contexto}."
        op_str = f"{a} + {b}"
    else:
        if b > a:
            a, b = b, a
        correto, errado_op = a - b, a + b
        enunciado = f"{nome} tinha {a} {contexto}. Perdeu {b} {contexto}."
        op_str = f"{a} - {b}"
    comando = f"Quantos {contexto} {nome} tem agora?"
    posicoes, gabarito = build_alternativas(rng, correto, errado_op)
    return montar(enunciado, comando, posicoes, gabarito, f"{op_str} = {correto}")


def gen_calc_mult_div(rng, faixas):
    faixa_fator1, faixa_fator2 = faixas
    if rng.random() < 0.5:
        a, b = rng.randint(*faixa_fator1), rng.randint(*faixa_fator2)
        correto, errado_op = a * b, a + b
        op_str = f"{a} x {b}"
        enunciado = f"Calcule o resultado da operação: {a} x {b}."
    else:
        quociente = rng.randint(*faixa_fator1)
        b = max(1, rng.randint(*faixa_fator2))
        a = b * quociente
        correto, errado_op = quociente, a - b
        op_str = f"{a} ÷ {b}"
        enunciado = f"Calcule o resultado da operação: {a} ÷ {b}."
    posicoes, gabarito = build_alternativas(rng, correto, errado_op)
    return montar(enunciado, "Qual é o resultado?", posicoes, gabarito, f"{op_str} = {correto}")


def gen_problema_mult_div(rng, faixas):
    faixa_grupos, faixa_itens = faixas
    nome, contexto = rng.choice(NOMES), rng.choice(CONTEXTOS)
    grupos, itens = rng.randint(*faixa_grupos), rng.randint(*faixa_itens)
    if rng.random() < 0.5:
        correto, errado_op = grupos * itens, grupos + itens
        enunciado = f"{nome} organizou {contexto} em {grupos} grupos com {itens} {contexto} cada um."
        comando = f"Quantos {contexto} há ao todo?"
        op_str = f"{grupos} x {itens}"
    else:
        total = grupos * itens
        correto, errado_op = itens, max(total - grupos, 0)
        enunciado = f"{nome} tem {total} {contexto} e quer separar em {grupos} grupos iguais."
        comando = f"Quantos {contexto} ficam em cada grupo?"
        op_str = f"{total} ÷ {grupos}"
    posicoes, gabarito = build_alternativas(rng, correto, errado_op)
    return montar(enunciado, comando, posicoes, gabarito, f"{op_str} = {correto}")


def gen_porcentagem(rng, faixas):
    base_mult_faixa, percentuais = faixas
    base = rng.randint(*base_mult_faixa) * 20  # múltiplo de 20 -> % múltiplo de 5 é exato
    percentual = rng.choice(percentuais)
    correto = base * percentual // 100
    errado_op = base - correto  # confunde "valor do desconto" com "valor restante"
    enunciado = f"Uma loja vende um produto de R$ {base} com {percentual}% de desconto."
    comando = "Qual é o valor, em reais, do desconto?"
    posicoes, gabarito = build_alternativas(rng, correto, errado_op)
    return montar(
        enunciado, comando, posicoes, gabarito,
        f"{base} x {percentual} ÷ 100 = {correto}",
    )


def gen_potenciacao(rng, faixas):
    faixa_base, faixa_exp = faixas
    base, exp = rng.randint(*faixa_base), rng.randint(*faixa_exp)
    correto, errado_op = base ** exp, base * exp
    posicoes, gabarito = build_alternativas(rng, correto, errado_op)
    return montar(
        f"Calcule o resultado da potência {base}^{exp} ({base} elevado ao expoente {exp}).",
        "Qual é o resultado?",
        posicoes, gabarito, f"{base}^{exp} = {correto}",
    )


def gen_calc_real_decimal(rng, faixa):
    a, b = round(rng.uniform(*faixa), 1), round(rng.uniform(*faixa), 1)
    op = rng.choice(["+", "-", "x"])
    if op == "-" and b > a:
        a, b = b, a
    if op == "+":
        correto, errado_op = round(a + b, 1), round(a - b, 1)
    elif op == "-":
        correto, errado_op = round(a - b, 1), round(a + b, 1)
    else:
        correto, errado_op = round(a * b, 1), round(a + b, 1)
    posicoes, gabarito = build_alternativas(rng, correto, errado_op, is_integer=False)
    return montar(
        f"Calcule o resultado da operação: {fmt(a)} {op} {fmt(b)}.",
        "Qual é o resultado?",
        posicoes, gabarito, f"{fmt(a)} {op} {fmt(b)} = {fmt(correto)}",
    )


# ---- Habilidades reais do banco (mesmos descritores usados em produção) ----

SKILLS = [
    {
        "ano": "2º", "habilidade": "H06",
        "descricao": "Calcular o resultado de adições ou subtrações, envolvendo números naturais de até 3 ordens.",
        "gerador": gen_calc_soma_subtracao,
        "faixas": {"Fácil": (1, 20), "Moderado": (1, 99), "Difícil": (100, 999)},
    },
    {
        "ano": "2º", "habilidade": "H08",
        "descricao": (
            "Resolver problemas de adição ou de subtração, envolvendo números naturais "
            "de até 3 ordens, com os significados de juntar, acrescentar, separar ou retirar."
        ),
        "gerador": gen_problema_soma_subtracao,
        "faixas": {"Fácil": (2, 20), "Moderado": (10, 99), "Difícil": (50, 300)},
    },
    {
        "ano": "5º", "habilidade": "H03",
        "descricao": "Calcular o resultado de adições ou subtrações envolvendo números naturais de até 6 ordens.",
        "gerador": gen_calc_soma_subtracao,
        "faixas": {"Fácil": (100, 999), "Moderado": (1000, 9999), "Difícil": (10000, 99999)},
    },
    {
        "ano": "5º", "habilidade": "H04",
        "descricao": "Calcular o resultado de multiplicações ou divisões envolvendo números naturais de até 6 ordens.",
        "gerador": gen_calc_mult_div,
        "faixas": {
            "Fácil": ((2, 9), (2, 9)),
            "Moderado": ((2, 20), (10, 50)),
            "Difícil": ((10, 50), (10, 99)),
        },
    },
    {
        "ano": "5º", "habilidade": "H06",
        "descricao": (
            "Resolver problemas de multiplicação ou de divisão, envolvendo números naturais de "
            "até 6 ordens, com os significados de formação de grupos iguais (incluindo repartição "
            "equitativa e medida), proporcionalidade ou disposição retangular."
        ),
        "gerador": gen_problema_mult_div,
        "faixas": {
            "Fácil": ((2, 6), (2, 9)),
            "Moderado": ((3, 10), (5, 20)),
            "Difícil": ((5, 15), (10, 40)),
        },
    },
    {
        "ano": "9º", "habilidade": "H03",
        "descricao": "Porcentagem, acréscimos, decréscimos e taxas sucessivas.",
        "gerador": gen_porcentagem,
        "faixas": {
            "Fácil": ((1, 5), (10, 25, 50)),
            "Moderado": ((5, 15), (15, 30, 40, 60)),
            "Difícil": ((15, 40), (35, 45, 65, 75, 85)),
        },
    },
    {
        "ano": "9º", "habilidade": "H02",
        "descricao": (
            "Resolver problemas de adição, subtração, multiplicação, divisão, potenciação ou "
            "radiciação envolvendo números reais, inclusive notação científica."
        ),
        "gerador": gen_potenciacao,
        "faixas": {"Fácil": ((2, 5), (2, 2)), "Moderado": ((2, 9), (2, 3)), "Difícil": ((2, 12), (2, 4))},
    },
    {
        "ano": "9º", "habilidade": "H06",
        "descricao": "Cálculo com números reais: adição, subtração, multiplicação e divisão.",
        "gerador": gen_calc_real_decimal,
        "faixas": {"Fácil": (1, 20), "Moderado": (1, 99), "Difícil": (1, 500)},
    },
]


def build_example(ano, habilidade, descricao, dificuldade, answer, idx):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    ano=ano, habilidade=habilidade, descricao=descricao, dificuldade=dificuldade
                ),
            },
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ],
        "meta": {
            "codigo_item": f"SINT-{habilidade}-{dificuldade}-{idx:04d}",
            "ano": ano,
            "habilidade": habilidade,
            "dificuldade": dificuldade,
            "sintetico": True,
        },
    }


def gerar_todos(num_per_skill, seed):
    rng = random.Random(seed)
    exemplos, idx = [], 0
    for skill in SKILLS:
        for dificuldade, faixa in skill["faixas"].items():
            for _ in range(num_per_skill):
                idx += 1
                answer = skill["gerador"](rng, faixa)
                exemplos.append(
                    build_example(skill["ano"], skill["habilidade"], skill["descricao"], dificuldade, answer, idx)
                )
    return exemplos


def validar(exemplos):
    """Confere schema + consistência gabarito/justificativa (sanity check dos geradores)."""
    problemas, verificaveis, ok = 0, 0, 0
    for ex in exemplos:
        obj = json.loads(ex["messages"][2]["content"])
        flags = check_structure(obj)
        if not (flags["schema_completo"] and flags["gabarito_valido"] and flags["alternativas_distintas"]):
            problemas += 1
        consistente, _ = check_consistency(obj)
        if consistente is not None:
            verificaveis += 1
            ok += int(consistente)
    return problemas, verificaveis, ok


def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-per-skill", type=int, default=DEFAULT_NUM_PER_SKILL,
                         help="exemplos gerados por (habilidade, dificuldade)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="só mostra estatísticas, não escreve")
    args = parser.parse_args()

    sinteticos = gerar_todos(args.num_per_skill, args.seed)
    problemas, verificaveis, ok = validar(sinteticos)

    print(f"Gerados: {len(sinteticos)} exemplos sintéticos ({len(SKILLS)} habilidades x "
          f"3 dificuldades x {args.num_per_skill})")
    print(f"Falhas de schema/gabarito/alternativas: {problemas} (esperado: 0)")
    print(f"Consistência gabarito<->justificativa: {ok}/{verificaveis} verificáveis "
          f"(o restante usa operações — potência, porcentagem — fora do regex simples do checker, "
          f"mas o gabarito continua garantido por construção)")

    if args.dry_run:
        print("\n--dry-run: nada escrito. Exemplo:")
        print(json.dumps(sinteticos[0], ensure_ascii=False, indent=2))
        return

    if problemas:
        raise SystemExit(f"Abortando: {problemas} exemplo(s) sintético(s) com schema inválido.")

    real = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")] if TRAIN_PATH.exists() else []
    real_sem_sintetico = [ex for ex in real if not ex.get("meta", {}).get("sintetico")]
    combinado = real_sem_sintetico + sinteticos
    write_jsonl(TRAIN_PATH, combinado)

    print(f"\nTreino real:      {len(real_sem_sintetico)}")
    print(f"Treino sintético: {len(sinteticos)}")
    print(f"Total escrito em {TRAIN_PATH}: {len(combinado)}")
    print(
        "\ndata/val.jsonl não foi tocado — a validação continua só com questões reais do "
        "SAEB, para medir o modelo no que ele de fato vai enfrentar em produção."
    )


if __name__ == "__main__":
    main()
