"""Extrai as questões de DB/questoes.db para JSONL em formato chat (SFT).

Filtros aplicados:
  - disciplina == 'Matemática'
  - sem imagem no enunciado nem nas alternativas (modelo final é texto puro)
  - enunciado, 4 alternativas e gabarito válidos (A-D)

Cada exemplo vira {"messages": [system, user, assistant]}, onde o assistant
responde um JSON compacto com a questão completa. Split 90/10 estratificado
por (ano, dificuldade).

Uso:
    python src/extract_data.py
"""

import argparse
import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from schema_utils import DIFFICULTY_MAP, normalize_math

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "DB" / "questoes.db"
OUT_DIR = ROOT / "data"
VAL_FRACTION = 0.10
SEED = 42

# Contrato fixo exigido pelo app mobile (schema definido pelos envolvidos, sem
# exceção): wrapper {"questoes": [...]}, 5 alternativas (A-E), resposta_correta
# (LETRA) e resolucao_passo_a_passo (texto único, sem justificativa por
# alternativa) e difficulty em inglês (EASY/MEDIUM/HARD). Não existe mais um
# campo dedicado ao VALOR da resposta — ver nota em schema_utils.check_consistency
# sobre o impacto disso na cobertura de verificação.
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

USER_TEMPLATE = (
    "Gere {quantidade} questão(ões) de matemática. Ano: {ano} ano. "
    "Habilidade: {habilidade} — {descricao}. Dificuldade: {dificuldade}."
)

# Alternativa E sintética para dados reais do banco (que só têm A-D). Nunca é
# a correta — o gabarito do banco permanece em A-D — só existe para cumprir o
# contrato de 5 alternativas exigido pelo app.
ALTERNATIVA_E_PADRAO = "Nenhuma das alternativas anteriores"


def clean(value):
    """Normaliza campo textual; retorna '' para nulos/'nan'.

    Converte operadores tipográficos Unicode (− × –) para ASCII (- x -).
    Motivo: 30% dos itens do banco usam a forma tipográfica e 70% a ASCII; um
    modelo de 1.7B treinado com 670 exemplos aprende essa inconsistência e
    passa a emitir os dois formatos de modo imprevisível na inferência — o que
    quebrava a verificação por texto (ver seção 5.4 da DOCUMENTACAO_CIENTIFICA).
    Padronizar a origem alinha os dados reais com os sintéticos, que já são ASCII.
    """
    if value is None:
        return ""
    text = normalize_math(str(value)).strip()
    return "" if text.lower() == "nan" else text


def load_rows():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM itens").fetchall()
    con.close()
    return rows


def build_example(row):
    """Converte uma linha do banco em exemplo de chat, ou None se inválida."""
    if clean(row["disciplina"]) != "Matemática":
        return None, "disciplina"

    image_cols = (
        "imagem",
        "alternativa_a_imagem",
        "alternativa_b_imagem",
        "alternativa_c_imagem",
        "alternativa_d_imagem",
    )
    if any(row[col] is not None for col in image_cols):
        return None, "imagem"

    enunciado = clean(row["enunciado_item"])
    comando = clean(row["texto_auxiliar"])
    gabarito = clean(row["gabarito"]).upper()
    alternativas = {
        letra: clean(row[f"alternativa_{letra.lower()}"]) for letra in "ABCD"
    }

    if not enunciado or gabarito not in "ABCD" or not all(alternativas.values()):
        return None, "campos"

    # Itens do banco em que duas alternativas têm o mesmo texto (a resposta
    # correta aparece duplicada) são malformados: treinar neles ensina que
    # repetir alternativa é aceitável, e tornam o gabarito ambíguo.
    if len({v.strip() for v in alternativas.values()}) < 4:
        return None, "alternativas_duplicadas"

    justificativas = {
        letra: clean(row[f"justificativa_alternativa_{letra.lower()}"])
        for letra in "ABCD"
    }
    geral = clean(row["justificativa_geral"])
    # Alternativa correta muitas vezes só tem a justificativa geral.
    if not justificativas[gabarito] and geral:
        justificativas[gabarito] = geral

    # 5ª alternativa (E) sintética — o banco só tem A-D, mas o contrato do app
    # exige 5. Nunca é a correta.
    alternativas["E"] = ALTERNATIVA_E_PADRAO

    dificuldade = clean(row["grau_resolucao"]) or "Moderado"
    # enunciado + comando fundidos: o novo contrato não tem campo separado
    # para a instrução complementar do SAEB.
    enunciado_completo = f"{enunciado} {comando}".strip() if comando else enunciado

    # A ordem de emissão treinada é enunciado -> alternativas ->
    # resolucao_passo_a_passo -> resposta_correta -> difficulty: o modelo
    # mostra o raciocínio ANTES de se comprometer com a letra (ver nota em
    # schema_utils.py). O contrato exige só as chaves abaixo, mesmo conjunto,
    # independente da ordem de serialização.
    questao = {
        "enunciado": enunciado_completo,
        "alternativas": alternativas,
        "resolucao_passo_a_passo": justificativas.get(gabarito, ""),
        "resposta_correta": gabarito,
        "difficulty": DIFFICULTY_MAP.get(dificuldade, "MEDIUM"),
    }
    answer = {"questoes": [questao]}

    ano = clean(row["ano"]) or "não informado"
    example = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    quantidade=1,
                    ano=ano,
                    habilidade=clean(row["habilidade"]) or "geral",
                    descricao=clean(row["descricao_item"]) or "não especificada",
                    dificuldade=dificuldade,
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(answer, ensure_ascii=False),
            },
        ],
        "meta": {
            "codigo_item": clean(row["codigo_item"]),
            "ano": ano,
            "habilidade": clean(row["habilidade"]),
            "dificuldade": clean(row["grau_resolucao"]),
            # Vazio para as linhas anteriores a 2026-09 (coluna criada por
            # importar_entregas.py). Permite separar lotes sem parsear strings.
            "lote": clean(row["lote"]) if "lote" in row.keys() else "",
        },
    }
    return example, None


def split_com_val_congelado(examples, codigos_val):
    """Mantém o conjunto de validação IDÊNTICO ao de um ciclo anterior.

    Comparar baseline e modelo novo exige que os dois sejam medidos no MESMO
    conjunto. stratified_split() re-sorteia a cada execução: acrescentar
    exemplos muda a composição dos buckets (ano, dificuldade) e, com ela, quem
    cai no val — o que tornaria a comparação entre os dois modelos inválida.
    Aqui o val é o do ciclo anterior, item a item, e tudo o mais vai para o
    treino. Nenhum exemplo novo pode entrar no conjunto de avaliação.
    """
    val = [ex for ex in examples if ex["meta"]["codigo_item"] in codigos_val]
    train = [ex for ex in examples if ex["meta"]["codigo_item"] not in codigos_val]
    return train, val


def separa_holdout(examples, por_habilidade, seed=SEED, apenas_lote=None):
    """Reserva N exemplos por (ano, habilidade) como conjunto de cobertura.

    Serve para medir o modelo nos anos que o lote novo trouxe (1º/3º/4º) sem
    contaminar o treino. NÃO substitui o val congelado: o baseline nunca viu
    esses anos, então comparar os dois modelos aqui não teria sentido — este
    conjunto é reportado à parte, como métrica informativa.
    """
    rng = random.Random(seed)
    grupos = defaultdict(list)
    for ex in examples:
        if apenas_lote and ex["meta"].get("lote") != apenas_lote:
            continue
        grupos[(ex["meta"]["ano"], ex["meta"]["habilidade"])].append(ex)

    reservados = set()
    holdout = []
    for chave in sorted(grupos):
        bucket = sorted(grupos[chave], key=lambda e: e["meta"]["codigo_item"])
        rng.shuffle(bucket)
        for ex in bucket[:por_habilidade]:
            holdout.append(ex)
            reservados.add(ex["meta"]["codigo_item"])
    restantes = [ex for ex in examples if ex["meta"]["codigo_item"] not in reservados]
    return restantes, holdout


def stratified_split(examples):
    """Split train/val estratificado por (ano, dificuldade)."""
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for ex in examples:
        groups[(ex["meta"]["ano"], ex["meta"]["dificuldade"])].append(ex)

    train, val = [], []
    for key in sorted(groups):
        bucket = groups[key]
        rng.shuffle(bucket)
        n_val = max(1, round(len(bucket) * VAL_FRACTION)) if len(bucket) > 3 else 0
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])
    rng.shuffle(train)
    return train, val


def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-congelado", default=None,
        help="jsonl cujo conjunto de codigo_item define o val (preserva a "
             "comparabilidade com o baseline); sem ele, faz o split 90/10 de sempre",
    )
    parser.add_argument(
        "--holdout-por-habilidade", type=int, default=0,
        help="reserva N exemplos por (ano, habilidade) num conjunto de cobertura",
    )
    parser.add_argument("--holdout-lote", default=None,
                        help="limita o holdout aos exemplos deste lote")
    parser.add_argument("--holdout-saida", default=str(OUT_DIR / "val_novos.jsonl"))
    args = parser.parse_args()

    rows = load_rows()
    examples, skipped = [], Counter()
    for row in rows:
        example, reason = build_example(row)
        if example:
            examples.append(example)
        else:
            skipped[reason] += 1

    holdout = []
    if args.val_congelado:
        codigos_val = {
            json.loads(linha)["meta"]["codigo_item"]
            for linha in open(args.val_congelado, encoding="utf-8")
        }
        pool, val = split_com_val_congelado(examples, codigos_val)
        faltando = len(codigos_val) - len(val)
        if faltando:
            raise SystemExit(
                f"{faltando} item(ns) do val congelado não foram encontrados no banco — "
                "o conjunto de avaliação deixaria de ser comparável ao baseline."
            )
        if args.holdout_por_habilidade:
            pool, holdout = separa_holdout(
                pool, args.holdout_por_habilidade, apenas_lote=args.holdout_lote)
        train = pool
        rng = random.Random(SEED)
        rng.shuffle(train)
    else:
        train, val = stratified_split(examples)

    OUT_DIR.mkdir(exist_ok=True)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)
    if holdout:
        write_jsonl(Path(args.holdout_saida), holdout)

    # Rede de segurança: nenhum codigo_item pode estar em dois conjuntos.
    conjuntos = {"train": train, "val": val, "holdout": holdout}
    for a in conjuntos:
        for b in conjuntos:
            if a >= b:
                continue
            comum = ({e["meta"]["codigo_item"] for e in conjuntos[a]}
                     & {e["meta"]["codigo_item"] for e in conjuntos[b]})
            if comum:
                raise SystemExit(f"CONTAMINAÇÃO {a}/{b}: {sorted(comum)[:5]}")

    print(f"Linhas no banco:       {len(rows)}")
    print(f"Exemplos válidos:      {len(examples)}")
    print(f"Descartadas:           {dict(skipped)}")
    print(f"Treino:                {len(train)} -> {OUT_DIR / 'train.jsonl'}")
    print(f"Validação:             {len(val)} -> {OUT_DIR / 'val.jsonl'}"
          + (" (CONGELADO)" if args.val_congelado else ""))
    if holdout:
        print(f"Holdout de cobertura:  {len(holdout)} -> {args.holdout_saida}")

    for campo in ("ano", "dificuldade"):
        dist = Counter(ex["meta"][campo] for ex in examples)
        print(f"Distribuição por {campo}: {dict(sorted(dist.items()))}")

    chars = [len(ex["messages"][2]["content"]) for ex in examples]
    print(
        "Tamanho da resposta (chars): "
        f"média={sum(chars) / len(chars):.0f}, máx={max(chars)} "
        f"(~{max(chars) // 3} tokens; max_seq_length=1024 cobre com folga)"
    )


if __name__ == "__main__":
    main()
