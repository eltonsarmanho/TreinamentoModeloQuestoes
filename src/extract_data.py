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

import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "DB" / "questoes.db"
OUT_DIR = ROOT / "data"
VAL_FRACTION = 0.10
SEED = 42

SYSTEM_PROMPT = (
    "Você é um gerador de questões de matemática no padrão SAEB para a educação "
    "básica brasileira. Gere exatamente UMA questão de múltipla escolha conforme o "
    "ano escolar, a habilidade e a dificuldade pedidos. Responda APENAS com JSON "
    "válido, sem texto extra, no schema: "
    '{"enunciado": str, "comando": str, "alternativas": {"A": str, "B": str, '
    '"C": str, "D": str}, "justificativas": {"A": str, "B": str, "C": str, '
    '"D": str}, "gabarito": "A|B|C|D"}. Resolva e justifique cada alternativa '
    "antes de decidir o gabarito, para que a letra escolhida seja consistente "
    "com as contas feitas nas justificativas. A questão deve ser autocontida, "
    "sem depender de figuras, imagens ou gráficos."
)

USER_TEMPLATE = (
    "Gere uma questão de matemática. Ano: {ano} ano. "
    "Habilidade: {habilidade} — {descricao}. Dificuldade: {dificuldade}."
)


def clean(value):
    """Normaliza campo textual; retorna '' para nulos/'nan'."""
    if value is None:
        return ""
    text = str(value).strip()
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

    justificativas = {
        letra: clean(row[f"justificativa_alternativa_{letra.lower()}"])
        for letra in "ABCD"
    }
    geral = clean(row["justificativa_geral"])
    # Alternativa correta muitas vezes só tem a justificativa geral.
    if not justificativas[gabarito] and geral:
        justificativas[gabarito] = geral
    justificativas = {k: v for k, v in justificativas.items() if v}

    # Ordem das chaves importa: JSON é gerado token a token, então a
    # justificativa (raciocínio) precisa vir ANTES do gabarito no schema —
    # senão o modelo comete a letra sem ter "mostrado o trabalho" ainda.
    answer = {
        "enunciado": enunciado,
        "comando": comando,
        "alternativas": alternativas,
    }
    if justificativas:
        answer["justificativas"] = justificativas
    answer["gabarito"] = gabarito

    ano = clean(row["ano"]) or "não informado"
    example = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    ano=ano,
                    habilidade=clean(row["habilidade"]) or "geral",
                    descricao=clean(row["descricao_item"]) or "não especificada",
                    dificuldade=clean(row["grau_resolucao"]) or "Moderado",
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
        },
    }
    return example, None


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
    rows = load_rows()
    examples, skipped = [], Counter()
    for row in rows:
        example, reason = build_example(row)
        if example:
            examples.append(example)
        else:
            skipped[reason] += 1

    train, val = stratified_split(examples)
    OUT_DIR.mkdir(exist_ok=True)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)

    print(f"Linhas no banco:       {len(rows)}")
    print(f"Exemplos válidos:      {len(examples)}")
    print(f"Descartadas:           {dict(skipped)}")
    print(f"Treino:                {len(train)} -> {OUT_DIR / 'train.jsonl'}")
    print(f"Validação:             {len(val)} -> {OUT_DIR / 'val.jsonl'}")

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
