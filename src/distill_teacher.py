"""Destilação de dados: gera questões novas com um modelo PROFESSOR grande
(via Hugging Face Inference Providers) e aceita só as que passam nos filtros
determinísticos — o caminho da literatura para ampliar a capacidade de um
modelo pequeno além do que os ~300 exemplos reais ensinam.

Diferente de generate_synthetic.py (aritmética pura com resposta_correta
calculada em Python), aqui o professor gera questões CONTEXTUALIZADAS no
estilo SAEB para qualquer habilidade do banco — inclusive as que não são só
cálculo. O preço é que o professor também erra; por isso NADA entra no treino
sem passar pelo filtro: schema completo, 5 alternativas distintas, sem menção
a figura, consistência resposta_correta<->conta não reprovada e sem duplicata.

Requer HF_TOKEN no .env com acesso a Inference Providers (billing do HF).
O professor padrão pode ser trocado com --teacher (qualquer modelo de chat
disponível nos providers).

Uso:
    python src/distill_teacher.py --dry-run                # mostra o plano, não chama API
    python src/distill_teacher.py --limit-tuples 5         # teste barato (5 tuplas x 3 questões)
    python src/distill_teacher.py                          # gera para todas as tuplas do banco
    python src/distill_teacher.py --merge                  # mescla data/distill.jsonl no train.jsonl

O arquivo data/distill.jsonl é APPEND-ONLY (seguro interromper e retomar);
duplicatas são descartadas na geração e na mesclagem. A mesclagem remove o
lote destilado anterior do train.jsonl antes de inserir o novo (idempotente,
igual ao generate_synthetic.py). data/val.jsonl nunca é tocado.
"""

import argparse
import json
import random
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from extract_data import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402
from schema_utils import (  # noqa: E402
    IMAGE_PATTERN,
    check_consistency,
    check_structure,
    extract_questao,
    parse_json,
)

DB_PATH = ROOT / "DB" / "questoes.db"
TRAIN_PATH = ROOT / "data" / "train.jsonl"
DISTILL_PATH = ROOT / "data" / "distill.jsonl"

# Qualquer modelo de chat forte disponível nos HF Inference Providers serve.
DEFAULT_TEACHER = "Qwen/Qwen3-235B-A22B-Instruct-2507"
DEFAULT_PER_TUPLE = 3
MAX_ANSWER_CHARS = 900  # mantém a resposta dentro do max_seq_length=1024 do treino

TEACHER_ADDENDUM = (
    " Regras adicionais para esta tarefa: a questão deve ser INÉDITA e "
    "contextualizada (situação do cotidiano brasileiro); os números devem ser "
    "escolhidos para que a conta seja exata; resolucao_passo_a_passo deve "
    "mostrar a conta completa e o resultado deve bater EXATAMENTE com o valor "
    "da alternativa de resposta_correta; cada distrator deve vir de um erro "
    "específico que um aluno cometeria; escreva o JSON em uma única linha."
)

TEMAS = [
    "feira livre", "campeonato de futebol", "receita de bolo", "viagem de ônibus",
    "biblioteca da escola", "horta comunitária", "festa junina", "mesada e cofrinho",
    "loja de brinquedos", "campanha de reciclagem", "clube de leitura", "padaria",
    "passeio ao parque", "coleção de figurinhas", "mercado do bairro", "gincana escolar",
]


def normalizar(texto):
    """Chave de deduplicação: minúsculas, sem acentos, só alfanumérico."""
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = texto.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", texto.lower())


def load_tuples(limit=None):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT DISTINCT ano, habilidade, descricao_item, grau_resolucao FROM itens "
        "WHERE disciplina='Matemática' AND habilidade IS NOT NULL AND habilidade != '' "
        "AND descricao_item IS NOT NULL AND grau_resolucao IS NOT NULL "
        "ORDER BY ano, habilidade, grau_resolucao"
    ).fetchall()
    con.close()
    rows = [r for r in rows if r[0] and str(r[0]).lower() != "nan"]
    return rows[:limit] if limit else rows


def filtrar(obj, raw_text, vistos, strict=False):
    """Aplica os filtros determinísticos. Retorna None se aprovado, ou o motivo.

    `obj` é o wrapper `{"questoes": [...]}` bruto devolvido pelo professor;
    só a primeira questão é considerada (o pedido é sempre quantidade=1)."""
    if obj is None:
        return "json_invalido"
    flags = check_structure(obj, quantidade_esperada=1)
    if not (flags["wrapper_valido"] and flags["schema_completo"] and flags["resposta_valida"]
            and flags["alternativas_distintas"] and flags["difficulty_valida"]):
        return "estrutura"
    questao = extract_questao(obj, 0)
    blob = json.dumps(questao, ensure_ascii=False)
    if IMAGE_PATTERN.search(blob):
        return "menciona_figura"
    if len(blob) > MAX_ANSWER_CHARS:
        return "muito_longa"
    consistente, _ = check_consistency(questao)
    if consistente is False:
        return "resposta_inconsistente"
    if strict and consistente is not True:
        return "consistencia_nao_verificavel"
    chave = normalizar(questao.get("enunciado", ""))
    if not chave or chave in vistos:
        return "duplicata"
    vistos.add(chave)
    return None


def build_example(ano, habilidade, descricao, dificuldade, questao, idx, teacher):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    quantidade=1, ano=ano, habilidade=habilidade,
                    descricao=descricao, dificuldade=dificuldade,
                ),
            },
            {"role": "assistant", "content": json.dumps({"questoes": [questao]}, ensure_ascii=False)},
        ],
        "meta": {
            "codigo_item": f"DIST-{habilidade}-{dificuldade}-{idx:05d}",
            "ano": ano,
            "habilidade": habilidade,
            "dificuldade": dificuldade,
            "destilado": True,
            "professor": teacher,
        },
    }


def gerar(args):
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=args.teacher, provider=args.provider)
    tuples = load_tuples(args.limit_tuples)
    rng = random.Random(args.seed)

    vistos = set()
    aceitos_previos = 0
    if DISTILL_PATH.exists():
        for line in open(DISTILL_PATH, encoding="utf-8"):
            ex = json.loads(line)
            obj = json.loads(ex["messages"][2]["content"])
            questao = extract_questao(obj, 0)
            vistos.add(normalizar((questao or {}).get("enunciado", "")))
            aceitos_previos += 1
        print(f"Retomando: {aceitos_previos} questões já aceitas em {DISTILL_PATH} "
              "(duplicatas serão descartadas).")

    rejeicoes = Counter()
    aceitos, erros_api, idx = 0, 0, aceitos_previos
    total_chamadas = len(tuples) * args.per_tuple
    print(f"Professor: {args.teacher} (provider: {args.provider})")
    print(f"{len(tuples)} tuplas (ano, habilidade, descrição, dificuldade) x "
          f"{args.per_tuple} = {total_chamadas} chamadas planejadas\n")

    with open(DISTILL_PATH, "a", encoding="utf-8") as saida:
        for t, (ano, habilidade, descricao, dificuldade) in enumerate(tuples, 1):
            for _ in range(args.per_tuple):
                tema = rng.choice(TEMAS)
                user = USER_TEMPLATE.format(
                    quantidade=1, ano=ano, habilidade=habilidade, descricao=descricao,
                    dificuldade=dificuldade,
                ) + f" Tema sugerido para o contexto: {tema}."
                try:
                    resp = client.chat_completion(
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT + TEACHER_ADDENDUM},
                            {"role": "user", "content": user},
                        ],
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    text = resp.choices[0].message.content or ""
                except Exception as exc:  # rede/quota/provider — segue para a próxima
                    erros_api += 1
                    print(f"  [erro API {erros_api}/{args.max_errors}] {exc}")
                    if erros_api >= args.max_errors:
                        raise SystemExit(
                            "Muitos erros de API — abortando. O que já foi aceito "
                            f"está salvo em {DISTILL_PATH}; rode de novo para retomar."
                        )
                    time.sleep(args.retry_wait)
                    continue

                obj = parse_json(text)
                motivo = filtrar(obj, text, vistos, strict=args.strict)
                if motivo:
                    rejeicoes[motivo] += 1
                    continue

                idx += 1
                questao = extract_questao(obj, 0)
                ex = build_example(ano, habilidade, descricao, dificuldade, questao, idx, args.teacher)
                saida.write(json.dumps(ex, ensure_ascii=False) + "\n")
                saida.flush()
                aceitos += 1
            print(f"[{t}/{len(tuples)}] {ano} {habilidade} ({dificuldade}): "
                  f"{aceitos} aceitas até agora")

    print(f"\nAceitas nesta rodada: {aceitos} (total no arquivo: {aceitos_previos + aceitos})")
    print(f"Rejeitadas por motivo: {dict(rejeicoes) or 'nenhuma'}")
    print(f"Erros de API: {erros_api}")
    print(f"\nPróximo passo: revisar amostras de {DISTILL_PATH} e mesclar com:")
    print("    python src/distill_teacher.py --merge")


def merge():
    if not DISTILL_PATH.exists():
        raise SystemExit(f"{DISTILL_PATH} não existe — rode a geração primeiro.")
    destilados = [json.loads(l) for l in open(DISTILL_PATH, encoding="utf-8")]

    # Dedup interno (retomadas podem ter re-aceitado enunciados parecidos).
    vistos, unicos = set(), []
    for ex in destilados:
        obj = json.loads(ex["messages"][2]["content"])
        questao = extract_questao(obj, 0)
        chave = normalizar((questao or {}).get("enunciado", ""))
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(ex)

    base = [json.loads(l) for l in open(TRAIN_PATH, encoding="utf-8")] if TRAIN_PATH.exists() else []
    base_sem_destilado = [ex for ex in base if not ex.get("meta", {}).get("destilado")]
    combinado = base_sem_destilado + unicos
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for ex in combinado:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    n_real = sum(1 for ex in base_sem_destilado if not ex.get("meta", {}).get("sintetico"))
    n_sint = len(base_sem_destilado) - n_real
    print(f"Treino real:      {n_real}")
    print(f"Treino sintético: {n_sint}")
    print(f"Treino destilado: {len(unicos)} ({len(destilados) - len(unicos)} duplicatas removidas)")
    print(f"Total escrito em {TRAIN_PATH}: {len(combinado)}")
    print("\ndata/val.jsonl não foi tocado — validação continua 100% com questões reais.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--teacher", default=DEFAULT_TEACHER,
                        help=f"modelo professor (padrão: {DEFAULT_TEACHER})")
    parser.add_argument("--provider", default="auto",
                        help="HF Inference Provider (padrão: auto)")
    parser.add_argument("--per-tuple", type=int, default=DEFAULT_PER_TUPLE,
                        help="questões pedidas por (ano, habilidade, descrição, dificuldade)")
    parser.add_argument("--limit-tuples", type=int, default=None,
                        help="limita o nº de tuplas (para teste barato)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="temperatura do professor (alta = mais diversidade)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--strict", action="store_true",
                        help="exige consistência VERIFICADA (descarta as não verificáveis)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    parser.add_argument("--merge", action="store_true",
                        help="mescla data/distill.jsonl no data/train.jsonl e sai")
    parser.add_argument("--dry-run", action="store_true",
                        help="mostra o plano de chamadas sem chamar a API")
    args = parser.parse_args()

    if args.merge:
        merge()
        return

    if args.dry_run:
        tuples = load_tuples(args.limit_tuples)
        print(f"Professor: {args.teacher} (provider: {args.provider})")
        print(f"Tuplas no banco: {len(tuples)} | questões por tupla: {args.per_tuple} "
              f"| chamadas planejadas: {len(tuples) * args.per_tuple}")
        print("Filtros: schema completo + 5 alternativas distintas + "
              "sem figura + consistência não reprovada + sem duplicata"
              + (" + consistência VERIFICADA (--strict)" if args.strict else ""))
        ano, habilidade, descricao, dificuldade = tuples[0]
        print("\nExemplo de prompt que será enviado ao professor:")
        print(f"  [system] {SYSTEM_PROMPT[:120]}... (+ addendum de destilação)")
        prompt_exemplo = USER_TEMPLATE.format(
            quantidade=1, ano=ano, habilidade=habilidade, descricao=descricao, dificuldade=dificuldade,
        )
        print(f"  [user]   {prompt_exemplo} Tema sugerido para o contexto: feira livre.")
        return

    gerar(args)


if __name__ == "__main__":
    main()
