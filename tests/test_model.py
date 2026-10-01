"""Testa o modelo REAL — o mesmo binário e arquivo .gguf que rodam offline no
app mobile — usando llama.cpp via subprocess. Não depende de torch/unsloth.

Diferente de evaluate.py (que mede o modelo em 4-bit via bitsandbytes/HF na
GPU, um caminho só de desenvolvimento), aqui a geração passa pelo artefato de
produção de fato: o .gguf quantizado Q4_K_M rodando em CPU pelo llama-cli,
igual ao que acontece no celular.

Modos:
  Interativo (padrão): escolha ano/habilidade/dificuldade num menu (as opções
  vêm do próprio banco DB/questoes.db) e veja a questão gerada, formatada.

      python tests/test_model.py

  Direto, uma pergunta específica:

      python tests/test_model.py --ano "5º" --habilidade H08 --descricao "Resolver problemas de adição ou subtração." --dificuldade Fácil

  Gerar N variações do mesmo pedido (ver estabilidade/variabilidade):

      python tests/test_model.py --ano "9º" --habilidade H17 --n 5

  Lote de N questões (padrão para N>1: modo PLANEJADO, uma questão por
  chamada com cobertura de subtemas, todas validadas — ver gerar_lote.py;
  --sem-planejamento volta ao pedido único antigo):

      python tests/test_model.py --ano "9º" --habilidade H17 --quantidade 5

  Lote real contra o conjunto de validação (sanity check fim a fim do
  artefato exportado, mesmas métricas estruturais do evaluate.py):

      python tests/test_model.py --batch
      python tests/test_model.py --batch --num-samples 10   # mais rápido
"""

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Este arquivo mora em tests/ (não é um teste pytest: é a CLI de inferência de
# produção, chamada por gerar_lote.py e avaliar_diversidade.py), então os
# módulos irmãos (extract_data, schema_utils) estão em src/, não ao lado dele.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from extract_data import SYSTEM_PROMPT, USER_TEMPLATE
from schema_utils import (
    ALTERNATIVE_LETTERS,
    DIFFICULTY_MAP,
    IMAGE_PATTERN,
    MOTIVO_FORA_DAS_ALTERNATIVAS,
    depende_de_visual_ausente,
    check_consistency,
    check_consistency_detalhado,
    check_structure,
    extract_questao,
    fix_gabarito,
    parse_json,
)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "DB" / "questoes.db"
VAL_PATH = ROOT / "data" / "val.jsonl"
REPORT_PATH = ROOT / "outputs" / "eval_report_gguf.json"
GRAMMAR_PATH = ROOT / "grammars" / "questao.gbnf"

DEFAULT_LLAMA_CLI = Path.home() / ".unsloth" / "llama.cpp" / "llama-cli"
# Q4_K_M: melhor tokens/s medido no benchmark (ver README) com poucas threads —
# a geração é limitada por banda de memória, não por núcleos de CPU.
DEFAULT_THREADS = 4
# MEDIDO (2026-09, tokenizer Qwen/Qwen3-1.7B) sobre as 1.188 respostas ACEITAS
# dos relatórios outputs/diversidade_*.json, reconstruídas como
# json.dumps({"questoes": [q]}, ensure_ascii=False) — a unidade real, já que
# gerar_lote_planejado chama generate_validated com quantidade=1:
#   mediana 176 | p90 261 | p95 302 | p99 389 | p99.9 482 | max 482
# Acumulada: <=384 98,8% | <=416 99,41% | <=448 99,75% | <=512 100,0%.
#
# NÃO reduzir para o p99 (389). Três motivos medidos:
#  1. A cauda observada encosta no teto: max = 482 = 94% de 512. A amostra é
#     CENSURADA — só entram respostas que couberam e parsearam; uma resposta
#     que já tivesse estourado 512 apareceria como JSON inválido, não como um
#     comprimento grande. O p99 real da distribuição não-censurada é >= 389.
#  2. Cortar em 389 truncaria pelo menos 1,2% das gerações BOAS no meio do JSON
#     — e truncar não degrada, invalida: vira parse_json() -> None -> "falha".
#     Trocaríamos 1,2% de questões entregues por ~0% de ganho.
#  3. Não há ganho de tempo a colher: max_new_tokens é TETO, não alocação. O
#     llama-cli para no EOS, então a mediana de 176 tokens custa 176 tokens de
#     geração com teto 512 ou com teto 389 — idêntico. O teto só cobra preço
#     quando é ATINGIDO, que é justamente o caso que queremos evitar.
# Ver `pendencias` do relatório da Fase 1: a economia de latência pertence ao
# cache de prefixo (item 1.4, lado React Native), não a este teto.
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.8

SPEED_PATTERN = re.compile(
    r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s"
)


def find_llama_cli(explicit=None):
    if explicit:
        return Path(explicit)
    if DEFAULT_LLAMA_CLI.exists():
        return DEFAULT_LLAMA_CLI
    found = shutil.which("llama-cli")
    if found:
        return Path(found)
    raise SystemExit(
        "llama-cli não encontrado. Rode antes: python src/export_gguf.py "
        "(ele baixa e compila o llama.cpp), ou informe o caminho com --llama-cli."
    )


def find_gguf(explicit=None):
    if explicit:
        return Path(explicit)
    candidates = sorted((ROOT / "outputs").glob("**/*.gguf"))
    if not candidates:
        raise SystemExit(
            "Nenhum .gguf em outputs/ — rode antes: python src/export_gguf.py"
        )
    return candidates[0]


def generate(llama_cli, gguf_path, user_prompt, threads, max_new_tokens, seed=None,
             grammar=None):
    """Chama llama-cli em modo single-turn e retorna (texto, prompt_tps, gen_tps, latencia_total_s)."""
    cmd = [
        str(llama_cli),
        "-m", str(gguf_path),
        "--jinja",
        "-sys", SYSTEM_PROMPT,
        "-p", user_prompt,
        "-st", "-no-cnv",
        "-rea", "off",
        "--temp", str(TEMPERATURE),
        "--top-p", str(TOP_P),
        "-n", str(max_new_tokens),
        "--no-display-prompt", "--simple-io",
        "-t", str(threads),
        "--log-disable",
    ]
    if grammar is not None:
        cmd += ["--grammar-file", str(grammar)]
    if seed is not None:
        cmd += ["-s", str(seed)]

    start = time.perf_counter()
    # errors="replace": o llama-cli ocasionalmente corta um caractere UTF-8
    # multibyte no limite do buffer de saída (mais provável em lotes longos,
    # com muitas chamadas independentes). Sem isto, decode() explode com
    # UnicodeDecodeError e derruba a avaliação inteira depois de já ter
    # gerado várias dezenas de questões. O byte truncado vira "�" só na
    # resposta daquela chamada; parse_json/check_structure tratam isso como
    # JSON inválido normalmente, sem mascarar nenhum outro erro.
    # Timeout: a chamada conta como geração inválida (texto vazio -> JSON
    # inválido -> best-of-N tenta de novo), não derruba o lote. O tempo gasto
    # entra na latência: é o que o usuário do app esperaria.
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, errors="replace")
        stdout = result.stdout
    except subprocess.TimeoutExpired:
        print(f"  [timeout] llama-cli > 300s (seed={seed}); tratado como geração inválida",
              file=sys.stderr)
        stdout = ""
    elapsed = time.perf_counter() - start

    match = SPEED_PATTERN.search(stdout)
    prompt_tps = float(match.group(1)) if match else None
    gen_tps = float(match.group(2)) if match else None

    # Remove o eco do prompt ("> ...") e a linha de timing antes de parsear.
    answer = stdout
    marker = f"> {user_prompt}"
    if marker in answer:
        answer = answer.split(marker, 1)[1]
    answer = SPEED_PATTERN.sub("", answer).strip()

    return answer, prompt_tps, gen_tps, elapsed


def _normaliza_alternativa(texto):
    """Forma canônica de uma alternativa para detectar duplicata DISFARÇADA.

    Colapsa só o que é invisível para o aluno: caixa, espaços repetidos e o
    ponto final. NÃO toca em dígitos nem em separador decimal — "2,5" e "2.5"
    continuam sendo alternativas distintas para efeito desta checagem, porque
    aqui não cabe decidir se o modelo quis dizer a mesma coisa.
    """
    return re.sub(r"\s+", " ", str(texto or "").strip().rstrip(".").strip()).casefold()


def alternativas_degeneradas(questao):
    """True se duas alternativas são a MESMA opção para quem responde.

    Complementa (não substitui) a flag `alternativas_distintas` de
    check_structure, que compara os textos literalmente. O caso real que
    motivou: D="Nenhuma das alternativas anteriores" e
    E="Nenhuma das alternativas anteriores." — a flag literal aprova por causa
    do ponto final, mas o aluno recebe duas opções idênticas e, pior, o
    gabarito pode apontar para uma delas, tornando a outra igualmente correta.

    Deliberadamente conservador: só a normalização acima. Nenhuma validação
    existente é removida — esta é uma checagem ADICIONAL.
    """
    alts = questao.get("alternativas") if isinstance(questao, dict) else None
    if not isinstance(alts, dict):
        return False
    textos = [_normaliza_alternativa(alts.get(L)) for L in ALTERNATIVE_LETTERS]
    if any(not t for t in textos):
        return True
    return len(set(textos)) != len(ALTERNATIVE_LETTERS)


# Faixas de _score_candidato, nomeadas para que os chamadores (e os testes) não
# dependam de literais espalhados pelo arquivo.
SCORE_ESTRUTURA_QUEBRADA = 0
SCORES_APROVADOS = (6, 8)          # estrutura ok + resolvível (retorno antecipado)
SCORES_COM_VISUAL_AUSENTE = (1, 3, 5, 7)


def _score_candidato(flags, consistente, texto="", fora_das_alternativas=False,
                     questao=None):
    """Ordena candidatos do melhor para o pior (maior é melhor).

    ESCALA (revista na Fase 1 — antes ia de 0 a 6):

    8 = estrutura ok + consistência VERIFICADA + resolvível sem ver nada
    7 = idem, mas aponta para um visual ausente
    6 = estrutura ok, consistência não verificável, resolvível  (aceitável)
    5 = idem, mas aponta para um visual ausente
    4 = estrutura ok, gabarito inconsistente mas CORRIGÍVEL por fix_gabarito
        (a conta da resolução bate com OUTRA alternativa: existe letra a sugerir)
    3 = idem, e ainda aponta para um visual ausente
    2 = estrutura ok, mas a resposta da resolução NÃO ESTÁ EM ALTERNATIVA NENHUMA
        (schema_utils.MOTIVO_FORA_DAS_ALTERNATIVAS) — questão IRRESPONDÍVEL
    1 = idem, e ainda aponta para um visual ausente
    0 = estrutura quebrada, incluindo alternativas degeneradas
                                                     (descartar se houver melhor)

    POR QUE 2 FICA ABAIXO DE 6 ("não verificável"). Uma questão cuja resposta
    certa não está entre as alternativas não tem como ser acertada: o aluno
    marca qualquer letra e erra. Uma questão "não verificável" é só uma questão
    que o verificador regex não conseguiu conferir — 86,6% do corpus está nessa
    faixa, e a esmagadora maioria está correta. Entregar a irrespondível no
    lugar da não verificável é trocar erro certo por dúvida. Ex. real
    (diversidade_exp_C_s2 / P12-5º-H21-N5 q#4): alternativas 60/100/110/120/130,
    gabarito C=110, resolução "150 - 80 = 70".

    POR QUE 4 FICA ACIMA DE 2. Na escala antiga as duas valiam 2, e como a
    comparação em generate_validated é `score > melhor` (estrita), a PRIMEIRA
    amostra vencia: o pipeline entregava a irreparável e jogava fora a que
    fix_gabarito consertaria para 100% consistente. Separar as faixas é o
    desempate, sem custo de chamada nenhuma.

    A penalidade por dependência visual entra como -1 DENTRO de cada faixa, em
    vez de zerar o candidato: o app é texto puro, então "Observe a imagem
    abaixo" entrega ao aluno uma questão que ele não tem como resolver. Se
    existir outro candidato resolvível, ele ganha; se TODOS dependerem de um
    visual, ainda se entrega o melhor — degradar é melhor que não responder.
    Ver depende_de_visual_ausente(): detecta a referência dêitica, não a mera
    palavra "figura" (que aparece legitimamente em "figura plana", geometria).

    Retorno antecipado do best-of-N continua nos scores APROVADOS E RESOLVÍVEIS
    (SCORES_APROVADOS = 6 e 8, o que antes eram 4 e 6): regra inalterada, só
    renumerada.
    """
    estrutura_ok = (flags["json_valido"] and flags["wrapper_valido"] and flags["quantidade_correta"]
                    and flags["schema_completo"] and flags["resposta_valida"]
                    and flags["alternativas_distintas"] and flags["difficulty_valida"])
    if not estrutura_ok:
        return SCORE_ESTRUTURA_QUEBRADA
    if questao is not None and alternativas_degeneradas(questao):
        return SCORE_ESTRUTURA_QUEBRADA
    if consistente is True:
        base = 4
    elif consistente is None:
        base = 3
    elif fora_das_alternativas:
        base = 1
    else:
        base = 2
    return base * 2 - (1 if depende_de_visual_ausente(texto) else 0)


def generate_validated(llama_cli, gguf_path, user_prompt, threads, max_new_tokens,
                       grammar=None, retries=1, quantidade=1, base_seed=None, gen_fn=None):
    """Pipeline de produção: best-of-N com verificador determinístico.

    1. Gera com grammar GBNF (estrutura garantida por construção, se disponível).
    2. Valida estrutura (check_structure, no wrapper {"questoes": [...]}) e
       consistência resposta_correta<->resolucao_passo_a_passo da primeira
       questão (check_consistency).
    3. Amostra até `retries`+1 candidatos, **parando assim que um passa** na
       verificação; entre os que reprovam, guarda o melhor (ver _score_candidato)
       em vez do último — é a diferença entre best-of-N e retry sequencial.
    4. Se nenhum passou, aplica fix_gabarito() sobre a primeira questão do
       melhor candidato: se a conta da resolução bate com outra alternativa,
       corrige a letra deterministicamente em vez de entregar uma questão errada.

    A seleção por verificador (em vez de voto majoritário simples) é o que a
    literatura reporta como mais eficaz para modelos pequenos — ver
    arXiv:2410.12608, que mede +5 a +7 pontos em modelos de 0.5B-1B.

    GUARDA DA FASE 1 (1.1): um candidato cuja resposta certa NÃO ESTÁ entre as
    alternativas é irrespondível e pontua ABAIXO de "não verificável" (ver
    _score_candidato), de modo que o best-of-N prefira re-amostrar a entregá-lo.
    Junto com o orçamento de qualidade de gerar_lote.TENTATIVAS_QUALIDADE, é o
    que faz essas questões serem regeneradas em vez de repassadas ao app.

    Retorna dict com: text, obj (a PRIMEIRA questão do wrapper, para exibição/
    métricas), flags, status, motivo_consistencia, regeneracoes, gen_tps, elapsed.
    status: "ok" | "nao_verificavel" | "corrigido" | "depende_de_visual" | "falha".
    motivo_consistencia (aditivo): o motivo de check_consistency_detalhado do
    candidato escolhido — vale schema_utils.MOTIVO_FORA_DAS_ALTERNATIVAS quando
    a falha foi por resposta fora das alternativas.

    gen_fn: substituto opcional de generate() (mesma assinatura). Existe para
    que gerar_lote.py e os testes rodem sem llama-cli; em produção fica None.
    """
    gen = gen_fn or generate
    total_elapsed, regeneracoes = 0.0, 0
    melhor = None  # (score, text, top_obj, questao, flags, gen_tps, motivo)
    for attempt in range(retries + 1):
        # base_seed fixa a amostragem por item: dois modelos diferentes recebem
        # exatamente as mesmas seeds, tornando a comparação PAREADA e
        # reproduzível. Sem isso, com n=30 e temperature=0.7, a diferença entre
        # baseline e candidato fica dentro do ruído da amostragem.
        if base_seed is not None:
            seed = base_seed * 100 + attempt
        else:
            seed = None if attempt == 0 else 1000 + attempt
        text, _, gen_tps, elapsed = gen(
            llama_cli, gguf_path, user_prompt, threads, max_new_tokens,
            seed=seed, grammar=grammar,
        )
        total_elapsed += elapsed
        top_obj = parse_json(text)
        flags = check_structure(top_obj, quantidade_esperada=quantidade)
        questao = extract_questao(top_obj, 0)
        # check_consistency_detalhado devolve também o MOTIVO, numa só passada.
        # É ele que separa "gabarito trocado" (fix_gabarito conserta) de
        # "resposta fora das alternativas" (irrecuperável: não há letra a
        # sugerir, só regenerar resolve). Ver _score_candidato.
        consistente, _sug, motivo = check_consistency_detalhado(questao)
        fora = motivo == MOTIVO_FORA_DAS_ALTERNATIVAS
        score = _score_candidato(flags, consistente, text,
                                 fora_das_alternativas=fora, questao=questao)

        if melhor is None or score > melhor[0]:
            melhor = (score, text, top_obj, questao, flags, gen_tps, motivo)

        # Só os scores PARES estão livres de dependência visual (a penalidade é
        # -1). Aceitar 7 ("consistente, mas mande o aluno olhar uma imagem que
        # não existe") entregaria uma questão irresolvível sem sequer tentar
        # de novo — a conta estar certa não ajuda quem não vê a figura.
        if score in SCORES_APROVADOS:  # aprovado e resolvível: para de gastar tempo
            return {"text": text, "obj": questao, "flags": flags,
                    "status": "ok" if score == 8 else "nao_verificavel",
                    "motivo_consistencia": motivo,
                    "regeneracoes": regeneracoes, "gen_tps": gen_tps,
                    "elapsed": total_elapsed}
        if attempt < retries:
            regeneracoes += 1

    melhor_score, text, top_obj, questao, flags, gen_tps, motivo = melhor
    if questao is not None and melhor_score == SCORE_ESTRUTURA_QUEBRADA:
        # estrutura quebrada (ex.: alternativas repetidas): trocar a letra do
        # gabarito não conserta — não pode sair rotulada como "corrigido"
        status = "falha"
    elif questao is not None:
        questao, fix_status = fix_gabarito(questao)
        if fix_status == "corrigido":
            status = "corrigido"
        elif melhor_score in SCORES_COM_VISUAL_AUSENTE and fix_status != "fora_das_alternativas":
            # Nenhum candidato ficou livre de dependência visual. A questão é
            # estruturalmente válida, mas manda o aluno olhar algo que o app
            # não tem — quem consome precisa saber para poder descartar.
            status = "depende_de_visual"
        else:
            # "falha" cobre também MOTIVO_FORA_DAS_ALTERNATIVAS. O status NÃO
            # ganha valor novo de propósito: test_model.batch grava
            # pos_processamento e promover_checkpoint.G2 exige `falha == 0`.
            # Criar um rótulo separado tiraria essas questões da contagem de G2
            # e AFROUXARIA o gate. Quem quiser o detalhe lê
            # `motivo_consistencia`, que é aditivo e não entra em gate nenhum.
            status = "falha"
    else:
        status = "falha"
    return {"text": text, "obj": questao, "flags": flags, "status": status,
            "motivo_consistencia": motivo,
            "regeneracoes": regeneracoes, "gen_tps": gen_tps,
            "elapsed": total_elapsed}


def print_question(obj, raw_text):
    if obj is None:
        print("  [FALHA AO PARSEAR JSON] Saída bruta do modelo:")
        print(f"  {raw_text[:600]}")
        return
    print(f"  Enunciado: {obj.get('enunciado', '?')}")
    alts = obj.get("alternativas", {})
    gabarito = obj.get("resposta_correta", "?")
    for letra in ALTERNATIVE_LETTERS:
        marca = " <- resposta_correta" if letra == gabarito else ""
        print(f"    {letra}) {alts.get(letra, '?')}{marca}")
    print(f"  Difficulty: {obj.get('difficulty', '?')}")
    if obj.get("resolucao_passo_a_passo"):
        print(f"  Resolução: {obj['resolucao_passo_a_passo']}")


STATUS_LABEL = {
    "depende_de_visual": ("[ATENÇÃO] estrutura ok, mas o enunciado aponta para uma "
                          "figura/gráfico que o app não tem — descartar"),
    "ok": "[ok] validado: estrutura e consistência aprovadas",
    "nao_verificavel": "[ok] estrutura aprovada (consistência não verificável — sem conta explícita)",
    "corrigido": "[corrigido] resposta_correta trocada deterministicamente para bater com a conta da resolução",
    "falha": "[FALHA] reprovado mesmo após regenerar — descartar esta questão",
}


def run_one(llama_cli, gguf_path, ano, habilidade, descricao, dificuldade, threads, n,
            grammar=None, retries=1, quantidade=1, planejado=True,
            max_tentativas_diversidade=2):
    # quantidade>1: por padrão usa o modo PLANEJADO (gerar_lote.py) — uma
    # questão por chamada, guiada por plano de subtemas, com TODAS as questões
    # validadas. --sem-planejamento volta ao pedido único "Gere N questões".
    if quantidade > 1 and planejado:
        return run_planejado(llama_cli, gguf_path, ano, habilidade, descricao, dificuldade,
                             threads, n, quantidade, grammar=grammar, retries=retries,
                             max_tentativas_diversidade=max_tentativas_diversidade)
    user_prompt = USER_TEMPLATE.format(
        quantidade=quantidade, ano=ano, habilidade=habilidade, descricao=descricao,
        dificuldade=dificuldade,
    )
    print(f"\nPrompt: {user_prompt}")
    print(f"Grammar: {'sim (' + grammar.name + ')' if grammar else 'não'} | "
          f"Regenerações máximas: {retries}\n")

    for i in range(n):
        if n > 1:
            print(f"--- Variação {i + 1}/{n} ---")
        r = generate_validated(
            llama_cli, gguf_path, user_prompt, threads, MAX_NEW_TOKENS,
            grammar=grammar, retries=retries, quantidade=quantidade,
        )
        print_question(r["obj"], r["text"])
        flags = r["flags"]
        figura = " | menciona figura!" if IMAGE_PATTERN.search(r["text"]) else ""
        print(
            f"  [json_valido={flags['json_valido']} "
            f"wrapper_valido={flags['wrapper_valido']} "
            f"quantidade_correta={flags['quantidade_correta']} "
            f"schema_completo={flags['schema_completo']} "
            f"resposta_valida={flags['resposta_valida']} "
            f"alternativas_distintas={flags['alternativas_distintas']} "
            f"difficulty_valida={flags['difficulty_valida']}{figura}]"
        )
        print(f"  {STATUS_LABEL[r['status']]}"
              + (f" | {r['regeneracoes']} regeneração(ões)" if r["regeneracoes"] else ""))
        gen_str = f"{r['gen_tps']:.1f} tok/s" if r["gen_tps"] else "n/d"
        print(f"  Tempo total: {r['elapsed']:.1f}s (inclui carregar o modelo) | Geração: {gen_str}\n")


def run_planejado(llama_cli, gguf_path, ano, habilidade, descricao, dificuldade, threads,
                  n, quantidade, grammar=None, retries=1, max_tentativas_diversidade=2):
    """Imprime lote(s) gerados por gerar_lote.gerar_lote_planejado."""
    from gerar_lote import gerar_lote_planejado  # import tardio: gerar_lote importa este módulo
    for i in range(n):
        if n > 1:
            print(f"--- Lote {i + 1}/{n} ---")
        r = gerar_lote_planejado(
            llama_cli, gguf_path, ano, habilidade, descricao, dificuldade, quantidade,
            threads, grammar=grammar, retries=retries,
            max_tentativas_diversidade=max_tentativas_diversidade, verbose=True,
        )
        for q, d in zip(r["questoes"], r["detalhes"]):
            cls = d["classificacao"] or {}
            print(f"[{d['indice'] + 1}] planejado={d['subtema_planejado']} "
                  f"obtido={cls.get('subtema', '?')} status={d['status']} "
                  f"tentativas={d['tentativas_diversidade']} tempo={d['tempo_s']}s")
            print_question(q, "")
            print()
        restantes = [v for v in r["violacoes"] if v["violacoes"]]
        m = r["metricas"] or {}
        print(f"Lote: {len(r['questoes'])}/{quantidade} questões | "
              f"quantidade_correta={r['flags']['quantidade_correta']} | "
              f"regenerações por diversidade={r['regeneracoes_diversidade']} | "
              f"slots com violação remanescente={len(restantes)}")
        if m:
            print(f"diversity_score={m['diversity_score']:.3f} coverage={m['coverage_score']:.3f} "
                  f"duplicate_rate={m['duplicate_rate']:.3f} "
                  f"subtemas={m['subtema_distribution']}")
        print(f"Tempo total: {r['tempo_s']}s ({r['tempo_medio_por_questao_s']}s/questão)\n")


def load_habilidades():
    """Carrega (ano, habilidade, descricao) distintos direto do banco, para o menu interativo."""
    if not DB_PATH.exists():
        return []
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT DISTINCT ano, habilidade, descricao_item FROM itens "
        "WHERE disciplina='Matemática' AND habilidade IS NOT NULL "
        "AND habilidade != '' ORDER BY ano, habilidade"
    ).fetchall()
    con.close()
    return [r for r in rows if r[0] and r[0].lower() != "nan"]


def interactive(llama_cli, gguf_path, threads, grammar=None, retries=1):
    opcoes = load_habilidades()
    print(f"llama-cli: {llama_cli}")
    print(f"modelo:    {gguf_path}\n")

    if not opcoes:
        print("DB/questoes.db não encontrado — informe os campos manualmente.")
        while True:
            ano = input("Ano escolar (ex: 5º) ou 'sair': ").strip()
            if ano.lower() in ("sair", "exit", "q"):
                return
            habilidade = input("Habilidade (ex: H08): ").strip()
            descricao = input("Descrição da habilidade: ").strip()
            dificuldade = input("Dificuldade (Fácil/Moderado/Difícil): ").strip()
            run_one(llama_cli, gguf_path, ano, habilidade, descricao, dificuldade,
                    threads, 1, grammar=grammar, retries=retries)
        return

    anos = sorted({o[0] for o in opcoes})
    while True:
        print("Anos disponíveis:", ", ".join(anos))
        ano = input("Escolha o ano (ou 'sair'): ").strip()
        if ano.lower() in ("sair", "exit", "q"):
            return
        do_ano = [o for o in opcoes if o[0] == ano]
        if not do_ano:
            print("Ano inválido.\n")
            continue

        print("\nHabilidades disponíveis:")
        for idx, (_, hab, desc) in enumerate(do_ano):
            print(f"  [{idx}] {hab} — {desc[:90]}")
        escolha = input("Escolha o número da habilidade: ").strip()
        if not escolha.isdigit() or not (0 <= int(escolha) < len(do_ano)):
            print("Escolha inválida.\n")
            continue
        _, habilidade, descricao = do_ano[int(escolha)]

        dificuldade = input("Dificuldade (Fácil/Moderado/Difícil) [Moderado]: ").strip() or "Moderado"
        n = input("Quantas variações gerar? [1]: ").strip()
        n = int(n) if n.isdigit() and int(n) > 0 else 1

        run_one(llama_cli, gguf_path, ano, habilidade, descricao, dificuldade,
                threads, n, grammar=grammar, retries=retries)
        print()


def batch(llama_cli, gguf_path, threads, num_samples, grammar=None, retries=1, raw=False,
          val_path=None, report_path=None):
    examples = [json.loads(line) for line in open(val_path or VAL_PATH, encoding="utf-8")]
    if num_samples:
        examples = examples[:num_samples]

    if raw:
        grammar, retries = None, 0
    modo = ("cru (mede o modelo sozinho, sem grammar/verificação)" if raw
            else "produção (grammar + gerar->checar->corrigir/regenerar)")
    print(f"Modo: {modo}\n")

    results, gen_tps_list, latencies, respostas_corretas = [], [], [], []
    image_mentions, visual_ausente = 0, 0
    consistencia_ok, consistencia_verificavel = 0, 0
    status_counter, regeneracoes_total = Counter(), 0
    dif_aderente, dif_avaliavel = 0, 0
    por_ano = defaultdict(lambda: {"n": 0, "estrutura_ok": 0, "consist_ok": 0,
                                   "consist_verif": 0, "dif_ok": 0})
    for i, ex in enumerate(examples, 1):
        user_msg = ex["messages"][1]["content"]
        print(f"[{i}/{len(examples)}] {user_msg[:80]}...")
        r = generate_validated(
            llama_cli, gguf_path, user_msg, threads, MAX_NEW_TOKENS,
            grammar=grammar, retries=retries, base_seed=i,
        )
        obj, flags, text = r["obj"], r["flags"], r["text"]
        status_counter[r["status"]] += 1
        regeneracoes_total += r["regeneracoes"]
        if obj:
            respostas_corretas.append(str(obj.get("resposta_correta")))
        achado_figura = IMAGE_PATTERN.search(text)
        if achado_figura:
            image_mentions += 1
        visual_ausente += int(depende_de_visual_ausente(text))
        if r["gen_tps"]:
            gen_tps_list.append(r["gen_tps"])
        latencies.append(r["elapsed"])

        consistente, sugestao = check_consistency(obj)
        if consistente is not None:
            consistencia_verificavel += 1
            consistencia_ok += int(consistente)

        # G5 — aderência à dificuldade PEDIDA no prompt. O lote incorporado em
        # 2026-09 é 69% "Fácil"; sem esta métrica, um modelo que passasse a
        # ignorar o pedido de "Difícil" continuaria pontuando 100% em todas as
        # flags estruturais, porque `difficulty` seria um enum válido — só que
        # o errado. É o campo do contrato mais exposto ao desvio do lote.
        esperado = DIFFICULTY_MAP.get(ex["meta"].get("dificuldade"))
        aderente = None
        if esperado and obj:
            dif_avaliavel += 1
            aderente = obj.get("difficulty") == esperado
            dif_aderente += int(aderente)

        # G7 — recorte por ano: 5º e 9º não recebem nenhum item do lote novo,
        # então é aí que um esquecimento catastrófico apareceria primeiro.
        ano = ex["meta"].get("ano", "?")
        estrutura_ok = all(flags[k] for k in (
            "json_valido", "wrapper_valido", "quantidade_correta",
            "schema_completo", "resposta_valida", "alternativas_distintas",
            "difficulty_valida"))
        por_ano[ano]["n"] += 1
        por_ano[ano]["estrutura_ok"] += int(estrutura_ok)
        por_ano[ano]["dif_ok"] += int(bool(aderente))
        if consistente is not None:
            por_ano[ano]["consist_verif"] += 1
            por_ano[ano]["consist_ok"] += int(consistente)

        results.append({
            "codigo_item_ref": ex["meta"]["codigo_item"],
            "ano": ano,
            **flags,
            "status": r["status"],
            "regeneracoes": r["regeneracoes"],
            "consistencia_resposta_correta": consistente,
            "sugestao_resposta_correta": sugestao,
            "difficulty_pedida": ex["meta"].get("dificuldade"),
            "difficulty_emitida": obj.get("difficulty") if obj else None,
            "difficulty_aderente": aderente,
            # Guardar o trecho permite distinguir menção REAL a uma imagem
            # inexistente ("conforme a figura abaixo" — questão quebrada) de
            # falso positivo do IMAGE_PATTERN sobre termo matemático
            # ("figura plana", em geometria — questão perfeitamente autocontida).
            # Sem isso, mencoes_figura_pct é um número que não se pode auditar.
            "trecho_figura": (
                text[max(0, achado_figura.start() - 90):achado_figura.end() + 90]
                if achado_figura else None
            ),
        })

    n = len(results)
    pct = lambda key: round(100 * sum(r[key] for r in results) / n, 1)
    # sha256 do binário avaliado: em 2026-09 três avaliações "de modelos
    # diferentes" rodaram sobre o mesmo .gguf sem que nada acusasse. Com o
    # hash no relatório, dois relatórios com o mesmo artefato são detectáveis.
    artefato_sha256 = hashlib.sha256(Path(gguf_path).read_bytes()).hexdigest()
    report = {
        "artefato": str(gguf_path),
        "artefato_sha256": artefato_sha256,
        "conjunto_avaliacao": str(val_path or VAL_PATH),
        "motor": "llama.cpp (llama-cli)",
        "modo": modo,
        "num_amostras": n,
        "estrutura": {
            "json_valido_pct": pct("json_valido"),
            "wrapper_valido_pct": pct("wrapper_valido"),
            "quantidade_correta_pct": pct("quantidade_correta"),
            "schema_completo_pct": pct("schema_completo"),
            "resposta_valida_pct": pct("resposta_valida"),
            "alternativas_distintas_pct": pct("alternativas_distintas"),
            "difficulty_valida_pct": pct("difficulty_valida"),
            "distribuicao_respostas_corretas": dict(Counter(respostas_corretas)),
            "mencoes_figura_pct": round(100 * image_mentions / n, 1),
            "mencoes_figura_nota": (
                "conta a PALAVRA figura/imagem/gráfico/desenho/ilustração em "
                "qualquer contexto — inclui falso positivo legítimo "
                "('figura plana', 'Desenho' como hobby). Para decidir, use "
                "depende_de_visual_ausente_pct."
            ),
            "depende_de_visual_ausente_pct": round(100 * visual_ausente / n, 1),
            "consistencia_resposta_correta_pct": (
                round(100 * consistencia_ok / consistencia_verificavel, 1)
                if consistencia_verificavel else None
            ),
            "consistencia_verificavel_n": consistencia_verificavel,
            "difficulty_aderente_pct": (
                round(100 * dif_aderente / dif_avaliavel, 1) if dif_avaliavel else None
            ),
            "difficulty_avaliavel_n": dif_avaliavel,
            "gabarito_letra_mais_frequente_pct": (
                round(100 * max(Counter(respostas_corretas).values()) / len(respostas_corretas), 1)
                if respostas_corretas else None
            ),
        },
        "por_ano": {
            ano: {
                "n": v["n"],
                "estrutura_ok_pct": round(100 * v["estrutura_ok"] / v["n"], 1),
                "difficulty_aderente_pct": round(100 * v["dif_ok"] / v["n"], 1),
                "consistencia_pct": (
                    round(100 * v["consist_ok"] / v["consist_verif"], 1)
                    if v["consist_verif"] else None
                ),
                "consistencia_verificavel_n": v["consist_verif"],
            }
            for ano, v in sorted(por_ano.items())
        },
        "pos_processamento": {
            **{k: status_counter.get(k, 0)
               for k in ("ok", "nao_verificavel", "corrigido",
                         "depende_de_visual", "falha")},
            "regeneracoes_total": regeneracoes_total,
        },
        "velocidade_cpu_real": {
            "tokens_por_segundo_geracao": round(statistics.mean(gen_tps_list), 1) if gen_tps_list else None,
            "latencia_media_total_s": round(statistics.mean(latencies), 2),
            "nota": (
                "latencia_media_total_s inclui o carregamento do modelo a cada "
                "chamada (processo novo por questão); tokens_por_segundo_geracao "
                "vem do próprio llama.cpp e reflete só a fase de geração."
            ),
        },
        "detalhes": results,
    }

    out_path = Path(report_path) if report_path else REPORT_PATH
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n===== Teste real (GGUF via llama.cpp) — {n} amostras =====")
    for section in ("estrutura", "por_ano", "pos_processamento", "velocidade_cpu_real"):
        print(f"[{section}]")
        for k, v in report[section].items():
            if k != "nota":
                print(f"  {k}: {v}")
    print(f"\nRelatório completo: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--llama-cli", help="caminho do binário llama-cli")
    parser.add_argument("--model", help="caminho do .gguf (padrão: acha em outputs/)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)

    parser.add_argument("--ano", help="ex: 5º — pula o modo interativo")
    parser.add_argument("--habilidade", help="ex: H08")
    parser.add_argument("--descricao", default="", help="descrição da habilidade")
    parser.add_argument("--dificuldade", default="Moderado")
    parser.add_argument("--n", type=int, default=1, help="variações a gerar")
    parser.add_argument("--quantidade", type=int, default=1,
                        help="quantas questões no lote. >1 usa o modo PLANEJADO por padrão "
                             "(uma questão por chamada, cobertura de subtemas; ver gerar_lote.py)")
    parser.add_argument("--sem-planejamento", action="store_true",
                        help="com --quantidade>1, volta ao modo antigo: uma única chamada pedindo N "
                             "questões (só a 1ª é verificada)")
    parser.add_argument("--max-tentativas-diversidade", type=int, default=2,
                        help="regenerações extras por slot quando a questão viola a diversidade")

    parser.add_argument("--batch", action="store_true", help="roda sobre data/val.jsonl")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--val", default=None, help="conjunto de avaliação (padrão: data/val.jsonl)")
    parser.add_argument("--report", default=None, help="arquivo de saída do relatório")

    parser.add_argument("--no-grammar", action="store_true",
                        help="não usar a grammar GBNF (estrutura fica por conta do modelo)")
    parser.add_argument("--retries", type=int, default=1,
                        help="regenerações máximas quando a validação reprova (padrão: 1)")
    parser.add_argument("--raw", action="store_true",
                        help="batch sem grammar nem verificação: mede o modelo cru")
    args = parser.parse_args()

    llama_cli = find_llama_cli(args.llama_cli)
    gguf_path = find_gguf(args.model)

    grammar = None
    if not args.no_grammar and not args.raw:
        if GRAMMAR_PATH.exists():
            grammar = GRAMMAR_PATH
        else:
            print(f"Aviso: {GRAMMAR_PATH} não encontrado — gerando sem grammar.")

    if args.batch:
        batch(llama_cli, gguf_path, args.threads, args.num_samples,
              grammar=grammar, retries=args.retries, raw=args.raw,
              val_path=args.val, report_path=args.report)
    elif args.ano and args.habilidade:
        run_one(
            llama_cli, gguf_path, args.ano, args.habilidade,
            args.descricao, args.dificuldade, args.threads, args.n,
            grammar=grammar, retries=args.retries, quantidade=args.quantidade,
            planejado=not args.sem_planejamento,
            max_tentativas_diversidade=args.max_tentativas_diversidade,
        )
    else:
        interactive(llama_cli, gguf_path, args.threads, grammar=grammar,
                    retries=args.retries)


if __name__ == "__main__":
    main()
