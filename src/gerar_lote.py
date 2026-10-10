"""Geração PLANEJADA de um lote de N questões (inferência offline, GGUF).

Por que existe: pedir "Gere N questões" numa única chamada deixava a
diversidade por conta da amostragem — e só a 1ª questão do lote passava por
check_consistency/fix_gabarito/depende_de_visual_ausente. No teste real, 9º H17
saiu 3/3 em triângulos. Aqui cada questão é gerada individualmente, guiada por
um plano de cobertura de subtemas (diversidade.planejar_lote), e TODAS passam
pelo mesmo pipeline validado de produção (test_model.generate_validated).

Fluxo, para cada slot do plano:
  1. prompt = USER_TEMPLATE(quantidade=1) + diversidade.sufixo_prompt(slot)
     (+ restrição explícita na regeneração). A gramática GBNF é mantida.
  2. generate_validated: estrutura, consistência, geometria, fix_gabarito,
     visual ausente.
  3. classificar + violacoes_diversidade contra o lote já aceito (inclui a
     guarda de dados insuficientes de slots de classificação).
  4. Se violar: regenera com diversidade.montar_restricao e seed diferente,
     até `max_tentativas_diversidade` vezes.
  5. Se esgotar: fica o MELHOR candidato, com QUALIDADE PRIMEIRO — uma questão
     válida nunca é trocada por uma inválida só por ser mais diversa. A
     violação remanescente é registrada.
Ao fim monta o wrapper {"questoes": [...]} (schema inalterado) e calcula as
métricas de diversidade do lote.

Custo: cada tentativa é uma chamada ao llama-cli (~10-15 s em CPU com 4
threads, recarregando o modelo). Um lote de N custa no mínimo N chamadas; as
regenerações de diversidade somam no pior caso N*max_tentativas chamadas.

Uso típico (pela CLI de test_model.py, onde é o padrão para --quantidade > 1):
    python tests/test_model.py --ano "9º" --habilidade H17 --quantidade 5
"""
import hashlib
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import diversidade as dv
import verificador_geometria as vg
from extract_data import USER_TEMPLATE
from schema_utils import (
    ALTERNATIVE_LETTERS,
    check_consistency_detalhado,
    check_structure,
)

# test_model.py mora em tests/ (é a CLI de inferência de produção, não um
# teste pytest — ver o cabeçalho dele), não ao lado deste arquivo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from test_model import MAX_NEW_TOKENS, generate_validated

# Qualidade do pós-processamento, do melhor para o pior. É o critério
# PRIMÁRIO na escolha entre candidatos de um slot; diversidade só desempata.
# Rank 0 = "não dá para entregar isto": é o gatilho do orçamento de qualidade
# abaixo. generate_validated devolve "falha" tanto para estrutura quebrada
# (alternativas repetidas) quanto para resposta fora das alternativas — os dois
# casos que a Fase 1 quer regenerar.
# "ok" (consistência VERIFICADA) > "nao_verificavel" (2026-10-10): antes valiam o
# mesmo, e com o verificador estendido (schema_utils) a diferença entre verificado
# e apenas não reprovado passou a ser informativa. Só desempata candidatos que o
# slot já gerou (nenhuma chamada extra).
RANK_STATUS = {"ok": 5, "nao_verificavel": 4, "corrigido": 3,
               "depende_de_visual": 1, "falha": 0}

# Orçamento de re-amostragem por QUALIDADE, independente do de diversidade.
# Consumido EXCLUSIVAMENTE por slots cujo melhor candidato ainda tem rank 0,
# então as ~97,8% de questões boas continuam custando 1 chamada, como hoje.
# Medido sobre outputs/diversidade_exp_C_s{0,1,2}.json (2,19% de falha):
# +0,026 chamadas por questão, ~+4 s num lote de 10 (+4,5%).
TENTATIVAS_QUALIDADE = 2


def _slots_fallback(quantidade, dificuldade):
    """Plano mínimo quando (ano, habilidade) não está na taxonomia: sem subtema
    (não inventamos um), mas o contexto ainda gira para não repetir cenário."""
    return [{"indice": i, "subtema": "geral", "subtema_rotulo": "geral",
             "tipo_raciocinio": "indefinido", "tipo_raciocinio_rotulo": "livre",
             "contexto": dv.CONTEXTOS[i % len(dv.CONTEXTOS)]["id"],
             "contexto_rotulo": dv.CONTEXTOS[i % len(dv.CONTEXTOS)]["rotulo"],
             "estrutura": "pergunta_direta", "estrutura_rotulo": "pergunta direta",
             "dificuldade": dificuldade} for i in range(quantidade)]


def montar_prompt(ano, habilidade, descricao, dificuldade, slot, restricao=None):
    """USER_TEMPLATE de produção (quantidade=1) + sufixo único de diversidade.
    O sufixo é o mesmo texto usado na destilação/treino (paridade)."""
    p = USER_TEMPLATE.format(quantidade=1, ano=ano, habilidade=habilidade,
                             descricao=descricao, dificuldade=dificuldade)
    if slot.get("subtema") != "geral" or slot.get("tipo_raciocinio") != "indefinido":
        p += dv.sufixo_prompt(slot)
    else:
        p += f" Contexto: {slot['contexto_rotulo']}."
    if restricao:
        p += f" Restrição: {restricao}"
    return p


# Inverso de USER_TEMPLATE: recupera (ano, habilidade, descricao, dificuldade) de
# um prompt de produção. A descrição pode terminar em ".." (ponto duplicado no
# banco), por isso o ponto final de USER_TEMPLATE não entra no grupo.
_RX_USER = re.compile(
    r"^Gere \d+ questão\(ões\) de matemática\. Ano: (\S+) ano\. "
    r"Habilidade: (\S+) — (.*)\. Dificuldade: ([^.\s]+)\.$", re.S)


def prompt_com_sufixo(ano, habilidade, descricao, dificuldade, seed, historico=None,
                      taxonomia=None):
    """Prompt de UMA questão com o sufixo de diversidade — o que o app deve mandar.

    92% do treino (2505/2717) tem o sufixo "Subtema/Tipo de raciocínio/Contexto";
    o prompt puro de USER_TEMPLATE cai no regime dos ~8% restantes (quase só itens
    reais em CAIXA ALTA com resolução de uma linha). Sorteia 1 slot do plano
    (determinístico dado seed/historico; sem taxonomia, só o contexto gira) e
    devolve (prompt, slot). O app deve variar `seed` a cada chamada e/ou passar
    `historico` (questões já geradas) para não repetir subtema/contexto.
    """
    try:
        slot = dv.planejar_lote(ano, habilidade, 1, dificuldade, seed=seed,
                                historico=historico, taxonomia=taxonomia)[0]
    except (KeyError, FileNotFoundError):
        slot = _slots_fallback(1, dificuldade)[0]
        slot["contexto"] = dv.CONTEXTOS[seed % len(dv.CONTEXTOS)]["id"]
        slot["contexto_rotulo"] = dv.CONTEXTOS[seed % len(dv.CONTEXTOS)]["rotulo"]
    return montar_prompt(ano, habilidade, descricao, dificuldade, slot), slot


def acrescentar_sufixo(user_msg, seed, historico=None, taxonomia=None):
    """Versão de prompt_com_sufixo para um prompt de produção já montado (val do
    gate). Prompt fora do formato de USER_TEMPLATE volta inalterado."""
    m = _RX_USER.match(user_msg.strip())
    if not m:
        return user_msg
    return prompt_com_sufixo(m.group(1), m.group(2), m.group(3), m.group(4), seed,
                             historico=historico, taxonomia=taxonomia)[0]


# Peso das violações no desempate entre candidatos de mesma qualidade.
# dados_ausentes torna a questão irresolvível no app (texto puro), então pesa
# mais que as violações de diversidade, que só empobrecem o lote.
# dados_insuficientes (slot de classificação sem nenhuma medida/propriedade no
# enunciado) é o mesmo defeito visto do lado da geometria: R1-Q2 "um ônibus
# passa por quatro pontos formando um quadrilátero; classifique quanto aos
# lados" e R2-Q7 "três garrafas PET formaram um triângulo" (9º H17,
# 2026-10-01) não têm resposta. Mesmo peso.
PESO_VIOLACAO = {"dados_ausentes": 3, "dados_insuficientes": 3}

# Nome de CLASSE nas alternativas: só então o slot de classificação produziu de
# fato uma pergunta de classificar. dv.dados_insuficientes_classificacao olha
# só o enunciado (de propósito: as alternativas sempre trazem nomes de classe e
# mascarariam a falta de dados); este filtro olha as alternativas para o
# inverso — "Quantos lados tem um triângulo?" (2/3/4/5/6) num slot de
# classificação não tem dígito nem propriedade no enunciado e está correta;
# acusá-la de "dados insuficientes" seria falso positivo. Regex simples sobre
# texto normalizado (sem acento), portável para TypeScript.
_CLASSE_NAS_ALTERNATIVAS = re.compile(
    r"equilater|isoscel|escalen|acutangul|obtusangul|retangul|quadrad|losang|trapezi"
    r"|paralelogram")


def _alternativas_classificam(questao):
    alts = questao.get("alternativas") if isinstance(questao, dict) else None
    if not isinstance(alts, dict):
        return False
    return bool(_CLASSE_NAS_ALTERNATIVAS.search(
        dv.normalizar_texto(" ".join(str(v) for v in alts.values()))))


def violacoes_slot(aceitas, obj, slot, ano, habilidade, quantidade, taxonomia=None):
    """violacoes_diversidade com a guarda de dados insuficientes LIGADA.

    A guarda é opt-in em diversidade.py porque muda quantas chamadas este
    módulo faz por slot; quem liga é aqui. Ela só vale para tipos com
    `exige_dados` (hoje: os de classificação de 9º H17) e só é mantida quando
    as alternativas são nomes de classe (ver _alternativas_classificam).
    Medido (2026-10-01, só leitura): nas 20 auditadas acusa R1-Q2 e R2-Q7 (os
    2 rotulados dados ausentes) e R1-Q8 (bolo/tampa, sem dado nenhum), nenhuma
    das 5 corretas. No corpus gerado de 9º H17 (133 únicas) a guarda bruta
    acusava 2 e com o filtro acusa 1 (G-9H17-0023); a que saiu tem alternativas
    descritivas ("Pode ter dois ângulos obtusos...") e não é pergunta de
    classificar. Nas 311 ocorrências de 9º H17 em outputs/: 16 -> 1, e 13 das
    15 retiradas são placeholders de dry-run com alternativas 10/12/14/16.
    """
    viol = dv.violacoes_diversidade(aceitas, obj, slot, ano, habilidade, quantidade,
                                    taxonomia=taxonomia, checar_dados_classificacao=True)
    if not _alternativas_classificam(obj):
        viol = [v for v in viol if v["tipo"] != "dados_insuficientes"]
    return viol


def _chave(cand):
    """Ordenação de candidatos: qualidade primeiro, depois menor peso de violações."""
    return (RANK_STATUS.get(cand["status"], 0) if cand["obj"] is not None else -1,
            -sum(PESO_VIOLACAO.get(v["tipo"], 1) for v in cand["violacoes"]))


def _utilizavel(cand):
    """O candidato pode ser entregue ao aluno como está?

    False para os dois casos que a Fase 1 quer regenerar: sem questão parseada
    e status de rank 0 ("falha" = estrutura quebrada OU resposta fora das
    alternativas). É o gatilho de TENTATIVAS_QUALIDADE.
    """
    return cand["obj"] is not None and RANK_STATUS.get(cand["status"], 0) > 0


# ---------------------------------------------------------------------------
# 1.3 — PERMUTAÇÃO DETERMINÍSTICA DA LETRA DO GABARITO
#
# Problema medido: nos 1.188 itens de outputs/diversidade_*.json a letra A é o
# gabarito em 38,5% das questões (por relatório chega a 53,3%), contra 20%
# esperados. O gate G6 de promover_checkpoint mede isso sobre n=30 e reprova.
# O modelo herdou e amplificou o viés do treino (A em 29,6% de train_curado).
#
# Fato estrutural que dita o desenho: em 63,2% das questões geradas a
# alternativa E é "Nenhuma das alternativas anteriores". Isso é uma ÂNCORA
# POSICIONAL — o texto só faz sentido na última posição, e permutá-la produz
# "A) Nenhuma das alternativas anteriores". Vetar essas questões custaria 2/3
# do corpus; a saída é CONGELAR o slot âncora e permutar só os livres.
#
# O predicado é conservador: qualquer dúvida devolve a questão intacta. O custo
# de um veto é só não corrigir o viés naquela questão; o custo de permutar algo
# que não podia ser permutado é entregar uma questão quebrada.
# ---------------------------------------------------------------------------

_ANCORA_PATTERN = re.compile(
    # MASCULINO TAMBÉM, e sem exigir substantivo conhecido depois do "das/dos".
    # A forma feminina sozinha deixava passar "Nenhum dos anteriores", "Nenhum
    # dos três", "Nenhum dos quadriláteros" — 10 questões dos relatórios reais
    # tinham a âncora DESLOCADA para o meio da lista, que é exatamente o modo de
    # falha que o congelamento existe para impedir. Aceitar um substantivo
    # qualquer depois de "nenhum d..." é a direção segura: o custo de congelar
    # uma alternativa que não era âncora é só não permutar aquela letra.
    r"nenhum[ao]?\s+d[aeo]s?\b"
    r"|todos\s+(?:os\s+)?(?:anteriores|itens|acima)"
    r"|todas\s+(?:as\s+)?(?:alternativas?|op[çc][õo]es|anteriores|est[ãa]o)"
    r"|\bn\.?\s*d\.?\s*a\.?\b"
    r"|(?:alternativas?|op[çc][õo]es|respostas?)\s+(?:anteriores|acima|abaixo)"
    # "Ambas as anteriores" / "As duas primeiras" não apareceram nos 2.669 itens
    # medidos, mas são âncoras de verdade e incluir custa zero.
    r"|\bambas\s+as\s+(?:anteriores|primeiras)\b"
    r"|\bas\s+duas\s+(?:primeiras|anteriores|[úu]ltimas)\b"
    r"|^\s*(?:nenhum|nenhuma|todos|todas)\b",
    re.I,
)

# Substantivos que introduzem uma REFERÊNCIA A LETRA em português. São
# insensíveis a caixa SÓ NELES: o `[A-E]` dos padrões abaixo NÃO pode ser
# case-insensitive, senão `\b[A-E]\b` passaria a casar o artigo "a", a
# conjunção "e" e a preposição "o", vetando praticamente todo o corpus. O bug
# era o inverso — sem nenhuma flag, "Alternativa D" e "Letra B" (início de
# frase, que é onde a citação mais aparece) escapavam do veto.
# Antes isso era a flag LOCAL (?i:...), que é ES2025 (RegExp modifiers): o V8
# do Node 24 aceita, mas Hermes e JavaScriptCore antigos dão SyntaxError e
# permutar_alternativas quebraria no app (revisão adversarial de 2026-10-01).
# _sem_caixa gera as classes explícitas ("op[çc]" -> "[Oo][Pp][çÇcC]"), que
# qualquer motor aceita; tests/test_integracao_geometria confere que o
# resultado casa exatamente o que a flag local casava.


def _sem_caixa(padrao):
    """Insensibilidade a caixa sem flag: cada letra vira [xX]. Só para padrões
    de palavras (letras, ?, |, parênteses e classes simples, sem escapes)."""
    assert "\\" not in padrao
    saida, em_classe = [], False
    for c in padrao:
        if c in "[]":
            em_classe = c == "["
            saida.append(c)
        elif c.isalpha() and c.upper() != c:
            saida.append(c + c.upper() if em_classe else f"[{c}{c.upper()}]")
        else:
            saida.append(c)
    return "".join(saida)


_SUBST_LETRA = "(?:" + _sem_caixa(r"alternativas?|letras?|op[çc](?:[ãa]o|[õo]es)|itens|item") + ")"
_SUBST_RESPOSTA = "(?:" + _sem_caixa(r"respostas?|afirma[çc](?:[ãa]o|[õo]es)|assertivas?") + ")"

# Uma alternativa que REFERENCIA outras letras ("A e C") deixa de fazer sentido
# quando as letras mudam de dono. 9 casos nos relatórios, 8 no treino.
_REF_CRUZADA = re.compile(
    r"\b[A-E]\s*(?:e|,|ou)\s*[A-E]\b|"
    + _SUBST_LETRA + r"\s+[A-E]\b"
    r"|\b(?:apenas|somente)\s+[IVX]+\b"
    r"|\b[IVX]+\s*(?:e|,)\s*[IVX]+\b",
)
# Alternativa cujo texto é só uma letra: "A" / "(B)" / "C." — o texto É o rótulo.
_ALT_SO_LETRA = re.compile(r"^\s*[\(\[]?\s*[A-E]\s*[\)\]\.]?\s*$")
# Enunciado que já embute a lista "A) ... B) ...". A janela de 120 chars entre
# A e B é o compromisso medido: pega 41 casos reais sem estourar em falso
# positivo (deixa passar listas muito longas — risco aceito e documentado).
_ENUN_LISTA = re.compile(r"\bA\s*[\.\):\-]\s*[^\n]{1,120}?\bB\s*[\.\):\-]\s*\S")
_CITA_LETRA = re.compile(
    _SUBST_LETRA + r"\s+[A-E]\b|[A-E]\s*\)\s*\S",
)
# Na RESOLUÇÃO o padrão é mais largo: "a resposta correta é C", 'a letra "B"',
# "C: 20" (tabela) e a referência por posição sem letra.
#
# A FORMA CANÔNICA EM PORTUGUÊS NÃO COLA A LETRA NO SUBSTANTIVO: "a alternativa
# correta é B", "Alternativa D (16 cm) é correta", "a afirmação correta é a A".
# O padrão antigo só reconhecia "alternativa B" com a letra imediatamente
# depois, então permutar_lote trocava a letra do gabarito de questões cuja
# resolução nomeava a letra antiga — medidas 10 questões assim nos 14
# relatórios reais, entregues ao aluno se contradizendo. Daí a janela curta
# `[^.\n]{0,24}?` entre o substantivo e a letra: ela cobre "correta é", "correta
# é a", "(16 cm) é" sem atravessar o ponto final para a frase seguinte.
#
# Vetar demais custa só cobertura de permutação; vetar de menos entrega uma
# questão quebrada. O veto é deliberadamente assimétrico nessa direção.
_RESOL_CITA_LETRA = re.compile(
    _SUBST_LETRA + r"\s+[A-E]\b"
    r"|" + _SUBST_RESPOSTA + r"\s+"
    r"(?:correta\s+)?(?:[ée]\s+)?(?:a\s+)?[A-E]\b"
    r"|(?:" + _SUBST_LETRA + r"|" + _SUBST_RESPOSTA + r")[^.\n]{0,24}?\b[A-E]\b"
    r"|[A-E]\s*\)\s*\S|\"[A-E]\"|'[A-E]'|[A-E]\s*[:\-]\s"
    r"|(?:primeir|segund|terceir|quart|quint|[úu]ltim)\w*\s+"
    r"(?:alternativa|op[çc][ãa]o|item|resposta)",
)
_ENUN_ORDEM = re.compile(
    r"em ordem|ordem\s+(?:crescente|decrescente|alfab)"
    r"|(?:primeir|segund|terceir|quart|quint|[úu]ltim)\w*\s+"
    r"(?:alternativa|op[çc][ãa]o|item|resposta)",
    re.I,
)


def _ancoras(alternativas):
    """Letras cujo TEXTO é uma âncora posicional — congeladas na permutação."""
    return {L for L in ALTERNATIVE_LETTERS
            if _ANCORA_PATTERN.search(str(alternativas.get(L) or ""))}


def vetos_permutacao(questao):
    """Motivos pelos quais ESTA questão não pode ter as letras permutadas.

    Lista vazia = segura. Cada veto é um caso em que a letra do gabarito carrega
    significado fora do campo `resposta_correta`, de modo que trocá-la quebraria
    a questão. Medido nos 1.188 itens dos relatórios: 10,4% de vetos (era 8,8%
    antes do aperto de V8 em 2026-09-30). Somados os gabaritos que já estão num
    slot âncora, 808 das 1.188 questões trocam de letra, contra 1.051 antes do
    aperto: a correção troca cobertura por correção, e é a troca certa — vetar
    demais só deixa de corrigir o viés naquela questão, vetar de menos entrega
    ao aluno uma questão que se contradiz.
    """
    vetos = []
    if not isinstance(questao, dict):
        return ["nao_e_dict"]
    alts = questao.get("alternativas")
    gab = questao.get("resposta_correta")
    if (not isinstance(alts, dict)
            or set(alts) != set(ALTERNATIVE_LETTERS)
            or not all(isinstance(alts[L], str) and alts[L].strip()
                       for L in ALTERNATIVE_LETTERS)
            or not (isinstance(gab, str) and gab in tuple(ALTERNATIVE_LETTERS))):
        return ["schema"]                                                   # V1
    textos = [alts[L].strip().lower() for L in ALTERNATIVE_LETTERS]
    if len(set(textos)) != len(ALTERNATIVE_LETTERS):
        vetos.append("alternativas_duplicadas")                             # V2
    if any(_REF_CRUZADA.search(alts[L]) for L in ALTERNATIVE_LETTERS):
        vetos.append("alternativa_referencia_outras")                       # V4
    if any(_ALT_SO_LETRA.match(alts[L]) for L in ALTERNATIVE_LETTERS):
        vetos.append("alternativa_e_so_uma_letra")                          # V5
    enunciado = str(questao.get("enunciado") or "")
    if _ENUN_LISTA.search(enunciado):
        vetos.append("enunciado_embute_lista")                              # V6
    if _CITA_LETRA.search(enunciado):
        vetos.append("enunciado_cita_letra")                                # V7
    if _RESOL_CITA_LETRA.search(str(questao.get("resolucao_passo_a_passo") or "")):
        vetos.append("resolucao_cita_letra")                                # V8
    if _ENUN_ORDEM.search(enunciado):
        vetos.append("enunciado_pede_ordem")                                # V9
    if len(set(ALTERNATIVE_LETTERS) - _ancoras(alts)) < 2:
        vetos.append("sem_slots_livres")                                    # V10
    return vetos


def _hash_int(*partes):
    """Inteiro determinístico e estável entre execuções (hash() do Python não é:
    PYTHONHASHSEED randomiza strings, o que quebraria a comparação pareada)."""
    h = hashlib.sha256("\x1f".join(str(p) for p in partes).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big")


def _veredito(questao):
    """(ok, motivo) de check_consistency_detalhado, com ok ordenável."""
    ok, _sug, motivo = check_consistency_detalhado(questao)
    return ({True: 2, None: 1, False: 0}[ok], motivo)


def _assinatura_geometria(questao):
    """O que o verificador de geometria conclui, SEM depender das letras.

    (veredito, motivo, {texto da alternativa: V/F/I/?}, texto da sugestão).
    Indexar pelo TEXTO e não pela letra é o que torna a assinatura comparável
    antes e depois de permutar: a alternativa "Obtusângulo" tem de continuar
    valendo V onde quer que caia.
    """
    veredito, det = vg.verificar_geometria(questao)
    if veredito == "nao_aplicavel":
        # O verificador não concluiu nada; os `valores` parciais que ele deixa
        # no detalhe (ex.: motivo alternativa_nao_mapeada, G-9H17-0131) não são
        # uma conclusão e não podem vetar a permutação — com eles, 10 de 9.575
        # permutações medidas eram revertidas sem motivo.
        return (veredito,)
    alts = questao.get("alternativas") or {}
    valores = det.get("valores") or {}
    sug = det.get("sugestao")
    return (veredito, det.get("motivo"),
            tuple(sorted((str(alts.get(L)), str(v)) for L, v in valores.items())),
            str(alts.get(sug)) if sug in alts else None)


def permutar_alternativas(questao, seed, alvo=None):
    """Reatribui as letras das alternativas mantendo o SIGNIFICADO da questão.

    O texto da alternativa correta viaja junto com `resposta_correta`, então a
    resolução continua batendo: validado sobre 808 questões reais dos
    relatórios — check_consistency não regrediu em nenhuma e check_structure não
    mudou em nenhuma.

    `alvo`: letra de destino do gabarito. Se None, escolhe por hash determinístico
    entre os slots livres (o balanceamento de lote fica em permutar_lote, que é
    quem enxerga o lote inteiro).

    Devolve SEMPRE uma questão válida: a original, intacta, quando há qualquer
    veto, quando o gabarito já está num slot âncora, ou quando a re-verificação
    pós-permutação piorar check_consistency / check_structure ou MUDAR o que o
    verificador de geometria conclui (_assinatura_geometria).
    """
    if vetos_permutacao(questao):
        return questao
    alts = questao["alternativas"]
    gab = questao["resposta_correta"]
    ancoras = _ancoras(alts)
    if gab in ancoras:
        # Gabarito é a própria âncora ("Nenhuma das alternativas anteriores"):
        # mover o texto para A destruiria a questão. No-op deliberado.
        return questao
    livres = [L for L in ALTERNATIVE_LETTERS if L not in ancoras]
    enunciado = str(questao.get("enunciado") or "")
    if alvo is None:
        alvo = min(livres, key=lambda L: _hash_int(seed, enunciado, L))
    if alvo not in livres:
        return questao

    textos = [alts[L] for L in livres if L != gab]
    random.Random(_hash_int(seed, enunciado, "s")).shuffle(textos)
    novo = dict(alts)
    novo[alvo] = alts[gab]
    for L, texto in zip([L for L in livres if L != alvo], textos):
        novo[L] = texto

    permutada = dict(questao)
    permutada["alternativas"] = novo
    permutada["resposta_correta"] = alvo

    # RE-VERIFICAÇÃO EXPLÍCITA (exigida pelo item 1.3). O predicado acima é
    # sintático; esta é a checagem semântica de que nada piorou. Medida: 0
    # reversões em 808 questões reais — e é exatamente por isso que ela é
    # barata de manter e valiosa se alguma heurística futura falhar.
    if _veredito(permutada)[0] < _veredito(questao)[0]:
        return questao
    if (check_structure({"questoes": [permutada]}, 1)
            != check_structure({"questoes": [questao]}, 1)):
        return questao
    # RE-VERIFICAÇÃO DE GEOMETRIA (2026-10-01). Questões de classificação de 9º
    # H17 nomeiam entidades com LETRAS que coincidem com as das alternativas:
    # R1-Q1 "a barraca A tem todos os ângulos agudos, a barraca B ..." com
    # alternativas compostas "A barraca A é um triângulo retângulo, a barraca B
    # é ...". O texto viaja inteiro (nada é reescrito), mas o verificador lê
    # "A"/"B"/"C" como nomes de entidade; se a leitura dele mudar com a troca
    # de posição — veredito, valor de alguma alternativa ou a alternativa
    # sugerida — a permutação não acontece. Exigir a assinatura INTEIRA igual
    # (não só o veredito) é a direção segura: vetar custa só não corrigir o
    # viés de letra naquela questão. Medido (20 auditadas + 198 do corpus
    # gerado + 311 de 9º H17 em outputs/, 5 seeds x 5 alvos): 9.575 permutações,
    # 0 revertidas por esta checagem e veredito idêntico em todas — inclusive
    # R1-Q1. Ver tests/test_integracao_geometria.py.
    if _assinatura_geometria(permutada) != _assinatura_geometria(questao):
        return questao
    return permutada


def permutar_lote(questoes, seed):
    """Permuta um LOTE inteiro balanceando a letra do gabarito (guloso).

    Alvo de cada questão = slot livre MENOS usado até aqui no lote, com desempate
    por sha256(seed|enunciado|letra). As questões vetadas continuam contando em
    `usados`, senão o balanceamento ignoraria o viés que elas próprias trazem.

    Efeito MEDIDO (re-medido em 2026-09-30, depois do aperto dos vetos V8 e da
    âncora masculina) nos 1.188 itens reais de outputs/diversidade_*.json, com
    permutar_lote(seed=12345) lote a lote: 808 questões trocam de letra e a
    letra mais frequente cai de 38,5% para 23,4% global, com faixa 22,4–30,3%
    por relatório×modo (pior caso 53,3% -> 25,3%). O balanceamento é por LOTE
    (3 a 15 questões), não por relatório, e é isso que explica a faixa ser mais
    larga que a de um balanceamento global.

    ATENÇÃO — ISTO NÃO CONSERTA O GATE G6. G6 lê
    `estrutura.gabarito_letra_mais_frequente_pct`, produzido por
    test_model.batch e evaluate.py, que chamam generate_validated questão a
    questão e nunca passam por gerar_lote_planejado. A permutação não toca a
    amostra que G6 mede; o teto absoluto do G6 continua sendo item aberto do
    plano. Os números acima valem para os lotes entregues, não para G6.

    Determinístico: mesma (seed, ordem do lote, conteúdo) => mesma saída, então
    a comparação pareada baseline/candidato continua válida.
    """
    usados, saida = Counter(), []
    for q in questoes:
        if not isinstance(q, dict):
            saida.append(q)
            continue
        gab = q.get("resposta_correta")
        alts = q.get("alternativas")
        if vetos_permutacao(q) or not isinstance(alts, dict):
            usados[gab] += 1
            saida.append(q)
            continue
        ancoras = _ancoras(alts)
        if gab in ancoras:
            usados[gab] += 1
            saida.append(q)
            continue
        enunciado = str(q.get("enunciado") or "")
        livres = [L for L in ALTERNATIVE_LETTERS if L not in ancoras]
        alvo = min(livres, key=lambda L: (usados[L], _hash_int(seed, enunciado, L)))
        nova = permutar_alternativas(q, seed, alvo=alvo)
        usados[nova.get("resposta_correta")] += 1
        saida.append(nova)
    return saida


def gerar_lote_planejado(llama_cli, gguf_path, ano, habilidade, descricao, dificuldade,
                         quantidade, threads, grammar=None, retries=1, base_seed=None,
                         max_tentativas_diversidade=2, gen_fn=None, historico=None,
                         taxonomia=None, verbose=False, permutar=True, modo_geometria=None):
    """Gera `quantidade` questões com plano de subtemas e regeneração guiada.

    gen_fn: substituto de test_model.generate (mesma assinatura) — para testes
    sem llama-cli. base_seed: torna o lote reprodutível (plano + seeds); None
    sorteia uma base por lote, de modo que tentativas/slots nunca repetem seed
    (no modo antigo toda regeneração usava a seed fixa 1001).

    permutar: balanceia a letra do gabarito no lote (permutar_lote). Roda no FIM
    do pipeline, depois de generate_validated/fix_gabarito — se rodasse antes,
    fix_gabarito poderia reescrever `resposta_correta` e desfazer o
    balanceamento. Reutiliza base_seed, de modo que o lote inteiro (plano,
    seeds e permutação) é reprodutível.

    Retorna dict: obj (wrapper {"questoes": [...]}), questoes, plano, metricas,
    violacoes (por slot, as que restaram), tempo_s, regeneracoes_diversidade,
    flags (check_structure do wrapper), detalhes (por slot),
    permutacoes_gabarito (quantas questões trocaram de letra).

    modo_geometria: repassado a generate_validated ("ativo" | "sombra" | None =
    verificador_geometria.MODO_GEOMETRIA, hoje "sombra"). Em modo ativo, questão
    reprovada pela geometria sai de generate_validated como "falha" (rank 0),
    então consome TENTATIVAS_QUALIDADE exatamente como a resposta fora das
    alternativas.
    """
    t0 = time.perf_counter()
    quantidade = int(quantidade)
    if base_seed is None:
        base_seed = random.randrange(1, 10 ** 6)
    try:
        plano = dv.planejar_lote(ano, habilidade, quantidade, dificuldade,
                                 seed=base_seed, historico=historico, taxonomia=taxonomia)
        tem_taxonomia = True
    except (KeyError, FileNotFoundError):
        plano, tem_taxonomia = _slots_fallback(quantidade, dificuldade), False

    aceitas, detalhes, violacoes_final = [], [], []
    regen_div = 0
    for slot_original in plano:
        t_slot = time.perf_counter()
        melhor, restricao, candidatos = None, None, []
        slot = slot_original  # pode virar cópia com contexto trocado (ver abaixo)
        # O teto do laço é o MAIOR dos dois orçamentos. Quem de fato decide
        # quando parar são os dois `break` do corpo: o de diversidade (candidato
        # bom e diverso) e o de qualidade (já existe candidato utilizável). Sem
        # o max(), um lote rodado com max_tentativas_diversidade=0 — como todos
        # os relatórios de outputs/ — entregava a primeira amostra mesmo quando
        # ela era estruturalmente quebrada ou irrespondível.
        for tent in range(max(max_tentativas_diversidade, TENTATIVAS_QUALIDADE) + 1):
            prompt = montar_prompt(ano, habilidade, descricao, dificuldade, slot, restricao)
            # Seed distinta por (lote, slot, tentativa de diversidade);
            # generate_validated ainda soma o índice da tentativa de qualidade.
            seed_item = base_seed * 1000 + slot["indice"] * 10 + tent
            r = generate_validated(llama_cli, gguf_path, prompt, threads, MAX_NEW_TOKENS,
                                   grammar=grammar, retries=retries, quantidade=1,
                                   base_seed=seed_item, gen_fn=gen_fn,
                                   modo_geometria=modo_geometria)
            viol = []
            if r["obj"] is not None and tem_taxonomia:
                viol = violacoes_slot(aceitas, r["obj"], slot, ano, habilidade,
                                      quantidade, taxonomia=taxonomia)
            cand = {"obj": r["obj"], "status": r["status"], "violacoes": viol,
                    "regeneracoes": r["regeneracoes"], "elapsed": r["elapsed"],
                    "gen_tps": r["gen_tps"], "tentativa": tent,
                    "geometria": (r.get("geometria") or {}).get("veredito"),
                    "reprovacoes_geometria": r.get("reprovacoes_geometria", 0)}
            candidatos.append(cand)
            if melhor is None or _chave(cand) > _chave(melhor):
                melhor = cand
            # Para quando o candidato é bom E diverso. Se é diverso mas de
            # qualidade inferior, generate_validated já esgotou os retries dele;
            # uma nova tentativa aqui só gastaria tempo sem motivo de diversidade.
            # Candidato estruturalmente quebrado/sem questão ("falha") usa as
            # tentativas restantes como novas chances de qualidade (seed nova),
            # mesmo sem violação de diversidade.
            if not viol and _utilizavel(cand):
                break
            if viol and tent < max_tentativas_diversidade:
                regen_div += 1
                restricao = dv.montar_restricao(viol, habilidade, slot)
                # Se a violação sugeriu um contexto diferente, o PRÓPRIO slot da
                # próxima tentativa passa a usar esse contexto — senão
                # sufixo_prompt continua emitindo "Contexto: X" logo acima da
                # restrição "use o contexto: Y", uma instrução contraditória.
                ctx_novo = next((v.get("sugerido") for v in viol if v["tipo"] == "contexto_repetido"), None)
                if ctx_novo and ctx_novo != slot.get("contexto"):
                    slot = {**slot, "contexto": ctx_novo,
                            "contexto_rotulo": dv.rotulo_contexto(ctx_novo, taxonomia)}
                if verbose:
                    print(f"  [slot {slot['indice']}] regenerando: {restricao}")
                continue
            # Daqui para baixo não há mais orçamento de DIVERSIDADE para o slot.
            # O orçamento de QUALIDADE só é consumido enquanto nenhum candidato
            # utilizável tiver aparecido; assim ele nunca infla
            # regeneracoes_diversidade (que alimenta o gate G11) nem injeta
            # restrição textual de diversidade indevida.
            if _utilizavel(melhor):
                break
            if verbose:
                print(f"  [slot {slot['indice']}] re-amostrando por qualidade "
                      f"(status={melhor['status']})")

        if melhor["obj"] is not None:
            aceitas.append(melhor["obj"])
        violacoes_final.append({"indice": slot["indice"],
                                "violacoes": melhor["violacoes"]})
        detalhes.append({
            "indice": slot["indice"], "subtema_planejado": slot["subtema"],
            "classificacao": (dv.classificar_questao(melhor["obj"], ano, habilidade, taxonomia)
                              if melhor["obj"] is not None and tem_taxonomia else None),
            "status": melhor["status"], "tentativas_diversidade": len(candidatos),
            "tentativa_escolhida": melhor["tentativa"],
            "regeneracoes_qualidade": sum(c["regeneracoes"] for c in candidatos),
            # aditivos (2026-10-01): veredito de geometria do candidato
            # escolhido e quantas amostras do slot a geometria reprovou.
            "geometria": melhor["geometria"],
            "reprovacoes_geometria": sum(c["reprovacoes_geometria"] for c in candidatos),
            "tempo_modelo_s": round(sum(c["elapsed"] for c in candidatos), 2),
            "tempo_s": round(time.perf_counter() - t_slot, 2),
            "gen_tps": melhor["gen_tps"],
        })

    # Permutação por ÚLTIMO: depois de fix_gabarito (senão ele desfaria o
    # balanceamento) e antes de montar o wrapper e as métricas, para que o que é
    # medido seja exatamente o que é entregue.
    permutacoes = 0
    if permutar and aceitas:
        antes = [q.get("resposta_correta") if isinstance(q, dict) else None
                 for q in aceitas]
        aceitas = permutar_lote(aceitas, base_seed)
        permutacoes = sum(
            1 for q, a in zip(aceitas, antes)
            if isinstance(q, dict) and q.get("resposta_correta") != a)

    wrapper = {"questoes": aceitas}
    metricas = None
    if tem_taxonomia and aceitas:
        metricas = dv.diversity_score(aceitas, ano, habilidade, taxonomia=taxonomia)
        metricas.pop("classificacoes", None)  # já estão em detalhes
    tempo = time.perf_counter() - t0
    return {
        "obj": wrapper, "questoes": aceitas, "plano": plano, "metricas": metricas,
        "violacoes": violacoes_final, "tempo_s": round(tempo, 2),
        "tempo_medio_por_questao_s": round(tempo / max(1, quantidade), 2),
        "regeneracoes_diversidade": regen_div,
        "permutacoes_gabarito": permutacoes,
        "flags": check_structure(wrapper, quantidade_esperada=quantidade),
        "detalhes": detalhes, "base_seed": base_seed, "planejado": tem_taxonomia,
    }
