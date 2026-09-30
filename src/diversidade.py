"""
Diversidade e cobertura semântica de lotes de questões (puro Python, sem deps).

Problema que resolve: tanto a destilação (professor) quanto a inferência (--n,
--quantidade) só variavam o CONTEXTO narrativo sorteado e a amostragem. Contexto
não é diversidade matemática, e trocar números também não. Na prática, 9º H17
saiu 3/3 em triângulos, sem nenhum quadrilátero.

Peças (todas determinísticas):
  - taxonomia (data/taxonomia_subtemas.json, gerada por src/build_taxonomia.py):
    (ano, habilidade) -> subtemas -> tipos de raciocínio; contextos e estruturas
    são listas GLOBAIS (não são subtema).
  - classificar_questao: léxico/regex -> subtema, tipo_raciocinio, objeto,
    contexto, estrutura, representacao.
  - planejar_lote: plano de slots com cobertura de subtemas (round-robin
    ponderado pelo histórico) e rotação de raciocínio/contexto/estrutura.
  - métricas decomponíveis e diversity_score com pesos expostos.
  - violacoes_diversidade / montar_restricao / gerar_lote_diverso: regeneração
    guiada por restrição textual, com limite de tentativas.
  - sufixo_prompt: formato ÚNICO anexado ao USER_TEMPLATE (treino, destilação e
    inferência usam o mesmo texto, para não quebrar a paridade).

A chave é sempre (ano, habilidade): o mesmo código H muda de sentido por ano.
Nada aqui altera o schema {"questoes":[...]} nem substitui as validações de
schema_utils — é uma camada a mais.
"""
import json
import math
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAXONOMIA_PATH = ROOT / "data" / "taxonomia_subtemas.json"

# Regenerações extras por slot quando o candidato viola a diversidade. Cada
# tentativa custa uma chamada inteira ao modelo (~10-15 s em CPU), por isso é baixo.
MAX_TENTATIVAS_DIVERSIDADE = 2
# Jaccard de trigramas de palavras (números mascarados) acima do qual duas
# questões são "a mesma questão".
LIMIAR_NEAR_DUP = 0.6

# Pesos do diversity_score (somam 1). Cobertura e duplicata pesam mais porque
# são as falhas observadas (concentração em um subtema; templates com números trocados).
PESOS_DIVERSIDADE = {
    "coverage_score": 0.30,
    "nao_duplicata": 0.25,
    "dissimilaridade": 0.15,
    "raciocinio_diversity": 0.10,
    "structural_diversity": 0.10,
    "context_diversity": 0.10,
}

# --------------------------------------------------------------------------
# Eixos globais. Regex sempre sobre texto normalizado (sem acento, minúsculo).
# --------------------------------------------------------------------------
CONTEXTOS = [
    {"id": "feira", "rotulo": "feira livre", "palavras_chave": r"feira|barraca|feirante"},
    {"id": "esporte", "rotulo": "campeonato esportivo", "palavras_chave": r"futebol|campeonato|\btime\b|\bgols?\b|partida|corrida|atleta|quadra|basquete|volei"},
    {"id": "culinaria", "rotulo": "receita de bolo", "palavras_chave": r"receita|\bbolo|cozinh|ingrediente|xicara|farinha"},
    {"id": "viagem", "rotulo": "viagem de ônibus", "palavras_chave": r"onibus|viagem|\bcarro|estrada|\btrem\b|passageir|rodoviaria"},
    {"id": "biblioteca", "rotulo": "biblioteca da escola", "palavras_chave": r"biblioteca|\blivros?\b|leitura|paginas?"},
    {"id": "horta", "rotulo": "horta comunitária", "palavras_chave": r"horta|canteiro|semente|jardim"},
    {"id": "festa", "rotulo": "festa junina", "palavras_chave": r"festa|junina|aniversario|convidad|balao|baloes"},
    {"id": "mesada", "rotulo": "mesada e cofrinho", "palavras_chave": r"mesada|cofrinho|economiz|poupanc|guardou"},
    {"id": "loja", "rotulo": "loja de brinquedos", "palavras_chave": r"\bloja|brinquedo|desconto|pagou|\bpreco|r\$|vendeu|comprou"},
    {"id": "reciclagem", "rotulo": "campanha de reciclagem", "palavras_chave": r"recicla|\blixo|garrafa|latinha|meio ambiente"},
    {"id": "padaria", "rotulo": "padaria", "palavras_chave": r"padaria|\bpao\b|\bpaes\b|padeiro"},
    {"id": "parque", "rotulo": "passeio ao parque", "palavras_chave": r"parque|\bpraca\b|passeio|piquenique"},
    {"id": "colecao", "rotulo": "coleção de figurinhas", "palavras_chave": r"figurinha|colec|album"},
    {"id": "mercado", "rotulo": "mercado do bairro", "palavras_chave": r"mercado|supermercado|mercearia"},
    {"id": "escola", "rotulo": "gincana escolar", "palavras_chave": r"gincana|escola|\bturma|alunos?\b|professora?\b|sala de aula"},
    {"id": "construcao", "rotulo": "reforma de uma casa", "palavras_chave": r"terreno|constru|\bobra\b|parede|\bpiso\b|calcada|\bmuro\b|reforma|pedreiro"},
]
SEM_CONTEXTO = "sem_contexto"

ESTRUTURAS = [
    {"id": "verdadeiro_falso_implicito", "rotulo": "julgamento de afirmações",
     "palavras_chave": r"afirma|\bcorreta\b|incorreta|verdadeir|\bfals[ao]\b"},
    {"id": "identificacao_propriedade", "rotulo": "identificação de propriedade",
     "palavras_chave": r"classific|chamad|recebe o nome|propriedade|qual (e|das|dos) .{0,20}(tipo|nome)|pode ser"},
    {"id": "comparacao", "rotulo": "comparação", "palavras_chave": r"\bmaior\b|\bmenor\b|compar|mais .{0,20}do que"},
    {"id": "problema_contextualizado", "rotulo": "problema contextualizado", "palavras_chave": None},
    {"id": "calculo", "rotulo": "cálculo", "palavras_chave": r"calcul|resultado|quant[oa]s?\b|valor"},
    {"id": "pergunta_direta", "rotulo": "pergunta direta", "palavras_chave": None},
]

# `habilita`: regex sobre a DESCRIÇÃO da habilidade (usada por build_taxonomia);
# `palavras_chave`: regex sobre a QUESTÃO (usada pelo classificador).
TIPOS_RACIOCINIO = [
    {"id": "identificacao", "rotulo": "identificação/reconhecimento",
     "habilita": r"identificar|reconhecer|nomear|\bler\b|ler/",
     "palavras_chave": r"identifi|reconhec|chamad|qual (figura|objeto|solido)|qual (e|das|dos) .{0,20}(nome|representa)"},
    {"id": "classificacao_lados", "rotulo": "classificação quanto aos lados",
     "habilita": r"classific.{0,60}lados",
     "palavras_chave": r"quanto aos lados|lados? (iguais|diferentes|congruentes)|isosceles|escaleno|equilater"},
    {"id": "classificacao_angulos", "rotulo": "classificação quanto aos ângulos",
     "habilita": r"classific.{0,80}angulos",
     "palavras_chave": r"quanto aos angulos|angulos? (reto|agudo|obtuso)s?|acutangul|obtusangul|angulos? internos"},
    {"id": "classificacao_atributos", "rotulo": "classificação por atributos",
     "habilita": r"classific.{0,60}atribut",
     "palavras_chave": r"classific|grupo|mesma (cor|forma)|atribut"},
    {"id": "calculo_medida", "rotulo": "cálculo de medida ou valor",
     "habilita": r"calcul|medir|\bmedidas?\b|determinar|perimetro|\barea\b|volume|valor numerico",
     "palavras_chave": r"calcul|quanto mede|qual (e|sera|foi) (o|a) (valor|resultado|medida|area|perimetro|volume|total)|resultado"},
    {"id": "resolucao_problema", "rotulo": "resolução de problema",
     "habilita": r"resolver|problema",
     "palavras_chave": r"quant[oa]s?\b"},
    {"id": "problema_inverso", "rotulo": "problema inverso (descobrir o dado inicial)",
     "habilita": r"resolver (e elaborar )?problemas|problemas que (envolv|possam)|equac",
     "palavras_chave": r"(original|inicial|antes|no inicio)\b|qual (era|foi) o (preco|valor|numero)|descubr"},
    {"id": "multiplas_etapas", "rotulo": "problema de múltiplas etapas",
     "habilita": r"problemas|sucessiv|sistema",
     "palavras_chave": r"(depois|em seguida|restante|sobrou|ainda).{0,120}(depois|em seguida|restante|sobrou|total|ainda)"},
    {"id": "comparacao", "rotulo": "comparação/ordenação",
     "habilita": r"compar|ordenar|\bmaior|\bmenor",
     "palavras_chave": r"\bmaior\b|\bmenor\b|mais .{0,20}(que|do que)|compar|ordem (crescente|decrescente)"},
    {"id": "conversao_representacao", "rotulo": "conversão entre representações",
     "habilita": r"convert|conversao|associar|relacionar|representa|escrever",
     "palavras_chave": r"equivale|correspond|convert|represent|por extenso|forma (decimal|fracionaria|percentual)|planifica"},
    {"id": "estimativa", "rotulo": "estimativa/aproximação",
     "habilita": r"estim|aproxim",
     "palavras_chave": r"aproximad|estim|cerca de|mais proximo"},
    {"id": "inferencia_padrao", "rotulo": "inferência de padrão",
     "habilita": r"inferir|padrao|regularidade|sequencia",
     "palavras_chave": r"padrao|proximo (numero|termo)|sequencia|regra|segue"},
    {"id": "verificacao_propriedade", "rotulo": "verificação de propriedade",
     "habilita": r"propriedade|condicao|relac(ao|oes) (de|entre)",
     "palavras_chave": r"possivel|verdadeir|corret[ao]|afirma|sempre|propriedade"},
    {"id": "localizacao_trajeto", "rotulo": "localização e trajeto",
     "habilita": r"localiza|desloca",
     "palavras_chave": r"direita|esquerda|caminho|localiz|\bonde\b"},
    {"id": "leitura_dados", "rotulo": "leitura e interpretação de dados",
     "habilita": r"dados|tabela|grafico",
     "palavras_chave": r"tabela|grafico|dados|pesquisa"},
]

# Objeto matemático principal (informativo; não guia o planejador).
OBJETOS = [
    ("triangulo", r"triangul"), ("quadrilatero", r"quadrilater|quadrado|(?<!triangulo )retangulo|losango|trapezio|paralelogramo"),
    ("circulo", r"circul|circunferen"), ("poligono", r"poligon|pentagon|hexagon"),
    ("solido", r"prisma|cilindr|piramide|\bcone|esfera|\bcubo|paralelepiped"),
    ("relogio", r"relogio|ponteiro|\bhoras?\b"), ("calendario", r"calendario|\bmes\b|semana"),
    ("tabela", r"tabela"), ("grafico", r"grafico"), ("dinheiro", r"moeda|cedula|reais|r\$"),
    ("porcentagem", r"%|porcent"), ("fracao", r"fraca|\d+/\d+"), ("decimal", r"\d,\d"),
    ("equacao", r"equac|incognita"), ("sequencia", r"sequencia"), ("numero", r"\d"),
]

# --------------------------------------------------------------------------
# Normalização e texto
# --------------------------------------------------------------------------

def normalizar_texto(texto):
    """Minúsculas, sem acentos, espaços colapsados. Mantém pontuação e dígitos
    (as regex usam %, +, /, dígitos). 'º' vira 'o' via NFKD."""
    texto = unicodedata.normalize("NFKD", str(texto or ""))
    texto = texto.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", texto.lower()).strip()


def mascarar_numeros(texto):
    """Troca todo número por '#': mudar só números não gera questão nova."""
    return re.sub(r"\d+(?:[.,]\d+)*", "#", normalizar_texto(texto))


def texto_questao(q, com_alternativas=True):
    """Aceita str, dict de questão ou wrapper {"questoes":[q]} e devolve o texto."""
    if isinstance(q, str):
        return q
    if not isinstance(q, dict):
        return ""
    if "questoes" in q and isinstance(q["questoes"], list) and q["questoes"]:
        q = q["questoes"][0]
    partes = [q.get("enunciado", "") or q.get("enunciado_item", "")]
    if com_alternativas:
        alts = q.get("alternativas", {})
        if isinstance(alts, dict):
            partes.extend(str(v) for v in alts.values())
        elif isinstance(alts, list):
            partes.extend(str(a.get("texto", a)) if isinstance(a, dict) else str(a) for a in alts)
    return " ".join(p for p in partes if p)


def _enunciado(q):
    return texto_questao(q, com_alternativas=False)


# --------------------------------------------------------------------------
# Taxonomia
# --------------------------------------------------------------------------
_CACHE = {}


def carregar_taxonomia(path=None):
    """Lê a taxonomia gerada (cache por caminho)."""
    path = Path(path or TAXONOMIA_PATH)
    if path not in _CACHE:
        _CACHE[path] = json.loads(path.read_text(encoding="utf-8"))
    return _CACHE[path]


def obter_habilidade(ano, habilidade, taxonomia=None):
    """Entrada da taxonomia para (ano, habilidade), ou None se não existir."""
    tax = taxonomia or carregar_taxonomia()
    return tax["habilidades"].get(f"{ano}|{habilidade}")


def _conta(padrao, texto):
    return len(re.findall(padrao, texto)) if padrao else 0


def _melhor(opcoes, texto, chave_regex, padrao_vazio):
    """Escolhe a opção com mais ocorrências de palavras-chave (empate: ordem da lista)."""
    melhor, melhor_n = padrao_vazio, 0
    for op in opcoes:
        regs = op[chave_regex]
        regs = regs if isinstance(regs, list) else [regs]
        n = sum(_conta(r, texto) for r in regs)
        if n > melhor_n:
            melhor, melhor_n = op["id"], n
    return melhor


# --------------------------------------------------------------------------
# Classificador
# --------------------------------------------------------------------------

def classificar_questao(questao, ano, habilidade, taxonomia=None):
    """Classifica uma questão (dict, wrapper ou texto) de forma determinística.

    Retorna {subtema, tipo_raciocinio, objeto_matematico, contexto, estrutura,
    representacao}. Sem casamento: subtema 'outros', tipo/objeto 'indefinido',
    contexto 'sem_contexto'.
    """
    bruto = texto_questao(questao)
    t = normalizar_texto(bruto)
    hab = obter_habilidade(ano, habilidade, taxonomia)
    subtemas = hab["subtemas"] if hab else []
    tipos_ids = hab["tipos_raciocinio"] if hab else [x["id"] for x in TIPOS_RACIOCINIO]
    tipos = [x for x in TIPOS_RACIOCINIO if x["id"] in tipos_ids]

    subtema = _melhor(subtemas, t, "palavras_chave", "outros")
    tipo = _melhor(tipos, t, "palavras_chave", "indefinido")
    contexto = _melhor(CONTEXTOS, t, "palavras_chave", SEM_CONTEXTO)

    objeto = "indefinido"
    for oid, rx in OBJETOS:
        if re.search(rx, t):
            objeto = oid
            break

    en = normalizar_texto(_enunciado(questao)) or t
    estrutura = "pergunta_direta"
    for e in ESTRUTURAS:
        if e["id"] == "problema_contextualizado":
            if contexto != SEM_CONTEXTO and len(en.split()) >= 15:
                estrutura = e["id"]
                break
        elif e["palavras_chave"] and re.search(e["palavras_chave"], en):
            estrutura = e["id"]
            break

    if re.search(r"\|.*\||tabela|\n.*\d.*\n.*\d", str(bruto).lower()):
        representacao = "tabela_textual"
    elif re.search(r"\d\s*(km|cm|mm|m|kg|g|l|ml|m2|cm2|m3|cm3|h|min)\b|\d\s*(°|graus)", t):
        representacao = "medidas"
    elif len(re.findall(r"\d+", t)) >= 2:
        representacao = "numerica"
    else:
        representacao = "verbal"

    return {"subtema": subtema, "tipo_raciocinio": tipo, "objeto_matematico": objeto,
            "contexto": contexto, "estrutura": estrutura, "representacao": representacao}


# --------------------------------------------------------------------------
# Planejador de cobertura
# --------------------------------------------------------------------------

def _rotulo(lista, id_):
    for x in lista:
        if x["id"] == id_:
            return x["rotulo"]
    return id_


def rotulo_contexto(contexto_id, taxonomia=None):
    """Rótulo em português de um id de contexto (função pública para quem
    monta slot fora deste módulo, ex.: gerar_lote ao trocar o contexto de
    um slot numa regeneração)."""
    return _rotulo((taxonomia or {}).get("contextos", CONTEXTOS) if taxonomia else CONTEXTOS, contexto_id)


def _contagem_historico(historico, ano, habilidade, taxonomia):
    """historico: lista de classificações (dict com 'subtema'), questões ou textos."""
    sub, ctx = Counter(), Counter()
    for h in historico or []:
        c = h if isinstance(h, dict) and "subtema" in h else classificar_questao(h, ano, habilidade, taxonomia)
        sub[c["subtema"]] += 1
        ctx[c.get("contexto", SEM_CONTEXTO)] += 1
    return sub, ctx


def planejar_lote(ano, habilidade, quantidade, dificuldade=None, seed=0, historico=None, taxonomia=None):
    """Plano de `quantidade` slots {subtema, tipo_raciocinio, contexto, estrutura}
    (+ rótulos, dificuldade, indice). Determinístico dado (seed, ano, habilidade,
    dificuldade, historico).

    Regras: round-robin sobre os K subtemas numa ordem ponderada (menos usados
    no histórico primeiro; empate desfeito pela seed). Isso garante min(N, K)
    subtemas distintos e no máximo ceil(N/K) por subtema — cobre N=1,2,3,5,10,20
    e é adaptativo a K. Com K=1 o subtema se repete, mas raciocínio, contexto e
    estrutura continuam girando (contexto e números não são a única variação).
    """
    tax = taxonomia or carregar_taxonomia()
    hab = obter_habilidade(ano, habilidade, tax)
    if hab is None:
        raise KeyError(f"(ano, habilidade) sem taxonomia: {ano} {habilidade}")
    n = int(quantidade)
    if n <= 0:
        return []
    # str como seed é determinístico no random do Python (sha512), ao contrário de hash()
    rng = random.Random(f"{seed}|{ano}|{habilidade}|{dificuldade}")
    hist_sub, hist_ctx = _contagem_historico(historico, ano, habilidade, tax)

    subtemas = list(hab["subtemas"])
    desempate = {s["id"]: rng.random() for s in subtemas}
    ordem = sorted(subtemas, key=lambda s: (hist_sub.get(s["id"], 0), desempate[s["id"]]))
    k = len(ordem)

    ctx_ids = [c["id"] for c in tax.get("contextos", CONTEXTOS)]
    ctx_desemp = {c: rng.random() for c in ctx_ids}
    ctx_ordem = sorted(ctx_ids, key=lambda c: (hist_ctx.get(c, 0), ctx_desemp[c]))
    est_ids = [e["id"] for e in tax.get("estruturas", ESTRUTURAS)]
    rng.shuffle(est_ids)
    tipos_off = rng.randrange(1000)

    slots, ocorr = [], Counter()
    for i in range(n):
        pos = i % k
        s = ordem[pos]
        tipos = s.get("tipos_raciocinio") or hab["tipos_raciocinio"]
        # desloca pela posição do subtema: subtemas diferentes começam em tipos diferentes
        tipo = tipos[(ocorr[s["id"]] + pos + tipos_off) % len(tipos)]
        ocorr[s["id"]] += 1
        ctx = ctx_ordem[i % len(ctx_ordem)]
        est = est_ids[i % len(est_ids)]
        slots.append({
            "indice": i, "subtema": s["id"], "subtema_rotulo": s["rotulo"],
            "tipo_raciocinio": tipo, "tipo_raciocinio_rotulo": _rotulo(TIPOS_RACIOCINIO, tipo),
            "contexto": ctx, "contexto_rotulo": _rotulo(CONTEXTOS, ctx),
            "estrutura": est, "estrutura_rotulo": _rotulo(ESTRUTURAS, est),
            "dificuldade": dificuldade,
        })
    return slots


# Subtemas cujo objeto matemático É a representação de dados (tabela, gráfico,
# histograma, lista). O app é texto puro: sem imagem, a questão só é resolvível
# se os dados vierem ESCRITOS no enunciado. Decidido por família de subtema,
# não por habilidade — vale para 2º/5º/9º e para qualquer habilidade nova.
_PREFIXOS_DADOS = ("tabela_", "grafico_", "histograma", "listas")

INSTRUCAO_DADOS_TEXTUAIS = (
    " Dados: escreva no enunciado todos os dados da tabela ou do gráfico em texto,"
    " um item por linha no formato 'rótulo: valor'."
)


def exige_dados_textuais(subtema):
    """True se o subtema é uma representação de dados (tabela/gráfico/...)."""
    return bool(subtema) and str(subtema).startswith(_PREFIXOS_DADOS)


def sufixo_prompt(slot):
    """Trecho ÚNICO anexado ao USER_TEMPLATE, igual em treino, destilação e
    inferência: ' Subtema: X. Tipo de raciocínio: Y. Contexto: Z.' — e, para
    subtemas de dados, a instrução fixa INSTRUCAO_DADOS_TEXTUAIS."""
    suf = (f" Subtema: {slot['subtema_rotulo']}. Tipo de raciocínio: {slot['tipo_raciocinio_rotulo']}."
           f" Contexto: {slot['contexto_rotulo']}.")
    if exige_dados_textuais(slot.get("subtema")):
        suf += INSTRUCAO_DADOS_TEXTUAIS
    return suf


# "Segunda: 12", "Azul - 15 votos", "Maçã = 8"; ou tabela com pipes.
_PAR_ROTULO_VALOR = re.compile(r"[A-Za-zÀ-ÿ][\wÀ-ÿ ]{0,30}?\s*[:=\-–]\s*\d+(?:[.,]\d+)?")
_ARTEFATO_CITADO = re.compile(r"\b(tabela|gr[áa]fico|histograma|quadro)s?\b", re.I)


def tem_dados_textuais(questao, minimo=3):
    """True se o enunciado traz os dados escritos: >= `minimo` pares
    'rótulo: valor' ou uma tabela com pipes. Números soltos numa frase
    ('12 azuis e 6 vermelhos') não contam como TABELA — mas contam como
    dados; por isso também aceita >= `minimo` números no enunciado (cobre
    '12 alunos gostam de X; 10 de Y; 8 de Z', formato valor-rótulo)."""
    en = _enunciado(questao) if isinstance(questao, dict) else str(questao or "")
    if en.count("|") >= 4:
        return True
    if len(_PAR_ROTULO_VALOR.findall(en)) >= minimo:
        return True
    return len(re.findall(r"\d+(?:[.,]\d+)?", en)) >= minimo


def dados_ausentes(questao, subtema=None):
    """True se a questão depende de uma tabela/gráfico que não está escrito.

    Casos: (a) o slot é de subtema de dados e o enunciado não traz os dados;
    (b) qualquer slot, o enunciado cita tabela/gráfico ('No gráfico, qual
    cor...') e não traz os dados. Complementa
    schema_utils.depende_de_visual_ausente, que só pega a referência dêitica
    explícita ('observe o gráfico abaixo')."""
    if tem_dados_textuais(questao):
        return False
    en = _enunciado(questao) if isinstance(questao, dict) else str(questao or "")
    return exige_dados_textuais(subtema) or bool(_ARTEFATO_CITADO.search(en))


# --------------------------------------------------------------------------
# Similaridade e métricas
# --------------------------------------------------------------------------
_STOP = set("a o as os de da do das dos e em um uma com por para que se no na nos nas ao aos "
            "qual quais sua seu ele ela eles elas foi sao ser tem".split())


def _tokens(texto):
    return [w for w in re.findall(r"[a-z#]+", mascarar_numeros(texto)) if w not in _STOP]


def shingles(texto, k=3):
    """Conjunto de k-gramas de palavras (números mascarados)."""
    toks = re.findall(r"[a-z#]+", mascarar_numeros(texto))
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def cosseno_tf(t1, t2):
    """Cosseno entre vetores de frequência de tokens (números mascarados)."""
    c1, c2 = Counter(_tokens(t1)), Counter(_tokens(t2))
    num = sum(c1[w] * c2[w] for w in c1.keys() & c2.keys())
    den = math.sqrt(sum(v * v for v in c1.values())) * math.sqrt(sum(v * v for v in c2.values()))
    return num / den if den else 0.0


def e_near_duplicata(q1, q2, limiar=LIMIAR_NEAR_DUP):
    """True se os enunciados são iguais após normalizar/mascarar números ou se o
    Jaccard de trigramas de palavras >= limiar."""
    e1, e2 = _enunciado(q1), _enunciado(q2)
    m1 = re.sub(r"[^a-z#]", "", mascarar_numeros(e1))
    m2 = re.sub(r"[^a-z#]", "", mascarar_numeros(e2))
    if m1 and m1 == m2:
        return True
    return jaccard(shingles(e1), shingles(e2)) >= limiar


def subtema_distribution(classificacoes):
    """{subtema: contagem} a partir de classificações (ou lista de ids)."""
    return dict(Counter(c["subtema"] if isinstance(c, dict) else c for c in classificacoes))


def coverage_score(subtemas, k):
    """distintos / min(N, K). 1.0 = cobertura máxima possível para o tamanho do lote."""
    subtemas = [s["subtema"] if isinstance(s, dict) else s for s in subtemas]
    n = len(subtemas)
    if n == 0 or k <= 0:
        return 0.0
    distintos = len({s for s in subtemas if s != "outros"})
    return min(1.0, distintos / min(n, k))


def duplicate_rate(questoes, limiar=LIMIAR_NEAR_DUP):
    """Fração de questões que são (near-)duplicata de alguma anterior no lote."""
    if not questoes:
        return 0.0
    dup = sum(1 for i in range(1, len(questoes))
              if any(e_near_duplicata(questoes[i], questoes[j], limiar) for j in range(i)))
    return dup / len(questoes)


def semantic_similarity(questoes):
    """Similaridade entre pares (cosseno TF e Jaccard de trigramas, números
    mascarados): {'cosseno_media','cosseno_max','jaccard_media','jaccard_max'}."""
    textos = [_enunciado(q) for q in questoes]
    cos, jac = [], []
    for i in range(len(textos)):
        for j in range(i + 1, len(textos)):
            cos.append(cosseno_tf(textos[i], textos[j]))
            jac.append(jaccard(shingles(textos[i]), shingles(textos[j])))
    if not cos:
        return {"cosseno_media": 0.0, "cosseno_max": 0.0, "jaccard_media": 0.0, "jaccard_max": 0.0}
    return {"cosseno_media": sum(cos) / len(cos), "cosseno_max": max(cos),
            "jaccard_media": sum(jac) / len(jac), "jaccard_max": max(jac)}


def _razao_distintos(valores, universo):
    n = len(valores)
    if n == 0:
        return 0.0
    return min(1.0, len(set(valores)) / min(n, max(universo, 1)))


def structural_diversity(classificacoes):
    """Média de distintos/min(N, universo) para estrutura e representação."""
    est = [c["estrutura"] for c in classificacoes]
    rep = [c["representacao"] for c in classificacoes]
    return (_razao_distintos(est, len(ESTRUTURAS)) + _razao_distintos(rep, 4)) / 2


def context_diversity(classificacoes):
    """Entropia normalizada dos contextos (sem_contexto conta como uma categoria).
    Lote de 1 questão = 1.0 (nada a repetir)."""
    ctx = [c["contexto"] for c in classificacoes]
    n = len(ctx)
    if n <= 1:
        return 1.0 if n == 1 else 0.0
    cont = Counter(ctx)
    h = -sum((v / n) * math.log(v / n) for v in cont.values())
    return h / math.log(min(n, len(CONTEXTOS) + 1))


def diversity_score(questoes, ano, habilidade, pesos=None, taxonomia=None, limiar=LIMIAR_NEAR_DUP):
    """Métricas decomponíveis de um lote + score ponderado (pesos expostos).

    Retorna dict com: n, k_subtemas, subtema_distribution, coverage_score,
    duplicate_rate, semantic_similarity, structural_diversity,
    context_diversity, raciocinio_diversity, pesos, componentes, diversity_score.
    """
    pesos = dict(pesos or PESOS_DIVERSIDADE)
    hab = obter_habilidade(ano, habilidade, taxonomia)
    k = len(hab["subtemas"]) if hab else 1
    cls = [classificar_questao(q, ano, habilidade, taxonomia) for q in questoes]
    sim = semantic_similarity(questoes)
    n_tipos = len(hab["tipos_raciocinio"]) if hab else len(TIPOS_RACIOCINIO)
    comp = {
        "coverage_score": coverage_score(cls, k),
        "nao_duplicata": 1.0 - duplicate_rate(questoes, limiar),
        "dissimilaridade": 1.0 - sim["cosseno_media"],
        "raciocinio_diversity": _razao_distintos([c["tipo_raciocinio"] for c in cls], n_tipos),
        "structural_diversity": structural_diversity(cls),
        "context_diversity": context_diversity(cls),
    }
    total = sum(pesos.values()) or 1.0
    score = sum(pesos.get(nome, 0.0) * v for nome, v in comp.items()) / total
    return {
        "n": len(questoes), "k_subtemas": k,
        "subtema_distribution": subtema_distribution(cls),
        "coverage_score": comp["coverage_score"],
        "duplicate_rate": 1.0 - comp["nao_duplicata"],
        "semantic_similarity": sim,
        "structural_diversity": comp["structural_diversity"],
        "context_diversity": comp["context_diversity"],
        "raciocinio_diversity": comp["raciocinio_diversity"],
        "pesos": pesos, "componentes": comp, "diversity_score": score,
        "classificacoes": cls,
    }


# --------------------------------------------------------------------------
# Regeneração guiada
# --------------------------------------------------------------------------

def violacoes_diversidade(lote_aceito, candidato, slot, ano, habilidade, quantidade=None,
                          limiar=LIMIAR_NEAR_DUP, taxonomia=None):
    """Lista de violações do candidato frente ao lote aceito e ao slot planejado.

    Tipos: 'subtema_fora_do_plano' (candidato caiu em OUTRO subtema conhecido e
    havia alternativa, K>=2), 'near_duplicata', 'subtema_saturado' (já há
    ceil(N/K) questões daquele subtema), 'contexto_repetido'. Candidato
    classificado como 'outros' não gera violação de subtema (o classificador é
    léxico; na dúvida não se pune).
    """
    hab = obter_habilidade(ano, habilidade, taxonomia)
    k = len(hab["subtemas"]) if hab else 1
    n = quantidade or (len(lote_aceito) + 1)
    c = classificar_questao(candidato, ano, habilidade, taxonomia)
    aceitos_cls = [classificar_questao(q, ano, habilidade, taxonomia) for q in lote_aceito]
    rot = {s["id"]: s["rotulo"] for s in (hab["subtemas"] if hab else [])}
    v = []
    if k >= 2 and c["subtema"] not in ("outros", slot.get("subtema")):
        v.append({"tipo": "subtema_fora_do_plano", "esperado": slot.get("subtema"),
                  "obtido": c["subtema"], "obtido_rotulo": rot.get(c["subtema"], c["subtema"])})
    for i, q in enumerate(lote_aceito):
        if e_near_duplicata(q, candidato, limiar):
            v.append({"tipo": "near_duplicata", "indice_aceito": i})
            break
    if k >= 2 and c["subtema"] != "outros":
        ja = sum(1 for a in aceitos_cls if a["subtema"] == c["subtema"])
        if ja >= math.ceil(n / k):
            v.append({"tipo": "subtema_saturado", "subtema": c["subtema"],
                      "subtema_rotulo": rot.get(c["subtema"], c["subtema"]), "quantidade": ja})
    if dados_ausentes(candidato, slot.get("subtema")):
        v.append({"tipo": "dados_ausentes"})
    ctx_pool = [x["id"] for x in (taxonomia or {}).get("contextos", CONTEXTOS)] or [x["id"] for x in CONTEXTOS]
    if c["contexto"] != SEM_CONTEXTO and len(lote_aceito) < len(ctx_pool) \
            and any(a["contexto"] == c["contexto"] for a in aceitos_cls):
        usados = {a["contexto"] for a in aceitos_cls} | {c["contexto"]}
        # Contexto ainda não usado no lote; se o próprio slot planejado já é o
        # repetido (o modelo seguiu o prompt e AINDA ASSIM colidiu com outro
        # aceito), a sugestão não pode ser o mesmo — senão a restrição vira
        # "use X" logo após "Contexto: X" no mesmo prompt, uma instrução
        # contraditória que o modelo não tem como seguir.
        livres = [cid for cid in ctx_pool if cid not in usados] or \
                 [cid for cid in ctx_pool if cid != c["contexto"]]
        sugerido = livres[0] if livres else c["contexto"]
        v.append({"tipo": "contexto_repetido", "contexto": c["contexto"],
                  "contexto_rotulo": _rotulo(CONTEXTOS, c["contexto"]),
                  "sugerido": sugerido, "sugerido_rotulo": _rotulo(CONTEXTOS, sugerido)})
    return v


def montar_restricao(violacoes, habilidade, slot):
    """Texto de restrição (português) a anexar ao prompt da regeneração."""
    frases = []
    for v in violacoes:
        if v["tipo"] == "subtema_saturado":
            frases.append(f"O lote já contém {v['quantidade']} questão(ões) sobre {v['subtema_rotulo']}. "
                          f"Gere uma nova questão utilizando outro subtema compatível com a habilidade "
                          f"{habilidade}: {slot['subtema_rotulo']}.")
        elif v["tipo"] == "subtema_fora_do_plano":
            frases.append(f"A questão deve tratar de {slot['subtema_rotulo']}, não de {v['obtido_rotulo']}.")
        elif v["tipo"] == "near_duplicata":
            frases.append("A questão ficou quase idêntica a uma já gerada; trocar apenas os números não "
                          "conta como questão nova. Mude a situação e o que é perguntado.")
        elif v["tipo"] == "dados_ausentes":
            frases.append("A questão cita uma tabela ou gráfico que o aluno não vê. Escreva no "
                          "enunciado todos os dados em texto, um item por linha no formato "
                          "'rótulo: valor'.")
        elif v["tipo"] == "contexto_repetido":
            frases.append(f"O contexto '{v['contexto_rotulo']}' já foi usado no lote; "
                          f"use o contexto: {v.get('sugerido_rotulo', slot['contexto_rotulo'])}.")
    # dedup preservando ordem (subtema saturado e fora do plano podem repetir a ideia)
    return " ".join(dict.fromkeys(frases))


def gerar_lote_diverso(gerar_fn, ano, habilidade, quantidade, dificuldade=None, seed=0,
                       max_tentativas=MAX_TENTATIVAS_DIVERSIDADE, historico=None, taxonomia=None):
    """Orquestra plano + geração + regeneração com restrição.

    gerar_fn(slot, restricao, tentativa) -> questão (dict) ou None. `restricao`
    é None na 1ª tentativa. Após `max_tentativas` regenerações, fica o candidato
    com menos violações (nunca descarta o slot por diversidade — as validações
    de schema_utils continuam sendo responsabilidade de gerar_fn).
    Retorna (aceitas, relatorio_por_slot).
    """
    plano = planejar_lote(ano, habilidade, quantidade, dificuldade, seed, historico, taxonomia)
    aceitas, relatorio = [], []
    for slot in plano:
        melhor, melhor_v, restricao, tent = None, None, None, 0
        for tent in range(max_tentativas + 1):
            cand = gerar_fn(slot, restricao, tent)
            if cand is None:
                continue
            v = violacoes_diversidade(aceitas, cand, slot, ano, habilidade, quantidade, taxonomia=taxonomia)
            if melhor is None or len(v) < len(melhor_v):
                melhor, melhor_v = cand, v
            if not v:
                break
            restricao = montar_restricao(v, habilidade, slot)
        if melhor is not None:
            aceitas.append(melhor)
        relatorio.append({"slot": slot, "tentativas": tent + 1,
                          "violacoes": [x["tipo"] for x in (melhor_v or [])]})
    return aceitas, relatorio
