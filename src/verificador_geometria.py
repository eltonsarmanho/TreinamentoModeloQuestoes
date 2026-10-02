"""Verificador simbólico de geometria: classificação de triângulos e quadriláteros.

POR QUE EXISTE
--------------
Teste real do usuário (2026-10-01): 20 questões de 9º H17 ("Classificar
triângulos ou quadriláteros em relação aos lados ou aos ângulos internos"),
Fácil, geradas pelo modelo promovido em modo planejado. A auditoria humana
(outputs/testes_locais/Log.txt) aprovou só 5 das 20, e o verificador aritmético
de schema_utils devolveu "sem_conta" nas 20: a resolução de uma questão de
classificação não tem conta "a op b = r", então 283 das 311 questões de 9º H17
do corpus eram invisíveis para ele. Os erros encontrados caem em três classes:
  (a) PREMISSA IMPOSSÍVEL: triângulo com dois ângulos retos (R1-Q5) ou dois
      obtusos (R1-Q9), quadrilátero com os 4 ângulos obtusos (R1-Q4), lados
      5, 5, 10 degenerados (R2-Q4);
  (b) RESPOSTA NÃO ÚNICA por hierarquia: 4 ângulos retos com gabarito
      "Quadrado" e "Retângulo" na lista (R2-Q2, R2-Q10); 90-45-45 com
      "Retângulo" e "Isósceles" (R2-Q3); "Quadrilátero com todos os lados
      iguais" e "Losango" como alternativas distintas (R1-Q10);
  (c) DADOS AUSENTES ou CONTA ERRADA: "passa por quatro pontos formando um
      quadrilátero; classifique quanto aos lados" (R1-Q2); lados 5, 7, 9 com
      gabarito "Acutângulo", mas 25 + 49 < 81 é obtusângulo (R1-Q3).
O padrão que a auditoria nomeou é "tem uma propriedade de X -> é X": o modelo
ignora que as classificações são HIERÁRQUICAS e que lados e ângulos são eixos
distintos. Este módulo checa exatamente isso, de forma determinística, sem LLM
e sem embeddings.

COMO FUNCIONA
-------------
Existe um universo FINITO e exato de 41 "estados abstratos" (UNIVERSO): 7 tipos
de triângulo (partição das igualdades de lados x tipo do maior ângulo) e 34
tipos de quadrilátero convexo (pares de lados paralelos, partição dos lados,
pares opostos iguais, pipa, contagem de ângulos retos/obtusos/agudos e, nos
trapézios, se as pernas são iguais). Cada estado guarda uma TESTEMUNHA com
aritmética inteira, e tests/test_verificador_geometria.py recalcula as
características a partir dela: prova que todo estado existe. A completude (não
falta estado) foi checada fora do repositório por enumeração exaustiva de
quadriláteros convexos no reticulado inteiro 29x29 (~1,4e10 casos, 34 estados),
por amostragem contínua e por derivação teórica (não existem: trapézio com 4
lados iguais, quadrilátero com exatamente 3 ângulos retos, quadrilátero com
três lados iguais e dois ângulos retos sem ser quadrado, etc.).

Cada rótulo ("quadrado", "isósceles"...) e cada fato extraído do texto ("dois
ângulos obtusos", "lados 5, 7, 9") é um SUBCONJUNTO do universo. Com S = a
interseção dos fatos, uma alternativa A é
    V (garantida)       se S ⊆ A,
    F (impossível)      se S ∩ A = ∅,
    I (indeterminada)   nos outros casos;
e S = ∅ é premissa impossível. A hierarquia sai de graça, porque é inclusão de
conjuntos (quadrado ⊂ retângulo). Fatos que o parser NÃO extraiu só poderiam
encolher S, então V e F são robustos a parse incompleto; vereditos que dependem
de I (dados insuficientes, "nenhuma das anteriores" verdadeira) exigem que o
texto tenha sido inteiramente compreendido (`completo`).

CONVENÇÕES (decididas aqui; contestáveis, por isso explícitas)
------------------------------------------------------------
 1. Hierarquia INCLUSIVA dos quadriláteros: quadrado ⊂ retângulo ⊂
    paralelogramo e quadrado ⊂ losango ⊂ paralelogramo. É o que a BNCC pede no
    EF06MA20 ("classificá-los em relação a lados e a ângulos e reconhecer a
    inclusão e a intersecção de classes entre eles") e o que a auditoria usou:
    "um quadrado é um tipo particular de retângulo"; "um quadrilátero com
    quatro lados iguais é um losango" (R2-Q4), com "Quadrado" também na lista.
 2. Trapézio EXCLUSIVO: exatamente um par de lados paralelos, como na maioria
    dos livros brasileiros ("apenas um par"). Paralelogramo não é trapézio.
 3. Isósceles: os livros divergem ("pelo menos dois lados iguais" x "apenas
    dois"). Em vez de escolher, o módulo avalia as DUAS LEITURAS (A inclusiva,
    B exclusiva) e só emite veredito quando elas concordam ou ambas rejeitam.
    A mesma dupla leitura cobre os quantificadores: "dois ângulos obtusos" é
    "pelo menos dois" na leitura A e "exatamente dois" na B ("apenas",
    "exatamente" e "pelo menos" fixam a leitura), e a ordem das medidas de um
    quadrilátero é cíclica (lado1, lado2, ... consecutivos) só na leitura A.
    Também "trapézio escaleno" (inclui ou não o retângulo) e "pipa" (inclui ou
    não o losango) têm uma leitura inclusiva e uma exclusiva.
 4. Resposta MAIS ESPECÍFICA dentro do MESMO EIXO: "Isósceles" verdadeiro não
    invalida o gabarito "Equilátero" (R2-Q5, aprovada pela auditoria), nem
    "Retângulo" invalida "Quadrado" quando ambos são garantidos.
 5. Lados e ângulos de um triângulo são EIXOS INDEPENDENTES: não há dominância
    entre eixos. 90-45-45 com "Retângulo" e "Isósceles" listados é não única
    (R2-Q3); 60-60-60 com "Equilátero" e "Acutângulo" também. Quando a pergunta
    fixa o eixo ("quanto aos lados", "considerando apenas os ângulos"), um
    rótulo do outro eixo vale F.
 6. Só quadriláteros CONVEXOS: ângulo >= 180° num quadrilátero sai do universo
    (nao_aplicavel); num triângulo é premissa impossível.
 7. Figura degenerada é impossível: desigualdade triangular ESTRITA (5, 5, 10 da
    R2-Q4 não é triângulo).
 8. Gabarito que NEGA a existência ("Esse triângulo não existe") com premissa
    impossível é "ok" (condição de existência, 9º H16), mesmo sem "existe" no
    enunciado; gabarito "Nenhuma das alternativas" ou "Sim/Não" com premissa
    impossível fica nao_aplicavel.
 9. "os lados iguais"/"os ângulos iguais" sem quantificador nem verbo de ligação
    é o PAR do isósceles, não "todos"; "em forma de X" é fato da figura só se
    nenhum outro objeto aparecer depois ("pátio em forma de quadrado... um
    quadrilátero pintado").
10. Na dúvida, nao_aplicavel: rótulo inventado ("Sínciclo", "Acrisângulo"),
    lista de nomes ("Quadrados e retângulos"), seleção de entidade ("Canteiro
    A"), negação, medidas parciais, conceito fora do modelo (diagonal, altura),
    pergunta de polaridade invertida ("qual NÃO se aplica?"), figura derivada
    (corpo e pergunta falando de figuras diferentes), medida decimal
    arredondada perto da fronteira (1,41 ≈ √2) e medidas de objetos diferentes
    num texto com várias figuras, texto todo em CAIXA ALTA, número com espaço
    de milhar ("1 000") e medida por extenso ("noventa graus") bloqueiam o
    veredito em vez de virar acusação.

MEDIDO em 2026-10-01 (só leitura em outputs/ e data/)
-----------------------------------------------------
  * As 20 auditadas: 20/20 (as 5 corretas dão "ok"; as 15 erradas reprovam).
    O verificador aritmético dava "sem_conta" nas 20.
  * 9º H17 dos relatórios outputs/diversidade_*.json (311 questões): 167
    decididas (59 ok, 108 reprovadas). O verificador aritmético dava
    "sem_conta" em 283 e só "aprovava" 26 placeholders sem rótulo de
    classificação ("Considere triângulos com medida 1 cm" -> "12"), que aqui
    ficam nao_aplicavel: os dois não se contradizem. Revisei uma a uma as 56
    reprovações ÚNICAS: todas são itens defeituosos pelos critérios da
    auditoria. NÃO é auditoria humana.
  * data/*.jsonl (train, train_curado, train_multi, distill, val*: 9.284
    questões) e os 1.941 itens de outras habilidades em outputs/: ZERO
    reprovações. Os 52 "ok" de data/ são itens corretos.
  * ~0,06 ms por questão.
  ATENÇÃO: as 20 auditadas e o corpus de 311 serviram para ajustar o parser
  (números dentro da amostra). Por isso tests/test_verificador_geometria.py
  trava também 30 questões CORRETAS escritas à mão com fraseado novo
  (TestSemFalsoPositivo); cada uma derrubou uma versão anterior com acusação
  falsa. Antes de ligar o modo "ativo", fazer auditoria humana nova de
  questões geradas depois disto.

PORTABILIDADE (o app é React Native; isto precisa virar TypeScript)
-------------------------------------------------------------------
Nenhuma regex usa lookbehind, grupos nomeados, flags embutidas ou classes
Unicode: o texto é normalizado para ASCII (NFD sem diacríticos) e toda regex
com \\b, \\w ou \\d é compilada com re.ASCII, que lhes dá a semântica do
JavaScript (o teste de portabilidade confere, inclusive os padrões montados em
tempo de execução). Medidas decimais viram inteiros escalados (sem float nas
comparações). As máscaras são conjuntos de índices de 0 a 40 (em TS:
boolean[41] ou dois uint32, sem BigInt). Armadilhas que o Python esconde
(revisão adversarial de 2026-10-01):
  * sorted() de números: Array.prototype.sort() ordena como TEXTO
    ([1000, 800, 600].sort() dá [1000, 600, 800]); use (a, b) => a - b. As
    ordenações de tuplas (_rotulos_no_texto, entidades) precisam de comparador.
  * int ilimitado: as medidas escaladas acima de _TETO_EXATO (2^26) fazem o
    verificador se abster, para que a² + b² - c² caiba exato em Number.
  * re.escape não existe no Hermes: escreva um helper (s.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&")).
  * dígitos de largura total ("３") viram ASCII na normalização (_TABELA).
  * _cortar_sufixo é um laço, não regex: a regex equivalente é quadrática em
    sequências de espaços e o Hermes não tem JIT de regex.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Vereditos públicos
# ---------------------------------------------------------------------------
VEREDITOS = ("ok", "gabarito_errado", "nao_unica", "premissa_impossivel",
             "dados_insuficientes", "nao_aplicavel")

# Vereditos que REPROVAM a questão.
GEO_REJEITA = frozenset({"gabarito_errado", "nao_unica", "premissa_impossivel",
                         "dados_insuficientes"})
# Modo da guarda no pipeline (tests/test_model.generate_validated, gerar_lote,
# avaliar_diversidade). FONTE ÚNICA: test_model reexporta estas constantes. Havia
# duas, contraditórias ("sombra" aqui, que nada lia, e "ativo" em test_model, a
# efetiva) — achado da revisão adversarial de 2026-10-01.
#   "sombra" (padrão) o veredito vai para o relatório; score/status não mudam.
#   "ativo"  reprovação re-amostra (como resposta fora das alternativas).
# Por que "sombra": com o modelo promovido, 15/20 questões de 9º H17 são
# reprovadas; o ativo custa então ~3,3 chamadas por questão de H17 (1,0 em
# sombra) e 17% a 45% dos slots esgotam as 6 chamadas e saem "falha" (G2 exige
# zero). Os 0,5-0,7 chamadas a mais da primeira estimativa vieram de corpora
# antigos (35-42% de rejeição), não do modelo atual. Ligar o ativo (CLI:
# --geometria ativo) exige medir esse custo com o modelo real e auditar à mão
# 30-40 questões novas de H17; baseline e candidato de um gate precisam do
# MESMO modo (os relatórios gravam o modo usado).
MODOS_GEOMETRIA = ("ativo", "sombra")
MODO_GEOMETRIA = "sombra"
# Convenção do gabarito, a MESMA do plano (diversidade.TIPOS_RACIOCINIO, campo
# "instrucao"): hierarquia inclusiva e o gabarito é a classe MAIS ESPECÍFICA
# garantida pelos dados; a superclasse do mesmo eixo pode ser distrator (R2-Q5,
# aprovada pela auditoria; MT9050/MT9081/MT9084 do banco). Não pode haver outra
# classe garantida em outro eixo ou em outro ramo (90-45-45 com "Isósceles" e
# "Retângulo"; "Retângulo" e "Losango" sem "Quadrado").
CONVENCAO_GABARITO = "mais_especifico_garantido"

_A = re.ASCII  # \b, \w, \d e \s com a semântica ASCII do JavaScript

# ===========================================================================
# 1. UNIVERSO DE ESTADOS
# ===========================================================================
# Triângulo: a testemunha são os QUADRADOS dos lados (a², b², c²). Com eles a
# classificação é exata em inteiros (recíproca de Pitágoras) e o equilátero,
# que não tem vértices inteiros, também ganha testemunha.
# Quadrilátero: a testemunha são 4 vértices inteiros em ordem convexa.


def _t(part, r, o, a, testemunha):
    return {"fig": "tri", "part": part, "r": r, "o": o, "a": a, "npar": 0,
            "opp": 0, "pipa": False, "pernas_iguais": None,
            "testemunha": testemunha}


def _q(part, npar, opp, pipa, r, o, a, pernas_iguais, testemunha):
    return {"fig": "quad", "part": part, "r": r, "o": o, "a": a, "npar": npar,
            "opp": opp, "pipa": pipa, "pernas_iguais": pernas_iguais,
            "testemunha": testemunha}


UNIVERSO = (
    # --- 7 triângulos (o equilátero só existe acutângulo)
    _t((3,), 0, 0, 3, (1, 1, 1)),
    _t((2, 1), 0, 0, 3, (25, 25, 36)),
    _t((2, 1), 1, 0, 2, (1, 1, 2)),
    _t((2, 1), 0, 1, 2, (25, 25, 64)),
    _t((1, 1, 1), 0, 0, 3, (16, 25, 36)),
    _t((1, 1, 1), 1, 0, 2, (9, 16, 25)),
    _t((1, 1, 1), 0, 1, 2, (16, 25, 49)),
    # --- 4 paralelogramos (npar=2): quadrado, losango, retângulo, "qualquer"
    _q((4,), 2, 2, True, 4, 0, 0, None, ((0, 0), (1, 0), (1, 1), (0, 1))),
    _q((4,), 2, 2, True, 0, 2, 2, None, ((4, 0), (9, 0), (5, 3), (0, 3))),
    _q((2, 2), 2, 2, False, 4, 0, 0, None, ((1, 0), (2, 0), (2, 2), (1, 2))),
    _q((2, 2), 2, 2, False, 0, 2, 2, None, ((1, 0), (2, 0), (1, 1), (0, 1))),
    # --- 6 trapézios (npar=1). Ângulos ao longo de cada perna são suplementares,
    # então só existem (0 retos, 2 obtusos, 2 agudos) ou (2, 1, 1) = retângulo.
    # O de 3 lados iguais exige coordenadas >= 11; a amostragem o achou.
    _q((3, 1), 1, 1, False, 0, 2, 2, True, ((3, 0), (8, 0), (11, 4), (0, 4))),
    _q((2, 1, 1), 1, 0, False, 2, 1, 1, False, ((1, 0), (2, 0), (2, 1), (0, 1))),
    _q((2, 1, 1), 1, 0, False, 0, 2, 2, False, ((0, 0), (1, 0), (9, 3), (4, 3))),
    _q((2, 1, 1), 1, 1, False, 0, 2, 2, True, ((0, 0), (1, 0), (2, 1), (2, 2))),
    _q((1, 1, 1, 1), 1, 0, False, 2, 1, 1, False, ((2, 0), (3, 0), (3, 2), (0, 2))),
    _q((1, 1, 1, 1), 1, 0, False, 0, 2, 2, False, ((2, 0), (3, 0), (2, 1), (0, 1))),
    # --- 24 quadriláteros sem lados paralelos (npar=0)
    _q((3, 1), 0, 1, False, 1, 1, 2, None, ((2, 0), (3, 0), (2, 2), (0, 1))),
    _q((3, 1), 0, 1, False, 0, 2, 2, None, ((1, 0), (4, 0), (8, 7), (0, 8))),
    _q((3, 1), 0, 1, False, 0, 1, 3, None, ((7, 0), (8, 0), (7, 8), (0, 4))),
    _q((2, 2), 0, 0, True, 2, 1, 1, None, ((5, 0), (10, 0), (10, 15), (1, 3))),
    _q((2, 2), 0, 0, True, 1, 2, 1, None, ((0, 0), (1, 0), (2, 2), (0, 1))),
    _q((2, 2), 0, 0, True, 1, 1, 2, None, ((0, 0), (3, 0), (3, 3), (1, 2))),
    _q((2, 2), 0, 0, True, 0, 3, 1, None, ((0, 0), (15, 0), (16, 8), (9, 12))),
    _q((2, 2), 0, 0, True, 0, 2, 2, None, ((7, 0), (12, 0), (8, 3), (0, 4))),
    _q((2, 2), 0, 0, True, 0, 1, 3, None, ((4, 0), (9, 0), (7, 9), (0, 3))),
    _q((2, 1, 1), 0, 0, False, 2, 1, 1, None, ((2, 0), (3, 0), (3, 3), (1, 2))),
    _q((2, 1, 1), 0, 0, False, 1, 2, 1, None, ((0, 0), (1, 0), (3, 2), (0, 1))),
    _q((2, 1, 1), 0, 0, False, 1, 1, 2, None, ((1, 0), (2, 0), (1, 2), (0, 1))),
    _q((2, 1, 1), 0, 0, False, 0, 3, 1, None, ((1, 0), (2, 0), (5, 5), (0, 2))),
    _q((2, 1, 1), 0, 0, False, 0, 2, 2, None, ((3, 0), (4, 0), (3, 2), (0, 1))),
    _q((2, 1, 1), 0, 0, False, 0, 1, 3, None, ((5, 0), (6, 0), (5, 8), (1, 1))),
    _q((2, 1, 1), 0, 1, False, 1, 1, 2, None, ((4, 0), (5, 0), (5, 5), (0, 3))),
    _q((2, 1, 1), 0, 1, False, 0, 2, 2, None, ((2, 0), (3, 0), (0, 2), (0, 1))),
    _q((2, 1, 1), 0, 1, False, 0, 1, 3, None, ((3, 0), (4, 0), (3, 3), (0, 1))),
    _q((1, 1, 1, 1), 0, 0, False, 2, 1, 1, None, ((1, 0), (2, 0), (2, 3), (0, 1))),
    _q((1, 1, 1, 1), 0, 0, False, 1, 2, 1, None, ((1, 0), (2, 0), (2, 4), (0, 1))),
    _q((1, 1, 1, 1), 0, 0, False, 1, 1, 2, None, ((1, 0), (2, 0), (2, 2), (0, 1))),
    _q((1, 1, 1, 1), 0, 0, False, 0, 3, 1, None, ((1, 0), (2, 0), (3, 5), (0, 1))),
    _q((1, 1, 1, 1), 0, 0, False, 0, 2, 2, None, ((1, 0), (2, 0), (3, 2), (0, 1))),
    _q((1, 1, 1, 1), 0, 0, False, 0, 1, 3, None, ((2, 0), (3, 0), (2, 3), (0, 1))),
)


def _particao(valores):
    """Tamanhos das classes de valores iguais, em ordem decrescente: 5,5,8 -> (2, 1)."""
    contagem = {}
    for v in valores:
        contagem[v] = contagem.get(v, 0) + 1
    return tuple(sorted(contagem.values(), reverse=True))


def caracteristicas_triangulo(lados2):
    """Características exatas do triângulo cujos QUADRADOS dos lados são `lados2`.

    Devolve None se os lados não formam triângulo. Aritmética só inteira:
    √A + √B > √C  <=>  C - A - B < 0  ou  4AB > (C - A - B)².
    """
    A, B, C = sorted(int(x) for x in lados2)
    if A <= 0 or not (C - A - B < 0 or 4 * A * B > (C - A - B) ** 2):
        return None
    if A + B == C:
        r, o, a = 1, 0, 2
    elif A + B < C:
        r, o, a = 0, 1, 2
    else:
        r, o, a = 0, 0, 3
    return {"fig": "tri", "part": _particao(lados2), "r": r, "o": o, "a": a,
            "npar": 0, "opp": 0, "pipa": False, "pernas_iguais": None}


def caracteristicas_quadrilatero(pontos):
    """Características exatas do quadrilátero de vértices inteiros `pontos`, em ordem.

    None se não for ESTRITAMENTE convexo. Produto escalar classifica o ângulo,
    produto vetorial decide paralelismo; nada de float.
    """
    P = [(int(x), int(y)) for x, y in pontos]
    if len(P) != 4:
        return None

    def vetor(i):
        return (P[(i + 1) % 4][0] - P[i][0], P[(i + 1) % 4][1] - P[i][1])

    def cruz(u, v):
        return u[0] * v[1] - u[1] * v[0]

    giros = [cruz(vetor(i), vetor((i + 1) % 4)) for i in range(4)]
    if not (all(g > 0 for g in giros) or all(g < 0 for g in giros)):
        return None
    L = [vetor(i)[0] ** 2 + vetor(i)[1] ** 2 for i in range(4)]
    r = o = a = 0
    for i in range(4):
        u = (P[i - 1][0] - P[i][0], P[i - 1][1] - P[i][1])
        v = vetor(i)
        d = u[0] * v[0] + u[1] * v[1]
        if d == 0:
            r += 1
        elif d > 0:
            a += 1
        else:
            o += 1
    p02 = cruz(vetor(0), vetor(2)) == 0
    p13 = cruz(vetor(1), vetor(3)) == 0
    npar = int(p02) + int(p13)
    pernas = None
    if npar == 1:
        pernas = (L[1] == L[3]) if p02 else (L[0] == L[2])
    return {"fig": "quad", "part": _particao(L), "r": r, "o": o, "a": a,
            "npar": npar, "opp": int(L[0] == L[2]) + int(L[1] == L[3]),
            "pipa": (L[0] == L[1] and L[2] == L[3]) or (L[1] == L[2] and L[3] == L[0]),
            "pernas_iguais": pernas}


def _mascara(pred):
    return frozenset(i for i, e in enumerate(UNIVERSO) if pred(e))


def _nlados(e):
    return 3 if e["fig"] == "tri" else 4


TODOS = _mascara(lambda e: True)
TRI = _mascara(lambda e: e["fig"] == "tri")
QUAD = _mascara(lambda e: e["fig"] == "quad")
FIGURA = {"tri": TRI, "quad": QUAD}
NUM_LADOS = {"tri": 3, "quad": 4}
LEITURAS = ("A", "B")


# ===========================================================================
# 2. RÓTULOS (predicados sobre estados), por leitura
# ===========================================================================
def _rotulos(leitura):
    incl = leitura == "A"
    T = lambda e: e["fig"] == "tri"  # noqa: E731
    Q = lambda e: e["fig"] == "quad"  # noqa: E731
    return {
        "triangulo": TRI,
        "quadrilatero": QUAD,
        "equilatero": _mascara(lambda e: T(e) and e["part"] == (3,)),
        "isosceles": _mascara(lambda e: T(e) and (e["part"] == (2, 1) or (incl and e["part"] == (3,)))),
        "escaleno": _mascara(lambda e: T(e) and e["part"] == (1, 1, 1)),
        "acutangulo": _mascara(lambda e: T(e) and e["a"] == 3),
        "retangulo_t": _mascara(lambda e: T(e) and e["r"] == 1),
        "obtusangulo": _mascara(lambda e: T(e) and e["o"] == 1),
        "quadrado": _mascara(lambda e: Q(e) and e["part"] == (4,) and e["r"] == 4),
        "retangulo_q": _mascara(lambda e: Q(e) and e["r"] == 4),
        "losango": _mascara(lambda e: Q(e) and e["part"] == (4,)),
        "paralelogramo": _mascara(lambda e: Q(e) and e["npar"] == 2),
        "trapezio": _mascara(lambda e: Q(e) and e["npar"] == 1),
        "trapezio_isosceles": _mascara(lambda e: Q(e) and e["npar"] == 1 and e["pernas_iguais"]),
        "trapezio_retangulo": _mascara(lambda e: Q(e) and e["npar"] == 1 and e["r"] == 2),
        "trapezio_escaleno": _mascara(lambda e: Q(e) and e["npar"] == 1 and not e["pernas_iguais"]
                                      and (incl or e["r"] != 2)),
        "trapezoide": _mascara(lambda e: Q(e) and e["npar"] == 0),
        "pipa": _mascara(lambda e: Q(e) and e["pipa"] and (incl or e["part"] != (4,))),
    }


ROTULOS = {L: _rotulos(L) for L in LEITURAS}

# Eixo de cada rótulo. Os de triângulo têm dois eixos independentes; os de
# quadrilátero formam UMA hierarquia só ("quad"); figura pura é "figura".
EIXO = {"equilatero": "lados", "isosceles": "lados", "escaleno": "lados",
        "acutangulo": "angulos", "retangulo_t": "angulos", "obtusangulo": "angulos"}
_FAMILIA = {r: ("tri" if r in EIXO or r == "triangulo" else "quad") for r in ROTULOS["A"]}

NOME_ROTULO = {
    "triangulo": "triângulo", "quadrilatero": "quadrilátero", "equilatero": "equilátero",
    "isosceles": "isósceles", "escaleno": "escaleno", "acutangulo": "acutângulo",
    "retangulo_t": "retângulo (triângulo)", "obtusangulo": "obtusângulo",
    "quadrado": "quadrado", "retangulo_q": "retângulo", "losango": "losango",
    "paralelogramo": "paralelogramo", "trapezio": "trapézio",
    "trapezio_isosceles": "trapézio isósceles", "trapezio_retangulo": "trapézio retângulo",
    "trapezio_escaleno": "trapézio escaleno", "trapezoide": "trapezoide", "pipa": "pipa",
}

# Vocabulário (texto normalizado -> rótulo). A ORDEM importa: compostos antes
# dos simples, para "trapézio retângulo" não virar trapézio + retângulo.
# "isoceles" é grafia errada real (G-9H17-0085 do corpus) e "rombo" é sinônimo
# de losango. "Retângulo" sozinho é ambíguo entre triângulo e quadrilátero e
# só é resolvido pela figura da questão (ver _resolver_retangulo).
VOCAB = (
    (r"\btrapezios? isos?celes\b", "trapezio_isosceles"),
    (r"\btrapezios? retangulos?\b", "trapezio_retangulo"),
    (r"\btrapezios? escalenos?\b", "trapezio_escaleno"),
    (r"\btriangulos? agud[oa]s?\b", "acutangulo"),
    (r"\btriangulos? obtus[oa]s?\b", "obtusangulo"),
    (r"\btriangulos? retangulos?\b", "retangulo_t"),
    (r"\bquadrilateros? retangulos?\b", "retangulo_q"),
    (r"\bequilater[oa]s?\b", "equilatero"),
    (r"\bisos?celes\b", "isosceles"),
    (r"\bescalen[oa]s?\b", "escaleno"),
    (r"\bacutangul[oa]s?\b", "acutangulo"),
    (r"\bobtusangul[oa]s?\b", "obtusangulo"),
    (r"\bquadrad[oa]s?\b", "quadrado"),
    (r"\blosangos?\b|\brombos?\b", "losango"),
    (r"\bparalelogramos?\b", "paralelogramo"),
    (r"\btrapezoides?\b", "trapezoide"),
    (r"\btrapezios?\b", "trapezio"),
    (r"\bpipas?\b|\bdeltoides?\b", "pipa"),
    (r"\bretangul[oa]s?\b|\bretangulares?\b", "RETANGULO"),
    (r"\btriangul[oa]s?\b|\btriangulares?\b", "triangulo"),
    (r"\bquadrilateros?\b", "quadrilatero"),
)
_VOCAB_RE = tuple((re.compile(p, _A), r) for p, r in VOCAB)

# ===========================================================================
# 3. NORMALIZAÇÃO
# ===========================================================================
# 'é' (verbo) vira 'eh' ANTES de tirar os acentos: sem isso 'é' e 'e'
# (conjunção) colapsam e "três tipos: equilátero, isósceles e escaleno" vira o
# fato "é escaleno" (defeito real achado na revisão do protótipo). A classe de
# pontuação é explícita para não depender de \p{L} no TypeScript.
_PONT = r"\s,.;:!?()\[\]\"'/«»“”‘’–—-"
_E_VERBO = re.compile(r"(^|[" + _PONT + r"])([éÉ])(?=[" + _PONT + r"]|$)")
_TABELA = str.maketrans({"º": "°", "˚": "°", "–": "-", "—": "-", "−": "-",
                         "×": "x", "“": '"', "”": '"', "‘": "'", "’": "'",
                         # Dígitos de largura total ("３ cm", teclados asiáticos/IME):
                         # o \d ASCII não os lê, e a questão virava "dados
                         # insuficientes" (12 "ok" mudavam na revisão de 2026-10-01).
                         **{chr(0xFF10 + i): str(i) for i in range(10)}})
# Separador de milhar pt-BR: "1.000 m" é mil metros, não 1,000 (o _NUM lia
# decimal e "1.000, 800 e 600" virava premissa impossível; schema_utils já
# trata "1.000" como milhar). Os pontos saem antes da extração. "1 000" (espaço)
# é ambíguo com lista de números: a questão é recusada (ver _verificar).
_MILHAR_PONTO = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b(?!\.\d)", _A)
_MILHAR_ESPACO = re.compile(r"\b\d{1,3}(?: \d{3})+\b", _A)


def _normalizar(texto):
    """Texto ASCII-izado, caixa preservada, espaços colapsados."""
    t = _E_VERBO.sub(lambda m: m.group(1) + ("Eh" if m.group(2) == "É" else "eh"),
                     str(texto or ""))
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if not ("̀" <= c <= "ͯ"))
    t = t.translate(_TABELA)
    t = _MILHAR_PONTO.sub(lambda m: m.group(0).replace(".", ""), t)
    return re.sub(r"\s+", " ", t).strip()


# Reescritas semânticas aplicadas ao texto em minúsculas, antes da extração:
#  * "cantos"/"quinas" são ângulos ("formas com os cantos retos", train_curado);
#  * "maior que 90°" é OBTUSO e "menor que 90°" é AGUDO. Sem isso o 90 virava
#    medida de ângulo: "todos os ângulos menores que 90°" era lido como um
#    ângulo de 90° (falso "retângulo" no protótipo).
_REESCRITAS = tuple((re.compile(p, _A | re.I), novo) for p, novo in (
    (r"\bcantos\b|\bquinas\b", "angulos"),
    (r"\bcanto\b|\bquina\b", "angulo"),
    (r"\bmaiores (?:que|de|do que) 90 ?(?:°|graus?\b)", "obtusos"),
    (r"\bmaior (?:que|de|do que) 90 ?(?:°|graus?\b)", "obtuso"),
    (r"\bmenores (?:que|de|do que) 90 ?(?:°|graus?\b)", "agudos"),
    (r"\bmenor (?:que|de|do que) 90 ?(?:°|graus?\b)", "agudo"),
    (r"\b(?:iguais|todos iguais) a 90 ?(?:°|graus?\b)", "retos"),
    (r"\bacut(os?)\b", r"agud\1"),
))


def _reescrever(texto_normalizado):
    """Reescritas sem mudar a caixa do resto: o texto com caixa (usado para
    achar "Canteiro A") e o minúsculo continuam alinhados posição a posição."""
    t = texto_normalizado
    for padrao, novo in _REESCRITAS:
        t = padrao.sub(novo, t)
    return t


def _minusculas(texto_normalizado):
    return _reescrever(texto_normalizado).lower()


def _frases(texto):
    """Divide em frases mantendo a pontuação final (split com grupo, sem lookbehind)."""
    partes = re.split(r"([.;?!])\s+", texto)
    frases = []
    for i in range(0, len(partes), 2):
        f = (partes[i] + (partes[i + 1] if i + 1 < len(partes) else "")).strip()
        if f:
            frases.append(f)
    return frases


def _apagar(texto, a, b):
    return texto[:a] + " " * (b - a) + texto[b:]


# ===========================================================================
# 4. EXTRAÇÃO DE FATOS
# ===========================================================================
_NUMW = {"um": 1, "uma": 1, "1": 1, "dois": 2, "duas": 2, "2": 2, "tres": 3, "3": 3,
         "quatro": 4, "4": 4}
_NUM = r"(\d+(?:[.,]\d+)?)"
# Unidade de comprimento; "cm²" (área) e "min" não casam.
_UNID = r"(centimetros?|milimetros?|quilometros?|metros?|mm|cm|dm|km|m)(?![a-z0-9²³])"
_FATOR_MM = {"mm": 1, "milimetro": 1, "milimetros": 1, "cm": 10, "centimetro": 10,
             "centimetros": 10, "dm": 100, "m": 1000, "metro": 1000, "metros": 1000,
             "km": 1000000, "quilometro": 1000000, "quilometros": 1000000}
# "9º ano" vira "9° ano" na normalização e NÃO é ângulo; "30°C" também não.
_GRAU = (r"\s*(?:°(?!\s*[cf]\b)|graus?\b)"
         r"(?!\s*(?:ano|anos|serie|lugar|andar|periodo|semestre|bimestre)\b)")
_TIPO_ANG = r"(retos?|agudos?|obtusos?)"
_QTD = r"(um|uma|dois|duas|tres|quatro|1|2|3|4)"
# "dois a dois"/"2 a 2"/"aos pares" depois de "iguais" é igualdade AOS PARES
# (paralelogramo), não "todos iguais": "lados iguais dois a dois" era lido
# como losango (falso positivo do teste adversarial).
_IGUAIS = (r"(?:iguais|congruentes|de mesm[oa] (?:medida|comprimento|tamanho)"
           r"|(?:com |tem )?(?:a )?mesm[oa] (?:medida|comprimento|tamanho)"
           r"|de medidas iguais|com medidas iguais|de comprimentos iguais)"
           r"(?! dois a dois| 2 a 2| aos pares| em pares)")
_DIFERENTES = (r"(?:diferentes?|distint[oa]s?|desiguais|desigual"
               r"|(?:de|com) (?:medidas|comprimentos|tamanhos) diferentes"
               r"|(?:medidas|comprimentos|tamanhos) diferentes)")
_NEGACAO = re.compile(r"\b(?:nao|nem|nenhum|nenhuma|sem|nunca|jamais)\b", _A)
_NOME_TIPO = {"r": "reto", "o": "obtuso", "a": "agudo"}


def _medida(texto_num):
    """'8,5' -> (85, 1): mantissa inteira e casas decimais (sem float)."""
    t = texto_num.replace(",", ".")
    if "." in t:
        inteiro, dec = t.split(".", 1)
        return int(inteiro + dec), len(dec)
    return int(t), 0


def _escalar(medidas, fatores=None):
    """[(mantissa, casas, ...)] -> inteiros na mesma escala, exatos (sem float).
    `fatores` (opcional) converte unidades diferentes para milímetros."""
    casas = max([m[1] for m in medidas] + [0])
    fatores = fatores or [1] * len(medidas)
    return [m[0] * (10 ** (casas - m[1])) * fat for m, fat in zip(medidas, fatores)]


def _texto_medida(m):
    v = str(m[0])
    if m[1]:
        v = (v[:-m[1]] or "0") + "," + v[-m[1]:]
    return v


class _Fatos:
    """Fatos extraídos de um trecho. Máscaras já calculadas para as duas leituras."""

    def __init__(self, texto):
        self.texto = texto
        # (descrição, máscara_A, máscara_B, figuras_modeladas, eixo)
        self.restricoes = []
        self.lados = []            # [(mantissa, casas, unidade|None)]
        self.lados_todos = None    # "todos os lados medem N" -> (mantissa, casas, unidade)
        self.angulos = []          # [(mantissa, casas)]
        self.angulos_todos = None  # "todos os ângulos medem N°"
        self.resto_cada = None     # (k|None, medida): "os outros ângulos medem N° cada"
        self.consumido = []        # spans consumidos (para a completude)
        self.pendencias = []       # reconhecido mas não modelado -> parse incompleto
        self.negacao = False
        self.frases_medidas = set()  # índices das frases com medida numérica

    def vazio(self):
        return not (self.restricoes or self.lados or self.angulos or self.lados_todos
                    or self.angulos_todos or self.resto_cada)

    def descricoes(self):
        d = [r[0] for r in self.restricoes]
        if self.lados:
            d.append("lados: " + ", ".join(_texto_medida(m) for m in self.lados))
        if self.lados_todos:
            d.append("todos os lados: " + _texto_medida(self.lados_todos))
        if self.angulos:
            d.append("ângulos: " + ", ".join(_texto_medida(m) + "°" for m in self.angulos))
        if self.angulos_todos:
            d.append("todos os ângulos: " + _texto_medida(self.angulos_todos) + "°")
        if self.resto_cada:
            k, m = self.resto_cada
            d.append(f"demais ângulos ({k or 'todos'}): {_texto_medida(m)}° cada")
        return d


def _restricao(descricao, pred_a, pred_b=None, figuras=("tri", "quad"), eixo="lados"):
    """Fato como máscara por leitura. Estados de figura NÃO modelada não são
    cortados (o fato vira pendência quando a figura dela for a da questão)."""
    pred_b = pred_b or pred_a
    livre = lambda e: e["fig"] not in figuras  # noqa: E731
    return (descricao,
            _mascara(lambda e: livre(e) or pred_a(e)),
            _mascara(lambda e: livre(e) or pred_b(e)),
            frozenset(figuras), eixo)


def _quantificador(prefixo):
    p = (prefixo or "").strip()
    if p in ("apenas", "somente", "exatamente", "so", "unicamente"):
        return "exato"
    if p in ("pelo menos", "no minimo", "ao menos"):
        return "minimo"
    return "ambiguo"


def _fato_contagem_angulo(c, n, modo):
    nome = _NOME_TIPO[c] + ("s" if n > 1 else "")
    if modo == "todos":
        return _restricao(f"todos os ângulos {_NOME_TIPO[c]}s",
                          lambda e: e[c] == _nlados(e), eixo="angulos")
    if modo == "exato":
        return _restricao(f"exatamente {n} ângulo(s) {nome}", lambda e: e[c] == n, eixo="angulos")
    if modo == "minimo":
        return _restricao(f"pelo menos {n} ângulo(s) {nome}", lambda e: e[c] >= n, eixo="angulos")
    return _restricao(f"{n} ângulo(s) {nome}", lambda e: e[c] >= n, lambda e: e[c] == n,
                      eixo="angulos")


def _compilar(padrao):
    return re.compile(padrao, _A)


# --- Teoremas e contas (não são dados da figura) ---------------------------
_SOMA = _compilar(r"\b(?:soma\w*|totaliz\w*)\b[^.;?]{0,60}?" + _NUM + _GRAU)
_CONTA_ANGULOS = _compilar(_NUM + _GRAU + r"\s*[+=]|[+=]\s*" + _NUM + _GRAU)

# --- Ângulos numéricos -----------------------------------------------------
_ANG_RESTO = _compilar(
    r"\bos (?:outros|demais) (?:(dois|duas|tres|2|3) )?(?:angulos )?(?:internos )?"
    r"(?:sao de|sao|medem|medindo|de|com|tem|possuem|valem) " + _NUM + _GRAU + r"( cada)?"
    r"(?!\s*(?:,|e)\s*\d)")  # "os outros dois medem 120° e 60°" é lista, não "cada"
# "todos os ângulos medem 90°" / "cada ângulo mede 60°" / "os ângulos medem
# 60° cada". Sem "todos"/"cada", "os ângulos medem 50°, 60° e 70°" é uma LISTA
# (G-9H17-0126): por isso o lookahead recusa continuação ", N" ou "e N".
_ANG_TODOS = _compilar(
    r"\b(?:todos os (?:seus )?(?:tres |quatro |3 |4 )?angulos (?:internos? )?(?:d[oa] \w+ )?"
    r"(?:sao de|sao|medem|medindo|de|com|iguais a|tem|com medida de|medida de|valem) "
    + _NUM + _GRAU + r"(?: cada)?"
    r"|cada (?:um dos )?angulos? (?:internos? )?(?:d[oa] \w+ )?(?:mede|medindo|tem|de|vale) "
    + _NUM + _GRAU +
    r"|os (?:tres |quatro |3 |4 )?angulos (?:internos? )?(?:d[oa] \w+ )?"
    r"(?:medem|sao de|tem|valem) " + _NUM + _GRAU + r" cada)"
    r"(?!\s*(?:,|e)\s*\d)")
_ANG_K = _compilar(
    r"\b(dois|duas|tres|quatro|2|3|4) (?:angulos )?(?:internos )?"
    r"(?:de|medindo|que medem|medem|com|iguais a|com medida de) " + _NUM + _GRAU
    + r"( cada)?(?!\s*(?:,|e)\s*\d)")
_ANG_NUM = _compilar(_NUM + _GRAU)

# --- Lados numéricos -------------------------------------------------------
_FRASE_DE_LADO = _compilar(r"\blados?\b|\bmedidas?\b|\bmedem\b|\bmede\b|\bmedindo\b|"
                           r"\bcomprimentos?\b|\bdimensoes\b|\b[a-z]{2} ?= ?\d")
# Frases com outra grandeza: os números podem não ser lados (altura, raio...).
_FRASE_OUTRA_GRANDEZA = _compilar(r"\baltura|\bperimetr|\barea\b|\bdiagona|\braio|\bdiametr|"
                                  r"\bbases?\b|\bdistancia|\bcatet|\bhipotenusa")
# "todos os lados medem 3 m", "cada lado mede 3 m", "lados medindo 3 m cada".
# Sem marca universal, "um triângulo com lados de 5 cm" é ambíguo (todos? um
# deles?) e fica como medida avulsa (parcial), que nunca gera acusação.
_LADOS_TODOS = _compilar(
    r"\b(?:(?:todos os (?:seus )?(?:tres |quatro |3 |4 )?lados|os (?:tres|quatro|3|4) lados|"
    r"cada (?:um dos )?lados?) (?:d[oa] \w+ )?"
    r"(?:medem|mede|medindo|de|com|tem|que medem|com medida de) " + _NUM + r"\s*" + _UNID
    + r"(?: cada)?"
    r"|lados (?:d[oa] \w+ )?(?:medem|medindo|de|com|que medem|com medida de) "
    + _NUM + r"\s*" + _UNID + r" cada)(?!\s*(?:,|e)\s*\d)")
_LADOS_K = _compilar(
    r"\b(dois|duas|tres|quatro|2|3|4) lados (?:iguais )?"
    r"(?:de|medindo|com|que medem|medem|mede|com medida de) " + _NUM + r"\s*" + _UNID
    + r"( cada)?(?!\s*(?:,|e)\s*\d)")
_LADOS_LISTA = _compilar(
    r"\blados?\s*(?:de|medindo|medem|com|que medem|sao|:|=)?\s*"
    r"(\d+(?:[.,]\d+)?(?:\s*,\s*\d+(?:[.,]\d+)?)*\s*(?:,|e)\s*\d+(?:[.,]\d+)?)"
    r"(?:\s*" + _UNID + r")?(?!\s*(?:°|graus?\b))")
_LADO_NUM = _compilar(_NUM + r"\s*" + _UNID)
# Introdução de uma lista JÁ consumida ("cujos lados medem [5 m, 7 m e 9 m]"):
# o texto apagado deixa 2+ espaços seguidos, que no texto normalizado só
# aparecem onde houve consumo. Sem isto, "lados medem" sobrava e a completude
# acusava trecho não lido (R1-Q3).
# "ângulos DA BASE medindo 40° cada" NÃO é introdução de todos os ângulos: a
# base e o vértice nomeiam ângulos específicos (falso "dados insuficientes" no
# teste adversarial quando o trecho era engolido aqui).
_INTRODUCAO = _compilar(r"\b(?:lados?|angulos?|medidas?)(?: internos?)?"
                        r"(?: (?:d[oa]|de um|de uma|desse|dessa|deste|desta) "
                        r"(?!base|vertice|topo|ponta)[a-z]+)?"
                        r"(?: (?:medem|medindo|mede|que medem|valem|sao de|sao|tem|de|com|"
                        r"iguais a|seguintes))* ?:?(?= {2})")

# --- Ângulos verbais -------------------------------------------------------
_ANG_TODOS_TIPO = _compilar(
    r"\b(?:todos os (?:seus )?(?:tres |quatro |3 |4 )?angulos (?:internos )?(?:d[oa] \w+ )?"
    r"(?:sao |sendo |eram |serao )?(?:todos )?"
    r"|(?:os|seus) (?:seus )?angulos (?:internos )?(?:d[oa] \w+ )?(?:sao |sendo |eram )?todos "
    r"|os (?:tres|quatro|3|4) angulos (?:internos )?(?:sao )?"
    r"|(?:apenas|somente|so) angulos (?:internos )?)" + _TIPO_ANG + r"\b")
_ANG_QTD_TIPO = _compilar(
    r"\b(?:(apenas|somente|exatamente|so|unicamente|pelo menos|no minimo|ao menos) )?"
    + _QTD + r" (?:unico )?angulos? (?:internos? )?(?:sao |e |eh )?" + _TIPO_ANG + r"\b"
    r"(?:,? e (?:os )?(?:outros )?(um|uma|dois|duas|tres|1|2|3) (?:outros? )?(?:angulos? )?"
    r"(?:internos? )?(?:sao |eh )?" + _TIPO_ANG + r"\b)?")
# "ângulo interno obtuso" no SINGULAR e sem numeral (G-9H17-0101, depois da
# reescrita de "maior que 90°") é um ângulo daquele tipo: o singular já conta.
_ANG_SINGULAR_TIPO = _compilar(r"\bangulo (?:interno )?(?:eh |e )?" + _TIPO_ANG + r"\b")
_ANG_IGUAIS_2 = _compilar(r"\b(?:dois|2) angulos (?:internos )?(?:sao )?" + _IGUAIS)
_ANG_IGUAIS = _compilar(
    r"\b((?:todos os |os tres |os quatro |os |seus )?)angulos (?:internos )?((?:sao )?(?:todos )?)"
    + _IGUAIS + r"(?! dois a dois| 2 a 2| entre si dois)")

# --- Lados verbais ---------------------------------------------------------
_PARES_IGUAIS = _compilar(r"\b(?:dois|2) pares de lados (opostos |consecutivos |adjacentes )?"
                          + _IGUAIS)
_LADOS_K_IGUAIS = _compilar(
    r"\b(dois|duas|tres|quatro|2|3|4) lados (?:sao )?" + _IGUAIS
    + r"( e (?:um|o outro|o terceiro|o quarto|os outros dois|outros dois|dois|o outro lado|"
      r"um lado|os demais|outro) (?:lados? )?" + _DIFERENTES + r")?")
_LADOS_IGUAIS = _compilar(
    r"\b((?:dois|duas|2|um|uma|1) )?((?:todos os |os tres |os quatro |os seus |seus |os )?)"
    r"lados (?:d[oa] \w+ )?((?:sao |tem |com )?(?:todos )?)" + _IGUAIS)
# "os lados iguais" SEM quantificador nem verbo de ligação é ATRIBUTIVO: nomeia
# o PAR de lados iguais do isósceles ("os lados iguais medem 5 cm e o outro
# mede 8 cm", "o ângulo entre os lados iguais mede 100°"). Lido como "todos os
# lados iguais", virava equilátero e reprovava questão CORRETA (5 sondas da
# revisão adversarial de 2026-10-01, uma com sugestão de letra errada: "os
# ângulos iguais medem 30° cada" é 30-30-120, obtusângulo). É universal só com
# "todos"/"os três"/"os quatro", com "são"/"todos" antes de "iguais" ou como
# complemento de ter/possuir/com fechando a oração ("tem os lados iguais.").
_UNIVERSAL_ANTES = _compilar(r"\b(?:tem|possui|possuem|apresenta|apresentam|com)\s*$")
_FIM_DE_ORACAO = _compilar(r"^\s*(?:[.,;:?!]|$|e\b)")


def _iguais_universal(f, m, artigo, verbo):
    """True se "lados/ângulos iguais" do casamento `m` quer dizer TODOS."""
    if re.search(r"todos|tres|quatro", artigo, _A) or verbo.strip():
        return True
    if artigo.strip() in ("os", "seus", "os seus"):
        antes = f.texto[max(0, m.start() - 12):m.start()]
        return bool(_UNIVERSAL_ANTES.search(antes) and _FIM_DE_ORACAO.search(f.texto[m.end():]))
    return False
_LADOS_DIF = _compilar(
    r"\b((?:dois|duas|2|um|uma|1) )?(?:todos os |os tres |os quatro |tres |quatro |3 |4 |"
    r"os seus |seus |os )?lados (?:d[oa] \w+ )?(?:sao |tem |com |possuem )?(?:todos )?"
    + _DIFERENTES + r"(?: entre si)?")
_PAR_PARALELO = _compilar(
    r"\b(?:(apenas|somente|so|exatamente|unicamente) )?(?:um|1) (?:unico )?par de lados "
    r"(?:opostos )?paralelos")
_DOIS_PARES_PARALELOS = _compilar(r"\b(?:dois|2) pares de lados (?:opostos )?paralelos"
                                  r"|\blados opostos (?:sao )?paralelos")
_DOIS_LADOS_PARALELOS = _compilar(r"\b(?:apenas |somente )?(?:dois|2) lados (?:opostos )?"
                                  r"(?:sao )?paralelos")
_OPOSTOS_IGUAIS = _compilar(r"\b((?:dois|2) )?lados opostos (?:sao )?" + _IGUAIS)


def _consumir(f, w, padrao, tratar):
    """Aplica `padrao` ao texto de trabalho `w`, chama `tratar(m)` e APAGA o
    trecho (mesmo comprimento), para que um padrão posterior não releia o que
    um anterior já interpretou. Era o defeito do protótipo em "um par de lados
    opostos paralelos": o padrão "lados opostos paralelos" relia o trecho e o
    fato virava "dois pares" + "um par" (falsa premissa impossível).
    Fato precedido de negação ("não tem todos os lados iguais") é consumido
    SEM virar restrição: aplicá-lo como afirmação inverteria o sentido."""
    spans = []
    for m in padrao.finditer(w):
        janela = f.texto[max(0, m.start() - 25):m.start()] + " " + m.group(0)
        if _NEGACAO.search(janela):
            f.negacao = True
            spans.append(m.span())
            continue
        if tratar(m) is not False:
            spans.append(m.span())
    for a, b in spans:
        w = _apagar(w, a, b)
        f.consumido.append((a, b))
    return w


def _indice_frase(texto, pos):
    return len(re.findall(r"[.;?!]\s", texto[:pos]))


def _extrair_fatos(texto):
    """Fatos de um trecho JÁ em minúsculas (ver _minusculas)."""
    f = _Fatos(texto)
    w = texto

    def medida_em(m):
        f.frases_medidas.add(_indice_frase(texto, m.start()))

    # 1) Teoremas e contas não são dados: "os ângulos internos somam 180°".
    def soma(m):
        if _medida(m.group(1)) not in ((180, 0), (360, 0)):
            f.pendencias.append("soma_de_angulos")
    w = _consumir(f, w, _SOMA, soma)
    w = _consumir(f, w, _CONTA_ANGULOS, lambda m: f.pendencias.append("conta_com_angulos"))

    # 2) Ângulos numéricos.
    def resto(m):
        k = _NUMW.get(m.group(1)) if m.group(1) else None
        if not m.group(3) and not k:
            f.pendencias.append("demais_angulos_sem_cada")
            return
        f.resto_cada = (k, _medida(m.group(2)))
        medida_em(m)
    w = _consumir(f, w, _ANG_RESTO, resto)

    def todos_ang(m):
        f.angulos_todos = _medida(next(g for g in m.groups() if g))
        medida_em(m)
    w = _consumir(f, w, _ANG_TODOS, todos_ang)

    def k_ang(m):
        f.angulos.extend([_medida(m.group(2))] * _NUMW[m.group(1)])
        medida_em(m)
    w = _consumir(f, w, _ANG_K, k_ang)

    def ang(m):
        f.angulos.append(_medida(m.group(1)))
        medida_em(m)
    w = _consumir(f, w, _ANG_NUM, ang)

    # 3) Lados numéricos, só em frases que falam de lado/medida.
    inicio = 0
    for frase in _frases(w):
        p0 = w.find(frase, inicio)
        if p0 < 0:
            continue
        inicio = p0 + len(frase)
        if not _FRASE_DE_LADO.search(frase):
            continue
        if _FRASE_OUTRA_GRANDEZA.search(frase):
            if _LADO_NUM.search(frase):
                f.pendencias.append("medidas_misturadas_com_outra_grandeza")
            continue
        sub = _Fatos(frase)
        ws = frase

        # "dois lados medindo 5 cm" ANTES de "lados medindo 5 cm" (= todos):
        # sem essa ordem o segundo padrão casava a partir de "lados" e lia
        # "todos os lados medem 5 cm".
        def k_lados(m):
            sub.lados.extend([_medida(m.group(2)) + (m.group(3),)] * _NUMW[m.group(1)])
        ws = _consumir(sub, ws, _LADOS_K, k_lados)

        def todos_lados(m):
            num, unid = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            sub.lados_todos = _medida(num) + (unid,)
        ws = _consumir(sub, ws, _LADOS_TODOS, todos_lados)

        def lista(m):
            unid = m.group(2)
            nums = re.findall(r"\d+(?:[.,]\d+)?", m.group(1), _A)
            sub.lados.extend(_medida(x) + (unid,) for x in nums)
        ws = _consumir(sub, ws, _LADOS_LISTA, lista)
        ws = _consumir(sub, ws, _LADO_NUM, lambda m: sub.lados.append(_medida(m.group(1)) + (m.group(2),)))
        f.lados.extend(sub.lados)
        if sub.lados_todos:
            f.lados_todos = sub.lados_todos
        if sub.lados or sub.lados_todos:
            f.frases_medidas.add(_indice_frase(w, p0))
        f.negacao = f.negacao or sub.negacao
        for a, b in sub.consumido:
            f.consumido.append((p0 + a, p0 + b))
            w = _apagar(w, p0 + a, p0 + b)
    w = _consumir(f, w, _INTRODUCAO, lambda m: None)

    # 4) Ângulos verbais.
    def todos_tipo(m):
        f.restricoes.append(_fato_contagem_angulo(m.group(1)[0], 0, "todos"))
    w = _consumir(f, w, _ANG_TODOS_TIPO, todos_tipo)

    def qtd_tipo(m):
        modo = _quantificador(m.group(1))
        f.restricoes.append(_fato_contagem_angulo(m.group(3)[0], _NUMW[m.group(2)], modo))
        if m.group(4):
            f.restricoes.append(_fato_contagem_angulo(m.group(5)[0], _NUMW[m.group(4)], modo))
    w = _consumir(f, w, _ANG_QTD_TIPO, qtd_tipo)
    w = _consumir(f, w, _ANG_SINGULAR_TIPO, lambda m: f.restricoes.append(
        _fato_contagem_angulo(m.group(1)[0], 1, "ambiguo")))

    # "dois ângulos iguais" só é modelado no triângulo (ângulos da base do
    # isósceles). No quadrilátero não diz nada que o universo represente: o
    # protótipo cortava TODOS os quadriláteros e acusava premissa impossível.
    w = _consumir(f, w, _ANG_IGUAIS_2, lambda m: f.restricoes.append(_restricao(
        "2 ângulos iguais", lambda e: e["part"][0] >= 2, lambda e: e["part"] == (2, 1),
        figuras=("tri",), eixo="lados")))
    def ang_iguais(m):
        if _iguais_universal(f, m, m.group(1), m.group(2)):
            f.restricoes.append(_restricao(
                "todos os ângulos iguais",
                lambda e: e["part"] == (3,) if e["fig"] == "tri" else e["r"] == 4, eixo="angulos"))
        else:
            # Leitura FRACA, verdadeira em qualquer interpretação: há (pelo
            # menos) dois ângulos iguais. Só o triângulo a modela, e a
            # pendência impede veredito que dependa de I (dados insuficientes).
            f.restricoes.append(_restricao("pelo menos 2 ângulos iguais",
                                           lambda e: e["part"][0] >= 2, figuras=("tri",),
                                           eixo="lados"))
            f.pendencias.append("iguais_sem_quantificador")
    w = _consumir(f, w, _ANG_IGUAIS, ang_iguais)

    # 5) Lados verbais.
    def pares_iguais(m):
        tipo = (m.group(1) or "").strip()
        if tipo == "opostos":
            f.restricoes.append(_restricao("2 pares de lados opostos iguais",
                                           lambda e: e["opp"] == 2, figuras=("quad",)))
        elif tipo:
            f.restricoes.append(_restricao("2 pares de lados consecutivos iguais",
                                           lambda e: e["pipa"], figuras=("quad",)))
        else:
            f.restricoes.append(_restricao(
                "2 pares de lados iguais", lambda e: e["part"] in ((2, 2), (4,)),
                lambda e: e["part"] == (2, 2), figuras=("quad",)))
    w = _consumir(f, w, _PARES_IGUAIS, pares_iguais)

    def k_iguais(m):
        k = _NUMW[m.group(1)]
        exato = bool(m.group(2))
        if k == 2:
            outros_dois = exato and re.search(r"(?:dois|outros dois|os demais)", m.group(2), _A)
            if outros_dois:
                f.restricoes.append(_restricao(
                    "2 lados iguais e os outros 2 diferentes",
                    lambda e: e["part"] == ((2, 1) if e["fig"] == "tri" else (2, 1, 1))))
            elif exato:
                f.restricoes.append(_restricao("exatamente 2 lados iguais",
                                               lambda e: e["part"][0] == 2))
            else:
                f.restricoes.append(_restricao("2 lados iguais", lambda e: e["part"][0] >= 2,
                                               lambda e: e["part"][0] == 2))
        elif k == 3:
            # No triângulo, "três lados iguais" é o equilátero: part (3,) cumpre
            # part[0] >= 3 e part[0] == 3. "Três iguais e um diferente" só
            # existe no quadrilátero (num triângulo seriam 4 lados).
            if exato:
                f.restricoes.append(_restricao("3 lados iguais e 1 diferente",
                                               lambda e: e["part"] == (3, 1)))
            else:
                f.restricoes.append(_restricao("3 lados iguais", lambda e: e["part"][0] >= 3,
                                               lambda e: e["part"][0] == 3))
        else:
            f.restricoes.append(_restricao("4 lados iguais", lambda e: e["part"] == (4,)))
    w = _consumir(f, w, _LADOS_K_IGUAIS, k_iguais)

    def lados_iguais(m):
        if m.group(1):
            return False  # "dois lados iguais" sem padrão reconhecido: não modela
        if _iguais_universal(f, m, m.group(2), m.group(3)):
            f.restricoes.append(_restricao("todos os lados iguais", lambda e: len(e["part"]) == 1))
        else:
            f.restricoes.append(_restricao("pelo menos 2 lados iguais",
                                           lambda e: e["part"][0] >= 2))
            f.pendencias.append("iguais_sem_quantificador")
    w = _consumir(f, w, _LADOS_IGUAIS, lados_iguais)

    def lados_dif(m):
        if m.group(1):
            return False  # "dois lados diferentes" não diz que TODOS são diferentes
        f.restricoes.append(_restricao("todos os lados diferentes",
                                       lambda e: all(p == 1 for p in e["part"])))
    w = _consumir(f, w, _LADOS_DIF, lados_dif)

    def par_paralelo(m):
        exato = bool(m.group(1))
        f.restricoes.append(_restricao(
            ("exatamente " if exato else "") + "1 par de lados paralelos",
            lambda e: e["npar"] == 1 if exato else e["npar"] >= 1,
            lambda e: e["npar"] == 1, eixo="paralelos"))
    w = _consumir(f, w, _PAR_PARALELO, par_paralelo)
    w = _consumir(f, w, _DOIS_PARES_PARALELOS, lambda m: f.restricoes.append(_restricao(
        "2 pares de lados paralelos", lambda e: e["npar"] == 2, eixo="paralelos")))
    w = _consumir(f, w, _DOIS_LADOS_PARALELOS, lambda m: f.restricoes.append(_restricao(
        "2 lados paralelos", lambda e: e["npar"] >= 1, lambda e: e["npar"] == 1,
        eixo="paralelos")))

    def opostos(m):
        if m.group(1):
            # "dois lados opostos iguais" é UM par (G-9H17-0103), não os dois.
            f.restricoes.append(_restricao("1 par de lados opostos iguais",
                                           lambda e: e["opp"] >= 1, figuras=("quad",)))
        else:
            f.restricoes.append(_restricao("lados opostos iguais", lambda e: e["opp"] == 2,
                                           figuras=("quad",)))
    w = _consumir(f, w, _OPOSTOS_IGUAIS, opostos)
    f.residuo = w
    return f


# ===========================================================================
# 5. ESTADOS COMPATÍVEIS COM OS FATOS
# ===========================================================================
def _particoes_parciais(conhecidos, n, ciclico):
    """(partição, opp) possíveis quando só parte dos n lados é conhecida.

    Cada lado desconhecido é igual a um conhecido ou é um valor novo; enumera
    todas as atribuições (n <= 4, barato). É uma SUPERaproximação: ignora a
    desigualdade triangular dos lados novos, o que só alarga S (seguro)."""
    k = len(conhecidos)
    candidatos = sorted(set(conhecidos)) + [("novo", i) for i in range(n - k)]
    saida = set()

    def rec(lista):
        if len(lista) == n:
            opp = None
            if n == 4 and ciclico:
                opp = int(lista[0] == lista[2]) + int(lista[1] == lista[3])
            saida.add((_particao(lista), opp))
            return
        for c in candidatos:
            rec(lista + [c])
    rec(list(conhecidos))
    return saida


# Teto das medidas ESCALADAS (inteiros). O app roda em TypeScript, onde
# Number só é exato até 2^53: com lados < 2^26, a² + b² - c² < 2^53 e a conta
# dá o mesmo resultado que o int ilimitado do Python. Acima disso o
# verificador se abstém (a terna 6074549952/8101350014/10125810050 mm dava
# dif = 0 no Python e -16384 em Number: vereditos diferentes nas duas pontas).
_TETO_EXATO = 2 ** 26


def _estados(f, figura, leitura, rotulos=()):
    """(S, motivos_de_impossibilidade, pendencias, fora_do_universo)."""
    S = set(FIGURA[figura]) if figura else set(TODOS)
    motivos, pend = [], []
    fora = False
    for desc, mA, mB, figs, _eixo in f.restricoes:
        S &= mA if leitura == "A" else mB
        if figura and figura not in figs:
            pend.append("fato_nao_modelado_para_a_figura:" + desc)
    for r in rotulos:
        S &= ROTULOS[leitura][r]
    n = NUM_LADOS.get(figura)

    # --- lados numéricos
    lados = list(f.lados)
    if f.lados_todos and n:
        lados = lados + [f.lados_todos] * max(0, n - len(lados))
    if lados and n:
        unidades = {m[2] for m in lados}
        fatores = None
        if len(unidades) > 1:
            if None in unidades:
                # "lados de 3, 4 e 50 cm" misturado com medida sem unidade: na
                # dúvida, não usa as medidas.
                pend.append("unidades_misturadas")
                lados = []
            else:
                fatores = [_FATOR_MM[m[2]] for m in lados]  # "50 cm" e "0,5 m": exato em mm
        valores = _escalar(lados, fatores) if lados else []
        if valores and max(valores) > _TETO_EXATO:
            pend.append("medida_acima_do_teto_exato")
            valores = []
        if len(valores) > n:
            pend.append("medidas_de_lado_demais")
        elif len(valores) == n:
            ordenados = sorted(valores)
            if ordenados[-1] >= sum(ordenados[:-1]):
                motivos.append(("desigualdade_triangular" if n == 3 else "desigualdade_poligonal")
                               + f"({', '.join(_texto_medida(m) for m in lados)})")
                S = set()
            else:
                part = _particao(valores)
                if n == 3:
                    a, b, c = ordenados
                    dif = a * a + b * b - c * c
                    if any(m[1] > 0 for m in lados) and dif != 0 and abs(dif) * 50 <= c * c:
                        # Decimal arredondado perto da fronteira da recíproca de
                        # Pitágoras: "1 m, 1 m e 1,41 m" é o √2 do triângulo
                        # retângulo isósceles, mas 1,41² < 2 o tornaria
                        # acutângulo. Com medida aproximada, o tipo de ângulo não
                        # é decidido (só a partição dos lados). Diferença EXATAMENTE
                        # zero não é arredondamento: "1,5 m, 2 m e 2,5 m" e "0,3 m,
                        # 0,4 m e 0,5 m" são ternas pitagóricas exatas e ficavam sem
                        # veredito (lacuna apontada na revisão de 2026-10-01).
                        pend.append("medidas_aproximadas_no_limite")
                        S &= _mascara(lambda e: e["fig"] == "tri" and e["part"] == part)
                    else:
                        tipo = (1, 0, 2) if dif == 0 else ((0, 1, 2) if dif < 0 else (0, 0, 3))
                        S &= _mascara(lambda e: e["fig"] == "tri" and e["part"] == part
                                      and (e["r"], e["o"], e["a"]) == tipo)
                else:
                    S &= _mascara(lambda e: e["part"] == part)
                    if leitura == "A":
                        opp = int(valores[0] == valores[2]) + int(valores[1] == valores[3])
                        S &= _mascara(lambda e: e["opp"] == opp)
        elif valores:
            pend.append("medidas_de_lado_parciais")
            possiveis = _particoes_parciais(valores, n, leitura == "A")
            S &= _mascara(lambda e: e["fig"] != figura or any(
                e["part"] == p and (o is None or e["opp"] == o) for p, o in possiveis))

    # --- ângulos numéricos
    ang = list(f.angulos)
    if "soma_de_angulos" in f.pendencias or "conta_com_angulos" in f.pendencias:
        # "a soma de 30° e 60° é 90°", "x + 2x + 90° = 180°": os números em grau
        # do texto são parcelas de conta, não ângulos da figura. Nenhum vira dado.
        ang = []
    if n and f.angulos_todos:
        if ang:
            pend.append("angulos_todos_e_avulsos")
        else:
            ang = [f.angulos_todos] * n
    if n and f.resto_cada:
        k, m = f.resto_cada
        ang = ang + [m] * (k if k else max(0, n - len(ang)))
    if ang and n and max(_escalar(ang)) > _TETO_EXATO:
        pend.append("medida_acima_do_teto_exato")
        ang = []
    if ang and n:
        valores = _escalar(ang)
        escala = 10 ** max(m[1] for m in ang)
        raso, reto = 180 * escala, 90 * escala
        total = (n - 2) * raso
        if any(v <= 0 for v in valores):
            motivos.append("angulo_nao_positivo")
            S = set()
        elif len(valores) > n:
            pend.append("medidas_de_angulo_demais")
        else:
            if len(valores) == n - 1 and sum(valores) < total:
                valores = valores + [total - sum(valores)]
            if any(v >= raso for v in valores):
                if n == 3:
                    motivos.append("angulo_de_180_ou_mais_no_triangulo")
                    S = set()
                else:
                    fora = True  # côncavo (ou degenerado): fora do universo convexo
            elif len(valores) == n:
                aproximado = any(m[1] > 0 for m in ang)
                if sum(valores) != total and aproximado and abs(sum(valores) - total) <= escala:
                    # "33,3°, 33,3° e 113,3°" soma 179,9° por arredondamento: com
                    # decimais, até 1° de diferença não é premissa impossível.
                    pend.append("soma_de_angulos_aproximada")
                elif sum(valores) != total:
                    motivos.append(f"soma_dos_angulos={_texto_soma(valores, escala)}"
                                   f"_deveria_ser_{(n - 2) * 180}")
                    S = set()
                else:
                    r = sum(1 for v in valores if v == reto)
                    o = sum(1 for v in valores if v > reto)
                    a = sum(1 for v in valores if v < reto)
                    S &= _mascara(lambda e: (e["r"], e["o"], e["a"]) == (r, o, a))
                    if n == 3:
                        part = _particao(valores)  # ângulos iguais <=> lados opostos iguais
                        S &= _mascara(lambda e: e["part"] == part)
                    elif leitura == "A":
                        # ordem cíclica: ângulos consecutivos suplementares <=> lados paralelos
                        npar = int(valores[0] + valores[1] == raso) + int(valores[0] + valores[3] == raso)
                        S &= _mascara(lambda e: e["npar"] == npar)
            elif sum(valores) >= total:
                motivos.append(f"angulos_conhecidos_somam_{_texto_soma(valores, escala)}")
                S = set()
            else:
                for c, cond in (("r", lambda v: v == reto), ("o", lambda v: v > reto),
                                ("a", lambda v: v < reto)):
                    k = sum(1 for v in valores if cond(v))
                    if k:
                        S &= _mascara(lambda e, c=c, k=k: e[c] >= k)
    if not S and not motivos:
        base = FIGURA[figura] if figura else TODOS
        sozinhos = [d for d, mA, mB, _f, _e in f.restricoes
                    if not ((mA if leitura == "A" else mB) & base)]
        if sozinhos:
            motivos.append("fato_impossivel:" + "; ".join(sozinhos))
        else:
            motivos.append("combinacao_impossivel:" + "; ".join(f.descricoes() + list(rotulos)))
    return frozenset(S), motivos, pend, fora


def _texto_soma(valores, escala):
    s = sum(valores)
    return str(s // escala) if s % escala == 0 else f"{s / escala:g}"


# ===========================================================================
# 6. FIGURA, PERGUNTA, EIXO, MODO
# ===========================================================================
_FIG_TRI = _compilar(r"\btriangul")
_FIG_QUAD = _compilar(r"\bquadrilater|\bquadrad|\blosango|\btrapezi|\bparalelogram|\brombos?\b")
_RET = _compilar(r"\bretangul")
_TRI_RET = _compilar(r"\btriangulos? retangul")
_QUALIFICA = (r"(?!\s+(?:iguais|congruentes|de mesm|diferentes|distintos|desiguais|retos?|agudos?|"
              r"obtusos?|paralelos|opostos|com medidas|de medidas|de comprimentos))")
_CONTA_QUAD = _compilar(r"\b(?:quatro|4) (?:lados|vertices|pontas)\b" + _QUALIFICA)
_CONTA_TRI = _compilar(r"\b(?:tres|3) (?:lados|vertices|pontas)\b" + _QUALIFICA)


def _familias_citadas(t):
    """Famílias de figura NOMEADAS no texto ("triângulo retângulo" é triângulo)."""
    familias = set()
    if _FIG_TRI.search(t):
        familias.add("tri")
    if _FIG_QUAD.search(t) or _RET.search(_TRI_RET.sub(" ", t)):
        familias.add("quad")
    return familias


def _figura_explicita(t):
    """'tri' | 'quad' | None. O NOME tem prioridade sobre contagens: "trapézio
    com três lados iguais" é quadrilátero (R1-Q10), não triângulo. Contagem só
    vale como TOTAL ("folha com 4 lados"), nunca qualificando ("três ângulos
    obtusos" é dado de um quadrilátero)."""
    familias = _familias_citadas(t)
    if len(familias) == 1:
        return next(iter(familias))
    if familias:
        return None
    if _CONTA_QUAD.search(t):
        return "quad"
    if _CONTA_TRI.search(t):
        return "tri"
    return None


def _figura_por_medidas(f):
    if f.lados and not f.angulos and len(f.lados) in (3, 4):
        return "tri" if len(f.lados) == 3 else "quad"
    if f.angulos and len(f.angulos) in (3, 4) and not f.resto_cada:
        return "tri" if len(f.angulos) == 3 else "quad"
    return None


_IMPERATIVO = _compilar(r"\b(?:classifique|assinale|indique|identifique|determine|marque|responda)\b")


def _separar_pergunta(t):
    """(corpo, pergunta). A pergunta é a ÚLTIMA frase interrogativa ou
    imperativa; o resto (antes E depois dela) é corpo: em R1-Q4 o dado "os
    ângulos do espelho são todos obtusos" vem DEPOIS da pergunta."""
    fs = _frases(t)
    idx = [i for i, f in enumerate(fs) if "?" in f or _IMPERATIVO.search(f)]
    if not idx:
        if fs and fs[-1].endswith(":"):
            idx = [len(fs) - 1]  # "A forma ... pode ser classificada como:" (R1-Q6)
        else:
            return t, ""
    i = idx[-1]
    return " ".join(fs[:i] + fs[i + 1:]), fs[i]


# Eixo nomeado na pergunta. Erro de eixo é perigoso NOS DOIS SENTIDOS: eixo
# perdido faz 90-45-45 virar falsa "não única"; eixo inventado faz o gabarito
# certo virar falso. Por isso: introdutores FORTES ("quanto aos", "em relação
# aos") têm prioridade sobre os FRACOS ("pelos", "considerando"), "lados e
# ângulos" juntos anulam o eixo, e "lados de 3, 4 e 5 cm" (dado) não é eixo.
_EIXO_MEIO = (r"\s+(?:(?:a|o|as|os|aos|ao|seus|suas|seu|sua|apenas|somente|d[oa]s?|tipos?|de|"
              r"medidas?|tamanhos?|comprimentos?|classificacao)\s+){0,4}")
_EIXO_FIM = (r"\b(?!\s*(?:de|medindo|que medem|medem|com|:|=)?\s*\d)"
             r"(?! (?:iguais|diferentes|congruentes|retos|agudos|obtusos|opostos|paralelos))")
_EIXO_FORTE = r"\b(?:quanto|relacao|referencia)"
_EIXO_FRACO = r"\b(?:considerando|segundo|pel[oa]s?|classificacao|criterio)"
_EIXO_AMBOS = _compilar(r"\blados?\b(?:,| e| ou)(?:\s+(?:a|o|as|os|aos|ao|quanto|seus|suas|"
                        r"pel[oa]s?|tambem|em relacao))*\s+angulos?\b|"
                        r"\bangulos?\b(?:,| e| ou)(?:\s+(?:a|o|as|os|aos|ao|quanto|seus|suas|"
                        r"pel[oa]s?|tambem|em relacao))*\s+lados?\b")


def _eixo_sugerido(corpo):
    """Eixo insinuado FORA da pergunta: "quanto aos lados" no corpo ou uma lista
    de categorias de um eixo só ("os triângulos podem ser equiláteros,
    isósceles ou escalenos"). Não restringe as alternativas; só impede que a
    falta de eixo NA PERGUNTA vire acusação de "não única"."""
    e = _eixo(corpo)
    if e:
        return e
    por_eixo = {}
    for _a, _b, r in _rotulos_no_texto(corpo):
        if r in EIXO:
            por_eixo.setdefault(EIXO[r], set()).add(r)
    listados = [e for e, rs in por_eixo.items() if len(rs) >= 2]
    return listados[0] if len(listados) == 1 else None


def _eixo(pergunta):
    if _EIXO_AMBOS.search(pergunta):
        return None
    for pre in (_EIXO_FORTE, _EIXO_FRACO):
        lados = bool(re.search(pre + _EIXO_MEIO + r"lados?" + _EIXO_FIM, pergunta, _A))
        angulos = bool(re.search(pre + _EIXO_MEIO + r"angulos?" + _EIXO_FIM, pergunta, _A))
        if lados != angulos:
            return "lados" if lados else "angulos"
        if lados and angulos:
            return None
    return None


_EXISTENCIA = _compilar(
    r"possivel (?:formar|construir|existir|desenhar|montar)|\bexiste\b|\bexistir\b|\bexistem\b|"
    r"pode(?:m)? (?:formar|existir|ser construid|ser formad|ser montad|ser desenhad)|"
    r"condicao de existencia|desigualdade triangular")
_REVERSO = _compilar(
    r"\bqua(?:l|is) (?:das alternativas|dos seguintes|das seguintes|dos quadrilateros|"
    r"dos triangulos|das figuras|triangulo|quadrilatero|figura|desses|destes|destas|dessas)\b"
    r"[^?]{0,40}\b(?:tem|possui|apresenta|mostra|representa|eh um exemplo|pode ter|"
    r"podem ter|pode possuir|pode apresentar)\b")
_MODAL = _compilar(r"\bpode(?:m|ria|riam)? (?:ter|possuir|apresentar)\b")
# Pergunta de polaridade INVERTIDA ("Qual classificação NÃO se aplica?",
# "assinale a incorreta", "qual é impossível?", "todos, exceto"): o gabarito
# certo é justamente a alternativa FALSA, e a lógica V/F daria acusação falsa
# (achado no teste adversarial). Na dúvida, nao_aplicavel.
_PERGUNTA_NEGATIVA = _compilar(r"\b(?:nao|nunca|jamais|exceto|incorret\w*|errad[oa]s?|fals[oa]s?|"
                               r"impossive(?:l|is)|inexistente)\b")

# Fora do escopo do universo (sólidos, planificação, contagem, círculo...).
_FORA_DE_ESCOPO = _compilar(
    r"perimetr|quadricul|coordenad|planifica|piramide|prisma|\bcubos?\b|\bsolidos?\b|"
    r"\bfaces?\b|arestas?|cilindr|\bcones?\b|esfera|poliedr|\bquantos\b|\bquantas\b|"
    r"circunferencia|\bcirculos?\b|\bextern")
_PORTA_ROTULO = _compilar(r"equilater|isos?cel|escalen|acutangul|obtusangul|retangul|quadrad|"
                          r"losango|\brombo|trapezi|paralelogram|triangul|quadrilater|\bpipa|deltoide")

# ===========================================================================
# 7. COMPLETUDE (exigida para vereditos que dependem de I)
# ===========================================================================
_NAO_MODELADO = _compilar(
    r"diagona|simetri|bissetri|mediana|mediatriz|\baltura|perimetr|\barea\b|\braio|inscrit|"
    r"circunscr|extern|perpendicular|paralel|congruent|semelhan|\bvertices?\b|catet|"
    r"hipotenusa|consecutiv|adjacent|\bopost")
_QUALIFICADO = _compilar(r"\b(?:lados?|angulos?)\b[^.;?,]{0,30}\b(?:igua|diferent|distint|"
                         r"desigua|mesm|reto|retang|agud|obtus|med|maior|menor|nao|\d)")
# Qualificador que SOBROU depois do consumo, sem "lado"/"ângulo" por perto: em
# "um ângulo reto e dois ângulos agudos DE MESMA MEDIDA" o padrão leu as
# contagens e o "de mesma medida" (o que faz o triângulo ser isósceles) ficou
# solto. Sem esta checagem a completude dava True e o gabarito certo
# "Isósceles e retângulo" virava "dados insuficientes" (falso positivo achado
# no teste adversarial, fora do corpus).
_QUALIFICADOR_SOLTO = _compilar(r"\b(?:iguais|igual|congruentes?|diferentes?|distint[oa]s?|"
                                r"desiguais|desigual|mesm[oa]s? (?:medida|comprimento|tamanho)s?|"
                                r"retos?|agudos?|obtusos?|paralel\w*|proporcion\w*)\b")
# Notação que o parser não lê: "AB = BC = CA", "AB ∥ CD", "AB ⊥ BC". Sem isto
# a relação passava despercebida e a questão (correta) virava "dados
# insuficientes".
_SIMBOLO = re.compile(r"[=∥∦⊥≅≡≠<>≤≥]|//")
# Número que nenhum padrão consumiu ("trechos de 300 m, 400 m e 500 m") ou um
# "cada" que sobrou ("ângulos da base medindo 40° cada"): havia medida no texto
# e o parser não a entendeu, logo "faltam dados" seria acusação falsa.
# Medida POR EXTENSO ("noventa graus", "cento e vinte graus", "cinco
# centímetros") também é medida não lida: sem isto a completude dava True e a
# questão CORRETA virava "dados insuficientes" (revisão de 2026-10-01).
_NUM_EXTENSO = (r"\b(?:zero|um|uma|dois|duas|tres|quatro|cinco|seis|sete|oito|nove|dez|onze|doze|"
                r"treze|quatorze|catorze|quinze|dezesseis|dezessete|dezoito|dezenove|vinte|trinta|"
                r"quarenta|cinquenta|sessenta|setenta|oitenta|noventa|cem|cento|duzentos|"
                r"trezentos|meio|meia|mil)\s+(?:graus?|centimetros?|metros?|milimetros?|"
                r"decimetros?|quilometros?|cm|mm|dm|km|m)\b")
_MEDIDA_NAO_LIDA = _compilar(r"\d|\bcada\b|" + _NUM_EXTENSO)
_ROTULO_SOLTO = _compilar(r"equilater|isos?cel|escalen|acutangul|obtusangul|quadrad|losango|"
                          r"\brombo|trapezi|paralelogram|\bretangul|\bpipa|deltoide")


def _completude(f, pend):
    """(completo, pendência). Falso se algo qualificado ficou sem modelar."""
    if f.negacao:
        return False, "negacao"
    if f.pendencias or pend:
        return False, (f.pendencias + pend)[0]
    # Setas ("ângulos agudos -> qual tipo?", G-9H17-0089) são pontuação, não relação.
    resto = re.sub(r"-+>|=+>|→", " ", f.residuo)
    for regex, nome in ((_NAO_MODELADO, "conceito_nao_modelado"),
                        (_QUALIFICADO, "trecho_qualificado_nao_lido"),
                        (_QUALIFICADOR_SOLTO, "qualificador_solto"),
                        (_SIMBOLO, "notacao_nao_lida"),
                        (_MEDIDA_NAO_LIDA, "medida_nao_lida"),
                        (_ROTULO_SOLTO, "rotulo_citado_sem_papel")):
        m = regex.search(resto)
        if m:
            return False, f"{nome}:{m.group(0)}"
    return True, None


# ===========================================================================
# 8. ALTERNATIVAS -> MÁSCARA
# ===========================================================================
_NDA = _compilar(r"nenhuma? d[oa]s (?:alternativas|opcoes|anteriores|respostas)|^n\.?d\.?a\.?$|"
                 r"nenhum dos (?:triangulos|quadrilateros|anteriores)\b")
_INDET = _compilar(r"nao (?:eh possivel|e possivel|se pode|da para|podemos) (?:determinar|classificar|"
                   r"saber|afirmar)|nao (?:eh |pode ser )?classificavel|impossivel (?:determinar|"
                   r"classificar)|dados insuficientes|faltam dados")
_NAO_EXISTE = _compilar(r"nao (?:existe|eh possivel (?:formar|construir)|e possivel (?:formar|construir)|"
                        r"pode ser (?:formad|construid)|forma(?:m)? (?:um )?(?:triangulo|quadrilatero))|"
                        r"impossivel (?:de )?(?:formar|construir|existir|ser construid)")
_SIM_NAO = _compilar(r"^(?:sim|nao)\b")
_CONTAGEM_FIGURAS = _compilar(r"\b(?:\d+|dois|duas|tres|quatro|cinco|seis)\s+(?:triangul|quadrad|"
                              r"retangul|losango|trapezi|quadrilater|paralelogram)")
_STOP_ALTERNATIVA = _compilar(
    r"\b(?:um|uma|o|a|os|as|e|com|de|do|da|dos|das|eh|sao|tipo|figura|formato|forma|apenas|"
    r"somente|so|que|tem|possui|seus|suas|seu|sua|todos|todas|ele|ela|classificado|"
    r"classificada|chamado|chamada)\b")


def _rotulos_no_texto(t):
    """[(início, fim, rótulo)] sem sobreposição, na ordem do texto."""
    usados = []
    for regex, rotulo in _VOCAB_RE:
        for m in regex.finditer(t):
            if any(not (m.end() <= a or m.start() >= b) for a, b, _ in usados):
                continue
            usados.append((m.start(), m.end(), rotulo))
    return sorted(usados)


def _resolver_retangulo(rotulo, figura):
    if rotulo != "RETANGULO":
        return rotulo
    return {"tri": "retangulo_t", "quad": "retangulo_q"}.get(figura)


def _retangulo_ambiguo(alternativas_norm, figura):
    """"Retângulo" sozinho é ambíguo quando as alternativas misturam nomes de
    figuras das duas famílias: em "Quadrado | Retângulo | Triângulo | Losango"
    numa questão de triângulo, "Retângulo" é o QUADRILÁTERO, não o triângulo
    retângulo (o protótipo lia como triângulo e acusava não unicidade)."""
    outra = "quad" if figura == "tri" else "tri"
    for t in alternativas_norm:
        for _a, _b, r in _rotulos_no_texto(t):
            if r not in ("RETANGULO", "triangulo", "quadrilatero") and _FAMILIA.get(r) == outra:
                return True
            if figura == "tri" and r == "quadrilatero":
                return True
    return False


def _mascara_alternativa(texto, figura, leitura, retangulo_ambiguo=False):
    """(máscara, grupo, rótulos) denotados por uma alternativa, ou None.

    Grupo é o eixo da alternativa ("lados", "angulos", "quad", "misto",
    "figura"), usado na regra da resposta mais específica. None = não mapeia
    com confiança (rótulo inventado, lista, contagem, sobra de texto)."""
    t = re.sub(r"[.;:!]+$", "", texto.strip()).strip()
    if not t or re.search(r"\bou\b", t, _A) or _CONTAGEM_FIGURAS.search(t):
        return None  # "Retângulo ou losango" (união) e "1 quadrado e 4 triângulos"
    rotulos = _rotulos_no_texto(t)
    fig_local = None
    for _a, _b, r in rotulos:
        if r in ("triangulo", "quadrilatero"):
            fig_local = "tri" if r == "triangulo" else "quad"
    classes = [r for _a, _b, r in rotulos if r not in ("triangulo", "quadrilatero")]
    if "RETANGULO" in classes and fig_local is None and retangulo_ambiguo:
        return None
    resolvidos = [_resolver_retangulo(r, fig_local or figura) for r in classes]
    if None in resolvidos:
        return None
    if len(resolvidos) >= 2:
        # Conjunção só entre eixos DIFERENTES do triângulo ("Isósceles e
        # retângulo", T-9H17-0008). "Quadrados e retângulos" ou "Equilátero,
        # isósceles, escaleno" são LISTAS de nomes, que o modelo não representa.
        eixos = [EIXO.get(r) for r in resolvidos]
        if None in eixos or len(set(eixos)) != len(eixos):
            return None
    fatos = _extrair_fatos(t)
    resto = fatos.residuo
    for a, b, _r in rotulos:
        resto = _apagar(resto, a, b)
    resto = _STOP_ALTERNATIVA.sub(" ", resto)
    # Qualquer sobra (letra de entidade "Triângulo A", número "Retângulo: 15
    # cm", palavra inventada "Sesquilátero") -> não é um rótulo puro.
    if re.search(r"[a-z]{3,}|\b[a-z]\b|\d", resto, _A):
        return None
    if not rotulos and fatos.vazio():
        return None
    fig = fig_local or figura
    S, motivos, pend, fora = _estados(fatos, fig, leitura, resolvidos)
    if fatos.negacao or pend or fora:
        return None
    if fig_local:
        S = S & FIGURA[fig_local]
    if resolvidos:
        grupos = {EIXO.get(r, "quad") for r in resolvidos}
    elif not fatos.vazio():
        eixos = {r[4] for r in fatos.restricoes}
        if fatos.lados or fatos.lados_todos:
            eixos.add("lados")
        if fatos.angulos or fatos.angulos_todos:
            eixos.add("angulos")
        grupos = {"quad"} if fig == "quad" else eixos
    else:
        grupos = {"figura"}
    grupo = grupos.pop() if len(grupos) == 1 else "misto"
    return S, grupo, resolvidos


def _valor(S, A):
    if S <= A:
        return "V"
    if not (S & A):
        return "F"
    return "I"


# ===========================================================================
# 9. ENTIDADES (várias figuras no mesmo enunciado)
# ===========================================================================
# Identificador: substantivo + letra/dígito SEGUIDO de pontuação ou verbo
# ("barraca A tem", "Triângulo 1:"). Sem essa exigência, "comprou 3 bandeiras"
# e "tem 2 lados" viravam entidades (defeito real do protótipo).
# Romanos ("figura I ... figura II") também identificam: sem eles os ângulos das
# duas figuras se somavam e davam falsa premissa impossível.
_NOME_ENTIDADE = re.compile(
    r"\b([A-Za-z]{3,}) ([A-H]|[1-9]|I{1,3}|IV|V)\b(?! ?=)(?!\))"
    r"(?= ?(?:[:,;.-]|tem\b|eh\b|possui\b|apresenta\b|mede\b|com\b|e\b|$))", _A)
_NAO_E_ENTIDADE = {"lado", "lados", "angulo", "angulos", "vertice", "vertices", "ponto",
                   "pontos", "alternativa", "alternativas", "letra", "opcao", "item",
                   "segmento", "questao", "ano", "serie", "medida", "medidas"}
# "Os ângulos internos DE UM triângulo somam 180°" é referência genérica, não
# uma segunda figura: o artigo indefinido depois de "de/qualquer/todo/cada" não
# abre entidade nova (o caso fazia um enunciado de uma figura só virar dois).
_MARCA_INDEFINIDA = _compilar(r"\b(?:(de|qualquer|todo|cada) )?(?:um|uma|outro|outra) "
                              r"(?:triangulo|quadrilatero)\b")
# "o primeiro tem ... o segundo tem ..." separa figuras; "o primeiro LADO" não.
_ORDINAL = _compilar(r"\b(?:uma|um) (?:eh|tem)\b|\b(um deles|uma delas|outra|outro)(?: eh| tem)?\b|"
                     r"\b(?:a|o) (primeir|segund|terceir|quart|quint)[oa]\b"
                     r"(?! (?:lado|angulo|vertice|ponto|par)s?\b)")
_INDEFINIDO_REPETIDO = _compilar(r"\b(?:um|uma) ([a-z]{4,})\b")
_NOMES_GENERICOS = {"angulo", "lado", "vertice", "ponto", "triangulo", "quadrilatero", "medida",
                    "figura", "forma", "classificacao", "alternativa", "vez", "parte"}
# Texto que fala de VÁRIAS figuras: contradição que só aparece COMBINANDO fatos
# pode ser falha de segmentação ("Duas bandeiras têm 2 lados iguais... A
# terceira tem todos os lados iguais", G-9H17-0036), não premissa impossível.
# "os outros dois ângulos", "o terceiro lado" falam da MESMA figura e não contam
# (G-9H17-0110 é premissa impossível legítima com "os outros ângulos").
_CHEIRO_MULTI = _compilar(
    r"\b(?:duas|dois|tres|quatro|cinco|seis|varios|varias|2|3|4|5|6) "
    r"(?!lados\b|angulos\b|pares\b|vertices\b|medidas\b|pontos\b|partes\b|vezes\b)[a-z]+s\b|"
    r"\b(?:outr[oa]s?|terceir[oa]|quart[oa]|segund[oa])\b"
    r"(?! (?:dois|duas|tres|\d|angulos?|lados?|medem|medindo|mede|medida)\b)|"
    r"\b(?:um deles|uma delas|seguintes|respectivamente)\b|"
    r"\b[a-z]+ (?:[a-h1-9]|i{1,3}|iv|v) ?[:-]")
# "Qual dos seguintes tipos de quadrilátero...?" fala das ALTERNATIVAS, não de
# várias figuras. Sem tirar isto, "seguintes" rebaixava uma premissa impossível
# já detectada a nao_aplicavel (revisão adversarial de 2026-10-01: "jardim com
# 4 lados iguais, um ângulo obtuso e 4 ângulos retos").
_SEGUINTES_DA_PERGUNTA = _compilar(r"\bqua(?:l|is) (?:d[oa]s )?seguintes\b")


def _cheiro_multi(en):
    return _CHEIRO_MULTI.search(_SEGUINTES_DA_PERGUNTA.sub(" ", en))


def _entidades_nomeadas(corpo_caixa):
    nomes = []
    for m in _NOME_ENTIDADE.finditer(corpo_caixa):
        sub = m.group(1).lower()
        ident = m.group(2)
        if sub in _NAO_E_ENTIDADE or (ident.isalpha() and not ident.isupper()):
            continue
        if [r for _a, _b, r in _rotulos_no_texto(sub) if r not in ("triangulo", "quadrilatero")]:
            continue  # "Escaleno B" é alternativa copiada, não entidade
        nomes.append((m.start(), f"{sub} {ident.lower()}"))
    if len({n for _p, n in nomes}) < 2:
        return None
    segs = {}
    for i, (p, n) in enumerate(nomes):
        fim = nomes[i + 1][0] if i + 1 < len(nomes) else len(corpo_caixa)
        segs[n] = segs.get(n, "") + " " + corpo_caixa[p:fim]
    return segs, corpo_caixa[:nomes[0][0]]


def _entidades_indefinidas(corpo):
    """"um triângulo com ... e um triângulo com ..." (R1-Q5) e "uma é ...,
    outra é ..., a terceira ..." (R1-Q10)."""
    marcas = [m.span() for m in _MARCA_INDEFINIDA.finditer(corpo) if not m.group(1)]
    contagem = {}
    for m in _INDEFINIDO_REPETIDO.finditer(corpo):
        contagem[m.group(1)] = contagem.get(m.group(1), 0) + 1
    for nome, n in contagem.items():
        if n >= 2 and nome not in _NOMES_GENERICOS:
            marcas += [m.span() for m in re.finditer(r"\b(?:um|uma) " + nome + r"\b", corpo, _A)]
    ordinais = list(_ORDINAL.finditer(corpo))
    if len(ordinais) >= 2 and any(m.group(1) or m.group(2) for m in ordinais):
        marcas += [m.span() for m in ordinais]
    unidas = []
    for a, b in sorted(marcas):
        if unidas and a < unidas[-1][1] + 12:  # "uma é um quadrilátero" = uma marca só
            continue
        unidas.append((a, b))
    if len(unidas) < 2:
        return None
    segs = {}
    for i, (a, _b) in enumerate(unidas):
        fim = unidas[i + 1][0] if i + 1 < len(unidas) else len(corpo)
        segs[f"#{i + 1}"] = corpo[a:fim]
    return segs, corpo[:unidas[0][0]]


# Rótulo citado no corpo como FATO: "a quarta é um losango", "foi classificado
# como retângulo", "tem formato de quadrado".
_ROTULO_FATO = _compilar(
    r"\b(?:eh|sao|seja|como|formato de|forma de|em forma de|um|uma) (?:um |uma )?"
    r"(?:triangulo |quadrilatero )?(trapezios? isos?celes|trapezios? retangulos?|equilater\w*|"
    r"isos?celes|escalen\w*|acutangul\w*|obtusangul\w*|quadrad\w*|losango\w*|paralelogram\w*|"
    r"trapezio\w*|retangul\w*)")


# "em forma/formato de X" dá a forma de UM objeto ("um pátio em forma de
# quadrado"); se OUTRO objeto aparece depois ("No pátio, os alunos pintaram UM
# QUADRILÁTERO com..."), a figura perguntada pode ser a outra e X não é dado
# dela. Lido como fato, reprovava questão CORRETA: 7/7 sondas da revisão
# adversarial de 2026-10-01 (pátio, caderno, lona, terreno, loja de pisos). O
# rótulo descartado fica no resíduo e bloqueia a completude (na dúvida, nada de
# "dados insuficientes"). Partes da própria figura (ângulo, lado...) não contam
# como outro objeto.
_FORMA_DE = _compilar(r"\bforma(?:to)? de\b")
_OUTRO_OBJETO = _compilar(
    r"\b(?:um|uma) (?!(?:angulos?|lados?|vertices?|pontos?|pares?|par|das|dos|de|vez|medidas?|"
    r"unic[oa]|so|pouco|total)\b)[a-z]{3,}")


def _rotulos_fato(f, texto, figura, limite=None):
    """Rótulos citados como dado. Só no CORPO (até `limite`): na pergunta o
    rótulo é o que se pergunta ("Esse triângulo é retângulo, acutângulo ou
    obtusângulo?"), e tomá-lo como dado inventaria um fato."""
    rotulos = []
    for m in _ROTULO_FATO.finditer(texto):
        if limite is not None and m.start() >= limite:
            continue
        if _FORMA_DE.search(m.group(0)) and _OUTRO_OBJETO.search(texto[m.end():]):
            continue
        if _NEGACAO.search(texto[max(0, m.start() - 25):m.start()]):
            f.negacao = True
            continue
        rs = _rotulos_no_texto(m.group(1))
        if not rs:
            continue
        r = _resolver_retangulo(rs[0][2], figura)
        if r and _FAMILIA.get(r) == figura:
            rotulos.append(r)
            f.consumido.append(m.span())
            f.residuo = _apagar(f.residuo, m.start(), m.end())
    return rotulos


# ===========================================================================
# 10. AVALIAÇÃO DE UMA LEITURA
# ===========================================================================
def _cortar_alternativas_copiadas(t):
    """Remove "A) Escaleno B) Isósceles ..." copiado no enunciado (R2-Q9)."""
    m = re.search(r"\s[aA]\)\s", t)
    if m and re.search(r"\s[bB]\)\s", t[m.end():]):
        return t[:m.start()]
    return t


def _avaliar(q, leitura):
    alts_orig = q["alternativas"]
    gab = q["resposta_correta"]
    caixa = _reescrever(_cortar_alternativas_copiadas(_normalizar(q.get("enunciado", ""))))
    en = caixa.lower()
    corpo, perg = _separar_pergunta(en)
    eixo = _eixo(perg) if perg else None
    alts = {k: _minusculas(_normalizar(v)) for k, v in alts_orig.items()}
    det = {"leitura": leitura, "eixo": eixo}
    existencia = bool(_EXISTENCIA.search(en))
    if _PERGUNTA_NEGATIVA.search(perg or en):
        return "nao_aplicavel", dict(det, motivo="pergunta_negativa")

    # Entidades vivem no CORPO: a pergunta ("Quais desses canteiros são
    # retângulos?") não pode emprestar figura à última entidade (R2-Q4).
    corpo_caixa = caixa
    if perg:
        p0 = en.find(perg)
        if p0 >= 0:
            corpo_caixa = caixa[:p0] + " " * len(perg) + caixa[p0 + len(perg):]
    ents = _entidades_nomeadas(corpo_caixa)
    multi = None
    if ents:
        segs, prefixo = ents
        multi, prefixo = {n: t.lower() for n, t in segs.items()}, prefixo.lower()
    else:
        e2 = _entidades_indefinidas(corpo)
        if e2:
            multi, prefixo = e2
    fig_pergunta = _figura_explicita(perg) if perg else None
    if multi:
        return _avaliar_entidades(leitura, multi, prefixo, perg, fig_pergunta, alts, gab, det,
                                  existencia, eixo)

    # ----- uma figura só
    # Triângulo E quadrilátero nomeados no corpo, ou corpo e pergunta falando
    # de figuras diferentes, é figura DERIVADA ("um quadrado dividido pela
    # diagonal: que triângulo se forma?") ou texto com várias figuras: os fatos
    # do quadrado seriam aplicados ao triângulo e o gabarito certo viraria F.
    if len(_familias_citadas(corpo)) == 2:
        return "nao_aplicavel", dict(det, motivo="triangulo_e_quadrilatero_no_mesmo_texto")
    fig_corpo = _figura_explicita(corpo)
    if fig_corpo and fig_pergunta and fig_corpo != fig_pergunta:
        return "nao_aplicavel", dict(det, motivo="figura_do_corpo_difere_da_pergunta")
    f_corpo = _extrair_fatos(corpo)
    f_perg = _extrair_fatos(perg) if perg else None
    figura = fig_corpo or fig_pergunta
    familias = set()
    for t in alts.values():
        for _a, _b, r in _rotulos_no_texto(t):
            if r != "RETANGULO":
                familias.add(_FAMILIA[r])
    if not figura and len(familias) == 1:
        figura = next(iter(familias))
    if not figura:
        figura = _figura_por_medidas(f_corpo) or (f_perg and _figura_por_medidas(f_perg))
    if not figura:
        return "nao_aplicavel", dict(det, motivo="figura_nao_identificada")
    det["figura"] = figura

    rot_corpo = _rotulos_fato(f_corpo, corpo, figura)
    corpo_tem_fatos = not f_corpo.vazio() or bool(rot_corpo)
    perg_tem_fatos = f_perg is not None and not f_perg.vazio()
    forma_reversa = perg_tem_fatos and bool(_REVERSO.search(perg))
    sugerido = None if eixo else _eixo_sugerido(corpo)
    ctx = (leitura, figura, eixo, alts, gab, existencia, en, perg, sugerido)
    if forma_reversa and not corpo_tem_fatos:
        # "Qual das alternativas mostra um quadrilátero com dois ângulos
        # obtusos?" (R2-Q6): a propriedade está na PERGUNTA e as alternativas
        # são candidatas a tê-la.
        return _avaliar_modo(ctx, det, f_perg, [], [f_corpo], reverso=True)
    if perg_tem_fatos and not corpo_tem_fatos:
        # "Triângulo com ângulos 80°, 60° e 40° — qual é o tipo?": a frase
        # interrogativa CARREGA os dados; é modo direto (G-9H17-0091).
        return _avaliar_modo(ctx, det, f_perg, [], [f_corpo], reverso=False)
    if perg_tem_fatos:
        # "Como os lados são diferentes entre si, qual é o tipo...?" (R2-Q1): a
        # oração da pergunta é mais um FATO da mesma figura.
        texto = corpo + " . " + perg
        fonte = _extrair_fatos(texto)
        rotulos = _rotulos_fato(fonte, texto, figura, limite=len(corpo))
        direto = _avaliar_modo(ctx, dict(det), fonte, rotulos, [], reverso=False)
        if not forma_reversa:
            return direto
        # "Vendeu produtos em formato de retângulo. Qual dos seguintes é um
        # exemplo de quadrilátero com todos os lados iguais?" (G-9H17-0002):
        # o corpo pode ser só contexto (pergunta reversa) ou a mesma figura
        # (fatos somados). Só vale veredito em que as duas leituras concordam.
        reverso = _avaliar_modo(ctx, dict(det), f_perg, [], [f_corpo], reverso=True)
        if direto[0] == reverso[0] or (direto[0] in GEO_REJEITA and reverso[0] in GEO_REJEITA):
            return direto
        return "nao_aplicavel", dict(det, motivo="pergunta_reversa_ou_direta",
                                     modos=[direto[0], reverso[0]])
    return _avaliar_modo(ctx, det, f_corpo, rot_corpo, [f_perg] if f_perg is not None else [],
                         reverso=False)


def _avaliar_modo(ctx, det, fonte, rotulos, extras, reverso):
    """Avalia uma figura só com os fatos `fonte` (+ rótulos citados como dado).
    `extras` são trechos sem fatos que ainda contam para a completude."""
    leitura, figura, eixo, alts, gab, existencia, en, perg, sugerido = ctx
    S, motivos, pend, fora = _estados(fonte, figura, leitura, rotulos)
    det["modo"] = "reverso" if reverso else "direto"
    det["fatos"] = fonte.descricoes() + ["rótulo: " + NOME_ROTULO[r] for r in rotulos]
    det["n_estados"] = len(S)
    if fora:
        return "nao_aplicavel", dict(det, motivo="quadrilatero_nao_convexo")
    if not S:
        if _NAO_EXISTE.search(alts.get(gab, "")):
            # Questão de condição de existência (9º H16): a impossibilidade É o
            # assunto, e o gabarito que a afirma está certo. O enunciado nem
            # sempre diz "existe"/"é possível formar": "Um triângulo tem dois
            # ângulos retos. Qual é a classificação?" com gabarito "Esse
            # triângulo não existe" é questão CORRETA, e era reprovada como
            # premissa impossível (11/11 sondas da revisão adversarial de
            # 2026-10-01; em modo ativo virava "falha" e pesava no G2).
            return "ok", dict(det, motivo="existencia_negada_corretamente")
        if existencia:
            return "nao_aplicavel", dict(det, motivo="pergunta_de_existencia")
        if _NDA.search(alts.get(gab, "")) or _SIM_NAO.search(alts.get(gab, "")):
            # "Qual triângulo possui dois ângulos obtusos?" -> "Nenhuma das
            # alternativas", ou "Pode ter dois obtusos?" -> "Não, pois...": a
            # resposta já recusa a figura; não há classificação a conferir.
            return "nao_aplicavel", dict(det, motivo="gabarito_recusa_a_figura")
        if any(m.startswith("combinacao") for m in motivos) and _cheiro_multi(en):
            return "nao_aplicavel", dict(det, motivo="contradicao_possivelmente_entre_figuras",
                                         motivos=motivos)
        numerico = any(m.startswith(("soma_dos_angulos", "desigualdade", "angulos_conhecidos",
                                     "angulo_")) for m in motivos)
        if numerico and len(fonte.frases_medidas) > 1 and _cheiro_multi(en):
            # "A barraca tem dois ângulos de 50°. Já a lona da OUTRA barraca tem
            # um de 100°": medidas de objetos diferentes, somadas como se
            # fossem da mesma figura, dariam falsa premissa impossível.
            return "nao_aplicavel", dict(det, motivo="medidas_possivelmente_de_figuras_diferentes",
                                         motivos=motivos)
        return "premissa_impossivel", dict(det, motivo="premissa_impossivel", motivos=motivos)
    if existencia:
        return "nao_aplicavel", dict(det, motivo="pergunta_de_existencia")

    comp, pendencia = _completude(fonte, pend)
    for extra in extras:
        if comp:
            comp, pendencia = _completude(extra, [])
    det["completo"] = comp
    if pendencia:
        det["pendencia"] = pendencia

    ret_amb = _retangulo_ambiguo(alts.values(), figura)
    masks, grupos, vals = {}, {}, {}
    for k, t in alts.items():
        if _NDA.search(t):
            vals[k] = "NDA"
            continue
        if _INDET.search(t):
            vals[k] = "INDET"
            continue
        mm = _mascara_alternativa(t, figura, leitura, ret_amb)
        if mm is None:
            vals[k] = "?"
            continue
        A, grupo, rs = mm
        masks[k], grupos[k] = A, grupo
        if eixo and figura == "tri" and rs and all(r in EIXO for r in rs):
            fora_eixo = [r for r in rs if EIXO[r] != eixo]
            if fora_eixo and len(fora_eixo) == len(rs):
                vals[k] = "F"  # respondeu no eixo errado ("classifique pelo ângulo": "Equilátero")
                det.setdefault("fora_do_eixo", []).append(k)
                if not reverso:
                    # Valor REAL da alternativa como afirmação sobre a figura,
                    # ignorando o eixo pedido. Não muda o veredito (convenção 5);
                    # serve à D5 do usuário (verificar_d5): no 6-6-7 "quanto
                    # aos lados", "Acutângulo" é verdadeiro por outro eixo.
                    det.setdefault("valores_fora_do_eixo", {})[k] = _valor(S, A)
                continue
            if fora_eixo:
                vals[k] = "?"  # conjunção que mistura o eixo pedido com o outro
                continue
        if reverso:
            # V: TODA figura do rótulo tem a propriedade; I: alguma tem; F: nenhuma.
            Af = A & FIGURA[figura]
            vals[k] = "F" if not Af else ("V" if Af <= S else ("I" if Af & S else "F"))
        else:
            vals[k] = _valor(S, A)
    modal = reverso and bool(_MODAL.search(perg))
    if reverso:
        # No modo reverso NÃO há "mais específica": "Qual quadrilátero tem os
        # lados iguais?" tem Losango E Quadrado verdadeiros, e ambos respondem
        # (G-9H17-0070). Dominância por inclusão aqui inverteria o sentido.
        masks, grupos = {}, {}
    return _decidir(vals, gab, masks, grupos, comp, det, reverso, modal, alts,
                    eixo_sugerido=sugerido if figura == "tri" else None)


def _avaliar_entidades(leitura, multi, prefixo, perg, fig_pergunta, alts, gab, det, existencia,
                       eixo):
    det["entidades"] = list(multi)
    fig_prefixo = _figura_explicita(prefixo)
    info = {}
    for nome, texto in multi.items():
        f = _extrair_fatos(texto)
        fig = _figura_explicita(texto) or fig_prefixo or _figura_por_medidas(f) or fig_pergunta
        rotulos = _rotulos_fato(f, texto, fig) if fig else []
        S, motivos, pend, fora = _estados(f, fig, leitura, rotulos)
        info[nome] = (f, fig, S, pend, fora)
        if fig and not S:
            if existencia or _NAO_EXISTE.search(alts.get(gab, "")):
                return "nao_aplicavel", dict(det, motivo="pergunta_de_existencia")
            return "premissa_impossivel", dict(det, motivo="premissa_impossivel", entidade=nome,
                                               fatos=f.descricoes(), motivos=motivos)
    nomes = [n for n in multi if not n.startswith("#")]
    if not nomes:
        return _equivalencia_ou_na(alts, gab, fig_pergunta or fig_prefixo, leitura, det,
                                   "varias_figuras_sem_nome")
    # Referência curta "A: escaleno; B: isósceles" (G-9H17-0111) só quando todas
    # as figuras têm o MESMO substantivo ("triângulo A/B/C"): aí "A:" não é ambíguo.
    substantivos = {n.rsplit(" ", 1)[0] for n in nomes}
    ident = {n.rsplit(" ", 1)[1]: n for n in nomes} if len(substantivos) == 1 else {}
    vals = {}
    for k, t in alts.items():
        if _NDA.search(t):
            vals[k] = "NDA"
            continue
        pos = []
        for n in nomes:
            for m in re.finditer(r"\b" + re.escape(n) + r"\b", t, _A):
                pos.append((m.start(), m.end(), n))
        if len(pos) < 2 and ident:
            pos = [(m.start(1), m.end(1), ident[m.group(1)])
                   for m in re.finditer(r"(?:^|[;,.] ?)([a-h1-9]) ?[:)-]", t, _A)
                   if m.group(1) in ident]
        pos.sort()
        if len(pos) < 2:
            return _equivalencia_ou_na(alts, gab, fig_pergunta or fig_prefixo, leitura, det,
                                       "selecao_de_entidade")
        afirmacoes = []
        for i, (_a, b, n) in enumerate(pos):
            fim = pos[i + 1][0] if i + 1 < len(pos) else len(t)
            # "a barraca A é um triângulo agudo, a barraca B ..." -> "um triângulo
            # agudo". Os cortes exigem separador: sem ele, "(eh|e)?" comia o "e"
            # de "escaleno" e o artigo final comia o "o" de "retângulo".
            trecho = t[b:fim].strip()
            trecho = re.sub(r"^[\s:,;-]*(?:(?:eh|sao|tem|e)\s+)?", "", trecho, flags=_A)
            trecho = _cortar_sufixo(trecho)
            f_n, fig_n, S_n, pend_n, fora_n = info[n]
            if not fig_n or pend_n or fora_n:
                return "nao_aplicavel", dict(det, motivo="entidade_incompleta", entidade=n)
            mm = _mascara_alternativa(trecho, fig_n, leitura)
            if mm is None:
                return "nao_aplicavel", dict(det, motivo="trecho_composto_nao_mapeado",
                                             trecho=trecho)
            A, _grupo, rs = mm
            if eixo and fig_n == "tri" and rs and all(r in EIXO for r in rs) \
                    and any(EIXO[r] != eixo for r in rs):
                afirmacoes.append("F" if all(EIXO[r] != eixo for r in rs) else "?")
                continue
            afirmacoes.append(_valor(S_n, A))
        if "F" in afirmacoes:
            vals[k] = "F"
        elif "?" in afirmacoes:
            vals[k] = "?"
        elif all(a == "V" for a in afirmacoes):
            vals[k] = "V"
        else:
            vals[k] = "I"
    completo = all(_completude(f_n, pend_n)[0] for f_n, _fig, _S, pend_n, _fo in info.values())
    for trecho in (prefixo, perg):
        if completo and trecho:
            completo = _completude(_extrair_fatos(trecho), [])[0]
    det["completo"] = completo
    return _decidir(vals, gab, {}, {}, completo, det, False, False, alts)


_SEPARADORES = " \t\n\r\f\v,;"


def _cortar_sufixo(t):
    """Tira o artigo que sobra no fim do trecho ("..., a", "e o") e a
    pontuação final. Equivale a re.sub(r"(?:[\\s,;]+(?:e\\s+)?(?:a|o|as|os))?
    [\\s,;.]*$", "", t), mas em tempo LINEAR: aquela regex é quadrática em
    sequência de espaços (que _apagar cria), e o Hermes do app não tem JIT de
    regex (20 mil espaços: 416 ms no Node 24, revisão de 2026-10-01)."""
    fim = len(t)
    while fim and t[fim - 1] in _SEPARADORES + ".":
        fim -= 1
    corpo = t[:fim]
    i = fim
    while i and corpo[i - 1] not in _SEPARADORES:
        i -= 1
    if 0 < i and corpo[i:] in ("a", "o", "as", "os"):
        j = i
        while j and corpo[j - 1] in " \t\n\r\f\v":
            j -= 1
        if j >= 1 and corpo[j - 1] == "e" and (j == 1 or corpo[j - 2] in _SEPARADORES) and j < i:
            k = j - 1
            if k and corpo[k - 1] in _SEPARADORES:
                while k and corpo[k - 1] in _SEPARADORES:
                    k -= 1
                return corpo[:k]
        k = i
        while k and corpo[k - 1] in _SEPARADORES:
            k -= 1
        return corpo[:k]
    return corpo


def _equivalencia_ou_na(alts, gab, figura, leitura, det, motivo):
    """Sem como avaliar as figuras uma a uma, ainda dá para pegar duas
    alternativas que dizem a MESMA coisa ("Quadrilátero com todos os lados
    iguais" e "Losango", R1-Q10): a questão não tem resposta única."""
    if gab in alts and figura:
        masks = {}
        amb = _retangulo_ambiguo(alts.values(), figura)
        for k, t in alts.items():
            if _NDA.search(t) or _INDET.search(t):
                continue
            mm = _mascara_alternativa(t, figura, leitura, amb)
            if mm and mm[0]:
                masks[k] = mm[0]
        if gab in masks:
            eq = sorted(k for k in masks if k != gab and masks[k] == masks[gab])
            if eq:
                return "nao_unica", dict(det, motivo="alternativas_equivalentes",
                                         equivalentes=[gab] + eq)
    return "nao_aplicavel", dict(det, motivo=motivo)



# ===========================================================================
# 11. DECISÃO
# ===========================================================================
def _valor_nda(outras, completo):
    """'Nenhuma das anteriores' é V só com TODAS as outras F (R1-Q6: trapézio
    possível com 4 lados diferentes basta para ela não valer)."""
    if "V" in outras:
        return "F"
    if outras and all(o == "F" for o in outras):
        return "V"
    if "I" in outras and completo:
        # Uma outra alternativa I vale em parte dos estados e, nesses, a NDA é
        # falsa: a NDA nunca é GARANTIDA. Uma alternativa não mapeada só
        # poderia torná-la F, nunca V (R2-Q7: "Acrisângulo" ao lado de
        # Equilátero I). "I" aqui significa "não garantida".
        return "I"
    return "?"


def _decidir(vals, gab, masks, grupos, completo, det, reverso, modal, alts,
             eixo_sugerido=None):
    if gab not in vals:
        return "nao_aplicavel", dict(det, motivo="sem_gabarito", valores=dict(vals))
    mapeadas = [k for k, v in vals.items() if v in ("V", "F", "I")]
    if len(mapeadas) < 2:
        return "nao_aplicavel", dict(det, motivo="poucas_alternativas_mapeadas",
                                     valores=dict(vals))
    desconhecidas = [k for k, v in vals.items() if v == "?"]
    verdadeiras = [k for k, v in vals.items() if v == "V"]
    if reverso and modal:
        # "Qual quadrilátero PODE ter dois ângulos obtusos?": quem pode (V ou I)
        # responde; a questão é única se só uma alternativa pode.
        verdadeiras = [k for k, v in vals.items() if v in ("V", "I")]

    def domina(j, k):
        """j (mais específica) domina k só no MESMO eixo, ou se k é figura pura."""
        if not (j in masks and k in masks and masks[j] < masks[k]):
            return False
        return grupos.get(k) == "figura" or grupos.get(j) == grupos.get(k)

    especificas = [k for k in verdadeiras if not any(j != k and domina(j, k) for j in verdadeiras)]
    det["mais_especificas"] = especificas

    # Valor das alternativas "Nenhuma das anteriores" e "não é possível determinar",
    # calculado sobre uma cópia (duas NDA no mesmo item não se contaminam).
    base = dict(vals)
    letras = sorted(base)
    for k in letras:
        outras = [base[j] for j in letras if j != k and base[j] not in ("NDA", "INDET")]
        if base[k] == "NDA":
            valor = _valor_nda(outras, completo)
            if "anterior" in alts.get(k, ""):
                # "anteriores" é posicional: se a NDA não é a última, as duas
                # leituras (todas as outras x só as anteriores) têm de concordar.
                antes = [base[j] for j in letras if j < k and base[j] not in ("NDA", "INDET")]
                if _valor_nda(antes, completo) != valor:
                    valor = "?"
            vals[k] = valor
        elif base[k] == "INDET":
            if "V" in outras:
                vals[k] = "F"
            elif "I" in outras and "?" not in outras and completo:
                vals[k] = "V"
            else:
                vals[k] = "?"
    det["valores"] = dict(vals)
    g = vals[gab]

    # Duas alternativas com o MESMO significado (e o gabarito é uma delas).
    if g != "F" and gab in masks and masks[gab]:
        eq = sorted(k for k in masks if k != gab and masks[k] == masks[gab])
        if eq:
            return "nao_unica", dict(det, motivo="alternativas_equivalentes",
                                     equivalentes=[gab] + eq)

    if reverso and modal and g in ("V", "I"):
        if len(especificas) > 1:
            return "nao_unica", dict(det, motivo="varias_alternativas_podem_ter_a_propriedade")
        if desconhecidas or (g == "I" and not completo):
            return "nao_aplicavel", dict(det, motivo="ok_nao_robusto")
        return "ok", dict(det, motivo="unica_que_pode_ter_a_propriedade")

    if g == "V" and gab not in verdadeiras:
        # gabarito "Nenhuma das anteriores"/"não é possível determinar" verdadeiro
        if verdadeiras:
            return "nao_unica", dict(det, motivo="nda_e_outra_verdadeira")
        return "ok", dict(det, motivo="nda_garantida")
    if g == "V":
        if len(especificas) > 1:
            eixos = {grupos.get(k) for k in especificas}
            if eixo_sugerido and {"lados", "angulos"} <= eixos:
                # "Os triângulos podem ser equiláteros, isósceles ou escalenos.
                # ... Como ele é classificado?": o corpo já fixou o eixo; a
                # pergunta não repetir não torna a questão ambígua para o aluno.
                return "nao_aplicavel", dict(det, motivo="eixo_sugerido_fora_da_pergunta",
                                             eixo_sugerido=eixo_sugerido)
            return "nao_unica", dict(det, motivo="duas_classificacoes_garantidas")
        if gab not in especificas:
            return "nao_unica", dict(det, motivo="gabarito_menos_especifico")
        if desconhecidas:
            return "nao_aplicavel", dict(det, motivo="alternativa_nao_mapeada",
                                         nao_mapeadas=desconhecidas)
        risco = [k for k, v in vals.items() if k != gab and v == "I"]
        if risco and not completo:
            return "nao_aplicavel", dict(det, motivo="ok_nao_robusto", risco=risco)
        return "ok", dict(det, motivo="gabarito_garantido_e_mais_especifico")
    if g == "F":
        candidatas = especificas + [k for k, v in vals.items() if v == "V" and k not in verdadeiras]
        if len(candidatas) == 1:
            return "gabarito_errado", dict(det, motivo="gabarito_falso_outra_garantida",
                                           sugestao=candidatas[0])
        if len(candidatas) > 1:
            return "nao_unica", dict(det, motivo="gabarito_falso_e_varias_garantidas")
        return "gabarito_errado", dict(det, motivo="gabarito_falso")
    if g == "I":
        if not completo:
            return "nao_aplicavel", dict(det, motivo="gabarito_indeterminado_parse_incompleto")
        if len(especificas) == 1:
            return "gabarito_errado", dict(det, motivo="gabarito_nao_garantido_outra_garantida",
                                           sugestao=especificas[0])
        if reverso and any(v == "I" for k, v in vals.items() if k != gab and base[k] != "NDA"):
            return "nao_unica", dict(det, motivo="varias_alternativas_podem_ter_a_propriedade")
        return "dados_insuficientes", dict(det, motivo="dados_nao_determinam_o_gabarito")
    return "nao_aplicavel", dict(det, motivo="gabarito_nao_mapeado")


# ===========================================================================
# 12. API PÚBLICA
# ===========================================================================
_EXPLICACAO = {
    "gabarito_garantido_e_mais_especifico": "o gabarito é garantido pelos dados e é a classificação mais específica",
    "nda_garantida": "nenhuma das outras alternativas é possível, então 'nenhuma das anteriores' é garantida",
    "existencia_negada_corretamente": "a figura descrita não existe e o gabarito diz exatamente isso",
    "unica_que_pode_ter_a_propriedade": "só a alternativa do gabarito pode ter a propriedade pedida",
    "premissa_impossivel": "o enunciado descreve uma figura que não existe",
    "gabarito_falso_outra_garantida": "o gabarito é impossível pelos dados; outra alternativa é a garantida",
    "gabarito_falso": "o gabarito é impossível pelos dados",
    "gabarito_falso_e_varias_garantidas": "o gabarito é impossível e mais de uma outra alternativa é garantida",
    "gabarito_nao_garantido_outra_garantida": "os dados não garantem o gabarito, mas garantem outra alternativa",
    "duas_classificacoes_garantidas": "mais de uma alternativa é garantida (eixos diferentes ou ramos da hierarquia)",
    "gabarito_menos_especifico": "há uma alternativa garantida mais específica que o gabarito",
    "alternativas_equivalentes": "duas alternativas descrevem a mesma classe de figuras",
    "nda_e_outra_verdadeira": "'nenhuma das anteriores' e outra alternativa são verdadeiras",
    "varias_alternativas_podem_ter_a_propriedade": "mais de uma alternativa pode ter a propriedade pedida",
    "dados_nao_determinam_o_gabarito": "os dados do enunciado não determinam a classificação",
}


def _explicar(veredito, det):
    motivo = det.get("motivo", "")
    base = _EXPLICACAO.get(motivo, motivo.replace("_", " "))
    extra = []
    if det.get("motivos"):
        extra.append("; ".join(det["motivos"]))
    if det.get("sugestao"):
        extra.append(f"alternativa garantida: {det['sugestao']}")
    if det.get("equivalentes"):
        extra.append("equivalentes: " + ", ".join(det["equivalentes"]))
    if det.get("mais_especificas") and veredito == "nao_unica":
        extra.append("garantidas: " + ", ".join(det["mais_especificas"]))
    return base + (" (" + " | ".join(extra) + ")" if extra else "")


def verificar_geometria(questao):
    """Verifica uma questão de classificação de triângulos/quadriláteros.

    Devolve (veredito, detalhe). veredito é um de VEREDITOS:
      "ok"                  gabarito garantido pelos dados, único e o mais específico;
      "gabarito_errado"     gabarito impossível ou não garantido; `sugestao` traz a
                            letra garantida quando as duas leituras concordam nela;
      "nao_unica"           mais de uma alternativa defensável (hierarquia, eixos,
                            alternativas equivalentes);
      "premissa_impossivel" o enunciado descreve figura que não existe;
      "dados_insuficientes" os dados não determinam a classificação pedida;
      "nao_aplicavel"       fora do escopo ou parse sem confiança (na dúvida).
    `detalhe` traz: motivo (código estável), explicacao (frase legível), fatos
    extraídos, valores V/F/I/? de cada alternativa, figura, eixo, modo,
    mais_especificas, completo/pendencia e leituras_concordam.

    Nunca levanta exceção: qualquer erro interno vira "nao_aplicavel".
    """
    try:
        return _verificar(questao)
    except Exception as exc:  # noqa: BLE001 - verificador opcional não derruba o pipeline
        return "nao_aplicavel", {"motivo": "erro_interno", "erro": repr(exc),
                                 "explicacao": "erro interno do verificador"}


def _verificar(q):
    if not isinstance(q, dict) or not isinstance(q.get("alternativas"), dict):
        return "nao_aplicavel", {"motivo": "sem_questao", "explicacao": "questão malformada"}
    if q.get("resposta_correta") not in q["alternativas"]:
        return "nao_aplicavel", {"motivo": "sem_gabarito", "explicacao": "sem gabarito"}
    alts = " | ".join(_minusculas(_normalizar(v)) for v in q["alternativas"].values())
    if not _PORTA_ROTULO.search(alts):
        return "nao_aplicavel", {"motivo": "sem_rotulo_nas_alternativas",
                                 "explicacao": "nenhuma alternativa é classificação de figura"}
    en_caixa = _normalizar(q.get("enunciado", ""))
    en = _minusculas(en_caixa)
    if _MILHAR_ESPACO.search(en):
        return "nao_aplicavel", {"motivo": "numero_com_espaco",
                                 "explicacao": "número com espaço (milhar ou lista?) é ambíguo"}
    if len(re.findall(r"[A-Z]", en_caixa)) >= 10 and not re.search(r"[a-z]", en_caixa):
        # Em CAIXA ALTA o artigo "A" e a entidade "barraca A" se confundem: a
        # R1-Q1 (aprovada pela auditoria) em maiúsculas fundia as três barracas
        # numa figura só e saía premissa impossível (revisão de 2026-10-01).
        return "nao_aplicavel", {"motivo": "texto_em_caixa_alta",
                                 "explicacao": "texto todo em maiúsculas: entidades ambíguas"}
    if _FORA_DE_ESCOPO.search(en) or _FORA_DE_ESCOPO.search(alts):
        return "nao_aplicavel", {"motivo": "fora_de_escopo",
                                 "explicacao": "sólidos, planificação, contagem ou círculo"}
    va, da = _avaliar(q, "A")
    vb, db = _avaliar(q, "B")
    if va == vb:
        det = dict(da, leituras_concordam=True)
        if da.get("sugestao") != db.get("sugestao"):
            det["sugestao"] = None
        return va, dict(det, explicacao=_explicar(va, det))
    if va in GEO_REJEITA and vb in GEO_REJEITA:
        # As duas convenções reprovam, por razões diferentes: reprova, mas sem
        # sugerir letra (cada leitura apontaria uma correção diferente).
        det = dict(da, leituras_concordam=False, veredito_leitura_B=vb,
                   motivo_leitura_B=db.get("motivo"), sugestao=None)
        return va, dict(det, explicacao=_explicar(va, det))
    det = {"motivo": "depende_de_convencao", "leituras_concordam": False,
           "leitura_A": [va, da], "leitura_B": [vb, db]}
    return "nao_aplicavel", dict(det, explicacao="as leituras inclusiva e exclusiva divergem "
                                                 f"(A: {va}, B: {vb})")


# ===========================================================================
# 13. D5 DO USUÁRIO: DISTRATOR VERDADEIRO (pré-filtro da injeção)
# ===========================================================================
# Decisão do usuário (2026-10-01, D5): "só o gabarito pode ser verdadeiro, por
# QUALQUER critério". É mais estrita que a convenção 4 deste módulo, que aceita
# a superclasse do mesmo eixo como distrator (R2-Q5). No piloto 2 o 9º H17
# rendeu 1 aceita em 12 candidatas; 8 foram barradas pelos juízes LLM (pagos:
# validador + revisor), quase sempre por isso: "Isósceles" para o equilátero,
# "Paralelogramo" para o losango, "Acutângulo" numa pergunta "quanto aos lados"
# de um triângulo com os três ângulos agudos. Esta função não muda verificar_geometria
# (o app e o gate usam a convenção 4); ela LÊ o detalhe de um veredito "ok" e
# aponta as alternativas verdadeiras além do gabarito:
#   superclasse   alternativa V que não é a mais específica (mesmo eixo, ou a
#                 figura pura: "Quadrilátero" para o quadrado);
#   outro_eixo    alternativa do eixo NÃO perguntado que é V para os dados.
# Leitura usada: a inclusiva (A), a do detalhe devolvido quando as duas
# leituras concordam — é a da BNCC e a do texto da D5 ("isósceles para o
# equilátero"). V é robusto a parse incompleto (fato não lido só encolhe S),
# então um V de alternativa que não é o gabarito vale mesmo quando o veredito
# principal se absteve SÓ por causa de valores I ("ok_nao_robusto",
# "alternativa_nao_mapeada", "gabarito_indeterminado_parse_incompleto": as duas
# leituras concordam e os fatos lidos foram lidos). Medido nas candidatas
# gravadas dos pilotos: das 14 barradas (todas violações reais da D5), 6 vêm
# dessas abstenções — ex.: "quatro lados de 8 cm e dois pares de lados
# paralelos" com "Losango" e "Paralelogramo" ambos V. Nas outras abstenções
# (convenção divergente, pergunta negativa ou reversa, figuras derivadas,
# várias entidades) e nos vereditos que já reprovam, não se pronuncia.
_MOTIVOS_D5_ABSTENCAO = frozenset({"ok_nao_robusto", "alternativa_nao_mapeada",
                                   "gabarito_indeterminado_parse_incompleto"})


def verificar_d5(questao, resultado=None):
    """(True, detalhe) quando há distrator verdadeiro pela D5; (False, None)
    caso contrário. `resultado` = (veredito, detalhe) já calculado por
    verificar_geometria (evita refazer o parse)."""
    try:
        veredito, det = resultado if resultado is not None else verificar_geometria(questao)
        if not isinstance(det, dict) or det.get("modo") != "direto" or det.get("entidades"):
            return False, None
        if veredito != "ok" and not (veredito == "nao_aplicavel" and det.get("leituras_concordam")
                                     and det.get("motivo") in _MOTIVOS_D5_ABSTENCAO):
            return False, None
        gab = questao.get("resposta_correta")
        valores = det.get("valores") or {}
        superclasse = sorted(k for k, v in valores.items() if k != gab and v == "V")
        outro_eixo = sorted(k for k, v in (det.get("valores_fora_do_eixo") or {}).items()
                            if k != gab and v == "V")
        if not superclasse and not outro_eixo:
            return False, None
        alts = questao.get("alternativas") or {}
        partes = []
        if superclasse:
            partes.append("outra classe também verdadeira (mais geral ou do mesmo ramo): "
                          + ", ".join(f"{k} ({alts.get(k)})" for k in superclasse))
        if outro_eixo:
            partes.append("verdadeira por outro critério: "
                          + ", ".join(f"{k} ({alts.get(k)})" for k in outro_eixo))
        return True, {"motivo": "distrator_verdadeiro", "superclasse": superclasse, "outro_eixo": outro_eixo,
                      "gabarito": gab, "eixo": det.get("eixo"), "explicacao": "; ".join(partes)}
    except Exception as exc:  # noqa: BLE001 - pré-filtro opcional nunca derruba o pipeline
        return False, {"motivo": "erro_interno", "erro": repr(exc)}
