"""Agentes LLM da base de conhecimento (Etapa 3): GERADOR, VALIDADOR e REVISOR.

Por que três papéis e não um "gerador cuidadoso": a auditoria humana das 20
questões do 9º H17 (outputs/testes_locais/Log.txt) mostrou que o erro central
— "tem uma propriedade de X, logo é X" (quadrado ⊂ retângulo; 90-45-45 também
é isósceles; 5-7-9 é obtusângulo) — passa intacto pelos filtros
determinísticos, porque check_consistency só confere CONTAS. Pedir ao próprio
autor que "revise" também não resolve (Huang et al. 2023: sem sinal externo o
LLM não se autocorrige e tende a trocar respostas certas). Por isso:

  VALIDADOR (rigor matemático) RESOLVE o item às cegas — sem gabarito, com as
  alternativas permutadas — e julga cada alternativa como V (garantida),
  F (contrariada) ou I (indeterminada). A resposta só é única com exatamente
  uma V e quatro F; isso pega de uma vez hierarquia, dados ausentes, NDA
  indevida e alternativas equivalentes. A fase 2 (resolução do autor) só roda
  quando a resposta cega já coincide com o gabarito: corta custo e impede o
  validador de "concordar" com o autor (viés de autoaprovação, Zheng et al.).

  REVISOR (pedagógico/SAEB, sabia-4-thinking desde 2026-10-01) também
  RESOLVE às cegas — sem gabarito, sem resolução do autor, numa permutação
  própria — e julga habilidade, ano, dificuldade real, distratores
  (estritamente falsos por qualquer critério), clareza, contexto (veta) e
  português. A comparação com o gabarito é feita no código. NUNCA recebe a
  saída do validador (e vice-versa): os juízes ficam independentes.

  O VEREDITO FINAL É CALCULADO AQUI, NO CÓDIGO, a partir dos campos do JSON
  (status V/F/I, critérios), e não aceito do LLM. Falha de API, JSON
  malformado ou truncado NUNCA vira aprovação (fail-closed): na injeção vira
  rejeição; na auditoria, "nao_avaliado" (retenta na próxima rodada).

Custo: toda chamada passa por Orcamento, que aborta ANTES de passar do teto
(--max-chamadas) e registra tokens em outputs/agentes/uso_api.jsonl. Chamadas
que falham também contam — o provedor pode cobrar. A chave da API nunca é
impressa, logada ou gravada: mensagens de erro passam por _sanitizar().

Clientes injetáveis: qualquer objeto com chat_completion(messages, max_tokens,
temperature) devolvendo .choices[0].message.content (e, opcionalmente,
.usage). ClienteSimulado responde sem rede (dry-run); os testes usam roteiros.

Este módulo NÃO edita distill_teacher.py, diversidade.py nem schema_utils.py:
só importa (somente leitura) o que precisa.
"""

import hashlib
import json
import os
import random
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import distill_teacher as dt  # noqa: E402  (carrega o .env; não imprimimos nada dele)
import diversidade  # noqa: E402
from extract_data import SYSTEM_PROMPT  # noqa: E402
from schema_utils import (  # noqa: E402
    ALTERNATIVE_LETTERS,
    DIFFICULTY_MAP,
    IMAGE_PATTERN,
    check_consistency,
    check_structure,
    classe_alternativa,
    depende_de_visual_ausente,
    extract_questao,
    parse_json,
)

# Verificador determinístico de geometria do OUTRO workflow (em construção em
# paralelo). Import opcional: se o arquivo não existe, está pela metade ou a
# assinatura não bate, seguimos sem ele e registramos "indisponivel". Nunca
# reprova por falha do verificador — só por um veredito explícito dele.
try:  # pragma: no cover - depende do estado do outro workflow
    import verificador_geometria as _vg
except Exception:  # noqa: BLE001
    _vg = None

ROOT = Path(__file__).resolve().parent.parent
LOG_USO_PADRAO = ROOT / "outputs" / "agentes" / "uso_api.jsonl"
LETRAS = ALTERNATIVE_LETTERS  # "ABCDE"

# Arquivos que NENHUM script desta etapa pode escrever (regra do projeto).
# _garantir_gravavel() é chamado antes de todo open(..., "w"/"a").
PROTEGIDOS = {
    (ROOT / "data" / n).resolve()
    for n in ("train.jsonl", "train_curado.jsonl", "val.jsonl", "val_frozen_v1.jsonl",
              "val_novos_v1.jsonl", "train_multi.jsonl")
}


def _garantir_gravavel(path):
    """Levanta se `path` é um arquivo protegido (train/val). Vale também para
    qualquer data/val*.jsonl futuro."""
    p = Path(path).resolve()
    if p in PROTEGIDOS or (p.parent == (ROOT / "data").resolve() and p.name.startswith("val")
                           and p.suffix == ".jsonl"):
        raise PermissionError(f"escrita proibida em arquivo protegido: {p.name}")
    return p


def abrir_para_gravar(path, modo="a"):
    p = _garantir_gravavel(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, modo, encoding="utf-8")
    # Arquivo append-only interrompido no meio de uma linha: sem esta quebra,
    # o próximo registro seria colado na linha truncada e viraria JSON
    # inválido — uma questão aceita (já paga) ou um registro de uso perdido
    # (reproduzido na revisão do piloto, /tmp/claude-1000/base/rev_trunc2).
    if "a" in modo and p.stat().st_size > 0:
        with open(p, "rb") as fb:
            fb.seek(-1, os.SEEK_END)
            if fb.read(1) != b"\n":
                f.write("\n")
    return f


def agora_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ===========================================================================
# PROMPTS (PT-BR). Mudar qualquer texto aqui muda VERSAO_PROMPTS, o que
# invalida o cache da auditoria (o registro antigo deixa de valer).
# ===========================================================================

GERADOR_ADDENDUM = """
Regras de qualidade do SAEB/INEP (obrigatórias):
1. Enunciado autocontido: todo dado necessário para responder está escrito no texto. Não cite figura, imagem, desenho ou malha. Se precisar de tabela ou gráfico, escreva os dados no enunciado, um item por linha, no formato "rótulo: valor".
2. Premissa possível: a situação precisa existir matematicamente. Respeite a soma dos ângulos internos (180° no triângulo, 360° no quadrilátero), a desigualdade triangular (o maior lado é menor que a soma dos outros dois), quantidades inteiras quando se contam objetos ou pessoas, porcentagens de um todo até 100% e preços e medidas realistas.
3. Uma única alternativa verdadeira. Antes de escolher o gabarito, julgue CADA alternativa como verdadeira, falsa ou indeterminada usando só os dados do enunciado, por QUALQUER critério e não só pelo critério da pergunta: exatamente uma deve ser verdadeira e as outras quatro, falsas. As classificações da BNCC são inclusivas: todo quadrado é retângulo, losango e paralelogramo; um triângulo pode ser retângulo e isósceles ao mesmo tempo. Ter UMA propriedade de uma figura não prova que ela é essa figura (quatro ângulos retos garantem retângulo, não quadrado; quatro lados iguais garantem losango, não quadrado; lados diferentes dizem que o triângulo é escaleno e nada dizem sobre os ângulos). Em classificação, escreva o critério na pergunta ("quanto aos lados" ou "quanto aos ângulos") e dê os dados que determinam a classe.
4. Alternativas não equivalentes: duas alternativas não podem dizer a mesma coisa de formas diferentes (1/2 e 0,5; 1 h 30 min e 90 min; "losango" e "quadrilátero com os quatro lados iguais").
5. Evite "Nenhuma das alternativas anteriores". Se usar, ela só pode ser o gabarito quando as outras quatro forem comprovadamente falsas com os dados dados.
6. Distratores ESTRITAMENTE FALSOS: cada alternativa errada tem de ser falsa por qualquer critério. Não use como distrator uma classe que é verdadeira por outro critério (numa pergunta "quanto aos lados" sobre um triângulo com os três ângulos agudos, "Acutângulo" é verdadeiro e não pode ser distrator) nem uma classe mais geral que também vale (isósceles para um equilátero; retângulo, losango ou paralelogramo para um quadrado). Cada alternativa errada vem de um erro típico de alunos deste ano (somar em vez de subtrair, esquecer o "vai um", confundir perímetro com área, contar um intervalo a mais, classificar por uma propriedade só). Use o mesmo formato e a mesma unidade da resposta correta e ordens de grandeza parecidas.
7. Linguagem e números do ano pedido: frases curtas e números pequenos nos anos iniciais; números na ordem de grandeza que a descrição da habilidade prevê; termos técnicos só quando já estudados no ano.
8. Contexto verossímil: use o contexto pedido só como cenário de uma situação real e plausível, com números, preços e quantidades realistas para essa situação (dinheiro escrito como "R$ 45,00", nunca "R$ 4.500 centavos"; nada como "234 álbuns de 6 figurinhas"). Se o contexto não combinar com o conteúdo, use uma situação simples de sala de aula; nunca force (um ônibus não "forma um quadrilátero").
9. Dificuldade genuína, dentro da habilidade e do ano: Fácil = um passo, aplicação direta; Moderado = dois passos ou uma interpretação; Difícil = três ou mais passos, um dado intermediário a descobrir, distratores mais próximos da resposta ou leitura mais elaborada. Use sempre os números e o conteúdo que a habilidade prevê para o ano. Nunca crie dificuldade com ambiguidade, dado escondido ou conteúdo de ano posterior.
10. Resolução: mostre a conta ou o argumento completo usando SÓ os dados do enunciado e termine com o valor (ou a classe) da alternativa correta.
11. Leitura única: o enunciado e o comando admitem uma só interpretação (o que se pede, em que unidade, por qual critério). Um aluno que sabe o conteúdo não pode ficar em dúvida sobre o que a pergunta quer.
12. Arredondamento: quando o valor exato é uma dízima ou um número irracional, a alternativa correta traz o valor com DUAS casas decimais (arredondado ou truncado, por exemplo 5/11 = 0,4545... vira 0,45). Nesse caso escreva no enunciado "aproximadamente" ou "com duas casas decimais" e, na resolução, mostre o valor exato e depois o arredondamento. Nenhuma outra alternativa pode coincidir com o valor arredondado ou truncado em duas casas.
Antes de responder, confira as regras 2, 3, 4, 5, 6, 11 e 12. Se alguma falhar, reescreva a questão inteira.
"""

# Decisão do usuário (2026-10-01, D4): 1º e 3º ano também recebem Difícil,
# que o banco real nunca usa nesses anos. Vai SÓ para o professor (não entra
# no prompt de treino) e só quando a dificuldade pedida é Difícil nos anos
# iniciais: a dificuldade tem de vir de mais etapas/distratores próximos/
# leitura, nunca de conteúdo de ano posterior. O revisor confere em C2.
ANOS_INICIAIS = ("1º", "2º", "3º")
DIFICIL_ANOS_INICIAIS_ADDENDUM = (
    " Esta questão é Difícil para o {ano} ano: a dificuldade deve vir de mais etapas (duas ou três "
    "operações encadeadas), de um dado intermediário a descobrir, de distratores próximos da resposta "
    "ou de uma leitura mais elaborada, SEM sair do conteúdo do {ano} ano: use só os números e as "
    "operações que a habilidade descreve."
)

# T4 (revisão do piloto): no 9º H17, "Retângulo" foi a resposta de 4 das 7
# aceitas. O planejador da injeção passa ao professor as respostas que já se
# repetem na habilidade (só respostas não numéricas: classes, nomes).
RESPOSTAS_USADAS_ADDENDUM = (
    " Nesta habilidade já há questões demais cuja resposta correta é {lista}. Crie uma situação "
    "cuja resposta correta seja OUTRA."
)

# Instrução de posição do gabarito: vai SÓ para o professor (não entra no
# prompt de treino). Sorteamos a letra para equilibrar A-E (hoje E = 3,1%,
# porque nos itens reais o E é sempre a NDA sintética).
LETRA_ALVO_ADDENDUM = " Coloque a alternativa correta na letra {letra}."

# H2 (decisão do usuário, 2026-10-01): no ensino básico, valor com DUAS casas
# decimais (arredondado ou truncado) conta como correto quando o exato é dízima
# ou irracional e o enunciado não pede outra precisão. Caso real: idx 1348
# (DIST-H09-Difícil-00569, 5/11 com gabarito "0,45"), que a revisão do piloto 2
# tinha tirado da v3 como "gabarito falso" — aluno do básico não escreve
# 0,4545..., e o item fica. A regra vale nos três prompts (gerador: regra 12;
# validador e revisor: "Arredondamento") e no verificador aritmético
# (verificar_aritmetica); duas alternativas que casam pela regra = não única.
#
# Recalibração r6 (2026-10-01, prompts 788dcddaf0e6; outputs/calibracao_agentes.json):
# critério GERAL nos dois juízes — ler cada alternativa sozinha como afirmação;
# a mesma coisa em outra forma/unidade é V; nome de polígono não implica
# regular; contagem de extremos (início + N − 1). No validador, "alternativas"
# passou a ser plano ({"A": "V"}) com "motivos" à parte: em r5, 4 de 37
# respostas vieram com o objeto aninhado sem fechar (malformado = reprova, T5).
VALIDADOR_SISTEMA = """Você é professor de Matemática e revisor técnico de itens do SAEB/INEP. Sua única tarefa é verificar se um item de múltipla escolha está matematicamente correto e tem UMA única resposta. Você NÃO conhece o gabarito do autor: resolva o item por conta própria. Use as definições inclusivas da BNCC.

Siga esta ordem:
1. DADOS: liste o que o enunciado realmente informa. Não invente medidas e não complete o texto com o que o autor "provavelmente" quis dizer. O nome de um polígono informa só o número de lados: "pentágono" ou "octógono" não diz que a figura é regular (lados e ângulos iguais); se a conta depende disso e o enunciado não afirma, é dados_insuficientes.
2. PREMISSA: verifique se a situação pode existir. São impossíveis, por exemplo: triângulo com dois ângulos retos ou com dois obtusos; quadrilátero com quatro ângulos agudos ou com quatro obtusos; lados que violam a desigualdade triangular (o maior lado precisa ser menor que a soma dos outros dois); "quadrilátero" com só três lados informados; parte maior que o todo; número fracionário de pessoas.
3. PERGUNTA: diga o que é pedido e em que modo. "É" ou "classifica-se como" exige alternativa NECESSARIAMENTE verdadeira com os dados. "Pode ser" aceita o que é possível. "Não pode ser" exige impossibilidade garantida. Registre o critério, se houver ("quanto aos lados", "quanto aos ângulos").
4. RESOLUÇÃO: resolva passo a passo, mostrando cada conta. Quando der, confira por outro caminho (operação inversa, estimativa, substituição). Em contagens de dias ou de termos de uma sequência, decida se os extremos entram: algo que começa num dia e dura N dias termina no N-ésimo dia contando o primeiro (início + N − 1); "N dias depois" é início + N. Com N pequeno, conte um a um.
5. ALTERNATIVAS: julgue A, B, C, D e E separadamente:
   "V" = verdadeira, garantida pelos dados;
   "F" = falsa, contrariada pelos dados;
   "I" = indeterminada, depende de informação que o enunciado não dá;
   "G" = verdadeira, mas é uma classe MAIS GERAL, do mesmo critério, que outra alternativa "V" (só em pergunta pelo nome ou tipo da figura; ver abaixo).
   Julgue cada alternativa como afirmação sobre a situação do enunciado, por QUALQUER critério, e não só pelo critério da pergunta: uma alternativa de outro critério que seja verdadeira com os dados é "V" (numa pergunta "quanto aos lados", "Acutângulo" é "V" se os três ângulos forem agudos). Antes de marcar "F", leia a alternativa sozinha como afirmação sobre os dados; se ela for verdadeira, é "V", mesmo que não responda ao que foi perguntado. Uma alternativa que diz o mesmo que outra verdadeira de outra forma (o mesmo valor em outra unidade, uma decomposição, uma descrição em palavras) também é "V". Só o gabarito pode ser verdadeiro.
   Arredondamento (ensino básico): se o valor exato é uma dízima ou um número irracional e o enunciado não pede outra precisão, a alternativa com o valor arredondado OU truncado em DUAS casas decimais é "V" (5/11 = 0,4545..., então "0,45" é "V"; raiz de 2 = 1,414..., então "1,41" é "V"). Um valor com outra precisão não é igual a ele. Se duas alternativas casarem com o valor por essa regra, registre resposta_nao_unica.
   Uma figura pode estar em várias classes ao mesmo tempo: todo quadrado é retângulo, losango e paralelogramo; um triângulo pode ser retângulo e isósceles; o equilátero tem dois lados iguais. Ter UMA propriedade de uma classe não prova que a figura está nela: quatro ângulos retos garantem retângulo, não quadrado; quatro lados iguais garantem losango, não quadrado; lados todos diferentes garantem escaleno e não dizem nada sobre os ângulos. Para os ângulos a partir dos lados a, b, c (c o maior): c² < a² + b² dá acutângulo; c² = a² + b² dá retângulo; c² > a² + b² dá obtusângulo.
   Pergunta pelo NOME ou pelo tipo da figura ("como se chama", "qual é o tipo", "como se classifica"): se os dados GARANTEM uma classe, a resposta é essa classe mais específica, e uma classe mais geral do MESMO critério que a contém (isósceles para o equilátero; retângulo, losango ou paralelogramo para o quadrado; paralelogramo para o retângulo) é "G". Isso só vale quando a classe específica está garantida; se ela depender de dado ausente, ela é "I" e a geral é "V". Classes de critérios DIFERENTES não se anulam, com ou sem critério na pergunta: retângulo (ângulos) e isósceles (lados) verdadeiros ao mesmo tempo são as duas "V".
   Trapézio: se o item não define, lembre que há duas definições usuais (pelo menos um par de lados paralelos, ou exatamente um par); se a resposta depender da definição adotada, registre pergunta_ambigua.
   "Nenhuma das alternativas anteriores" é "V" só se as outras quatro forem "F"; se alguma for "I", ela também é "I"; se houver uma "V", ela é "F".
6. EQUIVALÊNCIA: aponte pares de alternativas que dizem a mesma coisa (mesmo valor em formas diferentes, como 1/2 e 0,5, 1,2 km e 120 000 cm, "quarenta e dois" e "4 dezenas e 2 unidades", ou um nome e sua definição, como "losango" e "quadrilátero de quatro lados iguais").
7. DEPENDÊNCIA VISUAL: aponte se o item cita figura, malha, gráfico ou tabela cujos dados não estão escritos.

Decisão: "resposta_calculada" é a letra da ÚNICA alternativa "V" quando houver exatamente uma "V" e as outras quatro forem "F" ou "G". Em qualquer outro caso, é null. Não tente salvar o item: se faltam dados ou há ambiguidade, registre o problema em vez de escolher a alternativa "mais provável".

Códigos de problema (use só estes): premissa_impossivel, dados_insuficientes, pergunta_ambigua, resposta_nao_unica, distrator_verdadeiro, nenhuma_correta, alternativas_equivalentes, nda_indevida, dependencia_visual, conta_impossivel.

Padrões de itens DEFEITUOSOS (aprenda o padrão, não as palavras):
[N1] "Uma janela tem os quatro cantos em ângulo reto. Qual é a forma da janela? A) Quadrado B) Retângulo C) Losango D) Triângulo E) Pentágono". B é V; A e C são I (não se sabe se os lados são iguais). resposta_calculada = null; resposta_nao_unica. O autor costuma marcar A pelo raciocínio falso "tem ângulos retos, então é quadrado".
[N2] "Um esquadro tem dois lados de 12 cm que formam um ângulo de 90°. Que tipo de triângulo ele é? A) Retângulo B) Isósceles C) Equilátero D) Escaleno E) Obtusângulo". A e B são V. resposta_nao_unica; faltou dizer "quanto aos ângulos" ou "quanto aos lados".
[N3] "Um terreno tem a forma de um quadrilátero com quatro ângulos agudos." A soma ficaria abaixo de 360°: premissa_impossivel. O mesmo vale para "triângulo com dois ângulos obtusos" e para lados de 4 cm, 4 cm e 9 cm (4 + 4 < 9).
[N4] "Uma placa triangular tem lados de 4 m, 6 m e 8 m. Quanto aos ângulos, ela é: A) Retângulo B) Acutângulo C) Obtusângulo ...". 8² = 64 > 4² + 6² = 52, então só C é V. Se o autor marcar B com "os lados são diferentes, logo é acutângulo", o gabarito está errado: lados dizem a classe quanto aos lados, não aos ângulos.
[N5] "Três estacas foram fincadas formando um triângulo. Como ele se classifica quanto aos lados?" (sem medidas). Todas as classes são I: dados_insuficientes. Marcar "Nenhuma das anteriores" também é erro, porque as outras são I, não F.
[N6] Alternativas "Losango" e "Quadrilátero com os quatro lados de mesma medida" no mesmo item: alternativas_equivalentes.
[N7] "Ana comeu metade de um bolo. Que parte sobrou? A) 1/2 B) 0,5 C) 1/4 ...": alternativas_equivalentes (A e B são o mesmo número).
[N8] "Qual destes quadriláteros tem dois ângulos obtusos? A) Trapézio B) Losango C) Paralelogramo ...". Losango e paralelogramo também podem ter: resposta_nao_unica.
[N9] "Um triângulo tem lados de 6 cm, 6 cm e 7 cm. Quanto aos lados, ele é: A) Equilátero B) Isósceles C) Escaleno D) Acutângulo E) Obtusângulo". B é V; 7² = 49 < 6² + 6² = 72, então os três ângulos são agudos e D também é V, mesmo sendo de outro critério. resposta_nao_unica e distrator_verdadeiro.

Padrões de itens CORRETOS (não reprove sem motivo concreto):
[P1] "Um triângulo tem lados de 5 cm, 5 cm e 6 cm. Quanto aos lados, ele é: A) Equilátero B) Isósceles C) Escaleno D) Retângulo E) Obtusângulo". B é V; A e C são F; 6² = 36 < 5² + 5² = 50, logo D e E são F. Alternativas de outro critério só são aceitáveis quando são F.
[P2] "Pedro tinha 35 figurinhas e ganhou 18. Quantas tem agora? A) 17 B) 53 C) 43 D) 413 E) 52". B é V; os outros são erros típicos (subtrair, esquecer o "vai um", juntar os algarismos, errar a conta por um).

Responda APENAS com um JSON numa linha, nesta ordem de campos:
{"dados": [str], "premissa_possivel": bool, "pergunta": str, "resolucao": str, "alternativas": {"A": "V|F|I|G", "B": "V|F|I|G", "C": "V|F|I|G", "D": "V|F|I|G", "E": "V|F|I|G"}, "motivos": {"A": str, "B": str, "C": str, "D": str, "E": str}, "equivalentes": [[letra, letra]], "dependencia_visual": bool, "resposta_calculada": "A|B|C|D|E" ou null, "problemas": [{"codigo": str, "detalhe": str}], "confianca": número de 0 a 1}
Em "alternativas" vai SÓ a letra do status; o porquê vai em "motivos". Seja breve: "resolucao" com até 6 frases e cada motivo com até 1 frase."""

VALIDADOR_USUARIO = """Item a verificar.

Enunciado: {enunciado}

A) {A}
B) {B}
C) {C}
D) {D}
E) {E}"""

# Fase 2: chamada NOVA e curta (não continua a conversa da fase 1): mais
# barata, e pode usar um modelo sem "thinking".
VALIDADOR_FASE2_SISTEMA = ("Você é professor de Matemática e revisor técnico de itens do SAEB/INEP. "
                           "Responda APENAS com o JSON pedido, numa linha.")

VALIDADOR_FASE2 = """Sua análise concluiu que a resposta é {letra}, e o autor marcou a mesma letra. Agora leia a resolução do autor e verifique só três coisas:
a) Todos os passos e contas da resolução estão corretos?
b) A resolução usa algum dado que NÃO está no enunciado, ou uma justificativa falsa (por exemplo, "tem quatro ângulos retos, logo é quadrado"), mesmo chegando à letra certa?
c) A resolução termina no valor ou na classe da alternativa {letra}?

Resolução do autor: {resolucao}

Responda APENAS com JSON numa linha:
{{"resolucao_correta": bool, "usa_dado_ausente": bool, "justificativa_falsa": bool, "chega_na_alternativa": bool, "problemas": [{{"codigo": "resolucao_errada|resolucao_usa_dado_ausente|justificativa_falsa|resolucao_nao_conclui", "detalhe": str}}], "confianca": número de 0 a 1}}"""

# D1 (decisão de 2026-10-01): o REVISOR passa a ser sabia-4-thinking e
# resolve o item ÀS CEGAS, como o validador: não recebe gabarito, resolução
# do autor nem o campo difficulty. Antes ele via o gabarito "depois de
# resolver" e funcionava como carimbo (auditoria do piloto: kappa 0,195 com o
# validador; aprovou 26/26 destilados, inclusive 2 com defeito; pegou 1 dos 3
# defeitos matemáticos que o validador pegou). A comparação com o gabarito é
# feita no CÓDIGO. A resolução do autor continua conferida pela fase 2 do
# validador (por isso o antigo C10 saiu). Focos distintos: o validador cuida
# do rigor matemático; o revisor, da própria resolução + pedagogia/SAEB/ano/
# habilidade/clareza/contexto. As alternativas chegam numa permutação
# DIFERENTE da do validador (sal "revisor").
REVISOR_SISTEMA = """Você é especialista em avaliação educacional e elaborador experiente de itens do SAEB (INEP), com prática em sala de aula do Ensino Fundamental. Você vai dar um parecer sobre um item de Matemática antes de ele entrar num banco de itens. Você NÃO recebe o gabarito nem a resolução do autor: resolva o item você mesmo, do zero, e só depois avalie os critérios. O item só é aprovado se puder ser aplicado aos alunos sem nenhuma correção. Use as definições inclusivas da BNCC.

PRIMEIRO, RESOLVA. Escreva sua resolução e julgue cada alternativa como afirmação sobre a situação do enunciado, por QUALQUER critério (não só pelo critério da pergunta):
  "V" = verdadeira, garantida pelos dados;
  "F" = falsa, contrariada pelos dados;
  "I" = indeterminada, depende de dado que o enunciado não dá;
  "G" = verdadeira, mas é uma classe MAIS GERAL, do mesmo critério, que outra alternativa "V" (isósceles para o equilátero; retângulo, losango ou paralelogramo para o quadrado; paralelogramo para o retângulo).
Uma figura pode estar em várias classes ao mesmo tempo. Ter UMA propriedade de uma classe não prova que a figura está nela: quatro ângulos retos garantem retângulo, não quadrado; quatro lados iguais garantem losango, não quadrado; lados todos diferentes garantem escaleno e não dizem nada sobre os ângulos. Ângulos a partir dos lados a, b, c (c o maior): c² < a² + b² dá acutângulo; c² = a² + b² dá retângulo; c² > a² + b² dá obtusângulo; com as três medidas dadas, essas classes são V ou F, nunca I. Uma alternativa de outro critério que seja verdadeira é "V" (numa pergunta "quanto aos lados", "Acutângulo" é "V" se os três ângulos forem agudos). Antes de marcar "F", leia a alternativa sozinha como afirmação sobre os dados; se ela for verdadeira, é "V", mesmo que não responda ao que foi perguntado. Uma alternativa que diz o mesmo que outra verdadeira de outra forma (o mesmo valor em outra unidade, uma decomposição, uma descrição em palavras) também é "V".
Não complete o enunciado: o nome de um polígono informa só o número de lados ("pentágono" não diz que ele é regular); se a resposta depende de um dado que o texto não afirma, a alternativa é "I". Em contagens de dias ou de termos de uma sequência, decida se os extremos entram: algo que começa num dia e dura N dias termina no N-ésimo dia contando o primeiro (início + N − 1); "N dias depois" é início + N. Com N pequeno, conte um a um.
Arredondamento (ensino básico): se o valor exato é uma dízima ou um número irracional e o enunciado não pede outra precisão, a alternativa com o valor arredondado OU truncado em DUAS casas decimais é "V" (5/11 = 0,4545..., então "0,45" é "V"). Um valor com outra precisão não é igual a ele. Se duas alternativas casarem com o valor por essa regra, a resposta não é única.
"resposta_calculada" é a letra da única "V" quando houver exatamente uma "V" e as outras quatro forem "F" ou "G"; senão é null. Não escolha a alternativa "mais provável".

DEPOIS, avalie cada critério com "ok" (true ou false) e uma nota curta:
C1 HABILIDADE: o item mede a habilidade informada e não outra. Falha se exige só leitura, só uma conta de outra habilidade, conteúdo de outro ano, ou se o conteúdo não é o subtema informado (quando informado).
C2 ANO: vocabulário, tamanho das frases e grandeza dos números compatíveis com o ano e com o que a habilidade prevê; termos técnicos só se já forem ensinados nesse ano. Nos anos iniciais (1º ao 3º), se o item exige números ou conteúdos de ano posterior, C2 é false (ano_inadequado), mesmo que seja trabalhoso.
C3 DIFICULDADE (só informa; não há dificuldade de referência para comparar): escreva em "dificuldade_real" a dificuldade que o item tem para um aluno do ano informado, por esta RUBRICA ABSOLUTA. Conte quatro coisas: (a) etapas: operações ou inferências encadeadas até a resposta; (b) operação: se é a operação mais básica do ano ou uma das mais exigentes; (c) números: se o tamanho deles é o típico do ano ou maior; (d) leitura: dado em tabela, texto longo, informação a selecionar ou descartar, conversão de unidade, dado intermediário a descobrir, comando a interpretar.
   Fácil = uma etapa, com operação e números típicos do ano e sem exigência de leitura (aplicação direta de um fato, de uma conta ou de uma definição).
   Moderado = duas etapas; OU uma etapa com UMA exigência a mais: números maiores que o típico, operação mais exigente do ano, conversão, leitura de tabela ou gráfico, informação a selecionar.
   Difícil = três ou mais etapas; OU duas etapas com dado intermediário a descobrir ou com outra exigência a mais; OU distratores tão próximos que exigem conferir a conta.
   Números e operações TÍPICOS por ano: 1º ano: até 100, juntar, acrescentar, separar e retirar, contagem. 2º ano: até 1.000, adição e subtração com reagrupamento, dobro, metade, multiplicação e divisão por 2, 3, 4, 5 e 10. 3º ano: até 10.000, as quatro operações com números pequenos, multiplicação por um algarismo. 4º ano: dezenas de milhar, multiplicação por dois algarismos, divisão por um algarismo, frações simples. 5º ano: centenas de milhar, as quatro operações com naturais e com decimais de dinheiro e de medidas, frações e porcentagens simples (10%, 25%, 50%). 9º ano: números reais, potências, raízes, equações, proporcionalidade, Pitágoras e semelhança.
   O contexto da história não conta como etapa. Uma conta com mais algarismos só é exigência a mais quando passa do típico do ano. C3 é sempre true: ele só registra a dificuldade_real.
C4 DISTRATORES: toda alternativa que não é a sua resposta tem de ser FALSA por qualquer critério. Falha (distrator_verdadeiro) se alguma for verdadeira por outro critério ou for uma classe mais geral que também vale (status "G"). Falha também com distrator absurdo (distrator_implausivel), duas alternativas equivalentes, formatos misturados (unidades ou tipos diferentes) ou se a correta se destaca pela forma. Na nota, diga qual erro de aluno cada distrator representa.
C5 CLAREZA: há uma só leitura possível do enunciado e do comando (o que se pede, em que unidade, por qual critério). Qualquer dúvida razoável de interpretação de um aluno que sabe o conteúdo reprova (enunciado_ambiguo).
C6 AUTOCONTIDO: todos os dados estão no texto; nada depende de figura, tabela ou gráfico ausente.
C7 CONTEXTO (veta): a situação é verossímil e serve ao problema; números, preços, unidades e quantidades são realistas para ela. Reprove (contexto_forcado) contexto inverossímil ou forçado, como "R$ 4.500 centavos", "234 álbuns de 6 figurinhas" ou objetos do dia a dia que "formam figuras" sem explicação. Problema de contexto nunca vai só em "sugestoes": se ele existe, C7 é false.
C8 PORTUGUÊS: ortografia, concordância e pontuação corretas. Deslizes que não mudam o sentido vão em "sugestoes" e não reprovam.
C9 MATEMÁTICA: a premissa é possível, os dados bastam e sua resolução tem exatamente uma "V", com as outras quatro "F".

Veredito true somente se C1, C2 e C4 a C9 estiverem ok (C3 só informa a dificuldade real).

Códigos de problema (use só estes): habilidade_desalinhada, subtema_desalinhado, ano_inadequado, dificuldade_incoerente, distrator_implausivel, distrator_verdadeiro, alternativas_equivalentes, enunciado_ambiguo, nao_autocontida, contexto_forcado, portugues, premissa_impossivel, dados_insuficientes, resposta_nao_unica.

Responda APENAS com um JSON numa linha, nesta ordem de campos:
{"resolucao_propria": str, "alternativas": {"A": "V|F|I|G", "B": ..., "C": ..., "D": ..., "E": ...}, "resposta_calculada": "A|B|C|D|E" ou null, "dificuldade_real": "Fácil|Moderado|Difícil", "criterios": {"C1": {"ok": bool, "nota": str}, ..., "C9": {"ok": bool, "nota": str}}, "problemas": [{"codigo": str, "detalhe": str}], "sugestoes": [str], "veredito": bool, "confianca": número de 0 a 1}
Seja breve: "resolucao_propria" com até 5 frases e cada nota com até 1 frase."""

# Sem gabarito, sem resolução do autor e sem "difficulty": o revisor só vê o
# que um aluno veria, mais o que precisa para julgar a pedagogia (ano,
# habilidade e subtema pedidos).
#
# P3 (2026-10-01): a "Dificuldade pedida" SAIU da mensagem. Ela ancorava o
# julgamento: no piloto 2 o revisor disse Moderado para as 7 Difícil dos anos
# iniciais e confirmou 0 de 17 Difícil na auditoria, enquanto rebaixava 47% dos
# Moderado/Difícil para Fácil — o sinal que a D3 usa estava viciado pelo rótulo
# pedido. Agora a dificuldade_real sai de uma RUBRICA ABSOLUTA por ano (C3 do
# REVISOR_SISTEMA: etapas, operação, tamanho dos números, leitura).
REVISOR_USUARIO = """Ano: {ano} ano
Habilidade: {habilidade} — {descricao}
Subtema pedido: {subtema}

Enunciado: {enunciado}

A) {A}
B) {B}
C) {C}
D) {D}
E) {E}"""

# P3: marca dos julgamentos de dificuldade feitos SEM a âncora da dificuldade
# pedida, pela rubrica absoluta. A D3 só usa dificuldade_real de registros com
# esta marca (auditar_base.deve_rebaixar_para_facil): o sinal antigo rebaixava
# 47% dos Moderado/Difícil auditados e confirmava 0 de 17 Difícil.
RUBRICA_DIFICULDADE = "p3-rubrica-absoluta-sem-ancora"

VERSAO_PROMPTS = hashlib.sha1("\x1e".join([
    GERADOR_ADDENDUM, LETRA_ALVO_ADDENDUM, DIFICIL_ANOS_INICIAIS_ADDENDUM, RESPOSTAS_USADAS_ADDENDUM,
    VALIDADOR_SISTEMA, VALIDADOR_USUARIO, VALIDADOR_FASE2_SISTEMA, VALIDADOR_FASE2, REVISOR_SISTEMA,
    REVISOR_USUARIO,
]).encode("utf-8")).hexdigest()[:12]

CRITERIOS_REVISOR = [f"C{i}" for i in range(1, 10)]

# Critério do revisor que é registrado mas NÃO veta (decisão da calibração,
# outputs/calibracao_agentes.json, rodada r3): em 29 itens reais do banco,
# presumidamente corretos, o revisor (sabia-4) marcou C3/dificuldade em 13
# (45%) — quase sempre "é EASY" para itens que o banco rotula Moderado ou
# Difícil — e 9 foram reprovados só por isso. Um juiz que discorda da
# referência em quase metade dos casos não serve de veto; a dificuldade pedida
# continua checada pelo filtro determinístico (campo difficulty) e o aviso
# fica no registro ("avisos"). Reaplicada às MESMAS respostas (cache, 0
# chamadas), a regra não fez nenhuma das 15 ruins do 9º H17 ser aceita.
# Desde P3 (2026-10-01) o revisor nem recebe a dificuldade pedida: C3 só
# registra a dificuldade_real pela rubrica absoluta.
CRITERIOS_INFORMATIVOS = ("C3",)
CODIGOS_INFORMATIVOS = {"dificuldade_incoerente"}
# D3 (2026-10-01): a dificuldade REAL dita pelo revisor não veta, mas é usada
# para um único ajuste — rebaixar para Fácil um item matematicamente correto
# (os dois juízes true) rotulado Moderado/Difícil. Nunca reclassifica em
# outra direção. Ver auditar_base.rebaixar_para_facil e a injeção.
DIFICULDADES_VALIDAS = ("Fácil", "Moderado", "Difícil")
_DIF_ALIAS = {"facil": "Fácil", "easy": "Fácil", "moderado": "Moderado", "medio": "Moderado",
              "médio": "Moderado", "medium": "Moderado", "dificil": "Difícil", "difícil": "Difícil",
              "hard": "Difícil", "fácil": "Fácil"}

# Códigos que indicam MATEMÁTICA errada (ensinar isso ao modelo é o dano visto
# no 9º H17). Usado pela montagem da v3 para decidir remoção de destilados.
CODIGOS_MATEMATICOS = {
    "premissa_impossivel", "dados_insuficientes", "resposta_nao_unica", "gabarito_errado",
    "gabarito_sem_resposta_unica", "resolucao_errada", "alternativas_equivalentes",
    "nda_indevida", "justificativa_falsa", "nenhuma_correta", "conta_impossivel",
    "pergunta_ambigua", "distrator_correto", "distrator_verdadeiro", "resolucao_usa_dado_ausente",
    "resolucao_nao_conclui", "dependencia_visual", "nao_autocontida", "geometria",
}
# D2 (2026-10-01): "erro matemático CONFIRMADO" = o validador cego reprova por
# um destes códigos E (o revisor também reprova por um deles OU um filtro
# determinístico matemático reprova). É mais estreito que CODIGOS_MATEMATICOS:
# item INCOMPLETO (figura perdida na transcrição, dado ausente, dependência
# visual) não está matematicamente errado — vai para revisão humana, não sai.
CODIGOS_ERRO_MATEMATICO = {
    "premissa_impossivel", "resposta_nao_unica", "gabarito_errado", "gabarito_sem_resposta_unica",
    "nenhuma_correta", "conta_impossivel", "alternativas_equivalentes", "nda_indevida",
    "distrator_correto", "distrator_verdadeiro", "resolucao_errada", "justificativa_falsa",
    "resolucao_usa_dado_ausente", "resolucao_nao_conclui", "geometria",
}
CODIGOS_INCOMPLETO = {"dados_insuficientes", "dependencia_visual", "nao_autocontida"}
# Códigos que falam da RESOLUÇÃO do autor: num item com resolução vazia eles
# só repetem "não há resolução" — e resolução vazia não é erro matemático (D2).
CODIGOS_RESOLUCAO = {"resolucao_errada", "justificativa_falsa", "resolucao_usa_dado_ausente",
                     "resolucao_nao_conclui"}
# Filtros determinísticos que apontam matemática errada (não "incompleta").
# verificador_aritmetico (revisão do piloto 2): conta EXATA em padrões
# estreitos (dias da semana, fração -> decimal, decomposição aditiva).
FILTROS_MATEMATICOS = {"consistencia", "geometria", "resolucao_supoe_dado", "verificador_aritmetico"}
# Códigos com que o VALIDADOR diz que a falta de resposta única vem da
# REDAÇÃO (duas leituras do enunciado), não de uma conta ou classe errada:
# vai para revisão humana, não é "erro matemático" da D2 (ex.: MT9013, o "ou"
# de "mesma quantidade de bombons ou de pirulitos" — gabarito 4 certo na
# leitura usual).
CODIGOS_AMBIGUIDADE = {"pergunta_ambigua"}

# Versão da DECISÃO calculada no código a partir das respostas dos juízes
# (não do texto dos prompts). Muda quando a mesma resposta passa a dar outro
# veredito; a auditoria refaz os registros de outra versão — com
# --cache-juizes, a custo zero (mesma mensagem = mesma resposta).
# j2-d5-validador-g (revisão do piloto 2): o status G do validador reprova
# (distrator_verdadeiro), como já fazia no revisor.
VERSAO_JUIZES = "j2-d5-validador-g"
# Códigos só pedagógicos: reprovam na injeção, mas na base existente não
# justificam remover um exemplo cuja matemática passou.
CODIGOS_PEDAGOGICOS = {
    "dificuldade_incoerente", "contexto_forcado", "portugues", "distrator_implausivel",
    "ano_inadequado", "habilidade_desalinhada", "subtema_desalinhado", "enunciado_ambiguo",
}

PAPEIS = ("gerador", "validador", "validador_fase2", "revisor")
MODELOS_PADRAO = {
    "gerador": "sabia-4-thinking",
    "validador": "sabia-4-thinking",
    # Fase 2 é uma checagem curta da resolução do autor; modelo sem thinking
    # basta e custa menos.
    "validador_fase2": "sabia-4",
    # D1 (2026-10-01): revisor thinking, mesma família (o usuário não usará
    # HF/outra família). A descorrelação vem do desenho — resolução cega,
    # outra permutação, outro foco — e não do modelo.
    "revisor": "sabia-4-thinking",
}
TEMPERATURAS = {"gerador": 0.85, "validador": 0.0, "validador_fase2": 0.0, "revisor": 0.0}


def max_tokens_para(papel, modelo):
    """Modelos 'thinking' gastam max_tokens no raciocínio antes do JSON; com o
    teto baixo o JSON sai truncado e vira 'nao_avaliado'. Por isso o teto sobe."""
    thinking = "thinking" in str(modelo).lower()
    base = {"gerador": 1024, "validador": 1500, "validador_fase2": 300, "revisor": 1200}[papel]
    return max(base, 4096 if papel != "validador_fase2" else 2048) if thinking else base


# ===========================================================================
# Cache de respostas dos JUÍZES (reaproveitar o que já foi pago)
# ===========================================================================

# Só papéis com temperatura 0. O gerador (0,85) nunca usa cache: numa nova
# tentativa do mesmo slot a mensagem pode ser idêntica e o cache devolveria a
# mesma questão já rejeitada.
PAPEIS_CACHEAVEIS = ("validador", "validador_fase2", "revisor")


def chave_cache(papel, modelo, mensagens):
    """Chave por MENSAGEM EXATA (papel, modelo, max_tokens, temperatura e o
    texto enviado). Qualquer mudança de prompt, de questão, de permutação ou
    de contexto do revisor (ano, habilidade, subtema, dificuldade) muda a
    chave. É a mesma chave do cache da calibração (calibrar_agentes)."""
    base = json.dumps([papel, modelo, max_tokens_para(papel, modelo), TEMPERATURAS[papel], mensagens],
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


class CacheRespostas:
    """Respostas já pagas dos juízes, lidas de um ou mais .jsonl (formato do
    cache da calibração: {"chave", "papel", "modelo", "texto", ...}).

    Usado pela auditoria para não pagar de novo o validador (e o revisor,
    quando a mensagem é idêntica) de itens já julgados na calibração com os
    MESMOS prompts. Respostas novas são anexadas ao primeiro arquivo
    (`gravar`). A resposta em cache é reinterpretada pelo código atual.

    Só resposta INTERPRETÁVEL entra ou sai do cache (revisão do piloto 2):
    um JSON malformado/truncado reprova naquela rodada (T5), mas não pode ser
    servido de novo para sempre a custo zero — o item ficaria preso como
    "nao_avaliado" e a D2/D3 nunca o alcançariam."""

    def __init__(self, paths, gravar=None):
        self.paths = [Path(p) for p in paths]
        self.gravar = Path(gravar) if gravar else (self.paths[0] if self.paths else None)
        self.d = {}
        self.hits = {}
        self.novas = 0
        self.malformadas_ignoradas = 0
        self.malformadas_nao_gravadas = 0
        self.truncadas_nao_gravadas = 0
        for p in self.paths:
            if not p.exists():
                continue
            for linha in p.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(linha)
                except json.JSONDecodeError:
                    continue  # linha final truncada por queda
                if r.get("chave") and r.get("texto") is not None:
                    self.d.setdefault(r["chave"], r)

    def obter(self, papel, modelo, mensagens):
        if papel not in PAPEIS_CACHEAVEIS:
            return None
        r = self.d.get(chave_cache(papel, modelo, mensagens))
        if r is not None and not resposta_interpretavel(papel, r.get("texto")):
            # Resposta malformada/truncada gravada antes desta regra (o cache da
            # calibração tem algumas): não vale como cache — senão o
            # "nao_avaliado, retenta na próxima rodada" da T5 nunca retenta.
            self.malformadas_ignoradas += 1
            return None
        if r is not None:
            self.hits[papel] = self.hits.get(papel, 0) + 1
        return r

    def guardar(self, papel, modelo, mensagens, texto, info):
        if papel not in PAPEIS_CACHEAVEIS or texto is None:
            return
        if resposta_truncada(info):
            # P4 (2026-10-01): cortada pelo max_tokens, mesmo que o JSON final
            # ainda feche. Não vira cache: a próxima rodada refaz a chamada.
            self.truncadas_nao_gravadas += 1
            return
        if not resposta_interpretavel(papel, texto):
            self.malformadas_nao_gravadas += 1
            return
        reg = {"chave": chave_cache(papel, modelo, mensagens), "papel": papel, "modelo": modelo, "texto": texto,
               "prompt_tokens": info.get("prompt_tokens"), "completion_tokens": info.get("completion_tokens"),
               "ts": agora_iso(), "versao_prompts": VERSAO_PROMPTS}
        self.d[reg["chave"]] = reg
        self.novas += 1
        if self.gravar:
            with abrir_para_gravar(self.gravar, "a") as f:
                f.write(json.dumps(reg, ensure_ascii=False) + "\n")

    def resumo(self):
        return {"arquivos": [str(p) for p in self.paths], "respostas_carregadas": len(self.d),
                "acertos_por_papel": dict(self.hits), "acertos": sum(self.hits.values()),
                "respostas_novas_gravadas": self.novas,
                "malformadas_ignoradas": self.malformadas_ignoradas,
                "malformadas_nao_gravadas": self.malformadas_nao_gravadas,
                "truncadas_nao_gravadas": self.truncadas_nao_gravadas}


# ===========================================================================
# Orçamento, sanitização e log de uso
# ===========================================================================

class OrcamentoEsgotado(RuntimeError):
    """O próximo passo passaria do teto de chamadas/tokens: nada foi enviado."""


class Orcamento:
    """Teto DURO de chamadas (e opcionalmente de tokens).

    reservar() é chamado ANTES de cada envio e levanta se o envio passaria do
    teto — então o número de requisições HTTP nunca excede max_chamadas.
    Tentativas que falham contam (o provedor pode ter cobrado)."""

    def __init__(self, max_chamadas, max_tokens=None):
        self.max_chamadas = None if max_chamadas is None else int(max_chamadas)
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        self.chamadas = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.estimadas = 0
        self.por_papel = {}

    def restante(self):
        return float("inf") if self.max_chamadas is None else self.max_chamadas - self.chamadas

    def cabe(self, n):
        return self.restante() >= n

    def reservar(self, papel):
        if self.max_chamadas is not None and self.chamadas + 1 > self.max_chamadas:
            raise OrcamentoEsgotado(f"teto de {self.max_chamadas} chamadas atingido")
        if self.max_tokens is not None and self.prompt_tokens + self.completion_tokens >= self.max_tokens:
            raise OrcamentoEsgotado(f"teto de {self.max_tokens} tokens atingido")
        self.chamadas += 1
        self.por_papel.setdefault(papel, {"chamadas": 0, "prompt_tokens": 0, "completion_tokens": 0})
        self.por_papel[papel]["chamadas"] += 1

    def registrar_tokens(self, papel, prompt_tokens, completion_tokens, estimado):
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        self.estimadas += int(bool(estimado))
        p = self.por_papel.setdefault(papel, {"chamadas": 0, "prompt_tokens": 0, "completion_tokens": 0})
        p["prompt_tokens"] += int(prompt_tokens or 0)
        p["completion_tokens"] += int(completion_tokens or 0)

    def resumo(self):
        return {"chamadas": self.chamadas, "max_chamadas": self.max_chamadas,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "chamadas_com_tokens_estimados": self.estimadas, "por_papel": self.por_papel}


# GOOGLE_API_KEY (árbitro Gemini, H4 de 2026-10-01) entra na mesma limpeza:
# uma exceção do requests que ecoe cabeçalho nunca vai para log nem registro.
_SEGREDOS_ENV = ("MARITALK_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                 "GEMINI_API_KEY")


def _sanitizar(msg, limite=300):
    """Remove qualquer valor de chave do texto (mensagens de exceção podem
    ecoar cabeçalhos em bibliotecas mal comportadas) e trunca."""
    s = str(msg)
    for var in _SEGREDOS_ENV:
        val = os.environ.get(var)
        if val and len(val) >= 6:
            s = s.replace(val, "***")
    s = re.sub(r"(?i)(bearer|authorization|api[_-]?key|token)([\"':=\s]+)[A-Za-z0-9._\-]{6,}",
               r"\1\2***", s)
    return s[:limite]


def _estimar_tokens(texto):
    # ~3,6 caracteres por token em PT (medido com o tokenizer do Qwen); só usado
    # quando a API não devolve "usage" — o registro marca "estimado": true.
    return int(len(texto or "") / 3.6) + 1


def _ler_usage(resp):
    u = getattr(resp, "usage", None)
    if u is None and isinstance(resp, dict):
        u = resp.get("usage")
    if u is None:
        return None
    get = (lambda k: u.get(k)) if isinstance(u, dict) else (lambda k: getattr(u, k, None))
    pt, ct = get("prompt_tokens"), get("completion_tokens")
    if pt is None and ct is None:
        return None
    return int(pt or 0), int(ct or 0)


def _status_http(exc):
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None)


def _retentavel(exc):
    """401/403/400/404 não melhoram com nova tentativa (e cada tentativa custa);
    429, 5xx, timeout e erro de rede sim."""
    st = _status_http(exc)
    if st is None:
        return True
    return st == 429 or st >= 500


# ===========================================================================
# Clientes
# ===========================================================================

class ClienteMaritacaMedido(dt.MaritacaClient):
    """MaritacaClient que NÃO descarta o campo "usage" da resposta (o original
    descarta, então hoje não há contagem de tokens). Reusa self.api_key e a
    MARITACA_URL do distill_teacher sem editá-lo."""

    def chat_completion(self, messages, max_tokens, temperature):
        import requests
        r = requests.post(
            dt.MARITACA_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature},
            timeout=self.timeout,
        )
        r.raise_for_status()  # a mensagem do HTTPError traz a URL, não o cabeçalho
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        # finish_reason "length" = cortada pelo max_tokens (P4: não vai para o cache)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                     finish_reason=data["choices"][0].get("finish_reason"))],
            usage=SimpleNamespace(prompt_tokens=usage.get("prompt_tokens"),
                                  completion_tokens=usage.get("completion_tokens")) if usage else None,
        )

    def __repr__(self):  # nunca expor a chave num traceback/log
        return f"ClienteMaritacaMedido(model={self.model!r})"


def criar_cliente_real(backend, modelo):
    """Cliente por papel. backend: 'maritaca' (MARITALK_API_KEY) ou 'hf'
    (HF Inference Providers, HF_TOKEN; o InferenceClient já devolve usage)."""
    if backend == "maritaca":
        return ClienteMaritacaMedido(modelo, timeout=240)
    if backend == "hf":
        from huggingface_hub import InferenceClient
        return InferenceClient(model=modelo, provider="auto")
    raise ValueError(f"backend desconhecido: {backend}")


class ClienteSimulado:
    """Cliente SEM REDE para --dry-run: identifica o papel pelo prompt de sistema
    e responde de forma determinística.

    - gerador: questão de adição com contexto variado (para não virar
      near-duplicata) e, a cada `erro_a_cada` questões, gabarito errado de
      propósito — assim o dry-run exercita o caminho de rejeição.
    - validador/revisor: resolvem de verdade SÓ contas "a + b" escritas no
      enunciado; qualquer outra coisa vira tudo "I" (=> reprova, fail-closed).
    Devolve usage estimado, exercitando a contabilidade de tokens."""

    NOMES = ["Ana", "Bruno", "Carla", "Davi", "Elisa", "Fábio", "Gabi", "Heitor", "Iara", "João",
             "Kátia", "Lucas", "Marina", "Nicolas", "Olívia", "Pedro"]
    OBJETOS = ["figurinhas", "bolinhas de gude", "livros", "lápis de cor", "adesivos", "tampinhas",
               "pulseiras", "selos", "cartões", "chaveiros", "botões", "conchas"]
    LUGARES = ["na feira do bairro", "na biblioteca da escola", "durante a gincana", "no clube de leitura",
               "na festa junina", "no mercado", "na horta comunitária", "na campanha de reciclagem",
               "no passeio ao parque", "na loja de brinquedos", "na padaria", "no campeonato"]
    VERBOS = [("tinha", "ganhou", "Quantos itens ficou tendo ao todo?"),
              ("guardou", "recebeu mais", "Com quantos itens ficou?"),
              ("separou", "juntou outros", "Qual é o total de itens separados?"),
              ("contou", "encontrou mais", "Quantos itens contou no total?")]

    def __init__(self, erro_a_cada=4):
        self.n = 0
        self.erro_a_cada = erro_a_cada

    @staticmethod
    def _resp(texto, messages):
        pt = sum(_estimar_tokens(m.get("content", "")) for m in messages)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=texto))],
                               usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=_estimar_tokens(texto)))

    @staticmethod
    def _alternativas(user):
        alts = {}
        for L in LETRAS:
            m = re.search(rf"^{L}\) (.*)$", user, re.M)
            alts[L] = m.group(1).strip() if m else ""
        return alts

    @staticmethod
    def _conta(user):
        m = re.search(r"(\d+) \+ (\d+)", user.split("\nA) ")[0])
        return int(m.group(1)) + int(m.group(2)) if m else None

    def _status(self, user, sal=""):
        alvo = self._conta(user)
        alts = self._alternativas(user)
        if alvo is None:
            # Item que o simulador não sabe resolver: "chuta" uma letra pelo hash
            # do enunciado (sal diferente por papel), só para o dry-run passar
            # pelos caminhos alta/media/baixa da auditoria. NÃO é avaliação.
            en = user.split("\nA) ")[0]
            chute = LETRAS[int(hashlib.sha1((sal + en).encode("utf-8")).hexdigest(), 16) % 5]
            return {L: ("V" if alts[L] == alts[chute] else "F") for L in LETRAS}
        return {L: ("V" if re.match(rf"^{alvo}\b", alts[L]) else "F") for L in LETRAS}

    def chat_completion(self, messages, max_tokens, temperature):
        sistema = messages[0]["content"]
        user = messages[-1]["content"]
        if "Regras de qualidade do SAEB/INEP" in sistema:
            return self._resp(self._gerar(user), messages)
        if "Você NÃO conhece o gabarito" in sistema:
            st = self._status(user)
            vs = [L for L in LETRAS if st[L] == "V"]
            unica = vs[0] if len(vs) == 1 and list(st.values()).count("F") == 4 else None
            obj = {"dados": ["simulado"], "premissa_possivel": True, "pergunta": "simulada",
                   "resolucao": "simulada", "alternativas": {L: {"status": st[L], "motivo": "sim"} for L in LETRAS},
                   "equivalentes": [], "dependencia_visual": False, "resposta_calculada": unica,
                   "problemas": [] if unica else [{"codigo": "dados_insuficientes", "detalhe": "simulado"}],
                   "confianca": 0.9 if unica else 0.3}
            return self._resp("<think>simulado</think>" + json.dumps(obj, ensure_ascii=False), messages)
        if "Sua análise concluiu" in user:
            obj = {"resolucao_correta": True, "usa_dado_ausente": False, "justificativa_falsa": False,
                   "chega_na_alternativa": True, "problemas": [], "confianca": 0.9}
            return self._resp(json.dumps(obj), messages)
        if "especialista em avaliação educacional" in sistema:
            # Revisor CEGO: não há gabarito no prompt; ele resolve e o código
            # compara. Desde P3 ele também não vê a dificuldade pedida: a
            # dificuldade_real simulada sai do hash do enunciado (~1/3 Fácil, o
            # resto Moderado) — exercita a D3 no dry-run sem ler o rótulo.
            st = self._status(user, sal="revisor")
            vs = [L for L in LETRAS if st[L] == "V"]
            unica = vs[0] if len(vs) == 1 and list(st.values()).count("F") == 4 else None
            ok = unica is not None
            en = user.split("\nA) ")[0]
            facil = int(hashlib.sha1(("dif" + en).encode("utf-8")).hexdigest(), 16) % 3 == 0
            dif_real = "Fácil" if facil else "Moderado"
            obj = {"resolucao_propria": "simulada", "alternativas": st, "resposta_calculada": unica,
                   "dificuldade_real": dif_real,
                   "criterios": {c: {"ok": ok if c == "C9" else True, "nota": "sim"}
                                 for c in CRITERIOS_REVISOR},
                   "problemas": [] if ok else [{"codigo": "resposta_nao_unica", "detalhe": "simulado"}],
                   "sugestoes": [], "veredito": ok, "confianca": 0.85 if ok else 0.4}
            return self._resp("<think>simulado</think>" + json.dumps(obj, ensure_ascii=False), messages)
        return self._resp("{}", messages)

    def _gerar(self, user):
        self.n += 1
        i = self.n
        rng = random.Random(f"sim|{i}")
        nome, obj = self.NOMES[i % len(self.NOMES)], self.OBJETOS[(i * 5) % len(self.OBJETOS)]
        lugar = self.LUGARES[(i * 7) % len(self.LUGARES)]
        v1, v2, perg = self.VERBOS[(i * 3) % len(self.VERBOS)]
        a, b = rng.randint(12, 60), rng.randint(5, 39)
        certo = a + b
        m = re.search(r"na letra ([A-E])", user)
        letra = m.group(1) if m else LETRAS[i % 5]
        distr = [a - b, certo + 10, certo - 1, certo + 1]
        valores = {}
        for L in LETRAS:
            valores[L] = certo if L == letra else distr.pop(0)
        dmatch = re.search(r"Dificuldade: (Fácil|Moderado|Difícil)", user)
        dif = DIFFICULTY_MAP.get(dmatch.group(1) if dmatch else "Fácil", "EASY")
        gab = letra
        if self.erro_a_cada and i % self.erro_a_cada == 0:
            gab = LETRAS[(LETRAS.index(letra) + 1) % 5]  # erro proposital
        q = {"enunciado": f"{nome} {v1} {a} {obj} {lugar} e {v2} {b}. Para achar o total, fez {a} + {b}. {perg}",
             "alternativas": {L: f"{valores[L]} {obj}" for L in LETRAS},
             "resolucao_passo_a_passo": f"{a} + {b} = {certo}.",
             "resposta_correta": gab, "difficulty": dif}
        return json.dumps({"questoes": [q]}, ensure_ascii=False)


# ===========================================================================
# Parse robusto
# ===========================================================================

def extrair_json(texto, chaves=(), estrito=False):
    """Último objeto JSON do texto que contém todas as `chaves`.

    Mais tolerante que schema_utils.parse_json (que pega só o PRIMEIRO "{"):
    modelos thinking às vezes escrevem chaves soltas no raciocínio fora das
    tags, ou envolvem a resposta em ```json. Remove <think>…</think> (inclusive
    um <think> nunca fechado: aí não há resposta e devolvemos None).

    estrito=True (JUÍZES — T5, revisão do piloto): malformado = reprovação,
    sempre. Só vale o objeto que TERMINA no último "}" do texto, sem nenhum
    "{" depois dele, e sem reparo de sintaxe. Fecha dois caminhos em que uma
    resposta truncada virava aprovação: (a) _reparar_json fechando as chaves
    logo depois de "resposta_calculada"; (b) um rascunho JSON completo escrito
    no raciocínio sendo usado no lugar da resposta final truncada."""
    if not isinstance(texto, str) or not texto.strip():
        return None
    t = re.sub(r"<think>.*?</think>", "", texto, flags=re.S)
    if "<think>" in t:  # raciocínio truncado: o JSON nunca chegou
        t = t.split("<think>")[0]
    t = re.sub(r"```(?:json)?", "", t)
    if estrito:
        return _json_final(t, chaves)
    obj = parse_json(t)
    if isinstance(obj, dict) and all(k in obj for k in chaves):
        candidato = obj
    else:
        candidato = None
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", t):
        try:
            o, _ = dec.raw_decode(t[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and all(k in o for k in chaves):
            candidato = o
    if candidato is None and "{" in t:
        candidato = _reparar_json(t[t.index("{"):], chaves)
    return candidato


def _json_final(t, chaves):
    """Objeto JSON que termina exatamente no último "}" de `t` (o mais externo
    que termina ali), com todas as `chaves`; None se não houver, se houver
    algum "{" depois dele (outra resposta começada e truncada) ou se o JSON
    final não decodifica sem reparo."""
    t = t.rstrip()
    fim = t.rfind("}")
    if fim < 0 or "{" in t[fim + 1:]:
        return None
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", t[:fim + 1]):
        try:
            o, n = dec.raw_decode(t, m.start())
        except json.JSONDecodeError:
            continue
        if n == fim + 1:
            return o if isinstance(o, dict) and all(k in o for k in chaves) else None
    return None


def _reparar_json(t, chaves):
    """Conserta só SINTAXE (aspas escapadas fora de string, chave de
    fechamento faltando no fim). Usado SÓ fora do modo estrito (saída do
    gerador, que ainda passa pelos filtros e pelos dois juízes). Os juízes
    não usam reparo desde a decisão T5: um reparo que fecha chaves de uma
    resposta truncada pode apagar os problemas que o juiz ia listar."""
    t = re.sub(r'(?<=[:\[,])(\s*)\\"', r'\1"', t)
    t = re.sub(r'\\"(?=\s*[,}\]])', '"', t)
    # revisor: '}, {"C10": {...}}' (chave aberta a mais antes de um critério)
    t = re.sub(r'\}\s*,\s*\{\s*"(C\d{1,2})"', r'}, "\1"', t)
    dec = json.JSONDecoder()
    for extra in ("", "}", "}}", "}}}"):
        try:
            o, _ = dec.raw_decode(t.rstrip() + extra)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and all(k in o for k in chaves):
            return o
    return None


def _status_valido(v):
    """'V'/'F'/'I'/'G' a partir de 'V', 'verdadeira', {'status': 'V'}...; None se inválido.

    'G' (verdadeira, mas classe mais geral que outra 'V' do mesmo critério) só
    vale escrito exatamente como 'G' ou 'geral': pela primeira letra, um
    status como 'garantida' viraria 'G' e mudaria o veredito."""
    if isinstance(v, dict):
        v = v.get("status")
    if not isinstance(v, str) or not v.strip():
        return None
    t = v.strip().upper()
    if t in ("G", "GERAL"):
        return "G"
    c = t[0]
    return c if c in "VFI" else None


def _bool_estrito(v):
    return v if isinstance(v, bool) else None


def _confianca(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, x))


def _letra(v):
    if isinstance(v, str) and v.strip().upper() in tuple(LETRAS):
        return v.strip().upper()
    return None


def resposta_unica(status):
    """Letra da única 'V' quando há exatamente uma V e as outras quatro são F
    ou G; senão None.

    Desde a D5 (2026-10-01) um G REPROVA nos dois juízes (distrator_verdadeiro,
    decidido no código em validar() e revisar()); aqui ele só não impede
    identificar QUAL é a resposta do juiz. Histórico do G: em pergunta pelo NOME da figura, a classe
    mais geral do mesmo critério (isósceles para um equilátero garantido) é
    verdadeira, mas não é o nome pedido — a auditoria humana aprova esses itens.
    Na calibração (rodada r1) o validador entendeu a regra mas se recusou a
    chamar de "F" algo verdadeiro e marcou "I", reprovando um item correto; o
    status próprio evita forçar o modelo a mentir no rótulo. G nunca cria uma
    V: se a classe específica não está garantida, não há V e o item reprova."""
    vs = [L for L in LETRAS if status.get(L) == "V"]
    resto = [L for L in LETRAS if status.get(L) in ("F", "G")]
    return vs[0] if len(vs) == 1 and len(resto) == 4 else None


def _problemas(lista):
    out = []
    for p in lista if isinstance(lista, list) else []:
        if isinstance(p, dict) and str(p.get("codigo") or "").strip():
            out.append({"codigo": str(p["codigo"]).strip(), "detalhe": str(p.get("detalhe", ""))[:300]})
        elif isinstance(p, str) and p.strip():
            out.append({"codigo": p.strip()[:60], "detalhe": ""})
    return out


def _completo(o):
    """Os dois ÚLTIMOS campos do JSON pedido estão presentes e bem formados.

    Fail-closed contra truncamento: _reparar_json fecha as chaves de uma
    resposta cortada (max_tokens do modelo thinking) logo depois de
    "resposta_calculada"; sem esta exigência os problemas que o modelo ia
    listar (ex.: pergunta_ambigua) sumiam e o item seguia aprovado
    (reproduzido na revisão do piloto, /tmp/claude-1000/base/rev_indep/cego.py).
    Todas as 180 respostas reais em cache (calibração) trazem os dois campos."""
    return isinstance(o.get("problemas"), list) and _confianca(o.get("confianca")) is not None


def _alternativas_exatas(o):
    """'alternativas' é um dict com EXATAMENTE as letras A-E. Antes, campos
    caídos dentro dele (chave de fechamento esquecida depois de "E") eram
    devolvidos ao nível de cima; desde T5 isso é malformado (reprova)."""
    alts = o.get("alternativas")
    return isinstance(alts, dict) and set(alts) == set(LETRAS)


def dificuldade_valida(v):
    """'Fácil'/'Moderado'/'Difícil' a partir também de EASY/MEDIUM/HARD; None se inválido."""
    if not isinstance(v, str) or not v.strip():
        return None
    t = v.strip()
    return t if t in DIFICULDADES_VALIDAS else _DIF_ALIAS.get(t.lower())


def interpretar_fase1(texto):
    """Normaliza a saída da fase 1 do validador. None = malformado (=> reprova)."""
    o = extrair_json(texto, ("alternativas",), estrito=True)
    if o is None or not _alternativas_exatas(o):
        return None
    if not _completo(o):
        return None
    status = {L: _status_valido(o["alternativas"].get(L)) for L in LETRAS}
    if any(s is None for s in status.values()):
        return None
    premissa = _bool_estrito(o.get("premissa_possivel"))
    dep = _bool_estrito(o.get("dependencia_visual"))
    if premissa is None or dep is None:
        return None
    eq = []
    for par in o.get("equivalentes") or []:
        if isinstance(par, (list, tuple)) and len(par) >= 2:
            ls = [_letra(x) for x in par]
            if all(ls):
                eq.append(ls[:2])
    if o.get("equivalentes") not in (None, []) and not isinstance(o.get("equivalentes"), list):
        return None
    return {"status": status, "premissa_possivel": premissa, "dependencia_visual": dep,
            "equivalentes": eq, "resposta_llm": _letra(o.get("resposta_calculada")),
            "problemas": _problemas(o.get("problemas")), "confianca": _confianca(o.get("confianca")),
            "resolucao": str(o.get("resolucao", ""))[:800]}


def interpretar_fase2(texto):
    o = extrair_json(texto, ("resolucao_correta",), estrito=True)
    if o is None or not _completo(o):
        return None
    campos = {k: _bool_estrito(o.get(k)) for k in
              ("resolucao_correta", "usa_dado_ausente", "justificativa_falsa", "chega_na_alternativa")}
    if any(v is None for v in campos.values()):
        return None
    return dict(campos, problemas=_problemas(o.get("problemas")), confianca=_confianca(o.get("confianca")))


def interpretar_revisor(texto):
    o = extrair_json(texto, ("criterios", "alternativas"), estrito=True)
    if o is None or not isinstance(o.get("criterios"), dict) or not _alternativas_exatas(o):
        return None
    if not _completo(o):
        return None
    dif_real = dificuldade_valida(o.get("dificuldade_real"))
    if dif_real is None:
        return None
    status = {L: _status_valido(o["alternativas"].get(L)) for L in LETRAS}
    if any(s is None for s in status.values()):
        return None
    crit = {}
    for c in CRITERIOS_REVISOR:
        v = o["criterios"].get(c)
        ok = _bool_estrito(v.get("ok")) if isinstance(v, dict) else _bool_estrito(v)
        if ok is None:
            return None
        crit[c] = {"ok": ok, "nota": str(v.get("nota", ""))[:200] if isinstance(v, dict) else ""}
    ver = _bool_estrito(o.get("veredito"))
    if ver is None:
        return None
    return {"status": status, "criterios": crit, "veredito_llm": ver, "dificuldade_real": dif_real,
            "resposta_llm": _letra(o.get("resposta_calculada")),
            "problemas": _problemas(o.get("problemas")),
            "sugestoes": [str(s)[:200] for s in (o.get("sugestoes") or []) if isinstance(s, str)][:5],
            "confianca": _confianca(o.get("confianca"))}


# finish_reason de resposta cortada pelo teto de tokens: "length" (OpenAI/
# Maritaca) e "MAX_TOKENS" (Gemini).
FINS_TRUNCADOS = {"length", "max_tokens"}


def resposta_truncada(info):
    """A API disse que a resposta foi cortada pelo max_tokens (P4)."""
    fim = (info or {}).get("finish_reason")
    return fim is not None and str(fim).lower() in FINS_TRUNCADOS


def resposta_interpretavel(papel, texto):
    """A resposta de um juiz é interpretável pelo código atual? (cache: só
    estas entram e saem; ver CacheRespostas)."""
    f = {"validador": interpretar_fase1, "validador_fase2": interpretar_fase2,
         "revisor": interpretar_revisor}.get(papel)
    return f is not None and isinstance(texto, str) and f(texto) is not None


# ===========================================================================
# Permutação anti-viés de posição
# ===========================================================================

_REF_LETRAS = re.compile(r"\b[A-E]\s*(?:,|e|ou)\s*[A-E]\b")
_REF_POSICAO = re.compile(r"\b(anteriores|acima)\b", re.I)
# Enunciado que repete as alternativas com rótulo ("... A) Escaleno B) Isósceles").
# Permutar só as alternativas deixaria o enunciado e a lista exibida com letras
# trocadas, e o validador julgaria a letra errada (visto na calibração: item
# correto reprovado com "gabarito_errado").
_ROTULO_NO_ENUNCIADO = re.compile(r"(?<![A-Za-zÀ-ú])[A-E]\)\s")


def hash_questao(questao):
    base = json.dumps({"e": questao.get("enunciado"), "a": questao.get("alternativas")},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def permutacao(questao, sal=""):
    """{letra_exibida: letra_original}. Semente = hash da questão + `sal`
    (determinístico). O validador usa sal "" (a permutação de sempre); o
    revisor usa "revisor", então os dois juízes veem ordens diferentes.

    LLMs têm preferência por certas letras; permutar também descorrelaciona o
    validador do gerador. NDA e alternativas que citam posição ("anteriores")
    ficam no lugar; se alguma alternativa cita outras letras ("A e B"), não
    permuta nada (permutar mudaria o sentido)."""
    alts = questao.get("alternativas") or {}
    textos = {L: str(alts.get(L, "")) for L in LETRAS}
    if any(_REF_LETRAS.search(t) for t in textos.values()):
        return {L: L for L in LETRAS}
    if len(_ROTULO_NO_ENUNCIADO.findall(str(questao.get("enunciado") or ""))) >= 2:
        return {L: L for L in LETRAS}
    fixas = {L for L, t in textos.items() if classe_alternativa(t) == "nda" or _REF_POSICAO.search(t)}
    moveis = [L for L in LETRAS if L not in fixas]
    destino = list(moveis)
    random.Random(hash_questao(questao) + sal).shuffle(destino)
    perm = {L: L for L in fixas}
    perm.update(dict(zip(moveis, destino)))
    return perm


# ===========================================================================
# Filtros determinísticos (0 chamadas) — versões PURAS
# ===========================================================================

def veredito_geometria(questao, ano=None, habilidade=None):
    """Adaptador para src/verificador_geometria.py (API ainda em definição pelo
    outro workflow). Devolve {"status": indisponivel|sem_veredito|aprovada|
    reprovada, "detalhe"}. Só "reprovada" — um False explícito do verificador —
    reprova; qualquer exceção ou retorno irreconhecível é "sem_veredito"."""
    if _vg is None:
        return {"status": "indisponivel", "detalhe": "src/verificador_geometria.py ausente ou não importa"}
    # verificar_geometria é a API publicada pelo outro workflow: devolve
    # (veredito_str, detalhe) com veredito em _vg.VEREDITOS. Os outros nomes
    # ficam por compatibilidade com versões anteriores do módulo.
    fn = next((getattr(_vg, n) for n in ("verificar_geometria", "verificar_questao", "verificar",
                                         "avaliar_questao")
               if callable(getattr(_vg, n, None))), None)
    if fn is None:
        return {"status": "indisponivel", "detalhe": "nenhuma função reconhecida no verificador"}
    try:
        import inspect
        params = inspect.signature(fn).parameters
        kw = {k: v for k, v in (("ano", ano), ("habilidade", habilidade)) if k in params}
        r = fn(questao, **kw)
    except Exception as exc:  # noqa: BLE001 - módulo alheio em construção
        return {"status": "sem_veredito", "detalhe": _sanitizar(f"erro: {exc}", 160)}
    bruto = r
    detalhe_vg = None
    if isinstance(r, tuple) and r:
        if len(r) > 1 and isinstance(r[1], dict):
            detalhe_vg = r[1]
        r = r[0]
    # Vereditos textuais do verificador: os de GEO_REJEITA reprovam, "ok"
    # aprova e "nao_aplicavel" (fora do escopo / parse sem confiança) não
    # decide nada — na dúvida o verificador se abstém e quem julga são os LLMs.
    rejeita = set(getattr(_vg, "GEO_REJEITA", ()) or
                  ("gabarito_errado", "nao_unica", "premissa_impossivel", "dados_insuficientes"))
    if isinstance(r, str) and (r in rejeita or r in ("ok", "nao_aplicavel")):
        resumo = {"veredito": r}
        if detalhe_vg:
            resumo.update({k: detalhe_vg.get(k) for k in ("motivo", "explicacao") if k in detalhe_vg})
        if r == "nao_aplicavel":
            return {"status": "sem_veredito", "detalhe": _sanitizar(json.dumps(resumo, ensure_ascii=False), 300)}
        return {"status": "reprovada" if r in rejeita else "aprovada", "veredito_vg": r,
                "detalhe": _sanitizar(json.dumps(resumo, ensure_ascii=False), 300)}
    if isinstance(r, dict):
        for chave in ("ok", "valida", "aprovada", "veredito"):
            if chave in r:
                r = r[chave]
                break
    if isinstance(r, str):
        r = {"ok": True, "aprovada": True, "valida": True, "verdadeiro": True, "aprova": True,
             "reprovada": False, "invalida": False, "falso": False, "reprova": False}.get(r.lower())
    if not isinstance(r, bool):
        return {"status": "sem_veredito", "detalhe": _sanitizar(repr(bruto), 160)}
    return {"status": "aprovada" if r else "reprovada", "detalhe": _sanitizar(repr(bruto), 300)}


def veredito_d5_geometria(questao):
    """P1 (2026-10-01): D5 determinística sobre a geometria — {"status":
    "reprovada"|"sem_veredito"|"indisponivel", "detalhe"}. Reprova quando o
    verificador dá "ok" mas outra alternativa também é verdadeira (classe mais
    geral do gabarito ou classe verdadeira por outro eixo). Ver
    verificador_geometria.verificar_d5. Sem o verificador, nunca reprova."""
    fn = getattr(_vg, "verificar_d5", None) if _vg is not None else None
    if not callable(fn):
        return {"status": "indisponivel"}
    try:
        viola, det = fn(questao)
    except Exception as exc:  # noqa: BLE001
        return {"status": "sem_veredito", "detalhe": _sanitizar(f"erro: {exc}", 160)}
    if viola:
        return {"status": "reprovada", "detalhe": det}
    return {"status": "sem_veredito"}


# Versão dos filtros determinísticos. Muda quando um filtro novo pode
# reprovar o que antes passava: a auditoria reaplica os filtros aos registros
# em cache (0 chamadas) em vez de manter um rótulo "alta" que o filtro atual
# derrubaria.
# f3-aritmetica (revisão do piloto 2): verificar_aritmetica (dias da semana,
# fração -> decimal, decomposição aditiva).
# f4-h2-d5 (2026-10-01): H2 (duas casas decimais para dízima: o 1348 deixa de
# ser "contradiz") e o pré-filtro D5 de geometria (informativo na auditoria).
# f5-h2-d5-extenso (passo 2): verificador exato do número "por extenso" (idx
# 1052, que os dois juízes novos aprovaram juntos na regressão r7).
# f6-h2-d5-extenso-planilha (revisão do passo 2): D6, alternativa que é data da
# planilha (17 reais da base); H2 com "uma casa decimal" no singular, valor
# exato pedido e aproximação não pedida com o exato entre as alternativas.
VERSAO_FILTROS = "f6-h2-d5-extenso-planilha"

MIN_RESOLUCAO = 5
# Hipótese declarada pelo autor da resolução ("supondo que..."): se o
# enunciado não traz a mesma hipótese, a conta depende de dado ausente.
_HIPOTESE = re.compile(r"\b(?:supond[oa]|suponha(?:mos)?|supor que|assumind[oa]|presumind[oa]"
                       r"|vamos considerar que|se considerarmos)\b", re.I)
# "Regular" (todos os lados E todos os ângulos iguais) nunca decorre só do
# nome do polígono: "hexágono" não diz que os ângulos são iguais, e dividir a
# soma dos ângulos por n exige essa hipótese.
_REGULAR = re.compile(r"\bregular(?:es)?\b", re.I)


def defeitos_resolucao(questao):
    """Defeitos da RESOLUÇÃO detectáveis sem LLM: ["resolucao_vazia"] ou
    ["resolucao_supoe_dado"] ou [].

    Por quê: na auditoria do piloto os dois juízes (mesma família de modelo)
    deram "alta" a DIST-H15-Fácil-00589 ("Como o hexágono é regular..." sem
    "regular" no enunciado — o mesmo erro de dado suposto do 9º H17) e a um
    item real com resolução vazia (a fase 2 respondeu resolucao_correta=true
    para um texto vazio). Um filtro determinístico não compartilha o ponto
    cego dos LLMs. Na base inteira (1.481 questões) a regra de hipótese
    dispara só no item confirmado; resolução vazia, nos 38 reais MT9xxx sem
    resolução (vão para revisão humana, nunca saem sozinhos)."""
    res = str(questao.get("resolucao_passo_a_passo") or "").strip()
    if len(res) < MIN_RESOLUCAO:
        return ["resolucao_vazia"]
    texto = " ".join([str(questao.get("enunciado", ""))]
                     + [str(v) for v in (questao.get("alternativas") or {}).values()])
    for rx in (_REGULAR, _HIPOTESE):
        if rx.search(res) and not rx.search(texto):
            return ["resolucao_supoe_dado"]
    return []


# ---------------------------------------------------------------------------
# T1: duplicata SEMÂNTICA — mesmos dados numéricos + mesma resposta.
# O Jaccard de trigramas não vê o mesmo 30-60-90 em outro contexto nem a
# paráfrase de um item real; os NÚMEROS do enunciado e a resposta, sim.
# ---------------------------------------------------------------------------
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_MILHAR = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?")
MIN_NUMEROS_ASSINATURA = 2  # com 1 número só, a coincidência é comum demais


def _num_canonico(s):
    """'4.500' -> '4500'; '2,5' -> '2.5'; '12' -> '12'."""
    if _MILHAR.fullmatch(s):
        s = s.replace(".", "")
    s = s.replace(",", ".")
    try:
        x = float(s)
    except ValueError:
        return s
    return str(int(x)) if x == int(x) else repr(round(x, 6))


def numeros(texto):
    """Multiconjunto ordenado dos números do texto, em forma canônica."""
    return tuple(sorted(_num_canonico(m) for m in _NUM.findall(str(texto or ""))))


def assinatura_dados(questao):
    """(números do enunciado, resposta) ou None se não dá para comparar.
    A resposta é a tupla de números da alternativa correta (unidade/contexto
    ignorados: "53 figurinhas" = "53 livros") ou, sem números, o texto
    normalizado ("Retângulo")."""
    nums = numeros(questao.get("enunciado"))
    if len(nums) < MIN_NUMEROS_ASSINATURA:
        return None
    alt = (questao.get("alternativas") or {}).get(questao.get("resposta_correta"))
    if not alt:
        return None
    na = numeros(alt)
    resp = ("n",) + na if na else ("t", dt.normalizar(str(alt)))
    return nums, resp


def e_duplicata_semantica(q1, q2):
    a = assinatura_dados(q1)
    return a is not None and a == assinatura_dados(q2)


def resposta_conteudo(questao):
    """Texto normalizado da alternativa correta quando ela NÃO é numérica
    (classe, nome de figura...); None para respostas numéricas (T4)."""
    alt = (questao.get("alternativas") or {}).get(questao.get("resposta_correta"))
    if not alt or re.search(r"\d", str(alt)):
        return None
    return dt.normalizar(str(alt)) or None


# T2: contexto inverossímil detectável sem LLM (visto no piloto: "R$ 4.500
# centavos"). O resto (ex.: "234 álbuns de 6 figurinhas") é veto do revisor (C7).
_REAIS_EM_CENTAVOS = re.compile(r"R\$\s*\d[\d.,]*\s*centavos", re.I)


def contexto_inverossimil(questao):
    texto = " ".join([str(questao.get("enunciado", ""))]
                     + [str(v) for v in (questao.get("alternativas") or {}).values()])
    return bool(_REAIS_EM_CENTAVOS.search(texto))


# ---------------------------------------------------------------------------
# Verificadores aritméticos EXATOS (revisão do piloto 2).
#
# Os dois juízes são da mesma família (sabia-4-thinking) e erram JUNTOS: no
# idx 881 ("começou numa terça e durou 10 dias") os dois contaram "sexta" e um
# item CORRETO ia sair da v3 como "erro matemático confirmado"; no idx 1348
# (5/11 com gabarito "0,45") os dois aprovaram um gabarito falso. A
# concordância entre eles não é um sinal independente. Um verificador que
# CALCULA a resposta é: quando ele reconhece o padrão com segurança, o
# veredito dele vale nos dois sentidos — "contradiz" reprova (filtro
# matemático) e "confirma" impede a D2 de remover o item por um erro dos
# juízes. Padrões estreitos de propósito (0 disparos falsos na base inteira,
# val e injeção: ver Doc/RELATORIO_BASE.md): fora deles, "sem_veredito".
# ---------------------------------------------------------------------------
_DIAS = {"domingo": 0, "segunda": 1, "terca": 2, "terça": 2, "quarta": 3, "quinta": 4, "sexta": 5,
         "sabado": 6, "sábado": 6}
_DIA_RX = re.compile(r"\b(domingo|segunda|ter[çc]a|quarta|quinta|sexta|s[áa]bado)(?:-feira)?\b", re.I)
_DIA_ALT = re.compile(r"^\s*(domingo|segunda|ter[çc]a|quarta|quinta|sexta|s[áa]bado)(?:-feira)?\s*\.?\s*$", re.I)
# "durou/dura/durará (exatamente) N dias": o último dia é o N-ésimo contando o primeiro.
_DURACAO = re.compile(r"\b(?:dur(?:ou|a|ará|aria|ar)|vai durar|teve duração de|com duração de)\s+"
                      r"(?:exatamente\s+|apenas\s+|só\s+)?(\d+)\s+dias\b", re.I)
_PERGUNTA_FIM = re.compile(r"\b(?:termin|acab|encerr)\w*|\búltimo dia\b", re.I)
# "N dias depois/após", "daqui a N dias": início + N.
_DESLOCAMENTO = re.compile(r"\b(\d+)\s+dias\s+(?:depois|após|mais tarde)\b|\b(?:daqui a|dali a)\s+(\d+)\s+dias\b",
                           re.I)
_N_DIAS = re.compile(r"\b\d+\s+dias\b", re.I)
_NDA = re.compile(r"nenhuma d", re.I)


def _veredito_verificador(nome, gab, verdadeiras, esperado, alts):
    """confirma: só o gabarito é verdadeiro; contradiz: o gabarito é falso ou
    há outra alternativa verdadeira; NDA vale se nenhuma outra vale."""
    ndas = [L for L in LETRAS if _NDA.search(str(alts.get(L, "")))]
    if not verdadeiras and ndas:
        verdadeiras = ndas[:1]
    status = "confirma" if verdadeiras == [gab] else "contradiz"
    return {"status": status, "verificador": nome, "esperado": esperado, "verdadeiras": verdadeiras,
            "gabarito": gab,
            "detalhe": (f"{nome}: esperado {esperado}; verdadeira(s) {verdadeiras or 'nenhuma'}; gabarito {gab}")}


def _verificar_dias_semana(q, alts, gab):
    en = str(q.get("enunciado") or "")
    dias_alt = {}
    for L in LETRAS:
        m = _DIA_ALT.match(str(alts.get(L, "")))
        if m:
            dias_alt[L] = _DIAS[m.group(1).lower()]
        elif not _NDA.search(str(alts.get(L, ""))):
            return None  # alternativa que não é dia nem NDA: fora do padrão
    if len(dias_alt) < 3:
        return None
    dias_en = {_DIAS[m.group(1).lower()] for m in _DIA_RX.finditer(en)}
    if len(dias_en) != 1 or len(_N_DIAS.findall(en)) != 1:
        return None
    inicio = next(iter(dias_en))
    dur = _DURACAO.search(en)
    desl = _DESLOCAMENTO.search(en)
    if dur and _PERGUNTA_FIM.search(en) and not desl:
        alvo = (inicio + int(dur.group(1)) - 1) % 7
    elif desl and not dur:
        alvo = (inicio + int(desl.group(1) or desl.group(2))) % 7
    else:
        return None
    nomes = ["domingo", "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado"]
    verdadeiras = [L for L in LETRAS if dias_alt.get(L) == alvo]
    return _veredito_verificador("dias_semana", gab, verdadeiras, nomes[alvo], alts)


_PEDE_DECIMAL = re.compile(r"(número decimal|numero decimal|representação decimal|representacao decimal"
                           r"|forma decimal|na forma de decimal|em decimal)", re.I)
# "decima(?:is|l)": singular E plural. Revisão do passo 2 (2026-10-01): o
# padrão antigo "casas? decimais?" não casava "uma casa decimal" (o "?" só
# torna o "s" opcional: "decimai"), e "5/11 arredondado para uma casa
# decimal" com gabarito 0,45 saía CONFIRMADO pela regra das duas casas.
_APROX = re.compile(r"(casas? decima(?:is|l)|aproximad|arredond|cerca de|centésimo|centesimo|décimo|decimo"
                    r"|milésimo|milesimo)", re.I)
# Precisão pedida no enunciado: "duas casas decimais", "2 casas decimais",
# "até os centésimos". Qualquer outra ("uma casa", "décimos", "três casas")
# faz o verificador se abster.
_DUAS_CASAS = re.compile(r"\b(duas|2)\s+casas\s+decimais\b|cent[ée]sim", re.I)
_OUTRA_PRECISAO = re.compile(r"\b(uma|um|tr[êe]s|quatro|1|3|4)\s+casas?\s+decima(?:is|l)\b|d[ée]cimos?\b"
                             r"|mil[ée]sim", re.I)
_FRASE_CASAS = re.compile(r"\b(?:\d+|uma|duas|tr[êe]s|quatro)\s+casas?\s+decima(?:is|l)\b", re.I)
# H2 vale "quando o enunciado não exige outra precisão": pedir o valor EXATO
# ("representação decimal exata de 5/11") exige a dízima; 0,45 aí é falso.
_PEDE_EXATO = re.compile(r"\bexat[oa]s?\b", re.I)
_FRACAO = re.compile(r"(?<![\d,./])(\d+)\s*/\s*(\d+)(?![\d/])|\b(\d+)\s+de cada\s+(\d+)\b", re.I)
_DECIMAL_ALT = re.compile(r"^\s*(\d+)(?:[.,](\d+))?\s*(\.\.\.|…)?\s*$")
# H2: só DUAS casas decimais contam como resposta "do básico" para dízima.
CASAS_ARREDONDAMENTO = 2


def _truncar(fr, casas):
    from fractions import Fraction
    from math import floor
    return Fraction(floor(fr * 10 ** casas), 10 ** casas)


def _arredondar(fr, casas):
    """Arredondamento escolar (meio para cima), exato em Fraction."""
    from fractions import Fraction
    from math import floor
    return Fraction(floor(fr * 10 ** casas + Fraction(1, 2)), 10 ** casas)


def _verificar_fracao_decimal(q, alts, gab):
    """Fração -> decimal ("5 de cada 11 ... representação decimal").

    H2 (decisão do usuário, 2026-10-01): quando a expansão é uma dízima (ou o
    enunciado pede "aproximadamente"/"duas casas decimais"), a alternativa com
    o valor ARREDONDADO ou TRUNCADO em duas casas é verdadeira — no idx 1348,
    5/11 = 0,4545... e o gabarito "0,45" está certo para o ensino básico. Duas
    alternativas que casam pela regra (ex.: "0,45" e "0,4545...", ou o
    truncado e o arredondado quando diferem) deixam a resposta não única
    (contradiz). Aproximação com OUTRA precisão (uma casa, três casas) e
    precisão pedida diferente de duas casas: o verificador se abstém (None),
    porque a regra do usuário não cobre o caso e acusar seria palpite — a não
    ser que o valor EXATO esteja entre as alternativas (aí ele é a resposta e
    a aproximação não pedida é falsa). Enunciado que pede o valor EXATO
    ("representação decimal exata") desliga a regra das duas casas."""
    from fractions import Fraction
    en = str(q.get("enunciado") or "")
    if not _PEDE_DECIMAL.search(en):
        return None
    pede_aprox = bool(_APROX.search(en))
    pede_exato = bool(_PEDE_EXATO.search(en))
    if pede_aprox and pede_exato:
        return None  # "valor exato ... aproximado": contraditório, não decide
    if pede_aprox and _OUTRA_PRECISAO.search(en) and not _DUAS_CASAS.search(en):
        return None  # precisão pedida diferente de duas casas: fora da regra
    en_sem_casas = _FRASE_CASAS.sub(" ", en)
    fr = {(int(m.group(1) or m.group(3)), int(m.group(2) or m.group(4))) for m in _FRACAO.finditer(en_sem_casas)}
    if len(fr) != 1:
        return None
    a, b = next(iter(fr))
    if b == 0 or set(numeros(en_sem_casas)) != {str(a), str(b)}:
        return None  # há outros números no enunciado: o alvo pode não ser a fração
    alvo = Fraction(a, b)
    den = alvo.denominator
    while den % 2 == 0:
        den //= 2
    while den % 5 == 0:
        den //= 5
    periodica = den != 1
    # H2: dízima, ou aproximação pedida em duas casas (sem outra precisão);
    # nunca quando o enunciado pede o valor exato
    aceita_duas = (periodica or pede_aprox) and not pede_exato
    duas = {_truncar(alvo, CASAS_ARREDONDAMENTO), _arredondar(alvo, CASAS_ARREDONDAMENTO)}
    verdadeiras, n_num, exatas, aprox_fora = [], 0, [], []
    for L in LETRAS:
        t = str(alts.get(L, ""))
        m = _DECIMAL_ALT.match(t)
        if not m:
            if _NDA.search(t):
                continue
            return None
        n_num += 1
        inteiro, frac, reticencias = m.group(1), m.group(2) or "", m.group(3)
        casas = len(frac)
        valor = Fraction(int(inteiro + frac), 10 ** casas)
        if reticencias:
            # dízima escrita com reticências: os dígitos mostrados têm de ser o
            # começo da expansão e a fração tem de ser periódica
            ok = periodica and valor <= alvo < valor + Fraction(1, 10 ** casas)
            if ok:
                exatas.append(L)
        elif valor == alvo:
            ok = True
            exatas.append(L)
        elif casas == CASAS_ARREDONDAMENTO and aceita_duas and valor in duas:
            ok = True  # H2: 0,45 para 5/11
        elif valor != alvo and valor in (_truncar(alvo, casas), _arredondar(alvo, casas)) and not pede_exato:
            # aproximação do alvo com precisão que a regra não cobre ("0,5" ou
            # "0,455" para 5/11; "0,38" para 3/8 sem pedido de aproximação):
            # decide-se no fim, conforme o valor exato esteja ou não entre as
            # alternativas
            aprox_fora.append(L)
            ok = False
        else:
            ok = False
        if ok:
            verdadeiras.append(L)
    if n_num < 4:
        return None
    if aprox_fora and not exatas:
        # Sem o valor exato entre as alternativas, a aproximação fora da regra
        # pode ser a resposta pretendida: acusar seria palpite. COM o exato
        # presente (revisão do passo 2: "3/8" com 0,38 no gabarito e 0,375
        # entre as alternativas; "7/2" com 4 e 3,5), o exato é a resposta e a
        # aproximação não pedida é falsa — antes o verificador se abstinha e
        # o gabarito errado passava.
        return None
    esperado = (f"{a}/{b} = {float(alvo):.6g}{' (dízima periódica)' if periodica else ''}"
                + (f"; duas casas: {', '.join(sorted({str(float(x)).replace('.', ',') for x in duas}))}"
                   if aceita_duas else ""))
    return _veredito_verificador("fracao_decimal", gab, verdadeiras, esperado, alts)


_SOMA_ALT = re.compile(r"^\s*\d[\d.]*(?:\s*\+\s*\d[\d.]*)+\s*\.?\s*$")
_ORDENS = re.compile(r"(centena|dezena|unidade|ordem|ordens|milhar)", re.I)


def _verificar_decomposicao(q, alts, gab):
    """"Outra forma de representar N" com alternativas que são somas: conta
    quantas somas dão N. Só com UM número no enunciado e sem restrição de
    forma (centenas/dezenas/unidades restringem quais somas valem)."""
    en = str(q.get("enunciado") or "")
    nums = numeros(en)
    if len(nums) != 1 or _ORDENS.search(en):
        return None
    try:
        alvo = int(nums[0])
    except ValueError:
        return None
    verdadeiras, n_soma = [], 0
    for L in LETRAS:
        t = str(alts.get(L, ""))
        if not _SOMA_ALT.match(t):
            if _NDA.search(t):
                continue
            return None
        n_soma += 1
        parcelas = [int(_num_canonico(x)) for x in re.findall(r"\d[\d.]*\d|\d", t.rstrip(". "))]
        if sum(parcelas) == alvo:
            verdadeiras.append(L)
    if n_soma < 4:
        return None
    return _veredito_verificador("decomposicao_aditiva", gab, verdadeiras, str(alvo), alts)


# --- número "por extenso" (passo 2, 2026-10-01) ------------------------------
# Por que: na regressão r7 o validador novo (prompt com a regra H2) passou a
# APROVAR o idx 1052 (DIST-H03-Difícil-00273: 307 "por extenso", com o
# distrator C "Três centenas e sete unidades", que também descreve 307 — D5:
# só o gabarito pode ser verdadeiro). Os dois juízes aprovaram juntos. Em vez
# de mexer de novo no prompt (invalida o cache e custa a recalibração
# inteira), a conta vai para um verificador EXATO e estreito: enunciado com UM
# número que pede a escrita por extenso; cada alternativa é o nome do número
# (comparado com a grafia canônica, masculina ou feminina), uma decomposição
# em ordens ("3 centenas e 7 unidades"), uma soma ou o próprio numeral. Grafia
# errada ("Trezentos sete", sem o "e") e outro número são falsas. Qualquer
# alternativa fora desses formatos, ou a ambiguidade do "e" depois de "mil",
# faz o verificador se abster.
_EXTENSO_PEDIDO = re.compile(r"por extenso|como se escreve|como se l[êe]\b|escrit[oa] (?:corretamente )?(?:com|em) "
                             r"palavras", re.I)
_UNID = ["zero", "um", "dois", "tres", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez", "onze", "doze",
         "treze", "catorze", "quinze", "dezesseis", "dezessete", "dezoito", "dezenove"]
_DEZ = ["", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta", "oitenta", "noventa"]
_CENT = ["", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos", "seiscentos", "setecentos",
         "oitocentos", "novecentos"]
_PALAVRAS_NUMERO = set(_UNID + _DEZ[2:] + _CENT[1:] + ["cem", "mil", "e", "uma", "duas", "quatorze"]
                       + [c[:-2] + "as" for c in _CENT[2:]])
_QTD = {w: i for i, w in enumerate(_UNID[:11])}
_QTD.update({"uma": 1, "duas": 2})
_PESO_ORDEM = (("unidades de milhar", 1000), ("unidade de milhar", 1000), ("milhares", 1000), ("milhar", 1000),
               ("centenas", 100), ("centena", 100), ("dezenas", 10), ("dezena", 10), ("unidades", 1),
               ("unidade", 1))
_TERMO_ORDEM = re.compile(r"(\d+|" + "|".join(sorted(_QTD, key=len, reverse=True)) + r")\s+("
                          + "|".join(o for o, _ in _PESO_ORDEM) + r")\b")


def _ate_999(n):
    if n == 100:
        return "cem"
    c, r = divmod(n, 100)
    partes = [_CENT[c]] if c else []
    if r:
        if r < 20:
            partes.append(_UNID[r])
        else:
            t, u = divmod(r, 10)
            partes.append(_DEZ[t] + (" e " + _UNID[u] if u else ""))
    return " e ".join(partes)


def numero_por_extenso(n):
    """(grafia canônica, grafia com o outro conector depois de "mil" ou None),
    masculina, sem acentos, 0 <= n < 1.000.000. Regra usual: depois de "mil"
    vai "e" quando o resto é < 100 ou um número redondo de centenas
    ("mil e duzentos", "dois mil e cinco"); senão vai só espaço."""
    if not 0 <= n < 1_000_000:
        raise ValueError(n)
    if n == 0:
        return "zero", None
    mil, r = divmod(n, 1000)
    if not mil:
        return _ate_999(r), None
    cab = "mil" if mil == 1 else _ate_999(mil) + " mil"
    if not r:
        return cab, None
    com_e = r < 100 or r % 100 == 0
    return cab + (" e " if com_e else " ") + _ate_999(r), cab + (" " if com_e else " e ") + _ate_999(r)


def _feminino(t):
    t = re.sub(r"\bum\b", "uma", t)
    t = re.sub(r"\bdois\b", "duas", t)
    return re.sub(r"(\w+)entos\b", r"\1entas", t)


def _norm_extenso(t):
    t = unicodedata.normalize("NFKD", str(t or "")).encode("ascii", "ignore").decode().lower()
    t = t.replace("quatorze", "catorze")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+., ]", " ", t)).strip(" .,")


def _valor_ordens(t):
    """Valor de "3 centenas e 7 unidades" / "três centenas, zero dezenas e sete
    unidades"; None se o texto não for só isso."""
    termos = list(_TERMO_ORDEM.finditer(t))
    if not termos:
        return None
    resto = _TERMO_ORDEM.sub(" ", t)
    if re.sub(r"[ ,]|\be\b", "", resto):
        return None
    peso = dict(_PESO_ORDEM)
    total = 0
    for m in termos:
        q = m.group(1)
        total += (int(q) if q.isdigit() else _QTD[q]) * peso[m.group(2)]
    return total


def _verificar_extenso(q, alts, gab):
    en = str(q.get("enunciado") or "")
    nums = numeros(en)
    if len(nums) != 1 or not _EXTENSO_PEDIDO.search(en):
        return None
    try:
        alvo = int(nums[0])
        canon, outra = numero_por_extenso(alvo)
    except ValueError:
        return None
    validas = {canon, _feminino(canon)}
    verdadeiras = []
    for L in LETRAS:
        bruto = str(alts.get(L, ""))
        if _NDA.search(bruto):
            continue
        t = _norm_extenso(bruto)
        if outra is not None and t in {outra, _feminino(outra)}:
            return None                                   # conector do "mil" discutível: abstém-se
        if t in validas:
            verdadeiras.append(L)
            continue
        if _ORDENS.search(t):
            v = _valor_ordens(t)
            if v is None:
                return None
        elif _SOMA_ALT.match(t):
            v = sum(int(_num_canonico(x)) for x in re.findall(r"\d[\d.]*\d|\d", t))
        elif re.fullmatch(r"\d[\d.]*", t):
            v = int(_num_canonico(t))
        elif t and set(t.split()) <= _PALAVRAS_NUMERO:
            continue                                      # nome de outro número ou grafia errada: falsa
        else:
            return None
        if v == alvo:
            verdadeiras.append(L)
    return _veredito_verificador("numero_por_extenso", gab, verdadeiras, f"{alvo} = {canon}", alts)


def verificar_aritmetica(questao):
    """{"status": confirma|contradiz|sem_veredito, ...} do primeiro verificador
    exato que reconhece o item (ver comentário acima)."""
    alts = questao.get("alternativas") or {}
    gab = questao.get("resposta_correta")
    if not isinstance(alts, dict) or gab not in LETRAS:
        return {"status": "sem_veredito"}
    for fn in (_verificar_dias_semana, _verificar_fracao_decimal, _verificar_decomposicao, _verificar_extenso):
        try:
            r = fn(questao, alts, gab)
        except (ValueError, KeyError, ZeroDivisionError):  # padrão quase reconhecido: abstém-se
            r = None
        if r is not None:
            return r
    return {"status": "sem_veredito"}


# D6 (revisão do passo 2, 2026-10-01): alternativa CORROMPIDA PELA PLANILHA.
# Na importação do banco, o Excel converteu frações em data ("3/6" ->
# 3 de junho -> "2025-06-03 00:00:00"). 17 itens reais da base (9º H07, H08,
# H09, H25; ex.: MT90123MH25MT, "probabilidade de sair número par", com
# gabarito A = "2025-06-03 00:00:00") e 3 de cada val têm isso; nenhum filtro
# via, porque as alternativas são distintas e nenhum verificador reconhece o
# item. Uma questão assim não tem resposta: o modelo aprenderia a responder
# fração com data. O formato ISO com ano de 4 dígitos nunca aparece num item
# de verdade (datas do SAEB vêm como "12/03/2024" ou "12 de março").
_DATA_PLANILHA = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\s*$")


def alternativas_corrompidas_planilha(questao):
    """{letra: fração proposta} das alternativas que são uma data ISO vinda da
    planilha. A proposta é dia/mês (o Excel em pt-BR lê "3/6" como 3 de
    junho; nos 17 itens da base a resolução confere: MT9031, 3/4 -> 2025-04-03;
    MT9043, 2/10 -> 2025-10-02). É só PROPOSTA para revisão humana: a
    restauração não é aplicada automaticamente."""
    out = {}
    for L, v in ((questao or {}).get("alternativas") or {}).items():
        m = _DATA_PLANILHA.match(str(v))
        if m:
            out[L] = f"{int(m.group(3))}/{int(m.group(2))}"
    return out


def filtro_injecao(obj, raw, *, ano, habilidade, dificuldade, slot=None, vistos=(),
                   comparar_com=(), val_questoes=(), taxonomia=None, assinaturas_base=None,
                   assinaturas_val=None, respostas_usadas=None, limite_resposta=None):
    """Filtros determinísticos da INJEÇÃO. Devolve (motivo|None, detalhes).

    T1: `assinaturas_base`/`assinaturas_val` são índices {assinatura_dados:
    codigo} de TODA a base/validação (não só da habilidade); a duplicata
    semântica também é procurada em `comparar_com` (aceitas da habilidade).
    T4: `respostas_usadas` (Counter de resposta_conteudo das aceitas da
    habilidade) e `limite_resposta`: a mesma resposta não numérica não pode
    passar de `limite_resposta` aceitas na habilidade.

    Usa distill_teacher.filtrar com CÓPIA de `vistos` e aceitas=None: filtrar()
    MUTA vistos/aceitas quando aprova — chamado antes dos agentes LLM,
    registraria como "aceita" uma questão que o validador reprovaria depois e
    ela passaria a bloquear near-duplicatas. Aqui nada é mutado; quem grava a
    questão aceita atualiza o estado DEPOIS do veredito duplo.

    `comparar_com`: questões da mesma (ano, habilidade) na base + já aceitas
    (near-duplicata). `val_questoes`: questões de validação da mesma (ano,
    habilidade) — injetar algo parecido contaminaria a avaliação."""
    detalhes = {}
    motivo = dt.filtrar(obj, raw, set(vistos), strict=False, slot=slot, ano=ano,
                        habilidade=habilidade, aceitas=None, taxonomia=taxonomia)
    if motivo:
        return motivo, detalhes
    q = extract_questao(obj, 0)
    defeitos = defeitos_resolucao(q)
    if defeitos:
        return defeitos[0], detalhes
    arit = verificar_aritmetica(q)
    if arit["status"] == "contradiz":
        return "verificador_aritmetico", {"aritmetica": arit}
    if alternativas_corrompidas_planilha(q):
        return "alternativa_corrompida_planilha", {"planilha": alternativas_corrompidas_planilha(q)}
    texto =" ".join([str(q.get("enunciado", ""))] + [str(v) for v in q["alternativas"].values()])
    if depende_de_visual_ausente(texto):
        return "dependencia_visual", detalhes
    esperado = DIFFICULTY_MAP.get(dificuldade)
    if esperado and q.get("difficulty") != esperado:
        return "difficulty_divergente", {"esperado": esperado, "obtido": q.get("difficulty")}
    if contexto_inverossimil(q):
        return "contexto_inverossimil", detalhes
    for outra in comparar_com:
        if diversidade.e_near_duplicata(q, outra):
            return "near_duplicata", {"com": str(outra.get("enunciado", ""))[:120]}
    for outra in val_questoes:
        if diversidade.e_near_duplicata(q, outra):
            return "contamina_val", {"com": str(outra.get("enunciado", ""))[:120]}
    assin = assinatura_dados(q)
    if assin is not None:
        if assinaturas_val and assin in assinaturas_val:
            return "contamina_val", {"semantica": True, "com": assinaturas_val[assin]}
        for outra in val_questoes:
            if e_duplicata_semantica(q, outra):
                return "contamina_val", {"semantica": True, "com": str(outra.get("enunciado", ""))[:120]}
        if assinaturas_base and assin in assinaturas_base:
            return "duplicata_semantica", {"com": assinaturas_base[assin], "assinatura": repr(assin)[:160]}
        for outra in comparar_com:
            if e_duplicata_semantica(q, outra):
                return "duplicata_semantica", {"com": str(outra.get("enunciado", ""))[:120],
                                               "assinatura": repr(assin)[:160]}
    rc = resposta_conteudo(q)
    if rc and respostas_usadas is not None and limite_resposta and respostas_usadas.get(rc, 0) >= limite_resposta:
        return "resposta_concentrada", {"resposta": rc, "ja_usada": respostas_usadas.get(rc, 0),
                                        "limite": limite_resposta}
    geo = veredito_geometria(q, ano, habilidade)
    detalhes["geometria"] = geo
    if geo["status"] == "reprovada":
        return "geometria", detalhes
    # P1 (2026-10-01): D5 determinística ANTES de pagar os juízes. No piloto 2
    # o 9º H17 rendeu 1 aceita em 12 candidatas: o gerador insiste em
    # "Isósceles" no equilátero e "Paralelogramo" no losango, e cada uma dessas
    # custava validador + revisor para ser barrada.
    d5 = veredito_d5_geometria(q)
    if d5["status"] == "reprovada":
        detalhes["geometria_d5"] = d5["detalhe"]
        return "geometria_d5", detalhes
    return None, detalhes


def filtros_auditoria(questao, *, subtema=None):
    """Filtros PUROS para a auditoria da base (sem duplicata exata contra si
    mesma, sem estado). Devolve dict; filtros["reprovado"] diz se algum filtro
    que derruba a confiança falhou. muito_longa é só informativo."""
    obj = {"questoes": [questao]}
    flags = check_structure(obj, quantidade_esperada=1)
    estrutura = bool(flags["wrapper_valido"] and flags["schema_completo"] and flags["resposta_valida"]
                     and flags["alternativas_distintas"] and flags["difficulty_valida"])
    out = {"estrutura": estrutura, "visual": False, "dados_ausentes": False, "consistencia": None,
           "geometria": "indisponivel", "muito_longa": False}
    if not estrutura:
        out["reprovado"] = ["estrutura"]
        return out
    texto = " ".join([str(questao.get("enunciado", ""))] + [str(v) for v in questao["alternativas"].values()])
    out["visual"] = bool(depende_de_visual_ausente(texto))
    out["menciona_figura"] = bool(IMAGE_PATTERN.search(json.dumps(questao, ensure_ascii=False)))
    out["dados_ausentes"] = bool(diversidade.dados_ausentes(questao, subtema))
    ok, _ = check_consistency(questao)
    out["consistencia"] = ok
    out["muito_longa"] = len(json.dumps(questao, ensure_ascii=False)) > dt.MAX_ANSWER_CHARS
    geo = veredito_geometria(questao)
    out["geometria"] = geo if geo["status"] != "indisponivel" else "indisponivel"
    out["resolucao"] = defeitos_resolucao(questao)
    out["aritmetica"] = verificar_aritmetica(questao)
    # D5 de geometria: só INFORMA na auditoria (não entra em "reprovado" nem nos
    # filtros matemáticos da D2): remover item da base por ela exigiria a
    # confirmação dos juízes e do árbitro, como qualquer erro matemático.
    d5 = veredito_d5_geometria(questao)
    if d5["status"] == "reprovada":
        out["geometria_d5"] = d5["detalhe"]
    # D6: data da planilha no lugar da fração. Reprova (derruba a confiança e
    # vai para a lista), mas NÃO é filtro matemático da D2: quem tira o item da
    # v3 é a regra própria do --montar, que não depende de juiz nem de árbitro.
    out["planilha"] = alternativas_corrompidas_planilha(questao)
    reprov = [k for k, ruim in (("visual", out["visual"]), ("dados_ausentes", out["dados_ausentes"]),
                                ("alternativa_corrompida_planilha", bool(out["planilha"])),
                                ("consistencia", ok is False),
                                ("geometria", isinstance(geo, dict) and geo["status"] == "reprovada"),
                                ("verificador_aritmetico", out["aritmetica"]["status"] == "contradiz")) if ruim]
    reprov += out["resolucao"]
    out["reprovado"] = reprov
    return out


# ===========================================================================
# Agentes
# ===========================================================================

# P3 (2026-10-01): o revisor "cego" recebia o subtema, e em 8 itens de sólidos
# da base (5º H12/H13, 2º H12: subtema "cilindro", alternativa "Cilindro") o
# rótulo do subtema ERA a resposta. Pela regra de tokens abaixo, 15 itens da
# base vazam (inclui "raio e diâmetro" com "Raio"/"Diâmetro" e "retângulos e
# quadrados" com "Retângulo"/"Quadrado"). Nesses, o subtema não vai — e a
# mensagem diz "não especificado", igual a quando não há subtema, para não
# avisar que ele foi omitido.
_STOP_SUBTEMA = {"e", "de", "do", "da", "dos", "das", "o", "a", "os", "as", "com", "em", "um", "uma", "ou",
                 "no", "na", "nos", "nas"}


def _tokens_rotulo(texto):
    """Palavras normalizadas (sem acento, singular simples, sem artigos)."""
    ws = [w for w in re.findall(r"[a-z]+", diversidade.normalizar_texto(texto).replace("_", " "))
          if w not in _STOP_SUBTEMA]
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in ws}


def subtema_vaza_resposta(subtema, alternativas):
    """True se alguma alternativa NÃO numérica (até 3 palavras) está contida no
    rótulo do subtema: mandar o subtema entregaria a resposta ou estreitaria as
    opções ("cilindro" -> "Cilindro"; "raio e diâmetro" -> "Raio")."""
    st = _tokens_rotulo(subtema or "")
    if not st:
        return False
    for v in (alternativas or {}).values():
        if re.search(r"\d", str(v)):
            continue
        t = _tokens_rotulo(str(v))
        if t and len(t) <= 3 and t <= st:
            return True
    return False


def subtema_para_revisor(subtema, alternativas):
    """Subtema a mostrar ao revisor, ou None quando ele vaza a resposta."""
    if not subtema or subtema_vaza_resposta(subtema, alternativas):
        return None
    return subtema


def _falha(papel, erro, extra=None):
    """Resultado fail-closed: veredito False, avaliado False."""
    r = dict(extra or {})
    # aplicados POR ÚLTIMO: nenhum campo de `extra` pode reverter o fail-closed
    r.update({"veredito": False, "avaliado": False, "erro": erro, "resposta_calculada": None,
              "problemas": [{"codigo": erro, "detalhe": f"{papel}: {erro}"}], "confianca": None})
    return r


class Agentes:
    """Orquestra as chamadas dos papéis com orçamento, retries e log.

    clientes: {papel: cliente} — um cliente pode ser compartilhado.
    modelos:  {papel: nome do modelo} — só para registro.
    """

    def __init__(self, clientes, modelos=None, orcamento=None, log_uso=LOG_USO_PADRAO,
                 tentativas=3, backoff=2.0, dormir=time.sleep, simulado=False, cache_juizes=None):
        faltam = [p for p in PAPEIS if p not in clientes]
        if faltam:
            raise ValueError(f"clientes ausentes para os papéis: {faltam}")
        self.clientes = clientes
        self.modelos = dict(MODELOS_PADRAO, **(modelos or {}))
        self.orcamento = orcamento or Orcamento(None)
        self.log_uso = Path(log_uso) if log_uso else None
        self.tentativas = max(1, int(tentativas))
        self.backoff = backoff
        self.dormir = dormir
        self.simulado = simulado
        # CacheRespostas opcional (só juízes): acerto não gasta orçamento nem
        # vai para o log de uso; resposta nova paga é gravada nele.
        self.cache_juizes = cache_juizes

    # -- chamada genérica ---------------------------------------------------
    def _log(self, reg):
        if not self.log_uso:
            return
        with abrir_para_gravar(self.log_uso, "a") as f:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")

    def chamar(self, papel, fase, mensagens):
        """Envia com retries/backoff. Devolve (texto|None, info). Levanta
        OrcamentoEsgotado ANTES de enviar se o teto seria ultrapassado."""
        modelo = self.modelos[papel]
        if self.cache_juizes is not None:
            r = self.cache_juizes.obter(papel, modelo, mensagens)
            if r is not None:
                return r["texto"], {"cache": True, "tentativas": 0, "prompt_tokens": r.get("prompt_tokens"),
                                    "completion_tokens": r.get("completion_tokens")}
            texto, info = self._chamar_api(papel, fase, mensagens, modelo)
            self.cache_juizes.guardar(papel, modelo, mensagens, texto, info)
            return texto, info
        return self._chamar_api(papel, fase, mensagens, modelo)

    def _chamar_api(self, papel, fase, mensagens, modelo):
        cliente = self.clientes[papel]
        max_tokens = max_tokens_para(papel, modelo)
        temp = TEMPERATURAS[papel]
        ultimo_erro = None
        for tentativa in range(1, self.tentativas + 1):
            self.orcamento.reservar(papel)
            t0 = time.monotonic()
            reg = {"ts": agora_iso(), "agente": papel, "fase": fase, "modelo": modelo,
                   "tentativa": tentativa, "simulado": self.simulado}
            try:
                resp = cliente.chat_completion(messages=mensagens, max_tokens=max_tokens, temperature=temp)
                texto = resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - rede/quota/provedor
                ultimo_erro = _sanitizar(f"{type(exc).__name__}: {exc}")
                reg.update(ok=False, erro=ultimo_erro, latencia_s=round(time.monotonic() - t0, 2),
                           prompt_tokens=0, completion_tokens=0, estimado=True)
                self._log(reg)
                if tentativa < self.tentativas and _retentavel(exc):
                    self.dormir(self.backoff * (2 ** (tentativa - 1)))
                    continue
                return None, {"erro": ultimo_erro, "tentativas": tentativa}
            usage = _ler_usage(resp)
            estimado = usage is None
            if estimado:
                usage = (sum(_estimar_tokens(m.get("content", "")) for m in mensagens), _estimar_tokens(texto))
            self.orcamento.registrar_tokens(papel, usage[0], usage[1], estimado)
            fim = getattr(resp.choices[0], "finish_reason", None)
            reg.update(ok=True, latencia_s=round(time.monotonic() - t0, 2), prompt_tokens=usage[0],
                       completion_tokens=usage[1], estimado=estimado)
            if fim is not None:
                reg["finish_reason"] = str(fim)
            self._log(reg)
            return texto, {"tentativas": tentativa, "prompt_tokens": usage[0], "completion_tokens": usage[1],
                           "finish_reason": None if fim is None else str(fim)}
        return None, {"erro": ultimo_erro, "tentativas": self.tentativas}

    # -- GERADOR ------------------------------------------------------------
    def gerar(self, ano, habilidade, descricao, dificuldade, slot=None, letra_alvo=None, evitar_respostas=()):
        """Uma questão nova. Devolve {texto, obj, questao, erro}. O prompt do
        PROFESSOR leva aderência + regras SAEB + letra-alvo (+ Difícil nos anos
        iniciais, D4; + respostas a evitar, T4); o de TREINO (montado por quem
        grava) é só USER_TEMPLATE + sufixo de diversidade."""
        user = dt.prompt_professor(ano, habilidade, descricao, dificuldade, slot)
        if dificuldade == "Difícil" and ano in ANOS_INICIAIS:
            user += DIFICIL_ANOS_INICIAIS_ADDENDUM.format(ano=ano)
        if evitar_respostas:
            user += RESPOSTAS_USADAS_ADDENDUM.format(lista=", ".join(f'"{r}"' for r in evitar_respostas))
        if letra_alvo:
            user += LETRA_ALVO_ADDENDUM.format(letra=letra_alvo)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT + dt.TEACHER_ADDENDUM + "\n" + GERADOR_ADDENDUM},
                {"role": "user", "content": user}]
        texto, info = self.chamar("gerador", "geracao", msgs)
        if texto is None:
            return {"texto": None, "obj": None, "questao": None, "erro": "erro_api", "info": info}
        obj = extrair_json(texto, ("questoes",)) or parse_json(texto)
        return {"texto": texto, "obj": obj, "questao": extract_questao(obj, 0) if obj else None,
                "erro": None if obj else "json_invalido", "info": info}

    # -- VALIDADOR ----------------------------------------------------------
    def validar(self, questao):
        """Contrato: {veredito, resposta_calculada, problemas, confianca, avaliado,
        fases, chamadas, permutacao}. resposta_calculada na letra ORIGINAL."""
        alts = questao.get("alternativas") or {}
        gab = questao.get("resposta_correta")
        perm = permutacao(questao)
        exib = {L: alts.get(perm[L], "") for L in LETRAS}
        user1 = VALIDADOR_USUARIO.format(enunciado=questao.get("enunciado", ""), **exib)
        chamadas = 1
        texto, info = self.chamar("validador", "fase1", [{"role": "system", "content": VALIDADOR_SISTEMA},
                                                         {"role": "user", "content": user1}])
        if texto is None:
            return _falha("validador", "erro_api", {"chamadas": chamadas, "permutacao": perm})
        f1 = interpretar_fase1(texto)
        if f1 is None:
            return _falha("validador", "veredito_malformado", {"chamadas": chamadas, "permutacao": perm,
                                                               "bruto": _sanitizar(texto, 400)})
        status_orig = {perm[L]: f1["status"][L] for L in LETRAS}
        resp_exib = resposta_unica(f1["status"])
        resp = perm[resp_exib] if resp_exib else None
        problemas = list(f1["problemas"])
        if f1["resposta_llm"] != resp_exib:
            # vale o recálculo a partir dos status; a divergência é registrada
            problemas.append({"codigo": "incoerencia_interna",
                              "detalhe": f"LLM disse {f1['resposta_llm']}, status implicam {resp_exib}"})
        equivalentes = [[perm[a], perm[b]] for a, b in f1["equivalentes"]]
        if not f1["premissa_possivel"] and not any(p["codigo"] == "premissa_impossivel" for p in problemas):
            problemas.append({"codigo": "premissa_impossivel", "detalhe": "premissa_possivel=false"})
        if equivalentes and not any(p["codigo"] == "alternativas_equivalentes" for p in problemas):
            problemas.append({"codigo": "alternativas_equivalentes", "detalhe": str(equivalentes)})
        if f1["dependencia_visual"] and not any(p["codigo"] == "dependencia_visual" for p in problemas):
            problemas.append({"codigo": "dependencia_visual", "detalhe": "dependencia_visual=true"})
        # D5 (revisão do piloto 2): só o gabarito pode ser verdadeiro. Uma classe
        # mais geral que também vale (G) é distrator verdadeiro — no validador
        # também, não só no revisor. Antes o G contava como F aqui: no piloto 2
        # o validador aprovou, com confiança 0,95–1,0, os 5 candidatos do 9º H17
        # que violam a D5 (losango com "Paralelogramo", equilátero com
        # "Isósceles"...), e a D2 nunca confirmava esse erro.
        gerais = [perm[L] for L in LETRAS if f1["status"][L] == "G"]
        if gerais and not any(p["codigo"] == "distrator_verdadeiro" for p in problemas):
            problemas.append({"codigo": "distrator_verdadeiro",
                              "detalhe": f"classe mais geral também vale: {sorted(gerais)}"})
        base = {"resposta_calculada": resp, "status": status_orig, "equivalentes": equivalentes,
                "permutacao": perm, "avaliado": True, "fases": 1, "chamadas": chamadas,
                "resolucao": f1["resolucao"]}
        if resp != gab:
            problemas.append({"codigo": "gabarito_sem_resposta_unica" if resp is None else "gabarito_errado",
                              "detalhe": f"cego={resp} gabarito={gab}"})
            return dict(base, veredito=False, problemas=problemas, confianca=f1["confianca"])
        # Fail-closed: qualquer problema que o próprio validador apontou reprova,
        # mesmo com a letra batendo (ex.: pergunta_ambigua) — sem gastar a fase 2.
        if problemas:
            return dict(base, veredito=False, problemas=problemas, confianca=f1["confianca"])

        # Fase 2: ordem ORIGINAL (a resolução do autor pode citar letras).
        user2 = (VALIDADOR_USUARIO.format(enunciado=questao.get("enunciado", ""),
                                          **{L: alts.get(L, "") for L in LETRAS})
                 + "\n\n" + VALIDADOR_FASE2.format(letra=gab, resolucao=questao.get("resolucao_passo_a_passo", "")))
        chamadas += 1
        texto2, _ = self.chamar("validador_fase2", "fase2", [{"role": "system", "content": VALIDADOR_FASE2_SISTEMA},
                                                             {"role": "user", "content": user2}])
        base.update(fases=2, chamadas=chamadas)
        if texto2 is None:
            return _falha("validador", "erro_api", base)
        f2 = interpretar_fase2(texto2)
        if f2 is None:
            return _falha("validador", "veredito_malformado", dict(base, bruto=_sanitizar(texto2, 400)))
        problemas += f2["problemas"]
        for campo, ruim, codigo in (("resolucao_correta", False, "resolucao_errada"),
                                    ("usa_dado_ausente", True, "resolucao_usa_dado_ausente"),
                                    ("justificativa_falsa", True, "justificativa_falsa"),
                                    ("chega_na_alternativa", False, "resolucao_nao_conclui")):
            if f2[campo] is ruim and not any(p["codigo"] == codigo for p in problemas):
                problemas.append({"codigo": codigo, "detalhe": f"{campo}={f2[campo]}"})
        ok = (f2["resolucao_correta"] and not f2["usa_dado_ausente"] and not f2["justificativa_falsa"]
              and f2["chega_na_alternativa"] and not problemas)
        confs = [c for c in (f1["confianca"], f2["confianca"]) if c is not None]
        return dict(base, veredito=bool(ok), problemas=problemas, confianca=min(confs) if confs else None,
                    fase2={k: f2[k] for k in ("resolucao_correta", "usa_dado_ausente",
                                              "justificativa_falsa", "chega_na_alternativa")})

    # -- REVISOR ------------------------------------------------------------
    def revisar(self, questao, ano, habilidade, descricao, subtema=None, dificuldade=None):
        """Revisor CEGO (D1): uma chamada, sem gabarito, sem resolução do autor,
        sem difficulty, alternativas numa permutação própria. A resposta dele
        é comparada com o gabarito AQUI. Contrato igual ao anterior, mais
        dificuldade_real (D3) e permutacao."""
        alts = questao.get("alternativas") or {}
        gab = questao.get("resposta_correta")
        perm = permutacao(questao, sal="revisor")
        # `dificuldade` continua na assinatura (chamadores e calibração passam),
        # mas NÃO vai para a mensagem desde P3: ancorava a dificuldade_real.
        user = REVISOR_USUARIO.format(
            ano=ano, habilidade=habilidade, descricao=descricao or "",
            subtema=subtema_para_revisor(subtema, alts) or "não especificado",
            enunciado=questao.get("enunciado", ""), **{L: alts.get(perm[L], "") for L in LETRAS})
        texto, _ = self.chamar("revisor", "revisao_cega", [{"role": "system", "content": REVISOR_SISTEMA},
                                                           {"role": "user", "content": user}])
        if texto is None:
            return _falha("revisor", "erro_api", {"chamadas": 1, "permutacao": perm})
        r = interpretar_revisor(texto)
        if r is None:
            return _falha("revisor", "veredito_malformado", {"chamadas": 1, "permutacao": perm,
                                                             "bruto": _sanitizar(texto, 400)})
        resp_exib = resposta_unica(r["status"])
        resp = perm[resp_exib] if resp_exib else None
        status_orig = {perm[L]: r["status"][L] for L in LETRAS}
        problemas = list(r["problemas"])
        if r["resposta_llm"] != resp_exib:
            problemas.append({"codigo": "incoerencia_interna",
                              "detalhe": f"LLM disse {r['resposta_llm']}, status implicam {resp_exib}"})
        # D5: só o gabarito pode ser verdadeiro. Uma classe mais geral que também
        # vale (G) é um distrator verdadeiro — decidido pelo STATUS, no código.
        gerais = [L for L in LETRAS if status_orig[L] == "G"]
        if gerais and not any(p["codigo"] == "distrator_verdadeiro" for p in problemas):
            problemas.append({"codigo": "distrator_verdadeiro", "detalhe": f"classe mais geral também vale: {gerais}"})
        falhos = [c for c, v in r["criterios"].items() if not v["ok"]]
        # Avisos (C3/dificuldade) ficam registrados mas não vetam; ver
        # CRITERIOS_INFORMATIVOS.
        falhos_bloq = [c for c in falhos if c not in CRITERIOS_INFORMATIVOS]
        avisos = [p for p in problemas if p["codigo"] in CODIGOS_INFORMATIVOS]
        bloq_llm = [p for p in problemas if p["codigo"] not in CODIGOS_INFORMATIVOS]
        # O LLM foi instruído a dar veredito false se QUALQUER critério falhar;
        # quando tudo o que ele apontou é aviso, o false dele vem só do aviso.
        so_avisos = bool(falhos or avisos) and not falhos_bloq and not bloq_llm
        if resp != gab and not any(p["codigo"] in ("gabarito_errado", "resposta_nao_unica") for p in problemas):
            problemas.append({"codigo": "gabarito_errado" if resp else "resposta_nao_unica",
                              "detalhe": f"revisor={resp} gabarito={gab}"})
        bloq = [p for p in problemas if p["codigo"] not in CODIGOS_INFORMATIVOS]
        if r["veredito_llm"] and (falhos_bloq or bloq) and resp == gab:
            problemas.append({"codigo": "incoerencia_interna",
                              "detalhe": f"veredito true com critérios {falhos_bloq} / problemas"})
            bloq = [p for p in problemas if p["codigo"] not in CODIGOS_INFORMATIVOS]
        ok = (r["veredito_llm"] or so_avisos) and not falhos_bloq and resp == gab and not bloq
        return {"veredito": bool(ok), "avaliado": True, "resposta_calculada": resp, "status": status_orig,
                "criterios": r["criterios"], "criterios_falhos": falhos, "criterios_bloqueantes_falhos": falhos_bloq,
                "avisos": [p["codigo"] for p in avisos] + [c for c in falhos if c in CRITERIOS_INFORMATIVOS],
                "problemas": problemas, "sugestoes": r["sugestoes"], "confianca": r["confianca"],
                "dificuldade_real": r["dificuldade_real"], "permutacao": perm, "chamadas": 1}

    # -- par validador + revisor -------------------------------------------
    def julgar(self, questao, ano, habilidade, descricao, subtema=None, dificuldade=None,
               curto_circuito=True):
        """Validador; revisor só se o validador aprovar (curto_circuito=True, na
        injeção) ou sempre (auditoria: precisamos dos dois para o nível de
        confiança). Os juízes nunca veem a saída um do outro."""
        val = self.validar(questao)
        rev = None
        if val["veredito"] or not curto_circuito:
            rev = self.revisar(questao, ano, habilidade, descricao, subtema, dificuldade)
        ambos = bool(val["veredito"] and rev is not None and rev["veredito"])
        return {"validador": val, "revisor": rev, "ambos": ambos}


def montar_agentes(args, simulado=False, log_uso=LOG_USO_PADRAO, orcamento=None, dormir=time.sleep):
    """Agentes a partir dos argumentos de linha de comando (compartilhado pelos
    CLIs). Em simulado, um único ClienteSimulado para todos os papéis."""
    modelos = {"gerador": getattr(args, "modelo_gerador", None) or MODELOS_PADRAO["gerador"],
               "validador": getattr(args, "modelo_validador", None) or MODELOS_PADRAO["validador"],
               "validador_fase2": getattr(args, "modelo_validador_fase2", None) or MODELOS_PADRAO["validador_fase2"],
               "revisor": getattr(args, "modelo_revisor", None) or MODELOS_PADRAO["revisor"]}
    if simulado:
        c = ClienteSimulado()
        clientes = {p: c for p in PAPEIS}
        modelos = {p: f"simulado:{m}" for p, m in modelos.items()}
    else:
        backends = {"gerador": getattr(args, "backend_gerador", "maritaca"),
                    "validador": getattr(args, "backend_validador", "maritaca"),
                    "validador_fase2": getattr(args, "backend_validador", "maritaca"),
                    "revisor": getattr(args, "backend_revisor", "maritaca")}
        cache = {}
        clientes = {}
        for p in PAPEIS:
            chave = (backends[p], modelos[p])
            if chave not in cache:
                cache[chave] = criar_cliente_real(*chave)
            clientes[p] = cache[chave]
    orc = orcamento or Orcamento(getattr(args, "max_chamadas", None), getattr(args, "max_tokens_total", None))
    return Agentes(clientes, modelos, orc,
                   log_uso=log_uso, tentativas=getattr(args, "tentativas_api", 3),
                   backoff=getattr(args, "backoff", 2.0), dormir=dormir, simulado=simulado)


def adicionar_args_modelos(parser):
    """Flags comuns de modelo/backend/orçamento dos CLIs."""
    g = parser.add_argument_group("modelos e orçamento")
    g.add_argument("--modelo-gerador", default=MODELOS_PADRAO["gerador"])
    g.add_argument("--modelo-validador", default=MODELOS_PADRAO["validador"])
    g.add_argument("--modelo-validador-fase2", default=MODELOS_PADRAO["validador_fase2"])
    g.add_argument("--modelo-revisor", default=MODELOS_PADRAO["revisor"],
                   help="padrão sabia-4-thinking (decisão de 2026-10-01: sem outra família/HF); o revisor "
                        "resolve às cegas, numa permutação própria das alternativas")
    g.add_argument("--backend-gerador", choices=("maritaca", "hf"), default="maritaca")
    g.add_argument("--backend-validador", choices=("maritaca", "hf"), default="maritaca")
    g.add_argument("--backend-revisor", choices=("maritaca", "hf"), default="maritaca")
    g.add_argument("--max-chamadas", type=int, required=False, default=None,
                   help="teto DURO de chamadas à API nesta execução (obrigatório fora do --dry-run)")
    g.add_argument("--max-tokens-total", type=int, default=None,
                   help="teto opcional de tokens (entrada + saída) nesta execução")
    g.add_argument("--tentativas-api", type=int, default=3, help="tentativas por chamada (com backoff)")
    g.add_argument("--backoff", type=float, default=2.0, help="segundos da 1ª espera (dobra a cada tentativa)")
    g.add_argument("--dry-run", action="store_true",
                   help="cliente SIMULADO (sem rede, sem custo); saídas vão para outputs/agentes/dryrun/")
    return g
