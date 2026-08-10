# Fine-tuning de um LLM de 1.7B para Geração Offline de Questões de Matemática no Padrão SAEB: Metodologia, Protocolo e Resultados Preliminares

**Documento de trabalho interno — base para publicação futura.**
Última atualização: 2026-08-05.
Repositório: `TreinamentoModeloQuestoes` (código, dados e artefatos referenciados neste documento).

---


## Resumo

Este documento registra a metodologia, o protocolo experimental e os resultados
preliminares do fine-tuning do **Qwen3-1.7B** (Qwen Team, 2025) para geração
automática de questões de matemática de múltipla escolha no padrão do
**SAEB** (Sistema de Avaliação da Educação Básica, INEP), com restrição de
rodar **totalmente offline em dispositivo móvel**. O trabalho combina (i)
fine-tuning supervisionado com **QLoRA** (Dettmers et al., 2023) sobre um
dataset pequeno de itens reais extraídos de um banco SAEB, (ii) duas
estratégias de aumento de dados — síntese aritmética determinística e
destilação filtrada de um modelo professor — desenhadas para mitigar um modo
de falha específico observado empiricamente (inconsistência entre o gabarito
gerado e o raciocínio que o justifica), e (iii) um mecanismo de verificação em
tempo de inferência (decodificação restrita por gramática formal + laço
gerar→checar→corrigir) que dá uma garantia determinística sobre parte da
saída, independente da qualidade do modelo. Reportamos números medidos em
cada etapa, sinalizando explicitamente quais resultados vêm de amostras
pequenas (tratamento estatístico correspondentemente cauteloso) e quais
etapas foram implementadas mas ainda não avaliadas após retreino no momento
da escrita.

> **Nota de migração de contrato**: após a integração com o app mobile, o
> schema JSON de saída foi migrado para um contrato fixo definido pela equipe
> de integração (wrapper `{"questoes": [...]}`, 5 alternativas A–E,
> `resposta_correta`, `resolucao_passo_a_passo`, `difficulty`), substituindo o
> schema usado durante o desenvolvimento (`gabarito`/`justificativas` por
> alternativa/`resposta`). Ver detalhes e trade-offs na Seção 3.2. **O modelo
> já foi retreinado sobre esse contrato** (517 exemplos: 273 reais + 244
> sintéticos, incluindo frações H07–H09) — resultados em Seção 5.5.

---

## 1. Introdução e Motivação

Uma equipe de desenvolvimento mobile adotou o Qwen3-1.7B como modelo base
para gerar questões de matemática offline em um aplicativo educacional. Dois
problemas motivaram este trabalho:

1. **Desempenho**: o modelo base, executado em modo padrão, gera blocos de
   raciocínio (`<think>`) longos antes da resposta, tornando a latência em
   CPU mobile impraticável.
2. **Confiabilidade de formato**: sem fine-tuning, o modelo frequentemente
   falha em produzir uma saída estruturada e utilizável pelo aplicativo.

Um terceiro problema — não previsto no desenho inicial, e só evidenciado após
testes reais do artefato exportado — foi identificado durante a execução do
projeto: mesmo produzindo JSON estruturalmente válido, o modelo por vezes
gera um gabarito (a alternativa marcada como correta) **inconsistente** com o
cálculo apresentado na própria justificativa. Este documento cobre tanto o
desenho original (velocidade e formato) quanto a resposta metodológica a esse
terceiro problema, que se tornou o foco principal das iterações mais
recentes.

## 2. Trabalhos Relacionados

O desenho do pipeline se apoia em quatro linhas de trabalho:

- **Parameter-efficient fine-tuning**: LoRA (Hu et al., 2021) introduz
  adaptadores de baixo posto (*low-rank*) treináveis sobre um modelo base
  congelado. QLoRA (Dettmers et al., 2023) estende essa ideia quantizando o
  modelo base em 4-bit NormalFloat (NF4), viabilizando fine-tuning de modelos
  com poucos GB de VRAM sem perda relevante de qualidade frente ao
  fine-tuning completo em 16-bit.
- **SFT com poucos exemplos**: LIMA (Zhou et al., 2023) mostra que um
  conjunto pequeno (~1.000 exemplos) e cuidadosamente curado é suficiente
  para ensinar um modelo a produzir saídas no **formato e estilo**
  desejados, sob a hipótese de que conhecimento substantivo já reside no
  pré-treino. Este resultado é a base teórica para a decisão de tratar o
  dataset real (~300 exemplos) como adequado para ensinar formato, mas
  insuficiente para ampliar capacidade aritmética — motivando as estratégias
  de aumento de dados (Seções 3.4–3.5).
- **Decodificação restrita por gramática (constrained/grammar-based
  decoding)**: garante corretude sintática restringindo o espaço de tokens
  válidos a cada passo de geração a uma gramática formal (BNF-like). Banerjee
  et al. (2025, CRANE) mostram que restringir *toda* a geração pode reduzir a
  capacidade de raciocínio do modelo, propondo alternar geração livre
  (raciocínio) com geração restrita (resposta final estruturada). Resultados
  em modelos pequenos (~1–3B) mostram ganhos de até 39 pontos percentuais
  absolutos em tarefas de múltipla escolha ao restringir a decodificação a
  tokens válidos (arXiv:2506.09408). Nossa implementação (Seção 3.9) usa uma
  gramática GBNF (formato do llama.cpp) sobre o schema JSON completo, não
  apenas sobre a letra final — uma diferença de escopo frente ao CRANE que
  deve ser discutida como limitação/trabalho futuro (Seção 8).
- **Reinforcement Learning com recompensas verificáveis (RLVR/GRPO)**: GRPO
  (Shao et al., 2024, DeepSeekMath) elimina o modelo crítico do PPO e estima
  a vantagem a partir da recompensa relativa dentro de um grupo de amostras,
  reduzindo custo computacional — usado depois em DeepSeek-R1 (DeepSeek-AI,
  2025) para incentivar raciocínio via recompensas de resultado verificável.
  Um estudo controlado em modelos pequenos (Qwen2.5-0.5B, GSM8K) mostra que a
  granularidade da recompensa é uma decisão de primeira ordem: recompensa por
  processo obteve 63,7% de acurácia contra 53,8% de recompensa só por
  resultado (arXiv:2607.02869). Esta literatura motiva o desenho listado como
  trabalho futuro (Seção 8): usar o verificador determinístico já
  implementado (`check_consistency()`, Seção 3.9) como função de recompensa.

O domínio de aplicação (SAEB) segue a Matriz de Referência do INEP, baseada
em Teoria de Resposta ao Item (TRI), que define habilidades/descritores por
ano escolar — a mesma estrutura (ano, habilidade, descritor, dificuldade)
usada como condicionamento de entrada em todo o pipeline (Seção 3.1).

## 3. Metodologia

### 3.1 Fonte e extração dos dados

Os dados de origem são um banco SQLite (`DB/questoes.db`, tabela `itens`)
contendo **553 itens** de avaliação no padrão SAEB, com os campos: enunciado,
texto auxiliar (comando), 4 alternativas (A–D), gabarito, justificativa geral
e por alternativa, ano escolar (2º/5º/9º), código de habilidade (H01–H27),
descrição do descritor e grau de dificuldade (Fácil/Moderado/Difícil).

**Critérios de exclusão** (implementados em `src/extract_data.py`):
- Disciplina ≠ Matemática ou campos essenciais nulos/"nan": **19 itens**
  descartados.
- Presença de imagem no enunciado ou em qualquer alternativa (o modelo de
  produção é puramente textual; treinar sobre itens com imagem ensinaria o
  modelo a referenciar figuras inexistentes): **230 itens** descartados.

Resultado: **303 itens válidos** (304 antes do filtro de alternativas
duplicadas descrito na Seção 5.4), cobrindo os 3 anos escolares, as 3
dificuldades e 27 habilidades distintas.

### 3.2 Formato do exemplo de treino

> **Nota de migração de contrato.** O schema descrito nesta seção é o
> **atual**, definido pela equipe responsável pela integração com o app
> mobile como um contrato fixo, sem exceção: qualquer campo, tipo ou valor de
> enum fora dele quebra o parsing no app. Ele substitui um schema anterior
> (`enunciado`/`comando`/`alternativas` A-D/`justificativas` por
> alternativa/`resposta`/`gabarito`) usado durante a fase de desenvolvimento
> deste pipeline — os resultados preliminares da Seção 5 e a análise de
> cobertura da Seção 5.4 foram medidos sob esse schema anterior. As Seções
> 3.2, 3.9 e 6 foram atualizadas para o contrato atual; o restante do texto
> (motivação, metodologia de treino, protocolo de avaliação) permanece válido
> — o schema de saída é ortogonal ao método de fine-tuning.

Cada item vira um exemplo de conversa (formato *chat*) com três mensagens:

- `system`: instrução fixa (`SYSTEM_PROMPT`) definindo o papel do modelo e o
  schema JSON de saída esperado.
- `user`: template fixo —
  `"Gere {quantidade} questão(ões) de matemática. Ano: {ano} ano. Habilidade:
  {habilidade} — {descricao}. Dificuldade: {dificuldade}."` — o mesmo
  condicionamento (ano, habilidade, descrição, dificuldade) usado pelo
  aplicativo em produção, mais a quantidade de questões pedidas numa única
  chamada (ver adiante).
- `assistant`: JSON compacto no contrato `{"questoes": [...]}`, onde cada
  questão tem as chaves `enunciado`, `alternativas` (dict A–E, 5
  alternativas), `resolucao_passo_a_passo` (string única com o raciocínio),
  `resposta_correta` (a **letra** da alternativa correta, A–E) e `difficulty`
  (`EASY`/`MEDIUM`/`HARD`).

**Wrapper de lote (`questoes`)**: o app pode pedir mais de uma questão por
chamada (ex.: "gere 10 questões"); o contrato responde sempre com uma lista,
mesmo quando `quantidade=1`. Para ensinar esse comportamento sem duplicar o
custo de geração por questão em produção (a maioria dos pedidos reais é de 1
questão), uma fração dos exemplos sintéticos agrupa 2–5 questões da mesma
habilidade/dificuldade num único exemplo de treino (`generate_synthetic.py`,
`TAMANHOS_LOTE`), enquanto os exemplos do banco real (uma questão por linha)
permanecem sempre com `quantidade=1`.

**Decisão de protocolo relevante (ordem de emissão, não de contrato)**: a
ordem das chaves no JSON de saída não é arbitrária. Como a geração é
autorregressiva (token a token, esquerda para direita), a posição de uma
chave determina se o modelo já "viu" (gerou) o raciocínio associado antes de
se comprometer com outra chave. Na primeira versão do pipeline, `gabarito`
precedia `justificativas`; isto foi identificado como uma causa provável do
modo de falha descrito na Seção 5.2, e corrigido — o raciocínio passou a
preceder o compromisso com a resposta. O contrato atual não especifica ordem
de chaves (JSON não garante ordem para quem consome por nome, e o app lê por
chave), mas a sequência de **emissão** treinada mantém
`resolucao_passo_a_passo` antes de `resposta_correta`, preservando esse
sequenciamento de raciocínio mesmo sem um campo dedicado ao valor numérico
(ver próximo parágrafo). Como o modelo roda em modo *non-thinking* (Seção
3.6, sem bloco `<think>`), essa sequência é o único mecanismo disponível para
induzir esse tipo de raciocínio antes da resposta.

**Remoção do campo `resposta` (efeito colateral aceito da migração de
contrato)**: uma versão anterior deste pipeline introduziu um campo `resposta`
dedicado ao *valor* da resposta correta, separado da *letra* (`gabarito`),
especificamente para tornar `check_consistency()` uma comparação exata
(`alternativas[gabarito] == resposta`) — ver análise de cobertura na Seção
5.4. Esse campo não existe no contrato atual (que não permite campos além
dos listados acima) e foi removido; a verificação de consistência voltou a
depender de extrair uma equação do texto de `resolucao_passo_a_passo` via
regex — o mesmo mecanismo, com a mesma cobertura limitada, que motivou a
criação do campo `resposta` em primeiro lugar (Seção 5.4). Esse é o principal
trade-off da migração: o contrato exigido pela integração mobile tem
prioridade sobre a otimização de verificabilidade, mas o efeito é mensurável
e está documentado para não ser confundido com regressão não intencional.

### 3.3 Divisão treino/validação

Split 90/10 estratificado por (ano escolar, dificuldade), com seed fixa (42):
**273 exemplos de treino / 30 de validação** (`data/train.jsonl`,
`data/val.jsonl`). O conjunto de validação contém **exclusivamente itens
reais** do banco SAEB em todas as fases do projeto — nenhum dado sintético
ou destilado é inserido em `data/val.jsonl` em nenhuma etapa, para que a
métrica de avaliação sempre reflita o cenário de produção.

### 3.4 Aumento de dados I: síntese aritmética determinística

Motivação: seguindo LIMA (Zhou et al., 2023), tratamos os ~300 itens reais
como suficientes para ensinar formato/estilo, mas não como suficientes para
generalizar capacidade aritmética — hipótese corroborada empiricamente pelos
modos de falha da Seção 6. `src/generate_synthetic.py` gera questões de
adição, subtração, multiplicação, divisão, porcentagem, potenciação e
frações (representação pictórica em texto, equivalência e conversão
fração→porcentagem — habilidades H07/H08/H09 do 9º ano) em que **a resposta
correta é calculada em Python antes de montar a questão** — nunca produzida por um
LLM, portanto nunca incorreta por construção. As tuplas de condicionamento
(ano, habilidade, descrição, dificuldade) usadas são as mesmas presentes no
banco real, para que a distribuição de prompts de treino não diverja da
distribuição enfrentada em produção.

Para as questões de fração, a "representação pictórica" (habilidade que no
item real depende de uma figura) é reconstruída inteiramente em texto — um
todo descrito verbalmente como dividido em N partes iguais, das quais M são
destacadas — preservando a restrição de que o modelo de produção é
puramente textual (Seção 3.1).

**Distratores pedagógicos**: cada alternativa incorreta não é um número
aleatório, mas o resultado de um erro específico e nomeável (operação
trocada, erro de reagrupamento/"vai-um", erro de posição da vírgula decimal,
confundir "valor do desconto" com "valor final", entre outros), e sua
justificativa correspondente explica esse erro citando o valor gerado. Este
desenho replica o padrão observado nos itens reais do SAEB (onde distratores
também correspondem a erros conceituais típicos) e — ponto identificado
empiricamente durante a iteração do projeto — corrige uma regressão da
primeira versão do gerador, que produzia justificativa apenas para a
alternativa correta; o modelo fine-tunado sobre essa primeira versão passou
a gerar as 4 justificativas de forma degenerada (repetidas/idênticas) em
testes reais, sugerindo que a ausência de justificativas para distratores no
treino ensinava o modelo a tratar esse campo como acessório.

No momento da escrita: **396 questões sintéticas** geradas (11 combinações de
habilidade × 3 dificuldades × 12 exemplos), agrupadas em **244 exemplos de
treino** (uma fração dos exemplos agrupa 2–5 questões da mesma habilidade/
dificuldade num único `{"questoes": [...]}`, ensinando o modelo a responder
pedidos de lote — ver Seção 3.2), somando **517 exemplos de treino total**
(273 reais + 244 sintéticos) em `data/train.jsonl`. Validação determinística
interna (`schema_utils.check_structure` + `check_consistency`, rodada sobre a
geração antes de qualquer treino): 0 falhas de schema/resposta_correta/
alternativas/difficulty em 396 questões; consistência resposta_correta↔
resolucao_passo_a_passo de 100% nos casos onde a checagem por regex é
aplicável (298/298 verificáveis, ~75% de cobertura sobre as 396 questões —
as de fração/porcentagem tipicamente caem fora do que a heurística consegue
verificar, precisamente por não seguirem o padrão "a op b = r"; ver Seção 3.9
para os limites dessa checagem, incluindo por que a cobertura não chega a
100% neste schema).

**Nota de robustez identificada durante a implementação**: a primeira versão
do gerador de conversão fração→porcentagem (H09) continha um laço de
preenchimento de distratores que, para casos próximos dos limites 0% ou
100%, saturava sempre no mesmo valor e nunca terminava (loop infinito
determinístico, não uma falha estatística rara) — por exemplo, para a fração
9/10 (90%), tanto o distrator "+10%" quanto o preenchimento subsequente
saturavam em "100%", colidindo entre si indefinidamente. Identificado por um
teste de estresse (geração de milhares de exemplos antes de qualquer
treino) e corrigido tornando o preenchimento bidirecional (tentando valores
acima e abaixo do gabarito) com um limite explícito de iterações que levanta
um erro claro em vez de travar silenciosamente. Este episódio é registrado
aqui como evidência do valor de testes de estresse determinísticos sobre
geradores de dados sintéticos antes de consumir tempo de treino sobre eles.

### 3.5 Aumento de dados II: destilação com filtro determinístico

Para habilidades fora do escopo de cálculo puro (leitura/interpretação,
ordenação, sequências) e para diversificar o contexto das questões
aritméticas, `src/distill_teacher.py` implementa uma destilação com
**professor maior** (modelo de chat acessado via Hugging Face Inference
Providers, configurável — padrão `Qwen/Qwen3-235B-A22B-Instruct-2507`).
Diferente da Seção 3.4, aqui a resposta correta não é computável a priori: a
garantia de qualidade vem de um **filtro determinístico pós-geração**, não da
geração em si. Um exemplo só é aceito se, simultaneamente: schema completo;
5 alternativas distintas; ausência de menção a figura/imagem; tamanho dentro
do limite de contexto de treino (`max_seq_length`); `resposta_correta`
**não reprovada** por `check_consistency()` (Seção 3.9); e enunciado não
duplicado (deduplicação por normalização de texto). Este desenho segue o
princípio geral de RLVR/verificação
determinística (Seção 2): a confiabilidade do dado não depende de o
professor "acertar sempre", e sim de o filtro rejeitar o que não presta.

Nota de status: este componente está implementado e validado
estruturalmente (execução em modo `--dry-run`, sem custo), mas **não foi
executado em produção** até o momento da escrita — geração real depende de
créditos de inferência do provedor e de uma decisão explícita sobre o modelo
professor e o volume de chamadas, portanto os números de aceitação/rejeição
deste componente ainda não existem e não devem ser citados como resultado.

### 3.6 Modelo base e método de fine-tuning

- **Modelo base**: `unsloth/Qwen3-1.7B` (Qwen Team, 2025), variante
  pré-quantizada em 4-bit NF4 (`unsloth/qwen3-1.7b-unsloth-bnb-4bit`).
- **Modo non-thinking**: treino e inferência com `enable_thinking=False`,
  eliminando o bloco de raciocínio livre `<think>` do Qwen3 — a principal
  fonte de latência do modelo base em CPU mobile.
- **Método**: QLoRA (Dettmers et al., 2023) via framework **Unsloth**
  (implementação otimizada de LoRA/QLoRA) e **TRL `SFTTrainer`**
  (Hugging Face). O modelo base permanece congelado em 4-bit; apenas
  adaptadores LoRA (Hu et al., 2021) são treinados.
- **Loss mascarada na resposta** (*train on completions only*, via
  `unsloth.chat_templates.train_on_responses_only`): o gradiente de loss é
  computado apenas sobre os tokens da mensagem `assistant`, não sobre o
  prompt (`system`+`user`).

### 3.7 Protocolo de treinamento (hiperparâmetros)

| Hiperparâmetro | Valor |
|---|---|
| Modelo base | `unsloth/Qwen3-1.7B` (4-bit NF4) |
| `max_seq_length` | 1024 |
| LoRA rank (r) / alpha | 16 / 32 |
| LoRA dropout | 0,0 |
| Módulos-alvo | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Batch por dispositivo / grad. accumulation | 2 × 8 (efetivo 16) |
| Learning rate | 2e-4, cosine, warmup 5% |
| Épocas | até 3, com early stopping (`patience=2`) sobre `eval_loss` |
| Otimizador | `adamw_8bit`, weight decay 0,01 |
| Precisão | bf16 |
| Loss | somente tokens de resposta do assistant |
| Seed | 42 |
| Hardware de treino | GPU única, 6GB VRAM (RTX 3060 Laptop, Ampere) |
| Frameworks | TRL 0.24.0 · Transformers 5.5.0 · PyTorch 2.11.0 |

### 3.8 Exportação para deploy mobile

O adaptador LoRA é mesclado ao modelo base em fp16 e exportado via
`export_gguf.py` (usando a integração GGUF do Unsloth, que compila
`llama.cpp` automaticamente) para o formato **GGUF, quantização Q4_K_M**
(~1,1GB). Este é o artefato final consumido pelo aplicativo mobile via
`llama.cpp`, sem dependência de `torch`/`unsloth`/Python em produção.

### 3.9 Verificação em tempo de inferência

Independente da qualidade aprendida durante o fine-tuning, o pipeline de
inferência de teste (`src/test_model.py`, função `generate_validated()`)
implementa uma camada de verificação determinística em quatro estágios:

1. **Decodificação restrita por gramática GBNF** (`grammars/questao.gbnf`):
   restringe a decodificação do `llama-cli` ao contrato exato exigido pelo
   app, garantindo por construção JSON válido, wrapper `{"questoes": [...]}`,
   presença de todas as chaves em cada questão, exatamente 5 alternativas
   (A–E), `resposta_correta` em `{A,B,C,D,E}` e `difficulty` em
   `{EASY,MEDIUM,HARD}`. Não garante corretude semântica.
2. **Checagem de consistência** (`schema_utils.check_consistency`): extrai
   uma equação "a op b = r" de `resolucao_passo_a_passo`, recalcula o lado
   esquerdo e compara com o valor da alternativa apontada por
   `resposta_correta`. É a **única** via disponível no contrato atual — o
   contrato não permite um campo dedicado ao valor da resposta (o antigo
   campo `resposta`, que existia justamente para tornar esta checagem uma
   comparação exata; ver nota de migração na Seção 3.2 e a análise completa
   na Seção 5.4). Cobertura empírica medida no conjunto sintético (só
   aritmética) após a migração: ~75%; era ~25% antes de qualquer correção e
   chegou a 100% durante a janela em que o campo `resposta` existiu. A
   normalização de operadores tipográficos Unicode (Seção 5.4) permanece em
   vigor e ainda contribui para essa cobertura.
3. **Best-of-N com seleção por verificador**: são amostrados até *N*
   candidatos, **parando assim que um passa** na verificação. Entre os que
   reprovam, o pipeline retém o **melhor** segundo uma ordem parcial
   (verificado correto > não verificável > corrigível > estrutura quebrada),
   e não o último gerado — é o que distingue best-of-N de repetição
   sequencial. A seleção por verificador programático, em vez de voto
   majoritário simples, é a variante que a literatura reporta como mais
   eficaz em modelos pequenos: +4,9 pontos em Qwen2-0.5B e +7,4 em
   Llama-3.2-1B no GSM8K (arXiv:2410.12608), sobre a linha de base de
   *self-consistency* (Wang et al., arXiv:2203.11171). Ressalva metodológica
   registrada na mesma literatura: a técnica só ajuda quando a probabilidade
   de acerto por amostra já é superior a 0,5 — condição satisfeita aqui
   (≈0,9 observado empiricamente).
4. **Correção determinística**: esgotadas as tentativas, `fix_gabarito()`
   troca `resposta_correta` para a alternativa que bate com a conta extraída
   de `resolucao_passo_a_passo`; só quando nem isso é possível a questão é
   marcada para descarte (`status="falha"`).

Esta camada não substitui a melhoria do modelo (Seções 3.4–3.5): ela é a
garantia de produção que continua valendo **independentemente** de quão bem
o retreino corrigiu o comportamento do modelo, e seu custo é limitado à
latência das regenerações nos casos que reprovam na primeira tentativa.

## 4. Protocolo de avaliação

Duas famílias de scripts avaliam o modelo, usando as mesmas definições de
métrica (`schema_utils.py`, compartilhado):

- **`evaluate.py`** (caminho de desenvolvimento): carrega o adaptador LoRA
  sobre o modelo base em 4-bit via `bitsandbytes`/Transformers, gera sobre
  as 30 questões de validação real (`temperature=0.7`, `top_p=0.8` —
  parâmetros recomendados pela Qwen para modo non-thinking) e mede métricas
  estruturais, de linguagem e de velocidade **na GPU local** — não
  representativo do artefato mobile, mas útil para iteração rápida.
- **`test_model.py`** (caminho de produção): chama o mesmo binário
  (`llama-cli`) e o mesmo arquivo `.gguf` que rodam no aplicativo, com
  `--batch` reproduzindo a avaliação sobre `data/val.jsonl` e o pipeline de
  verificação da Seção 3.9 opcionalmente ativo (`--raw` para medir o modelo
  "cru", sem grammar/verificação, como comparação controlada).

**Métricas estruturais** (definidas em `schema_utils.check_structure`, sobre
o wrapper `{"questoes": [...]}`): % JSON sintaticamente válido; % wrapper
válido (`questoes` é lista não vazia); % quantidade de questões bate com a
pedida; % schema completo (todas as chaves presentes em cada questão); %
`resposta_correta` ∈ {A,B,C,D,E}; % 5 alternativas distintas; % `difficulty`
∈ {EASY,MEDIUM,HARD}; distribuição das letras de `resposta_correta` geradas
(detecta viés); % de saídas que mencionam indevidamente
"figura/imagem/gráfico".

**Métrica de consistência semântica**: % de casos em que
`check_consistency()` (Seção 3.9) confirma que `resposta_correta` bate com a
conta extraída de `resolucao_passo_a_passo`, calculada apenas sobre o
subconjunto de amostras em que a equação é reconhecível pela heurística de
regex — reportada sempre junto com o denominador (n verificável), dado o
tamanho pequeno do conjunto de validação.

**Métrica de linguagem**: perplexity da resposta de referência (loss
calculada apenas sobre os tokens do assistant, análogo ao treino).

**Métricas de velocidade**: latência média/p95 por questão e tokens/segundo
de geração — medidas tanto no caminho de desenvolvimento (GPU) quanto no
artefato real (CPU, via `llama-bench`/`llama-cli`), sendo o segundo o número
relevante para a decisão de deploy.

## 5. Resultados preliminares

**Aviso metodológico**: o conjunto de validação real tem apenas 30 itens, e
o subconjunto onde a consistência é verificável pela heurística de regex é
ainda menor (n=3 a n=11 nas rodadas reportadas). Os números desta seção
devem ser lidos como **direcionais**, não como estimativas estatisticamente
robustas — um ponto a ser corrigido antes de qualquer submissão, expandindo o
conjunto de validação real ou reportando intervalos de confiança sobre um n
maior.

**Nota sobre a evolução das seções abaixo**: as Seções 5.1–5.4 documentam a
trajetória do projeto **antes** da migração de contrato (Seção 3.2) — schema
com `gabarito`/`justificativas`/`resposta`, dataset em versões sucessivas
(562, depois 669 exemplos). Mantidas como registro histórico do raciocínio
que levou ao desenho atual (em particular, a análise de cobertura da Seção
5.4 é a motivação direta do campo `resposta`, posteriormente removido pela
migração). A **Seção 5.5** reporta os resultados do modelo **retreinado sob
o contrato atual** (517 exemplos, schema com `resposta_correta`/
`resolucao_passo_a_passo`/`difficulty`) e é a referência para o estado
presente do projeto.

### 5.1 Modelo treinado sobre 562 exemplos (274 reais + 288 sintéticos, geração de distratores v1) — schema pré-migração, histórico

Medido via `evaluate.py` (GPU, caminho de desenvolvimento), 30 amostras de
validação:

| Métrica estrutural | Resultado |
|---|---|
| JSON válido | 96,7% |
| Schema completo | 96,7% |
| Gabarito válido (A–D) | 96,7% |
| 4 alternativas distintas | 90,0% |
| Menções indevidas a figura | 6,7% |
| Consistência gabarito↔justificativa | 37,5% (3/8 verificáveis) |
| Perplexity (resposta de referência) | 2,223 |

Medido via `test_model.py --batch` (artefato `.gguf` real, CPU, 10 amostras):

| Métrica estrutural (GGUF real) | Resultado |
|---|---|
| JSON válido / schema completo / gabarito válido / alternativas distintas | 100% |
| Menções indevidas a figura | 0% |
| Consistência gabarito↔justificativa | 66,7% (2/3 verificáveis) |

**Velocidade** (`llama-bench`, CPU, máquina ociosa, referência de deploy):

| Configuração | Prompt processing | Geração |
|---|---|---|
| 4 threads | 123 tok/s | 29,7 tok/s |
| 16 threads | 116 tok/s | 17,1 tok/s (banda de memória satura antes dos núcleos) |

### 5.2 Modos de falha identificados em testes reais (mesmo modelo da Seção 5.1)

Três padrões de erro distintos foram observados em geração real (via app e
via `test_model.py`), documentados aqui por serem a motivação direta das
correções da Seção 3.4 e 3.9:

1. **Justificativas degeneradas**: as 4 justificativas idênticas entre si
   (ex.: todas repetindo a mesma equação), atribuído à ausência de
   justificativa para distratores na primeira versão do gerador sintético
   (Seção 3.4) — corrigido na versão atual do gerador, ainda **não
   validado após retreino** no momento da escrita.
2. **Erro de vínculo número→letra**: o modelo calcula corretamente (ex.:
   120 + 35 = 155) mas associa o resultado à letra errada no campo
   `gabarito`, enquanto a justificativa textual do valor correto aparece
   atribuída a uma alternativa diferente. Mitigado, não eliminado, pela
   reordenação de schema (Seção 3.2) e coberto em produção pelo verificador
   da Seção 3.9.
3. **Incoerência semântica de enunciado**: em cenários fora da distribuição
   de treino (ex.: quantidade consumida maior que o total disponível em um
   contexto de porcentagem), o modelo produziu um enunciado sem sentido
   físico e uma inconsistência adicional entre a justificativa (300%) e o
   gabarito (200%). Este modo de falha é atribuído a limite de conhecimento/
   raciocínio do modelo de 1,7B em contextos incomuns, e não é atacado por
   nenhuma correção de schema ou dado sintético puro — é o principal
   argumento para a destilação de professor (Seção 3.5) e para RLVR/GRPO
   (Seção 8) como próximos passos.

### 5.3 Validação end-to-end do pipeline de verificação (Seção 3.9)

Uma geração real sobre o artefato `.gguf` da Seção 5.1 (portanto, ainda sob
o modo de falha nº 1 acima) com grammar GBNF + verificador ativos produziu:
estrutura 100% válida, 1 regeneração automática disparada, e gabarito final
consistente com a conta (240 + 278 = 518, gabarito B = 518) — confirmando
que o mecanismo de verificação entrega a garantia pretendida (gabarito
correto) mesmo quando o modelo subjacente ainda apresenta o modo de falha
nº 1 (justificativas repetidas, não corrigido pelo verificador, que atua
apenas sobre a letra do gabarito).

### 5.4 Análise de cobertura do verificador: a rede de segurança tinha um buraco

Testes de uso continuado reportaram uma taxa de acerto de aproximadamente 90%,
com casos residuais de gabarito incorreto **atravessando o verificador sem
serem sinalizados**. A investigação desses casos revelou que o problema não
estava apenas no modelo, mas na própria camada de verificação.

**Medição de cobertura.** Recontando os relatórios de avaliação já existentes
pelo estado retornado por `check_consistency()`:

| Relatório | n | verificado correto | verificado incorreto | **não verificável** |
|---|---|---|---|---|
| `eval_report.json` (GPU) | 30 | 4 | 3 | **23 (77%)** |
| `eval_report_gguf.json` (CPU) | 10 | 2 | 1 | **7 (70%)** |

Ou seja: em 70–77% das gerações o verificador **não emitia julgamento algum**,
e uma questão não julgada é indistinguível, do ponto de vista do sistema, de
uma questão aprovada. A garantia de produção declarada na Seção 3.9 valia,
na prática, para cerca de um quarto dos casos.

**Causa 1 — operadores tipográficos Unicode.** Um caso reportado trazia a
justificativa `"12 − 7 = 5."` associada a uma alternativa de valor `4`,
enquanto o valor `5` estava em outra alternativa: gabarito objetivamente
errado, e ainda assim classificado como "não verificável". A inspeção do
código de ponto revelou que o caractere emitido era U+2212 (MINUS SIGN),
enquanto a expressão regular aceitava apenas U+002D (HYPHEN-MINUS). O modelo
emite a forma tipográfica de maneira **intermitente** (uma amostragem
independente de cinco gerações não a produziu nenhuma vez), o que explica por
que a falha era esporádica e difícil de reproduzir.

A origem dessa intermitência foi localizada nos **próprios dados de treino**:
auditando o corpus real extraído do banco, **30% dos itens (84 de 273)**
contêm operadores tipográficos (143 ocorrências de `×`, 50 de `–`, 41 de `−`),
enquanto os 70% restantes — e a totalidade do corpus sintético — usam ASCII.
O modelo não estava introduzindo um artefato próprio: estava reproduzindo
fielmente uma inconsistência de notação presente na sua supervisão. Para um
modelo de 1,7B ajustado sobre poucas centenas de exemplos, uma convenção
dividida em 30/70 é aprendida como variação livre, e emitida de forma
imprevisível na inferência.

Corrigido em dois pontos: (i) `normalize_math()`, aplicada antes de qualquer
casamento textual na verificação, mapeando `− – — × ⋅ ∙ ∕` e espaços não
separáveis para os equivalentes ASCII; e (ii) a mesma normalização aplicada
na **extração** (`extract_data.py`), padronizando o corpus real para ASCII e
alinhando-o ao sintético. A segunda correção ataca a causa, a primeira
protege contra o efeito residual.

A mesma auditoria identificou um item do banco com duas alternativas de texto
idêntico (a resposta correta duplicada em A e D), que tornava o gabarito
ambíguo; passou a ser descartado na extração pelo filtro
`alternativas_duplicadas`. Corpus final: 273 itens reais (de 304) + 396
sintéticos = 669 exemplos de treino, 0 com operador Unicode, 0 com schema
inválido, 100% verificáveis.

**Causa 2 — raciocínio verbal sem equação explícita (dominante).** A mesma
amostragem de cinco gerações produziu justificativas como
`"Subtraindo 5 de 12, obtemos 7."` — aritmeticamente correta, mas sem nenhuma
equação no formato `a op b = r`. Nenhuma normalização de caracteres resolve
este caso: extrair a operação exigiria interpretar linguagem natural, uma
corrida armamentista contra a variedade de formulações possíveis. Esta é a
causa majoritária dos 70–77%.

**Consequência metodológica.** A Causa 2 estabelece que aumentar a
sofisticação do extrator é uma estratégia sem teto de garantia: qualquer
heurística sobre texto livre terá cobertura parcial e desconhecida *a priori*.
A alternativa adotada foi mudar o **contrato de saída** em vez do extrator —
o campo `resposta` (Seção 3.2) coloca o valor da resposta em um slot
dedicado, tornando a verificação uma comparação exata. A cobertura sobre o
conjunto sintético, onde o campo é gerado por construção, passa de 76% (fração
com equação explícita reconhecível) para **100%** (396/396 exemplos
verificáveis). A cobertura sobre gerações reais do modelo só poderá ser medida
integralmente após o retreino, e é a métrica a reportar na próxima iteração
deste documento.

**Medição intermediária** (artefato ainda **não** retreinado, 8 amostras, com
gramática e verificador novos ativos): a cobertura sobe de ~25–30% para
**50%** (4 de 8 verificáveis) apenas com a normalização Unicode, com 100% de
consistência entre os casos julgados, nenhuma questão descartada e nenhuma
correção de letra necessária. O ganho residual até 100% depende do modelo
aprender a preencher `resposta` com o valor — isto é, do retreino. Este é o
comportamento esperado e não deve ser lido como resultado final.

**Nota sobre hiperparâmetros.** Foi considerada e descartada a hipótese de
que o modo de falha fosse mitigável por ajuste de hiperparâmetros de treino.
A configuração atual (r=16, α=32, lr 2e-4) já corresponde ao ótimo relatado
para datasets pequenos — desempenho reportado com pico em r=16, sem ganho em
r=32 e subajuste em r=8 (documentação técnica do Unsloth) — de modo que o
espaço de melhoria está no contrato de saída e na verificação, não na
parametrização do ajuste fino.

### 5.5 Modelo retreinado sob o contrato de schema atual (517 exemplos, pós-migração)

Após a migração de contrato (Seção 3.2) — que introduziu o wrapper
`{"questoes": [...]}`, 5 alternativas (A–E), `resposta_correta`,
`resolucao_passo_a_passo` e `difficulty`, e **removeu** o campo `resposta`
que sustentava a verificação exata da Seção 5.4 — o modelo foi retreinado
sobre **517 exemplos** (273 reais + 244 sintéticos agrupando 396 questões,
incluindo as frações H07–H09). Estes são os primeiros resultados medidos
sob o contrato atual, e substituem as Seções 5.1–5.4 como referência do
estado presente do projeto.

Medido via `evaluate.py` (GPU, caminho de desenvolvimento), 30 amostras de
validação real:

| Métrica estrutural | Resultado |
|---|---|
| JSON válido | 100% |
| Wrapper `{"questoes": [...]}` válido | 100% |
| Schema completo | 100% |
| `resposta_correta` válida (A–E) | 100% |
| 5 alternativas distintas | 90,0% |
| `difficulty` válida (EASY/MEDIUM/HARD) | 100% |
| Menções indevidas a figura | 13,3% |
| Consistência `resposta_correta` ↔ `resolucao_passo_a_passo` | 72,7% (11/30 verificáveis) |
| Perplexity (resposta de referência) | 1,772 |

| Velocidade (GPU, dev) | Resultado |
|---|---|
| Latência média | 5,09 s |
| Latência p95 | 6,82 s |
| Tokens/s de geração | 30,3 |
| Tokens de saída (média) | 152,7 |

**Leitura destes números frente à Seção 5.4.** A hipótese registrada na
Seção 5.4 e no antigo item 1 da Seção 8 era que a cobertura de verificação
chegaria a ~100% após o retreino, sob a premissa de que o campo `resposta`
seguiria no schema. Essa premissa deixou de valer: o contrato exigido pela
integração mobile não permite esse campo (Seção 3.2), então a cobertura
medida (72,7%, n=11/30) reflete o teto da via por regex sobre
`resolucao_passo_a_passo` — mais alta que os ~25% da primeira medição
(Causa 1 da Seção 5.4 corrigida por `normalize_math`), mas sem o salto a
100% que o campo dedicado teria dado. Este é o resultado esperado do
trade-off descrito na Seção 3.2, não uma regressão de treino: o contrato
correto tem prioridade sobre a métrica de verificabilidade.

As outras métricas estruturais saíram em 100% (à exceção de 5 alternativas
distintas, 90%) — acima dos 96,7% medidos na Seção 5.1 sob o schema
anterior, indicando que a grammar GBNF (Seção 3.9) e o dataset ampliado
continuam entregando estrutura confiável mesmo com o contrato mais rígido
(5 alternativas em vez de 4, wrapper de lista, enum `difficulty`).

**Pendências de medição**: a validação do artefato `.gguf` real
(`test_model.py --batch`) ainda não foi remedida sob o contrato atual — o
`eval_report_gguf.json` existente é de uma rodada anterior à migração
(schema `gabarito`/`justificativas`) e não deve ser citado como resultado
do modelo atual. O teste manual de uma única questão (via
`test_model.py --ano "5º" --habilidade H08 ... --dificuldade Fácil`)
confirmou geração correta no formato exato do contrato, mas não substitui
uma rodada `--batch` com métricas agregadas. O suporte a lote (`--quantidade
N` no CLI, `"questoes"` com múltiplos itens) também ainda não tem uma
rodada de avaliação agregada — só verificação pontual.

## 6. Limitações

- **Amostra de validação pequena** (n=30, e n≤11 para a métrica de
  consistência) — ver aviso metodológico da Seção 5.
- **A verificação por expressão regular permanece sintaticamente limitada**
  (Seção 5.4, Causa 2) e, após a migração de contrato (Seção 3.2), é a
  **única** via disponível — o contrato do app não admite o campo `resposta`
  que, numa versão anterior deste pipeline, havia elevado a cobertura a
  100% no conjunto sintético por comparação exata. Cobertura atual medida:
  ~75% no conjunto sintético (só aritmética, Seção 3.4), 72,7% (n=11/30) no
  conjunto de validação real após o retreino (Seção 5.5) — presumivelmente
  menor nas habilidades mais interpretativas do banco real fora dessa
  amostra. Além da cobertura parcial, a checagem só confere *coerência
  interna* entre a conta extraída e a letra escolhida, não a corretude da
  conta em si: um modelo que calcule errado e seja consistente com o próprio
  erro passa pela verificação. Não cobre raciocínio verbal sem equação
  explícita nem respostas textuais (frações, porcentagens) — mitigado apenas
  pelo best-of-N e pela correção determinística (Seção 3.9), não eliminado.
  Mitigar a limitação de corretude (não só de cobertura) exigiria um
  solucionador independente (viável apenas para os subtipos de questão
  gerados deterministicamente) ou RLVR (Seção 8).
- **Validação do artefato `.gguf` real ainda não remedida sob o contrato
  atual** (Seção 5.5): o retreino sobre os 517 exemplos já foi feito e
  avaliado no caminho de desenvolvimento (GPU, `evaluate.py`), mas
  `test_model.py --batch` — que mede o mesmo binário/arquivo que roda no
  app — ainda não foi rodado após a migração de schema; o relatório
  existente (`eval_report_gguf.json`) é de uma rodada anterior, com o schema
  antigo, e não deve ser citado como resultado do modelo atual.
- **Escopo de cálculo determinístico ainda parcial**: `generate_synthetic.py`
  cobre adição, subtração, multiplicação, divisão exata, porcentagem,
  potenciação e frações (representação pictórica textual, equivalência,
  conversão para porcentagem — H07/H08/H09), já incorporadas ao dataset de
  treino atual (Seção 3.4/5.5). Fora do escopo atual: conversão
  fração↔decimal explícita (H09 cobre só fração→porcentagem) e operações
  aritméticas diretamente entre frações (soma/subtração com denominadores
  diferentes) — candidatas naturais para uma próxima extensão.
- **Grammar GBNF restringe a saída inteira**, não apenas a resposta final
  como no desenho do CRANE (Banerjee et al., 2025) — a literatura sugere que
  restringir tudo pode custar capacidade de raciocínio; não medimos
  separadamente esse efeito neste trabalho (comparação `--raw` vs. com
  grammar ainda não quantificada sobre o modelo retreinado).
- **Destilação (Seção 3.5) não executada em produção** até o momento da
  escrita — implementada e validada apenas em modo dry-run.
- **Suporte a lote (`quantidade` > 1, Seção 3.2) sem avaliação agregada**:
  verificado pontualmente (uma chamada manual com `--quantidade 5` retornou
  5 questões no formato correto), mas não há, até o momento da escrita,
  uma rodada de `test_model.py`/`evaluate.py` que meça estrutura e
  consistência especificamente sobre pedidos de lote (N > 1).

## 7. Reprodutibilidade

Todos os scripts citados estão em `src/`, com README.md do repositório
documentando os comandos de cada etapa. Sequência completa:

```
extract_data.py → generate_synthetic.py [→ distill_teacher.py --merge]
  → train.py → evaluate.py → export_gguf.py → test_model.py
```

`data/val.jsonl` nunca é modificado por nenhum script de aumento de dados —
garantia estrutural, não apenas convenção, verificável lendo
`generate_synthetic.py`/`distill_teacher.py` (ambos operam exclusivamente
sobre `data/train.jsonl`).

## 8. Trabalhos futuros

1. **Remedir o artefato `.gguf` real** (`test_model.py --batch`) sob o
   contrato de schema atual — o retreino da Seção 5.5 só foi avaliado no
   caminho de desenvolvimento (GPU); falta confirmar que as mesmas métricas
   estruturais e de consistência se sustentam no binário/arquivo que roda no
   app (CPU, `llama-cli`, Q4_K_M).
2. **Avaliar especificamente pedidos de lote** (`quantidade` > 1, Seção 3.2):
   hoje só há verificação pontual manual; falta uma rodada agregada que meça
   estrutura e consistência sobre N > 1 questões por chamada.
3. **Executar a destilação em escala** (Seção 3.5) e reportar taxas de
   aceitação/rejeição por motivo de filtro — dado ainda inexistente.
4. **RLVR/GRPO** (Shao et al., 2024; DeepSeek-AI, 2025): usar
   `check_consistency()` (Seção 3.9) diretamente como função de recompensa
   verificável, seguindo a evidência de que recompensa por processo supera
   recompensa só por resultado em modelos pequenos (arXiv:2607.02869).
   Viabilidade de hardware já confirmada (Unsloth reporta GRPO em
   Qwen3-1.7B com FP8 em ~5GB de VRAM, dentro da RTX 3060 6GB disponível).
   Nota: com o campo `resposta` removido do contrato (Seção 3.2), a função
   de recompensa fica limitada à mesma via por regex da Seção 3.9/5.5 — o
   ganho esperado de GRPO aqui é sobre a corretude da conta, não sobre a
   cobertura da verificação, que tem teto conhecido (~75%).
5. **Estender `generate_synthetic.py`** para conversão fração↔decimal
   explícita e aritmética direta entre frações (soma/subtração com
   denominadores diferentes) — únicos subtipos de cálculo determinístico
   ainda fora de escopo (H07–H09 já cobertas desde a Seção 3.4/5.5).
6. **Quantificar separadamente o custo de raciocínio da grammar GBNF total**
   frente a uma variante estilo CRANE (grammar só na resposta final,
   raciocínio livre antes) — pergunta em aberto na Seção 6.
7. **Ampliar o conjunto de validação real** para reduzir a incerteza
   estatística da Seção 5 antes de reportar números em publicação.

## Referências bibliográficas

- Banerjee, D., Suresh, T., Ugare, S., Misailovic, S., & Singh, G. (2025).
  *CRANE: Reasoning with constrained LLM generation*. ICML 2025.
  arXiv:2502.09061.
- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in
  LLMs via Reinforcement Learning*. arXiv:2501.12948.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA:
  Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang,
  L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language
  Models*. arXiv:2106.09685.
- Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira
  (INEP). *Sistema de Avaliação da Educação Básica (SAEB) — Documentos de
  Referência*. Ministério da Educação, Brasil.
  (https://download.inep.gov.br/publicacoes/institucionais/avaliacoes_e_exames_da_educacao_basica/saeb_documentos_referencia_versao_preliminar.pdf)
- Qwen Team (2025). *Qwen3 Technical Report*. arXiv:2505.09388.
- Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang,
  M., Li, Y. K., Wu, Y., & Guo, D. (2024). *DeepSeekMath: Pushing the
  Limits of Mathematical Reasoning in Open Language Models*.
  arXiv:2402.03300.
- Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A.,
  Yu, P., Yu, L., Zhang, S., Ghosh, G., Lewis, M., Zettlemoyer, L., & Levy,
  O. (2023). *LIMA: Less Is More for Alignment*. arXiv:2305.11206.
- Artigo sem autoria individual identificada nos resultados de busca (2025).
  *Reward Granularity in RLVR: Comparing Process and Outcome Reward
  Structures for Mathematical Reasoning in Small Language Models*.
  arXiv:2607.02869. **(verificar autoria completa antes de citar em
  publicação — não confirmada nesta revisão.)**
- Artigo sem autoria individual identificada nos resultados de busca (2025).
  *Token Constraint Decoding Improves Robustness on Question Answering for
  Large Language Models*. arXiv:2506.09408. **(verificar autoria completa
  antes de citar em publicação — não confirmada nesta revisão.)**
- Unsloth AI. *Reinforcement Learning (RL) Guide* e *Qwen3 — How to Run &
  Fine-tune* (documentação técnica). https://unsloth.ai/docs
- von Werra, L. et al. *TRL: Transformer Reinforcement Learning* (biblioteca
  de software, Hugging Face). https://github.com/huggingface/trl
- Gerganov, G. et al. *llama.cpp* (projeto de software, especificação de
  gramática GBNF). https://github.com/ggml-org/llama.cpp

---

**Nota de manutenção deste documento**: o modelo **já foi retreinado** sobre
o contrato de schema atual (517 exemplos: 273 reais + 244 sintéticos/396
questões, Seção 3.4) — resultados em **Seção 5.5**, que substitui as Seções
5.1–5.4 como referência do estado presente (mantidas como registro
histórico do raciocínio que levou ao desenho atual, em particular à
introdução e posterior remoção do campo `resposta`). Pendências que ainda
exigem atualização deste documento: (a) remedir `test_model.py --batch`
sobre o `.gguf` exportado após o retreino (Seção 5.5/6/8, item 1) — o
`eval_report_gguf.json` existente é de uma rodada anterior à migração de
contrato e não deve ser citado como resultado atual; (b) avaliar
especificamente pedidos de lote (`quantidade` > 1); (c) executar a
destilação (Seção 3.5) em escala. Os itens marcados "verificar autoria
completa" na lista de referências foram localizados via busca automatizada
e precisam de conferência manual do preprint antes de qualquer submissão
formal.
