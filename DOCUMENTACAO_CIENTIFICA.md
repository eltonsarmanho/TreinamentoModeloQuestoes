# Fine-tuning de um LLM de 1.7B para Geração Offline de Questões de Matemática no Padrão SAEB: Metodologia, Protocolo e Resultados Preliminares

**Documento de trabalho interno — base para publicação futura.**
Última atualização: 2026-08-01.
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

Resultado: **304 itens válidos**, cobrindo os 3 anos escolares, as 3
dificuldades e 27 habilidades distintas.

### 3.2 Formato do exemplo de treino

Cada item vira um exemplo de conversa (formato *chat*) com três mensagens:

- `system`: instrução fixa (`SYSTEM_PROMPT`) definindo o papel do modelo e o
  schema JSON de saída esperado.
- `user`: template fixo —
  `"Gere uma questão de matemática. Ano: {ano} ano. Habilidade: {habilidade}
  — {descricao}. Dificuldade: {dificuldade}."` — o mesmo condicionamento
  (ano, habilidade, descrição, dificuldade) usado pelo aplicativo em
  produção.
- `assistant`: JSON compacto com as chaves, **nesta ordem**: `enunciado`,
  `comando`, `alternativas` (dict A–D), `justificativas` (dict A–D),
  `gabarito` (letra).

**Decisão de protocolo relevante**: a ordem das chaves no JSON de saída não é
arbitrária. Como a geração é autorregressiva (token a token, esquerda para
direita), a posição de uma chave determina se o modelo já "viu" (gerou) o
raciocínio associado antes de se comprometer com outra chave. Na primeira
versão do pipeline, `gabarito` precedia `justificativas`; isto foi
identificado como uma causa provável do modo de falha descrito na Seção 6, e
corrigido — `justificativas` passou a preceder `gabarito`, forçando o modelo
a "mostrar o trabalho" antes de emitir a letra final. Como o modelo roda em
modo *non-thinking* (Seção 3.6, sem bloco `<think>`), a ordem das chaves do
JSON é o único mecanismo disponível para induzir esse tipo de sequenciamento
de raciocínio.

### 3.3 Divisão treino/validação

Split 90/10 estratificado por (ano escolar, dificuldade), com seed fixa (42):
**274 exemplos de treino / 30 de validação** (`data/train.jsonl`,
`data/val.jsonl`). O conjunto de validação contém **exclusivamente itens
reais** do banco SAEB em todas as fases do projeto — nenhum dado sintético
ou destilado é inserido em `data/val.jsonl` em nenhuma etapa, para que a
métrica de avaliação sempre reflita o cenário de produção.

### 3.4 Aumento de dados I: síntese aritmética determinística

Motivação: seguindo LIMA (Zhou et al., 2023), tratamos os 304 itens reais
como suficientes para ensinar formato/estilo, mas não como suficientes para
generalizar capacidade aritmética — hipótese corroborada empiricamente pelos
modos de falha da Seção 6. `src/generate_synthetic.py` gera questões de
adição, subtração, multiplicação, divisão, porcentagem, potenciação e
frações (representação pictórica em texto, equivalência e conversão
fração→porcentagem — habilidades H07/H08/H09 do 9º ano) em que **o gabarito
é calculado em Python antes de montar a questão** — nunca produzido por um
LLM, portanto nunca incorreto por construção. As tuplas de condicionamento
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

No momento da escrita: **396 exemplos sintéticos** gerados (11 combinações de
habilidade × 3 dificuldades × 12 exemplos), somando **670 exemplos de treino
total** (274 reais + 396 sintéticos) em `data/train.jsonl`. Validação
determinística interna (`schema_utils.check_structure` +
`check_consistency`, rodada sobre a geração antes de qualquer treino): 0
falhas de schema/gabarito/alternativas/justificativas em 396 exemplos; 100%
de consistência gabarito↔justificativa nos casos onde a checagem por regex é
aplicável (297/297 na rodada correspondente a este tamanho de lote — as
questões de fração/porcentagem tipicamente caem fora do que a heurística
consegue verificar, precisamente por não seguirem o padrão "a op b = r"; ver
Seção 3.9 para os limites dessa checagem).

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
Diferente da Seção 3.4, aqui o gabarito não é computável a priori: a garantia
de qualidade vem de um **filtro determinístico pós-geração**, não da
geração em si. Um exemplo só é aceito se, simultaneamente: schema completo;
4 alternativas e 4 justificativas distintas; ausência de menção a
figura/imagem; tamanho dentro do limite de contexto de treino
(`max_seq_length`); gabarito **não reprovado** por `check_consistency()`
(Seção 3.9); e enunciado não duplicado (deduplicação por normalização de
texto). Este desenho segue o princípio geral de RLVR/verificação
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
implementa uma camada de verificação determinística em três estágios:

1. **Decodificação restrita por gramática GBNF** (`grammars/questao.gbnf`):
   restringe a decodificação do `llama-cli` ao schema JSON exato (mesma
   ordem de chaves da Seção 3.2), garantindo por construção JSON válido,
   presença de todas as chaves, exatamente 4 alternativas e 4 justificativas,
   e gabarito em `{A,B,C,D}`. Não garante corretude semântica.
2. **Checagem de consistência gabarito↔justificativa**
   (`schema_utils.check_consistency`): extrai, via expressão regular, uma
   equação do tipo "a op b = r" do texto da justificativa correspondente ao
   gabarito, recalcula o lado esquerdo e compara com o valor numérico
   apresentado na alternativa apontada como correta. Retorna um entre três
   estados — consistente, inconsistente (com sugestão de letra correta, se
   encontrada), ou não verificável (quando o texto não contém uma equação
   simples reconhecível — limitação conhecida: casos com raciocínio verbal
   sem equação explícita, ou com múltiplas operações encadeadas, não são
   cobertos por esta heurística).
3. **Regenerar → corrigir**: se a checagem reprova, o pipeline regenera com
   uma seed diferente (número de tentativas configurável). Esgotadas as
   tentativas, `fix_gabarito()` aplica correção determinística — se a conta
   da justificativa aponta para o valor de outra alternativa, a letra do
   gabarito é trocada para essa alternativa; só quando isso também falha a
   questão é marcada para descarte (`status="falha"`).

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

**Métricas estruturais** (definidas em `schema_utils.check_structure`): %
JSON sintaticamente válido; % schema completo (todas as chaves presentes); %
gabarito ∈ {A,B,C,D}; % 4 alternativas distintas; % 4 justificativas
distintas (métrica introduzida após o modo de falha da Seção 6); distribuição
das letras de gabarito geradas (detecta viés); % de saídas que mencionam
indevidamente "figura/imagem/gráfico".

**Métrica de consistência semântica**: % de casos em que
`check_consistency()` (Seção 3.9) confirma que o gabarito bate com a conta
da justificativa, calculada apenas sobre o subconjunto de amostras em que a
equação é reconhecível pela heurística de regex — reportada sempre junto com
o denominador (n verificável), dado o tamanho pequeno do conjunto de
validação.

**Métrica de linguagem**: perplexity da resposta de referência (loss
calculada apenas sobre os tokens do assistant, análogo ao treino).

**Métricas de velocidade**: latência média/p95 por questão e tokens/segundo
de geração — medidas tanto no caminho de desenvolvimento (GPU) quanto no
artefato real (CPU, via `llama-bench`/`llama-cli`), sendo o segundo o número
relevante para a decisão de deploy.

## 5. Resultados preliminares

**Aviso metodológico**: o conjunto de validação real tem apenas 30 itens, e
o subconjunto onde a consistência gabarito↔justificativa é verificável pela
heurística de regex é ainda menor (n=3 a n=8 nas rodadas reportadas). Os
números desta seção devem ser lidos como **direcionais**, não como
estimativas estatisticamente robustas — um ponto a ser corrigido antes de
qualquer submissão, expandindo o conjunto de validação real ou reportando
intervalos de confiança sobre um n maior.

**Nota de defasagem**: os resultados abaixo foram medidos sobre um modelo
treinado com **562** exemplos (274 reais + 288 sintéticos, cobrindo apenas
aritmética inteira/decimal). Desde então, `generate_synthetic.py` passou a
cobrir também frações (H07–H09, Seção 3.4), elevando o dataset de treino
para **670** exemplos (274 + 396). Os números desta seção **antecedem** essa
expansão e precisam ser remedidos após o próximo retreino.

### 5.1 Modelo treinado sobre 562 exemplos (274 reais + 288 sintéticos, geração de distratores v1)

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

## 6. Limitações

- **Amostra de validação pequena** (n=30, e n≤8 para a métrica de
  consistência) — ver aviso metodológico da Seção 5.
- **Heurística de consistência é sintaticamente limitada**: cobre apenas
  equações simples de uma operação ("a op b = r"); raciocínio verbal sem
  equação explícita, ou problemas de porcentagem/potenciação com múltiplas
  operações encadeadas, não são verificados por ela — o gabarito nesses
  casos pode estar correto ou incorreto sem que o sistema saiba dizer.
- **Resultados da Seção 5 antecedem o retreino com os dados sintéticos
  corrigidos (Seção 3.4) e a destilação (Seção 3.5)** — no momento da
  escrita, a correção mais recente foi validada apenas estruturalmente
  (geração determinística, sem custo de treino), não através de um modelo
  retreinado. Este documento deve ser atualizado com os resultados
  pós-retreino antes de qualquer submissão.
- **Escopo de cálculo determinístico ainda parcial**: `generate_synthetic.py`
  cobre adição, subtração, multiplicação, divisão exata, porcentagem,
  potenciação e frações (representação pictórica textual, equivalência,
  conversão para porcentagem — H07/H08/H09). Fora do escopo atual: conversão
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

1. **Retreinar e reavaliar** com o gerador sintético corrigido (Seção 3.4) e
   quantificar o impacto isolado sobre a métrica de consistência e sobre
   `justificativas_distintas_pct` (comparação direta com a Seção 5.1).
2. **Executar a destilação em escala** (Seção 3.5) e reportar taxas de
   aceitação/rejeição por motivo de filtro — dado ainda inexistente.
3. **RLVR/GRPO** (Shao et al., 2024; DeepSeek-AI, 2025): usar
   `check_consistency()` (Seção 3.9) diretamente como função de recompensa
   verificável, seguindo a evidência de que recompensa por processo supera
   recompensa só por resultado em modelos pequenos (arXiv:2607.02869).
   Viabilidade de hardware já confirmada (Unsloth reporta GRPO em
   Qwen3-1.7B com FP8 em ~5GB de VRAM, dentro da RTX 3060 6GB disponível).
4. **Retreinar e reavaliar com as questões de fração** (H07–H09, adicionadas
   em `generate_synthetic.py` após a Seção 5 ter sido escrita — ver "Nota de
   defasagem" na abertura da Seção 5) e, na sequência, estender o gerador
   para conversão fração↔decimal e aritmética direta entre frações.
5. **Quantificar separadamente o custo de raciocínio da grammar GBNF total**
   frente a uma variante estilo CRANE (grammar só na resposta final,
   raciocínio livre antes) — pergunta em aberto na Seção 6.
6. **Ampliar o conjunto de validação real** para reduzir a incerteza
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

**Nota de manutenção deste documento**: as Seções 5 e 6 devem ser
atualizadas assim que (a) o modelo for retreinado sobre os 670 exemplos
atuais (274 reais + 396 sintéticos, já incluindo frações H07–H09), e (b) a
destilação (Seção 3.5) for executada em escala. Os itens marcados "verificar
autoria completa" na lista de referências foram localizados via busca
automatizada e precisam de conferência manual do preprint antes de qualquer
submissão formal.
