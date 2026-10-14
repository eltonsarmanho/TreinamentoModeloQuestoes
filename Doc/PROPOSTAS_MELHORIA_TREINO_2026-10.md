# Auditoria de arquitetura e treino — propostas de melhoria

Data: 2026-10-14. Escopo: pipeline de treino (DB → dataset → QLoRA → GGUF → gate), hiperparâmetros,
base de dados e critério de aceitação. Não cobre o pipeline de agentes da base de conhecimento
(`agentes_questoes.py` e afins) além do que afeta o dado de treino.

Método: auditoria somente leitura do código e dos artefatos (`outputs/checkpoints_*`, `VEREDITO_*`,
`apuracao.json`), medições sobre `data/train_curado_v*.jsonl` com o tokenizer do Qwen3, e revisão de
evidência externa (2024–2026) listada na seção 7. Cada proposta diz o que muda, por que, qual evidência
sustenta, como medir e o custo estimado.

---

## 1. Sumário executivo

O problema central **não é hiperparâmetro**. A configuração QLoRA atual (r=16, α=32, LR 2e-4, batch
efetivo 16, LoRA em todas as projeções lineares, bf16, 3 épocas com melhor época por `eval_loss`) já
coincide com a receita de referência mais citada hoje (Schulman et al., *LoRA Without Regret*, 2025).
Trocar `max_seq_length` (v7c) não produziu efeito distinguível de ruído.

O que a evidência do próprio projeto mostra:

| Achado | Evidência |
|---|---|
| Dois treinos idênticos (v7b e v7d: mesma base, seed, config) deram vereditos opostos no gate | `VEREDITO_v7b` PROMOVIDO vs `VEREDITO_v7d` NÃO PROMOVIDO; eval_loss 0,6720 vs 0,6724 |
| Três ciclos de base (v6, v7, v7b) não superaram o baseline de 963 exemplos na amostra humana | v6: 35% vs 50%; v7b: 35% vs 45% (n=20/braço, IC ±20 pp) |
| O verificador determinístico cobre ~50% dos itens; o professor rejeita itens que o gate marca `ok` | G3* decide sobre n=68–84 de 150; `correcao_matematica` 60–70% na amostra humana |
| O gate mede um regime de prompt diferente da produção | gate roda `sufixo=false`; app e 90% do treino usam sufixo de diversidade |
| `val.jsonl` e `val_frozen_v1.jsonl` não são idênticos (mesmo conjunto, ordem diferente) | sha256 `72c59e7d…` vs `4f56633d…`; seed por índice de linha em `test_model.batch` |

Por isso as propostas estão ordenadas assim: **(P0) tornar a medição confiável**, **(P1) alinhar o dado
de treino ao que a produção pede e ao que o professor cobra**, **(P2) só então** experimentos de
treino, cada um com protocolo multi-seed de treino. Propostas de modelo maior, RL ou DPO ficam em P3,
condicionadas a P0/P1.

---

## 2. Diagnóstico por camada

### 2.1 Arquitetura e acoplamentos

Fatos (com caminho):

- Fluxo: `DB/questoes.db` → `src/extract_data.py` → `data/*.jsonl` → `src/train.py` → `src/export_gguf.py`
  → `tests/test_model.py::generate_validated` (GBNF + verificador + best-of-N) → `src/promover_checkpoint.py`.
- `SYSTEM_PROMPT`/`USER_TEMPLATE` têm fonte única (`extract_data.py:37-56`) e são idênticos em 100% dos
  exemplos de treino e val. Bom.
- Constantes duplicadas sem fonte única: `MAX_SEQ_LENGTH` em `train.py:44`, `export_gguf.py:37`,
  `evaluate.py:53`; `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_P` em `test_model.py` e `evaluate.py`.
- `congelar_baseline.py:91-99` lê hiperparâmetros **por regex sobre o fonte** de `train.py`.
- `test_model.find_gguf` sem `--model` pega o primeiro `.gguf` em ordem alfabética entre 14 diretórios
  `outputs/gguf*` (`test_model.py:216-224`).
- Template de chat: treino usa Jinja HF com `enable_thinking=False` (insere `<think>\n\n</think>` vazio);
  inferência usa llama.cpp `--jinja`/`-rea off` ou `/v1/chat/completions` sem flag. Equivalência
  não é coberta por teste; a grammar começa em `{` e mascara qualquer divergência.
- `outputs/gguf_*_gguf/Modelfile` (gerado pelo Unsloth) publica `temperature 0.6, top_k 20, top_p 0.95`,
  diferente do harness (`0.7 / 0.8`).
- Suíte: 834 testes, 3 falhas crônicas acopladas a estado externo (`run_base_conhecimento.sh`
  inexistente; limiar de viés `> 35%` sobre corpus que agora dá 31,2%).

### 2.2 Base de dados (`train_curado_v7b.jsonl`, 2 846 exemplos / 2 948 questões)

- Séries: 1º 232 / 2º 836 / 3º 192 / 4º 59 / 5º 723 / 9º 804. O 2º ano é 29% da base; o val
  (30 itens) não tem 1º/3º/4º e G7 só vigia 5º e 9º.
- 79 pares (ano, habilidade); mínimo (9º, H17 geometria) com 12 exemplos.
- Letras do gabarito balanceadas (≈20% cada) desde a v5. Alternativa E fixa "Nenhuma das alternativas
  anteriores" em 23% dos itens, **nunca gabarito** — o modelo aprende que E é sempre descartável e o
  gate mede "Nenhuma das…" em 13–25% das saídas.
- Assistant: média 253 tokens, p99 495, máx 1 221; 27 exemplos (0,95%) acima de 512 tokens, que é o
  `MAX_NEW_TOKENS` de inferência; 14 sequências completas acima de 1 024.
- 47 exemplos (1,7%) pedem `quantidade` 2–5 numa única resposta, **incluindo todos os 75 itens
  `REFORCO-`** que motivaram a v7b. A produção pede sempre `quantidade=1` e só verifica a 1ª questão.
- Sintético: 134 (4,7%). Resoluções iniciadas por "Correto": 16% na v3, 0,5% na v7b (melhoria real do ciclo v6).
- Val: 30 itens reais, 19 do 9º ano, letra mais frequente 36,7%, assistant máx 302 tokens.

### 2.3 Treino (`src/train.py`)

Hiperparâmetros e ausências:

| Item | Valor | Comentário |
|---|---|---|
| LoRA | r=16, α=32, dropout 0, todas as projeções | conforme LoRA Without Regret |
| LR / schedule | 2e-4, cosine, warmup 5%, wd 0,01 | conforme; referência usou LR constante |
| Batch | 2 × 8 = 16 | abaixo do teto 32 recomendado |
| Épocas / parada | 3, `patience=2` | com 3 épocas a parada antecipada **nunca dispara**; só `load_best_model_at_end` atua |
| Determinismo | `seed=42`, `random_state=42` | sem `full_determinism`, `data_seed`, `cudnn.deterministic`, `CUBLAS_WORKSPACE_CONFIG` |
| Packing / NEFTune / group_by_length | ausentes | ver P2 |
| Truncagem | silenciosa | nenhum aviso quando exemplo > `MAX_SEQ_LENGTH` |

Curvas (train loss médio / eval loss por época):

| run | ép. 1 | ép. 2 | ép. 3 | melhor |
|---|---|---|---|---|
| v6 | 0,589 / 0,6808 | 0,428 / 0,6684 | 0,353 / 0,6929 | ép. 2 |
| v7 | 0,591 / 0,6678 | 0,430 / 0,6638 | 0,355 / 0,6843 | ép. 2 |
| v7b | 0,589 / 0,6744 | 0,425 / 0,6720 | 0,349 / 0,6886 | ép. 2 |
| v7c (seq 2048) | 0,588 / 0,6750 | 0,424 / 0,6736 | 0,349 / 0,6907 | ép. 2 |
| v7d (= v7b) | 0,589 / 0,6743 | 0,425 / 0,6724 | 0,350 / 0,6897 | ép. 2 |

Leitura: o padrão é estável e reprodutível em cinco treinos: eval mínima na época 2, sobe na 3
(overfitting leve, train 0,59 → 0,35). O `eval_loss` sobre 30 itens separa épocas mas **não separa
treinos**: v7b e v7d diferem em 0,0004 e têm vereditos opostos. Já no step 5 o `grad_norm` difere na
3ª casa (4,17 / 4,16 / 4,17): a divergência começa no primeiro passo, não no early stopping.

### 2.4 Export e inferência

- Export: merge do adaptador (treinado sobre NF4) em base fp16 recarregada, depois `q4_k_m` sem
  imatrix. A perda QLoRA→merge→Q4 não é medida (G9 "não medida").
- Sampling: `temp 0,7`, `top_p 0,8`, sem `top_k`/`min_p`/`repeat_penalty` explícitos (defaults do
  llama.cpp valem). Best-of-N com `retries=1` (até 2 amostras). `MAX_NEW_TOKENS=512` < p99 do treino.
- Timeout de 300 s vira "geração inválida" e entra como falha em G2.

### 2.5 Gate e aceitação

- G1 e G7 decidem **sem tolerância** por rodada, em subconjuntos de 7 (5º ano) a 30 itens: 1 item = 3,3 pp
  no G1 e 14 pp no G7-5º. É assim que v7c e v7d reprovaram em G1/G7.
- G2*/G3* usam McNemar em n=150 pares, mas G3* só vê a metade verificável.
- A amostra humana (n=20/braço) tem IC de Wilson de ±20 pp: não detecta diferenças menores que ~25 pp.
- O baseline em produção foi treinado com 963 exemplos (`train.jsonl`); três bases 3× maiores não o
  superaram no julgamento humano. Isso é o fato mais importante do projeto e nenhuma hipótese
  foi testada formalmente sobre ele (ver P1.1).

---

## 3. Evidência externa relevante

Resumo do que a literatura recente sustenta ou contradiz nas decisões do projeto. Fontes na seção 7.

1. **Hiperparâmetros LoRA** (Schulman et al., 2025; replicação parcial HF/TRL): LR ótima de LoRA ≈ 10×
   a de full fine-tuning e quase independente do rank sob escala α/r; LoRA em **todas** as camadas
   (MLP incluída) é necessário; batch grande penaliza LoRA mais que full FT; capacidade raramente é o
   gargalo em datasets de pós-treino. **O projeto já cumpre tudo isso.** Não há ganho esperado em
   mexer em r, α ou LR sem evidência de instabilidade.
2. **Variância de seed em fine-tuning** (Zhou, Savova & Wang, IJCNLP-AACL 2025): 5–10 seeds são
   necessárias para estimar variância; dois modelos com a mesma acurácia agregada podem concordar em
   apenas uma fração das predições individuais; LoRA reduz a variância macro mas **não** a inconsistência
   por exemplo; diferenças dentro da variância de seed devem ser tratadas como inconclusivas. O achado
   v7b≠v7d é exatamente esse fenômeno.
3. **Determinismo em GPU** (guias Megatron-Core/NVIDIA e Primus/ROCm, 2025–2026): reprodutibilidade
   bit a bit exige algoritmos determinísticos, `CUBLAS_WORKSPACE_CONFIG`, autotune do Triton fixado e
   custo de velocidade; Unsloth não documenta chaves equivalentes. Logo: **reduzir variância por
   determinismo é incerto; medir e integrar a variância é o caminho robusto.**
4. **Soups / média de adaptadores** (Wortsman et al., 2022; Soup-Adapters, 2025; AdapterSoup, 2023):
   a média de pesos de runs independentes do mesmo ponto inicial (variando só seed) tende a superar
   qualquer run isolada e reduz sensibilidade a seed, desde que os pesos fiquem próximos. É a forma mais
   barata de converter "5 treinos" em "1 artefato melhor".
5. **Filtro por verificador vale mais que o otimizador** (Xiong et al., 2025 "RFT ≈ GRPO"; DPO
   ponderado por verificador em 1,5B, 2025; TrajFusion, 2026): grande parte do ganho de RL vem do
   filtro implícito de dados; rejection sampling com verificador forte é competitivo em modelos
   pequenos. O projeto **já tem** um verificador determinístico (`check_consistency`) e um pipeline de
   juízes cegos — pode fazer RFT sem RL.
6. **Qualidade × quantidade em modelos pequenos** (LIMA 2023; MindLLM; Zhao et al. 2024; "Data Diversity
   Matters" 2023): poucos exemplos de alta qualidade superam bases maiores e mais ruidosas em modelos
   pequenos; diversidade melhora o pior caso. Consistente com "baseline de 963 > bases de 2,8k".
7. **NEFTune** (Jain et al., ICLR 2024): ganhos em avaliação conversacional, nenhum em benchmarks
   objetivos; sem replicação independente em bases pequenas. Evidência fraca para este caso; só como
   experimento barato.
8. **Qwen3 e thinking** (docs Unsloth/Qwen): SFT só com dados sem raciocínio pode enfraquecer o
   raciocínio; sugerem mistura 75/25 com dados de raciocínio. Não há ablação publicada. O projeto treina
   100% `enable_thinking=False`; a `resolucao_passo_a_passo` funciona como raciocínio curto dentro do JSON.
9. **Quantização** (llama.cpp, Unsloth Dynamic 2.0): em modelos < 3B a perda de Q4_K_M é maior
   proporcionalmente; Q5_K_M custa ~20% a mais de tamanho com ganho perceptível; imatrix ajuda quando
   o corpus de calibração casa com o domínio. Nada disso foi medido no projeto (G9 vazio).
10. **Distratores gerados por LLM** (D-GEN 2025; estudos com professores 2025): ~50% dos distratores
    gerados são aproveitáveis como estão; métricas de superfície não capturam plausibilidade. Reforça
    que o julgamento humano continua sendo o critério, e que a amostra precisa ser maior.

---

## 4. Propostas

Formato: **o quê** · **por quê** · **evidência** · **como medir** · **custo** · **risco**.

### P0 — Tornar a medição confiável (pré-requisito de tudo)

**P0.1 Protocolo multi-seed de treino no gate.**
Toda comparação de hiperparâmetro ou base passa a exigir ≥3 treinos (seeds 42, 43, 44) por braço,
cada um com 5 rodadas de geração, e o gate decide sobre a distribuição (mediana das falhas G2*, G3*;
McNemar sobre os 3×150 pares; G1/G7 com tolerância de 1 item ou binomial). Documentar no código como
exige a regra do projeto ("mudança de critério só após veredito reprovado, documentada").
Por quê: v7b≠v7d. Evidência: item 2 da seção 3. Como medir: rodar os 3 treinos da config v7b atual
**antes** de qualquer outro experimento; a dispersão observada é o "piso de ruído" que todo ganho
futuro precisa superar. Custo: ~3× tempo de treino (≈ horas em RTX 3060), sem API paga.
Risco: nenhum; é só mais compute.

**P0.2 Tolerância estatística em G1 e G7.**
Hoje 1 item em 7 reprova. Trocar "não pode cair" por teste binomial/McNemar com os mesmos n agregados
das 5 rodadas (35 pares para o 5º ano, 95 para o 9º). Mantém a intenção (sem esquecimento) sem
reprovar por ruído. Custo: 1–2 h de código + testes.

**P0.3 Corrigir `val.jsonl`.**
Regerar `data/val.jsonl` como cópia byte a byte de `val_frozen_v1.jsonl` (mesmo conjunto, só ordem e
`meta.lote` diferem) e adicionar teste em `tests/test_entradas_imutaveis.py` que compara os dois sha256.
Motivo: `batch` deriva a seed do índice da linha; o CLAUDE.md promete identidade que hoje não vale.
Custo: minutos. Requer confirmação explícita do usuário porque toca `val*.jsonl`.

**P0.4 Gate no regime de produção.**
Rodar as 5 rodadas com `--sufixo` (como o app) ou, no mínimo, rodar os dois regimes e reportar ambos.
Hoje o gate mede o regime que corresponde a ~10% do treino. Custo: só compute de avaliação.

**P0.5 Amostra humana com poder estatístico.**
Com n=20/braço o IC é ±20 pp. Para detectar 15 pp de diferença (45% → 60%) com 80% de poder são
necessários ~170 itens por braço; para 20 pp, ~100. Propostas práticas:
(a) n=100/braço, pareado (mesmo prompt e seed nos dois braços), professor cego, 1 sessão de ~3 h;
(b) **auto-juiz calibrado**: usar o par validador+revisor cego (`agentes_questoes`, família diferente
via Gemini) sobre os mesmos itens que o professor julgou (v6 + v7b: 80 itens) e medir concordância;
se κ ≥ 0,6, usar o juiz como pré-filtro em n=300+ e reservar o professor para discordâncias.
Custo: (a) tempo de professor; (b) API paga, ~R$ baixo por item, requer `--max-chamadas`.

**P0.6 Medir o que o gate não vê.**
Cruzar as 40 rejeições de professor (v6 + v7b) com o `status` do verificador no relatório: quantas
estavam `ok`? Isso dá a taxa de falso-negativo do verificador e diz quanto do erro é
"resolução coerente com questão errada" (cego por construção) versus aritmética. Custo: 1 h, sem API.

### P1 — Alinhar o dado de treino

**P1.1 Testar a hipótese "menos e melhor".**
O baseline (963 ex.) vence bases de 2,8k no julgamento humano. Experimento controlado com P0.1:
(i) v7b completa; (ii) v7b filtrada a itens com `status=ok` do verificador **e** aprovados por juiz cego
(CONFIRMADO + JUIZES do `auditar_gabaritos_base`), ~1,3–1,6k; (iii) os 963 originais + só as injeções
CONFIRMADO. Mesma config, 3 seeds cada. Evidência: item 6 da seção 3. Custo: 9 treinos.

**P1.2 Reforço no formato da produção.**
Os 75 itens `REFORCO-` (e os demais 47 multi-questão) devem ser divididos em exemplos de
`quantidade=1`, um por questão, com o mesmo `USER_TEMPLATE` do app. Hoje o modelo aprende o reforço
num formato que nunca é usado. Custo: script de 30 linhas + verificar com `verificar_v7`.

**P1.3 Rejection-sampling fine-tuning (RFT) com o verificador existente.**
Gerar com o próprio candidato (ou com o baseline) 5–10 amostras por prompt do plano de cobertura
(`diversidade.py`), manter só as que passam em `check_consistency` com `status=ok`, não são near-dup e
passam no juiz cego; adicionar ao treino como exemplos extras rotulados `RFT-`. Isso ataca
diretamente G2/G3 (o item que o verificador reprova é o que a produção regenera). Evidência: item 5.
Custo: compute local de geração (horas) + API do juiz com teto. Risco: reforçar viés do próprio
modelo (mitigar com teto de 30% de itens RFT e deduplicação contra o banco).

**P1.4 Alternativa E.**
Hoje "Nenhuma das alternativas anteriores" é fixa e nunca correta em 23% dos itens, e o gate observa
13–25% de "Nenhuma" nas saídas. Opções: (a) remover E dos itens de 4 alternativas na extração e deixar
o contrato aceitar 4 ou 5; (b) manter E mas torná-la gabarito numa fração (5–10%) com resolução que o
verificador consiga checar (`fora_das_alternativas` → E). (b) exige mudar grammar e verificador; (a) é
mais simples e elimina um artefato que o modelo memoriza. Custo: (a) 2–3 h incluindo testes.

**P1.5 Cobertura do val e de G7.**
Val de 30 itens, 19 do 9º, nenhum de 1º/3º/4º. Manter `val_frozen_v1` congelado (comparabilidade
histórica) e criar `val_frozen_v2` com 60–90 itens estratificados por série e dificuldade, usado em
paralelo por um ciclo antes de virar o padrão. G7 passa a vigiar todas as séries. Custo: curadoria
humana de 30–60 itens do banco (sem API).

### P2 — Experimentos de treino (só com P0.1 em vigor)

Ordem por relação custo/evidência:

| # | Mudança | Hipótese | Evidência | Custo |
|---|---|---|---|---|
| P2.1 | Época 2 fixa + `EPOCHS=2` com cosine terminando em 0 | 5/5 runs têm mínimo na ép. 2; cosine até o fim da ép. 2 em vez de cortar no meio | curvas §2.3 | nulo |
| P2.2 | Média de adaptadores das 3 seeds (soup) antes do merge | reduz variância e tende a superar a melhor run | item 4 §3 | script de 20 linhas; exige P0.1 |
| P2.3 | Avisar/contar truncagem e `MAX_NEW_TOKENS` ≥ p99 do treino (512 → 640) | 1% do treino excede 512 tokens; inferência corta antes do fim | §2.2 | trivial; passa pelo gate |
| P2.4 | `lora_dropout=0.05` ou `weight_decay=0.05` | overfitting leve ép. 3 | fraco (referência usa dropout 0) | 3 treinos |
| P2.5 | NEFTune α=5 | regularização em base pequena | fraca, sem replicação | 3 treinos |
| P2.6 | Mistura 10–25% de exemplos com raciocínio explícito (`enable_thinking=True`, resolução no `<think>`) | preservar raciocínio do Qwen3 | guia do fornecedor, sem ablação | gera dado + grammar precisa aceitar `<think>` → caro |
| P2.7 | r=32/α=64 | capacidade | referência diz que não é gargalo | baixo valor esperado |

Não recomendado agora: mudar LR, mudar `max_seq_length` (já testado, sem efeito detectável),
aumentar batch (LoRA penaliza), mudar de modelo base (decisão já tomada: Qwen3-1.7B).

### P3 — Export, inferência e engenharia

**P3.1 Medir a quantização (G9).** Gerar logits fp16 nos 30 prompts do val e calcular KL-divergência e
top-1 agreement para Q4_K_M, Q4_K_M+imatrix (calibrado com 200 exemplos do treino), Q5_K_M e Q8_0.
Se a KL Q4→Q5 for grande e o app aguentar +20% de tamanho, Q5_K_M é ganho gratuito. Custo: 2 h.

**P3.2 Best-of-N.** `retries=1` dá 2 amostras. Medir o custo marginal de `retries=2` em latência real
no dispositivo contra a queda de falhas G2; o gate `--comparar-inferencia` já existe para isso.

**P3.3 Fonte única de constantes.** Criar `src/config_modelo.py` com `MAX_SEQ_LENGTH`, `MAX_NEW_TOKENS`,
sampling, `N_CTX`; `congelar_baseline` passa a importar em vez de regex. Gerar o `Modelfile` com os
mesmos parâmetros do harness ou não publicá-lo.

**P3.4 Teste de equivalência de template.** Teste que tokeniza um exemplo com o Jinja HF
(`enable_thinking=False`) e com o template do GGUF (`llama-cli --jinja`) e compara o prefixo até o
assistant. Fecha a única fenda não testada entre treino e inferência.

**P3.5 `find_gguf` sem default silencioso.** Exigir `--model` ou falhar com lista dos candidatos.

**P3.6 Manifesto por treino.** `train.py` grava `outputs/lora_<v>/MANIFESTO.json` com hash do
dataset, constantes, versões de unsloth/trl/torch, seed, GPU e `trainer_state` resumido. Hoje só
existe para o baseline, feito à mão.

**P3.7 Suíte verde.** Corrigir os 3 testes que falham por estado externo (script renomeado; limiar
de viés calculado sobre o corpus em vez de constante). Instalar `pytest-timeout`.

---

## 5. Plano sugerido (ordem e critério de parada)

1. **Semana 1 — medir o ruído.** P0.6 (1 h), P0.3 (com confirmação), P3.7, P0.2. Depois P0.1: 3 treinos
   da config v7b atual. Resultado: distribuição de G2*/G3*/G6* por seed. Se a dispersão entre seeds for
   maior que a diferença candidato–baseline, **nenhuma** conclusão de v6/v7/v7b era válida e o relatório
   deve dizer isso.
2. **Semana 2 — alinhar dado.** P1.2 e P1.4(a) (baratos). P1.1 com 3 seeds por braço. Soup (P2.2)
   sobre o braço vencedor. Gate no regime do app (P0.4).
3. **Semana 3 — aceitação.** Amostra humana n≥100/braço (P0.5a) para o vencedor; em paralelo, calibrar
   o auto-juiz (P0.5b) nos 80 itens já julgados.
4. **Só depois:** P1.3 (RFT), P3.1 (quantização), P2.4–P2.6.

Critério de publicação (mantém a política atual): gate multiseed com 3 seeds de treino **e** amostra
humana com IC inferior do candidato acima da média do baseline. Enquanto isso, o baseline continua.

---

## 6. O que não fazer

- Não publicar v7b/v7c/v7d (decisão já tomada em `ciclo_v7/SPEC.md`).
- Não mudar r, α, LR ou modelo base sem P0.1; a literatura e as curvas dizem que não é aí.
- Não escrever em `train_curado.jsonl`, `train.jsonl`, `val*.jsonl` sem pedido explícito.
- Não ativar o verificador de geometria em modo ativo sem a auditoria humana prevista.
- Não tratar `eval_loss` em 30 itens como critério de seleção entre treinos (só entre épocas).

---

## 6b. Decisão com segunda opinião do Jev (2026-10-14)

Jev (TypeSafe System One, `typesafe/jev-1.13`, via OpenRouter `POST /api/v1/systemone`) recebeu como
estado os fatos medidos, este documento e `Doc/PROPOSTAS_MELHORIA_ARQUITETURA_TREINAMENTO.md`, e
respondeu 22 ações × (Score de prioridade + Noul de ganho) + 7 escolhas. Resposta completa e válida
(sem truncagem) em `outputs/melhoria_v7/jev_prioriza_propostas.json`; custo US$ 0,001.
Script: `jev_prioriza_propostas.py` (scratchpad da sessão).

**Prioridade "Imediato" (pré-requisito, fazer antes de qualquer treino), P(Imediato):**

| Ação | P(Imediato) | P(ganho ≥10 pp em 2 ciclos) |
|---|---|---|
| Multiseed de treino (3–5 seeds, gate sobre a distribuição) | 0,98 | 0,21 |
| Manifesto por execução + README/CLI apontando ao corpus certo | 0,95 | 0,19 |
| Gate no regime do app (com sufixo) | 0,90 | 0,22 |
| Corrigir `val.jsonl` = `val_frozen_v1.jsonl` | 0,87 | 0,10 |
| Tolerância estatística em G1/G7 | 0,59 | 0,20 |
| Cruzar rejeições do professor × status do verificador | 0,53 | 0,12 |

**Prioridade "Próximo ciclo":** reforço em `quantidade=1` (0,71), remover alternativa E fixa (0,73),
experimento "menos e melhor" (0,48; maior ganho esperado da lista, 0,45), auto-juiz calibrado,
amostra humana n≥100, `val_frozen_v2`, fonte única de constantes.

**"Adiar":** RFT (0,79 adiar; ganho esperado alto 0,44, mas condicionado a medir o verificador
primeiro), soup de adaptadores, medir quantização, proveniência SQLite, `EPOCHS=2`.

**"Descartar" por enquanto:** sweep de LR/rank (0,88 descartar), dropout/NEFTune (0,92), mistura de
exemplos com thinking (0,97), migrar para `assistant_only_loss` do TRL (0,91).

**Escolhas:**

| Pergunta | Resposta | Distribuição |
|---|---|---|
| Gargalo principal | **qualidade/alinhamento do dado** | 0,98 (ruído de medição 0,01) |
| Primeira ação | manifesto + README/CLI | 0,56 (multiseed 0,25; val.jsonl 0,10) |
| Sweep LR/rank (doc A) × não mexer (doc B) | não fazer sweep | 0,66 (só após dado 0,34; fazer 0,00) |
| Alternativa E fixa | remover no próximo ciclo e medir | 0,81 |
| RFT | medir falso-negativo do verificador antes | 0,97 |
| 2 épocas × 3 com best | fixar 2 | 0,49 (testar ambos no multiseed 0,41) — indeciso |
| Documento mais acionável | este (B) | 1,00 |
| Publicar v7b/v7c/v7d agora | não | P(sim) = 0,04 |

**Decisão final adotada:**

1. **Nenhum v7 é publicado.** Baseline continua.
2. **Antes de qualquer novo treino:** manifesto por execução e `train.py` exigindo `--train`/`--val`
   explícitos; `val.jsonl` idêntico ao congelado (com autorização, pois toca `val*`); gate rodando com
   sufixo; tolerância estatística em G1/G7 documentada no código; cruzar rejeições humanas com o
   verificador. Depois, 3 treinos da config atual com seeds 42/43/44 para medir o piso de ruído.
3. **Próximo ciclo de base (v8):** reforço em `quantidade=1`; remover a alternativa E fixa; três
   braços do experimento "menos e melhor", 3 seeds cada; amostra humana pareada n≥100 no vencedor,
   com auto-juiz calibrado nos 80 itens já julgados.
4. **Hiperparâmetros ficam como estão.** Sem sweep de LR/rank, sem dropout/NEFTune, sem thinking,
   sem trocar o mecanismo de máscara de loss. A única mudança de treino admitida é 2 vs 3 épocas,
   e só dentro do protocolo multiseed, porque o Jev ficou dividido (0,49 × 0,41).
5. **RFT, soup, quantização e proveniência SQLite** entram só depois que o item 2 produzir a
   medida de falso-negativo do verificador e o piso de ruído.

---

## 6c. Sequência de ações até o objetivo

**Objetivo:** um modelo cuja validade, julgada por professor cego, supere o baseline com significância,
e só então publicar. Critério de publicação: gate multiseed com 3 seeds de treino e limite inferior do
IC de 95% do candidato acima da média do baseline na amostra humana n≥100/braço.

Legenda: **[sem API]** só compute local · **[API]** exige teto `--max-chamadas` · **[OK?]** precisa de
autorização sua antes · **[humano]** depende do professor.

### Fase 0: medir o que já temos (1 dia, sem treino)

| # | Ação | Saída | Decide |
|---|---|---|---|
| 0.1 | Cruzar as 40 rejeições do professor (v6 + v7b) com o `status` do verificador [sem API] | taxa de falso-negativo do verificador e tipo de erro dominante | se RFT é viável; onde focar a curadoria |
| 0.2 | Rodar `pytest` e corrigir os 3 testes que falham por estado externo; instalar `pytest-timeout` [sem API] | suíte verde | base segura para mexer no gate |

### Fase 1: tornar o treino e a medição confiáveis (2–3 dias)

| # | Ação | Saída | Critério de pronto |
|---|---|---|---|
| 1.1 | `train.py` exige `--train` e `--val`; grava `MANIFESTO.json` por treino (hash dos dados, versões, seed, GPU, truncagens, melhor checkpoint); README atualizado | treino rastreável | qualquer LoRA liga a dados e ambiente sem ambiguidade |
| 1.2 | Regerar `data/val.jsonl` idêntico a `val_frozen_v1.jsonl` e teste de sha256 **[OK?]** (toca `val*`) | validação imutável de fato | sha256 iguais; teste passa |
| 1.3 | Gate com tolerância estatística em G1/G7, documentada no código como mudança pós-reprovação | critério que não reprova por 1 item em 7 | testes do gate atualizados |
| 1.4 | Rodar o gate também no regime do app (`--sufixo`) | veredito nos dois regimes | os dois reportados lado a lado |

### Fase 2: piso de ruído de treino (≈1 dia de GPU, você roda o treino)

| # | Ação | Saída |
|---|---|---|
| 2.1 | 3 treinos da config v7b atual com seeds 42, 43, 44 (você executa no seu terminal; eu preparo os comandos) | 3 LoRAs |
| 2.2 | Export GGUF + 5 rodadas + gate para cada seed | distribuição de G2*, G3*, G6* por seed |
| 2.3 | Registrar o piso de ruído em `ciclo_v7/SPEC.md` | **parada:** se a dispersão entre seeds for maior que a diferença candidato–baseline, v6/v7/v7b eram inconclusivas e o relatório diz isso |

### Fase 3: ciclo v8 de dados (3–4 dias, sem API ou com teto)

| # | Ação | Custo |
|---|---|---|
| 3.1 | Dividir os 75 `REFORCO-` e os demais multi-questão em exemplos de `quantidade=1` | [sem API] |
| 3.2 | Remover a alternativa E fixa na extração (contrato aceita 4 ou 5 alternativas; grammar e verificador ajustados) | [sem API] |
| 3.3 | Montar os 3 braços: (i) v7b completa, (ii) v7b filtrada a CONFIRMADO + `ok` + juiz cego, (iii) 963 originais + injeções CONFIRMADO | [sem API] |
| 3.4 | Treinar 3 braços × 3 seeds = 9 treinos; export e gate multiseed em cada | GPU local |
| 3.5 | Teste dentro do protocolo: 2 épocas contra 3 épocas no melhor braço | GPU local |

### Fase 4: aceitação humana (1 semana, depende do professor)

| # | Ação | Custo |
|---|---|---|
| 4.1 | Calibrar o auto-juiz (validador + revisor cego) contra os 80 itens já julgados; κ ≥ 0,6 para usar como triagem | [API] com teto |
| 4.2 | Amostra humana cega pareada, n≥100/braço, só do melhor braço contra o baseline | [humano] |
| 4.3 | Apuração com Wilson 95% por unidade temática | [sem API] |

### Fase 5: decisão e só então outras alavancas

| Resultado da 4.3 | Ação |
|---|---|
| IC inferior do candidato acima da média do baseline | pedir sua autorização para publicar (publicação segue pausada até lá) |
| Sem diferença detectável | manter baseline; abrir RFT (se 0.1 mostrar verificador confiável), depois soup de adaptadores e medição de quantização |
| Pior que o baseline | descartar o braço; revisar curadoria a partir dos erros do professor |

### Regras que valem em todas as fases

- Nenhum push ao Hugging Face sem autorização explícita.
- Nenhuma escrita em `train_curado.jsonl`, `train.jsonl` ou `val*.jsonl` sem o **[OK?]** marcado.
- Nenhum job de GPU lançado por mim sem pedido; os treinos rodam no seu terminal.
- Qwen3-1.7B fixo. Sem sweep de LR/rank, dropout, NEFTune ou thinking neste ciclo.
- Toda mudança de critério do gate vem **depois** de um veredito reprovado e fica comentada no código.

---

## 7. Fontes

Internas: `src/train.py`, `src/promover_checkpoint.py`, `tests/test_model.py`, `src/extract_data.py`,
`outputs/checkpoints_v{6,7,7b,7c,7d}/checkpoint-*/trainer_state.json`,
`outputs/relatorios/VEREDITO_v7{b,c,d}_multiseed_n150.json`,
`outputs/amostra_humana_v7b_vs_baseline/apuracao.json`, `ciclo_v7/SPEC.md`, `Doc/PROCEDIMENTO_AMOSTRA_HUMANA.md`.

Externas:
- Schulman et al., *LoRA Without Regret*, Thinking Machines, 2025-09. https://thinkingmachines.ai/blog/lora/ ; replicação TRL: https://huggingface.co/docs/trl/main/en/lora_without_regret
- Zhou, Savova, Wang, *Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning LLMs*, IJCNLP-AACL 2025. https://arxiv.org/abs/2503.07329
- Wortsman et al., *Model soups*, ICML 2022. https://arxiv.org/abs/2203.05482 ; *Soup-Adapters*, 2025. https://arxiv.org/abs/2507.05807 ; AdapterSoup, 2023. https://arxiv.org/abs/2302.07027
- Xiong et al., *A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce*, 2025. https://arxiv.org/abs/2504.11343
- *Hard Negative Sample-Augmented DPO Post-Training for Small Language Models*, 2025. https://arxiv.org/abs/2512.19728
- *Beyond Rejection Sampling: Trajectory Fusion*, Findings ACL 2026. https://arxiv.org/abs/2602.04391
- *Small Language Models Need Strong Verifiers to Self-Correct Reasoning*, 2024. https://arxiv.org/abs/2404.17140
- Jain et al., *NEFTune*, ICLR 2024. https://arxiv.org/abs/2310.05914
- *Is Training Data Quality or Quantity More Impactful to Small Language Model Performance?*, 2024/25. https://arxiv.org/abs/2411.15821 ; *Data Diversity Matters for Robust Instruction Tuning*, 2023. https://arxiv.org/abs/2311.14736
- *An Empirical Study of SFT-DPO Interaction and Parameterization in Small Language Models*, 2026. https://arxiv.org/abs/2603.20100
- Unsloth, *Qwen3: How to Run & Fine-tune*. https://docs.unsloth.ai/basics/qwen3-how-to-run-and-finetune ; Unsloth Dynamic 2.0 GGUFs. https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
- HF blog, *The 4 Things Qwen-3's Chat Template Teaches Us*. https://huggingface.co/blog/qwen-3-chat-template-deep-dive
- NVIDIA Megatron-Core, *Deterministic Training*. https://docs.nvidia.com/megatron-core/developer-guide/nightly/user-guide/deterministic-training.html ; AMD Primus, *Determinism and reproducibility*. https://rocm.docs.amd.com/projects/primus/en/main/04-technical-guides/determinism-and-reproducibility.html
- llama.cpp, `llama-perplexity` (KL-divergence). https://github.com/ggml-org/llama.cpp/tree/master/tools/perplexity
- D-GEN, *Distractor generation*, 2025. https://arxiv.org/abs/2504.13439 ; *Exploring Automated Distractor Generation for Math MCQs via LLMs*, 2024. https://arxiv.org/abs/2404.02124
- FMBench (SFT+GRPO em Qwen3-1.7B para formatação), 2026. https://arxiv.org/abs/2602.06384
