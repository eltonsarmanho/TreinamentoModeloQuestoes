# Ciclo v7 da base: curadoria item a item

Entrada: `data/train_curado_v6.jsonl` (2.807 exemplos, 2.849 questões). Saída: `data/train_curado_v7.jsonl`.

## O que foi feito

`src/curar_base_v7.py` passou CADA questão da v6 por um professor de Matemática SAEB (Maritaca
`sabia-4`, papel "curador"), que resolve o item às cegas (sem ver gabarito, resolução nem
dificuldade pedida) e devolve veredito `aprovado | corrigivel | inviavel`. O código compara com o
gabarito e decide `confirmada | corrigida | gabarito_discorda | correcao_rejeitada | inviavel`.
Toda correção passa por uma 2ª chamada cega (o validador já calibrado de `agentes_questoes`) antes
de entrar. Custo real: R$ 42,68 para as 2.849 questões (0 abstenções sem retry; 6 abstenções
ficaram sem decisão e o item continuou como estava).

Política (nunca muda sem decisão humana, ver `src/curar_base_v7.py` docstring):
- item NÃO real (injetado/destilado/sintético): correção e remoção automáticas, documentadas em
  `outputs/melhoria_v7/proveniencia_v7.jsonl`;
- item REAL (`real_banco`, `real_L2`): correção e remoção viram PROPOSTA em
  `outputs/melhoria_v7/propostas_reais.jsonl` — só entram com decisão humana explícita.

## Decisão humana sobre os itens reais (2026-10-13)

7 propostas de itens reais. O responsável aprovou as 4 **correções** e rejeitou as 3 **remoções**:

| código | tipo | decisão |
|---|---|---|
| EF04MA04-005-L2-2026-09 | correção | aprovada (completou dado ausente: "cada músico usa um instrumento") |
| EF01MA02-016-L2-2026-09 | correção | aprovada, com ajuste do responsável (pergunta trocada para "caixas") |
| MT2077FH16MT | correção | aprovada (unidade "kg" explicitada no enunciado) |
| MT9055DH11MT | correção | aprovada, **com valor diferente do proposto pelo curador**: o responsável trocou o total para R$ 140,00 (dá solução inteira b=2, c=4), não os R$ 130,00 que o curador sugeriu (que também não dava inteiro) |
| MT9077MH16MT | remoção | aprovada → item removido da base ("vamos excluir") |
| EF03MA01-015-L2-2026-09 | remoção | aprovada → item removido da base ("vamos excluir") |
| MT5068MH14MT | remoção | aprovada → item removido da base ("vamos excluir") |

Registro em `outputs/melhoria_v7/EDICOES_REAIS_HUMANAS.json` (as 4 correções, antes/depois) e
`outputs/melhoria_v7/QUARENTENA_REAIS_HUMANA.json` (as 3 remoções de itens reais, aplicadas após
a decisão "vamos excluir").

`data/train_curado_v7.jsonl`: 2.807 → 2.774 (33 não-reais em quarentena pelo curador) → 2.771
(−3 itens reais removidos por decisão humana explícita: MT9077MH16MT, EF03MA01-015-L2-2026-09,
MT5068MH14MT).

## Verificação (`src/verificar_v7.py`, espelha `verificar_candidata_v6.py`)

8 de 9 critérios passam sem ressalva (schema, sem duplicata/contaminação com val*, 0
`matematicamente_problematica`, % verificável e near-dup não pioram, letras 18–22%,
explicabilidade de distratores não piora).

### Exceção registrada (2026-10-13)

**V7-03a** (REFUTADO = 0 e FRACO = 0, herdado do v6): a v7 tem 2 REFUTADO, exatamente as 2 edições
reais acima cujo texto/valor diverge do registro oficial do banco SQLite por causa da própria
correção aprovada, não de um erro novo:
- `EF01MA02-016-L2-2026-09`: o banco guarda o texto oficial "9 REMEDIOS." para a alternativa
  correta; a correção trocou a pergunta para "quantas caixas" (elimina a ambiguidade de unidade
  "caixas de remédio" × "remédios"), então o texto virou "9 caixas." — diverge na PALAVRA, não no
  valor.
- `MT9055DH11MT`: o banco registra R$ 120,00, que não fecha em número inteiro de camisas/bermudas
  (b = 1,6). A correção aprovada usa R$ 140,00 (b = 2, c = 4, confere: 4×20 + 2×30 = 140) — diverge
  do banco porque o banco tinha o defeito que motivou a correção, não porque a correção está errada.

`auditar_gabaritos_base.py` compara o texto/valor da alternativa correta contra o registro oficial
do SQLite; ele não distingue "edição humana aprovada" de "erro novo introduzido". O critério NÃO foi
alterado para os próximos ciclos — a exceção vale só para estes 2 códigos, verificados caso a caso
nesta decisão. `venv/bin/python src/verificar_v7.py --excecao=V7-03a` grava a aprovação com a
exceção registrada em `ciclo_v7/ANTES_DEPOIS_v7.json`.

## Treino v7, gate de promoção e investigação (2026-10-13)

Treinado no `train_curado_v7.jsonl` (2.771 exemplos). O gate mecânico
(`promover_checkpoint.py --perfil multiseed`) deu **NÃO PROMOVIDO** (3 rodadas, n=90):
gates bloqueantes reprovados G1, G2, G5, G7. Rodada ampliada para 5 rodadas (n=150) CONFIRMOU
o efeito como estatisticamente significativo (McNemar G2 p=0,006, G3 p=0,021 — G3 passou a
reprovar também), não ruído de amostra pequena: `outputs/relatorios/VEREDITO_v7_multiseed_n150.json`.

**Diagnóstico** (Jev via OpenRouter + agente Claude independente, convergentes; depois confirmado
nos itens reais gerados): a curadoria derrubou a taxa de "Nenhuma das alternativas anteriores" no
candidato de 58,0% para 23,3% (ganho real de viés de gabarito, G6 passa: 42,2%→28,9%). Sem essa
muleta, o modelo de 1,7B precisa gerar 4 distratores plausíveis de verdade e erra a própria
aritmética de números grandes, concentrado em 3 habilidades: `5º H03` (soma/subtração até 6
ordens), `5º H01` (escrever número por extenso) e `9º H02` (operações com números reais). A
quantidade de exemplos reais dessas habilidades não caiu na curadoria (v6→v7: 32→31, 39→38, 32→32)
— o problema é padrão de distrator mal aprendido, não falta de volume.

## Reforço sintético (`src/reforco_distratores_v7.py`, 2026-10-13)

Gera `data/train_curado_v7b.jsonl` = v7 (2.771) + 85 exemplos / 135 questões de reforço nas 3
habilidades acima, com gabarito e distratores computados em Python via `num2words`/aritmética
exata (nunca por LLM): 100% verificável, 0 falha de schema, 0 contaminação com val*.

`src/verificar_v7.py` contra `v7b` falha em `V7-03d` (pouco_explicativa) e `V7-03e` (near-dup) —
**exceção aceita, não corrigida**: esses critérios assumem curadoria sem item novo; um lote de
reforço repete deliberadamente o mesmo padrão de distrator pedagógico em várias habilidades/
dificuldades, e o detector de near-dup (trigramas de palavras sobre o enunciado com números
mascarados) não distingue "mesmo padrão ensinado de novo" de duplicata real — a maior parte da
frase é texto fixo do template, então variar nome/contexto não derruba o Jaccard o bastante quando
~135 questões se concentram em só 3 habilidades. `V7-03a` continua exceção herdada da v7 pura.
O critério de aceitação real para `v7b` é o retreino e o gate `G1/G5/G7`, não esta auditoria
estática — ainda não rodado.

    venv/bin/python src/reforco_distratores_v7.py --num-per-skill 15   # já gerado
    venv/bin/python src/train.py --train data/train_curado_v7b.jsonl --val data/val.jsonl \
      --out outputs/lora_v7b --checkpoints outputs/checkpoints_v7b

## Gate v7b e amostra humana (2026-10-13)

Treinado em `train_curado_v7b.jsonl`, exportado, 5 rodadas de seed. O gate mecânico
(`promover_checkpoint.py --perfil multiseed`, n=150) deu **PROMOVIDO**: todos os gates
bloqueantes passaram, G2*/G3* pioraram sem significância (6 falhas novas pareadas, McNemar
p=0,125), viés de gabarito melhorou (40,0%→35,3%), "Nenhuma das alternativas anteriores" caiu de
58,0% para 12,7%. Veredito: `outputs/relatorios/VEREDITO_v7b_multiseed_n150.json`.

A **amostra humana cega** (40 questões, 20 por origem, `outputs/amostra_humana_v7b_vs_baseline/`)
contradisse o gate: validade geral 35,0% (v7b) contra 45,0% (baseline), IC95 sobrepostos
(não significativo, mas sem sinal de melhora — mesmo padrão já visto na v6: 35% vs 50%).
`correcao_matematica` caiu de 70% (baseline) para 60% (v7b), o ponto que o gate tinha dado como
ganho. Os 6 itens que o professor reprovou por conta errada batem com o padrão do G2*/G3*:
resolução calcula o valor certo e as 5 alternativas não o contêm. Outros motivos de reprovação
não são do modelo: figura/tabela ausente no enunciado, geometria com premissa impossível
(paralelogramo/triângulo), enunciado contraditório.

**Conclusão: o gate mecânico não substitui a amostra humana — v7b não está pronta para
publicação**, pelo mesmo motivo que travou a v6.

## Correção do auditor de gabaritos (2026-10-13)

`auditar_gabaritos_base.origem()` não reconhecia o prefixo `REFORCO-` dos itens de reforço
sintético e caía no `real_banco` por padrão, classificando as 135 questões como `FRACO`
("nunca_julgado_com_os_prompts_atuais") em vez de `sintetico`/`CONFIRMADO`. Bug só de
rotulagem na auditoria — não afeta `check_consistency` nem o treino. Corrigido em
`src/auditar_gabaritos_base.py` (prefixo `REFORCO-` tratado como `SINT-`); confirmado:
0 FRACO, 135 questões agora `CONFIRMADO/sintetico`.

## Não-determinismo de treino encontrado (2026-10-14)

Testada a hipótese de que truncar 17/2.846 exemplos (0,6%) em `max_seq_length=1024`
prejudicava o modelo: treino **v7c** (mesma base v7b, mesmos hiperparâmetros, `SEED=42`,
só `max_seq_length=2048`) deu **NÃO PROMOVIDO** no gate (G1, G2*, G3*, G7 falharam).
Para isolar a causa, rodou-se **v7d**: repetição **exata** da v7b (mesma base, mesmos
hiperparâmetros, `SEED=42`, `max_seq_length=1024` de volta) — sem nenhuma mudança
deliberada. **v7d também deu NÃO PROMOVIDO**, pelos mesmos 4 gates, com magnitude parecida
à da v7c (`outputs/relatorios/VEREDITO_v7{b,c,d}_multiseed_n150.json`).

| | v7b (PROMOVIDO) | v7c (seq=2048) | v7d (repetição de v7b) |
|---|---|---|---|
| G2* falhas | 1→6/150 (p=0,125) | 1→11/150 (p=0,006) | 1→9/150 (p=0,021) |
| G3* consistência | 98,5%→92,9% (p=0,125) | 98,5%→88,3% (p=0,021) | 98,5%→89,7% (p=0,039) |
| eval_loss (melhor época, sempre época 2) | 0,67204 | 0,67356 | 0,67242 |

As três rodadas escolheram a **mesma época** como melhor checkpoint, com `eval_loss`
quase idêntico (diferença na 4ª casa decimal) — então o early stopping escolhendo época
diferente **não** é a causa. `src/train.py` fixa `SEED=42` em `get_peft_model(random_state=...)`
e `SFTConfig(seed=...)`, mas não usa `full_determinism=True`, `cudnn.deterministic`, nem
`CUBLAS_WORKSPACE_CONFIG`; o treino roda em bf16 com kernels Triton do Unsloth. Perturbações
de ponto flutuante não controladas por essas flags, acumuladas ao longo do treino, bastam
para produzir pesos finais que mudam o veredito do gate.

**Conclusão (subagente Claude + segunda opinião do Jev, `typesafe/jev-router`, convergentes;
resposta do Jev veio truncada por `max_tokens`, finish_reason=length, então não conta como
veredito estruturado validado — só como leitura qualitativa):** a pergunta original sobre
`max_seq_length` continua **sem resposta** — v7c e v7d reprovam pelos mesmos gates com
magnitude parecida, então não dá para atribuir a reprovação ao `max_seq_length`
especificamente. Mais grave: o gate mecânico, no tamanho de efeito que está em jogo aqui
(poucos pontos percentuais em n=150, subconjuntos de ~14-19 itens em G7), decide **dentro do
próprio ruído de treino-a-treino** — v7b e v7d são o mesmo experimento e deram vereditos
opostos. Isso é o terceiro ciclo (depois de v6 e v7) em que PROMOVIDO mecânico não é
confiável sozinho.

**Decisão: não publicar v7b, v7c nem v7d.** v7b já tinha falhado na amostra humana (35%
contra 45% do baseline, `correcao_matematica` caindo de 70% para 60%) antes mesmo deste
achado. v7c/v7d reprovam no próprio gate, mas isso pode ser ruído de treino, não prova de
que sejam piores. O baseline permanece em produção.

**Pendências para o próximo ciclo, em ordem de custo-benefício** (não executadas ainda):
1. Estudo de variância de seed: treinar a mesma config com 3-5 `SEED` diferentes e comparar
   a distribuição dos gates, não um único ponto, antes de aceitar qualquer veredito de
   promoção ou de hiperparâmetro.
2. Ampliar a amostra humana (hoje n=20 por braço, ICs largos) para n≥100 por braço, ou
   calibrar um juiz automático contra as 40 anotações já feitas.
3. Medir quantos dos itens que o professor reprovou por "conta errada" tinham
   `status="ok"` no verificador determinístico — quantifica a taxa de falso-positivo do
   G2*/G3* atual.

## Fase 1 do plano de melhoria: confiabilidade do treino e da medição (2026-10-14)

Plano em `Doc/PROPOSTAS_MELHORIA_TREINO_2026-10.md`, seção 6c. Nenhum treino, export ou avaliação
foi executado nesta fase.

**1.1 Treino rastreável (feito).** `src/train.py` agora exige `--train` e `--val` (antes o padrão
era `data/train.jsonl`, 963 exemplos, enquanto o ciclo usava `train_curado_v7b.jsonl`, 2.846),
aceita `--seed` e grava `MANIFESTO_TREINO.json` em `--checkpoints` ao iniciar e ao lado do LoRA ao
terminar: sha256 e nº de linhas de treino, validação e `DB/questoes.db`, commit do git, versões de
unsloth/trl/transformers/peft/torch/bitsandbytes, GPU, hiperparâmetros, seed, nº de exemplos
truncados por `MAX_SEQ_LENGTH` e melhor checkpoint. Código em `src/manifesto_treino.py`
(`tests/test_manifesto_treino.py`). README e CLAUDE.md atualizados.

**1.3 G1* e G7* agregados (feito, regra revisada DEPOIS de v7c e v7d reprovarem).** Em
`src/promover_checkpoint.py`, perfil multiseed: G1 e G7 deixam de reprovar por 1 item numa
rodada de 30 (ou de 7 no 5º ano). G1*: cada uma das 7 flags precisa de >= 99% nos pares somados
e não pode piorar de forma significativa (McNemar). G7*: por ano de risco, >= 90% de estrutura
ok nos pares somados e sem piora significativa. Vale só quando o relatório traz as flags e o ano
por item; senão segue a regra antiga. Reaplicado aos relatórios reais, sem sobrescrever vereditos:

| Candidato | Veredito com a regra nova | Gates que reprovam |
|---|---|---|
| v7b | PROMOVIDO (igual ao anterior) | nenhum |
| v7c | NÃO PROMOVIDO | G1* (alternativas_distintas 100 -> 98,7%, 2 itens), G2*, G3* |
| v7d | NÃO PROMOVIDO | G2*, G3* |

A mudança não aprova v7c nem v7d: G2* e G3* continuam reprovando. Ela só retira G1/G7 como causa
de reprovação por um item. `tests/test_gate_multiseed.py` cobre o caso real.

**1.4 Regime de prompt (feito no gate; medição pendente).** O gate multiseed passa a abortar se
baseline e candidato foram medidos em regimes diferentes (com e sem `--sufixo`), imprime o regime
no veredito e grava `regime_prompt` no JSON. Os vereditos v6, v7b, v7c e v7d foram todos
`sem_sufixo`, que é o regime de cerca de 10% do treino; o app e 90% do treino usam o sufixo.
Para medir no regime do app é preciso reavaliar o baseline e o candidato com `--sufixo`
(5 rodadas cada); isso é compute de GPU e fica para a Fase 2.

**1.2 `val.jsonl` não é intercambiável com o congelado (resolvido sem tocar em dados).**
`data/val.jsonl` e `data/val_frozen_v1.jsonl` têm os mesmos 30 `codigo_item` em ordem diferente, e
`val.jsonl` tem `meta.lote` a mais (sha256 distintos; `baseline_v1/val.jsonl` é idêntico ao
congelado). Como a seed de geração vem do índice da linha, os arquivos não se substituem.
Decisão: em vez de regerar `val.jsonl` (que exigiria trocar o hash em `ENTRADAS_IMUTAVEIS.json`),
o padrão de `tests/test_model.py` e `src/evaluate.py` passou a ser `val_frozen_v1.jsonl`;
`val.jsonl` fica só como histórico. `tests/test_val_congelado_padrao.py` falha se algum script de
avaliação voltar a usá-lo como padrão. CLAUDE.md corrigido. Nenhum `val*.jsonl` foi alterado.

## Fase 2: treinos com seeds 42, 43 e 44 concluídos (2026-10-14)

Mesma base (`train_curado_v7b.jsonl`, sha256 `82e7ea18…`, 2.846 linhas nos três manifestos), mesma
config, `--val data/val_frozen_v1.jsonl`. Manifestos em `outputs/lora_v7b_s{42,43,44}/MANIFESTO_TREINO.json`.

| Seed | eval_loss ép.1 / ép.2 / ép.3 | Melhor checkpoint |
|---|---|---|
| 42 | 0,6807 / 0,6804 / 0,6974 | ép. 2 (356) |
| 43 | 0,6769 / 0,6789 / 0,6954 | **ép. 1 (178)** |
| 44 | 0,6812 / 0,6807 / 0,6998 | ép. 2 (356) |

Achados: (1) a seed 43 escolheu a época 1; nas cinco runs anteriores o melhor era sempre a época 2.
(2) Nas três seeds a diferença entre as épocas 1 e 2 é de 0,0003 a 0,0020, então o `eval_loss` em 30
itens não discrimina entre elas e a escolha do checkpoint é mais uma fonte de variância.
(3) Os `eval_loss` desta rodada não são comparáveis aos de v7b/v7c/v7d: aqueles usaram `val.jsonl`
(outra ordem de itens, outra composição de batch). (4) Os três manifestos registram árvore de
`src/` e `tests/` com alterações não commitadas (as da Fase 1). Avaliação e export:
`fase2_avaliar_seeds.sh`.

### Fase 2: avaliação da seed 43 (2026-10-14); seeds 44 e 42 NÃO avaliadas

`fase2_avaliar_seeds.sh` abortou antes do export das seeds 44 e 42 pela guarda de disco (9 GB livres,
export pede 9,5 GB). Só a seed 43 tem GGUF e vereditos nos dois regimes. Baseline com sufixo medido
(`eval_baseline_suf_r{0..4}_gguf.json`). Vereditos: `VEREDITO_v7b_s43_{sem,com}_sufixo_multiseed_n150.json`.

| | sem sufixo (histórico) | com sufixo (regime do app) |
|---|---|---|
| Veredito | NÃO PROMOVIDO: G1*, G5, G8 | NÃO PROMOVIDO: G2*, G8 |
| Falhas pós best-of-N, baseline -> cand. | 1 -> 7 (4,7%), piorou 7 / melhorou 1, p=0,070 | 6 -> 8 (5,3%), piorou 7 / melhorou 5, p=0,774 |
| Consistência (verificáveis) | 98,5% (68) -> 94,3% (87), p=0,219 | 94,0% (67) -> 91,7% (84), p=0,508 |
| Viés de gabarito (letra mais frequente) | 40,0% -> 41,3% | 46,7% -> 32,0% |
| G1* / G7* | G1* falha (alternativas_distintas 98,7%, 2 itens); G7* ok | ambos ok |

Leituras e ressalvas (uma seed apenas, não conclusivo):
- O baseline com sufixo já falha 4,0% (6/150) contra 0,7% sem sufixo: o regime do app é mais difícil
  para os dois. O G2* reprova o candidato com sufixo só pelo teto absoluto de 5% (8/150 = 5,3%);
  McNemar não distingue os dois (p=0,774).
- G5 reprovou numa rodada sem sufixo (aderência 100 -> 93,3%, 2 itens de 30); agregado 97,3%.
  Mesmo defeito de resolução já tratado em G1/G7; NÃO alterado aqui para não mudar critério depois
  de ver resultados novos. Decisão fica para depois das 3 seeds, aplicada a todas.
- G8 (desconsiderado por ordem do usuário): a velocidade do candidato cai de 26,4 para 20,0 tok/s
  dentro da mesma avaliação com sufixo (rodadas 2 a 4), com o mesmo GGUF; sinal de estrangulamento
  térmico ou carga concorrente na GPU, não do modelo.
- Piso de ruído: ainda não estimável. Falta avaliar seeds 44 e 42 (e compará-las com v7b/v7d).

### Fase 2: seeds 44 e 42 + amostra humana cega (2026-10-14)

Seeds 44 e 42 exportadas e avaliadas nos dois regimes (disco liberado antes). Vereditos:
`VEREDITO_v7b_s{42,43,44}_{sem,com}_sufixo_multiseed_n150.json`.

| Seed | sem sufixo | com sufixo (app) |
|---|---|---|
| 42 | NÃO: G2* (1->12/150, p=0,003), G3* | NÃO: G2* (6->15/150, 10%), G3* (p=0,031) |
| 43 | NÃO: G1*, G5, G8 | NÃO: G2* (6->8/150, 5,3%), G8 |
| 44 | NÃO: G1*, G2* (1->10/150, p=0,012) | PROMOVIDO (G2* 6->5/150; G3* 94,0->93,8%) |

1 de 6 vereditos passou. Mesma configuração, vereditos opostos entre seeds e entre regimes: o ruído
de treino é maior que o efeito medido pelo gate. Ganho consistente nas três seeds: viés de letra
(letra mais frequente 40-47% -> 28-32%) e "E = nenhuma das anteriores" (~60% -> 4-8%, só observação).

**Amostra humana cega** (`outputs/amostra_humana/`, seed 2026, n=40, 20 por origem, 8 por unidade,
regime com sufixo, baseline x v7b s44; `apuracao.json`). Professor: valida_geral = s.
- Geral 8/40 (20%) IC95 [10,5; 34,8]. Baseline 3/20 (15%) [5,2; 36,0]; candidato 5/20 (25%) [11,2; 46,9].
  ICs se sobrepõem: sem diferença demonstrada na validade.
- Correção matemática: baseline 45%, **candidato 25%**. Dados suficientes 70% -> 95%; resposta única
  80% -> 90%; dificuldade 95% -> 100%; contexto plausível 70% -> 65%; aderência 95% = 95%.
- Geometria e probabilidade/estatística: 0/8 válidas nos dois braços.
- Verificador automático x professor (correção matemática = s): candidato com status `ok` 4/13 (31%),
  baseline com `ok` 6/8 (75%). O `ok` do verificador não garante a correção no candidato: 9 dos 13
  itens `ok` foram reprovados pelo professor (gabarito x resolução divergentes, valor correto fora
  das alternativas, dados contraditórios, resolução que contradiz o enunciado). Os 7 itens do
  candidato não verificáveis: 1 correto.
- Ressalvas: n=20 por braço; amostra não estratificada por ano (maioria 9º); o baseline humano aqui
  (15%) é bem menor que a medida anterior do projeto (50%), com outra amostra e outro avaliador.

**Decisão:** v7b NÃO é adotado como superior. Gate e humano discordam; o critério de aceitação é o
humano. A queda em correção matemática do candidato é o sinal vermelho. Não gastar mais seeds para
decidir; a validade baixa nos dois braços indica problema de geração que o v7b não resolve.
Próximo: analisar os 15 itens do candidato com correção = n (arquivo `ANALISE_ERROS_CORRECAO.md`
em `outputs/amostra_humana/`) antes de desenhar a Fase 3 (dados v8).

### Fase 3.0 (2026-10-14): aviso `gabarito_so_em_passo_intermediario`

`schema_utils.avisos_gabarito` (novo): aviso NÃO BLOQUEANTE quando o veredito é `valor_bate` mas o
resultado final da resolução não é o valor do gabarito. Não muda status nem veredito; entra em
`avisos_consistencia` do resultado de `generate_validated` e em `detalhe_consistencia`.
`avisos_consistencia` (contado por `triar_resolucoes` como `dado_inventado`) não foi alterado.
Testes: `tests/test_furos_verificador_amostra_humana.py`. Suíte: 853 passam; as 3 falhas são as
mesmas já conhecidas (Fase 0.2), sem relação.
Atinge 66 itens de `train_curado_v7b` (2,2%); 30 sorteados (seed 7) para revisão em
`outputs/revisao_aviso_intermediario/amostra_30.csv` (coluna `gabarito_certo(s/n)`). Lista completa
em `todos_os_atingidos.json`. Resultado da revisão: pendente.

### Teto da base (2026-10-14): `src/amostra_humana_base.py`

Pergunta: que fração dos itens de `train_curado_v7b` o professor aprova com a rubrica usada nos
modelos? Folha cega com 48 itens, 12 por origem (banco real MT, lote L2, injeção LLM, destilado
`DIST-`), em `outputs/amostra_base_v7b/` (folha, `chave_oculta.json`, instruções). Fora da amostra:
`REFORCO-` e `SINT-` (correção por construção). Apuração: `src/amostra_humana.py apurar --folha
<preenchida> --chave outputs/amostra_base_v7b/chave_oculta.json`; a taxa da base inteira pondera as
origens pelo `pool_por_origem` (L2 444, banco 232, destilado 610, injeção 1426). Ressalva: o lote L2
só tem unidade Números, então a estimativa por unidade usa quase só as outras três origens.
Resultado: pendente.

**Resultado (professor, 2026-10-14)**, `outputs/amostra_base_v7b/folha_preenchida.csv`, `apuracao.json`:
validade 26/48 (54,2%, IC95 [40,3; 67,4]); correção matemática 48/48 (100%). Por origem: L2 9/12
(75%), banco 7/12 (58%), injeção 6/12 (50%), destilado 4/12 (33%). Por unidade: Geometria 1/6, Álgebra
4/9, Números 13/21, Grandezas 4/6, Prob./Estat. 4/6. Das 22 reprovadas, 15 têm `dificuldade_adequada=n`
(dificuldade rotulada acima do conteúdo, muito em itens "Difícil" de adição/subtração simples) e 5
`aderencia_ano_habilidade=n`. **Nenhuma reprovação é por conta errada.** Conclusão: o teto da base em
correção é ~100%; o limite de validade (54%) vem do rótulo de dificuldade, e não da matemática.
Comparação com os modelos (mesma rubrica, outras amostras): correção matemática baseline 45-70%,
candidato 25-60%; base 100%. O modelo erra a matemática que a base acerta: o gargalo de correção está no
modelo/geração, não nos dados. Ressalvas: n=48; a amostra tem muito 1º/2º ano (21 Números) e o L2 só
cobre Números.
Avaliador LLM externo (folha colada antes, não é o professor): 44/48 (91,7%); concordou com o
professor em 30 de 48 e aprovou 18 itens que o professor reprovou, quase todos por dificuldade. Não serve como juiz
da rubrica de dificuldade.

### Rótulo de dificuldade: validação humana (2026-10-14)

Decisão do responsável: a referência do rótulo de dificuldade vem do professor, não de um método
automático. `src/amostra_dificuldade.py` (`gerar` / `apurar`, testes em `tests/test_amostra_dificuldade.py`)
sorteia 108 itens de `train_curado_v7b` (6 por célula ano x rótulo atual; sem `REFORCO-`/`SINT-`) e o
professor classifica F/M/D às cegas (`outputs/amostra_dificuldade/`). A apuração dá a matriz, a taxa de
rótulo inflado/baixo por ano e origem e o kappa ponderado. Só depois disso se decide como corrigir a base
(regra, Jev validado contra esta folha, ou revisão manual). Itens reais do banco continuam exigindo
decisão humana. **Resultado (professor, 2026-10-14, n=108, `outputs/amostra_dificuldade/apuracao.json`):**
rótulo atual = professor em 44/108 (40,7%, IC95 [31,9; 50,2]); kappa ponderado 0,20. Rótulo mais alto
que o do professor em 57 (52,8%); mais baixo em 7 (6,5%). **O professor não classificou nenhum item como
Difícil** (59 Fácil, 49 Moderado). Dos 36 itens rotulados "Difícil", 9 viraram F e 27 M; dos 36 "Moderado",
21 F e 15 M; dos 36 "Fácil", 29 F e 7 M. Inflação por origem: injeção 27/46, L2 15/31, destilado 15/30; só 1
item do banco real na amostra. Achado extra: `INJ-5-H01-D-00238` (D4861) tem enunciado/gabarito
inconsistentes ("seiscentos e cinquenta e sete milésimos" lido como 650,007); 1 erro de conta em 108.
Leitura: o rótulo da base está uma casa acima do critério do professor, e a escala dele não usa Difícil
nos itens gerados; antes de relabelar é preciso saber se isso é inflação dos geradores ou régua mais
rígida do professor (teste âncora com itens reais do banco rotulados Difícil).

Teste-âncora (2026-10-14): `src/amostra_dificuldade.py gerar --origens banco --n 27` →
`outputs/amostra_dificuldade_ancora/` (27 itens reais do banco SAEB, 3 por célula ano x rótulo em
2º/5º/9º). Pergunta: o professor usa "Difícil" nos itens reais rotulados Difícil? Sim => a inflação está
nos geradores; não => a régua dele é mais rígida e o remapeamento deve partir dela. Resultado: pendente.

**Resultado do teste-âncora (professor, 2026-10-14, 27 itens reais do banco SAEB,
`outputs/amostra_dificuldade_ancora/apuracao.json`):** rótulo SAEB = professor em 12/27 (44,4%, IC95
[27,6; 62,7]), kappa ponderado 0,24; rótulo mais alto que o do professor em 15 (55,6%), mais baixo em 0.
Dos 9 itens reais rotulados Difícil, o professor marcou 4 F, 4 M e 1 D; dos 9 Moderado, 7 F e 2 M.
A inflação aparente é a MESMA nos itens reais (55,6%) e nos gerados (52,8%; kappa 0,20): o professor
usa uma régua uma casa abaixo do rótulo empírico do SAEB. Conclusão: o rótulo da base não está "errado"
por causa dos geradores; os dois critérios medem coisas diferentes (desempenho dos alunos x dificuldade
percebida). **Decisão proposta: não relabelar a base** e não exigir `dificuldade_adequada` em `valida_geral`
nas próximas amostras (manter a coluna como informativa).
Efeito na validade (recalculado, mesma rubrica sem a dificuldade): modelos (v7b s44 e baseline) 8/40, igual
com ou sem dificuldade (baseline 3/20, candidato 5/20); base 40/48 (83%): L2 11/12, injeção 10/12,
destilado 10/12, banco 9/12. A diferença base x modelo (83% x 20%) não vem do rótulo de dificuldade.

### Decisões 2026-10-14

**Critério NÚCLEO adotado como leitura principal.** `valida_nucleo` = `correcao_matematica` ∧
`resposta_unica` ∧ `dados_suficientes` (todas `s`). `dificuldade_adequada`, `aderencia_ano_habilidade`
e `contexto_plausivel` continuam na folha e na apuração, mas são informativos — não decidem
`valida_nucleo`. Implementado em `src/amostra_humana.py` (constante `NUCLEO`, campo
`validade_nucleo` na apuração), testado em `tests/test_validade_nucleo.py`.

Recálculo com o critério núcleo:
- Amostra de modelo (v7b s44 + baseline, 40 itens): `valida_geral` 20% → `valida_nucleo` **25%**
  (baseline 25%, candidato 25%).
- Amostra de base (4 origens, 48 itens): `valida_geral` 54,2% → `valida_nucleo` **97,9%**
  (L2 100%, banco 100%, injeção 100%, destilado 91,7%).

**Correção do item D4861 (INJ-5-H01-D-00238).** Enunciado "seiscentos e cinquenta inteiros e sete
milésimos de quilograma" estava ambíguo (650,007 vs 650,7); resolução dizia "650." sem casas. Correção:
resolução atualizada para `650` inteiro, `meta.edicao_pos_v7b` adicionado. Escrito em
`data/train_curado_v7b_fix1.jsonl` (sha256 registrado em `outputs/melhoria_v7/EDICAO_D4861.json`);
`train_curado_v7b.jsonl` original não foi tocado.

### Fase 3.1: teste de teto com Qwen3-4B zero-shot (resultado, 2026-10-14)

Objetivo: isolar se o gargalo de qualidade está nos dados de treino ou no tamanho do modelo
(1.7B). Gerado 1 lote de 30 questões com `Qwen/Qwen3-4B-GGUF` Q4_K_M (2,4 GB, **sem fine-tuning**,
zero-shot), mesmos 30 prompts de `val_frozen_v1.jsonl` com `--sufixo`, seed-rodada 1, grammar GBNF
idêntica. `outputs/qwen3_4b/eval_report_4b_s1.json`: 0 falhas de estrutura/schema, 100%
`consistencia_resposta_correta_pct` nos 8 itens verificáveis automaticamente, mas viés forte de
letra A (60%, 18/30) e 14,7 tok/s (mais lento que o 1.7B, esperado).

**Amostra humana cega** (`outputs/amostra_qwen4b/`, 40 itens: 20 fine-tune v7b-s44-suf-r1 + 20
Qwen3-4B zero-shot, mesma rubrica, critério núcleo):

| | correcao_matematica | validade_nucleo |
|---|---|---|
| fine-tune 1.7B (v7b s44) | 50,0% (10/20) | 50,0% IC95 [29,9; 70,1] |
| Qwen3-4B zero-shot | 70,0% (14/20) | 70,0% IC95 [48,1; 85,5] |

ICs se sobrepõem (n=20/braço), mas a direção é consistente: um 4B **sem nenhum fine-tuning** supera
o 1.7B fine-tunado em correção matemática e validade núcleo. Geometria: 4B 2/2, fine-tune 0/2 (n
mínimo, mas no mesmo sentido das amostras anteriores). O 4B perde em `aderencia_ano_habilidade`
(70% x 85%) e `dificuldade_adequada` (45% x 60%) — não foi condicionado ao formato SAEB.

**Conclusão provisória:** combinada com o teto da base medido em núcleo (~98%), a evidência aponta
que o gargalo de qualidade **não é majoritariamente a curadoria dos dados de treino** — é a
capacidade do modelo de 1.7B em explorar esses dados durante a geração. Um modelo maior, mesmo
zero-shot, já supera o fine-tune atual na dimensão que mais importa (correção matemática). Isso
não decide a escolha de modelo final — publicação segue pausada — mas redireciona o esforço:
antes de investir mais em curadoria de dados (Fase 3), vale considerar fine-tuning de um modelo
maior (ex.: Qwen3-4B) ou destilação mais agressiva do 4B/professor maior para o 1.7B.

### Fase 3.1: teste de teto com Qwen maior (em andamento)

Objetivo: medir o teto de geração zero-shot de um Qwen3-4B Q4_K_M (não fine-tunado) nos mesmos
30 prompts do val_frozen_v1.jsonl com `--sufixo`. Se a taxa de `valida_nucleo` superar o modelo
fine-tunado de 1.7B, o gargalo está no tamanho do modelo e não nos dados de treino.

### Fase 3.2: Qwen3-4B fine-tunado x zero-shot x 1.7B (amostra humana, 2026-10-15)

Treino: Qwen3-4B QLoRA, mesma receita do v7b s44 (seed 44, r=16, LR 2e-4, batch efetivo 16 = 1x16),
corpus `train_curado_v7b_fix1`. Avaliação: adaptador convertido para GGUF e aplicado sobre o Q4_K_M
oficial (`--lora` em `tests/test_model.py`), `--sufixo`, rodada 1 de cada braço (mesmos 30 prompts).
Folha cega de 60 itens, 20 por braço (`outputs/amostra_3bracos/`, `apuracao.json`).

| Braço | validade núcleo | correcao_matematica | valida_geral |
|---|---|---|---|
| 1.7B fine-tunado (v7b s44) | 7/20 = 35% IC95 [18; 57] | 40% | 5% |
| 4B fine-tunado | 13/20 = 65% IC95 [43; 82] | 75% | 10% |
| 4B zero-shot | 13/20 = 65% IC95 [43; 82] | 65% | 35% |

Leitura: o 4B (com ou sem fine-tuning) fica acima do 1.7B em núcleo (65% x 35%), mas os ICs ainda
se sobrepõem levemente (n=20/braço). O fine-tuning do 4B NÃO melhorou o núcleo sobre o zero-shot (empate
13/20); melhorou correção matemática (75% x 65%, dentro do ruído) e piorou valida_geral (10% x 35%),
puxado por aderência e dificuldade. Nesta amostra o 1.7B caiu de 50% (amostra anterior) para 35%:
variação de amostragem, o que mostra o tamanho do ruído com n=20.

**Gate multiseed 4B fine-tunado (candidato) x 4B zero-shot (baseline)** (`outputs/relatorios/VEREDITO_4b_ft_vs_zs_multiseed.json`,
3 rodadas, 90 pares): PROMOVIDO, todos os gates bloqueantes passaram. Ganhos do candidato: viés de letra
52,9% -> 40,0%, falhas pós best-of-N 5 -> 3 (p=0,727), cobertura `ok` do verificador 30% -> 49%. G2*/G3* sem
diferença significativa (p=0,727 e 0,625). O gate só garante ausência de regressão: não contradiz o empate
em núcleo da amostra humana (65% x 65%), e o ganho do candidato é de formato (letra, verificabilidade),
não de correção medida pelo professor. Publicação segue pausada.

**Gate multiseed 4B fine-tunado (candidato) x baseline_v1 1.7B (baseline)** (`outputs/relatorios/VEREDITO_4b_ft_vs_baseline_v1_multiseed.json`,
rodadas 0-2 com sufixo, 90 pares): veredito formal NÃO PROMOVIDO, reprovado só em G8 (velocidade: 27 -> 13,8 tok/s,
-48% a -49% nas 3 rodadas, esperado para um modelo ~2,3x maior). G1*, G2* (3/90 -> 3/90, p=1,0), G3* (p=1,0), G4, G5, G6*
(44,4% -> 40,0%), G7* e G9 passaram. G8 está desconsiderado por decisão do responsável, então o único gate
reprovado é justamente o custo de execução no dispositivo. Observação não bloqueante: "E = Nenhuma das anteriores"
em 58,9% dos itens do baseline contra 2,2% do candidato. O gate não detecta melhora de correção; a evidência
disso vem só da amostra humana (núcleo 65% x 35% contra o 1.7B v7b s44, ICs ainda sobrepostos).
