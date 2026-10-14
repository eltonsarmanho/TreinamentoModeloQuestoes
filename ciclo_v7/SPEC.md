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

**1.2 `val.jsonl` idêntico ao congelado (pendente de autorização).** Confirmado: `data/val.jsonl`
e `data/val_frozen_v1.jsonl` têm os mesmos 30 `codigo_item`, em ordem diferente, e `val.jsonl`
tem `meta.lote` a mais (sha256 distintos; `baseline_v1/val.jsonl` é idêntico ao congelado).
Como a seed de geração vem do índice da linha, os dois arquivos não são intercambiáveis. O gate
já aborta se os nomes diferem. Falta decidir se `val.jsonl` é regerado como cópia do congelado.
