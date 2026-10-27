# Ciclo v8: resolver antes de escrever as alternativas (Qwen3-4B)

Data: 2026-10-19. Origem da decisão: consulta ao Jev (TypeSafe System One, `typesafe/jev-1.13` via
OpenRouter) sobre seis caminhos. Resultado em `outputs/melhoria_v8/jev_decisao_resultado.json`
(cópia do scratchpad da sessão): a opção A foi o melhor primeiro passo nas 4 ordens de opções testadas
(probabilidade média 0,76, confiança 0,68 a 0,75). A medição (opção F) foi apontada como pré-requisito
(0,94) e roda em paralelo. Hiperparâmetros: sem evidência de efeito acima do ruído entre seeds (0,05).

## Hipótese

O erro mais frequente que o professor aponta é "a resolução calcula o valor certo, mas ele não está
entre as cinco alternativas, ou a letra não bate". Os 2.948 alvos de treino emitem as chaves na ordem
`enunciado, alternativas, resolucao_passo_a_passo, resposta_correta, difficulty`: as cinco opções são
escritas antes de a conta existir. Trocar para `enunciado, resolucao_passo_a_passo, alternativas,
resposta_correta, difficulty` põe o valor calculado antes das opções.

O contrato do app não muda (mesmas chaves, 5 alternativas A-E, leitura por nome de chave). O que muda:
o alvo de treino e a gramática enviada ao llama.cpp.

## O que foi feito

- `src/construir_v8.py` (determinístico, sem API) gera `data/train_curado_v8.jsonl` a partir de
  `data/train_curado_v7b_fix1.jsonl` (entrada intacta) e `data/holdout_loss_v8.jsonl` a partir de
  `data/val_frozen_v1.jsonl` (só para o eval_loss do treino; o gate continua usando o val congelado).
  Relatório em `outputs/melhoria_v8/relatorio_v8.json`.
- Transformações: ordem das chaves; 47 exemplos multi-questão divididos em um exemplo por questão
  (formato real do app); frases da resolução que citam "alternativa", letra ou lista A) B) saem, porque
  na nova ordem as alternativas ainda não existem. "3 opções de camisa" num problema de contagem fica.
- Trava: depois da limpeza, `check_consistency_detalhado` não pode mudar de veredito. Questão que
  mudaria sai do corpus (30 injetadas). Item real do banco/L2 nunca é removido (política do ciclo v7);
  2 itens reais ficaram reordenados, com a resolução original.
- Resultado: 2.948 → 2.918 questões, uma por exemplo. Enunciado, alternativas, gabarito e dificuldade
  idênticos aos de origem em 100%; letras do gabarito balanceadas (19% a 21% cada); 0 contaminação com
  o val congelado; 5 resoluções ainda citam alternativas (2 reais mantidas + 3 casos de borda).
- `grammars/questao_v8.gbnf`: só a regra `questao` muda; o resto é idêntico (testado).
- `tests/test_model.py --gramatica` e `src/gerar_folha_professor.py --gramatica`; o relatório de
  avaliação passou a gravar `gramatica` (arquivo e sha256).
- `tests/test_ordem_v8.py`: 17 testes.

## Treino

Mesma receita do 4B v7b s44 (r=16, alpha=32, LR 2e-4, cosseno, 3 épocas, lote efetivo 16, seed 44,
`max_seq_length` 1024); mudam só os dados e a ordem. A confusão é aceita de propósito: ordem e limpeza
vêm no mesmo treino, então um ganho não se atribui a uma delas. Controle (mesma limpeza, ordem antiga)
custa outras 2,5 h de GPU e só vale a pena se o resultado de v8 for ambíguo.

    venv/bin/python src/train.py --train data/train_curado_v8.jsonl --val data/holdout_loss_v8.jsonl \
      --out outputs/lora_4b_v8_s44 --checkpoints outputs/checkpoints_4b_v8_s44 \
      --base-model unsloth/Qwen3-4B-unsloth-bnb-4bit --batch-size 1 --grad-accum 16 --seed 44

O eval_loss do v8 NÃO é comparável ao do v7b (outro holdout, outra ordem).

## Avaliação (depois do treino)

1. Converter o adaptador para GGUF e aplicar sobre o Q4_K_M oficial (`--lora`), como no v7b.
2. Verificar o template (P3.4) antes do gate.
3. Gate multiseed com `--sufixo`, 5 rodadas. Candidato com `--gramatica grammars/questao_v8.gbnf`;
   baseline (4B v7b s44 e 4B zero-shot) com a gramática antiga. Cada braço usa a gramática do seu treino.
4. Amostra humana pareada. Critério de decisão: núcleo (correção ∧ resposta única ∧ dados suficientes),
   n ≥ 100 por braço, só depois da medição da opção F.
5. Métrica de mecanismo, sem professor: taxa de itens em que o valor final da resolução está entre as
   alternativas (`fora_das_alternativas` do verificador), v8 contra v7b nas mesmas seeds.

## Fora do escopo desta rodada (e por quê)

- Remover a alternativa E fixa "Nenhuma das anteriores": o contrato do app é 5 alternativas "sem
  exceção". Exigiria trocar a E por um distrator real (gerado e verificado) e a decisão é do responsável.
- Reescrever as resoluções curtas: usa API paga (`melhorar_resolucoes.py` só em dry-run).
- RFT/ORPO (opção B), contrato do app (C), inferência (D), hiperparâmetros (E): ver ordem do Jev.

## Implantação

Um modelo v8 só funciona com a gramática v8. O app envia a gramática inline (Doc/GUIA_SUFIXO_APP.md):
publicar o v8 exige trocar a string da gramática no app junto com o modelo. Nada disso foi alterado.

## Resultado da medição F (2026-10-19)

Folha cega com 222 itens do 4B v7b (`outputs/melhoria_v8/F/folha_lote_01_preenchida.csv`, apuração em
`apuracao_parcial.json`). Todos preenchidos, todos na chave.

- Núcleo (correção ∧ resposta única ∧ dados suficientes): **153/222 = 68,9%**, IC95 [62,5; 74,6].
  Com as 30 avaliações antigas do mesmo modelo: 171/252 = 67,9%, IC95 [61,9; 73,3].
- Por ano: 9º 54% (42/78), 5º 70% (48/69), 2º 82% (55/67). Por dificuldade: Difícil 54%, Moderado 74%, Fácil 78%.
- Por unidade (núcleo): Geometria 54% (26/48), Álgebra 59% (22/37), Prob./Estat. 72% (31/43), Números 76% (34/45),
  Grandezas 82% (40/49). Correção 2026-10-23: a versão anterior trazia 42/51/56/58/67%, que são `valida_geral`
  (bloco `por_unidade_e_origem` de `amostra_humana.py`, que só mede validade geral), não núcleo.
- Verificador × professor: `ok` 83% (81/98) válidos; `nao_verificavel` 58% (72/124).
- Dimensões: correção 75,7%, resposta única 75,2%, dados suficientes 86,9%.

Causa principal dos 69 reprovados (classificação minha sobre o texto do professor, uma causa por item):

| Causa | Itens | % |
|---|---|---|
| Enunciado mal posto: dados contraditórios, impossíveis, insuficientes ou inventados | 24 | 35 |
| Conta ou conceito errado na resolução ou no gabarito | 19 | 28 |
| Mais de uma alternativa correta (frações equivalentes, ambiguidade) | 14 | 20 |
| Resolução chega ao valor certo, mas as alternativas não o contêm ou a letra está errada | 12 | 17 |

**Consequência para o A:** ele só ataca a última linha (12/222 = 5,4 pontos). Teto de ganho em núcleo
de cerca de 5 pontos, não mais. É barato (um treino) e continua valendo, mas não é a alavanca principal.

### Juiz Jev, validação fora da amostra

Regra congelada antes de ver estes itens: reprovar se `resposta_unica` < 0,9 (`src/f_jev_validar.py`,
`validacao_jev_fora_da_amostra.json`; 222 requisições, US$ 0,0098).

| | Resultado |
|---|---|
| κ contra o professor (valida_nucleo) | 0,795, IC95 [0,703; 0,876] |
| Precisão do reprovado / recall dos inválidos | 90,3% / 81,2% |
| Válidos perdidos | 6 de 153 (3,9%) |
| Núcleo entre os aceitos | 91,9%, contra 68,9% sem filtro |
| AUC de `resposta_unica` | 0,956 [0,926; 0,982] |

Meta κ ≥ 0,6 atingida na amostra nova. Limites: o Jev roda por API, não no celular offline; vale como
filtro de dados, juiz de gate e proxy de avaliação, não como guarda em produção. `referencia_a_figura_ausente`
fica perto do acaso (AUC 0,53).

### Segunda consulta ao Jev (`outputs/melhoria_v8/jev_decisao_2_resultado.json`)

Com a evidência acima, o primeiro passo escolhido mudou para **B** nas 4 ordens (probabilidade média B 0,67,
A 0,30; confiança 0,47 a 0,71). A produto "ataca falha × mensurável" ainda dá A 0,26 contra B 0,22, então a
escolha é sensível à pergunta; F caiu a 0 por já estar feita. Leitura: A e B são complementares. A é um treino
de 2,5 h com teto de 5 pontos; B ataca as quatro causas, pois o juiz filtra enunciado mal posto, resposta
múltipla e conta errada.

## Auditoria das etapas B (2026-10-22)

- **B2 (filtro Jev) teve dois problemas.** (1) O OpenRouter ficou sem crédito no meio: 606 das 2.067 amostras
  (prompts 289 a 399, corte temporal, não aleatório) nunca foram julgadas e ficaram fora. Todas as 79
  habilidades ainda têm extras. (2) **Erro meu:** os extras entraram sem `permutar_lote`; herdaram o viés de letra
  do gerador (A 34,5%, E 7,7%). O B3 treinado em `train_curado_v8_rft.jsonl` saiu com **44,1% de gabaritos em A**
  (v7b: 33,4%) nas 800 amostras pareadas; o teto do G6 é 40%. Corrigido em `src/rft_filtrar.py`
  (`permutar_lote`, seed 20261022, teste `test_extras_saem_com_letra_balanceada`). Corpus corrigido:
  `data/train_curado_v8_rft_bal.jsonl` (3.793 linhas, letras 20% cada, verificador inalterado: 478 ok / 397 n.v.).
  O corpus já treinado NÃO foi sobrescrito (sha `dfed9cee…` confere com o manifesto).
- **B3 (treino)**: adaptador salvo = checkpoint-238 (época 1), sha `3a8bbe05…`; GGUF convertido em 21/10 19:38.
- **Comparação pareada sem API (800 prompts/seeds iguais):** v8rft entrega 750/800 contra 698/800 do v7b
  (só v7b 44, só v8rft 96, McNemar p < 0,0001); verificador `ok` 374 contra 310 (p < 0,0001). Ressalva: o v8rft
  foi treinado em itens filtrados pelo verificador, então parte do ganho é por construção.
- **Gate r0 do v8rft** (`eval_v8rft_suf_r0.json`): 1 JSON inválido em 30 (MT90119DH24MT, 9º) e 1 par de
  alternativas repetidas; 2 falhas contra 0 do v7b r0. G1* exige ≥ 99% nos 150 pares: no máximo 1 falha de JSON
  nas 5 rodadas. Observação: "E = Nenhuma" subiu para 26,1% (v7b 12,8%) sem que os extras tivessem isso (12,5%).
- **Etapa 2 (template):** sem arquivo de evidência; a saída v8 válida no gate (96,7% JSON, ordem v8) é evidência
  indireta. Falta o teste de equivalência do `<think>` vazio (P3.4), ainda não escrito.
- **Juiz Maritaca:** calibração em andamento (75 de 222 no cache, 2 ininterpretáveis, R$ 1,20 até aqui, mediana
  408 tokens de saída, abaixo da estimativa de 700).

## Resultados 2026-10-22: juízes e gate

- **Juiz Maritaca calibrado** (`outputs/melhoria_v8/calibracao_maritaca.json`, 222 itens, R$ 5,36, 3 abstenções):
  κ 0,725 contra o professor; recall dos inválidos 90,9%; precisão do reprovado 74,1%; válidos perdidos 13,7%;
  núcleo entre aceitos 95,7% (sem filtro 69,9%). Passa a meta (κ ≥ 0,6). Mais rígido que o Jev (κ 0,80, 3,9%
  perdidos): serve como 2º juiz, de outra família, para o filtro RFT e para o gate.
- **Etapa 4 com Jev (créditos repostos), 800 pares:** v7b 517/800 = 64,6% [61,2; 67,9]; v8rft 589/800 = 73,6%
  [70,5; 76,6]; só v7b 110, só v8rft 182, McNemar p < 0,001. 9º ano 55% → 69%; 5º 67% → 78%. Ressalvas: o Jev
  foi o filtro do treino do v8rft (ganho em parte por construção) e o v8rft tem 44% de gabaritos em A (G6).
- **Gate v8rft:** as rodadas 1 a 4 foram gravadas sobre o mesmo arquivo; só a rodada 4 sobreviveu
  (renomeada para `eval_v8rft_suf_r4.json`: JSON 100%, 0 falhas, 16 ok). As rodadas 0 a 3 precisam ser refeitas
  com nomes distintos. Baseline v7b com sufixo só tem r0 a r2 (`outputs/qwen3_4b/eval_4b_ft_suf_r{0,1,2}.json`).

## Decisão do responsável sobre os gates (2026-10-22)

G5 e G8 deixam de bloquear no perfil multiseed e G9 sai da lista (`src/promover_checkpoint.py`,
`sem_g5_g8_g9`; `--gates-completos` restaura a regra antiga). Tomada ANTES de medir o v8rftb, então não é
revisão pós-reprovação; o responsável foi avisado da prática do projeto e decidiu assim. Efeito verificado:
o veredito 4B FT × 1.7B baseline_v1 (reprovado só por G8) passa a PROMOVIDO com a regra nova e continua
NÃO PROMOVIDO com `--gates-completos`. Testes: `TestDecisao20261022` em `tests/test_gate_multiseed.py`.
Bloqueantes restantes: G1*, G2*, G3*, G4, G6*, G7*.


## Gate do v8rftb e revisão do G1* (2026-10-23)

- Veredito original: NÃO PROMOVIDO só por G1* (`alternativas_distintas` 100% -> 98,9%, 1 item em 90,
  p=1,000). Item: rodada 1, item 0 (canteiros por cor), B = D, `status="falha"` (descartado em produção,
  já contado no G2* 3/90 -> 1/90). Guardado em `outputs/relatorios/VEREDITO_v8rftb_vs_v7b_multiseed_G1_todos_reprovado.json`.
- Revisão pós-reprovação, decisão do responsável: G1* conta só itens entregues. Item entregue com flag
  ruim continua reprovando (teste `TestG1SoEntregues`). `--gates-completos` reproduz o veredito antigo.
- Veredito novo: PROMOVIDO (`VEREDITO_v8rftb_vs_v7b_multiseed.json`). Ganhos: consistência 93,6 -> 100,
  verificáveis 47 -> 51, falhas 3 -> 1. G6* 40,0 -> 41,1% (limite 45%). G8 passou.
- Juízes (recalculados do cache, sem gasto): Jev 64,6% -> 73,4% (110 x 180, p<0,001); Maritaca
  60,2% -> 66,5% (400 pares, 72 x 97, p=0,065). Mesma direção.
- Observação: "Nenhuma das anteriores" 2,2% -> 32,2% no gate (corpus de treino tem ~21%). Vai para a
  revisão do professor na Etapa 8.

## Folha do professor, v8rftb (2026-10-23)

Folha cega de 78 itens (`outputs/melhoria_v8/folha_v8rftb/`; `gerar_folha_professor.py`, seed 20261101; apuração
oficial em `apuracao.json`, com `chave_enriquecida.json` porque a chave desse gerador não traz unidade/origem).

- Núcleo: **51/78 = 65,4%**, IC95 [54,3; 75,0]. v7b: 68,9% [62,5; 74,6]. Diferença −3,5 pontos, z = −0,57 (ruído).
  Meta de 80% não atingida. `valida_geral` 55,1%.
- Por ano: 2º 83% (19/23), 5º 59% (13/22), 9º 50% (13/26). Por unidade (núcleo): Geometria 62% (8/13), Grandezas 69% (9/13),
  Números 67% (22/33), Álgebra 56% (5/9), Prob./Estat. 70% (7/10). (Corrigido em 2026-10-23: antes 46/46/58%, que eram `valida_geral`.) Dimensões: correção 69,2%, resposta única 73,1%, dados suficientes 80,8%.
- Verificador × professor: `ok` 33/42 = 79%; `nao_verificavel` 18/36 = 50%. Letra do gabarito, "Nenhuma das anteriores"
  (59% contra 68%) e resolução longa (n=8) não separam com significância.
- Causas das 27 reprovadas (recontadas item a item em 2026-10-23; a estimativa anterior era ~11/~6/~4/resto): 9 situação mal definida ou
  dado que falta, 7 conta ou resolução errada, 6 mais de uma correta, 5 enunciado contraditório.
- O filtro determinístico atual (`check_consistency_detalhado` + `avisos_consistencia`) sinaliza 18 das 27 inválidas, mas 17 desses
  18 sinais são `sem_conta` (não verificável) e 18 válidas também são sinalizadas: não é filtro útil. Sinais reais (`resultado_na_cauda`,
  `gabarito_nda`, `sem_correspondencia`) pegam 4 das 27.
- **Leitura:** o ganho de +8,8 (Jev) e +6,3 (Maritaca) não apareceu no professor. O RFT filtrado por juiz de API não moveu o núcleo.
  Recomendação: não publicar o v8rftb; manter o v7b; adiar o ORPO (mesmos juízes filtrariam os pares); atacar enunciado incoerente
  com checagem numérica sem API e auditar com o professor uma amostra dos 875 extras do RFT.

## Auditoria dos extras do RFT (2026-10-23)

`src/auditar_extras_rft.py` gera `outputs/melhoria_v8/auditoria_extras_rft/`: 80 questões, 60 extras `rft_v8` (estratificados
por ano; 29 `ok`, 31 `nao_verificavel`) e 20 controles `injecao_saeb`, misturados às cegas (ids opacos, sem pista de origem na folha).
Pergunta: qual a taxa de núcleo dos 875 extras que o Jev aprovou? Os controles (base já medida em ~98%) calibram a régua do professor.
Critério de leitura: extras ≥ ~80% = o filtro funcionou e o gargalo é o treino; extras ≈ 65% ou menos = o Jev deixou passar o mesmo tipo
de erro e o corpus precisa de outra curadoria. Apuração: `amostra_humana.py apurar` com a chave desta pasta (traz `origem`).

### Resultado da auditoria dos extras (2026-10-23)

Apuração oficial (`outputs/melhoria_v8/auditoria_extras_rft/apuracao.json`). O professor devolveu `folha_preenchida(1).csv`
SEM a coluna `aderencia_ano_habilidade`; a apuração rodou numa cópia com essa coluna preenchida com "s" (só dimensão informativa,
fora do núcleo; a linha de aderência do relatório não vale). O arquivo dele não foi alterado.

- Extras do RFT (Jev aceitou): núcleo **53/60 = 88,3%**, IC95 [77,8; 94,2]. Controles da base: **20/20 = 100%** [83,9; 100].
  Fisher extras × controle p = 0,18 (sem diferença detectável com n=20).
- Mesmo professor, mesma rubrica: v8rftb 51/78 = 65,4% × extras 88,3% (Fisher p = 0,0025); v7b 68,9% × extras (p = 0,0028).
- Extras por status: `ok` 26/29 = 90%, `nao_verificavel` 27/31 = 87%. Geometria 4/8, Números 16/17, Álgebra 11/12.
- 7 extras reprovados: enunciado incoerente (3), resultado errado com resolução contraditória (2), mais de uma ou nenhuma
  correta (2). Os mesmos tipos das reprovações do modelo.
- **Leitura:** o corpus RFT é bom (~88%, perto da base) e o filtro do Jev funcionou como filtro de dados. O v8rftb produz 65%
  mesmo treinado em dados ~90-98% válidos: o gargalo é a capacidade de geração do modelo, não a curadoria. Isso fecha a hipótese de
  melhorar só pelos dados, consistente com a conclusão do ciclo v7 (teto da base ~98%).

## Experimento "gerar e conferir" (2026-10-23, preparado; execução do responsável)

`src/auto_conferencia.py` + `tests/test_auto_conferencia.py` (5 testes). Hipótese: se o modelo resolve a questão às cegas (só enunciado e
alternativas) e escolhe a letra do gabarito, a questão tende a ser válida; discordância = descartar. Medido sem treino nas 350 questões
rotuladas pelo professor e únicas de `outputs/melhoria_v8/F/rotulados.jsonl` (229 válidas, 65,4% de núcleo sem filtro). Solver: Qwen3-4B
Q4_K_M base, sem LoRA, temperatura 0. `--permutar` embaralha as alternativas (a "Nenhuma das ..." fica por último). Modos `direto` (~1 s) e
`cot` (~30 s). Saída em `outputs/melhoria_v8/auto_conferencia/`. Critério de leitura: núcleo entre as mantidas ≥ 80% com perda de válidas
≤ 30% justifica custo de ~2× tempo por questão no app; abaixo disso o ganho não compensa. O conjunto mistura saídas de vários modelos e
itens da base, então o resultado mede o sinal, não o ganho final do v8rftb.

### Resultado "gerar e conferir" (2026-10-23)

350 questões rotuladas pelo professor, resolvedor Qwen3-4B Q4_K_M base, alternativas permutadas ("Nenhuma" fica por último).

| Modo | Mantidas | Núcleo entre mantidas | Válidas perdidas | Inválidas descartadas |
|---|---|---|---|---|
| sem filtro | 350 | 65,4% | – | – |
| direto (só a letra) | 154 (44%) | 74,0% [66,6; 80,3] | 50,2% | 66,9% |
| **cot (raciocínio curto)** | **260 (74%)** | **82,7% [77,6; 86,8]** | **6,1%** | **62,8%** |
| verificador `ok` | 158 (45%) | 82,9% | – | – |
| cot e `ok` | 137 (39%) | 91,2% | – | – |

- O direto falha porque o resolvedor sem conta acerta só 50% das válidas. Com raciocínio, concorda em 215/229 válidas (94%) e em 45/121
  inválidas (37%).
- A média de 82,7% é puxada pelos itens da base (96% válidos). Só saídas de modelo: 40,7% -> 64,6% (perde 8% das válidas); Qwen:
  49,4% -> 68,6%; 4B ft+zs (n=53): 64,2% -> 75,0% (perde 12%); 9º ano: 37,0% -> 61,5%.
- 21 respostas sem letra são quase todas "Resposta: Nenhuma das alternativas anteriores" por extenso; o leitor passou a mapeá-las para a
  alternativa "Nenhuma" quando existe. O resultado não mudou (continuam descarte).
- Custo medido: 350 itens em ~67 min no PC, ~11,5 s por conferência (CPU, 8 threads).
- Projeção para o v8rftb (65,4% bruto) com as taxas do subconjunto 4B: ~76%; com as taxas globais: ~83%. Falta a medida direta:
  `--folha outputs/melhoria_v8/folha_v8rftb/folha_preenchida.csv --chave .../chave_oculta.json --rotulo v8rftb`.

### Validação direta no v8rftb (2026-10-23)

`auto_conferencia.py apurar --modo cot --permutar --folha folha_v8rftb/... --rotulo v8rftb` (78 itens do professor, 51 válidos):

| Regra | Mantidas | Núcleo entre mantidas | Válidas perdidas |
|---|---|---|---|
| Sem filtro | 78 | 51/78 = 65,4% | – |
| cot | 66 (84,6%) | 48/66 = 72,7% [61,0; 82,0] | 3/51 = 5,9% |
| verificador ok | 42 (53,8%) | 33/42 = 78,6% | – |
| cot ∧ ok | 37 (47,4%) | 31/37 = 83,8% | – |

Abaixo da projeção (~76–83%): a conferência descarta só 33% das inválidas do v8rftb (contra 63% em F/rotulados).
Das 18 inválidas mantidas, nenhuma é "conta errada com gabarito errado". São erros que um resolvedor que só compara a
própria letra com o gabarito não vê (mesma classificação das 27 reprovadas acima): situação mal definida ou dado que falta
(Q1032, Q1040, Q1050, Q1051, Q1055, Q1057, Q1076), mais de uma alternativa correta (Q1003, Q1016, Q1019, Q1049, Q1069),
dados contraditórios (Q1017, Q1021, Q1043, Q1052) e resolução com justificativa errada ou que vaza a resposta (Q1028, Q1077).
Por ano: 2º 95%, 5º 67%, 9º 55% entre mantidas. Critério de decisão (≥80%) não atingido pela cot sozinha.

### Modo `critica` (2026-10-23, preparado; execução do responsável)

`auto_conferencia.py --modo critica` (testes em `tests/test_auto_conferencia.py`, 8 no total). O mesmo 4B base, às cegas, julga cada
alternativa e termina com `CORRETAS: ...`, `CONTRADICAO: SIM|NAO`, `SUFICIENTE: SIM|NAO`. Passa só com exatamente uma correta
(= gabarito, mapeada de volta da permutação), sem contradição e com dados suficientes; linha faltando ou valor fora de SIM/NAO =
descarte (`motivo` = formato). `apurar --modo critica` também cruza com a `cot` gravada (`cot_e_critica`, `cot_e_critica_e_ok`).
Critério de leitura nas 78 do v8rftb: núcleo entre mantidas ≥ 80% com perda de válidas ≤ ~15%.
- **v1 testada em 5 itens (descartada):** formato ok (0 descartes por formato), mas 4 de 5 respostas trouxeram só as três linhas
  finais, sem conta (Q1001: "A,D" para 70 − 45 = 25), e a Q1005 inverteu SIM/NAO. Perdeu 2 das 3 válidas. Evidência mantida em
  `respostas_critica_perm_v8rftb.jsonl`. **v2:** análise obrigatória antes do veredito (resolver, julgar cada alternativa, checar
  dados) e polaridade única (`CONTRADICAO`/`FALTA_DADO`, SIM = problema), `max_tokens` 1200. Saída em `respostas_criticav2_*`.

### Resultado do modo `critica` v2 no v8rftb (2026-10-23)

`apurar --modo critica --permutar --folha folha_v8rftb/... --rotulo v8rftb` (78 itens, 51 válidos; `apuracao_criticav2_perm_v8rftb.json`):

| Regra | Mantidas | Núcleo entre mantidas | Válidas perdidas | Inválidas descartadas |
|---|---|---|---|---|
| Sem filtro | 78 | 65,4% | – | – |
| cot | 66 (85%) | 72,7% [61,0; 82,0] | 5,9% | 33% |
| **critica** | **54 (69%)** | **85,2% [73,4; 92,3]** | **9,8% (5/51)** | **70% (19/27)** |
| critica sem os 5 itens usados no ajuste do prompt | 51/73 | 84,3% [72,0; 91,8] | 10,4% (5/48) | – |
| cot ∧ critica | 51 (65%) | 88,2% | 11,8% | – |
| critica ∧ verificador ok | 31 (40%) | 96,8% | – | – |

- Passa o critério combinado (≥ 80% com ≤ ~15% de válidas perdidas), mas o limite inferior do IC (73%) ainda está abaixo de 80% e o
  método foi desenhado olhando os erros DESTES 78 itens: falta validar em uma amostra nova antes de levar ao app.
- Pareado com a cot: a crítica descarta 12 inválidas que a cot mantinha, ao custo de 3 válidas; o contrário ocorre em 2 inválidas e 1 válida.
- 8 inválidas mantidas: ambiguidade que o modelo não percebe (losango, 1,4 = 1,40, variáveis não definidas, planificação, posição inicial
  não informada: Q1016, Q1049, Q1050, Q1057, Q1076) e **resolução errada com enunciado e gabarito defensáveis** (Q1070, Q1071, Q1077). A
  crítica não vê a resolução, então esses três tipos não podem ser pegos por ela.
- 5 válidas perdidas: 2 de classificação de quadriláteros em que o resolvedor exige dado a mais (Q1058, Q1059), 1 resolvida errado (Q1075),
  1 em que achou "nenhuma correta" (Q1009) e 1 truncada (Q1022).
- 3 respostas truncadas em 1.200 tokens (`formato`, descarte seguro): Q1022, Q1040, Q1055.
- Custo: 73 itens em ~1 h no PC (pelo horário do arquivo), ~50 s por crítica. Com geração (~21 s) e 69% mantidas: ~105 s por questão
  entregue no PC, contra ~21 s hoje. No celular será maior. Este é o obstáculo para o app.

## Validação da conferência em folha nova (2026-10-25, protocolo fixado ANTES de gerar os dados)

**Pergunta:** a crítica v2 (85,2% [73,4; 92,3] nas 78 questões do v8rftb, método desenhado olhando esses erros) mantém o ganho em
questões que nunca viu?

**Protocolo** (`src/validacao_conferencia.py`, `tests/test_validacao_conferencia.py`):
- 100 questões do **v7b** (gerador diferente do v8rftb, com o qual o prompt foi ajustado), seed 20261201 (as anteriores foram 20261015,
  20261101 e as da auditoria), pelo mesmo `gerar_folha_professor.py` que o app usa (só entram as ENTREGUES). Sem repetição de enunciado
  com as folhas de 78 e de 222 (`validacao_conferencia.py novidade`).
- **O professor avalia TODAS às cegas**, as que a crítica mantém e as que descarta; a decisão da crítica não aparece na folha.
- Conferência rodada com `--permutar`, temperatura 0, Qwen3-4B Q4_K_M base. **Prompt congelado:** crítica v2, sha256[:16] `831469ec8262eeed`
  (system + formato + max_tokens), travado por `test_prompt_critica_congelado`. A cot também roda, só como referência.
- Apuração exige respostas para as 100 questões e a folha completa (sem apuração parcial).

**Critério (todos precisam valer, sobre as 100 questões da folha nova):**
1. núcleo entre as mantidas pela crítica ≥ 80%;
2. limite inferior do IC95 de Wilson ≥ 70%;
3. válidas perdidas ≤ 15%;
4. ganho ≥ 10 pontos sobre o núcleo sem filtro da MESMA folha.

**Leitura do resultado:** VALIDADO → medir no celular e seguir para o app. NÃO VALIDADO com núcleo entre 70% e 80% → a crítica ajuda mas
não basta; avaliar cot ∧ crítica e crítica ∧ verificador `ok` (já com custo de perda maior) e a checagem da resolução. Abaixo de 70% → o
85,2% era ajuste às 78 questões; descartar. Nenhum ajuste de prompt depois de ver esta folha; mudança de prompt = nova versão e nova folha.

Estimativa de precisão: com ~70 mantidas, o IC95 tem meia-largura de cerca de ±8 pontos. Gerar 100 questões leva ~40 min e a crítica
~85 min no PC; o professor leva ~4 a 5 h.

**Desvio do protocolo (2026-10-25):** a geração entregou **77** questões, não 100: `gerar_folha_professor.py` sorteia cada par
(ano, habilidade) uma vez e a taxonomia tem 79 pares, então o teto é 79 menos as falhas (2 não entregues). O critério e o prompt
não mudaram. Com ~54 mantidas esperadas, o IC95 tem meia-largura de ~±9 pontos, e o limite inferior ≥ 70% só é atingido com
núcleo observado de cerca de 83% ou mais. Se o veredito ficar na fronteira (núcleo 75–85%), uma segunda folha com outra seed é o
próximo passo, não mudar o critério. Tempos com 77 questões: crítica ~65 min, cot ~15 min, professor ~3 a 4 h.

**Folha de validação devolvida (2026-10-25), ANTES de rodar a cot e a crítica nela:** 77 de 77 preenchidas, mesmos ids e
enunciados da folha gerada. Núcleo sem filtro pelo professor: **61/77 = 79,2%** [68,9; 86,8], bem acima dos 68,9% do v7b na
medição F (n=222). Com esse ponto de partida, o critério 4 (ganho ≥ 10 pp) exige ≥ 89,2% entre as mantidas.

Seis rótulos "válida" parecem errados na leitura técnica (registrado agora, antes do resultado da crítica):
Q1006 (o primeiro em ordem crescente não é Davi; gabarito errado), Q1007 (alternativas sem sentido: "Cocô", "Quarto"...),
Q1008 (total 5.600, nenhuma alternativa certa), Q1013 (ponteiro dos minutos no 10 = 8h50, nenhuma alternativa certa),
Q1014 (8 e 18 são pares; nenhuma alternativa verdadeira) e Q1017 (resolução diz que as faces do dado são hexágonos).
**Regra fixada agora:** a apuração OFICIAL usa os rótulos do professor como vieram. Os seis vão para o professor rever às cegas,
misturados a outras questões; se ele mudar algum, roda-se uma apuração de SENSIBILIDADE, reportada ao lado da oficial, sem
substituí-la. Nenhum rótulo é trocado por leitura técnica (política de decisão humana do projeto).

**Resultado OFICIAL (2026-10-26): NÃO VALIDADO.** `apuracao_val_v7b.json`, prompt `831469ec8262eeed`, 77/77 respostas.

| regra | mantidas | núcleo entre mantidas | válidas perdidas |
|---|---|---|---|
| sem filtro | 77 | 79,2% [68,9; 86,8] | 0% |
| **crítica** | **54 (70%)** | **88,9% [77,8; 94,8]** | **21,3% (13/61)** |
| cot | 64 (83%) | 85,9% [75,4; 92,4] | 9,8% |
| cot ∧ crítica | 53 | 90,6% [79,7; 95,9] | 21,3% |
| crítica ∧ verificador ok | 32 | 90,6% [75,8; 96,8] | 52,5% |

- Critérios 1 e 2 passam (88,9% ≥ 80; 77,8 ≥ 70). Falham o 3 (perdidas 21,3% > 15%) e o 4 (ganho 9,7 pp < 10 pp).
- A precisão se repetiu fora da amostra de ajuste (88,9% contra 85,2%). O que não passou foi o custo: a crítica descarta válidas demais.
- Motivos entre as 13 válidas perdidas: 6 `insuficiente`, 4 `nenhuma_correta`, 2 `varias_corretas`, 1 `formato`. A cot mantém 7 delas.
- 6 inválidas passam: Q1023, Q1027, Q1033, Q1042, Q1050, Q1067.
- **5 das 13 válidas perdidas são rótulos suspeitos registrados antes da crítica** (Q1006 e Q1007 `insuficiente`; Q1008, Q1013 e Q1014
  `nenhuma_correta`). Q1017 foi mantida pela crítica.
- Sem nenhum ajuste de prompt ou de critério. Pela regra fixada, a revisão cega dos seis vai ao professor (`validacao_conferencia.py revisao`:
  6 suspeitas + 10 controles sorteados, seed 20261026, ids novos; `revisao_cega/`). A sensibilidade usa os rótulos revistos de TODAS as 16,
  inclusive dos controles, e é reportada ao lado da oficial, sem substituí-la.
- Cenário hipotético (calculado agora para saber o que está em jogo, não é resultado): se o professor reprovar os 5 descartados, ou os 6,
  a sensibilidade dá crítica 88,9% (ou 87,0%), perdidas 14,3% (14,5%), ganho 16 pp: passaria, mas com 0,5 pp de folga no critério 3. Uma
  válida a mais perdida, ou um controle que vire válido entre os descartados, reverte. Fronteira: se a sensibilidade passar e a oficial não,
  o próximo passo é uma segunda folha com outra seed (regra do desvio), não levar ao app direto.

## Teste de amostragem do card do Qwen3 (pré-registrado em 2026-10-26, antes de medir)

**Hipótese.** O pedido do app envia só `temperature 0,7` e `top_p 0,8`; o llama-server completa com `top_k 40` e `min_p 0,05`. O card do
Qwen3-4B recomenda, sem pensamento, `top_k 20` e `min_p 0`. Hipótese: alinhar reduz a cauda de tokens improváveis e o erro de conta.
Não há medição prévia; pode não mudar nada.

**Braços (mesmo .gguf e mesmo LoRA v7b s44, gramática `questao.gbnf`, `--sufixo`, 30 itens do `val_frozen_v1`):**
baseline = `outputs/hf_4b/avaliacao/eval_4b_ft_suf_r{0,1,2}.json` (já medido, padrão do llama.cpp);
candidato = `--top-k 20 --min-p 0`, rodadas 0, 1 e 2 (mesmas seeds). `report["inferencia"]["amostragem"]` registra o que mudou.

**Decisão, mecânica:** gate `multiseed --comparar-inferencia`. NÃO PROMOVIDO ou empate = fica o padrão atual, fim. PROMOVIDO pelo gate
NÃO basta para mudar o app: o gate mede o verificador, não o núcleo. Antes disso, folha cega do professor com o mesmo protocolo da
`val_v7b` (núcleo, sem rótulos de origem), e só adota se o núcleo do candidato não ficar abaixo do baseline (Fisher unilateral,
p > 0,10 contra queda). Sem folha, o resultado do gate é informativo.

**Proibido depois do resultado:** trocar os valores (ex.: testar `top_k 10`, outra temperatura) como se fosse o mesmo teste. Cada
configuração nova é outro teste, com novo registro aqui, e conta como mais um braço para quem lê o resultado.

**Resultado (2026-10-26): gate PROMOVIDO, sem efeito detectável.** `outputs/relatorios/VEREDITO_4b_topk20.json`, 3 rodadas, 90 pares.
G2* 3/90 -> 3/90 (p=1,0); G3* 93,6% -> 95,9% (piorou 2, melhorou 3, p=1,0); G6* 40,0% -> 40,4%. Os "ganhos" listados pelo gate
(consistência, verificabilidade, regenerações) são diferenças de 1 a 2 itens, sem significância. "Nenhuma das anteriores" 2,2% -> 4,5%
(n.s., a acompanhar). Pela regra acima, sem folha do professor o resultado é só informativo: **o padrão do app não muda**. Conclusão:
a amostragem não é alavanca mensurável com n=90; não justifica gastar uma folha do professor.

**Sensibilidade (2026-10-26): VALIDADO, oficial segue NÃO VALIDADO.** O professor mudou as 6 suspeitas para inválidas e nenhum dos 10 controles.
Núcleo sem filtro 71,4% (55/77). Crítica: 87,0% [75,6; 93,6], perdidas 14,5% (8/55), ganho 15,6 pp. Folga de 0,5 pp no critério 3.
Pela regra fixada: segunda folha com outra seed antes de levar ao app.

## Segunda folha de validação da crítica (pré-registrado em 2026-10-26, antes de gerar)

**Por quê:** a oficial deu NÃO VALIDADO (perdidas 21,3%) e a sensibilidade VALIDADO com 0,5 pp de folga; regra fixada: segunda folha, outra seed.
**Protocolo igual ao da primeira, sem mudança:** v7b s44 (`lora_4b_v7b_s44.f16.gguf`), `gerar_folha_professor.py --n 100`, seed **20261208**
(nova), rótulo `val_v7b_2`; `novidade` contra a folha de 77 (`--contra`) e as antigas; professor avalia todas às cegas; conferência `--permutar`,
temperatura 0, prompt crítica v2 `831469ec8262eeed` congelado. Mesmo critério (núcleo mantidas ≥ 80%; IC95 inferior ≥ 70%; perdidas ≤ 15%;
ganho ≥ 10 pp), aplicado à **segunda folha sozinha**, com rótulos do professor como vierem. Reportado ao lado, só informativo: as duas folhas
somadas (rótulos oficiais da 1ª com as 6 revisões, marcado como tal). VALIDADO = só então medir no celular. NÃO VALIDADO = a crítica
sai do plano do app; nada de ajustar prompt ou critério.

**Resultado da segunda folha (2026-10-26): VALIDADO**, `val_v7b_2`, 74/74, seed 20261208, prompt `831469ec8262eeed`, rótulos do professor como vieram.
Sem filtro 73,0% [61,9; 81,8]. Crítica: mantém 50 (67,6%), núcleo 94,0% [83,8; 97,9], perdidas 13,0% (7/54), ganho 21,0 pp. Os 4 critérios passam.
cot: 82,5%, perdidas 3,7%. Informativo (1ª folha com as 6 revisões + 2ª): crítica ~90% de núcleo, ~14% de válidas perdidas, ~18 pp de ganho.
Próximo passo conforme a leitura fixada: medir latência no celular. Descartes da crítica: ~1/3, logo ~1,5 geração por questão entregue.
