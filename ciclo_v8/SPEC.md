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
- Por unidade: Geometria 42%, Álgebra 51%, Prob./Estat. 56%, Números 58%, Grandezas 67%.
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
- Por ano: 2º 83% (19/23), 5º 59% (13/22), 9º 50% (13/26). Por unidade: Geometria 46%, Grandezas 46%, Números 58%,
  Álgebra 56%, Prob./Estat. 70%. Dimensões: correção 69,2%, resposta única 73,1%, dados suficientes 80,8%.
- Verificador × professor: `ok` 33/42 = 79%; `nao_verificavel` 18/36 = 50%. Letra do gabarito, "Nenhuma das anteriores"
  (59% contra 68%) e resolução longa (n=8) não separam com significância.
- Causas das 27 reprovadas (leitura minha): ~11 enunciado contraditório, ~6 mais de uma correta, ~4 conta errada, resto situação mal definida.
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
