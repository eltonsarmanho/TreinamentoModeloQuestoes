# Relatório da Base de Conhecimento (treino)

Gerado por `src/relatorio_base.py` em 2026-10-04 16:41 a partir de `data/train_curado_v3.jsonl`, `DB/questoes.db` e `data/taxonomia_subtemas.json`. Reexecute o script para atualizar: este arquivo é gerado e não deve ser editado à mão.

## 1. Resumo executivo

- **2717 exemplos** de treino, **2759 questões**, **79 habilidades** (ano × código).
- Origem (exemplos): real do banco **693** (25.5%), sintético **59**, destilado **1965** (72.3%).
- Dificuldade (exemplos): Fácil 1061 · Moderado 852 · Difícil 804.
- Gabarito (`check_consistency`): ok **1310** (47.5%), inconsistente **0**, não verificável **1449** (52.5%).
- Near-duplicatas dentro da habilidade: **29** questões (1.1%).
- Subtemas da taxonomia com ≥1 exemplo: **228/252**.
- Questões **efetivas** (sem gabarito inconsistente e sem near-dup): **2730**.
- **Meta total: 2370 questões efetivas. Déficit total a injetar: 198** (Fácil 41 · Moderado 42 · Difícil 28 só para equilibrar dificuldade). 34/79 habilidades estão abaixo da meta.
- Itens reais **textuais livres** no banco (sem imagem, fora do treino e da validação): **33** — âncoras possíveis para a injeção.
- Letra da resposta correta: A 688 (24.9%) · B 643 (23.3%) · C 619 (22.4%) · D 500 (18.1%) · E 309 (11.2%). (Equilíbrio seria 20% cada.)
- Verificador de geometria (`src/verificador_geometria.py`): disponivel.

### Distribuição por ano

| Ano | Matriz | Hab. | Exemplos | Questões | Efetivas | Real/Sint/Dest | F/M/D | Subtemas cobertos | Near-dup | Gab. incons. | Livres no banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1º | BNCC | 3 | 232 | 232 | 232 | 140/0/92 | 149/53/30 | 8/8 | 0 | 0 | 1 | 90 | **0** |
| 2º | BNCC+SAEB | 24 | 831 | 844 | 842 | 168/24/639 | 307/256/268 | 58/61 | 2 | 0 | 6 | 720 | **54** |
| 3º | BNCC | 3 | 193 | 193 | 192 | 123/0/70 | 111/52/30 | 6/6 | 1 | 0 | 0 | 90 | **0** |
| 4º | BNCC | 1 | 59 | 59 | 59 | 47/0/12 | 26/23/10 | 4/4 | 0 | 0 | 0 | 30 | **0** |
| 5º | SAEB | 22 | 655 | 671 | 657 | 64/19/572 | 217/221/217 | 65/75 | 14 | 0 | 4 | 660 | **68** |
| 9º | SAEB | 26 | 747 | 760 | 748 | 151/16/580 | 251/247/249 | 87/98 | 12 | 0 | 22 | 780 | **76** |

### As 15 piores habilidades (maior fração da meta faltando)

| # | Ano | Hab. | Descrição | Efetivas | Meta | Déficit | F/M/D | Subtemas | Livres no banco |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2º | H21 | Ler/Identificar OU Comparar dados estatísticos expressos em… | 26 | 30 | **20** | 0/0/26 | 3/3 | 0 |
| 2 | 9º | H17 | Classificação de triângulos e quadriláteros quanto aos lado… | 12 | 30 | **18** | 2/4/6 | 2/2 | 5 |
| 3 | 2º | H11 | Reconhecer/nomear figuras geométricas planas (círculo, quad… | 17 | 30 | **13** | 3/6/8 | 3/5 | 2 |
| 4 | 5º | H01 | Escrever números racionais (naturais de até 6 ordens, repre… | 30 | 30 | **12** | 10/10/10 | 1/5 | 0 |
| 5 | 5º | H11 | Descrever OU Esboçar o deslocamento de pessoas e/ou objetos… | 20 | 30 | **10** | 7/8/5 | 3/3 | 0 |
| 6 | 9º | H01 | Escrever números racionais (representação fracionária ou de… | 30 | 30 | **9** | 10/10/10 | 1/4 | 1 |
| 7 | 2º | H13 | Identificar a localização OU a descrição/esboço do deslocam… | 22 | 30 | **8** | 8/7/7 | 3/3 | 0 |
| 8 | 5º | H10 | Identificar a localização OU a descrição/esboço do deslocam… | 23 | 30 | **7** | 8/8/7 | 3/3 | 0 |
| 9 | 5º | H14 | Reconhecer/nomear figuras geométricas planas (polígonos, ci… | 23 | 30 | **7** | 7/8/8 | 4/5 | 0 |
| 10 | 9º | H14 | Relacionar objetos tridimensionais às suas planificações ou… | 24 | 30 | **6** | 8/8/8 | 7/7 | 0 |
| 11 | 9º | H02 | Resolver problemas de adição, subtração, multiplicação, div… | 27 | 30 | **6** | 10/9/8 | 6/7 | 0 |
| 12 | 9º | H24 | Representar dados em listas, tabelas ou gráficos. | 29 | 30 | **6** | 13/7/9 | 8/9 | 0 |
| 13 | 5º | H21 | Ler/Identificar OU Comparar dados estatísticos expressos em… | 30 | 30 | **6** | 10/10/10 | 0/2 | 0 |
| 14 | 9º | H03 | Porcentagem, acréscimos, decréscimos e taxas sucessivas. | 30 | 30 | **6** | 10/10/10 | 3/5 | 0 |
| 15 | 9º | H23 | Leitura e interpretação de tabelas e gráficos. | 30 | 30 | **6** | 10/10/10 | 7/8 | 0 |

## 2. Critério da meta (como o déficit é calculado)

1. **Mínimo de 30 questões efetivas por habilidade.** O prompt de treino é condicionado em (ano, habilidade, dificuldade); 30 dá 10 por dificuldade, abaixo disso o modelo tende a decorar exemplos em vez de aprender o padrão. O banco SAEB real tem só 8 itens por descritor, então 30 mantém a âncora real relevante (~25%) sem ser engolida pelos sintéticos.
2. **Pelo menos 3 por subtema da taxonomia.** Com 1 exemplo o modelo copia o molde; 3 é o mínimo para variar contexto e estrutura dentro do subtema. Habilidade com K subtemas tem meta ≥ 3·K.
3. **Equilíbrio entre as 3 dificuldades:** a meta é arredondada para múltiplo de 3 e o alvo por dificuldade é meta/3. Desequilíbrio ensina o modelo a ignorar a dificuldade pedida no prompt.
4. Só contam questões **efetivas**: questão com gabarito não-inconsistente e que não é near-dup de outra da mesma habilidade. Gabarito errado ensina errado; near-dup não ensina nada novo.
5. **Déficit = max(meta − efetivas, Σ faltas por dificuldade, Σ faltas por subtema).** Cada questão nova tem uma dificuldade e um subtema escolhidos livremente; por isso o mínimo que cumpre os três critérios ao mesmo tempo é o maior dos três, não a soma.

Limites conhecidos: o subtema vem do classificador léxico de `diversidade.py` (pode errar em enunciados atípicos — veja a coluna 'outros'); `check_consistency` só verifica contas explícitas e **não detecta erro conceitual** (ex.: o erro central do caso 9º H17 auditado, "tem uma propriedade de X ⇒ é X"). "ok" aqui é necessário, não suficiente.

## 3. Alertas

### Gabarito inconsistente (a conta da resolução aponta outra alternativa ou nenhuma) — 0

Nenhum.

### Desequilíbrio de dificuldade (alguma fora de 15–60%, n≥6; ou dificuldade ausente) — 6

- **1º EF01MA02**: F/M/D = 37/10/10 (fora: {'Fácil': 0.65})
- **1º EF01MA06**: F/M/D = 55/17/10 (fora: {'Fácil': 0.67, 'Difícil': 0.12})
- **1º EF01MA08**: F/M/D = 57/26/10 (fora: {'Fácil': 0.61, 'Difícil': 0.11})
- **2º H21**: F/M/D = 0/0/26 (fora: {'Fácil': 0.0, 'Moderado': 0.0, 'Difícil': 1.0})
- **3º EF03MA03**: F/M/D = 33/29/10 (fora: {'Difícil': 0.14})
- **3º EF03MA07**: F/M/D = 48/13/10 (fora: {'Fácil': 0.68, 'Difícil': 0.14})

### Subtemas da taxonomia sem nenhum exemplo (K≥2) — 16

- **2º H11**: 2/5 vazios — trapezio, poligono
- **2º H18**: 1/4 vazios — calendario
- **5º H01**: 4/5 vazios — naturais_6_ordens, fracao, decimal, registro_lingua_materna
- **5º H06**: 1/5 vazios — multiplicacao
- **5º H09**: 1/4 vazios — seq_figuras
- **5º H13**: 1/6 vazios — esfera
- **5º H14**: 1/5 vazios — trapezio
- **5º H21**: 2/2 vazios — tabela_dupla, tabela_simples
- **9º H01**: 3/4 vazios — fracao, decimal, registro_lingua_materna
- **9º H02**: 1/7 vazios — radiciacao
- **9º H03**: 2/5 vazios — taxas_sucessivas, taxa_percentual
- **9º H13**: 1/2 vazios — sistema
- **9º H19**: 1/6 vazios — volume_medida
- **9º H21**: 1/5 vazios — poligono
- **9º H23**: 1/8 vazios — tabela_dupla
- **9º H24**: 1/9 vazios — tabela_simples

### Near-duplicatas altas (≥10% das questões) — 3

- **5º H03**: 12% (4 questões)
- **5º H06**: 15% (9 questões)
- **9º H09**: 12% (4 questões)

### Gabarito sem verificação automática (≥80% não verificável): exige validação conceitual — 28

- **2º H02**: 28/30 não verificáveis
- **2º H03**: 25/30 não verificáveis
- **2º H04**: 25/30 não verificáveis
- **2º H11**: 16/17 não verificáveis
- **2º H12**: 27/30 não verificáveis
- **2º H13**: 21/22 não verificáveis
- **2º H14**: 29/30 não verificáveis
- **2º H17**: 29/30 não verificáveis
- **2º H18**: 26/27 não verificáveis
- **5º H10**: 21/23 não verificáveis
- **5º H11**: 19/20 não verificáveis
- **5º H12**: 30/30 não verificáveis
- **5º H13**: 28/29 não verificáveis
- **5º H14**: 20/23 não verificáveis
- **5º H18**: 28/30 não verificáveis
- **5º H19**: 24/30 não verificáveis
- **5º H20**: 30/30 não verificáveis
- **9º H01**: 24/30 não verificáveis
- **9º H04**: 27/30 não verificáveis
- **9º H07**: 32/33 não verificáveis
- **9º H08**: 26/32 não verificáveis
- **9º H11**: 28/30 não verificáveis
- **9º H13**: 26/30 não verificáveis
- **9º H14**: 24/24 não verificáveis
- **9º H15**: 26/30 não verificáveis
- **9º H16**: 29/30 não verificáveis
- **9º H17**: 12/12 não verificáveis
- **9º H25**: 25/30 não verificáveis

### Classificador não reconhece o subtema (≥30% em 'outros') — 19

- **1º EF01MA06**: 26/82 em 'outros'
- **2º EF02MA05**: 27/79 em 'outros'
- **2º H19**: 28/30 em 'outros'
- **2º H20**: 11/27 em 'outros'
- **2º H21**: 14/26 em 'outros'
- **3º EF03MA01**: 35/50 em 'outros'
- **3º EF03MA03**: 29/72 em 'outros'
- **3º EF03MA07**: 24/71 em 'outros'
- **4º EF04MA04**: 22/59 em 'outros'
- **5º H04**: 10/31 em 'outros'
- **5º H08**: 15/30 em 'outros'
- **5º H09**: 9/29 em 'outros'
- **5º H21**: 30/30 em 'outros'
- **9º H05**: 19/27 em 'outros'
- **9º H10**: 20/31 em 'outros'
- **9º H13**: 29/30 em 'outros'
- **9º H15**: 10/30 em 'outros'
- **9º H23**: 10/30 em 'outros'
- **9º H25**: 17/30 em 'outros'

### Campo `difficulty` do JSON diverge da dificuldade do meta — 0

Nenhum.

### Sem âncora real (nenhum item real no treino e nenhum textual livre no banco) — 13

- **2º H10**: banco tem 8 itens, 0 textuais
- **2º H13**: banco tem 8 itens, 0 textuais
- **2º H14**: banco tem 8 itens, 0 textuais
- **2º H19**: banco tem 8 itens, 0 textuais
- **2º H20**: banco tem 7 itens, 0 textuais
- **2º H21**: banco tem 1 itens, 0 textuais
- **5º H10**: banco tem 8 itens, 0 textuais
- **5º H11**: banco tem 8 itens, 0 textuais
- **5º H15**: banco tem 8 itens, 0 textuais
- **5º H16**: banco tem 8 itens, 0 textuais
- **5º H19**: banco tem 8 itens, 0 textuais
- **5º H21**: banco tem 8 itens, 0 textuais
- **5º H22**: banco tem 8 itens, 0 textuais

### Descrição idêntica para códigos diferentes no mesmo ano — 1

- **2º** H20, H21: "Ler/Identificar OU Comparar dados estatísticos expressos em gráficos (barras simples, col…"

### Dificuldade exigida pela meta que o banco real nunca usa nessa habilidade (confirmar com pedagogia) — 6

- **1º EF01MA02**: banco só tem Fácil, Moderado; a meta pede {'Difícil': 0} a mais nas faixas ausentes
- **1º EF01MA06**: banco só tem Fácil, Moderado; a meta pede {'Difícil': 0} a mais nas faixas ausentes
- **1º EF01MA08**: banco só tem Fácil, Moderado; a meta pede {'Difícil': 0} a mais nas faixas ausentes
- **2º H21**: banco só tem Difícil; a meta pede {'Fácil': 10, 'Moderado': 10} a mais nas faixas ausentes
- **3º EF03MA03**: banco só tem Fácil, Moderado; a meta pede {'Difícil': 0} a mais nas faixas ausentes
- **3º EF03MA07**: banco só tem Fácil, Moderado; a meta pede {'Difícil': 0} a mais nas faixas ausentes

### Letra da resposta correta desbalanceada (fora de 10–30% no global) — 0

Nenhum.

### Origem desconhecida (codigo_item fora do banco, sem flag de sintético/destilado) — 0

Nenhum.

## 4. Detalhe por habilidade

Colunas: Ex = exemplos; Q = questões; Ef = efetivas; R/S/D = real do banco/sintético/destilado; F/M/D = dificuldade (exemplos); Sub = subtemas cobertos/K; Dup = taxa de near-dup; Gab ok/inc/nv = gabarito ok/inconsistente/não verificável; Banco = itens totais/textuais/livres.

### 1º ano

| Hab. | Descrição | Ex | Q | Ef | R/S/D | F/M/D | Sub | Vazios | Dup | Gab ok/inc/nv | Banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EF01MA02 | Contar de maneira exata ou aproximada, utilizando diferentes estratég… | 57 | 57 | 57 | 47/0/10 | 37/10/10 | 1/1 | — | 0% | 54/0/3 | 50/50/0 | 30 | **0** |
| EF01MA06 | Construir fatos fundamentais da adição e utilizá-los em procedimentos… | 82 | 82 | 82 | 47/0/35 | 55/17/10 | 1/1 | — | 0% | 76/0/6 | 50/50/0 | 30 | **0** |
| EF01MA08 | Resolver e elaborar problemas de adição e de subtração, envolvendo nú… | 93 | 93 | 93 | 46/0/47 | 57/26/10 | 6/6 | — | 0% | 89/0/4 | 50/50/1 | 30 | **0** |

### 2º ano

| Hab. | Descrição | Ex | Q | Ef | R/S/D | F/M/D | Sub | Vazios | Dup | Gab ok/inc/nv | Banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EF02MA03 | Comparar quantidades de objetos de dois conjuntos, por estimativa e/o… | 51 | 51 | 51 | 42/0/9 | 29/12/10 | 1/1 | — | 0% | 50/0/1 | 50/45/0 | 30 | **0** |
| EF02MA05 | Construir fatos básicos da adição e subtração e utilizá-los no cálcul… | 79 | 79 | 79 | 47/0/32 | 45/22/12 | 2/2 | — | 0% | 71/0/8 | 50/50/0 | 30 | **0** |
| EF02MA06 | Resolver e elaborar problemas de adição e de subtração, envolvendo nú… | 81 | 81 | 81 | 46/0/35 | 37/26/18 | 6/6 | — | 0% | 75/0/6 | 49/49/0 | 30 | **0** |
| H01 | Reconhecer o que os números naturais indicam em diferentes situações:… | 30 | 30 | 30 | 1/0/29 | 10/10/10 | 1/1 | — | 0% | 7/0/23 | 8/1/0 | 30 | **0** |
| H02 | Identificar a posição ordinal de um objeto ou termo em uma sequência … | 30 | 30 | 30 | 3/0/27 | 10/10/10 | 1/1 | — | 0% | 2/0/28 | 8/3/0 | 30 | **0** |
| H03 | Escrever números naturais de até 3 ordens em sua representação por al… | 30 | 30 | 30 | 5/0/25 | 10/10/10 | 2/2 | — | 0% | 5/0/25 | 8/5/0 | 30 | **0** |
| H04 | Comparar OU Ordenar quantidades de objetos (até 2 ordens). | 30 | 30 | 30 | 1/0/29 | 10/10/10 | 1/1 | — | 0% | 5/0/25 | 8/1/0 | 30 | **0** |
| H05 | Comparar OU Ordenar números naturais, de até 3 ordens, com ou sem sup… | 30 | 30 | 30 | 2/0/28 | 10/10/10 | 1/1 | — | 0% | 7/0/23 | 8/2/0 | 30 | **0** |
| H06 | Calcular o resultado de adições ou subtrações, envolvendo números nat… | 30 | 32 | 30 | 5/1/24 | 10/10/10 | 2/2 | — | 6% | 25/0/7 | 8/5/0 | 30 | **0** |
| H07 | Compor OU Decompor números naturais de até 3 ordens por meio de difer… | 30 | 30 | 30 | 1/0/29 | 10/10/10 | 1/1 | — | 0% | 9/0/21 | 8/3/1 | 30 | **0** |
| H08 | Resolver problemas de adição ou de subtração, envolvendo números natu… | 53 | 64 | 64 | 6/23/24 | 16/17/20 | 6/6 | — | 0% | 61/0/3 | 8/7/0 | 30 | **0** |
| H09 | Identificar a classificação OU Classificar objetos ou representações … | 30 | 30 | 30 | 3/0/27 | 10/10/10 | 1/1 | — | 0% | 7/0/23 | 8/4/0 | 30 | **0** |
| H10 | Inferir os elementos ausentes em uma sequência de números naturais or… | 28 | 28 | 28 | 0/0/28 | 10/9/9 | 2/2 | — | 0% | 16/0/12 | 8/0/0 | 30 | **2** |
| H11 | Reconhecer/nomear figuras geométricas planas (círculo, quadrado, retâ… | 17 | 17 | 17 | 1/0/16 | 3/6/8 | 3/5 | trapezio, poligono | 0% | 1/0/16 | 8/3/2 | 30 | **13** |
| H12 | Reconhecer/nomear figuras geométricas espaciais (cubo, bloco retangul… | 30 | 30 | 30 | 2/0/28 | 10/10/10 | 5/5 | — | 0% | 3/0/27 | 8/2/0 | 30 | **1** |
| H13 | Identificar a localização OU a descrição/esboço do deslocamento de pe… | 22 | 22 | 22 | 0/0/22 | 8/7/7 | 3/3 | — | 0% | 1/0/21 | 8/0/0 | 30 | **8** |
| H14 | Comparar comprimentos, capacidades ou massas OU Ordenar imagens de ob… | 30 | 30 | 30 | 0/0/30 | 10/10/10 | 3/3 | — | 0% | 1/0/29 | 8/0/0 | 30 | **0** |
| H15 | Relacionar valores de moedas e/ou cédulas do sistema monetário brasil… | 30 | 30 | 30 | 0/0/30 | 10/10/10 | 2/2 | — | 0% | 7/0/23 | 8/3/3 | 30 | **0** |
| H16 | Estimar/Inferir medida de comprimento, capacidade ou massa de objetos… | 30 | 30 | 30 | 1/0/29 | 10/10/10 | 3/3 | — | 0% | 9/0/21 | 8/1/0 | 30 | **0** |
| H17 | Identificar sequência de acontecimentos relativos a um dia. | 30 | 30 | 30 | 1/0/29 | 10/10/10 | 1/1 | — | 0% | 1/0/29 | 8/1/0 | 30 | **0** |
| H18 | Identificar datas, dias da semana, ou meses do ano em calendário OU E… | 27 | 27 | 27 | 1/0/26 | 9/9/9 | 3/4 | calendario | 0% | 1/0/26 | 8/2/0 | 30 | **3** |
| H19 | Ler/Identificar OU Comparar dados estatísticos ou informações express… | 30 | 30 | 30 | 0/0/30 | 10/10/10 | 2/2 | — | 0% | 7/0/23 | 8/0/0 | 30 | **4** |
| H20 | Ler/Identificar OU Comparar dados estatísticos expressos em gráficos … | 27 | 27 | 27 | 0/0/27 | 10/8/9 | 3/3 | — | 0% | 12/0/15 | 7/0/0 | 30 | **3** |
| H21 | Ler/Identificar OU Comparar dados estatísticos expressos em gráficos … | 26 | 26 | 26 | 0/0/26 | 0/0/26 | 3/3 | — | 0% | 18/0/8 | 1/0/0 | 30 | **20** |

### 3º ano

| Hab. | Descrição | Ex | Q | Ef | R/S/D | F/M/D | Sub | Vazios | Dup | Gab ok/inc/nv | Banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EF03MA01 | Ler, escrever e comparar números naturais de até a ordem de unidade d… | 50 | 50 | 50 | 34/0/16 | 30/10/10 | 1/1 | — | 0% | 37/0/13 | 37/37/0 | 30 | **0** |
| EF03MA03 | Construir e utilizar fatos básicos da adição e da multiplicação para … | 72 | 72 | 71 | 47/0/25 | 33/29/10 | 2/2 | — | 1% | 66/0/6 | 50/50/0 | 30 | **0** |
| EF03MA07 | Resolver e elaborar problemas de multiplicação (por 2, 3, 4, 5 e 10) … | 71 | 71 | 71 | 42/0/29 | 48/13/10 | 3/3 | — | 0% | 67/0/4 | 50/45/0 | 30 | **0** |

### 4º ano

| Hab. | Descrição | Ex | Q | Ef | R/S/D | F/M/D | Sub | Vazios | Dup | Gab ok/inc/nv | Banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EF04MA04 | Utilizar as relações entre adição e subtração, bem como entre multipl… | 59 | 59 | 59 | 47/0/12 | 26/23/10 | 4/4 | — | 0% | 54/0/5 | 50/50/0 | 30 | **0** |

### 5º ano

| Hab. | Descrição | Ex | Q | Ef | R/S/D | F/M/D | Sub | Vazios | Dup | Gab ok/inc/nv | Banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H01 | Escrever números racionais (naturais de até 6 ordens, representação f… | 30 | 30 | 30 | 5/0/25 | 10/10/10 | 1/5 | naturais_6_ordens, fracao, decimal, registro_lingua_materna | 0% | 7/0/23 | 8/6/0 | 30 | **12** |
| H02 | Identificar a ordem ocupada por um algarismo OU seu valor posicional … | 30 | 30 | 30 | 4/0/26 | 10/10/10 | 1/1 | — | 0% | 20/0/10 | 8/6/0 | 30 | **0** |
| H03 | Calcular o resultado de adições ou subtrações envolvendo números natu… | 30 | 34 | 30 | 2/1/27 | 10/10/10 | 4/4 | — | 12% | 25/0/9 | 8/4/0 | 30 | **2** |
| H04 | Calcular o resultado de multiplicações ou divisões envolvendo números… | 30 | 31 | 30 | 1/2/27 | 10/10/10 | 2/2 | — | 3% | 20/0/11 | 8/1/0 | 30 | **0** |
| H05 | Resolver problemas de adição ou de subtração, envolvendo números natu… | 34 | 34 | 34 | 5/0/29 | 10/10/14 | 8/8 | — | 0% | 27/0/7 | 8/7/1 | 30 | **6** |
| H06 | Resolver problemas de multiplicação ou de divisão, envolvendo números… | 48 | 59 | 50 | 7/16/25 | 17/18/13 | 4/5 | multiplicacao | 15% | 48/0/11 | 8/7/0 | 30 | **6** |
| H07 | Inferir OU Descrever atributos ou propriedades comuns que os elemento… | 30 | 30 | 30 | 4/0/26 | 10/10/10 | 1/1 | — | 0% | 19/0/11 | 8/5/0 | 30 | **0** |
| H08 | Inferir o padrão ou a regularidade de uma sequência de números natura… | 30 | 30 | 30 | 6/0/24 | 10/10/10 | 2/2 | — | 0% | 11/0/19 | 8/6/0 | 30 | **0** |
| H09 | Inferir os elementos ausentes em uma sequência de números naturais or… | 29 | 29 | 29 | 4/0/25 | 9/10/10 | 3/4 | seq_figuras | 0% | 22/0/7 | 8/4/0 | 30 | **3** |
| H10 | Identificar a localização OU a descrição/esboço do deslocamento de pe… | 23 | 23 | 23 | 0/0/23 | 8/8/7 | 3/3 | — | 0% | 2/0/21 | 8/0/0 | 30 | **7** |
| H11 | Descrever OU Esboçar o deslocamento de pessoas e/ou objetos em repres… | 20 | 20 | 20 | 0/0/20 | 7/8/5 | 3/3 | — | 0% | 1/0/19 | 8/0/0 | 30 | **10** |
| H12 | Reconhecer/nomear figuras geométricas espaciais (prismas, pirâmides, … | 30 | 30 | 30 | 6/0/24 | 10/10/10 | 5/5 | — | 0% | 0/0/30 | 8/6/0 | 30 | **2** |
| H13 | Relacionar figuras geométricas espaciais (prismas retos, pirâmides re… | 29 | 29 | 29 | 4/0/25 | 9/10/10 | 5/6 | esfera | 0% | 1/0/28 | 8/5/1 | 30 | **5** |
| H14 | Reconhecer/nomear figuras geométricas planas (polígonos, circunferênc… | 23 | 23 | 23 | 4/0/19 | 7/8/8 | 4/5 | trapezio | 0% | 3/0/20 | 8/4/0 | 30 | **7** |
| H15 | Medir OU Comparar perímetro de figuras planas desenhadas em malha qua… | 30 | 30 | 30 | 0/0/30 | 10/10/10 | 3/3 | — | 0% | 9/0/21 | 8/0/0 | 30 | **0** |
| H16 | Medir OU Comparar área de figuras planas desenhadas em malha quadricu… | 30 | 30 | 30 | 0/0/30 | 10/10/10 | 3/3 | — | 0% | 12/0/18 | 8/0/0 | 30 | **0** |
| H17 | Identificar horas em relógios analógicos OU Associar horas em relógio… | 30 | 30 | 30 | 2/0/28 | 10/10/10 | 2/2 | — | 0% | 10/0/20 | 8/3/1 | 30 | **0** |
| H18 | Determinar o horário de início, o horário de término ou a duração de … | 30 | 30 | 30 | 5/0/25 | 10/10/10 | 3/3 | — | 0% | 2/0/28 | 8/5/0 | 30 | **1** |
| H19 | Relacionar valores de moedas e/ou cédulas do sistema monetário brasil… | 30 | 30 | 30 | 0/0/30 | 10/10/10 | 2/2 | — | 0% | 6/0/24 | 8/0/0 | 30 | **0** |
| H20 | Identificar, entre eventos aleatórios, aqueles que têm menor, maior o… | 30 | 30 | 30 | 5/0/25 | 10/10/10 | 2/2 | — | 0% | 0/0/30 | 8/6/1 | 30 | **0** |
| H21 | Ler/Identificar OU Comparar dados estatísticos expressos em tabelas (… | 30 | 30 | 30 | 0/0/30 | 10/10/10 | 0/2 | tabela_dupla, tabela_simples | 0% | 7/0/23 | 8/0/0 | 30 | **6** |
| H22 | Ler/Identificar OU Comparar dados estatísticos expressos em gráficos … | 29 | 29 | 29 | 0/0/29 | 10/9/10 | 4/4 | — | 0% | 8/0/21 | 8/0/0 | 30 | **1** |

### 9º ano

| Hab. | Descrição | Ex | Q | Ef | R/S/D | F/M/D | Sub | Vazios | Dup | Gab ok/inc/nv | Banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H01 | Escrever números racionais (representação fracionária ou decimal fini… | 30 | 30 | 30 | 7/0/23 | 10/10/10 | 1/4 | fracao, decimal, registro_lingua_materna | 0% | 6/0/24 | 8/8/1 | 30 | **9** |
| H02 | Resolver problemas de adição, subtração, multiplicação, divisão, pote… | 27 | 27 | 27 | 7/1/19 | 10/9/8 | 6/7 | radiciacao | 0% | 10/0/17 | 8/8/0 | 30 | **6** |
| H03 | Porcentagem, acréscimos, decréscimos e taxas sucessivas. | 30 | 32 | 30 | 7/1/22 | 10/10/10 | 3/5 | taxas_sucessivas, taxa_percentual | 6% | 12/0/20 | 8/8/0 | 30 | **6** |
| H04 | Comparar, ordenar ou aproximar números reais. | 30 | 30 | 30 | 6/0/24 | 10/10/10 | 1/1 | — | 0% | 3/0/27 | 8/8/2 | 30 | **0** |
| H05 | Múltiplos, divisores, MDC e MMC. | 27 | 27 | 27 | 7/0/20 | 10/8/9 | 4/4 | — | 0% | 8/0/19 | 8/8/1 | 30 | **4** |
| H06 | Cálculo com números reais: adição, subtração, multiplicação e divisão. | 30 | 30 | 30 | 6/1/23 | 10/10/10 | 4/4 | — | 0% | 17/0/13 | 8/8/1 | 30 | **1** |
| H07 | Representar ou associar frações a representações pictóricas. | 28 | 33 | 30 | 4/11/13 | 9/10/9 | 1/1 | — | 9% | 1/0/32 | 8/6/1 | 30 | **0** |
| H08 | Identificar frações equivalentes. | 30 | 32 | 30 | 6/1/23 | 10/10/10 | 1/1 | — | 6% | 6/0/26 | 8/6/0 | 30 | **0** |
| H09 | Converter entre representações de números racionais positivos (fraçõe… | 30 | 34 | 30 | 5/1/24 | 10/10/10 | 3/3 | — | 12% | 20/0/14 | 8/8/1 | 30 | **0** |
| H10 | Resolver uma equação polinomial de 1º grau. | 31 | 31 | 30 | 6/0/25 | 10/11/10 | 1/1 | — | 3% | 9/0/22 | 8/8/1 | 30 | **0** |
| H11 | Inferir equações, inequações ou sistemas de 1º grau que representam s… | 30 | 30 | 30 | 6/0/24 | 10/10/10 | 3/3 | — | 0% | 2/0/28 | 8/8/0 | 30 | **3** |
| H12 | Cálculo do valor numérico de expressões algébricas | 30 | 30 | 30 | 6/0/24 | 10/10/10 | 1/1 | — | 0% | 14/0/16 | 8/8/1 | 30 | **0** |
| H13 | Resolver problemas com sistemas de equações de 1º grau (duas incógnit… | 30 | 30 | 30 | 7/0/23 | 10/10/10 | 1/2 | sistema | 0% | 4/0/26 | 8/8/1 | 30 | **5** |
| H14 | Relacionar objetos tridimensionais às suas planificações ou vistas. | 24 | 24 | 24 | 5/0/19 | 8/8/8 | 7/7 | — | 0% | 0/0/24 | 5/5/0 | 30 | **6** |
| H15 | Ângulos em polígonos, retas paralelas e elementos notáveis (cevianas). | 30 | 30 | 30 | 4/0/26 | 10/10/10 | 3/3 | — | 0% | 4/0/26 | 5/5/1 | 30 | **0** |
| H16 | Propriedades dos triângulos. | 30 | 30 | 30 | 6/0/24 | 10/10/10 | 5/5 | — | 0% | 1/0/29 | 8/8/1 | 30 | **0** |
| H17 | Classificação de triângulos e quadriláteros quanto aos lados ou ângul… | 12 | 12 | 12 | 1/0/11 | 2/4/6 | 2/2 | — | 0% | 0/0/12 | 7/7/5 | 30 | **18** |
| H18 | Elementos da circunferência e do círculo | 30 | 30 | 30 | 7/0/23 | 10/10/10 | 4/4 | — | 0% | 10/0/20 | 7/7/0 | 30 | **0** |
| H19 | Medidas e conversão de unidades | 30 | 30 | 30 | 6/0/24 | 10/10/10 | 5/6 | volume_medida | 0% | 12/0/18 | 8/8/1 | 30 | **3** |
| H20 | Perímetro de figuras planas. | 30 | 30 | 30 | 7/0/23 | 10/10/10 | 5/5 | — | 0% | 14/0/16 | 8/8/0 | 30 | **0** |
| H21 | Área de figuras planas | 29 | 29 | 29 | 7/0/22 | 9/10/10 | 4/5 | poligono | 0% | 10/0/19 | 8/8/0 | 30 | **3** |
| H22 | Volume de prismas retos e cilindros retos | 30 | 30 | 30 | 7/0/23 | 10/10/10 | 2/2 | — | 0% | 12/0/18 | 8/8/0 | 30 | **0** |
| H23 | Leitura e interpretação de tabelas e gráficos. | 30 | 30 | 30 | 4/0/26 | 10/10/10 | 7/8 | tabela_dupla | 0% | 12/0/18 | 7/5/0 | 30 | **6** |
| H24 | Representar dados em listas, tabelas ou gráficos. | 29 | 29 | 29 | 6/0/23 | 13/7/9 | 8/9 | tabela_simples | 0% | 7/0/22 | 7/7/0 | 30 | **6** |
| H25 | Probabilidade de eventos aleatórios equiprováveis | 30 | 30 | 30 | 6/0/24 | 10/10/10 | 2/2 | — | 0% | 5/0/25 | 8/8/1 | 30 | **0** |
| H26 | Média, moda e mediana. | 30 | 30 | 30 | 5/0/25 | 10/10/10 | 3/3 | — | 0% | 7/0/23 | 8/8/3 | 30 | **0** |

Detalhes completos (contagem por subtema, falta por dificuldade/subtema, pares de near-dup, códigos livres no banco, itens inconsistentes) estão em `outputs/relatorio_base.json`.
