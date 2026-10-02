# Relatório da Base de Conhecimento (treino)

Gerado por `src/relatorio_base.py` em 2026-10-01 11:12 a partir de `data/train_curado.jsonl`, `DB/questoes.db` e `data/taxonomia_subtemas.json`. Reexecute o script para atualizar: este arquivo é gerado e não deve ser editado à mão.

## 1. Resumo executivo

- **1435 exemplos** de treino, **1481 questões**, **79 habilidades** (ano × código).
- Origem (exemplos): real do banco **719** (50.1%), sintético **61**, destilado **655** (45.6%).
- Dificuldade (exemplos): Fácil 658 · Moderado 413 · Difícil 364.
- Gabarito (`check_consistency`): ok **886** (59.8%), inconsistente **1**, não verificável **594** (40.1%).
- Near-duplicatas dentro da habilidade: **34** questões (2.3%).
- Subtemas da taxonomia com ≥1 exemplo: **215/252**.
- Questões **efetivas** (sem gabarito inconsistente e sem near-dup): **1446**.
- **Meta total: 2370 questões efetivas. Déficit total a injetar: 1386** (Fácil 444 · Moderado 481 · Difícil 448 só para equilibrar dificuldade). 76/79 habilidades estão abaixo da meta.
- Itens reais **textuais livres** no banco (sem imagem, fora do treino e da validação): **7** — âncoras possíveis para a injeção.
- Letra da resposta correta: A 438 (29.6%) · B 386 (26.1%) · C 372 (25.1%) · D 239 (16.1%) · E 46 (3.1%). (Equilíbrio seria 20% cada.)
- Verificador de geometria (`src/verificador_geometria.py`): indisponivel.

### Distribuição por ano

| Ano | Matriz | Hab. | Exemplos | Questões | Efetivas | Real/Sint/Dest | F/M/D | Subtemas cobertos | Near-dup | Gab. incons. | Livres no banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1º | BNCC | 3 | 203 | 203 | 203 | 141/0/62 | 150/53/0 | 8/8 | 0 | 0 | 0 | 90 | **30** |
| 2º | BNCC+SAEB | 24 | 408 | 421 | 419 | 169/25/214 | 162/117/129 | 53/61 | 2 | 0 | 5 | 720 | **457** |
| 3º | BNCC | 3 | 167 | 167 | 166 | 123/0/44 | 111/52/4 | 6/6 | 1 | 0 | 0 | 90 | **26** |
| 4º | BNCC | 1 | 58 | 58 | 58 | 47/0/11 | 27/23/8 | 4/4 | 0 | 0 | 0 | 30 | **2** |
| 5º | SAEB | 22 | 247 | 263 | 249 | 67/19/161 | 79/71/97 | 58/75 | 14 | 0 | 1 | 660 | **438** |
| 9º | SAEB | 26 | 352 | 369 | 351 | 172/17/163 | 129/97/126 | 86/98 | 17 | 1 | 1 | 780 | **433** |

### As 15 piores habilidades (maior fração da meta faltando)

| # | Ano | Hab. | Descrição | Efetivas | Meta | Déficit | F/M/D | Subtemas | Livres no banco |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2º | H07 | Compor OU Decompor números naturais de até 3 ordens por mei… | 3 | 30 | **27** | 0/2/1 | 1/1 | 0 |
| 2 | 2º | H09 | Identificar a classificação OU Classificar objetos ou repre… | 3 | 30 | **27** | 1/1/1 | 1/1 | 0 |
| 3 | 2º | H17 | Identificar sequência de acontecimentos relativos a um dia. | 3 | 30 | **27** | 1/0/2 | 1/1 | 0 |
| 4 | 2º | H04 | Comparar OU Ordenar quantidades de objetos (até 2 ordens). | 4 | 30 | **26** | 1/1/2 | 1/1 | 0 |
| 5 | 2º | H05 | Comparar OU Ordenar números naturais, de até 3 ordens, com … | 4 | 30 | **26** | 2/1/1 | 1/1 | 0 |
| 6 | 5º | H02 | Identificar a ordem ocupada por um algarismo OU seu valor p… | 4 | 30 | **26** | 1/2/1 | 1/1 | 0 |
| 7 | 5º | H07 | Inferir OU Descrever atributos ou propriedades comuns que o… | 4 | 30 | **26** | 2/1/1 | 1/1 | 0 |
| 8 | 2º | H01 | Reconhecer o que os números naturais indicam em diferentes … | 5 | 30 | **25** | 2/0/3 | 1/1 | 0 |
| 9 | 2º | H13 | Identificar a localização OU a descrição/esboço do deslocam… | 5 | 30 | **25** | 1/1/3 | 3/3 | 0 |
| 10 | 2º | H19 | Ler/Identificar OU Comparar dados estatísticos ou informaçõ… | 5 | 30 | **25** | 1/2/2 | 0/2 | 0 |
| 11 | 5º | H10 | Identificar a localização OU a descrição/esboço do deslocam… | 5 | 30 | **25** | 0/2/3 | 2/3 | 0 |
| 12 | 5º | H11 | Descrever OU Esboçar o deslocamento de pessoas e/ou objetos… | 5 | 30 | **25** | 1/1/3 | 2/3 | 0 |
| 13 | 2º | H10 | Inferir os elementos ausentes em uma sequência de números n… | 6 | 30 | **24** | 1/1/4 | 2/2 | 0 |
| 14 | 5º | H14 | Reconhecer/nomear figuras geométricas planas (polígonos, ci… | 6 | 30 | **24** | 3/1/2 | 2/5 | 0 |
| 15 | 5º | H17 | Identificar horas em relógios analógicos OU Associar horas … | 6 | 30 | **24** | 3/1/2 | 2/2 | 1 |

## 2. Critério da meta (como o déficit é calculado)

1. **Mínimo de 30 questões efetivas por habilidade.** O prompt de treino é condicionado em (ano, habilidade, dificuldade); 30 dá 10 por dificuldade, abaixo disso o modelo tende a decorar exemplos em vez de aprender o padrão. O banco SAEB real tem só 8 itens por descritor, então 30 mantém a âncora real relevante (~25%) sem ser engolida pelos sintéticos.
2. **Pelo menos 3 por subtema da taxonomia.** Com 1 exemplo o modelo copia o molde; 3 é o mínimo para variar contexto e estrutura dentro do subtema. Habilidade com K subtemas tem meta ≥ 3·K.
3. **Equilíbrio entre as 3 dificuldades:** a meta é arredondada para múltiplo de 3 e o alvo por dificuldade é meta/3. Desequilíbrio ensina o modelo a ignorar a dificuldade pedida no prompt.
4. Só contam questões **efetivas**: questão com gabarito não-inconsistente e que não é near-dup de outra da mesma habilidade. Gabarito errado ensina errado; near-dup não ensina nada novo.
5. **Déficit = max(meta − efetivas, Σ faltas por dificuldade, Σ faltas por subtema).** Cada questão nova tem uma dificuldade e um subtema escolhidos livremente; por isso o mínimo que cumpre os três critérios ao mesmo tempo é o maior dos três, não a soma.

Limites conhecidos: o subtema vem do classificador léxico de `diversidade.py` (pode errar em enunciados atípicos — veja a coluna 'outros'); `check_consistency` só verifica contas explícitas e **não detecta erro conceitual** (ex.: o erro central do caso 9º H17 auditado, "tem uma propriedade de X ⇒ é X"). "ok" aqui é necessário, não suficiente.

## 3. Alertas

### Gabarito inconsistente (a conta da resolução aponta outra alternativa ou nenhuma) — 1

- **9º H06**: 1 — `MT9017MH06TD` (resposta_fora_das_alternativas)

### Desequilíbrio de dificuldade (alguma fora de 15–60%, n≥6; ou dificuldade ausente) — 22

- **1º EF01MA02**: F/M/D = 37/10/0 (fora: {'Fácil': 0.79, 'Difícil': 0.0})
- **1º EF01MA06**: F/M/D = 55/17/0 (fora: {'Fácil': 0.76, 'Difícil': 0.0})
- **1º EF01MA08**: F/M/D = 58/26/0 (fora: {'Fácil': 0.69, 'Difícil': 0.0})
- **2º EF02MA03**: F/M/D = 29/12/1 (fora: {'Fácil': 0.69, 'Difícil': 0.02})
- **2º H01**: F/M/D = 2/0/3 (fora: {'n_pequeno': 5})
- **2º H02**: F/M/D = 0/2/5 (fora: {'Fácil': 0.0, 'Difícil': 0.71})
- **2º H07**: F/M/D = 0/2/1 (fora: {'n_pequeno': 3})
- **2º H10**: F/M/D = 1/1/4 (fora: {'Difícil': 0.67})
- **2º H17**: F/M/D = 1/0/2 (fora: {'n_pequeno': 3})
- **2º H18**: F/M/D = 2/5/7 (fora: {'Fácil': 0.14})
- **2º H21**: F/M/D = 0/0/10 (fora: {'Fácil': 0.0, 'Moderado': 0.0, 'Difícil': 1.0})
- **3º EF03MA01**: F/M/D = 30/10/4 (fora: {'Fácil': 0.68, 'Difícil': 0.09})
- **3º EF03MA03**: F/M/D = 33/29/0 (fora: {'Difícil': 0.0})
- **3º EF03MA07**: F/M/D = 48/13/0 (fora: {'Fácil': 0.79, 'Difícil': 0.0})
- **4º EF04MA04**: F/M/D = 27/23/8 (fora: {'Difícil': 0.14})
- **5º H01**: F/M/D = 7/2/7 (fora: {'Moderado': 0.12})
- **5º H08**: F/M/D = 1/3/3 (fora: {'Fácil': 0.14})
- **5º H10**: F/M/D = 0/2/3 (fora: {'n_pequeno': 5})
- **5º H16**: F/M/D = 1/3/5 (fora: {'Fácil': 0.11})
- **9º H12**: F/M/D = 1/3/3 (fora: {'Fácil': 0.14})
- **9º H17**: F/M/D = 1/3/3 (fora: {'Fácil': 0.14})
- **9º H24**: F/M/D = 13/2/4 (fora: {'Fácil': 0.68, 'Moderado': 0.11})

### Subtemas da taxonomia sem nenhum exemplo (K≥2) — 24

- **2º H03**: 1/2 vazios — registro_lingua_materna
- **2º H11**: 2/5 vazios — trapezio, poligono
- **2º H12**: 1/5 vazios — cone
- **2º H18**: 1/4 vazios — calendario
- **2º H19**: 2/2 vazios — tabela_dupla, tabela_simples
- **2º H20**: 1/3 vazios — grafico_pictorico
- **5º H01**: 4/5 vazios — naturais_6_ordens, fracao, decimal, registro_lingua_materna
- **5º H06**: 2/5 vazios — multiplicacao, divisao
- **5º H09**: 2/4 vazios — adicao, seq_figuras
- **5º H10**: 1/3 vazios — mapas_croquis
- **5º H11**: 1/3 vazios — plantas
- **5º H12**: 1/5 vazios — cone
- **5º H14**: 3/5 vazios — circulo, triangulo, trapezio
- **5º H20**: 1/2 vazios — chance_igual
- **5º H21**: 2/2 vazios — tabela_dupla, tabela_simples
- **9º H01**: 3/4 vazios — fracao, decimal, registro_lingua_materna
- **9º H03**: 2/5 vazios — taxas_sucessivas, taxa_percentual
- **9º H05**: 1/4 vazios — multiplo
- **9º H13**: 1/2 vazios — sistema
- **9º H18**: 1/4 vazios — corda
- **9º H19**: 1/6 vazios — volume_medida
- **9º H21**: 1/5 vazios — poligono
- **9º H23**: 1/8 vazios — tabela_dupla
- **9º H24**: 1/9 vazios — tabela_simples

### Near-duplicatas altas (≥10% das questões) — 10

- **2º H06**: 20% (2 questões)
- **5º H03**: 24% (4 questões)
- **5º H04**: 10% (1 questões)
- **5º H06**: 17% (9 questões)
- **9º H03**: 12% (2 questões)
- **9º H06**: 22% (4 questões)
- **9º H07**: 14% (3 questões)
- **9º H08**: 22% (2 questões)
- **9º H09**: 31% (4 questões)
- **9º H10**: 25% (2 questões)

### Gabarito sem verificação automática (≥80% não verificável): exige validação conceitual — 37

- **2º H01**: 4/5 não verificáveis
- **2º H02**: 6/7 não verificáveis
- **2º H04**: 4/4 não verificáveis
- **2º H05**: 4/4 não verificáveis
- **2º H07**: 3/3 não verificáveis
- **2º H09**: 3/3 não verificáveis
- **2º H11**: 10/10 não verificáveis
- **2º H12**: 15/15 não verificáveis
- **2º H13**: 5/5 não verificáveis
- **2º H14**: 12/12 não verificáveis
- **2º H16**: 10/11 não verificáveis
- **2º H17**: 3/3 não verificáveis
- **2º H18**: 14/14 não verificáveis
- **2º H21**: 8/10 não verificáveis
- **5º H02**: 4/4 não verificáveis
- **5º H07**: 4/4 não verificáveis
- **5º H08**: 7/7 não verificáveis
- **5º H10**: 5/5 não verificáveis
- **5º H12**: 17/17 não verificáveis
- **5º H13**: 15/15 não verificáveis
- **5º H14**: 6/6 não verificáveis
- **5º H17**: 6/6 não verificáveis
- **5º H18**: 11/11 não verificáveis
- **5º H20**: 8/8 não verificáveis
- **5º H21**: 5/6 não verificáveis
- **9º H01**: 19/19 não verificáveis
- **9º H04**: 8/8 não verificáveis
- **9º H07**: 21/21 não verificáveis
- **9º H10**: 8/8 não verificáveis
- **9º H11**: 10/12 não verificáveis
- **9º H13**: 11/13 não verificáveis
- **9º H14**: 13/13 não verificáveis
- **9º H15**: 11/12 não verificáveis
- **9º H16**: 17/18 não verificáveis
- **9º H17**: 7/7 não verificáveis
- **9º H25**: 8/10 não verificáveis
- **9º H26**: 10/12 não verificáveis

### Classificador não reconhece o subtema (≥30% em 'outros') — 16

- **1º EF01MA06**: 23/72 em 'outros'
- **2º EF02MA05**: 27/79 em 'outros'
- **2º H06**: 3/10 em 'outros'
- **2º H19**: 5/5 em 'outros'
- **2º H20**: 3/10 em 'outros'
- **2º H21**: 5/10 em 'outros'
- **3º EF03MA01**: 34/44 em 'outros'
- **3º EF03MA03**: 39/62 em 'outros'
- **3º EF03MA07**: 35/61 em 'outros'
- **4º EF04MA04**: 22/58 em 'outros'
- **5º H04**: 5/10 em 'outros'
- **5º H21**: 6/6 em 'outros'
- **9º H05**: 13/18 em 'outros'
- **9º H13**: 12/13 em 'outros'
- **9º H23**: 6/20 em 'outros'
- **9º H25**: 6/10 em 'outros'

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

- **1º EF01MA02**: banco só tem Fácil, Moderado; a meta pede {'Difícil': 10} a mais nas faixas ausentes
- **1º EF01MA06**: banco só tem Fácil, Moderado; a meta pede {'Difícil': 10} a mais nas faixas ausentes
- **1º EF01MA08**: banco só tem Fácil, Moderado; a meta pede {'Difícil': 10} a mais nas faixas ausentes
- **2º H21**: banco só tem Difícil; a meta pede {'Fácil': 10, 'Moderado': 10} a mais nas faixas ausentes
- **3º EF03MA03**: banco só tem Fácil, Moderado; a meta pede {'Difícil': 10} a mais nas faixas ausentes
- **3º EF03MA07**: banco só tem Fácil, Moderado; a meta pede {'Difícil': 10} a mais nas faixas ausentes

### Letra da resposta correta desbalanceada (fora de 10–30% no global) — 1

- Global {'A': 438, 'B': 386, 'C': 372, 'D': 239, 'E': 46}; fora da faixa: {'E': 0.031}. Na injeção, sortear a posição da correta de modo uniforme e não usar 'Nenhuma das anteriores' como preenchimento fixo do E.

### Origem desconhecida (codigo_item fora do banco, sem flag de sintético/destilado) — 0

Nenhum.

## 4. Detalhe por habilidade

Colunas: Ex = exemplos; Q = questões; Ef = efetivas; R/S/D = real do banco/sintético/destilado; F/M/D = dificuldade (exemplos); Sub = subtemas cobertos/K; Dup = taxa de near-dup; Gab ok/inc/nv = gabarito ok/inconsistente/não verificável; Banco = itens totais/textuais/livres.

### 1º ano

| Hab. | Descrição | Ex | Q | Ef | R/S/D | F/M/D | Sub | Vazios | Dup | Gab ok/inc/nv | Banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EF01MA02 | Contar de maneira exata ou aproximada, utilizando diferentes estratég… | 47 | 47 | 47 | 47/0/0 | 37/10/0 | 1/1 | — | 0% | 47/0/0 | 50/50/0 | 30 | **10** |
| EF01MA06 | Construir fatos fundamentais da adição e utilizá-los em procedimentos… | 72 | 72 | 72 | 47/0/25 | 55/17/0 | 1/1 | — | 0% | 70/0/2 | 50/50/0 | 30 | **10** |
| EF01MA08 | Resolver e elaborar problemas de adição e de subtração, envolvendo nú… | 84 | 84 | 84 | 47/0/37 | 58/26/0 | 6/6 | — | 0% | 80/0/4 | 50/50/0 | 30 | **10** |

### 2º ano

| Hab. | Descrição | Ex | Q | Ef | R/S/D | F/M/D | Sub | Vazios | Dup | Gab ok/inc/nv | Banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EF02MA03 | Comparar quantidades de objetos de dois conjuntos, por estimativa e/o… | 42 | 42 | 42 | 42/0/0 | 29/12/1 | 1/1 | — | 0% | 42/0/0 | 50/45/0 | 30 | **9** |
| EF02MA05 | Construir fatos básicos da adição e subtração e utilizá-los no cálcul… | 79 | 79 | 79 | 47/0/32 | 45/22/12 | 2/2 | — | 0% | 71/0/8 | 50/50/0 | 30 | **0** |
| EF02MA06 | Resolver e elaborar problemas de adição e de subtração, envolvendo nú… | 81 | 81 | 81 | 46/0/35 | 37/26/18 | 6/6 | — | 0% | 75/0/6 | 49/49/0 | 30 | **0** |
| H01 | Reconhecer o que os números naturais indicam em diferentes situações:… | 5 | 5 | 5 | 1/0/4 | 2/0/3 | 1/1 | — | 0% | 1/0/4 | 8/1/0 | 30 | **25** |
| H02 | Identificar a posição ordinal de um objeto ou termo em uma sequência … | 7 | 7 | 7 | 3/0/4 | 0/2/5 | 1/1 | — | 0% | 1/0/6 | 8/3/0 | 30 | **23** |
| H03 | Escrever números naturais de até 3 ordens em sua representação por al… | 9 | 9 | 9 | 5/0/4 | 3/3/3 | 1/2 | registro_lingua_materna | 0% | 2/0/7 | 8/5/0 | 30 | **21** |
| H04 | Comparar OU Ordenar quantidades de objetos (até 2 ordens). | 4 | 4 | 4 | 1/0/3 | 1/1/2 | 1/1 | — | 0% | 0/0/4 | 8/1/0 | 30 | **26** |
| H05 | Comparar OU Ordenar números naturais, de até 3 ordens, com ou sem sup… | 4 | 4 | 4 | 2/0/2 | 2/1/1 | 1/1 | — | 0% | 0/0/4 | 8/2/0 | 30 | **26** |
| H06 | Calcular o resultado de adições ou subtrações, envolvendo números nat… | 8 | 10 | 8 | 5/1/2 | 2/2/4 | 2/2 | — | 20% | 9/0/1 | 8/5/0 | 30 | **22** |
| H07 | Compor OU Decompor números naturais de até 3 ordens por meio de difer… | 3 | 3 | 3 | 2/0/1 | 0/2/1 | 1/1 | — | 0% | 0/0/3 | 8/3/0 | 30 | **27** |
| H08 | Resolver problemas de adição ou de subtração, envolvendo números natu… | 54 | 65 | 65 | 6/24/24 | 16/18/20 | 6/6 | — | 0% | 62/0/3 | 8/7/0 | 30 | **0** |
| H09 | Identificar a classificação OU Classificar objetos ou representações … | 3 | 3 | 3 | 3/0/0 | 1/1/1 | 1/1 | — | 0% | 0/0/3 | 8/4/0 | 30 | **27** |
| H10 | Inferir os elementos ausentes em uma sequência de números naturais or… | 6 | 6 | 6 | 0/0/6 | 1/1/4 | 2/2 | — | 0% | 4/0/2 | 8/0/0 | 30 | **24** |
| H11 | Reconhecer/nomear figuras geométricas planas (círculo, quadrado, retâ… | 10 | 10 | 10 | 1/0/9 | 2/2/6 | 3/5 | trapezio, poligono | 0% | 0/0/10 | 8/3/2 | 30 | **20** |
| H12 | Reconhecer/nomear figuras geométricas espaciais (cubo, bloco retangul… | 15 | 15 | 15 | 2/0/13 | 3/6/6 | 4/5 | cone | 0% | 0/0/15 | 8/2/0 | 30 | **15** |
| H13 | Identificar a localização OU a descrição/esboço do deslocamento de pe… | 5 | 5 | 5 | 0/0/5 | 1/1/3 | 3/3 | — | 0% | 0/0/5 | 8/0/0 | 30 | **25** |
| H14 | Comparar comprimentos, capacidades ou massas OU Ordenar imagens de ob… | 12 | 12 | 12 | 0/0/12 | 3/3/6 | 3/3 | — | 0% | 0/0/12 | 8/0/0 | 30 | **18** |
| H15 | Relacionar valores de moedas e/ou cédulas do sistema monetário brasil… | 8 | 8 | 8 | 0/0/8 | 2/2/4 | 2/2 | — | 0% | 2/0/6 | 8/3/3 | 30 | **22** |
| H16 | Estimar/Inferir medida de comprimento, capacidade ou massa de objetos… | 11 | 11 | 11 | 1/0/10 | 5/2/4 | 3/3 | — | 0% | 1/0/10 | 8/1/0 | 30 | **19** |
| H17 | Identificar sequência de acontecimentos relativos a um dia. | 3 | 3 | 3 | 1/0/2 | 1/0/2 | 1/1 | — | 0% | 0/0/3 | 8/1/0 | 30 | **27** |
| H18 | Identificar datas, dias da semana, ou meses do ano em calendário OU E… | 14 | 14 | 14 | 1/0/13 | 2/5/7 | 3/4 | calendario | 0% | 0/0/14 | 8/2/0 | 30 | **16** |
| H19 | Ler/Identificar OU Comparar dados estatísticos ou informações express… | 5 | 5 | 5 | 0/0/5 | 1/2/2 | 0/2 | tabela_dupla, tabela_simples | 0% | 2/0/3 | 8/0/0 | 30 | **25** |
| H20 | Ler/Identificar OU Comparar dados estatísticos expressos em gráficos … | 10 | 10 | 10 | 0/0/10 | 3/3/4 | 2/3 | grafico_pictorico | 0% | 4/0/6 | 7/0/0 | 30 | **20** |
| H21 | Ler/Identificar OU Comparar dados estatísticos expressos em gráficos … | 10 | 10 | 10 | 0/0/10 | 0/0/10 | 3/3 | — | 0% | 2/0/8 | 1/0/0 | 30 | **20** |

### 3º ano

| Hab. | Descrição | Ex | Q | Ef | R/S/D | F/M/D | Sub | Vazios | Dup | Gab ok/inc/nv | Banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EF03MA01 | Ler, escrever e comparar números naturais de até a ordem de unidade d… | 44 | 44 | 44 | 34/0/10 | 30/10/4 | 1/1 | — | 0% | 37/0/7 | 37/37/0 | 30 | **6** |
| EF03MA03 | Construir e utilizar fatos básicos da adição e da multiplicação para … | 62 | 62 | 61 | 47/0/15 | 33/29/0 | 2/2 | — | 2% | 59/0/3 | 50/50/0 | 30 | **10** |
| EF03MA07 | Resolver e elaborar problemas de multiplicação (por 2, 3, 4, 5 e 10) … | 61 | 61 | 61 | 42/0/19 | 48/13/0 | 3/3 | — | 0% | 58/0/3 | 50/45/0 | 30 | **10** |

### 4º ano

| Hab. | Descrição | Ex | Q | Ef | R/S/D | F/M/D | Sub | Vazios | Dup | Gab ok/inc/nv | Banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EF04MA04 | Utilizar as relações entre adição e subtração, bem como entre multipl… | 58 | 58 | 58 | 47/0/11 | 27/23/8 | 4/4 | — | 0% | 55/0/3 | 50/50/0 | 30 | **2** |

### 5º ano

| Hab. | Descrição | Ex | Q | Ef | R/S/D | F/M/D | Sub | Vazios | Dup | Gab ok/inc/nv | Banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H01 | Escrever números racionais (naturais de até 6 ordens, representação f… | 16 | 16 | 16 | 5/0/11 | 7/2/7 | 1/5 | naturais_6_ordens, fracao, decimal, registro_lingua_materna | 0% | 4/0/12 | 8/6/0 | 30 | **14** |
| H02 | Identificar a ordem ocupada por um algarismo OU seu valor posicional … | 4 | 4 | 4 | 4/0/0 | 1/2/1 | 1/1 | — | 0% | 0/0/4 | 8/6/0 | 30 | **26** |
| H03 | Calcular o resultado de adições ou subtrações envolvendo números natu… | 13 | 17 | 13 | 2/1/10 | 6/2/5 | 4/4 | — | 24% | 11/0/6 | 8/4/0 | 30 | **17** |
| H04 | Calcular o resultado de multiplicações ou divisões envolvendo números… | 9 | 10 | 9 | 1/2/6 | 4/2/3 | 2/2 | — | 10% | 8/0/2 | 8/1/0 | 30 | **21** |
| H05 | Resolver problemas de adição ou de subtração, envolvendo números natu… | 27 | 27 | 27 | 6/0/21 | 7/8/12 | 8/8 | — | 0% | 20/0/7 | 8/7/0 | 30 | **8** |
| H06 | Resolver problemas de multiplicação ou de divisão, envolvendo números… | 41 | 52 | 43 | 7/16/18 | 15/16/10 | 3/5 | multiplicacao, divisao | 17% | 44/0/8 | 8/7/0 | 30 | **9** |
| H07 | Inferir OU Descrever atributos ou propriedades comuns que os elemento… | 4 | 4 | 4 | 4/0/0 | 2/1/1 | 1/1 | — | 0% | 0/0/4 | 8/5/0 | 30 | **26** |
| H08 | Inferir o padrão ou a regularidade de uma sequência de números natura… | 7 | 7 | 7 | 6/0/1 | 1/3/3 | 2/2 | — | 0% | 0/0/7 | 8/6/0 | 30 | **23** |
| H09 | Inferir os elementos ausentes em uma sequência de números naturais or… | 12 | 12 | 12 | 4/0/8 | 4/3/5 | 2/4 | adicao, seq_figuras | 0% | 8/0/4 | 8/4/0 | 30 | **18** |
| H10 | Identificar a localização OU a descrição/esboço do deslocamento de pe… | 5 | 5 | 5 | 0/0/5 | 0/2/3 | 2/3 | mapas_croquis | 0% | 0/0/5 | 8/0/0 | 30 | **25** |
| H11 | Descrever OU Esboçar o deslocamento de pessoas e/ou objetos em repres… | 5 | 5 | 5 | 0/0/5 | 1/1/3 | 2/3 | plantas | 0% | 2/0/3 | 8/0/0 | 30 | **25** |
| H12 | Reconhecer/nomear figuras geométricas espaciais (prismas, pirâmides, … | 17 | 17 | 17 | 6/0/11 | 4/6/7 | 4/5 | cone | 0% | 0/0/17 | 8/6/0 | 30 | **13** |
| H13 | Relacionar figuras geométricas espaciais (prismas retos, pirâmides re… | 15 | 15 | 15 | 5/0/10 | 5/4/6 | 6/6 | — | 0% | 0/0/15 | 8/5/0 | 30 | **15** |
| H14 | Reconhecer/nomear figuras geométricas planas (polígonos, circunferênc… | 6 | 6 | 6 | 4/0/2 | 3/1/2 | 2/5 | circulo, triangulo, trapezio | 0% | 0/0/6 | 8/4/0 | 30 | **24** |
| H15 | Medir OU Comparar perímetro de figuras planas desenhadas em malha qua… | 9 | 9 | 9 | 0/0/9 | 3/2/4 | 3/3 | — | 0% | 2/0/7 | 8/0/0 | 30 | **21** |
| H16 | Medir OU Comparar área de figuras planas desenhadas em malha quadricu… | 9 | 9 | 9 | 0/0/9 | 1/3/5 | 3/3 | — | 0% | 5/0/4 | 8/0/0 | 30 | **21** |
| H17 | Identificar horas em relógios analógicos OU Associar horas em relógio… | 6 | 6 | 6 | 2/0/4 | 3/1/2 | 2/2 | — | 0% | 0/0/6 | 8/3/1 | 30 | **24** |
| H18 | Determinar o horário de início, o horário de término ou a duração de … | 11 | 11 | 11 | 5/0/6 | 3/3/5 | 3/3 | — | 0% | 0/0/11 | 8/5/0 | 30 | **19** |
| H19 | Relacionar valores de moedas e/ou cédulas do sistema monetário brasil… | 6 | 6 | 6 | 0/0/6 | 2/1/3 | 2/2 | — | 0% | 2/0/4 | 8/0/0 | 30 | **24** |
| H20 | Identificar, entre eventos aleatórios, aqueles que têm menor, maior o… | 8 | 8 | 8 | 6/0/2 | 3/2/3 | 1/2 | chance_igual | 0% | 0/0/8 | 8/6/0 | 30 | **22** |
| H21 | Ler/Identificar OU Comparar dados estatísticos expressos em tabelas (… | 6 | 6 | 6 | 0/0/6 | 1/2/3 | 0/2 | tabela_dupla, tabela_simples | 0% | 1/0/5 | 8/0/0 | 30 | **24** |
| H22 | Ler/Identificar OU Comparar dados estatísticos expressos em gráficos … | 11 | 11 | 11 | 0/0/11 | 3/4/4 | 4/4 | — | 0% | 3/0/8 | 8/0/0 | 30 | **19** |

### 9º ano

| Hab. | Descrição | Ex | Q | Ef | R/S/D | F/M/D | Sub | Vazios | Dup | Gab ok/inc/nv | Banco | Meta | Déficit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H01 | Escrever números racionais (representação fracionária ou decimal fini… | 19 | 19 | 19 | 8/0/11 | 7/6/6 | 1/4 | fracao, decimal, registro_lingua_materna | 0% | 0/0/19 | 8/8/0 | 30 | **11** |
| H02 | Resolver problemas de adição, subtração, multiplicação, divisão, pote… | 20 | 20 | 20 | 7/1/12 | 10/7/3 | 7/7 | — | 0% | 8/0/12 | 8/8/0 | 30 | **10** |
| H03 | Porcentagem, acréscimos, decréscimos e taxas sucessivas. | 14 | 16 | 14 | 7/1/6 | 7/3/4 | 3/5 | taxas_sucessivas, taxa_percentual | 12% | 5/0/11 | 8/8/0 | 30 | **16** |
| H04 | Comparar, ordenar ou aproximar números reais. | 8 | 8 | 8 | 8/0/0 | 2/3/3 | 1/1 | — | 0% | 0/0/8 | 8/8/0 | 30 | **22** |
| H05 | Múltiplos, divisores, MDC e MMC. | 18 | 18 | 18 | 8/0/10 | 6/4/8 | 3/4 | multiplo | 0% | 4/0/14 | 8/8/0 | 30 | **12** |
| H06 | Cálculo com números reais: adição, subtração, multiplicação e divisão. | 14 | 18 | 13 | 7/2/5 | 6/3/5 | 4/4 | — | 22% | 11/1/6 | 8/8/0 | 30 | **17** |
| H07 | Representar ou associar frações a representações pictóricas. | 16 | 21 | 18 | 5/11/0 | 8/4/4 | 1/1 | — | 14% | 0/0/21 | 8/6/0 | 30 | **12** |
| H08 | Identificar frações equivalentes. | 7 | 9 | 7 | 6/1/0 | 2/3/2 | 1/1 | — | 22% | 3/0/6 | 8/6/0 | 30 | **23** |
| H09 | Converter entre representações de números racionais positivos (fraçõe… | 9 | 13 | 9 | 6/1/2 | 2/4/3 | 3/3 | — | 31% | 6/0/7 | 8/8/0 | 30 | **21** |
| H10 | Resolver uma equação polinomial de 1º grau. | 8 | 8 | 6 | 7/0/1 | 2/2/4 | 1/1 | — | 25% | 0/0/8 | 8/8/0 | 30 | **24** |
| H11 | Inferir equações, inequações ou sistemas de 1º grau que representam s… | 12 | 12 | 12 | 6/0/6 | 4/3/5 | 3/3 | — | 0% | 2/0/10 | 8/8/0 | 30 | **18** |
| H12 | Cálculo do valor numérico de expressões algébricas | 7 | 7 | 7 | 7/0/0 | 1/3/3 | 1/1 | — | 0% | 3/0/4 | 8/8/0 | 30 | **23** |
| H13 | Resolver problemas com sistemas de equações de 1º grau (duas incógnit… | 13 | 13 | 13 | 8/0/5 | 4/4/5 | 1/2 | sistema | 0% | 2/0/11 | 8/8/0 | 30 | **17** |
| H14 | Relacionar objetos tridimensionais às suas planificações ou vistas. | 13 | 13 | 13 | 5/0/8 | 4/3/6 | 7/7 | — | 0% | 0/0/13 | 5/5/0 | 30 | **17** |
| H15 | Ângulos em polígonos, retas paralelas e elementos notáveis (cevianas). | 12 | 12 | 12 | 4/0/8 | 4/3/5 | 3/3 | — | 0% | 1/0/11 | 5/5/1 | 30 | **18** |
| H16 | Propriedades dos triângulos. | 18 | 18 | 18 | 7/0/11 | 5/6/7 | 5/5 | — | 0% | 1/0/17 | 8/8/0 | 30 | **12** |
| H17 | Classificação de triângulos e quadriláteros quanto aos lados ou ângul… | 7 | 7 | 7 | 6/0/1 | 1/3/3 | 2/2 | — | 0% | 0/0/7 | 7/7/0 | 30 | **23** |
| H18 | Elementos da circunferência e do círculo | 14 | 14 | 14 | 7/0/7 | 4/5/5 | 3/4 | corda | 0% | 5/0/9 | 7/7/0 | 30 | **16** |
| H19 | Medidas e conversão de unidades | 19 | 19 | 19 | 7/0/12 | 9/4/6 | 5/6 | volume_medida | 0% | 7/0/12 | 8/8/0 | 30 | **11** |
| H20 | Perímetro de figuras planas. | 18 | 18 | 18 | 7/0/11 | 4/5/9 | 5/5 | — | 0% | 8/0/10 | 8/8/0 | 30 | **12** |
| H21 | Área de figuras planas | 15 | 15 | 15 | 7/0/8 | 6/3/6 | 4/5 | poligono | 0% | 5/0/10 | 8/8/0 | 30 | **15** |
| H22 | Volume de prismas retos e cilindros retos | 10 | 10 | 10 | 7/0/3 | 3/3/4 | 2/2 | — | 0% | 5/0/5 | 8/8/0 | 30 | **20** |
| H23 | Leitura e interpretação de tabelas e gráficos. | 20 | 20 | 20 | 4/0/16 | 10/5/5 | 7/8 | tabela_dupla | 0% | 7/0/13 | 7/5/0 | 30 | **11** |
| H24 | Representar dados em listas, tabelas ou gráficos. | 19 | 19 | 19 | 6/0/13 | 13/2/4 | 8/9 | tabela_simples | 0% | 5/0/14 | 7/7/0 | 30 | **14** |
| H25 | Probabilidade de eventos aleatórios equiprováveis | 10 | 10 | 10 | 7/0/3 | 2/3/5 | 2/2 | — | 0% | 2/0/8 | 8/8/0 | 30 | **20** |
| H26 | Média, moda e mediana. | 12 | 12 | 12 | 8/0/4 | 3/3/6 | 3/3 | — | 0% | 2/0/10 | 8/8/0 | 30 | **18** |

Detalhes completos (contagem por subtema, falta por dificuldade/subtema, pares de near-dup, códigos livres no banco, itens inconsistentes) estão em `outputs/relatorio_base.json`.

<!-- secao-preservada:inicio -->
## 5. Calibração dos agentes (validador e revisor)

Medição com chamadas **reais** à API Maritaca antes de usar os juízes na injeção e na auditoria. Dados completos, item a item, em `outputs/calibracao_agentes.json`. O texto exato dos prompts de cada rodada está em `outputs/agentes/calibracao_prompts/<versão>.json`, e cada chamada fica registrada em `outputs/agentes/uso_api_calibracao.jsonl`. Ferramenta: `src/calibrar_agentes.py`.

**Conjuntos**
- **cal20**: as 20 questões do 9º H17 (R1/R2) com o rótulo humano do `Log.txt`. São 5 para aprovar (R1-Q1, R1-Q7, R2-Q1, R2-Q5, R2-Q9) e 15 para reprovar.
- **real30**: 30 itens do `DB/questoes.db`, convertidos pelo mesmo `build_example` do treino.
  - Exclusões: itens com imagem, com descrição de imagem, com remissão a figura no texto ou sem justificativa. Restaram 490 elegíveis.
  - Sorteio com semente 42: 6 itens de geometria plana e 24 estratificados por ano.
  - Foram avaliados 29. O 30º (REAL-MT9067MH14MT) ficou de fora porque restavam 2 chamadas no orçamento e um item custa até 3.

**Regra de aceite.** Uma questão entra só se o validador **e** o revisor aprovarem. Na matriz de confusão, a classe positiva é "aprovar":
- VP = boa aprovada;
- FN = boa reprovada;
- FP = ruim aprovada;
- VN = ruim reprovada.

### Rodadas

| Rodada | O que mudou | Chamadas | Tokens (entrada/saída) | cal20: validador | cal20: revisor | cal20: par | Reais: o par rejeita |
|---|---|---|---|---|---|---|---|
| r0 `25d590a6b75a` | prompts originais | 43 | 70.903 / 32.798 | VP3 FN2 FP0 VN15 | VP4 FN1 FP0 VN15 | VP2 FN3 FP0 VN15 | — |
| r1 `a0f80d090f96` | validador com a regra da classe mais específica; a permutação é suspensa quando o enunciado repete as alternativas rotuladas | 21 | 48.546 / 17.115 | VP4 FN1 FP0 VN15 | = r0 (cache) | VP3 FN2 FP0 VN15 | — |
| r2 `b1604b6e3e76` | status **G** no validador e reparo de sintaxe do JSON. Nesta rodada foram refeitos 8 itens de hierarquia; os outros 12 vieram de r1 | 94 (9 cal + 85 reais) | 137.693 / 43.541 | VP5 FN0 FP0 VN15 | VP4 FN1 FP0 VN15 | VP4 FN1 FP0 VN15 | **14/29 = 48,3%** (IC95 0,31–0,66) |
| r3 (regra no código, mesmos prompts) | no revisor, C3/dificuldade passa a ser **aviso** e não veta | **0** (todas as respostas vieram do cache) | — | VP5 FN0 FP0 VN15 | VP4 FN1 FP0 VN15 | **VP4 FN1 FP0 VN15** | **5/29 = 17,2%** (IC95 0,08–0,35) |

**Total gasto: 158 das 160 chamadas** do orçamento, com 257.142 tokens de entrada e 93.454 de saída. Por papel:
- validador fase 1 (sabia-4-thinking): 77 chamadas, 182.535 / 61.461 tokens;
- validador fase 2 (sabia-4): 32 chamadas, 12.958 / 1.733 tokens;
- revisor (sabia-4): 49 chamadas, 61.649 / 30.260 tokens.

Todos os tokens vieram do campo `usage` da API; nenhum foi estimado. Nenhuma chamada falhou por erro HTTP.

### Configuração final (r2 + regra r3)

**cal20**
- O par reprova as ruins em **15/15** (IC95 de Wilson 0,80–1,00) e aprova as boas em **4/5** (IC95 0,38–0,96).
- Nenhuma ruim foi aceita, nem pelo par nem por algum juiz sozinho.
- Kappa entre validador e revisor: 0,857.
- A boa rejeitada é R2-Q1 (8-10-12, escaleno). O revisor marcou "Obtusângulo" como indeterminado porque não aplicou a comparação c² × a² + b². Isso ficaria F: 144 < 164.

**Reais (29)**
- O validador rejeita **2/29**. As duas rejeições têm defeito real:
  - **MT5023MH05MT**: o gabarito do banco está errado. 2.450 − 1.728 = 722, que é a alternativa B; o banco marca A = 732.
  - **MT5060DH20TD**: faltam os dados das três brincadeiras.
- O revisor rejeita **5/29**: as duas acima, mais MT9050 e MT9084 (hierarquia: "paralelogramo/retângulo/losango também são corretos") e MT90107 ("volume do cubo é conteúdo do 7º ano").
- Falsa rejeição do par:
  - bruta: 5/29 = 17,2%;
  - descontando as 2 rejeições corretas: **3/27 = 11,1%** (IC95 0,04–0,28).
- Fica abaixo do limite de 20%, mas o IC é largo.

### O que cada ajuste corrigiu (critério geral, sem reconhecer item)

1. **Enunciado com as alternativas rotuladas.** R2-Q9 traz "A) Escaleno B) Isósceles…" dentro do enunciado. A permutação anti-viés embaralhava só a lista, e o validador julgou letras trocadas. Agora a permutação é suspensa quando o enunciado tem 2 ou mais rótulos "X) ". Na amostra, só R2-Q9 é afetado.
2. **Pergunta pelo nome da figura.** "Como se chama um triângulo de lados iguais?" tem um único nome: equilátero. Isósceles também é verdade, mas é mais geral. Em r1 o validador entendeu a regra, mas se recusou a rotular como "F" algo verdadeiro e marcou "I". Em r2 ganhou o status **G** ("verdadeira, mas mais geral, do mesmo critério, que outra V").
   - O código aceita G só escrito como "G" ou "geral".
   - G nunca cria uma resposta: sem uma V garantida, o item reprova.
   - Classes de critérios diferentes, como retângulo (ângulos) e isósceles (lados), continuam valendo as duas V. Por isso R2-Q3 (90-45-45) segue reprovado.
3. **Reparo de sintaxe do JSON.** 3 das 126 respostas fase 1/revisor vinham com aspas escapadas fora de string ou com uma chave a mais ou a menos. Cada uma era uma chamada paga e descartada (fail-closed). O reparo mexe só na sintaxe; a interpretação continua estrita.
4. **C3 vira aviso.** O revisor marcou dificuldade incoerente em 13 dos 29 reais (45%), quase sempre dizendo "é EASY" de itens que o banco rotula Moderado ou Difícil. Desses, 9 foram reprovados só por isso.
   - Um juiz que discorda da referência em quase metade dos casos não serve de veto.
   - O aviso fica registrado (`avisos`), e o campo `difficulty` continua conferido pelo filtro determinístico.
   - A regra foi medida reaplicando as mesmas respostas, com custo zero e orçamento travado em 0 chamadas.
   - Contrapartida: a injeção deixa de barrar uma questão "Difícil" que o revisor considera fácil.

### Limitações (leia antes de escalar)

- **cal20 é otimista.** Os few-shots N1–N8 do prompt do validador descrevem os mesmos padrões de erro destas 20 questões. Os 15/15 medem se o validador aprendeu o padrão, não se generaliza. Os reais medem só a falsa rejeição. Falta um conjunto cego de ruins de outras habilidades.
- **r2 em cal20 foi parcial.** 12 itens foram herdados de r1, todos casos que a regra G não alcança: premissa impossível, conta errada, dado ausente e as boas já aprovadas. R1-Q5 entrou como "malformado" (rejeição fail-closed). Em r0, o mesmo item tinha sido reprovado por premissa impossível.
- **"Reais" não são itens oficiais do INEP.** São 534 do banco original (MT/TD) e 486 do lote L2-2026-09, e 2 dos 29 tinham defeito. Os casos de hierarquia (MT9050, MT9084) são discutíveis: com definição inclusiva estrita, a resposta não é única.
- **O revisor ainda não tem a regra G nem a comparação c² × a² + b².** Daí vêm R2-Q1 e MT9050/MT9084. A correção do prompt do revisor **não foi medida**, porque o orçamento acabou. Medir custa cerca de 49 chamadas: só o revisor, já que o validador vem do cache:
  `python src/calibrar_agentes.py --rodada r4 --conjunto ambos --reavaliar-revisor r3 --max-chamadas 50 --teto-global 210`
  Depois de editar `REVISOR_SISTEMA`, `--reavaliar-revisor` passa a chamar a API para os itens que não estão no cache.
- **n pequeno.** 15/15 tem limite inferior do IC95 de 0,80, e 4/5 vai de 0,38 a 0,96. O gate do `run_base_conhecimento.sh` (`auditar_base.py --calibrar`) gastaria até 80 chamadas para repetir esta medição. Use `PULAR_CALIBRACAO=1` se os prompts não mudarem.

**Reproduzir sem custo** (só cache): `python src/calibrar_agentes.py --rodada r3 --metricas`.
<!-- secao-preservada:fim -->

<!-- secao-preservada:inicio -->
## 6. Decisões de 2026-10-01 (regras novas do pipeline)

Prompts: versão **`d685c9b1e5fd`** (era `7570dee3dd46`). Mudar os prompts invalida o cache da auditoria e da calibração; isso é esperado. Os 59 registros de `data/auditoria_base.jsonl` e as 10 aceitas do piloto em `data/injecao_saeb.jsonl` foram julgados com os prompts antigos. Nesta etapa, **nenhuma chamada real** foi feita.

### Decisões do usuário

| # | Regra | Onde está |
|---|---|---|
| D1 | O **revisor** usa `sabia-4-thinking` (sem HF ou outra família) e **resolve às cegas**. Não recebe gabarito, resolução do autor nem `difficulty`. Vê as alternativas numa permutação própria, diferente da do validador. A comparação com o gabarito é feita no código. A resolução do autor continua sendo conferida pela fase 2 do validador; por isso o antigo C10 saiu e o revisor tem C1–C9. Os focos ficam separados: o validador cuida do rigor matemático; o revisor, da própria resolução e de habilidade, ano, distratores, clareza, contexto e português. | `REVISOR_SISTEMA`/`REVISOR_USUARIO`, `Agentes.revisar` |
| D2 | **Erro matemático confirmado sai da v3, qualquer que seja a origem** (inclusive item real, como o MT5023MH05MT). Conta como confirmado quando o **validador cego reprova** por motivo matemático **e** (o **revisor também reprova** por motivo matemático **ou** um filtro determinístico matemático reprova: consistência, geometria ou resolução que supõe dado). Item *incompleto* não conta como *errado*: dados insuficientes, dependência visual ou figura perdida vão para revisão humana. **Resolução vazia não é erro matemático**: o item não é removido por isso, e os códigos sobre a resolução são ignorados nesse caso. | `auditar_base.erro_matematico_confirmado`, `decidir`; lista em `outputs/agentes/removidos_v3.jsonl` (motivo e evidência de cada juiz); itens com resolução vazia em `outputs/agentes/resolucao_vazia_v3.jsonl` |
| D3 | Item **matematicamente correto** (os dois juízes `true`, nenhum filtro matemático reprovado) rotulado Moderado/Difícil que o revisor julga **Fácil** (`dificuldade_real`) é **rebaixado para Fácil**, sempre nos três lugares ao mesmo tempo: `meta.dificuldade`, `"difficulty": "EASY"` de cada questão e `"Dificuldade: Fácil."` do prompt de usuário. A regra só rebaixa para Fácil. Se um dos três lugares não bate, o item não é alterado. Na injeção, uma candidata pedida como Moderado/Difícil que sai Fácil entra como Fácil, desde que haja um slot Fácil pendente na habilidade. O slot Moderado/Difícil continua aberto. Sem lacuna de Fácil, ela é rejeitada (`dificuldade_real_facil_sem_lacuna`) e nunca entra com o rótulo errado. | `auditar_base.deve_rebaixar_para_facil`/`rebaixar_para_facil`, `injetar_questoes.tentar`/`construir_exemplo`; lista em `outputs/agentes/rotulos_corrigidos_v3.jsonl` (base e injetadas) |
| D4 | **Difícil também no 1º e no 3º ano**, por padrão (`ANOS_DIFICIL_LIBERADO`). Só para o professor, o prompt pede dificuldade genuína: mais etapas, dado intermediário, distratores próximos ou leitura mais elaborada, **sem conteúdo de ano posterior**. O revisor reprova em C2 (`ano_inadequado`) quando o item extrapola o ano. Outras dificuldades ausentes no banco, como Fácil e Moderado no 2º H21, continuam fora sem `--incluir-dificuldade-ausente`. | `injetar_questoes.lacunas`, `DIFICIL_ANOS_INICIAIS_ADDENDUM` |
| D5 | **Distratores estritamente falsos**: só o gabarito pode ser verdadeiro, por qualquer critério. O gerador recebe a instrução explícita (regras 3, 6 e 11). O validador julga V/F sem se restringir ao eixo perguntado; o exemplo novo [N9] é o 6-6-7 "quanto aos lados" com "Acutângulo" → `resposta_nao_unica` e `distrator_verdadeiro`. No revisor, C4 falha com distrator verdadeiro por outro eixo, e o código reprova qualquer status **G** (classe mais geral que também vale). C5 reprova qualquer dúvida razoável de interpretação. | `GERADOR_ADDENDUM`, `VALIDADOR_SISTEMA`, `REVISOR_SISTEMA`, `Agentes.revisar` |

**Consequência da D5 no gate, a confirmar.** No `Log.txt`, o R2-Q5 ("três lados iguais, como se chama?", com "Isósceles" entre as alternativas) estava aprovado. Pela D5, "Isósceles" é um distrator verdadeiro, então o item passa a **ruim** no gate (`ROTULOS_REVISTOS_D5` em `auditar_base.py`; o rótulo original fica em `rotulo_log`). Com isso o gate vira "reprova 16/16 ruins e aprova 4/4 boas". Para voltar ao rótulo humano, basta esvaziar o dict. O validador mantém o status G: a matemática do item continua consistente, e quem barra o item é o revisor. Itens reais nesse padrão (MT9050, MT9084) **não** saem pela D2, porque o validador aprova; vão para revisão humana.

### Correções técnicas da revisão do piloto

- **T1 Duplicata semântica.** Conta como duplicata a questão com os mesmos números no enunciado (multiconjunto canônico: "4.500" = 4500) **e** a mesma resposta (os números da alternativa correta, ou o texto normalizado dela). São necessários pelo menos 2 números. A busca é feita contra **toda** a base, a validação (`contamina_val`) e as aceitas. É determinística e roda antes dos juízes. Pega o mesmo 30-60-90 em outro contexto, que o Jaccard não vê. Funções: `assinatura_dados` e `e_duplicata_semantica`.
- **T2 Contexto forçado.** Virou veto. O critério C7 do revisor é bloqueante e traz exemplos ("R$ 4.500 centavos", "234 álbuns de 6 figurinhas"), e problema de contexto nunca vai só para `sugestoes`. Há também um filtro determinístico, `contexto_inverossimil`, que barra "R$ … centavos".
- **T3 Retomada sem perder questão paga.** Cada aceita é gravada primeiro num diário `<saída>.pendente`, de forma atômica. Depois vem o append com fsync, e só então o diário é apagado. Antes de qualquer append, `reparar_saida` faz três coisas:
  - move a linha truncada para `<stem>_truncadas.jsonl` (quarentena), reservando o `codigo_item` que der para ler do fragmento;
  - recupera do diário a aceita cortada, inteira e sem duplicar código;
  - reescreve o arquivo de forma atômica.
- **T4 Concentração de resposta.** Por habilidade, a mesma resposta não numérica não passa de `max(2, ⌈0,25·alvo⌉)` aceitas (`resposta_concentrada`). As respostas que chegam perto do limite são passadas ao professor ("crie uma situação cuja resposta seja OUTRA"); esse aviso não vai para o prompt de treino. `diversidade.py` não foi alterado.
- **T5 JSON dos juízes estrito: malformado = reprovação, sempre.** Só vale o objeto que termina no último "}", sem nenhum "{" depois. Não há mais reparo de sintaxe nem "içamento" de campos caídos dentro de `alternativas`. Isso fecha dois caminhos: a resposta truncada logo depois de `resposta_calculada` e o rascunho completo no raciocínio seguido da resposta final truncada. O gerador continua tolerante. Custo: as cerca de 3 em 126 respostas que o reparo salvava agora reprovam.
- **T6 Desempate de dificuldade.** O empate é sorteado com semente fixa por habilidade, então não começa sempre em Fácil.
- **Auditoria.** Os juízes rodam mesmo quando um filtro reprova, porque a D2 precisa do validador para confirmar. Um registro sem juízo do validador é refeito.

### Testes e dry-run

- `tests/test_agentes_questoes.py` e `tests/test_calibrar_agentes.py`: **88 passaram**. Todos usam cliente simulado.
  - D1: as mensagens enviadas ao revisor foram gravadas e conferidas; não trazem gabarito, resolução nem `difficulty`.
  - D2: remove o real com gabarito errado, grava a lista e mantém (listando) o item com resolução vazia.
  - D3: rebaixa nos três lugares e nada mais muda; só rebaixa para Fácil; vale também na injeção.
  - D4: libera Difícil no 1º e no 3º ano.
  - D5: alternativa verdadeira por outro eixo reprova; status G reprova no revisor.
  - T1 a T6: cobertos.
- Dry-run de ponta a ponta: `DIR_DRYRUN=/tmp/claude-1000/base2/dryrun ANOS="1º,3º,9º" META_POR_HAB=3 ORC_INJECAO=200 ./run_base_conhecimento.sh --fg`. Foram 573 chamadas simuladas e 0 reais.
  - Injeção: 38 aceitas, 4 delas Difícil no 1º e no 3º ano, 3 rebaixadas para Fácil, códigos únicos.
  - v3: 1.351 exemplos, com rótulo consistente nos três lugares em 1.351 de 1.351.
  - `train.jsonl`, `train_curado.jsonl` e `val*.jsonl` têm o mesmo sha1 de antes.
  - As remoções do dry-run são ruído do simulador, que "chuta" itens fora do padrão `a + b`, e não medem nada.

### Antes de escalar: recalibrar (chamadas reais, a autorizar)

Os dois juízes mudaram (D1, D5, T5), então a calibração precisa ser refeita. A fase 2 do validador não mudou e sai do cache.

```
python src/calibrar_agentes.py --rodada r5 --conjunto ambos --max-chamadas 110 --teto-global 300 \
    --nota "D1-D5/T5: revisor sabia-4-thinking cego; distratores estritamente falsos; JSON estrito"
```

Estimativa: cerca de 50 chamadas de fase 1, 50 do revisor e até 10 de fase 2, com thinking nos dois juízes. O ledger está em 180 chamadas, e o teto global padrão de 160 já foi ultrapassado na r4. A rodada passa a medir também `dificuldade_rotulo_x_real` e `rebaixaria_para_facil` nos reais, ou seja, o tamanho do efeito da D3. **Atenção:** na r3, o revisor sabia-4 disse "é EASY" para 45% dos reais. Se o thinking repetir esse padrão, a D3 rebaixa quase metade dos Moderado/Difícil reais.
<!-- secao-preservada:fim -->

<!-- secao-preservada:inicio -->
## 7. Recalibração com chamadas reais (r5 e r6, 2026-10-01)

Recalibração dos dois juízes depois das decisões D1–D5 e T5, feita com chamadas **reais** (Maritaca). Orçamento duro da etapa: **200 chamadas**. Foram gastas **193**: o ledger foi de 180 para 373, com teto global de 380.
- r5: 81 chamadas, 168.434 tokens de entrada e 71.543 de saída.
- r6: 112 chamadas, 272.841 de entrada e 84.080 de saída.

Nenhuma chamada falhou por HTTP, e todos os tokens vieram do campo `usage`. Os dados de cada item estão em `outputs/calibracao_agentes.json` (rodadas `r5` e `r6`), e o texto dos prompts em `outputs/agentes/calibracao_prompts/{d685c9b1e5fd,788dcddaf0e6}.json`.

Modelos: validador fase 1 `sabia-4-thinking`, fase 2 `sabia-4`, revisor `sabia-4-thinking` (cego, com permutação própria).

### Conjuntos (67 itens)

| Conjunto | Itens | Rótulo |
|---|---|---|
| cal20 | 20 do 9º H17 (Log.txt) | Rótulo D5: 16 ruins e 4 boas, porque o R2-Q5 passou a ruim. Também medido com o rótulo do Log (15 e 5). |
| real30 | 30 do banco (semente 42), agora incluindo o MT9067 | Presumidos bons. São descontados 2 adjudicados como defeituosos (MT5023 com gabarito errado; MT5060 sem dados). |
| defeitos | idx 1368, 1080, 1052 e 561 (ruins) e 881 (boa), de `data/train_curado.jsonl` (só leitura) | Adjudicação manual do piloto (`DEFEITOS_CONHECIDOS`). |
| d5 | 6 ruins (distrator verdadeiro por outro eixo) e 6 controles, em `data/calibracao_d5.json` | B1 e B2 são cópias literais de INJ-9-H17-F-00001 ("Acutângulo") e 00005 ("Escaleno"). B3–B6 são polígono de lados iguais, 91 "múltiplo de 7", 2,35 km = "235 000 cm" e 35°/55° "Agudos". Os controles têm o mesmo formato, com todos os distratores falsos. |

### Resultado

| Conjunto | Juiz | r5 (`d685c9b1e5fd`) | r6 (`788dcddaf0e6`) |
|---|---|---|---|
| cal20 (rótulo D5) | par | reprova 16/16 ruins, aprova 4/4 boas | **16/16 e 4/4** ¹ |
| cal20 (rótulo do Log) | par | 15/15 e 4/5 (R2-Q5) | 15/15 e 4/5 (R2-Q5) |
| reais | par | — (não rodado em r5) | rejeita 5/30 no bruto; **3/28 = 10,7% ajustado** (IC95 0,04–0,27) ² |
| defeitos | validador | 3/4 ruins (aceitou 1368); 881 rejeitado | **4/4**; 881 rejeitado |
| defeitos | revisor | 2/4 ruins (aceitou 1052 e 1368); 881 rejeitado | 2/4 (aceitou 1052 e 1368); **881 aprovado** |
| defeitos | par | 3/4 (aceitou 1368); 881 rejeitado | **4/4**; 881 rejeitado (pelo validador) |
| d5 | validador | 5/6 ruins, 5/6 controles (B5 aceito; G2 malformado) | 4/6 (aceitou B1 e B6); **6/6 controles** |
| d5 | revisor | 4/6 (aceitou B5 e B6), 6/6 controles | **6/6 e 6/6** ³ |
| d5 | par | 5/6 e 5/6 | **6/6 e 6/6** |

¹ Em r6 foram refeitos 7 itens do cal20: as 4 boas, o R2-Q5 e as 2 ruins que tinham vindo malformadas. As outras 13 ruins foram **herdadas de r5** (`--herdar r5`), por orçamento.

² Reais rejeitados pelo par:
- MT5023 e MT5060: os dois adjudicados como defeituosos.
- MT9050 e MT9084: hierarquia. "Paralelogramo", "retângulo" e "losango" são verdadeiros, o que pela D5 conta como distrator verdadeiro. São rejeições esperadas pela regra nova.
- EF04MA04-045: o revisor marcou habilidade desalinhada ("só subtração direta"). É a única rejeição contestável.

Descontando também as duas de hierarquia, a falsa rejeição fica em 1/26.

³ Das 6 reprovações do revisor no d5 em r6, a do B1 é fail-closed: a resposta veio malformada, não houve detecção.

**Geral (67 itens, r6 com os herdados):**
- kappa entre validador e revisor: **0,786** (era 0,195 na auditoria do piloto, com o revisor antigo).
- Taxa de reprovação: validador 41,8%, revisor 43,3%, par 47,8%.
- Só o validador reprova: DEF-1052, DEF-1368 e DEF-881.
- Só o revisor reprova: D5-B1 (malformado), D5-B6, R2-Q5 e EF04MA04-045.

**JSON malformado (T5):**
- validador: 4/37 em r5 e **0/54** em r6;
- revisor: 1/37 em r5 e 1/54 em r6.

### Metas: o que passou e o que não passou

- **Passou:**
  - cal20: 16/16 e 4/4 com o rótulo D5.
  - Falsa rejeição nos reais ≤ 15%: 10,7%.
  - D5 pelo par: 6/6 ruins reprovadas e 6/6 controles aprovados.
  - Os 4 defeituosos são reprovados pelo par.
- **Não passou:**
  1. **O revisor sozinho ainda aprova o DEF-1368** (hexágono suposto regular) **e o DEF-1052** ("Três centenas e sete unidades"). Nos dois casos ele escreve a regra certa e conclui errado. O par barra os dois porque o validador reprova, e no 1368 o filtro determinístico `resolucao_supoe_dado` também reprova.
  2. **O DEF-881 continua rejeitado pelo par.** O revisor agora acerta (quinta-feira), mas o validador erra a contagem um a um ("dia 9 quarta, 10 sexta", pulando a quinta). É um erro aritmético do modelo, não falta de regra: a regra de extremos está no prompt e ele a aplicou de forma errada.
  - Consequência pela D2: o 881 **não** é removido, porque o revisor não confirma o erro. Ele cai em "media" e vai para revisão humana.

### Ajuste feito (1 rodada, critério geral) e casos usados para ajustar

- **Regras** (casos DEF-881, DEF-1368, DEF-1052, D5-B5 e D5-B6), nos dois juízes:
  - ler cada alternativa sozinha, como afirmação sobre os dados: se for verdadeira, é V, mesmo fora do critério perguntado;
  - a mesma coisa dita de outra forma (outra unidade, decomposição, por extenso) é V;
  - o nome do polígono não implica que ele seja regular;
  - contagem de extremos: começa num dia e dura N dias → termina em início + N − 1; "N dias depois" → início + N.
  - Os exemplos do prompt usam números e figuras diferentes dos casos.
- **Formato** (casos D5-B1, D5-G2, R1-Q10 e R2-Q2): no validador, `alternativas` passou a ser plano (`{"A": "V"}`), com `motivos` à parte. Isso zerou os malformados da fase 1.
- **Fora da amostra do ajuste:** real30, cal20, D5-B2/B3/B4, D5-G1/G3–G6, DEF-1080 e DEF-561. Os resultados nos casos de ajuste são otimistas por construção.
- **Não houve 2ª rodada.** Medir de forma honesta exigiria refazer os dois juízes nos 67 itens (~140 chamadas), e restavam 7.

### Efeito medido da D3 (rebaixar para Fácil)

Nos 30 reais, o revisor thinking julgou assim os itens rotulados Moderado ou Difícil:
- Moderado → Fácil: 11 de 13;
- Difícil → Fácil: 3 de 5.

Com os dois juízes de acordo (`true`), a D3 rebaixaria **11 dos 30 reais (61% dos 18 Moderado/Difícil)**. É o mesmo padrão do sabia-4 na r3 (45%), agora mais forte. **Recomendação:** confirmar a D3 com o usuário antes de montar a v3 real, ou exigir concordância de um segundo julgamento de dificuldade. Do jeito atual, a base perde a maior parte dos Moderado/Difícil reais.

### Efeito medido da D2 nos reais (r6)

- O MT5023 (gabarito errado) é **removido**: os dois juízes deram `gabarito_errado`.
- O MT9084 (quadrado com "retângulo", "losango" e "paralelogramo" como distratores) também é **removido**, porque os dois deram `resposta_nao_unica`. É coerente com a D5, mas é um item real do banco: vale confirmar.
- O MT9050 vai para revisão humana: o validador marcou `pergunta_ambigua` (definição de trapézio), que não é código de erro confirmado.

**Reproduzir sem custo** (só cache): `python src/calibrar_agentes.py --rodada r6 --metricas --herdar r5 --conjunto todos`.
<!-- secao-preservada:fim -->

<!-- secao-preservada:inicio -->
## 8. Piloto 2 com chamadas reais (2026-10-01, prompts `788dcddaf0e6`)

Orçamento duro: **260 chamadas**. Gasto: **258**. A auditoria usou 161 e a injeção 97, rodando em paralelo. Nenhuma chamada falhou por HTTP e nenhum token foi estimado. Os números estão em `outputs/agentes/piloto2.json`, e o uso, chamada a chamada, em `outputs/agentes/uso_api_{auditoria,injecao}_piloto2.jsonl`.

**Mudanças no código** (sem mudar prompts, por isso a versão continua `788dcddaf0e6`):
- `auditar_base.py --idx-registrados` reaudita a mesma amostra.
- `--cache-juizes` reaproveita respostas já pagas dos juízes quando a mensagem é idêntica. Foram 9 acertos de cache: 7 do validador fase 1 e 2 da fase 2. O revisor não acerta, porque recebe o subtema.
- `--montar --previa` grava as listas sem escrever a v3.
- `injetar_questoes.py --meta-relativa` soma N às aceitas que já existem.
- Contagem por dificuldade.
- O motivo de rejeição do revisor passa a ser o 1º problema que veta (antes saía "dificuldade_incoerente", que é só aviso).
- Os testes deixaram de escrever em `outputs/agentes/removidos_v3.jsonl` real.

### Injeção (5 habilidades, alvo +5 cada)

| Habilidade | Pedida | Candidatos | Aceitas | Taxa (IC95) | Dificuldade real (revisor) nas aceitas | Rejeições |
|---|---|---|---|---|---|---|
| 1º EF01MA08 | Difícil (D4) | 4 | 3 | 0,75 (0,30–0,95) | Moderado 3 | subtema_divergente 1 |
| 3º EF03MA07 | Difícil (D4) | 6 | 2 | 0,33 (0,10–0,70) | Moderado 2 | subtema_divergente 2, duplicata_semantica 2 |
| 2º H07 | F 3, D 2 | 5 | 4 | 0,80 (0,38–0,96) | Fácil 2, Moderado 2 | validador 1 (falso: errou 100+20+5) |
| 5º H21 (tabela) | F 3, M 2 | 5 | 4 | 0,80 (0,38–0,96) | Fácil 2, Moderado 2 | subtema_divergente 1 |
| 9º H17 | M 6, D 6 | 12 | **1** | 0,08 (0,01–0,35) | Difícil 1 | revisor: distrator_verdadeiro 4, resposta_nao_unica 3; validador 1; menciona_figura 2; muito_longa 1 |
| **Total** | | **32** | **14** | **0,44 (0,28–0,61)** | | filtro 9, validador 2, revisor 7 |

**Aceitação por dificuldade pedida:**
- Fácil: 4 de 6;
- Moderado: 2 de 8;
- Difícil: 8 de 18.

**Rebaixadas para Fácil (D3): 0.** Nenhum Difícil do 1º/3º saiu Fácil, então o caso "sem lacuna de Fácil" não ocorreu.

**Custo:**
- 3,03 chamadas por candidato;
- **6,93 chamadas, 17,9 mil tokens (13,0 mil de entrada e 4,9 mil de saída) e 76 s por aceita.**
- Foram 1.060 s de parede para 14 aceitas, com a auditoria rodando em paralelo.

**Revisão manual das 14 aceitas:**
- A matemática está correta em todas.
- **Ponto de atenção da D4:** o revisor julgou **Moderado** as 5 Difícil aceitas do 1º e do 3º ano, e também as 2 Difícil do 2º H07. Mesmo assim, elas entram rotuladas como Difícil, porque a D3 só rebaixa para Fácil.
- O `INJ-1-EF01MA08-D-00017` (3×12 + 2×15) é duvidoso para o 1º ano, mas o revisor aprovou o C2.

**9º H17:**
- O gerador continua pondo como distrator a classe mais geral ou a de outro eixo (Isósceles ou Acutângulo para o equilátero, Paralelogramo para o losango), apesar da regra 6.
- Pela D5, o revisor reprova certo em 6 dos 9 casos.
- Falsos negativos dos juízes: 3 em 9. São dois 50-60-70 (status incoerente do revisor) e o 100+20+5 do validador.

### Reauditoria dos 59 (mesma amostra)

| | Antes (`7570dee3dd46`) | Depois (`788dcddaf0e6`) |
|---|---|---|
| alta / media / baixa | 49 / 6 / 4 | **45 / 7 / 7** |
| real | 20 / 2 / 3 | 19 / 2 / 4 |
| destilado | 28 / 3 / 1 | 25 / 4 / 3 |
| validador × revisor (TT / TF / FF / FT) | 52 / 3 / 1 / 3 | 46 / 6 / 4 / 3 |
| kappa | 0,195 | **0,385** |
| JSON malformado | — | 0 / 0 |

**Transições:** alta→media 4; media→baixa 3; o resto ficou igual.

**Custo da reauditoria:** 2,73 chamadas, 6,5 mil tokens e 24 s por item.

**Dificuldade (revisor × rótulo):**
- Moderado→Fácil: 13 de 19;
- Difícil→Fácil: 9 de 17;
- Difícil→Moderado: 8 de 17.

**Prévia da montagem** (`--montar --previa`; a v3 **não** foi escrita; `train*`/`val*` têm o mesmo sha1):
- **D2:** 4 removidos por erro matemático confirmado: 561, 599, 881 e 1080. A regra anterior (destilado media/baixa com problema matemático) remove mais 4: 1052, 1127, 1187 e 1368.
  - **Remoção falsa: 881.** "Terça + 10 dias" termina na quinta, e os **dois** juízes contaram sexta. É erro correlacionado (mesma família de modelo), justamente o que a D2 não detecta.
  - **Remoção discutível: 1187.** O revisor marcou "Dez mil e quatrocentos e oito" como G.
- **D3:** **17 rótulos rebaixados para Fácil** de 36 Moderado/Difícil auditados: 47% (IC95 0,32–0,63). São 10 reais e 7 destilados.
- 38 itens com resolução vazia foram listados e não removidos.
- 7 itens vão para revisão humana.
- Listas: `outputs/agentes/{removidos_v3,rotulos_corrigidos_v3,resolucao_vazia_v3}.jsonl` e `montagem_v3_previa.json`.

### Projeção da execução completa (números do piloto 2)

**Base da projeção:**
- Injeção: 1.362 questões (déficit 1.386 menos 24 já injetadas). A faixa vem do IC95 de Wilson da taxa de aceite.
- Auditoria: as 1.422 questões restantes, com 2,73 a 3,0 chamadas por item.

| | Chamadas | Tokens | Horas (1 processo) |
|---|---|---|---|
| Injeção (otimista / central / pessimista) | 6,8 / 9,4 / 14,6 mil | 17,6 / 24,4 / 37,8 M | 21 / 29 / 44 |
| Auditoria da base | 3,9–4,3 mil | 9,3 M | 9,5 |
| **Total** | **10,7 / 13,5 / 18,9 mil** | **27 / 34 / 47 M** | **30 / 38 / 54** (≈15 / 19 / 27 com 2 processos) |

**Por que subiu em relação à projeção anterior** (9,4–14,5 mil chamadas; 9–14 h): o revisor agora é thinking e cego. Ele leva ~14 s por chamada, contra 3 s antes, e passou a reprovar de verdade.

**O pessimista não é teto:** o 9º H17 custou cerca de 37 chamadas por aceita (estimativa pelo funil: 12 candidatos, 1 aceita), e habilidades de classificação geométrica vão puxar a média para cima.
<!-- secao-preservada:fim -->

<!-- secao-preservada:inicio -->
## 9. Correções da revisão independente do piloto 2 (2026-10-01)

A auditoria independente do piloto 2 encontrou 8 problemas altos. Todos foram reproduzidos antes da correção (`/tmp/claude-1000/base2/fix/repro_antes.txt`) e têm teste que falha no código anterior e passa no atual (`TestRevisaoPiloto2`, 18 testes). Os prompts **não mudaram** (versão continua `788dcddaf0e6`); mudou a decisão calculada no código (`VERSAO_JUIZES = j2-d5-validador-g`) e os filtros (`VERSAO_FILTROS = f3-aritmetica`).

**Chamadas reais nesta etapa: 17** (orçamento duro de 60):
- 12 no rejulgamento das 4 aceitas do 5º H04 do piloto 1;
- 5 para medir a regra do G no validador com os 5 candidatos reais do 9º H17 que violam a D5.

A reauditoria dos 59 com a decisão nova custou **0 chamadas**: as 170 respostas vieram do cache.

### O que mudou no critério

| Achado (alto) | Correção |
|---|---|
| O validador aceitava o status G (classe mais geral verdadeira), contrariando a D5. | G reprova no validador (`distrator_verdadeiro`), como já fazia no revisor. Medido com a API nos 5 candidatos do piloto 2 que o validador tinha aprovado: **5/5 agora reprovados**; em 3 deles só o G mudou o veredito. |
| Os dois juízes erram juntos (881: "terça + 10 dias" → os dois contaram sexta; 1348: 5/11 = "0,45" → os dois aprovaram). | `verificar_aritmetica`: conta exata em 3 padrões estreitos (dias da semana, fração → decimal, "outra forma de representar N" com somas). Ele vale nos dois sentidos: **contradiz** = filtro matemático, que confirma a D2 sozinho; **confirma** = a D2 nunca remove por erro dos juízes, e o item vai para revisão humana. Na base inteira, val e injeção ele disparou 11 vezes, todas certas na conferência manual (9 confirma, 2 contradiz: 561 e 1348), e 0 vezes nos 29 candidatos rejeitados do piloto. |
| `decidir()` removia destilado "baixa" ou "media com problema matemático" sem a confirmação da D2 (1187 correto saiu com `evidencia: null`). | Só sai erro matemático **confirmado**. Reprovado sem confirmação fica e vai para `auditoria_revisao_humana.jsonl` com a evidência. `--manter-media` não tem mais efeito. |
| A D2 tratava como erro matemático a ambiguidade de redação (599) e não via a resolução que supõe o dado ausente (1368). | Quando o validador diz que a falta de resposta única vem de `pergunta_ambigua`, isso é redação e vai para revisão humana. `dados_insuficientes` + filtro `resolucao_supoe_dado` vira `resolucao_usa_dado_ausente`, que é erro. |
| O `--montar` copiava todas as aceitas, inclusive as 10 do piloto 1 (prompts antigos), 6 delas violando a D5. | Só entra injetada julgada com os **prompts atuais**, na injeção ou em `injetar_questoes.py --rejulgar` (`meta.rejulgamento`), e que o verificador aritmético não contradiz. As outras vão para `outputs/agentes/injetadas_excluidas_v3.jsonl`. |

**Também corrigido (médio, código barato):**
- **Cache dos juízes:** só guarda e só devolve resposta interpretável (`resposta_interpretavel`), no cache da auditoria e no da calibração. Antes, um JSON malformado ficava "nao_avaliado" para sempre a custo zero. Dentro do cache, a auditoria não faz mais a pré-checagem de orçamento, então a reauditoria com `--max-chamadas 0` roda só do cache.
- **Trava por saída da injeção** (`<saída>.lock`, flock): um 2º processo na mesma saída para antes de gastar. Antes os dois calculavam o mesmo código e o 2º abortava no `.pendente`.
- **Registros de auditoria:** passam a guardar o `status` V/F/I/G de cada alternativa, nas letras originais. O "detalhe" do LLM cita letras da permutação dele, o que confundiu a leitura da evidência do 561.
- **Pipeline (`run_base_conhecimento.sh`):** ganhou a etapa de rejulgamento antes da montagem (`ORC_REJULGAMENTO=60`). Em modo real, a injeção grava as respostas dos juízes no cache.

### Dados

**`data/injecao_saeb.jsonl`: 24 → 17 aceitas.** Saíram 7 para `outputs/injecao_rejeitadas.jsonl`, com motivo `auditoria_independente`, nota por item e o exemplo inteiro:
- violam a D5: `INJ-9-H17-F-00001` (o "Acutângulo"), `F-00003`, `F-00005`, `D-00009`, `F-00011` e `M-00013`;
- viola a D4 (conteúdo de ano posterior no 1º ano): `INJ-1-EF01MA08-D-00017`.

Os códigos ficam reservados. A calibração D5 (B1/B2) passou a ler a cópia literal das rejeitadas.

**Rejulgamento das 4 aceitas do 5º H04 do piloto 1** (12 chamadas): as 4 foram aprovadas pelos dois juízes atuais e ganharam `meta.rejulgamento`. Pela D3, `INJ-5-H04-D-00006` (18.480 ÷ 24) foi rebaixada para Fácil, porque o revisor julgou Fácil. **Esse rebaixamento é discutível:** é uma divisão de 5 algarismos por 2 no 5º ano.

**`INJ-9-H17-D-00016`** (13-14-15 "quanto aos lados") foi rebaixada para Fácil nos três lugares, por decisão da auditoria independente (`fonte: auditoria_independente`).

**Prévia da v3** (não escrita):

| | Antes (seção 8) | Agora |
|---|---|---|
| Removidos | 8 (4 pela D2, 4 pela regra antiga) | **4, todos D2 com evidência**: 561 (verificador: A e C dão 245), 1080, 1348 (verificador: 5/11 ≠ 0,45), 1368 (resolução supõe "regular") |
| Removidos indevidamente | 881, 1187 (e 599 discutível) | nenhum dos três; 881 vai para revisão humana por `verificador_confirma_gabarito` |
| Revisão humana | 7 | 12 (inclui 881, 1052, 1127, 1187, 1429) |
| Rótulos rebaixados | 17 | 19 (17 da base + 00006 + 00016) |
| Injetadas na v3 | 24 (6 violando a D5) | 17 (0 violando a D5; 0 excluídas por versão) |
| Total v3 | 1.451 | 1.448 |

**1052 e 1127** (defeitos reais segundo a auditoria independente) **ficam na v3** e vão para revisão humana: só um juiz os reprovou, então a D2 não os confirma. Remover só com um juiz é exatamente o que tirou o 1187 correto.

### Correções de leitura das seções anteriores

- **Kappa:** a alta de 0,195 para 0,385 nos 59 **não** é distinguível do acaso (IC95 ≈ 0,03–0,69; P(Δ ≤ 0) ≈ 0,23). O 0,786 da calibração mistura duas versões de prompt, num conjunto com cerca de 45% de itens ruins. Nenhum dos dois prova que os juízes são independentes, e os erros conjuntos (881, 1348, 1387) mostram que não são.
- **Projeção (seção 8):**
  - Com injeção e auditoria em paralelo, o tempo de parede é o **máximo** das duas, não a soma dividida por 2: cerca de **29 h no cenário central e 44 h no pessimista**, não 19 e 27.
  - A injeção em si não pode rodar em 2 processos na mesma saída (agora barrado pela trava).
  - As lacunas vêm do relatório **antes** da D2/D3. Rebaixando cerca de 47% dos Moderado/Difícil (IC 0,32–0,63), a D3 cria um déficit novo de Moderado/Difícil de cerca de 365 itens (250–490), que a projeção não cobre. É preciso refazer o relatório sobre a v3 antes de dimensionar a escala.

### Pendências (não corrigidas aqui)

- **Ancoragem da dificuldade:** o revisor recebe "Dificuldade pedida" antes de julgar `dificuldade_real`. Ele disse Moderado para as 7 Difícil dos anos iniciais e Difícil para o 13-14-15, e na auditoria confirmou 0 de 17 Difícil. Tirar essa linha muda o prompt: nova versão, cache invalidado e recalibração (cerca de 140 chamadas). A D3 continua dependendo desse sinal enviesado.
- **Calibração com entrada diferente da produção:** `calibrar_agentes.julgar_item` passa `subtema=None`. O 881 foi aprovado pelo revisor na calibração e reprovado na auditoria.
- **Revisor "cego" recebe o subtema**, que em 8 itens da base é a própria resposta (ex.: "cilindro").
- **Contexto forçado:** o veto T2/C7 não disparou no 1387 (ônibus/ângulo inscrito), que segue rebaixado para Fácil.
- **INJ-3-EF03MA07-D-00025** resolve por divisão (EF03MA08) e fala em "outra parede" sem citar a primeira. A auditoria independente classificou como desvio leve, então o item foi mantido.
- **T1:** a assinatura numérica colide em problemas de números pequenos, com 109 grupos na base/val, e não cobre classificação com 0 ou 1 número.
- **T3:** a reserva de código lida de um fragmento truncado quase nunca dispara (o `codigo_item` fica no fim da linha). O `--montar` só avisa quando há `.pendente`.
- O arquivo `data/injecao_saeb.jsonl.lock` é criado pela trava: é vazio e pode ficar.
<!-- secao-preservada:fim -->

<!-- secao-preservada:inicio -->
## 10. Decisões de 2026-10-01 (tarde) e passo 1 sem custo de Maritaca

Prompts: versão **`3358bf312d22`** (era `788dcddaf0e6`); filtros `f4-h2-d5` (era `f3-aritmetica`). Chamadas nesta etapa: **0 à Maritaca** e **2 ao Gemini** (de 3 permitidas): 1 listagem de modelos e 1 resolução às cegas do MT9018MH04MT, com 2.876 tokens de entrada e 1.089 de saída (684 de raciocínio). Registro em `outputs/agentes/uso_api_gemini.jsonl`.

### Decisões do usuário

| # | Regra | Onde está |
|---|---|---|
| H1 | Questão matematicamente **errada sai** da v3. Remoções e correções valem só na montagem (arquivo novo); `train_curado.jsonl` continua com sha1 `5033cb42…`. | `auditar_base.montar` |
| H2 | **Arredondamento no ensino básico:** quando o valor exato é dízima ou irracional e o enunciado não pede outra precisão, a alternativa com **duas casas decimais**, arredondada ou truncada, é correta. Se duas alternativas casarem por essa regra, a resposta não é única. Caso real: **idx 1348** (5/11, gabarito "0,45") **fica**. | gerador (regra 12), validador e revisor ("Arredondamento"), `agentes_questoes.verificar_aritmetica` |
| H3 | Dos 12 itens da revisão humana, **só o MT9018MH04MT (idx 264) fica**, com a resolução comentada cadastrada e a dificuldade **Moderado → Fácil** nos três lugares. Os outros 11 **saem**. As decisões estão num arquivo **versionável**, que tem precedência sobre as regras automáticas. | `Doc/decisoes_humanas.json` (o `data/` está no `.gitignore`), `auditar_base.carregar_decisoes`/`aplicar_edicoes_humanas` |
| H4 | **Árbitro de outra família** (Google Gemini) é **obrigatório antes de qualquer remoção pela D2**. Ele resolve às cegas e o item só sai se ele também achar erro matemático. Se discordar, ou se ainda não tiver arbitrado, o item fica e vai para revisão humana. | `src/arbitro_gemini.py`, `auditar_base.py --arbitrar`, `decidir_com_arbitro`; etapa nova no `run_base_conhecimento.sh` (`ORC_ARBITRO=20`) |

**Troca de modelo do árbitro (registrada, não silenciosa).** O modelo pedido, `gemini-2.0-flash-thinking-exp-01-21`, **não existe mais** na lista da API (61 modelos, `outputs/agentes/arbitro_modelo.json`). O critério é: flash com thinking, estável e de versão mais próxima da pedida. Por esse critério, o substituto é **`gemini-2.5-flash`** (constante `MODELO_ARBITRO`, com o motivo em `MOTIVO_TROCA_MODELO`, e opção `--modelo-arbitro`). Há flash mais novos com thinking (3.5 a 3.8); escolher um deles fica a critério do usuário.

### Itens do passo 1

| # | Mudança | Medida |
|---|---|---|
| P1 | **Pré-filtro D5 de geometria antes dos juízes.** `verificador_geometria.verificar_d5` lê o detalhe do veredito e reprova quando outra alternativa também é V. São dois casos: a classe mais geral ("Isósceles" no equilátero, "Paralelogramo" no losango) e a classe verdadeira por outro eixo ("Acutângulo" numa pergunta "quanto aos lados"). Vale quando o veredito é "ok" e nas abstenções que só se devem a valores I. O `verificar_geometria` público não mudou (convenção 4). | Candidatas gravadas: **14 barradas**, todas violações reais da D5 (conferidas uma a uma). Isso daria **34 chamadas de juízes economizadas**: 24 no piloto 1 (7 aceitas que a auditoria independente tirou depois + 1 rejeitada) e 10 no piloto 2 (4 do 9º H17). **0 das 17 aceitas** foram barradas. No cal20 + d5, foram barradas **18 de 22 ruins** e **0 de 10 boas**; B3–B6 não são de geometria. Na base, é informativo: 4 reais (MT9050, MT9081, MT9083, MT9084) violam a D5, e a D2 não os remove sozinha. |
| P2 | **Taxonomia: multiplicação × divisão pela estrutura.** "cada … ao todo/total" passa a contar como multiplicação. "cada" dentro da pergunta, sem número até o "?", passa a contar como divisão. Saiu o "cada um" solto. As regex não usam lookbehind. | `build_taxonomia` regenerado: **mudaram só as 7 habilidades** que têm esses subtemas (3º EF03MA03/07, 4º EF04MA04, 5º H04/H06, 9º H02/H06). Foram 27 itens do banco reclassificados, conferidos (ex.: EF03MA07 multiplicação de 1 para 12). Falsa rejeição do 5º H04 no piloto 1: **4/11 → 0/11**. |
| P3 | **Revisor sem âncora:** a "Dificuldade pedida" saiu da mensagem. A `dificuldade_real` agora segue uma **rubrica absoluta por ano** (etapas, operação, tamanho dos números, leitura), com números e operações típicos de cada ano. O **subtema não é enviado** quando alguma alternativa está contida nele. | O subtema vaza em 15 itens da base: os 8 sólidos de H12/H13, "raio e diâmetro", "gráfico de colunas" e "retângulos e quadrados". A **D3 só usa registros julgados com a rubrica nova** (`rubrica_dificuldade`). Os 16 rebaixamentos da prévia anterior, que vinham do sinal ancorado, ficam **suspensos** até a recalibração (passo 2). |
| P4 | **Cache:** a resposta com `finish_reason` "length" ou "MAX_TOKENS" não entra no cache, mesmo que o JSON feche. Malformadas já ficavam fora desde a seção 9. Vale para o cache da auditoria, o da calibração e o do árbitro. | `aq.resposta_truncada`, `CacheRespostas.truncadas_nao_gravadas` |

**Também:**
- `refiltrar` passa a **absolver**: um filtro que deixou de reprovar (o 1348 com a H2) sai dos problemas, e o nível volta ao que os juízes deram.
- O `--montar` refiltra na hora os registros de outra versão de filtros (0 chamadas).
- A `GOOGLE_API_KEY` entra na limpeza de segredos e vai só no cabeçalho `x-goog-api-key`, nunca na URL.
- O `check_consistency` (contrato congelado) **não muda**: conta aproximada não é "a op b = r" exata, então o item fica não verificável, nunca acusado (teste no 1348).

### MT9018MH04MT (idx 264, 9º H04)

> Durante uma prova, a professora pediu aos alunos que organizassem os seguintes números em ordem crescente: 2,5; 2,45 e 2,75. Qual é a ordem correta?
> A) 2,75 - 2,5 - 2,45 · **B) 2,45 - 2,5 - 2,75** · C) 2,5 - 2,75 - 2,45 · D) 2,5 - 2,45 - 2,75 · E) Nenhuma das alternativas anteriores

Resolução cadastrada: "Para comparar números decimais, escrevemos todos com a mesma quantidade de casas decimais: 2,5 = 2,50; 2,45; 2,75. Os três têm a mesma parte inteira, 2, então comparamos a parte decimal em centésimos: 45 < 50 < 75. Logo, 2,45 < 2,5 < 2,75. Cuidado: 2,45 não é maior que 2,5 só porque tem mais algarismos depois da vírgula (2,45 tem 45 centésimos e 2,5 tem 50 centésimos). A ordem crescente é 2,45 - 2,5 - 2,75."

Conferência:
- **Script** (Fraction): a ordem exata é 2,45 < 2,5 < 2,75, que é a alternativa B. A, C e D são outras permutações, e E é falsa.
- **Gemini às cegas** (`gemini-2.5-flash`, `outputs/agentes/arbitro_mt9018.json`): respondeu B, com status B=V e as outras F, sem problemas e com confiança 1,0.

### Prévia da montagem (v3 não escrita em `data/`; arquivos em `/tmp/claude-1000/base3/previa/`)

| | Prévia anterior (seção 9) | Agora |
|---|---|---|
| Removidos | 4 pela D2 (561, 1080, 1348, 1368) | **11 por decisão humana**, **0 pela D2** |
| D2 aguardando árbitro (ficam, revisão humana) | — | **2**: 1080 e 1368 |
| 1348 | removido | **fica** (H2 + decisão "manter") |
| MT9018 (264) | revisão humana | **fica, editado** (resolução + Fácil) |
| Rótulos corrigidos | 19 | **1** (o 264); a D3 está suspensa (P3) |
| Resolução vazia listada | 38 | 37 |
| Injetadas na v3 | 17 | **0**: as 17 foram julgadas com os prompts `788dcddaf0e6` e precisam de rejulgamento (até 51 chamadas) |
| Base na v3 | 1.431 | **1.424** (+17 injetadas após o rejulgamento) |
| Déficit (relatório sobre a prévia) | 1.386 (F 444 · M 481 · D 448) | **1.393 (F 445 · M 485 · D 452)**, sem a D3 |

### Consequências para o passo 2

- **Cache:** como os prompts mudaram, o cache do validador fase 1 e o do revisor não valem mais. O da fase 2 continua válido, porque a mensagem é a mesma.
- **Custo da regressão:** a "regressão de qualidade" do passo 2 paga validador e revisor de novo: cerca de 37 itens × 2 a 3 chamadas.
- **MT9018MH04MT no conjunto de referência:** o código do banco tem "M" (Moderado), mas o usuário reclassificou o item para Fácil. Ele deve ficar fora do conjunto de dificuldade do passo 2, ou entrar com o rótulo humano.
- **Pendências da seção 9 resolvidas aqui:**
  - ancoragem da dificuldade (P3);
  - subtema que vaza a resposta (P3);
  - cache de resposta truncada (P4).
- **Continuam pendentes:** a calibração passa `subtema=None`, e o T1 colide em números pequenos.
<!-- secao-preservada:fim -->

<!-- secao-preservada:inicio -->
## 11. Passo 2: recalibração com chamadas reais (rodada r7, 2026-10-01)

Prompts `3358bf312d22` (sem a dificuldade pedida no revisor, rubrica absoluta, H2); filtros `f5-h2-d5-extenso`. Resultados em `outputs/calibracao_agentes.json` → `rodadas.r7` (itens, métricas e `passo2_resumo`).

**Chamadas pagas nesta etapa** (orçamento duro: 160 Maritaca + 40 Gemini):

| Provedor | Chamadas | Onde | Tokens (entrada / saída) | Registro |
|---|---|---|---|---|
| Maritaca | **121** | 45 revisor (referência de dificuldade) + 76 regressão (37 validador fase 1, 2 fase 2, 37 revisor; 10 acertos de cache da fase 2) | 331 mil / 112 mil | `uso_api_calibracao.jsonl`, ledger 373 → 494 |
| Gemini (`gemini-2.5-flash`) | **38** | 2 árbitro da D2 (1080, 1368) + 36 dificuldade | 33 mil / 73 mil | `uso_api_gemini.jsonl`, `calibracao_ledger_gemini.json`, `arbitragem_d2.jsonl` |

Nenhuma chamada falhou; nenhuma saiu truncada. Uma resposta do revisor veio malformada (MT5083MH17MT) e ficou fora do cache e das métricas.

### Referência de dificuldade (45 reais do banco)

- **Padrão confirmado no banco:** a letra depois do número no `codigo_item` (`MT9018MH04MT` → M) é a dificuldade; bate com `grau_resolucao` em 524 de 530 códigos nesse formato, mais 4 códigos malformados (ex.: `MT9020MfH07TD`) que ficam fora. Os 6 que divergem ficam fora. *(Corrigido na seção 12: antes dizia 526 de 532.)* Os itens do lote L2 (`EF..-L2-…`) não têm a letra.
- **Conjunto:** 15 F, 15 M, 15 D; 5 de cada no 2º, 5º e 9º ano (só esses anos têm o código com letra); habilidades diferentes primeiro; sem figura nem remissão visual. Fora: MT9018 (reclassificado por você), os 5 reais removidos por decisão humana, os 2 adjudicados defeituosos e os 4 reais que violam a D5.
- **Achado que muda a leitura:** cada habilidade do banco tem cerca de 1/3 de F, 1/3 de M e 1/3 de D (69 habilidades; 22 com 3/3/2, 17 com 2/3/3...). O rótulo do banco é **relativo, por cota dentro da habilidade**; a rubrica do revisor é **absoluta**. Há "Difícil" do banco que são de fato de um passo (2º H11: "a peça com três lados" = triângulo; 2º H17: "o que Sofia fez por último").

**Revisor novo (sem âncora) × banco** (n = 44; linhas = banco, colunas = revisor):

| banco \ revisor | Fácil | Moderado | Difícil |
|---|---|---|---|
| Fácil | 15 | 0 | 0 |
| Moderado | 12 | 2 | 0 |
| Difícil | 12 | 3 | 0 |

- Concordância exata 0,39; **kappa ponderado quadrático 0,09** (linear 0,08; simples 0,07).
- **Rebaixaria para Fácil 24 de 29 M/D = 83% (IC95 0,66–0,92)**: 2º ano 10/10, 5º 6/9, 9º 8/10. Nunca disse Difícil.
- Sem a âncora o revisor ficou mais "Fácil", não mais calibrado (no piloto 2, com a âncora, eram 47%).
- Veredito do revisor nesses reais: reprova 5 de 44 (11%).

**Gemini (mesma rubrica, às cegas)** em todos os M/D e em 6 F (n = 35):

| banco \ Gemini | Fácil | Moderado | Difícil |
|---|---|---|---|
| Fácil | 6 | 0 | 0 |
| Moderado | 9 | 4 | 1 |
| Difícil | 5 | 7 | 3 |

- Kappa quadrático 0,31; rebaixaria 14/29 = 48% (0,31–0,66). Kappa revisor × Gemini: 0,17.
- **Regra 2 de 2** (revisor E Gemini dizem Fácil): rebaixaria **12 de 29 = 41% (IC95 0,26–0,59)**. Confirma Fácil em 6 de 6 F.

### Regra D3 fixada: SUSPENSA

Critério **fixado antes dos dados** (`calibrar_agentes.regra_d3`): o revisor sozinho só rebaixa se, nos reais M/D, rebaixar no máximo 10% e o kappa quadrático for ≥ 0,60. Se não cumprir, a regra 2 de 2 só vale se ficar nos mesmos 10%.

| Regra | Rebaixamento falso nos M/D reais | Passa? |
|---|---|---|
| Revisor sozinho | 83% (kappa 0,09) | não |
| Revisor + Gemini (2 de 2) | 41% | não |

- **Implementado:** `auditar_base.REGRA_D3 = "suspensa"` (`--regra-d3`; a opção `revisor` continua disponível como escolha explícita). Na injeção (`injetar_questoes.py --regra-d3`, padrão `suspensa`), a aprovada entra com a dificuldade **pedida**; a `dificuldade_real` fica só registrada na meta. Antes ela era rebaixada ou rejeitada (`dificuldade_real_facil_sem_lacuna`). No rejulgamento, nada é rebaixado.
- **Itens reais:** o rótulo do banco **não** é sobrescrito por regra automática, em nenhuma regra (inclusive com `--regra-d3 revisor`). É a referência contra a qual o revisor foi medido, e o modelo aprende a convenção relativa do banco com eles. Só uma decisão humana troca o rótulo de um real (ex.: MT9018MH04MT).
- **Contrafactual** (0 chamadas, sobre a prévia): rebaixar 83% dos M/D não reais tiraria 355 rótulos e levaria o déficit de 1.393 para **1.460 (F 248 · M 576 · D 636)**; a regra 2 de 2 (41%) tiraria 174 e levaria a 1.404 (F 341 · M 524 · D 533). Com a D3 suspensa, **o déficit novo do rebaixamento é zero**.

### Regressão de qualidade (cal20 + defeitos + D5, 37 itens)

| | r6 (`788dcddaf0e6`) | r7 (`3358bf312d22`) |
|---|---|---|
| cal20 (rótulos D5), par | 0 ruins aceitas, 0 boas rejeitadas | 16/16, 4/4 (o validador sozinho deixou de aceitar R2-Q5) |
| D5, par | 6/6 ruins, 6/6 boas | 6/6 ruins, **5/6 boas** (D5-G2: revisor devolveu status incoerente, E = "I") |
| defeitos, par | 4/4 ruins, 0/1 boa (881) | **3/4 ruins: aceitou o 1052**; 0/1 boa (881) |
| **Pipeline (filtro exato + par)** | — | **0 ruins aceitas**; boas rejeitadas: D5-G2 e 881 |

- **O 1052 piorou:** "307 por extenso", com o distrator C "Três centenas e sete unidades", que também é 307. O validador novo passou a aprovar, junto com o revisor, que já aprovava na r6. Mexer de novo no prompt invalidaria o cache e custaria outra recalibração inteira.
  - **Correção (0 chamadas):** novo verificador exato `agentes_questoes._verificar_extenso`. Ele compara cada alternativa com a grafia canônica do número (masculina ou feminina), com a decomposição em ordens, com a soma ou com o numeral. Abstém-se no "e" depois de "mil" e em texto que não reconhece.
  - **Medido:** na base, val, injeção e rejeitadas, disparou em 6 itens, todos conferidos: 5 confirmam (MT2012, EF03MA01-015, 853, 1156, 1161) e 1 contradiz (o 1052). Ele se absteve nos 3 itens do conector do "mil" (889, 1187, 1188).
  - **Efeito no pipeline:** barra o 1052 antes dos juízes na injeção e, na auditoria, confirma a D2 sozinho (e o árbitro ainda tem de concordar).
- **881 ("terça + 10 dias")** voltou a ser reprovado pelo revisor (conta sexta). É o erro correlacionado conhecido. O verificador exato confirma o gabarito, então a D2 não o remove (está fora da v3 por decisão sua; ver a pendência abaixo).
- Os 20 itens que o filtro exato barra na regressão (geometria, D5 e aritmética) são todos ruins; **nenhuma boa foi barrada**.

### Árbitro da D2 (H4) — 1080 e 1368

O Gemini **discordou nos dois**, então nenhum dos dois sai pela D2; ficam na v3 e vão para revisão humana:
- **1080** ("quatro lados iguais" é quadrado?): o Gemini viu o defeito. Respondeu sem resposta única por falta do dado dos ângulos, e as alternativas ficaram I. Pelo critério da D2, dado ausente é item **incompleto**, não errado: revisão humana. A regra funcionou como escrita, mas o item é ruim.
- **1368** (ângulo interno do "hexágono" sem "regular"): o Gemini supôs regular e respondeu 120° (B), como o revisor; só o validador apontou o dado ausente.

### Prévia da montagem (v3 não escrita em `data/`; `/tmp/claude-1000/base3/previa2/`)

| | Seção 10 | Agora |
|---|---|---|
| Removidos | 11 humanos; D2 aguardando árbitro: 2 | **11 humanos, 0 pela D2** (árbitro discordou de 1080 e 1368: revisão humana) |
| Rótulos corrigidos | 1 (MT9018) | **1** (MT9018, humano); D3 suspensa: 0 automáticos |
| Base na v3 | 1.424 | **1.424** (+17 injetadas depois do rejulgamento) |
| Déficit | 1.393 (F 445 · M 485 · D 452) | **1.393 (F 445 · M 485 · D 452)**; déficit novo da D3 = 0 |

`train_curado.jsonl` continua com sha1 `5033cb42…`.

### Projeção da escala (atualizada)

**Base:**
- Injeção de 1.376 questões (déficit de 1.393 menos as 17 que voltam após o rejulgamento).
- Funil do piloto 2: 3,03 chamadas por candidato; aceite 0,44 (IC95 0,28–0,61); 2,6 mil tokens e 11 s por chamada. Nesta etapa a latência medida do revisor foi 10,9 s.
- Auditoria da base inteira (1.424; os 59 já auditados também precisam ser refeitos com os prompts novos): 2,73–3,0 chamadas, 6,5 mil tokens e 24 s por item.

| Cenário | Injeção | Auditoria | **Total de chamadas** | Tokens | **Parede (2 processos em paralelo = máximo)** |
|---|---|---|---|---|---|
| Otimista | 6,8 mil (21 h) | 3,9 mil (9,5 h) | **10,7 mil** | 27 M | **≈ 21 h** |
| Central | 9,5 mil (29 h) | 4,1 mil (9,5 h) | **13,6 mil** | 34 M | **≈ 29 h** |
| Pessimista | 14,9 mil (45 h) | 4,3 mil (9,5 h) | **19,2 mil** | 48 M | **≈ 45 h** |

**Além disso:**
- rejulgamento das 17 injetadas (até 51 chamadas Maritaca);
- árbitro Gemini para a D2 na base inteira: cerca de 50 chamadas, extrapolando os 2 em 59 da amostra (IC largo). Isso passa do "baixo volume" de 40 e precisa de orçamento próprio.

A D3 suspensa não acrescenta déficit. O pré-filtro P1 economiza chamadas sobretudo nas habilidades de classificação geométrica (34 nas candidatas gravadas); isso não foi descontado da projeção. O pessimista continua não sendo teto (9º H17: cerca de 37 chamadas por aceita).

### Pendências e decisões para você

1. **881 (DIST-H18-Moderado-00102)** está na sua lista de remoção, mas é **matematicamente correto**. O verificador exato confirma, e os dois sabiá erram juntos de novo na r7. Para mantê-lo, basta trocar a ação em `Doc/decisoes_humanas.json`.
2. **1080 e 1368** estão em revisão humana: o árbitro discordou da remoção automática. O 1080 é defeituoso (resposta não única); o 1368 depende de aceitar "hexágono" como regular.
3. **Item 889 (DIST-EF03MA01-Difícil-00110):** o gabarito é "dois mil quarenta e oito", mas a grafia usual é "dois mil **e** quarenta e oito", que é a alternativa B. O verificador se absteve (é o caso do conector do "mil"), então o item precisa de conferência humana.
4. **Dificuldade como sinal:** nenhum juiz reproduz a dificuldade relativa do banco. Para voltar a ter D3, o caminho que os dados indicam é uma régua **relativa por habilidade** (ordenar itens da mesma habilidade), não uma rubrica absoluta. Isso exigiria outro desenho e outra calibração.
5. **INJ-5-H04-D-00006** foi rebaixada para Fácil pela D3 antiga (seção 9) e continua Fácil no arquivo. Com a D3 suspensa, cabe a você decidir se ela volta a Difícil.
6. **D5-G2:** o revisor devolveu um JSON incoerente (disse C, mas os status não fecham). Ele reprova (fail-closed). Não foi refeito, para economizar chamadas.
<!-- secao-preservada:fim -->

<!-- secao-preservada:inicio -->
## 12. Correções da revisão do passo 2 (2026-10-01, noite)

Prompts `3358bf312d22` (sem mudança: os caches da r7 continuam valendo); filtros `f6-h2-d5-extenso-planilha`. Prévia da v3 em `/tmp/claude-1000/base3/previa3/` (nada escrito em `data/` além do registro de auditoria; `train_curado.jsonl` continua `5033cb42…`).

**Chamadas pagas nesta etapa** (teto: 30 Maritaca + 10 Gemini):

| Provedor | Chamadas | Onde | Tokens (entrada / saída) |
|---|---|---|---|
| Maritaca | **12** | 6 na auditoria dos idx 139, 254 e 401 (`auditar_base.py --idx`); 6 na rodada r8 da calibração (5 itens novos da referência + o MT5083MH17MT, que tinha vindo malformado) | 17,2 mil / 4,1 mil + 15,6 mil / 5,2 mil |
| Gemini (`gemini-2.5-flash`) | **7** | 2 árbitro da D2 (139, 401); 5 dificuldade na r8 (31 vieram do cache) | 5,7 mil / 1,6 mil + 3,6 mil / 10,5 mil |

### O que a revisão achou (confirmado) e o que foi feito

1. **17 itens reais com a fração trocada por data pela planilha** (`2025-06-03 00:00:00` no lugar de 3/6; ex.: MT90123MH25MT, "probabilidade de sair número par", gabarito A = data). Nenhum filtro via. Todos no 9º ano (H07, H08, H09, H25).
   - **Nova regra D6** (`agentes_questoes.alternativas_corrompidas_planilha`, `auditar_base.corrompido_planilha`): o item sai da v3 **sem juiz nem árbitro**, porque não é opinião sobre a matemática; o gabarito gravado é uma data e a questão não tem resposta (H1). Precedência: decisão humana > D6 > D2 com árbitro > D3.
   - A **proposta de restauração** (dia/mês: `2025-06-03` → `3/6`; confere com a resolução nos 17) vai na evidência de cada removido em `removidos_v3.jsonl`. Restaurar é decisão sua: uma decisão "manter" em `Doc/decisoes_humanas.json` vence a D6.
   - A D6 também barra na injeção, nas injetadas da montagem e na auditoria (filtro `alternativa_corrompida_planilha`; não é filtro matemático da D2).
2. **Os 5 "falsos" da referência de dificuldade eram defeitos de verdade.** Na r7, o revisor reprovou 5 de 44 reais, e a métrica contou isso como falsa rejeição. O revisor acertou nos 5: 3 têm datas (MT9073FH25TD, MT90123MH25MT, MT90125DH25MT) e 2 têm o gabarito do banco errado:
   - **MT9036DH12TD** (base, idx 401): 2·4² − 3·4 + 10 = 30 (D); o banco marca C = 26.
   - **MT9009DH03TD** (está em `val.jsonl` e `val_frozen_v1`, não na base): (900 − 135) × 1,05 ÷ 4 = 200,81; o banco marca C = R$ 212,25.
   - **Corrigido:** `calibrar_agentes.DEFEITOS_REAIS_CONFIRMADOS` e a detecção de datas tiram o item defeituoso **no sorteio**: o próximo da mesma fila (nível, ano) entra no lugar e os outros 40 sorteados não mudam. A métrica só usa a seleção atual. Os defeitos que o revisor reprovou aparecem como `revisor_rejeicao_correta_defeitos`.
   - **Itens reais com gabarito errado na base:** MT5023MH05MT (idx 139, já adjudicado) e MT9036DH12TD (idx 401) nunca tinham passado pela auditoria. Auditados agora: validador e revisor apontam `gabarito_errado` nos dois, e o **árbitro Gemini confirmou nos dois** (B e D). Saem pela D2. O MT5060DH20TD (idx 254, enunciado sem os dados) é incompleto, não errado: fica e vai para revisão humana.
3. **H2 com brechas** (`_verificar_fracao_decimal`):
   - "uma casa decimal" no **singular** não era reconhecido ("decimais?" não casa "decimal"), e 0,45 saía confirmado para "5/11 com uma casa decimal";
   - o valor **exato** pedido ("representação decimal exata") desligava nada: 0,45 saía confirmado como exato;
   - a aproximação **não pedida** fazia o verificador se abster mesmo com o valor exato entre as alternativas (3/8 com gabarito 0,38 e 0,375 presente).
   - Agora: o singular é reconhecido; com "exata", a regra das duas casas não vale; e, com o valor exato presente, a aproximação não pedida conta como falsa. Sem o exato, o verificador continua se abstendo. Na base, val e injeção os vereditos não mudaram (o 1348 continua confirmado).
4. **Árbitro "discorda" no 1080:** o Gemini também viu que não há resposta única, mas atribuiu à falta de dado. Agora o motivo é `d2_arbitro_ve_item_incompleto_revisao_humana`. O item continua na v3, em revisão humana.
5. **Documentação:** a docstring da D2 ainda citava o 1348 como erro. Os números "526 de 532" e "~50%" foram corrigidos para 524 de 530 (mais 4 códigos malformados) e 41% (12 de 29, r7).

### Referência de dificuldade só com itens bons (rodada r8)

Entraram MT9007FH03TD (F), MT9057MH12MT (M), MT9063DH21TD, MT9066DH22TD e MT9030DH10TD (D), todos do 9º ano e conferidos por conta. A cobertura do 9º H25 saiu junto com os itens corrompidos.

| banco \ revisor (n = 45) | Fácil | Moderado | Difícil |
|---|---|---|---|
| Fácil | 15 | 0 | 0 |
| Moderado | 12 | 3 | 0 |
| Difícil | 12 | 3 | 0 |

- Revisor: concordância exata 0,40; kappa quadrático **0,09**. **Rebaixaria 24 de 30 M/D = 80% (IC95 0,63–0,91)**: 2º 10/10, 5º 6/10, 9º 8/10.
- Falsa rejeição do revisor nos reais: **0 de 45** (IC95 0–0,08). Na r7 eram 5/44, todos defeitos.
- Gemini (n = 36): kappa quadrático 0,31; rebaixaria 14/30 = 47%.
- **Regra 2 de 2: 12 de 30 = 40% (IC95 0,25–0,58)**; confirma Fácil em 5 de 6 F.
- **A D3 continua SUSPENSA**, pela mesma regra fixada antes dos dados. Os itens corrompidos não mudavam a conclusão.

### Prévia da montagem (`/tmp/claude-1000/base3/previa3/`)

| | Seção 11 | Agora |
|---|---|---|
| Removidos | 11 humanos | **30**: 11 humanos, **17 pela D6** (datas), **2 pela D2 com árbitro** (MT5023MH05MT, MT9036DH12TD) |
| Revisão humana | 1080, 1368 | 139 e 401 (reais removidos, para conferência), 254 (MT5060, incompleto), 1080 (árbitro vê item incompleto), 1368 |
| Base na v3 | 1.424 | **1.405** (+17 injetadas depois do rejulgamento) |
| Déficit (relatório) | 1.393 (F 445 · M 485 · D 452) | **1.411 (F 448 · M 494 · D 459)** |

**Projeção da escala:** a injeção passa a cerca de 1.394 questões (+1,3%) e a auditoria a 1.405 itens. Central: cerca de **13,7 mil chamadas**, 34 M de tokens e **≈ 29 h de parede** (máximo dos 2 processos). Otimista 10,7 mil (≈ 21 h); pessimista 19,3 mil (≈ 45 h).

### Avaliação contaminada (val não pode ser alterada)

`val.jsonl` e `val_frozen_v1.jsonl` (30 itens cada) têm **4 itens quebrados em cada**:
- 3 com datas da planilha: MT9074MH25TD, MT9032MH07MT e MT9044DH09MT;
- MT9009DH03TD, com gabarito errado.

São 13% da validação. Uma avaliação que pontua o modelo nesses itens mede ruído; eles devem ser descontados na leitura das métricas.

### Pendências (não corrigidas aqui)

- **Restaurar ou remover os 17 itens com datas:** a v3 remove (padrão). Restaurar por dia/mês recupera 17 itens reais, mas o **MT9020MfH07TD continua defeituoso mesmo restaurado**: 5/16 e 10/32 são iguais, e o gabarito 10/8 depende de leitura.
- **Regressão piorou no nível dos juízes (r7, `metas_ok` falso):**
  - o par aceita o DEF-1052, e a D5-G2 virou falsa rejeição;
  - o "0 ruins aceitas" do pipeline depende do verificador de número por extenso, escrito depois de ver o 1052: é uma correção dentro da amostra;
  - não há conjunto separado para validar os prompts;
  - a causa da regressão do validador no 1052 não foi diagnosticada.
- **Subtema na mensagem do revisor:** a calibração ainda manda `subtema=None`, então a omissão do subtema que entrega a resposta (P3) nunca foi medida com chamadas reais.
- **1348:** fica, pela H2, mas a resolução escreve 5/11 "escrita como 0,45", sem "aproximadamente". Uma decisão `manter_editar` poderia ajustar o texto à regra 12 do gerador. Não editei, porque você aceitou o gabarito, não a redação.
<!-- secao-preservada:fim -->
