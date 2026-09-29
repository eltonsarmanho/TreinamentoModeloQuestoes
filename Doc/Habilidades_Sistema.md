# Habilidades (Descritores) do Sistema

Levantamento gerado a partir da base de dados `DB/questoes.db` (tabela `itens`), refletindo as habilidades/descritores efetivamente associados às questões cadastradas.

## Resumo

- Total de itens na base: **1039**
- Itens com habilidade classificada: **1020** (19 sem classificação, valor `nan`)
- Habilidades BNCC (códigos `EFxxMAxx`): **10**, todas de Matemática
- Descritores tipo SAEB/Prova Brasil (códigos `H01`–`H26`): **69** combinações (código × ano escolar)

Existem dois sistemas de classificação de habilidade coexistindo na base:

1. **Códigos BNCC** (`codigo_bncc` preenchido, ex.: `EF01MA02`) — usados para os anos 1º ao 4º do Ensino Fundamental, cada código com descrição fixa e única.
2. **Descritores `H01`–`H26`** (sem `codigo_bncc`) — reaproveitados por ano escolar (2º, 5º e 9º), ou seja, **o mesmo código `H01` representa uma habilidade diferente dependendo do ano**. É o modelo de matriz de referência tipo SAEB/Prova Brasil.

## 1. Habilidades BNCC (`EFxxMAxx`)

| Código | Ano | Descrição | Nº de itens |
|---|---|---|---|
| EF01MA02 | 1º | Contar de maneira exata ou aproximada, utilizando diferentes estratégias como o pareamento e outros agrupamentos. | 50 |
| EF01MA06 | 1º | Construir fatos fundamentais da adição e utilizá-los em procedimentos de cálculo para resolver problemas. | 50 |
| EF01MA08 | 1º | Resolver e elaborar problemas de adição e de subtração, envolvendo números de até dois algarismos, com os significados de juntar, acrescentar, separar e retirar, com o suporte de imagens e/ou material manipulável, utilizando estratégias e formas de registro pessoais. | 50 |
| EF02MA03 | 2º | Comparar quantidades de objetos de dois conjuntos, por estimativa e/ou por correspondência (um a um, dois a dois, entre outros), para indicar “tem mais”, “tem menos” ou “tem a mesma quantidade”, indicando, quando for o caso, quantos a mais e quantos a menos. | 50 |
| EF02MA05 | 2º | Construir fatos básicos da adição e subtração e utilizá-los no cálculo mental ou escrito. | 50 |
| EF02MA06 | 2º | Resolver e elaborar problemas de adição e de subtração, envolvendo números de até três ordens, com os significados de juntar, acrescentar, separar, retirar, utilizando estratégias pessoais ou convencionais. | 49 |
| EF03MA01 | 3º | Ler, escrever e comparar números naturais de até a ordem de unidade de milhar, estabelecendo relações entre os registros numéricos e em língua materna. | 37 |
| EF03MA03 | 3º | Construir e utilizar fatos básicos da adição e da multiplicação para o cálculo mental ou escrito. | 50 |
| EF03MA07 | 3º | Resolver e elaborar problemas de multiplicação (por 2, 3, 4, 5 e 10) com os significados de adição de parcelas iguais e elementos apresentados em disposição retangular, utilizando diferentes estratégias de cálculo e registros. | 50 |
| EF04MA04 | 4º | Utilizar as relações entre adição e subtração, bem como entre multiplicação e divisão, para ampliar as estratégias de cálculo. | 50 |

## 2. Descritores `H01`–`H26` (por ano escolar)

Cada descritor é reutilizado nos três anos avaliados (2º, 5º e 9º), com significado próprio em cada um. Quando a descrição variou ligeiramente entre itens (fraseados diferentes para o mesmo descritor), a mais frequente é mostrada; a coluna final indica quantas variações de texto existem para esse código.

### Ano: 2º

| Código | Descrição | Nº de itens | Variações de texto |
|---|---|---|---|
| H01 | Reconhecer o que os números naturais indicam em diferentes situações: quantidade, ordem, medida ou código de identificação. | 8 | 1 |
| H02 | Identificar a posição ordinal de um objeto ou termo em uma sequência (1º, 2º etc.). | 8 | 1 |
| H03 | Escrever números naturais de até 3 ordens em sua representação por algarismos ou em língua materna. OU Associar o registro numérico de números naturais de até 3 ordens ao registro em língua materna. | 8 | 2 |
| H04 | Comparar OU Ordenar quantidades de objetos (até 2 ordens). | 8 | 1 |
| H05 | Comparar OU Ordenar números naturais, de até 3 ordens, com ou sem suporte da reta numérica. | 8 | 1 |
| H06 | Calcular o resultado de adições ou subtrações, envolvendo números naturais de até 3 ordens. | 8 | 1 |
| H07 | Compor OU Decompor números naturais de até 3 ordens por meio de diferentes adições. | 8 | 1 |
| H08 | Resolver problemas de adição ou de subtração, envolvendo números naturais de até 3 ordens, com os significados de juntar, acrescentar, separar ou retirar. | 8 | 1 |
| H09 | Identificar a classificação OU Classificar objetos ou representações por figuras, por meio de atributos, tais como cor, forma e medida. | 8 | 1 |
| H10 | Inferir os elementos ausentes em uma sequência de números naturais ordenados, de objetos ou de figuras. | 8 | 1 |
| H11 | Reconhecer/nomear figuras geométricas planas (círculo, quadrado, retângulo e triângulo). | 8 | 1 |
| H12 | Reconhecer/nomear figuras geométricas espaciais (cubo, bloco retangular, pirâmide, cone, cilindro e esfera), relacionando-as com objetos do mundo físico. | 8 | 1 |
| H13 | Identificar a localização OU a descrição/esboço do deslocamento de pessoas e/ou de objetos em representações bidimensionais (mapas, croquis etc.). | 8 | 2 |
| H14 | Comparar comprimentos, capacidades ou massas OU Ordenar imagens de objetos com base na comparação visual de seus comprimentos, capacidades ou massas. | 8 | 1 |
| H15 | Relacionar valores de moedas e/ou cédulas do sistema monetário brasileiro, com base nas imagens desses objetos. | 8 | 1 |
| H16 | Estimar/Inferir medida de comprimento, capacidade ou massa de objetos, utilizando unidades de medida convencionais ou não OU Medir comprimento, capacidade ou massa de objetos. | 8 | 1 |
| H17 | Identificar sequência de acontecimentos relativos a um dia. | 8 | 1 |
| H18 | Identificar datas, dias da semana, ou meses do ano em calendário OU Escrever uma data, apresentando o dia, o mês e o ano. | 8 | 1 |
| H19 | Ler/Identificar OU Comparar dados estatísticos ou informações expressos em tabelas (simples ou de dupla entrada). | 8 | 1 |
| H20 | Ler/Identificar OU Comparar dados estatísticos expressos em gráficos (barras simples, colunas simples ou pictóricos). | 7 | 1 |
| H21 | Ler/Identificar OU Comparar dados estatísticos expressos em gráficos (barras simples, colunas simples ou pictóricos). | 1 | 1 |

### Ano: 5º

| Código | Descrição | Nº de itens | Variações de texto |
|---|---|---|---|
| H01 | Escrever números racionais (naturais de até 6 ordens, representação fracionária ou decimal finita até a ordem dos milésimos) em sua representação por algarismos ou em língua materna OU Associar o registro numérico ao registro em língua materna. | 8 | 2 |
| H02 | Identificar a ordem ocupada por um algarismo OU seu valor posicional (ou valor relativo) em um número natural de até 6 ordens. | 8 | 2 |
| H03 | Calcular o resultado de adições ou subtrações envolvendo números naturais de até 6 ordens. | 8 | 2 |
| H04 | Calcular o resultado de multiplicações ou divisões envolvendo números naturais de até 6 ordens. | 8 | 2 |
| H05 | Resolver problemas de adição ou de subtração, envolvendo números naturais de até 6 ordens, com os significados de juntar, acrescentar, separar, retirar, comparar ou completar. | 8 | 1 |
| H06 | Resolver problemas de multiplicação ou de divisão, envolvendo números naturais de até 6 ordens, com os significados de formação de grupos iguais (incluindo repartição equitativa e medida), proporcionalidade ou disposição retangular. | 8 | 2 |
| H07 | Inferir OU Descrever atributos ou propriedades comuns que os elementos que constituem uma sequência recursiva de números naturais apresentam. | 8 | 2 |
| H08 | Inferir o padrão ou a regularidade de uma sequência de números naturais ordenados, objetos ou figuras. | 8 | 1 |
| H09 | Inferir os elementos ausentes em uma sequência de números naturais ordenados, objetos ou figuras. | 8 | 2 |
| H10 | Identificar a localização OU a descrição/esboço do deslocamento de pessoas e/ou de objetos em representações bidimensionais (mapas, croquis etc.). | 8 | 1 |
| H11 | Descrever OU Esboçar o deslocamento de pessoas e/ou objetos em representações bidimensionais (mapas, croquis etc.) ou plantas de ambientes, de acordo com condições dadas. | 8 | 1 |
| H12 | Reconhecer/nomear figuras geométricas espaciais (prismas, pirâmides, cilindros, cones ou esferas). | 8 | 1 |
| H13 | Relacionar figuras geométricas espaciais (prismas retos, pirâmides retas, cilindros retos ou cones retos) a suas planificações. | 8 | 1 |
| H14 | Reconhecer/nomear figuras geométricas planas (polígonos, circunferência ou círculo). | 8 | 1 |
| H15 | Medir OU Comparar perímetro de figuras planas desenhadas em malha quadriculada. | 8 | 1 |
| H16 | Medir OU Comparar área de figuras planas desenhadas em malha quadriculada. | 8 | 1 |
| H17 | Identificar horas em relógios analógicos OU Associar horas em relógios analógicos e digitais. | 8 | 1 |
| H18 | Determinar o horário de início, o horário de término ou a duração de um acontecimento. | 8 | 1 |
| H19 | Relacionar valores de moedas e/ou cédulas do sistema monetário brasileiro, com base nas imagens desses objetos. | 8 | 1 |
| H20 | Identificar, entre eventos aleatórios, aqueles que têm menor, maior ou iguais chances de ocorrência, sem utilizar frações. | 8 | 1 |
| H21 | Ler/Identificar OU Comparar dados estatísticos expressos em tabelas (simples ou de dupla entrada). | 8 | 1 |
| H22 | Ler/Identificar OU Comparar dados estatísticos expressos em gráficos (barras simples ou agrupadas, colunas simples ou agrupadas, pictóricos, ou de linhas). | 8 | 1 |

### Ano: 9º

| Código | Descrição | Nº de itens | Variações de texto |
|---|---|---|---|
| H01 | Escrever números racionais (representação fracionária ou decimal finita) em sua representação por algarismos ou em língua materna OU Associar o registro numérico ao registro em língua materna. | 8 | 1 |
| H02 | Resolver problemas de adição, subtração, multiplicação, divisão, potenciação ou radiciação envolvendo números reais, inclusive notação científica. | 8 | 1 |
| H03 | Porcentagem, acréscimos, decréscimos e taxas sucessivas. | 8 | 2 |
| H04 | Comparar, ordenar ou aproximar números reais. | 8 | 2 |
| H05 | Múltiplos, divisores, MDC e MMC. | 8 | 2 |
| H06 | Cálculo com números reais: adição, subtração, multiplicação e divisão. | 8 | 2 |
| H07 | Representar ou associar frações a representações pictóricas. | 8 | 2 |
| H08 | Identificar frações equivalentes. | 8 | 1 |
| H09 | Converter entre representações de números racionais positivos (frações, decimais, porcentagens) | 8 | 2 |
| H10 | Resolver uma equação polinomial de 1º grau. | 8 | 1 |
| H11 | Inferir equações, inequações ou sistemas de 1º grau que representam situações-problema | 8 | 2 |
| H12 | Cálculo do valor numérico de expressões algébricas | 8 | 2 |
| H13 | Resolver problemas com sistemas de equações de 1º grau (duas incógnitas) | 8 | 2 |
| H14 | Relacionar objetos tridimensionais às suas planificações ou vistas. | 5 | 1 |
| H15 | Ângulos em polígonos, retas paralelas e elementos notáveis (cevianas). | 5 | 1 |
| H16 | Propriedades dos triângulos. | 8 | 2 |
| H17 | Classificação de triângulos e quadriláteros quanto aos lados ou ângulos. | 7 | 2 |
| H18 | Elementos da circunferência e do círculo | 7 | 2 |
| H19 | Medidas e conversão de unidades | 8 | 2 |
| H20 | Perímetro de figuras planas. | 8 | 2 |
| H21 | Área de figuras planas | 8 | 2 |
| H22 | Volume de prismas retos e cilindros retos | 8 | 2 |
| H23 | Leitura e interpretação de tabelas e gráficos. | 7 | 2 |
| H24 | Representar dados em listas, tabelas ou gráficos. | 7 | 2 |
| H25 | Probabilidade de eventos aleatórios equiprováveis | 8 | 2 |
| H26 | Média, moda e mediana. | 8 | 2 |

## 3. Exemplos de Prompt para Gerar Questões — 5º Ano

O sistema gera questões usando dois prompts fixos, definidos em [`src/extract_data.py`](../src/extract_data.py) (`SYSTEM_PROMPT` e `USER_TEMPLATE`). O `SYSTEM_PROMPT` é sempre o mesmo; o `USER_TEMPLATE` é preenchido por questão com `quantidade`, `ano`, `habilidade`, `descricao` e `dificuldade`:

```
Gere {quantidade} questão(ões) de matemática. Ano: {ano} ano. Habilidade: {habilidade} — {descricao}. Dificuldade: {dificuldade}.
```

Abaixo, exemplos reais desse template já preenchidos com descritores do **5º ano** (seção 2, tabela "Ano: 5º"):

- `Gere 1 questão de matemática. Ano: 5º ano. Habilidade: H01 — Escrever números racionais (naturais de até 6 ordens, representação fracionária ou decimal finita até a ordem dos milésimos) em sua representação por algarismos ou em língua materna OU Associar o registro numérico ao registro em língua materna. Dificuldade: Fácil.`
- `Gere 1 questão de matemática. Ano: 5º ano. Habilidade: H03 — Calcular o resultado de adições ou subtrações envolvendo números naturais de até 6 ordens. Dificuldade: Moderado.`
- `Gere 1 questão de matemática. Ano: 5º ano. Habilidade: H04 — Calcular o resultado de multiplicações ou divisões envolvendo números naturais de até 6 ordens. Dificuldade: Difícil.`
- `Gere 1 questão de matemática. Ano: 5º ano. Habilidade: H05 — Resolver problemas de adição ou de subtração, envolvendo números naturais de até 6 ordens, com os significados de juntar, acrescentar, separar, retirar, comparar ou completar. Dificuldade: Moderado.`
- `Gere 1 questão de matemática. Ano: 5º ano. Habilidade: H09 — Inferir os elementos ausentes em uma sequência de números naturais ordenados, objetos ou figuras. Dificuldade: Fácil.`
- `Gere 1 questão de matemática. Ano: 5º ano. Habilidade: H16 — Medir OU Comparar área de figuras planas desenhadas em malha quadriculada. Dificuldade: Moderado.`
- `Gere 1 questão de matemática. Ano: 5º ano. Habilidade: H19 — Relacionar valores de moedas e/ou cédulas do sistema monetário brasileiro, com base nas imagens desses objetos. Dificuldade: Fácil.`
- `Gere 1 questão de matemática. Ano: 5º ano. Habilidade: H22 — Ler/Identificar OU Comparar dados estatísticos expressos em gráficos (barras simples ou agrupadas, colunas simples ou agrupadas, pictóricos, ou de linhas). Dificuldade: Difícil.`

Exemplo de lote (`quantidade` > 1, formato aceito pelo schema `{"questoes": [...]}`):

- `Gere 3 questões de matemática. Ano: 5º ano. Habilidade: H06 — Resolver problemas de multiplicação ou de divisão, envolvendo números naturais de até 6 ordens, com os significados de formação de grupos iguais (incluindo repartição equitativa e medida), proporcionalidade ou disposição retangular. Dificuldade: Moderado.`

Valores válidos: `dificuldade` ∈ {Fácil, Moderado, Difícil} (mapeados para `EASY`/`MEDIUM`/`HARD` na resposta); `habilidade` e `descricao` devem corresponder a um dos códigos `H01`–`H22` do 5º ano listados na seção 2.

## 4. Itens sem habilidade classificada

- 19 itens estão com o campo `habilidade` vazio/`nan` na base (`disciplina` também `nan` nesses casos), representando dados incompletos a serem revisados/classificados.

## Fonte

- Banco: `DB/questoes.db`, tabela `itens`
- Campos utilizados: `habilidade`, `codigo_bncc`, `ano`, `disciplina`, `descricao_item`
- Documento gerado em 2026-09-25
