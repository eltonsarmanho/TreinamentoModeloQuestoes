# Procedimento da amostra humana (avaliação de questões)

Para: responsável pedagógico. Tempo estimado: **2 a 3 horas** para 40 questões
(3 a 4 minutos por questão), em uma ou duas sessões. Não é preciso instalar nada:
a folha abre no Excel ou no LibreOffice Calc.

## Por que isto existe

Os testes automáticos do projeto conferem o formato, as contas e o rótulo de
dificuldade. Eles não dizem se a questão **está certa e serve para a prova**.
Só uma pessoa da área responde isso. Sua avaliação é o dado que decide se uma
nova versão do modelo melhora de verdade.

A folha é **cega**: as questões vêm embaralhadas, sem indicar de onde vieram.
Avalie cada uma por si só, sem tentar adivinhar a origem.

## O que você recebe

Um arquivo **`folha_professor.csv`**, uma questão por linha:

- `id`: código da questão (não altere).
- `ano`, `habilidade`, `descricao_habilidade`, `dificuldade_pedida`: o que foi solicitado ao gerador.
- `enunciado`, `A` a `E`, `resolucao`: o texto da questão.
- `gabarito_modelo`: a letra que o modelo marcou como correta. **Ela pode estar errada**: conferi-la faz parte da avaliação.

## Como preencher

Abra o arquivo (no Excel: *Dados > De Texto/CSV*, codificação **UTF-8**). Para cada
questão, preencha as **7 colunas em branco** com **`s`** (sim) ou **`n`** (não).
Resolva a questão por conta própria antes de olhar a resolução.

| Coluna | Responda `s` se... |
|---|---|
| `correcao_matematica` | a resolução está matematicamente correta e a letra de `gabarito_modelo` é de fato a resposta certa. |
| `resposta_unica` | existe **exatamente uma** alternativa correta (nenhuma outra também serve, e as alternativas não se repetem). |
| `dados_suficientes` | o enunciado traz tudo o que é preciso para resolver e não depende de figura, gráfico ou tabela que não aparece no texto. |
| `aderencia_ano_habilidade` | o conteúdo corresponde ao **ano** e à **habilidade** indicados (use `descricao_habilidade`). |
| `contexto_plausivel` | a situação é realista e a linguagem é adequada à idade (sem valores, objetos ou cenas absurdos). |
| `dificuldade_adequada` | o nível condiz com `dificuldade_pedida` (Fácil, Moderado, Difícil) para aquele ano. |
| `valida_geral` | você **usaria a questão numa avaliação como ela está**. Em geral exige as seis acima em `s`; se marcar `s` com alguma em `n`, o sistema apenas avisa. |

`comentario` é opcional: use para anotar o motivo de um `n` (ex.: "duas alternativas
corretas: B e D"). Comentários curtos ajudam a melhorar o modelo.

Regras do preenchimento:

1. Use somente `s` ou `n` (minúsculas; "sim" e "não" também são aceitos). Qualquer outro valor invalida a linha.
2. **Não deixe células em branco** nas 7 colunas. Se a questão for incompreensível, marque `n` e explique no comentário.
3. Não apague, reordene colunas nem altere o `id`. Pode reordenar ou filtrar as linhas se isso ajudar.
4. Não é preciso (nem permitido, para manter a folha cega) perguntar de onde vem cada questão.

## Como devolver

1. Salve como **CSV UTF-8** com o nome **`folha_preenchida.csv`** (Excel: *Salvar como > CSV UTF-8*; Calc: *Salvar como > Texto CSV*, conjunto de caracteres Unicode UTF-8).
2. Envie o arquivo ao responsável técnico do projeto (por e-mail ou pasta compartilhada).
3. Se algo não puder ser avaliado, devolva mesmo assim e liste os `id` no corpo da mensagem.

## O que acontece depois (responsável técnico)

```bash
python src/amostra_humana.py apurar --folha folha_preenchida.csv \
    --chave outputs/amostra_humana/chave_oculta.json --saida outputs/amostra_humana/apuracao.json
```

A apuração informa a taxa de questões válidas por unidade temática (Números,
Álgebra, Geometria, Grandezas e medidas, Probabilidade e estatística) e por
versão do modelo, com intervalo de confiança de 95%, e a taxa de `s` em cada
dimensão. A **chave de origem** (`chave_oculta.json`) fica com o responsável
técnico e nunca é enviada junto com a folha.
