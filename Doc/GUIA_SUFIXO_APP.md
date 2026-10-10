# Guia de integração: prompt de 1 questão com sufixo de diversidade (app mobile)

Público: equipe do app (TypeScript/React Native + llama.cpp). Fonte da verdade em Python:
`src/gerar_lote.py` (`montar_prompt`, `prompt_com_sufixo`), `src/diversidade.py` (`planejar_lote`,
`sufixo_prompt`) e `data/taxonomia_subtemas.json`. Os 6 vetores de teste abaixo (e em
`Doc/vetores_sufixo_app.json`) foram gerados por esse código Python real.

## Resumo do que foi medido (90 gerações com sufixo x 90 sem, mesmo GGUF, mesmas seeds, 3 rodadas)

O sufixo **tira os defeitos de estilo** do prompt puro e **faz o modelo seguir o subtema pedido**, mas
**não melhorou a correção das questões** (leitura manual de 30 itens: 9/30 limpas com sufixo, 11/30 sem; diferença
dentro do ruído). Por isso: use o sufixo para variar o que é perguntado, e mantenha a verificação de gabarito do app.

| Métrica (n=90 de cada lado) | sem sufixo | com sufixo |
|---|---|---|
| alternativa E "Nenhuma das alternativas anteriores" | 15 | 2 |
| enunciado em CAIXA ALTA | 8 | 0 |
| resolução com menos de 40 caracteres | 8 | 0 |
| subtema gerado = subtema pedido (habilidades com 2+ subtemas, n=66) | 15 | 36 |
| contexto gerado = contexto pedido | 1 | 68 |
| custo | 211 tokens gerados/chamada | 243 tokens gerados/chamada (+15%) e +30 tokens de prompt |

Relatórios: `outputs/relatorios/sufixo_cand_server_s{0,1,2}.json` e `outputs/relatorios/VEREDITO_sufixo_vs_semsufixo.json`.
Limitação: a leitura manual foi feita por um LLM (não por professor) em 30 itens; trate como indício, não como prova.


## 1. Quando e como montar o prompt

Regra: **toda chamada de geração pede 1 questão e leva o sufixo**. Para um lote de N questões,
faça N chamadas em sequência (cada uma com seu slot), nunca uma chamada pedindo N.

```
prompt = USER_TEMPLATE(ano, habilidade, descricao, dificuldade) + SUFIXO(slot)
```

`USER_TEMPLATE` (texto literal, `quantidade` sempre `1`):

```
Gere 1 questão(ões) de matemática. Ano: {ano} ano. Habilidade: {habilidade} — {descricao}. Dificuldade: {dificuldade}.
```

- `{ano}` já vem com o ordinal: `"5º"` -> `Ano: 5º ano.`
- `{descricao}` é o texto do banco SAEB **sem nenhuma alteração**. Muitas terminam em `.`, então o
  prompt tem `..` (ponto duplo) de propósito: o modelo foi treinado assim. Não faça `trim` de pontuação.
- `{dificuldade}` é um de `Fácil`, `Moderado`, `Difícil` (com acento).
- O system prompt é fixo (constante `SYSTEM_PROMPT` em `src/extract_data.py`) e deve ser idêntico em
  toda chamada, senão o cache de prefixo do llama-server não funciona.

`SUFIXO(slot)` (cada trecho começa com um espaço; use os **rótulos**, não os ids):

```
" Subtema: {subtema_rotulo}. Tipo de raciocínio: {tipo_rotulo}. Contexto: {contexto_rotulo}."
+ instrucao_do_tipo            // só se o tipo tem "instrucao" E o subtema casa "aplica_a" (ver 2)
+ INSTRUCAO_DADOS             // só se o id do subtema começa com tabela_ | grafico_ | histograma | listas
```

`INSTRUCAO_DADOS` **não está no JSON**; copie literalmente (começa com espaço):

```
" Dados: escreva no enunciado todos os dados da tabela ou do gráfico em texto, um item por linha no formato 'rótulo: valor'."
```

Caminho de fallback (habilidade ou ano que **não** existe em `habilidades`, por exemplo `3º|H99`):
sem subtema e sem tipo; o prompt termina só com `" Contexto: {contexto_rotulo}."`. Os contextos giram
(ver 3). Isso é o `_slots_fallback` do Python.

## 2. Campos de `data/taxonomia_subtemas.json` (versão 1)

Carregue o JSON uma vez (asset do app). Chave de habilidade: `"<ano>|<habilidade>"`, p. ex. `"9º|H17"`.

| Campo | O que é | Usa no prompt de 1 questão? |
|---|---|---|
| `versao`, `fonte`, `nota` | metadados | não |
| `contextos[]` | lista **global** de 16 cenários: `{id, rotulo, palavras_chave}`. `rotulo` vai no sufixo ("feira livre", "festa junina"...) | `id` e `rotulo` sim; `palavras_chave` não (só classificação) |
| `sem_contexto` | `{id:"sem_contexto", rotulo:"sem contexto narrativo"}`. O rótulo é um texto que o treino já viu 299 vezes. Só entra se a habilidade o listar em `contextos` (hoje nenhuma lista) | sim, se aparecer |
| `estruturas[]` | 6 formatos de enunciado (`pergunta_direta`, `calculo`...). Servem ao planejador de lotes e ao verificador de diversidade | **não** entra no prompt |
| `tipos_raciocinio[]` | 16 tipos: `{id, rotulo, palavras_chave}`; 3 deles (`classificacao_lados`, `classificacao_angulos`, `classificacao_propriedades`) têm ainda `aplica_a` (regex sobre o **id do subtema**), `instrucao` (texto do "Eixo: ...", começa com espaço) e `exige_dados` | `rotulo` sempre; `instrucao` só quando `aplica_a` casa o id do subtema |
| `habilidades["ano|H"]` | `{ano, habilidade, descricoes[], graus_resolucao[], tipos_raciocinio[] (ids), subtemas[], n_itens_db, ...}` (79 entradas: 1º a 5º e 9º) | é o ponto de partida do sorteio |
| `habilidades[..].subtemas[]` | `{id, rotulo, tipos_raciocinio[] (ids permitidos para ESTE subtema), palavras_chave, origem, n_exemplos_db}`. 1 a 9 por habilidade | `rotulo` no sufixo; `id` decide `aplica_a` e "Dados:" |
| `habilidades[..].tipos_raciocinio` | tipos da habilidade; use só se o subtema não trouxer lista própria | fallback |
| `habilidades[..].contextos` | **opcional**: pool de contextos só desta habilidade. Ausente em todas hoje; se existir, substitui a lista global | sim, se existir |
| `habilidades[..].conceitual`, `.estruturas` | marcas das 7 habilidades conceituais (restringem estruturas) | não |

Detalhes que costumam pegar o dev:

- Há subtemas com `id: "geral"` (habilidades sem subtemas mapeados, p. ex. `2º|H09`): o `rotulo` é a
  própria descrição da habilidade, em minúsculas e **sem acento**. Isso é esperado; copie o rótulo como está.
- `aplica_a` é regex Python (`triangul`, `quadrilater|paralelogram|trapezi|losang|retangulo_quadrado`):
  em JS, `new RegExp(aplica_a).test(subtemaId)` funciona igual.
- A instrução do eixo vem **depois** das 3 frases e **antes** do "Dados:".
- `Eixo` nunca é inventado pelo app: só o texto que está em `tipos_raciocinio[].instrucao`.

## 3. Sorteio do slot para N = 1 (portável, sem Python)

O Python usa `random.Random` com semente em string; **não tente reproduzir a sequência**, não é
portável. O que importa é a regra, que é determinística exceto pelo desempate aleatório.

O app guarda, por `(ano, habilidade)`, um **histórico de slots já usados**: lista de
`{subtema, tipo, contexto}` (ids). Não reclassifique texto de questões antigas: guardar o slot que foi
pedido é mais simples e o `planejar_lote` aceita exatamente isso.

```ts
function sortearSlot(tax, ano, hab, hist /* slots (ano,hab) já usados */) {
  const h = tax.habilidades[`${ano}|${hab}`];
  const ctxPool = (h?.contextos ?? tax.contextos.map(c => c.id));
  // 1) contexto: o MENOS usado no histórico; empate -> aleatório
  const usoCtx = (id) => hist.filter(s => s.contexto === id).length;
  const contexto = menosUsado(ctxPool, usoCtx);               // empate: Math.random()
  if (!h) return { subtema: null, tipo: null, contexto };      // fallback (seção 1)

  // 2) subtema: o MENOS usado no histórico; empate -> aleatório
  const usoSub = (s) => hist.filter(x => x.subtema === s.id).length;
  const subtema = menosUsado(h.subtemas, usoSub, s => s.id);

  // 3) tipo: sorteio uniforme entre subtema.tipos_raciocinio (se vazio, h.tipos_raciocinio)
  const tipos = subtema.tipos_raciocinio?.length ? subtema.tipos_raciocinio : h.tipos_raciocinio;
  const tipo = tipos[Math.floor(Math.random() * tipos.length)];
  return { subtema: subtema.id, tipo, contexto };
}
// menosUsado: calcula o uso de cada item, pega o mínimo, sorteia entre os empatados.
```

Depois de **aceitar** uma questão (passou nas verificações do app), faça `hist.push(slot)`.
Esse é o mecanismo que dá a diversidade: o subtema e o contexto menos usados vão primeiro, e com K
subtemas você cobre `min(N, K)` subtemas distintos em N chamadas.

Em regeneração (a questão reprovou na verificação), **mantenha o mesmo prompt** e só troque a seed
(`seed = base_seed * 100 + tentativa`, como em `generate_validated`).

## 4. O que NÃO fazer

- Não pedir `N > 1` questões numa chamada (a diversidade vira loteria e só a 1ª recebe verificação).
- Não mandar o prompt sem sufixo: é o regime de ~8% do treino (itens reais em CAIXA ALTA, resolução de uma linha).
- Não usar `id` no lugar de `rotulo`, não traduzir, não normalizar acentos, não trocar a ordem das frases.
- Não aparar o `..` do final da descrição, nem acrescentar `\n`, espaço final ou aspas ao prompt.
- Não inventar `Eixo:`/`Dados:` próprios nem anexá-los a subtemas que não os pedem.
- Não mudar o system prompt, a gramática GBNF, `temperature`/`top_p`/`n_ctx` sem passar pelo gate de promoção.
- Não reaproveitar a mesma seed em chamadas diferentes (mesma seed + mesmo prompt = mesma questão).
- Não confiar no sufixo como garantia de correção: ele muda **o que** é perguntado, não valida a resposta.
  A verificação (`check_consistency` portada, dados ausentes, descarte de `falha`) continua obrigatória.

## 5. Chamada ao llama.cpp (llama-server)

Parâmetros de `baseline_v3/MANIFESTO.json` (seção `inferencia`): motor `server`, `n_ctx 2048`,
`threads 8`, `temperatura 0.7`, `top_p 0.8`, `max_tokens 512`, `np 1`, `cache_prompt true`.

Subir o servidor (uma vez; o modelo fica carregado e o prefixo do system prompt fica no cache KV):

```bash
llama-server -m qwen3-1.7b.Q4_K_M.gguf --jinja -rea off -c 2048 -np 1 -t 8 \
  --host 127.0.0.1 --port 8080 --log-disable
```

`-c 2048` e `-np 1` são obrigatórios (sem `-c`, o llama.cpp reserva os 40960 tokens do modelo, ~4,4 GB de KV;
com slots automáticos o contexto seria dividido).

Requisição (a gramática vai **inline**, como string, lida de `grammars/questao.gbnf`):

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d @- <<'EOF'
{
  "messages": [
    {"role": "system", "content": "<SYSTEM_PROMPT de src/extract_data.py, byte a byte>"},
    {"role": "user",   "content": "<prompt da seção 1>"}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "max_tokens": 512,
  "seed": 1234,
  "cache_prompt": true,
  "grammar": "<conteúdo inteiro de grammars/questao.gbnf>"
}
EOF
```

A resposta é `{"questoes":[{enunciado, alternativas{A..E}, resolucao_passo_a_passo, resposta_correta, difficulty}]}`
em `choices[0].message.content`. A gramática garante o JSON; **não** garante a letra certa.
Se o app usar um binding embarcado em vez do servidor HTTP, mantenha os mesmos valores
(temperature 0.7, top_p 0.8, 512 tokens, seed por chamada, gramática, cache do prefixo).

## 6. Vetores de teste (valide o port por igualdade de string)

Para cada vetor, dado `ano`, `habilidade`, `descricao`, `dificuldade` e o **slot** (subtema, tipo,
contexto), a sua função `montarPrompt` deve devolver **exatamente** o texto do bloco. Teste também
o fallback (V6). O sorteio do slot (seção 3) é aleatório e não entra nos vetores; as `seed` citadas
só servem para reproduzir o slot em Python.

**V1** — `5º H01`, dificuldade `Moderado`  
origem do slot: diversidade.planejar_lote(..., 1, dif, seed=4)  
entrada: subtema `registro_lingua_materna` (rótulo "registro por extenso"), tipo `resolucao_problema` (rótulo "resolução de problema"), contexto `biblioteca` (rótulo "biblioteca da escola")
descrição: `Escrever números racionais (naturais de até 6 ordens, representação fracionária ou decimal finita até a ordem dos milésimos) em sua representação por algarismos ou em língua materna OU Associar o registro numérico ao registro em língua materna.`

```text
Gere 1 questão(ões) de matemática. Ano: 5º ano. Habilidade: H01 — Escrever números racionais (naturais de até 6 ordens, representação fracionária ou decimal finita até a ordem dos milésimos) em sua representação por algarismos ou em língua materna OU Associar o registro numérico ao registro em língua materna.. Dificuldade: Moderado. Subtema: registro por extenso. Tipo de raciocínio: resolução de problema. Contexto: biblioteca da escola.
```

**V2** — `9º H17`, dificuldade `Moderado`  
origem do slot: diversidade.planejar_lote(..., 1, dif, seed=7)  
entrada: subtema `triangulo` (rótulo "triângulos"), tipo `classificacao_lados` (rótulo "classificação quanto aos lados"), contexto `festa` (rótulo "festa junina")
descrição: `Classificar triângulos ou quadriláteros em relação aos lados ou aos ângulos internos.`

```text
Gere 1 questão(ões) de matemática. Ano: 9º ano. Habilidade: H17 — Classificar triângulos ou quadriláteros em relação aos lados ou aos ângulos internos.. Dificuldade: Moderado. Subtema: triângulos. Tipo de raciocínio: classificação quanto aos lados. Contexto: festa junina. Eixo: pergunte quanto aos lados e dê a medida dos três lados; a resposta é a classe mais específica.
```

**V3** — `9º H17`, dificuldade `Moderado`  
origem do slot: diversidade.planejar_lote(..., 1, dif, seed=1)  
entrada: subtema `quadrilatero` (rótulo "quadriláteros"), tipo `classificacao_propriedades` (rótulo "classificação por lados, paralelismo e ângulos retos"), contexto `festa` (rótulo "festa junina")
descrição: `Classificar triângulos ou quadriláteros em relação aos lados ou aos ângulos internos.`

```text
Gere 1 questão(ões) de matemática. Ano: 9º ano. Habilidade: H17 — Classificar triângulos ou quadriláteros em relação aos lados ou aos ângulos internos.. Dificuldade: Moderado. Subtema: quadriláteros. Tipo de raciocínio: classificação por lados, paralelismo e ângulos retos. Contexto: festa junina. Eixo: informe os lados iguais, os lados paralelos e os ângulos retos; a resposta é o nome mais específico garantido por eles.
```

**V4** — `9º H24`, dificuldade `Fácil`  
origem do slot: diversidade.planejar_lote(..., 1, dif, seed=3)  
entrada: subtema `grafico_setores` (rótulo "gráfico de setores"), tipo `leitura_dados` (rótulo "leitura e interpretação de dados"), contexto `mercado` (rótulo "mercado do bairro")
descrição: `Representar dados em listas, tabelas ou gráficos.`

```text
Gere 1 questão(ões) de matemática. Ano: 9º ano. Habilidade: H24 — Representar dados em listas, tabelas ou gráficos.. Dificuldade: Fácil. Subtema: gráfico de setores. Tipo de raciocínio: leitura e interpretação de dados. Contexto: mercado do bairro. Dados: escreva no enunciado todos os dados da tabela ou do gráfico em texto, um item por linha no formato 'rótulo: valor'.
```

**V5** — `2º H09`, dificuldade `Difícil`  
origem do slot: diversidade.planejar_lote(..., 1, dif, seed=3)  
entrada: subtema `geral` (rótulo "identificar a classificacao ou classificar objetos ou representacoes por figuras, por meio de atributos, tais como cor, forma e medida"), tipo `classificacao_atributos` (rótulo "classificação por atributos"), contexto `horta` (rótulo "horta comunitária")
descrição: `Identificar a classificação OU Classificar objetos ou representações por figuras, por meio de atributos, tais como cor, forma e medida.`

```text
Gere 1 questão(ões) de matemática. Ano: 2º ano. Habilidade: H09 — Identificar a classificação OU Classificar objetos ou representações por figuras, por meio de atributos, tais como cor, forma e medida.. Dificuldade: Difícil. Subtema: identificar a classificacao ou classificar objetos ou representacoes por figuras, por meio de atributos, tais como cor, forma e medida. Tipo de raciocínio: classificação por atributos. Contexto: horta comunitária.
```

**V6** — `3º H99`, dificuldade `Fácil`  
origem do slot: fallback: contexto = CONTEXTOS[3 % 16], sem subtema  
entrada: subtema `geral` (rótulo "geral"), tipo `indefinido` (rótulo "livre"), contexto `viagem` (rótulo "viagem de ônibus")
descrição: `Habilidade fora da taxonomia (exemplo do caminho de fallback).`

```text
Gere 1 questão(ões) de matemática. Ano: 3º ano. Habilidade: H99 — Habilidade fora da taxonomia (exemplo do caminho de fallback).. Dificuldade: Fácil. Contexto: viagem de ônibus.
```


Arquivo para teste automatizado: `Doc/vetores_sufixo_app.json` (campos `ano`, `habilidade`,
`descricao`, `dificuldade`, `slot`, `prompt_esperado`).

Para checar um vetor novo em Python:

```bash
venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "src")
import gerar_lote as gl
print(gl.prompt_com_sufixo("9º", "H17", "Classificar triângulos ou quadriláteros em relação aos lados ou aos ângulos internos.", "Moderado", seed=7)[0])
EOF
```
