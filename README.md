# Treinamento do Qwen3-1.7B — Gerador Offline de Questões de Matemática

Fine-tuning do Qwen3-1.7B com as questões SAEB de `DB/questoes.db` para gerar
questões de matemática em JSON, rodando **offline em dispositivo mobile** via
llama.cpp (GGUF Q4_K_M).

## Pipeline

```
DB/questoes.db ──▶ extract_data.py ──▶ data/{train,val}.jsonl
                                            │
                    generate_synthetic.py ──┤ (aritmética, gabarito calculado em Python)
                    distill_teacher.py ─────┤ (professor maior + filtro determinístico)
                                            │
                                        train.py (QLoRA 4-bit, Unsloth)
                                            │
                                     outputs/lora/ ──▶ evaluate.py ──▶ outputs/eval_report.json
                                            │
                                     export_gguf.py ──▶ outputs/gguf/*.gguf  (deploy mobile)
                                            │
                                     test_model.py (grammar GBNF + gerar→checar→corrigir)
                                     ──▶ outputs/eval_report_gguf.json
                                     (testa o .gguf real via llama.cpp, sem torch/unsloth)
```

```bash
source venv/bin/activate
pip install -r requirements.txt

python src/extract_data.py            # 1. SQLite -> JSONL (303 questões textuais, split 90/10)
python src/generate_synthetic.py      # 1b. (opcional) aumenta data/train.jsonl com aritmética sintética
python src/train.py                   # 2. fine-tuning QLoRA na RTX 3060 6GB
python src/evaluate.py                # 3. métricas de qualidade + tempo de resposta (GPU, dev)
python src/evaluate.py --baseline     # (opcional) comparação A/B com o modelo base
python src/export_gguf.py             # 4. GGUF Q4_K_M (~1.1GB) para o app
python src/test_model.py              # 5. teste real: gera questões com o .gguf via llama.cpp
python src/test_model.py --batch      # ou valida em lote contra data/val.jsonl
```

### Aumentando o dataset com aritmética sintética (`generate_synthetic.py`)

As ~300 questões reais ensinam formato/estilo (ver "Evolução futura" abaixo),
mas são poucas para o modelo generalizar cálculo. `generate_synthetic.py`
gera questões de adição, subtração, multiplicação, divisão, porcentagem,
potenciação e frações (representação pictórica em texto, equivalência e
conversão fração→porcentagem — H07/H08/H09 do 9º ano) com o **gabarito
calculado em Python antes de montar a questão**
— nunca por um LLM, então nunca pode estar errado — usando as mesmas tuplas
(ano, habilidade, descrição, dificuldade) reais do banco, para o prompt de
treino continuar idêntico ao que o app manda em produção.

Cada questão tem **5 alternativas** (A-E, contrato do app — ver "Formato dos
dados" abaixo). O distrator não é um número aleatório: é o resultado de um
**erro pedagógico específico** (operação trocada, erro de vai-um, erro de
casa decimal, confundir "valor do desconto" com "valor final"...); só a
justificativa da alternativa CORRETA vira `resolucao_passo_a_passo` — o
contrato não tem mais um campo de justificativa por alternativa:

```bash
python src/generate_synthetic.py --dry-run             # só mostra estatísticas, não escreve nada
python src/generate_synthetic.py                       # mescla em data/train.jsonl (12 por habilidade/dificuldade)
python src/generate_synthetic.py --num-per-skill 20     # lote maior
```

Só mexe em `data/train.jsonl` (é seguro rodar de novo — remove o lote
sintético anterior antes de gerar um novo). **`data/val.jsonl` nunca é
tocado**: a validação continua só com questões reais do SAEB, para medir o
modelo no que ele de fato vai enfrentar em produção. Isso não muda nada no
deploy — o `.gguf` exportado continua o mesmo formato/tamanho, offline e
mobile como antes; só a qualidade do fine-tuning tende a melhorar.

### Destilação de um professor maior (`distill_teacher.py`)

`generate_synthetic.py` só cobre cálculo puro (as habilidades onde dá para
computar o gabarito em Python). Para habilidades mais interpretativas
(leitura de tabela, ordem/comparação, sequências...) e para variar o
contexto das questões de cálculo, `distill_teacher.py` usa um **modelo
professor maior** via Hugging Face Inference Providers para gerar questões
novas — mas **nada entra no treino sem passar pelo filtro determinístico**:
schema completo, 5 alternativas distintas, sem menção a figura, tamanho
dentro do `max_seq_length`, `resposta_correta` não reprovada por
`check_consistency()` e sem duplicata de enunciado. O professor erra às
vezes; o filtro é o que garante que só entra dado limpo.

```bash
python src/distill_teacher.py --dry-run --limit-tuples 5   # mostra o plano, sem chamar a API (grátis)
python src/distill_teacher.py --limit-tuples 5              # teste barato: 5 tuplas x 3 questões
python src/distill_teacher.py                                # roda para todas as tuplas do banco
python src/distill_teacher.py --merge                        # mescla data/distill.jsonl em data/train.jsonl
```

Requer um `HF_TOKEN` com acesso a Inference Providers (billing do HF — não é
o mesmo custo zero de `generate_synthetic.py`). `data/distill.jsonl` é
append-only (seguro interromper e retomar: duplicatas são descartadas tanto
na geração quanto na mesclagem). `--merge` é uma etapa separada de propósito
— dá para revisar amostras de `data/distill.jsonl` manualmente antes de
decidir incorporar ao treino.

### Verificação em produção: grammar GBNF + gerar→checar→corrigir (`test_model.py`)

Mesmo com o dataset melhor, um modelo de 1.7B ainda erra às vezes. Para não
depender só de "o modelo aprendeu direito", `test_model.py` roda um pipeline
de verificação sobre o artefato real (`generate_validated()`):

1. **Grammar GBNF** (`grammars/questao.gbnf`) restringe a decodificação do
   `llama-cli` ao contrato exato exigido pelo app — garante por construção
   JSON válido, wrapper `{"questoes": [...]}`, todas as chaves, 5
   alternativas (A-E), `resposta_correta` em `{A,B,C,D,E}`, `difficulty` em
   `{EASY,MEDIUM,HARD}`. Não garante consistência lógica (passo 2).
2. **`check_consistency()`** extrai uma conta "a op b = r" de
   `resolucao_passo_a_passo` e compara com `alternativas[resposta_correta]`.
   Cobre bem contas simples; não cobre raciocínio verbal sem equação nem
   frações/porcentagens textuais (ver "Limitações conhecidas" abaixo — esse é
   o efeito colateral aceito de remover o campo `resposta`, que existia só
   para dar cobertura exata e foi descartado para seguir o contrato do app).
3. Se reprovar, **best-of-N**: amostra até `--retries N`+1 candidatos,
   parando no primeiro que passa e guardando o **melhor** entre os que
   reprovam (não o último).
4. Esgotadas as tentativas, **`fix_gabarito()`** troca `resposta_correta`
   para a alternativa que bate com a conta de `resolucao_passo_a_passo` — só
   descarta a questão (`status="falha"`) se nem isso resolver.

```bash
python src/test_model.py --ano "5º" --habilidade H08 --descricao "..." --dificuldade Fácil
python src/test_model.py --quantidade 5             # pede um lote de 5 questões numa chamada
python src/test_model.py --batch                    # pipeline completo (grammar + verificação)
python src/test_model.py --batch --raw              # mede o modelo cru, sem grammar/verificação (comparação)
python src/test_model.py --no-grammar               # só o verificador, sem a grammar
```

`evaluate.py` mede o modelo em 4-bit via `bitsandbytes`/HF na GPU — um caminho
só de desenvolvimento, que não reflete o app. `test_model.py` (seção acima)
chama o mesmo binário e o mesmo `.gguf` que rodam no celular (`llama-cli`,
CPU, Q4_K_M), sem dependência de `torch`/`unsloth` — só precisa do
`llama-cli` (gerado por `export_gguf.py` em `~/.unsloth/llama.cpp/llama-cli`)
e do `.gguf` exportado.

### Gabarito inconsistente com a resolução

Em alguns testes o cálculo na resolução está certo, mas a letra de
`resposta_correta` não bate com ele. Causas identificadas, e o que ataca cada
uma:

1. **Ordem das chaves no JSON treinado**: a geração é autorregressiva —
   `resolucao_passo_a_passo` é emitido ANTES de `resposta_correta` na
   sequência de treino (mesmo conjunto de chaves do contrato do app, que não
   garante nem depende de ordem posicional — ver "Formato dos dados" abaixo),
   para o modelo "mostrar o trabalho" antes de se comprometer com a letra (o
   modo non-thinking não tem `<think>` para isso).
2. **Distratores sintéticos sem valor plausível**: cada gerador de
   `generate_synthetic.py` produz o distrator a partir de um **erro
   pedagógico específico** (operação trocada, erro de vai-um, erro de casa
   decimal...), não um número aleatório — evita que o modelo aprenda padrões
   de distrator "óbvios" que não ensinam nada sobre o erro real do aluno.
3. **Verificador cego em boa parte dos casos**: o contrato do app (schema
   fixo, sem exceção) não tem um campo dedicado ao VALOR da resposta — só
   `resposta_correta` (a LETRA). `check_consistency()` depende então de
   extrair uma equação "a op b = r" de `resolucao_passo_a_passo` via regex,
   o que cobre contas simples mas não raciocínio verbal sem equação nem
   frações/porcentagens textuais (medido: ~75% de cobertura no conjunto
   sintético, que é só aritmética — menor em habilidades mais interpretativas
   do banco real). Normalização Unicode (`normalize_math`) evita que
   operadores tipográficos (`−` U+2212, `×`, `–`) quebrem o regex.
4. **Qwen3-1.7B é pequeno**: aritmética multi-dígito e coerência semântica em
   contextos incomuns são pontos fracos conhecidos de LLMs nessa faixa sem
   chain-of-thought — nenhuma correção de schema/dataset elimina isso
   totalmente. Atacado por dataset maior/mais limpo (1 e 2 acima, mais
   `distill_teacher.py`) e mitigado em produção pelo verificador acima.

Para produção, `test_model.py` já roda o verificador gerar→checar→corrigir
(seção "Verificação em produção" acima) — é a garantia que não depende do
modelo ter aprendido perfeitamente.

## Por que essa abordagem (literatura atual)

O problema relatado — modelo lento no mobile e com falhas de formato — tem duas
causas independentes, e o pipeline ataca as duas:

### 1. Qualidade/raciocínio → SFT com QLoRA

- **QLoRA** (Dettmers et al., 2023, *QLoRA: Efficient Finetuning of Quantized
  LLMs*): congela o modelo base quantizado em NF4 4-bit e treina só adaptadores
  LoRA (~1–2% dos parâmetros). Recupera a qualidade do fine-tuning completo em
  16-bit com fração da VRAM — é o que torna o treino viável numa RTX 3060 6GB.
- **Unsloth**: implementação otimizada de LoRA/QLoRA (~2x mais rápida, ~40%
  menos VRAM que PEFT puro), com suporte oficial ao Qwen3 e exportação GGUF
  integrada.
- **Loss só na resposta** (*train on completions only*): mascara os tokens do
  prompt no cálculo da loss. Prática padrão do TRL que melhora o resultado em
  datasets pequenos.
- **Dataset pequeno (~300 exemplos)**: suficiente para ensinar **formato,
  estilo e estrutura pedagógica** (o objetivo aqui), que é onde SFT com poucas
  centenas de exemplos comprovadamente funciona (cf. *LIMA: Less Is More for
  Alignment*, Zhou et al., 2023). Hiperparâmetros calibrados para isso: 3
  épocas com early stopping, LoRA r=16, lr 2e-4 cosine.

### 2. Velocidade no mobile → menos tokens + quantização

- **Modo non-thinking do Qwen3**: o Qwen3 por padrão gera blocos `<think>`
  longos antes da resposta — a principal causa de lentidão percebida. Treinamos
  e inferimos com `enable_thinking=False`; o raciocínio pedagógico fica nas
  justificativas do JSON, não em cadeia de pensamento solta.
- **Saída JSON compacta**: geração é autorregressiva — o custo é proporcional
  aos tokens gerados. A saída treinada tem ~150–300 tokens (vs. milhares com
  thinking).
- **GGUF Q4_K_M** (k-quants do llama.cpp): ~1.1GB, perda de qualidade mínima
  para 4 bits, formato de fato para LLM offline em Android/iOS.
- **Grammar GBNF no app**: o llama.cpp permite restringir a decodificação a uma
  gramática JSON, garantindo saída parseável mesmo nos casos raros de erro.

### Evolução futura

Com ~300 exemplos reais o modelo aprende o formato, mas não amplia
conhecimento matemático sozinho — por isso os dois caminhos de aumento de
dados já implementados (`generate_synthetic.py` para cálculo puro, incluindo
frações, H07–H09 do 9º ano; `distill_teacher.py` para destilação de um
professor maior com filtro determinístico). Próximos passos considerados:
**GRPO/RLVR** (reinforcement learning com recompensa verificável) usando
`schema_utils.check_consistency()` diretamente como função de recompensa —
Unsloth já roda GRPO em Qwen3-1.7B com ~5GB de VRAM, dentro da RTX 3060 6GB
disponível.

## Formato dos dados

Contrato fixo exigido pelo app mobile (definido pelos envolvidos, sem
exceção — qualquer desvio de chave/tipo/enum quebra o parsing no app).

Entrada (user): `Gere 1 questão(ões) de matemática. Ano: 5º ano. Habilidade:
H08 — <descritor>. Dificuldade: Fácil.` (o "1" muda se o usuário pedir um
lote — ex.: `Gere 10 questão(ões)...` — e o modelo deve devolver 10 itens em
`questoes`.)

Saída (assistant):

```json
{
  "questoes": [
    {
      "enunciado": "...",
      "alternativas": {"A": "...", "B": "12", "C": "...", "D": "...", "E": "..."},
      "resolucao_passo_a_passo": "...",
      "resposta_correta": "B",
      "difficulty": "EASY"
    }
  ]
}
```

`difficulty` é `EASY`/`MEDIUM`/`HARD` (mapeado de Fácil/Moderado/Difícil).
Internamente o modelo é treinado a emitir `resolucao_passo_a_passo` ANTES de
`resposta_correta` (mostra o trabalho antes de se comprometer com a letra —
ver "Gabarito inconsistente com a resolução" abaixo); como JSON não garante
ordem para quem consome por nome, isso não afeta o contrato, só a sequência
de geração token a token.

Esse schema **não tem mais** um campo dedicado ao VALOR da resposta nem
justificativa por alternativa (ambos existiam numa versão anterior só para
uso interno de verificação) — foram removidos para seguir exatamente o
contrato do app. Efeito colateral aceito: `check_consistency()` volta a
depender de regex sobre `resolucao_passo_a_passo` (ver seção acima), com
cobertura menor do que a comparação exata que o campo `resposta` permitia.

Questões com imagem (230 de 553) são excluídas — o modelo em produção é só
texto e não deve aprender a citar figuras inexistentes. Alternativa **E** não
existe no banco real (que só tem A-D); para esses casos é preenchida com um
distrator fixo ("Nenhuma das alternativas anteriores"), nunca a correta.

## Hardware

Calibrado para **RTX 3060 Laptop 6GB** (bf16/Ampere), CUDA 12.x, Python 3.11.
Se ocorrer OOM no treino: em `src/train.py`, reduza `MAX_SEQ_LENGTH` para 768
ou `BATCH_SIZE` para 1 (dobrando `GRAD_ACCUMULATION`).

## Métricas de validação (`outputs/eval_report.json`)

| Grupo | Métrica |
|---|---|
| Estrutura | % JSON válido, % schema completo, % gabarito válido, % alternativas distintas, % justificativas distintas, distribuição de gabaritos, % menções a figura, % consistência gabarito↔justificativa (ver seção acima) |
| Linguagem | perplexity da resposta de referência |
| Velocidade | latência média/p95 por questão, tokens/s, tokens de saída |

A velocidade medida por `evaluate.py` é na GPU local; o número real do mobile
deve ser medido com o GGUF: `llama-bench -m outputs/gguf/<modelo>.gguf -p 256 -n 256`.
