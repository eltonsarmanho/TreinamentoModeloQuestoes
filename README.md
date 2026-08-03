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

Cada questão tem as **4 justificativas** (uma por alternativa), no mesmo
padrão do SAEB real: cada distrator é o resultado de um **erro pedagógico
específico** (operação trocada, erro de vai-um, erro de casa decimal,
confundir "valor do desconto" com "valor final"...), e sua justificativa
explica esse erro citando o valor — não um número aleatório. Isso reforça o
vínculo letra↔valor↔raciocínio no treino, atacando o problema de gabarito
inconsistente com a justificativa (ver "Limitações conhecidas" abaixo):

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
schema completo, 4 alternativas/justificativas distintas, sem menção a
figura, tamanho dentro do `max_seq_length`, gabarito não reprovado por
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
   `llama-cli` ao schema exato — garante por construção JSON válido, todas
   as chaves (incluindo `resposta`), 4 alternativas e 4 justificativas,
   gabarito em `{A,B,C,D}`. Não garante consistência lógica (passo 2).
2. **`check_consistency()`** compara `alternativas[gabarito]` com `resposta`
   por **igualdade exata** — sem regex, sem depender de como a justificativa
   foi escrita, funciona para fração/porcentagem/qualquer tipo. Para
   artefatos antigos (sem o campo) cai num fallback por regex sobre a conta
   da justificativa.
3. Se reprovar, **best-of-N**: amostra até `--retries N`+1 candidatos,
   parando no primeiro que passa e guardando o **melhor** entre os que
   reprovam (não o último).
4. Esgotadas as tentativas, **`fix_gabarito()`** troca a letra do gabarito
   para a alternativa que contém a `resposta` — só descarta a questão
   (`status="falha"`) se nem isso resolver.

```bash
python src/test_model.py --ano "5º" --habilidade H08 --descricao "..." --dificuldade Fácil
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

### Gabarito inconsistente com a justificativa

Em alguns testes o cálculo na justificativa está certo, mas a letra do
`gabarito` não bate com ele (ou as 4 justificativas saem repetidas/genéricas).
Causas identificadas, e o que ataca cada uma:

1. **Ordem das chaves no JSON treinado**: a geração é autorregressiva — antes
   desta mudança, `gabarito` vinha *antes* de `justificativas` no schema,
   então o modelo comprometia a letra sem ter "mostrado o trabalho" ainda (o
   modo non-thinking não tem `<think>` para isso). Corrigido: `justificativas`
   agora vem antes de `gabarito` no schema (`extract_data.py`/`SYSTEM_PROMPT`
   e `grammars/questao.gbnf`).
2. **Distratores sintéticos sem justificativa própria** (regressão
   introduzida na primeira versão de `generate_synthetic.py`, identificada em
   testes reais no app): gerar só a justificativa do gabarito e nada para as
   outras 3 alternativas ensinava o modelo a tratar "justificativa" como
   enfeite, e ele passava a repetir a mesma frase nas 4. Corrigido: cada
   gerador agora produz **4 justificativas distintas**, uma por distrator
   pedagógico (ver seção do `generate_synthetic.py` acima). Métrica nova:
   `justificativas_distintas_pct` em `evaluate.py`/`test_model.py`.
3. **Verificador cego em 70–77% dos casos** (medido em `eval_report.json`:
   23 de 30 "não verificável"). Duas causas: (a) o modelo emite operadores
   Unicode (`−` U+2212, `×`, `–`) que o regex ASCII não casava; (b) mais
   grave, justificativas em texto corrido sem equação ("Subtraindo 5 de 12,
   obtemos 7") que nenhum regex cobre de forma robusta. Corrigido em duas
   frentes: normalização Unicode (`normalize_math`) e, sobretudo, o campo
   **`resposta`** no schema — o valor da resposta num slot dedicado torna a
   verificação uma comparação exata, levando a cobertura de ~25% para 100%
   no conjunto sintético.
4. **Qwen3-1.7B é pequeno**: aritmética multi-dígito e coerência semântica em
   contextos incomuns são pontos fracos conhecidos de LLMs nessa faixa sem
   chain-of-thought — nenhuma correção de schema/dataset elimina isso
   totalmente. Atacado por dataset maior/mais limpo (1 e 2 acima, mais
   `distill_teacher.py`) e mitigado em produção pelo verificador acima.

Os itens 1–2 só têm efeito **depois de retreinar** (`python src/train.py`
sobre o `data/train.jsonl` regenerado). Para produção, independente de
quando/se o retreino aconteceu, `test_model.py` já roda o verificador
gerar→checar→corrigir (seção "Verificação em produção" acima) — é a garantia
que não depende do modelo ter aprendido perfeitamente.

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

Entrada (user): `Gere uma questão de matemática. Ano: 5º ano. Habilidade: H08 —
<descritor>. Dificuldade: Fácil.`

Saída (assistant):

```json
{
  "enunciado": "...",
  "comando": "...",
  "alternativas": {"A": "...", "B": "12", "C": "...", "D": "..."},
  "justificativas": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "resposta": "12",
  "gabarito": "B"
}
```

Ordem das chaves importa (ver "Gabarito inconsistente com a justificativa"
abaixo): o modelo primeiro **mostra o trabalho** (`justificativas`), depois
se compromete com o **valor** (`resposta`) e só então escolhe a **letra**
(`gabarito`). Isso torna a verificação uma comparação exata
`alternativas[gabarito] == resposta`, sem depender de interpretar texto.

Questões com imagem (230 de 553) são excluídas — o modelo em produção é só
texto e não deve aprender a citar figuras inexistentes.

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
