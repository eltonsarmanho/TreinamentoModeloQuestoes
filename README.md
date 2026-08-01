# Treinamento do Qwen3-1.7B — Gerador Offline de Questões de Matemática

Fine-tuning do Qwen3-1.7B com as questões SAEB de `DB/questoes.db` para gerar
questões de matemática em JSON, rodando **offline em dispositivo mobile** via
llama.cpp (GGUF Q4_K_M).

## Pipeline

```
DB/questoes.db ──▶ extract_data.py ──▶ data/{train,val}.jsonl
                                            │
                                        train.py (QLoRA 4-bit, Unsloth)
                                            │
                                     outputs/lora/ ──▶ evaluate.py ──▶ outputs/eval_report.json
                                            │
                                     export_gguf.py ──▶ outputs/gguf/*.gguf  (deploy mobile)
                                            │
                                     test_model.py ──▶ outputs/eval_report_gguf.json
                                     (testa o .gguf real via llama.cpp, sem torch/unsloth)
```

```bash
source venv/bin/activate
pip install -r requirements.txt

python src/extract_data.py            # 1. SQLite -> JSONL (304 questões textuais, split 90/10)
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
gera questões de adição, subtração, multiplicação, divisão, porcentagem e
potenciação com o **gabarito calculado em Python antes de montar a questão**
— nunca por um LLM, então nunca pode estar errado — usando as mesmas tuplas
(ano, habilidade, descrição, dificuldade) reais do banco, para o prompt de
treino continuar idêntico ao que o app manda em produção:

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

### Testando o modelo de verdade (`test_model.py`)

`evaluate.py` mede o modelo em 4-bit via `bitsandbytes`/HF na GPU — um caminho
só de desenvolvimento, que não reflete o app. `test_model.py` chama o mesmo
binário e o mesmo `.gguf` que rodam no celular (`llama-cli`, CPU, Q4_K_M):

```bash
python src/test_model.py                          # menu interativo (usa o DB para sugerir habilidades)
python src/test_model.py --ano "5º" --habilidade H08 \
    --descricao "Resolver problemas de adição ou subtração." --dificuldade Fácil
python src/test_model.py --ano "9º" --habilidade H17 --n 5   # 5 variações do mesmo pedido
python src/test_model.py --batch                   # roda data/val.jsonl inteiro pelo .gguf
python src/test_model.py --batch --num-samples 10  # versão rápida
```

Sem dependência de `torch`/`unsloth` — só precisa do `llama-cli` (gerado por
`export_gguf.py` em `~/.unsloth/llama.cpp/llama-cli`) e do `.gguf` exportado.

### Gabarito inconsistente com a justificativa

Em alguns testes o cálculo na justificativa está certo, mas a letra do
`gabarito` não bate com ele. Duas causas somadas:

1. **Ordem das chaves no JSON treinado**: a geração é autorregressiva —
   antes desta mudança, `gabarito` vinha *antes* de `justificativas` no
   schema, então o modelo comprometia a letra sem ter "mostrado o trabalho"
   ainda (o modo non-thinking não tem `<think>` para isso). Corrigido em
   `extract_data.py`/`SYSTEM_PROMPT`: `justificativas` agora vem antes de
   `gabarito`, forçando um raciocínio curto antes da resposta final. Só tem
   efeito treinando de novo (`python src/extract_data.py` já regera os
   dados; falta rodar `python src/train.py` para o modelo aprender a nova
   ordem).
2. **Qwen3-1.7B é pequeno**: aritmética multi-dígito é um ponto fraco
   conhecido de LLMs nessa faixa sem chain-of-thought, e ~300 exemplos de
   treino ensinam formato/estilo, não ampliam capacidade de cálculo (ver
   "Evolução futura" abaixo).

Como rede de segurança independente de retreino, `schema_utils.check_consistency()`
faz uma checagem determinística: extrai uma expressão "a op b = r" do texto da
justificativa do gabarito e confere se bate com o valor da alternativa
apontada. `evaluate.py` e `test_model.py` reportam isso como
`consistencia_gabarito_pct` (só sobre as amostras onde dá pra verificar —
é uma heurística best-effort para aritmética simples, não substitui
curadoria humana). Quando a conta não bate, a função também tenta indicar
qual alternativa seria a correta (`sugestao_gabarito`), útil tanto para
auditar o dataset quanto para uma futura correção automática no app antes
de exibir a questão ao usuário.

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

Com ~300 exemplos o modelo aprende o formato, mas não amplia conhecimento
matemático. O caminho da literatura para melhorar small language models é
**destilação de dados sintéticos**: usar um modelo grande para gerar milhares
de questões novas por habilidade, filtrá-las (validação automática +
professor) e re-treinar. O pipeline atual já suporta isso — basta adicionar
exemplos ao `data/train.jsonl`.

## Formato dos dados

Entrada (user): `Gere uma questão de matemática. Ano: 5º ano. Habilidade: H08 —
<descritor>. Dificuldade: Fácil.`

Saída (assistant):

```json
{
  "enunciado": "...",
  "comando": "...",
  "alternativas": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "gabarito": "B",
  "justificativas": {"A": "...", "B": "...", "C": "...", "D": "..."}
}
```

Questões com imagem (230 de 553) são excluídas — o modelo em produção é só
texto e não deve aprender a citar figuras inexistentes.

## Hardware

Calibrado para **RTX 3060 Laptop 6GB** (bf16/Ampere), CUDA 12.x, Python 3.11.
Se ocorrer OOM no treino: em `src/train.py`, reduza `MAX_SEQ_LENGTH` para 768
ou `BATCH_SIZE` para 1 (dobrando `GRAD_ACCUMULATION`).

## Métricas de validação (`outputs/eval_report.json`)

| Grupo | Métrica |
|---|---|
| Estrutura | % JSON válido, % schema completo, % gabarito válido, % alternativas distintas, distribuição de gabaritos, % menções a figura, % consistência gabarito↔justificativa (ver seção acima) |
| Linguagem | perplexity da resposta de referência |
| Velocidade | latência média/p95 por questão, tokens/s, tokens de saída |

A velocidade medida por `evaluate.py` é na GPU local; o número real do mobile
deve ser medido com o GGUF: `llama-bench -m outputs/gguf/<modelo>.gguf -p 256 -n 256`.
