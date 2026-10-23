# Pipeline de treino — variante Gemma 3 1B-it

Scripts em `src/gemma/`. Modelo: `unsloth/gemma-3-1b-it` (Apache 2.0), 1B parâmetros, puro texto
(sem visão/áudio). É `Gemma3ForCausalLM` — pega o kernel de cross-entropy eficiente do Unsloth, o
que o Gemma 4 E2B (`Gemma4ForConditionalGeneration`) não pegava, e foi por isso que ele estourou os
5,67 GB utilizáveis da GPU mesmo em QLoRA 4-bit (ver histórico abaixo).

A variante fica neste repositório e reusa a base, o val congelado, o harness de inferência
(`tests/test_model.py`), o manifesto e o gate. Os artefatos vão para `outputs/gemma3/`.

## Por que Gemma 3 1B e não Gemma 2 2B

- Geração mais nova, com mais dados multilíngues no pré-treino — importa para pt-BR.
- Ganhos documentados em matemática/seguimento de instrução sobre o Gemma 2, na mesma faixa de
  tamanho — o ponto fraco que o projeto já mediu no ciclo v7 (correção matemática).
- Os dois são `*ForCausalLM` e cabem tranquilos em 6 GB; a diferença é de qualidade esperada, não
  de capacidade da máquina.
- Ressalva: o Gemma 2 2B tem mais parâmetros (2,6B x 1B). Se o Gemma 3 1B ficar aquém na correção
  matemática medida pela amostra humana, o Gemma 2 2B é o próximo candidato a testar — não um
  beco sem saída.
- Nenhum dos dois tem um turno de `system` de verdade: o Gemma 2 **rejeita** mensagem de `system`
  (levanta exceção no template); o Gemma 3 apenas funde o conteúdo do `system` como prefixo do
  primeiro turno de `user`, sem erro. `gemma_formato.py` e `train_gemma.py` já lidam com isso —
  passam as mensagens (system, user, assistant) direto para `apply_chat_template`, que faz a fusão.

## Histórico: por que não o Gemma 4 E2B

A primeira tentativa foi o Gemma 4 E2B-it (2,3B efetivos, 5,1B com as Per-Layer Embeddings).
Deu `CUDA out of memory` tanto com `max_seq_length=1024` quanto com 768. Causa raiz: o checkpoint
é `Gemma4ForConditionalGeneration`, e o kernel de cross-entropy fundido do Unsloth
(`unsloth_zoo/fused_losses/forward_install.py`, `_is_eligible_class`) só se aplica a classes que
terminam em `ForCausalLM` — o E2B ficava de fora, materializando os logits inteiros (262.144 de
vocabulário) em memória cheia durante o backward. Isso, somado ao próprio piso do Unsloth de 8–10 GB
para LoRA/QLoRA do E2B, tornava o treino inviável nesta GPU. Os artefatos dessa tentativa (GGUF
zero-shot parcial, cache HF de ~17 GB) foram removidos; não há vestígio de Gemma 4 no repositório.

## Regras da variante

1. **Gemma só se compara com Gemma.** O baseline do gate é o **Gemma 3 1B-it zero-shot**,
   exportado pelo mesmo caminho do candidato (`export_gguf_gemma.py --base-only`). O gate aborta
   se os relatórios trazem `familia_modelo` diferente (`gemma3` x `qwen3`).
2. **Gates:** são os do perfil `multiseed`, com `--sem-dificuldade`. O G5 (aderência à
   dificuldade) passa a ser informativo e a aderência sai da lista de ganhos. Motivo: o professor
   usa uma régua uma casa abaixo do rótulo SAEB, com kappa de 0,20 a 0,24 (`ciclo_v7/SPEC.md`).
   Os demais gates seguem bloqueantes como estão: G1\*, G2\*, G3\*, G4, G6\*, G7\* e G8.
3. **Base:** `data/train_curado_v7b_fix1.jsonl`, sha256 `7e63bc2c…`, com 2.846 exemplos. É a
   v7b (curadoria item a item + reforço de distratores) com a correção humana do D4861. O val é
   o `data/val_frozen_v1.jsonl`.
4. **Regime de prompt:** a avaliação usa `--sufixo`, que é o regime do app e de ~90% do treino.
5. **O gate não basta:** o PROMOVIDO mecânico já discordou da amostra humana três vezes na linha
   Qwen (v6, v7b, v7b-s44). A decisão final vem da amostra humana cega, pelo critério núcleo.
6. **Sem publicação** no HF sem decisão explícita do responsável.

## Hiperparâmetros (e por quê)

| Parâmetro | Valor | Fonte / motivo |
|---|---|---|
| Quantização do treino | QLoRA NF4 4-bit | Cabe com folga em 6 GB para 1B parâmetros; mesma receita da linha Qwen. |
| Loader | `FastModel`, só camadas de linguagem | Notebook oficial do Unsloth para Gemma 3 (1B/4B). |
| Módulos | attention + MLP | LoRA só em attention rende menos, mesmo com o mesmo número de parâmetros ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/), Thinking Machines, 2025). |
| r / alpha / dropout | 16 / 32 / 0 | Base de ~1,4M tokens supervisionados: r=16 fica dentro da capacidade em que o LoRA empata com o full FT (LoRA Without Regret). O notebook oficial usa r=8/α=8 (demo de 30-60 passos); mantivemos a receita já usada no projeto. |
| Learning rate | 2e-4 | LR ótimo do LoRA ≈ 10x o do full FT (LoRA Without Regret). Padrão do QLoRA e do notebook oficial. |
| Batch efetivo | 16 (= 1 x 16) | O LoRA perde mais que o full FT com batch grande (LoRA Without Regret). Com 1B parâmetros há folga de VRAM para subir `--batch-size` se quiser velocidade; não mudamos o padrão sem medir no gate. |
| Scheduler / warmup | cosine / 5% | Igual à linha Qwen. |
| max_grad_norm | 0,3 | QLoRA (Dettmers et al., 2023). |
| Weight decay | 0,01 | Igual à linha Qwen. |
| Épocas | **2** (não 3) | Nas 8 runs Qwen sobre a v7b, a melhor época foi sempre a 1 ou a 2, e a 3ª sempre piorou o eval_loss (`ciclo_v7/SPEC.md`, Fase 2). O melhor checkpoint é escolhido por eval_loss. |
| Precisão | bf16 | RTX 3060 (Ampere) suporta bf16 nativamente. |
| max_seq_length | 1024 | Confirme com `--checar-formato` antes de treinar (a contagem de truncados depende do tokenizer do Gemma 3, diferente do Qwen). |
| Loss | só nos tokens do modelo | `train_on_responses_only` com `<start_of_turn>user\n` / `<start_of_turn>model\n`. Texto vai sem `<bos>` (o SFTTrainer põe um). |
| Seeds | 42, 43, 44 | O ruído de treino a treino já mudou o veredito do gate na linha Qwen (v7b x v7d). Uma seed sozinha não decide. |

Cautelas de literatura:
- [LoRA Learns Less and Forgets Less](https://iclr.cc/virtual/2025/poster/31465) (Biderman et
  al.) acha o LoRA abaixo do full FT em domínios novos. Aqui o alvo é formato e estilo SAEB, o
  regime em que o LoRA empata, e o lado "esquece menos" protege a capacidade geral do modelo.
- A amostragem do harness (temp 0,7, top_p 0,8, herdada do Qwen) não é a recomendada pelo Gemma
  (1,0 / 0,95 / top_k 64). Ela fica igual nos dois braços para o pareamento valer. Trocar a
  amostragem é outro experimento e precisa passar pelo gate.

## Comandos (rodar manualmente, nesta ordem)

```bash
cd ~/Documents/VSCodigo/TreinamentoModeloQuestoes
source venv/bin/activate
TRAIN=data/train_curado_v7b_fix1.jsonl
VAL=data/val_frozen_v1.jsonl
mkdir -p outputs/gemma3 outputs/relatorios/gemma3
```

### 0. Pré-checagem do formato (sem GPU, ~1 min)
Renderiza os 2.876 exemplos com o chat template nativo e checa marcadores, `<bos>`, se o prompt
de geração é prefixo do texto de treino e as truncagens.
```bash
python src/gemma/train_gemma.py --checar-formato --train $TRAIN --val $VAL
```

### 1. Baseline zero-shot (export do base sem LoRA)
O Unsloth grava o `.gguf` em `<out>/` ou em `<out>_gguf/` (varia); o fim do export imprime o
caminho exato — use-o. O glob abaixo pega qualquer um dos dois.
```bash
python src/gemma/export_gguf_gemma.py --base-only --out outputs/gemma3/gguf_zeroshot
ZS=$(ls outputs/gemma3/gguf_zeroshot/*.gguf outputs/gemma3/gguf_zeroshot_gguf/*.gguf 2>/dev/null | head -1)
python src/gemma/verificar_template_gemma.py --gguf "$ZS"
```

### 2. Treino (uma seed por vez; repita para 43 e 44)
```bash
S=42
python src/gemma/train_gemma.py --train $TRAIN --val $VAL --seed $S \
  --out outputs/gemma3/lora_s$S --checkpoints outputs/gemma3/checkpoints_s$S
```
Se faltar memória (improvável em 1B, mas por segurança): `--max-seq 768` ou `--batch-size 1
--grad-accum 16` (já é o padrão). Não deve ocorrer no 1B — se ocorrer, é sinal de outro processo
disputando a GPU (`nvidia-smi`).

### 3. Export do candidato + checagem do template
```bash
python src/gemma/export_gguf_gemma.py --lora outputs/gemma3/lora_s$S --out outputs/gemma3/gguf_s$S
FT=$(ls outputs/gemma3/gguf_s$S/*.gguf outputs/gemma3/gguf_s${S}_gguf/*.gguf 2>/dev/null | head -1)
python src/gemma/verificar_template_gemma.py --tokenizer outputs/gemma3/lora_s$S --gguf "$FT"
```

### 4. Avaliação multiseed (5 rodadas por braço, regime do app)
O baseline roda uma vez só e serve para as três seeds de treino.
```bash
ZS=$(ls outputs/gemma3/gguf_zeroshot/*.gguf outputs/gemma3/gguf_zeroshot_gguf/*.gguf 2>/dev/null | head -1)
FT=$(ls outputs/gemma3/gguf_s$S/*.gguf outputs/gemma3/gguf_s${S}_gguf/*.gguf 2>/dev/null | head -1)
for R in 0 1 2 3 4; do
  python tests/test_model.py --batch --val $VAL --sufixo --seed-rodada $R --model $ZS \
    --report outputs/relatorios/gemma3/eval_zs_suf_r$R.json
done
for R in 0 1 2 3 4; do
  python tests/test_model.py --batch --val $VAL --sufixo --seed-rodada $R --model $FT \
    --report outputs/relatorios/gemma3/eval_ft_s${S}_suf_r$R.json
done
```

### 5. Gate de promoção (Gemma x Gemma, sem o gate de dificuldade)
```bash
python src/promover_checkpoint.py --perfil multiseed --sem-dificuldade \
  --baseline-gguf-seeds outputs/relatorios/gemma3/eval_zs_suf_r{0,1,2,3,4}.json \
  --candidato-gguf-seeds outputs/relatorios/gemma3/eval_ft_s${S}_suf_r{0,1,2,3,4}.json \
  --saida outputs/relatorios/gemma3/VEREDITO_gemma3_s${S}_vs_zs_multiseed_n150.json
```

### 6. Amostra humana cega (critério núcleo; decide de fato)
```bash
python src/amostra_humana.py gerar \
  --relatorios baseline=outputs/relatorios/gemma3/eval_zs_suf_r1.json \
               candidato=outputs/relatorios/gemma3/eval_ft_s${S}_suf_r1.json \
  --n 40 --saida-dir outputs/gemma3/amostra_humana_s$S
# entregar só folha_professor.csv ao professor (Doc/PROCEDIMENTO_AMOSTRA_HUMANA.md), depois:
python src/amostra_humana.py apurar --folha <folha_preenchida.csv> \
  --chave outputs/gemma3/amostra_humana_s$S/chave_oculta.json
```

### Disco e tempo (estimativas, não medidas)
- O export pede ≥ 8 GB livres. O Q4_K_M do Gemma 3 1B deve ficar em torno de 0,7–0,9 GB —
  menor que o 1,1 GB do Qwen3-1.7B.
- Meça o pico de RSS com `--perfil` antes de pensar no mobile.

## Arquivos

| Arquivo | Papel |
|---|---|
| `src/gemma/train_gemma.py` | Treino QLoRA (e `--checar-formato`) |
| `src/gemma/export_gguf_gemma.py` | `save_pretrained_gguf` direto (Gemma 3 é `ForCausalLM`, sem contorno); `--base-only` gera o baseline |
| `src/gemma/verificar_template_gemma.py` | Compara o prompt do tokenizer de treino com o do llama-server sobre o `.gguf` |
| `src/gemma/gemma_formato.py` | Marcadores e checagens do formato de chat (stdlib) |
| `src/gguf_meta.py` | Lê `general.architecture` do `.gguf` (stdlib). Fica em `src/` porque o harness e o gate da linha Qwen também o usam |
| `tests/test_gemma_pipeline.py` | Testes de tudo acima e da trava de família do gate |
