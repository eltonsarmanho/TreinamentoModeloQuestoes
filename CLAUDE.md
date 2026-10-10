# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

QLoRA fine-tune of Qwen3-1.7B that generates SAEB-style (Brazilian basic-education
assessment) multiple-choice math questions as structured JSON, quantized to GGUF
Q4_K_M and run **fully offline on a mobile device** via llama.cpp. Published model:
https://huggingface.co/eltonsarmanho/qwen3-1.7b-questoes-matematica.

Two independent subsystems live in this repo:
1. **Training pipeline** — `DB/questoes.db` → dataset → QLoRA fine-tune → GGUF export → gate → publish.
2. **Knowledge-base pipeline** (`src/agentes_questoes.py` and friends) — an LLM-judge
   agent system that audits and expands the training bank itself (separate from
   training/inference code, costs real API money, has its own safety rails).

## Commands

```bash
source venv/bin/activate   # or prefix every command with venv/bin/python

# Training pipeline (see README.md for the full pipeline diagram and rationale)
python src/extract_data.py              # DB/questoes.db -> data/{train,val}.jsonl
python src/generate_synthetic.py        # (optional) adds arithmetic items with Python-computed answers
python src/distill_teacher.py           # (optional) distills from a larger teacher model, costs HF credits
python src/train.py                     # QLoRA fine-tune (RTX 3060 6GB); --train/--val/--out/--checkpoints for control runs
python src/export_gguf.py               # merges LoRA + quantizes to GGUF Q4_K_M
python tests/test_model.py              # interactive: generate one question via the real .gguf/llama-cli
python tests/test_model.py --batch --val data/val_frozen_v1.jsonl --report outputs/relatorios/eval.json  # add --perfil for a latency breakdown
python src/promover_checkpoint.py --perfil multiseed \
  --baseline-gguf-seeds <3 reports> --candidato-gguf-seeds <3 reports>  # promotion gate, see below
python src/push_to_hub.py --repo-id <org>/<name> --lora-dir ... --gguf-dir ... --data-dir ... --model-card ...

# Tests (plain pytest; unittest-style TestCase classes, no pytest.ini)
venv/bin/python -m pytest -q tests                              # full suite
venv/bin/python -m pytest -q tests/test_gate_multiseed.py       # one file
venv/bin/python -m pytest -q tests/test_gate_multiseed.py::TestMultiseed::test_caso_v3_um_item_nao_reprova  # one test
```

There is no lint/format command configured in this repo — don't invent one.

### Running the knowledge-base agent pipeline

Everything under `src/agentes_questoes.py`, `src/auditar_base.py`,
`src/injetar_questoes.py`, `src/arbitro_gemini.py`, `src/calibrar_agentes.py`
calls paid LLM APIs (Maritaca `MARITALK_API_KEY`, optionally Gemini
`GOOGLE_API_KEY` as a second reviewer). **Every one of these CLIs defaults to
`--dry-run` or requires an explicit `--real`/budget flag** — never flip that
default without being asked to spend money. `run_escala_base.sh` is the
idempotent wrapper that runs the full audit+injection cycle in the background
(`--status` / `--parar`); it is retryable because every stage is append-only
and resumes from what's already on disk.

## Architecture

### Training data lineage (the part that isn't obvious from file names alone)

```
DB/questoes.db  (SAEB item bank, SQLite)
     │  extract_data.py (filters: Matemática, no image, A-D complete, valid gabarito)
     ▼
data/train.jsonl, data/val.jsonl          (val.jsonl is FROZEN once created — never regenerate it)
     │  + generate_synthetic.py (arithmetic, answer computed in Python, never by an LLM)
     │  + distill_teacher.py --merge (teacher-model items, filtered by schema_utils.check_consistency)
     │  + curar_diversidade.py --apply (drops near-duplicates, dedupes)
     ▼
data/train_curado.jsonl                   (the curated corpus most tooling reads by default)
     │  + src/auditar_base.py (re-judges every existing item, flags/removes math errors)
     │  + src/injetar_questoes.py (adds items for skills below the per-skill/subtopic/difficulty target)
     ▼
data/train_curado_v3.jsonl                (current training set: ~2.7k examples, 79 skills, all 6 grades)
```

`data/val.jsonl` and `data/val_frozen_v1.jsonl` are **byte-identical across
every cycle** — no new item is ever added — so every promotion gate compares
models on exactly the same questions. `data/val_novos_v1.jsonl` is a separate
holdout used only to measure coverage of grades the base set didn't originally
have. **Never write to `data/train_curado.jsonl`, `data/train.jsonl` or any
`val*.jsonl`** unless explicitly asked — most pipeline scripts treat these as
read-only inputs and write to a different, versioned output file instead.

### The knowledge-base agent pipeline (quality control for the training bank)

This is a separate system from training/inference, built because a blind
deterministic checker (`schema_utils.check_consistency`, which only verifies
arithmetic in `resolucao_passo_a_passo`) cannot catch conceptual math errors
(e.g. "has one property of X, therefore is X" — a 90-45-45 triangle is both
right and isosceles, a square is a rectangle, etc.).

```
generator (Maritaca, sabiá-4-thinking)
   │
   ▼
deterministic filters (schema, consistency, missing data, image, duplicate,
                        verificador_geometria.py, implausible context)
   │
   ▼
validador  — resolves the item BLIND (no gabarito/resolution/difficulty shown), independently
   │
   ▼
revisor    — resolves it independently too, checks skill/grade/distractors/clarity
   │  both must approve
   ▼
[optional] 2º revisor (Gemini, arbitro_gemini.py) — different model family, same
           reviewer prompt + anti-false-positive addendum; only called if the
           first two already approved; degrades gracefully (keeps going without
           Gemini) on budget exhaustion or 3 consecutive API failures
   │
   ▼
entra na base (data/train_curado_v3.jsonl)
```

Key invariant: **validator and reviewer never see the gabarito, the
resolution, or each other's answer** — they each solve the item from scratch
and only then compare. This is what catches the "plausible but wrong"
failure mode a same-family judge pair alone would miss (two same-family
judges can and did agree on the same wrong answer).

A **Gemini arbiter** (`src/arbitro_gemini.py`) sits in front of any
*automatic removal* of an existing item for a confirmed math error (D2) —
single-model mistakes never remove an item unseconded. Human decisions
(`Doc/decisoes_humanas.json`) take precedence over every automatic rule.

A **symbolic geometry verifier** (`src/verificador_geometria.py`) checks
grade-9 triangle/quadrilateral classification questions against their stated
premises. It runs in **shadow mode**: it logs a verdict but does not block or
regenerate — it still abstains on ~40% of grade-9 geometry items. Don't flip
it to active mode without being asked; that's gated on a blind human audit.

### Promotion gate (`src/promover_checkpoint.py`)

Decides PROMOVIDO / NÃO PROMOVIDO mechanically — never by eyeballing numbers.
Three profiles, selected with `--perfil`:
- `n1` (historical): one pass, G1–G9 bloqueantes, gates documented inline in
  the file (each revision is commented with *why* and *when*, e.g. G4's
  2026-09-16 metric fix, G11's 2026-09-30 tolerance unification).
- `planejado`: the app's actual generation mode (N planned calls, 0 regens).
- `multiseed` (current default for new promotions): requires ≥3 seed rounds
  (`tests/test_model.py --seed-rodada N`, same prompts/pairing, different
  generation seeds) so G2/G3 decide on ≥90 paired samples via McNemar, and G6
  (answer-letter bias) on the letter counts summed over all rounds, instead
  of a single 30-item pass. This profile exists because a single item on a
  30-sample pass once failed both G2 and G3 simultaneously — not
  distinguishable from noise. **Any further change to gate criteria must be
  made *after* a reproved verdict and documented in the code as such** — this
  is the established practice (see git history / comments in the file), not
  a one-off.
  `--comparar-inferencia` reuses this profile to compare two *inference
  configurations* of the same `.gguf` (e.g. `--motor cli` vs `--motor server`):
  it requires the same sha256 and a different `report["inferencia"]`, and
  counts latency as a gain.

**Observations and the blind human sample (P0-4, informative only).** Every
verdict (profiles `n1` and `multiseed`) ends with an `OBSERVAÇÃO (não
bloqueante)` block and writes it to `observacoes` in the output JSON
(`src/observacoes_gate.py`): (1) verification coverage, `ok` vs
`nao_verificavel` per round and aggregate; (2) between-round variability of the
same item (mean Jaccard / near-dup % of the delivered statements); (3) rate of
"E = Nenhuma das alternativas anteriores" and of ALL-CAPS statements. 2 and 3
need the question text, which `test_model.batch` now stores in
`detalhes[].obj` (older reports lack it and print `n/d`). They never touch
PROMOVIDO / NÃO PROMOVIDO; making any of them blocking follows the rule above
(only after a reproved verdict, documented in the code). A teacher-graded blind
sample complements the automatic metrics: `python src/amostra_humana.py gerar
--relatorios baseline=a.json candidato=b.json ... --n 40` writes
`folha_professor.csv` (shuffled, opaque ids, no origin hints; hand it to the
teacher, see `Doc/PROCEDIMENTO_AMOSTRA_HUMANA.md`) and `chave_oculta.json`
(keep it away from the teacher); `... apurar --folha folha_preenchida.csv
--chave chave_oculta.json` reports validity per thematic unit
(`src/unidades_tematicas.py`) and per origin with Wilson 95% intervals.

Baseline and candidate are always evaluated on the exact same seeds
(`base_seed` in `test_model.generate_validated`) — this is what makes the
comparison paired instead of two independent noisy samples. `baseline_v1/`
holds the frozen rollback artifact (GGUF + sha256 + `PROCEDENCIA.txt`); any
new baseline should follow the same convention.

### Production inference guard (`generate_validated()` in `tests/test_model.py`)

The path every real generation (app and eval) goes through: GBNF grammar
(`grammars/questao.gbnf`) constrains decoding to the exact JSON contract →
`schema_utils.check_consistency()` checks the answer against the arithmetic
in the solution → best-of-N retry on failure, keeping the best failed
candidate if none pass → `fix_gabarito()` as a last resort, which only
discards the item (`status="falha"`) if nothing else works. `src/gerar_lote.py`
builds a planned N-question batch this way (one call per question against a
subtopic-coverage plan from `src/diversidade.py`), instead of one call asking
for N questions at once — the latter let diversity ride on sampling alone and
only the first question in a batch got the full check.

Generation runs on a persistent `llama-server` by default (`MOTOR = "server"`,
`--motor cli` = one `llama-cli` process per call, kept to reproduce old
baselines): the model loads once and the system-prompt prefix stays in the KV
cache (`cache_prompt`). Always `-c N_CTX` (2048; without it llama.cpp
reserves the model's full 40960-token context, ~4.4 GB of KV) and `-np 1` (auto
slots would split that context). Prompt cache and thread count change the
generated text through floating-point rounding (not bit-identical), so changing
them goes through the multiseed gate. `--perfil` breaks each call into
load / prefill / decode time and peak RSS.

The arithmetic check (`schema_utils._analisar_contas`, extended 2026-10-10) now
reads n-ary sums/products, parentheses, powers, chained equalities
(`a + b = c = d`), units (`cm`, `°`, `R$`, `%`...), pt-BR decimals/thousands
(`1.250,50`), `p% de N`, clock times (`10h30 + 45 min = 11h15`) and `≈` rounding
(a plain `=` stays exact: an approximate `=` never verifies). A resolution whose
final computed result is in *no* alternative (numeric-valued options, tolerant of
`33,3` vs `33%`, `1.250` vs `1250 reais`, `0,25` vs `25%`) is `fora_das_alternativas`
→ `status="falha"` → regenerated; a result that sits in a *different* alternative
still goes to `fix_gabarito`. Measured on `train_curado_v3`: verified (`ok`)
47.5% → 58.9%, `nao_verificavel` 52.5% → 41.1%, 0 new rejections of training
items; the raw rule alone would have rejected 42 correct items, which the tail
guard (`_valor_na_cauda`) and the mention guard (`_valor_mencionado`) absolve — do
not remove them. `avisos_consistencia()` adds a non-blocking `dado_inventado`
warning (operand absent from the statement and from earlier steps; ~5% of items
with a computation, mostly benign) exposed as `avisos_consistencia` in the
`generate_validated` result; it never changes status. `gerar_lote.RANK_STATUS`
now ranks `ok` above `nao_verificavel`. Still blind: a wrong question with a
coherent resolution ("12 m² → 28 m², how many m² now?" marked 16), a gabarito that
matches a wrong calculation the resolution itself made (prism volume without the
`/2`), verbal reasoning without an equation, and label-type alternatives (textual
branch, which only runs for items that already had a binary `a op b = r`). Note
that `promover_checkpoint.mcnemar_planejado` re-judges stored reports with the
*current* verifier, so re-running a gate on old reports can now give different
paired counts (e.g. base_k0 x exp_C: 13 worse / 3 better, p≈0.02).

### Auditoria e padronização da base (2026-10-10)

`src/padronizar_enunciado.py` converte CAIXA ALTA em caixa normal e restaura acentos (Maritaca `sabia-4`; 20 de 20 passaram no invariante contra 6 de 20 do `sabiazinho-4`). O invariante é a trava: o texto novo tem de ser igual ao original sem acentos e sem diferença de maiúsculas, então o modelo não pode trocar palavra nem número; resposta que viola cai no método determinístico. Escreve só um arquivo novo (`data/train_curado_v4.jsonl`), nunca `train_curado.jsonl`, `train.jsonl` nem `val*`; `--real` exige `--max-chamadas`. `src/auditar_gabaritos_base.py` classifica cada gabarito em CONFIRMADO / JUIZES / FRACO / REFUTADO sem chamada paga, e `src/reverificar_gabaritos.py` resolve às cegas (2 permutações) só os FRACO. As trocas de gabarito que ela sugere são PROPOSTAS: item real do banco e `Doc/decisoes_humanas.json` têm precedência e exigem revisão humana. CONFIRMADO só certifica a aritmética.

`src/consolidar_base.py` fecha o ciclo e gera `data/train_curado_v5.jsonl` (v4 menos os itens sem sustentação, com a letra do gabarito balanceada em 20% cada por `permutar_lote`, um único lote para a base inteira). Fases: `figura` (sabia-4: resolvível só com o texto?), `resolver` (sabia-4-thinking, 2 permutações: lê, resolve, marca o gabarito e escreve a resolução, só para os itens do banco sem justificativa "Correto"; troca de gabarito só com confirmação aritmética, senão remove) e `montar` (sem API). Todas têm `--max-chamadas` e abortam, em vez de remover, se o teto ou a API falharem. `auditar_gabaritos_base.py --base data/train_curado_v5.jsonl --rotulo v5` compara o gabarito do banco por CONTEÚDO (o extrator troca "×" por "x" e travessões por hífen; o banco guarda frações como data de planilha). O banco SQLite e a v3 não são alterados.

### Secrets

`.env` holds `MARITALK_API_KEY`, `GOOGLE_API_KEY`, `HF_TOKEN` — never read,
print, or log these. `agentes_questoes.py` has an explicit scrub list
(`_SEGREDOS_ENV`) for anything that touches logs.
