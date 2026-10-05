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
python tests/test_model.py --batch --val data/val_frozen_v1.jsonl --report outputs/relatorios/eval.json
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
  generation seeds) so G2/G3 decide on ≥90 paired samples via McNemar instead
  of a single 30-item pass. This profile exists because a single item on a
  30-sample pass once failed both G2 and G3 simultaneously — not
  distinguishable from noise. **Any further change to gate criteria must be
  made *after* a reproved verdict and documented in the code as such** — this
  is the established practice (see git history / comments in the file), not
  a one-off.

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

### Secrets

`.env` holds `MARITALK_API_KEY`, `GOOGLE_API_KEY`, `HF_TOKEN` — never read,
print, or log these. `agentes_questoes.py` has an explicit scrub list
(`_SEGREDOS_ENV`) for anything that touches logs.
