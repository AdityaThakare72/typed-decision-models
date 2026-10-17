# typed-decision-models

A small, reproducible benchmark: how good is **Laya**, a typed decision model from Hugging Face, at an
agent-routing task, compared with Random Forest, XGBoost and a fine-tuned BERT?

## What's a typed decision model?

A generative LLM writes text and you parse an answer out of it. A **typed decision model** skips the writing:
you give it a state and some *typed questions* (pick one of N, rate on a scale, yes/no), and it returns a
probability for every option in a single forward pass. That makes it a good fit for the "router" step at the start
of an agent workflow, where you need a few structured fields fast and reliably.

Here every model has to produce the same structured output for a support ticket:

```python
TypedDecision(route_target="fraud", priority_level="high", requires_human=True)
```

from the ticket text plus tabular metadata (`account_tier`, `channel`, `urgency_score`, `past_failures`).

## Why compare against RF / XGBoost / BERT?

- **RF / XGBoost** on TF-IDF + tabular features are the cheap, fast, strong default. If a 400M-parameter model
  can't beat them, it isn't worth the GPU.
- **BERT** (`bert-base-uncased`, fine-tuned) is the standard "small transformer classifier" baseline.
- **Laya** is tested three ways: zero-shot base, zero-shot `laya-typed-decisions`, and fine-tuned on our data.

See [RESEARCH.md](RESEARCH.md) for the Hugging Face model IDs, the API, and caveats.

## How to run

Locally (classical models + tests, CPU only):

```bash
uv sync --group dev
uv run python -m src.dataset      # builds data/train.csv, val.csv, test.csv
uv run pytest                     # offline tests
```

Optional extras: `uv sync --extra laya --extra bert --extra graph --extra viz`.

On Colab (GPU for Laya / BERT), open the notebooks in order. Each one clones this repo in its first cell:

1. `notebooks/01_dataset_and_classical_ml.ipynb`: EDA, RF + XGBoost
2. `notebooks/02_laya_base_vs_finetuned.ipynb`: Laya zero-shot vs fine-tuned
3. `notebooks/03_full_benchmark_laya_vs_xgboost_rf_bert.ipynb`: everything head to head, with charts
4. `notebooks/04_langgraph_decision_router.ipynb`: the model as a LangGraph router node

## Project structure

```
src/
  dataset.py            dataset builder (offline synthetic + Hugging Face loader), Laya JSONL helpers
  interface.py          DecisionModel base class, DecisionOutput, metrics, p50/p95 latency, run_benchmark
  models/
    classical_ml.py     RF + XGBoost (TF-IDF + one-hot + scaler in a ColumnTransformer)
    laya_model.py       Laya wrapper (any checkpoint) + finetune_laya
    bert_baseline.py    bert-base-uncased with 3 heads
tests/                  offline pytest suite (no GPU / internet needed)
notebooks/              Colab-ready notebooks 01-04
data/                   generated CSVs (git-ignored)
RESEARCH.md             Laya findings + links
NOTES.md                file-by-file explainer + concepts
```

## Results

Test set: 450 tickets (synthetic, seed 42). Latency is single-ticket p50 / p95 over 100 calls.
Classical models on CPU, transformers on a Colab T4.

| model | exact match | route F1 | priority F1 | requires-human F1 | size (MB) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|---|
| Random Forest | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| XGBoost | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| BERT (fine-tuned) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Laya base (zero-shot) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Laya typed-decisions (zero-shot) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Laya fine-tuned | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

Notebook 03 writes this table to `artifacts/results_full.md`.

## Licenses of the models used

Laya checkpoints: Apache-2.0. bert-base-uncased: Apache-2.0. See each model card.
