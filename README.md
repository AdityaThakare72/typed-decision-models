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

Optional extras: `uv sync --extra laya --extra bert --extra graph --extra llm --extra viz`.

On Colab (GPU for Laya / BERT), open the notebooks in order. Each one clones this repo in its first cell:

1. `notebooks/01_dataset_and_classical_ml.ipynb`: EDA, RF + XGBoost
2. `notebooks/02_laya_base_vs_finetuned.ipynb`: Laya zero-shot vs fine-tuned
3. `notebooks/03_full_benchmark_laya_vs_xgboost_rf_bert.ipynb`: everything head to head, with charts
4. `notebooks/04_langgraph_decision_router.ipynb`: fine-tuned Laya as a LangGraph router node
5. `notebooks/05_laya_vs_llm_router.ipynb`: same graph, Laya vs Gemini vs hybrid (Laya first, Gemini when unsure),
   latency + $ cost (needs the fine-tuned Laya on Drive and a Gemini API key)

## Project structure

```
src/
  dataset.py            dataset builder (offline synthetic + Hugging Face loader), Laya JSONL helpers
  interface.py          DecisionModel base class, DecisionOutput, metrics, p50/p95 latency, run_benchmark
  router_graph.py       LangGraph support workflow with any DecisionModel as the router node
  models/
    classical_ml.py     RF + XGBoost (TF-IDF + one-hot + scaler in a ColumnTransformer)
    laya_model.py       Laya wrapper (any checkpoint) + finetune_laya
    bert_baseline.py    bert-base-uncased with 3 heads
    llm_router.py       Gemini as a router (structured JSON output) + token / $ tracking
    hybrid.py           decision model first, LLM only for unsure tickets
tests/                  offline pytest suite (no GPU / internet / API key needed)
notebooks/              Colab-ready notebooks 01-05
data/                   generated CSVs (git-ignored)
RESEARCH.md             Laya findings + links
NOTES.md                file-by-file explainer + concepts
```

## Results

Test set: 450 tickets (synthetic, seed 42). Latency is single-ticket p50 / p95 over 100 calls.
Classical models on Colab CPU, transformers on a Colab T4 GPU. One run, one seed.

| model | exact match | route F1 | priority F1 | requires-human F1 | size (MB) | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.598 | 0.944 | 0.538 | 0.892 | 67.5 | 71.5 | 88.3 |
| **XGBoost** | **0.736** | 0.951 | 0.641 | **0.924** | **5.4** | 17.4 | 19.0 |
| BERT (fine-tuned, 3 epochs) | 0.667 | 0.955 | 0.555 | 0.921 | 417.7 | **10.5** | **14.9** |
| Laya base (zero-shot) | 0.173 | 0.615 | 0.190 | 0.530 | 1607.1 | 45.8 | 49.3 |
| Laya typed-decisions (zero-shot) | 0.296 | 0.674 | 0.302 | 0.574 | 1607.1 | 46.1 | 51.8 |
| Laya fine-tuned (2 epochs) | 0.720 | **0.958** | **0.679** | 0.916 | 1607.1 | 46.5 | 54.1 |

Exact match = all three fields right. Full numbers (accuracy too) are in notebook 03 / `artifacts/results_full.md`.

### What it says

- **Fine-tuned Laya and XGBoost tie on exact match** (0.720 vs 0.736, ~7 tickets apart; one standard error on 450
  tickets is ~0.02). Laya is best on priority, the field that needs text and metadata together; XGBoost is best on
  requires-human.
- **Routing is saturated.** With 5% label noise the best possible route accuracy is ~0.958, and fine-tuned Laya hits
  it. The models only really differ on priority.
- **Zero-shot Laya doesn't work for a custom schema** (0.17 / 0.30 exact match). Fine-tuning took it to 0.72 and fixed
  its calibration (held-out ECE 0.17 → 0.015).
- **Cost:** for about the same quality, XGBoost is ~300x smaller and runs faster on a CPU than Laya does on a GPU.
  Laya is the pick when you need calibrated confidences, or want to add new questions without re-engineering features.
- **BERT** is the fastest (10.5 ms on GPU) but reads numbers as text, which hurts priority.
- In the LangGraph workflow (notebook 04) with fine-tuned Laya as the router, 90 of 100 test tickets reached the right
  node; the router took p50 53 / p95 136 ms of the 57 / 142 ms per ticket on a T4. (XGBoost as the router: 92 of 100,
  17 / 24 ms on CPU.)

Caveats: synthetic data with rule-based labels (trees are well suited to learning thresholds), one seed, CPU vs GPU
latency.

### Laya vs LLM router (notebook 05)

Same LangGraph workflow, same 100 test tickets, only the router node changes. Gemini gets the routing policy in words
plus 12 labelled examples and answers in schema-checked JSON. The hybrid runs Laya on every ticket and sends it to
Gemini only when Laya's least confident field is below the threshold.

| router | correct destination | exact match | router p50 / p95 (ms) | LLM share | $ per 1k tickets |
|---|---|---|---|---|---|
| Laya fine-tuned (T4) | TODO | TODO | TODO | 0% | TODO |
| Gemini only (`gemini-3.1-flash-lite`) | TODO | TODO | TODO | 100% | TODO |
| Hybrid Laya→Gemini | TODO | TODO | TODO | TODO | TODO |
| XGBoost (reference, CPU) | TODO | TODO | TODO | 0% | TODO |

Gemini prices from the [official pricing page](https://ai.google.dev/gemini-api/docs/pricing) (checked 2026-10-19).
Local compute is priced at an assumed $0.04/h CPU and $0.35/h T4 GPU.

## Licenses of the models used

Laya checkpoints: Apache-2.0. bert-base-uncased: Apache-2.0. See each model card.
