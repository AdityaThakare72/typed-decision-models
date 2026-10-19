# Research notes: Laya on Hugging Face

Checked October 2026. Goal: find the exact model IDs and usage pattern for Laya before writing any wrapper code.

## What Laya is

Laya is a **non-autoregressive "System 1" decision engine**. You don't generate text token by token. You give it
a `state` (any text / dict) plus a set of typed `questions`, and it answers every question in **one forward
pass** of an encoder, with a probability for each option.

- Code + docs: https://github.com/NandhaKishorM/laya (docs: https://nandhakishorm.github.io/laya/)
- PyPI package: `laya` (checked v0.4.11, Apache-2.0, Python >= 3.10)
- Deps: `torch`, `transformers`, `safetensors`, `huggingface_hub`, `numpy`. Extras include `langgraph`, `serve`, `onnx`.

## Checkpoints

| HF model ID | Encoder | Params | Context | Notes |
|---|---|---|---|---|
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large + decision head | 421M | 512 | General English base. This is our **zero-shot base**. |
| [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large | 421M | 1024 | Fine-tuned on 4 synthetic workflows (agent traces, customer service, invoices, security). Also loadable as `laya.load("convaiinnovations/laya", subfolder="typed-decisions")`. |
| [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | mmBERT-base | 322M | 1024 | 100+ languages. Not used here. |

The model card reports, on its own 400-case typed-decisions benchmark: base `laya` 0.362 accuracy vs
`laya-typed-decisions` 0.766. That is self-reported and on their data, not ours, so we measure ourselves.

There are many community mirrors and conversions (GGUF, ONNX, MLX, e.g. `mys/laya-typed-decisions-GGUF`,
`ggml-org/Laya-GGUF`). We stick to the official `convaiinnovations/*` repos.

## Usage pattern

```python
import laya

agent = laya.load("convaiinnovations/laya")   # ~800 MB download

state = {"query": "I was charged twice", "account_tier": "pro", "urgency_score": 0.8}
questions = {
    "route_target": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "payments, refunds, invoices", "tech_support": "bugs, errors, outages"},
    },
    "priority_level": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["low", "medium", "high", "critical"],
    },
    "requires_human": {
        "type": "noul",  # yes / no question
        "instructions": "This ticket needs a human agent.",
    },
}

result = agent.predict(state, questions)
result["answers"]["route_target"]["choice"]         # chosen label
result["answers"]["priority_level"]["probabilities"]  # {"0": p, "1": p, ...} keyed by level index
result["answers"]["requires_human"]["noul"]          # P(true)
result["answers"][qid]["answer_confidence"]          # prob of the reported answer
```

Our three output fields map straight onto Laya's three question types:

| Our field | Laya question type |
|---|---|
| `route_target` | `choice` |
| `priority_level` | `score` (ordinal levels) |
| `requires_human` | `noul` (yes/no) |

Also: `agent.predict_batch(states, questions)`, `laya.Router` (picks a checkpoint per request), and
`laya.decide(...)` for pydantic / JSON-schema output.

## Fine-tuning

The package ships its own trainer:

```python
from laya.train import finetune, TrainConfig

finetune("train.jsonl", "convaiinnovations/laya", "laya-finetuned/", TrainConfig(epochs=2))
agent = laya.load("laya-finetuned/")
```

- Data: JSONL rows of `{"state": ..., "questions": ..., "expected": {qid: answer}}`. For `score` questions
  `expected` is the level index; for `noul` it is a bool.
- Default loss is **RLCD**: reinforcement learning where the reward is a proper scoring rule (log / spherical /
  ranked probability score), plus soft cross-entropy. It also refits calibration temperatures after training.
- Useful knobs: `epochs`, `micro_batch`, `grad_accum`, `encoder_lr`, `freeze_encoder`, `eval_data`.
- The authors' reference run takes ~4-5 h on Kaggle's free 2xT4. Our dataset is much smaller.

### Why no PEFT / LoRA

Laya is **not** a stock `transformers` class. It is a ModernBERT encoder plus a custom decision head
(loaded through the package's own code), and the package has no PEFT/LoRA support. We could wrap the encoder
with `peft` by hand, but that is an unsupported path. So the "fine-tuned Laya" arm uses the official
`laya.train.finetune`. If GPU memory is tight on Colab, `freeze_encoder=True` trains only the head.

## Gotchas

- Set `USE_TF=0` before importing: an installed TensorFlow can deadlock `transformers` at import (Colab has TF).
- Confidences are **overconfident** (model card ECE ~0.2). Don't treat `confidence` as a calibrated probability
  without refitting temperatures on held-out data.
- Keep `choice` questions under ~20 options. We use 6.
- English only for these checkpoints.

## Baselines used in this repo

- Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- XGBoost: https://xgboost.readthedocs.io/
- BERT: [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased) (110M params)
- Real query text for the Colab dataset variant:
  [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
