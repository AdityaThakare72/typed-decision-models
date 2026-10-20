# Notes: how this repo works, from first principles

These notes explain each file and the ideas behind it in plain language. Read them alongside the code.

---

## Part 1: The files

### `src/dataset.py`: where the data comes from

**What it does:** makes the support tickets every model trains and tests on.

Each ticket has:
- **text**: `query`, e.g. "my card was used for 3 orders I never placed"
- **tabular columns**: `account_tier` (free/pro/enterprise), `channel` (email/chat/phone),
  `urgency_score` (0–1), `past_failures` (0–5)
- **labels**: a `TypedDecision` with `route_target`, `priority_level`, `requires_human`

**How labels are made:** simple, readable rules.
- `decide_priority` adds up "points" (urgency, tier, past failures, fraud bonus) and buckets them into low/medium/high/critical.
- `decide_requires_human` says yes for fraud, 3+ past failures, critical priority, or high-priority enterprise tickets.

Then 5% of labels are flipped at random (`_add_label_noise`). Real data is messy; without noise every model scores
~100% and the benchmark tells you nothing.

**Why the rules mix text and table:** route depends mostly on the text, while priority and requires_human depend
heavily on the tabular columns. A model has to use *both* to do well. That's the "hybrid" part.

**Two builders:**
- `build_synthetic_dataset(n, seed)`: templates + random slot fills. Fully offline. Same seed → identical rows
  (`numpy.random.default_rng(seed)` is a seeded random generator, like shuffling a deck in a fixed order).
- `load_hf_dataset()`: real customer queries from the Bitext dataset on Hugging Face, with the same metadata and rules
  added. Bitext has no fraud tickets, so that route is empty there.

**Laya helpers:** `laya_questions()`, `row_to_state()`, `to_laya_example()`, `write_laya_jsonl()` turn a ticket into
the `{state, questions, expected}` format Laya's trainer reads.

### `src/interface.py`: the contract every model signs

**What it does:** defines what "a model" means in this repo, and measures it.

- `DecisionOutput`: the answer for one ticket: route, priority, requires_human, confidence, latency_ms.
- `DecisionModel`: an **abstract base class**. Any model must implement `predict_batch(df)`. It gets `predict_one`
  (timed) and `fit` / `size_mb` defaults for free.
- `compute_metrics`: accuracy + macro-F1 for each field, plus **exact match** (all three right at once).
- `measure_latency`: times 100 single-ticket calls, reports p50 / p95.
- `run_benchmark`: loops over models and builds one results table.

**Analogy:** a power socket. RF, XGBoost, BERT and Laya are different appliances, but they all have the same plug
(`DecisionModel`), so one benchmark "socket" works for all of them, and the comparison is fair because every model is
measured the same way.

### `src/models/classical_ml.py`: Random Forest and XGBoost

- `build_preprocessor()` is a `ColumnTransformer`: it sends each column to the right converter and glues the results
  side by side into one big number matrix:
  - text → `TfidfVectorizer` (word and word-pair scores)
  - categories → `OneHotEncoder` (one 0/1 column per value)
  - numbers → `StandardScaler` (shift to mean 0, std 1)
- `ClassicalDecisionModel("rf" | "xgb")` trains **three** pipelines, one per output field, because a standard tree
  classifier predicts one label at a time.
- XGBoost only accepts class ids 0,1,2..., so a `LabelEncoder` maps "billing" ↔ 0 etc.
- Trains with all CPU cores but predicts with one (`n_jobs=1`), because starting a thread pool for a single ticket
  costs more than the prediction.

### `src/models/bert_baseline.py`: BERT

- BERT only reads text, so the metadata is written into the text:
  `"tier: pro | channel: chat | urgency: 0.82 | failures: 2 [SEP] my app keeps crashing"`.
- One model with **12 outputs** = 6 routes + 4 priorities + 2 yes/no. We slice the 12 into three groups and add up
  three cross-entropy losses. One forward pass gives the whole decision.
- A plain PyTorch training loop: batch → forward → loss → backward → optimizer step.

### `src/models/laya_model.py`: Laya

- `LayaDecisionModel(model_id)` wraps any Laya checkpoint: a Hub ID for zero-shot, or a local folder for our fine-tune.
- It sends Laya the ticket as a `state` dict and three typed questions: `choice` for route, `score` for priority,
  `noul` (yes/no) for requires_human.
- `parse_laya_answers` turns Laya's answer dict into a `DecisionOutput`. For priority it takes the most likely level
  from the probabilities instead of Laya's "expected level" (like 1.7), because we need a single label.
- `finetune_laya` writes JSONL and calls `laya.train.finetune` (the package's own trainer). No PEFT/LoRA: Laya is not a
  stock `transformers` model and the package doesn't support it (details in RESEARCH.md).
- Heavy imports (`laya`, `torch`) live inside functions, so tests can import this file on a laptop with no GPU.

### `src/models/llm_router.py`: Gemini as the router

- `GeminiDecisionModel` asks Gemini for the decision. It sends a **system prompt** (team descriptions, the routing
  policy in plain words, 12 labelled example tickets) and the ticket as JSON.
- **Structured output:** we pass a JSON schema (`DECISION_JSON_SCHEMA`) with enums for team and priority, so Gemini
  must answer `{"route_target": "fraud", ...}` instead of free text. `parse_llm_json` still checks it, and a broken
  answer sends the ticket to a human (and is counted as a failure).
- Every call records latency and **token counts** from `usage_metadata`; `cost_usd` turns tokens into dollars with
  the prices in `GEMINI_PRICES_PER_1M`.
- Retries on rate limits (HTTP 429) and server errors with a doubling pause. An optional pause between calls keeps a
  free-tier key under its requests-per-minute limit; that pause is not counted as latency.

### `src/models/hybrid.py`: decision model first, LLM when unsure

`HybridDecisionModel(fast, slow, threshold)` runs the fast model on every ticket and only calls the slow one when the
fast model's **least confident field** (`DecisionOutput.min_confidence()`) is below the threshold.

### `src/router_graph.py`: the LangGraph workflow, reusable

`build_router_graph(model)` builds the notebook-04 graph around any DecisionModel; `run_tickets(graph, df)` sends
tickets through and records where each one landed and how long it took. Notebook 05 uses it to swap routers without
touching the graph.

### `tests/`

Offline pytest suite: no internet, no torch.
- `test_dataset.py`: determinism, valid labels, rules applied correctly, split sizes, Laya JSONL format.
- `test_features.py`: the ColumnTransformer output shape, scaling, unseen categories.
- `test_interface.py`: metrics on a hand-worked example, latency percentiles, the benchmark table, using a
  `DummyModel` (always says billing/medium/no) and an `OracleModel` (reads the answers, must score 1.0).
- `test_models.py`: real RF/XGB fits on 300 rows, plus Laya/BERT decoding logic fed fake outputs.
- `test_llm_router.py`: the Gemini router with a **fake client** (canned answers, fake token counts, fake 429 errors),
  the hybrid's escalation rule, and the LangGraph wrapper end to end. No API key needed.

Run with `uv run pytest`.

### `notebooks/`

01 data + trees · 02 Laya three ways · 03 everything together with charts · 04 the model inside a LangGraph agent ·
05 decision model vs LLM vs hybrid as the router, with latency and dollar cost.
Each has a markdown explanation before every code cell, and a first cell that clones the repo on Colab.

---

## Part 2: The concepts

### Tree ensembles (Random Forest, XGBoost)

A **decision tree** is a flowchart of yes/no questions: "is urgency > 0.7?" → "is tier enterprise?" → "high priority".
One tree overfits easily, so we use many:

- **Random Forest** = many trees, each trained on a random sample of rows and features, then they **vote**.
  Like asking 300 people who each saw part of the evidence; the crowd is more reliable than any one person.
- **XGBoost** = trees built **one after another**, each one trying to fix the mistakes of the trees before it
  ("boosting"). Like a student who reviews only the questions they got wrong last time.

Trees are great with tabular columns (they split on numbers directly) and fast on CPU. They don't understand language;
with TF-IDF they see words as independent counts, not meaning.

**TF-IDF** gives each word a score: high if it appears in *this* ticket (term frequency) but is rare across *all*
tickets (inverse document frequency). "refund" is informative; "the" is not.

### Encoder attention (BERT, and Laya's backbone)

An **encoder** reads the whole input at once. **Attention** lets every word look at every other word and decide which
ones matter for its meaning: in "the charge I did not make", "charge" attends to "not make" and the meaning becomes
"unauthorised charge", not just "charge".

The output is a vector per token. A small classification layer on top turns that into label probabilities. Fine-tuning
nudges all the weights so those vectors become useful for *our* labels.

Trade-off: understands language far better than TF-IDF, but has 100M+ parameters, so it wants a GPU and is slower
per request than a tree.

### Typed decision heads (Laya)

Laya = an encoder (ModernBERT-large) + a **decision head** built for typed questions. Instead of fixed output
classes baked in at training time, the options are *part of the input*: the question's instructions and each option's
description are fed in together with the state, and the head scores every option.

Consequences:
- **Zero-shot works at all:** you can ask about labels it never trained on, because it reads their descriptions.
- **One forward pass answers several questions** of different *types*: choice (pick one), score (ordered levels),
  noul (yes/no).
- **Fine-tuning** teaches it what *your* labels really mean in *your* data.

**Analogy:** a classifier is a vending machine with fixed buttons. A typed decision model is a clerk who reads the
menu you hand over and points at an item, and you can hand them a different menu each time.

### Macro-F1 vs accuracy

**Accuracy** = share of correct answers. If 90% of tickets are "not critical", a model that never says "critical" gets
90% and is useless for critical tickets.

**F1** for one class balances *precision* (when I say critical, am I right?) and *recall* (of all critical tickets, how
many did I catch?). **Macro-F1** averages F1 over classes with equal weight, so ignoring a rare class hurts.

**Exact match** = all three fields correct together. That's what a router actually needs.

### p50 / p95 latency

Time 100 requests and sort them.
- **p50** (median): the 50th one. A typical request.
- **p95**: the 95th one. 95% of requests are faster than this; the slow tail.

**Analogy:** your commute. p50 is a normal day; p95 is a day with bad traffic. You plan around p95, because that's
when you're late. Averages hide the tail, so we report percentiles.

We drop a few **warm-up** calls first: the first requests pay one-off costs (loading, caches, GPU kernel setup) that
real steady-state traffic doesn't.

### Dataclasses

```python
@dataclass
class DecisionOutput:
    route: str
    priority: str
    requires_human: bool
    confidence: float
    latency_ms: float = 0.0
```

`@dataclass` writes `__init__`, `__repr__` and `__eq__` for you from the field list. You get a clean, typed record
(`out.route` rather than `out["route"]`), it prints nicely, and two outputs with the same fields compare equal, which
makes testing easy. **Analogy:** a pre-printed form with labelled boxes instead of a blank sheet of paper.

### Abstract base classes (ABC)

`DecisionModel(ABC)` with an `@abstractmethod predict_batch` means: "you can't create a DecisionModel directly, and any
subclass *must* implement `predict_batch`". Python raises an error at creation time if you forget, instead of failing
halfway through a benchmark.

### Confidence and calibration

A model's "confidence 0.9" should mean "right 90% of the time". Often it doesn't: Laya's model card says it is
**over-confident**. Notebook 02 checks this by bucketing predictions by confidence and comparing with real accuracy.
Notebook 04 uses confidence as a safety valve (low confidence → human), so this matters.

### LangGraph router

LangGraph models an agent workflow as a **graph**: nodes are steps, edges are "what next", and a shared **state** dict
flows through. A **conditional edge** is a function that reads the state and returns the next node's name. Our router
node fills in the typed decision; the conditional edge reads `route` / `requires_human` / `confidence` and picks the
specialist. Because the decision is already typed, there is no text parsing.

### LLM routing, tokens and cost

An LLM router is a generative model asked to fill in the decision. It needs no training data, only a prompt, and it
understands language well. But:
- **Every call is a network round trip** to someone else's servers, plus the time to read the prompt and write the
  answer. Hundreds of milliseconds instead of a few.
- **You pay per token.** A *token* is a chunk of text, roughly ¾ of a word. Input tokens (the prompt, which includes
  our policy and examples on *every* call) and output tokens (the answer, plus any "thinking") are priced separately,
  per million. Cost per ticket = input tokens × input price + output tokens × output price.
- **Numbers are hard to follow from words.** "Weigh urgency, tier and past failures" is fuzzy; a tree learns the exact
  thresholds from data.

**Structured output** (a JSON schema with enums) removes the parsing problem: the model can only answer with a valid
team name and priority level.

### Cascades (the hybrid router)

Send every request to the cheap, fast model; escalate only the ones it is unsure about to the expensive one. Most
traffic is easy, so you pay LLM prices for a small slice.

**Analogy:** a hospital triage nurse sees everyone and calls the specialist only for the hard cases. The trick is the
escalation signal. Here it's the decision model's **least confident field**, because its mistakes are mostly in
priority while its route confidence stays high. That is also why calibrated confidence (see above) matters: a cascade
is only as good as the model's sense of when it might be wrong.
