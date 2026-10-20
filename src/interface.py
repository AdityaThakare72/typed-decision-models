"""The common interface every model implements, plus the benchmark runner.

Idea: RF, XGBoost, BERT and Laya are very different inside, but from the outside they all
take a ticket and return a DecisionOutput. If they share one interface, the benchmark code
only has to be written once and the comparison is fair (same data, same metrics, same timer).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import LABEL_COLUMNS


@dataclass
class DecisionOutput:
    """What any model returns for one ticket."""

    route: str
    priority: str
    requires_human: bool
    confidence: float      # model's confidence in the route (0..1). Not always calibrated!
    latency_ms: float = 0.0
    # Optional per-field confidences, e.g. {"route_target": 0.98, "priority_level": 0.55, ...}.
    # Route confidence alone hides most mistakes (they're usually in priority), so a router
    # that wants to escalate unsure tickets should look at the weakest field.
    field_confidences: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def min_confidence(self) -> float:
        """Confidence of the least sure field (falls back to the route confidence)."""
        if self.field_confidences:
            return min(self.field_confidences.values())
        return self.confidence


class DecisionModel(ABC):
    """Base class. Subclasses must implement `predict_batch`; everything else has a default."""

    name: str = "base"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "DecisionModel":
        """Train on a DataFrame with feature + label columns. Zero-shot models just skip this."""
        return self

    @abstractmethod
    def predict_batch(self, df: pd.DataFrame) -> list[DecisionOutput]:
        """Predict a decision for every row of df."""

    def predict_one(self, row: dict | pd.Series) -> DecisionOutput:
        """Predict a single ticket and record how long it took.

        This is the call an agent router makes at runtime (one request at a time),
        so it is also what we time for latency.
        """
        one_row_df = pd.DataFrame([dict(row)])
        start = time.perf_counter()
        output = self.predict_batch(one_row_df)[0]
        output.latency_ms = (time.perf_counter() - start) * 1000.0
        return output

    def size_mb(self) -> float:
        """Approximate model size on disk/in memory. 0 means 'unknown'."""
        return 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def outputs_to_frame(outputs: list[DecisionOutput]) -> pd.DataFrame:
    """List of DecisionOutput -> DataFrame with the same column names as the labels."""
    return pd.DataFrame({
        "route_target": [o.route for o in outputs],
        "priority_level": [o.priority for o in outputs],
        "requires_human": [bool(o.requires_human) for o in outputs],
    })


def compute_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    """Accuracy and macro-F1 for each of the three heads, plus exact match of the whole decision.

    Macro-F1 = average F1 over classes, each class counted equally. It punishes a model that
    ignores rare classes (like "critical") even when plain accuracy looks fine.
    """
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    metrics = {}
    for column in LABEL_COLUMNS:
        truth = y_true[column].astype(str)
        guess = y_pred[column].astype(str)
        short = column.split("_")[0]  # route / priority / requires
        metrics[f"{short}_acc"] = accuracy_score(truth, guess)
        metrics[f"{short}_f1"] = f1_score(truth, guess, average="macro", zero_division=0)

    # Exact match: all three fields right at once. This is what matters for an agent router.
    all_right = np.ones(len(y_true), dtype=bool)
    for column in LABEL_COLUMNS:
        all_right &= (y_true[column].astype(str) == y_pred[column].astype(str)).to_numpy()
    metrics["exact_match"] = float(all_right.mean()) if len(y_true) else 0.0
    return metrics


def measure_latency(model: DecisionModel, df: pd.DataFrame, n_runs: int = 100, warmup: int = 5) -> dict:
    """Time `n_runs` single-ticket predictions and report p50 / p95 in milliseconds.

    p50 = median (a typical request). p95 = 95% of requests are faster than this
    (the slow tail your users actually notice). Warm-up runs are thrown away because the
    first calls pay one-off costs (lazy loading, caches, GPU kernel compile).
    """
    rows = df.to_dict(orient="records")
    if not rows:
        raise ValueError("need at least one row to time")
    for i in range(warmup):
        model.predict_one(rows[i % len(rows)])

    timings = []
    for i in range(n_runs):
        out = model.predict_one(rows[i % len(rows)])
        timings.append(out.latency_ms)
    return {
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def evaluate_model(model: DecisionModel, test_df: pd.DataFrame, n_latency_runs: int = 100) -> dict:
    """Quality metrics on the whole test set + latency + size, as one flat dict."""
    outputs = model.predict_batch(test_df)
    y_true = test_df[LABEL_COLUMNS]
    result = {"model": model.name}
    result.update(compute_metrics(y_true, outputs_to_frame(outputs)))
    result.update(measure_latency(model, test_df, n_runs=n_latency_runs))
    result["size_mb"] = model.size_mb()
    return result


def run_benchmark(models: list[DecisionModel], test_df: pd.DataFrame, n_latency_runs: int = 100) -> pd.DataFrame:
    """Evaluate every model on the same test set and return one results table."""
    rows = []
    for model in models:
        print(f"evaluating {model.name} ...")
        rows.append(evaluate_model(model, test_df, n_latency_runs))
    return pd.DataFrame(rows).set_index("model")
