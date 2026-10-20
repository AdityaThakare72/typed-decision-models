"""Hybrid router: a fast decision model first, the LLM only when the fast model is unsure.

Most tickets are easy, and a 5 MB tree model gets them right in a few milliseconds. The LLM is
slower and costs money per call, so we only pay for it on the tickets where the fast model's
weakest field confidence is below `threshold`.
"""

import pandas as pd

from src.interface import DecisionModel, DecisionOutput


class HybridDecisionModel(DecisionModel):
    """Wraps two DecisionModels: `fast` always runs, `slow` runs only when `fast` is unsure."""

    def __init__(self, fast: DecisionModel, slow: DecisionModel, threshold: float = 0.7):
        self.fast = fast
        self.slow = slow
        self.threshold = threshold
        self.name = f"hybrid:{fast.name}+{slow.name}@{threshold}"
        self.escalations = 0   # how many tickets went to the slow model
        self.total = 0
        self.last_pause_ms = 0.0  # rate-limit pause of the slow model on the latest ticket (not latency)

    def predict_row(self, row) -> DecisionOutput:
        quick = self.fast.predict_one(row)
        self.total += 1
        self.last_pause_ms = 0.0
        if quick.min_confidence() >= self.threshold:
            return quick
        self.escalations += 1
        careful = self.slow.predict_one(row)
        self.last_pause_ms = getattr(self.slow, "last_pause_ms", 0.0)
        # Report the time of both calls: the fast model ran first either way.
        careful.latency_ms += quick.latency_ms
        return careful

    def predict_batch(self, df: pd.DataFrame) -> list[DecisionOutput]:
        return [self.predict_row(dict(row)) for _, row in df.iterrows()]

    def predict_one(self, row) -> DecisionOutput:
        return self.predict_row(dict(row))

    def escalation_rate(self) -> float:
        return self.escalations / self.total if self.total else 0.0

    def size_mb(self) -> float:
        return self.fast.size_mb()  # the LLM runs on someone else's servers
