"""Hybrid router: a fast decision model first, the LLM only when the fast model is unsure.

Most tickets are easy, and a fast model gets them right in a few milliseconds. The LLM is
slower and costs money per call, so we only pay for it on tickets where the fast model's
confidence in some field is below `threshold`.

Two ways to combine the answers when a ticket is escalated:
  - "full":  the LLM's whole decision replaces the fast model's.
  - "field": only the fields the fast model was unsure about are replaced; confident fields
             stay. Then a cross-field check fixes combinations the policy rules out
             (a critical or fraud ticket always goes to a human), because mixing two models'
             answers can produce a decision neither of them would have made.
"""

import pandas as pd

from src.interface import DecisionModel, DecisionOutput

# DecisionOutput attribute <- label column name, for the three fields.
FIELDS = {"route_target": "route", "priority_level": "priority", "requires_human": "requires_human"}


def uncertain_fields(output: DecisionOutput, threshold: float) -> list[str]:
    """Label columns whose confidence is below the threshold."""
    if not output.field_confidences:
        return list(FIELDS) if output.confidence < threshold else []
    return [field for field, conf in output.field_confidences.items() if conf < threshold]


def apply_consistency(output: DecisionOutput) -> int:
    """Enforce the policy's cross-field rules in place; returns how many fields it changed."""
    if (output.priority == "critical" or output.route == "fraud") and not output.requires_human:
        output.requires_human = True
        return 1
    return 0


def merge_decisions(fast: DecisionOutput, slow: DecisionOutput, fields: list[str], mode: str) -> tuple[DecisionOutput, int]:
    """Combine an escalated ticket's two answers. Returns (decision, consistency fixes)."""
    if mode == "full":
        return slow, 0
    if mode != "field":
        raise ValueError(f"unknown mode {mode!r}, use 'full' or 'field'")
    merged = DecisionOutput(route=fast.route, priority=fast.priority, requires_human=fast.requires_human,
                            confidence=fast.confidence)
    for field in fields:
        setattr(merged, FIELDS[field], getattr(slow, FIELDS[field]))
    if "route_target" in fields:
        merged.confidence = slow.confidence
    fixes = apply_consistency(merged)
    return merged, fixes


class HybridDecisionModel(DecisionModel):
    """Wraps two DecisionModels: `fast` always runs, `slow` runs only when `fast` is unsure."""

    def __init__(self, fast: DecisionModel, slow: DecisionModel, threshold: float = 0.7, mode: str = "full"):
        if mode not in ("full", "field"):
            raise ValueError(f"unknown mode {mode!r}, use 'full' or 'field'")
        self.fast = fast
        self.slow = slow
        self.threshold = threshold
        self.mode = mode
        self.name = f"hybrid-{mode}:{fast.name}+{slow.name}@{threshold}"
        self.escalations = 0   # how many tickets went to the slow model
        self.total = 0
        self.history: list[dict] = []  # per ticket: escalated?, which fields were replaced, consistency fixes
        self.last_pause_ms = 0.0  # rate-limit pause of the slow model on the latest ticket (not latency)

    def predict_row(self, row) -> DecisionOutput:
        quick = self.fast.predict_one(row)
        self.total += 1
        self.last_pause_ms = 0.0
        fields = uncertain_fields(quick, self.threshold)
        if not fields:
            self.history.append({"escalated": False, "fields": [], "consistency_fixes": 0})
            return quick
        self.escalations += 1
        careful = self.slow.predict_one(row)
        self.last_pause_ms = getattr(self.slow, "last_pause_ms", 0.0)
        decision, fixes = merge_decisions(quick, careful, fields, self.mode)
        # Report the time of both calls: the fast model ran first either way.
        decision.latency_ms = quick.latency_ms + careful.latency_ms
        self.history.append({"escalated": True, "fields": fields, "consistency_fixes": fixes})
        return decision

    def predict_batch(self, df: pd.DataFrame) -> list[DecisionOutput]:
        return [self.predict_row(dict(row)) for _, row in df.iterrows()]

    def predict_one(self, row) -> DecisionOutput:
        return self.predict_row(dict(row))

    def escalation_rate(self) -> float:
        return self.escalations / self.total if self.total else 0.0

    def size_mb(self) -> float:
        return self.fast.size_mb()  # the LLM runs on someone else's servers
