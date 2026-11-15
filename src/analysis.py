"""Paired analysis tools: did a change actually help, ticket by ticket?

Comparing two accuracy numbers hides what happened underneath. When a hybrid router replaces
some of Laya's answers with Gemini's, we want to know, on exactly those tickets:
  - fixed:  Laya was wrong, the hybrid is right
  - broke:  Laya was right, the hybrid is wrong
  - unchanged right / unchanged wrong
Net gain = fixed - broke. A small test set also needs an honest error bar, so we bootstrap
the *paired* difference (same tickets, two routers) instead of comparing two separate means.
"""

import numpy as np
import pandas as pd

from src.dataset import LABEL_COLUMNS
from src.interface import DecisionOutput, outputs_to_frame
from src.models.hybrid import merge_decisions, uncertain_fields


def correctness(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """True/False per ticket for each field, plus `exact` (all three right)."""
    pred = predictions[LABEL_COLUMNS].reset_index(drop=True).astype(str)
    true = truth[LABEL_COLUMNS].reset_index(drop=True).astype(str)
    right = pd.DataFrame({column: pred[column] == true[column] for column in LABEL_COLUMNS})
    right["exact"] = right.all(axis=1)
    return right


def transition_table(before: pd.DataFrame, after: pd.DataFrame, truth: pd.DataFrame,
                     mask=None) -> pd.DataFrame:
    """Fixed / broke / unchanged counts per field, on the tickets where `mask` is True (default: all)."""
    right_before = correctness(before, truth)
    right_after = correctness(after, truth)
    if mask is None:
        mask = np.ones(len(right_before), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    rows = []
    for column in right_before.columns:
        b, a = right_before[column].to_numpy()[mask], right_after[column].to_numpy()[mask]
        fixed, broke = int((~b & a).sum()), int((b & ~a).sum())
        rows.append({"field": column, "tickets": int(mask.sum()), "fixed": fixed, "broke": broke,
                     "kept_right": int((b & a).sum()), "stayed_wrong": int((~b & ~a).sum()),
                     "net": fixed - broke})
    return pd.DataFrame(rows).set_index("field")


def paired_bootstrap(correct_a, correct_b, n_boot: int = 10_000, seed: int = 0) -> dict:
    """95% interval for mean(b) - mean(a) on the same tickets, by resampling tickets."""
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return {"diff": float(b.mean() - a.mean()),
            "low": float(np.percentile(diffs, 2.5)), "high": float(np.percentile(diffs, 97.5))}


def simulate_hybrid(fast_outputs: list[DecisionOutput], slow_outputs: list, threshold: float,
                    mode: str) -> tuple[list[DecisionOutput], np.ndarray]:
    """What a HybridDecisionModel would answer, given both models' answers per ticket.

    The hybrid is a fixed function of the two answers, so this needs no extra model calls.
    `slow_outputs[i]` may be None for tickets that never needed escalation at this threshold.
    """
    decisions, escalated = [], []
    for fast, slow in zip(fast_outputs, slow_outputs):
        fields = uncertain_fields(fast, threshold)
        if not fields:
            decisions.append(fast)
            escalated.append(False)
            continue
        if slow is None:
            raise ValueError("a ticket needs escalation but has no slow-model answer")
        decision, _ = merge_decisions(fast, slow, fields, mode)
        decisions.append(decision)
        escalated.append(True)
    return decisions, np.array(escalated)


def pick_threshold(fast_outputs, slow_outputs, truth: pd.DataFrame, thresholds, mode: str) -> tuple[float, pd.DataFrame]:
    """Choose the threshold with the best simulated exact match (ties -> fewer escalations)."""
    rows = []
    for t in thresholds:
        decisions, escalated = simulate_hybrid(fast_outputs, slow_outputs, t, mode)
        rows.append({"threshold": t, "share_escalated": escalated.mean(),
                     "exact_match": correctness(outputs_to_frame(decisions), truth)["exact"].mean()})
    table = pd.DataFrame(rows).set_index("threshold")
    best = table.sort_values(["exact_match", "share_escalated"], ascending=[False, True]).index[0]
    return float(best), table
