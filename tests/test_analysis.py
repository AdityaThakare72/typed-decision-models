"""Tests for the hybrid merge modes and the paired analysis helpers."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import correctness, paired_bootstrap, pick_threshold, simulate_hybrid, transition_table
from src.interface import DecisionOutput, outputs_to_frame
from src.models.hybrid import apply_consistency, merge_decisions, uncertain_fields


def out(route, priority, human, conf=0.9, priority_conf=0.9, human_conf=0.9):
    return DecisionOutput(route, priority, human, confidence=conf,
                          field_confidences={"route_target": conf, "priority_level": priority_conf,
                                             "requires_human": human_conf})


def test_uncertain_fields():
    assert uncertain_fields(out("billing", "low", False, priority_conf=0.4), 0.7) == ["priority_level"]
    assert uncertain_fields(out("billing", "low", False), 0.7) == []
    assert uncertain_fields(DecisionOutput("billing", "low", False, confidence=0.3), 0.7) == [
        "route_target", "priority_level", "requires_human"]


def test_full_mode_takes_everything_from_slow():
    fast = out("billing", "low", False, priority_conf=0.4)
    slow = out("orders", "high", True)
    merged, fixes = merge_decisions(fast, slow, ["priority_level"], "full")
    assert (merged.route, merged.priority, merged.requires_human) == ("orders", "high", True)
    assert fixes == 0


def test_field_mode_only_replaces_unsure_fields():
    fast = out("billing", "low", False, priority_conf=0.4)
    slow = out("orders", "high", True)
    merged, fixes = merge_decisions(fast, slow, ["priority_level"], "field")
    assert (merged.route, merged.priority, merged.requires_human) == ("billing", "high", False)
    assert fixes == 0


def test_field_mode_consistency_fix():
    fast = out("billing", "low", False, priority_conf=0.4)
    slow = out("billing", "critical", False)
    merged, fixes = merge_decisions(fast, slow, ["priority_level"], "field")
    assert merged.priority == "critical" and merged.requires_human is True and fixes == 1
    fraud = DecisionOutput("fraud", "low", False, confidence=0.9)
    assert apply_consistency(fraud) == 1 and fraud.requires_human is True


def test_unknown_mode():
    with pytest.raises(ValueError):
        merge_decisions(out("billing", "low", False), out("billing", "low", False), [], "half")


TRUTH = pd.DataFrame({"route_target": ["billing", "orders", "fraud"],
                      "priority_level": ["low", "high", "high"],
                      "requires_human": [False, False, True]})


def test_correctness_and_transitions():
    before = outputs_to_frame([out("billing", "low", False), out("orders", "low", False), out("fraud", "high", True)])
    after = outputs_to_frame([out("billing", "low", False), out("orders", "high", False), out("fraud", "low", True)])
    right = correctness(after, TRUTH)
    assert right["exact"].tolist() == [True, True, False]
    table = transition_table(before, after, TRUTH)
    assert table.loc["priority_level", "fixed"] == 1
    assert table.loc["priority_level", "broke"] == 1
    assert table.loc["exact", "net"] == 0
    masked = transition_table(before, after, TRUTH, mask=[False, True, False])
    assert masked.loc["exact", "tickets"] == 1 and masked.loc["exact", "fixed"] == 1


def test_paired_bootstrap():
    a = np.array([1, 0, 1, 0] * 25)
    result = paired_bootstrap(a, a)
    assert result["diff"] == 0 and result["low"] == 0 and result["high"] == 0
    better = paired_bootstrap(np.zeros(100), np.ones(100))
    assert better["diff"] == 1 and better["low"] == 1


def test_simulate_and_pick_threshold():
    fast = [out("billing", "low", False), out("orders", "low", False, priority_conf=0.6),
            out("fraud", "high", True)]
    slow = [None, out("orders", "high", False), None]
    decisions, escalated = simulate_hybrid(fast, slow, 0.7, "field")
    assert escalated.tolist() == [False, True, False]
    assert decisions[1].priority == "high"
    best, table = pick_threshold(fast, slow, TRUTH, [0.0, 0.7], "field")
    assert best == 0.7 and table.loc[0.7, "exact_match"] == 1.0
    with pytest.raises(ValueError):
        simulate_hybrid(fast, [None, None, None], 0.7, "field")
