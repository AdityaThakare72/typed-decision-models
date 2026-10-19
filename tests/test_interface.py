"""Tests for the DecisionModel interface, metrics and latency measurement, using a dummy model."""

import time

import pandas as pd
import pytest

from src.dataset import LABEL_COLUMNS
from src.interface import (
    DecisionModel, DecisionOutput, compute_metrics, measure_latency, outputs_to_frame, run_benchmark,
)


class DummyModel(DecisionModel):
    """Always says billing / medium / no human. Optionally sleeps to fake slowness."""

    name = "dummy"

    def __init__(self, sleep_s: float = 0.0):
        self.sleep_s = sleep_s

    def predict_batch(self, df):
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return [DecisionOutput("billing", "medium", False, confidence=0.5) for _ in range(len(df))]


class OracleModel(DecisionModel):
    """Cheats by reading the labels - should score a perfect 1.0 everywhere."""

    name = "oracle"

    def __init__(self, truth: pd.DataFrame):
        self.truth = truth.reset_index(drop=True)
        self.calls = 0

    def predict_batch(self, df):
        out = []
        for _, row in df.iterrows():
            match = self.truth[self.truth["query"] == row["query"]].iloc[0]
            out.append(DecisionOutput(match["route_target"], match["priority_level"],
                                      bool(match["requires_human"]), confidence=1.0))
        return out


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        DecisionModel()


def test_predict_one_sets_latency():
    out = DummyModel(sleep_s=0.002).predict_one({"query": "hi"})
    assert out.route == "billing"
    assert out.latency_ms >= 2.0


def test_outputs_to_frame():
    frame = outputs_to_frame([DecisionOutput("fraud", "high", True, 0.9)])
    assert list(frame.columns) == LABEL_COLUMNS
    assert frame.iloc[0].to_dict() == {"route_target": "fraud", "priority_level": "high", "requires_human": True}


def test_metrics_on_hand_made_example():
    y_true = pd.DataFrame({
        "route_target": ["billing", "billing", "fraud", "orders"],
        "priority_level": ["low", "medium", "high", "low"],
        "requires_human": [False, False, True, False],
    })
    y_pred = pd.DataFrame({
        "route_target": ["billing", "fraud", "fraud", "orders"],   # 3/4 right
        "priority_level": ["low", "medium", "high", "low"],        # 4/4 right
        "requires_human": [False, False, True, True],              # 3/4 right
    })
    m = compute_metrics(y_true, y_pred)
    assert m["route_acc"] == pytest.approx(0.75)
    assert m["priority_acc"] == pytest.approx(1.0)
    assert m["priority_f1"] == pytest.approx(1.0)
    assert m["requires_acc"] == pytest.approx(0.75)
    # rows 0 and 2 are fully right -> 2/4
    assert m["exact_match"] == pytest.approx(0.5)
    # route macro-F1 by hand: billing P=1 R=.5 -> .667, fraud P=.5 R=1 -> .667, orders 1 -> mean .778
    assert m["route_f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_dummy_model_scores_match_class_share(small_df):
    outputs = DummyModel().predict_batch(small_df)
    m = compute_metrics(small_df[LABEL_COLUMNS], outputs_to_frame(outputs))
    assert m["route_acc"] == pytest.approx((small_df["route_target"] == "billing").mean())


def test_oracle_is_perfect(small_df):
    test = small_df.drop_duplicates("query").head(30)
    result = run_benchmark([OracleModel(test)], test, n_latency_runs=5)
    assert result.loc["oracle", "exact_match"] == 1.0
    assert result.loc["oracle", "route_f1"] == 1.0


def test_latency_percentiles(small_df):
    lat = measure_latency(DummyModel(sleep_s=0.001), small_df.head(10), n_runs=20, warmup=2)
    assert 0 < lat["p50_ms"] <= lat["p95_ms"]
    assert lat["p50_ms"] >= 1.0


def test_latency_needs_rows():
    with pytest.raises(ValueError):
        measure_latency(DummyModel(), pd.DataFrame(), n_runs=1)


def test_run_benchmark_table_columns(small_df):
    table = run_benchmark([DummyModel()], small_df.head(20), n_latency_runs=5)
    expected = {"route_acc", "route_f1", "priority_acc", "priority_f1", "requires_acc", "requires_f1",
                "exact_match", "p50_ms", "p95_ms", "size_mb"}
    assert expected <= set(table.columns)
    assert list(table.index) == ["dummy"]
