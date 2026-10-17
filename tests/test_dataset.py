"""Tests for the dataset builder, labelling rules, splits and Laya JSONL format."""

import json

import pandas as pd

from src.dataset import (
    FEATURE_COLUMNS, LABEL_COLUMNS, PRIORITIES, ROUTES, ACCOUNT_TIERS, CHANNELS,
    TypedDecision, bitext_route, build_synthetic_dataset, decide_priority, decide_requires_human,
    label_row, load_splits, split_features_labels, to_laya_example, train_val_test_split, write_laya_jsonl,
)


def test_same_seed_gives_same_data():
    a = build_synthetic_dataset(n_rows=100, seed=1)
    b = build_synthetic_dataset(n_rows=100, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_data():
    a = build_synthetic_dataset(n_rows=100, seed=1)
    b = build_synthetic_dataset(n_rows=100, seed=2)
    assert not a.equals(b)


def test_columns_and_value_ranges(small_df):
    assert list(small_df.columns) == FEATURE_COLUMNS + LABEL_COLUMNS
    assert set(small_df["route_target"]) <= set(ROUTES)
    assert set(small_df["priority_level"]) <= set(PRIORITIES)
    assert set(small_df["account_tier"]) <= set(ACCOUNT_TIERS)
    assert set(small_df["channel"]) <= set(CHANNELS)
    assert small_df["requires_human"].dtype == bool
    assert small_df["urgency_score"].between(0, 1).all()
    assert small_df["past_failures"].between(0, 5).all()
    assert small_df["query"].str.len().gt(0).all()


def test_zero_noise_labels_follow_the_rules():
    df = build_synthetic_dataset(n_rows=200, seed=3, label_noise=0.0)
    for _, row in df.iterrows():
        expected = label_row(row["route_target"], row["account_tier"], row["urgency_score"], row["past_failures"])
        assert row["priority_level"] == expected.priority_level
        assert row["requires_human"] == expected.requires_human


def test_priority_rule_extremes():
    assert decide_priority("general", "free", 0.0, 0) == "low"
    assert decide_priority("fraud", "enterprise", 1.0, 5) == "critical"


def test_requires_human_rule():
    assert decide_requires_human("fraud", "low", "free", 0) is True
    assert decide_requires_human("billing", "low", "free", 3) is True
    assert decide_requires_human("billing", "low", "free", 0) is False
    assert decide_requires_human("orders", "high", "enterprise", 0) is True


def test_typed_decision_to_dict():
    d = TypedDecision("billing", "high", True).to_dict()
    assert d == {"route_target": "billing", "priority_level": "high", "requires_human": True}


def test_split_sizes_and_no_overlap(small_df, small_splits):
    train, val, test = small_splits
    assert len(train) + len(val) + len(test) == len(small_df)
    assert abs(len(test) - 0.15 * len(small_df)) <= 2
    # every route should show up in the test split thanks to stratification
    assert set(test["route_target"]) == set(small_df["route_target"])


def test_split_features_labels(small_df):
    X, y = split_features_labels(small_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert list(y.columns) == LABEL_COLUMNS


def test_load_splits_builds_csvs_when_missing(tmp_path):
    train, val, test = load_splits(str(tmp_path), n_rows=120, seed=5)
    assert (tmp_path / "train.csv").exists()
    assert train["requires_human"].dtype == bool
    # second call reads from disk and gives the same data
    train2, _, _ = load_splits(str(tmp_path), n_rows=120, seed=5)
    pd.testing.assert_frame_equal(train, train2)


def test_bitext_mapping():
    assert bitext_route("ORDER", "track_order") == "orders"
    assert bitext_route("ACCOUNT", "recover_password") == "tech_support"
    assert bitext_route("SOMETHING_NEW", "x") == "general"


def test_laya_example_shape(small_df):
    row = small_df.iloc[0]
    example = to_laya_example(row)
    assert set(example) == {"state", "questions", "expected"}
    assert set(example["questions"]) == set(LABEL_COLUMNS)
    assert example["questions"]["route_target"]["type"] == "choice"
    assert example["questions"]["priority_level"]["type"] == "score"
    assert example["questions"]["requires_human"]["type"] == "noul"
    # score answers are level indexes, not strings
    assert example["expected"]["priority_level"] == PRIORITIES.index(row["priority_level"])
    assert isinstance(example["expected"]["requires_human"], bool)


def test_write_laya_jsonl(small_df, tmp_path):
    path = write_laya_jsonl(small_df.head(5), str(tmp_path / "out" / "train.jsonl"))
    lines = open(path, encoding="utf-8").read().strip().split("\n")
    assert len(lines) == 5
    assert json.loads(lines[0])["state"]["query"] == small_df.iloc[0]["query"]
