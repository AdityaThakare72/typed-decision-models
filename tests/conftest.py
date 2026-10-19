"""Shared fixtures: a small deterministic dataset reused across test files."""

import pytest

from src.dataset import build_synthetic_dataset, train_val_test_split


@pytest.fixture(scope="session")
def small_df():
    # 300 rows is enough to exercise every route/priority but keeps tests fast.
    return build_synthetic_dataset(n_rows=300, seed=7)


@pytest.fixture(scope="session")
def small_splits(small_df):
    return train_val_test_split(small_df, seed=7)
