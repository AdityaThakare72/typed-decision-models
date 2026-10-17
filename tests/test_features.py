"""Tests for the TF-IDF + one-hot + scaler feature pipeline."""

import numpy as np
import pandas as pd

from src.dataset import FEATURE_COLUMNS
from src.models.classical_ml import build_preprocessor


def test_preprocessor_output_shape(small_df):
    pre = build_preprocessor()
    X = pre.fit_transform(small_df[FEATURE_COLUMNS])
    assert X.shape[0] == len(small_df)

    vocab_size = len(pre.named_transformers_["text"].vocabulary_)
    n_onehot = sum(len(c) for c in pre.named_transformers_["cat"].categories_)
    n_numeric = 2
    assert X.shape[1] == vocab_size + n_onehot + n_numeric


def test_numeric_columns_are_standardised(small_df):
    pre = build_preprocessor()
    X = pre.fit_transform(small_df[FEATURE_COLUMNS])
    X = X.toarray() if hasattr(X, "toarray") else X
    numeric = X[:, -2:]
    assert np.allclose(numeric.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(numeric.std(axis=0), 1, atol=1e-6)


def test_unseen_category_does_not_crash(small_df):
    pre = build_preprocessor()
    pre.fit(small_df[FEATURE_COLUMNS])
    weird = pd.DataFrame([{
        "query": "totally new words here", "account_tier": "platinum", "channel": "pigeon",
        "urgency_score": 0.5, "past_failures": 1,
    }])
    out = pre.transform(weird)
    assert out.shape[0] == 1


def test_max_tfidf_features_is_respected(small_df):
    pre = build_preprocessor(max_tfidf_features=10)
    pre.fit(small_df[FEATURE_COLUMNS])
    assert len(pre.named_transformers_["text"].vocabulary_) <= 10
