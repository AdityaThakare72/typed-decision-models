"""Random Forest and XGBoost baselines.

Both use the same feature pipeline:
  query text            -> TF-IDF (word + bigram counts, down-weighting very common words)
  account_tier, channel -> one-hot columns
  urgency, past_failures -> standardised numbers (mean 0, std 1)

We train one classifier per output field (route, priority, requires_human), because
classic tree models predict a single label at a time.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.dataset import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, LABEL_COLUMNS, NUMERIC_COLUMNS, TEXT_COLUMN
from src.interface import DecisionModel, DecisionOutput


def build_preprocessor(max_tfidf_features: int = 5000) -> ColumnTransformer:
    """Text + tabular features -> one big numeric matrix that trees can split on."""
    return ColumnTransformer([
        # A single column name (not a list) because TfidfVectorizer wants 1-D text input.
        ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=max_tfidf_features, sublinear_tf=True),
         TEXT_COLUMN),
        # handle_unknown="ignore": a tier we never saw in training becomes all-zeros, not a crash.
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ("num", StandardScaler(), NUMERIC_COLUMNS),
    ])


def make_classifier(kind: str, seed: int = 42):
    """Fresh, untrained classifier of the requested kind."""
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    if kind == "xgb":
        from xgboost import XGBClassifier  # imported here so the RF path works without xgboost
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             tree_method="hist", n_jobs=-1, random_state=seed)
    raise ValueError(f"unknown kind {kind!r}, use 'rf' or 'xgb'")


class ClassicalDecisionModel(DecisionModel):
    """RF or XGBoost wrapped in the common DecisionModel interface."""

    def __init__(self, kind: str = "rf", seed: int = 42):
        self.kind = kind
        self.seed = seed
        self.name = {"rf": "random_forest", "xgb": "xgboost"}.get(kind, kind)
        self.pipelines: dict[str, Pipeline] = {}
        # XGBoost only accepts integer class ids, so we keep an encoder per label to map back.
        self.label_encoders: dict[str, LabelEncoder] = {}

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "ClassicalDecisionModel":
        X = train_df[FEATURE_COLUMNS]
        for column in LABEL_COLUMNS:
            encoder = LabelEncoder()
            y = encoder.fit_transform(train_df[column].astype(str))
            pipeline = Pipeline([
                ("features", build_preprocessor()),
                ("clf", make_classifier(self.kind, self.seed)),
            ])
            pipeline.fit(X, y)
            # Train on all cores, but predict on one: spinning up a thread pool for a single
            # ticket costs far more than the prediction itself (RF went ~390ms -> a few ms).
            pipeline.named_steps["clf"].set_params(n_jobs=1)
            self.pipelines[column] = pipeline
            self.label_encoders[column] = encoder
        return self

    def predict_batch(self, df: pd.DataFrame) -> list[DecisionOutput]:
        if not self.pipelines:
            raise RuntimeError("call fit() before predicting")
        X = df[FEATURE_COLUMNS]
        decoded = {}
        route_confidence = None
        for column in LABEL_COLUMNS:
            probs = self.pipelines[column].predict_proba(X)
            best = probs.argmax(axis=1)
            decoded[column] = self.label_encoders[column].inverse_transform(best)
            if column == "route_target":
                route_confidence = probs.max(axis=1)

        outputs = []
        for i in range(len(df)):
            outputs.append(DecisionOutput(
                route=str(decoded["route_target"][i]),
                priority=str(decoded["priority_level"][i]),
                # the encoder stored "True"/"False" strings, turn them back into a bool
                requires_human=str(decoded["requires_human"][i]) == "True",
                confidence=float(route_confidence[i]),
            ))
        return outputs

    def size_mb(self) -> float:
        """Size of the pickled pipelines, a fair proxy for the deployed artifact."""
        if not self.pipelines:
            return 0.0
        blob = pickle.dumps((self.pipelines, self.label_encoders))
        return len(blob) / (1024 * 1024)


def feature_names(model: ClassicalDecisionModel, column: str = "route_target") -> np.ndarray:
    """Names of the features after preprocessing (handy for feature-importance plots)."""
    return model.pipelines[column].named_steps["features"].get_feature_names_out()
