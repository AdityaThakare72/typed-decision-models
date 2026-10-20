"""Model tests that run offline: real RF/XGB fits on tiny data, and the Laya/BERT
output-decoding logic (which doesn't need torch or the actual weights)."""

import numpy as np
import pytest

from src.dataset import PRIORITIES, ROUTES
from src.interface import DecisionOutput, evaluate_model
from src.models.bert_baseline import NUM_LOGITS, encode_labels, logits_to_outputs, row_to_text
from src.models.classical_ml import ClassicalDecisionModel, make_classifier
from src.models.laya_model import LayaDecisionModel, parse_laya_answers


@pytest.mark.parametrize("kind", ["rf", "xgb"])
def test_classical_fit_predict(kind, small_splits):
    train, _, test = small_splits
    model = ClassicalDecisionModel(kind).fit(train)
    outputs = model.predict_batch(test)
    assert len(outputs) == len(test)
    assert all(isinstance(o, DecisionOutput) for o in outputs)
    assert {o.route for o in outputs} <= set(ROUTES)
    assert {o.priority for o in outputs} <= set(PRIORITIES)
    assert all(isinstance(o.requires_human, bool) for o in outputs)
    assert all(0.0 <= o.confidence <= 1.0 for o in outputs)
    assert all(set(o.field_confidences) == {"route_target", "priority_level", "requires_human"} for o in outputs)
    assert all(o.min_confidence() <= o.confidence for o in outputs)
    assert model.size_mb() > 0


def test_classical_beats_chance(small_splits):
    train, _, test = small_splits
    result = evaluate_model(ClassicalDecisionModel("xgb").fit(train), test, n_latency_runs=3)
    # 6 routes -> chance is ~0.17. Templates are easy to tell apart, so expect far better.
    assert result["route_acc"] > 0.6


def test_classical_predict_before_fit():
    with pytest.raises(RuntimeError):
        ClassicalDecisionModel("rf").predict_batch(None)


def test_unknown_kind():
    with pytest.raises(ValueError):
        make_classifier("svm")


def test_parse_laya_answers():
    fake_answers = {
        "route_target": {"type": "choice", "choice": "fraud", "answer_confidence": 0.81},
        "priority_level": {"type": "score", "score": 2.1,
                           "probabilities": {"0": 0.05, "1": 0.15, "2": 0.6, "3": 0.2}},
        "requires_human": {"type": "noul", "noul": 0.73},
    }
    out = parse_laya_answers(fake_answers)
    assert (out.route, out.priority, out.requires_human, out.confidence) == ("fraud", "high", True, 0.81)
    assert out.field_confidences == {"route_target": 0.81, "priority_level": 0.6, "requires_human": 0.73}
    assert out.min_confidence() == 0.6
    # stricter threshold flips the yes/no answer
    assert parse_laya_answers(fake_answers, human_threshold=0.9).requires_human is False


def test_laya_model_does_not_load_on_init():
    model = LayaDecisionModel()
    assert model.agent is None
    assert model.name == "laya:laya"
    assert model.size_mb() == 0.0


def test_bert_text_includes_tabular(small_df):
    text = row_to_text(small_df.iloc[0])
    assert small_df.iloc[0]["account_tier"] in text
    assert small_df.iloc[0]["query"] in text


def test_bert_label_encoding(small_df):
    labels = encode_labels(small_df.head(10))
    assert labels.shape == (10, 3)
    assert labels[:, 0].max() < len(ROUTES)


def test_bert_logits_decoding():
    logits = np.zeros((1, NUM_LOGITS))
    logits[0, ROUTES.index("orders")] = 5.0                       # route block
    logits[0, len(ROUTES) + PRIORITIES.index("critical")] = 5.0   # priority block
    logits[0, NUM_LOGITS - 1] = 5.0                               # human block: True
    out = logits_to_outputs(logits)[0]
    assert (out.route, out.priority, out.requires_human) == ("orders", "critical", True)
    assert 0.9 < out.confidence <= 1.0
