"""Offline tests for the Gemini router, the hybrid router and the LangGraph wrapper.

No API key or network: a fake client stands in for google-genai and returns canned answers.
"""

import json
from types import SimpleNamespace

import pytest

from src.dataset import ROUTES
from src.interface import DecisionModel, DecisionOutput
from src.models.hybrid import HybridDecisionModel
from src.models.llm_router import (
    GeminiDecisionModel, build_system_prompt, cost_usd, parse_llm_json, pick_examples,
)
from src.router_graph import build_router_graph, expected_destination, run_tickets

GOOD = json.dumps({"route_target": "fraud", "priority_level": "high", "requires_human": True, "confidence": 0.9})


class FakeAPIError(Exception):
    """Mimics google.genai.errors.APIError, which carries an HTTP status in `.code`."""

    def __init__(self, code, message="error"):
        super().__init__(message)
        self.code = code


class FakeModels:
    def __init__(self, answers):
        self.answers = list(answers)
        self.configs = []

    def generate_content(self, model, contents, config):
        self.configs.append(config)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        usage = SimpleNamespace(prompt_token_count=1000, candidates_token_count=30, thoughts_token_count=10)
        return SimpleNamespace(text=answer, usage_metadata=usage)


def fake_gemini(answers, **kwargs):
    client = SimpleNamespace(models=FakeModels(answers))
    return GeminiDecisionModel(client=client, retry_wait=0.0, **kwargs), client


def test_parse_llm_json():
    out = parse_llm_json(GOOD)
    assert (out.route, out.priority, out.requires_human, out.confidence) == ("fraud", "high", True, 0.9)
    with pytest.raises(ValueError):
        parse_llm_json(json.dumps({"route_target": "sales", "priority_level": "high", "requires_human": True}))


def test_cost_usd():
    assert cost_usd(1_000_000, 0, {"input": 0.25, "output": 1.5}) == pytest.approx(0.25)
    assert cost_usd(1000, 40, {"input": 0.25, "output": 1.5}) == pytest.approx(0.00031)


def test_system_prompt_has_teams_and_examples(small_df):
    examples = pick_examples(small_df, per_route=2)
    assert len(examples) == 2 * len(ROUTES)
    assert set(examples["route_target"]) == set(ROUTES)
    prompt = build_system_prompt(examples)
    for route in ROUTES:
        assert route in prompt
    assert examples.iloc[0]["query"] in prompt


def test_predict_records_tokens_and_cost(small_df):
    model, _ = fake_gemini([GOOD])
    out = model.predict_one(small_df.iloc[0])
    assert out.route == "fraud" and out.latency_ms >= 0
    call = model.calls[0]
    assert call["input_tokens"] == 1000
    assert call["output_tokens"] == 40          # 30 answer + 10 thinking
    assert call["cost_usd"] == pytest.approx(cost_usd(1000, 40, model.prices))
    summary = model.usage_summary()
    assert summary["calls"] == 1 and summary["failed_parses"] == 0


def test_broken_answer_goes_to_human(small_df):
    model, _ = fake_gemini(["not json at all"])
    out = model.predict_one(small_df.iloc[0])
    assert out.requires_human is True and out.confidence == 0.0
    assert model.usage_summary()["failed_parses"] == 1


def test_retries_on_rate_limit(small_df):
    model, client = fake_gemini([FakeAPIError(429), FakeAPIError(503), GOOD])
    assert model.predict_one(small_df.iloc[0]).route == "fraud"
    assert len(client.models.configs) == 3


def test_gives_up_on_other_errors(small_df):
    model, _ = fake_gemini([FakeAPIError(403, "bad key")])
    with pytest.raises(FakeAPIError):
        model.predict_one(small_df.iloc[0])


def test_drops_thinking_setting_if_unsupported(small_df):
    model, client = fake_gemini([FakeAPIError(400, "thinking_level is not supported"), GOOD])
    model.predict_one(small_df.iloc[0])
    assert "thinking_config" in client.models.configs[0]
    assert "thinking_config" not in client.models.configs[1]


def test_rate_limit_pause_is_not_latency(small_df):
    # Regression: the free-tier pause used to be inside the timer, so every call looked ~6 s slow.
    model, _ = fake_gemini([GOOD, GOOD], min_seconds_between_calls=0.3)
    model.predict_one(small_df.iloc[0])
    second = model.predict_one(small_df.iloc[1])
    assert model.last_pause_ms > 200        # it did wait...
    assert second.latency_ms < 100          # ...but the wait isn't counted
    assert model.calls[1]["latency_ms"] < 100


def test_graph_leaves_out_rate_limit_pause(small_df):
    model, _ = fake_gemini([GOOD, GOOD, GOOD], min_seconds_between_calls=0.3)
    results = run_tickets(build_router_graph(model), small_df.head(3))
    assert (results["graph_ms"] < 150).all()


def test_unknown_model_needs_prices():
    with pytest.raises(ValueError):
        GeminiDecisionModel(model="gemini-made-up", client=object())
    GeminiDecisionModel(model="gemini-made-up", prices={"input": 1, "output": 2}, client=object())


class FixedModel(DecisionModel):
    """Returns the same decision every time, with a chosen weakest-field confidence."""

    def __init__(self, name, route, min_conf):
        self.name, self.route, self.min_conf = name, route, min_conf

    def predict_batch(self, df):
        return [DecisionOutput(self.route, "low", False, confidence=0.99,
                               field_confidences={"route_target": 0.99, "priority_level": self.min_conf,
                                                  "requires_human": 0.99})
                for _ in range(len(df))]


def test_hybrid_escalates_only_unsure_tickets(small_df):
    sure = HybridDecisionModel(FixedModel("fast", "billing", 0.9), FixedModel("slow", "fraud", 0.9), threshold=0.7)
    assert sure.predict_one(small_df.iloc[0]).route == "billing"
    assert sure.escalation_rate() == 0.0

    unsure = HybridDecisionModel(FixedModel("fast", "billing", 0.5), FixedModel("slow", "fraud", 0.9), threshold=0.7)
    assert unsure.predict_one(small_df.iloc[0]).route == "fraud"
    assert unsure.escalation_rate() == 1.0


def test_expected_destination():
    assert expected_destination({"requires_human": True, "route_target": "billing"}) == "human_handoff"
    assert expected_destination({"requires_human": False, "route_target": "billing"}) == "billing_agent"


def test_graph_runs_and_scores(small_df):
    graph = build_router_graph(FixedModel("fixed", "billing", 0.9))
    results = run_tickets(graph, small_df.head(20))
    assert len(results) == 20
    assert set(results["handled_by"]) == {"billing_agent"}
    # the fixed model is right exactly when the ticket should go to billing_agent
    expected_share = (results["expected"] == "billing_agent").mean()
    assert results["correct_destination"].mean() == pytest.approx(expected_share)
    assert (results["graph_ms"] >= results["router_ms"]).all()
