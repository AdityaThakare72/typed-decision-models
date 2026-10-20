"""LLM router: ask Gemini for the typed decision instead of using a trained model.

This is the "just use an LLM for everything" baseline. Gemini gets:
  - the team descriptions and the routing policy written in plain words
  - a few labelled example tickets (few-shot)
  - the ticket itself as JSON
and must answer with JSON matching a schema (Gemini's structured output), so we don't have
to parse free text.

Every call records its latency and token counts, so we can put a dollar figure next to
the trained models. `google-genai` is imported lazily: the rest of the repo (and the tests)
work without it.
"""

import json
import os
import time

import pandas as pd

from src.dataset import LABEL_COLUMNS, PRIORITIES, ROUTE_DESCRIPTIONS, ROUTES, row_to_state
from src.interface import DecisionModel, DecisionOutput

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

# USD per 1M tokens, standard paid tier, text in / text out (thinking tokens bill as output).
# Source: https://ai.google.dev/gemini-api/docs/pricing (checked 2026-10-19). Prices change,
# so check the page and edit this dict (or pass your own prices) before quoting numbers.
GEMINI_PRICES_PER_1M = {
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},   # introductory price until 2026-12-31
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}

# The JSON shape Gemini must return. Enums stop it from inventing new team names.
DECISION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "route_target": {"type": "string", "enum": ROUTES},
        "priority_level": {"type": "string", "enum": PRIORITIES},
        "requires_human": {"type": "boolean"},
        "confidence": {"type": "number", "description": "how sure you are about the whole decision, 0 to 1"},
    },
    "required": ["route_target", "priority_level", "requires_human", "confidence"],
}

# The routing policy, written the way an ops team would write it for a human agent.
# It describes the same signals the labels are built from, but in words, not exact thresholds.
POLICY = """You are the router for a customer-support system. For each ticket decide:

1. route_target: which team handles it.
{teams}

2. priority_level: low, medium, high or critical. Weigh these together:
   - urgency_score (0 to 1): higher means more urgent
   - account_tier: enterprise counts most, then pro, then free
   - past_failures: each earlier failed attempt makes it more urgent
   - fraud tickets are more urgent, tech_support tickets slightly more urgent
   Most tickets are medium. Critical is rare: only the most urgent combinations.

3. requires_human: true if ANY of these holds:
   - the ticket is fraud
   - past_failures is 3 or more
   - priority is critical
   - the account is enterprise and priority is high
   otherwise false.

Answer only with the JSON object."""


def build_system_prompt(examples: pd.DataFrame | None = None) -> str:
    """Policy text + optional labelled examples. Same for every call, so it can be cached."""
    teams = "\n".join(f"   - {route}: {desc}" for route, desc in ROUTE_DESCRIPTIONS.items())
    prompt = POLICY.format(teams=teams)
    if examples is not None and len(examples):
        lines = ["\n\nLabelled examples:"]
        for _, row in examples.iterrows():
            answer = {column: row[column] for column in LABEL_COLUMNS}
            answer["requires_human"] = bool(answer["requires_human"])
            lines.append(f"ticket: {json.dumps(row_to_state(row))}\nanswer: {json.dumps(answer)}")
        prompt += "\n\n".join(lines)
    return prompt


def pick_examples(train_df: pd.DataFrame, per_route: int = 2, seed: int = 42) -> pd.DataFrame:
    """A small, balanced few-shot set: `per_route` random tickets from each route."""
    parts = []
    for _, group in train_df.groupby("route_target"):
        parts.append(group.sample(n=min(per_route, len(group)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def parse_llm_json(text: str) -> DecisionOutput:
    """Turn Gemini's JSON text into a DecisionOutput. Raises ValueError if it doesn't fit the schema."""
    data = json.loads(text)
    route, priority = data.get("route_target"), data.get("priority_level")
    if route not in ROUTES or priority not in PRIORITIES or not isinstance(data.get("requires_human"), bool):
        raise ValueError(f"answer does not match the schema: {data}")
    confidence = float(data.get("confidence", 0.0))
    return DecisionOutput(route=route, priority=priority, requires_human=data["requires_human"],
                          confidence=min(max(confidence, 0.0), 1.0))


def cost_usd(input_tokens: int, output_tokens: int, prices: dict) -> float:
    """Dollar cost of one call from token counts and per-1M prices."""
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


class GeminiDecisionModel(DecisionModel):
    """Gemini behind the common DecisionModel interface, with per-call usage tracking."""

    def __init__(self, model: str = DEFAULT_GEMINI_MODEL, api_key: str | None = None,
                 examples: pd.DataFrame | None = None, thinking_level: str | None = "minimal",
                 prices: dict | None = None, min_seconds_between_calls: float = 0.0,
                 max_retries: int = 5, retry_wait: float = 2.0, client=None):
        self.model = model
        self.name = "llm:" + model
        self.system_prompt = build_system_prompt(examples)
        self.thinking_level = thinking_level
        self.prices = prices or GEMINI_PRICES_PER_1M.get(model)
        if self.prices is None:
            raise ValueError(f"no price known for {model!r}; pass prices={{'input': ..., 'output': ...}}")
        # Free-tier keys allow only a few requests per minute; a small pause avoids 429 errors.
        # The pause is NOT counted in the measured latency.
        self.min_seconds_between_calls = min_seconds_between_calls
        self.max_retries = max_retries
        self.retry_wait = retry_wait  # first back-off pause in seconds, doubles each retry
        self.api_key = api_key
        self.client = client          # tests pass a fake client; real one is made on first use
        self.calls: list[dict] = []   # one record per API call: latency, tokens, cost, ok
        self._last_call = 0.0
        self.last_pause_ms = 0.0      # rate-limit pause before the latest call (not latency)

    def _get_client(self):
        if self.client is None:
            from google import genai
            self.client = genai.Client(api_key=self.api_key or os.environ.get("GEMINI_API_KEY"))
        return self.client

    def _config(self) -> dict:
        config = {
            "system_instruction": self.system_prompt,
            "response_mime_type": "application/json",
            "response_json_schema": DECISION_JSON_SCHEMA,
            "temperature": 0.0,   # same ticket -> same answer, as far as the API allows
        }
        if self.thinking_level:
            # Thinking tokens cost money and time; a router decision doesn't need much reasoning.
            config["thinking_config"] = {"thinking_level": self.thinking_level}
        return config

    def _wait_for_rate_limit(self) -> float:
        """Sleep until `min_seconds_between_calls` has passed since the last call; returns ms slept."""
        pause = self.min_seconds_between_calls - (time.monotonic() - self._last_call)
        if pause <= 0:
            return 0.0
        time.sleep(pause)
        return pause * 1000.0

    def _call(self, ticket_json: str):
        """One API call with retry on rate limits (429) and server errors (5xx)."""
        wait = self.retry_wait
        for attempt in range(self.max_retries + 1):
            self._last_call = time.monotonic()
            try:
                return self._get_client().models.generate_content(
                    model=self.model, contents=ticket_json, config=self._config())
            except Exception as error:  # the SDK raises google.genai.errors.APIError subclasses
                code = getattr(error, "code", None)
                if code == 400 and self.thinking_level and "think" in str(error).lower():
                    # This model doesn't accept the thinking setting: drop it and try again.
                    self.thinking_level = None
                    continue
                if code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(wait)
                    wait *= 2
                    continue
                raise

    def predict_row(self, row) -> DecisionOutput:
        ticket_json = json.dumps(row_to_state(row))
        # The free-tier pause happens BEFORE the timer starts, so it never counts as latency.
        # The graph reads `last_pause_ms` to take it out of the end-to-end time too.
        self.last_pause_ms = self._wait_for_rate_limit()
        start = time.perf_counter()
        response = self._call(ticket_json)
        latency_ms = (time.perf_counter() - start) * 1000.0

        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count or 0
        # Thinking tokens are billed as output tokens.
        output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        record = {"latency_ms": latency_ms, "input_tokens": input_tokens, "output_tokens": output_tokens,
                  "cost_usd": cost_usd(input_tokens, output_tokens, self.prices), "ok": True}
        try:
            output = parse_llm_json(response.text)
        except (ValueError, TypeError) as error:
            # A broken answer still costs money. Send the ticket to a human, the safe default.
            record["ok"] = False
            record["error"] = str(error)[:200]
            output = DecisionOutput(route="general", priority="medium", requires_human=True, confidence=0.0)
        self.calls.append(record)
        output.latency_ms = latency_ms
        return output

    def predict_batch(self, df: pd.DataFrame) -> list[DecisionOutput]:
        # A router handles one request at a time, so we call the API once per ticket.
        return [self.predict_row(row) for _, row in df.iterrows()]

    def predict_one(self, row) -> DecisionOutput:
        return self.predict_row(dict(row))

    def usage_summary(self) -> dict:
        """Totals and averages over every call made so far."""
        calls = pd.DataFrame(self.calls)
        if calls.empty:
            return {"calls": 0}
        return {
            "calls": len(calls),
            "failed_parses": int((~calls["ok"]).sum()),
            "avg_input_tokens": float(calls["input_tokens"].mean()),
            "avg_output_tokens": float(calls["output_tokens"].mean()),
            "total_cost_usd": float(calls["cost_usd"].sum()),
            "cost_per_1k_tickets_usd": float(calls["cost_usd"].mean() * 1000),
        }
