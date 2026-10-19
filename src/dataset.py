"""Builds and loads the hybrid "decision + agent routing" dataset.

Every row is one support ticket:
  - text:    `query` (what the customer wrote)
  - tabular: `account_tier`, `channel`, `urgency_score`, `past_failures`
  - labels:  a TypedDecision -> `route_target`, `priority_level`, `requires_human`

Two ways to get data:
  1. `build_synthetic_dataset()` - fully offline and deterministic (same seed -> same rows).
     Used by the tests and as the default everywhere.
  2. `load_hf_dataset()` - real customer-support queries from Hugging Face (Bitext), with the
     same tabular features and labelling rules bolted on. Needs internet, meant for Colab.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# The schema: what a decision looks like
# ---------------------------------------------------------------------------

ROUTES = ["billing", "orders", "account", "tech_support", "fraud", "general"]
PRIORITIES = ["low", "medium", "high", "critical"]  # ordered: index = how urgent
ACCOUNT_TIERS = ["free", "pro", "enterprise"]
CHANNELS = ["email", "chat", "phone"]

TEXT_COLUMN = "query"
CATEGORICAL_COLUMNS = ["account_tier", "channel"]
NUMERIC_COLUMNS = ["urgency_score", "past_failures"]
FEATURE_COLUMNS = [TEXT_COLUMN] + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
LABEL_COLUMNS = ["route_target", "priority_level", "requires_human"]

# Short descriptions of each route. Laya reads these as the "criteria" for its choice question,
# so they double as documentation of what each team handles.
ROUTE_DESCRIPTIONS = {
    "billing": "payments, charges, refunds, invoices, subscriptions",
    "orders": "placing, changing, tracking or cancelling orders and deliveries",
    "account": "login details, profile changes, creating or deleting an account",
    "tech_support": "bugs, errors, crashes, outages, the app or site not working",
    "fraud": "unauthorised charges, hacked accounts, scams, suspicious activity",
    "general": "feedback, general questions, anything else",
}


@dataclass
class TypedDecision:
    """The structured answer every model must produce for a ticket.

    A dataclass is just a class that auto-writes __init__ / __repr__ / __eq__ for us,
    so a decision is easy to create, print and compare.
    """

    route_target: str
    priority_level: str
    requires_human: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Text templates for the synthetic builder
# ---------------------------------------------------------------------------

# A few phrasings per route. {x} slots get filled from SLOT_VALUES so we get variety
# without needing an LLM to write the data.
TEMPLATES = {
    "billing": [
        "I was charged twice for my {plan} plan this month",
        "can I get a refund for my last {plan} payment",
        "my invoice shows the wrong amount, please fix it",
        "why did my card get declined when paying for {plan}",
        "how do I cancel my {plan} subscription and stop billing",
        "need a copy of the invoice for order {num}",
    ],
    "orders": [
        "where is my order {num}, it has not arrived",
        "I want to change the delivery address for order {num}",
        "please cancel order {num} before it ships",
        "the package for order {num} came damaged",
        "how long does shipping take to {city}",
        "tracking for order {num} has not updated in days",
    ],
    "account": [
        "I forgot my password and the reset email never comes",
        "how do I change the email on my account",
        "please delete my account and all my data",
        "I can't log in after changing my phone number",
        "how do I add a teammate to my {plan} account",
        "I want to switch my account to a different email",
    ],
    "tech_support": [
        "the app keeps crashing when I open {feature}",
        "{feature} is showing error 500 since this morning",
        "your site is down and I can't access {feature}",
        "the export to {feature} feature is broken",
        "sync between devices stopped working",
        "the dashboard is very slow and times out",
    ],
    "fraud": [
        "there is a charge I did not make on my account",
        "someone logged into my account from {city}, I think I was hacked",
        "I got an email asking for my password, is it a scam",
        "my card was used for {num} orders I never placed",
        "suspicious login alerts keep coming in",
        "someone changed my account email without permission",
    ],
    "general": [
        "do you have a student discount",
        "just wanted to say the new {feature} is great",
        "what are your support hours",
        "can I suggest a feature for {feature}",
        "is there a way to talk to someone about partnerships",
        "how do I leave a review",
    ],
}

SLOT_VALUES = {
    "plan": ["basic", "pro", "team", "annual", "monthly"],
    "num": ["1042", "88213", "5531", "77", "40920", "3"],
    "city": ["Pune", "Berlin", "Austin", "Lagos", "Tokyo"],
    "feature": ["reports", "the editor", "CSV export", "the mobile app", "notifications"],
}

# Optional bits glued onto the end of a query. Some push urgency up, some are neutral.
URGENT_SUFFIXES = [" asap", " this is urgent", " please help now", " we are losing money", " third time asking"]
CALM_SUFFIXES = ["", "", " thanks", " when you get a chance", " no rush"]


def _fill_template(template: str, rng: np.random.Generator) -> str:
    """Replace every {slot} in the template with a random value for that slot."""
    text = template
    for slot, values in SLOT_VALUES.items():
        placeholder = "{" + slot + "}"
        if placeholder in text:
            text = text.replace(placeholder, str(rng.choice(values)))
    return text


# ---------------------------------------------------------------------------
# Labelling rules
# ---------------------------------------------------------------------------

def decide_priority(route: str, account_tier: str, urgency_score: float, past_failures: int) -> str:
    """Turn the ticket's signals into a priority level with a simple points system.

    Kept deliberately readable: the point is that the label depends on BOTH text (route)
    and tabular columns, so a text-only or table-only model can't get it perfect.
    """
    points = urgency_score * 2.0                     # 0 .. 2
    points += {"free": 0.0, "pro": 0.5, "enterprise": 1.0}[account_tier]
    points += min(past_failures, 5) * 0.3            # 0 .. 1.5
    if route == "fraud":
        points += 1.0
    if route == "tech_support":
        points += 0.4

    if points < 1.2:
        return "low"
    if points < 2.2:
        return "medium"
    if points < 3.2:
        return "high"
    return "critical"


def decide_requires_human(route: str, priority: str, account_tier: str, past_failures: int) -> bool:
    """A human takes over for risky or repeatedly-failed tickets."""
    if route == "fraud":
        return True
    if past_failures >= 3:
        return True
    if priority == "critical":
        return True
    if account_tier == "enterprise" and priority == "high":
        return True
    return False


def label_row(route: str, account_tier: str, urgency_score: float, past_failures: int) -> TypedDecision:
    """Apply both rules to get the full TypedDecision for one ticket."""
    priority = decide_priority(route, account_tier, urgency_score, past_failures)
    human = decide_requires_human(route, priority, account_tier, past_failures)
    return TypedDecision(route_target=route, priority_level=priority, requires_human=human)


def _random_metadata(rng: np.random.Generator, is_urgent_text: bool) -> dict:
    """Random tabular columns. Urgent-sounding text gets a higher urgency score on average,
    like a real upstream sentiment/urgency scorer would give."""
    if is_urgent_text:
        urgency = rng.beta(5, 2)   # skewed high
    else:
        urgency = rng.beta(2, 4)   # skewed low
    return {
        "account_tier": str(rng.choice(ACCOUNT_TIERS, p=[0.5, 0.35, 0.15])),
        "channel": str(rng.choice(CHANNELS, p=[0.4, 0.45, 0.15])),
        "urgency_score": round(float(urgency), 3),
        "past_failures": int(min(rng.poisson(0.8), 5)),
    }


def _add_label_noise(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Flip a small fraction of labels at random.

    Real support data is messy (agents disagree, tickets get mis-tagged). Without noise every
    model would hit ~100% and the benchmark would tell us nothing.
    """
    df = df.copy()
    n = len(df)
    for column, choices in [("route_target", ROUTES), ("priority_level", PRIORITIES)]:
        flip = rng.random(n) < rate
        df.loc[flip, column] = rng.choice(choices, size=int(flip.sum()))
    flip = rng.random(n) < rate
    df.loc[flip, "requires_human"] = ~df.loc[flip, "requires_human"].astype(bool)
    df["requires_human"] = df["requires_human"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# Builder #1: offline synthetic data
# ---------------------------------------------------------------------------

def build_synthetic_dataset(n_rows: int = 3000, seed: int = 42, label_noise: float = 0.05) -> pd.DataFrame:
    """Create `n_rows` labelled tickets. Same arguments -> exactly the same DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        route = str(rng.choice(ROUTES))
        template = str(rng.choice(TEMPLATES[route]))
        query = _fill_template(template, rng)

        is_urgent_text = rng.random() < 0.3
        if is_urgent_text:
            query += str(rng.choice(URGENT_SUFFIXES))
        else:
            query += str(rng.choice(CALM_SUFFIXES))

        meta = _random_metadata(rng, is_urgent_text)
        decision = label_row(route, meta["account_tier"], meta["urgency_score"], meta["past_failures"])
        rows.append({"query": query, **meta, **decision.to_dict()})

    df = pd.DataFrame(rows)
    return _add_label_noise(df, label_noise, rng)


# ---------------------------------------------------------------------------
# Builder #2: real queries from Hugging Face (Colab)
# ---------------------------------------------------------------------------

HF_DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# Bitext is an e-commerce dataset, so it has no "fraud" tickets. Specific intents are
# mapped first, then the broad category as a fallback.
BITEXT_INTENT_TO_ROUTE = {
    "recover_password": "tech_support",
    "registration_problems": "tech_support",
    "payment_issue": "billing",
    "complaint": "general",
}
BITEXT_CATEGORY_TO_ROUTE = {
    "PAYMENT": "billing", "INVOICE": "billing", "REFUND": "billing",
    "CANCEL": "billing", "SUBSCRIPTION": "billing",
    "ORDER": "orders", "DELIVERY": "orders", "SHIPPING": "orders",
    "ACCOUNT": "account",
    "CONTACT": "general", "FEEDBACK": "general",
}


def bitext_route(category: str, intent: str) -> str:
    """Map a Bitext (category, intent) pair onto one of our ROUTES."""
    if intent in BITEXT_INTENT_TO_ROUTE:
        return BITEXT_INTENT_TO_ROUTE[intent]
    return BITEXT_CATEGORY_TO_ROUTE.get(category, "general")


def load_hf_dataset(n_rows: int = 5000, seed: int = 42, label_noise: float = 0.05) -> pd.DataFrame:
    """Real customer queries + synthetic metadata + the same labelling rules.

    `datasets` is imported here (not at the top) so the rest of this file works offline.
    """
    from datasets import load_dataset

    raw = load_dataset(HF_DATASET_ID, split="train").to_pandas()
    raw = raw.sample(n=min(n_rows, len(raw)), random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)

    rows = []
    for _, r in raw.iterrows():
        route = bitext_route(r["category"], r["intent"])
        # Bitext keeps placeholders like {{Order Number}}; swap them for something readable.
        query = str(r["instruction"]).replace("{{", "").replace("}}", "")
        # Bitext flags: "W" = offensive/angry, "Q" = colloquial. Treat angry text as urgent.
        is_urgent_text = "W" in str(r["flags"]) or rng.random() < 0.2
        meta = _random_metadata(rng, is_urgent_text)
        decision = label_row(route, meta["account_tier"], meta["urgency_score"], meta["past_failures"])
        rows.append({"query": query, **meta, **decision.to_dict()})

    return _add_label_noise(pd.DataFrame(rows), label_noise, rng)


# ---------------------------------------------------------------------------
# Splitting, saving, loading
# ---------------------------------------------------------------------------

def train_val_test_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15,
                         seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split on route_target so every split has every route in the same mix."""
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(df, test_size=test_frac, random_state=seed,
                                       stratify=df["route_target"])
    # val_frac is a fraction of the whole dataset, so rescale it to the remaining part.
    val_share = val_frac / (1.0 - test_frac)
    train, val = train_test_split(train_val, test_size=val_share, random_state=seed,
                                  stratify=train_val["route_target"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, data_dir: str = "data") -> None:
    """Write train/val/test CSVs into data_dir."""
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)


def load_splits(data_dir: str = "data", n_rows: int = 3000, seed: int = 42):
    """Load the CSVs if they exist, otherwise build + save the synthetic dataset first."""
    folder = Path(data_dir)
    paths = [folder / "train.csv", folder / "val.csv", folder / "test.csv"]
    if not all(p.exists() for p in paths):
        train, val, test = train_val_test_split(build_synthetic_dataset(n_rows, seed), seed=seed)
        save_splits(train, val, test, data_dir)
    splits = []
    for p in paths:
        df = pd.read_csv(p)
        df["requires_human"] = df["requires_human"].astype(bool)
        splits.append(df)
    return tuple(splits)


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """X = the input columns, y = the three label columns."""
    return df[FEATURE_COLUMNS].copy(), df[LABEL_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Laya format helpers
# ---------------------------------------------------------------------------

def laya_questions() -> dict:
    """The three typed questions Laya answers for every ticket (see RESEARCH.md)."""
    return {
        "route_target": {
            "type": "choice",
            "instructions": "Which support team should handle this ticket?",
            "criteria": dict(ROUTE_DESCRIPTIONS),
        },
        "priority_level": {
            "type": "score",
            "instructions": "How urgent is this ticket, given the customer's tier, urgency and past failures?",
            "criteria": list(PRIORITIES),
        },
        "requires_human": {
            "type": "noul",
            "instructions": "This ticket should be handed to a human agent instead of an automated agent.",
        },
    }


def row_to_state(row) -> dict:
    """The ticket as Laya's `state`: the text plus the tabular columns as plain fields."""
    return {
        "query": str(row["query"]),
        "account_tier": str(row["account_tier"]),
        "channel": str(row["channel"]),
        "urgency_score": float(row["urgency_score"]),
        "past_failures": int(row["past_failures"]),
    }


def to_laya_example(row) -> dict:
    """One training row in Laya's fine-tuning JSONL format: {state, questions, expected}."""
    return {
        "state": row_to_state(row),
        "questions": laya_questions(),
        "expected": {
            "route_target": str(row["route_target"]),
            "priority_level": PRIORITIES.index(str(row["priority_level"])),  # score -> level index
            "requires_human": bool(row["requires_human"]),
        },
    }


def write_laya_jsonl(df: pd.DataFrame, path: str) -> str:
    """Dump a DataFrame as Laya fine-tuning JSONL and return the path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(to_laya_example(row)) + "\n")
    return path


if __name__ == "__main__":
    # `uv run python -m src.dataset` -> builds data/train.csv, val.csv, test.csv
    train_df, val_df, test_df = train_val_test_split(build_synthetic_dataset())
    save_splits(train_df, val_df, test_df)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(train_df.head())
