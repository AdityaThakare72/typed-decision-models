"""Fine-tuned bert-base-uncased baseline.

BERT only reads text, so the tabular columns are written into the input as a short prefix:
    "tier: pro | channel: chat | urgency: 0.82 | failures: 2 [SEP] my app keeps crashing"

One BERT, three answers: we use a single sequence-classification model with
6 + 4 + 2 = 12 output logits and slice them into three groups (route / priority / human),
each with its own cross-entropy loss. One forward pass gives the whole decision,
like Laya, instead of running three separate 110M-parameter models.

torch / transformers are imported inside functions so the file imports without them.
"""

import numpy as np
import pandas as pd

from src.dataset import PRIORITIES, ROUTES
from src.interface import DecisionModel, DecisionOutput

BERT_MODEL_ID = "bert-base-uncased"
HUMAN_CLASSES = [False, True]

# Where each head's logits live inside the 12-wide output vector.
ROUTE_SLICE = slice(0, len(ROUTES))
PRIORITY_SLICE = slice(len(ROUTES), len(ROUTES) + len(PRIORITIES))
HUMAN_SLICE = slice(len(ROUTES) + len(PRIORITIES), len(ROUTES) + len(PRIORITIES) + 2)
NUM_LOGITS = HUMAN_SLICE.stop


def row_to_text(row) -> str:
    """Flatten one ticket (text + tabular) into a single string BERT can read."""
    prefix = (f"tier: {row['account_tier']} | channel: {row['channel']} | "
              f"urgency: {float(row['urgency_score']):.2f} | failures: {int(row['past_failures'])}")
    return f"{prefix} [SEP] {row['query']}"


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    """Labels -> integer ids, one column per head. Shape (n_rows, 3)."""
    route_ids = df["route_target"].map(ROUTES.index)
    priority_ids = df["priority_level"].map(PRIORITIES.index)
    human_ids = df["requires_human"].astype(bool).astype(int)
    return np.stack([route_ids, priority_ids, human_ids], axis=1)


def split_logits(logits):
    """12 logits -> (route logits, priority logits, human logits). Works on numpy or torch."""
    return logits[:, ROUTE_SLICE], logits[:, PRIORITY_SLICE], logits[:, HUMAN_SLICE]


def softmax(x: np.ndarray) -> np.ndarray:
    """Turn raw scores into probabilities that sum to 1 (row-wise)."""
    x = x - x.max(axis=1, keepdims=True)  # subtract max for numerical stability
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def logits_to_outputs(logits: np.ndarray) -> list[DecisionOutput]:
    """Decode a (n, 12) numpy array of logits into DecisionOutputs."""
    route_logits, priority_logits, human_logits = split_logits(logits)
    route_probs = softmax(route_logits)
    outputs = []
    for i in range(len(logits)):
        outputs.append(DecisionOutput(
            route=ROUTES[int(route_probs[i].argmax())],
            priority=PRIORITIES[int(priority_logits[i].argmax())],
            requires_human=HUMAN_CLASSES[int(human_logits[i].argmax())],
            confidence=float(route_probs[i].max()),
        ))
    return outputs


class BertDecisionModel(DecisionModel):
    """bert-base-uncased fine-tuned on all three heads at once."""

    def __init__(self, model_id: str = BERT_MODEL_ID, max_length: int = 96, epochs: int = 3,
                 batch_size: int = 32, lr: float = 3e-5, device: str | None = None, seed: int = 42):
        self.model_id = model_id
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.name = "bert:" + model_id.split("/")[-1]
        self.device = device
        self.model = None
        self.tokenizer = None

    def _setup(self):
        """Load tokenizer + model with a fresh 12-logit classification layer on top."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=NUM_LOGITS)
        self.model.to(self.device)

    def _tokenize(self, texts: list[str]):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                              return_tensors="pt").to(self.device)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "BertDecisionModel":
        """A plain PyTorch training loop: batches -> forward -> 3 losses -> backward -> step."""
        import torch
        import torch.nn.functional as F

        self._setup()
        texts = [row_to_text(row) for _, row in train_df.iterrows()]
        labels = torch.tensor(encode_labels(train_df), dtype=torch.long)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        generator = np.random.default_rng(self.seed)

        for epoch in range(self.epochs):
            self.model.train()
            order = generator.permutation(len(texts))
            total_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                batch = self._tokenize([texts[i] for i in idx])
                y = labels[idx].to(self.device)

                logits = self.model(**batch).logits
                route_logits, priority_logits, human_logits = split_logits(logits)
                # Sum of the three heads' losses: the shared encoder learns all tasks together.
                loss = (F.cross_entropy(route_logits, y[:, 0])
                        + F.cross_entropy(priority_logits, y[:, 1])
                        + F.cross_entropy(human_logits, y[:, 2]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
            print(f"epoch {epoch + 1}/{self.epochs} loss={total_loss / len(texts):.4f}")
        return self

    def predict_batch(self, df: pd.DataFrame) -> list[DecisionOutput]:
        import torch

        if self.model is None:
            raise RuntimeError("call fit() or load() before predicting")
        self.model.eval()
        texts = [row_to_text(row) for _, row in df.iterrows()]
        all_logits = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = self._tokenize(texts[start:start + self.batch_size])
                all_logits.append(self.model(**batch).logits.float().cpu().numpy())
        return logits_to_outputs(np.concatenate(all_logits, axis=0))

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> "BertDecisionModel":
        """Load a model saved with save()."""
        self.model_id = path
        self._setup()
        return self

    def size_mb(self) -> float:
        if self.model is None:
            return 0.0
        total = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return total / (1024 * 1024)
