"""Laya wrapper: zero-shot base checkpoints and our own fine-tuned checkpoint.

Laya answers typed questions about a "state" in one forward pass (see RESEARCH.md).
Our three output fields map straight onto its three question types:
    route_target   -> "choice"
    priority_level -> "score"  (ordered levels)
    requires_human -> "noul"   (yes/no, returns P(true))

Heavy imports (laya / torch) happen inside functions, so this file can be imported
(and its parsing logic tested) on a machine without a GPU or torch installed.
"""

import os

import pandas as pd

from src.dataset import PRIORITIES, laya_questions, row_to_state, write_laya_jsonl
from src.interface import DecisionModel, DecisionOutput

BASE_MODEL_ID = "convaiinnovations/laya"
TYPED_DECISIONS_MODEL_ID = "convaiinnovations/laya-typed-decisions"


def parse_laya_answers(answers: dict, human_threshold: float = 0.5) -> DecisionOutput:
    """Turn Laya's `result["answers"]` dict into our DecisionOutput.

    Kept as a plain function (no model needed) so it can be unit-tested offline.
    """
    route_answer = answers["route_target"]
    priority_answer = answers["priority_level"]
    human_answer = answers["requires_human"]

    # For "score" questions Laya reports an *expected* level (e.g. 1.7). We want a single label,
    # so we take the most likely level from the probabilities ({"0": p, "1": p, ...}).
    level_probs = priority_answer["probabilities"]
    best_level = max(level_probs, key=lambda level: level_probs[level])
    priority = PRIORITIES[int(best_level)]

    return DecisionOutput(
        route=str(route_answer["choice"]),
        priority=priority,
        requires_human=float(human_answer["noul"]) >= human_threshold,
        confidence=float(route_answer.get("answer_confidence", route_answer.get("confidence", 0.0))),
    )


class LayaDecisionModel(DecisionModel):
    """Any Laya checkpoint (Hub ID or local folder) behind the common interface."""

    def __init__(self, model_id: str = BASE_MODEL_ID, name: str | None = None, subfolder: str | None = None,
                 device: str | None = None, batch_size: int = 32, human_threshold: float = 0.5):
        self.model_id = model_id
        self.subfolder = subfolder
        self.device = device
        self.batch_size = batch_size
        self.human_threshold = human_threshold
        self.name = name or "laya:" + model_id.split("/")[-1]
        self.questions = laya_questions()  # built once, same for every ticket
        self.agent = None                  # loaded lazily on first use

    def load(self) -> "LayaDecisionModel":
        """Download (first time) and load the checkpoint."""
        if self.agent is None:
            # Colab has TensorFlow installed, which can deadlock transformers at import.
            os.environ.setdefault("USE_TF", "0")
            import laya
            self.agent = laya.load(self.model_id, device=self.device, subfolder=self.subfolder)
        return self

    def predict_batch(self, df: pd.DataFrame) -> list[DecisionOutput]:
        self.load()
        states = [row_to_state(row) for _, row in df.iterrows()]
        if len(states) == 1:
            # Single ticket = the router's hot path; plain predict avoids batching overhead.
            results = [self.agent.predict(states[0], self.questions)]
        else:
            results = self.agent.predict_batch(states, self.questions, batch_size=self.batch_size)
        return [parse_laya_answers(r["answers"], self.human_threshold) for r in results]

    def size_mb(self) -> float:
        """Bytes of all weights, as loaded (fp32 on CPU, maybe fp16 on GPU)."""
        if self.agent is None or getattr(self.agent, "model", None) is None:
            return 0.0
        total = sum(p.numel() * p.element_size() for p in self.agent.model.parameters())
        return total / (1024 * 1024)


def finetune_laya(train_df: pd.DataFrame, output_dir: str = "artifacts/laya-finetuned",
                  base_model: str = BASE_MODEL_ID, val_df: pd.DataFrame | None = None,
                  epochs: int = 2, micro_batch: int = 8, grad_accum: int = 4,
                  freeze_encoder: bool = False, device: str = "auto") -> dict:
    """Fine-tune Laya on our tickets with the package's own trainer (RLCD loss).

    Steps: write train (and val) rows as Laya JSONL -> `laya.train.finetune` -> a checkpoint
    folder you can load with `LayaDecisionModel(output_dir)`. Returns the trainer's summary.

    `freeze_encoder=True` trains only the small decision head: much faster and lighter on
    GPU memory, usually somewhat less accurate.
    """
    os.environ.setdefault("USE_TF", "0")
    from laya.train import TrainConfig, finetune

    train_path = write_laya_jsonl(train_df, os.path.join(output_dir + "_data", "train.jsonl"))
    eval_path = None
    if val_df is not None:
        eval_path = write_laya_jsonl(val_df, os.path.join(output_dir + "_data", "val.jsonl"))

    config = TrainConfig(epochs=epochs, micro_batch=micro_batch, grad_accum=grad_accum,
                         freeze_encoder=freeze_encoder, eval_data=eval_path)
    return finetune(train_path, base_model, output_dir, config, device=device)
