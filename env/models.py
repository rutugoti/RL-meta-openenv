from __future__ import annotations
from typing import Any
from pydantic import BaseModel, model_validator

# ── valid action operations ──────────────────────────────────────
VALID_OPS = {
    "rename_column",
    "cast_dtype",
    "drop_column",
    "fill_nulls",
    "strip_whitespace",
}

# ── Observation ──────────────────────────────────────────────────
# What the agent sees at every step.
class Observation(BaseModel):
    columns:     list[str]
    dtypes:      dict[str, str]
    null_counts: dict[str, int]
    sample_rows: list[dict[str, Any]]
    step_count:  int
    task_id:     int
    done:        bool

# ── Action ───────────────────────────────────────────────────────
# What the agent submits each step.
class Action(BaseModel):
    op:     str
    params: dict[str, Any] = {}

    @model_validator(mode="after")
    def validate_op(self) -> Action:
        if self.op not in VALID_OPS:
            raise ValueError(
                f"Invalid op '{self.op}'. Must be one of: {VALID_OPS}"
            )
        return self

# ── Reward ───────────────────────────────────────────────────────
# Returned by step() every turn — always partial, never binary.
class Reward(BaseModel):
    total:        float   # 0.0 – 1.0 final weighted score
    name_score:   float   # fraction of cols named correctly
    dtype_score:  float   # fraction of cols with correct dtype
    null_score:   float   # how well nulls are handled
    step_penalty: float   # negative when action was invalid