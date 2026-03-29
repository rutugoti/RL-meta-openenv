from __future__ import annotations
import pandas as pd
from env.models import Observation, Action, Reward
from tasks.task_definitions import load_dirty_df, TASKS

class DataCleaningEnv:
    """
    OpenEnv-compliant data cleaning environment.
    An agent applies cleaning operations to a dirty DataFrame
    and is rewarded for progress toward the target schema.
    """

    MAX_STEPS = 30

    def __init__(self, task_id: int = 1, seed: int = 42):
        if task_id not in (1, 2, 3):
            raise ValueError(f"task_id must be 1, 2, or 3. Got {task_id}")
        self.task_id  = task_id
        self.seed     = seed
        self.df:      pd.DataFrame | None = None
        self.step_count: int = 0

    # ── OpenEnv required: reset() ────────────────────────────────
    def reset(self) -> Observation:
        """Start a new episode. Returns initial observation."""
        self.df = load_dirty_df(self.task_id, self.seed)
        self.step_count = 0
        return self._make_observation()

    # ── OpenEnv required: state() ────────────────────────────────
    def state(self) -> dict:
        """Return current internal state snapshot."""
        if self.df is None:
            raise RuntimeError("Call reset() before state()")
        return {
            "task_id":     self.task_id,
            "seed":        self.seed,
            "step_count":  self.step_count,
            "max_steps":   self.MAX_STEPS,
            "columns":     list(self.df.columns),
            "dtypes":      {c: str(t) for c, t in self.df.dtypes.items()},
            "null_counts": self.df.isnull().sum().to_dict(),
            "shape":       list(self.df.shape),
        }

    # ── OpenEnv required: step() ─────────────────────────────────
    # Implemented on Day 3.
    def step(self, action: Action):
        raise NotImplementedError("step() implemented on Day 3")

    # ── Internal helpers ─────────────────────────────────────────
    def _make_observation(self) -> Observation:
        """Build Observation from current DataFrame state."""
        safe_rows = (
            self.df.head(3)
            .astype(str)           # convert everything to str for JSON safety
            .to_dict(orient="records")
        )
        return Observation(
            columns     = list(self.df.columns),
            dtypes      = {c: str(t) for c, t in self.df.dtypes.items()},
            null_counts = {c: int(v)
                           for c, v in self.df.isnull().sum().items()},
            sample_rows = safe_rows,
            step_count  = self.step_count,
            task_id     = self.task_id,
            done        = self.step_count >= self.MAX_STEPS,
        )