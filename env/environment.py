from __future__ import annotations
import pandas as pd
from env.models import Observation, Action, Reward
from tasks.task_definitions import load_dirty_df, TASKS


class DataCleaningEnv:
    """
    OpenEnv-compliant Data Cleaning Agent Environment.
    Agent applies cleaning ops to a dirty DataFrame and
    is rewarded for progress toward the target schema.
    """

    MAX_STEPS = 30

    def __init__(self, task_id: int = 1, seed: int = 42):
        if task_id not in (1, 2, 3):
            raise ValueError(f"task_id must be 1, 2, or 3. Got {task_id}")
        self.task_id     = task_id
        self.seed        = seed
        self.df: pd.DataFrame | None = None
        self.step_count  = 0

    # ── OpenEnv required ─────────────────────────────────────────

    def reset(self) -> Observation:
        self.df          = load_dirty_df(self.task_id, self.seed)
        self.step_count  = 0
        return self._make_observation()

    def state(self) -> dict:
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

    def step(self, action: Action):
        if self.df is None:
            raise RuntimeError("Call reset() before step()")
        action_valid = self._apply_action(action)
        self.step_count += 1
        done   = self.step_count >= self.MAX_STEPS
        target = TASKS[self.task_id]
        reward = self._compute_reward(action_valid, target)
        obs    = self._make_observation()
        # build rich info for debugging and agent logging
        target      = TASKS[self.task_id]
        tgt_cols    = set(target["target_columns"])
        curr_cols   = set(self.df.columns)
        matched     = curr_cols & tgt_cols
        unmatched   = tgt_cols - curr_cols
        total_nulls = int(self.df.isnull().sum().sum())

        info = {
            "action_valid":      action_valid,
            "step":              self.step_count,
            "op":                action.op,
            "params":            action.params,
            "columns_now":       list(self.df.columns),
            "columns_matched":   list(matched),
            "columns_missing":   list(unmatched),
            "total_nulls":       total_nulls,
            "steps_remaining":   self.MAX_STEPS - self.step_count,
            "reward_breakdown":  {
                "name_score":    reward.name_score,
                "dtype_score":   reward.dtype_score,
                "null_score":    reward.null_score,
                "step_penalty":  reward.step_penalty,
                "total":         reward.total,
            },
        }
        return obs, reward, done, info

    # ── Internal: apply action ────────────────────────────────────

    def _apply_action(self, action: Action) -> bool:
        op = action.op
        p  = action.params
        try:
            if op == "rename_column":
                src, dst = p.get("from_col",""), p.get("to_col","")
                if not src or not dst:         return False
                if src not in self.df.columns: return False
                if src == dst:                 return False
                self.df.rename(columns={src: dst}, inplace=True)

            elif op == "cast_dtype":
                col, dtype = p.get("col",""), p.get("dtype","")
                if not col or not dtype:       return False
                if col not in self.df.columns: return False
                valid = {"int64","float64","object","str",
                         "datetime64[ns]","bool"}
                if dtype not in valid:         return False
                if dtype == "datetime64[ns]":
                    self.df[col] = pd.to_datetime(
                        self.df[col], errors="coerce")
                else:
                    self.df[col] = self.df[col].astype(
                        dtype, errors="ignore")

            elif op == "drop_column":
                col = p.get("col","")
                if not col:                    return False
                if col not in self.df.columns: return False
                if len(self.df.columns) <= 1:  return False
                self.df.drop(columns=[col], inplace=True)

            elif op == "fill_nulls":
                col      = p.get("col","")
                strategy = p.get("strategy","")
                if not col or not strategy:    return False
                if col not in self.df.columns: return False
                if not self.df[col].isnull().any(): return False
                if strategy == "mean":
                    if not pd.api.types.is_numeric_dtype(
                            self.df[col]):     return False
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == "median":
                    if not pd.api.types.is_numeric_dtype(
                            self.df[col]):     return False
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif strategy == "mode":
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif strategy == "value":
                    val = p.get("value", "unknown")
                    self.df[col] = self.df[col].fillna(val)
                else:
                    return False

            elif op == "strip_whitespace":
                col     = p.get("col","all")
                str_cols = self.df.select_dtypes(include=["object","string"]).columns.tolist()
                targets = str_cols if col == "all" else [col]
                for c in targets:
                    if c in self.df.columns:
                        self.df[c] = self.df[c].str.strip()

            return True
        except Exception:
            return False

    # ── Internal: compute reward ──────────────────────────────────

    def _compute_reward(self, action_valid: bool, target: dict) -> Reward:
        tgt_cols    = target["target_columns"]
        tgt_dtypes  = target.get("target_dtypes", {})
        null_policy = target.get("null_policy", {})

        # name score
        current_cols = set(self.df.columns)
        matched      = len(current_cols & set(tgt_cols))
        name_score   = matched / len(tgt_cols) if tgt_cols else 1.0

        # dtype score
        if tgt_dtypes:
            hits = sum(
                1 for col, dt in tgt_dtypes.items()
                if col in self.df.columns
                and str(self.df[col].dtype) == dt
            )
            dtype_score = hits / len(tgt_dtypes)
        else:
            dtype_score = 1.0

        # null score
        if null_policy:
            penalties = sum(
                1 for col in null_policy
                if col in self.df.columns
                and int(self.df[col].isnull().sum()) > 0
            )
            null_score = 1.0 - (penalties / len(null_policy))
        else:
            null_score = 1.0

        step_penalty = 0.0 if action_valid else -0.05
        raw   = (name_score*0.35 + dtype_score*0.40 +
                 null_score*0.25 + step_penalty)
        total = round(max(0.0, min(1.0, raw)), 4)

        return Reward(
            total        = total,
            name_score   = round(name_score,  4),
            dtype_score  = round(dtype_score, 4),
            null_score   = round(null_score,  4),
            step_penalty = step_penalty,
        )

    # ── Internal: build observation ───────────────────────────────

    def _make_observation(self) -> Observation:
        safe_rows = (
            self.df.head(3).astype(str)
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