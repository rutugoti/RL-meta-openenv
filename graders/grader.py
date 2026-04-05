from __future__ import annotations
import pandas as pd
from tasks.task_definitions import TASKS


def grade(task_id: int, df: pd.DataFrame,
          steps_taken: int = 0) -> float:
    """
    Public grader. Returns float 0.0–1.0.
    steps_taken used by Task 3 to penalise inefficiency.
    Always deterministic. Never mutates df.
    """
    if task_id not in TASKS:
        raise ValueError(f"task_id must be 1, 2, or 3. Got {task_id}")
    target  = TASKS[task_id]
    graders = {
        1: _grade_task1,
        2: _grade_task2,
        3: _grade_task3,
    }
    raw = graders[task_id](df.copy(), target, steps_taken)
    return round(max(0.0, min(1.0, float(raw))), 4)


# ── Task 1: easy ──────────────────────────────────────────────────
def _grade_task1(df: pd.DataFrame, target: dict,
                 steps_taken: int = 0) -> float:
    """
    Score = exact_name_score (80%) + whitespace_score (20%)
    No step penalty — easy task should be fully solvable.
    """
    tgt_cols   = target["target_columns"]
    current    = set(df.columns)
    target_set = set(tgt_cols)

    # exact name score
    name_score = len(current & target_set) / len(tgt_cols)

    # whitespace score
    str_cols = df.select_dtypes(
        include=["object", "string"]).columns
    if len(str_cols) == 0:
        ws_score = 1.0
    else:
        clean_cols = 0
        for col in str_cols:
            series = df[col].dropna().astype(str)
            if series.empty:
                clean_cols += 1
                continue
            has_ws = (
                series.str.startswith(" ") |
                series.str.endswith(" ")
            ).any()
            if not has_ws:
                clean_cols += 1
        ws_score = clean_cols / len(str_cols)

    return round(name_score * 0.80 + ws_score * 0.20, 4)


# ── Task 2: medium ────────────────────────────────────────────────
def _grade_task2(df: pd.DataFrame, target: dict,
                 steps_taken: int = 0) -> float:
    """
    Score = name(30%) + dtype(50%) + null(20%)
    No step penalty — medium task, steps not penalised.
    """
    tgt_cols    = target["target_columns"]
    tgt_dtypes  = target.get("target_dtypes", {})
    null_policy = target.get("null_policy", {})

    # name score — partial credit for wrong case
    name_hits   = 0.0
    current_low = {c.strip().lower(): c for c in df.columns}
    for col in tgt_cols:
        if col in df.columns:
            name_hits += 1.0
        elif col.lower() in current_low:
            name_hits += 0.4
    name_score = name_hits / len(tgt_cols)

    # dtype score — partial credit for near-matches
    if tgt_dtypes:
        dtype_hits = 0.0
        for col, expected_dt in tgt_dtypes.items():
            if col not in df.columns:
                continue
            actual_dt = str(df[col].dtype)
            if actual_dt == expected_dt:
                dtype_hits += 1.0
            elif (expected_dt in ("int64", "float64") and
                  actual_dt  in ("int64", "float64")):
                dtype_hits += 0.5
        dtype_score = dtype_hits / len(tgt_dtypes)
    else:
        dtype_score = 1.0

    # null score — partial credit
    if null_policy:
        null_hits = 0.0
        for col in null_policy:
            actual_col = None
            if col in df.columns:
                actual_col = col
            else:
                for c in df.columns:
                    if c.strip().lower() == col.lower():
                        actual_col = c
                        break
            if actual_col is None:
                continue
            remaining = int(df[actual_col].isnull().sum())
            if remaining == 0:
                null_hits += 1.0
            else:
                null_hits += max(0.0, 1.0 - remaining / len(df))
        null_score = null_hits / len(null_policy)
    else:
        null_score = 1.0

    total = (name_score  * 0.30 +
             dtype_score * 0.50 +
             null_score  * 0.20)
    return round(total, 4)


# ── Task 3: hard — WITH step penalty ─────────────────────────────
def _grade_task3(df: pd.DataFrame, target: dict,
                 steps_taken: int = 0) -> float:
    """
    Field-by-field schema diff. Partial credit per field.
    STEP PENALTY: each step costs 0.01 (max penalty 0.30).
    This makes Task 3 genuinely hard — efficiency matters.
    Mock agent (9 steps) → -0.09 penalty.
    Random agent (30 steps) → -0.30 penalty.
    Frontier model (15-20 steps) → -0.15 to -0.20 penalty.
    """
    tgt_cols    = target["target_columns"]
    tgt_dtypes  = target.get("target_dtypes", {})
    null_policy = target.get("null_policy", {})
    fk_col      = target.get("foreign_key", None)

    scores = []

    # column presence — partial for close snake_case match
    for col in tgt_cols:
        if col in df.columns:
            scores.append(1.0)
        else:
            snake = col.replace(" ", "_").lower()
            alt   = any(
                c.replace(" ", "_").lower() == snake
                for c in df.columns
            )
            scores.append(0.3 if alt else 0.0)

    # dtype correctness
    for col, expected_dt in tgt_dtypes.items():
        if col not in df.columns:
            scores.append(0.0)
            continue
        actual_dt = str(df[col].dtype)
        if actual_dt == expected_dt:
            scores.append(1.0)
        elif (expected_dt == "datetime64[ns]" and
              "datetime" in actual_dt):
            scores.append(0.8)
        elif (expected_dt in ("float64", "int64") and
              actual_dt  in ("float64", "int64")):
            scores.append(0.5)
        else:
            scores.append(0.0)

    # null handling
    for col in null_policy:
        if col not in df.columns:
            scores.append(0.0)
            continue
        remaining = int(df[col].isnull().sum())
        scores.append(
            1.0 if remaining == 0
            else max(0.0, 1.0 - remaining / len(df))
        )

    # FK detection
    if fk_col:
        if fk_col in df.columns:
            scores.append(1.0)
        else:
            fk_like = any(
                "id" in c.lower() and c.lower() != "order_id"
                for c in df.columns
            )
            scores.append(0.2 if fk_like else 0.0)

    base_score = (
        round(sum(scores) / len(scores), 4)
        if scores else 0.0
    )

    # ── Step penalty — what makes Task 3 genuinely hard ──────────
    step_penalty = min(0.30, steps_taken * 0.01)
    final = round(max(0.0, base_score - step_penalty), 4)
    return final