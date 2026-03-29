import pandas as pd
import numpy as np
from typing import Any

# ── Target schemas ────────────────────────────────────────────────
# Graders compare the agent's final df against these.
TASKS: dict[int, dict[str, Any]] = {
    1: {
        "id": 1,
        "name": "basic-cleanup",
        "difficulty": "easy",
        "description": "Rename columns, drop duplicate, strip whitespace",
        "target_columns": ["user_id", "first_name", "age", "email"],
        "target_dtypes":  {
            "user_id":    "int64",
            "age":        "int64",
            "first_name": "object",
            "email":      "object",
        },
        "null_policy": {},   # no nulls expected
    },
    2: {
        "id": 2,
        "name": "type-fixing",
        "difficulty": "medium",
        "description": "Fill nulls, cast dtypes, rename 5 columns",
        "target_columns": ["id","name","score","joined_date","status","region"],
        "target_dtypes": {
            "id":          "int64",
            "score":       "float64",
            "joined_date": "datetime64[ns]",
        },
        "null_policy": {
            "score":  "fill_mean",
            "status": "fill_value",
        },
    },
    3: {
        "id": 3,
        "name": "schema-inference",
        "difficulty": "hard",
        "description": "Multi-col: infer schema, normalize dates, detect FK",
        "target_columns": [
            "order_id","customer_id","order_date",
            "amount","status","customer_name"
        ],
        "target_dtypes": {
            "order_id":    "int64",
            "amount":      "float64",
            "order_date":  "datetime64[ns]",
        },
        "null_policy": {"customer_name": "fill_value"},
        "foreign_key":  "customer_id",
    },
}

# ── Dirty dataset generators ──────────────────────────────────────
# Each returns a messy DataFrame. Deterministic given seed.

def _make_task1_df(rng: np.random.Generator) -> pd.DataFrame:
    n = 20
    return pd.DataFrame({
        "User ID ":      rng.integers(100, 999, n),   # trailing space
        "user_id":       rng.integers(100, 999, n),   # duplicate — drop
        "First Name":    [f"  {c}  " for c in       # leading/trailing ws
                          rng.choice(["alice","bob","carol","dan"], n)],
        "AGE":           rng.integers(18, 65, n),     # wrong case
        "email_address": [f"user{i}@example.com"      # name too long
                          for i in range(n)],
    })

def _make_task2_df(rng: np.random.Generator) -> pd.DataFrame:
    n = 25
    scores = rng.uniform(50, 100, n).tolist()
    # inject nulls at random positions
    null_idx = rng.choice(n, 5, replace=False)
    for i in null_idx:
        scores[i] = None
    statuses = rng.choice(["active","inactive","pending"], n).tolist()
    for i in rng.choice(n, 4, replace=False):
        statuses[i] = None
    return pd.DataFrame({
        "ID":        rng.integers(1, 500, n).astype(float),  # int as float
        "Full Name": [f"Person {i}" for i in range(n)],
        "Score":     scores,
        "JoinDate":  ["2023-01-15"] * n,                     # str not datetime
        "STATUS":    statuses,
        "Region ":   rng.choice(["North","South","East"], n), # trailing space
    })

def _make_task3_df(rng: np.random.Generator) -> pd.DataFrame:
    n = 20
    amounts = [f"${rng.integers(10,999)}.{rng.integers(0,99):02d}"
               for _ in range(n)]                            # $ strings
    dates = rng.choice(
        ["2024/01/15","Jan 15 2024","15-01-2024"], n)        # mixed formats
    names = [f"Customer {i}" for i in range(n)]
    for i in rng.choice(n, 3, replace=False):
        names[i] = None                                      # nulls
    return pd.DataFrame({
        "OrderID":       rng.integers(1000, 9999, n),
        "cust":          rng.integers(1, 100, n),            # wrong FK name
        "Order Date":    dates,
        "Amount":        amounts,
        "STATUS":        rng.choice(["shipped","pending","cancelled"], n),
        "CustomerName":  names,
    })

# ── Public loader ─────────────────────────────────────────────────
def load_dirty_df(task_id: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    makers = {1: _make_task1_df, 2: _make_task2_df, 3: _make_task3_df}
    if task_id not in makers:
        raise ValueError(f"task_id must be 1, 2, or 3. Got {task_id}")
    return makers[task_id](rng)