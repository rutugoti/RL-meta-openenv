---
title: Data Cleaning Env
emoji: 💻
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Data Cleaning Environment
This Space provides a Docker environment for data cleaning in RL.


# Data Cleaning Agent Environment

An OpenEnv-compliant reinforcement learning environment where an
agent learns to repair messy, real-world tabular datasets by applying
a sequence of cleaning operations.

Every data team does this daily. No OpenEnv environment trains agents
for it. This fills that gap.

---

## Environment description

The environment presents an agent with a dirty pandas DataFrame and
a target schema. The agent applies cleaning operations step by step
and is rewarded for progress toward the target. Episodes end after
30 steps or when the agent is done.

**Real-world motivation:** Data cleaning consumes 60-80% of a data
scientist's time. An agent trained in this environment could automate
column renaming, dtype normalization, null imputation, and schema
alignment — tasks that are tedious, repetitive, and well-defined
enough for RL.

---

## Action space

Each step the agent submits exactly one action as JSON:

```json
{"op": "rename_column", "params": {"from_col": "OLD", "to_col": "NEW"}}
```

| Op | Required params | Description |
|---|---|---|
| `rename_column` | `from_col`, `to_col` | Rename a column |
| `cast_dtype` | `col`, `dtype` | Cast column to dtype (int64 / float64 / object / datetime64[ns]) |
| `drop_column` | `col` | Drop a column (cannot drop last column) |
| `fill_nulls` | `col`, `strategy` | Fill nulls — strategy: mean / median / mode / value |
| `strip_whitespace` | `col` | Strip leading/trailing spaces — use "all" for all string columns |

Invalid actions (non-existent column, bad dtype) return `False`
from `_apply_action()` and apply a -0.05 step penalty. They never crash.

---

## Observation space

Returned by `reset()` and `step()` as a typed Pydantic model:

| Field | Type | Description |
|---|---|---|
| `columns` | `list[str]` | Current column names |
| `dtypes` | `dict[str, str]` | Column → dtype string |
| `null_counts` | `dict[str, int]` | Null count per column |
| `sample_rows` | `list[dict]` | First 3 rows as dicts |
| `step_count` | `int` | Steps taken this episode |
| `task_id` | `int` | Active task (1, 2, or 3) |
| `done` | `bool` | True when episode has ended |

---

## Tasks

### Task 1 — Easy (expected agent score: 0.70–0.90)

The dirty DataFrame has wrong column names, a duplicate column,
and leading/trailing whitespace in string values.

**Agent must:** rename 4 columns to target names, drop 1 duplicate
column, strip whitespace from all string columns.

**Grader:** exact column name match (80%) + whitespace cleanliness (20%).

---

### Task 2 — Medium (expected agent score: 0.40–0.65)

The dirty DataFrame has wrong column names, nulls in two columns,
incorrect dtypes (int stored as float, dates stored as strings).

**Agent must:** rename 6 columns, fill nulls by policy (mean for
numeric, "unknown" for categorical), cast 2 columns to correct dtypes.

**Grader:** name score (30%) + dtype score (50%) + null score (20%).

---

### Task 3 — Hard (expected agent score: 0.15–0.35)

A multi-column dirty table with mixed date formats stored as strings,
a foreign key column with wrong name, amounts stored as "$X.XX"
strings, and missing customer names.

**Agent must:** rename all columns to snake_case, normalize date
column to datetime64[ns], detect and rename the FK column, cast
amount to float64, fill null customer names.

**Grader:** field-by-field schema diff — partial credit per field.

---

## Reward function

Partial progress signal provided at every step (not only at episode end):

```
reward = name_score × 0.35
       + dtype_score × 0.40
       + null_score  × 0.25
       + step_penalty        # -0.05 for invalid actions
```

Scores are clamped to [0.0, 1.0]. The reward varies meaningfully
with each action — a correct rename immediately raises name_score.

---

## Setup

```bash
git clone https://github.com/rutugoti/RL-meta-openenv
cd RL-meta-openenv
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Run baseline (requires OpenAI API key):

```bash
export OPENAI_API_KEY=sk-...
python baseline/run.py
```


Run tests:

```bash
pytest tests/ -v
```

---

## Docker

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

Test after running:

```bash
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" -d "{}"
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Start new episode, returns Observation |
| POST | `/step` | Submit action, returns obs + reward + done + info |
| GET | `/state` | Current environment state |
| GET | `/tasks` | All tasks + action schema |
| POST | `/baseline` | Run baseline script, returns scores |
| POST | `/grader` | Grade a final state |

Swagger UI: `https://ruuuuq-data-cleaning-env.hf.space/docs`

---

## Baseline scores

Evaluated with seed=42 on all 3 tasks:
| Task | Difficulty | Baseline score |
|------|------------|----------------|
| Task 1 | Easy   | **0.8000**     |
| Task 2 | Medium | **0.7673**     |
| Task 3 | Hard   | **0.8045**     |

Scores produced by `python baseline/run.py` with seed=42.
this can only run with your  own `OPENAI_API_KEY` for GPT-4o evaluation.

---

## Project structure

```
data-cleaning-env/
├── env/
│   ├── models.py          # Pydantic models: Observation, Action, Reward
│   └── environment.py     # DataCleaningEnv — reset(), step(), state()
├── tasks/
│   └── task_definitions.py # Target schemas + dirty dataset generators
├── graders/
│   └── grader.py          # Deterministic graders for all 3 tasks
├── baseline/
│   └── run.py             # GPT-4o agent loop 
├── api/
│   └── main.py            # FastAPI server — all 6 required endpoints
├── tests/
│   ├── test_env.py        # 11 environment tests
│   ├── test_graders.py    # 8 grader tests
│   └── test_exploits.py   # 5 exploit proofing tests
├── openenv.yaml           # OpenEnv spec metadata
├── requirements.txt       # Pinned dependencies
└── Dockerfile             # HF Spaces Docker container
```

---

## Author

RUTU GOTI

HuggingFace Space: https://ruuuuq-data-cleaning-env.hf.space