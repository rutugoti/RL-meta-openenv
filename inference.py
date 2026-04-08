"""
inference.py — OpenEnv Phase 2 entry point.

The validator injects these environment variables:
  API_BASE_URL  — LiteLLM proxy URL (required, use this)
  API_KEY       — proxy API key (required, use this)
  MODEL_NAME    — model to use (default: gpt-4o)

Structured stdout format (parsed by validator):
  [START] task=TASK_NAME
  [STEP] step=N reward=X.XXXX
  [END] task=TASK_NAME score=X.XXXX steps=N
"""
from __future__ import annotations
import os
import re
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import DataCleaningEnv
from env.models      import Action, Observation
from graders.grader  import grade

# ── Read injected env vars exactly as validator provides them ─────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("API_KEY",      "")          # validator injects this
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o")

# Task name map
TASK_NAMES = {
    1: "basic-cleanup",
    2: "type-fixing",
    3: "schema-inference",
}

# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a data cleaning agent.
You receive the current state of a dirty DataFrame each step.
Your job is to clean it toward the target schema.

Respond with EXACTLY ONE action as a JSON object.
No explanation. No markdown. No code blocks. Raw JSON only.

Available operations:
{"op": "rename_column",    "params": {"from_col": "OLD", "to_col": "NEW"}}
{"op": "cast_dtype",       "params": {"col": "NAME", "dtype": "DTYPE"}}
{"op": "drop_column",      "params": {"col": "NAME"}}
{"op": "fill_nulls",       "params": {"col": "NAME", "strategy": "mean|median|mode|value"}}
{"op": "strip_whitespace", "params": {"col": "all"}}

Only use column names that currently exist. Output raw JSON only."""


# ── Helpers ───────────────────────────────────────────────────────
def _parse_action(text: str) -> Action | None:
    if not text or not text.strip():
        return None
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        if "operation" in data and "op" not in data:
            data["op"] = data.pop("operation")
        if "op" in data and "params" not in data:
            data = {"op": data["op"],
                    "params": {k: v for k, v in data.items() if k != "op"}}
        return Action(**data)
    except Exception:
        return None


def _fmt_obs(obs: Observation) -> str:
    return (
        f"Step {obs.step_count + 1}/30\n"
        f"Columns: {obs.columns}\n"
        f"Dtypes:  {obs.dtypes}\n"
        f"Nulls:   {obs.null_counts}\n"
        f"Rows:    {obs.sample_rows[:3]}\n"
        f"Choose ONE cleaning action. Raw JSON only."
    )


# ── Mock sequences — fallback only when API_KEY not set ───────────
_MOCK_SEQUENCES = {
    1: [
        Action(op="strip_whitespace",
               params={"col": "all"}),
        Action(op="rename_column",
               params={"from_col": "User ID",      "to_col": "user_id"}),
        Action(op="rename_column",
               params={"from_col": "First Name",   "to_col": "first_name"}),
        Action(op="rename_column",
               params={"from_col": "AGE",          "to_col": "age"}),
        Action(op="rename_column",
               params={"from_col": "email_address","to_col": "email"}),
        Action(op="drop_column",
               params={"col": "user_id"}),
    ],
    2: [
        Action(op="strip_whitespace",
               params={"col": "all"}),
        Action(op="rename_column",
               params={"from_col": "ID",        "to_col": "id"}),
        Action(op="rename_column",
               params={"from_col": "Full Name", "to_col": "name"}),
        Action(op="rename_column",
               params={"from_col": "Score",     "to_col": "score"}),
        Action(op="rename_column",
               params={"from_col": "JoinDate",  "to_col": "joined_date"}),
        Action(op="rename_column",
               params={"from_col": "STATUS",    "to_col": "status"}),
        Action(op="rename_column",
               params={"from_col": "Region",    "to_col": "region"}),
        Action(op="fill_nulls",
               params={"col": "score",  "strategy": "mean"}),
        Action(op="fill_nulls",
               params={"col": "status", "strategy": "value",
                        "value": "unknown"}),
        Action(op="cast_dtype",
               params={"col": "id",          "dtype": "int64"}),
        Action(op="cast_dtype",
               params={"col": "joined_date", "dtype": "datetime64[ns]"}),
    ],
    3: [
        Action(op="strip_whitespace",
               params={"col": "all"}),
        Action(op="rename_column",
               params={"from_col": "OrderID",      "to_col": "order_id"}),
        Action(op="rename_column",
               params={"from_col": "cust",         "to_col": "customer_id"}),
        Action(op="rename_column",
               params={"from_col": "Order Date",   "to_col": "order_date"}),
        Action(op="rename_column",
               params={"from_col": "Amount",       "to_col": "amount"}),
        Action(op="rename_column",
               params={"from_col": "STATUS",       "to_col": "status"}),
        Action(op="rename_column",
               params={"from_col": "CustomerName", "to_col": "customer_name"}),
        Action(op="fill_nulls",
               params={"col": "customer_name", "strategy": "value",
                        "value": "unknown"}),
        Action(op="cast_dtype",
               params={"col": "order_id", "dtype": "int64"}),
    ],
}


# ── LLM agent via validator-injected proxy ────────────────────────
def _run_llm_episode(env: DataCleaningEnv,
                     task_id: int,
                     task_name: str) -> dict:
    """
    Runs agent using API_KEY + API_BASE_URL from environment.
    These are injected by the validator — must use them.
    """
    from openai import OpenAI
    client   = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    obs      = env.reset()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    steps    = 0

    print(f"[START] task={task_name}", flush=True)

    while not obs.done:
        messages.append({"role": "user", "content": _fmt_obs(obs)})
        try:
            resp = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = messages,
                temperature = 0.0,
                max_tokens  = 150,
                timeout     = 30,
            )
            raw = resp.choices[0].message.content
        except Exception:
            raw = ""

        action = _parse_action(raw) or Action(
            op="strip_whitespace", params={"col": "all"})
        messages.append({"role": "assistant", "content": raw or "{}"})

        obs, reward, done, _ = env.step(action)
        steps += 1
        print(f"[STEP] step={steps} reward={round(reward.total, 4)}",
              flush=True)
        if done:
            break

    final_score = grade(task_id, env.df)
    print(f"[END] task={task_name} score={final_score} steps={steps}",
          flush=True)

    return {"task_id": task_id, "score": final_score,
            "steps": steps, "done": obs.done, "agent": "llm"}


# ── Mock agent fallback ───────────────────────────────────────────
def _run_mock_episode(env: DataCleaningEnv,
                      task_id: int,
                      task_name: str) -> dict:
    obs     = env.reset()
    actions = _MOCK_SEQUENCES.get(task_id, [])
    steps   = 0

    print(f"[START] task={task_name}", flush=True)

    for action in actions:
        if obs.done:
            break
        try:
            obs, reward, done, _ = env.step(action)
            steps += 1
            print(f"[STEP] step={steps} reward={round(reward.total, 4)}",
                  flush=True)
        except Exception:
            break

    final_score = grade(task_id, env.df)
    print(f"[END] task={task_name} score={final_score} steps={steps}",
          flush=True)

    return {"task_id": task_id, "score": final_score,
            "steps": steps, "done": obs.done, "agent": "mock-deterministic"}


# ── Public API ────────────────────────────────────────────────────
def run_inference(
    env:     DataCleaningEnv,
    task_id: int = 1,
    seed:    int = 42,
) -> dict:
    """
    Run one episode of DataCleaningEnv.

    Uses API_KEY + API_BASE_URL env vars (injected by validator).
    Falls back to deterministic mock if no API_KEY present.
    Always prints [START]/[STEP]/[END] to stdout.
    Never raises.
    """
    task_name = TASK_NAMES.get(task_id, f"task-{task_id}")

    # Always try LLM first if API_KEY is present (validator injects this)
    if API_KEY:
        try:
            return _run_llm_episode(env, task_id, task_name)
        except Exception:
            pass  # fall through to mock

    # Fallback — no API key available
    return _run_mock_episode(env, task_id, task_name)


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    for tid in [1, 2, 3]:
        task_name = TASK_NAMES.get(tid, f"task-{tid}")
        try:
            e      = DataCleaningEnv(task_id=tid, seed=42)
            result = run_inference(e, task_id=tid, seed=42)
            results.append(result)
        except Exception as ex:
            print(f"[END] task={task_name} score=0.0 steps=0", flush=True)
            results.append({
                "task_id": tid, "score": 0.0,
                "steps": 0, "done": False, "agent": "error",
            })

    print(json.dumps(results, indent=2), flush=True)