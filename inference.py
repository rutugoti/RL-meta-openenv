"""
inference.py — OpenEnv required entry point.
The automated validator imports this file and calls run_inference().

This wraps the GPT-4o agent in baseline/run.py.
Set OPENAI_API_KEY in environment before running.
"""
from __future__ import annotations
import os
import sys

# ensure project root is on path regardless of how this file is called
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import DataCleaningEnv
from env.models      import Action, Observation
from graders.grader  import grade


def run_inference(
    env:     DataCleaningEnv,
    task_id: int = 1,
    seed:    int = 42,
) -> dict:
    """
    Run a GPT-4o agent on one episode of DataCleaningEnv.

    Args:
        env:     An instantiated DataCleaningEnv.
        task_id: Task to run (1=easy, 2=medium, 3=hard).
        seed:    Random seed for reproducibility.

    Returns:
        dict with keys: task_id, score, steps, done
    """
    import re
    import json
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Export it before running inference."
        )

    client = OpenAI(api_key=api_key)

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

    def _parse(text: str) -> Action | None:
        if not text:
            return None
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`")
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if not m:
            return None
        try:
            d = json.loads(m.group())
            if "op" in d and "params" not in d:
                d = {"op": d["op"],
                     "params": {k: v for k, v in d.items() if k != "op"}}
            return Action(**d)
        except Exception:
            return None

    def _fmt(obs: Observation) -> str:
        return (
            f"Step {obs.step_count + 1}/30\n"
            f"Columns: {obs.columns}\n"
            f"Dtypes:  {obs.dtypes}\n"
            f"Nulls:   {obs.null_counts}\n"
            f"Rows:    {obs.sample_rows[:3]}\n"
            f"Choose ONE cleaning action. Raw JSON only."
        )

    obs      = env.reset()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    steps    = 0
    score    = 0.0

    while not obs.done:
        messages.append({"role": "user", "content": _fmt(obs)})
        try:
            resp = client.chat.completions.create(
                model       = "gpt-4o",
                messages    = messages,
                temperature = 0.0,
                max_tokens  = 150,
            )
            raw = resp.choices[0].message.content
        except Exception:
            raw = ""

        action = _parse(raw) or Action(
            op="strip_whitespace", params={"col": "all"})
        messages.append({"role": "assistant", "content": raw or "{}"})

        obs, reward, done, _ = env.step(action)
        score = reward.total
        steps += 1
        if done:
            break

    return {
        "task_id": task_id,
        "score":   grade(task_id, env.df),
        "steps":   steps,
        "done":    obs.done,
    }


# ── Standalone usage ─────────────────────────────────────────────
if __name__ == "__main__":
    import json as _json
    results = []
    for tid in [1, 2, 3]:
        e = DataCleaningEnv(task_id=tid, seed=42)
        r = run_inference(e, task_id=tid, seed=42)
        results.append(r)
        print(f"Task {tid}: {r['score']}", flush=True)
    print(_json.dumps(results, indent=2))