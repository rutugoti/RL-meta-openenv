"""
baseline/run.py — Baseline inference script.

Uses the same LLM proxy as inference.py.
Reads API_KEY, API_BASE_URL, MODEL_NAME from environment.

Usage:
    export API_KEY=sk-...
    export API_BASE_URL=https://api.openai.com/v1
    python baseline/run.py
    python baseline/run.py --verbose
    python baseline/run.py --task 1
"""
from __future__ import annotations
import os
import re
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai          import OpenAI
from env.environment import DataCleaningEnv
from env.models      import Action, Observation
from graders.grader  import grade

# ── Env vars ──────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("API_KEY",      os.environ.get("OPENAI_API_KEY", ""))
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o")

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
    lines = [
        f"Step {obs.step_count + 1}/30", "",
        f"Columns:     {obs.columns}",
        f"Dtypes:      {obs.dtypes}",
        f"Null counts: {obs.null_counts}", "",
        "Sample rows (first 3):",
    ]
    for i, row in enumerate(obs.sample_rows):
        lines.append(f"  Row {i+1}: {row}")
    lines += ["", "Choose ONE cleaning action. Raw JSON only."]
    return "\n".join(lines)


def run_task(
    task_id: int,
    seed:    int  = 42,
    verbose: bool = False,
) -> dict:
    if not API_KEY:
        print("ERROR: API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it with: export API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)

    client   = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env      = DataCleaningEnv(task_id=task_id, seed=seed)
    obs      = env.reset()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    steps    = 0
    parse_fails = 0

    while not obs.done:
        messages.append({"role": "user", "content": _fmt_obs(obs)})
        try:
            response = client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = messages,
                temperature = 0.0,
                max_tokens  = 150,
                timeout     = 30,
            )
            raw_text = response.choices[0].message.content
        except Exception as e:
            if verbose:
                print(f"  [API error] {e}", file=sys.stderr)
            raw_text = ""

        action = _parse_action(raw_text)
        if action is None:
            parse_fails += 1
            action = Action(op="strip_whitespace", params={"col": "all"})
            if verbose:
                print(f"  [parse fail] {repr(raw_text[:60])}", file=sys.stderr)

        messages.append({"role": "assistant", "content": raw_text or "{}"})
        obs, reward, done, info = env.step(action)
        steps += 1

        if verbose:
            status = "OK " if info["action_valid"] else "BAD"
            print(f"  [{status}] step {steps:2d} | "
                  f"{action.op:<20} | score={reward.total:.4f}",
                  file=sys.stderr)
        if done:
            break

    return {
        "task_id":      task_id,
        "score":        grade(task_id, env.df),
        "reward_score": round(reward.total, 4),
        "steps":        steps,
        "parse_fails":  parse_fails,
        "done":         obs.done,
        "model":        MODEL_NAME,
    }


def main():
    parser = argparse.ArgumentParser(
        description="DataCleaningEnv baseline — runs LLM agent on all 3 tasks"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--task",    type=int, default=0,
                        help="Run single task (1/2/3). Default: all")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    task_ids = [args.task] if args.task in (1, 2, 3) else [1, 2, 3]

    if args.verbose:
        print(f"model={MODEL_NAME} base_url={API_BASE_URL} seed={args.seed}",
              file=sys.stderr)

    results = []
    for task_id in task_ids:
        if args.verbose:
            print(f"\nTask {task_id}:", file=sys.stderr)
        result = run_task(task_id, seed=args.seed, verbose=args.verbose)
        results.append(result)
        if args.verbose:
            print(f"  Score: {result['score']}", file=sys.stderr)

    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()