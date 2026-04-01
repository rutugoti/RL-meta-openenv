"""
Baseline inference script for DataCleaningEnv.
Runs a GPT-4o agent against all 3 tasks with seed=42.
Usage:
    set OPENAI_API_KEY=sk-...        (Windows)
    export OPENAI_API_KEY=sk-...     (Mac/Linux)
    python baseline/run.py
    python baseline/run.py --verbose
"""
from __future__ import annotations
import os
import re
import sys
import json
import argparse

from openai import OpenAI

# add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.environment import DataCleaningEnv
from env.models      import Action, Observation
from graders.grader  import grade


# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a data cleaning agent.
You receive the current state of a dirty DataFrame each step.
Your job is to clean it toward the target schema.

Respond with EXACTLY ONE action as a JSON object.
No explanation. No markdown. No code blocks. Raw JSON only.

Available operations:

1. rename_column
   {"op": "rename_column", "params": {"from_col": "OLD", "to_col": "NEW"}}

2. cast_dtype
   {"op": "cast_dtype", "params": {"col": "NAME", "dtype": "DTYPE"}}
   Valid dtypes: int64, float64, object, datetime64[ns]

3. drop_column
   {"op": "drop_column", "params": {"col": "NAME"}}

4. fill_nulls
   {"op": "fill_nulls", "params": {"col": "NAME", "strategy": "STRATEGY"}}
   Valid strategies: mean, median, mode, value
   For value strategy add: "value": "unknown"

5. strip_whitespace
   {"op": "strip_whitespace", "params": {"col": "all"}}

Rules:
- Only use column names that exist in the current DataFrame
- Choose the action most likely to move toward the target schema
- Output raw JSON only — no other text whatsoever
"""


# ── Helpers ───────────────────────────────────────────────────────
def parse_action(text: str) -> Action | None:
    if not text or not text.strip():
        return None
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.strip("`").strip()
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        if "operation" in data and "op" not in data:
            data["op"] = data.pop("operation")
        if "op" in data and "params" not in data:
            op     = data["op"]
            params = {k: v for k, v in data.items() if k != "op"}
            data   = {"op": op, "params": params}
        return Action(**data)
    except Exception:
        return None


def obs_to_prompt(obs: Observation) -> str:
    lines = [
        f"Step {obs.step_count + 1}/30",
        "",
        f"Current columns: {obs.columns}",
        f"Current dtypes:  {obs.dtypes}",
        f"Null counts:     {obs.null_counts}",
        "",
        "Sample rows (first 3):",
    ]
    for i, row in enumerate(obs.sample_rows):
        lines.append(f"  Row {i+1}: {row}")
    lines.append("")
    lines.append("Choose ONE cleaning action to apply.")
    return "\n".join(lines)


# ── Core agent loop ───────────────────────────────────────────────
def run_task(
    client:  OpenAI,
    task_id: int,
    seed:    int  = 42,
    verbose: bool = False,
) -> dict:
    env      = DataCleaningEnv(task_id=task_id, seed=seed)
    obs      = env.reset()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    final_score = 0.0
    steps_taken = 0
    parse_fails = 0

    while not obs.done:
        user_msg = obs_to_prompt(obs)
        messages.append({"role": "user", "content": user_msg})

        # call LLM
        try:
            response = client.chat.completions.create(
                model       = "gpt-4o",
                messages    = messages,
                temperature = 0.0,
                max_tokens  = 150,
            )
            raw_text = response.choices[0].message.content
        except Exception as e:
            if verbose:
                print(f"  [API error] {e}", file=sys.stderr)
            raw_text = ""

        # parse action — fallback to safe no-op if parsing fails
        action = parse_action(raw_text)
        if action is None:
            parse_fails += 1
            action = Action(op="strip_whitespace",
                            params={"col": "all"})

        messages.append({
            "role":    "assistant",
            "content": raw_text or "{}"
        })

        # step environment
        obs, reward, done, info = env.step(action)
        final_score = reward.total
        steps_taken += 1

        if verbose:
            status = "OK " if info["action_valid"] else "BAD"
            print(
                f"  [{status}] step {steps_taken:2d} | "
                f"{action.op:<20} | score={reward.total:.4f}"
            )

        if done:
            break

    # official grader score
    grader_score = grade(task_id, env.df)

    return {
        "task_id":      task_id,
        "score":        grader_score,
        "reward_score": round(final_score, 4),
        "steps":        steps_taken,
        "parse_fails":  parse_fails,
        "done":         obs.done,
    }


# ── Entry point ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="Print step-by-step agent actions")
    parser.add_argument("--task", type=int, default=0,
                        help="Run single task (1/2/3). Default: all")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set",
              file=sys.stderr)
        sys.exit(1)

    client   = OpenAI(api_key=api_key)
    task_ids = [args.task] if args.task in (1, 2, 3) else [1, 2, 3]

    if args.verbose:
        print(f"Running baseline with seed={args.seed}", file=sys.stderr)

    results = []
    for task_id in task_ids:
        if args.verbose:
            print(f"\nTask {task_id}:", file=sys.stderr)
        result = run_task(client, task_id,
                          seed=args.seed, verbose=args.verbose)
        results.append(result)

    # output JSON to stdout — this is what /baseline endpoint returns
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()