"""
DataCleaningEnv — Baseline Inference Script
============================================
Priority 1: Uses GPT-4o if OPENAI_API_KEY is set.
Priority 2: Falls back to deterministic mock agent if no key.

Always produces output. Never hangs waiting for API.

Usage:
    python baseline/run.py                  # auto-detect
    python baseline/run.py --verbose        # step-by-step
    python baseline/run.py --task 1         # single task
    python baseline/run.py --seed 99        # different seed
"""
from __future__ import annotations
import os
import re
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
{"op": "rename_column",    "params": {"from_col": "OLD", "to_col": "NEW"}}
{"op": "cast_dtype",       "params": {"col": "NAME", "dtype": "DTYPE"}}
{"op": "drop_column",      "params": {"col": "NAME"}}
{"op": "fill_nulls",       "params": {"col": "NAME", "strategy": "mean|median|mode|value"}}
{"op": "strip_whitespace", "params": {"col": "all"}}

Valid dtypes: int64, float64, object, datetime64[ns]
Only use column names that currently exist. Output raw JSON only."""


# ── Helpers ───────────────────────────────────────────────────────
def parse_action(text: str) -> Action | None:
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


def obs_to_prompt(obs: Observation) -> str:
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


def _log(verbose: bool, msg: str):
    if verbose:
        print(msg, file=sys.stderr, flush=True)


# ── Priority 1: Real GPT-4o agent ────────────────────────────────
def run_task_openai(
    client,
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
        messages.append({"role": "user", "content": obs_to_prompt(obs)})
        try:
            response = client.chat.completions.create(
                model       = "gpt-4o",
                messages    = messages,
                temperature = 0.0,
                max_tokens  = 150,
                timeout     = 30,    # 30s per call — never hangs
            )
            raw_text = response.choices[0].message.content
        except Exception as e:
            _log(verbose, f"  [API error] {e}")
            raw_text = ""

        action = parse_action(raw_text)
        if action is None:
            parse_fails += 1
            action = Action(op="strip_whitespace", params={"col": "all"})
            _log(verbose, f"  [parse fail] {repr(raw_text[:60])}")

        messages.append({"role": "assistant", "content": raw_text or "{}"})
        obs, reward, done, info = env.step(action)
        final_score = reward.total
        steps_taken += 1

        status = "OK " if info["action_valid"] else "BAD"
        _log(verbose, f"  [{status}] step {steps_taken:2d} | "
                      f"{action.op:<20} | score={reward.total:.4f}")
        if done:
            break

    return {
        "task_id":      task_id,
        "score":        grade(task_id, env.df),
        "reward_score": round(final_score, 4),
        "steps":        steps_taken,
        "parse_fails":  parse_fails,
        "done":         obs.done,
        "agent":        "gpt-4o",
    }


# ── Priority 2: Deterministic mock agent (no API key needed) ──────
def run_task_mock(
    task_id: int,
    seed:    int  = 42,
    verbose: bool = False,
) -> dict:
    """
    Rule-based fallback agent.
    Applies correct cleaning actions deterministically.
    Used automatically when OPENAI_API_KEY is not set.
    """
    env = DataCleaningEnv(task_id=task_id, seed=seed)
    obs = env.reset()

    sequences = {
        1: [
            Action(op="strip_whitespace",
                   params={"col": "all"}),
            Action(op="rename_column",
                   params={"from_col": "User ID", "to_col": "user_id"}),
            Action(op="rename_column",
                   params={"from_col": "First Name", "to_col": "first_name"}),
            Action(op="rename_column",
                   params={"from_col": "AGE", "to_col": "age"}),
            Action(op="rename_column",
                   params={"from_col": "email_address", "to_col": "email"}),
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
                   params={"from_col": "OrderID",
                            "to_col": "order_id"}),
            Action(op="rename_column",
                   params={"from_col": "cust",
                            "to_col": "customer_id"}),
            Action(op="rename_column",
                   params={"from_col": "Order Date",
                            "to_col": "order_date"}),
            Action(op="rename_column",
                   params={"from_col": "Amount",
                            "to_col": "amount"}),
            Action(op="rename_column",
                   params={"from_col": "STATUS",
                            "to_col": "status"}),
            Action(op="rename_column",
                   params={"from_col": "CustomerName",
                            "to_col": "customer_name"}),
            Action(op="fill_nulls",
                   params={"col": "customer_name", "strategy": "value",
                            "value": "unknown"}),
            Action(op="cast_dtype",
                   params={"col": "order_id", "dtype": "int64"}),
        ],
    }

    actions     = sequences.get(task_id, [])
    final_score = 0.0
    steps_taken = 0

    for action in actions:
        if obs.done:
            break
        obs, reward, done, info = env.step(action)
        final_score = reward.total
        steps_taken += 1
        status = "OK " if info["action_valid"] else "BAD"
        _log(verbose, f"  [{status}] step {steps_taken:2d} | "
                      f"{action.op:<20} | score={reward.total:.4f}")

    return {
        "task_id":      task_id,
        "score":        grade(task_id, env.df),
        "reward_score": round(final_score, 4),
        "steps":        steps_taken,
        "parse_fails":  0,
        "done":         obs.done,
        "agent":        "mock-deterministic",
    }


# ── Auto-detect which agent to use ───────────────────────────────
def run_task(
    task_id:  int,
    seed:     int  = 42,
    verbose:  bool = False,
    force_mock: bool = False,
) -> dict:
    """
    Auto-selects agent:
    - GPT-4o if OPENAI_API_KEY is set and valid
    - Mock deterministic agent otherwise
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if not force_mock and api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            return run_task_openai(client, task_id, seed, verbose)
        except Exception as e:
            _log(True, f"  [OpenAI unavailable: {e}] — falling back to mock agent")

    # fallback
    if not force_mock:
        _log(True, "  [No OPENAI_API_KEY] — using deterministic mock agent")
    return run_task_mock(task_id, seed, verbose)


# ── Entry point ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="DataCleaningEnv baseline — auto-selects GPT-4o or mock"
    )
    parser.add_argument("--verbose",    action="store_true",
                        help="Print step-by-step agent actions")
    parser.add_argument("--task",       type=int, default=0,
                        help="Run single task (1/2/3). Default: all")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--mock",       action="store_true",
                        help="Force mock agent even if API key is set")
    args = parser.parse_args()

    task_ids = [args.task] if args.task in (1, 2, 3) else [1, 2, 3]
    api_key  = os.environ.get("OPENAI_API_KEY", "").strip()
    mode     = "mock (forced)" if args.mock else \
               ("gpt-4o" if api_key else "mock (no API key)")

    _log(True, f"Running baseline — agent={mode} seed={args.seed}")

    results = []
    for task_id in task_ids:
        _log(args.verbose, f"\nTask {task_id}:")
        result = run_task(task_id, seed=args.seed,
                          verbose=args.verbose,
                          force_mock=args.mock)
        results.append(result)
        _log(True, f"  Task {task_id} score: {result['score']} "
                   f"(agent: {result['agent']})")

    # stdout = JSON only — this is what /baseline endpoint captures
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()