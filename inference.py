#!/usr/bin/env python3
"""
inference.py — OpenEnv Phase 2 entry point.
Corrected: MODEL_NAME now has a default value to prevent crash.
"""
import os
import sys
import json
import re
from typing import List, Optional

# Ensure local modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

# Import your specific environment and models
# Ensure these paths match your repository structure
try:
    from env.environment import DataCleaningEnv
    from env.models      import Action, Observation
    from graders.grader  import grade
except ImportError as e:
    print(f"[CRITICAL] Import failed: {e}. Ensure running from repo root.", flush=True)
    sys.exit(1)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# CRITICAL: Proxy variables MUST be present (Injected by Validator)
try:
    API_KEY = os.environ["API_KEY"]
    API_BASE_URL = os.environ["API_BASE_URL"]
except KeyError as e:
    # Use SystemExit to ensure a clean non-zero exit for the validator
    raise SystemExit(f"Error: Missing required environment variable {e}. Validator proxy vars required.")

# OPTIONAL: Model Name defaults to a sensible value if not injected
# (Validator does not always inject this, unlike API_KEY)
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# Task configuration
TASK_IDS = [1, 2, 3]
TASK_NAMES = {
    1: "basic-cleanup",
    2: "type-fixing",
    3: "schema-inference",
}
BENCHMARK_NAME = "data-cleaning-env"
MAX_STEPS = 30  # Safety limit

# ==============================================================================
# 2. LOGGING FUNCTIONS (Strictly from Sample Script)
# ==============================================================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ==============================================================================
# 3. AGENT HELPERS
# ==============================================================================

SYSTEM_PROMPT = """You are an expert Data Cleaning Agent.
Your goal is to transform a dirty DataFrame into a clean, analysis-ready dataset.

INSTRUCTIONS:
1. ANALYZE: Look at the Column Names and Dtypes.
2. STANDARDIZE: Rename columns to snake_case (e.g., "First Name" -> "first_name").
3. FIX TYPES: Convert numeric strings to numbers, date strings to datetime objects.
4. HANDLE NULLS: Fill missing values appropriately (mean for numbers, mode for strings).
5. TRIM: Remove whitespace from all string columns.

OUTPUT FORMAT:
Respond with EXACTLY ONE JSON object. No markdown. No code blocks.
Example: {"op": "rename_column", "params": {"from_col": "User ID", "to_col": "user_id"}}

Available Operations:
- rename_column: Standardize column names.
- cast_dtype: Fix data types (int64, float64, datetime64[ns], bool).
- fill_nulls: Fill missing data (strategies: mean, median, mode, value).
- drop_column: Remove useless columns.
- strip_whitespace: Clean string columns.

Current Observation:
"""

def _parse_action(text: str) -> Action:
    """Parses LLM response into an Action object."""
    if not text or not text.strip():
        return Action(op="strip_whitespace", params={"col": "all"})
    
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        return Action(op="strip_whitespace", params={"col": "all"})
        
    try:
        data = json.loads(match.group())
        if "operation" in data and "op" not in data:
            data["op"] = data.pop("operation")
        if "op" in data and "params" not in data:
            data = {"op": data["op"], "params": {k: v for k, v in data.items() if k != "op"}}
        return Action(**data)
    except Exception:
        return Action(op="strip_whitespace", params={"col": "all"})

def _format_observation(obs: Observation) -> str:
    """Formats observation into a string prompt for the LLM."""
    return (
        f"Step {obs.step_count + 1}/{MAX_STEPS}\n"
        f"Columns: {obs.columns}\n"
        f"Dtypes:  {obs.dtypes}\n"
        f"Nulls:   {obs.null_counts}\n"
        f"Rows Preview: {obs.sample_rows[:3]}\n"
        f"Choose ONE cleaning action. Raw JSON only."
    )

# ==============================================================================
# 4. MAIN EXECUTION LOGIC
# ==============================================================================

def run_episode(task_id: int):
    """
    Runs a single episode for the given task ID using the Validator's Proxy.
    """
    task_name = TASK_NAMES.get(task_id, f"task-{task_id}")
    
    # Initialize Client (Standard OpenAI SDK pointing to Proxy)
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    
    # Initialize Environment
    env = DataCleaningEnv(task_id=task_id, seed=42)
    
    # State tracking
    rewards: List[float] = []
    steps_taken = 0
    success = False
    final_score = 0.0
    
    log_start(task=task_name, env=BENCHMARK_NAME, model=MODEL_NAME)
    
    obs = None
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    try:
        # 1. Reset Environment (Can fail if Docker container isn't ready)
        obs = env.reset()
        
        while not obs.done and steps_taken < MAX_STEPS:
            # 2. Construct Prompt
            user_content = _format_observation(obs)
            messages.append({"role": "user", "content": user_content})
            
            # 3. Call LLM
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=150,
                    timeout=30,
                )
                raw_action_str = response.choices[0].message.content or ""
            except Exception as e:
                print(f"[DEBUG] API Error: {e}", flush=True)
                raw_action_str = ""
            
            # 4. Parse Action
            action = _parse_action(raw_action_str)
            messages.append({"role": "assistant", "content": raw_action_str or "{}"})
            
            # 5. Step Environment
            obs, reward_obj, done, info = env.step(action)
            
            # Extract float reward
            current_reward = float(reward_obj.total) if hasattr(reward_obj, 'total') else float(reward_obj)
            
            steps_taken += 1
            rewards.append(current_reward)
            
            # 6. Log Step
            action_str = action.model_dump_json() if hasattr(action, 'model_dump_json') else str(action)
            
            log_step(
                step=steps_taken,
                action=action_str,
                reward=current_reward,
                done=done,
                error=None
            )
            
            if done:
                break
        
        # 7. Calculate Final Score
        if obs and hasattr(env, 'df'):
            final_score = grade(task_id, env.df)
        success = final_score >= 0.5
        
    except Exception as e:
        # Catch any crash during reset, step, or grading
        print(f"[DEBUG] Episode crashed: {e}", flush=True)
        if not obs: 
            # If crash happened at reset, log a dummy step to satisfy parser
            log_step(step=0, action="init_error", reward=0.0, done=True, error=str(e))
        final_score = 0.0
        success = False
        
    finally:
        # Always Log End
        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards
        )
        if hasattr(env, 'close'):
            try:
                env.close()
            except:
                pass

# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    # Run all required tasks sequentially
    for tid in TASK_IDS:
        try:
            run_episode(tid)
        except Exception as e:
            print(f"CRITICAL FAILURE on Task {tid}: {e}", flush=True)
            # Ensure we print a formatted END line even on critical failure
            print(f"[END] success=false steps=0 score=0.000 rewards=0.00", flush=True)