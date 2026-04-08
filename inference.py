#!/usr/bin/env python3
"""
inference.py — OpenEnv Phase 2 entry point.
Corrected to strictly follow sample script format and logging requirements.
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
from env.environment import DataCleaningEnv
from env.models      import Action, Observation
from graders.grader  import grade

# ==============================================================================
# 1. STRICT CONFIGURATION
# ==============================================================================
# CRITICAL: No fallback values. This forces the script to use the Proxy variables
# injected by the Phase 2 validator. Failure to find them will raise a clear error.
try:
    API_KEY = os.environ["API_KEY"]
    API_BASE_URL = os.environ["API_BASE_URL"]
    MODEL_NAME = os.environ["MODEL_NAME"]
except KeyError as e:
    raise SystemExit(f"Error: Missing required environment variable {e}. "
                     "Cannot proceed without Proxy configuration.")

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
    # Formatting: 2 decimal places for reward, lowercase bools
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

def _parse_action(text: str) -> Action:
    """Parses LLM response into an Action object."""
    if not text or not text.strip():
        # Default fallback action if empty
        return Action(op="strip_whitespace", params={"col": "all"})
    
    # Clean markdown code blocks if present
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    
    # Find JSON object
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not match:
        return Action(op="strip_whitespace", params={"col": "all"})
        
    try:
        data = json.loads(match.group())
        # Normalize keys
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
    
    # Log Start
    log_start(task=task_name, env=BENCHMARK_NAME, model=MODEL_NAME)
    
    obs = env.reset()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    try:
        while not obs.done and steps_taken < MAX_STEPS:
            # 1. Construct Prompt
            user_content = _format_observation(obs)
            messages.append({"role": "user", "content": user_content})
            
            # 2. Call LLM
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
                # Log error and use default action
                print(f"[DEBUG] API Error: {e}", flush=True)
                raw_action_str = ""
            
            # 3. Parse Action
            action = _parse_action(raw_action_str)
            
            # Log assistant response for history context
            messages.append({"role": "assistant", "content": raw_action_str or "{}"})
            
            # 4. Step Environment
            # Note: Your env returns (obs, reward, done, info). 
            # We assume reward is an object with .total based on your previous code.
            obs, reward_obj, done, info = env.step(action)
            
            # Extract float reward
            current_reward = float(reward_obj.total) if hasattr(reward_obj, 'total') else float(reward_obj)
            
            steps_taken += 1
            rewards.append(current_reward)
            
            # 5. Log Step
            # Convert action to string for logging
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
        
        # 6. Calculate Final Score
        final_score = grade(task_id, env.df)
        success = final_score >= 0.5  # Example threshold logic
        
    except Exception as e:
        # If anything crashes, we still need to print [END]
        print(f"[DEBUG] Episode crashed: {e}", flush=True)
        # Log the crash step if possible, or just finish
        log_step(
            step=steps_taken + 1,
            action="error",
            reward=0.0,
            done=True,
            error=str(e)
        )
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
        # Ensure env is closed if it has a close method
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
            # Ensure we print a formatted END line even on critical failure to keep parser happy
            print(f"[END] success=false steps=0 score=0.000 rewards=0.00", flush=True)