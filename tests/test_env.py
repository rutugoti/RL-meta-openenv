import pytest
from env.environment import DataCleaningEnv
from env.models import Observation

# ── Test 1: reset() returns correct Observation type ─────────────
def test_reset_returns_observation():
    env = DataCleaningEnv(task_id=1, seed=42)
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert isinstance(obs.columns, list)
    assert isinstance(obs.dtypes, dict)
    assert isinstance(obs.null_counts, dict)
    assert obs.step_count == 0
    assert obs.task_id == 1
    assert obs.done == False
    print("✓ reset() returns valid Observation")

# ── Test 2: reset() is deterministic (same seed = same result) ───
def test_reset_deterministic():
    env = DataCleaningEnv(task_id=1, seed=42)
    obs1 = env.reset()
    obs2 = env.reset()   # call again on same instance
    assert obs1.columns == obs2.columns
    assert obs1.dtypes   == obs2.dtypes
    print("✓ reset() is deterministic on same instance")

# ── Test 3: deterministic across different instances ─────────────
def test_reset_deterministic_across_instances():
    obs1 = DataCleaningEnv(task_id=1, seed=42).reset()
    obs2 = DataCleaningEnv(task_id=1, seed=42).reset()
    assert obs1.columns == obs2.columns
    assert obs1.dtypes   == obs2.dtypes
    print("✓ reset() is deterministic across new instances")

# ── Test 4: different seeds give different starting states ────────
def test_different_seeds_give_different_states():
    obs_42 = DataCleaningEnv(task_id=1, seed=42).reset()
    obs_99 = DataCleaningEnv(task_id=1, seed=99).reset()
    # null_counts or sample_rows should differ
    assert (obs_42.null_counts != obs_99.null_counts or
            obs_42.sample_rows != obs_99.sample_rows)
    print("✓ different seeds produce different starting states")

# ── Test 5: state() returns correct dict ─────────────────────────
def test_state_returns_dict():
    env = DataCleaningEnv(task_id=2, seed=42)
    env.reset()
    s = env.state()
    assert isinstance(s, dict)
    assert "columns" in s
    assert "dtypes" in s
    assert "step_count" in s
    assert s["step_count"] == 0
    assert s["task_id"] == 2
    print("✓ state() returns correct dict with all fields")

if __name__ == "__main__":
    test_reset_returns_observation()
    test_reset_deterministic()
    test_reset_deterministic_across_instances()
    test_different_seeds_give_different_states()
    test_state_returns_dict()
    print("\n All Day 2 tests passed. Ready for Day 3.")