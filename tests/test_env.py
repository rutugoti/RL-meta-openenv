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


from env.models import Action

# ── Test 6: step() returns correct 4-tuple ───────────────────────
def test_step_returns_correct_tuple():
    env = DataCleaningEnv(task_id=1, seed=42)
    env.reset()
    action = Action(op="strip_whitespace", params={"col": "all"})
    result = env.step(action)
    assert len(result) == 4, "step() must return 4 values"
    obs, reward, done, info = result
    assert isinstance(obs, Observation)
    assert isinstance(reward.total, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    print("✓ step() returns correct (obs, reward, done, info)")

# ── Test 7: reward varies — partial progress works ───────────────
def test_reward_varies_with_actions():
    env = DataCleaningEnv(task_id=1, seed=42)
    env.reset()
    # invalid action → low reward
    _, r_invalid, _, _ = env.step(
        Action(op="rename_column",
               params={"from_col": "nonexistent", "to_col": "x"}))
    # reset and do a valid rename
    env.reset()
    _, r_valid, _, _ = env.step(
        Action(op="rename_column",
               params={"from_col": "AGE", "to_col": "age"}))
    assert r_valid.total != r_invalid.total, \
        "DISQUALIFICATION RISK: reward same for valid and invalid actions"
    print(f"✓ Reward varies: invalid={r_invalid.total} valid={r_valid.total}")

# ── Test 8: determinism — same action same state same reward ──────
def test_reward_is_deterministic():
    action = Action(op="rename_column",
                    params={"from_col": "AGE", "to_col": "age"})
    env1 = DataCleaningEnv(task_id=1, seed=42)
    env1.reset()
    _, r1, _, _ = env1.step(action)

    env2 = DataCleaningEnv(task_id=1, seed=42)
    env2.reset()
    _, r2, _, _ = env2.step(action)

    assert r1.total == r2.total, \
        f"DISQUALIFICATION RISK: same action gives different rewards: {r1.total} vs {r2.total}"
    print(f"✓ Reward deterministic: both = {r1.total}")

# ── Test 9: invalid action penalised but not fatal ────────────────
def test_invalid_action_penalised():
    env = DataCleaningEnv(task_id=1, seed=42)
    env.reset()
    _, reward, _, info = env.step(
        Action(op="drop_column", params={"col": "doesnotexist"}))
    assert reward.step_penalty < 0, "Invalid action must apply penalty"
    assert not info["action_valid"], "info must flag action as invalid"
    print(f"✓ Invalid action penalised: {reward.step_penalty}")

# ── Test 10: exploit check — doing nothing stays low ──────────────
def test_exploit_doing_nothing():
    env = DataCleaningEnv(task_id=1, seed=42)
    env.reset()
    last_reward = None
    for _ in range(5):
        _, reward, _, _ = env.step(
            Action(op="strip_whitespace",
                   params={"col": "nonexistent_col"}))
        last_reward = reward
    assert last_reward.total < 0.6, \
        "EXPLOIT: spamming invalid actions scores too high"
    print(f"✓ Exploit check passed: spam score = {last_reward.total}")

# ── Test 11: episode ends at MAX_STEPS ───────────────────────────
def test_episode_ends_at_max_steps():
    env = DataCleaningEnv(task_id=1, seed=42)
    env.reset()
    done = False
    for _ in range(35):
        action = Action(op="strip_whitespace", params={"col":"all"})
        _, _, done, _ = env.step(action)
        if done:
            break
    assert done, "Episode must end at MAX_STEPS"
    assert env.step_count == env.MAX_STEPS
    print(f"✓ Episode ends correctly at step {env.step_count}")

if __name__ == "__main__":
    # run all tests including Day 2 tests
    test_reset_returns_observation()
    test_reset_deterministic()
    test_reset_deterministic_across_instances()
    test_different_seeds_give_different_states()
    test_state_returns_dict()
    test_step_returns_correct_tuple()
    test_reward_varies_with_actions()
    test_reward_is_deterministic()
    test_invalid_action_penalised()
    test_exploit_doing_nothing()
    test_episode_ends_at_max_steps()
    print("\n All Day 3 tests passed. Ready for Day 4.")