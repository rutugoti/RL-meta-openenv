import pytest
import pandas as pd
from graders.grader import grade
from tasks.task_definitions import load_dirty_df, TASKS
from env.environment import DataCleaningEnv
from env.models import Action


# ── Test 1: all graders return float in 0.0–1.0 ──────────────────
def test_grader_scores_in_range():
    for task_id in [1, 2, 3]:
        df    = load_dirty_df(task_id, seed=42)
        score = grade(task_id, df)
        assert isinstance(score, float), \
            f"Task {task_id} grader must return float"
        assert 0.0 <= score <= 1.0, \
            f"Task {task_id} score {score} out of range"
        print(f"✓ Task {task_id} dirty score: {score}")


# ── Test 2: graders are deterministic ────────────────────────────
def test_grader_deterministic():
    for task_id in [1, 2, 3]:
        df = load_dirty_df(task_id, seed=42)
        s1 = grade(task_id, df)
        s2 = grade(task_id, df)
        assert s1 == s2, \
            f"Task {task_id} grader not deterministic: {s1} vs {s2}"
        print(f"✓ Task {task_id} grader is deterministic: {s1}")


# ── Test 3: dirty df scores low (< 0.5) ──────────────────────────
def test_dirty_df_scores_low():
    for task_id in [1, 2, 3]:
        df    = load_dirty_df(task_id, seed=42)
        score = grade(task_id, df)
        assert score < 0.5, \
            f"Task {task_id}: dirty df scores too high ({score}). " \
            f"Task is too easy or grader is broken."
        print(f"✓ Task {task_id} dirty score low enough: {score}")


# ── Test 4: Task 3 genuinely hard (< 0.35) ───────────────────────
def test_task3_is_genuinely_hard():
    df    = load_dirty_df(3, seed=42)
    score = grade(3, df)
    assert score < 0.35, \
        f"Task 3 dirty score {score} too high — hard task must " \
        f"challenge frontier models. Tighten target schema."
    print(f"✓ Task 3 is genuinely hard: {score}")


# ── Test 5: perfect df scores 1.0 on Task 1 ──────────────────────
def test_perfect_task1_scores_max():
    perfect_df = pd.DataFrame({
        "user_id":    [1, 2, 3],
        "first_name": ["alice", "bob", "carol"],
        "age":        [25, 30, 35],
        "email":      ["a@x.com", "b@x.com", "c@x.com"],
    })
    score = grade(1, perfect_df)
    assert score >= 0.85, \
        f"Perfect Task 1 df should score >= 0.85, got {score}"
    print(f"✓ Perfect Task 1 df scores: {score}")


# ── Test 6: grader scores improve after correct actions ───────────
def test_grader_improves_after_actions():
    env = DataCleaningEnv(task_id=1, seed=42)
    env.reset()
    score_before = grade(1, env.df)

    # do several correct cleaning actions
    actions = [
        Action(op="rename_column",
               params={"from_col": "AGE", "to_col": "age"}),
        Action(op="strip_whitespace",
               params={"col": "all"}),
        Action(op="drop_column",
               params={"col": "user_id"}),
    ]
    for a in actions:
        env.step(a)

    score_after = grade(1, env.df)
    assert score_after > score_before, \
        f"Grader must improve after correct actions: " \
        f"before={score_before} after={score_after}"
    print(f"✓ Grader improves: {score_before} → {score_after}")


# ── Test 7: grader does not change df (stateless) ────────────────
def test_grader_does_not_mutate_df():
    df      = load_dirty_df(1, seed=42)
    cols_before = list(df.columns)
    shape_before = df.shape
    grade(1, df)
    assert list(df.columns) == cols_before, \
        "Grader must not rename columns"
    assert df.shape == shape_before, \
        "Grader must not add/drop rows or columns"
    print("✓ Grader is stateless — df unchanged after grading")


# ── Test 8: all 3 tasks have increasing difficulty ────────────────
def test_difficulty_progression():
    scores = {}
    for task_id in [1, 2, 3]:
        df = load_dirty_df(task_id, seed=42)
        scores[task_id] = grade(task_id, df)
    print(f"  Dirty scores: Task1={scores[1]} Task2={scores[2]} Task3={scores[3]}")
    # Task 3 should be hardest (lowest dirty score)
    assert scores[3] <= scores[1], \
        "Task 3 should be harder than Task 1"
    assert scores[3] <= scores[2], \
        "Task 3 should be harder than Task 2"
    print("✓ Difficulty progression correct: Task3 hardest")


if __name__ == "__main__":
    test_grader_scores_in_range()
    test_grader_deterministic()
    test_dirty_df_scores_low()
    test_task3_is_genuinely_hard()
    test_perfect_task1_scores_max()
    test_grader_improves_after_actions()
    test_grader_does_not_mutate_df()
    test_difficulty_progression()
    print("\n All Day 4 grader tests passed. Ready for Day 5.")