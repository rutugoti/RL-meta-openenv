"""
env/curriculum.py
=================
Adaptive curriculum manager for DataCleaningEnv.

Tracks agent performance per task and:
  1. Unlocks harder tasks when agent is ready
  2. Samples weak seeds more frequently (focus on failures)

Usage in notebook reward_func:
    from env.curriculum import CurriculumManager
    curriculum = CurriculumManager()
    # inside reward_func, after scoring:
    curriculum.update(task_id, seed, score)
    active_tasks = curriculum.get_active_tasks()
    next_seed    = curriculum.get_next_seed(task_id)
"""
from __future__ import annotations
import random
import collections

# Thresholds to unlock next task
UNLOCK_THRESHOLDS = {
    1: 0.70,   # Task 1 avg > 0.70 unlocks Task 2
    2: 0.55,   # Task 2 avg > 0.55 unlocks Task 3
}

# Rolling window size per task
WINDOW = 20


class CurriculumManager:
    """
    Adaptive curriculum for DataCleaningEnv.

    Tracks rolling average reward per task and unlocks
    harder tasks as the agent improves. Also tracks
    which seeds the agent struggles with and samples
    them more frequently to focus training on failures.

    Design principle:
        Task 1 unlocked always.
        Task 2 unlocked when Task 1 rolling avg > 0.70.
        Task 3 unlocked when Task 2 rolling avg > 0.55.
        Once unlocked, a task stays unlocked.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._scores: dict[int, collections.deque] = {
            1: collections.deque(maxlen=WINDOW),
            2: collections.deque(maxlen=WINDOW),
            3: collections.deque(maxlen=WINDOW),
        }
        self._unlocked = {1: True, 2: False, 3: False}
        self._seed_scores: dict[int, dict[int, list]] = {
            1: {}, 2: {}, 3: {}
        }
        self.step = 0

    # ── Public API ──────────────────────────────────────────────────

    def update(self, task_id: int, seed: int, score: float) -> None:
        """Record one episode result. Call after every completion is scored."""
        self.step += 1
        if task_id in self._scores:
            self._scores[task_id].append(score)
        if task_id in self._seed_scores:
            self._seed_scores[task_id].setdefault(seed, []).append(score)
        self._maybe_unlock()

    def get_active_tasks(self) -> list[int]:
        """Return list of currently unlocked task IDs (always at least [1])."""
        return [t for t, unlocked in self._unlocked.items() if unlocked]

    def get_next_seed(self, task_id: int, max_seed: int = 999) -> int:
        """
        70% chance: sample a weak seed (agent scored < 0.50).
        30% chance: sample a completely new unseen seed.
        """
        weak = self._get_weak_seeds(task_id)
        if weak and random.random() < 0.70:
            return random.choice(weak)
        seen = set(self._seed_scores.get(task_id, {}).keys())
        for _ in range(20):
            candidate = random.randint(0, max_seed)
            if candidate not in seen:
                return candidate
        return random.randint(0, max_seed)

    def avg_score(self, task_id: int) -> float:
        """Rolling average score for a task. 0.0 if no data yet."""
        scores = self._scores.get(task_id, [])
        return sum(scores) / len(scores) if scores else 0.0

    def status(self) -> str:
        """One-line status string for logging."""
        avgs = {t: self.avg_score(t) for t in [1, 2, 3]}
        return (
            f"Step={self.step} | Active={self.get_active_tasks()} | "
            f"T1={avgs[1]:.3f} T2={avgs[2]:.3f} T3={avgs[3]:.3f} | "
            f"Unlocked: T2={'yes' if self._unlocked[2] else 'no'} "
            f"T3={'yes' if self._unlocked[3] else 'no'}"
        )

    # ── Internal ────────────────────────────────────────────────────

    def _maybe_unlock(self) -> None:
        for task_id, threshold in UNLOCK_THRESHOLDS.items():
            next_task = task_id + 1
            if (not self._unlocked.get(next_task, True)
                    and self.avg_score(task_id) >= threshold):
                self._unlocked[next_task] = True
                print(
                    f"\n*** CURRICULUM: Task {next_task} UNLOCKED "
                    f"(Task {task_id} avg={self.avg_score(task_id):.3f} "
                    f">= {threshold}) ***\n"
                )

    def _get_weak_seeds(self, task_id: int) -> list[int]:
        """Seeds where agent averaged below 0.50."""
        weak = []
        for seed, scores in self._seed_scores.get(task_id, {}).items():
            if scores and (sum(scores) / len(scores)) < 0.50:
                weak.append(seed)
        return weak
