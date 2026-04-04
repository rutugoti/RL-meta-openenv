"""
DataCleaningEnv Python Client
==============================
A simple HTTP client for interacting with the DataCleaningEnv server.
Use this to connect to the local server or the HF Space.

Usage:
    from client import DataCleaningClient
    client = DataCleaningClient("https://ruuuuq-data-cleaning-env.hf.space")
    obs = client.reset(task_id=1)
    result = client.step({"op": "strip_whitespace", "params": {"col": "all"}})
"""
from __future__ import annotations
import requests


class DataCleaningClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()

    def reset(self, task_id: int = 1, seed: int = 42) -> dict:
        """Start a new episode. Returns initial observation."""
        r = self.session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: dict, task_id: int = 1, seed: int = 42) -> dict:
        """Submit one action. Returns observation, reward, done, info."""
        r = self.session.post(
            f"{self.base_url}/step",
            json={"action": action, "task_id": task_id, "seed": seed},
        )
        r.raise_for_status()
        return r.json()

    def state(self, task_id: int = 1, seed: int = 42) -> dict:
        """Get current environment state."""
        r = self.session.get(
            f"{self.base_url}/state",
            params={"task_id": task_id, "seed": seed},
        )
        r.raise_for_status()
        return r.json()

    def tasks(self) -> dict:
        """List all tasks and action schema."""
        r = self.session.get(f"{self.base_url}/tasks")
        r.raise_for_status()
        return r.json()

    def grade(self, task_id: int, final_state: dict) -> float:
        """Grade a completed episode."""
        r = self.session.post(
            f"{self.base_url}/grader",
            json={"task_id": task_id, "final_state": final_state},
        )
        r.raise_for_status()
        return r.json()["score"]

    def baseline(self) -> list:
        """Trigger baseline run. Returns scores for all 3 tasks."""
        r = self.session.post(f"{self.base_url}/baseline")
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    import json
    client = DataCleaningClient()
    print("Connecting to local server...")
    obs = client.reset(task_id=1, seed=42)
    print(f"Reset OK — columns: {obs['columns']}")
    result = client.step(
        {"op": "strip_whitespace", "params": {"col": "all"}})
    print(f"Step OK — score: {result['reward']['total']:.4f}")
    print(f"Tasks: {json.dumps(client.tasks()['tasks'], indent=2)}")