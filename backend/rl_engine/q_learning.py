# backend/rl_engine/q_learning.py
"""
Tabular Q-Learning helper for discrete action spaces.

This module provides:
 - StateHasher: produces compact discrete state ids from arbitrary observations (via hashing/bucketing).
 - QTable: manages the Q-table storage, load/save, and basic ops.
 - QLearning: training helper that performs Q-learning updates given transitions.

Design notes:
 - Intended for simple, interpretable discrete-space experiments in the hybrid trainer.
 - Use StateHasher to convert continuous observations into discrete buckets if needed.
 - Persist Q-table as a JSON or numpy file for easy inspection.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Optional
import json
import os
import hashlib
from pathlib import Path

import numpy as np


@dataclass
class StateHasher:
    """
    Convert arbitrary observation vectors or dicts into a discrete state id (string).
    Supports:
      - direct hashing of JSON-serializable objects
      - optional discretization of numeric arrays via bin edges
    """
    bin_edges: Optional[np.ndarray] = None  # if provided, used to discretize numeric vectors

    def hash_state(self, obs: Any) -> str:
        """
        Return a stable string id for the given observation.
        If obs is list/tuple/np.ndarray of numbers and bin_edges provided, discretize first.
        """
        try:
            # handle numeric sequences
            if isinstance(obs, (list, tuple, np.ndarray)):
                arr = np.array(obs, dtype=float)
                if self.bin_edges is not None:
                    # digitize returns indices for bins
                    idxs = np.digitize(arr, bins=self.bin_edges).tolist()
                    key = json.dumps(idxs, separators=(",", ":"), sort_keys=True)
                else:
                    key = json.dumps(arr.tolist(), separators=(",", ":"), sort_keys=True)
            elif isinstance(obs, dict):
                # deterministic JSON dump
                key = json.dumps(obs, sort_keys=True, separators=(",", ":"))
            else:
                key = str(obs)
        except Exception:
            # fallback safe string
            key = repr(obs)

        # hash to keep keys compact
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        return h


@dataclass
class QTable:
    num_actions: int
    table: Dict[str, np.ndarray] = field(default_factory=dict)

    def ensure_state(self, state_id: str):
        if state_id not in self.table:
            self.table[state_id] = np.zeros(self.num_actions, dtype=float)

    def get_action_values(self, state_id: str) -> np.ndarray:
        self.ensure_state(state_id)
        return self.table[state_id]

    def get_best_action(self, state_id: str) -> int:
        vals = self.get_action_values(state_id)
        return int(np.argmax(vals))

    def update(self, state_id: str, action: int, delta: float):
        self.ensure_state(state_id)
        self.table[state_id][int(action)] += float(delta)

    def to_serializable(self) -> Dict[str, list]:
        return {s: v.tolist() for s, v in self.table.items()}

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # save as JSON for easy inspection
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_serializable(), f, indent=2)

    def load(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.table = {k: np.array(v, dtype=float) for k, v in raw.items()}


class QLearning:
    """
    Tabular Q-learning update logic.
    """

    def __init__(self, num_actions: int, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.2):
        self.num_actions = int(num_actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.qtable = QTable(self.num_actions)
        self.hasher = StateHasher()

    def select_action(self, obs: Any) -> Tuple[str, int]:
        """
        Hash observation -> state_id, then choose an action (epsilon-greedy).
        Returns (state_id, action_index).
        """
        state_id = self.hasher.hash_state(obs)
        self.qtable.ensure_state(state_id)
        if np.random.rand() < self.epsilon:
            return state_id, int(np.random.randint(0, self.num_actions))
        else:
            return state_id, int(self.qtable.get_best_action(state_id))

    def update(self, state_id: str, action: int, reward: float, next_obs: Any, done: bool = False):
        """
        Perform Q-learning update for a single transition.
        Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
        target = r + gamma * max_a' Q(s', a') if not done else r
        """
        self.qtable.ensure_state(state_id)
        q_sa = self.qtable.table[state_id][int(action)]

        next_state_id = self.hasher.hash_state(next_obs)
        self.qtable.ensure_state(next_state_id)
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.qtable.table[next_state_id]))

        td = target - q_sa
        self.qtable.table[state_id][int(action)] += self.alpha * td

    def save(self, path: str):
        self.qtable.save(path)

    def load(self, path: str):
        self.qtable.load(path)

    def get_policy(self) -> Dict[str, int]:
        """
        Return greedy policy mapping state_id -> best_action
        """
        return {s: int(np.argmax(v)) for s, v in self.qtable.table.items()}
