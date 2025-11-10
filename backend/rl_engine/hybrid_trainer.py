# backend/rl_engine/hybrid_trainer.py
"""
Hybrid RL Trainer orchestrator.

Provides:
- QLearningAgent: Tabular Q-learning for discrete small action spaces.
- PPOAgent: A PyTorch-based PPO implementation for discrete action spaces.
- GeneticSearcher: Simple GA-style mutation over policy parameter dicts.
- HybridTrainer: Orchestrates evaluation (via an evaluator callable), policy updates,
  and checkpointing.

Usage:
  from hybrid_trainer import HybridTrainer, QLearningAgent, PPOAgent

  def evaluator(action_spec) -> dict:
      # run tests according to action_spec, return metrics: {"coverage": 42.0, "failures": 1, "duration": 3.2}
      ...

  trainer = HybridTrainer(evaluator=evaluator, job_id="job123", save_dir="models/policies")
  q_agent = QLearningAgent(num_actions=5)
  ppo_agent = PPOAgent(num_actions=5, obs_dim=8)
  trainer.register_agent("q", q_agent)
  trainer.register_agent("ppo", ppo_agent)
  trainer.run_iteration(num_evals=10)  # evaluate policies and update
"""

import os
import time
import json
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.config import settings
from ..test_executor.test_runner import execute_tests_for_job  # used by example evaluator
from ..utils.logger import get_logger

logger = get_logger(__name__)

# ----- Q-Learning Agent (tabular) -----
class QLearningAgent:
    def __init__(self, num_actions: int, learning_rate: float = 0.1, gamma: float = 0.99, epsilon: float = 0.2):
        self.num_actions = int(num_actions)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        # For a demo, state is simplified to a single discrete index; we use a default single-state Q
        # For real use, you would use a hashed state representation (e.g., discretized coverage).
        self.q_table = np.zeros((1, self.num_actions), dtype=np.float32)

    def select_action(self, state: Any = 0) -> int:
        # epsilon-greedy over available actions
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        state_idx = 0
        return int(np.argmax(self.q_table[state_idx]))

    def update(self, state: Any, action: int, reward: float, next_state: Any = 0, done: bool = False):
        s = 0
        ns = 0
        a = int(action)
        target = reward
        if not done:
            target = reward + self.gamma * np.max(self.q_table[ns])
        td = target - self.q_table[s, a]
        self.q_table[s, a] += self.lr * td

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path: str):
        self.q_table = np.load(path)

    def get_params(self) -> Dict:
        return {"q_table": self.q_table.tolist(), "lr": self.lr, "gamma": self.gamma, "epsilon": self.epsilon}

    def set_params(self, params: Dict):
        self.q_table = np.array(params.get("q_table", self.q_table.tolist()), dtype=np.float32)
        self.lr = params.get("lr", self.lr)
        self.gamma = params.get("gamma", self.gamma)
        self.epsilon = params.get("epsilon", self.epsilon)

# ----- PPO Agent (PyTorch) -----
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

class PPOAgent:
    """
    Discrete-action PPO agent with a small MLP actor-critic.
    Observations are user-supplied fixed-size vectors (obs_dim).
    """
    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes: Tuple[int, ...] = (128, 64),
                 lr: float = 3e-4, clip_epsilon: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01):
        if torch is None:
            raise RuntimeError("PyTorch is required for PPOAgent. Install torch.")
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # simple MLP actor-critic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = self._build_actor().to(self.device)
        self.critic = self._build_critic().to(self.device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)

    def _build_actor(self):
        layers = []
        input_dim = self.obs_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, self.num_actions))
        return nn.Sequential(*layers)

    def _build_critic(self):
        layers = []
        input_dim = self.obs_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)

    def select_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """
        Select discrete action and return (action_index, log_prob).
        obs: 1D numpy array of shape (obs_dim,)
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # shape [1, obs_dim]
        logits = self.actor(obs_t)  # shape [1, num_actions]
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logp = dist.log_prob(action).item()
        return int(action.item()), float(logp)

    def value(self, obs: np.ndarray) -> float:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        v = self.critic(obs_t)
        return float(v.item())

    def train_batch(self, obs_batch: np.ndarray, act_batch: np.ndarray, adv_batch: np.ndarray, ret_batch: np.ndarray,
                    logp_old_batch: np.ndarray, epochs: int = 4, batch_size: int = 64):
        """
        Perform PPO update given collected batches.
        obs_batch: (N, obs_dim)
        act_batch: (N,)
        adv_batch: (N,)
        ret_batch: (N,)
        logp_old_batch: (N,)
        """
        if torch is None:
            raise RuntimeError("PyTorch not available")

        N = obs_batch.shape[0]
        obs = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        acts = torch.tensor(act_batch, dtype=torch.int64, device=self.device)
        advs = torch.tensor(adv_batch, dtype=torch.float32, device=self.device)
        rets = torch.tensor(ret_batch, dtype=torch.float32, device=self.device)
        logp_old = torch.tensor(logp_old_batch, dtype=torch.float32, device=self.device)

        for _ in range(epochs):
            # simple minibatch iteration
            indices = np.arange(N)
            np.random.shuffle(indices)
            for start in range(0, N, batch_size):
                mb_idx = indices[start:start+batch_size]
                mb_obs = obs[mb_idx]
                mb_acts = acts[mb_idx]
                mb_advs = advs[mb_idx]
                mb_rets = rets[mb_idx]
                mb_logp_old = logp_old[mb_idx]

                logits = self.actor(mb_obs)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                mb_logp = dist.log_prob(mb_acts)
                entropy = dist.entropy().mean()

                ratio = torch.exp(mb_logp - mb_logp_old)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                value_preds = self.critic(mb_obs).squeeze(-1)
                value_loss = ((value_preds - mb_rets) ** 2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save(self, path_prefix: str):
        Path(path_prefix).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path_prefix}.actor.pt")
        torch.save(self.critic.state_dict(), f"{path_prefix}.critic.pt")

    def load(self, path_prefix: str):
        self.actor.load_state_dict(torch.load(f"{path_prefix}.actor.pt", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{path_prefix}.critic.pt", map_location=self.device))

    def get_params(self) -> Dict:
        # Not serializing whole model here; caller should use save/load for weights.
        return {"obs_dim": self.obs_dim, "num_actions": self.num_actions, "hidden_sizes": self.hidden_sizes}

# ----- Genetic Searcher -----
class GeneticSearcher:
    """
    Simple genetic-style mutation over parameter dicts or numeric arrays.
    Used to produce candidate modifications to policies or action parameters.
    """

    def __init__(self, mutation_rate: float = 0.1, mutation_scale: float = 0.2):
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.rng = np.random.RandomState(seed=int(time.time()))

    def mutate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate numeric parameters in-place (shallow pass). Params expected to be simple key->numeric.
        """
        out = dict(params)
        for k, v in params.items():
            if isinstance(v, (int, float)):
                if random.random() < self.mutation_rate:
                    delta = float(self.rng.normal(scale=self.mutation_scale * max(1.0, abs(v))))
                    out[k] = v + delta
        return out

    def crossover(self, parent_a: Dict[str, Any], parent_b: Dict[str, Any]) -> Dict[str, Any]:
        child = {}
        for k in set(parent_a.keys()).union(parent_b.keys()):
            child[k] = parent_a.get(k) if random.random() < 0.5 else parent_b.get(k)
        return child

# ----- HybridTrainer orchestration -----
@dataclass
class PolicyRecord:
    id: str
    agent_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridTrainer:
    """
    Orchestrates evaluation and training of hybrid agents.

    :param evaluator: Callable that accepts an 'action_spec' or policy id and returns a metrics dict.
                      Example return: {"coverage": 41.2, "failures": 2, "duration": 3.2}
    :param job_id: identifier used for saving artifacts
    :param save_dir: base dir for saving models and checkpoints
    """

    def __init__(self, evaluator: Callable[[Dict[str,Any], str], Dict[str, Any]], job_id: str, save_dir: Optional[str] = None):
        self.evaluator = evaluator
        self.job_id = job_id
        self.save_dir = Path(save_dir or (Path(settings.MODELS_DIR) / "policies"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.agents: Dict[str, Any] = {}  # name -> agent instance
        self.policy_pool: List[PolicyRecord] = []
        self.searcher = GeneticSearcher()
        self.metrics_log_path = self.save_dir / f"metrics_{job_id}.jsonl"

    def register_agent(self, name: str, agent: Any):
        self.agents[name] = agent

    def seed_policy_pool(self, base_actions: List[Dict[str, Any]]):
        """
        Seed the policy pool with initial action specifications.
        Each action is a small dict that the evaluator understands.
        Example: {"name":"random_payload", "payload_strategy":"random", "param_mutation":0.1}
        """
        for i, a in enumerate(base_actions):
            pid = f"policy_{len(self.policy_pool)+1}"
            pr = PolicyRecord(id=pid, agent_type="spec", params=a)
            self.policy_pool.append(pr)

    def evaluate_policy(self, policy: PolicyRecord) -> float:
        """
        Evaluate a policy via the evaluator callable. Computes a scalar reward:
        reward = coverage_percent - alpha * failures - beta * duration (simple)
        Adjust alpha/beta to tune desired behaviour.
        """
        try:
            metrics = self.evaluator(policy.params, self.job_id)
            coverage = float(metrics.get("coverage", 0.0))
            failures = float(metrics.get("failures", metrics.get("errors", 0)))
            duration = float(metrics.get("duration", 0.0))
            # reward design: prioritize coverage, penalize failures and long runs
            reward = coverage - 5.0 * failures - 0.1 * duration
            # update record
            policy.score = reward
            policy.metadata.update(metrics)
            # log metrics
            self._log_metrics(policy.id, metrics, reward)
            return reward
        except Exception as exc:
            logger.error(f"Evaluation failed for {policy.id}: {exc}")
            return -999.0

    def _log_metrics(self, policy_id: str, metrics: Dict[str, Any], reward: float):
        rec = {"timestamp": time.time(), "policy_id": policy_id, "metrics": metrics, "reward": reward}
        try:
            with open(self.metrics_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            logger.exception("Failed to write metrics log")

    def run_iteration(self, num_evals: int = 5):
        """
        Run an iteration: sample policies from pool, evaluate them, and perform updates:
          - apply Q-learning update if a Q agent is registered (for discrete chosen actions)
          - apply PPO training for collected trajectories (if PPO agent present)
          - run genetic mutations and add new candidates
        This function keeps things synchronous for simplicity (you can offload to workers).
        """
        # Ensure we have candidates
        if not self.policy_pool:
            raise RuntimeError("No policies in pool. Call seed_policy_pool first.")

        # Sample policies to evaluate
        candidates = random.choices(self.policy_pool, k=min(num_evals, len(self.policy_pool)))
        results = []
        for p in candidates:
            logger.info(f"Evaluating policy {p.id} ({p.params})")
            r = self.evaluate_policy(p)
            results.append((p, r))

        # Example Q-learning update: treat each policy index as discrete action
        if "q" in self.agents:
            q_agent: QLearningAgent = self.agents["q"]
            # state simplification: single-state; update based on reward
            for p, r in results:
                action_idx = int(p.id.split("_")[-1]) - 1  # simple mapping
                q_agent.update(state=0, action=action_idx, reward=r, next_state=0, done=True)
            # save Q-table
            q_agent.save(str(self.save_dir / f"q_agent_{self.job_id}.npy"))

        # Example PPO update: collect synthetic observation/action/logp batches from evaluation metadata
        if "ppo" in self.agents:
            ppo_agent: PPOAgent = self.agents["ppo"]
            # Build batches from results: this is an interface point — you should map your environment state to obs
            obs_batch = []
            act_batch = []
            adv_batch = []
            ret_batch = []
            logp_old_batch = []
            # For demonstration, create synthetic obs and logp; in a real setup you'd collect trajectories
            for p, r in results:
                # obs: simple vector [coverage, failures, duration, ...] normalized
                m = p.metadata
                obs = np.array([m.get("coverage", 0.0)/100.0, m.get("failures", 0)/10.0, m.get("duration", 0.0)/10.0] + [0.0]*(ppo_agent.obs_dim-3), dtype=np.float32)
                action = int(p.id.split("_")[-1]) % ppo_agent.num_actions
                # synthetic logp_old: small negative float
                logp_old = -np.log(max(1e-6, 1.0/ppo_agent.num_actions))
                # adv and ret: use reward as both advantage and return (toy)
                obs_batch.append(obs)
                act_batch.append(action)
                adv_batch.append(r)
                ret_batch.append(r)
                logp_old_batch.append(logp_old)

            if len(obs_batch) >= 2:
                obs_arr = np.vstack(obs_batch)
                act_arr = np.array(act_batch)
                adv_arr = np.array(adv_batch, dtype=np.float32)
                ret_arr = np.array(ret_batch, dtype=np.float32)
                logp_arr = np.array(logp_old_batch, dtype=np.float32)
                ppo_agent.train_batch(obs_arr, act_arr, adv_arr, ret_arr, logp_arr, epochs=4, batch_size=8)
                ppo_agent.save(str(self.save_dir / f"ppo_agent_{self.job_id}"))

        # Genetic exploration: mutate top policies to create new candidates
        # pick top-K by reward
        self.policy_pool.sort(key=lambda x: x.score, reverse=True)
        topk = self.policy_pool[:max(1, len(self.policy_pool)//4)]
        new_candidates = []
        for parent in topk:
            child_params = self.searcher.mutate_params(parent.params)
            pid = f"policy_{len(self.policy_pool)+len(new_candidates)+1}"
            pr = PolicyRecord(id=pid, agent_type="genetic", params=child_params)
            new_candidates.append(pr)
        # append new candidates to pool
        self.policy_pool.extend(new_candidates)
        logger.info(f"Added {len(new_candidates)} genetic candidates to pool. Pool size now {len(self.policy_pool)}")

        # Save pool metadata
        pool_path = self.save_dir / f"policy_pool_{self.job_id}.json"
        with open(pool_path, "w", encoding="utf-8") as f:
            json.dump([{"id": p.id, "score": p.score, "params": p.params, "metadata": p.metadata} for p in self.policy_pool], f, indent=2)

        return results

    # Convenience: a default evaluator that calls the test runner on a collection path or spec
    @staticmethod
    def default_evaluator(action_spec: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        """
        Default evaluator demonstrates how you might run tests to obtain metrics.
        Expects action_spec to provide keys like:
          - 'tests_dir' or 'collection_path' or 'run_target' which points to folder or collection.
          - or more complex spec that your test harness understands.
        Returns: {"coverage":float, "failures":int, "duration":float}
        """
        start = time.time()
        target = action_spec.get("tests_dir") or action_spec.get("collection_path") or action_spec.get("run_target")
        if not target:
            # if action_spec is itself a generator directive, you would create tests first. For now return low reward.
            return {"coverage": 0.0, "failures": 1, "duration": 0.0}

        try:
            summary = execute_tests_for_job(str(target), job_id)
            # try to interpret summary: Newman or pytest shapes
            failures = 0
            coverage = 0.0
            duration = float(summary.get("ran_at", 0))  # ran_at is timestamp string, not duration; we'll attempt alternatives
            # try Newman report reading
            if summary.get("runner") == "newman" and summary.get("report_path"):
                # parse newman report for assertions and failures
                try:
                    rp = Path(summary["report_path"])
                    if rp.exists():
                        data = json.loads(rp.read_text(encoding="utf-8"))
                        run = data.get("run", {})
                        stats = run.get("stats", {})
                        failures = stats.get("assertions", {}).get("failed", 0)
                        # coverage not provided by newman; keep 0
                except Exception:
                    pass
            elif summary.get("runner") == "pytest":
                # parse pytest log for failures
                logp = summary.get("log_path")
                if logp:
                    try:
                        txt = Path(logp).read_text(encoding="utf-8")
                        # naive failure count
                        failures = txt.count("FAILED")
                    except Exception:
                        pass
            # compute duration
            try:
                duration = float(summary.get("duration", 0.0))
            except Exception:
                duration = time.time() - start
            # reward defaults
            return {"coverage": coverage, "failures": failures, "duration": duration}
        except Exception as exc:
            logger.exception("Default evaluator failed")
            return {"coverage": 0.0, "failures": 1, "duration": time.time() - start}

