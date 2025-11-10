# backend/rl_engine/ppo.py
"""
PPO implementation (PyTorch) for discrete-action environments.

Provides:
 - ActorCritic: small MLP actor + critic
 - RolloutBuffer: collect (obs, actions, rewards, dones, logps) and compute advantages (GAE)
 - PPO: training loop utilities (update from buffer, save/load)

Usage:
    from ppo import PPO, ActorCritic

    agent = PPO(obs_dim=8, num_actions=5)
    # collect transitions via environment or custom evaluator, then call agent.update(buffer)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_sizes: Tuple[int, ...] = (128, 64)):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        # actor
        actor_layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            actor_layers.append(nn.Linear(in_dim, h))
            actor_layers.append(nn.ReLU())
            in_dim = h
        actor_layers.append(nn.Linear(in_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # critic
        critic_layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            critic_layers.append(nn.Linear(in_dim, h))
            critic_layers.append(nn.ReLU())
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value

    def act(self, obs: np.ndarray) -> Tuple[int, float]:
        """Return sampled action and log prob for a single observation (numpy)."""
        self.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, _ = self.forward(obs_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)
            return int(action.item()), float(logp.item())


@dataclass
class RolloutBuffer:
    obs_buf: List[np.ndarray] = field(default_factory=list)
    act_buf: List[int] = field(default_factory=list)
    rew_buf: List[float] = field(default_factory=list)
    done_buf: List[bool] = field(default_factory=list)
    logp_buf: List[float] = field(default_factory=list)
    val_buf: List[float] = field(default_factory=list)

    def store(self, obs: np.ndarray, act: int, rew: float, done: bool, logp: float, val: float):
        self.obs_buf.append(np.array(obs, dtype=np.float32))
        self.act_buf.append(int(act))
        self.rew_buf.append(float(rew))
        self.done_buf.append(bool(done))
        self.logp_buf.append(float(logp))
        self.val_buf.append(float(val))

    def clear(self):
        self.obs_buf.clear()
        self.act_buf.clear()
        self.rew_buf.clear()
        self.done_buf.clear()
        self.logp_buf.clear()
        self.val_buf.clear()

    def size(self) -> int:
        return len(self.obs_buf)

    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        """
        Compute GAE advantages and discounted returns.
        Returns:
            obs_arr, act_arr, logp_arr, adv_arr, ret_arr, val_arr
        """
        rewards = np.array(self.rew_buf, dtype=np.float32)
        values = np.array(self.val_buf + [last_value], dtype=np.float32)  # append last value
        dones = np.array(self.done_buf, dtype=np.bool_)

        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + values[:-1]

        obs_arr = np.vstack(self.obs_buf) if self.obs_buf else np.zeros((0,))
        act_arr = np.array(self.act_buf, dtype=np.int64)
        logp_arr = np.array(self.logp_buf, dtype=np.float32)
        val_arr = np.array(self.val_buf, dtype=np.float32)
        adv_arr = (adv - np.mean(adv)) / (np.std(adv) + 1e-8) if adv.size > 0 else adv
        ret_arr = returns
        return obs_arr, act_arr, logp_arr, adv_arr, ret_arr, val_arr


class PPO:
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: Tuple[int, ...] = (128, 64),
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: torch.device = DEVICE
    ):
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.device = device

        self.model = ActorCritic(self.obs_dim, self.num_actions, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """
        Sample action and return (action, logp, value)
        """
        self.model.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.model(obs_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)
            return int(action.item()), float(logp.item()), float(value.item())

    def update(self, buffer: RolloutBuffer, epochs: int = 4, batch_size: int = 64, gamma: float = 0.99, lam: float = 0.95):
        """
        Perform PPO update from buffer. Buffer should have stored transitions for one or more episodes.
        """
        if buffer.size() == 0:
            return {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0}

        # compute last_value = 0 (assuming episode terminated) as safe default
        last_value = 0.0
        obs_arr, act_arr, logp_arr, adv_arr, ret_arr, val_arr = buffer.compute_returns_and_advantages(last_value, gamma=gamma, lam=lam)

        N = obs_arr.shape[0]
        if N == 0:
            return {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0}

        # convert to tensors
        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(act_arr, dtype=torch.int64, device=self.device)
        adv_t = torch.tensor(adv_arr, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(ret_arr, dtype=torch.float32, device=self.device)
        logp_old_t = torch.tensor(logp_arr, dtype=torch.float32, device=self.device)

        losses = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0}
        idxs = np.arange(N)
        for _epoch in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, batch_size):
                mb_idx = idxs[start:start+batch_size]
                mb_obs = obs_t[mb_idx]
                mb_acts = act_t[mb_idx]
                mb_advs = adv_t[mb_idx]
                mb_rets = ret_t[mb_idx]
                mb_logp_old = logp_old_t[mb_idx]

                logits, values = self.model(mb_obs)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                logp = dist.log_prob(mb_acts)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - mb_logp_old)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values.squeeze(-1) - mb_rets) ** 2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses["loss_pi"] += float(policy_loss.item())
                losses["loss_v"] += float(value_loss.item())
                losses["entropy"] += float(entropy.item())

        # average losses
        ns = max(1, (N // batch_size) * epochs)
        losses = {k: v / ns for k, v in losses.items()}

        # clear buffer after update
        buffer.clear()

        return losses

    def save(self, path_prefix: str):
        p = Path(path_prefix)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(p.with_suffix(".pt")))

    def load(self, path_prefix: str):
        p = Path(path_prefix)
        state_path = str(p.with_suffix(".pt"))
        if Path(state_path).exists():
            self.model.load_state_dict(torch.load(state_path, map_location=self.device))

    def export_config(self) -> dict:
        return {"obs_dim": self.obs_dim, "num_actions": self.num_actions}
