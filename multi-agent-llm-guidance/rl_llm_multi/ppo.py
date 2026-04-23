"""Shared-policy PPO with centralized critic and per-agent LLM advice fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from configs import (
    ACTION_BINS,
    EPS_CLIP,
    GAMMA,
    GAE_LAMBDA,
    K_EPOCHS,
    LAMBDA_ALIGN,
    LR_ACTOR,
    LR_CRITIC,
    MINIBATCH_SIZE,
    NORMALIZE_REWARD,
    USE_ALIGNMENT,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ActionRecord:
    local_obs: np.ndarray
    global_obs: np.ndarray
    action: int
    logprob: float
    value: float
    advice_embed: np.ndarray
    advice_active: float
    target_bin: int


class RunningMeanStd:
    """Track running mean and variance for observation normalization."""

    def __init__(self, shape: Tuple[int, ...], eps: float = 1e-4) -> None:
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = eps

    @torch.no_grad()
    def update(self, x_val: torch.Tensor) -> None:
        if x_val.dim() == 1:
            x_val = x_val.unsqueeze(0)
        batch_mean = x_val.mean(dim=0)
        batch_var = x_val.var(dim=0, unbiased=False)
        batch_count = x_val.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        new_var = m_2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x_val: torch.Tensor) -> torch.Tensor:
        return (x_val - self.mean) / torch.sqrt(self.var + 1e-8)


class RolloutBuffer:
    def __init__(self) -> None:
        self.local_obs: List[torch.Tensor] = []
        self.global_obs: List[torch.Tensor] = []
        self.next_global_obs: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.advice_embeds: List[torch.Tensor] = []
        self.advice_active: List[torch.Tensor] = []
        self.target_bins: List[int] = []

    def clear(self) -> None:
        self.__init__()


class SharedActorCentralCritic(nn.Module):
    """Actor uses local observations; critic uses centralized global state."""

    def __init__(
        self,
        local_obs_dim: int,
        global_obs_dim: int,
        advice_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.local_encoder = nn.Sequential(
            nn.Linear(local_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.advice_proj = nn.Linear(advice_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.actor = nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1))
        self.global_encoder = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def fuse(
        self,
        local_obs: torch.Tensor,
        advice_embed: torch.Tensor,
        advice_active: torch.Tensor,
    ) -> torch.Tensor:
        hidden_local = self.local_encoder(local_obs)
        if advice_embed is None or advice_active is None or torch.all(advice_active <= 0.0):
            return hidden_local
        hidden_advice = torch.tanh(self.advice_proj(advice_embed))
        tokens = torch.stack([hidden_local, hidden_advice], dim=1)
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return attn_out.mean(dim=1)

    def value(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.global_encoder(global_obs))


class SharedActorCentralCriticPPO:
    """Shared-policy PPO with optional LLM alignment for active advice states."""

    def __init__(
        self,
        local_obs_dim: int,
        global_obs_dim: int,
        *,
        advice_dim: int = 7,
        action_dim: int = len(ACTION_BINS),
        lr_actor: float = LR_ACTOR,
        lr_critic: float = LR_CRITIC,
        gamma: float = GAMMA,
        k_epochs: int = K_EPOCHS,
        eps_clip: float = EPS_CLIP,
        mb_size: int = MINIBATCH_SIZE,
        gae_lambda: float = GAE_LAMBDA,
        normalize_reward: bool = NORMALIZE_REWARD,
        use_alignment: bool = USE_ALIGNMENT,
        lambda_align: float = LAMBDA_ALIGN,
    ) -> None:
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.mb_size = mb_size
        self.gae_lambda = gae_lambda
        self.normalize_reward = normalize_reward
        self.use_alignment = use_alignment
        self.lambda_align = lambda_align

        self.buffer = RolloutBuffer()
        self.local_rms = RunningMeanStd((local_obs_dim,))
        self.global_rms = RunningMeanStd((global_obs_dim,))
        self.policy = SharedActorCentralCritic(local_obs_dim, global_obs_dim, advice_dim, action_dim).to(device)
        self.policy_old = SharedActorCentralCritic(local_obs_dim, global_obs_dim, advice_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.local_encoder.parameters(), "lr": lr_actor},
                {"params": self.policy.advice_proj.parameters(), "lr": lr_actor},
                {"params": self.policy.attn.parameters(), "lr": lr_actor},
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.global_encoder.parameters(), "lr": lr_critic},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )
        self.mse_loss = nn.MSELoss()

    def _normalize_local(self, obs: np.ndarray, *, train: bool) -> torch.Tensor:
        tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        if train:
            self.local_rms.update(tensor)
        return self.local_rms.normalize(tensor)

    def _normalize_global(self, obs: np.ndarray, *, train: bool) -> torch.Tensor:
        tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        if train:
            self.global_rms.update(tensor)
        return self.global_rms.normalize(tensor)

    def clear_advice_cache(self) -> None:
        return None

    def _prepare_advice_tensor(
        self,
        advice_embedding: Optional[np.ndarray],
        advice_active: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if advice_embedding is None:
            embed = torch.zeros(7, dtype=torch.float32, device=device)
        else:
            embed = torch.tensor(advice_embedding, dtype=torch.float32, device=device)
        active = torch.tensor([float(advice_active)], dtype=torch.float32, device=device)
        return embed, active

    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        global_state: np.ndarray,
        *,
        advice_embeddings: Optional[Dict[str, np.ndarray]] = None,
        advice_active_masks: Optional[Dict[str, float]] = None,
        target_bins: Optional[Dict[str, int]] = None,
        deterministic: bool = False,
        train: bool = True,
    ) -> Tuple[Dict[str, int], Dict[str, ActionRecord]]:
        actions: Dict[str, int] = {}
        records: Dict[str, ActionRecord] = {}
        global_tensor = self._normalize_global(global_state, train=train)

        for agent_id, local_obs in obs_dict.items():
            local_tensor = self._normalize_local(local_obs, train=train)
            advice_embed_np = None if advice_embeddings is None else advice_embeddings.get(agent_id)
            advice_active = 0.0 if advice_active_masks is None else float(advice_active_masks.get(agent_id, 0.0))
            advice_tensor, active_tensor = self._prepare_advice_tensor(advice_embed_np, advice_active)

            with torch.no_grad():
                fused = self.policy_old.fuse(
                    local_tensor.unsqueeze(0),
                    advice_tensor.unsqueeze(0),
                    active_tensor,
                )
                probs = self.policy_old.actor(fused)
                dist = Categorical(probs)
                if deterministic:
                    action = torch.argmax(probs, dim=-1)
                else:
                    action = dist.sample()
                logprob = dist.log_prob(action)
                value = self.policy_old.value(global_tensor.unsqueeze(0)).squeeze(-1)

            action_idx = int(action.item())
            actions[agent_id] = action_idx
            records[agent_id] = ActionRecord(
                local_obs=np.array(local_tensor.detach().cpu(), dtype=np.float32),
                global_obs=np.array(global_tensor.detach().cpu(), dtype=np.float32),
                action=action_idx,
                logprob=float(logprob.item()),
                value=float(value.item()),
                advice_embed=np.zeros(7, dtype=np.float32)
                if advice_embed_np is None
                else np.array(advice_embed_np, dtype=np.float32),
                advice_active=advice_active,
                target_bin=-1 if target_bins is None else int(target_bins.get(agent_id, -1)),
            )
        return actions, records

    def greedy_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        global_state: np.ndarray,
        *,
        advice_embeddings: Optional[Dict[str, np.ndarray]] = None,
        advice_active_masks: Optional[Dict[str, float]] = None,
    ) -> Dict[str, int]:
        actions, _ = self.select_actions(
            obs_dict,
            global_state,
            advice_embeddings=advice_embeddings,
            advice_active_masks=advice_active_masks,
            deterministic=True,
            train=False,
        )
        return actions

    def record_transition(
        self,
        record: ActionRecord,
        *,
        reward: float,
        done: bool,
        next_global_state: np.ndarray,
    ) -> None:
        self.buffer.local_obs.append(torch.tensor(record.local_obs, dtype=torch.float32, device=device))
        self.buffer.global_obs.append(torch.tensor(record.global_obs, dtype=torch.float32, device=device))
        self.buffer.next_global_obs.append(
            self._normalize_global(next_global_state, train=False).detach()
        )
        self.buffer.actions.append(torch.tensor(record.action, dtype=torch.long, device=device))
        self.buffer.logprobs.append(torch.tensor(record.logprob, dtype=torch.float32, device=device))
        self.buffer.values.append(torch.tensor(record.value, dtype=torch.float32, device=device))
        self.buffer.rewards.append(float(reward))
        self.buffer.dones.append(float(done))
        self.buffer.advice_embeds.append(torch.tensor(record.advice_embed, dtype=torch.float32, device=device))
        self.buffer.advice_active.append(torch.tensor(record.advice_active, dtype=torch.float32, device=device))
        self.buffer.target_bins.append(int(record.target_bin))

    def update(self) -> None:
        if not self.buffer.local_obs:
            return

        local_obs = torch.stack(self.buffer.local_obs, dim=0)
        global_obs = torch.stack(self.buffer.global_obs, dim=0)
        next_global_obs = torch.stack(self.buffer.next_global_obs, dim=0)
        actions = torch.stack(self.buffer.actions, dim=0)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32, device=device)
        advice_embeds = torch.stack(self.buffer.advice_embeds, dim=0)
        advice_active = torch.stack(self.buffer.advice_active, dim=0)
        target_bins = torch.tensor(self.buffer.target_bins, dtype=torch.long, device=device)

        if self.normalize_reward:
            rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)

        with torch.no_grad():
            next_values = self.policy.value(next_global_obs).squeeze(-1)
        values = torch.stack(self.buffer.values, dim=0)
        targets = rewards + self.gamma * next_values * (1.0 - dones)
        advantages = targets - values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = local_obs.size(0)
        minibatch_size = min(self.mb_size, batch_size)
        for _ in range(self.k_epochs):
            permutation = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                idx = permutation[start : start + minibatch_size]
                mb_local = local_obs[idx]
                mb_global = global_obs[idx]
                mb_next = next_global_obs[idx]
                mb_actions = actions[idx]
                mb_old_logprobs = old_logprobs[idx]
                mb_targets = targets[idx]
                mb_advantages = advantages[idx]
                mb_advice = advice_embeds[idx]
                mb_active = advice_active[idx]
                mb_target_bins = target_bins[idx]

                fused = self.policy.fuse(mb_local, mb_advice, mb_active)
                probs = self.policy.actor(fused)
                dist = Categorical(probs)
                logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                state_values = self.policy.value(mb_global).squeeze(-1)

                ratios = torch.exp(logprobs - mb_old_logprobs.detach())
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.mse_loss(state_values, mb_targets.detach())
                align_loss = torch.zeros((), dtype=torch.float32, device=device)

                if self.use_alignment:
                    valid = (mb_active > 0.5) & (mb_target_bins >= 0)
                    if valid.any():
                        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                        chosen = torch.log(probs[valid_idx, mb_target_bins[valid_idx]] + 1e-8)
                        align_loss = -chosen.mean() * self.lambda_align

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy + align_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, path: str) -> None:
        torch.save(self.policy_old.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=device)
        self.policy.load_state_dict(state_dict)
        self.policy_old.load_state_dict(state_dict)

