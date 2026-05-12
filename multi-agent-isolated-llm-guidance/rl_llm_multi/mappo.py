"""MAPPO-style PPO for the multi-agent sector simulator."""

from __future__ import annotations

import os
import random
import time
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from configs import ACTION_BINS, MAX_AGENTS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SCHEMA_VERSION = 4
torch.manual_seed(0)
if device.type == "cuda":
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True


def _activation_module(name: str) -> nn.Module:
    key = str(name).lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported MAPPO activation: {name}")


def _hidden_gain(name: str) -> float:
    key = str(name).lower()
    if key == "tanh":
        return float(nn.init.calculate_gain("tanh"))
    if key == "relu":
        return float(nn.init.calculate_gain("relu"))
    return float(np.sqrt(2.0))


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    *,
    activation: str,
    use_layer_norm: bool,
    final_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    dims = [int(input_dim), *[int(dim) for dim in hidden_dims]]
    layers: List[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(_activation_module(activation))
    layers.append(nn.Linear(dims[-1], int(output_dim)))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


def _orthogonal_init(module: nn.Module, *, activation: str, output_gain: float) -> None:
    hidden_gain = _hidden_gain(activation)
    linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    for idx, layer in enumerate(linear_layers):
        is_output = idx == len(linear_layers) - 1
        gain = float(output_gain) if is_output else hidden_gain
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)


def _atomic_torch_save(obj, path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def agent_id_to_index(agent_id: str, max_agents: int = MAX_AGENTS) -> int:
    label = str(agent_id).strip().upper()
    if not label.startswith("A") or not label[1:].isdigit():
        raise ValueError(f"Unsupported agent id: {agent_id}")
    idx = int(label[1:]) - 1
    if idx < 0 or idx >= int(max_agents):
        raise ValueError(f"Agent id {agent_id} is outside max_agents={max_agents}")
    return idx


class RunningMeanStd:
    """Track running mean and variance for global-state normalization."""

    def __init__(self, shape: int | Sequence[int], eps: float = 1e-4) -> None:
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = float(eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = float(total_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

    def state_dict(self) -> Dict[str, object]:
        return {
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.mean = torch.as_tensor(state["mean"], dtype=torch.float32, device=device)
        self.var = torch.as_tensor(state["var"], dtype=torch.float32, device=device)
        self.count = float(state["count"])


@dataclass
class ActionRecord:
    episode_id: int
    agent_id: str
    agent_index: int
    local_obs: torch.Tensor
    global_state: torch.Tensor
    action: torch.Tensor
    logprob: torch.Tensor
    value: torch.Tensor


class RolloutBuffer:
    """Rollout storage for all active-agent decisions."""

    def __init__(self) -> None:
        self.local_observations: List[torch.Tensor] = []
        self.global_states: List[torch.Tensor] = []
        self.agent_indices: List[int] = []
        self.actions: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.terminals: List[bool] = []
        self.episode_ids: List[int] = []
        self.agent_ids: List[str] = []

    def add(self, record: ActionRecord, *, reward: float, terminal: bool) -> None:
        self.local_observations.append(record.local_obs.detach())
        self.global_states.append(record.global_state.detach())
        self.agent_indices.append(int(record.agent_index))
        self.actions.append(record.action.detach())
        self.logprobs.append(record.logprob.detach())
        self.values.append(record.value.detach())
        self.rewards.append(float(reward))
        self.terminals.append(bool(terminal))
        self.episode_ids.append(int(record.episode_id))
        self.agent_ids.append(str(record.agent_id))

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.rewards)

    def add_terminal_bonus(self, episode_id: int, reward: float | Mapping[str, float]) -> None:
        if isinstance(reward, MappingABC):
            rewards_by_agent = {
                str(agent_id): float(value)
                for agent_id, value in reward.items()
                if abs(float(value)) > 1e-12
            }
            if not rewards_by_agent:
                return
        else:
            scalar_reward = float(reward)
            if abs(scalar_reward) <= 1e-12:
                return
            rewards_by_agent = None

        last_index_by_agent: Dict[str, int] = {}
        for idx, (stored_episode_id, agent_id) in enumerate(zip(self.episode_ids, self.agent_ids)):
            if int(stored_episode_id) == int(episode_id):
                last_index_by_agent[str(agent_id)] = idx
        for agent_id, idx in last_index_by_agent.items():
            bonus = (
                rewards_by_agent.get(agent_id, 0.0)
                if rewards_by_agent is not None
                else scalar_reward
            )
            self.rewards[idx] = float(self.rewards[idx]) + float(bonus)


class MAPPOActorCritic(nn.Module):
    """Decentralized actor with a centralized critic."""

    def __init__(
        self,
        local_obs_dim: int,
        global_state_dim: int,
        max_agents: int,
        action_dim: int,
        hidden_dim: int = 128,
        hidden_dims: Optional[Sequence[int]] = None,
        activation: str = "tanh",
        use_layer_norm: bool = False,
        orthogonal_init: bool = False,
    ) -> None:
        super().__init__()
        self.local_obs_dim = int(local_obs_dim)
        self.global_state_dim = int(global_state_dim)
        self.max_agents = int(max_agents)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(dim) for dim in (hidden_dims or (hidden_dim, hidden_dim)))
        self.activation = str(activation)
        self.use_layer_norm = bool(use_layer_norm)
        self.orthogonal_init = bool(orthogonal_init)
        actor_dim = self.local_obs_dim + self.max_agents
        critic_dim = self.global_state_dim + self.max_agents
        self.actor = _build_mlp(
            actor_dim,
            self.action_dim,
            self.hidden_dims,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
        )
        self.critic = _build_mlp(
            critic_dim,
            1,
            self.hidden_dims,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
        )
        if self.orthogonal_init:
            _orthogonal_init(self.actor, activation=self.activation, output_gain=0.01)
            _orthogonal_init(self.critic, activation=self.activation, output_gain=1.0)

    def _conditioned_input(self, features: torch.Tensor, agent_indices: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        agent_indices = agent_indices.long().view(-1)
        if features.size(0) == 1 and agent_indices.numel() > 1:
            features = features.expand(agent_indices.numel(), -1)
        one_hot = F.one_hot(agent_indices, num_classes=self.max_agents).to(
            dtype=torch.float32,
            device=features.device,
        )
        return torch.cat([features, one_hot], dim=-1)

    def _actor_input(self, local_observations: torch.Tensor, agent_indices: torch.Tensor) -> torch.Tensor:
        return self._conditioned_input(local_observations, agent_indices)

    def _critic_input(self, global_states: torch.Tensor, agent_indices: torch.Tensor) -> torch.Tensor:
        return self._conditioned_input(global_states, agent_indices)

    def act(
        self,
        local_obs: torch.Tensor,
        global_state: torch.Tensor,
        agent_index: int,
        *,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.tensor([int(agent_index)], dtype=torch.long, device=global_state.device)
        actor_input = self._actor_input(local_obs, idx)
        critic_input = self._critic_input(global_state, idx)
        logits = self.actor(actor_input).squeeze(0)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.critic(critic_input).squeeze(-1).squeeze(0)
        entropy = dist.entropy()
        return action.detach(), logprob.detach(), value.detach(), entropy.detach()

    def evaluate(
        self,
        local_observations: torch.Tensor,
        global_states: torch.Tensor,
        agent_indices: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_input = self._actor_input(local_observations, agent_indices)
        critic_input = self._critic_input(global_states, agent_indices)
        logits = self.actor(actor_input)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(critic_input).squeeze(-1)
        return logprobs, values, entropy

    @torch.no_grad()
    def value(self, global_state: torch.Tensor, agent_indices: torch.Tensor | int) -> torch.Tensor:
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        if isinstance(agent_indices, int):
            indices = torch.tensor([agent_indices], dtype=torch.long, device=global_state.device)
        else:
            indices = torch.as_tensor(agent_indices, dtype=torch.long, device=global_state.device).view(-1)
        critic_input = self._critic_input(global_state, indices)
        return self.critic(critic_input).squeeze(-1)


class MAPPO:
    """PPO wrapper for centralized multi-agent training."""

    def __init__(
        self,
        global_state_dim: int,
        local_obs_dim: int,
        action_dim: int,
        *,
        max_agents: int = MAX_AGENTS,
        hidden_dim: int = 128,
        hidden_dims: Optional[Sequence[int]] = None,
        activation: str = "tanh",
        use_layer_norm: bool = False,
        orthogonal_init: bool = False,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        K_epochs: int = 5,
        eps_clip: float = 0.2,
        mb_size: int = 256,
        gae_lambda: float = 0.95,
        normalize_reward: bool = False,
    ) -> None:
        self.global_state_dim = int(global_state_dim)
        self.local_obs_dim = int(local_obs_dim)
        self.action_dim = int(action_dim)
        self.max_agents = int(max_agents)
        self.hidden_dims = tuple(int(dim) for dim in (hidden_dims or (hidden_dim, hidden_dim)))
        self.activation = str(activation)
        self.use_layer_norm = bool(use_layer_norm)
        self.orthogonal_init = bool(orthogonal_init)
        self.gamma = float(gamma)
        self.K_epochs = int(K_epochs)
        self.eps_clip = float(eps_clip)
        self.mb_size = int(mb_size)
        self.gae_lambda = float(gae_lambda)
        self.normalize_reward = bool(normalize_reward)

        self.buffer = RolloutBuffer()
        self.global_obs_rms = RunningMeanStd(shape=self.global_state_dim)
        self.local_obs_rms = RunningMeanStd(shape=self.local_obs_dim)
        self.obs_rms = self.global_obs_rms
        self.policy = MAPPOActorCritic(
            self.local_obs_dim,
            self.global_state_dim,
            self.max_agents,
            self.action_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            orthogonal_init=self.orthogonal_init,
        ).to(device)
        self.policy_old = MAPPOActorCritic(
            self.local_obs_dim,
            self.global_state_dim,
            self.max_agents,
            self.action_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            orthogonal_init=self.orthogonal_init,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )
        self.loss_fn = nn.MSELoss()

    def _float_tensor(self, value) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float32, device=device)

    def _normalize_global_train(self, global_state_tensor: torch.Tensor) -> torch.Tensor:
        self.global_obs_rms.update(global_state_tensor)
        return self.global_obs_rms.normalize(global_state_tensor)

    def _normalize_global_eval(self, global_state_tensor: torch.Tensor) -> torch.Tensor:
        return self.global_obs_rms.normalize(global_state_tensor)

    def _normalize_local_train(self, local_obs_tensor: torch.Tensor) -> torch.Tensor:
        self.local_obs_rms.update(local_obs_tensor)
        return self.local_obs_rms.normalize(local_obs_tensor)

    def _normalize_local_eval(self, local_obs_tensor: torch.Tensor) -> torch.Tensor:
        return self.local_obs_rms.normalize(local_obs_tensor)

    def select_actions(
        self,
        global_state,
        local_observations: Mapping[str, object],
        active_agent_ids: Sequence[str],
        *,
        episode_id: int,
        deterministic: bool = False,
        store: bool = True,
    ) -> Tuple[Dict[str, int], List[ActionRecord], float]:
        state_tensor = self._float_tensor(global_state)
        norm_state = self._normalize_global_train(state_tensor) if store else self._normalize_global_eval(state_tensor)
        actions: Dict[str, int] = {}
        records: List[ActionRecord] = []
        entropies: List[float] = []
        with torch.no_grad():
            for agent_id in active_agent_ids:
                if agent_id not in local_observations:
                    raise KeyError(f"Missing local observation for active agent {agent_id}")
                local_tensor = self._float_tensor(local_observations[agent_id])
                norm_local = (
                    self._normalize_local_train(local_tensor)
                    if store
                    else self._normalize_local_eval(local_tensor)
                )
                agent_index = agent_id_to_index(agent_id, self.max_agents)
                action, logprob, value, entropy = self.policy_old.act(
                    norm_local,
                    norm_state,
                    agent_index,
                    deterministic=deterministic,
                )
                actions[str(agent_id)] = int(action.item())
                entropies.append(float(entropy.item()))
                if store:
                    records.append(
                        ActionRecord(
                            episode_id=int(episode_id),
                            agent_id=str(agent_id),
                            agent_index=agent_index,
                            local_obs=norm_local,
                            global_state=norm_state,
                            action=action,
                            logprob=logprob,
                            value=value,
                        )
                    )
        mean_entropy = float(np.mean(entropies)) if entropies else 0.0
        return actions, records, mean_entropy

    def store_outcomes(
        self,
        records: Iterable[ActionRecord],
        *,
        reward: float | Mapping[str, float],
        terminals: Mapping[str, bool],
    ) -> None:
        for record in records:
            record_reward = (
                float(reward.get(record.agent_id, 0.0))
                if isinstance(reward, MappingABC)
                else float(reward)
            )
            self.buffer.add(
                record,
                reward=record_reward,
                terminal=bool(terminals.get(record.agent_id, False)),
            )

    def add_terminal_bonus(self, episode_id: int, reward: float | Mapping[str, float]) -> None:
        self.buffer.add_terminal_bonus(episode_id, reward)

    @torch.no_grad()
    def bootstrap_values(
        self,
        global_state,
        active_agent_ids: Sequence[str],
        *,
        episode_id: int,
    ) -> Dict[Tuple[int, str], float]:
        if not active_agent_ids:
            return {}
        state_tensor = self._float_tensor(global_state)
        norm_state = self._normalize_global_eval(state_tensor)
        values: Dict[Tuple[int, str], float] = {}
        for agent_id in active_agent_ids:
            agent_index = agent_id_to_index(agent_id, self.max_agents)
            value = float(self.policy_old.value(norm_state, agent_index).squeeze(0).item())
            values[(int(episode_id), str(agent_id))] = value
        return values

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        *,
        bootstrap_values: Optional[Mapping[Tuple[int, str], float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bootstrap_values = bootstrap_values or {}
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        grouped: Dict[Tuple[int, str], List[int]] = defaultdict(list)
        for idx, key in enumerate(zip(self.buffer.episode_ids, self.buffer.agent_ids)):
            grouped[(int(key[0]), str(key[1]))].append(idx)

        for key, indices in grouped.items():
            next_value = float(bootstrap_values.get(key, 0.0))
            last_gae = 0.0
            for idx in reversed(indices):
                nonterminal = 1.0 - float(dones[idx].item())
                delta = (
                    float(rewards[idx].item())
                    + self.gamma * next_value * nonterminal
                    - float(values[idx].item())
                )
                last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
                advantages[idx] = float(last_gae)
                returns[idx] = advantages[idx] + values[idx]
                next_value = float(values[idx].item())
        return advantages, returns

    def update(
        self,
        bootstrap_values: Optional[Mapping[Tuple[int, str], float]] = None,
    ) -> Dict[str, float]:
        if len(self.buffer) == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        local_observations = torch.stack(self.buffer.local_observations, dim=0).to(device)
        states = torch.stack(self.buffer.global_states, dim=0).to(device)
        agent_indices = torch.tensor(self.buffer.agent_indices, dtype=torch.long, device=device)
        actions = torch.stack(self.buffer.actions, dim=0).long().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).to(device)
        values = torch.stack(self.buffer.values, dim=0).float().to(device)
        rewards_raw = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.buffer.terminals, dtype=torch.float32, device=device)

        rewards = rewards_raw
        if self.normalize_reward and rewards_raw.numel() > 1:
            rewards = (rewards_raw - rewards_raw.mean()) / (rewards_raw.std(unbiased=False) + 1e-8)

        advantages, returns = self._compute_gae(
            rewards,
            values,
            dones,
            bootstrap_values=bootstrap_values,
        )
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = local_observations.size(0)
        mb_size = min(self.mb_size, batch_size)
        metric_sums = defaultdict(float)
        metric_count = 0
        for _ in range(self.K_epochs):
            order = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                idx = order[start : start + mb_size]
                mb_local_observations = local_observations[idx]
                mb_states = states[idx]
                mb_agents = agent_indices[idx]
                mb_actions = actions[idx]
                mb_old_logprobs = old_logprobs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                logprobs, state_values, entropy = self.policy.evaluate(
                    mb_local_observations,
                    mb_states,
                    mb_agents,
                    mb_actions,
                )
                ratios = torch.exp(logprobs - mb_old_logprobs.detach())
                logratio = logprobs - mb_old_logprobs.detach()
                approx_kl = ((ratios - 1.0) - logratio).mean()
                clip_fraction = (
                    torch.abs(ratios - 1.0) > self.eps_clip
                ).to(dtype=torch.float32).mean()
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.loss_fn(state_values, mb_returns)
                entropy_mean = entropy.mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                metric_sums["loss"] += float(loss.item())
                metric_sums["policy_loss"] += float(policy_loss.item())
                metric_sums["value_loss"] += float(value_loss.item())
                metric_sums["entropy"] += float(entropy_mean.item())
                metric_sums["approx_kl"] += float(approx_kl.item())
                metric_sums["clip_fraction"] += float(clip_fraction.item())
                metric_count += 1

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        return {
            key: value / max(metric_count, 1)
            for key, value in metric_sums.items()
        }

    def _model_payload(self) -> Dict[str, object]:
        return {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "policy_old": self.policy_old.state_dict(),
            "global_obs_rms": self.global_obs_rms.state_dict(),
            "local_obs_rms": self.local_obs_rms.state_dict(),
            "obs_rms": self.global_obs_rms.state_dict(),
            "local_obs_dim": self.local_obs_dim,
            "global_state_dim": self.global_state_dim,
            "action_dim": self.action_dim,
            "max_agents": self.max_agents,
        }

    def _validate_model_payload(self, payload: Mapping[str, object]) -> None:
        schema_version = int(payload.get("model_schema_version", 1))
        if schema_version != MODEL_SCHEMA_VERSION:
            raise ValueError(
                "Incompatible MAPPO checkpoint schema. "
                f"Expected schema {MODEL_SCHEMA_VERSION} with local-observation K-neighbor actor, "
                f"got schema {schema_version}. Retrain from scratch or use a matching checkpoint."
            )
        payload_local_dim = int(payload.get("local_obs_dim", -1))
        payload_global_dim = int(payload.get("global_state_dim", -1))
        if payload_local_dim != self.local_obs_dim or payload_global_dim != self.global_state_dim:
            raise ValueError(
                "Incompatible MAPPO checkpoint dimensions. "
                f"Expected local_obs_dim={self.local_obs_dim}, global_state_dim={self.global_state_dim}; "
                f"got local_obs_dim={payload_local_dim}, global_state_dim={payload_global_dim}."
            )

    def save_model(self, path: str | os.PathLike[str]) -> None:
        _atomic_torch_save(self._model_payload(), str(path))

    def load_model(self, path: str | os.PathLike[str]) -> None:
        payload = _torch_load(str(path), map_location=device)
        self._validate_model_payload(payload)
        self.policy_old.load_state_dict(payload["policy_old"])
        self.policy.load_state_dict(payload["policy_old"])
        self.global_obs_rms.load_state_dict(payload["global_obs_rms"])
        self.local_obs_rms.load_state_dict(payload["local_obs_rms"])
        self.obs_rms = self.global_obs_rms

    def save_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        agent_steps: int,
        env_steps: int,
        episode: int,
    ) -> None:
        payload = {
            **self._model_payload(),
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "agent_steps": int(agent_steps),
            "env_steps": int(env_steps),
            "episode": int(episode),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "py_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        _atomic_torch_save(payload, str(path))

    def load_checkpoint(self, path: str | os.PathLike[str]) -> Tuple[int, int, int]:
        payload = _torch_load(str(path), map_location=device)
        self._validate_model_payload(payload)
        self.policy.load_state_dict(payload["policy"])
        self.policy_old.load_state_dict(payload["policy_old"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_obs_rms.load_state_dict(payload["global_obs_rms"])
        self.local_obs_rms.load_state_dict(payload["local_obs_rms"])
        self.obs_rms = self.global_obs_rms
        torch_rng = payload.get("torch_rng")
        if torch_rng is not None:
            torch_rng_tensor = torch.as_tensor(torch_rng, dtype=torch.uint8, device="cpu")
            torch.set_rng_state(torch_rng_tensor)
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["py_rng"])
        cuda_rng = payload.get("cuda_rng")
        if torch.cuda.is_available() and cuda_rng is not None:
            cuda_rng_tensors = [
                torch.as_tensor(state, dtype=torch.uint8, device="cpu")
                for state in cuda_rng
            ]
            torch.cuda.set_rng_state_all(cuda_rng_tensors)
        return int(payload["agent_steps"]), int(payload["env_steps"]), int(payload["episode"])


def evaluate_policy(
    agent: MAPPO,
    make_env: Callable[[], object],
    *,
    n_episodes: int = 10,
    seed: int = 0,
    reset_options: Optional[Mapping[str, object]] = None,
) -> Dict[str, float]:
    returns: List[float] = []
    lengths: List[int] = []
    successes = 0
    collisions = 0
    weather_failures = 0
    truncations = 0
    reset_options = dict(reset_options or {})

    for ep_idx in range(int(n_episodes)):
        env = make_env()
        observations, info = env.reset(seed=int(seed) + ep_idx, options=reset_options)
        done = bool(info["__common__"]["episode_done"])
        truncated = bool(info["__common__"]["episode_truncated"])
        episode_return = 0.0
        steps = 0
        while not done and not truncated:
            active_ids = list(env.agents)
            if not active_ids:
                observations, _, _, _, info = env.step({})
            else:
                actions, _, _ = agent.select_actions(
                    global_state=env.state(),
                    local_observations=observations,
                    active_agent_ids=active_ids,
                    episode_id=ep_idx,
                    deterministic=True,
                    store=False,
                )
                observations, _, _, _, info = env.step(actions)
            common = info["__common__"]
            episode_return += float(common["team_reward"])
            done = bool(common["episode_done"])
            truncated = bool(common["episode_truncated"])
            steps += 1
            if steps > env.core.max_step + 5:
                break

        common = info["__common__"]
        failure_reason = common["failure_reason"]
        returns.append(float(episode_return))
        lengths.append(int(steps))
        successes += int(bool(common["episode_success"]))
        collisions += int(failure_reason == "collision")
        weather_failures += int(failure_reason == "weather")
        truncations += int(bool(common["episode_truncated"]))

    n = max(int(n_episodes), 1)
    total_episode_reward = float(np.sum(returns)) if returns else 0.0
    mean_episode_total_reward = float(np.mean(returns)) if returns else 0.0
    return {
        "total_episode_reward": total_episode_reward,
        "mean_episode_total_reward": mean_episode_total_reward,
        "mean_return": mean_episode_total_reward,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "mean_steps": float(np.mean(lengths)) if lengths else 0.0,
        "success_rate": successes / n,
        "collision_rate": collisions / n,
        "weather_rate": weather_failures / n,
        "truncation_rate": truncations / n,
    }
