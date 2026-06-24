"""Teacher policies and trajectory dataset for dual-teacher distillation.

A "teacher" is a small network that imitates one guidance source (the LLM guide
or the preview heuristic). It is trained offline by ``train_teachers.py`` on
trajectories produced by ``generate_guided_trajectories.py``, and then queried
during MAPPO training to provide a distillation target for the student policy.

Design notes
------------
* A teacher uses the **same input convention** as the student actor:
  ``local_obs`` concatenated with a one-hot ``agent_index`` (see
  ``MAPPOActorCritic._conditioned_input``), and outputs logits over
  ``ACTION_BINS``. This makes ``action_logits`` drop-in for the distillation
  loss.
* The student stores *normalized* observations (running mean/std updated online),
  but offline trajectories are *raw*. To stay consistent, each teacher carries
  its **own frozen input standardization** (mean/std fit on the training
  dataset) and is always fed raw observations. The student keeps its own
  normalization; only the output distributions are compared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from configs import ACTION_BINS, MAX_AGENTS
from .mappo import _atomic_torch_save, _build_mlp, _torch_load, device

TEACHER_SCHEMA_VERSION = 1


class TeacherPolicy(nn.Module):
    """Actor-only MLP that imitates one guidance source.

    Input layout matches the student actor: ``[standardize(local_obs),
    one_hot(agent_index)]``. Output: logits over ``ACTION_BINS``.
    """

    def __init__(
        self,
        local_obs_dim: int,
        action_dim: int,
        *,
        max_agents: int = MAX_AGENTS,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "silu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.local_obs_dim = int(local_obs_dim)
        self.action_dim = int(action_dim)
        self.max_agents = int(max_agents)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.activation = str(activation)
        self.use_layer_norm = bool(use_layer_norm)
        in_dim = self.local_obs_dim + self.max_agents
        self.net = _build_mlp(
            in_dim,
            self.action_dim,
            self.hidden_dims,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
        )
        # Frozen input standardization, fit on the training dataset.
        self.register_buffer("input_mean", torch.zeros(self.local_obs_dim))
        self.register_buffer("input_std", torch.ones(self.local_obs_dim))

    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean_t = torch.as_tensor(mean, dtype=torch.float32).view(-1)
        std_t = torch.as_tensor(std, dtype=torch.float32).view(-1)
        std_t = torch.clamp(std_t, min=1e-6)
        with torch.no_grad():
            self.input_mean.copy_(mean_t.to(self.input_mean.device))
            self.input_std.copy_(std_t.to(self.input_std.device))

    def _conditioned_input(
        self,
        local_observations: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> torch.Tensor:
        features = local_observations
        if features.dim() == 1:
            features = features.unsqueeze(0)
        features = (features - self.input_mean) / self.input_std
        agent_indices = agent_indices.long().view(-1)
        if features.size(0) == 1 and agent_indices.numel() > 1:
            features = features.expand(agent_indices.numel(), -1)
        one_hot = nn.functional.one_hot(agent_indices, num_classes=self.max_agents).to(
            dtype=torch.float32,
            device=features.device,
        )
        return torch.cat([features, one_hot], dim=-1)

    def action_logits(
        self,
        local_observations: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(self._conditioned_input(local_observations, agent_indices))

    def forward(
        self,
        local_observations: torch.Tensor,
        agent_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.action_logits(local_observations, agent_indices)


# ---------------------------------------------------------------------------
# Teacher checkpoint save / load
# ---------------------------------------------------------------------------
def save_teacher(teacher: TeacherPolicy, path: str | Path, meta: Optional[Dict] = None) -> None:
    payload = {
        "teacher_schema_version": TEACHER_SCHEMA_VERSION,
        "state_dict": teacher.state_dict(),
        "local_obs_dim": int(teacher.local_obs_dim),
        "action_dim": int(teacher.action_dim),
        "max_agents": int(teacher.max_agents),
        "hidden_dims": list(teacher.hidden_dims),
        "activation": str(teacher.activation),
        "use_layer_norm": bool(teacher.use_layer_norm),
        "action_bins": list(ACTION_BINS),
        "meta": dict(meta or {}),
    }
    _atomic_torch_save(payload, str(path))


def load_teacher(path: str | Path, map_location=device) -> TeacherPolicy:
    payload = _torch_load(str(path), map_location=map_location)
    teacher = TeacherPolicy(
        local_obs_dim=int(payload["local_obs_dim"]),
        action_dim=int(payload["action_dim"]),
        max_agents=int(payload["max_agents"]),
        hidden_dims=tuple(payload.get("hidden_dims", (128, 128))),
        activation=str(payload.get("activation", "silu")),
        use_layer_norm=bool(payload.get("use_layer_norm", True)),
    )
    teacher.load_state_dict(payload["state_dict"])
    teacher.to(map_location)
    teacher.eval()
    return teacher


# ---------------------------------------------------------------------------
# Trajectory dataset (produced by generate_guided_trajectories.py)
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryArrays:
    """In-memory arrays for one guidance source's trajectories."""

    local_obs: np.ndarray              # (N, obs_dim) float32, raw (un-normalized)
    agent_index: np.ndarray            # (N,) int64
    action_idx: np.ndarray             # (N,) int64 -- the guide's chosen action bin
    episode_id: np.ndarray             # (N,) int64
    step: np.ndarray                   # (N,) int64
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.local_obs.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.local_obs.shape[1])

    def input_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.local_obs.mean(axis=0)
        std = self.local_obs.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)


def save_trajectory_shard(
    out_dir: str | Path,
    shard_index: int,
    *,
    local_obs: np.ndarray,
    agent_index: np.ndarray,
    action_idx: np.ndarray,
    episode_id: np.ndarray,
    step: np.ndarray,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{int(shard_index):04d}.npz"
    tmp_path = out_dir / f"shard_{int(shard_index):04d}.npz.tmp"
    # Write through a file handle so numpy does not append a ".npz" suffix to
    # the temp name, which would break the atomic replace below.
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(
            handle,
            local_obs=local_obs.astype(np.float32),
            agent_index=agent_index.astype(np.int64),
            action_idx=action_idx.astype(np.int64),
            episode_id=episode_id.astype(np.int64),
            step=step.astype(np.int64),
        )
    tmp_path.replace(shard_path)
    return shard_path


def write_dataset_meta(out_dir: str | Path, meta: Dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "dataset_meta.json"
    tmp_path = meta_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    tmp_path.replace(meta_path)
    return meta_path


def load_trajectory_dataset(path: str | Path) -> TrajectoryArrays:
    """Load all shards under ``path`` (a directory) into one TrajectoryArrays."""
    path = Path(path)
    meta: Dict = {}
    meta_path = path / "dataset_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    shard_paths = sorted(path.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz files found under {path}")
    chunks: Dict[str, List[np.ndarray]] = {
        "local_obs": [], "agent_index": [], "action_idx": [], "episode_id": [], "step": []
    }
    for shard_path in shard_paths:
        with np.load(shard_path) as data:
            for key in chunks:
                chunks[key].append(data[key])
    return TrajectoryArrays(
        local_obs=np.concatenate(chunks["local_obs"], axis=0),
        agent_index=np.concatenate(chunks["agent_index"], axis=0),
        action_idx=np.concatenate(chunks["action_idx"], axis=0),
        episode_id=np.concatenate(chunks["episode_id"], axis=0),
        step=np.concatenate(chunks["step"], axis=0),
        meta=meta,
    )
