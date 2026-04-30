"""Environment wrappers for the multi-agent sector simulator."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces

from configs import ACTION_BINS, MAX_AGENTS, NUM_WEATHER_CELLS_DEFAULT, SAFE_R
from .core import MultiAgentSectorCore
from .utils import action_idx_to_deg

try:  # pragma: no cover
    from pettingzoo import ParallelEnv as PettingZooParallelEnv
except Exception:  # pragma: no cover
    class PettingZooParallelEnv:  # type: ignore[override]
        metadata: Dict[str, Any] = {}


def _active_info_dict(core: MultiAgentSectorCore) -> Dict[str, Dict[str, Any]]:
    common = core.get_infos()
    action_mask = np.ones(len(ACTION_BINS), dtype=np.int8)
    infos = {
        agent_id: {
            "global_state": common["global_state"],
            "action_mask": action_mask.copy(),
            "route_assignment": common["route_assignments"][agent_id],
        }
        for agent_id in core.active_agent_ids
    }
    infos["__common__"] = common
    return infos


class MultiAgentParallelEnv(PettingZooParallelEnv):
    """PettingZoo-style simultaneous-action wrapper around the shared simulator."""

    metadata = {"name": "multi_agent_sector_parallel_v0", "render_modes": ["human"]}

    def __init__(
        self,
        *,
        num_agents: int = 2,
        max_agents: int = MAX_AGENTS,
        safe_r: float = SAFE_R,
        num_weather_cells: int = NUM_WEATHER_CELLS_DEFAULT,
    ) -> None:
        super().__init__()
        self.core = MultiAgentSectorCore(
            num_agents=num_agents,
            max_agents=max_agents,
            safe_r=safe_r,
            num_weather_cells=num_weather_cells,
        )
        self.possible_agents = list(self.core.possible_agents)
        self.agents = list(self.possible_agents[:num_agents])
        self._local_obs_shape = (self.core.local_obs_dim,)
        self._global_obs_shape = self.core.global_state_vector().shape
        self.observation_spaces = {
            agent_id: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self._local_obs_shape,
                dtype=np.float64,
            )
            for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: spaces.Discrete(len(ACTION_BINS))
            for agent_id in self.possible_agents
        }

    def observation_space(self, agent: str) -> spaces.Box:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Discrete:
        return self.action_spaces[agent]

    def state(self) -> np.ndarray:
        return self.core.global_state_vector()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        options = options or {}
        observations, _ = self.core.reset(
            seed=seed,
            num_agents=options.get("num_agents"),
            route_ids=options.get("route_ids"),
            num_weather_cells=options.get("num_weather_cells"),
        )
        self.agents = list(self.core.active_agent_ids)
        return observations, _active_info_dict(self.core)

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        degree_actions = {
            agent_id: action_idx_to_deg(int(action), ACTION_BINS)
            for agent_id, action in (actions or {}).items()
        }
        observations, rewards, terminations, truncations, _ = self.core.step(degree_actions)
        team_done = self.core.last_team_done or self.core.last_team_truncated
        self.agents = [] if team_done else list(self.core.active_agent_ids)
        return observations, rewards, terminations, truncations, _active_info_dict(self.core)

    def render(self, *, show: bool = False, folder: Optional[str] = None):
        return self.core.render(show=show, folder=folder)

    def close(self) -> None:
        return None

    def action_masks(self) -> Dict[str, np.ndarray]:
        mask = np.ones(len(ACTION_BINS), dtype=np.int8)
        return {agent_id: mask.copy() for agent_id in self.core.active_agent_ids}


class JointGuidanceEnv:
    """Centralized wrapper for pure-LLM and debugging rollouts."""

    def __init__(
        self,
        *,
        num_agents: int = 2,
        max_agents: int = MAX_AGENTS,
        safe_r: float = SAFE_R,
        num_weather_cells: int = NUM_WEATHER_CELLS_DEFAULT,
    ) -> None:
        self.core = MultiAgentSectorCore(
            num_agents=num_agents,
            max_agents=max_agents,
            safe_r=safe_r,
            num_weather_cells=num_weather_cells,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        *,
        num_agents: Optional[int] = None,
        route_ids: Optional[Sequence[str]] = None,
        num_weather_cells: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        local_obs, info = self.core.reset(
            seed=seed,
            num_agents=num_agents,
            route_ids=route_ids,
            num_weather_cells=num_weather_cells,
        )
        return self.joint_observation(local_obs), info

    def joint_observation(self, local_obs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        observations = self.core.get_active_observations() if local_obs is None else local_obs
        return {
            "local_observations": observations,
            "global_state": self.core.global_state_vector(),
            "active_agents": list(self.core.active_agent_ids),
            "possible_agents": list(self.core.possible_agents),
            "route_assignments": self.core.get_infos()["route_assignments"],
        }

    def step(
        self,
        turn_deg_actions: Optional[Dict[str, int]] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        local_obs, _, _, _, info = self.core.step(turn_deg_actions or {})
        joint_obs = self.joint_observation(local_obs)
        return joint_obs, float(self.core.reward), bool(self.core.last_team_done), bool(self.core.last_team_truncated), info

    def render(self, *, show: bool = False, folder: Optional[str] = None):
        return self.core.render(show=show, folder=folder)

    @property
    def n_step(self) -> int:
        return self.core.n_step

    @property
    def MAX_STEP(self) -> int:
        return self.core.max_step
