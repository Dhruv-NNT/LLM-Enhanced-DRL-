"""Framework-agnostic multi-agent sector simulator."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from shapely.geometry import Point

from configs import AGENT_SPEED, GOAL_RADIUS, MAX_AGENTS, MAX_STEP, SAFE_R
from .utils import (
    AssignedRoute,
    assign_routes,
    build_route_catalog,
    clip_position,
    heading_from_line,
    line_signed_cross_track,
    load_sector_polygon,
    plot_sector,
    possible_agent_ids,
    save_rendered_figure,
)


@dataclass
class AgentState:
    agent_id: str
    route: Any
    color: str
    start_step: int
    position: Tuple[float, float]
    destination: Tuple[float, float]
    heading_rad: float
    route_progress: float = 0.0
    launched: bool = False
    active: bool = False
    finished: bool = False
    inside_sector: bool = False
    has_entered_sector: bool = False
    sector_entry_step: Optional[int] = None
    safe_streak: int = 0
    last_turn_deg: int = 0
    conflict_predicted: bool = False
    boundary_warning_active: bool = False
    last_event: Optional[str] = None
    just_finished: bool = False


class MultiAgentSectorCore:
    """Stateful multi-agent simulator shared by RL, LLM, and evaluation wrappers."""

    local_obs_dim = 17

    def __init__(
        self,
        *,
        num_agents: int = 2,
        max_agents: int = MAX_AGENTS,
        safe_r: float = SAFE_R,
        goal_radius: float = GOAL_RADIUS,
        max_step: int = MAX_STEP,
        speed: float = AGENT_SPEED,
    ) -> None:
        self.num_agents = int(num_agents)
        self.max_agents = int(max_agents)
        self.safe_r = float(safe_r)
        self.goal_radius = float(goal_radius)
        self.max_step = int(max_step)
        self.speed = float(speed)
        self.sector_polygon = load_sector_polygon()
        self.possible_agents = possible_agent_ids(self.max_agents)

        self.assignments: List[AssignedRoute] = []
        self.agent_states: Dict[str, AgentState] = {}
        self.n_step = 0
        self.reward = 0.0
        self.total_reward = 0.0
        self.last_team_done = False
        self.last_team_truncated = False
        self.last_team_success = False
        self.last_pairwise_separations: Dict[Tuple[str, str], float] = {}

    def clone(self) -> "MultiAgentSectorCore":
        return copy.deepcopy(self)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        num_agents: Optional[int] = None,
        route_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if num_agents is not None:
            self.num_agents = int(num_agents)
        self.assignments = assign_routes(self.num_agents, seed=seed, route_ids=route_ids)
        self.agent_states = {}
        self.n_step = 0
        self.reward = 0.0
        self.total_reward = 0.0
        self.last_team_done = False
        self.last_team_truncated = False
        self.last_team_success = False
        self.last_pairwise_separations = {}

        for assignment in self.assignments:
            route = assignment.route
            start_position = tuple(route.points[0])
            heading = heading_from_line(route.linestring, 0.0)
            launched = assignment.start_step == 0
            self.agent_states[assignment.agent_id] = AgentState(
                agent_id=assignment.agent_id,
                route=route,
                color=assignment.color,
                start_step=assignment.start_step,
                position=start_position,
                destination=tuple(route.points[-1]),
                heading_rad=heading,
                launched=launched,
                active=launched,
            )

        observations = self.get_active_observations()
        infos = self.get_infos()
        return observations, infos

    @property
    def all_agents_finished(self) -> bool:
        return bool(self.agent_states) and all(state.finished for state in self.agent_states.values())

    @property
    def active_agent_ids(self) -> List[str]:
        return [
            agent_id
            for agent_id, state in self.agent_states.items()
            if state.launched and not state.finished
        ]

    def get_agent(self, agent_id: str) -> AgentState:
        return self.agent_states[agent_id]

    def _launch_agents(self) -> None:
        for state in self.agent_states.values():
            if not state.launched and self.n_step >= state.start_step:
                state.launched = True
                state.active = True
                state.last_event = f"launch@{self.n_step}"

    def _route_step(self, state: AgentState) -> None:
        route = state.route
        line = route.linestring
        next_progress = min(float(line.length), state.route_progress + self.speed)
        point = line.interpolate(next_progress)
        state.position = (float(point.x), float(point.y))
        state.route_progress = next_progress
        state.heading_rad = heading_from_line(line, next_progress)

    def _free_step(self, state: AgentState, turn_deg: int) -> None:
        state.heading_rad += math.radians(float(turn_deg))
        new_x = state.position[0] + self.speed * math.cos(state.heading_rad)
        new_y = state.position[1] + self.speed * math.sin(state.heading_rad)
        state.position = clip_position(new_x, new_y)

    def _update_agent_flags(self, state: AgentState) -> None:
        state.just_finished = False
        inside_now = bool(self.sector_polygon.covers(Point(state.position)))
        if inside_now and not state.has_entered_sector:
            state.has_entered_sector = True
            state.sector_entry_step = self.n_step
            state.last_event = f"entered_sector@{self.n_step}"
        state.inside_sector = inside_now

        if self.distance_to_destination(state) <= self.goal_radius:
            state.finished = True
            state.active = False
            state.just_finished = True
            state.last_event = f"finished@{self.n_step}"

    def _pairwise_distances(self) -> Dict[Tuple[str, str], float]:
        distances: Dict[Tuple[str, str], float] = {}
        active_ids = self.active_agent_ids
        for idx, agent_id in enumerate(active_ids):
            pos_a = np.array(self.agent_states[agent_id].position, dtype=float)
            for other_id in active_ids[idx + 1 :]:
                pos_b = np.array(self.agent_states[other_id].position, dtype=float)
                distances[(agent_id, other_id)] = float(np.linalg.norm(pos_a - pos_b))
        return distances

    def distance_to_destination(self, state: AgentState) -> float:
        return float(np.linalg.norm(np.array(state.position) - np.array(state.destination)))

    def nearest_neighbor(self, agent_id: str) -> Tuple[Optional[str], float]:
        state = self.agent_states[agent_id]
        best_id = None
        best_dist = float("inf")
        for other_id in self.active_agent_ids:
            if other_id == agent_id:
                continue
            other_state = self.agent_states[other_id]
            dist = float(np.linalg.norm(np.array(state.position) - np.array(other_state.position)))
            if dist < best_dist:
                best_dist = dist
                best_id = other_id
        return best_id, best_dist if best_id is not None else float("inf")

    def _bearing_to_neighbor_deg(self, agent_id: str, other_id: Optional[str]) -> float:
        if other_id is None:
            return 0.0
        state = self.agent_states[agent_id]
        other = self.agent_states[other_id]
        absolute = math.degrees(
            math.atan2(other.position[1] - state.position[1], other.position[0] - state.position[0])
        )
        own_deg = math.degrees(state.heading_rad)
        return ((absolute - own_deg + 180.0) % 360.0) - 180.0

    def boundary_distance(self, state: AgentState) -> Optional[float]:
        if not state.inside_sector:
            return None
        return float(self.sector_polygon.exterior.distance(Point(state.position)))

    def project_free_motion(
        self,
        *,
        agent_id: str,
        heading_rad: Optional[float] = None,
        horizon_steps: int = 1,
    ) -> List[Tuple[float, float]]:
        state = self.agent_states[agent_id]
        heading = state.heading_rad if heading_rad is None else float(heading_rad)
        positions: List[Tuple[float, float]] = []
        x_val, y_val = state.position
        for _ in range(int(horizon_steps)):
            x_val += self.speed * math.cos(heading)
            y_val += self.speed * math.sin(heading)
            x_val, y_val = clip_position(x_val, y_val)
            positions.append((x_val, y_val))
        return positions

    def step(
        self,
        action_dict: Optional[Dict[str, int]] = None,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        action_dict = action_dict or {}
        previous_distances = {
            agent_id: self.distance_to_destination(state)
            for agent_id, state in self.agent_states.items()
        }
        self.n_step += 1
        self._launch_agents()

        for agent_id, state in self.agent_states.items():
            if not state.launched or state.finished:
                continue
            turn_deg = int(action_dict.get(agent_id, 0))
            state.last_turn_deg = turn_deg
            if not state.has_entered_sector:
                self._route_step(state)
            else:
                self._free_step(state, turn_deg)
            self._update_agent_flags(state)

        pairwise = self._pairwise_distances()
        self.last_pairwise_separations = pairwise
        active_ids = self.active_agent_ids
        conflict_count = 0
        for state in self.agent_states.values():
            if not state.launched or state.finished:
                continue
            neighbor_id, neighbor_dist = self.nearest_neighbor(state.agent_id)
            state.conflict_predicted = neighbor_id is not None and neighbor_dist < 1.5 * self.safe_r
            state.boundary_warning_active = bool(
                state.inside_sector
                and self.boundary_distance(state) is not None
                and float(self.boundary_distance(state)) < self.safe_r
            )
            if state.conflict_predicted:
                conflict_count += 1
                state.safe_streak = 0
            else:
                state.safe_streak += 1

        collision = any(distance < self.safe_r for distance in pairwise.values())
        team_done = collision or self.all_agents_finished
        team_truncated = self.n_step >= self.max_step and not team_done
        team_success = self.all_agents_finished and not collision and not team_truncated

        progress_values: List[float] = []
        cross_tracks: List[float] = []
        newly_finished = 0
        for state in self.agent_states.values():
            if not state.launched:
                continue
            progress_values.append(previous_distances[state.agent_id] - self.distance_to_destination(state))
            signed_xtrk, _, _ = line_signed_cross_track(state.route.linestring, state.position)
            cross_tracks.append(abs(signed_xtrk))
            if state.just_finished:
                newly_finished += 1

        reward = 0.0
        if progress_values:
            reward += float(np.mean(progress_values))
        reward -= 0.01
        if cross_tracks:
            reward -= 0.01 * float(np.mean(cross_tracks))
        reward -= 0.05 * conflict_count
        reward += 2.0 * newly_finished
        if collision:
            reward -= 10.0
        if team_success:
            reward += 20.0

        self.reward = reward
        self.total_reward += reward
        self.last_team_done = team_done
        self.last_team_truncated = team_truncated
        self.last_team_success = team_success

        observations = self.get_active_observations()
        rewards = {agent_id: reward for agent_id in active_ids}
        terminations = {agent_id: team_done for agent_id in active_ids}
        truncations = {agent_id: team_truncated for agent_id in active_ids}
        infos = self.get_infos()
        return observations, rewards, terminations, truncations, infos

    def get_local_observation(self, agent_id: str) -> np.ndarray:
        state = self.agent_states[agent_id]
        route_heading_deg = math.degrees(heading_from_line(state.route.linestring, state.route_progress))
        own_heading_deg = math.degrees(state.heading_rad)
        heading_error_to_route = ((route_heading_deg - own_heading_deg + 180.0) % 360.0) - 180.0
        signed_xtrk, _, _ = line_signed_cross_track(state.route.linestring, state.position)
        nearest_id, nearest_dist = self.nearest_neighbor(agent_id)
        nearest_bearing = self._bearing_to_neighbor_deg(agent_id, nearest_id)
        boundary_distance = self.boundary_distance(state)
        obs = np.array(
            [
                state.position[0],
                state.position[1],
                state.destination[0],
                state.destination[1],
                self.distance_to_destination(state),
                state.heading_rad,
                route_heading_deg,
                heading_error_to_route,
                signed_xtrk,
                1.0 if state.inside_sector else 0.0,
                1.0 if state.has_entered_sector else 0.0,
                -1.0 if boundary_distance is None else boundary_distance,
                0.0 if nearest_id is None else nearest_dist,
                nearest_bearing,
                1.0 if state.conflict_predicted else 0.0,
                1.0 if state.boundary_warning_active else 0.0,
                float(state.safe_streak),
            ],
            dtype=np.float64,
        )
        return obs

    def get_active_observations(self) -> Dict[str, np.ndarray]:
        return {agent_id: self.get_local_observation(agent_id) for agent_id in self.active_agent_ids}

    def global_state_vector(self) -> np.ndarray:
        blocks: List[float] = []
        for agent_id in self.possible_agents:
            state = self.agent_states.get(agent_id)
            if state is None:
                blocks.extend([0.0] * 12)
                continue
            blocks.extend(
                [
                    1.0 if state.launched else 0.0,
                    1.0 if state.active else 0.0,
                    1.0 if state.finished else 0.0,
                    state.position[0],
                    state.position[1],
                    state.destination[0],
                    state.destination[1],
                    state.heading_rad,
                    1.0 if state.inside_sector else 0.0,
                    1.0 if state.has_entered_sector else 0.0,
                    float(state.start_step),
                    self.distance_to_destination(state),
                ]
            )
        blocks.extend([float(self.n_step), float(self.reward), float(self.total_reward)])
        return np.array(blocks, dtype=np.float64)

    def get_infos(self) -> Dict[str, Any]:
        return {
            "global_state": self.global_state_vector(),
            "possible_agents": list(self.possible_agents),
            "active_agents": list(self.active_agent_ids),
            "episode_done": self.last_team_done,
            "episode_truncated": self.last_team_truncated,
            "episode_success": self.last_team_success,
            "team_reward": self.reward,
            "max_agents_possible": len(build_route_catalog()),
            "route_assignments": {
                assignment.agent_id: {
                    "route_id": assignment.route.route_id,
                    "origin": assignment.route.origin,
                    "destination": assignment.route.destination,
                    "start_step": assignment.start_step,
                    "entry_step": assignment.route.entry_step,
                    "shared_prefix_key": assignment.route.shared_prefix_key,
                }
                for assignment in self.assignments
            },
        }

    def render(self, *, show: bool = False, folder: Optional[str] = None) -> plt.Figure:
        fig = plot_sector()
        ax = fig.gca()

        for assignment in self.assignments:
            state = self.agent_states[assignment.agent_id]
            route = assignment.route
            xs = [point[0] for point in route.points]
            ys = [point[1] for point in route.points]
            ax.plot(xs, ys, color=assignment.color, linestyle="--", alpha=0.9, linewidth=1.2)
            ax.scatter(xs[0], ys[0], c=assignment.color, s=50, marker="^")
            ax.scatter(xs[-1], ys[-1], c=assignment.color, s=20, marker="*")
            ax.add_patch(Circle((xs[-1], ys[-1]), self.goal_radius, color=assignment.color, alpha=0.15))
            if state.launched:
                ax.scatter(state.position[0], state.position[1], c=assignment.color, s=40)
                ax.text(
                    state.position[0] + 0.8,
                    state.position[1] + 0.8,
                    assignment.agent_id,
                    color=assignment.color,
                    fontsize=9,
                    weight="bold",
                )

        ax.set_title(
            "Step {} - Reward {:.3f} - Total Reward {:.3f} - Agents {}".format(
                self.n_step,
                self.reward,
                self.total_reward,
                ",".join(self.active_agent_ids) or "none",
            )
        )
        if show:
            plt.show()
        elif folder is not None:
            output_path = Path(folder) / f"image_{self.n_step:03d}.png"
            save_rendered_figure(fig, output_path)
            print(str(output_path))
        return fig

