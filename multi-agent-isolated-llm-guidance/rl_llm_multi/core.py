"""Framework-agnostic multi-agent sector simulator with mirrored routes and moving weather."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse
from shapely.geometry import Point

from configs import (
    AGENT_SPEED,
    BOUNDARY_WARNING_DISTANCE_UNITS,
    CLEAR_STREAK_REQUIRED,
    DESTINATION_ALIGNMENT_DEG,
    GOAL_RADIUS,
    HAZARD_LOOKAHEAD_STEPS,
    MAX_AGENTS,
    MAX_STEP,
    PAIR_RISK_BUFFER,
    ROUTE_RECOVERY_XTRACK_UNITS,
    SAFE_R,
    WEATHER_CLEARANCE_BUFFER_NM,
    WEATHER_GROWTH_NM_PER_STEP_MAX,
    WEATHER_GROWTH_NM_PER_STEP_MIN,
    WEATHER_INIT_TRIES,
    WEATHER_MAJOR_RADIUS_NM_MAX,
    WEATHER_MAJOR_RADIUS_NM_MIN,
    WEATHER_MINOR_RATIO_MAX,
    WEATHER_MINOR_RATIO_MIN,
    WEATHER_SPEED_NM_PER_STEP_MAX,
    WEATHER_SPEED_NM_PER_STEP_MIN,
    WEATHER_TRAIL_STEPS,
)
from .utils import (
    AssignedRoute,
    WeatherCell,
    assign_routes,
    build_route_catalog,
    clip_position,
    heading_from_line,
    line_signed_cross_track,
    load_sector_polygon,
    nm_to_grid_mean,
    plot_sector,
    possible_agent_ids,
    save_rendered_figure,
    weather_axis_units,
    weather_polygon,
    weather_signed_clearance,
    weather_trail,
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
    weather_predicted: bool = False
    hazard_predicted: bool = False
    boundary_warning_active: bool = False
    predicted_min_sep: float = float("inf")
    predicted_pair_loss_step: Optional[int] = None
    predicted_weather_clearance: float = float("inf")
    predicted_weather_entry_step: Optional[int] = None
    predicted_boundary_exit_step: Optional[int] = None
    threat_agents: Tuple[str, ...] = field(default_factory=tuple)
    last_event: Optional[str] = None
    just_finished: bool = False


class MultiAgentSectorCore:
    """Stateful multi-agent simulator shared by the isolated LLM runner and evaluator."""

    local_obs_dim = 24

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
        self.weather_clearance_buffer_units = nm_to_grid_mean(WEATHER_CLEARANCE_BUFFER_NM)
        self.sector_polygon = load_sector_polygon()
        self.possible_agents = possible_agent_ids(self.max_agents)

        self.assignments: List[AssignedRoute] = []
        self.agent_states: Dict[str, AgentState] = {}
        self.weather_cell: Optional[WeatherCell] = None
        self.n_step = 0
        self.reward = 0.0
        self.total_reward = 0.0
        self.last_team_done = False
        self.last_team_truncated = False
        self.last_team_success = False
        self.last_failure_reason: Optional[str] = None
        self.last_pairwise_separations: Dict[Tuple[str, str], float] = {}
        self.risky_pairs: List[Dict[str, Any]] = []
        self.rng = random.Random()

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
        self.rng = random.Random(seed)
        self.assignments = assign_routes(self.num_agents, seed=seed, route_ids=route_ids)
        self.agent_states = {}
        self.weather_cell = None
        self.n_step = 0
        self.reward = 0.0
        self.total_reward = 0.0
        self.last_team_done = False
        self.last_team_truncated = False
        self.last_team_success = False
        self.last_failure_reason = None
        self.last_pairwise_separations = {}
        self.risky_pairs = []

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

        self.weather_cell = self._init_weather()
        self._update_hazard_predictions()
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

    def _wrap_angle(self, angle_rad: float) -> float:
        return ((float(angle_rad) + math.pi) % (2.0 * math.pi)) - math.pi

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
        state.heading_rad = self._wrap_angle(state.heading_rad + math.radians(float(turn_deg)))
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

    def _pairwise_distances(self, active_ids: Optional[Sequence[str]] = None) -> Dict[Tuple[str, str], float]:
        distances: Dict[Tuple[str, str], float] = {}
        agent_ids = list(self.active_agent_ids if active_ids is None else active_ids)
        for idx, agent_id in enumerate(agent_ids):
            pos_a = np.array(self.agent_states[agent_id].position, dtype=float)
            for other_id in agent_ids[idx + 1 :]:
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

    def weather_signed_clearance(self, position: Tuple[float, float]) -> float:
        if self.weather_cell is None:
            return float("inf")
        return weather_signed_clearance(self.weather_cell, position)

    def weather_center_distance(self, position: Tuple[float, float]) -> float:
        if self.weather_cell is None:
            return float("inf")
        return float(np.linalg.norm(np.array(position) - np.array(self.weather_cell.center)))

    def weather_relative_bearing_deg(self, state: AgentState) -> float:
        if self.weather_cell is None:
            return 0.0
        absolute = math.degrees(
            math.atan2(
                self.weather_cell.center[1] - state.position[1],
                self.weather_cell.center[0] - state.position[0],
            )
        )
        own_deg = math.degrees(state.heading_rad)
        return ((absolute - own_deg + 180.0) % 360.0) - 180.0

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

    def _init_weather(self) -> WeatherCell:
        launches = [
            state.position
            for state in self.agent_states.values()
            if state.launched
        ]
        destinations = [tuple(assignment.route.points[-1]) for assignment in self.assignments]
        min_x, min_y, max_x, max_y = self.sector_polygon.bounds

        for _ in range(WEATHER_INIT_TRIES):
            major_radius_nm = self.rng.uniform(WEATHER_MAJOR_RADIUS_NM_MIN, WEATHER_MAJOR_RADIUS_NM_MAX)
            minor_ratio = self.rng.uniform(WEATHER_MINOR_RATIO_MIN, WEATHER_MINOR_RATIO_MAX)
            angle_rad = self.rng.uniform(0.0, math.pi)
            motion_heading_rad = self.rng.uniform(-math.pi, math.pi)
            speed_units = nm_to_grid_mean(
                self.rng.uniform(WEATHER_SPEED_NM_PER_STEP_MIN, WEATHER_SPEED_NM_PER_STEP_MAX)
            )
            growth_nm = self.rng.uniform(WEATHER_GROWTH_NM_PER_STEP_MIN, WEATHER_GROWTH_NM_PER_STEP_MAX)
            if self.rng.random() < 0.5:
                growth_nm *= -1.0
            center = (
                self.rng.uniform(min_x, max_x),
                self.rng.uniform(min_y, max_y),
            )
            cell = WeatherCell(
                center=center,
                major_radius_nm=major_radius_nm,
                minor_ratio=minor_ratio,
                angle_rad=angle_rad,
                motion_heading_rad=motion_heading_rad,
                speed_units_per_step=speed_units,
                major_growth_nm_per_step=growth_nm,
            )
            if not self._weather_geometry_valid(cell):
                continue
            if any(weather_signed_clearance(cell, launch) <= 0.0 for launch in launches):
                continue
            if any(weather_polygon(cell).distance(Point(dest)) <= self.goal_radius for dest in destinations):
                continue
            return cell

        raise RuntimeError("Failed to initialize a valid weather ellipse inside the sector.")

    def _weather_geometry_valid(self, cell: WeatherCell) -> bool:
        poly = weather_polygon(cell)
        return bool(poly.within(self.sector_polygon))

    def _advance_weather(self) -> None:
        if self.weather_cell is None:
            return

        cell = copy.deepcopy(self.weather_cell)
        growth_nm = cell.major_growth_nm_per_step
        next_major_nm = cell.major_radius_nm + growth_nm
        if next_major_nm < WEATHER_MAJOR_RADIUS_NM_MIN or next_major_nm > WEATHER_MAJOR_RADIUS_NM_MAX:
            growth_nm *= -1.0
            next_major_nm = min(
                WEATHER_MAJOR_RADIUS_NM_MAX,
                max(WEATHER_MAJOR_RADIUS_NM_MIN, cell.major_radius_nm + growth_nm),
            )

        proposed_center = (
            cell.center[0] + cell.speed_units_per_step * math.cos(cell.motion_heading_rad),
            cell.center[1] + cell.speed_units_per_step * math.sin(cell.motion_heading_rad),
        )
        proposed = WeatherCell(
            center=proposed_center,
            major_radius_nm=next_major_nm,
            minor_ratio=cell.minor_ratio,
            angle_rad=cell.angle_rad,
            motion_heading_rad=cell.motion_heading_rad,
            speed_units_per_step=cell.speed_units_per_step,
            major_growth_nm_per_step=growth_nm,
        )

        destinations = [tuple(assignment.route.points[-1]) for assignment in self.assignments]
        if self._weather_geometry_valid(proposed) and all(
            weather_polygon(proposed).distance(Point(dest)) > self.goal_radius for dest in destinations
        ):
            self.weather_cell = proposed
            return

        bounced_heading = self._wrap_angle(cell.motion_heading_rad + math.pi)
        bounced_growth = -growth_nm
        bounced_major_nm = min(
            WEATHER_MAJOR_RADIUS_NM_MAX,
            max(WEATHER_MAJOR_RADIUS_NM_MIN, cell.major_radius_nm + bounced_growth),
        )
        bounced_center = (
            cell.center[0] + cell.speed_units_per_step * math.cos(bounced_heading),
            cell.center[1] + cell.speed_units_per_step * math.sin(bounced_heading),
        )
        bounced = WeatherCell(
            center=bounced_center,
            major_radius_nm=bounced_major_nm,
            minor_ratio=cell.minor_ratio,
            angle_rad=cell.angle_rad,
            motion_heading_rad=bounced_heading,
            speed_units_per_step=cell.speed_units_per_step,
            major_growth_nm_per_step=bounced_growth,
        )
        if self._weather_geometry_valid(bounced) and all(
            weather_polygon(bounced).distance(Point(dest)) > self.goal_radius for dest in destinations
        ):
            self.weather_cell = bounced
            return

        self.weather_cell.motion_heading_rad = bounced_heading
        self.weather_cell.major_growth_nm_per_step = bounced_growth

    def _update_hazard_predictions(self, *, horizon_steps: int = HAZARD_LOOKAHEAD_STEPS) -> None:
        active_ids = list(self.active_agent_ids)
        if not active_ids:
            self.risky_pairs = []
            return

        pair_summary: Dict[Tuple[str, str], Dict[str, Any]] = {}
        agent_summary = {
            agent_id: {
                "min_sep": float("inf"),
                "pair_loss_step": None,
                "min_weather_clearance": float("inf"),
                "weather_entry_step": None,
                "boundary_exit_step": None,
                "threats": {},
            }
            for agent_id in active_ids
        }

        sim = self.clone()
        for tick in range(1, int(horizon_steps) + 1):
            sim.step({}, compute_predictions=False)
            sim_active = list(sim.active_agent_ids)
            pairwise = sim._pairwise_distances(sim_active)
            for (agent_a, agent_b), distance in pairwise.items():
                key = tuple(sorted((agent_a, agent_b)))
                summary = pair_summary.setdefault(
                    key,
                    {"min_sep": float("inf"), "min_step": None, "loss_step": None},
                )
                if distance < summary["min_sep"]:
                    summary["min_sep"] = float(distance)
                    summary["min_step"] = tick
                if distance < self.safe_r and summary["loss_step"] is None:
                    summary["loss_step"] = tick

                for src, dst in ((agent_a, agent_b), (agent_b, agent_a)):
                    if src not in agent_summary:
                        continue
                    agent_summary[src]["min_sep"] = min(agent_summary[src]["min_sep"], float(distance))
                    threat_summary = agent_summary[src]["threats"].setdefault(
                        dst,
                        {"min_sep": float("inf"), "min_step": None, "loss_step": None},
                    )
                    if distance < threat_summary["min_sep"]:
                        threat_summary["min_sep"] = float(distance)
                        threat_summary["min_step"] = tick
                    if distance < self.safe_r and threat_summary["loss_step"] is None:
                        threat_summary["loss_step"] = tick
                    if distance < self.safe_r and agent_summary[src]["pair_loss_step"] is None:
                        agent_summary[src]["pair_loss_step"] = tick

            for agent_id in active_ids:
                if agent_id not in sim.agent_states:
                    continue
                state = sim.agent_states[agent_id]
                if not state.launched or state.finished:
                    continue
                weather_clearance = sim.weather_signed_clearance(state.position)
                agent_summary[agent_id]["min_weather_clearance"] = min(
                    agent_summary[agent_id]["min_weather_clearance"],
                    float(weather_clearance),
                )
                if weather_clearance < 0.0 and agent_summary[agent_id]["weather_entry_step"] is None:
                    agent_summary[agent_id]["weather_entry_step"] = tick
                if state.has_entered_sector and not state.inside_sector and agent_summary[agent_id]["boundary_exit_step"] is None:
                    agent_summary[agent_id]["boundary_exit_step"] = tick

            if sim.last_team_done or sim.last_team_truncated:
                break

        risky_pairs: List[Dict[str, Any]] = []
        for (agent_a, agent_b), summary in pair_summary.items():
            if summary["min_sep"] < 2.0 * self.safe_r or summary["loss_step"] is not None:
                risky_pairs.append(
                    {
                        "agent_a": agent_a,
                        "agent_b": agent_b,
                        "min_sep": float(summary["min_sep"]),
                        "min_step": summary["min_step"],
                        "loss_step": summary["loss_step"],
                    }
                )
        risky_pairs.sort(
            key=lambda item: (
                999 if item["loss_step"] is None else item["loss_step"],
                item["min_sep"],
                999 if item["min_step"] is None else item["min_step"],
            )
        )
        self.risky_pairs = risky_pairs

        for agent_id in active_ids:
            state = self.agent_states[agent_id]
            summary = agent_summary[agent_id]
            boundary_distance = self.boundary_distance(state)
            state.predicted_min_sep = float(summary["min_sep"])
            state.predicted_pair_loss_step = summary["pair_loss_step"]
            state.predicted_weather_clearance = float(summary["min_weather_clearance"])
            state.predicted_weather_entry_step = summary["weather_entry_step"]
            state.predicted_boundary_exit_step = summary["boundary_exit_step"]
            state.boundary_warning_active = bool(
                state.has_entered_sector
                and (
                    summary["boundary_exit_step"] is not None
                    or (
                        boundary_distance is not None
                        and float(boundary_distance) < BOUNDARY_WARNING_DISTANCE_UNITS
                    )
                )
            )
            state.conflict_predicted = bool(
                summary["pair_loss_step"] is not None
                or (
                    math.isfinite(summary["min_sep"])
                    and float(summary["min_sep"]) < self.safe_r * PAIR_RISK_BUFFER
                )
            )
            state.weather_predicted = bool(
                summary["weather_entry_step"] is not None
                or (
                    math.isfinite(summary["min_weather_clearance"])
                    and float(summary["min_weather_clearance"]) < self.weather_clearance_buffer_units
                )
            )
            state.hazard_predicted = bool(state.conflict_predicted or state.weather_predicted)
            if state.hazard_predicted:
                state.safe_streak = 0
            else:
                state.safe_streak += 1

            ranked_threats = sorted(
                summary["threats"].items(),
                key=lambda item: (
                    999 if item[1]["loss_step"] is None else item[1]["loss_step"],
                    item[1]["min_sep"],
                ),
            )
            state.threat_agents = tuple(agent for agent, _ in ranked_threats[:3])

    def step(
        self,
        action_dict: Optional[Dict[str, int]] = None,
        *,
        compute_predictions: bool = True,
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
        self._advance_weather()

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
        collision = any(distance < self.safe_r for distance in pairwise.values())
        weather_violation_agents = [
            agent_id
            for agent_id in active_ids
            if self.weather_signed_clearance(self.agent_states[agent_id].position) < 0.0
        ]
        weather_failure = bool(weather_violation_agents)

        team_done = collision or weather_failure or self.all_agents_finished
        team_truncated = self.n_step >= self.max_step and not team_done
        team_success = self.all_agents_finished and not collision and not weather_failure and not team_truncated

        if collision:
            self.last_failure_reason = "collision"
        elif weather_failure:
            self.last_failure_reason = "weather"
        elif team_truncated:
            self.last_failure_reason = "truncated"
        elif team_success:
            self.last_failure_reason = "success"
        else:
            self.last_failure_reason = None

        if compute_predictions and not team_done and not team_truncated:
            self._update_hazard_predictions()
        elif compute_predictions:
            self.risky_pairs = []

        progress_values: List[float] = []
        cross_tracks: List[float] = []
        newly_finished = 0
        threat_count = 0
        weather_risk_count = 0
        for state in self.agent_states.values():
            if not state.launched:
                continue
            progress_values.append(previous_distances[state.agent_id] - self.distance_to_destination(state))
            signed_xtrk, _, _ = line_signed_cross_track(state.route.linestring, state.position)
            cross_tracks.append(abs(signed_xtrk))
            if state.just_finished:
                newly_finished += 1
            if state.hazard_predicted:
                threat_count += 1
            if state.weather_predicted:
                weather_risk_count += 1

        reward = 0.0
        if progress_values:
            reward += float(np.mean(progress_values))
        reward -= 0.01
        if cross_tracks:
            reward -= 0.01 * float(np.mean(cross_tracks))
        reward -= 0.05 * threat_count
        reward -= 0.05 * weather_risk_count
        reward += 2.0 * newly_finished
        if collision:
            reward -= 10.0
        if weather_failure:
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
        weather_center_distance = self.weather_center_distance(state.position)
        weather_relative_bearing = self.weather_relative_bearing_deg(state)
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
                1.0 if state.weather_predicted else 0.0,
                1.0 if state.hazard_predicted else 0.0,
                float(state.safe_streak),
                -1.0 if state.predicted_pair_loss_step is None else float(state.predicted_pair_loss_step),
                -1.0 if state.predicted_weather_entry_step is None else float(state.predicted_weather_entry_step),
                999.0 if not math.isfinite(state.predicted_min_sep) else float(state.predicted_min_sep),
                999.0 if not math.isfinite(state.predicted_weather_clearance) else float(state.predicted_weather_clearance),
                999.0 if not math.isfinite(weather_center_distance) else float(weather_center_distance),
                float(weather_relative_bearing),
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
                blocks.extend([0.0] * 16)
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
                    999.0 if not math.isfinite(state.predicted_min_sep) else float(state.predicted_min_sep),
                    999.0 if not math.isfinite(state.predicted_weather_clearance) else float(state.predicted_weather_clearance),
                    -1.0 if state.predicted_pair_loss_step is None else float(state.predicted_pair_loss_step),
                    -1.0 if state.predicted_weather_entry_step is None else float(state.predicted_weather_entry_step),
                ]
            )

        if self.weather_cell is None:
            blocks.extend([0.0] * 8)
        else:
            blocks.extend(
                [
                    1.0,
                    self.weather_cell.center[0],
                    self.weather_cell.center[1],
                    self.weather_cell.major_radius_nm,
                    self.weather_cell.minor_radius_nm,
                    self.weather_cell.angle_rad,
                    self.weather_cell.motion_heading_rad,
                    self.weather_cell.speed_units_per_step,
                ]
            )
        blocks.extend([float(self.n_step), float(self.reward), float(self.total_reward)])
        return np.array(blocks, dtype=np.float64)

    def weather_dict(self) -> Optional[Dict[str, Any]]:
        if self.weather_cell is None:
            return None
        major_units, minor_units = weather_axis_units(self.weather_cell)
        return {
            "center": list(self.weather_cell.center),
            "major_radius_nm": float(self.weather_cell.major_radius_nm),
            "minor_radius_nm": float(self.weather_cell.minor_radius_nm),
            "major_radius_units": float(major_units),
            "minor_radius_units": float(minor_units),
            "minor_ratio": float(self.weather_cell.minor_ratio),
            "angle_rad": float(self.weather_cell.angle_rad),
            "motion_heading_rad": float(self.weather_cell.motion_heading_rad),
            "speed_units_per_step": float(self.weather_cell.speed_units_per_step),
            "major_growth_nm_per_step": float(self.weather_cell.major_growth_nm_per_step),
        }

    def get_infos(self) -> Dict[str, Any]:
        return {
            "global_state": self.global_state_vector(),
            "possible_agents": list(self.possible_agents),
            "active_agents": list(self.active_agent_ids),
            "episode_done": self.last_team_done,
            "episode_truncated": self.last_team_truncated,
            "episode_success": self.last_team_success,
            "team_reward": self.reward,
            "failure_reason": self.last_failure_reason,
            "max_agents_possible": len(build_route_catalog()),
            "weather": self.weather_dict(),
            "risky_pairs": copy.deepcopy(self.risky_pairs),
            "route_assignments": {
                assignment.agent_id: {
                    "route_id": assignment.route.route_id,
                    "canonical_route_id": assignment.route.canonical_route_id,
                    "is_reversed": assignment.route.is_reversed,
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
            linestyle = "--" if not route.is_reversed else "-."
            ax.plot(xs, ys, color=assignment.color, linestyle=linestyle, alpha=0.95, linewidth=1.15)
            ax.scatter(xs[0], ys[0], c=assignment.color, s=55, marker="^")
            ax.scatter(xs[-1], ys[-1], c=assignment.color, s=28, marker="*")
            ax.add_patch(Circle((xs[-1], ys[-1]), self.goal_radius, color=assignment.color, alpha=0.12))
            if state.launched:
                ax.scatter(state.position[0], state.position[1], c=assignment.color, s=45)
                ax.text(
                    state.position[0] + 0.8,
                    state.position[1] + 0.8,
                    assignment.agent_id,
                    color=assignment.color,
                    fontsize=9,
                    weight="bold",
                )

        if self.weather_cell is not None:
            major_units, minor_units = weather_axis_units(self.weather_cell)
            weather_patch = Ellipse(
                xy=self.weather_cell.center,
                width=2.0 * major_units,
                height=2.0 * minor_units,
                angle=math.degrees(self.weather_cell.angle_rad),
                edgecolor="#dc2626",
                facecolor=(220 / 255.0, 38 / 255.0, 38 / 255.0, 0.18),
                linewidth=2.0,
            )
            ax.add_patch(weather_patch)
            trail = weather_trail(self.weather_cell, WEATHER_TRAIL_STEPS)
            trail_x = [point[0] for point in trail]
            trail_y = [point[1] for point in trail]
            ax.plot(trail_x, trail_y, color="#dc2626", alpha=0.35, linewidth=1.0, linestyle=":")
            arrow_dx = 3.0 * self.weather_cell.speed_units_per_step * math.cos(self.weather_cell.motion_heading_rad)
            arrow_dy = 3.0 * self.weather_cell.speed_units_per_step * math.sin(self.weather_cell.motion_heading_rad)
            ax.arrow(
                self.weather_cell.center[0],
                self.weather_cell.center[1],
                arrow_dx,
                arrow_dy,
                color="#991b1b",
                width=0.08,
                head_width=1.5,
                length_includes_head=True,
                alpha=0.9,
            )

        failure_text = self.last_failure_reason or "-"
        ax.set_title(
            "Step {} - Reward {:.3f} - Total Reward {:.3f} - Agents {} - Failure {}".format(
                self.n_step,
                self.reward,
                self.total_reward,
                ",".join(self.active_agent_ids) or "none",
                failure_text,
            )
        )
        if show:
            plt.show()
        elif folder is not None:
            output_path = Path(folder) / f"image_{self.n_step:03d}.png"
            save_rendered_figure(fig, output_path)
            print(str(output_path))
        return fig
