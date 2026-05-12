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
    HAZARD_RISK_DECAY_EXPONENT,
    LAUNCH_SEPARATION_R,
    LOCAL_TRAFFIC_NEIGHBOR_COUNT,
    MAX_AGENTS,
    MAX_WEATHER_CELLS,
    MAX_STEP,
    NUM_WEATHER_CELLS_DEFAULT,
    PAIR_RISK_BUFFER,
    REWARD_COLLISION_PENALTY,
    REWARD_CROSS_TRACK_CLIP,
    REWARD_CROSS_TRACK_HAZARD_RELIEF,
    REWARD_CROSS_TRACK_RECOVERY_CLIP,
    REWARD_CROSS_TRACK_RECOVERY_SCALE,
    REWARD_CROSS_TRACK_SCALE,
    REWARD_DEST_HEADING_SCALE,
    REWARD_FINISHED_AIRCRAFT,
    REWARD_GREEN_WEATHER_PENETRATION_SCALE,
    REWARD_MERGE_BACK_RISK_THRESHOLD,
    REWARD_MERGE_BACK_SCALE,
    REWARD_PROGRESS_CLIP,
    REWARD_PROGRESS_HAZARD_RELIEF,
    REWARD_PROGRESS_SCALE,
    REWARD_RISK_REDUCTION_CLIP,
    REWARD_STEP_PENALTY,
    REWARD_TEAM_SUCCESS,
    REWARD_TERMINAL_WEATHER_PENALTY,
    REWARD_TRAFFIC_RISK_REDUCTION_SCALE,
    REWARD_TRUNCATION_PENALTY,
    REWARD_TRUNCATION_DISTANCE_PENALTY_SCALE,
    REWARD_TRAFFIC_RISK_PENALTY,
    REWARD_WEATHER_RISK_REDUCTION_SCALE,
    REWARD_WEATHER_RISK_PENALTY,
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
    weather_terminal_signed_clearance,
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

    local_neighbor_feature_dim = 6
    local_obs_dim = 22 + LOCAL_TRAFFIC_NEIGHBOR_COUNT * local_neighbor_feature_dim

    def __init__(
        self,
        *,
        num_agents: int = 2,
        max_agents: int = MAX_AGENTS,
        safe_r: float = SAFE_R,
        goal_radius: float = GOAL_RADIUS,
        max_step: int = MAX_STEP,
        speed: float = AGENT_SPEED,
        num_weather_cells: int = NUM_WEATHER_CELLS_DEFAULT,
    ) -> None:
        self.num_agents = int(num_agents)
        self.max_agents = int(max_agents)
        self.safe_r = float(safe_r)
        self.goal_radius = float(goal_radius)
        self.max_step = int(max_step)
        self.speed = float(speed)
        self.num_weather_cells = self._validate_num_weather_cells(num_weather_cells)
        self.weather_clearance_buffer_units = nm_to_grid_mean(WEATHER_CLEARANCE_BUFFER_NM)
        self.sector_polygon = load_sector_polygon()
        self.possible_agents = possible_agent_ids(self.max_agents)

        self.assignments: List[AssignedRoute] = []
        self.agent_states: Dict[str, AgentState] = {}
        self.weather_cells: List[WeatherCell] = []
        self.weather_cell: Optional[WeatherCell] = None
        self.n_step = 0
        self.reward = 0.0
        self.total_reward = 0.0
        self.agent_rewards: Dict[str, float] = {}
        self.agent_dense_rewards: Dict[str, float] = {}
        self.agent_terminal_rewards: Dict[str, float] = {}
        self.last_reward_components: Dict[str, Dict[str, float]] = {}
        self.last_terminal_reward = 0.0
        self.last_team_done = False
        self.last_team_truncated = False
        self.last_team_success = False
        self.last_failure_reason: Optional[str] = None
        self.last_pairwise_separations: Dict[Tuple[str, str], float] = {}
        self.risky_pairs: List[Dict[str, Any]] = []
        self.rng = random.Random()

    @staticmethod
    def _validate_num_weather_cells(value: int) -> int:
        num_cells = int(value)
        if num_cells not in (1, 2):
            raise ValueError(f"num_weather_cells must be 1 or 2, got {value}")
        return num_cells

    def _sync_primary_weather_cell(self) -> None:
        self.weather_cell = self.weather_cells[0] if self.weather_cells else None

    def clone(self) -> "MultiAgentSectorCore":
        return copy.deepcopy(self)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        num_agents: Optional[int] = None,
        route_ids: Optional[Sequence[str]] = None,
        num_weather_cells: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if num_agents is not None:
            self.num_agents = int(num_agents)
        if num_weather_cells is not None:
            self.num_weather_cells = self._validate_num_weather_cells(num_weather_cells)
        self.rng = random.Random(seed)
        self.assignments = assign_routes(self.num_agents, seed=seed, route_ids=route_ids)
        self.agent_states = {}
        self.weather_cells = []
        self.weather_cell = None
        self.n_step = 0
        self.reward = 0.0
        self.total_reward = 0.0
        self.agent_rewards = {}
        self.agent_dense_rewards = {}
        self.agent_terminal_rewards = {}
        self.last_reward_components = {}
        self.last_terminal_reward = 0.0
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

        self.weather_cells = self._init_weather_cells()
        self._sync_primary_weather_cell()
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

    def _launch_position_clear(self, state: AgentState) -> bool:
        launch_position = np.array(state.route.points[0], dtype=float)
        for other in self.agent_states.values():
            if other.agent_id == state.agent_id or not other.launched or other.finished:
                continue
            distance = float(np.linalg.norm(launch_position - np.array(other.position, dtype=float)))
            if distance < float(LAUNCH_SEPARATION_R):
                return False
        return True

    def _launch_agents(self) -> None:
        for state in self.agent_states.values():
            if not state.launched and self.n_step >= state.start_step:
                if not self._launch_position_clear(state):
                    state.start_step = self.n_step + 1
                    state.last_event = f"launch_delayed@{self.n_step}"
                    continue
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

    def heading_error_to_destination(self, state: AgentState) -> float:
        if self.distance_to_destination(state) <= self.goal_radius:
            return 0.0
        dest_heading = math.atan2(
            state.destination[1] - state.position[1],
            state.destination[0] - state.position[0],
        )
        return abs(float(self._wrap_angle(dest_heading - state.heading_rad)))

    def nearest_neighbor(self, agent_id: str) -> Tuple[Optional[str], float]:
        neighbors = self.nearest_neighbors(agent_id, limit=1)
        if not neighbors:
            return None, float("inf")
        return neighbors[0]

    def nearest_neighbors(self, agent_id: str, *, limit: int = LOCAL_TRAFFIC_NEIGHBOR_COUNT) -> List[Tuple[str, float]]:
        state = self.agent_states[agent_id]
        neighbors: List[Tuple[str, float]] = []
        for other_id in self.active_agent_ids:
            if other_id == agent_id:
                continue
            other_state = self.agent_states[other_id]
            dist = float(np.linalg.norm(np.array(state.position) - np.array(other_state.position)))
            neighbors.append((other_id, dist))
        neighbors.sort(key=lambda item: item[1])
        return neighbors[: max(int(limit), 0)]

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

    def _relative_heading_to_neighbor_deg(self, agent_id: str, other_id: Optional[str]) -> float:
        if other_id is None:
            return 0.0
        state = self.agent_states[agent_id]
        other = self.agent_states[other_id]
        own_deg = math.degrees(state.heading_rad)
        other_deg = math.degrees(other.heading_rad)
        return ((other_deg - own_deg + 180.0) % 360.0) - 180.0

    def _neighbor_proximity_risk(self, distance: float) -> float:
        caution_distance = max(self.safe_r * PAIR_RISK_BUFFER, 1e-6)
        return float(np.clip((caution_distance - float(distance)) / caution_distance, 0.0, 1.0))

    def _closing_risk_to_neighbor(self, agent_id: str, other_id: Optional[str]) -> float:
        if other_id is None:
            return 0.0
        state = self.agent_states[agent_id]
        other = self.agent_states[other_id]
        relative_position = np.array(other.position, dtype=float) - np.array(state.position, dtype=float)
        distance = float(np.linalg.norm(relative_position))
        if distance <= 1e-6:
            return 1.0
        line_of_sight = relative_position / distance
        own_velocity = self.speed * np.array(
            [math.cos(state.heading_rad), math.sin(state.heading_rad)],
            dtype=float,
        )
        other_velocity = self.speed * np.array(
            [math.cos(other.heading_rad), math.sin(other.heading_rad)],
            dtype=float,
        )
        closing_speed = float(np.dot(own_velocity - other_velocity, line_of_sight))
        max_closing_speed = max(2.0 * float(self.speed), 1e-6)
        return float(np.clip(closing_speed / max_closing_speed, 0.0, 1.0))

    def _local_neighbor_risk(self, distance: float, closing_risk: float) -> float:
        proximity_risk = self._neighbor_proximity_risk(distance)
        closing_boost = 1.0 + 0.5 * float(np.clip(closing_risk, 0.0, 1.0))
        return float(np.clip(proximity_risk * closing_boost, 0.0, 1.0))

    def _local_traffic_risk_score(self, agent_id: str) -> float:
        risks = [
            self._local_neighbor_risk(distance, self._closing_risk_to_neighbor(agent_id, other_id))
            for other_id, distance in self.nearest_neighbors(agent_id, limit=LOCAL_TRAFFIC_NEIGHBOR_COUNT)
        ]
        return max(risks) if risks else 0.0

    def _local_neighbor_features(self, agent_id: str) -> List[float]:
        features: List[float] = []
        caution_distance = max(self.safe_r * PAIR_RISK_BUFFER, 1e-6)
        padded_distance = 2.0 * caution_distance
        for other_id, distance in self.nearest_neighbors(agent_id, limit=LOCAL_TRAFFIC_NEIGHBOR_COUNT):
            proximity_risk = self._neighbor_proximity_risk(distance)
            closing_risk = self._closing_risk_to_neighbor(agent_id, other_id)
            features.extend(
                [
                    float(distance),
                    self._bearing_to_neighbor_deg(agent_id, other_id),
                    self._relative_heading_to_neighbor_deg(agent_id, other_id),
                    1.0 if float(distance) <= caution_distance else 0.0,
                    proximity_risk,
                    closing_risk,
                ]
            )
        missing = LOCAL_TRAFFIC_NEIGHBOR_COUNT - len(features) // self.local_neighbor_feature_dim
        for _ in range(max(missing, 0)):
            features.extend([padded_distance, 0.0, 0.0, 0.0, 0.0, 0.0])
        return features

    def boundary_distance(self, state: AgentState) -> Optional[float]:
        if not state.inside_sector:
            return None
        return float(self.sector_polygon.exterior.distance(Point(state.position)))

    def weather_signed_clearance(self, position: Tuple[float, float]) -> float:
        if not self.weather_cells:
            return float("inf")
        return min(weather_signed_clearance(cell, position) for cell in self.weather_cells)

    def weather_terminal_signed_clearance(self, position: Tuple[float, float]) -> float:
        if not self.weather_cells:
            return float("inf")
        return min(weather_terminal_signed_clearance(cell, position) for cell in self.weather_cells)

    def weather_center_distance(self, position: Tuple[float, float]) -> float:
        if not self.weather_cells:
            return float("inf")
        return min(
            float(np.linalg.norm(np.array(position) - np.array(cell.center)))
            for cell in self.weather_cells
        )

    def weather_relative_bearing_deg(self, state: AgentState) -> float:
        if not self.weather_cells:
            return 0.0
        nearest = min(
            self.weather_cells,
            key=lambda cell: float(np.linalg.norm(np.array(state.position) - np.array(cell.center))),
        )
        absolute = math.degrees(
            math.atan2(
                nearest.center[1] - state.position[1],
                nearest.center[0] - state.position[0],
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

    def _init_weather_cells(self) -> List[WeatherCell]:
        shared_motion_heading_rad = self.rng.uniform(-math.pi, math.pi)
        return [
            self._init_weather_cell(
                cell_id=f"W{idx + 1}",
                motion_heading_rad=shared_motion_heading_rad,
            )
            for idx in range(self.num_weather_cells)
        ]

    def _init_weather(self) -> WeatherCell:
        return self._init_weather_cell(
            cell_id="W1",
            motion_heading_rad=self.rng.uniform(-math.pi, math.pi),
        )

    def _init_weather_cell(self, *, cell_id: str, motion_heading_rad: float) -> WeatherCell:
        launches = [state.position for state in self.agent_states.values()]
        destinations = [tuple(assignment.route.points[-1]) for assignment in self.assignments]
        min_x, min_y, max_x, max_y = self.sector_polygon.bounds

        for _ in range(WEATHER_INIT_TRIES):
            major_radius_nm = self.rng.uniform(WEATHER_MAJOR_RADIUS_NM_MIN, WEATHER_MAJOR_RADIUS_NM_MAX)
            minor_ratio = self.rng.uniform(WEATHER_MINOR_RATIO_MIN, WEATHER_MINOR_RATIO_MAX)
            angle_rad = self.rng.uniform(0.0, math.pi)
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
                cell_id=cell_id,
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

    def _weather_destinations_clear(self, cell: WeatherCell) -> bool:
        destinations = [tuple(assignment.route.points[-1]) for assignment in self.assignments]
        return all(
            weather_polygon(cell).distance(Point(dest)) > self.goal_radius
            for dest in destinations
        )

    def _next_weather_size(self, cell: WeatherCell) -> Tuple[float, float, bool]:
        if cell.growth_stopped or abs(float(cell.major_growth_nm_per_step)) <= 1e-9:
            return float(cell.major_radius_nm), 0.0, bool(cell.growth_stopped)
        growth_nm = float(cell.major_growth_nm_per_step)
        next_major_nm = float(cell.major_radius_nm) + growth_nm
        if next_major_nm >= WEATHER_MAJOR_RADIUS_NM_MAX:
            return float(WEATHER_MAJOR_RADIUS_NM_MAX), 0.0, True
        if next_major_nm <= WEATHER_MAJOR_RADIUS_NM_MIN:
            return float(WEATHER_MAJOR_RADIUS_NM_MIN), 0.0, True
        return float(next_major_nm), growth_nm, False

    def _advance_weather_cell(self, cell: WeatherCell) -> WeatherCell:
        current = copy.deepcopy(cell)
        if current.movement_stopped:
            current.speed_units_per_step = 0.0
            current.major_growth_nm_per_step = 0.0
            current.growth_stopped = True
            return current

        next_major_nm, next_growth_nm, growth_stopped = self._next_weather_size(current)
        proposed_center = (
            current.center[0] + current.speed_units_per_step * math.cos(current.motion_heading_rad),
            current.center[1] + current.speed_units_per_step * math.sin(current.motion_heading_rad),
        )
        proposed = WeatherCell(
            center=proposed_center,
            major_radius_nm=next_major_nm,
            minor_ratio=current.minor_ratio,
            angle_rad=current.angle_rad,
            motion_heading_rad=current.motion_heading_rad,
            speed_units_per_step=current.speed_units_per_step,
            major_growth_nm_per_step=next_growth_nm,
            cell_id=current.cell_id,
            growth_stopped=growth_stopped,
            movement_stopped=False,
        )
        if self._weather_geometry_valid(proposed) and self._weather_destinations_clear(proposed):
            return proposed

        current.speed_units_per_step = 0.0
        current.major_growth_nm_per_step = 0.0
        current.growth_stopped = True
        current.movement_stopped = True
        return current

    def _advance_weather(self) -> None:
        if not self.weather_cells:
            if self.weather_cell is not None:
                self.weather_cells = [self.weather_cell]
            else:
                return
        self.weather_cells = [self._advance_weather_cell(cell) for cell in self.weather_cells]
        self._sync_primary_weather_cell()

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

    def _lookahead_risk(self, step: Optional[int], horizon_steps: int = HAZARD_LOOKAHEAD_STEPS) -> float:
        if step is None:
            return 0.0
        horizon = max(float(horizon_steps), 1.0)
        linear_risk = float(np.clip(1.0 - ((float(step) - 1.0) / horizon), 0.0, 1.0))
        exponent = max(float(HAZARD_RISK_DECAY_EXPONENT), 1e-6)
        return float(linear_risk**exponent)

    def _traffic_risk_parts(self, state: AgentState, collision_agents: set[str]) -> Dict[str, float]:
        step_risk = self._lookahead_risk(state.predicted_pair_loss_step)
        sep_risk = 0.0
        if math.isfinite(state.predicted_min_sep):
            caution_distance = max(self.safe_r * PAIR_RISK_BUFFER, 1e-6)
            sep_risk = float(np.clip((caution_distance - state.predicted_min_sep) / caution_distance, 0.0, 1.0))
        predicted_risk = max(step_risk, sep_risk)
        local_risk = self._local_traffic_risk_score(state.agent_id)
        traffic_risk = max(predicted_risk, local_risk)
        if state.agent_id in collision_agents:
            predicted_risk = 1.0
            local_risk = 1.0
            traffic_risk = 1.0
        return {
            "traffic_step_risk": float(step_risk),
            "traffic_separation_risk": float(sep_risk),
            "predicted_traffic_risk": float(predicted_risk),
            "local_traffic_risk": float(local_risk),
            "traffic_risk": float(traffic_risk),
        }

    def _traffic_risk_score(self, state: AgentState, collision_agents: set[str]) -> float:
        return self._traffic_risk_parts(state, collision_agents)["traffic_risk"]

    def _weather_risk_score(self, state: AgentState) -> float:
        terminal_clearance = self.weather_terminal_signed_clearance(state.position)
        if terminal_clearance < 0.0:
            return 1.0

        step_risk = self._lookahead_risk(state.predicted_weather_entry_step)
        buffer = max(float(self.weather_clearance_buffer_units), 1e-6)
        clearance_risk = 0.0
        if math.isfinite(state.predicted_weather_clearance):
            clearance_risk = float(np.clip((buffer - state.predicted_weather_clearance) / buffer, 0.0, 1.0))

        green_risk = 0.0
        green_clearance = self.weather_signed_clearance(state.position)
        if green_clearance < 0.0:
            green_risk = float(
                np.clip(
                    REWARD_GREEN_WEATHER_PENETRATION_SCALE * abs(float(green_clearance)) / buffer,
                    0.0,
                    1.0,
                )
            )
        return max(step_risk, clearance_risk, green_risk)

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
        previously_finished = {
            agent_id
            for agent_id, state in self.agent_states.items()
            if state.finished
        }
        previous_distances = {
            agent_id: self.distance_to_destination(state)
            for agent_id, state in self.agent_states.items()
        }
        previous_cross_tracks = {
            agent_id: abs(float(line_signed_cross_track(state.route.linestring, state.position)[0]))
            for agent_id, state in self.agent_states.items()
        }
        previous_reward_agent_ids = [
            agent_id
            for agent_id, state in self.agent_states.items()
            if state.launched and not state.finished
        ]
        previous_traffic_parts = {
            agent_id: self._traffic_risk_parts(self.agent_states[agent_id], set())
            for agent_id in previous_reward_agent_ids
        }
        previous_traffic_risks = {
            agent_id: parts["traffic_risk"]
            for agent_id, parts in previous_traffic_parts.items()
        }
        previous_weather_risks = {
            agent_id: self._weather_risk_score(self.agent_states[agent_id])
            for agent_id in previous_reward_agent_ids
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
        collision_pairs = {
            pair
            for pair, distance in pairwise.items()
            if distance < self.safe_r
        }
        collision_agents = {agent_id for pair in collision_pairs for agent_id in pair}
        collision = bool(collision_pairs)
        weather_violation_agents = [
            agent_id
            for agent_id in active_ids
            if self.weather_terminal_signed_clearance(self.agent_states[agent_id].position) < 0.0
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

        reward_agent_ids = [
            agent_id
            for agent_id, state in self.agent_states.items()
            if state.launched and agent_id not in previously_finished
        ]
        dense_rewards: Dict[str, float] = {}
        reward_components: Dict[str, Dict[str, float]] = {}
        for state in self.agent_states.values():
            if state.agent_id not in reward_agent_ids:
                continue
            raw_progress = previous_distances[state.agent_id] - self.distance_to_destination(state)
            progress = float(
                np.clip(
                    raw_progress / max(float(self.speed), 1e-6),
                    -REWARD_PROGRESS_CLIP,
                    REWARD_PROGRESS_CLIP,
                )
            )
            signed_xtrk, _, _ = line_signed_cross_track(state.route.linestring, state.position)
            cross_track = float(
                np.clip(
                    abs(float(signed_xtrk)) / max(float(self.safe_r), 1e-6),
                    0.0,
                    REWARD_CROSS_TRACK_CLIP,
                )
            )
            traffic_parts = self._traffic_risk_parts(state, collision_agents)
            traffic_risk = traffic_parts["traffic_risk"]
            weather_risk = self._weather_risk_score(state)
            previous_traffic_part = previous_traffic_parts.get(state.agent_id, {})
            previous_traffic_risk = float(previous_traffic_risks.get(state.agent_id, 0.0))
            previous_weather_risk = float(previous_weather_risks.get(state.agent_id, 0.0))
            combined_traffic_risk_reduction = float(
                np.clip(
                    previous_traffic_risk - traffic_risk,
                    0.0,
                    REWARD_RISK_REDUCTION_CLIP,
                )
            )
            local_traffic_risk_reduction = float(
                np.clip(
                    float(previous_traffic_part.get("local_traffic_risk", 0.0))
                    - float(traffic_parts["local_traffic_risk"]),
                    0.0,
                    REWARD_RISK_REDUCTION_CLIP,
                )
            )
            traffic_risk_reduction = max(combined_traffic_risk_reduction, local_traffic_risk_reduction)
            weather_risk_reduction = float(
                np.clip(
                    previous_weather_risk - weather_risk,
                    0.0,
                    REWARD_RISK_REDUCTION_CLIP,
                )
            )
            progress_risk_gate = float(np.clip(max(traffic_risk, weather_risk), 0.0, 1.0))
            cross_track_penalty_scale = float(
                np.clip(
                    1.0 - REWARD_CROSS_TRACK_HAZARD_RELIEF * progress_risk_gate,
                    0.0,
                    1.0,
                )
            )
            relaxed_cross_track = cross_track * cross_track_penalty_scale
            progress_safety_scale = float(
                np.clip(
                    1.0 - REWARD_PROGRESS_HAZARD_RELIEF * progress_risk_gate,
                    0.0,
                    1.0,
                )
            )
            recovery_safety_scale = 1.0 - progress_risk_gate
            safe_progress = progress * progress_safety_scale
            raw_cross_track_recovery = previous_cross_tracks[state.agent_id] - abs(float(signed_xtrk))
            cross_track_recovery = float(
                np.clip(
                    raw_cross_track_recovery / max(float(self.safe_r), 1e-6),
                    0.0,
                    REWARD_CROSS_TRACK_RECOVERY_CLIP,
                )
            )
            safe_cross_track_recovery = cross_track_recovery * recovery_safety_scale
            merge_back_scale = float(
                np.clip(
                    (REWARD_MERGE_BACK_RISK_THRESHOLD - progress_risk_gate)
                    / max(float(REWARD_MERGE_BACK_RISK_THRESHOLD), 1e-6),
                    0.0,
                    1.0,
                )
            )
            merge_back_bonus = REWARD_MERGE_BACK_SCALE * merge_back_scale * cross_track_recovery
            heading_error = float(self.heading_error_to_destination(state))
            heading_error_norm = float(np.clip(heading_error / math.pi, 0.0, 1.0))
            safe_heading_penalty = heading_error_norm * recovery_safety_scale
            finish_bonus = REWARD_FINISHED_AIRCRAFT if state.just_finished else 0.0
            dense_reward = (
                REWARD_PROGRESS_SCALE * safe_progress
                - REWARD_STEP_PENALTY
                - REWARD_CROSS_TRACK_SCALE * relaxed_cross_track
                + REWARD_CROSS_TRACK_RECOVERY_SCALE * safe_cross_track_recovery
                - REWARD_DEST_HEADING_SCALE * safe_heading_penalty
                - REWARD_TRAFFIC_RISK_PENALTY * traffic_risk
                - REWARD_WEATHER_RISK_PENALTY * weather_risk
                + REWARD_TRAFFIC_RISK_REDUCTION_SCALE * traffic_risk_reduction
                + REWARD_WEATHER_RISK_REDUCTION_SCALE * weather_risk_reduction
                + merge_back_bonus
                + finish_bonus
            )
            dense_rewards[state.agent_id] = float(dense_reward)
            reward_components[state.agent_id] = {
                "raw_progress": float(raw_progress),
                "progress": float(progress),
                "safe_progress": float(safe_progress),
                "progress_risk_gate": float(progress_risk_gate),
                "progress_safety_scale": float(progress_safety_scale),
                "cross_track": float(cross_track),
                "cross_track_penalty_scale": float(cross_track_penalty_scale),
                "relaxed_cross_track": float(relaxed_cross_track),
                "raw_cross_track_recovery": float(raw_cross_track_recovery),
                "cross_track_recovery": float(cross_track_recovery),
                "safe_cross_track_recovery": float(safe_cross_track_recovery),
                "merge_back_scale": float(merge_back_scale),
                "merge_back_bonus": float(merge_back_bonus),
                "heading_error_to_destination": float(math.degrees(heading_error)),
                "heading_error_norm": float(heading_error_norm),
                "safe_heading_penalty": float(safe_heading_penalty),
                "previous_traffic_risk": float(previous_traffic_risk),
                "previous_local_traffic_risk": float(previous_traffic_part.get("local_traffic_risk", 0.0)),
                "traffic_step_risk": float(traffic_parts["traffic_step_risk"]),
                "traffic_separation_risk": float(traffic_parts["traffic_separation_risk"]),
                "predicted_traffic_risk": float(traffic_parts["predicted_traffic_risk"]),
                "local_traffic_risk": float(traffic_parts["local_traffic_risk"]),
                "traffic_risk": float(traffic_risk),
                "combined_traffic_risk_reduction": float(combined_traffic_risk_reduction),
                "local_traffic_risk_reduction": float(local_traffic_risk_reduction),
                "traffic_risk_reduction": float(traffic_risk_reduction),
                "previous_weather_risk": float(previous_weather_risk),
                "weather_risk": float(weather_risk),
                "weather_risk_reduction": float(weather_risk_reduction),
                "finish_bonus": float(finish_bonus),
                "dense_reward": float(dense_reward),
            }

        terminal_agent_ids = [
            agent_id
            for agent_id, state in self.agent_states.items()
            if state.launched
        ]
        terminal_rewards = {agent_id: 0.0 for agent_id in terminal_agent_ids}
        shared_scale = math.sqrt(max(float(self.num_agents), 1.0))
        if collision:
            shared_penalty = -REWARD_COLLISION_PENALTY / shared_scale
            terminal_rewards = {agent_id: shared_penalty for agent_id in terminal_agent_ids}
            for agent_id in collision_agents:
                if agent_id in terminal_rewards:
                    terminal_rewards[agent_id] = -REWARD_COLLISION_PENALTY
        elif weather_failure:
            shared_penalty = -REWARD_TERMINAL_WEATHER_PENALTY / shared_scale
            terminal_rewards = {agent_id: shared_penalty for agent_id in terminal_agent_ids}
            for agent_id in weather_violation_agents:
                if agent_id in terminal_rewards:
                    terminal_rewards[agent_id] = -REWARD_TERMINAL_WEATHER_PENALTY
        elif team_truncated:
            shared_penalty = -REWARD_TRUNCATION_PENALTY / shared_scale
            terminal_rewards = {agent_id: shared_penalty for agent_id in terminal_agent_ids}
            for agent_id, state in self.agent_states.items():
                if agent_id in terminal_rewards and state.launched and not state.finished:
                    route_length = max(float(state.route.linestring.length), self.goal_radius, 1e-6)
                    remaining_fraction = float(
                        np.clip(self.distance_to_destination(state) / route_length, 0.0, 1.0)
                    )
                    distance_penalty = REWARD_TRUNCATION_DISTANCE_PENALTY_SCALE * remaining_fraction
                    terminal_rewards[agent_id] = -REWARD_TRUNCATION_PENALTY - distance_penalty
        elif team_success:
            terminal_rewards = {
                agent_id: REWARD_TEAM_SUCCESS
                for agent_id in terminal_agent_ids
            }

        terminal_reward = (
            float(np.mean(list(terminal_rewards.values())))
            if terminal_rewards
            else 0.0
        )
        reward_scope_ids = (
            terminal_agent_ids
            if team_done or team_truncated
            else list(dense_rewards)
        )

        agent_rewards = {
            agent_id: float(
                dense_rewards.get(agent_id, 0.0)
                + terminal_rewards.get(agent_id, 0.0)
            )
            for agent_id in reward_scope_ids
        }
        reward = (
            float(np.mean(list(agent_rewards.values())))
            if agent_rewards
            else float(terminal_reward)
        )

        self.reward = reward
        self.total_reward += reward
        self.agent_rewards = agent_rewards
        self.agent_dense_rewards = dense_rewards
        self.agent_terminal_rewards = terminal_rewards
        self.last_reward_components = reward_components
        self.last_terminal_reward = float(terminal_reward)
        self.last_team_done = team_done
        self.last_team_truncated = team_truncated
        self.last_team_success = team_success

        observations = self.get_active_observations()
        rewards = {
            agent_id: float(agent_rewards.get(agent_id, reward))
            for agent_id in reward_agent_ids
        }
        terminations = {
            agent_id: bool(team_done or self.agent_states[agent_id].finished)
            for agent_id in reward_agent_ids
        }
        truncations = {agent_id: team_truncated for agent_id in reward_agent_ids}
        infos = self.get_infos()
        return observations, rewards, terminations, truncations, infos

    def get_local_observation(self, agent_id: str) -> np.ndarray:
        state = self.agent_states[agent_id]
        route_heading_deg = math.degrees(heading_from_line(state.route.linestring, state.route_progress))
        own_heading_deg = math.degrees(state.heading_rad)
        heading_error_to_route = ((route_heading_deg - own_heading_deg + 180.0) % 360.0) - 180.0
        signed_xtrk, _, _ = line_signed_cross_track(state.route.linestring, state.position)
        neighbor_features = self._local_neighbor_features(agent_id)
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
                *neighbor_features,
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

        for idx in range(MAX_WEATHER_CELLS):
            cell = self.weather_cells[idx] if idx < len(self.weather_cells) else None
            if cell is None:
                blocks.extend([0.0] * 8)
                continue
            blocks.extend(
                [
                    1.0,
                    cell.center[0],
                    cell.center[1],
                    cell.major_radius_nm,
                    cell.minor_radius_nm,
                    cell.angle_rad,
                    cell.motion_heading_rad,
                    cell.speed_units_per_step,
                ]
            )
        blocks.extend([float(self.n_step), float(self.reward), float(self.total_reward)])
        return np.array(blocks, dtype=np.float64)

    def weather_dict(self) -> Optional[Dict[str, Any]]:
        if not self.weather_cells:
            return None

        def cell_dict(cell: WeatherCell) -> Dict[str, Any]:
            major_units, minor_units = weather_axis_units(cell)
            return {
                "cell_id": str(cell.cell_id),
                "center": list(cell.center),
                "major_radius_nm": float(cell.major_radius_nm),
                "minor_radius_nm": float(cell.minor_radius_nm),
                "major_radius_units": float(major_units),
                "minor_radius_units": float(minor_units),
                "minor_ratio": float(cell.minor_ratio),
                "angle_rad": float(cell.angle_rad),
                "motion_heading_rad": float(cell.motion_heading_rad),
                "speed_units_per_step": float(cell.speed_units_per_step),
                "major_growth_nm_per_step": float(cell.major_growth_nm_per_step),
                "growth_stopped": bool(cell.growth_stopped),
                "movement_stopped": bool(cell.movement_stopped),
            }

        cells = [cell_dict(cell) for cell in self.weather_cells]
        primary = cells[0]
        return {
            **primary,
            "cells": cells,
            "num_weather_cells": len(cells),
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
            "agent_rewards": dict(self.agent_rewards),
            "agent_dense_rewards": dict(self.agent_dense_rewards),
            "agent_terminal_rewards": dict(self.agent_terminal_rewards),
            "terminal_reward": float(self.last_terminal_reward),
            "reward_components": copy.deepcopy(self.last_reward_components),
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

        weather_severity_bands = [
            (1.00, "#22c55e", (34 / 255.0, 197 / 255.0, 94 / 255.0, 0.24)),
            (0.75, "#facc15", (250 / 255.0, 204 / 255.0, 21 / 255.0, 0.34)),
            (0.50, "#dc2626", (220 / 255.0, 38 / 255.0, 38 / 255.0, 0.44)),
            (0.25, "#d946ef", (217 / 255.0, 70 / 255.0, 239 / 255.0, 0.56)),
        ]
        arrow_color = "#86198f"
        trail_color = "#166534"
        for cell in self.weather_cells:
            major_units, minor_units = weather_axis_units(cell)
            for scale, edge_color, face_color in weather_severity_bands:
                weather_patch = Ellipse(
                    xy=cell.center,
                    width=2.0 * major_units * scale,
                    height=2.0 * minor_units * scale,
                    angle=math.degrees(cell.angle_rad),
                    edgecolor=edge_color,
                    facecolor=face_color,
                    linewidth=1.6 if scale < 1.0 else 2.0,
                )
                ax.add_patch(weather_patch)
            if cell.movement_stopped or abs(float(cell.speed_units_per_step)) <= 1e-9:
                ax.scatter(cell.center[0], cell.center[1], c=arrow_color, s=28, marker="x")
                continue
            trail = weather_trail(cell, WEATHER_TRAIL_STEPS)
            future_centers = trail[1:]
            future_count = max(len(future_centers), 1)
            for future_idx, future_center in enumerate(future_centers, start=1):
                alpha = max(0.10, 0.38 * (1.0 - (future_idx - 1) / future_count))
                ghost_patch = Ellipse(
                    xy=future_center,
                    width=2.0 * major_units,
                    height=2.0 * minor_units,
                    angle=math.degrees(cell.angle_rad),
                    edgecolor=trail_color,
                    facecolor="none",
                    linewidth=1.2,
                    linestyle="--",
                    alpha=alpha,
                )
                ax.add_patch(ghost_patch)

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
