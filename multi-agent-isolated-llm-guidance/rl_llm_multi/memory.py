"""Compact route-gated decision memory for global LLM guidance."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from shapely.geometry import Point
from shapely.ops import nearest_points

from configs import (
    BOUNDARY_WARNING_DISTANCE_UNITS,
    DECISION_MEMORY_MAX_EPISODES,
    DECISION_MEMORY_MIN_SIMILARITY,
    SAFE_R,
    TRAFFIC_CAUTION_R,
)
from .utils import clip_position, line_signed_cross_track


SCHEMA_VERSION = 1


def _stable_case_id(episode_id: str, case: Mapping[str, Any]) -> str:
    existing = case.get("case_id")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    identity = {
        "episode_id": str(episode_id or ""),
        "step": _int_or_none(case.get("step")),
        "agent": str(case.get("agent", "")),
        "route": str(case.get("route", "")),
        "call": str(case.get("call", "")),
        "type": str(case.get("type", "")),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"case_{hashlib.sha1(encoded.encode('utf-8')).hexdigest()[:16]}"


def _wrap_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _round_or_none(value: Any, digits: int = 3) -> Optional[float]:
    number = _finite_float(value)
    return None if number is None else round(number, int(digits))


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _action_or_none(value: Any) -> Optional[int]:
    action = _int_or_none(value)
    if action is None:
        return None
    return int(action)


def _bool_or_false(value: Any) -> bool:
    return bool(value)


def _route_fingerprint_from_state(state: Any) -> str:
    route = getattr(state, "route", None)
    waypoint_names = tuple(getattr(route, "waypoint_names", ()) or ())
    origin = str(getattr(route, "origin", "") or (waypoint_names[0] if waypoint_names else "unknown"))
    destination = str(
        getattr(route, "destination", "") or (waypoint_names[-1] if waypoint_names else "unknown")
    )
    route_id = str(
        getattr(route, "route_id", "")
        or getattr(route, "canonical_route_id", "")
        or f"{origin}->{destination}"
    )
    is_reversed = int(bool(getattr(route, "is_reversed", False)))
    return f"{route_id}|{origin}|{destination}|{is_reversed}"


def _heading_error_to_destination(state: Any) -> float:
    dest_heading = math.degrees(
        math.atan2(
            float(state.destination[1]) - float(state.position[1]),
            float(state.destination[0]) - float(state.position[0]),
        )
    )
    return _wrap_deg(dest_heading - math.degrees(float(state.heading_rad)))


def _relative_bearing_deg(
    *,
    origin: Tuple[float, float],
    heading_rad: float,
    target: Tuple[float, float],
) -> float:
    absolute = math.degrees(
        math.atan2(float(target[1]) - float(origin[1]), float(target[0]) - float(origin[0]))
    )
    return _wrap_deg(absolute - math.degrees(float(heading_rad)))


def _project_no_turn_point(core: Any, state: Any) -> Tuple[float, float]:
    speed = float(getattr(core, "speed", 0.0) or 0.0)
    x_val = float(state.position[0]) + speed * math.cos(float(state.heading_rad))
    y_val = float(state.position[1]) + speed * math.sin(float(state.heading_rad))
    return clip_position(x_val, y_val)


def _signed_boundary_distance(core: Any, point: Tuple[float, float]) -> Optional[float]:
    polygon = getattr(core, "sector_polygon", None)
    if polygon is None:
        return None
    cur = Point(point)
    distance = float(polygon.exterior.distance(cur))
    return distance if polygon.covers(cur) else -distance


def _boundary_bearing(core: Any, state: Any) -> Optional[float]:
    polygon = getattr(core, "sector_polygon", None)
    if polygon is None:
        return None
    try:
        _, boundary_point = nearest_points(Point(state.position), polygon.exterior)
    except Exception:
        return None
    return _relative_bearing_deg(
        origin=tuple(state.position),
        heading_rad=float(state.heading_rad),
        target=(float(boundary_point.x), float(boundary_point.y)),
    )


def _nearest_weather_center(core: Any, state: Any) -> Optional[Tuple[float, float]]:
    cells = list(getattr(core, "weather_cells", []) or [])
    centers: List[Tuple[float, float]] = []
    for cell in cells:
        center = getattr(cell, "center", None)
        if center is not None:
            centers.append((float(center[0]), float(center[1])))
    if not centers:
        try:
            weather = core.weather_dict()
        except Exception:
            weather = None
        if isinstance(weather, dict):
            raw_cells = weather.get("cells") if isinstance(weather.get("cells"), list) else [weather]
            for raw in raw_cells:
                if isinstance(raw, dict) and isinstance(raw.get("center"), (list, tuple)):
                    center = raw["center"]
                    centers.append((float(center[0]), float(center[1])))
    if not centers:
        return None
    own = tuple(float(value) for value in state.position)
    return min(centers, key=lambda center: math.hypot(center[0] - own[0], center[1] - own[1]))


def _weather_signature(core: Any, state: Any) -> Dict[str, Any]:
    clear_now = _finite_float(core.weather_signed_clearance(tuple(state.position)))
    clear_pred = _finite_float(getattr(state, "predicted_weather_clearance", None))
    projected = _project_no_turn_point(core, state)
    projected_clearance = _finite_float(core.weather_signed_clearance(projected))
    center = _nearest_weather_center(core, state)
    bearing = (
        None
        if center is None
        else _relative_bearing_deg(
            origin=tuple(state.position),
            heading_rad=float(state.heading_rad),
            target=center,
        )
    )
    closing = (
        False
        if clear_now is None or projected_clearance is None
        else bool(projected_clearance < clear_now)
    )
    return {
        "clear_now": _round_or_none(clear_now),
        "clear_pred": _round_or_none(clear_pred),
        "bearing": _round_or_none(bearing),
        "entry": _int_or_none(getattr(state, "predicted_weather_entry_step", None)),
        "closing": bool(closing),
    }


def _boundary_signature(core: Any, state: Any) -> Dict[str, Any]:
    current = _signed_boundary_distance(core, tuple(state.position))
    projected = _signed_boundary_distance(core, _project_no_turn_point(core, state))
    closing = False if current is None or projected is None else bool(projected < current)
    return {
        "dist": _round_or_none(current),
        "bearing": _round_or_none(_boundary_bearing(core, state)),
        "exit": _int_or_none(getattr(state, "predicted_boundary_exit_step", None)),
        "closing": bool(closing),
    }


def _place_signature(core: Any, state: Any) -> Dict[str, Any]:
    line = state.route.linestring
    signed_xtrack, route_progress, _ = line_signed_cross_track(line, tuple(state.position))
    route_length = max(float(getattr(line, "length", 0.0)), 1e-6)
    route_frac = max(0.0, min(1.0, float(route_progress) / route_length))
    return {
        "route_frac": round(route_frac, 4),
        "xtrack": _round_or_none(signed_xtrack),
        "heading_err": _round_or_none(_heading_error_to_destination(state)),
    }


def _traffic_signature(core: Any, threat_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    traffic: List[Dict[str, Any]] = []
    for row in list(threat_rows or [])[:2]:
        intruder_id = str(row.get("intruder_id", ""))
        if not intruder_id:
            continue
        try:
            intruder_state = core.get_agent(intruder_id)
        except Exception:
            continue
        traffic.append(
            {
                "route": _route_fingerprint_from_state(intruder_state),
                "dist": _round_or_none(row.get("current_distance")),
                "bearing": _round_or_none(row.get("bearing_error_deg")),
                "loss": _int_or_none(row.get("predicted_loss_step")),
                "min_sep": _round_or_none(row.get("predicted_min_sep")),
            }
        )
    return traffic


def _preview_signature(preview_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(preview_rows or [])
    safe_rows = [row for row in rows if bool(row.get("safe_over_preview", False))]
    unsafe_rows = [row for row in rows if not bool(row.get("safe_over_preview", False))]
    progress_pool = safe_rows if safe_rows else rows
    best_safe = safe_rows[0] if safe_rows else (rows[0] if rows else None)
    best_progress = (
        max(progress_pool, key=lambda row: float(row.get("progress_to_destination", 0.0)))
        if progress_pool
        else None
    )
    return {
        "safe": [int(row.get("heading_change_deg", 0)) for row in safe_rows],
        "unsafe": [int(row.get("heading_change_deg", 0)) for row in unsafe_rows],
        "best_safe": None if best_safe is None else int(best_safe.get("heading_change_deg", 0)),
        "best_progress": None
        if best_progress is None
        else int(best_progress.get("heading_change_deg", 0)),
    }


def _scenario_type(
    *,
    core: Any,
    state: Any,
    call_name: str,
    traffic: Sequence[Mapping[str, Any]],
    weather: Mapping[str, Any],
    boundary: Mapping[str, Any],
) -> str:
    tags: List[str] = []
    primary = traffic[0] if traffic else {}
    primary_min_sep = _finite_float(primary.get("min_sep"))
    has_traffic = bool(
        primary
        and (
            primary.get("loss") is not None
            or (primary_min_sep is not None and primary_min_sep < 2.0 * float(SAFE_R))
        )
    )
    weather_clear_now = _finite_float(weather.get("clear_now"))
    weather_clear_pred = _finite_float(weather.get("clear_pred"))
    weather_buffer = float(getattr(core, "weather_clearance_buffer_units", 0.0) or 0.0)
    has_weather = bool(
        weather.get("entry") is not None
        or (
            weather_buffer > 0.0
            and (
                (weather_clear_now is not None and weather_clear_now < 2.0 * weather_buffer)
                or (weather_clear_pred is not None and weather_clear_pred < 2.0 * weather_buffer)
            )
        )
    )
    boundary_dist = _finite_float(boundary.get("dist"))
    has_boundary = bool(
        boundary.get("exit") is not None
        or bool(getattr(state, "boundary_warning_active", False))
        or (
            boundary_dist is not None
            and boundary_dist < 2.0 * float(BOUNDARY_WARNING_DISTANCE_UNITS)
            and (boundary_dist < float(BOUNDARY_WARNING_DISTANCE_UNITS) or bool(boundary.get("closing")))
        )
    )
    if has_traffic:
        tags.append("traffic")
    if has_weather:
        tags.append("weather")
    if has_boundary:
        tags.append("boundary")
    if tags:
        return "+".join(tags)
    if str(call_name) == "MERGE_BACK":
        return "merge_back"
    return "nominal"


def build_decision_case(
    core: Any,
    *,
    step: int,
    agent_id: str,
    call_name: str,
    call_reason: str,
    threat_rows: Sequence[Mapping[str, Any]],
    preview_rows: Sequence[Mapping[str, Any]],
    action_deg: Optional[int] = None,
) -> Dict[str, Any]:
    state = core.get_agent(agent_id)
    traffic = _traffic_signature(core, threat_rows)
    weather = _weather_signature(core, state)
    boundary = _boundary_signature(core, state)
    scenario_type = _scenario_type(
        core=core,
        state=state,
        call_name=call_name,
        traffic=traffic,
        weather=weather,
        boundary=boundary,
    )
    result = {"action": None, "corrected": None, "note": None}
    if action_deg is not None:
        result["action"] = int(action_deg)
    return {
        "step": int(step),
        "agent": str(agent_id),
        "route": _route_fingerprint_from_state(state),
        "call": str(call_name),
        "reason": str(call_reason or ""),
        "type": scenario_type,
        "place": _place_signature(core, state),
        "traffic": traffic,
        "weather": weather,
        "boundary": boundary,
        "preview": _preview_signature(preview_rows),
        "result": result,
    }


def _closeness(a: Any, b: Any, scale: float) -> float:
    a_val = _finite_float(a)
    b_val = _finite_float(b)
    if a_val is None and b_val is None:
        return 1.0
    if a_val is None or b_val is None:
        return 0.0
    return max(0.0, 1.0 - min(abs(a_val - b_val) / max(float(scale), 1e-6), 1.0))


def _angle_closeness(a: Any, b: Any, scale: float = 90.0) -> float:
    a_val = _finite_float(a)
    b_val = _finite_float(b)
    if a_val is None and b_val is None:
        return 1.0
    if a_val is None or b_val is None:
        return 0.0
    return max(0.0, 1.0 - min(abs(_wrap_deg(a_val - b_val)) / max(float(scale), 1e-6), 1.0))


def _step_closeness(a: Any, b: Any, scale: float = 8.0) -> float:
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    return _closeness(a, b, scale)


def _bool_similarity(a: Any, b: Any) -> float:
    return 1.0 if bool(a) == bool(b) else 0.0


def _set_similarity(a: Any, b: Any) -> float:
    set_a = set(int(value) for value in (a or []))
    set_b = set(int(value) for value in (b or []))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / float(len(set_a.union(set_b)))


def _place_similarity(query: Mapping[str, Any], case: Mapping[str, Any]) -> float:
    return (
        0.45 * _closeness(query.get("route_frac"), case.get("route_frac"), 0.08)
        + 0.25 * _closeness(query.get("xtrack"), case.get("xtrack"), SAFE_R)
        + 0.30 * _angle_closeness(query.get("heading_err"), case.get("heading_err"), 45.0)
    )


def _traffic_similarity(query: Sequence[Mapping[str, Any]], case: Sequence[Mapping[str, Any]]) -> float:
    if not query and not case:
        return 1.0
    if not query or not case:
        return 0.0

    def row_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        route_score = 1.0 if left.get("route") == right.get("route") else 0.0
        return (
            0.20 * route_score
            + 0.20 * _closeness(left.get("dist"), right.get("dist"), 15.0)
            + 0.20 * _angle_closeness(left.get("bearing"), right.get("bearing"), 90.0)
            + 0.20 * _step_closeness(left.get("loss"), right.get("loss"), 8.0)
            + 0.20 * _closeness(left.get("min_sep"), right.get("min_sep"), 8.0)
        )

    primary = row_similarity(query[0], case[0])
    if len(query) < 2 or len(case) < 2:
        return primary
    return 0.80 * primary + 0.20 * row_similarity(query[1], case[1])


def _weather_similarity(query: Mapping[str, Any], case: Mapping[str, Any]) -> float:
    return (
        0.25 * _closeness(query.get("clear_now"), case.get("clear_now"), 10.0)
        + 0.25 * _closeness(query.get("clear_pred"), case.get("clear_pred"), 10.0)
        + 0.20 * _angle_closeness(query.get("bearing"), case.get("bearing"), 90.0)
        + 0.20 * _step_closeness(query.get("entry"), case.get("entry"), 8.0)
        + 0.10 * _bool_similarity(query.get("closing"), case.get("closing"))
    )


def _boundary_similarity(query: Mapping[str, Any], case: Mapping[str, Any]) -> float:
    return (
        0.35 * _closeness(query.get("dist"), case.get("dist"), 5.0)
        + 0.25 * _angle_closeness(query.get("bearing"), case.get("bearing"), 90.0)
        + 0.25 * _step_closeness(query.get("exit"), case.get("exit"), 8.0)
        + 0.15 * _bool_similarity(query.get("closing"), case.get("closing"))
    )


def _preview_similarity(query: Mapping[str, Any], case: Mapping[str, Any]) -> float:
    return (
        0.35 * _set_similarity(query.get("safe"), case.get("safe"))
        + 0.25 * _set_similarity(query.get("unsafe"), case.get("unsafe"))
        + 0.20 * _closeness(query.get("best_safe"), case.get("best_safe"), 30.0)
        + 0.20 * _closeness(query.get("best_progress"), case.get("best_progress"), 30.0)
    )


def _misc_similarity(
    query: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    query_seed: Optional[int],
    case_seed: Optional[int],
    query_num_agents: Optional[int],
    case_num_agents: Optional[int],
) -> float:
    reason_score = 1.0 if query.get("reason") == case.get("reason") else 0.4
    seed_score = 0.5
    if query_seed is not None and case_seed is not None:
        seed_score = 1.0 if int(query_seed) == int(case_seed) else 0.0
    return (
        0.35 * reason_score
        + 0.25 * seed_score
        + 0.20 * _closeness(query_num_agents, case_num_agents, 6.0)
        + 0.20 * _closeness(query.get("step"), case.get("step"), 20.0)
    )


def _case_action(case: Mapping[str, Any]) -> Optional[int]:
    result = case.get("result") if isinstance(case.get("result"), dict) else {}
    corrected = _action_or_none(result.get("corrected"))
    if corrected is not None:
        return corrected
    return _action_or_none(result.get("action"))


def _case_note(case: Mapping[str, Any]) -> Optional[str]:
    result = case.get("result") if isinstance(case.get("result"), dict) else {}
    note = result.get("note")
    if isinstance(note, str) and note.strip():
        return note.strip()
    return None


def _action_row_is_currently_acceptable(
    preview_rows: Sequence[Mapping[str, Any]],
    action_deg: int,
) -> bool:
    rows = list(preview_rows or [])
    row_by_turn = {int(row.get("heading_change_deg", 0)): row for row in rows}
    selected = row_by_turn.get(int(action_deg))
    if selected is None:
        return False
    if not rows:
        return False
    if selected.get("pair_loss_step") is not None and any(row.get("pair_loss_step") is None for row in rows):
        return False
    if selected.get("weather_entry_step") is not None and any(row.get("weather_entry_step") is None for row in rows):
        return False
    if not bool(selected.get("traffic_buffer_ok", True)) and any(
        bool(row.get("traffic_buffer_ok", True)) for row in rows
    ):
        return False
    if not bool(selected.get("weather_buffer_ok", True)) and any(
        bool(row.get("weather_buffer_ok", True)) for row in rows
    ):
        return False
    if not bool(selected.get("safe_over_preview", False)) and any(
        bool(row.get("safe_over_preview", False)) for row in rows
    ):
        return False
    return True


def _hard_gate_passes(
    query: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    current_preview_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if query.get("route") != case.get("route"):
        return False
    if query.get("call") != case.get("call"):
        return False
    if query.get("type") != case.get("type"):
        return False
    if "traffic" in str(query.get("type", "")):
        query_traffic = query.get("traffic") if isinstance(query.get("traffic"), list) else []
        case_traffic = case.get("traffic") if isinstance(case.get("traffic"), list) else []
        if not query_traffic or not case_traffic:
            return False
        if query_traffic[0].get("route") != case_traffic[0].get("route"):
            return False
    action = _case_action(case)
    if action is None:
        return False
    return _action_row_is_currently_acceptable(current_preview_rows, action)


def score_case_similarity(
    query: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    query_seed: Optional[int],
    case_seed: Optional[int],
    query_num_agents: Optional[int],
    case_num_agents: Optional[int],
) -> float:
    return (
        0.25 * _place_similarity(query.get("place", {}), case.get("place", {}))
        + 0.20 * _traffic_similarity(query.get("traffic", []), case.get("traffic", []))
        + 0.15 * _weather_similarity(query.get("weather", {}), case.get("weather", {}))
        + 0.15 * _boundary_similarity(query.get("boundary", {}), case.get("boundary", {}))
        + 0.15 * _preview_similarity(query.get("preview", {}), case.get("preview", {}))
        + 0.10
        * _misc_similarity(
            query,
            case,
            query_seed=query_seed,
            case_seed=case_seed,
            query_num_agents=query_num_agents,
            case_num_agents=case_num_agents,
        )
    )


class DecisionMemoryStore:
    """Load, update, and query compact decision memory."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_episodes: int = DECISION_MEMORY_MAX_EPISODES,
        min_similarity: float = DECISION_MEMORY_MIN_SIMILARITY,
    ) -> None:
        self.path = Path(path)
        self.max_episodes = int(max_episodes)
        self.min_similarity = float(min_similarity)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "episodes": []}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {"schema_version": SCHEMA_VERSION, "episodes": []}
        if not isinstance(payload, dict):
            return {"schema_version": SCHEMA_VERSION, "episodes": []}
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            episodes = []
        return {"schema_version": SCHEMA_VERSION, "episodes": episodes}

    def save(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f"{self.path.name}.tmp.{os.getpid()}")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(tmp_path, self.path)

    def retrieve(
        self,
        core: Any,
        *,
        step: int,
        actionable: Sequence[Mapping[str, Any]],
        threat_rows_by_agent: Mapping[str, Sequence[Mapping[str, Any]]],
        preview_rows_by_agent: Mapping[str, Sequence[Mapping[str, Any]]],
        exclude_episode_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        payload = self.load()
        episodes = list(payload.get("episodes", []))
        query_seed = getattr(core, "reset_seed", None)
        query_num_agents = getattr(core, "reset_num_agents", getattr(core, "num_agents", None))
        matches: List[Dict[str, Any]] = []
        for item in actionable:
            agent_id = str(item["agent_id"])
            query = build_decision_case(
                core,
                step=int(step),
                agent_id=agent_id,
                call_name=str(item.get("call_name", "")),
                call_reason=str(item.get("call_reason", "")),
                threat_rows=threat_rows_by_agent.get(agent_id, []),
                preview_rows=preview_rows_by_agent.get(agent_id, []),
                action_deg=None,
            )
            best: Optional[Dict[str, Any]] = None
            for episode in episodes:
                if not isinstance(episode, dict):
                    continue
                episode_id = str(episode.get("episode_id", ""))
                if exclude_episode_id is not None and episode_id == str(exclude_episode_id):
                    continue
                case_seed = _int_or_none(episode.get("seed"))
                case_num_agents = _int_or_none(episode.get("num_agents"))
                for case in episode.get("cases", []):
                    if not isinstance(case, dict):
                        continue
                    if not _hard_gate_passes(
                        query,
                        case,
                        current_preview_rows=preview_rows_by_agent.get(agent_id, []),
                    ):
                        continue
                    score = score_case_similarity(
                        query,
                        case,
                        query_seed=query_seed,
                        case_seed=case_seed,
                        query_num_agents=query_num_agents,
                        case_num_agents=case_num_agents,
                    )
                    if score < self.min_similarity:
                        continue
                    case_id = _stable_case_id(episode_id, case)
                    case_with_id = {**case, "case_id": case_id}
                    candidate = {
                        "agent": agent_id,
                        "similarity": round(float(score), 2),
                        "action": _action_or_none((case.get("result") or {}).get("action")),
                        "corrected": _action_or_none((case.get("result") or {}).get("corrected")),
                        "note": _case_note(case),
                        "memory_ref": {
                            "episode_id": episode_id,
                            "case_id": case_id,
                        },
                        "case": case_with_id,
                    }
                    if best is None or float(candidate["similarity"]) > float(best["similarity"]):
                        best = candidate
            if best is not None:
                matches.append(best)

        matches.sort(key=lambda item: (-float(item.get("similarity", 0.0)), str(item.get("agent", ""))))
        return matches[:2]

    def append_cases(
        self,
        core: Any,
        *,
        episode_id: str,
        step: int,
        actionable: Sequence[Mapping[str, Any]],
        final_actions: Mapping[str, Mapping[str, Any]],
        threat_rows_by_agent: Mapping[str, Sequence[Mapping[str, Any]]],
        preview_rows_by_agent: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        for item in actionable:
            agent_id = str(item["agent_id"])
            action = final_actions.get(agent_id, {})
            if not isinstance(action, Mapping):
                continue
            action_deg = _action_or_none(action.get("heading_change_deg"))
            if action_deg is None:
                continue
            case = build_decision_case(
                core,
                step=int(step),
                agent_id=agent_id,
                call_name=str(item.get("call_name", "")),
                call_reason=str(item.get("call_reason", "")),
                threat_rows=threat_rows_by_agent.get(agent_id, []),
                preview_rows=preview_rows_by_agent.get(agent_id, []),
                action_deg=action_deg,
            )
            case["case_id"] = _stable_case_id(str(episode_id), case)
            cases.append(case)
        if not cases:
            return []

        payload = self.load()
        episodes = [
            episode
            for episode in payload.get("episodes", [])
            if isinstance(episode, dict)
        ]
        target = None
        for episode in episodes:
            if episode.get("episode_id") == episode_id:
                target = episode
                break
        if target is None:
            target = {
                "episode_id": str(episode_id),
                "seed": getattr(core, "reset_seed", None),
                "num_agents": int(getattr(core, "reset_num_agents", getattr(core, "num_agents", 0)) or 0),
                "num_weather_cells": int(
                    getattr(core, "reset_num_weather_cells", getattr(core, "num_weather_cells", 0)) or 0
                ),
                "cases": [],
            }
            episodes.append(target)

        existing_cases = [
            case
            for case in target.get("cases", [])
            if not (
                isinstance(case, dict)
                and int(case.get("step", -1)) == int(step)
                and str(case.get("agent", "")) in {case_item["agent"] for case_item in cases}
                and str(case.get("call", "")) in {case_item["call"] for case_item in cases}
            )
        ]
        target["cases"] = existing_cases + cases
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            current_episode_id = str(episode.get("episode_id", ""))
            normalized_cases = []
            for case in episode.get("cases", []):
                if not isinstance(case, dict):
                    continue
                if not isinstance(case.get("case_id"), str) or not str(case.get("case_id")).strip():
                    case["case_id"] = _stable_case_id(current_episode_id, case)
                normalized_cases.append(case)
            episode["cases"] = normalized_cases
        payload = {
            "schema_version": SCHEMA_VERSION,
            "episodes": episodes[-max(self.max_episodes, 1) :],
        }
        self.save(payload)
        return cases
