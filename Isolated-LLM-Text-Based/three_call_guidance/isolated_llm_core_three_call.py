"""
Three-call sector-entry LLM guidance runtime for an isolated episode.

This file is a sibling variant of the existing isolated controllers. It keeps
the original isolated files untouched and centralizes all tunable behavior in
one config block at the top of the file.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import ollama
from shapely.geometry import Point, Polygon


@dataclass(frozen=True)
class ThreeCallGuidanceConfig:
    allowed_turn_bins: tuple[int, ...] = (-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30)
    safe_r: float = 5.0
    conflict_lookahead_steps: int = 8
    safe_streak_required: int = 5
    execute_allow_zero: bool = True
    emergency_allow_zero: bool = False
    default_execute_hold_bounds: tuple[int, int] = (0, 3)
    default_emergency_hold_bounds: tuple[int, int] = (1, 3)
    merge_back_min_hold: int = 1
    merge_back_max_hold: int = 60
    merge_back_distance_buffer: float = 2.0
    heading_alignment_tolerance_deg: float = 2.5
    hold_speed_fallback: float = 1.0
    model: str = "llama3:8b"
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 8192
    default_start_llm_at_step: int = 0
    annotation_entered_sector: str = "ENTERED SECTOR"
    annotation_execute_turn: str = "EXECUTE TURN"
    annotation_emergency: str = "EMERGENCY MANEUVER"
    annotation_merge_back: str = "MERGE BACK"


DEFAULT_CONFIG = ThreeCallGuidanceConfig()
ALLOWED_BINS: List[int] = list(DEFAULT_CONFIG.allowed_turn_bins)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT.parent
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_SECTOR_POLYGON_CACHE: Optional[Polygon] = None


def _rescaling(r_min: float, r_max: float, m: float, t_min: float = 0.0, t_max: float = 100.0) -> float:
    delta_r = r_max - r_min
    delta_t = t_max - t_min
    return (m - r_min) * delta_t / delta_r + t_min


def _load_sector_polygon() -> Polygon:
    global _SECTOR_POLYGON_CACHE
    if _SECTOR_POLYGON_CACHE is not None:
        return _SECTOR_POLYGON_CACHE

    df = gpd.read_file(DATA_ROOT / "test.geojson")
    poly = df.geometry[5]
    x, y = poly.exterior.coords.xy
    pts = [(_rescaling(103, 108, float(px)), _rescaling(1, 5, float(py))) for px, py in zip(x, y)]
    sector_poly = Polygon(pts)
    if not sector_poly.is_valid:
        sector_poly = sector_poly.buffer(0)
    _SECTOR_POLYGON_CACHE = sector_poly
    return sector_poly


@dataclass
class ObsSnapshot:
    own_x: float
    own_y: float
    intr_x: float
    intr_y: float
    dest_x: float
    dest_y: float
    sep_oi: float
    dist_o_dest: float
    dist_i_dest: float
    ctd_dist: float
    dist_to_atco_path: float
    own_heading_rad: float
    path_heading_deg: Optional[float] = None
    heading_error_to_path_deg: Optional[float] = None
    signed_cross_track_to_path: Optional[float] = None
    step: Optional[int] = None

    @staticmethod
    def from_vector(vec: np.ndarray, step: Optional[int] = None) -> "ObsSnapshot":
        snap = ObsSnapshot(
            own_x=float(vec[0]),
            own_y=float(vec[1]),
            intr_x=float(vec[2]),
            intr_y=float(vec[3]),
            dest_x=float(vec[4]),
            dest_y=float(vec[5]),
            sep_oi=float(vec[6]),
            dist_o_dest=float(vec[7]),
            dist_i_dest=float(vec[8]),
            ctd_dist=float(vec[9]),
            dist_to_atco_path=float(vec[10]),
            own_heading_rad=float(vec[11]),
            step=step,
        )
        if len(vec) >= 13:
            snap.path_heading_deg = float(vec[12])
        if len(vec) >= 14:
            snap.heading_error_to_path_deg = float(vec[13])
        if len(vec) >= 15:
            snap.signed_cross_track_to_path = float(vec[14])
        return snap


def _rad2deg(radians_value: float) -> float:
    return float(radians_value) * 180.0 / math.pi


def _deg2rad(degrees_value: float) -> float:
    return float(degrees_value) * math.pi / 180.0


def _wrap_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _bearing_deg(dy: float, dx: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def _side_word(x: float, left: str = "left", right: str = "right", center: str = "centerline") -> str:
    if x > 0:
        return left
    if x < 0:
        return right
    return center


def _ahead_or_behind(abs_deg: float) -> str:
    return "ahead" if abs_deg < 90.0 else "abeam/behind"


def _extract_json(text: str) -> Dict[str, Any]:
    match = JSON_OBJECT_RE.search(text.strip())
    if not match:
        raise ValueError("No JSON object found in LLM output.")
    return json.loads(match.group(0))


def _snap_to_allowed_bin(value: float, allowed_bins: List[int]) -> int:
    return int(min(allowed_bins, key=lambda turn_bin: abs(turn_bin - value)))


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def _deg_to_action_idx(delta_deg: int) -> int:
    degrees_value = int(round(delta_deg))
    if degrees_value % 5 != 0 or abs(degrees_value) > 30:
        raise ValueError(f"delta must be multiple of 5 and |d|<=30, got {delta_deg}")
    if degrees_value >= 0:
        return degrees_value // 5
    return 6 + (abs(degrees_value) // 5)


def _heading_error_to_dest_deg(snap: ObsSnapshot) -> float:
    heading_to_dest_deg = _bearing_deg(snap.dest_y - snap.own_y, snap.dest_x - snap.own_x)
    return _wrap_deg(heading_to_dest_deg - _rad2deg(snap.own_heading_rad))


def _bearing_error_to_intruder_deg(snap: ObsSnapshot) -> float:
    own_to_intr_angle_deg = _bearing_deg(snap.intr_y - snap.own_y, snap.intr_x - snap.own_x)
    return _wrap_deg(own_to_intr_angle_deg - _rad2deg(snap.own_heading_rad))


def _target_heading_to_destination_rad(snap: ObsSnapshot) -> float:
    return math.atan2(snap.dest_y - snap.own_y, snap.dest_x - snap.own_x)


def _validate_and_normalize(payload: Dict[str, Any], allowed_bins: List[int]) -> Dict[str, Any]:
    if "answer" not in payload or not isinstance(payload["answer"], dict):
        raise ValueError("Missing 'answer' object.")

    answer = payload["answer"]
    if not isinstance(answer.get("scenario_summary"), str):
        answer["scenario_summary"] = ""
    if not isinstance(answer.get("technical_summary"), str):
        answer["technical_summary"] = ""
    if not isinstance(answer.get("rationale"), str):
        answer["rationale"] = ""

    answer.setdefault("metrics", {})
    metrics = answer["metrics"]
    for key in ("ttcp", "separation_distance"):
        if key in metrics and metrics[key] is not None:
            try:
                metrics[key] = float(metrics[key])
            except Exception:
                metrics[key] = None
        else:
            metrics[key] = None

    answer.setdefault("maneuver", {})
    maneuver = answer["maneuver"]
    raw_heading = maneuver.get("heading_change_deg", 0)
    try:
        snapped_heading = _snap_to_allowed_bin(float(raw_heading), allowed_bins)
    except Exception:
        snapped_heading = 0
    maneuver["heading_change_deg"] = snapped_heading

    raw_hold = maneuver.get("hold_steps", 0)
    hold_steps = _coerce_int(raw_hold, default=0)
    maneuver["hold_steps"] = max(0, hold_steps)

    return {
        "answer": {
            "scenario_summary": answer["scenario_summary"],
            "technical_summary": answer["technical_summary"],
            "rationale": answer["rationale"],
            "metrics": metrics,
            "maneuver": maneuver,
        }
    }


def ollama_invoke(
    prompt_text: str,
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        options={"temperature": temperature, "top_p": top_p, "num_predict": max_tokens},
    )
    return (response.get("message") or {}).get("content", "") or ""


class ThreeCallPromptBuilder:
    def __init__(self, config: ThreeCallGuidanceConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._last_kinematics: Optional[dict] = None

    @property
    def last_kinematics(self) -> Optional[dict]:
        return self._last_kinematics

    def build_prompt(
        self,
        obs_vector: np.ndarray,
        *,
        step: int,
        prev_obs_vector: Optional[np.ndarray],
        prev_step: Optional[int],
        guidance_state: Dict[str, Any],
        call_name: str,
    ) -> str:
        snap = ObsSnapshot.from_vector(obs_vector, step=step)
        prev = ObsSnapshot.from_vector(prev_obs_vector, step=prev_step) if prev_obs_vector is not None else None
        parts = [
            self._instruction_text(call_name, guidance_state),
            self._current_observation_text(snap),
            self._derived_signals_text(snap, prev),
            self._kinematics_text(snap, prev),
            self._guidance_state_text(call_name, guidance_state),
        ]
        return "\n\n".join(part for part in parts if part)

    def _instruction_text(self, call_name: str, guidance_state: Dict[str, Any]) -> str:
        cfg = self.config
        hold_bounds = guidance_state.get("hold_bounds")
        if isinstance(hold_bounds, tuple):
            hold_text = f"{hold_bounds[0]}..{hold_bounds[1]}"
        else:
            hold_text = "controller-computed"

        if call_name == "EXECUTE_TURN":
            stage_text = (
                f"STAGE: EXECUTE_TURN. Ownship has just entered the sector polygon.\n"
                f"GOAL: Choose a turn and hold that preserves safe separation and avoids unnecessary deviation from the destination heading.\n"
                f"ZERO-TURN RULE: Return 0 deg only if no loss of safe separation is predicted within the next {cfg.conflict_lookahead_steps} steps.\n"
                f"HOLD RANGE: {hold_text}.\n"
            )
        elif call_name == "EMERGENCY_MANEUVER":
            stage_text = (
                f"STAGE: EMERGENCY_MANEUVER. A previous execute/emergency hold has finished, but conflict is still predicted within the next {cfg.conflict_lookahead_steps} steps.\n"
                "GOAL: Choose a non-zero avoidance turn and short hold that improves safety immediately while limiting destination deviation.\n"
                "ZERO-TURN RULE: 0 deg is NOT allowed in this stage.\n"
                f"HOLD RANGE: {hold_text}.\n"
            )
        elif call_name == "MERGE_BACK":
            stage_text = (
                f"STAGE: MERGE_BACK. Safe separation has been maintained for {cfg.safe_streak_required} observed steps after avoidance.\n"
                "GOAL: Align ownship back toward its destination heading.\n"
                "HOLD RULE: The controller computes the straight-flight hold length after your turn decision. You may leave hold_steps as 0.\n"
            )
        else:
            raise ValueError(f"Unsupported call name: {call_name}")

        allowed_values = ", ".join(str(v) for v in cfg.allowed_turn_bins)
        return (
            "ROLE: You are a cautious ATCO helper.\n"
            f"{stage_text}"
            "\n"
            "OUTPUT RULES:\n"
            "- Return EXACTLY ONE JSON object.\n"
            f"- heading_change_deg must be one of {{{allowed_values}}}.\n"
            "- Use positive values for left turns and negative values for right turns.\n"
            "\n"
            "STRICT JSON SHAPE:\n"
            "{\n"
            "  \"answer\": {\n"
            "    \"scenario_summary\": \"<1-3 short sentences>\",\n"
            "    \"technical_summary\": \"<brief technical summary>\",\n"
            "    \"rationale\": \"<why this maneuver helps>\",\n"
            "    \"metrics\": { \"ttcp\": null, \"separation_distance\": null },\n"
            "    \"maneuver\": { \"heading_change_deg\": -30..30 step 5, \"hold_steps\": <int>=0 }\n"
            "  }\n"
            "}\n"
            "\n"
            "Current observation and derived signals are:"
        )

    def _kinematics_text(self, snap: ObsSnapshot, prev: Optional[ObsSnapshot]) -> str:
        if prev is None:
            self._last_kinematics = {
                "ttcp_steps": None,
                "own_speed": 0.0,
                "intr_speed": 0.0,
                "rel_speed": 0.0,
                "sep": float(snap.sep_oi),
            }
            return (
                "Kinematics (needs a previous observation to compute):\n"
                "- Ownship speed: N/A\n"
                "- Intruder speed: N/A\n"
                "- Relative speed: N/A\n"
                "- TTCP (to CPA): N/A"
            )

        dt_steps = 1
        if snap.step is not None and prev.step is not None and snap.step != prev.step:
            dt_steps = max(1, snap.step - prev.step)

        dx_o = snap.own_x - prev.own_x
        dy_o = snap.own_y - prev.own_y
        dx_i = snap.intr_x - prev.intr_x
        dy_i = snap.intr_y - prev.intr_y

        own_speed = math.hypot(dx_o, dy_o) / dt_steps
        intr_speed = math.hypot(dx_i, dy_i) / dt_steps

        rx = snap.intr_x - snap.own_x
        ry = snap.intr_y - snap.own_y
        vrx = (dx_i - dx_o) / dt_steps
        vry = (dy_i - dy_o) / dt_steps
        rel_speed2 = vrx * vrx + vry * vry

        if rel_speed2 > 0:
            t_star = -((rx * vrx + ry * vry) / rel_speed2)
            ttcp_steps = max(0.0, t_star)
            ttcp_text = f"{ttcp_steps:.2f} steps"
        else:
            ttcp_steps = None
            ttcp_text = "N/A (relative speed ~ 0)"

        rel_speed = math.sqrt(rel_speed2)
        self._last_kinematics = {
            "ttcp_steps": ttcp_steps,
            "own_speed": float(own_speed),
            "intr_speed": float(intr_speed),
            "rel_speed": float(rel_speed),
            "sep": float(snap.sep_oi),
        }

        lines = [
            "Kinematics (scenario-units per step; 1 step = 2 min):",
            f"- Ownship speed: {own_speed:.3f}",
            f"- Intruder speed: {intr_speed:.3f}",
            f"- Relative speed |v_rel|: {rel_speed:.3f}",
            f"- TTCP (to CPA): {ttcp_text}",
        ]
        return "\n".join(lines)

    def _current_observation_text(self, snap: ObsSnapshot) -> str:
        lines = ["Current situation (all numbers are in scenario units):"]
        lines.append(f"- Step index: {snap.step}")
        lines.append(f"- Ownship position: x={snap.own_x:.1f}, y={snap.own_y:.1f}")
        lines.append(f"- Intruder position: x={snap.intr_x:.1f}, y={snap.intr_y:.1f}")
        lines.append(f"- Ownship destination: x={snap.dest_x:.1f}, y={snap.dest_y:.1f}")
        lines.append(f"- Separation (ownship ↔ intruder): {snap.sep_oi:.2f}")
        lines.append(f"- Distances: ownship→dest={snap.dist_o_dest:.2f}, intruder→dest={snap.dist_i_dest:.2f}")
        lines.append(
            f"- Deviation: cross-track-to-original={snap.ctd_dist:.2f}, distance-to-ATCO-path={snap.dist_to_atco_path:.2f}"
        )
        lines.append(
            f"- Ownship heading: {_rad2deg(snap.own_heading_rad):.1f}° (0°=east, positive is counter-clockwise)"
        )
        if snap.path_heading_deg is not None:
            lines.append(f"- ATCO path heading: {snap.path_heading_deg:+.1f}°")
        if snap.heading_error_to_path_deg is not None:
            lines.append(
                f"- heading_error_to_path_deg: {snap.heading_error_to_path_deg:+.1f}° "
                f"(path is to your {_side_word(snap.heading_error_to_path_deg)})"
            )
        return "\n".join(lines)

    def _derived_signals_text(self, snap: ObsSnapshot, prev: Optional[ObsSnapshot]) -> str:
        heading_error_to_dest_deg = _heading_error_to_dest_deg(snap)
        bearing_error_to_intr_deg = _bearing_error_to_intruder_deg(snap)
        intr_pos_rel = _ahead_or_behind(abs(bearing_error_to_intr_deg))

        if prev is not None:
            sep_trend = snap.sep_oi - prev.sep_oi
            sep_trend_text = (
                f"{sep_trend:+.3f} per step "
                f"({'closing' if sep_trend < 0 else 'opening' if sep_trend > 0 else 'steady'})"
            )
        else:
            sep_trend_text = "N/A (no previous observation)"

        lines = [
            "Derived signals (simple math, for decision support):",
            f"- Heading error to destination: {heading_error_to_dest_deg:+.1f}° "
            f"(destination is to your {_side_word(heading_error_to_dest_deg)})",
            f"- Bearing error to intruder:   {bearing_error_to_intr_deg:+.1f}° "
            f"(intruder is to your {_side_word(bearing_error_to_intr_deg)}; {intr_pos_rel})",
            f"- Separation trend (Δsep):      {sep_trend_text}",
        ]
        return "\n".join(lines)

    def _guidance_state_text(self, call_name: str, guidance_state: Dict[str, Any]) -> str:
        lines = ["Controller-supplied guidance state:"]
        lines.append(f"- call_name: {call_name}")
        lines.append(f"- current_stage: {guidance_state.get('current_stage')}")
        lines.append(f"- inside_sector_now: {guidance_state.get('inside_sector')}")
        lines.append(f"- has_entered_sector: {guidance_state.get('has_entered_sector')}")
        lines.append(f"- sector_entry_step: {guidance_state.get('sector_entry_step')}")
        lines.append(f"- predicted_loss_step: {guidance_state.get('predicted_loss_step')}")
        lines.append(f"- conflict_within_lookahead: {guidance_state.get('conflict_within_lookahead')}")
        lines.append(f"- current_ttcp_steps: {guidance_state.get('current_ttcp_steps')}")
        lines.append(f"- safe_streak_steps: {guidance_state.get('safe_streak_steps')}")
        lines.append(f"- last_turn_deg: {guidance_state.get('last_turn_deg')}")
        lines.append(f"- emergency_count: {guidance_state.get('emergency_count')}")
        lines.append(f"- zero_turn_allowed: {guidance_state.get('zero_turn_allowed')}")
        lines.append(f"- hold_bounds: {guidance_state.get('hold_bounds')}")
        return "\n".join(lines)


class LLMThreeCallEpisodeController:
    def __init__(
        self,
        builder: ThreeCallPromptBuilder,
        *,
        config: ThreeCallGuidanceConfig = DEFAULT_CONFIG,
        start_llm_at_step: Optional[int] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.builder = builder
        self.config = config
        self.start_llm_at_step = (
            config.default_start_llm_at_step if start_llm_at_step is None else int(start_llm_at_step)
        )
        self.model = config.model if model is None else str(model)
        self.temperature = config.temperature if temperature is None else float(temperature)
        self.top_p = config.top_p if top_p is None else float(top_p)
        self.max_tokens = config.max_tokens if max_tokens is None else int(max_tokens)
        self.sector_polygon = _load_sector_polygon()
        self.reset()

    def reset(self) -> None:
        self.stage = "FOLLOW_LINESTRING"
        self.inside_sector = False
        self.has_entered_sector = False
        self.sector_entry_step: Optional[int] = None
        self.safe_streak_steps = 0
        self.hold_remaining = 0
        self.last_turn_deg = 0
        self.last_call_name: Optional[str] = None
        self.last_tag: Optional[str] = None
        self.last_prompt_text: Optional[str] = None
        self.last_raw_text: Optional[str] = None
        self.last_normalized: Optional[dict] = None
        self.last_event: Optional[str] = None
        self.last_annotations: List[str] = []
        self.did_call_llm = False
        self.last_predicted_ttcp_steps: Optional[float] = None
        self.last_predicted_loss_step: Optional[int] = None
        self.last_conflict_within_lookahead = False
        self.last_applied_hold_steps = 0
        self.has_called_execute_turn = False
        self.emergency_count = 0
        self.merge_back_count = 0
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_step: Optional[int] = None

    def mark_finished(self) -> None:
        self.stage = "FINISHED"

    def _within_sector_polygon(self, snap: ObsSnapshot) -> bool:
        return bool(self.sector_polygon.covers(Point(snap.own_x, snap.own_y)))

    def _estimate_motion(self, cur: ObsSnapshot, prev: Optional[ObsSnapshot]) -> Dict[str, float]:
        if prev is None:
            return {"own_speed": 0.0, "intr_vx": 0.0, "intr_vy": 0.0}

        dt = 1
        if cur.step is not None and prev.step is not None and cur.step != prev.step:
            dt = max(1, cur.step - prev.step)

        own_dx = (cur.own_x - prev.own_x) / dt
        own_dy = (cur.own_y - prev.own_y) / dt
        intr_vx = (cur.intr_x - prev.intr_x) / dt
        intr_vy = (cur.intr_y - prev.intr_y) / dt
        own_speed = math.hypot(own_dx, own_dy)
        return {"own_speed": own_speed, "intr_vx": intr_vx, "intr_vy": intr_vy}

    def _estimate_ttcp_steps(self, cur: ObsSnapshot, prev: Optional[ObsSnapshot]) -> Optional[float]:
        if prev is None:
            return None

        dt = 1
        if cur.step is not None and prev.step is not None and cur.step != prev.step:
            dt = max(1, cur.step - prev.step)

        dx_o = (cur.own_x - prev.own_x) / dt
        dy_o = (cur.own_y - prev.own_y) / dt
        dx_i = (cur.intr_x - prev.intr_x) / dt
        dy_i = (cur.intr_y - prev.intr_y) / dt

        rx = cur.intr_x - cur.own_x
        ry = cur.intr_y - cur.own_y
        vrx = dx_i - dx_o
        vry = dy_i - dy_o
        vrel2 = vrx * vrx + vry * vry
        if vrel2 <= 0:
            return None

        t_star = -((rx * vrx + ry * vry) / vrel2)
        return max(0.0, float(t_star))

    def _predict_loss_step(
        self,
        cur: ObsSnapshot,
        prev: Optional[ObsSnapshot],
        *,
        heading_rad: float,
        horizon_steps: Optional[int] = None,
    ) -> Optional[int]:
        cfg = self.config
        preview_steps = cfg.conflict_lookahead_steps if horizon_steps is None else int(horizon_steps)
        motion = self._estimate_motion(cur, prev)
        own_speed = float(motion["own_speed"])
        intr_vx = float(motion["intr_vx"])
        intr_vy = float(motion["intr_vy"])
        if own_speed <= 0.0:
            return None

        own_x = float(cur.own_x)
        own_y = float(cur.own_y)
        intr_x = float(cur.intr_x)
        intr_y = float(cur.intr_y)
        ux = math.cos(float(heading_rad))
        uy = math.sin(float(heading_rad))

        for step_ahead in range(1, preview_steps + 1):
            own_x += own_speed * ux
            own_y += own_speed * uy
            intr_x += intr_vx
            intr_y += intr_vy
            sep = math.hypot(intr_x - own_x, intr_y - own_y)
            if sep < cfg.safe_r:
                return step_ahead
        return None

    def _preview_heading_rad(self, cur: ObsSnapshot) -> float:
        if not self.has_entered_sector and cur.path_heading_deg is not None:
            return _deg2rad(cur.path_heading_deg)
        return cur.own_heading_rad

    def _update_safe_streak(self, cur: ObsSnapshot) -> None:
        countable_stages = {"EXECUTE_TURN", "EXECUTE_HOLD", "EMERGENCY_MANEUVER", "EMERGENCY_HOLD", "WAIT_FOR_SAFE"}
        if self.has_called_execute_turn and self.stage in countable_stages:
            if cur.sep_oi >= self.config.safe_r:
                self.safe_streak_steps += 1
            else:
                self.safe_streak_steps = 0

    def _follow_linestring_turn_deg(self, snap: ObsSnapshot, step: int) -> int:
        if step < self.start_llm_at_step:
            return 0
        if snap.heading_error_to_path_deg is None:
            return 0
        err = float(snap.heading_error_to_path_deg)
        if abs(err) < self.config.heading_alignment_tolerance_deg:
            return 0
        return int(_snap_to_allowed_bin(err, ALLOWED_BINS))

    def _guidance_state(
        self,
        cur: ObsSnapshot,
        *,
        call_name: str,
        zero_turn_allowed: bool,
        hold_bounds: Any,
    ) -> Dict[str, Any]:
        return {
            "call_name": call_name,
            "current_stage": self.stage,
            "inside_sector": self.inside_sector,
            "has_entered_sector": self.has_entered_sector,
            "sector_entry_step": self.sector_entry_step,
            "predicted_loss_step": self.last_predicted_loss_step,
            "conflict_within_lookahead": self.last_conflict_within_lookahead,
            "current_ttcp_steps": None
            if self.last_predicted_ttcp_steps is None
            else round(self.last_predicted_ttcp_steps, 3),
            "safe_streak_steps": self.safe_streak_steps,
            "last_turn_deg": self.last_turn_deg,
            "emergency_count": self.emergency_count,
            "zero_turn_allowed": zero_turn_allowed,
            "hold_bounds": hold_bounds,
            "step": cur.step,
        }

    def _simulate_turn_candidate(
        self,
        cur: ObsSnapshot,
        prev: Optional[ObsSnapshot],
        *,
        turn_deg: int,
        horizon_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        preview_steps = cfg.conflict_lookahead_steps if horizon_steps is None else int(horizon_steps)
        motion = self._estimate_motion(cur, prev)
        own_speed = float(motion["own_speed"])
        intr_vx = float(motion["intr_vx"])
        intr_vy = float(motion["intr_vy"])
        if own_speed <= 0.0:
            own_speed = cfg.hold_speed_fallback

        heading_rad = cur.own_heading_rad + _deg2rad(turn_deg)
        own_vx = own_speed * math.cos(heading_rad)
        own_vy = own_speed * math.sin(heading_rad)

        own_x = float(cur.own_x)
        own_y = float(cur.own_y)
        intr_x = float(cur.intr_x)
        intr_y = float(cur.intr_y)
        min_sep = float(cur.sep_oi)
        loss_step = None

        for step_ahead in range(1, preview_steps + 1):
            own_x += own_vx
            own_y += own_vy
            intr_x += intr_vx
            intr_y += intr_vy
            sep = math.hypot(intr_x - own_x, intr_y - own_y)
            min_sep = min(min_sep, sep)
            if loss_step is None and sep < cfg.safe_r:
                loss_step = step_ahead

        end_dist_to_dest = math.hypot(cur.dest_x - own_x, cur.dest_y - own_y)
        heading_to_dest_deg = _bearing_deg(cur.dest_y - own_y, cur.dest_x - own_x)
        end_heading_err_deg = _wrap_deg(heading_to_dest_deg - _rad2deg(heading_rad))

        rx = intr_x - own_x
        ry = intr_y - own_y
        vrx = intr_vx - own_vx
        vry = intr_vy - own_vy
        vrel2 = vrx * vrx + vry * vry
        if vrel2 > 0:
            end_ttcp = max(0.0, -((rx * vrx + ry * vry) / vrel2))
        else:
            end_ttcp = None

        return {
            "turn_deg": int(turn_deg),
            "loss_step": loss_step,
            "min_sep": float(min_sep),
            "end_ttcp_steps": end_ttcp,
            "end_dist_to_dest": float(end_dist_to_dest),
            "end_heading_error_to_dest_deg": float(abs(end_heading_err_deg)),
        }

    def _best_preview_turn(self, cur: ObsSnapshot, prev: Optional[ObsSnapshot], *, allow_zero: bool) -> int:
        candidates = []
        for turn_deg in ALLOWED_BINS:
            if not allow_zero and turn_deg == 0:
                continue
            candidates.append(self._simulate_turn_candidate(cur, prev, turn_deg=turn_deg))

        candidates.sort(
            key=lambda item: (
                item["loss_step"] is not None,
                -(item["loss_step"] or (self.config.conflict_lookahead_steps + 1)),
                -item["min_sep"],
                -(item["end_ttcp_steps"] if item["end_ttcp_steps"] is not None else -1.0),
                item["end_dist_to_dest"],
                item["end_heading_error_to_dest_deg"],
                abs(item["turn_deg"]),
            )
        )
        return int(candidates[0]["turn_deg"]) if candidates else 0

    def _deterministic_merge_back_turn(self, cur: ObsSnapshot) -> int:
        err = _heading_error_to_dest_deg(cur)
        if abs(err) < self.config.heading_alignment_tolerance_deg:
            return 0
        return int(_snap_to_allowed_bin(err, ALLOWED_BINS))

    def _compute_merge_back_hold_steps(self, cur: ObsSnapshot, prev: Optional[ObsSnapshot], *, step: int) -> int:
        cfg = self.config
        motion = self._estimate_motion(cur, prev)
        own_speed = float(motion["own_speed"])
        if own_speed <= 0.0:
            own_speed = cfg.hold_speed_fallback

        remaining_dist = max(0.0, float(cur.dist_o_dest) - cfg.merge_back_distance_buffer)
        raw_steps = int(math.ceil(remaining_dist / max(1e-6, own_speed)))
        raw_steps = max(cfg.merge_back_min_hold, raw_steps)
        remaining_budget = max(0, cfg.merge_back_max_hold - int(step))
        if remaining_budget <= 0:
            return 0
        return int(min(raw_steps, remaining_budget))

    def _build_fallback_answer(self, cur: ObsSnapshot) -> Dict[str, Any]:
        return {
            "answer": {
                "scenario_summary": "",
                "technical_summary": "",
                "rationale": "",
                "metrics": {
                    "ttcp": self.last_predicted_ttcp_steps,
                    "separation_distance": cur.sep_oi,
                },
                "maneuver": {
                    "heading_change_deg": 0,
                    "hold_steps": 0,
                },
            }
        }

    def _resolve_turn_for_call(
        self,
        call_name: str,
        requested_turn: int,
        cur: ObsSnapshot,
        prev: Optional[ObsSnapshot],
    ) -> int:
        requested_turn = int(_snap_to_allowed_bin(requested_turn, ALLOWED_BINS))

        if call_name == "EXECUTE_TURN":
            if not self.last_conflict_within_lookahead:
                return 0
            if requested_turn == 0:
                return self._best_preview_turn(cur, prev, allow_zero=False)
            return requested_turn

        if call_name == "EMERGENCY_MANEUVER":
            if requested_turn == 0:
                return self._best_preview_turn(cur, prev, allow_zero=False)
            return requested_turn

        if call_name == "MERGE_BACK":
            desired_turn = self._deterministic_merge_back_turn(cur)
            current_err = _heading_error_to_dest_deg(cur)
            requested_new_err = abs(_wrap_deg(current_err - requested_turn))
            desired_new_err = abs(_wrap_deg(current_err - desired_turn))
            if abs(current_err) <= self.config.heading_alignment_tolerance_deg:
                return 0
            if requested_turn == 0 or requested_new_err > desired_new_err:
                return desired_turn
            return requested_turn

        raise ValueError(f"Unsupported call name: {call_name}")

    def _resolve_hold_for_call(
        self,
        call_name: str,
        requested_hold: int,
        cur: ObsSnapshot,
        prev: Optional[ObsSnapshot],
        *,
        step: int,
    ) -> int:
        requested_hold = max(0, int(requested_hold))
        if call_name == "EXECUTE_TURN":
            hold_min, hold_max = self.config.default_execute_hold_bounds
            return int(max(hold_min, min(hold_max, requested_hold)))
        if call_name == "EMERGENCY_MANEUVER":
            hold_min, hold_max = self.config.default_emergency_hold_bounds
            return int(max(hold_min, min(hold_max, requested_hold)))
        if call_name == "MERGE_BACK":
            return self._compute_merge_back_hold_steps(cur, prev, step=step)
        raise ValueError(f"Unsupported call name: {call_name}")

    def _call_llm_for_stage(
        self,
        call_name: str,
        obs_vec: np.ndarray,
        step: int,
        cur: ObsSnapshot,
    ) -> int:
        prev = ObsSnapshot.from_vector(self._prev_obs, step=self._prev_step) if self._prev_obs is not None else None

        if call_name == "EXECUTE_TURN":
            hold_bounds: Any = self.config.default_execute_hold_bounds
            zero_turn_allowed = self.config.execute_allow_zero
            tag_suffix = "execute_turn"
            self.stage = "EXECUTE_TURN"
            self.last_event = f"execute_turn@{step}"
            self.last_annotations.append(self.config.annotation_execute_turn)
        elif call_name == "EMERGENCY_MANEUVER":
            hold_bounds = self.config.default_emergency_hold_bounds
            zero_turn_allowed = self.config.emergency_allow_zero
            tag_suffix = f"emergency_maneuver_{self.emergency_count + 1:02d}"
            self.stage = "EMERGENCY_MANEUVER"
            self.last_event = f"emergency_maneuver@{step}"
            self.last_annotations.append(self.config.annotation_emergency)
        elif call_name == "MERGE_BACK":
            hold_bounds = "controller_computed"
            zero_turn_allowed = True
            tag_suffix = f"merge_back_{self.merge_back_count + 1:02d}"
            self.stage = "MERGE_BACK"
            self.last_event = f"merge_back@{step}"
            self.last_annotations.append(self.config.annotation_merge_back)
        else:
            raise ValueError(f"Unsupported call name: {call_name}")

        prompt_text = self.builder.build_prompt(
            obs_vector=obs_vec,
            step=step,
            prev_obs_vector=self._prev_obs,
            prev_step=self._prev_step,
            guidance_state=self._guidance_state(
                cur,
                call_name=call_name,
                zero_turn_allowed=zero_turn_allowed,
                hold_bounds=hold_bounds,
            ),
            call_name=call_name,
        )
        raw_text = ollama_invoke(
            prompt_text,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        try:
            parsed = _extract_json(raw_text)
            normalized = _validate_and_normalize(parsed, ALLOWED_BINS)
        except Exception:
            normalized = self._build_fallback_answer(cur)

        maneuver = normalized["answer"]["maneuver"]
        requested_turn = int(maneuver.get("heading_change_deg", 0))
        requested_hold = int(maneuver.get("hold_steps", 0))
        applied_turn = self._resolve_turn_for_call(call_name, requested_turn, cur, prev)
        applied_hold = self._resolve_hold_for_call(call_name, requested_hold, cur, prev, step=step)

        maneuver["heading_change_deg"] = int(applied_turn)
        maneuver["hold_steps"] = int(applied_hold)

        self.did_call_llm = True
        self.last_call_name = call_name
        self.last_tag = f"t{step:03d}_{tag_suffix}"
        self.last_prompt_text = prompt_text
        self.last_raw_text = raw_text
        self.last_normalized = normalized
        self.last_turn_deg = int(applied_turn)
        self.last_applied_hold_steps = int(applied_hold)
        self.hold_remaining = int(applied_hold)

        if call_name == "EXECUTE_TURN":
            self.has_called_execute_turn = True
            self.safe_streak_steps = 0
        elif call_name == "EMERGENCY_MANEUVER":
            self.emergency_count += 1
            self.safe_streak_steps = 0
        elif call_name == "MERGE_BACK":
            self.merge_back_count += 1

        return int(applied_turn)

    def _advance_post_call_stage(self) -> None:
        if self.stage == "EXECUTE_TURN":
            self.stage = "EXECUTE_HOLD" if self.hold_remaining > 0 else "WAIT_FOR_SAFE"
        elif self.stage == "EMERGENCY_MANEUVER":
            self.stage = "EMERGENCY_HOLD" if self.hold_remaining > 0 else "WAIT_FOR_SAFE"
        elif self.stage == "MERGE_BACK":
            self.stage = "MERGE_BACK_HOLD"

    def next_turn_deg(self, obs_vec: np.ndarray, step: int) -> int:
        self.did_call_llm = False
        self.last_call_name = None
        self.last_event = None
        self.last_annotations = []

        cur = ObsSnapshot.from_vector(obs_vec, step=step)
        prev = ObsSnapshot.from_vector(self._prev_obs, step=self._prev_step) if self._prev_obs is not None else None

        inside_now = self._within_sector_polygon(cur)
        if inside_now and not self.inside_sector:
            self.last_event = f"entered_sector@{step}"
            self.last_annotations.append(self.config.annotation_entered_sector)
        self.inside_sector = inside_now
        if inside_now and not self.has_entered_sector:
            self.has_entered_sector = True
            self.sector_entry_step = step

        self.last_predicted_ttcp_steps = self._estimate_ttcp_steps(cur, prev)
        preview_heading = self._preview_heading_rad(cur)
        self.last_predicted_loss_step = self._predict_loss_step(cur, prev, heading_rad=preview_heading)
        self.last_conflict_within_lookahead = self.last_predicted_loss_step is not None

        self._update_safe_streak(cur)
        self._advance_post_call_stage()

        if not self.has_entered_sector:
            turn_now = self._follow_linestring_turn_deg(cur, step)
            self._prev_obs, self._prev_step = obs_vec.copy(), step
            self.last_turn_deg = int(turn_now)
            return int(turn_now)

        if not self.has_called_execute_turn:
            if step < self.start_llm_at_step:
                turn_now = self._follow_linestring_turn_deg(cur, step)
            else:
                turn_now = self._call_llm_for_stage("EXECUTE_TURN", obs_vec, step, cur)
            self._prev_obs, self._prev_step = obs_vec.copy(), step
            return int(turn_now)

        if self.stage == "EXECUTE_HOLD":
            if self.hold_remaining > 0:
                self.hold_remaining -= 1
                turn_now = 0
            else:
                self.stage = "WAIT_FOR_SAFE"
                turn_now = 0
        elif self.stage == "EMERGENCY_HOLD":
            if self.hold_remaining > 0:
                self.hold_remaining -= 1
                turn_now = 0
            else:
                self.stage = "WAIT_FOR_SAFE"
                turn_now = 0
        elif self.stage == "WAIT_FOR_SAFE":
            if self.last_conflict_within_lookahead and step >= self.start_llm_at_step:
                turn_now = self._call_llm_for_stage("EMERGENCY_MANEUVER", obs_vec, step, cur)
            elif self.safe_streak_steps >= self.config.safe_streak_required and step >= self.start_llm_at_step:
                turn_now = self._call_llm_for_stage("MERGE_BACK", obs_vec, step, cur)
            else:
                turn_now = 0
        elif self.stage == "MERGE_BACK_HOLD":
            if self.last_conflict_within_lookahead and step >= self.start_llm_at_step:
                turn_now = self._call_llm_for_stage("EMERGENCY_MANEUVER", obs_vec, step, cur)
            else:
                if self.hold_remaining > 0:
                    self.hold_remaining -= 1
                turn_now = 0
        elif self.stage == "FOLLOW_LINESTRING":
            turn_now = self._follow_linestring_turn_deg(cur, step)
        else:
            turn_now = 0

        self._prev_obs, self._prev_step = obs_vec.copy(), step
        self.last_turn_deg = int(turn_now)
        return int(turn_now)


__all__ = [
    "ALLOWED_BINS",
    "DEFAULT_CONFIG",
    "LLMThreeCallEpisodeController",
    "ThreeCallGuidanceConfig",
    "ThreeCallPromptBuilder",
    "_deg_to_action_idx",
]
