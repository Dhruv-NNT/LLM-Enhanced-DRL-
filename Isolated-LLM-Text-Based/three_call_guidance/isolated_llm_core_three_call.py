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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import ollama
from shapely.geometry import Point, Polygon

from three_call_memory import (
    MemoryFeatures,
    ThreeCallMemoryStore,
    format_memory_card,
)


@dataclass(frozen=True)
class ThreeCallGuidanceConfig:
    allowed_turn_bins: tuple[int, ...] = (-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30)
    safe_r: float = 5.0
    # safe_r: float = 2.5
    conflict_lookahead_steps: int = 8
    turn_preview_steps: int = 5
    boundary_lookahead_steps: int = 7
    safe_streak_required: int = 5
    execute_allow_zero: bool = True
    emergency_allow_zero: bool = False
    merge_back_repeat_error_deg: float = 30.0
    merge_back_recheck_steps: int = 2
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
    use_memory: bool = False
    memory_top_k: int = 2
    memory_min_similarity: float = 0.70
    memory_store_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "memory" / "three_call_memory.jsonl"
    )


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

    return {
        "answer": {
            "scenario_summary": answer["scenario_summary"],
            "technical_summary": answer["technical_summary"],
            "rationale": answer["rationale"],
            "metrics": metrics,
            "maneuver": {
                "heading_change_deg": snapped_heading,
            },
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
            self._turn_preview_text(guidance_state),
            self._boundary_lookahead_text(guidance_state),
            self._memory_cases_text(guidance_state),
            self._guidance_state_text(call_name, guidance_state),
        ]
        return "\n\n".join(part for part in parts if part)

    def _instruction_text(self, call_name: str, guidance_state: Dict[str, Any]) -> str:
        cfg = self.config

        if call_name == "EXECUTE_TURN":
            stage_text = (
                f"STAGE: EXECUTE_TURN. Ownship has just entered the sector polygon.\n"
                "GOAL: Turn on the same side as the intruder while keeping safe separation.\n"
                "- Intruder on left -> choose a left turn.\n"
                "- Intruder on right -> choose a right turn.\n"
                f"- Use the {cfg.turn_preview_steps}-step angle table below.\n"
                f"- Return 0 deg only if no conflict is predicted within the next {cfg.turn_preview_steps} steps.\n"
                "- After you choose the heading change, the controller will keep flying that heading until a later trigger causes another call.\n"
            )
        elif call_name == "EMERGENCY_MANEUVER":
            stage_text = (
                f"STAGE: EMERGENCY_MANEUVER. Conflict is still predicted within the next {cfg.conflict_lookahead_steps} steps.\n"
                "GOAL: Choose a non-zero turn on the same side as the intruder that restores safety.\n"
                "- Intruder on left -> choose a left turn.\n"
                "- Intruder on right -> choose a right turn.\n"
                f"- Use the {cfg.turn_preview_steps}-step angle table below.\n"
                "- ZERO-TURN RULE: 0 deg is NOT allowed in this stage.\n"
                "- After you choose the heading change, the controller will keep flying that heading until a later trigger causes another call.\n"
            )
        elif call_name == "MERGE_BACK":
            stage_text = (
                "STAGE: MERGE_BACK. The controller has decided it is time to recover toward the ownship destination.\n"
                f"- This can happen after safe separation has been maintained for {cfg.safe_streak_required} observed steps, or earlier if the boundary guard requests recovery.\n"
                "GOAL: Turn toward the ownship destination and then keep flying straight.\n"
                f"- If |heading error to destination| is greater than {cfg.merge_back_repeat_error_deg:.0f} deg, this is ALIGNMENT MODE. Use a destination-side turn; MERGE_BACK may be called again after a short controller-side recheck.\n"
                f"- If |heading error to destination| is at most {cfg.merge_back_repeat_error_deg:.0f} deg, this is DESTINATION MODE. Use a destination-side turn that best resumes flight toward the destination.\n"
                "- After you choose the heading change, the controller will keep flying that heading until a later trigger causes another call.\n"
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
            "    \"maneuver\": { \"heading_change_deg\": -30..30 step 5 }\n"
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

    def _turn_preview_text(self, guidance_state: Dict[str, Any]) -> str:
        rows = guidance_state.get("turn_preview_rows") or []
        if not rows:
            return ""

        lines = [
            f"{guidance_state.get('turn_preview_steps')}-step angle preview table (choose only from these rows):",
        ]
        for row in rows:
            lines.append(
                "- "
                f"turn={int(row['heading_change_deg']):+d} | "
                f"safe_over_preview={int(bool(row['safe_preserved_over_preview']))} | "
                f"predicted_loss_step={row['predicted_loss_step']} | "
                f"stays_inside_sector={int(bool(row['stays_inside_sector_over_preview']))} | "
                f"boundary_exit_step={row['boundary_exit_step']} | "
                f"merge_back_turn_after_preview={int(row['merge_back_turn_deg_after_preview']):+d}"
            )
        return "\n".join(lines)

    def _boundary_lookahead_text(self, guidance_state: Dict[str, Any]) -> str:
        boundary_distance = guidance_state.get("boundary_distance")
        boundary_exit_step = guidance_state.get("boundary_exit_step")
        warning_active = guidance_state.get("boundary_warning_active")
        boundary_distance_text = "N/A" if boundary_distance is None else f"{float(boundary_distance):.2f}"
        exit_step_text = "None" if boundary_exit_step is None else str(int(boundary_exit_step))
        return "\n".join(
            [
                f"Boundary lookahead ({self.config.boundary_lookahead_steps}-step bad-exit guard):",
                f"- Current distance to nearest sector boundary: {boundary_distance_text}",
                f"- Predicted sector exit step if current heading is held: {exit_step_text}",
                f"- Boundary warning active: {warning_active}",
            ]
        )

    def _memory_cases_text(self, guidance_state: Dict[str, Any]) -> str:
        cards = guidance_state.get("memory_prompt_cards") or []
        if not cards:
            return ""
        return "\n".join(["Relevant successful past cases:"] + [str(card) for card in cards])

    def _guidance_state_text(self, call_name: str, guidance_state: Dict[str, Any]) -> str:
        lines = ["Controller-supplied guidance state:"]
        lines.append(f"- call_name: {call_name}")
        lines.append(f"- current_stage: {guidance_state.get('current_stage')}")
        lines.append(f"- inside_sector_now: {guidance_state.get('inside_sector')}")
        lines.append(f"- has_entered_sector: {guidance_state.get('has_entered_sector')}")
        lines.append(f"- sector_entry_step: {guidance_state.get('sector_entry_step')}")
        lines.append(f"- predicted_loss_step: {guidance_state.get('predicted_loss_step')}")
        lines.append(f"- conflict_within_lookahead: {guidance_state.get('conflict_within_lookahead')}")
        lines.append(f"- immediate_predicted_loss_step: {guidance_state.get('execute_predicted_loss_step')}")
        lines.append(f"- immediate_conflict_within_preview: {guidance_state.get('execute_conflict_within_preview')}")
        lines.append(f"- boundary_exit_step: {guidance_state.get('boundary_exit_step')}")
        lines.append(f"- boundary_warning_active: {guidance_state.get('boundary_warning_active')}")
        lines.append(f"- boundary_distance: {guidance_state.get('boundary_distance')}")
        lines.append(f"- heading_error_to_destination_deg: {guidance_state.get('heading_error_to_destination_deg')}")
        lines.append(f"- intruder_side: {guidance_state.get('intruder_side')}")
        lines.append(f"- allowed_turn_bins: {guidance_state.get('allowed_turn_bins')}")
        lines.append(f"- merge_back_mode: {guidance_state.get('merge_back_mode')}")
        lines.append(f"- merge_back_reason: {guidance_state.get('merge_back_reason')}")
        lines.append(f"- current_ttcp_steps: {guidance_state.get('current_ttcp_steps')}")
        lines.append(f"- safe_streak_steps: {guidance_state.get('safe_streak_steps')}")
        lines.append(f"- last_turn_deg: {guidance_state.get('last_turn_deg')}")
        lines.append(f"- emergency_count: {guidance_state.get('emergency_count')}")
        lines.append(f"- merge_back_count: {guidance_state.get('merge_back_count')}")
        lines.append(f"- zero_turn_allowed: {guidance_state.get('zero_turn_allowed')}")
        lines.append(f"- merge_back_recheck_remaining: {guidance_state.get('merge_back_recheck_remaining')}")
        if guidance_state.get("memory_enabled"):
            lines.append(f"- memory_enabled: {guidance_state.get('memory_enabled')}")
            lines.append(f"- retrieved_memory_count: {guidance_state.get('retrieved_memory_count')}")
        lines.append(f"- turn_preview_steps: {guidance_state.get('turn_preview_steps')}")
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
        self.memory_store = ThreeCallMemoryStore(Path(config.memory_store_path))
        self.reset()

    def reset(self) -> None:
        self.stage = "FOLLOW_LINESTRING"
        self.inside_sector = False
        self.has_entered_sector = False
        self.sector_entry_step: Optional[int] = None
        self.safe_streak_steps = 0
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
        self.last_execute_predicted_loss_step: Optional[int] = None
        self.last_execute_conflict_within_preview = False
        self.last_boundary_exit_step: Optional[int] = None
        self.last_boundary_warning_active = False
        self.last_boundary_distance: Optional[float] = None
        self.last_turn_preview_rows: List[Dict[str, Any]] = []
        self.last_allowed_turn_bins: List[int] = []
        self.last_intruder_side = "centerline"
        self.last_merge_back_mode = "ALIGNMENT"
        self.last_merge_back_reason = "SAFE_STREAK"
        self.merge_back_recheck_remaining = 0
        self.last_memory_prompt_cards: List[str] = []
        self.last_memory_match_debug: List[Dict[str, Any]] = []
        self.last_memory_features: Optional[MemoryFeatures] = None
        self.has_called_execute_turn = False
        self.emergency_count = 0
        self.merge_back_count = 0
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_step: Optional[int] = None
        self.memory_store.discard_buffer()

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

    def _predict_boundary_exit_step(
        self,
        cur: ObsSnapshot,
        prev: Optional[ObsSnapshot],
        *,
        heading_rad: float,
        horizon_steps: Optional[int] = None,
    ) -> Optional[int]:
        preview_steps = self.config.boundary_lookahead_steps if horizon_steps is None else int(horizon_steps)
        if not self._within_sector_polygon(cur):
            return None

        motion = self._estimate_motion(cur, prev)
        own_speed = float(motion["own_speed"])
        if own_speed <= 0.0:
            return None

        own_x = float(cur.own_x)
        own_y = float(cur.own_y)
        ux = math.cos(float(heading_rad))
        uy = math.sin(float(heading_rad))

        for step_ahead in range(1, preview_steps + 1):
            own_x += own_speed * ux
            own_y += own_speed * uy
            if not self.sector_polygon.covers(Point(own_x, own_y)):
                return step_ahead
        return None

    def _boundary_distance(self, cur: ObsSnapshot) -> Optional[float]:
        if not self._within_sector_polygon(cur):
            return None
        return float(self.sector_polygon.exterior.distance(Point(cur.own_x, cur.own_y)))

    def _preview_heading_rad(self, cur: ObsSnapshot) -> float:
        if not self.has_entered_sector and cur.path_heading_deg is not None:
            return _deg2rad(cur.path_heading_deg)
        return cur.own_heading_rad

    def _update_safe_streak(self, cur: ObsSnapshot) -> None:
        countable_stages = {"EXECUTE_TURN", "EMERGENCY_MANEUVER", "WAIT_FOR_SAFE"}
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

    def _intruder_side_sign(self, cur: ObsSnapshot) -> int:
        bearing_err = _bearing_error_to_intruder_deg(cur)
        if bearing_err > 0:
            return 1
        if bearing_err < 0:
            return -1
        return 0

    def _intruder_side_label(self, cur: ObsSnapshot) -> str:
        return _side_word(_bearing_error_to_intruder_deg(cur))

    def _merge_back_mode(self, cur: ObsSnapshot) -> str:
        return (
            "ALIGNMENT"
            if abs(_heading_error_to_dest_deg(cur)) > self.config.merge_back_repeat_error_deg
            else "DESTINATION"
        )

    def _allowed_turn_bins_for_call(self, call_name: str, cur: ObsSnapshot) -> List[int]:
        if call_name == "MERGE_BACK":
            err = _heading_error_to_dest_deg(cur)
            if abs(err) <= self.config.heading_alignment_tolerance_deg:
                return [0]
            if err > 0:
                return [turn_deg for turn_deg in ALLOWED_BINS if turn_deg > 0]
            return [turn_deg for turn_deg in ALLOWED_BINS if turn_deg < 0]

        if call_name not in {"EXECUTE_TURN", "EMERGENCY_MANEUVER"}:
            return list(ALLOWED_BINS)

        side = self._intruder_side_sign(cur)
        if side > 0:
            bins = [turn_deg for turn_deg in ALLOWED_BINS if turn_deg > 0]
        elif side < 0:
            bins = [turn_deg for turn_deg in ALLOWED_BINS if turn_deg < 0]
        else:
            bins = [turn_deg for turn_deg in ALLOWED_BINS if turn_deg != 0]

        allow_zero = (
            call_name == "EXECUTE_TURN"
            and self.config.execute_allow_zero
            and not self.last_execute_conflict_within_preview
        )
        if allow_zero:
            bins = [0] + bins
        return bins

    def _guidance_state(
        self,
        cur: ObsSnapshot,
        *,
        call_name: str,
        zero_turn_allowed: bool,
    ) -> Dict[str, Any]:
        return {
            "call_name": call_name,
            "current_stage": self.stage,
            "inside_sector": self.inside_sector,
            "has_entered_sector": self.has_entered_sector,
            "sector_entry_step": self.sector_entry_step,
            "predicted_loss_step": self.last_predicted_loss_step,
            "conflict_within_lookahead": self.last_conflict_within_lookahead,
            "execute_predicted_loss_step": self.last_execute_predicted_loss_step,
            "execute_conflict_within_preview": self.last_execute_conflict_within_preview,
            "boundary_exit_step": self.last_boundary_exit_step,
            "boundary_warning_active": self.last_boundary_warning_active,
            "boundary_distance": None
            if self.last_boundary_distance is None
            else round(self.last_boundary_distance, 3),
            "heading_error_to_destination_deg": round(_heading_error_to_dest_deg(cur), 3),
            "intruder_side": self.last_intruder_side,
            "allowed_turn_bins": self.last_allowed_turn_bins,
            "merge_back_mode": self.last_merge_back_mode,
            "merge_back_reason": self.last_merge_back_reason,
            "current_ttcp_steps": None
            if self.last_predicted_ttcp_steps is None
            else round(self.last_predicted_ttcp_steps, 3),
            "safe_streak_steps": self.safe_streak_steps,
            "last_turn_deg": self.last_turn_deg,
            "emergency_count": self.emergency_count,
            "merge_back_count": self.merge_back_count,
            "zero_turn_allowed": zero_turn_allowed,
            "merge_back_recheck_remaining": self.merge_back_recheck_remaining,
            "memory_enabled": self.config.use_memory,
            "retrieved_memory_count": len(self.last_memory_prompt_cards),
            "memory_prompt_cards": self.last_memory_prompt_cards,
            "turn_preview_steps": self.config.turn_preview_steps,
            "turn_preview_rows": self.last_turn_preview_rows,
            "step": cur.step,
        }

    def _make_preview_snapshot(
        self,
        base: ObsSnapshot,
        *,
        own_x: float,
        own_y: float,
        intr_x: float,
        intr_y: float,
        own_heading_rad: float,
        step: Optional[int] = None,
    ) -> ObsSnapshot:
        return ObsSnapshot(
            own_x=float(own_x),
            own_y=float(own_y),
            intr_x=float(intr_x),
            intr_y=float(intr_y),
            dest_x=float(base.dest_x),
            dest_y=float(base.dest_y),
            sep_oi=float(math.hypot(intr_x - own_x, intr_y - own_y)),
            dist_o_dest=float(math.hypot(base.dest_x - own_x, base.dest_y - own_y)),
            dist_i_dest=float(base.dist_i_dest),
            ctd_dist=float(base.ctd_dist),
            dist_to_atco_path=float(base.dist_to_atco_path),
            own_heading_rad=float(own_heading_rad),
            path_heading_deg=base.path_heading_deg,
            heading_error_to_path_deg=base.heading_error_to_path_deg,
            signed_cross_track_to_path=base.signed_cross_track_to_path,
            step=step,
        )

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
        boundary_exit_step = None
        started_inside_sector = self._within_sector_polygon(cur)

        for step_ahead in range(1, preview_steps + 1):
            own_x += own_vx
            own_y += own_vy
            intr_x += intr_vx
            intr_y += intr_vy
            sep = math.hypot(intr_x - own_x, intr_y - own_y)
            min_sep = min(min_sep, sep)
            if loss_step is None and sep < cfg.safe_r:
                loss_step = step_ahead
            if started_inside_sector and boundary_exit_step is None and not self.sector_polygon.covers(Point(own_x, own_y)):
                boundary_exit_step = step_ahead

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

        preview_snapshot = self._make_preview_snapshot(
            cur,
            own_x=own_x,
            own_y=own_y,
            intr_x=intr_x,
            intr_y=intr_y,
            own_heading_rad=heading_rad,
            step=(cur.step + preview_steps) if cur.step is not None else None,
        )

        return {
            "turn_deg": int(turn_deg),
            "loss_step": loss_step,
            "boundary_exit_step": boundary_exit_step,
            "min_sep": float(min_sep),
            "end_ttcp_steps": end_ttcp,
            "end_dist_to_dest": float(end_dist_to_dest),
            "end_heading_error_to_dest_deg": float(abs(end_heading_err_deg)),
            "safe_preserved_over_preview": loss_step is None,
            "stays_inside_sector_over_preview": started_inside_sector and boundary_exit_step is None,
            "merge_back_turn_deg_after_preview": int(self._deterministic_merge_back_turn(preview_snapshot)),
        }

    def _best_turn_from_bins(
        self,
        cur: ObsSnapshot,
        prev: Optional[ObsSnapshot],
        candidate_bins: List[int],
        *,
        horizon_steps: Optional[int] = None,
    ) -> int:
        candidates = [self._simulate_turn_candidate(cur, prev, turn_deg=turn_deg, horizon_steps=horizon_steps) for turn_deg in candidate_bins]
        preview_steps = self.config.conflict_lookahead_steps if horizon_steps is None else int(horizon_steps)

        candidates.sort(
            key=lambda item: (
                item["loss_step"] is not None,
                -(item["loss_step"] or (preview_steps + 1)),
                -item["min_sep"],
                -(item["end_ttcp_steps"] if item["end_ttcp_steps"] is not None else -1.0),
                item["end_dist_to_dest"],
                item["end_heading_error_to_dest_deg"],
                abs(item["turn_deg"]),
            )
        )
        return int(candidates[0]["turn_deg"]) if candidates else 0

    def _best_same_side_turn(self, call_name: str, cur: ObsSnapshot, prev: Optional[ObsSnapshot]) -> int:
        candidate_bins = [turn_deg for turn_deg in self._allowed_turn_bins_for_call(call_name, cur) if turn_deg != 0]
        horizon_steps = self.config.turn_preview_steps if call_name == "EXECUTE_TURN" else self.config.conflict_lookahead_steps
        return self._best_turn_from_bins(cur, prev, candidate_bins, horizon_steps=horizon_steps)

    def _deterministic_merge_back_turn(self, cur: ObsSnapshot) -> int:
        err = _heading_error_to_dest_deg(cur)
        if abs(err) < self.config.heading_alignment_tolerance_deg:
            return 0
        return int(_snap_to_allowed_bin(err, ALLOWED_BINS))

    def _call_merge_back_for_reason(
        self,
        obs_vec: np.ndarray,
        step: int,
        cur: ObsSnapshot,
        *,
        reason: str,
    ) -> int:
        self.last_merge_back_reason = reason
        return self._call_llm_for_stage("MERGE_BACK", obs_vec, step, cur)

    def _build_turn_preview(self, call_name: str, cur: ObsSnapshot, prev: Optional[ObsSnapshot]) -> None:
        if call_name not in {"EXECUTE_TURN", "EMERGENCY_MANEUVER"}:
            self.last_turn_preview_rows = []
            return

        rows: List[Dict[str, Any]] = []
        for turn_deg in self._allowed_turn_bins_for_call(call_name, cur):
            candidate = self._simulate_turn_candidate(
                cur,
                prev,
                turn_deg=turn_deg,
                horizon_steps=self.config.turn_preview_steps,
            )
            rows.append(
                {
                    "heading_change_deg": int(turn_deg),
                    "safe_preserved_over_preview": bool(candidate["safe_preserved_over_preview"]),
                    "predicted_loss_step": candidate["loss_step"],
                    "stays_inside_sector_over_preview": bool(candidate["stays_inside_sector_over_preview"]),
                    "boundary_exit_step": candidate["boundary_exit_step"],
                    "merge_back_turn_deg_after_preview": int(candidate["merge_back_turn_deg_after_preview"]),
                }
            )
        self.last_turn_preview_rows = rows

    def _should_repeat_merge_back(self, cur: ObsSnapshot) -> bool:
        return abs(_heading_error_to_dest_deg(cur)) > self.config.merge_back_repeat_error_deg

    def _build_memory_features(self, call_name: str, cur: ObsSnapshot) -> MemoryFeatures:
        predicted_loss_step = (
            self.last_execute_predicted_loss_step if call_name == "EXECUTE_TURN" else self.last_predicted_loss_step
        )
        return MemoryFeatures(
            call_name=call_name,
            intruder_bearing_deg=round(_bearing_error_to_intruder_deg(cur), 3),
            intruder_distance=round(cur.sep_oi, 3),
            destination_bearing_deg=round(_heading_error_to_dest_deg(cur), 3),
            destination_distance=round(cur.dist_o_dest, 3),
            predicted_loss_step=None if predicted_loss_step is None else int(predicted_loss_step),
            ttcp_steps=None if self.last_predicted_ttcp_steps is None else round(self.last_predicted_ttcp_steps, 3),
            boundary_warning_active=bool(self.last_boundary_warning_active),
            boundary_exit_step=None if self.last_boundary_exit_step is None else int(self.last_boundary_exit_step),
            boundary_distance=None if self.last_boundary_distance is None else round(self.last_boundary_distance, 3),
            safe_streak_steps=int(self.safe_streak_steps),
            merge_back_mode=str(self.last_merge_back_mode),
        )

    def _prepare_memory_context(self, call_name: str, cur: ObsSnapshot) -> None:
        self.last_memory_prompt_cards = []
        self.last_memory_match_debug = []
        self.last_memory_features = None

        if not self.config.use_memory:
            return

        query_features = self._build_memory_features(call_name, cur)
        self.last_memory_features = query_features
        matches = self.memory_store.retrieve(
            query_features,
            top_k=self.config.memory_top_k,
            min_similarity=self.config.memory_min_similarity,
            safe_r=self.config.safe_r,
            turn_preview_steps=self.config.turn_preview_steps,
            conflict_lookahead_steps=self.config.conflict_lookahead_steps,
            boundary_lookahead_steps=self.config.boundary_lookahead_steps,
            safe_streak_required=self.config.safe_streak_required,
        )
        self.last_memory_prompt_cards = [
            format_memory_card(index + 1, match) for index, match in enumerate(matches)
        ]
        self.last_memory_match_debug = [
            {
                "case_id": match.case.case_id,
                "similarity": round(match.similarity, 3),
                "heading_change_deg": int(match.case.action_heading_change_deg),
                "outcome_label": match.case.outcome_label,
            }
            for match in matches
        ]

    def _buffer_memory_case(self, call_name: str, step: int, applied_turn: int, normalized: Dict[str, Any]) -> None:
        if not self.config.use_memory or self.last_memory_features is None or self.last_tag is None:
            return

        answer = normalized.get("answer") or {}
        audit_summary = (
            str(answer.get("technical_summary") or "")
            or str(answer.get("scenario_summary") or "")
            or str(answer.get("rationale") or "")
        )
        self.memory_store.buffer_case(
            call_tag=self.last_tag,
            call_name=call_name,
            step=step,
            features=self.last_memory_features,
            heading_change_deg=int(applied_turn),
            audit_summary=audit_summary[:200],
        )

    def finalize_episode_memory(self, *, run_id: str, episode_id: str, success: bool, outcome_label: str) -> int:
        if not self.config.use_memory:
            self.memory_store.discard_buffer()
            return 0
        if not success:
            self.memory_store.discard_buffer()
            return 0
        return self.memory_store.commit_buffer(
            run_id=str(run_id),
            episode_id=str(episode_id),
            outcome_label=str(outcome_label),
        )

    def _build_fallback_answer(
        self,
        call_name: str,
        cur: ObsSnapshot,
        prev: Optional[ObsSnapshot],
    ) -> Dict[str, Any]:
        if call_name == "EXECUTE_TURN":
            if self.last_execute_conflict_within_preview:
                fallback_turn = self._best_same_side_turn(call_name, cur, prev)
            else:
                fallback_turn = 0 if 0 in self._allowed_turn_bins_for_call(call_name, cur) else self._best_same_side_turn(call_name, cur, prev)
        elif call_name == "EMERGENCY_MANEUVER":
            fallback_turn = self._best_same_side_turn(call_name, cur, prev)
        elif call_name == "MERGE_BACK":
            fallback_turn = self._deterministic_merge_back_turn(cur)
        else:
            fallback_turn = 0

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
                    "heading_change_deg": int(fallback_turn),
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
            allowed_bins = self._allowed_turn_bins_for_call(call_name, cur)
            if requested_turn in allowed_bins:
                return requested_turn
            if requested_turn == 0 and 0 in allowed_bins:
                return 0
            side = self._intruder_side_sign(cur)
            if side > 0:
                return int(_snap_to_allowed_bin(abs(requested_turn), allowed_bins))
            if side < 0:
                return int(_snap_to_allowed_bin(-abs(requested_turn), allowed_bins))
            return self._best_same_side_turn(call_name, cur, prev)

        if call_name == "EMERGENCY_MANEUVER":
            allowed_bins = self._allowed_turn_bins_for_call(call_name, cur)
            if requested_turn in allowed_bins and requested_turn != 0:
                return requested_turn
            side = self._intruder_side_sign(cur)
            if side > 0:
                return int(_snap_to_allowed_bin(max(5, abs(requested_turn)), allowed_bins))
            if side < 0:
                return int(_snap_to_allowed_bin(-max(5, abs(requested_turn)), allowed_bins))
            return self._best_same_side_turn(call_name, cur, prev)

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

    def _call_llm_for_stage(
        self,
        call_name: str,
        obs_vec: np.ndarray,
        step: int,
        cur: ObsSnapshot,
    ) -> int:
        prev = ObsSnapshot.from_vector(self._prev_obs, step=self._prev_step) if self._prev_obs is not None else None
        self.last_intruder_side = self._intruder_side_label(cur)
        self.last_allowed_turn_bins = self._allowed_turn_bins_for_call(call_name, cur)
        self.last_merge_back_mode = self._merge_back_mode(cur)
        self._build_turn_preview(call_name, cur, prev)
        self._prepare_memory_context(call_name, cur)

        if call_name == "EXECUTE_TURN":
            zero_turn_allowed = 0 in self.last_allowed_turn_bins
            tag_suffix = "execute_turn"
            self.stage = "EXECUTE_TURN"
            self.last_event = f"execute_turn@{step}"
            self.last_annotations.append(self.config.annotation_execute_turn)
        elif call_name == "EMERGENCY_MANEUVER":
            zero_turn_allowed = False
            tag_suffix = f"emergency_maneuver_{self.emergency_count + 1:02d}"
            self.stage = "EMERGENCY_MANEUVER"
            self.last_event = f"emergency_maneuver@{step}"
            self.last_annotations.append(self.config.annotation_emergency)
        elif call_name == "MERGE_BACK":
            zero_turn_allowed = abs(_heading_error_to_dest_deg(cur)) <= self.config.heading_alignment_tolerance_deg
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
            normalized = self._build_fallback_answer(call_name, cur, prev)

        maneuver = normalized["answer"]["maneuver"]
        requested_turn = int(maneuver.get("heading_change_deg", 0))
        applied_turn = self._resolve_turn_for_call(call_name, requested_turn, cur, prev)

        maneuver["heading_change_deg"] = int(applied_turn)
        if call_name == "MERGE_BACK":
            self.merge_back_recheck_remaining = int(self.config.merge_back_recheck_steps)
        else:
            self.merge_back_recheck_remaining = 0

        normalized["controller_debug"] = {
            "intruder_side": self.last_intruder_side,
            "allowed_turn_bins": self.last_allowed_turn_bins,
            "turn_preview_rows": self.last_turn_preview_rows,
            "merge_back_mode": self.last_merge_back_mode,
            "merge_back_reason": self.last_merge_back_reason,
            "boundary_exit_step": self.last_boundary_exit_step,
            "boundary_warning_active": self.last_boundary_warning_active,
            "boundary_distance": self.last_boundary_distance,
            "merge_back_recheck_remaining": int(self.merge_back_recheck_remaining),
            "memory_enabled": bool(self.config.use_memory),
            "retrieved_memory_case_ids": [item["case_id"] for item in self.last_memory_match_debug],
            "retrieved_memory_scores": [item["similarity"] for item in self.last_memory_match_debug],
            "retrieved_memory_actions": [item["heading_change_deg"] for item in self.last_memory_match_debug],
        }

        self.did_call_llm = True
        self.last_call_name = call_name
        self.last_tag = f"t{step:03d}_{tag_suffix}"
        self.last_prompt_text = prompt_text
        self.last_raw_text = raw_text
        self.last_normalized = normalized
        self.last_turn_deg = int(applied_turn)
        self._buffer_memory_case(call_name, step, int(applied_turn), normalized)

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
            self.stage = "WAIT_FOR_SAFE"
        elif self.stage == "EMERGENCY_MANEUVER":
            self.stage = "WAIT_FOR_SAFE"
        elif self.stage == "MERGE_BACK":
            self.stage = "MERGE_BACK_RECHECK"

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
        self.last_execute_predicted_loss_step = self._predict_loss_step(
            cur,
            prev,
            heading_rad=preview_heading,
            horizon_steps=self.config.turn_preview_steps,
        )
        self.last_execute_conflict_within_preview = self.last_execute_predicted_loss_step is not None
        self.last_boundary_exit_step = self._predict_boundary_exit_step(cur, prev, heading_rad=preview_heading)
        self.last_boundary_distance = self._boundary_distance(cur)
        self.last_boundary_warning_active = (
            self.inside_sector and self.has_called_execute_turn and self.last_boundary_exit_step is not None
        )
        self.last_intruder_side = self._intruder_side_label(cur)
        self.last_merge_back_mode = self._merge_back_mode(cur)
        self.last_allowed_turn_bins = []

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

        if self.stage == "WAIT_FOR_SAFE":
            if self.last_conflict_within_lookahead and step >= self.start_llm_at_step:
                turn_now = self._call_llm_for_stage("EMERGENCY_MANEUVER", obs_vec, step, cur)
            elif self.last_boundary_warning_active and step >= self.start_llm_at_step:
                turn_now = self._call_merge_back_for_reason(
                    obs_vec,
                    step,
                    cur,
                    reason="BOUNDARY_LOOKAHEAD",
                )
            elif self.safe_streak_steps >= self.config.safe_streak_required and step >= self.start_llm_at_step:
                turn_now = self._call_merge_back_for_reason(
                    obs_vec,
                    step,
                    cur,
                    reason="SAFE_STREAK",
                )
            else:
                turn_now = 0
        elif self.stage == "MERGE_BACK_RECHECK":
            if self.last_conflict_within_lookahead and step >= self.start_llm_at_step:
                turn_now = self._call_llm_for_stage("EMERGENCY_MANEUVER", obs_vec, step, cur)
            elif self.merge_back_recheck_remaining > 0:
                self.merge_back_recheck_remaining -= 1
                turn_now = 0
            else:
                should_repeat = self._should_repeat_merge_back(cur)
                if step >= self.start_llm_at_step and (should_repeat or self.last_boundary_warning_active):
                    turn_now = self._call_merge_back_for_reason(
                        obs_vec,
                        step,
                        cur,
                        reason="ALIGNMENT_REPEAT" if should_repeat else "BOUNDARY_LOOKAHEAD",
                    )
                else:
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
