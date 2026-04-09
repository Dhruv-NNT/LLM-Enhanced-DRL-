"""
Sector-entry LLM guidance runtime for an isolated episode.

This file is a sibling variant of `isolated_llm_core.py`.
It keeps the original isolated files untouched and implements a different
3-stage controller:

1) FOLLOW_LINESTRING
2) AVOID_INTRUDER
3) RETURN_TO_DESTINATION
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent

PROMPT_INSTRUCTIONS = (
    "ROLE: You are a very cautious ATCO helper.\n"
    "STAGE: You are being called only for the AVOID_INTRUDER stage after ownship has entered the sector polygon.\n"
    "GOAL: Choose a short avoidance maneuver that improves safety and increases TTCP.\n"
    "\n"
    "LOCKED RULES:\n"
    "- SAFE_R = 5 units.\n"
    "- Turn TOWARD the intruder.\n"
    "- If intruder is on LEFT (bearing_error_to_intruder > 0°), choose a LEFT turn (positive).\n"
    "- If intruder is on RIGHT (bearing_error_to_intruder < 0°), choose a RIGHT turn (negative).\n"
    "- If intruder is nearly ahead and closing, you may choose a larger same-side turn.\n"
    "- Do NOT plan the return to destination. The controller handles recovery separately after 5 observed safe steps.\n"
    "\n"
    "OUTPUT RULES:\n"
    "- Return EXACTLY ONE JSON object.\n"
    "- heading_change_deg must be one of {-30,-25,-20,-15,-10,-5,5,10,15,20,25,30}. Do not return 0 unless you truly cannot parse the situation.\n"
    "- hold_steps should be short, usually 0..3.\n"
    "\n"
    "STRICT JSON SHAPE:\n"
    "{\n"
    "  \"answer\": {\n"
    "    \"scenario_summary\": \"<1-3 short sentences>\",\n"
    "    \"technical_summary\": \"<brief technical summary>\",\n"
    "    \"rationale\": \"<why this same-side avoidance turn helps>\",\n"
    "    \"metrics\": { \"ttcp\": null, \"separation_distance\": null },\n"
    "    \"maneuver\": { \"heading_change_deg\": -30..30 step 5, \"hold_steps\": <int>=0 }\n"
    "  }\n"
    "}\n"
    "\n"
    "Current observation and derived signals are:\n"
)

ALLOWED_BINS: List[int] = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
SAFE_R = 5.0
LOSS_PREVIEW_STEPS = 5
SAFE_STREAK_REQUIRED = 5
AVOID_HOLD_MAX = 3
OLLAMA_MODEL = "llama3:8b"
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
        s = ObsSnapshot(
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
            s.path_heading_deg = float(vec[12])
        if len(vec) >= 14:
            s.heading_error_to_path_deg = float(vec[13])
        if len(vec) >= 15:
            s.signed_cross_track_to_path = float(vec[14])
        return s


def _rad2deg(r: float) -> float:
    return float(r) * 180.0 / math.pi


def _deg2rad(d: float) -> float:
    return float(d) * math.pi / 180.0


def _wrap_deg(a: float) -> float:
    return ((a + 180.0) % 360.0) - 180.0


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
    m = JSON_OBJECT_RE.search(text.strip())
    if not m:
        raise ValueError("No JSON object found in LLM output.")
    return json.loads(m.group(0))


def _snap_to_allowed_bin(x: float) -> int:
    return min(ALLOWED_BINS, key=lambda b: abs(b - x))


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default


def _deg_to_action_idx(delta_deg: int) -> int:
    d = int(round(delta_deg))
    if d % 5 != 0 or abs(d) > 30:
        raise ValueError(f"delta must be multiple of 5 and |d|<=30, got {delta_deg}")
    if d >= 0:
        return d // 5
    return 6 + (abs(d) // 5)


def _heading_error_to_dest_deg(s: ObsSnapshot) -> float:
    heading_to_dest_deg = _bearing_deg(s.dest_y - s.own_y, s.dest_x - s.own_x)
    return _wrap_deg(heading_to_dest_deg - _rad2deg(s.own_heading_rad))


def _bearing_error_to_intruder_deg(s: ObsSnapshot) -> float:
    own_to_intr_angle_deg = _bearing_deg(s.intr_y - s.own_y, s.intr_x - s.own_x)
    return _wrap_deg(own_to_intr_angle_deg - _rad2deg(s.own_heading_rad))


def _target_heading_to_destination_rad(s: ObsSnapshot) -> float:
    return math.atan2(s.dest_y - s.own_y, s.dest_x - s.own_x)


def _same_side_turn_toward_intruder_deg(s: ObsSnapshot, magnitude: int = 5) -> int:
    bearing_err = _bearing_error_to_intruder_deg(s)
    mag = int(abs(magnitude))
    if bearing_err > 0:
        return mag
    if bearing_err < 0:
        return -mag
    return mag


def _validate_and_normalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "answer" not in payload or not isinstance(payload["answer"], dict):
        raise ValueError("Missing 'answer' object.")
    ans = payload["answer"]

    if "scenario_summary" not in ans or not isinstance(ans["scenario_summary"], str):
        ans["scenario_summary"] = ""
    if "technical_summary" not in ans or not isinstance(ans["technical_summary"], str):
        ans["technical_summary"] = ""
    if "rationale" not in ans or not isinstance(ans["rationale"], str):
        ans["rationale"] = ""

    ans.setdefault("metrics", {})
    metrics = ans["metrics"]
    for k in ("ttcp", "separation_distance"):
        if k in metrics and metrics[k] is not None:
            try:
                metrics[k] = float(metrics[k])
            except Exception:
                metrics[k] = None
        else:
            metrics[k] = None

    ans.setdefault("maneuver", {})
    man = ans["maneuver"]

    raw_h = man.get("heading_change_deg", 0)
    try:
        h = float(raw_h)
    except Exception:
        h = 0.0
    man["heading_change_deg"] = int(_snap_to_allowed_bin(h))

    raw_hold = man.get("hold_steps", 0)
    hold = _coerce_int(raw_hold, default=0)
    if hold < 0:
        hold = 0
    man["hold_steps"] = int(hold)

    return {
        "answer": {
            "scenario_summary": ans["scenario_summary"],
            "technical_summary": ans["technical_summary"],
            "rationale": ans["rationale"],
            "metrics": metrics,
            "maneuver": man,
        }
    }


def ollama_invoke(
    prompt_text: str,
    *,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 8192,
) -> str:
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        options={"temperature": temperature, "top_p": top_p, "num_predict": max_tokens},
    )
    return (resp.get("message") or {}).get("content", "") or ""


class SectorEntryPromptBuilder:
    def __init__(self) -> None:
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
    ) -> str:
        snap = ObsSnapshot.from_vector(obs_vector, step=step)
        prev = ObsSnapshot.from_vector(prev_obs_vector, step=prev_step) if prev_obs_vector is not None else None
        parts = [
            PROMPT_INSTRUCTIONS,
            self._current_observation_text(snap),
            self._derived_signals_text(snap, prev),
            self._kinematics_text(snap, prev),
            self._guidance_state_text(guidance_state),
        ]
        return "\n\n".join(p for p in parts if p)

    def _kinematics_text(self, s: ObsSnapshot, prev: Optional[ObsSnapshot]) -> str:
        if prev is None:
            self._last_kinematics = {
                "ttcp_steps": None,
                "own_speed": 0.0,
                "intr_speed": 0.0,
                "rel_speed": 0.0,
                "sep": float(s.sep_oi),
            }
            return (
                "Kinematics (needs a previous observation to compute):\n"
                "- Ownship speed: N/A\n- Intruder speed: N/A\n"
                "- Relative speed: N/A\n- TTCP (to CPA): N/A"
            )

        dt_steps = 1
        if s.step is not None and prev.step is not None and s.step != prev.step:
            dt_steps = max(1, s.step - prev.step)

        dx_o = s.own_x - prev.own_x
        dy_o = s.own_y - prev.own_y
        dx_i = s.intr_x - prev.intr_x
        dy_i = s.intr_y - prev.intr_y

        v_o = math.hypot(dx_o, dy_o) / dt_steps
        v_i = math.hypot(dx_i, dy_i) / dt_steps

        rx = s.intr_x - s.own_x
        ry = s.intr_y - s.own_y
        vrx = (dx_i - dx_o) / dt_steps
        vry = (dy_i - dy_o) / dt_steps
        vrel_mag2 = vrx * vrx + vry * vry

        if vrel_mag2 > 0:
            t_star = -((rx * vrx + ry * vry) / vrel_mag2)
            ttcp_steps = max(0.0, t_star)
            ttcp_text = f"{ttcp_steps:.2f} steps"
        else:
            ttcp_steps = None
            ttcp_text = "N/A (relative speed ~ 0)"

        rel_speed = math.sqrt(vrel_mag2)
        self._last_kinematics = {
            "ttcp_steps": ttcp_steps,
            "own_speed": float(v_o),
            "intr_speed": float(v_i),
            "rel_speed": float(rel_speed),
            "sep": float(s.sep_oi),
        }

        lines = [
            "Kinematics (scenario-units per step; 1 step = 2 min):",
            f"- Ownship speed: {v_o:.3f}",
            f"- Intruder speed: {v_i:.3f}",
            f"- Relative speed |v_rel|: {rel_speed:.3f}",
            f"- TTCP (to CPA): {ttcp_text}",
        ]
        return "\n".join(lines)

    def _current_observation_text(self, s: ObsSnapshot) -> str:
        lines = ["Current situation (all numbers are in scenario units):"]
        lines.append(f"- Step index: {s.step}")
        lines.append(f"- Ownship position: x={s.own_x:.1f}, y={s.own_y:.1f}")
        lines.append(f"- Intruder position: x={s.intr_x:.1f}, y={s.intr_y:.1f}")
        lines.append(f"- Ownship destination: x={s.dest_x:.1f}, y={s.dest_y:.1f}")
        lines.append(f"- Separation (ownship ↔ intruder): {s.sep_oi:.2f}")
        lines.append(f"- Distances: ownship→dest={s.dist_o_dest:.2f}, intruder→dest={s.dist_i_dest:.2f}")
        lines.append(
            f"- Deviation: cross-track-to-original={s.ctd_dist:.2f}, distance-to-ATCO-path={s.dist_to_atco_path:.2f}"
        )
        lines.append(f"- Ownship heading: {_rad2deg(s.own_heading_rad):.1f}° (0°=east, positive is counter-clockwise)")
        if s.path_heading_deg is not None:
            lines.append(f"- ATCO path heading: {s.path_heading_deg:+.1f}°")
        if s.heading_error_to_path_deg is not None:
            lines.append(
                f"- heading_error_to_path_deg: {s.heading_error_to_path_deg:+.1f}° "
                f"(path is to your {_side_word(s.heading_error_to_path_deg)})"
            )
        return "\n".join(lines)

    def _derived_signals_text(self, s: ObsSnapshot, prev: Optional[ObsSnapshot]) -> str:
        heading_error_to_dest_deg = _heading_error_to_dest_deg(s)
        bearing_error_to_intr_deg = _bearing_error_to_intruder_deg(s)
        intr_pos_rel = _ahead_or_behind(abs(bearing_error_to_intr_deg))

        if prev is not None:
            sep_trend = s.sep_oi - prev.sep_oi
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

    def _guidance_state_text(self, guidance_state: Dict[str, Any]) -> str:
        lines = ["Controller-supplied guidance state:"]
        lines.append(f"- current_stage: {guidance_state.get('current_stage')}")
        lines.append(f"- inside_sector: {guidance_state.get('inside_sector')}")
        lines.append(f"- sector_entry_step: {guidance_state.get('sector_entry_step')}")
        lines.append(f"- predicted_loss_step: {guidance_state.get('predicted_loss_step')}")
        lines.append(f"- current_ttcp_steps: {guidance_state.get('current_ttcp_steps')}")
        lines.append(f"- safe_streak_steps: {guidance_state.get('safe_streak_steps')}")
        lines.append("- turn_rule: turn TOWARD the intruder using the same-side sign rule.")
        lines.append("- output: choose an avoidance turn and a short hold only.")
        return "\n".join(lines)


class LLMSectorEntryEpisodeController:
    def __init__(
        self,
        builder: SectorEntryPromptBuilder,
        *,
        start_llm_at_step: int = 0,
        model: str = OLLAMA_MODEL,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 8192,
    ):
        self.builder = builder
        self.start_llm_at_step = int(start_llm_at_step)
        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.sector_polygon = _load_sector_polygon()
        self.reset()

    def reset(self) -> None:
        self.stage = "FOLLOW_LINESTRING"
        self.inside_sector = False
        self.sector_entry_step: Optional[int] = None
        self.recovery_start_step: Optional[int] = None
        self.safe_streak_steps = 0
        self.hold_remaining = 0
        self.last_turn_deg = 0
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_step: Optional[int] = None
        self.did_call_llm = False
        self.last_tag: Optional[str] = None
        self.last_prompt_text: Optional[str] = None
        self.last_raw_text: Optional[str] = None
        self.last_normalized: Optional[dict] = None
        self.last_predicted_ttcp_steps: Optional[float] = None
        self.last_predicted_loss_step: Optional[int] = None
        self.last_safe_loss_trigger = False
        self.last_event: Optional[str] = None

    def mark_finished(self) -> None:
        self.stage = "FINISHED"

    def _within_sector_polygon(self, s: ObsSnapshot) -> bool:
        return bool(self.sector_polygon.covers(Point(s.own_x, s.own_y)))

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
        horizon_steps: int = LOSS_PREVIEW_STEPS,
    ) -> Optional[int]:
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

        for step_ahead in range(1, horizon_steps + 1):
            own_x += own_speed * ux
            own_y += own_speed * uy
            intr_x += intr_vx
            intr_y += intr_vy
            sep = math.hypot(intr_x - own_x, intr_y - own_y)
            if sep < SAFE_R:
                return step_ahead
        return None

    def _follow_linestring_turn_deg(self, s: ObsSnapshot, step: int) -> int:
        if step < self.start_llm_at_step:
            return 0
        if s.heading_error_to_path_deg is None:
            return 0
        err = float(s.heading_error_to_path_deg)
        if abs(err) < 2.5:
            return 0
        return int(_snap_to_allowed_bin(err))

    def _return_to_destination_turn_deg(self, s: ObsSnapshot) -> int:
        err = _heading_error_to_dest_deg(s)
        if abs(err) < 2.5:
            return 0
        return int(_snap_to_allowed_bin(err))

    def _guidance_state(self, cur: ObsSnapshot) -> Dict[str, Any]:
        return {
            "current_stage": self.stage,
            "inside_sector": self.inside_sector,
            "sector_entry_step": self.sector_entry_step,
            "predicted_loss_step": self.last_predicted_loss_step,
            "current_ttcp_steps": None if self.last_predicted_ttcp_steps is None else round(self.last_predicted_ttcp_steps, 3),
            "safe_streak_steps": self.safe_streak_steps,
            "last_turn_deg": self.last_turn_deg,
            "step": cur.step,
        }

    def _call_llm_for_avoidance(self, obs_vec: np.ndarray, step: int, cur: ObsSnapshot) -> int:
        prompt_text = self.builder.build_prompt(
            obs_vector=obs_vec,
            step=step,
            prev_obs_vector=self._prev_obs,
            prev_step=self._prev_step,
            guidance_state=self._guidance_state(cur),
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
            normalized = _validate_and_normalize(parsed)
        except Exception:
            normalized = {
                "answer": {
                    "scenario_summary": "",
                    "technical_summary": "",
                    "rationale": "",
                    "metrics": {"ttcp": self.last_predicted_ttcp_steps, "separation_distance": cur.sep_oi},
                    "maneuver": {"heading_change_deg": 0, "hold_steps": 0},
                }
            }

        man = normalized["answer"]["maneuver"]
        turn_now = int(man.get("heading_change_deg", 0))
        if turn_now == 0:
            turn_now = _same_side_turn_toward_intruder_deg(cur, magnitude=5)
            normalized["answer"]["maneuver"]["heading_change_deg"] = int(turn_now)

        hold_steps = int(max(0, int(man.get("hold_steps", 0))))
        hold_steps = min(AVOID_HOLD_MAX, hold_steps)

        self.did_call_llm = True
        self.last_tag = f"t{step:03d}_avoid_intruder"
        self.last_prompt_text = prompt_text
        self.last_raw_text = raw_text
        self.last_normalized = normalized
        self.hold_remaining = hold_steps
        self.last_turn_deg = int(turn_now)
        return int(turn_now)

    def next_turn_deg(self, obs_vec: np.ndarray, step: int) -> int:
        self.did_call_llm = False
        self.last_event = None

        cur = ObsSnapshot.from_vector(obs_vec, step=step)
        prev = ObsSnapshot.from_vector(self._prev_obs, step=self._prev_step) if self._prev_obs is not None else None

        inside_now = self._within_sector_polygon(cur)
        if inside_now and not self.inside_sector:
            self.inside_sector = True
            self.sector_entry_step = step
            self.last_event = f"entered_sector@{step}"
        elif not inside_now:
            self.inside_sector = False

        self.last_predicted_ttcp_steps = self._estimate_ttcp_steps(cur, prev)

        if self.stage == "FOLLOW_LINESTRING":
            preview_heading_rad = _deg2rad(cur.path_heading_deg) if cur.path_heading_deg is not None else cur.own_heading_rad
        elif self.stage == "RETURN_TO_DESTINATION":
            preview_heading_rad = _target_heading_to_destination_rad(cur)
        else:
            preview_heading_rad = cur.own_heading_rad

        self.last_predicted_loss_step = self._predict_loss_step(cur, prev, heading_rad=preview_heading_rad)
        self.last_safe_loss_trigger = (
            self.last_predicted_loss_step is not None and self.last_predicted_loss_step < LOSS_PREVIEW_STEPS
        )

        if self.stage in ("AVOID_INTRUDER", "RETURN_TO_DESTINATION") and self.inside_sector:
            if cur.sep_oi >= SAFE_R:
                self.safe_streak_steps += 1
            else:
                self.safe_streak_steps = 0
        else:
            self.safe_streak_steps = 0

        if not self.inside_sector:
            self.stage = "FOLLOW_LINESTRING"
            self.hold_remaining = 0
            turn_now = self._follow_linestring_turn_deg(cur, step)
            self._prev_obs, self._prev_step = obs_vec.copy(), step
            self.last_turn_deg = int(turn_now)
            return int(turn_now)

        if self.stage == "FOLLOW_LINESTRING" and self.last_safe_loss_trigger and step >= self.start_llm_at_step:
            self.stage = "AVOID_INTRUDER"
            self.safe_streak_steps = 0
            self.last_event = f"avoidance_trigger@{step}"

        if self.stage == "AVOID_INTRUDER" and self.safe_streak_steps >= SAFE_STREAK_REQUIRED:
            self.stage = "RETURN_TO_DESTINATION"
            self.recovery_start_step = step
            self.hold_remaining = 0
            self.last_event = f"recovery_start@{step}"

        if self.stage == "RETURN_TO_DESTINATION" and self.last_safe_loss_trigger and step >= self.start_llm_at_step:
            self.stage = "AVOID_INTRUDER"
            self.safe_streak_steps = 0
            self.last_event = f"conflict_reappeared@{step}"

        if self.stage == "FOLLOW_LINESTRING":
            turn_now = self._follow_linestring_turn_deg(cur, step)
        elif self.stage == "AVOID_INTRUDER":
            if step < self.start_llm_at_step:
                turn_now = self._follow_linestring_turn_deg(cur, step)
            elif self.hold_remaining > 0:
                self.hold_remaining -= 1
                turn_now = 0
            elif self.last_safe_loss_trigger:
                turn_now = self._call_llm_for_avoidance(obs_vec, step, cur)
            else:
                turn_now = 0
        elif self.stage == "RETURN_TO_DESTINATION":
            turn_now = self._return_to_destination_turn_deg(cur)
            self.last_turn_deg = int(turn_now)
        else:
            turn_now = 0

        self._prev_obs, self._prev_step = obs_vec.copy(), step
        return int(turn_now)


__all__ = [
    "ALLOWED_BINS",
    "LLMSectorEntryEpisodeController",
    "SectorEntryPromptBuilder",
    "_deg_to_action_idx",
]
