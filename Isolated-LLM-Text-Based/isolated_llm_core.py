"""
Minimal LLM guidance runtime for an **LLM-only** episode.

This file is intentionally a lightweight copy of the core pieces in:
`LLM-DRL restructured code/rl_llm/llm.py`

It omits (on purpose):
- PPO / hybrid fusion
- evaluator engine
- memory retrieval / _index.jsonl
- ghost compare

The goal is to let you iterate on prompt/signals in isolation, without touching the
original restructured pipeline.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import ollama
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURE_CSV = str(PROJECT_ROOT / "featurefile_transformed_2.csv")


PROMPT_INSTRUCTIONS = (
    "ROLE: You are a very cautious ATCO helper.\n"
    "GOALS: (1) Keep ownship safely away from the intruder. (2) Help start a safe turn.\n"
    "\n"
    "SAFETY NUMBER:\n"
    "- SAFE_R = 5 units. If separation is near/below this, fix safety first.\n"
    "\n"
    "WHAT YOU GET:\n"
    "- Current step, positions, destination, separation, heading, simple trends, TTCP (time to closest approach).\n"
    "- You will also be told which PHASE we are in. DO NOT change it. Only two phases exist: WAIT_TURN and EXECUTE_TURN.\n"
    "\n"
    "YOUR OUTPUT (strict):\n"
    "- Return EXACTLY ONE JSON with fields: scenario_summary, metrics (ttcp=null, separation_distance=null),\n"
    "  and maneuver { phase, heading_change_deg, hold_steps }.\n"
    "- heading_change_deg ∈ {-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30}.\n"
    "- Positive = LEFT. Negative = RIGHT. 0 = no turn.\n"
    "\n"
    "SIGN RULE (simple):\n"
    "- If intruder on LEFT (bearing_error_to_intruder > 0°), TURN LEFT (positive).\n"
    "- If intruder on RIGHT (bearing_error_to_intruder < 0°), TURN RIGHT (negative).\n"
    "- If intruder ~ahead (|bearing| < 15°) AND closing, use a BIGGER same-side turn.\n"
    "- SAFETY OVERRIDE: if separation < SAFE_R*1.2 or TTCP ≤ 3, pick the direction that increases separation fastest.\n"
    "\n"
    "PHASES (only these two; obey the phase you are given):\n"
    "- WAIT_TURN: heading_change_deg MUST BE 0. Only choose hold_steps (watch and wait). Keep hold short if risky.\n"
    "- EXECUTE_TURN: choose ONE turn using the sign rule. Prefer the smallest bin that improves safety (never output 0).\n"
    "  Also choose a small hold_steps to keep that heading for a few steps.\n"
    "\n"
    "PRIORITY ORDER:\n"
    "1) Safety: open separation / increase TTCP.\n"
    "2) Do not make ownship→destination distance worse if avoidable.\n"
    "3) Compare the LLM_return and ppo_return rewards, and in future similar contexts, favor the action and holding step associated with the higher reward."
    "\n"
    "\n"
    "YOUR OUTPUT (strict JSON):\n"
    "- Return EXACTLY ONE JSON object with this shape:\n"
    "{\n"
    "  \"answer\": {\n"
    "    \"scenario_summary\": \"<1–3 short sentences>\",\n"
    "    \"technical_summary\": \"<detailed technical description referencing observation, derived signals, kinematics, and recent trends>\",\n"
    "    \"rationale\": \"<clear reason for the chosen maneuver; mention if a PAST CASE informed the decision>\",\n"
    "    \"metrics\": { \"ttcp\": null, \"separation_distance\": null },\n"
    "    \"maneuver\": { \"phase\": \"WAIT_TURN\"|\"EXECUTE_TURN\", \"heading_change_deg\": -30..30 step 5, \"hold_steps\": <int≥0> }\n"
    "  }\n"
    "}\n"
    "A PHASE STATE block will be provided below. Use it exactly as given.\n"
    "=========================\n"
    "Current observation, derived signals, and recent history are:\n"
)


# --- Small typed view over your observation vector ---
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

    # Optional ATCO-path fields (present in current env observation)
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


def _wrap_deg(a: float) -> float:
    """Wrap any angle in degrees to (-180, 180]."""
    return ((a + 180.0) % 360.0) - 180.0


def _bearing_deg(dy: float, dx: float) -> float:
    """atan2(dy, dx) in degrees."""
    return math.degrees(math.atan2(dy, dx))


def _side_word(x: float, left: str = "left", right: str = "right", center: str = "centerline") -> str:
    if x > 0:
        return left
    if x < 0:
        return right
    return center


def _ahead_or_behind(abs_deg: float) -> str:
    return "ahead" if abs_deg < 90.0 else "abeam/behind"


class BasePromptBuilder:
    def __init__(self, *, feature_csv_path: str = DEFAULT_FEATURE_CSV, include_bc_baseline: bool = True):
        self._bc_text_cache: Optional[str] = None
        self._last_kinematics: Optional[dict] = None
        self._feature_csv_path = feature_csv_path
        self._include_bc_baseline = bool(include_bc_baseline)

    def build_base_prompt(
        self,
        obs_vector: np.ndarray,
        *,
        step: Optional[int] = None,
        prev_obs_vector: Optional[np.ndarray] = None,
        prev_step: Optional[int] = None,
    ) -> str:
        snap = ObsSnapshot.from_vector(obs_vector, step=step)
        prev = ObsSnapshot.from_vector(prev_obs_vector, step=prev_step) if prev_obs_vector is not None else None
        parts = [
            PROMPT_INSTRUCTIONS,
            self._current_observation_text(snap),
            self._derived_signals_text(snap, prev),
            self._kinematics_text(snap, prev),
            self._bc_baseline_text(csv_path=self._feature_csv_path) if self._include_bc_baseline else "",
        ]
        return "\n\n".join(p for p in parts if p)

    @property
    def last_kinematics(self) -> Optional[dict]:
        return self._last_kinematics

    def _kinematics_text(self, s: ObsSnapshot, prev: Optional[ObsSnapshot]) -> str:
        if prev is None:
            self._last_kinematics = {
                "ttcp_steps": None,
                "own_speed": None,
                "intr_speed": None,
                "rel_speed": None,
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
            ttcp_minutes = 2.0 * ttcp_steps
            ttcp_text = f"{ttcp_steps:.2f} steps (~{ttcp_minutes:.1f} min)"
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
        if s.step is not None:
            lines.append(f"- Step index: {s.step}")
        lines.append(f"- Ownship position: x={s.own_x:.1f}, y={s.own_y:.1f}")
        lines.append(f"- Intruder position: x={s.intr_x:.1f}, y={s.intr_y:.1f}")
        lines.append(f"- Ownship destination: x={s.dest_x:.1f}, y={s.dest_y:.1f}")
        lines.append(f"- Separation (ownship ↔ intruder): {s.sep_oi:.2f}")
        lines.append(f"- Distances: ownship→dest={s.dist_o_dest:.2f}, intruder→dest={s.dist_i_dest:.2f}")
        lines.append(
            f"- Deviation: cross-track-to-original={s.ctd_dist:.2f}, distance-to-ATCO-path={s.dist_to_atco_path:.2f}"
        )
        lines.append(
            f"- Ownship heading: {_rad2deg(s.own_heading_rad):.1f}° (0°=east, positive is counter-clockwise)"
        )

        if s.path_heading_deg is not None:
            lines.append(f"- ATCO path heading: {s.path_heading_deg:+.1f}°")
        if s.heading_error_to_path_deg is not None:
            lines.append(
                f"- heading_error_to_path_deg: {s.heading_error_to_path_deg:+.1f}° "
                f"(path is to your {_side_word(s.heading_error_to_path_deg)})"
            )
        if s.signed_cross_track_to_path is not None:
            lines.append(
                f"- signed_cross_track_to_path: {s.signed_cross_track_to_path:+.2f} (left of path is positive)"
            )

        # Optional sanity check if we have both headings.
        if s.path_heading_deg is not None and s.heading_error_to_path_deg is not None:
            own_h = _rad2deg(s.own_heading_rad)
            err_path_from_heading = _wrap_deg(float(s.path_heading_deg) - own_h)
            diff = abs(_wrap_deg(err_path_from_heading - float(s.heading_error_to_path_deg)))
            lines.append(
                f"- provided_heading_error_to_path: {float(s.heading_error_to_path_deg):+.1f}° (Δ={diff:.1f}°)"
            )

        return "\n".join(lines)

    def _derived_signals_text(self, s: ObsSnapshot, prev: Optional[ObsSnapshot]) -> str:
        own_heading_deg = _rad2deg(s.own_heading_rad)

        heading_to_dest_deg = _bearing_deg(s.dest_y - s.own_y, s.dest_x - s.own_x)
        heading_error_to_dest_deg = _wrap_deg(heading_to_dest_deg - own_heading_deg)

        own_to_intr_angle_deg = _bearing_deg(s.intr_y - s.own_y, s.intr_x - s.own_x)
        bearing_error_to_intr_deg = _wrap_deg(own_to_intr_angle_deg - own_heading_deg)
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

    def _bc_baseline_text(
        self,
        df: Optional[pd.DataFrame] = None,
        csv_path: Optional[str] = None,
        *,
        min_bucket_count: int = 5,  # kept for API compatibility; unused now
    ) -> str:
        if self._bc_text_cache is not None:
            return self._bc_text_cache

        if df is None:
            if csv_path is None:
                return ""
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                return f"(BC baseline unavailable: failed to load CSV: {e})"

        lines: List[str] = []
        lines.append("ATCO baseline (two-phase mode):")

        STEP_MINUTES = 2.0

        def _wrap_signed_deg(d):
            return (d + 180.0) % 360.0 - 180.0

        def _to_time(s):
            return pd.to_datetime(s, format="%H:%M:%S", errors="coerce")

        def _diff_steps(start_s, event_s):
            start_t = _to_time(start_s)
            event_t = _to_time(event_s)
            if pd.isna(start_t) or pd.isna(event_t):
                return np.nan
            delta = (event_t - start_t).total_seconds()
            if delta < 0:
                delta += 24 * 3600
            return int(round(delta / (STEP_MINUTES * 60.0)))

        def _hist_str(series_steps: pd.Series) -> str:
            if series_steps.empty:
                return "n/a"
            bins = [-0.5, 2.5, 5.5, 10.5, 15.5, 20.5, np.inf]
            labels = ["0–2", "3–5", "6–10", "11–15", "16–20", "21+"]
            vc = (
                pd.cut(series_steps.astype(float), bins=bins, labels=labels)
                .value_counts(normalize=True)
                .reindex(labels, fill_value=0.0)
            )
            return ", ".join(f"{lab}: {pct*100:.0f}%" for lab, pct in vc.items())

        # 1) Turn magnitude distribution
        turn_bins_series = None
        if {"initialhead_resolved", "initialhead_unresolved"}.issubset(df.columns):
            cols = df[["initialhead_resolved", "initialhead_unresolved"]].dropna()
            if len(cols):
                d_signed = cols.apply(
                    lambda r: _wrap_signed_deg(
                        float(r["initialhead_resolved"]) - float(r["initialhead_unresolved"])
                    ),
                    axis=1,
                )
                mags = d_signed.abs()
                bins = [0, 5, 10, 15, 20, 25, 30]
                turn_bins_series = mags.apply(lambda x: min(bins, key=lambda b: abs(b - x)))

        if turn_bins_series is not None and len(turn_bins_series):
            counts = turn_bins_series.value_counts(normalize=True).reindex([0, 5, 10, 15, 20, 25, 30], fill_value=0.0)
            top_bins = counts.sort_values(ascending=False).head(3)
            small_share = counts.loc[[0, 5, 10]].sum()
            large_share = counts.loc[[20, 25, 30]].sum()
            dist_str = ", ".join(f"{b}°:{pct*100:.0f}%" for b, pct in counts.items())
            top_str = ", ".join(f"{b}° ({pct*100:.0f}%)" for b, pct in top_bins.items())
            lines.append(
                "- Turn magnitudes (how big the first turn usually is): "
                f"{dist_str}. Most common bins → {top_str}. "
                f"Rule-of-thumb: small turns (≤10°) ≈ {small_share*100:.0f}% of cases; "
                f"large turns (≥20°) ≈ {large_share*100:.0f}%."
            )
        else:
            lines.append("- Turn magnitudes: not enough data.")

        # 2) Cross-track deviation percentiles
        ctd_col = None
        for name in ("CTD_distance", "ctd_distance", "cross_track_distance", "cross_track"):
            if name in df.columns:
                ctd_col = name
                break
        if ctd_col:
            s = pd.to_numeric(df[ctd_col], errors="coerce").dropna()
            if len(s):
                p10, p25, p50, p75, p90 = [float(s.quantile(q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90)]
                lines.append(
                    "- Off-route distance after turning (max cross-track): "
                    f"p10≈{p10:.2f}, p25≈{p25:.2f}, p50≈{p50:.2f}, p75≈{p75:.2f}, p90≈{p90:.2f}. "
                    "Interpretation: staying near the median is “typical”; exceeding p90 is “unusual”."
                )
            else:
                lines.append("- Off-route distance (cross-track): not enough data.")
        else:
            lines.append("- Off-route distance (cross-track): column not found.")

        # 3) First-turn timing summary
        start_time = None
        if "flightstarttime_original_res" in df.columns:
            start_time = df["flightstarttime_original_res"].copy()
            if "flightstarttime_original_unres" in df.columns:
                start_time = start_time.fillna(df["flightstarttime_original_unres"])
        elif "flightstarttime_original_unres" in df.columns:
            start_time = df["flightstarttime_original_unres"].copy()

        first_turn_steps = pd.Series(dtype=float)
        if start_time is not None and "manstrt_time" in df.columns:
            first_turn_steps = start_time.combine(df["manstrt_time"], _diff_steps).dropna().astype(int)

        if len(first_turn_steps):
            p25 = int(first_turn_steps.quantile(0.25))
            p50 = int(first_turn_steps.quantile(0.50))
            p75 = int(first_turn_steps.quantile(0.75))
            p80 = int(first_turn_steps.quantile(0.80))
            lines.append(
                "- When to start the first turn (steps from scenario start): "
                f"median ≈ {p50} (50% lie in {p25}–{p75}); ~80% by {p80}. "
                f"Histogram → { _hist_str(first_turn_steps) }."
            )
        else:
            lines.append("- First-turn timing: not enough data.")

        lines.append(
            "Baseline decision hints: start turns inside the typical window; "
            "prefer smaller bins when geometry is mild; avoid cross-track values beyond p90 unless necessary."
        )

        text = "\n".join(lines)
        self._bc_text_cache = text
        return text


# === Phase block insertion ===
PHASE_MARKER = "\n=========================\n"
ALLOWED_BINS: List[int] = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]


def attach_phase_state_block(
    prompt: str,
    *,
    current_phase: str,
    step_index: int,
    allowed_bins: Optional[list] = None,
    extra_notes: Optional[str] = None,
) -> str:
    bins = allowed_bins or ALLOWED_BINS
    block_lines = [
        "PHASE STATE (controller-supplied):",
        f"- current_phase: {current_phase}",
        f"- step_index: {step_index}",
        f"- allowed_bins: {bins}",
        "- VALID_PHASES: ['WAIT_TURN','EXECUTE_TURN']",
        "- enforcement:",
        "  - WAIT_TURN: heading_change_deg must be 0 (only set hold_steps).",
        "  - EXECUTE_TURN: turn AWAY from intruder; pick the smallest bin that opens separation; do not return 0.",
    ]
    if extra_notes:
        block_lines.append(f"- notes: {extra_notes}")
    block = "\n".join(block_lines) + "\n\n"

    idx = prompt.find(PHASE_MARKER)
    if idx == -1:
        return block + prompt
    # Insert with a newline so the last sentence before the marker doesn't run into the phase header.
    return prompt[:idx] + "\n" + block + prompt[idx:]


# === Ollama client ===
OLLAMA_MODEL = "llama3:8b"


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


# === JSON extraction + normalization ===
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _snap_to_allowed_bin(x: float) -> int:
    return min(ALLOWED_BINS, key=lambda b: abs(b - x))


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default


def _extract_json(text: str) -> Dict[str, Any]:
    m = JSON_OBJECT_RE.search(text.strip())
    if not m:
        raise ValueError("No JSON object found in LLM output.")
    return json.loads(m.group(0))


def _validate_and_normalize(payload: Dict[str, Any], expected_phase: Optional[str] = None) -> Dict[str, Any]:
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

    allowed_phases = {"WAIT_TURN", "EXECUTE_TURN"}
    raw_phase = man.get("phase", "")
    phase = expected_phase if expected_phase in allowed_phases else (raw_phase if raw_phase in allowed_phases else "WAIT_TURN")
    man["phase"] = phase

    raw_h = man.get("heading_change_deg", 0)
    try:
        h = float(raw_h)
    except Exception:
        h = 0.0
    snapped = _snap_to_allowed_bin(h)

    if phase == "WAIT_TURN":
        snapped = 0
    elif phase == "EXECUTE_TURN":
        if snapped == 0:
            snapped = 5
    man["heading_change_deg"] = int(snapped)

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


# === Action bin mapping ===
def _deg_to_action_idx(delta_deg: int) -> int:
    d = int(round(delta_deg))
    if d % 5 != 0 or abs(d) > 30:
        raise ValueError(f"delta must be multiple of 5 and |d|<=30, got {delta_deg}")
    if d >= 0:
        return d // 5
    return 6 + (abs(d) // 5)


# === Minimal LLM-only controller (two-phase + hold) ===
PHASE_MIN_HOLD = {"WAIT_TURN": 0, "EXECUTE_TURN": 3}
PHASE_MAX_HOLD = {"WAIT_TURN": 1, "EXECUTE_TURN": 6}


class LLMOnlyEpisodeController:
    """
    LLM-only controller:
    - Calls the LLM once per phase (WAIT_TURN then EXECUTE_TURN).
    - Applies the chosen heading change ONCE, then holds straight flight (0°) for `hold_steps` future steps.
    """

    def __init__(
        self,
        builder: BasePromptBuilder,
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

        self.phase = "WAIT_TURN"
        self.hold_remaining = 0
        self._phase_called = False

        self._prev_obs: Optional[np.ndarray] = None
        self._prev_step: Optional[int] = None

        # Exposed for the runner to save for debugging.
        self.did_call_llm: bool = False
        self.last_tag: Optional[str] = None
        self.last_prompt_text: Optional[str] = None
        self.last_raw_text: Optional[str] = None
        self.last_normalized: Optional[dict] = None

    def reset(self, initial_phase: str = "WAIT_TURN") -> None:
        self.phase = initial_phase
        self.hold_remaining = 0
        self._phase_called = False
        self._prev_obs = None
        self._prev_step = None
        self.did_call_llm = False
        self.last_tag = None
        self.last_prompt_text = None
        self.last_raw_text = None
        self.last_normalized = None

    def _clamp_hold(self, phase: str, hold: int) -> int:
        lo = PHASE_MIN_HOLD.get(phase, 0)
        hi = PHASE_MAX_HOLD.get(phase, hold)
        return max(lo, min(hi, int(hold)))

    def _build_prompt(self, obs_vec: np.ndarray, step: int) -> str:
        prompt_core = self.builder.build_base_prompt(
            obs_vector=obs_vec,
            step=step,
            prev_obs_vector=self._prev_obs,
            prev_step=self._prev_step,
        )
        return attach_phase_state_block(
            prompt_core,
            current_phase=self.phase,
            step_index=step,
            allowed_bins=ALLOWED_BINS,
            extra_notes=None,
        )

    def next_turn_deg(self, obs_vec: np.ndarray, step: int) -> int:
        self.did_call_llm = False

        # Gate calls until the configured start step.
        if step < self.start_llm_at_step:
            self._prev_obs, self._prev_step = obs_vec.copy(), step
            return 0

        # During hold: fly straight (0°), but detect when WAIT_TURN hold ends to advance phase.
        if self.hold_remaining > 0:
            self.hold_remaining -= 1
            if self.hold_remaining == 0 and self.phase == "WAIT_TURN":
                self.phase = "EXECUTE_TURN"
                self._phase_called = False
            self._prev_obs, self._prev_step = obs_vec.copy(), step
            return 0

        # If we've already called the LLM for EXECUTE_TURN, do nothing else for the episode.
        if self._phase_called and self.phase == "EXECUTE_TURN":
            self._prev_obs, self._prev_step = obs_vec.copy(), step
            return 0

        # Call LLM once for the current phase.
        phase_at_call = self.phase
        prompt_text = self._build_prompt(obs_vec, step)
        raw_text = ollama_invoke(
            prompt_text,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        try:
            parsed = _extract_json(raw_text)
        except Exception:
            parsed = {
                "answer": {
                    "scenario_summary": "",
                    "technical_summary": "",
                    "rationale": "",
                    "metrics": {"ttcp": None, "separation_distance": None},
                    "maneuver": {"phase": self.phase, "heading_change_deg": 0, "hold_steps": 0},
                }
            }

        normalized = _validate_and_normalize(parsed, expected_phase=phase_at_call)
        man = normalized["answer"]["maneuver"]
        turn_now = int(man.get("heading_change_deg", 0))
        hold_steps = int(max(0, int(man.get("hold_steps", 0))))
        hold_steps = self._clamp_hold(phase_at_call, hold_steps)

        self._phase_called = True
        self.hold_remaining = hold_steps

        # If WAIT_TURN asks for hold=0, advance phase immediately (next call will query EXECUTE_TURN).
        if self.hold_remaining == 0 and phase_at_call == "WAIT_TURN":
            self.phase = "EXECUTE_TURN"
            self._phase_called = False

        self.did_call_llm = True
        self.last_tag = f"t{step:03d}_{phase_at_call.lower()}"
        self.last_prompt_text = prompt_text
        self.last_raw_text = raw_text
        self.last_normalized = normalized

        self._prev_obs, self._prev_step = obs_vec.copy(), step
        return turn_now


__all__ = ["ALLOWED_BINS", "BasePromptBuilder", "LLMOnlyEpisodeController", "_deg_to_action_idx"]
