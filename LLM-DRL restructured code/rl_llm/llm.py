"""LLM guidance helpers derived from RL - LLM (Complete Code Pipeline).ipynb."""
from __future__ import annotations

# Copy lets us duplicate objects without sharing state.
import copy
# Glob helps locate files based on patterns.
import glob
# Hashlib lets us build stable fingerprints for caching.
import hashlib
# Json is used for serialising prompts and answers.
import json
# Math supplies trigonometry and constants.
import math
# Os handles directories, paths, and process info.
import os
# Random is used for sampling replay cases.
import random
# Re powers regular expression checks.
import re
# Shutil helps with folder operations like moving logs.
import shutil
# Traceback prints readable exception stacks.
import traceback
# Deque gives us a fast append/pop history buffer.
from collections import deque
# Dataclass helps define simple containers with defaults.
from dataclasses import dataclass
# Datetime records timestamps with timezone info.
from datetime import datetime, timezone
# SimpleNamespace stores flexible attribute bags.
from types import SimpleNamespace
# Typing gives readable type hints to guide usage.
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

# NumPy handles vector math for observations and cases.
import numpy as np
# Ollama is the local LLM client.
import ollama
# Pandas loads answer history tables.
import pandas as pd
# Torch handles tensors used inside PPO helper calls.
import torch

# Shared configuration paths for memory and logs.
from configs import JSON_ANSWERS_DIR, MEMORY_DIR, EVAL_MEMORY_PATH, FEATUREFILE_PATH
# We reuse the same torch device as the PPO module.
from .ppo import device

if TYPE_CHECKING:  # pragma: no cover
    from .env import StructuredEnv
    from .ppo import PPO

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

EVALUATOR_PROMPT_TEMPLATE = (
      "You are an evaluator of air-traffic maneuvers.\n"
      "Apply the rubric BELOW based on the PHASE in the provided CONTEXT (maneuver.phase).\n"
      "Judge whether the suggested action improved safety and is phase-consistent.\n"
      "\n"
      
      "PHASE-AWARE RUBRIC\n"
        "\n"
        "ROLE & GOALS:\n"
        "- You are a very cautious ATCO helper.\n"
        "- Goals: (1) Keep ownship safely away from the intruder. (2) Help start a safe turn.\n"
        "SAFETY NUMBER: SAFE_R = 5 units. If separation is near/below this, fix safety first.\n"
        "SIGN RULE (for turn direction):\n"
        "- If intruder on LEFT (bearing_error_to_intruder > 0°), prefer a LEFT (positive) turn.\n"
        "- If intruder on RIGHT (bearing_error_to_intruder < 0°), prefer a RIGHT (negative) turn.\n"
        "- If intruder is ~ahead and closing, use a bigger same-side turn.\n"
        "\n"
        "PHASE INTENT:\n"
        "- WAIT_TURN:\n"
        "  * Intention: watch & wait; do not change heading now.\n"
        "  * heading_change_deg MUST be 0; only judge/adjust hold_steps.\n"
        "  * OK: EVEN if the separation or TTCP degrades slightly across the hold window.\n"
        "    keep hold short if risk is high (sep < SAFE_R*1.2 or TTCP ≤ 3).\n"
        "  * NOT OK: non-zero turn, or long hold while TTCP/separation degrades significantly.\n"
        "- EXECUTE_TURN:\n"
        "  * Intention: start one turn now; heading_change_deg must be non-zero (bins − {-30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30} but except for 0).\n"
        "  * Use the sign rule; prefer the smallest bin that improves safety.\n"
        "  * Hold should be small (≈3–6) to maintain the chosen heading briefly.\n"
        "  * OK if separation/TTCP improves during the hold (or min-sep increases).\n"
        "  * NOT OK: zero turn, wrong sign, oversize turn without benefit i.e, separation or TTCP degrades over the hold steps.\n"
      "\n"
      "SCORING HINTS:\n"
      "- Favor decisions that increase separation or TTCP over the hold window.\n"
      "- Penalize phase violations (e.g., non-zero heading in WAIT_TURN, zero turn in EXECUTE_TURN).\n"
      "- Prefer smallest non-zero bin that helps in EXECUTE_TURN; keep hold modest.\n"
      "\n"
      "Return STRICT JSON ONLY with keys exactly as below (omit fields you cannot infer):\n"
      "{\n"
      "  \"ok\": true/false,\n"
      "  \"score\": number between 0 and 1,\n"
      "  \"comment\": \"short reason (mention phase and why)\",\n"
      "  \"delta_action\": {\"heading_change_deg\": <int>, \"hold_steps\": <int>, \"reason\": \"short\"}},\n"
    #   "  \"ppo\": {\"turn_deg\": <int|null>, \"hold_used\": <int|null>, \"rewards\": {\"total\": <number|null>}}\n"
      "}\n"
      "If your recommended action exactly matches the LLM's (same heading and hold), say so in \"comment\".\n"
      "\n"
      "CONTEXT:\n"
    )

# --- Small typed view over your observation vector ---
@dataclass
class ObsSnapshot:
    # Direct x coordinate of ownship.
    own_x: float
    # Direct y coordinate of ownship.
    own_y: float
    # Intruder x coordinate.
    intr_x: float
    # Intruder y coordinate.
    intr_y: float
    # Ownship destination x coordinate.
    dest_x: float
    # Ownship destination y coordinate.
    dest_y: float
    # Current separation distance between aircraft.
    sep_oi: float
    # Distance from ownship to its destination.
    dist_o_dest: float
    # Distance from intruder to its destination.
    dist_i_dest: float
    # Cross-track distance to unresolved path.
    ctd_dist: float
    # Distance to ATCO-approved path.
    dist_to_atco_path: float
    # Current ownship heading in radians.
    own_heading_rad: float
    # --- Optional ATCO-path fields (okay to be None if not supplied) ---
    # Optional heading of ATCO path where projected.
    path_heading_deg: Optional[float] = None
    # Optional difference between path heading and ownship.
    heading_error_to_path_deg: Optional[float] = None
    # Optional signed cross-track offset.
    signed_cross_track_to_path: Optional[float] = None
    
    # Step index for logging context.
    step: Optional[int] = None

    @staticmethod
    def from_vector(vec: np.ndarray, step: Optional[int] = None) -> "ObsSnapshot":
        """
        Maps env.observation_func() vector (current ordering) into fields.
        Index map:
          0..1: own (x,y)
          2..3: intruder (x,y)
          4..5: own destination (x,y)
          6:    separation own↔intruder
          7:    own→dest distance
          8:    intr→dest distance
          9:    cross-track to original path
          10:   distance to ATCO path
          11:   own heading (radians)
        """
        # We map the observation vector entries into readable names.
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
        # Later elements are optional; we fill them if present.
        if len(vec) >= 13: s.path_heading_deg = float(vec[12])
        if len(vec) >= 14: s.heading_error_to_path_deg = float(vec[13])
        if len(vec) >= 15: s.signed_cross_track_to_path = float(vec[14])
        return s

def _rad2deg(r: float) -> float:
    # Convert radians into degrees.
    return float(r) * 180.0 / math.pi

def _wrap_deg(a: float) -> float:
    """Wrap any angle in degrees to (-180, 180]."""
    return ((a + 180.0) % 360.0) - 180.0

def _bearing_deg(dy: float, dx: float) -> float:
    """atan2(dy, dx) in degrees."""
    return math.degrees(math.atan2(dy, dx))

def _side_word(x: float, left="left", right="right", center="centerline") -> str:
    # Return a friendly label for the sign of a value.
    if x > 0: return left
    if x < 0: return right
    return center

def _ahead_or_behind(abs_deg: float) -> str:
    return "ahead" if abs_deg < 90.0 else "abeam/behind"

INDEX_NAME = "_index.jsonl" 

# --- Extensible builder: now includes derived signals; history/BC stats can be added later.
class BasePromptBuilder:
    def __init__(self):
        # We lazily load the behaviour cloning baseline text.
        self._bc_text_cache: Optional[str] = None  # Placeholder for future BC baseline text
        # We keep the latest kinematic numbers for the controller.
        self._last_kinematics: Optional[dict] = None  # <— ADD
    
    def _instructions_text(self) -> str:
        # We return the static instructions used in every prompt.
        return PROMPT_INSTRUCTIONS


    # Public entrypoint (stable API)
    def build_base_prompt(
        self,
        obs_vector: np.ndarray,
        *,
        step: Optional[int] = None,
        prev_obs_vector: Optional[np.ndarray] = None,
        prev_step: Optional[int] = None,
        history_vectors: Optional[Sequence[np.ndarray]] = None,  # <-- NEW
        history_steps:   Optional[Sequence[int]] = None,         # <-- NEW
    ) -> str:
        # We convert the latest observation array into friendly names.
        snap = ObsSnapshot.from_vector(obs_vector, step=step)
        # We do the same for the previous snapshot if provided.
        prev = ObsSnapshot.from_vector(prev_obs_vector, step=prev_step) if prev_obs_vector is not None else None

        # Build history snapshots (oldest → newest). History should NOT include 'snap' itself.
        hist_snaps: List[ObsSnapshot] = []
        if history_vectors is not None:
            if history_steps is not None and len(history_steps) != len(history_vectors):
                raise ValueError("history_steps must match history_vectors length")
            for i, v in enumerate(history_vectors):
                s_i = history_steps[i] if history_steps is not None else None
                # We collect each historic observation as a snapshot object.
                hist_snaps.append(ObsSnapshot.from_vector(v, step=s_i))

        # We assemble all prompt sections and skip any empty ones.
        parts = [
            self._instructions_text(),
            self._current_observation_text(snap),
            self._derived_signals_text(snap, prev),
            self._kinematics_text(snap, prev),
            self._recent_history_text(hist_snaps, current=snap),   # <-- NEW (only prints if history given)
            self._bc_baseline_text(csv_path=FEATUREFILE_PATH),                           
        ]
        return "\n\n".join(p for p in parts if p)

    @property
    def last_kinematics(self) -> Optional[dict]:
        """
        Returns the last computed kinematics as a dictionary.
        This is useful for debugging or further processing.
        """
        # We simply expose the cached numbers.
        return self._last_kinematics
    
    def _kinematics_text(self, s: ObsSnapshot, prev: Optional[ObsSnapshot]) -> str:
        """
        Speeds and TTCP-to-CPA from current & previous snapshots.
        Units: scenario-units per step (1 step = 2 minutes).
        TTCP: time (in steps) to closest approach assuming constant velocity & straight lines.
        """
        if prev is None:
            # Without a previous frame we cannot compute motion numbers.
            self._last_kinematics = {
                "ttcp_steps": None,
                "own_speed": None,
                "intr_speed": None,
                "rel_speed": None,
                "sep": float(s.sep_oi),
            }
            return ("Kinematics (needs a previous observation to compute):\n"
                    "- Ownship speed: N/A\n- Intruder speed: N/A\n"
                    "- Relative speed: N/A\n- TTCP (to CPA): N/A")

        # Δt in steps (fallback to 1 if step indices missing or equal)
        dt_steps = 1
        if s.step is not None and prev.step is not None and s.step != prev.step:
            # We use the actual difference in step numbers when provided.
            dt_steps = max(1, s.step - prev.step)

        # Position deltas
        dx_o = s.own_x  - prev.own_x
        dy_o = s.own_y  - prev.own_y
        dx_i = s.intr_x - prev.intr_x
        dy_i = s.intr_y - prev.intr_y

        # Speeds (scenario-units per step)
        # We compute simple straight-line speeds for each aircraft.
        v_o = math.hypot(dx_o, dy_o) / dt_steps
        v_i = math.hypot(dx_i, dy_i) / dt_steps

        # Relative position r (intruder relative to ownship) at current step
        rx = s.intr_x - s.own_x
        ry = s.intr_y - s.own_y

        # Relative velocity v_rel = v_i - v_o
        vrx = (dx_i - dx_o) / dt_steps
        vry = (dy_i - dy_o) / dt_steps
        vrel_mag2 = vrx*vrx + vry*vry

        # TTCP-to-CPA (in steps): t* = - (r · v_rel) / |v_rel|^2, clamped to >= 0
        if vrel_mag2 > 0:
            # We project the relative motion to find the closest approach time.
            t_star = - (rx*vrx + ry*vry) / vrel_mag2
            ttcp_steps = max(0.0, t_star)
            ttcp_minutes = 2.0 * ttcp_steps  # 1 step = 2 minutes
            ttcp_text = f"{ttcp_steps:.2f} steps (~{ttcp_minutes:.1f} min)"
        else:
            # With no relative motion we cannot define TTCP.
            ttcp_steps = None
            rel_speed = None
            ttcp_text = "N/A (relative speed ~ 0)"

        rel_speed = math.sqrt(vrel_mag2)
        # ---- cache numeric results for the controller ----
        # We save the numbers so the controller can reuse them later.
        self._last_kinematics = {
            "ttcp_steps": ttcp_steps,
            "own_speed": float(v_o),
            "intr_speed": float(v_i),
            "rel_speed": float(rel_speed) if rel_speed is not None else None,
            "sep": float(s.sep_oi),
        }
        lines = []
        # We describe each value in plain text for the prompt.
        lines.append("Kinematics (scenario-units per step; 1 step = 2 min):")
        lines.append(f"- Ownship speed: {v_o:.3f}")
        lines.append(f"- Intruder speed: {v_i:.3f}")
        lines.append(f"- Relative speed |v_rel|: {rel_speed:.3f}")
        lines.append(f"- TTCP (to CPA): {ttcp_text}")
        return "\n".join(lines)

    def _current_observation_text(self, s: ObsSnapshot) -> str:
        lines = []
        # We open with a short heading to orient the reader.
        lines.append("Current situation (all numbers are in scenario units):")
        if s.step is not None:
            # We include the step index when provided.
            lines.append(f"- Step index: {s.step}")
        # We list all positions and distances the LLM might need.
        lines.append(f"- Ownship position: x={s.own_x:.1f}, y={s.own_y:.1f}")
        lines.append(f"- Intruder position: x={s.intr_x:.1f}, y={s.intr_y:.1f}")
        lines.append(f"- Ownship destination: x={s.dest_x:.1f}, y={s.dest_y:.1f}")
        lines.append(f"- Separation (ownship ↔ intruder): {s.sep_oi:.2f}")
        lines.append(f"- Distances: ownship→dest={s.dist_o_dest:.2f}, intruder→dest={s.dist_i_dest:.2f}")
        lines.append(f"- Deviation: cross-track-to-original={s.ctd_dist:.2f}, distance-to-ATCO-path={s.dist_to_atco_path:.2f}")
        lines.append(f"- Ownship heading: {_rad2deg(s.own_heading_rad):.1f}° (0°=east, positive is counter-clockwise)")
        # --- NEW: make ATCO path signals explicit in the prompt ---
        if s.path_heading_deg is not None:
            # We only mention the path heading if it exists.
            lines.append(f"- ATCO path heading: {s.path_heading_deg:+.1f}°")

        if s.heading_error_to_path_deg is not None:
            # We add the signed heading error and a simple left/right label.
            lines.append(
                f"- heading_error_to_path_deg: {s.heading_error_to_path_deg:+.1f}° "
                f"(path is to your {_side_word(s.heading_error_to_path_deg)})"
            )

        if s.signed_cross_track_to_path is not None:
            # We show how far we are off the path with a sign.
            lines.append(
                f"- signed_cross_track_to_path: {s.signed_cross_track_to_path:+.2f} "
                f"(left of path is positive)"
            )

        # Adding sanity check
        if s.path_heading_deg is not None:
            # We double-check that the error value matches the two headings.
            own_h = _rad2deg(s.own_heading_rad)
            err_path_from_heading = _wrap_deg(float(s.path_heading_deg) - own_h)
        if s.heading_error_to_path_deg is not None:
            diff = abs(_wrap_deg(err_path_from_heading - float(s.heading_error_to_path_deg)))
            lines.append(f"- provided_heading_error_to_path: {float(s.heading_error_to_path_deg):+.1f}° (Δ={diff:.1f}°)")

        return "\n".join(lines)

    def _derived_signals_text(self, s: ObsSnapshot, prev: Optional[ObsSnapshot]) -> str:
        """
        Three simple, informative signals:
          1) heading_error_to_dest_deg
          2) bearing_error_to_intr_deg
          3) sep_trend_per_step (requires previous obs; otherwise N/A)
        """
        # We reuse the ownship heading in degrees for easy comparisons.
        own_heading_deg = _rad2deg(s.own_heading_rad)

        # 1) Heading error to destination
        # We find the direction to the destination and compare it to the nose.
        heading_to_dest_deg = _bearing_deg(s.dest_y - s.own_y, s.dest_x - s.own_x)
        heading_error_to_dest_deg = _wrap_deg(heading_to_dest_deg - own_heading_deg)

        # 2) Bearing error to intruder
        # We do the same for the intruder to learn which side it sits on.
        own_to_intr_angle_deg = _bearing_deg(s.intr_y - s.own_y, s.intr_x - s.own_x)
        bearing_error_to_intr_deg = _wrap_deg(own_to_intr_angle_deg - own_heading_deg)
        intr_pos_rel = _ahead_or_behind(abs(bearing_error_to_intr_deg))

        # 3) Separation trend per step (Δsep)
        if prev is not None:
            # We subtract the previous separation to know if we are closing or opening.
            sep_trend = s.sep_oi - prev.sep_oi  # negative = closing, positive = opening
            sep_trend_text = f"{sep_trend:+.3f} per step ({'closing' if sep_trend < 0 else 'opening' if sep_trend > 0 else 'steady'})"
        else:
            sep_trend_text = "N/A (no previous observation)"

        lines = []
        # We bundle all three human readable sentences.
        lines.append("Derived signals (simple math, for decision support):")
        lines.append(f"- Heading error to destination: {heading_error_to_dest_deg:+.1f}° "
                     f"(destination is to your {_side_word(heading_error_to_dest_deg)})")
        lines.append(f"- Bearing error to intruder:   {bearing_error_to_intr_deg:+.1f}° "
                     f"(intruder is to your {_side_word(bearing_error_to_intr_deg)}; {intr_pos_rel})")
        lines.append(f"- Separation trend (Δsep):      {sep_trend_text}")
        return "\n".join(lines)

    # ---- Stubs to fill later ----
    # NOTE: make sure you have: from typing import List

    def _recent_history_text(self, hist_snaps: List[ObsSnapshot], *, current: ObsSnapshot) -> str:
        """
        Simple-language trend summary over a fixed history window of size K=5 (oldest → current).
        """
        K = 5
        if len(hist_snaps) < K:
            # Not enough past points to describe a trend.
            return ""

        window = hist_snaps[-K:]
        oldest = window[0]
        newest = window[-1]

        def heading_err_to_dest_deg(s: ObsSnapshot) -> float:
            own_h_deg = _rad2deg(s.own_heading_rad)
            head_to_dest_deg = _bearing_deg(s.dest_y - s.own_y, s.dest_x - s.own_x)
            return _wrap_deg(head_to_dest_deg - own_h_deg)

        def rel_bearing_to_intr_deg(s: ObsSnapshot) -> float:
            own_h_deg = _rad2deg(s.own_heading_rad)
            b = _bearing_deg(s.intr_y - s.own_y, s.intr_x - s.own_x)
            return _wrap_deg(b - own_h_deg)

        # Helpers to use friendlier words
        def _closer_or_farther(x: float) -> str:
            return "getting closer" if x < 0 else "moving farther apart" if x > 0 else "no change"

        def _path_side(deg: float) -> str:
            if deg > 0: return "left of the direct path"
            if deg < 0: return "right of the direct path"
            return "exactly on the direct path"

        def _nose_quadrant(deg: float) -> str:
            a = abs(deg)
            side = "left" if deg > 0 else "right" if deg < 0 else ""
            if a < 10:
                return "dead-ahead"
            elif a < 90:
                return f"{side}-front" if side else "ahead"
            else:
                return f"{side}-side/behind" if side else "side/behind"

        # ---- 1) Separation ----
        sep_start = float(oldest.sep_oi)
        sep_prev  = float(newest.sep_oi)
        sep_now   = float(current.sep_oi)

        sep_net   = sep_now - sep_start
        sep_last  = sep_now - sep_prev

        # ---- 2) Heading alignment to destination ----
        he_start = heading_err_to_dest_deg(oldest)
        he_prev  = heading_err_to_dest_deg(newest)
        he_now   = heading_err_to_dest_deg(current)

        he_net   = he_now - he_start
        he_last  = he_now - he_prev

        align_trend = "better aligned than at the start" if abs(he_now) < abs(he_start) \
                    else "worse aligned than at the start" if abs(he_now) > abs(he_start) \
                    else "same alignment as at the start"

        # ---- 3) Intruder relative bearing (vs nose) ----
        r_start = rel_bearing_to_intr_deg(oldest)
        r_prev  = rel_bearing_to_intr_deg(newest)
        r_now   = rel_bearing_to_intr_deg(current)

        rmag_net  = abs(r_now) - abs(r_start)   # <0 toward nose, >0 away
        rmag_last = abs(r_now) - abs(r_prev)

        t_old = f"t={oldest.step}" if oldest.step is not None else "oldest"
        t_cur = f"t={current.step}" if current.step is not None else "current"

        lines = []
        lines.append(f"Last {K} steps ({t_old} → {t_cur}):")

        # Separation — plain language first, numbers in parentheses
        lines.append(
            f"- Distance between aircraft: {sep_start:.2f} → {sep_now:.2f} "
            f"({sep_net:+.2f}; {_closer_or_farther(sep_net)}). "
            f"Last step: {sep_last:+.2f} ({_closer_or_farther(sep_last)})."
        )

        # Destination alignment — where you are relative to the direct path
        lines.append(
            f"- Facing the destination: now {abs(he_now):.1f}° {_path_side(he_now)} "
            f"(was {abs(he_start):.1f}° {_path_side(he_start)}). "
            f"Trend: {align_trend}. Last step change: {he_last:+.1f}°."
        )

        # Intruder location relative to nose — quadrant words + nose-angle trend
        lines.append(
            f"- Intruder position vs your nose: {r_start:+.1f}° → {r_now:+.1f}° "
            f"({ _nose_quadrant(r_start) } → { _nose_quadrant(r_now) }). "
            f"Angle from your nose changed by {rmag_net:+.1f}° "
            f"({'moving away from nose' if rmag_net > 0 else 'moving toward nose' if rmag_net < 0 else 'no change'}); "
            f"last step: {rmag_last:+.1f}° "
            f"({'toward nose' if rmag_last < 0 else 'away from nose' if rmag_last > 0 else 'no change'})."
        )

        return "\n".join(lines)

    def _bc_baseline_text(
        self,
        df: Optional[pd.DataFrame] = None,
        csv_path: Optional[str] = None,
        *,
        min_bucket_count: int = 5  # kept for API compatibility; unused now
    ) -> str:
        """
        ATCO baseline (TWO-PHASE: WAIT_TURN & EXECUTE_TURN).

        Summaries included (1 step = 2 minutes):
        1) Distribution of turn magnitudes (0–30° by 5°) + small/large share
        2) Cross-track deviation percentiles (p10,p25,p50,p75,p90)
        3) When first turn typically starts (steps) + coarse histogram
        """
        if self._bc_text_cache is not None:
            # We reuse the cached text so we only compute once.
            return self._bc_text_cache

        # ---- load dataframe ----
        if df is None:
            if csv_path is None:
                return ""  # nothing to compute yet
            try:
                # We load the CSV containing baseline stats.
                df = pd.read_csv(csv_path)
            except Exception as e:
                return f"(BC baseline unavailable: failed to load CSV: {e})"

        lines: List[str] = []
        # We open the section with a simple heading.
        lines.append("ATCO baseline (two-phase mode):")

        # -------------------- helpers --------------------
        STEP_MINUTES = 2.0  # 2 minutes per environment step

        def _wrap_signed_deg(d):
            # Wrap to (-180, 180]
            return (d + 180.0) % 360.0 - 180.0

        def _to_time(s):
            # Parse "HH:MM:SS" (returns NaT if bad)
            return pd.to_datetime(s, format="%H:%M:%S", errors="coerce")

        def _diff_steps(start_s, event_s):
            # Steps from start to event (handles cross-midnight). Integer steps by rounding.
            start_t = _to_time(start_s)
            event_t = _to_time(event_s)
            if pd.isna(start_t) or pd.isna(event_t):
                return np.nan
            delta = (event_t - start_t).total_seconds()
            if delta < 0:
                delta += 24 * 3600
            return int(round(delta / (STEP_MINUTES * 60.0)))

        def _hist_str(series_steps: pd.Series) -> str:
            # Coarse, decision-friendly bins
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

        # -------------------- 1) Turn magnitude distribution --------------------
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
                mag_bins = d_signed.abs().apply(lambda x: min(30, int(round(x / 5.0) * 5)))  # 0..30 step 5
                turn_bins_series = mag_bins.value_counts(normalize=True).sort_index()
        elif "headingAngle" in df.columns:
            s = pd.to_numeric(df["headingAngle"], errors="coerce").dropna().astype(float).abs()
            if len(s):
                mag_bins = s.apply(lambda x: min(30, int(round(x / 5.0) * 5)))
                turn_bins_series = mag_bins.value_counts(normalize=True).sort_index()

        if turn_bins_series is not None and len(turn_bins_series):
            all_bins = [0, 5, 10, 15, 20, 25, 30]
            # We build a friendly string for each possible turn size.
            dist_str = ", ".join(f"{b}°:{turn_bins_series.get(b, 0.0)*100:.0f}%" for b in all_bins)
            # We compute the share of small and large turns for human rules of thumb.
            small_share = sum(turn_bins_series.get(b, 0.0) for b in [0, 5, 10]) * 100.0
            large_share = sum(turn_bins_series.get(b, 0.0) for b in [20, 25, 30]) * 100.0
            # We call out the two most popular turn sizes.
            top2 = turn_bins_series.sort_values(ascending=False).head(2)
            top_str = ", ".join(f"{int(k)}° ({v*100:.0f}%)" for k, v in top2.items())
            lines.append(
                "- Turn magnitudes (how big the first turn usually is): "
                f"{dist_str}. Most common bins → {top_str}. "
                f"Rule-of-thumb: small turns (≤10°) ≈ {small_share:.0f}% of cases; "
                f"large turns (≥20°) ≈ {large_share:.0f}%."
            )
        else:
            lines.append("- Turn magnitudes: not enough data to estimate typical bins.")

        # -------------------- 2) Cross-track deviation percentiles --------------------
        if "crosstrack_dist_max" in df.columns:
            s = pd.to_numeric(df["crosstrack_dist_max"], errors="coerce").dropna().astype(float)
            if len(s):
                q = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
                lines.append(
                    "- Off-route distance after turning (max cross-track): "
                    f"p10≈{q.loc[0.10]:.2f}, p25≈{q.loc[0.25]:.2f}, p50≈{q.loc[0.50]:.2f}, "
                    f"p75≈{q.loc[0.75]:.2f}, p90≈{q.loc[0.90]:.2f}. "
                    "Interpretation: staying near the median is “typical”; exceeding p90 is “unusual”."
                )
            else:
                lines.append("- Off-route distance: no valid values.")
        else:
            lines.append("- Off-route distance: column 'crosstrack_dist_max' not found.")

        # -------------------- 3) When first turn starts (steps) --------------------
        start_time = None
        if "flightstarttime_original_res" in df.columns:
            start_time = df["flightstarttime_original_res"].copy()
            if "flightstarttime_original_unres" in df.columns:
                # We fill missing resolved times with unresolved start times.
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

        # -------------------- Decision hints (baseline-only) --------------------
        lines.append(
            "Baseline decision hints: start turns inside the typical window; "
            "prefer smaller bins when geometry is mild; avoid cross-track values beyond p90 unless necessary."
        )

        text = "\n".join(lines)
        self._bc_text_cache = text
        return text


# === Phase 1.2: Attach a PHASE STATE block to the built prompt ===
PHASE_MARKER = "\n=========================\n"
ALLOWED_BINS: List[int] = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]

def attach_phase_state_block(
    prompt: str,
    *,
    current_phase: str,      # "WAIT_TURN" | "EXECUTE_TURN"
    step_index: int,
    allowed_bins: Optional[list] = None,
    extra_notes: Optional[str] = None
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
    return prompt[:idx] + block + prompt[idx:]


# optional but handy):
# OLLAMA_MODEL = "gemma3:12b"
OLLAMA_MODEL = "llama3:8b"

ALLOWED_BINS: List[int] = [-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30]

# NEW: local invoke via Ollama (returns model text; we’ll parse JSON downstream)
def ollama_invoke(
    prompt_text: str,
    *,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 8192
) -> str:
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        options={
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
        },
    )
    # Only return assistant content (no <think> saved anywhere)
    return (resp.get("message") or {}).get("content", "") or ""

EXTRA_HEADERS = {
    "HTTP-Referer": "http://localhost",  # replace with your app/site if you have one
    "X-Title": "ATCO-sim"
}

# --- Helpers: files, JSON extraction + validation ---
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

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# CHANGED
def start_llm_log_dir(base_dir: str = MEMORY_DIR):
    os.makedirs(base_dir, exist_ok=True)
    # run_id is the ID (no leading 'run_'); run_dir has 'run_' prefix
    stamp  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"run_{stamp}"
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id

def save_eval_notes(new_snippets: list, path: Optional[str] = None):
    path = path or EVAL_MEMORY_PATH
    mem = {"notes": ""}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                mem = json.load(f)
        except:
            pass
    cur = set([s.strip() for s in mem.get("notes","").split("\n") if s.strip()])
    for s in new_snippets:
        if isinstance(s, str) and s.strip():
            cur.add(s.strip())
    mem["notes"] = "\n".join(sorted(cur))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2)

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")

def _save_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _eq_int(a, b) -> bool:
    """True if both not-None and equal as ints (after safe coercion)."""
    try:
        return int(round(float(a))) == int(round(float(b)))
    except Exception:
        return False

def load_eval_notes(path: Optional[str] = None):
    path = path or EVAL_MEMORY_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("notes", "")
    except:
        return ""
    
# NEW: snapshot builder used for observation_before / observation_after
def make_obs_snapshot(cur_vec: np.ndarray, prev_vec: Optional[np.ndarray], step: int) -> dict:
    cur = ObsSnapshot.from_vector(np.array(cur_vec, dtype=float), step=step)
    prev = ObsSnapshot.from_vector(np.array(prev_vec, dtype=float), step=step-1) if prev_vec is not None else None

    # TTCP math (same as BasePromptBuilder._kinematics_text)
    if prev is None:
        ttcp = None
    else:
        dt = 1 if (cur.step is None or prev.step is None or cur.step == prev.step) else max(1, cur.step - prev.step)
        dx_o, dy_o = (cur.own_x - prev.own_x), (cur.own_y - prev.own_y)
        dx_i, dy_i = (cur.intr_x - prev.intr_x), (cur.intr_y - prev.intr_y)
        rx, ry = (cur.intr_x - cur.own_x), (cur.intr_y - cur.own_y)
        vrx, vry = (dx_i - dx_o)/dt, (dy_i - dy_o)/dt
        vrel2 = vrx*vrx + vry*vry
        ttcp = max(0.0, - (rx*vrx + ry*vry) / vrel2) if vrel2 > 0 else None

    snapshot = {
        "step": int(step),
        "sep_oi": float(cur.sep_oi),
        "dist_o_dest": float(cur.dist_o_dest),
        "dist_i_dest": float(cur.dist_i_dest),
        "ttcp": None if ttcp is None else float(ttcp),
        "own_heading_deg": float(_rad2deg(cur.own_heading_rad)),
        "path_heading_deg": None if cur.path_heading_deg is None else float(cur.path_heading_deg),
        "heading_err_path_deg": None if cur.heading_error_to_path_deg is None else float(cur.heading_error_to_path_deg),
    }
    return snapshot

def update_memory_record(path: str, hold_effect: dict, obs_after: dict, index_path: Optional[str] = None):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)

    rec["hold_effect"] = hold_effect
    rec["observation_after"] = obs_after

    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    # (Plan 7) append one-line index for fast filtering
    if index_path:
        # Ensure the index folder exists before appending.
        os.path.dirname(index_path) and os.makedirs(os.path.dirname(index_path), exist_ok=True)
        before = rec.get("observation_before") or {}
        meta   = rec.get("meta") or {}
        ans    = rec.get("answer") or {}
        man    = (ans.get("maneuver") or {})
        # Pull branch-comparison numbers if present
        br     = rec.get("branch_compare") or {}
        man    = (rec.get("answer") or {}).get("maneuver") or {}
        def _num(x):
            try:
                return None if x is None else float(x)
            except Exception:
                return None
                br     = rec.get("branch_compare") or {}
        man    = (rec.get("answer") or {}).get("maneuver") or {}

        # DEBUG: show what we’re about to mirror so we can spot 'None'
        print(f"[INDEX] tag={meta.get('tag')} br_keys={list(br.keys())} "
              f"llm_return={br.get('llm_return')} ppo_turn_deg={br.get('ppo_turn_deg')} ppo_return={br.get('ppo_return')}")

        idx_obj = {
            "run_id": meta.get("run_id"),
            "tag": meta.get("tag"),
            "phase": meta.get("phase"),
            "step": meta.get("step"),

            "ttcp_before": before.get("ttcp"),
            "sep_before":  before.get("sep_oi"),
            "dist_before": before.get("dist_o_dest"),
            "head_err_before": before.get("heading_err_path_deg"),

            "ttcp_end":    obs_after.get("ttcp"),
            "sep_min":     hold_effect.get("min_sep_during_hold"),
            "sep_end":     obs_after.get("sep_oi"),
            "dist_end":    obs_after.get("dist_o_dest"),

            # LLM suggestion (from decision)
            "turn_deg":         man.get("heading_change_deg"),
            "hold_steps":       man.get("hold_steps"),
            "llm_turn_deg":     man.get("heading_change_deg"),
            "llm_hold_steps":   man.get("hold_steps"),

            # Ghost A/B compare (from branch_compare)
            "llm_return":       _num(br.get("llm_return")),
            "ppo_turn_deg":     br.get("ppo_turn_deg"),
            "ppo_hold_steps":   br.get("return_horizon") or man.get("hold_steps"),  # horizon N used for both branches
            "ppo_return":       _num(br.get("ppo_return")),
            "return_horizon":   br.get("return_horizon") or man.get("hold_steps"),

            "tech_summary_snip": ans.get("technical_summary", ""),
            "rationale_snip":    ans.get("rationale", ""),

            "mem_path": path,

            # Evaluator mirrors (filled by apply_eval_to_record)
            "eval_ok": None,
            "eval_score": None,
            "eval_heading": None,
            "eval_hold": None,
            # "eval_reward": None
        }
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(idx_obj) + "\n")
            print(f"[INDEX] appended line to {index_path}")


def _rewrite_index_line(index_path: str, tag: str, patch: dict):
    """Update the single line with matching tag in _index.jsonl by merging 'patch'."""
    if not os.path.exists(index_path):
        return
    rows = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
            except:
                continue
            if j.get("tag") == tag:
                j.update({k: v for k, v in patch.items()})
            rows.append(j)
    with open(index_path, "w", encoding="utf-8") as f:
        for j in rows:
            f.write(json.dumps(j) + "\n")

def enforce_index_cap(index_path: str, cap: int = 100):
    """
    Keep at most 'cap' items in _index.jsonl by recency only:
    - Allow duplicate tags.
    - If over cap, drop the OLDEST lines and keep the NEWEST 'cap' lines.
    - Optionally deletes per-phase JSON files of pruned rows (via 'mem_path').

    Assumes rows are appended in chronological order (oldest at top).
    """
    if not os.path.exists(index_path):
        return

    # Read all valid JSONL rows in order
    rows = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                # Skip broken lines
                pass

    # Nothing to do if within the cap
    n = len(rows)
    if n <= cap:
        return

    # Oldest rows to prune, newest to keep
    over = n - cap
    pruned = rows[:over]
    kept   = rows[over:]

    # Rewrite file with just the newest 'cap' rows
    with open(index_path, "w", encoding="utf-8") as f:
        for j in kept:
            f.write(json.dumps(j) + "\n")

    # OPTIONAL: remove heavy per-phase JSON files we just pruned
    for j in pruned:
        mp = j.get("mem_path")
        if mp and isinstance(mp, str) and os.path.exists(mp):
            try:
                os.remove(mp)
            except:
                pass

def _save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# REPLACE the whole _validate_and_normalize with this version
def _validate_and_normalize(payload: Dict[str, Any], expected_phase: Optional[str] = None) -> Dict[str, Any]:
    """
    Normalizes and prints messages whenever we correct/replace anything.
    Also used by the saver to persist both raw and normalized outputs.
    """
    corrections: List[str] = []

    if "answer" not in payload or not isinstance(payload["answer"], dict):
        raise ValueError("Missing 'answer' object.")
    ans = payload["answer"]

    # scenario_summary
    if "scenario_summary" not in ans or not isinstance(ans["scenario_summary"], str):
        corrections.append("scenario_summary missing → set to empty string.")
        ans["scenario_summary"] = ""

    # NEW: technical_summary + rationale (both strings)
    if "technical_summary" not in ans or not isinstance(ans["technical_summary"], str):
        corrections.append("technical_summary missing → set to empty string.")
        ans["technical_summary"] = ""
    if "rationale" not in ans or not isinstance(ans["rationale"], str):
        corrections.append("rationale missing → set to empty string.")
        ans["rationale"] = ""

    # metrics (ttcp, separation_distance) → float or null
    ans.setdefault("metrics", {})
    metrics = ans["metrics"]
    for k in ("ttcp", "separation_distance"):
        if k in metrics and metrics[k] is not None:
            try:
                metrics[k] = float(metrics[k])
            except Exception:
                corrections.append(f"metrics.{k} not numeric → set to null.")
                metrics[k] = None
        else:
            metrics[k] = None

    # maneuver
    ans.setdefault("maneuver", {})
    man = ans["maneuver"]

    # Phase: only two allowed
    allowed_phases = {"WAIT_TURN", "EXECUTE_TURN"}
    raw_phase = man.get("phase", "")
    phase = expected_phase if expected_phase in allowed_phases else (raw_phase if raw_phase in allowed_phases else "WAIT_TURN")
    man["phase"] = phase

    # Heading
    raw_h = man.get("heading_change_deg", 0)
    try:
        h = float(raw_h)
    except Exception:
        corrections.append(f"heading_change_deg not numeric ('{raw_h}') → set to 0.")
        h = 0.0
    snapped = _snap_to_allowed_bin(h)
    if int(round(h)) != snapped:
        corrections.append(f"heading_change_deg {h:+.1f}° snapped to nearest bin {snapped:+d}°.")

    if phase == "WAIT_TURN":
        if snapped != 0:
            corrections.append("WAIT_TURN forces heading_change_deg to 0°.")
        snapped = 0
    elif phase == "EXECUTE_TURN":
        if snapped == 0:
            corrections.append("EXECUTE_TURN must not be 0° → set to +5°.")
            snapped = 5
    man["heading_change_deg"] = snapped

    # hold_steps (>=0 integer)
    raw_hold = man.get("hold_steps", 0)
    hold = _coerce_int(raw_hold, default=0)
    if isinstance(raw_hold, (str, float)) and hold != raw_hold:
        corrections.append(f"hold_steps coerced to integer ({raw_hold} → {hold}).")
    if hold < 0:
        corrections.append(f"hold_steps negative ({hold}) → set to 0.")
        hold = 0
    man["hold_steps"] = hold

    if corrections:
        print("LLM output normalized:")
        for c in corrections:
            print("  •", c)

    # Return full answer including the two new text fields
    return {
        "answer": {
            "scenario_summary": ans["scenario_summary"],
            "technical_summary": ans["technical_summary"],
            "rationale": ans["rationale"],
            "metrics": metrics,
            "maneuver": man,
        }
    }

def find_similar_cases(obs_before: dict, phase: str, index_path: str, k: int = 3) -> List[dict]:
    """
    Relaxed, feature-flexible matcher:
    - Same phase only
    - Score = average of available deltas over these features:
        sep_before, ttcp_before, dist_before, head_err_before (deg wrap)
    - Missing fields are simply skipped (no big penalty).
    """
    def _ang_diff_deg(a: float, b: float) -> float:
        # smallest absolute difference on a circle (-180, 180]
        return abs(((a - b + 180.0) % 360.0) - 180.0)

    if not os.path.exists(index_path):
        return []

    # Current features we’ll try to match on
    cur_sep   = obs_before.get("sep_oi")
    cur_ttcp  = obs_before.get("ttcp")
    cur_dist  = obs_before.get("dist_o_dest")
    cur_herr  = obs_before.get("heading_err_path_deg")

    rows = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
            except:
                continue
            if j.get("phase") != phase:
                continue

            score_sum = 0.0
            used = 0

            # 1) separation
            if (cur_sep is not None) and (j.get("sep_before") is not None):
                try:
                    score_sum += abs(float(cur_sep) - float(j["sep_before"]))
                    used += 1
                except:
                    pass

            # 2) TTCP
            if (cur_ttcp is not None) and (j.get("ttcp_before") is not None):
                try:
                    score_sum += abs(float(cur_ttcp) - float(j["ttcp_before"]))
                    used += 1
                except:
                    pass

            # 3) distance to own destination
            if (cur_dist is not None) and (j.get("dist_before") is not None):
                try:
                    score_sum += abs(float(cur_dist) - float(j["dist_before"]))
                    used += 1
                except:
                    pass

            # 4) heading error to ATCO path (angles)
            if (cur_herr is not None) and (j.get("head_err_before") is not None):
                try:
                    score_sum += _ang_diff_deg(float(cur_herr), float(j["head_err_before"]))
                    used += 1
                except:
                    pass

            if used == 0:
                # nothing to compare reliably → skip this row
                continue

            j["_score"] = score_sum / used  # average to be fair
            rows.append(j)

    rows.sort(key=lambda r: r.get("_score", 9e9))
    return rows[:k]

def _format_past_cases_block(matches: List[dict]) -> str:
    if not matches:
        return ""
    # Small helpers to format numbers or degrees safely.
    def _fmt_num(x, nd=1):
        return "n/a" if x is None else f"{float(x):.{nd}f}"
    def _fmt_deg(x, nd=1):
        return "n/a" if x is None else f"{float(x):.{nd}f}°"

    lines = []
    # Heading line to separate this block inside the prompt.
    lines.append("PAST CASES (closest matches):")
    for m in matches:
        # We grab the stored tag to identify the memory file.
        tag  = m.get("tag", "n/a")
        # We show what turn the LLM suggested in that memory.
        turn = m.get("turn_deg", "n/a")
        # We also show how long the hold lasted.
        hold = m.get("hold_steps", "n/a")

        # BEFORE features (the ones we match on)
        sep0  = m.get("sep_before")
        tt0   = m.get("ttcp_before")
        dist0 = m.get("dist_before")
        hep0  = m.get("head_err_path_before") or m.get("heading_err_path_before")


        # one compact line with the four BEFORE features
        lines.append(
            f"- {tag}: sep≈{_fmt_num(sep0)}, ttcp≈{_fmt_num(tt0)}, "
            f"dist≈{_fmt_num(dist0)}, head_err_path≈{_fmt_deg(hep0)}; "
            f"decision: turn {turn}°, hold {hold}"
        )
       
        # optional short text
        ts = m.get("tech_summary_snip", "")
        if ts:
            # We include the technical summary if it was stored.
            lines.append(f"  summary: {ts}")
        ra = m.get("rationale_snip", "")
        if ra:
            # We include the free form rationale too.
            lines.append(f"  rationale: {ra}")

    return "\n".join(lines)


# CHANGED: now writes full memory record and returns its path
def call_llm_for_maneuver(
    prompt_text: str,
    save_dir: str,
    tag: str,                              # e.g., "t012_wait_turn"
    override_metrics: Optional[Dict[str, Any]] = None,
    expected_phase: Optional[str] = None,
    *,
    obs_before: Optional[dict] = None,     # NEW
    run_id: Optional[str] = None,          # NEW
    step: Optional[int] = None,            # NEW
    fileinfo: Optional[Any] = None,        # NEW
    append_after_path: Optional[str] = None # NEW (kept for API compatibility)
) -> dict:
    # We make sure the memory folder exists before writing to it.
    _ensure_dir(save_dir)

    # File naming for memory (Plan 1): one JSON per decision
    # memory/run_<ts>/t{step}_{phase}.json
    fname_json   = f"{tag}.json"
    fname_prompt = f"{tag}.prompt.txt"
    mem_path     = os.path.join(save_dir, fname_json)
    prompt_path  = os.path.join(save_dir, fname_prompt)

    # Save exact prompt (+ sha1 ref)
    _save_text(prompt_path, prompt_text)
    prompt_sha1 = _sha1_text(prompt_text)

    # LLM call
    # resp     = llm.invoke(prompt_text)
    # raw_text = getattr(resp, "content", str(resp))
    # raw_text = groq_invoke(prompt_text)
    raw_text = ollama_invoke(prompt_text)
    try:
        # We try to parse the answer as strict JSON.
        parsed = _extract_json(raw_text)
    except Exception:
        # If parsing fails we fall back to a safe default answer.
        parsed = {
            "answer": {
                "scenario_summary": "",
                "metrics": {"ttcp": None, "separation_distance": None},
                "maneuver": {"phase": expected_phase or "WAIT_TURN", "heading_change_deg": 0, "hold_steps": 0}
            }
        }
    # We normalise everything so the shape is consistent.
    normalized = _validate_and_normalize(parsed, expected_phase=expected_phase)

    # override metrics (Plan 2)
    if override_metrics:
        # We optionally overwrite TTCP and separation if caller provided numbers.
        m = normalized["answer"].setdefault("metrics", {})
        for k in ("ttcp", "separation_distance"):
            if k in override_metrics:
                v = override_metrics[k]
                m[k] = None if v is None else float(v)

    # Build memory record (Plan 2 schema)
    rec = {
        "meta": {
            "run_id": run_id,
            "tag": tag,                                 # "t{step}_{phase}"
            "phase": normalized["answer"]["maneuver"]["phase"],
            "step": int(step) if step is not None else None,
            "fileinfo": fileinfo,
        },
        "prompt_ref": {
            "path": prompt_path,
            "sha1": prompt_sha1,
        },
        "observation_before": obs_before,               # Plan 4
        "answer": normalized["answer"],                 # current normalized JSON
        # "hold_effect":  ... (added later)
        # "observation_after": ... (added later)
    }

    # We persist the early memory record so later steps can append extra fields.
    _save_json(mem_path, rec)
    # We return the normalized answer plus the memory path for callers.
    return {"normalized": normalized, "mem_path": mem_path, "tag": tag}

def build_eval_context(rec: dict, hold_effect: dict, obs_after: dict) -> str:
    """
    Minimal, self-contained JSON context for the evaluator.
    """
    ans = rec.get("answer", {})
    man = ans.get("maneuver", {})
    # We gather the most important pieces in a clean JSON bundle.
    ctx = {
        "meta": rec.get("meta"),
        "maneuver": {
            "phase": man.get("phase"),
            "heading_change_deg": man.get("heading_change_deg"),
            "hold_steps": man.get("hold_steps")
        },
        "observation_before": rec.get("observation_before"),
        "hold_effect": hold_effect,
        "observation_after": obs_after,
        "notes": {
            "ttcp_rule": "larger is safer",
            "sep_rule":  "larger is safer"
        }
    }
    # We return the pretty printed JSON string to feed into the evaluator prompt.
    return json.dumps(ctx, indent=2)

def call_llm_evaluator(context_text: str, save_dir: str, tag: str):
    """
    Evaluates the last advice with PHASE-AWARE rules and (optionally) proposes an absolute action.
    Expected JSON:
    {
      "ok": <bool>, "score": 0..1,
      "comment": "<short>",
      "delta_action": {"heading_change_deg": <int>, "hold_steps": <int>, "reason": "<why>"},
      "ppo": {"turn_deg": <int|null>, "hold_used": <int|null>, "rewards": {"total": <float|null>}}
    }
    """
    if EVALUATOR_PROMPT_TEMPLATE:
        # We wrap the given context with the fixed instructions template.
        prompt = EVALUATOR_PROMPT_TEMPLATE.format(context=context_text)
    else:
        prompt = context_text

    # txt = groq_invoke(prompt)
    txt = ollama_invoke(prompt)
    # We try to capture the first JSON object in the reply.
    m = JSON_OBJECT_RE.search(txt.strip())
    obj = {"ok": False, "score": 0.0, "comment": "", "delta_action": None, "ppo": None}
    if m:
        try:
            obj = json.loads(m.group(0))
        except:
            pass

    outp = os.path.join(save_dir, f"{_slug(tag)}.eval.json")
    _save_json(outp, obj)
    # We return the parsed evaluation dictionary for further handling.
    return obj


def apply_eval_to_record(mem_path: str, eval_obj: dict, index_path: str):
    """
    Add an 'eval' block + optional 'delta_action' to the phase JSON.
    Also mirror key fields into _index.jsonl and enforce cap afterwards.
    """
    try:
        with open(mem_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
    except:
        return

    # Attach into the detailed record
    # We store everything the evaluator returned so we can inspect it later.
    rec["eval"] = {
        "ok": bool(eval_obj.get("ok", False)),
        "score": float(eval_obj.get("score", 0.0)),
        "comment": eval_obj.get("comment", ""),
        "delta_action": eval_obj.get("delta_action"),
        # "ppo": eval_obj.get("ppo")
    }
    with open(mem_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    # Mirror fields onto the index row for fast filtering
    patch = {
        "eval_ok":    bool(eval_obj.get("ok", False)),
        "eval_score": float(eval_obj.get("score", 0.0)),
        "eval_heading": None,
        "eval_hold": None,
        # "ppo_return": None,
        "eval_detailed": None,
    }
    da = eval_obj.get("delta_action") or {}
    if isinstance(da, dict):
        patch["eval_heading"] = da.get("heading_change_deg")
        patch["eval_hold"]    = da.get("hold_steps")

    # --- NEW: enforce eval_ok=True if evaluator == LLM action ---
    # Get the original LLM action from the detailed record
    man = (rec.get("answer") or {}).get("maneuver") or {}
    llm_heading = man.get("heading_change_deg")
    llm_hold    = man.get("hold_steps")
    ev_heading  = patch.get("eval_heading")
    ev_hold     = patch.get("eval_hold")

    # If both match, force ok=True both in rec and index
    if _eq_int(ev_heading, llm_heading) and _eq_int(ev_hold, llm_hold):
        patch["eval_ok"] = True               # index view
        rec["eval"]["ok"] = True              # detailed JSON view
        # Friendly detail
        patch["eval_detailed"] = (
            f"Evaluator agrees with LLM action: turn {int(llm_heading)}°, hold {int(llm_hold)}."
        )
    else:
        # Disagreement: provide a concise why-string
        comment = (eval_obj.get("comment") or "").strip()
        reason  = ""
        if isinstance(da, dict):
            reason = (da.get("reason") or "").strip()
        # fallbacks to avoid empty strings
        why = reason or comment or "Evaluator recommends a different action."
        patch["eval_detailed"] = (
            f"Evaluator recommends turn {ev_heading}°, hold {ev_hold} "
            f"instead of LLM turn {llm_heading}°, hold {llm_hold}. Reason: {why}"
        )

        # Also write the updated 'rec' file so eval.ok stays consistent there too
        with open(mem_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)

    # Apply patch to the index line
    tag = (rec.get("meta") or {}).get("tag")
    if tag:
        _rewrite_index_line(index_path, tag, patch)

    # finally, enforce the cap
    enforce_index_cap(index_path, cap=100)



# HELPER FUNCTIONS
# bins are ...,-30,-25,...,0,...,25,30 (5° steps)
def _deg_to_action_idx(delta_deg: int) -> int:
    """Map ±(0,5,...,30)° to env action index:
       0..6 produce +0,+5,...,+30°, 7..12 produce -5,...,-30°."""
    # We round to be safe and make sure the value lives on the 5° grid.
    d = int(round(delta_deg))
    if d % 5 != 0 or abs(d) > 30:
        raise ValueError(f"delta must be multiple of 5 and |d|<=30, got {delta_deg}")
    if d >= 0:
        return d // 5           # 0°→0, +5°→1, ..., +30°→6
    else:
        return 6 + (abs(d) // 5)  # -5°→7, ..., -30°→12
    
# ADD: inverse mapping for debugging/ghost PPO branch
def _action_idx_to_deg(idx: int) -> int:
    """Inverse of _deg_to_action_idx: 0..6 -> +0,+5,...,+30; 7..12 -> -5,...,-30"""
    i = int(idx)
    if i < 0 or i > 12:
        raise ValueError(f"idx must be in [0,12], got {idx}")
    if i <= 6:
        return 5 * i
    return -5 * (i - 6)

def _ensure_clean_dir(path: str):
    # We make sure the directory exists first.
    os.makedirs(path, exist_ok=True)
    # wipe only PNGs, keep any prior GIFs/logs
    for p in glob.glob(os.path.join(path, "image_*.png")):
        try: os.remove(p)
        except OSError: pass

def _save_gif(frames_dir: str, out_gif_path: str, fps: float = 5.0):
    import imageio.v2 as imageio
    frames = sorted(glob.glob(os.path.join(frames_dir, "image_*.png")))
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}. Did env.render() run?")
    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(out_gif_path, images, duration=1.0/max(1e-6, fps))

def ghost_compare_hold(env: StructuredEnv,
                       ppo: PPO,
                       obs_vec: np.ndarray,
                       llm_turn_deg: int,
                       hold_steps: int,
                       intent_embed: Optional[np.ndarray],
                       phase_active: int) -> dict:
    """
    Compare two ghost futures (LLM vs PPO) for horizon N=max(1, hold_steps).
    Leaves the real env untouched. Very verbose for debugging.
    """
    print(f"[GHOST] start: llm_turn_deg={llm_turn_deg}, hold_steps={hold_steps}")

    # 1) Horizon from hold
    N = max(1, int(hold_steps))
    print(f"[GHOST] horizon N={N}")

    # 2) Build fused embedding & PPO greedy first-turn
    state_t  = torch.tensor(obs_vec, dtype=torch.float32, device=device)
    norm_st  = ppo._normalize_obs_eval(state_t)  # do NOT update stats during ghost compare

    if intent_embed is None:
        intent_t = torch.zeros(4, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t   = torch.tensor([0.0], dtype=torch.float32, device=device)
    else:
        intent_t = torch.tensor(intent_embed, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t   = torch.tensor([1.0 if phase_active else 0.0], dtype=torch.float32, device=device)

    with torch.no_grad():
        z = ppo.policy_old.fuse(norm_st.unsqueeze(0), intent_t, mask_t)     # (1, d_model)
        probs = ppo.policy_old.actor(z)                                      # (1, 13)
        ppo_idx = torch.argmax(probs, dim=-1).item()
    ppo_turn_deg = int(_action_idx_to_deg(ppo_idx))
    print(f"[GHOST] ppo_idx={ppo_idx} → ppo_turn_deg={ppo_turn_deg}")

    # --- helper: minimal "clone" if deepcopy fails ---
    def _clone_env_min(src: StructuredEnv) -> StructuredEnv:
        print("[GHOST] deepcopy failed -> using minimal clone()")
        e = StructuredEnv(Reward_Params=src.Reward_Params, start_llm_at_step=src.start_llm_at_step)
        # carry over dynamic state
        try:
            e.agent = copy.deepcopy(src.agent)
        except Exception as ce:
            print(f"[GHOST] copy.deepcopy(agent) failed: {ce}")
            traceback.print_exc()
            # last resort: manual agent fields (positions/heading only)
            # We manually copy the critical pieces so replay keeps working.
            e.agent.o_position = list(src.agent.o_position)
            e.agent.i_position = list(src.agent.i_position)
            e.agent.o_heading  = float(src.agent.o_heading)
            e.agent.o_newpathlist = [list(p) for p in src.agent.o_newpathlist]
        e.reward = float(src.reward)
        e.n_step = int(src.n_step)
        e.total_reward = float(src.total_reward)
        return e

    def _copy_env(src: StructuredEnv) -> StructuredEnv:
        try:
            # We prefer a full deepcopy so every field is duplicated.
            e = copy.deepcopy(src)
            print("[GHOST] deepcopy(env) OK")
            return e
        except Exception as ce:
            print(f"[GHOST] deepcopy(env) FAILED: {ce}")
            traceback.print_exc()
            return _clone_env_min(src)

    # Helper: run one branch for N steps, first-step action is 'first_deg', then hold straight (0°)
    def _run_branch(env_src: StructuredEnv, first_deg: int, N: int) -> float:
        # We clone the environment so we do not touch the real one.
        e = _copy_env(env_src)
        total = 0.0
        for k in range(N):
            # The very first step uses the chosen turn; later steps fly straight.
            a_idx = _deg_to_action_idx(first_deg if k == 0 else 0)
            _, r, term, trunc, _ = e.step(a_idx)
            total += r
            if term or trunc:
                print(f"[GHOST] branch(first_deg={first_deg}) terminated at k={k+1}/{N}, partial_return={total:.4f}")
                break
        return float(total)

    # 3) Roll each branch
    llm_return = _run_branch(env, llm_turn_deg, N)
    ppo_return = _run_branch(env, ppo_turn_deg, N)
    print(f"[GHOST] returns: llm_return={llm_return:.6f}, ppo_return={ppo_return:.6f}, horizon={N}")

    return {
        "llm_return": llm_return,
        "ppo_return": ppo_return,
        "ppo_turn_deg": ppo_turn_deg,
        "return_horizon": N
    }


# === Phase 3: Minimal controller (FSM + hold semantics) ===


NEXT_PHASE = {
    "WAIT_TURN":   "EXECUTE_TURN",
    "EXECUTE_TURN":"EXECUTE_TURN",
}

def _round_down_bin(x: float) -> int:
    """Largest allowed bin ≤ x (5,10,15,20,25,30). Returns 0 if x<5."""
    # We normalise the value to a positive degree amount before bucketing.
    x = abs(float(x))
    for b in (30, 25, 20, 15, 10, 5):
        if x >= b: return b
    return 0

def _heading_error_to_dest(cur: ObsSnapshot) -> float:
    own_h = _rad2deg(cur.own_heading_rad)
    to_dest = _bearing_deg(cur.dest_y - cur.own_y, cur.dest_x - cur.own_x)
    # Positive result means the destination sits to the left of the nose.
    return _wrap_deg(to_dest - own_h)  # positive = destination to LEFT; negative = to RIGHT

@dataclass
class ControllerState:
    phase: str = "WAIT_TURN"
    hold_remaining: int = 0
    last_turn_deg: int = 0  # last non-zero command we applied (for logging)
    step: int = -1

from typing import Tuple, Optional, Deque
import numpy as np
from collections import deque

# ---- Controller policy knobs (simple, readable) ----
SAFE_R = 5.0

PHASE_MIN_HOLD = {"WAIT_TURN": 0, "EXECUTE_TURN": 3}
PHASE_MAX_HOLD = {"WAIT_TURN": 1, "EXECUTE_TURN": 6}

# Look-ahead limits
EXTRA_HOLD_MAX = 6         # we’ll search N in [1..6]


class LLMGuidanceController:
    """
    Calls the LLM only at the start of each phase (when hold_remaining == 0).
    Applies the chosen heading_change_deg ONCE, then flies straight for `hold_steps` future steps.
    Transitions phases linearly: WAIT_TURN → EXECUTE_TURN.
    """
    def __init__(self, builder: BasePromptBuilder, k_history: int = 5, save_dir: str = JSON_ANSWERS_DIR, start_llm_at_step: int = 10, memory_path: str = EVAL_MEMORY_PATH, run_id: str = None):
        self.builder = builder
        # We remember how many past observations to include in prompts.
        self.k_history = int(k_history)
        self.save_dir = save_dir
        self.run_id = run_id

        self.state = ControllerState()
        # We cache the previous observation and step for derived signals.
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_step: Optional[int] = None
        # These deques hold older observations so we can summarise trends.
        self._hist: Deque[np.ndarray] = deque(maxlen=self.k_history)  # oldest → newest (previous snapshots only)
        self._hist_steps: Deque[int] = deque(maxlen=self.k_history)
        self.start_llm_at_step = int(start_llm_at_step)  # call LLM only once ≥ this controller step
        self._phase_called = False  # NEW: ensure one LLM call per phase
        self.last_advice = {"phase": "WAIT_TURN", "turn_deg": 0, "hold": 0, "active": 0}
        self.memory_path = memory_path
        # NEW: memory bookkeeping for Plan 5
        self._hold_log: List[np.ndarray] = []
        # We store the latest memory file path so we can update it after the hold.
        self._current_mem_path: Optional[str] = None
        self._last_tag: Optional[str] = None
        # We remember the observation snapshot from before the hold started.
        self._obs_before_last: Optional[dict] = None

    def reset(self, initial_phase: str = "WAIT_TURN"):
        self.state = ControllerState(phase=initial_phase, hold_remaining=0, last_turn_deg=0, step=-1)
        self._prev_obs = None
        self._prev_step = None
        self._hist.clear()
        self._hist_steps.clear()
        self._phase_called = False   # NEW
        self._hold_log.clear()
        self._current_mem_path = None
        self._last_tag = None
        self._obs_before_last = None

    def _advice_to_embed(self, phase: str, turn_deg: int, hold_steps: int):
        """
        Returns: small numpy vector for PPO [norm_turn, norm_hold, is_wait, is_exec]
        - turn normalized to [-1, +1] by /30
        - hold normalized to [0, 1] by /6 (your current max hold is 6)
        - flags: WAIT_TURN, EXECUTE_TURN (else both 0)
        """
        # We scale the turn so +30° becomes +1.0 and -30° becomes -1.0.
        norm_turn = float(turn_deg) / 30.0
        # We fit the hold count into [0,1] using the maximum allowed hold.
        norm_hold = float(min(max(hold_steps, 0), 6)) / 6.0
        # Flags let the network know which phase we are in.
        is_wait   = 1.0 if phase == "WAIT_TURN" else 0.0
        is_exec   = 1.0 if phase == "EXECUTE_TURN" else 0.0
        return np.array([norm_turn, norm_hold, is_wait, is_exec], dtype=np.float32)

    def get_advice_for_rl(self):
        """
        Safe getter for PPO.

        Returns:
        - embed_vec: np.array length-4 [norm_turn, norm_hold, is_wait, is_exec]
        - target_deg: int snapped degrees
        - phase_mask: int {0,1} -> 1 means "use attention + cache now" (active in BOTH WAIT_TURN & EXECUTE_TURN)
        """
        phase = self.last_advice["phase"]
        emb = self._advice_to_embed(
            phase,
            int(self.last_advice["turn_deg"]),
            int(self.last_advice["hold"]),
        )
        target_deg = int(self.last_advice["turn_deg"])
        phase_mask = 1 if self.last_advice.get("active", 0) == 1 else 0
        # PPO expects the embedding vector, the snapped heading, and an "active" mask.
        return emb, target_deg, phase_mask

    def _snap(self, vec, step) -> ObsSnapshot:
        # Convenience helper that converts a raw vector into a snapshot dataclass.
        return ObsSnapshot.from_vector(vec, step=step)

    def _estimate_vel(self, prev: ObsSnapshot, cur: ObsSnapshot):
        # We estimate simple per-step velocities for both aircraft.
        dt = 1 if (prev.step is None or cur.step is None or cur.step == prev.step) else max(1, cur.step - prev.step)
        vox, voy = (cur.own_x - prev.own_x)/dt, (cur.own_y - prev.own_y)/dt
        vix, viy = (cur.intr_x - prev.intr_x)/dt, (cur.intr_y - prev.intr_y)/dt
        return vox, voy, vix, viy, dt

    def _predict_N(self, cur: ObsSnapshot, prev: Optional[ObsSnapshot], N: int):
        """
        Straight-line preview for N steps:
        - ownship keeps its CURRENT heading (after last turn) and current speed
        - intruder keeps its current velocity vector
        Returns: sep_N, dist_to_dest_N, ttcp_N
        """
        # own speed & heading
        kin = self.builder.last_kinematics or {}
        v_o = kin.get("own_speed", None)
        if v_o is None:
            # fallback from last two snapshots
            if prev is None: return None, None, None
            vox, voy, vix, viy, _ = self._estimate_vel(prev, cur)
            v_o = math.hypot(vox, voy)
        head_deg = _rad2deg(cur.own_heading_rad)
        ux, uy = math.cos(math.radians(head_deg)), math.sin(math.radians(head_deg))

        # intruder velocity
        if prev is None:
            # Without a previous frame we assume the intruder is stationary.
            vix = viy = 0.0
        else:
            _, _, vix, viy, _ = self._estimate_vel(prev, cur)

        # positions after N
        # We project both aircraft forward using the straight-line assumption.
        own_xN = cur.own_x + v_o * ux * N
        own_yN = cur.own_y + v_o * uy * N
        intr_xN = cur.intr_x + vix * N
        intr_yN = cur.intr_y + viy * N

        # separation
        # Hypotenuse gives us the distance between the projected aircraft.
        sepN = math.hypot(intr_xN - own_xN, intr_yN - own_yN)

        # own→dest distance
        dist_destN = math.hypot(cur.dest_x - own_xN, cur.dest_y - own_yN)

        # TTCP at predicted state (relative motion constant)
        rx, ry = (intr_xN - own_xN), (intr_yN - own_yN)
        vrx, vry = (vix - v_o*ux), (viy - v_o*uy)
        vrel2 = vrx*vrx + vry*vry
        if vrel2 > 0:
            t_star = - (rx*vrx + ry*vry) / vrel2
            ttcpN = max(0.0, t_star)
        else:
            ttcpN = None

        return sepN, dist_destN, ttcpN

    def _can_advance(self, phase: str, cur: ObsSnapshot, prev: Optional[ObsSnapshot]) -> bool:
        # WAIT_TURN → EXECUTE_TURN: always allowed (the controller decides when to leave WAIT based on hold)
        if phase == "WAIT_TURN":
            return True

    def _clamp_hold(self, phase: str, hold: int) -> int:
        lo = PHASE_MIN_HOLD.get(phase, 0); hi = PHASE_MAX_HOLD.get(phase, hold)
        # We cap the requested hold between the configured minimum and maximum.
        return max(lo, min(hi, hold))

    def _choose_extra_hold(self, cur: ObsSnapshot, prev: Optional[ObsSnapshot], phase: str) -> int:
        """
        If the gate is not met, try to find the smallest extra hold N (1..EXTRA_HOLD_MAX)
        that improves the right thing for this phase.
        """
        if phase == "EXECUTE_TURN":
            cur_sep = cur.sep_oi
            for N in range(1, EXTRA_HOLD_MAX+1):
                sepN, _, ttcpN = self._predict_N(cur, prev, N)
                if sepN is None: break
                # We keep holding only if separation or TTCP looks better in the preview.
                good = (sepN > cur_sep) or (ttcpN is not None and self.builder.last_kinematics and
                                            self.builder.last_kinematics.get("ttcp_steps") is not None and
                                            ttcpN > self.builder.last_kinematics["ttcp_steps"])
                if good:
                    return N
            return 0


    def _advance_phase_if_needed(self):
        nxt = NEXT_PHASE.get(self.state.phase, "EXECUTE_TURN")
        if nxt != self.state.phase:
            print(f"[FSM] Phase {self.state.phase} → {nxt}")
            self.state.phase = nxt
            self._phase_called = False  # NEW: allow one LLM call in the new phase
        else:
            self.state.phase = nxt


    def _should_call_llm(self) -> bool:
        if self.state.hold_remaining > 0:
            return False
        # We only dial the LLM when the phase has not already used it.
        return not self._phase_called   # NEW: one call per phase

    # REPLACE the whole _build_prompt method with this version
    def _build_prompt(self, obs_vec: np.ndarray, step: int) -> str:
        # We start with the shared prompt boilerplate plus observation details.
        prompt_core = self.builder.build_base_prompt(
            obs_vector=obs_vec,
            step=step,
            prev_obs_vector=self._prev_obs,
            prev_step=self._prev_step,
            history_vectors=list(self._hist),
            history_steps=list(self._hist_steps)
        )

        # Optional eval notes you already had
        notes = load_eval_notes(getattr(self, "memory_path", EVAL_MEMORY_PATH))

        # NEW: lightweight retrieval from memory/_index.jsonl (if present)
        idx_path = os.path.join(os.path.dirname(self.save_dir), "_index.jsonl")
        cur_before = make_obs_snapshot(obs_vec, self._prev_obs, step)
        matches = find_similar_cases(cur_before, self.state.phase, idx_path, k=3)
        past_block = _format_past_cases_block(matches)
        if past_block:
            # append the past-cases section AFTER the marker content (keeps PHASE block position intact)
            prompt_core = prompt_core + "\n\n" + past_block

        # PHASE block goes right before the marker as before
        return attach_phase_state_block(
            prompt_core,
            current_phase=self.state.phase,
            step_index=step,
            allowed_bins=ALLOWED_BINS,
            extra_notes=notes
        )

    
    # CHANGED signature to accept fileinfo so we can store it in meta
    def next_action(self, obs_vec: np.ndarray, step: int, fileinfo: Optional[Any] = None,
                env: Optional[StructuredEnv] = None, ppo: Optional[PPO] = None) -> Tuple[int, ControllerState]:
        threshold = self.start_llm_at_step
        cur_snap = self._snap(obs_vec, step)
        prev_snap = self._snap(self._prev_obs, self._prev_step) if self._prev_obs is not None else None

        # --- During HOLD: collect obs, decrement, and on 0 finalize memory (Plan 5) ---
        if not self._should_call_llm():
            if self.state.hold_remaining > 0:
                # collect current obs vector for min-sep computation
                self._hold_log.append(np.array(obs_vec, dtype=float))
                self.state.hold_remaining -= 1
                print(f"[HOLD] Step {step}: phase={self.state.phase}, remaining={self.state.hold_remaining}")

            # HOLD ended this tick → compute hold_effect + observation_after and update JSON + run evaluator
            if self.state.hold_remaining == 0 and self._phase_called and self._current_mem_path:
                # Build observation_after
                obs_after = make_obs_snapshot(obs_vec, self._prev_obs, step)
                # min separation during the hold window
                if len(self._hold_log):
                    sep_min = float(np.min([v[6] for v in self._hold_log]))
                else:
                    sep_min = float(obs_after["sep_oi"])

                before = self._obs_before_last or {}
                hold_effect = {
                    "n_steps": len(self._hold_log),
                    "min_sep_during_hold": sep_min,
                    "ttcp_end": obs_after["ttcp"],
                    "dist_o_dest_end": obs_after["dist_o_dest"],
                    "deltas": {
                        "sep_oi": None if before.get("sep_oi") is None else float(obs_after["sep_oi"]) - float(before["sep_oi"]),
                        "ttcp":   None if (before.get("ttcp") is None or obs_after.get("ttcp") is None) else float(obs_after["ttcp"]) - float(before["ttcp"]),
                        "dist_o_dest": None if before.get("dist_o_dest") is None else float(obs_after["dist_o_dest"]) - float(before["dist_o_dest"]),
                    },
                }

                index_path = os.path.join(os.path.dirname(self.save_dir), "_index.jsonl")
                update_memory_record(self._current_mem_path, hold_effect, obs_after, index_path=index_path)

                # ---- EVALUATOR: build context from the just-saved record, then write results back ----
                try:
                    with open(self._current_mem_path, "r", encoding="utf-8") as f:
                        rec_now = json.load(f)
                    # We build a compact context for the evaluator and run it right away.
                    ctx = build_eval_context(rec_now, hold_effect, obs_after)
                    ev  = call_llm_evaluator(ctx, self.save_dir, self._last_tag or rec_now.get("meta", {}).get("tag", "phase"))
                    apply_eval_to_record(self._current_mem_path, ev, index_path)
                except Exception as _e:
                    # evaluator is best-effort; keep going even if it fails
                    pass

                # clear window for next phase/decision
                # We reset the temporary buffers now that this decision is finished.
                self._hold_log.clear()
                self._current_mem_path = None
                self._obs_before_last = None

                # phase transitions as before
                if self.state.phase == "WAIT_TURN":
                    # After waiting we step into the execute phase and unlock a new LLM query.
                    self._advance_phase_if_needed()  # WAIT_TURN -> EXECUTE_TURN
                    self._phase_called = False
                elif self.state.phase == "EXECUTE_TURN":
                    # We mark that no advice is currently being enforced.
                    self.last_advice["active"] = 0

            # history bookkeeping (unchanged)
            if self._prev_obs is not None:
                self._hist.append(self._prev_obs)
                self._hist_steps.append(self._prev_step if self._prev_step is not None else (step - 1))
            self._prev_obs, self._prev_step = obs_vec.copy(), step
            self.state.step = step
            return 0, self.state  # keep heading during hold

        # --- Gate LLM calls until threshold (unchanged) ---
        if step < threshold:
            if self._prev_obs is not None:
                self._hist.append(self._prev_obs)
                self._hist_steps.append(self._prev_step if self._prev_step is not None else (step - 1))
            # We save the current observation for the next iteration and keep heading zero.
            self._prev_obs, self._prev_step = obs_vec.copy(), step
            self.state.step = step
            return 0, self.state

        # === Call LLM ONCE for this phase ===
        prompt_text = self._build_prompt(obs_vec, step)
        print(f"[LLM] Calling model at step {step} for phase {self.state.phase}")
        tag = f"t{step:03d}_{self.state.phase.lower()}"

        # build observation_before (Plan 4)
        obs_before = make_obs_snapshot(obs_vec, self._prev_obs, step)
        nums = self.builder.last_kinematics or {}
        # We reuse the latest kinematic stats when available for logging metrics.
        sep_val  = nums.get("sep", float(cur_snap.sep_oi))
        ttcp_val = nums.get("ttcp_steps", None)

        # Save memory record with 'before' and meta (Plan 2 + 4)
        ret = call_llm_for_maneuver(
            prompt_text,
            save_dir=self.save_dir,
            tag=tag,
            override_metrics={"separation_distance": sep_val, "ttcp": ttcp_val},
            expected_phase=self.state.phase,
            obs_before=obs_before,
            run_id=self.run_id,
            step=step,
            fileinfo=fileinfo,
        )
        result   = ret["normalized"]
        mem_path = ret["mem_path"]

        man = result["answer"]["maneuver"]
        turn_now   = int(man.get("heading_change_deg", 0))
        hold_steps = max(0, int(man.get("hold_steps", 0)))
        # We respect the per-phase min and max hold limits.
        hold_steps = self._clamp_hold(self.state.phase, hold_steps)

        # --- NEW: A/B ghost compare over the hold window using *same* fused intent ---
        try:
            print(f"[GHOST] invoking from controller: phase={self.state.phase}, step={step}, mem_path={mem_path}")
            if env is not None and ppo is not None:
                intent_for_fuse = self._advice_to_embed(self.state.phase, turn_now, hold_steps)
                compare = ghost_compare_hold(env, ppo, obs_vec, turn_now, hold_steps,
                                             intent_embed=intent_for_fuse, phase_active=1)

                print(f"[GHOST] compare result: {compare}")
                # Persist so index writer can include it later
                with open(mem_path, "r", encoding="utf-8") as f:
                    rec_now = json.load(f)
                rec_now["branch_compare"] = compare
                with open(mem_path, "w", encoding="utf-8") as f:
                    json.dump(rec_now, f, indent=2, ensure_ascii=False)
                print(f"[GHOST] branch_compare persisted to {mem_path}")
            else:
                # If we cannot run the comparison we just log it for transparency.
                print("[GHOST] skipped: env or ppo is None")
        except Exception as e:
            print(f"[GHOST] ERROR during ghost_compare_hold: {e}")
            traceback.print_exc()
            # Minimal fallback so index fields aren’t null
            try:
                with open(mem_path, "r", encoding="utf-8") as f:
                    rec_now = json.load(f)
            except Exception:
                rec_now = {}
            rec_now.setdefault("branch_compare", {})
            # We fill in a neutral comparison so downstream code still works.
            rec_now["branch_compare"].update({
                "llm_return": None,
                "ppo_return": None,
                "ppo_turn_deg": 0,
                "return_horizon": int(hold_steps)
            })
            try:
                with open(mem_path, "w", encoding="utf-8") as f:
                    json.dump(rec_now, f, indent=2, ensure_ascii=False)
                print(f"[GHOST] wrote fallback branch_compare to {mem_path}")
            except Exception as e2:
                print(f"[GHOST] fallback write failed: {e2}")
                traceback.print_exc()


        self._phase_called = True
        self.state.last_turn_deg  = turn_now
        self.state.hold_remaining = hold_steps
        self.state.step = step

        # start a fresh hold window for this decision (Plan 5)
        # We clear previous logs and remember the new memory file path.
        self._hold_log.clear()
        self._current_mem_path = mem_path
        self._last_tag = tag
        self._obs_before_last = obs_before

        # history bookkeeping (unchanged)
        if self._prev_obs is not None:
            self._hist.append(self._prev_obs)
            self._hist_steps.append(self._prev_step if self._prev_step is not None else (step - 1))
        # The latest observation becomes the "previous" frame for next turn.
        self._prev_obs, self._prev_step = obs_vec.copy(), step

        # advance phase if needed when hold=0 (unchanged control flow)
        if hold_steps == 0:
            if self._can_advance(self.state.phase, cur_snap, prev_snap):
                self._advance_phase_if_needed()
            else:
                extra = self._choose_extra_hold(cur_snap, prev_snap, self.state.phase)
                extra = self._clamp_hold(self.state.phase, max(extra, 1))
                # Force at least one straight step so we gather another observation.
                self.state.hold_remaining = extra

        self.last_advice = {
            "phase": self.state.phase,
            "turn_deg": int(turn_now),
            "hold": int(self.state.hold_remaining),
            "active": 1 if self.state.phase in ("WAIT_TURN","EXECUTE_TURN") else 0
        }
        # We return the chosen heading change along with the updated controller state.
        return turn_now,self.state
