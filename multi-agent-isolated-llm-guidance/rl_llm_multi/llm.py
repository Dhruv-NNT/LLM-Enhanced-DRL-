"""Centralized multi-agent three-call controller and prompt helpers for isolated pure-LLM episodes."""

from __future__ import annotations

import base64
import copy
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from configs import (
    ACTION_BINS,
    CLEAR_STREAK_REQUIRED,
    DESTINATION_ALIGNMENT_DEG,
    EMERGENCY_LOOKAHEAD_STEPS,
    EVAL_MEMORY_PATH,
    HAZARD_LOOKAHEAD_STEPS,
    JSON_ANSWERS_DIR,
    MERGE_BACK_RECHECK_STEPS,
    MERGE_BACK_REPEAT_ERROR_DEG,
    OLLAMA_HOST,
    OLLAMA_MAX_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_TOP_P,
    PREVIEW_TOP_K,
    ROUTE_RECOVERY_XTRACK_UNITS,
    SAFE_R,
    TURN_PREVIEW_STEPS,
)
from .utils import line_signed_cross_track

if TYPE_CHECKING:  # pragma: no cover
    from .core import AgentState, MultiAgentSectorCore


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AgentControllerState:
    stage: str = "FOLLOW_ROUTE"
    execute_called: bool = False
    emergency_count: int = 0
    merge_back_count: int = 0
    merge_back_recheck_remaining: int = 0
    last_advice_turn: int = 0
    last_call_name: Optional[str] = None
    last_merge_back_reason: Optional[str] = None


class MultiAgentPromptBuilder:
    """Build local per-aircraft prompts for sequential guidance."""

    def __init__(self, safe_r: float = SAFE_R) -> None:
        self.safe_r = float(safe_r)

    def build_prompt(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
        call_name: str,
        threat_rows: List[Dict[str, Any]],
        preview_rows: List[Dict[str, Any]],
        fixed_actions: Optional[Dict[str, int]] = None,
    ) -> str:
        sections = [
            self._instructions_text(call_name),
            self._ownship_text(core, agent_id, call_name),
            self._committed_actions_text(fixed_actions or {}),
            self._weather_text(core, agent_id),
            self._primary_threat_text(threat_rows),
            self._other_threats_text(threat_rows),
            self._preview_rows_text(preview_rows),
        ]
        return "\n\n".join(section for section in sections if section)

    def _instructions_text(self, call_name: str) -> str:
        bins_text = ", ".join(str(value) for value in ACTION_BINS)
        if call_name == "EXECUTE_TURN":
            stage_text = (
                "STAGE: EXECUTE_TURN. This aircraft needs its first avoidance turn.\n"
                "- Focus on this ownship only.\n"
                "- Use the local threat and preview information below.\n"
            )
        elif call_name == "EMERGENCY_MANEUVER":
            stage_text = (
                "STAGE: EMERGENCY_MANEUVER. A near-term hazard is predicted for this aircraft.\n"
                "- Focus on immediate safety for this ownship.\n"
                "- Prefer a non-zero turn if that improves safety.\n"
            )
        elif call_name == "MERGE_BACK":
            stage_text = (
                "STAGE: MERGE_BACK. This aircraft should recover toward destination when safe.\n"
                "- Bias the turn toward the destination side.\n"
                "- Keep recovery safe with respect to nearby traffic and weather.\n"
                "MERGE_BACK PRIORITY:\n"
                "- Choose a safe candidate that improves recovery toward destination.\n"
                "- Prefer larger progress_to_destination.\n"
                "- Prefer reducing cross_track_abs.\n"
                "- Use 0 only if it remains aligned and still makes good destination progress.\n"
            )
        else:
            stage_text = f"STAGE: {call_name}.\n"
        return (
            "ROLE: You are a cautious ATCO helper for one aircraft.\n"
            f"{stage_text}"
            f"- SAFE_R = {self.safe_r:.1f} units.\n"
            "- The weather ellipse is forbidden airspace.\n"
            "- Return exactly one action for the ownship only.\n"
            "OUTPUT RULES:\n"
            "- Return EXACTLY ONE JSON object.\n"
            f"- heading_change_deg must be one of {{{bins_text}}}.\n"
            "- Positive is left, negative is right, 0 means hold current heading.\n"
            "STRICT JSON SHAPE:\n"
            "{\n"
            "  \"answer\": {\n"
            "    \"scenario_summary\": \"<brief>\",\n"
            "    \"technical_summary\": \"<brief>\",\n"
            "    \"rationale\": \"<brief>\",\n"
            "    \"maneuver\": { \"heading_change_deg\": 0 }\n"
            "  }\n"
            "}"
        )

    def _ownship_text(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
        call_name: str,
    ) -> str:
        state = core.get_agent(agent_id)
        signed_xtrk, _, _ = line_signed_cross_track(state.route.linestring, state.position)
        boundary_distance = core.boundary_distance(state)
        heading_error_to_dest = _heading_error_to_destination(state)
        return "\n".join(
            [
                f"OWNSHIP {agent_id}:",
                f"- call_name={call_name}",
                f"- position=({state.position[0]:.1f},{state.position[1]:.1f}) heading_deg={math.degrees(state.heading_rad):+.1f}",
                (
                    f"- destination=({state.destination[0]:.1f},{state.destination[1]:.1f}) "
                    f"distance_to_destination={core.distance_to_destination(state):.2f} "
                    f"heading_error_to_destination_deg={heading_error_to_dest:+.1f}"
                ),
                (
                    f"- route_recovery_error: cross_track={signed_xtrk:+.2f} "
                    f"boundary_distance={'n/a' if boundary_distance is None else f'{boundary_distance:.2f}'}"
                ),
                (
                    f"- predicted_pair_loss_step={state.predicted_pair_loss_step} "
                    f"predicted_weather_entry_step={state.predicted_weather_entry_step} "
                    f"predicted_boundary_exit_step={state.predicted_boundary_exit_step}"
                ),
                (
                    f"- predicted_weather_clearance={state.predicted_weather_clearance:.2f} "
                    f"safe_streak={state.safe_streak} "
                    f"boundary_warning_active={int(bool(state.boundary_warning_active))}"
                ),
            ]
        )

    def _committed_actions_text(self, fixed_actions: Dict[str, int]) -> str:
        if not fixed_actions:
            return "EARLIER COMMITTED ACTIONS THIS STEP:\n- none"
        lines = ["EARLIER COMMITTED ACTIONS THIS STEP:"]
        for agent_id, turn_deg in sorted(fixed_actions.items()):
            lines.append(f"- {agent_id}: {int(turn_deg):+d} deg")
        return "\n".join(lines)

    def _weather_text(self, core: "MultiAgentSectorCore", agent_id: str) -> str:
        state = core.get_agent(agent_id)
        current_clearance = core.weather_signed_clearance(state.position)
        weather = core.weather_dict()
        if weather is None:
            return "LOCAL WEATHER RISK:\n- none"
        return "\n".join(
            [
                "LOCAL WEATHER RISK:",
                f"- current_clearance={current_clearance:.2f}",
                (
                    f"- predicted_weather_entry_step={state.predicted_weather_entry_step} "
                    f"predicted_min_clearance={state.predicted_weather_clearance:.2f}"
                ),
                (
                    f"- weather_center=({weather['center'][0]:.1f},{weather['center'][1]:.1f}) "
                    f"orientation_deg={math.degrees(weather['angle_rad']):+.1f}"
                ),
            ]
        )

    def _primary_threat_text(self, threat_rows: List[Dict[str, Any]]) -> str:
        if not threat_rows:
            return "PRIMARY THREAT:\n- none"
        row = threat_rows[0]
        return "\n".join(
            [
                f"PRIMARY THREAT: {row['intruder_id']}",
                (
                    f"- intruder_position=({row['intruder_position'][0]:.1f},{row['intruder_position'][1]:.1f}) "
                    f"intruder_heading_deg={row['intruder_heading_deg']:+.1f}"
                ),
                (
                    f"- current_distance={row['current_distance']:.2f} "
                    f"bearing_error_deg={row['bearing_error_deg']:+.1f}"
                ),
                (
                    f"- predicted_loss_step={row['predicted_loss_step']} "
                    f"predicted_min_sep={row['predicted_min_sep']:.2f}"
                ),
            ]
        )

    def _other_threats_text(self, threat_rows: List[Dict[str, Any]]) -> str:
        if len(threat_rows) <= 1:
            return "OTHER THREATS:\n- none"
        lines = ["OTHER THREATS:"]
        for row in threat_rows[1:4]:
            lines.append(
                "- "
                f"{row['intruder_id']}: distance={row['current_distance']:.2f} "
                f"predicted_loss_step={row['predicted_loss_step']} "
                f"predicted_min_sep={row['predicted_min_sep']:.2f}"
            )
        return "\n".join(lines)

    def _preview_rows_text(self, preview_rows: List[Dict[str, Any]]) -> str:
        lines = ["LOCAL TURN PREVIEW (already conditioned on earlier committed actions this step):"]
        if not preview_rows:
            lines.append("- none")
            return "\n".join(lines)
        for row in preview_rows[:PREVIEW_TOP_K]:
            lines.append(
                "- "
                f"turn={int(row['heading_change_deg']):+d} | "
                f"safe={int(bool(row['safe_over_preview']))} | "
                f"pair_loss_step={row['pair_loss_step']} | "
                f"weather_entry_step={row['weather_entry_step']} | "
                f"boundary_exit_step={row['boundary_exit_step']} | "
                f"min_sep_to_any={float(row['min_sep_to_any']):.2f} | "
                f"min_weather_clearance={float(row['min_weather_clearance']):.2f} | "
                f"progress_to_destination={float(row.get('progress_to_destination', 0.0)):.2f} | "
                f"cross_track_reduction={float(row.get('cross_track_reduction', 0.0)):.2f} | "
                f"end_distance_to_destination={float(row.get('end_distance_to_destination', 999.0)):.2f} | "
                f"end_cross_track_abs={float(row.get('end_cross_track_abs', 999.0)):.2f} | "
                f"end_heading_error_to_dest_deg={float(row['end_heading_error_to_dest_deg']):.1f}"
            )
        return "\n".join(lines)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _extract_json(text: str) -> Dict[str, Any]:
    match = JSON_OBJECT_RE.search(text.strip())
    if not match:
        raise ValueError("No JSON object found in LLM output.")
    return json.loads(match.group(0))


def _snap_to_allowed_bin(value: float) -> int:
    return int(min(ACTION_BINS, key=lambda candidate: abs(candidate - float(value))))


def _snap_to_allowed_turn(value: float, allowed_turns: Sequence[int]) -> int:
    turns = list(allowed_turns)
    if not turns:
        return 0
    return int(min(turns, key=lambda candidate: abs(candidate - float(value))))


def _wrap_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _pair_key(agent_a: str, agent_b: str) -> Tuple[str, str]:
    return tuple(sorted((str(agent_a), str(agent_b))))


def _heading_error_to_destination(state: "AgentState") -> float:
    dest_heading = math.degrees(
        math.atan2(
            state.destination[1] - state.position[1],
            state.destination[0] - state.position[0],
        )
    )
    return _wrap_deg(dest_heading - math.degrees(state.heading_rad))


def start_llm_log_dir(base_dir: str = str(JSON_ANSWERS_DIR)) -> Tuple[str, str]:
    _ensure_dir(base_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"run_{stamp}"
    run_dir = os.path.join(base_dir, run_id)
    _ensure_dir(run_dir)
    return run_dir, run_id


def ollama_invoke(
    prompt_text: str,
    *,
    image_paths: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    base_url = os.environ.get("OLLAMA_HOST", OLLAMA_HOST).rstrip("/")
    usable_images: List[str] = []
    for image_path in image_paths or []:
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as handle:
                usable_images.append(base64.b64encode(handle.read()).decode("ascii"))

    message: Dict[str, Any] = {"role": "user", "content": prompt_text}
    if usable_images:
        message["images"] = usable_images

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [message],
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "num_predict": OLLAMA_MAX_TOKENS,
        },
    }
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(OLLAMA_TIMEOUT_SECONDS)) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        return {
            "llm_status": "http_error",
            "raw_text": body,
            "error": str(exc),
            "used_vision": bool(usable_images),
        }
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        return {
            "llm_status": "connection_error",
            "raw_text": "",
            "error": str(exc),
            "used_vision": bool(usable_images),
        }

    try:
        parsed = json.loads(raw_body)
        content = (parsed.get("message") or {}).get("content", "")
        if content:
            return {
                "llm_status": "ok",
                "raw_text": str(content),
                "error": "",
                "used_vision": bool(usable_images),
            }
        return {
            "llm_status": "empty_content",
            "raw_text": raw_body,
            "error": "Ollama response did not contain message.content",
            "used_vision": bool(usable_images),
        }
    except Exception as exc:
        return {
            "llm_status": "bad_response",
            "raw_text": raw_body,
            "error": str(exc),
            "used_vision": bool(usable_images),
        }


class MultiAgentThreeCallController:
    """Per-aircraft pure-LLM controller with sequential local calls."""

    def __init__(
        self,
        builder: Optional[MultiAgentPromptBuilder] = None,
        *,
        save_dir: str = str(JSON_ANSWERS_DIR),
        memory_path: str = str(EVAL_MEMORY_PATH),
    ) -> None:
        self.builder = MultiAgentPromptBuilder() if builder is None else builder
        self.save_dir = save_dir
        self.memory_path = memory_path
        self.run_id: Optional[str] = None
        self.episode_id: Optional[int] = None
        self.agent_states: Dict[str, AgentControllerState] = {}
        self.last_prompt_text: Optional[str] = None
        self.last_raw_text: Optional[str] = None
        self.last_normalized: Optional[Dict[str, Any]] = None
        self.last_tag: Optional[str] = None
        self.last_annotations: List[str] = []
        self.last_joint_actions: Dict[str, int] = {}

    def reset(self) -> None:
        self.agent_states = {}
        self.last_prompt_text = None
        self.last_raw_text = None
        self.last_normalized = None
        self.last_tag = None
        self.last_annotations = []
        self.last_joint_actions = {}

    def _controller_state(self, agent_id: str) -> AgentControllerState:
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = AgentControllerState()
        return self.agent_states[agent_id]

    def _call_decision_for_agent(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
    ) -> Optional[Dict[str, str]]:
        state = core.get_agent(agent_id)
        ctrl = self._controller_state(agent_id)
        if state.finished or not state.launched:
            return None
        if not state.has_entered_sector:
            ctrl.stage = "FOLLOW_ROUTE"
            ctrl.merge_back_recheck_remaining = 0
            return None

        signed_xtrk, _, _ = line_signed_cross_track(state.route.linestring, state.position)
        heading_err = abs(_heading_error_to_destination(state))
        immediate_hazard = bool(
            (state.predicted_pair_loss_step is not None and state.predicted_pair_loss_step <= EMERGENCY_LOOKAHEAD_STEPS)
            or (
                state.predicted_weather_entry_step is not None
                and state.predicted_weather_entry_step <= EMERGENCY_LOOKAHEAD_STEPS
            )
        )
        any_hazard = bool(state.hazard_predicted)

        if any_hazard and not ctrl.execute_called:
            return {
                "agent_id": str(agent_id),
                "call_name": "EXECUTE_TURN",
                "call_reason": "FIRST_HAZARD",
            }
        if immediate_hazard:
            return {
                "agent_id": str(agent_id),
                "call_name": "EMERGENCY_MANEUVER",
                "call_reason": "IMMEDIATE_HAZARD",
            }
        if ctrl.stage == "MERGE_BACK_RECHECK":
            if ctrl.merge_back_recheck_remaining > 0:
                ctrl.merge_back_recheck_remaining -= 1
                return None
            if state.boundary_warning_active:
                return {
                    "agent_id": str(agent_id),
                    "call_name": "MERGE_BACK",
                    "call_reason": "BOUNDARY_LOOKAHEAD",
                }
            if heading_err > MERGE_BACK_REPEAT_ERROR_DEG:
                return {
                    "agent_id": str(agent_id),
                    "call_name": "MERGE_BACK",
                    "call_reason": "ALIGNMENT_REPEAT",
                }
            ctrl.stage = "WAIT_CLEAR"
            return None
        if ctrl.execute_called and state.boundary_warning_active:
            return {
                "agent_id": str(agent_id),
                "call_name": "MERGE_BACK",
                "call_reason": "BOUNDARY_LOOKAHEAD",
            }
        if (
            ctrl.execute_called
            and state.safe_streak >= CLEAR_STREAK_REQUIRED
            and (
                abs(signed_xtrk) > ROUTE_RECOVERY_XTRACK_UNITS
                or heading_err > DESTINATION_ALIGNMENT_DEG
            )
        ):
            return {
                "agent_id": str(agent_id),
                "call_name": "MERGE_BACK",
                "call_reason": "SAFE_STREAK",
            }
        return None

    def _phase_label_text(self, agent_id: str, call_name: str, call_reason: str) -> str:
        label_map = {
            "EXECUTE_TURN": "EXECUTE TURN",
            "EMERGENCY_MANEUVER": "EMERGENCY MANEUVER",
            "MERGE_BACK": "MERGE BACK",
        }
        label = label_map.get(call_name, call_name.replace("_", " "))
        if call_name == "MERGE_BACK" and call_reason == "BOUNDARY_LOOKAHEAD":
            label = f"{label} (BOUNDARY)"
        return f"{agent_id}: {label}"

    def _entered_sector_annotations(self, core: "MultiAgentSectorCore", step: int) -> List[str]:
        annotations: List[str] = []
        for agent_id in core.active_agent_ids:
            state = core.get_agent(agent_id)
            if state.sector_entry_step is not None and int(state.sector_entry_step) == int(step):
                annotations.append(f"{agent_id}: ENTERED SECTOR")
        return annotations

    def _allowed_turns(self, core: "MultiAgentSectorCore", agent_id: str, call_name: str) -> List[int]:
        state = core.get_agent(agent_id)
        if call_name == "MERGE_BACK":
            heading_err = _heading_error_to_destination(state)
            if abs(heading_err) <= 2.5:
                return [0]
            if heading_err > 0:
                return [value for value in ACTION_BINS if value >= 0]
            return [value for value in ACTION_BINS if value <= 0]
        if call_name == "EMERGENCY_MANEUVER":
            return [value for value in ACTION_BINS if value != 0]
        return list(ACTION_BINS)

    def _simulate_joint_horizon(
        self,
        core: "MultiAgentSectorCore",
        first_step_turns: Dict[str, int],
        *,
        horizon_steps: int,
    ) -> Dict[str, Any]:
        sim = core.clone()
        cumulative_reward = 0.0
        active_ids = list(core.active_agent_ids)
        min_sep_any = float("inf")
        pair_loss_step = {agent_id: None for agent_id in active_ids}
        per_agent_min_sep = {agent_id: float("inf") for agent_id in active_ids}
        min_weather_clearance = {agent_id: float("inf") for agent_id in active_ids}
        weather_entry_step = {agent_id: None for agent_id in active_ids}
        boundary_exit_step = {agent_id: None for agent_id in active_ids}
        pairwise_summary: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for tick in range(1, int(horizon_steps) + 1):
            turns = first_step_turns if tick == 1 else {}
            sim.step(turns, compute_predictions=False)
            cumulative_reward += float(sim.reward)

            for pair_key, distance in sim.last_pairwise_separations.items():
                agent_a, agent_b = pair_key
                key = _pair_key(agent_a, agent_b)
                summary = pairwise_summary.setdefault(
                    key,
                    {"min_sep": float("inf"), "min_step": None, "loss_step": None},
                )
                min_sep_any = min(min_sep_any, float(distance))
                if float(distance) < float(summary["min_sep"]):
                    summary["min_sep"] = float(distance)
                    summary["min_step"] = tick
                if float(distance) < core.safe_r and summary["loss_step"] is None:
                    summary["loss_step"] = tick
                per_agent_min_sep[agent_a] = min(per_agent_min_sep[agent_a], float(distance))
                per_agent_min_sep[agent_b] = min(per_agent_min_sep[agent_b], float(distance))
                if float(distance) < core.safe_r and pair_loss_step[agent_a] is None:
                    pair_loss_step[agent_a] = tick
                if float(distance) < core.safe_r and pair_loss_step[agent_b] is None:
                    pair_loss_step[agent_b] = tick

            for agent_id in active_ids:
                state = sim.agent_states.get(agent_id)
                if state is None or not state.launched or state.finished:
                    continue
                clearance = sim.weather_signed_clearance(state.position)
                min_weather_clearance[agent_id] = min(min_weather_clearance[agent_id], float(clearance))
                if clearance < 0.0 and weather_entry_step[agent_id] is None:
                    weather_entry_step[agent_id] = tick
                if state.has_entered_sector and not state.inside_sector and boundary_exit_step[agent_id] is None:
                    boundary_exit_step[agent_id] = tick

            if sim.last_team_done or sim.last_team_truncated:
                break

        if min_sep_any == float("inf"):
            min_sep_any = 999.0
        for agent_id in active_ids:
            if per_agent_min_sep[agent_id] == float("inf"):
                per_agent_min_sep[agent_id] = 999.0
            if min_weather_clearance[agent_id] == float("inf"):
                min_weather_clearance[agent_id] = 999.0
        for summary in pairwise_summary.values():
            if float(summary["min_sep"]) == float("inf"):
                summary["min_sep"] = 999.0

        end_heading_errors = {}
        end_cross_tracks = {}
        end_distances = {}
        for agent_id in active_ids:
            if agent_id not in sim.agent_states:
                continue
            state = sim.get_agent(agent_id)
            end_heading_errors[agent_id] = abs(_heading_error_to_destination(state))
            signed_xtrk, _, _ = line_signed_cross_track(state.route.linestring, state.position)
            end_cross_tracks[agent_id] = abs(signed_xtrk)
            end_distances[agent_id] = sim.distance_to_destination(state)
        return {
            "min_sep_any": float(min_sep_any),
            "pair_loss_step": pair_loss_step,
            "per_agent_min_sep": per_agent_min_sep,
            "min_weather_clearance": min_weather_clearance,
            "weather_entry_step": weather_entry_step,
            "boundary_exit_step": boundary_exit_step,
            "pairwise_summary": pairwise_summary,
            "team_done": bool(sim.last_team_done),
            "team_truncated": bool(sim.last_team_truncated),
            "cumulative_reward": float(cumulative_reward),
            "end_heading_error_to_dest_deg": end_heading_errors,
            "end_cross_track_abs": end_cross_tracks,
            "end_distance_to_destination": end_distances,
        }

    def _actionable_sort_key(
        self,
        core: "MultiAgentSectorCore",
        item: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        state = core.get_agent(item["agent_id"])
        earliest_bad_event = min(
            [
                step
                for step in [
                    state.predicted_pair_loss_step,
                    state.predicted_weather_entry_step,
                    state.predicted_boundary_exit_step,
                ]
                if step is not None
            ]
            or [999]
        )
        return (
            earliest_bad_event,
            float(state.predicted_min_sep),
            str(item["agent_id"]),
        )

    def _threat_sort_key(self, row: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            999 if row["predicted_loss_step"] is None else row["predicted_loss_step"],
            float(row["predicted_min_sep"]),
            float(row["current_distance"]),
            str(row["intruder_id"]),
        )

    def _ranked_threats(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
        *,
        fixed_actions: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        fixed_actions = {
            other_id: int(turn_deg)
            for other_id, turn_deg in (fixed_actions or {}).items()
            if other_id != agent_id
        }
        sim = self._simulate_joint_horizon(
            core,
            fixed_actions,
            horizon_steps=HAZARD_LOOKAHEAD_STEPS,
        )
        own_state = core.get_agent(agent_id)
        rows: List[Dict[str, Any]] = []
        for other_id in core.active_agent_ids:
            if other_id == agent_id:
                continue
            other_state = core.get_agent(other_id)
            pair = sim["pairwise_summary"].get(
                _pair_key(agent_id, other_id),
                {"min_sep": 999.0, "min_step": None, "loss_step": None},
            )
            current_distance = float(
                math.hypot(
                    own_state.position[0] - other_state.position[0],
                    own_state.position[1] - other_state.position[1],
                )
            )
            rows.append(
                {
                    "intruder_id": other_id,
                    "intruder_position": tuple(other_state.position),
                    "intruder_heading_deg": float(math.degrees(other_state.heading_rad)),
                    "current_distance": current_distance,
                    "bearing_error_deg": float(core._bearing_to_neighbor_deg(agent_id, other_id)),
                    "predicted_loss_step": pair["loss_step"],
                    "predicted_min_sep": float(pair["min_sep"]),
                    "predicted_min_step": pair["min_step"],
                }
            )
        rows.sort(key=self._threat_sort_key)
        return rows[:4]

    def _candidate_sort_key(self, row: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            0 if bool(row.get("safe_over_preview", False)) else 1,
            999 if row.get("pair_loss_step") is None else row.get("pair_loss_step"),
            999 if row.get("weather_entry_step") is None else row.get("weather_entry_step"),
            999 if row.get("boundary_exit_step") is None else row.get("boundary_exit_step"),
            -float(row.get("min_sep_to_any", 999.0)),
            -float(row.get("min_weather_clearance", 999.0)),
            float(row.get("end_heading_error_to_dest_deg", 999.0)),
            float(row.get("end_cross_track_abs", 999.0)),
            abs(int(row.get("heading_change_deg", 0))),
        )

    def _merge_back_candidate_sort_key(self, row: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            0 if bool(row.get("safe_over_preview", False)) else 1,
            999 if row.get("boundary_exit_step") is None else row.get("boundary_exit_step"),
            999 if row.get("weather_entry_step") is None else row.get("weather_entry_step"),
            999 if row.get("pair_loss_step") is None else row.get("pair_loss_step"),
            -float(row.get("progress_to_destination", 0.0)),
            -float(row.get("cross_track_reduction", 0.0)),
            float(row.get("end_heading_error_to_dest_deg", 999.0)),
            abs(int(row.get("heading_change_deg", 0))),
        )

    def _candidate_sort_key_for_call(self, call_name: str, row: Dict[str, Any]) -> Tuple[Any, ...]:
        if call_name == "MERGE_BACK":
            return self._merge_back_candidate_sort_key(row)
        return self._candidate_sort_key(row)

    def _preview_rows(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
        call_name: str,
        *,
        fixed_actions: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        fixed_actions = {} if fixed_actions is None else dict(fixed_actions)
        current_state = core.get_agent(agent_id)
        start_distance = core.distance_to_destination(current_state)
        start_signed_xtrk, _, _ = line_signed_cross_track(current_state.route.linestring, current_state.position)
        start_cross_track_abs = abs(start_signed_xtrk)
        for turn_deg in self._allowed_turns(core, agent_id, call_name):
            actions = dict(fixed_actions)
            actions[agent_id] = int(turn_deg)
            sim = self._simulate_joint_horizon(
                core,
                actions,
                horizon_steps=TURN_PREVIEW_STEPS,
            )
            row = {
                "heading_change_deg": int(turn_deg),
                "pair_loss_step": sim["pair_loss_step"].get(agent_id),
                "weather_entry_step": sim["weather_entry_step"].get(agent_id),
                "boundary_exit_step": sim["boundary_exit_step"].get(agent_id),
                "min_sep_to_any": float(sim["per_agent_min_sep"].get(agent_id, 999.0)),
                "min_weather_clearance": float(sim["min_weather_clearance"].get(agent_id, 999.0)),
                "safe_over_preview": bool(
                    sim["pair_loss_step"].get(agent_id) is None
                    and sim["weather_entry_step"].get(agent_id) is None
                    and sim["boundary_exit_step"].get(agent_id) is None
                    and float(sim["per_agent_min_sep"].get(agent_id, 999.0)) >= core.safe_r
                ),
                "end_heading_error_to_dest_deg": float(
                    sim["end_heading_error_to_dest_deg"].get(agent_id, 999.0)
                ),
                "start_distance_to_destination": float(start_distance),
                "end_distance_to_destination": float(
                    sim["end_distance_to_destination"].get(agent_id, 999.0)
                ),
                "progress_to_destination": float(
                    start_distance - sim["end_distance_to_destination"].get(agent_id, 999.0)
                ),
                "start_cross_track_abs": float(start_cross_track_abs),
                "end_cross_track_abs": float(sim["end_cross_track_abs"].get(agent_id, 999.0)),
                "cross_track_reduction": float(
                    start_cross_track_abs - sim["end_cross_track_abs"].get(agent_id, 999.0)
                ),
            }
            rows.append(row)
        rows.sort(key=lambda row: self._candidate_sort_key_for_call(call_name, row))
        return rows

    def _deterministic_best_turn(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
        call_name: str,
        *,
        fixed_actions: Optional[Dict[str, int]] = None,
    ) -> int:
        best_turn = 0
        best_key: Optional[Tuple[Any, ...]] = None
        for row in self._preview_rows(
            core,
            agent_id,
            call_name,
            fixed_actions=fixed_actions,
        ):
            key = self._candidate_sort_key_for_call(call_name, row)
            if best_key is None or key < best_key:
                best_key = key
                best_turn = int(row["heading_change_deg"])
        return best_turn

    def _fallback_answer(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
        call_name: str,
        *,
        fixed_actions: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        return {
            "answer": {
                "scenario_summary": "",
                "technical_summary": "",
                "rationale": "deterministic fallback choice",
                "maneuver": {
                    "heading_change_deg": int(
                        self._deterministic_best_turn(
                            core,
                            agent_id,
                            call_name,
                            fixed_actions=fixed_actions,
                        )
                    ),
                },
            }
        }

    def _normalize_local_answer(
        self,
        payload: Dict[str, Any],
        agent_id: str,
        call_name: str,
    ) -> Dict[str, Any]:
        answer = payload.get("answer") if isinstance(payload, dict) else {}
        answer = answer if isinstance(answer, dict) else {}
        maneuver = answer.get("maneuver") if isinstance(answer.get("maneuver"), dict) else {}
        raw_heading = maneuver.get("heading_change_deg", answer.get("heading_change_deg", 0))
        try:
            snapped = _snap_to_allowed_bin(float(raw_heading))
        except Exception:
            snapped = 0
        return {
            "answer": {
                "agent_id": agent_id,
                "call_name": call_name,
                "scenario_summary": answer.get("scenario_summary", "")
                if isinstance(answer.get("scenario_summary"), str)
                else "",
                "technical_summary": answer.get("technical_summary", "")
                if isinstance(answer.get("technical_summary"), str)
                else "",
                "rationale": answer.get("rationale", "")
                if isinstance(answer.get("rationale"), str)
                else "",
                "maneuver": {"heading_change_deg": int(snapped)},
            }
        }

    def _resolve_turn_for_call(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
        call_name: str,
        requested_turn: int,
    ) -> int:
        allowed_turns = self._allowed_turns(core, agent_id, call_name)
        snapped_turn = _snap_to_allowed_turn(int(requested_turn), allowed_turns)
        return int(snapped_turn)

    def _log_local_call(
        self,
        *,
        step: int,
        agent_id: str,
        call_name: str,
        prompt_text: str,
        raw_text: str,
        normalized: Dict[str, Any],
        threat_rows: List[Dict[str, Any]],
        preview_rows: List[Dict[str, Any]],
        debug: Dict[str, Any],
    ) -> None:
        _ensure_dir(self.save_dir)
        tag = f"t{step:03d}_{agent_id.lower()}_{call_name.lower()}"
        self.last_tag = tag
        self.last_prompt_text = prompt_text
        self.last_raw_text = raw_text
        self.last_normalized = normalized
        prompt_path = os.path.join(self.save_dir, f"{tag}.prompt.txt")
        response_path = os.path.join(self.save_dir, f"{tag}.response.txt")
        normalized_path = os.path.join(self.save_dir, f"{tag}.normalized.json")
        _save_text(prompt_path, prompt_text)
        _save_text(response_path, raw_text)
        payload = copy.deepcopy(normalized)
        payload["threat_rows"] = threat_rows
        payload["preview_rows"] = preview_rows
        payload["debug"] = debug
        _save_json(normalized_path, payload)
        with open(os.path.join(self.save_dir, "_index.jsonl"), "a", encoding="utf-8") as handle:
            row = {
                "tag": tag,
                "step": step,
                "run_id": self.run_id,
                "episode_id": self.episode_id,
                "agent_id": agent_id,
                "call_name": call_name,
                "llm_status": debug.get("llm_status"),
                "parse_status": debug.get("parse_status"),
                "used_vision": debug.get("used_vision"),
            }
            handle.write(json.dumps(row) + "\n")

    def update(
        self,
        core: "MultiAgentSectorCore",
        *,
        step: int,
        latest_frame_path: Optional[str] = None,
        use_vision: bool = False,
    ) -> Dict[str, int]:
        annotations = self._entered_sector_annotations(core, step)
        actionable: List[Dict[str, Any]] = []
        for agent_id in core.active_agent_ids:
            decision = self._call_decision_for_agent(core, agent_id)
            if decision is not None:
                actionable.append(decision)

        actionable.sort(key=lambda item: self._actionable_sort_key(core, item))
        if not actionable:
            self.last_annotations = annotations
            self.last_joint_actions = {}
            return {}

        planned_actions: Dict[str, int] = {}
        for item in actionable:
            agent_id = str(item["agent_id"])
            call_name = str(item["call_name"])
            call_reason = str(item.get("call_reason", ""))
            fixed_actions = dict(planned_actions)
            threat_rows = self._ranked_threats(core, agent_id, fixed_actions=fixed_actions)
            preview_rows = self._preview_rows(
                core,
                agent_id,
                call_name,
                fixed_actions=fixed_actions,
            )
            prompt_text = self.builder.build_prompt(
                core,
                agent_id,
                call_name,
                threat_rows,
                preview_rows,
                fixed_actions=fixed_actions,
            )
            fallback = self._fallback_answer(
                core,
                agent_id,
                call_name,
                fixed_actions=fixed_actions,
            )

            llm_result = {"llm_status": "skipped", "raw_text": "", "error": "", "used_vision": False}
            parse_status = "not_attempted"
            fallback_reason = ""
            parsed: Dict[str, Any] = fallback

            if latest_frame_path and use_vision:
                llm_result = ollama_invoke(prompt_text, image_paths=[latest_frame_path])
            else:
                llm_result = ollama_invoke(prompt_text, image_paths=None)

            raw_text = str(llm_result.get("raw_text", ""))
            if llm_result.get("llm_status") == "ok":
                try:
                    parsed = _extract_json(raw_text)
                    parse_status = "ok"
                except Exception:
                    parsed = fallback
                    parse_status = "invalid_json"
                    fallback_reason = "parse_error"
            else:
                parsed = fallback
                parse_status = "not_attempted"
                fallback_reason = str(llm_result.get("llm_status"))

            normalized = self._normalize_local_answer(parsed, agent_id, call_name)
            applied_turn = self._resolve_turn_for_call(
                core,
                agent_id,
                call_name,
                int(normalized["answer"]["maneuver"]["heading_change_deg"]),
            )
            normalized["answer"]["maneuver"]["heading_change_deg"] = int(applied_turn)
            planned_actions[agent_id] = int(applied_turn)

            ctrl = self._controller_state(agent_id)
            ctrl.last_advice_turn = int(applied_turn)
            ctrl.last_call_name = call_name
            ctrl.last_merge_back_reason = call_reason or None
            if call_name == "EXECUTE_TURN":
                ctrl.execute_called = True
                ctrl.stage = "WAIT_CLEAR"
                ctrl.merge_back_recheck_remaining = 0
            elif call_name == "EMERGENCY_MANEUVER":
                ctrl.execute_called = True
                ctrl.emergency_count += 1
                ctrl.stage = "WAIT_CLEAR"
                ctrl.merge_back_recheck_remaining = 0
            elif call_name == "MERGE_BACK":
                ctrl.merge_back_count += 1
                ctrl.stage = "MERGE_BACK_RECHECK"
                ctrl.merge_back_recheck_remaining = int(MERGE_BACK_RECHECK_STEPS)
            annotations.append(self._phase_label_text(agent_id, call_name, call_reason))

            debug = {
                "llm_status": llm_result.get("llm_status"),
                "llm_error": llm_result.get("error", ""),
                "parse_status": parse_status,
                "fallback_reason": fallback_reason,
                "call_reason": call_reason,
                "used_vision": bool(llm_result.get("used_vision", False)),
                "frame_path": latest_frame_path if latest_frame_path and use_vision else None,
                "fixed_actions": fixed_actions,
                "threat_ids": [row["intruder_id"] for row in threat_rows],
            }
            self._log_local_call(
                step=step,
                agent_id=agent_id,
                call_name=call_name,
                prompt_text=prompt_text,
                raw_text=raw_text,
                normalized=normalized,
                threat_rows=threat_rows,
                preview_rows=preview_rows,
                debug=debug,
            )

        self.last_annotations = annotations
        self.last_joint_actions = planned_actions
        return planned_actions

    def choose_llm_actions(
        self,
        core: "MultiAgentSectorCore",
        *,
        step: int,
        latest_frame_path: Optional[str] = None,
        use_vision: bool = False,
    ) -> Dict[str, int]:
        return self.update(
            core,
            step=step,
            latest_frame_path=latest_frame_path,
            use_vision=use_vision,
        )


class GlobalPromptBuilder:
    """Build one global prompt for all entered aircraft due for guidance."""

    def __init__(self, safe_r: float = SAFE_R) -> None:
        self.safe_r = float(safe_r)

    def build_prompt(
        self,
        core: "MultiAgentSectorCore",
        *,
        step: int,
        actionable: List[Dict[str, Any]],
        entered_agent_ids: List[str],
        threat_rows_by_agent: Dict[str, List[Dict[str, Any]]],
        preview_rows_by_agent: Dict[str, List[Dict[str, Any]]],
        recent_decisions: Sequence[Dict[str, Any]],
        frame_paths: Sequence[str],
    ) -> str:
        sections = [
            self._instructions_text(actionable),
            self._merge_back_priority_text(actionable),
            self._eligible_agents_text(core, actionable),
            self._entered_context_text(core, entered_agent_ids),
            self._weather_text(core),
            self._threats_text(threat_rows_by_agent),
            self._preview_text(preview_rows_by_agent),
            self._recent_memory_text(recent_decisions),
            self._frame_context_text(frame_paths),
        ]
        return "\n\n".join(section for section in sections if section)

    def _instructions_text(self, actionable: List[Dict[str, Any]]) -> str:
        bins_text = ", ".join(str(value) for value in ACTION_BINS)
        agent_ids = ", ".join(str(item["agent_id"]) for item in actionable) or "none"
        return (
            "ROLE: You are a cautious ATCO helper for the entered aircraft listed below.\n"
            "GLOBAL GUIDANCE TASK:\n"
            f"- Return one maneuver for every GUIDANCE-ELIGIBLE agent: {agent_ids}.\n"
            "- Do not return actions for aircraft that have not entered the sector.\n"
            "- Use the simulator-generated candidate previews as the main decision evidence.\n"
            f"- SAFE_R = {self.safe_r:.1f} units.\n"
            "- The weather ellipse is forbidden airspace.\n"
            "OUTPUT RULES:\n"
            "- Return EXACTLY ONE JSON object and no extra text.\n"
            f"- heading_change_deg must be one of {{{bins_text}}}.\n"
            "- Positive is left, negative is right, 0 means hold current heading.\n"
            "STRICT JSON SHAPE:\n"
            "{\n"
            "  \"answer\": {\n"
            "    \"scenario_summary\": \"<brief>\",\n"
            "    \"actions\": {\n"
            "      \"A1\": {\n"
            "        \"call_name\": \"EXECUTE_TURN\",\n"
            "        \"rationale\": \"<brief>\",\n"
            "        \"maneuver\": { \"heading_change_deg\": 0 }\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}"
        )

    def _merge_back_priority_text(self, actionable: List[Dict[str, Any]]) -> str:
        if not any(str(item.get("call_name")) == "MERGE_BACK" for item in actionable):
            return ""
        return (
            "MERGE_BACK PRIORITY:\n"
            "- Choose a safe candidate that improves recovery toward destination.\n"
            "- Prefer larger progress_to_destination.\n"
            "- Prefer reducing cross_track_abs.\n"
            "- Use 0 only if it remains aligned and still makes good destination progress."
        )

    def _agent_state_line(self, core: "MultiAgentSectorCore", agent_id: str) -> str:
        state = core.get_agent(agent_id)
        signed_xtrk, _, _ = line_signed_cross_track(state.route.linestring, state.position)
        boundary_distance = core.boundary_distance(state)
        return (
            f"{agent_id}: pos=({state.position[0]:.1f},{state.position[1]:.1f}) "
            f"heading={math.degrees(state.heading_rad):+.1f} "
            f"dest=({state.destination[0]:.1f},{state.destination[1]:.1f}) "
            f"dist_dest={core.distance_to_destination(state):.2f} "
            f"heading_err={_heading_error_to_destination(state):+.1f} "
            f"xtrack={signed_xtrk:+.2f} "
            f"boundary_dist={'n/a' if boundary_distance is None else f'{boundary_distance:.2f}'} "
            f"pair_loss={state.predicted_pair_loss_step} "
            f"weather_entry={state.predicted_weather_entry_step} "
            f"boundary_exit={state.predicted_boundary_exit_step} "
            f"weather_clearance={state.predicted_weather_clearance:.2f} "
            f"safe_streak={state.safe_streak}"
        )

    def _eligible_agents_text(
        self,
        core: "MultiAgentSectorCore",
        actionable: List[Dict[str, Any]],
    ) -> str:
        lines = ["GUIDANCE-ELIGIBLE AGENTS THIS STEP:"]
        if not actionable:
            lines.append("- none")
            return "\n".join(lines)
        for item in actionable:
            agent_id = str(item["agent_id"])
            lines.append(
                f"- {self._agent_state_line(core, agent_id)} "
                f"call_name={item['call_name']} call_reason={item.get('call_reason', '')}"
            )
        return "\n".join(lines)

    def _entered_context_text(self, core: "MultiAgentSectorCore", entered_agent_ids: List[str]) -> str:
        lines = ["ENTERED TRAFFIC CONTEXT:"]
        if not entered_agent_ids:
            lines.append("- none")
            return "\n".join(lines)
        for agent_id in entered_agent_ids:
            lines.append(f"- {self._agent_state_line(core, agent_id)}")
        return "\n".join(lines)

    def _weather_text(self, core: "MultiAgentSectorCore") -> str:
        weather = core.weather_dict()
        if weather is None:
            return "GLOBAL WEATHER:\n- none"
        return "\n".join(
            [
                "GLOBAL WEATHER:",
                (
                    f"- center=({weather['center'][0]:.1f},{weather['center'][1]:.1f}) "
                    f"orientation_deg={math.degrees(weather['angle_rad']):+.1f} "
                    f"motion_heading_deg={math.degrees(weather['motion_heading_rad']):+.1f} "
                    f"speed_units_per_step={float(weather['speed_units_per_step']):.2f}"
                ),
                (
                    f"- major_radius_nm={float(weather['major_radius_nm']):.2f} "
                    f"minor_radius_nm={float(weather['minor_radius_nm']):.2f}"
                ),
            ]
        )

    def _threats_text(self, threat_rows_by_agent: Dict[str, List[Dict[str, Any]]]) -> str:
        lines = ["RANKED TRAFFIC THREATS:"]
        if not threat_rows_by_agent:
            lines.append("- none")
            return "\n".join(lines)
        for agent_id, rows in sorted(threat_rows_by_agent.items()):
            if not rows:
                lines.append(f"- {agent_id}: none")
                continue
            compact_rows = []
            for row in rows[:4]:
                compact_rows.append(
                    f"{row['intruder_id']} dist={row['current_distance']:.2f} "
                    f"loss={row['predicted_loss_step']} min_sep={row['predicted_min_sep']:.2f}"
                )
            lines.append(f"- {agent_id}: " + "; ".join(compact_rows))
        return "\n".join(lines)

    def _preview_text(self, preview_rows_by_agent: Dict[str, List[Dict[str, Any]]]) -> str:
        lines = ["PER-AGENT TURN PREVIEWS:"]
        if not preview_rows_by_agent:
            lines.append("- none")
            return "\n".join(lines)
        for agent_id, rows in sorted(preview_rows_by_agent.items()):
            lines.append(f"- {agent_id}:")
            for row in rows[:PREVIEW_TOP_K]:
                lines.append(
                    "  "
                    f"turn={int(row['heading_change_deg']):+d} | "
                    f"safe={int(bool(row['safe_over_preview']))} | "
                    f"pair_loss={row['pair_loss_step']} | "
                    f"weather_entry={row['weather_entry_step']} | "
                    f"boundary_exit={row['boundary_exit_step']} | "
                    f"min_sep={float(row['min_sep_to_any']):.2f} | "
                    f"min_weather_clearance={float(row['min_weather_clearance']):.2f} | "
                    f"progress_to_destination={float(row.get('progress_to_destination', 0.0)):.2f} | "
                    f"cross_track_reduction={float(row.get('cross_track_reduction', 0.0)):.2f} | "
                    f"end_distance={float(row.get('end_distance_to_destination', 999.0)):.2f} | "
                    f"end_cross_track={float(row.get('end_cross_track_abs', 999.0)):.2f} | "
                    f"end_heading_error={float(row['end_heading_error_to_dest_deg']):.1f}"
                )
        return "\n".join(lines)

    def _recent_memory_text(self, recent_decisions: Sequence[Dict[str, Any]]) -> str:
        lines = ["RECENT GLOBAL DECISION MEMORY:"]
        if not recent_decisions:
            lines.append("- none")
            return "\n".join(lines)
        for item in recent_decisions[-4:]:
            actions = item.get("actions", {})
            fragments = []
            for agent_id, action in sorted(actions.items()):
                fragments.append(
                    f"{agent_id}:{action.get('call_name')} "
                    f"{int(action.get('heading_change_deg', 0)):+d}deg"
                )
            lines.append(f"- step={item.get('step')}: " + "; ".join(fragments))
        return "\n".join(lines)

    def _frame_context_text(self, frame_paths: Sequence[str]) -> str:
        lines = ["VISION FRAME CONTEXT:"]
        if not frame_paths:
            lines.append("- none")
            return "\n".join(lines)
        for idx, frame_path in enumerate(frame_paths):
            lines.append(f"- frame_{idx}: {os.path.basename(frame_path)}")
        return "\n".join(lines)


class GlobalLangGraphGuidanceController(MultiAgentThreeCallController):
    """Global LangGraph controller that asks for all due entered-agent actions at once."""

    def __init__(
        self,
        builder: Optional[GlobalPromptBuilder] = None,
        *,
        save_dir: str = str(JSON_ANSWERS_DIR),
        memory_path: str = str(EVAL_MEMORY_PATH),
        retry_limit: int = 2,
    ) -> None:
        super().__init__(save_dir=save_dir, memory_path=memory_path)
        self.global_builder = GlobalPromptBuilder() if builder is None else builder
        self.retry_limit = int(retry_limit)
        self.recent_decisions: List[Dict[str, Any]] = []
        self.graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "langgraph is required for GlobalLangGraphGuidanceController. "
                "Install it in the llmdrl conda environment."
            ) from exc

        graph = StateGraph(dict)
        graph.add_node("collect_state", self._graph_collect_state)
        graph.add_node("assign_stages", self._graph_assign_stages)
        graph.add_node("build_global_context", self._graph_build_global_context)
        graph.add_node("call_llm", self._graph_call_llm)
        graph.add_node("parse_validate_retry", self._graph_parse_validate_retry)
        graph.add_node("guardrail_actions", self._graph_guardrail_actions)
        graph.add_node("fallback_missing", self._graph_fallback_missing)
        graph.add_node("commit_and_log", self._graph_commit_and_log)
        graph.set_entry_point("collect_state")
        graph.add_edge("collect_state", "assign_stages")
        graph.add_edge("assign_stages", "build_global_context")
        graph.add_edge("build_global_context", "call_llm")
        graph.add_edge("call_llm", "parse_validate_retry")
        graph.add_edge("parse_validate_retry", "guardrail_actions")
        graph.add_edge("guardrail_actions", "fallback_missing")
        graph.add_edge("fallback_missing", "commit_and_log")
        graph.add_edge("commit_and_log", END)
        return graph.compile()

    def reset(self) -> None:
        super().reset()
        self.recent_decisions = []

    def _global_call_decision_for_agent(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
    ) -> Optional[Dict[str, str]]:
        state = core.get_agent(agent_id)
        ctrl = self._controller_state(agent_id)
        if state.finished or not state.launched:
            return None
        if not state.has_entered_sector:
            ctrl.stage = "FOLLOW_ROUTE"
            ctrl.merge_back_recheck_remaining = 0
            return None

        signed_xtrk, _, _ = line_signed_cross_track(state.route.linestring, state.position)
        heading_err = abs(_heading_error_to_destination(state))
        immediate_hazard = bool(
            (state.predicted_pair_loss_step is not None and state.predicted_pair_loss_step <= EMERGENCY_LOOKAHEAD_STEPS)
            or (
                state.predicted_weather_entry_step is not None
                and state.predicted_weather_entry_step <= EMERGENCY_LOOKAHEAD_STEPS
            )
        )

        if not ctrl.execute_called:
            return {
                "agent_id": str(agent_id),
                "call_name": "EXECUTE_TURN",
                "call_reason": "SECTOR_ENTRY",
            }
        if immediate_hazard:
            return {
                "agent_id": str(agent_id),
                "call_name": "EMERGENCY_MANEUVER",
                "call_reason": "IMMEDIATE_HAZARD",
            }
        if ctrl.stage == "MERGE_BACK_RECHECK":
            if ctrl.merge_back_recheck_remaining > 0:
                ctrl.merge_back_recheck_remaining -= 1
                return None
            if state.boundary_warning_active:
                return {
                    "agent_id": str(agent_id),
                    "call_name": "MERGE_BACK",
                    "call_reason": "BOUNDARY_LOOKAHEAD",
                }
            if heading_err > MERGE_BACK_REPEAT_ERROR_DEG:
                return {
                    "agent_id": str(agent_id),
                    "call_name": "MERGE_BACK",
                    "call_reason": "ALIGNMENT_REPEAT",
                }
            ctrl.stage = "WAIT_CLEAR"
            return None
        if ctrl.execute_called and state.boundary_warning_active:
            return {
                "agent_id": str(agent_id),
                "call_name": "MERGE_BACK",
                "call_reason": "BOUNDARY_LOOKAHEAD",
            }
        if (
            ctrl.execute_called
            and state.safe_streak >= CLEAR_STREAK_REQUIRED
            and (
                abs(signed_xtrk) > ROUTE_RECOVERY_XTRACK_UNITS
                or heading_err > DESTINATION_ALIGNMENT_DEG
            )
        ):
            return {
                "agent_id": str(agent_id),
                "call_name": "MERGE_BACK",
                "call_reason": "SAFE_STREAK",
            }
        return None

    def _recent_frame_paths(
        self,
        latest_frame_path: Optional[str],
        *,
        use_vision: bool,
        count: int = 4,
    ) -> List[str]:
        if not latest_frame_path or not use_vision:
            return []
        latest = Path(latest_frame_path)
        match = re.match(r"image_(\d+)\.png$", latest.name)
        if not match:
            return [str(latest)] if latest.exists() else []
        step = int(match.group(1))
        paths: List[str] = []
        for value in range(step, max(-1, step - int(count)), -1):
            candidate = latest.with_name(f"image_{value:03d}.png")
            if candidate.exists():
                paths.append(str(candidate))
        return paths

    def _graph_collect_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        core = state["core"]
        step = int(state["step"])
        annotations = self._entered_sector_annotations(core, step)
        entered_agent_ids = [
            agent_id
            for agent_id in core.active_agent_ids
            if core.get_agent(agent_id).has_entered_sector
            and core.get_agent(agent_id).launched
            and not core.get_agent(agent_id).finished
        ]
        frame_paths = self._recent_frame_paths(
            state.get("latest_frame_path"),
            use_vision=bool(state.get("use_vision", False)),
        )
        return {
            **state,
            "annotations": annotations,
            "entered_agent_ids": entered_agent_ids,
            "frame_paths": frame_paths,
        }

    def _graph_assign_stages(self, state: Dict[str, Any]) -> Dict[str, Any]:
        core = state["core"]
        actionable: List[Dict[str, Any]] = []
        for agent_id in state.get("entered_agent_ids", []):
            decision = self._global_call_decision_for_agent(core, agent_id)
            if decision is not None:
                actionable.append(decision)
        actionable.sort(key=lambda item: self._actionable_sort_key(core, item))
        return {**state, "actionable": actionable}

    def _graph_build_global_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        core = state["core"]
        actionable = list(state.get("actionable", []))
        threat_rows_by_agent: Dict[str, List[Dict[str, Any]]] = {}
        preview_rows_by_agent: Dict[str, List[Dict[str, Any]]] = {}
        for item in actionable:
            agent_id = str(item["agent_id"])
            call_name = str(item["call_name"])
            threat_rows_by_agent[agent_id] = self._ranked_threats(core, agent_id, fixed_actions=None)
            preview_rows_by_agent[agent_id] = self._preview_rows(
                core,
                agent_id,
                call_name,
                fixed_actions=None,
            )
        prompt_text = self.global_builder.build_prompt(
            core,
            step=int(state["step"]),
            actionable=actionable,
            entered_agent_ids=list(state.get("entered_agent_ids", [])),
            threat_rows_by_agent=threat_rows_by_agent,
            preview_rows_by_agent=preview_rows_by_agent,
            recent_decisions=self.recent_decisions,
            frame_paths=list(state.get("frame_paths", [])),
        )
        return {
            **state,
            "threat_rows_by_agent": threat_rows_by_agent,
            "preview_rows_by_agent": preview_rows_by_agent,
            "prompt_text": prompt_text,
        }

    def _graph_call_llm(self, state: Dict[str, Any]) -> Dict[str, Any]:
        actionable = list(state.get("actionable", []))
        if not actionable:
            return {**state, "llm_attempts": [], "parsed_payload": {}, "candidate_actions": {}}
        image_paths = list(state.get("frame_paths", [])) if state.get("use_vision") else None
        llm_result = ollama_invoke(str(state.get("prompt_text", "")), image_paths=image_paths)
        return {**state, "llm_attempts": [llm_result]}

    def _actions_from_payload(
        self,
        payload: Dict[str, Any],
        expected_agent_ids: Sequence[str],
    ) -> Tuple[Dict[str, Any], List[str]]:
        answer = payload.get("answer") if isinstance(payload, dict) else {}
        answer = answer if isinstance(answer, dict) else {}
        actions = answer.get("actions") if isinstance(answer.get("actions"), dict) else {}
        found: Dict[str, Any] = {}
        missing: List[str] = []
        for agent_id in expected_agent_ids:
            action = actions.get(agent_id)
            if isinstance(action, dict):
                found[agent_id] = action
            else:
                missing.append(agent_id)
        return found, missing

    def _graph_parse_validate_retry(self, state: Dict[str, Any]) -> Dict[str, Any]:
        actionable = list(state.get("actionable", []))
        expected_agent_ids = [str(item["agent_id"]) for item in actionable]
        if not expected_agent_ids:
            return {
                **state,
                "llm_attempts": list(state.get("llm_attempts", [])),
                "parsed_payload": {},
                "candidate_actions": {},
                "missing_agent_ids": [],
                "parse_status": "not_attempted",
                "fallback_reason": "",
            }
        attempts = list(state.get("llm_attempts", []))
        parsed_payload: Dict[str, Any] = {}
        candidate_actions: Dict[str, Any] = {}
        missing_agent_ids = list(expected_agent_ids)
        parse_status = "not_attempted" if not expected_agent_ids else "invalid_json"
        fallback_reason = ""

        for attempt_index in range(self.retry_limit + 1):
            if attempt_index > 0:
                image_paths = list(state.get("frame_paths", [])) if state.get("use_vision") else None
                attempts.append(ollama_invoke(str(state.get("prompt_text", "")), image_paths=image_paths))
            result = attempts[-1] if attempts else {"llm_status": "skipped", "raw_text": "", "error": ""}
            if result.get("llm_status") != "ok":
                parse_status = "not_attempted"
                fallback_reason = str(result.get("llm_status"))
                continue
            try:
                parsed_payload = _extract_json(str(result.get("raw_text", "")))
                candidate_actions, missing_agent_ids = self._actions_from_payload(
                    parsed_payload,
                    expected_agent_ids,
                )
                parse_status = "ok" if not missing_agent_ids else "missing_actions"
                fallback_reason = "" if not missing_agent_ids else "missing_actions"
                if not missing_agent_ids:
                    break
            except Exception:
                parsed_payload = {}
                candidate_actions = {}
                missing_agent_ids = list(expected_agent_ids)
                parse_status = "invalid_json"
                fallback_reason = "parse_error"

        return {
            **state,
            "llm_attempts": attempts,
            "parsed_payload": parsed_payload,
            "candidate_actions": candidate_actions,
            "missing_agent_ids": missing_agent_ids,
            "parse_status": parse_status,
            "fallback_reason": fallback_reason,
        }

    def _raw_requested_turn(self, action_payload: Dict[str, Any]) -> int:
        maneuver = action_payload.get("maneuver") if isinstance(action_payload.get("maneuver"), dict) else {}
        raw_heading = maneuver.get("heading_change_deg", action_payload.get("heading_change_deg", 0))
        try:
            return _snap_to_allowed_bin(float(raw_heading))
        except Exception:
            return 0

    def _graph_guardrail_actions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        core = state["core"]
        candidate_actions = dict(state.get("candidate_actions", {}))
        final_actions: Dict[str, Dict[str, Any]] = {}
        invalid_agent_ids: List[str] = []
        stage_by_agent = {str(item["agent_id"]): str(item["call_name"]) for item in state.get("actionable", [])}
        for agent_id, action_payload in candidate_actions.items():
            if agent_id not in stage_by_agent:
                continue
            if not isinstance(action_payload, dict):
                invalid_agent_ids.append(agent_id)
                continue
            call_name = stage_by_agent[agent_id]
            requested_turn = self._raw_requested_turn(action_payload)
            applied_turn = self._resolve_turn_for_call(core, agent_id, call_name, requested_turn)
            rationale = action_payload.get("rationale", "")
            final_actions[agent_id] = {
                "call_name": call_name,
                "heading_change_deg": int(applied_turn),
                "requested_heading_change_deg": int(requested_turn),
                "rationale": rationale if isinstance(rationale, str) else "",
                "source": "llm",
            }
        missing = set(state.get("missing_agent_ids", []))
        missing.update(invalid_agent_ids)
        return {
            **state,
            "final_actions": final_actions,
            "missing_agent_ids": sorted(missing),
        }

    def _graph_fallback_missing(self, state: Dict[str, Any]) -> Dict[str, Any]:
        core = state["core"]
        final_actions = dict(state.get("final_actions", {}))
        fallback_agents: List[str] = []
        stage_by_agent = {str(item["agent_id"]): str(item["call_name"]) for item in state.get("actionable", [])}
        for agent_id in stage_by_agent:
            if agent_id in final_actions:
                continue
            call_name = stage_by_agent[agent_id]
            fallback = self._fallback_answer(core, agent_id, call_name, fixed_actions=None)
            requested_turn = int(fallback["answer"]["maneuver"]["heading_change_deg"])
            applied_turn = self._resolve_turn_for_call(core, agent_id, call_name, requested_turn)
            final_actions[agent_id] = {
                "call_name": call_name,
                "heading_change_deg": int(applied_turn),
                "requested_heading_change_deg": int(requested_turn),
                "rationale": "deterministic fallback choice",
                "source": "fallback",
            }
            fallback_agents.append(agent_id)
        return {**state, "final_actions": final_actions, "fallback_agents": fallback_agents}

    def _normalized_global_payload(self, state: Dict[str, Any]) -> Dict[str, Any]:
        parsed_payload = state.get("parsed_payload", {})
        answer = parsed_payload.get("answer") if isinstance(parsed_payload, dict) else {}
        answer = answer if isinstance(answer, dict) else {}
        scenario_summary = answer.get("scenario_summary", "")
        actions = {}
        for agent_id, action in sorted(state.get("final_actions", {}).items()):
            actions[agent_id] = {
                "call_name": action["call_name"],
                "rationale": action.get("rationale", ""),
                "source": action.get("source", ""),
                "requested_heading_change_deg": int(action.get("requested_heading_change_deg", 0)),
                "maneuver": {"heading_change_deg": int(action["heading_change_deg"])},
            }
        return {
            "answer": {
                "scenario_summary": scenario_summary if isinstance(scenario_summary, str) else "",
                "actions": actions,
            }
        }

    def _log_global_call(
        self,
        *,
        step: int,
        prompt_text: str,
        raw_text: str,
        normalized: Dict[str, Any],
        debug: Dict[str, Any],
        threat_rows_by_agent: Dict[str, List[Dict[str, Any]]],
        preview_rows_by_agent: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        _ensure_dir(self.save_dir)
        tag = f"t{step:03d}_global_guidance"
        self.last_tag = tag
        self.last_prompt_text = prompt_text
        self.last_raw_text = raw_text
        prompt_path = os.path.join(self.save_dir, f"{tag}.prompt.txt")
        response_path = os.path.join(self.save_dir, f"{tag}.response.txt")
        normalized_path = os.path.join(self.save_dir, f"{tag}.normalized.json")
        _save_text(prompt_path, prompt_text)
        _save_text(response_path, raw_text)
        payload = copy.deepcopy(normalized)
        payload["threat_rows_by_agent"] = threat_rows_by_agent
        payload["preview_rows_by_agent"] = preview_rows_by_agent
        payload["debug"] = debug
        self.last_normalized = payload
        _save_json(normalized_path, payload)
        with open(os.path.join(self.save_dir, "_index.jsonl"), "a", encoding="utf-8") as handle:
            row = {
                "tag": tag,
                "step": step,
                "run_id": self.run_id,
                "episode_id": self.episode_id,
                "agent_id": "GLOBAL",
                "call_name": "GLOBAL_GUIDANCE",
                "llm_status": debug.get("llm_status"),
                "parse_status": debug.get("parse_status"),
                "used_vision": debug.get("used_vision"),
            }
            handle.write(json.dumps(row) + "\n")

    def _apply_controller_state(self, agent_id: str, call_name: str, call_reason: str, applied_turn: int) -> None:
        ctrl = self._controller_state(agent_id)
        ctrl.last_advice_turn = int(applied_turn)
        ctrl.last_call_name = call_name
        ctrl.last_merge_back_reason = call_reason or None
        if call_name == "EXECUTE_TURN":
            ctrl.execute_called = True
            ctrl.stage = "WAIT_CLEAR"
            ctrl.merge_back_recheck_remaining = 0
        elif call_name == "EMERGENCY_MANEUVER":
            ctrl.execute_called = True
            ctrl.emergency_count += 1
            ctrl.stage = "WAIT_CLEAR"
            ctrl.merge_back_recheck_remaining = 0
        elif call_name == "MERGE_BACK":
            ctrl.merge_back_count += 1
            ctrl.stage = "MERGE_BACK_RECHECK"
            ctrl.merge_back_recheck_remaining = int(MERGE_BACK_RECHECK_STEPS)

    def _graph_commit_and_log(self, state: Dict[str, Any]) -> Dict[str, Any]:
        step = int(state["step"])
        actionable = list(state.get("actionable", []))
        annotations = list(state.get("annotations", []))
        if not actionable:
            self.last_annotations = annotations
            self.last_joint_actions = {}
            return {**state, "actions": {}}

        call_reason_by_agent = {
            str(item["agent_id"]): str(item.get("call_reason", ""))
            for item in actionable
        }
        actions = {
            agent_id: int(action["heading_change_deg"])
            for agent_id, action in sorted(state.get("final_actions", {}).items())
        }
        for agent_id, action in sorted(state.get("final_actions", {}).items()):
            call_name = str(action["call_name"])
            call_reason = call_reason_by_agent.get(agent_id, "")
            self._apply_controller_state(agent_id, call_name, call_reason, int(action["heading_change_deg"]))
            annotations.append(self._phase_label_text(agent_id, call_name, call_reason))

        normalized = self._normalized_global_payload(state)
        attempts = list(state.get("llm_attempts", []))
        last_attempt = attempts[-1] if attempts else {"llm_status": "skipped", "raw_text": "", "error": ""}
        debug = {
            "llm_status": last_attempt.get("llm_status"),
            "llm_error": last_attempt.get("error", ""),
            "parse_status": state.get("parse_status"),
            "fallback_reason": state.get("fallback_reason", ""),
            "fallback_agents": list(state.get("fallback_agents", [])),
            "retry_count": max(0, len(attempts) - 1),
            "used_vision": bool(state.get("use_vision", False) and state.get("frame_paths")),
            "frame_paths": list(state.get("frame_paths", [])) if state.get("use_vision") else [],
            "eligible_agent_ids": [str(item["agent_id"]) for item in actionable],
            "entered_agent_ids": list(state.get("entered_agent_ids", [])),
            "stage_by_agent": {
                str(item["agent_id"]): str(item["call_name"])
                for item in actionable
            },
            "model": OLLAMA_MODEL,
        }
        self._log_global_call(
            step=step,
            prompt_text=str(state.get("prompt_text", "")),
            raw_text=str(last_attempt.get("raw_text", "")),
            normalized=normalized,
            debug=debug,
            threat_rows_by_agent=state.get("threat_rows_by_agent", {}),
            preview_rows_by_agent=state.get("preview_rows_by_agent", {}),
        )

        self.recent_decisions.append(
            {
                "step": step,
                "actions": {
                    agent_id: {
                        "call_name": state["final_actions"][agent_id]["call_name"],
                        "heading_change_deg": int(turn_deg),
                        "rationale": state["final_actions"][agent_id].get("rationale", ""),
                    }
                    for agent_id, turn_deg in actions.items()
                },
            }
        )
        self.recent_decisions = self.recent_decisions[-4:]
        self.last_annotations = annotations
        self.last_joint_actions = actions
        return {**state, "actions": actions}

    def update(
        self,
        core: "MultiAgentSectorCore",
        *,
        step: int,
        latest_frame_path: Optional[str] = None,
        use_vision: bool = False,
    ) -> Dict[str, int]:
        result = self.graph.invoke(
            {
                "core": core,
                "step": int(step),
                "latest_frame_path": latest_frame_path,
                "use_vision": bool(use_vision),
            }
        )
        return dict(result.get("actions", {}))
