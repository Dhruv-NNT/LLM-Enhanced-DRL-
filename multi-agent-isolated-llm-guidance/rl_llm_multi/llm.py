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
        for agent_id in active_ids:
            if agent_id not in sim.agent_states:
                continue
            state = sim.get_agent(agent_id)
            end_heading_errors[agent_id] = abs(_heading_error_to_destination(state))
            signed_xtrk, _, _ = line_signed_cross_track(state.route.linestring, state.position)
            end_cross_tracks[agent_id] = abs(signed_xtrk)
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
                "end_cross_track_abs": float(sim["end_cross_track_abs"].get(agent_id, 999.0)),
            }
            rows.append(row)
        rows.sort(key=self._candidate_sort_key)
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
            key = self._candidate_sort_key(row)
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
