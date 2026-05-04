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

from shapely.geometry import Point

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
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_TOP_P,
    PREVIEW_TOP_K,
    ROUTE_RECOVERY_XTRACK_UNITS,
    SAFE_R,
    TRAFFIC_CAUTION_R,
    TURN_PREVIEW_STEPS,
)
from .utils import heading_from_line, line_signed_cross_track, load_waypoint_map

if TYPE_CHECKING:  # pragma: no cover
    from .core import AgentState, MultiAgentSectorCore


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

WEATHER_SEVERITY_LEGEND = (
    "- Weather severity rings: green=light precipitation, yellow=moderate rain/rain and hail, "
    "red=heavy to very heavy precipitation, magenta=extreme convective activity.\n"
    "- Treat the green outer ring as a caution/buffer region: avoid it with the required clearance "
    "buffer whenever any safe alternative exists.\n"
    "- Yellow, red, and magenta are the terminal convective core; never plan through them."
)


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


@dataclass
class HLTPPlan:
    agent_id: str
    first_waypoint_name: str
    merge_back_waypoint_name: str
    short_context: str
    source: str
    created_step: int
    conflict_partner_ids: Tuple[str, ...]
    rationale: str = ""


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
                "- Do not merge back through another aircraft's path.\n"
                "- If merge-back rows show pair_loss or low min_sep, choose a hold/diverge candidate instead of forcing recovery.\n"
                "- Weather clearance and traffic separation override destination progress.\n"
                "- Once pair_loss_step=None, traffic_buffer_ok=1, weather_entry_step=None, and weather_buffer_ok=1 are satisfied, do not choose the row only because it has the largest min_sep.\n"
                "- Among traffic/weather-clear rows, prioritize recovery in this order: no boundary_exit_step, larger progress_to_destination, larger cross_track_reduction, smaller end_heading_error_to_dest_deg.\n"
                "- Use min_sep as a tie-breaker after the traffic caution buffer is satisfied.\n"
                "- If rows are unsafe only because boundary_exit_step is present, treat that as recovery urgency: choose the candidate with the strongest destination-side recovery, usually the largest allowed turn magnitude.\n"
                "- When heading_error_to_destination_deg is large, do not keep using +/-5 unless larger same-direction bins materially worsen traffic or weather safety.\n"
                "- Prefer larger progress_to_destination.\n"
                "- Prefer reducing cross_track_abs.\n"
                "- Do not choose 0 just because it is safe if a nonzero safe row gives materially better progress, cross-track recovery, or heading alignment.\n"
                "- Use 0 only if it remains aligned and still makes good destination progress.\n"
            )
        else:
            stage_text = f"STAGE: {call_name}.\n"
        return (
            "ROLE: You are a cautious ATCO helper for one aircraft.\n"
            f"{stage_text}"
            f"- SAFE_R = {self.safe_r:.1f} units.\n"
            "- Avoid weather with buffer: stay outside the green ring when possible and never enter yellow/red/magenta.\n"
            f"{WEATHER_SEVERITY_LEGEND}\n"
            f"- Internal traffic caution buffer: prefer min_sep >= {TRAFFIC_CAUTION_R:.1f} units; the simulator still terminates only below SAFE_R={self.safe_r:.1f}.\n"
            "SAFETY SELECTION RULES:\n"
            "- Prefer a preview row with safe=1 over any row with safe=0.\n"
            "- Prefer traffic_buffer_ok=1; do not choose traffic_buffer_ok=0 unless every candidate has traffic_buffer_ok=0.\n"
            "- Prefer weather_buffer_ok=1; do not choose weather_buffer_ok=0 unless every candidate has weather_buffer_ok=0.\n"
            "- If all rows are unsafe due to traffic or weather, choose the least bad row: no weather_entry, no pair_loss, largest min_sep, then largest min_weather_clearance.\n"
            "- If all rows are unsafe only because boundary_exit_step is present, prioritize recovering back inside/toward destination over small weather-clearance differences.\n"
            "TRAFFIC HARD VETO:\n"
            "- Never choose a row with pair_loss_step not None if any displayed row has pair_loss_step=None.\n"
            "- pair_loss_step at step 1 or 2 is an imminent collision and overrides MERGE_BACK, boundary recovery, destination progress, and HLTP.\n"
            "- If choosing between boundary_exit_step and pair_loss_step, choose boundary_exit_step unless every row has pair_loss_step.\n"
            f"- Do not claim a maneuver avoids conflict unless pair_loss_step=None and min_sep>={TRAFFIC_CAUTION_R:.1f}.\n"
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
        cells = weather.get("cells") if isinstance(weather.get("cells"), list) else [weather]
        lines = [
            "LOCAL WEATHER RISK:",
            "- avoid green with buffer when possible; yellow/red/magenta are terminal no-go regions",
            WEATHER_SEVERITY_LEGEND,
            f"- required_clearance_buffer={core.weather_clearance_buffer_units:.2f} simulator units",
            f"- current_min_clearance_across_cells={current_clearance:.2f}",
            (
                f"- predicted_weather_entry_step={state.predicted_weather_entry_step} "
                f"predicted_min_clearance={state.predicted_weather_clearance:.2f}"
            ),
        ]
        for cell in cells:
            lines.append(
                (
                    f"- {cell.get('cell_id', 'W?')}: center=({cell['center'][0]:.1f},{cell['center'][1]:.1f}) "
                    f"major_nm={float(cell['major_radius_nm']):.2f} "
                    f"minor_nm={float(cell['minor_radius_nm']):.2f} "
                    f"orientation_deg={math.degrees(float(cell['angle_rad'])):+.1f} "
                    f"motion_heading_deg={math.degrees(float(cell['motion_heading_rad'])):+.1f} "
                    f"growth_stopped={int(bool(cell.get('growth_stopped', False)))} "
                    f"movement_stopped={int(bool(cell.get('movement_stopped', False)))}"
                )
            )
        return "\n".join(lines)

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
                f"traffic_buffer_ok={int(bool(row.get('traffic_buffer_ok', True)))} | "
                f"min_weather_clearance={float(row['min_weather_clearance']):.2f} | "
                f"weather_buffer_ok={int(bool(row.get('weather_buffer_ok', True)))} | "
                f"progress_to_destination={float(row.get('progress_to_destination', 0.0)):.2f} | "
                f"cross_track_reduction={float(row.get('cross_track_reduction', 0.0)):.2f} | "
                f"end_distance_to_destination={float(row.get('end_distance_to_destination', 999.0)):.2f} | "
                f"end_cross_track_abs={float(row.get('end_cross_track_abs', 999.0)):.2f} | "
                f"end_heading_error_to_dest_deg={float(row['end_heading_error_to_dest_deg']):.1f}"
            )
        return "\n".join(lines)


class HLTPPromptBuilder:
    """Build text-only high-level tactical planning prompts."""

    def build_prompt(
        self,
        core: "MultiAgentSectorCore",
        *,
        step: int,
        contexts: Sequence[Dict[str, Any]],
        frame_paths: Sequence[str],
    ) -> str:
        sections = [
            self._instructions_text(step, contexts),
            self._agent_context_text(contexts),
            self._weather_text(core),
            self._vision_placeholder_text(frame_paths),
        ]
        return "\n\n".join(section for section in sections if section)

    def _instructions_text(self, step: int, contexts: Sequence[Dict[str, Any]]) -> str:
        json_shape = self._strict_json_shape_text(contexts)
        return (
            "ROLE: You are a high-level tactical planner (HLTP) for aircraft that just entered the sector.\n"
            "TASK:\n"
            f"- Look ahead from current step {int(step)} to step 60.\n"
            "- If two aircraft are in conflict, aviation convention says both should turn right.\n"
            "- For each listed aircraft, choose exactly two waypoint names.\n"
            "- first_waypoint_name must be a named waypoint on the right side of the aircraft's original flight plan.\n"
            "- merge_back_waypoint_name must be a later named waypoint on the aircraft's original route.\n"
            "- Keep weather clear of the route with buffer; avoid the green ring when possible and never plan through yellow/red/magenta.\n"
            f"{WEATHER_SEVERITY_LEGEND}\n"
            "- This is advisory high-level context only; low-level safety guidance may override it.\n"
            "OUTPUT RULES:\n"
            "- Return EXACTLY ONE JSON object and no extra text.\n"
            "- Use only waypoint names shown in the candidate lists.\n"
            "- The placeholder strings below are not waypoint names; replace them with candidate names for each listed aircraft.\n"
            "- The plans object must include every aircraft shown in this JSON shape.\n"
            "STRICT JSON SHAPE:\n"
            f"{json_shape}"
        )

    def _strict_json_shape_text(self, contexts: Sequence[Dict[str, Any]]) -> str:
        agent_ids = [str(context["agent_id"]) for context in contexts]
        if not agent_ids:
            agent_ids = ["A1"]
        lines = [
            "{\n"
            "  \"answer\": {\n"
            "    \"plans\": {\n"
        ]
        for idx, agent_id in enumerate(agent_ids):
            comma = "," if idx < len(agent_ids) - 1 else ""
            lines.extend(
                [
                    f"      \"{agent_id}\": {{\n",
                    "        \"first_waypoint_name\": \"<FIRST_WAYPOINT_FROM_CANDIDATES>\",\n",
                    "        \"merge_back_waypoint_name\": \"<MERGE_BACK_WAYPOINT_FROM_CANDIDATES>\",\n",
                    "        \"rationale\": \"<brief>\"\n",
                    f"      }}{comma}\n",
                ]
            )
        lines.extend(
            [
                "    }\n",
                "  }\n",
                "}",
            ]
        )
        return "".join(lines)

    def _agent_context_text(self, contexts: Sequence[Dict[str, Any]]) -> str:
        lines = ["HLTP AIRCRAFT CONTEXT:"]
        if not contexts:
            lines.append("- none")
            return "\n".join(lines)
        for context in contexts:
            agent_id = str(context["agent_id"])
            state = context["state"]
            conflict = context["conflict"]
            lines.append(
                (
                    f"- {agent_id}: pos=({state['position'][0]:.1f},{state['position'][1]:.1f}) "
                    f"heading={state['heading_deg']:+.1f} route={'->'.join(context['route_waypoint_names'])}"
                )
            )
            lines.append(
                (
                    f"  conflict_partners={','.join(context['conflict_partner_ids'])} "
                    f"loss_step={conflict['loss_step']} min_sep={conflict['min_sep']:.2f} "
                    f"conflict_point=({conflict['point'][0]:.1f},{conflict['point'][1]:.1f})"
                )
            )
            lines.append("  right-side first waypoint candidates:")
            for row in context["right_waypoint_candidates"]:
                lines.append(
                    (
                        f"    - {row['name']}: pos=({row['point'][0]:.1f},{row['point'][1]:.1f}) "
                        f"distance_to_conflict={row['distance_to_conflict']:.2f} "
                        f"right_offset={abs(row['side_offset']):.2f}"
                    )
                )
            lines.append("  original-route merge-back candidates:")
            for row in context["merge_back_candidates"]:
                lines.append(
                    (
                        f"    - {row['name']}: pos=({row['point'][0]:.1f},{row['point'][1]:.1f}) "
                        f"route_progress={row['route_progress']:.2f}"
                    )
                )
        return "\n".join(lines)

    def _weather_text(self, core: "MultiAgentSectorCore") -> str:
        weather = core.weather_dict()
        if weather is None:
            return "HLTP WEATHER CONTEXT:\n- none"
        cells = weather.get("cells") if isinstance(weather.get("cells"), list) else [weather]
        lines = [
            "HLTP WEATHER CONTEXT:",
            "- keep routes outside green with buffer when possible; never plan through yellow/red/magenta",
            WEATHER_SEVERITY_LEGEND,
        ]
        lines.append(f"- required_clearance_buffer={core.weather_clearance_buffer_units:.2f} simulator units")
        for cell in cells:
            lines.append(
                (
                    f"- {cell.get('cell_id', 'W?')}: center=({cell['center'][0]:.1f},{cell['center'][1]:.1f}) "
                    f"major_nm={float(cell['major_radius_nm']):.2f} "
                    f"minor_nm={float(cell['minor_radius_nm']):.2f} "
                    f"motion_heading_deg={math.degrees(float(cell['motion_heading_rad'])):+.1f} "
                    f"growth_nm_per_step={float(cell.get('major_growth_nm_per_step', 0.0)):+.2f} "
                    f"growth_stopped={int(bool(cell.get('growth_stopped', False)))} "
                    f"movement_stopped={int(bool(cell.get('movement_stopped', False)))}"
                )
            )
        return "\n".join(lines)

    def _vision_placeholder_text(self, frame_paths: Sequence[str]) -> str:
        lines = ["VISION HLTP CONTEXT:"]
        if not frame_paths:
            lines.append("- none")
        else:
            lines.append("- placeholder only; HLTP is currently text-only and does not inspect frames.")
            for idx, frame_path in enumerate(frame_paths):
                lines.append(f"- frame_{idx}: {os.path.basename(frame_path)}")
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


def _bad_step_sort_value(step: Optional[int]) -> Tuple[int, int]:
    if step is None:
        return (0, 0)
    return (1, -int(step))


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
            "num_ctx": OLLAMA_NUM_CTX,
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
            or (
                state.weather_predicted
                and math.isfinite(state.predicted_weather_clearance)
                and float(state.predicted_weather_clearance) < core.weather_clearance_buffer_units
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

        def ensure_tracked(agent_id: str) -> None:
            if agent_id not in active_ids:
                active_ids.append(agent_id)
            pair_loss_step.setdefault(agent_id, None)
            per_agent_min_sep.setdefault(agent_id, float("inf"))
            min_weather_clearance.setdefault(agent_id, float("inf"))
            weather_entry_step.setdefault(agent_id, None)
            boundary_exit_step.setdefault(agent_id, None)

        for tick in range(1, int(horizon_steps) + 1):
            turns = first_step_turns if tick == 1 else {}
            sim.step(turns, compute_predictions=False)
            cumulative_reward += float(sim.reward)

            for agent_id in sim.active_agent_ids:
                ensure_tracked(agent_id)

            for pair_key, distance in sim.last_pairwise_separations.items():
                agent_a, agent_b = pair_key
                ensure_tracked(agent_a)
                ensure_tracked(agent_b)
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

            for agent_id in list(active_ids):
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
            _bad_step_sort_value(row.get("pair_loss_step")),
            0 if bool(row.get("traffic_buffer_ok", True)) else 1,
            _bad_step_sort_value(row.get("weather_entry_step")),
            0 if bool(row.get("weather_buffer_ok", True)) else 1,
            _bad_step_sort_value(row.get("boundary_exit_step")),
            -float(row.get("min_sep_to_any", 999.0)),
            -float(row.get("min_weather_clearance", 999.0)),
            float(row.get("end_heading_error_to_dest_deg", 999.0)),
            float(row.get("end_cross_track_abs", 999.0)),
            abs(int(row.get("heading_change_deg", 0))),
        )

    def _merge_back_candidate_sort_key(self, row: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            0 if bool(row.get("safe_over_preview", False)) else 1,
            _bad_step_sort_value(row.get("pair_loss_step")),
            0 if bool(row.get("traffic_buffer_ok", True)) else 1,
            _bad_step_sort_value(row.get("weather_entry_step")),
            0 if bool(row.get("weather_buffer_ok", True)) else 1,
            _bad_step_sort_value(row.get("boundary_exit_step")),
            -float(row.get("progress_to_destination", 0.0)),
            -float(row.get("cross_track_reduction", 0.0)),
            float(row.get("end_heading_error_to_dest_deg", 999.0)),
            -float(row.get("min_sep_to_any", 999.0)),
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
                "traffic_buffer_ok": bool(
                    float(sim["per_agent_min_sep"].get(agent_id, 999.0)) >= TRAFFIC_CAUTION_R
                ),
                "weather_buffer_ok": bool(
                    float(sim["min_weather_clearance"].get(agent_id, 999.0))
                    >= core.weather_clearance_buffer_units
                ),
                "safe_over_preview": bool(
                    sim["pair_loss_step"].get(agent_id) is None
                    and sim["weather_entry_step"].get(agent_id) is None
                    and sim["boundary_exit_step"].get(agent_id) is None
                    and float(sim["per_agent_min_sep"].get(agent_id, 999.0)) >= TRAFFIC_CAUTION_R
                    and float(sim["min_weather_clearance"].get(agent_id, 999.0))
                    >= core.weather_clearance_buffer_units
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
                "weather": core.weather_dict(),
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
        hltp_plans: Dict[str, Dict[str, Any]],
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
            self._hltp_plans_text(hltp_plans),
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
            "- The controller already assigned each aircraft's phase; do not classify, rename, or output phases.\n"
            "- Use the simulator-generated candidate previews as the main decision evidence.\n"
            "- Preview rows are sorted by a deterministic heuristic, but you must judge the tradeoffs and may choose any displayed row that best satisfies the safety and phase priorities.\n"
            f"- SAFE_R = {self.safe_r:.1f} units.\n"
            "- Avoid weather with buffer: stay outside the green ring when possible and never enter yellow/red/magenta.\n"
            f"{WEATHER_SEVERITY_LEGEND}\n"
            f"- Internal traffic caution buffer: prefer min_sep >= {TRAFFIC_CAUTION_R:.1f} units; the simulator still terminates only below SAFE_R={self.safe_r:.1f}.\n"
            "SAFETY SELECTION RULES:\n"
            "- For each aircraft, prefer a preview row with safe=1 over any row with safe=0.\n"
            "- Prefer traffic_buffer_ok=1; do not choose traffic_buffer_ok=0 unless every candidate for that aircraft has traffic_buffer_ok=0.\n"
            "- Prefer weather_buffer_ok=1; do not choose weather_buffer_ok=0 unless every candidate for that aircraft has weather_buffer_ok=0.\n"
            "- If all rows are unsafe due to traffic or weather, choose the least bad row: no weather_entry, no pair_loss, largest min_sep, then largest min_weather_clearance.\n"
            "- If all rows are unsafe only because boundary_exit is present, prioritize recovering back inside/toward destination over small weather-clearance differences.\n"
            "- During MERGE_BACK, traffic separation and weather clearance override destination recovery.\n"
            "TRAFFIC HARD VETO:\n"
            "- Never choose a row with pair_loss not None if any displayed row for that aircraft has pair_loss=None.\n"
            "- pair_loss at step 1 or 2 is an imminent collision and overrides MERGE_BACK, boundary recovery, destination progress, and HLTP.\n"
            "- If choosing between boundary_exit and pair_loss, choose boundary_exit unless every row for that aircraft has pair_loss.\n"
            f"- Do not claim a maneuver avoids conflict unless pair_loss=None and min_sep>={TRAFFIC_CAUTION_R:.1f}.\n"
            "OUTPUT RULES:\n"
            "- Return EXACTLY ONE JSON object and no extra text.\n"
            "- Under actions, output only the listed aircraft IDs with rationale and maneuver; do not output call_name.\n"
            f"- heading_change_deg must be one of {{{bins_text}}}.\n"
            "- heading_change_deg should match one of that aircraft's displayed preview row turns.\n"
            "- In the rationale, cite the chosen row's most important consequences, such as no traffic loss, no weather entry, better boundary recovery, or better progress.\n"
            "- Positive is left, negative is right, 0 means hold current heading.\n"
            "STRICT JSON SHAPE:\n"
            "{\n"
            "  \"answer\": {\n"
            "    \"scenario_summary\": \"<brief>\",\n"
            "    \"actions\": {\n"
            "      \"A1\": {\n"
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
            "- Do not merge back through another aircraft's path.\n"
            "- If merge-back rows show pair_loss or low min_sep, choose a hold/diverge candidate instead of forcing recovery.\n"
            "- Weather clearance and traffic separation override destination progress.\n"
            "- Once pair_loss=None, traffic_buffer_ok=1, weather_entry=None, and weather_buffer_ok=1 are satisfied, do not choose the row only because it has the largest min_sep.\n"
            "- Among traffic/weather-clear rows, prioritize recovery in this order: no boundary_exit, larger progress_to_destination, larger cross_track_reduction, smaller end_heading_error.\n"
            "- Use min_sep as a tie-breaker after the traffic caution buffer is satisfied.\n"
            "- If rows are unsafe only because boundary_exit is present, treat that as recovery urgency: choose the candidate with the strongest destination-side recovery, usually the largest allowed turn magnitude.\n"
            "- When heading_err magnitude is large, do not keep using +/-5 unless larger same-direction bins materially worsen traffic or weather safety.\n"
            "- Prefer larger progress_to_destination.\n"
            "- Prefer reducing cross_track_abs.\n"
            "- Do not choose 0 just because it is safe if a nonzero safe row gives materially better progress, cross-track recovery, or heading alignment.\n"
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

    def _hltp_plans_text(self, hltp_plans: Dict[str, Dict[str, Any]]) -> str:
        lines = ["HIGH-LEVEL TACTICAL PLANS:"]
        if not hltp_plans:
            lines.append("- none")
            return "\n".join(lines)
        for agent_id, plan in sorted(hltp_plans.items()):
            context = str(plan.get("short_context", "")).strip()
            if not context:
                continue
            rationale = str(plan.get("rationale", "")).strip()
            if rationale:
                lines.append(
                    f"- {agent_id}: {context}. HLTP rationale: {rationale}. "
                    "Advisory only; current safety logic may override."
                )
            else:
                lines.append(f"- {agent_id}: {context}. Advisory only; current safety logic may override.")
        if len(lines) == 1:
            lines.append("- none")
        return "\n".join(lines)

    def _weather_text(self, core: "MultiAgentSectorCore") -> str:
        weather = core.weather_dict()
        if weather is None:
            return "GLOBAL WEATHER:\n- none"
        cells = weather.get("cells") if isinstance(weather.get("cells"), list) else [weather]
        lines = [
            "GLOBAL WEATHER:",
            "- green is a caution/buffer ring; yellow/red/magenta are terminal no-go regions",
            WEATHER_SEVERITY_LEGEND,
        ]
        lines.append(f"- required_clearance_buffer={core.weather_clearance_buffer_units:.2f} simulator units")
        for cell in cells:
            lines.append(
                (
                    f"- {cell.get('cell_id', 'W?')}: center=({cell['center'][0]:.1f},{cell['center'][1]:.1f}) "
                    f"major_radius_nm={float(cell['major_radius_nm']):.2f} "
                    f"minor_radius_nm={float(cell['minor_radius_nm']):.2f} "
                    f"orientation_deg={math.degrees(float(cell['angle_rad'])):+.1f} "
                    f"motion_heading_deg={math.degrees(float(cell['motion_heading_rad'])):+.1f} "
                    f"speed_units_per_step={float(cell['speed_units_per_step']):.2f} "
                    f"growth_nm_per_step={float(cell.get('major_growth_nm_per_step', 0.0)):+.2f} "
                    f"growth_stopped={int(bool(cell.get('growth_stopped', False)))} "
                    f"movement_stopped={int(bool(cell.get('movement_stopped', False)))}"
                )
            )
        return "\n".join(lines)

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
        lines = [
            "PER-AGENT TURN PREVIEWS:",
            "- each row predicts consequences if that turn is applied now",
            "- safe=1 means no predicted traffic loss, traffic-buffer loss, weather entry, or boundary exit over the preview",
            "- pair_loss/weather_entry/boundary_exit show the first bad step; None means not predicted",
            f"- traffic_buffer_ok=1 means min_sep stayed at or above {TRAFFIC_CAUTION_R:.1f} units",
            "- progress_to_destination and cross_track_reduction: larger is better for recovery",
            "- end_heading_error: smaller is better alignment to destination",
        ]
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
                    f"traffic_buffer_ok={int(bool(row.get('traffic_buffer_ok', True)))} | "
                    f"min_weather_clearance={float(row['min_weather_clearance']):.2f} | "
                    f"weather_buffer_ok={int(bool(row.get('weather_buffer_ok', True)))} | "
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
        lines.extend(
            [
                "- Solid concentric weather rings show each cell's current position.",
                "- Faint dashed green outer ellipses show predicted future weather-cell positions; motion is from the solid cell toward the dashed outlines.",
                "- Keep aircraft clear of both the current solid cell and its dashed future outlines with margin.",
            ]
        )
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
        self.hltp_builder = HLTPPromptBuilder()
        self.retry_limit = int(retry_limit)
        self.recent_decisions: List[Dict[str, Any]] = []
        self.hltp_plans: Dict[str, HLTPPlan] = {}
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
        graph.add_node("hltp_on_sector_entry", self._graph_hltp_on_sector_entry)
        graph.add_node("assign_stages", self._graph_assign_stages)
        graph.add_node("build_global_context", self._graph_build_global_context)
        graph.add_node("call_llm", self._graph_call_llm)
        graph.add_node("parse_validate_retry", self._graph_parse_validate_retry)
        graph.add_node("guardrail_actions", self._graph_guardrail_actions)
        graph.add_node("fallback_missing", self._graph_fallback_missing)
        graph.add_node("commit_and_log", self._graph_commit_and_log)
        graph.set_entry_point("collect_state")
        graph.add_edge("collect_state", "hltp_on_sector_entry")
        graph.add_edge("hltp_on_sector_entry", "assign_stages")
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
        self.hltp_plans = {}

    def _hltp_plan_to_dict(self, plan: HLTPPlan) -> Dict[str, Any]:
        return {
            "agent_id": plan.agent_id,
            "first_waypoint_name": plan.first_waypoint_name,
            "merge_back_waypoint_name": plan.merge_back_waypoint_name,
            "short_context": plan.short_context,
            "source": plan.source,
            "created_step": int(plan.created_step),
            "conflict_partner_ids": list(plan.conflict_partner_ids),
            "rationale": plan.rationale,
        }

    def _active_hltp_plan_dicts(self, agent_ids: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
        allowed = None if agent_ids is None else set(str(agent_id) for agent_id in agent_ids)
        return {
            agent_id: self._hltp_plan_to_dict(plan)
            for agent_id, plan in sorted(self.hltp_plans.items())
            if allowed is None or agent_id in allowed
        }

    def _route_side_offset(
        self,
        line: Any,
        point: Tuple[float, float],
    ) -> Tuple[float, float]:
        route_progress = float(line.project(Point(point)))
        foot = line.interpolate(route_progress)
        route_heading = heading_from_line(line, route_progress)
        tx, ty = math.cos(route_heading), math.sin(route_heading)
        dx, dy = float(point[0]) - float(foot.x), float(point[1]) - float(foot.y)
        side_offset = tx * dy - ty * dx
        return float(side_offset), float(route_progress)

    def _hltp_pair_conflicts(
        self,
        core: "MultiAgentSectorCore",
        *,
        horizon_steps: int,
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        sim = core.clone()
        summaries: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for tick in range(1, int(max(1, horizon_steps)) + 1):
            sim.step({}, compute_predictions=False)
            for pair_key, distance in sim.last_pairwise_separations.items():
                agent_a, agent_b = pair_key
                key = _pair_key(agent_a, agent_b)
                state_a = sim.agent_states.get(agent_a)
                state_b = sim.agent_states.get(agent_b)
                if state_a is None or state_b is None:
                    continue
                midpoint = (
                    0.5 * (float(state_a.position[0]) + float(state_b.position[0])),
                    0.5 * (float(state_a.position[1]) + float(state_b.position[1])),
                )
                summary = summaries.setdefault(
                    key,
                    {
                        "agent_ids": key,
                        "min_sep": float("inf"),
                        "min_step": None,
                        "loss_step": None,
                        "point": midpoint,
                    },
                )
                if float(distance) < float(summary["min_sep"]):
                    summary["min_sep"] = float(distance)
                    summary["min_step"] = tick
                    if summary["loss_step"] is None:
                        summary["point"] = midpoint
                if float(distance) < core.safe_r and summary["loss_step"] is None:
                    summary["loss_step"] = tick
                    summary["point"] = midpoint
            if sim.last_team_done or sim.last_team_truncated:
                break
        return {
            key: value
            for key, value in summaries.items()
            if value.get("loss_step") is not None
        }

    def _best_hltp_conflict_for_agent(
        self,
        conflicts: Dict[Tuple[str, str], Dict[str, Any]],
        agent_id: str,
    ) -> Optional[Dict[str, Any]]:
        candidates = [
            conflict
            for pair, conflict in conflicts.items()
            if str(agent_id) in pair
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                999 if item.get("loss_step") is None else int(item["loss_step"]),
                float(item.get("min_sep", 999.0)),
            ),
        )

    def _hltp_right_waypoint_candidates(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
        conflict_point: Tuple[float, float],
    ) -> List[Dict[str, Any]]:
        state = core.get_agent(agent_id)
        line = state.route.linestring
        current_progress = float(line.project(Point(state.position)))
        conflict_progress = float(line.project(Point(conflict_point)))
        rows: List[Dict[str, Any]] = []
        for name, point in load_waypoint_map().items():
            side_offset, route_progress = self._route_side_offset(line, point)
            if route_progress + 1e-6 < current_progress:
                continue
            if side_offset >= -0.10:
                continue
            distance_to_conflict = float(
                math.hypot(float(point[0]) - conflict_point[0], float(point[1]) - conflict_point[1])
            )
            rows.append(
                {
                    "name": str(name),
                    "point": (float(point[0]), float(point[1])),
                    "route_progress": float(route_progress),
                    "side_offset": float(side_offset),
                    "distance_to_conflict": distance_to_conflict,
                    "progress_delta_to_conflict": abs(float(route_progress) - conflict_progress),
                }
            )
        rows.sort(
            key=lambda row: (
                float(row["distance_to_conflict"]),
                float(row["progress_delta_to_conflict"]),
                abs(float(row["side_offset"])),
                str(row["name"]),
            )
        )
        return rows[:8]

    def _hltp_merge_back_candidates(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
        conflict_point: Tuple[float, float],
    ) -> List[Dict[str, Any]]:
        state = core.get_agent(agent_id)
        line = state.route.linestring
        waypoint_map = load_waypoint_map()
        current_progress = float(line.project(Point(state.position)))
        conflict_progress = float(line.project(Point(conflict_point)))
        minimum_progress = max(current_progress, conflict_progress)
        rows: List[Dict[str, Any]] = []
        for name in state.route.waypoint_names:
            if name not in waypoint_map:
                continue
            point = waypoint_map[name]
            route_progress = float(line.project(Point(point)))
            if route_progress + 1e-6 < minimum_progress:
                continue
            rows.append(
                {
                    "name": str(name),
                    "point": (float(point[0]), float(point[1])),
                    "route_progress": float(route_progress),
                }
            )
        if not rows:
            for name in reversed(state.route.waypoint_names):
                if name not in waypoint_map:
                    continue
                point = waypoint_map[name]
                route_progress = float(line.project(Point(point)))
                if route_progress + 1e-6 >= current_progress:
                    rows.append(
                        {
                            "name": str(name),
                            "point": (float(point[0]), float(point[1])),
                            "route_progress": float(route_progress),
                        }
                    )
                    break
        rows.sort(key=lambda row: (float(row["route_progress"]), str(row["name"])))
        return rows[:6]

    def _hltp_context_for_agent(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
        conflict: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        state = core.get_agent(agent_id)
        conflict_point = tuple(conflict["point"])
        right_candidates = self._hltp_right_waypoint_candidates(core, agent_id, conflict_point)
        merge_candidates = self._hltp_merge_back_candidates(core, agent_id, conflict_point)
        if not right_candidates or not merge_candidates:
            return None
        partner_ids = tuple(
            other_id
            for other_id in conflict["agent_ids"]
            if str(other_id) != str(agent_id)
        )
        return {
            "agent_id": str(agent_id),
            "state": {
                "position": tuple(float(value) for value in state.position),
                "heading_deg": float(math.degrees(state.heading_rad)),
            },
            "route_waypoint_names": list(state.route.waypoint_names),
            "conflict_partner_ids": list(partner_ids),
            "conflict": {
                "agent_ids": list(conflict["agent_ids"]),
                "loss_step": conflict.get("loss_step"),
                "min_sep": float(conflict.get("min_sep", 999.0)),
                "point": tuple(float(value) for value in conflict_point),
            },
            "right_waypoint_candidates": right_candidates,
            "merge_back_candidates": merge_candidates,
        }

    def _hltp_contexts_for_step(
        self,
        core: "MultiAgentSectorCore",
        *,
        step: int,
        newly_entered: Optional[Sequence[str]] = None,
        conflicts: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        newly_entered = list(
            self._newly_entered_hltp_agent_ids(core, step)
            if newly_entered is None
            else newly_entered
        )
        if not newly_entered:
            return {}

        if conflicts is None:
            conflicts = self._hltp_pair_conflicts(core, horizon_steps=max(1, 60 - int(step)))
        selected: Dict[str, Dict[str, Any]] = {}
        for agent_id in newly_entered:
            conflict = self._best_hltp_conflict_for_agent(conflicts, agent_id)
            if conflict is None:
                continue
            selected[agent_id] = conflict
            for partner_id in conflict["agent_ids"]:
                partner_id = str(partner_id)
                if partner_id == agent_id or partner_id in self.hltp_plans or partner_id in selected:
                    continue
                partner_state = core.get_agent(partner_id)
                if partner_state.launched and partner_state.has_entered_sector and not partner_state.finished:
                    selected[partner_id] = conflict

        contexts: Dict[str, Dict[str, Any]] = {}
        for agent_id, conflict in selected.items():
            context = self._hltp_context_for_agent(core, agent_id, conflict)
            if context is not None:
                contexts[agent_id] = context
        return contexts

    def _newly_entered_hltp_agent_ids(
        self,
        core: "MultiAgentSectorCore",
        step: int,
    ) -> List[str]:
        return [
            agent_id
            for agent_id in core.active_agent_ids
            if core.get_agent(agent_id).sector_entry_step is not None
            and int(core.get_agent(agent_id).sector_entry_step) == int(step)
            and agent_id not in self.hltp_plans
        ]

    def _fallback_hltp_plan(
        self,
        *,
        context: Dict[str, Any],
        step: int,
        source: str = "fallback",
        rationale: str = "deterministic right-side waypoint fallback",
    ) -> HLTPPlan:
        first = context["right_waypoint_candidates"][0]
        first_progress = float(first["route_progress"])
        merge_candidates = [
            row
            for row in context["merge_back_candidates"]
            if float(row["route_progress"]) + 1e-6 >= first_progress
        ]
        merge = merge_candidates[0] if merge_candidates else context["merge_back_candidates"][0]
        short_context = f"HLTP {first['name']}->{merge['name']}"
        return HLTPPlan(
            agent_id=str(context["agent_id"]),
            first_waypoint_name=str(first["name"]),
            merge_back_waypoint_name=str(merge["name"]),
            short_context=short_context,
            source=source,
            created_step=int(step),
            conflict_partner_ids=tuple(str(value) for value in context["conflict_partner_ids"]),
            rationale=rationale,
        )

    def _hltp_plan_from_payload(
        self,
        *,
        agent_id: str,
        payload: Dict[str, Any],
        context: Dict[str, Any],
        step: int,
    ) -> Optional[HLTPPlan]:
        first_name = str(payload.get("first_waypoint_name", "")).strip()
        merge_name = str(payload.get("merge_back_waypoint_name", "")).strip()
        right_names = {str(row["name"]) for row in context["right_waypoint_candidates"]}
        merge_names = {str(row["name"]) for row in context["merge_back_candidates"]}
        if first_name not in right_names or merge_name not in merge_names:
            return None
        rationale = payload.get("rationale", "")
        short_context = f"HLTP {first_name}->{merge_name}"
        return HLTPPlan(
            agent_id=str(agent_id),
            first_waypoint_name=first_name,
            merge_back_waypoint_name=merge_name,
            short_context=short_context,
            source="llm",
            created_step=int(step),
            conflict_partner_ids=tuple(str(value) for value in context["conflict_partner_ids"]),
            rationale=rationale if isinstance(rationale, str) else "",
        )

    def _extract_hltp_plans_from_payload(
        self,
        payload: Dict[str, Any],
        expected_agent_ids: Sequence[str],
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        answer = payload.get("answer") if isinstance(payload, dict) else {}
        answer = answer if isinstance(answer, dict) else {}
        plans = answer.get("plans") if isinstance(answer.get("plans"), dict) else {}
        found: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        for agent_id in expected_agent_ids:
            plan = plans.get(agent_id)
            if isinstance(plan, dict):
                found[agent_id] = plan
            else:
                missing.append(agent_id)
        return found, missing

    def _apply_hltp_annotations(
        self,
        core: "MultiAgentSectorCore",
        *,
        step: int,
        annotations: Sequence[str],
        plans: Dict[str, HLTPPlan],
    ) -> List[str]:
        updated = list(annotations)
        for agent_id, plan in plans.items():
            state = core.get_agent(agent_id)
            if state.sector_entry_step is None or int(state.sector_entry_step) != int(step):
                continue
            plain = f"{agent_id}: ENTERED SECTOR"
            with_plan = f"{agent_id}: ENTERED SECTOR: {plan.short_context}"
            replaced = False
            for idx, text in enumerate(updated):
                if text == plain:
                    updated[idx] = with_plan
                    replaced = True
                    break
            if not replaced:
                updated.append(with_plan)
        return updated

    def _apply_no_hltp_conflict_annotations(
        self,
        core: "MultiAgentSectorCore",
        *,
        step: int,
        annotations: Sequence[str],
        agent_ids: Sequence[str],
    ) -> List[str]:
        updated = list(annotations)
        for agent_id in agent_ids:
            state = core.get_agent(agent_id)
            if state.sector_entry_step is None or int(state.sector_entry_step) != int(step):
                continue
            plain = f"{agent_id}: ENTERED SECTOR"
            with_reason = f"{agent_id}: ENTERED SECTOR: HLTP NO CONFLICT DETECTED"
            replaced = False
            for idx, text in enumerate(updated):
                if text == plain:
                    updated[idx] = with_reason
                    replaced = True
                    break
            if not replaced and not any(text.startswith(f"{agent_id}: ENTERED SECTOR:") for text in updated):
                updated.append(with_reason)
        return updated

    def _log_hltp_call(
        self,
        *,
        step: int,
        prompt_text: str,
        raw_text: str,
        normalized: Dict[str, Any],
        contexts_by_agent: Dict[str, Dict[str, Any]],
        debug: Dict[str, Any],
    ) -> None:
        _ensure_dir(self.save_dir)
        tag = f"t{step:03d}_hltp"
        prompt_path = os.path.join(self.save_dir, f"{tag}.prompt.txt")
        response_path = os.path.join(self.save_dir, f"{tag}.response.txt")
        normalized_path = os.path.join(self.save_dir, f"{tag}.normalized.json")
        _save_text(prompt_path, prompt_text)
        _save_text(response_path, raw_text)
        payload = copy.deepcopy(normalized)
        payload["contexts_by_agent"] = contexts_by_agent
        payload["debug"] = debug
        _save_json(normalized_path, payload)
        with open(os.path.join(self.save_dir, "_index.jsonl"), "a", encoding="utf-8") as handle:
            row = {
                "tag": tag,
                "step": step,
                "run_id": self.run_id,
                "episode_id": self.episode_id,
                "agent_id": "HLTP",
                "call_name": "HLTP",
                "llm_status": debug.get("llm_status"),
                "parse_status": debug.get("parse_status"),
                "used_vision": False,
            }
            handle.write(json.dumps(row) + "\n")

    def _graph_hltp_on_sector_entry(self, state: Dict[str, Any]) -> Dict[str, Any]:
        core = state["core"]
        step = int(state["step"])
        newly_entered = self._newly_entered_hltp_agent_ids(core, step)
        conflicts = (
            self._hltp_pair_conflicts(core, horizon_steps=max(1, 60 - step))
            if newly_entered
            else {}
        )
        contexts_by_agent = self._hltp_contexts_for_step(
            core,
            step=step,
            newly_entered=newly_entered,
            conflicts=conflicts,
        )
        no_conflict_agent_ids = [
            agent_id
            for agent_id in newly_entered
            if self._best_hltp_conflict_for_agent(conflicts, agent_id) is None
        ]
        annotations = self._apply_no_hltp_conflict_annotations(
            core,
            step=step,
            annotations=list(state.get("annotations", [])),
            agent_ids=no_conflict_agent_ids,
        )
        if not contexts_by_agent:
            return {
                **state,
                "annotations": annotations,
                "hltp_plans_created": {},
                "hltp_no_conflict_agent_ids": no_conflict_agent_ids,
            }

        prompt_text = self.hltp_builder.build_prompt(
            core,
            step=step,
            contexts=list(contexts_by_agent.values()),
            frame_paths=list(state.get("frame_paths", [])),
        )
        llm_result = ollama_invoke(prompt_text, image_paths=None)
        expected_agent_ids = sorted(contexts_by_agent)
        parsed_payload: Dict[str, Any] = {}
        candidate_plans: Dict[str, Dict[str, Any]] = {}
        missing_agent_ids = list(expected_agent_ids)
        parse_status = "not_attempted"
        fallback_agents: List[str] = []

        if llm_result.get("llm_status") == "ok":
            try:
                parsed_payload = _extract_json(str(llm_result.get("raw_text", "")))
                candidate_plans, missing_agent_ids = self._extract_hltp_plans_from_payload(
                    parsed_payload,
                    expected_agent_ids,
                )
                parse_status = "ok" if not missing_agent_ids else "missing_plans"
            except Exception:
                parsed_payload = {}
                candidate_plans = {}
                missing_agent_ids = list(expected_agent_ids)
                parse_status = "invalid_json"

        final_plans: Dict[str, HLTPPlan] = {}
        invalid_agent_ids: List[str] = []
        invalid_rationales: Dict[str, str] = {}
        for agent_id, plan_payload in candidate_plans.items():
            plan = self._hltp_plan_from_payload(
                agent_id=agent_id,
                payload=plan_payload,
                context=contexts_by_agent[agent_id],
                step=step,
            )
            if plan is None:
                invalid_agent_ids.append(agent_id)
                rationale = plan_payload.get("rationale", "")
                if isinstance(rationale, str) and rationale.strip():
                    invalid_rationales[agent_id] = rationale.strip()
                continue
            final_plans[agent_id] = plan

        for agent_id in sorted(set(missing_agent_ids).union(invalid_agent_ids)):
            fallback_rationale = invalid_rationales.get(agent_id, "")
            if fallback_rationale:
                fallback_rationale = (
                    "LLM rationale from rejected waypoint proposal: "
                    f"{fallback_rationale}"
                )
            final_plans[agent_id] = self._fallback_hltp_plan(
                context=contexts_by_agent[agent_id],
                step=step,
                rationale=fallback_rationale or "deterministic right-side waypoint fallback",
            )
            fallback_agents.append(agent_id)

        for agent_id, plan in final_plans.items():
            self.hltp_plans[agent_id] = plan

        normalized = {
            "answer": {
                "plans": {
                    agent_id: self._hltp_plan_to_dict(plan)
                    for agent_id, plan in sorted(final_plans.items())
                }
            }
        }
        debug = {
            "llm_status": llm_result.get("llm_status"),
            "llm_error": llm_result.get("error", ""),
            "parse_status": parse_status,
            "fallback_agents": fallback_agents,
            "eligible_agent_ids": expected_agent_ids,
            "used_vision": False,
            "frame_paths": [],
            "model": OLLAMA_MODEL,
            "weather": core.weather_dict(),
        }
        self._log_hltp_call(
            step=step,
            prompt_text=prompt_text,
            raw_text=str(llm_result.get("raw_text", "")),
            normalized=normalized,
            contexts_by_agent=contexts_by_agent,
            debug=debug,
        )
        annotations = self._apply_hltp_annotations(
            core,
            step=step,
            annotations=annotations,
            plans=final_plans,
        )
        return {
            **state,
            "annotations": annotations,
            "hltp_plans_created": normalized["answer"]["plans"],
            "hltp_no_conflict_agent_ids": no_conflict_agent_ids,
        }

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
            or (
                state.weather_predicted
                and math.isfinite(state.predicted_weather_clearance)
                and float(state.predicted_weather_clearance) < core.weather_clearance_buffer_units
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
            hltp_plans=self._active_hltp_plan_dicts(state.get("entered_agent_ids", [])),
            threat_rows_by_agent=threat_rows_by_agent,
            preview_rows_by_agent=preview_rows_by_agent,
            recent_decisions=self.recent_decisions,
            frame_paths=list(state.get("frame_paths", [])),
        )
        return {
            **state,
            "threat_rows_by_agent": threat_rows_by_agent,
            "preview_rows_by_agent": preview_rows_by_agent,
            "hltp_plans": self._active_hltp_plan_dicts(state.get("entered_agent_ids", [])),
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
        hltp_plans: Dict[str, Dict[str, Any]],
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
        payload["hltp_plans"] = hltp_plans
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
        core = state["core"]
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
            "hltp_plans": state.get("hltp_plans", {}),
            "weather": core.weather_dict(),
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
            hltp_plans=state.get("hltp_plans", {}),
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
