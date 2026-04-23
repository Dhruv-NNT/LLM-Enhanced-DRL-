"""Centralized multi-agent three-call controller and prompt helpers."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from configs import (
    ACTION_BINS,
    BOUNDARY_LOOKAHEAD_STEPS,
    CONFLICT_LOOKAHEAD_STEPS,
    EVAL_MEMORY_PATH,
    JSON_ANSWERS_DIR,
    MERGE_BACK_REPEAT_ERROR_DEG,
    OLLAMA_HOST,
    OLLAMA_MAX_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_TOP_P,
    SAFE_R,
    SAFE_STREAK_REQUIRED,
    TURN_PREVIEW_STEPS,
)
from .utils import deg_to_action_idx

if TYPE_CHECKING:  # pragma: no cover
    from .core import AgentState, MultiAgentSectorCore
    from .ppo import SharedActorCentralCriticPPO


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AgentControllerState:
    stage: str = "FOLLOW_ROUTE"
    execute_called: bool = False
    emergency_count: int = 0
    merge_back_count: int = 0
    last_advice_turn: int = 0
    advice_active: bool = False
    last_call_name: Optional[str] = None


class MultiAgentPromptBuilder:
    """Build text-first prompts for the centralized multi-agent controller."""

    def __init__(self, safe_r: float = SAFE_R) -> None:
        self.safe_r = float(safe_r)

    def build_prompt(
        self,
        core: "MultiAgentSectorCore",
        actionable: List[Dict[str, Any]],
        preview_rows: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        sections = [self._instructions_text()]
        sections.append(self._global_scene_text(core))
        for item in actionable:
            sections.append(
                self._per_agent_text(
                    core,
                    item["agent_id"],
                    item["call_name"],
                    preview_rows.get(item["agent_id"], []),
                )
            )
        return "\n\n".join(section for section in sections if section)

    def _instructions_text(self) -> str:
        bins_text = ", ".join(str(value) for value in ACTION_BINS)
        return (
            "ROLE: You are a cautious centralized ATCO helper for a multi-agent sector.\n"
            "GOAL: Guide every aircraft safely to its destination while maintaining pairwise separation.\n"
            "SAFETY: SAFE_R = 5.0 units for all aircraft pairs.\n"
            "TASK:\n"
            "- Only return decisions for the actionable agents listed below.\n"
            "- Use the preview table and controller state for each actionable agent.\n"
            "- Favor the smallest safe turn that improves safety and/or recovery.\n"
            "- EXECUTE_TURN and EMERGENCY_MANEUVER are conflict-first.\n"
            "- MERGE_BACK must bias toward the destination side.\n"
            "OUTPUT RULES:\n"
            "- Return EXACTLY ONE JSON object.\n"
            f"- heading_change_deg must be one of {{{bins_text}}}.\n"
            "- Positive means left, negative means right, 0 means keep heading.\n"
            "STRICT JSON SHAPE:\n"
            "{\n"
            "  \"answer\": {\n"
            "    \"global_summary\": \"<1-3 short sentences>\",\n"
            "    \"agents\": [\n"
            "      {\n"
            "        \"agent_id\": \"A1\",\n"
            "        \"call_name\": \"EXECUTE_TURN|EMERGENCY_MANEUVER|MERGE_BACK\",\n"
            "        \"heading_change_deg\": -30..30 step 5,\n"
            "        \"scenario_summary\": \"<brief>\",\n"
            "        \"technical_summary\": \"<brief technical explanation>\",\n"
            "        \"rationale\": \"<why this helps>\"\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}"
        )

    def _global_scene_text(self, core: "MultiAgentSectorCore") -> str:
        lines = ["GLOBAL SCENE:"]
        lines.append(f"- step_index: {core.n_step + 1}")
        lines.append(f"- active_agents: {core.active_agent_ids}")
        for agent_id in core.active_agent_ids:
            state = core.get_agent(agent_id)
            lines.append(
                f"- {agent_id}: pos=({state.position[0]:.1f},{state.position[1]:.1f}) "
                f"dest=({state.destination[0]:.1f},{state.destination[1]:.1f}) "
                f"heading={math.degrees(state.heading_rad):+.1f}° "
                f"inside_sector={int(state.inside_sector)} "
                f"conflict_predicted={int(state.conflict_predicted)}"
            )
        risky_pairs = [
            f"{a}<->{b}:{distance:.2f}"
            for (a, b), distance in core.last_pairwise_separations.items()
            if distance < 2.0 * core.safe_r
        ]
        lines.append(f"- risky_pairs: {risky_pairs or ['none']}")
        return "\n".join(lines)

    def _per_agent_text(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
        call_name: str,
        preview_rows: List[Dict[str, Any]],
    ) -> str:
        state = core.get_agent(agent_id)
        dist_dest = core.distance_to_destination(state)
        route_heading = math.degrees(
            math.atan2(
                state.destination[1] - state.position[1],
                state.destination[0] - state.position[0],
            )
        )
        heading_error_to_dest = _wrap_deg(route_heading - math.degrees(state.heading_rad))
        boundary_distance = core.boundary_distance(state)
        nearest_id, nearest_dist = core.nearest_neighbor(agent_id)
        nearest_bearing = core._bearing_to_neighbor_deg(agent_id, nearest_id)
        lines = [f"ACTIONABLE AGENT {agent_id} ({call_name}):"]
        lines.append(f"- position: ({state.position[0]:.1f}, {state.position[1]:.1f})")
        lines.append(f"- destination: ({state.destination[0]:.1f}, {state.destination[1]:.1f})")
        lines.append(f"- distance_to_destination: {dist_dest:.2f}")
        lines.append(f"- heading_deg: {math.degrees(state.heading_rad):+.1f}")
        lines.append(f"- heading_error_to_destination_deg: {heading_error_to_dest:+.1f}")
        lines.append(
            f"- nearest_neighbor: {nearest_id or 'none'} "
            f"(distance={0.0 if nearest_id is None else nearest_dist:.2f}, "
            f"bearing_error_deg={nearest_bearing:+.1f})"
        )
        lines.append(f"- boundary_distance: {'n/a' if boundary_distance is None else f'{boundary_distance:.2f}'}")
        lines.append(f"- safe_streak: {state.safe_streak}")
        lines.append(f"- boundary_warning_active: {int(state.boundary_warning_active)}")
        lines.append(f"- conflict_predicted: {int(state.conflict_predicted)}")
        lines.append("- candidate_turn_preview:")
        for row in preview_rows:
            lines.append(
                "  "
                f"turn={int(row['heading_change_deg']):+d} | "
                f"safe={int(bool(row['safe_over_preview']))} | "
                f"min_sep_any={float(row['min_sep_any']):.2f} | "
                f"boundary_exit_step={row['boundary_exit_step']} | "
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


def _wrap_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def start_llm_log_dir(base_dir: str = str(JSON_ANSWERS_DIR)) -> Tuple[str, str]:
    _ensure_dir(base_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"run_{stamp}"
    run_dir = os.path.join(base_dir, run_id)
    _ensure_dir(run_dir)
    return run_dir, run_id


def ollama_invoke(prompt_text: str) -> str:
    base_url = os.environ.get("OLLAMA_HOST", OLLAMA_HOST).rstrip("/")
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt_text}],
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
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    parsed = json.loads(raw_body)
    content = (parsed.get("message") or {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama response did not contain message.content")
    return str(content)


class MultiAgentThreeCallController:
    """Centralized controller that manages per-agent three-call semantics."""

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
        self.last_joint_actions: Dict[str, int] = {}

    def reset(self) -> None:
        self.agent_states = {}
        self.last_prompt_text = None
        self.last_raw_text = None
        self.last_normalized = None
        self.last_tag = None
        self.last_joint_actions = {}

    def _controller_state(self, agent_id: str) -> AgentControllerState:
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = AgentControllerState()
        return self.agent_states[agent_id]

    def _heading_error_to_destination(self, state: "AgentState") -> float:
        dest_heading = math.degrees(
            math.atan2(
                state.destination[1] - state.position[1],
                state.destination[0] - state.position[0],
            )
        )
        return _wrap_deg(dest_heading - math.degrees(state.heading_rad))

    def _call_name_for_agent(self, core: "MultiAgentSectorCore", agent_id: str) -> Optional[str]:
        state = core.get_agent(agent_id)
        ctrl = self._controller_state(agent_id)
        if state.finished or not state.launched:
            ctrl.advice_active = False
            return None
        if not state.has_entered_sector:
            ctrl.stage = "FOLLOW_ROUTE"
            return None
        if not ctrl.execute_called:
            return "EXECUTE_TURN"
        if state.conflict_predicted:
            return "EMERGENCY_MANEUVER"
        heading_err = abs(self._heading_error_to_destination(state))
        if state.boundary_warning_active or state.safe_streak >= SAFE_STREAK_REQUIRED:
            return "MERGE_BACK"
        if ctrl.stage == "MERGE_BACK_RECHECK" and heading_err > MERGE_BACK_REPEAT_ERROR_DEG:
            return "MERGE_BACK"
        if ctrl.stage == "MERGE_BACK_RECHECK" and heading_err <= MERGE_BACK_REPEAT_ERROR_DEG:
            ctrl.stage = "WAIT_FOR_SAFE"
        return None

    def _allowed_turns(self, core: "MultiAgentSectorCore", agent_id: str, call_name: str) -> List[int]:
        state = core.get_agent(agent_id)
        if call_name == "MERGE_BACK":
            heading_err = self._heading_error_to_destination(state)
            if abs(heading_err) <= 2.5:
                return [0]
            if heading_err > 0:
                return [value for value in ACTION_BINS if value >= 0]
            return [value for value in ACTION_BINS if value <= 0]

        neighbor_id, _ = core.nearest_neighbor(agent_id)
        bearing = core._bearing_to_neighbor_deg(agent_id, neighbor_id)
        if bearing > 0:
            bins = [value for value in ACTION_BINS if value >= 0]
        elif bearing < 0:
            bins = [value for value in ACTION_BINS if value <= 0]
        else:
            bins = [value for value in ACTION_BINS if value != 0]
        if call_name == "EMERGENCY_MANEUVER":
            bins = [value for value in bins if value != 0]
        return bins or [0]

    def _simulate_joint_horizon(
        self,
        core: "MultiAgentSectorCore",
        first_step_turns: Dict[str, int],
        *,
        horizon_steps: int,
    ) -> Dict[str, Any]:
        sim = core.clone()
        cumulative_reward = 0.0
        min_sep_any = float("inf")
        boundary_exit_step: Dict[str, Optional[int]] = {agent_id: None for agent_id in first_step_turns}
        for tick in range(int(horizon_steps)):
            turns = first_step_turns if tick == 0 else {}
            sim.step(turns)
            cumulative_reward += float(sim.reward)
            if sim.last_pairwise_separations:
                min_sep_any = min(min_sep_any, min(sim.last_pairwise_separations.values()))
            for agent_id in first_step_turns:
                state = sim.agent_states.get(agent_id)
                if state is None:
                    continue
                if state.has_entered_sector and not state.inside_sector and boundary_exit_step[agent_id] is None:
                    boundary_exit_step[agent_id] = tick + 1
            if sim.last_team_done or sim.last_team_truncated:
                break

        end_heading_errors = {
            agent_id: abs(self._heading_error_to_destination(sim.get_agent(agent_id)))
            for agent_id in first_step_turns
            if agent_id in sim.agent_states
        }
        if min_sep_any == float("inf"):
            min_sep_any = 999.0
        return {
            "min_sep_any": float(min_sep_any),
            "safe_over_preview": float(min_sep_any) >= core.safe_r,
            "boundary_exit_step": boundary_exit_step,
            "team_done": bool(sim.last_team_done),
            "team_truncated": bool(sim.last_team_truncated),
            "cumulative_reward": float(cumulative_reward),
            "end_heading_error_to_dest_deg": end_heading_errors,
        }

    def _preview_rows(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
        call_name: str,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for turn_deg in self._allowed_turns(core, agent_id, call_name):
            sim = self._simulate_joint_horizon(
                core,
                {agent_id: int(turn_deg)},
                horizon_steps=TURN_PREVIEW_STEPS,
            )
            rows.append(
                {
                    "heading_change_deg": int(turn_deg),
                    "safe_over_preview": bool(sim["safe_over_preview"]),
                    "min_sep_any": float(sim["min_sep_any"]),
                    "boundary_exit_step": sim["boundary_exit_step"].get(agent_id),
                    "end_heading_error_to_dest_deg": float(
                        sim["end_heading_error_to_dest_deg"].get(agent_id, 0.0)
                    ),
                    "cumulative_reward": float(sim["cumulative_reward"]),
                }
            )
        return rows

    def _deterministic_best_turn(
        self,
        core: "MultiAgentSectorCore",
        agent_id: str,
        call_name: str,
        fixed_actions: Optional[Dict[str, int]] = None,
    ) -> int:
        best_turn = 0
        best_key = None
        fixed_actions = {} if fixed_actions is None else dict(fixed_actions)
        for turn_deg in self._allowed_turns(core, agent_id, call_name):
            actions = dict(fixed_actions)
            actions[agent_id] = int(turn_deg)
            sim = self._simulate_joint_horizon(core, actions, horizon_steps=CONFLICT_LOOKAHEAD_STEPS)
            key = (
                0 if sim["safe_over_preview"] else 1,
                0 if sim["boundary_exit_step"].get(agent_id) is None else 1,
                -float(sim["min_sep_any"]),
                float(sim["end_heading_error_to_dest_deg"].get(agent_id, 0.0)),
                abs(int(turn_deg)),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_turn = int(turn_deg)
        return best_turn

    def _fallback_answer(
        self,
        core: "MultiAgentSectorCore",
        actionable: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        agents = []
        for item in actionable:
            best_turn = self._deterministic_best_turn(core, item["agent_id"], item["call_name"])
            agents.append(
                {
                    "agent_id": item["agent_id"],
                    "call_name": item["call_name"],
                    "heading_change_deg": int(best_turn),
                    "scenario_summary": "",
                    "technical_summary": "",
                    "rationale": "fallback deterministic choice",
                }
            )
        return {"answer": {"global_summary": "", "agents": agents}}

    def _normalize_answer(
        self,
        payload: Dict[str, Any],
        actionable: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        expected = {item["agent_id"]: item["call_name"] for item in actionable}
        answer = payload.get("answer") or {}
        global_summary = answer.get("global_summary")
        if not isinstance(global_summary, str):
            global_summary = ""
        raw_agents = answer.get("agents")
        if not isinstance(raw_agents, list):
            raw_agents = []
        normalized_agents: List[Dict[str, Any]] = []
        indexed = {
            str(item.get("agent_id")): item
            for item in raw_agents
            if isinstance(item, dict) and str(item.get("agent_id")) in expected
        }
        for agent_id, call_name in expected.items():
            item = indexed.get(agent_id, {})
            try:
                turn_deg = float(item.get("heading_change_deg", 0))
            except Exception:
                turn_deg = 0.0
            normalized_agents.append(
                {
                    "agent_id": agent_id,
                    "call_name": call_name,
                    "heading_change_deg": int(_snap_to_allowed_bin(turn_deg)),
                    "scenario_summary": item.get("scenario_summary", "")
                    if isinstance(item.get("scenario_summary"), str)
                    else "",
                    "technical_summary": item.get("technical_summary", "")
                    if isinstance(item.get("technical_summary"), str)
                    else "",
                    "rationale": item.get("rationale", "")
                    if isinstance(item.get("rationale"), str)
                    else "",
                }
            )
        return {"answer": {"global_summary": global_summary, "agents": normalized_agents}}

    def _repair_joint_actions(
        self,
        core: "MultiAgentSectorCore",
        normalized: Dict[str, Any],
        actionable: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        agents = normalized["answer"]["agents"]
        actions = {item["agent_id"]: int(item["heading_change_deg"]) for item in agents}
        sim = self._simulate_joint_horizon(core, actions, horizon_steps=CONFLICT_LOOKAHEAD_STEPS)
        if sim["safe_over_preview"] and all(
            step is None for step in sim["boundary_exit_step"].values()
        ):
            return normalized

        order = sorted(
            actionable,
            key=lambda item: (
                0 if core.get_agent(item["agent_id"]).conflict_predicted else 1,
                0 if core.get_agent(item["agent_id"]).boundary_warning_active else 1,
            ),
        )
        for item in order:
            fixed = {agent_id: turn for agent_id, turn in actions.items() if agent_id != item["agent_id"]}
            actions[item["agent_id"]] = self._deterministic_best_turn(
                core,
                item["agent_id"],
                item["call_name"],
                fixed_actions=fixed,
            )
        for item in agents:
            item["heading_change_deg"] = int(actions[item["agent_id"]])
        return normalized

    def _advice_embedding(
        self,
        turn_deg: int,
        call_name: Optional[str],
        boundary_warning_active: bool,
        conflict_predicted: bool,
        advice_active: bool,
    ) -> np.ndarray:
        return np.array(
            [
                float(turn_deg) / 30.0,
                1.0 if call_name == "EXECUTE_TURN" else 0.0,
                1.0 if call_name == "EMERGENCY_MANEUVER" else 0.0,
                1.0 if call_name == "MERGE_BACK" else 0.0,
                1.0 if boundary_warning_active else 0.0,
                1.0 if conflict_predicted else 0.0,
                1.0 if advice_active else 0.0,
            ],
            dtype=np.float32,
        )

    def _ghost_compare_joint_horizon(
        self,
        core: "MultiAgentSectorCore",
        ppo: Optional["SharedActorCentralCriticPPO"],
        *,
        llm_turns: Dict[str, int],
        advice_embeddings: Dict[str, np.ndarray],
        advice_active_masks: Dict[str, float],
        horizon_steps: int = TURN_PREVIEW_STEPS,
    ) -> Optional[Dict[str, Any]]:
        if ppo is None or not llm_turns:
            return None
        obs_dict = core.get_active_observations()
        global_state = core.global_state_vector()
        try:
            ppo_actions_idx = ppo.greedy_actions(
                obs_dict,
                global_state,
                advice_embeddings=advice_embeddings,
                advice_active_masks=advice_active_masks,
            )
            ppo_turns = {
                agent_id: int(ACTION_BINS[action_idx])
                for agent_id, action_idx in ppo_actions_idx.items()
                if agent_id in llm_turns
            }
            llm_result = self._simulate_joint_horizon(core, llm_turns, horizon_steps=horizon_steps)
            ppo_result = self._simulate_joint_horizon(core, ppo_turns, horizon_steps=horizon_steps)
            return {
                "llm_turns": llm_turns,
                "ppo_turns": ppo_turns,
                "horizon_steps": horizon_steps,
                "llm_reward": llm_result["cumulative_reward"],
                "ppo_reward": ppo_result["cumulative_reward"],
                "llm_min_sep_any": llm_result["min_sep_any"],
                "ppo_min_sep_any": ppo_result["min_sep_any"],
            }
        except Exception:
            traceback.print_exc()
            return None

    def _log_joint_call(
        self,
        *,
        step: int,
        prompt_text: str,
        raw_text: str,
        normalized: Dict[str, Any],
        preview_rows: Dict[str, List[Dict[str, Any]]],
        ghost_compare: Optional[Dict[str, Any]],
    ) -> None:
        _ensure_dir(self.save_dir)
        tag = f"t{step:03d}_joint_guidance"
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
        payload["preview_rows"] = preview_rows
        payload["ghost_compare"] = ghost_compare
        _save_json(normalized_path, payload)
        with open(os.path.join(self.save_dir, "_index.jsonl"), "a", encoding="utf-8") as handle:
            row = {
                "tag": tag,
                "step": step,
                "run_id": self.run_id,
                "episode_id": self.episode_id,
                "n_agents": len(normalized["answer"]["agents"]),
                "agent_ids": [item["agent_id"] for item in normalized["answer"]["agents"]],
            }
            handle.write(json.dumps(row) + "\n")

    def update(
        self,
        core: "MultiAgentSectorCore",
        *,
        step: int,
        ppo: Optional["SharedActorCentralCriticPPO"] = None,
    ) -> Dict[str, int]:
        for agent_id in list(self.agent_states):
            if agent_id not in core.agent_states or core.get_agent(agent_id).finished:
                self.agent_states[agent_id].advice_active = False

        actionable: List[Dict[str, Any]] = []
        preview_rows: Dict[str, List[Dict[str, Any]]] = {}
        for agent_id in core.active_agent_ids:
            call_name = self._call_name_for_agent(core, agent_id)
            if call_name is None:
                continue
            actionable.append({"agent_id": agent_id, "call_name": call_name})
            preview_rows[agent_id] = self._preview_rows(core, agent_id, call_name)

        if not actionable:
            self.last_joint_actions = {}
            return {}

        prompt_text = self.builder.build_prompt(core, actionable, preview_rows)
        fallback = self._fallback_answer(core, actionable)
        try:
            raw_text = ollama_invoke(prompt_text)
            parsed = _extract_json(raw_text)
        except Exception:
            parsed = fallback
            raw_text = json.dumps(fallback)

        normalized = self._normalize_answer(parsed, actionable)
        normalized = self._repair_joint_actions(core, normalized, actionable)

        joint_actions = {
            item["agent_id"]: int(item["heading_change_deg"])
            for item in normalized["answer"]["agents"]
        }

        for item in normalized["answer"]["agents"]:
            ctrl = self._controller_state(item["agent_id"])
            ctrl.last_advice_turn = int(item["heading_change_deg"])
            ctrl.advice_active = True
            ctrl.last_call_name = item["call_name"]
            if item["call_name"] == "EXECUTE_TURN":
                ctrl.execute_called = True
                ctrl.stage = "WAIT_FOR_SAFE"
            elif item["call_name"] == "EMERGENCY_MANEUVER":
                ctrl.emergency_count += 1
                ctrl.stage = "WAIT_FOR_SAFE"
            elif item["call_name"] == "MERGE_BACK":
                ctrl.merge_back_count += 1
                ctrl.stage = "MERGE_BACK_RECHECK"

        advice_embeddings, _, advice_active_masks = self.get_advice_for_rl(core.active_agent_ids, core=core)
        ghost_compare = self._ghost_compare_joint_horizon(
            core,
            ppo,
            llm_turns=joint_actions,
            advice_embeddings=advice_embeddings,
            advice_active_masks=advice_active_masks,
        )
        self._log_joint_call(
            step=step,
            prompt_text=prompt_text,
            raw_text=raw_text,
            normalized=normalized,
            preview_rows=preview_rows,
            ghost_compare=ghost_compare,
        )
        self.last_joint_actions = joint_actions
        return joint_actions

    def get_advice_for_rl(
        self,
        active_agent_ids: Sequence[str],
        *,
        core: Optional["MultiAgentSectorCore"] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, float]]:
        embeddings: Dict[str, np.ndarray] = {}
        target_bins: Dict[str, int] = {}
        active_masks: Dict[str, float] = {}
        for agent_id in active_agent_ids:
            ctrl = self._controller_state(agent_id)
            boundary_warning = False
            conflict_predicted = False
            if core is not None and agent_id in core.agent_states:
                state = core.get_agent(agent_id)
                if state.finished:
                    ctrl.advice_active = False
                boundary_warning = bool(state.boundary_warning_active)
                conflict_predicted = bool(state.conflict_predicted)
            embeddings[agent_id] = self._advice_embedding(
                ctrl.last_advice_turn,
                ctrl.last_call_name,
                boundary_warning,
                conflict_predicted,
                ctrl.advice_active,
            )
            target_bins[agent_id] = -1 if not ctrl.advice_active else deg_to_action_idx(ctrl.last_advice_turn, ACTION_BINS)
            active_masks[agent_id] = 1.0 if ctrl.advice_active else 0.0
        return embeddings, target_bins, active_masks

    def choose_llm_actions(
        self,
        core: "MultiAgentSectorCore",
        *,
        step: int,
        ppo: Optional["SharedActorCentralCriticPPO"] = None,
    ) -> Dict[str, int]:
        return self.update(core, step=step, ppo=ppo)
