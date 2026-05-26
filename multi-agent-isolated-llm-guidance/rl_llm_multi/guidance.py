"""Guidance interfaces and shadow rollout evaluation for LLM-guided MAPPO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from configs import (
    ACTION_BINS,
    DECISION_MEMORY_PATH,
    DECISION_MEMORY_VISUAL_AUDIT_DIR,
    DECISION_MEMORY_VISUAL_AUDIT_ENABLED,
    DECISION_MEMORY_VISUAL_AUDIT_FRAME_COUNT,
    GUIDANCE_RANDOM_SEED,
    GUIDANCE_SOURCE,
    JSON_ANSWERS_DIR,
    LLM_GUIDANCE_USE_VISION,
    LLM_SHADOW_GAMMA,
    LLM_SHADOW_HORIZON,
)
from .utils import action_idx_to_deg, deg_to_action_idx


@dataclass(frozen=True)
class GuidanceAdvice:
    """Structured per-agent LLM advice for MAPPO training labels."""

    agent_id: str
    llm_action_deg: int
    llm_action_idx: int
    phase: str
    source: str = ""
    fallback: bool = False
    requested_action_deg: Optional[int] = None
    memory_action_deg: Optional[int] = None
    memory_action_idx: Optional[int] = None
    memory_ref: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ShadowEvaluationResult:
    """Discounted returns from cloned simulator branches."""

    mappo_returns: Dict[str, float]
    llm_returns: Dict[str, float]
    memory_returns: Dict[str, float]
    mappo_joint_return: float
    llm_joint_return: float
    memory_joint_return: float
    memory_agent_ids: list[str]
    memory_attribution_returns: Dict[str, Dict[str, float]]
    memory_attribution_joint_returns: Dict[str, float]


class NoGuidanceProvider:
    """Return no advice while preserving the trainer call site."""

    def reset(self) -> None:
        return None

    def actions_for_step(
        self,
        core,
        *,
        step: int,
        proposed_actions: Mapping[str, int],
    ) -> Dict[str, int]:
        return {}

    def advice_for_step(
        self,
        core,
        *,
        step: int,
        proposed_actions: Mapping[str, int],
        latest_frame_path: Optional[str] = None,
    ) -> Dict[str, GuidanceAdvice]:
        return {}

    def get_advice_for_rl(self, agent_id: Optional[str] = None):
        return None

    def memory_episode_id(self, core) -> str:
        return ""


class LLMGuidanceProvider:
    """Call the global LLM controller and expose advice without overriding MAPPO."""

    def __init__(
        self,
        *,
        save_dir: str | Path = JSON_ANSWERS_DIR,
        memory_path: str | Path = DECISION_MEMORY_PATH,
        use_vision: bool = LLM_GUIDANCE_USE_VISION,
        memory_visual_audit_enabled: bool = DECISION_MEMORY_VISUAL_AUDIT_ENABLED,
        memory_visual_audit_dir: str | Path = DECISION_MEMORY_VISUAL_AUDIT_DIR,
        memory_visual_audit_frame_count: int = DECISION_MEMORY_VISUAL_AUDIT_FRAME_COUNT,
        controller: Optional[Any] = None,
        guidance_source: str = GUIDANCE_SOURCE,
        guidance_random_seed: Optional[int] = GUIDANCE_RANDOM_SEED,
    ) -> None:
        if controller is None:
            from .llm import GlobalLangGraphGuidanceController

            controller = GlobalLangGraphGuidanceController(
                save_dir=str(save_dir),
                memory_path=str(memory_path),
                memory_visual_audit_enabled=bool(memory_visual_audit_enabled),
                memory_visual_audit_dir=str(memory_visual_audit_dir),
                memory_visual_audit_frame_count=int(memory_visual_audit_frame_count),
                guidance_source=guidance_source,
                guidance_random_seed=guidance_random_seed,
            )
        self.controller = controller
        self.use_vision = bool(use_vision)
        self.last_advice: Dict[str, GuidanceAdvice] = {}

    def reset(self) -> None:
        self.last_advice = {}
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset()

    def actions_for_step(
        self,
        core,
        *,
        step: int,
        proposed_actions: Mapping[str, int],
    ) -> Dict[str, int]:
        return {}

    def advice_for_step(
        self,
        core,
        *,
        step: int,
        proposed_actions: Mapping[str, int],
        latest_frame_path: Optional[str] = None,
    ) -> Dict[str, GuidanceAdvice]:
        if not proposed_actions:
            self.last_advice = {}
            return {}

        choose_labels = getattr(self.controller, "advisory_update", None)
        if not callable(choose_labels):
            choose_labels = getattr(self.controller, "update")
        actions = choose_labels(
            core,
            step=int(step),
            latest_frame_path=latest_frame_path,
            use_vision=self.use_vision,
        )
        normalized = getattr(self.controller, "last_normalized", {}) or {}
        answer = normalized.get("answer") if isinstance(normalized, dict) else {}
        answer = answer if isinstance(answer, dict) else {}
        normalized_actions = answer.get("actions") if isinstance(answer.get("actions"), dict) else {}
        debug = normalized.get("debug") if isinstance(normalized, dict) else {}
        debug = debug if isinstance(debug, dict) else {}
        stage_by_agent = debug.get("stage_by_agent") if isinstance(debug.get("stage_by_agent"), dict) else {}
        fallback_agents = set(str(agent_id) for agent_id in debug.get("fallback_agents", []) or [])
        retrieved_by_agent = _retrieved_memory_by_agent(debug.get("retrieved_memories", []))

        advice: Dict[str, GuidanceAdvice] = {}
        for agent_id, action_deg in sorted((actions or {}).items()):
            agent_id = str(agent_id)
            if agent_id not in proposed_actions:
                continue
            final_action = normalized_actions.get(agent_id, {})
            final_action = final_action if isinstance(final_action, dict) else {}
            phase = str(final_action.get("call_name") or stage_by_agent.get(agent_id) or "")
            source = str(final_action.get("source", ""))
            requested = _optional_int(final_action.get("requested_heading_change_deg"))
            memory = retrieved_by_agent.get(agent_id)
            memory_action_deg = _memory_action_deg(memory)
            memory_ref = None
            if isinstance(memory, dict) and isinstance(memory.get("memory_ref"), dict):
                memory_ref = dict(memory["memory_ref"])
            advice[agent_id] = GuidanceAdvice(
                agent_id=agent_id,
                llm_action_deg=int(action_deg),
                llm_action_idx=deg_to_action_idx(int(action_deg), ACTION_BINS),
                phase=phase,
                source=source,
                fallback=bool(source == "fallback" or agent_id in fallback_agents),
                requested_action_deg=requested,
                memory_action_deg=memory_action_deg,
                memory_action_idx=(
                    None
                    if memory_action_deg is None
                    else deg_to_action_idx(int(memory_action_deg), ACTION_BINS)
                ),
                memory_ref=memory_ref,
            )
        self.last_advice = advice
        return advice

    def get_advice_for_rl(self, agent_id: Optional[str] = None):
        if agent_id is None:
            return dict(self.last_advice)
        return self.last_advice.get(str(agent_id))

    def memory_episode_id(self, core) -> str:
        memory_episode_id = getattr(self.controller, "_memory_episode_id", None)
        if callable(memory_episode_id):
            return str(memory_episode_id(core))
        return ""


def shadow_evaluate_actions(
    core: Any,
    agent: Any,
    *,
    proposed_actions: Mapping[str, int],
    guidance_by_agent: Mapping[str, GuidanceAdvice],
    horizon: int = LLM_SHADOW_HORIZON,
    gamma: float = LLM_SHADOW_GAMMA,
    episode_id: int = 0,
) -> ShadowEvaluationResult:
    """Compare MAPPO, joint LLM, and memory actions on cloned cores.

    proposed_actions are MAPPO action indexes. LLM and memory actions are stored
    as heading-change degrees in GuidanceAdvice. Only the first shadow step is
    forced; later steps use deterministic MAPPO actions from the current policy.
    """

    guidance_by_id = {str(agent_id): advice for agent_id, advice in guidance_by_agent.items()}
    tracked_ids = sorted(set(str(agent_id) for agent_id in proposed_actions) | set(guidance_by_id))
    mappo_first = _degree_actions_from_indices(proposed_actions)
    llm_first = dict(mappo_first)
    memory_first = dict(mappo_first)
    memory_agent_ids: list[str] = []

    for agent_id, advice in guidance_by_id.items():
        llm_first[agent_id] = int(advice.llm_action_deg)
        if advice.memory_action_deg is not None:
            memory_first[agent_id] = int(advice.memory_action_deg)
            memory_agent_ids.append(agent_id)
    memory_agent_ids = sorted(memory_agent_ids)

    mappo_returns = _rollout_branch(
        core,
        agent,
        first_actions_deg=mappo_first,
        tracked_agent_ids=tracked_ids,
        horizon=int(horizon),
        gamma=float(gamma),
        episode_id=int(episode_id),
    )
    llm_returns = _rollout_branch(
        core,
        agent,
        first_actions_deg=llm_first,
        tracked_agent_ids=tracked_ids,
        horizon=int(horizon),
        gamma=float(gamma),
        episode_id=int(episode_id),
    )
    memory_returns = (
        _rollout_branch(
            core,
            agent,
            first_actions_deg=memory_first,
            tracked_agent_ids=tracked_ids,
            horizon=int(horizon),
            gamma=float(gamma),
            episode_id=int(episode_id),
        )
        if memory_agent_ids
        else {}
    )
    memory_attribution_returns: Dict[str, Dict[str, float]] = {}
    memory_attribution_joint_returns: Dict[str, float] = {}
    if len(memory_agent_ids) > 1:
        for agent_id in memory_agent_ids:
            advice = guidance_by_id[agent_id]
            first_actions = dict(mappo_first)
            first_actions[agent_id] = int(advice.memory_action_deg)
            branch_returns = _rollout_branch(
                core,
                agent,
                first_actions_deg=first_actions,
                tracked_agent_ids=tracked_ids,
                horizon=int(horizon),
                gamma=float(gamma),
                episode_id=int(episode_id),
            )
            memory_attribution_returns[agent_id] = branch_returns
            memory_attribution_joint_returns[agent_id] = _mean_return_for_ids(branch_returns, tracked_ids)

    return ShadowEvaluationResult(
        mappo_returns=mappo_returns,
        llm_returns=llm_returns,
        memory_returns=memory_returns,
        mappo_joint_return=_mean_return_for_ids(mappo_returns, tracked_ids),
        llm_joint_return=_mean_return_for_ids(llm_returns, tracked_ids),
        memory_joint_return=_mean_return_for_ids(memory_returns, tracked_ids) if memory_agent_ids else 0.0,
        memory_agent_ids=memory_agent_ids,
        memory_attribution_returns=memory_attribution_returns,
        memory_attribution_joint_returns=memory_attribution_joint_returns,
    )


def _mean_return_for_ids(returns: Mapping[str, float], agent_ids: Sequence[str]) -> float:
    values = [float(returns[str(agent_id)]) for agent_id in agent_ids if str(agent_id) in returns]
    return sum(values) / float(len(values)) if values else 0.0

def _rollout_branch(
    core: Any,
    agent: Any,
    *,
    first_actions_deg: Mapping[str, int],
    tracked_agent_ids: Sequence[str],
    horizon: int,
    gamma: float,
    episode_id: int,
) -> Dict[str, float]:
    sim = core.clone()
    returns = {str(agent_id): 0.0 for agent_id in tracked_agent_ids}
    discount = 1.0
    actions_deg = {str(agent_id): int(action) for agent_id, action in first_actions_deg.items()}
    for _ in range(max(0, int(horizon))):
        _, rewards, _, _, info = sim.step(actions_deg, compute_predictions=True)
        for agent_id in returns:
            returns[agent_id] += discount * _reward_for_agent(rewards, info, agent_id)
        if bool(getattr(sim, "last_team_done", False) or getattr(sim, "last_team_truncated", False)):
            break
        discount *= float(gamma)
        actions_deg = _deterministic_policy_degree_actions(agent, sim, episode_id=episode_id)
    return returns


def _deterministic_policy_degree_actions(agent: Any, core: Any, *, episode_id: int) -> Dict[str, int]:
    active_ids = _decision_agent_ids(core)
    if not active_ids:
        return {}
    observations = {agent_id: core.get_local_observation(agent_id) for agent_id in active_ids}
    action_indices, _, _ = agent.select_actions(
        global_state=core.global_state_vector(),
        local_observations=observations,
        active_agent_ids=active_ids,
        episode_id=int(episode_id),
        deterministic=True,
        store=False,
    )
    return _degree_actions_from_indices(action_indices)


def _decision_agent_ids(core: Any) -> list[str]:
    ids = []
    for agent_id in list(getattr(core, "active_agent_ids", []) or []):
        try:
            state = core.get_agent(agent_id)
        except Exception:
            state = getattr(core, "agent_states", {}).get(agent_id)
        if state is None:
            continue
        if bool(getattr(state, "has_entered_sector", False)):
            ids.append(str(agent_id))
    return ids


def _degree_actions_from_indices(actions: Mapping[str, int]) -> Dict[str, int]:
    return {
        str(agent_id): action_idx_to_deg(int(action_idx), ACTION_BINS)
        for agent_id, action_idx in (actions or {}).items()
    }


def _reward_for_agent(rewards: Mapping[str, float], info: Mapping[str, Any], agent_id: str) -> float:
    if isinstance(rewards, Mapping) and agent_id in rewards:
        return float(rewards[agent_id])
    for key in ("agent_rewards", "agent_dense_rewards", "agent_terminal_rewards"):
        values = info.get(key) if isinstance(info, Mapping) else None
        if isinstance(values, Mapping) and agent_id in values:
            return float(values[agent_id])
    return 0.0


def _retrieved_memory_by_agent(value: Any) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not isinstance(value, (list, tuple)):
        return result
    for item in value:
        if not isinstance(item, dict):
            continue
        agent_id = item.get("agent")
        if agent_id is None:
            continue
        result[str(agent_id)] = item
    return result


def _memory_action_deg(memory: Optional[Mapping[str, Any]]) -> Optional[int]:
    if not isinstance(memory, Mapping):
        return None
    corrected = _optional_int(memory.get("corrected"))
    if corrected is not None:
        return corrected
    return _optional_int(memory.get("action"))


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None
