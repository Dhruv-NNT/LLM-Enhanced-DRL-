"""Dual-teacher distillation helpers for MAPPO training (Stage C).

This module is the bridge used by ``train.py`` when ``DISTILL_ENABLED`` is set.
It loads the two frozen teacher networks, queries them at decision time to get
per-agent target distributions + actions, computes a competence weight for each
teacher (shadow / weather-rule / equal), and packages everything into the
per-agent metadata dict consumed by ``MAPPO.store_outcomes`` -> ``RolloutBuffer``.

The MAPPO student is always what acts in the real environment; teachers only
shape the auxiliary distillation loss.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from configs import (
    ACTION_BINS,
    COMPETENCE_MODE,
    DISTILL_DECAY_ENABLED,
    DISTILL_DECAY_END_STEP,
    DISTILL_DECAY_START_STEP,
    DISTILL_LLM_TEACHER_NAMES,
    DISTILL_LOSS_WEIGHT,
    DISTILL_MIN_WEIGHT,
    DISTILL_TEACHER_PATHS,
    DISTILL_TEACHERS,
    LLM_RETURN_WEIGHT_AGENT_COEF,
    LLM_RETURN_WEIGHT_JOINT_COEF,
    LLM_SHADOW_GAMMA,
    LLM_SHADOW_HORIZON,
    LLM_SHADOW_MAX_WEIGHT,
    LLM_SHADOW_RETURN_MARGIN,
    LLM_SHADOW_RETURN_SCALE,
    LLM_SHADOW_RETURN_WEIGHT_CLIP,
    WEATHER_RULE_CLEARANCE_MULT,
    WEATHER_RULE_HEUR_WHEN_FAR,
    WEATHER_RULE_HEUR_WHEN_NEAR,
    WEATHER_RULE_LLM_WHEN_FAR,
    WEATHER_RULE_LLM_WHEN_NEAR,
    WEATHER_RULE_SOFTNESS_MULT,
    WEATHER_CLEARANCE_BUFFER_NM,
)
from .guidance import _mean_return_for_ids, shadow_evaluate_teachers
from .mappo import agent_id_to_index, device
from .teacher import TeacherPolicy, load_teacher
from .utils import action_idx_to_deg


def current_distill_weight(agent_steps: int) -> float:
    """Decay the distillation weight from DISTILL_LOSS_WEIGHT toward
    DISTILL_MIN_WEIGHT over [DISTILL_DECAY_START_STEP, DISTILL_DECAY_END_STEP],
    so late training is pure PPO fine-tune (mirrors _current_llm_loss_weight)."""
    base = float(DISTILL_LOSS_WEIGHT)
    if not bool(DISTILL_DECAY_ENABLED):
        return base
    start = int(DISTILL_DECAY_START_STEP)
    end = int(DISTILL_DECAY_END_STEP)
    if end <= start:
        return max(float(DISTILL_MIN_WEIGHT), base)
    progress = (float(agent_steps) - float(start)) / max(float(end - start), 1.0)
    progress = max(0.0, min(1.0, progress))
    decayed = base + progress * (float(DISTILL_MIN_WEIGHT) - base)
    return max(float(DISTILL_MIN_WEIGHT), decayed)


def load_distill_teachers(
    teacher_names: Sequence[str] = DISTILL_TEACHERS,
    paths: Optional[Mapping[str, str | Path]] = None,
) -> Dict[str, TeacherPolicy]:
    """Load the frozen teacher networks named in ``teacher_names`` from the
    registry. Only the requested teachers are loaded, so single-teacher and
    two-teacher runs do not require checkpoints for teachers they will not use.
    Raises if a requested teacher is unregistered or its checkpoint is missing."""
    registry = dict(DISTILL_TEACHER_PATHS if paths is None else paths)
    teachers: Dict[str, TeacherPolicy] = {}
    for name in teacher_names:
        if name not in registry:
            raise KeyError(
                f"Teacher '{name}' is not registered in DISTILL_TEACHER_PATHS. "
                f"Known teachers: {sorted(registry)}."
            )
        path = Path(registry[name])
        if not path.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint for '{name}' not found at {path}. "
                f"Run generate_guided_trajectories.py and train_teachers.py first."
            )
        teacher = load_teacher(path, map_location=device)
        for param in teacher.parameters():
            param.requires_grad_(False)
        teacher.eval()
        teachers[name] = teacher
    return teachers


def _max_weight() -> float:
    return float(LLM_SHADOW_MAX_WEIGHT if LLM_SHADOW_MAX_WEIGHT is not None else LLM_SHADOW_RETURN_WEIGHT_CLIP)


def _return_weight(g_teacher: float, g_mappo: float) -> float:
    delta = float(g_teacher) - float(g_mappo)
    if delta <= float(LLM_SHADOW_RETURN_MARGIN):
        return 0.0
    scale = max(float(LLM_SHADOW_RETURN_SCALE), 1e-6)
    return max(0.0, min(_max_weight(), delta / scale))


def _hybrid_weight(g_t_i: float, g_mappo_i: float, g_t_joint: float, g_mappo_joint: float) -> float:
    agent_weight = _return_weight(g_t_i, g_mappo_i)
    joint_weight = _return_weight(g_t_joint, g_mappo_joint)
    hybrid = (
        float(LLM_RETURN_WEIGHT_AGENT_COEF) * agent_weight
        + float(LLM_RETURN_WEIGHT_JOINT_COEF) * joint_weight
    )
    return max(0.0, min(_max_weight(), hybrid))


@torch.no_grad()
def _teacher_action_and_probs(teacher: TeacherPolicy, obs_np, agent_index: int):
    dev = next(teacher.parameters()).device
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=dev).view(1, -1)
    idx_t = torch.tensor([int(agent_index)], dtype=torch.long, device=dev)
    logits = teacher.action_logits(obs_t, idx_t)
    probs = F.softmax(logits, dim=-1).view(-1)
    action_idx = int(torch.argmax(logits, dim=-1).item())
    return action_idx, probs.detach().cpu()


def _weather_buffer_units(core) -> float:
    units = getattr(core, "weather_clearance_buffer_units", None)
    if units is None or not math.isfinite(float(units)) or float(units) <= 0.0:
        return float(WEATHER_CLEARANCE_BUFFER_NM)
    return float(units)


def weather_rule_weights(core, agent_id: str) -> tuple[float, float]:
    """Fixed competence rule: trust the heuristic more when weather is near,
    the LLM more otherwise. 'Near' is measured against the simulator's own
    weather buffer (grid units)."""
    try:
        state = core.get_agent(agent_id)
    except Exception:
        state = getattr(core, "agent_states", {}).get(agent_id)
    clearance = float("inf")
    if state is not None:
        clearance = float(getattr(state, "predicted_weather_clearance", float("inf")))
    buffer_units = _weather_buffer_units(core)
    thresh = float(WEATHER_RULE_CLEARANCE_MULT) * buffer_units
    soft = float(WEATHER_RULE_SOFTNESS_MULT) * buffer_units
    if not math.isfinite(clearance):
        near = 0.0
    elif soft <= 0.0:
        near = 1.0 if clearance < thresh else 0.0
    else:
        near = 1.0 / (1.0 + math.exp((clearance - thresh) / soft))
    w_heur = float(WEATHER_RULE_HEUR_WHEN_FAR) + near * (
        float(WEATHER_RULE_HEUR_WHEN_NEAR) - float(WEATHER_RULE_HEUR_WHEN_FAR)
    )
    w_llm = float(WEATHER_RULE_LLM_WHEN_FAR) + near * (
        float(WEATHER_RULE_LLM_WHEN_NEAR) - float(WEATHER_RULE_LLM_WHEN_FAR)
    )
    return w_llm, w_heur


def weather_rule_weight_for(core, agent_id: str, teacher_name: str) -> float:
    """Per-teacher weather-rule weight: LLM-type teachers (named in
    DISTILL_LLM_TEACHER_NAMES) get the 'far' branch, all other teachers get the
    heuristic 'near' branch."""
    w_llm, w_heur = weather_rule_weights(core, agent_id)
    return w_llm if teacher_name in set(DISTILL_LLM_TEACHER_NAMES) else w_heur


@torch.no_grad()
def build_distill_metadata(
    core,
    agent,
    observations: Mapping[str, object],
    active_ids: Sequence[str],
    *,
    teachers: Mapping[str, TeacherPolicy],
    proposed_actions: Mapping[str, int],
    competence_mode: str = COMPETENCE_MODE,
    episode_id: int = 0,
) -> Dict[str, Dict[str, object]]:
    """Build per-agent distillation metadata: teacher distributions, teacher
    actions, and competence weights, ready for RolloutBuffer.add via
    MAPPO.store_outcomes(llm_metadata_by_agent=...)."""
    if not active_ids or not teachers:
        return {}
    # Honor DISTILL_TEACHERS: distill only from teachers that are both loaded and
    # listed as active. This drives the single-teacher (T1/T2), two-teacher, and
    # three-teacher runs from one code path.
    active_names = [str(name) for name in DISTILL_TEACHERS if str(name) in teachers]
    if not active_names:
        return {}

    # Per-agent teacher action index + probability vector for each active teacher.
    info: Dict[str, Dict[str, object]] = {}
    actions_deg_by_teacher: Dict[str, Dict[str, int]] = {name: {} for name in active_names}
    for agent_id in active_ids:
        agent_id = str(agent_id)
        obs_np = observations.get(agent_id)
        if obs_np is None:
            continue
        try:
            agent_index = agent_id_to_index(agent_id)
        except ValueError:
            continue
        rec: Dict[str, object] = {}
        for name in active_names:
            a, p = _teacher_action_and_probs(teachers[name], obs_np, agent_index)
            rec[name] = {"a": int(a), "p": p}
            actions_deg_by_teacher[name][agent_id] = action_idx_to_deg(int(a), ACTION_BINS)
        info[agent_id] = rec

    if not info:
        return {}

    mode = str(competence_mode).lower()
    # weights[agent_id][teacher_name] -> competence weight.
    weights: Dict[str, Dict[str, float]] = {agent_id: {} for agent_id in info}
    if mode == "shadow":
        mappo_returns, teacher_returns, tracked = shadow_evaluate_teachers(
            core,
            agent,
            proposed_actions=proposed_actions,
            teacher_actions_deg=actions_deg_by_teacher,
            horizon=int(LLM_SHADOW_HORIZON),
            gamma=float(LLM_SHADOW_GAMMA),
            episode_id=int(episode_id),
        )
        g_mappo_joint = _mean_return_for_ids(mappo_returns, tracked)
        joint_by_teacher = {
            name: _mean_return_for_ids(teacher_returns.get(name, {}), tracked)
            for name in active_names
        }
        for agent_id in info:
            g_mappo_i = float(mappo_returns.get(agent_id, 0.0))
            for name in active_names:
                g_t_i = float(teacher_returns.get(name, {}).get(agent_id, 0.0))
                weights[agent_id][name] = _hybrid_weight(
                    g_t_i, g_mappo_i, joint_by_teacher[name], g_mappo_joint
                )
    elif mode == "weather_rule":
        for agent_id in info:
            for name in active_names:
                weights[agent_id][name] = weather_rule_weight_for(core, agent_id, name)
    else:  # "equal"
        for agent_id in info:
            for name in active_names:
                weights[agent_id][name] = 1.0

    metadata: Dict[str, Dict[str, object]] = {}
    for agent_id, rec in info.items():
        metadata[agent_id] = {
            "distill_label_available": True,
            "distill_phase": "",  # no controller phase in distillation mode
            "teacher_probs": {name: rec[name]["p"] for name in active_names},
            "teacher_action_idx": {name: int(rec[name]["a"]) for name in active_names},
            "distill_weight": {
                name: float(weights[agent_id].get(name, 0.0)) for name in active_names
            },
        }
    return metadata
