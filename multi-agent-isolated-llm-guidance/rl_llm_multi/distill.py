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
    DISTILL_LOSS_WEIGHT,
    DISTILL_MIN_WEIGHT,
    DISTILL_TEACHERS,
    LLM_RETURN_WEIGHT_AGENT_COEF,
    LLM_RETURN_WEIGHT_JOINT_COEF,
    LLM_SHADOW_GAMMA,
    LLM_SHADOW_HORIZON,
    LLM_SHADOW_MAX_WEIGHT,
    LLM_SHADOW_RETURN_MARGIN,
    LLM_SHADOW_RETURN_SCALE,
    LLM_SHADOW_RETURN_WEIGHT_CLIP,
    TEACHER_HEUR_PATH,
    TEACHER_LLM_PATH,
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
    llm_path: str | Path = TEACHER_LLM_PATH,
    heur_path: str | Path = TEACHER_HEUR_PATH,
) -> Dict[str, TeacherPolicy]:
    """Load the two frozen teacher networks. Raises if a checkpoint is missing."""
    teachers: Dict[str, TeacherPolicy] = {}
    for name, path in (("llm", llm_path), ("heur", heur_path)):
        path = Path(path)
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
    teacher_llm = teachers["llm"]
    teacher_heur = teachers["heur"]

    info: Dict[str, Dict[str, object]] = {}
    llm_actions_deg: Dict[str, int] = {}
    heur_actions_deg: Dict[str, int] = {}
    for agent_id in active_ids:
        agent_id = str(agent_id)
        obs_np = observations.get(agent_id)
        if obs_np is None:
            continue
        try:
            agent_index = agent_id_to_index(agent_id)
        except ValueError:
            continue
        llm_a, llm_p = _teacher_action_and_probs(teacher_llm, obs_np, agent_index)
        heur_a, heur_p = _teacher_action_and_probs(teacher_heur, obs_np, agent_index)
        info[agent_id] = {"llm_a": llm_a, "llm_p": llm_p, "heur_a": heur_a, "heur_p": heur_p}
        llm_actions_deg[agent_id] = action_idx_to_deg(int(llm_a), ACTION_BINS)
        heur_actions_deg[agent_id] = action_idx_to_deg(int(heur_a), ACTION_BINS)

    if not info:
        return {}

    mode = str(competence_mode).lower()
    weights: Dict[str, tuple[float, float]] = {}
    if mode == "shadow":
        mappo_returns, teacher_returns, tracked = shadow_evaluate_teachers(
            core,
            agent,
            proposed_actions=proposed_actions,
            teacher_actions_deg={"llm": llm_actions_deg, "heur": heur_actions_deg},
            horizon=int(LLM_SHADOW_HORIZON),
            gamma=float(LLM_SHADOW_GAMMA),
            episode_id=int(episode_id),
        )
        llm_returns = teacher_returns.get("llm", {})
        heur_returns = teacher_returns.get("heur", {})
        g_mappo_joint = _mean_return_for_ids(mappo_returns, tracked)
        g_llm_joint = _mean_return_for_ids(llm_returns, tracked)
        g_heur_joint = _mean_return_for_ids(heur_returns, tracked)
        for agent_id in info:
            g_mappo_i = float(mappo_returns.get(agent_id, 0.0))
            g_llm_i = float(llm_returns.get(agent_id, 0.0))
            g_heur_i = float(heur_returns.get(agent_id, 0.0))
            w_llm = _hybrid_weight(g_llm_i, g_mappo_i, g_llm_joint, g_mappo_joint)
            w_heur = _hybrid_weight(g_heur_i, g_mappo_i, g_heur_joint, g_mappo_joint)
            weights[agent_id] = (w_llm, w_heur)
    elif mode == "weather_rule":
        for agent_id in info:
            weights[agent_id] = weather_rule_weights(core, agent_id)
    else:  # "equal"
        for agent_id in info:
            weights[agent_id] = (1.0, 1.0)

    # Honor DISTILL_TEACHERS: zero the weight of any teacher not in the active
    # set. This enables the single-teacher ablations (T1 = llm-only via
    # DISTILL_TEACHERS=("llm",), T2 = heur-only) in every competence mode.
    active = {str(name).lower() for name in DISTILL_TEACHERS}
    llm_on = "llm" in active
    heur_on = "heur" in active
    if not (llm_on and heur_on):
        for agent_id, (w_llm, w_heur) in list(weights.items()):
            weights[agent_id] = (w_llm if llm_on else 0.0, w_heur if heur_on else 0.0)

    metadata: Dict[str, Dict[str, object]] = {}
    for agent_id, rec in info.items():
        w_llm, w_heur = weights.get(agent_id, (0.0, 0.0))
        metadata[agent_id] = {
            "distill_label_available": True,
            "distill_phase": "",  # no controller phase in distillation mode
            "teacher_llm_probs": rec["llm_p"],
            "teacher_heur_probs": rec["heur_p"],
            "teacher_llm_action_idx": int(rec["llm_a"]),
            "teacher_heur_action_idx": int(rec["heur_a"]),
            "distill_weight_llm": float(w_llm),
            "distill_weight_heur": float(w_heur),
        }
    return metadata
