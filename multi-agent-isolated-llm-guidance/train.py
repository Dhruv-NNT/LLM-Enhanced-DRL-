"""Train a pure-RL MAPPO baseline for the multi-agent sector simulator."""

from __future__ import annotations

import argparse
import json
import sys
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import configs as config_module
from configs import (
    ACTION_BINS,
    MAPPO_ACTIVATION,
    MAPPO_CUDA_VISIBLE_DEVICES,
    MAPPO_EPS_CLIP,
    MAPPO_EVAL_FREQ,
    MAPPO_GAE_LAMBDA,
    MAPPO_GAMMA,
    MAPPO_HIDDEN_DIMS,
    MAPPO_K_EPOCHS,
    TRAIN_LOG_DIR,
    MAPPO_LR_ACTOR,
    MAPPO_LR_CRITIC,
    MAPPO_MAX_AGENT_STEPS,
    MAPPO_MINIBATCH_SIZE,
    MAPPO_N_EVAL_EPISODES,
    MAPPO_NORMALIZE_REWARD,
    MAPPO_ORTHOGONAL_INIT,
    MAPPO_SAVE_MODEL_FREQ,
    MAPPO_UPDATE_AGENT_STEPS,
    MAPPO_USE_LAYER_NORM,
    LLM_ADVISORY_MEMORY_WRITE_ENABLED,
    LLM_ADVISORY_MEMORY_WRITE_MARGIN,
    LLM_GUIDED_MAPPO_AUDIT_ENABLED,
    LLM_GUIDED_MAPPO_AUDIT_MAX_ROWS,
    LLM_GUIDANCE_END_STEP,
    LLM_GUIDANCE_MEMORY_ISOLATED_PER_RUN,
    LLM_GUIDANCE_MEMORY_PATH,
    LLM_MEMORY_PATH,
    LLM_MEMORY_RUN_ID,
    DECISION_MEMORY_PATH,
    LLM_GUIDANCE_START_STEP,
    LLM_GUIDANCE_USE_VISION,
    LLM_LOSS_DECAY_ENABLED,
    LLM_LOSS_MIN_WEIGHT,
    LLM_LOSS_WEIGHT,
    LLM_GUIDED_MEMORY_MAX_EPISODES,
    LLM_MEMORY_EVALUATOR_ENABLED,
    LLM_MEMORY_UPDATE_MARGIN,
    LLM_MEMORY_UPDATE_WITH_BEST_OF,
    LLM_RETURN_WEIGHT_AGENT_COEF,
    LLM_RETURN_WEIGHT_JOINT_COEF,
    LLM_SHADOW_EVAL_ENABLED,
    LLM_SHADOW_GAMMA,
    LLM_SHADOW_HORIZON,
    LLM_SHADOW_RETURN_MARGIN,
    LLM_SHADOW_RETURN_SCALE,
    LLM_SHADOW_MAX_WEIGHT,
    LLM_SHADOW_RETURN_WEIGHT_CLIP,
    USE_LLM_GUIDED_TRAINING,
    DECISION_MEMORY_VISUAL_AUDIT_DIR,
    DECISION_MEMORY_VISUAL_AUDIT_ENABLED,
    DECISION_MEMORY_VISUAL_AUDIT_FRAME_COUNT,
    MAX_AGENTS,
    NUM_AGENTS_DEFAULT,
    NUM_WEATHER_CELLS_DEFAULT,
    ROUTE_IDS_DEFAULT,
    SEED,
)

if MAPPO_CUDA_VISIBLE_DEVICES is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(MAPPO_CUDA_VISIBLE_DEVICES)

from rl_llm_multi import LLMGuidanceProvider, MAPPO, MultiAgentParallelEnv, NoGuidanceProvider, shadow_evaluate_actions
from rl_llm_multi.mappo import evaluate_policy
from rl_llm_multi.memory import DecisionMemoryStore
from rl_llm_multi.utils import action_idx_to_deg

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *_, **__) -> None:
            pass

        def add_scalar(self, *_, **__) -> None:
            return None

        def close(self) -> None:
            return None


def _normalize_route_ids(route_ids: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not route_ids:
        return None
    values = [str(route_id).strip() for route_id in route_ids if str(route_id).strip()]
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS_DEFAULT)
    parser.add_argument("--num-weather-cells", type=int, choices=(1, 2), default=NUM_WEATHER_CELLS_DEFAULT)
    parser.add_argument("--route-ids", nargs="*", default=ROUTE_IDS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-agent-steps", type=int, default=MAPPO_MAX_AGENT_STEPS)
    parser.add_argument("--update-agent-steps", type=int, default=MAPPO_UPDATE_AGENT_STEPS)
    parser.add_argument("--eval-freq", type=int, default=MAPPO_EVAL_FREQ)
    parser.add_argument("--n-eval-episodes", type=int, default=MAPPO_N_EVAL_EPISODES)
    parser.add_argument("--save-model-freq", type=int, default=MAPPO_SAVE_MODEL_FREQ)
    parser.add_argument("--log-dir", type=str, default=str(TRAIN_LOG_DIR), help="Run root directory; TensorBoard, weights, and run artifacts are written under this folder.")
    parser.add_argument("--best-model-path", type=str, default=None, help="Optional override; defaults to <log-dir>/weights/best_model.pt.")
    parser.add_argument("--last-ckpt-path", type=str, default=None, help="Optional override; defaults to <log-dir>/weights/last_checkpoint.pt.")
    parser.add_argument("--episode-step-cap", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def make_env(
    *,
    num_agents: int,
    num_weather_cells: int,
    episode_step_cap: Optional[int] = None,
) -> MultiAgentParallelEnv:
    env = MultiAgentParallelEnv(
        num_agents=int(num_agents),
        max_agents=MAX_AGENTS,
        num_weather_cells=int(num_weather_cells),
    )
    if episode_step_cap is not None:
        env.core.max_step = int(episode_step_cap)
    return env


def build_agent(global_state_dim: int, local_obs_dim: int) -> MAPPO:
    return MAPPO(
        global_state_dim=global_state_dim,
        local_obs_dim=local_obs_dim,
        action_dim=len(ACTION_BINS),
        max_agents=MAX_AGENTS,
        hidden_dims=MAPPO_HIDDEN_DIMS,
        activation=MAPPO_ACTIVATION,
        use_layer_norm=MAPPO_USE_LAYER_NORM,
        orthogonal_init=MAPPO_ORTHOGONAL_INIT,
        lr_actor=MAPPO_LR_ACTOR,
        lr_critic=MAPPO_LR_CRITIC,
        gamma=MAPPO_GAMMA,
        K_epochs=MAPPO_K_EPOCHS,
        eps_clip=MAPPO_EPS_CLIP,
        mb_size=MAPPO_MINIBATCH_SIZE,
        gae_lambda=MAPPO_GAE_LAMBDA,
        normalize_reward=MAPPO_NORMALIZE_REWARD,
    )


def _llm_guidance_active(agent_steps: int) -> bool:
    return bool(
        USE_LLM_GUIDED_TRAINING
        and int(agent_steps) >= int(LLM_GUIDANCE_START_STEP)
        and int(agent_steps) < int(LLM_GUIDANCE_END_STEP)
    )


def _llm_guidance_active_for_episode(agent_steps_at_episode_start: int) -> bool:
    """Latch LLM guidance for a whole episode using episode-start agent steps."""

    return _llm_guidance_active(agent_steps_at_episode_start)


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    tensorboard_dir: Path
    tensorboard_run_dir: Path
    weights_dir: Path
    best_model_path: Path
    last_ckpt_path: Path


def _resolve_run_paths(args: argparse.Namespace) -> RunPaths:
    run_dir = Path(args.log_dir)
    weights_dir = run_dir / "weights"
    tensorboard_dir = run_dir / "tensorboard"
    run_name = run_dir.name or "run"
    tensorboard_run_dir = tensorboard_dir / run_name
    best_model_path = Path(args.best_model_path) if args.best_model_path else weights_dir / "best_model.pt"
    last_ckpt_path = Path(args.last_ckpt_path) if args.last_ckpt_path else weights_dir / "last_checkpoint.pt"
    return RunPaths(
        run_dir=run_dir,
        tensorboard_dir=tensorboard_dir,
        tensorboard_run_dir=tensorboard_run_dir,
        weights_dir=weights_dir,
        best_model_path=best_model_path,
        last_ckpt_path=last_ckpt_path,
    )


def _resolve_llm_memory_path(
    run_dir: Path,
    *,
    seed: int,
    num_agents: int,
    num_weather_cells: int,
) -> Path:
    explicit_path = LLM_MEMORY_PATH or LLM_GUIDANCE_MEMORY_PATH
    if explicit_path:
        return Path(str(explicit_path))
    if bool(LLM_GUIDANCE_MEMORY_ISOLATED_PER_RUN):
        if LLM_MEMORY_RUN_ID:
            return Path(run_dir) / "memory" / str(LLM_MEMORY_RUN_ID) / "decision_memory.json"
        return Path(run_dir) / "memory" / "decision_memory.json"
    return Path(DECISION_MEMORY_PATH)


def _format_hparam_value(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        parts = [f"{key}: {_format_hparam_value(val)}" for key, val in sorted(value.items(), key=lambda item: str(item[0]))]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, (list, tuple, set)):
        return "[" + ", ".join(_format_hparam_value(item) for item in value) + "]"
    return repr(value)


def _config_group(name: str) -> str:
    if name.startswith("REWARD_"):
        return "Reward Hyperparameters"
    if name.startswith("MAPPO_"):
        return "MAPPO Hyperparameters"
    if name.startswith("LLM_") or name.startswith("OLLAMA_") or name.startswith("USE_LLM"):
        return "LLM Guidance Hyperparameters"
    if name.startswith("WEATHER_") or name in {"NUM_WEATHER_CELLS_DEFAULT", "MAX_WEATHER_CELLS"}:
        return "Weather Hyperparameters"
    if name.startswith("HAZARD_") or name in {
        "EMERGENCY_LOOKAHEAD_STEPS",
        "TURN_PREVIEW_STEPS",
        "PREVIEW_TOP_K",
        "PAIR_RISK_BUFFER",
        "BOUNDARY_WARNING_DISTANCE_UNITS",
        "CLEAR_STREAK_REQUIRED",
        "ROUTE_RECOVERY_XTRACK_UNITS",
        "DESTINATION_ALIGNMENT_DEG",
        "MERGE_BACK_REPEAT_ERROR_DEG",
        "MERGE_BACK_RECHECK_STEPS",
    }:
        return "Hazard And Guidance Geometry"
    if name.endswith("_DIR") or name.endswith("_PATH") or name.endswith("_CSV") or name.endswith("_GEOJSON"):
        return "Configured Paths"
    if name in {
        "MAX_AGENTS",
        "NUM_AGENTS_DEFAULT",
        "ROUTE_IDS_DEFAULT",
        "SEED",
        "SAFE_R",
        "TRAFFIC_CAUTION_R",
        "LOCAL_TRAFFIC_NEIGHBOR_COUNT",
        "LAUNCH_SEPARATION_R",
        "GOAL_RADIUS",
        "AGENT_SPEED",
        "MAX_STEP",
        "ACTION_BINS",
        "ROUTE_MIRROR_SUFFIX",
    }:
        return "Environment Hyperparameters"
    return "Other Config Values"


def _iter_config_hparams() -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    for name in sorted(dir(config_module)):
        if not name.isupper():
            continue
        value = getattr(config_module, name)
        if callable(value) or name.startswith("__"):
            continue
        grouped.setdefault(_config_group(name), {})[name] = value
    return grouped


def _write_run_hyperparameters_file(
    path: Path,
    *,
    args: argparse.Namespace,
    run_paths: RunPaths,
    llm_memory_path: Path,
    reset_options: Mapping[str, object],
) -> None:
    if path.exists():
        print(f"Run hyperparameters file already exists; leaving unchanged: {path}")
        return
    lines: List[str] = []
    lines.append("Run Hyperparameters")
    lines.append("===================")
    lines.append("")
    lines.append(f"Created UTC: {datetime.utcnow().isoformat()}Z")
    lines.append(f"Python: {sys.version.split()[0]}")
    lines.append("")
    lines.append("Run Paths")
    lines.append("---------")
    run_path_values = {
        "run_dir": run_paths.run_dir,
        "tensorboard_dir": run_paths.tensorboard_dir,
        "tensorboard_run_dir": run_paths.tensorboard_run_dir,
        "weights_dir": run_paths.weights_dir,
        "best_model_path": run_paths.best_model_path,
        "last_ckpt_path": run_paths.last_ckpt_path,
        "llm_memory_path": llm_memory_path,
        "llm_guidance_json_dir": run_paths.run_dir / "llm_guidance_json",
        "llm_guided_mappo_audit_path": run_paths.run_dir / "llm_guided_mappo_audit.jsonl",
    }
    for key, value in run_path_values.items():
        lines.append(f"{key}: {_format_hparam_value(value)}")
    lines.append("")
    lines.append("Command Line Arguments")
    lines.append("----------------------")
    for key, value in sorted(vars(args).items()):
        lines.append(f"{key}: {_format_hparam_value(value)}")
    lines.append("")
    lines.append("Resolved Reset Options")
    lines.append("----------------------")
    for key, value in sorted(reset_options.items()):
        lines.append(f"{key}: {_format_hparam_value(value)}")
    for group_name, values in _iter_config_hparams().items():
        lines.append("")
        lines.append(group_name)
        lines.append("-" * len(group_name))
        for key, value in values.items():
            lines.append(f"{key}: {_format_hparam_value(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Created run hyperparameters file: {path}")


def _llm_return_weight(g_llm: float, g_mappo: float) -> float:
    delta = float(g_llm) - float(g_mappo)
    if delta <= float(LLM_SHADOW_RETURN_MARGIN):
        return 0.0
    scale = max(float(LLM_SHADOW_RETURN_SCALE), 1e-6)
    max_weight = float(LLM_SHADOW_MAX_WEIGHT if LLM_SHADOW_MAX_WEIGHT is not None else LLM_SHADOW_RETURN_WEIGHT_CLIP)
    return max(0.0, min(max_weight, delta / scale))


def _current_llm_loss_weight(agent_steps: int) -> float:
    base = float(LLM_LOSS_WEIGHT)
    if not bool(LLM_LOSS_DECAY_ENABLED):
        return base
    start = int(LLM_GUIDANCE_START_STEP)
    end = int(LLM_GUIDANCE_END_STEP)
    if end <= start:
        return max(float(LLM_LOSS_MIN_WEIGHT), base)
    progress = (float(agent_steps) - float(start)) / max(float(end - start), 1.0)
    progress = max(0.0, min(1.0, progress))
    decayed = base + progress * (float(LLM_LOSS_MIN_WEIGHT) - base)
    return max(float(LLM_LOSS_MIN_WEIGHT), decayed)


def _hybrid_llm_return_weight(
    g_llm: float,
    g_mappo: float,
    g_llm_joint: float,
    g_mappo_joint: float,
) -> tuple[float, float, float]:
    agent_weight = _llm_return_weight(g_llm, g_mappo)
    joint_weight = _llm_return_weight(g_llm_joint, g_mappo_joint)
    hybrid = (
        float(LLM_RETURN_WEIGHT_AGENT_COEF) * agent_weight
        + float(LLM_RETURN_WEIGHT_JOINT_COEF) * joint_weight
    )
    max_weight = float(LLM_SHADOW_MAX_WEIGHT if LLM_SHADOW_MAX_WEIGHT is not None else LLM_SHADOW_RETURN_WEIGHT_CLIP)
    return agent_weight, joint_weight, max(0.0, min(max_weight, hybrid))


@dataclass
class MemoryCorrectionResult:
    updates: Dict[str, bool]
    sources: Dict[str, str]
    note_sources: Dict[str, str]
    attribution_modes: Dict[str, str]
    skipped_count: int = 0


def _clean_memory_note(note: object, *, max_chars: int = 120) -> str:
    text = " ".join(str(note or "").split())
    return text[: int(max_chars)].strip()


def _fallback_memory_correction_note(source: str) -> str:
    source_text = "LLM" if str(source).lower() == "llm" else "MAPPO"
    return f"{source_text} branch improved short-horizon safety over retrieved memory"


def _preview_row_for_turn(rows: object, turn_deg: int) -> Optional[Mapping[str, object]]:
    if not isinstance(rows, (list, tuple)):
        return None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        try:
            if int(row.get("heading_change_deg", 0)) == int(turn_deg):
                return row
        except Exception:
            continue
    return None


def _preview_row_summary(row: Optional[Mapping[str, object]]) -> str:
    if not isinstance(row, Mapping):
        return "unlisted candidate"
    fields = [
        f"turn={int(row.get('heading_change_deg', 0)):+d}",
        f"safe={int(bool(row.get('safe_over_preview', False)))}",
        f"pair_loss={row.get('pair_loss_step')}",
        f"weather_entry={row.get('weather_entry_step')}",
        f"boundary_exit={row.get('boundary_exit_step')}",
        f"progress={row.get('progress_to_destination')}",
    ]
    return ", ".join(fields)


def _generate_memory_correction_note(
    *,
    agent_id: str,
    phase: str,
    call_reason: str,
    original_action_deg: int,
    corrected_action_deg: int,
    source: str,
    g_mappo: float,
    g_llm: float,
    g_memory: float,
    preview_rows: object,
) -> tuple[str, str]:
    original_row = _preview_row_for_turn(preview_rows, int(original_action_deg))
    corrected_row = _preview_row_for_turn(preview_rows, int(corrected_action_deg))
    fallback = _fallback_memory_correction_note(source)
    prompt = (
        "ROLE: You write concise evaluator notes for aircraft guidance memory corrections.\n"
        "Return exactly one JSON object: {\"note\": \"<12-18 word operational reason>\"}.\n"
        "Do not mention rewards or training. Explain why the corrected action is safer/better.\n"
        f"agent={agent_id} phase={phase} reason={call_reason}\n"
        f"original_memory_action={int(original_action_deg):+d} corrected_action={int(corrected_action_deg):+d} source={source}\n"
        f"shadow_returns: MAPPO={float(g_mappo):.3f} LLM={float(g_llm):.3f} MEMORY={float(g_memory):.3f}\n"
        f"original_preview: {_preview_row_summary(original_row)}\n"
        f"corrected_preview: {_preview_row_summary(corrected_row)}\n"
    )
    try:
        from rl_llm_multi.llm import _extract_json, ollama_invoke

        result = ollama_invoke(prompt, image_paths=None)
        if result.get("llm_status") == "ok":
            payload = _extract_json(str(result.get("raw_text", "")))
            note = _clean_memory_note(payload.get("note") if isinstance(payload, dict) else "")
            if note:
                return note, "llm"
    except Exception:
        pass
    return fallback, "fallback"


def _llm_call_payload_from_guidance(guidance: object) -> Dict[str, object]:
    controller = getattr(guidance, "controller", None)
    payload = getattr(controller, "last_normalized", {}) if controller is not None else {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _memory_episode_id_for_guidance(guidance: object, core: object) -> str:
    method = getattr(guidance, "memory_episode_id", None)
    if callable(method):
        value = method(core)
        if value:
            return str(value)
    controller = getattr(guidance, "controller", None)
    method = getattr(controller, "_memory_episode_id", None)
    if callable(method):
        return str(method(core))
    seed = getattr(core, "reset_seed", None)
    seed_text = "none" if seed is None else str(seed)
    return f"llm_guided_mappo__seed_{seed_text}"


def _append_advisory_memory_cases(
    memory_store: Optional[DecisionMemoryStore],
    *,
    guidance: object,
    core: object,
    step: int,
    advice_by_agent: Mapping[str, object],
    shadow_result: Optional[object],
) -> Dict[str, bool]:
    writes: Dict[str, bool] = {}
    if (
        not bool(LLM_ADVISORY_MEMORY_WRITE_ENABLED)
        or memory_store is None
        or shadow_result is None
        or not advice_by_agent
    ):
        return writes

    payload = _llm_call_payload_from_guidance(guidance)
    answer = payload.get("answer") if isinstance(payload.get("answer"), Mapping) else {}
    actions_payload = answer.get("actions") if isinstance(answer.get("actions"), Mapping) else {}
    debug = payload.get("debug") if isinstance(payload.get("debug"), Mapping) else {}
    stage_by_agent = debug.get("stage_by_agent") if isinstance(debug.get("stage_by_agent"), Mapping) else {}
    call_reason_by_agent = (
        debug.get("call_reason_by_agent") if isinstance(debug.get("call_reason_by_agent"), Mapping) else {}
    )
    eligible_agent_ids_raw = debug.get("eligible_agent_ids", [])
    eligible_agent_ids = (
        {str(agent_id) for agent_id in eligible_agent_ids_raw}
        if isinstance(eligible_agent_ids_raw, (list, tuple, set))
        else set()
    )
    threat_rows_by_agent = payload.get("threat_rows_by_agent") if isinstance(payload.get("threat_rows_by_agent"), Mapping) else {}
    preview_rows_by_agent = payload.get("preview_rows_by_agent") if isinstance(payload.get("preview_rows_by_agent"), Mapping) else {}

    actionable: List[Dict[str, str]] = []
    final_actions: Dict[str, Dict[str, object]] = {}
    for agent_id, advice in sorted(advice_by_agent.items()):
        agent_id = str(agent_id)
        if eligible_agent_ids and agent_id not in eligible_agent_ids:
            continue
        g_mappo = float(getattr(shadow_result, "mappo_returns", {}).get(agent_id, 0.0))
        g_llm = float(getattr(shadow_result, "llm_returns", {}).get(agent_id, 0.0))
        if g_llm <= g_mappo + float(LLM_ADVISORY_MEMORY_WRITE_MARGIN):
            continue
        action_payload = actions_payload.get(agent_id, {}) if isinstance(actions_payload, Mapping) else {}
        action_payload = action_payload if isinstance(action_payload, Mapping) else {}
        call_name = str(action_payload.get("call_name") or stage_by_agent.get(agent_id) or getattr(advice, "phase", ""))
        if not call_name:
            continue
        actionable.append(
            {
                "agent_id": agent_id,
                "call_name": call_name,
                "call_reason": str(call_reason_by_agent.get(agent_id, "")),
            }
        )
        requested_action_deg = getattr(advice, "requested_action_deg", None)
        if requested_action_deg is None:
            requested_action_deg = getattr(advice, "llm_action_deg")
        final_actions[agent_id] = {
            "call_name": call_name,
            "heading_change_deg": int(getattr(advice, "llm_action_deg")),
            "requested_heading_change_deg": int(requested_action_deg),
            "rationale": str(action_payload.get("rationale", "")),
            "source": str(action_payload.get("source", getattr(advice, "source", ""))),
        }

    if not actionable:
        return writes
    try:
        cases = memory_store.append_cases(
            core,
            episode_id=_memory_episode_id_for_guidance(guidance, core),
            step=int(step),
            actionable=actionable,
            final_actions=final_actions,
            threat_rows_by_agent=threat_rows_by_agent,
            preview_rows_by_agent=preview_rows_by_agent,
        )
    except Exception:
        return writes
    for case in cases:
        if isinstance(case, Mapping) and case.get("agent") is not None:
            writes[str(case["agent"])] = True
    return writes


def _write_jsonl_rows_with_cap(path: Path, rows: Sequence[Mapping[str, object]], max_rows: Optional[int]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [json.dumps(dict(row), sort_keys=True) for row in rows]
    if max_rows is None:
        with path.open("a", encoding="utf-8") as handle:
            for line in serialized:
                handle.write(line + "\n")
        return
    limit = int(max_rows)
    if limit <= 0:
        return
    existing = []
    if path.exists():
        existing = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = (existing + serialized)[-limit:]
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def _append_llm_training_audit(
    audit_path: Path,
    *,
    episode: int,
    env_step: int,
    agent_steps: int,
    memory_path: Path,
    policy_actions: Mapping[str, int],
    executed_actions: Mapping[str, int],
    advice_by_agent: Mapping[str, object],
    llm_metadata_by_agent: Mapping[str, Mapping[str, object]],
    memory_update_sources: Mapping[str, str],
) -> None:
    if not bool(LLM_GUIDED_MAPPO_AUDIT_ENABLED) or not advice_by_agent:
        return
    llm_global_weight = _current_llm_loss_weight(agent_steps)
    rows: List[Dict[str, object]] = []
    for agent_id, advice in sorted(advice_by_agent.items()):
        agent_id = str(agent_id)
        if agent_id not in policy_actions:
            continue
        policy_idx = int(policy_actions[agent_id])
        executed_idx = int(executed_actions.get(agent_id, policy_idx))
        metadata = dict(llm_metadata_by_agent.get(agent_id, {}))
        rows.append({
            "episode": int(episode),
            "env_step": int(env_step),
            "agent_steps_before_step": int(agent_steps),
            "agent_id": agent_id,
            "mappo_action_idx": policy_idx,
            "mappo_action_deg": action_idx_to_deg(policy_idx, ACTION_BINS),
            "real_action_idx": executed_idx,
            "real_action_deg": action_idx_to_deg(executed_idx, ACTION_BINS),
            "real_action_equals_mappo": bool(executed_idx == policy_idx),
            "llm_label_only": True,
            "llm_action_idx": int(getattr(advice, "llm_action_idx")),
            "llm_action_deg": int(getattr(advice, "llm_action_deg")),
            "llm_phase": str(metadata.get("llm_phase", "")),
            "ppo_action_logprob_source": "mappo_sampled_action",
            "llm_loss_label_source": "llm_action_label",
            "llm_loss_global_weight": llm_global_weight,
            "llm_return_weight": float(metadata.get("llm_return_weight", 0.0) or 0.0),
            "llm_return_weight_agent": float(metadata.get("llm_return_weight_agent", 0.0) or 0.0),
            "llm_return_weight_joint": float(metadata.get("llm_return_weight_joint", 0.0) or 0.0),
            "llm_return_weight_hybrid": float(metadata.get("llm_return_weight_hybrid", 0.0) or 0.0),
            "shadow_G_MAPPO": metadata.get("G_MAPPO_shadow"),
            "shadow_G_LLM": metadata.get("G_LLM_shadow"),
            "shadow_G_MEMORY": metadata.get("G_MEMORY_shadow"),
            "shadow_G_MAPPO_joint": metadata.get("G_MAPPO_joint_shadow"),
            "shadow_G_LLM_joint": metadata.get("G_LLM_joint_shadow"),
            "shadow_G_MEMORY_joint": metadata.get("G_MEMORY_joint_shadow"),
            "memory_path": str(memory_path),
            "memory_ref": getattr(advice, "memory_ref", None),
            "normal_memory_write_allowed": False,
            "advisory_memory_written": bool(metadata.get("advisory_memory_written", False)),
            "memory_correction_updated": bool(metadata.get("memory_update_applied", False)),
            "memory_correction_source": str(memory_update_sources.get(agent_id, "")),
            "memory_attribution_mode": str(metadata.get("memory_attribution_mode", "")),
            "memory_correction_note_source": str(metadata.get("memory_correction_note_source", "")),
        })
    max_rows = None if LLM_GUIDED_MAPPO_AUDIT_MAX_ROWS is None else int(LLM_GUIDED_MAPPO_AUDIT_MAX_ROWS)
    _write_jsonl_rows_with_cap(audit_path, rows, max_rows)


def _build_llm_metadata_by_agent(
    advice_by_agent: Mapping[str, object],
    shadow_result: Optional[object],
    memory_corrections: MemoryCorrectionResult,
    advisory_memory_writes: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Dict[str, object]]:
    metadata: Dict[str, Dict[str, object]] = {}
    advisory_memory_writes = advisory_memory_writes or {}
    g_mappo_joint = float(getattr(shadow_result, "mappo_joint_return", 0.0) if shadow_result is not None else 0.0)
    g_llm_joint = float(getattr(shadow_result, "llm_joint_return", 0.0) if shadow_result is not None else 0.0)
    g_memory_joint = float(getattr(shadow_result, "memory_joint_return", 0.0) if shadow_result is not None else 0.0)
    for agent_id, advice in advice_by_agent.items():
        agent_id = str(agent_id)
        g_mappo = 0.0
        g_llm = 0.0
        g_memory = 0.0
        if shadow_result is not None:
            g_mappo = float(getattr(shadow_result, "mappo_returns", {}).get(agent_id, 0.0))
            g_llm = float(getattr(shadow_result, "llm_returns", {}).get(agent_id, 0.0))
            g_memory = float(getattr(shadow_result, "memory_returns", {}).get(agent_id, 0.0))
        agent_weight, joint_weight, hybrid_weight = _hybrid_llm_return_weight(
            g_llm,
            g_mappo,
            g_llm_joint,
            g_mappo_joint,
        )
        metadata[agent_id] = {
            "llm_label_available": True,
            "llm_action_idx": int(getattr(advice, "llm_action_idx")),
            "llm_action_deg": int(getattr(advice, "llm_action_deg")),
            "llm_phase": str(getattr(advice, "phase", "")),
            "G_MAPPO_shadow": g_mappo,
            "G_LLM_shadow": g_llm,
            "G_MEMORY_shadow": g_memory,
            "G_MAPPO_joint_shadow": g_mappo_joint,
            "G_LLM_joint_shadow": g_llm_joint,
            "G_MEMORY_joint_shadow": g_memory_joint,
            "llm_return_weight_agent": agent_weight,
            "llm_return_weight_joint": joint_weight,
            "llm_return_weight_hybrid": hybrid_weight,
            "llm_return_weight": hybrid_weight,
            "memory_action_idx": getattr(advice, "memory_action_idx", None) or 0,
            "memory_action_deg": getattr(advice, "memory_action_deg", None) or 0,
            "memory_ref": getattr(advice, "memory_ref", None),
            "memory_update_applied": bool(memory_corrections.updates.get(agent_id, False)),
            "memory_attribution_mode": str(memory_corrections.attribution_modes.get(agent_id, "")),
            "memory_correction_note_source": str(memory_corrections.note_sources.get(agent_id, "")),
            "advisory_memory_written": bool(advisory_memory_writes.get(agent_id, False)),
        }
    return metadata


def _apply_memory_corrections(
    memory_store: Optional[DecisionMemoryStore],
    *,
    advice_by_agent: Mapping[str, object],
    shadow_result: Optional[object],
    policy_actions: Mapping[str, int],
    llm_call_payload: Optional[Mapping[str, object]] = None,
) -> MemoryCorrectionResult:
    result = MemoryCorrectionResult(updates={}, sources={}, note_sources={}, attribution_modes={})
    if not bool(LLM_MEMORY_EVALUATOR_ENABLED) or memory_store is None or shadow_result is None:
        return result
    memory_agent_ids = [str(agent_id) for agent_id in (getattr(shadow_result, "memory_agent_ids", []) or [])]
    if not memory_agent_ids:
        return result

    payload = dict(llm_call_payload or {})
    debug = payload.get("debug") if isinstance(payload.get("debug"), Mapping) else {}
    call_reason_by_agent = (
        debug.get("call_reason_by_agent") if isinstance(debug.get("call_reason_by_agent"), Mapping) else {}
    )
    preview_rows_by_agent = payload.get("preview_rows_by_agent") if isinstance(payload.get("preview_rows_by_agent"), Mapping) else {}
    memory_agent_set = set(memory_agent_ids)
    g_mappo_joint = float(getattr(shadow_result, "mappo_joint_return", 0.0))
    g_llm_joint = float(getattr(shadow_result, "llm_joint_return", 0.0))
    g_memory_joint = float(getattr(shadow_result, "memory_joint_return", 0.0))
    attribution_joint_returns = getattr(shadow_result, "memory_attribution_joint_returns", {}) or {}

    for agent_id, advice in advice_by_agent.items():
        agent_id = str(agent_id)
        memory_ref = getattr(advice, "memory_ref", None)
        memory_action_deg = getattr(advice, "memory_action_deg", None)
        if not isinstance(memory_ref, dict) or memory_action_deg is None or agent_id not in memory_agent_set:
            continue

        if len(memory_agent_ids) == 1:
            attribution_mode = "single"
            g_memory_compare = g_memory_joint
        else:
            attribution_mode = "one_at_a_time"
            if agent_id not in attribution_joint_returns:
                result.skipped_count += 1
                continue
            g_memory_compare = float(attribution_joint_returns[agent_id])
        result.attribution_modes[agent_id] = attribution_mode

        update_mode = str(LLM_MEMORY_UPDATE_WITH_BEST_OF).lower()
        if update_mode == "llm":
            best_return = g_llm_joint
            corrected = int(getattr(advice, "llm_action_deg"))
            source = "llm"
        elif update_mode == "mappo":
            if agent_id not in policy_actions:
                result.skipped_count += 1
                continue
            best_return = g_mappo_joint
            corrected = action_idx_to_deg(int(policy_actions[agent_id]), ACTION_BINS)
            source = "mappo"
        elif g_llm_joint >= g_mappo_joint:
            best_return = g_llm_joint
            corrected = int(getattr(advice, "llm_action_deg"))
            source = "llm"
        else:
            if agent_id not in policy_actions:
                result.skipped_count += 1
                continue
            best_return = g_mappo_joint
            corrected = action_idx_to_deg(int(policy_actions[agent_id]), ACTION_BINS)
            source = "mappo"

        if best_return <= g_memory_compare + float(LLM_MEMORY_UPDATE_MARGIN):
            result.skipped_count += 1
            continue

        note, note_source = _generate_memory_correction_note(
            agent_id=agent_id,
            phase=str(getattr(advice, "phase", "")),
            call_reason=str(call_reason_by_agent.get(agent_id, "")),
            original_action_deg=int(memory_action_deg),
            corrected_action_deg=int(corrected),
            source=source,
            g_mappo=g_mappo_joint,
            g_llm=g_llm_joint,
            g_memory=g_memory_compare,
            preview_rows=preview_rows_by_agent.get(agent_id, []) if isinstance(preview_rows_by_agent, Mapping) else [],
        )
        applied = memory_store.update_case_result(memory_ref, corrected=corrected, note=note)
        result.updates[agent_id] = bool(applied)
        if applied:
            result.sources[agent_id] = source
            result.note_sources[agent_id] = note_source
        else:
            result.skipped_count += 1
    return result


def _log_eval(writer: SummaryWriter, stats: Dict[str, float], agent_steps: int) -> None:
    for key, value in stats.items():
        writer.add_scalar(f"eval/{key}", float(value), agent_steps)


def _eval_score(stats: Dict[str, float]) -> float:
    return float(stats.get("total_episode_reward", float("-inf")))


def _mean_mapping_value(mapping: object) -> Optional[float]:
    if not isinstance(mapping, dict) or not mapping:
        return None
    values = [float(value) for value in mapping.values()]
    return sum(values) / float(len(values))


def _mean_agent_value(value: object, agent_ids: Sequence[str]) -> Optional[float]:
    agent_ids = [str(agent_id) for agent_id in agent_ids]
    if not agent_ids:
        return None
    if isinstance(value, dict):
        values = [
            float(value[agent_id])
            for agent_id in agent_ids
            if agent_id in value
        ]
        return sum(values) / float(len(values)) if values else None
    if value is None:
        return None
    return float(value)


ESSENTIAL_REWARD_COMPONENT_KEYS = (
    "raw_progress",
    "progress",
    "safe_progress",
    "progress_risk_gate",
    "progress_safety_scale",
    "cross_track",
    "cross_track_penalty_scale",
    "relaxed_cross_track",
    "raw_cross_track_recovery",
    "cross_track_recovery",
    "safe_cross_track_recovery",
    "merge_back_scale",
    "merge_back_bonus",
    "heading_error_to_destination",
    "heading_error_norm",
    "safe_heading_penalty",
    "previous_traffic_risk",
    "previous_local_traffic_risk",
    "traffic_step_risk",
    "traffic_separation_risk",
    "predicted_traffic_risk",
    "local_traffic_risk",
    "traffic_risk",
    "combined_traffic_risk_reduction",
    "local_traffic_risk_reduction",
    "traffic_risk_reduction",
    "previous_weather_risk",
    "weather_risk",
    "weather_risk_reduction",
    "finish_bonus",
    "dense_reward",
)


def _accumulate_reward_debug(
    common: Dict[str, object],
    sums: Dict[str, float],
    counts: Dict[str, int],
) -> None:
    dense_mean = _mean_mapping_value(common.get("agent_dense_rewards"))
    if dense_mean is not None:
        sums["agent_dense_mean"] = sums.get("agent_dense_mean", 0.0) + float(dense_mean)
        counts["agent_dense_mean"] = counts.get("agent_dense_mean", 0) + 1
    components = common.get("reward_components")
    if not isinstance(components, dict) or not components:
        return

    for key in ESSENTIAL_REWARD_COMPONENT_KEYS:
        values = [
            float(parts[key])
            for parts in components.values()
            if isinstance(parts, dict) and key in parts
        ]
        if values:
            sums[key] = sums.get(key, 0.0) + sum(values) / float(len(values))
            counts[key] = counts.get(key, 0) + 1


def _log_episode_reward_debug(
    writer: SummaryWriter,
    sums: Dict[str, float],
    counts: Dict[str, int],
    common: Dict[str, object],
    episode: int,
) -> None:
    terminal_reward = float(common.get("terminal_reward", 0.0))
    writer.add_scalar("train/reward/terminal_episode", terminal_reward, episode)
    dense_count = int(counts.get("agent_dense_mean", 0))
    writer.add_scalar(
        "train/reward/mean_step_behavior_reward",
        (sums.get("agent_dense_mean", 0.0) / float(dense_count)) if dense_count > 0 else 0.0,
        episode,
    )
    for key in ESSENTIAL_REWARD_COMPONENT_KEYS:
        count = int(counts.get(key, 0))
        writer.add_scalar(
            f"train/reward_component/{key}_mean",
            (sums.get(key, 0.0) / float(count)) if count > 0 else 0.0,
            episode,
        )


def main() -> None:
    args = parse_args()
    route_ids = _normalize_route_ids(args.route_ids)
    reset_options = {
        "num_agents": int(args.num_agents),
        "route_ids": route_ids,
        "num_weather_cells": int(args.num_weather_cells),
    }
    run_paths = _resolve_run_paths(args)
    log_dir = run_paths.run_dir
    tensorboard_dir = run_paths.tensorboard_dir
    tensorboard_run_dir = run_paths.tensorboard_run_dir
    weights_dir = run_paths.weights_dir
    best_model_path = run_paths.best_model_path
    last_ckpt_path = run_paths.last_ckpt_path
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    last_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    probe_env = make_env(
        num_agents=args.num_agents,
        num_weather_cells=args.num_weather_cells,
        episode_step_cap=args.episode_step_cap,
    )
    probe_env.reset(seed=args.seed, options=reset_options)
    global_state_dim = int(probe_env.state().shape[0])
    local_obs_dim = int(probe_env.observation_space(probe_env.possible_agents[0]).shape[0])
    agent = build_agent(global_state_dim, local_obs_dim)
    llm_memory_path = _resolve_llm_memory_path(
        log_dir,
        seed=int(args.seed),
        num_agents=int(args.num_agents),
        num_weather_cells=int(args.num_weather_cells),
    )
    _write_run_hyperparameters_file(
        log_dir / "run_hyperparameters.txt",
        args=args,
        run_paths=run_paths,
        llm_memory_path=llm_memory_path,
        reset_options=reset_options,
    )
    llm_audit_path = log_dir / "llm_guided_mappo_audit.jsonl"
    if USE_LLM_GUIDED_TRAINING:
        guidance = LLMGuidanceProvider(
            save_dir=log_dir / "llm_guidance_json",
            memory_path=llm_memory_path,
            use_vision=bool(LLM_GUIDANCE_USE_VISION),
            memory_visual_audit_enabled=bool(DECISION_MEMORY_VISUAL_AUDIT_ENABLED),
            memory_visual_audit_dir=DECISION_MEMORY_VISUAL_AUDIT_DIR,
            memory_visual_audit_frame_count=int(DECISION_MEMORY_VISUAL_AUDIT_FRAME_COUNT),
        )
        memory_store: Optional[DecisionMemoryStore] = DecisionMemoryStore(llm_memory_path, max_episodes=LLM_GUIDED_MEMORY_MAX_EPISODES)
    else:
        guidance = NoGuidanceProvider()
        memory_store = None

    agent_steps = 0
    env_steps = 0
    episode = 0
    if not args.no_resume and last_ckpt_path.exists():
        try:
            agent_steps, env_steps, episode = agent.load_checkpoint(last_ckpt_path)
            print(
                f"Resumed MAPPO checkpoint from {last_ckpt_path} | "
                f"agent_steps={agent_steps} env_steps={env_steps} episode={episode}"
            )
        except Exception as exc:
            print(f"Failed to load checkpoint {last_ckpt_path}: {exc}. Starting fresh.")

    writer = SummaryWriter(str(tensorboard_run_dir), purge_step=agent_steps)
    print(f"Run directory: {log_dir}")
    print(f"TensorBoard directory: {tensorboard_dir}")
    print(f"TensorBoard run directory: {tensorboard_run_dir}")
    print(f"Weights directory: {weights_dir}")
    print(f"Guided memory path: {llm_memory_path}")
    best_eval_stats: Optional[Dict[str, float]] = None
    next_update = agent_steps + int(args.update_agent_steps)
    next_eval = agent_steps + int(args.eval_freq)
    next_save = agent_steps + int(args.save_model_freq)
    start_time = datetime.now()

    failure_counts = {"collision": 0, "weather": 0, "truncated": 0, "success": 0}
    print(
        "Starting MAPPO training | num_agents={} weather_cells={} global_dim={} local_dim={} max_agent_steps={}".format(
            args.num_agents,
            args.num_weather_cells,
            global_state_dim,
            local_obs_dim,
            args.max_agent_steps,
        )
    )

    def make_eval_env() -> MultiAgentParallelEnv:
        return make_env(
            num_agents=args.num_agents,
            num_weather_cells=args.num_weather_cells,
            episode_step_cap=args.episode_step_cap,
        )

    try:
        while agent_steps < int(args.max_agent_steps):
            env = make_env(
                num_agents=args.num_agents,
                num_weather_cells=args.num_weather_cells,
                episode_step_cap=args.episode_step_cap,
            )
            observations, info = env.reset(seed=int(args.seed) + int(episode), options=reset_options)
            guidance.reset()
            episode_llm_guidance_active = _llm_guidance_active_for_episode(agent_steps)
            done = bool(info["__common__"]["episode_done"])
            truncated = bool(info["__common__"]["episode_truncated"])
            ep_return = 0.0
            ep_policy_controlled_return = 0.0
            ep_pre_sector_return = 0.0
            ep_env_steps = 0
            ep_reward_sums: Dict[str, float] = {}
            ep_reward_counts: Dict[str, int] = {}
            ep_policy_agent_ids: set[str] = set()
            ep_llm_guided_decisions = 0
            ep_llm_positive_labels = 0
            ep_llm_action_matches = 0
            ep_llm_abs_turn_difference_sum = 0.0
            ep_shadow_mappo_returns: List[float] = []
            ep_shadow_llm_returns: List[float] = []
            ep_shadow_memory_returns: List[float] = []
            ep_shadow_mappo_joint_returns: List[float] = []
            ep_shadow_llm_joint_returns: List[float] = []
            ep_shadow_memory_joint_returns: List[float] = []
            ep_shadow_memory_mappo_joint_returns: List[float] = []
            ep_llm_agent_weights: List[float] = []
            ep_llm_joint_weights: List[float] = []
            ep_llm_hybrid_weights: List[float] = []
            ep_memory_retrieval_count = 0
            ep_memory_update_count = 0
            ep_memory_update_to_mappo = 0
            ep_memory_update_to_llm = 0
            ep_memory_attribution_skipped_count = 0
            ep_memory_note_llm_count = 0
            ep_memory_note_fallback_count = 0
            ep_advisory_memory_write_count = 0
            ep_phase_counts = {
                "EXECUTE_TURN": 0,
                "EMERGENCY_MANEUVER": 0,
                "MERGE_BACK": 0,
            }

            while not done and not truncated:
                active_ids = list(env.agents)
                if not active_ids:
                    observations, _, _, _, info = env.step({})
                    common = info["__common__"]
                    team_reward = float(common["team_reward"])
                    done = bool(common["episode_done"])
                    truncated = bool(common["episode_truncated"])
                    if done or truncated:
                        agent.add_terminal_bonus(
                            episode,
                            common.get("agent_terminal_rewards", common.get("terminal_reward", 0.0)),
                        )
                        terminal_policy_reward = _mean_agent_value(
                            common.get("agent_terminal_rewards", common.get("terminal_reward", 0.0)),
                            sorted(ep_policy_agent_ids),
                        )
                        if terminal_policy_reward is not None:
                            ep_policy_controlled_return += float(terminal_policy_reward)
                    env_steps += 1
                    ep_env_steps += 1
                    ep_return += team_reward
                    ep_pre_sector_return += team_reward
                    _accumulate_reward_debug(common, ep_reward_sums, ep_reward_counts)
                    continue

                policy_actions, records, _entropy = agent.select_actions(
                    global_state=env.state(),
                    local_observations=observations,
                    active_agent_ids=active_ids,
                    episode_id=episode,
                    deterministic=False,
                    store=True,
                )
                advice_by_agent: Dict[str, object] = {}
                shadow_result = None
                memory_corrections = MemoryCorrectionResult(updates={}, sources={}, note_sources={}, attribution_modes={})
                memory_update_sources: Dict[str, str] = {}
                advisory_memory_writes: Dict[str, bool] = {}
                llm_metadata_by_agent: Dict[str, Dict[str, object]] = {}
                llm_call_payload: Dict[str, object] = {}
                if episode_llm_guidance_active:
                    advice_by_agent = guidance.advice_for_step(
                        env.core,
                        step=env.core.n_step,
                        proposed_actions=policy_actions,
                        latest_frame_path=None,
                    )
                    llm_call_payload = _llm_call_payload_from_guidance(guidance)
                    if advice_by_agent and bool(LLM_SHADOW_EVAL_ENABLED):
                        shadow_result = shadow_evaluate_actions(
                            env.core,
                            agent,
                            proposed_actions=policy_actions,
                            guidance_by_agent=advice_by_agent,
                            horizon=int(LLM_SHADOW_HORIZON),
                            gamma=float(LLM_SHADOW_GAMMA),
                            episode_id=episode,
                        )
                        memory_corrections = _apply_memory_corrections(
                            memory_store,
                            advice_by_agent=advice_by_agent,
                            shadow_result=shadow_result,
                            policy_actions=policy_actions,
                            llm_call_payload=llm_call_payload,
                        )
                        memory_update_sources = memory_corrections.sources
                        advisory_memory_writes = _append_advisory_memory_cases(
                            memory_store,
                            guidance=guidance,
                            core=env.core,
                            step=env.core.n_step,
                            advice_by_agent=advice_by_agent,
                            shadow_result=shadow_result,
                        )
                    llm_metadata_by_agent = _build_llm_metadata_by_agent(
                        advice_by_agent,
                        shadow_result,
                        memory_corrections,
                        advisory_memory_writes,
                    )

                    if shadow_result is not None:
                        ep_shadow_mappo_joint_returns.append(float(getattr(shadow_result, "mappo_joint_return", 0.0)))
                        ep_shadow_llm_joint_returns.append(float(getattr(shadow_result, "llm_joint_return", 0.0)))
                        if getattr(shadow_result, "memory_agent_ids", []):
                            ep_shadow_memory_joint_returns.append(float(getattr(shadow_result, "memory_joint_return", 0.0)))
                            ep_shadow_memory_mappo_joint_returns.append(float(getattr(shadow_result, "mappo_joint_return", 0.0)))

                    for agent_id, advice in advice_by_agent.items():
                        agent_id = str(agent_id)
                        metadata = llm_metadata_by_agent.get(agent_id, {})
                        ep_llm_guided_decisions += 1
                        ep_llm_agent_weights.append(float(metadata.get("llm_return_weight_agent", 0.0) or 0.0))
                        ep_llm_joint_weights.append(float(metadata.get("llm_return_weight_joint", 0.0) or 0.0))
                        ep_llm_hybrid_weights.append(float(metadata.get("llm_return_weight", 0.0) or 0.0))
                        if float(metadata.get("llm_return_weight", 0.0) or 0.0) > 0.0:
                            ep_llm_positive_labels += 1
                        if int(policy_actions.get(agent_id, -1)) == int(getattr(advice, "llm_action_idx")):
                            ep_llm_action_matches += 1
                        ep_llm_abs_turn_difference_sum += abs(
                            action_idx_to_deg(int(policy_actions[agent_id]), ACTION_BINS)
                            - int(getattr(advice, "llm_action_deg"))
                        )
                        phase = str(getattr(advice, "phase", ""))
                        if phase in ep_phase_counts:
                            ep_phase_counts[phase] += 1
                        if getattr(advice, "memory_ref", None):
                            ep_memory_retrieval_count += 1
                        if shadow_result is not None:
                            ep_shadow_mappo_returns.append(
                                float(getattr(shadow_result, "mappo_returns", {}).get(agent_id, 0.0))
                            )
                            ep_shadow_llm_returns.append(
                                float(getattr(shadow_result, "llm_returns", {}).get(agent_id, 0.0))
                            )
                            if agent_id in getattr(shadow_result, "memory_returns", {}):
                                ep_shadow_memory_returns.append(
                                    float(getattr(shadow_result, "memory_returns", {}).get(agent_id, 0.0))
                                )
                    ep_memory_update_count += len(memory_update_sources)
                    ep_memory_update_to_mappo += sum(1 for value in memory_update_sources.values() if value == "mappo")
                    ep_memory_update_to_llm += sum(1 for value in memory_update_sources.values() if value == "llm")
                    ep_memory_attribution_skipped_count += int(memory_corrections.skipped_count)
                    ep_memory_note_llm_count += sum(1 for value in memory_corrections.note_sources.values() if value == "llm")
                    ep_memory_note_fallback_count += sum(1 for value in memory_corrections.note_sources.values() if value == "fallback")
                    ep_advisory_memory_write_count += sum(1 for value in advisory_memory_writes.values() if value)

                final_actions = dict(policy_actions)
                if advice_by_agent:
                    _append_llm_training_audit(
                        llm_audit_path,
                        episode=episode,
                        env_step=env_steps,
                        agent_steps=agent_steps,
                        memory_path=llm_memory_path,
                        policy_actions=policy_actions,
                        executed_actions=final_actions,
                        advice_by_agent=advice_by_agent,
                        llm_metadata_by_agent=llm_metadata_by_agent,
                        memory_update_sources=memory_update_sources,
                    )
                observations, _, _, _, info = env.step(final_actions)
                common = info["__common__"]
                team_reward = float(common["team_reward"])
                done = bool(common["episode_done"])
                truncated = bool(common["episode_truncated"])
                ep_policy_agent_ids.update(active_ids)
                post_active = set(env.agents)
                team_terminal = bool(done or truncated)
                terminals = {
                    agent_id: bool(team_terminal or agent_id not in post_active)
                    for agent_id in active_ids
                }
                controlled_reward_source = common.get("agent_dense_rewards", common.get("agent_rewards", team_reward))
                step_policy_reward = _mean_agent_value(controlled_reward_source, active_ids)
                if step_policy_reward is not None:
                    ep_policy_controlled_return += float(step_policy_reward)
                agent.store_outcomes(
                    records,
                    reward=controlled_reward_source,
                    terminals=terminals,
                    llm_metadata_by_agent=llm_metadata_by_agent,
                )
                if team_terminal:
                    agent.add_terminal_bonus(
                        episode,
                        common.get("agent_terminal_rewards", common.get("terminal_reward", 0.0)),
                    )
                    terminal_policy_reward = _mean_agent_value(
                        common.get("agent_terminal_rewards", common.get("terminal_reward", 0.0)),
                        sorted(ep_policy_agent_ids),
                    )
                    if terminal_policy_reward is not None:
                        ep_policy_controlled_return += float(terminal_policy_reward)

                step_agent_count = len(records)
                agent_steps += step_agent_count
                env_steps += 1
                ep_env_steps += 1
                ep_return += team_reward

                _accumulate_reward_debug(common, ep_reward_sums, ep_reward_counts)

            common = info["__common__"]
            failure_reason = common["failure_reason"] or "none"
            if bool(common["episode_success"]):
                failure_counts["success"] += 1
            elif failure_reason in failure_counts:
                failure_counts[failure_reason] += 1

            episode += 1
            total_episodes = max(episode, 1)
            collision_or_weather_failures = failure_counts["collision"] + failure_counts["weather"]
            writer.add_scalar("train/reward/episode_total_return", ep_return, episode)
            writer.add_scalar("train/reward/policy_controlled_return", ep_policy_controlled_return, episode)
            writer.add_scalar("train/reward/pre_sector_return", ep_pre_sector_return, episode)
            writer.add_scalar("train/episode_env_steps", ep_env_steps, episode)
            writer.add_scalar("train/success", int(bool(common["episode_success"])), episode)
            writer.add_scalar("train/collision", int(failure_reason == "collision"), episode)
            writer.add_scalar("train/weather_failure", int(failure_reason == "weather"), episode)
            writer.add_scalar("train/truncated", int(failure_reason == "truncated"), episode)
            writer.add_scalar(
                "train/success_rate_running",
                failure_counts["success"] / total_episodes,
                agent_steps,
            )
            writer.add_scalar(
                "train/collision_or_weather_rate_running",
                collision_or_weather_failures / total_episodes,
                agent_steps,
            )
            writer.add_scalar(
                "train/truncation_rate_running",
                failure_counts["truncated"] / total_episodes,
                agent_steps,
            )
            _log_episode_reward_debug(writer, ep_reward_sums, ep_reward_counts, common, episode)
            writer.add_scalar("train/llm/enabled", int(bool(USE_LLM_GUIDED_TRAINING)), agent_steps)
            writer.add_scalar("train/llm/num_guided_decisions", ep_llm_guided_decisions, agent_steps)
            writer.add_scalar("train/llm/positive_label_count", ep_llm_positive_labels, agent_steps)
            writer.add_scalar(
                "train/llm/action_match_rate",
                ep_llm_action_matches / max(float(ep_llm_guided_decisions), 1.0),
                agent_steps,
            )
            writer.add_scalar(
                "train/llm/mean_abs_turn_difference",
                ep_llm_abs_turn_difference_sum / max(float(ep_llm_guided_decisions), 1.0),
                agent_steps,
            )
            writer.add_scalar(
                "train/llm/return_weight_agent_mean",
                sum(ep_llm_agent_weights) / max(float(len(ep_llm_agent_weights)), 1.0),
                agent_steps,
            )
            writer.add_scalar(
                "train/llm/return_weight_joint_mean",
                sum(ep_llm_joint_weights) / max(float(len(ep_llm_joint_weights)), 1.0),
                agent_steps,
            )
            writer.add_scalar(
                "train/llm/return_weight_hybrid_mean",
                sum(ep_llm_hybrid_weights) / max(float(len(ep_llm_hybrid_weights)), 1.0),
                agent_steps,
            )
            writer.add_scalar(
                "train/shadow/G_MAPPO_mean",
                sum(ep_shadow_mappo_returns) / max(float(len(ep_shadow_mappo_returns)), 1.0),
                agent_steps,
            )
            writer.add_scalar(
                "train/shadow/G_LLM_mean",
                sum(ep_shadow_llm_returns) / max(float(len(ep_shadow_llm_returns)), 1.0),
                agent_steps,
            )
            writer.add_scalar(
                "train/shadow/G_MEMORY_mean",
                sum(ep_shadow_memory_returns) / max(float(len(ep_shadow_memory_returns)), 1.0),
                agent_steps,
            )
            writer.add_scalar(
                "train/shadow/G_MAPPO_joint_mean",
                sum(ep_shadow_mappo_joint_returns) / max(float(len(ep_shadow_mappo_joint_returns)), 1.0),
                agent_steps,
            )
            writer.add_scalar(
                "train/shadow/G_LLM_joint_mean",
                sum(ep_shadow_llm_joint_returns) / max(float(len(ep_shadow_llm_joint_returns)), 1.0),
                agent_steps,
            )
            writer.add_scalar(
                "train/shadow/G_MEMORY_joint_mean",
                sum(ep_shadow_memory_joint_returns) / max(float(len(ep_shadow_memory_joint_returns)), 1.0),
                agent_steps,
            )
            writer.add_scalar(
                "train/shadow/llm_beats_mappo_rate",
                sum(
                    1
                    for g_llm, g_mappo in zip(ep_shadow_llm_returns, ep_shadow_mappo_returns)
                    if g_llm > g_mappo
                )
                / max(float(len(ep_shadow_llm_returns)), 1.0),
                agent_steps,
            )
            writer.add_scalar(
                "train/shadow/memory_beats_mappo_rate",
                sum(
                    1
                    for g_memory, g_mappo in zip(ep_shadow_memory_joint_returns, ep_shadow_memory_mappo_joint_returns)
                    if g_memory > g_mappo
                )
                / max(float(len(ep_shadow_memory_joint_returns)), 1.0),
                agent_steps,
            )
            writer.add_scalar(
                "train/shadow/mappo_beats_memory_rate",
                sum(
                    1
                    for g_memory, g_mappo in zip(ep_shadow_memory_joint_returns, ep_shadow_memory_mappo_joint_returns)
                    if g_mappo > g_memory
                )
                / max(float(len(ep_shadow_memory_joint_returns)), 1.0),
                agent_steps,
            )
            writer.add_scalar("train/memory/retrieval_count", ep_memory_retrieval_count, agent_steps)
            writer.add_scalar("train/memory/update_count", ep_memory_update_count, agent_steps)
            writer.add_scalar("train/memory/corrected_to_mappo_count", ep_memory_update_to_mappo, agent_steps)
            writer.add_scalar("train/memory/corrected_to_llm_count", ep_memory_update_to_llm, agent_steps)
            writer.add_scalar("train/memory/attribution_skipped_count", ep_memory_attribution_skipped_count, agent_steps)
            writer.add_scalar("train/memory/correction_note_llm_count", ep_memory_note_llm_count, agent_steps)
            writer.add_scalar("train/memory/correction_note_fallback_count", ep_memory_note_fallback_count, agent_steps)
            writer.add_scalar("train/memory/advisory_write_count", ep_advisory_memory_write_count, agent_steps)
            writer.add_scalar("train/llm/execute_turn_count", ep_phase_counts["EXECUTE_TURN"], agent_steps)
            writer.add_scalar("train/llm/emergency_maneuver_count", ep_phase_counts["EMERGENCY_MANEUVER"], agent_steps)
            writer.add_scalar("train/llm/merge_back_count", ep_phase_counts["MERGE_BACK"], agent_steps)

            if agent_steps >= next_update and len(agent.buffer) > 0:
                metrics = agent.update(llm_loss_weight=_current_llm_loss_weight(agent_steps))
                for key, value in metrics.items():
                    writer.add_scalar(f"loss/{key}", value, agent_steps)
                agent.save_checkpoint(
                    last_ckpt_path,
                    agent_steps=agent_steps,
                    env_steps=env_steps,
                    episode=episode,
                )
                next_update = agent_steps + int(args.update_agent_steps)

            if agent_steps >= next_eval:
                stats = evaluate_policy(
                    agent,
                    make_eval_env,
                    n_episodes=int(args.n_eval_episodes),
                    seed=int(args.seed) + 10_000,
                    reset_options=reset_options,
                )
                _log_eval(writer, stats, agent_steps)
                agent.save_checkpoint(
                    last_ckpt_path,
                    agent_steps=agent_steps,
                    env_steps=env_steps,
                    episode=episode,
                )
                if best_eval_stats is None or _eval_score(stats) > _eval_score(best_eval_stats):
                    best_eval_stats = dict(stats)
                    agent.save_model(best_model_path)
                    print(
                        f"[{agent_steps}] Best MAPPO model saved to {best_model_path} "
                        f"total_episode_reward={stats['total_episode_reward']:.3f} "
                        f"success={stats['success_rate']:.3f} "
                        f"collision={stats['collision_rate']:.3f} "
                        f"weather={stats['weather_rate']:.3f} "
                        f"truncated={stats['truncation_rate']:.3f} "
                        f"mean_episode_total_reward={stats['mean_episode_total_reward']:.3f}"
                    )
                next_eval = agent_steps + int(args.eval_freq)

            if agent_steps >= next_save:
                checkpoint_path = weights_dir / f"checkpoint_{agent_steps}.pt"
                agent.save_model(checkpoint_path)
                print(f"[{agent_steps}] Saved MAPPO model checkpoint to {checkpoint_path}")
                next_save = agent_steps + int(args.save_model_freq)

            if episode % 10 == 0:
                elapsed = datetime.now() - start_time
                print(
                    "Episode {} | agent_steps={} env_steps={} ep_return={:.3f} "
                    "failure={} elapsed={}".format(
                        episode,
                        agent_steps,
                        env_steps,
                        ep_return,
                        failure_reason,
                        elapsed,
                    )
                )

        if len(agent.buffer) > 0:
            metrics = agent.update(llm_loss_weight=_current_llm_loss_weight(agent_steps))
            for key, value in metrics.items():
                writer.add_scalar(f"loss/{key}", value, agent_steps)
        agent.save_checkpoint(
            last_ckpt_path,
            agent_steps=agent_steps,
            env_steps=env_steps,
            episode=episode,
        )
        if not best_model_path.exists():
            agent.save_model(best_model_path)

    except KeyboardInterrupt:
        agent.save_checkpoint(
            last_ckpt_path,
            agent_steps=agent_steps,
            env_steps=env_steps,
            episode=episode,
        )
        print(f"\nInterrupted. Saved checkpoint to {last_ckpt_path}")
        raise
    finally:
        writer.add_scalar("train/episodes_total", episode, agent_steps)
        writer.close()
        print(
            "MAPPO training complete | episodes={} agent_steps={} env_steps={} elapsed={}".format(
                episode,
                agent_steps,
                env_steps,
                datetime.now() - start_time,
            )
        )


if __name__ == "__main__":
    main()
