"""Train a pure-RL MAPPO baseline for the multi-agent sector simulator."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from configs import (
    ACTION_BINS,
    MAPPO_BEST_MODEL_PATH,
    MAPPO_ACTIVATION,
    MAPPO_CUDA_VISIBLE_DEVICES,
    MAPPO_EPS_CLIP,
    MAPPO_EVAL_FREQ,
    MAPPO_GAE_LAMBDA,
    MAPPO_GAMMA,
    MAPPO_HIDDEN_DIMS,
    MAPPO_K_EPOCHS,
    MAPPO_LAST_CKPT_PATH,
    MAPPO_LOG_DIR,
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
    LLM_GUIDANCE_END_STEP,
    LLM_GUIDANCE_MEMORY_ISOLATED_PER_RUN,
    LLM_GUIDANCE_MEMORY_PATH,
    DECISION_MEMORY_PATH,
    LLM_GUIDANCE_START_STEP,
    LLM_GUIDANCE_USE_VISION,
    LLM_MEMORY_EVALUATOR_ENABLED,
    LLM_MEMORY_UPDATE_MARGIN,
    LLM_SHADOW_EVAL_ENABLED,
    LLM_SHADOW_GAMMA,
    LLM_SHADOW_HORIZON,
    LLM_SHADOW_RETURN_MARGIN,
    LLM_SHADOW_RETURN_SCALE,
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
    parser.add_argument("--log-dir", type=str, default=str(MAPPO_LOG_DIR))
    parser.add_argument("--best-model-path", type=str, default=str(MAPPO_BEST_MODEL_PATH))
    parser.add_argument("--last-ckpt-path", type=str, default=str(MAPPO_LAST_CKPT_PATH))
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


def _resolve_llm_memory_path(log_dir: Path) -> Path:
    if LLM_GUIDANCE_MEMORY_PATH:
        return Path(str(LLM_GUIDANCE_MEMORY_PATH))
    if bool(LLM_GUIDANCE_MEMORY_ISOLATED_PER_RUN):
        return Path(log_dir) / "decision_memory.json"
    return Path(DECISION_MEMORY_PATH)


def _llm_return_weight(g_llm: float, g_mappo: float) -> float:
    advantage = float(g_llm) - float(g_mappo) - float(LLM_SHADOW_RETURN_MARGIN)
    if advantage <= 0.0:
        return 0.0
    scale = max(float(LLM_SHADOW_RETURN_SCALE), 1e-6)
    return max(0.0, min(float(LLM_SHADOW_RETURN_WEIGHT_CLIP), advantage / scale))


def _build_llm_metadata_by_agent(
    advice_by_agent: Mapping[str, object],
    shadow_result: Optional[object],
    memory_updates: Mapping[str, bool],
) -> Dict[str, Dict[str, object]]:
    metadata: Dict[str, Dict[str, object]] = {}
    for agent_id, advice in advice_by_agent.items():
        agent_id = str(agent_id)
        g_mappo = 0.0
        g_llm = 0.0
        g_memory = 0.0
        if shadow_result is not None:
            g_mappo = float(getattr(shadow_result, "mappo_returns", {}).get(agent_id, 0.0))
            g_llm = float(getattr(shadow_result, "llm_returns", {}).get(agent_id, 0.0))
            g_memory = float(getattr(shadow_result, "memory_returns", {}).get(agent_id, 0.0))
        metadata[agent_id] = {
            "llm_label_available": True,
            "llm_action_idx": int(getattr(advice, "llm_action_idx")),
            "llm_action_deg": int(getattr(advice, "llm_action_deg")),
            "llm_phase": str(getattr(advice, "phase", "")),
            "G_MAPPO_shadow": g_mappo,
            "G_LLM_shadow": g_llm,
            "G_MEMORY_shadow": g_memory,
            "llm_return_weight": _llm_return_weight(g_llm, g_mappo),
            "memory_action_idx": getattr(advice, "memory_action_idx", None) or 0,
            "memory_action_deg": getattr(advice, "memory_action_deg", None) or 0,
            "memory_ref": getattr(advice, "memory_ref", None),
            "memory_update_applied": bool(memory_updates.get(agent_id, False)),
        }
    return metadata


def _apply_memory_corrections(
    memory_store: Optional[DecisionMemoryStore],
    *,
    advice_by_agent: Mapping[str, object],
    shadow_result: Optional[object],
    policy_actions: Mapping[str, int],
) -> tuple[Dict[str, bool], Dict[str, str]]:
    updates: Dict[str, bool] = {}
    sources: Dict[str, str] = {}
    if not bool(LLM_MEMORY_EVALUATOR_ENABLED) or memory_store is None or shadow_result is None:
        return updates, sources
    memory_returns = getattr(shadow_result, "memory_returns", {})
    if not memory_returns:
        return updates, sources

    for agent_id, advice in advice_by_agent.items():
        agent_id = str(agent_id)
        memory_ref = getattr(advice, "memory_ref", None)
        if not isinstance(memory_ref, dict) or getattr(advice, "memory_action_deg", None) is None:
            continue
        if agent_id not in memory_returns:
            continue
        g_memory = float(memory_returns.get(agent_id, 0.0))
        g_mappo = float(getattr(shadow_result, "mappo_returns", {}).get(agent_id, 0.0))
        g_llm = float(getattr(shadow_result, "llm_returns", {}).get(agent_id, 0.0))
        if g_llm >= g_mappo:
            best_return = g_llm
            corrected = int(getattr(advice, "llm_action_deg"))
            source = "llm"
            note = "short-horizon LLM return improved memory"
        else:
            best_return = g_mappo
            corrected = action_idx_to_deg(int(policy_actions[agent_id]), ACTION_BINS)
            source = "mappo"
            note = "short-horizon MAPPO return improved memory"
        if best_return <= g_memory + float(LLM_MEMORY_UPDATE_MARGIN):
            continue
        applied = memory_store.update_case_result(memory_ref, corrected=corrected, note=note)
        updates[agent_id] = bool(applied)
        if applied:
            sources[agent_id] = source
    return updates, sources


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
    "safe_progress",
    "progress_risk_gate",
    "progress_safety_scale",
    "cross_track",
    "cross_track_penalty_scale",
    "relaxed_cross_track",
    "safe_cross_track_recovery",
    "merge_back_scale",
    "merge_back_bonus",
    "heading_error_to_destination",
    "safe_heading_penalty",
    "predicted_traffic_risk",
    "local_traffic_risk",
    "traffic_risk",
    "local_traffic_risk_reduction",
    "traffic_risk_reduction",
    "weather_risk",
    "weather_risk_reduction",
    "finish_bonus",
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
    if abs(terminal_reward) > 1e-12:
        writer.add_scalar("train/reward/terminal_episode", terminal_reward, episode)
    if counts.get("agent_dense_mean", 0) > 0:
        writer.add_scalar(
            "train/reward/mean_step_behavior_reward",
            sums["agent_dense_mean"] / float(counts["agent_dense_mean"]),
            episode,
        )
    for key in ESSENTIAL_REWARD_COMPONENT_KEYS:
        if counts.get(key, 0) > 0:
            writer.add_scalar(
                f"train/reward_component/{key}_mean",
                sums[key] / float(counts[key]),
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
    log_dir = Path(args.log_dir)
    best_model_path = Path(args.best_model_path)
    last_ckpt_path = Path(args.last_ckpt_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    probe_env = make_env(
        num_agents=args.num_agents,
        num_weather_cells=args.num_weather_cells,
        episode_step_cap=args.episode_step_cap,
    )
    probe_env.reset(seed=args.seed, options=reset_options)
    global_state_dim = int(probe_env.state().shape[0])
    local_obs_dim = int(probe_env.observation_space(probe_env.possible_agents[0]).shape[0])
    agent = build_agent(global_state_dim, local_obs_dim)
    llm_memory_path = _resolve_llm_memory_path(log_dir)
    if USE_LLM_GUIDED_TRAINING:
        guidance = LLMGuidanceProvider(
            save_dir=log_dir / "llm_guidance_json",
            memory_path=llm_memory_path,
            use_vision=bool(LLM_GUIDANCE_USE_VISION),
            memory_visual_audit_enabled=bool(DECISION_MEMORY_VISUAL_AUDIT_ENABLED),
            memory_visual_audit_dir=DECISION_MEMORY_VISUAL_AUDIT_DIR,
            memory_visual_audit_frame_count=int(DECISION_MEMORY_VISUAL_AUDIT_FRAME_COUNT),
        )
        memory_store: Optional[DecisionMemoryStore] = DecisionMemoryStore(llm_memory_path)
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

    writer = SummaryWriter(str(log_dir), purge_step=agent_steps)
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
            ep_memory_retrieval_count = 0
            ep_memory_update_count = 0
            ep_memory_update_to_mappo = 0
            ep_memory_update_to_llm = 0
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
                memory_updates: Dict[str, bool] = {}
                memory_update_sources: Dict[str, str] = {}
                llm_metadata_by_agent: Dict[str, Dict[str, object]] = {}
                if _llm_guidance_active(agent_steps):
                    advice_by_agent = guidance.advice_for_step(
                        env.core,
                        step=env.core.n_step,
                        proposed_actions=policy_actions,
                        latest_frame_path=None,
                    )
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
                        memory_updates, memory_update_sources = _apply_memory_corrections(
                            memory_store,
                            advice_by_agent=advice_by_agent,
                            shadow_result=shadow_result,
                            policy_actions=policy_actions,
                        )
                    llm_metadata_by_agent = _build_llm_metadata_by_agent(
                        advice_by_agent,
                        shadow_result,
                        memory_updates,
                    )

                    for agent_id, advice in advice_by_agent.items():
                        agent_id = str(agent_id)
                        metadata = llm_metadata_by_agent.get(agent_id, {})
                        ep_llm_guided_decisions += 1
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

                final_actions = dict(policy_actions)
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
                "train/shadow/llm_beats_mappo_rate",
                sum(
                    1
                    for g_llm, g_mappo in zip(ep_shadow_llm_returns, ep_shadow_mappo_returns)
                    if g_llm > g_mappo
                )
                / max(float(len(ep_shadow_llm_returns)), 1.0),
                agent_steps,
            )
            writer.add_scalar("train/memory/retrieval_count", ep_memory_retrieval_count, agent_steps)
            writer.add_scalar("train/memory/update_count", ep_memory_update_count, agent_steps)
            writer.add_scalar("train/memory/corrected_to_mappo_count", ep_memory_update_to_mappo, agent_steps)
            writer.add_scalar("train/memory/corrected_to_llm_count", ep_memory_update_to_llm, agent_steps)
            writer.add_scalar("train/llm/execute_turn_count", ep_phase_counts["EXECUTE_TURN"], agent_steps)
            writer.add_scalar("train/llm/emergency_maneuver_count", ep_phase_counts["EMERGENCY_MANEUVER"], agent_steps)
            writer.add_scalar("train/llm/merge_back_count", ep_phase_counts["MERGE_BACK"], agent_steps)

            if agent_steps >= next_update and len(agent.buffer) > 0:
                metrics = agent.update()
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
                checkpoint_path = log_dir / f"checkpoint_{agent_steps}.pt"
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
            metrics = agent.update()
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
