"""Train a pure-RL MAPPO baseline for the multi-agent sector simulator."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
    MAX_AGENTS,
    NUM_AGENTS_DEFAULT,
    NUM_WEATHER_CELLS_DEFAULT,
    ROUTE_IDS_DEFAULT,
    SEED,
)

if MAPPO_CUDA_VISIBLE_DEVICES is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(MAPPO_CUDA_VISIBLE_DEVICES)

from rl_llm_multi import MAPPO, MultiAgentParallelEnv, NoGuidanceProvider
from rl_llm_multi.mappo import evaluate_policy

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


def build_agent(global_state_dim: int) -> MAPPO:
    return MAPPO(
        global_state_dim=global_state_dim,
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


ESSENTIAL_REWARD_COMPONENT_KEYS = (
    "safe_progress",
    "cross_track",
    "cross_track_penalty_scale",
    "relaxed_cross_track",
    "safe_cross_track_recovery",
    "heading_error_to_destination",
    "safe_heading_penalty",
    "traffic_risk",
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
    agent = build_agent(global_state_dim)
    guidance = NoGuidanceProvider()

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
        "Starting MAPPO training | num_agents={} weather_cells={} global_dim={} max_agent_steps={}".format(
            args.num_agents,
            args.num_weather_cells,
            global_state_dim,
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
            _, info = env.reset(seed=int(args.seed) + int(episode), options=reset_options)
            guidance.reset()
            done = bool(info["__common__"]["episode_done"])
            truncated = bool(info["__common__"]["episode_truncated"])
            ep_return = 0.0
            ep_env_steps = 0
            ep_reward_sums: Dict[str, float] = {}
            ep_reward_counts: Dict[str, int] = {}

            while not done and not truncated:
                active_ids = list(env.agents)
                if not active_ids:
                    _, _, _, _, info = env.step({})
                    common = info["__common__"]
                    team_reward = float(common["team_reward"])
                    done = bool(common["episode_done"])
                    truncated = bool(common["episode_truncated"])
                    if done or truncated:
                        agent.add_terminal_bonus(
                            episode,
                            common.get("agent_terminal_rewards", common.get("terminal_reward", 0.0)),
                        )
                    env_steps += 1
                    ep_env_steps += 1
                    ep_return += team_reward
                    _accumulate_reward_debug(common, ep_reward_sums, ep_reward_counts)
                    continue

                policy_actions, records, _entropy = agent.select_actions(
                    env.state(),
                    active_ids,
                    episode_id=episode,
                    deterministic=False,
                    store=True,
                )
                overrides = guidance.actions_for_step(
                    env.core,
                    step=env.core.n_step,
                    proposed_actions=policy_actions,
                )
                final_actions = dict(policy_actions)
                for agent_id, action_idx in overrides.items():
                    if agent_id in active_ids:
                        final_actions[agent_id] = int(action_idx)

                _, _, _, _, info = env.step(final_actions)
                common = info["__common__"]
                team_reward = float(common["team_reward"])
                done = bool(common["episode_done"])
                truncated = bool(common["episode_truncated"])
                post_active = set(common["active_agents"])
                team_terminal = bool(done or truncated)
                terminals = {
                    agent_id: bool(team_terminal or agent_id not in post_active)
                    for agent_id in active_ids
                }
                agent.store_outcomes(
                    records,
                    reward=common.get("agent_dense_rewards", common.get("agent_rewards", team_reward)),
                    terminals=terminals,
                )
                if team_terminal:
                    agent.add_terminal_bonus(
                        episode,
                        common.get("agent_terminal_rewards", common.get("terminal_reward", 0.0)),
                    )

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
