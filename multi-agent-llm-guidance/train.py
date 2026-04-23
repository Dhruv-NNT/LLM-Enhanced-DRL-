"""Training entrypoint for RL-only, hybrid, and pure-LLM multi-agent episodes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from configs import (
    CHECKPOINT_DIR,
    EPISODE_MODE_CYCLE,
    EVAL_DIR,
    EVAL_INTERVAL_EPISODES,
    MAX_AGENTS,
    MAX_TIMESTEPS,
    MEMORY_DIR,
    MODEL_NAME,
    NUM_AGENTS_DEFAULT,
    PLOT_DIR,
    ROUTE_IDS_DEFAULT,
    RUN_MODES,
    SAVE_MODEL_FREQ,
    SEED,
    UPDATE_TIMESTEP,
)
from rl_llm_multi import (
    JointGuidanceEnv,
    MultiAgentParallelEnv,
    MultiAgentThreeCallController,
    SharedActorCentralCriticPPO,
)
from rl_llm_multi.utils import max_agents_possible


@dataclass
class EpisodeResult:
    episode_idx: int
    mode: str
    num_agents: int
    seed: Optional[int]
    route_ids: Optional[List[str]]
    steps: int
    total_reward: float
    success: bool
    truncated: bool
    collision: bool
    active_calls_logged: int


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _episode_log_dir(episode_idx: int) -> Path:
    return _ensure_dir(PLOT_DIR / f"run_{_timestamp()}__ep{episode_idx:06d}")


def _memory_log_dir(episode_idx: int) -> Path:
    return _ensure_dir(MEMORY_DIR / f"run_{_timestamp()}__ep{episode_idx:06d}")


def _checkpoint_path(tag: str) -> Path:
    return CHECKPOINT_DIR / f"{MODEL_NAME}_{tag}.pth"


def _mode_for_episode(episode_idx: int, fixed_mode: Optional[str]) -> str:
    if fixed_mode is not None:
        return fixed_mode
    return EPISODE_MODE_CYCLE[(episode_idx - 1) % len(EPISODE_MODE_CYCLE)]


def _normalize_route_ids(route_ids: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not route_ids:
        return None
    values = [value.strip() for value in route_ids if value.strip()]
    return values or None


def _validate_num_agents(num_agents: int) -> int:
    capacity = max_agents_possible()
    if num_agents < 1 or num_agents > capacity:
        raise ValueError(f"num_agents must satisfy 1 <= num_agents <= {capacity}, got {num_agents}")
    return int(num_agents)


def _build_policy(env: MultiAgentParallelEnv) -> SharedActorCentralCriticPPO:
    obs_dim = env.core.local_obs_dim
    global_dim = env.state().shape[0]
    return SharedActorCentralCriticPPO(local_obs_dim=obs_dim, global_obs_dim=global_dim)


def _run_policy_episode(
    *,
    env: MultiAgentParallelEnv,
    ppo: SharedActorCentralCriticPPO,
    controller: MultiAgentThreeCallController,
    mode: str,
    episode_idx: int,
    seed: Optional[int],
    num_agents: int,
    route_ids: Optional[Sequence[str]],
    render_dir: Optional[Path] = None,
) -> Tuple[EpisodeResult, int]:
    route_ids = _normalize_route_ids(route_ids)
    obs_dict, infos = env.reset(
        seed=seed,
        options={"num_agents": num_agents, "route_ids": route_ids},
    )
    controller.reset()
    if mode == "HYBRID" and render_dir is not None:
        env.render(folder=str(render_dir))

    total_transitions = 0
    total_reward = 0.0
    llm_calls = 0
    team_done = False
    team_truncated = False

    while env.agents:
        common_info = infos["__common__"]
        global_state = common_info["global_state"]
        advice_embeddings: Optional[Dict[str, np.ndarray]] = None
        target_bins: Optional[Dict[str, int]] = None
        active_masks: Optional[Dict[str, float]] = None

        if mode == "HYBRID":
            actions_preview = controller.update(env.core, step=env.core.n_step, ppo=ppo)
            if actions_preview:
                llm_calls += 1
            advice_embeddings, target_bins, active_masks = controller.get_advice_for_rl(
                env.core.active_agent_ids,
                core=env.core,
            )

        actions, records = ppo.select_actions(
            obs_dict,
            global_state,
            advice_embeddings=advice_embeddings,
            advice_active_masks=active_masks,
            target_bins=target_bins,
            deterministic=False,
            train=True,
        )

        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        next_common = infos["__common__"]
        next_global_state = next_common["global_state"]
        step_done = bool(next_common["episode_done"] or next_common["episode_truncated"])
        team_done = bool(next_common["episode_done"])
        team_truncated = bool(next_common["episode_truncated"])
        team_reward = float(next_common["team_reward"])
        total_reward += team_reward

        for agent_id, record in records.items():
            ppo.record_transition(
                record,
                reward=float(rewards.get(agent_id, team_reward)),
                done=step_done,
                next_global_state=next_global_state,
            )
            total_transitions += 1

        if render_dir is not None:
            env.render(folder=str(render_dir))

        obs_dict = next_obs

    result = EpisodeResult(
        episode_idx=episode_idx,
        mode=mode,
        num_agents=num_agents,
        seed=seed,
        route_ids=route_ids,
        steps=env.core.n_step,
        total_reward=total_reward,
        success=bool(env.core.last_team_success),
        truncated=team_truncated,
        collision=bool(team_done and not env.core.last_team_success and not team_truncated),
        active_calls_logged=llm_calls,
    )
    return result, total_transitions


def _run_llm_only_episode(
    *,
    env: JointGuidanceEnv,
    controller: MultiAgentThreeCallController,
    ppo: Optional[SharedActorCentralCriticPPO],
    episode_idx: int,
    seed: Optional[int],
    num_agents: int,
    route_ids: Optional[Sequence[str]],
    render_dir: Optional[Path] = None,
) -> EpisodeResult:
    route_ids = _normalize_route_ids(route_ids)
    env.reset(seed=seed, num_agents=num_agents, route_ids=route_ids)
    controller.reset()
    if render_dir is not None:
        env.render(folder=str(render_dir))

    total_reward = 0.0
    llm_calls = 0
    done = False
    truncated = False

    while not done and not truncated:
        actions = controller.choose_llm_actions(env.core, step=env.n_step, ppo=ppo)
        if actions:
            llm_calls += 1
        _, reward, done, truncated, _ = env.step(actions)
        total_reward += float(reward)
        if render_dir is not None:
            env.render(folder=str(render_dir))

    return EpisodeResult(
        episode_idx=episode_idx,
        mode="LLM_ONLY",
        num_agents=num_agents,
        seed=seed,
        route_ids=route_ids,
        steps=env.core.n_step,
        total_reward=total_reward,
        success=bool(env.core.last_team_success),
        truncated=bool(truncated),
        collision=bool(done and not env.core.last_team_success and not truncated),
        active_calls_logged=llm_calls,
    )


def _write_episode_summary(path: Path, result: EpisodeResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-timesteps", type=int, default=MAX_TIMESTEPS)
    parser.add_argument("--episode-step-cap", type=int, default=None)
    parser.add_argument("--episode-mode", choices=RUN_MODES, default=None)
    parser.add_argument("--route-ids", nargs="*", default=ROUTE_IDS_DEFAULT)
    parser.add_argument("--save-renders", action="store_true")
    parser.add_argument("--render-every", type=int, default=EVAL_INTERVAL_EPISODES)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_agents = _validate_num_agents(args.num_agents)
    route_ids = _normalize_route_ids(args.route_ids)

    _ensure_dir(CHECKPOINT_DIR)
    _ensure_dir(EVAL_DIR)
    _ensure_dir(PLOT_DIR)

    parallel_env = MultiAgentParallelEnv(num_agents=num_agents, max_agents=MAX_AGENTS)
    llm_env = JointGuidanceEnv(num_agents=num_agents, max_agents=MAX_AGENTS)
    if args.episode_step_cap is not None:
        parallel_env.core.max_step = int(args.episode_step_cap)
        llm_env.core.max_step = int(args.episode_step_cap)
    ppo = _build_policy(parallel_env)
    if args.checkpoint:
        ppo.load(args.checkpoint)

    controller = MultiAgentThreeCallController()

    episode_idx = 0
    total_transitions = 0
    results: List[EpisodeResult] = []

    while total_transitions < int(args.max_timesteps):
        episode_idx += 1
        mode = _mode_for_episode(episode_idx, args.episode_mode)
        render_dir = None
        if args.save_renders and episode_idx % max(1, int(args.render_every)) == 0:
            render_dir = _episode_log_dir(episode_idx)

        if mode == "LLM_ONLY":
            controller.save_dir = str(_memory_log_dir(episode_idx))
            result = _run_llm_only_episode(
                env=llm_env,
                controller=controller,
                ppo=ppo,
                episode_idx=episode_idx,
                seed=args.seed + episode_idx - 1 if args.seed is not None else None,
                num_agents=num_agents,
                route_ids=route_ids,
                render_dir=render_dir,
            )
        else:
            if mode == "HYBRID":
                controller.save_dir = str(_memory_log_dir(episode_idx))
            result, transitions = _run_policy_episode(
                env=parallel_env,
                ppo=ppo,
                controller=controller,
                mode=mode,
                episode_idx=episode_idx,
                seed=args.seed + episode_idx - 1 if args.seed is not None else None,
                num_agents=num_agents,
                route_ids=route_ids,
                render_dir=render_dir,
            )
            total_transitions += transitions

        if ppo.buffer.local_obs:
            ppo.update()

        results.append(result)
        summary_path = EVAL_DIR / f"train_episode_{episode_idx:06d}.json"
        _write_episode_summary(summary_path, result)

        if episode_idx % max(1, SAVE_MODEL_FREQ) == 0:
            ppo.save(str(_checkpoint_path(f"ep{episode_idx:06d}")))
        ppo.save(str(_checkpoint_path("latest")))

        print(
            "episode={} mode={} steps={} reward={:.3f} success={} truncated={} transitions={}".format(
                result.episode_idx,
                result.mode,
                result.steps,
                result.total_reward,
                int(result.success),
                int(result.truncated),
                total_transitions,
            )
        )

    manifest_path = EVAL_DIR / f"train_manifest_{_timestamp()}.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(result) for result in results], handle, indent=2)


if __name__ == "__main__":
    main()
