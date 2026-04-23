"""Evaluation helpers for RL-only, hybrid, and pure-LLM multi-agent episodes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image

from configs import (
    CHECKPOINT_DIR,
    EVAL_DIR,
    MAX_AGENTS,
    MEMORY_DIR,
    NUM_AGENTS_DEFAULT,
    PLOT_DIR,
    ROUTE_IDS_DEFAULT,
    RUN_MODES,
    SEED,
)
from rl_llm_multi import (
    JointGuidanceEnv,
    MultiAgentParallelEnv,
    MultiAgentThreeCallController,
    SharedActorCentralCriticPPO,
)
from rl_llm_multi.utils import max_agents_possible
from train import EpisodeResult


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _write_gif(frame_dir: Path, gif_path: Path, duration_ms: int = 250) -> None:
    frame_paths = sorted(frame_dir.glob("image_*.png"))
    if not frame_paths:
        return
    frames = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    for frame in frames:
        frame.close()


def _build_policy(env: MultiAgentParallelEnv, checkpoint: Optional[str]) -> SharedActorCentralCriticPPO:
    ppo = SharedActorCentralCriticPPO(
        local_obs_dim=env.core.local_obs_dim,
        global_obs_dim=env.state().shape[0],
    )
    if checkpoint:
        ppo.load(checkpoint)
    return ppo


def evaluate_episode(
    *,
    mode: str,
    num_agents: int,
    seed: Optional[int],
    route_ids: Optional[Sequence[str]],
    checkpoint: Optional[str],
    output_dir: Path,
    episode_step_cap: Optional[int] = None,
) -> EpisodeResult:
    route_ids = _normalize_route_ids(route_ids)
    memory_dir = _ensure_dir(MEMORY_DIR / f"run_{_timestamp()}__ep000001")
    controller = MultiAgentThreeCallController(save_dir=str(memory_dir))

    if mode == "LLM_ONLY":
        env = JointGuidanceEnv(num_agents=num_agents, max_agents=MAX_AGENTS)
        if episode_step_cap is not None:
            env.core.max_step = int(episode_step_cap)
        env.reset(seed=seed, num_agents=num_agents, route_ids=route_ids)
        controller.reset()
        env.render(folder=str(output_dir))
        total_reward = 0.0
        llm_calls = 0
        done = False
        truncated = False
        while not done and not truncated:
            actions = controller.choose_llm_actions(env.core, step=env.n_step, ppo=None)
            if actions:
                llm_calls += 1
            _, reward, done, truncated, _ = env.step(actions)
            total_reward += float(reward)
            env.render(folder=str(output_dir))
        return EpisodeResult(
            episode_idx=1,
            mode=mode,
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

    env = MultiAgentParallelEnv(num_agents=num_agents, max_agents=MAX_AGENTS)
    if episode_step_cap is not None:
        env.core.max_step = int(episode_step_cap)
    ppo = _build_policy(env, checkpoint)
    obs_dict, infos = env.reset(seed=seed, options={"num_agents": num_agents, "route_ids": route_ids})
    controller.reset()
    env.render(folder=str(output_dir))

    total_reward = 0.0
    llm_calls = 0
    while env.agents:
        common = infos["__common__"]
        advice_embeddings = None
        target_bins = None
        advice_masks = None
        if mode == "HYBRID":
            preview = controller.update(env.core, step=env.core.n_step, ppo=ppo)
            if preview:
                llm_calls += 1
            advice_embeddings, target_bins, advice_masks = controller.get_advice_for_rl(
                env.core.active_agent_ids,
                core=env.core,
            )
        actions = ppo.greedy_actions(
            obs_dict,
            common["global_state"],
            advice_embeddings=advice_embeddings,
            advice_active_masks=advice_masks,
        )
        obs_dict, _, _, _, infos = env.step(actions)
        total_reward += float(infos["__common__"]["team_reward"])
        env.render(folder=str(output_dir))

    return EpisodeResult(
        episode_idx=1,
        mode=mode,
        num_agents=num_agents,
        seed=seed,
        route_ids=route_ids,
        steps=env.core.n_step,
        total_reward=total_reward,
        success=bool(env.core.last_team_success),
        truncated=bool(env.core.last_team_truncated),
        collision=bool(env.core.last_team_done and not env.core.last_team_success and not env.core.last_team_truncated),
        active_calls_logged=llm_calls,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=RUN_MODES, default="HYBRID")
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--route-ids", nargs="*", default=ROUTE_IDS_DEFAULT)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_DIR / "multi_agent_shared_policy_latest.pth"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--gif-name", type=str, default=None)
    parser.add_argument("--episode-step-cap", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_agents = _validate_num_agents(args.num_agents)
    output_dir = Path(args.output_dir) if args.output_dir else PLOT_DIR / f"eval_{args.mode.lower()}_{_timestamp()}"
    _ensure_dir(output_dir)
    _ensure_dir(EVAL_DIR)

    checkpoint = None if args.mode == "LLM_ONLY" else args.checkpoint
    if checkpoint and not Path(checkpoint).exists():
        checkpoint = None

    result = evaluate_episode(
        mode=args.mode,
        num_agents=num_agents,
        seed=args.seed,
        route_ids=args.route_ids,
        checkpoint=checkpoint,
        output_dir=output_dir,
        episode_step_cap=args.episode_step_cap,
    )

    gif_name = args.gif_name or f"{args.mode.lower()}_{_timestamp()}.gif"
    gif_path = output_dir / gif_name
    _write_gif(output_dir, gif_path)

    summary_path = EVAL_DIR / f"eval_{args.mode.lower()}_{_timestamp()}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)

    print(
        "mode={} steps={} reward={:.3f} success={} truncated={} gif={}".format(
            result.mode,
            result.steps,
            result.total_reward,
            int(result.success),
            int(result.truncated),
            gif_path,
        )
    )


if __name__ == "__main__":
    main()
