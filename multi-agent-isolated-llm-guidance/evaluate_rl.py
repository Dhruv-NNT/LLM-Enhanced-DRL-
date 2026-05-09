"""Evaluate a trained MAPPO policy and optionally save a rollout GIF."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from configs import (
    ACTION_BINS,
    FRAME_DURATION_MS,
    MAPPO_ACTIVATION,
    MAPPO_BEST_MODEL_PATH,
    MAPPO_CUDA_VISIBLE_DEVICES,
    MAPPO_EPS_CLIP,
    MAPPO_EVAL_OUTPUT_DIR,
    MAPPO_GAE_LAMBDA,
    MAPPO_GAMMA,
    MAPPO_HIDDEN_DIMS,
    MAPPO_K_EPOCHS,
    MAPPO_LR_ACTOR,
    MAPPO_LR_CRITIC,
    MAPPO_METRICS_DIR,
    MAPPO_MINIBATCH_SIZE,
    MAPPO_N_EVAL_EPISODES,
    MAPPO_NORMALIZE_REWARD,
    MAPPO_ORTHOGONAL_INIT,
    MAPPO_USE_LAYER_NORM,
    MAX_AGENTS,
    NUM_AGENTS_DEFAULT,
    NUM_WEATHER_CELLS_DEFAULT,
    ROUTE_IDS_DEFAULT,
    SEED,
)

if MAPPO_CUDA_VISIBLE_DEVICES is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(MAPPO_CUDA_VISIBLE_DEVICES)

from rl_llm_multi import MAPPO, MultiAgentParallelEnv


@dataclass
class RLEpisodeResult:
    episode_index: int
    seed: int
    steps: int
    total_reward: float
    success: bool
    truncated: bool
    failure_reason: Optional[str]
    num_agents: int
    num_weather_cells: int
    gif_path: Optional[str] = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_route_ids(route_ids: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not route_ids:
        return None
    values = [str(route_id).strip() for route_id in route_ids if str(route_id).strip()]
    return values or None


def _write_gif(frame_dir: Path, gif_path: Path, duration_ms: int = FRAME_DURATION_MS) -> None:
    frame_paths = sorted(frame_dir.glob("image_*.png"))
    if not frame_paths:
        return
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
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


def rollout_episode(
    agent: MAPPO,
    *,
    episode_index: int,
    seed: int,
    num_agents: int,
    num_weather_cells: int,
    route_ids: Optional[Sequence[str]],
    episode_step_cap: Optional[int],
    frame_dir: Optional[Path] = None,
    gif_path: Optional[Path] = None,
) -> RLEpisodeResult:
    reset_options = {
        "num_agents": int(num_agents),
        "route_ids": _normalize_route_ids(route_ids),
        "num_weather_cells": int(num_weather_cells),
    }
    env = make_env(
        num_agents=num_agents,
        num_weather_cells=num_weather_cells,
        episode_step_cap=episode_step_cap,
    )
    _, info = env.reset(seed=seed, options=reset_options)
    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)
        env.render(folder=str(frame_dir))

    done = bool(info["__common__"]["episode_done"])
    truncated = bool(info["__common__"]["episode_truncated"])
    total_reward = 0.0
    steps = 0
    while not done and not truncated:
        active_ids = list(env.agents)
        if not active_ids:
            _, _, _, _, info = env.step({})
        else:
            actions, _, _ = agent.select_actions(
                env.state(),
                active_ids,
                episode_id=episode_index,
                deterministic=True,
                store=False,
            )
            _, _, _, _, info = env.step(actions)
        common = info["__common__"]
        total_reward += float(common["team_reward"])
        done = bool(common["episode_done"])
        truncated = bool(common["episode_truncated"])
        steps += 1
        if frame_dir is not None:
            env.render(folder=str(frame_dir))
        if steps > env.core.max_step + 5:
            break

    written_gif = None
    if frame_dir is not None and gif_path is not None:
        _write_gif(frame_dir, gif_path)
        written_gif = str(gif_path)

    common = info["__common__"]
    return RLEpisodeResult(
        episode_index=int(episode_index),
        seed=int(seed),
        steps=int(steps),
        total_reward=float(total_reward),
        success=bool(common["episode_success"]),
        truncated=bool(common["episode_truncated"]),
        failure_reason=common["failure_reason"],
        num_agents=int(num_agents),
        num_weather_cells=int(num_weather_cells),
        gif_path=written_gif,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default=str(MAPPO_BEST_MODEL_PATH))
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS_DEFAULT)
    parser.add_argument("--num-weather-cells", type=int, choices=(1, 2), default=NUM_WEATHER_CELLS_DEFAULT)
    parser.add_argument("--route-ids", nargs="*", default=ROUTE_IDS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-episodes", type=int, default=MAPPO_N_EVAL_EPISODES)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--episode-step-cap", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MAPPO model not found: {model_path}")

    route_ids = _normalize_route_ids(args.route_ids)
    probe_env = make_env(
        num_agents=args.num_agents,
        num_weather_cells=args.num_weather_cells,
        episode_step_cap=args.episode_step_cap,
    )
    probe_env.reset(
        seed=args.seed,
        options={
            "num_agents": int(args.num_agents),
            "route_ids": route_ids,
            "num_weather_cells": int(args.num_weather_cells),
        },
    )
    agent = build_agent(int(probe_env.state().shape[0]))
    agent.load_model(model_path)

    output_root = Path(args.output_dir) if args.output_dir else MAPPO_EVAL_OUTPUT_DIR / f"eval_{_timestamp()}"
    metrics_dir = MAPPO_METRICS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[RLEpisodeResult] = []
    for ep_idx in range(int(args.n_episodes)):
        frame_dir = None
        gif_path = None
        if args.save_gif and ep_idx == 0:
            frame_dir = output_root / f"ep{ep_idx + 1:03d}"
            gif_path = output_root / f"ep{ep_idx + 1:03d}.gif"
        result = rollout_episode(
            agent,
            episode_index=ep_idx,
            seed=int(args.seed) + ep_idx,
            num_agents=int(args.num_agents),
            num_weather_cells=int(args.num_weather_cells),
            route_ids=route_ids,
            episode_step_cap=args.episode_step_cap,
            frame_dir=frame_dir,
            gif_path=gif_path,
        )
        results.append(result)

    returns = [result.total_reward for result in results]
    summary = {
        "model_path": str(model_path),
        "num_episodes": len(results),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "success_rate": float(np.mean([result.success for result in results])) if results else 0.0,
        "collision_rate": float(np.mean([result.failure_reason == "collision" for result in results])) if results else 0.0,
        "weather_rate": float(np.mean([result.failure_reason == "weather" for result in results])) if results else 0.0,
        "truncation_rate": float(np.mean([result.truncated for result in results])) if results else 0.0,
        "episodes": [asdict(result) for result in results],
    }

    metrics_path = Path(args.metrics_path) if args.metrics_path else metrics_dir / f"eval_mappo_{_timestamp()}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "episodes={} mean_return={:.3f} success_rate={:.2f} collision_rate={:.2f} "
        "weather_rate={:.2f} truncation_rate={:.2f} metrics={}".format(
            summary["num_episodes"],
            summary["mean_return"],
            summary["success_rate"],
            summary["collision_rate"],
            summary["weather_rate"],
            summary["truncation_rate"],
            metrics_path,
        )
    )
    if args.save_gif and results and results[0].gif_path:
        print(f"gif={results[0].gif_path}")


if __name__ == "__main__":
    main()
