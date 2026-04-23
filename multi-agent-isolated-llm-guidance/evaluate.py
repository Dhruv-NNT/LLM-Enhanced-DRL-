"""Pure-LLM evaluation helpers for isolated multi-agent guidance episodes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image

from configs import (
    EVAL_DIR,
    FRAME_DURATION_MS,
    MAX_AGENTS,
    MEMORY_DIR,
    NUM_AGENTS_DEFAULT,
    PLOT_DIR,
    ROUTE_IDS_DEFAULT,
    RUN_MODES,
    SEED,
)
from rl_llm_multi import JointGuidanceEnv, MultiAgentThreeCallController
from rl_llm_multi.utils import max_agents_possible


@dataclass
class EpisodeResult:
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


def _write_gif(frame_dir: Path, gif_path: Path, duration_ms: int = FRAME_DURATION_MS) -> None:
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


def evaluate_episode(
    *,
    num_agents: int,
    seed: Optional[int],
    route_ids: Optional[Sequence[str]],
    output_dir: Path,
    episode_step_cap: Optional[int] = None,
) -> EpisodeResult:
    route_ids = _normalize_route_ids(route_ids)
    memory_dir = _ensure_dir(MEMORY_DIR / f"run_{_timestamp()}__ep000001")
    controller = MultiAgentThreeCallController(save_dir=str(memory_dir))
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
        actions = controller.choose_llm_actions(env.core, step=env.n_step)
        if actions:
            llm_calls += 1
        _, reward, done, truncated, _ = env.step(actions)
        total_reward += float(reward)
        env.render(folder=str(output_dir))

    return EpisodeResult(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=RUN_MODES, default="LLM_ONLY")
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--route-ids", nargs="*", default=ROUTE_IDS_DEFAULT)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--gif-name", type=str, default=None)
    parser.add_argument("--episode-step-cap", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_agents = _validate_num_agents(args.num_agents)
    output_dir = Path(args.output_dir) if args.output_dir else PLOT_DIR / f"llm_only_eval_{_timestamp()}"
    _ensure_dir(output_dir)
    _ensure_dir(EVAL_DIR)

    result = evaluate_episode(
        num_agents=num_agents,
        seed=args.seed,
        route_ids=args.route_ids,
        output_dir=output_dir,
        episode_step_cap=args.episode_step_cap,
    )

    gif_name = args.gif_name or f"llm_only_{_timestamp()}.gif"
    gif_path = output_dir / gif_name
    _write_gif(output_dir, gif_path)

    summary_path = EVAL_DIR / f"eval_llm_only_{_timestamp()}.json"
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
