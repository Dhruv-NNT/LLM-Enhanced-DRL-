"""Run a pure-LLM multi-agent episode and save frames plus a GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs import NUM_AGENTS_DEFAULT, PLOT_DIR, ROUTE_IDS_DEFAULT, SEED
from evaluate import _timestamp, _validate_num_agents, _write_gif, evaluate_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    output_dir = Path(args.output_dir) if args.output_dir else PLOT_DIR / f"llm_only_{_timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate_episode(
        num_agents=num_agents,
        seed=args.seed,
        route_ids=args.route_ids,
        output_dir=output_dir,
        episode_step_cap=args.episode_step_cap,
    )
    gif_path = output_dir / (args.gif_name or f"llm_only_{_timestamp()}.gif")
    _write_gif(output_dir, gif_path)
    print(
        "steps={} reward={:.3f} success={} truncated={} gif={}".format(
            result.steps,
            result.total_reward,
            int(result.success),
            int(result.truncated),
            gif_path,
        )
    )


if __name__ == "__main__":
    main()
