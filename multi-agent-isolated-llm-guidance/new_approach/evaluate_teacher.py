"""Quick sanity check: fly a trained teacher network as the controller.

Loads a teacher checkpoint (teacher_*.pt) and lets it choose every aircraft's
action directly in the environment (argmax over its action logits), then reports
success / collision / weather / truncation rates over N games.

This is a standalone sanity check of the teacher ON ITS OWN. It is NOT how the
teacher is used in the method (there the teacher only shapes MAPPO's training
loss; MAPPO flies). But it answers "does this teacher, by itself, get aircraft to
their destinations?" before you commit to the full training runs.

Example (CPU):
    export CUDA_VISIBLE_DEVICES=""
    python new_approach/evaluate_teacher.py \
        --teacher-path teachers/teacher_best_preview.pt \
        --n-episodes 100 --num-agents 4 --num-weather-cells 2
"""

from __future__ import annotations

# --- path bootstrap: make the project root importable when this script is run
# from its subfolder, so `import configs` / `from rl_llm_multi ...` still resolve.
import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, _bootstrap_os.path.dirname(
    _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))

import argparse
from typing import Optional, Sequence

import torch
from tqdm import tqdm

from configs import NUM_AGENTS_DEFAULT, NUM_WEATHER_CELLS_DEFAULT, ROUTE_IDS_DEFAULT, SEED
from rl_llm_multi import MultiAgentParallelEnv
from rl_llm_multi.mappo import MAX_AGENTS, agent_id_to_index
from rl_llm_multi.teacher import (
    load_teacher,
    load_trajectory_dataset,
    teacher_agreement_metrics,
)


def _normalize_route_ids(route_ids: Optional[Sequence[str]]):
    if not route_ids:
        return None
    values = [str(r).strip() for r in route_ids if str(r).strip()]
    return values or None


@torch.no_grad()
def _teacher_actions(teacher, observations, active_ids):
    """Build {agent_id: action_index} from the teacher's argmax for each aircraft.
    The teacher takes RAW local obs (it standardizes internally)."""
    dev = next(teacher.parameters()).device
    actions = {}
    for agent_id in active_ids:
        obs = observations.get(agent_id)
        if obs is None:
            continue
        idx = agent_id_to_index(agent_id, MAX_AGENTS)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev).view(1, -1)
        idx_t = torch.tensor([idx], dtype=torch.long, device=dev)
        logits = teacher.action_logits(obs_t, idx_t)
        actions[agent_id] = int(torch.argmax(logits, dim=-1).item())
    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher-path", required=True, help="Path to a teacher_*.pt checkpoint.")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS_DEFAULT)
    parser.add_argument("--num-weather-cells", type=int, choices=(1, 2), default=NUM_WEATHER_CELLS_DEFAULT)
    parser.add_argument("--route-ids", nargs="*", default=ROUTE_IDS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--episode-step-cap", type=int, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="If given, also report how closely the teacher's argmax "
                             "matches the guide's labels on this dataset (exact / "
                             "within-1-bin / mean degree error).")
    parser.add_argument("--no-flight", action="store_true",
                        help="Skip flying; only run the --dataset-dir agreement check.")
    return parser.parse_args()


def _report_agreement(teacher, dataset_dir: str) -> None:
    """Near-miss-aware agreement of the teacher vs the guide on a labeled dataset.

    Exact top-1 undercounts an ordered action space; within-1-bin and mean degree
    error show the teacher's true quality (a good teacher can be ~5-10 deg off even
    at ~50% exact-match)."""
    data = load_trajectory_dataset(dataset_dir)
    m = teacher_agreement_metrics(teacher, data.local_obs, data.agent_index, data.action_idx)
    print(f"\n==== Teacher-vs-guide agreement on {dataset_dir} ====")
    print(f"  records           : {int(m['n'])}")
    print(f"  exact (top-1)     : {m['exact']:.1%}")
    print(f"  within 1 bin (5deg): {m['within1']:.1%}")
    print(f"  mean degree error : {m['deg_err']:.2f} deg")


def main() -> None:
    args = parse_args()
    teacher = load_teacher(args.teacher_path)
    teacher.eval()
    dev = next(teacher.parameters()).device
    print(f"Loaded teacher: {args.teacher_path}  (device={dev})")

    if args.dataset_dir:
        _report_agreement(teacher, args.dataset_dir)
    if args.no_flight:
        return

    print(f"\nFlying it as the controller over {args.n_episodes} games "
          f"| agents={args.num_agents} weather={args.num_weather_cells}")

    reset_options = {
        "num_agents": int(args.num_agents),
        "route_ids": _normalize_route_ids(args.route_ids),
        "num_weather_cells": int(args.num_weather_cells),
    }

    outcomes = {"success": 0, "collision": 0, "weather": 0, "truncated": 0}
    returns = []
    progress = tqdm(range(int(args.n_episodes)), desc="eval:teacher", unit="ep", dynamic_ncols=True)
    for ep_idx in progress:
        env = MultiAgentParallelEnv(
            num_agents=int(args.num_agents),
            max_agents=MAX_AGENTS,
            num_weather_cells=int(args.num_weather_cells),
        )
        if args.episode_step_cap is not None:
            env.core.max_step = int(args.episode_step_cap)
        observations, info = env.reset(seed=int(args.seed) + ep_idx, options=reset_options)
        common = info["__common__"]
        done = bool(common["episode_done"])
        truncated = bool(common["episode_truncated"])
        ep_return = 0.0
        steps = 0
        while not done and not truncated:
            active_ids = list(env.agents)
            if not active_ids:
                observations, _, _, _, info = env.step({})
            else:
                actions = _teacher_actions(teacher, observations, active_ids)
                observations, _, _, _, info = env.step(actions)
            common = info["__common__"]
            ep_return += float(common["team_reward"])
            done = bool(common["episode_done"])
            truncated = bool(common["episode_truncated"])
            steps += 1
            if steps > env.core.max_step + 5:
                break

        common = info["__common__"]
        reason = common["failure_reason"]
        if bool(common["episode_success"]):
            outcomes["success"] += 1
        elif reason == "collision":
            outcomes["collision"] += 1
        elif reason == "weather":
            outcomes["weather"] += 1
        elif bool(common["episode_truncated"]):
            outcomes["truncated"] += 1
        returns.append(ep_return)
        n_done = ep_idx + 1
        progress.set_postfix(
            succ=f"{outcomes['success'] / n_done:.0%}",
            coll=outcomes["collision"], wx=outcomes["weather"], trunc=outcomes["truncated"],
        )

    n = max(int(args.n_episodes), 1)
    mean_return = sum(returns) / len(returns) if returns else 0.0
    print("\n==== Teacher-as-controller results ====")
    print(f"  episodes        : {n}")
    print(f"  success   : {outcomes['success']:4d}  ({outcomes['success'] / n:.1%})")
    print(f"  collision : {outcomes['collision']:4d}  ({outcomes['collision'] / n:.1%})")
    print(f"  weather   : {outcomes['weather']:4d}  ({outcomes['weather'] / n:.1%})")
    print(f"  truncated : {outcomes['truncated']:4d}  ({outcomes['truncated'] / n:.1%})")
    print(f"  mean team reward : {mean_return:.2f}")


if __name__ == "__main__":
    main()
