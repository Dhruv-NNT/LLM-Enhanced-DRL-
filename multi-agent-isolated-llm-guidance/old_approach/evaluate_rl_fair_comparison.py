"""Evaluate a trained MAPPO policy across multiple late checkpoints to mitigate
PPO snapshot noise and report mean +/- std on outcome rates.

Use this when comparing methods against each other for benchmarking. A single
checkpoint can swing 5-15 reward points across two PPO updates near the end of
training; averaging across the last N checkpoints absorbs that noise so the
final comparison number is defensible.

Compared to evaluate_rl.py (which evaluates one checkpoint), this script:
  - Accepts a weights folder via --weights-dir and auto-selects the last K
    numbered checkpoints (plus last_checkpoint.pt / last.ckpt), OR
  - Accepts an explicit list via --model-paths
  - Runs the same N-episode eval on each checkpoint with the same --seed
  - Aggregates per-checkpoint outcome rates into mean +/- std
  - Writes a combined JSON containing every checkpoint's per-episode detail
    plus the aggregate summary

The original evaluate_rl.py is unchanged.
"""

from __future__ import annotations

# --- path bootstrap: make the project root importable when this script is run
# from its subfolder, so `import configs` / `from rl_llm_multi ...` still resolve.
import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, _bootstrap_os.path.dirname(
    _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

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


# ---------------------------------------------------------------------------
# Per-episode result type (identical to evaluate_rl.py).
# ---------------------------------------------------------------------------
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


@dataclass
class CheckpointSummary:
    """Aggregate outcomes from one checkpoint's eval pass."""
    model_path: str
    model_label: str           # short label e.g. "checkpoint_4956248" or "last"
    agent_steps_in_filename: Optional[int]
    num_episodes: int
    mean_reward: float
    std_reward: float
    success_rate: float
    collision_rate: float
    weather_rate: float
    truncation_rate: float
    mean_steps: float
    episodes: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers shared with evaluate_rl.py (identical implementations).
# ---------------------------------------------------------------------------
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
    observations, info = env.reset(seed=seed, options=reset_options)
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
            observations, _, _, _, info = env.step({})
        else:
            actions, _, _ = agent.select_actions(
                global_state=env.state(),
                local_observations=observations,
                active_agent_ids=active_ids,
                episode_id=episode_index,
                deterministic=True,
                store=False,
            )
            observations, _, _, _, info = env.step(actions)
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


# ---------------------------------------------------------------------------
# New helpers for multi-checkpoint comparison.
# ---------------------------------------------------------------------------
_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)\.pt$")
_LAST_NAMES = ("last_checkpoint.pt", "last.ckpt")


def _extract_step_from_filename(path: Path) -> Optional[int]:
    """Return the agent_steps encoded in checkpoint_<step>.pt names; None for last_*."""
    m = _CHECKPOINT_RE.match(path.name)
    if m is None:
        return None
    return int(m.group(1))


def _resolve_checkpoint_list(
    *,
    weights_dir: Optional[str],
    num_checkpoints: int,
    include_last: bool,
    explicit_paths: Optional[Sequence[str]],
) -> List[Path]:
    """Build the ordered list of checkpoint paths to evaluate.

    Precedence:
      1. If --model-paths was given, use that list verbatim.
      2. Else if --weights-dir was given:
           - select the last num_checkpoints numbered checkpoints (by step)
           - optionally append last_checkpoint.pt / last.ckpt if include_last is True
    """
    if explicit_paths:
        resolved = [Path(p) for p in explicit_paths]
        for p in resolved:
            if not p.exists():
                raise FileNotFoundError(f"Model path does not exist: {p}")
        return resolved

    if weights_dir is None:
        raise ValueError("Either --model-paths or --weights-dir must be provided")

    folder = Path(weights_dir)
    if not folder.is_dir():
        raise NotADirectoryError(f"Weights directory does not exist: {folder}")

    numbered: List[Path] = []
    for entry in folder.iterdir():
        step = _extract_step_from_filename(entry)
        if step is not None:
            numbered.append(entry)
    numbered.sort(key=lambda p: _extract_step_from_filename(p) or 0)

    if num_checkpoints > 0:
        numbered = numbered[-num_checkpoints:]

    resolved: List[Path] = list(numbered)
    if include_last:
        for last_name in _LAST_NAMES:
            candidate = folder / last_name
            if candidate.exists():
                resolved.append(candidate)
                break

    if not resolved:
        raise FileNotFoundError(
            f"No checkpoint files matched in {folder}. "
            "Looked for checkpoint_<step>.pt and last_checkpoint.pt / last.ckpt."
        )
    return resolved


def _checkpoint_label(path: Path) -> str:
    """Short human-readable label for a checkpoint (e.g. 'checkpoint_4956248' or 'last')."""
    if path.name in _LAST_NAMES:
        return "last"
    if path.suffix == ".pt" and path.stem.startswith("checkpoint_"):
        return path.stem
    return path.stem


def _evaluate_one_checkpoint(
    *,
    model_path: Path,
    agent: MAPPO,
    args: argparse.Namespace,
    route_ids: Optional[List[str]],
    output_root: Path,
    checkpoint_idx: int,
    total_checkpoints: int,
) -> CheckpointSummary:
    """Run --n-episodes evaluation episodes on the model at model_path."""
    agent.load_model(model_path)
    n_total = int(args.n_episodes)

    desc = f"[{checkpoint_idx + 1}/{total_checkpoints}] {model_path.name}"
    progress = tqdm(
        range(n_total),
        total=n_total,
        desc=desc,
        unit="ep",
        dynamic_ncols=True,
    )
    results: List[RLEpisodeResult] = []
    successes = collisions = weather_failures = truncations = 0
    for ep_idx in progress:
        result = rollout_episode(
            agent,
            episode_index=ep_idx,
            seed=int(args.seed) + ep_idx,
            num_agents=int(args.num_agents),
            num_weather_cells=int(args.num_weather_cells),
            route_ids=route_ids,
            episode_step_cap=args.episode_step_cap,
            frame_dir=None,
            gif_path=None,
        )
        results.append(result)
        successes += int(bool(result.success))
        collisions += int(result.failure_reason == "collision")
        weather_failures += int(result.failure_reason == "weather")
        truncations += int(bool(result.truncated))
        denom = float(len(results))
        progress.set_postfix(
            success=f"{successes / denom:.2f}",
            collision=f"{collisions / denom:.2f}",
            weather=f"{weather_failures / denom:.2f}",
            trunc=f"{truncations / denom:.2f}",
            mean_R=f"{float(np.mean([r.total_reward for r in results])):.1f}",
        )
    progress.close()

    rewards = np.array([r.total_reward for r in results], dtype=float)
    steps = np.array([r.steps for r in results], dtype=float)

    return CheckpointSummary(
        model_path=str(model_path),
        model_label=_checkpoint_label(model_path),
        agent_steps_in_filename=_extract_step_from_filename(model_path),
        num_episodes=len(results),
        mean_reward=float(rewards.mean()) if rewards.size else 0.0,
        std_reward=float(rewards.std(ddof=1)) if rewards.size > 1 else 0.0,
        success_rate=float(np.mean([r.success for r in results])) if results else 0.0,
        collision_rate=float(np.mean([r.failure_reason == "collision" for r in results])) if results else 0.0,
        weather_rate=float(np.mean([r.failure_reason == "weather" for r in results])) if results else 0.0,
        truncation_rate=float(np.mean([r.truncated for r in results])) if results else 0.0,
        mean_steps=float(steps.mean()) if steps.size else 0.0,
        episodes=[asdict(r) for r in results],
    )


def _aggregate_across_checkpoints(summaries: Sequence[CheckpointSummary]) -> Dict[str, float]:
    """Compute mean / std (ddof=1) of each rate across the per-checkpoint summaries."""
    if not summaries:
        return {}
    fields = ("mean_reward", "success_rate", "collision_rate", "weather_rate",
              "truncation_rate", "mean_steps")
    out: Dict[str, float] = {"num_checkpoints": len(summaries)}
    for field in fields:
        values = np.array([getattr(s, field) for s in summaries], dtype=float)
        out[f"{field}__mean"] = float(values.mean())
        out[f"{field}__std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        out[f"{field}__sem"] = (
            float(values.std(ddof=1) / np.sqrt(values.size))
            if values.size > 1 else 0.0
        )
        out[f"{field}__min"] = float(values.min())
        out[f"{field}__max"] = float(values.max())
    return out


def _format_pct(mean: float, std: float, sem: float) -> str:
    return f"{mean*100:5.2f}% +/- {std*100:4.2f}% (SEM {sem*100:4.2f}%)"


def _print_report(
    *,
    method_label: str,
    summaries: Sequence[CheckpointSummary],
    aggregate: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    line = "=" * 96
    print()
    print(line)
    print(f"FAIR COMPARISON REPORT  |  method={method_label}  "
          f"|  checkpoints={len(summaries)}  |  episodes_each={args.n_episodes}  "
          f"|  seed={args.seed}")
    print(line)

    print(f"{'#':>2}  {'checkpoint':<28}  {'agent_steps':>12}  "
          f"{'mean_R':>8}  {'succ':>6}  {'coll':>6}  {'wx':>6}  {'trnc':>6}  {'len':>5}")
    print("-" * 96)
    for i, s in enumerate(summaries, 1):
        steps_str = f"{s.agent_steps_in_filename}" if s.agent_steps_in_filename else "(last_ckpt)"
        print(
            f"{i:>2}  {s.model_label:<28}  {steps_str:>12}  "
            f"{s.mean_reward:>8.2f}  {s.success_rate:>6.3f}  {s.collision_rate:>6.3f}  "
            f"{s.weather_rate:>6.3f}  {s.truncation_rate:>6.3f}  {s.mean_steps:>5.1f}"
        )
    print("-" * 96)
    print()

    print("AGGREGATE  (mean +/- std across the checkpoints above)")
    print(f"  mean_reward     :  {aggregate.get('mean_reward__mean', 0):8.2f}  "
          f"+/- {aggregate.get('mean_reward__std', 0):6.2f}  "
          f"(SEM {aggregate.get('mean_reward__sem', 0):5.2f})  "
          f"[min {aggregate.get('mean_reward__min', 0):6.2f}, "
          f"max {aggregate.get('mean_reward__max', 0):6.2f}]")
    for key, label in (
        ("success_rate", "success_rate   "),
        ("collision_rate", "collision_rate "),
        ("weather_rate", "weather_rate   "),
        ("truncation_rate", "truncation_rate"),
    ):
        m = aggregate.get(f"{key}__mean", 0.0)
        sd = aggregate.get(f"{key}__std", 0.0)
        sem = aggregate.get(f"{key}__sem", 0.0)
        mn = aggregate.get(f"{key}__min", 0.0)
        mx = aggregate.get(f"{key}__max", 0.0)
        print(f"  {label}:  {_format_pct(m, sd, sem)}  "
              f"[min {mn*100:5.2f}%, max {mx*100:5.2f}%]")
    print("-" * 96)


# ---------------------------------------------------------------------------
# Argument parsing.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Checkpoint selection (mutually-exclusive in spirit, but argparse allows both).
    parser.add_argument(
        "--weights-dir", type=str, default=None,
        help="Folder containing checkpoint_<step>.pt and last_checkpoint.pt / last.ckpt. "
             "The last --num-checkpoints numbered files are selected and (by default) "
             "last_checkpoint.pt is appended."
    )
    parser.add_argument(
        "--model-paths", nargs="+", default=None,
        help="Explicit list of checkpoint files. Overrides --weights-dir if both are passed."
    )
    parser.add_argument(
        "--num-checkpoints", type=int, default=5,
        help="Number of most recent numbered checkpoints to include from --weights-dir (default 5)."
    )
    parser.add_argument(
        "--no-last-checkpoint", action="store_true",
        help="Do NOT append last_checkpoint.pt / last.ckpt to the list."
    )

    # Method labeling for the report.
    parser.add_argument(
        "--method-label", type=str, default="(unnamed)",
        help="Human-readable label for this method, printed in the report."
    )

    # Standard eval flags (same defaults as evaluate_rl.py).
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS_DEFAULT)
    parser.add_argument("--num-weather-cells", type=int, choices=(1, 2), default=NUM_WEATHER_CELLS_DEFAULT)
    parser.add_argument("--route-ids", nargs="*", default=ROUTE_IDS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-episodes", type=int, default=MAPPO_N_EVAL_EPISODES)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to place transient frame artifacts (currently unused without --save-gif).")
    parser.add_argument("--metrics-path", type=str, default=None,
                        help="Combined JSON output path. Defaults to "
                             "<MAPPO_METRICS_DIR>/eval_mappo_fair_<timestamp>.json")
    parser.add_argument("--episode-step-cap", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    model_paths = _resolve_checkpoint_list(
        weights_dir=args.weights_dir,
        num_checkpoints=int(args.num_checkpoints),
        include_last=not bool(args.no_last_checkpoint),
        explicit_paths=args.model_paths,
    )

    route_ids = _normalize_route_ids(args.route_ids)

    # Probe env once to learn dims; agent is reused across checkpoints via load_model.
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
    agent = build_agent(
        int(probe_env.state().shape[0]),
        int(probe_env.observation_space(probe_env.possible_agents[0]).shape[0]),
    )

    output_root = Path(args.output_dir) if args.output_dir else MAPPO_EVAL_OUTPUT_DIR / f"eval_fair_{_timestamp()}"
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_dir = MAPPO_METRICS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating method '{args.method_label}' across {len(model_paths)} checkpoint(s):")
    for idx, p in enumerate(model_paths, 1):
        print(f"  [{idx}] {p}")
    print(f"Per-checkpoint episodes : {args.n_episodes}")
    print(f"Episode seed offset     : {args.seed} -> {args.seed + args.n_episodes - 1}")
    print(f"Agents / weather cells  : {args.num_agents} / {args.num_weather_cells}")

    summaries: List[CheckpointSummary] = []
    for idx, model_path in enumerate(model_paths):
        summary = _evaluate_one_checkpoint(
            model_path=model_path,
            agent=agent,
            args=args,
            route_ids=route_ids,
            output_root=output_root,
            checkpoint_idx=idx,
            total_checkpoints=len(model_paths),
        )
        summaries.append(summary)

    aggregate = _aggregate_across_checkpoints(summaries)
    _print_report(method_label=args.method_label, summaries=summaries,
                  aggregate=aggregate, args=args)

    metrics_path = (
        Path(args.metrics_path)
        if args.metrics_path
        else metrics_dir / f"eval_mappo_fair_{_timestamp()}.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method_label": args.method_label,
        "seed": int(args.seed),
        "n_episodes_per_checkpoint": int(args.n_episodes),
        "num_agents": int(args.num_agents),
        "num_weather_cells": int(args.num_weather_cells),
        "route_ids": route_ids,
        "episode_step_cap": args.episode_step_cap,
        "model_paths": [str(p) for p in model_paths],
        "aggregate": aggregate,
        "per_checkpoint": [asdict(s) for s in summaries],
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote combined report -> {metrics_path}")


if __name__ == "__main__":
    main()
