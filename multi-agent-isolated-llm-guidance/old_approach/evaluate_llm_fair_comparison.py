"""Pure-LLM (no MAPPO) evaluation across many episodes for fair comparison
between guidance sources.

This script runs the LLM controller directly on the environment, choosing every
action via the configured GUIDANCE_SOURCE. There is no policy network. There is
no MAPPO. The only thing being evaluated is the LLM controller (or one of its
synthetic substitutes) as an action selector.

Available guidance sources (one per invocation, via --guidance-source):
  - "real"          Call the configured Ollama model every guided step (slow, ~minutes/episode)
  - "uniform"       Uniform random from stage-allowed turns (fast)
  - "preview_safe"  Random from preview rows marked safe_over_preview=True (fast)
  - "best_preview"  Deterministic top-sorted preview row (fast, no randomness)

Output:
  Evaluation_LLM_only/<run-name>/llm_eval_<run-name>.json
    (override the parent with --output-dir; override the child folder with --run-name)

The JSON contains per-episode results plus aggregate outcome rates, in the same
shape as evaluate_rl.py for downstream comparison.

No GIFs or frames are saved by default. Per-episode metadata (seed, steps,
reward, success, failure_reason, num_llm_calls) is in the JSON.

The original evaluate_llm.py is left unchanged.
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
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from configs import (
    GUIDANCE_RANDOM_SEED,
    GUIDANCE_SOURCE,
    MAX_AGENTS,
    NUM_AGENTS_DEFAULT,
    OLLAMA_MODEL,
    NUM_WEATHER_CELLS_DEFAULT,
    ROUTE_IDS_DEFAULT,
    SEED,
)
from rl_llm_multi import GlobalLangGraphGuidanceController, JointGuidanceEnv
from rl_llm_multi.utils import max_agents_possible


_VALID_SOURCES = ("real", "uniform", "preview_safe", "best_preview")


@dataclass
class LLMEpisodeResult:
    episode_index: int
    seed: int
    steps: int
    total_reward: float
    success: bool
    truncated: bool
    failure_reason: Optional[str]
    num_agents: int
    num_weather_cells: int
    num_llm_calls: int          # number of steps where the controller produced actions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_route_ids(route_ids: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not route_ids:
        return None
    values = [str(rid).strip() for rid in route_ids if str(rid).strip()]
    return values or None


def _validate_num_agents(num_agents: int) -> int:
    capacity = max_agents_possible()
    if num_agents < 1 or num_agents > capacity:
        raise ValueError(
            f"num_agents must satisfy 1 <= num_agents <= {capacity}, got {num_agents}"
        )
    return int(num_agents)


def _validate_num_weather_cells(num_weather_cells: int) -> int:
    value = int(num_weather_cells)
    if value not in (1, 2):
        raise ValueError(f"num_weather_cells must be 1 or 2, got {num_weather_cells}")
    return value


def _validate_guidance_source(value: str) -> str:
    norm = str(value or "").strip().lower()
    if norm not in _VALID_SOURCES:
        raise ValueError(
            f"--guidance-source must be one of {_VALID_SOURCES}, got {value!r}"
        )
    return norm


def _safe_path_name(value: str) -> str:
    text = str(value or "").strip()
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    safe = "_".join(part for part in safe.split("_") if part)
    if safe in ("", ".", ".."):
        raise ValueError(f"Invalid output run name derived from {value!r}")
    return safe


def _effective_ollama_model(model_override: Optional[str]) -> str:
    value = model_override or os.environ.get("OLLAMA_MODEL") or OLLAMA_MODEL
    return value.strip() or OLLAMA_MODEL


def _default_output_run_name(args: argparse.Namespace, ollama_model: str) -> str:
    if args.output_run_name:
        return _safe_path_name(args.output_run_name)
    has_model_override = bool(args.ollama_model or os.environ.get("OLLAMA_MODEL"))
    if args.guidance_source == "real" and has_model_override:
        return _safe_path_name(f"real_{ollama_model}")
    return args.guidance_source


def _make_isolated_memory_path(tmp_root: Path, label: str) -> Path:
    """Create a fresh memory file outside the project so the eval does not
    pollute the project's persistent decision memory.
    """
    tmp_root.mkdir(parents=True, exist_ok=True)
    return tmp_root / f"decision_memory__{label}.json"


def _build_controller(
    *,
    guidance_source: str,
    guidance_random_seed: Optional[int],
    save_dir: Path,
    memory_path: Path,
) -> GlobalLangGraphGuidanceController:
    return GlobalLangGraphGuidanceController(
        save_dir=str(save_dir),
        memory_path=str(memory_path),
        memory_visual_audit_enabled=False,
        guidance_source=guidance_source,
        guidance_random_seed=guidance_random_seed,
    )


def _run_one_episode(
    *,
    env: JointGuidanceEnv,
    controller: GlobalLangGraphGuidanceController,
    episode_index: int,
    seed: int,
    num_agents: int,
    num_weather_cells: int,
    route_ids: Optional[List[str]],
    episode_step_cap: Optional[int],
) -> LLMEpisodeResult:
    """Run one episode entirely under LLM-controller actions, no rendering."""
    env.reset(
        seed=seed,
        num_agents=num_agents,
        route_ids=route_ids,
        num_weather_cells=num_weather_cells,
    )
    if episode_step_cap is not None:
        env.core.max_step = int(episode_step_cap)
    controller.reset()

    total_reward = 0.0
    llm_call_count = 0
    done = False
    truncated = False
    safety_step_limit = int(env.core.max_step) + 5

    while not done and not truncated:
        # latest_frame_path points at a file that does not exist on disk; the
        # controller is in text mode and handles missing frame files gracefully.
        actions = controller.choose_llm_actions(
            env.core,
            step=env.n_step,
            latest_frame_path=None,
            use_vision=False,
        )
        if actions:
            llm_call_count += 1
        _, reward, done, truncated, _ = env.step(actions)
        total_reward += float(reward)
        if env.core.n_step > safety_step_limit:
            break

    return LLMEpisodeResult(
        episode_index=int(episode_index),
        seed=int(seed),
        steps=int(env.core.n_step),
        total_reward=float(total_reward),
        success=bool(env.core.last_team_success),
        truncated=bool(truncated),
        failure_reason=env.core.last_failure_reason,
        num_agents=int(num_agents),
        num_weather_cells=int(num_weather_cells),
        num_llm_calls=int(llm_call_count),
    )


def _aggregate(results: Sequence[LLMEpisodeResult]) -> Dict[str, float]:
    if not results:
        return {
            "num_episodes": 0,
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "weather_rate": 0.0,
            "truncation_rate": 0.0,
            "mean_steps": 0.0,
            "mean_llm_calls": 0.0,
        }
    rewards = np.array([r.total_reward for r in results], dtype=float)
    steps = np.array([r.steps for r in results], dtype=float)
    calls = np.array([r.num_llm_calls for r in results], dtype=float)
    return {
        "num_episodes": int(len(results)),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std(ddof=1)) if rewards.size > 1 else 0.0,
        "sem_reward": (
            float(rewards.std(ddof=1) / np.sqrt(rewards.size))
            if rewards.size > 1 else 0.0
        ),
        "success_rate": float(np.mean([r.success for r in results])),
        "collision_rate": float(np.mean([r.failure_reason == "collision" for r in results])),
        "weather_rate": float(np.mean([r.failure_reason == "weather" for r in results])),
        "truncation_rate": float(np.mean([r.truncated for r in results])),
        "mean_steps": float(steps.mean()),
        "mean_llm_calls": float(calls.mean()),
    }


def _print_report(method_label: str, aggregate: Dict[str, float], args: argparse.Namespace) -> None:
    line = "=" * 90
    print()
    print(line)
    print(f"PURE-LLM EVAL REPORT  |  method={method_label}  "
          f"|  source={args.guidance_source}  |  episodes={aggregate['num_episodes']}  "
          f"|  seed={args.seed}")
    print(line)
    print(f"  mean_reward      : {aggregate['mean_reward']:8.2f}  +/- "
          f"{aggregate['std_reward']:.2f}  (SEM {aggregate['sem_reward']:.2f})")
    print(f"  success_rate     : {aggregate['success_rate']*100:6.2f}%")
    print(f"  collision_rate   : {aggregate['collision_rate']*100:6.2f}%")
    print(f"  weather_rate     : {aggregate['weather_rate']*100:6.2f}%")
    print(f"  truncation_rate  : {aggregate['truncation_rate']*100:6.2f}%")
    print(f"  mean_steps       : {aggregate['mean_steps']:6.2f}  (max=60)")
    print(f"  mean_llm_calls   : {aggregate['mean_llm_calls']:6.2f}  (per episode)")
    # Sanity check: 4 outcome rates should sum to ~1.0
    outcome_sum = (aggregate["success_rate"] + aggregate["collision_rate"]
                   + aggregate["weather_rate"] + aggregate["truncation_rate"])
    print(f"  sanity: rates sum= {outcome_sum:.4f}  (should be 1.0000)")
    print(line)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--guidance-source", type=str, default=GUIDANCE_SOURCE,
        help=f"One of {_VALID_SOURCES}. Defaults to configs.GUIDANCE_SOURCE "
             f"({GUIDANCE_SOURCE!r})."
    )
    parser.add_argument(
        "--guidance-random-seed", type=int, default=GUIDANCE_RANDOM_SEED,
        help="Seed for the controller's synthesis RNG (uniform / preview_safe). "
             "Has no effect for 'real' or 'best_preview'."
    )
    parser.add_argument(
        "--method-label", type=str, default=None,
        help="Human-readable label for the report and JSON. Defaults to the "
             "output run name."
    )
    parser.add_argument(
        "--ollama-model", type=str, default=None,
        help="Ollama model to use when --guidance-source=real. Defaults to "
             "the OLLAMA_MODEL environment variable, then configs.OLLAMA_MODEL "
             f"({OLLAMA_MODEL!r})."
    )

    # Scenario settings -- match evaluate_rl_fair_comparison.py defaults.
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS_DEFAULT)
    parser.add_argument("--num-weather-cells", type=int, choices=(1, 2), default=NUM_WEATHER_CELLS_DEFAULT)
    parser.add_argument("--route-ids", nargs="*", default=ROUTE_IDS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-episodes", type=int, default=300)
    parser.add_argument("--episode-step-cap", type=int, default=None)

    # Output.
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Parent folder; results go under <output-dir>/<run-name>/. "
             "Defaults to ./Evaluation_LLM_only"
    )
    parser.add_argument(
        "--run-name", dest="output_run_name", type=str, default=None,
        help="Output subfolder name. Defaults to <source>, or real_<ollama_model> "
             "when an Ollama model override is supplied for real guidance."
    )
    parser.add_argument(
        "--metrics-name", type=str, default=None,
        help="JSON filename (no extension). Defaults to llm_eval_<run-name>."
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    args.guidance_source = _validate_guidance_source(args.guidance_source)
    args.num_agents = _validate_num_agents(args.num_agents)
    args.num_weather_cells = _validate_num_weather_cells(args.num_weather_cells)
    route_ids = _normalize_route_ids(args.route_ids)
    effective_ollama_model = _effective_ollama_model(args.ollama_model)
    if args.ollama_model:
        os.environ["OLLAMA_MODEL"] = effective_ollama_model
    output_run_name = _default_output_run_name(args, effective_ollama_model)
    method_label = args.method_label or output_run_name

    # Output paths.
    output_root = (
        Path(args.output_dir) if args.output_dir
        else Path(__file__).resolve().parent / "Evaluation_LLM_only"
    )
    method_dir = output_root / output_run_name
    method_dir.mkdir(parents=True, exist_ok=True)
    metrics_stem = args.metrics_name or f"llm_eval_{output_run_name}"
    metrics_path = method_dir / f"{metrics_stem}.json"

    # Isolate transient artifacts (LLM prompt/response logs, decision memory)
    # from the project's persistent ones so the eval does not pollute training
    # memory or audit logs.
    tmp_root = Path(tempfile.mkdtemp(prefix=f"llm_eval_{output_run_name}_"))
    controller_log_dir = tmp_root / "controller_logs"
    controller_log_dir.mkdir(parents=True, exist_ok=True)
    memory_path = _make_isolated_memory_path(tmp_root, output_run_name)

    print(f"Pure-LLM evaluation for guidance_source = '{args.guidance_source}'")
    print(f"  output run name       : {output_run_name}")
    if args.guidance_source == "real":
        print(f"  ollama model          : {effective_ollama_model}")
    print(f"  episodes              : {args.n_episodes}")
    print(f"  seed range            : {args.seed} .. {args.seed + args.n_episodes - 1}")
    print(f"  agents / weather      : {args.num_agents} / {args.num_weather_cells}")
    print(f"  guidance_random_seed  : {args.guidance_random_seed}")
    print(f"  output JSON           : {metrics_path}")
    print(f"  scratch dir (auto)    : {tmp_root}")

    # Build env and controller once; reset per episode.
    env = JointGuidanceEnv(
        num_agents=args.num_agents,
        max_agents=MAX_AGENTS,
        num_weather_cells=args.num_weather_cells,
    )
    controller = _build_controller(
        guidance_source=args.guidance_source,
        guidance_random_seed=args.guidance_random_seed,
        save_dir=controller_log_dir,
        memory_path=memory_path,
    )

    n_total = int(args.n_episodes)
    progress = tqdm(
        range(n_total),
        total=n_total,
        desc=f"LLM:{args.guidance_source}",
        unit="ep",
        dynamic_ncols=True,
    )

    results: List[LLMEpisodeResult] = []
    successes = collisions = weather_failures = truncations = 0
    start_time = time.time()
    for ep_idx in progress:
        result = _run_one_episode(
            env=env,
            controller=controller,
            episode_index=ep_idx,
            seed=int(args.seed) + ep_idx,
            num_agents=int(args.num_agents),
            num_weather_cells=int(args.num_weather_cells),
            route_ids=route_ids,
            episode_step_cap=args.episode_step_cap,
        )
        results.append(result)
        successes += int(result.success)
        collisions += int(result.failure_reason == "collision")
        weather_failures += int(result.failure_reason == "weather")
        truncations += int(result.truncated)
        denom = float(len(results))
        progress.set_postfix(
            success=f"{successes/denom:.2f}",
            collision=f"{collisions/denom:.2f}",
            weather=f"{weather_failures/denom:.2f}",
            trunc=f"{truncations/denom:.2f}",
            mean_R=f"{float(np.mean([r.total_reward for r in results])):.1f}",
        )
    progress.close()
    elapsed = time.time() - start_time

    aggregate = _aggregate(results)
    aggregate["wall_clock_seconds"] = float(elapsed)

    _print_report(method_label=method_label, aggregate=aggregate, args=args)

    payload = {
        "method_label": method_label,
        "guidance_source": args.guidance_source,
        "guidance_random_seed": args.guidance_random_seed,
        "output_run_name": output_run_name,
        "ollama_model": effective_ollama_model if args.guidance_source == "real" else None,
        "seed": int(args.seed),
        "n_episodes": int(args.n_episodes),
        "num_agents": int(args.num_agents),
        "num_weather_cells": int(args.num_weather_cells),
        "route_ids": route_ids,
        "episode_step_cap": args.episode_step_cap,
        "aggregate": aggregate,
        "episodes": [asdict(r) for r in results],
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote results -> {metrics_path}")
    print(f"Wall clock      : {elapsed:.1f} sec  "
          f"({elapsed/max(1, len(results)):.2f} sec/episode)")
    print(f"Scratch dir     : {tmp_root}  (safe to delete after this run)")


if __name__ == "__main__":
    main()
