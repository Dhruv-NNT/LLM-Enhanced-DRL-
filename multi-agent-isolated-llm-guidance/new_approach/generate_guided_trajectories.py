"""Offline trajectory generation for dual-teacher distillation (Stage A).

Runs the guidance controller as a pure controller (no MAPPO) and records, for
every advised agent at every guided step, the raw local observation and the
action the guide chose. One invocation = one guidance source = one dataset.

Run it twice to build both teachers' data:

    python3 generate_guided_trajectories.py --guidance-source best_preview --n-episodes 200
    python3 generate_guided_trajectories.py --guidance-source real         --n-episodes 200

Outputs (under --out-dir, default ./distill_data/<source>/):
    shard_0000.npz, shard_0001.npz, ...   arrays: local_obs, agent_index,
                                          action_idx, episode_id, step
    dataset_meta.json                     dims, action bins, generation args,
                                          outcome counts

The "best_preview" source needs no Ollama; only "real" calls the LLM. This is
the one-time amortization cost: the slow LLM is paid here, never during MAPPO
training.
"""

from __future__ import annotations

# --- path bootstrap: make the project root importable when this script is run
# from its subfolder, so `import configs` / `from rl_llm_multi ...` still resolve.
import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, _bootstrap_os.path.dirname(
    _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))

import argparse
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from configs import (
    ACTION_BINS,
    DISTILL_DATASET_DIR,
    GUIDANCE_RANDOM_SEED,
    MAX_AGENTS,
    NUM_AGENTS_DEFAULT,
    NUM_WEATHER_CELLS_DEFAULT,
    OLLAMA_MODEL,
    ROUTE_IDS_DEFAULT,
    SEED,
)
from rl_llm_multi import GlobalLangGraphGuidanceController, JointGuidanceEnv
from rl_llm_multi.mappo import agent_id_to_index
from rl_llm_multi.teacher import save_trajectory_shard, write_dataset_meta
from rl_llm_multi.utils import deg_to_action_idx, max_agents_possible

_VALID_SOURCES = ("real", "uniform", "preview_safe", "best_preview")


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
        raise ValueError(f"num_agents must satisfy 1 <= n <= {capacity}, got {num_agents}")
    return int(num_agents)


def _validate_guidance_source(value: str) -> str:
    norm = str(value or "").strip().lower()
    if norm not in _VALID_SOURCES:
        raise ValueError(f"--guidance-source must be one of {_VALID_SOURCES}, got {value!r}")
    return norm


class _ShardWriter:
    """Buffers records and flushes them to npz shards of bounded size."""

    def __init__(self, out_dir: Path, shard_size: int) -> None:
        self.out_dir = out_dir
        self.shard_size = int(shard_size)
        self.shard_index = 0
        self.total = 0
        self._reset_buffers()
        self._stage: List[tuple] = []

    def _reset_buffers(self) -> None:
        self._obs: List[np.ndarray] = []
        self._agent_idx: List[int] = []
        self._action_idx: List[int] = []
        self._episode_id: List[int] = []
        self._step: List[int] = []

    def add(self, *, local_obs, agent_index, action_idx, episode_id, step) -> None:
        self._obs.append(np.asarray(local_obs, dtype=np.float32))
        self._agent_idx.append(int(agent_index))
        self._action_idx.append(int(action_idx))
        self._episode_id.append(int(episode_id))
        self._step.append(int(step))
        if len(self._obs) >= self.shard_size:
            self.flush()

    # --- Episode-level staging: hold an episode's records until its outcome is
    # known, then keep (commit) or discard (drop) them as a unit. Lets us write
    # successful-episode-only datasets (see --keep).
    def stage(self, *, local_obs, agent_index, action_idx, episode_id, step) -> None:
        self._stage.append(
            (np.asarray(local_obs, dtype=np.float32), int(agent_index),
             int(action_idx), int(episode_id), int(step))
        )

    def commit_episode(self) -> int:
        n = len(self._stage)
        for obs, agent_index, action_idx, episode_id, step in self._stage:
            self.add(local_obs=obs, agent_index=agent_index,
                     action_idx=action_idx, episode_id=episode_id, step=step)
        self._stage = []
        return n

    def drop_episode(self) -> None:
        self._stage = []

    def flush(self) -> None:
        if not self._obs:
            return
        save_trajectory_shard(
            self.out_dir,
            self.shard_index,
            local_obs=np.stack(self._obs, axis=0),
            agent_index=np.asarray(self._agent_idx, dtype=np.int64),
            action_idx=np.asarray(self._action_idx, dtype=np.int64),
            episode_id=np.asarray(self._episode_id, dtype=np.int64),
            step=np.asarray(self._step, dtype=np.int64),
        )
        self.total += len(self._obs)
        self.shard_index += 1
        self._reset_buffers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guidance-source", type=str, default="best_preview")
    parser.add_argument("--guidance-random-seed", type=int, default=GUIDANCE_RANDOM_SEED)
    parser.add_argument("--ollama-model", type=str, default=None)
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS_DEFAULT)
    parser.add_argument("--num-weather-cells", type=int, choices=(1, 2), default=NUM_WEATHER_CELLS_DEFAULT)
    parser.add_argument("--route-ids", nargs="*", default=ROUTE_IDS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--n-episodes", type=int, default=6000,
        help="TARGET number of KEPT episodes (not games played). The generator "
             "keeps playing new games until this many episodes have passed the "
             "--keep filter. With --keep success this is the number of successful "
             "episodes collected.",
    )
    parser.add_argument(
        "--keep", choices=("all", "success", "success_or_truncated"), default="success",
        help="Which episodes to keep records from. 'success' (default) keeps only "
             "episodes where all aircraft reached their destination (cleanest "
             "distillation target); 'success_or_truncated' also keeps "
             "ran-out-of-time episodes but drops collision/weather crashes; "
             "'all' keeps every episode.",
    )
    parser.add_argument("--episode-step-cap", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory. Defaults to configs.DISTILL_DATASET_DIR/<source>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.guidance_source = _validate_guidance_source(args.guidance_source)
    args.num_agents = _validate_num_agents(args.num_agents)
    route_ids = _normalize_route_ids(args.route_ids)
    if args.ollama_model:
        os.environ["OLLAMA_MODEL"] = args.ollama_model
    effective_model = os.environ.get("OLLAMA_MODEL") or OLLAMA_MODEL

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else Path(DISTILL_DATASET_DIR) / args.guidance_source
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Isolate controller scratch artifacts (prompt logs, decision memory) so we
    # do not touch the project's persistent memory/logs.
    tmp_root = Path(tempfile.mkdtemp(prefix=f"gen_traj_{args.guidance_source}_"))
    controller_log_dir = tmp_root / "controller_logs"
    controller_log_dir.mkdir(parents=True, exist_ok=True)
    memory_path = tmp_root / "decision_memory.json"

    print(f"Generating trajectories | source={args.guidance_source}")
    print(f"  target kept   : {args.n_episodes} episodes (keep='{args.keep}')")
    print(f"  seed start    : {args.seed} (plays as many games as needed)")
    print(f"  agents/weather: {args.num_agents} / {args.num_weather_cells}")
    if args.guidance_source == "real":
        print(f"  ollama model  : {effective_model}")
    print(f"  out dir       : {out_dir}")

    env = JointGuidanceEnv(
        num_agents=args.num_agents,
        max_agents=MAX_AGENTS,
        num_weather_cells=args.num_weather_cells,
    )
    controller = GlobalLangGraphGuidanceController(
        save_dir=str(controller_log_dir),
        memory_path=str(memory_path),
        memory_visual_audit_enabled=False,
        guidance_source=args.guidance_source,
        guidance_random_seed=args.guidance_random_seed,
    )

    writer = _ShardWriter(out_dir, args.shard_size)
    outcomes = {"success": 0, "collision": 0, "weather": 0, "truncated": 0}
    episodes_kept = 0
    start = time.time()

    # `--n-episodes` is the TARGET number of KEPT episodes. We keep generating
    # (playing new games with fresh seeds) until that many episodes have passed
    # the --keep filter. There is no safety cap: with --keep success the loop runs
    # until it has collected exactly `--n-episodes` successful episodes.
    target_kept = int(args.n_episodes)
    progress = tqdm(total=target_kept, desc=f"gen:{args.guidance_source}", unit="ep", dynamic_ncols=True)
    attempt = 0
    while episodes_kept < target_kept:
        seed = int(args.seed) + attempt
        attempt += 1
        joint_obs, _ = env.reset(
            seed=seed,
            num_agents=args.num_agents,
            route_ids=route_ids,
            num_weather_cells=args.num_weather_cells,
        )
        if args.episode_step_cap is not None:
            env.core.max_step = int(args.episode_step_cap)
        controller.reset()

        done = False
        truncated = False
        safety_limit = int(env.core.max_step) + 5
        while not done and not truncated:
            step_now = int(env.n_step)
            local_obs_map = joint_obs.get("local_observations", {}) or {}
            actions = controller.choose_llm_actions(
                env.core,
                step=step_now,
                latest_frame_path=None,
                use_vision=False,
            )
            for agent_id, action_deg in (actions or {}).items():
                obs_vec = local_obs_map.get(agent_id)
                if obs_vec is None:
                    continue
                try:
                    agent_index = agent_id_to_index(agent_id, MAX_AGENTS)
                except ValueError:
                    continue
                writer.stage(
                    local_obs=obs_vec,
                    agent_index=agent_index,
                    action_idx=deg_to_action_idx(int(action_deg), ACTION_BINS),
                    episode_id=episodes_kept,   # id among kept episodes
                    step=step_now,
                )
            joint_obs, _, done, truncated, _ = env.step(actions)
            if env.core.n_step > safety_limit:
                break

        team_success = bool(env.core.last_team_success)
        failure_reason = env.core.last_failure_reason
        if team_success:
            outcomes["success"] += 1
        elif failure_reason == "collision":
            outcomes["collision"] += 1
        elif failure_reason == "weather":
            outcomes["weather"] += 1
        elif bool(truncated):
            outcomes["truncated"] += 1

        # Keep or discard this episode's staged records as a unit (see --keep).
        if args.keep == "success":
            keep_episode = team_success
        elif args.keep == "success_or_truncated":
            keep_episode = team_success or failure_reason not in ("collision", "weather")
        else:  # "all"
            keep_episode = True
        if keep_episode:
            writer.commit_episode()
            episodes_kept += 1
            progress.update(1)
        else:
            writer.drop_episode()
        progress.set_postfix(
            played=attempt,
            records=writer.total + len(writer._obs),
            **{k: outcomes[k] for k in outcomes},
        )

    episodes_played = attempt
    writer.flush()
    progress.close()
    elapsed = time.time() - start

    meta = {
        "created_at": _timestamp(),
        "guidance_source": args.guidance_source,
        "ollama_model": effective_model if args.guidance_source == "real" else None,
        "obs_dim": int(env.core.local_obs_dim),
        "action_dim": int(len(ACTION_BINS)),
        "action_bins": list(ACTION_BINS),
        "max_agents": int(MAX_AGENTS),
        "num_agents": int(args.num_agents),
        "num_weather_cells": int(args.num_weather_cells),
        "route_ids": route_ids,
        "seed_start": int(args.seed),
        "target_kept": int(args.n_episodes),
        "keep": str(args.keep),
        "episodes_kept": int(episodes_kept),
        "episodes_played": int(episodes_played),
        "n_records": int(writer.total),
        "n_shards": int(writer.shard_index),
        "episode_step_cap": args.episode_step_cap,
        "outcomes": outcomes,
        "wall_clock_seconds": float(elapsed),
    }
    write_dataset_meta(out_dir, meta)

    print(f"\nWrote {writer.total} records in {writer.shard_index} shard(s) -> {out_dir}")
    print(f"Kept {episodes_kept} episodes from {episodes_played} played (keep='{args.keep}')")
    print(f"Outcomes: {outcomes}")
    print(f"Wall clock: {elapsed:.1f}s ({elapsed / max(1, args.n_episodes):.2f}s/episode)")
    print(f"Scratch dir (safe to delete): {tmp_root}")


if __name__ == "__main__":
    main()
