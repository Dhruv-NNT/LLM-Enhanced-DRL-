"""Parallel wrapper around generate_guided_trajectories.py.

Splits a target number of KEPT episodes across N worker processes, each of which
runs the single-process generator with a disjoint seed range and writes to its
own ``part_XX/`` subdirectory. The teacher dataset loader reads the parent
directory recursively, so the parts merge automatically — no manual merge step.

Because generation is CPU-simulator-bound (the safety-preview rollouts), running
several processes in parallel is the way to scale, not a GPU. Workers inherit
this shell's ``CUDA_VISIBLE_DEVICES``; export it empty for CPU before launching.

Example (CPU, no Ollama):
    export CUDA_VISIBLE_DEVICES=""
    python generate_parallel.py --guidance-source best_preview \
        --total-episodes 6000 --workers 12 --keep success

Output layout:
    distill_data/best_preview/
        part_00/  shard_*.npz  dataset_meta.json  gen.log
        part_01/  ...
        ...
"""

from __future__ import annotations

# --- path bootstrap: make the project root importable when this script is run
# from its subfolder, so `import configs` / `from rl_llm_multi ...` still resolve.
import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, _bootstrap_os.path.dirname(
    _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from configs import (
    DISTILL_DATASET_DIR,
    GUIDANCE_RANDOM_SEED,
    NUM_AGENTS_DEFAULT,
    NUM_WEATHER_CELLS_DEFAULT,
)

# Spacing between worker seed ranges. Each worker plays an unknown number of
# games (it runs until its kept-target is met); this stride is far larger than
# any worker will ever consume, so the per-worker seed ranges never overlap.
SEED_STRIDE = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--guidance-source", required=True,
                        help="best_preview | preview_safe | uniform | real")
    parser.add_argument("--total-episodes", type=int, default=6000,
                        help="Total KEPT episodes across all workers (default 6000).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel generator processes.")
    parser.add_argument("--keep", choices=("all", "success", "success_or_truncated"),
                        default="success")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for worker 0.")
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS_DEFAULT)
    parser.add_argument("--num-weather-cells", type=int, choices=(1, 2),
                        default=NUM_WEATHER_CELLS_DEFAULT)
    parser.add_argument("--route-ids", nargs="*", default=None)
    parser.add_argument("--ollama-model", type=str, default=None,
                        help="Only for --guidance-source real; note workers hit "
                             "Ollama concurrently, so use few workers.")
    parser.add_argument("--guidance-random-seed", type=int, default=GUIDANCE_RANDOM_SEED)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--episode-step-cap", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Parent output dir. Defaults to DISTILL_DATASET_DIR/<source>.")
    parser.add_argument("--poll-seconds", type=int, default=60,
                        help="How often to print a running-status line.")
    parser.add_argument("--math-threads", type=int, default=1,
                        help="Threads each worker may use for math (BLAS/OpenMP). "
                             "Default 1 so N workers map cleanly to N cores and do "
                             "not oversubscribe. Raise only if you run few workers.")
    parser.add_argument("--nice", type=int, default=10,
                        help="CPU-scheduling niceness for workers (0-19; higher = "
                             "lower priority). Default 10 so a shared machine's "
                             "interactive/other jobs are not starved. 0 = normal.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the worker commands and exit without launching.")
    return parser.parse_args()


# Env vars that control per-process math threading in the common BLAS/OpenMP
# backends. Pinning these keeps `--workers` processes from each spawning many
# math threads and thrashing the CPU.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def worker_env(math_threads: int) -> dict:
    """Copy the current environment and pin the math-thread count for workers."""
    env = dict(os.environ)
    for name in _THREAD_ENV_VARS:
        env[name] = str(max(1, int(math_threads)))
    return env


def split_target(total: int, workers: int) -> list[int]:
    """Split `total` into `workers` near-equal positive chunks summing to total."""
    workers = max(1, min(int(workers), int(total)))
    base = total // workers
    rem = total % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(DISTILL_DATASET_DIR) / args.guidance_source
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = split_target(int(args.total_episodes), int(args.workers))
    n_workers = len(chunks)
    generator = str(Path(__file__).resolve().parent / "generate_guided_trajectories.py")

    def worker_cmd(i: int, target: int) -> list[str]:
        part_dir = out_dir / f"part_{i:02d}"
        cmd = [
            sys.executable, generator,
            "--guidance-source", args.guidance_source,
            "--keep", args.keep,
            "--n-episodes", str(target),
            "--seed", str(int(args.seed) + i * SEED_STRIDE),
            "--guidance-random-seed", str(int(args.guidance_random_seed) + i),
            "--num-agents", str(args.num_agents),
            "--num-weather-cells", str(args.num_weather_cells),
            "--shard-size", str(args.shard_size),
            "--out-dir", str(part_dir),
        ]
        if args.route_ids:
            cmd += ["--route-ids", *[str(r) for r in args.route_ids]]
        if args.ollama_model:
            cmd += ["--ollama-model", str(args.ollama_model)]
        if args.episode_step_cap is not None:
            cmd += ["--episode-step-cap", str(args.episode_step_cap)]
        return cmd

    print(f"Parallel generation | source={args.guidance_source} | keep={args.keep}")
    print(f"  total kept target : {args.total_episodes}")
    print(f"  workers           : {n_workers}  (per-worker targets: {chunks})")
    print(f"  parent out dir    : {out_dir}")
    print(f"  device            : CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')!r}")
    print(f"  math threads/wkr  : {max(1, int(args.math_threads))}  "
          f"(pins {', '.join(_THREAD_ENV_VARS)})")
    print(f"  worker niceness   : {int(args.nice)}  (higher = lower priority; be a "
          f"good neighbor on shared machines)")

    if args.dry_run:
        for i, target in enumerate(chunks):
            print(f"\n[worker {i:02d}] " + " ".join(worker_cmd(i, target)))
        return

    env = worker_env(args.math_threads)
    nice_level = max(0, min(19, int(args.nice)))
    # preexec_fn runs in the child after fork: lower its scheduling priority so
    # co-tenants' jobs are not starved (POSIX only).
    preexec = (lambda: os.nice(nice_level)) if nice_level > 0 and os.name == "posix" else None
    procs = []
    logs = []
    start = time.time()
    for i, target in enumerate(chunks):
        part_dir = out_dir / f"part_{i:02d}"
        part_dir.mkdir(parents=True, exist_ok=True)
        log_path = part_dir / "gen.log"
        log = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            worker_cmd(i, target), stdout=log, stderr=subprocess.STDOUT,
            env=env, preexec_fn=preexec,
        )
        procs.append(proc)
        logs.append(log)
        print(f"  launched worker {i:02d} (pid {proc.pid}, target {target}) -> {log_path}")

    # Poll until all workers exit.
    try:
        while True:
            running = [i for i, p in enumerate(procs) if p.poll() is None]
            if not running:
                break
            elapsed = int(time.time() - start)
            kept = _sum_kept(out_dir, n_workers)
            print(f"  [{elapsed}s] running={len(running)}/{n_workers} "
                  f"workers={running} | kept so far ~{kept}/{args.total_episodes}", flush=True)
            time.sleep(max(5, int(args.poll_seconds)))
    except KeyboardInterrupt:
        print("Interrupted — terminating workers ...")
        for p in procs:
            p.terminate()
        raise
    finally:
        for log in logs:
            log.close()

    codes = [p.wait() for p in procs]
    failed = [i for i, c in enumerate(codes) if c != 0]
    total_kept, total_records, total_played = _aggregate(out_dir, n_workers)
    elapsed = int(time.time() - start)
    print(f"\nDone in {elapsed}s. kept={total_kept} records={total_records} "
          f"played={total_played} -> {out_dir}")
    if failed:
        print(f"WARNING: workers with non-zero exit: {failed} "
              f"(see part_XX/gen.log). Dataset still usable from the successful parts.")
        sys.exit(1)


def _sum_kept(out_dir: Path, n_workers: int) -> int:
    """Best-effort running count from any metas already written (finished workers)."""
    total = 0
    for i in range(n_workers):
        meta = out_dir / f"part_{i:02d}" / "dataset_meta.json"
        if meta.exists():
            try:
                total += int(json.loads(meta.read_text()).get("episodes_kept", 0))
            except Exception:
                pass
    return total


def _aggregate(out_dir: Path, n_workers: int) -> tuple[int, int, int]:
    kept = records = played = 0
    for i in range(n_workers):
        meta = out_dir / f"part_{i:02d}" / "dataset_meta.json"
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                kept += int(m.get("episodes_kept", 0))
                records += int(m.get("n_records", 0))
                played += int(m.get("episodes_played", 0))
            except Exception:
                pass
    return kept, records, played


if __name__ == "__main__":
    main()
