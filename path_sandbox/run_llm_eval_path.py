#!/usr/bin/env python3
"""Pure-LLM evaluation harness (standalone, no PPO)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_only_eval import configs as cfg

from configs import EVAL_MEMORY_PATH, MEMORY_DIR, REWARD_PARAMS, START_LLM_AT_STEP
from rl_llm.env import StructuredEnv
from rl_llm.llm import (
    BasePromptBuilder,
    LLMGuidanceController,
    make_obs_snapshot,
    start_llm_log_dir,
)
import rl_llm.llm as llm_mod

ACTION_TO_DEG = [0, 5, 10, 15, 20, 25, 30, -5, -10, -15, -20, -25, -30]


def deg_to_action(turn_deg: int) -> int:
    """Map a heading delta in degrees to the closest discrete action bin."""
    return min(range(len(ACTION_TO_DEG)), key=lambda i: abs(ACTION_TO_DEG[i] - int(turn_deg)))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_gif(frames_dir: Path, out_gif_path: Path, fps: float) -> None:
    import imageio.v2 as imageio

    frames = sorted(frames_dir.glob("image_*.png"))
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}")
    images = [imageio.imread(str(f)) for f in frames]
    imageio.mimsave(str(out_gif_path), images, duration=1.0 / max(1e-6, fps))


def _cleanup_frames(frames_dir: Path) -> None:
    for png in frames_dir.glob("image_*.png"):
        try:
            png.unlink()
        except OSError:
            pass
    try:
        frames_dir.rmdir()
    except OSError:
        pass


def _write_summary(run_dir: Path, rows: List[dict]) -> None:
    total = len(rows)
    safe_count = sum(1 for r in rows if r["safe"])
    mean_min_sep = sum(r["min_sep"] for r in rows) / total if total else 0.0

    ttcp_vals = [r["min_ttcp"] for r in rows if r["min_ttcp"] is not None]
    mean_min_ttcp = sum(ttcp_vals) / len(ttcp_vals) if ttcp_vals else None

    mean_return = sum(r["return"] for r in rows) / total if total else 0.0

    summary = {
        "episodes": total,
        "safe_count": safe_count,
        "safe_rate": safe_count / total if total else 0.0,
        "mean_min_sep": mean_min_sep,
        "mean_min_ttcp": mean_min_ttcp,
        "mean_return": mean_return,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"episodes: {summary['episodes']}",
        f"safe: {summary['safe_count']}/{summary['episodes']} ({summary['safe_rate']:.2%})",
        f"mean min_sep: {summary['mean_min_sep']:.3f}",
        f"mean min_ttcp: {summary['mean_min_ttcp'] if summary['mean_min_ttcp'] is not None else 'n/a'}",
        f"mean return: {summary['mean_return']:.3f}",
    ]
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stub_evaluator(*_args, **_kwargs) -> dict:
    return {"ok": False, "score": 0.0, "comment": "disabled", "delta_action": None, "ppo": None}

def _heading_deg_from_obs(obs: np.ndarray) -> float:
    # obs[11] is own_heading_rad in the env observation vector.
    return float(math.degrees(float(obs[11])))

def _wrap_deg(d: float) -> float:
    return ((d + 180.0) % 360.0) - 180.0

def _log_apply_once(mem_path: Optional[str], seen: set, prev_obs: np.ndarray, obs: np.ndarray, turn_deg: int, step: int) -> None:
    if not mem_path or mem_path in seen:
        return
    try:
        rec = json.loads(Path(mem_path).read_text(encoding="utf-8"))
    except Exception:
        return
    before_h = _heading_deg_from_obs(prev_obs)
    after_h = _heading_deg_from_obs(obs)
    delta = _wrap_deg(after_h - before_h)
    rec["apply_log"] = {
        "applied_step": int(step),
        "heading_before_apply_deg": float(before_h),
        "heading_after_apply_deg": float(after_h),
        "delta_heading_deg": float(delta),
        "turn_commanded_deg": float(turn_deg),
    }
    try:
        Path(mem_path).write_text(json.dumps(rec, indent=2), encoding="utf-8")
        seen.add(mem_path)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone pure-LLM evaluation runner.")
    parser.add_argument("--episodes", type=int, default=cfg.DEFAULT_EPISODES)
    parser.add_argument("--mode", choices=["with_memory", "no_memory"], default="with_memory")
    parser.add_argument("--seed", type=int, default=cfg.BASE_SEED)
    parser.add_argument("--fps", type=float, default=cfg.FPS)
    parser.add_argument("--keep-frames", action="store_true", default=cfg.KEEP_FRAMES)
    parser.add_argument("--model", type=str, default=None, help="Override Ollama model name.")
    parser.add_argument("--enable-evaluator", action="store_true", help="Enable evaluator LLM call.")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = cfg.OUTPUT_ROOT / f"{args.mode}_{stamp}"
    gifs_dir = run_dir / "gifs"
    frames_root = run_dir / "frames"
    _ensure_dir(run_dir)
    _ensure_dir(gifs_dir)
    _ensure_dir(frames_root)

    if args.model:
        llm_mod.OLLAMA_MODEL = args.model

    if not args.enable_evaluator:
        llm_mod.call_llm_evaluator = _stub_evaluator

    if args.mode == "with_memory":
        mem_run_dir, mem_run_id = start_llm_log_dir(base_dir=MEMORY_DIR)
        save_dir = Path(mem_run_dir)
        memory_path = Path(EVAL_MEMORY_PATH)
        memory_mode = "shared"
    else:
        save_dir = run_dir / "memory"
        _ensure_dir(save_dir)
        mem_run_id = save_dir.name
        memory_path = run_dir / "eval_memory.json"
        memory_mode = "isolated"

    run_meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "mode": args.mode,
        "memory_mode": memory_mode,
        "episodes": args.episodes,
        "base_seed": args.seed,
        "seeds": [args.seed + i for i in range(args.episodes)],
        "output_dir": str(run_dir),
        "gifs_dir": str(gifs_dir),
        "frames_root": str(frames_root),
        "memory_dir": str(save_dir),
        "memory_index": str(save_dir.parent / "_index.jsonl"),
        "memory_path": str(memory_path),
        "ollama_model": llm_mod.OLLAMA_MODEL,
        "start_llm_at_step": START_LLM_AT_STEP,
        "enable_evaluator": bool(args.enable_evaluator),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    env = StructuredEnv(Reward_Params=REWARD_PARAMS, start_llm_at_step=START_LLM_AT_STEP)
    builder = BasePromptBuilder()
    ctl = LLMGuidanceController(
        builder=builder,
        k_history=5,
        save_dir=str(save_dir),
        start_llm_at_step=START_LLM_AT_STEP,
        memory_path=str(memory_path),
        run_id=mem_run_id,
    )

    rows: List[dict] = []
    applied_logged: set = set()

    for ep in range(1, args.episodes + 1):
        seed_i = args.seed + (ep - 1)
        random.seed(seed_i)
        np.random.seed(seed_i)

        obs, _ = env.reset(seed=seed_i)
        ctl.reset("WAIT_TURN")
        ctl.episode_id = ep

        frames_dir = frames_root / f"ep{ep:03d}"
        _ensure_dir(frames_dir)

        # Render initial frame
        env.render(show=False, folder=str(frames_dir) + "/")
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

        done = False
        total_reward = 0.0
        min_sep = float("inf")
        min_ttcp: Optional[float] = None
        prev_obs: Optional[np.ndarray] = None

        while not done:
            turn_deg, _ = ctl.next_action(
                obs,
                step=env.n_step + 1,
                fileinfo=getattr(env.agent, "fileinfo", None),
                env=None,
                ppo=None,
            )
            action = deg_to_action(turn_deg)

            prev_obs = obs
            obs, reward, term, trunc, _ = env.step(action)
            _log_apply_once(getattr(ctl, "_current_mem_path", None), applied_logged, prev_obs, obs, turn_deg, env.n_step)
            total_reward += float(reward)

            min_sep = min(min_sep, float(obs[6]))
            if prev_obs is not None:
                snap = make_obs_snapshot(np.array(obs, dtype=float), np.array(prev_obs, dtype=float), step=env.n_step)
                ttcp = snap.get("ttcp")
                if ttcp is not None:
                    min_ttcp = ttcp if min_ttcp is None else min(min_ttcp, float(ttcp))

            env.render(show=False, folder=str(frames_dir) + "/")
            try:
                import matplotlib.pyplot as plt
                plt.close("all")
            except Exception:
                pass

            done = term or trunc
            if env.n_step > env.MAX_STEP + 5:
                break

        gif_path = gifs_dir / f"ep{ep:03d}.gif"
        _save_gif(frames_dir, gif_path, fps=args.fps)
        if not args.keep_frames:
            _cleanup_frames(frames_dir)

        row = {
            "ep": ep,
            "min_sep": float(min_sep),
            "min_ttcp": None if min_ttcp is None else float(min_ttcp),
            "steps": int(env.n_step),
            "return": float(total_reward),
            "safe": bool(min_sep >= cfg.SAFE_R),
        }
        rows.append(row)

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ep", "min_sep", "min_ttcp", "steps", "return", "safe"],
        )
        writer.writeheader()
        writer.writerows(rows)

    _write_summary(run_dir, rows)
    print(f"[LLM_ONLY] Done. Outputs in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
