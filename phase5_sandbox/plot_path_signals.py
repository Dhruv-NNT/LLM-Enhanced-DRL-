#!/usr/bin/env python3
"""Plot key path-related signals for one LLM-guided episode.

Usage:
  python phase5_sandbox/plot_path_signals.py --seed 123 --steps 40 --episodes 1
Outputs PNGs under phase5_sandbox/plots/ and a CSV of sampled signals.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Use the top-level configs (two directories up from here)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import REWARD_PARAMS, START_LLM_AT_STEP  # type: ignore
from rl_llm.env import StructuredEnv
from phase5_sandbox.rl_llm_phase5.llm import (
    BasePromptBuilder,
    LLMGuidanceController,
    make_obs_snapshot,
)

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def wrap_deg(a: float) -> float:
    return ((a + 180.0) % 360.0) - 180.0


def bearing_error_to_intruder(obs: np.ndarray) -> float:
    own_x, own_y, intr_x, intr_y = obs[0], obs[1], obs[2], obs[3]
    own_h_deg = math.degrees(float(obs[11]))
    intr_bearing = math.degrees(math.atan2(intr_y - own_y, intr_x - own_x))
    return wrap_deg(intr_bearing - own_h_deg)


def heading_error_to_dest_deg(obs: np.ndarray) -> float:
    own_x, own_y, dest_x, dest_y = obs[0], obs[1], obs[4], obs[5]
    own_h_deg = math.degrees(float(obs[11]))
    dest_bearing = math.degrees(math.atan2(dest_y - own_y, dest_x - own_x))
    return wrap_deg(dest_bearing - own_h_deg)


ACTION_TO_DEG = [0, 5, 10, 15, 20, 25, 30, -5, -10, -15, -20, -25, -30]


def deg_to_action(turn_deg: int) -> int:
    """Map a heading delta in degrees to the closest discrete action bin."""
    return min(range(len(ACTION_TO_DEG)), key=lambda i: abs(ACTION_TO_DEG[i] - int(turn_deg)))


def run_episode(seed: int, steps_limit: int, model: str, use_llm: bool) -> List[dict]:
    env = StructuredEnv(Reward_Params=REWARD_PARAMS, start_llm_at_step=START_LLM_AT_STEP)

    if use_llm:
        builder = BasePromptBuilder()
        ctl = LLMGuidanceController(
            builder=builder,
            k_history=5,
            save_dir=str(PLOTS_DIR / "memory"),
            start_llm_at_step=START_LLM_AT_STEP,
            memory_path=str(PLOTS_DIR / "eval_memory.json"),
            run_id=f"plots_seed{seed}"
        )
        ctl.reset()
    else:
        ctl = None

    np.random.seed(seed)
    env.reset(seed=seed)

    records: List[dict] = []
    obs, _ = env.reset(seed=seed)
    done = False
    while not done and env.n_step < steps_limit:
        if ctl is not None:
            turn_deg, _ = ctl.next_action(
                obs,
                step=env.n_step + 1,
                fileinfo=getattr(env.agent, "fileinfo", None),
                env=None,
                ppo=None,
            )
        else:
            # No LLM available: keep straight (0° turn) so we can still plot signals.
            turn_deg = 0

        action = deg_to_action(turn_deg)

        prev_obs = obs
        obs, reward, term, trunc, _ = env.step(action)

        snap = make_obs_snapshot(np.array(obs, dtype=float), np.array(prev_obs, dtype=float), step=env.n_step)
        records.append({
            "step": env.n_step,
            "dist_to_atco_path": float(obs[10]),
            "path_heading_deg": float(obs[12]),
            "heading_error_to_path_deg": float(obs[13]),
            "signed_cross_track_to_path": float(obs[14]),
            "own_heading_deg": math.degrees(float(obs[11])),
            "bearing_error_to_intruder": bearing_error_to_intruder(obs),
            "heading_error_to_dest_deg": heading_error_to_dest_deg(obs),
            "turn_deg": turn_deg,
        })

        done = term or trunc

    return records


def plot_lines(records: List[dict], fname: str, y_key: str, ylabel: str, zero_line: bool=False, stem_turns: bool=False):
    steps = [r["step"] for r in records]
    ys = [r[y_key] for r in records]
    plt.figure(figsize=(8,3))
    plt.plot(steps, ys, marker='o', label=ylabel)
    if zero_line:
        plt.axhline(0, color='k', linewidth=0.8, linestyle='--')
    if stem_turns:
        turns = [r["turn_deg"] for r in records]
        # Older matplotlib versions don't support use_line_collection.
        plt.stem(steps, turns, linefmt='C1-', markerfmt='C1o', basefmt='k-', label='turn_deg')
        plt.legend()
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    out = PLOTS_DIR / fname
    plt.savefig(out)
    plt.close()
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--model', type=str, default='llama3:8b')
    ap.add_argument('--no-llm', action='store_true', help='Skip LLM calls and fly straight (for offline plotting).')
    args = ap.parse_args()

    records = run_episode(seed=args.seed, steps_limit=args.steps, model=args.model, use_llm=not args.no_llm)

    # Save CSV
    csv_path = PLOTS_DIR / "signals.csv"
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"saved {csv_path}")

    # Plots
    plot_lines(records, "dist_to_path.png", "dist_to_atco_path", "dist_to_atco_path")
    plot_lines(records, "signed_xtrack.png", "signed_cross_track_to_path", "signed_cross_track_to_path", zero_line=True)
    plot_lines(records, "headings.png", "own_heading_deg", "own_heading_deg")
    # overlay path heading and error on the same fig
    steps = [r["step"] for r in records]
    plt.figure(figsize=(8,3))
    plt.plot(steps, [r["own_heading_deg"] for r in records], label='own_heading_deg')
    plt.plot(steps, [r["path_heading_deg"] for r in records], label='path_heading_deg')
    plt.plot(steps, [r["heading_error_to_path_deg"] for r in records], label='heading_error_to_path_deg')
    plt.axhline(0, color='k', linewidth=0.8, linestyle='--')
    plt.xlabel('step'); plt.ylabel('deg'); plt.legend(); plt.tight_layout()
    out = PLOTS_DIR / "heading_vs_path.png"
    plt.savefig(out); plt.close(); print(f"saved {out}")

    plot_lines(records, "bearing_vs_turn.png", "bearing_error_to_intruder", "bearing_error_to_intruder", zero_line=True, stem_turns=True)
    plot_lines(records, "heading_error_dest.png", "heading_error_to_dest_deg", "heading_error_to_dest_deg", zero_line=True)

    # Print a small preview
    print("first 5:")
    for r in records[:5]:
        print(r)
    print("last 5:")
    for r in records[-5:]:
        print(r)


if __name__ == "__main__":
    main()
