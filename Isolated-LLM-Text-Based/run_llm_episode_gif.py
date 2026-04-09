#!/usr/bin/env python
"""
Run a single **LLM-only** episode in StructuredEnv and save a GIF for visual inspection.

This script imports the environment from `LLM-DRL restructured code/` (read-only).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from isolated_llm_core import BasePromptBuilder, LLMOnlyEpisodeController, _deg_to_action_idx


def _ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for png in path.glob("image_*.png"):
        try:
            png.unlink()
        except OSError:
            pass


def _save_gif(frames_dir: Path, out_gif_path: Path, fps: float = 5.0) -> None:
    import imageio.v2 as imageio

    frames = sorted(frames_dir.glob("image_*.png"))
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}. Did env.render() run?")
    images = [imageio.imread(str(f)) for f in frames]
    imageio.mimsave(str(out_gif_path), images, duration=1.0 / max(1e-6, fps))


def _inject_restructured_code_on_path(repo_root: Path) -> Path:
    if (repo_root / "rl_llm").exists():
        restructured_root = repo_root
    else:
        restructured_root = repo_root / "LLM-DRL restructured code"
    if not restructured_root.exists():
        raise FileNotFoundError(f"Not found: {restructured_root}")
    sys.path.insert(0, str(restructured_root))
    return restructured_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3:8b")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--start-llm-at-step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    _inject_restructured_code_on_path(repo_root)

    # Import after sys.path injection so "configs.py" resolves inside the restructured folder.
    from rl_llm.env import StructuredEnv  # type: ignore

    if args.seed is not None:
        np.random.seed(int(args.seed))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "outputs" / f"run_{stamp}"
    frames_dir = out_dir / "ep001_frames"
    _ensure_clean_dir(frames_dir)

    env = StructuredEnv(start_llm_at_step=int(args.start_llm_at_step))
    obs, _ = env.reset()

    builder = BasePromptBuilder()
    ctl = LLMOnlyEpisodeController(
        builder,
        start_llm_at_step=int(args.start_llm_at_step),
        model=str(args.model),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
    )
    ctl.reset(initial_phase="WAIT_TURN")

    ep_ret = 0.0
    done = False

    # Render initial frame
    env.render(show=False, folder=str(frames_dir) + "/")

    while not done:
        step_index = int(env.n_step) + 1
        turn_deg = int(ctl.next_turn_deg(obs, step=step_index))
        action_idx = _deg_to_action_idx(turn_deg)

        if ctl.did_call_llm and ctl.last_tag and ctl.last_prompt_text is not None and ctl.last_raw_text is not None:
            # Save prompt/response for later refinement.
            (out_dir / f"{ctl.last_tag}.prompt.txt").write_text(ctl.last_prompt_text, encoding="utf-8")
            (out_dir / f"{ctl.last_tag}.response.txt").write_text(ctl.last_raw_text, encoding="utf-8")
            if ctl.last_normalized is not None:
                (out_dir / f"{ctl.last_tag}.normalized.json").write_text(
                    __import__("json").dumps(ctl.last_normalized, indent=2),
                    encoding="utf-8",
                )

        obs, reward, term, trunc, _ = env.step(action_idx)
        ep_ret += float(reward)
        done = bool(term or trunc)

        env.render(show=False, folder=str(frames_dir) + "/")

        # Safety guard
        if env.n_step > env.MAX_STEP + 5:
            break

    gif_path = out_dir / "ep001.gif"
    _save_gif(frames_dir, gif_path, fps=5.0)
    print(f"[ISO-LLM] Saved GIF → {gif_path} | Return={ep_ret:.3f}")


if __name__ == "__main__":
    main()
