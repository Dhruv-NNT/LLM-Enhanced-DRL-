#!/usr/bin/env python
"""
Run a single LLM-guided sector-entry episode variant and save a GIF.

This runner is a sibling of `run_llm_episode_gif.py` and keeps the original
isolated workflow untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from isolated_llm_core_sector_entry import (
    LLMSectorEntryEpisodeController,
    SectorEntryPromptBuilder,
    _deg_to_action_idx,
)


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
    images = [np.array(Image.open(f).convert("RGB")) for f in frames]
    imageio.mimsave(str(out_gif_path), images, duration=1.0 / max(1e-6, fps))


def _annotate_latest_frame(frames_dir: Path, text: str) -> None:
    frames = sorted(frames_dir.glob("image_*.png"))
    if not frames:
        return

    frame_path = frames[-1]
    with Image.open(frame_path).convert("RGBA") as img:
        draw = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except OSError:
            font = ImageFont.load_default()

        left = 24
        top = 24
        pad_x = 16
        pad_y = 10
        bbox = draw.textbbox((left, top), text, font=font)
        box = (
            bbox[0] - pad_x,
            bbox[1] - pad_y,
            bbox[2] + pad_x,
            bbox[3] + pad_y,
        )
        draw.rounded_rectangle(box, radius=12, fill=(220, 30, 30, 210))
        draw.text((left, top), text, fill=(255, 255, 255, 255), font=font)
        img.convert("RGB").save(frame_path)


def _inject_restructured_code_on_path(repo_root: Path) -> Path:
    if (repo_root / "rl_llm").exists():
        restructured_root = repo_root
    else:
        restructured_root = repo_root / "LLM-DRL restructured code"
    if not restructured_root.exists():
        raise FileNotFoundError(f"Not found: {restructured_root}")
    sys.path.insert(0, str(restructured_root))
    return restructured_root


def _print_step_log(step_index: int, ctl: LLMSectorEntryEpisodeController, turn_deg: int) -> None:
    ttcp_text = "n/a" if ctl.last_predicted_ttcp_steps is None else f"{ctl.last_predicted_ttcp_steps:.2f}"
    loss_text = "n/a" if ctl.last_predicted_loss_step is None else str(int(ctl.last_predicted_loss_step))
    print(
        "[SECTOR-ENTRY] "
        f"step={step_index:03d} "
        f"stage={ctl.stage} "
        f"inside_sector={int(ctl.inside_sector)} "
        f"ttcp={ttcp_text} "
        f"loss_step={loss_text} "
        f"trigger={int(ctl.last_safe_loss_trigger)} "
        f"safe_streak={ctl.safe_streak_steps} "
        f"turn={turn_deg:+d}"
    )
    if ctl.last_event:
        print(f"[SECTOR-ENTRY] event={ctl.last_event}")
    if ctl.recovery_start_step is not None:
        print(f"[SECTOR-ENTRY] recovery_start_step={ctl.recovery_start_step}")


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

    from rl_llm.env import StructuredEnv  # type: ignore

    if args.seed is not None:
        np.random.seed(int(args.seed))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "outputs" / f"sector_entry_run_{stamp}"
    frames_dir = out_dir / "ep001_frames"
    _ensure_clean_dir(frames_dir)

    env = StructuredEnv(start_llm_at_step=int(args.start_llm_at_step))
    obs, _ = env.reset()

    builder = SectorEntryPromptBuilder()
    ctl = LLMSectorEntryEpisodeController(
        builder,
        start_llm_at_step=int(args.start_llm_at_step),
        model=str(args.model),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
    )
    ctl.reset()

    ep_ret = 0.0
    done = False

    env.render(show=False, folder=str(frames_dir) + "/")

    while not done:
        step_index = int(env.n_step) + 1
        turn_deg = int(ctl.next_turn_deg(obs, step=step_index))
        action_idx = _deg_to_action_idx(turn_deg)

        if ctl.last_event and ctl.last_event.startswith("entered_sector@"):
            _annotate_latest_frame(frames_dir, "ENTERED SECTOR")

        _print_step_log(step_index, ctl, turn_deg)

        if ctl.did_call_llm and ctl.last_tag and ctl.last_prompt_text is not None and ctl.last_raw_text is not None:
            (out_dir / f"{ctl.last_tag}.prompt.txt").write_text(ctl.last_prompt_text, encoding="utf-8")
            (out_dir / f"{ctl.last_tag}.response.txt").write_text(ctl.last_raw_text, encoding="utf-8")
            if ctl.last_normalized is not None:
                (out_dir / f"{ctl.last_tag}.normalized.json").write_text(
                    json.dumps(ctl.last_normalized, indent=2),
                    encoding="utf-8",
                )

        obs, reward, term, trunc, _ = env.step(action_idx)
        ep_ret += float(reward)
        done = bool(term or trunc)

        env.render(show=False, folder=str(frames_dir) + "/")

        if env.n_step > env.MAX_STEP + 5:
            break

    ctl.mark_finished()
    gif_path = out_dir / "ep001.gif"
    _save_gif(frames_dir, gif_path, fps=5.0)
    print(f"[SECTOR-ENTRY] Saved GIF -> {gif_path} | Return={ep_ret:.3f}")


if __name__ == "__main__":
    main()
