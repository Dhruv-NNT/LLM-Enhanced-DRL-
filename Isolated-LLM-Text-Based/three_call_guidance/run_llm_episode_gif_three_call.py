#!/usr/bin/env python
"""
Run a single LLM-guided three-call sector-entry episode and save a GIF.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from isolated_llm_core_three_call import (
    DEFAULT_CONFIG,
    LLMThreeCallEpisodeController,
    ThreeCallPromptBuilder,
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
    images = [np.array(Image.open(frame).convert("RGB")) for frame in frames]
    imageio.mimsave(str(out_gif_path), images, duration=1.0 / max(1e-6, fps))


def _annotate_latest_frame(frames_dir: Path, texts: list[str]) -> None:
    frames = sorted(frames_dir.glob("image_*.png"))
    if not frames or not texts:
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
        gap_y = 12
        for text in texts:
            bbox = draw.textbbox((left, top), text, font=font)
            box = (
                bbox[0] - pad_x,
                bbox[1] - pad_y,
                bbox[2] + pad_x,
                bbox[3] + pad_y,
            )
            draw.rounded_rectangle(box, radius=12, fill=(220, 30, 30, 210))
            draw.text((left, top), text, fill=(255, 255, 255, 255), font=font)
            top = box[3] + gap_y
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


def _print_step_log(step_index: int, ctl: LLMThreeCallEpisodeController, turn_deg: int) -> None:
    ttcp_text = "n/a" if ctl.last_predicted_ttcp_steps is None else f"{ctl.last_predicted_ttcp_steps:.2f}"
    loss_text = "n/a" if ctl.last_predicted_loss_step is None else str(int(ctl.last_predicted_loss_step))
    call_text = ctl.last_call_name or "-"
    print(
        "[THREE-CALL] "
        f"step={step_index:03d} "
        f"stage={ctl.stage} "
        f"inside_sector={int(ctl.inside_sector)} "
        f"entered_sector={int(ctl.has_entered_sector)} "
        f"ttcp={ttcp_text} "
        f"loss_step={loss_text} "
        f"conflict={int(ctl.last_conflict_within_lookahead)} "
        f"safe_streak={ctl.safe_streak_steps} "
        f"merge_back_recheck_remaining={ctl.merge_back_recheck_remaining} "
        f"call={call_text} "
        f"turn={turn_deg:+d} "
        f"merge_back_mode={ctl.last_merge_back_mode} "
        f"merge_back_count={ctl.merge_back_count}"
    )
    if ctl.last_event:
        print(f"[THREE-CALL] event={ctl.last_event}")
    if ctl.last_call_name in {"EXECUTE_TURN", "EMERGENCY_MANEUVER", "MERGE_BACK"}:
        print(
            "[THREE-CALL] "
            f"chosen_turn={turn_deg:+d} "
            f"merge_back_recheck_remaining={ctl.merge_back_recheck_remaining} "
            f"intruder_side={ctl.last_intruder_side} "
            f"allowed_turn_bins={ctl.last_allowed_turn_bins}"
        )


def _episode_outcome(env: object, *, term: bool, trunc: bool) -> tuple[bool, str]:
    if not hasattr(env, "agent"):
        return False, "unknown"

    own_position = np.array(env.agent.o_position, dtype=float)
    intr_position = np.array(env.agent.i_position, dtype=float)
    own_destination = np.array(env.agent.o_destination, dtype=float)
    intr_destination = np.array(env.agent.i_destination, dtype=float)

    own_at_goal = np.linalg.norm(own_position - own_destination) < 5.0
    intr_at_goal = np.linalg.norm(intr_position - intr_destination) < 5.0
    sep_dist = np.linalg.norm(own_position - intr_position)
    collision_risk = sep_dist < float(env.SAFE_R)
    success = bool(term and not trunc and own_at_goal and intr_at_goal and not collision_risk)

    if success:
        return True, "goal_reached"
    if trunc:
        return False, "truncated"
    if collision_risk:
        return False, "collision_risk"
    if term:
        return False, "terminated_not_success"
    return False, "manual_break"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_CONFIG.model)
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG.temperature)
    parser.add_argument("--top-p", type=float, default=DEFAULT_CONFIG.top_p)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_CONFIG.max_tokens)
    parser.add_argument("--start-llm-at-step", type=int, default=DEFAULT_CONFIG.default_start_llm_at_step)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    _inject_restructured_code_on_path(repo_root)

    from rl_llm.env import StructuredEnv  # type: ignore

    if args.seed is not None:
        np.random.seed(int(args.seed))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "outputs" / f"three_call_run_{stamp}"
    frames_dir = out_dir / "ep001_frames"
    _ensure_clean_dir(frames_dir)

    env = StructuredEnv(start_llm_at_step=int(args.start_llm_at_step))
    obs, _ = env.reset()

    builder = ThreeCallPromptBuilder(DEFAULT_CONFIG)
    ctl = LLMThreeCallEpisodeController(
        builder,
        config=DEFAULT_CONFIG,
        start_llm_at_step=int(args.start_llm_at_step),
        model=str(args.model),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
    )
    ctl.reset()

    ep_ret = 0.0
    done = False
    last_term = False
    last_trunc = False

    env.render(show=False, folder=str(frames_dir) + "/")

    while not done:
        step_index = int(env.n_step) + 1
        turn_deg = int(ctl.next_turn_deg(obs, step=step_index))
        action_idx = _deg_to_action_idx(turn_deg)

        if ctl.last_annotations:
            _annotate_latest_frame(frames_dir, ctl.last_annotations)

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
        last_term = bool(term)
        last_trunc = bool(trunc)

        env.render(show=False, folder=str(frames_dir) + "/")

        if env.n_step > env.MAX_STEP + 5:
            break

    success, outcome_label = _episode_outcome(env, term=last_term, trunc=last_trunc)
    committed_cases = ctl.finalize_episode_memory(
        run_id=out_dir.name,
        episode_id="ep001",
        success=success,
        outcome_label=outcome_label,
    )
    ctl.mark_finished()
    gif_path = out_dir / "ep001.gif"
    _save_gif(frames_dir, gif_path, fps=5.0)
    print(
        f"[THREE-CALL] Saved GIF -> {gif_path} | Return={ep_ret:.3f} "
        f"| outcome={outcome_label} | memory_cases_committed={committed_cases}"
    )


if __name__ == "__main__":
    main()
