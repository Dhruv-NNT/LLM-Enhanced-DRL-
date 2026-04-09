#!/usr/bin/env python3
"""
Generate rollout GIFs or diagnose non-terminating scenarios without touching the
core LLM-DRL environment. The new --diagnose mode scans the dataset, watches the
distance-to-goal deltas, and emits JSON artifacts whenever the drift-away or
max-steps rules should have fired but did not.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch

import rl_llm.env as env_module
from configs import EVAL_BEST_MODEL_PATH, FEATUREFILE_PATH, START_LLM_AT_STEP
from rl_llm.env import StructuredEnv
from rl_llm.ppo import PPO, warmup_obs_norm, device
from rl_llm.utils import getfilenames, project_path

FAILURE_DIR = Path(__file__).resolve().parent / "identifying failure cases"
DIAG_DIR = FAILURE_DIR / "diagnostics"
SUMMARY_CSV = FAILURE_DIR / "failure_case_summary.csv"
NON_TERMINATION_INDEX = FAILURE_DIR / "non_termination_index.csv"

DRIFT_EPS = 0.1
MAX_STEP = StructuredEnv.MAX_STEP


@dataclass
class StepTelemetry:
    step_idx: int
    ownship_pos: List[float]
    goal_pos: List[float]
    dist_to_goal: float
    delta_dist_to_goal: Optional[float]
    action: Dict[str, float]
    terminated: bool
    truncated: bool
    reasons: List[str]


@dataclass
class ViolationInfo:
    step: int
    dist: Optional[float]
    delta: Optional[float]


@dataclass
class ScenarioResult:
    row_index: int
    scenario_name: str
    gif_path: Optional[Path]
    n_steps: int
    total_reward: float
    reasons: Sequence[str]
    drift_violation: Optional[ViolationInfo]
    maxstep_violation: Optional[ViolationInfo]


@dataclass
class BugReport:
    scenario_name: str
    bug_type: str
    violation: ViolationInfo
    json_path: Path
    highlight_gif: Optional[Path]


def slugify(text: str, max_len: int = 64) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return (cleaned or "scenario")[:max_len]


def ensure_clean_frames(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for png in dir_path.glob("image_*.png"):
        try:
            png.unlink()
        except OSError:
            pass


def save_gif_from_paths(frame_paths: List[Path], out_path: Path, fps: float) -> None:
    if not frame_paths:
        raise RuntimeError(f"No frames available to save GIF at {out_path}")
    images = [imageio.imread(str(frame)) for frame in frame_paths]
    imageio.mimsave(str(out_path), images, duration=1.0 / max(1e-6, fps))


def save_full_run_gif(frames_dir: Path, out_path: Path, fps: float) -> None:
    frames = sorted(frames_dir.glob("image_*.png"))
    save_gif_from_paths(frames, out_path, fps)


def classify_termination(env: StructuredEnv, terminated: bool, truncated: bool) -> List[str]:
    reasons: List[str] = []
    agent = env.agent

    own = np.array(agent.o_position, dtype=float)
    intr = np.array(agent.i_position, dtype=float)
    own_dest = np.array(agent.o_destination, dtype=float)
    intr_dest = np.array(agent.i_destination, dtype=float)

    dist_own = float(np.linalg.norm(own - own_dest))
    dist_intr = float(np.linalg.norm(intr - intr_dest))
    sep = float(np.linalg.norm(own - intr))

    if dist_own < 5 and dist_intr < 5:
        reasons.append("arrived")

    if sep < env.SAFE_R:
        reasons.append("loss_of_separation")

    if len(agent.o_newpathlist) > 1:
        prev_loc = np.array(agent.o_newpathlist[-2], dtype=float)
        prev_dist = float(np.linalg.norm(own_dest - prev_loc))
        if prev_dist - dist_own < -DRIFT_EPS:
            reasons.append("drift_away")

    if truncated and env.n_step >= env.MAX_STEP:
        reasons.append("max_steps")

    if not reasons:
        if terminated:
            reasons.append("terminated_unspecified")
        elif truncated:
            reasons.append("truncated_unspecified")

    return reasons


def infer_expected_reasons(
    dist_own: float,
    dist_intr: float,
    sep: float,
    delta: Optional[float],
    n_step: int,
    truncated: bool,
) -> List[str]:
    reasons: List[str] = []
    if dist_own < 5 and dist_intr < 5:
        reasons.append("arrived")
    if sep < StructuredEnv.SAFE_R:
        reasons.append("loss_of_separation")
    if delta is not None and delta > DRIFT_EPS:
        reasons.append("drift_away")
    if truncated and n_step >= StructuredEnv.MAX_STEP:
        reasons.append("max_steps")
    return reasons


def build_agent() -> PPO:
    if not os.path.exists(EVAL_BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Trained weights not found at {EVAL_BEST_MODEL_PATH}. "
            "Please update configs.py or train the policy first."
        )

    dummy_env = StructuredEnv(start_llm_at_step=START_LLM_AT_STEP)
    obs_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n

    agent = PPO(
        obs_dim,
        action_dim,
        lr_actor=2e-4,
        lr_critic=8e-4,
        gamma=0.99,
        K_epochs=5,
        eps_clip=0.2,
        mb_size=256,
        gae_lambda=0.95,
        normalize_reward=True,
    )
    agent.load(EVAL_BEST_MODEL_PATH)
    warmup_obs_norm(agent, dummy_env, steps=5000)
    dummy_env.close()
    return agent


def select_action(agent: PPO, obs: np.ndarray) -> int:
    tensor_obs = torch.tensor(obs, dtype=torch.float32, device=device)
    norm_obs = agent._normalize_obs_eval(tensor_obs)
    with torch.no_grad():
        feats = agent.policy_old.net(norm_obs)
        logits = agent.policy_old.actor(feats)
        action = torch.argmax(logits, dim=-1).item()
    return int(action)


def scenario_dataframe_row(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    row = df.iloc[idx: idx + 1].copy()
    row.reset_index(drop=True, inplace=True)
    return row


def scenario_has_required_files(row: pd.Series) -> bool:
    resolved_dir = project_path("allresolvedtrajectories")
    unresolved_dir = project_path("allflighttrajectories")
    resflight, unresflight = getfilenames(row["filenames"])
    needed_paths = [
        resolved_dir / resflight,
        unresolved_dir / resflight,
        unresolved_dir / unresflight,
    ]
    return all(path.exists() for path in needed_paths)


def iterate_valid_indices(df: pd.DataFrame) -> List[int]:
    valid: List[int] = []
    for idx, row in df.iterrows():
        if scenario_has_required_files(row):
            valid.append(idx)
    return valid


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.SubprocessError:
        return "unknown"


def extract_trace_sample(
    telemetry: List[StepTelemetry],
    center_step: int,
) -> List[Dict[str, object]]:
    if not telemetry:
        return []
    lookup = {t.step_idx: t for t in telemetry}
    selected_steps = set()
    for offset in range(-2, 3):
        selected_steps.add(center_step + offset)
    for entry in telemetry[-2:]:
        selected_steps.add(entry.step_idx)
    sample: List[Dict[str, object]] = []
    for step in sorted(selected_steps):
        if step not in lookup:
            continue
        t = lookup[step]
        sample.append(
            {
                "step": t.step_idx,
                "ownship_pos": t.ownship_pos,
                "goal_pos": t.goal_pos,
                "dist": t.dist_to_goal,
                "delta": t.delta_dist_to_goal,
                "action": t.action,
                "terminated": t.terminated,
                "truncated": t.truncated,
                "reasons": t.reasons,
            }
        )
    return sample


def append_to_index_csv(row: Dict[str, object]) -> None:
    header_needed = not NON_TERMINATION_INDEX.exists()
    with NON_TERMINATION_INDEX.open("a", encoding="utf-8") as fh:
        if header_needed:
            fh.write(
                "scenario_id,bug_type,first_violation_step,delta_at_violation,"
                "total_steps,termination_reason_observed,reward_total,json_path\n"
            )
        fh.write(
            f"{row['scenario_id']},{row['bug_type']},"
            f"{row['first_violation_step']},{row['delta_at_violation']},"
            f"{row['total_steps']},{row['termination_reason_observed']},"
            f"{row['reward_total']},{row['json_path']}\n"
        )


def create_highlight_gif(
    frames_dir: Path,
    violation_step: int,
    out_path: Path,
    fps: float,
) -> Optional[Path]:
    if not frames_dir.exists():
        return None
    start = max(0, violation_step - 5)
    end = violation_step + 5
    frame_paths: List[Path] = []
    for step in range(start, end + 1):
        frame = frames_dir / f"image_{step:03d}.png"
        if frame.exists():
            frame_paths.append(frame)
    if not frame_paths:
        return None
    save_gif_from_paths(frame_paths, out_path, fps)
    return out_path


def write_bug_json(
    scenario_id: str,
    bug_type: str,
    env_commit: str,
    seed: int,
    env: StructuredEnv,
    violation: ViolationInfo,
    telemetry: List[StepTelemetry],
    reasons: Sequence[str],
) -> Path:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    safe_bug = bug_type.replace(":", "_")
    json_path = DIAG_DIR / f"{slugify(scenario_id)}_{safe_bug}.json"
    agent = env.agent
    final_dist = float(
        np.linalg.norm(
            np.array(agent.o_position, dtype=float)
            - np.array(agent.o_destination, dtype=float)
        )
    )
    payload = {
        "scenario_id": scenario_id,
        "bug_type": bug_type,
        "env_commit": env_commit,
        "ownship_entry_coordinates": list(agent.o_start),
        "seed": seed,
        "thresholds": {"drift_eps": DRIFT_EPS, "max_steps": MAX_STEP},
        "summary": {
            "first_violation_step": violation.step,
            "dist_at_violation": violation.dist,
            "delta_at_violation": violation.delta,
            "total_steps": env.n_step,
            "final_dist": final_dist,
            "termination_reason_observed": "|".join(reasons),
            "reward_total": float(env.total_reward),
        },
        "trace_sample": extract_trace_sample(telemetry, violation.step),
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return json_path


def run_episode_for_row(
    agent: PPO,
    df: pd.DataFrame,
    row_index: int,
    scenario_rank: int,
    total_scenarios: int,
    *,
    record_full_gif: bool,
    preserve_frames: bool,
    fps: float,
    diagnose: bool,
    keep_highlight_frames: bool,
    base_seed: int,
    env_commit: str,
) -> Tuple[ScenarioResult, List[BugReport]]:
    scenario_name = df.loc[row_index, "filenames"]
    slug = slugify(scenario_name)
    scenario_dir = FAILURE_DIR / f"{scenario_rank:04d}_{slug}"
    frames_dir = scenario_dir / "frames"
    gif_path = scenario_dir / f"{slug}.gif"
    capture_frames = record_full_gif or (diagnose and keep_highlight_frames)
    if capture_frames:
        scenario_dir.mkdir(parents=True, exist_ok=True)
        ensure_clean_frames(frames_dir)

    env_module.featurefile = scenario_dataframe_row(df, row_index)
    scenario_seed = base_seed + row_index
    set_global_seed(scenario_seed)
    env = StructuredEnv(start_llm_at_step=START_LLM_AT_STEP)

    print(
        f"[RUN] Scenario {scenario_rank}/{total_scenarios} "
        f"(row {row_index}) → {scenario_name}"
    )

    obs, _ = env.reset()
    if capture_frames:
        env.render(show=False, folder=str(frames_dir) + "/")

    own_goal = np.array(env.agent.o_destination, dtype=float)
    prev_dist = float(
        np.linalg.norm(np.array(env.agent.o_position, dtype=float) - own_goal)
    )
    telemetry: List[StepTelemetry] = []
    drift_violation: Optional[ViolationInfo] = None
    maxstep_violation: Optional[ViolationInfo] = None

    done = False
    terminated = False
    truncated = False

    while not done:
        action = select_action(agent, obs)
        obs, _, terminated, truncated, _ = env.step(action)

        own_pos = np.array(env.agent.o_position, dtype=float)
        intr_pos = np.array(env.agent.i_position, dtype=float)
        intr_goal = np.array(env.agent.i_destination, dtype=float)

        current_dist = float(np.linalg.norm(own_pos - own_goal))
        delta = None if prev_dist is None else current_dist - prev_dist
        sep = float(np.linalg.norm(own_pos - intr_pos))
        dist_intr = float(np.linalg.norm(intr_pos - intr_goal))
        expected_reasons = infer_expected_reasons(
            dist_own=current_dist,
            dist_intr=dist_intr,
            sep=sep,
            delta=delta,
            n_step=env.n_step,
            truncated=truncated,
        )

        if diagnose:
            telemetry.append(
                StepTelemetry(
                    step_idx=env.n_step,
                    ownship_pos=list(map(float, own_pos)),
                    goal_pos=list(map(float, own_goal)),
                    dist_to_goal=current_dist,
                    delta_dist_to_goal=None if delta is None else float(delta),
                    action={
                        "index": float(action),
                        "ownship_heading_deg": float(
                            math.degrees(env.agent.o_heading)
                        ),
                    },
                    terminated=terminated,
                    truncated=truncated,
                    reasons=expected_reasons,
                )
            )

        if delta is not None and delta > DRIFT_EPS and not terminated:
            if drift_violation is None:
                drift_violation = ViolationInfo(
                    step=env.n_step,
                    dist=current_dist,
                    delta=float(delta),
                )

        if env.n_step >= env.MAX_STEP and not truncated and not terminated:
            if maxstep_violation is None:
                maxstep_violation = ViolationInfo(
                    step=env.n_step,
                    dist=current_dist,
                    delta=None,
                )

        if capture_frames:
            env.render(show=False, folder=str(frames_dir) + "/")

        done = terminated or truncated
        prev_dist = current_dist

        if env.n_step > env.MAX_STEP + 10:
            print("[WARN] Safety break hit; stopping rollout early.")
            break

    reasons = classify_termination(env, terminated, truncated)

    generated_gif: Optional[Path] = None
    if record_full_gif and capture_frames:
        save_full_run_gif(frames_dir, gif_path, fps=fps)
        generated_gif = gif_path
        if not preserve_frames:
            for png in frames_dir.glob("image_*.png"):
                try:
                    png.unlink()
                except OSError:
                    pass
            try:
                frames_dir.rmdir()
            except OSError:
                pass

    bug_reports: List[BugReport] = []
    reasons_set = set(reasons)
    if diagnose and drift_violation and "drift_away" not in reasons_set:
        json_path = write_bug_json(
            scenario_id=scenario_name,
            bug_type="BUG:DRIFT_NOT_FIRED",
            env_commit=env_commit,
            seed=scenario_seed,
            env=env,
            violation=drift_violation,
            telemetry=telemetry,
            reasons=reasons,
        )
        highlight_path = None
        if keep_highlight_frames and capture_frames:
            highlight_path = create_highlight_gif(
                frames_dir,
                drift_violation.step,
                scenario_dir / f"{slug}_drift_bug.gif",
                fps,
            )
        bug_reports.append(
            BugReport(
                scenario_name=scenario_name,
                bug_type="BUG:DRIFT_NOT_FIRED",
                violation=drift_violation,
                json_path=json_path,
                highlight_gif=highlight_path,
            )
        )
        append_to_index_csv(
            {
                "scenario_id": scenario_name,
                "bug_type": "BUG:DRIFT_NOT_FIRED",
                "first_violation_step": drift_violation.step,
                "delta_at_violation": drift_violation.delta,
                "total_steps": env.n_step,
                "termination_reason_observed": "|".join(reasons),
                "reward_total": float(env.total_reward),
                "json_path": json_path,
            }
        )

    if diagnose and maxstep_violation and "max_steps" not in reasons_set:
        json_path = write_bug_json(
            scenario_id=scenario_name,
            bug_type="BUG:MAX_STEPS_NOT_FIRED",
            env_commit=env_commit,
            seed=scenario_seed,
            env=env,
            violation=maxstep_violation,
            telemetry=telemetry,
            reasons=reasons,
        )
        highlight_path = None
        if keep_highlight_frames and capture_frames:
            highlight_path = create_highlight_gif(
                frames_dir,
                maxstep_violation.step,
                scenario_dir / f"{slug}_maxsteps_bug.gif",
                fps,
            )
        bug_reports.append(
            BugReport(
                scenario_name=scenario_name,
                bug_type="BUG:MAX_STEPS_NOT_FIRED",
                violation=maxstep_violation,
                json_path=json_path,
                highlight_gif=highlight_path,
            )
        )
        append_to_index_csv(
            {
                "scenario_id": scenario_name,
                "bug_type": "BUG:MAX_STEPS_NOT_FIRED",
                "first_violation_step": maxstep_violation.step,
                "delta_at_violation": maxstep_violation.delta,
                "total_steps": env.n_step,
                "termination_reason_observed": "|".join(reasons),
                "reward_total": float(env.total_reward),
                "json_path": json_path,
            }
        )

    if diagnose and capture_frames and not record_full_gif:
        # Clean up any step-by-step PNGs created for diagnostics only.
        for png in frames_dir.glob("image_*.png"):
            try:
                png.unlink()
            except OSError:
                pass
        try:
            frames_dir.rmdir()
        except OSError:
            pass
        # Remove the scenario directory as well if it is now empty and no bug artifacts were produced.
        if not bug_reports and scenario_dir.exists():
            try:
                scenario_dir.rmdir()
            except OSError:
                pass

    print(
        f"[DONE] {scenario_name} → steps={env.n_step}, reward={env.total_reward:.2f}, "
        f"reason={','.join(reasons) or 'none'}"
        f"{' | GIF=' + str(generated_gif) if generated_gif else ''}"
    )

    env.close()

    scenario_result = ScenarioResult(
        row_index=row_index,
        scenario_name=scenario_name,
        gif_path=generated_gif,
        n_steps=env.n_step,
        total_reward=float(env.total_reward),
        reasons=reasons,
        drift_violation=drift_violation,
        maxstep_violation=maxstep_violation,
    )

    return scenario_result, bug_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate GIFs for every valid scenario or diagnose missing termination cases."
        )
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Optional cap on how many scenarios to process.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip dataset rows before this index.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Playback rate (frames per second) for the output GIFs.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep per-step PNGs after writing the full GIF (non-diagnose mode).",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Enable diagnostics mode (no full GIFs unless explicitly requested).",
    )
    parser.add_argument(
        "--keep-gifs",
        action="store_true",
        help="When diagnosing, store highlight GIFs around any detected violation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed for deterministic rollouts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    NON_TERMINATION_INDEX.parent.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(FEATUREFILE_PATH)
    valid_indices = iterate_valid_indices(full_df)
    valid_indices = [idx for idx in valid_indices if idx >= args.start_index]
    if args.max_scenarios is not None:
        valid_indices = valid_indices[: args.max_scenarios]

    if not valid_indices:
        raise RuntimeError(
            "No scenarios satisfied the file-availability constraints. "
            "Verify that the CSV paths in configs.py are correct."
        )

    print(
        f"[INFO] Found {len(valid_indices)} valid scenarios between indices "
        f"{valid_indices[0]} and {valid_indices[-1]}."
    )

    agent = build_agent()
    env_commit = get_git_commit()

    results: List[ScenarioResult] = []
    total = len(valid_indices)
    drift_bug_count = 0
    max_bug_count = 0

    for rank, row_index in enumerate(valid_indices, start=1):
        try:
            res, bugs = run_episode_for_row(
                agent=agent,
                df=full_df,
                row_index=row_index,
                scenario_rank=rank,
                total_scenarios=total,
                record_full_gif=not args.diagnose,
                preserve_frames=args.keep_frames,
                fps=args.fps,
                diagnose=args.diagnose,
                keep_highlight_frames=args.keep_gifs,
                base_seed=args.seed,
                env_commit=env_commit,
            )
            results.append(res)
            for bug in bugs:
                if bug.bug_type == "BUG:DRIFT_NOT_FIRED":
                    drift_bug_count += 1
                elif bug.bug_type == "BUG:MAX_STEPS_NOT_FIRED":
                    max_bug_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Scenario row {row_index} failed: {exc}")

    if results:
        summary_df = pd.DataFrame(
            {
                "row_index": [r.row_index for r in results],
                "scenario_name": [r.scenario_name for r in results],
                "gif_path": [
                    str(r.gif_path) if r.gif_path is not None else "" for r in results
                ],
                "n_steps": [r.n_steps for r in results],
                "total_reward": [r.total_reward for r in results],
                "termination_reasons": [",".join(r.reasons) for r in results],
            }
        )
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print(f"[INFO] Wrote summary CSV → {SUMMARY_CSV}")
    else:
        print("[WARN] No scenarios completed.")

    total_bugs = drift_bug_count + max_bug_count
    print(
        "[SUMMARY] Scanned "
        f"{len(results)} scenario(s); flagged {total_bugs} bug(s)\n"
        f"  - BUG:DRIFT_NOT_FIRED: {drift_bug_count} cases\n"
        f"  - BUG:MAX_STEPS_NOT_FIRED: {max_bug_count} cases\n"
        "Artifacts: identifying failure cases/diagnostics/ and "
        "identifying failure cases/non_termination_index.csv"
    )


if __name__ == "__main__":
    main()
