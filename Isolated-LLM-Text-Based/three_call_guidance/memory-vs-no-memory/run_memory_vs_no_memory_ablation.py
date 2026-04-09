#!/usr/bin/env python3
"""
Run a paired memory-vs-no-memory ablation for the isolated three-call guidance.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=FutureWarning, module=r"rl_llm\.utils")
warnings.filterwarnings("ignore", message="More than 20 figures have been opened", category=RuntimeWarning)


THIS_DIR = Path(__file__).resolve().parent
THREE_CALL_DIR = THIS_DIR.parent
REPO_ROOT = THREE_CALL_DIR.parent.parent

if str(THREE_CALL_DIR) not in sys.path:
    sys.path.insert(0, str(THREE_CALL_DIR))

from isolated_llm_core_three_call import (  # noqa: E402
    DEFAULT_CONFIG,
    LLMThreeCallEpisodeController,
    ThreeCallPromptBuilder,
    _deg_to_action_idx,
)
from run_llm_episode_gif_three_call import (  # noqa: E402
    _annotate_latest_frame,
    _ensure_clean_dir,
    _episode_outcome,
    _inject_restructured_code_on_path,
    _save_gif,
)


STAGE_KEYS: tuple[str, ...] = ("EXECUTE_TURN", "EMERGENCY_MANEUVER", "MERGE_BACK")
STAGE_LABELS: Dict[str, str] = {
    "EXECUTE_TURN": "Execute_Turn",
    "EMERGENCY_MANEUVER": "Emergency_maneuver",
    "MERGE_BACK": "Merge_back",
}
SETTING_LABELS: Dict[str, str] = {
    "no_memory": "No Memory",
    "with_memory": "With Memory",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=20260409)
    parser.add_argument("--model", default=DEFAULT_CONFIG.model)
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG.temperature)
    parser.add_argument("--top-p", type=float, default=DEFAULT_CONFIG.top_p)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_CONFIG.max_tokens)
    parser.add_argument("--start-llm-at-step", type=int, default=DEFAULT_CONFIG.default_start_llm_at_step)
    return parser.parse_args()


def _episode_id(episode_index: int) -> str:
    return f"ep{episode_index:03d}"


def _setting_dirs(output_root: Path, setting: str, episode_index: int) -> tuple[Path, Path, Path]:
    episode_id = _episode_id(episode_index)
    episode_dir = output_root / setting / episode_id
    frames_dir = episode_dir / "frames"
    gif_path = episode_dir / f"{episode_id}.gif"
    return episode_dir, frames_dir, gif_path


def _serialize_scenario_fileinfo(fileinfo: Sequence[Any]) -> str:
    return json.dumps(list(fileinfo), ensure_ascii=True)


def _terminal_metrics(env: object) -> Dict[str, float]:
    own_position = np.array(env.agent.o_position, dtype=float)
    intr_position = np.array(env.agent.i_position, dtype=float)
    own_destination = np.array(env.agent.o_destination, dtype=float)
    intr_destination = np.array(env.agent.i_destination, dtype=float)
    return {
        "terminal_ownship_distance": float(np.linalg.norm(own_position - own_destination)),
        "terminal_intruder_distance": float(np.linalg.norm(intr_position - intr_destination)),
        "terminal_separation": float(np.linalg.norm(own_position - intr_position)),
    }


def _study_success(*, done: bool, trunc: bool, terminal_ownship_distance: float, terminal_separation: float, safe_r: float) -> bool:
    return bool(done and not trunc and terminal_ownship_distance < 5.0 and terminal_separation >= float(safe_r))


def _study_outcome_label(*, success: bool, term: bool, trunc: bool, terminal_separation: float, safe_r: float) -> str:
    if success:
        return "goal_reached"
    if trunc:
        return "truncated"
    if terminal_separation < float(safe_r):
        return "collision_risk"
    if term:
        return "terminated_not_success"
    return "manual_break"


def _write_csv(records: List[Dict[str, Any]], csv_path: Path) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _aggregate(records: List[Dict[str, Any]], episodes_per_setting: int) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for setting in ("no_memory", "with_memory"):
        subset = [record for record in records if record["setting"] == setting]
        successful_count = sum(1 for record in subset if bool(record["success"]))
        unsuccessful_count = len(subset) - successful_count
        call_totals = {
            stage_key: int(sum(int(record[f"{stage_key.lower()}_calls"]) for record in subset))
            for stage_key in STAGE_KEYS
        }
        call_averages = {
            stage_key: float(call_totals[stage_key]) / float(max(1, episodes_per_setting))
            for stage_key in STAGE_KEYS
        }
        summary[setting] = {
            "episodes": len(subset),
            "successful_count": successful_count,
            "unsuccessful_count": unsuccessful_count,
            "call_totals": call_totals,
            "call_averages": call_averages,
        }
    return summary


def _plot_success_counts(summary: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    success_color = "#2e7d32"
    failure_color = "#c62828"
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    for axis, setting in zip(axes, ("no_memory", "with_memory")):
        counts = [
            int(summary[setting]["successful_count"]),
            int(summary[setting]["unsuccessful_count"]),
        ]
        axis.bar(["Successful", "Unsuccessful"], counts, color=[success_color, failure_color], width=0.65)
        axis.set_title(SETTING_LABELS[setting])
        axis.set_ylabel("Episodes")
        axis.set_ylim(0, max(counts + [1]) * 1.15)
    fig.legend(
        handles=[
            Patch(color=success_color, label="Successful"),
            Patch(color=failure_color, label="Unsuccessful"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Successful vs Unsuccessful Episodes", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_stage_counts(summary: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    stage_order = list(STAGE_KEYS)
    x = np.arange(len(stage_order))
    width = 0.35

    no_memory_counts = [summary["no_memory"]["call_totals"][stage_key] for stage_key in stage_order]
    with_memory_counts = [summary["with_memory"]["call_totals"][stage_key] for stage_key in stage_order]

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, no_memory_counts, width=width, label=SETTING_LABELS["no_memory"], color="#1565c0")
    axis.bar(x + width / 2, with_memory_counts, width=width, label=SETTING_LABELS["with_memory"], color="#ef6c00")
    axis.set_xticks(x, [STAGE_LABELS[stage_key] for stage_key in stage_order])
    axis.set_ylabel("Total Calls")
    axis.set_title("Stage Call Totals by Setting")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_episode(
    *,
    setting: str,
    episode_index: int,
    seed: int,
    args: argparse.Namespace,
    output_root: Path,
    expected_scenario_fileinfo: Optional[Sequence[Any]],
    memory_store_path: Path,
) -> Dict[str, Any]:
    _, frames_dir, gif_path = _setting_dirs(output_root, setting, episode_index)
    frames_dir.parent.mkdir(parents=True, exist_ok=True)
    _ensure_clean_dir(frames_dir)

    use_memory = setting == "with_memory"
    config = replace(
        DEFAULT_CONFIG,
        use_memory=use_memory,
        memory_store_path=Path(memory_store_path),
    )
    builder = ThreeCallPromptBuilder(config)

    from rl_llm.env import StructuredEnv  # type: ignore

    env = StructuredEnv(start_llm_at_step=int(args.start_llm_at_step))
    np.random.seed(int(seed))
    obs, _ = env.reset(seed=int(seed))

    scenario_fileinfo = list(getattr(env.agent, "fileinfo", []))
    if expected_scenario_fileinfo is not None and list(expected_scenario_fileinfo) != scenario_fileinfo:
        raise RuntimeError(
            f"Scenario mismatch for {setting} episode {episode_index}: "
            f"expected {list(expected_scenario_fileinfo)!r}, got {scenario_fileinfo!r}"
        )

    ctl = LLMThreeCallEpisodeController(
        builder,
        config=config,
        start_llm_at_step=int(args.start_llm_at_step),
        model=str(args.model),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
    )
    ctl.reset()

    stage_counts = {stage_key: 0 for stage_key in STAGE_KEYS}
    last_term = False
    last_trunc = False
    manual_break = False
    memory_case_count_before_episode = len(ctl.memory_store.cases)

    env.render(show=False, folder=str(frames_dir) + "/")
    plt.close("all")

    done = False
    while not done:
        step_index = int(env.n_step) + 1
        turn_deg = int(ctl.next_turn_deg(obs, step=step_index))
        action_idx = _deg_to_action_idx(turn_deg)

        if ctl.last_annotations:
            _annotate_latest_frame(frames_dir, ctl.last_annotations)
        if ctl.last_call_name in stage_counts:
            stage_counts[str(ctl.last_call_name)] += 1

        obs, _, term, trunc, _ = env.step(action_idx)
        last_term = bool(term)
        last_trunc = bool(trunc)
        done = bool(term or trunc)

        env.render(show=False, folder=str(frames_dir) + "/")
        plt.close("all")

        if env.n_step > env.MAX_STEP + 5:
            manual_break = True
            done = True

    terminal = _terminal_metrics(env)
    success = _study_success(
        done=not manual_break,
        trunc=last_trunc,
        terminal_ownship_distance=terminal["terminal_ownship_distance"],
        terminal_separation=terminal["terminal_separation"],
        safe_r=float(env.SAFE_R),
    )
    outcome_label = _study_outcome_label(
        success=success,
        term=last_term,
        trunc=last_trunc,
        terminal_separation=terminal["terminal_separation"],
        safe_r=float(env.SAFE_R),
    )
    runtime_success, runtime_outcome_label = _episode_outcome(env, term=last_term, trunc=last_trunc)
    committed_cases = ctl.finalize_episode_memory(
        run_id=f"{output_root.name}_{setting}",
        episode_id=_episode_id(episode_index),
        success=runtime_success,
        outcome_label=runtime_outcome_label,
    )
    memory_case_count_after_episode = len(ctl.memory_store.cases)
    ctl.mark_finished()
    _save_gif(frames_dir, gif_path, fps=5.0)

    print(
        f"[ABLATION] setting={setting} episode={_episode_id(episode_index)} seed={seed} "
        f"scenario={scenario_fileinfo} success={int(success)} "
        f"calls=({stage_counts['EXECUTE_TURN']}, {stage_counts['EMERGENCY_MANEUVER']}, {stage_counts['MERGE_BACK']}) "
        f"memory_before={memory_case_count_before_episode} committed={committed_cases} "
        f"memory_after={memory_case_count_after_episode}"
    , flush=True)

    return {
        "setting": setting,
        "episode_index": episode_index,
        "seed": int(seed),
        "scenario_fileinfo": _serialize_scenario_fileinfo(scenario_fileinfo),
        "terminated": bool(last_term),
        "truncated": bool(last_trunc),
        "success": bool(success),
        "memory_commit_success": bool(runtime_success),
        "study_outcome_label": outcome_label,
        "memory_outcome_label": runtime_outcome_label,
        "terminal_ownship_distance": round(terminal["terminal_ownship_distance"], 6),
        "terminal_intruder_distance": round(terminal["terminal_intruder_distance"], 6),
        "terminal_separation": round(terminal["terminal_separation"], 6),
        "execute_turn_calls": int(stage_counts["EXECUTE_TURN"]),
        "emergency_maneuver_calls": int(stage_counts["EMERGENCY_MANEUVER"]),
        "merge_back_calls": int(stage_counts["MERGE_BACK"]),
        "memory_case_count_before_episode": int(memory_case_count_before_episode),
        "memory_cases_committed": int(committed_cases),
        "memory_case_count_after_episode": int(memory_case_count_after_episode),
        "gif_path": str(gif_path),
    }


def main() -> None:
    args = _parse_args()
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive.")

    _inject_restructured_code_on_path(REPO_ROOT)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = THIS_DIR / "outputs" / f"memory_vs_no_memory_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = [int(args.base_seed) + episode_offset for episode_offset in range(int(args.episodes))]
    expected_scenarios: Dict[int, List[Any]] = {}
    records: List[Dict[str, Any]] = []

    no_memory_store_path = output_root / "no_memory" / "memory" / "three_call_memory.jsonl"
    with_memory_store_path = output_root / "with_memory" / "memory" / "three_call_memory.jsonl"
    if with_memory_store_path.exists():
        with_memory_store_path.unlink()

    for setting, memory_store_path in (
        ("no_memory", no_memory_store_path),
        ("with_memory", with_memory_store_path),
    ):
        for episode_offset, seed in enumerate(seeds, start=1):
            expected = expected_scenarios.get(episode_offset)
            record = _run_episode(
                setting=setting,
                episode_index=episode_offset,
                seed=seed,
                args=args,
                output_root=output_root,
                expected_scenario_fileinfo=expected if setting == "with_memory" else None,
                memory_store_path=memory_store_path,
            )
            if setting == "no_memory":
                expected_scenarios[episode_offset] = json.loads(record["scenario_fileinfo"])
            records.append(record)

    summary = _aggregate(records, int(args.episodes))
    summary_payload = {
        "created_at_utc": stamp,
        "output_root": str(output_root),
        "episodes_per_setting": int(args.episodes),
        "base_seed": int(args.base_seed),
        "model": str(args.model),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_tokens),
        "start_llm_at_step": int(args.start_llm_at_step),
        "paired_scenarios_verified": True,
        "memory_store_path": str(with_memory_store_path),
        "settings": summary,
    }

    _write_csv(records, output_root / "episode_metrics.csv")
    (output_root / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _plot_success_counts(summary, output_root / "success_counts.png")
    _plot_stage_counts(summary, output_root / "stage_call_counts.png")

    print("\n[Ablation Summary]", flush=True)
    for setting in ("no_memory", "with_memory"):
        setting_summary = summary[setting]
        print(
            f"{setting}: successful={setting_summary['successful_count']} "
            f"unsuccessful={setting_summary['unsuccessful_count']}"
        , flush=True)
        for stage_key in STAGE_KEYS:
            print(
                f"{setting} {STAGE_LABELS[stage_key]} average="
                f"{setting_summary['call_averages'][stage_key]:.3f}"
            , flush=True)
    print(f"Artifacts saved under: {output_root}", flush=True)


if __name__ == "__main__":
    main()
