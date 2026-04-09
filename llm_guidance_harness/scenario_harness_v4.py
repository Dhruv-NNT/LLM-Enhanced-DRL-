#!/usr/bin/env python3
"""
Standalone harness to sanity-check LLM guidance for a few deterministic scenarios.

Key capabilities
----------------
1. Simulation harness: run labeled scenarios (A–D by default) inside `StructuredEnv`
   and record observations, actions, rewards, and prompts step-by-step.
2. Direct prompt capture: every step dumps the exact prompt text that would be sent
   to the LLM so it can be inspected without touching the main pipeline.
3. Replay tests: the recorded prompts can be replayed later (`--replay-file`) to
   compare how the current LLM outputs differ from the earlier log.
4. Glue-code unit tests: `--run-tests` executes quick checks around prompt/action
   conversions so regressions in the adapter layer are caught early.
5. Manual inspection: `--inspect-log` prints a concise timeline so we can trace
   whether the controller honoured the LLM (or heuristic) instructions.

Usage examples
--------------
Run all four canned scenarios and keep prompts only (default dry run):
    python scenario_harness.py

Run the simulator but actually call the local LLM (requires Ollama running):
    python scenario_harness.py --call-llm

Replay a previous log and compare fresh LLM outputs:
    python scenario_harness.py --call-llm --replay-file logs/run_.../A/trace.jsonl

Print a quick textual trace from a run:
    python scenario_harness.py --inspect-log logs/run_.../B/trace.jsonl

Execute glue-code unit tests only:
    python scenario_harness.py --run-tests
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import unittest
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Ensure imports work when executing from this folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs import REWARD_PARAMS, START_LLM_AT_STEP
from rl_llm.env import StructuredEnv
from rl_llm.llm import (
    ALLOWED_BINS,
    BasePromptBuilder,
    ObsSnapshot,
    attach_phase_state_block,
    ollama_invoke,
)

# Reusable regex to pull the first JSON block out of an LLM answer.
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Env action bins start at 0° on index 0, then +5…+30, then -5…-30.
ACTION_TO_DEG = [0, 5, 10, 15, 20, 25, 30, -5, -10, -15, -20, -25, -30]


@dataclass
class ScenarioDefinition:
    label: str
    seed: int
    steps: int
    description: str


DEFAULT_SCENARIOS: Dict[str, ScenarioDefinition] = {
    "A": ScenarioDefinition(
        label="A",
        seed=101,
        steps=18,
        description="Head-on closing encounter with both at cruise speed.",
    ),
    "B": ScenarioDefinition(
        label="B",
        seed=202,
        steps=22,
        description="Intruder crosses from right-to-left with mild offset.",
    ),
    "C": ScenarioDefinition(
        label="C",
        seed=303,
        steps=20,
        description="Ownship starts late and must chase intruder already turning.",
    ),
    "D": ScenarioDefinition(
        label="D",
        seed=404,
        steps=25,
        description="Diverging track but dangerously low separation at spawn.",
    ),
}


@dataclass
class GuidanceDecision:
    heading_deg: int
    hold_steps: int
    phase: str
    source: str
    issued_step: int
    raw_text: Optional[str] = None
    parsed: Optional[dict] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        data = {
            "heading_deg": int(self.heading_deg),
            "hold_steps": int(self.hold_steps),
            "phase": self.phase,
            "source": self.source,
            "issued_step": int(self.issued_step),
        }
        if self.note:
            data["note"] = self.note
        if self.raw_text is not None:
            data["raw_text"] = self.raw_text
        if self.parsed is not None:
            data["parsed"] = self.parsed
        return data


class ScenarioLogger:
    """Helper that creates per-scenario folders and writes prompts/traces."""

    def __init__(self, run_dir: Path, scenario: ScenarioDefinition):
        self.scenario = scenario
        self.dir = run_dir / scenario.label
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prompt_dir = self.dir / "prompts"
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        self.response_dir = self.dir / "responses"
        self.response_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = (self.dir / "trace.jsonl").open("w", encoding="utf-8")
        meta_path = self.dir / "scenario_meta.json"
        meta_payload = {
            "scenario": asdict(scenario),
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    def save_prompt(self, step: int, prompt_text: str) -> Tuple[str, str]:
        rel_path = Path("prompts") / f"step_{step:03d}.txt"
        abs_path = self.dir / rel_path
        abs_path.write_text(prompt_text, encoding="utf-8")
        digest = sha1(prompt_text.encode("utf-8")).hexdigest()
        return str(rel_path), digest

    def save_response(self, step: int, payload: Dict[str, object]) -> str:
        rel_path = Path("responses") / f"step_{step:03d}.json"
        abs_path = self.dir / rel_path
        abs_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(rel_path)

    def log_step(self, record: Dict[str, object]) -> None:
        self.trace_file.write(json.dumps(record) + "\n")
        self.trace_file.flush()

    def close(self) -> None:
        self.trace_file.close()


def clamp_turn_deg(turn: float) -> int:
    """Snap arbitrary angles to the allowed [-30, 30] grid in 5° increments."""
    snapped = 5 * round(float(turn) / 5.0)
    return int(max(-30, min(30, snapped)))


def heading_to_action(turn_deg: int) -> Tuple[int, int]:
    """Map a heading delta in degrees to the closest discrete action bin."""
    snapped = clamp_turn_deg(turn_deg)
    best_idx = min(range(len(ACTION_TO_DEG)), key=lambda i: abs(ACTION_TO_DEG[i] - snapped))
    return best_idx, ACTION_TO_DEG[best_idx]


def extract_json_payload(raw_text: str) -> Optional[dict]:
    if not raw_text:
        return None
    match = JSON_OBJECT_RE.search(raw_text.strip())
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def parse_maneuver(payload: Optional[dict], fallback_phase: str) -> Tuple[int, int, str]:
    if not payload:
        return 0, 0, fallback_phase
    answer = payload.get("answer") if isinstance(payload, dict) else None
    if not isinstance(answer, dict):
        return 0, 0, fallback_phase
    maneuver = answer.get("maneuver")
    if not isinstance(maneuver, dict):
        return 0, 0, fallback_phase
    heading = clamp_turn_deg(maneuver.get("heading_change_deg", 0))
    hold_steps = int(max(0, float(maneuver.get("hold_steps", 0))))
    phase = maneuver.get("phase") or fallback_phase
    if phase not in ("WAIT_TURN", "EXECUTE_TURN", "MERGE_BACK", "JOINED"):
        phase = fallback_phase
    return heading, hold_steps, phase


def heuristic_guidance(snapshot: ObsSnapshot, phase: str, step: int) -> GuidanceDecision:
    """Very small sign-rule heuristic so we can run without a live LLM."""
    if phase == "WAIT_TURN":
        return GuidanceDecision(
            heading_deg=0,
            hold_steps=1,
            phase=phase,
            source="heuristic",
            issued_step=step,
            note="WAIT_TURN enforces straight flight; only observing.",
        )

    dx = snapshot.intr_x - snapshot.own_x
    dy = snapshot.intr_y - snapshot.own_y
    intr_bearing_deg = math.degrees(math.atan2(dy, dx))
    own_heading_deg = math.degrees(snapshot.own_heading_rad)
    bearing_error = ((intr_bearing_deg - own_heading_deg + 180.0) % 360.0) - 180.0
    direction = 1 if bearing_error >= 0 else -1

    magnitude = 10
    if abs(bearing_error) < 15:
        magnitude = 25
    elif snapshot.sep_oi < StructuredEnv.SAFE_R * 1.2:
        magnitude = 20

    heading = clamp_turn_deg(direction * magnitude)
    hold = 2 if abs(heading) <= 10 else 3
    return GuidanceDecision(
        heading_deg=heading,
        hold_steps=hold,
        phase=phase,
        source="heuristic",
        issued_step=step,
        note="Sign-rule fallback decision.",
    )


def request_guidance(
    prompt_text: str,
    snapshot: ObsSnapshot,
    *,
    use_llm: bool,
    phase: str,
    step: int,
) -> GuidanceDecision:
    if not use_llm:
        return heuristic_guidance(snapshot, phase, step)

    raw_text: Optional[str] = None
    parsed: Optional[dict] = None
    note: Optional[str] = None
    try:
        raw_text = ollama_invoke(prompt_text)
        parsed = extract_json_payload(raw_text)
    except Exception as exc:  # pragma: no cover - only hit if Ollama is unreachable
        note = f"Ollama call failed, falling back to heuristic: {exc}"
        decision = heuristic_guidance(snapshot, phase, step)
        decision.raw_text = raw_text
        decision.note = note
        return decision

    heading, hold_steps, decided_phase = parse_maneuver(parsed, phase)
    if parsed is None:
        note = "LLM response missing JSON, using heuristic result."
        decision = heuristic_guidance(snapshot, phase, step)
        decision.raw_text = raw_text
        decision.note = note
        return decision

    return GuidanceDecision(
        heading_deg=heading,
        hold_steps=hold_steps,
        phase=decided_phase,
        source="llm",
        issued_step=step,
        raw_text=raw_text,
        parsed=parsed,
        note=note,
    )


def run_scenario(
    scenario: ScenarioDefinition,
    *,
    run_dir: Path,
    args: argparse.Namespace,
    builder: BasePromptBuilder,
) -> None:
    logger = ScenarioLogger(run_dir, scenario)
    env = StructuredEnv(Reward_Params=REWARD_PARAMS, start_llm_at_step=START_LLM_AT_STEP)

    np.random.seed(scenario.seed)
    random.seed(scenario.seed)
    obs, _ = env.reset(seed=scenario.seed)

    prev_obs: Optional[np.ndarray] = None
    prev_step: Optional[int] = None
    history_vecs: Deque[np.ndarray] = deque(maxlen=args.history_length)
    history_steps: Deque[int] = deque(maxlen=args.history_length)

    hold_counter = 0
    last_guidance: Optional[GuidanceDecision] = None
    last_response_path: Optional[str] = None
    phase = "WAIT_TURN"

    max_steps = args.steps if args.steps is not None else scenario.steps
    for step_idx in range(max_steps):
        # Simple 4-phase progression: WAIT_TURN -> EXECUTE_TURN -> MERGE_BACK -> JOINED
        if phase == "WAIT_TURN" and step_idx >= args.wait_turn_steps and hold_counter <= 0:
            phase = "EXECUTE_TURN"
        elif phase == "EXECUTE_TURN" and hold_counter <= 0 and last_guidance is not None:
            phase = "MERGE_BACK"
        elif phase == "MERGE_BACK" and hold_counter <= 0 and last_guidance is not None:
            phase = "JOINED"
        history_snapshot = list(history_vecs)
        history_steps_snapshot = list(history_steps)
        prev_snapshot = prev_obs.copy() if prev_obs is not None else None
        prompt_core = builder.build_base_prompt(
            obs_vector=obs,
            step=step_idx,
            prev_obs_vector=prev_snapshot,
            prev_step=prev_step,
            history_vectors=history_snapshot,
            history_steps=history_steps_snapshot,
        )
        prompt_text = attach_phase_state_block(
            prompt_core,
            current_phase=phase,
            step_index=step_idx,
            allowed_bins=ALLOWED_BINS,
        )
        prompt_rel, prompt_sha1 = logger.save_prompt(step_idx, prompt_text)

        snap = ObsSnapshot.from_vector(obs, step=step_idx)
        new_guidance = False
        hold_display = hold_counter
        if hold_counter <= 0 or last_guidance is None:
            if phase == "JOINED":
                # Joined: no further LLM calls, keep straight flight.
                last_guidance = GuidanceDecision(
                    heading_deg=0,
                    hold_steps=1,
                    phase=phase,
                    source="joined-fixed",
                    issued_step=step_idx,
                    note="Joined phase forces straight flight.",
                )
            else:
                last_guidance = request_guidance(
                    prompt_text,
                    snap,
                    use_llm=args.call_llm,
                    phase=phase,
                    step=step_idx,
                )
            hold_counter = max(0, last_guidance.hold_steps)
            hold_display = hold_counter
            new_guidance = True
        else:
            hold_display = hold_counter
        guidance_src = "cached" if not new_guidance else last_guidance.source
        action_idx, snapped_turn = heading_to_action(last_guidance.heading_deg)
        next_obs, reward, terminated, truncated, _ = env.step(action_idx)

        response_payload = {
            "guidance": last_guidance.to_dict(),
            "used_source": guidance_src,
            "new_call": new_guidance,
            "hold_remaining": int(hold_display),
        }
        if not new_guidance and last_guidance is not None:
            response_payload["cached_from_step"] = last_guidance.issued_step
        response_rel = logger.save_response(step_idx, response_payload)
        last_response_path = response_rel

        record = {
            "scenario": scenario.label,
            "description": scenario.description,
            "step": step_idx,
            "phase": phase,
            "observation": obs.tolist(),
            "prev_observation": prev_snapshot.tolist() if prev_snapshot is not None else None,
            "history": [
                {"step": s, "vector": v.tolist()}
                for s, v in zip(history_steps_snapshot, history_snapshot)
            ],
            "action_index": action_idx,
            "applied_heading_deg": snapped_turn,
            "reward": reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "prompt_path": prompt_rel,
            "prompt_sha1": prompt_sha1,
            "response_path": last_response_path,
            "guidance": response_payload,
            "hold_remaining": int(hold_display),
        }
        logger.log_step(record)

        if terminated or truncated:
            break

        if prev_obs is not None:
            history_vecs.append(prev_obs.copy())
            history_steps.append(prev_step if prev_step is not None else max(0, step_idx - 1))
        prev_obs = obs.copy()
        prev_step = step_idx
        obs = next_obs

        if hold_counter > 0:
            hold_counter -= 1

    logger.close()


def parse_scenarios(arg: str, default_steps: Optional[int]) -> List[ScenarioDefinition]:
    labels = [token.strip() for token in arg.split(",") if token.strip()]
    scenarios: List[ScenarioDefinition] = []
    for label in labels:
        base = DEFAULT_SCENARIOS.get(label.upper())
        if base:
            steps = default_steps if default_steps is not None else base.steps
            scenarios.append(
                ScenarioDefinition(
                    label=base.label,
                    seed=base.seed,
                    steps=steps,
                    description=base.description,
                )
            )
        else:
            derived_seed = abs(hash(label)) % 10_000 + 500
            steps = default_steps if default_steps is not None else 20
            scenarios.append(
                ScenarioDefinition(
                    label=label,
                    seed=derived_seed,
                    steps=steps,
                    description="Auto-generated seed based on label.",
                )
            )
    return scenarios


def inspect_trace(trace_path: Path, limit: Optional[int]) -> None:
    if not trace_path.exists():
        raise FileNotFoundError(f"No trace file at {trace_path}")
    with trace_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            entry = json.loads(line)
            reward = entry.get("reward", 0.0)
            guidance = entry.get("guidance", {})
            heading = guidance.get("guidance", {}).get("heading_deg")
            source = guidance.get("used_source")
            hold_remaining = entry.get("hold_remaining", guidance.get("hold_remaining"))
            print(
                f"[{entry['scenario']}] step={entry['step']:02d} "
                f"phase={entry['phase']} heading={heading} hold={hold_remaining} "
                f"src={source} reward={reward:.3f}"
            )
            if limit is not None and idx + 1 >= limit:
                break


def replay_prompts(trace_path: Path, *, call_llm: bool) -> int:
    if not trace_path.exists():
        print(f"Replay file not found: {trace_path}", file=sys.stderr)
        return 1

    mismatches: List[str] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            prompt_rel = entry["prompt_path"]
            prompt_path = trace_path.parent / prompt_rel
            prompt_text = prompt_path.read_text(encoding="utf-8")
            original = entry.get("guidance", {}).get("guidance", {})
            if call_llm:
                raw = ollama_invoke(prompt_text)
                parsed = extract_json_payload(raw)
                heading, hold, _ = parse_maneuver(parsed, entry["phase"])
                if heading != original.get("heading_deg") or hold != original.get("hold_steps"):
                    mismatches.append(
                        f"Scenario {entry['scenario']} step {entry['step']}: "
                        f"recorded heading {original.get('heading_deg')} "
                        f"vs new {heading}"
                    )
            else:
                current_sha = sha1(prompt_text.encode("utf-8")).hexdigest()
                if current_sha != entry["prompt_sha1"]:
                    mismatches.append(
                        f"Scenario {entry['scenario']} step {entry['step']}: "
                        "prompt text changed (sha mismatch)."
                    )

    if mismatches:
        print("Replay detected differences:")
        for msg in mismatches:
            print(" -", msg)
        return 2

    print("Replay completed without differences.")
    return 0


class GlueTests(unittest.TestCase):
    def test_heading_clamp(self):
        self.assertEqual(clamp_turn_deg(33), 30)
        self.assertEqual(clamp_turn_deg(-17), -15)

    def test_action_mapping(self):
        idx, snapped = heading_to_action(12)
        self.assertEqual(snapped, 10)
        self.assertEqual(idx, 2)

    def test_parse_maneuver(self):
        payload = {
            "answer": {
                "maneuver": {
                    "heading_change_deg": 12,
                    "hold_steps": 4,
                    "phase": "EXECUTE_TURN",
                }
            }
        }
        heading, hold, phase = parse_maneuver(payload, "WAIT_TURN")
        self.assertEqual(heading, 10)
        self.assertEqual(hold, 4)
        self.assertEqual(phase, "EXECUTE_TURN")


def run_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GlueTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def ensure_log_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_simulations(args: argparse.Namespace) -> int:
    log_root = ensure_log_root(Path(args.log_root).resolve())
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = log_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scenarios = parse_scenarios(args.scenarios, args.steps)
    builder = BasePromptBuilder()
    run_meta = {
        "created_at": timestamp,
        "call_llm": args.call_llm,
        "history_length": args.history_length,
        "wait_turn_steps": args.wait_turn_steps,
        "scenarios": [asdict(s) for s in scenarios],
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    for scenario in scenarios:
        print(f"[Harness] Running scenario {scenario.label} (seed={scenario.seed}) …")
        run_scenario(scenario, run_dir=run_dir, args=args, builder=builder)

    print(f"Logs saved under: {run_dir}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM guidance regression harness.")
    parser.add_argument(
        "--scenarios",
        default="A,B,C,D",
        help="Comma-separated scenario labels (default: A,B,C,D).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override number of steps per scenario (default: use preset).",
    )
    parser.add_argument(
        "--wait-turn-steps",
        type=int,
        default=2,
        help="How many initial steps stay in WAIT_TURN before EXECUTE_TURN.",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=5,
        help="How many past observations to keep when building prompts.",
    )
    parser.add_argument(
        "--log-root",
        default=str(Path(__file__).resolve().parent / "logs"),
        help="Directory to store run folders.",
    )
    parser.add_argument(
        "--call-llm",
        action="store_true",
        help="If set, call the configured Ollama model instead of the heuristic stub.",
    )
    parser.add_argument(
        "--replay-file",
        type=str,
        help="Path to a trace.jsonl file to replay instead of running simulations.",
    )
    parser.add_argument(
        "--inspect-log",
        type=str,
        help="Path to a trace.jsonl file for quick textual inspection.",
    )
    parser.add_argument(
        "--inspect-limit",
        type=int,
        default=10,
        help="Maximum number of rows to print when using --inspect-log.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run glue-code unit tests and exit.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.run_tests:
        return run_tests()

    if args.inspect_log:
        inspect_trace(Path(args.inspect_log), limit=args.inspect_limit)
        return 0

    if args.replay_file:
        return replay_prompts(Path(args.replay_file), call_llm=args.call_llm)

    return run_simulations(args)


if __name__ == "__main__":
    sys.exit(main())
