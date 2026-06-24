"""Complementarity diagnostic (the go/no-go test for the dual-teacher idea).

Question it answers: across many decision states drawn from a trained MAPPO
policy, where does each guide's short-horizon outcome beat MAPPO and beat the
LLM? If a heuristic wins in a meaningful slice of states (especially within one
scenario type), a second teacher is worth distilling. If the LLM dominates
everywhere, the dual-teacher headline is weak and we should reframe.

How it works (no training needed):
  * An existing MAPPO checkpoint drives the episodes (deterministic) and also
    plays out the short shadow rollouts -- it is the reference "competent
    controller", not something we train here.
  * At every decision step we ask each guide (LLM via Ollama, best_preview,
    preview_safe) for its action on that exact state, then shadow-roll each
    first action 5 steps forward under MAPPO and record the discounted return.
  * We compare those returns per agent, bucketed by scenario type
    (traffic / weather / boundary / nominal).

This mirrors how guidance is used in training: the controller advises on
MAPPO-driven states; the environment is driven by MAPPO.

Example:
    python3 analyze_complementarity.py \
        --mappo-ckpt llm_mappo_logs/ERandom_ce_5percent/weights/best_model.pt \
        --ollama-model gptoss:20b --n-episodes 60 \
        --num-agents 4 --num-weather-cells 2 --seed 3000

Quick mechanics check without Ollama (heuristics only):
    python3 analyze_complementarity.py --mappo-ckpt <ckpt> --no-llm --n-episodes 3
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from configs import (
    ACTION_BINS,
    LLM_SHADOW_GAMMA,
    LLM_SHADOW_HORIZON,
    LLM_SHADOW_RETURN_MARGIN,
    MAX_AGENTS,
    NUM_AGENTS_DEFAULT,
    NUM_WEATHER_CELLS_DEFAULT,
    OLLAMA_MODEL,
    ROUTE_IDS_DEFAULT,
    SEED,
)
from rl_llm_multi import GlobalLangGraphGuidanceController, JointGuidanceEnv
from rl_llm_multi.guidance import shadow_evaluate_teachers
from rl_llm_multi.utils import action_idx_to_deg, max_agents_possible
from evaluate_rl import build_agent

SCENARIOS = ("traffic", "weather", "boundary", "nominal")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_route_ids(route_ids: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not route_ids:
        return None
    values = [str(rid).strip() for rid in route_ids if str(rid).strip()]
    return values or None


def _decision_agent_ids(core) -> List[str]:
    out = []
    for agent_id in list(getattr(core, "active_agent_ids", []) or []):
        state = core.agent_states.get(agent_id)
        if state is not None and bool(getattr(state, "has_entered_sector", False)):
            out.append(str(agent_id))
    return out


def _classify_scenario(core, agent_id: str) -> str:
    st = core.get_agent(agent_id)
    if bool(getattr(st, "conflict_predicted", False)):
        return "traffic"
    if bool(getattr(st, "weather_predicted", False)):
        return "weather"
    if bool(getattr(st, "boundary_warning_active", False)):
        return "boundary"
    return "nominal"


def _build_controller(name: str, guidance_source: str, tmp_root: Path) -> GlobalLangGraphGuidanceController:
    save_dir = tmp_root / f"ctrl_{name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    return GlobalLangGraphGuidanceController(
        save_dir=str(save_dir),
        memory_path=str(tmp_root / f"memory_{name}.json"),
        memory_visual_audit_enabled=False,
        guidance_source=guidance_source,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mappo-ckpt", type=str, required=True,
                        help="Reference MAPPO checkpoint (drives rollouts + MAPPO action).")
    parser.add_argument("--ollama-model", type=str, default=None,
                        help="Ollama model for the LLM guide (e.g. gptoss:20b).")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip the LLM guide (heuristics only). For a quick mechanics check without Ollama.")
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS_DEFAULT)
    parser.add_argument("--num-weather-cells", type=int, choices=(1, 2), default=NUM_WEATHER_CELLS_DEFAULT)
    parser.add_argument("--route-ids", nargs="*", default=ROUTE_IDS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-episodes", type=int, default=60)
    parser.add_argument("--episode-step-cap", type=int, default=None)
    parser.add_argument("--shadow-horizon", type=int, default=int(LLM_SHADOW_HORIZON))
    parser.add_argument("--gamma", type=float, default=float(LLM_SHADOW_GAMMA))
    parser.add_argument("--margin", type=float, default=float(LLM_SHADOW_RETURN_MARGIN),
                        help="A guide 'beats' another only if its return is higher by this margin.")
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    route_ids = _normalize_route_ids(args.route_ids)
    if args.num_agents < 1 or args.num_agents > max_agents_possible():
        raise ValueError(f"num_agents must be 1..{max_agents_possible()}")
    if args.ollama_model:
        os.environ["OLLAMA_MODEL"] = args.ollama_model
    effective_model = os.environ.get("OLLAMA_MODEL") or OLLAMA_MODEL

    ckpt = Path(args.mappo_ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"MAPPO checkpoint not found: {ckpt}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("complementarity") / f"diag_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="complementarity_"))

    # Which guides to compare.
    guide_specs = [] if args.no_llm else [("llm", "real")]
    guide_specs += [("best_preview", "best_preview"), ("preview_safe", "preview_safe")]
    guide_names = [name for name, _ in guide_specs]
    heuristic_names = [n for n in guide_names if n != "llm"]
    has_llm = "llm" in guide_names

    print("Complementarity diagnostic")
    print(f"  reference MAPPO : {ckpt}")
    print(f"  guides          : {guide_names}")
    if has_llm:
        print(f"  ollama model    : {effective_model}")
    print(f"  episodes        : {args.n_episodes}  (seeds {args.seed}..{args.seed + args.n_episodes - 1})")
    print(f"  agents / weather: {args.num_agents} / {args.num_weather_cells}")
    print(f"  shadow horizon  : {args.shadow_horizon}  gamma={args.gamma}  margin={args.margin}")
    print(f"  out dir         : {out_dir}")

    env = JointGuidanceEnv(
        num_agents=args.num_agents,
        max_agents=MAX_AGENTS,
        num_weather_cells=args.num_weather_cells,
    )
    env.reset(seed=args.seed, num_agents=args.num_agents, route_ids=route_ids,
              num_weather_cells=args.num_weather_cells)
    global_state_dim = int(env.core.global_state_vector().shape[0])
    local_obs_dim = int(env.core.local_obs_dim)
    agent = build_agent(global_state_dim, local_obs_dim)
    agent.load_model(ckpt)

    controllers = {name: _build_controller(name, src, tmp_root) for name, src in guide_specs}

    rows: List[Dict[str, float]] = []  # one per (state, eligible agent)
    n_llm_calls = 0
    start = time.time()

    for ep in tqdm(range(int(args.n_episodes)), desc="diag", unit="ep", dynamic_ncols=True):
        env.reset(seed=int(args.seed) + ep, num_agents=args.num_agents,
                  route_ids=route_ids, num_weather_cells=args.num_weather_cells)
        if args.episode_step_cap is not None:
            env.core.max_step = int(args.episode_step_cap)
        for ctrl in controllers.values():
            ctrl.reset()

        done = False
        truncated = False
        safety_limit = int(env.core.max_step) + 5
        while not done and not truncated:
            core = env.core
            step_now = int(env.n_step)
            decision_ids = _decision_agent_ids(core)
            drive_deg: Dict[str, int] = {}

            if decision_ids:
                local_obs = {aid: core.get_local_observation(aid) for aid in decision_ids}
                mappo_idx, _, _ = agent.select_actions(
                    global_state=core.global_state_vector(),
                    local_observations=local_obs,
                    active_agent_ids=decision_ids,
                    episode_id=ep,
                    deterministic=True,
                    store=False,
                )
                drive_deg = {aid: action_idx_to_deg(int(idx), ACTION_BINS) for aid, idx in mappo_idx.items()}

                # Each guide's action on this exact state (controllers clone the
                # core internally, so calling them does not mutate it).
                guide_acts: Dict[str, Dict[str, int]] = {}
                for name, ctrl in controllers.items():
                    acts = ctrl.choose_llm_actions(core, step=step_now, latest_frame_path=None, use_vision=False)
                    guide_acts[name] = {str(a): int(d) for a, d in (acts or {}).items()}
                if has_llm and guide_acts.get("llm"):
                    n_llm_calls += 1

                # Agents every guide gave an opinion on (so comparisons are fair).
                eligible = None
                for name in guide_names:
                    keys = set(guide_acts[name].keys())
                    eligible = keys if eligible is None else (eligible & keys)
                eligible = sorted(eligible or set())

                if eligible:
                    mappo_returns, teacher_returns, _tracked = shadow_evaluate_teachers(
                        core,
                        agent,
                        proposed_actions=mappo_idx,
                        teacher_actions_deg=guide_acts,
                        horizon=int(args.shadow_horizon),
                        gamma=float(args.gamma),
                        episode_id=ep,
                    )
                    for aid in eligible:
                        row = {
                            "scenario": _classify_scenario(core, aid),
                            "g_mappo": float(mappo_returns.get(aid, 0.0)),
                        }
                        for name in guide_names:
                            row[f"g_{name}"] = float(teacher_returns.get(name, {}).get(aid, 0.0))
                        rows.append(row)

            _, _, done, truncated, _ = env.step(drive_deg)
            if env.core.n_step > safety_limit:
                break

    elapsed = time.time() - start
    _report_and_save(rows, guide_names, heuristic_names, has_llm, args, out_dir,
                     effective_model if has_llm else None, n_llm_calls, elapsed)


def _rate(values: List[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def _report_and_save(rows, guide_names, heuristic_names, has_llm, args, out_dir,
                     model, n_llm_calls, elapsed) -> None:
    margin = float(args.margin)
    by_scenario: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_scenario[r["scenario"]].append(r)

    def block(label: str, group: List[Dict]) -> Dict:
        n = len(group)
        out: Dict[str, object] = {"n": n}
        if n == 0:
            return out
        out["mean_g_mappo"] = float(np.mean([r["g_mappo"] for r in group]))
        for name in guide_names:
            out[f"mean_g_{name}"] = float(np.mean([r[f"g_{name}"] for r in group]))
            out[f"{name}_beats_mappo"] = _rate([r[f"g_{name}"] > r["g_mappo"] + margin for r in group])
        for name in heuristic_names:
            if has_llm:
                out[f"{name}_beats_llm"] = _rate([r[f"g_{name}"] > r["g_llm"] + margin for r in group])
        # Which source is the single best (argmax) on each state.
        sources = ["mappo"] + guide_names
        best_counts = {s: 0 for s in sources}
        for r in group:
            vals = {"mappo": r["g_mappo"], **{name: r[f"g_{name}"] for name in guide_names}}
            best_counts[max(vals, key=vals.get)] += 1
        out["best_source_share"] = {s: best_counts[s] / n for s in sources}
        return out

    report = {scen: block(scen, by_scenario.get(scen, [])) for scen in SCENARIOS}
    report["ALL"] = block("ALL", rows)

    # ---- pretty print ----
    line = "=" * 96
    print("\n" + line)
    print(f"COMPLEMENTARITY DIAGNOSTIC  |  states scored={len(rows)}  |  guides={guide_names}")
    print(line)
    for scen in list(SCENARIOS) + ["ALL"]:
        b = report[scen]
        if b.get("n", 0) == 0:
            print(f"\n[{scen}]  n=0  (no states of this type)")
            continue
        print(f"\n[{scen}]  n={b['n']}")
        print(f"  mean return: mappo={b['mean_g_mappo']:.3f}  " +
              "  ".join(f"{name}={b[f'mean_g_{name}']:.3f}" for name in guide_names))
        print("  beats MAPPO : " + "  ".join(f"{name}={b[f'{name}_beats_mappo']*100:.0f}%" for name in guide_names))
        if has_llm:
            print("  beats LLM   : " + "  ".join(f"{name}={b[f'{name}_beats_llm']*100:.0f}%" for name in heuristic_names))
        share = b["best_source_share"]
        print("  best source : " + "  ".join(f"{s}={share[s]*100:.0f}%" for s in share))
    print("\n" + line)

    # ---- headline read ----
    all_block = report["ALL"]
    if all_block.get("n", 0) > 0 and has_llm:
        heur_best = sum(all_block["best_source_share"].get(n, 0.0) for n in heuristic_names)
        heur_beats_llm = max(all_block.get(f"{n}_beats_llm", 0.0) for n in heuristic_names)
        print(f"HEADLINE: a heuristic is the single best choice in {heur_best*100:.0f}% of states; "
              f"a heuristic beats the LLM in up to {heur_beats_llm*100:.0f}% of states.")
        print("  Rule of thumb: >~15-20% (especially concentrated in one scenario) => a second teacher is worth it.")
    print(line)

    payload = {
        "created_at": _timestamp(),
        "mappo_ckpt": str(args.mappo_ckpt),
        "ollama_model": model,
        "guides": guide_names,
        "num_agents": int(args.num_agents),
        "num_weather_cells": int(args.num_weather_cells),
        "route_ids": _normalize_route_ids(args.route_ids),
        "seed": int(args.seed),
        "n_episodes": int(args.n_episodes),
        "shadow_horizon": int(args.shadow_horizon),
        "gamma": float(args.gamma),
        "margin": margin,
        "states_scored": len(rows),
        "n_llm_calls": int(n_llm_calls),
        "wall_clock_seconds": float(elapsed),
        "report": report,
    }
    out_path = out_dir / "complementarity_report.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Wall clock: {elapsed:.1f}s | LLM calls: {n_llm_calls}")


if __name__ == "__main__":
    main()
