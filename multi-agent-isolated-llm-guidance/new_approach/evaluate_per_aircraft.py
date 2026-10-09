"""Per-aircraft evaluation of a trained MAPPO checkpoint (AAMAS-style report card).

Flies a saved policy exactly like evaluate_rl.py (deterministic actions, same seeds), but in
addition to per-episode outcomes it records, for EVERY aircraft in every episode:
    reached          - aircraft reached its destination
    in_los           - aircraft was in a loss-of-separation pair (< SAFE_R) at termination
    in_storm         - aircraft was inside a terminal weather core at termination
    flown_distance   - path length actually flown (sum of per-step displacements)
    planned_distance - length of its planned route polyline
and the wall-clock latency of each policy decision (ms).

From these it reports the four AAMAS 2025 (IDDPG paper) metrics, all PER AIRCRAFT:
    aircraft_los_rate     = N_aircraft_in_LOS   / N_total_aircraft
    storm_los_rate        = N_aircraft_in_storm / N_total_aircraft
    goal_reach_rate       = N_aircraft_reached  / N_total_aircraft
    flight_distance_ratio = mean(flown/planned) over aircraft that reached
plus the usual per-episode rates so results can be cross-checked against evaluate_rl.py.

Evaluation only: reads a checkpoint, never writes to any run folder except --metrics-path.

Example:
    CUDA_VISIBLE_DEVICES="" python new_approach/evaluate_per_aircraft.py \
        --model-path Evaluation_distillation_approach/S_LLM_seed0/weights/last_checkpoint.pt \
        --n-episodes 300 --num-agents 8 --num-weather-cells 2 --seed 42 \
        --metrics-path Evaluation_distillation_approach/S_LLM_seed0/eval_pa_ac8.json
"""

from __future__ import annotations

# --- path bootstrap: make the project root importable when run from new_approach/
import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, _bootstrap_os.path.dirname(
    _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from configs import NUM_AGENTS_DEFAULT, NUM_WEATHER_CELLS_DEFAULT, ROUTE_IDS_DEFAULT, SEED
from evaluate_rl import _normalize_route_ids, build_agent, make_env


def _dist(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _safe_clearance(core, position) -> float:
    try:
        return float(core.weather_terminal_signed_clearance(position))
    except Exception:
        return float("inf")


def run_episode(agent, *, episode_index: int, seed: int, num_agents: int, num_weather_cells: int,
                route_ids: Optional[Sequence[str]], episode_step_cap: Optional[int]) -> Dict:
    reset_options = {
        "num_agents": int(num_agents),
        "route_ids": _normalize_route_ids(route_ids),
        "num_weather_cells": int(num_weather_cells),
    }
    env = make_env(num_agents=num_agents, num_weather_cells=num_weather_cells, episode_step_cap=episode_step_cap)
    observations, info = env.reset(seed=seed, options=reset_options)
    core = env.core
    ids: List[str] = list(core.agent_states.keys())
    prev_pos = {i: tuple(core.agent_states[i].position) for i in ids}
    flown = {i: 0.0 for i in ids}
    latency_ms: List[float] = []

    common = info["__common__"]
    done = bool(common["episode_done"])
    truncated = bool(common["episode_truncated"])
    total_reward = 0.0
    steps = 0
    while not done and not truncated:
        active_ids = list(env.agents)
        if not active_ids:
            observations, _, _, _, info = env.step({})
        else:
            t0 = time.perf_counter()
            actions, _, _ = agent.select_actions(
                global_state=env.state(),
                local_observations=observations,
                active_agent_ids=active_ids,
                episode_id=episode_index,
                deterministic=True,
                store=False,
            )
            latency_ms.append((time.perf_counter() - t0) * 1000.0)
            observations, _, _, _, info = env.step(actions)
        for i in ids:  # accumulate flown distance (unlaunched aircraft do not move -> adds 0)
            p = tuple(core.agent_states[i].position)
            flown[i] += _dist(prev_pos[i], p)
            prev_pos[i] = p
        common = info["__common__"]
        total_reward += float(common["team_reward"])
        done = bool(common["episode_done"])
        truncated = bool(common["episode_truncated"])
        steps += 1
        if steps > core.max_step + 5:
            break

    common = info["__common__"]
    reason = common["failure_reason"]

    # --- per-aircraft terminal outcomes (recomputed from simulator state; robust fallbacks) ---
    seps = getattr(core, "last_pairwise_separations", None) or {}
    los_agents = set()
    for pair, d in seps.items():
        if float(d) < float(core.safe_r):
            los_agents.update(pair)
    if reason == "collision" and not los_agents and seps:  # fallback: the closest pair
        los_agents.update(min(seps, key=seps.get))

    storm_agents = set()
    for i in ids:
        st = core.agent_states[i]
        if st.launched and not st.finished and _safe_clearance(core, st.position) < 0.0:
            storm_agents.add(i)
    if reason == "weather" and not storm_agents:  # fallback: the aircraft deepest in weather
        cand = [i for i in ids if core.agent_states[i].launched and not core.agent_states[i].finished]
        if cand:
            storm_agents.add(min(cand, key=lambda i: _safe_clearance(core, core.agent_states[i].position)))

    aircraft = []
    for i in ids:
        st = core.agent_states[i]
        try:
            planned = float(st.route.linestring.length)
        except Exception:
            planned = float("nan")
        ratio = (flown[i] / planned) if (planned and planned > 0 and not math.isnan(planned)) else None
        aircraft.append({
            "agent_id": i,
            "launched": bool(st.launched),
            "reached": bool(st.finished),
            "in_los": i in los_agents,
            "in_storm": i in storm_agents,
            "flown_distance": float(flown[i]),
            "planned_distance": planned,
            "distance_ratio": ratio,
        })

    return {
        "episode_index": int(episode_index),
        "seed": int(seed),
        "steps": int(steps),
        "total_reward": float(total_reward),
        "success": bool(common["episode_success"]),
        "truncated": bool(common["episode_truncated"]),
        "failure_reason": reason,
        "num_agents": int(num_agents),
        "num_weather_cells": int(num_weather_cells),
        "n_reached": sum(a["reached"] for a in aircraft),
        "n_in_los": sum(a["in_los"] for a in aircraft),
        "n_in_storm": sum(a["in_storm"] for a in aircraft),
        "mean_decision_ms": (statistics.fmean(latency_ms) if latency_ms else None),
        "n_decisions": len(latency_ms),
        "aircraft": aircraft,
    }


def summarize(episodes: List[Dict], model_path: str) -> Dict:
    n_ep = len(episodes)
    n_total = sum(e["num_agents"] for e in episodes)
    n_reached = sum(e["n_reached"] for e in episodes)
    n_los = sum(e["n_in_los"] for e in episodes)
    n_storm = sum(e["n_in_storm"] for e in episodes)
    ratios_reached = [a["distance_ratio"] for e in episodes for a in e["aircraft"]
                      if a["reached"] and a["distance_ratio"] is not None]
    ratios_launched = [a["distance_ratio"] for e in episodes for a in e["aircraft"]
                       if a["launched"] and a["distance_ratio"] is not None]
    lat = [e["mean_decision_ms"] for e in episodes if e["mean_decision_ms"] is not None]
    all_lat_weighted = (sum(e["mean_decision_ms"] * e["n_decisions"] for e in episodes if e["mean_decision_ms"] is not None)
                        / max(1, sum(e["n_decisions"] for e in episodes)))
    def _sd(x): return statistics.pstdev(x) if len(x) > 1 else 0.0
    return {
        "model_path": model_path,
        "num_episodes": n_ep,
        "num_agents": episodes[0]["num_agents"] if episodes else None,
        "num_weather_cells": episodes[0]["num_weather_cells"] if episodes else None,
        "total_aircraft": n_total,
        # --- AAMAS-style per-aircraft metrics ---
        "aircraft_los_rate": n_los / max(1, n_total),
        "storm_los_rate": n_storm / max(1, n_total),
        "goal_reach_rate": n_reached / max(1, n_total),
        "flight_distance_ratio_mean": (statistics.fmean(ratios_reached) if ratios_reached else None),
        "flight_distance_ratio_std": (_sd(ratios_reached) if ratios_reached else None),
        "flight_distance_ratio_n": len(ratios_reached),
        "flight_distance_ratio_launched_mean": (statistics.fmean(ratios_launched) if ratios_launched else None),
        # --- decision latency ---
        "mean_decision_ms": all_lat_weighted if lat else None,
        "decision_ms_p50": (statistics.median(lat) if lat else None),
        "decision_ms_max_episode_mean": (max(lat) if lat else None),
        # --- per-episode rates (cross-check with evaluate_rl.py) ---
        "success_rate": sum(e["success"] for e in episodes) / max(1, n_ep),
        "collision_rate": sum(e["failure_reason"] == "collision" for e in episodes) / max(1, n_ep),
        "weather_rate": sum(e["failure_reason"] == "weather" for e in episodes) / max(1, n_ep),
        "truncation_rate": sum(e["truncated"] for e in episodes) / max(1, n_ep),
        "mean_return": (statistics.fmean(e["total_reward"] for e in episodes) if episodes else 0.0),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--n-episodes", type=int, default=300)
    p.add_argument("--num-agents", type=int, default=NUM_AGENTS_DEFAULT)
    p.add_argument("--num-weather-cells", type=int, choices=(1, 2), default=NUM_WEATHER_CELLS_DEFAULT)
    p.add_argument("--route-ids", nargs="*", default=ROUTE_IDS_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--episode-step-cap", type=int, default=None)
    p.add_argument("--metrics-path", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MAPPO model not found: {model_path}")
    route_ids = _normalize_route_ids(args.route_ids)
    probe = make_env(num_agents=args.num_agents, num_weather_cells=args.num_weather_cells, episode_step_cap=args.episode_step_cap)
    probe.reset(seed=args.seed, options={"num_agents": int(args.num_agents), "route_ids": route_ids, "num_weather_cells": int(args.num_weather_cells)})
    agent = build_agent(int(probe.state().shape[0]), int(probe.observation_space(probe.possible_agents[0]).shape[0]))
    agent.load_model(model_path)

    episodes: List[Dict] = []
    bar = tqdm(range(int(args.n_episodes)), desc=f"per-aircraft eval (ac={args.num_agents})", unit="ep", dynamic_ncols=True)
    for ep in bar:
        rec = run_episode(agent, episode_index=ep, seed=int(args.seed) + ep, num_agents=int(args.num_agents),
                          num_weather_cells=int(args.num_weather_cells), route_ids=route_ids,
                          episode_step_cap=args.episode_step_cap)
        episodes.append(rec)
        n = len(episodes); tot = sum(e["num_agents"] for e in episodes)
        bar.set_postfix(succ=f"{sum(e['success'] for e in episodes)/n:.2f}",
                        acLOS=f"{sum(e['n_in_los'] for e in episodes)/tot:.3f}",
                        storm=f"{sum(e['n_in_storm'] for e in episodes)/tot:.3f}",
                        reach=f"{sum(e['n_reached'] for e in episodes)/tot:.3f}")
    bar.close()

    summary = summarize(episodes, str(model_path))
    out = {**summary, "episodes": episodes}
    mp = Path(args.metrics_path); mp.parent.mkdir(parents=True, exist_ok=True)
    with mp.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("PER-AIRCRAFT SUMMARY | episodes={num_episodes} agents={num_agents} | "
          "aircraft_LOS={aircraft_los_rate:.4f} storm_LOS={storm_los_rate:.4f} goal_reach={goal_reach_rate:.4f} "
          "dist_ratio={dr} | ep_success={success_rate:.3f} collision={collision_rate:.3f} weather={weather_rate:.3f} "
          "trunc={truncation_rate:.3f} | decision_ms={ms} | metrics={mp}".format(
              dr=(f"{summary['flight_distance_ratio_mean']:.3f}" if summary['flight_distance_ratio_mean'] is not None else "n/a"),
              ms=(f"{summary['mean_decision_ms']:.2f}" if summary['mean_decision_ms'] is not None else "n/a"),
              mp=mp, **summary))


if __name__ == "__main__":
    main()
