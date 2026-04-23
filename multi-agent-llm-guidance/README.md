# Multi-Agent LLM Guidance

This folder contains a self-contained multi-agent extension of the original three-call guidance setup. All new code lives here; shared sector data is read from the existing root-level CSV and GeoJSON files.

## Design

- `rl_llm_multi/core.py`: framework-agnostic multi-agent simulator.
- `rl_llm_multi/env.py`: PettingZoo-style parallel wrapper and centralized joint wrapper.
- `rl_llm_multi/ppo.py`: shared-actor PPO with centralized critic and advice fusion.
- `rl_llm_multi/llm.py`: centralized three-call controller, logging, normalization, and safety repair.
- `rl_llm_multi/utils.py`: route catalog building, route staggering, sector plotting, rendering, and helpers.

## Route Capacity

The multi-agent capacity is derived from the unique `(linestring, origin, destination)` route signatures in `../22feb_paths.csv`.

- `max_agents_possible = 6`
- shared-entry groups:
  - `PATH2` and `PATH5`
  - `PATH3` and `PATH4`

Agents sharing the same pre-entry prefix are staggered so the earlier agent enters the sector before the later one launches.

## Episode Modes

- `RL_ONLY`: no LLM advice, shared-policy PPO acts directly.
- `HYBRID`: the centralized LLM emits advice embeddings; PPO remains the acting policy.
- `LLM_ONLY`: the centralized LLM emits the joint action dictionary directly.

## Outputs

- Frames and GIFs are written under `multi-agent-llm-guidance/outputs/`.
- Prompt, response, normalized JSON, and compact call indexes are written under `multi-agent-llm-guidance/memory/`.
- Rendered frames preserve the original `1200x800` output size.

## Usage

Train:

```bash
python3 multi-agent-llm-guidance/train.py --num-agents 4 --save-renders
```

Evaluate:

```bash
python3 multi-agent-llm-guidance/evaluate.py --mode HYBRID --num-agents 4
```

Pure LLM GIF episode:

```bash
python3 multi-agent-llm-guidance/run_llm_episode_gif_multi_agent.py --num-agents 4
```
