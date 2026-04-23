# Multi-Agent Isolated LLM Guidance

This folder isolates the pure-LLM episode path from `multi-agent-llm-guidance/` so you can iterate on centralized LLM guidance without touching the RL-coupled folder.

## Included

- `configs.py`
- `evaluate.py`
- `run_llm_episode_gif_multi_agent.py`
- `rl_llm_multi/utils.py`
- `rl_llm_multi/core.py`
- `rl_llm_multi/env.py`
- `rl_llm_multi/llm.py`

## Purpose

- Keep the same sector data, route catalog logic, rendering size, and multi-agent simulator.
- Keep the centralized three-call LLM controller.
- Remove the training and PPO entrypoints from the working surface so changes can focus on pure `LLM_ONLY` episodes.

## Run

From the repo root:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --num-agents 2 --route-ids PATH2 PATH5
```

For a short smoke run:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --num-agents 2 --route-ids PATH2 PATH5 --episode-step-cap 8
```

Frames and GIFs are written under `multi-agent-isolated-llm-guidance/outputs/`. Prompt, response, and normalized JSON logs are written under `multi-agent-isolated-llm-guidance/memory/`.
