# LLM-DRL Restructured Code

## Preface: What This Project Does
This code takes an air-traffic control scenario and teaches an agent to help the ownship avoid the intruder. It uses **PPO** (a classic reinforcement learning algorithm) and can also ask help from a **Large Language Model** to suggest safe turns.

The repo started as a big notebook. Now every part sits in a small Python file. This README gives you a plain-English map before you dive into individual modules.

### Big-Picture Flow
1. Set up global knobs and file paths (`configs.py`).
2. Start training (`train.py`). This builds PPO, creates the gym environment, loops over episodes, and (sometimes) asks the LLM for advice.
3. The environment (`rl_llm/env.py`) loads conflict scenarios, moves both aircraft, computes rewards, and checks safety.
4. PPO logic (`rl_llm/ppo.py`) keeps running statistics, stores rollouts, updates actor and critic weights, and saves/loads checkpoints.
5. LLM support (`rl_llm/llm.py`) builds prompts, logs answers, manages when to query the model, and compares LLM advice vs PPO behavior.
6. Utility helpers (`rl_llm/utils.py`) convert raw flight data into the normalized grid, resample paths, and draw the sector map.
7. Evaluation (`evaluate.py`) reloads a trained model, runs a deterministic rollout, renders frames, and writes a GIF.

### Key Classes and Functions
- `StructuredEnv` (`env.py`): gym-style environment; `reset()` returns observations, `step(action)` applies the discrete turn, moves the planes, collects reward.
- `Agent`, `Ownship`, `Intruder`: internal state carriers inside the environment. They know positions, headings, offsets, and destination points.
- `HybridActorCritic` + `RolloutBuffer` (`ppo.py`): store the current policy, value predictions, actions, and rewards. `PPO.update()` handles training across mini-batches.
- `LLMGuidanceController.next_action()` (`llm.py`): core function that builds a prompt, calls the LLM, and turns the JSON response into a heading change and hold duration.
- `ghost_compare_hold()` (`llm.py`): simulates what would happen if PPO took the lead vs if the LLM suggestion ran, without touching the real environment.
- `generate_transformed_sector()` / `get_conflict_scenario()` (`utils.py`): pull a random conflict case from CSV files, rescale coordinates, and prepare everything for the environment.

### Data and Prompt Notes
- Place the datasets referenced in `configs.py` (`featurefile_transformed_2.csv`, `allresolvedtrajectories/`, `allflighttrajectories/`, etc.) in the same relative layout as the original notebook.
- Inside `rl_llm/llm.py` fill in `PROMPT_INSTRUCTIONS` and `EVALUATOR_PROMPT_TEMPLATE` with the text used in the notebook. They are left blank in the repository because they came from user-specific prompt wording.

### Running the Code
```bash
# Train with PPO + optional LLM guidance
python train.py

# Evaluate and create GIFs of the learned policy
python evaluate.py
```
Checkpoints, memory logs, and GIFs will appear in the directories printed on screen. Those paths all come from `configs.py`.

### Suggested Reading Order
If you want to inspect the code step by step, a good order is:
1. `configs.py`: see all the knobs in one place.
2. `train.py`: main training loop and how PPO + LLM interact.
3. `rl_llm/env.py`: how the scenario is loaded and stepped.
4. `rl_llm/ppo.py`: the PPO implementation details.
5. `rl_llm/llm.py`: prompts, controller, and memory handling.
6. `rl_llm/utils.py`: data prep and plotting.
7. `evaluate.py`: clean evaluation flow.

With this overview you should be able to navigate the modules quickly and understand how the pieces fit together.
