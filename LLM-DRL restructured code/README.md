# LLM-DRL Restructured Code (Easy Guide)

This folder turns the original notebook pipeline into small Python files. The goal is to follow the same PPO + LLM idea but make each step clear and reusable.

## How the code runs (simple steps)
1. **Set knobs** – `configs.py` holds run names, log folders, reward weights, and when the LLM should join. Think of it as the single place to change paths or schedules.
2. **Start training** – `train.py` imports everything, builds the `PPO` agent, creates the `StructuredEnv`, and loops over episodes. During some episodes it also talks to the LLM through `LLMGuidanceController`.
3. **Load a scenario** – `rl_llm/env.py` creates the air-traffic world. Important pieces are:
   - `Agent`, `Ownship`, `Intruder` for aircraft state.
   - `StructuredEnv.step()` which applies an action, moves both planes, checks safety, and computes reward.
4. **Learn actions** – `rl_llm/ppo.py` keeps all PPO tools:
   - `HybridActorCritic` and `RolloutBuffer` store experience.
   - `PPO.update()` learns from the buffer.
   - `save_training_checkpoint()` and `load_training_checkpoint()` handle resumes.
5. **Talk to the LLM** – `rl_llm/llm.py` manages prompts and decisions:
   - `BasePromptBuilder` explains the current state to the model.
   - `LLMGuidanceController` decides when to ask for help, stores answers, and replays them.
   - `ghost_compare_hold()` compares the LLM turn against the pure PPO turn.
6. **Use geometry helpers** – `rl_llm/utils.py` converts raw flight data into the 0–100 grid, chooses random conflict cases, and draws the sector map.
7. **Review results** – `evaluate.py` loads the best PPO weights, plays one deterministic episode, saves frames, and turns them into a GIF.

## Important helpers (quick reference)
- `StructuredEnv.reset()` / `step()` – start and advance the episode.
- `PPO.select_action_hybrid()` – choose an action that can blend LLM advice.
- `LLMGuidanceController.next_action()` – build a prompt, get the LLM answer, and convert it into a turn + hold time.
- `ghost_compare_hold()` – simulate PPO vs LLM turns without touching the live environment.

## Before running
- Make sure `featurefile_transformed_2.csv` and the path folders under `configs.py` exist.

## Command line cheatsheet
```bash
# Train PPO + optional LLM guidance
python train.py

# Evaluate the saved policy and build GIFs
python evaluate.py
```

Logs, checkpoints, and GIFs appear in the folders printed on screen (they come from the paths in `configs.py`).
