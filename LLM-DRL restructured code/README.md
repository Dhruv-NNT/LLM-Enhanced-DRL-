# LLM-DRL Restructured Code

This directory converts the former `RL - LLM (Complete Code Pipeline).ipynb` notebook into importable Python modules and small entry-point scripts. The goal is to keep the number of files low while making the hybrid PPO + LLM pipeline easier to modify and reuse.

## Layout
- `configs.py` – central knobs (run names, logging paths, PPO schedule, reward weights).
- `train.py` – hybrid training loop with PPO + optional LLM guidance episodes.
- `evaluate.py` – deterministic rollouts that load saved PPO weights and export GIFs.
- `rl_llm/env.py` – scenario loader, `Agent`, and `StructuredEnv` gym environment.
- `rl_llm/ppo.py` – running statistics, hybrid actor-critic network, PPO wrapper, checkpoint helpers.
- `rl_llm/llm.py` – prompt builder, controller FSM, Ollama helpers, memory bookkeeping, ghost comparison.
- `rl_llm/utils.py` – geometry, plotting, and scenario-selection helpers copied from the original `utils.py`.
- `rl_llm/__init__.py` – convenience re-export of the main classes.

## Prompts to Restore
Two placeholders must be populated manually (per user request):
1. `rl_llm/llm.py` → `PROMPT_INSTRUCTIONS`: paste the “Base prompt without the comments” text from notebook cell 21.
2. `rl_llm/llm.py` → `EVALUATOR_PROMPT_TEMPLATE`: paste the evaluator rubric prompt from notebook cell 23 (inside `call_llm_evaluator`).

Leave the surrounding formatting intact; both placeholders accept multi-line strings. The evaluator template should include a `{context}` token where the episode JSON is injected.

## Usage
1. Ensure the dataset referenced in `configs.FEATUREFILE_PATH` (`featurefile_transformed_2.csv`) is available relative to the project root.
2. Install the dependencies listed in the original repository (e.g., `pip install -r requirements.txt`).
3. Launch training:
   ```bash
   python train.py
   ```
   Checkpoints and TensorBoard logs are written to the run directory printed at startup (inside `log/PPO/`).
4. Run evaluation rollouts and export GIFs:
   ```bash
   python evaluate.py
   ```
   GIFs are written to `configs.EVAL_OUTPUT_DIR`.

## Notes
- All logic is copied verbatim from the notebook cells, apart from minimal structural glue, the prompt placeholders, and the run-directory safeguards.

- Training runs now call `configs.ensure_unique_subdir` so a new timestamped log directory is created if the base name already exists; set `ALLOW_TRAIN_OVERWRITE = True` in `configs.py` if you explicitly want to reuse an existing folder.
- GIF exports use the same helper, preventing accidental overwrites; toggle `ALLOW_EVAL_OVERWRITE` when you need to reuse an output folder.
- The training script resumes from `configs.LAST_CKPT_PATH` when present, otherwise falls back to weights-only checkpoints (`checkpoint_*.pth`) or starts from scratch.
- Memory logs (`memory/run_*`) are pruned automatically to avoid runaway storage.
