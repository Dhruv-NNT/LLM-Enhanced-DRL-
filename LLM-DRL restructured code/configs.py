"""Top-level configuration constants for the restructured LLM-DRL pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUN_NAME = "PPO_v8_LLM_RL"
LOG_DIR = str(PROJECT_ROOT / "log" / "PPO" / RUN_NAME)
BEST_MODEL_PATH = str(Path(LOG_DIR) / "best_model.pth")
LAST_CKPT_PATH = str(Path(LOG_DIR) / "last.ckpt")

# Environment / reward settings
REWARD_PARAMS = [-1, -1, -1, -1, -1, 1]
START_LLM_AT_STEP = 0  # match the notebook default

# PPO scheduling
MAX_TIMESTEPS = 15_000_000
UPDATE_TIMESTEP = 2048
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 10
SAVE_MODEL_FREQ = 500_000

# LLM usage policy
LLM_EVERY_K = 100
LLM_FIRST_N = 1_000_000

# Evaluation defaults (pure PPO rollout)
EVAL_RUN_NAME = "PPO_v7_LLM_RL"
EVAL_LOG_DIR = str(PROJECT_ROOT / "log" / "PPO" / EVAL_RUN_NAME)
EVAL_BEST_MODEL_PATH = str(Path(EVAL_LOG_DIR) / "best_model.pth")
EVAL_OUTPUT_DIR = str(PROJECT_ROOT / "Evaluation GiFs")

# Dataset
FEATUREFILE_PATH = str(PROJECT_ROOT / "featurefile_transformed_2.csv")

# Memory / prompt storage
MEMORY_DIR = str(PROJECT_ROOT / "memory")
JSON_ANSWERS_DIR = str(PROJECT_ROOT / "json_answers")
EVAL_MEMORY_PATH = str(PROJECT_ROOT / "json_answers" / "eval_memory.json")
