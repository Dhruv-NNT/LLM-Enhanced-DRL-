"""Configuration for the multi-agent LLM guidance prototype."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT.parent

# Logging and checkpoints stay inside the new folder only.
RUN_NAME = "shared_policy_multi_agent_llm"
MODEL_NAME = "multi_agent_shared_policy"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "log" / "PPO" / RUN_NAME
BEST_MODEL_PATH = CHECKPOINT_DIR / f"{MODEL_NAME}_best.pth"
LAST_CKPT_PATH = CHECKPOINT_DIR / f"{MODEL_NAME}_latest.pth"

# Evaluation / artifacts stay inside the new folder only.
EVAL_RUN_NAME = "shared_policy_multi_agent_llm_eval"
PLOT_DIR = PROJECT_ROOT / "outputs"
EVAL_OUTPUT_DIR = PLOT_DIR / EVAL_RUN_NAME
EVAL_DIR = PROJECT_ROOT / "evaluation"
MEMORY_DIR = PROJECT_ROOT / "memory"
JSON_ANSWERS_DIR = MEMORY_DIR / "json_answers"
EVAL_MEMORY_PATH = MEMORY_DIR / "eval_memory.json"

# Shared data still lives one level above the workspace root.
ROUTES_CSV = DATA_ROOT / "22feb_paths.csv"
WAYPOINTS_CSV = DATA_ROOT / "path_Waypoints22feb.csv"
SECTOR_GEOJSON = DATA_ROOT / "test.geojson"

# Environment defaults.
MAX_AGENTS = 6
NUM_AGENTS_DEFAULT = 2
ROUTE_IDS_DEFAULT = None
SEED = 42
SAFE_R = 5.0
GOAL_RADIUS = 5.0
AGENT_SPEED = 2.0
MAX_STEP = 60
START_LLM_AT_STEP = 0
ACTION_BINS = (-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30)

# Controller defaults.
CONFLICT_LOOKAHEAD_STEPS = 8
TURN_PREVIEW_STEPS = 5
BOUNDARY_LOOKAHEAD_STEPS = 7
SAFE_STREAK_REQUIRED = 5
MERGE_BACK_REPEAT_ERROR_DEG = 30.0
USE_VISION_DEFAULT = False

# Training defaults.
REWARD_SCALE_STEP = 0.01
RUN_MODES = ("RL_ONLY", "HYBRID", "LLM_ONLY")
EPISODE_MODE_CYCLE = ("RL_ONLY", "HYBRID", "RL_ONLY", "LLM_ONLY")
MAX_TIMESTEPS = 100_000
UPDATE_TIMESTEP = 1024
EVAL_INTERVAL_EPISODES = 5
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 3
SAVE_MODEL_FREQ = 25_000

# Shared-policy PPO defaults.
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
GAMMA = 0.99
K_EPOCHS = 4
EPS_CLIP = 0.2
MINIBATCH_SIZE = 256
GAE_LAMBDA = 0.95
NORMALIZE_REWARD = True
USE_ALIGNMENT = True
LAMBDA_ALIGN = 0.10

# LLM defaults.
OLLAMA_MODEL = "llama3:8b"
OLLAMA_TEMPERATURE = 0.4
OLLAMA_TOP_P = 0.95
OLLAMA_MAX_TOKENS = 4096
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_SECONDS = 10.0
