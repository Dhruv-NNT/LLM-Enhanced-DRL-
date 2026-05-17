"""Configuration for the isolated multi-agent LLM/RL guidance sandbox."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT.parent

# Pure-LLM artifacts stay inside the new folder only.
RUN_NAME = "isolated_multi_agent_llm"
MODEL_NAME = "isolated_multi_agent_llm"
LOG_DIR = PROJECT_ROOT / "log" / RUN_NAME

EVAL_RUN_NAME = "isolated_multi_agent_llm_eval"
PLOT_DIR = PROJECT_ROOT / "outputs"
EVAL_OUTPUT_DIR = PLOT_DIR / EVAL_RUN_NAME
EVAL_DIR = PROJECT_ROOT / "evaluation"

# Pure-LLM prompt/response artifacts are audit logs, not persistent memory.
LLM_EPISODE_LOG_DIR = PROJECT_ROOT / "llm_episode_logs"
LLM_EPISODE_LOG_RETENTION_ENABLED = True
LLM_EPISODE_LOG_MAX_EPISODE_DIRS = 3

# Reserved for future persistent memory design.
MEMORY_DIR = PROJECT_ROOT / "memory"
JSON_ANSWERS_DIR = LLM_EPISODE_LOG_DIR / "json_answers"
EVAL_MEMORY_PATH = MEMORY_DIR / "eval_memory.json"
DECISION_MEMORY_PATH = MEMORY_DIR / "decision_memory.json"
DECISION_MEMORY_MAX_EPISODES = 100
DECISION_MEMORY_MIN_SIMILARITY = 0.75
DECISION_MEMORY_VISUAL_AUDIT_ENABLED = False
DECISION_MEMORY_VISUAL_AUDIT_DIR = MEMORY_DIR / "visual_audit"
DECISION_MEMORY_VISUAL_AUDIT_FRAME_COUNT = 4
DECISION_MEMORY_PROMPT_UNVERIFIED_ACTIONS = True

# Shared data still lives one level above the workspace root.
ROUTES_CSV = DATA_ROOT / "22feb_paths.csv"
WAYPOINTS_CSV = DATA_ROOT / "path_Waypoints22feb.csv"
SECTOR_GEOJSON = DATA_ROOT / "test.geojson"

# Environment defaults.
ROUTE_MIRROR_SUFFIX = "_REV"
MAX_AGENTS = 12
MAX_WEATHER_CELLS = 2
NUM_AGENTS_DEFAULT = 2
ROUTE_IDS_DEFAULT = None
SEED = 42
SAFE_R = 5.0
TRAFFIC_CAUTION_R = 6.5
LOCAL_TRAFFIC_NEIGHBOR_COUNT = 3
LAUNCH_SEPARATION_R = TRAFFIC_CAUTION_R
GOAL_RADIUS = 5.0
AGENT_SPEED = 2.0
MAX_STEP = 60
ACTION_BINS = (-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30)

# Reward defaults.
#
# Dense rewards are per aircraft and bounded component-by-component so training
# curves stay readable across 2-12 aircraft. Terminal outcomes are shared by all
# aircraft trajectories and are intentionally large enough to dominate normal
# step progress.
#
# Coefficient intent:
# - risk penalties/reduction bonuses should dominate normal progress while a
#   conflict exists, so deviation is acceptable;
# - progress, cross-track recovery, and heading alignment should regain strength
#   as risk falls, so aircraft merge back toward the route/destination.
REWARD_PROGRESS_SCALE = 1.0
REWARD_PROGRESS_CLIP = 1.0
REWARD_PROGRESS_HAZARD_RELIEF = 0.60
REWARD_STEP_PENALTY = 0.05
REWARD_CROSS_TRACK_SCALE = 0.35
REWARD_CROSS_TRACK_CLIP = 2.0
REWARD_CROSS_TRACK_HAZARD_RELIEF = 0.75
REWARD_CROSS_TRACK_RECOVERY_SCALE = 0.55
REWARD_CROSS_TRACK_RECOVERY_CLIP = 1.0
REWARD_MERGE_BACK_RISK_THRESHOLD = 0.30
REWARD_MERGE_BACK_SCALE = 0.40
REWARD_DEST_HEADING_SCALE = 0.50
REWARD_TRAFFIC_RISK_PENALTY = 2.5
REWARD_WEATHER_RISK_PENALTY = 2.5
REWARD_RISK_REDUCTION_CLIP = 1.0
REWARD_TRAFFIC_RISK_REDUCTION_SCALE = 1.8
REWARD_WEATHER_RISK_REDUCTION_SCALE = 1.8
REWARD_GREEN_WEATHER_PENETRATION_SCALE = 2.0
REWARD_FINISHED_AIRCRAFT = 30.0
REWARD_COLLISION_PENALTY = 100.0
REWARD_TERMINAL_WEATHER_PENALTY = 100.0
REWARD_TRUNCATION_PENALTY = 80.0
REWARD_TRUNCATION_DISTANCE_PENALTY_SCALE = 60.0
REWARD_TEAM_SUCCESS = 120.0

# Hazard detection and recovery defaults.
HAZARD_LOOKAHEAD_STEPS = 12
HAZARD_RISK_DECAY_EXPONENT = 0.4
EMERGENCY_LOOKAHEAD_STEPS = 5
TURN_PREVIEW_STEPS = 10
PREVIEW_TOP_K = 6
PAIR_RISK_BUFFER = 1.8
WEATHER_CLEARANCE_BUFFER_NM = 8.0
BOUNDARY_WARNING_DISTANCE_UNITS = 2.0
CLEAR_STREAK_REQUIRED = 2
ROUTE_RECOVERY_XTRACK_UNITS = 2.5
DESTINATION_ALIGNMENT_DEG = 8.0
MERGE_BACK_REPEAT_ERROR_DEG = 30.0
MERGE_BACK_RECHECK_STEPS = 2

# Weather defaults.
WEATHER_MAJOR_RADIUS_NM_MIN = 2.5
WEATHER_MAJOR_RADIUS_NM_MAX = 25.0
WEATHER_MINOR_RATIO_MIN = 0.60
WEATHER_MINOR_RATIO_MAX = 0.65
WEATHER_SPEED_NM_PER_STEP_MIN = 0.5
WEATHER_SPEED_NM_PER_STEP_MAX = 1.1666666667
WEATHER_GROWTH_NM_PER_STEP_MIN = 0.15
WEATHER_GROWTH_NM_PER_STEP_MAX = 0.60
WEATHER_TRAIL_STEPS = 5
WEATHER_INIT_TRIES = 300
NUM_WEATHER_CELLS_DEFAULT = 1

# Pure-RL MAPPO defaults.
MAPPO_RUN_NAME = "local_obs_k3_neighbor_risk_13th_may"
MAPPO_LOG_DIR = PROJECT_ROOT / "log" / "MAPPO" / MAPPO_RUN_NAME
MAPPO_BEST_MODEL_PATH = MAPPO_LOG_DIR / "best_model.pt"
MAPPO_LAST_CKPT_PATH = MAPPO_LOG_DIR / "last.ckpt"
MAPPO_EVAL_OUTPUT_DIR = PLOT_DIR / "mappo_eval"
MAPPO_METRICS_DIR = PROJECT_ROOT / "evaluation" / "mappo"

# GPU visibility for MAPPO train/eval.
# Examples:
#   None   -> respect shell CUDA_VISIBLE_DEVICES
#   "0"    -> use only physical GPU 0
#   "1"    -> use only physical GPU 1
#   "0,1"  -> expose GPUs 0 and 1; current MAPPO still uses one visible GPU
#   ""     -> hide GPUs and run on CPU
MAPPO_CUDA_VISIBLE_DEVICES = "0,1"

MAPPO_MAX_AGENT_STEPS = 5_000_000
MAPPO_UPDATE_AGENT_STEPS = 2_048
MAPPO_EVAL_FREQ = 10_000
MAPPO_N_EVAL_EPISODES = 50
MAPPO_SAVE_MODEL_FREQ = 50_000

# MAPPO_LR_ACTOR = 1e-4 # Original setting, which produced good results in early testing but may have been a bit high for stable convergence in the more complex environment. Left unchanged in the config for reference and easy reversion if desired.
# MAPPO_LR_CRITIC = 1e-4 # Original setting, which produced good results in early testing but may have been a bit high for stable convergence in the more complex environment. Left unchanged in the config for reference and easy reversion if desired.
MAPPO_LR_ACTOR = 1e-4
# MAPPO_LR_CRITIC = 5e-5 # Planned LR based on the training curves observed in the 12th may local observation run but to compare the LLM+RL on equal grounds need to switch back to 1e-4
MAPPO_LR_CRITIC = 1e-4

MAPPO_GAMMA = 0.99
MAPPO_K_EPOCHS = 5
MAPPO_EPS_CLIP = 0.2
MAPPO_MINIBATCH_SIZE = 256
MAPPO_GAE_LAMBDA = 0.95
MAPPO_HIDDEN_DIMS = (256, 256, 128)
MAPPO_ACTIVATION = "silu"
MAPPO_USE_LAYER_NORM = True
MAPPO_ORTHOGONAL_INIT = True
# Legacy single-width setting kept for scripts that still import it directly.
MAPPO_HIDDEN_DIM = MAPPO_HIDDEN_DIMS[0]
MAPPO_NORMALIZE_REWARD = False

# Pure-LLM defaults.
RUN_MODES = ("LLM_ONLY",)
FRAME_DURATION_MS = 250
USE_VISION_DEFAULT = False

# LLM defaults.
# OLLAMA_MODEL = "llama4:latest"
OLLAMA_MODEL = "gemma3:12b"
OLLAMA_TEMPERATURE = 0
OLLAMA_TOP_P = 0.95
OLLAMA_MAX_TOKENS = 6000 
OLLAMA_NUM_CTX = 10000 
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_SECONDS = 30
