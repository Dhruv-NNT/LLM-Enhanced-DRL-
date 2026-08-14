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
NUM_AGENTS_DEFAULT = 4
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
NUM_WEATHER_CELLS_DEFAULT = 2

# Pure-RL MAPPO defaults.
MAPPO_RUN_NAME = "local_obs_k3_neighbor_risk_13th_may"
MAPPO_LOG_DIR = PROJECT_ROOT / "log" / "MAPPO" / MAPPO_RUN_NAME
# Default value for train.py --log-dir. Override this in config.py or pass --log-dir on the command line.
TRAIN_LOG_DIR = MAPPO_LOG_DIR
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
# MAPPO_CUDA_VISIBLE_DEVICES = "0"

MAPPO_CUDA_VISIBLE_DEVICES = None

MAPPO_MAX_AGENT_STEPS = 5000000
MAPPO_UPDATE_AGENT_STEPS = 2_048
MAPPO_EVAL_FREQ = 10_000
MAPPO_N_EVAL_EPISODES = 50
MAPPO_SAVE_MODEL_FREQ = 500_000

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

# LLM-guided MAPPO defaults. The master switch is intentionally false so the
# existing train.py path remains pure MAPPO unless explicitly enabled here.
USE_LLM_GUIDED_TRAINING = False
LLM_GUIDANCE_START_STEP = 0
LLM_GUIDANCE_END_STEP = 3_000_000
LLM_GUIDANCE_USE_VISION = False
LLM_GUIDANCE_MODE = "label_only_shadow_eval"
LLM_GUIDANCE_MEMORY_ISOLATED_PER_RUN = True
LLM_GUIDANCE_MEMORY_PATH = None  # Backward-compatible alias; prefer LLM_MEMORY_PATH.
LLM_MEMORY_RUN_ID = None
LLM_MEMORY_PATH = None

# Guidance source for the controller. Determines how the per-step advisory
# action is chosen. The rest of the pipeline (turn previews, shadow evaluation,
# hybrid weighting, memory writes / corrections, auxiliary loss) runs
# identically regardless of this setting.
#
#   "real"          -> call Ollama as today (production behaviour).
#   "uniform"       -> skip the LLM, uniform random pick from the stage-allowed
#                      turn bins. Tests whether the LLM beats random noise that
#                      passes through shadow eval.
#   "preview_safe"  -> skip the LLM, uniform random pick from preview rows
#                      marked safe_over_preview=True (falls back to top-sorted
#                      row if no row is safe). Tests whether the LLM beats
#                      random selection from preview-safe options.
#   "best_preview"  -> skip the LLM, deterministically pick the top-sorted
#                      preview row. Tests whether the LLM beats the
#                      controller's deterministic heuristic.
GUIDANCE_SOURCE = "real"
# Optional explicit seed for the random-guidance RNG (used by "uniform" and
# "preview_safe"). If None, Python's default random module is used.
GUIDANCE_RANDOM_SEED = 42

# LLM-guided MAPPO artifact logging.
LLM_GUIDANCE_ARTIFACTS_ENABLED = False
LLM_GUIDANCE_ARTIFACT_MAX_CALLS = 2
LLM_GUIDED_MAPPO_AUDIT_ENABLED = True
LLM_GUIDED_MAPPO_AUDIT_MAX_ROWS = 30

# Shadow rollout evaluator.
LLM_SHADOW_EVAL_ENABLED = True
LLM_SHADOW_HORIZON = 5
LLM_SHADOW_GAMMA = 0.99
LLM_SHADOW_RETURN_MARGIN = 0.05
LLM_SHADOW_RETURN_SCALE = 1.0
LLM_SHADOW_RETURN_WEIGHT_CLIP = 1.0
LLM_SHADOW_MAX_WEIGHT = LLM_SHADOW_RETURN_WEIGHT_CLIP
LLM_RETURN_WEIGHT_AGENT_COEF = 0.65
LLM_RETURN_WEIGHT_JOINT_COEF = 0.35
LLM_ADVISORY_MEMORY_WRITE_ENABLED = True
LLM_ADVISORY_MEMORY_WRITE_MARGIN = LLM_SHADOW_RETURN_MARGIN
LLM_GUIDED_MEMORY_MAX_EPISODES = 300

# Auxiliary LLM imitation loss.
LLM_LOSS_ENABLED = True
LLM_LOSS_TYPE = "ce"  # "ce" or "soft_kl"
LLM_LOSS_WEIGHT = 1.0
LLM_LOSS_DECAY_ENABLED = True
LLM_LOSS_MIN_WEIGHT = 0.05
LLM_SOFT_KL_SIGMA_DEG = 5.0
LLM_PHASE_WEIGHTS = {
    "EXECUTE_TURN": 0.5,
    "EMERGENCY_MANEUVER": 1.0,
    "MERGE_BACK": 0.6,
}

# Memory evaluator for LLM-guided MAPPO.
LLM_MEMORY_EVALUATOR_ENABLED = True
LLM_MEMORY_UPDATE_MARGIN = 0.10
LLM_MEMORY_UPDATE_WITH_BEST_OF = "mappo_or_llm"

# ---------------------------------------------------------------------------
# Dual-teacher distillation (competence-weighted).
#
# Idea: no single guide is best everywhere. We distill the MAPPO student from
# TWO complementary teachers at once -- an LLM guide and the cheap preview
# heuristic -- trusting each where it is stronger.
#
# Offline (run once): generate_guided_trajectories.py records (state -> guide
# action) for each guide; train_teachers.py turns each guide into a small
# network (pi_LLM, pi_heur). This removes the slow LLM from the training loop.
#
# Training: train.py distills BOTH teacher networks into the student with a
# competence-weighted loss, reusing the existing shadow-evaluation / auxiliary-
# loss machinery. The environment is always MAPPO-controlled.
# ---------------------------------------------------------------------------
DISTILL_DATASET_DIR = PROJECT_ROOT / "distill_data"
DISTILL_TEACHER_DIR = PROJECT_ROOT / "teachers"

# Master switch. When True, train.py distills from the two teacher networks
# instead of calling the live LLM controller / Ollama shadow evaluator.
DISTILL_ENABLED = False

# Teacher checkpoints produced by train_teachers.py.
TEACHER_LLM_PATH = DISTILL_TEACHER_DIR / "teacher_llm.pt"
TEACHER_HEUR_PATH = DISTILL_TEACHER_DIR / "teacher_heur.pt"

# Registry of all known teacher checkpoints. DISTILL_TEACHERS selects an
# arbitrary subset of these names to distill from (1, 2, or 3 teachers). The
# "heur" alias is kept for backward compatibility with the two-teacher setup;
# "best_preview"/"preview_safe" are the named heuristic teachers used in the
# three-teacher study. Add more names here to register more teachers.
DISTILL_TEACHER_PATHS = {
    "llm": TEACHER_LLM_PATH,
    "heur": TEACHER_HEUR_PATH,
    "best_preview": DISTILL_TEACHER_DIR / "teacher_best_preview.pt",
    "preview_safe": DISTILL_TEACHER_DIR / "teacher_preview_safe.pt",
}

# Teacher network architecture. Made identical to the MAPPO actor (same hidden
# dims, activation, LayerNorm, and orthogonal init) so the teacher and the
# student actor are the exact same network — removing any capacity confound in
# the distillation experiments. To shrink the teacher again, override these here.
TEACHER_HIDDEN_DIMS = MAPPO_HIDDEN_DIMS
TEACHER_ACTIVATION = MAPPO_ACTIVATION
TEACHER_USE_LAYER_NORM = MAPPO_USE_LAYER_NORM
TEACHER_ORTHOGONAL_INIT = MAPPO_ORTHOGONAL_INIT

# Teacher supervised-training defaults (train_teachers.py).
TEACHER_LR = 1e-5
TEACHER_EPOCHS = 500
TEACHER_BATCH_SIZE = 64
TEACHER_VAL_FRACTION = 0.20
TEACHER_LABEL_SMOOTHING = 0.15  # regularizer: softens targets to curb overfitting/overconfidence
TEACHER_WEIGHT_DECAY = 3e-4     # L2 regularizer: penalizes large weights to curb memorization
TEACHER_EARLY_STOP_PATIENCE = 500 

# Distillation loss applied to the MAPPO student.
#   "js"      -> Jensen-Shannon divergence between student and teacher (default).
#   "ce"      -> cross-entropy toward the teacher's argmax action.
#   "soft_kl" -> KL toward a Gaussian-smoothed target around the teacher action.
DISTILL_LOSS_TYPE = "js"
DISTILL_LOSS_WEIGHT = 0.50
DISTILL_DECAY_ENABLED = True
DISTILL_MIN_WEIGHT = 0.10
# Decay window for the distillation weight (agent steps). Past the end step the
# weight is held at DISTILL_MIN_WEIGHT so late training is pure PPO fine-tune.
DISTILL_DECAY_START_STEP = 0
DISTILL_DECAY_END_STEP = LLM_GUIDANCE_END_STEP

# Which teachers participate. Any subset of DISTILL_TEACHER_PATHS keys. Examples:
#   ("llm",)                                -> single LLM teacher (prior work)
#   ("llm", "heur")                         -> two teachers (default)
#   ("llm", "best_preview", "preview_safe") -> three-teacher study
DISTILL_TEACHERS = ("llm", "heur")
# For weather_rule competence: teacher names treated as "LLM-type" (favored when
# weather is FAR). Every other active teacher is treated as heuristic-type
# (favored when weather is NEAR). The shadow/equal modes ignore this setting.
DISTILL_LLM_TEACHER_NAMES = ("llm",)
# Reuse the existing LLM phase weights for the LLM teacher; flat for others.
DISTILL_USE_PHASE_WEIGHTS = True

# ----- Competence weighting (which teacher to trust in a given state) -----
#   "shadow"        -> reuse the shadow-rollout advantage of each teacher over
#                      MAPPO (same machinery already used for the LLM). A teacher
#                      is trusted in proportion to how much its action beats
#                      MAPPO over LLM_SHADOW_HORIZON steps.
#   "weather_rule"  -> fixed rule: trust the heuristic more when a weather cell
#                      is near, the LLM more otherwise.
#   "equal"         -> both teachers weighted equally (ablation).
COMPETENCE_MODE = "shadow"

# weather_rule parameters. "Near" means the (predicted) weather clearance is
# below CLEARANCE_MULT * core.weather_clearance_buffer_units. The threshold and
# softness are expressed as multiples of the simulator's own weather buffer
# (grid units), so they stay unit-correct regardless of the nm->grid scaling.
WEATHER_RULE_CLEARANCE_MULT = 1.5    # "near" if predicted clearance < mult * weather buffer
WEATHER_RULE_SOFTNESS_MULT = 0.5     # sigmoid width as a fraction of the buffer; <=0 -> hard switch
WEATHER_RULE_HEUR_WHEN_NEAR = 1.0
WEATHER_RULE_LLM_WHEN_NEAR = 0.0
WEATHER_RULE_HEUR_WHEN_FAR = 0.0
WEATHER_RULE_LLM_WHEN_FAR = 1.0

# Pure-LLM defaults.
RUN_MODES = ("LLM_ONLY",)
FRAME_DURATION_MS = 250
USE_VISION_DEFAULT = False

# LLM defaults.
# OLLAMA_MODEL = "llama4:latest"
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_TEMPERATURE = 0
OLLAMA_TOP_P = 0.95
OLLAMA_MAX_TOKENS = 6000 
OLLAMA_NUM_CTX = 10000 
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_SECONDS = 30
