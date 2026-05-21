# CLAUDE.md — Project Context for AI Assistants

This file documents the full context of the `multi-agent-isolated-llm-guidance` project for use by AI coding assistants. It covers the research question, system architecture, training flow, memory design, and a critical comparison against related published work. Read this before making any code changes.

---

## 1. Research Question

**Can shadow-validated LLM advisory guidance improve MAPPO training in a multi-agent aircraft sector environment while keeping real environment execution fully MAPPO-controlled?**

The core idea is to use an LLM as a teacher that proposes advisory heading changes, validate those proposals through short cloned rollouts ("shadow evaluation"), and inject only validated, improvement-weighted signals into the MAPPO policy via an auxiliary imitation loss. The LLM never directly controls the real environment. MAPPO remains the sole source of actions executed in the environment.

---

## 2. Project Structure

```
multi-agent-isolated-llm-guidance/
├── configs.py                    # All hyperparameters, paths, constants
├── train.py                      # Main MAPPO training loop
├── evaluate_llm.py               # Pure-LLM episode evaluation (text + vision)
├── evaluate_rl.py                # MAPPO policy evaluation metrics
├── rl_llm_multi/
│   ├── __init__.py               # Public API exports
│   ├── core.py                   # Simulator: MultiAgentSectorCore, AgentState
│   ├── mappo.py                  # RL algorithm: MAPPO, RolloutBuffer, Actor-Critic
│   ├── guidance.py               # Interfaces: LLMGuidanceProvider, ShadowEvaluationResult
│   ├── llm.py                    # LLM controller: GlobalLangGraphGuidanceController
│   ├── memory.py                 # Decision memory: DecisionMemoryStore, case building
│   ├── utils.py                  # Routes, weather, rendering, helpers
│   └── env.py                    # Gymnasium wrappers: MultiAgentParallelEnv, JointGuidanceEnv
├── memory/
│   └── decision_memory.json      # Persistent cross-episode LLM decision memory
├── LLM-Guidance-Readme.md        # LLM controller design, prompt structure, vision mode
├── Training Flow.md              # Complete training algorithm and experiment design
├── Memory-readme.md              # Decision memory system design and usage
└── CLAUDE.md                     # This file
```

**Data files required (not in repo):**
- `22feb_paths.csv` — route definitions
- `path_Waypoints22feb.csv` — waypoint lat/lon
- `test.geojson` — sector polygon

---

## 3. Environment: Multi-Agent Sector Simulator

**File:** `rl_llm_multi/core.py` — `MultiAgentSectorCore`

The environment simulates multiple aircraft moving through a controlled airspace sector. It is a custom simulator with no community benchmark equivalent.

### Key Constants (`configs.py`)

| Constant | Value | Meaning |
|---|---|---|
| `SAFE_R` | 5.0 | Actual collision threshold (terminates episode) |
| `TRAFFIC_CAUTION_R` | 6.5 | Soft caution buffer shown to LLM |
| `GOAL_RADIUS` | 5.0 | Distance to destination for task completion |
| `AGENT_SPEED` | 2.0 | Units per simulator step |
| `MAX_STEP` | 60 | Episode step limit |
| `ACTION_BINS` | (-30,-25,...,25,30) | 13 discrete heading changes in degrees |
| `HAZARD_LOOKAHEAD_STEPS` | 12 | Forward-look horizon for hazard prediction |
| `TURN_PREVIEW_STEPS` | 10 | Steps simulated per candidate turn before showing to LLM |

**Positive heading = left turn. Negative heading = right turn. Zero = hold.**

### Aircraft Lifecycle

1. **Before sector entry**: follows pre-planned route exactly (no LLM/RL heading control)
2. **After sector entry**: free flight — RL/LLM chooses heading changes
3. **At destination** (`dist < GOAL_RADIUS`): marked finished, removed from active

### Hazard Prediction (12-step lookahead)

The simulator clones itself and simulates forward to predict:
- Minimum traffic separation and first predicted pair-loss step
- Weather entry step and minimum weather clearance
- Boundary exit step
- `safe_streak` counter (consecutive hazard-free updates, used for merge-back eligibility)

### Reward Structure

**Per-aircraft dense reward (per step):**
```
agent_dense_reward =
  + 1.00 * safe_progress          (gated by risk)
  - 0.05                          (time penalty)
  - 0.35 * relaxed_cross_track    (gated by risk)
  + 0.55 * safe_cross_track_recovery
  - 0.50 * safe_heading_penalty
  - 2.50 * traffic_risk
  - 2.50 * weather_risk
  + 1.80 * traffic_risk_reduction
  + 1.80 * weather_risk_reduction
  + merge_back_bonus
  + 30.0  if just finished
```

**Terminal reward (shared):**
```
team success:    +120 per launched aircraft
collision:       -100 for collision aircraft, -100/sqrt(n_agents) for others
weather failure: -100 for violating aircraft, -100/sqrt(n_agents) for others
truncation:      -80/sqrt(n_agents) + distance penalty per unfinished aircraft
```

Design rationale: dense rewards are per-aircraft and component-clipped to stay bounded across 2–12 agent scenarios; terminal outcomes dominate episode quality.

### Episode Termination

| Condition | Reason |
|---|---|
| Any pair distance < `SAFE_R` | `collision` |
| Any aircraft enters terminal weather core | `weather` |
| All aircraft finished | `success` |
| `n_step >= MAX_STEP` | `truncated` |

---

## 4. RL Algorithm: MAPPO

**File:** `rl_llm_multi/mappo.py`

### Architecture

- **Actor (decentralized):** local observation (~40 dims) → action probabilities over 13 action bins
- **Critic (centralized, agent-conditioned):** global state + one-hot agent ID → value estimate
- **Hidden dims:** (256, 256, 128), SiLU activations, LayerNorm after each hidden layer, orthogonal initialization
- **Separate networks** for actor and critic (not shared lower layers)

### Training Hyperparameters

| Param | Value |
|---|---|
| Learning rate (both) | 1e-4 |
| Discount γ | 0.99 |
| PPO clip ε | 0.2 |
| K epochs | 5 |
| Minibatch size | 256 |
| GAE λ | 0.95 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |

**Updates happen only after a complete episode ends**, ensuring terminal rewards are in the buffer before computing returns.

### PPO Loss

```
ratio = exp(new_logprob - old_logprob)
policy_loss = -mean(min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv))
L_MAPPO = policy_loss + 0.5 * value_loss - 0.01 * entropy_mean
```

### LLM Auxiliary Imitation Loss

Active only when: `LLM_LOSS_ENABLED=True`, label available, `llm_return_weight > 0`.

```
CE_i = -log π_θ(llm_action_i | obs_i)
effective_weight_i = llm_return_weight_i * phase_weight_i
L_LLM = mean(CE_i * effective_weight_i)
L_total = L_MAPPO + 0.25 * L_LLM
```

**Phase weights:**
- `EXECUTE_TURN`: 0.5
- `EMERGENCY_MANEUVER`: 1.0
- `MERGE_BACK`: 0.6

An alternative `soft_kl` loss type is also available (soft target distribution centered on LLM action), controlled by `LLM_LOSS_TYPE`.

### Local Observation Vector (24 dims)

Position (x,y), destination (x,y), dist-to-dest, heading (rad), route heading (deg), heading-error-to-route, signed cross-track, inside-sector flag, entered-sector flag, boundary dist, nearest-neighbor dist, nearest-neighbor bearing, conflict-predicted, weather-predicted, hazard-predicted, safe-streak, predicted pair-loss step, predicted weather-entry step, predicted min-sep, predicted weather clearance, weather-center dist, weather relative bearing.

---

## 5. LLM Guidance System

**File:** `rl_llm_multi/llm.py`

### LangGraph Controller Pipeline

```
collect_state
  → hltp_on_sector_entry    (high-level tactical planning on sector entry)
  → assign_stages           (EXECUTE_TURN / EMERGENCY_MANEUVER / MERGE_BACK / WAIT_CLEAR)
  → build_global_context    (single prompt for all eligible aircraft)
  → call_llm                (Ollama API call)
  → parse_validate_retry    (JSON extraction, retry up to 2x)
  → guardrail_actions       (snap to action bin, snap to call-type allowed turns)
  → fallback_missing        (best preview candidate for any missing aircraft)
  → commit_and_log          (finalize, update controller state, store memory)
```

### Ollama Settings

| Setting | Value |
|---|---|
| Model | `gemma3:12b` |
| Host | `http://127.0.0.1:11434` |
| Temperature | 0 |
| top_p | 0.95 |
| num_ctx | 32768 |
| Timeout | 30s |

### Guidance Stages

| Stage | Trigger | Zero allowed? |
|---|---|---|
| `EXECUTE_TURN` | First call after sector entry | Yes |
| `EMERGENCY_MANEUVER` | Hazard predicted within `EMERGENCY_LOOKAHEAD_STEPS` | No (must turn) |
| `MERGE_BACK` | Safe streak reached / off-route / boundary warning | Only turns toward destination |
| `WAIT_CLEAR` | No guidance needed | (no call made) |

### Turn Preview System

Before every LLM call, for each allowed turn:
1. Clone simulator
2. Apply candidate turn on step 1
3. Simulate `TURN_PREVIEW_STEPS` (10) steps deterministically (no new turns)
4. Record per-row: `safe_over_preview`, `pair_loss_step`, `weather_entry_step`, `boundary_exit_step`, `min_sep`, `traffic_buffer_ok`, `min_weather_clearance`, `weather_buffer_ok`, `progress_to_destination`, `cross_track_reduction`, `end_distance`, `end_cross_track`, `end_heading_error`

**The LLM selects from pre-computed consequences, not from first principles.** This is the key guardrail reducing LLM hallucination.

Row sorting (normal/emergency): safe → no pair loss → traffic buffer → no weather → weather buffer → no boundary exit → larger min_sep → larger clearance → smaller heading error → smaller cross-track → smaller turn size.

Row sorting (merge-back): safe first, then progress → cross-track reduction → heading error → min_sep → turn size.

### Prompt Sections (text mode)

```
GLOBAL GUIDANCE TASK
MERGE_BACK PRIORITY
GUIDANCE-ELIGIBLE AGENTS THIS STEP
ENTERED TRAFFIC CONTEXT
HIGH-LEVEL TACTICAL PLANS
GLOBAL WEATHER
RANKED TRAFFIC THREATS
PER-AGENT TURN PREVIEWS
RECENT GLOBAL DECISION MEMORY
FRAME CONTEXT
```

### Vision Mode

Enabled with `--use-vision`. Attaches up to 4 recent rendered frames as base64 images to the Ollama request. Vision prompt tells LLM to inspect frames first, then anchor with text. Adds an extra candidate-row safety resolver: if LLM-requested turn has pair loss but another displayed row avoids it, the safer row is substituted.

### HLTP (High-Level Tactical Planning)

Runs once per aircraft on sector entry. Looks ahead to step 60 to detect conflicts, suggests right-side avoidance waypoints and merge-back waypoints. HLTP plans are stored and included in later step prompts as advisory context. Does not directly apply heading changes.

### Action Guardrails

1. Snap LLM-requested heading change to nearest `ACTION_BINS` value
2. Snap again to the allowed turns for the current call type
3. Fallback to best-sorted preview candidate if result is still invalid

---

## 6. Training Flow

**File:** `Training Flow.md` (read in full for complete details)

### Core Isolation Property

```python
final_actions = policy_actions  # always MAPPO
env.step(final_actions)          # LLM never touches real env
```

### Episode-Level Guidance Latching

```python
episode_llm_guidance_active = (
    USE_LLM_GUIDED_TRAINING
    and LLM_GUIDANCE_START_STEP <= agent_steps_at_episode_start < LLM_GUIDANCE_END_STEP
)
```

**Current defaults:** `USE_LLM_GUIDED_TRAINING = False`, window `[0, 100000)`.

The decision is latched at episode start — guidance cannot switch on/off mid-episode.

### Per-Step Flow (when guidance active)

1. **MAPPO selects actions** for all active aircraft → `policy_actions`
2. **LLM generates advisory labels** for eligible aircraft via `guidance.advice_for_step()`
3. **Shadow evaluation:** clone simulator, run 3 branches for `LLM_SHADOW_HORIZON=5` steps:
   - MAPPO branch: all aircraft use MAPPO first-step action
   - LLM branch: eligible aircraft use LLM first-step action, others use MAPPO
   - Memory branch: aircraft with retrieved memory use memory first-step action, others use MAPPO
   - Steps 2–5: all branches use deterministic MAPPO
4. **Hybrid return weight:**
   ```
   delta_agent_i = G_LLM_i - G_MAPPO_i
   w_agent_i = 0 if delta <= 0.05 else clip(delta / 1.0)
   delta_joint = G_LLM_joint - G_MAPPO_joint
   w_joint = 0 if delta <= 0.05 else clip(delta / 1.0)
   llm_return_weight_i = clip(0.65 * w_agent_i + 0.35 * w_joint)
   ```
5. **Advisory memory write** (if `G_LLM_i > G_MAPPO_i + 0.05`): store LLM action in `decision_memory.json`
6. **Memory correction** (if retrieved memory beat by MAPPO or LLM by margin 0.10): fill `result.corrected` and `result.note`
7. **Real step:** `env.step(policy_actions)`
8. **Buffer storage:** MAPPO transition + full LLM metadata attached

### Key Buffer Metadata Fields

`llm_label_available`, `llm_action_idx`, `llm_action_deg`, `llm_phase`, `G_MAPPO_shadow`, `G_LLM_shadow`, `G_MEMORY_shadow`, joint equivalents, `llm_return_weight`, `memory_ref`, `memory_update_applied`, `advisory_memory_written`

---

## 7. Decision Memory System

**File:** `rl_llm_multi/memory.py`, documented in `Memory-readme.md`

### Persistent File

```
memory/decision_memory.json
```

Top-level shape: `{ "schema_version": 1, "episodes": [...] }`

Retention: latest 100 episodes (not cases). One episode can have many cases.

### Case Fields

Each case stores: `case_id`, `step`, `agent`, `route` (fingerprint: `route_id|origin|dest|reversed`), `call`, `reason`, `type` (traffic/weather/boundary/merge_back/nominal), `place` (route_frac, xtrack, heading_err), `traffic` (up to 2 threats: route, dist, bearing, loss_step, min_sep), `weather` (clear_now, clear_pred, bearing, entry, closing), `boundary` (dist, bearing, exit, closing), `preview` (safe set, unsafe set, best_safe, best_progress), `result` (action, corrected, note).

### Hard Gates (all must pass before scoring)

1. Same ownship route fingerprint
2. Same call name
3. Same scenario type
4. Traffic cases: same primary intruder route
5. Remembered action must exist in current preview
6. If corrected action exists, it must also exist in current preview
7. Neither action has pair loss/weather entry if safer alternatives exist

### Similarity Score (threshold: 0.75)

| Component | Weight |
|---|---|
| place (route_frac 45%, xtrack 25%, heading_err 30%) | 25% |
| traffic (dist, bearing, loss_step, min_sep, secondary) | 20% |
| weather (clear_now, clear_pred, bearing, entry, closing) | 15% |
| boundary (dist, bearing, exit, closing) | 15% |
| preview (safe set, unsafe set, best_safe, best_progress) | 15% |
| misc (reason, seed, num_agents, step) | 10% |

### Prompt Insertion

Max 2 memory lines total, max 1 per aircraft. Only shown if similarity >= 0.75 and action passes safety checks.

Unverified memory:
```
- A2 sim=0.82 unverified memory comparison only: similar past case used -15 deg. Do not copy unless current preview independently supports it.
```

Corrected memory:
```
- A2 sim=0.86 evaluator-corrected memory: past -15 deg was poor because "traffic buffer kept shrinking"; corrected -25 deg is preferred only if current preview is safe.
```

**Design principle: prefer no memory over wrong memory.**

### What Is Not Yet Implemented

- Standalone evaluator LLM call path from `evaluate_llm.py` (memory correction currently only triggered from within the MAPPO training loop shadow evaluation)
- The evaluator engine described in `Memory-readme.md` as "next stage" is partially implemented in `train.py` but not in the pure LLM evaluation path

---

## 8. Experiment Design

**File:** `Training Flow.md` — Suggested Experiments section

| Experiment | Key Settings | Purpose |
|---|---|---|
| E0 | `USE_LLM_GUIDED_TRAINING=False` | MAPPO baseline |
| E1 | LLM on, shadow on, memory off | Test if shadow-validated labels help |
| E2 | LLM on, shadow on, memory on | Full method |
| E3 | `agent_coef=1.0, joint_coef=0.0` | Individual-only weighting |
| E4 | `agent_coef=0.0, joint_coef=1.0` | Joint-only weighting |
| E5 | Memory write on, correction off | Isolate memory write from correction |
| E6 | Three margin variants (0.00/0.05/0.10) | Margin sensitivity |

**Recommended run order:** E0 → E1 → E2 → E3 → E5 → E6 → E4

**Primary decision rules:**
- E1 beats E0 → LLM labels are useful
- E2 beats E1 → memory adds value
- Positive-label count near zero → margin too strict or LLM rarely beats MAPPO

**Recommended seeds:** `[0, 1, 2]` for pilot runs, `[0, 1, 2, 3, 4]` for full runs.

### TensorBoard Metrics

Key training meters: `train/success`, `train/collision`, `train/weather_failure`, `train/truncated`, `train/success_rate_running`.

Key LLM meters: `train/llm/positive_label_count`, `train/shadow/llm_beats_mappo_rate`, `train/llm/return_weight_hybrid_mean`.

Key memory meters: `train/memory/advisory_write_count`, `train/memory/update_count`, `train/memory/corrected_to_mappo_count`.

### Run Directory Layout

```
runs/<log-dir>/
  run_hyperparameters.txt
  tensorboard/<run-name>/
  weights/
    best_model.pt
    last_checkpoint.pt
  memory/
    decision_memory.json
  llm_guidance_json/           (optional, LLM_GUIDANCE_ARTIFACTS_ENABLED)
  llm_guided_mappo_audit.jsonl (compact per-decision audit, latest 5000 rows)
```

---

## 9. Running the System

### LLM Evaluation (no MAPPO training)

```bash
# Text mode (default)
python3 multi-agent-isolated-llm-guidance/evaluate_llm.py

# Vision mode
python3 multi-agent-isolated-llm-guidance/evaluate_llm.py --use-vision

# More aircraft
python3 multi-agent-isolated-llm-guidance/evaluate_llm.py --num-agents 8

# Two weather cells
python3 multi-agent-isolated-llm-guidance/evaluate_llm.py --num-weather-cells 2

# Specific routes
python3 multi-agent-isolated-llm-guidance/evaluate_llm.py --num-agents 2 --route-ids PATH5_REV PATH6

# With memory visual audit
python3 multi-agent-isolated-llm-guidance/evaluate_llm.py --memory-visual-audit
```

### MAPPO Training

```bash
# E0: baseline
python3 train.py --log-dir runs/E0_baseline_seed0 --seed 0

# E2: full method
python3 train.py --log-dir runs/E2_full_seed0 --seed 0
```

### Tests

```bash
python3 -m unittest discover multi-agent-isolated-llm-guidance/tests
```

Many tests patch Ollama, so a live server is not required.

---

## 10. Critical Analysis vs. Related Published Work

The following papers were used as inspiration and comparison points. They are in `LLM-DRL Research Papers/`.

---

### Paper 1: ULTRA — LLM-Guided RL (Lehigh University, May 2025)

**Core idea:** Two-stage post-hoc RL fine-tuning. LLM reads historical trajectories from a pre-trained sub-optimal agent, identifies "critical states," then suggests replacement actions (stored in a lookup table) or generates dense rewards to guide further training.

**Genuine novelty:**
- Two-stage separation: LLM as trajectory analyst first, corrector second
- Lookup-table cache for critical-state corrections avoids per-step LLM calls at fine-tuning time
- Case-analysis-first reward generation reduces LLM per-step inconsistency

**Real weaknesses:**
- Lookup table uses direct state equality matching — brittle in continuous state spaces; most states will never match a stored entry
- No ablation proves LLM identifies actually-critical states vs. any random subset
- Critical missing control: does extended PPO training with the same compute budget beat LLM-guided fine-tuning? This is never tested
- Only GPT-4o tested; no analysis of weaker or local models
- LLM actions injected directly into training trajectories without validation — if LLM is wrong, it corrupts training with no recourse
- Benchmarks (Pong, Hopper, Walker2d, Ant) are precisely the settings where LLM world knowledge is most applicable; results unlikely to generalize to novel domains

**Evaluation quality: Weak-to-moderate.** No equivalent-compute control. Improvement margins are modest (5–14% on MuJoCo tasks).

---

### Paper 2: TeLL-Drive — Teacher LLM + DRL (Tongji/MSU, Feb 2025)

**Core idea:** Teacher-student architecture for autonomous driving. LLM teacher with chain-of-thought reasoning, scenario memory retrieval, and a reflective evaluator. Self-attention mechanism fuses teacher and student embeddings. KL divergence constraint keeps student policy close to teacher during early training.

**Genuine novelty:**
- Self-attention fusion of LLM and RL embeddings is more architecturally integrated than action-overlay approaches
- Reflective evaluator that reviews risky episodes and updates prompts is a meaningful design
- TTCP (Time-to-Conflict-Point) as structured domain-specific LLM input is good engineering

**Real weaknesses:**
- Teacher-student via KL divergence is standard policy distillation — novelty is incremental
- LLM teacher is **only active for the first 10% of training steps**: is it actually teaching, or just adding early noise diversity?
- No analysis of how attention weights evolve — student may simply learn to ignore teacher embeddings after KL constraint is relaxed
- Highway-Env scenarios: small discrete state spaces with directly applicable LLM driving knowledge; unlikely to generalize
- Real-vehicle validation provides no quantitative comparison against baselines
- No independent ablation of memory vs. reflective evaluator vs. attention mechanism

**Evaluation quality: Moderate.** Training only 100k steps — baselines may not be converged. Right ablation structure but incomplete.

---

### Paper 3: iLLM-TSC — Traffic Signal Control (CUHK-Shenzhen, Jul 2024)

**Core idea:** RL decides first, LLM evaluates and overrides if unreasonable. Targets degraded communication (packet loss, noise) and rare out-of-distribution events (emergency vehicles) not encoded in reward.

**Genuine novelty:**
- Problem framing is the strongest contribution: explicit degraded communication model + long-tail event handling is realistic
- Plug-in architecture requires no modification to existing RL systems

**Real weaknesses:**
- Conceptually, this is just "LLM as rule-based fallback" — the LLM does not learn; contribution is engineering, not research
- LLM inference takes 6+ seconds per step vs. 0.002s for RL — critical failure for real-time control, acknowledged but unresolved
- Missing baseline: a simple rule-based fallback (fixed timing when observation is missing) would likely match the LLM at zero latency cost
- Single intersection, four phases — bare minimum to demonstrate concept
- Prompt ablation shows logically inconsistent LLM decisions under simpler prompts — correctness is entirely prompt-engineering-dependent

**Evaluation quality: Weak.** The 17.5% improvement is against degraded RL, not against a deterministic fallback.

---

### This Project: Where It Genuinely Advances on All Three Papers

**1. Shadow validation as an isolation guarantee (vs. all three)**

ULTRA injects LLM actions directly into training trajectories. TeLL-Drive's KL constraint actively pulls the policy toward LLM-preferred actions. iLLM-TSC overrides RL at inference time. This project validates LLM advisory actions through cloned shadow rollouts before deciding whether to weight the auxiliary loss. The real environment always executes MAPPO. This is the only approach among the four with a formal isolation guarantee.

**2. Candidate turn preview (vs. all three)**

Pre-computing kinematic consequences for all legal heading changes before querying the LLM reduces the LLM's epistemic burden from trajectory prediction to selection among validated options. This substantially reduces hallucination-driven errors. None of the three papers have an equivalent.

**3. Multi-agent joint return weighting (vs. ULTRA and iLLM-TSC)**

The hybrid weight (65% individual, 35% joint) addresses a genuine multi-agent concern. ULTRA and iLLM-TSC are single-agent. TeLL-Drive has multiple vehicles but treats LLM guidance as individual, not joint.

**4. Correctable episodic memory with hard retrieval gates (vs. TeLL-Drive)**

Inter-episode memory correctable by shadow evaluation, with hard gates on route/call/scenario type and explicit safety-first overrides, is more principled than TeLL-Drive's reflective evaluator (which relies on the LLM itself to generate improved policies without ground-truth comparison).

**5. Local model (Gemma 3:12B via Ollama) vs. GPT-4o**

Cost, reproducibility, and data privacy advantages — though with capability trade-offs.

---

### Where This Project Does NOT Yet Improve, or Introduces New Problems

**1. No empirical results exist (most serious problem)**

All three papers have numbers. This project has a well-designed experimental plan and prototype code. Until E0 vs. E1 vs. E2 runs complete across multiple seeds, no improvement claim can be made. Design elegance does not substitute for evidence.

**2. Local LLM (Gemma 3:12B) may be too weak for this task**

All three papers use GPT-4o. Aviation conflict resolution with multi-agent dynamics, weather, and sector boundary constraints is harder reasoning than highway driving or traffic signals. If Gemma 3:12B beats MAPPO fewer than 15–20% of the time in shadow evaluation, the auxiliary loss provides near-zero training signal. Check `train/shadow/llm_beats_mappo_rate` immediately in early training.

**3. 5-step shadow horizon is too short for a 60-step sparse-terminal episode**

Terminal rewards are ±100–120; dense step rewards are ~1–5. In a 5-step shadow window, most episodes will not reach a terminal state. Shadow returns are dominated by small dense signals. The LLM's marginal improvement over 5 steps may not correlate with actual episode outcomes. The 0.05 margin threshold is unjustified — may be too tight (captures no signal) or too loose (captures noise). No hyperparameter search is documented for shadow horizon or margin.

**4. Computational cost may be prohibitive during the guidance window**

Each guided step requires: state clone, 3 parallel shadow branches x 5 steps, Ollama API call (slow with a local 12B model), response parsing, weight computation. Multi-memory attribution multiplies this for each active memory agent. This may limit training throughput substantially during the 100k-step guidance window.

**5. Memory correction is partially implemented**

Memory correction is triggered from within the MAPPO training loop's shadow evaluation. The standalone pure-LLM evaluation path (`evaluate_llm.py`) does not yet trigger corrections. The evaluator engine described in `Memory-readme.md` as "next stage" is not yet complete.

**6. Custom simulator with unpublished data files**

No community baseline exists. Without `22feb_paths.csv`, `path_Waypoints22feb.csv`, `test.geojson`, results cannot be reproduced. There is no published MAPPO baseline for this task to compare against.

**7. Memory similarity weights are unjustified**

25% place, 20% traffic, 15% weather, 15% boundary, 15% preview, 10% misc — stated without ablation or learned calibration. `Memory-readme.md` explicitly acknowledges "similarity is still hand-written."

**8. `LLM_LOSS_WEIGHT = 0.25` may be too large if positive-label rate is high**

If shadow evaluation is permissive, the LLM is essentially doing behavioral cloning into MAPPO, which can cause policy collapse toward LLM-preferred actions even if those actions are only marginally better.

---

### Honest Bottom-Line Assessment

The design is technically the most principled of the four approaches. The shadow validation isolation property is a genuine conceptual advance, and the turn preview design is smart engineering. However:

**Four things must happen before claiming novelty:**

1. **Run E0 and E1 first, fully, with >= 3 seeds.** If shadow-gated LLM labels (E1) do not beat pure MAPPO (E0) statistically, the core claim fails regardless of design elegance.

2. **Measure `llm_beats_mappo_rate` early in training.** If consistently below 15%, the mechanism provides near-zero signal. You need to know this before long runs.

3. **Add the missing baseline all three papers skip:** MAPPO with equivalent wall-clock training time but without LLM calls. If extended MAPPO matches LLM-guided MAPPO, the LLM is not adding value.

4. **Either increase shadow horizon or use a terminal-conditioned return estimate.** 5-step returns in a 60-step sparse-terminal episode are unlikely to be reliable signals for what actually matters.

---

## 11. Key Invariants — Never Violate These

1. `final_actions = policy_actions` — MAPPO actions always go to the real environment
2. Guidance latching happens at episode start and never changes mid-episode
3. LLM auxiliary loss only activates when `llm_return_weight > 0`
4. Memory never overrides current preview safety
5. `real_action_equals_mappo = true` in every audit log row

---

## 12. Debugging Entry Points

```
rl_llm_multi/core.py
  → reset(), step(), _update_hazard_predictions()

rl_llm_multi/llm.py
  → _global_call_decision_for_agent()
  → _preview_rows(), _candidate_sort_key(), _merge_back_candidate_sort_key()
  → GlobalPromptBuilder.build_prompt(), build_vision_prompt()
  → _graph_parse_validate_retry()
  → _graph_guardrail_actions()
  → _graph_fallback_missing()
  → _graph_commit_and_log()

rl_llm_multi/memory.py
  → DecisionMemoryStore.find_similar_cases()
  → _score_similarity()

evaluate_llm.py
  → evaluate_episode()

train.py
  → main training loop, shadow_evaluate_actions()
```

Audit file for checking isolation: `llm_guided_mappo_audit.jsonl` — field `real_action_equals_mappo` must always be `true`.
