# Training Flow

## Research Question

Can shadow-validated LLM advisory guidance improve MAPPO training in the multi-agent aircraft sector environment while keeping real environment execution fully MAPPO-controlled?

## Short Answer

The code tests whether an LLM can improve MAPPO as a teacher, not as a real-time controller. MAPPO still chooses every real action sent to the environment. When LLM guidance is enabled, the LLM proposes advisory actions, those actions are checked in short cloned shadow rollouts, and only shadow-validated labels are allowed to influence the policy through a weighted auxiliary imitation loss. Memory is also updated cautiously: new advisory memory is written only when the LLM beats MAPPO for that agent, and old memory is corrected only when shadow evidence shows it is worse than a better MAPPO or LLM branch.

## Whole Flow In One Pass

At the start of each episode, the trainer checks whether the current `agent_steps` count is inside the configured LLM guidance window. This decision is latched for the whole episode. If an episode starts before the LLM window, LLM guidance stays off until the episode ends. If an episode starts inside the window, LLM guidance stays on until the episode ends. This prevents guidance from suddenly starting or stopping in the middle of an episode.

During each active environment step, MAPPO first samples actions for all active aircraft. These sampled MAPPO actions are the real actions for the environment. If LLM guidance is active for the episode, the controller asks the LLM for advisory labels for eligible aircraft. The code does not immediately trust these labels. Instead, it copies the simulator state and runs short shadow rollouts comparing three first-step alternatives: the MAPPO action, the LLM action, and, when available, a retrieved memory action. After the first shadow step, all branches continue with deterministic MAPPO actions.

The shadow returns decide how much the LLM label should matter. A label receives weight only when the LLM branch beats the MAPPO branch by the configured return margin. The final label weight combines two signals: how much the LLM helped the individual aircraft and how much it helped the tracked group jointly. That weighted label is stored in the rollout buffer as metadata attached to the original MAPPO transition.

Memory is handled separately. If the LLM action beats MAPPO for a specific eligible agent, the code may write that LLM action into advisory memory. If a retrieved memory action was used and shadow evaluation shows that memory was worse than a better MAPPO or LLM branch, the memory case is corrected by filling `result.corrected` and `result.note`. The original stored action remains in `result.action`.

Finally, the real environment steps using MAPPO's original sampled actions. The LLM action is not executed in the real episode. During PPO update, MAPPO learns from the normal PPO objective, plus a small auxiliary supervised loss toward useful LLM labels. The auxiliary loss is multiplied by the return-based label weight and by a phase-specific weight, so LLM guidance has the strongest effect only when shadow evaluation says it is useful.

## Core Guarantees

- Real environment execution is MAPPO-only.
- LLM guidance is episode-aligned, not switched mid-episode.
- LLM labels affect training only through an auxiliary loss.
- LLM labels receive positive weight only when shadow rollouts show improvement over MAPPO.
- Advisory memory writes happen only for eligible agents whose LLM action beats MAPPO.
- Memory correction is conservative and does not overwrite the original stored action.
- Audit logs and TensorBoard metrics are available to verify each of these claims.

# Detailed Training Stages

## 1. Episode Start And Guidance Latching

At episode reset, the trainer computes:

```text
episode_llm_guidance_active = (
    USE_LLM_GUIDED_TRAINING
    and LLM_GUIDANCE_START_STEP <= agent_steps_at_episode_start < LLM_GUIDANCE_END_STEP
)
```

Current defaults:

```text
USE_LLM_GUIDED_TRAINING = False
LLM_GUIDANCE_START_STEP = 0
LLM_GUIDANCE_END_STEP = 100000
```

If `episode_llm_guidance_active` is true, LLM guidance is available for the entire episode. If it is false, LLM guidance is unavailable for the entire episode.

## 2. MAPPO Action Selection

For each active environment step:

- The trainer reads active aircraft from `env.agents`.
- MAPPO samples actions with `agent.select_actions(...)`.
- The sampled MAPPO action records contain local observation, global state, agent id, sampled action, log probability, and value estimate.
- These sampled MAPPO actions are saved as `policy_actions`.

This happens before any LLM guidance. MAPPO is always the source of real environment actions.

## 3. LLM Advisory Label Generation

If guidance is active for the episode, the trainer calls:

```text
guidance.advice_for_step(...)
```

The guidance provider returns per-agent `GuidanceAdvice` objects containing:

- LLM action in heading degrees.
- LLM action index.
- Guidance phase, such as `EXECUTE_TURN`, `EMERGENCY_MANEUVER`, or `MERGE_BACK`.
- Requested action before normalization, when available.
- Source and fallback metadata.
- Retrieved memory action and memory reference, when memory was used.

The controller also stores a normalized payload used for audit and memory writing:

```text
debug.eligible_agent_ids
debug.stage_by_agent
debug.call_reason_by_agent
threat_rows_by_agent
preview_rows_by_agent
normalized final LLM actions
```

## 4. Shadow Evaluation

If LLM advice exists and shadow evaluation is enabled, the trainer runs cloned short-horizon rollouts.

Current shadow settings:

```text
LLM_SHADOW_EVAL_ENABLED = True
LLM_SHADOW_HORIZON = 5
LLM_SHADOW_GAMMA = 0.99
```

The shadow evaluator compares:

- MAPPO branch: every aircraft uses the MAPPO first-step action.
- LLM branch: advised aircraft use the LLM first-step action; others use MAPPO.
- Memory branch: aircraft with retrieved memory use the memory first-step action; others use MAPPO.

After the forced first step, later shadow steps use deterministic MAPPO. The evaluator reports per-agent and joint returns:

```text
G_MAPPO_i
G_LLM_i
G_MEMORY_i
G_MAPPO_joint = mean(G_MAPPO_i over tracked agents)
G_LLM_joint = mean(G_LLM_i over tracked agents)
G_MEMORY_joint = mean(G_MEMORY_i over tracked agents)
```

## 5. Multi-Memory Attribution

If exactly one memory action was used, the code uses the normal joint memory branch to decide whether that memory case should be corrected.

If multiple memory actions were used, the code runs one-at-a-time attribution rollouts. For each target memory agent:

- The target agent uses its memory action on the first step.
- Every other aircraft uses its MAPPO first-step action.
- Later steps use deterministic MAPPO.

This avoids blaming every retrieved memory case when only one memory action caused poor joint behavior.

## 6. Hybrid LLM Return Weight

Each LLM label receives a return-based weight. The current implementation uses the raw improvement after passing a margin test.

Per-agent label weight:

```text
delta_agent_i = G_LLM_i - G_MAPPO_i

if delta_agent_i <= LLM_SHADOW_RETURN_MARGIN:
    w_agent_i = 0
else:
    w_agent_i = clip(delta_agent_i / LLM_SHADOW_RETURN_SCALE)
```

Joint label weight:

```text
delta_joint = G_LLM_joint - G_MAPPO_joint

if delta_joint <= LLM_SHADOW_RETURN_MARGIN:
    w_joint = 0
else:
    w_joint = clip(delta_joint / LLM_SHADOW_RETURN_SCALE)
```

Current settings:

```text
LLM_SHADOW_RETURN_MARGIN = 0.05
LLM_SHADOW_RETURN_SCALE = 1.0
LLM_SHADOW_RETURN_WEIGHT_CLIP = 1.0
LLM_SHADOW_MAX_WEIGHT = 1.0
LLM_RETURN_WEIGHT_AGENT_COEF = 0.65
LLM_RETURN_WEIGHT_JOINT_COEF = 0.35
```

Final hybrid weight:

```text
llm_return_weight_i = clip(0.65 * w_agent_i + 0.35 * w_joint)
```

This means the label is strongest when it helps both the individual aircraft and the team-level joint return.

## 7. Advisory Memory Writes

After shadow evaluation, the trainer can write new advisory memory cases.

Current settings:

```text
LLM_ADVISORY_MEMORY_WRITE_ENABLED = True
LLM_ADVISORY_MEMORY_WRITE_MARGIN = LLM_SHADOW_RETURN_MARGIN
LLM_GUIDED_MEMORY_MAX_EPISODES = 300
```

A new advisory memory case is written for agent `i` only if:

```text
G_LLM_i > G_MAPPO_i + LLM_ADVISORY_MEMORY_WRITE_MARGIN
```

Additional conditions:

- If `debug.eligible_agent_ids` is present, the agent must be in that list.
- The stored action is the normalized final LLM action.
- The stored action goes into `result.action`.
- The guided MAPPO memory store keeps only the latest 300 episodes.

## 8. Cautious Memory Correction

If retrieved memory was used, the trainer checks whether that memory case should be corrected.

Current settings:

```text
LLM_MEMORY_EVALUATOR_ENABLED = True
LLM_MEMORY_UPDATE_MARGIN = 0.10
LLM_MEMORY_UPDATE_WITH_BEST_OF = "mappo_or_llm"
```

Correction source selection:

- `llm`: correct toward the LLM action.
- `mappo`: correct toward the MAPPO action.
- `mappo_or_llm`: correct toward whichever joint branch is better.

A correction is applied only if:

```text
best_return > memory_compare_return + LLM_MEMORY_UPDATE_MARGIN
```

The memory schema is:

```json
{
  "result": {
    "action": "<original stored action>",
    "corrected": "<better action>",
    "note": "<short reason>"
  }
}
```

Correction does not overwrite `result.action`. It fills only `result.corrected` and `result.note`.

## 9. LLM-Generated Correction Notes

When a memory correction is applied, the trainer asks the LLM for a concise note explaining why the corrected action is better.

The note prompt includes:

- agent id
- phase
- call reason
- original memory action
- corrected action
- MAPPO, LLM, and memory shadow returns
- preview summaries for the original and corrected actions, when available

Expected response:

```json
{"note": "<concise reason>"}
```

If the note call fails, the correction still happens and a deterministic fallback note is stored.

## 10. Real Environment Step

After all advisory logic is complete, the trainer sets:

```text
final_actions = policy_actions
```

Then it calls:

```text
env.step(final_actions)
```

This is the main isolation property. LLM actions do not directly control the real environment.

## 11. Buffer Storage

After the real MAPPO-only environment step:

- The trainer reads rewards and terminal flags.
- MAPPO records are stored in the rollout buffer.
- LLM metadata is attached to the corresponding agent record when available.

Important metadata fields include:

```text
llm_label_available
llm_action_idx
llm_action_deg
llm_phase
G_MAPPO_shadow
G_LLM_shadow
G_MEMORY_shadow
G_MAPPO_joint_shadow
G_LLM_joint_shadow
G_MEMORY_joint_shadow
llm_return_weight_agent
llm_return_weight_joint
llm_return_weight_hybrid
llm_return_weight
memory_action_idx
memory_action_deg
memory_ref
memory_update_applied
memory_attribution_mode
memory_correction_note_source
advisory_memory_written
```

# Losses And Weights

## MAPPO PPO Loss

MAPPO uses the standard clipped PPO objective:

```text
ratio = exp(new_logprob - old_logprob)
surr1 = ratio * advantage
surr2 = clip(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
policy_loss = -mean(min(surr1, surr2))
```

The value and entropy terms are:

```text
value_loss = MSE(value_prediction, return)
entropy_bonus = entropy_mean
```

The base MAPPO loss is:

```text
L_MAPPO = policy_loss + 0.5 * value_loss - 0.01 * entropy_mean
```

## LLM Auxiliary Imitation Loss

The LLM auxiliary loss is used only when all of these are true:

```text
LLM_LOSS_ENABLED = True
current_llm_loss_weight > 0
llm_label_available = True
llm_return_weight > 0
```

Current settings:

```text
LLM_LOSS_ENABLED = True
LLM_LOSS_TYPE = "ce"
LLM_LOSS_WEIGHT = 0.25
LLM_LOSS_DECAY_ENABLED = False
LLM_LOSS_MIN_WEIGHT = 0.05
LLM_SOFT_KL_SIGMA_DEG = 5.0
```

Current phase weights:

```text
EXECUTE_TURN = 0.5
EMERGENCY_MANEUVER = 1.0
MERGE_BACK = 0.6
```

For cross-entropy mode:

```text
CE_i = -log pi_theta(llm_action_i | observation_i)
effective_weight_i = llm_return_weight_i * phase_weight_i
L_LLM = mean(CE_i * effective_weight_i)
```

Total loss:

```text
L_total = L_MAPPO + current_llm_loss_weight * L_LLM
```

With the current defaults:

```text
current_llm_loss_weight = 0.25
```

If `LLM_LOSS_TYPE = "soft_kl"`, the code uses a soft target distribution centered on the LLM action instead of a hard CE target.

# Run Directory Layout

Each experiment should use a unique `--log-dir`. That directory is treated as the run root. For example:

```bash
python train.py --log-dir runs/E2_full_llm_seed0 --seed 0
```

The trainer derives the main artifact paths from that run root:

```text
runs/E2_full_llm_seed0/
  run_hyperparameters.txt
  tensorboard/
    E2_full_llm_seed0/
  weights/
    best_model.pt
    last_checkpoint.pt
    checkpoint_<agent_steps>.pt
  memory/
    decision_memory.json
  llm_guidance_json/
  llm_guided_mappo_audit.jsonl
```

`run_hyperparameters.txt` is created once at the beginning of training. It records the command-line arguments, resolved run paths, reset options, and config hyperparameters, including all reward coefficients.

TensorBoard event files are written under `tensorboard/<run-name>/`, where `<run-name>` is derived from the final folder name in `--log-dir`. For example, `--log-dir runs/E2_full_llm_seed0` writes events under `runs/E2_full_llm_seed0/tensorboard/E2_full_llm_seed0/`.

`llm_guidance_json/` contains detailed LLM call artifacts. It is for debugging individual LLM guidance decisions: prompts, raw responses, normalized JSON payloads, candidate previews, threat rows, and debug metadata. This folder can become large, so it is controlled by:

```text
LLM_GUIDANCE_ARTIFACTS_ENABLED = False
LLM_GUIDANCE_ARTIFACT_MAX_CALLS = 2
```

When disabled, the trainer still keeps the latest normalized LLM payload in memory for training and memory writes, but it does not write prompt/response/normalized files to disk. When enabled, only the latest configured number of detailed calls are kept.

`llm_guided_mappo_audit.jsonl` is the compact per-guided-decision audit file. Each line records one guided decision with MAPPO action, LLM label, shadow returns, hybrid weights, memory write/correction flags, and whether the real executed action stayed MAPPO. It is controlled by:

```text
LLM_GUIDED_MAPPO_AUDIT_ENABLED = True
LLM_GUIDED_MAPPO_AUDIT_MAX_ROWS = 5000
```

This file is better for analysis scripts than the full JSON artifacts, and the cap keeps only the latest rows.

This keeps TensorBoard logs, checkpoints, LLM call artifacts, audit logs, hyperparameters, and guided memory separated across experiments. The optional `--best-model-path` and `--last-ckpt-path` flags still exist as overrides, but normal experiment runs should not need them.

# Logging And Validation Signals

The audit JSONL records per-guided-decision fields such as:

- MAPPO action and real executed action.
- Whether the real action equals MAPPO.
- LLM label action.
- Per-agent and joint shadow returns.
- Agent, joint, and hybrid LLM weights.
- Memory reference and advisory write flag.
- Memory correction source, attribution mode, and note source.

The most important sanity check is:

```text
real_action_equals_mappo = true
```

TensorBoard metrics include the core episode and reward meters:

```text
train/reward/episode_total_return
train/reward/policy_controlled_return
train/reward/pre_sector_return
train/episode_env_steps
train/reward/terminal_episode
train/reward/mean_step_behavior_reward
train/reward_component/cross_track_mean
train/reward_component/finish_bonus_mean
train/success
train/collision
train/weather_failure
train/truncated
train/success_rate_running
train/collision_or_weather_rate_running
train/truncation_rate_running
```

It also logs LLM, shadow, and memory meters:

```text
train/llm/enabled
train/llm/num_guided_decisions
train/llm/positive_label_count
train/llm/action_match_rate
train/llm/return_weight_agent_mean
train/llm/return_weight_joint_mean
train/llm/return_weight_hybrid_mean
train/shadow/G_MAPPO_mean
train/shadow/G_LLM_mean
train/shadow/G_MEMORY_mean
train/shadow/G_MAPPO_joint_mean
train/shadow/G_LLM_joint_mean
train/shadow/G_MEMORY_joint_mean
train/shadow/llm_beats_mappo_rate
train/shadow/memory_beats_mappo_rate
train/shadow/mappo_beats_memory_rate
train/memory/retrieval_count
train/memory/update_count
train/memory/corrected_to_mappo_count
train/memory/corrected_to_llm_count
train/memory/attribution_skipped_count
train/memory/correction_note_llm_count
train/memory/correction_note_fallback_count
train/memory/advisory_write_count
```

# Suggested Experiments

## Common Setup

Use the same MAPPO and environment settings across conditions. Change only the listed LLM guidance settings.

Recommended seeds:

```text
pilot_seeds = [0, 1, 2]
preferred_full_seeds = [0, 1, 2, 3, 4]
```

Common LLM settings unless overridden:

```text
LLM_GUIDANCE_START_STEP = 0
LLM_GUIDANCE_END_STEP = 100000
LLM_SHADOW_HORIZON = 5
LLM_SHADOW_GAMMA = 0.99
LLM_SHADOW_RETURN_SCALE = 1.0
LLM_SHADOW_RETURN_WEIGHT_CLIP = 1.0
LLM_SHADOW_MAX_WEIGHT = 1.0
LLM_LOSS_TYPE = "ce"
LLM_LOSS_WEIGHT = 0.25
LLM_LOSS_DECAY_ENABLED = False
LLM_LOSS_MIN_WEIGHT = 0.05
LLM_PHASE_WEIGHTS = {
    "EXECUTE_TURN": 0.5,
    "EMERGENCY_MANEUVER": 1.0,
    "MERGE_BACK": 0.6,
}
```

Primary metrics:

- Evaluation total episode reward.
- Policy-controlled return.
- Success rate, if available.
- Loss-of-separation or conflict count.
- Weather-entry count.
- Boundary-exit count.
- LLM positive-label rate.
- LLM beats MAPPO rate.
- Hybrid return-weight mean.
- Advisory memory write count.
- Memory correction count and source split.

## E0: MAPPO Baseline

Purpose: establish the no-LLM baseline.

```text
USE_LLM_GUIDED_TRAINING = False
```

## E1: LLM Labels With Shadow Weighting, No Memory

Purpose: test whether shadow-validated LLM labels help without memory.

```text
USE_LLM_GUIDED_TRAINING = True
LLM_SHADOW_EVAL_ENABLED = True
LLM_SHADOW_RETURN_MARGIN = 0.05
LLM_RETURN_WEIGHT_AGENT_COEF = 0.65
LLM_RETURN_WEIGHT_JOINT_COEF = 0.35
LLM_ADVISORY_MEMORY_WRITE_ENABLED = False
LLM_MEMORY_EVALUATOR_ENABLED = False
LLM_LOSS_ENABLED = True
LLM_LOSS_WEIGHT = 0.25
LLM_LOSS_TYPE = "ce"
```

Interpretation: if E1 beats E0, the LLM label signal is useful even without memory.

## E2: Full Current Method

Purpose: test the full hybrid LLM plus memory method.

```text
USE_LLM_GUIDED_TRAINING = True
LLM_SHADOW_EVAL_ENABLED = True
LLM_SHADOW_RETURN_MARGIN = 0.05
LLM_RETURN_WEIGHT_AGENT_COEF = 0.65
LLM_RETURN_WEIGHT_JOINT_COEF = 0.35
LLM_ADVISORY_MEMORY_WRITE_ENABLED = True
LLM_ADVISORY_MEMORY_WRITE_MARGIN = 0.05
LLM_GUIDED_MEMORY_MAX_EPISODES = 300
LLM_MEMORY_EVALUATOR_ENABLED = True
LLM_MEMORY_UPDATE_MARGIN = 0.10
LLM_MEMORY_UPDATE_WITH_BEST_OF = "mappo_or_llm"
LLM_LOSS_ENABLED = True
LLM_LOSS_WEIGHT = 0.25
LLM_LOSS_TYPE = "ce"
```

Interpretation: if E2 beats E1, guided memory is adding value.

## E3: Per-Agent Weight Only

Purpose: test whether individual-agent improvement is enough.

Use E2 settings except:

```text
LLM_RETURN_WEIGHT_AGENT_COEF = 1.0
LLM_RETURN_WEIGHT_JOINT_COEF = 0.0
```

Interpretation: if E3 beats E2, the joint term may be too noisy or restrictive.

## E4: Joint Weight Only

Purpose: test whether team-level improvement alone is enough.

Use E2 settings except:

```text
LLM_RETURN_WEIGHT_AGENT_COEF = 0.0
LLM_RETURN_WEIGHT_JOINT_COEF = 1.0
```

Interpretation: if E4 performs poorly, individual-agent attribution is needed.

## E5: Full Method Without Memory Correction

Purpose: separate memory writing from memory correction.

Use E2 settings except:

```text
LLM_ADVISORY_MEMORY_WRITE_ENABLED = True
LLM_MEMORY_EVALUATOR_ENABLED = False
```

Interpretation: if E5 beats E2, memory correction may be too aggressive. If E2 beats E5, correction is probably improving memory quality.

## E6: Margin Sensitivity

Purpose: test how conservative shadow filtering should be.

Run E6 as three variants using E2 as the base.

Loose margins:

```text
LLM_SHADOW_RETURN_MARGIN = 0.00
LLM_ADVISORY_MEMORY_WRITE_MARGIN = 0.00
LLM_MEMORY_UPDATE_MARGIN = 0.05
```

Default margins:

```text
LLM_SHADOW_RETURN_MARGIN = 0.05
LLM_ADVISORY_MEMORY_WRITE_MARGIN = 0.05
LLM_MEMORY_UPDATE_MARGIN = 0.10
```

Strict margins:

```text
LLM_SHADOW_RETURN_MARGIN = 0.10
LLM_ADVISORY_MEMORY_WRITE_MARGIN = 0.10
LLM_MEMORY_UPDATE_MARGIN = 0.20
```

Interpretation: loose margins produce more LLM signal but more noise; strict margins are safer but may produce too few positive labels.

## Recommended Run Order

1. E0: MAPPO baseline.
2. E1: LLM labels with shadow weighting, no memory.
3. E2: full current method.
4. E3: per-agent weight only.
5. E5: full method without memory correction.
6. E6: margin sensitivity.
7. E4: joint-only weighting, if E2/E3 suggests the joint term needs deeper analysis.

## Decision Rules

- If E1 beats E0, LLM labels are useful.
- If E2 beats E1, guided memory is useful.
- If E1 beats E2, memory should be made simpler or more conservative.
- If E2 beats E3, hybrid joint weighting is useful.
- If E3 beats E2, per-agent weighting should be preferred.
- If positive-label count is near zero, the margin is too strict or the LLM rarely beats MAPPO.
- If positive-label count is high but reward drops, the labels may be noisy or `LLM_LOSS_WEIGHT` may be too high.
- If memory correction count is high and performance drops, correction may be too aggressive.
