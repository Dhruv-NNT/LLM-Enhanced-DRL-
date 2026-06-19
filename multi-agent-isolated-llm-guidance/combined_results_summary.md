# Combined Results Summary: LLM-Only Evaluation + 5% LLM-Guided MAPPO Evaluation

This document combines two related experiments that together test one question:

> Is the real LLM (Gemma 3:12B) adding value over cheaper non-LLM alternatives that use the same shadow-validation / preview pipeline?

The two experiments look at this question from two different angles:

1. **Experiment A — Pure LLM control (no MAPPO):** the controller chooses every action. We compare the real LLM against random and preview-based heuristics.
2. **Experiment B — MAPPO trained with 5% LLM guidance:** MAPPO is trained while the LLM (or a synthetic alternative) provides advisory labels through the shadow-validation framework. We then compare the final trained policies.

---

## Part 1 — Pure LLM Evaluation (no MAPPO)

Source folder: `Evaluation_LLM_only`
Source summary: `llm_only_results_summary.md`

### What was evaluated

The guidance controller was run on its own, with no learned policy. The controller directly chooses the heading-change actions for the aircraft at each guided step.

The purpose was to measure how useful the real LLM guidance is compared with simpler non-LLM guidance sources that use the same controller pipeline (preview rows, guardrails, memory).

Episode outcomes measured:

- `success`: all aircraft reached their destinations.
- `collision`: at least one aircraft pair violated the separation limit.
- `weather`: at least one aircraft entered the terminal weather region.
- `truncated`: the episode reached the 60-step limit before all aircraft finished.

Also recorded: mean reward, mean steps per episode, mean number of guidance calls, wall-clock runtime.

### Methods compared

- `real`: calls the configured Ollama LLM (Gemma 3:12B) and uses the validated LLM-suggested actions.
- `uniform`: skips the LLM and chooses a random allowed turn.
- `preview_safe`: skips the LLM and randomly chooses from preview rows marked safe (with a fallback if no safe row exists).
- `best_preview`: skips the LLM and chooses the top-ranked preview row deterministically.

### Setup

300 episodes per method, base seed 42, 4 agents, 2 weather cells. Episode seeds: 42 through 341 (identical across methods).

### Results

| Method | Success | Collision | Weather | Truncated | Mean reward | Mean steps | Mean guidance calls | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `best_preview` | 63.3% (190/300) | 1.7% (5/300) | 14.0% (42/300) | 21.0% (63/300) | 83.58 | 40.79 | 24.06 | 6.0 h |
| `preview_safe` | 55.3% (166/300) | 0.3% (1/300) | 14.7% (44/300) | 29.7% (89/300) | 65.20 | 42.17 | 24.74 | 6.2 h |
| `real` | 39.0% (117/300) | 1.3% (4/300) | 23.3% (70/300) | 36.3% (109/300) | 26.82 | 39.49 | 23.05 | 22.4 h |
| `uniform` | 4.0% (12/300) | 53.0% (159/300) | 35.3% (106/300) | 7.7% (23/300) | -65.58 | 17.90 | 11.96 | 3.2 h |

### Interpretation

- `best_preview` performed best overall: highest success rate and highest mean reward.
- `preview_safe` was second-best with the lowest collision rate, but more episodes hit the step limit than `best_preview`.
- `real` LLM guidance was better than uniform random, but weaker than both preview-based methods. Its main issue is a high truncation rate — many episodes did not fail immediately, but did not finish in time either.
- `uniform` performed poorly — large number of collisions and weather failures, as expected.
- The real LLM is also roughly 4 times slower than the preview-based methods because of actual Ollama calls.

**Headline finding (Part 1):** when the LLM is the *only* decision-maker, the preview-based heuristics are more reliable than the real LLM. The LLM's reasoning does not improve on a simple deterministic ranker that already has access to the same pre-computed previews.

---

## Part 2 — MAPPO Trained with 5% LLM Guidance

Source folder: `Evaluation_wts_5percent`
Source file used for the new row: `E2_ce_5percent.json`

### What was evaluated

Each method was used to train a MAPPO policy (with the same shadow-validation pipeline and identical hyperparameters). After training, the last 5 checkpoints of each run were each evaluated on 300 episodes (seed 42, 4 agents, 2 weather cells). The mean and range across the 5 checkpoints is reported below — this reduces the snapshot noise that affected earlier single-checkpoint comparisons.

The 5% number refers to the fraction of the training timeline during which LLM (or alternative) guidance was active.

### Success rate table — 5% guidance variants (mean and range across 5 checkpoints x 300 episodes)

| Method | Mean | Range | Spread (max − min) |
|---|---:|---|---:|
| Baseline (no guidance) | 80.13% | 77.3 – 82.0% | 4.7 pts |
| E_random (uniform) | 83.00% | 81.3 – 84.3% | 3.0 pts |
| E_prev_safe | 84.67% | 81.0 – 86.3% | 5.3 pts |
| E_best_prev | 84.27% | 79.0 – 87.3% | 8.3 pts |
| **E2_ce (real LLM)** | **84.87%** | **83.67 – 86.33%** | **2.7 pts** |

### Interpretation

- All four guidance variants — random, prev_safe, best_prev, and real LLM — beat the unguided baseline by roughly 3 to 5 percentage points.
- Among the four guidance variants, the differences are small. They all cluster between 83% and 85%.
- The real LLM produces the highest mean (84.87%) — but only by 0.2 points over E_prev_safe and 0.6 points over E_best_prev. Those gaps are well within the run-to-run noise.
- The real LLM has the *tightest* spread (2.7 pts). It is the most consistent variant across the 5 checkpoints. This is meaningful: even if the average is similar, a more stable training signal is desirable.

### Headline finding (Part 2)

Most of the gain from "LLM-augmented" training comes from the shadow-validation / preview / memory framework, not from the LLM's reasoning. A uniform random action picker, gated through the same framework, captures almost all of the benefit. The real LLM does have one advantage that the heuristics do not: it produces the most stable improvement across checkpoints.

---

## Combined Reading — What Both Parts Together Tell Us

The two experiments rule out two opposite explanations of the original result.

| If only Part 1 had been run | One would conclude | But Part 2 shows |
|---|---|---|
| Pure LLM control loses to best_preview | The LLM is useless | LLM-guided MAPPO still ties or marginally beats the best heuristic |
| | | so the LLM as a *training signal* is not the same as the LLM as a *controller* |

| If only Part 2 had been run | One would conclude | But Part 1 shows |
|---|---|---|
| LLM-guided MAPPO is the best | The LLM is what is helping | Random and preview heuristics inside the same framework do almost as well |
| | | so the framework, not the LLM, is doing most of the work |

Together, the picture is:

1. **The shadow-validation + preview + guardrail framework is the load-bearing contribution.** Even a uniform random picker inside this framework lifts MAPPO success rate by about 3 percentage points over the unguided baseline.
2. **The real LLM does not "understand" the task better than a simple deterministic ranker over the preview rows.** As a pure controller it actually does worse than `best_preview`.
3. **The real LLM still contributes something small but consistent when used as a training signal.** It produces the highest mean and the lowest variance across checkpoints. This is consistent with the LLM acting as a soft regularizer rather than as a teacher.

---

## What This Means For The Research Question

Original question: *Can shadow-validated LLM advisory guidance improve MAPPO training while keeping real environment execution fully MAPPO-controlled?*

The honest answer based on these results:

- **Yes**, MAPPO training is improved when the shadow-validated guidance pipeline is active.
- **Partly no**, the improvement is not specific to the LLM. The LLM is interchangeable with cheap heuristics that use the same pipeline.
- **Partly yes**, the real LLM produces the most stable improvement across checkpoints, even if the mean gain is similar to the heuristic baselines.

This is a defensible, publishable result if reframed carefully. The novelty is the *shadow-validation gated guidance injection framework*, not the LLM itself.

---

## Suggested Next Research Directions

These are listed in increasing order of compute and risk.

### Direction 1 — Statistical confirmation (recommended first, very cheap)

Run paired statistical tests (paired t-test or bootstrap confidence interval) on the per-episode success-rate differences across matched seeds for:

- E2_ce (real LLM) vs E_best_prev
- E2_ce (real LLM) vs E_prev_safe
- E2_ce (real LLM) vs E_random
- Each variant vs Baseline

With 1500 episodes per method (5 checkpoints x 300 eps) this gives tight confidence intervals. The goal is to convert "they look similar" into a statement like "the real LLM is statistically tied with the best heuristic at p > 0.1" or "the real LLM is significantly better at p < 0.05". Without this, all comparisons are visual.

### Direction 2 — Find a setting where the LLM matters

The current 4-agent + 2-weather-cell + same-route-set setup may simply be too narrow for semantic reasoning to dominate. Promising variants to test:

- **Heterogeneous agents** with different priorities or roles (the LLM can reason about role; a deterministic ranker cannot).
- **Out-of-distribution evaluation**: train with the current setup but evaluate on routes / weather configurations / agent counts not seen during training. If the LLM beats the heuristics here, it is because of generalization, not memorization.
- **Higher agent counts (8 – 12)** where coordination semantics matter more than local kinematics.
- **Sparse-event scenarios** (rare emergency events, conflicting priorities) where the preview ranker has no useful signal but a language model with prior knowledge might.

If the real LLM only wins in these harder regimes, that is a clean novel result: "the LLM is useful exactly when the preview heuristic runs out of structure."

### Direction 3 — Reframe the contribution

Rewrite the thesis as:

> A shadow-validation-gated guidance injection framework improves MAPPO training. The choice of guidance generator (LLM vs cheap heuristic vs random) is interchangeable in this benchmark, with the LLM offering modest stability gains at much higher compute cost.

This is publishable as a negative result on LLM reasoning combined with a positive result on the gating framework, with strong controls. It also has a practical implication: cheap heuristics give you almost the same benefit at a fraction of the runtime.

### Direction 4 — Replace shadow-only validation with longer-horizon shadow

Currently shadow rollouts are 5 steps in a 60-step sparse-terminal episode. The terminal rewards dominate episode quality but are invisible to most shadow windows. Either:

- Extend shadow horizon to 15 – 30 steps (compute-heavy).
- Or add a learned value head that estimates terminal return from the shadow's final state.

This may let the LLM's advantage (long-horizon semantic reasoning) actually show up — if it exists.

### Direction 5 — Lower-cost LLM variants

If the LLM is essentially acting as a noise-shaping signal, you can probably drop to a much smaller model (3B class) without losing the stability benefit. Worth testing: 4B / 7B local models, and rule out "we are just paying 22 hours of compute to get a fancy random seed."

---

## My Recommendation

Do Direction 1 first. It is cheap (about a day, no new training) and converts the current observations from visual to statistical. The outcome of Direction 1 should drive the next choice:

- If the LLM is statistically tied with the best heuristic: go to Direction 3 (reframe).
- If the LLM is statistically better but only marginally: go to Direction 2 (find the harder regime).
- If the LLM is statistically worse: go to Direction 5 (cheaper LLM) and re-test in Direction 2.

In all cases the framework itself is the defensible contribution. The role of the LLM is what the next experiment should pin down.
