# Combined Results Summary: LLM-Only Evaluation + 5% LLM-Guided MAPPO Evaluation

This document combines two related experiments that together test one question:

> Is the real LLM (Gemma 3:12B) adding value over cheaper non-LLM alternatives that use the same shadow-validation / preview pipeline?

The two experiments look at this question from two different angles:

1. **Experiment A — Pure LLM control (no MAPPO):** the controller chooses every action. We compare the real LLM against random and preview-based heuristics.
2. **Experiment B — MAPPO trained with 5% LLM guidance:** MAPPO is trained while the LLM (or a synthetic alternative) provides advisory labels through the shadow-validation framework. We then compare the final trained policies.

---

## Part 1 — Pure LLM Evaluation (no MAPPO)

Source folder: `Evaluation_old_approach/Evaluation_LLM_only`
Source summary: `archive_docs/llm_only_results_summary.md`

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

Source folder: `Evaluation_old_approach/Evaluation_wts_5percent`
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

---

# Three-Teacher Complementary Distillation Study (gpt-oss-20B)

This section documents the *next* research direction, decided after the Gemma study
above. Two things changed since Part 1/Part 2:

1. **The LLM was upgraded** from Gemma 3:12B to the much stronger **gpt-oss-20B**.
   As a pure controller it now clearly beats the heuristics on average (R1 below),
   which threatened the premise that a cheap heuristic is worth keeping.
2. **A per-decision diagnostic (R3) showed the threat is unfounded:** even though
   the strong LLM is the best *overall* guide, it is *not* the best guide at every
   individual decision. A cheap heuristic is the single best choice in a meaningful
   minority of states, concentrated in safety-critical (traffic / weather) moments.

This motivates a shift from a single teacher to **competence-weighted distillation
from multiple complementary teachers**: an LLM teacher plus one or two cheap
simulator-based heuristic teachers (`best_preview`, `preview_safe`), where the
student trusts whichever teacher is more reliable *in that state*. The student
(MAPPO) always remains the only thing that acts in the real environment.

## Direction-finding evidence already collected

All numbers below are factual outputs from the direction-finding runs (R1/R2/R3).

**R1 — controller comparison, easy (4 aircraft, 100 episodes):** success / collision / weather / truncation (%)

| Controller | Success | Coll | Weather | Trunc |
|---|---:|---:|---:|---:|
| gpt-oss-20B | 76 | 1 | 12 | 11 |
| best_preview (heuristic) | 64 | 2 | 11 | 23 |
| preview_safe (heuristic) | 54 | 1 | 15 | 30 |

**R2 — controller comparison, hard (8 aircraft, 60 episodes):**

| Controller | Success | Coll | Weather | Trunc |
|---|---:|---:|---:|---:|
| gpt-oss-20B | 18 | 3 | 57 | 22 |
| best_preview (heuristic) | 15 | 3 | 50 | 32 |
| preview_safe (heuristic) | 13 | 2 | 57 | 28 |

In the hard regime every controller collapses (success < 20%, weather failures
dominate), but the heuristics are not worse on safety (best_preview has the fewest
weather failures, preview_safe the fewest collisions).

**R3 — per-decision diagnostic (`direction_finding/analyze_complementarity.py`).** At ~2,750 decision
states a fixed reference MAPPO policy is queried, then each guide's suggested action
is shadow-rolled 5 steps and scored. "% of states" = % of those individual decision
moments. Share of decisions where each guide is the single best of all four (the
strictest test — beating every other guide including the second heuristic):

| Single best guide | Easy (2752 states) | Hard (2746 states) |
|---|---:|---:|
| MAPPO (student — already best) | 40% | 40% |
| LLM | 38% | 39% |
| best_preview | 15% | 10% |
| preview_safe | 8% | 11% |

Reading: in ~40% of states the student is already best (no teacher needed); in the
remaining ~60% a heuristic is the single best in ~23% and the LLM in ~38%. Crucially
**both heuristics hold a distinct, non-trivial slice** — preview_safe is outright best
(beating best_preview, the LLM, and MAPPO) in 8–11% of states — so they are
complementary to *each other*, not redundant. This is the diagnostic pre-check that
justifies investigating a three-teacher (rather than two-teacher) study.

## Ladder of claims

"No single teacher is best" is really four nested claims; each experiment maps to one:

1. **Per-decision (diagnostic):** no single teacher is the best advisor in all
   situations. *(Supported by R3 above.)*
2. **Trained outcome:** a combination of teachers produces a better trained policy
   than *any* single teacher.
3. **Mechanism:** the gain comes from *competence weighting*, not merely from
   averaging several teachers.
4. **Marginal value:** inside the winning combination, *every* teacher pulls its
   weight — drop any one and the trained result gets worse. (Strongest, most direct
   evidence; requires the all-three-teacher runs.)

## Experiment matrix (three teachers: LLM, best_preview, preview_safe)

The three "leave-one-out" runs are exactly the three pairwise combinations, so the
pairs come for free. All runs train a *separate* MAPPO policy and differ only in
which teachers are active and how they are weighted; evaluate each with
`evaluate_rl.py` and compare success rate (paired by seed).

| Run | Teachers used | Weighting | Tests |
|---|---|---|---|
| **T0** | none (baseline) | — | reference point |
| **S-LLM** | LLM only | shadow | single-teacher baseline (prior work) |
| **S-BP** | best_preview only | shadow | single-teacher baseline |
| **S-PS** | preview_safe only | shadow | single-teacher baseline |
| **TRI-equal** | all three | equal | Claim 3 (is weighting needed?) |
| **TRI-shadow** ⭐ | all three | shadow | the proposed method |
| **LOO−LLM** | best_preview + preview_safe | shadow | Claim 4: marginal value of LLM |
| **LOO−BP** | LLM + preview_safe | shadow | Claim 4: marginal value of best_preview |
| **LOO−PS** | LLM + best_preview | shadow | Claim 4: marginal value of preview_safe |

How the comparisons read:

- **TRI-shadow beats S-LLM, S-BP, S-PS** → no single teacher is enough (Claims 1–2).
- **TRI-shadow beats TRI-equal** → it is the competence weighting, not just
  ensembling (Claim 3).
- **TRI-shadow beats each LOO run** → each teacher contributes something the others
  cannot supply (Claim 4). If, e.g., TRI-shadow ≈ LOO−PS, that is honest evidence
  preview_safe is redundant and the method drops back to two teachers.

Suggested sequencing: (1) diagnostic pre-check — already done, see R3; (2) pilot —
TRI-shadow vs S-LLM vs LOO−LLM, seeds [0,1,2], short steps; (3) full matrix, seeds
[0,1,2,3,4], full steps. Full matrix = 9 configs × 5 seeds = 45 runs, so the tiering
matters.

## Code-readiness audit (as of this study)

The distillation pipeline is currently built for **exactly two teachers** (`llm` +
`heur`). The trajectory-generation and shadow-evaluation cores are already generic;
the buffer, loss, metadata, loader and config are hard-coded to two named slots.

| Component | File | Ready for 3? | Note |
|---|---|---|---|
| Trajectory generation | `new_approach/generate_guided_trajectories.py` | yes | generic via `--guidance-source` |
| Teacher training | `new_approach/train_teachers.py` | yes | explicit `--dataset-dir/--out-path/--name` trains any teacher (`--both` is the only hardwired-to-2 path) |
| Shadow evaluation | `guidance.py::shadow_evaluate_teachers` | yes | already loops an arbitrary teacher dict |
| Teacher loading | `distill.py::load_distill_teachers` | no | hardwired to `("llm","heur")`; loads both unconditionally |
| Metadata build | `distill.py::build_distill_metadata` | no | fixed `llm`/`heur` keys and 2-tuple weights |
| Buffer storage | `mappo.py::RolloutBuffer` | no | fixed `teacher_llm_*` / `teacher_heur_*` fields |
| Loss loop | `mappo.py::MAPPO.update` | no | `teacher_specs` is a literal 2-tuple; metrics per fixed name |
| Config | `configs.py` | no | only two teacher paths; `DISTILL_TEACHERS` fixed to `{llm,heur}` |

**Key nuance:** the two slots are labelled `"llm"`/`"heur"` but, for the `shadow` and
`equal` competence modes, those labels are cosmetic — any teacher can occupy either
slot (only `weather_rule` mode is semantically asymmetric). Consequences:

- **Runnable today, no code changes** (with `shadow` or `equal` weighting): **T0, all
  three singles (S-LLM/S-BP/S-PS), and all three pairs (LOO−LLM/LOO−BP/LOO−PS)** — 7
  of the 9 runs. For the BP+PS pair, put one heuristic in each slot and use shadow/equal.
- **Blocked, needs code changes:** **TRI-equal and TRI-shadow** (all three teachers
  at once). A genuine third slot must be threaded end-to-end.

So Claims 1–3 are testable with the current code; **Claim 4 (the full leave-one-out
story) requires the three-teacher generalization.**

### Three-teacher generalization — change map (IMPLEMENTED)

Status: done. The distillation pipeline now supports an arbitrary set of named
teachers (1..N); all 20 distillation unit tests pass and a three-teacher
`MAPPO.update()` runs end-to-end (teacher columns aligned, competence-weighted
JS loss finite, per-teacher weight metrics logged, gradients flow). To run the
three-teacher study set, e.g., `DISTILL_TEACHERS = ("llm", "best_preview",
"preview_safe")` and register the checkpoints in `DISTILL_TEACHER_PATHS`
(`teacher_best_preview.pt`, `teacher_preview_safe.pt`). The change touched:

1. **`configs.py`** — replace the two fixed paths with a registry
   `DISTILL_TEACHER_PATHS = {name: path, ...}`; `DISTILL_TEACHERS` becomes the active
   subset.
2. **`distill.py::load_distill_teachers`** — load only the teachers named in
   `DISTILL_TEACHERS` from the registry (also fixes the "must exist even if unused"
   wart that currently forces a placeholder `teacher_llm.pt`).
3. **`distill.py::build_distill_metadata`** — build per-teacher action/probs in a loop
   over active names; shadow returns one weight per teacher; emit metadata as dicts
   keyed by teacher name.
4. **`mappo.py::RolloutBuffer`** — store teacher probs/actions/weights as name-keyed
   dicts of lists instead of fixed `llm`/`heur` fields.
5. **`mappo.py::MAPPO.update`** — build `teacher_specs` by iterating active teacher
   names; `contrib_mask` = OR over all teacher weights; per-teacher metrics keyed by
   name.
6. **`distill.py` weather_rule** — generalize the 2-tuple to a per-teacher weight
   (only needed if `weather_rule` is used with 3 teachers; `shadow`/`equal` are already
   general once the plumbing is dict-based).
7. **`tests/test_distill.py`** — extend teacher-gate and buffer tests to N teachers.

Each run is now fully specified on the command line via `train.py` flags
(`--distill` / `--no-distill`, `--distill-teachers`, `--competence-mode`), so
concurrent runs in separate tmux sessions never race on `configs.py`.

## Runbook — Commands (CPU, Ollama-free first)

This runbook runs every experiment that does **not** need Ollama first, on CPU.
Only the LLM teacher's trajectory generation needs Ollama; once that teacher
checkpoint exists, even the LLM runs are Ollama-free. The MAPPO training itself
never calls Ollama in distillation mode.

### One-time setup

Force CPU everywhere. The machine has GPUs and PyTorch sees them, and `train.py`
otherwise overrides the shell's `CUDA_VISIBLE_DEVICES` from the config, so you
need both of these:

1. In `configs.py` set `MAPPO_CUDA_VISIBLE_DEVICES = None` (so `train.py` stops
   overriding and respects the shell). This makes the shell line below the single
   control for every script.
2. In every tmux session, set the shell line:

```bash
export CUDA_VISIBLE_DEVICES=""        # "" = CPU; later set to a free GPU id, e.g. "3"
PY=/home/aradhya.dhruv/anaconda3/envs/llmdrl/bin/python
cd "/home/aradhya.dhruv/rl/behaviour_cloning/LLM enhanced DRL/LLM-DRL restructured code/multi-agent-isolated-llm-guidance"
```

To switch to a GPU later, leave the config as `None` and just change the shell
line to a free GPU id (e.g. `export CUDA_VISIBLE_DEVICES="3"`) — no config edits.

CPU note: MAPPO training is slow on CPU. Phase A (data + teachers) is cheap and
finishes quickly. For the Phase B training runs, use a reduced `STEPS` for a CPU
pilot (the distillation window ends at 250k, so `STEPS=300000` covers it; drop to
`STEPS=60000` for a faster first signal). Final-quality numbers will want more
steps / a GPU.

### Recommended order (one experiment per tmux session)

**Phase A — offline, Ollama-free (datasets + heuristic teachers).** A1 and A2 can
run concurrently; A3/A4 depend on them.

Notes on the generation settings:
- `--keep success` keeps only episodes where all aircraft reached their
  destination — a clean, outcome-validated distillation target.
- `--n-episodes` / `--total-episodes` is now a **target number of KEPT episodes**,
  not games played: the generator keeps playing new games (fresh seeds) until it
  has collected that many successful episodes. Default target is **6000**
  successful episodes (~200k+ records — ample for the small teacher).
- Generation is CPU-simulator-bound, so use **`new_approach/generate_parallel.py`** to split the
  target across many worker processes. Each worker gets a disjoint seed range and
  writes its own `part_XX/` subdirectory; the teacher loader reads the parent
  directory recursively and merges the parts automatically (no merge step). The
  launcher also auto-pins each worker to one math thread (OMP/BLAS), so workers map
  cleanly to cores instead of oversubscribing — no manual `OMP_NUM_THREADS` needed.
- Choose `--workers` from **physical** cores, not the inflated logical count
  (`lscpu`: cores/socket × sockets = 32 here). **This box is shared, so use ~16
  workers** (about half the cores) to leave room for co-tenants; bump toward 30
  only when the machine is idle (`uptime` / `htop` to check). `--math-threads N`
  overrides the 1-thread pin if you run few workers.
- The launcher runs workers at **niceness 10** by default (lower CPU priority), so
  others' jobs are not starved. Tune with `--nice` (0 = normal priority).
- Route coverage is not a concern: the catalog has only **12 routes** (PATH1–PATH6
  × forward/reversed); even 100 episodes already sees all 12 origin/destination
  pairs, so 6000 covers every pair many times over.
- Trade-off of success-only: it under-represents the hardest configurations (which
  fail most), but a large 6000-episode target still captures many hard-but-handled
  cases. If early distillation shows the student weak in conflict/weather moments,
  regenerate with `--keep success_or_truncated` (keeps ran-out-of-time episodes,
  still drops collision/weather crashes).

```bash
# Session A1: best_preview dataset — 6000 successful episodes across 12 CPU workers
$PY new_approach/generate_parallel.py --guidance-source best_preview --keep success \
    --total-episodes 6000 --workers 16 --seed 42 --num-agents 4 --num-weather-cells 2

# Session A2: preview_safe dataset — 6000 successful episodes across 12 CPU workers
$PY new_approach/generate_parallel.py --guidance-source preview_safe --keep success \
    --total-episodes 6000 --workers 16 --seed 42 --num-agents 4 --num-weather-cells 2

# Session A3: train the best_preview teacher (loader merges all part_XX/ dirs)
$PY new_approach/train_teachers.py --dataset-dir distill_data/best_preview \
    --out-path teachers/teacher_best_preview.pt --name best_preview

# Session A4: train the preview_safe teacher
$PY new_approach/train_teachers.py --dataset-dir distill_data/preview_safe \
    --out-path teachers/teacher_preview_safe.pt --name preview_safe
```

(Single-process alternative, if you don't want parallelism:
`$PY new_approach/generate_guided_trajectories.py --guidance-source best_preview --keep success --n-episodes 6000 ...`)

**Phase B — Ollama-free training runs (4 experiments × seeds 0,1,2).** Each block
is one tmux session running its 3 seeds sequentially. Set `STEPS` first.

```bash
STEPS=300000          # CPU pilot: lower to 60000 for a faster first look

# Session B0: T0 baseline (no teachers)
for s in 0 1 2; do
  $PY train.py --no-distill --seed $s --max-agent-steps $STEPS \
      --num-agents 4 --num-weather-cells 2 \
      --log-dir runs/T0_baseline_seed$s --no-resume
done

# Session B1: S-BP (best_preview teacher only)
for s in 0 1 2; do
  $PY train.py --distill --distill-teachers best_preview --competence-mode shadow \
      --seed $s --max-agent-steps $STEPS --num-agents 4 --num-weather-cells 2 \
      --log-dir runs/S_BP_seed$s --no-resume
done

# Session B2: S-PS (preview_safe teacher only)
for s in 0 1 2; do
  $PY train.py --distill --distill-teachers preview_safe --competence-mode shadow \
      --seed $s --max-agent-steps $STEPS --num-agents 4 --num-weather-cells 2 \
      --log-dir runs/S_PS_seed$s --no-resume
done

# Session B3: LOO-LLM = both heuristics, no LLM (best_preview + preview_safe)
for s in 0 1 2; do
  $PY train.py --distill --distill-teachers best_preview preview_safe \
      --competence-mode shadow --seed $s --max-agent-steps $STEPS \
      --num-agents 4 --num-weather-cells 2 \
      --log-dir runs/LOO_noLLM_seed$s --no-resume
done
```

**Phase C — evaluate every Phase-B run.** Replace `<TAG>` with `T0_baseline`,
`S_BP`, `S_PS`, `LOO_noLLM`.

```bash
for s in 0 1 2; do
  $PY evaluate_rl.py --model-path runs/<TAG>_seed$s/weights/best_model.pt \
      --n-episodes 100 --num-agents 4 --num-weather-cells 2 \
      --metrics-path runs/<TAG>_seed$s/eval.json
done
```

### Phase D — needs Ollama (run later, when gpt-oss is available)

```bash
# D1: LLM teacher dataset — target 250 successful episodes (needs Ollama; slow, so
#     a smaller target than Phase A). Use FEW workers since they share one Ollama
#     server (or run the single-process form).
$PY new_approach/generate_parallel.py --guidance-source real --keep success \
    --total-episodes 250 --workers 2 --ollama-model gpt-oss:20b \
    --seed 42 --num-agents 4 --num-weather-cells 2

# D2: train the LLM teacher
$PY new_approach/train_teachers.py --dataset-dir distill_data/real \
    --out-path teachers/teacher_llm.pt --name llm
```

Once `teacher_llm.pt` exists, these runs are Ollama-free (same `train.py` flags):

| Run | `--distill-teachers` | `--competence-mode` |
|---|---|---|
| S-LLM | `llm` | `shadow` |
| LOO-BP (llm + preview_safe) | `llm preview_safe` | `shadow` |
| LOO-PS (llm + best_preview) | `llm best_preview` | `shadow` |
| TRI-equal | `llm best_preview preview_safe` | `equal` |
| TRI-shadow ⭐ | `llm best_preview preview_safe` | `shadow` |

These flags override `configs.py` per run, so each tmux session is fully
self-contained.
