# Experiment Plan — Multi-Teacher Distillation (CPU, no GPU)

A step-by-step runbook you can follow even after a long break. It explains the
**idea**, **why each experiment exists**, the **exact commands** (one per tmux
session), and **every hyperparameter you can change and when to change it** — all in
plain language.

Companion docs: `DISTILLATION_FLOW.md` (how the code works end to end) and
`CODE_MAP.md` (which script lives in which folder).

---

## 0. The idea, in plain words

We have an air-traffic controller trained with reinforcement learning, called
**MAPPO**. It flies several aircraft through a sector and tries to get them all to
their destinations without collisions or flying into weather.

We want to make MAPPO **train better** by letting it learn from a **teacher**.

- A **guide** is something that already knows a decent move in each situation. We
  have three guides:
  - **LLM** — a language model (gpt-oss) that reasons about the traffic. Needs Ollama.
  - **best_preview** — a cheap rule: simulate each possible turn a few steps ahead
    and pick the top-ranked safe one. No Ollama.
  - **preview_safe** — a cheap rule: pick randomly among the turns that look safe.
    No Ollama.
- A **teacher** is a small network that has *copied* one guide, so we can ask it
  instantly during training (no slow guide needed).
- **Distillation** = while MAPPO trains, we add a gentle extra push in its loss that
  nudges it toward what the teacher would do.

**The one rule that never breaks:** the aircraft are **always flown by MAPPO's own
action**. Teachers only shape the training loss — they never touch the environment.

**The research question:** the papers we compare against use *one* LLM teacher and
assume it is always best. We suspect **no single teacher is best everywhere** — the
LLM is strong overall, but a cheap rule is better in some safety-critical moments.
So we test whether learning from **several teachers at once**, trusting each where it
is stronger, beats using any single teacher.

---

## 1. Why we run these specific experiments (the ladder of claims)

Each experiment is a separately trained controller. We compare their success rates.
The comparisons build a chain of evidence:

1. **Teaching helps at all** → the teacher runs beat the no-teacher baseline (T0).
2. **No single teacher is enough** → the all-three-teacher run beats every
   single-teacher run.
3. **Smart trust matters (not just having many teachers)** → the all-three run with
   *competence weighting* beats the all-three run with *equal* weighting.
4. **Every teacher pulls its weight** → the all-three run beats every "drop one
   teacher" run. (If dropping one changes nothing, that teacher was not needed —
   also a useful, honest result.)

### The experiments

Each row is one trained controller, run 3 times (seeds 0, 1, 2) so results are not a
fluke. "Needs Ollama?" = must wait until the LLM is available.

| Name | Teachers it learns from | Trust rule | Needs Ollama? | Why it exists |
|---|---|---|---|---|
| **T0** | none (plain MAPPO) | — | No | the baseline everything must beat |
| **S-BP** | best_preview | shadow | No | how good is one heuristic teacher alone |
| **S-PS** | preview_safe | shadow | No | how good is the other heuristic alone |
| **LOO-noLLM** | best_preview + preview_safe | shadow | No | do the two heuristics help together |
| **S-LLM** | llm | shadow | Yes | single-LLM teacher = the prior-work baseline |
| **LOO-noBP** | llm + preview_safe | shadow | Yes | is best_preview needed? (drop it) |
| **LOO-noPS** | llm + best_preview | shadow | Yes | is preview_safe needed? (drop it) |
| **TRI-equal** | all three | equal | Yes | do we need smart trust, or is averaging enough |
| **TRI-shadow** ⭐ | all three | shadow | Yes | **the proposed method** |

**Do now (no Ollama):** T0, S-BP, S-PS, LOO-noLLM.
**Do later (needs Ollama):** the other five.

"Trust rule" = how much to believe each teacher in a given moment (explained in
Section 10 under *Competence mode*).

---

## 2. One-time setup

**a) Make the code use the CPU.** In `configs.py` set:

```python
MAPPO_CUDA_VISIBLE_DEVICES = None
```

This makes `train.py` obey the shell instead of grabbing a GPU.

**b) Start every tmux session with these lines** (the "gentle profile" — forces CPU
and makes every command a polite neighbour on the shared server):

```bash
conda activate llmdrl               # so `python` = the llmdrl env's python
export OMP_NUM_THREADS=2            # each training run uses ~2 cores, not all of them
cd "/home/aradhya.dhruv/rl/behaviour_cloning/LLM enhanced DRL/LLM-DRL restructured code/multi-agent-isolated-llm-guidance"
```

The commands below call `python` directly (run them from the activated `llmdrl` env).

What the setup does:
- **GPU choice** is handled in `configs.py` via `MAPPO_CUDA_VISIBLE_DEVICES` (e.g. `"1"`
  for GPU 1, `""` for CPU); it **overrides** any shell `CUDA_VISIBLE_DEVICES`.
- `OMP_NUM_THREADS=2` → caps how many cores one **training** process may use. (Data
  generation pins its own workers to 1 core each automatically.)
- **Polite neighbour (optional but recommended on the shared box):** prefix training with
  `nice -n 19 python ...` (lowest CPU priority, so your jobs yield when a colleague needs
  the machine). Data generation already takes `--nice 19`.

**Shared-machine etiquette:** this box has 32 real cores and is shared. Generate data
with `--workers 12 --nice 19`, keep to **~3–4 training sessions at once**, and check
`htop` first.

---

## 3. tmux quick reference

```bash
tmux new -s NAME       # start a new named session
#   (run your command inside it)
# detach (leave it running):  press Ctrl-b, then d
tmux ls                # list running sessions
tmux attach -t NAME    # go back into a session
```

Use one session per step below; the suggested NAME is given each time.

---

## 4. Phase A — Generate the teacher training data (no Ollama)

**Why:** to build a teacher we first need examples of the guide's decisions. We play
games with a guide and record, for every decision, `(situation → the move the guide
chose)`. We keep **only successful games** so the teacher copies good behaviour.

`--total-episodes` is the number of **successful** games to collect — the tool keeps
playing until it has that many. It splits the work across 12 CPU workers.

**How many?** `--total-episodes 6000` (≈240k records) is generous; 3000–4000 already
trains a good teacher. More games = more *time*, not more memory (records stream to
disk). Full list of choices in Section 10.

> Run these **one after another** (each uses ~12 cores).

**Session `gen_bp`** — best_preview data:
```bash
python new_approach/generate_parallel.py --guidance-source best_preview --keep success \
    --total-episodes 6000 --workers 12 --nice 19 --seed 42 \
    --num-agents 4 --num-weather-cells 2
# output: distill_data/best_preview/part_00 .. part_11
```

**Session `gen_ps`** — preview_safe data:
```bash
python new_approach/generate_parallel.py --guidance-source preview_safe --keep success \
    --total-episodes 6000 --workers 12 --nice 19 --seed 42 \
    --num-agents 4 --num-weather-cells 2
# output: distill_data/preview_safe/part_00 .. part_11
```

Tip: add `--dry-run` first to preview what it will launch. If the box is busy, use
`--workers 8`.

---

## 5. Phase B — Train the teacher networks (no Ollama)

**Why:** turn the recorded examples into a fast network. The teacher learns (by
supervised learning) to predict the guide's move from the situation. It is now the
**same size as the MAPPO actor**, so it can copy the guide faithfully. Fast (a few
minutes). The trainer merges all `part_XX/` folders automatically.

**Session `teach_bp`** (after `gen_bp`):
```bash
python new_approach/train_teachers.py --dataset-dir distill_data/best_preview \
    --out-path teachers/teacher_best_preview.pt --name best_preview
```

**Session `teach_ps`** (after `gen_ps`):
```bash
python new_approach/train_teachers.py --dataset-dir distill_data/preview_safe \
    --out-path teachers/teacher_preview_safe.pt --name preview_safe
```

**Check it worked:** the printout shows `best_val_acc` (how often the teacher matches
the guide on held-out examples) vs `majority_val` (always guessing the most common
move). `best_val_acc` clearly above `majority_val` = the teacher really learned.

**TensorBoard for teacher training:** curves are also written to
`teachers/tensorboard/<name>/` (per-epoch `teacher/train_loss`, `teacher/val_acc`, and
a flat `teacher/majority_val_acc` reference line). View both teachers with:
```bash
tensorboard --logdir teachers/tensorboard --port 6007
```
You want `teacher/val_acc` to climb well above the flat `teacher/majority_val_acc`
line and plateau.

---

## 6. Phase C — Train the controllers (no-Ollama runs)

**Why:** this is the actual test. Each run trains MAPPO with a different set of
teachers, then we compare. Each block is **one tmux session**, 3 seeds one after
another. The gentle profile from Section 2 (`OMP_NUM_THREADS=2`, niced `PY`) keeps
each run low-priority and core-limited automatically.

> **Why 3 seeds per setup (and what a seed actually changes).** A single run is
> *already* diverse **inside** itself: the RNG advances every episode, so one run walks
> through thousands of different routes, start positions, and weather placements —
> scenario variety is handled internally. So why repeat 3 times?
>
> Because the starting seed fixes the run's **whole random journey**: its **initial
> network weights** *and* the exact sequence of scenarios/sampled actions it happens to
> take. That makes one run a **single sample** of the outcome — deep RL is famous for
> two identical-recipe runs finishing at, say, 74% and 78% purely from different
> starting weights and paths. Internal scenario diversity does **not** remove this
> run-to-run variation.
>
> So seeds 0/1/2 are **not** about adding scenario variety (you already have that) —
> they sample the **outcome variation**, so we compare setups by their **average across
> seeds**, not one lucky or unlucky journey. A gap that shows up in *all* seeds is real;
> a gap smaller than the seed-to-seed spread is just noise. Use 3 seeds for a pilot, 5
> for the final numbers.

> **Steps:** these commands do **not** pass `--max-agent-steps`, so each run uses
> `MAPPO_MAX_AGENT_STEPS` (5,000,000) from `configs.py` — the full run.

**Session `T0`** — baseline, no teacher:

> ⚠️ **`--no-distill` is NOT enough for a pure baseline.** It only turns off the
> *distillation* teachers. If `USE_LLM_GUIDED_TRAINING = True` in `configs.py` (the
> current default), a no-distill run **falls through to the old LLM-guidance path** and
> silently calls **Ollama every step** — so T0 runs the real LLM, becomes far *slower*
> than the distill runs, and is not a valid baseline. (Tell-tale signs in the run
> folder: a `memory/` subfolder and an `llm_guided_mappo_audit.jsonl` file — a clean T0
> has neither.)
>
> **Before launching T0, set `USE_LLM_GUIDED_TRAINING = False` in `configs.py`.** Then
> `--no-distill` lands in the pure-MAPPO branch (no Ollama, no shadow eval, no memory) —
> a true, fast baseline. This is **safe for the distill runs**: they set
> `distill_active = True`, which takes precedence and ignores this switch, so you can
> leave it `False` for the whole study. (Config is read at process start, so set it
> *before* launching.) There is no CLI flag for this — it must be the config edit.

```bash
# One-time (in configs.py):  USE_LLM_GUIDED_TRAINING = False
for s in 0 1 2; do
  python train.py --no-distill --seed $s \
      --num-agents 4 --num-weather-cells 2 \
      --log-dir Evaluation_distillation_approach/T0_baseline_seed$s
done
```

**Session `S_BP`** — best_preview teacher only:
```bash
for s in 0 1 2; do
  python train.py --distill --distill-teachers best_preview --competence-mode shadow \
      --seed $s --num-agents 4 --num-weather-cells 2 \
      --log-dir Evaluation_distillation_approach/S_BP_seed$s
done
```

**Session `S_PS`** — preview_safe teacher only:
```bash
for s in 0 1 2; do
  python train.py --distill --distill-teachers preview_safe --competence-mode shadow \
      --seed $s --num-agents 4 --num-weather-cells 2 \
      --log-dir Evaluation_distillation_approach/S_PS_seed$s
done
```

**Session `LOO_noLLM`** — both heuristics, no LLM:
```bash
for s in 0 1 2; do
  python train.py --distill --distill-teachers best_preview preview_safe \
      --competence-mode shadow --seed $s \
      --num-agents 4 --num-weather-cells 2 \
      --log-dir Evaluation_distillation_approach/LOO_noLLM_seed$s
done
```

**Reality check:** each run trains to the full `MAPPO_MAX_AGENT_STEPS = 5,000,000`
(no `--max-agent-steps` override). On a GPU the `shadow` runs still take days because of
the per-step look-ahead (see Section 10); the teacher influence fully decays by
`DISTILL_DECAY_END_STEP = 3,000,000`, so the last 2M steps are pure RL fine-tuning. If a
run is interrupted, just re-launch the same command in its session — `train.py`
auto-resumes from `weights/last_checkpoint.pt` (see "Resuming an interrupted run" below).

### Resuming an interrupted run, and watching TensorBoard

**Resume is automatic.** `train.py` saves a full checkpoint to
`<log-dir>/weights/last_checkpoint.pt` — model weights, optimizer state, all
random-number states, observation-normalization stats, and the step counter. It saves
periodically during training **and** if the run is interrupted (Ctrl-C or crash). To
continue after an interruption, just **re-run the exact same command** (same
`--log-dir`); it loads that checkpoint and picks up where it stopped. Because each
run/seed has its own `--log-dir`, a resume only ever continues its own run.

- The `for s in 0 1 2` loops are resumable too: a finished seed re-loads its final
  checkpoint and moves on; an interrupted seed continues; a not-yet-started seed runs
  fresh. So if a session dies, just start the same loop again.
- To instead force a **fresh** restart (e.g. after changing a hyperparameter), either
  delete that run's `--log-dir` or add `--no-resume` to that one command.

**Watch training live with TensorBoard.** Curves are written under
`<log-dir>/tensorboard/` as training runs. View all experiments at once:

```bash
tensorboard --logdir Evaluation_distillation_approach --port 6006
# then open http://localhost:6006
# (remote server? forward the port first: ssh -L 6006:localhost:6006 user@server)
```

Useful curves: `train/success`, `train/collision`, `train/weather_failure` (outcome
rates over time), and `loss/distill_loss`, `loss/distill_weight_mean` (how strongly
the teachers are influencing the student). Put several runs on the same chart to see
which teacher setup learns faster and higher.

---

## 7. Phase D — Evaluate the trained controllers

**Why:** measure each finished controller on 100 fresh games and compare success
rates. Replace `<TAG>` with a finished run name.

**Session `eval`** (reuse one session):
```bash
for TAG in T0_baseline S_BP S_PS LOO_noLLM; do
  for s in 0 1 2; do
    python evaluate_rl.py \
        --model-path Evaluation_distillation_approach/${TAG}_seed$s/weights/best_model.pt \
        --n-episodes 100 --num-agents 4 --num-weather-cells 2 \
        --metrics-path Evaluation_distillation_approach/${TAG}_seed$s/eval.json
  done
done
```

Compare the **success rate** across the four, averaged over the 3 seeds. This already
tells you whether the heuristic teachers help and whether combining them beats using
one alone.

---

## 8. Phase E — The Ollama runs (do later, when gpt-oss is back)

**E1 — make the LLM teacher** (needs Ollama; slower, so fewer games and few workers
because they share one Ollama server).

Session `gen_real`:
```bash
python new_approach/generate_parallel.py --guidance-source real --keep success \
    --total-episodes 250 --workers 2 --nice 19 --ollama-model gpt-oss:20b \
    --seed 42 --num-agents 4 --num-weather-cells 2
```

Session `teach_llm`:
```bash
python new_approach/train_teachers.py --dataset-dir distill_data/real \
    --out-path teachers/teacher_llm.pt --name llm
```

**E2 — the five remaining controllers.** Same pattern as Phase C. Only the
`--distill-teachers` / `--competence-mode` change.

**Steps:** no `--max-agent-steps` here — each run uses `MAPPO_MAX_AGENT_STEPS`
(5,000,000) from `configs.py`, as intended.

**GPU (use GPU 1):** the GPU is chosen by `MAPPO_CUDA_VISIBLE_DEVICES` in `configs.py`,
which **overrides** any shell `export CUDA_VISIBLE_DEVICES`. It is currently `"0,1"`
(→ physical GPU 0). To run Phase E on **GPU 1**, do **one** of these once, before
launching:
- **Simplest (all these runs → GPU 1):** set `MAPPO_CUDA_VISIBLE_DEVICES = "1"` in
  `configs.py`.
- **Per-session control:** set `MAPPO_CUDA_VISIBLE_DEVICES = None` in `configs.py`, then
  start each tmux session with `export CUDA_VISIBLE_DEVICES=1`. (With the config left at
  `"0,1"`, the `export` is ignored — so you must change the config for the export route
  to work.)

**tmux layout:** each block below is **its own tmux session** (5 sessions:
`S_LLM`, `LOO_noBP`, `LOO_noPS`, `TRI_equal`, `TRI_shadow`). The `for s in 0 1 2` loop
runs the **3 seeds one after another** inside that session. You may run the 5 sessions in
parallel (each run ≈ 1 CPU core + tiny GPU memory) or stagger them. Do **not** paste all
five loops into a single session unless you want them to run fully sequentially (very
long: 5 × 3 × 5M steps back-to-back).

```bash
# Session S_LLM  (single LLM teacher = prior-work baseline)
for s in 0 1 2; do
  python train.py --distill --distill-teachers llm --competence-mode shadow \
      --seed $s --num-agents 4 --num-weather-cells 2 \
      --log-dir Evaluation_distillation_approach/S_LLM_seed$s
done

# Session LOO_noBP  (llm + preview_safe — drop best_preview)
for s in 0 1 2; do
  python train.py --distill --distill-teachers llm preview_safe --competence-mode shadow \
      --seed $s --num-agents 4 --num-weather-cells 2 \
      --log-dir Evaluation_distillation_approach/LOO_noBP_seed$s
done


# Session LOO_noPS  (llm + best_preview — drop preview_safe)
for s in 0 1 2; do
  python train.py --distill --distill-teachers llm best_preview --competence-mode shadow \
      --seed $s --num-agents 4 --num-weather-cells 2 \
      --log-dir Evaluation_distillation_approach/LOO_noPS_seed$s
done

# Session TRI_equal  (all three, equal trust)
for s in 0 1 2; do
  python train.py --distill --distill-teachers llm best_preview preview_safe \
      --competence-mode equal --seed $s \
      --num-agents 4 --num-weather-cells 2 \
      --log-dir Evaluation_distillation_approach/TRI_equal_seed$s
done

# Session TRI_shadow  (all three, smart trust) -- the proposed method
for s in 0 1 2; do
  python train.py --distill --distill-teachers llm best_preview preview_safe \
      --competence-mode shadow --seed $s \
      --num-agents 4 --num-weather-cells 2 \
      --log-dir Evaluation_distillation_approach/TRI_shadow_seed$s
done
```

**E3 — evaluate them** (same as Phase D, new tags):
```bash
for TAG in S_LLM LOO_noBP LOO_noPS TRI_equal TRI_shadow; do
  for s in 0 1 2; do
    python evaluate_rl.py \
        --model-path Evaluation_distillation_approach/${TAG}_seed$s/weights/best_model.pt \
        --n-episodes 100 --num-agents 4 --num-weather-cells 2 \
        --metrics-path Evaluation_distillation_approach/${TAG}_seed$s/eval.json
  done
done
```

---

## 9. How to read the final results

### The naming code (run folders in `Evaluation_distillation_approach/`)

- **T0** = baseline, zero teachers
- **S_** = **S**ingle teacher
- **LOO_** = **L**eave **O**ne **O**ut (all three teachers *minus* one)
- **TRI_** = all **thr**ee teachers
- **seedN** = which random seed (same experiment, different dice — see the Phase C note)

### Folder by folder (alphabetical, as they appear in VS Code)

| Folder | Teachers used | What it answers |
|---|---|---|
| **LOO_noBP_seed\*** | llm + preview_safe | Drop **best_preview** — does best_preview actually contribute? |
| **LOO_noLLM_seed\*** | best_preview + preview_safe | Drop the **LLM** — how much does the expensive LLM really add over the two cheap heuristics? |
| **LOO_noPS_seed\*** | llm + best_preview | Drop **preview_safe** — does preview_safe actually contribute? |
| **S_BP_seed\*** | best_preview only | How good is one cheap heuristic teacher alone? |
| **S_LLM_seed\*** | llm only | **The prior-work baseline** — what published papers do (distill from one LLM teacher) |
| **S_PS_seed\*** | preview_safe only | How good is the other cheap heuristic alone? |
| **T0_baseline_seed\*** | none (pure MAPPO) | **The floor** — what you get with no teaching at all |
| **TRI_equal_seed\*** | all three, **equal** trust | Ablation: is *smart* trust needed, or is plain averaging enough? |
| **TRI_shadow_seed\*** | all three, **shadow** trust | **The proposed method** — trust each teacher where it's actually better |

### How they combine (line up average success rates over the seeds, then check in order)

1. **Any teacher beats no teacher?** → S_\* vs **T0** — teaching helps at all.
2. **Proposed beats prior work?** → **TRI_shadow** vs **S_LLM** ← *the headline claim*
   (multiple complementary teachers beat one LLM teacher).
3. **Is the smart weighting doing the work?** → **TRI_shadow** vs **TRI_equal** — the
   competence rule matters, not just having more teachers.
4. **Does each teacher earn its place?** → each **LOO_** vs **TRI_shadow** — if dropping
   a teacher hurts, that teacher contributes.
5. **Is the LLM worth its cost?** → **LOO_noLLM** vs **TRI_shadow**.

The two most important comparisons are **#2** (TRI_shadow vs S_LLM) and **#3**
(TRI_shadow vs TRI_equal).

If TRI_shadow ≈ LOO_noPS, that honestly says preview_safe was not needed and two
teachers suffice — a perfectly good result to report. Remember: no comparison is
trustworthy until each experiment has **all its seeds** run (a gap smaller than the
seed-to-seed spread is just noise).

---

## 10. Hyperparameters — the knobs, organized by phase

Everything below has a sensible default already set. This section is so that, later,
you know **what each knob does and when to change it**. Values shown are the current
defaults. Knobs marked **[CLI]** are set on the command line; **[configs.py]** are
edited in `configs.py`. The knobs are grouped by the phase (A–E) where you actually
touch them, plus a cross-cutting block at the top and a troubleshooting block at the end.

### Across all phases — the scenario (keep identical everywhere for a fair comparison)
- `--num-agents 4`, `--num-weather-cells 2` **[CLI]** — how many aircraft and weather
  cells per game. This is the difficulty. **Do not vary it between runs** (generation,
  training, or eval) or the comparison is unfair. (Harder settings like 8 aircraft
  exist, but pick one and stick to it.)
- `MAX_STEP = 60` **[configs.py]** — max steps per game. Leave as-is.
- 13 possible turns, from −30° to +30° **[configs.py `ACTION_BINS`]**. Leave as-is.

### Phase A — Data generation (`generate_parallel.py`)
- `--total-episodes 6000` **[CLI]** — number of **kept successful** games to collect
  (it keeps playing until it reaches this many). For a fair multi-teacher claim, use the
  **same target for every source**; the LLM is the bottleneck, so a common value like
  1000 is reasonable. Affects time, not quality much, and not memory.
- `--keep success` **[CLI]** — which games to keep. Options:
  - `success` (default): only games where all aircraft landed — cleanest teacher.
  - `success_or_truncated`: also keep "ran out of time" games (more hard-situation
    coverage), still drop crashes. Try this if the trained student is weak specifically
    in traffic/weather moments.
  - `all`: keep everything (not recommended — teaches bad moves too).
- `--workers 12`, `--nice 19` **[CLI]** — parallelism and politeness (Section 2).
  **Heuristics scale with cores** (use 12). **The LLM does not** — all workers share one
  Ollama GPU, so use `--workers 2` there (see Phase E).
- `--seed 42` **[CLI]** — base random seed. When *adding* a batch on top of existing
  data, use a **new far-away seed** (e.g. `100000000`) and a nested `--out-dir
  distill_data/<source>/batch2` so you neither duplicate games nor overwrite `part_XX`.

### Phase B — Teacher training (`train_teachers.py`) — mostly leave alone
- Architecture = the MAPPO actor's: `TEACHER_HIDDEN_DIMS = (256, 256, 128)`,
  `TEACHER_ORTHOGONAL_INIT = True` **[configs.py]**. Kept identical on purpose (no
  "teacher too small" objection). Shrink only for a speed test.
- `TEACHER_EPOCHS = 30`, `TEACHER_LR = 3e-4`, `TEACHER_BATCH_SIZE = 256`
  **[configs.py, also CLI flags on train_teachers.py]** — standard training settings.
  **Don't crank `--epochs` high** — early stopping should end the run near the val peak;
  extra epochs only overfit and the saved checkpoint is the **best-val** one anyway.
- `TEACHER_VAL_FRACTION = 0.1` — 10% of examples held out to measure `best_val_acc`.
- `TEACHER_LABEL_SMOOTHING = 0.15`, `TEACHER_WEIGHT_DECAY = 3e-4`,
  `TEACHER_EARLY_STOP_PATIENCE = 5` **[configs.py]** — anti-overfitting / stopping.
  Signature of overfitting: `train_loss` keeps falling while `val_acc` peaks then
  declines → raise label smoothing / weight decay, **or add data (Phase A)**, which is
  usually the stronger lever. If `val_acc` is low *and* flat (underfitting), raise epochs.
- **Reading teacher quality (the right way):** the 13 turns are an *ordered* number
  line, so plain `val_acc` (exact top-1) is harsh. Also watch `teacher/val_within1_acc`
  (within one 5° bin) and `teacher/val_deg_err` (mean degree error) — a good teacher can
  be ~5–10° off even at ~50% exact-match. Compare all of them to `majority_val_acc`
  (the lazy baseline the teacher must clearly beat). A noisier guide like `preview_safe`
  has a **lower learnable ceiling** than the deterministic `best_preview` — expected, not
  a bug.

### Phase C — Controller (distillation) training (`train.py`)

**Which teachers + how strongly (the experiment dials):**
- `--distill / --no-distill` **[CLI]** — turn teaching on/off (on = a teacher run,
  off = T0 baseline). Overrides `DISTILL_ENABLED`.
- `--distill-teachers ...` **[CLI]** — which teachers to use, any subset of
  `llm best_preview preview_safe`. This is what makes each experiment different.
  Overrides `DISTILL_TEACHERS`.
- `DISTILL_LOSS_TYPE = "js"` **[configs.py]** — how the student is pulled toward a
  teacher:
  - `js` (default): match the teacher's whole spread of preferences.
  - `ce`: copy just the teacher's single top move.
  - `soft_kl`: match a softened target around the teacher's move.
- `DISTILL_LOSS_WEIGHT = 0.25` **[configs.py]** — strength of the teacher pull vs
  normal RL. Lower it (e.g. 0.1) if the student collapses into just imitating; raise it
  (e.g. 0.5) if teachers barely move the student.
- Decay: `DISTILL_DECAY_ENABLED = True`, from `DISTILL_DECAY_START_STEP = 0` to
  `DISTILL_DECAY_END_STEP = 250000`, down to `DISTILL_MIN_WEIGHT = 0.0`
  **[configs.py]**. Meaning: the teacher pull starts at 0.25 and **fades to zero by
  step 250,000**, so late training is pure RL and the student can *surpass* its
  teachers. This is why `STEPS = 250000` is the "covers the whole teaching window" number.

**Competence mode — how much to trust each teacher moment-to-moment:**
- `--competence-mode shadow` **[CLI]** — options:
  - `shadow` (default, proposed): at each decision, quickly simulate a few steps ahead
    for MAPPO's move and each teacher's move; trust a teacher in proportion to how much
    its move beats MAPPO. Data-driven, no hand-tuning.
  - `equal`: trust all active teachers equally (the "is smart trust needed?" ablation).
  - `weather_rule`: a fixed rule — trust heuristics more near weather, the LLM more
    otherwise.
- `LLM_SHADOW_HORIZON = 5` **[configs.py]** — how many steps the look-ahead simulates.
  Raise to 10–15 for a longer-sighted judgment (costs more compute per step).
- `LLM_SHADOW_RETURN_MARGIN = 0.05` **[configs.py]** — a teacher must beat MAPPO by at
  least this much to earn any trust (filters out noise). Raise to be stricter.
- `LLM_RETURN_WEIGHT_AGENT_COEF = 0.65`, `LLM_RETURN_WEIGHT_JOINT_COEF = 0.35`
  **[configs.py]** — blends "did it help this aircraft" (65%) vs "did it help the whole
  team" (35%).
- `LLM_SHADOW_GAMMA = 0.99` **[configs.py]** — discount inside the look-ahead. Leave.

**Training length + core RL:**
- `--max-agent-steps` **[CLI, our `STEPS`]** — how long to train. `60000` = quick CPU
  pilot; `250000` = covers the full teaching window; the code default is 5,000,000 for
  a full-scale run. Longer = better but slower.
- `--seed 0/1/2` **[CLI]** — repeat runs so results are not luck.
- Core RL settings (`MAPPO_LR_ACTOR/CRITIC = 1e-4`, `MAPPO_GAMMA = 0.99`,
  `MAPPO_HIDDEN_DIMS = (256,256,128)`) **[configs.py]** — **keep identical across all
  runs** so any difference comes from the teachers, not the RL settings.

### Phase D — Evaluation (`evaluate_rl.py`)
- `--n-episodes 100` **[CLI]** — fresh games to score each controller on. More = steadier
  numbers; 100 is a reasonable default.
- `--num-agents 4`, `--num-weather-cells 2` **[CLI]** — **must match** the training
  scenario (see cross-cutting block) or the comparison is meaningless.
- Evaluate **every seed** (0/1/2) and average — report the spread, not one lucky run.

### Phase E — Ollama LLM generation (later, `--guidance-source real`)
- `--workers 2` **[CLI]** — **not 12.** All LLM workers share one Ollama GPU, so 2–4 is
  the ceiling; more just queue and can OOM. Match `OLLAMA_NUM_PARALLEL` to it.
- `--ollama-model gpt-oss:20b` **[CLI]** — the served model tag. Ollama must be running.
- `--total-episodes` **[CLI]** — keep this **modest** (the LLM is slow) and, for a fair
  claim, **equal to** the heuristics' target. New `--seed` + nested `--out-dir` when
  topping up, same as Phase A.
- Everything downstream (teacher training, the five E2 controller runs) is identical to
  Phases B and C — only `--distill-teachers` / `--competence-mode` change per experiment.

### If the pilot looks disappointing, sweep in this order (cross-cutting)
1. `--keep success_or_truncated` when regenerating data (more hard cases) [Phase A].
2. Add more data and retrain the teachers [Phases A→B] — usually the strongest lever.
3. `LLM_SHADOW_HORIZON` 5 → 10 (longer-sighted trust) and/or
   `LLM_SHADOW_RETURN_MARGIN` (stricter/looser) [Phase C].
4. `DISTILL_LOSS_WEIGHT` (0.1 / 0.25 / 0.5) — how hard teachers push [Phase C].
5. Longer `STEPS` [Phase C].

---

## 11. Order to run things (summary)

1. One-time setup (Section 2).
2. `gen_bp` → `gen_ps` (Phase A, one after the other).
3. `teach_bp`, `teach_ps` (Phase B).
4. `T0`, `S_BP`, `S_PS`, `LOO_noLLM` (Phase C — separate sessions; keep total load modest).
5. `eval` (Phase D).
6. Later, when Ollama is back: `gen_real` → `teach_llm` → the five E2 runs → E3 eval.

Everything on CPU for now. When a GPU frees up, the only change is the shell line
`export CUDA_VISIBLE_DEVICES="<free id>"` before launching — no code or command
changes.

---

## Appendix A — Runs to start individually after the July 2026 interruption

**Why this appendix exists.** In **late July 2026** a machine/tmux interruption killed
the tmux server and all training processes. The **seed-0** runs (and `TRI_equal_seed1`,
which already had a checkpoint) were **resumed** from `weights/last_checkpoint.pt` and
are fine. But the **remaining seeds were never trained** — there is no checkpoint to
resume, so each must be **started fresh** (from step 0) as its **own run**. A resume
command (`--seed 0` + a seed-0 log-dir) only continues that one seed; it does **not**
create the other seeds.

### Status snapshot (as of 14 Aug 2026)

| Experiment | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| T0_baseline | ✅ complete | ✅ complete | ✅ complete |
| S_BP | ✅ complete | ⬜ **start** | ⬜ **start** |
| S_PS | ✅ complete | ⬜ **start** | ⬜ **start** |
| S_LLM | 🔄 running | ⬜ **start** | ⬜ **start** |
| LOO_noBP | 🔄 running | ⬜ **start** | ⬜ **start** |
| LOO_noLLM | 🔄 running | ⬜ **start** | ⬜ **start** |
| LOO_noPS | 🔄 running | ⬜ **start** | ⬜ **start** |
| TRI_equal | ✅ complete | 🔄 running (resumed) | ⬜ **start** |
| TRI_shadow | 🔄 running | ⬜ **start** | ⬜ **start** |

Legend: ✅ complete (5M) · 🔄 running · ⬜ must be started fresh.

**15 seed-runs still to start** (each = a separate `train.py` run + its own tmux session).

### How to launch each one (fresh run recipe)

These are **fresh** runs (no checkpoint), so there is **no device-count/RNG concern** —
just pin each to **one free GPU** and start it. Per session:

```bash
tmux new-session -d -s <SESSION>          # names below
tmux send-keys -t <SESSION> 'cd "…/multi-agent-isolated-llm-guidance"' C-m
tmux send-keys -t <SESSION> 'conda activate llmdrl' C-m
tmux send-keys -t <SESSION> 'export CUDA_VISIBLE_DEVICES=<free id: 2, 3, or 4>' C-m
tmux send-keys -t <SESSION> '<command from the table>' C-m
```

**Shared-machine etiquette:** these runs are CPU-bound and compete with each other, so
**start ~3–5 at a time** (spread across free GPUs 2/3/4), not all 15 at once. Prioritise
the headline experiments' seeds first: **TRI_shadow, S_LLM, TRI_equal**.

### The 15 runs, with tmux session names and commands

Every command below already omits `--max-agent-steps` (uses the 5,000,000 default) and
uses `python` directly (activate `llmdrl` first).

**S_BP** — sessions `S_BP_seed1`, `S_BP_seed2`
```bash
python train.py --distill --distill-teachers best_preview --competence-mode shadow --seed 1 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/S_BP_seed1
python train.py --distill --distill-teachers best_preview --competence-mode shadow --seed 2 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/S_BP_seed2
```

**S_PS** — sessions `S_PS_seed1`, `S_PS_seed2`
```bash
python train.py --distill --distill-teachers preview_safe --competence-mode shadow --seed 1 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/S_PS_seed1
python train.py --distill --distill-teachers preview_safe --competence-mode shadow --seed 2 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/S_PS_seed2
```

**S_LLM** — sessions `S_LLM_seed1`, `S_LLM_seed2`
```bash
python train.py --distill --distill-teachers llm --competence-mode shadow --seed 1 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/S_LLM_seed1
python train.py --distill --distill-teachers llm --competence-mode shadow --seed 2 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/S_LLM_seed2
```

**LOO_noBP** (llm + preview_safe) — sessions `LOO_noBP_seed1`, `LOO_noBP_seed2`
```bash
python train.py --distill --distill-teachers llm preview_safe --competence-mode shadow --seed 1 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/LOO_noBP_seed1
python train.py --distill --distill-teachers llm preview_safe --competence-mode shadow --seed 2 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/LOO_noBP_seed2
```

**LOO_noLLM** (best_preview + preview_safe) — sessions `LOO_noLLM_seed1`, `LOO_noLLM_seed2`
```bash
python train.py --distill --distill-teachers best_preview preview_safe --competence-mode shadow --seed 1 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/LOO_noLLM_seed1
python train.py --distill --distill-teachers best_preview preview_safe --competence-mode shadow --seed 2 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/LOO_noLLM_seed2
```

**LOO_noPS** (llm + best_preview) — sessions `LOO_noPS_seed1`, `LOO_noPS_seed2`
```bash
python train.py --distill --distill-teachers llm best_preview --competence-mode shadow --seed 1 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/LOO_noPS_seed1
python train.py --distill --distill-teachers llm best_preview --competence-mode shadow --seed 2 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/LOO_noPS_seed2
```

**TRI_equal** (all three, equal) — session `TRI_equal_seed2` *(seed 1 already running)*
```bash
python train.py --distill --distill-teachers llm best_preview preview_safe --competence-mode equal --seed 2 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/TRI_equal_seed2
```

**TRI_shadow** (all three, shadow — the proposed method) — sessions `TRI_shadow_seed1`, `TRI_shadow_seed2`
```bash
python train.py --distill --distill-teachers llm best_preview preview_safe --competence-mode shadow --seed 1 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/TRI_shadow_seed1
python train.py --distill --distill-teachers llm best_preview preview_safe --competence-mode shadow --seed 2 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/TRI_shadow_seed2
```

**Note:** because these are fresh runs, on launch they will **not** print
`Resumed MAPPO checkpoint …` — they start at `agent_steps=0`. That is expected here (a
resume line only appears when a `last_checkpoint.pt` already exists for that log-dir).
