# Session Handoff — Distillation Study State & Operations

**Purpose:** a single reference so progress and hard-won operational know-how are not lost
between chat sessions. Snapshot time: **3 Sep 2026, 10:00**. Sister docs: `DISTILLATION_RESULTS.md`
(results table), `EXPERIMENT_PLAN.md` (training runbook + Appendix A = pending seed-runs),
**`CONCLUSION_AND_BENCHMARK_PLAN.md` (the current evaluation plan — read this first)**,
`new_approach/LLM_MAPPO_Research_Documentation.docx` (write-up).

Python: `/home/aradhya.dhruv/anaconda3/envs/llmdrl/bin/python` (conda env `llmdrl`).
Project dir: `.../multi-agent-isolated-llm-guidance`. All runs live under
`Evaluation_distillation_approach/<TAG>_seed<N>/`.

---

## 1. Run status (12 run folders exist; target = 5,000,000 steps each)

Steps read from `Evaluation_distillation_approach/<run>/weights/last_checkpoint.pt`.

| Run | State | Step |
|---|---|---|
| T0_baseline_seed0 / seed1 / seed2 | ✅ complete | 5.00M each |
| S_BP_seed0, S_PS_seed0, S_LLM_seed0 | ✅ complete | 5.00M each |
| LOO_noLLM_seed0, TRI_equal_seed0, TRI_equal_seed1 | ✅ complete | 5.00M each |
| **LOO_noBP_seed0** | 🔄 training (GPU 1) | ~4.50M |
| **LOO_noPS_seed0** | 🔄 training (GPU 1) | ~4.46M |
| **TRI_shadow_seed0** | 🔄 training (GPU 1) | ~3.41M |

**9 complete, 3 training, 0 halted.** ETA: LOO_noBP/LOO_noPS ~4–6 days; **TRI_shadow ~3 weeks**
(see §6e for why it never speeds up).

**Evaluation scope decision (3 Sep):** only **seed 0** is evaluated — 9 runs. The seed-1/seed-2
folders (`T0_baseline_seed1/2`, `TRI_equal_seed1`) are kept but deliberately **not** scored, so all
evaluated runs are treated identically.

**Still never trained: 15 seed-runs** (seeds 1 & 2 of most experiments). Commands in
`EXPERIMENT_PLAN.md` Appendix A. Not planned for now — there is no time to train.

## 2. Evaluation results

⚠ **Two separate result sets — never mix them.** See `CONCLUSION_AND_BENCHMARK_PLAN.md` Part 5.

**(a) Old scores, still valid** — `Evaluation_distillation_approach/<run>/eval_last.json`
(9 files; 300 episodes, seed 42, 4 aircraft, 2 storms, on the 5M checkpoints). These produced the
two findings we already trust:
- **Finding 1 (headline):** the rule-taught run (`S_PS`) causes significantly fewer collisions
  (6 vs 18 discordant episodes, p=0.023); the LLM-taught run (`S_LLM`) causes significantly fewer
  storm entries (4 vs 16, p=0.012). **No single teacher is best at both.**
- **Finding 2:** `S_LLM` beats the baseline by **+6.0 points** (rescued 42, lost 24, p=0.036).
- Reproduce either with:
  `python new_approach/analyze_pairwise.py --pattern "Evaluation_distillation_approach/*/eval_last.json" --baseline T0_baseline_seed0 --out /tmp/verify.md`

**(b) New two-setting results (in progress)** — `new_approach/eval_results/`. Setting 2 (~3M, all
9 runs, equal training) is the **primary** one; Setting 1 (latest checkpoint) is a companion valid
only for the 6 finished runs. Nothing conclusive yet.

**Caveat that governs every conclusion:** one training seed per method. The three baseline seeds
scored **81.7 / 80.7 / 59.7 %** — training luck exceeds the 2–6 point method gaps.

## 3. Current tmux sessions

**Training (3 — never touch these):** `LOO_noBP`, `LOO_noPS`, `TRI_shadow` — all on GPU 1.

**Evaluation (7 running, launched 3 Sep 09:05):** `S1_baseB`, `S1_D1`, `S1_D2`, `S2_baseB`,
`S2_D1`, `S2_D2`, `shared_F`. Each runs `new_approach/experiment_jobs/jobs_<session>.sh`
sequentially on CPU; logs in `new_approach/experiment_jobs/logs/<session>.log`.
`shared_E` (decision latency) already finished — a session closing itself is normal.

**Idle leftovers, safe to ignore or kill:** `LOO_noLLM` (its training finished), `eval_S_LLM`,
`eval_LOO_noLLM`, `eval_TRI_equal_s1` (from an older evaluation round).

## 4. GPU layout of running work
| Run | GPU (`CUDA_VISIBLE_DEVICES`) |
|---|---|
| LOO_noBP_seed0 | 1 |
| LOO_noPS_seed0 | 1 |
| TRI_shadow_seed0 | 1 |

Avoid GPUs 2/3/7 (other users, ~100%). GPUs 0,1,4,5,6 usually have room (each run ≈ 250 MB).
Runs are **CPU-bound**, so which GPU barely affects speed — CPU contention does.

---

## 5. How to resume / move a run (the critical procedure)

### Golden rule — match the GPU **device count**, or it fresh-starts
A checkpoint stores one RNG state per GPU visible when it was saved (`cuda_rng`). On resume
you must expose **≥ that many** GPUs, else you get `tuple index out of range` → "Starting
fresh" → it silently **overwrites your progress from step 0**. Detect the count:
```bash
python -c "import torch;cr=torch.load('<run>/weights/last_checkpoint.pt',map_location='cpu',weights_only=False).get('cuda_rng');print(len(cr) if cr else 1)"
```
- count = 1 → `export CUDA_VISIBLE_DEVICES=1` (any single GPU)
- count = 2 → `export CUDA_VISIBLE_DEVICES=0,1` (two GPUs; it runs on the first)

The 3 currently-running runs were last on 1 GPU → **resume with a single GPU** (count=1).

### Resume commands (if these halt again — same args as launch)
```bash
# in a fresh tmux session; set the GPU first
export CUDA_VISIBLE_DEVICES=1
python train.py --distill --distill-teachers llm preview_safe --competence-mode shadow --seed 0 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/LOO_noBP_seed0
python train.py --distill --distill-teachers llm best_preview --competence-mode shadow --seed 0 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/LOO_noPS_seed0
python train.py --distill --distill-teachers llm best_preview preview_safe --competence-mode shadow --seed 0 --num-agents 4 --num-weather-cells 2 --log-dir Evaluation_distillation_approach/TRI_shadow_seed0
```
On resume it prints `Resumed MAPPO checkpoint … agent_steps=<N>`. **If it says "Starting
fresh", stop it immediately (SIGKILL, not Ctrl-C) before it overwrites the checkpoint.**

### To STOP or MOVE a run without losing progress
`train.py` saves a checkpoint on **Ctrl-C** (`KeyboardInterrupt` handler). So to move a run
to another GPU: `tmux send-keys -t <session> C-c` → wait for "Interrupted. Saved checkpoint"
→ relaunch with the new GPU (its interrupt-save has 1 RNG state if it was on 1 GPU).
- **Never SIGKILL a run you want to keep** (loses progress since last save).
- **Never relaunch on top of a live process** (two processes on one log-dir = corruption).
- Verify a process is truly dead with `ps` before relaunching (`pgrep -f` false-matches your
  own command line).

### Fresh runs (new seeds) are simpler
No checkpoint → they start at step 0, so **no device-count concern** — just
`export CUDA_VISIBLE_DEVICES=<free gpu>` and launch. A fresh run does **not** print "Resumed"
(expected). See `EXPERIMENT_PLAN.md` Appendix A for the 15 commands + session names.

---

## 6. Gotchas & facts learned this session

**a. Two HPC interruptions.** The tmux server + all processes died in **late July** and again
**~Aug 25**. Completed runs were unaffected; in-progress runs recovered from checkpoints. No
data was ever lost because checkpoints were intact and we resumed correctly.

**b. T0 baseline is NOT pure MAPPO with `--no-distill` alone.** With
`USE_LLM_GUIDED_TRAINING = True` in `configs.py`, a no-distill run falls through to the old
LLM path and **calls Ollama every step** (slow, invalid baseline). Tell-tale: a `memory/`
folder + `llm_guided_mappo_audit.jsonl` in the run dir. Fix: set
`USE_LLM_GUIDED_TRAINING = False` before launching T0. (It's currently `False`.) Distill runs
are unaffected — `distill_active=True` takes precedence.

**c. Which checkpoint to evaluate — two rules.**
(i) Use `last_checkpoint.pt`, **not** `best_model.pt`: for a *finished* run the true final 5M model
lives only in `last_checkpoint.pt` (the numbered snapshots stop earlier). `evaluate_rl.py`'s
`load_model` reads it fine and ignores optimizer/RNG, so there is no device-count issue on eval.
(ii) **BUT never use `last_checkpoint.pt` for a run that is still training** — it is rewritten every
few minutes, so the score could never be reproduced. Use that run's newest **frozen**
`checkpoint_<step>.pt` instead (currently LOO_noBP `4147227`, LOO_noPS `4120186`,
TRI_shadow `3280741`).
Always evaluate on CPU (`CUDA_VISIBLE_DEVICES=""`) so it doesn't disturb GPU training. Every output
JSON records per-episode `seed`, `num_agents`, `num_weather_cells` — that is the proof of settings.

**d. Eval command templates (results now go to `new_approach/eval_results/`, never into a
training folder).**
```bash
# standard per-episode score (Setting 2 example: the run's ~3M checkpoint)
CUDA_VISIBLE_DEVICES="" python evaluate_rl.py \
  --model-path Evaluation_distillation_approach/<RUN>/weights/<checkpoint>.pt \
  --n-episodes 300 --num-agents 4 --num-weather-cells 2 --seed 42 \
  --metrics-path new_approach/eval_results/setting2_3M/<RUN>/evaluation.json

# per-aircraft AAMAS metrics + decision latency (used for the 8/6/4-aircraft benchmark)
CUDA_VISIBLE_DEVICES="" python new_approach/evaluate_per_aircraft.py \
  --model-path Evaluation_distillation_approach/<RUN>/weights/<checkpoint>.pt \
  --n-episodes 300 --num-agents 8 --num-weather-cells 2 --seed 42 \
  --metrics-path new_approach/eval_results/setting2_3M/<RUN>/evaluation_scal_ac8.json
```
The 9 older `Evaluation_distillation_approach/<RUN>/eval_last.json` files are kept as historical
5M scores (they back Findings 1 and 2 in §2a) — do not overwrite them.

**e. Why TRI_shadow is slow (~3 wk), and why it never speeds up.** The expensive shadow
evaluation only runs while `current_distill_weight(step) > 0` (train.py:1343). BUT the decay
config is `DISTILL_LOSS_WEIGHT=0.50` → floor `DISTILL_MIN_WEIGHT=0.10` by step 3M, then
**held at 0.10 forever** (never 0). So the gate never closes and the shadow eval runs the
**entire 5M**. TRI_shadow = `shadow` × **3 teachers** = 3 teacher rollouts every step for all
5M → slowest. This is the deliberate "keep teacher influence alive" floor, **not a bug**. If
"shadow only early, then plain MAPPO" were wanted, set `DISTILL_MIN_WEIGHT=0` — but apply it
to **all** seeds/experiments for comparability (the current shadow runs all used 0.10).

**f. Key decay config (as used by these runs):** `DISTILL_LOSS_WEIGHT=0.50`,
`DISTILL_MIN_WEIGHT=0.10`, `DISTILL_DECAY_START_STEP=0`, `DISTILL_DECAY_END_STEP=3,000,000`,
`LLM_SHADOW_HORIZON=5`, `MAPPO_MAX_AGENT_STEPS=5,000,000`, `MAPPO_CUDA_VISIBLE_DEVICES=None`
(so the shell `CUDA_VISIBLE_DEVICES` is respected).

---

## 7. Immediate next actions (when picking this back up)

1. **Check the evaluations:** `tmux ls`; per-session progress in
   `new_approach/experiment_jobs/logs/<session>.log`. A finished session disappears — that is normal.
2. **First thing to read when it lands:** `new_approach/eval_results/analysis/analysis_S2_base.md`
   (the primary setting's comparison, with p-values), then the same for Setting 1.
3. **Fill in the AAMAS benchmark table** from `evaluation_scal_ac{8,6,4}.json` — compare our
   degradation 4→8 aircraft against the paper's (IDDPG and FMT numbers are in
   `CONCLUSION_AND_BENCHMARK_PLAN.md` Part 4).
4. **Let the 3 training runs finish** untouched. If LOO_noBP/LOO_noPS reach 5M, their Setting-1
   entry can be re-scored at 5M by editing one line (`w1()`) in the job scripts.
5. **Do not re-time decision latency on this loaded box** — the `shared_E` numbers
   (~130–170 ms) are an upper bound taken on CPU under heavy load, not a fair figure.

## 8. The evaluation now running (3 Sep 2026, from 09:05)

Because no new training is possible, the study was re-planned around **evaluating existing
checkpoints only**. Full detail: **`CONCLUSION_AND_BENCHMARK_PLAN.md`**.

**Two settings, kept completely separate:**
- **Setting 2 "~3M" — PRIMARY.** Every one of the 9 runs at its ~3M checkpoint, so all have roughly
  equal training. The only fair comparison for the proposed method `TRI_shadow`.
- **Setting 1 "latest" — companion.** Finished runs at 5M; the 3 still-training runs at their newest
  **frozen** checkpoint. **Not valid** for `TRI_shadow` / `LOO_noBP` / `LOO_noPS` — `TRI_shadow`
  would face 5M rivals with 34% less training.

**Hard rule discovered here:** never evaluate a still-training run's `last_checkpoint.pt` — it is
rewritten every few minutes. Use its newest frozen `checkpoint_<step>.pt` instead
(LOO_noBP `4147227`, LOO_noPS `4120186`, TRI_shadow `3280741`).

**Five tests** (per setting where applicable): main score (300 episodes, seed 42) · two extra
episode sets (seeds 1042/2042) · traffic density 8/6/4 aircraft with per-aircraft AAMAS metrics ·
decision latency · learning curves from earlier checkpoints. **153 jobs total.**

**New evaluation tooling** (added 3 Sep; no existing file was modified):
- `new_approach/evaluate_per_aircraft.py` — per-aircraft AAMAS metrics + decision latency.
  Validated: per-episode outcomes match `evaluate_rl.py` on 8/8 identical seeds.
- `new_approach/analyze_pairwise.py` — episode-by-episode McNemar / bootstrap comparison tables.

**Results layout** — `new_approach/eval_results/` (nothing is written into training folders):
```
setting2_3M/<run>/evaluation.json            <- PRIMARY main score
setting2_3M/<run>/evaluation_seed{1042,2042}.json
setting2_3M/<run>/evaluation_scal_ac{8,6,4}.json
setting1_latest/<run>/…                      <- same names, latest checkpoints
shared/<run>/latency_ac{4,8}.json, curve_<step>.json
analysis/analysis_S{1,2}_{base,seed1042,seed2042}.md
```

**Verified before launch:** all 153 jobs expand to checkpoint files that exist; no still-training
`last_checkpoint.pt` is read; no seed-1/seed-2 run is referenced; smoke tests passed.
