# Conclusion & Benchmark Plan

*Last updated 3 Sep 2026. Rule for everything here: **no new training and no resuming** — we only
score checkpoints that already exist. Older drafts are kept as `CONCLUSION_AND_BENCHMARK_PLAN.v*_backup.md`.*

**Terms used throughout (literal, no metaphors):**
- **run** = one trained MAPPO policy = one folder, e.g. `S_LLM_seed0`. We have **9** (seed 0 only).
- **checkpoint** = a saved weights file, e.g. `checkpoint_3103544.pt`; the number is the training
  step. `last_checkpoint.pt` is the newest — and is **rewritten while a run is still training**.
- **episode** = one simulated scenario (4 aircraft cross the sector, 2 storms move). It **succeeds**
  only if *every* aircraft arrives; it **fails** by collision, weather entry, or time-out.
- **teacher** = the source of advice used **during training only** (the LLM, or the rule-based
  `best_preview` / `preview_safe`). No teacher is used at evaluation time.

**Where every claim comes from.** Each number below says either the **file path** it was read from,
or the **script + inputs** that computed it (so you can re-run it), or is marked *not yet produced*.

---

## Part 1 — Where we are

The 9 runs, and how much training each has.
**Source:** step counts read from `Evaluation_distillation_approach/<run>/weights/last_checkpoint.pt`.

| Run | Teacher(s) during training | Steps | State |
|---|---|---|---|
| `T0_baseline_seed0` | none — plain MAPPO (the baseline) | 5,000,051 | finished |
| `S_BP_seed0` | best_preview | 5,000,036 | finished |
| `S_PS_seed0` | preview_safe | 5,000,012 | finished |
| `S_LLM_seed0` | LLM | 5,000,004 | finished |
| `LOO_noLLM_seed0` | best_preview + preview_safe | 5,000,094 | finished |
| `TRI_equal_seed0` | all three, equal trust | 5,000,033 | finished |
| `LOO_noBP_seed0` | LLM + preview_safe | ~4.49M | still training |
| `LOO_noPS_seed0` | LLM + best_preview | ~4.45M | still training |
| `TRI_shadow_seed0` | all three, competence-weighted — **the proposed method** | ~3.40M | still training |

Only **seed 0** is evaluated — all 9 runs above. Seed-1/seed-2 folders exist for a few
experiments (`T0_baseline_seed1`, `T0_baseline_seed2`, `TRI_equal_seed1`) but are deliberately **not**
scored and appear nowhere in this document's results, so every evaluated run is treated identically.

**Related documents, and exactly what they contain.**
- `DISTILLATION_RESULTS.md` — the older results tracker. Its table is the same 5M scoring quoted in
  Part 3 (plus the seed-1/2 rows we exclude here). Its stated protocol is: *"Model evaluated:
  `weights/last_checkpoint.pt` (the final 5,000,000-step model). 300 episodes, `--seed 42` (episodes
  use seeds 42…341 — same 300 scenarios for all). 4 agents, 2 weather cells. Metrics saved to
  `<run>/eval_last.json`."* Its seed-0 numbers are, verbatim: `T0_baseline` 81.7 / `S_BP` 82.3 /
  `S_PS` 85.0 / `S_LLM` **87.7** / `LOO_noLLM` 84.7 / `TRI_equal` 85.3 success %.
  **These are the same numbers as Part 3 below** — that file is not a second source of truth.
- `SESSION_HANDOFF.md` — operational state and procedures (which tmux sessions exist, how to resume a
  training run safely). No results of its own.
- `EXPERIMENT_PLAN.md` — the *training* runbook (how the 9 runs were produced, and the 15 seed-runs
  never started). No evaluation results.

---

## Part 2 — The problem with our original story

Our original headline was *"the three-teacher competence-weighted run (TRI_shadow) beats everything."*
We cannot prove that now, for two reasons:

1. **TRI_shadow is only ~68% trained** (3.40M of 5M) and finishing it would take ~3 more weeks.
2. **The gaps between the teacher runs are smaller than luck.** With 300 episodes, "85%" really
   means "somewhere between 81% and 89%". The teacher runs score 82–88%, all inside that band. And
   each method was trained only **once**, so we cannot separate "this method is better" from "this
   training run got lucky". *(Background, not part of our results: the baseline was once trained with
   three different random seeds and they landed 22 points apart — evidence that training luck can be
   large. Those extra seeds are **not** evaluated or reported anywhere in this study; we use seed 0
   for every run.)*

So we change the headline to something we **can** prove (Part 3), and we benchmark against a
published paper (Part 4).

---

## Part 3 — What our runs already prove

**Where this comes from.** All of Part 3 is computed from 9 files that exist right now:
`Evaluation_distillation_approach/<run>/eval_last.json` (300 episodes each, seed 42, 4 aircraft,
2 storms, scored on the 5M checkpoints). The comparisons were produced by
`new_approach/analyze_pairwise.py`; re-run them any time with:
```bash
python new_approach/analyze_pairwise.py \
    --pattern "Evaluation_distillation_approach/*/eval_last.json" \
    --baseline T0_baseline_seed0 --out /tmp/verify.md
```
### How we decide whether a difference is real (the method, with a worked example)

Every run was scored on the **same 300 episodes** (scenarios 42–341). So instead of comparing two
overall percentages, we can go **episode by episode** and put each of the 300 episodes into exactly
one of four groups. Here are the real counts for collisions, `S_PS` versus `S_LLM`:

| In this episode… | How many | Useful? |
|---|---|---|
| Neither run collided | 275 | no — they agreed |
| Both runs collided | 1 | no — they agreed |
| **Only `S_PS` collided** (so `S_LLM` was better here) | **6** | **yes** |
| **Only `S_LLM` collided** (so `S_PS` was better here) | **18** | **yes** |
| *Total* | *300* | |

The first two groups are episodes where the two runs did the **same thing** (275 + 1 = 276 of them).
They tell us nothing about which run is safer, so we set them aside. Everything hinges on the
**24 episodes where the runs disagreed**: `S_PS` was the one that crashed in **6** of them, and
`S_LLM` was the one that crashed in **18**.

Now the question: *if the two runs were really equally safe, how surprising is a 6-vs-18 split?*
That is exactly a coin-flip question. With 24 disagreements, a fair coin would land 6-or-fewer heads
(or 6-or-fewer tails) only **2.3%** of the time. That 2.3% **is the p-value** — the probability of
seeing a split this lopsided by luck alone. Because 0.023 is below the usual 0.05 threshold, we call
the difference **real** rather than chance. (This is the standard *McNemar test for paired data*; it
is computed exactly, by counting, in `new_approach/analyze_pairwise.py`.)

Two things to note: the p-value depends **only on the two disagreement counts**, and a bigger, more
lopsided split gives a smaller p-value. That is why more episodes (Test 2 in Part 6) can turn a
"maybe" into a "yes".

### The scores (source: the 9 `eval_last.json` files above)

**Seed 0 only.** Six of our nine runs have finished and been scored. (`LOO_noBP`, `LOO_noPS`,
`TRI_shadow` are still training, so they have no `eval_last.json` yet — they enter the story through
the new evaluations in Part 6.)

| Run | Teacher(s) | Success % | Collision % | Weather % | Time-out % |
|---|---|---|---|---|---|
| `S_LLM_seed0` | LLM | **87.7** | 6.3 | **3.7** | 2.3 |
| `TRI_equal_seed0` | all three, equal | 85.3 | 7.7 | 4.3 | 2.7 |
| `S_PS_seed0` | preview_safe | 85.0 | **2.3** | 7.7 | 5.0 |
| `LOO_noLLM_seed0` | best_preview + preview_safe | 84.7 | 6.7 | 5.0 | 3.7 |
| `S_BP_seed0` | best_preview | 82.3 | 4.7 | 6.3 | 6.7 |
| `T0_baseline_seed0` | none (baseline) | 81.7 | 5.7 | 7.0 | 5.7 |

Each row: 300 episodes; the four percentages add to 100 (an episode either succeeds or fails in one
of the three ways). **Read the two bold columns together:** `S_LLM` has the fewest weather failures
(3.7%) but *more* collisions than `S_PS` (6.3 vs 2.3); `S_PS` is the reverse. That contrast is what
Finding 1 tests.

### Finding 1 (the headline) — the two kinds of teacher prevent *different* failures

**What we compared:** `S_PS_seed0` (taught by the rule-based `preview_safe`) against `S_LLM_seed0`
(taught by the LLM), on the same 300 episodes, using the method just described.

| Failure type | both failed | neither failed | only `S_PS` failed | only `S_LLM` failed | disagreements | p-value | real? |
|---|---|---|---|---|---|---|---|
| Collision with another aircraft | 1 | 275 | **6** | 18 | 24 | **0.023** | yes |
| Flying into a storm | 7 | 273 | 16 | **4** | 20 | **0.012** | yes |

(Each row adds to 300. Only the two "only…" columns decide the p-value.)

**How to read it.** Row 1: when the two disagreed about colliding, it was the LLM-taught run at fault
three times as often (18 vs 6) → **the rule-based teacher is better at preventing collisions.**
Row 2 flips: when they disagreed about storms, the rule-taught run was at fault four times as often
(16 vs 4) → **the LLM teacher is better at preventing storm entries.** Both splits are too lopsided
to be luck (p < 0.05).

**Why this matters.** Each teacher is genuinely good at one thing and genuinely worse at the other.
**No single teacher is best at everything** — which is precisely the premise this project set out to
test, and it is now supported by data we already have. This is our headline claim.

*(The same numbers appear as `T3` in the report produced by `new_approach/analyze_pairwise.py`.)*

### Finding 2 — the LLM teacher makes the baseline measurably better

**What we compared:** `S_LLM_seed0` against the untaught `T0_baseline_seed0`, same 300 episodes,
this time on *success* rather than on a failure type.

Same idea as before — sort all 300 episodes into four groups, this time by *who succeeded*:

| In this episode… | How many | Useful? |
|---|---|---|
| Both succeeded | 221 | no — they agreed |
| Both failed | 13 | no — they agreed |
| **Only `S_LLM` succeeded** — it *rescued* an episode the baseline lost | **42** | **yes** |
| **Only the baseline succeeded** — `S_LLM` *lost* one the baseline won | **24** | **yes** |
| *Total* | *300* | |

**Step 1 — the success rates come straight out of this table.**
`S_LLM` succeeded in 221 + 42 = **263** episodes → 263/300 = **87.7%**.
The baseline succeeded in 221 + 24 = **245** episodes → 245/300 = **81.7%**.

**Step 2 — the gain.** 263 − 245 = **18 more episodes won**, out of 300 → **+6.0 percentage points**.
Notice the 234 agreed episodes cancel out entirely; the gain is just **42 − 24 = 18**.

**Step 3 — is it luck?** Everything rests on the 66 disagreements, split **42 to 24**. If the two runs
were truly equal, a coin would produce a split at least this lopsided only **3.6%** of the time.
So **p = 0.036 → real** (below the 0.05 threshold).

**Plain reading:** teaching MAPPO with the LLM turned 42 losses into wins while costing 24 wins —
a genuine net improvement, not a fluke.

### Finding 3 — the other teachers help, but we cannot yet prove it

Exactly the same table as Finding 2, but for each of the remaining teachers against the baseline.
Read each row left to right the same way: the two "agreed" columns cancel, and **rescued − lost** is
the net gain.

| Run vs baseline | both ok | both failed | rescued | lost | net (resc.−lost) | Success % | Gain | p-value | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `S_LLM_seed0` *(Finding 2)* | 221 | 13 | **42** | 24 | **+18** | 87.7 | **+6.0** | **0.036** | **real** |
| `TRI_equal_seed0` | 217 | 16 | 39 | 28 | +11 | 85.3 | +3.7 | 0.22 | not proven |
| `S_PS_seed0` | 215 | 15 | 40 | 30 | +10 | 85.0 | +3.3 | 0.28 | not proven |
| `LOO_noLLM_seed0` | 215 | 16 | 39 | 30 | +9 | 84.7 | +3.0 | 0.34 | not proven |
| `S_BP_seed0` | 207 | 15 | 40 | 38 | +2 | 82.3 | +0.7 | 0.91 | no effect |

(Every row adds to 300. Gain = net ÷ 300, e.g. TRI_equal: 11/300 = +3.7 points.)

**Why the same-looking rows get different verdicts.** Compare `S_LLM` (42 vs 24) with `TRI_equal`
(39 vs 28). Both rescue about 40 episodes — but `S_LLM` gives back only 24 while `TRI_equal` gives
back 28. That small difference in *lopsidedness* is what separates p = 0.036 from p = 0.22. A
39-vs-28 split is the kind of thing a coin produces 22% of the time, so we cannot rule out luck.
`S_BP` is the extreme case: 40 rescued and 38 lost is almost a perfect tie (p = 0.91) — no effect.

**The fix is more episodes, not a different method.** With 300 episodes the uncertainty on a success
rate is about ±4 points, which swallows a 3-point gain. **Test 2** in Part 6 scores each run on 900
episodes instead, shrinking that to about ±2.5 points — enough to decide these three rows.

### Finding 4 (the capstone) — the proposed method helps, but does not beat the single best teacher

`TRI_shadow` is the proposed method: **all three teachers, with competence weighting** (trust each
teacher where a short look-ahead shows it beats the policy). This is the run the whole study was built
to test. Its final 5M checkpoint was evaluated on **9 Oct 2026** (`jobs_final_TRI_shadow.sh`), and the
paired comparison below pools all **900 episodes** (the three seed blocks 42/1042/2042) for power.

| `TRI_shadow` vs | rescued | lost | net | McNemar p | verdict |
|---|---|---|---|---|---|
| `T0_baseline` (no teacher) | 120 | 87 | **+33** | **0.026** | **real — it beats plain MAPPO** |
| `S_LLM` (single best teacher) | 86 | 92 | **−6** | 0.71 | **tied — no gain over one LLM teacher** |
| `TRI_equal` (equal-weight ablation) | 108 | 83 | +25 | 0.082 | trend, not yet proven |

Success rate, mean over the 3 seeds, 4 aircraft: **`TRI_shadow` 85.8%**, `S_LLM` 86.4%, `TRI_equal`
83.0%, baseline 82.1%.

**Read honestly, three things are true:**
1. **It beats no-teacher** (p = 0.026). The competence-weighted teacher mix does improve on plain RL.
2. **It does *not* beat the single best teacher.** `TRI_shadow` and `S_LLM` are a statistical tie at
   ~86% (net −6 of 900, p = 0.71). The headline the study hoped for — *"combining complementary
   teachers beats the best single one"* — **is not supported on overall success rate.**
3. **It does beat naive averaging** (`TRI_equal`, +25 net), but only as a trend (p = 0.082), and
   `TRI_equal` is actually the weakest of the teacher runs (83.0%) — so *equal* trust in three teachers
   hurt, and the competence weighting is what pulled `TRI_shadow` back up into the top cluster.

**Why it tied `S_LLM` instead of beating it (the telling detail).** Look at the failure types
(counts over the 900 episodes). `TRI_shadow` entered storms **40** times — *exactly matching `S_LLM`'s
40* — but caused **75** collisions: **more than the single LLM teacher (61), and far more than the
rule-taught runs** (`S_PS` 23, `S_BP` 36). (It is not the single worst — `TRI_equal` at 91 and
`LOO_noBP` at 81 are higher — but it is nowhere near the rule teachers' conflict-avoidance.) So the
competence weighting leaned **LLM-ward**: `TRI_shadow` inherited the LLM teacher's storm avoidance but
**did not** inherit the rule teacher's conflict avoidance. It behaved like its LLM component rather
than blending the best of both — which is exactly why it matched, rather than beat, the single LLM
teacher.

**Where it does look best: dense traffic.** On the scalability sweep (seed 42), `TRI_shadow` holds up
best as aircraft are added, and clearly beats both the baseline and the equal ablation there:

| aircraft | `T0` | `TRI_equal` | `S_LLM` | `TRI_shadow` |
|---|---|---|---|---|
| 4 | 81.7 | 85.3 | 87.7 | 87.0 |
| 6 | 50.3 | 55.0 | 60.3 | **64.3** |
| 8 | 22.3 | 21.3 | 32.0 | **33.3** |

So the method's advantage shows up under *harder* conditions — at 6 and 8 aircraft it is tied-best —
even though at the easy 4-aircraft setting it only matches the best single teacher.

**The caveat that still governs everything: one training seed per method.** The top runs (`S_PS`,
`S_LLM`, `LOO_noPS`, `TRI_shadow`) all cluster at ~86%; the 0.5–3 point gaps between them are inside
the ±2–4 point seed/sampling noise. **Strongest honest claim:** the proposed method is in the top
cluster, significantly beats no-teacher, beats naive averaging, and scales best — but it does **not**
demonstrably beat the single best LLM teacher on base success. Closing that would need multiple
training seeds, and likely a competence rule that does not collapse toward the LLM on conflicts.

### The new story in one sentence
*Teaching MAPPO with different kinds of teachers improves safety in complementary ways — the LLM
teacher prevents storm entries, the rule-based teacher prevents aircraft conflicts, and no single
teacher does both; the taught runs cost no more to run than plain MAPPO, and we test how they hold up
as traffic grows using the AAMAS paper's own protocol.*

---

## Part 4 — The AAMAS paper, explained (you don't need to have read it)

**Source:** `new_approach/Manuscript_AAMAS 2025.pdf` (text extracted; numbers below are from its
Tables 4 and 6).

**What it is.** A 2025 paper solving almost our exact problem: several aircraft cross a region while
thunderstorms move through it; each aircraft has its own learned policy (their algorithm is called
**IDDPG**). Their setup matches ours closely — 5 nm separation, ±30° turns per step, 2 moving storms,
4 to 8 aircraft.

**Their report card — four measures, all counted per aircraft:**
1. **Aircraft-LOS rate** — how often an aircraft came within 5 nm of another. ("LOS" = loss of
   separation, the aviation term for "too close".)
2. **Storm-LOS rate** — how often an aircraft flew into a storm.
3. **Goal-reach rate** — what fraction of aircraft reached their exit point.
4. **Flight distance ratio** — how much longer the flown path was than the direct route (1.10 = 10%
   extra).

**Their headline test is scalability** — does it still work when the sky is crowded? Their numbers:

| | 4 aircraft | 8 aircraft |
|---|---|---|
| **IDDPG** (their method): aircraft-LOS / storm-LOS / goal-reach | 0% / 0% / 100% | 0% / 5% / 95% |
| **FMT** (their older, non-learning planner) | 5.2–7.3% / 0.05% / 100% | 16.8% / 0.05% / 93% |

They also report decision speed: IDDPG **<20 ms** per step; FMT up to **41 s**.

**Two cautions before comparing.** (a) Their simulator is not ours — real oceanic routes, 150-step
episodes vs our 60-step ones — so we must **never** claim to beat their absolute numbers. (b) They
count per *aircraft*; our `eval_last.json` counts per *episode* (all 4 must arrive), which makes us
look worse for identical behaviour. **The fair comparisons are:** how much our scores degrade from 4
to 8 aircraft next to how theirs did, and where we sit between IDDPG and FMT.

---

## Part 5 — The two settings (why the same tests are run twice)

The three still-training runs cannot be scored on `last_checkpoint.pt` — that file is rewritten every
few minutes, so a score from it could never be reproduced. For them we use the **newest frozen
checkpoint** instead. That creates a choice, so we run **both** options and keep them completely
separate.

| | **Setting 1 — "latest"** | **Setting 2 — "3M" ← PRIMARY** |
|---|---|---|
| Which checkpoint | finished runs → their 5M file; still-training runs → newest frozen file | every run → its ~3M file |
| Is it fair? | **No** for the still-training runs — `TRI_shadow` (3.28M) would face 5M rivals with **34% less training** | **Yes** — everyone at roughly the same step |
| What to use it for | comparing the **6 finished runs** with each other (all at 5M) | **every comparison involving the proposed method** |

**Why Setting 2 is the primary one:** it is the only setting where a difference can fairly be blamed
on the *method* instead of on unequal training. Small print we will print with every Setting-2 table:
the "~3M" checkpoints span **2,999,287–3,332,359 steps (10.5%)**, and `TRI_shadow` (3,280,741) has
**9.3% more** training than `T0_baseline` and `TRI_equal` (~3,000,300) — a slight advantage to the
proposed method, so a win must be read with that in mind and a loss is all the more convincing. We
checked 2.5M as an alternative: its spread is **13.3%**, i.e. worse. So ~3M is the best common point
available. *(Computed from the checkpoint filenames in each `Evaluation_distillation_approach/<run>/weights/` folder.)*

### The exact checkpoint each run uses — every file verified to exist

| Run | Setting 1 file | Setting 2 file |
|---|---|---|
| `T0_baseline_seed0` | `last_checkpoint.pt` | `checkpoint_3000307.pt` |
| `S_BP_seed0` | `last_checkpoint.pt` | `checkpoint_3332359.pt` |
| `S_PS_seed0` | `last_checkpoint.pt` | `checkpoint_3263303.pt` |
| `S_LLM_seed0` | `last_checkpoint.pt` | `checkpoint_3103544.pt` |
| `LOO_noLLM_seed0` | `last_checkpoint.pt` | `checkpoint_2999287.pt` |
| `TRI_equal_seed0` | `last_checkpoint.pt` | `checkpoint_3000458.pt` |
| `LOO_noBP_seed0` | `last_checkpoint.pt` | `checkpoint_3304236.pt` |
| `LOO_noPS_seed0` | `last_checkpoint.pt` | `checkpoint_3285558.pt` |
| `TRI_shadow_seed0` | `last_checkpoint.pt` | `checkpoint_3280741.pt` |

All paths are `Evaluation_distillation_approach/<run>/weights/<file>`. **Update (9 Oct 2026):** the
three distill runs that were still training when this table was first written — `LOO_noBP`,
`LOO_noPS`, and `TRI_shadow` — have since **all finished at 5M**, so their Setting-1 file is now
`last_checkpoint.pt` (the true final model), re-evaluated via `jobs_final_<run>.sh`. `TRI_shadow`'s
Setting-2 stays at its ~3M checkpoint (`checkpoint_3280741.pt`), so its two settings now differ as
intended (they were identical only while it was frozen mid-training). `LOO_noLLM`'s Setting-2 file is
713 steps *under* 3M; that is the boundary snapshot (its next one is 3.50M, which would over-train it
relative to the others).

---


## Part 5b — From the findings to the experiments: what we still need, and where each answer will land

Part 3 gave us two proven findings and one unproven one, all from **6 finished runs at 5M steps**.
Three things are still missing, and each planned test exists to supply exactly one of them:

| What is missing after Part 3 | The test that supplies it | The file that will hold the answer |
|---|---|---|
| The proposed method `TRI_shadow` is not in any table (it never finished, so it has no `eval_last.json`) | **Test 1** — score every run at a common ~3M checkpoint | `new_approach/eval_results/setting2_3M/TRI_shadow_seed0/evaluation.json` |
| Finding 3's three teachers are unproven (p = 0.2–0.3) because 300 episodes is too few | **Test 2** — two more sets of 300 episodes → 900 per run | `…/setting2_3M/<run>/evaluation_seed1042.json`, `…_seed2042.json` |
| We have no comparison against published work | **Test 3** — the AAMAS protocol at 8 / 6 / 4 aircraft | `…/setting2_3M/<run>/evaluation_scal_ac{8,6,4}.json` |
| (supporting) Does the teacher cost anything at run time? | **Test 4** — decision latency | `new_approach/eval_results/shared/<run>/latency_ac{4,8}.json` |
| (supporting) Do taught runs learn faster? | **Test 5** — earlier checkpoints | `new_approach/eval_results/shared/<run>/curve_<step>.json` |
| Are the new differences real or luck? | **Test 6** — the same paired method as Part 3, re-run automatically | `new_approach/eval_results/analysis/analysis_S2_base.md` |

Tests 1, 2 and 3 are run **twice** — once per setting (Part 5) — which is why each appears under both
`setting1_latest/` and `setting2_3M/`. Tests 4 and 5 do not depend on the setting, so they run once
and live in `shared/`.

### The `new_approach/` folder, and what each part is for

```text
new_approach/
├── evaluate_per_aircraft.py      the scorer that produces the AAMAS per-aircraft metrics (Test 3, 4)
├── analyze_pairwise.py           the comparison tool: builds the 2x2 tables and p-values (Test 6)
├── experiment_jobs/              one shell script per tmux session — the exact commands being run
│   ├── jobs_S1_baseB.sh          Setting 1: Tests 1+2   (30 jobs)
│   ├── jobs_S1_D1.sh / _D2.sh    Setting 1: Test 3      (15 + 12 jobs)
│   ├── jobs_S2_baseB.sh          Setting 2: Tests 1+2   (30 jobs)
│   ├── jobs_S2_D1.sh / _D2.sh    Setting 2: Test 3      (15 + 12 jobs)
│   ├── jobs_shared_E.sh          Test 4  (4 jobs, finished)
│   ├── jobs_shared_F.sh          Test 5  (35 jobs)
│   └── logs/<session>.log        every command that ran, timestamped
├── eval_results/                 ALL new results (nothing is written into training folders)
│   ├── setting2_3M/<run>/        PRIMARY setting — 9 run folders
│   ├── setting1_latest/<run>/    companion setting — the same 9 run folders, same filenames
│   ├── shared/<run>/             Tests 4 and 5 (setting-independent)
│   └── analysis/                 Test 6 reports, per setting
├── Manuscript_AAMAS 2025.pdf     the paper we benchmark against (Part 4)
└── LLM_MAPPO_Research_Documentation.docx   the write-up these results feed into
```

**The one rule for reading results:** the *setting* folder says which checkpoint was scored, the
*run* folder says which training run, and the *filename* says which test. So
`setting2_3M/TRI_shadow_seed0/evaluation_scal_ac8.json` = "the proposed method, at its ~3M
checkpoint, flown with 8 aircraft".

## Part 6 — The experiments, one at a time

Nothing here trains. Every command only scores existing checkpoints. Each runs in its own tmux
session, on the CPU, one job at a time.

**The lines every script starts with** (shown once — they are identical in all 8 scripts, which live
in `new_approach/experiment_jobs/jobs_<session>.sh`):
```bash
cd ".../multi-agent-isolated-llm-guidance"
export CUDA_VISIBLE_DEVICES=""      # CPU only -> never disturbs the 3 GPU training runs
PY=/home/aradhya.dhruv/anaconda3/envs/llmdrl/bin/python
EV=Evaluation_distillation_approach # trained weights live here (read-only)
OUT=new_approach/eval_results       # NEW results are written here
run() { echo "[job $(date +%H:%M:%S)] $*"; "$@"; }   # log each command with a timestamp, then run it
RUNS="TRI_shadow_seed0 S_LLM_seed0 TRI_equal_seed0 T0_baseline_seed0 S_PS_seed0 \
      LOO_noLLM_seed0 LOO_noBP_seed0 LOO_noPS_seed0 S_BP_seed0"   # seed 0 only, headline runs first
w1() { … }   # returns the Setting-1 file for a run (the table in Part 5)
w2() { … }   # returns the Setting-2 file for a run
```
*Why CPU only?* Scoring barely uses the GPU (the network is tiny; the slow part is the simulator,
which is CPU work), and the three training runs are on GPU 1 — so this costs nothing in speed and
never disturbs them. *Why `run`?* So each log line records exactly what ran, with a timestamp.

---

### Test 1 + 2 — Main score, then more episodes — sessions `S2_baseB` and `S1_baseB`
- **Do:** score all 9 runs on 300 episodes (the main table), write the comparison report, then repeat
  on two *further* sets of 300 episodes.
- **Why:** the main table says who is best; the extra sets shrink the error bars from ±4 points to
  about ±2.5, which is what decides the "not provable yet" gaps of Finding 3.
- **Tells us:** the headline ranking, and which differences are real rather than luck.
- **Time:** ~18 h per session (30 jobs); the main table lands after the first ~6 h.

**The command (this is `S2_baseB`, the primary setting; `S1_baseB` is identical with `w1` and
`setting1_latest`):**
```bash
# --- main score: 300 episodes, seed 42, 4 aircraft, 2 storms ---
for RUN in $RUNS; do
  run $PY evaluate_rl.py --model-path $EV/$RUN/weights/$(w2 $RUN) \
      --n-episodes 300 --num-agents 4 --num-weather-cells 2 --seed 42 \
      --metrics-path $OUT/setting2_3M/$RUN/evaluation.json
done
run $PY new_approach/analyze_pairwise.py --pattern "$OUT/setting2_3M/*/evaluation.json" \
    --baseline T0_baseline_seed0 --out $OUT/analysis/analysis_S2_base.md

# --- two further sets of 300 episodes ---
for SEED in 1042 2042; do
  for RUN in $RUNS; do
    run $PY evaluate_rl.py --model-path $EV/$RUN/weights/$(w2 $RUN) \
        --n-episodes 300 --num-agents 4 --num-weather-cells 2 --seed $SEED \
        --metrics-path $OUT/setting2_3M/$RUN/evaluation_seed$SEED.json
  done
  run $PY new_approach/analyze_pairwise.py --pattern "$OUT/setting2_3M/*/evaluation_seed$SEED.json" \
      --baseline T0_baseline_seed0 --out $OUT/analysis/analysis_S2_seed$SEED.md
done
```
**Why each choice:**
- `$(w2 $RUN)` picks the ~3M checkpoint from the Part-5 table; `$(w1 $RUN)` in `S1_baseB` picks the
  latest one. **This single difference is the only thing separating the two settings.**
- `--seed 42` fixes *which* 300 episodes are flown (scenarios 42–341) and is the same for every run —
  that is what makes the episode-by-episode comparison possible.
- `--seed 1042` / `2042` are two further, non-overlapping sets of 300 (1042–1341, 2042–2341).
  Pooling all three gives 900 episodes per run.
- `--num-agents 4 --num-weather-cells 2` = the conditions the runs were trained in; this test is
  about error bars, not difficulty.
- The analysis step uses `T0_baseline_seed0` as the yardstick because every teacher run was trained
  with the same seed 0 — the like-for-like partner.
- **Results:** `new_approach/eval_results/setting2_3M/<run>/evaluation.json` (and
  `evaluation_seed1042.json`, `evaluation_seed2042.json`); reports in
  `new_approach/eval_results/analysis/analysis_S2_base.md` etc.

---

### Test 3 — Traffic density (the AAMAS benchmark) — sessions `S2_D1`, `S2_D2`, `S1_D1`, `S1_D2`
- **Do:** fly each run with **8**, then **6**, then **4** aircraft, 300 episodes each, recording the
  paper's four per-aircraft measures.
- **Why:** this *is* the paper's headline test (Part 4). It shows whether the safety habits the
  teachers instilled survive when traffic grows — and lets us place ourselves between IDDPG and FMT.
- **Tells us:** our degradation from 4 to 8 aircraft, next to theirs.
- **Time:** ~10 h (`D1`, 5 runs) and ~8 h (`D2`, 4 runs), per setting.

**The command (`S2_D1`; the other three are the same with `w1`/`setting1_latest` or the `D2` group):**
```bash
# D1 group: TRI_shadow_seed0 S_LLM_seed0 TRI_equal_seed0 T0_baseline_seed0 S_PS_seed0
# D2 group: LOO_noLLM_seed0 LOO_noBP_seed0 LOO_noPS_seed0 S_BP_seed0
for AC in 8 6 4; do
  for RUN in $GROUP; do
    run $PY new_approach/evaluate_per_aircraft.py --model-path $EV/$RUN/weights/$(w2 $RUN) \
        --n-episodes 300 --num-agents $AC --num-weather-cells 2 --seed 42 \
        --metrics-path $OUT/setting2_3M/$RUN/evaluation_scal_ac$AC.json
  done
done
```
**Why each choice:**
- **A different scorer** — `new_approach/evaluate_per_aircraft.py` instead of `evaluate_rl.py` —
  because with 6–8 aircraft the "all must arrive" per-episode success collapses toward zero and hides
  what happened. This scorer records, for every aircraft, whether it arrived, came too close, or
  entered a storm, plus how far it flew versus its planned route: exactly the paper's four measures.
- **8 aircraft first** — the most informative and least certain case, so the full comparison table
  lands earliest.
- **Split into two sessions per setting** so the crowded results arrive in ~10 h rather than ~20 h;
  the 5 headline runs are in `D1`.
- `--n-episodes 300` — three times the paper's 100.
- **Cautions to print with the results:** our episodes are 60 steps (theirs 150), so time-outs are
  reported separately; and with 8 aircraft, 8 of our 12 routes are in use, so crossings are denser.
- **Results:** `new_approach/eval_results/setting2_3M/<run>/evaluation_scal_ac{8,6,4}.json` (fields
  `aircraft_los_rate`, `storm_los_rate`, `goal_reach_rate`, `flight_distance_ratio_mean`).

---

### Test 4 — Decision speed — session `shared_E` *(already finished)*
- **Do:** time one steering decision, in milliseconds, at 4 and at 8 aircraft.
- **Why:** the paper reports this (IDDPG <20 ms; FMT up to 41 s). It is also our selling point: the
  teachers — including the LLM — are used **only during training**, so the finished policy never
  calls an LLM and costs no more to run than plain MAPPO.
- **Time:** minutes. **Status: done.**

```bash
for AC in 4 8; do
  for RUN in T0_baseline_seed0 S_LLM_seed0; do
    run $PY new_approach/evaluate_per_aircraft.py --model-path $EV/$RUN/weights/$(w1 $RUN) \
        --n-episodes 20 --num-agents $AC --num-weather-cells 2 --seed 42 \
        --metrics-path $OUT/shared/$RUN/latency_ac$AC.json
  done
done
```
**Why each choice:** only **two runs**, because every run uses the *identical* network, so any two are
representative — one untaught and one taught proves the teacher adds zero runtime cost. Only **20
episodes**, because latency is measured per *decision* and 20 episodes contain hundreds of them
(ignore the safety numbers in these files; Test 3 has the real ones). Setting-independent, so it runs
once.
- **Results (exist now):** `new_approach/eval_results/shared/T0_baseline_seed0/latency_ac4.json`,
  `…/latency_ac8.json`, and the same two for `S_LLM_seed0` — field `mean_decision_ms`.

---

### Test 5 — Learning curves — session `shared_F`
- **Do:** score each run's *earlier* checkpoints (nearest 1.0, 1.5, 2.0 and 2.5M steps), 100 episodes
  each, and plot success against training step.
- **Why:** the point of teachers is to speed up learning. If a taught run reaches 80% success at 1.5M
  steps and the baseline only at 3M, that is a clear claim.
- **Tells us:** whether taught runs need fewer training steps.
- **Time:** ~9 h (35 jobs).

```bash
while read -r RUN STEP; do
  run $PY evaluate_rl.py --model-path $EV/$RUN/weights/checkpoint_$STEP.pt \
      --n-episodes 100 --num-agents 4 --num-weather-cells 2 --seed 42 \
      --metrics-path $OUT/shared/$RUN/curve_$STEP.json
done <<'LIST'
   (one "RUN STEP" pair per line — 35 in total; the full list is in
    new_approach/experiment_jobs/jobs_shared_F.sh)
LIST
```
**Why each choice:**
- The steps differ between runs (e.g. `LOO_noBP` sits at 1.30M, 1.80M…) because checkpoints were saved
  every 500,000 steps *counted from each restart*, and the training interruptions shifted every run's
  grid. We take each run's nearest checkpoint and plot it at its **true** step.
- **100 episodes** (not 300): 35 checkpoints × 300 would take days, and 100 is enough to see the
  *shape* of a curve, which is the claim.
- `--seed 42` — the same first 100 episodes for every checkpoint, so the points are comparable.
- The two end points are reused free: the ~3M point comes from **Test 1 in Setting 2**, the latest
  point from **Test 1 in Setting 1**.
- **Caveat to print alongside:** the teacher-based runs used ~10× more computer time per step, so
  "fewer training steps" is true but "cheaper overall" is not.
- **Results:** `new_approach/eval_results/shared/<run>/curve_<step>.json`.

---

### How all 8 sessions were started
```bash
for X in S1_baseB S1_D1 S1_D2 S2_baseB S2_D1 S2_D2 shared_E shared_F; do
  tmux new -d -s $X "bash new_approach/experiment_jobs/jobs_$X.sh 2>&1 \
      | tee new_approach/experiment_jobs/logs/$X.log"
done
```
Attach with `tmux a -t S2_baseB`. A session closes by itself when its last job finishes (that is why
`shared_E` is already gone) — the full record stays in its `.log` file.

**Checked before launching:** all 8 scripts pass a syntax check; a dry run expands to **153 jobs** and
every one of the 153 checkpoint paths exists; **no script reads a still-training run's
`last_checkpoint.pt`**; **no seed-1/seed-2 run is referenced**; and one-episode smoke tests passed for
a Setting-1 frozen checkpoint, a Setting-2 checkpoint at 8 aircraft, and an alternative episode seed.

---

## Part 7 — Where the results are stored

Everything new goes under `new_approach/eval_results/`. Nothing is written into the training folders.
The layout is **setting → run → files**:

```
new_approach/eval_results/
├── setting2_3M/        ← PRIMARY setting; one folder per run (9 folders)
│   └── S_LLM_seed0/
│       ├── evaluation.json               Test 1 — the main score
│       ├── evaluation_seed1042.json      Test 2
│       ├── evaluation_seed2042.json      Test 2
│       ├── evaluation_scal_ac8.json      Test 3 — 8 aircraft
│       ├── evaluation_scal_ac6.json      Test 3 — 6 aircraft
│       └── evaluation_scal_ac4.json      Test 3 — 4 aircraft
├── setting1_latest/    ← same 9 run folders, same file names, different checkpoints
├── shared/             ← Test 4 (latency_ac4/8.json) and Test 5 (curve_<step>.json)
└── analysis/           ← Test 1/2 reports: analysis_S1_*.md, analysis_S2_*.md
```

**How to find something:**

| You want | Open |
|---|---|
| The headline number for a run | `setting2_3M/<run>/evaluation.json` → `success_rate` |
| The same run, fully trained | `setting1_latest/<run>/evaluation.json` |
| Who beat the baseline, with p-values | `analysis/analysis_S2_base.md` |
| The AAMAS benchmark at 8 aircraft | `setting2_3M/<run>/evaluation_scal_ac8.json` → `aircraft_los_rate`, `storm_los_rate`, `goal_reach_rate`, `flight_distance_ratio_mean` |
| Decision speed | `shared/<run>/latency_ac4.json` → `mean_decision_ms` |
| Which command produced a file | `new_approach/experiment_jobs/logs/<session>.log` |

---

## Part 8 — Keeping the conclusion honest

1. **Same episodes for everyone.** Every run always flies the identical episode sets, so every
   comparison is episode by episode.
2. **More episodes = smaller error bars.** Report the error bar every time; never claim a gap the
   error bars do not support.
3. **One training seed per method** — the biggest limitation. Every run in this study is seed 0,
   so we cannot separate "this method is better" from "this training run got lucky"; past experience
   on this project suggests training luck can move a result by more than the 2–6 point differences we
   are measuring. Lead with Finding 1 (a large, mechanistic effect); treat the overall ranking as
   weak evidence; say plainly that multi-seed training is the obvious next step.
4. **Never mix the settings.** Report Setting 1 and Setting 2 in separate tables, and use Setting 2
   for anything involving `TRI_shadow`, `LOO_noBP` or `LOO_noPS`.

---

## Part 9 — What we will be able to say at the end

1. *Different teachers prevent different failures; none is best at both.* — already proven (Finding 1).
2. *The LLM teacher significantly improves plain MAPPO.* — already proven (Finding 2); the other
   teachers are settled by Test 2.
3. *Does the competence-weighted combination beat the best single teacher?* — Tests 1–2 in **Setting 2**
   (the only fair comparison for it).
4. *Do taught runs learn in fewer steps?* — Test 5, with the compute caveat.
5. *On the AAMAS scalability protocol, our runs degrade [less than / about the same as] the untaught
   one, sit between IDDPG and FMT on safety, and decide in milliseconds.* — Tests 3 and 4.

**If Test 3 shows the teachers give no advantage at 8 aircraft**, the story still stands on points 1,
2 and 4, and we write: *the safety gains hold at the traffic level the runs were trained on but do not
transfer to denser traffic — likely because the policy watches only its three nearest neighbours and
episodes are short — which points to density-aware teaching as the next step.* That is a legitimate,
useful result, not a failure.

---

## Part 10 — Current status (11 Oct 2026) — **study complete**

- **All training complete.** Every run reached 5M; `TRI_shadow_seed0` finished at **5,000,009** steps.
- **All Setting-1 evaluation complete.** `final_TRI_shadow` finished on 9 Oct at 12:14 — all six
  output files in `setting1_latest/TRI_shadow_seed0/` are on the true final checkpoint
  (`last_checkpoint.pt`), replacing the earlier frozen 3.78M snapshot. All 9 seed-0 runs now have
  final Setting-1 numbers (3 seed blocks + 3 scalability blocks each).
- **Headline verdict is in — see Finding 4 (Part 3).** The proposed competence-weighted method
  (`TRI_shadow`) **significantly beats plain MAPPO** (p = 0.026) and **beats naive averaging**
  (`TRI_equal`, trend, p = 0.082), but it is a **statistical tie with the single best teacher**
  (`S_LLM`, p = 0.71). It scales best in dense traffic (tied-best at 6 and 8 aircraft). The telling
  detail: it inherited the LLM teacher's storm avoidance (40 storm entries, same as `S_LLM`) but a high
  collision count (75 — more than `S_LLM`'s 61 and far above the rule-taught runs), i.e. the weighting
  leaned LLM-ward rather than blending both teachers.
- **Governing caveat unchanged:** one training seed per method; the top four runs cluster at ~86%
  inside ±2–4 points of noise. Firming up "combining beats the single best teacher" would need
  multiple training seeds and a competence rule that doesn't collapse toward the LLM on conflicts.
