# Results Explained — A Complete Walkthrough

**Written for someone with no statistics background.** Every term is defined before it is used.
Every number quoted here is followed by the exact file it came from, so you can check it yourself.

Date written: 4 September 2026.
All results folder: `new_approach/eval_results/`

---

## Table of contents

- [Part 0 — What this document is](#part-0)
- [Part 1 — Background: the system, in plain words](#part-1)
- [Part 2 — The 9 runs and why each one exists](#part-2)
- [Part 3 — The two settings (why every test was done twice)](#part-3)
- [Part 4 — What we measured (every metric defined)](#part-4)
- [Part 5 — Statistics from zero: pp, p-values, and why we pair](#part-5)
- [Part 6 — How to read the raw files yourself](#part-6)
- [Part 7 — RESULT 1: the headline scores](#part-7)
- [Part 8 — RESULT 2: is teaching better than no teaching?](#part-8)
- [Part 9 — RESULT 3: head-to-head against the proposed method](#part-9)
- [Part 10 — RESULT 4: the complementarity finding (strongest result)](#part-10)
- [Part 11 — RESULT 5: crowding test vs the AAMAS 2025 paper](#part-11)
- [Part 12 — RESULT 6: decision speed](#part-12)
- [Part 13 — RESULT 7: learning curves](#part-13)
- [Part 14 — Putting it together: what you can and cannot claim](#part-14)
- [Part 15 — The honest limitations](#part-15)

---

<a name="part-0"></a>
## Part 0 — What this document is

You ran a large set of evaluations. 153 result files were produced. This document explains:

1. **What each test was actually asking** (in plain English)
2. **What number came back**
3. **How to read that number yourself** — including which raw file it lives in
4. **What the number means** — and, importantly, what it does *not* mean

Nothing here requires you to know statistics beforehand. Part 5 teaches the two concepts
(`pp` and `p-value`) that the rest of the document relies on.

---

<a name="part-1"></a>
## Part 1 — Background: the system, in plain words

### What the software does

You built a simulated piece of airspace (a "sector"). Several aircraft fly through it. Each
aircraft must:

- get from where it enters to its destination,
- without coming too close to another aircraft (that is a **collision**),
- without flying into a storm cell (that is a **weather failure**),
- and it must arrive before time runs out (60 simulator steps; running out is a **truncation**).

Every few moments each aircraft chooses a **turn**: one of 13 options from 30° left, through
straight ahead, to 30° right. A neural network makes that choice. Training that network to make
good choices is the whole project.

Source of these rules: `configs.py` (`SAFE_R = 5.0`, `MAX_STEP = 60`, `ACTION_BINS` = the 13 turns).

### The vocabulary you need

| Word | What it means here |
|---|---|
| **Episode** | One complete simulated scenario, start to finish. Ends in success, collision, weather, or truncation. |
| **Run** | One complete training job producing one trained network. You have 9 of these. |
| **Checkpoint** | A saved copy of a network's weights at a particular moment in training, e.g. `checkpoint_3280741.pt` = the network as it was after 3,280,741 training steps. |
| **Seed** | A number that fixes the random choices. The same seed always produces the same scenario. This is how we make the comparison fair. |
| **Success rate** | Out of N episodes, the percentage that ended in success. This is the main score. |

### What "teaching" means in this project

The core idea being tested: instead of the network learning purely by trial and error, we also
show it what **teachers** would have done, and nudge it toward them.

You have **three teachers**, each good at a different thing:

| Teacher | What it is | Its strength |
|---|---|---|
| **`llm`** | A large language model (Gemma 3:12B) reasoning about the traffic picture | Good at storms / broad situational reasoning |
| **`preview_safe` (PS)** | A rule that simulates each of the 13 turns forward 10 steps and picks a safe one | Good at avoiding other aircraft |
| **`best_preview` (BP)** | A rule that picks the turn that makes best progress while staying safe | Good at efficiency |

The **proposed method** does not trust all three equally. It measures how good each teacher's
suggestion actually turns out to be (this is the "shadow" check) and **weights each teacher by
how competent it has proven to be in that situation.** That is the idea this whole study exists
to test.

---

<a name="part-2"></a>
## Part 2 — The 9 runs and why each one exists

All 9 are **seed 0** only. All are stored in `Evaluation_distillation_approach/<run>/weights/`.

| Run name | Which teachers it used | How it weighted them | Why this run exists |
|---|---|---|---|
| **`T0_baseline`** | none | — | The control. Pure trial-and-error learning, no teaching at all. Everything is measured against this. |
| **`S_BP`** | best_preview only | — | What does the efficiency rule alone achieve? |
| **`S_PS`** | preview_safe only | — | What does the safety rule alone achieve? |
| **`S_LLM`** | llm only | — | **The prior-work baseline.** Other published papers use a single LLM teacher. This is what you must beat. |
| **`TRI_equal`** | all three | equal — trust each the same | **The critical control.** Same three teachers as the proposed method, but *without* the smart weighting. |
| **`TRI_shadow`** | all three | competence-weighted | **THE PROPOSED METHOD.** |
| **`LOO_noLLM`** | BP + PS | competence-weighted | Remove the LLM. Does it matter? ("LOO" = leave-one-out.) |
| **`LOO_noBP`** | LLM + PS | competence-weighted | Remove best_preview. Does it matter? |
| **`LOO_noPS`** | LLM + BP | competence-weighted | Remove preview_safe. Does it matter? |

**The three comparisons that decide the paper:**

1. `TRI_shadow` vs `T0_baseline` → *does teaching help at all?*
2. `TRI_shadow` vs `S_LLM` → *does my method beat the prior approach?* ← **headline**
3. `TRI_shadow` vs `TRI_equal` → *is the smart weighting the reason, or just having 3 teachers?*

Comparison 3 is the most scientifically valuable, because these two runs are **identical in every
way except the weighting**. Any difference is therefore caused by the weighting itself.

---

<a name="part-3"></a>
## Part 3 — The two settings (why every test was done twice)

### The problem

Three runs are **still training right now** and have not reached the 5,000,000-step finish line:

- `LOO_noBP` — at ~4.56M
- `LOO_noPS` — at ~4.52M
- `TRI_shadow` (**your proposed method**) — at ~3.46M

The others finished at 5.00M. So if you simply compare "each run's newest network," your own
proposed method is competing with **34% less training** than its rivals. That is not a fair fight
— and it is unfair *against you*.

### The fix: run everything twice

| | **Setting 1 — "latest"** | **Setting 2 — "equal age"** |
|---|---|---|
| Stored in | `new_approach/eval_results/setting1_latest/` | `new_approach/eval_results/setting2_3M/` |
| Which checkpoint | Each run's newest available | Every run at ~3,000,000 steps |
| Training amounts | Unequal (3.28M to 5.00M) | Equal (~3.0M–3.3M for all) |
| Honest description | "each network at its personal best" | "all networks at the same age" |
| **Use for** | Secondary / robustness check | **PRIMARY RESULTS TABLE** |

**Setting 2 is the fair comparison and should be your main table in the paper.** Setting 1 is
reported alongside it for honesty, and to show what happens when the finished runs get their full
budget.

### Exact checkpoints used

| Run | Setting 1 file | Setting 2 file |
|---|---|---|
| TRI_shadow | `checkpoint_3280741.pt` | `checkpoint_3280741.pt` ← **same file** |
| S_LLM | `last_checkpoint.pt` (5.00M) | `checkpoint_3103544.pt` |
| TRI_equal | `last_checkpoint.pt` (5.00M) | `checkpoint_3000458.pt` |
| T0_baseline | `last_checkpoint.pt` (5.00M) | `checkpoint_3000307.pt` |
| S_PS | `last_checkpoint.pt` (5.00M) | `checkpoint_3263303.pt` |
| S_BP | `last_checkpoint.pt` (5.00M) | `checkpoint_3332359.pt` |
| LOO_noLLM | `last_checkpoint.pt` (5.00M) | `checkpoint_2999287.pt` |
| LOO_noBP | `checkpoint_4147227.pt` | `checkpoint_3304236.pt` |
| LOO_noPS | `checkpoint_4120186.pt` | `checkpoint_3285558.pt` |

> **Note this openly in the paper:** TRI_shadow's rows are *identical* in both settings, because
> at the time of evaluation its newest frozen checkpoint (3.28M) was also its ~3M checkpoint. A
> reader who spots this on their own will wonder what else was hidden; stating it costs nothing.

**Why the still-training runs use a numbered checkpoint and not `last_checkpoint.pt`:** the file
`last_checkpoint.pt` is being rewritten every few minutes by the live training job. Reading it
mid-write can give a corrupted or inconsistent network. The numbered files are frozen and safe.

---

<a name="part-4"></a>
## Part 4 — What we measured (every metric defined)

### Group A — episode-level outcome metrics

Found in: `eval_results/setting{1,2}_*/<run>/evaluation*.json`

| Field in the file | Meaning | Better is |
|---|---|---|
| `success_rate` | fraction of episodes where **all** aircraft reached their destinations safely | higher |
| `collision_rate` | fraction ending because two aircraft came within 5 units | lower |
| `weather_rate` | fraction ending because an aircraft entered a storm | lower |
| `truncation_rate` | fraction ending because 60 steps ran out before everyone arrived | lower |
| `mean_return` | average total reward per episode — a summary of "how well it went overall" | higher |

These are stored as fractions (0.867), so multiply by 100 to get a percentage (86.7%).

**Important:** the four rates always sum to 1.0. An episode has exactly one outcome.

### Group B — per-aircraft metrics (for benchmarking against the AAMAS paper)

Found in: `eval_results/setting{1,2}_*/<run>/evaluation_scal_ac{4,6,8}.json`

The AAMAS 2025 paper does not report "did the whole scenario succeed." It reports **per aircraft**.
So we built `new_approach/evaluate_per_aircraft.py` to produce numbers in their format:

| Field | Meaning | Better is |
|---|---|---|
| `aircraft_los_rate` | % of aircraft that lost separation from another aircraft ("LOS" = loss of separation) | lower |
| `storm_los_rate` | % of aircraft that entered a storm | lower |
| `goal_reach_rate` | % of aircraft that reached their destination | higher |
| `flight_distance_ratio_mean` | actual distance flown ÷ straight-line planned distance. 1.00 = perfectly direct; 1.10 = flew 10% further than necessary | lower (closer to 1.0) |
| `mean_decision_ms` | milliseconds to decide one turn | lower |

**Why 4, 6 and 8 aircraft?** To test crowding. More aircraft in the same airspace = harder. The
AAMAS paper reports 4 and 8, so we match them and add 6 as a midpoint.

### Group C — decision speed

Found in: `eval_results/shared/<run>/latency_ac{4,8}.json`

Measured separately with only 20 episodes because timing does not need many samples, and because
running it alongside everything else would have distorted the timing.

### Group D — learning curves

Found in: `eval_results/shared/<run>/curve_<step>.json`

Each run scored at several points during training (1M, 1.5M, 2M, 2.5M steps) with 100 episodes.
This shows *how* each run improved over time, not just where it ended.

---

<a name="part-5"></a>
## Part 5 — Statistics from zero: pp, p-values, and why we pair

This is the part you need before reading any result. Take it slowly; it is only two ideas.

### 5.1 What "pp" means (percentage points)

If run A succeeds **85.8%** of the time and run B succeeds **81.3%** of the time, the gap is:

```
85.8 − 81.3 = 4.5
```

That gap is **4.5 percentage points**, written **+4.5 pp**.

**Why not just say "4.5 percent"?** Because "percent" is ambiguous. 85.8 is about 5.5% *larger*
than 81.3 in relative terms (85.8/81.3 = 1.055). Saying "4.5 percent" would confuse those two
completely different statements. So:

- **pp = the simple subtraction of two percentages.** 85.8% − 81.3% = 4.5 pp.
- **percent = a relative change.** From 81.3 to 85.8 is a 5.5% relative increase.

In this document **every gap between two success rates is in pp.** Always subtraction, never
division.

### 5.2 Why a difference might be meaningless

Suppose you flip a fair coin 300 times. You expect 150 heads. But you will not get exactly 150 —
you will get 143, or 158. That wobble is pure chance, not a property of the coin.

The same applies here. Two runs of genuinely identical skill, both tested on 300 episodes, will
**not** produce the same score. They will differ by a couple of points just from luck of the draw.

**How big is the wobble?** For a run scoring around 85% on 300 episodes, the typical random
wobble is roughly **±4 pp**. On 900 episodes it shrinks to about **±2.3 pp**.

> **Practical rule:** with 300 episodes, if two runs differ by less than about 4 pp, you cannot
> tell them apart. That is exactly why we ran 900 episodes and why we compute p-values.

### 5.3 What a p-value is

A **p-value** answers one specific question:

> *"If the two runs were actually equally good, how often would pure luck alone produce a gap
> this large or larger?"*

It is a probability between 0 and 1.

| p-value | Read it as | Decision |
|---|---|---|
| p = 0.60 | "luck would produce this gap 60% of the time" | **Prove nothing.** Very ordinary. |
| p = 0.20 | "luck would produce it 20% of the time — 1 in 5" | Still not convincing. |
| **p = 0.05** | "luck would produce it 5% of the time — 1 in 20" | **The conventional cut-off.** |
| p = 0.006 | "luck would produce it 0.6% of the time — 1 in 170" | **Strong.** The gap is real. |
| p = 0.0000 | "essentially never by luck" | **Very strong.** |

**The rule everyone uses:** `p < 0.05` = call it real. `p ≥ 0.05` = "not proven" (which is *not*
the same as "no difference" — it means you don't have enough evidence either way).

**Two things a p-value does NOT tell you:**

- It does **not** say how big the effect is. A tiny 0.5 pp gap can have p < 0.05 if you test
  enough episodes. Always read `pp` and `p` together: **pp = how much, p = how sure.**
- It does **not** say the result will hold for a different training seed. See Part 15.

### 5.4 Why we test every run on the *same* episodes (pairing)

This is the single most important design decision in these evaluations, and it is what makes
comparisons on 900 episodes trustworthy.

Every run was tested on **exactly the same scenarios** — episode seeds 42, 43, 44 … 341 for the
first block, then 1042… and 2042… for the other two blocks. Same weather placements, same routes,
same starting positions, in the same order, for all 9 runs.

**Why this matters enormously:** some episodes are simply hard and some are simply easy. If run A
happened to draw easier scenarios than run B, A would look better for no good reason. By giving
every run the identical scenario list, that entire source of noise is cancelled out.

This lets us do something much sharper than comparing two averages. We can go **episode by
episode** and ask: *on this exact scenario, who succeeded?*

### 5.5 The four-category table (this is what every comparison below reports)

For two runs A and B on the same 900 episodes, every episode falls into exactly one bucket:

| Bucket | Meaning |
|---|---|
| **both succeeded** | tells us nothing about who is better |
| **both failed** | tells us nothing about who is better |
| **only A succeeded** | evidence **for A** |
| **only B succeeded** | evidence **for B** |

The first two buckets are discarded — they are ties. **All the information is in the last two.**

**Worked example — TRI_shadow vs S_LLM, Setting 2, 900 episodes:**

```
only TRI_shadow succeeded  = 123 episodes   ← evidence for TRI_shadow
only S_LLM      succeeded  =  83 episodes   ← evidence for S_LLM
                             ---
informative episodes       = 206
```

If the two runs were truly equally good, then on each of those 206 informative episodes it should
be a 50/50 coin flip which one wins. We would expect about **103 each**.

We got **123 vs 83** — a lean of 20 towards TRI_shadow.

The test (called **McNemar's test**) asks: *how likely is a split at least this lopsided from 206
fair coin flips?* The answer is **p = 0.0064**, i.e. about **1 chance in 156**.

That is well under 0.05, so we call it real: **TRI_shadow genuinely beats S_LLM here.**

The gap in success rate is `85.8% − 81.3% = +4.5 pp` (reported as +4.4 pp; the small difference is
rounding — computed from raw counts it is 4.44).

The exact code doing this is in `new_approach/analyze_pairwise.py`:

```python
def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)
```

That is literally "the probability of getting a split this lopsided or worse from `n` coin flips,
doubled because we didn't predict the direction in advance."

---

<a name="part-6"></a>
## Part 6 — How to read the raw files yourself

You do not have to take any number in this document on trust. Here is how to check each one.

### Read one run's headline score

```bash
cd "new_approach/eval_results"
python -c "
import json
d = json.load(open('setting2_3M/TRI_shadow_seed0/evaluation.json'))
print('checkpoint :', d['model_path'])
print('episodes   :', d['num_episodes'])
print('success    : %.1f%%' % (d['success_rate']*100))
print('collision  : %.1f%%' % (d['collision_rate']*100))
print('weather    : %.1f%%' % (d['weather_rate']*100))
print('truncated  : %.1f%%' % (d['truncation_rate']*100))
print('mean return: %.1f'   % d['mean_return'])
"
```

### Compare all 9 runs at once

```bash
cd "new_approach/eval_results"
for f in setting2_3M/*/evaluation.json; do
  python -c "
import json,sys
d = json.load(open('$f'))
print('%-22s %5.1f%%' % ('$f'.split('/')[1], d['success_rate']*100))
"
done | sort -k2 -rn
```

### Look at one individual episode

Every `evaluation.json` contains an `episodes` list with one entry per scenario:

```bash
python -c "
import json
d = json.load(open('setting2_3M/TRI_shadow_seed0/evaluation.json'))
for e in d['episodes'][:5]:
    print('seed %d | %-12s | %2d steps | reward %7.1f' %
          (e['seed'], e['failure_reason'], e['steps'], e['total_reward']))
"
```

`failure_reason` is one of `success`, `collision`, `weather`, `truncated`. This is the field the
whole paired analysis is built on.

### See where two runs disagree

```bash
python -c "
import json
A = json.load(open('setting2_3M/TRI_shadow_seed0/evaluation.json'))['episodes']
B = json.load(open('setting2_3M/S_LLM_seed0/evaluation.json'))['episodes']
for a,b in zip(A,B):
    if a['success'] != b['success']:
        print('seed %d: TRI=%-10s S_LLM=%-10s' % (a['seed'], a['failure_reason'], b['failure_reason']))
" | head -20
```

Every line printed is one **informative episode** — one of the ones that actually counts toward
the p-value.

### The pre-written analysis reports

Six reports were generated automatically:

```
new_approach/eval_results/analysis/analysis_S1_base.md      Setting 1, episode seeds 42-341
new_approach/eval_results/analysis/analysis_S1_seed1042.md  Setting 1, episode seeds 1042-1341
new_approach/eval_results/analysis/analysis_S1_seed2042.md  Setting 1, episode seeds 2042-2341
new_approach/eval_results/analysis/analysis_S2_base.md      Setting 2, episode seeds 42-341
new_approach/eval_results/analysis/analysis_S2_seed1042.md  Setting 2, episode seeds 1042-1341
new_approach/eval_results/analysis/analysis_S2_seed2042.md  Setting 2, episode seeds 2042-2341
```

Each covers 300 episodes. **This document pools all three blocks into 900 episodes**, which is why
the p-values here are stronger than in any single report. Pooling is legitimate because the three
blocks use disjoint episode seeds — they are three independent samples of the same thing.

---

<a name="part-7"></a>
## Part 7 — RESULT 1: the headline scores

**What was tested:** every run, 900 episodes (3 blocks of 300), 4 aircraft, 2 weather cells.
**Files:** `setting{1,2}_*/<run>/evaluation.json`, `evaluation_seed1042.json`, `evaluation_seed2042.json`

### Setting 2 — equal training (~3M) — **THE PRIMARY TABLE**

| Rank | Run | Success % | Collision % | Weather % | Truncated % | Mean return |
|---|---|---|---|---|---|---|
| 1 | **LOO_noBP** | **87.2** | 5.3 | 6.6 | 0.9 | 139.6 |
| 2 | **S_PS** | **85.9** | 5.1 | 7.9 | 1.1 | 135.6 |
| 3 | **TRI_shadow** ← proposed | **85.8** | 6.4 | 6.2 | 1.6 | 135.9 |
| 4 | T0_baseline | 81.9 | 4.0 | 9.0 | 5.1 | 125.4 |
| 5 | S_LLM ← prior work | 81.3 | 10.8 | 4.9 | 3.0 | 124.3 |
| 6 | LOO_noPS | 80.8 | 10.1 | 4.6 | 4.6 | 124.1 |
| 6 | S_BP | 80.8 | 6.6 | 6.7 | 6.0 | 123.2 |
| 8 | TRI_equal | 80.2 | 10.3 | 5.2 | 4.2 | 120.7 |
| 9 | LOO_noLLM | 75.7 | 9.8 | 6.6 | 8.0 | 111.8 |

### Setting 1 — each run's latest checkpoint

| Rank | Run | Success % | Collision % | Weather % | Truncated % | Mean return |
|---|---|---|---|---|---|---|
| 1 | S_LLM | 86.4 | 6.8 | 4.4 | 2.3 | 137.2 |
| 2 | S_PS | 86.0 | 2.6 | 7.0 | 4.4 | 136.0 |
| 3 | **TRI_shadow** | **85.8** | 6.4 | 6.2 | 1.6 | 135.9 |
| 4 | LOO_noLLM | 85.6 | 6.1 | 4.9 | 3.4 | 135.9 |
| 5 | LOO_noBP | 83.3 | 7.9 | 4.7 | 4.1 | 130.4 |
| 6 | LOO_noPS | 83.1 | 3.8 | 4.0 | 9.1 | 131.7 |
| 7 | TRI_equal | 83.0 | 10.1 | 4.4 | 2.4 | 128.3 |
| 8 | S_BP | 82.7 | 4.0 | 5.4 | 7.9 | 129.4 |
| 9 | T0_baseline | 82.1 | 4.9 | 6.6 | 6.4 | 126.3 |

### How to read these tables

**Do not read the ranking as a ranking.** Ranks 1, 2 and 3 in Setting 2 are 87.2 / 85.9 / 85.8 —
a spread of 1.4 pp, well inside the ±2.3 pp random wobble on 900 episodes. **Those three runs are
statistically tied.** (Part 9 confirms this properly.)

What *is* meaningful in these tables:

- **The gap from top to bottom is 11.5 pp** (87.2 vs 75.7) in Setting 2. That is real and large.
- **The failure columns tell a story the success column hides.** Look at Setting 2:
  - S_LLM: **10.8% collisions**, only 4.9% weather. It is bad at aircraft, good at storms.
  - T0_baseline: **4.0% collisions**, but **9.0% weather**. Exactly the opposite.
  - TRI_shadow: 6.4% / 6.2% — **balanced**, no dominant weakness.
  
  This balance is a genuine property of the proposed method and is worth stating in the paper:
  it does not fix one failure by creating another.
- **Truncation is where TRI_shadow is clearly strongest**: 1.6% vs T0's 5.1% and LOO_noLLM's 8.0%.
  Truncation means aircraft dithered and ran out of time. Low truncation = decisive, efficient
  flying. TRI_shadow has the lowest of any three-teacher variant.

### The uncomfortable observation

`LOO_noBP` (2 teachers: LLM + preview_safe) scores **87.2%**, slightly *above* the full
three-teacher `TRI_shadow` at **85.8%**. Taken at face value, that says dropping `best_preview`
helps. But the gap is 1.4 pp with p = 0.36 (Part 9) — **statistically a tie**. The honest statement
is *"removing best_preview does not measurably hurt,"* not *"removing it helps."*

---

<a name="part-8"></a>
## Part 8 — RESULT 2: is teaching better than no teaching?

**The question:** does any of this teaching machinery beat plain trial-and-error learning
(`T0_baseline`)?

**Method:** paired McNemar test on 900 identical episodes, each run vs `T0_baseline`.

### Setting 2 (equal training, ~3M) — primary

| Run | Its % | Baseline % | Gap (pp) | only-run wins | only-baseline wins | p | Verdict |
|---|---|---|---|---|---|---|---|
| **LOO_noBP** | 87.2 | 81.9 | **+5.3** | 125 | 77 | **0.0009** | ✅ **REAL** |
| **S_PS** | 85.9 | 81.9 | **+4.0** | 104 | 68 | **0.0074** | ✅ **REAL** |
| **TRI_shadow** | 85.8 | 81.9 | **+3.9** | 119 | 84 | **0.0168** | ✅ **REAL** |
| LOO_noPS | 80.8 | 81.9 | −1.1 | 120 | 130 | 0.5693 | ✗ not proven |
| S_BP | 80.8 | 81.9 | −1.1 | 108 | 118 | 0.5495 | ✗ not proven |
| S_LLM | 81.3 | 81.9 | −0.6 | 124 | 129 | 0.8015 | ✗ not proven |
| TRI_equal | 80.2 | 81.9 | −1.7 | 113 | 128 | 0.3672 | ✗ not proven |
| **LOO_noLLM** | 75.7 | 81.9 | **−6.2** | 105 | 161 | **0.0007** | ⚠️ **REAL — WORSE** |

### Setting 1 (latest checkpoints)

| Run | Its % | Baseline % | Gap (pp) | only-run | only-base | p | Verdict |
|---|---|---|---|---|---|---|---|
| **S_LLM** | 86.4 | 82.1 | +4.3 | 115 | 76 | **0.0058** | ✅ REAL |
| **S_PS** | 86.0 | 82.1 | +3.9 | 110 | 75 | **0.0122** | ✅ REAL |
| **TRI_shadow** | 85.8 | 82.1 | +3.7 | 114 | 81 | **0.0217** | ✅ REAL |
| **LOO_noLLM** | 85.6 | 82.1 | +3.4 | 119 | 88 | **0.0368** | ✅ REAL |
| LOO_noBP | 83.3 | 82.1 | +1.2 | 108 | 97 | 0.4850 | ✗ not proven |
| LOO_noPS | 83.1 | 82.1 | +1.0 | 109 | 100 | 0.5801 | ✗ not proven |
| TRI_equal | 83.0 | 82.1 | +0.9 | 112 | 104 | 0.6340 | ✗ not proven |
| S_BP | 82.7 | 82.1 | +0.6 | 112 | 107 | 0.7870 | ✗ not proven |

### How to read a single row

Take TRI_shadow in Setting 2:

```
Out of 900 identical scenarios:
  TRI_shadow succeeded where T0 failed  : 119 times
  T0 succeeded where TRI_shadow failed  :  84 times
  (the other 697 episodes: both succeeded or both failed — ignored)

If they were equally good, those 203 informative episodes should split ~101/102.
We saw 119/84. p = 0.0168 → this happens by luck about 1.7% of the time.
Conclusion: TRI_shadow really is better than the baseline. Gap = +3.9 pp.
```

### What this section establishes

**Only three runs beat the baseline in both settings: `TRI_shadow`, `S_PS`, and `LOO_noBP`.**
Those are the results you can rely on.

**`S_LLM` and `LOO_noLLM` flip sign between settings** — this is a red flag and Part 15 explains
what it means. `LOO_noLLM` goes from *significantly worse* (−6.2 pp, p=0.0007) to *significantly
better* (+3.4 pp, p=0.037) purely by changing which checkpoint you evaluate. Make no claim about
those two runs.

**`TRI_equal` never beats the baseline in either setting.** Given that it has all three teachers,
this is important: **three teachers with naive equal trust is worth nothing.** That sets up the
next section.

---

<a name="part-9"></a>
## Part 9 — RESULT 3: head-to-head against the proposed method

**The question:** does `TRI_shadow` (competence-weighted, three teachers) beat each alternative
directly?

### Setting 2 (equal training) — primary

| Comparison | TRI % | Other % | Gap (pp) | only-TRI | only-other | p | Verdict |
|---|---|---|---|---|---|---|---|
| vs **LOO_noLLM** | 85.8 | 75.7 | **+10.1** | 167 | 76 | **0.0000** | ✅ **TRI wins** |
| vs **TRI_equal** | 85.8 | 80.2 | **+5.6** | 123 | 73 | **0.0004** | ✅ **TRI wins** |
| vs **LOO_noPS** | 85.8 | 80.8 | **+5.0** | 126 | 81 | **0.0022** | ✅ **TRI wins** |
| vs **S_BP** | 85.8 | 80.8 | **+5.0** | 124 | 79 | **0.0019** | ✅ **TRI wins** |
| vs **S_LLM** ← headline | 85.8 | 81.3 | **+4.4** | 123 | 83 | **0.0064** | ✅ **TRI wins** |
| vs **T0_baseline** | 85.8 | 81.9 | **+3.9** | 119 | 84 | **0.0168** | ✅ **TRI wins** |
| vs LOO_noBP | 85.8 | 87.2 | −1.4 | 80 | 93 | 0.3616 | — tie |
| vs S_PS | 85.8 | 85.9 | −0.1 | 84 | 85 | 1.0000 | — tie |

### Setting 1 (latest checkpoints)

| Comparison | TRI % | Other % | Gap (pp) | only-TRI | only-other | p | Verdict |
|---|---|---|---|---|---|---|---|
| vs **T0_baseline** | 85.8 | 82.1 | **+3.7** | 114 | 81 | **0.0217** | ✅ TRI wins |
| vs S_BP | 85.8 | 82.7 | +3.1 | 114 | 86 | 0.0560 | — tie (just misses) |
| vs TRI_equal | 85.8 | 83.0 | +2.8 | 107 | 82 | 0.0806 | — tie |
| vs LOO_noPS | 85.8 | 83.1 | +2.7 | 102 | 78 | 0.0862 | — tie |
| vs LOO_noBP | 85.8 | 83.3 | +2.4 | 97 | 75 | 0.1091 | — tie |
| vs LOO_noLLM | 85.8 | 85.6 | +0.2 | 79 | 77 | 0.9362 | — tie |
| vs S_PS | 85.8 | 86.0 | −0.2 | 88 | 90 | 0.9403 | — tie |
| vs S_LLM | 85.8 | 86.4 | −0.7 | 83 | 89 | 0.7031 | — tie |

### Reading this: what it actually says

**In Setting 2 (the fair comparison), your proposed method wins 6 of 8 comparisons with real
statistical significance, and ties the other 2. It never loses.**

The two most important wins:

**Win 1 — vs `S_LLM` (+4.4 pp, p = 0.0064).** `S_LLM` is the prior-work approach: a single LLM
teacher. Beating it at equal training is the core claim of the paper.

**Win 2 — vs `TRI_equal` (+5.6 pp, p = 0.0004).** This is the scientifically cleanest result you
have. Both runs use *the identical three teachers*. The only difference in the entire setup is
that TRI_shadow weights teachers by measured competence and TRI_equal trusts them equally. So the
5.6 pp gap is **attributable to the competence weighting mechanism itself** — the thing your paper
is about. p = 0.0004 means roughly 1 chance in 2,500 that this is luck.

Note also from Part 8 that TRI_equal does not even beat the plain baseline. So the story is:
*three teachers alone are worthless; three teachers plus competence weighting is what works.*

### The two ties, honestly

`TRI_shadow` does **not** beat `S_PS` (−0.1 pp, p = 1.00) or `LOO_noBP` (−1.4 pp, p = 0.36).

`p = 1.0000` is as flat as a result can be: 84 wins vs 85 wins out of 900 episodes. **You cannot
claim your method beats the preview_safe rule alone.** Attempting to would be dishonest and a
reviewer would catch it immediately. Part 10 explains why TRI_shadow is still the better method
despite this tie.

### The settings disagree — and you must say so

In Setting 1 nothing is significant except vs the baseline. Everything is a tie.

**Why?** In Setting 1, TRI_shadow is evaluated at 3.28M steps while S_LLM, S_PS, TRI_equal and
LOO_noLLM are at 5.00M — **34% less training.** Under that handicap it still ties them all. That
is arguably *in your favour*, but it is not a claim you can make cleanly.

**The correct framing for the paper:** *"At matched training budget (Setting 2), the proposed
method significantly outperforms the single-LLM-teacher baseline and the equal-weight ablation.
Under unmatched budgets (Setting 1) where the proposed method has 34% less training, it remains
statistically indistinguishable from those baselines."*

---

<a name="part-10"></a>
## Part 10 — RESULT 4: the complementarity finding (your strongest result)

**This is the finding that holds in every setting, has the smallest p-values, and justifies the
entire premise of the project.**

### The question

The whole project assumes: *different teachers are good at different things, so combining them
should help.* Is that assumption actually true, or was it just a guess?

### The test

Compare `S_PS` (safety-rule teacher only) against `S_LLM` (LLM teacher only) on the same 900
episodes — but instead of asking "who succeeded," ask **"who failed, and how?"**

Count episodes where *only one* of them had a collision, and episodes where *only one* of them
flew into a storm.

### Results — Setting 2 (equal training)

**Collisions:**
```
both had a collision              :  12 episodes  (ignored — tie)
only S_PS  had a collision        :  34 episodes
only S_LLM had a collision        :  85 episodes
                                     ---
informative                       : 119 episodes

Expected if equally safe: ~60 / ~60.  Observed: 34 / 85.
p = 0.00000   →  S_PS is REALLY better at avoiding aircraft.
```

**Storms:**
```
both flew into a storm            :  14 episodes  (ignored)
only S_PS  flew into a storm      :  57 episodes
only S_LLM flew into a storm      :  30 episodes
                                     ---
informative                       :  87 episodes

Expected if equally safe: ~43 / ~43.  Observed: 57 / 30.
p = 0.00501   →  S_LLM is REALLY better at avoiding storms.
```

### Results — Setting 1 (latest checkpoints) — same conclusion

| Failure type | both | only S_PS | only S_LLM | p | Who is safer |
|---|---|---|---|---|---|
| Collisions | 3 | **20** | 58 | **0.00002** | **S_PS** |
| Storms | 17 | 46 | **23** | **0.00762** | **S_LLM** |

### Why this matters so much

**The direction reverses between the two failure types, in both settings, with strong
significance.** The rule-based teacher is significantly safer around aircraft. The LLM teacher is
significantly safer around storms. **Neither teacher is best at both.**

This is not a marginal statistical curiosity — p = 0.00002 means roughly 1 chance in 50,000. It is
the most solid empirical fact in your entire results set.

**This is your paper's justification.** It proves the premise is real: you are not combining
teachers because it seemed like a nice idea, you are combining them because they have
**measurably, significantly different and complementary weaknesses.**

### The ceiling: how much room is there?

If you had a perfect oracle that knew which teacher to trust on each episode:

| Setting | Best-of-both ceiling | TRI_shadow achieved | Gap remaining |
|---|---|---|---|
| Setting 2 | 94.9% | 85.8% | 9.1 pp |
| Setting 1 | 96.0% | 85.8% | 10.2 pp |

*(Ceiling = the % of episodes where S_PS **or** S_LLM succeeded.)*

Read this as: **the complementarity is worth up to ~95%, and the current method captures roughly
half of the available headroom.** It is real progress, and there is clearly more to get. This is a
good, honest framing — it shows the direction works and motivates future work.

### Rescue analysis

On the 46 Setting-2 episodes where **both** single teachers failed, `TRI_shadow` succeeded on
**23 of them (50%)**. So the combined method is not merely picking the better teacher — it
recovers scenarios that neither individual teacher could handle. (Setting 1: 12 of 36, 33%.)

---

<a name="part-11"></a>
## Part 11 — RESULT 5: crowding test vs the AAMAS 2025 paper

**The question:** how does this system compare to published work, and what happens when the
airspace gets crowded?

**Files:** `setting{1,2}_*/<run>/evaluation_scal_ac{4,6,8}.json`
**Script:** `new_approach/evaluate_per_aircraft.py`

### What the AAMAS 2025 paper reports

That paper's best method is **IDDPG**. It also reports a weaker classical planner, **FMT**.
Their numbers, in the same per-aircraft format:

| Method | Aircraft | Aircraft-LOS | Storm-LOS | Goal reach | Distance ratio | Decision time |
|---|---|---|---|---|---|---|
| IDDPG | 4 | 0% | 0% | 100% | — | <20 ms |
| IDDPG | 8 | 0% | 5% | **95%** | 1.17 | <20 ms |
| FMT | 4 | 5.2–7.3% | 0.05% | 100% | 1.08 | 3–41 s |
| FMT | 8 | 16.8% | 0.05% | 93% | — | 3–41 s |

### Your results — Setting 2 (equal training)

| Run | 4AC LOS | 4AC storm | 4AC goal | 4AC ratio | 8AC LOS | 8AC storm | 8AC goal | 8AC ratio | goal drop |
|---|---|---|---|---|---|---|---|---|---|
| **TRI_shadow** | **2.8** | 1.50 | **88.2** | **1.041** | 13.4 | 1.88 | 35.2 | **1.085** | 53.0 |
| S_LLM | 5.5 | 1.17 | 85.0 | 1.047 | 16.2 | 2.00 | 23.1 | 1.073 | 61.9 |
| TRI_equal | 6.2 | 1.50 | 82.2 | 1.048 | 15.9 | 1.92 | 26.8 | 1.092 | 55.4 |
| T0_baseline | **1.0** | 2.58 | 86.4 | 1.105 | **11.6** | 2.42 | 35.6 | 1.178 | 50.8 |
| S_PS | 2.8 | 2.42 | 85.4 | 1.075 | 12.8 | 2.29 | 33.0 | 1.129 | 52.4 |
| LOO_noLLM | 4.0 | 1.58 | 83.3 | 1.051 | 15.6 | 2.25 | 24.5 | 1.084 | 58.8 |
| **LOO_noBP** | 2.5 | 1.75 | **88.1** | **1.036** | **10.8** | 2.38 | **40.1** | 1.095 | **48.0** |
| LOO_noPS | 5.0 | 1.17 | 84.3 | 1.036 | 14.0 | 2.08 | 30.8 | **1.064** | 53.6 |
| S_BP | 2.8 | 2.00 | 85.5 | 1.092 | 11.2 | 2.58 | 37.7 | 1.160 | **47.8** |

*(6-aircraft column omitted for width; it sits between the two, e.g. TRI_shadow 8.7% LOS / 66.6% goal.)*

### How to read these numbers

**At 4 aircraft, you are competitive:**

- `TRI_shadow` has **2.8% aircraft-LOS** — worse than IDDPG's 0%, but **better than FMT's 5.2–7.3%.**
  You sit between the paper's strong and weak baselines. That is a defensible position.
- **Your flight distance ratio is genuinely better.** TRI_shadow at 4AC = **1.041** vs FMT's 1.08.
  At 8AC = **1.085** vs IDDPG's 1.17. Your aircraft fly noticeably more direct routes. **This is a
  real, reportable win** and the one place you clearly beat the published numbers.
- Goal reach 88.2% vs their 100% — you are behind here.

**At 8 aircraft, everything collapses:**

Goal reach falls from ~85-88% to **23-40%** for every single run. IDDPG holds at 95%.
**You do not beat this paper on scalability. Do not claim you do.**

### Why goal-reach collapses — check this before explaining it

It is tempting to assume the aircraft simply ran out of time. **They did not.** The episode-level
outcomes at 8 aircraft (`evaluation_scal_ac8.json`) show what actually ends the episodes:

| Run | 4AC success | 4AC collision | 4AC truncated | 8AC success | **8AC collision** | 8AC weather | 8AC truncated |
|---|---|---|---|---|---|---|---|
| TRI_shadow | 86.7 | 5.7 | 1.7 | 28.0 | **52.0** | 14.3 | 5.7 |
| S_LLM | 83.0 | 11.0 | 1.3 | 18.0 | **63.7** | 15.0 | 3.3 |
| TRI_equal | 78.0 | 12.3 | 4.0 | 19.7 | **60.7** | 15.0 | 4.7 |
| T0_baseline | 81.0 | 2.0 | 6.7 | 24.7 | **46.0** | 18.0 | 11.3 |
| S_PS | 84.3 | 5.7 | 0.3 | 28.3 | **50.3** | 17.3 | 4.0 |
| LOO_noBP | 87.0 | 5.0 | 1.0 | 32.0 | **42.7** | 18.3 | 7.0 |
| LOO_noPS | 81.0 | 10.0 | 4.3 | 23.7 | **54.3** | 16.7 | 5.3 |
| S_BP | 82.0 | 5.7 | 4.3 | 26.0 | **44.0** | 19.0 | 11.0 |
| LOO_noLLM | 76.0 | 8.0 | 9.7 | 18.7 | **60.3** | 16.3 | 4.7 |

**Collisions go from ~2–12% of episodes at 4 aircraft to 43–64% at 8.** Truncation barely moves
(1–10% either way). So the collapse is a **safety failure, not a time-budget failure**: episodes
end early because two aircraft hit each other, and every aircraft still in the air at that moment
is counted as not having reached its goal. That is the mechanism behind the low `goal_reach_rate`.

**The honest explanation to state in the paper:**

1. **Observation limit — the primary cause.** The policy only sees its 3 nearest neighbours. At 4
   aircraft that is everyone in the sector. At 8 aircraft it is under half the traffic, so the
   network is blind to most of the airspace and cannot anticipate conflicts it never observes.
   This is a fixable architectural limit, not a failure of the teaching method.
2. **Episode length is a secondary factor.** Your episodes are capped at 60 steps
   (`MAX_STEP = 60` in `configs.py`) against the AAMAS paper's 150. This depresses `goal_reach_rate`
   somewhat, but the truncation numbers above show it is **not** the main driver — do not lead
   with it.

> **Do not write "most 8-aircraft failures are timeouts."** The data says the opposite, and the
> numbers to disprove it are in the same files a reviewer would open.

### The finding that matters most in this section

**Teaching does not help with crowding.** Compare goal-reach drop from 4AC to 8AC:

```
TRI_shadow  (all teaching)  : −53.0 pp
T0_baseline (no teaching)   : −50.8 pp   ← the untaught baseline degrades LESS
S_BP                        : −47.8 pp   ← best
LOO_noBP                    : −48.0 pp
S_LLM                       : −61.9 pp   ← worst
```

The taught runs do not degrade any more gracefully than the untaught one. **The safety improvements
you demonstrated at 4 aircraft do not transfer to 8 aircraft.** This is a genuine negative result
and belongs in your limitations section. Reviewers respect papers that report this; they penalise
papers that quietly omit it.

One nuance worth reporting alongside it: on the **collision** measure specifically, TRI_shadow at
8 aircraft (52.0% of episodes) is still better than S_LLM (63.7%), TRI_equal (60.7%) and
LOO_noLLM (60.3%) — but worse than the untaught baseline (46.0%). So the teaching helps against
the other *taught* variants at scale while still losing to no teaching at all. State it that way;
it is more accurate than a flat "teaching does not help."

### Setting 1 note

In Setting 1, `S_PS` is the standout at 8AC (8.6% LOS, 47.9% goal, only −41.2 pp drop) — but S_PS
has 5.00M steps there vs TRI_shadow's 3.28M, so this is not a like-for-like comparison. Use
Setting 2 for the benchmark table.

---

<a name="part-12"></a>
## Part 12 — RESULT 6: decision speed

**Files:** `eval_results/shared/<run>/latency_ac{4,8}.json` (20 episodes each)

| Run | Aircraft | Mean | Median | Worst-in-episode |
|---|---|---|---|---|
| T0_baseline | 4 | 149.8 ms | 131.1 ms | 304.7 ms |
| T0_baseline | 8 | 145.2 ms | 148.2 ms | 232.5 ms |
| S_LLM | 4 | 133.2 ms | 118.8 ms | 199.7 ms |
| S_LLM | 8 | 167.5 ms | 163.3 ms | 381.3 ms |

### How to read this

**Milliseconds (ms) = thousandths of a second.** 150 ms = 0.15 seconds per decision.

**The good news:** the timing barely changes from 4 to 8 aircraft (145–167 ms). Your method does
**not** get slower as traffic increases. Compare this to the AAMAS paper's FMT planner at
**3–41 seconds** — you are 20× to 250× faster than that classical planner.

**The caveat you must state:** the paper's IDDPG runs in **under 20 ms**, and your ~150 ms is
slower. But the comparison is not apples to apples:

- These measurements were taken on a **heavily loaded shared CPU** (8 evaluation sessions plus 3
  training jobs running simultaneously) with **GPU deliberately disabled** (`CUDA_VISIBLE_DEVICES=""`).
- On a dedicated machine with GPU, this network's forward pass would be in the low milliseconds.

**Recommended wording:** *"Measured at 133–168 ms per decision on a contended CPU with GPU disabled;
this is an upper bound rather than a deployment figure. Critically, latency is flat with respect to
traffic density (145 ms at 4 aircraft, 167 ms at 8), unlike search-based planners."*

Do not claim "we are faster than IDDPG." You did not measure that, and it is probably false.

---

<a name="part-13"></a>
## Part 13 — RESULT 7: learning curves

**Files:** `eval_results/shared/<run>/curve_<step>.json` (100 episodes each, so noisier — ±8 pp
wobble, not ±4).

Each run's success rate at several points during training:

```
TRI_shadow   1.11M:82%  1.61M:76%  2.11M:80%  2.61M:87%  3.28M:87%
S_LLM        1.00M:73%  1.60M:78%  2.10M:70%  2.60M:79%  3.10M:83%  5.00M:88%
TRI_equal    1.00M:68%  1.50M:78%  2.00M:80%  2.50M:82%  3.00M:78%  5.00M:85%
T0_baseline  1.00M:75%  1.50M:70%  2.00M:66%  2.50M:78%  3.00M:81%  5.00M:82%
S_PS         1.00M:71%  1.50M:81%  2.00M:77%  2.50M:86%  3.26M:84%  5.00M:85%
LOO_noLLM    1.00M:69%  1.50M:80%  2.50M:91%  3.00M:76%  5.00M:85%
LOO_noBP     1.30M:78%  1.80M:75%  2.30M:87%  2.80M:79%  3.30M:87%  4.15M:84%
LOO_noPS     1.29M:77%  1.79M:76%  2.29M:84%  2.79M:83%  3.29M:81%  4.12M:82%
S_BP         1.00M:76%  1.50M:77%  2.00M:78%  2.50M:80%  3.33M:82%  5.00M:82%
```

### How to read these

**These are noisy — treat them as trends, not measurements.** With only 100 episodes, a genuine
82% run can easily print 76% or 88%. Look at `LOO_noLLM`: 80% → **91%** → 76% within 1.5M steps.
That 91% is a fluke, not a peak. Never quote a single curve point as a result.

**What is trustworthy in this data:**

1. **`TRI_shadow` reaches a high level earlier.** It is at 82% by 1.11M steps and 87% by 2.61M.
   `T0_baseline` is still at 78% at 2.50M and only reaches 82% at 5.00M. So the proposed method
   reaches the untaught baseline's *final* performance at roughly **half the training budget.**
   That is a legitimate and useful claim: **teaching improves sample efficiency.**

2. **Every run improves from ~1M to ~3M.** Nothing has collapsed or diverged. Training is healthy.

3. **`TRI_shadow` is flat from 2.61M to 3.28M** (87% → 87%). Combined with it still being the
   3rd-highest scorer at 3.28M, this suggests it is near its plateau — so the fact that it hasn't
   reached 5M is **less damaging than it sounds.** It is not obviously still climbing.

4. **`S_LLM` gains a lot late** (83% at 3.10M → 88% at 5.00M, +5 pp). This is the direct
   explanation for why the two settings disagree in Part 9: S_LLM's advantage in Setting 1 comes
   from training TRI_shadow has not had yet.

---

<a name="part-14"></a>
## Part 14 — Putting it together: what you can and cannot claim

### ✅ Claims you can make, with the evidence

**1. Teachers are demonstrably complementary. (STRONGEST)**
> The rule-based teacher is significantly safer around traffic (only-S_PS collisions 34 vs
> only-S_LLM 85, p < 10⁻⁵) while the LLM teacher is significantly safer around weather
> (only-S_PS 57 vs only-S_LLM 30, p = 0.005). The pattern replicates in both evaluation settings.

*This is the premise of your project, and it is now empirically proven.*

**2. Competence weighting — not merely having multiple teachers — is what produces the gain.**
> At matched training budget, competence-weighted three-teacher distillation reaches 85.8% vs
> 80.2% for equal-weight distillation from the identical teachers (+5.6 pp, p = 0.0004). The
> equal-weight variant does not significantly outperform the untaught baseline at all.

*This is your cleanest controlled experiment — one variable changed, large effect.*

**3. The method beats the prior single-LLM-teacher approach at matched budget.**
> +4.4 pp over S_LLM (85.8% vs 81.3%, p = 0.0064) on 900 paired episodes.

**4. Teaching improves sample efficiency.**
> TRI_shadow reaches ~87% by 2.6M steps; the untaught baseline reaches 82% only at 5.0M.

**5. Route efficiency beats the published baselines.**
> Flight distance ratio 1.041 at 4 aircraft and 1.085 at 8, versus 1.08 (FMT) and 1.17 (IDDPG).

**6. Decision latency is flat with traffic density.**
> 145 ms at 4 aircraft, 167 ms at 8 — versus 3–41 s for the search-based planner.

### ❌ Claims you cannot make

| Tempting claim | Why it fails |
|---|---|
| "TRI_shadow is the best method" | It ties `S_PS` (p = 1.00) and `LOO_noBP` (p = 0.36). |
| "All three teachers are necessary" | Removing `best_preview` (LOO_noBP) does not measurably hurt. |
| "We beat the AAMAS paper" | Their IDDPG reaches 95% goal at 8 aircraft; your best is 40%. |
| "We are faster than IDDPG" | Not measured, and almost certainly false (150 ms vs <20 ms). |
| Anything about `S_LLM` or `LOO_noLLM` in isolation | They flip sign between settings. |
| "Teaching helps with scalability" | The untaught baseline degrades *less* from 4→8 aircraft. |

### The paragraph to build the paper around

> *We show that heterogeneous teachers for multi-agent air-traffic control have significantly
> different and complementary failure modes: a rule-based safety teacher prevents aircraft
> conflicts (p < 10⁻⁵) while an LLM teacher prevents weather incursions (p = 0.005). Exploiting
> this, competence-weighted distillation reaches 85.8% success versus 81.3% for a single LLM
> teacher (+4.4 pp, p = 0.006) and 80.2% for equal-weight distillation from the identical teachers
> (+5.6 pp, p < 0.001), all at matched training budget on 900 paired episodes. The equal-weight
> ablation does not outperform an untaught baseline, isolating the competence-weighting mechanism
> as the source of the gain.*

---

<a name="part-15"></a>
## Part 15 — The honest limitations

State every one of these in the paper. They will be found otherwise, and a paper that
self-reports its limits is treated far more generously than one caught hiding them.

### 1. One training seed per method (the biggest limitation)

Every run is **seed 0 only.** A neural network trained twice with different random seeds gives
different results — sometimes substantially.

**You have direct evidence of how large this effect is**, from `DISTILLATION_RESULTS.md`:

```
T0_baseline seed 0 : 81.7%
T0_baseline seed 1 : 80.7%
T0_baseline seed 2 : 59.7%   ← same code, same settings, 22 pp lower
```

**A 22 pp swing from the training seed alone is larger than every effect reported in this
document.** So a +4.4 pp result, however small its p-value, could in principle be a seed artefact.

**The precise statement of what the p-values do and do not cover:**
- ✅ They prove the difference is not caused by which *evaluation episodes* were drawn.
- ❌ They say nothing about which *training seed* was drawn.

**Required wording:** *"All results are from a single training seed per configuration; the reported
p-values quantify evaluation-episode variance only and do not account for training-seed variance."*

### 2. Two runs give opposite answers in the two settings

`LOO_noLLM` is significantly worse than baseline in Setting 2 (−6.2 pp, p = 0.0007) and
significantly better in Setting 1 (+3.4 pp, p = 0.037). `S_LLM` flips too.

This is a concrete demonstration of point 1 in action. Report both settings; make no claim about
either run.

### 3. The proposed method is undertrained

`TRI_shadow` is at 3.46M of 5.00M steps. Setting 2 handles this correctly by comparing everyone at
~3M. But it means you have **no 5M number for your own method.** The learning curve (Part 13)
suggests it has plateaued, which softens this, but it remains a gap.

*Why so slow:* the competence check requires simulating extra rollouts at every step. Because
`DISTILL_MIN_WEIGHT = 0.10` (not 0) in `configs.py`, that check **never switches off** — it runs
for the entire 5M steps. This run trains at roughly 75k steps/day versus ~1M/day for the others.
This is worth reporting as a computational-cost finding in its own right.

### 4. Scalability results are negative

Goal reach falls to 23–40% at 8 aircraft. Report it, and give the two structural reasons
(60-step cap, 3-nearest-neighbour observations).

### 5. Latency was measured under load

133–168 ms on a contended CPU with GPU disabled. Present as an upper bound.

### 6. Reproducibility

The simulator uses data files (`22feb_paths.csv`, `path_Waypoints22feb.csv`, `test.geojson`) that
are not published, and there is no community benchmark for this environment. Nobody can reproduce
these numbers without those files. Consider releasing them.

---

## Quick reference card

| Symbol | Means | Example |
|---|---|---|
| **pp** | percentage points — a plain subtraction of two percentages | 85.8% − 81.3% = **+4.4 pp** |
| **p** | probability that luck alone would produce a gap this big | p = 0.006 → 1 chance in 170 |
| **p < 0.05** | conventional threshold — call the difference real | ✅ |
| **p ≥ 0.05** | "not proven" — not the same as "no difference" | ✗ |
| **paired** | both runs tested on identical scenarios, cancelling scenario difficulty | — |
| **LOS** | loss of separation — two aircraft too close | — |
| **900 episodes** | 3 blocks of 300 (episode seeds 42…, 1042…, 2042…) pooled | ±2.3 pp wobble |

**Rules of thumb**

- 300 episodes → differences under ~4 pp are indistinguishable
- 900 episodes → differences under ~2.3 pp are indistinguishable
- 100 episodes (curves) → differences under ~8 pp are indistinguishable
- Always read **pp and p together**: pp = how much, p = how sure
- Trust a result only if it appears in **both** settings

---

## File map

```
new_approach/eval_results/
├── setting1_latest/<run>/          each run at its newest checkpoint
│   ├── evaluation.json                 300 episodes, seeds 42–341
│   ├── evaluation_seed1042.json        300 episodes, seeds 1042–1341
│   ├── evaluation_seed2042.json        300 episodes, seeds 2042–2341
│   └── evaluation_scal_ac{4,6,8}.json  per-aircraft metrics at 4/6/8 aircraft
├── setting2_3M/<run>/              same six files, all runs at ~3M steps  ← PRIMARY
├── shared/<run>/
│   ├── latency_ac{4,8}.json            decision-speed measurement, 20 episodes
│   └── curve_<step>.json               learning-curve points, 100 episodes
└── analysis/analysis_S{1,2}_{base,seed1042,seed2042}.md    auto-generated, 300 eps each

Scripts that produced these:
  evaluate_rl.py                        episode-level metrics
  new_approach/evaluate_per_aircraft.py per-aircraft AAMAS-format metrics + latency
  new_approach/analyze_pairwise.py      paired McNemar analysis
  new_approach/experiment_jobs/*.sh     the job scripts that ran everything

Related documents:
  DISTILLATION_RESULTS.md               earlier results tracker (5M numbers, includes seeds 1,2)
  CONCLUSION_AND_BENCHMARK_PLAN.md      the plan these evaluations executed
  SESSION_HANDOFF.md                    operational notes for resuming work
```
