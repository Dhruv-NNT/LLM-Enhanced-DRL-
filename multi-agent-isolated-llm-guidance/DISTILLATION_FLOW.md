# Distillation Flow — How It All Fits Together

A plain-language map of the multi-teacher distillation pipeline: the big idea, what
each script you run does, and the detailed flow during training.

---

## The big idea (two stages + one rule)

We train a controller (**MAPPO student**) by letting it copy one or more **teachers**.
A teacher is a small network that learned to imitate one "guide" (the LLM, or a cheap
simulator rule like `best_preview` / `preview_safe`).

- **Stage 1 (offline, run once):** turn each guide into a teacher network.
- **Stage 2 (online):** train the MAPPO student, nudging it toward the teachers it
  should trust in each situation.

**The one rule that never breaks:** the real environment is **always driven by MAPPO's
own action**. Teachers only shape an extra *loss term* during training — they never
fly the planes. (This is the "isolation" property of the project.)

---

## The scripts you run, in order

### 1. `new_approach/generate_parallel.py` → (launches) `new_approach/generate_guided_trajectories.py`
**Purpose:** record the guide's decisions and save them to disk.

- Plays games where a guide chooses each heading. For every decision it stores a
  "flashcard": `(situation → guide's move)`.
- `--keep success` keeps only games where all aircraft landed; `--total-episodes` is
  the *target number of kept games* (it keeps playing until it has that many).
- `new_approach/generate_parallel.py` just splits that target across many CPU workers (each with a
  disjoint seed range, low priority) and writes to `distill_data/<source>/part_XX/`.

**Output:** `distill_data/<source>/part_XX/shard_*.npz` (the flashcards) — obs (40
numbers), agent index, chosen action, episode id, step.
**Key code:** `new_approach/generate_guided_trajectories.py::main` (the play loop + `_ShardWriter`).

### 2. `new_approach/train_teachers.py`
**Purpose:** turn the saved flashcards into a teacher network.

- Loads all shards (merges every `part_XX/` automatically), splits into a study pile
  and a held-out test pile.
- Trains a `TeacherPolicy` network with **cross-entropy** to predict the guide's move
  from the situation (standard supervised learning / behavioral cloning).
- Reports `best_val_acc` (accuracy on the unseen test pile) vs `majority_val` (always
  guessing the most common move). Higher-than-majority = it really learned.

**Output:** `teachers/teacher_<name>.pt` — a frozen teacher network.
**Key code:** `new_approach/train_teachers.py::train_one`; `rl_llm_multi/teacher.py::TeacherPolicy`,
`load_trajectory_dataset`, `save_teacher`. The teacher is now the **exact same
architecture as the MAPPO actor** (same hidden dims, activation, LayerNorm, init).

### 3. `train.py` (distillation mode)
**Purpose:** train the MAPPO student while the frozen teachers shape its loss.

Turned on with `--distill` (or `DISTILL_ENABLED`), selecting which teachers with
`--distill-teachers` and how to weight them with `--competence-mode`. **No Ollama is
called here** — the teachers are just the small networks from Stage 1. Detailed flow
below.

**Output:** `Evaluation_distillation_approach/<tag>/weights/best_model.pt` — the trained student.

### 4. `evaluate_rl.py`
**Purpose:** measure a trained student on fresh games (success/collision/weather rates).
**Output:** a metrics JSON you compare across runs.

---

## What happens each step inside `train.py` (the core)

For every decision step during training (while the distillation weight is > 0):

1. **MAPPO chooses the real actions** for all active aircraft (`policy_actions`).
   These are what actually get executed in the environment. (The rule.)

2. **Query the teachers.** For each active teacher and each aircraft, get its suggested
   action + full probability distribution over the 13 turns.
   → `rl_llm_multi/distill.py::build_distill_metadata` → `_teacher_action_and_probs`.

3. **Decide how much to trust each teacher here (competence weight).** Controlled by
   `--competence-mode`:
   - **`shadow`** (default): clone the simulator and roll ~5 steps ahead for MAPPO's
     action and for each teacher's action; a teacher earns weight in proportion to how
     much its action beats MAPPO over that short horizon (blend of per-aircraft and
     joint outcome). → `rl_llm_multi/guidance.py::shadow_evaluate_teachers`.
   - **`equal`**: every active teacher gets weight 1 (naive averaging — an ablation).
   - **`weather_rule`**: a fixed rule — trust heuristic teachers more near weather,
     the LLM teacher more otherwise.

4. **Store it all in the buffer.** MAPPO's own transition, plus each teacher's action,
   probabilities, and competence weight, are saved per aircraft.
   → `rl_llm_multi/mappo.py::RolloutBuffer` (name-keyed per-teacher columns).

5. **Step the environment with `policy_actions`** (MAPPO only) and continue.

Then, at each update (after enough steps), `MAPPO.update()` computes the loss:

```
L_total = PPO_loss                              # normal MAPPO (policy + value + entropy)
        + current_distill_weight *              # a global weight that DECAYS over training
          mean over samples of
            Σ_teacher  competence_weight[teacher] * divergence(student, teacher)
```

- **`divergence`** defaults to **Jensen–Shannon** between the student's and teacher's
  action distributions (`DISTILL_LOSS_TYPE`; also `ce` or `soft_kl`).
- Each teacher's term is scaled by the **competence weight** from step 3, so the
  student is pulled toward a teacher only where that teacher was judged better.
- **`current_distill_weight`** starts high and decays to ~0 over
  `[0, DISTILL_DECAY_END_STEP]`, so late training is pure MAPPO fine-tuning (the
  student can surpass its teachers).
→ `rl_llm_multi/mappo.py::MAPPO.update` (the `for name in teacher_names` loop);
`_js_divergence`; `rl_llm_multi/distill.py::current_distill_weight`.

---

## Multi-teacher selection (how the experiments differ)

`--distill-teachers` picks any subset of the registered teachers, and
`--competence-mode` picks the trust rule. That single mechanism produces every
experiment:

| Run | `--distill-teachers` | `--competence-mode` |
|---|---|---|
| S-LLM / S-BP / S-PS | one teacher | shadow |
| pairs (LOO runs) | two teachers | shadow |
| TRI-equal | all three | equal |
| TRI-shadow (proposed) | all three | shadow |
| T0 baseline | `--no-distill` | — |

The loss loop simply iterates over whichever teachers are active, so 1, 2, or 3
teachers all run through the same code.

---

## File map (where each piece lives)

| File | Role |
|---|---|
| `new_approach/generate_parallel.py` | split data generation across CPU workers |
| `new_approach/generate_guided_trajectories.py` | play games, record `(situation → guide move)` shards |
| `rl_llm_multi/teacher.py` | `TeacherPolicy` net; dataset loader; save/load teachers |
| `new_approach/train_teachers.py` | supervised training of a teacher from shards |
| `rl_llm_multi/distill.py` | build per-step teacher targets + competence weights; weight decay |
| `rl_llm_multi/guidance.py` | `shadow_evaluate_teachers` (cloned short rollouts) |
| `rl_llm_multi/mappo.py` | `RolloutBuffer` teacher columns; distillation loss in `update()` |
| `train.py` | the training loop; wires distillation in; keeps env MAPPO-only |
| `evaluate_rl.py` | score a trained student |
| `configs.py` | all the knobs (teacher registry, loss type/weight, decay, competence mode) |

---

## Key config knobs (`configs.py`)

- `DISTILL_TEACHER_PATHS` — registry of teacher-name → checkpoint file.
- `DISTILL_TEACHERS` — which teachers are active (overridable with `--distill-teachers`).
- `COMPETENCE_MODE` — `shadow` / `equal` / `weather_rule` (overridable with `--competence-mode`).
- `DISTILL_LOSS_TYPE` — `js` (default) / `ce` / `soft_kl`.
- `DISTILL_LOSS_WEIGHT`, `DISTILL_DECAY_END_STEP` — starting strength and how fast it decays.
- `TEACHER_HIDDEN_DIMS` etc. — teacher architecture (currently identical to the MAPPO actor).

---

## One-sentence summary

Record a guide's decisions (Stage 1a) → train a small network to copy it (Stage 1b) →
during MAPPO training, at each step ask the frozen teachers, judge how much to trust
each one (usually by a short look-ahead), and add a decaying loss that pulls the
student toward the trusted teachers — while MAPPO alone always flies the planes.
