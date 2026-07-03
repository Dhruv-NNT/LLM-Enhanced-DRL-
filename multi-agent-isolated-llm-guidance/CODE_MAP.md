# Code Map — which script belongs to which effort

The scripts are grouped into folders by effort. **Shared code stays at the project
root** (it is imported by everything). Each moved script has a small *path bootstrap*
at the top so that `import configs` and `from rl_llm_multi ...` still resolve after the
move:

```python
import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, _bootstrap_os.path.dirname(
    _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))
```

Run everything **from the project root**, e.g. `python new_approach/train_teachers.py`.
Nothing is duplicated: files used by both approaches (`configs.py`, `rl_llm_multi/`,
`train.py`, `evaluate_rl.py`) live once, at the root.

## Folder layout

```
.                              (project root)
  configs.py                   shared: all knobs/paths
  rl_llm_multi/                shared: the engine (env, MAPPO, teachers, distill, ...)
  train.py                     shared: MAPPO training loop (old + new approaches)
  evaluate_rl.py               shared: evaluate any trained policy

  new_approach/                current multi-teacher distillation scripts
  old_approach/                previous LLM-guided / pure-LLM scripts
  direction_finding/           R1/R2/R3 direction scripts + their logs
```

## Shared core (root — used by everything, never move/duplicate)

| File / dir | Role |
|---|---|
| `configs.py` | All hyperparameters, paths, switches. Imported by every script. |
| `rl_llm_multi/` | The engine: env, MAPPO, teachers, distillation, guidance, memory, utils. |
| `train.py` | MAPPO training loop. Both approaches: old via `USE_LLM_GUIDED_TRAINING`, new via `--distill` / `--distill-teachers` / `--competence-mode`. |
| `evaluate_rl.py` | Evaluates any trained MAPPO policy. Used by both; also imported by `direction_finding/analyze_complementarity.py`. |

## `new_approach/` — multi-teacher distillation (current)

| File | Role | Runbook |
|---|---|---|
| `new_approach/generate_guided_trajectories.py` | Play games with a guide and record `(situation → move)` shards. | `EXPERIMENT_PLAN.md` Phase A |
| `new_approach/generate_parallel.py` | Split generation across CPU workers; writes `part_XX/` dirs. | Phase A |
| `new_approach/train_teachers.py` | Supervised-train a `TeacherPolicy` (same architecture as the MAPPO actor) from the shards. | Phase B |

Related shared code: `rl_llm_multi/teacher.py`, `rl_llm_multi/distill.py`, and the
distillation loss in `rl_llm_multi/mappo.py`. See `DISTILLATION_FLOW.md`.

## `old_approach/` — LLM-guided / pure-LLM (superseded)

| File | Role |
|---|---|
| `old_approach/evaluate_llm.py` | Pure-LLM episode evaluation (the LLM controller flies, no MAPPO). |
| `old_approach/evaluate_llm_fair_comparison.py` | Fair comparison of the real LLM vs cheap heuristics as controllers. |
| `old_approach/evaluate_rl_fair_comparison.py` | Fair comparison for the LLM-guided MAPPO variants. |

Old results live in `Evaluation_old_approach/`; old design docs in `archive_docs/`.

## `direction_finding/` — one-off direction scripts + logs (R1/R2/R3)

| File | Role |
|---|---|
| `direction_finding/analyze_complementarity.py` | The R3 per-decision diagnostic: score MAPPO / LLM / best_preview / preview_safe by short shadow rollouts and report who is best. Justifies the multi-teacher direction. Imports `evaluate_rl` from the root. |

Its outputs are alongside it: `direction_finding/direction_logs/`, `direction_finding/complementarity/`.

## Quick "what do I run?" index (run from the project root)

- Make teacher data → `python new_approach/generate_parallel.py` (or `.../generate_guided_trajectories.py`)
- Make teachers → `python new_approach/train_teachers.py`
- Train a controller → `python train.py`
- Score a controller → `python evaluate_rl.py`
- (Old) score the LLM controller → `python old_approach/evaluate_llm.py`
- (Direction) per-decision teacher comparison → `python direction_finding/analyze_complementarity.py`
