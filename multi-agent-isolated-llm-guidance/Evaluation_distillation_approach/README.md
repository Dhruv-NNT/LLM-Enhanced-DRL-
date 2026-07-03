# Evaluation — Multi-Teacher Distillation Approach

All training runs and evaluations for the **current** (multi-teacher distillation)
research direction live here, one subfolder per run.

Layout:

```
Evaluation_distillation_approach/
  <TAG>_seed<s>/
    weights/best_model.pt         # trained student
    weights/last_checkpoint.pt
    tensorboard/                  # training curves
    eval.json                     # evaluate_rl.py metrics
    run_hyperparameters.txt
```

`<TAG>` is one of: `T0_baseline`, `S_BP`, `S_PS`, `LOO_noLLM`, `S_LLM`, `LOO_noBP`,
`LOO_noPS`, `TRI_equal`, `TRI_shadow` (see `../EXPERIMENT_PLAN.md`).

Inputs for these runs live outside this folder: the recorded trajectories in
`../distill_data/` and the trained teacher networks in `../teachers/`.

Older evaluations from the previous (pre-distillation) approach are kept separately
and are not part of this study.
