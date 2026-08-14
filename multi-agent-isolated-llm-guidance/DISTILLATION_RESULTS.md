# Distillation Study — Results Tracker

Evaluation protocol (identical for every model, for a fair paired comparison):
- **Model evaluated:** `weights/last_checkpoint.pt` (the final 5,000,000-step model).
- **300 episodes**, `--seed 42` (episodes use seeds 42…341 — same 300 scenarios for all).
- **4 agents, 2 weather cells.** Metrics saved to `<run>/eval_last.json`.
- Higher **success%** is better; lower collision/weather/truncated is better.

Status legend: ✅ complete (5M) · 🔄 training · ⏸ stopped (incomplete) · ⬜ not started.

## Results (per run)

| Experiment | Teachers | Competence | Seed | Steps | Status | Success % | Collision % | Weather % | Truncated % | Mean return | Eval |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T0_baseline | — (pure MAPPO) | — | 0 | 5.00M | ✅ | _pending_ | | | | | 🔄 running |
| T0_baseline | — (pure MAPPO) | — | 1 | 5.00M | ✅ | _pending_ | | | | | 🔄 running |
| T0_baseline | — (pure MAPPO) | — | 2 | 5.00M | ✅ | _pending_ | | | | | 🔄 running |
| S_BP | best_preview | shadow | 0 | 5.00M | ✅ | _pending_ | | | | | 🔄 running |
| S_PS | preview_safe | shadow | 0 | 5.00M | ✅ | _pending_ | | | | | 🔄 running |
| TRI_equal | llm+best_preview+preview_safe | equal | 0 | 5.00M | ✅ | _pending_ | | | | | 🔄 running |
| S_LLM | llm | shadow | 0 | 3.51M | 🔄 | — | | | | | — |
| LOO_noLLM | best_preview+preview_safe | shadow | 0 | 3.66M | 🔄 | — | | | | | — |
| LOO_noBP | llm+preview_safe | shadow | 0 | 2.50M | 🔄 | — | | | | | — |
| LOO_noPS | llm+best_preview | shadow | 0 | 2.47M | 🔄 | — | | | | | — |
| TRI_shadow | llm+best_preview+preview_safe | shadow | 0 | 1.90M | 🔄 | — | | | | | — |
| TRI_equal | llm+best_preview+preview_safe | equal | 1 | 2.03M | ⏸ | — | | | | | — |

Seeds 1 & 2 for the eight non-T0 experiments: ⬜ not started.

## Headline comparisons to fill in (averaged over seeds once available)

1. **Teaching helps?** S_* vs T0.
2. **Proposed beats prior work?** TRI_shadow vs S_LLM ← headline.
3. **Smart trust needed?** TRI_shadow vs TRI_equal.
4. **Each teacher earns its place?** each LOO_* vs TRI_shadow.
5. **LLM worth its cost?** LOO_noLLM vs TRI_shadow.
