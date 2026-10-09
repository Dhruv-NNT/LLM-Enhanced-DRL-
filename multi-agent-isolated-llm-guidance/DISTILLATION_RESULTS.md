# Distillation Study — Results Tracker

Evaluation protocol (identical for every model, for a fair paired comparison):
- **Model evaluated:** `weights/last_checkpoint.pt` (the final 5,000,000-step model).
- **300 episodes**, `--seed 42` (episodes use seeds 42…341 — same 300 scenarios for all).
- **4 agents, 2 weather cells.** Metrics saved to `<run>/eval_last.json`.
- Higher **success%** is better; lower collision/weather/truncated is better.

Status legend: ✅ complete (5M) · 🔄 training · ⏸ stopped (incomplete) · ⬜ not started.

## Results (per run) — updated 2 Sep 2026

Success/collision/weather/truncation are percentages over the 300 eval episodes; mean
return is the mean episode reward. **9 of 12 runs evaluated; 3 still training.**

| Experiment | Teachers | Competence | Seed | Steps | Status | Success % | Collision % | Weather % | Truncated % | Mean return | Eval |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T0_baseline | — (pure MAPPO) | — | 0 | 5.00M | ✅ | 81.7 | 5.7 | 7.0 | 5.7 | 125.6 | ✅ |
| T0_baseline | — (pure MAPPO) | — | 1 | 5.00M | ✅ | 80.7 | 5.7 | 6.7 | 7.0 | 123.6 | ✅ |
| T0_baseline | — (pure MAPPO) | — | 2 | 5.00M | ✅ | 59.7 | 3.7 | 9.3 | 27.3 | 78.0 | ✅ |
| S_BP | best_preview | shadow | 0 | 5.00M | ✅ | 82.3 | 4.7 | 6.3 | 6.7 | 128.5 | ✅ |
| S_PS | preview_safe | shadow | 0 | 5.00M | ✅ | 85.0 | 2.3 | 7.7 | 5.0 | 134.1 | ✅ |
| S_LLM | llm | shadow | 0 | 5.00M | ✅ | **87.7** | 6.3 | 3.7 | 2.3 | 139.9 | ✅ |
| LOO_noLLM | best_preview+preview_safe | shadow | 0 | 5.00M | ✅ | 84.7 | 6.7 | 5.0 | 3.7 | 134.0 | ✅ |
| LOO_noBP | llm+preview_safe | shadow | 0 | ~4.39M | 🔄 | — | — | — | — | — | — |
| LOO_noPS | llm+best_preview | shadow | 0 | ~4.35M | 🔄 | — | — | — | — | — | — |
| TRI_equal | llm+best_preview+preview_safe | equal | 0 | 5.00M | ✅ | 85.3 | 7.7 | 4.3 | 2.7 | 134.2 | ✅ |
| TRI_equal | llm+best_preview+preview_safe | equal | 1 | 5.00M | ✅ | 83.7 | 2.3 | 6.7 | 7.3 | 131.5 | ✅ |
| TRI_shadow | llm+best_preview+preview_safe | shadow | 0 | ~3.33M | 🔄 | — | — | — | — | — | — |

Still training (seed 0): **LOO_noBP** (~4.39M), **LOO_noPS** (~4.35M), **TRI_shadow** (~3.33M).
Seeds 1 & 2 for the non-T0 experiments: ⬜ not started (TRI_equal seed 1 done; seed 2 pending).

> **Preliminary — do not conclude yet.** These are mostly single-seed, and the **proposed
> method (TRI_shadow) has not been evaluated** (still training). Right now S_LLM (single LLM
> teacher) leads at 87.7% and the three-teacher `equal` ablation is ~84–85%, but the headline
> comparisons below stay open until TRI_shadow finishes and seeds are averaged. T0_seed2 is a
> low outlier (59.7%, 27.3% truncation) that drags the T0 average — worth noting when averaging.

## Headline comparisons to fill in (averaged over seeds once available)

1. **Teaching helps?** S_* vs T0.
2. **Proposed beats prior work?** TRI_shadow vs S_LLM ← headline.
3. **Smart trust needed?** TRI_shadow vs TRI_equal.
4. **Each teacher earns its place?** each LOO_* vs TRI_shadow.
5. **LLM worth its cost?** LOO_noLLM vs TRI_shadow.
