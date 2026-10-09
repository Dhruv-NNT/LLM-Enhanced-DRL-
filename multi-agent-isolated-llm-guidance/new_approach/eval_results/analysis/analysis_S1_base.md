# Paired analysis — pattern `new_approach/eval_results/setting1_latest/*/evaluation.json`

Baseline: **T0_baseline_seed0**. Runs: 9. Common paired scenarios: **300**.

## T1. Each run vs baseline (paired on identical scenarios)

| run | success % | Δ pp | rescued | lost | McNemar p | 95% CI Δpp | real? |
|---|---|---|---|---|---|---|---|
| T0_baseline_seed0 (baseline) | 81.7 | 0.0 | – | – | – | – | – |
| S_LLM_seed0 | 87.7 | +6.0 | 42 | 24 | 0.036 | [+1.0, +11.3] | YES |
| TRI_shadow_seed0 | 86.7 | +5.0 | 38 | 23 | 0.072 | [+0.0, +10.0] | no |
| TRI_equal_seed0 | 85.3 | +3.7 | 39 | 28 | 0.222 | [-1.3, +9.0] | no |
| S_PS_seed0 | 85.0 | +3.3 | 40 | 30 | 0.282 | [-2.3, +8.7] | no |
| LOO_noLLM_seed0 | 84.7 | +3.0 | 39 | 30 | 0.336 | [-2.3, +8.7] | no |
| LOO_noBP_seed0 | 84.3 | +2.7 | 40 | 32 | 0.410 | [-2.7, +8.0] | no |
| S_BP_seed0 | 82.3 | +0.7 | 40 | 38 | 0.910 | [-5.3, +6.7] | no |
| LOO_noPS_seed0 | 82.0 | +0.3 | 37 | 36 | 1.000 | [-5.3, +6.0] | no |

*rescued* = baseline failed, run succeeded; *lost* = the reverse. *real?* = p < 0.05 (the split of disagreements is too lopsided to be a coin flip).

## T2. Failure-mode split (% of scenarios)

| run | success | collision | weather | truncated |
|---|---|---|---|---|
| S_LLM_seed0 | 87.7 | 6.3 | 3.7 | 2.3 |
| TRI_shadow_seed0 | 86.7 | 5.7 | 6.0 | 1.7 |
| TRI_equal_seed0 | 85.3 | 7.7 | 4.3 | 2.7 |
| S_PS_seed0 | 85.0 | 2.3 | 7.7 | 5.0 |
| LOO_noLLM_seed0 | 84.7 | 6.7 | 5.0 | 3.7 |
| LOO_noBP_seed0 | 84.3 | 7.0 | 5.3 | 3.3 |
| S_BP_seed0 | 82.3 | 4.7 | 6.3 | 6.7 |
| LOO_noPS_seed0 | 82.0 | 4.0 | 4.3 | 9.7 |
| T0_baseline_seed0 | 81.7 | 5.7 | 7.0 | 5.7 |

## T3. Pairwise complementarity (McNemar on each failure type; counts are 'A-only' vs 'B-only')

| A | B | collision A-only / B-only (p) | weather A-only / B-only (p) | success A-only / B-only (p) |
|---|---|---|---|---|
| LOO_noBP_seed0 | LOO_noLLM_seed0 | 18 / 17 (1.000) | 6 / 5 (1.000) | 25 / 26 (1.000) |
| LOO_noBP_seed0 | LOO_noPS_seed0 | 20 / 11 (0.150) | 8 / 5 (0.581) | 36 / 29 (0.457) |
| LOO_noBP_seed0 | S_BP_seed0 | 20 / 13 (0.296) | 7 / 10 (0.629) | 31 / 25 (0.504) |
| LOO_noBP_seed0 | S_LLM_seed0 | 17 / 15 (0.860) | 9 / 4 (0.267) | 22 / 32 (0.220) |
| LOO_noBP_seed0 | S_PS_seed0 | 20 / 6 (0.009) * | 7 / 14 (0.189) | 30 / 32 (0.899) |
| LOO_noBP_seed0 | T0_baseline_seed0 | 21 / 17 (0.627) | 7 / 12 (0.359) | 40 / 32 (0.410) |
| LOO_noBP_seed0 | TRI_equal_seed0 | 18 / 20 (0.871) | 8 / 5 (0.581) | 29 / 32 (0.798) |
| LOO_noBP_seed0 | TRI_shadow_seed0 | 17 / 13 (0.585) | 6 / 8 (0.791) | 21 / 28 (0.392) |
| LOO_noLLM_seed0 | LOO_noPS_seed0 | 18 / 10 (0.185) | 8 / 6 (0.791) | 33 / 25 (0.358) |
| LOO_noLLM_seed0 | S_BP_seed0 | 19 / 13 (0.377) | 7 / 11 (0.481) | 33 / 26 (0.435) |
| LOO_noLLM_seed0 | S_LLM_seed0 | 19 / 18 (1.000) | 8 / 4 (0.388) | 22 / 31 (0.272) |
| LOO_noLLM_seed0 | S_PS_seed0 | 20 / 7 (0.019) * | 7 / 15 (0.134) | 30 / 31 (1.000) |
| LOO_noLLM_seed0 | T0_baseline_seed0 | 18 / 15 (0.728) | 7 / 13 (0.263) | 39 / 30 (0.336) |
| LOO_noLLM_seed0 | TRI_equal_seed0 | 15 / 18 (0.728) | 8 / 6 (0.791) | 28 / 30 (0.896) |
| LOO_noLLM_seed0 | TRI_shadow_seed0 | 17 / 14 (0.720) | 6 / 9 (0.607) | 21 / 27 (0.471) |
| LOO_noPS_seed0 | S_BP_seed0 | 11 / 13 (0.839) | 6 / 12 (0.238) | 34 / 35 (1.000) |
| LOO_noPS_seed0 | S_LLM_seed0 | 10 / 17 (0.248) | 6 / 4 (0.754) | 25 / 42 (0.050) * |
| LOO_noPS_seed0 | S_PS_seed0 | 12 / 7 (0.359) | 5 / 15 (0.041) * | 27 / 36 (0.314) |
| LOO_noPS_seed0 | T0_baseline_seed0 | 12 / 17 (0.458) | 5 / 13 (0.096) | 37 / 36 (1.000) |
| LOO_noPS_seed0 | TRI_equal_seed0 | 10 / 21 (0.071) | 5 / 5 (1.000) | 30 / 40 (0.282) |
| LOO_noPS_seed0 | TRI_shadow_seed0 | 12 / 17 (0.458) | 5 / 10 (0.302) | 22 / 36 (0.087) |
| S_BP_seed0 | S_LLM_seed0 | 11 / 16 (0.442) | 12 / 4 (0.077) | 23 / 39 (0.056) |
| S_BP_seed0 | S_PS_seed0 | 14 / 7 (0.189) | 11 / 15 (0.557) | 30 / 38 (0.396) |
| S_BP_seed0 | T0_baseline_seed0 | 13 / 16 (0.711) | 11 / 13 (0.839) | 40 / 38 (0.910) |
| S_BP_seed0 | TRI_equal_seed0 | 13 / 22 (0.175) | 11 / 5 (0.210) | 27 / 36 (0.314) |
| S_BP_seed0 | TRI_shadow_seed0 | 14 / 17 (0.720) | 9 / 8 (1.000) | 26 / 39 (0.136) |
| S_LLM_seed0 | S_PS_seed0 | 18 / 6 (0.023) * | 4 / 16 (0.012) * | 30 / 22 (0.332) |
| S_LLM_seed0 | T0_baseline_seed0 | 18 / 16 (0.864) | 4 / 14 (0.031) * | 42 / 24 (0.036) * |
| S_LLM_seed0 | TRI_equal_seed0 | 13 / 17 (0.585) | 4 / 6 (0.754) | 28 / 21 (0.392) |
| S_LLM_seed0 | TRI_shadow_seed0 | 16 / 14 (0.856) | 2 / 9 (0.065) | 24 / 21 (0.766) |
| S_PS_seed0 | T0_baseline_seed0 | 7 / 17 (0.064) | 10 / 8 (0.815) | 40 / 30 (0.282) |
| S_PS_seed0 | TRI_equal_seed0 | 6 / 22 (0.004) * | 15 / 5 (0.041) * | 27 / 28 (1.000) |
| S_PS_seed0 | TRI_shadow_seed0 | 7 / 17 (0.064) | 13 / 8 (0.383) | 26 / 31 (0.597) |
| T0_baseline_seed0 | TRI_equal_seed0 | 17 / 23 (0.430) | 13 / 5 (0.096) | 28 / 39 (0.222) |
| T0_baseline_seed0 | TRI_shadow_seed0 | 15 / 15 (1.000) | 11 / 8 (0.648) | 23 / 38 (0.072) |
| TRI_equal_seed0 | TRI_shadow_seed0 | 19 / 13 (0.377) | 4 / 9 (0.267) | 24 / 28 (0.678) |

`*` = p < 0.05. A lower 'A-only' count means A makes that mistake less often than B.

## T4. Rescue analysis on the 55 scenarios the baseline failed

| run | rescued | % of baseline failures |
|---|---|---|
| S_LLM_seed0 | 42 | 76.4 |
| LOO_noBP_seed0 | 40 | 72.7 |
| S_BP_seed0 | 40 | 72.7 |
| S_PS_seed0 | 40 | 72.7 |
| LOO_noLLM_seed0 | 39 | 70.9 |
| TRI_equal_seed0 | 39 | 70.9 |
| TRI_shadow_seed0 | 38 | 69.1 |
| LOO_noPS_seed0 | 37 | 67.3 |
