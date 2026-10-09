# Paired analysis — pattern `new_approach/eval_results/setting2_3M/*/evaluation_seed1042.json`

Baseline: **T0_baseline_seed0**. Runs: 9. Common paired scenarios: **300**.

## T1. Each run vs baseline (paired on identical scenarios)

| run | success % | Δ pp | rescued | lost | McNemar p | 95% CI Δpp | real? |
|---|---|---|---|---|---|---|---|
| T0_baseline_seed0 (baseline) | 81.0 | 0.0 | – | – | – | – | – |
| S_PS_seed0 | 87.3 | +6.3 | 38 | 19 | 0.016 | [+1.7, +11.0] | YES |
| LOO_noBP_seed0 | 86.3 | +5.3 | 45 | 29 | 0.081 | [+0.0, +10.7] | no |
| TRI_shadow_seed0 | 84.7 | +3.7 | 41 | 30 | 0.235 | [-1.7, +9.0] | no |
| S_BP_seed0 | 81.0 | +0.0 | 39 | 39 | 1.000 | [-5.7, +6.0] | no |
| S_LLM_seed0 | 81.0 | +0.0 | 44 | 44 | 1.000 | [-6.3, +6.3] | no |
| LOO_noPS_seed0 | 80.7 | -0.3 | 42 | 43 | 1.000 | [-6.3, +5.7] | no |
| TRI_equal_seed0 | 80.0 | -1.0 | 41 | 44 | 0.828 | [-7.0, +5.0] | no |
| LOO_noLLM_seed0 | 75.0 | -6.0 | 40 | 58 | 0.085 | [-12.7, +0.7] | no |

*rescued* = baseline failed, run succeeded; *lost* = the reverse. *real?* = p < 0.05 (the split of disagreements is too lopsided to be a coin flip).

## T2. Failure-mode split (% of scenarios)

| run | success | collision | weather | truncated |
|---|---|---|---|---|
| S_PS_seed0 | 87.3 | 3.7 | 7.3 | 1.7 |
| LOO_noBP_seed0 | 86.3 | 4.7 | 8.3 | 0.7 |
| TRI_shadow_seed0 | 84.7 | 7.3 | 6.7 | 1.3 |
| S_BP_seed0 | 81.0 | 6.0 | 5.7 | 7.3 |
| S_LLM_seed0 | 81.0 | 9.7 | 6.7 | 2.7 |
| T0_baseline_seed0 | 81.0 | 3.7 | 10.0 | 5.3 |
| LOO_noPS_seed0 | 80.7 | 10.0 | 5.3 | 4.0 |
| TRI_equal_seed0 | 80.0 | 8.0 | 6.3 | 5.7 |
| LOO_noLLM_seed0 | 75.0 | 9.7 | 6.3 | 9.0 |

## T3. Pairwise complementarity (McNemar on each failure type; counts are 'A-only' vs 'B-only')

| A | B | collision A-only / B-only (p) | weather A-only / B-only (p) | success A-only / B-only (p) |
|---|---|---|---|---|
| LOO_noBP_seed0 | LOO_noLLM_seed0 | 11 / 26 (0.020) * | 18 / 12 (0.362) | 60 / 26 (0.000) * |
| LOO_noBP_seed0 | LOO_noPS_seed0 | 12 / 28 (0.017) * | 19 / 10 (0.136) | 40 / 23 (0.043) * |
| LOO_noBP_seed0 | S_BP_seed0 | 13 / 17 (0.585) | 21 / 13 (0.229) | 49 / 33 (0.097) |
| LOO_noBP_seed0 | S_LLM_seed0 | 9 / 24 (0.014) * | 15 / 10 (0.424) | 38 / 22 (0.052) |
| LOO_noBP_seed0 | S_PS_seed0 | 12 / 9 (0.664) | 18 / 15 (0.728) | 27 / 30 (0.791) |
| LOO_noBP_seed0 | T0_baseline_seed0 | 13 / 10 (0.678) | 18 / 23 (0.533) | 45 / 29 (0.081) |
| LOO_noBP_seed0 | TRI_equal_seed0 | 13 / 23 (0.132) | 17 / 11 (0.345) | 44 / 25 (0.029) * |
| LOO_noBP_seed0 | TRI_shadow_seed0 | 14 / 22 (0.243) | 19 / 14 (0.487) | 35 / 30 (0.620) |
| LOO_noLLM_seed0 | LOO_noPS_seed0 | 23 / 24 (1.000) | 12 / 9 (0.664) | 38 / 55 (0.097) |
| LOO_noLLM_seed0 | S_BP_seed0 | 25 / 14 (0.108) | 13 / 11 (0.839) | 32 / 50 (0.060) |
| LOO_noLLM_seed0 | S_LLM_seed0 | 27 / 27 (1.000) | 14 / 15 (1.000) | 41 / 59 (0.089) |
| LOO_noLLM_seed0 | S_PS_seed0 | 26 / 8 (0.003) * | 11 / 14 (0.690) | 21 / 58 (0.000) * |
| LOO_noLLM_seed0 | T0_baseline_seed0 | 28 / 10 (0.005) * | 11 / 22 (0.080) | 40 / 58 (0.085) |
| LOO_noLLM_seed0 | TRI_equal_seed0 | 23 / 18 (0.533) | 12 / 12 (1.000) | 32 / 47 (0.115) |
| LOO_noLLM_seed0 | TRI_shadow_seed0 | 26 / 19 (0.371) | 10 / 11 (1.000) | 31 / 60 (0.003) * |
| LOO_noPS_seed0 | S_BP_seed0 | 26 / 14 (0.081) | 11 / 12 (1.000) | 38 / 39 (1.000) |
| LOO_noPS_seed0 | S_LLM_seed0 | 28 / 27 (1.000) | 12 / 16 (0.572) | 37 / 38 (1.000) |
| LOO_noPS_seed0 | S_PS_seed0 | 29 / 10 (0.003) * | 8 / 14 (0.286) | 23 / 43 (0.019) * |
| LOO_noPS_seed0 | T0_baseline_seed0 | 29 / 10 (0.003) * | 8 / 22 (0.016) * | 42 / 43 (1.000) |
| LOO_noPS_seed0 | TRI_equal_seed0 | 22 / 16 (0.418) | 9 / 12 (0.664) | 38 / 36 (0.908) |
| LOO_noPS_seed0 | TRI_shadow_seed0 | 27 / 19 (0.302) | 10 / 14 (0.541) | 32 / 44 (0.207) |
| S_BP_seed0 | S_LLM_seed0 | 13 / 24 (0.099) | 14 / 17 (0.720) | 40 / 40 (1.000) |
| S_BP_seed0 | S_PS_seed0 | 15 / 8 (0.210) | 8 / 13 (0.383) | 20 / 39 (0.018) * |
| S_BP_seed0 | T0_baseline_seed0 | 16 / 9 (0.230) | 7 / 20 (0.019) * | 39 / 39 (1.000) |
| S_BP_seed0 | TRI_equal_seed0 | 15 / 21 (0.405) | 11 / 13 (0.839) | 39 / 36 (0.818) |
| S_BP_seed0 | TRI_shadow_seed0 | 16 / 20 (0.618) | 10 / 13 (0.678) | 28 / 39 (0.222) |
| S_LLM_seed0 | S_PS_seed0 | 24 / 6 (0.001) * | 16 / 18 (0.864) | 28 / 47 (0.037) * |
| S_LLM_seed0 | T0_baseline_seed0 | 28 / 10 (0.005) * | 15 / 25 (0.154) | 44 / 44 (1.000) |
| S_LLM_seed0 | TRI_equal_seed0 | 26 / 21 (0.560) | 14 / 13 (1.000) | 42 / 39 (0.824) |
| S_LLM_seed0 | TRI_shadow_seed0 | 28 / 21 (0.392) | 15 / 15 (1.000) | 34 / 45 (0.260) |
| S_PS_seed0 | T0_baseline_seed0 | 9 / 9 (1.000) | 8 / 16 (0.152) | 38 / 19 (0.016) * |
| S_PS_seed0 | TRI_equal_seed0 | 7 / 20 (0.019) * | 16 / 13 (0.711) | 44 / 22 (0.009) * |
| S_PS_seed0 | TRI_shadow_seed0 | 9 / 20 (0.061) | 13 / 11 (0.839) | 31 / 23 (0.341) |
| T0_baseline_seed0 | TRI_equal_seed0 | 10 / 23 (0.035) * | 22 / 11 (0.080) | 44 / 41 (0.828) |
| T0_baseline_seed0 | TRI_shadow_seed0 | 10 / 21 (0.071) | 20 / 10 (0.099) | 30 / 41 (0.235) |
| TRI_equal_seed0 | TRI_shadow_seed0 | 22 / 20 (0.878) | 11 / 12 (1.000) | 27 / 41 (0.114) |

`*` = p < 0.05. A lower 'A-only' count means A makes that mistake less often than B.

## T4. Rescue analysis on the 57 scenarios the baseline failed

| run | rescued | % of baseline failures |
|---|---|---|
| LOO_noBP_seed0 | 45 | 78.9 |
| S_LLM_seed0 | 44 | 77.2 |
| LOO_noPS_seed0 | 42 | 73.7 |
| TRI_equal_seed0 | 41 | 71.9 |
| TRI_shadow_seed0 | 41 | 71.9 |
| LOO_noLLM_seed0 | 40 | 70.2 |
| S_BP_seed0 | 39 | 68.4 |
| S_PS_seed0 | 38 | 66.7 |
