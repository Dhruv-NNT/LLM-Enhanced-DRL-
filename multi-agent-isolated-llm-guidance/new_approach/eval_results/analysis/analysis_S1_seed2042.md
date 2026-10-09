# Paired analysis — pattern `new_approach/eval_results/setting1_latest/*/evaluation_seed2042.json`

Baseline: **T0_baseline_seed0**. Runs: 9. Common paired scenarios: **300**.

## T1. Each run vs baseline (paired on identical scenarios)

| run | success % | Δ pp | rescued | lost | McNemar p | 95% CI Δpp | real? |
|---|---|---|---|---|---|---|---|
| T0_baseline_seed0 (baseline) | 82.0 | 0.0 | – | – | – | – | – |
| S_PS_seed0 | 86.3 | +4.3 | 38 | 25 | 0.130 | [-1.0, +9.3] | no |
| LOO_noLLM_seed0 | 86.0 | +4.0 | 39 | 27 | 0.175 | [-1.3, +9.3] | no |
| TRI_shadow_seed0 | 86.0 | +4.0 | 41 | 29 | 0.188 | [-1.7, +9.7] | no |
| LOO_noPS_seed0 | 85.7 | +3.7 | 37 | 26 | 0.207 | [-1.7, +8.7] | no |
| S_LLM_seed0 | 85.3 | +3.3 | 40 | 30 | 0.282 | [-2.0, +9.0] | no |
| TRI_equal_seed0 | 84.0 | +2.0 | 39 | 33 | 0.556 | [-3.7, +7.7] | no |
| LOO_noBP_seed0 | 82.0 | +0.0 | 33 | 33 | 1.000 | [-5.3, +5.0] | no |
| S_BP_seed0 | 82.0 | +0.0 | 39 | 39 | 1.000 | [-5.3, +6.0] | no |

*rescued* = baseline failed, run succeeded; *lost* = the reverse. *real?* = p < 0.05 (the split of disagreements is too lopsided to be a coin flip).

## T2. Failure-mode split (% of scenarios)

| run | success | collision | weather | truncated |
|---|---|---|---|---|
| S_PS_seed0 | 86.3 | 3.3 | 6.0 | 4.3 |
| LOO_noLLM_seed0 | 86.0 | 5.3 | 5.0 | 3.7 |
| TRI_shadow_seed0 | 86.0 | 6.3 | 6.0 | 1.7 |
| LOO_noPS_seed0 | 85.7 | 5.0 | 2.0 | 7.3 |
| S_LLM_seed0 | 85.3 | 8.0 | 4.3 | 2.3 |
| TRI_equal_seed0 | 84.0 | 10.3 | 4.3 | 1.3 |
| LOO_noBP_seed0 | 82.0 | 10.0 | 4.0 | 4.0 |
| S_BP_seed0 | 82.0 | 3.7 | 3.7 | 10.7 |
| T0_baseline_seed0 | 82.0 | 5.3 | 6.0 | 6.7 |

## T3. Pairwise complementarity (McNemar on each failure type; counts are 'A-only' vs 'B-only')

| A | B | collision A-only / B-only (p) | weather A-only / B-only (p) | success A-only / B-only (p) |
|---|---|---|---|---|
| LOO_noBP_seed0 | LOO_noLLM_seed0 | 26 / 12 (0.034) * | 6 / 9 (0.607) | 20 / 32 (0.126) |
| LOO_noBP_seed0 | LOO_noPS_seed0 | 22 / 7 (0.008) * | 9 / 3 (0.146) | 24 / 35 (0.193) |
| LOO_noBP_seed0 | S_BP_seed0 | 28 / 9 (0.003) * | 6 / 5 (1.000) | 38 / 38 (1.000) |
| LOO_noBP_seed0 | S_LLM_seed0 | 25 / 19 (0.451) | 8 / 9 (1.000) | 28 / 38 (0.268) |
| LOO_noBP_seed0 | S_PS_seed0 | 27 / 7 (0.001) * | 8 / 14 (0.286) | 28 / 41 (0.148) |
| LOO_noBP_seed0 | T0_baseline_seed0 | 24 / 10 (0.024) * | 7 / 13 (0.263) | 33 / 33 (1.000) |
| LOO_noBP_seed0 | TRI_equal_seed0 | 22 / 23 (1.000) | 8 / 9 (1.000) | 28 / 34 (0.526) |
| LOO_noBP_seed0 | TRI_shadow_seed0 | 25 / 14 (0.108) | 7 / 13 (0.263) | 23 / 35 (0.148) |
| LOO_noLLM_seed0 | LOO_noPS_seed0 | 12 / 11 (1.000) | 12 / 3 (0.035) * | 27 / 26 (1.000) |
| LOO_noLLM_seed0 | S_BP_seed0 | 15 / 10 (0.424) | 11 / 7 (0.481) | 39 / 27 (0.175) |
| LOO_noLLM_seed0 | S_LLM_seed0 | 15 / 23 (0.256) | 11 / 9 (0.824) | 34 / 32 (0.902) |
| LOO_noLLM_seed0 | S_PS_seed0 | 14 / 8 (0.286) | 8 / 11 (0.648) | 27 / 28 (1.000) |
| LOO_noLLM_seed0 | T0_baseline_seed0 | 13 / 13 (1.000) | 9 / 12 (0.664) | 39 / 27 (0.175) |
| LOO_noLLM_seed0 | TRI_equal_seed0 | 14 / 29 (0.032) * | 10 / 8 (0.815) | 34 / 28 (0.526) |
| LOO_noLLM_seed0 | TRI_shadow_seed0 | 15 / 18 (0.728) | 9 / 12 (0.664) | 24 / 24 (1.000) |
| LOO_noPS_seed0 | S_BP_seed0 | 12 / 8 (0.503) | 3 / 8 (0.227) | 40 / 29 (0.228) |
| LOO_noPS_seed0 | S_LLM_seed0 | 11 / 20 (0.150) | 3 / 10 (0.092) | 34 / 33 (1.000) |
| LOO_noPS_seed0 | S_PS_seed0 | 13 / 8 (0.383) | 3 / 15 (0.008) * | 30 / 32 (0.899) |
| LOO_noPS_seed0 | T0_baseline_seed0 | 10 / 11 (1.000) | 3 / 15 (0.008) * | 37 / 26 (0.207) |
| LOO_noPS_seed0 | TRI_equal_seed0 | 12 / 28 (0.017) * | 3 / 10 (0.092) | 33 / 28 (0.609) |
| LOO_noPS_seed0 | TRI_shadow_seed0 | 15 / 19 (0.608) | 3 / 15 (0.008) * | 29 / 30 (1.000) |
| S_BP_seed0 | S_LLM_seed0 | 8 / 21 (0.024) * | 7 / 9 (0.804) | 24 / 34 (0.237) |
| S_BP_seed0 | S_PS_seed0 | 11 / 10 (1.000) | 6 / 13 (0.167) | 27 / 40 (0.142) |
| S_BP_seed0 | T0_baseline_seed0 | 11 / 16 (0.442) | 7 / 14 (0.189) | 39 / 39 (1.000) |
| S_BP_seed0 | TRI_equal_seed0 | 7 / 27 (0.001) * | 6 / 8 (0.791) | 29 / 35 (0.532) |
| S_BP_seed0 | TRI_shadow_seed0 | 11 / 19 (0.200) | 5 / 12 (0.143) | 28 / 40 (0.182) |
| S_LLM_seed0 | S_PS_seed0 | 22 / 8 (0.016) * | 8 / 13 (0.383) | 30 / 33 (0.801) |
| S_LLM_seed0 | T0_baseline_seed0 | 22 / 14 (0.243) | 9 / 14 (0.405) | 40 / 30 (0.282) |
| S_LLM_seed0 | TRI_equal_seed0 | 18 / 25 (0.360) | 9 / 9 (1.000) | 33 / 29 (0.704) |
| S_LLM_seed0 | TRI_shadow_seed0 | 23 / 18 (0.533) | 10 / 15 (0.424) | 32 / 34 (0.902) |
| S_PS_seed0 | T0_baseline_seed0 | 6 / 12 (0.238) | 10 / 10 (1.000) | 38 / 25 (0.130) |
| S_PS_seed0 | TRI_equal_seed0 | 8 / 29 (0.001) * | 14 / 9 (0.405) | 39 / 32 (0.477) |
| S_PS_seed0 | TRI_shadow_seed0 | 9 / 18 (0.122) | 11 / 11 (1.000) | 30 / 29 (1.000) |
| T0_baseline_seed0 | TRI_equal_seed0 | 14 / 29 (0.032) * | 13 / 8 (0.383) | 33 / 39 (0.556) |
| T0_baseline_seed0 | TRI_shadow_seed0 | 15 / 18 (0.728) | 12 / 12 (1.000) | 29 / 41 (0.188) |
| TRI_equal_seed0 | TRI_shadow_seed0 | 29 / 17 (0.104) | 8 / 13 (0.383) | 29 / 35 (0.532) |

`*` = p < 0.05. A lower 'A-only' count means A makes that mistake less often than B.

## T4. Rescue analysis on the 54 scenarios the baseline failed

| run | rescued | % of baseline failures |
|---|---|---|
| TRI_shadow_seed0 | 41 | 75.9 |
| S_LLM_seed0 | 40 | 74.1 |
| LOO_noLLM_seed0 | 39 | 72.2 |
| S_BP_seed0 | 39 | 72.2 |
| TRI_equal_seed0 | 39 | 72.2 |
| S_PS_seed0 | 38 | 70.4 |
| LOO_noPS_seed0 | 37 | 68.5 |
| LOO_noBP_seed0 | 33 | 61.1 |
