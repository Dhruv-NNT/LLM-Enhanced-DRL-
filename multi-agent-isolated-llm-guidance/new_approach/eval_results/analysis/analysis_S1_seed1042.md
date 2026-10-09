# Paired analysis — pattern `new_approach/eval_results/setting1_latest/*/evaluation_seed1042.json`

Baseline: **T0_baseline_seed0**. Runs: 9. Common paired scenarios: **300**.

## T1. Each run vs baseline (paired on identical scenarios)

| run | success % | Δ pp | rescued | lost | McNemar p | 95% CI Δpp | real? |
|---|---|---|---|---|---|---|---|
| T0_baseline_seed0 (baseline) | 82.7 | 0.0 | – | – | – | – | – |
| S_PS_seed0 | 86.7 | +4.0 | 32 | 20 | 0.126 | [-0.7, +8.7] | no |
| S_LLM_seed0 | 86.3 | +3.7 | 33 | 22 | 0.177 | [-1.3, +8.7] | no |
| LOO_noLLM_seed0 | 86.0 | +3.3 | 41 | 31 | 0.289 | [-2.0, +8.3] | no |
| TRI_shadow_seed0 | 84.7 | +2.0 | 35 | 29 | 0.532 | [-3.3, +7.0] | no |
| LOO_noBP_seed0 | 83.7 | +1.0 | 35 | 32 | 0.807 | [-4.3, +6.3] | no |
| S_BP_seed0 | 83.7 | +1.0 | 33 | 30 | 0.801 | [-4.3, +6.3] | no |
| LOO_noPS_seed0 | 81.7 | -1.0 | 35 | 38 | 0.815 | [-6.3, +4.7] | no |
| TRI_equal_seed0 | 79.7 | -3.0 | 34 | 43 | 0.362 | [-8.7, +2.7] | no |

*rescued* = baseline failed, run succeeded; *lost* = the reverse. *real?* = p < 0.05 (the split of disagreements is too lopsided to be a coin flip).

## T2. Failure-mode split (% of scenarios)

| run | success | collision | weather | truncated |
|---|---|---|---|---|
| S_PS_seed0 | 86.7 | 2.0 | 7.3 | 4.0 |
| S_LLM_seed0 | 86.3 | 6.0 | 5.3 | 2.3 |
| LOO_noLLM_seed0 | 86.0 | 6.3 | 4.7 | 3.0 |
| TRI_shadow_seed0 | 84.7 | 7.3 | 6.7 | 1.3 |
| LOO_noBP_seed0 | 83.7 | 6.7 | 4.7 | 5.0 |
| S_BP_seed0 | 83.7 | 3.7 | 6.3 | 6.3 |
| T0_baseline_seed0 | 82.7 | 3.7 | 6.7 | 7.0 |
| LOO_noPS_seed0 | 81.7 | 2.3 | 5.7 | 10.3 |
| TRI_equal_seed0 | 79.7 | 12.3 | 4.7 | 3.3 |

## T3. Pairwise complementarity (McNemar on each failure type; counts are 'A-only' vs 'B-only')

| A | B | collision A-only / B-only (p) | weather A-only / B-only (p) | success A-only / B-only (p) |
|---|---|---|---|---|
| LOO_noBP_seed0 | LOO_noLLM_seed0 | 18 / 17 (1.000) | 10 / 10 (1.000) | 25 / 32 (0.427) |
| LOO_noBP_seed0 | LOO_noPS_seed0 | 17 / 4 (0.007) * | 9 / 12 (0.664) | 38 / 32 (0.550) |
| LOO_noBP_seed0 | S_BP_seed0 | 18 / 9 (0.122) | 10 / 15 (0.424) | 35 / 35 (1.000) |
| LOO_noBP_seed0 | S_LLM_seed0 | 18 / 16 (0.864) | 10 / 12 (0.832) | 29 / 37 (0.389) |
| LOO_noBP_seed0 | S_PS_seed0 | 19 / 5 (0.007) * | 8 / 16 (0.152) | 30 / 39 (0.336) |
| LOO_noBP_seed0 | T0_baseline_seed0 | 18 / 9 (0.122) | 6 / 12 (0.238) | 35 / 32 (0.807) |
| LOO_noBP_seed0 | TRI_equal_seed0 | 17 / 34 (0.024) * | 9 / 9 (1.000) | 45 / 33 (0.213) |
| LOO_noBP_seed0 | TRI_shadow_seed0 | 18 / 20 (0.871) | 8 / 14 (0.286) | 31 / 34 (0.804) |
| LOO_noLLM_seed0 | LOO_noPS_seed0 | 19 / 7 (0.029) * | 8 / 11 (0.648) | 38 / 25 (0.130) |
| LOO_noLLM_seed0 | S_BP_seed0 | 18 / 10 (0.185) | 9 / 14 (0.405) | 38 / 31 (0.470) |
| LOO_noLLM_seed0 | S_LLM_seed0 | 15 / 14 (1.000) | 6 / 8 (0.791) | 26 / 27 (1.000) |
| LOO_noLLM_seed0 | S_PS_seed0 | 19 / 6 (0.015) * | 8 / 16 (0.152) | 32 / 34 (0.902) |
| LOO_noLLM_seed0 | T0_baseline_seed0 | 19 / 11 (0.200) | 8 / 14 (0.286) | 41 / 31 (0.289) |
| LOO_noLLM_seed0 | TRI_equal_seed0 | 15 / 33 (0.013) * | 10 / 10 (1.000) | 44 / 25 (0.029) * |
| LOO_noLLM_seed0 | TRI_shadow_seed0 | 18 / 21 (0.749) | 6 / 12 (0.238) | 32 / 28 (0.699) |
| LOO_noPS_seed0 | S_BP_seed0 | 7 / 11 (0.481) | 14 / 16 (0.856) | 35 / 41 (0.567) |
| LOO_noPS_seed0 | S_LLM_seed0 | 5 / 16 (0.027) * | 13 / 12 (1.000) | 28 / 42 (0.120) |
| LOO_noPS_seed0 | S_PS_seed0 | 6 / 5 (1.000) | 8 / 13 (0.383) | 24 / 39 (0.077) |
| LOO_noPS_seed0 | T0_baseline_seed0 | 5 / 9 (0.424) | 11 / 14 (0.690) | 35 / 38 (0.815) |
| LOO_noPS_seed0 | TRI_equal_seed0 | 4 / 34 (0.000) * | 11 / 8 (0.648) | 41 / 35 (0.567) |
| LOO_noPS_seed0 | TRI_shadow_seed0 | 6 / 21 (0.006) * | 8 / 11 (0.648) | 27 / 36 (0.314) |
| S_BP_seed0 | S_LLM_seed0 | 11 / 18 (0.265) | 12 / 9 (0.664) | 26 / 34 (0.366) |
| S_BP_seed0 | S_PS_seed0 | 11 / 6 (0.332) | 16 / 19 (0.736) | 28 / 37 (0.321) |
| S_BP_seed0 | T0_baseline_seed0 | 11 / 11 (1.000) | 14 / 15 (1.000) | 33 / 30 (0.801) |
| S_BP_seed0 | TRI_equal_seed0 | 9 / 35 (0.000) * | 14 / 9 (0.405) | 48 / 36 (0.230) |
| S_BP_seed0 | TRI_shadow_seed0 | 11 / 22 (0.080) | 14 / 15 (1.000) | 32 / 35 (0.807) |
| S_LLM_seed0 | S_PS_seed0 | 18 / 6 (0.023) * | 11 / 17 (0.345) | 30 / 31 (1.000) |
| S_LLM_seed0 | T0_baseline_seed0 | 15 / 8 (0.210) | 9 / 13 (0.523) | 33 / 22 (0.177) |
| S_LLM_seed0 | TRI_equal_seed0 | 11 / 30 (0.004) * | 11 / 9 (0.824) | 45 / 25 (0.022) * |
| S_LLM_seed0 | TRI_shadow_seed0 | 16 / 20 (0.618) | 10 / 14 (0.541) | 33 / 28 (0.609) |
| S_PS_seed0 | T0_baseline_seed0 | 4 / 9 (0.267) | 13 / 11 (0.839) | 32 / 20 (0.126) |
| S_PS_seed0 | TRI_equal_seed0 | 6 / 37 (0.000) * | 16 / 8 (0.152) | 50 / 29 (0.024) * |
| S_PS_seed0 | TRI_shadow_seed0 | 6 / 22 (0.004) * | 14 / 12 (0.845) | 34 / 28 (0.526) |
| T0_baseline_seed0 | TRI_equal_seed0 | 9 / 35 (0.000) * | 14 / 8 (0.286) | 43 / 34 (0.362) |
| T0_baseline_seed0 | TRI_shadow_seed0 | 9 / 20 (0.061) | 9 / 9 (1.000) | 29 / 35 (0.532) |
| TRI_equal_seed0 | TRI_shadow_seed0 | 32 / 17 (0.044) * | 8 / 14 (0.286) | 29 / 44 (0.101) |

`*` = p < 0.05. A lower 'A-only' count means A makes that mistake less often than B.

## T4. Rescue analysis on the 52 scenarios the baseline failed

| run | rescued | % of baseline failures |
|---|---|---|
| LOO_noLLM_seed0 | 41 | 78.8 |
| LOO_noBP_seed0 | 35 | 67.3 |
| LOO_noPS_seed0 | 35 | 67.3 |
| TRI_shadow_seed0 | 35 | 67.3 |
| TRI_equal_seed0 | 34 | 65.4 |
| S_BP_seed0 | 33 | 63.5 |
| S_LLM_seed0 | 33 | 63.5 |
| S_PS_seed0 | 32 | 61.5 |
