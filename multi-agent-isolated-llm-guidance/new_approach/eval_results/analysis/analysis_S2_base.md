# Paired analysis — pattern `new_approach/eval_results/setting2_3M/*/evaluation.json`

Baseline: **T0_baseline_seed0**. Runs: 9. Common paired scenarios: **300**.

## T1. Each run vs baseline (paired on identical scenarios)

| run | success % | Δ pp | rescued | lost | McNemar p | 95% CI Δpp | real? |
|---|---|---|---|---|---|---|---|
| T0_baseline_seed0 (baseline) | 81.0 | 0.0 | – | – | – | – | – |
| LOO_noBP_seed0 | 87.0 | +6.0 | 41 | 23 | 0.033 | [+1.0, +11.3] | YES |
| TRI_shadow_seed0 | 86.7 | +5.7 | 43 | 26 | 0.053 | [+0.3, +11.0] | no |
| S_PS_seed0 | 84.3 | +3.3 | 37 | 27 | 0.260 | [-1.7, +8.7] | no |
| S_LLM_seed0 | 83.0 | +2.0 | 45 | 39 | 0.586 | [-4.0, +8.0] | no |
| S_BP_seed0 | 82.0 | +1.0 | 39 | 36 | 0.818 | [-4.7, +6.7] | no |
| LOO_noPS_seed0 | 81.0 | +0.0 | 44 | 44 | 1.000 | [-6.3, +6.0] | no |
| TRI_equal_seed0 | 78.0 | -3.0 | 38 | 47 | 0.386 | [-9.0, +3.3] | no |
| LOO_noLLM_seed0 | 76.0 | -5.0 | 40 | 55 | 0.151 | [-11.3, +1.3] | no |

*rescued* = baseline failed, run succeeded; *lost* = the reverse. *real?* = p < 0.05 (the split of disagreements is too lopsided to be a coin flip).

## T2. Failure-mode split (% of scenarios)

| run | success | collision | weather | truncated |
|---|---|---|---|---|
| LOO_noBP_seed0 | 87.0 | 5.0 | 7.0 | 1.0 |
| TRI_shadow_seed0 | 86.7 | 5.7 | 6.0 | 1.7 |
| S_PS_seed0 | 84.3 | 5.7 | 9.7 | 0.3 |
| S_LLM_seed0 | 83.0 | 11.0 | 4.7 | 1.3 |
| S_BP_seed0 | 82.0 | 5.7 | 8.0 | 4.3 |
| LOO_noPS_seed0 | 81.0 | 10.0 | 4.7 | 4.3 |
| T0_baseline_seed0 | 81.0 | 2.0 | 10.3 | 6.7 |
| TRI_equal_seed0 | 78.0 | 12.3 | 5.7 | 4.0 |
| LOO_noLLM_seed0 | 76.0 | 8.0 | 6.3 | 9.7 |

## T3. Pairwise complementarity (McNemar on each failure type; counts are 'A-only' vs 'B-only')

| A | B | collision A-only / B-only (p) | weather A-only / B-only (p) | success A-only / B-only (p) |
|---|---|---|---|---|
| LOO_noBP_seed0 | LOO_noLLM_seed0 | 14 / 23 (0.188) | 12 / 10 (0.832) | 58 / 25 (0.000) * |
| LOO_noBP_seed0 | LOO_noPS_seed0 | 14 / 29 (0.032) * | 14 / 7 (0.189) | 44 / 26 (0.041) * |
| LOO_noBP_seed0 | S_BP_seed0 | 13 / 15 (0.851) | 11 / 14 (0.690) | 38 / 23 (0.072) |
| LOO_noBP_seed0 | S_LLM_seed0 | 12 / 30 (0.008) * | 13 / 6 (0.167) | 35 / 23 (0.148) |
| LOO_noBP_seed0 | S_PS_seed0 | 14 / 16 (0.856) | 11 / 19 (0.200) | 32 / 24 (0.350) |
| LOO_noBP_seed0 | T0_baseline_seed0 | 15 / 6 (0.078) | 11 / 21 (0.110) | 41 / 23 (0.033) * |
| LOO_noBP_seed0 | TRI_equal_seed0 | 13 / 35 (0.002) * | 13 / 9 (0.523) | 49 / 22 (0.002) * |
| LOO_noBP_seed0 | TRI_shadow_seed0 | 13 / 15 (0.851) | 13 / 10 (0.678) | 27 / 26 (1.000) |
| LOO_noLLM_seed0 | LOO_noPS_seed0 | 18 / 24 (0.441) | 11 / 6 (0.332) | 28 / 43 (0.096) |
| LOO_noLLM_seed0 | S_BP_seed0 | 22 / 15 (0.324) | 9 / 14 (0.405) | 31 / 49 (0.057) |
| LOO_noLLM_seed0 | S_LLM_seed0 | 18 / 27 (0.233) | 11 / 6 (0.332) | 29 / 50 (0.024) * |
| LOO_noLLM_seed0 | S_PS_seed0 | 23 / 16 (0.337) | 9 / 19 (0.087) | 25 / 50 (0.005) * |
| LOO_noLLM_seed0 | T0_baseline_seed0 | 23 / 5 (0.001) * | 10 / 22 (0.050) | 40 / 55 (0.151) |
| LOO_noLLM_seed0 | TRI_equal_seed0 | 20 / 33 (0.098) | 11 / 9 (0.824) | 45 / 51 (0.610) |
| LOO_noLLM_seed0 | TRI_shadow_seed0 | 22 / 15 (0.324) | 11 / 10 (1.000) | 23 / 55 (0.000) * |
| LOO_noPS_seed0 | S_BP_seed0 | 28 / 15 (0.066) | 7 / 17 (0.064) | 34 / 37 (0.813) |
| LOO_noPS_seed0 | S_LLM_seed0 | 23 / 26 (0.775) | 7 / 7 (1.000) | 32 / 38 (0.550) |
| LOO_noPS_seed0 | S_PS_seed0 | 28 / 15 (0.066) | 7 / 22 (0.008) * | 29 / 39 (0.275) |
| LOO_noPS_seed0 | T0_baseline_seed0 | 30 / 6 (0.000) * | 6 / 23 (0.002) * | 44 / 44 (1.000) |
| LOO_noPS_seed0 | TRI_equal_seed0 | 26 / 33 (0.435) | 7 / 10 (0.629) | 43 / 34 (0.362) |
| LOO_noPS_seed0 | TRI_shadow_seed0 | 24 / 11 (0.041) * | 7 / 11 (0.481) | 23 / 40 (0.043) * |
| S_BP_seed0 | S_LLM_seed0 | 15 / 31 (0.026) * | 16 / 6 (0.052) | 34 / 37 (0.813) |
| S_BP_seed0 | S_PS_seed0 | 16 / 16 (1.000) | 8 / 13 (0.383) | 27 / 34 (0.443) |
| S_BP_seed0 | T0_baseline_seed0 | 16 / 5 (0.027) * | 10 / 17 (0.248) | 39 / 36 (0.818) |
| S_BP_seed0 | TRI_equal_seed0 | 12 / 32 (0.004) * | 15 / 8 (0.210) | 42 / 30 (0.195) |
| S_BP_seed0 | TRI_shadow_seed0 | 16 / 16 (1.000) | 14 / 8 (0.286) | 23 / 37 (0.092) |
| S_LLM_seed0 | S_PS_seed0 | 32 / 16 (0.029) * | 7 / 22 (0.008) * | 31 / 35 (0.712) |
| S_LLM_seed0 | T0_baseline_seed0 | 33 / 6 (0.000) * | 7 / 24 (0.003) * | 45 / 39 (0.586) |
| S_LLM_seed0 | TRI_equal_seed0 | 24 / 28 (0.678) | 6 / 9 (0.607) | 46 / 31 (0.110) |
| S_LLM_seed0 | TRI_shadow_seed0 | 30 / 14 (0.023) * | 6 / 10 (0.454) | 24 / 35 (0.193) |
| S_PS_seed0 | T0_baseline_seed0 | 16 / 5 (0.027) * | 13 / 15 (0.851) | 37 / 27 (0.260) |
| S_PS_seed0 | TRI_equal_seed0 | 15 / 35 (0.007) * | 20 / 8 (0.036) * | 45 / 26 (0.032) * |
| S_PS_seed0 | TRI_shadow_seed0 | 17 / 17 (1.000) | 19 / 8 (0.052) | 26 / 33 (0.435) |
| T0_baseline_seed0 | TRI_equal_seed0 | 6 / 37 (0.000) * | 22 / 8 (0.016) * | 47 / 38 (0.386) |
| T0_baseline_seed0 | TRI_shadow_seed0 | 6 / 17 (0.035) * | 21 / 8 (0.024) * | 26 / 43 (0.053) |
| TRI_equal_seed0 | TRI_shadow_seed0 | 35 / 15 (0.007) * | 8 / 9 (1.000) | 21 / 47 (0.002) * |

`*` = p < 0.05. A lower 'A-only' count means A makes that mistake less often than B.

## T4. Rescue analysis on the 57 scenarios the baseline failed

| run | rescued | % of baseline failures |
|---|---|---|
| S_LLM_seed0 | 45 | 78.9 |
| LOO_noPS_seed0 | 44 | 77.2 |
| TRI_shadow_seed0 | 43 | 75.4 |
| LOO_noBP_seed0 | 41 | 71.9 |
| LOO_noLLM_seed0 | 40 | 70.2 |
| S_BP_seed0 | 39 | 68.4 |
| TRI_equal_seed0 | 38 | 66.7 |
| S_PS_seed0 | 37 | 64.9 |
