# Paired analysis — pattern `new_approach/eval_results/setting2_3M/*/evaluation_seed2042.json`

Baseline: **T0_baseline_seed0**. Runs: 9. Common paired scenarios: **300**.

## T1. Each run vs baseline (paired on identical scenarios)

| run | success % | Δ pp | rescued | lost | McNemar p | 95% CI Δpp | real? |
|---|---|---|---|---|---|---|---|
| T0_baseline_seed0 (baseline) | 83.7 | 0.0 | – | – | – | – | – |
| LOO_noBP_seed0 | 88.3 | +4.7 | 39 | 25 | 0.103 | [-0.3, +10.0] | no |
| S_PS_seed0 | 86.0 | +2.3 | 29 | 22 | 0.401 | [-2.3, +7.0] | no |
| TRI_shadow_seed0 | 86.0 | +2.3 | 35 | 28 | 0.450 | [-3.0, +7.7] | no |
| TRI_equal_seed0 | 82.7 | -1.0 | 34 | 37 | 0.813 | [-6.7, +4.3] | no |
| LOO_noPS_seed0 | 80.7 | -3.0 | 34 | 43 | 0.362 | [-8.7, +2.7] | no |
| S_LLM_seed0 | 80.0 | -3.7 | 35 | 46 | 0.266 | [-9.3, +2.0] | no |
| S_BP_seed0 | 79.3 | -4.3 | 30 | 43 | 0.160 | [-10.0, +1.3] | no |
| LOO_noLLM_seed0 | 76.0 | -7.7 | 25 | 48 | 0.010 | [-13.3, -2.3] | YES |

*rescued* = baseline failed, run succeeded; *lost* = the reverse. *real?* = p < 0.05 (the split of disagreements is too lopsided to be a coin flip).

## T2. Failure-mode split (% of scenarios)

| run | success | collision | weather | truncated |
|---|---|---|---|---|
| LOO_noBP_seed0 | 88.3 | 6.3 | 4.3 | 1.0 |
| S_PS_seed0 | 86.0 | 6.0 | 6.7 | 1.3 |
| TRI_shadow_seed0 | 86.0 | 6.3 | 6.0 | 1.7 |
| T0_baseline_seed0 | 83.7 | 6.3 | 6.7 | 3.3 |
| TRI_equal_seed0 | 82.7 | 10.7 | 3.7 | 3.0 |
| LOO_noPS_seed0 | 80.7 | 10.3 | 3.7 | 5.3 |
| S_LLM_seed0 | 80.0 | 11.7 | 3.3 | 5.0 |
| S_BP_seed0 | 79.3 | 8.0 | 6.3 | 6.3 |
| LOO_noLLM_seed0 | 76.0 | 11.7 | 7.0 | 5.3 |

## T3. Pairwise complementarity (McNemar on each failure type; counts are 'A-only' vs 'B-only')

| A | B | collision A-only / B-only (p) | weather A-only / B-only (p) | success A-only / B-only (p) |
|---|---|---|---|---|
| LOO_noBP_seed0 | LOO_noLLM_seed0 | 14 / 30 (0.023) * | 5 / 13 (0.096) | 54 / 17 (0.000) * |
| LOO_noBP_seed0 | LOO_noPS_seed0 | 16 / 28 (0.096) | 10 / 8 (0.815) | 46 / 23 (0.008) * |
| LOO_noBP_seed0 | S_BP_seed0 | 16 / 21 (0.511) | 8 / 14 (0.286) | 46 / 19 (0.001) * |
| LOO_noBP_seed0 | S_LLM_seed0 | 15 / 31 (0.026) * | 9 / 6 (0.607) | 44 / 19 (0.002) * |
| LOO_noBP_seed0 | S_PS_seed0 | 18 / 17 (1.000) | 9 / 16 (0.230) | 29 / 22 (0.401) |
| LOO_noBP_seed0 | T0_baseline_seed0 | 18 / 18 (1.000) | 8 / 15 (0.210) | 39 / 25 (0.103) |
| LOO_noBP_seed0 | TRI_equal_seed0 | 18 / 31 (0.085) | 9 / 7 (0.804) | 43 / 26 (0.053) |
| LOO_noBP_seed0 | TRI_shadow_seed0 | 19 / 19 (1.000) | 8 / 13 (0.383) | 31 / 24 (0.419) |
| LOO_noLLM_seed0 | LOO_noPS_seed0 | 21 / 17 (0.627) | 17 / 7 (0.064) | 29 / 43 (0.125) |
| LOO_noLLM_seed0 | S_BP_seed0 | 28 / 17 (0.135) | 12 / 10 (0.832) | 33 / 43 (0.302) |
| LOO_noLLM_seed0 | S_LLM_seed0 | 30 / 30 (1.000) | 17 / 6 (0.035) * | 39 / 51 (0.246) |
| LOO_noLLM_seed0 | S_PS_seed0 | 28 / 11 (0.009) * | 14 / 13 (1.000) | 23 / 53 (0.001) * |
| LOO_noLLM_seed0 | T0_baseline_seed0 | 32 / 16 (0.029) * | 13 / 12 (1.000) | 25 / 48 (0.010) * |
| LOO_noLLM_seed0 | TRI_equal_seed0 | 27 / 24 (0.780) | 16 / 6 (0.052) | 28 / 48 (0.029) * |
| LOO_noLLM_seed0 | TRI_shadow_seed0 | 31 / 15 (0.026) * | 14 / 11 (0.690) | 22 / 52 (0.001) * |
| LOO_noPS_seed0 | S_BP_seed0 | 25 / 18 (0.360) | 8 / 16 (0.152) | 39 / 35 (0.728) |
| LOO_noPS_seed0 | S_LLM_seed0 | 26 / 30 (0.689) | 8 / 7 (1.000) | 41 / 39 (0.911) |
| LOO_noPS_seed0 | S_PS_seed0 | 25 / 12 (0.047) * | 8 / 17 (0.108) | 27 / 43 (0.072) |
| LOO_noPS_seed0 | T0_baseline_seed0 | 27 / 15 (0.088) | 7 / 16 (0.093) | 34 / 43 (0.362) |
| LOO_noPS_seed0 | TRI_equal_seed0 | 25 / 26 (1.000) | 8 / 8 (1.000) | 35 / 41 (0.567) |
| LOO_noPS_seed0 | TRI_shadow_seed0 | 27 / 15 (0.088) | 7 / 14 (0.189) | 26 / 42 (0.068) |
| S_BP_seed0 | S_LLM_seed0 | 23 / 34 (0.185) | 14 / 5 (0.064) | 40 / 42 (0.912) |
| S_BP_seed0 | S_PS_seed0 | 20 / 14 (0.392) | 8 / 9 (1.000) | 21 / 41 (0.015) * |
| S_BP_seed0 | T0_baseline_seed0 | 19 / 14 (0.487) | 10 / 11 (1.000) | 30 / 43 (0.160) |
| S_BP_seed0 | TRI_equal_seed0 | 18 / 26 (0.291) | 14 / 6 (0.115) | 33 / 43 (0.302) |
| S_BP_seed0 | TRI_shadow_seed0 | 23 / 18 (0.533) | 12 / 11 (1.000) | 28 / 48 (0.029) * |
| S_LLM_seed0 | S_PS_seed0 | 29 / 12 (0.012) * | 7 / 17 (0.064) | 22 / 40 (0.030) * |
| S_LLM_seed0 | T0_baseline_seed0 | 34 / 18 (0.036) * | 7 / 17 (0.064) | 35 / 46 (0.266) |
| S_LLM_seed0 | TRI_equal_seed0 | 26 / 23 (0.775) | 6 / 7 (1.000) | 35 / 43 (0.428) |
| S_LLM_seed0 | TRI_shadow_seed0 | 30 / 14 (0.023) * | 7 / 15 (0.134) | 25 / 43 (0.038) * |
| S_PS_seed0 | T0_baseline_seed0 | 13 / 14 (1.000) | 11 / 11 (1.000) | 29 / 22 (0.401) |
| S_PS_seed0 | TRI_equal_seed0 | 11 / 25 (0.029) * | 16 / 7 (0.093) | 38 / 28 (0.268) |
| S_PS_seed0 | TRI_shadow_seed0 | 15 / 16 (1.000) | 14 / 12 (0.845) | 28 / 28 (1.000) |
| T0_baseline_seed0 | TRI_equal_seed0 | 14 / 27 (0.060) | 16 / 7 (0.093) | 37 / 34 (0.813) |
| T0_baseline_seed0 | TRI_shadow_seed0 | 18 / 18 (1.000) | 13 / 11 (0.839) | 28 / 35 (0.450) |
| TRI_equal_seed0 | TRI_shadow_seed0 | 28 / 15 (0.066) | 6 / 13 (0.167) | 25 / 35 (0.245) |

`*` = p < 0.05. A lower 'A-only' count means A makes that mistake less often than B.

## T4. Rescue analysis on the 49 scenarios the baseline failed

| run | rescued | % of baseline failures |
|---|---|---|
| LOO_noBP_seed0 | 39 | 79.6 |
| S_LLM_seed0 | 35 | 71.4 |
| TRI_shadow_seed0 | 35 | 71.4 |
| LOO_noPS_seed0 | 34 | 69.4 |
| TRI_equal_seed0 | 34 | 69.4 |
| S_BP_seed0 | 30 | 61.2 |
| S_PS_seed0 | 29 | 59.2 |
| LOO_noLLM_seed0 | 25 | 51.0 |
