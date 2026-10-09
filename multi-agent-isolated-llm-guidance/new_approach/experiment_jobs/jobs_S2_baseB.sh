#!/bin/bash
set -u
cd "/home/aradhya.dhruv/rl/behaviour_cloning/LLM enhanced DRL/LLM-DRL restructured code/multi-agent-isolated-llm-guidance" || exit 1
export CUDA_VISIBLE_DEVICES=""                       # CPU only: never disturbs the 3 GPU training runs
PY=/home/aradhya.dhruv/anaconda3/envs/llmdrl/bin/python
EV=Evaluation_distillation_approach                  # where the trained weights live (read-only)
OUT=new_approach/eval_results                        # where NEW results are written
run() { echo "[job $(date +%H:%M:%S)] $*"; [ "${DRY:-0}" = 1 ] && return 0; "$@"; }

# ---- SEED-0 PILOTS ONLY (9). Headline pilots first so key results land earliest. ----
RUNS="TRI_shadow_seed0 S_LLM_seed0 TRI_equal_seed0 T0_baseline_seed0 S_PS_seed0 LOO_noLLM_seed0 LOO_noBP_seed0 LOO_noPS_seed0 S_BP_seed0"

# ---- SETTING 1 "latest weight": finished -> last_checkpoint.pt ; still-training -> newest FROZEN snapshot ----
w1() { case "$1" in
  T0_baseline_seed0) echo last_checkpoint.pt ;;      # 5,000,051 steps (finished)
  S_BP_seed0)        echo last_checkpoint.pt ;;      # 5,000,036 (finished)
  S_PS_seed0)        echo last_checkpoint.pt ;;      # 5,000,012 (finished)
  S_LLM_seed0)       echo last_checkpoint.pt ;;      # 5,000,004 (finished)
  LOO_noLLM_seed0)   echo last_checkpoint.pt ;;      # 5,000,094 (finished)
  TRI_equal_seed0)   echo last_checkpoint.pt ;;      # 5,000,033 (finished)
  LOO_noBP_seed0)    echo checkpoint_4147227.pt ;;   # STILL TRAINING -> newest frozen (4.15M)
  LOO_noPS_seed0)    echo checkpoint_4120186.pt ;;   # STILL TRAINING -> newest frozen (4.12M)
  TRI_shadow_seed0)  echo checkpoint_3280741.pt ;;   # STILL TRAINING -> newest frozen (3.28M)
esac; }

# ---- SETTING 2 "3M snapshot": every pilot at the same age (end of the teaching ramp) ----
w2() { case "$1" in
  T0_baseline_seed0) echo checkpoint_3000307.pt ;;
  S_BP_seed0)        echo checkpoint_3332359.pt ;;
  S_PS_seed0)        echo checkpoint_3263303.pt ;;
  S_LLM_seed0)       echo checkpoint_3103544.pt ;;
  LOO_noLLM_seed0)   echo checkpoint_2999287.pt ;;   # 713 steps under 3M = the boundary snapshot
  TRI_equal_seed0)   echo checkpoint_3000458.pt ;;
  LOO_noBP_seed0)    echo checkpoint_3304236.pt ;;
  LOO_noPS_seed0)    echo checkpoint_3285558.pt ;;
  TRI_shadow_seed0)  echo checkpoint_3280741.pt ;;   # same file as Setting 1 (its newest frozen IS ~3.3M)
esac; }

echo "=== S2: BASE (seed 42) then EXTRA SEEDS (1042, 2042) — 300 flights, 4 aircraft, 2 storms ==="
for RUN in $RUNS; do mkdir -p $OUT/setting2_3M/$RUN; done

# --- Base: the headline table for this setting ---
for RUN in $RUNS; do
  run $PY evaluate_rl.py --model-path $EV/$RUN/weights/$(w2 $RUN) \
      --n-episodes 300 --num-agents 4 --num-weather-cells 2 --seed 42 \
      --metrics-path $OUT/setting2_3M/$RUN/evaluation.json
done
run $PY new_approach/analyze_pairwise.py --pattern "$OUT/setting2_3M/*/evaluation.json" \
    --baseline T0_baseline_seed0 --out $OUT/analysis/analysis_S2_base.md

# --- Extra flight sets: shrink the error bars ---
for SEED in 1042 2042; do
  for RUN in $RUNS; do
    run $PY evaluate_rl.py --model-path $EV/$RUN/weights/$(w2 $RUN) \
        --n-episodes 300 --num-agents 4 --num-weather-cells 2 --seed $SEED \
        --metrics-path $OUT/setting2_3M/$RUN/evaluation_seed$SEED.json
  done
  run $PY new_approach/analyze_pairwise.py --pattern "$OUT/setting2_3M/*/evaluation_seed$SEED.json" \
      --baseline T0_baseline_seed0 --out $OUT/analysis/analysis_S2_seed$SEED.md
done
echo "=== S2_baseB DONE ==="
