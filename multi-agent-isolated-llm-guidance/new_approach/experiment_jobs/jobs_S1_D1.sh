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

echo "=== S1_D1: AAMAS scalability — per-aircraft report card at 8, 6, 4 aircraft (300 flights) ==="
GROUP="TRI_shadow_seed0 S_LLM_seed0 TRI_equal_seed0 T0_baseline_seed0 S_PS_seed0"
for RUN in $GROUP; do mkdir -p $OUT/setting1_latest/$RUN; done
for AC in 8 6 4; do
  for RUN in $GROUP; do
    run $PY new_approach/evaluate_per_aircraft.py --model-path $EV/$RUN/weights/$(w1 $RUN) \
        --n-episodes 300 --num-agents $AC --num-weather-cells 2 --seed 42 \
        --metrics-path $OUT/setting1_latest/$RUN/evaluation_scal_ac$AC.json
  done
done
echo "=== S1_D1 DONE ==="
