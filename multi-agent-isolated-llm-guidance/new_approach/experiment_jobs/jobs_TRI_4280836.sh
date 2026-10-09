#!/bin/bash
set -u
cd "/home/aradhya.dhruv/rl/behaviour_cloning/LLM enhanced DRL/LLM-DRL restructured code/multi-agent-isolated-llm-guidance" || exit 1

# CPU ONLY. TRI_shadow training (pid 66093) is still on the GPU and must not be disturbed.
export CUDA_VISIBLE_DEVICES=""
PY=/home/aradhya.dhruv/anaconda3/envs/llmdrl/bin/python
EV=Evaluation_distillation_approach
RUN=TRI_shadow_seed0
W=checkpoint_4280836.pt                               # frozen snapshot at 4,280,836 steps
OUT=new_approach/eval_results/staging_TRI_4280836     # staging: nothing existing is overwritten

run() { echo "[$(date +%H:%M:%S)] $*"; [ "${DRY:-0}" = 1 ] && return 0; "$@"; }

echo "=== TRI_shadow @ 4,280,836 steps | weights=$W ==="
[ -f "$EV/$RUN/weights/$W" ] || { echo "FATAL: weights not found"; exit 1; }
mkdir -p $OUT/$RUN

# --- 1-3: the three 300-scenario paired blocks (same episode seeds as every other run) ---
for SEED in 42 1042 2042; do
  if [ "$SEED" = 42 ]; then OUTFILE=evaluation.json; else OUTFILE=evaluation_seed$SEED.json; fi
  run $PY evaluate_rl.py --model-path $EV/$RUN/weights/$W \
      --n-episodes 300 --num-agents 4 --num-weather-cells 2 --seed $SEED \
      --metrics-path $OUT/$RUN/$OUTFILE
done

# --- 4-6: per-aircraft scalability for the AAMAS benchmark table ---
for AC in 8 6 4; do
  run $PY new_approach/evaluate_per_aircraft.py --model-path $EV/$RUN/weights/$W \
      --n-episodes 300 --num-agents $AC --num-weather-cells 2 --seed 42 \
      --metrics-path $OUT/$RUN/evaluation_scal_ac$AC.json
done

echo "=== DONE at $(date +%H:%M:%S) — results in $OUT/$RUN ==="
echo "Session kept open as requested. Type 'exit' to close it."
