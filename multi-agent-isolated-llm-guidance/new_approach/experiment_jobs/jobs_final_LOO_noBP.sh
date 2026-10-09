#!/bin/bash
set -u
cd "/home/aradhya.dhruv/rl/behaviour_cloning/LLM enhanced DRL/LLM-DRL restructured code/multi-agent-isolated-llm-guidance" || exit 1

# CPU ONLY. TRI_shadow_seed0 training (pid 66093) is still on the GPU and must not be disturbed.
export CUDA_VISIBLE_DEVICES=""
PY=/home/aradhya.dhruv/anaconda3/envs/llmdrl/bin/python
EV=Evaluation_distillation_approach
OUT=new_approach/eval_results/setting1_latest

RUN=LOO_noBP_seed0
W=last_checkpoint.pt          # 5,000,094 steps (training FINISHED)

run() { echo "[$(date +%H:%M:%S)] $*"; [ "${DRY:-0}" = 1 ] && return 0; "$@"; }

echo "=== FINAL Setting-1 evaluation | $RUN | weights=$W ==="
[ -f "$EV/$RUN/weights/$W" ] || { echo "FATAL: weights not found"; exit 1; }
mkdir -p $OUT/$RUN

# --- 1-3: the three 300-scenario paired blocks that feed every significance test ---
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

echo "=== DONE $RUN at $(date +%H:%M:%S) ==="
