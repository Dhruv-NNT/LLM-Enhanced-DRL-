# Pure-LLM Evaluation Harness

This folder contains a standalone runner to evaluate LLM-only guidance (no PPO). It saves one GIF per episode and logs simple safety metrics.

## Quick start

With memory context (uses `memory/_index.jsonl` and `json_answers/eval_memory.json`):
```bash
python llm_only_eval/run_llm_eval.py --mode with_memory --episodes 10
```

Without memory context (isolated run folder, no prior context):
```bash
python llm_only_eval/run_llm_eval.py --mode no_memory --episodes 10
```

## Analysis helper
To reproduce the step-by-step diagnosis summary:
```bash
python llm_only_eval/analyze_llm_runs.py \
  --no-run llm_only_eval/runs/no_memory_<timestamp> \
  --with-run llm_only_eval/runs/with_memory_<timestamp>
```

## Notes
- Make sure the Ollama server is running and the model is available.
- Outputs are written under `llm_only_eval/runs/<mode>_<timestamp>/`.
- Memory logs are written to:
  - `memory/run_<timestamp>/` for `with_memory`
  - `llm_only_eval/runs/<...>/memory/` for `no_memory`

## Optional flags
- `--model <name>`: override the Ollama model name.
- `--seed <int>`: base seed for paired runs.
- `--keep-frames`: keep PNG frames after GIF generation.
- `--enable-evaluator`: enable the evaluator LLM call (extra LLM cost).
