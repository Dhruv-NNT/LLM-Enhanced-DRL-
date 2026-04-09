# Four-Phase LLM Guidance Sandbox

This sandbox runs a 4-phase LLM controller (WAIT_TURN → EXECUTE_TURN → MERGE_BACK → JOINED) without touching the original code.

## Run (LLM-only)

No-memory:
```bash
python phase5_sandbox/run_llm_eval_phase5.py --mode no_memory --episodes 10
```

With-memory:
```bash
python phase5_sandbox/run_llm_eval_phase5.py --mode with_memory --episodes 10
```

## Analyze
```bash
python phase5_sandbox/analyze_llm_runs.py \
  --no-run phase5_sandbox/runs/no_memory_<timestamp> \
  --with-run phase5_sandbox/runs/with_memory_<timestamp>
```
