# Four-Phase LLM Guidance Sandbox

This sandbox runs a 4-phase LLM controller (WAIT_TURN → EXECUTE_TURN → MERGE_BACK → JOINED) without touching the original code.

## Run (LLM-only)

No-memory:
```bash
python phase4_sandbox/run_llm_eval_phase4.py --mode no_memory --episodes 10
```

With-memory:
```bash
python phase4_sandbox/run_llm_eval_phase4.py --mode with_memory --episodes 10
```

## Analyze
```bash
python phase4_sandbox/analyze_llm_runs.py \
  --no-run phase4_sandbox/runs/no_memory_<timestamp> \
  --with-run phase4_sandbox/runs/with_memory_<timestamp>
```
