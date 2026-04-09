# Path-First LLM Guidance Sandbox

LLM is called every step; top goal (after safety) is to stay on the ATCO path toward destination. Safety overrides still apply (can deviate if separation is low).

Run:
```bash
python path_sandbox/run_llm_eval_path.py --mode no_memory --episodes 10
python path_sandbox/run_llm_eval_path.py --mode with_memory --episodes 10
```

Analyze:
```bash
python path_sandbox/analyze_llm_runs.py \
  --no-run path_sandbox/runs/no_memory_<ts> \
  --with-run path_sandbox/runs/with_memory_<ts>
```
