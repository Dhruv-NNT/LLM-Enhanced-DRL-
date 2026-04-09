# Isolated LLM (Text-Based)

This folder isolates the **LLM-driven episode logic** from the main `LLM-DRL restructured code/` pipeline so you can iterate on prompts/signals safely.

- It does **not** modify anything inside `LLM-DRL restructured code/`.
- It runs an **LLM-only** episode in `StructuredEnv` and saves a GIF for visual inspection.
- It also contains an isolated guidance-audit script so you can debug one signal/turn choice at a time.

## Run

From the repository root:

```bash
python Isolated-LLM-Text-Based/run_llm_episode_gif.py
```

Outputs are written to:

- `Isolated-LLM-Text-Based/outputs/run_<UTC timestamp>/ep001.gif`

## Guidance Audit

Use this when you want to inspect the exact prompt-side geometry signals at one controller step and compare every turn bin over a short horizon.

```bash
python Isolated-LLM-Text-Based/audit_guidance_signals.py --seed 0 --start-llm-at-step 10 --horizon 6
```

The audit writes:

- `inspect.prompt.txt` with the exact observation/derived-signal text the LLM would see
- `turn_audit.json` with ranked turn-bin outcomes for that frozen scenario

## Notes

- Default Ollama model is `llama3:8b` (override with `--model`).
- GIF creation uses `imageio` (same dependency pattern as `LLM-DRL restructured code/evaluate.py`).
