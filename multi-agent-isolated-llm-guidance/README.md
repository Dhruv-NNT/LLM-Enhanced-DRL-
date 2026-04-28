# Multi-Agent Isolated LLM Guidance

This folder is the pure-LLM guidance sandbox for the multi-aircraft sector simulation.

The current default evaluation path uses a **global LangGraph guidance controller**:

- The simulator owns the world state, aircraft motion, weather motion, hazard prediction, rewards, and rendering.
- At each simulator step, the controller finds the aircraft that have entered the sector and are due for guidance.
- If at least one entered aircraft needs guidance, the controller builds one global prompt for all guidance-eligible aircraft in that step.
- The LLM must return one JSON action for every guidance-eligible aircraft.
- The controller parses, retries, guardrails, and fills missing actions before applying the final maneuver dictionary to the simulator.

The folder name still says "isolated" because this project started from a per-aircraft isolated prompting design. The older local controller still exists as `MultiAgentThreeCallController`, and some tests cover it, but `evaluate.py` now instantiates `GlobalLangGraphGuidanceController`.

## Current Control Path

The active path is:

1. `evaluate.py` creates `GlobalLangGraphGuidanceController`.
2. `evaluate.py` creates `JointGuidanceEnv`.
3. The environment resets the multi-agent simulator.
4. Each loop calls `controller.choose_llm_actions(...)`.
5. The controller's LangGraph pipeline decides whether an LLM call is needed.
6. If needed, one prompt is sent to Ollama.
7. The LLM response is parsed into per-agent actions.
8. Invalid, missing, or unsafe actions are corrected or replaced.
9. The environment applies the final `{agent_id: heading_change_deg}` action map.
10. Frames, GIFs, prompt traces, raw responses, and normalized JSON logs are saved.

The active controller is implemented in `rl_llm_multi/llm.py` as `GlobalLangGraphGuidanceController`.

## Important Files

- `configs.py`: environment constants, allowed action bins, hazard lookahead windows, recovery thresholds, Ollama settings, and output paths.
- `evaluate.py`: runs a pure-LLM episode using `GlobalLangGraphGuidanceController` and saves memory traces plus rendered frames.
- `run_llm_episode_gif_multi_agent.py`: command-line entrypoint for running an episode and writing a GIF.
- `rl_llm_multi/core.py`: multi-agent simulator. It moves aircraft and weather, predicts hazards, computes rewards, and renders frames.
- `rl_llm_multi/env.py`: environment wrappers. The pure-LLM rollout uses `JointGuidanceEnv`.
- `rl_llm_multi/llm.py`: prompt builders, Ollama invocation, JSON parsing, local legacy controller, and the current global LangGraph controller.
- `rl_llm_multi/utils.py`: route loading, geometry helpers, weather helpers, route assignment, and cross-track calculations.
- `tests/test_llm_controller.py`: unit tests for both the older local controller and the current global controller.

## One-Sentence Summary

At each step, the current controller asks the LLM for a joint JSON decision for all entered aircraft that are currently due for guidance, then applies controller-side validation and deterministic fallback before the simulator moves.

## What The LLM Actually Receives

The global prompt is built by `GlobalPromptBuilder.build_prompt()` in `rl_llm_multi/llm.py`.

It is plain text, not a chat-history template. In text mode it is sent as the single user message to Ollama. In vision mode, recent rendered frames are also attached as image inputs.

The prompt includes these sections:

- `ROLE` and `GLOBAL GUIDANCE TASK`
- `OUTPUT RULES`
- strict JSON schema
- optional `MERGE_BACK PRIORITY`
- `GUIDANCE-ELIGIBLE AGENTS THIS STEP`
- `ENTERED TRAFFIC CONTEXT`
- `GLOBAL WEATHER`
- `RANKED TRAFFIC THREATS`
- `PER-AGENT TURN PREVIEWS`
- `RECENT GLOBAL DECISION MEMORY`
- `VISION FRAME CONTEXT`

The most important section is `PER-AGENT TURN PREVIEWS`. The LLM is not asked to invent maneuver consequences from scratch. The controller simulates candidate turns first, ranks them, and gives the model structured preview rows such as:

```text
turn=+30 | safe=1 | pair_loss=None | weather_entry=None | boundary_exit=None | min_sep=7.49 | min_weather_clearance=19.88 | progress_to_destination=20.00 | cross_track_reduction=-0.01 | end_distance=37.16 | end_cross_track=1.01 | end_heading_error=1.6
```

The LLM chooses among those controller-generated options using the prompt rules.

## Required LLM Output

The current global prompt asks for exactly one JSON object:

```json
{
  "answer": {
    "scenario_summary": "<brief>",
    "actions": {
      "A1": {
        "call_name": "EXECUTE_TURN",
        "rationale": "<brief>",
        "maneuver": { "heading_change_deg": 0 }
      }
    }
  }
}
```

Rules enforced by the controller:

- `actions` must include every guidance-eligible agent for the current step.
- Actions for aircraft that have not entered the sector are ignored.
- `heading_change_deg` must snap to one of `ACTION_BINS`.
- Positive turn values mean left, negative values mean right, and `0` means hold current heading.

## LangGraph Pipeline

`GlobalLangGraphGuidanceController` compiles a LangGraph with these nodes:

1. `collect_state`: finds entered aircraft, gathers sector-entry annotations, and optionally collects recent frame paths for vision mode.
2. `assign_stages`: decides which entered aircraft need guidance and labels each one with a call type.
3. `build_global_context`: builds threat rows, candidate preview rows, recent decision memory, and the final global prompt.
4. `call_llm`: sends the prompt to Ollama when at least one aircraft is actionable.
5. `parse_validate_retry`: extracts JSON, checks whether every expected agent has an action, and retries when output is invalid or incomplete.
6. `guardrail_actions`: snaps requested turns to allowed bins and applies stage-specific maneuver constraints.
7. `fallback_missing`: uses deterministic fallback for missing or invalid per-agent actions.
8. `commit_and_log`: updates controller state, logs prompt/response/normalized artifacts, records recent decisions, and returns the final action map.

## Which Aircraft Are Eligible For Guidance

The global controller only considers aircraft that are:

- launched,
- not finished,
- already inside or past sector entry,
- present in `core.active_agent_ids`.

Aircraft that have not entered the sector are excluded from the prompt and from the returned action map. This is intentional: current guidance is mainly about behavior after sector entry.

## Call Types

The controller labels every guidance-eligible aircraft with one of three per-agent call types.

### `EXECUTE_TURN`

Used for the first guidance call after an aircraft enters the sector.

In the current global controller, the first call is triggered by sector entry, not only by a predicted hazard. This means every entered aircraft gets one initial chance to choose an active maneuver.

### `EMERGENCY_MANEUVER`

Used after an aircraft has already received its first guidance call and an immediate hazard is predicted.

Immediate hazards are based on the short emergency lookahead window:

- predicted pairwise loss of separation within `EMERGENCY_LOOKAHEAD_STEPS`,
- or predicted weather entry within `EMERGENCY_LOOKAHEAD_STEPS`.

The guardrail layer does not allow an emergency action to remain at `0`; if the LLM requests `0`, the controller resolves it to a nonzero permitted turn.

### `MERGE_BACK`

Used after initial avoidance when the aircraft should recover toward the destination or route corridor.

Common triggers include:

- boundary warning,
- enough safe streak with excessive cross-track error,
- enough safe streak with excessive destination heading error,
- repeated merge-back recheck after a prior merge-back maneuver.

For merge-back calls, the prompt adds `MERGE_BACK PRIORITY`, and the guardrail layer restricts the action toward the destination side when necessary.

## Candidate Preview Rows

Before asking the LLM, the controller simulates allowed heading-change bins for each actionable aircraft.

Each preview row reports:

- candidate `heading_change_deg`,
- whether the candidate is safe over the preview horizon,
- first predicted pair-loss step,
- first predicted weather-entry step,
- first predicted boundary-exit step,
- minimum separation to any aircraft,
- minimum weather clearance,
- progress toward destination,
- cross-track reduction,
- end distance to destination,
- end cross-track error,
- end heading error to destination.

Preview rows are sorted by a controller-side preference function before they reach the model. This means the LLM sees a curated decision table, not raw world state alone.

## Threat Rows

For each guidance-eligible aircraft, the controller ranks nearby traffic threats and includes the top threats in the prompt.

Threat ranking prefers:

- aircraft with predicted loss of separation,
- then smaller predicted minimum separation,
- then smaller current distance.

The prompt does not dump all pairwise geometry in full detail. It gives the model a compact threat table that is most relevant for the current decision.

## Hazard Prediction

Hazards are computed by the simulator before the LLM is called.

`MultiAgentSectorCore._update_hazard_predictions()` clones the simulator and rolls it forward for `HAZARD_LOOKAHEAD_STEPS`. For each aircraft, it records:

- predicted minimum separation,
- predicted pair-loss step,
- predicted weather clearance,
- predicted weather-entry step,
- predicted boundary-exit step.

These predictions appear in the prompt and also influence call timing.

## Guardrails And Fallback

The LLM is advisory. The controller is the final authority.

After parsing the model response, the controller:

- ignores actions for agents that were not expected,
- snaps each requested turn to the nearest allowed action bin,
- applies stage-specific constraints,
- retries if the JSON is invalid or missing expected agents,
- falls back per missing agent after retries are exhausted,
- logs both requested and applied heading changes.

Examples from the tests:

- If the global LLM response omits `A2`, the controller retries. If `A2` is still missing, `A2` gets a deterministic fallback while valid LLM actions for other agents can still be used.
- If an emergency response asks for `0`, the controller changes it to a nonzero maneuver.
- If a merge-back response asks for a turn away from the destination side, the controller can resolve it to `0` or another allowed turn.

## Vision Mode

Text mode is the default.

When `--use-vision` is enabled, the controller attaches the current frame plus up to the previous three frames to the Ollama request:

```text
image_003.png, image_002.png, image_001.png, image_000.png
```

The prompt still includes text references in `VISION FRAME CONTEXT`; the actual images are sent through the Ollama message payload.

## Memory And Logs

Every global guidance call writes files under a timestamped folder in `multi-agent-isolated-llm-guidance/memory/`.

Current global call artifacts use this naming pattern:

- `tXXX_global_guidance.prompt.txt`
- `tXXX_global_guidance.response.txt`
- `tXXX_global_guidance.normalized.json`
- `_index.jsonl`

The normalized JSON is usually the best debugging artifact. It contains:

- final normalized answer,
- per-agent requested heading changes,
- per-agent applied heading changes,
- whether each action came from `llm` or `fallback`,
- threat rows by agent,
- preview rows by agent,
- debug metadata such as parse status, retry count, fallback agents, used vision, frame paths, eligible agent ids, entered agent ids, stage by agent, and model name.

The `_index.jsonl` file gives a compact one-line record per guidance call. For global calls, `agent_id` is `GLOBAL` and `call_name` is `GLOBAL_GUIDANCE`.

## Recent Decision Memory

The global controller keeps the last four committed global decisions in `recent_decisions`.

Those decisions are inserted into the next prompt under `RECENT GLOBAL DECISION MEMORY`, for example:

```text
- step=5: A1:EXECUTE_TURN +25deg; A3:EXECUTE_TURN -30deg
```

This gives the LLM short-term continuity without turning the prompt into an unbounded conversation history.

## Outputs

Runtime outputs are kept inside this folder:

- rendered frames and GIFs go to `outputs/`,
- evaluation summaries go to `evaluation/`,
- prompts, raw responses, normalized traces, and indexes go to `memory/`.

## Running An Episode

From the repository root:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py
```

Run with explicit routes:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --num-agents 2 --route-ids PATH2 PATH5
```

Run a short debugging episode:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --num-agents 2 --episode-step-cap 8
```

Run with vision frames attached to the LLM request:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --num-agents 2 --use-vision
```

The default Ollama settings are in `configs.py`:

```python
OLLAMA_MODEL = "gemma3:12b"
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_SECONDS = 120
```

## Running Tests

```bash
python3 -m unittest discover multi-agent-isolated-llm-guidance/tests
```

The tests patch `ollama_invoke`, so they do not require a live Ollama server.

## Legacy Local Controller

`MultiAgentThreeCallController` still exists in `rl_llm_multi/llm.py`.

That older controller:

- builds one local prompt per actionable aircraft,
- processes actionable aircraft sequentially,
- lets later aircraft in the same step see earlier committed actions,
- writes `tXXX_<agent>_<call>.prompt.txt` style local traces.

This is no longer the controller used by `evaluate.py`, but it remains useful for comparison and regression tests.

## Practical Mental Model

The current system is not "the LLM flies every aircraft directly."

It is:

- simulator-owned dynamics,
- simulator-owned hazard prediction,
- controller-owned call timing,
- controller-owned candidate previews,
- LLM-selected global JSON actions for due entered aircraft,
- controller-owned validation, correction, fallback, logging, and state updates.

When changing guidance behavior, the strongest levers are usually:

- call timing in `_global_call_decision_for_agent()`,
- prompt structure in `GlobalPromptBuilder`,
- candidate ranking in the preview sorting logic,
- guardrail behavior in `_resolve_turn_for_call()`,
- fallback behavior in `_fallback_answer()`,
- hazard prediction in `core.py`.
