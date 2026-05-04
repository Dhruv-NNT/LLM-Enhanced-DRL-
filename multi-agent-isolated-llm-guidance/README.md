# Multi-Agent Isolated LLM Guidance

This folder contains the pure-LLM guidance sandbox for the multi-aircraft sector simulation.
The current evaluation path uses a global LangGraph controller: one prompt is built for all entered aircraft that need guidance at the current simulator step, and the controller converts the LLM response into a validated `{agent_id: heading_change_deg}` action map.

The folder is still named "isolated" because the project started with per-aircraft isolated prompting. That older controller still exists as `MultiAgentThreeCallController`, but `evaluate.py` and `run_llm_episode_gif_multi_agent.py` use `GlobalLangGraphGuidanceController`.

## Current Control Flow

1. `evaluate.py` creates `GlobalLangGraphGuidanceController` and `JointGuidanceEnv`.
2. `JointGuidanceEnv` owns `MultiAgentSectorCore`, which handles aircraft motion, weather motion, rewards, rendering, and hazard prediction.
3. At each simulator step, the controller checks which launched aircraft have entered the sector.
4. New sector entries may receive an advisory HLTP plan.
5. The controller assigns each guidance-eligible aircraft a call type such as `EXECUTE_TURN`, `EMERGENCY_MANEUVER`, or `MERGE_BACK`.
6. For each actionable aircraft, the controller simulates candidate heading changes and builds preview rows.
7. The global prompt asks the LLM to choose one maneuver for every guidance-eligible aircraft.
8. The controller parses the JSON, retries if needed, applies guardrails, fills missing actions with deterministic fallback, logs artifacts, and returns final actions to the simulator.

## Important Files

- `configs.py`: constants for agents, weather, action bins, lookahead windows, buffers, output paths, and Ollama settings.
- `evaluate.py`: programmatic and CLI evaluation entrypoint.
- `run_llm_episode_gif_multi_agent.py`: convenience CLI for running one episode and writing a GIF.
- `rl_llm_multi/core.py`: simulator core for aircraft, launch timing, weather, hazard prediction, rewards, and rendering.
- `rl_llm_multi/env.py`: environment wrappers, including `JointGuidanceEnv` for LLM rollouts.
- `rl_llm_multi/llm.py`: prompt builders, Ollama calls, JSON parsing, HLTP logic, local legacy controller, and global LangGraph controller.
- `rl_llm_multi/utils.py`: route loading, waypoint geometry, weather geometry, unit conversion, and route assignment.
- `tests/test_llm_controller.py`: regression tests for controller behavior, weather, route launch spacing, and prompt/response handling.

## Data Inputs

The code expects shared data files outside this folder, as configured in `configs.py`:

- `22feb_paths.csv`
- `path_Waypoints22feb.csv`
- `test.geojson`

`utils.py` loads these files to build forward and reversed routes, waypoint positions, and the sector polygon.

## Running An Episode

From the repository root:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py
```

Useful options:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --num-agents 8 --num-weather-cells 2
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --num-agents 2 --route-ids PATH5_REV PATH6
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --episode-step-cap 10
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --use-vision
```

The script prints the number of steps, reward, success/truncation/failure status, whether vision was used, number of weather cells, and the GIF path.

Default Ollama settings are in `configs.py`:

```python
OLLAMA_MODEL = "gemma3:12b"
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_SECONDS = 120
```

## Guidance Eligibility

The global guidance prompt only asks for actions for aircraft that are:

- launched,
- not finished,
- already entered into the sector.

Aircraft that have not entered the sector are not controlled by the LLM yet. Delayed aircraft can still appear as future traffic inside controller previews once they launch during the lookahead.

## Launch Spacing

Route assignment now staggers aircraft that start too close to each other.

- `SAFE_R = 5.0` remains the simulator collision threshold.
- `TRAFFIC_CAUTION_R = 6.5` is the internal caution buffer.
- `LAUNCH_SEPARATION_R = TRAFFIC_CAUTION_R` is used to delay same-origin or nearby-origin launches.

There are two protections:

- `assign_routes()` computes a scheduled `start_step` that avoids launching aircraft too close to earlier scheduled traffic.
- `MultiAgentSectorCore._launch_agents()` checks again at runtime and postpones a launch if the start point is still too close to an active aircraft.

This prevents immediate collisions when many agents share an origin such as `OMBAP`.

## Weather Model

The simulator supports one or two weather cells.

Each weather cell has:

- an elliptical shape,
- a current center,
- a motion heading and speed,
- a major radius in nautical miles,
- a random growth or shrink rate in nautical miles per simulator step.

Weather speed is sampled from `0.5` to `1.1666666667` NM per step. Weather growth/shrink rate is sampled from `0.15` to `0.60` NM per step, with sign chosen randomly. If two weather cells exist, they share the same motion heading but can have different speed and growth/shrink rates.

Weather rings are interpreted as:

- green: caution/buffer region,
- yellow/red/magenta: terminal no-go region.

The prompt tells the LLM to avoid the green ring when possible and never plan through yellow/red/magenta.

## Prompt Shape

The global prompt is built by `GlobalPromptBuilder.build_prompt()` in `rl_llm_multi/llm.py`.

Main sections include:

- global task and output rules,
- safety rules and traffic hard veto,
- optional merge-back priority rules,
- guidance-eligible aircraft,
- entered traffic context,
- advisory HLTP plans,
- global weather,
- ranked traffic threats,
- per-agent turn previews,
- recent global decision memory,
- optional vision frame context.

The most important section is `PER-AGENT TURN PREVIEWS`. The controller simulates candidate turns first, then gives the LLM rows such as:

```text
turn=+10 | safe=1 | pair_loss=None | weather_entry=None | boundary_exit=None | min_sep=8.20 | traffic_buffer_ok=1 | min_weather_clearance=11.50 | weather_buffer_ok=1 | progress_to_destination=3.40 | cross_track_reduction=1.20 | end_distance=25.10 | end_cross_track=2.00 | end_heading_error=6.5
```

The LLM should choose from the displayed turns. It is not expected to invent new maneuver consequences.

## Required LLM Output

The global prompt asks for exactly one JSON object and no extra text:

```json
{
  "answer": {
    "scenario_summary": "<brief>",
    "actions": {
      "A1": {
        "rationale": "<brief>",
        "maneuver": { "heading_change_deg": 0 }
      }
    }
  }
}
```

Important details:

- The LLM should output only the guidance-eligible aircraft IDs.
- The LLM should not output `call_name`; the controller already assigned each phase.
- `heading_change_deg` must be one of the configured `ACTION_BINS`.
- Positive means left, negative means right, and `0` means hold current heading.

The normalized logs may include extra controller-owned fields such as `call_name`, `source`, and `requested_heading_change_deg`, but those are added after parsing.

## LangGraph Pipeline

`GlobalLangGraphGuidanceController` builds a LangGraph with these nodes:

1. `collect_state`
2. `hltp_on_sector_entry`
3. `assign_stages`
4. `build_global_context`
5. `call_llm`
6. `parse_validate_retry`
7. `guardrail_actions`
8. `fallback_missing`
9. `commit_and_log`

If no aircraft is actionable, the graph returns an empty action map for that step.

## Guardrails And Fallback

The LLM is advisory. The controller is the final authority.

After parsing the response, the controller:

- ignores unexpected aircraft IDs,
- checks that every expected aircraft has an action,
- retries invalid or incomplete JSON up to the retry limit,
- snaps requested turns to allowed bins,
- enforces call-specific constraints,
- uses deterministic fallback for missing or invalid actions,
- logs requested and applied turns.

Examples:

- Emergency maneuvers cannot stay at `0`; the controller resolves them to a nonzero turn.
- Merge-back maneuvers are constrained toward destination recovery when appropriate.
- Rows with imminent pair loss are avoided when any no-pair-loss row exists.

## HLTP Plans

HLTP is an advisory high-level tactical plan generated when aircraft enter the sector.

It can suggest:

- a first waypoint for deviation,
- a later route waypoint for merge-back,
- short rationale and conflict context.

Low-level candidate previews, traffic separation, weather clearance, and guardrails can override HLTP guidance.

## Vision Mode

Text mode is the default.

When `--use-vision` is enabled, the controller attaches the current frame and recent previous frames to the Ollama request when available. The text prompt still contains the structured state and preview rows, so vision is extra context rather than the only evidence.

## Outputs And Logs

Runtime outputs stay inside `multi-agent-isolated-llm-guidance/`:

- `outputs/`: rendered frames and GIFs,
- `evaluation/`: evaluation summaries,
- `memory/`: prompt, response, normalized JSON, and index logs.

Global guidance logs use names like:

- `t001_global_guidance.prompt.txt`
- `t001_global_guidance.response.txt`
- `t001_global_guidance.normalized.json`
- `_index.jsonl`

The normalized JSON is usually the best debugging file. It contains final actions, threat rows, preview rows, HLTP plans, weather, eligible agent IDs, entered agent IDs, fallback metadata, parse status, retry count, model name, and frame paths when vision is used.

## Running Tests

```bash
python3 -m unittest discover multi-agent-isolated-llm-guidance/tests
```

Many tests patch `ollama_invoke`, so they do not require a live Ollama server. The global controller tests do require `langgraph` to be installed in the active Python environment.

## Practical Mental Model

The LLM does not directly fly every aircraft at every step.

The actual division of responsibility is:

- simulator owns motion, weather, reward, rendering, and hazard prediction,
- controller owns call timing, candidate previews, guardrails, fallback, and logs,
- LLM selects maneuvers from controller-generated preview rows for due entered aircraft.

When changing behavior, the main places to inspect are:

- call timing in `_global_call_decision_for_agent()`,
- prompt text in `GlobalPromptBuilder`,
- preview generation and sorting in `_preview_rows()`,
- turn correction in `_resolve_turn_for_call()`,
- fallback in `_fallback_answer()`,
- launch spacing in `assign_routes()` and `_launch_agents()`,
- hazard prediction in `MultiAgentSectorCore._update_hazard_predictions()`.
