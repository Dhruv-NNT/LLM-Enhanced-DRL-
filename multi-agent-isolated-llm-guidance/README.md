# Multi-Agent Isolated LLM Guidance

This folder is the pure-LLM sandbox for the multi-aircraft guidance setup.

The main idea is:

- The simulator still contains multiple aircraft, shared airspace, shared weather, and shared collision risk.
- But the LLM is **not** asked to solve all aircraft at once in one big global answer.
- Instead, the controller makes **one local LLM call per aircraft that currently needs advice**.
- Each call is about **one ownship only**.
- The controller does this sequentially inside a single simulator step, so later aircraft can see what earlier aircraft in the same step have already committed to doing.

This is why the setup is best described as:

- **multi-agent** at the environment level,
- **centralized** at the controller level,
- **isolated/local** at the prompt level.

That distinction is the most important thing to understand before making improvements.

## What This Folder Contains

- `configs.py`
  Defines constants such as action bins, lookahead lengths, route recovery thresholds, model name, Ollama host, and output paths.
- `evaluate.py`
  Runs a pure-LLM episode from start to finish and saves memory traces and rendered frames.
- `run_llm_episode_gif_multi_agent.py`
  Small command-line entrypoint that runs one episode and builds a GIF.
- `rl_llm_multi/core.py`
  The actual simulator. It moves aircraft, moves weather, computes predicted hazards, calculates reward, and renders frames.
- `rl_llm_multi/env.py`
  Thin environment wrappers around the simulator. The pure-LLM path uses `JointGuidanceEnv`.
- `rl_llm_multi/llm.py`
  The most important file for prompt understanding. It decides when to call the model, builds the prompt, calls Ollama, parses the answer, falls back if needed, and converts the answer into actual turn commands.
- `rl_llm_multi/utils.py`
  Route loading, geometry helpers, weather shape helpers, route assignment logic, and cross-track calculations.
- `tests/test_llm_controller.py`
  Good place to read if you want to see the intended behavior in compact form.

## One-Sentence Summary Of The System

At every simulator step, the controller checks each active aircraft, decides whether that aircraft needs an LLM call, builds a local prompt containing only that aircraft's relevant situation plus a preview of candidate turns, asks the LLM for one JSON action, corrects or replaces the answer if needed, and then applies the chosen heading change to the simulator for the next step.

## Why This Is Called "Isolated" Guidance

The word "isolated" does **not** mean aircraft are independent in the simulator.

They are not independent.

- They still share the same weather ellipse.
- They still share the same sector boundary.
- They still share the same risk of pairwise conflict.
- One aircraft's move can change another aircraft's safety.

What is isolated is the **prompting surface**.

The LLM does not receive:

- a global list of all actions to output at once,
- a global joint JSON for every aircraft,
- a centralized multi-aircraft optimization instruction.

Instead, each LLM prompt says, in effect:

> "You are helping one aircraft only. Here is that aircraft's local situation. Here are the nearby threats. Here is a preview of candidate turns. Return exactly one JSON action for this ownship."

That is the isolation.

## End-To-End Flow

Here is the full flow in plain language.

### 1. An episode starts

`evaluate.py` creates:

- a `MultiAgentThreeCallController`,
- a `JointGuidanceEnv`,
- a new timestamped memory folder under `memory/`.

The environment is reset with:

- `num_agents`,
- `seed`,
- optional explicit `route_ids`.

### 2. Routes are assigned

In `rl_llm_multi/utils.py`, `assign_routes()` selects route definitions from the route catalog.

Important details:

- Each route comes from the shared route CSV data.
- The code also creates mirrored reverse routes using the suffix `"_REV"`.
- If two routes share the same prefix before entering the sector, the code staggers their `start_step` so they do not launch on top of each other.

This matters because the multi-agent traffic pattern is created before the LLM ever sees anything.

### 3. The simulator builds aircraft state

In `rl_llm_multi/core.py`, each aircraft gets an `AgentState` containing:

- current position,
- destination,
- current heading,
- whether it has launched,
- whether it has entered the sector,
- whether it has finished,
- predicted conflict and weather information,
- safe streak count,
- last turn,
- threat metadata.

### 4. Weather is initialized

The simulator creates one moving weather ellipse.

The ellipse is constrained so that:

- it stays inside the sector,
- it does not start on top of launched aircraft,
- it does not cover the destination goals.

The prompt later refers to this weather ellipse as forbidden airspace.

### 5. Hazard predictions are computed

Before the LLM is asked anything, the simulator forecasts future risk.

The key function is `_update_hazard_predictions()` in `core.py`.

It clones the simulator and rolls it forward for `HAZARD_LOOKAHEAD_STEPS` steps.

During that rollout it records, for each aircraft:

- predicted minimum separation from other aircraft,
- first predicted step where separation drops below `SAFE_R`,
- predicted minimum weather clearance,
- first predicted step where the aircraft enters weather,
- first predicted step where the aircraft exits the sector boundary.

These predicted values are exactly what later appear inside the prompt.

### 6. A rendered frame is saved

At the beginning of the episode and after every step, the environment renders a PNG frame.

If `use_vision=True`, the latest frame is also attached to the Ollama call as an image.

Important detail:

- The prompt text itself is still text-only.
- Vision is added by attaching the current frame as base64 image data in the Ollama request payload.

### 7. The controller decides who needs LLM advice

This happens in `MultiAgentThreeCallController.update()` inside `rl_llm_multi/llm.py`.

For each active aircraft, `_call_decision_for_agent()` decides whether to do nothing or to trigger one of three call types:

- `EXECUTE_TURN`
- `EMERGENCY_MANEUVER`
- `MERGE_BACK`

If no aircraft need advice at that step, the controller returns `{}` and the environment advances with no explicit turns.

### 8. Actionable aircraft are sorted

If multiple aircraft need a call in the same step, the controller sorts them.

The sorting priority is based on:

- earliest predicted bad event,
- then smaller predicted minimum separation,
- then aircraft id for deterministic tie breaking.

So the system handles the most urgent aircraft first.

### 9. For each actionable aircraft, the controller builds a local prompt

For the current aircraft only, the controller computes:

- `threat_rows`
- `preview_rows`
- `fixed_actions`

Then it builds one prompt.

### 10. The LLM returns one JSON action

The controller calls Ollama through `ollama_invoke()`.

If the call succeeds and valid JSON is found, the answer is parsed.

If the call fails, times out, or returns invalid JSON, the controller uses a deterministic fallback action.

### 11. The answer is normalized and corrected

The controller does not trust the raw LLM answer completely.

It always:

- extracts one JSON object from the raw text,
- snaps the requested turn to the nearest allowed action bin,
- further restricts the turn if the call type only allows a subset of bins.

Example:

- In one saved `MERGE_BACK` trace, the raw LLM answer requested `-10`.
- But that aircraft was only allowed to turn toward the destination side.
- So the controller corrected the applied action to `0`.

This correction step is important.

The LLM is advisory, not authoritative.

### 12. The chosen turn is committed for that aircraft

Once one aircraft's action is accepted, it becomes part of `planned_actions`.

Then the next actionable aircraft in the same simulator step is processed.

That next aircraft will see the earlier committed action inside:

- `EARLIER COMMITTED ACTIONS THIS STEP`

and its turn preview will already be simulated while assuming those earlier actions happen.

This is the heart of the isolated multi-agent design.

### 13. The environment steps forward

After the controller finishes the current step's sequential local decisions, the environment applies the final action dictionary to the simulator.

Then:

- aircraft move,
- weather moves,
- rewards update,
- predictions update again,
- a new frame is rendered,
- the next loop begins.

## The Three Call Types

The controller only asks the LLM for help in three situations.

### 1. `EXECUTE_TURN`

Meaning:

- "This aircraft has encountered its first predicted hazard and needs its first avoidance turn."

Trigger logic:

- If any hazard is predicted and `execute_called` is still false for that aircraft.

Hazards include:

- traffic conflict risk,
- weather risk.

Prompt bias:

- focus on first avoidance,
- local ownship view only.

Allowed turns:

- all bins from `ACTION_BINS`.

State update after the call:

- `execute_called=True`
- internal controller stage becomes `WAIT_CLEAR`

### 2. `EMERGENCY_MANEUVER`

Meaning:

- "A bad event is predicted very soon, so this aircraft needs an urgent non-zero turn."

Trigger logic:

- If predicted pair loss or predicted weather entry happens within `EMERGENCY_LOOKAHEAD_STEPS`.

Prompt bias:

- immediate safety,
- prefer non-zero turn.

Allowed turns:

- all bins except `0`.

This is stricter than `EXECUTE_TURN`.

The controller is explicitly telling the model:

- do not just hold course,
- do something.

Note:

- The saved memory traces currently present in this folder did not include an `EMERGENCY_MANEUVER` example, so the description here comes from the code path rather than a captured runtime sample.

### 3. `MERGE_BACK`

Meaning:

- "The aircraft has already deviated for safety and should now recover toward the route or destination when safe."

Trigger logic:

- boundary warning after an executed turn,
- or safe streak long enough plus route/destination misalignment,
- or repeated merge-back rechecks.

Prompt bias:

- recover toward destination side,
- stay safe while recovering.

Allowed turns:

- only the direction that points back toward the destination,
- or just `0` if already almost aligned.

This is very important.

`MERGE_BACK` is not a free-choice turn stage.

The controller restricts the allowed action set:

- if destination heading error is positive, allowed turns are only nonnegative bins,
- if destination heading error is negative, allowed turns are only nonpositive bins,
- if the aircraft is already almost aligned, only `0` is allowed.

That is why a raw LLM answer can be changed after parsing.

## Internal Controller Stages

Besides call names, each aircraft has an internal controller state:

- `FOLLOW_ROUTE`
- `WAIT_CLEAR`
- `MERGE_BACK_RECHECK`

These stages are not directly sent to the model, but they decide when the model gets called again.

### `FOLLOW_ROUTE`

Default state before the aircraft has needed intervention.

### `WAIT_CLEAR`

Used after an execution or emergency turn.

The controller waits for the aircraft to become safe enough before merge-back logic starts.

### `MERGE_BACK_RECHECK`

Used after a merge-back advice is applied.

The controller waits a small number of steps, then checks again whether another merge-back call is needed.

This prevents the system from asking for merge-back every single step.

## What Exactly Goes Into The Prompt

The prompt is built by `MultiAgentPromptBuilder.build_prompt()` in `rl_llm_multi/llm.py`.

It concatenates these sections:

1. instructions text
2. ownship text
3. earlier committed actions
4. local weather risk
5. primary threat
6. other threats
7. local turn preview

That means the prompt is structured, hand-built text, not a chat history and not a chain-of-thought template.

## The Exact Prompt Shape

The saved traces show that a real prompt looks like this:

```text
ROLE: You are a cautious ATCO helper for one aircraft.
STAGE: EXECUTE_TURN. This aircraft needs its first avoidance turn.
- Focus on this ownship only.
- Use the local threat and preview information below.
- SAFE_R = 5.0 units.
- The weather ellipse is forbidden airspace.
- Return exactly one action for the ownship only.
OUTPUT RULES:
- Return EXACTLY ONE JSON object.
- heading_change_deg must be one of {-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30}.
- Positive is left, negative is right, 0 means hold current heading.
STRICT JSON SHAPE:
{
  "answer": {
    "scenario_summary": "<brief>",
    "technical_summary": "<brief>",
    "rationale": "<brief>",
    "maneuver": { "heading_change_deg": 0 }
  }
}
```

Then the prompt continues with situation-specific sections such as:

```text
OWNSHIP A3:
- call_name=EXECUTE_TURN
- position=(46.8,72.8) heading_deg=-47.7
- destination=(82.0,34.2) distance_to_destination=52.21 heading_error_to_destination_deg=+0.0
- route_recovery_error: cross_track=+0.00 boundary_distance=0.48
- predicted_pair_loss_step=8 predicted_weather_entry_step=None predicted_boundary_exit_step=None
- predicted_weather_clearance=11.92 safe_streak=0 boundary_warning_active=1
```

Then:

```text
EARLIER COMMITTED ACTIONS THIS STEP:
- none
```

Then weather, threats, and the preview table.

## Prompt Section By Section

### 1. Instructions section

This section fixes the model's job very tightly.

Important constraints:

- the model is helping one aircraft only,
- output must be exactly one JSON object,
- the action must be a heading change,
- the heading change must come from fixed discrete bins,
- sign convention is defined explicitly.

This is the main anti-drift control in the prompt.

### 2. Ownship section

This gives the LLM the aircraft's current state.

Fields:

- `position`
  Current simulator position in rescaled grid units, not raw latitude/longitude.
- `heading_deg`
  Current aircraft heading in degrees.
- `destination`
  Final target point of the route.
- `distance_to_destination`
  Straight-line distance to the destination.
- `heading_error_to_destination_deg`
  How far the aircraft would need to rotate to point directly at the destination.
- `cross_track`
  Signed route offset. This says how far the aircraft is from its route centerline.
- `boundary_distance`
  Distance from the sector boundary if inside the sector.
- `predicted_pair_loss_step`
  First predicted future step where pair separation will go below `SAFE_R`.
- `predicted_weather_entry_step`
  First predicted future step where the aircraft enters weather.
- `predicted_boundary_exit_step`
  First predicted future step where it leaves the sector.
- `predicted_weather_clearance`
  Minimum forecast clearance from weather over the lookahead.
- `safe_streak`
  Number of recent prediction updates where the aircraft has not been predicted hazardous.
- `boundary_warning_active`
  Whether the controller already considers the aircraft close enough to the boundary to care.

### 3. Earlier committed actions section

This is the most important coordination mechanism.

If the controller has already chosen actions for earlier aircraft in the current simulator step, those are listed here.

Real example from a saved `EXECUTE_TURN` prompt for `A4`:

```text
EARLIER COMMITTED ACTIONS THIS STEP:
- A3: -20 deg
```

This means:

- `A3` was processed first,
- `A3` already committed to `-20`,
- `A4` is being prompted second,
- `A4` should reason while assuming that `A3` turn will happen.

This is how the system stays isolated per prompt without becoming blind to same-step coordination.

### 4. Local weather risk section

This section tells the model about the weather ellipse from the ownship's perspective.

Fields:

- `current_clearance`
  Signed distance to the weather boundary right now.
  Positive means outside.
  Negative means already inside forbidden weather.
- `predicted_weather_entry_step`
  First future step the ownship is forecast to enter weather.
- `predicted_min_clearance`
  Smallest forecast weather clearance over the prediction horizon.
- `weather_center`
  Ellipse center in simulator units.
- `orientation_deg`
  Ellipse rotation angle.

The model is not given the full weather polygon geometry as raw coordinates.

Instead, it gets a compact summary plus the visual frame if vision is enabled.

### 5. Primary threat section

The controller ranks nearby threats and picks the highest-priority one as the primary threat.

Threat ordering is based on:

- earliest predicted loss step,
- then smallest predicted minimum separation,
- then current distance,
- then intruder id for tie breaking.

The primary threat section includes:

- intruder id,
- intruder position,
- intruder heading,
- current distance,
- bearing error,
- predicted loss step,
- predicted minimum separation.

### 6. Other threats section

This lists additional ranked threats after the primary one.

The controller keeps at most four threat rows total.

So the prompt does not dump the full traffic state.

It gives the LLM the most relevant nearby traffic summary.

### 7. Local turn preview section

This is arguably the most important section in the whole prompt.

The controller does not ask the LLM to imagine every consequence from scratch.

Instead, it precomputes a preview for every allowed candidate turn and shows the top-ranked candidates.

For each candidate turn, the preview includes:

- the turn angle,
- whether the preview is considered safe,
- first pair-loss step,
- first weather-entry step,
- first boundary-exit step,
- minimum separation to any aircraft,
- minimum weather clearance,
- end heading error to destination.

The prompt label says:

```text
LOCAL TURN PREVIEW (already conditioned on earlier committed actions this step):
```

That wording is accurate.

The preview is generated by actually simulating the horizon while assuming:

- all earlier committed aircraft turns happen,
- plus this candidate turn for the current ownship,
- and no new turns after the first previewed step.

So the LLM is choosing from controller-precomputed consequences, not free-form imagination alone.

## How `preview_rows` Are Computed

This logic lives in `_preview_rows()` and `_simulate_joint_horizon()` inside `rl_llm_multi/llm.py`.

For every allowed turn:

1. the controller clones the current simulator,
2. injects `fixed_actions` from earlier aircraft plus the candidate ownship turn,
3. rolls the simulator forward for `TURN_PREVIEW_STEPS`,
4. records safety and recovery metrics,
5. stores the result as one preview row.

Each row is then sorted by `_candidate_sort_key()`.

The ranking prefers:

- safe rows first,
- later or nonexistent pair loss,
- later or nonexistent weather entry,
- later or nonexistent boundary exit,
- larger minimum separation,
- larger weather clearance,
- smaller destination heading error,
- smaller cross-track error,
- smaller absolute turn magnitude.

This ranking is very important.

It means the controller already has a built-in preference system before the LLM speaks.

## The LLM Is Not Choosing From Raw World State Alone

This is worth stating clearly.

The LLM is not doing all reasoning from scratch.

It is being given:

- a compact local state summary,
- a ranked local threat summary,
- a controller-generated candidate-turn consequence table.

So the model's real job is closer to:

- interpret,
- choose,
- justify,

rather than:

- fully simulate,
- fully optimize,
- fully predict.

That is a deliberate design choice.

## What The Model Is Expected To Return

The prompt asks for exactly one JSON object in this shape:

```json
{
  "answer": {
    "scenario_summary": "<brief>",
    "technical_summary": "<brief>",
    "rationale": "<brief>",
    "maneuver": { "heading_change_deg": 0 }
  }
}
```

Important notes:

- The model returns only one ownship action.
- It is not allowed to return a list of actions.
- It is not allowed to return free text outside the JSON object.
- The controller expects `heading_change_deg` to be the actionable field.

The summary fields are currently informational only.

The actual control output is:

- `answer.maneuver.heading_change_deg`

## Real Example Of A Successful LLM Reply

Saved `EXECUTE_TURN` response:

```json
{
  "answer": {
    "scenario_summary": "Executing first avoidance turn for A3",
    "technical_summary": "Predicted primary threat A4 is closest at 31.78 units, requiring a -20 degree heading change to maintain safe separation.",
    "rationale": "Aiming to create space with the intruder while maintaining a safe distance and avoiding weather risk.",
    "maneuver": {
      "heading_change_deg": -20
    }
  }
}
```

This is a clean, valid reply.

The controller then normalizes it into a richer saved record that also adds:

- `agent_id`
- `call_name`
- `threat_rows`
- `preview_rows`
- `debug`

## What Happens If The Model Returns Bad Output

The controller is defensive.

There are two big failure classes:

### 1. The model call fails

Examples:

- timeout,
- connection error,
- HTTP error,
- malformed Ollama response.

In that case the controller does **not** stop the episode.

It uses `_fallback_answer()`.

### 2. The model call succeeds but JSON extraction fails

The controller uses a regex to find one JSON object in the returned text.

If that fails:

- parse status becomes `invalid_json`,
- the controller again uses fallback.

## What The Fallback Does

The fallback does not guess randomly.

It calls `_deterministic_best_turn()`, which picks the best candidate using the same preview ranking logic.

So fallback behavior is:

- deterministic,
- local,
- preview-driven,
- usually sensible.

Real saved fallback example:

- `llm_status = "connection_error"`
- `llm_error = "timed out"`
- fallback rationale becomes `"deterministic fallback choice"`
- applied turn becomes the best preview turn

This is important for reliability.

The system can keep running even when the LLM is unavailable.

## Normalization And Safety Correction

After parsing, the controller still performs two correction passes.

### Pass 1: snap to global allowed bins

Any returned value is snapped to the nearest value in:

```text
(-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30)
```

So if the model returns `-17`, the controller would snap it to `-15` or `-20` depending on nearest distance.

### Pass 2: enforce call-specific constraints

Even after bin snapping, the controller checks whether that turn is allowed for the current call type.

Examples:

- `EMERGENCY_MANEUVER` cannot be `0`
- `MERGE_BACK` may only allow turns on the destination side

Real example:

Raw `MERGE_BACK` LLM response:

```json
{
  "answer": {
    "scenario_summary": "Merging back to destination",
    "technical_summary": "Recovering from merge maneuver",
    "rationale": "Biasing turn towards destination while ensuring safe distance and clearance",
    "maneuver": { "heading_change_deg": -10 }
  }
}
```

But the normalized saved output for that same call used:

```json
"maneuver": {
  "heading_change_deg": 0
}
```

Why?

Because for that aircraft, `MERGE_BACK` only allowed nonnegative turns, so `-10` was illegal and got snapped to the closest allowed value, which was `0`.

This is a very important design feature.

The controller, not the LLM, is the final gatekeeper.

## How Same-Step Multi-Agent Coordination Works

This is the key design idea of the whole folder.

Imagine two aircraft need action in the same simulator step.

The controller does **not** ask:

> "LLM, give me both actions together."

Instead it does this:

1. pick the highest-priority aircraft,
2. build a local prompt,
3. get that aircraft's action,
4. commit it to `planned_actions`,
5. move to the next actionable aircraft,
6. build that second aircraft's prompt while showing the first aircraft's committed action,
7. build preview rows that already assume the first aircraft's action.

So the prompts stay local, but the coordination is still real.

This is why the design is stronger than naive independent prompts.

The later aircraft are not blind.

They just do not receive a giant global instruction block.

## What The Controller Logs

For every local call, the controller saves three files:

- `tXXX_agent_call.prompt.txt`
- `tXXX_agent_call.response.txt`
- `tXXX_agent_call.normalized.json`

It also appends one row to `_index.jsonl`.

The normalized JSON is the most informative artifact because it contains:

- final normalized answer,
- threat rows,
- preview rows,
- debug metadata.

Debug fields include:

- `llm_status`
- `llm_error`
- `parse_status`
- `fallback_reason`
- `call_reason`
- `used_vision`
- `frame_path`
- `fixed_actions`
- `threat_ids`

If you want to understand a strange action, always start with the normalized trace file.

## What The Ollama Request Looks Like

The runtime call is made by `ollama_invoke()`.

It sends a chat payload roughly like this:

```json
{
  "model": "llama3:8b",
  "stream": false,
  "messages": [
    {
      "role": "user",
      "content": "<prompt text here>",
      "images": ["<base64 image>"]
    }
  ],
  "options": {
    "temperature": 0.4,
    "top_p": 0.95,
    "num_predict": 4096
  }
}
```

If vision is off, the `images` field is omitted.

Defaults come from `configs.py`.

Important defaults:

- model: `llama3:8b`
- temperature: `0.4`
- top-p: `0.95`
- max tokens: `4096`
- timeout: `10.0` seconds
- host: `http://127.0.0.1:11434`

## What Happens In The Simulator When Actions Are Applied

In `core.py`, aircraft behave differently depending on whether they have entered the sector.

Before sector entry:

- they follow the route automatically using `_route_step()`

After sector entry:

- they move in free-flight mode using `_free_step()`
- the action is applied as a heading change

This means LLM guidance is mainly about behavior after sector entry.

The controller also annotates frames with labels like:

- `A1: ENTERED SECTOR`
- `A1: EXECUTE TURN`
- `A4: MERGE BACK (BOUNDARY)`

These labels come from the controller state, not the LLM.

## Reward Exists, But The Pure-LLM Path Does Not Train On It

The simulator still computes reward.

Reward includes:

- progress toward destination,
- small step penalty,
- cross-track penalty,
- hazard penalties,
- weather-risk penalties,
- finish bonus,
- large failure penalties,
- success bonus.

But in this isolated folder, the pure-LLM episode path is not using reward to update a policy.

The reward is just part of evaluation and diagnostics.

That matters because if you change prompting behavior, you are not training through reward here.

You are changing inference-time decision making.

## Important Practical Observations About The Current Design

These are not criticisms. They are simply the current facts of the implementation.

### 1. The prompt is strongly scaffolded by controller-generated preview rows

The LLM is deciding with heavy structured assistance.

### 2. The final control space is discrete

The model cannot ask for arbitrary degrees.

### 3. The system is robust to model failure

Timeouts and invalid JSON fall back to deterministic controller logic.

### 4. Later aircraft in the same step are partially coordinated

They see earlier committed actions and previews conditioned on those actions.

### 5. `MERGE_BACK` is controller-constrained

The LLM can suggest something outside the allowed merge-back direction, but the controller can override it.

### 6. The visual input is optional

The prompt logic works without vision.

When vision is enabled, the image supplements the text rather than replacing it.

## Where To Edit If You Want To Improve The System

If you want to improve this setup, these are the highest-value edit points.

### If you want to change when the LLM gets called

Edit:

- `MultiAgentThreeCallController._call_decision_for_agent()` in `rl_llm_multi/llm.py`

This controls:

- when first-turn calls happen,
- when emergency calls happen,
- when merge-back calls happen,
- how safe streak and boundary logic affect call timing.

### If you want to change what the LLM sees

Edit:

- `MultiAgentPromptBuilder` in `rl_llm_multi/llm.py`

This controls:

- wording,
- section order,
- how much state is exposed,
- whether the prompt is more narrative or more tabular,
- whether additional fields are shown.

### If you want to change the candidate-turn preview logic

Edit:

- `_preview_rows()`
- `_candidate_sort_key()`
- `_simulate_joint_horizon()`

These functions largely define the quality of the "advice context" given to the model.

### If you want to change post-processing or trust level of the model

Edit:

- `_normalize_local_answer()`
- `_resolve_turn_for_call()`
- `_fallback_answer()`

These decide how much freedom the LLM really has.

### If you want to change hazard prediction itself

Edit:

- `_update_hazard_predictions()` in `rl_llm_multi/core.py`

That function determines the forecast values that show up throughout the prompt.

### If you want to change route recovery behavior

Edit:

- `ROUTE_RECOVERY_XTRACK_UNITS`
- `DESTINATION_ALIGNMENT_DEG`
- `MERGE_BACK_REPEAT_ERROR_DEG`
- `MERGE_BACK_RECHECK_STEPS`

in `configs.py`

These directly affect when and how merge-back is triggered.

### If you want to tune urgency

Edit:

- `HAZARD_LOOKAHEAD_STEPS`
- `EMERGENCY_LOOKAHEAD_STEPS`
- `TURN_PREVIEW_STEPS`
- `CLEAR_STREAK_REQUIRED`

These constants shape both call timing and preview quality.

## A Very Short Mental Model

If you want the fastest correct intuition, use this:

1. the simulator predicts hazards ahead of time
2. the controller decides which aircraft need help now
3. for each such aircraft, the controller simulates candidate turns
4. the controller gives the LLM a local summary plus those previews
5. the LLM picks one discrete turn in JSON
6. the controller validates or overrides that choice if necessary
7. the environment advances one step

That is the system.

## Suggested Reading Order In Code

If you want to modify the system confidently, read in this order:

1. `rl_llm_multi/llm.py`
2. `rl_llm_multi/core.py`
3. `tests/test_llm_controller.py`
4. `configs.py`
5. `rl_llm_multi/utils.py`
6. `evaluate.py`

This order gives the fastest understanding of both control flow and geometry.

## How To Run

From the repo root:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --num-agents 2 --route-ids PATH2 PATH5
```

For a short smoke run:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --num-agents 2 --route-ids PATH2 PATH5 --episode-step-cap 8
```

To enable image input to the model:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --num-agents 2 --use-vision
```

Outputs:

- frames and GIFs go to `multi-agent-isolated-llm-guidance/outputs/`
- prompt/response/normalized traces go to `multi-agent-isolated-llm-guidance/memory/`

## Final Takeaway

This system is not "LLM controls everything directly."

It is:

- simulator-driven,
- prediction-assisted,
- prompt-scaffolded,
- per-aircraft locally prompted,
- sequentially coordinated,
- controller-validated.

The LLM contributes the final choice among structured local options, but the simulator and controller define the world, define the hazards, define the candidate action consequences, and define the final safety constraints.

If you want to improve the system, the best question is usually not only:

> "How should I change the prompt?"

It is also:

> "How should I change the information architecture around the prompt, the preview ranking, the call timing, and the post-processing rules?"

That is where most of the real behavior comes from.
