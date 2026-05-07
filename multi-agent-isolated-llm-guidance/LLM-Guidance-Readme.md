# Integrated LLM Guidance README

This document explains how both text-based and vision-based LLM guidance work
in the `multi-agent-isolated-llm-guidance` folder.

Both modes use the same simulator, the same global controller, the same stage
logic, the same candidate-turn previews, the same fallback logic, and the same
reward/termination logic.

The main difference is the LLM input:

- text mode sends only structured text,
- vision mode sends structured text plus recent rendered frames.

The shared code path is:

```text
run_llm_episode_gif_multi_agent.py
  -> evaluate_episode()
  -> JointGuidanceEnv
  -> MultiAgentSectorCore
  -> GlobalLangGraphGuidanceController
  -> text prompt or vision prompt
  -> Ollama response
  -> validated heading changes
  -> simulator step
```

Text mode is used by default. Vision mode is enabled by passing `--use-vision`.

## 1. What The System Does

The simulation contains multiple aircraft moving through a sector.

Each aircraft has:

- a route,
- a start point,
- a destination,
- a current position,
- a current heading,
- a fixed speed,
- a launch time,
- a state such as not launched, active, entered sector, or finished.

The sector also contains one or two moving weather cells. Weather is drawn as
colored ellipses:

- green means caution or buffer,
- yellow means dangerous weather,
- red means more dangerous weather,
- magenta means extreme convective weather.

The LLM does not own the simulator. The simulator owns aircraft movement,
weather movement, reward calculation, rendering, collision checks, weather
checks, and episode termination.

The LLM is used as an air traffic control helper.

In text mode, it receives:

- text that names aircraft and destinations,
- text that lists predicted threats,
- text that lists simulator-tested candidate turns,
- text that lists recent controller memory.

In vision mode, it receives all of the above plus:

- rendered frames showing aircraft, routes, waypoints, weather, and recent
  motion.

In both modes, it chooses a heading change for each aircraft that needs guidance
in the current step.

The controller validates the LLM response before passing the final actions back
to the simulator.

## 2. Main Files

`configs.py`

Contains constants for routes, agents, weather, action bins, lookahead windows,
output paths, and Ollama settings.

`evaluate.py`

Runs a complete episode. It creates the environment, creates the global
controller, saves frames, passes frame paths to the controller, runs the step
loop, and returns an episode result.

`run_llm_episode_gif_multi_agent.py`

Command-line wrapper around `evaluate_episode()`. It runs one episode and writes
a GIF from the saved frames.

`rl_llm_multi/core.py`

The simulator core. It handles aircraft state, route following, free flight,
weather movement, hazard prediction, rewards, rendering, and termination.

`rl_llm_multi/env.py`

Environment wrappers. The vision LLM runner uses `JointGuidanceEnv`.

`rl_llm_multi/llm.py`

The LLM guidance logic. It builds text and vision prompts, attaches frame
images to Ollama requests, parses JSON, creates candidate turn previews, applies
turn constraints, creates fallback actions, logs prompt/response files, and runs
the LangGraph controller.

`rl_llm_multi/utils.py`

Route loading, waypoint loading, sector loading, weather geometry, unit
conversion, route assignment, and plotting helpers.

## 3. Important Constants

The important constants are in `configs.py`.

`SAFE_R = 5.0`

This is the actual collision threshold. If any pair of active aircraft becomes
closer than this distance, the episode terminates with collision failure.

`TRAFFIC_CAUTION_R = 6.5`

This is a larger caution buffer. The prompt and preview rows tell the LLM to
prefer actions that keep aircraft at or above this separation. The simulator
does not terminate at this value.

`GOAL_RADIUS = 5.0`

An aircraft is finished when it is within this distance of its destination.

`AGENT_SPEED = 2.0`

Each active aircraft moves 2 simulator units per step.

`MAX_STEP = 60`

If the episode has not succeeded or failed by this step count, it truncates.

`ACTION_BINS = (-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30)`

These are the only legal heading changes. Positive means turn left. Negative
means turn right. Zero means hold the current heading.

`HAZARD_LOOKAHEAD_STEPS = 12`

The simulator looks ahead this many steps to predict traffic, weather, and
boundary hazards.

`TURN_PREVIEW_STEPS = 10`

The controller simulates each candidate turn for this many steps before showing
the result to the LLM.

`PREVIEW_TOP_K = 6`

Only the best few candidate rows are displayed in the prompt for each aircraft.

`CLEAR_STREAK_REQUIRED = 2`

After an aircraft has been predicted safe for this many updates, it may become
eligible for merge-back guidance if it is off route or poorly aligned.

`MERGE_BACK_RECHECK_STEPS = 2`

After a merge-back instruction, the controller waits this many cycles before
asking for another merge-back, unless a more urgent hazard appears.

## 4. Route And Sector Preparation

The code reads shared data files configured in `configs.py`:

```text
22feb_paths.csv
path_Waypoints22feb.csv
test.geojson
```

Waypoint latitude and longitude values are rescaled into a simulator grid. The
grid is roughly `0` to `100` in both x and y.

The sector polygon is loaded from the GeoJSON file and rescaled into the same
grid.

For each route in `22feb_paths.csv`, the route catalog creates:

1. the forward route,
2. a reversed route with the `_REV` suffix.

Each route stores:

- route id,
- original route id,
- whether it is reversed,
- origin waypoint,
- destination waypoint,
- waypoint names,
- waypoint coordinates,
- route `LineString`,
- estimated sector entry distance,
- estimated sector entry step.

## 5. Aircraft Route Assignment

When the episode resets, `assign_routes()` selects routes for the requested
number of aircraft.

If no explicit route ids are given, routes are randomly sampled using the
episode seed.

If route ids are given, the simulator uses those route ids in order.

Aircraft ids are:

```text
A1, A2, A3, ...
```

The assignment also computes a start step for each aircraft.

The code tries to avoid launching aircraft too close together. It uses
`LAUNCH_SEPARATION_R`, which currently equals `TRAFFIC_CAUTION_R`.

There are two launch protections:

1. `assign_routes()` staggers scheduled starts when routes share an early route
   prefix.
2. `_launch_agents()` checks the launch position again at runtime. If another
   active aircraft is too close, launch is delayed by one step.

This helps avoid immediate collisions at shared or nearby origins.

## 6. What Happens When A Simulation Starts

The normal text command is:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py
```

The normal vision command is:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --use-vision
```

The only mode flag is `--use-vision`. Without it, the runner uses text mode.

Startup happens in this order:

1. `run_llm_episode_gif_multi_agent.py` parses command-line arguments.
2. It chooses an output directory under `outputs/` unless one is provided.
3. It calls `evaluate_episode()`.
4. `evaluate_episode()` creates a memory directory under `memory/`.
5. It creates a `GlobalLangGraphGuidanceController`.
6. It creates a `JointGuidanceEnv`.
7. `JointGuidanceEnv` creates a `MultiAgentSectorCore`.
8. The environment is reset.
9. The first frame is rendered as `image_000.png`.
10. The controller can use that frame path during guidance.
11. The main episode loop starts.

## 7. What Environment Reset Does

Reset happens in `MultiAgentSectorCore.reset()`.

At reset, the simulator:

1. stores the requested number of agents,
2. stores the requested number of weather cells,
3. creates a seeded random generator,
4. assigns routes to aircraft,
5. clears old aircraft state,
6. clears old weather state,
7. sets `n_step = 0`,
8. sets current reward and total reward to zero,
9. clears previous done, truncated, and success flags,
10. clears the previous failure reason,
11. creates one `AgentState` for each assigned aircraft,
12. creates the weather cells,
13. runs initial hazard prediction,
14. returns active observations and info.

Each aircraft starts at the first point of its route. Its initial heading follows
the route tangent. Its destination is the final point of the route.

An aircraft can be launched immediately if its start step is `0`, but it is not
marked as having entered the sector until `_update_agent_flags()` runs during a
simulator step.

## 8. Weather Initialization

The simulator creates one or two weather cells.

Each weather cell is an ellipse with:

- center,
- major radius,
- minor radius,
- orientation angle,
- motion heading,
- speed,
- growth or shrink rate.

Weather is randomly sampled, but it must be valid:

- the full weather ellipse must stay inside the sector,
- it must not cover any launch point,
- it must not block any destination goal area.

If a sampled weather cell fails those checks, another one is sampled.

If two weather cells are used, they share the same motion heading but can have
different positions, sizes, speeds, and growth rates.

## 9. The Main Episode Loop

After reset, `evaluate_episode()` loops until the episode is done or truncated.

The loop is:

```text
while not done and not truncated:
    latest_frame_path = image_<current step>.png
    actions = controller.choose_llm_actions(..., use_vision=<mode>)
    annotate current frame if there are labels
    env.step(actions)
    render next frame
```

Guidance is chosen before the next simulator step.
In text mode, the current frame path is mostly used for saved-frame annotation.
In vision mode, it also tells the controller which rendered image can be
attached to the LLM call.

Example:

1. At `n_step = 0`, the controller sees the reset state.
2. If no aircraft has entered the sector yet, there is no LLM action.
3. The environment steps to `n_step = 1`.
4. During that step, an aircraft may enter the sector.
5. The next frame is rendered.
6. On the next loop, the controller sees that sector entry and can request LLM
   guidance. In vision mode, that request can include the latest rendered frame
   and recent older frames.

## 10. Route Following Before Sector Entry

Before an aircraft enters the sector, it does not use LLM turns.

It follows its original route with `_route_step()`.

Route following does this:

1. Move forward along the route `LineString`.
2. Increase route progress by `AGENT_SPEED`.
3. Set the heading to the route tangent at the new route progress.

This keeps aircraft on the planned route before they enter the controlled
sector.

## 11. Free Flight After Sector Entry

After an aircraft has entered the sector, it uses free flight.

Free flight happens in `_free_step()`.

The movement is:

1. Read the aircraft turn from the action dictionary.
2. If no action exists for the aircraft, use `0`.
3. Add the turn to the current heading.
4. Move forward by `AGENT_SPEED` along the new heading.
5. Clip the position to the broad simulator play area.

A heading change is applied at the start of the step. The new heading then
persists until another turn changes it.

If the controller returns no action for an entered aircraft, the aircraft keeps
its current heading.

## 12. One Simulator Step In Detail

The main step method is `MultiAgentSectorCore.step()`.

One step does this in order:

1. Read the action dictionary.
2. Store each aircraft previous distance to destination.
3. Increase `n_step` by one.
4. Launch aircraft whose start step has arrived and whose launch position is
   clear.
5. Advance weather.
6. Move every launched, unfinished aircraft.
7. Update aircraft flags.
8. Compute pairwise distances.
9. Check collision.
10. Check dangerous weather entry.
11. Check whether all aircraft are finished.
12. Check step-limit truncation.
13. Set success or failure reason.
14. Update hazard predictions for the next guidance decision.
15. Compute reward.
16. Return observations, rewards, termination flags, truncation flags, and info.

Weather advances before aircraft move.

Hazard prediction is updated after movement and after termination checks.

## 13. Sector Entry

After an aircraft moves, `_update_agent_flags()` checks whether the aircraft
position is inside the sector polygon.

If the aircraft is inside for the first time:

```text
has_entered_sector = True
sector_entry_step = current n_step
last_event = entered_sector@<step>
```

This matters because the global LLM controller only guides aircraft that have
entered the sector.

Aircraft outside the sector stay on route and are not included in the action
prompt.

## 14. Finishing An Aircraft

After movement, the simulator checks the aircraft distance to destination.

If the distance is less than or equal to `GOAL_RADIUS`, the aircraft is marked
finished:

```text
finished = True
active = False
just_finished = True
last_event = finished@<step>
```

Finished aircraft are no longer active and no longer need guidance.

## 15. Hazard Prediction

Hazard prediction happens in `_update_hazard_predictions()`.

The simulator clones itself and simulates forward without applying new guidance
turns.

For each active aircraft, it predicts:

- minimum traffic separation,
- first future step where separation falls below `SAFE_R`,
- minimum weather clearance,
- first future step where weather is entered,
- first future step where the aircraft leaves the sector after entry,
- ranked threat aircraft.

The default lookahead is `HAZARD_LOOKAHEAD_STEPS`, currently 12.

For traffic:

- if predicted pair distance drops below `SAFE_R`, pair loss is predicted,
- if minimum separation is below `SAFE_R * PAIR_RISK_BUFFER`, conflict risk is
  predicted.

For weather:

- if predicted signed clearance is below zero, weather entry is predicted,
- if clearance is below the configured weather buffer, weather risk is
  predicted.

For boundary:

- if an entered aircraft is predicted to leave the sector, boundary warning is
  active,
- if an entered aircraft is already close to the boundary, boundary warning is
  also active.

Hazard prediction updates fields such as:

```text
conflict_predicted
weather_predicted
hazard_predicted
boundary_warning_active
safe_streak
threat_agents
predicted_min_sep
predicted_pair_loss_step
predicted_weather_clearance
predicted_weather_entry_step
predicted_boundary_exit_step
```

If a hazard is predicted, `safe_streak` becomes zero. If no hazard is predicted,
`safe_streak` increases by one.

## 16. The Global LLM Controller

The default controller is `GlobalLangGraphGuidanceController`.

It is called once per simulator step. The `use_vision` flag selects the prompt
mode:

```python
controller.choose_llm_actions(
    env.core,
    step=env.n_step,
    latest_frame_path=latest_frame_path,
    use_vision=False,  # text mode
)
```

or:

```python
controller.choose_llm_actions(
    env.core,
    step=env.n_step,
    latest_frame_path=latest_frame_path,
    use_vision=True,   # vision mode
)
```

When `use_vision=False`, the controller builds a text-only prompt.

When `use_vision=True`, the controller looks for the current rendered frame and
up to three recent older frames. Existing frame files are attached to the Ollama
request as base64 images.

The controller still asks for all due aircraft in one global prompt. It does not
make one separate LLM call per aircraft.

The controller graph has these nodes:

```text
collect_state
hltp_on_sector_entry
assign_stages
build_global_context
call_llm
parse_validate_retry
guardrail_actions
fallback_missing
commit_and_log
```

If no aircraft is actionable, the graph returns an empty action dictionary.

## 17. collect_state

This node finds entered aircraft.

An entered aircraft is:

- active,
- launched,
- not finished,
- already inside or past sector entry.

The node also creates frame annotations for aircraft that entered the sector at
the current step.

In text mode, this node does not attach images to the LLM call.

In vision mode, this node calls `_recent_frame_paths()`. It starts from the
latest frame path, such as `image_012.png`, then searches backward for recent
frames:

```text
image_012.png
image_011.png
image_010.png
image_009.png
```

Only files that actually exist are used. The prompt calls the newest image
`frame_0`; later frame indexes are older snapshots.

## 18. hltp_on_sector_entry

HLTP means high-level tactical planning.

HLTP runs when an aircraft has just entered the sector and does not already have
an HLTP plan.

HLTP is advisory. It does not directly apply a heading change.

The node:

1. finds newly entered aircraft,
2. looks ahead from the current step to step 60,
3. detects predicted pair conflicts,
4. builds planning context for conflicted aircraft,
5. finds right-side waypoint candidates,
6. finds later merge-back waypoint candidates,
7. builds an HLTP prompt,
8. calls the LLM,
9. validates returned waypoint names,
10. uses deterministic fallback if the LLM plan is missing or invalid,
11. stores the plan.

In text mode, the HLTP prompt asks the LLM to use the listed aircraft context,
conflict context, candidate waypoints, and weather context.

In vision mode, the HLTP prompt also asks the LLM to inspect the snapshots,
judge route geometry, closure, weather rings, and safe right-side deviations.

Both modes return JSON like:

```json
{
  "answer": {
    "plans": {
      "A1": {
        "first_waypoint_name": "ABC",
        "merge_back_waypoint_name": "DEF",
        "rationale": "brief reason"
      }
    }
  }
}
```

The stored HLTP plan is later included in the normal global guidance prompt. It
can help the LLM understand tactical intent, but current safety previews and
turn constraints can override it.

If no conflict is predicted for a newly entered aircraft, the annotation says
that no HLTP conflict was detected.

## 19. assign_stages

This node decides which entered aircraft need LLM guidance now.

The controller keeps a small state for each aircraft.

Important call names are:

```text
EXECUTE_TURN
EMERGENCY_MANEUVER
MERGE_BACK
```

In the current global controller, the first guidance call after sector entry is
`EXECUTE_TURN`.

After that, the aircraft is usually not prompted again unless a condition is met.

The main conditions are:

`IMMEDIATE_HAZARD`

A near-term traffic or weather hazard is predicted. The call becomes
`EMERGENCY_MANEUVER`.

`BOUNDARY_LOOKAHEAD`

The aircraft is near the boundary or predicted to leave the sector. The call
becomes `MERGE_BACK`.

`SAFE_STREAK`

The aircraft has been predicted safe for enough updates, but it is still off
route or badly aligned with its destination. The call becomes `MERGE_BACK`.

`ALIGNMENT_REPEAT`

A previous merge-back did not align the aircraft enough after the recheck wait.
The call becomes another `MERGE_BACK`.

If no condition is met, the aircraft is not included in the prompt for that
step. It will hold heading unless another action is returned later.

Actionable aircraft are sorted by urgency:

1. earliest predicted bad event,
2. smaller predicted minimum separation,
3. aircraft id.

## 20. Allowed Turns

The controller restricts turns by call type.

For `EXECUTE_TURN`, all action bins are allowed:

```text
-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30
```

For `EMERGENCY_MANEUVER`, zero is not allowed:

```text
-30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30
```

For `MERGE_BACK`, the controller computes heading error to the destination.

If the destination is already almost straight ahead, only `0` is allowed.

If the destination is to the left, only nonnegative turns are allowed.

If the destination is to the right, only nonpositive turns are allowed.

This keeps merge-back focused on destination recovery.

## 21. build_global_context

This node builds the LLM prompt for the current mode.

For each actionable aircraft, it creates:

- ranked traffic threat rows,
- candidate turn preview rows.

It also gathers:

- entered traffic context,
- advisory HLTP plans,
- global weather context,
- recent decision memory.

In vision mode, it also gathers recent frame paths.

When `use_vision=False`, the controller calls:

```text
GlobalPromptBuilder.build_prompt()
```

The main text prompt sections are:

```text
GLOBAL GUIDANCE TASK
MERGE_BACK PRIORITY
GUIDANCE-ELIGIBLE AGENTS THIS STEP
ENTERED TRAFFIC CONTEXT
HIGH-LEVEL TACTICAL PLANS
GLOBAL WEATHER
RANKED TRAFFIC THREATS
PER-AGENT TURN PREVIEWS
RECENT GLOBAL DECISION MEMORY
FRAME CONTEXT
```

When `use_vision=True`, the controller calls:

```text
GlobalPromptBuilder.build_vision_prompt()
```

The main vision prompt sections are:

```text
VISION-FIRST GLOBAL GUIDANCE TASK
MERGE_BACK VISUAL PRIORITY
VISION FRAME CONTEXT
GUIDANCE-ELIGIBLE DIRECTION CONTEXT
ENTERED TRAFFIC DIRECTION CONTEXT
HIGH-LEVEL TACTICAL PLANS
WEATHER MOTION SUMMARY
RANKED TRAFFIC THREATS
VISION ACTION CANDIDATES
RECENT GLOBAL DECISION MEMORY
VISION MEMORY ADVISORY
```

The text prompt tells the LLM to use structured state and simulator-tested
preview rows.

The vision prompt tells the LLM to inspect frames first, then use text signals
to anchor aircraft ids, destinations, threats, weather, memory, and candidate
turns.

## 22. Turn Previews And Vision Action Candidates

Turn previews are the most important guardrail in both guidance modes.

In text mode, this prompt section is called:

```text
PER-AGENT TURN PREVIEWS
```

In vision mode, this prompt section is called:

```text
VISION ACTION CANDIDATES
```

The controller does not simply ask:

```text
What should the aircraft do?
```

It asks:

```text
Here are the legal turns.
Here is what the simulator predicts for each turn.
Choose one of these listed turns.
```

In vision mode, the prompt also says to inspect frames first so the LLM can use
the image to understand the tactical situation before choosing from the listed
turns.

For each allowed turn, the controller:

1. clones the simulator,
2. applies that candidate turn on the first preview tick,
3. applies no additional turns after the first preview tick,
4. simulates forward for `TURN_PREVIEW_STEPS`,
5. records the result.

Each row contains:

```text
heading_change_deg
pair_loss_step
weather_entry_step
boundary_exit_step
min_sep_to_any
min_weather_clearance
traffic_buffer_ok
weather_buffer_ok
safe_over_preview
end_heading_error_to_dest_deg
start_distance_to_destination
end_distance_to_destination
progress_to_destination
start_cross_track_abs
end_cross_track_abs
cross_track_reduction
```

A row is marked `safe_over_preview = 1` only if:

- no pair loss is predicted,
- no weather entry is predicted,
- no boundary exit is predicted,
- traffic separation stays at or above `TRAFFIC_CAUTION_R`,
- weather clearance stays above the configured weather buffer.

The prompt shows compact rows like:

```text
turn=+10 | safe=1 | pair_loss=None | weather_entry=None | boundary_exit=None |
min_sep=8.20 | traffic_buffer_ok=1 | min_weather_clearance=11.50 |
weather_buffer_ok=1 | progress_to_destination=3.40 |
cross_track_reduction=1.20 | end_distance=25.10 |
end_cross_track=2.00 | end_heading_error=6.5
```

This tells the LLM what is expected to happen if that turn is selected now.

In text mode, the LLM uses these rows as the main decision evidence.

In vision mode, the image helps the LLM choose tactical intent, while the
candidate row turns that intent into an executable simulator action.

## 23. Candidate Row Sorting

Preview rows are sorted before they are shown to the LLM.

For normal and emergency guidance, sorting prefers:

1. safe rows,
2. no pair loss,
3. traffic buffer satisfied,
4. no weather entry,
5. weather buffer satisfied,
6. no boundary exit,
7. larger minimum separation,
8. larger weather clearance,
9. smaller final heading error,
10. smaller final cross-track error,
11. smaller absolute turn size.

For merge-back, sorting is safety-first and then recovery-focused:

1. safe rows,
2. no pair loss,
3. traffic buffer satisfied,
4. no weather entry,
5. weather buffer satisfied,
6. no boundary exit,
7. larger progress toward destination,
8. larger cross-track reduction,
9. smaller final heading error,
10. larger minimum separation,
11. smaller absolute turn size.

This means merge-back still avoids traffic and weather first. Once actions are
safe, it prefers actions that recover the aircraft toward its destination.

## 24. Prompt Safety Rules

Both prompts tell the LLM to act as a cautious air traffic control helper.

Rules shared by both modes:

- return one maneuver for every listed guidance-eligible aircraft,
- do not return actions for aircraft that are not listed,
- do not rename or output the call stages,
- use candidate rows as guardrails,
- prefer safe rows over unsafe rows,
- prefer traffic and weather buffer satisfaction,
- never choose a row with pair loss if another displayed row avoids pair loss,
- treat pair loss at step 1 or 2 as imminent collision,
- avoid the green weather ring when possible,
- never enter yellow, red, or magenta weather,
- during merge-back, do not recover through traffic or weather,
- return exactly one JSON object and no extra text.

Extra vision-mode rules:

- inspect the recent snapshots first,
- infer current motion from the frames,
- use text to anchor aircraft ids, destinations, threats, weather, and memory,
- choose visual tactical intent first,
- then choose a listed vision action candidate,
- ignore reward, total reward, success/failure, and scoreboard overlays in the
  images,
- avoid repeating recent ineffective turn patterns unless the frames show that
  the repeated turn is now helping.

The prompt also states:

```text
positive = left
negative = right
0 = hold current heading
```

## 25. Required LLM Output

The expected global guidance output is:

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

The LLM should include only the aircraft listed in the prompt.

The LLM should not output `call_name`; the controller already knows the call
type for each aircraft.

The simulator uses only the final heading change. Rationale text is kept in the
logs for debugging.

In text mode, the rationale should cite the chosen candidate row consequences.

In vision mode, the rationale should cite both visual evidence and the selected
candidate row consequence.

## 26. call_llm

If there are no actionable aircraft, the controller skips the LLM call.

If there are actionable aircraft, it sends the prompt to Ollama:

```text
<OLLAMA_HOST>/api/chat
```

The default host is:

```text
http://127.0.0.1:11434
```

The default model is:

```text
gemma3:12b
```

The request uses:

- `stream = False`,
- `temperature = 0`,
- `top_p = 0.95`,
- `num_ctx = 32768`,
- the configured max token value.

In text mode, the request contains:

- the text prompt only.

In vision mode, the request contains:

- the vision prompt as text,
- base64-encoded frame images under the chat message `images` field.

The controller only attaches frames that exist on disk. If no frame files are
available, the request falls back to text-only content even though `use_vision`
was requested.

The returned status can be:

- `ok`,
- `http_error`,
- `connection_error`,
- `empty_content`,
- `bad_response`.

If Ollama is not running, the status is usually `connection_error`.

## 27. parse_validate_retry

This node extracts JSON from the raw LLM response.

It searches for a JSON object and runs `json.loads()`.

Then it checks that every expected aircraft id appears under:

```text
answer.actions
```

If JSON is invalid or actions are missing, the controller retries.

The default retry limit is `2`, so it can make:

1. the original attempt,
2. retry 1,
3. retry 2.

If the response is still unusable, deterministic fallback handles the missing
actions.

## 28. guardrail_actions

This node turns parsed LLM actions into final candidate actions.

It:

1. ignores unexpected aircraft ids,
2. ignores malformed action payloads,
3. reads `maneuver.heading_change_deg`,
4. snaps the value to the nearest action bin,
5. snaps the value again to the allowed turns for the call type,
6. stores both requested and applied turns.

Example:

```text
LLM requested: +7
nearest action bin: +5
allowed for this call: +5
applied turn: +5
```

For emergency guidance, this prevents a zero-degree hold.

For merge-back, this prevents a turn that points away from destination recovery.

In text mode, this is the main turn guardrail: the requested value is snapped to
a legal action bin and then to a legal turn for the current call type.

Vision mode has an extra candidate-row safety resolver:

1. Find the displayed candidate row for the LLM-requested turn.
2. If the requested turn was not displayed, use the best sorted candidate.
3. If any displayed row avoids pair loss, reject a requested row that has pair
   loss.
4. If any displayed row avoids weather entry, reject a requested row that enters
   weather.
5. For `MERGE_BACK`, if every displayed row still has boundary exit, use the
   best sorted recovery row.
6. If the requested non-merge row is marked safe, keep it.
7. Otherwise, use the best sorted candidate.

This means vision mode lets the LLM use the images to choose among viable
candidates, but it still blocks some clearly unsafe visual choices when the
candidate table shows a safer option.

## 29. fallback_missing

If an expected aircraft still has no valid action, fallback creates one.

Fallback uses the same candidate preview rows and the same sorting rules. It
chooses the best sorted candidate turn.

Fallback is used when:

- the LLM call fails,
- the LLM returns invalid JSON,
- the LLM omits an aircraft,
- the LLM gives a malformed action,
- the HLTP LLM returns invalid waypoint names.

This allows the episode to continue when the LLM is unavailable or unusable.

## 30. commit_and_log

This node finalizes the action map.

It:

1. applies controller state updates,
2. adds frame annotations such as `A1: EXECUTE TURN`,
3. builds the final `{agent_id: heading_change_deg}` dictionary,
4. builds normalized JSON for logging,
5. writes prompt, response, normalized JSON, and index logs,
6. stores recent decision memory,
7. returns the actions to `evaluate_episode()`.

Controller state updates decide future guidance timing.

For `EXECUTE_TURN`:

- `execute_called` becomes true,
- stage becomes `WAIT_CLEAR`,
- merge-back recheck is cleared.

For `EMERGENCY_MANEUVER`:

- `execute_called` becomes true,
- emergency count increases,
- stage becomes `WAIT_CLEAR`,
- merge-back recheck is cleared.

For `MERGE_BACK`:

- merge-back count increases,
- stage becomes `MERGE_BACK_RECHECK`,
- merge-back recheck counter is reset.

Recent decision memory keeps the last four global decisions and includes them in
later prompts.

## 31. How Directional Guidance Actually Happens

The low-level action is always a heading change in degrees.

The directional guidance loop is:

1. The controller reads current aircraft state.
2. It decides which entered aircraft need guidance.
3. It assigns a call type to each one.
4. It determines legal turns for each call.
5. It simulates each legal turn.
6. It shows the best preview rows to the LLM.
7. The LLM chooses one heading change per listed aircraft.
8. The controller validates the chosen turns.
9. The simulator applies the final heading changes.
10. Each aircraft moves forward on its new heading.

If the LLM chooses:

```json
{ "heading_change_deg": -15 }
```

the aircraft turns 15 degrees right, then moves forward by `AGENT_SPEED`.

If the LLM chooses:

```json
{ "heading_change_deg": 0 }
```

the aircraft keeps its current heading for that step.

## 32. Why Guidance Does Not Happen Every Step

The controller is event-driven.

An entered aircraft is prompted only when the controller believes guidance is
due.

An aircraft may not be prompted if:

- it already received its first execute turn,
- no immediate hazard is predicted,
- it is not near the boundary,
- it does not need merge-back,
- it is waiting during merge-back recheck.

If no action is returned for an active entered aircraft, the simulator uses `0`
for that aircraft, so it holds heading.

## 33. Merge-Back Guidance

Merge-back is the recovery phase.

It is used when an aircraft should move back toward its destination after
avoidance, or when it is near the boundary.

Merge-back can trigger when:

- boundary warning is active,
- the aircraft has been safe for enough prediction updates,
- the aircraft is too far from its route,
- the aircraft heading is poorly aligned with destination,
- previous merge-back did not align it enough.

During merge-back:

1. the controller computes heading error to destination,
2. it allows only turns toward the destination side,
3. it simulates those turns,
4. it sorts safe rows by destination recovery,
5. the prompt tells the LLM not to recover through traffic or weather.

Merge-back does not mean returning exactly to the original route immediately.
It means recovering toward the destination and route direction when safe.

## 34. Emergency Maneuver Guidance

Emergency maneuver is used for near-term hazards.

A near-term hazard exists when:

- predicted pair loss is within `EMERGENCY_LOOKAHEAD_STEPS`,
- predicted weather entry is within `EMERGENCY_LOOKAHEAD_STEPS`,
- predicted weather clearance is already below the weather buffer.

Emergency maneuver disallows `0`, so the aircraft must turn.

The prompt tells the LLM to focus on immediate safety.

## 35. Weather Effects On Guidance

Weather affects guidance in three ways.

First, weather moves and changes size every simulator step.

Second, hazard prediction checks whether aircraft are predicted to enter weather
or get too close to the weather buffer.

Third, preview rows show weather consequences for every candidate turn.

The prompt tells the LLM:

- green is a caution or buffer ring,
- yellow, red, and magenta are no-go regions,
- avoid green with buffer when possible,
- never plan through yellow, red, or magenta.

The simulator termination check uses the terminal weather core. Entering the
softer green buffer is a risk, but entering the terminal core ends the episode
with weather failure.

## 36. Boundary Warnings

Boundary warnings help prevent aircraft from drifting out of the sector.

An aircraft gets `boundary_warning_active = True` when:

- it has entered the sector, and
- it is predicted to leave the sector, or
- it is already close to the boundary.

Boundary warning usually triggers merge-back guidance.

Boundary exit is not a direct termination condition in the current environment.
It is treated as a guidance risk and appears in preview rows.

## 37. Reward

The reward is one team reward value for the whole step.

The simulator computes one scalar called `reward`. It stores it in
`core.reward`, adds it into `core.total_reward`, and gives the same value to
every aircraft that is still active after the step. In `JointGuidanceEnv`, this
same scalar is returned as `team_reward`.

Exact formula:

```text
reward = 0.0

if any aircraft has launched:
    reward += mean(progress_toward_destination)

reward -= 0.01

if any aircraft has launched:
    reward -= 0.01 * mean(abs(cross_track_error))

reward -= 0.05 * threat_count
reward -= 0.05 * weather_risk_count
reward += 2.0 * newly_finished

if collision:
    reward -= 10.0

if weather_failure:
    reward -= 10.0

if team_success:
    reward += 20.0
```

The exact parts are:

- `mean(progress_toward_destination)`: average progress made by launched
  aircraft. For each launched aircraft, progress is
  `previous_distance_to_destination - current_distance_to_destination`. If an
  aircraft gets closer to its destination, this value is positive. If it moves
  away, this value is negative.
- `-0.01`: fixed step cost. This is applied every simulator step.
- `-0.01 * mean(abs(cross_track_error))`: route-deviation penalty. Cross-track
  error means how far launched aircraft are from their route line. Larger
  route deviation gives a larger penalty.
- `-0.05 * threat_count`: hazard penalty. `threat_count` is the number of
  launched aircraft with `hazard_predicted = True`.
- `-0.05 * weather_risk_count`: weather-risk penalty. `weather_risk_count` is
  the number of launched aircraft with `weather_predicted = True`.
- `+2.0 * newly_finished`: destination bonus. Each aircraft that newly reaches
  its destination during this step adds `+2.0`.
- `-10.0` for `collision`: collision penalty. This is added when any active
  aircraft pair is closer than `SAFE_R`.
- `-10.0` for `weather_failure`: terminal weather penalty. This is added when
  any active aircraft enters the terminal weather core.
- `+20.0` for `team_success`: success bonus. This is added only when all
  aircraft finish without collision, weather failure, or truncation.

Important details:

- There is no separate numeric penalty for truncation in the reward formula.
  A truncated step still receives the normal progress, cross-track, hazard,
  weather-risk, and step-cost terms.
- Collision and weather failure are separate checks. If both happen in the same
  step, both `-10.0` penalties apply.
- Weather risk can be penalized twice. If `weather_predicted = True`, then
  `hazard_predicted` is also true, so one aircraft can add `-0.05` through
  `threat_count` and another `-0.05` through `weather_risk_count`.
- A just-finished aircraft contributes to the reward calculation and can add
  `+2.0`, even though it is no longer active after the step.

Simple example:

```text
mean progress toward destination = +1.50
mean abs cross-track error = 4.00
threat_count = 2
weather_risk_count = 1
newly_finished = 1
collision = False
weather_failure = False
team_success = False

reward = 0.0
reward += 1.50
reward -= 0.01
reward -= 0.01 * 4.00      = -0.04
reward -= 0.05 * 2         = -0.10
reward -= 0.05 * 1         = -0.05
reward += 2.0 * 1          = +2.00

final reward = 3.30
```

The LLM does not directly optimize reward. It sees rendered frames, structured
vision prompts, and candidate rows.

## 38. Environment Termination

The episode can end in four main ways.

### Collision Failure

After aircraft move, the simulator computes active pairwise distances.

If any pair distance is less than `SAFE_R`, the episode terminates.

The failure reason becomes:

```text
collision
```

### Weather Failure

After aircraft move, the simulator checks whether any active aircraft is inside
the terminal weather core.

If yes, the episode terminates.

The failure reason becomes:

```text
weather
```

### Team Success

If all aircraft finish and there is no collision, weather failure, or
truncation, the episode succeeds.

The status field `failure_reason` is set to:

```text
success
```

The name is reused even for success.

### Step Truncation

If `n_step >= MAX_STEP` and the episode is not otherwise done, the episode
truncates.

The failure reason becomes:

```text
truncated
```

## 39. What A Step Returns

The lower-level simulator returns:

```text
observations
rewards
terminations
truncations
infos
```

The joint LLM environment simplifies this to:

```text
joint_observation
team_reward
done
truncated
info
```

The joint observation contains:

- local observations,
- global state vector,
- active agents,
- possible agents,
- route assignments.

The info dictionary contains:

- global state,
- possible agents,
- active agents,
- episode done flag,
- episode truncated flag,
- episode success flag,
- team reward,
- failure reason,
- max agents possible,
- weather dictionary,
- risky pairs,
- route assignments.

## 40. Local Observation Vector

Each local observation has 24 values:

1. aircraft x position,
2. aircraft y position,
3. destination x,
4. destination y,
5. distance to destination,
6. current heading in radians,
7. route heading in degrees,
8. heading error to route,
9. signed cross-track error,
10. inside-sector flag,
11. has-entered-sector flag,
12. boundary distance,
13. nearest-neighbor distance,
14. nearest-neighbor bearing,
15. conflict-predicted flag,
16. weather-predicted flag,
17. hazard-predicted flag,
18. safe streak,
19. predicted pair-loss step,
20. predicted weather-entry step,
21. predicted minimum separation,
22. predicted weather clearance,
23. weather center distance,
24. weather relative bearing.

The LLM does not see this raw vector directly. In both modes, the prompt builder
turns important parts of this state into readable text. In vision mode, the LLM
also receives rendered frames.

## 41. Logs And Outputs

Runtime artifacts stay inside:

```text
multi-agent-isolated-llm-guidance/
```

Rendered frames and GIFs go under:

```text
outputs/
```

Evaluation summaries go under:

```text
evaluation/
```

Prompt and response logs go under:

```text
memory/
```

Each run gets a directory like:

```text
memory/run_YYYYMMDDTHHMMSSZ__ep000001/
```

Global guidance logs look like:

```text
t001_global_guidance.prompt.txt
t001_global_guidance.response.txt
t001_global_guidance.normalized.json
_index.jsonl
```

HLTP logs look like:

```text
t001_hltp.prompt.txt
t001_hltp.response.txt
t001_hltp.normalized.json
```

The prompt file shows the text part of what the LLM saw.

In vision mode, the attached images are the frame files listed in the normalized
JSON debug section. They are not copied into the prompt text file because they
are sent to Ollama as encoded image payloads.

The response file shows exactly what the LLM returned.

The normalized JSON file is usually the best debugging file. It includes:

- final actions,
- requested turns,
- applied turns,
- source values such as `llm` or `fallback`,
- threat rows,
- preview rows,
- HLTP plans,
- parse status,
- retry count,
- fallback agents,
- weather,
- model name,
- eligible agent ids,
- entered agent ids,
- frame paths when images were used.

## 42. Frame Rendering

The simulator renders a frame after reset and after every step.

Frames are named:

```text
image_000.png
image_001.png
image_002.png
```

The frame shows:

- sector boundary,
- waypoints,
- route lines,
- aircraft positions and ids,
- destinations,
- goal circles,
- weather rings,
- weather future trail,
- step number,
- reward,
- total reward,
- active agents,
- failure reason.

If the controller has annotations, `evaluate_episode()` draws them on the
current frame. Examples:

```text
A1: ENTERED SECTOR
A1: EXECUTE TURN
A2: MERGE BACK (BOUNDARY)
```

After the episode ends, the runner writes a GIF from the saved frames.

## 43. If The LLM Is Wrong Or Unavailable

The episode can continue when the LLM response is unusable.

If Ollama is unavailable:

- `ollama_invoke()` returns `connection_error`,
- parsing is not attempted,
- fallback actions are generated for actionable aircraft.

If the LLM returns non-JSON content:

- JSON extraction fails,
- the controller retries,
- fallback actions are used if retries fail.

If the LLM omits an aircraft:

- the controller retries,
- fallback fills the missing aircraft if needed.

If the LLM returns a value outside the action bins:

- the value is snapped to the nearest action bin.

If the snapped value is not allowed for the call type:

- it is snapped to the nearest allowed turn.

All of this is logged.

## 44. What The LLM Controls

The LLM controls:

- selected heading change for each guidance-eligible aircraft,
- rationale text in the logs,
- advisory HLTP waypoint choices when HLTP is triggered.

The LLM does not control:

- aircraft speed,
- weather movement,
- route assignment,
- launch timing,
- collision threshold,
- weather failure threshold,
- reward calculation,
- episode termination,
- simulator physics,
- which aircraft are guidance-eligible,
- which call type an aircraft receives.

## 45. Legacy Per-Aircraft Controller

`rl_llm_multi/llm.py` also contains `MultiAgentThreeCallController`.

That older controller makes sequential local calls for one aircraft at a time.

The current default runner does not use it. The current runner uses
`GlobalLangGraphGuidanceController`, which creates one global prompt for all due
entered aircraft in the current step.

The phrase "three-call" refers to the three main low-level guidance phases:

1. `EXECUTE_TURN`,
2. `EMERGENCY_MANEUVER`,
3. `MERGE_BACK`.

## 46. Relationship Between Text And Vision Modes

Text mode is the default:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py
```

Vision mode is enabled with:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --use-vision
```

Vision mode does not replace text mode. It adds images on top of the same
structured text, simulator state, candidate rows, fallback logic, reward logic,
and termination logic.

The main differences are:

- text mode uses `GlobalPromptBuilder.build_prompt()`,
- vision mode uses `GlobalPromptBuilder.build_vision_prompt()`,
- text mode sends only text to Ollama,
- vision mode sends text plus available frame images,
- text mode calls the candidate section `PER-AGENT TURN PREVIEWS`,
- vision mode calls the candidate section `VISION ACTION CANDIDATES`,
- vision mode tells the LLM to inspect frames first,
- vision mode has extra candidate-row safety replacement logic.

## 47. Running Episodes

Basic text run:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py
```

Basic vision run:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --use-vision
```

Text run with more aircraft:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --num-agents 8
```

Vision run with more aircraft:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --use-vision --num-agents 8
```

Text run with two weather cells:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --num-weather-cells 2
```

Vision run with two weather cells:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --use-vision --num-weather-cells 2
```

Text run with specific routes:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --num-agents 2 --route-ids PATH5_REV PATH6
```

Vision run with specific routes:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --use-vision --num-agents 2 --route-ids PATH5_REV PATH6
```

Text run with shorter time limit:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --episode-step-cap 10
```

Vision run with shorter time limit:

```bash
python3 multi-agent-isolated-llm-guidance/run_llm_episode_gif_multi_agent.py --use-vision --episode-step-cap 10
```

The printed result includes:

- steps,
- total reward,
- success flag,
- truncated flag,
- failure reason,
- whether vision was used,
- number of weather cells,
- GIF path.

## 48. Running Tests

Run tests with:

```bash
python3 -m unittest discover multi-agent-isolated-llm-guidance/tests
```

Many tests patch the LLM call, so they do not need a live Ollama server.

Tests that instantiate the global controller need `langgraph` installed.

## 49. Practical Mental Model

The simplest model is:

1. The simulator moves aircraft and weather.
2. The simulator predicts near-future hazards.
3. The controller decides whether any entered aircraft needs guidance.
4. The controller simulates possible turns before asking the LLM.
5. The LLM chooses from the displayed turn consequences. In vision mode, it also
   inspects frames before choosing.
6. The controller cleans up the answer and fills missing actions.
7. The simulator applies the final heading changes.
8. The episode ends when all aircraft finish, a collision happens, terminal
   weather is entered, or the step limit is reached.

The LLM is a decision helper. The environment remains the final authority for
movement, safety checks, and termination.

When debugging, start with:

```text
rl_llm_multi/core.py
  reset()
  step()
  _update_hazard_predictions()

rl_llm_multi/llm.py
  _global_call_decision_for_agent()
  _preview_rows()
  _candidate_sort_key()
  _merge_back_candidate_sort_key()
  GlobalPromptBuilder.build_prompt()
  GlobalPromptBuilder.build_vision_prompt()
  _graph_parse_validate_retry()
  _graph_guardrail_actions()
  _graph_fallback_missing()
  _graph_commit_and_log()

evaluate.py
  evaluate_episode()
```

These functions explain almost all behavior in the integrated LLM guidance
loop.
