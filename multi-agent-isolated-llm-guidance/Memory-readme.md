# Decision Memory README

## Start Here

The memory system does three simple things:

1. After an LLM decision is made, it saves a small summary of that decision.
2. Before a future LLM decision, it searches older saved decisions for a similar
   case.
3. If it finds a strong match, it adds one short hint to the LLM prompt.

That is all.

The memory is not the main controller. It is not the MAPPO policy. It is not a
planner. It is only a small helper that says:

```text
This current situation looks similar to this older situation.
That older situation used this turn.
Only use this hint if the current preview still says it is safe.
```

The most important safety rule is:

```text
Current simulator preview always overrides memory.
```

If memory says `0 deg` but the current preview says `0 deg` is unsafe, the
memory is ignored.

## What Happened When You Ran Visual Audit

You ran:

```bash
python3 multi-agent-isolated-llm-guidance/evaluate_llm.py --memory-visual-audit
```

This did two things:

1. It ran a normal LLM-guided episode.
2. It copied some image frames into `memory/visual_audit/` so you can manually
   check whether a retrieved memory case really looked similar.

The visual audit does not change the LLM prompt. It does not change matching.
It only copies images and writes small `manifest.json` files for debugging.

## Why Two Visual Audit Folders Exist

There are two kinds of visual audit folders.

### 1. `cases/<source_episode_id>/<case_id>/`

This folder shows the old saved memory case.

Example:

```text
memory/visual_audit/cases/run_20260515T032627Z__ep000001__seed_42/case_26990aed88b33646/
```

Inside it you will see:

```text
manifest.json
past_00_image_019.png
past_01_image_018.png
past_02_image_017.png
past_03_image_016.png
```

Meaning:

- this memory case happened at step 19
- it involved agent `A2`
- it was a `MERGE_BACK` call
- it was a `boundary` case
- the remembered action was `0 deg`
- the copied images show what the scene looked like around that saved case

### 2. `retrievals/<current_episode_id>/<source_episode_id>/<case_id>/<step_agent>/`

This folder shows the current moment where memory was retrieved.

Example:

```text
memory/visual_audit/retrievals/run_20260515T033034Z__ep000001__seed_42/run_20260515T032627Z__ep000001__seed_42/case_26990aed88b33646/t022__A2/
```

The folder name means:

```text
run_20260515T033034Z__ep000001__seed_42  current episode
run_20260515T032627Z__ep000001__seed_42  source episode where memory was saved
case_26990aed88b33646                     source memory case
t022__A2                                  current step 22, current agent A2
```

Inside it you will see:

```text
manifest.json
current_00_image_022.png
current_01_image_021.png
current_02_image_020.png
current_03_image_019.png
```

Meaning:

- memory was retrieved at current step 22
- the current agent was `A2`
- the matched past memory case was `case_26990aed88b33646`
- similarity was recorded in `manifest.json`
- the copied images show what the current scene looked like at retrieval time

## How To Visually Check A Match

To check one memory match:

1. Open the retrieval manifest.
2. Look at `case_visual_dir`.
3. Open images from the retrieval folder.
4. Open images from the case folder.
5. Compare them with your eyes.

You are asking:

- Is this the same route?
- Is the aircraft in a similar part of the route?
- Is it close to the same boundary, weather, or traffic?
- Is it moving in a similar direction?
- Does the remembered action make sense in the current image?

If the answer is no, then the numeric matcher is too loose or missing a field.

## Important Change

The live controller now skips memory cases from the current episode when it
retrieves memory.

This means memory should come from older episodes, not from a decision that
happened a few steps earlier in the same episode.

Why this matters:

- it makes memory easier to understand
- it avoids calling a same-episode action a "past case"
- it avoids using an action before we know whether it was actually useful

If you run visual audit on a brand-new empty memory file, you may see case
folders but no retrieval folders. That is normal. The first run creates memory.
A later run can retrieve it.

## My Honest Recommendation

The current memory design is useful for testing and as a small LLM hint.

It is not yet strong enough to be treated as learned truth.

Good parts:

- it requires the same route
- it requires the same LLM call type
- traffic cases require the same primary intruder route
- it checks the current preview before showing the memory
- it gives only a tiny prompt hint

Weak parts:

- similarity is still hand-written
- weather cell identity is not stable yet
- a visually different case can still score high if the numbers are close
- the evaluator update is not implemented yet
- without horizon rollout, we cannot know whether a memory action was truly good

So the safe use is:

```text
Use memory as an advisory hint only.
Use visual audit to inspect whether matching is reasonable.
Do not let memory override preview safety.
Do not update memory quality until shadow rollout over N steps exists.
```

The rest of this README gives the detailed technical version.

This document explains the decision memory used by the LLM-guided MAPPO
episode runner.

The goal of this memory is simple:

When the LLM is asked to guide an aircraft, we look for a very similar past
decision. If one exists, we add one short memory line to the prompt. That line
does not replace the simulator preview. It only gives the LLM a small hint from
a past case.

The memory is intentionally small. It stores only the fields needed to answer:

- Is this the same route?
- Is this the same kind of problem?
- Is this aircraft in the same part of the route?
- Is the traffic, weather, or boundary geometry similar?
- Did the same kind of turn look safe or unsafe before?
- Was the past action later corrected by the evaluator?

Full prompts, full LLM responses, and detailed episode logs remain in
`llm_episode_logs/`. Decision memory only stores compact decision summaries.

## Files Involved

The current implementation is split across these files:

- `rl_llm_multi/memory.py`
  - Builds compact memory cases.
  - Saves memory to disk.
  - Loads memory from disk.
  - Finds similar past cases.
  - Scores memory similarity.

- `rl_llm_multi/llm.py`
  - Calls the memory retrieval before building the LLM prompt.
  - Inserts retrieved memory lines into the prompt.
  - Writes new memory cases after a global LLM decision is committed.

- `rl_llm_multi/core.py`
  - Stores reset-level metadata needed by memory:
    - seed
    - number of agents
    - number of weather cells
    - route IDs

- `configs.py`
  - Defines the memory path and memory limits.

Current config values:

```python
DECISION_MEMORY_PATH = MEMORY_DIR / "decision_memory.json"
DECISION_MEMORY_MAX_EPISODES = 100
DECISION_MEMORY_MIN_SIMILARITY = 0.75
DECISION_MEMORY_VISUAL_AUDIT_ENABLED = False
DECISION_MEMORY_VISUAL_AUDIT_DIR = MEMORY_DIR / "visual_audit"
DECISION_MEMORY_VISUAL_AUDIT_FRAME_COUNT = 4
DECISION_MEMORY_PROMPT_UNVERIFIED_ACTIONS = True
```

## Memory File Location

The persistent memory file is:

```text
multi-agent-isolated-llm-guidance/memory/decision_memory.json
```

This file is created automatically when the first memory cases are written.

If the file does not exist, retrieval simply returns no memory. The episode can
still run normally.

If the file is malformed, unreadable, or has the wrong shape, the memory loader
falls back to an empty memory object. The LLM run should not fail only because
memory failed.

## Top-Level JSON Shape

The memory file has one top-level object:

```json
{
  "schema_version": 1,
  "episodes": []
}
```

`schema_version` allows the memory format to change later without confusing old
files with new files.

`episodes` is a list of compact episode records.

## Episode Shape

Each memory episode looks like this:

```json
{
  "episode_id": "run_20260514T045457Z__ep000001__seed_42",
  "seed": 42,
  "num_agents": 4,
  "num_weather_cells": 1,
  "cases": []
}
```

### `episode_id`

A string that identifies the run and seed.

The current implementation builds it from:

- the log directory name
- the reset seed

Example:

```text
run_20260514T045457Z__ep000001__seed_42
```

This is useful for debugging. It is not used as the main similarity identity.
The route fingerprint is more important.

### `seed`

The environment seed used for the episode.

Seed is useful because it can reproduce the same traffic and weather setup.
However, seed alone is not enough. The same seed can still be unsafe to rely on
if route generation, number of agents, or curriculum settings change.

So seed is only a small scoring feature, not a hard gate.

### `num_agents`

The number of aircraft in the episode.

This helps separate a 2-agent problem from a 12-agent problem. It is not a hard
gate because the same local conflict can still appear with a different number
of background aircraft.

### `num_weather_cells`

The number of weather cells in the episode.

This is stored for context and future evaluator/debugging use.

### `cases`

The list of compact decision cases from that episode.

Each case is one agent decision at one step.

Each stored case has a stable `case_id`. This is not used for similarity. It is
used so a retrieved memory can be traced back to the exact stored case that
produced the prompt hint.

## Episode Retention

The memory keeps the latest 100 episodes, not the latest 100 cases.

This matters because one episode can contain many agent decisions. Keeping 100
episodes gives more complete recent experience than keeping 100 individual
decisions.

When a new episode is appended and the file has more than 100 episodes, older
episodes are dropped from the front of the list.

## Duplicate Case Handling

When cases are appended to an episode, the store avoids duplicate entries for
the same:

- step
- agent
- call

If a case with the same `(step, agent, call)` already exists in that episode, it
is replaced by the newer case.

This prevents repeated writes from creating multiple copies of the same
decision.

## Case Shape

Each case is one compact memory of one decision.

Example:

```json
{
  "case_id": "case_8dfca8c01a842011",
  "step": 7,
  "agent": "A2",
  "route": "R3|O1|D5|0",
  "call": "EXECUTE_TURN",
  "reason": "SECTOR_ENTRY",
  "type": "traffic",
  "place": {
    "route_frac": 0.42,
    "xtrack": -0.03,
    "heading_err": -8.5
  },
  "traffic": [
    {
      "route": "R7|O4|D2|0",
      "dist": 20.2,
      "bearing": -45.2,
      "loss": null,
      "min_sep": 6.62
    }
  ],
  "weather": {
    "clear_now": 18.0,
    "clear_pred": 18.0,
    "bearing": 37.0,
    "entry": null,
    "closing": false
  },
  "boundary": {
    "dist": 1.57,
    "bearing": 82.0,
    "exit": null,
    "closing": false
  },
  "preview": {
    "safe": [-30, -25, -20, -15, -10, -5, 0],
    "unsafe": [5, 10],
    "best_safe": -30,
    "best_progress": 0
  },
  "result": {
    "action": -15,
    "corrected": null,
    "note": null
  }
}
```

The case intentionally does not store everything. It stores only enough to find
a similar situation and give the LLM a useful one-line hint.

## Case Fields

### `case_id`

Stable ID for this exact stored decision case.

Example:

```text
case_8dfca8c01a842011
```

This ID is generated from stable decision identity fields:

- episode ID
- step
- agent label
- route fingerprint
- call name
- scenario type

It is not used as a similarity feature. Its purpose is exact traceability.

When memory retrieval finds a useful past case, the retrieval result carries a
reference to this exact case. Later, the evaluator should update this exact
case, not search again and hope it finds the same similar-looking memory.

### `step`

The simulation step when this decision happened.

Step is only a weak similarity signal. Two similar conflicts can happen at
different absolute steps, especially if the route is longer or the episode has
more agents.

Route position is more important than raw step.

### `agent`

The local agent label, for example `A2`.

This is stored for readability and prompt display. It is not treated as stable
identity.

Agent IDs can change meaning across seeds or curriculum settings. `A2` in one
episode may not represent the same route or same aircraft role in another
episode.

For matching, the ownship `route` is the real identity.

### `route`

The ownship route fingerprint.

Format:

```text
route_id|origin|destination|is_reversed
```

Example:

```text
R3|O1|D5|0
```

This is the strongest hard gate in memory retrieval.

A memory case is rejected if the current aircraft route does not match the past
case route.

Why this matters:

- Two aircraft can have similar distances to traffic but be on different routes.
- Two cases can have similar heading errors but be moving toward different
  destinations.
- A turn that worked on one route can be wrong on another route.

The route fingerprint keeps memory grounded in the same origin-destination
problem.

### `call`

The high-level call name that triggered the LLM guidance.

Examples may include values such as:

- `EXECUTE_TURN`
- `MERGE_BACK`
- other controller call names used by the current runner

This is a hard gate.

If the past case had a different `call`, it is rejected.

Reason:

The same geometry can require different behavior depending on what the
controller is asking the LLM to do. A conflict-avoidance turn and a merge-back
decision should not blindly share memory.

### `reason`

The more detailed reason for the call.

Example:

```text
SECTOR_ENTRY
```

This is not a hard gate. It is a score feature.

Reason:

Some reasons can describe the same useful geometry. For example, a boundary
lookahead and a safe-streak merge-back can both describe a similar return-to-route
problem.

So `call` is strict, but `reason` is flexible.

### `type`

The scenario type detected from the current state.

Common values:

- `traffic`
- `weather`
- `boundary`
- `merge_back`
- `nominal`

It can also combine types:

```text
traffic+boundary
traffic+weather
weather+boundary
traffic+weather+boundary
```

This is a hard gate.

If the current case is a traffic case, memory should not retrieve a pure weather
case. If the current case is a boundary case, memory should not retrieve a pure
traffic case.

### How `type` Is Chosen

The memory builder checks the local state:

- It marks `traffic` when there is a primary traffic threat, a predicted pair
  loss, or a low predicted minimum separation.
- It marks `weather` when weather entry is predicted or the current/predicted
  weather clearance is low.
- It marks `boundary` when boundary exit is predicted, boundary warning is
  active, or the aircraft is close to the boundary and still closing.
- It marks `merge_back` when the call is merge-back and no stronger threat type
  is present.
- Otherwise it marks `nominal`.

This keeps retrieval focused on the same kind of problem.

## `place`

`place` describes where the aircraft is relative to its own route.

```json
{
  "route_frac": 0.42,
  "xtrack": -0.03,
  "heading_err": -8.5
}
```

### `place.route_frac`

Normalized route progress.

It is computed as:

```text
progress along route / total route length
```

The value is usually between `0.0` and `1.0`.

Why normalized progress is used:

Raw distance to destination is not enough. A distance of 30 units means
something different on a short route than on a long route.

`route_frac` lets the matcher compare "same part of the route" across cases.

### `place.xtrack`

Signed cross-track distance from the route line.

Simple meaning:

- near `0`: aircraft is close to the route centerline
- positive or negative: aircraft is off to one side of the route

The sign matters. Being left of the route and right of the route are not the
same. A correction that helped on one side can be wrong on the other side.

### `place.heading_err`

Signed heading error relative to the destination.

Simple meaning:

- near `0`: aircraft is pointing toward the destination
- positive: aircraft is angled to one side
- negative: aircraft is angled to the other side

The sign and size both matter.

This answers:

"Am I deviating away from my destination in the same manner as a past case?"

## `traffic`

`traffic` stores up to two ranked traffic threats.

Example:

```json
[
  {
    "route": "R7|O4|D2|0",
    "dist": 20.2,
    "bearing": -45.2,
    "loss": null,
    "min_sep": 6.62
  }
]
```

The list is sorted so the primary threat comes first.

Only up to two threats are stored because the memory should stay small. The
primary threat is the most important one. A secondary threat can still improve
matching when two cases have the same primary conflict but different background
risk.

### `traffic[].route`

The intruder aircraft route fingerprint.

This uses the same format as ownship route:

```text
route_id|origin|destination|is_reversed
```

For traffic cases, the primary intruder route is a hard gate.

This prevents a bad match where the current aircraft has a similar distance to
some aircraft, but that aircraft is actually coming from a different route and
creating a different conflict geometry.

### `traffic[].dist`

Current distance to the traffic aircraft.

This helps answer:

"Is my proximity to traffic the same as the past case?"

### `traffic[].bearing`

Relative bearing from ownship heading to the traffic aircraft.

Simple meaning:

- negative: traffic is on one side
- positive: traffic is on the other side
- near `0`: traffic is roughly ahead

Distance alone is ambiguous. A traffic aircraft 20 units ahead is not the same
as a traffic aircraft 20 units behind or beside the ownship.

### `traffic[].loss`

Predicted pair-loss step, if one is predicted.

`null` means no pair loss is predicted in the current preview horizon.

This helps distinguish:

- immediate conflict
- later conflict
- no predicted conflict but low separation

### `traffic[].min_sep`

Predicted minimum separation.

This is one of the most important traffic risk fields. Two aircraft can be at
the same current distance but have very different future closest approach.

## `weather`

`weather` describes current and predicted weather geometry.

Example:

```json
{
  "clear_now": 18.0,
  "clear_pred": 18.0,
  "bearing": 37.0,
  "entry": null,
  "closing": false
}
```

### `weather.clear_now`

Current signed clearance to the nearest weather cell.

Positive means outside the protected weather ring.

Negative means inside the protected weather ring.

This answers:

"Is my proximity to the weather cell similar to a past case?"

### `weather.clear_pred`

Predicted weather clearance over the preview horizon.

This matters because the aircraft may be safe now but moving toward weather.

### `weather.bearing`

Relative bearing from ownship heading to the nearest weather cell center.

Weather clearance alone is ambiguous. Being close to weather on the left is not
the same as being close to weather on the right.

### `weather.entry`

Predicted weather entry step.

`null` means no entry is predicted in the preview horizon.

### `weather.closing`

Boolean flag.

`true` means that if the aircraft continues with no turn for one step, the
weather clearance becomes smaller.

This tells us whether the current motion is moving toward the weather risk.

## `boundary`

`boundary` describes current and predicted sector-boundary geometry.

Example:

```json
{
  "dist": 1.57,
  "bearing": 82.0,
  "exit": null,
  "closing": false
}
```

### `boundary.dist`

Signed distance to the nearest sector boundary.

Positive usually means inside the sector.

Negative usually means outside the sector.

This answers:

"Is my proximity to the boundary similar to a past case?"

### `boundary.bearing`

Relative bearing from ownship heading to the nearest boundary point.

Boundary distance alone is ambiguous. Being close to the north edge is not the
same as being close to the east edge.

Bearing helps identify which boundary edge the aircraft is near and whether the
aircraft is pointed toward it.

### `boundary.exit`

Predicted boundary exit step.

`null` means no boundary exit is predicted in the preview horizon.

### `boundary.closing`

Boolean flag.

`true` means that if the aircraft continues with no turn for one step, its
signed boundary distance becomes smaller.

This answers:

"Am I close to the same boundary and moving toward it in the same way?"

## `preview`

`preview` summarizes the simulator turn preview for the current decision.

Example:

```json
{
  "safe": [-30, -25, -20, -15, -10, -5, 0],
  "unsafe": [5, 10],
  "best_safe": -30,
  "best_progress": 0
}
```

The preview is extremely important because memory is never allowed to override
current safety.

### `preview.safe`

List of turn degrees that were safe in the current preview.

These are rows where `safe_over_preview` is true.

### `preview.unsafe`

List of turn degrees that were unsafe in the current preview.

These are rows where `safe_over_preview` is false.

### `preview.best_safe`

The best safe turn according to the current sorted preview rows.

If no turn is safe, it falls back to the first available preview row.

### `preview.best_progress`

The turn degree with the best progress toward destination.

If safe rows exist, it chooses among safe rows. Otherwise it chooses among all
rows.

This helps the matcher compare cases where the safe set is similar but the
route-progress preference is different.

## `result`

`result` stores the action remembered from the case and any future evaluator
correction.

Example:

```json
{
  "action": -15,
  "corrected": null,
  "note": null
}
```

### `result.action`

The action that was committed in the past case.

This is the action used by memory if no correction exists.

### `result.corrected`

Future evaluator field.

When the evaluator decides the remembered action was poor, it can write a better
turn degree here.

Example:

```json
"corrected": -25
```

When this field exists, retrieval prefers the corrected action over the original
action.

### `result.note`

Future evaluator field.

This is a short plain-language reason explaining why the original memory action
was poor.

Example:

```json
"note": "traffic buffer kept shrinking"
```

This note can be shown to the LLM in the retrieved memory line.

## When Memory Is Read

Memory is read before the global LLM prompt is built.

Current flow:

1. The controller identifies actionable agents.
2. For each actionable agent, the memory system builds a compact current query
   case.
3. The memory store scans previous episodes and cases.
4. It rejects cases that fail hard gates.
5. It scores the remaining cases.
6. It keeps only strong matches.
7. It returns a `memory_ref` for each selected match.
8. It inserts at most two short memory lines into the prompt.

If no strong match is found, the prompt has no memory section.

During a live controller run, retrieval skips the current episode ID. This keeps
the memory focused on older episodes instead of reusing a decision from a few
steps earlier in the same episode.

The `memory_ref` looks like this:

```json
{
  "episode_id": "run_20260514T045457Z__ep000001__seed_42",
  "case_id": "case_8dfca8c01a842011"
}
```

This reference is not shown to the LLM. It is kept in logs/debug data so the
future evaluator can update the exact memory case that was retrieved.

## When Memory Is Written

Memory is written after the global LLM decision is committed and logged.

Current flow:

1. The LLM produces guidance.
2. The controller parses and applies the final action.
3. The final action is used to build a compact memory case.
4. The memory case is appended to `decision_memory.json`.
5. The file is saved atomically.

The remembered action is the committed action, not just the raw text the LLM
requested.

This is important because the controller may clamp, parse, or adjust the action
before it is actually used.

## Hard Gates Used Before Scoring

A memory case must pass all hard gates before it can be scored.

The current hard gates are:

1. Same ownship route.
2. Same call name.
3. Same scenario type.
4. For traffic cases, same primary intruder route.
5. The remembered action must exist in the current preview.
6. If a corrected action exists, the corrected action must exist in the current
   preview.
7. The remembered or corrected action must not violate current hard safety when
   safer alternatives exist.

These gates are strict on purpose.

The biggest risk with memory is retrieving a case that looks numerically similar
but is actually a different route, different traffic conflict, or different
safety situation.

## Current Safety Check For Remembered Actions

Before memory is shown to the LLM, the memory action is checked against the
current preview.

The selected remembered action is rejected if:

- it is not present in the current preview rows
- it has predicted pair loss while another row avoids pair loss
- it has predicted weather entry while another row avoids weather entry
- its traffic buffer is bad while another row has good traffic buffer
- its weather buffer is bad while another row has good weather buffer
- it is unsafe over preview while another row is safe

This means memory cannot say:

"Use -15 degrees"

if the current simulator preview says -15 degrees is unsafe and there are safer
choices.

Memory is advisory only.

## Similarity Score

After hard gates, each remaining case gets a numeric similarity score from `0`
to `1`.

The configured threshold is:

```text
0.75
```

If the best score is below `0.75`, no memory is shown.

The score is weighted like this:

```text
25% place
20% traffic
15% weather
15% boundary
15% preview pattern
10% misc context
```

## Place Similarity - 25%

Place compares:

- route fraction
- signed cross-track distance
- signed heading error

Weights inside place:

```text
45% route_frac
25% xtrack
30% heading_err
```

This is designed to answer:

"Is the aircraft in the same part of the same route and deviating in the same
way?"

## Traffic Similarity - 20%

Traffic compares:

- primary intruder route
- current distance
- relative bearing
- predicted loss step
- predicted minimum separation
- secondary threat, if present in both cases

For traffic cases, primary intruder route is already a hard gate. The score then
checks whether the details are close.

If both cases have a secondary threat, the score uses:

```text
80% primary threat
20% secondary threat
```

If only one side has traffic, traffic similarity is low.

If neither side has traffic, traffic similarity is perfect for that part.

## Weather Similarity - 15%

Weather compares:

- current clearance
- predicted clearance
- bearing to nearest weather cell
- predicted entry step
- closing flag

Weights inside weather:

```text
25% clear_now
25% clear_pred
20% bearing
20% entry
10% closing
```

This is designed to answer:

"Is the aircraft near weather in the same way and moving toward or away from it
in the same way?"

## Boundary Similarity - 15%

Boundary compares:

- signed boundary distance
- bearing to nearest boundary point
- predicted exit step
- closing flag

Weights inside boundary:

```text
35% dist
25% bearing
25% exit
15% closing
```

This is designed to answer:

"Is the aircraft near the same kind of boundary situation and moving toward it
in the same way?"

## Preview Similarity - 15%

Preview compares:

- safe turn set
- unsafe turn set
- best safe turn
- best progress turn

Weights inside preview:

```text
35% safe set
25% unsafe set
20% best_safe
20% best_progress
```

Safe and unsafe sets are compared with set overlap.

This matters because two cases can have the same geometry but different
available actions due to current traffic, weather, or boundary constraints.

## Misc Similarity - 10%

Misc compares:

- call reason
- seed
- number of agents
- step proximity

Weights inside misc:

```text
35% reason
25% seed
20% num_agents
20% step
```

These are intentionally weak signals.

They help break ties, but they should not dominate route, traffic, weather,
boundary, or preview similarity.

## Prompt Insertion

Memory prompt insertion is intentionally surgical.

If there is no strong match, the prompt does not include a memory section.

By default, unverified past actions are inserted into the LLM prompt only as a
comparison point.

Reason:

- a past action is only proof that the LLM chose it before
- it is not proof that the action was good
- showing an unverified action can anchor the LLM and make the run worse

So the wording avoids saying "worked" or "preferred".

For an unverified memory, the prompt says:

```text
RETRIEVED DECISION MEMORY:
- A2 sim=0.82 unverified memory comparison only: similar past case used -15 deg. Do not copy unless current preview independently supports it.
- Memory is not proof the action is good; current preview, safety rules, and phase priorities decide.
```

If the evaluator has corrected the past memory:

```text
RETRIEVED DECISION MEMORY:
- A2 sim=0.86 evaluator-corrected memory: past -15 deg was poor because "traffic buffer kept shrinking"; corrected -25 deg is preferred only if current preview is safe.
- Memory is not proof the action is good; current preview, safety rules, and phase priorities decide.
```

Rules:

- max one memory line per agent
- max two memory lines total
- omit the whole section when no strong match exists
- never include full past prompt text
- never include long LLM rationale
- never tell the LLM to blindly copy memory

## Why The Prompt Text Is Short

The memory line should help the LLM, not distract it.

The LLM already receives the current simulator state and current preview. Those
are more important than memory.

The memory should only say:

- this looks like a past case
- the similarity score is strong
- this action was corrected by evaluator, if correction exists
- current preview safety still wins

## Retrieval Output In Debug Logs

The controller stores retrieved memory information in the normalized debug state
under `retrieved_memories`.

Each retrieved memory includes a `memory_ref`:

```json
{
  "memory_ref": {
    "episode_id": "run_20260514T045457Z__ep000001__seed_42",
    "case_id": "case_8dfca8c01a842011"
  }
}
```

This is important for evaluator updates. The evaluator should update by
`memory_ref`, not by searching for a similar case again.

If memory retrieval or writing fails, the controller records `memory_error`
instead of crashing the run.

This keeps memory as an optional guidance layer, not a required dependency.

## Optional Visual Audit

The memory system has an optional visual audit mode.

It is off by default:

```python
DECISION_MEMORY_VISUAL_AUDIT_ENABLED = False
```

You can enable it from the evaluation script:

```bash
python3 multi-agent-isolated-llm-guidance/evaluate_llm.py --memory-visual-audit
```

When enabled, the controller copies recent rendered frames into:

```text
multi-agent-isolated-llm-guidance/memory/visual_audit/
```

This is only for testing and human inspection. It does not change the memory
similarity score. It does not change the prompt. It does not send extra images
to the LLM unless `--use-vision` is also enabled.

### Visual Audit Folder Shape

Stored memory case frames:

```text
memory/visual_audit/cases/<source_episode_id>/<case_id>/
  manifest.json
  past_00_image_007.png
  past_01_image_006.png
  ...
```

Retrieval-event frames:

```text
memory/visual_audit/retrievals/<current_episode_id>/<source_episode_id>/<case_id>/t007__A2/
  manifest.json
  current_00_image_007.png
  current_01_image_006.png
  ...
```

The case folder shows what the stored memory case looked like when it was
created.

The retrieval folder shows what the current situation looked like when that
memory case was retrieved.

The retrieval manifest points back to the exact stored memory:

```json
{
  "kind": "memory_retrieval",
  "memory_ref": {
    "episode_id": "run_20260514T045457Z__ep000001__seed_42",
    "case_id": "case_8dfca8c01a842011"
  },
  "case_visual_dir": "multi-agent-isolated-llm-guidance/memory/visual_audit/cases/run_20260514T045457Z__ep000001__seed_42/case_8dfca8c01a842011"
}
```

This lets you visually compare:

- the current retrieval frame
- the past case frame
- the numeric similarity score
- the exact stored case that produced the memory hint

### Visual Audit Limits

Visual audit copies whole-sector frames, not cropped per-agent images.

That is deliberate. The matching logic depends on route, traffic, weather,
boundary, and preview context. A crop around one aircraft can hide why a match
was good or bad.

Visual audit can produce many image files, so keep it off during normal
training. It is meant for debugging the memory design.

## Why Route Fingerprint Is More Important Than Agent ID

Agent names like `A2` are local labels.

Across different seeds or curriculum settings:

- `A2` may use a different origin
- `A2` may use a different destination
- `A2` may meet different traffic
- `A2` may have a different role in the scenario

So memory stores `agent` mostly for readability.

The real identity is:

```text
route_id|origin|destination|is_reversed
```

This makes retrieval much less likely to confuse unrelated cases.

## Why Boundary Bearing Is Stored

Boundary distance alone is weak.

Two aircraft can both be 2 units from a boundary but near completely different
edges.

The memory stores:

- distance to boundary
- bearing to nearest boundary point
- predicted exit step
- closing flag

Together these answer:

"Am I close to the same kind of boundary in the same direction?"

## Why Weather Bearing And Closing Are Stored

Weather clearance alone is weak.

Two aircraft can both be 10 units from weather, but one may have weather on the
left and the other may have weather on the right.

The memory stores:

- current clearance
- predicted clearance
- bearing to nearest weather cell
- predicted entry step
- closing flag

Together these answer:

"Am I near weather in the same way, and am I moving toward it?"

## Why Up To Two Traffic Threats Are Stored

The primary traffic threat is the most important.

But a second aircraft can change the best turn. For example:

- turning left may avoid the primary threat but move toward a secondary threat
- turning right may avoid both

So memory stores up to two traffic threats.

The primary threat is used as a hard gate for traffic cases. The secondary
threat only improves or lowers the score.

## Why Preview Pattern Is Stored

The same physical-looking situation can have a different safe action set.

Example:

- In one case, `-15` and `-20` are safe.
- In another case, only `-30` is safe because weather is closer.

The preview pattern protects against copying a past action into a current case
where the simulator says it is no longer safe.

## Evaluator Engine Plan

The evaluator engine is reserved for the next stage.

The memory schema already has two fields for it:

```json
{
  "corrected": null,
  "note": null
}
```

The intended evaluator flow is:

1. A memory case is retrieved and shown to the LLM.
2. The exact retrieved case is recorded through `memory_ref`.
3. The LLM-guided action is simulated.
4. A shadow policy action from the RL/MAPPO policy is also simulated.
5. Returns are compared over a short horizon, for example 5 steps.
6. If the memory-backed action performs poorly, the evaluator LLM writes:
   - a short reason in `result.note`
   - a better degree in `result.corrected`
7. The memory store updates the exact case pointed to by `memory_ref`.
8. Future retrieval uses `corrected` instead of the original `action`.

Example after evaluator update:

```json
"result": {
  "action": -15,
  "corrected": -25,
  "note": "traffic buffer kept shrinking"
}
```

Then the future prompt can say:

```text
past -15 deg was poor because "traffic buffer kept shrinking"; corrected -25 deg is preferred only if current preview is safe.
```

## Evaluator Should Stay Small

The evaluator should not write long explanations.

Good evaluator note:

```text
traffic buffer kept shrinking
```

Bad evaluator note:

```text
The aircraft started at a difficult angle and there were many possible tactical
choices, but one of them seemed better after considering the route and the
traffic movement in detail...
```

The prompt needs a short correction, not a long story.

## Evaluator Should Not Rewrite The Whole Case

The evaluator should only update:

- `result.corrected`
- `result.note`

It should not rewrite:

- route
- traffic geometry
- weather geometry
- boundary geometry
- preview safe set

Those fields describe what happened at decision time. They should stay stable.

## Evaluator Should Respect Current Preview Safety

Even if a past case has a corrected action, retrieval still checks that action
against the current preview.

If the corrected action is not available or is unsafe in the current case, the
memory is not shown.

This prevents stale corrections from becoming unsafe instructions.

## What Is Implemented Now

Implemented now:

- compact memory JSON file
- episode-level retention
- stable `case_id` on stored memory cases
- case building
- route fingerprinting
- traffic/weather/boundary/place/preview fields
- hard-gated retrieval
- numeric similarity scoring
- `memory_ref` returned for retrieved matches
- max two prompt memory lines
- prompt omission when no strong match exists
- unverified past actions shown as comparison-only prompt context by default
- reserved evaluator fields
- optional visual audit frames grouped by current/source episode, off by default

Not implemented yet:

- shadow MAPPO versus LLM return comparison
- evaluator LLM call
- automatic update of `corrected`
- automatic update of `note`

## Practical Example

Suppose the current aircraft is `A2`.

It is:

- on route `R3|O1|D5|0`
- at route fraction `0.42`
- heading slightly left of destination
- close to traffic on route `R7|O4|D2|0`
- seeing a safe preview where left turns are safer than right turns

Memory scans old cases.

A past case is rejected if:

- it was a different ownship route
- it had a different call
- it was a weather case instead of a traffic case
- its primary traffic was on a different route
- its remembered action is unsafe now

If a past case passes those gates, memory scores it.

If the score is `0.82`, the prompt may get:

```text
RETRIEVED DECISION MEMORY:
- A2 sim=0.82 unverified memory comparison only: similar past case used -15 deg. Do not copy unless current preview independently supports it.
- Memory is not proof the action is good; current preview, safety rules, and phase priorities decide.
```

The LLM can use that hint, but it still must obey the current preview.

## How To Debug Memory

Check the persistent memory file:

```text
multi-agent-isolated-llm-guidance/memory/decision_memory.json
```

Check per-step prompt logs:

```text
multi-agent-isolated-llm-guidance/llm_episode_logs/
```

Look for:

```text
RETRIEVED DECISION MEMORY:
```

If the section is missing, likely reasons are:

- no memory file exists yet
- no past case has the same route
- no past case has the same call
- no past case has the same scenario type
- traffic case has a different primary intruder route
- similarity score is below `0.75`
- remembered action is not available in the current preview
- remembered action is unsafe while safer alternatives exist

Check normalized debug JSON for:

```text
retrieved_memories
memory_error
```

## Important Limitations

Memory is not a planner.

Memory is not a replacement for MAPPO.

Memory is not a replacement for simulator preview safety.

Memory is only a small retrieval system that says:

"This looks like a past decision on the same route and same kind of conflict.
Here is what happened before."

The current preview remains the source of truth.

## Design Principle

The memory system should stay boring and strict.

It should prefer no memory over wrong memory.

A missing hint is acceptable. A misleading hint can degrade training.

That is why the implementation uses:

- strict route gates
- strict call gates
- strict scenario type gates
- primary intruder route gates for traffic
- current-preview safety checks
- a similarity threshold
- very short prompt insertion

This keeps memory useful without letting it dominate the LLM guidance.
