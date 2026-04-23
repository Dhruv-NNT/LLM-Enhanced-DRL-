# How memory is given to the LLM

This file explains, in simple words, how memory works in the `three_call_guidance` setup.

Think of this as an extra note on top of `ownship.md`.

`ownship.md` explains how the ownship is guided.

This file explains how past successful LLM calls can be shown again to the LLM during a new episode.

## Big idea

- Memory does not replace the controller.
- Memory does not call the LLM by itself.
- Memory does not decide when the ownship should turn.

- The controller still does the main jobs:
  - it watches the aircraft every step
  - it decides when to call the LLM
  - it builds the prompt
  - it checks and applies the final turn

- Memory only adds one extra thing:
  - if a similar successful past case exists, the controller can show it to the LLM as context

- So memory is like a small list of examples.
- It tells the LLM:
  - "In a similar situation before, this turn worked."

## Where memory is stored

- Memory is stored in a JSONL file.
- JSONL means:
  - one JSON object per line
  - each line is one saved memory case

- In the normal three-call setup, the default memory file is:
  - `three_call_guidance/memory/three_call_memory.jsonl`

- In the memory-vs-no-memory ablation, the memory file is kept inside the output folder for that run.
- This is done so the ablation does not overwrite or depend on the normal memory file.

- For example, a memory ablation run may create:
  - `memory-vs-no-memory/outputs/<run_id>/with_memory/memory/three_call_memory.jsonl`

## What one memory case contains

- A memory case is not the whole episode.
- It is one LLM call from a successful episode.

- A saved case stores:
  - which LLM call it came from
  - the step number
  - the situation features at that step
  - the turn that was finally applied
  - the outcome label
  - a short audit summary from the LLM response

- The call name is important.
- A case can come from:
  - `EXECUTE_TURN`
  - `EMERGENCY_MANEUVER`
  - `MERGE_BACK`

- These are kept separate during retrieval.
- An `EXECUTE_TURN` prompt only retrieves old `EXECUTE_TURN` cases.
- An `EMERGENCY_MANEUVER` prompt only retrieves old `EMERGENCY_MANEUVER` cases.
- A `MERGE_BACK` prompt only retrieves old `MERGE_BACK` cases.

## What features are used for memory

- Before the LLM is called, the controller builds a small memory signature for the current situation.

- This signature includes things like:
  - which call is happening now
  - intruder bearing from the ownship
  - separation distance
  - bearing to the destination
  - distance to the destination
  - predicted loss-of-separation step
  - time to closest point of approach
  - boundary warning information
  - safe streak count
  - merge-back mode

- These features are used only to find similar past cases.
- They are not a new action.
- They are just a way to compare the current situation with old successful situations.

## How retrieval works

- When memory is enabled, the controller asks the memory store:
  - "Do we have successful past cases like this one?"

- The memory store only considers cases that:
  - were saved from successful episodes
  - have the same call name as the current call

- Then it computes a similarity score.
- The score compares the current situation with the saved situation.

- The similarity score is normalized.
- This means different quantities are first converted into a common `0.0` to `1.0` range.

- A score near `1.0` means:
  - this feature is very similar

- A score near `0.0` means:
  - this feature is very different

- For numeric features, the memory store uses this idea:
  - compare the current value with the old value
  - divide the difference by a scale
  - convert that into a similarity score

- In simple terms:
  - small difference gives high similarity
  - large difference gives low similarity

- The formula is basically:
  - `similarity = 1 - abs(current - old) / scale`

- If the result would go below `0`, it is clipped to `0`.
- So every feature similarity stays between `0` and `1`.

## What scales are used for normalization

- The scale depends on what kind of feature is being compared.

- Angle-like values use a `45 deg` scale.
- This is used for things like:
  - intruder bearing
  - destination bearing

- Intruder distance uses a scale based on the safety radius:
  - `2 * safe_r`

- Boundary distance also uses a scale based on the safety radius:
  - `2 * safe_r`

- Destination distance uses the larger of:
  - `10`
  - `3 * safe_r`

- Conflict timing uses the relevant lookahead window.
- For example:
  - `EXECUTE_TURN` uses the turn preview window
  - `EMERGENCY_MANEUVER` uses the conflict lookahead window

- Safe streak uses the configured required safe streak as its scale.

- This normalization matters because the raw numbers have different meanings.
- For example:
  - a `10 deg` bearing difference is not the same kind of difference as a `10 unit` distance difference
  - so each feature needs its own scale before being combined

## How missing timing values are handled

- Sometimes a predicted event does not exist.
- For example:
  - there may be no predicted loss-of-separation step
  - there may be no predicted boundary exit step

- In that case, the memory store uses a sentinel value.
- The sentinel usually means:
  - "one step beyond the lookahead window"

- This lets the store compare:
  - no predicted event now
  - no predicted event in the saved case

- If both cases have no predicted event, they can still look similar.
- If one case has an event and the other does not, they look less similar.

## How different features are weighted

- Not every feature has the same importance.
- After normalization, the feature scores are combined using a weighted average.

- A larger weight means:
  - this feature matters more for matching

- A smaller weight means:
  - this feature still matters, but less strongly

- For `EXECUTE_TURN`, memory gives more weight to:
  - intruder bearing
  - predicted loss step
  - intruder distance
  - destination bearing

- For `EMERGENCY_MANEUVER`, memory gives more weight to:
  - predicted loss step
  - intruder bearing
  - intruder distance
  - time to closest point of approach

- For `MERGE_BACK`, memory gives more weight to:
  - destination bearing
  - merge-back mode
  - boundary warning state

- This is because each call type has a different purpose.

- `EXECUTE_TURN` and `EMERGENCY_MANEUVER` care more about conflict geometry.
- `MERGE_BACK` cares more about returning toward the destination and staying inside boundary constraints.

## What the final similarity score means

- After all feature scores are normalized and weighted, the memory store produces one final score.

- This final score is also between `0.0` and `1.0`.

- A score like `0.90` means:
  - the saved case is very similar to the current situation

- A score like `0.40` means:
  - the saved case is probably not useful enough

- The current setup only keeps cases with score at least:
  - `0.70`

- Then the matching cases are sorted:
  - highest similarity first
  - then earlier step
  - then case ID

- Finally, the controller keeps only the top cases.
- The current setup keeps at most:
  - `2` cases

## What the retrieval check uses

- The retrieval check uses things like:
  - intruder direction
  - intruder distance
  - destination direction
  - destination distance
  - predicted conflict timing
  - boundary state
  - merge-back state

- So the LLM is not given a long memory dump.
- It only sees a small number of relevant successful examples.

## How memory is written into the prompt

- The prompt builder creates the normal LLM prompt first.
- It includes:
  - the stage instruction
  - the current observation
  - derived safety signals
  - kinematics
  - turn preview table
  - boundary lookahead

- If memory finds matching cases, one extra section is added.

- That section starts with:
  - `Relevant successful past cases:`

- Each case is shown as a short memory card.

- A memory card includes:
  - similarity score
  - important situation features
  - the old successful turn
  - the old outcome label

- Example shape:
  - `1) sim=0.82 | intr=+20deg sep=7.4 loss=3 dest=-40deg bnd=None | turn=+15 | outcome=goal_reached`

- The exact fields depend on the call type.
- `MERGE_BACK` cards focus more on destination and boundary state.
- `EXECUTE_TURN` and `EMERGENCY_MANEUVER` cards focus more on intruder and conflict state.

## What the LLM should do with memory

- Memory is only context.
- It is not a command.

- The LLM still has to answer the current situation.
- The LLM still outputs only:
  - `heading_change_deg`

- The controller still checks the answer.
- If the LLM asks for an invalid turn, the controller snaps or resolves it using the normal rules.

- This means memory can guide the LLM, but it cannot force an unsafe or invalid action.

## What happens during an episode with memory

- At the start of the episode, the controller is created with:
  - `use_memory=True`

- The controller opens the memory store path.
- It loads the successful cases already saved in that file.

- When an LLM call is needed:
  - the controller builds memory features for the current situation
  - the memory store retrieves similar successful past cases
  - the prompt builder adds those cases to the prompt
  - the LLM gives a turn
  - the controller validates and applies the final turn

- After the turn is applied, the controller buffers the case.
- Buffering means:
  - keep it temporarily
  - do not save it permanently yet

- The buffered case stores the final applied turn.
- This is important because the LLM may have requested one turn, but the controller may have corrected it.
- Memory saves the action that was actually used.

## Why memory is only saved at the end

- The controller does not immediately write every LLM call to memory.
- It waits until the episode is over.

- This is because we only want memory from successful episodes.
- If an episode fails, then the turns from that episode should not become examples for the future.

- So each memory-enabled episode has two phases:
  - collect temporary memory during the episode
  - decide at the end whether to save or discard it

## What happens when the episode succeeds

- If the episode is successful, the controller commits the buffered cases.

- Committing means:
  - create final memory case IDs
  - mark the cases as successful
  - write them to the JSONL memory file
  - keep them available for future retrieval

- Each saved case gets an ID based on:
  - run ID
  - episode ID
  - call tag

- After this, later episodes can retrieve those cases.

## What happens when the episode fails

- If the episode is not successful, the controller discards the buffer.

- This means:
  - no failed turns are written to memory
  - no unsuccessful episode becomes a future example

- This keeps the memory store focused on successful behavior.

## What cold start means

- Cold start means:
  - memory is enabled
  - but the memory file has no saved cases yet

- This happens when:
  - the JSONL file does not exist yet
  - or the JSONL file exists but has no successful cases

- In cold start, the controller still checks memory.
- But retrieval returns nothing.

- So the LLM prompt does not get the `Relevant successful past cases:` section.

- The episode still runs normally.
- It is almost like no-memory for that first episode, except the controller is ready to save memory if the episode succeeds.

## What happens after a cold-start success

- If the cold-start episode succeeds, its LLM calls are saved.

- Then the next memory-enabled episode is no longer a pure cold start.
- It can retrieve those successful cases if they are similar enough.

- This creates an online learning pattern:
  - first memory episode has no examples
  - successful episodes add examples
  - later episodes can use those examples

## What happens if cold-start episodes fail

- If the first memory-enabled episode fails, nothing is saved.
- If the second memory-enabled episode also fails, nothing is saved.

- Memory stays empty until a successful episode happens.

- So memory does not automatically improve the controller.
- It only helps after there are useful successful cases to retrieve.

## What happens in no-memory mode

- In no-memory mode, the controller is created with:
  - `use_memory=False`

- Then the memory preparation step returns immediately.
- No memory features are used for retrieval.
- No memory cards are added to the prompt.
- No cases are committed at the end.

- The LLM still receives all the normal non-memory information:
  - observation
  - safety signals
  - kinematics
  - turn preview table
  - boundary information
  - controller state

- The only missing part is the successful-past-cases section.

## What the memory-vs-no-memory ablation does

- The ablation runs two settings:
  - `no_memory`
  - `with_memory`

- The same seeds are used in both settings.
- This means episode 1 in `no_memory` is paired with episode 1 in `with_memory`.
- Episode 2 is paired with episode 2, and so on.

- The goal is to compare:
  - how the controller behaves without successful past examples
  - how it behaves when successful past examples can be retrieved

- The memory-enabled setting uses a fresh study-local memory file.
- This gives a clean cold start for the first memory episode in that run.

## What memory does not change

- Memory does not change when the LLM is called.
- Memory does not change the three call types.
- Memory does not change the allowed turn bins.
- Memory does not remove controller-side validation.
- Memory does not store failed episodes.
- Memory does not store every step.
- Memory does not change the scenario seed.

- It only adds relevant successful past cases to the prompt when such cases exist.

## Very simple summary

- The controller still decides when to call the LLM.
- When memory is off, the LLM sees only the current situation.
- When memory is on, the controller first searches for similar successful past calls.
- If similar cases exist, the prompt includes short memory cards.
- The LLM can use those cards as examples.
- The controller still validates the turn.
- During the episode, memory cases are only buffered.
- If the episode succeeds, the buffered cases are saved.
- If the episode fails, the buffered cases are thrown away.
- Cold start means memory is on, but there are no saved successful cases yet.
