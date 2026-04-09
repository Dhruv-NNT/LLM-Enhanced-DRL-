# How the ownship is guided to its destination

This file explains, in simple words, how the ownship moves when you run the `three_call_guidance` setup.

## Big idea

- The ownship is not asking the LLM at every single step.
- Most of the time, the ownship just keeps flying in its current heading.
- The LLM is only called at important moments:
  - when the ownship first enters the sector
  - when danger is still building and a stronger avoidance turn is needed
  - when it is time to go back toward the destination

- So the controller does two jobs:
  - it watches the situation every step
  - it decides when the LLM should be asked for a new turn

## What happens before the sector entry

- At the beginning, the ownship follows the normal planned path.
- This means it tries to stay aligned with the original ATCO path.
- In this phase, the LLM is not yet controlling the ownship.

## What happens when the ownship enters the sector

- The moment the ownship enters the sector, the controller switches to the LLM-guided part.
- The first important LLM call is `EXECUTE_TURN`.
- In this call, the LLM is shown:
  - the current ownship and intruder positions
  - the ownship destination
  - separation and heading information
  - a short preview table of possible turns
  - boundary warning information
  - controller state information

- The LLM chooses only one thing now:
  - `heading_change_deg`

- This is a turn amount like:
  - `+5`, `+10`, `+15`
  - `-5`, `-10`, `-15`
  - or `0` in some cases

- Positive means left turn.
- Negative means right turn.

## What happens after the LLM gives a turn

- After the LLM gives a turn, the controller applies that turn.
- From the next step onward, the ownship usually keeps flying in that same heading.
- The controller does this by sending action `0`.
- In this environment, action `0` means:
  - do not add a new turn
  - keep the current heading

- So even though the LLM no longer gives "hold steps", the ownship still continues in the chosen direction.

## What the controller keeps checking every step

- At every step, the controller checks:
  - is conflict predicted soon?
  - is the ownship getting too close to leaving the sector boundary?
  - has the ownship been safe for enough steps to start returning toward destination?
  - after merge-back, is the ownship still badly misaligned with the destination?

- This means the LLM does not need to keep repeating the same instruction.
- The controller keeps the aircraft moving, and only asks again when a real change is needed.

## How conflict checking works

- The controller looks ahead a few steps and predicts whether the ownship and intruder may lose safe separation.
- If conflict is predicted, the current heading is no longer enough.
- Then the controller asks for `EMERGENCY_MANEUVER`.

## What `EMERGENCY_MANEUVER` means

- `EMERGENCY_MANEUVER` is a new LLM call used when danger is still likely.
- The LLM is asked for another turn that helps restore safety.
- This turn is usually on the same side as the intruder logic used by the controller.
- After that emergency turn is applied, the ownship again keeps flying in that new heading until another trigger happens.

## How the boundary check works

- The controller also looks ahead to see if the ownship is about to leave the sector boundary.
- This is important because sometimes a safe avoidance turn can still carry the ownship too far outward.
- If the controller predicts that the ownship may leave the sector within the boundary lookahead window, it turns on a boundary warning.

- When this happens, and there is no more immediate conflict, the controller can ask for `MERGE_BACK` early.
- This helps the ownship start turning back before it actually crosses the boundary.

## How the ownship starts going back to the destination

- The controller does not send the ownship back immediately after one turn.
- It waits until one of these conditions happens:
  - the ownship has stayed safe for enough steps
  - or the boundary warning says it is time to come back early

- Then it asks the LLM for `MERGE_BACK`.

## What `MERGE_BACK` means

- `MERGE_BACK` means:
  - choose a turn that points the ownship back toward its destination side
  - then keep flying that heading

- This is the recovery phase.
- The goal is not only to avoid the intruder.
- The goal is also to make the ownship continue its trip toward the destination.

## What happens after merge-back

- After one merge-back turn, the controller does not keep calling the LLM every step.
- It waits for a short controller-side recheck period.
- During this small waiting period, the ownship keeps flying in the current heading.

- After that short wait, the controller checks again:
  - is the ownship still badly misaligned with the destination?
  - is the boundary warning still active?

- If yes, the controller can call `MERGE_BACK` again.
- If no, the ownship just keeps flying.

## Why repeated merge-back may happen

- One merge-back turn may not always be enough.
- For example:
  - the ownship may still be pointing too far away from the destination
  - or it may still be near the sector boundary

- In that case, the controller can ask for another merge-back turn.
- This helps the ownship gradually realign instead of making one overly large correction.

## How the ownship finally reaches the destination direction

- The overall pattern is:
  - follow the original path first
  - make an LLM-guided turn when entering the sector
  - keep flying that heading
  - do an emergency turn if conflict is still predicted
  - start merge-back when it becomes safe, or when the boundary guard says it is time
  - keep checking alignment and safety
  - repeat merge-back only if needed

- So the ownship is guided toward the destination in a step-by-step way.
- It does not constantly ask the LLM what to do.
- Instead, it uses:
  - a chosen heading
  - automatic continuation of that heading
  - safety checks
  - boundary checks
  - merge-back corrections

## Very simple summary

- The ownship first follows the normal path.
- Then the LLM tells it how much to turn.
- The ownship keeps flying that way automatically.
- If danger increases, the LLM gives an emergency turn.
- If things become safe, or if the boundary gets too close, the LLM gives a merge-back turn.
- The controller keeps repeating this logic until the ownship is safely moving back toward its destination.
