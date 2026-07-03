# LLM-Only Evaluation Results Summary

Source folder: `Evaluation_LLM_only`

## What I Evaluated

I evaluated the guidance controller by itself, without using MAPPO or any learned policy. In these runs, the controller directly chooses the heading-change actions for the aircraft at each guided step.

The purpose of this experiment was to measure how useful the real LLM guidance is compared with simpler non-LLM guidance sources that use the same controller pipeline. I wanted to check whether calling the real LLM gives better decisions than simpler baselines such as random turns, random preview-safe turns, or the deterministic best preview option.

The main outcome I measured was whether all aircraft reached their destinations successfully before any terminal failure occurred. I also measured the types of failures:

- `success`: all aircraft reached their destinations.
- `collision`: at least one aircraft pair violated the separation limit.
- `weather`: at least one aircraft entered the terminal weather region.
- `truncated`: the episode reached the 60-step limit before all aircraft finished.

I also recorded mean reward, mean number of steps, mean number of guidance calls, and wall-clock runtime.

## Methods Compared

- `real`: calls the configured Ollama LLM and uses the validated LLM-suggested actions.
- `uniform`: skips the LLM and chooses a random allowed turn.
- `preview_safe`: skips the LLM and randomly chooses from preview rows marked safe, with fallback if no safe row exists.
- `best_preview`: skips the LLM and chooses the top-ranked preview row deterministically.

The intention was not only to see whether real LLM guidance works, but also to test whether it adds value beyond the preview logic already available inside the controller.

## Evaluation Setup

All runs used the same evaluation setup: 300 episodes, base seed 42, 4 agents, and 2 weather cells. Episode seeds therefore cover 42 through 341.

| Method | Success | Collision | Weather | Truncated | Mean reward | Mean steps | Mean guidance calls | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `best_preview` | 63.3% (190/300) | 1.7% (5/300) | 14.0% (42/300) | 21.0% (63/300) | 83.58 | 40.79 | 24.06 | 6.0 h |
| `preview_safe` | 55.3% (166/300) | 0.3% (1/300) | 14.7% (44/300) | 29.7% (89/300) | 65.20 | 42.17 | 24.74 | 6.2 h |
| `real` | 39.0% (117/300) | 1.3% (4/300) | 23.3% (70/300) | 36.3% (109/300) | 26.82 | 39.49 | 23.05 | 22.4 h |
| `uniform` | 4.0% (12/300) | 53.0% (159/300) | 35.3% (106/300) | 7.7% (23/300) | -65.58 | 17.90 | 11.96 | 3.2 h |

## Interpretation

`best_preview` performed best overall. It had the highest success rate and the highest mean reward.

`preview_safe` was the second-best method. It had the lowest collision rate, but more episodes reached the step limit than `best_preview`.

`real` LLM guidance was better than uniform random guidance, but it was weaker than the preview-based methods in this run. Its main issue was a high truncation rate: many episodes did not fail immediately, but did not get all aircraft to their destinations before the 60-step limit.

`uniform` performed poorly. It caused many collisions and weather failures, which is expected because it chooses turns without using the preview ranking.

Overall, the preview-based heuristics were more reliable than the real LLM in this saved evaluation. The real LLM was also much slower because it made actual Ollama calls, while the other methods used synthetic guidance choices.
