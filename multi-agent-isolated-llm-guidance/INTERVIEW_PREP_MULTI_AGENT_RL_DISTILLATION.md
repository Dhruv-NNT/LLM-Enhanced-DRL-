# Interview Prep: Multi-Agent RL Air-Traffic Controller with Multi-Teacher LLM Distillation

These notes explain your own research back to you in plain language, so you can talk about it
confidently. Read the **"one-breath summary"** and the **"How to tell the story"** sections out loud
a few times — those are the parts that matter most when they say "tell me about your research."

---

## 1. What this project actually is (the honest origin story)

This is your research on **teaching a reinforcement-learning air-traffic controller using several
"teachers" at once**, in a custom multi-aircraft sector simulator.

There are really **two layers** to it, and it helps to keep them straight:

- **The base system (built first):** a **MAPPO** controller — multiple aircraft in a sector, each
  learning to choose heading changes to reach their destination without colliding or flying into a
  storm. On top of plain RL, an **LLM acts as an advisory teacher** during training, but with a hard
  safety rule: the LLM never flies the plane. Its suggestions are *validated by short look-ahead
  simulations* ("shadow evaluation") and only then used to gently shape the RL policy's training loss.

- **The new direction (your main contribution now):** **multi-teacher competence-weighted
  distillation.** Instead of trusting one teacher, the controller is distilled from **several
  complementary teachers at once** — one LLM teacher plus two cheap rule-based heuristic teachers —
  and at every decision the system works out *which teacher is actually right in this situation* and
  trusts that one more. The thesis you're testing: **"no single teacher is best everywhere."**

**Here's what you can honestly claim you did:**

> - Built a multi-agent RL (MAPPO) controller for a safety-critical air-traffic-control sector
>   environment, where several aircraft simultaneously learn conflict-free routing.
> - Designed and implemented a **multi-teacher knowledge-distillation** framework: turned each
>   "guide" (an LLM and two heuristics) into a small teacher network via behavioural cloning, then
>   distilled a MAPPO student from all of them with a **competence-weighted** loss that trusts each
>   teacher only where a short look-ahead shows it beats the current policy.
> - Preserved a strict **isolation guarantee** — the RL policy is the only thing that ever controls
>   the real environment; teachers shape training only.
> - Ran the full experiment ladder (baseline, single-teacher, leave-one-out, and combined runs),
>   generated the teacher training data, trained the teachers, and evaluated with statistical
>   comparisons across held-out scenarios.

**Important nuance for the interview:** this is **research in progress**, not a finished paper with
final numbers. Be upfront about it — it's a strength, not a weakness, if you're precise about *what
is already solid* and *what is still running*. Two things are solid (an early finding that no single
teacher wins on everything); the headline experiment (the proposed competence-weighted combination)
is **still training** at the time of writing. Saying that plainly shows scientific honesty.

*(If asked "is this published?": "It's ongoing research — I built and ran the whole distillation
pipeline and I have early, statistically-tested findings; the final headline run is still training,
so I'm careful not to over-claim the combined result yet.")*

---

## 2. The one-breath summary (memorize this)

> "It's a multi-agent reinforcement-learning controller for air traffic control — several aircraft in
> a sector, each choosing heading changes to reach their destination without colliding or entering a
> storm. The interesting part is *how* it's taught. Instead of trusting one expert, I distil the
> controller from several complementary teachers at once — an LLM teacher plus two cheap rule-based
> heuristics — and at every decision the system simulates a few steps ahead to see which teacher is
> actually right *here*, and trusts that one more. The core finding is that no single teacher is best
> everywhere: the rule-based teacher avoids collisions better, the LLM teacher avoids weather better.
> And throughout, a strict safety rule holds — the RL policy always flies the plane; the teachers only
> shape its training, they never touch the real environment."

---

## 3. How the method works — the actual pipeline (know this cold)

The system has two stages: **turn each guide into a teacher network (offline)**, then **train the RL
controller while the teachers shape its loss (online)**.

**Stage 1 — make the teachers (offline, run once):**
```
For each "guide" — LLM (via a local Ollama model) and two heuristics:
   Play many games; at every decision record  (situation  →  the guide's chosen heading)
        │
        ▼
   Train a small "teacher network" by supervised behavioural cloning:
   given the situation, predict the guide's move
        │
        ▼
   Output: three frozen teacher networks  (llm, best_preview, preview_safe)
```
The two heuristics come free of the LLM: **best_preview** deterministically picks the top-ranked
"turn preview" candidate; **preview_safe** picks randomly among the candidates flagged safe. The LLM
teacher is the expensive one (a real language-model call per decision at data-generation time).

**Stage 2 — train the RL controller (online), with the teachers shaping its loss:**
```
At each step during MAPPO training:
   1. MAPPO chooses the real action for every aircraft         ← this is what actually flies
   2. Ask each active teacher what IT would do (cheap — just a network forward pass)
   3. COMPETENCE WEIGHTING (the key step): clone the simulator and roll ~5 steps ahead
      for MAPPO's move and for each teacher's move; a teacher earns trust in proportion
      to how much its move beats MAPPO over that short horizon
   4. Add a distillation loss that pulls the policy toward the teachers it trusts here
   5. Step the real environment with MAPPO's action ONLY  (the isolation rule)

Loss = PPO_loss  +  w(t) · mean over teachers[ competence_weight · divergence(student, teacher) ]
        │            │
        │            └─ w(t): a global weight that DECAYS early on, then holds at a small floor
        └─ normal multi-agent PPO (policy + value + a little entropy)
```

**The experiment ladder (this is how you prove the thesis):**
```
T0          — no teacher (plain MAPPO baseline)
S_BP / S_PS / S_LLM   — single-teacher runs (one heuristic, or the LLM, alone)
LOO_noBP / noPS / noLLM — "leave one out": all three teachers minus one
TRI_equal   — all three teachers, trusted EQUALLY (the ablation)
TRI_shadow  — all three teachers, SMART competence weighting  ← the proposed method
```
The comparisons that matter: **TRI_shadow vs S_LLM** (does combining beat the single best teacher?)
and **TRI_shadow vs TRI_equal** (is the smart weighting doing the work, or is just having more
teachers enough?).

**Results to have ready (be precise that these are early / single-seed):**
- **Finding 1 (headline so far):** *no single teacher is best at both risks.* The rule-taught run
  causes significantly **fewer collisions**; the LLM-taught run causes significantly **fewer storm
  entries** — a real, statistically-tested complementarity (p ≈ 0.02 on each).
- **Finding 2:** the LLM-taught run beats the plain baseline by about **+6 points** of success.
- **Honest caveat:** each method has **one training seed** so far, and one baseline seed was unlucky
  (dropped to ~60% success while the others were ~81%) — so training luck currently exceeds the small
  method gaps. The proposed **TRI_shadow** run is still training. This is exactly why the combined
  claim isn't final yet.

---

## 4. Key concepts you must be fluent explaining

### MAPPO, actor-critic, and CTDE (the absolute basics)
- **RL (reinforcement learning)** = an agent learns by trial and error to maximise reward. Here the
  reward rewards reaching the destination and heavily penalises collisions / storms.
- **PPO (Proximal Policy Optimization)** = a stable, popular RL algorithm; it improves the policy in
  small clipped steps so training doesn't blow up.
- **MAPPO** = **Multi-Agent PPO**. Several aircraft learn together. The design pattern is **CTDE —
  Centralised Training, Decentralised Execution**: each aircraft acts on its *own local view* (an
  actor), but during training a **central critic** sees the *global* state to judge how good things
  are. That's what lets selfish local decisions add up to good team behaviour.
- **Actions:** 13 discrete heading changes, −30° to +30°. **The action space is a number line**
  (a small turn is "close" to a slightly bigger turn) — a detail that matters for how you score the
  teachers (below).

### The isolation guarantee (your single strongest talking point)
- The real environment is **always** driven by the RL policy's own action. The teachers (and the LLM)
  only ever contribute an **extra loss term** during training — they never choose what the aircraft
  actually does.
- **Why this matters:** it means a bad teacher can, at worst, slow learning — it can never fly a plane
  into danger. Most "LLM + RL" papers don't have this; they either inject the LLM's actions into the
  training data directly, pull the policy toward the LLM with a hard constraint, or let the LLM
  override the controller at run time. Yours validates first and isolates always. **Say this early —
  it's the cleanest "why is your approach different" answer.**

### Shadow evaluation / competence weighting (the heart of the method)
- **Shadow evaluation** = before trusting a teacher, *clone the simulator and simulate a few steps
  into the future* for the policy's move and for the teacher's move, and compare the returns. The
  teacher earns trust only if its move actually leads somewhere better.
- **Competence weighting** = do that per teacher, per decision, so each teacher is trusted **where it
  is strong and ignored where it is weak**. The weight blends "did it help *this* aircraft" (65%) with
  "did it help the *whole team*" (35%) — a genuinely multi-agent concern.
- **Why not just average the teachers?** That's the `TRI_equal` ablation. Averaging trusts a bad
  teacher as much as a good one in every state; competence weighting is the hypothesis that *knowing
  where to trust whom* is what actually helps. Comparing the two isolates that.

### Knowledge distillation and behavioural cloning
- **Distillation** = train a "student" model to imitate a "teacher." Here the student is the RL policy;
  the teachers are the small imitation networks.
- **Behavioural cloning** = the plain supervised way the teacher networks are made: given a situation,
  predict the guide's action. Standard classification over the 13 turns.
- **The divergence in the loss** = by default **Jensen–Shannon divergence** between the student's and
  teacher's action *distributions* (a symmetric, bounded "how different are these two probability
  distributions" measure). Cross-entropy or a soft-KL variant are alternatives. JS is gentle and
  stable, which matters when you're pulling a policy toward several teachers at once.

### Why the teacher influence decays but never fully dies (a design decision you can defend)
- The distillation weight **starts moderate and decays** so late training is mostly pure RL — that's
  the standard idea that lets the student eventually **surpass** imperfect teachers.
- But you set a **small non-zero floor** (it decays to ~0.1, not 0), deliberately keeping a light,
  permanent teacher signal. **Trade-off to state out loud:** the floor keeps useful guidance alive,
  but because the expensive shadow evaluation only runs while the weight is above zero, a non-zero
  floor means the look-ahead runs for the *whole* training run — which is exactly why the
  three-teacher shadow run is the slowest. That's a real cost/robustness trade-off you chose knowingly.

### The turn-preview guardrail (why the LLM doesn't hallucinate)
- Before any LLM decision, the simulator **pre-computes the consequences of every legal turn** (simulate
  each candidate ~10 steps and record whether it stays safe, keeps separation, avoids weather, makes
  progress). The LLM then **selects among validated options** instead of inventing a heading from
  scratch. This turns the LLM's job from hard trajectory prediction into easy selection, which
  massively cuts hallucination-driven errors.

### The reward and the outcomes (so you can talk about "success")
- **Dense per-step reward** nudges progress toward the destination while penalising traffic and
  weather risk; **terminal reward** dominates: big bonus if everyone lands, big penalty for a
  collision or a storm entry.
- **Episode outcomes** = success (all land) / collision / weather / truncated (ran out of time). Your
  headline metrics are **success rate** and the two failure rates (collision %, weather %).

---

## 5. Be upfront about the gaps — map them to the JD honestly

If the role is about **agentic AI + operations research / optimization**, this project is a *better*
fit than pure ML work — it's literally sequential multi-agent decision-making in a safety-critical
optimisation-flavoured domain. But be precise about two honest gaps.

**Gap 1 — "agentic AI" here means multi-agent RL, not LLM tool-using agents.**
When people say "agentic AI" today they often mean an LLM that plans, calls tools, and acts in a loop.
Your agents are RL policies, and your LLM is a *teacher/advisor*, not an autonomous tool-using agent.
- **Bridge:** *"There are genuine agents here — several RL agents making sequential decisions toward a
  goal under constraints — and there's an LLM in the loop as a validated advisor. What's directly
  transferable to agentic systems is the discipline I built around it: never let the LLM act without
  validating it first, and keep a hard isolation boundary so a wrong LLM suggestion can't cause harm.
  That 'validate-then-trust, never hand it the controls' pattern is exactly what production LLM agents
  need."*

**Gap 2 — it's reinforcement learning, not classical OR (LP / MIP / exact optimization).**
Conflict-free routing is a classic OR problem, but you solve it with RL, not mathematical programming.
- **Bridge:** Acknowledge it plainly: *"I solved this with reinforcement learning, not linear or
  integer programming. The honest conceptual link is that RL is sequential decision **optimisation** —
  maximising expected return under constraints — and conflict resolution is an OR domain. But if the
  role centres on LP/MIP scheduling, that's an area I'd be ramping on rather than claiming."* (Then
  point to any actual OR study you have.)

**Other honest limitations, if pushed on rigor:**
- **Single training seed per method so far**, and RL is high-variance — so present the small gaps as
  *directional*, not conclusive, and note the proposed headline run is still training. Saying this
  before they push is a credibility win.
- **Custom simulator, no community benchmark** — results are internally consistent but not directly
  comparable to a published leaderboard.
- The **three-teacher shadow run is expensive** (weeks of compute) because the look-ahead runs the
  whole time — a real scalability caveat you can discuss.

---

## 6. How to tell the story when they say "walk me through your research"

Use this shape — context → what you built → *the* clever decision → the result. Keep it under 2
minutes, then let them steer.

1. **Context (10s):** "I built a multi-agent reinforcement-learning controller for air traffic control
   — several aircraft in a sector each choosing headings to reach their destination without colliding
   or flying into a storm. Mistakes there are a safety issue, so how you *teach* the controller matters."
2. **What you built (40s):** "Rather than trust one expert teacher, I distil the controller from
   several complementary teachers at once — an LLM plus two cheap rule-based heuristics. The key idea
   is competence weighting: at every decision I simulate a few steps ahead to see which teacher is
   actually right in that situation, and trust that one more. And there's a hard safety rule — the RL
   policy always flies the plane; the teachers only shape its training loss, never the real environment."
3. **The clever decision (40s):** "The central design choice was *validate before you trust*. A naive
   approach either averages the teachers or lets the LLM act directly. I do neither — I look ahead a
   few steps and weight each teacher by how much it beats the current policy right there. That let me
   test a specific hypothesis: no single teacher is best everywhere."
4. **Result / insight (20s):** "And that held: the rule-based teacher avoids collisions better, the LLM
   teacher avoids weather better — a statistically real complementarity. The single-LLM-taught policy
   also beat the plain baseline. The proposed combined run is still training, so I'm careful there, but
   the 'no single teacher wins on everything' finding is solid."
5. **Invite the dig:** "Happy to go deeper on the competence weighting, the isolation guarantee, or how
   I turned the LLM into a cheap teacher network — whichever's more useful."

---

## 7. Likely interview questions + tight answers

**Q: In one line, what did *you* do?**
I built a multi-agent RL air-traffic controller and a multi-teacher distillation framework that
distils it from an LLM plus two heuristics, trusting each teacher only where a short look-ahead shows
it beats the current policy — while the RL policy alone ever controls the environment.

**Q: What is MAPPO / why multi-agent?**
Multi-Agent PPO. Several aircraft learn together with centralised-training / decentralised-execution:
each acts on its own local view, but a central critic sees the global state during training, so local
decisions add up to safe team behaviour.

**Q: Why not just let the LLM fly the aircraft?**
Safety and reliability. A local LLM hallucinates and is slow. So the LLM is a *validated advisor*: its
suggestions are checked by look-ahead simulation and only used to shape training — the RL policy always
executes. A wrong LLM suggestion can at worst slow learning, never cause a collision.

**Q: What is "competence weighting" and why does it matter?**
Per decision, per teacher, I clone the simulator and roll a few steps ahead to measure how much that
teacher's move beats the current policy, and weight it by that. So each teacher is trusted where it's
strong and ignored where it's weak — instead of blindly averaging them.

**Q: Why not just average all the teachers?**
That's my ablation (`TRI_equal`). Averaging trusts a bad teacher as much as a good one everywhere.
Comparing smart weighting against equal weighting isolates whether *knowing where to trust whom* is
what actually helps.

**Q: What's your main finding?**
No single teacher is best everywhere: the rule-based teacher significantly reduces collisions, the LLM
teacher significantly reduces storm entries. That complementarity is the whole motivation for combining
them — and it's statistically tested, not just eyeballed.

**Q: How did you turn a slow LLM into something usable in an RL loop?**
Offline, I recorded the LLM's decisions once and trained a small network to imitate them (behavioural
cloning). During RL training I query that cheap network instead of the LLM — no live LLM calls in the
training loop.

**Q: Why does the teacher influence decay, and why keep a floor?**
Decaying it lets the student eventually surpass imperfect teachers (late training is mostly pure RL).
I keep a small non-zero floor to retain a light permanent signal — knowing the trade-off that it keeps
the expensive look-ahead running the whole time, which is why the three-teacher run is slow.

**Q: How do you know the RL policy is really the one in control?**
It's an invariant I check in an audit log: the action sent to the environment always equals the RL
policy's action. The teachers only ever add a loss term.

**Q: How is this different from the papers you compare against?**
Two ways. First, the isolation guarantee — I validate and isolate, they inject/constrain/override.
Second, the multi-teacher angle — prior work distils from a *single* LLM teacher assuming the LLM is
best; my premise is that no single teacher is best, so I combine complementary teachers by competence.

**Q: What are the weaknesses / what would you improve?**
Single training seed per method so far, so I treat the small gaps as directional; the proposed combined
run is still training; and it's a custom simulator with no public benchmark. I'd run multiple seeds,
finish the combined run, and make the look-ahead cheaper so the three-teacher version scales.

**Q (trap): explain a detail you didn't build.**
"My focus was [the distillation framework / competence weighting]. At a high level that piece does
[one true sentence]. I can go deep on the parts I built." Confident, honest, pivots to strength.

---

## 8. Quick-glance cheat sheet (skim this right before the interview)

| Question | One-line answer |
|---|---|
| What is it | A multi-agent RL air-traffic controller, distilled from several complementary teachers at once |
| The domain | Aircraft in a sector choosing headings to land safely without collisions or storm entries |
| Your role | Built the RL controller + the multi-teacher distillation framework; ran and evaluated the experiments |
| Base RL | MAPPO — multi-agent PPO, centralised critic / decentralised actors (CTDE) |
| Teachers | One LLM teacher + two cheap heuristics, each a small network cloned from a "guide" |
| Key idea | Competence weighting: look a few steps ahead, trust each teacher where it beats the policy |
| Safety rule | Isolation — the RL policy always flies; teachers only shape the training loss |
| Distillation loss | PPO + decaying, competence-weighted JS divergence to each trusted teacher |
| The hypothesis | "No single teacher is best everywhere" |
| Finding 1 | Rule teacher → fewer collisions; LLM teacher → fewer storms (statistically real complementarity) |
| Finding 2 | Single-LLM-taught policy beats the plain baseline by ~+6 points |
| Honest caveat | One seed per method so far; proposed combined run still training; custom sim |
| Ablation | `TRI_equal` (average the teachers) vs `TRI_shadow` (smart weighting) — isolates the weighting |
| Why not LLM flies | Slow + hallucinates; used as a *validated advisor*, never the controller |
| LLM made cheap | Recorded its decisions once, cloned into a small network — no live LLM calls in training |
| vs prior work | They distil from one LLM assuming it's best; I combine complementary teachers by competence + isolate |
| Gaps vs JD | Multi-agent RL (not LLM tool-agents); RL optimisation (not LP/MIP) — bridge honestly |
