# LLM-Enhanced DRL for Air-Traffic Control ✈

## Overview
Integrate a Large Language Model (LLM) as a “teacher” to boost a PPO-based agent’s decision-making on standard conflict scenarios. The LLM provides high-level advice that the RL policy learns to interpret and execute.

## Core Workflow
1. **State + LLM prompt**  
   - At each conflict step, send current state summary and separation rules to the LLM.  
2. **High-level suggestion**  
   - LLM returns a maneuver hint (e.g. “turn +10°”).  
3. **Fusion & action**  
   - Encode the hint, combine it with the state, and let the PPO actor choose the precise control.  
4. **Learning**  
   - Apply standard PPO updates plus a small distillation penalty so the policy aligns with the LLM’s advice.

## Next Steps
- Once the baseline LLM+RL integration is stable, extend to rare “long-tail” situations (storms, emergencies, sensor loss).
