"""PPO components derived from RL - LLM (Complete Code Pipeline).ipynb."""
# We use glob to search for saved model files when resuming training.
import glob
# Math provides constants and helpers like exp and sqrt.
import math
# OS functions help us work with folders and file paths.
import os
# Random allows us to sync Python's own random number generator for checkpoints.
import random
# Regular expressions help with parsing numbers from filenames.
import re
# Time gives access to timestamps and sleep for safe file writes.
import time
# Optional lets us note when a value may be missing.
from typing import Optional

# NumPy lets us handle arrays for checkpoint data.
import numpy as np
# Torch is our main deep learning framework.
import torch
# Torch.nn provides neural network building blocks.
import torch.nn as nn
# Categorical handles discrete action distributions for policy sampling.
from torch.distributions import Categorical

# We detect whether to run on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# We fix the random seed for reproducible runs.
torch.manual_seed(0)
if device.type == 'cuda':
    # We also fix CUDA level randomness and enable cudnn tuning.
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

def _atomic_torch_save(obj, path: str):
    # We find the folder where the final file should live.
    parent = os.path.dirname(os.path.abspath(path))
    # We make sure that folder exists before saving.
    os.makedirs(parent, exist_ok=True)  # Ensure folder exists

    # We build a unique temporary name so multiple saves do not clash.
    tmp = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
    try:
        # We first save to the temporary file.
        torch.save(obj, tmp)
        # We then atomically move the file into place.
        os.replace(tmp, path)
    except FileNotFoundError:
        # If the folder disappeared we wait a bit and try to recover.
        time.sleep(0.05)
        if os.path.exists(tmp):
            # If the temp file still exists we move it again.
            os.replace(tmp, path)
        else:
            # Otherwise we raise a clear error for the caller.
            raise FileNotFoundError(f"Failed to save checkpoint to {path}: temporary file {tmp} not found")
    except Exception as e:
        # Any other error is wrapped with path information.
        raise Exception(f"Failed to save checkpoint to {path}: {e}")


def save_training_checkpoint(path: str, agent, time_step: int, episode: int):
    # We gather all parts of the agent that are needed to restart training.
    ckpt = {
        # models
        "policy": agent.policy.state_dict(),
        "policy_old": agent.policy_old.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        # obs normalizer
        "obs_mean": agent.obs_rms.mean.detach().cpu(),
        "obs_var":  agent.obs_rms.var.detach().cpu(),
        "obs_count": float(agent.obs_rms.count),
        # counters
        "time_step": int(time_step),
        "episode": int(episode),
        # RNG states (optional but useful)
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "py_rng":    random.getstate(),
        "cuda_rng":  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        # (optional) hyperparams snapshot if you want
        "hparams": {
            "gamma": agent.gamma,
            "K_epochs": agent.K_epochs,
            "eps_clip": agent.eps_clip,
            "mb_size": agent.mb_size,
            "gae_lambda": agent.gae_lambda,
        },
    }
    # We write the checkpoint safely to disk.
    _atomic_torch_save(ckpt, path)

def load_training_checkpoint(path: str, agent, device):
    # We read the checkpoint data onto the requested device.
    ckpt = torch.load(path, map_location=device)

    # We restore policy networks and optimizer state.
    agent.policy.load_state_dict(ckpt["policy"])
    agent.policy_old.load_state_dict(ckpt["policy_old"])
    agent.optimizer.load_state_dict(ckpt["optimizer"])

    # We bring back the running observation statistics.
    agent.obs_rms.mean  = ckpt["obs_mean"].to(device)
    agent.obs_rms.var   = ckpt["obs_var"].to(device)
    agent.obs_rms.count = ckpt["obs_count"]

    # restore RNG states (optional)
    torch.set_rng_state(ckpt["torch_rng"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    random.setstate(ckpt["py_rng"])

    # We return the progress counters so training can resume.
    return int(ckpt["time_step"]), int(ckpt["episode"])


# 0. Utilities: Running Mean/Std for Obs Normalization
class RunningMeanStd:
    """Track running mean and variance of observations."""
    def __init__(self, shape, eps: float = 1e-4):
        # We start with zero mean for each observation dimension.
        self.mean  = torch.zeros(shape, dtype=torch.float32, device=device)
        # We start with ones for variance to avoid division by zero.
        self.var   = torch.ones(shape,  dtype=torch.float32, device=device)
        # We keep track of how many samples we have seen.
        self.count = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        # x: (..., shape)
        # We ensure input has a batch dimension.
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # We compute batch statistics for the new data.
        batch_mean = x.mean(dim=0)
        batch_var  = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        # We measure the difference between batch mean and stored mean.
        delta = batch_mean - self.mean
        # We combine counts to compute the new totals.
        tot_count = self.count + batch_count
        # We update the running mean using a weighted average.
        new_mean = self.mean + delta * (batch_count / tot_count)
        # We compute total squared differences for old data.
        m_a = self.var * self.count
        # We compute total squared differences for batch data.
        m_b = batch_var * batch_count
        # We fuse both sums and adjust for the mean shift.
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        # We compute the final variance.
        new_var = M2 / tot_count
        # We commit the new statistics to the object.
        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # We subtract the running mean and scale by the running std.
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)

# 1. Rollout Buffer
class RolloutBuffer:
    def __init__(self):
        self.states       = []  # list of torch tensors (normalized obs) on device
        self.actions      = []  # list of torch tensors on device
        self.logprobs     = []  # list of torch tensors on device
        self.state_values = []  # list of torch tensors on device
        self.rewards      = []  # list of floats
        self.is_terminals = []  # list of bools
        # NEW: for hybrid path with LLM intent
        self.intents          = []  # torch float32, shape [4]
        self.active_masks     = []  # torch float32 scalar {0,1}
        self.llm_target_bin   = []  # python int (0..12) or -1 if N/A
        self.llm_target_deg   = []  # python float (snapped deg) or 0.0 if N/A


    def clear(self):
        # Reset all buffer lists by reinitializing
        self.__init__()

# 2. Hybrid Actor-Critic Network (attention-ready, backward compatible)
class HybridActorCritic(nn.Module):
    """
    Backward-compatible drop-in for ActorCritic:
      - Keeps .net (state-only MLP), .actor, .critic so existing calls keep working.
      - Adds .intent_proj and .attn + a fuse() path we will enable in Step 2.
      - Right now, PPO continues to use the state-only path (identical behavior).
    """
    def __init__(self, state_dim, action_dim, d_model: int = 64, n_heads: int = 1):
        # We initialize the PyTorch base class.
        super().__init__()
        # --- State-only path (kept for backward-compat) ---
        # We build a tiny MLP that turns raw states into features.
        self.net = nn.Sequential(
            nn.Linear(state_dim, d_model), nn.Tanh(),
            nn.Linear(d_model, d_model),   nn.Tanh()
        )
        # We map features into action probabilities with a softmax layer.
        self.actor  = nn.Sequential(nn.Linear(d_model, action_dim), nn.Softmax(dim=-1))
        # We predict the state value for critic updates.
        self.critic = nn.Linear(d_model, 1)

        # --- New components for fusion (will be used in Step 2) ---
        # We project the 4-value LLM intent vector into the model space.
        self.intent_proj = nn.Linear(4, d_model)
        # We prepare attention so the state can mix with the intent token.
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Cache to support "hold" reuse later (Step 3)
        # We remember the last fused vector if the LLM wants to reuse it.
        self._last_fused: Optional[torch.Tensor] = None

    # ---- Backward-compatible API used by your current PPO ----
    def act(self, state: torch.Tensor):
        """
        Current training/eval uses state-only path to keep behavior identical.
        We'll switch this to fused path in Step 2.
        """
        # We turn the state into hidden features.
        features     = self.net(state)
        # We get action probabilities from the actor head.
        action_probs = self.actor(features)
        # We build a categorical distribution over actions.
        dist         = Categorical(action_probs)

        # We sample one discrete action from the distribution.
        action      = dist.sample()
        # We compute the log probability for PPO updates.
        action_logp = dist.log_prob(action)
        # We ask the critic for the value estimate.
        value       = self.critic(features)
        # We detach tensors to store them without gradients.
        return action.detach(), action_logp.detach(), value.detach()

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Batch evaluation on the state-only path (unchanged behavior).
        """
        # We process the batch of states through the shared network.
        features        = self.net(states)
        # We get batch action probabilities for each state.
        action_probs    = self.actor(features)
        # We create a categorical distribution for the batch.
        dist            = Categorical(action_probs)

        # We compute log probabilities of the provided actions.
        action_logprobs = dist.log_prob(actions)
        # We track the entropy for the PPO loss.
        dist_entropy    = dist.entropy()
        # We compute value predictions and remove the last dimension.
        state_values    = self.critic(features).squeeze(-1)
        return action_logprobs, state_values, dist_entropy

    # ---- New fusion utilities (will be used starting Step 2) ----
    def fuse(self, state_emb: torch.Tensor,
             intent_emb: Optional[torch.Tensor] = None,
             active_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Build a two-token sequence [state_token, intent_token] and run MHSA.
        If no intent or active_mask==0, bypass and return state token.
        Shapes:
          state_emb:  (B, d_model)  from a state MLP/projection
          intent_emb: (B, 4)        raw intent (norm_turn, norm_hold, is_wait, is_exec)
          active_mask: scalar or (B,) 1=use attention, 0=bypass
        Returns:
          fused embedding z: (B, d_model)
        """
        # State token (project to d_model if needed)
        # We reuse the stored network layer if the input size differs.
        h_s = self.net[0](state_emb) if state_emb.shape[-1] != self.net[0].in_features else state_emb
        if state_emb.shape[-1] == self.net[0].in_features:
            # Run the same two-layer MLP to stay consistent with current features
            h_s = self.net(state_emb)  # (B, d_model)

        # Bypass unless explicitly active
        # We only enter attention if intent info exists and the mask allows it.
        use_attn = (intent_emb is not None) and (active_mask is not None) and (
            (torch.is_tensor(active_mask) and torch.any(active_mask != 0)) or
            (not torch.is_tensor(active_mask) and active_mask != 0)
        )
        if not use_attn:
            # Without intent we fall back to plain state features.
            return h_s

        # Intent token
        # We push the four intent numbers through a tanh projection.
        h_i = torch.tanh(self.intent_proj(intent_emb))  # (B, d_model)

        # Tokens: (B, 2, d_model)
        # We build a two-token sequence for multi-head attention.
        tokens = torch.stack([h_s, h_i], dim=1)
        # We run self-attention so the state can attend to the intent token.
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        # We average the output tokens to get a single fused vector.
        z = attn_out.mean(dim=1)  # simple pooling over the 2 tokens
        return z

    def act_from_fused(self, z: torch.Tensor):
        # We turn the fused features into action probabilities.
        action_probs = self.actor(z)
        # We create a categorical distribution from those probabilities.
        dist         = Categorical(action_probs)
        # We sample one discrete action for the environment.
        action      = dist.sample()
        # We compute the log probability for PPO's ratio term.
        action_logp = dist.log_prob(action)
        # We predict the value using the critic head.
        value       = self.critic(z)
        # We return detached copies for safe storage.
        return action.detach(), action_logp.detach(), value.detach()

    def evaluate_from_fused(self, z: torch.Tensor, actions: torch.Tensor):
        # We compute probabilities for the provided fused batch.
        action_probs    = self.actor(z)
        # We build the categorical distribution that matches those probabilities.
        dist            = Categorical(action_probs)
        # We measure how likely the chosen actions were.
        action_logprobs = dist.log_prob(actions)
        # We record the entropy term to encourage exploration.
        dist_entropy    = dist.entropy()
        # We compute critic values and squeeze to a vector.
        state_values    = self.critic(z).squeeze(-1)
        return action_logprobs, state_values, dist_entropy


# 3. PPO Agent Wrapper
class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=2e-4,
        lr_critic=8e-4,
        gamma=0.99,
        K_epochs=5,
        eps_clip=0.2,
        mb_size=256,
        gae_lambda=0.95,
        normalize_reward=True,
    ):
        # We remember key hyperparameters that guide learning.
        self.gamma      = gamma
        self.eps_clip   = eps_clip
        self.K_epochs   = K_epochs
        self.mb_size    = mb_size
        self.gae_lambda = gae_lambda
        # This switch controls whether we normalize rewards per batch.
        self.normalize_reward = normalize_reward

        # We store past interactions until we update the policy.
        self.buffer = RolloutBuffer()

        # Observation normalizer
        # We track running mean and variance of observations.
        self.obs_rms = RunningMeanStd(shape=state_dim)

        # Initialize current & old policy networks on device
        # self.policy     = ActorCritic(state_dim, action_dim).to(device)
        # self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        # We build the active policy network that learns from new data.
        self.policy     = HybridActorCritic(state_dim, action_dim).to(device)
        # We keep a separate frozen copy for sampling actions.
        self.policy_old = HybridActorCritic(state_dim, action_dim).to(device)
        # We sync both models so they start identical.
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizer for actor and critic parameters
        # We train actor, critic, and fusion parts with Adam.
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.net.parameters(),        'lr': lr_actor},   # state MLP
            {'params': self.policy.actor.parameters(),      'lr': lr_actor},
            {'params': self.policy.critic.parameters(),     'lr': lr_critic},
            {'params': self.policy.intent_proj.parameters(),'lr': lr_actor},   # new
            {'params': self.policy.attn.parameters(),       'lr': lr_actor},   # new
        ])
        # We reuse MSELoss for value function training.
        self.MseLoss = nn.MSELoss()
        # --- NEW: alignment knobs (defaults keep behavior almost identical) ---
        # These flags control optional losses that align with LLM advice.
        self.use_kl_to_llm = True
        self.use_mse_to_llm = False
        # These weights decide how strong each alignment loss should be.
        self.lambda_kl = 0.10
        self.lambda_mse = 0.00

        # Bin → degrees map for expected-heading MSE
        # We map discrete action bins to actual degrees for debugging losses.
        self._deg_by_idx = torch.tensor(
            [0, 5,10,15,20,25,30,  -5,-10,-15,-20,-25,-30],
            dtype=torch.float32, device=device
        )
        # Cache for "hold_steps" fused embedding reuse (episode-scoped)
        # We reuse the fused vector when the LLM says to hold a command.
        self._fused_cache = None

    def _normalize_obs_train(self, state_tensor: torch.Tensor) -> torch.Tensor:
        # Update running stats with current state, then normalize
        self.obs_rms.update(state_tensor)
        return self.obs_rms.normalize(state_tensor)

    def _normalize_obs_eval(self, state_tensor: torch.Tensor) -> torch.Tensor:
        # Use current running stats, do NOT update during evaluation
        return self.obs_rms.normalize(state_tensor)

        # --- helpers for LLM-intent (unused = zeros) ---
    def _prepare_intent_tensors(self, intent_embed, phase_active: int):
        """
        Returns (intent_t [4], mask_t [1]) on device.
        If intent_embed is None, returns zeros and mask=0.
        """
        if intent_embed is None:
            # No LLM tip is available, so we fill zeros.
            intent_t = torch.zeros(4, dtype=torch.float32, device=device)
            mask_t   = torch.tensor([0.0], dtype=torch.float32, device=device)
        else:
            # expect a length-4 iterable: [norm_turn, norm_hold, is_wait, is_exec]
            # We convert the advice into a tensor on the correct device.
            intent_t = torch.tensor(intent_embed, dtype=torch.float32, device=device)
            # We flag whether the advice should actually be used.
            mask_t   = torch.tensor([1.0 if phase_active else 0.0], dtype=torch.float32, device=device)
        return intent_t, mask_t

    def _snap_deg_to_bin_index(self, deg: float) -> int:
        """
        Map degrees in {0,±5,±10,...,±30} to action index 0..12.
        Returns -1 if deg is invalid (not multiple of 5 or |deg|>30).
        Bins: 0..6 => +0,+5,+10,+15,+20,+25,+30; 7..12 => -5,-10,-15,-20,-25,-30
        """
        # We round the degree to the nearest integer for stability.
        d = int(round(float(deg)))
        if d % 5 != 0 or abs(d) > 30:
            # If the angle is outside the allowed set we return -1.
            return -1
        # We map positive and zero values to the first half of bins.
        return d // 5 if d >= 0 else 6 + (abs(d) // 5)

    def select_action_hybrid(self, state, intent_embed=None, phase_active: int = 0,
                             target_deg: float = None, reuse_cached: bool = False):
        """
        Hybrid selection on policy_old using fused embedding.
        If reuse_cached is True and a cache exists, we reuse the last fused z (ignores new obs).
        """
        # 1) normalize obs (train-time)
        # We convert the incoming state into a float tensor on the device.
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        # We normalize the state with running statistics.
        norm_state   = self._normalize_obs_train(state_tensor)  # [state_dim]

        # 2) prepare intent + mask
        # We turn optional LLM advice into tensors the network understands.
        intent_t, mask_t = self._prepare_intent_tensors(intent_embed, phase_active)

        # 3) decide fused embedding (reuse or recompute)
        z: torch.Tensor
        if reuse_cached and (self._fused_cache is not None):
            # We reuse the cached fused vector when the LLM says to hold the action.
            z = self._fused_cache  # (d_model,)
        else:
            # We build a new fused vector from the normalized state and intent.
            z = self.policy_old.fuse(
                norm_state.unsqueeze(0),     # (1, state_dim)
                intent_t.unsqueeze(0),       # (1, 4)
                mask_t                       # (1,)
            ).squeeze(0)                      # (d_model,)
            # Only keep cache while an LLM phase is active (mask=1)
            self._fused_cache = z.detach() if (mask_t.item() > 0.5) else None

        # 4) sample from fused embedding (policy_old heads)
        # We draw an action and supporting numbers from the old policy.
        action, logp, value = self.policy_old.act_from_fused(z)

        # 5) store rollout (keep old fields; add new)
        # We push the state and outputs into the rollout buffer.
        self.buffer.states.append(norm_state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logp)
        self.buffer.state_values.append(value)

        # We also store the intent information for later analysis.
        self.buffer.intents.append(intent_t)         # [4]
        self.buffer.active_masks.append(mask_t[0])   # scalar

        if target_deg is None:
            # If no heading target was given we mark the slot as unused.
            self.buffer.llm_target_bin.append(-1)
            self.buffer.llm_target_deg.append(0.0)
        else:
            # We convert the target degrees into a discrete bin.
            idx = self._snap_deg_to_bin_index(target_deg)
            self.buffer.llm_target_bin.append(idx)
            self.buffer.llm_target_deg.append(float(int(round(target_deg))))

        # We return the plain integer action for the environment.
        return action.cpu().item()

    def clear_fused_cache(self):
        # We forget any stored fused vector when a new episode starts.
        self._fused_cache = None

    def select_action(self, state):
        # Convert raw state to tensor on device and normalize (train-time)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        norm_state   = self._normalize_obs_train(state_tensor)
        # We sample using the state-only path for backward compatibility.
        action, logp, value = self.policy_old.act(norm_state)

        # Store tensors directly (already on device)
        # We push all pieces into the buffer for PPO updates.
        self.buffer.states.append(norm_state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logp)
        self.buffer.state_values.append(value)

        # We return the chosen discrete action index.
        return action.cpu().item()

    def update(self):
        """ PPO update on fused embeddings with optional LLM alignment (phase-gated). """
        # ---- Stack rollout tensors (on device) ----
        # We gather all stored states into one tensor.
        states      = torch.stack(self.buffer.states,       dim=0)  # (T, obs_dim) normalized
        # We stack the actions taken during the rollout.
        actions     = torch.stack(self.buffer.actions,      dim=0)  # (T,)
        # We save the old log probabilities from sampling time.
        old_logprob = torch.stack(self.buffer.logprobs,     dim=0)  # (T,)
        # We stack the value estimates that were recorded.
        values      = torch.stack(self.buffer.state_values, dim=0).squeeze(-1)  # (T,)
        # We build a tensor of raw rewards for this rollout.
        rewards_raw = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=device)
        # We mark where episodes ended so GAE can reset.
        dones       = torch.tensor(self.buffer.is_terminals, dtype=torch.float32, device=device)

        # ---- LLM fields (backward-compatible) ----
        if len(self.buffer.intents) == 0:
            # Pure-PPO rollout: fabricate zeros so fuse() bypasses attention
            # Without LLM data we create zero placeholders that skip fusion.
            B = states.size(0)
            intents    = torch.zeros(B, 4, dtype=torch.float32, device=device)
            masks      = torch.zeros(B,    dtype=torch.float32, device=device)
            target_idx = torch.full((B,), -1, dtype=torch.long, device=device)
            target_deg = torch.zeros(B,    dtype=torch.float32, device=device)
        else:
            # We gather all stored intents, masks, and targets for training.
            intents    = torch.stack(self.buffer.intents,      dim=0)  # (T, 4)
            masks      = torch.stack(self.buffer.active_masks, dim=0)  # (T,)
            target_idx = torch.tensor(self.buffer.llm_target_bin, dtype=torch.long,   device=device)
            target_deg = torch.tensor(self.buffer.llm_target_deg, dtype=torch.float32, device=device)


        # ---- Reward normalization (per-batch) ----
        rewards = rewards_raw
        if self.normalize_reward:
            # We center and scale rewards to stabilize learning.
            r_mean = rewards_raw.mean()
            r_std  = rewards_raw.std(unbiased=False)
            rewards = (rewards_raw - r_mean) / (r_std + 1e-8)

        # ---- GAE(λ) advantages using (normalized) rewards ----
        # We compute advantage estimates with Generalized Advantage Estimation.
        T = rewards.size(0)
        advantages = torch.zeros(T, dtype=torch.float32, device=device)
        last_gae = 0.0
        values_ext = torch.cat([values, torch.zeros(1, device=device)], dim=0)
        for t in reversed(range(T)):
            # We look one step ahead while respecting episode boundaries.
            next_value = values_ext[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae
        # We compute target returns for the critic.
        returns = advantages + values
        # We normalize advantages to zero mean and unit variance.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- Mini-batch SGD over K epochs on FUSED embeddings ----
        # We iterate over shuffled batches for multiple epochs.
        batch_size = states.size(0)
        mb_size = min(self.mb_size, batch_size)
        for _ in range(self.K_epochs):
            # We create a random order of indices for the epoch.
            idx = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                mb_idx = idx[start:start + mb_size]

                # We slice out mini-batch tensors from the rollout.
                mb_states  = states[mb_idx]        # (B, obs_dim)
                mb_intents = intents[mb_idx]       # (B, 4)
                mb_masks   = masks[mb_idx]         # (B,)
                mb_actions = actions[mb_idx]       # (B,)
                mb_old_lp  = old_logprob[mb_idx]   # (B,)
                mb_adv     = advantages[mb_idx]    # (B,)
                mb_ret     = returns[mb_idx]       # (B,)
                mb_tgt_idx = target_idx[mb_idx]    # (B,)
                mb_tgt_deg = target_deg[mb_idx]    # (B,)

                # -- FUSE with current policy --
                # We fuse state and intent using the up-to-date policy.
                z = self.policy.fuse(mb_states, mb_intents, mb_masks)  # (B, d_model)

                # -- Evaluate taken actions on current policy --
                # We compute log probabilities and values under the new policy.
                logprobs, state_vals, entropy = self.policy.evaluate_from_fused(z, mb_actions)
                # We build the PPO ratio between new and old probabilities.
                ratios = torch.exp(logprobs - mb_old_lp.detach())
                # We form the unclipped surrogate loss.
                surr1  = ratios * mb_adv
                # We build the clipped version to keep updates stable.
                surr2  = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv

                # Policy loss encourages higher advantage actions.
                policy_loss   = -torch.min(surr1, surr2).mean()
                # Value loss is a mean squared error to the returns.
                value_loss    = self.MseLoss(state_vals, mb_ret)
                # Entropy bonus encourages exploration; negative sign for minimization.
                entropy_bonus = -0.01 * entropy.mean()

                # -- OPTIONAL: LLM alignment during phases only (mask>0 & valid target idx) --
                # We start with no alignment penalty.
                align_loss = torch.zeros((), dtype=torch.float32, device=device)

                if self.use_kl_to_llm or self.use_mse_to_llm:
                    # We compute full action probabilities for alignment checks.
                    probs_all = self.policy.actor(z)                 # (B, 13)
                    log_all   = torch.log(probs_all + 1e-8)

                    # We select samples where the LLM gave actionable advice.
                    valid = (mb_masks > 0.5) & (mb_tgt_idx >= 0)     # boolean mask
                    if valid.any():
                        # We find the indices of those valid samples.
                        v_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)

                        if self.use_kl_to_llm and self.lambda_kl > 0.0:
                            # CE to one-hot target (equiv. KL to delta target)
                            # We encourage the policy to match the LLM suggested bin.
                            chosen = log_all[v_idx, mb_tgt_idx[v_idx]]
                            ce_loss = -chosen.mean()
                            align_loss = align_loss + self.lambda_kl * ce_loss

                        if self.use_mse_to_llm and self.lambda_mse > 0.0:
                            # We compute expected heading and compare to the target degrees.
                            exp_deg = (probs_all[v_idx] * self._deg_by_idx.unsqueeze(0)).sum(dim=-1)
                            mse = ((exp_deg - mb_tgt_deg[v_idx]) ** 2).mean()
                            align_loss = align_loss + self.lambda_mse * mse

                # Final PPO loss mixes policy, value, entropy, and alignment terms.
                loss = policy_loss + 0.5 * value_loss + entropy_bonus + align_loss

                # We perform one optimization step on the current mini-batch.
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        # ---- Sync old policy and clear buffer ----
        # After updates we copy weights into the old policy.
        self.policy_old.load_state_dict(self.policy.state_dict())
        # We reset the buffer ready for the next rollout.
        self.buffer.clear()


    def save(self, path):
        # We store the old policy weights because they are used for acting.
        torch.save(self.policy_old.state_dict(), path)

    def load(self, path):
        # We read the saved weights onto the current device.
        state_dict = torch.load(path, map_location=device)
        # We load the weights into both policy copies to keep them in sync.
        self.policy_old.load_state_dict(state_dict)
        self.policy.load_state_dict(state_dict)

# 4. Evaluation Function
def evaluate_policy(agent, env, n_episodes=10):
    """Deterministic evaluation (argmax over action probs) for smoother curves.
    Uses observation normalization stats but does not update them.
    """
    # We store the return from each evaluation episode.
    returns = []
    for _ in range(n_episodes):
        # We reset the environment before each evaluation run.
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            # We normalize observations the same way as during training.
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            norm_state   = agent._normalize_obs_eval(state_tensor)
            with torch.no_grad():
                # We forward pass without gradients to get action scores.
                features = agent.policy_old.net(norm_state)
                probs    = agent.policy_old.actor(features)
                # We pick the most likely action instead of sampling.
                action   = torch.argmax(probs, dim=-1).item()
            # We step the environment using the chosen action.
            obs, reward, term, trunc, _ = env.step(action)
            # We stop if the episode ended for any reason.
            done = term or trunc
            # We accumulate total reward for this episode.
            ep_ret += reward
        # We record the final return for later averaging.
        returns.append(ep_ret)
    # We return the mean and spread of the evaluation returns.
    return np.mean(returns), np.std(returns)

def find_latest_weights(log_dir: str, best_model_path: str):
    """
    Find and return the newest weights-only checkpoint.

    Returns:
        (path, step):
            path: absolute path to 'checkpoint_*.pth' with largest step, or fall back to best_model.pth if present,
                  or None if nothing found.
            step: int step parsed from filename for checkpoint_*.pth; None if falling back to best_model.pth.

    Why:
        We are skipping 'last.ckpt' entirely to avoid unpickling issues. This loader only touches weights-only files.
    """
    # We look for checkpoint files that fit the naming pattern.
    pattern = os.path.join(log_dir, "checkpoint_*.pth")
    files = glob.glob(pattern)
    latest_path, latest_step = None, -1
    for f in files:
        # We try to read the step number from the filename.
        m = re.search(r"checkpoint_(\d+)\.pth$", f)
        if m:
            step = int(m.group(1))
            if step > latest_step:
                # We keep the file with the highest step number so far.
                latest_step, latest_path = step, f
    if latest_path is not None:
        return latest_path, latest_step
    if os.path.exists(best_model_path):
        # If no numbered checkpoint exists we fall back to the best model.
        return best_model_path, None
    # Nothing was found so we return a pair of None values.
    return None, None


def warmup_obs_norm(agent, env, steps: int = 20000):
    """
    Quickly rebuild observation normalization stats after resuming from weights-only.

    Args:
        agent: PPO agent instance (has 'obs_rms' and 'policy_old').
        env:   Gymnasium environment.
        steps: Number of environment steps to feed through the normalizer (no PPO updates).

    What it does:
        - Feeds observations to agent.obs_rms.update() so mean/var are sensible again.
        - Takes deterministic actions to roll the env (no learning).
        - Clears any rollout buffer leftovers at the end.

    Why:
        Weights-only resume doesn't restore obs normalizer; this prevents a noisy restart.
    """
    # We start from a fresh environment observation.
    obs, _ = env.reset()
    for _ in range(steps):
        # We build a tensor so the normalizer can update on device.
        st = torch.tensor(obs, dtype=torch.float32, device=device)
        agent.obs_rms.update(st)
        with torch.no_grad():
            # We compute a deterministic action using the old policy.
            feats  = agent.policy_old.net(agent.obs_rms.normalize(st))
            probs  = agent.policy_old.actor(feats)
            action = torch.argmax(probs, dim=-1).item()
        # We roll the environment with that action but ignore rewards.
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            # If the episode ends we reset and keep feeding the normalizer.
            obs, _ = env.reset()
    # We clear the buffer so normalizer warmup leaves no stale data.
    agent.buffer.clear()
