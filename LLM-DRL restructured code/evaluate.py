"""Evaluation script rebuilt from RL - LLM (Complete Code Pipeline).ipynb."""

import os
from pathlib import Path
from typing import List

import numpy as np
import torch

from configs import EVAL_BEST_MODEL_PATH, EVAL_OUTPUT_DIR, START_LLM_AT_STEP
from rl_llm.env import StructuredEnv
from rl_llm.ppo import PPO, warmup_obs_norm, device


def _ensure_clean_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    for png in Path(path).glob("image_*.png"):
        try:
            png.unlink()
        except OSError:
            pass


def _save_gif(frames_dir: str, out_gif_path: str, fps: float = 5.0) -> None:
    import imageio.v2 as imageio

    frames = sorted(Path(frames_dir).glob("image_*.png"))
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}. Did env.render() run?")
    images = [imageio.imread(str(f)) for f in frames]
    imageio.mimsave(out_gif_path, images, duration=1.0 / max(1e-6, fps))


def rollout_to_gif(env: StructuredEnv, agent: PPO, ep_index: int = 1) -> float:
    frames_dir = os.path.join(EVAL_OUTPUT_DIR, f"ep{ep_index:03d}")
    _ensure_clean_dir(frames_dir)

    obs, _ = env.reset()
    ep_ret = 0.0
    done = False

    env.render(show=False, folder=frames_dir + "/")

    while not done:
        st = torch.tensor(obs, dtype=torch.float32, device=device)
        norm_st = agent._normalize_obs_eval(st)
        with torch.no_grad():
            feats = agent.policy_old.net(norm_st)
            probs = agent.policy_old.actor(feats)
            action = torch.argmax(probs, dim=-1).item()

        obs, reward, term, trunc, _ = env.step(action)
        ep_ret += reward
        done = term or trunc

        env.render(show=False, folder=frames_dir + "/")

        if env.n_step > env.MAX_STEP + 5:
            break

    gif_path = os.path.join(EVAL_OUTPUT_DIR, f"ep{ep_index:03d}.gif")
    _save_gif(frames_dir, gif_path, fps=5.0)
    print(f"[EVAL] Saved GIF → {gif_path} | Return={ep_ret:.3f}")
    return float(ep_ret)


def main() -> None:
    assert os.path.exists(
        EVAL_BEST_MODEL_PATH
    ), f"Not found: {EVAL_BEST_MODEL_PATH}"
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    env = StructuredEnv(start_llm_at_step=START_LLM_AT_STEP)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(
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
    )

    agent.load(EVAL_BEST_MODEL_PATH)
    warmup_obs_norm(agent, env, steps=5000)

    returns: List[float] = []
    for ep in range(1, 2):
        ret = rollout_to_gif(env, agent, ep)
        returns.append(ret)

    mean_return = float(np.mean(returns)) if returns else 0.0
    print(f"[EVAL] Mean return over {len(returns)} ep(s): {mean_return:.3f}")


if __name__ == "__main__":
    main()
