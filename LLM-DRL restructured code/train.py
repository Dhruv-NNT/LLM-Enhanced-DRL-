"""Training entry point rebuilt from RL - LLM (Complete Code Pipeline).ipynb."""

import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from configs import (
    BEST_MODEL_PATH,
    EVAL_FREQ,
    LLM_EVERY_K,
    LLM_FIRST_N,
    LOG_DIR,
    MAX_TIMESTEPS,
    N_EVAL_EPISODES,
    REWARD_PARAMS,
    RUN_NAME,
    SAVE_MODEL_FREQ,
    START_LLM_AT_STEP,
    UPDATE_TIMESTEP,
    LAST_CKPT_PATH,
    MEMORY_DIR,
    EVAL_MEMORY_PATH,
)
from rl_llm.env import StructuredEnv
from rl_llm.llm import BasePromptBuilder, LLMGuidanceController, start_llm_log_dir
from rl_llm.ppo import (
    PPO,
    evaluate_policy,
    find_latest_weights,
    load_training_checkpoint,
    save_training_checkpoint,
)


def prune_memory_runs(base_dir: str = MEMORY_DIR, keep: int = 15) -> None:
    """
    Keep only the newest `keep` run folders under `base_dir`.
    """
    if not os.path.isdir(base_dir):
        return

    run_dirs = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith("run_"):
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                mtime = 0.0
            run_dirs.append((mtime, full))

    if len(run_dirs) <= keep:
        return

    run_dirs.sort(key=lambda x: x[0])
    for _, path in run_dirs[:-keep]:
        try:
            shutil.rmtree(path)
            print(f"[MEM] Pruned old run folder: {path}")
        except Exception as exc:
            print(f"[MEM] Failed to prune {path}: {exc}")


@dataclass
class LLMUsageConfig:
    mode: str = "every_k_first_n"
    every_k: int = LLM_EVERY_K
    first_n: int = LLM_FIRST_N


class LLMUsageManager:
    def __init__(self, cfg: LLMUsageConfig):
        self.cfg = cfg
        self.total_llm_episodes = 0

    def start_episode(self, episode_index: int, _: int) -> bool:
        if self.cfg.mode == "every_k_first_n":
            ep1 = episode_index + 1
            use = (ep1 <= self.cfg.first_n) and (ep1 % self.cfg.every_k == 0)
            if use:
                self.total_llm_episodes += 1
            return use
        return False


def build_agent() -> PPO:
    env = StructuredEnv(Reward_Params=REWARD_PARAMS, start_llm_at_step=START_LLM_AT_STEP)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(
        state_dim,
        action_dim,
        lr_actor=1e-4,
        lr_critic=3e-4,
        gamma=0.99,
        K_epochs=5,
        eps_clip=0.2,
        mb_size=256,
        gae_lambda=0.95,
        normalize_reward=True,
    )
    return agent


def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

    env = StructuredEnv(Reward_Params=REWARD_PARAMS, start_llm_at_step=START_LLM_AT_STEP)
    builder = BasePromptBuilder()
    ppo_agent = build_agent()

    ctl = LLMGuidanceController(
        builder=builder,
        k_history=5,
        save_dir=os.path.join(MEMORY_DIR, "_deferred"),
        start_llm_at_step=env.start_llm_at_step,
        memory_path=EVAL_MEMORY_PATH,
        run_id="deferred",
    )

    llm_usage = LLMUsageManager(LLMUsageConfig())

    time_step = 0
    episode = 0
    best_mean_reward = -math.inf
    start_time = datetime.now()

    if os.path.exists(LAST_CKPT_PATH):
        try:
            time_step, episode = load_training_checkpoint(
                LAST_CKPT_PATH,
                ppo_agent,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print(
                f"Resumed FULL checkpoint from {LAST_CKPT_PATH} | time_step={time_step}, episode={episode}"
            )
        except Exception as exc:
            print(
                f"Failed to load {LAST_CKPT_PATH}: {exc}. Falling back to weights-only checkpoint."
            )
            pth_path, resume_step = find_latest_weights(LOG_DIR, BEST_MODEL_PATH)
            if pth_path is not None:
                ppo_agent.load(pth_path)
                time_step = resume_step if resume_step is not None else 5_000_000
                episode = 0
                print(f"Resumed WEIGHTS from {pth_path} | time_step set to {time_step}")
            else:
                print("No checkpoints found; starting from scratch.")
    else:
        pth_path, resume_step = find_latest_weights(LOG_DIR, BEST_MODEL_PATH)
        if pth_path is not None:
            ppo_agent.load(pth_path)
            time_step = resume_step if resume_step is not None else 5_000_000
            episode = 0
            print(f"Resumed WEIGHTS from {pth_path} | time_step set to {time_step}")
        else:
            print("No checkpoints found; starting from scratch.")

    writer = SummaryWriter(LOG_DIR, purge_step=time_step)
    print(f"Starting training run {RUN_NAME} ...")

    try:
        while time_step < MAX_TIMESTEPS:
            obs, _ = env.reset()
            ctl.reset(initial_phase="WAIT_TURN")
            ppo_agent.clear_fused_cache()

            done = False
            ep_reward = 0.0

            use_llm_episode = llm_usage.start_episode(episode_index=episode, _=time_step)
            writer.add_scalar("llm/use_this_episode", int(use_llm_episode), episode)

            if use_llm_episode:
                ep1 = episode + 1
                run_dir, base_run_id = start_llm_log_dir(base_dir=MEMORY_DIR)
                run_dir_ep = f"{run_dir}__ep{ep1:06d}"
                try:
                    os.rename(run_dir, run_dir_ep)
                except OSError:
                    os.makedirs(run_dir_ep, exist_ok=True)
                    try:
                        os.rmdir(run_dir)
                    except OSError:
                        pass
                ctl.save_dir = run_dir_ep
                ctl.run_id = f"{base_run_id}__ep{ep1:06d}"
                print(f"[LLM] Episode {ep1}: memory → {ctl.save_dir} | run_id={ctl.run_id}")

            for t in range(1, env.MAX_STEP + 1):
                if use_llm_episode:
                    turn_deg, ctl_state = ctl.next_action(
                        obs,
                        step=env.n_step + 1,
                        fileinfo=env.agent.fileinfo,
                        env=env,
                        ppo=ppo_agent,
                    )
                    intent_embed, target_deg, is_exec_mask = ctl.get_advice_for_rl()
                    reuse_cached = ctl_state.hold_remaining > 0
                else:
                    intent_embed = None
                    target_deg = 0.0
                    is_exec_mask = False
                    reuse_cached = False
                    turn_deg = 0.0

                action = ppo_agent.select_action_hybrid(
                    state=obs,
                    intent_embed=intent_embed,
                    phase_active=is_exec_mask,
                    target_deg=target_deg,
                    reuse_cached=reuse_cached,
                )

                obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc

                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                ep_reward += reward
                time_step += 1

                if time_step % UPDATE_TIMESTEP == 0:
                    ppo_agent.update()
                    save_training_checkpoint(LAST_CKPT_PATH, ppo_agent, time_step, episode)

                if time_step % EVAL_FREQ == 0:
                    mean_ret, std_ret = evaluate_policy(
                        ppo_agent, env, N_EVAL_EPISODES
                    )
                    writer.add_scalar("eval/mean_reward", mean_ret, time_step)
                    writer.add_scalar("eval/std_reward", std_ret, time_step)
                    save_training_checkpoint(LAST_CKPT_PATH, ppo_agent, time_step, episode)
                    if mean_ret > best_mean_reward:
                        best_mean_reward = mean_ret
                        ppo_agent.save(BEST_MODEL_PATH)
                        print(
                            f"Best model saved at {BEST_MODEL_PATH} "
                            f"[{time_step}] mean reward {mean_ret:.2f}"
                        )

                if time_step % SAVE_MODEL_FREQ == 0:
                    path = os.path.join(LOG_DIR, f"checkpoint_{time_step}.pth")
                    ppo_agent.save(path)
                    print(f"[{time_step}] Checkpoint saved to {path}")

                if done:
                    break

            episode += 1
            writer.add_scalar("train/episode_reward", ep_reward, episode)
            writer.add_scalar("train/steps_per_episode", t, episode)

            if use_llm_episode:
                prune_memory_runs(base_dir=MEMORY_DIR, keep=15)

            if episode % 10 == 0:
                elapsed = datetime.now() - start_time
                print(
                    f"Episode {episode} | Step {time_step} | "
                    f"Ep Reward {ep_reward:.2f} | Elapsed {elapsed} | "
                    f"LLM={'ON' if use_llm_episode else 'OFF'}"
                )

    except KeyboardInterrupt:
        emergency = os.path.join(LOG_DIR, "interrupted.ckpt")
        save_training_checkpoint(emergency, ppo_agent, time_step, episode)
        print(f"\nInterrupted — saved {emergency}")
        raise
    except Exception:
        emergency = os.path.join(LOG_DIR, "interrupted.ckpt")
        save_training_checkpoint(emergency, ppo_agent, time_step, episode)
        print(f"\nCrashed — saved {emergency}")
        raise
    finally:
        writer.add_scalar("llm/episodes_total", llm_usage.total_llm_episodes, time_step)
        writer.close()
        print("Training complete in", datetime.now() - start_time)


if __name__ == "__main__":
    main()
