"""Training entry point rebuilt from RL - LLM (Complete Code Pipeline).ipynb."""
# This import gives us math helpers like infinity for tracking best scores.
import math
# This import lets us talk to the operating system for file paths and checks.
import os
# This import lets us delete folders when they are old.
import shutil
# This decorator helper makes it easy to build simple data containers.
from dataclasses import dataclass
# We use datetime to track when training started and ended.
from datetime import datetime

# Torch is the deep learning library that powers our PPO agent.
import torch
# We log training numbers into TensorBoard using this writer class.
from torch.utils.tensorboard import SummaryWriter

# These are all the knobs and paths that control the training run.
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
# StructuredEnv wraps the air traffic scenario into an RL friendly gym.
from rl_llm.env import StructuredEnv
# These classes handle LLM prompts, responses, and logging of LLM guidance.
from rl_llm.llm import BasePromptBuilder, LLMGuidanceController, start_llm_log_dir
# PPO and helper utilities handle training, saving, loading, and evaluation.
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
    # We skip all work if the memory folder does not exist yet.
    if not os.path.isdir(base_dir):
        return

    # This list will hold pairs of (modified time, folder path).
    run_dirs = []
    # We look through everything inside the memory folder.
    for name in os.listdir(base_dir):
        # We build the full path for the potential run folder.
        full = os.path.join(base_dir, name)
        # We only care about folders that follow the run_ naming pattern.
        if os.path.isdir(full) and name.startswith("run_"):
            try:
                # We remember when this folder was last touched.
                mtime = os.path.getmtime(full)
            except OSError:
                # If we cannot read the time we treat it as very old.
                mtime = 0.0
            # We store the time and folder so we can sort later.
            run_dirs.append((mtime, full))

    # If we already have only a few folders, we do nothing.
    if len(run_dirs) <= keep:
        return

    # We sort folders by time so the oldest ones come first.
    run_dirs.sort(key=lambda x: x[0])
    # We remove every old folder except for the newest `keep` ones.
    for _, path in run_dirs[:-keep]:
        try:
            # We delete the folder and everything inside it.
            shutil.rmtree(path)
            print(f"[MEM] Pruned old run folder: {path}")
        except Exception as exc:
            # If delete fails we show a message but keep going.
            print(f"[MEM] Failed to prune {path}: {exc}")


@dataclass
class LLMUsageConfig:
    # Name of the simple rule that decides when to call the LLM.
    mode: str = "every_k_first_n"
    # Only every_k-th episode will ask the LLM at the start of training.
    every_k: int = LLM_EVERY_K
    # We limit LLM use to the first_n episodes to save time.
    first_n: int = LLM_FIRST_N


class LLMUsageManager:
    def __init__(self, cfg: LLMUsageConfig):
        # We remember the configuration that controls usage.
        self.cfg = cfg
        # We count how many times the LLM actually helped.
        self.total_llm_episodes = 0

    def start_episode(self, episode_index: int, _: int) -> bool:
        # We only support the simple every k episodes rule for now.
        if self.cfg.mode == "every_k_first_n":
            # Convert zero-based episode index into a human friendly number.
            ep1 = episode_index + 1
            # We call the LLM if we are still within the limit and on the correct stride.
            use = (ep1 <= self.cfg.first_n) and (ep1 % self.cfg.every_k == 0)
            # We track how many episodes actually used the LLM.
            if use:
                self.total_llm_episodes += 1
            return use
        # If the mode is unknown we do not use the LLM.
        return False


def build_agent() -> PPO:
    # We make a temporary environment to learn the observation and action sizes.
    env = StructuredEnv(Reward_Params=REWARD_PARAMS, start_llm_at_step=START_LLM_AT_STEP)
    # The state dimension is the length of the observation vector.
    state_dim = env.observation_space.shape[0]
    # The action dimension is how many discrete turns the agent can pick.
    action_dim = env.action_space.n
    # We create the PPO agent with hand-picked hyperparameters.
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
    # We hand the ready to use agent back to the caller.
    return agent


def main() -> None:
    # We make sure the logging folder exists before training starts.
    os.makedirs(LOG_DIR, exist_ok=True)

    # We create a fresh environment that the agent will interact with.
    env = StructuredEnv(Reward_Params=REWARD_PARAMS, start_llm_at_step=START_LLM_AT_STEP)
    # We build a helper that prepares prompts for the language model.
    builder = BasePromptBuilder()
    # We set up the PPO learning agent with the helper function defined above.
    ppo_agent = build_agent()

    # This controller keeps track of LLM advice, memory, and prompts.
    ctl = LLMGuidanceController(
        # We pass in the prompt builder that crafts the question text.
        builder=builder,
        # We remember up to five turns when talking with the LLM.
        k_history=5,
        # We store LLM conversations inside a deferred memory folder by default.
        save_dir=os.path.join(MEMORY_DIR, "_deferred"),
        # We tell the controller when the agent should start asking for LLM help.
        start_llm_at_step=env.start_llm_at_step,
        # We pass the memory file used during evaluation to keep prompts consistent.
        memory_path=EVAL_MEMORY_PATH,
        # We tag the run so we can track these conversations on disk.
        run_id="deferred",
    )

    # This helper decides which episodes should use the LLM.
    llm_usage = LLMUsageManager(LLMUsageConfig())

    # We start counting timesteps of interactions from zero.
    time_step = 0
    # We start counting episodes from zero as well.
    episode = 0
    # We remember the best evaluation reward seen so far.
    best_mean_reward = -math.inf
    # We record when training began to show runtime later.
    start_time = datetime.now()

    # We first check if a full training checkpoint is available.
    if os.path.exists(LAST_CKPT_PATH):
        try:
            # We load the checkpoint so we can continue training.
            time_step, episode = load_training_checkpoint(
                LAST_CKPT_PATH,
                ppo_agent,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            # We explain that training resumed from the saved state.
            print(
                f"Resumed FULL checkpoint from {LAST_CKPT_PATH} | time_step={time_step}, episode={episode}"
            )
        except Exception as exc:
            # If the full checkpoint fails we fall back to plain weight files.
            print(
                f"Failed to load {LAST_CKPT_PATH}: {exc}. Falling back to weights-only checkpoint."
            )
            # We look for the newest model weights inside the log directory.
            pth_path, resume_step = find_latest_weights(LOG_DIR, BEST_MODEL_PATH)
            if pth_path is not None:
                # We load only the neural network weights into the agent.
                ppo_agent.load(pth_path)
                # We guess the timestep if the saved file told us, otherwise pick a default.
                time_step = resume_step if resume_step is not None else 5_000_000
                # We restart episode counting because we only know the weights.
                episode = 0
                # We let the user know we loaded from the lighter backup.
                # We explain that training will continue from the recovered weights.
                print(f"Resumed WEIGHTS from {pth_path} | time_step set to {time_step}")
            else:
                # No checkpoint was found so we will train from scratch.
                print("No checkpoints found; starting from scratch.")
    else:
        # If no full checkpoint path existed we still look for weight-only files.
        pth_path, resume_step = find_latest_weights(LOG_DIR, BEST_MODEL_PATH)
        if pth_path is not None:
            # We load the best weights that were found.
            ppo_agent.load(pth_path)
            # We restore the saved timestep if possible.
            time_step = resume_step if resume_step is not None else 5_000_000
            # We reset the episode counter because only weights were stored.
            episode = 0
            # We share the resume information with the user.
            print(f"Resumed WEIGHTS from {pth_path} | time_step set to {time_step}")
        else:
            print("No checkpoints found; starting from scratch.")

    # We start a TensorBoard writer that resumes from the loaded step.
    writer = SummaryWriter(LOG_DIR, purge_step=time_step)
    # We inform the user that training is about to begin.
    print(f"Starting training run {RUN_NAME} ...")

    try:
        # We keep training until we hit the target number of timesteps.
        while time_step < MAX_TIMESTEPS:
            # We reset the environment to get a fresh start for the episode.
            obs, _ = env.reset()
            # We reset the LLM controller so it knows the new episode phase.
            ctl.reset(initial_phase="WAIT_TURN")
            # We clear any cached fused features from the PPO agent.
            ppo_agent.clear_fused_cache()

            # We track whether the episode finished.
            done = False
            # We sum rewards earned in this episode.
            ep_reward = 0.0

            # We ask the helper if this episode should use LLM advice.
            use_llm_episode = llm_usage.start_episode(episode_index=episode, _=time_step)
            # We log the LLM usage decision to TensorBoard.
            writer.add_scalar("llm/use_this_episode", int(use_llm_episode), episode)

            if use_llm_episode:
                # We switch to human-friendly numbering for logging folders.
                ep1 = episode + 1
                # We start a new LLM memory folder on disk.
                run_dir, base_run_id = start_llm_log_dir(base_dir=MEMORY_DIR)
                # We append the episode number to keep folders unique.
                run_dir_ep = f"{run_dir}__ep{ep1:06d}"
                try:
                    # We rename the folder to include the episode number.
                    os.rename(run_dir, run_dir_ep)
                except OSError:
                    # If the rename fails we make the folder ourselves.
                    os.makedirs(run_dir_ep, exist_ok=True)
                    try:
                        # We try to remove the original empty folder.
                        os.rmdir(run_dir)
                    except OSError:
                        # It is okay if removal fails; we just ignore it.
                        pass
                # We point the controller at the final memory folder.
                ctl.save_dir = run_dir_ep
                # We tag the run id with the same episode number.
                ctl.run_id = f"{base_run_id}__ep{ep1:06d}"
                # We log where the LLM memories will be stored.
                print(f"[LLM] Episode {ep1}: memory → {ctl.save_dir} | run_id={ctl.run_id}")

            for t in range(1, env.MAX_STEP + 1):
                # We handle two paths depending on whether LLM advice is active.
                if use_llm_episode:
                    # We ask the controller for the next suggestion and state info.
                    turn_deg, ctl_state = ctl.next_action(
                        obs,
                        step=env.n_step + 1,
                        fileinfo=env.agent.fileinfo,
                        env=env,
                        ppo=ppo_agent,
                    )
                    # We collect the embedding, target angle, and mask for PPO.
                    intent_embed, target_deg, is_exec_mask = ctl.get_advice_for_rl()
                    # We check if the LLM wants to reuse the previous fused plan.
                    reuse_cached = ctl_state.hold_remaining > 0
                else:
                    # When no LLM is used we leave all helper values in plain form.
                    intent_embed = None
                    target_deg = 0.0
                    is_exec_mask = False
                    reuse_cached = False
                    turn_deg = 0.0

                # We pick an action from the hybrid policy that can mix in LLM hints.
                action = ppo_agent.select_action_hybrid(
                    state=obs,
                    intent_embed=intent_embed,
                    phase_active=is_exec_mask,
                    target_deg=target_deg,
                    reuse_cached=reuse_cached,
                )

                # We send the chosen action to the environment and get the result.
                obs, reward, term, trunc, _ = env.step(action)
                # The episode ends if either a terminal or truncation flag is set.
                done = term or trunc

                # We store the reward for policy updates later on.
                ppo_agent.buffer.rewards.append(reward)
                # We record if the step finished the episode.
                ppo_agent.buffer.is_terminals.append(done)

                # We keep track of the running reward for this episode.
                ep_reward += reward
                # We advance the global timestep counter.
                time_step += 1

                # Every fixed number of steps we trigger a PPO update.
                if time_step % UPDATE_TIMESTEP == 0:
                    ppo_agent.update()
                    # We save a full checkpoint so we can resume from this point.
                    save_training_checkpoint(LAST_CKPT_PATH, ppo_agent, time_step, episode)

                # At evaluation intervals we test the current policy.
                if time_step % EVAL_FREQ == 0:
                    mean_ret, std_ret = evaluate_policy(
                        ppo_agent, env, N_EVAL_EPISODES
                    )
                    # We log the evaluation mean reward to TensorBoard.
                    writer.add_scalar("eval/mean_reward", mean_ret, time_step)
                    # We log the evaluation standard deviation too.
                    writer.add_scalar("eval/std_reward", std_ret, time_step)
                    # We save the checkpoint after evaluation for safety.
                    save_training_checkpoint(LAST_CKPT_PATH, ppo_agent, time_step, episode)
                    if mean_ret > best_mean_reward:
                        # We remember the new best reward.
                        best_mean_reward = mean_ret
                        # We save the agent weights to the best model path.
                        ppo_agent.save(BEST_MODEL_PATH)
                        # We log that a better model has been stored.
                        print(
                            f"Best model saved at {BEST_MODEL_PATH} "
                            f"[{time_step}] mean reward {mean_ret:.2f}"
                        )

                # We periodically save the model regardless of performance.
                if time_step % SAVE_MODEL_FREQ == 0:
                    path = os.path.join(LOG_DIR, f"checkpoint_{time_step}.pth")
                    ppo_agent.save(path)
                    print(f"[{time_step}] Checkpoint saved to {path}")

                # If the episode is done we exit the step loop early.
                if done:
                    break

            # We move to the next episode count now that the loop finished.
            episode += 1
            # We log how much reward we earned this episode.
            writer.add_scalar("train/episode_reward", ep_reward, episode)
            # We log how many steps the episode lasted.
            writer.add_scalar("train/steps_per_episode", t, episode)

            # If LLM advice was active we trim old memory folders.
            if use_llm_episode:
                prune_memory_runs(base_dir=MEMORY_DIR, keep=15)

            # Every ten episodes we print a short progress update.
            if episode % 10 == 0:
                elapsed = datetime.now() - start_time
                print(
                    f"Episode {episode} | Step {time_step} | "
                    f"Ep Reward {ep_reward:.2f} | Elapsed {elapsed} | "
                    f"LLM={'ON' if use_llm_episode else 'OFF'}"
                )

    except KeyboardInterrupt:
        # If the user stops the run we save progress to an emergency file.
        emergency = os.path.join(LOG_DIR, "interrupted.ckpt")
        save_training_checkpoint(emergency, ppo_agent, time_step, episode)
        print(f"\nInterrupted — saved {emergency}")
        raise
    except Exception:
        # If anything unexpected happens we also save to the same emergency file.
        emergency = os.path.join(LOG_DIR, "interrupted.ckpt")
        save_training_checkpoint(emergency, ppo_agent, time_step, episode)
        print(f"\nCrashed — saved {emergency}")
        raise
    finally:
        # We log the total number of LLM-assisted episodes at the final step.
        writer.add_scalar("llm/episodes_total", llm_usage.total_llm_episodes, time_step)
        # We close the TensorBoard writer to flush all pending data.
        writer.close()
        # We print how long the full training loop took.
        print("Training complete in", datetime.now() - start_time)


if __name__ == "__main__":
    # Running this file directly will launch the training routine.
    main()
