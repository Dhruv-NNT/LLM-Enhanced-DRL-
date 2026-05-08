from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from configs import (  # noqa: E402
    ACTION_BINS,
    MAX_AGENTS,
    MAX_WEATHER_CELLS,
    REWARD_CROSS_TRACK_SCALE,
    REWARD_GREEN_WEATHER_PENETRATION_SCALE,
    REWARD_STEP_PENALTY,
)
from rl_llm_multi import MAPPO, MultiAgentParallelEnv, MultiAgentSectorCore, WeatherCell  # noqa: E402
from rl_llm_multi.mappo import evaluate_policy  # noqa: E402
from rl_llm_multi.utils import line_signed_cross_track, weather_axis_units  # noqa: E402


def _make_env(max_step: int = 5) -> MultiAgentParallelEnv:
    env = MultiAgentParallelEnv(num_agents=2, max_agents=MAX_AGENTS, num_weather_cells=1)
    env.core.max_step = int(max_step)
    return env


def _make_agent(env: MultiAgentParallelEnv) -> MAPPO:
    return MAPPO(
        global_state_dim=int(env.state().shape[0]),
        action_dim=len(ACTION_BINS),
        max_agents=MAX_AGENTS,
        hidden_dim=32,
        K_epochs=1,
        mb_size=8,
    )


class MAPPOTests(unittest.TestCase):
    def test_global_state_has_fixed_weather_slots(self) -> None:
        one_weather = MultiAgentSectorCore(num_agents=2, num_weather_cells=1)
        one_weather.reset(seed=42, num_weather_cells=1)
        state_one = one_weather.global_state_vector()

        two_weather = MultiAgentSectorCore(num_agents=2, num_weather_cells=2)
        two_weather.reset(seed=42, num_weather_cells=2)
        state_two = two_weather.global_state_vector()

        expected_dim = MAX_AGENTS * 16 + MAX_WEATHER_CELLS * 8 + 3
        weather_offset = MAX_AGENTS * 16
        self.assertEqual(state_one.shape, (expected_dim,))
        self.assertEqual(state_two.shape, (expected_dim,))
        self.assertEqual(state_one[weather_offset], 1.0)
        self.assertEqual(state_one[weather_offset + 8], 0.0)
        self.assertEqual(state_two[weather_offset], 1.0)
        self.assertEqual(state_two[weather_offset + 8], 1.0)

    def test_mappo_selects_one_action_per_active_agent(self) -> None:
        env = _make_env()
        env.reset(seed=42)
        agent = _make_agent(env)

        actions, records, entropy = agent.select_actions(
            env.state(),
            list(env.agents),
            episode_id=0,
        )

        self.assertEqual(set(actions), set(env.agents))
        self.assertEqual({record.agent_id for record in records}, set(env.agents))
        self.assertGreaterEqual(entropy, 0.0)
        for action in actions.values():
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, len(ACTION_BINS))

    def test_green_weather_ring_adds_penalty_without_termination(self) -> None:
        core = MultiAgentSectorCore(num_agents=1, num_weather_cells=1)
        core.reset(seed=42, num_weather_cells=1)
        state = core.get_agent("A1")
        cell = WeatherCell(
            (50.0, 50.0),
            15.0,
            0.8,
            0.0,
            0.0,
            0.0,
            0.0,
            cell_id="W1",
            movement_stopped=True,
        )
        major_units, _ = weather_axis_units(cell)
        target_position = (cell.center[0] + 0.90 * major_units, cell.center[1])
        state.position = (target_position[0] - core.speed, target_position[1])
        state.destination = (90.0, 90.0)
        state.heading_rad = 0.0
        state.launched = True
        state.active = True
        state.has_entered_sector = True
        state.inside_sector = True
        state.hazard_predicted = False
        state.weather_predicted = False
        core.weather_cells = [cell]
        core._sync_primary_weather_cell()

        previous_distance = core.distance_to_destination(state)
        _, _, terminations, _, info = core.step({}, compute_predictions=False)

        self.assertFalse(any(terminations.values()))
        self.assertIsNone(info["failure_reason"])
        self.assertLess(core.weather_signed_clearance(state.position), 0.0)
        self.assertGreaterEqual(core.weather_terminal_signed_clearance(state.position), 0.0)

        progress = previous_distance - core.distance_to_destination(state)
        signed_xtrk, _, _ = line_signed_cross_track(state.route.linestring, state.position)
        green_penalty = (
            REWARD_GREEN_WEATHER_PENETRATION_SCALE
            * abs(core.weather_signed_clearance(state.position))
        )
        expected_without_green = (
            progress
            - REWARD_STEP_PENALTY
            - REWARD_CROSS_TRACK_SCALE * abs(signed_xtrk)
        )
        self.assertAlmostEqual(core.reward, expected_without_green - green_penalty, places=6)

    def test_short_rollout_update_clears_buffer(self) -> None:
        env = _make_env()
        _, info = env.reset(seed=42)
        agent = _make_agent(env)
        done = bool(info["__common__"]["episode_done"])
        truncated = bool(info["__common__"]["episode_truncated"])

        for _ in range(3):
            if done or truncated:
                break
            active_ids = list(env.agents)
            actions, records, _ = agent.select_actions(env.state(), active_ids, episode_id=0)
            _, _, _, _, info = env.step(actions)
            common = info["__common__"]
            team_terminal = bool(common["episode_done"] or common["episode_truncated"])
            post_active = set(common["active_agents"])
            terminals = {
                agent_id: bool(team_terminal or agent_id not in post_active)
                for agent_id in active_ids
            }
            agent.store_outcomes(records, reward=float(common["team_reward"]), terminals=terminals)
            done = bool(common["episode_done"])
            truncated = bool(common["episode_truncated"])

        self.assertGreater(len(agent.buffer), 0)
        metrics = agent.update()
        self.assertIn("loss", metrics)
        self.assertEqual(len(agent.buffer), 0)

    def test_checkpoint_roundtrip_restores_counters(self) -> None:
        env = _make_env()
        env.reset(seed=42)
        agent = _make_agent(env)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mappo.ckpt"
            agent.save_checkpoint(path, agent_steps=12, env_steps=7, episode=3)

            restored = _make_agent(env)
            counters = restored.load_checkpoint(path)

        self.assertEqual(counters, (12, 7, 3))

    def test_deterministic_evaluation_returns_metrics(self) -> None:
        env = _make_env(max_step=3)
        env.reset(seed=42)
        agent = _make_agent(env)

        def make_eval_env() -> MultiAgentParallelEnv:
            return _make_env(max_step=3)

        stats = evaluate_policy(agent, make_eval_env, n_episodes=1, seed=42)
        self.assertIn("mean_return", stats)
        self.assertIn("success_rate", stats)
        self.assertGreaterEqual(stats["mean_steps"], 0.0)


if __name__ == "__main__":
    unittest.main()
