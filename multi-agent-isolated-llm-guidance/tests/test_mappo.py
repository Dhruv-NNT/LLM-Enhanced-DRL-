from __future__ import annotations

import math
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
    HAZARD_RISK_DECAY_EXPONENT,
    LOCAL_TRAFFIC_NEIGHBOR_COUNT,
    REWARD_COLLISION_PENALTY,
    REWARD_CROSS_TRACK_CLIP,
    REWARD_CROSS_TRACK_HAZARD_RELIEF,
    REWARD_CROSS_TRACK_RECOVERY_CLIP,
    REWARD_CROSS_TRACK_RECOVERY_SCALE,
    REWARD_CROSS_TRACK_SCALE,
    REWARD_DEST_HEADING_SCALE,
    REWARD_GREEN_WEATHER_PENETRATION_SCALE,
    REWARD_MERGE_BACK_RISK_THRESHOLD,
    REWARD_MERGE_BACK_SCALE,
    REWARD_PROGRESS_CLIP,
    REWARD_PROGRESS_HAZARD_RELIEF,
    REWARD_PROGRESS_SCALE,
    REWARD_STEP_PENALTY,
    REWARD_TRAFFIC_RISK_PENALTY,
    REWARD_TRAFFIC_RISK_REDUCTION_SCALE,
    REWARD_TRUNCATION_DISTANCE_PENALTY_SCALE,
    REWARD_TRUNCATION_PENALTY,
    REWARD_WEATHER_RISK_REDUCTION_SCALE,
    REWARD_WEATHER_RISK_PENALTY,
    PAIR_RISK_BUFFER,
    SAFE_R,
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
        local_obs_dim=int(env.observation_space(env.possible_agents[0]).shape[0]),
        action_dim=len(ACTION_BINS),
        max_agents=MAX_AGENTS,
        hidden_dim=32,
        K_epochs=1,
        mb_size=8,
    )


def _advance_to_decision_agent(
    env: MultiAgentParallelEnv,
    *,
    seed: int = 42,
) -> tuple[dict[str, object], dict[str, object]]:
    observations, info = env.reset(seed=seed)
    for _ in range(env.core.max_step + 5):
        if env.agents:
            return observations, info
        observations, _, _, _, info = env.step({})
        common = info["__common__"]
        if bool(common["episode_done"] or common["episode_truncated"]):
            break
    return observations, info


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

    def test_local_observation_includes_three_nearest_neighbors(self) -> None:
        core = MultiAgentSectorCore(num_agents=4, num_weather_cells=1)
        core.reset(seed=42, num_weather_cells=1)
        own_id, near_id, middle_id, far_id = core.possible_agents[:4]

        positions = {
            own_id: (0.0, 0.0),
            near_id: (3.0, 0.0),
            middle_id: (0.0, 4.0),
            far_id: (-10.0, 0.0),
        }
        headings = {
            own_id: 0.0,
            near_id: math.radians(10.0),
            middle_id: math.radians(20.0),
            far_id: math.radians(30.0),
        }
        for agent_id, state in core.agent_states.items():
            if agent_id in positions:
                state.launched = True
                state.finished = False
                state.position = positions[agent_id]
                state.heading_rad = headings[agent_id]
            else:
                state.launched = False
                state.finished = True

        obs = core.get_local_observation(own_id)
        neighbor_start = 12
        neighbor_width = 6
        neighbor_block = obs[
            neighbor_start : neighbor_start + LOCAL_TRAFFIC_NEIGHBOR_COUNT * neighbor_width
        ]

        self.assertEqual(obs.shape, (core.local_obs_dim,))
        self.assertEqual(core.local_obs_dim, 22 + LOCAL_TRAFFIC_NEIGHBOR_COUNT * neighbor_width)
        self.assertEqual(list(neighbor_block[0::neighbor_width]), [3.0, 4.0, 10.0])
        caution_distance = SAFE_R * PAIR_RISK_BUFFER
        self.assertEqual(
            list(neighbor_block[3::neighbor_width]),
            [1.0 if distance <= caution_distance else 0.0 for distance in (3.0, 4.0, 10.0)],
        )
        expected_proximity = [
            max(0.0, (caution_distance - distance) / caution_distance)
            for distance in (3.0, 4.0, 10.0)
        ]
        for actual, expected in zip(neighbor_block[4::neighbor_width], expected_proximity):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(neighbor_block[2], 10.0)
        self.assertAlmostEqual(neighbor_block[8], 20.0)
        self.assertAlmostEqual(neighbor_block[14], 30.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in neighbor_block[5::neighbor_width]))

    def test_local_observation_pads_missing_neighbors_as_non_caution(self) -> None:
        core = MultiAgentSectorCore(num_agents=1, num_weather_cells=1)
        core.reset(seed=42, num_weather_cells=1)
        agent_id = core.possible_agents[0]

        obs = core.get_local_observation(agent_id)
        neighbor_start = 12
        neighbor_width = 6
        neighbor_block = obs[
            neighbor_start : neighbor_start + LOCAL_TRAFFIC_NEIGHBOR_COUNT * neighbor_width
        ]
        padded_distance = 2.0 * SAFE_R * PAIR_RISK_BUFFER

        self.assertEqual(list(neighbor_block[0::neighbor_width]), [padded_distance] * LOCAL_TRAFFIC_NEIGHBOR_COUNT)
        self.assertEqual(list(neighbor_block[3::neighbor_width]), [0.0] * LOCAL_TRAFFIC_NEIGHBOR_COUNT)
        self.assertEqual(list(neighbor_block[4::neighbor_width]), [0.0] * LOCAL_TRAFFIC_NEIGHBOR_COUNT)
        self.assertEqual(list(neighbor_block[5::neighbor_width]), [0.0] * LOCAL_TRAFFIC_NEIGHBOR_COUNT)

    def test_mappo_selects_one_action_per_active_agent(self) -> None:
        env = _make_env(max_step=30)
        observations, _ = _advance_to_decision_agent(env, seed=42)
        agent = _make_agent(env)

        actions, records, entropy = agent.select_actions(
            global_state=env.state(),
            local_observations=observations,
            active_agent_ids=list(env.agents),
            episode_id=0,
        )

        self.assertEqual(set(actions), set(env.agents))
        self.assertEqual({record.agent_id for record in records}, set(env.agents))
        self.assertGreaterEqual(entropy, 0.0)
        for action in actions.values():
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, len(ACTION_BINS))

    def test_critic_is_agent_conditioned(self) -> None:
        env = _make_env()
        env.reset(seed=42)
        agent = _make_agent(env)

        expected_dim = int(env.state().shape[0]) + MAX_AGENTS
        self.assertEqual(agent.policy.critic[0].in_features, expected_dim)
        self.assertEqual(
            agent.policy.actor[0].in_features,
            int(env.observation_space(env.possible_agents[0]).shape[0]) + MAX_AGENTS,
        )

    def test_configurable_mlp_architecture_uses_layer_norm_and_silu(self) -> None:
        env = _make_env()
        env.reset(seed=42)
        agent = MAPPO(
            global_state_dim=int(env.state().shape[0]),
            local_obs_dim=int(env.observation_space(env.possible_agents[0]).shape[0]),
            action_dim=len(ACTION_BINS),
            max_agents=MAX_AGENTS,
            hidden_dims=(32, 32, 16),
            activation="silu",
            use_layer_norm=True,
            orthogonal_init=True,
            K_epochs=1,
            mb_size=8,
        )

        actor_names = [module.__class__.__name__ for module in agent.policy.actor]
        critic_names = [module.__class__.__name__ for module in agent.policy.critic]

        self.assertEqual(actor_names.count("Linear"), 4)
        self.assertEqual(critic_names.count("Linear"), 4)
        self.assertIn("LayerNorm", actor_names)
        self.assertIn("SiLU", actor_names)
        self.assertEqual(actor_names[-1], "Linear")

    def test_lookahead_risk_uses_slow_decay_curve(self) -> None:
        core = MultiAgentSectorCore(num_agents=1, num_weather_cells=1)

        self.assertAlmostEqual(core._lookahead_risk(1, horizon_steps=12), 1.0)
        self.assertAlmostEqual(
            core._lookahead_risk(6, horizon_steps=12),
            (7.0 / 12.0) ** HAZARD_RISK_DECAY_EXPONENT,
            places=6,
        )
        self.assertAlmostEqual(
            core._lookahead_risk(12, horizon_steps=12),
            (1.0 / 12.0) ** HAZARD_RISK_DECAY_EXPONENT,
            places=6,
        )
        self.assertEqual(core._lookahead_risk(None, horizon_steps=12), 0.0)
        self.assertEqual(core._lookahead_risk(13, horizon_steps=12), 0.0)

    def test_reward_coefficients_prioritize_avoidance_then_recovery(self) -> None:
        max_progress = REWARD_PROGRESS_SCALE * REWARD_PROGRESS_CLIP
        max_progress_at_full_risk = max_progress * (1.0 - REWARD_PROGRESS_HAZARD_RELIEF)
        max_cross_track_penalty = REWARD_CROSS_TRACK_SCALE * REWARD_CROSS_TRACK_CLIP
        max_cross_track_penalty_at_full_risk = max_cross_track_penalty * (
            1.0 - REWARD_CROSS_TRACK_HAZARD_RELIEF
        )

        self.assertGreaterEqual(REWARD_PROGRESS_HAZARD_RELIEF, 0.0)
        self.assertLessEqual(REWARD_PROGRESS_HAZARD_RELIEF, 1.0)
        self.assertGreaterEqual(REWARD_CROSS_TRACK_HAZARD_RELIEF, 0.0)
        self.assertLessEqual(REWARD_CROSS_TRACK_HAZARD_RELIEF, 1.0)
        self.assertGreater(REWARD_TRAFFIC_RISK_PENALTY, max_progress)
        self.assertGreater(REWARD_WEATHER_RISK_PENALTY, max_progress)
        self.assertGreater(REWARD_TRAFFIC_RISK_PENALTY, max_progress_at_full_risk)
        self.assertGreater(REWARD_WEATHER_RISK_PENALTY, max_progress_at_full_risk)
        self.assertLess(max_cross_track_penalty_at_full_risk, max_cross_track_penalty)
        self.assertGreater(REWARD_TRAFFIC_RISK_REDUCTION_SCALE, max_cross_track_penalty)
        self.assertGreater(REWARD_WEATHER_RISK_REDUCTION_SCALE, max_cross_track_penalty)
        self.assertGreater(REWARD_CROSS_TRACK_RECOVERY_SCALE, REWARD_STEP_PENALTY)
        self.assertGreater(REWARD_DEST_HEADING_SCALE, REWARD_STEP_PENALTY)
        self.assertGreater(REWARD_MERGE_BACK_RISK_THRESHOLD, 0.0)
        self.assertLess(REWARD_MERGE_BACK_RISK_THRESHOLD, 1.0)
        self.assertGreater(REWARD_MERGE_BACK_SCALE, REWARD_STEP_PENALTY)
        self.assertLess(REWARD_MERGE_BACK_SCALE, REWARD_TRAFFIC_RISK_REDUCTION_SCALE)
        self.assertLess(REWARD_MERGE_BACK_SCALE, REWARD_WEATHER_RISK_REDUCTION_SCALE)
        self.assertLess(
            REWARD_CROSS_TRACK_RECOVERY_SCALE + REWARD_MERGE_BACK_SCALE,
            REWARD_TRAFFIC_RISK_PENALTY,
        )
        self.assertLess(
            REWARD_CROSS_TRACK_RECOVERY_SCALE + REWARD_MERGE_BACK_SCALE,
            REWARD_WEATHER_RISK_PENALTY,
        )

    def test_local_traffic_risk_uses_proximity_and_closing(self) -> None:
        core = MultiAgentSectorCore(num_agents=2, num_weather_cells=1)
        core.reset(seed=42, num_weather_cells=1)
        core.weather_cells = []
        core._sync_primary_weather_cell()

        states = {
            "A1": ((40.0, 40.0), 0.0),
            "A2": ((48.0, 40.0), math.pi),
        }
        for agent_id, (position, heading) in states.items():
            state = core.get_agent(agent_id)
            state.position = position
            state.destination = (90.0, position[1])
            state.heading_rad = heading
            state.launched = True
            state.active = True
            state.finished = False
            state.has_entered_sector = True
            state.inside_sector = True
            state.predicted_pair_loss_step = None
            state.predicted_min_sep = float("inf")

        parts = core._traffic_risk_parts(core.get_agent("A1"), set())
        caution_distance = SAFE_R * PAIR_RISK_BUFFER
        proximity = (caution_distance - 8.0) / caution_distance
        expected_local_risk = min(1.0, proximity * 1.5)

        self.assertAlmostEqual(parts["traffic_step_risk"], 0.0)
        self.assertAlmostEqual(parts["traffic_separation_risk"], 0.0)
        self.assertAlmostEqual(parts["local_traffic_risk"], expected_local_risk)
        self.assertAlmostEqual(parts["traffic_risk"], expected_local_risk)

    def test_dense_reward_uses_local_neighbor_traffic_risk(self) -> None:
        core = MultiAgentSectorCore(num_agents=2, num_weather_cells=1)
        core.reset(seed=42, num_weather_cells=1)
        core.weather_cells = []
        core._sync_primary_weather_cell()

        for agent_id, position in {"A1": (40.0, 40.0), "A2": (48.0, 40.0)}.items():
            state = core.get_agent(agent_id)
            state.position = position
            state.destination = (90.0, position[1])
            state.heading_rad = 0.0
            state.launched = True
            state.active = True
            state.finished = False
            state.has_entered_sector = True
            state.inside_sector = True
            state.predicted_pair_loss_step = None
            state.predicted_min_sep = float("inf")

        _, _, terminations, _, info = core.step({}, compute_predictions=False)

        self.assertFalse(any(terminations.values()))
        components = info["reward_components"]["A1"]
        expected_local_risk = (SAFE_R * PAIR_RISK_BUFFER - 8.0) / (SAFE_R * PAIR_RISK_BUFFER)
        self.assertAlmostEqual(components["predicted_traffic_risk"], 0.0)
        self.assertAlmostEqual(components["local_traffic_risk"], expected_local_risk)
        self.assertAlmostEqual(components["traffic_risk"], expected_local_risk)
        self.assertAlmostEqual(components["progress_risk_gate"], expected_local_risk)
        self.assertAlmostEqual(
            components["merge_back_scale"],
            (REWARD_MERGE_BACK_RISK_THRESHOLD - expected_local_risk)
            / REWARD_MERGE_BACK_RISK_THRESHOLD,
        )
        self.assertAlmostEqual(
            components["progress_safety_scale"],
            1.0 - REWARD_PROGRESS_HAZARD_RELIEF * expected_local_risk,
        )
        self.assertAlmostEqual(
            components["safe_progress"],
            components["progress"] * components["progress_safety_scale"],
        )
        self.assertLess(components["cross_track_penalty_scale"], 1.0)

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
        previous_cross_track = abs(float(line_signed_cross_track(state.route.linestring, state.position)[0]))
        _, _, terminations, _, info = core.step({}, compute_predictions=False)

        self.assertFalse(any(terminations.values()))
        self.assertIsNone(info["failure_reason"])
        self.assertLess(core.weather_signed_clearance(state.position), 0.0)
        self.assertGreaterEqual(core.weather_terminal_signed_clearance(state.position), 0.0)

        components = info["reward_components"]["A1"]
        expected_reward = (
            REWARD_PROGRESS_SCALE * components["safe_progress"]
            - REWARD_STEP_PENALTY
            - REWARD_CROSS_TRACK_SCALE * components["relaxed_cross_track"]
            + REWARD_CROSS_TRACK_RECOVERY_SCALE * components["safe_cross_track_recovery"]
            - REWARD_DEST_HEADING_SCALE * components["safe_heading_penalty"]
            - REWARD_TRAFFIC_RISK_PENALTY * components["traffic_risk"]
            - REWARD_WEATHER_RISK_PENALTY * components["weather_risk"]
            + REWARD_TRAFFIC_RISK_REDUCTION_SCALE * components["traffic_risk_reduction"]
            + REWARD_WEATHER_RISK_REDUCTION_SCALE * components["weather_risk_reduction"]
            + components["merge_back_bonus"]
            + components["finish_bonus"]
        )
        self.assertAlmostEqual(core.reward, expected_reward, places=6)

    def test_collision_terminal_penalty_dominates_progress(self) -> None:
        core = MultiAgentSectorCore(num_agents=2, num_weather_cells=1)
        core.reset(seed=42, num_weather_cells=1)
        core.weather_cells = []
        core._sync_primary_weather_cell()

        for idx, agent_id in enumerate(("A1", "A2")):
            state = core.get_agent(agent_id)
            state.position = (40.0, 40.0 + idx * (core.safe_r * 0.5))
            state.destination = (90.0, 40.0 + idx * (core.safe_r * 0.5))
            state.heading_rad = 0.0
            state.launched = True
            state.active = True
            state.finished = False
            state.has_entered_sector = True
            state.inside_sector = True

        _, rewards, terminations, _, info = core.step({}, compute_predictions=False)

        self.assertEqual(info["failure_reason"], "collision")
        self.assertEqual(info["terminal_reward"], -REWARD_COLLISION_PENALTY)
        self.assertAlmostEqual(info["reward_components"]["A1"]["merge_back_scale"], 0.0)
        self.assertAlmostEqual(info["reward_components"]["A1"]["merge_back_bonus"], 0.0)
        self.assertLess(core.reward, -70.0)
        self.assertTrue(all(value < -70.0 for value in rewards.values()))
        self.assertTrue(all(terminations.values()))

    def test_collision_terminal_penalty_blames_colliding_agents_more_than_bystander(self) -> None:
        core = MultiAgentSectorCore(num_agents=3, num_weather_cells=1)
        core.reset(seed=42, num_weather_cells=1)
        core.weather_cells = []
        core._sync_primary_weather_cell()

        positions = {
            "A1": (40.0, 40.0),
            "A2": (40.0, 40.0 + core.safe_r * 0.5),
            "A3": (10.0, 10.0),
        }
        for agent_id, position in positions.items():
            state = core.get_agent(agent_id)
            state.position = position
            state.destination = (90.0, position[1])
            state.heading_rad = 0.0
            state.launched = True
            state.active = True
            state.finished = False
            state.has_entered_sector = True
            state.inside_sector = True

        _, _, _, _, info = core.step({}, compute_predictions=False)

        shared_penalty = -REWARD_COLLISION_PENALTY / math.sqrt(3.0)
        terminal_rewards = info["agent_terminal_rewards"]
        self.assertEqual(info["failure_reason"], "collision")
        self.assertAlmostEqual(terminal_rewards["A1"], -REWARD_COLLISION_PENALTY)
        self.assertAlmostEqual(terminal_rewards["A2"], -REWARD_COLLISION_PENALTY)
        self.assertAlmostEqual(terminal_rewards["A3"], shared_penalty)
        self.assertAlmostEqual(
            info["terminal_reward"],
            (-2.0 * REWARD_COLLISION_PENALTY + shared_penalty) / 3.0,
        )

    def test_truncation_penalty_scales_with_remaining_distance_for_unfinished_aircraft(self) -> None:
        core = MultiAgentSectorCore(num_agents=1, max_step=1, num_weather_cells=1)
        core.reset(seed=42, num_weather_cells=1)
        core.weather_cells = []
        core._sync_primary_weather_cell()

        state = core.get_agent("A1")
        state.position = (20.0, 20.0)
        state.destination = (90.0, 20.0)
        state.heading_rad = 0.0
        state.launched = True
        state.active = True
        state.finished = False
        state.has_entered_sector = True
        state.inside_sector = True

        _, _, terminations, truncations, info = core.step({}, compute_predictions=False)

        route_length = max(float(state.route.linestring.length), core.goal_radius, 1e-6)
        remaining_fraction = min(max(core.distance_to_destination(state) / route_length, 0.0), 1.0)
        expected_penalty = (
            -REWARD_TRUNCATION_PENALTY
            - REWARD_TRUNCATION_DISTANCE_PENALTY_SCALE * remaining_fraction
        )

        self.assertEqual(info["failure_reason"], "truncated")
        self.assertTrue(info["episode_truncated"])
        self.assertFalse(any(terminations.values()))
        self.assertTrue(all(truncations.values()))
        self.assertAlmostEqual(info["agent_terminal_rewards"]["A1"], expected_penalty)
        self.assertAlmostEqual(info["terminal_reward"], expected_penalty)

    def test_short_rollout_update_clears_buffer(self) -> None:
        env = _make_env(max_step=30)
        observations, info = _advance_to_decision_agent(env, seed=42)
        agent = _make_agent(env)
        done = bool(info["__common__"]["episode_done"])
        truncated = bool(info["__common__"]["episode_truncated"])

        for _ in range(3):
            if done or truncated:
                break
            active_ids = list(env.agents)
            if not active_ids:
                observations, _, _, _, info = env.step({})
                common = info["__common__"]
                done = bool(common["episode_done"])
                truncated = bool(common["episode_truncated"])
                continue
            actions, records, _ = agent.select_actions(
                global_state=env.state(),
                local_observations=observations,
                active_agent_ids=active_ids,
                episode_id=0,
            )
            observations, _, _, _, info = env.step(actions)
            common = info["__common__"]
            team_terminal = bool(common["episode_done"] or common["episode_truncated"])
            post_active = set(env.agents)
            terminals = {
                agent_id: bool(team_terminal or agent_id not in post_active)
                for agent_id in active_ids
            }
            agent.store_outcomes(
                records,
                reward=common.get("agent_dense_rewards", common["agent_rewards"]),
                terminals=terminals,
            )
            if team_terminal:
                agent.add_terminal_bonus(
                    0,
                    common.get("agent_terminal_rewards", common.get("terminal_reward", 0.0)),
                )
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
