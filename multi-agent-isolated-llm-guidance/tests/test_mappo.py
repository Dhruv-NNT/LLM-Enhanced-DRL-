from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from configs import (  # noqa: E402
    ACTION_BINS,
    USE_LLM_GUIDED_TRAINING,
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
from rl_llm_multi import GuidanceAdvice, LLMGuidanceProvider, MAPPO, MultiAgentParallelEnv, MultiAgentSectorCore, WeatherCell, shadow_evaluate_actions  # noqa: E402
from rl_llm_multi.llm import _write_guidance_artifact_record  # noqa: E402
from rl_llm_multi.mappo import evaluate_policy  # noqa: E402
from rl_llm_multi.memory import DecisionMemoryStore  # noqa: E402
from rl_llm_multi.utils import action_idx_to_deg, deg_to_action_idx, line_signed_cross_track, weather_axis_units  # noqa: E402
from train import _accumulate_reward_debug, _append_advisory_memory_cases, _apply_memory_corrections, _build_llm_metadata_by_agent, _current_llm_loss_weight, _hybrid_llm_return_weight, _llm_guidance_active_for_episode, _llm_return_weight, _log_episode_reward_debug, _resolve_llm_memory_path, _resolve_run_paths, _write_jsonl_rows_with_cap, _write_run_hyperparameters_file  # noqa: E402


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


class _FakeShadowState:
    has_entered_sector = True


class _FakeShadowCore:
    def __init__(self) -> None:
        self.active_agent_ids = ["A1", "A2"]
        self.agent_states = {agent_id: _FakeShadowState() for agent_id in self.active_agent_ids}
        self.step_calls: list[dict[str, int]] = []
        self.last_team_done = False
        self.last_team_truncated = False
        self.n_step = 0

    def clone(self):
        return copy.deepcopy(self)

    def get_agent(self, agent_id: str):
        return self.agent_states[agent_id]

    def step(self, action_dict, *, compute_predictions: bool = True):
        self.step_calls.append(dict(action_dict))
        self.n_step += 1
        self.last_team_done = True
        rewards = {agent_id: float(action_dict.get(agent_id, 0)) for agent_id in self.active_agent_ids}
        return {}, rewards, {agent_id: True for agent_id in self.active_agent_ids}, {}, {"agent_rewards": rewards}


class _UnusedShadowAgent:
    def select_actions(self, *args, **kwargs):
        raise AssertionError("deterministic continuation should not run for horizon=1")


class _FakeGuidanceController:
    def __init__(self) -> None:
        self.last_normalized = {}
        self.reset_called = False
        self.advisory_update_called = False
        self.update_called = False

    def reset(self) -> None:
        self.reset_called = True

    def advisory_update(self, core, *, step: int, latest_frame_path=None, use_vision: bool = False):
        self.advisory_update_called = True
        self.last_normalized = {
            "answer": {
                "actions": {
                    "A1": {
                        "call_name": "EXECUTE_TURN",
                        "source": "llm",
                        "requested_heading_change_deg": 9,
                    }
                }
            },
            "debug": {
                "stage_by_agent": {"A1": "EXECUTE_TURN"},
                "fallback_agents": [],
                "retrieved_memories": [
                    {
                        "agent": "A1",
                        "action": -15,
                        "corrected": -25,
                        "memory_ref": {"episode_id": "ep1", "case_id": "case1"},
                    }
                ],
            },
        }
        return {"A1": 10}

    def update(self, core, *, step: int, latest_frame_path=None, use_vision: bool = False):
        self.update_called = True
        return self.advisory_update(
            core,
            step=step,
            latest_frame_path=latest_frame_path,
            use_vision=use_vision,
        )


class _CapturingMemoryStore:
    def __init__(self) -> None:
        self.calls = []

    def append_cases(self, *args, **kwargs):
        self.calls.append(kwargs)
        return [
            {"agent": str(item["agent_id"]), "case_id": f"case_{item['agent_id']}"}
            for item in kwargs.get("actionable", [])
        ]


class _UpdatingMemoryStore:
    def __init__(self) -> None:
        self.updates = []

    def update_case_result(self, memory_ref, *, corrected: int, note: str) -> bool:
        self.updates.append({"memory_ref": memory_ref, "corrected": corrected, "note": note})
        return True


class MAPPOTests(unittest.TestCase):
    def test_llm_guided_training_is_disabled_by_default(self) -> None:
        self.assertFalse(USE_LLM_GUIDED_TRAINING)

    def test_llm_guidance_episode_latch_uses_episode_start_step(self) -> None:
        with patch("train.USE_LLM_GUIDED_TRAINING", True), patch("train.LLM_GUIDANCE_START_STEP", 100), patch("train.LLM_GUIDANCE_END_STEP", 200):
            self.assertFalse(_llm_guidance_active_for_episode(99))
            self.assertTrue(_llm_guidance_active_for_episode(100))
            self.assertTrue(_llm_guidance_active_for_episode(199))
            self.assertFalse(_llm_guidance_active_for_episode(200))

    def test_llm_degree_label_maps_to_action_index(self) -> None:
        for degrees in ACTION_BINS:
            index = deg_to_action_idx(degrees, ACTION_BINS)
            self.assertEqual(action_idx_to_deg(index, ACTION_BINS), degrees)

    def test_run_paths_are_derived_from_log_dir(self) -> None:
        args = SimpleNamespace(log_dir="runs/E0_mappo_seed0", best_model_path=None, last_ckpt_path=None)

        paths = _resolve_run_paths(args)

        self.assertEqual(paths.run_dir, Path("runs/E0_mappo_seed0"))
        self.assertEqual(paths.tensorboard_dir, Path("runs/E0_mappo_seed0/tensorboard"))
        self.assertEqual(paths.tensorboard_run_dir, Path("runs/E0_mappo_seed0/tensorboard/E0_mappo_seed0"))
        self.assertEqual(paths.weights_dir, Path("runs/E0_mappo_seed0/weights"))
        self.assertEqual(paths.best_model_path, Path("runs/E0_mappo_seed0/weights/best_model.pt"))
        self.assertEqual(paths.last_ckpt_path, Path("runs/E0_mappo_seed0/weights/last_checkpoint.pt"))

    def test_llm_memory_path_is_derived_from_log_dir(self) -> None:
        with patch("train.LLM_MEMORY_PATH", None), patch("train.LLM_GUIDANCE_MEMORY_PATH", None), patch("train.LLM_GUIDANCE_MEMORY_ISOLATED_PER_RUN", True), patch("train.LLM_MEMORY_RUN_ID", None):
            path = _resolve_llm_memory_path(Path("runs/E2_full_llm_seed0"), seed=0, num_agents=3, num_weather_cells=1)
        self.assertEqual(path, Path("runs/E2_full_llm_seed0/memory/decision_memory.json"))

    def test_jsonl_row_writer_keeps_latest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.jsonl"
            _write_jsonl_rows_with_cap(path, [{"idx": 1}, {"idx": 2}], max_rows=2)
            _write_jsonl_rows_with_cap(path, [{"idx": 3}], max_rows=2)
            lines = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(lines, [{"idx": 2}, {"idx": 3}])

    def test_guidance_artifacts_can_be_capped_to_latest_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("rl_llm_multi.llm.LLM_GUIDANCE_ARTIFACTS_ENABLED", True), patch("rl_llm_multi.llm.LLM_GUIDANCE_ARTIFACT_MAX_CALLS", 2):
                for idx in range(3):
                    tag = f"t{idx:03d}_global_guidance"
                    _write_guidance_artifact_record(
                        tmpdir,
                        tag=tag,
                        prompt_text=f"prompt {idx}",
                        raw_text=f"response {idx}",
                        payload={"idx": idx},
                        index_row={"tag": tag, "idx": idx},
                    )
            names = sorted(path.name for path in Path(tmpdir).iterdir())
            index_lines = [json.loads(line) for line in (Path(tmpdir) / "_index.jsonl").read_text().splitlines()]

        self.assertEqual([row["idx"] for row in index_lines], [1, 2])
        self.assertFalse(any(name.startswith("t000_global_guidance") for name in names))
        self.assertTrue(any(name.startswith("t001_global_guidance") for name in names))
        self.assertTrue(any(name.startswith("t002_global_guidance") for name in names))

    def test_run_hyperparameters_file_is_written_once_with_rewards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                log_dir=str(Path(tmpdir) / "run"),
                best_model_path=None,
                last_ckpt_path=None,
                seed=7,
                num_agents=2,
                num_weather_cells=1,
            )
            paths = _resolve_run_paths(args)
            hparams_path = paths.run_dir / "run_hyperparameters.txt"
            _write_run_hyperparameters_file(
                hparams_path,
                args=args,
                run_paths=paths,
                llm_memory_path=paths.run_dir / "memory" / "decision_memory.json",
                reset_options={"num_agents": 2, "route_ids": None, "num_weather_cells": 1},
            )
            first_text = hparams_path.read_text()
            _write_run_hyperparameters_file(
                hparams_path,
                args=args,
                run_paths=paths,
                llm_memory_path=paths.run_dir / "memory" / "decision_memory.json",
                reset_options={"num_agents": 3},
            )
            second_text = hparams_path.read_text()

        self.assertEqual(first_text, second_text)
        self.assertIn("Run Hyperparameters", first_text)
        self.assertIn("Command Line Arguments", first_text)
        self.assertIn("Reward Hyperparameters", first_text)
        self.assertIn("REWARD_COLLISION_PENALTY", first_text)
        self.assertIn("tensorboard_dir", first_text)
        self.assertIn("tensorboard_run_dir", first_text)
        self.assertIn("llm_memory_path", first_text)

    def test_reward_debug_logs_requested_tensorboard_tags(self) -> None:
        class _FakeWriter:
            def __init__(self) -> None:
                self.scalars = {}

            def add_scalar(self, tag, value, step):
                self.scalars[str(tag)] = (float(value), int(step))

        sums = {}
        counts = {}
        common = {
            "agent_dense_rewards": {"A1": 1.0, "A2": 3.0},
            "terminal_reward": 0.0,
            "reward_components": {
                "A1": {"cross_track": 2.0, "finish_bonus": 0.0, "dense_reward": 1.0},
                "A2": {"cross_track": 4.0, "finish_bonus": 10.0, "dense_reward": 3.0},
            },
        }
        _accumulate_reward_debug(common, sums, counts)
        writer = _FakeWriter()

        _log_episode_reward_debug(writer, sums, counts, common, episode=7)

        self.assertIn("train/reward/terminal_episode", writer.scalars)
        self.assertIn("train/reward/mean_step_behavior_reward", writer.scalars)
        self.assertIn("train/reward_component/cross_track_mean", writer.scalars)
        self.assertIn("train/reward_component/finish_bonus_mean", writer.scalars)
        self.assertEqual(writer.scalars["train/reward/terminal_episode"], (0.0, 7))
        self.assertEqual(writer.scalars["train/reward/mean_step_behavior_reward"], (2.0, 7))
        self.assertEqual(writer.scalars["train/reward_component/cross_track_mean"], (3.0, 7))
        self.assertEqual(writer.scalars["train/reward_component/finish_bonus_mean"], (5.0, 7))

    def test_structured_guidance_provider_returns_metadata_without_overrides(self) -> None:
        controller = _FakeGuidanceController()
        provider = LLMGuidanceProvider(controller=controller)

        self.assertEqual(
            provider.actions_for_step(_FakeShadowCore(), step=0, proposed_actions={"A1": 0}),
            {},
        )
        advice = provider.advice_for_step(
            _FakeShadowCore(),
            step=0,
            proposed_actions={"A1": deg_to_action_idx(0, ACTION_BINS)},
        )

        self.assertEqual(advice["A1"].llm_action_deg, 10)
        self.assertEqual(advice["A1"].llm_action_idx, deg_to_action_idx(10, ACTION_BINS))
        self.assertEqual(advice["A1"].phase, "EXECUTE_TURN")
        self.assertEqual(advice["A1"].memory_action_deg, -25)
        self.assertEqual(advice["A1"].memory_ref, {"episode_id": "ep1", "case_id": "case1"})
        self.assertTrue(controller.advisory_update_called)
        self.assertFalse(controller.update_called)

    def test_llm_return_weight_preserves_improvement_magnitude(self) -> None:
        self.assertEqual(_llm_return_weight(1.0, 1.0), 0.0)
        small = _llm_return_weight(1.10, 1.0)
        large = _llm_return_weight(1.90, 1.0)
        self.assertGreater(small, 0.0)
        self.assertGreater(large, small)

    def test_llm_loss_weight_decay_respects_minimum(self) -> None:
        self.assertGreaterEqual(_current_llm_loss_weight(10**9), 0.0)

    def test_shadow_evaluation_uses_joint_llm_swap_and_preserves_core(self) -> None:
        core = _FakeShadowCore()
        proposed = {"A1": deg_to_action_idx(0, ACTION_BINS), "A2": deg_to_action_idx(0, ACTION_BINS)}
        advice = {
            "A1": GuidanceAdvice(
                agent_id="A1",
                llm_action_deg=10,
                llm_action_idx=deg_to_action_idx(10, ACTION_BINS),
                phase="EXECUTE_TURN",
                memory_action_deg=20,
                memory_action_idx=deg_to_action_idx(20, ACTION_BINS),
                memory_ref={"episode_id": "ep1", "case_id": "case1"},
            ),
            "A2": GuidanceAdvice(
                agent_id="A2",
                llm_action_deg=-10,
                llm_action_idx=deg_to_action_idx(-10, ACTION_BINS),
                phase="EXECUTE_TURN",
            ),
        }

        result = shadow_evaluate_actions(
            core,
            _UnusedShadowAgent(),
            proposed_actions=proposed,
            guidance_by_agent=advice,
            horizon=1,
            gamma=0.99,
        )

        self.assertEqual(core.step_calls, [])
        self.assertEqual(result.mappo_returns, {"A1": 0.0, "A2": 0.0})
        self.assertEqual(result.llm_returns, {"A1": 10.0, "A2": -10.0})
        self.assertEqual(result.mappo_joint_return, 0.0)
        self.assertEqual(result.llm_joint_return, 0.0)
        self.assertEqual(result.memory_joint_return, 10.0)
        self.assertEqual(result.memory_agent_ids, ["A1"])

    def test_hybrid_llm_return_weight_combines_agent_and_joint_weights(self) -> None:
        agent_weight, joint_weight, hybrid_weight = _hybrid_llm_return_weight(1.2, 1.0, 2.0, 1.0)

        self.assertEqual(agent_weight, _llm_return_weight(1.2, 1.0))
        self.assertEqual(joint_weight, _llm_return_weight(2.0, 1.0))
        self.assertAlmostEqual(hybrid_weight, 0.65 * agent_weight + 0.35 * joint_weight)

    def test_llm_metadata_uses_hybrid_weight(self) -> None:
        advice = {
            "A1": GuidanceAdvice(
                agent_id="A1",
                llm_action_deg=10,
                llm_action_idx=deg_to_action_idx(10, ACTION_BINS),
                phase="EXECUTE_TURN",
            )
        }
        shadow = SimpleNamespace(
            mappo_returns={"A1": 0.0},
            llm_returns={"A1": 0.2},
            memory_returns={},
            mappo_joint_return=0.0,
            llm_joint_return=1.0,
            memory_joint_return=0.0,
        )
        corrections = SimpleNamespace(updates={}, note_sources={}, attribution_modes={})

        metadata = _build_llm_metadata_by_agent(advice, shadow, corrections)

        self.assertGreater(metadata["A1"]["llm_return_weight_agent"], 0.0)
        self.assertGreater(metadata["A1"]["llm_return_weight_joint"], 0.0)
        self.assertEqual(metadata["A1"]["llm_return_weight"], metadata["A1"]["llm_return_weight_hybrid"])

    def test_advisory_memory_write_only_when_llm_beats_mappo(self) -> None:
        store = _CapturingMemoryStore()
        guidance = SimpleNamespace(
            controller=SimpleNamespace(
                last_normalized={
                    "answer": {
                        "actions": {
                            "A1": {"call_name": "EXECUTE_TURN", "source": "llm", "rationale": "clearer turn"},
                            "A2": {"call_name": "EXECUTE_TURN", "source": "llm", "rationale": "worse turn"},
                        }
                    },
                    "debug": {
                        "stage_by_agent": {"A1": "EXECUTE_TURN", "A2": "EXECUTE_TURN"},
                        "call_reason_by_agent": {"A1": "SECTOR_ENTRY", "A2": "SECTOR_ENTRY"},
                    },
                    "threat_rows_by_agent": {},
                    "preview_rows_by_agent": {},
                }
            )
        )
        advice = {
            "A1": GuidanceAdvice("A1", 10, deg_to_action_idx(10, ACTION_BINS), "EXECUTE_TURN"),
            "A2": GuidanceAdvice("A2", -10, deg_to_action_idx(-10, ACTION_BINS), "EXECUTE_TURN"),
        }
        shadow = SimpleNamespace(
            mappo_returns={"A1": 0.0, "A2": 1.0},
            llm_returns={"A1": 0.2, "A2": 0.9},
        )

        writes = _append_advisory_memory_cases(
            store,
            guidance=guidance,
            core=_FakeShadowCore(),
            step=5,
            advice_by_agent=advice,
            shadow_result=shadow,
        )

        self.assertEqual(writes, {"A1": True})
        self.assertEqual([item["agent_id"] for item in store.calls[0]["actionable"]], ["A1"])
        self.assertEqual(store.calls[0]["final_actions"]["A1"]["heading_change_deg"], 10)

    def test_advisory_memory_write_respects_eligible_agent_ids(self) -> None:
        store = _CapturingMemoryStore()
        guidance = SimpleNamespace(
            controller=SimpleNamespace(
                last_normalized={
                    "answer": {
                        "actions": {
                            "A1": {"call_name": "EXECUTE_TURN", "source": "llm"},
                            "A2": {"call_name": "EXECUTE_TURN", "source": "llm"},
                        }
                    },
                    "debug": {
                        "eligible_agent_ids": ["A1"],
                        "stage_by_agent": {"A1": "EXECUTE_TURN", "A2": "EXECUTE_TURN"},
                        "call_reason_by_agent": {"A1": "SECTOR_ENTRY", "A2": "SECTOR_ENTRY"},
                    },
                    "threat_rows_by_agent": {},
                    "preview_rows_by_agent": {},
                }
            )
        )
        advice = {
            "A1": GuidanceAdvice("A1", 10, deg_to_action_idx(10, ACTION_BINS), "EXECUTE_TURN"),
            "A2": GuidanceAdvice("A2", 20, deg_to_action_idx(20, ACTION_BINS), "EXECUTE_TURN"),
        }
        shadow = SimpleNamespace(
            mappo_returns={"A1": 0.0, "A2": 0.0},
            llm_returns={"A1": 0.2, "A2": 0.2},
        )

        writes = _append_advisory_memory_cases(
            store,
            guidance=guidance,
            core=_FakeShadowCore(),
            step=5,
            advice_by_agent=advice,
            shadow_result=shadow,
        )

        self.assertEqual(writes, {"A1": True})
        self.assertEqual([item["agent_id"] for item in store.calls[0]["actionable"]], ["A1"])
        self.assertNotIn("A2", store.calls[0]["final_actions"])

    def test_guided_memory_store_caps_latest_episodes(self) -> None:
        core = MultiAgentSectorCore(num_agents=2, num_weather_cells=1)
        core.reset(seed=42, num_weather_cells=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DecisionMemoryStore(Path(tmpdir) / "decision_memory.json", max_episodes=300)
            for idx in range(305):
                store.append_cases(
                    core,
                    episode_id=f"ep{idx:03d}",
                    step=idx,
                    actionable=[{"agent_id": "A1", "call_name": "EXECUTE_TURN", "call_reason": "SECTOR_ENTRY"}],
                    final_actions={"A1": {"heading_change_deg": 0}},
                    threat_rows_by_agent={"A1": []},
                    preview_rows_by_agent={"A1": []},
                )
            payload = store.load()

        self.assertEqual(len(payload["episodes"]), 300)
        self.assertEqual(payload["episodes"][0]["episode_id"], "ep005")
        self.assertEqual(payload["episodes"][-1]["episode_id"], "ep304")

    def test_single_memory_correction_updates_corrected_and_note(self) -> None:
        store = _UpdatingMemoryStore()
        advice = {
            "A1": GuidanceAdvice(
                agent_id="A1",
                llm_action_deg=10,
                llm_action_idx=deg_to_action_idx(10, ACTION_BINS),
                phase="EXECUTE_TURN",
                memory_action_deg=-15,
                memory_action_idx=deg_to_action_idx(-15, ACTION_BINS),
                memory_ref={"episode_id": "ep1", "case_id": "case1"},
            )
        }
        shadow = SimpleNamespace(
            memory_agent_ids=["A1"],
            mappo_joint_return=0.0,
            llm_joint_return=2.0,
            memory_joint_return=0.0,
            memory_attribution_joint_returns={},
        )

        with patch("train._generate_memory_correction_note", return_value=("traffic buffer improved", "llm")):
            result = _apply_memory_corrections(
                store,
                advice_by_agent=advice,
                shadow_result=shadow,
                policy_actions={"A1": deg_to_action_idx(0, ACTION_BINS)},
                llm_call_payload={"debug": {"call_reason_by_agent": {"A1": "SECTOR_ENTRY"}}},
            )

        self.assertEqual(result.sources, {"A1": "llm"})
        self.assertEqual(result.note_sources, {"A1": "llm"})
        self.assertEqual(store.updates[0]["corrected"], 10)
        self.assertEqual(store.updates[0]["note"], "traffic buffer improved")

    def test_multi_memory_correction_uses_one_at_a_time_attribution(self) -> None:
        store = _UpdatingMemoryStore()
        advice = {
            "A1": GuidanceAdvice("A1", 10, deg_to_action_idx(10, ACTION_BINS), "EXECUTE_TURN", memory_action_deg=-15, memory_action_idx=deg_to_action_idx(-15, ACTION_BINS), memory_ref={"episode_id": "ep1", "case_id": "case1"}),
            "A2": GuidanceAdvice("A2", 20, deg_to_action_idx(20, ACTION_BINS), "EXECUTE_TURN", memory_action_deg=-20, memory_action_idx=deg_to_action_idx(-20, ACTION_BINS), memory_ref={"episode_id": "ep1", "case_id": "case2"}),
        }
        shadow = SimpleNamespace(
            memory_agent_ids=["A1", "A2"],
            mappo_joint_return=0.0,
            llm_joint_return=2.0,
            memory_joint_return=-1.0,
            memory_attribution_joint_returns={"A1": 0.0, "A2": 1.95},
        )

        with patch("train._generate_memory_correction_note", return_value=("LLM keeps more separation", "fallback")):
            result = _apply_memory_corrections(
                store,
                advice_by_agent=advice,
                shadow_result=shadow,
                policy_actions={"A1": deg_to_action_idx(0, ACTION_BINS), "A2": deg_to_action_idx(0, ACTION_BINS)},
                llm_call_payload={},
            )

        self.assertEqual(result.sources, {"A1": "llm"})
        self.assertEqual(result.attribution_modes["A1"], "one_at_a_time")
        self.assertEqual(len(store.updates), 1)
        self.assertEqual(store.updates[0]["memory_ref"], {"episode_id": "ep1", "case_id": "case1"})

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

    def test_labeled_rollout_adds_llm_loss_metrics(self) -> None:
        env = _make_env(max_step=30)
        observations, info = _advance_to_decision_agent(env, seed=42)
        self.assertTrue(env.agents)
        agent = _make_agent(env)
        active_ids = list(env.agents)
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
        labeled_agent = active_ids[0]
        action_idx = int(actions[labeled_agent])
        metadata = {
            labeled_agent: {
                "llm_label_available": True,
                "llm_action_idx": action_idx,
                "llm_action_deg": action_idx_to_deg(action_idx, ACTION_BINS),
                "llm_phase": "EXECUTE_TURN",
                "llm_return_weight": 1.0,
            }
        }
        agent.store_outcomes(
            records,
            reward=common.get("agent_dense_rewards", common["agent_rewards"]),
            terminals=terminals,
            llm_metadata_by_agent=metadata,
        )

        metrics = agent.update()

        self.assertGreater(metrics["llm_label_count"], 0.0)
        self.assertGreaterEqual(metrics["llm_loss"], 0.0)
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
