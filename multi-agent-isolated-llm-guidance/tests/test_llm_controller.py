from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shapely.geometry import LineString


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from configs import MAX_AGENTS, SAFE_R  # noqa: E402
from rl_llm_multi import (  # noqa: E402
    GlobalLangGraphGuidanceController,
    JointGuidanceEnv,
    MultiAgentThreeCallController,
)


def _answer_json(turn_deg: int) -> str:
    return json.dumps(
        {
            "answer": {
                "scenario_summary": "",
                "technical_summary": "",
                "rationale": "",
                "maneuver": {"heading_change_deg": int(turn_deg)},
            }
        }
    )


def _global_answer_json(actions: dict[str, int]) -> str:
    return json.dumps(
        {
            "answer": {
                "scenario_summary": "",
                "actions": {
                    agent_id: {
                        "call_name": "EXECUTE_TURN",
                        "rationale": "",
                        "maneuver": {"heading_change_deg": int(turn_deg)},
                    }
                    for agent_id, turn_deg in actions.items()
                },
            }
        }
    )


def _make_state(
    agent_id: str,
    *,
    position: tuple[float, float],
    destination: tuple[float, float] = (90.0, 10.0),
    heading_deg: float = 0.0,
    predicted_pair_loss_step: int | None = None,
    predicted_weather_entry_step: int | None = None,
    predicted_boundary_exit_step: int | None = None,
    predicted_min_sep: float = 999.0,
    predicted_weather_clearance: float = 999.0,
    safe_streak: int = 0,
    boundary_warning_active: bool = False,
    hazard_predicted: bool = False,
    launched: bool = True,
    finished: bool = False,
    has_entered_sector: bool = True,
    sector_entry_step: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=agent_id,
        route=SimpleNamespace(linestring=LineString([(0.0, 0.0), (100.0, 0.0)])),
        position=tuple(position),
        destination=tuple(destination),
        heading_rad=math.radians(float(heading_deg)),
        safe_streak=int(safe_streak),
        boundary_warning_active=bool(boundary_warning_active),
        predicted_pair_loss_step=predicted_pair_loss_step,
        predicted_weather_entry_step=predicted_weather_entry_step,
        predicted_boundary_exit_step=predicted_boundary_exit_step,
        predicted_min_sep=float(predicted_min_sep),
        predicted_weather_clearance=float(predicted_weather_clearance),
        hazard_predicted=bool(hazard_predicted),
        launched=bool(launched),
        finished=bool(finished),
        has_entered_sector=bool(has_entered_sector),
        inside_sector=True,
        sector_entry_step=sector_entry_step,
        threat_agents=(),
    )


class FakeCore:
    def __init__(self, states: list[SimpleNamespace]) -> None:
        self.agent_states = {state.agent_id: state for state in states}
        self.active_agent_ids = [
            state.agent_id for state in states if state.launched and not state.finished
        ]
        self.safe_r = SAFE_R
        self.n_step = 5

    def get_agent(self, agent_id: str) -> SimpleNamespace:
        return self.agent_states[agent_id]

    def distance_to_destination(self, state: SimpleNamespace) -> float:
        return float(
            math.hypot(
                state.destination[0] - state.position[0],
                state.destination[1] - state.position[1],
            )
        )

    def boundary_distance(self, state: SimpleNamespace) -> float | None:
        return 7.5 if state.inside_sector else None

    def nearest_neighbor(self, agent_id: str) -> tuple[str | None, float]:
        own = self.get_agent(agent_id)
        best_id = None
        best_dist = float("inf")
        for other_id in self.active_agent_ids:
            if other_id == agent_id:
                continue
            other = self.get_agent(other_id)
            dist = math.hypot(
                own.position[0] - other.position[0],
                own.position[1] - other.position[1],
            )
            if dist < best_dist:
                best_dist = dist
                best_id = other_id
        return best_id, best_dist

    def _bearing_to_neighbor_deg(self, agent_id: str, other_id: str | None) -> float:
        if other_id is None:
            return 0.0
        own = self.get_agent(agent_id)
        other = self.get_agent(other_id)
        absolute = math.degrees(
            math.atan2(
                other.position[1] - own.position[1],
                other.position[0] - own.position[0],
            )
        )
        own_deg = math.degrees(own.heading_rad)
        return ((absolute - own_deg + 180.0) % 360.0) - 180.0

    def weather_dict(self) -> dict[str, object]:
        return {
            "center": [50.0, 50.0],
            "major_radius_nm": 10.0,
            "minor_radius_nm": 6.0,
            "angle_rad": 0.0,
            "motion_heading_rad": 0.0,
            "speed_units_per_step": 1.0,
            "major_growth_nm_per_step": 0.0,
        }

    def weather_signed_clearance(self, position: tuple[float, float]) -> float:
        return 8.0


class ControllerTests(unittest.TestCase):
    def test_actionable_sort_key_prefers_earliest_event_then_min_sep_then_id(self) -> None:
        controller = MultiAgentThreeCallController()
        core = FakeCore(
            [
                _make_state(
                    "A1",
                    position=(10.0, 10.0),
                    predicted_weather_entry_step=4,
                    predicted_min_sep=3.0,
                ),
                _make_state(
                    "A2",
                    position=(20.0, 10.0),
                    predicted_pair_loss_step=4,
                    predicted_min_sep=2.0,
                ),
                _make_state(
                    "A3",
                    position=(30.0, 10.0),
                    predicted_boundary_exit_step=2,
                    predicted_min_sep=10.0,
                ),
            ]
        )
        actionable = [
            {"agent_id": "A1", "call_name": "EXECUTE_TURN"},
            {"agent_id": "A2", "call_name": "EXECUTE_TURN"},
            {"agent_id": "A3", "call_name": "MERGE_BACK"},
        ]

        ordered = sorted(actionable, key=lambda item: controller._actionable_sort_key(core, item))
        self.assertEqual([item["agent_id"] for item in ordered], ["A3", "A2", "A1"])

    def test_threat_sort_key_prefers_loss_then_min_sep_then_distance(self) -> None:
        controller = MultiAgentThreeCallController()
        rows = [
            {"intruder_id": "A2", "predicted_loss_step": None, "predicted_min_sep": 1.0, "current_distance": 8.0},
            {"intruder_id": "A3", "predicted_loss_step": 5, "predicted_min_sep": 4.0, "current_distance": 4.0},
            {"intruder_id": "A4", "predicted_loss_step": 5, "predicted_min_sep": 2.0, "current_distance": 6.0},
            {"intruder_id": "A5", "predicted_loss_step": None, "predicted_min_sep": 0.5, "current_distance": 3.0},
        ]

        ordered = sorted(rows, key=controller._threat_sort_key)
        self.assertEqual([row["intruder_id"] for row in ordered], ["A4", "A3", "A5", "A2"])

    def test_merge_back_candidate_sort_prefers_recovery_progress_after_safety(self) -> None:
        controller = MultiAgentThreeCallController()
        rows = [
            {
                "heading_change_deg": 0,
                "safe_over_preview": True,
                "pair_loss_step": None,
                "weather_entry_step": None,
                "boundary_exit_step": None,
                "progress_to_destination": 2.0,
                "cross_track_reduction": -1.0,
                "end_heading_error_to_dest_deg": 4.0,
            },
            {
                "heading_change_deg": 10,
                "safe_over_preview": True,
                "pair_loss_step": None,
                "weather_entry_step": None,
                "boundary_exit_step": None,
                "progress_to_destination": 6.0,
                "cross_track_reduction": 2.0,
                "end_heading_error_to_dest_deg": 8.0,
            },
            {
                "heading_change_deg": 15,
                "safe_over_preview": False,
                "pair_loss_step": 2,
                "weather_entry_step": None,
                "boundary_exit_step": None,
                "progress_to_destination": 9.0,
                "cross_track_reduction": 4.0,
                "end_heading_error_to_dest_deg": 2.0,
            },
        ]

        ordered = sorted(rows, key=lambda row: controller._candidate_sort_key_for_call("MERGE_BACK", row))

        self.assertEqual([row["heading_change_deg"] for row in ordered], [10, 0, 15])

    def test_prompt_is_local_and_mentions_primary_and_other_threats(self) -> None:
        controller = MultiAgentThreeCallController()
        core = FakeCore(
            [
                _make_state("A1", position=(10.0, 10.0), heading_deg=0.0),
                _make_state("A2", position=(14.0, 10.0), heading_deg=180.0),
                _make_state("A3", position=(18.0, 12.0), heading_deg=90.0),
            ]
        )
        threat_rows = [
            {
                "intruder_id": "A2",
                "intruder_position": (14.0, 10.0),
                "intruder_heading_deg": 180.0,
                "current_distance": 4.0,
                "bearing_error_deg": 0.0,
                "predicted_loss_step": 2,
                "predicted_min_sep": 3.0,
            },
            {
                "intruder_id": "A3",
                "intruder_position": (18.0, 12.0),
                "intruder_heading_deg": 90.0,
                "current_distance": 8.2,
                "bearing_error_deg": 15.0,
                "predicted_loss_step": None,
                "predicted_min_sep": 6.0,
            },
        ]
        preview_rows = [
            {
                "heading_change_deg": 5,
                "safe_over_preview": True,
                "pair_loss_step": None,
                "weather_entry_step": None,
                "boundary_exit_step": None,
                "min_sep_to_any": 6.0,
                "min_weather_clearance": 7.0,
                "end_heading_error_to_dest_deg": 3.0,
            }
        ]

        prompt = controller.builder.build_prompt(
            core,
            "A1",
            "EXECUTE_TURN",
            threat_rows,
            preview_rows,
            {"A9": 10},
        )

        self.assertIn("PRIMARY THREAT: A2", prompt)
        self.assertIn("OTHER THREATS:", prompt)
        self.assertIn("A3", prompt)
        self.assertIn("\"maneuver\": { \"heading_change_deg\": 0 }", prompt)
        self.assertNotIn("\"global_summary\"", prompt)
        self.assertNotIn("\"agents\"", prompt)
        self.assertNotIn("actionable agents listed below", prompt)

    def test_update_processes_actionable_agents_in_risk_order(self) -> None:
        core = FakeCore(
            [
                _make_state(
                    "A1",
                    position=(10.0, 10.0),
                    predicted_pair_loss_step=1,
                    predicted_min_sep=3.0,
                    hazard_predicted=True,
                ),
                _make_state(
                    "A2",
                    position=(20.0, 10.0),
                    predicted_pair_loss_step=3,
                    predicted_min_sep=4.0,
                    hazard_predicted=True,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = MultiAgentThreeCallController(save_dir=tmpdir)
            preview_calls: list[tuple[str, dict[str, int]]] = []

            def preview_side_effect(
                core_arg: FakeCore,
                agent_id: str,
                call_name: str,
                *,
                fixed_actions: dict[str, int] | None = None,
            ) -> list[dict[str, object]]:
                preview_calls.append((agent_id, dict(fixed_actions or {})))
                return [
                    {
                        "heading_change_deg": 5,
                        "safe_over_preview": True,
                        "pair_loss_step": None,
                        "weather_entry_step": None,
                        "boundary_exit_step": None,
                        "min_sep_to_any": 6.0,
                        "min_weather_clearance": 7.0,
                        "end_heading_error_to_dest_deg": 3.0,
                    }
                ]

            with patch.object(controller, "_ranked_threats", return_value=[]), patch.object(
                controller,
                "_preview_rows",
                side_effect=preview_side_effect,
            ), patch.object(controller.builder, "build_prompt", return_value="prompt"), patch(
                "rl_llm_multi.llm.ollama_invoke",
                return_value={
                    "llm_status": "ok",
                    "raw_text": _answer_json(5),
                    "error": "",
                    "used_vision": False,
                },
            ), patch.object(controller, "_deterministic_best_turn", return_value=0), patch.object(
                controller,
                "_log_local_call",
                return_value=None,
            ):
                actions = controller.choose_llm_actions(core, step=core.n_step)

        self.assertEqual(actions, {"A1": 5, "A2": 5})
        self.assertEqual(preview_calls[0], ("A1", {}))
        self.assertEqual(preview_calls[1], ("A2", {"A1": 5}))

    def test_invalid_json_uses_single_agent_fallback_turn(self) -> None:
        core = FakeCore(
            [
                _make_state(
                    "A1",
                    position=(10.0, 10.0),
                    predicted_pair_loss_step=1,
                    predicted_min_sep=3.0,
                    hazard_predicted=True,
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = MultiAgentThreeCallController(save_dir=tmpdir)
            with patch.object(controller, "_ranked_threats", return_value=[]), patch.object(
                controller,
                "_preview_rows",
                return_value=[],
            ), patch.object(controller.builder, "build_prompt", return_value="prompt"), patch.object(
                controller,
                "_deterministic_best_turn",
                return_value=15,
            ), patch(
                "rl_llm_multi.llm.ollama_invoke",
                return_value={
                    "llm_status": "ok",
                    "raw_text": "not-json",
                    "error": "",
                    "used_vision": False,
                },
            ), patch.object(controller, "_log_local_call", return_value=None):
                actions = controller.choose_llm_actions(core, step=core.n_step)

        self.assertEqual(actions, {"A1": 15})

    def test_boundary_warning_triggers_merge_back_without_safe_streak(self) -> None:
        controller = MultiAgentThreeCallController()
        core = FakeCore(
            [
                _make_state(
                    "A1",
                    position=(10.0, 10.0),
                    heading_deg=35.0,
                    predicted_boundary_exit_step=2,
                    boundary_warning_active=True,
                    safe_streak=0,
                )
            ]
        )
        ctrl_state = controller._controller_state("A1")
        ctrl_state.execute_called = True
        ctrl_state.stage = "WAIT_CLEAR"

        decision = controller._call_decision_for_agent(core, "A1")

        self.assertEqual(
            decision,
            {
                "agent_id": "A1",
                "call_name": "MERGE_BACK",
                "call_reason": "BOUNDARY_LOOKAHEAD",
            },
        )

    def test_annotations_include_entered_sector_and_phase_labels(self) -> None:
        core = FakeCore(
            [
                _make_state(
                    "A1",
                    position=(10.0, 10.0),
                    predicted_pair_loss_step=1,
                    predicted_min_sep=3.0,
                    hazard_predicted=True,
                    sector_entry_step=5,
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = MultiAgentThreeCallController(save_dir=tmpdir)
            with patch.object(controller, "_ranked_threats", return_value=[]), patch.object(
                controller,
                "_preview_rows",
                return_value=[],
            ), patch.object(controller.builder, "build_prompt", return_value="prompt"), patch(
                "rl_llm_multi.llm.ollama_invoke",
                return_value={
                    "llm_status": "ok",
                    "raw_text": _answer_json(5),
                    "error": "",
                    "used_vision": False,
                },
            ), patch.object(controller, "_log_local_call", return_value=None):
                controller.choose_llm_actions(core, step=5)

        self.assertEqual(controller.last_annotations, ["A1: ENTERED SECTOR", "A1: EXECUTE TURN"])

    def test_global_controller_excludes_agents_before_sector_entry(self) -> None:
        core = FakeCore(
            [
                _make_state("A1", position=(10.0, 10.0), has_entered_sector=True),
                _make_state("A2", position=(20.0, 10.0), has_entered_sector=False),
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = GlobalLangGraphGuidanceController(save_dir=tmpdir)
            with patch.object(controller, "_ranked_threats", return_value=[]), patch.object(
                controller,
                "_preview_rows",
                return_value=[],
            ), patch.object(controller, "_deterministic_best_turn", return_value=0), patch(
                "rl_llm_multi.llm.ollama_invoke",
                return_value={
                    "llm_status": "ok",
                    "raw_text": _global_answer_json({"A1": 5, "A2": 10}),
                    "error": "",
                    "used_vision": False,
                },
            ):
                actions = controller.choose_llm_actions(core, step=core.n_step)

        self.assertEqual(actions, {"A1": 5})
        self.assertTrue(controller._controller_state("A1").execute_called)
        self.assertFalse(controller._controller_state("A2").execute_called)

    def test_global_controller_executes_all_entered_agents_once_after_entry(self) -> None:
        core = FakeCore(
            [
                _make_state("A1", position=(10.0, 10.0), has_entered_sector=True),
                _make_state("A2", position=(20.0, 10.0), has_entered_sector=True),
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = GlobalLangGraphGuidanceController(save_dir=tmpdir)
            with patch.object(controller, "_ranked_threats", return_value=[]), patch.object(
                controller,
                "_preview_rows",
                return_value=[],
            ), patch.object(controller, "_deterministic_best_turn", return_value=0), patch(
                "rl_llm_multi.llm.ollama_invoke",
                return_value={
                    "llm_status": "ok",
                    "raw_text": _global_answer_json({"A1": 5, "A2": -5}),
                    "error": "",
                    "used_vision": False,
                },
            ):
                actions = controller.choose_llm_actions(core, step=core.n_step)

        self.assertEqual(actions, {"A1": 5, "A2": -5})
        self.assertTrue(controller._controller_state("A1").execute_called)
        self.assertTrue(controller._controller_state("A2").execute_called)

    def test_global_controller_retries_missing_actions_then_falls_back_per_agent(self) -> None:
        core = FakeCore(
            [
                _make_state("A1", position=(10.0, 10.0), has_entered_sector=True),
                _make_state("A2", position=(20.0, 10.0), has_entered_sector=True),
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = GlobalLangGraphGuidanceController(save_dir=tmpdir)
            with patch.object(controller, "_ranked_threats", return_value=[]), patch.object(
                controller,
                "_preview_rows",
                return_value=[],
            ), patch.object(controller, "_deterministic_best_turn", return_value=15), patch(
                "rl_llm_multi.llm.ollama_invoke",
                return_value={
                    "llm_status": "ok",
                    "raw_text": _global_answer_json({"A1": 5}),
                    "error": "",
                    "used_vision": False,
                },
            ) as invoke_mock:
                actions = controller.choose_llm_actions(core, step=core.n_step)

        self.assertEqual(invoke_mock.call_count, 3)
        self.assertEqual(actions, {"A1": 5, "A2": 15})
        self.assertEqual(controller.last_normalized["debug"]["fallback_agents"], ["A2"])
        self.assertEqual(controller.last_normalized["debug"]["retry_count"], 2)

    def test_global_controller_emergency_disallows_zero_turn(self) -> None:
        core = FakeCore(
            [
                _make_state(
                    "A1",
                    position=(10.0, 10.0),
                    predicted_pair_loss_step=1,
                    has_entered_sector=True,
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = GlobalLangGraphGuidanceController(save_dir=tmpdir)
            ctrl_state = controller._controller_state("A1")
            ctrl_state.execute_called = True
            ctrl_state.stage = "WAIT_CLEAR"
            with patch.object(controller, "_ranked_threats", return_value=[]), patch.object(
                controller,
                "_preview_rows",
                return_value=[],
            ), patch(
                "rl_llm_multi.llm.ollama_invoke",
                return_value={
                    "llm_status": "ok",
                    "raw_text": _global_answer_json({"A1": 0}),
                    "error": "",
                    "used_vision": False,
                },
            ):
                actions = controller.choose_llm_actions(core, step=core.n_step)

        self.assertEqual(actions, {"A1": -5})

    def test_global_controller_merge_back_enforces_destination_side_turn(self) -> None:
        core = FakeCore(
            [
                _make_state(
                    "A1",
                    position=(10.0, 10.0),
                    heading_deg=-30.0,
                    boundary_warning_active=True,
                    has_entered_sector=True,
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = GlobalLangGraphGuidanceController(save_dir=tmpdir)
            ctrl_state = controller._controller_state("A1")
            ctrl_state.execute_called = True
            ctrl_state.stage = "WAIT_CLEAR"
            with patch.object(controller, "_ranked_threats", return_value=[]), patch.object(
                controller,
                "_preview_rows",
                return_value=[],
            ), patch(
                "rl_llm_multi.llm.ollama_invoke",
                return_value={
                    "llm_status": "ok",
                    "raw_text": _global_answer_json({"A1": -10}),
                    "error": "",
                    "used_vision": False,
                },
            ):
                actions = controller.choose_llm_actions(core, step=core.n_step)

        self.assertEqual(actions, {"A1": 0})

    def test_global_controller_vision_uses_current_plus_previous_three_frames(self) -> None:
        core = FakeCore([_make_state("A1", position=(10.0, 10.0), has_entered_sector=True)])
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"
            frame_dir.mkdir()
            for step in range(4):
                (frame_dir / f"image_{step:03d}.png").touch()
            controller = GlobalLangGraphGuidanceController(save_dir=tmpdir)
            with patch.object(controller, "_ranked_threats", return_value=[]), patch.object(
                controller,
                "_preview_rows",
                return_value=[],
            ), patch.object(controller, "_deterministic_best_turn", return_value=0), patch(
                "rl_llm_multi.llm.ollama_invoke",
                return_value={
                    "llm_status": "ok",
                    "raw_text": _global_answer_json({"A1": 5}),
                    "error": "",
                    "used_vision": True,
                },
            ) as invoke_mock:
                actions = controller.choose_llm_actions(
                    core,
                    step=3,
                    latest_frame_path=str(frame_dir / "image_003.png"),
                    use_vision=True,
                )

        self.assertEqual(actions, {"A1": 5})
        image_paths = invoke_mock.call_args.kwargs["image_paths"]
        self.assertEqual([Path(path).name for path in image_paths], ["image_003.png", "image_002.png", "image_001.png", "image_000.png"])
        self.assertEqual(
            [Path(path).name for path in controller.last_normalized["debug"]["frame_paths"]],
            ["image_003.png", "image_002.png", "image_001.png", "image_000.png"],
        )

    def test_global_controller_text_mode_uses_no_image_paths(self) -> None:
        core = FakeCore([_make_state("A1", position=(10.0, 10.0), has_entered_sector=True)])
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = Path(tmpdir) / "image_003.png"
            frame_path.touch()
            controller = GlobalLangGraphGuidanceController(save_dir=tmpdir)
            with patch.object(controller, "_ranked_threats", return_value=[]), patch.object(
                controller,
                "_preview_rows",
                return_value=[],
            ), patch.object(controller, "_deterministic_best_turn", return_value=0), patch(
                "rl_llm_multi.llm.ollama_invoke",
                return_value={
                    "llm_status": "ok",
                    "raw_text": _global_answer_json({"A1": 5}),
                    "error": "",
                    "used_vision": False,
                },
            ) as invoke_mock:
                actions = controller.choose_llm_actions(
                    core,
                    step=3,
                    latest_frame_path=str(frame_path),
                    use_vision=False,
                )

        self.assertEqual(actions, {"A1": 5})
        self.assertIsNone(invoke_mock.call_args.kwargs["image_paths"])
        self.assertEqual(controller.last_normalized["debug"]["frame_paths"], [])

    def test_global_controller_smoke_runs_with_two_and_four_agents(self) -> None:
        for num_agents in (2, 4):
            with self.subTest(num_agents=num_agents):
                with tempfile.TemporaryDirectory() as tmpdir:
                    env = JointGuidanceEnv(num_agents=num_agents, max_agents=MAX_AGENTS)
                    env.reset(seed=42, num_agents=num_agents, route_ids=None)
                    controller = GlobalLangGraphGuidanceController(save_dir=tmpdir)
                    controller.reset()
                    with patch(
                        "rl_llm_multi.llm.ollama_invoke",
                        return_value={
                            "llm_status": "ok",
                            "raw_text": _global_answer_json({f"A{idx}": 0 for idx in range(1, MAX_AGENTS + 1)}),
                            "error": "",
                            "used_vision": False,
                        },
                    ):
                        for _ in range(3):
                            actions = controller.choose_llm_actions(env.core, step=env.n_step)
                            self.assertIsInstance(actions, dict)
                            _, _, done, truncated, _ = env.step(actions)
                            if done or truncated:
                                break

    def test_smoke_runs_with_two_and_four_agents(self) -> None:
        for num_agents in (2, 4):
            with self.subTest(num_agents=num_agents):
                with tempfile.TemporaryDirectory() as tmpdir:
                    env = JointGuidanceEnv(num_agents=num_agents, max_agents=MAX_AGENTS)
                    env.reset(seed=42, num_agents=num_agents, route_ids=None)
                    controller = MultiAgentThreeCallController(save_dir=tmpdir)
                    controller.reset()
                    with patch(
                        "rl_llm_multi.llm.ollama_invoke",
                        return_value={
                            "llm_status": "ok",
                            "raw_text": _answer_json(0),
                            "error": "",
                            "used_vision": False,
                        },
                    ):
                        for _ in range(3):
                            actions = controller.choose_llm_actions(env.core, step=env.n_step)
                            self.assertIsInstance(actions, dict)
                            _, _, done, truncated, _ = env.step(actions)
                            if done or truncated:
                                break


if __name__ == "__main__":
    unittest.main()
