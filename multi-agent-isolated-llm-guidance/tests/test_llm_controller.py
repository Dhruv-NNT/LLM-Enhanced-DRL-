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

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Ellipse
from shapely.geometry import LineString


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from configs import (  # noqa: E402
    AGENT_SPEED,
    LAUNCH_SEPARATION_R,
    MAX_AGENTS,
    SAFE_R,
    WEATHER_TRAIL_STEPS,
    WEATHER_MAJOR_RADIUS_NM_MAX,
    WEATHER_MAJOR_RADIUS_NM_MIN,
)
from rl_llm_multi import (  # noqa: E402
    GlobalLangGraphGuidanceController,
    JointGuidanceEnv,
    MultiAgentSectorCore,
    MultiAgentThreeCallController,
    WeatherCell,
)
from rl_llm_multi.llm import GlobalPromptBuilder, HLTPPromptBuilder  # noqa: E402
from rl_llm_multi.memory import DecisionMemoryStore  # noqa: E402
from rl_llm_multi.utils import assign_routes, weather_axis_units  # noqa: E402


def _global_controller(save_dir: str, **kwargs) -> GlobalLangGraphGuidanceController:
    return GlobalLangGraphGuidanceController(
        save_dir=save_dir,
        memory_path=str(Path(save_dir) / "decision_memory.json"),
        **kwargs,
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
    weather_predicted: bool = False,
    hazard_predicted: bool = False,
    launched: bool = True,
    finished: bool = False,
    has_entered_sector: bool = True,
    sector_entry_step: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=agent_id,
        route=SimpleNamespace(
            linestring=LineString([(0.0, 0.0), (100.0, 0.0)]),
            origin="ORIG",
            destination="DEST",
            waypoint_names=("ORIG", "DEST"),
        ),
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
        weather_predicted=bool(weather_predicted),
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
        self.weather_clearance_buffer_units = 2.0
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


class LaunchSpacingTests(unittest.TestCase):
    def test_same_origin_routes_are_staggered_by_launch_separation(self) -> None:
        assignments = assign_routes(2, route_ids=["PATH5_REV", "PATH6"])

        self.assertEqual(assignments[0].route.origin, "OMBAP")
        self.assertEqual(assignments[1].route.origin, "OMBAP")
        self.assertEqual(assignments[0].start_step, 0)
        self.assertGreater(assignments[1].start_step, 0)

        first_route = assignments[0].route
        second_origin = assignments[1].route.points[0]
        elapsed_before_launch = max(0, assignments[1].start_step - 1)
        first_progress = float(elapsed_before_launch) * float(AGENT_SPEED)
        first_point = first_route.linestring.interpolate(first_progress)
        distance_at_launch = math.hypot(
            first_point.x - second_origin[0],
            first_point.y - second_origin[1],
        )
        self.assertGreaterEqual(distance_at_launch, LAUNCH_SEPARATION_R)

    def test_reset_does_not_initially_launch_too_close_aircraft(self) -> None:
        core = MultiAgentSectorCore(num_agents=8, num_weather_cells=2)
        core.reset(seed=42, num_agents=8, route_ids=None, num_weather_cells=2)

        active_ids = list(core.active_agent_ids)
        for idx, agent_a in enumerate(active_ids):
            for agent_b in active_ids[idx + 1 :]:
                state_a = core.get_agent(agent_a)
                state_b = core.get_agent(agent_b)
                distance = math.hypot(
                    state_a.position[0] - state_b.position[0],
                    state_a.position[1] - state_b.position[1],
                )
                self.assertGreaterEqual(distance, LAUNCH_SEPARATION_R)

    def test_runtime_launch_gate_waits_for_active_aircraft_to_clear_origin(self) -> None:
        core = MultiAgentSectorCore(num_agents=2, num_weather_cells=1)
        core.reset(
            seed=42,
            num_agents=2,
            route_ids=["PATH5_REV", "PATH6"],
            num_weather_cells=1,
        )
        blocker = core.get_agent("A1")
        delayed = core.get_agent("A2")
        origin = delayed.route.points[0]

        blocker.launched = True
        blocker.active = True
        blocker.finished = False
        blocker.position = origin
        delayed.launched = False
        delayed.active = False
        delayed.start_step = core.n_step

        core._launch_agents()
        self.assertFalse(delayed.launched)
        self.assertEqual(delayed.start_step, core.n_step + 1)

        blocker.position = (origin[0] + LAUNCH_SEPARATION_R + 1.0, origin[1])
        core.n_step = delayed.start_step
        core._launch_agents()
        self.assertTrue(delayed.launched)

    def test_preview_tracks_aircraft_that_launch_during_horizon(self) -> None:
        core = MultiAgentSectorCore(num_agents=8, num_weather_cells=2)
        core.reset(seed=42, num_agents=8, num_weather_cells=2)
        self.assertNotIn("A7", core.active_agent_ids)
        self.assertGreater(core.get_agent("A7").start_step, core.n_step)

        controller = MultiAgentThreeCallController()
        preview = controller._simulate_joint_horizon(
            core,
            {},
            horizon_steps=core.get_agent("A7").start_step + 2,
        )

        self.assertIn("A7", preview["per_agent_min_sep"])
        self.assertIn("A7", preview["pair_loss_step"])


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

    def test_preview_marks_weather_buffer_breach_unsafe(self) -> None:
        controller = MultiAgentThreeCallController()
        core = FakeCore([_make_state("A1", position=(10.0, 10.0), has_entered_sector=True)])
        with patch.object(controller, "_allowed_turns", return_value=[0]), patch.object(
            controller,
            "_simulate_joint_horizon",
            return_value={
                "pair_loss_step": {"A1": None},
                "weather_entry_step": {"A1": None},
                "boundary_exit_step": {"A1": None},
                "per_agent_min_sep": {"A1": SAFE_R + 1.0},
                "min_weather_clearance": {"A1": core.weather_clearance_buffer_units - 0.1},
                "end_heading_error_to_dest_deg": {"A1": 0.0},
                "end_distance_to_destination": {"A1": 70.0},
                "end_cross_track_abs": {"A1": 0.0},
            },
        ):
            rows = controller._preview_rows(core, "A1", "EXECUTE_TURN")

        self.assertFalse(rows[0]["safe_over_preview"])
        self.assertFalse(rows[0]["weather_buffer_ok"])

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
        merge_prompt = controller.builder.build_prompt(core, "A1", "MERGE_BACK", [], preview_rows, {})
        self.assertIn("largest allowed turn magnitude", merge_prompt)
        self.assertIn("boundary_exit_step", merge_prompt)
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
            controller = _global_controller(tmpdir)
            with patch.object(controller, "_ranked_threats", return_value=[]), patch.object(
                controller,
                "_preview_rows",
                return_value=[
                    {
                        "heading_change_deg": 5,
                        "safe_over_preview": True,
                        "pair_loss_step": None,
                        "weather_entry_step": None,
                        "boundary_exit_step": None,
                        "min_sep_to_any": 8.0,
                        "min_weather_clearance": 7.0,
                        "progress_to_destination": 5.0,
                        "cross_track_reduction": 1.0,
                        "end_heading_error_to_dest_deg": 4.0,
                    }
                ],
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
            controller = _global_controller(tmpdir)
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
            controller = _global_controller(tmpdir)
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
            controller = _global_controller(tmpdir)
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
            controller = _global_controller(tmpdir)
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

    def test_default_core_uses_one_weather_cell(self) -> None:
        core = MultiAgentSectorCore(num_agents=2)
        core.reset(seed=42)

        self.assertEqual(len(core.weather_cells), 1)
        self.assertIs(core.weather_cell, core.weather_cells[0])
        self.assertEqual(core.weather_dict()["num_weather_cells"], 1)
        self.assertEqual(len(core.weather_dict()["cells"]), 1)

    def test_two_weather_cells_initialize_with_shared_motion_heading(self) -> None:
        core = MultiAgentSectorCore(num_agents=2, num_weather_cells=2)
        core.reset(seed=42, num_weather_cells=2)

        self.assertEqual(len(core.weather_cells), 2)
        self.assertEqual([cell.cell_id for cell in core.weather_cells], ["W1", "W2"])
        self.assertAlmostEqual(
            core.weather_cells[0].motion_heading_rad,
            core.weather_cells[1].motion_heading_rad,
        )
        self.assertEqual(len(core.weather_dict()["cells"]), 2)

    def test_weather_growth_clamps_and_stops_at_max(self) -> None:
        core = MultiAgentSectorCore(num_agents=1)
        core.reset(seed=42)
        cell = core.weather_cell
        cell.major_radius_nm = WEATHER_MAJOR_RADIUS_NM_MAX - 0.05
        cell.major_growth_nm_per_step = 0.5
        cell.speed_units_per_step = 0.0

        with patch.object(core, "_weather_geometry_valid", return_value=True), patch.object(
            core,
            "_weather_destinations_clear",
            return_value=True,
        ):
            advanced = core._advance_weather_cell(cell)

        self.assertEqual(advanced.major_radius_nm, WEATHER_MAJOR_RADIUS_NM_MAX)
        self.assertEqual(advanced.major_growth_nm_per_step, 0.0)
        self.assertTrue(advanced.growth_stopped)

    def test_weather_shrink_clamps_and_stops_at_min(self) -> None:
        core = MultiAgentSectorCore(num_agents=1)
        core.reset(seed=42)
        cell = core.weather_cell
        cell.major_radius_nm = WEATHER_MAJOR_RADIUS_NM_MIN + 0.05
        cell.major_growth_nm_per_step = -0.5
        cell.speed_units_per_step = 0.0

        with patch.object(core, "_weather_geometry_valid", return_value=True), patch.object(
            core,
            "_weather_destinations_clear",
            return_value=True,
        ):
            advanced = core._advance_weather_cell(cell)

        self.assertEqual(advanced.major_radius_nm, WEATHER_MAJOR_RADIUS_NM_MIN)
        self.assertEqual(advanced.major_growth_nm_per_step, 0.0)
        self.assertTrue(advanced.growth_stopped)

    def test_weather_boundary_hit_freezes_position_and_stops_movement(self) -> None:
        core = MultiAgentSectorCore(num_agents=1)
        core.reset(seed=42)
        cell = core.weather_cell
        start_center = tuple(cell.center)

        with patch.object(core, "_weather_geometry_valid", return_value=False):
            advanced = core._advance_weather_cell(cell)

        self.assertEqual(tuple(advanced.center), start_center)
        self.assertEqual(advanced.speed_units_per_step, 0.0)
        self.assertEqual(advanced.major_growth_nm_per_step, 0.0)
        self.assertTrue(advanced.movement_stopped)

    def test_weather_signed_clearance_uses_most_dangerous_cell(self) -> None:
        core = MultiAgentSectorCore(num_agents=1)
        core.reset(seed=42)
        core.weather_cells = [
            WeatherCell((10.0, 10.0), 2.5, 0.8, 0.0, 0.0, 0.0, 0.0, cell_id="W1"),
            WeatherCell((50.0, 50.0), 15.0, 0.8, 0.0, 0.0, 0.0, 0.0, cell_id="W2"),
        ]
        core._sync_primary_weather_cell()

        self.assertLess(core.weather_signed_clearance((50.0, 50.0)), 0.0)

    def test_aircraft_entering_second_weather_cell_triggers_failure(self) -> None:
        core = MultiAgentSectorCore(num_agents=1, num_weather_cells=2)
        core.reset(seed=42, num_weather_cells=2)
        state = core.get_agent("A1")
        state.position = (50.0, 50.0)
        state.heading_rad = 0.0
        state.launched = True
        state.active = True
        state.has_entered_sector = True
        state.inside_sector = True
        core.weather_cells = [
            WeatherCell((10.0, 10.0), 2.5, 0.8, 0.0, 0.0, 0.0, 0.0, cell_id="W1", movement_stopped=True),
            WeatherCell((52.0, 50.0), 15.0, 0.8, 0.0, 0.0, 0.0, 0.0, cell_id="W2", movement_stopped=True),
        ]
        core._sync_primary_weather_cell()

        _, _, terminations, _, info = core.step({})

        self.assertTrue(any(terminations.values()))
        self.assertEqual(info["failure_reason"], "weather")

    def test_aircraft_entering_only_green_weather_ring_does_not_terminate(self) -> None:
        core = MultiAgentSectorCore(num_agents=1, num_weather_cells=1)
        core.reset(seed=42, num_weather_cells=1)
        state = core.get_agent("A1")
        cell = WeatherCell((50.0, 50.0), 15.0, 0.8, 0.0, 0.0, 0.0, 0.0, cell_id="W1", movement_stopped=True)
        major_units, _ = weather_axis_units(cell)
        state.position = (cell.center[0] + 0.90 * major_units - core.speed, cell.center[1])
        state.destination = (90.0, 90.0)
        state.heading_rad = 0.0
        state.launched = True
        state.active = True
        state.has_entered_sector = True
        state.inside_sector = True
        core.weather_cells = [cell]
        core._sync_primary_weather_cell()

        self.assertLess(core.weather_signed_clearance((cell.center[0] + 0.90 * major_units, cell.center[1])), 0.0)
        self.assertGreater(core.weather_terminal_signed_clearance((cell.center[0] + 0.90 * major_units, cell.center[1])), 0.0)

        _, _, terminations, _, info = core.step({})

        self.assertFalse(any(terminations.values()))
        self.assertIsNone(info["failure_reason"])

    def test_global_prompt_includes_both_weather_cells(self) -> None:
        core = MultiAgentSectorCore(num_agents=2, num_weather_cells=2)
        core.reset(seed=42, num_weather_cells=2)
        prompt = GlobalPromptBuilder().build_prompt(
            core,
            step=0,
            actionable=[],
            entered_agent_ids=[],
            hltp_plans={},
            threat_rows_by_agent={},
            preview_rows_by_agent={},
            recent_decisions=[],
            frame_paths=["/tmp/image_000.png"],
        )

        self.assertIn("W1", prompt)
        self.assertIn("W2", prompt)
        self.assertIn("green is a caution/buffer ring", prompt)
        self.assertIn("green=light precipitation", prompt)
        self.assertIn("never enter yellow/red/magenta", prompt)
        self.assertIn("dashed green outer ellipses", prompt)
        self.assertIn("motion is from the solid cell toward the dashed outlines", prompt)
        self.assertIn("do not output call_name", prompt)
        self.assertIn("may choose any displayed row", prompt)
        self.assertIn("each row predicts consequences", prompt)
        self.assertNotIn("\"call_name\": \"EXECUTE_TURN\"", prompt)

        merge_prompt = GlobalPromptBuilder().build_prompt(
            core,
            step=0,
            actionable=[{"agent_id": "A1", "call_name": "MERGE_BACK", "call_reason": "BOUNDARY_LOOKAHEAD"}],
            entered_agent_ids=["A1"],
            hltp_plans={
                "A1": {
                    "short_context": "HLTP MUMSO->MABAL",
                    "rationale": "Turn right around the conflict, then rejoin the original route.",
                }
            },
            threat_rows_by_agent={},
            preview_rows_by_agent={},
            recent_decisions=[],
            frame_paths=[],
        )
        self.assertIn("largest allowed turn magnitude", merge_prompt)
        self.assertIn("boundary_exit", merge_prompt)
        self.assertIn("HLTP MUMSO->MABAL", merge_prompt)
        self.assertIn("Turn right around the conflict", merge_prompt)

    def test_global_prompt_uses_retrieved_memory_instead_of_recent_memory(self) -> None:
        core = FakeCore([_make_state("A1", position=(10.0, 10.0), has_entered_sector=True)])
        unverified_prompt = GlobalPromptBuilder().build_prompt(
            core,
            step=5,
            actionable=[{"agent_id": "A1", "call_name": "EXECUTE_TURN", "call_reason": "SECTOR_ENTRY"}],
            entered_agent_ids=["A1"],
            hltp_plans={},
            threat_rows_by_agent={},
            preview_rows_by_agent={},
            recent_decisions=[{"step": 4, "actions": {"A1": {"call_name": "EXECUTE_TURN", "heading_change_deg": 5}}}],
            frame_paths=[],
            retrieved_memories=[
                {
                    "agent": "A1",
                    "similarity": 0.82,
                    "action": -15,
                    "corrected": None,
                    "note": None,
                }
            ],
        )

        self.assertIn("RETRIEVED DECISION MEMORY", unverified_prompt)
        self.assertIn("A1 sim=0.82 unverified memory comparison only", unverified_prompt)
        self.assertIn("similar past case used -15 deg", unverified_prompt)
        self.assertIn("Do not copy unless current preview independently supports it", unverified_prompt)
        self.assertIn("Memory is not proof the action is good", unverified_prompt)
        self.assertNotIn("RECENT GLOBAL DECISION MEMORY", unverified_prompt)

        corrected_prompt = GlobalPromptBuilder().build_prompt(
            core,
            step=5,
            actionable=[{"agent_id": "A1", "call_name": "EXECUTE_TURN", "call_reason": "SECTOR_ENTRY"}],
            entered_agent_ids=["A1"],
            hltp_plans={},
            threat_rows_by_agent={},
            preview_rows_by_agent={},
            recent_decisions=[],
            frame_paths=[],
            retrieved_memories=[
                {
                    "agent": "A1",
                    "similarity": 0.82,
                    "action": -15,
                    "corrected": -25,
                    "note": "traffic buffer kept shrinking",
                }
            ],
        )

        self.assertIn("RETRIEVED DECISION MEMORY", corrected_prompt)
        self.assertIn("A1 sim=0.82 evaluator-corrected memory", corrected_prompt)
        self.assertIn("corrected -25 deg is preferred only if current preview is safe", corrected_prompt)
        self.assertIn("Memory is not proof the action is good", corrected_prompt)

    def test_decision_memory_retrieval_is_route_gated(self) -> None:
        preview_rows = [
            {
                "heading_change_deg": -15,
                "safe_over_preview": True,
                "pair_loss_step": None,
                "weather_entry_step": None,
                "boundary_exit_step": None,
                "traffic_buffer_ok": True,
                "weather_buffer_ok": True,
                "min_sep_to_any": 8.0,
                "min_weather_clearance": 7.0,
                "progress_to_destination": 5.0,
            },
            {
                "heading_change_deg": 0,
                "safe_over_preview": False,
                "pair_loss_step": 3,
                "weather_entry_step": None,
                "boundary_exit_step": None,
                "traffic_buffer_ok": False,
                "weather_buffer_ok": True,
                "min_sep_to_any": 4.0,
                "min_weather_clearance": 7.0,
                "progress_to_destination": 6.0,
            },
        ]
        threat_rows = [
            {
                "intruder_id": "A2",
                "current_distance": 9.0,
                "bearing_error_deg": 5.0,
                "predicted_loss_step": 3,
                "predicted_min_sep": 4.0,
            }
        ]
        core = FakeCore(
            [
                _make_state("A1", position=(10.0, 0.0), heading_deg=0.0, predicted_min_sep=4.0),
                _make_state("A2", position=(19.0, 0.0), heading_deg=180.0),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DecisionMemoryStore(Path(tmpdir) / "decision_memory.json", min_similarity=0.75)
            store.append_cases(
                core,
                episode_id="ep1",
                step=5,
                actionable=[{"agent_id": "A1", "call_name": "EXECUTE_TURN", "call_reason": "SECTOR_ENTRY"}],
                final_actions={"A1": {"heading_change_deg": -15}},
                threat_rows_by_agent={"A1": threat_rows},
                preview_rows_by_agent={"A1": preview_rows},
            )
            same_route = store.retrieve(
                core,
                step=5,
                actionable=[{"agent_id": "A1", "call_name": "EXECUTE_TURN", "call_reason": "SECTOR_ENTRY"}],
                threat_rows_by_agent={"A1": threat_rows},
                preview_rows_by_agent={"A1": preview_rows},
            )
            memory_payload = json.loads((Path(tmpdir) / "decision_memory.json").read_text())
            stored_case_id = memory_payload["episodes"][0]["cases"][0]["case_id"]

            changed_route_core = FakeCore(
                [
                    _make_state("A1", position=(10.0, 0.0), heading_deg=0.0, predicted_min_sep=4.0),
                    _make_state("A2", position=(19.0, 0.0), heading_deg=180.0),
                ]
            )
            changed_route_core.get_agent("A1").route.origin = "OTHER"
            changed_route = store.retrieve(
                changed_route_core,
                step=5,
                actionable=[{"agent_id": "A1", "call_name": "EXECUTE_TURN", "call_reason": "SECTOR_ENTRY"}],
                threat_rows_by_agent={"A1": threat_rows},
                preview_rows_by_agent={"A1": preview_rows},
            )

        self.assertEqual(same_route[0]["action"], -15)
        self.assertTrue(stored_case_id.startswith("case_"))
        self.assertEqual(same_route[0]["memory_ref"], {"episode_id": "ep1", "case_id": stored_case_id})
        self.assertEqual(same_route[0]["case"]["case_id"], stored_case_id)
        self.assertEqual(changed_route, [])

    def test_advisory_global_commit_does_not_append_normal_memory(self) -> None:
        class FailingMemory:
            def append_cases(self, *args, **kwargs):
                raise AssertionError("advisory labels must not append normal memory cases")

        controller = GlobalLangGraphGuidanceController.__new__(GlobalLangGraphGuidanceController)
        controller.decision_memory = FailingMemory()
        controller.recent_decisions = []
        controller.last_annotations = []
        controller.last_joint_actions = {}
        controller._memory_episode_id = lambda core: "ep1"
        controller._phase_label_text = lambda agent_id, call_name, call_reason: f"{agent_id}:{call_name}"
        applied = []

        def apply_state(agent_id, call_name, call_reason, applied_turn):
            applied.append((agent_id, call_name, applied_turn))

        controller._apply_controller_state = apply_state
        controller._normalized_global_payload = GlobalLangGraphGuidanceController._normalized_global_payload.__get__(
            controller,
            GlobalLangGraphGuidanceController,
        )
        logged = {}

        def log_call(**kwargs):
            logged.update(kwargs)
            controller.last_normalized = {
                **kwargs["normalized"],
                "debug": kwargs["debug"],
            }

        controller._log_global_call = log_call
        core = FakeCore([_make_state("A1", position=(10.0, 0.0), heading_deg=0.0)])

        result = GlobalLangGraphGuidanceController._graph_commit_and_log(
            controller,
            {
                "core": core,
                "step": 5,
                "advisory_only": True,
                "actionable": [
                    {"agent_id": "A1", "call_name": "EXECUTE_TURN", "call_reason": "SECTOR_ENTRY"}
                ],
                "annotations": [],
                "final_actions": {
                    "A1": {
                        "call_name": "EXECUTE_TURN",
                        "heading_change_deg": 10,
                        "requested_heading_change_deg": 10,
                        "source": "llm",
                        "rationale": "",
                    }
                },
                "llm_attempts": [{"llm_status": "ok", "raw_text": "{}", "error": ""}],
                "parse_status": "ok",
                "fallback_reason": "",
                "fallback_agents": [],
                "entered_agent_ids": ["A1"],
                "stage_by_agent": {"A1": "EXECUTE_TURN"},
                "hltp_plans": {},
                "retrieved_memories": [],
                "threat_rows_by_agent": {},
                "preview_rows_by_agent": {},
            },
        )

        self.assertEqual(result["actions"], {"A1": 10})
        self.assertEqual(applied, [("A1", "EXECUTE_TURN", 10)])
        self.assertTrue(controller.last_normalized["answer"]["advisory_only"])
        self.assertTrue(controller.last_normalized["debug"]["advisory_only"])
        self.assertEqual(controller.last_normalized["debug"]["memory_error"], "advisory_only_no_normal_memory_write")

    def test_decision_memory_update_case_result_only_updates_correction_fields(self) -> None:
        preview_rows = [
            {
                "heading_change_deg": -15,
                "safe_over_preview": True,
                "pair_loss_step": None,
                "weather_entry_step": None,
                "boundary_exit_step": None,
                "traffic_buffer_ok": True,
                "weather_buffer_ok": True,
                "min_sep_to_any": 8.0,
                "min_weather_clearance": 7.0,
                "progress_to_destination": 5.0,
            }
        ]
        core = FakeCore([_make_state("A1", position=(10.0, 0.0), heading_deg=0.0)])

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DecisionMemoryStore(Path(tmpdir) / "decision_memory.json", min_similarity=0.75)
            store.append_cases(
                core,
                episode_id="ep1",
                step=5,
                actionable=[{"agent_id": "A1", "call_name": "EXECUTE_TURN", "call_reason": "SECTOR_ENTRY"}],
                final_actions={"A1": {"heading_change_deg": -15}},
                threat_rows_by_agent={"A1": []},
                preview_rows_by_agent={"A1": preview_rows},
            )
            before = store.load()
            case_id = before["episodes"][0]["cases"][0]["case_id"]
            expected = copy.deepcopy(before)
            expected["episodes"][0]["cases"][0]["result"]["corrected"] = -25
            expected["episodes"][0]["cases"][0]["result"]["note"] = "short-horizon LLM return improved memory"

            updated = store.update_case_result(
                {"episode_id": "ep1", "case_id": case_id},
                corrected=-25,
                note="short-horizon LLM return improved memory",
            )
            after = store.load()

        self.assertTrue(updated)
        self.assertEqual(after, expected)

    def test_global_vision_prompt_includes_action_candidates_not_text_table(self) -> None:
        core = FakeCore([_make_state("A1", position=(10.0, 10.0), has_entered_sector=True)])
        prompt = GlobalPromptBuilder().build_vision_prompt(
            core,
            step=5,
            actionable=[{"agent_id": "A1", "call_name": "EXECUTE_TURN", "call_reason": "SECTOR_ENTRY"}],
            entered_agent_ids=["A1"],
            hltp_plans={},
            threat_rows_by_agent={},
            preview_rows_by_agent={
                "A1": [
                    {
                        "heading_change_deg": 5,
                        "safe_over_preview": True,
                        "pair_loss_step": None,
                        "weather_entry_step": None,
                        "boundary_exit_step": None,
                        "min_sep_to_any": 8.0,
                        "min_weather_clearance": 7.0,
                        "progress_to_destination": 5.0,
                        "cross_track_reduction": 1.0,
                        "end_heading_error_to_dest_deg": 4.0,
                    }
                ]
            },
            recent_decisions=[],
            frame_paths=["/tmp/image_005.png"],
        )

        self.assertIn("VISION ACTION CANDIDATES", prompt)
        self.assertIn("Use vision to choose tactical intent", prompt)
        self.assertIn("choose among listed VISION ACTION CANDIDATES", prompt)
        self.assertIn("choose only from the listed turns", prompt)
        self.assertIn("Rows are guidance, not orders", prompt)
        self.assertIn("visual common sense may justify another viable listed row", prompt)
        self.assertIn("displayed_turns=[+5]", prompt)
        self.assertIn("turn=+5", prompt)
        self.assertNotIn("trust the candidate row", prompt)
        self.assertNotIn("prefer the first row unless", prompt)
        self.assertNotIn("Do not use a preview table", prompt)
        self.assertNotIn("PER-AGENT TURN PREVIEWS", prompt)

    def test_invalid_hltp_waypoint_proposal_can_preserve_llm_rationale(self) -> None:
        controller = GlobalLangGraphGuidanceController.__new__(GlobalLangGraphGuidanceController)
        context = {
            "agent_id": "A3",
            "conflict_partner_ids": ["A4"],
            "right_waypoint_candidates": [
                {"name": "MABAL", "route_progress": 10.0},
            ],
            "merge_back_candidates": [
                {"name": "LEBIN", "route_progress": 20.0},
            ],
        }
        payload = {
            "first_waypoint_name": "MUMSO",
            "merge_back_waypoint_name": "MABAL",
            "rationale": "Turn right towards MUMSO, then merge back towards MABAL.",
        }

        self.assertIsNone(
            controller._hltp_plan_from_payload(
                agent_id="A3",
                payload=payload,
                context=context,
                step=5,
            )
        )
        fallback = controller._fallback_hltp_plan(
            context=context,
            step=5,
            rationale=(
                "LLM rationale from rejected waypoint proposal: "
                f"{payload['rationale']}"
            ),
        )

        self.assertEqual(fallback.short_context, "HLTP MABAL->LEBIN")
        self.assertIn("Turn right towards MUMSO", fallback.rationale)

    def test_hltp_prompt_uses_neutral_waypoint_placeholders(self) -> None:
        core = MultiAgentSectorCore(num_agents=1, num_weather_cells=1)
        core.reset(seed=42, num_weather_cells=1)

        prompt = HLTPPromptBuilder().build_prompt(core, step=5, contexts=[], frame_paths=[])

        self.assertIn("<FIRST_WAYPOINT_FROM_CANDIDATES>", prompt)
        self.assertIn("<MERGE_BACK_WAYPOINT_FROM_CANDIDATES>", prompt)
        self.assertNotIn("\"first_waypoint_name\": \"MUMSO\"", prompt)
        self.assertNotIn("\"merge_back_waypoint_name\": \"MABAL\"", prompt)

        multi_prompt = HLTPPromptBuilder()._instructions_text(
            5,
            [{"agent_id": "A3"}, {"agent_id": "A4"}],
        )
        self.assertIn("\"A3\":", multi_prompt)
        self.assertIn("\"A4\":", multi_prompt)
        self.assertIn("must include every aircraft shown", multi_prompt)

    def test_hltp_vision_prompt_uses_frames_as_real_context(self) -> None:
        core = MultiAgentSectorCore(num_agents=1, num_weather_cells=1)
        core.reset(seed=42, num_weather_cells=1)
        contexts = [
            {
                "agent_id": "A1",
                "state": {"position": (10.0, 10.0), "heading_deg": 45.0},
                "origin_name": "ORIG",
                "destination_name": "DEST",
                "destination": (90.0, 10.0),
                "route_waypoint_names": ["ORIG", "MID", "DEST"],
                "conflict_partner_ids": ["A2"],
                "conflict": {
                    "agent_ids": ["A1", "A2"],
                    "loss_step": 5,
                    "min_step": 4,
                    "min_sep": 4.2,
                    "point": (20.0, 15.0),
                },
                "right_waypoint_candidates": [
                    {
                        "name": "RIGHT",
                        "point": (25.0, 5.0),
                        "route_progress": 20.0,
                        "side_offset": -8.0,
                        "distance_to_conflict": 6.0,
                    },
                ],
                "merge_back_candidates": [
                    {"name": "MID", "point": (50.0, 0.0), "route_progress": 50.0},
                ],
            }
        ]

        prompt = HLTPPromptBuilder().build_prompt(
            core,
            step=5,
            contexts=contexts,
            frame_paths=["/tmp/image_005.png", "/tmp/image_004.png"],
        )
        text_prompt = HLTPPromptBuilder().build_prompt(core, step=5, contexts=contexts, frame_paths=[])

        self.assertIn("vision-first high-level tactical planner", prompt)
        self.assertIn("HLTP VISION AIRCRAFT CONTEXT", prompt)
        self.assertIn("frame_0 is the current snapshot", prompt)
        self.assertIn("image_005.png", prompt)
        self.assertIn("Ignore reward", prompt)
        self.assertIn("origin=ORIG destination=DEST dest=(90.0,10.0)", prompt)
        self.assertIn("min_step=4", prompt)
        self.assertIn("RIGHT: pos=(25.0,5.0)", prompt)
        self.assertIn("MID: pos=(50.0,0.0)", prompt)
        self.assertNotIn("distance_to_conflict", prompt)
        self.assertNotIn("right_offset", prompt)
        self.assertNotIn("route_progress", prompt)
        self.assertNotIn("placeholder only", prompt)
        self.assertIn("distance_to_conflict", text_prompt)
        self.assertIn("right_offset", text_prompt)

    def test_render_labels_two_weather_cells(self) -> None:
        core = MultiAgentSectorCore(num_agents=2, num_weather_cells=2)
        core.reset(seed=42, num_weather_cells=2)
        fig = core.render(show=False)
        try:
            labels = {text.get_text() for text in fig.gca().texts}
            self.assertNotIn("W1", labels)
            self.assertNotIn("W2", labels)
            severity_colors = {"#22c55e", "#facc15", "#dc2626", "#d946ef"}
            weather_patches = [
                patch
                for patch in fig.gca().patches
                if isinstance(patch, Ellipse)
                and to_hex(patch.get_facecolor(), keep_alpha=False) in severity_colors
            ]
            self.assertEqual(len(weather_patches), 8)
            face_colors = {to_hex(patch.get_facecolor(), keep_alpha=False) for patch in weather_patches}
            self.assertEqual(face_colors, severity_colors)
            ghost_patches = [
                patch
                for patch in fig.gca().patches
                if isinstance(patch, Ellipse)
                and to_hex(patch.get_edgecolor(), keep_alpha=False) == "#166534"
                and patch.get_facecolor()[3] == 0.0
            ]
            self.assertEqual(len(ghost_patches), 2 * WEATHER_TRAIL_STEPS)
        finally:
            plt.close(fig)

    def test_global_controller_vision_uses_current_plus_previous_three_frames(self) -> None:
        core = FakeCore([_make_state("A1", position=(10.0, 10.0), has_entered_sector=True)])
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"
            frame_dir.mkdir()
            for step in range(4):
                (frame_dir / f"image_{step:03d}.png").touch()
            controller = _global_controller(tmpdir)
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
        prompt_text = invoke_mock.call_args.args[0]
        self.assertIn("VISION-FIRST GLOBAL GUIDANCE TASK", prompt_text)
        self.assertIn("VISION ACTION CANDIDATES", prompt_text)
        self.assertIn("choose only from the listed turns", prompt_text)
        self.assertIn("turn=+5", prompt_text)
        self.assertIn("Ignore reward", prompt_text)
        self.assertIn("origin=ORIG destination=DEST", prompt_text)
        self.assertIn("dest=(90.0,10.0)", prompt_text)
        self.assertIn("WEATHER MOTION SUMMARY", prompt_text)
        self.assertIn("moving from near", prompt_text)
        self.assertIn("toward", prompt_text)
        self.assertNotIn("RETRIEVED DECISION MEMORY", prompt_text)
        self.assertNotIn("PER-AGENT TURN PREVIEWS", prompt_text)
        self.assertNotIn("each row predicts consequences", prompt_text)
        self.assertEqual(
            [Path(path).name for path in controller.last_normalized["debug"]["frame_paths"]],
            ["image_003.png", "image_002.png", "image_001.png", "image_000.png"],
        )

    def test_global_controller_text_mode_uses_no_image_paths(self) -> None:
        core = FakeCore([_make_state("A1", position=(10.0, 10.0), has_entered_sector=True)])
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = Path(tmpdir) / "image_003.png"
            frame_path.touch()
            controller = _global_controller(tmpdir)
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
        self.assertIn("PER-AGENT TURN PREVIEWS", invoke_mock.call_args.args[0])
        self.assertEqual(controller.last_normalized["debug"]["frame_paths"], [])

    def test_global_controller_vision_invalid_json_still_falls_back(self) -> None:
        core = FakeCore([_make_state("A1", position=(10.0, 10.0), has_entered_sector=True)])
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = Path(tmpdir) / "frames"
            frame_dir.mkdir()
            for step in range(4):
                (frame_dir / f"image_{step:03d}.png").touch()
            controller = _global_controller(tmpdir, retry_limit=0)
            with patch.object(controller, "_ranked_threats", return_value=[]), patch.object(
                controller,
                "_preview_rows",
                return_value=[],
            ), patch.object(controller, "_deterministic_best_turn", return_value=15), patch(
                "rl_llm_multi.llm.ollama_invoke",
                return_value={
                    "llm_status": "ok",
                    "raw_text": "not-json",
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

        self.assertEqual(actions, {"A1": 15})
        self.assertIn("VISION-FIRST GLOBAL GUIDANCE TASK", invoke_mock.call_args.args[0])
        self.assertNotIn("PER-AGENT TURN PREVIEWS", invoke_mock.call_args.args[0])
        self.assertEqual(controller.last_normalized["debug"]["fallback_agents"], ["A1"])

    def test_vision_guardrail_preserves_safe_listed_merge_back_candidate(self) -> None:
        controller = GlobalLangGraphGuidanceController.__new__(GlobalLangGraphGuidanceController)
        core = FakeCore([
            _make_state(
                "A1",
                position=(10.0, 10.0),
                heading_deg=-30.0,
                boundary_warning_active=True,
                has_entered_sector=True,
            )
        ])
        preview_rows = [
            {
                "heading_change_deg": 0,
                "safe_over_preview": True,
                "pair_loss_step": None,
                "weather_entry_step": None,
                "boundary_exit_step": None,
                "min_sep_to_any": 9.0,
                "min_weather_clearance": 6.0,
                "traffic_buffer_ok": True,
                "weather_buffer_ok": True,
                "progress_to_destination": 2.0,
                "cross_track_reduction": 0.0,
                "end_heading_error_to_dest_deg": 25.0,
            },
            {
                "heading_change_deg": 10,
                "safe_over_preview": True,
                "pair_loss_step": None,
                "weather_entry_step": None,
                "boundary_exit_step": None,
                "min_sep_to_any": 8.0,
                "min_weather_clearance": 5.0,
                "traffic_buffer_ok": True,
                "weather_buffer_ok": True,
                "progress_to_destination": 7.0,
                "cross_track_reduction": 2.0,
                "end_heading_error_to_dest_deg": 5.0,
            },
        ]

        state = controller._graph_guardrail_actions(
            {
                "core": core,
                "use_vision": True,
                "actionable": [{"agent_id": "A1", "call_name": "MERGE_BACK"}],
                "candidate_actions": {
                    "A1": {
                        "rationale": "vision intentionally holds for tactical spacing",
                        "maneuver": {"heading_change_deg": 0},
                    }
                },
                "missing_agent_ids": [],
                "preview_rows_by_agent": {"A1": preview_rows},
            }
        )

        self.assertEqual(state["final_actions"]["A1"]["requested_heading_change_deg"], 0)
        self.assertEqual(state["final_actions"]["A1"]["heading_change_deg"], 0)

    def test_vision_guardrail_falls_back_for_unlisted_or_hard_unsafe_turn(self) -> None:
        controller = GlobalLangGraphGuidanceController.__new__(GlobalLangGraphGuidanceController)
        core = FakeCore([_make_state("A1", position=(10.0, 10.0), has_entered_sector=True)])
        preview_rows = [
            {
                "heading_change_deg": 0,
                "safe_over_preview": True,
                "pair_loss_step": None,
                "weather_entry_step": None,
                "boundary_exit_step": None,
                "min_sep_to_any": 9.0,
                "min_weather_clearance": 6.0,
                "traffic_buffer_ok": True,
                "weather_buffer_ok": True,
                "progress_to_destination": 2.0,
                "cross_track_reduction": 0.0,
                "end_heading_error_to_dest_deg": 10.0,
            },
            {
                "heading_change_deg": 10,
                "safe_over_preview": False,
                "pair_loss_step": 2,
                "weather_entry_step": None,
                "boundary_exit_step": None,
                "min_sep_to_any": 3.0,
                "min_weather_clearance": 6.0,
                "traffic_buffer_ok": False,
                "weather_buffer_ok": True,
                "progress_to_destination": 8.0,
                "cross_track_reduction": 2.0,
                "end_heading_error_to_dest_deg": 2.0,
            },
        ]

        unsafe_state = controller._graph_guardrail_actions(
            {
                "core": core,
                "use_vision": True,
                "actionable": [{"agent_id": "A1", "call_name": "MERGE_BACK"}],
                "candidate_actions": {"A1": {"maneuver": {"heading_change_deg": 10}}},
                "missing_agent_ids": [],
                "preview_rows_by_agent": {"A1": preview_rows},
            }
        )
        unlisted_state = controller._graph_guardrail_actions(
            {
                "core": core,
                "use_vision": True,
                "actionable": [{"agent_id": "A1", "call_name": "MERGE_BACK"}],
                "candidate_actions": {"A1": {"maneuver": {"heading_change_deg": -20}}},
                "missing_agent_ids": [],
                "preview_rows_by_agent": {"A1": preview_rows},
            }
        )

        self.assertEqual(unsafe_state["final_actions"]["A1"]["requested_heading_change_deg"], 10)
        self.assertEqual(unsafe_state["final_actions"]["A1"]["heading_change_deg"], 0)
        self.assertEqual(unlisted_state["final_actions"]["A1"]["requested_heading_change_deg"], -20)
        self.assertEqual(unlisted_state["final_actions"]["A1"]["heading_change_deg"], 0)

    def test_vision_guardrail_uses_strongest_recovery_when_all_merge_back_candidates_exit_boundary(self) -> None:
        controller = GlobalLangGraphGuidanceController.__new__(GlobalLangGraphGuidanceController)
        core = FakeCore([_make_state("A1", position=(10.0, 10.0), has_entered_sector=True)])
        preview_rows = [
            {
                "heading_change_deg": 0,
                "safe_over_preview": False,
                "pair_loss_step": None,
                "weather_entry_step": None,
                "boundary_exit_step": 1,
                "min_sep_to_any": 9.0,
                "min_weather_clearance": 6.0,
                "traffic_buffer_ok": True,
                "weather_buffer_ok": True,
                "progress_to_destination": 2.0,
                "cross_track_reduction": 0.0,
                "end_heading_error_to_dest_deg": 25.0,
            },
            {
                "heading_change_deg": 10,
                "safe_over_preview": False,
                "pair_loss_step": None,
                "weather_entry_step": None,
                "boundary_exit_step": 1,
                "min_sep_to_any": 8.0,
                "min_weather_clearance": 5.0,
                "traffic_buffer_ok": True,
                "weather_buffer_ok": True,
                "progress_to_destination": 7.0,
                "cross_track_reduction": 2.0,
                "end_heading_error_to_dest_deg": 5.0,
            },
        ]

        state = controller._graph_guardrail_actions(
            {
                "core": core,
                "use_vision": True,
                "actionable": [{"agent_id": "A1", "call_name": "MERGE_BACK"}],
                "candidate_actions": {"A1": {"maneuver": {"heading_change_deg": 0}}},
                "missing_agent_ids": [],
                "preview_rows_by_agent": {"A1": preview_rows},
            }
        )

        self.assertEqual(state["final_actions"]["A1"]["requested_heading_change_deg"], 0)
        self.assertEqual(state["final_actions"]["A1"]["heading_change_deg"], 10)

    def test_hltp_vision_call_receives_recent_frames(self) -> None:
        core = FakeCore([
            _make_state("A1", position=(10.0, 10.0), has_entered_sector=True, sector_entry_step=5),
        ])
        context = {
            "agent_id": "A1",
            "state": {"position": (10.0, 10.0), "heading_deg": 0.0},
            "route_waypoint_names": ["ORIG", "MERGE"],
            "conflict_partner_ids": ["A2"],
            "conflict": {
                "agent_ids": ["A1", "A2"],
                "loss_step": 4,
                "min_sep": 4.5,
                "point": (20.0, 10.0),
            },
            "right_waypoint_candidates": [
                {
                    "name": "RIGHT",
                    "point": (20.0, 5.0),
                    "route_progress": 20.0,
                    "side_offset": -5.0,
                    "distance_to_conflict": 5.0,
                },
            ],
            "merge_back_candidates": [
                {"name": "MERGE", "point": (40.0, 0.0), "route_progress": 40.0},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_paths = []
            for step in range(4):
                path = Path(tmpdir) / f"image_{3 - step:03d}.png"
                path.touch()
                frame_paths.append(str(path))
            controller = _global_controller(tmpdir)
            with patch.object(controller, "_newly_entered_hltp_agent_ids", return_value=["A1"]), patch.object(
                controller,
                "_hltp_pair_conflicts",
                return_value={("A1", "A2"): {"agent_ids": ("A1", "A2"), "loss_step": 4}},
            ), patch.object(controller, "_hltp_contexts_for_step", return_value={"A1": context}), patch(
                "rl_llm_multi.llm.ollama_invoke",
                return_value={
                    "llm_status": "ok",
                    "raw_text": json.dumps(
                        {
                            "answer": {
                                "plans": {
                                    "A1": {
                                        "first_waypoint_name": "RIGHT",
                                        "merge_back_waypoint_name": "MERGE",
                                        "rationale": "visual right deviation",
                                    }
                                }
                            }
                        }
                    ),
                    "error": "",
                    "used_vision": True,
                },
            ) as invoke_mock:
                state = controller._graph_hltp_on_sector_entry(
                    {
                        "core": core,
                        "step": 5,
                        "annotations": ["A1: ENTERED SECTOR"],
                        "frame_paths": frame_paths,
                        "use_vision": True,
                    }
                )

            normalized = json.loads((Path(tmpdir) / "t005_hltp.normalized.json").read_text())

        self.assertIn("A1", state["hltp_plans_created"])
        self.assertEqual([Path(path).name for path in invoke_mock.call_args.kwargs["image_paths"]], ["image_003.png", "image_002.png", "image_001.png", "image_000.png"])
        self.assertIn("vision-first high-level tactical planner", invoke_mock.call_args.args[0])
        self.assertNotIn("distance_to_conflict", invoke_mock.call_args.args[0])
        self.assertNotIn("right_offset", invoke_mock.call_args.args[0])
        self.assertTrue(normalized["debug"]["used_vision"])
        self.assertEqual(
            [Path(path).name for path in normalized["debug"]["frame_paths"]],
            ["image_003.png", "image_002.png", "image_001.png", "image_000.png"],
        )

    def test_global_controller_smoke_runs_with_two_and_four_agents(self) -> None:
        for num_agents in (2, 4):
            for num_weather_cells in (1, 2):
                with self.subTest(num_agents=num_agents, num_weather_cells=num_weather_cells):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        env = JointGuidanceEnv(
                            num_agents=num_agents,
                            max_agents=MAX_AGENTS,
                            num_weather_cells=num_weather_cells,
                        )
                        env.reset(
                            seed=42,
                            num_agents=num_agents,
                            route_ids=None,
                            num_weather_cells=num_weather_cells,
                        )
                        controller = _global_controller(tmpdir)
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
            for num_weather_cells in (1, 2):
                with self.subTest(num_agents=num_agents, num_weather_cells=num_weather_cells):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        env = JointGuidanceEnv(
                            num_agents=num_agents,
                            max_agents=MAX_AGENTS,
                            num_weather_cells=num_weather_cells,
                        )
                        env.reset(
                            seed=42,
                            num_agents=num_agents,
                            route_ids=None,
                            num_weather_cells=num_weather_cells,
                        )
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
