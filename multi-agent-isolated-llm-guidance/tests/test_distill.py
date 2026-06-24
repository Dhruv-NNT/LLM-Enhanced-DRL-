"""Unit tests for dual-teacher distillation (teacher nets, JS loss, competence,
dataset round-trip, decay, and buffer wiring)."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from configs import ACTION_BINS  # noqa: E402
from rl_llm_multi.mappo import RolloutBuffer, ActionRecord, _js_divergence  # noqa: E402
from rl_llm_multi.teacher import (  # noqa: E402
    TeacherPolicy,
    load_teacher,
    load_trajectory_dataset,
    save_teacher,
    save_trajectory_shard,
    write_dataset_meta,
)
from rl_llm_multi import distill  # noqa: E402


ACTION_DIM = len(ACTION_BINS)


def _random_log_probs(batch: int, dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, dim, generator=g)
    return torch.log_softmax(logits, dim=-1)


class TestJSDivergence(unittest.TestCase):
    def test_zero_when_equal(self):
        logp = _random_log_probs(8, ACTION_DIM, seed=1)
        probs = logp.exp()
        js = _js_divergence(logp, probs)
        self.assertTrue(torch.all(js >= -1e-6))
        self.assertTrue(torch.all(js < 1e-5), f"JS(p,p) should be ~0, got {js}")

    def test_bounded_by_ln2(self):
        logp = _random_log_probs(16, ACTION_DIM, seed=2)
        # An unrelated teacher distribution.
        teacher = torch.log_softmax(torch.randn(16, ACTION_DIM, generator=torch.Generator().manual_seed(3)), dim=-1).exp()
        js = _js_divergence(logp, teacher)
        self.assertTrue(torch.all(js >= -1e-6))
        self.assertTrue(torch.all(js <= math.log(2.0) + 1e-4), f"JS exceeds ln2: max={js.max().item()}")

    def test_symmetric(self):
        p_logp = _random_log_probs(8, ACTION_DIM, seed=4)
        q_logp = _random_log_probs(8, ACTION_DIM, seed=5)
        js_pq = _js_divergence(p_logp, q_logp.exp())
        js_qp = _js_divergence(q_logp, p_logp.exp())
        self.assertTrue(torch.allclose(js_pq, js_qp, atol=1e-5))

    def test_gradient_flows_to_student_only(self):
        logits = torch.randn(4, ACTION_DIM, requires_grad=True)
        student_logp = torch.log_softmax(logits, dim=-1)
        teacher = torch.softmax(torch.randn(4, ACTION_DIM), dim=-1)  # constant target
        loss = _js_divergence(student_logp, teacher).mean()
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())


class TestTeacherPolicy(unittest.TestCase):
    def test_action_logits_shape(self):
        teacher = TeacherPolicy(local_obs_dim=40, action_dim=ACTION_DIM, max_agents=12)
        obs = torch.randn(5, 40)
        idx = torch.tensor([0, 1, 2, 3, 4])
        logits = teacher.action_logits(obs, idx)
        self.assertEqual(tuple(logits.shape), (5, ACTION_DIM))

    def test_input_stats_applied(self):
        teacher = TeacherPolicy(local_obs_dim=6, action_dim=ACTION_DIM, max_agents=4)
        mean = torch.arange(6, dtype=torch.float32)
        std = torch.full((6,), 2.0)
        teacher.set_input_stats(mean, std)
        self.assertTrue(torch.allclose(teacher.input_mean, mean))
        self.assertTrue(torch.allclose(teacher.input_std, std))
        # std clamped away from zero
        teacher.set_input_stats(torch.zeros(6), torch.zeros(6))
        self.assertTrue(torch.all(teacher.input_std >= 1e-6))

    def test_save_load_roundtrip(self):
        teacher = TeacherPolicy(local_obs_dim=8, action_dim=ACTION_DIM, max_agents=4)
        teacher.set_input_stats(torch.randn(8), torch.rand(8) + 1.0)
        obs = torch.randn(3, 8)
        idx = torch.tensor([0, 1, 2])
        with torch.no_grad():
            expected = teacher.action_logits(obs, idx)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.pt"
            save_teacher(teacher, path, meta={"note": "x"})
            loaded = load_teacher(path, map_location="cpu")
            with torch.no_grad():
                got = loaded.action_logits(obs, idx)
        self.assertTrue(torch.allclose(expected, got, atol=1e-5))


class TestDatasetRoundtrip(unittest.TestCase):
    def test_shard_save_and_load(self):
        n = 17
        obs = np.random.randn(n, 40).astype(np.float32)
        agent_index = np.random.randint(0, 12, size=n).astype(np.int64)
        action_idx = np.random.randint(0, ACTION_DIM, size=n).astype(np.int64)
        episode_id = np.zeros(n, dtype=np.int64)
        step = np.arange(n, dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            save_trajectory_shard(
                tmp, 0, local_obs=obs, agent_index=agent_index,
                action_idx=action_idx, episode_id=episode_id, step=step,
            )
            write_dataset_meta(tmp, {"obs_dim": 40, "action_dim": ACTION_DIM})
            data = load_trajectory_dataset(tmp)
        self.assertEqual(len(data), n)
        self.assertEqual(data.obs_dim, 40)
        self.assertTrue(np.allclose(data.local_obs, obs))
        self.assertTrue(np.array_equal(data.action_idx, action_idx))
        mean, std = data.input_stats()
        self.assertEqual(mean.shape[0], 40)
        self.assertTrue(np.all(std > 0))


class TestWeatherRule(unittest.TestCase):
    def _fake_core(self, clearance, buffer_units=4.0):
        state = SimpleNamespace(predicted_weather_clearance=clearance)
        return SimpleNamespace(
            weather_clearance_buffer_units=buffer_units,
            get_agent=lambda aid: state,
            agent_states={"A1": state},
        )

    def test_near_weather_favors_heuristic(self):
        core = self._fake_core(clearance=0.1)  # very close to weather
        w_llm, w_heur = distill.weather_rule_weights(core, "A1")
        self.assertGreater(w_heur, w_llm)

    def test_far_weather_favors_llm(self):
        core = self._fake_core(clearance=1000.0)  # far from weather
        w_llm, w_heur = distill.weather_rule_weights(core, "A1")
        self.assertGreater(w_llm, w_heur)

    def test_infinite_clearance_is_far(self):
        core = self._fake_core(clearance=float("inf"))
        w_llm, w_heur = distill.weather_rule_weights(core, "A1")
        self.assertGreater(w_llm, w_heur)


class TestDistillWeightDecay(unittest.TestCase):
    def test_decay_monotonic_within_window(self):
        # Uses configured DISTILL_* values; just check shape of the schedule.
        from configs import DISTILL_DECAY_END_STEP, DISTILL_DECAY_START_STEP, DISTILL_LOSS_WEIGHT, DISTILL_MIN_WEIGHT
        start = int(DISTILL_DECAY_START_STEP)
        end = int(DISTILL_DECAY_END_STEP)
        w_start = distill.current_distill_weight(start)
        w_mid = distill.current_distill_weight((start + end) // 2)
        w_end = distill.current_distill_weight(end)
        w_past = distill.current_distill_weight(end + 10_000)
        self.assertGreaterEqual(w_start + 1e-9, w_mid)
        self.assertGreaterEqual(w_mid + 1e-9, w_end)
        self.assertAlmostEqual(w_end, w_past, places=6)
        self.assertGreaterEqual(w_past, float(DISTILL_MIN_WEIGHT) - 1e-9)
        self.assertLessEqual(w_start, float(DISTILL_LOSS_WEIGHT) + 1e-9)


class TestBufferDistillFields(unittest.TestCase):
    def _record(self):
        return ActionRecord(
            episode_id=0,
            agent_id="A1",
            agent_index=0,
            local_obs=torch.zeros(40),
            global_state=torch.zeros(40),
            action=torch.tensor(0),
            logprob=torch.tensor(0.0),
            value=torch.tensor(0.0),
        )

    def test_distill_metadata_stored(self):
        buf = RolloutBuffer()
        probs = torch.softmax(torch.randn(ACTION_DIM), dim=-1)
        meta = {
            "distill_label_available": True,
            "distill_phase": "",
            "teacher_llm_probs": probs,
            "teacher_heur_probs": probs,
            "teacher_llm_action_idx": 3,
            "teacher_heur_action_idx": 5,
            "distill_weight_llm": 0.4,
            "distill_weight_heur": 0.7,
        }
        buf.add(self._record(), reward=1.0, terminal=False, llm_metadata=meta)
        self.assertEqual(buf.distill_available, [True])
        self.assertEqual(buf.teacher_llm_action_indices, [3])
        self.assertEqual(buf.teacher_heur_action_indices, [5])
        self.assertAlmostEqual(buf.distill_weight_llm[0], 0.4)
        self.assertAlmostEqual(buf.distill_weight_heur[0], 0.7)
        self.assertEqual(tuple(buf.teacher_llm_probs[0].shape), (ACTION_DIM,))

    def test_placeholder_when_absent(self):
        buf = RolloutBuffer()
        buf.add(self._record(), reward=0.0, terminal=False, llm_metadata={})
        self.assertEqual(buf.distill_available, [False])
        self.assertEqual(tuple(buf.teacher_llm_probs[0].shape), (ACTION_DIM,))
        self.assertTrue(torch.allclose(buf.teacher_llm_probs[0], torch.zeros(ACTION_DIM)))


class TestTeacherGate(unittest.TestCase):
    """DISTILL_TEACHERS must enable single-teacher ablations (T1/T2)."""

    def _teachers(self, obs_dim=6):
        return {
            "llm": TeacherPolicy(local_obs_dim=obs_dim, action_dim=ACTION_DIM, max_agents=4),
            "heur": TeacherPolicy(local_obs_dim=obs_dim, action_dim=ACTION_DIM, max_agents=4),
        }

    def _meta(self):
        obs = {"A1": np.zeros(6, dtype=np.float32)}
        return distill.build_distill_metadata(
            None, None, obs, ["A1"], teachers=self._teachers(6),
            proposed_actions={}, competence_mode="equal", episode_id=0,
        )["A1"]

    def test_llm_only(self):
        orig = distill.DISTILL_TEACHERS
        try:
            distill.DISTILL_TEACHERS = ("llm",)
            meta = self._meta()
            self.assertEqual(meta["distill_weight_llm"], 1.0)
            self.assertEqual(meta["distill_weight_heur"], 0.0)
        finally:
            distill.DISTILL_TEACHERS = orig

    def test_heur_only(self):
        orig = distill.DISTILL_TEACHERS
        try:
            distill.DISTILL_TEACHERS = ("heur",)
            meta = self._meta()
            self.assertEqual(meta["distill_weight_llm"], 0.0)
            self.assertEqual(meta["distill_weight_heur"], 1.0)
        finally:
            distill.DISTILL_TEACHERS = orig

    def test_both(self):
        orig = distill.DISTILL_TEACHERS
        try:
            distill.DISTILL_TEACHERS = ("llm", "heur")
            meta = self._meta()
            self.assertEqual(meta["distill_weight_llm"], 1.0)
            self.assertEqual(meta["distill_weight_heur"], 1.0)
        finally:
            distill.DISTILL_TEACHERS = orig


if __name__ == "__main__":
    unittest.main()
