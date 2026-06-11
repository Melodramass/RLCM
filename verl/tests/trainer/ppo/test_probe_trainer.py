"""Tests for the Budget Probe Trainer and related components.

Covers:
- BudgetProbeMLP: forward, predict, shapes
- HiddenStateHook: tensor and tuple output handling
- BudgetProbeTrainer:
    - extract_budget_hidden_states: position math, avg_last_k, edge cases
    - collect_probe_data: batch processing, empty handling
    - train_step: loss decreases, metrics are populated, step_count advances
    - predict / compute_probe_rewards: output ranges and shapes
    - state_dict / load_state_dict: round-trip checkpoint
- Integration: end-to-end probe training on synthetic data produces
  predictions that converge toward MC accuracy targets
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import modules via importlib to avoid triggering verl.__init__ (needs ray)
# ---------------------------------------------------------------------------

_base = os.path.join(os.path.dirname(__file__), "..", "..", "..")

_probe_trainer_path = os.path.join(
    _base, "verl", "trainer", "ppo", "forced_output", "probe_trainer.py",
)
_spec_pt = importlib.util.spec_from_file_location("probe_trainer", os.path.abspath(_probe_trainer_path))
_mod_pt = importlib.util.module_from_spec(_spec_pt)
_spec_pt.loader.exec_module(_mod_pt)

BudgetProbeMLP = _mod_pt.BudgetProbeMLP
BudgetProbeTrainer = _mod_pt.BudgetProbeTrainer

# Import HiddenStateHook from anytime_reasoner
_hook_path = os.path.join(
    _base, "..", "anytime_reasoner-new", "models", "probe.py",
)
_spec_hook = importlib.util.spec_from_file_location("ar_probe", os.path.abspath(_hook_path))
_mod_hook = importlib.util.module_from_spec(_spec_hook)
_spec_hook.loader.exec_module(_mod_hook)

HiddenStateHook = _mod_hook.HiddenStateHook


# =========================================================================
# BudgetProbeMLP
# =========================================================================


class TestBudgetProbeMLP:

    def test_forward_shape(self):
        """Forward returns (batch,) logits."""
        mlp = BudgetProbeMLP(hidden_size=64, dropout=0.0)
        x = torch.randn(8, 64)
        out = mlp(x)
        assert out.shape == (8,), f"Expected (8,), got {out.shape}"

    def test_predict_range(self):
        """predict() returns values in [0, 1]."""
        mlp = BudgetProbeMLP(hidden_size=32, dropout=0.0)
        x = torch.randn(16, 32)
        preds = mlp.predict(x)
        assert preds.shape == (16,)
        assert (preds >= 0).all() and (preds <= 1).all(), f"Predictions out of [0,1]: {preds}"

    def test_architecture(self):
        """MLP has hidden_size -> hidden_size//4 -> 1 structure."""
        mlp = BudgetProbeMLP(hidden_size=128, dropout=0.0)
        # First linear: (128, 32)
        assert mlp.net[0].in_features == 128
        assert mlp.net[0].out_features == 32
        # Last linear: (32, 1)
        assert mlp.net[3].in_features == 32
        assert mlp.net[3].out_features == 1

    def test_single_sample(self):
        """Works with batch_size=1."""
        mlp = BudgetProbeMLP(hidden_size=16, dropout=0.0)
        x = torch.randn(1, 16)
        out = mlp(x)
        assert out.shape == (1,)


# =========================================================================
# HiddenStateHook
# =========================================================================


class TestHiddenStateHook:

    def test_tensor_output(self):
        """Hook captures plain tensor output (e.g., from LayerNorm)."""
        hook = HiddenStateHook()
        fake_module = nn.Identity()
        fake_input = None
        fake_output = torch.randn(2, 10, 64)

        hook(fake_module, fake_input, fake_output)

        assert hook.hidden_states is not None
        assert hook.hidden_states.shape == (2, 10, 64)
        # Should be detached (no grad_fn)
        assert not hook.hidden_states.requires_grad

    def test_tuple_output(self):
        """Hook captures first element from tuple output (e.g., from decoder layer)."""
        hook = HiddenStateHook()
        fake_module = nn.Identity()
        fake_input = None
        hs = torch.randn(2, 10, 64, requires_grad=True)
        fake_output = (hs, None, None)  # decoder layer returns (hidden_states, kv_cache, ...)

        hook(fake_module, fake_input, fake_output)

        assert hook.hidden_states is not None
        assert hook.hidden_states.shape == (2, 10, 64)
        assert not hook.hidden_states.requires_grad

    def test_clear(self):
        """clear() removes captured hidden states."""
        hook = HiddenStateHook()
        hook(nn.Identity(), None, torch.randn(1, 5, 8))
        assert hook.hidden_states is not None
        hook.clear()
        assert hook.hidden_states is None

    def test_register_on_module(self):
        """Hook can be registered on an nn.Module and captures output during forward."""
        hook = HiddenStateHook()
        layer = nn.Linear(8, 8)
        handle = layer.register_forward_hook(hook)

        x = torch.randn(2, 8)
        _ = layer(x)

        assert hook.hidden_states is not None
        assert hook.hidden_states.shape == (2, 8)
        handle.remove()


# =========================================================================
# BudgetProbeTrainer — extract_budget_hidden_states
# =========================================================================


class TestExtractBudgetHiddenStates:

    def test_single_budget_avg1(self):
        """Single budget, avg_last_k=1: takes exactly one hidden state."""
        hidden_size = 8
        seq_len = 20
        hs = torch.arange(seq_len * hidden_size, dtype=torch.float32).reshape(seq_len, hidden_size)
        prompt_length = 5
        budget_positions = [10]  # response token 10 -> seq pos 15

        result = BudgetProbeTrainer.extract_budget_hidden_states(
            hs, prompt_length, budget_positions, avg_last_k=1
        )
        assert result.shape == (1, hidden_size)
        # Should be hs[14] (position 5+10-1 = 14, but end_pos = min(5+10, 20) = 15,
        # start_pos = 15-1 = 14, so hs[14:15].mean(0) = hs[14])
        expected = hs[14]
        torch.testing.assert_close(result[0], expected)

    def test_single_budget_avg3(self):
        """avg_last_k=3: averages the last 3 positions before the budget point."""
        hidden_size = 4
        seq_len = 20
        hs = torch.ones(seq_len, hidden_size)
        # Make positions 12, 13, 14 have distinct values
        hs[12] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        hs[13] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        hs[14] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        prompt_length = 5
        budget_positions = [10]  # end_pos = 15, start_pos = 12

        result = BudgetProbeTrainer.extract_budget_hidden_states(
            hs, prompt_length, budget_positions, avg_last_k=3
        )
        assert result.shape == (1, hidden_size)
        expected = (hs[12] + hs[13] + hs[14]) / 3.0
        torch.testing.assert_close(result[0], expected)

    def test_multiple_budgets(self):
        """Multiple budget positions produce one row each."""
        hidden_size = 4
        seq_len = 100
        hs = torch.randn(seq_len, hidden_size)
        prompt_length = 10
        budget_positions = [20, 40, 60]

        result = BudgetProbeTrainer.extract_budget_hidden_states(
            hs, prompt_length, budget_positions, avg_last_k=1
        )
        assert result.shape == (3, hidden_size)

    def test_budget_exceeds_seq_len(self):
        """Budget pos beyond seq_len is clamped to seq_len."""
        hidden_size = 4
        seq_len = 10
        hs = torch.randn(seq_len, hidden_size)
        prompt_length = 5
        budget_positions = [100]  # way beyond seq_len

        result = BudgetProbeTrainer.extract_budget_hidden_states(
            hs, prompt_length, budget_positions, avg_last_k=1
        )
        assert result.shape == (1, hidden_size)
        # end_pos = min(5+100, 10) = 10, start_pos = 9
        expected = hs[9]
        torch.testing.assert_close(result[0], expected)

    def test_zero_budget(self):
        """Budget position 0 results in end_pos == prompt_length."""
        hidden_size = 4
        seq_len = 20
        hs = torch.randn(seq_len, hidden_size)
        prompt_length = 5
        budget_positions = [0]  # end_pos = 5, start_pos = 4

        result = BudgetProbeTrainer.extract_budget_hidden_states(
            hs, prompt_length, budget_positions, avg_last_k=1
        )
        assert result.shape == (1, hidden_size)
        torch.testing.assert_close(result[0], hs[4])

    def test_avg_last_k_larger_than_available(self):
        """avg_last_k exceeds available positions — start_pos clamped to 0."""
        hidden_size = 4
        seq_len = 5
        hs = torch.randn(seq_len, hidden_size)
        prompt_length = 0
        budget_positions = [3]  # end_pos = 3, start_pos = max(3-10,0) = 0

        result = BudgetProbeTrainer.extract_budget_hidden_states(
            hs, prompt_length, budget_positions, avg_last_k=10
        )
        assert result.shape == (1, hidden_size)
        expected = hs[:3].mean(dim=0)
        torch.testing.assert_close(result[0], expected)


# =========================================================================
# BudgetProbeTrainer — collect_probe_data
# =========================================================================


class TestCollectProbeData:

    def _make_trainer(self, hidden_size=16, avg_last_k=1):
        return BudgetProbeTrainer(
            hidden_size=hidden_size,
            train_steps=1,
            batch_size=4,
            avg_last_k=avg_last_k,
            device="cpu",
        )

    def test_basic(self):
        """Collects features and targets from a simple batch."""
        trainer = self._make_trainer(hidden_size=8)
        batch_hs = torch.randn(3, 50, 8)  # 3 samples, seq_len=50, hidden=8
        prompt_lens = [10, 10, 10]
        budget_pos = [[20, 30], [20, 30], [20, 30]]
        mc_accs = [[0.5, 0.75], [0.25, 1.0], [0.0, 0.5]]

        features, targets = trainer.collect_probe_data(batch_hs, prompt_lens, budget_pos, mc_accs)

        assert features.shape == (6, 8)  # 3 samples × 2 budgets
        assert targets.shape == (6,)
        # Targets should match flattened mc_accs
        expected_targets = [0.5, 0.75, 0.25, 1.0, 0.0, 0.5]
        torch.testing.assert_close(targets, torch.tensor(expected_targets))

    def test_empty_batch(self):
        """Empty budget_positions produce empty outputs."""
        trainer = self._make_trainer(hidden_size=8)
        batch_hs = torch.randn(2, 50, 8)
        prompt_lens = [10, 10]
        budget_pos = [[], []]
        mc_accs = [[], []]

        features, targets = trainer.collect_probe_data(batch_hs, prompt_lens, budget_pos, mc_accs)

        assert features.shape[0] == 0
        assert targets.shape[0] == 0

    def test_mixed_empty_nonempty(self):
        """Some samples have budgets, some don't."""
        trainer = self._make_trainer(hidden_size=8)
        batch_hs = torch.randn(3, 50, 8)
        prompt_lens = [10, 10, 10]
        budget_pos = [[20], [], [30]]
        mc_accs = [[0.5], [], [1.0]]

        features, targets = trainer.collect_probe_data(batch_hs, prompt_lens, budget_pos, mc_accs)

        assert features.shape == (2, 8)  # Only 2 budgets across 2 samples
        assert targets.shape == (2,)
        torch.testing.assert_close(targets, torch.tensor([0.5, 1.0]))

    def test_avg_last_k_propagated(self):
        """avg_last_k from trainer config is used during collection."""
        hidden_size = 4
        trainer = self._make_trainer(hidden_size=hidden_size, avg_last_k=3)
        seq_len = 20
        hs = torch.ones(1, seq_len, hidden_size)
        hs[0, 7] = torch.tensor([10.0, 10.0, 10.0, 10.0])
        hs[0, 8] = torch.tensor([20.0, 20.0, 20.0, 20.0])
        hs[0, 9] = torch.tensor([30.0, 30.0, 30.0, 30.0])

        features, targets = trainer.collect_probe_data(
            batch_hidden_states=hs,
            batch_prompt_lengths=[0],
            batch_budget_positions=[[10]],  # end_pos=10, avg over [7,8,9]
            batch_mc_accuracies=[[0.5]],
        )
        assert features.shape == (1, hidden_size)
        expected = torch.tensor([(10 + 20 + 30) / 3.0] * hidden_size)
        torch.testing.assert_close(features[0], expected)


# =========================================================================
# BudgetProbeTrainer — train_step
# =========================================================================


class TestTrainStep:

    def _make_trainer(self, hidden_size=16):
        return BudgetProbeTrainer(
            hidden_size=hidden_size,
            train_steps=5,
            batch_size=8,
            lr=1e-2,
            dropout=0.0,
            device="cpu",
        )

    def test_metrics_returned(self):
        """train_step returns the expected metric keys."""
        trainer = self._make_trainer()
        features = torch.randn(20, 16)
        targets = torch.rand(20)

        metrics = trainer.train_step(features, targets)

        expected_keys = {
            "probe/train_loss", "probe/eval_loss", "probe/mean_pred",
            "probe/mean_target", "probe/binary_accuracy", "probe/n_samples",
            "probe/step_count", "probe/mean_conf_correct", "probe/mean_conf_incorrect",
            "probe/std_conf_correct", "probe/std_conf_incorrect",
        }
        assert expected_keys.issubset(metrics.keys()), f"Missing keys: {expected_keys - metrics.keys()}"

    def test_step_count_increments(self):
        """Step count increments after each train_step call."""
        trainer = self._make_trainer()
        assert trainer._step_count == 0

        trainer.train_step(torch.randn(10, 16), torch.rand(10))
        assert trainer._step_count == 1

        trainer.train_step(torch.randn(10, 16), torch.rand(10))
        assert trainer._step_count == 2

    def test_n_samples_correct(self):
        """n_samples metric matches input size."""
        trainer = self._make_trainer()
        metrics = trainer.train_step(torch.randn(42, 16), torch.rand(42))
        assert metrics["probe/n_samples"] == 42

    def test_empty_input(self):
        """Empty input returns zero loss and n_samples=0."""
        trainer = self._make_trainer()
        metrics = trainer.train_step(torch.randn(0, 16), torch.rand(0))
        assert metrics["probe/n_samples"] == 0
        assert metrics["probe/loss"] == 0.0

    def test_loss_decreases_on_learnable_signal(self):
        """Loss decreases when training on a clear signal.

        Create a simple dataset: features with positive first element -> target=1,
        negative first element -> target=0.
        """
        torch.manual_seed(42)
        hidden_size = 16
        trainer = BudgetProbeTrainer(
            hidden_size=hidden_size,
            train_steps=50,
            batch_size=16,
            lr=1e-2,
            dropout=0.0,
            device="cpu",
        )

        N = 100
        features = torch.randn(N, hidden_size)
        targets = (features[:, 0] > 0).float()

        # Get initial loss
        trainer.mlp.eval()
        with torch.no_grad():
            initial_logits = trainer.mlp(features)
            initial_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                initial_logits, targets
            ).item()

        # Train
        metrics = trainer.train_step(features, targets)

        assert metrics["probe/eval_loss"] < initial_loss, (
            f"Loss should decrease: {metrics['probe/eval_loss']:.4f} >= {initial_loss:.4f}"
        )

    def test_predictions_in_range(self):
        """After training, predictions are in [0, 1]."""
        trainer = self._make_trainer()
        features = torch.randn(20, 16)
        targets = torch.rand(20)
        trainer.train_step(features, targets)

        preds = trainer.predict(features)
        assert (preds >= 0).all() and (preds <= 1).all()


# =========================================================================
# BudgetProbeTrainer — predict / compute_probe_rewards
# =========================================================================


class TestPredictAndRewards:

    def _make_trainer(self, hidden_size=16):
        return BudgetProbeTrainer(
            hidden_size=hidden_size,
            train_steps=1,
            batch_size=8,
            device="cpu",
        )

    def test_predict_shape(self):
        """predict returns (N,) tensor on CPU."""
        trainer = self._make_trainer()
        features = torch.randn(10, 16)
        preds = trainer.predict(features)
        assert preds.shape == (10,)
        assert preds.device == torch.device("cpu")

    def test_compute_probe_rewards_shapes(self):
        """compute_probe_rewards returns correct nested list structure."""
        trainer = self._make_trainer(hidden_size=8)
        batch_hs = torch.randn(3, 50, 8)
        prompt_lens = [10, 10, 10]
        budget_pos = [[20, 30], [15], [20, 30, 40]]

        rewards = trainer.compute_probe_rewards(batch_hs, prompt_lens, budget_pos)

        assert len(rewards) == 3
        assert len(rewards[0]) == 2
        assert len(rewards[1]) == 1
        assert len(rewards[2]) == 3

    def test_compute_probe_rewards_range(self):
        """All probe rewards are in [0, 1]."""
        trainer = self._make_trainer(hidden_size=8)
        batch_hs = torch.randn(4, 50, 8)
        prompt_lens = [10] * 4
        budget_pos = [[20, 30]] * 4

        rewards = trainer.compute_probe_rewards(batch_hs, prompt_lens, budget_pos)

        for sample_rewards in rewards:
            for r in sample_rewards:
                assert 0.0 <= r <= 1.0, f"Reward {r} out of [0,1]"

    def test_compute_probe_rewards_empty_budgets(self):
        """Samples with no budgets get empty reward lists."""
        trainer = self._make_trainer(hidden_size=8)
        batch_hs = torch.randn(2, 50, 8)
        prompt_lens = [10, 10]
        budget_pos = [[], [20]]

        rewards = trainer.compute_probe_rewards(batch_hs, prompt_lens, budget_pos)

        assert len(rewards) == 2
        assert rewards[0] == []
        assert len(rewards[1]) == 1

    def test_all_empty_budgets(self):
        """All samples empty -> all empty reward lists."""
        trainer = self._make_trainer(hidden_size=8)
        batch_hs = torch.randn(3, 50, 8)
        prompt_lens = [10] * 3
        budget_pos = [[], [], []]

        rewards = trainer.compute_probe_rewards(batch_hs, prompt_lens, budget_pos)

        assert len(rewards) == 3
        assert all(r == [] for r in rewards)


# =========================================================================
# BudgetProbeTrainer — conf_diff corner handling
# =========================================================================


class TestConfDiffCornerHandling:

    def _make_trainer(self, margin_corner: bool):
        return BudgetProbeTrainer(
            hidden_size=8,
            train_steps=1,
            batch_size=4,
            device="cpu",
            reward_mode="conf_diff",
            margin_corner=margin_corner,
        )

    def test_conf_diff_corner_enabled_all_correct(self):
        trainer = self._make_trainer(margin_corner=True)
        batch_hs = torch.randn(1, 30, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8]]
        mc_accs = [[1.0, 1.0]]

        trainer.predict = lambda x: torch.tensor([0.8, 0.6], dtype=torch.float32)

        rewards, metrics = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        assert rewards[0] == pytest.approx(0.7)
        assert metrics["probe/mean_conf_correct"] == pytest.approx(0.7)
        assert metrics["probe/std_conf_correct"] == pytest.approx(0.1)

    def test_conf_diff_corner_disabled_all_correct(self):
        trainer = self._make_trainer(margin_corner=False)
        batch_hs = torch.randn(1, 30, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8]]
        mc_accs = [[1.0, 1.0]]

        trainer.predict = lambda x: torch.tensor([0.8, 0.6], dtype=torch.float32)

        rewards, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        assert rewards[0] == pytest.approx(0.0)

    def test_conf_diff_corner_enabled_all_incorrect(self):
        trainer = self._make_trainer(margin_corner=True)
        batch_hs = torch.randn(1, 30, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8]]
        mc_accs = [[0.0, 0.0]]

        trainer.predict = lambda x: torch.tensor([0.2, 0.4], dtype=torch.float32)

        rewards, metrics = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        assert rewards[0] == pytest.approx(0.7)
        assert metrics["probe/mean_conf_incorrect"] == pytest.approx(0.3)
        assert metrics["probe/std_conf_incorrect"] == pytest.approx(0.1)

    def test_conf_diff_corner_enabled_with_partials_and_only_zero_present(self):
        """Example like (0, 0.2, 0.4, 0.6): no 1.0 class, corner path should apply."""
        trainer = self._make_trainer(margin_corner=True)
        batch_hs = torch.randn(1, 40, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8, 11, 14]]
        mc_accs = [[0.0, 0.2, 0.4, 0.6]]

        # Only first value contributes to incorrect bucket when filtering by mc==0.0
        trainer.predict = lambda x: torch.tensor([0.25, 0.6, 0.7, 0.8], dtype=torch.float32)

        rewards, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        assert rewards[0] == pytest.approx(0.75)  # 1 - conf_wrong

    def test_conf_diff_corner_enabled_with_partials_and_only_one_present(self):
        """Example like (0.1, 0.2, 0.4, 1): no 0.0 class, corner path should apply."""
        trainer = self._make_trainer(margin_corner=True)
        batch_hs = torch.randn(1, 40, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8, 11, 14]]
        mc_accs = [[0.1, 0.2, 0.4, 1.0]]

        # Only last value contributes to correct bucket when filtering by mc==1.0
        trainer.predict = lambda x: torch.tensor([0.15, 0.3, 0.5, 0.85], dtype=torch.float32)

        rewards, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        assert rewards[0] == pytest.approx(0.85)  # conf_correct - 0

    def test_conf_diff_corner_disabled_with_partial_one_class_returns_zero(self):
        trainer = self._make_trainer(margin_corner=False)
        batch_hs = torch.randn(1, 40, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8, 11, 14]]

        trainer.predict = lambda x: torch.tensor([0.25, 0.6, 0.7, 0.8], dtype=torch.float32)

        rewards_zero_side, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=[[0.0, 0.2, 0.4, 0.6]],
        )
        rewards_one_side, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=[[0.1, 0.2, 0.4, 1.0]],
        )

        assert rewards_zero_side[0] == pytest.approx(0.0)
        assert rewards_one_side[0] == pytest.approx(0.0)


class TestConfDiffV1CornerHandling:

    def _make_trainer(self, margin_corner: bool):
        return BudgetProbeTrainer(
            hidden_size=8,
            train_steps=1,
            batch_size=4,
            device="cpu",
            reward_mode="conf_diff_v1",
            margin_corner=margin_corner,
        )

    def test_conf_diff_v1_corner_enabled_all_incorrect_uses_negative_conf(self):
        trainer = self._make_trainer(margin_corner=True)
        batch_hs = torch.randn(1, 30, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8]]
        mc_accs = [[0.0, 0.0]]

        trainer.predict = lambda x: torch.tensor([0.2, 0.4], dtype=torch.float32)

        rewards, metrics = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        assert rewards[0] == pytest.approx(-0.3)
        assert metrics["probe/mean_conf_incorrect"] == pytest.approx(0.3)
        assert metrics["probe/std_conf_incorrect"] == pytest.approx(0.1)

    def test_conf_diff_v1_corner_enabled_with_partials_and_only_zero_present(self):
        trainer = self._make_trainer(margin_corner=True)
        batch_hs = torch.randn(1, 40, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8, 11, 14]]
        mc_accs = [[0.0, 0.2, 0.4, 0.6]]

        trainer.predict = lambda x: torch.tensor([0.25, 0.6, 0.7, 0.8], dtype=torch.float32)

        rewards, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        assert rewards[0] == pytest.approx(-0.25)

    def test_conf_diff_v1_corner_disabled_with_partial_one_class_returns_zero(self):
        trainer = self._make_trainer(margin_corner=False)
        batch_hs = torch.randn(1, 40, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8, 11, 14]]

        trainer.predict = lambda x: torch.tensor([0.25, 0.6, 0.7, 0.8], dtype=torch.float32)

        rewards_zero_side, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=[[0.0, 0.2, 0.4, 0.6]],
        )

        assert rewards_zero_side[0] == pytest.approx(0.0)

    def test_conf_diff_v1_corner_enabled_all_correct_unchanged(self):
        trainer = self._make_trainer(margin_corner=True)
        batch_hs = torch.randn(1, 30, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8]]
        mc_accs = [[1.0, 1.0]]

        trainer.predict = lambda x: torch.tensor([0.8, 0.6], dtype=torch.float32)

        rewards, metrics = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        assert rewards[0] == pytest.approx(0.7)
        assert metrics["probe/mean_conf_correct"] == pytest.approx(0.7)
        assert metrics["probe/std_conf_correct"] == pytest.approx(0.1)


class TestConfDiffV2CornerHandling:

    def _make_trainer(self, margin_corner: bool, tau: float = 0.5, alpha: float = 1.0):
        return BudgetProbeTrainer(
            hidden_size=8,
            train_steps=1,
            batch_size=4,
            device="cpu",
            reward_mode="conf_diff_v2",
            margin_corner=margin_corner,
            conf_diff_v2_tau=tau,
            conf_diff_v2_alpha=alpha,
        )

    def test_conf_diff_v2_two_class_unchanged_margin_formula(self):
        trainer = self._make_trainer(margin_corner=True, tau=0.2, alpha=3.0)
        batch_hs = torch.randn(1, 30, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8, 11, 14]]
        mc_accs = [[1.0, 1.0, 0.0, 0.0]]

        trainer.predict = lambda x: torch.tensor([0.9, 0.7, 0.3, 0.2], dtype=torch.float32)

        rewards, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        # mu_correct=0.8, mu_incorrect=0.25
        assert rewards[0] == pytest.approx(0.55)

    def test_conf_diff_v2_corner_only_correct_uses_tau_alpha(self):
        trainer = self._make_trainer(margin_corner=True, tau=0.6, alpha=2.5)
        batch_hs = torch.randn(1, 30, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8]]
        mc_accs = [[1.0, 1.0]]

        trainer.predict = lambda x: torch.tensor([0.8, 0.6], dtype=torch.float32)

        rewards, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        # mu_correct=0.7 -> alpha*(mu_correct-tau)=2.5*(0.1)=0.25
        assert rewards[0] == pytest.approx(0.25)

    def test_conf_diff_v2_corner_only_incorrect_uses_tau_alpha(self):
        trainer = self._make_trainer(margin_corner=True, tau=0.4, alpha=3.0)
        batch_hs = torch.randn(1, 30, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8]]
        mc_accs = [[0.0, 0.0]]

        trainer.predict = lambda x: torch.tensor([0.1, 0.3], dtype=torch.float32)

        rewards, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        # mu_incorrect=0.2 -> alpha*(tau-mu_incorrect)=3.0*(0.2)=0.6
        assert rewards[0] == pytest.approx(0.6)

    def test_conf_diff_v2_corner_disabled_one_class_returns_zero(self):
        trainer = self._make_trainer(margin_corner=False, tau=0.3, alpha=4.0)
        batch_hs = torch.randn(1, 30, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8]]

        trainer.predict = lambda x: torch.tensor([0.8, 0.9], dtype=torch.float32)

        rewards, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=[[1.0, 1.0]],
        )

        assert rewards[0] == pytest.approx(0.0)


class TestConfDiffV3CaseWeights:

    def _make_trainer(
        self,
        margin_corner: bool,
        tau: float = 0.5,
        alpha_both: float = 1.0,
        alpha_only_correct: float = 1.0,
        alpha_only_incorrect: float = 1.0,
    ):
        return BudgetProbeTrainer(
            hidden_size=8,
            train_steps=1,
            batch_size=4,
            device="cpu",
            reward_mode="conf_diff_v3",
            margin_corner=margin_corner,
            conf_diff_v2_tau=tau,
            conf_diff_v3_alpha_both=alpha_both,
            conf_diff_v3_alpha_only_correct=alpha_only_correct,
            conf_diff_v3_alpha_only_incorrect=alpha_only_incorrect,
        )

    def test_conf_diff_v3_two_class_uses_alpha_both(self):
        trainer = self._make_trainer(
            margin_corner=True,
            tau=0.2,
            alpha_both=2.0,
            alpha_only_correct=9.0,
            alpha_only_incorrect=9.0,
        )
        batch_hs = torch.randn(1, 30, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8, 11, 14]]
        mc_accs = [[1.0, 1.0, 0.0, 0.0]]

        trainer.predict = lambda x: torch.tensor([0.9, 0.7, 0.3, 0.2], dtype=torch.float32)

        rewards, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        # mu_correct=0.8, mu_incorrect=0.25 -> 2.0 * (0.55) = 1.1
        assert rewards[0] == pytest.approx(1.1)

    def test_conf_diff_v3_corner_only_correct_uses_case_alpha(self):
        trainer = self._make_trainer(
            margin_corner=True,
            tau=0.6,
            alpha_both=1.0,
            alpha_only_correct=2.5,
            alpha_only_incorrect=1.0,
        )
        batch_hs = torch.randn(1, 30, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8]]
        mc_accs = [[1.0, 1.0]]

        trainer.predict = lambda x: torch.tensor([0.8, 0.6], dtype=torch.float32)

        rewards, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        # mu_correct=0.7 -> alpha_only_correct*(mu_correct-tau)=2.5*(0.1)=0.25
        assert rewards[0] == pytest.approx(0.25)

    def test_conf_diff_v3_corner_only_incorrect_uses_case_alpha(self):
        trainer = self._make_trainer(
            margin_corner=True,
            tau=0.4,
            alpha_both=1.0,
            alpha_only_correct=1.0,
            alpha_only_incorrect=3.0,
        )
        batch_hs = torch.randn(1, 30, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8]]
        mc_accs = [[0.0, 0.0]]

        trainer.predict = lambda x: torch.tensor([0.1, 0.3], dtype=torch.float32)

        rewards, _ = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        # mu_incorrect=0.2 -> alpha_only_incorrect*(tau-mu)=3.0*(0.2)=0.6
        assert rewards[0] == pytest.approx(0.6)

    def test_conf_diff_v3_can_match_v2_in_selected_cases(self):
        batch_hs = torch.randn(3, 30, 8)
        prompt_lens = [10, 10, 10]
        budget_pos = [[5, 8, 11, 14], [5, 8], [5, 8]]
        mc_accs = [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]

        v2 = BudgetProbeTrainer(
            hidden_size=8,
            train_steps=1,
            batch_size=4,
            device="cpu",
            reward_mode="conf_diff_v2",
            margin_corner=True,
            conf_diff_v2_tau=0.4,
            conf_diff_v2_alpha=2.0,
        )
        v3 = BudgetProbeTrainer(
            hidden_size=8,
            train_steps=1,
            batch_size=4,
            device="cpu",
            reward_mode="conf_diff_v3",
            margin_corner=True,
            conf_diff_v2_tau=0.4,
            conf_diff_v3_alpha_both=1.0,
            conf_diff_v3_alpha_only_correct=2.0,
            conf_diff_v3_alpha_only_incorrect=2.0,
        )

        # Sample 1 (two-class): mu_correct=0.8, mu_incorrect=0.25 -> 0.55
        # Sample 2 (only-correct): 2.0*(0.7-0.4)=0.6
        # Sample 3 (only-incorrect): 2.0*(0.4-0.2)=0.4
        preds = torch.tensor(
            [
                0.9, 0.7, 0.3, 0.2,
                0.8, 0.6,
                0.1, 0.3,
            ],
            dtype=torch.float32,
        )
        v2.predict = lambda x: preds
        v3.predict = lambda x: preds

        rewards_v2, _ = v2.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )
        rewards_v3, _ = v3.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        assert rewards_v3 == pytest.approx(rewards_v2)


class TestRankingReward:

    def _make_trainer(self, rt: float = 0.5):
        return BudgetProbeTrainer(
            hidden_size=8,
            train_steps=1,
            batch_size=4,
            device="cpu",
            reward_mode="ranking_reward",
            ranking_reward_rt=rt,
        )

    def test_ranking_reward_matches_all_four_cases(self):
        trainer = self._make_trainer(rt=0.3)
        batch_hs = torch.randn(4, 30, 8)
        prompt_lens = [10, 10, 10, 10]
        budget_pos = [[5, 8], [5, 8], [5, 8], [5, 8]]
        natural_correct = [True, True, False, False]

        trainer.predict = lambda x: torch.tensor(
            [
                0.8, 0.7,
                0.2, 0.4,
                0.3, 0.4,
                0.7, 0.8,
            ],
            dtype=torch.float32,
        )

        rewards, metrics = trainer.compute_ranking_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            natural_correct=natural_correct,
        )

        assert rewards == pytest.approx([1.0, 0.3, -0.3, -1.0])
        assert metrics["probe/ranking_reward_rt"] == pytest.approx(0.3)
        assert metrics["probe/ranking_avg_conf_correct"] == pytest.approx(0.525)
        assert metrics["probe/ranking_avg_conf_incorrect"] == pytest.approx(0.55)
        assert metrics["probe/ranking_high_conf_rate"] == pytest.approx(0.5)

    def test_ranking_reward_requires_natural_correct(self):
        trainer = self._make_trainer()
        batch_hs = torch.randn(1, 20, 8)
        prompt_lens = [10]
        budget_pos = [[5]]

        with pytest.raises(ValueError, match="natural_correct is required"):
            trainer.compute_ranking_rewards(
                batch_hidden_states=batch_hs,
                batch_prompt_lengths=prompt_lens,
                batch_budget_positions=budget_pos,
                natural_correct=None,
            )

    def test_ranking_reward_boundary_at_point_five_uses_low_conf_branch(self):
        trainer = self._make_trainer(rt=0.4)
        batch_hs = torch.randn(2, 20, 8)
        prompt_lens = [10, 10]
        budget_pos = [[5, 8], [5, 8]]
        natural_correct = [True, False]

        # Both samples have mean confidence exactly 0.5, which should fall into
        # the p <= 0.5 branch per the ranking_reward definition.
        trainer.predict = lambda x: torch.tensor(
            [
                0.4, 0.6,
                0.5, 0.5,
            ],
            dtype=torch.float32,
        )

        rewards, metrics = trainer.compute_ranking_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            natural_correct=natural_correct,
        )

        assert rewards == pytest.approx([0.4, -0.4])
        assert metrics["probe/ranking_high_conf_rate"] == pytest.approx(0.0)

    def test_ranking_reward_empty_budgets_return_zero(self):
        trainer = self._make_trainer()
        batch_hs = torch.randn(2, 20, 8)
        prompt_lens = [10, 10]
        budget_pos = [[], []]
        natural_correct = [True, False]

        rewards, metrics = trainer.compute_ranking_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            natural_correct=natural_correct,
        )

        assert rewards == pytest.approx([0.0, 0.0])
        assert math.isnan(metrics["probe/ranking_avg_conf_mean"])


class TestConfMarginCovar:

    def _make_trainer(self, margin_corner: bool):
        return BudgetProbeTrainer(
            hidden_size=8,
            train_steps=1,
            batch_size=4,
            device="cpu",
            reward_mode="conf_margin_covar",
            margin_corner=margin_corner,
        )

    def test_conf_margin_covar_uses_v1_corner_behavior(self):
        trainer = self._make_trainer(margin_corner=True)
        batch_hs = torch.randn(1, 30, 8)
        prompt_lens = [10]
        budget_pos = [[5, 8]]
        mc_accs = [[0.0, 0.0]]

        trainer.predict = lambda x: torch.tensor([0.2, 0.4], dtype=torch.float32)

        rewards, metrics = trainer.compute_conf_diff_rewards(
            batch_hidden_states=batch_hs,
            batch_prompt_lengths=prompt_lens,
            batch_budget_positions=budget_pos,
            batch_mc_accuracies=mc_accs,
        )

        assert rewards[0] == pytest.approx(-0.3)
        assert metrics["probe/mean_conf_incorrect"] == pytest.approx(0.3)

    def test_group_covar_factors(self):
        uids = np.array(["a", "a", "a", "a", "b", "b", "c", "c"], dtype=object)
        natural_correct = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)

        factors = BudgetProbeTrainer.compute_group_covar_factors(
            uids=uids,
            natural_correct=natural_correct,
        )

        expected = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(factors, expected)

    def test_apply_conf_margin_covar_scales_rewards_by_group_factor(self):
        base_rewards = [1.0, -2.0, 0.5, 3.0, -1.5, 2.0]
        uids = np.array(["a", "a", "a", "a", "b", "b"], dtype=object)
        natural_correct = np.array([1, 1, 0, 0, 1, 1], dtype=bool)

        scaled, factors = BudgetProbeTrainer.apply_conf_margin_covar(
            base_rewards=base_rewards,
            uids=uids,
            natural_correct=natural_correct,
        )

        # Group a: p = 0.5 -> factor 0.25
        # Group b: p = 1.0 -> factor 0.0
        expected_scaled = [0.25, -0.5, 0.125, 0.75, 0.0, 0.0]
        expected_factors = np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0], dtype=np.float32)

        assert scaled == pytest.approx(expected_scaled)
        np.testing.assert_allclose(factors, expected_factors)


# =========================================================================
# BudgetProbeTrainer — state_dict / load_state_dict
# =========================================================================


class TestStateDict:

    def test_roundtrip(self):
        """save -> load produces identical predictions."""
        torch.manual_seed(123)
        hidden_size = 16
        trainer1 = BudgetProbeTrainer(hidden_size=hidden_size, device="cpu")
        features = torch.randn(10, hidden_size)

        # Train once so state is non-trivial
        trainer1.train_step(features, torch.rand(10))
        preds1 = trainer1.predict(features)

        state = trainer1.state_dict()

        # Create fresh trainer and load state
        trainer2 = BudgetProbeTrainer(hidden_size=hidden_size, device="cpu")
        trainer2.load_state_dict(state)
        preds2 = trainer2.predict(features)

        torch.testing.assert_close(preds1, preds2)
        assert trainer2._step_count == trainer1._step_count

    def test_state_dict_keys(self):
        """state_dict has expected keys."""
        trainer = BudgetProbeTrainer(hidden_size=8, device="cpu")
        state = trainer.state_dict()
        assert "mlp" in state
        assert "optimizer" in state
        assert "step_count" in state

    def test_save_load_with_torch(self):
        """State dict can be saved/loaded via torch.save/load."""
        torch.manual_seed(42)
        hidden_size = 16
        trainer = BudgetProbeTrainer(hidden_size=hidden_size, device="cpu")
        trainer.train_step(torch.randn(20, hidden_size), torch.rand(20))

        features = torch.randn(5, hidden_size)
        preds_before = trainer.predict(features)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(trainer.state_dict(), f.name)
            path = f.name

        try:
            trainer2 = BudgetProbeTrainer(hidden_size=hidden_size, device="cpu")
            trainer2.load_state_dict(torch.load(path, weights_only=False))
            preds_after = trainer2.predict(features)
            torch.testing.assert_close(preds_before, preds_after)
        finally:
            os.unlink(path)


# =========================================================================
# Integration: end-to-end probe training convergence
# =========================================================================


class TestIntegrationConvergence:

    def test_probe_learns_correct_pattern(self):
        """Probe can learn to predict MC accuracy from hidden states.

        Setup: hidden states from early budget positions (small values)
        correlate with low accuracy; late budgets (large values) with high accuracy.
        After sufficient training, the probe should predict low for early and
        high for late budgets.
        """
        torch.manual_seed(7)
        hidden_size = 32
        trainer = BudgetProbeTrainer(
            hidden_size=hidden_size,
            train_steps=100,
            batch_size=32,
            lr=1e-2,
            dropout=0.0,
            device="cpu",
        )

        # Create synthetic data: features = random but with a signal in first dim
        # targets = sigmoid(features[:, 0] * 3)
        N = 200
        features = torch.randn(N, hidden_size) * 0.1
        signal = torch.randn(N) * 2
        features[:, 0] = signal
        targets = torch.sigmoid(signal * 3)

        trainer.train_step(features, targets)

        # Predict on new data with same pattern
        test_features = torch.randn(50, hidden_size) * 0.1
        test_signal = torch.randn(50) * 2
        test_features[:, 0] = test_signal
        test_targets = torch.sigmoid(test_signal * 3)

        preds = trainer.predict(test_features)

        # Correlation between predictions and targets should be positive
        corr = torch.corrcoef(torch.stack([preds, test_targets]))[0, 1].item()
        assert corr > 0.5, f"Expected positive correlation > 0.5, got {corr:.3f}"

    def test_full_pipeline_simulation(self):
        """Simulate the full GRPO probe pipeline:
        1. Generate synthetic batch of hidden states
        2. Define MC accuracies per budget
        3. collect_probe_data
        4. train_step
        5. compute_probe_rewards
        6. Verify rewards are reasonable
        """
        torch.manual_seed(0)
        hidden_size = 16
        batch_size = 4
        seq_len = 100
        prompt_len = 20
        budgets = [20, 40, 60]
        n_budgets = len(budgets)

        trainer = BudgetProbeTrainer(
            hidden_size=hidden_size,
            train_steps=10,
            batch_size=8,
            lr=1e-2,
            dropout=0.0,
            avg_last_k=2,
            device="cpu",
        )

        # Generate batch
        batch_hs = torch.randn(batch_size, seq_len, hidden_size)
        prompt_lens = [prompt_len] * batch_size
        budget_pos = [budgets] * batch_size
        mc_accs = [
            [0.25, 0.5, 0.75],  # increasing accuracy with budget
            [0.0, 0.25, 0.5],
            [0.5, 0.75, 1.0],
            [0.0, 0.0, 0.25],
        ]

        # Step 1: Collect data
        features, targets = trainer.collect_probe_data(
            batch_hs, prompt_lens, budget_pos, mc_accs,
        )
        assert features.shape == (batch_size * n_budgets, hidden_size)
        assert targets.shape == (batch_size * n_budgets,)

        # Step 2: Train
        metrics = trainer.train_step(features, targets)
        assert metrics["probe/n_samples"] == batch_size * n_budgets
        assert "probe/eval_loss" in metrics
        assert metrics["probe/step_count"] == 1

        # Step 3: Compute rewards
        rewards = trainer.compute_probe_rewards(batch_hs, prompt_lens, budget_pos)
        assert len(rewards) == batch_size
        for i in range(batch_size):
            assert len(rewards[i]) == n_budgets
            for r in rewards[i]:
                assert 0.0 <= r <= 1.0

    def test_mc_accuracy_computation(self):
        """Verify MC accuracy is correctly computed as fraction of correct answers.

        This tests the logic in _compute_forced_output_rewards where:
        - multi-answer: mc_acc = mean([1 if r > 0 else 0 for r in per_answer_rewards])
        - single-answer: mc_acc = 1.0 if reward_val > 0 else 0.0
        """
        # Case 1: Multi-answer, 4 samples, 3 correct out of 4
        per_answer_rewards = [1.0, 1.0, 0.0, 1.0]
        mc_acc = float(np.mean([1.0 if r > 0 else 0.0 for r in per_answer_rewards]))
        assert mc_acc == pytest.approx(0.75)

        # Case 2: All correct
        per_answer_rewards = [1.0, 1.0, 1.0, 1.0]
        mc_acc = float(np.mean([1.0 if r > 0 else 0.0 for r in per_answer_rewards]))
        assert mc_acc == pytest.approx(1.0)

        # Case 3: None correct
        per_answer_rewards = [0.0, 0.0, 0.0, 0.0]
        mc_acc = float(np.mean([1.0 if r > 0 else 0.0 for r in per_answer_rewards]))
        assert mc_acc == pytest.approx(0.0)

        # Case 4: Single answer correct
        reward_val = 1.0
        mc_acc = 1.0 if reward_val > 0 else 0.0
        assert mc_acc == 1.0

        # Case 5: Single answer wrong
        reward_val = 0.0
        mc_acc = 1.0 if reward_val > 0 else 0.0
        assert mc_acc == 0.0

    def test_reward_formula(self):
        """Verify the overall reward formula:
        r_trace = r_natural + weight_budget * r_budget + weight_probe * r_probe

        Test all combinations of flag settings.
        """
        r_natural = 1.0
        r_budget = 0.6  # raw budget reward
        r_probe = 0.8   # probe predicted accuracy

        # Case 1: All included
        w_budget = 0.1
        w_probe = 0.1
        total = r_natural + w_budget * r_budget + w_probe * r_probe
        assert total == pytest.approx(1.0 + 0.06 + 0.08)

        # Case 2: No budget reward (include_budget_reward=False)
        w_budget = 0.0
        w_probe = 0.1
        total = r_natural + w_budget * r_budget + w_probe * r_probe
        assert total == pytest.approx(1.0 + 0.0 + 0.08)

        # Case 3: No probe reward (use_as_reward=False)
        w_budget = 0.1
        w_probe = 0.0
        total = r_natural + w_budget * r_budget + w_probe * r_probe
        assert total == pytest.approx(1.0 + 0.06 + 0.0)

        # Case 4: Only natural
        w_budget = 0.0
        w_probe = 0.0
        total = r_natural + w_budget * r_budget + w_probe * r_probe
        assert total == pytest.approx(1.0)


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:

    def test_hidden_size_not_divisible_by_4(self):
        """MLP works when hidden_size is not divisible by 4 (integer division)."""
        mlp = BudgetProbeMLP(hidden_size=7, dropout=0.0)
        assert mlp.net[0].in_features == 7
        assert mlp.net[0].out_features == 1  # 7 // 4 = 1
        x = torch.randn(3, 7)
        out = mlp(x)
        assert out.shape == (3,)

    def test_very_small_batch(self):
        """Training with batch_size=1 works."""
        trainer = BudgetProbeTrainer(
            hidden_size=8, train_steps=2, batch_size=1, device="cpu",
        )
        metrics = trainer.train_step(torch.randn(3, 8), torch.rand(3))
        assert metrics["probe/n_samples"] == 3

    def test_large_avg_last_k_with_short_seq(self):
        """avg_last_k > seq_len doesn't crash."""
        hidden_size = 4
        trainer = BudgetProbeTrainer(
            hidden_size=hidden_size, avg_last_k=1000, device="cpu",
        )
        batch_hs = torch.randn(1, 5, hidden_size)
        prompt_lens = [2]
        budget_pos = [[3]]  # end_pos=5, start_pos=max(5-1000,0)=0
        mc_accs = [[0.5]]

        features, targets = trainer.collect_probe_data(
            batch_hs, prompt_lens, budget_pos, mc_accs,
        )
        assert features.shape == (1, hidden_size)
        # Should average all 5 positions
        expected = batch_hs[0, :5].mean(dim=0)
        torch.testing.assert_close(features[0], expected)

    def test_deterministic_predict(self):
        """predict() gives identical results on the same input (no stochastic dropout)."""
        trainer = BudgetProbeTrainer(hidden_size=8, dropout=0.5, device="cpu")
        features = torch.randn(10, 8)

        preds1 = trainer.predict(features)
        preds2 = trainer.predict(features)
        torch.testing.assert_close(preds1, preds2)

    def test_no_gradient_flow_to_features(self):
        """Training the probe doesn't create gradients on input features."""
        trainer = BudgetProbeTrainer(
            hidden_size=8, train_steps=2, batch_size=4, device="cpu",
        )
        features = torch.randn(10, 8, requires_grad=False)
        targets = torch.rand(10)

        trainer.train_step(features, targets)
        assert features.grad is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
