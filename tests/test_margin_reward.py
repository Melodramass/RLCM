"""Freeze the RLCM margin reward (paper: R_margin; code: reward_mode="conf_diff").

compute_conf_diff_rewards() splits each sample's budget checkpoints into
all-correct (MC accuracy == 1.0) and all-incorrect (MC accuracy == 0.0)
groups, ignores partial-accuracy budgets, and returns
mu_conf_correct - mu_conf_incorrect per sample.

Probe predictions are stubbed so the aggregation logic is tested in
isolation from MLP weights.
"""

import pytest
import torch

from verl.trainer.ppo.forced_output.probe_trainer import BudgetProbeTrainer

HIDDEN = 8


def make_trainer(**kwargs):
    kwargs.setdefault("device", "cpu")
    return BudgetProbeTrainer(HIDDEN, **kwargs)


def stub_predict(trainer, preds):
    """Make the probe return fixed confidences regardless of hidden states."""
    preds = torch.tensor(preds, dtype=torch.float32)
    trainer.predict = lambda features: preds
    return trainer


def fake_batch(budget_lists):
    """Build (hidden_states, prompt_lengths, budget_positions) for n samples."""
    n = len(budget_lists)
    seq_len = 32
    hs = torch.zeros(n, seq_len, HIDDEN)
    prompt_lengths = [4] * n
    return hs, prompt_lengths, budget_lists


class TestConfDiffBothClasses:
    def test_margin_is_mean_correct_minus_mean_incorrect(self):
        trainer = stub_predict(make_trainer(reward_mode="conf_diff"), [0.9, 0.7, 0.2])
        hs, pl, budgets = fake_batch([[5, 10, 15]])
        rewards, metrics = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[1.0, 1.0, 0.0]]
        )
        # mu_correct = (0.9 + 0.7) / 2 = 0.8, mu_incorrect = 0.2
        assert rewards == pytest.approx([0.6], abs=1e-6)
        assert metrics["probe/mean_conf_correct"] == pytest.approx(0.8, abs=1e-6)
        assert metrics["probe/mean_conf_incorrect"] == pytest.approx(0.2, abs=1e-6)

    def test_partial_accuracy_budgets_are_ignored(self):
        trainer = stub_predict(make_trainer(reward_mode="conf_diff"), [0.9, 0.5, 0.1])
        hs, pl, budgets = fake_batch([[5, 10, 15]])
        rewards, _ = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[1.0, 0.5, 0.0]]
        )
        # The 0.5-accuracy budget (conf 0.5) contributes to neither group.
        assert rewards == pytest.approx([0.8], abs=1e-6)

    def test_multiple_samples_split_correctly(self):
        # Sample 0 gets preds [0.8, 0.3]; sample 1 gets preds [0.6, 0.4].
        trainer = stub_predict(make_trainer(reward_mode="conf_diff"), [0.8, 0.3, 0.6, 0.4])
        hs, pl, budgets = fake_batch([[5, 10], [5, 10]])
        rewards, _ = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[1.0, 0.0], [0.0, 1.0]]
        )
        assert rewards == pytest.approx([0.5, -0.2], abs=1e-6)


class TestConfDiffCornerCases:
    """One-class traces: behavior depends on margin_corner and reward_mode."""

    def test_one_class_returns_zero_without_margin_corner(self):
        trainer = stub_predict(make_trainer(reward_mode="conf_diff", margin_corner=False), [0.9, 0.8])
        hs, pl, budgets = fake_batch([[5, 10]])
        rewards, _ = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[1.0, 1.0]]
        )
        assert rewards == [0.0]

    def test_all_correct_with_margin_corner_rewards_confidence(self):
        trainer = stub_predict(make_trainer(reward_mode="conf_diff", margin_corner=True), [0.9, 0.7])
        hs, pl, budgets = fake_batch([[5, 10]])
        rewards, _ = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[1.0, 1.0]]
        )
        assert rewards == pytest.approx([0.8], abs=1e-6)  # conf - 0

    def test_all_incorrect_conf_diff_uses_one_minus_conf(self):
        trainer = stub_predict(make_trainer(reward_mode="conf_diff", margin_corner=True), [0.3, 0.1])
        hs, pl, budgets = fake_batch([[5, 10]])
        rewards, _ = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[0.0, 0.0]]
        )
        assert rewards == pytest.approx([0.8], abs=1e-6)  # 1 - 0.2

    def test_all_incorrect_conf_diff_v1_uses_negative_conf(self):
        trainer = stub_predict(make_trainer(reward_mode="conf_diff_v1", margin_corner=True), [0.3, 0.1])
        hs, pl, budgets = fake_batch([[5, 10]])
        rewards, _ = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[0.0, 0.0]]
        )
        assert rewards == pytest.approx([-0.2], abs=1e-6)  # 0 - 0.2

    def test_conf_diff_v2_corner_uses_tau_and_alpha(self):
        trainer = stub_predict(
            make_trainer(
                reward_mode="conf_diff_v2",
                margin_corner=True,
                conf_diff_v2_tau=0.5,
                conf_diff_v2_alpha=2.0,
            ),
            [0.9],
        )
        hs, pl, budgets = fake_batch([[5]])
        rewards, _ = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[1.0]]
        )
        assert rewards == pytest.approx([2.0 * (0.9 - 0.5)], abs=1e-6)

        trainer = stub_predict(
            make_trainer(
                reward_mode="conf_diff_v2",
                margin_corner=True,
                conf_diff_v2_tau=0.5,
                conf_diff_v2_alpha=2.0,
            ),
            [0.2],
        )
        rewards, _ = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[0.0]]
        )
        assert rewards == pytest.approx([2.0 * (0.5 - 0.2)], abs=1e-6)

    def test_conf_diff_v3_alphas(self):
        common = dict(
            reward_mode="conf_diff_v3",
            margin_corner=True,
            conf_diff_v2_tau=0.4,
            conf_diff_v3_alpha_both=3.0,
            conf_diff_v3_alpha_only_correct=5.0,
            conf_diff_v3_alpha_only_incorrect=7.0,
        )
        # Both classes: alpha_both * margin
        trainer = stub_predict(make_trainer(**common), [0.9, 0.1])
        hs, pl, budgets = fake_batch([[5, 10]])
        rewards, _ = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[1.0, 0.0]]
        )
        assert rewards == pytest.approx([3.0 * 0.8], abs=1e-6)

        # Only correct: alpha_only_correct * (conf - tau)
        trainer = stub_predict(make_trainer(**common), [0.9])
        hs, pl, budgets = fake_batch([[5]])
        rewards, _ = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[1.0]]
        )
        assert rewards == pytest.approx([5.0 * (0.9 - 0.4)], abs=1e-6)

        # Only incorrect: alpha_only_incorrect * (tau - conf)
        trainer = stub_predict(make_trainer(**common), [0.1])
        rewards, _ = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[0.0]]
        )
        assert rewards == pytest.approx([7.0 * (0.4 - 0.1)], abs=1e-6)

    def test_empty_budget_list_yields_zero(self):
        trainer = stub_predict(make_trainer(reward_mode="conf_diff"), [0.9, 0.2])
        hs, pl, budgets = fake_batch([[], [5, 10]])
        rewards, _ = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[], [1.0, 0.0]]
        )
        assert rewards == pytest.approx([0.0, 0.7], abs=1e-6)

    def test_call_site_margin_corner_overrides_trainer_default(self):
        trainer = stub_predict(make_trainer(reward_mode="conf_diff", margin_corner=False), [0.9])
        hs, pl, budgets = fake_batch([[5]])
        rewards, _ = trainer.compute_conf_diff_rewards(
            hs, pl, budgets, batch_mc_accuracies=[[1.0]], margin_corner=True
        )
        assert rewards == pytest.approx([0.9], abs=1e-6)


class TestConfMarginCovar:
    def test_group_covar_factors_are_p_times_one_minus_p(self):
        # Group "a": 2/4 correct -> 0.25; group "b": 1/1 correct -> 0.0
        factors = BudgetProbeTrainer.compute_group_covar_factors(
            uids=["a", "a", "a", "a", "b"],
            natural_correct=[True, True, False, False, True],
        )
        assert factors.tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25, 0.0], abs=1e-6)

    def test_apply_conf_margin_covar_scales_rewards(self):
        scaled, factors = BudgetProbeTrainer.apply_conf_margin_covar(
            base_rewards=[1.0, -0.5, 0.8, 0.8, 1.0],
            uids=["a", "a", "a", "a", "b"],
            natural_correct=[True, True, False, False, True],
        )
        assert scaled == pytest.approx([0.25, -0.125, 0.2, 0.2, 0.0], abs=1e-6)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            BudgetProbeTrainer.apply_conf_margin_covar(
                base_rewards=[1.0],
                uids=["a", "b"],
                natural_correct=[True, False],
            )


class TestRewardModeValidation:
    def test_unknown_reward_mode_rejected(self):
        with pytest.raises(ValueError, match="Unknown probe reward_mode"):
            make_trainer(reward_mode="nonsense")

    def test_all_documented_modes_accepted(self):
        for mode in BudgetProbeTrainer.VALID_REWARD_MODES:
            make_trainer(reward_mode=mode)
