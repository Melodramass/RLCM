"""Freeze probe MLP architecture, per-budget reward signals, ranking reward,
hidden-state extraction, and the probe training loop."""

import pytest
import torch

from verl.trainer.ppo.forced_output.probe_trainer import BudgetProbeMLP, BudgetProbeTrainer

HIDDEN = 16


class TestBudgetProbeMLP:
    def test_architecture(self):
        mlp = BudgetProbeMLP(HIDDEN)
        layers = list(mlp.net)
        assert isinstance(layers[0], torch.nn.Linear)
        assert layers[0].in_features == HIDDEN
        assert layers[0].out_features == HIDDEN // 4
        assert isinstance(layers[1], torch.nn.ReLU)
        assert isinstance(layers[2], torch.nn.Dropout)
        assert isinstance(layers[3], torch.nn.Linear)
        assert layers[3].out_features == 1

    def test_forward_squeezes_last_dim(self):
        mlp = BudgetProbeMLP(HIDDEN)
        out = mlp(torch.zeros(5, HIDDEN))
        assert out.shape == (5,)

    def test_predict_is_sigmoid_of_forward(self):
        mlp = BudgetProbeMLP(HIDDEN, dropout=0.0)
        mlp.eval()
        x = torch.randn(7, HIDDEN, generator=torch.Generator().manual_seed(0))
        assert torch.allclose(mlp.predict(x), torch.sigmoid(mlp(x)))
        assert ((mlp.predict(x) >= 0) & (mlp.predict(x) <= 1)).all()


class TestRewardSignal:
    """_compute_reward_signal: per-budget scalar reward modes."""

    def test_brier(self):
        preds = torch.tensor([0.9, 0.2, 0.5])
        targets = torch.tensor([1.0, 0.0, 1.0])
        out = BudgetProbeTrainer._compute_reward_signal(preds, targets, "brier")
        expected = torch.tensor([-0.01, -0.04, -0.25])
        assert torch.allclose(out, expected, atol=1e-6)

    def test_cross_entropy(self):
        preds = torch.tensor([0.9, 0.2])
        targets = torch.tensor([1.0, 0.0])
        out = BudgetProbeTrainer._compute_reward_signal(preds, targets, "cross_entropy")
        expected = torch.tensor([0.9, 0.8]).log()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_cross_entropy_clamps_extreme_preds(self):
        preds = torch.tensor([0.0, 1.0])
        targets = torch.tensor([1.0, 0.0])
        out = BudgetProbeTrainer._compute_reward_signal(preds, targets, "cross_entropy")
        assert torch.isfinite(out).all()

    def test_accuracy(self):
        preds = torch.tensor([0.9, 0.2, 0.6, 0.4])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        out = BudgetProbeTrainer._compute_reward_signal(preds, targets, "accuracy")
        assert out.tolist() == [1.0, 0.0, 0.0, 1.0]

    def test_per_sample_modes_rejected(self):
        preds = torch.tensor([0.5])
        targets = torch.tensor([1.0])
        for mode in ("conf_diff", "conf_diff_v1", "conf_diff_v2", "conf_diff_v3",
                     "conf_margin_covar", "ranking_reward"):
            with pytest.raises(ValueError):
                BudgetProbeTrainer._compute_reward_signal(preds, targets, mode)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            BudgetProbeTrainer._compute_reward_signal(
                torch.tensor([0.5]), torch.tensor([1.0]), "bogus"
            )


class TestRankingReward:
    def _trainer_with_preds(self, preds, rt=0.5):
        trainer = BudgetProbeTrainer(HIDDEN, device="cpu", reward_mode="ranking_reward",
                                     ranking_reward_rt=rt)
        trainer.predict = lambda features: torch.tensor(preds, dtype=torch.float32)
        return trainer

    def test_four_quadrants(self):
        # 4 samples x 1 budget; mean confidences: 0.9, 0.3, 0.3, 0.9
        trainer = self._trainer_with_preds([0.9, 0.3, 0.3, 0.9], rt=0.5)
        hs = torch.zeros(4, 16, HIDDEN)
        rewards, metrics = trainer.compute_ranking_rewards(
            hs, [2, 2, 2, 2], [[5], [5], [5], [5]],
            natural_correct=[True, True, False, False],
        )
        # correct & high -> 1, correct & low -> rt, incorrect & low -> -rt,
        # incorrect & high -> -1
        assert rewards == pytest.approx([1.0, 0.5, -0.5, -1.0])
        assert metrics["probe/ranking_high_conf_rate"] == pytest.approx(0.5)

    def test_missing_natural_correct_raises(self):
        trainer = self._trainer_with_preds([0.9])
        hs = torch.zeros(1, 16, HIDDEN)
        with pytest.raises(ValueError, match="natural_correct"):
            trainer.compute_ranking_rewards(hs, [2], [[5]], natural_correct=None)


class TestExtractBudgetHiddenStates:
    def test_takes_mean_of_last_k_positions(self):
        # hidden_states[t] = t for easy averaging
        seq_len, hidden = 10, 4
        hs = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1).expand(seq_len, hidden)
        out = BudgetProbeTrainer.extract_budget_hidden_states(
            hs, prompt_length=2, budget_positions=[3, 6], avg_last_k=2
        )
        # budget 3: positions [3, 4] -> mean 3.5; budget 6: positions [6, 7] -> mean 6.5
        assert out.shape == (2, hidden)
        assert out[0, 0].item() == pytest.approx(3.5)
        assert out[1, 0].item() == pytest.approx(6.5)

    def test_budget_beyond_seq_len_clamps_to_end(self):
        seq_len, hidden = 8, 4
        hs = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1).expand(seq_len, hidden)
        out = BudgetProbeTrainer.extract_budget_hidden_states(
            hs, prompt_length=2, budget_positions=[100], avg_last_k=1
        )
        assert out[0, 0].item() == pytest.approx(7.0)  # last position

    def test_avg_last_k_one_takes_single_position(self):
        seq_len, hidden = 10, 4
        hs = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1).expand(seq_len, hidden)
        out = BudgetProbeTrainer.extract_budget_hidden_states(
            hs, prompt_length=3, budget_positions=[4], avg_last_k=1
        )
        # end_pos = 3 + 4 = 7, single position 6
        assert out[0, 0].item() == pytest.approx(6.0)


class TestProbeTraining:
    def test_train_step_learns_separable_data(self):
        torch.manual_seed(0)
        trainer = BudgetProbeTrainer(HIDDEN, device="cpu", train_steps=50, lr=1e-2, dropout=0.0)
        n = 64
        x = torch.randn(n, HIDDEN)
        # Label depends on the sign of the first feature: linearly separable.
        y = (x[:, 0] > 0).float()
        metrics = trainer.train_step(x, y)
        assert metrics["probe/n_samples"] == n
        assert metrics["probe/binary_accuracy"] > 0.9
        assert metrics["probe/eval_loss"] < 0.5
        # Bucket metrics exposed for conf_diff diagnostics
        assert "probe/mean_conf_correct" in metrics
        assert "probe/mean_conf_incorrect" in metrics

    def test_train_step_empty_input(self):
        trainer = BudgetProbeTrainer(HIDDEN, device="cpu")
        metrics = trainer.train_step(torch.zeros(0, HIDDEN), torch.zeros(0))
        assert metrics == {"probe/loss": 0.0, "probe/n_samples": 0}

    def test_collect_probe_data_shapes_and_targets(self):
        trainer = BudgetProbeTrainer(HIDDEN, device="cpu")
        hs = torch.randn(2, 12, HIDDEN)
        features, targets = trainer.collect_probe_data(
            hs,
            batch_prompt_lengths=[2, 2],
            batch_budget_positions=[[3, 6], [4]],
            batch_mc_accuracies=[[1.0, 0.5], [0.0]],
        )
        assert features.shape == (3, HIDDEN)
        assert targets.tolist() == [1.0, 0.5, 0.0]

    def test_state_dict_roundtrip(self):
        torch.manual_seed(0)
        a = BudgetProbeTrainer(HIDDEN, device="cpu", dropout=0.0)
        b = BudgetProbeTrainer(HIDDEN, device="cpu", dropout=0.0)
        x = torch.randn(4, HIDDEN)
        assert not torch.allclose(a.predict(x), b.predict(x))
        b.load_state_dict(a.state_dict())
        assert torch.allclose(a.predict(x), b.predict(x))
