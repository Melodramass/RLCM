"""Tests for C2GSPG policy loss (compute_policy_loss_c2gspg).

Covers:
- BCE calibration term activates only when flags are on
- With bce_beta=0 or c2_binary_reward=None, output is identical to vanilla PPO
- BCE loss value is mathematically correct
- Metrics are logged correctly
- get_policy_loss_fn("c2gspg") resolves to the right function
- GRPO / vanilla loss_mode ignores c2_binary_reward entirely
"""

import math
import types
import unittest

import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_c2gspg, get_policy_loss_fn


def _make_config(bce_beta: float = 0.0, clip_ratio: float = 0.2) -> types.SimpleNamespace:
    """Minimal ActorConfig-like object sufficient for compute_policy_loss_c2gspg."""
    policy_loss = types.SimpleNamespace(
        loss_mode="c2gspg",
        c2gspg_bce_beta=bce_beta,
    )
    # BaseConfig.get() is used in the real code; patch it here.
    policy_loss.get = lambda key, default=None: getattr(policy_loss, key, default)

    cfg = types.SimpleNamespace(
        clip_ratio=clip_ratio,
        clip_ratio_low=clip_ratio,
        clip_ratio_high=clip_ratio,
        clip_ratio_c=3.0,
        loss_agg_mode="token-mean",
        global_batch_info={},
        policy_loss=policy_loss,
    )
    cfg.get = lambda key, default=None: getattr(cfg, key, default)
    return cfg


def _make_inputs(bsz: int = 4, seq_len: int = 8, seed: int = 0):
    """Random tensors for a mini-batch."""
    torch.manual_seed(seed)
    log_prob = torch.randn(bsz, seq_len) * 0.1 - 2.0   # small negative
    old_log_prob = log_prob + torch.randn(bsz, seq_len) * 0.05
    advantages = torch.randn(bsz, seq_len)
    response_mask = torch.ones(bsz, seq_len, dtype=torch.bool)
    # mask out last 2 tokens to test masking
    response_mask[:, -2:] = False
    return log_prob, old_log_prob, advantages, response_mask


class TestC2GSPGPolicyLossRegistration(unittest.TestCase):
    def test_registered_under_c2gspg(self):
        fn = get_policy_loss_fn("c2gspg")
        self.assertIs(fn, compute_policy_loss_c2gspg)

    def test_vanilla_still_resolves(self):
        from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
        fn = get_policy_loss_fn("vanilla")
        self.assertIs(fn, compute_policy_loss_vanilla)


class TestC2GSPGPolicyLossNoBCE(unittest.TestCase):
    """When BCE is disabled, output must equal vanilla PPO exactly."""

    def _vanilla_loss(self, log_prob, old_log_prob, advantages, response_mask, cfg):
        from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
        return compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=cfg.loss_agg_mode,
            config=cfg,
        )

    def test_beta_zero_matches_vanilla(self):
        log_prob, old_log_prob, advantages, response_mask = _make_inputs()
        cfg = _make_config(bce_beta=0.0)

        c2_loss, c2_metrics = compute_policy_loss_c2gspg(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=cfg.loss_agg_mode,
            config=cfg,
            c2_binary_reward=torch.ones(4),
        )
        v_loss, v_metrics = self._vanilla_loss(log_prob, old_log_prob, advantages, response_mask, cfg)

        self.assertAlmostEqual(c2_loss.item(), v_loss.item(), places=6)
        self.assertAlmostEqual(c2_metrics["actor/pg_clipfrac"], v_metrics["actor/pg_clipfrac"], places=6)
        self.assertNotIn("actor/c2gspg_bce_loss", c2_metrics)

    def test_no_c2_binary_reward_matches_vanilla(self):
        log_prob, old_log_prob, advantages, response_mask = _make_inputs(seed=1)
        cfg = _make_config(bce_beta=0.5)  # beta > 0 but no labels

        c2_loss, c2_metrics = compute_policy_loss_c2gspg(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=cfg.loss_agg_mode,
            config=cfg,
            c2_binary_reward=None,
        )
        v_loss, _ = self._vanilla_loss(log_prob, old_log_prob, advantages, response_mask, cfg)

        self.assertAlmostEqual(c2_loss.item(), v_loss.item(), places=6)
        self.assertNotIn("actor/c2gspg_bce_loss", c2_metrics)

    def test_omitted_c2_binary_reward_kwarg_matches_vanilla(self):
        """c2_binary_reward defaults to None if not passed at all."""
        log_prob, old_log_prob, advantages, response_mask = _make_inputs(seed=2)
        cfg = _make_config(bce_beta=1.0)

        c2_loss, c2_metrics = compute_policy_loss_c2gspg(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=cfg.loss_agg_mode,
            config=cfg,
        )
        v_loss, _ = self._vanilla_loss(log_prob, old_log_prob, advantages, response_mask, cfg)

        self.assertAlmostEqual(c2_loss.item(), v_loss.item(), places=6)


class TestC2GSPGPolicyLossBCE(unittest.TestCase):
    """When BCE is enabled with labels, loss should differ from vanilla."""

    def test_bce_adds_positive_term(self):
        log_prob, old_log_prob, advantages, response_mask = _make_inputs(seed=3)
        cfg = _make_config(bce_beta=1.0)
        c2_binary_reward = torch.tensor([1.0, 0.0, 1.0, 0.0])

        c2_loss, c2_metrics = compute_policy_loss_c2gspg(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=cfg.loss_agg_mode,
            config=cfg,
            c2_binary_reward=c2_binary_reward,
        )

        self.assertIn("actor/c2gspg_bce_loss", c2_metrics)
        self.assertIn("actor/c2gspg_bce_beta", c2_metrics)
        self.assertEqual(c2_metrics["actor/c2gspg_bce_beta"], 1.0)
        # BCE loss should be non-negative
        self.assertGreaterEqual(c2_metrics["actor/c2gspg_bce_loss"], 0.0)

    def test_bce_term_is_additive(self):
        """total_loss = pg_loss + beta * bce_loss."""
        log_prob, old_log_prob, advantages, response_mask = _make_inputs(seed=4)
        cfg_no_bce = _make_config(bce_beta=0.0)
        cfg_bce = _make_config(bce_beta=2.0)
        c2_binary_reward = torch.tensor([1.0, 1.0, 0.0, 0.0])

        pg_loss, _ = compute_policy_loss_c2gspg(
            old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
            response_mask=response_mask, loss_agg_mode=cfg_no_bce.loss_agg_mode,
            config=cfg_no_bce, c2_binary_reward=c2_binary_reward,
        )
        total_loss, metrics = compute_policy_loss_c2gspg(
            old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
            response_mask=response_mask, loss_agg_mode=cfg_bce.loss_agg_mode,
            config=cfg_bce, c2_binary_reward=c2_binary_reward,
        )

        expected = pg_loss.item() + 2.0 * metrics["actor/c2gspg_bce_loss"]
        self.assertAlmostEqual(total_loss.item(), expected, places=5)

    def test_bce_value_manual(self):
        """Verify BCE computation on a hand-crafted example."""
        bsz, seq_len = 2, 4
        # log_prob = -1.0 everywhere -> avg log prob = -1.0 -> conf = exp(-1)
        log_prob = torch.full((bsz, seq_len), -1.0)
        old_log_prob = torch.full((bsz, seq_len), -1.0)
        advantages = torch.zeros(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len, dtype=torch.bool)
        c2_binary_reward = torch.tensor([1.0, 0.0])  # seq0 correct, seq1 incorrect

        cfg = _make_config(bce_beta=1.0)

        _, metrics = compute_policy_loss_c2gspg(
            old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
            response_mask=response_mask, loss_agg_mode="token-mean",
            config=cfg, c2_binary_reward=c2_binary_reward,
        )

        eps = 1e-6
        conf = math.exp(-1.0)
        conf = max(eps, min(1.0 - eps, conf))
        # seq0 (r=1): -log(conf); seq1 (r=0): -log(1-conf)
        bce_seq0 = -math.log(conf)
        bce_seq1 = -math.log(1.0 - conf)
        expected_bce = (bce_seq0 + bce_seq1) / 2.0  # mean over 2 sequences

        self.assertAlmostEqual(metrics["actor/c2gspg_bce_loss"], expected_bce, places=4)

    def test_high_conf_correct_low_bce(self):
        """A highly confident correct sequence should have low BCE loss."""
        bsz, seq_len = 2, 8
        # very high confidence (log_prob near 0)
        log_prob_high = torch.full((bsz, seq_len), -0.01)
        # very low confidence
        log_prob_low = torch.full((bsz, seq_len), -5.0)
        old_log_prob = torch.zeros(bsz, seq_len)
        advantages = torch.zeros(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len, dtype=torch.bool)
        all_correct = torch.ones(bsz)
        cfg = _make_config(bce_beta=1.0)

        _, metrics_high = compute_policy_loss_c2gspg(
            old_log_prob=old_log_prob, log_prob=log_prob_high, advantages=advantages,
            response_mask=response_mask, config=cfg, c2_binary_reward=all_correct,
        )
        _, metrics_low = compute_policy_loss_c2gspg(
            old_log_prob=old_log_prob, log_prob=log_prob_low, advantages=advantages,
            response_mask=response_mask, config=cfg, c2_binary_reward=all_correct,
        )
        # High confidence on correct sequences → lower BCE
        self.assertLess(metrics_high["actor/c2gspg_bce_loss"], metrics_low["actor/c2gspg_bce_loss"])


class TestC2GSPGFallbackToGRPO(unittest.TestCase):
    """When loss_mode is 'vanilla', c2_binary_reward must be ignored entirely."""

    def test_vanilla_loss_mode_ignores_c2_binary_reward(self):
        from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
        log_prob, old_log_prob, advantages, response_mask = _make_inputs(seed=5)
        cfg = _make_config(bce_beta=0.0)  # vanilla config

        loss_vanilla, _ = compute_policy_loss_vanilla(
            old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
            response_mask=response_mask, config=cfg,
        )

        # Simulate what dp_actor.py does for loss_mode != "c2gspg": no extra kwargs
        fn = get_policy_loss_fn("vanilla")
        loss_via_fn, metrics = fn(
            old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
            response_mask=response_mask, config=cfg,
        )

        self.assertAlmostEqual(loss_vanilla.item(), loss_via_fn.item(), places=6)
        self.assertNotIn("actor/c2gspg_bce_loss", metrics)

    def test_c2gspg_loss_mode_without_c2_binary_reward_still_works(self):
        """c2gspg loss mode with no binary reward must not crash and equal vanilla."""
        from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
        log_prob, old_log_prob, advantages, response_mask = _make_inputs(seed=6)
        cfg_c2 = _make_config(bce_beta=0.5)
        cfg_v = _make_config(bce_beta=0.0)

        loss_c2, _ = compute_policy_loss_c2gspg(
            old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
            response_mask=response_mask, config=cfg_c2,
            c2_binary_reward=None,  # no labels → BCE disabled
        )
        loss_v, _ = compute_policy_loss_vanilla(
            old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
            response_mask=response_mask, config=cfg_v,
        )
        self.assertAlmostEqual(loss_c2.item(), loss_v.item(), places=6)


class TestC2GSPGGradients(unittest.TestCase):
    """Ensure gradients flow through both terms."""

    def test_gradients_flow_through_bce_term(self):
        bsz, seq_len = 2, 4
        log_prob = torch.randn(bsz, seq_len, requires_grad=True)
        old_log_prob = log_prob.detach()
        advantages = torch.zeros(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len, dtype=torch.bool)
        c2_binary_reward = torch.tensor([1.0, 0.0])
        cfg = _make_config(bce_beta=1.0)

        loss, _ = compute_policy_loss_c2gspg(
            old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
            response_mask=response_mask, config=cfg, c2_binary_reward=c2_binary_reward,
        )
        loss.backward()
        self.assertIsNotNone(log_prob.grad)
        self.assertFalse(torch.all(log_prob.grad == 0))

    def test_gradients_zero_when_bce_disabled(self):
        """When advantages=0 and bce_beta=0, ratio=1 means pg gradient ~0."""
        bsz, seq_len = 2, 4
        # ratio = exp(log_prob - old_log_prob) = 1 when equal
        log_prob = torch.full((bsz, seq_len), -1.0, requires_grad=True)
        old_log_prob = torch.full((bsz, seq_len), -1.0)
        advantages = torch.zeros(bsz, seq_len)
        response_mask = torch.ones(bsz, seq_len, dtype=torch.bool)
        cfg = _make_config(bce_beta=0.0)

        loss, _ = compute_policy_loss_c2gspg(
            old_log_prob=old_log_prob, log_prob=log_prob, advantages=advantages,
            response_mask=response_mask, config=cfg,
        )
        loss.backward()
        # With advantages=0 and ratio=1 (on-policy), pg gradient should be ~0
        self.assertTrue(torch.allclose(log_prob.grad, torch.zeros_like(log_prob.grad), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
