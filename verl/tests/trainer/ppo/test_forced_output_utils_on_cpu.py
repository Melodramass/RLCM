"""Tests for forced output utility functions.

Covers:
- Original single-answer functions (backward compat)
- New multi-answer aggregation function
- Integration of multi-answer with budget aggregation pipeline
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np
import pytest
import torch

# Import the utils module directly from file path to avoid triggering
# verl.__init__ which requires ray (not always available in test envs).
_utils_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "verl", "trainer", "ppo", "forced_output", "forced_output_utils.py",
)
_spec = importlib.util.spec_from_file_location("forced_output_utils", os.path.abspath(_utils_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

aggregate_multi_answer_rewards = _mod.aggregate_multi_answer_rewards
aggregate_rewards = _mod.aggregate_rewards
aggregate_rewards_batch = _mod.aggregate_rewards_batch
compute_unanimous_group_keep_mask = _mod.compute_unanimous_group_keep_mask
compute_budget_truncation_lengths = _mod.compute_budget_truncation_lengths
create_token_level_reward_tensor = _mod.create_token_level_reward_tensor
resolve_budget_checkpoint = _mod.resolve_budget_checkpoint


# =========================================================================
# Tests for compute_budget_truncation_lengths (original, unchanged)
# =========================================================================

class TestComputeBudgetTruncationLengths:
    def test_basic(self):
        assert compute_budget_truncation_lengths(6000, [4000, 8000]) == [4000, 6000]

    def test_all_clamped(self):
        assert compute_budget_truncation_lengths(3000, [2000, 4000, 8000]) == [2000, 3000]

    def test_no_clamping_needed(self):
        assert compute_budget_truncation_lengths(10000, [2000, 4000, 8000]) == [2000, 4000, 8000]

    def test_dedup(self):
        # Both 4000 and 8000 clamp to 3000
        assert compute_budget_truncation_lengths(3000, [4000, 8000]) == [3000]

    def test_empty_budgets(self):
        assert compute_budget_truncation_lengths(5000, []) == []

    def test_zero_trace_length(self):
        # All budgets clamp to 0, which is filtered (> 0 check)
        assert compute_budget_truncation_lengths(0, [1000, 2000]) == []

    def test_single_budget(self):
        assert compute_budget_truncation_lengths(5000, [3000]) == [3000]

    def test_unsorted_budgets(self):
        # Budgets don't need to be sorted in input; output is always sorted
        assert compute_budget_truncation_lengths(10000, [8000, 2000, 4000]) == [2000, 4000, 8000]

    def test_relative_budgets(self):
        assert compute_budget_truncation_lengths(6000, [0.25, 0.5, 1.0]) == [1500, 3000, 6000]

    def test_relative_budgets_are_clamped_and_deduped(self):
        assert compute_budget_truncation_lengths(3, [0.2, 0.34, 0.49, 1.0]) == [1, 3]

    def test_mixed_absolute_and_relative_budgets(self):
        assert compute_budget_truncation_lengths(6000, [0.5, 2000, 1.0]) == [2000, 3000, 6000]


class TestResolveBudgetCheckpoint:
    def test_absolute_budget(self):
        assert resolve_budget_checkpoint(6000, 2000) == 2000

    def test_relative_budget(self):
        assert resolve_budget_checkpoint(6000, 0.5) == 3000

    def test_relative_full_trace_budget(self):
        assert resolve_budget_checkpoint(6000, 1.0) == 6000

    def test_absolute_budget_is_clamped(self):
        assert resolve_budget_checkpoint(6000, 8000) == 6000


# =========================================================================
# Tests for aggregate_rewards (original, unchanged)
# =========================================================================

class TestAggregateRewards:
    def test_mean_basic(self):
        assert aggregate_rewards([1.0, 0.0, 1.0], "mean") == pytest.approx(2 / 3)

    def test_mean_all_same(self):
        assert aggregate_rewards([0.5, 0.5, 0.5], "mean") == pytest.approx(0.5)

    def test_linear_weighted(self):
        # weights = [1, 2, 3], normalized = [1/6, 2/6, 3/6]
        # result = 1/6 * 1.0 + 2/6 * 0.0 + 3/6 * 1.0 = 4/6
        result = aggregate_rewards([1.0, 0.0, 1.0], "linear_weighted")
        assert result == pytest.approx(4 / 6)

    def test_single_value(self):
        assert aggregate_rewards([0.7], "mean") == pytest.approx(0.7)
        assert aggregate_rewards([0.7], "linear_weighted") == pytest.approx(0.7)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            aggregate_rewards([], "mean")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown reward aggregation mode"):
            aggregate_rewards([1.0, 0.5], "unknown")


# =========================================================================
# Tests for aggregate_rewards_batch (original, unchanged)
# =========================================================================

class TestAggregateRewardsBatch:
    def test_basic_batch(self):
        matrix = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        result = aggregate_rewards_batch(matrix, "mean")
        assert result == [pytest.approx(0.5), pytest.approx(0.5), pytest.approx(1.0)]

    def test_linear_weighted_batch(self):
        # weights [1, 2] -> normalized [1/3, 2/3]
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        result = aggregate_rewards_batch(matrix, "linear_weighted")
        assert result == [pytest.approx(1 / 3), pytest.approx(2 / 3)]

    def test_empty_batch(self):
        assert aggregate_rewards_batch([], "mean") == []


# =========================================================================
# Tests for create_token_level_reward_tensor (original, unchanged)
# =========================================================================

class TestCreateTokenLevelRewardTensor:
    def test_no_mask(self):
        tensor = create_token_level_reward_tensor([1.0, 0.5], response_length=5)
        assert tensor.shape == (2, 5)
        # Reward placed at last position
        assert tensor[0, 4].item() == pytest.approx(1.0)
        assert tensor[1, 4].item() == pytest.approx(0.5)
        # All other positions should be 0
        assert tensor[0, :4].sum().item() == 0.0

    def test_with_mask(self):
        mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ], dtype=torch.float32)
        tensor = create_token_level_reward_tensor([1.0, 0.5], response_length=5, response_mask=mask)
        # Last valid position for sample 0 is index 2, sample 1 is index 1
        assert tensor[0, 2].item() == pytest.approx(1.0)
        assert tensor[1, 1].item() == pytest.approx(0.5)
        # All other positions should be 0
        assert tensor[0, :2].sum().item() == 0.0
        assert tensor[0, 3:].sum().item() == 0.0

    def test_all_zeros_mask(self):
        mask = torch.zeros(1, 5)
        tensor = create_token_level_reward_tensor([1.0], response_length=5, response_mask=mask)
        # No valid tokens -> all zeros
        assert tensor.sum().item() == 0.0

    def test_single_sample(self):
        tensor = create_token_level_reward_tensor([0.8], response_length=3)
        assert tensor.shape == (1, 3)
        assert tensor[0, 2].item() == pytest.approx(0.8)


# =========================================================================
# Tests for aggregate_multi_answer_rewards (NEW)
# =========================================================================

class TestAggregateMultiAnswerRewards:
    """Tests for the new multi-answer aggregation function."""

    # --- mean method ---

    def test_mean_basic(self):
        result = aggregate_multi_answer_rewards([1.0, 0.0, 1.0, 0.0], "mean")
        assert result == pytest.approx(0.5)

    def test_mean_all_correct(self):
        result = aggregate_multi_answer_rewards([1.0, 1.0, 1.0, 1.0], "mean")
        assert result == pytest.approx(1.0)

    def test_mean_all_wrong(self):
        result = aggregate_multi_answer_rewards([0.0, 0.0, 0.0, 0.0], "mean")
        assert result == pytest.approx(0.0)

    def test_mean_partial_scores(self):
        # Non-binary rewards
        result = aggregate_multi_answer_rewards([0.2, 0.4, 0.6, 0.8], "mean")
        assert result == pytest.approx(0.5)

    # --- max method ---

    def test_max_basic(self):
        result = aggregate_multi_answer_rewards([0.0, 0.5, 1.0, 0.2], "max")
        assert result == pytest.approx(1.0)

    def test_max_all_zero(self):
        result = aggregate_multi_answer_rewards([0.0, 0.0, 0.0], "max")
        assert result == pytest.approx(0.0)

    def test_max_negative(self):
        result = aggregate_multi_answer_rewards([-0.5, -0.2, -0.8], "max")
        assert result == pytest.approx(-0.2)

    # --- edge cases ---

    def test_single_answer(self):
        """Single answer should return that answer for all methods."""
        assert aggregate_multi_answer_rewards([0.7], "mean") == pytest.approx(0.7)
        assert aggregate_multi_answer_rewards([0.7], "max") == pytest.approx(0.7)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            aggregate_multi_answer_rewards([], "mean")

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown multi-answer aggregation method"):
            aggregate_multi_answer_rewards([1.0, 0.5], "unknown_method")


# =========================================================================
# Integration tests: multi-answer + budget aggregation pipeline
# =========================================================================

class TestMultiAnswerBudgetPipeline:
    """Test the full pipeline: multi-answer per budget -> budget aggregation."""

    def test_multi_answer_mean_then_budget_mean(self):
        """4 answers per budget, 3 budgets. Mean everything."""
        # Budget 1: [1, 0, 1, 0] -> mean = 0.5
        # Budget 2: [1, 1, 0, 0] -> mean = 0.5
        # Budget 3: [1, 1, 1, 0] -> mean = 0.75
        per_budget_multi = [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
        # Step 1: aggregate multi-answers at each budget
        per_budget_rewards = [
            aggregate_multi_answer_rewards(answers, "mean")
            for answers in per_budget_multi
        ]
        assert per_budget_rewards == [
            pytest.approx(0.5),
            pytest.approx(0.5),
            pytest.approx(0.75),
        ]
        # Step 2: aggregate across budgets
        final = aggregate_rewards(per_budget_rewards, "mean")
        assert final == pytest.approx((0.5 + 0.5 + 0.75) / 3)

    def test_multi_answer_mean_then_budget_linear(self):
        """Mean aggregation per budget, linear_weighted across budgets."""
        # Budget 1: [1, 0, 0, 0] -> mean = 0.25
        # Budget 2: [1, 1, 0, 0] -> mean = 0.5
        # Budget 3: [1, 1, 1, 1] -> mean = 1.0
        per_budget_multi = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
        per_budget_rewards = [
            aggregate_multi_answer_rewards(answers, "mean")
            for answers in per_budget_multi
        ]
        assert per_budget_rewards == [
            pytest.approx(0.25),
            pytest.approx(0.5),
            pytest.approx(1.0),
        ]
        # linear_weighted: weights [1, 2, 3], normalized [1/6, 2/6, 3/6]
        final = aggregate_rewards(per_budget_rewards, "linear_weighted")
        expected = 1/6 * 0.25 + 2/6 * 0.5 + 3/6 * 1.0
        assert final == pytest.approx(expected)

    def test_single_answer_backward_compat(self):
        """With 1 answer per budget, pipeline should behave like original."""
        per_budget_multi = [[1.0], [0.0], [1.0]]
        per_budget_rewards = [
            aggregate_multi_answer_rewards(answers, "mean")
            for answers in per_budget_multi
        ]
        assert per_budget_rewards == [1.0, 0.0, 1.0]
        # Same as aggregate_rewards directly
        final_via_multi = aggregate_rewards(per_budget_rewards, "mean")
        final_direct = aggregate_rewards([1.0, 0.0, 1.0], "mean")
        assert final_via_multi == pytest.approx(final_direct)

    def test_batch_pipeline(self):
        """Full batch of 2 traces, 2 budgets, 4 answers each."""
        # Trace 0: budget1 [1,1,0,0]->0.5, budget2 [1,1,1,0]->0.75
        # Trace 1: budget1 [0,0,0,0]->0.0, budget2 [1,0,0,0]->0.25
        trace0_multi = [[1, 1, 0, 0], [1, 1, 1, 0]]
        trace1_multi = [[0, 0, 0, 0], [1, 0, 0, 0]]

        rewards_matrix = []
        for trace_budgets in [trace0_multi, trace1_multi]:
            per_budget = [
                aggregate_multi_answer_rewards(ans, "mean")
                for ans in trace_budgets
            ]
            rewards_matrix.append(per_budget)

        assert rewards_matrix == [
            [pytest.approx(0.5), pytest.approx(0.75)],
            [pytest.approx(0.0), pytest.approx(0.25)],
        ]

        aggregated = aggregate_rewards_batch(rewards_matrix, "mean")
        assert aggregated == [pytest.approx(0.625), pytest.approx(0.125)]

    def test_multi_answer_max_then_budget_mean(self):
        """Max aggregation per budget (optimistic), mean across budgets."""
        per_budget_multi = [
            [0.0, 0.0, 0.0, 1.0],  # max = 1.0
            [0.0, 0.0, 0.0, 0.0],  # max = 0.0
        ]
        per_budget_rewards = [
            aggregate_multi_answer_rewards(answers, "max")
            for answers in per_budget_multi
        ]
        assert per_budget_rewards == [pytest.approx(1.0), pytest.approx(0.0)]
        final = aggregate_rewards(per_budget_rewards, "mean")
        assert final == pytest.approx(0.5)


# =========================================================================
# Tests for token-level reward tensor with multi-answer rewards
# =========================================================================

class TestTokenLevelRewardWithMultiAnswer:
    """Verify create_token_level_reward_tensor works correctly with
    aggregated multi-answer rewards."""

    def test_multi_answer_aggregated_to_token_level(self):
        """Simulate full pipeline: multi-answer -> budget agg -> token tensor."""
        # 2 samples, 3 budgets, 4 answers each
        sample_0_budgets = [[1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0]]
        sample_1_budgets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]

        rewards_matrix = []
        for sample_budgets in [sample_0_budgets, sample_1_budgets]:
            per_budget = [
                aggregate_multi_answer_rewards(
                    [float(x) for x in ans], "mean"
                )
                for ans in sample_budgets
            ]
            rewards_matrix.append(per_budget)

        aggregated = aggregate_rewards_batch(rewards_matrix, "mean")

        # Sample 0: mean = [0.75, 1.0, 0.75] -> mean = 0.833...
        # Sample 1: mean = [0.0, 0.25, 0.25] -> mean = 0.1666...
        assert aggregated[0] == pytest.approx(5 / 6)
        assert aggregated[1] == pytest.approx(1 / 6)

        mask = torch.tensor([
            [1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0],
        ], dtype=torch.float32)

        tensor = create_token_level_reward_tensor(
            aggregated, response_length=5, response_mask=mask
        )

        # Sample 0: last valid = idx 3
        assert tensor[0, 3].item() == pytest.approx(5 / 6)
        # Sample 1: last valid = idx 1
        assert tensor[1, 1].item() == pytest.approx(1 / 6)


# =========================================================================
# Tests for unanimous natural-trace group filtering helper
# =========================================================================

class TestComputeUnanimousGroupKeepMask:
    def test_mixed_and_unanimous_groups(self):
        uids = np.array(["u1", "u1", "u2", "u2", "u3", "u3"], dtype=object)
        natural_correct = np.array([True, False, True, True, False, False], dtype=bool)

        keep_mask, stats = compute_unanimous_group_keep_mask(uids, natural_correct)

        np.testing.assert_array_equal(keep_mask, np.array([True, True, False, False, False, False]))
        assert stats["groups_total"] == 3
        assert stats["groups_kept_mixed"] == 1
        assert stats["groups_dropped_all_correct"] == 1
        assert stats["groups_dropped_all_incorrect"] == 1
        assert stats["samples_kept"] == 2
        assert stats["samples_dropped"] == 4

    def test_all_groups_unanimous(self):
        uids = np.array(["u1", "u1", "u2", "u2"], dtype=object)
        natural_correct = np.array([True, True, False, False], dtype=bool)

        keep_mask, stats = compute_unanimous_group_keep_mask(uids, natural_correct)

        assert not keep_mask.any()
        assert stats["groups_kept_mixed"] == 0
        assert stats["groups_dropped_all_correct"] == 1
        assert stats["groups_dropped_all_incorrect"] == 1
        assert stats["samples_kept"] == 0
        assert stats["samples_dropped"] == 4

    def test_singleton_group_is_unanimous(self):
        uids = np.array(["u1", "u2", "u2"], dtype=object)
        natural_correct = np.array([True, True, False], dtype=bool)

        keep_mask, stats = compute_unanimous_group_keep_mask(uids, natural_correct)

        np.testing.assert_array_equal(keep_mask, np.array([False, True, True]))
        assert stats["groups_total"] == 2
        assert stats["groups_kept_mixed"] == 1
        assert stats["groups_dropped_all_correct"] == 1
        assert stats["groups_dropped_all_incorrect"] == 0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_unanimous_group_keep_mask(
                np.array(["u1", "u1"], dtype=object),
                np.array([True], dtype=bool),
            )
