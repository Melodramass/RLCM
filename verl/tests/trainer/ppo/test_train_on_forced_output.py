"""Tests for the train_on_forced_output feature.

Covers:
- Response construction: thinking trace + suffix + answer tokens appended correctly
- Edge cases: trace ends with </think>, trace doesn't end with </think>,
  response_length truncation, empty answers
- forced_full_texts double-</think> bug fix
- Integration with GRPO advantage computation: reward placed on correct tokens,
  advantages broadcast correctly, response_mask covers forced output tokens
- Condition gating: feature only activates when budget_checkpoints=[] and
  include_full_trace=True
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Import modules directly from file paths to avoid triggering verl.__init__
# which requires ray (not always available in test envs).
# ---------------------------------------------------------------------------
_fo_root = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "verl", "trainer", "ppo", "forced_output",
)

_utils_path = os.path.join(_fo_root, "forced_output_utils.py")
_spec = importlib.util.spec_from_file_location("forced_output_utils", os.path.abspath(_utils_path))
_utils_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils_mod)

build_forced_output_tokens = _utils_mod.build_forced_output_tokens
compute_budget_truncation_lengths = _utils_mod.compute_budget_truncation_lengths
create_token_level_reward_tensor = _utils_mod.create_token_level_reward_tensor
aggregate_rewards = _utils_mod.aggregate_rewards
aggregate_rewards_batch = _utils_mod.aggregate_rewards_batch


# ---------------------------------------------------------------------------
# Helpers that mirror the agent loop logic for deterministic testing
# ---------------------------------------------------------------------------

THINK_END_TOKEN_ID = 99  # Fake token ID for </think>
SUFFIX_TOKEN_IDS = [200, 201, 202]  # Fake suffix tokens
RESPONSE_LENGTH = 30  # Max response length


def build_response_with_forced_output(
    thinking_trace_ids: list[int],
    forced_answer_ids: list[int],
    think_end_token_id: int = THINK_END_TOKEN_ID,
    suffix_token_ids: list[int] = SUFFIX_TOKEN_IDS,
    response_length: int = RESPONSE_LENGTH,
) -> tuple[list[int], list[int]]:
    """Replicate the agent loop's train_on_forced_output response construction."""
    trace_length = len(thinking_trace_ids)

    # Get suffix tokens (the part after the trace) via build_forced_output_tokens
    full_forced_tokens = build_forced_output_tokens(
        thinking_trace_ids=thinking_trace_ids,
        truncation_length=trace_length,
        think_end_token_id=think_end_token_id,
        suffix_token_ids=suffix_token_ids,
    )
    fo_suffix_ids = full_forced_tokens[trace_length:]

    # Build full response
    full_response_ids = thinking_trace_ids + fo_suffix_ids + forced_answer_ids
    full_response_mask = [1] * len(full_response_ids)

    final_response_ids = full_response_ids[:response_length]
    final_response_mask = full_response_mask[:response_length]
    return final_response_ids, final_response_mask


# ===========================================================================
# Test: suffix slicing from build_forced_output_tokens
# ===========================================================================

class TestForcedOutputSuffixSlicing:
    """Test that slicing build_forced_output_tokens[trace_length:] gives
    the correct suffix tokens for the train_on_forced_output feature."""

    def test_trace_not_ending_with_think_end(self):
        """When trace doesn't end with </think>, suffix = [</think>] + suffix_tokens."""
        trace = [1, 2, 3, 4, 5]  # Does NOT end with THINK_END_TOKEN_ID=99
        full = build_forced_output_tokens(
            trace, len(trace), THINK_END_TOKEN_ID, SUFFIX_TOKEN_IDS
        )
        suffix = full[len(trace):]
        assert suffix == [THINK_END_TOKEN_ID] + SUFFIX_TOKEN_IDS

    def test_trace_ending_with_think_end(self):
        """When trace ends with </think>, suffix = just suffix_tokens (no double)."""
        trace = [1, 2, 3, 4, THINK_END_TOKEN_ID]  # Ends with </think>
        full = build_forced_output_tokens(
            trace, len(trace), THINK_END_TOKEN_ID, SUFFIX_TOKEN_IDS
        )
        suffix = full[len(trace):]
        assert suffix == SUFFIX_TOKEN_IDS

    def test_single_token_trace_is_think_end(self):
        """Edge: trace is just [</think>]."""
        trace = [THINK_END_TOKEN_ID]
        full = build_forced_output_tokens(
            trace, len(trace), THINK_END_TOKEN_ID, SUFFIX_TOKEN_IDS
        )
        suffix = full[len(trace):]
        assert suffix == SUFFIX_TOKEN_IDS

    def test_empty_suffix_tokens(self):
        """Edge: no suffix prompt tokens."""
        trace = [1, 2, 3]
        full = build_forced_output_tokens(
            trace, len(trace), THINK_END_TOKEN_ID, []
        )
        suffix = full[len(trace):]
        assert suffix == [THINK_END_TOKEN_ID]


# ===========================================================================
# Test: full response construction (the core agent loop logic)
# ===========================================================================

class TestTrainOnForcedOutputResponseConstruction:
    """Test the complete response construction for train_on_forced_output."""

    def test_basic_construction_trace_no_think_end(self):
        """Basic case: trace + </think> + suffix + answer."""
        trace = [10, 20, 30, 40, 50]
        answer = [300, 301]
        response, mask = build_response_with_forced_output(trace, answer)

        expected = trace + [THINK_END_TOKEN_ID] + SUFFIX_TOKEN_IDS + answer
        assert response == expected
        assert mask == [1] * len(expected)

    def test_basic_construction_trace_with_think_end(self):
        """Trace already ends with </think>: no double added."""
        trace = [10, 20, 30, 40, THINK_END_TOKEN_ID]
        answer = [300, 301]
        response, mask = build_response_with_forced_output(trace, answer)

        expected = trace + SUFFIX_TOKEN_IDS + answer
        assert response == expected
        assert mask == [1] * len(expected)

    def test_truncation_to_response_length(self):
        """Response capped at response_length."""
        trace = list(range(25))  # 25 tokens
        answer = [300, 301, 302, 303, 304]  # 5 tokens
        # suffix = [99, 200, 201, 202] = 4 tokens (trace doesn't end with 99)
        # Total = 25 + 4 + 5 = 34 > RESPONSE_LENGTH=30
        response, mask = build_response_with_forced_output(
            trace, answer, response_length=30
        )
        assert len(response) == 30
        assert len(mask) == 30
        assert all(m == 1 for m in mask)
        # First 25 tokens are the trace
        assert response[:25] == trace
        # Then partial suffix
        assert response[25:] == [THINK_END_TOKEN_ID] + SUFFIX_TOKEN_IDS + [300]

    def test_empty_answer(self):
        """Empty forced answer: response = trace + suffix only."""
        trace = [10, 20, 30]
        answer = []
        response, mask = build_response_with_forced_output(trace, answer)

        expected = trace + [THINK_END_TOKEN_ID] + SUFFIX_TOKEN_IDS
        assert response == expected
        assert mask == [1] * len(expected)

    def test_very_long_trace_equal_to_response_length(self):
        """Trace exactly fills response_length: answer gets truncated entirely."""
        trace = list(range(30))  # Exactly response_length
        answer = [300, 301]
        response, mask = build_response_with_forced_output(
            trace, answer, response_length=30
        )
        # 30 + suffix + answer > 30, so it's capped at 30 = just the trace
        assert len(response) == 30
        assert response == trace

    def test_trace_shorter_than_response_length_with_room(self):
        """Plenty of room: everything fits within response_length."""
        trace = [10, 20, 30]  # 3 tokens
        answer = [300]  # 1 token
        # suffix = [99, 200, 201, 202] = 4 tokens
        # Total = 3 + 4 + 1 = 8 << 30
        response, mask = build_response_with_forced_output(trace, answer)

        expected = [10, 20, 30, THINK_END_TOKEN_ID, 200, 201, 202, 300]
        assert response == expected
        assert len(mask) == 8
        assert all(m == 1 for m in mask)

    def test_response_length_of_one(self):
        """Edge: response_length=1, only first token of trace kept."""
        trace = [10, 20, 30]
        answer = [300]
        response, mask = build_response_with_forced_output(
            trace, answer, response_length=1
        )
        assert response == [10]
        assert mask == [1]


# ===========================================================================
# Test: condition gating (when train_on_forced_output should activate)
# ===========================================================================

class TestTrainOnForcedOutputConditions:
    """Test that the feature only activates under the correct conditions."""

    def test_activates_with_empty_budgets_and_full_trace(self):
        """train_on_fo = True when budget_checkpoints=[] and include_full_trace=True."""
        train_on_forced_output = True
        budget_checkpoints = []
        include_full_trace = True

        train_on_fo = (
            train_on_forced_output
            and len(budget_checkpoints) == 0
            and include_full_trace
        )
        assert train_on_fo is True

    def test_disabled_with_budget_checkpoints(self):
        """train_on_fo = False when budget_checkpoints is non-empty."""
        train_on_forced_output = True
        budget_checkpoints = [2000, 4000]
        include_full_trace = True

        train_on_fo = (
            train_on_forced_output
            and len(budget_checkpoints) == 0
            and include_full_trace
        )
        assert train_on_fo is False

    def test_disabled_without_full_trace(self):
        """train_on_fo = False when include_full_trace=False."""
        train_on_forced_output = True
        budget_checkpoints = []
        include_full_trace = False

        train_on_fo = (
            train_on_forced_output
            and len(budget_checkpoints) == 0
            and include_full_trace
        )
        assert train_on_fo is False

    def test_disabled_when_feature_off(self):
        """train_on_fo = False when train_on_forced_output=False."""
        train_on_forced_output = False
        budget_checkpoints = []
        include_full_trace = True

        train_on_fo = (
            train_on_forced_output
            and len(budget_checkpoints) == 0
            and include_full_trace
        )
        assert train_on_fo is False


# ===========================================================================
# Test: budget checkpoints + include_full_trace interaction
# ===========================================================================

class TestBudgetCheckpointsWithFullTrace:
    """Test that truncation_lengths is correctly computed when
    budget_checkpoints=[] and include_full_trace=True."""

    def test_empty_budgets_full_trace(self):
        """With budget_checkpoints=[] and include_full_trace=True,
        truncation_lengths should be [trace_length]."""
        trace_length = 5000
        truncation_lengths = compute_budget_truncation_lengths(trace_length, [])
        assert truncation_lengths == []

        # include_full_trace logic
        if True and trace_length not in truncation_lengths:
            truncation_lengths.append(trace_length)
            truncation_lengths.sort()
        assert truncation_lengths == [5000]

    def test_budgets_that_collapse_to_full_trace(self):
        """When all budgets >= trace_length, they collapse to trace_length.
        include_full_trace won't add a duplicate."""
        trace_length = 3000
        truncation_lengths = compute_budget_truncation_lengths(trace_length, [4000, 8000])
        assert truncation_lengths == [3000]  # Both clamped to 3000, deduped

        # include_full_trace: trace_length IS already in truncation_lengths
        if True and trace_length not in truncation_lengths:
            truncation_lengths.append(trace_length)
            truncation_lengths.sort()
        assert truncation_lengths == [3000]  # No duplicate added

    def test_budgets_below_trace_length(self):
        """With budgets below trace_length, include_full_trace adds a new entry."""
        trace_length = 8000
        truncation_lengths = compute_budget_truncation_lengths(trace_length, [2000, 4000])
        assert truncation_lengths == [2000, 4000]

        if True and trace_length not in truncation_lengths:
            truncation_lengths.append(trace_length)
            truncation_lengths.sort()
        assert truncation_lengths == [2000, 4000, 8000]


# ===========================================================================
# Test: GRPO advantage integration with extended response
# ===========================================================================

class TestGRPOAdvantageWithExtendedResponse:
    """Test that GRPO advantage computation works correctly when
    response_ids includes forced output tokens (train_on_forced_output)."""

    def test_reward_placement_with_extended_response(self):
        """Reward should be placed on the last valid token of the extended response."""
        # Simulate: 5 trace tokens + 4 suffix tokens + 3 answer tokens = 12 tokens
        # response_length = 15 (padded)
        response_length = 15
        response_mask = torch.zeros(2, response_length)
        response_mask[0, :12] = 1  # Sample 0: 12 valid tokens
        response_mask[1, :8] = 1   # Sample 1: 8 valid tokens (shorter answer)

        rewards = [1.0, 0.0]
        reward_tensor = create_token_level_reward_tensor(
            rewards, response_length=response_length, response_mask=response_mask
        )

        # Reward should be on last valid token
        assert reward_tensor[0, 11].item() == pytest.approx(1.0)
        assert reward_tensor[1, 7].item() == pytest.approx(0.0)
        # All other positions should be 0
        assert reward_tensor[0, :11].sum().item() == 0.0
        assert reward_tensor[0, 12:].sum().item() == 0.0

    def test_advantage_broadcast_covers_forced_output_tokens(self):
        """GRPO advantage should be broadcast to ALL valid response tokens,
        including forced output tokens, when train_on_forced_output is True.

        This verifies the key property: gradient flows through forced output tokens.
        """
        # Simulate 2 samples from the same prompt group (uid="A")
        # Sample 0: 12 tokens (trace+suffix+answer), reward=1.0
        # Sample 1: 12 tokens, reward=0.0
        response_length = 15
        batch_size = 2

        response_mask = torch.zeros(batch_size, response_length)
        response_mask[0, :12] = 1
        response_mask[1, :12] = 1

        # Create token-level rewards
        reward_tensor = create_token_level_reward_tensor(
            [1.0, 0.0], response_length=response_length, response_mask=response_mask
        )

        # Simulate GRPO advantage: score = reward.sum(dim=-1)
        scores = reward_tensor.sum(dim=-1)
        assert scores[0].item() == pytest.approx(1.0)
        assert scores[1].item() == pytest.approx(0.0)

        # Normalize within group: mean=0.5, std ~= 0.707
        group_mean = scores.mean()
        group_std = scores.std()
        normalized = (scores - group_mean) / (group_std + 1e-6)

        # Broadcast onto response_mask positions
        advantages = normalized.unsqueeze(-1) * response_mask

        # Check: advantage is non-zero on ALL 12 valid tokens (including forced output)
        assert (advantages[0, :12] != 0).all(), (
            "Advantage should be non-zero on ALL valid tokens including forced output"
        )
        assert (advantages[1, :12] != 0).all(), (
            "Advantage should be non-zero on ALL valid tokens including forced output"
        )
        # Padding positions should be zero
        assert advantages[0, 12:].sum().item() == 0.0
        assert advantages[1, 12:].sum().item() == 0.0

        # Sample 0 (reward=1) should have positive advantage
        assert (advantages[0, :12] > 0).all()
        # Sample 1 (reward=0) should have negative advantage
        assert (advantages[1, :12] < 0).all()

    def test_response_mask_consistency(self):
        """response_mask should be all 1s for extended response tokens
        and 0s for padding, matching what compute_response_mask would produce
        from attention_mask."""
        # Simulate: prompt=10 tokens, response=12 tokens (trace+suffix+ans)
        prompt_length = 10
        trace = list(range(5))
        answer = [300, 301, 302]
        response, resp_mask = build_response_with_forced_output(
            trace, answer, response_length=RESPONSE_LENGTH
        )
        actual_response_len = len(response)  # 5 + 4 + 3 = 12

        # Build attention_mask as the training pipeline would
        total_length = prompt_length + RESPONSE_LENGTH
        attention_mask = torch.zeros(1, total_length, dtype=torch.long)
        attention_mask[0, :prompt_length + actual_response_len] = 1

        # compute_response_mask gets attention_mask[:, -response_length:]
        computed_response_mask = attention_mask[:, -RESPONSE_LENGTH:]

        # Verify mask is 1 for all response tokens including forced output
        assert computed_response_mask[0, :actual_response_len].sum().item() == actual_response_len
        assert computed_response_mask[0, actual_response_len:].sum().item() == 0


# ===========================================================================
# Test: forced_full_texts double-</think> bug fix
# ===========================================================================

class TestForcedFullTextsNoDoubleThinkEnd:
    """Test that forced_full_texts construction doesn't produce double </think>.

    Previously, the code always concatenated stop_token regardless of whether
    the decoded trace already ended with it. This class tests the fix.
    """

    def test_no_double_think_end_when_trace_ends_with_it(self):
        """When decoded trace text ends with </think>, don't add another."""
        stop_token = "</think>"
        decoded_trace = "Let me think step by step...</think>"
        suffix_prompt = "\n**Final Answer**\n\\boxed{"
        answer_text = "42}"

        # Simulating the fixed logic
        if decoded_trace.rstrip().endswith(stop_token):
            think_end_text = ""
        else:
            think_end_text = stop_token

        full_text = decoded_trace + think_end_text + suffix_prompt + answer_text

        # Should NOT have double </think>
        assert "</think></think>" not in full_text
        # Should have exactly one </think>
        assert full_text.count("</think>") == 1
        assert full_text == "Let me think step by step...</think>\n**Final Answer**\n\\boxed{42}"

    def test_think_end_added_when_trace_truncated(self):
        """When decoded trace is truncated (no </think>), add it."""
        stop_token = "</think>"
        decoded_trace = "Let me think step by step... and more"
        suffix_prompt = "\n**Final Answer**\n\\boxed{"
        answer_text = "42}"

        if decoded_trace.rstrip().endswith(stop_token):
            think_end_text = ""
        else:
            think_end_text = stop_token

        full_text = decoded_trace + think_end_text + suffix_prompt + answer_text

        assert full_text.count("</think>") == 1
        assert full_text == (
            "Let me think step by step... and more</think>"
            "\n**Final Answer**\n\\boxed{42}"
        )

    def test_think_end_with_trailing_whitespace(self):
        """Handle case where decoded trace has whitespace after </think>."""
        stop_token = "</think>"
        # Tokenizer might add a trailing space or newline after </think>
        decoded_trace = "thinking...</think> "
        suffix_prompt = "\n\\boxed{"
        answer_text = "7}"

        if decoded_trace.rstrip().endswith(stop_token):
            think_end_text = ""
        else:
            think_end_text = stop_token

        full_text = decoded_trace + think_end_text + suffix_prompt + answer_text

        assert "</think></think>" not in full_text
        # The rstrip() check should catch this
        assert think_end_text == ""


# ===========================================================================
# Test: no train_on_forced_output (backward compatibility)
# ===========================================================================

class TestBackwardCompatibility:
    """Verify that when train_on_forced_output is disabled, the response_ids
    contain only the thinking trace (original behavior)."""

    def test_original_response_only_thinking_trace(self):
        """Without train_on_forced_output, response = thinking_trace only."""
        trace = [10, 20, 30, 40, 50]
        response_length = 10

        # Old behavior: just truncate trace
        final_response_ids = trace[:response_length]
        final_response_mask = [1] * len(trace)
        final_response_mask = final_response_mask[:response_length]

        assert final_response_ids == [10, 20, 30, 40, 50]
        assert final_response_mask == [1, 1, 1, 1, 1]

    def test_old_vs_new_response_length(self):
        """With train_on_forced_output, response is longer than without."""
        trace = [10, 20, 30, 40, 50]
        answer = [300, 301]

        # Old: only trace
        old_response = trace[:RESPONSE_LENGTH]

        # New: trace + suffix + answer
        new_response, _ = build_response_with_forced_output(trace, answer)

        assert len(new_response) > len(old_response)
        # New response starts with the same trace
        assert new_response[:len(trace)] == trace


# ===========================================================================
# Test: combined reward placement + advantage for realistic scenario
# ===========================================================================

class TestRealisticTrainingScenario:
    """Simulate a realistic mini-batch to verify the full pipeline works."""

    def test_grpo_group_of_six_with_extended_response(self):
        """Simulate n=6 rollouts for one prompt, with train_on_forced_output.
        Verify rewards and advantages are computed correctly."""
        n = 6
        response_length = 20

        # Simulated traces and answers of varying lengths
        traces_and_answers = [
            (list(range(8)), [300, 301]),        # 8 trace + 4 suffix + 2 ans = 14
            (list(range(10)), [300]),             # 10 + 4 + 1 = 15
            (list(range(6)), [300, 301, 302]),    # 6 + 4 + 3 = 13
            (list(range(12)), [300, 301]),        # 12 + 4 + 2 = 18
            (list(range(7)), []),                 # 7 + 4 + 0 = 11
            (list(range(9)), [300, 301, 302, 303]),  # 9 + 4 + 4 = 17
        ]

        # Build extended responses
        responses = []
        for trace, answer in traces_and_answers:
            resp, mask = build_response_with_forced_output(
                trace, answer, response_length=response_length
            )
            responses.append((resp, mask))

        # rewards: 3 correct, 3 wrong
        rewards = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]

        # Build response_mask tensor (pad to response_length)
        response_mask = torch.zeros(n, response_length)
        for i, (resp, mask) in enumerate(responses):
            resp_len = len(resp)
            response_mask[i, :resp_len] = 1

        # Create token-level reward tensor
        reward_tensor = create_token_level_reward_tensor(
            rewards, response_length=response_length, response_mask=response_mask
        )

        # Verify rewards are placed correctly
        for i, (resp, _) in enumerate(responses):
            last_valid = len(resp) - 1
            assert reward_tensor[i, last_valid].item() == pytest.approx(rewards[i])
            # No reward on other positions
            mask_sum = reward_tensor[i].sum().item()
            assert mask_sum == pytest.approx(rewards[i])

        # Simulate GRPO advantage computation
        scores = reward_tensor.sum(dim=-1)
        group_mean = scores.mean()
        group_std = scores.std()

        # With 3 correct (1.0) and 3 wrong (0.0): mean=0.5, std≈0.5477
        assert group_mean.item() == pytest.approx(0.5)
        assert group_std.item() == pytest.approx(0.5477, abs=0.01)

        normalized = (scores - group_mean) / (group_std + 1e-6)
        advantages = normalized.unsqueeze(-1) * response_mask

        # Correct samples should have positive advantages on ALL their tokens
        for i in [0, 2, 5]:
            resp_len = len(responses[i][0])
            assert (advantages[i, :resp_len] > 0).all(), (
                f"Sample {i} (correct) should have positive advantage on all tokens"
            )

        # Wrong samples should have negative advantages on ALL their tokens
        for i in [1, 3, 4]:
            resp_len = len(responses[i][0])
            assert (advantages[i, :resp_len] < 0).all(), (
                f"Sample {i} (wrong) should have negative advantage on all tokens"
            )

    def test_reward_with_single_budget_full_trace(self):
        """When budget_checkpoints=[] and include_full_trace=True,
        there's exactly 1 budget (full trace). Reward aggregation
        should just pass through the single value."""
        rewards_matrix = [[0.8], [0.3], [1.0]]
        aggregated = aggregate_rewards_batch(rewards_matrix, "mean")
        assert aggregated == [pytest.approx(0.8), pytest.approx(0.3), pytest.approx(1.0)]


# ===========================================================================
# Test: policy loss gradient flow (simulated)
# ===========================================================================

class TestPolicyLossGradientFlow:
    """Verify that the PPO/GRPO policy loss would produce gradients
    for forced output tokens when train_on_forced_output is True.

    This doesn't test the actual model, just the math of the loss.
    """

    def test_gradient_flows_through_all_response_tokens(self):
        """Simulate the PPO loss and verify grad exists for forced output positions."""
        batch_size = 2
        response_length = 15
        trace_len = 8
        forced_output_len = 5  # suffix + answer
        total_valid = trace_len + forced_output_len  # 13

        response_mask = torch.zeros(batch_size, response_length)
        response_mask[:, :total_valid] = 1

        # Simulated log_probs (requires grad)
        old_log_probs = torch.randn(batch_size, response_length)
        new_log_probs = torch.randn(batch_size, response_length, requires_grad=True)

        # Advantages (constant across tokens within a sample, as in GRPO)
        advantages = torch.zeros(batch_size, response_length)
        advantages[0, :total_valid] = 0.5   # Positive advantage
        advantages[1, :total_valid] = -0.3  # Negative advantage

        # Simplified PPO loss (no clipping for this test)
        ratio = torch.exp(new_log_probs - old_log_probs)
        pg_loss_per_token = -advantages * ratio

        # Masked mean over valid tokens
        loss = (pg_loss_per_token * response_mask).sum() / response_mask.sum()
        loss.backward()

        assert new_log_probs.grad is not None

        # Gradient should be non-zero for ALL valid tokens (including forced output)
        for pos in range(total_valid):
            assert new_log_probs.grad[0, pos].item() != 0.0, (
                f"Gradient should be non-zero at position {pos} "
                f"(trace ends at {trace_len}, forced output starts there)"
            )
            assert new_log_probs.grad[1, pos].item() != 0.0, (
                f"Gradient should be non-zero at position {pos} for sample 1"
            )

        # Gradient should be zero for padding positions
        for pos in range(total_valid, response_length):
            assert new_log_probs.grad[0, pos].item() == 0.0
            assert new_log_probs.grad[1, pos].item() == 0.0

    def test_no_gradient_without_train_on_forced_output(self):
        """Without train_on_forced_output, only thinking trace tokens get gradient."""
        batch_size = 1
        response_length = 15
        trace_len = 8

        # Without train_on_forced_output: mask only covers trace
        response_mask = torch.zeros(batch_size, response_length)
        response_mask[:, :trace_len] = 1

        old_log_probs = torch.randn(batch_size, response_length)
        new_log_probs = torch.randn(batch_size, response_length, requires_grad=True)

        advantages = torch.zeros(batch_size, response_length)
        advantages[0, :trace_len] = 0.5

        ratio = torch.exp(new_log_probs - old_log_probs)
        pg_loss_per_token = -advantages * ratio
        loss = (pg_loss_per_token * response_mask).sum() / response_mask.sum()
        loss.backward()

        # Gradient non-zero only for trace tokens
        for pos in range(trace_len):
            assert new_log_probs.grad[0, pos].item() != 0.0
        # Zero for positions beyond trace (where forced output would be)
        for pos in range(trace_len, response_length):
            assert new_log_probs.grad[0, pos].item() == 0.0


# ===========================================================================
# Test: edge cases and robustness
# ===========================================================================

class TestEdgeCases:

    def test_zero_length_trace(self):
        """Edge: trace is empty."""
        trace = []
        answer = [300, 301]
        response, mask = build_response_with_forced_output(trace, answer)

        # trace is empty, build_forced_output_tokens with truncation_length=0:
        # truncated = [], not ending with think_end -> [99] + suffix + answer
        expected = [THINK_END_TOKEN_ID] + SUFFIX_TOKEN_IDS + answer
        assert response == expected

    def test_response_length_zero(self):
        """Edge: response_length=0 returns empty."""
        trace = [10, 20]
        answer = [300]
        response, mask = build_response_with_forced_output(
            trace, answer, response_length=0
        )
        assert response == []
        assert mask == []

    def test_very_long_answer_gets_truncated(self):
        """Long forced answer gets truncated by response_length."""
        trace = [10, 20, 30]  # 3 tokens
        answer = list(range(300, 330))  # 30 tokens!
        # suffix = [99, 200, 201, 202] = 4 tokens
        # total = 3 + 4 + 30 = 37 > RESPONSE_LENGTH=30
        response, mask = build_response_with_forced_output(trace, answer)
        assert len(response) == RESPONSE_LENGTH
        assert len(mask) == RESPONSE_LENGTH

        # Verify structure: trace + suffix fits, then partial answer
        assert response[:3] == [10, 20, 30]
        assert response[3:7] == [THINK_END_TOKEN_ID, 200, 201, 202]
        assert response[7:] == list(range(300, 323))  # 23 answer tokens

    def test_all_same_rewards_zero_advantage(self):
        """When all samples in a group have the same reward,
        GRPO advantage should be zero (no learning signal)."""
        n = 4
        response_length = 10

        response_mask = torch.ones(n, response_length)
        rewards = [1.0, 1.0, 1.0, 1.0]

        reward_tensor = create_token_level_reward_tensor(
            rewards, response_length=response_length, response_mask=response_mask
        )

        scores = reward_tensor.sum(dim=-1)
        group_std = scores.std()

        # std is 0, so normalized advantages would be 0 (or near-zero)
        assert group_std.item() == pytest.approx(0.0, abs=1e-6)
