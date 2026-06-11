# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities for the Forced Output GRPO trainer.

Provides:
- Budget checkpoint computation (truncation + dedup)
- Forced output suffix construction
- Reward aggregation across budgets (mean / linear_weighted)
- Group filtering helpers for unanimous natural-trace correctness
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def compute_budget_truncation_lengths(
    trace_length: int,
    budget_checkpoints: list[int | float],
) -> list[int]:
    """Compute the actual truncation lengths for a trace given budget checkpoints.

    Each budget is first resolved into an absolute token count:

    - integer budgets are treated as absolute token budgets
    - fractional budgets in ``(0, 1]`` are treated as relative budgets and
      converted to ``round(trace_length * fraction)``

    Then each resolved budget is clamped to ``min(trace_length, budget)``.
    Duplicates are removed while preserving order.

    Args:
        trace_length: Number of tokens in the thinking trace (excluding prompt).
        budget_checkpoints: List of target token budgets (e.g. [2000, 4000, 8000]).

    Returns:
        Deduplicated list of truncation lengths, sorted ascending.

    Example:
        >>> compute_budget_truncation_lengths(6000, [4000, 8000])
        [4000, 6000]
        >>> compute_budget_truncation_lengths(3000, [2000, 4000, 8000])
        [2000, 3000]
        >>> compute_budget_truncation_lengths(6000, [0.25, 0.5, 1.0])
        [1500, 3000, 6000]
    """
    clamped = [resolve_budget_checkpoint(trace_length, b) for b in budget_checkpoints]
    # Deduplicate while keeping sorted order
    seen = set()
    deduped = []
    for length in sorted(clamped):
        if length > 0 and length not in seen:
            seen.add(length)
            deduped.append(length)
    return deduped


def resolve_budget_checkpoint(
    trace_length: int,
    budget_checkpoint: int | float,
) -> int:
    """Resolve one configured budget into an absolute token count.

    Relative budgets are detected by type/value:
    - ``float`` in ``(0, 1]`` => relative fraction of ``trace_length``
    - everything else => absolute token budget
    """
    if isinstance(budget_checkpoint, float) and 0.0 < budget_checkpoint <= 1.0:
        resolved = int(round(trace_length * budget_checkpoint))
    else:
        resolved = int(budget_checkpoint)
    return min(trace_length, resolved)


def build_forced_output_tokens(
    thinking_trace_ids: list[int],
    truncation_length: int,
    think_end_token_id: int,
    suffix_token_ids: list[int],
) -> list[int]:
    """Build the token sequence for a forced output at a given budget.

    Truncates the thinking trace, appends </think> token, then appends
    the forced output suffix tokens.

    Args:
        thinking_trace_ids: The full thinking trace token IDs (response only, no prompt).
        truncation_length: Number of tokens to keep from the thinking trace.
        think_end_token_id: Token ID for </think>.
        suffix_token_ids: Token IDs for the forced output suffix
            (e.g. "If I were to output the final answer now, it would be \\boxed{").

    Returns:
        List of token IDs: truncated_trace + [</think>] + suffix_token_ids.
    """
    truncated = list(thinking_trace_ids[:truncation_length])
    # If the trace already ends with </think>, don't double it
    if truncated and truncated[-1] == think_end_token_id:
        return truncated + suffix_token_ids
    return truncated + [think_end_token_id] + suffix_token_ids


def aggregate_rewards(
    rewards_per_budget: list[float],
    aggregation_mode: str = "mean",
) -> float:
    """Aggregate rewards across multiple budget checkpoints into a single scalar.

    Args:
        rewards_per_budget: List of reward values, one per budget checkpoint.
        aggregation_mode: How to aggregate.
            - "mean": Simple average.
            - "linear_weighted": Linearly increasing weights
              (weight_i = i+1 for i in 0..n-1, then normalized).

    Returns:
        Aggregated scalar reward.

    Raises:
        ValueError: If aggregation_mode is unknown or rewards list is empty.
    """
    if not rewards_per_budget:
        raise ValueError("rewards_per_budget cannot be empty")

    if len(rewards_per_budget) == 1:
        return rewards_per_budget[0]

    if aggregation_mode == "mean":
        return float(np.mean(rewards_per_budget))
    elif aggregation_mode == "linear_weighted":
        n = len(rewards_per_budget)
        # weights: 1, 2, 3, ..., n  (later budgets get higher weight)
        weights = np.arange(1, n + 1, dtype=np.float64)
        weights = weights / weights.sum()
        return float(np.dot(weights, rewards_per_budget))
    else:
        raise ValueError(
            f"Unknown reward aggregation mode: {aggregation_mode}. "
            f"Supported: 'mean', 'linear_weighted'."
        )


def aggregate_rewards_batch(
    rewards_matrix: list[list[float]],
    aggregation_mode: str = "mean",
) -> list[float]:
    """Aggregate rewards for a batch of traces, each with multiple budget checkpoints.

    Args:
        rewards_matrix: List of lists. rewards_matrix[i] contains the rewards
            for trace i across its budget checkpoints.
        aggregation_mode: "mean" or "linear_weighted".

    Returns:
        List of aggregated rewards, one per trace.
    """
    return [aggregate_rewards(r, aggregation_mode) for r in rewards_matrix]


def aggregate_multi_answer_rewards(
    per_answer_rewards: list[float],
    method: str = "mean",
) -> float:
    """Aggregate rewards from multiple forced-output answers at a single budget.

    When the model generates multiple answers at the same budget checkpoint,
    this function reduces them to a single scalar reward for that budget.

    Args:
        per_answer_rewards: List of reward values, one per generated answer.
        method: How to aggregate.
            - "mean": Simple average of all answer rewards.
            - "max": Maximum reward across answers (optimistic).

    Returns:
        Aggregated scalar reward for the budget.

    Raises:
        ValueError: If method is unknown or per_answer_rewards is empty.
    """
    if not per_answer_rewards:
        raise ValueError("per_answer_rewards cannot be empty")

    if len(per_answer_rewards) == 1:
        return per_answer_rewards[0]

    if method == "mean":
        return float(np.mean(per_answer_rewards))
    elif method == "max":
        return float(max(per_answer_rewards))
    else:
        raise ValueError(
            f"Unknown multi-answer aggregation method: {method}. "
            f"Supported: 'mean', 'max'."
        )


def create_token_level_reward_tensor(
    aggregated_rewards: list[float],
    response_length: int,
    response_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert per-trace scalar rewards into a token-level reward tensor.

    Places the reward on the last valid token position of each sequence
    (same convention as standard outcome-based reward in GRPO).

    Args:
        aggregated_rewards: List of scalar rewards, one per trace in the batch.
        response_length: Number of response token positions.
        response_mask: Optional (batch_size, response_length) mask. If provided,
            the reward is placed on the last '1' position.

    Returns:
        Tensor of shape (batch_size, response_length) with rewards placed
        at the last valid token position.
    """
    batch_size = len(aggregated_rewards)
    reward_tensor = torch.zeros(batch_size, response_length, dtype=torch.float32)

    for i, reward in enumerate(aggregated_rewards):
        if response_mask is not None:
            # Find the last valid token position
            valid_positions = response_mask[i].nonzero(as_tuple=True)[0]
            if len(valid_positions) > 0:
                last_pos = valid_positions[-1].item()
                reward_tensor[i, last_pos] = reward
            # else: no valid tokens, leave as zeros
        else:
            # Place at last position
            reward_tensor[i, -1] = reward

    return reward_tensor


def compute_unanimous_group_keep_mask(
    uids: np.ndarray,
    natural_correct: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build a keep mask that drops unanimous-correct / unanimous-incorrect groups.

    Groups are defined by identical ``uid`` values. For each group:
    - Keep the group if it is mixed (contains both True and False).
    - Drop the group if all values are True or all values are False.

    Args:
        uids: 1D uid array, one uid per sample.
        natural_correct: 1D bool-like array, one natural-trace correctness value
            per sample.

    Returns:
        Tuple of:
            - keep_mask: boolean mask of shape (num_samples,)
            - stats: dictionary with group/sample drop counts.

    Raises:
        ValueError: If inputs are not 1D or have mismatched lengths.
    """
    uids_np = np.asarray(uids)
    natural_np = np.asarray(natural_correct).reshape(-1).astype(bool)

    if uids_np.ndim != 1:
        raise ValueError(f"uids must be 1D, got shape={uids_np.shape}")
    if natural_np.ndim != 1:
        raise ValueError(f"natural_correct must be 1D, got shape={natural_np.shape}")
    if uids_np.shape[0] != natural_np.shape[0]:
        raise ValueError(
            "uids and natural_correct must have the same length, "
            f"got {uids_np.shape[0]} and {natural_np.shape[0]}"
        )

    keep_mask = np.ones(uids_np.shape[0], dtype=bool)
    unique_uids, inv = np.unique(uids_np, return_inverse=True)

    dropped_all_correct_groups = 0
    dropped_all_incorrect_groups = 0
    kept_mixed_groups = 0

    for g in range(len(unique_uids)):
        group_mask = inv == g
        group_correct = natural_np[group_mask]
        if np.all(group_correct):
            keep_mask[group_mask] = False
            dropped_all_correct_groups += 1
        elif not np.any(group_correct):
            keep_mask[group_mask] = False
            dropped_all_incorrect_groups += 1
        else:
            kept_mixed_groups += 1

    stats = {
        "groups_total": int(len(unique_uids)),
        "groups_kept_mixed": int(kept_mixed_groups),
        "groups_dropped_all_correct": int(dropped_all_correct_groups),
        "groups_dropped_all_incorrect": int(dropped_all_incorrect_groups),
        "samples_dropped": int((~keep_mask).sum()),
        "samples_kept": int(keep_mask.sum()),
    }
    return keep_mask, stats
