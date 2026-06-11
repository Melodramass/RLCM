"""
Metric utilities for probe training and evaluation.

Trimmed to the metrics the evaluation actually reports:
  * Expected Calibration Error (ECE)  -- natural-trace calibration
  * Brier score                       -- used internally by probe training
  * SARA numerical helpers            -- alternative correctness for numeric data

The earlier PCE / log-loss / over-confidence / AUROC helpers were only consumed
by the deprecated plotting scripts and have been removed.
"""

import math
from typing import Optional

import numpy as np
import torch

# Import robust answer extraction utilities from math_dapo
from math_dapo import (
    last_boxed_only_string,
    remove_boxed,
    normalize_answer,
)


def _compute_ece(
    confidences: torch.Tensor,
    accuracies: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    bins: int = 10,
) -> Optional[float]:
    """Compute Expected Calibration Error (ECE).

    Args:
        confidences: Tensor of confidence scores in [0, 1].
        accuracies: Tensor of accuracy scores (0 or 1).
        mask: Optional mask for valid samples.
        bins: Number of equal-width confidence bins.

    Returns:
        ECE value or None if computation fails.
    """
    if confidences is None or accuracies is None or confidences.numel() == 0:
        return None
    if mask is None:
        valid_mask = torch.ones_like(confidences, dtype=torch.bool)
    else:
        valid_mask = mask > 0
    if not torch.any(valid_mask):
        return None
    valid_conf = confidences[valid_mask]
    valid_acc = accuracies[valid_mask]
    bin_boundaries = torch.linspace(0, 1, bins + 1, device=valid_conf.device)
    total = valid_conf.numel()
    ece = torch.tensor(0.0, device=valid_conf.device)
    for i in range(bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1] if i < bins - 1 else bin_boundaries[i + 1] + 1e-6
        bin_mask = (valid_conf >= lower) & (valid_conf < upper)
        if torch.any(bin_mask):
            weight = bin_mask.float().sum() / total
            bin_conf = valid_conf[bin_mask].mean()
            bin_acc = valid_acc[bin_mask].mean()
            ece += weight * torch.abs(bin_acc - bin_conf)
    return ece.detach().item()


def _compute_brier_score(
    confidences: torch.Tensor,
    accuracies: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Optional[float]:
    """Compute Brier score (mean squared confidence error).

    Args:
        confidences: Tensor of confidence scores in [0, 1].
        accuracies: Tensor of accuracy scores (0 or 1).
        mask: Optional mask for valid samples.

    Returns:
        Brier score or None if computation fails.
    """
    if confidences is None or accuracies is None or confidences.numel() == 0:
        return None
    if mask is None:
        valid_mask = torch.ones_like(confidences, dtype=torch.bool)
    else:
        valid_mask = mask > 0
    if not torch.any(valid_mask):
        return None
    valid_conf = confidences[valid_mask]
    valid_acc = accuracies[valid_mask]
    brier = (valid_conf - valid_acc.float()) ** 2
    return brier.mean().detach().item()


def _compute_sara_risk(
    delta_y: torch.Tensor, y_actual: torch.Tensor, refused: torch.Tensor
) -> torch.Tensor:
    # Use torch.maximum for element-wise comparison (Python max() fails on tensors).
    threshold = torch.maximum(torch.full_like(y_actual, 5000.0), 0.1 * y_actual)

    # Nested torch.where for piecewise logic. 'refused' must be a BoolTensor.
    costs = torch.where(
        refused,
        torch.tensor(270.0, device=delta_y.device),
        torch.where(
            delta_y < 0,
            -delta_y,
            torch.where(
                delta_y > threshold,
                0.2 * delta_y,
                torch.tensor(0.0, device=delta_y.device),
            ),
        ),
    )
    return torch.mean(costs)


def _extract_numeric_answer(text: str) -> Optional[float]:
    """Extract numeric answer from solution text using robust math_dapo utilities.

    Uses the same extraction logic as the scoring function for consistency:
    boxed extraction -> remove wrapper -> normalize -> parse as float.
    """
    boxed_pred = last_boxed_only_string(text)
    if boxed_pred is None:
        return None

    try:
        extracted_pred = remove_boxed(boxed_pred)
    except (AssertionError, IndexError):
        return None

    normalized = normalize_answer(extracted_pred)
    if normalized is None:
        return None

    cleaned = normalized.replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def compute_sara_numerical_risk(solution_str: str, ground_truth: str) -> Optional[float]:
    """Compute SARA risk for numerical reasoning problems."""
    predicted = _extract_numeric_answer(solution_str)

    try:
        gt_normalized = normalize_answer(str(ground_truth))
        if gt_normalized is None:
            return None
        gt_cleaned = gt_normalized.replace("$", "").replace(",", "").strip()
        gt_value = float(gt_cleaned)
    except (ValueError, AttributeError, TypeError):
        return None

    refused = predicted is None
    pred_value = predicted if predicted is not None else 0.0
    delta = pred_value - gt_value

    delta_tensor = torch.tensor([delta], dtype=torch.float32)
    gt_tensor = torch.tensor([gt_value], dtype=torch.float32)
    refused_tensor = torch.tensor([refused], dtype=torch.bool)

    risk_tensor = _compute_sara_risk(delta_tensor, gt_tensor, refused_tensor)
    return risk_tensor.item()


def compute_sara_integer_part_accuracy(solution_str: str, ground_truth: str) -> Optional[bool]:
    """Compute SARA numerical correctness by integer-part match only.

    Keeps boxed-answer extraction but ignores decimal differences:
    e.g., 123.9 and 123.1 are both treated as integer part 123.
    """
    predicted = _extract_numeric_answer(solution_str)
    if predicted is None:
        return False

    try:
        gt_normalized = normalize_answer(str(ground_truth))
        if gt_normalized is None:
            return None
        gt_cleaned = gt_normalized.replace("$", "").replace(",", "").strip()
        gt_value = float(gt_cleaned)
    except (ValueError, AttributeError, TypeError):
        return None

    def _round_half_away_from_zero(x: float) -> int:
        if x >= 0:
            return int(math.floor(x + 0.5))
        return int(math.ceil(x - 0.5))

    return _round_half_away_from_zero(predicted) == _round_half_away_from_zero(gt_value)
