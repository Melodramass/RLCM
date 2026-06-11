"""Freeze the eval pipeline's calibration metrics (ECE, Brier) and the SARA
numeric-extraction helper against hand-computed values."""

import pytest
import torch

import metric_utils


# Shared fixture: 4 samples, 10 bins.
#   bin [0.9, 1.0]: conf 0.95, 0.95 -> mean conf 0.95, mean acc 0.5, weight 0.5
#   bin [0.6, 0.7): conf 0.65       -> acc 1.0, weight 0.25  (underconfident)
#   bin [0.0, 0.1): conf 0.05       -> acc 0.0, weight 0.25  (overconfident)
CONF = torch.tensor([0.95, 0.95, 0.65, 0.05])
ACC = torch.tensor([1.0, 0.0, 1.0, 0.0])


class TestECE:
    def test_hand_computed_value(self):
        ece = metric_utils._compute_ece(CONF, ACC, bins=10)
        expected = 0.5 * 0.45 + 0.25 * 0.35 + 0.25 * 0.05
        assert ece == pytest.approx(expected, abs=1e-6)

    def test_perfectly_calibrated_bin(self):
        conf = torch.tensor([0.75, 0.75, 0.75, 0.75])
        acc = torch.tensor([1.0, 1.0, 1.0, 0.0])
        assert metric_utils._compute_ece(conf, acc, bins=10) == pytest.approx(0.0, abs=1e-6)

    def test_mask_filters_samples(self):
        mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
        # Only the two 0.95-conf samples remain: |0.5 - 0.95| = 0.45
        assert metric_utils._compute_ece(CONF, ACC, mask=mask, bins=10) == pytest.approx(0.45, abs=1e-6)

    def test_empty_returns_none(self):
        assert metric_utils._compute_ece(torch.zeros(0), torch.zeros(0)) is None
        assert metric_utils._compute_ece(CONF, ACC, mask=torch.zeros(4)) is None


class TestBrier:
    def test_brier_hand_computed(self):
        brier = metric_utils._compute_brier_score(CONF, ACC)
        expected = (0.05**2 + 0.95**2 + 0.35**2 + 0.05**2) / 4
        assert brier == pytest.approx(expected, abs=1e-6)


class TestNumericExtraction:
    def test_extract_numeric_answer(self):
        assert metric_utils._extract_numeric_answer("so \\boxed{42}") == pytest.approx(42.0)
        assert metric_utils._extract_numeric_answer("\\boxed{1,000}") == pytest.approx(1000.0)
        assert metric_utils._extract_numeric_answer("no box") is None
        assert metric_utils._extract_numeric_answer("\\boxed{\\frac{1}{2}}") is None
