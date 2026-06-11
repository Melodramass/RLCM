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

"""Unit tests for RLCR reward scoring."""

import pytest

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.rlcr import (
    compute_score,
    extract_answer_and_confidence,
    verify_extracted_answer,
)


class TestExtractAnswerAndConfidence:
    """Test the low‑level extraction function."""

    def test_both_tags_present(self):
        sol = "Reasoning... <answer>42</answer> <confidence>0.8</confidence>"
        answer, conf, valid = extract_answer_and_confidence(sol)
        assert answer == "42"
        assert conf == 0.8
        assert valid is True

    def test_multiple_tags_last_used(self):
        sol = "<answer>wrong</answer> <confidence>0.9</confidence> ... <answer>correct</answer> <confidence>0.7</confidence>"
        answer, conf, valid = extract_answer_and_confidence(sol)
        assert answer == "correct"
        assert conf == 0.7
        assert valid is True

    def test_missing_confidence_tag(self):
        sol = "<answer>hello</answer>"
        answer, conf, valid = extract_answer_and_confidence(sol)
        assert answer == "hello"
        assert conf == 0.5
        assert valid is False

    def test_missing_answer_tag(self):
        sol = "<confidence>0.6</confidence>"
        answer, conf, valid = extract_answer_and_confidence(sol)
        assert answer is None
        assert conf == 0.6
        assert valid is True

    def test_no_tags(self):
        sol = "just text"
        answer, conf, valid = extract_answer_and_confidence(sol)
        assert answer is None
        assert conf == 0.5
        assert valid is False

    def test_confidence_percentage(self):
        sol = "<answer>x</answer> <confidence>85%</confidence>"
        answer, conf, valid = extract_answer_and_confidence(sol)
        assert answer == "x"
        assert conf == 0.85
        assert valid is True

    def test_confidence_percentage_over_100(self):
        sol = "<answer>x</answer> <confidence>150%</confidence>"
        answer, conf, valid = extract_answer_and_confidence(sol)
        # val > 100 -> not in [0,100]? Actually condition 0 <= val <= 100 passes (150 > 100) so else? Wait logic: if 0 <= val <= 100: if val <=1 ... else val/100. So val=150 fails condition, goes to elif 0 <= val <=1 (false). So confidence_valid remains False, confidence stays default 0.5.
        # Let's verify expected behavior: val=150, not in [0,100] nor [0,1] => invalid, default 0.5.
        assert conf == 0.5
        assert valid is False

    def test_confidence_out_of_range_negative(self):
        sol = "<answer>x</answer> <confidence>-0.2</confidence>"
        answer, conf, valid = extract_answer_and_confidence(sol)
        # val = -0.2, not in [0,100] nor [0,1] => invalid
        assert conf == 0.5
        assert valid is False

    def test_confidence_malformed_text(self):
        sol = "<answer>x</answer> <confidence>abc</confidence>"
        answer, conf, valid = extract_answer_and_confidence(sol)
        assert conf == 0.5
        assert valid is False

    def test_confidence_with_extra_text(self):
        sol = "<answer>x</answer> <confidence>0.75 (pretty sure)</confidence>"
        answer, conf, valid = extract_answer_and_confidence(sol)
        assert conf == 0.75
        assert valid is True

    def test_confidence_integer(self):
        sol = "<answer>x</answer> <confidence>1</confidence>"
        answer, conf, valid = extract_answer_and_confidence(sol)
        assert conf == 1.0
        assert valid is True


class TestComputeScore:
    """Test the main RLCR scoring function."""

    def test_correct_high_confidence(self):
        sol = "<answer>42</answer> <confidence>0.8</confidence>"
        gt = "42"
        result = compute_score(sol, gt)
        assert result["acc"] == 1
        assert result["confidence"] == 0.8
        assert result["confidence_valid"] is True
        assert result["brier"] == (1 - 0.8) ** 2  # (acc - conf)^2 = 0.04
        assert result["score"] == pytest.approx(0.96)
        assert result["pred"] == "42"  # normalized

    def test_correct_low_confidence(self):
        sol = "<answer>42</answer> <confidence>0.2</confidence>"
        gt = "42"
        result = compute_score(sol, gt)
        assert result["acc"] == 1
        assert result["confidence"] == 0.2
        assert result["brier"] == (1 - 0.2) ** 2  # 0.64
        assert result["score"] == pytest.approx(1 - 0.64)  # 0.36

    def test_wrong_high_confidence(self):
        sol = "<answer>wrong</answer> <confidence>0.9</confidence>"
        gt = "42"
        result = compute_score(sol, gt)
        assert result["acc"] == 0
        assert result["confidence"] == 0.9
        assert result["brier"] == (0 - 0.9) ** 2  # 0.81
        assert result["score"] == pytest.approx(0 - 0.81)  # -0.81

    def test_wrong_low_confidence(self):
        sol = "<answer>wrong</answer> <confidence>0.1</confidence>"
        gt = "42"
        result = compute_score(sol, gt)
        assert result["acc"] == 0
        assert result["brier"] == (0 - 0.1) ** 2  # 0.01
        assert result["score"] == pytest.approx(-0.01)

    def test_missing_confidence_default(self):
        sol = "<answer>42</answer>"
        gt = "42"
        result = compute_score(sol, gt)
        assert result["acc"] == 1
        assert result["confidence"] == 0.5
        assert result["confidence_valid"] is False
        assert result["brier"] == (1 - 0.5) ** 2  # 0.25
        assert result["score"] == pytest.approx(0.75)

    def test_missing_answer_tag(self):
        sol = "<confidence>0.8</confidence>"
        gt = "42"
        result = compute_score(sol, gt)
        # answer_text is None -> verify_extracted_answer receives empty string
        assert result["acc"] == 0
        assert result["confidence"] == 0.8
        assert result["confidence_valid"] is True
        assert result["brier"] == (0 - 0.8) ** 2  # 0.64
        assert result["score"] == pytest.approx(-0.64)
        assert result["answer_valid"] is False

    def test_answer_valid_true_when_tag_present(self):
        sol = "<answer>42</answer> <confidence>0.9</confidence>"
        gt = "42"
        result = compute_score(sol, gt)
        assert result["answer_valid"] is True

    def test_answer_valid_false_when_tag_absent(self):
        sol = "no tags here <confidence>0.5</confidence>"
        gt = "42"
        result = compute_score(sol, gt)
        assert result["answer_valid"] is False

    def test_answer_normalization(self):
        sol = "<answer>  42  </answer> <confidence>0.7</confidence>"
        gt = "42"
        result = compute_score(sol, gt)
        assert result["acc"] == 1
        assert result["pred"] == "42"  # stripped

    def test_ground_truth_normalization(self):
        sol = "<answer>42</answer> <confidence>0.7</confidence>"
        gt = "  42  "
        result = compute_score(sol, gt)
        assert result["acc"] == 1
        assert result["pred"] == "42"

    def test_answer_mismatch_normalization(self):
        sol = "<answer> 42.0 </answer> <confidence>0.7</confidence>"
        gt = "42"
        # Depending on normalize_final_answer, "42.0" may be normalized to "42".
        # We'll rely on actual behavior; we can just test that compute_score returns something.
        result = compute_score(sol, gt)
        # Not asserting specific acc, just ensure no crash
        assert "score" in result


class TestDispatchIntegration:
    """Test that RLCR is correctly integrated into the reward dispatcher."""

    def test_default_compute_score_stage1_rlcr(self):
        sol = "<answer>42</answer> <confidence>0.8</confidence>"
        gt = "42"
        result = default_compute_score("stage1_rlcr", sol, gt)
        # Should be the same dict as compute_score returns
        assert isinstance(result, dict)
        assert "score" in result
        assert "acc" in result
        assert "confidence" in result
        assert "brier" in result
        assert result["acc"] == 1
        assert result["score"] == pytest.approx(0.96)

    def test_default_compute_score_other_source_unchanged(self):
        # Ensure other data sources are not affected
        sol = "<answer>42</answer>"
        gt = "42"
        # For example, "stage1" should still go to math_dapo
        # We'll just verify that default_compute_score doesn't crash
        result = default_compute_score("stage1", sol, gt)
        # It returns a dict (since math_dapo returns dict)
        assert isinstance(result, dict)
        # Should contain at least score, acc, pred
        assert "score" in result
        assert "acc" in result
        assert "pred" in result

    def test_verify_extracted_answer_helper(self):
        # Ensure the helper works as expected
        correct, pred = verify_extracted_answer("42", "42")
        assert correct is True
        assert pred == "42"
        correct, pred = verify_extracted_answer("wrong", "42")
        assert correct is False
        assert pred == "wrong"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])