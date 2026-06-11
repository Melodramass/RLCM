"""Freeze the RLCR baseline scoring: <answer>/<confidence> tag parsing and
score = acc - (acc - confidence)^2."""

import pytest

from verl.utils.reward_score.rlcr import compute_score, extract_answer_and_confidence


class TestExtractAnswerAndConfidence:
    def test_basic_extraction(self):
        s = "reasoning... <answer>42</answer> <confidence>0.8</confidence>"
        answer, conf, valid = extract_answer_and_confidence(s)
        assert answer == "42"
        assert conf == pytest.approx(0.8)
        assert valid is True

    def test_last_tags_win(self):
        s = (
            "<answer>1</answer> <confidence>0.1</confidence>"
            "更新后 <answer>2</answer> <confidence>0.9</confidence>"
        )
        answer, conf, valid = extract_answer_and_confidence(s)
        assert answer == "2"
        assert conf == pytest.approx(0.9)

    def test_percentage_style_confidence(self):
        _, conf, valid = extract_answer_and_confidence("<confidence>85</confidence>")
        assert conf == pytest.approx(0.85)
        assert valid is True

    def test_boundary_value_one_kept_as_is(self):
        _, conf, valid = extract_answer_and_confidence("<confidence>1</confidence>")
        assert conf == pytest.approx(1.0)
        assert valid is True

    def test_missing_confidence_defaults_to_half(self):
        answer, conf, valid = extract_answer_and_confidence("<answer>7</answer>")
        assert answer == "7"
        assert conf == pytest.approx(0.5)
        assert valid is False

    def test_missing_answer_returns_none(self):
        answer, conf, valid = extract_answer_and_confidence("no tags at all")
        assert answer is None
        assert conf == pytest.approx(0.5)
        assert valid is False

    def test_non_numeric_confidence_invalid(self):
        _, conf, valid = extract_answer_and_confidence("<confidence>high</confidence>")
        assert conf == pytest.approx(0.5)
        assert valid is False

    def test_out_of_range_confidence_invalid(self):
        _, conf, valid = extract_answer_and_confidence("<confidence>150</confidence>")
        assert conf == pytest.approx(0.5)
        assert valid is False

    def test_confidence_embedded_in_text(self):
        _, conf, valid = extract_answer_and_confidence(
            "<confidence>I'd say 0.75 roughly</confidence>"
        )
        assert conf == pytest.approx(0.75)
        assert valid is True

    def test_multiline_tags(self):
        s = "<answer>\n  x = 3\n</answer><confidence>\n0.6\n</confidence>"
        answer, conf, _ = extract_answer_and_confidence(s)
        assert answer == "x = 3"
        assert conf == pytest.approx(0.6)


class TestComputeScore:
    def test_correct_high_confidence(self):
        out = compute_score("<answer>42</answer><confidence>0.9</confidence>", "42")
        assert out["acc"] == 1
        assert out["confidence"] == pytest.approx(0.9)
        assert out["brier"] == pytest.approx(0.01)
        assert out["score"] == pytest.approx(0.99)

    def test_incorrect_high_confidence_penalized(self):
        out = compute_score("<answer>41</answer><confidence>0.9</confidence>", "42")
        assert out["acc"] == 0
        assert out["brier"] == pytest.approx(0.81)
        assert out["score"] == pytest.approx(-0.81)

    def test_incorrect_low_confidence_mild_penalty(self):
        out = compute_score("<answer>41</answer><confidence>0.1</confidence>", "42")
        assert out["acc"] == 0
        assert out["score"] == pytest.approx(-0.01)

    def test_missing_answer_scored_as_incorrect(self):
        out = compute_score("no tags", "42")
        assert out["acc"] == 0
        assert out["answer_valid"] is False
        # default confidence 0.5 -> score = 0 - 0.25
        assert out["score"] == pytest.approx(-0.25)

    def test_answer_normalization_applies(self):
        # normalize_final_answer strips spaces/dollar signs etc.
        out = compute_score("<answer>$1,000$</answer><confidence>1</confidence>", "1000")
        assert out["acc"] == 1
        assert out["score"] == pytest.approx(1.0)
