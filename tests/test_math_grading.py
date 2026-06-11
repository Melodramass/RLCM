"""Freeze answer extraction/grading used by training rewards (verl math_dapo)
and the eval pipeline's extended variant (sympy symbolic equality)."""

import pytest

from verl.utils.reward_score.math_dapo import (
    compute_score,
    is_correct_minerva,
    is_correct_strict_box,
    last_boxed_only_string,
    normalize_answer,
    normalize_final_answer,
    remove_boxed,
    verify_extracted_answer,
)


class TestBoxedExtraction:
    def test_simple_box(self):
        assert last_boxed_only_string("the answer is \\boxed{42}.") == "\\boxed{42}"

    def test_nested_braces(self):
        s = "\\boxed{\\frac{1}{2}}"
        assert last_boxed_only_string(s) == "\\boxed{\\frac{1}{2}}"

    def test_last_box_wins(self):
        s = "\\boxed{1} then \\boxed{2}"
        assert last_boxed_only_string(s) == "\\boxed{2}"

    def test_no_box_returns_none(self):
        assert last_boxed_only_string("no box here") is None

    def test_unclosed_box_returns_none(self):
        assert last_boxed_only_string("\\boxed{42") is None

    def test_remove_boxed(self):
        assert remove_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


class TestNormalization:
    def test_normalize_final_answer_strips_units_and_spaces(self):
        assert normalize_final_answer("5 degrees") == "5"
        assert normalize_final_answer("x = 5") == "5"

    def test_normalize_final_answer_comma_digits(self):
        assert normalize_final_answer("1,000") == "1000"

    def test_normalize_final_answer_frac_shorthand(self):
        assert normalize_final_answer("\\frac12") == "\\frac{1}{2}"

    def test_normalize_answer_leading_zeros_and_decimal(self):
        assert normalize_answer("073") == "73"
        assert normalize_answer("7.0") == "7"
        assert normalize_answer("-007") == "-7"
        assert normalize_answer("000") == "0"
        assert normalize_answer(None) is None


class TestGrading:
    def test_strict_box_correct(self):
        score, pred = is_correct_strict_box("thus \\boxed{42}", "42")
        assert score == 1
        assert pred == "42"

    def test_strict_box_incorrect(self):
        score, pred = is_correct_strict_box("thus \\boxed{41}", "42")
        assert score == -1

    def test_strict_box_missing_box(self):
        score, pred = is_correct_strict_box("no final answer", "42")
        assert score == -1
        assert pred is None

    def test_strict_box_normalizes_both_sides(self):
        score, _ = is_correct_strict_box("\\boxed{042}", "42.0")
        assert score == 1

    def test_minerva_answer_pattern(self):
        correct, pred = is_correct_minerva("blah\nAnswer: 42", "42")
        assert correct is True
        assert pred == "42"

    def test_verify_extracted_answer(self):
        assert verify_extracted_answer("1,000", "1000")[0] is True
        assert verify_extracted_answer("41", "42")[0] is False
        assert verify_extracted_answer("", "42")[0] is False

    def test_compute_score_strict_box(self):
        out = compute_score("\\boxed{42}", "42", strict_box_verify=True)
        assert out["score"] == 1.0
        assert out["acc"] is True
        out = compute_score("\\boxed{41}", "42", strict_box_verify=True)
        assert out["score"] == -1.0


class TestEvalSideSymbolicEquality:
    """The eval pipeline's math_dapo adds sympy-based equivalence on top."""

    def test_symbolic_equivalence(self):
        import math_dapo as eval_math_dapo

        assert eval_math_dapo.is_symbolically_equal("1/2", "0.5") is True
        assert eval_math_dapo.is_symbolically_equal("x + 1", "1 + x") is True
        assert eval_math_dapo.is_symbolically_equal("2", "3") is False

    def test_unparseable_returns_false(self):
        import math_dapo as eval_math_dapo

        assert eval_math_dapo.is_symbolically_equal("\\fakecmd{", "42") is False
