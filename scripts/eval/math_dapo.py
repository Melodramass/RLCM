# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
from typing import Optional

from sympy import simplify, sympify, SympifyError
from sympy.parsing.latex import parse_latex


def is_symbolically_equal(expr1: str, expr2: str) -> bool:
    """Check if two LaTeX/math expressions are symbolically equivalent using sympy.

    Tries parse_latex first, falls back to sympify for simpler expressions.
    Returns False on any parsing failure rather than raising.
    """
    def _parse(s):
        # Unwrap single-element lists
        if isinstance(s, (list, tuple)) and len(s) == 1:
            s = s[0]
        if not isinstance(s, str):
            try:
                s = str(s)
            except Exception:
                return None
        # Strip surrounding $ if present
        s = s.strip().strip("$").strip()
        # Try LaTeX parsing first
        try:
            return parse_latex(s)
        except Exception:
            pass
        # Fall back to sympify
        try:
            return sympify(s)
        except (SympifyError, TypeError, ValueError):
            return None

    a = _parse(expr1)
    b = _parse(expr2)
    if a is None or b is None:
        return False
    try:
        return simplify(a - b) == 0
    except Exception:
        return False


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize LaTeX variants
    final_answer = final_answer.replace("\\dfrac", "\\frac")
    final_answer = final_answer.replace("\\tfrac", "\\frac")
    final_answer = final_answer.replace("\\left(", "(").replace("\\right)", ")")
    final_answer = final_answer.replace("\\left[", "[").replace("\\right]", "]")
    final_answer = final_answer.replace("\\cdot", "*")
    final_answer = final_answer.replace("\\times", "*")
    # Remove braces around single characters in exponents/subscripts: ^{k} -> ^k, _{n} -> _n
    final_answer = re.sub(r'\^{([^{}])}', r'^\1', final_answer)
    final_answer = re.sub(r'_{([^{}])}', r'_\1', final_answer)

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def normalize_answer(s: Optional[str]) -> Optional[str]:
    """Normalize an answer string for comparison.

    Handles:
    - Leading zeros (e.g., "073" -> "73")
    - Trailing .0 for integers (e.g., "7.0" -> "7")
    - Whitespace stripping
    """
    if s is None:
        return None
    s = s.strip()
    # Remove trailing .0 for integers
    if s.endswith('.0'):
        s = s[:-2]
    # Remove leading zeros (but keep '0' itself and handle negative numbers)
    if s.startswith('-'):
        # Handle negative numbers
        rest = s[1:].lstrip('0')
        s = '-' + (rest if rest else '0')
    else:
        stripped = s.lstrip('0')
        s = stripped if stripped else '0'
    return s


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract and check the boxed answer from the full prediction
    # (no truncation here - truncation is done in compute_score if needed)
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    def _normalize_strict_candidate(value: Optional[str]) -> Optional[str]:
        """Normalize strict-box strings robustly while preserving boxed-only constraint."""
        if value is None:
            return None
        # Coerce safely so non-str ground truth formats do not crash normalization.
        normalized = normalize_final_answer(str(value))
        return normalize_answer(normalized)

    # Normalize both for comparison.
    # This keeps strict boxed extraction, but allows equivalent GT formats
    # such as "$2n$" or "$f(x)=2 x$" to match normalized predictions.
    extracted_norm = _normalize_strict_candidate(extracted_pred)
    gt_norm = _normalize_strict_candidate(gt)

    # 1. Try exact (normalized) string match first
    if extracted_norm == gt_norm and extracted_norm is not None:
        return 1, extracted_pred

    # 2. Try numeric evaluation (catches e.g. "147556443852" vs "\frac{4\cdot999^4}{27}")
    if extracted_pred is not None:
        try:
            # Use normalized forms which already have \cdot -> * etc.
            pred_val = float(eval(extracted_norm.replace("^", "**"))) if extracted_norm else None
            gt_val = float(eval(gt_norm.replace("^", "**"))) if gt_norm else None
            if pred_val is not None and gt_val is not None and abs(pred_val - gt_val) < 1e-6 * max(1, abs(gt_val)):
                return 1, extracted_pred
        except Exception:
            pass

    # 3. Fall back to symbolic equivalence check
    if extracted_pred is not None and is_symbolically_equal(extracted_pred, gt):
        return 1, extracted_pred

    return -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None
) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Use full solution string for boxed answer extraction
    # The boxed answer can appear anywhere in long reasoning outputs
    # No truncation needed since last_boxed_only_string searches for the last \boxed{}

    # Verify the solution
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }
