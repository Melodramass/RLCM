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

"""RLCR (Reward with Learned Confidence Rewards) scoring module.

Extracts answer and confidence from RLCR‑formatted solution strings
(<answer>...</answer> and <confidence>...</confidence>) and computes the
RLCR reward: score = accuracy - (accuracy - confidence)^2.
"""

import re
from typing import Optional, Tuple

from .math_dapo import verify_extracted_answer


def extract_answer_and_confidence(
    solution_str: str,
) -> Tuple[Optional[str], Optional[float], bool]:
    """Extract the last <answer> and <confidence> tags from a solution string.

    Args:
        solution_str: The raw solution string (may contain reasoning, analysis, etc.)

    Returns:
        Tuple of (answer_text, confidence_value, confidence_valid).
        - answer_text: The content of the last <answer> tag, or None if not found.
        - confidence_value: Parsed confidence float in [0,1]; defaults to 0.5 if missing/invalid.
        - confidence_valid: True if a valid confidence tag was found and parsed.
    """
    # Extract last <answer>...</answer>
    answer_match = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    answer_text = answer_match[-1].group(1).strip() if answer_match else None

    # Extract last <confidence>...</confidence>
    conf_match = list(re.finditer(r"<confidence>(.*?)</confidence>", solution_str, re.DOTALL))
    conf_raw = conf_match[-1].group(1).strip() if conf_match else None

    confidence = 0.5
    confidence_valid = False

    if conf_raw is not None:
        # Try to parse a number from the tag content
        num_match = re.search(r"[-+]?\d*\.?\d+", conf_raw)
        if num_match:
            try:
                val = float(num_match.group())
                # Handle percentage‑style values (e.g., "85" -> 0.85)
                if 0 <= val <= 100:
                    if val <= 1:
                        # Already in [0,1] range
                        confidence = val
                    else:
                        confidence = val / 100.0
                    confidence_valid = True
                elif 0 <= val <= 1:
                    confidence = val
                    confidence_valid = True
                # If val outside [0,100] we treat as invalid and keep default 0.5
            except ValueError:
                pass

    return answer_text, confidence, confidence_valid


def compute_score(
    solution_str: str,
    ground_truth: str,
    **kwargs,
) -> dict:
    """Compute the RLCR reward score.

    The RLCR reward is defined as:
        acc = 1 if extracted answer matches ground truth else 0
        brier = (acc - confidence)^2
        score = acc - brier

    Returns a dictionary with keys:
        score (float): the RLCR reward (acc - brier)
        acc (int): 0 or 1
        pred (str): the extracted answer text (or empty string)
        confidence (float): confidence value used (0‑1)
        confidence_valid (bool): whether confidence was successfully parsed
        brier (float): (acc - confidence)^2
    """
    answer_text, confidence, confidence_valid = extract_answer_and_confidence(solution_str)

    # Use the shared verification helper (normalizes both strings)
    correct, norm_pred = verify_extracted_answer(
        answer_text if answer_text is not None else "",
        ground_truth,
    )
    acc = 1 if correct else 0

    brier = (acc - confidence) ** 2
    score = acc - brier

    return {
        "score": score,
        "acc": acc,
        "pred": norm_pred,
        "confidence": confidence,
        "confidence_valid": confidence_valid,
        "answer_valid": answer_text is not None,
        "brier": brier,
    }