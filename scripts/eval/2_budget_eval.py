#!/usr/bin/env python3
"""
Budget-wise forced output evaluation script.

This script takes a reproduced jsonl file (e.g., reproduced_aime25_step200.jsonl),
cuts the chain-of-thought at predefined budgets, appends a forced answer prompt,
generates multiple forced outputs, and evaluates accuracy at each budget.

Usage:
    python reproduce_budget_eval.py \
        --input_file reproduced_aime25_step200.jsonl \
        --model_path /path/to/model \
        --output_file budget_eval_results.jsonl \
        --n_forced 8 \
        --budgets 500,1000,1500,2000
"""

import os
import sys
import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import multiprocessing
from tqdm import tqdm
from vllm import LLM, SamplingParams, TokensPrompt
from transformers import AutoTokenizer

# Ensure local eval utilities are importable regardless of launch directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from math_dapo import compute_score, last_boxed_only_string

from metric_utils import compute_sara_integer_part_accuracy, compute_sara_numerical_risk

# Try importing RLCR-specific functions (optional, for RLCR mode)
try:
    from verl.utils.reward_score.rlcr import (
        extract_answer_and_confidence,
        compute_score as compute_rlcr_score,
    )
    HAS_RLCR = True
except ImportError:
    HAS_RLCR = False

from verl.utils.reward_score.prime_code import compute_score as compute_code_score

# Default settings
DEFAULT_BUDGETS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
THINK_END_TAG = "</think>"
FORCED_ANSWER_PROMPT = "If I were to give the final answer now, the final answer would be \\boxed{"
FORCED_ANSWER_PROMPT_GPQA_XML = (
    "Answer now with exactly one XML tag. Use only A, B, C, or D.\n"
    "<answer>"
)
RLCR_RESPONSE_INSTRUCTION = (
    "I am out of time, let me just output the final answer and confidence analysis "
    "in exactly this format:\n"
    "<answer> final answer here </answer>\n"
    "<analysis> confidence and uncertainty analysis here </analysis>\n"
    "<confidence> number between 0 and 1 </confidence>"
)
FORCED_ANSWER_PROMPT_RLCR = (
    f"{RLCR_RESPONSE_INSTRUCTION}\n\n"
    "<answer>"
)
# Code tasks: just close the thinking and let the model write the final solution.
FORCED_ANSWER_PROMPT_CODE = "Here is my final Python solution:\n"
CODE_DATA_SOURCES = {"codecontests", "apps", "codeforces", "taco", "livecodebench"}
GPQA_DATA_SOURCES = {"gpqa", "gpqa_rlcr"}


def extract_gpqa_choice(text: Optional[str]) -> Optional[str]:
    """Normalize GPQA multiple-choice answers like '(A)' or 'Option A' to 'A'."""
    if text is None:
        return None
    value = str(text).strip().upper()
    if not value:
        return None

    match = re.search(r"\(([A-D])\)", value)
    if match:
        return match.group(1)

    match = re.search(r"\b(?:OPTION|CHOICE|ANSWER)\s*[:\-]?\s*([A-D])\b", value)
    if match:
        return match.group(1)

    if value[0] in "ABCD" and (len(value) == 1 or value[1] in {".", ")", ":", "-", " ", "\n", "\t"}):
        return value[0]

    match = re.search(r"\b([A-D])\b", value)
    if match:
        return match.group(1)

    return None


def extract_xml_answer_content(text: Optional[str]) -> Optional[str]:
    """Extract content from an <answer>...</answer> completion.

    The forced prompt already ends with "<answer>", so many completions start
    with "A</answer>" instead of repeating the opening tag.
    """
    if text is None:
        return None

    value = str(text).strip()
    if not value:
        return None

    candidates = [value]
    if "<answer>" not in value.lower() and "</answer>" in value.lower():
        candidates.append("<answer>" + value)

    for candidate in candidates:
        matches = list(re.finditer(r"<answer>(.*?)</answer>", candidate, re.IGNORECASE | re.DOTALL))
        if matches:
            answer = matches[-1].group(1).strip()
            return answer or None

    return None

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Monkey-patch get_context: force spawn only when no method is explicitly requested.
# Explicit "fork" requests (e.g. from code-scoring subprocesses that don't use CUDA)
# are left as-is so they benefit from near-instant fork startup (~0.01s vs ~2s spawn).
_original_get_context = multiprocessing.get_context
def _patched_get_context(method=None):
    if method is None:
        method = "spawn"
    return _original_get_context(method)
multiprocessing.get_context = _patched_get_context

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_reproduced_data(input_file: str) -> List[Dict]:
    """Load reproduced jsonl file and extract relevant fields."""
    data = []
    with open(input_file, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    logging.info(f"Loaded {len(data)} rollouts from {input_file}")
    return data


def extract_thinking_content(output: str) -> str:
    """Extract the thinking content from the output.

    If </think> is present, remove everything after it.
    This isolates just the chain-of-thought reasoning.
    """
    if THINK_END_TAG in output:
        # Find the position of </think> and extract only the thinking part
        think_end_idx = output.find(THINK_END_TAG)
        return output[:think_end_idx]
    return output


def build_budget_prompts(
    prompt: str,
    thinking_content: str,
    tokenizer: AutoTokenizer,
    budgets: List[int],
    closing_prompt: Optional[str] = None,
    always_generate: bool = False,
) -> List[Tuple[int, Optional[TokensPrompt], str, int, bool]]:
    """Build prompts with budget-cut CoT and forced answer prompt.

    If thinking is already complete within the budget (and always_generate is False),
    returns None for the TokensPrompt (caller should use the original output's answer instead).

    Args:
        closing_prompt: Override for the text appended after the truncated thinking.
            Defaults to "..." + THINK_END_TAG + FORCED_ANSWER_PROMPT (math boxed).
        always_generate: If True, never short-circuit on complete thinking
            (used for code and GPQA forced-answer evaluation).

    Returns list of (budget, token_prompt_or_None, truncated_cot_text, thinking_len, thinking_complete) tuples.
    """
    # Tokenize the prompt and thinking content
    prompt_token_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    thinking_token_ids = tokenizer(thinking_content, add_special_tokens=False).input_ids

    # Closing tokens for truncated thinking
    if closing_prompt is None:
        closing_prompt = "..." + THINK_END_TAG + "\n\n" + FORCED_ANSWER_PROMPT
    truncated_close_ids = tokenizer(closing_prompt, add_special_tokens=False).input_ids

    # Check if the original thinking contains a \boxed{} answer (only relevant for math)
    thinking_has_boxed = (not always_generate) and last_boxed_only_string(thinking_content) is not None

    result = []
    for budget in budgets:
        if len(thinking_token_ids) <= budget and thinking_has_boxed:
            # Thinking already finished with a boxed answer — no need to re-generate
            truncated_cot_text = tokenizer.decode(thinking_token_ids, skip_special_tokens=False)
            result.append((budget, None, truncated_cot_text, len(thinking_token_ids), True))
        else:
            # Thinking not finished — truncate and use closing prompt
            truncated_thinking_ids = thinking_token_ids[:budget]
            full_prompt_ids = prompt_token_ids + truncated_thinking_ids + truncated_close_ids
            truncated_cot_text = tokenizer.decode(truncated_thinking_ids, skip_special_tokens=False)
            result.append((budget, TokensPrompt(prompt_token_ids=full_prompt_ids), truncated_cot_text, len(truncated_thinking_ids), False))

    return result


def extract_boxed_content(text: str) -> Optional[str]:
    """Extract the content inside \\boxed{} from the forced output.

    Since we prompt with "The final answer is \\boxed{", the model just needs
    to complete the content and close the brace.
    """
    # The text should start right after \boxed{, so we need to find the closing brace
    # Handle nested braces
    brace_count = 1
    end_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break

    if end_idx == -1:
        # No proper closing brace found, try to extract what we can
        # Look for common terminators
        for term in ['}', '\n', '<']:
            if term in text:
                end_idx = text.find(term)
                break

    if end_idx > 0:
        return text[:end_idx].strip()
    elif len(text.strip()) > 0:
        return text.strip()
    return None


def build_budget_prompts_rlcr(
    prompt: str,
    thinking_content: str,
    tokenizer: AutoTokenizer,
    budgets: List[int],
) -> List[Tuple[int, TokensPrompt, str, int]]:
    """Build prompts with budget-cut CoT and RLCR forced answer prompt.
    
    RLCR always generates (no complete thinking check).

    Returns list of (budget, TokensPrompt, truncated_cot_text, truncated_length) tuples.
    """
    # Tokenize the prompt and thinking content
    prompt_token_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    thinking_token_ids = tokenizer(thinking_content, add_special_tokens=False).input_ids

    # Prepare the closing tokens for RLCR: "...</think>\n\n" + forced answer prompt
    ellipsis_close = "..." + THINK_END_TAG + "\n\n" + FORCED_ANSWER_PROMPT_RLCR
    ellipsis_close_ids = tokenizer(ellipsis_close, add_special_tokens=False).input_ids

    result = []
    for budget in budgets:
        # Cut thinking at budget tokens
        truncated_thinking_ids = thinking_token_ids[:budget]

        # Build the full prompt
        full_prompt_ids = prompt_token_ids + truncated_thinking_ids + ellipsis_close_ids

        # Decode truncated thinking for saving
        truncated_cot_text = tokenizer.decode(truncated_thinking_ids, skip_special_tokens=False)

        result.append((budget, TokensPrompt(prompt_token_ids=full_prompt_ids), truncated_cot_text, len(truncated_thinking_ids)))

    return result


def extract_rlcr_content(text: str) -> Tuple[Optional[str], float, bool]:
    """Extract answer and confidence from RLCR-formatted output.
    
    Returns:
        Tuple of (answer_text, confidence_value, confidence_valid)
    """
    if not HAS_RLCR:
        return None, 0.5, False
    answer_text, confidence, confidence_valid = extract_answer_and_confidence(text)
    return answer_text, confidence, confidence_valid


def evaluate_budget_forced_outputs(
    llm: LLM,
    tokenizer: AutoTokenizer,
    data: List[Dict],
    budgets: List[int],
    n_forced: int,
    forced_max_tokens: int,
    top_p: float = 0.95,
    rlcr: bool = False,
    code_forced_max_tokens: int = 2048,
    gpqa_answer_format: str = "boxed",
) -> Tuple[List[Dict], Dict[int, Dict]]:
    """Generate forced outputs at each budget and evaluate accuracy.

    Args:
        llm: vLLM model instance
        tokenizer: Tokenizer for the model
        data: List of reproduced rollouts with 'input', 'output', 'gts' fields
        budgets: List of budget values (in tokens) to evaluate
        n_forced: Number of forced outputs per budget per rollout
        forced_max_tokens: Maximum tokens for forced answer generation (math tasks)
        rlcr: If True, use RLCR-specific extraction and scoring
        code_forced_max_tokens: Maximum tokens for code forced answer generation

    Returns:
        - List of result records (for saving to jsonl)
        - Dict of budget -> accuracy metrics
    """
    all_results = []
    budget_stats_base = {"correct": 0, "total": 0, "extracted": 0, "avg_tokens": 0, "total_risk": 0.0, "risk_count": 0}

    # Add RLCR-specific stats if in RLCR mode
    if rlcr:
        budget_stats_base.update({
            "total_confidence": 0.0, "confidence_count": 0,
            "total_rlcr_score": 0.0, "rlcr_count": 0,
            "total_brier": 0.0, "brier_count": 0,
        })

    budget_stats = {b: budget_stats_base.copy() for b in budgets}

    # Prepare sampling params for forced answer generation
    math_sampling_params = SamplingParams(
        temperature=0.6,
        top_p=top_p,
        max_tokens=forced_max_tokens,
        n=n_forced,
    )
    code_sampling_params = SamplingParams(
        temperature=0.6,
        top_p=top_p,
        max_tokens=code_forced_max_tokens,
        n=n_forced,
    )

    # Step 1: Build ALL prompts for ALL rollouts and ALL budgets upfront
    logging.info("Building all budget-cut prompts...")
    # Separate lists for math and code prompts (different token limits)
    math_prompts: List[TokensPrompt] = []
    math_prompt_to_meta_idx: List[int] = []
    code_prompts: List[TokensPrompt] = []
    code_prompt_to_meta_idx: List[int] = []
    all_metadata: List[Dict] = []  # All entries (both complete and incomplete)

    code_closing = "..." + THINK_END_TAG + "\n\n" + FORCED_ANSWER_PROMPT_CODE
    gpqa_xml_closing = "..." + THINK_END_TAG + "\n\n" + FORCED_ANSWER_PROMPT_GPQA_XML

    for rollout_idx, item in enumerate(tqdm(data, desc="Preparing prompts")):
        prompt = item["input"]
        output = item["output"]
        ground_truth = item.get("gts", item.get("ground_truth", ""))
        if isinstance(ground_truth, (list,)) and len(ground_truth) == 1:
            ground_truth = ground_truth[0]
        data_source = item.get("data_source", "unknown")
        is_code = data_source in CODE_DATA_SOURCES
        is_gpqa = str(data_source).lower() in GPQA_DATA_SOURCES

        # Extract just the thinking content (remove everything after </think> if present)
        thinking_content = extract_thinking_content(output)

        # Build budget-cut prompts for all budgets
        if rlcr:
            # RLCR mode: always generate, no complete thinking check
            budget_prompts = build_budget_prompts_rlcr(prompt, thinking_content, tokenizer, budgets)
            for budget, token_prompt, truncated_cot, truncated_cot_length in budget_prompts:
                meta_idx = len(all_metadata)
                all_metadata.append({
                    "rollout_idx": rollout_idx,
                    "budget": budget,
                    "truncated_cot": truncated_cot,
                    "truncated_cot_length": truncated_cot_length,
                    "ground_truth": ground_truth,
                    "prompt": prompt,
                    "data_source": data_source,
                    "is_code": is_code,
                    "thinking_complete": False,  # RLCR always generates
                    "original_output": output,
                    "generated_output": None,
                })
                math_prompts.append(token_prompt)
                math_prompt_to_meta_idx.append(meta_idx)
        elif is_code:
            # Code mode: always generate with code-specific closing prompt
            budget_prompts = build_budget_prompts(
                prompt, thinking_content, tokenizer, budgets,
                closing_prompt=code_closing, always_generate=True,
            )
            for budget, token_prompt, truncated_cot, truncated_cot_length, thinking_complete in budget_prompts:
                meta_idx = len(all_metadata)
                all_metadata.append({
                    "rollout_idx": rollout_idx,
                    "budget": budget,
                    "truncated_cot": truncated_cot,
                    "truncated_cot_length": truncated_cot_length,
                    "ground_truth": ground_truth,
                    "prompt": prompt,
                    "data_source": data_source,
                    "is_code": True,
                    "thinking_complete": False,  # code always generates
                    "original_output": output,
                    "generated_output": None,
                })
                code_prompts.append(token_prompt)
                code_prompt_to_meta_idx.append(meta_idx)
        else:
            # Standard math mode: handle complete vs incomplete thinking.
            # GPQA is multiple-choice and format-sensitive, so always force a
            # short boxed letter instead of trusting the natural rollout format.
            closing_prompt = gpqa_xml_closing if is_gpqa and gpqa_answer_format == "xml" else None
            budget_prompts = build_budget_prompts(
                prompt,
                thinking_content,
                tokenizer,
                budgets,
                closing_prompt=closing_prompt,
                always_generate=is_gpqa,
            )

            for budget, token_prompt, truncated_cot, truncated_cot_length, thinking_complete in budget_prompts:
                meta_idx = len(all_metadata)
                all_metadata.append({
                    "rollout_idx": rollout_idx,
                    "budget": budget,
                    "truncated_cot": truncated_cot,
                    "truncated_cot_length": truncated_cot_length,
                    "ground_truth": ground_truth,
                    "prompt": prompt,
                    "data_source": data_source,
                    "is_code": False,
                    "thinking_complete": False if is_gpqa else thinking_complete,
                    "original_output": output,
                    "generated_output": None,  # filled after generation for incomplete traces
                })
                if is_gpqa or not thinking_complete:
                    math_prompts.append(token_prompt)
                    math_prompt_to_meta_idx.append(meta_idx)

    n_complete = sum(1 for m in all_metadata if m["thinking_complete"])
    n_math = len(math_prompts)
    n_code = len(code_prompts)
    logging.info(
        f"Total entries: {len(all_metadata)} ({n_complete} complete, "
        f"{n_math} math need generation, {n_code} code need generation, {n_forced} samples each)"
    )

    # Step 2: Generate outputs only for incomplete traces (math and code batched separately)
    if math_prompts:
        logging.info(f"Generating forced outputs for {n_math} math/RLCR prompts...")
        math_outputs = llm.generate(math_prompts, math_sampling_params)
        for output_obj, meta_idx in zip(math_outputs, math_prompt_to_meta_idx):
            all_metadata[meta_idx]["generated_output"] = output_obj

    if code_prompts:
        logging.info(f"Generating forced outputs for {n_code} code prompts (max_tokens={code_forced_max_tokens})...")
        code_outputs = llm.generate(code_prompts, code_sampling_params)
        for output_obj, meta_idx in zip(code_outputs, code_prompt_to_meta_idx):
            all_metadata[meta_idx]["generated_output"] = output_obj

    # Step 3a: Pre-score code outputs in parallel (fork-based, ~0.01s startup vs ~2s spawn).
    # This builds a cache keyed by (meta_idx, seq_idx) so the main loop below can do
    # an O(1) lookup instead of a blocking subprocess call per sample.
    code_score_cache: Dict[tuple, tuple] = {}
    code_items_to_score = [
        (meta_idx, seq_idx, seq.text, all_metadata[meta_idx]["ground_truth"])
        for meta_idx, meta in enumerate(all_metadata)
        if meta.get("is_code") and meta.get("generated_output") is not None
        for seq_idx, seq in enumerate(meta["generated_output"].outputs)
    ]
    if code_items_to_score:
        logging.info(f"Pre-scoring {len(code_items_to_score)} code outputs in parallel...")

        def _score_code_item(args):
            m_idx, s_idx, text, gt = args
            score, _smeta = compute_code_score(text, gt, continuous=False)
            return (m_idx, s_idx), (float(score), bool(score))

        n_code_workers = min(64, len(code_items_to_score))
        with ThreadPoolExecutor(max_workers=n_code_workers) as executor:
            futures = {executor.submit(_score_code_item, item): item for item in code_items_to_score}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring code"):
                key, val = future.result()
                code_score_cache[key] = val

    # Step 3b: Process results
    logging.info("Processing outputs...")
    for meta_idx, meta in enumerate(all_metadata):
        rollout_idx = meta["rollout_idx"]
        budget = meta["budget"]
        truncated_cot = meta["truncated_cot"]
        truncated_cot_length = meta["truncated_cot_length"]
        ground_truth = meta["ground_truth"]
        original_prompt = meta["prompt"]
        data_source = meta["data_source"]
        is_gpqa = str(data_source).lower() in GPQA_DATA_SOURCES

        forced_answers = []
        forced_accuracies = []
        forced_risks = []
        
        # RLCR-specific lists
        forced_outputs = []
        forced_rlcr_texts = []
        forced_answer_valid = []
        forced_confidence_valid = []
        model_confidences = []
        rlcr_scores = []
        brier_scores = []

        # Check if this is a SARA numerical problem
        is_sara_numerical = False
        if "sara" in data_source.lower():
            try:
                float(str(ground_truth).replace('$', '').replace(',', ''))
                is_sara_numerical = True
            except (ValueError, AttributeError):
                pass

        if rlcr:
            # RLCR mode: always use generated outputs
            output_obj = meta["generated_output"]
            for seq in output_obj.outputs:
                generated_text = seq.text

                # Extract answer and confidence from RLCR format.
                # If the prompt already contains '<answer>', many models emit only
                # '... </answer>' in completion, so inject the missing opening tag.
                rlcr_text = generated_text
                if "<answer>" not in rlcr_text and "</answer>" in rlcr_text:
                    rlcr_text = "<answer>" + rlcr_text

                forced_outputs.append(generated_text)
                forced_rlcr_texts.append(rlcr_text)

                answer_text, confidence, confidence_valid = extract_rlcr_content(rlcr_text)

                # Use RLCR scoring first.
                result = compute_rlcr_score(rlcr_text, ground_truth)

                use_boxed_fallback = (
                    (not result.get("answer_valid", True) or result.get("pred") in {None, ""})
                    and "\\boxed{" in generated_text
                )

                if use_boxed_fallback:
                    boxed_result = compute_score(generated_text, ground_truth, strict_box_verify=True)
                    is_correct = bool(boxed_result["acc"])
                    pred = boxed_result["pred"] if boxed_result["pred"] is not None else ""
                    # No parsed confidence in fallback mode; use neutral confidence.
                    confidence = 0.5
                    confidence_valid = False
                    acc_int = 1 if is_correct else 0
                    brier = (acc_int - confidence) ** 2
                    rlcr_score = acc_int - brier
                else:
                    is_correct = result["acc"]
                    pred = result["pred"]
                    rlcr_score = result["score"]
                    brier = result["brier"]

                if is_gpqa:
                    pred_choice = extract_gpqa_choice(pred)
                    gt_choice = extract_gpqa_choice(ground_truth)
                    if pred_choice is not None and gt_choice is not None:
                        pred = pred_choice
                        is_correct = pred_choice == gt_choice
                        acc_int = 1 if is_correct else 0
                        brier = (acc_int - confidence) ** 2
                        rlcr_score = acc_int - brier

                # SARA numerical branch: correctness uses integer-part match only.
                if is_sara_numerical:
                    sara_acc = compute_sara_integer_part_accuracy(
                        solution_str=generated_text,
                        ground_truth=ground_truth,
                    )
                    if sara_acc is not None:
                        is_correct = bool(sara_acc)

                forced_answers.append(pred if pred else "")
                forced_accuracies.append(1 if is_correct else 0)
                forced_answer_valid.append(pred is not None and pred != "")
                forced_confidence_valid.append(bool(confidence_valid))
                model_confidences.append(confidence)
                rlcr_scores.append(rlcr_score)
                brier_scores.append(brier)
                
                # Compute SARA risk for numerical problems
                if is_sara_numerical:
                    risk = compute_sara_numerical_risk(solution_str=generated_text, ground_truth=ground_truth)
                    forced_risks.append(risk if risk is not None else None)
                    if risk is not None:
                        budget_stats[budget]["total_risk"] += risk
                        budget_stats[budget]["risk_count"] += 1

                # Update budget stats
                budget_stats[budget]["total"] += 1
                if pred is not None and pred != "":
                    budget_stats[budget]["extracted"] += 1
                if is_correct:
                    budget_stats[budget]["correct"] += 1
                budget_stats[budget]["avg_tokens"] += truncated_cot_length
                
                # RLCR-specific stats
                budget_stats[budget]["total_confidence"] += confidence
                budget_stats[budget]["confidence_count"] += 1
                budget_stats[budget]["total_rlcr_score"] += rlcr_score
                budget_stats[budget]["rlcr_count"] += 1
                budget_stats[budget]["total_brier"] += brier
                budget_stats[budget]["brier_count"] += 1
        elif meta.get("is_code", False):
            # Code mode: always use generated outputs and score with prime_code.
            # Scores were pre-computed in parallel above — just look them up.
            output_obj = meta["generated_output"]
            for seq_idx, seq in enumerate(output_obj.outputs):
                generated_text = seq.text
                score_val, is_correct = code_score_cache.get((meta_idx, seq_idx), (0.0, False))
                pred = generated_text

                forced_answers.append(pred)
                forced_accuracies.append(1 if is_correct else 0)

            # Update budget stats for code mode
            for fa, fc in zip(forced_answers, forced_accuracies):
                budget_stats[budget]["total"] += 1
                # For code: extracted = generated text is non-empty
                if fa:
                    budget_stats[budget]["extracted"] += 1
                if fc:
                    budget_stats[budget]["correct"] += 1
                budget_stats[budget]["avg_tokens"] += truncated_cot_length
        else:
            # Standard math mode: handle complete vs incomplete thinking
            if meta["thinking_complete"]:
                # Thinking was complete — use the original output's boxed answer directly
                original_output = meta["original_output"]
                result = compute_score(original_output, ground_truth, strict_box_verify=True)
                is_correct = result["acc"]
                pred = result["pred"]

                if is_sara_numerical:
                    sara_acc = compute_sara_integer_part_accuracy(
                        solution_str=original_output,
                        ground_truth=ground_truth,
                    )
                    if sara_acc is not None:
                        is_correct = bool(sara_acc)

                # Replicate the same answer n_forced times (deterministic — no sampling needed)
                for _ in range(n_forced):
                    forced_answers.append(pred)
                    forced_accuracies.append(1 if is_correct else 0)
            else:
                # Thinking was truncated — use generated forced outputs
                output_obj = meta["generated_output"]
                for seq in output_obj.outputs:
                    generated_text = seq.text
                    if is_gpqa and gpqa_answer_format == "xml":
                        answer_text = extract_xml_answer_content(generated_text)
                        pred = extract_gpqa_choice(answer_text)
                        if pred is None:
                            pred = extract_gpqa_choice(generated_text)
                        gt_choice = extract_gpqa_choice(ground_truth)
                        is_correct = pred is not None and gt_choice is not None and pred == gt_choice
                        full_answer_text = f"<answer>{answer_text}</answer>" if answer_text else generated_text
                    else:
                        # Since we prompted with "The final answer is \boxed{", the model's output
                        # is just the content after \boxed{. We need to prepend \boxed{ so that
                        # compute_score can find and extract the boxed answer properly.
                        full_answer_text = "\\boxed{" + generated_text
                        # Use strict_box_verify=True to extract \boxed{} content (not "Answer:" pattern)
                        result = compute_score(full_answer_text, ground_truth, strict_box_verify=True)
                        is_correct = result["acc"]
                        pred = result["pred"]

                        if is_gpqa:
                            pred_choice = extract_gpqa_choice(pred)
                            gt_choice = extract_gpqa_choice(ground_truth)
                            if pred_choice is not None and gt_choice is not None:
                                pred = pred_choice
                                is_correct = pred_choice == gt_choice

                    # SARA numerical branch: correctness uses integer-part match only.
                    if is_sara_numerical:
                        sara_acc = compute_sara_integer_part_accuracy(
                            solution_str=full_answer_text,
                            ground_truth=ground_truth,
                        )
                        if sara_acc is not None:
                            is_correct = bool(sara_acc)

                    forced_answers.append(pred)
                    forced_accuracies.append(1 if is_correct else 0)

                    # Compute SARA risk for numerical problems
                    if is_sara_numerical:
                        risk = compute_sara_numerical_risk(solution_str=full_answer_text, ground_truth=ground_truth)
                        forced_risks.append(risk if risk is not None else None)
                        if risk is not None:
                            budget_stats[budget]["total_risk"] += risk
                            budget_stats[budget]["risk_count"] += 1

            # Compute SARA risk for complete traces
            if meta["thinking_complete"] and is_sara_numerical:
                original_output = meta["original_output"]
                risk = compute_sara_numerical_risk(solution_str=original_output, ground_truth=ground_truth)
                for _ in range(n_forced):
                    forced_risks.append(risk if risk is not None else None)
                    if risk is not None:
                        budget_stats[budget]["total_risk"] += risk
                        budget_stats[budget]["risk_count"] += 1

            # Update budget stats for non-RLCR mode
            if not rlcr:
                for fa, fc in zip(forced_answers, forced_accuracies):
                    budget_stats[budget]["total"] += 1
                    if fa is not None:
                        budget_stats[budget]["extracted"] += 1
                    if fc:
                        budget_stats[budget]["correct"] += 1
                    budget_stats[budget]["avg_tokens"] += truncated_cot_length

        # Save the result record
        result_record = {
            "rollout_idx": rollout_idx,
            "prompt": original_prompt,
            "budget": budget,
            "truncated_cot": truncated_cot,
            "truncated_cot_length": truncated_cot_length,
            "ground_truth": ground_truth,
            "forced_answers": forced_answers,
            "forced_accuracies": forced_accuracies,
            "mean_accuracy": sum(forced_accuracies) / len(forced_accuracies) if forced_accuracies else 0,
            "data_source": data_source,
            "gpqa_answer_format": gpqa_answer_format if str(data_source).lower() in GPQA_DATA_SOURCES else None,
        }
        
        # Add RLCR-specific fields
        if rlcr:
            result_record.update({
                "forced_outputs": forced_outputs,
                "forced_rlcr_texts": forced_rlcr_texts,
                "forced_answer_valid": forced_answer_valid,
                "forced_confidence_valid": forced_confidence_valid,
                "model_confidences": model_confidences,
                "mean_confidence": sum(model_confidences) / len(model_confidences) if model_confidences else 0.5,
                "rlcr_scores": rlcr_scores,
                "mean_rlcr_score": sum(rlcr_scores) / len(rlcr_scores) if rlcr_scores else 0.0,
                "brier_scores": brier_scores,
                "mean_brier": sum(brier_scores) / len(brier_scores) if brier_scores else 0.5,
            })
        
        # Add risk metrics for SARA numerical problems
        if is_sara_numerical and forced_risks:
            result_record["forced_risks"] = forced_risks
            valid_risks = [r for r in forced_risks if r is not None]
            if valid_risks:
                result_record["mean_risk"] = sum(valid_risks) / len(valid_risks)
        
        all_results.append(result_record)

    return all_results, budget_stats


def print_budget_table(budget_stats: Dict[int, Dict], budgets: List[int], rlcr: bool = False) -> None:
    """Print a table of performance at each budget."""
    # Check if any budget has risk data
    has_risk = any(stats.get("risk_count", 0) > 0 for stats in budget_stats.values())
    
    if rlcr:
        # RLCR-specific table
        print("\n" + "=" * 120)
        print("BUDGET-WISE FORCED OUTPUT PERFORMANCE (RLCR)")
        print("=" * 120)
        if has_risk:
            print(f"{'Budget':<10} {'Accuracy':<12} {'Mean Conf':<12} {'Mean RLCR':<12} {'Mean Brier':<12} {'Extract':<10} {'Avg Risk':<12}")
        else:
            print(f"{'Budget':<10} {'Accuracy':<12} {'Mean Conf':<12} {'Mean RLCR':<12} {'Mean Brier':<12} {'Extract':<10}")
        print("-" * 120 if has_risk else "-" * 108)

        for budget in sorted(budgets):
            stats = budget_stats[budget]
            total = stats["total"]
            correct = stats["correct"]
            extracted = stats["extracted"]
            avg_tokens = stats["avg_tokens"] / total if total > 0 else 0.0
            accuracy = correct / total if total > 0 else 0.0
            extraction_rate = extracted / total if total > 0 else 0.0
            
            # RLCR-specific metrics
            mean_conf = stats["total_confidence"] / stats["confidence_count"] if stats["confidence_count"] > 0 else 0.5
            mean_rlcr = stats["total_rlcr_score"] / stats["rlcr_count"] if stats["rlcr_count"] > 0 else 0.0
            mean_brier = stats["total_brier"] / stats["brier_count"] if stats["brier_count"] > 0 else 0.5
            
            if has_risk:
                avg_risk = stats["total_risk"] / stats["risk_count"] if stats["risk_count"] > 0 else 0.0
                print(f"{budget:<10} {accuracy:.4f}{'':<6} {mean_conf:.4f}{'':<6} {mean_rlcr:.4f}{'':<6} {mean_brier:.4f}{'':<6} {extraction_rate:.4f}{'':<4} {avg_risk:<12.2f}")
            else:
                print(f"{budget:<10} {accuracy:.4f}{'':<6} {mean_conf:.4f}{'':<6} {mean_rlcr:.4f}{'':<6} {mean_brier:.4f}{'':<6} {extraction_rate:.4f}")

        print("=" * 120 if has_risk else "=" * 108)
        
        # Print overall stats
        total_all = sum(s["total"] for s in budget_stats.values())
        correct_all = sum(s["correct"] for s in budget_stats.values())
        overall_acc = correct_all / total_all if total_all > 0 else 0.0
        
        total_conf_all = sum(s["total_confidence"] for s in budget_stats.values())
        conf_count_all = sum(s["confidence_count"] for s in budget_stats.values())
        overall_conf = total_conf_all / conf_count_all if conf_count_all > 0 else 0.5
        
        total_rlcr_all = sum(s["total_rlcr_score"] for s in budget_stats.values())
        rlcr_count_all = sum(s["rlcr_count"] for s in budget_stats.values())
        overall_rlcr = total_rlcr_all / rlcr_count_all if rlcr_count_all > 0 else 0.0
        
        total_brier_all = sum(s["total_brier"] for s in budget_stats.values())
        brier_count_all = sum(s["brier_count"] for s in budget_stats.values())
        overall_brier = total_brier_all / brier_count_all if brier_count_all > 0 else 0.5
        
        print(f"Overall Accuracy: {overall_acc:.4f} ({correct_all}/{total_all})")
        print(f"Overall Mean Confidence: {overall_conf:.4f}")
        print(f"Overall Mean RLCR Score: {overall_rlcr:.4f}")
        print(f"Overall Mean Brier: {overall_brier:.4f}")
        
        if has_risk:
            total_risk_all = sum(s["total_risk"] for s in budget_stats.values())
            risk_count_all = sum(s["risk_count"] for s in budget_stats.values())
            overall_risk = total_risk_all / risk_count_all if risk_count_all > 0 else 0.0
            print(f"Overall Avg Risk: {overall_risk:.2f} (computed from {risk_count_all} samples)")
        
        print("=" * 120 if has_risk else "=" * 108)
    else:
        # Standard table
        print("\n" + "=" * 90 if has_risk else "\n" + "=" * 70)
        print("BUDGET-WISE FORCED OUTPUT PERFORMANCE")
        print("=" * 90 if has_risk else "=" * 70)
        
        if has_risk:
            print(f"{'Budget':<10} {'Accuracy':<15} {'Extracted':<15} {'Total':<10} {'Avg Tokens':<12} {'Avg Risk':<12}")
        else:
            print(f"{'Budget':<10} {'Accuracy':<15} {'Extracted':<15} {'Total':<10} {'Avg Tokens':<10}")
        print("-" * 90 if has_risk else "-" * 70)

        for budget in sorted(budgets):
            stats = budget_stats[budget]
            total = stats["total"]
            correct = stats["correct"]
            extracted = stats["extracted"]
            avg_tokens = stats["avg_tokens"] / total if total > 0 else 0.0
            accuracy = correct / total if total > 0 else 0.0
            extraction_rate = extracted / total if total > 0 else 0.0
            
            if has_risk:
                avg_risk = stats["total_risk"] / stats["risk_count"] if stats["risk_count"] > 0 else 0.0
                print(f"{budget:<10} {accuracy:.4f} ({correct}/{total}){'':<3} {extraction_rate:.4f}{'':<6} {total:<10} {avg_tokens:<12.1f} {avg_risk:<12.2f}")
            else:
                print(f"{budget:<10} {accuracy:.4f} ({correct}/{total}){'':<3} {extraction_rate:.4f}{'':<6} {total:<10} {avg_tokens:<10}")

        print("=" * 90 if has_risk else "=" * 70)

        # Print overall stats
        total_all = sum(s["total"] for s in budget_stats.values())
        correct_all = sum(s["correct"] for s in budget_stats.values())
        overall_acc = correct_all / total_all if total_all > 0 else 0.0
        print(f"Overall Accuracy: {overall_acc:.4f} ({correct_all}/{total_all})")
        
        if has_risk:
            total_risk_all = sum(s["total_risk"] for s in budget_stats.values())
            risk_count_all = sum(s["risk_count"] for s in budget_stats.values())
            overall_risk = total_risk_all / risk_count_all if risk_count_all > 0 else 0.0
            print(f"Overall Avg Risk: {overall_risk:.2f} (computed from {risk_count_all} samples)")
        
        print("=" * 90 if has_risk else "=" * 70)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Budget-wise forced output evaluation"
    )
    parser.add_argument("--input_file",type=str,required=True,help="Path to reproduced jsonl file",)
    parser.add_argument("--model_path",type=str,required=True,help="Path to the model checkpoint",)
    parser.add_argument("--output_file",type=str,default="budget_eval_results.jsonl",help="Output jsonl file for detailed results",)
    parser.add_argument("--n_forced",type=int,default=8,help="Number of forced outputs per budget per rollout",)
    parser.add_argument(
        "--budgets",type=str,default=",".join(map(str, DEFAULT_BUDGETS)),help="Comma-separated list of budget values (in tokens)",
    )
    parser.add_argument("--forced_max_tokens",type=int,default=64,help="Maximum tokens for forced answer generation (math tasks)",)
    parser.add_argument("--code_forced_max_tokens",type=int,default=2048,help="Maximum tokens for forced code generation (code tasks such as livecodebench)",)
    parser.add_argument("--top_p",type=float,default=0.95,help="Top-p sampling parameter for forced answer generation",)
    parser.add_argument("--gpqa_answer_format", type=str, default="boxed", choices=["boxed", "xml"], help="Forced-answer format for GPQA tasks")
    parser.add_argument("--tensor_parallel_size",type=int,default=2,help="Number of GPUs for tensor parallelism",)
    parser.add_argument("--gpu_memory_utilization",type=float,default=0.9,help="GPU memory utilization for vLLM",)
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument("--log_file",type=str,default="budget_eval_results.log",help="Path to log file",)
    parser.add_argument("--rlcr",action="store_true",help="RLCR mode: use RLCR-specific extraction and scoring",)

    return parser.parse_args()



def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.log_level)

    # Parse budgets
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    logging.info("=" * 70)
    logging.info("BUDGET-WISE FORCED OUTPUT EVALUATION")
    logging.info("=" * 70)
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Output file: {args.output_file}")
    logging.info(f"N forced outputs: {args.n_forced}")
    logging.info(f"Budgets: {budgets}")
    logging.info(f"Forced max tokens: {args.forced_max_tokens}")
    logging.info(f"Forced top_p: {args.top_p}")
    if args.rlcr:
        logging.info("Mode: RLCR (uses RLCR-specific extraction and scoring)")
    logging.info(f"GPQA answer format: {args.gpqa_answer_format}")
    logging.info("=" * 70)

    # Load reproduced data
    data = load_reproduced_data(args.input_file)

    # Load tokenizer
    logging.info(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # Initialize vLLM
    logging.info(f"Initializing vLLM model: {args.model_path}...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    # Evaluate
    logging.info("Starting budget-wise forced output evaluation...")
    results, budget_stats = evaluate_budget_forced_outputs(
        llm=llm,
        tokenizer=tokenizer,
        data=data,
        budgets=budgets,
        n_forced=args.n_forced,
        forced_max_tokens=args.forced_max_tokens,
        top_p=args.top_p,
        rlcr=args.rlcr,
        code_forced_max_tokens=args.code_forced_max_tokens,
        gpqa_answer_format=args.gpqa_answer_format,
    )

    # Save results to jsonl
    logging.info(f"Saving {len(results)} result records to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    # Print performance table
    print_budget_table(budget_stats, budgets, rlcr=args.rlcr)

    # Also save summary statistics
    summary_file = args.output_file.replace(".jsonl", "_summary.json")
    summary = {
        "budgets": budgets,
        "n_forced": args.n_forced,
        "input_file": args.input_file,
        "model_path": args.model_path,
        "budget_stats": {
            str(b): {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
                "extraction_rate": stats["extracted"] / stats["total"] if stats["total"] > 0 else 0.0,
                "correct": stats["correct"],
                "extracted": stats["extracted"],
                "total": stats["total"],
                "avg_tokens": stats["avg_tokens"] / stats["total"] if stats["total"] > 0 else 0.0,
                "avg_risk": stats["total_risk"] / stats["risk_count"] if stats["risk_count"] > 0 else None,
                "risk_samples": stats["risk_count"],
            }
            for b, stats in budget_stats.items()
        },
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary saved to {summary_file}")

    logging.info("Evaluation completed!")


if __name__ == "__main__":
    main()
