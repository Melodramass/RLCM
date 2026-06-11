#!/usr/bin/env python3
"""
Budget-wise forced output evaluation script for RLCR-formatted models.

This script is adapted from 2_budget_eval.py for the e27-baseline_RLCR model
which uses <answer>, <analysis>, and <confidence> tags instead of \boxed{}.

Key differences from standard 2_budget_eval.py:
- Uses RLCR-style forced answer prompt that asks for answer, analysis, and confidence tags
- Uses compute_rlcr_score() to extract answer AND confidence
- Saves model_confidence in results for downstream Steps 5-7

Usage:
    python 2_budget_eval_rlcr.py \
        --input_file rollouts_rlcr.jsonl \
        --model_path /path/to/model \
        --output_file budget_eval_rlcr.jsonl \
        --n_forced 16 \
        --budgets 500,1000,1500,2000
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Optional, Tuple
import multiprocessing
from tqdm import tqdm
from vllm import LLM, SamplingParams, TokensPrompt
from transformers import AutoTokenizer

# Ensure local eval utilities are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from math_dapo import compute_score
from metric_utils import compute_sara_integer_part_accuracy, compute_sara_numerical_risk

# Import RLCR-specific scoring functions
from verl.utils.reward_score.rlcr import (
    extract_answer_and_confidence,
    compute_score as compute_rlcr_score,
)

# Default settings
DEFAULT_BUDGETS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]

# RLCR-specific: The model uses <think></ Thinking_END> for thinking, then outputs <answer>, <analysis>, <confidence>
THINK_END_TAG = "</think>"

# RLCR-specific forced answer prompt
# Instead of \boxed{, we use <answer> tag
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

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Monkey-patch get_context to always return spawn context
_original_get_context = multiprocessing.get_context
def _patched_get_context(method=None):
    if method is None or method == "fork":
        method = "spawn"
    return _original_get_context(method)
multiprocessing.get_context = _patched_get_context


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%%Y-%%m-%%d %%H:%%M:%%S",
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


def build_budget_prompts_rlcr(
    prompt: str,
    thinking_content: str,
    tokenizer: AutoTokenizer,
    budgets: List[int],
) -> List[Tuple[int, TokensPrompt, str, int]]:
    """Build prompts with budget-cut CoT and RLCR forced answer prompt.

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


def extract_rlcr_content(text: str) -> Tuple[Optional[str], Optional[float], bool]:
    """Extract answer and confidence from RLCR-formatted output.

    Returns:
        Tuple of (answer_text, confidence_value, confidence_valid)
    """
    answer_text, confidence, confidence_valid = extract_answer_and_confidence(text)
    return answer_text, confidence, confidence_valid


def evaluate_budget_forced_outputs_rlcr(
    llm: LLM,
    tokenizer: AutoTokenizer,
    data: List[Dict],
    budgets: List[int],
    n_forced: int,
    forced_max_tokens: int,
) -> Tuple[List[Dict], Dict[int, Dict]]:
    """Generate forced outputs at each budget and evaluate using RLCR scoring.

    This version extracts both answer AND confidence from the model's output,
    then computes RLCR scores.

    Args:
        llm: vLLM model instance
        tokenizer: Tokenizer for the model
        data: List of reproduced rollouts with 'input', 'output', 'gts' fields
        budgets: List of budget values (in tokens) to evaluate
        n_forced: Number of forced outputs per budget per rollout
        forced_max_tokens: Maximum tokens for forced answer generation

    Returns:
        - List of result records (for saving to jsonl)
        - Dict of budget -> accuracy metrics
    """
    all_results = []
    budget_stats = {b: {
        "correct": 0, "total": 0, "extracted": 0, 
        "avg_tokens": 0, "total_risk": 0.0, "risk_count": 0,
        # RLCR-specific stats
        "total_confidence": 0.0, "confidence_count": 0,
        "total_rlcr_score": 0.0, "rlcr_count": 0,
        "total_brier": 0.0, "brier_count": 0,
    } for b in budgets}

    # Prepare sampling params for forced answer generation
    # Note: For RLCR, we need more tokens since model outputs <answer><analysis><confidence>
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=forced_max_tokens,
        n=n_forced,
    )

    # Step 1: Build ALL prompts for ALL rollouts and ALL budgets upfront
    logging.info("Building all budget-cut prompts (RLCR format)...")
    all_prompts = []  # List of TokensPrompt
    prompt_metadata = []  # List of (rollout_idx, budget, truncated_cot, ground_truth, original_prompt)

    for rollout_idx, item in enumerate(tqdm(data, desc="Preparing prompts")):
        prompt = item["input"]
        output = item["output"]
        ground_truth = item.get("gts", item.get("ground_truth", ""))
        data_source = item.get("data_source", "unknown")

        # Extract just the thinking content (remove everything after </think> if present)
        thinking_content = extract_thinking_content(output)

        # Build budget-cut prompts for all budgets using RLCR format
        budget_prompts = build_budget_prompts_rlcr(prompt, thinking_content, tokenizer, budgets)

        for budget, token_prompt, truncated_cot, truncated_cot_length in budget_prompts:
            all_prompts.append(token_prompt)
            prompt_metadata.append((rollout_idx, budget, truncated_cot, truncated_cot_length, ground_truth, prompt, data_source))

    logging.info(f"Total prompts to generate: {len(all_prompts)} (will generate {n_forced} samples each)")

    # Step 2: Generate ALL outputs in one batched call
    logging.info("Generating all forced outputs in one batch...")
    all_outputs = llm.generate(all_prompts, sampling_params)

    # Step 3: Process results and map back to rollout/budget
    logging.info("Processing outputs (RLCR scoring)...")
    for output, metadata in zip(all_outputs, prompt_metadata):
        rollout_idx, budget, truncated_cot, truncated_cot_length, ground_truth, original_prompt, data_source = metadata

        forced_answers = []
        forced_accuracies = []
        forced_risks = []
        
        # NEW: RLCR-specific: store model's self-reported confidence
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

        for seq in output.outputs:
            generated_text = seq.text

            # If the prompt already contains '<answer>', the completion may
            # start with only the answer text and closing tag.
            rlcr_text = generated_text
            if "<answer>" not in rlcr_text and "</answer>" in rlcr_text:
                rlcr_text = "<answer>" + rlcr_text

            forced_outputs.append(generated_text)
            forced_rlcr_texts.append(rlcr_text)
            
            # RLCR extraction: extract both answer AND confidence
            answer_text, confidence, confidence_valid = extract_rlcr_content(rlcr_text)
            
            # Use RLCR scoring function
            # Pass the RLCR text directly - it will extract answer/confidence internally
            result = compute_rlcr_score(rlcr_text, ground_truth)
            
            is_correct = result["acc"]
            pred = result["pred"]
            rlcr_score = result["score"]
            brier = result["brier"]

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
                # Track risk in budget stats
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

        # Save the result record with RLCR-specific fields
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
            
            # RLCR-specific fields (for Steps 5-7 - matching probe output format!)
            "forced_outputs": forced_outputs,
            "forced_rlcr_texts": forced_rlcr_texts,
            "forced_answer_valid": forced_answer_valid,
            "forced_confidence_valid": forced_confidence_valid,
            "model_confidences": model_confidences,
            "mean_confidence": sum(model_confidences) / len(model_confidences) if model_confidences else 0.5,  # Same field name as probe!
            "rlcr_scores": rlcr_scores,
            "mean_rlcr_score": sum(rlcr_scores) / len(rlcr_scores) if rlcr_scores else 0.0,
            "brier_scores": brier_scores,
            "mean_brier": sum(brier_scores) / len(brier_scores) if brier_scores else 0.5,
        }
        
        # Add risk metrics for SARA numerical problems
        if is_sara_numerical and forced_risks:
            result_record["forced_risks"] = forced_risks
            valid_risks = [r for r in forced_risks if r is not None]
            if valid_risks:
                result_record["mean_risk"] = sum(valid_risks) / len(valid_risks)
        
        all_results.append(result_record)

    return all_results, budget_stats


def print_budget_table_rlcr(budget_stats: Dict[int, Dict], budgets: List[int]) -> None:
    """Print a table of performance at each budget with RLCR metrics."""
    # Check if any budget has risk data
    has_risk = any(stats.get("risk_count", 0) > 0 for stats in budget_stats.values())
    
    print("\n" + "=" * 100)
    print("BUDGET-WISE FORCED OUTPUT PERFORMANCE (RLCR)")
    print("=" * 100)
    
    # Extended header for RLCR
    header = f"{'Budget':<10} {'Accuracy':<15} {'Mean Conf':<12} {'Mean RLCR':<12} {'Mean Brier':<12} {'Extract':<10}"
    if has_risk:
        header += f" {'Avg Risk':<12}"
    print(header)
    print("-" * 100)

    for budget in sorted(budgets):
        stats = budget_stats[budget]
        total = stats["total"]
        correct = stats["correct"]
        extracted = stats["extracted"]
        avg_tokens = stats["avg_tokens"] / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        extraction_rate = extracted / total if total > 0 else 0.0
        
        # RLCR-specific - use mean_confidence to match probe format!
        mean_conf = stats["total_confidence"] / stats["confidence_count"] if stats["confidence_count"] > 0 else 0.0
        mean_rlcr = stats["total_rlcr_score"] / stats["rlcr_count"] if stats["rlcr_count"] > 0 else 0.0
        mean_brier = stats["total_brier"] / stats["brier_count"] if stats["brier_count"] > 0 else 0.0
        
        row = f"{budget:<10} {accuracy:.4f} ({correct}/{total}){'':<3} {mean_conf:.4f}      {mean_rlcr:.4f}      {mean_brier:.4f}      {extraction_rate:.4f}"
        
        if has_risk:
            avg_risk = stats["total_risk"] / stats["risk_count"] if stats["risk_count"] > 0 else 0.0
            row += f" {avg_risk:.2f}"
        
        print(row)

    print("=" * 100)

    # Print overall stats
    total_all = sum(s["total"] for s in budget_stats.values())
    correct_all = sum(s["correct"] for s in budget_stats.values())
    overall_acc = correct_all / total_all if total_all > 0 else 0.0
    print(f"Overall Accuracy: {overall_acc:.4f} ({correct_all}/{total_all})")
    
    # Overall RLCR stats
    total_conf = sum(s["total_confidence"] for s in budget_stats.values())
    conf_count = sum(s["confidence_count"] for s in budget_stats.values())
    overall_conf = total_conf / conf_count if conf_count > 0 else 0.0
    
    total_rlcr = sum(s["total_rlcr_score"] for s in budget_stats.values())
    rlcr_count = sum(s["rlcr_count"] for s in budget_stats.values())
    overall_rlcr = total_rlcr / rlcr_count if rlcr_count > 0 else 0.0
    
    total_brier = sum(s["total_brier"] for s in budget_stats.values())
    brier_count = sum(s["brier_count"] for s in budget_stats.values())
    overall_brier = total_brier / brier_count if brier_count > 0 else 0.0
    
    print(f"Overall Mean Confidence: {overall_conf:.4f}")
    print(f"Overall Mean RLCR Score: {overall_rlcr:.4f}")
    print(f"Overall Mean Brier: {overall_brier:.4f}")
    
    if has_risk:
        total_risk_all = sum(s["total_risk"] for s in budget_stats.values())
        risk_count_all = sum(s["risk_count"] for s in budget_stats.values())
        overall_risk = total_risk_all / risk_count_all if risk_count_all > 0 else 0.0
        print(f"Overall Avg Risk: {overall_risk:.2f} (computed from {risk_count_all} samples)")
    
    print("=" * 100)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Budget-wise forced output evaluation for RLCR models"
    )
    parser.add_argument("--input_file",type=str,required=True,help="Path to reproduced jsonl file",)
    parser.add_argument("--model_path",type=str,required=True,help="Path to the model checkpoint",)
    parser.add_argument("--output_file",type=str,default="budget_eval_results_rlcr.jsonl",help="Output jsonl file for detailed results",)
    parser.add_argument("--n_forced",type=int,default=8,help="Number of forced outputs per budget per rollout",)
    parser.add_argument(
        "--budgets",type=str,default=",".join(map(str, DEFAULT_BUDGETS)),help="Comma-separated list of budget values (in tokens)",
    )
    parser.add_argument("--forced_max_tokens",type=int,default=128,help="Maximum tokens for forced answer generation (higher for RLCR)",)
    parser.add_argument("--tensor_parallel_size",type=int,default=2,help="Number of GPUs for tensor parallelism",)
    parser.add_argument("--gpu_memory_utilization",type=float,default=0.9,help="GPU memory utilization for vLLM",)
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument("--log_file",type=str,default="budget_eval_rlcr_results.log",help="Path to log file",)

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.log_level)

    # Parse budgets
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    logging.info("=" * 70)
    logging.info("BUDGET-WISE FORCED OUTPUT EVALUATION (RLCR)")
    logging.info("=" * 70)
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Model path: {args.model_path}")
    logging.info(f"Output file: {args.output_file}")
    logging.info(f"N forced outputs: {args.n_forced}")
    logging.info(f"Budgets: {budgets}")
    logging.info(f"Forced max tokens: {args.forced_max_tokens}")
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

    # Evaluate with RLCR scoring
    logging.info("Starting budget-wise forced output evaluation (RLCR)...")
    results, budget_stats = evaluate_budget_forced_outputs_rlcr(
        llm=llm,
        tokenizer=tokenizer,
        data=data,
        budgets=budgets,
        n_forced=args.n_forced,
        forced_max_tokens=args.forced_max_tokens,
    )

    # Save results to jsonl
    logging.info(f"Saving {len(results)} result records to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print performance table with RLCR metrics
    print_budget_table_rlcr(budget_stats, budgets)

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
                # RLCR-specific (same field name as probe!)
                "mean_confidence": stats["total_confidence"] / stats["confidence_count"] if stats["confidence_count"] > 0 else 0.0,
                "mean_rlcr_score": stats["total_rlcr_score"] / stats["rlcr_count"] if stats["rlcr_count"] > 0 else 0.0,
                "mean_brier": stats["total_brier"] / stats["brier_count"] if stats["brier_count"] > 0 else 0.0,
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
