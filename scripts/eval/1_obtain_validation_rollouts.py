#!/usr/bin/env python3
"""
Standalone script to reproduce validation rollout generation results.

This script reproduces the validation rollouts generated during training by:
1. Loading the validation datasets (aime_filtered.parquet, amc_filtered.parquet)
2. Applying the same chat template as verl-new training
3. Generating outputs with the same vLLM configuration
4. Computing rewards using the same reward function
5. Saving outputs in the same JSONL format

Usage:
    python reproduce_validation_rollouts.py \
        --model_path /path/to/checkpoint \
        --output_path output.jsonl \
        --n_samples 8 \
        --temperature 0.6

The default settings match the training config:
- temperature: 0.6
- top_p: 1.0
- do_sample: True  
- n: 8 (samples per problem)
- max_tokens: 8000
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Ensure local eval utilities are importable regardless of launch directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

REPO_ROOT = Path(SCRIPT_DIR).resolve().parents[1]


from math_dapo import compute_score

# Import RLCR scoring for RLCR models
from verl.utils.reward_score.rlcr import compute_score as compute_rlcr_score
from verl.utils.reward_score import default_compute_score

from metric_utils import compute_sara_integer_part_accuracy, compute_sara_numerical_risk

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
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

GPQA_DATA_SOURCES = {"gpqa", "gpqa_rlcr"}
LETS_THINK_MARKER = "Let's think step by step and output the final answer within \\boxed{}."


def extract_gpqa_choice(text):
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


def apply_gpqa_choice_scoring(score_result, ground_truth):
    pred_choice = extract_gpqa_choice(score_result.get("pred"))
    gt_choice = extract_gpqa_choice(ground_truth)
    if pred_choice is None or gt_choice is None:
        return score_result

    acc = 1 if pred_choice == gt_choice else 0
    confidence = float(score_result.get("confidence", 0.5))
    brier = (acc - confidence) ** 2
    score_result = dict(score_result)
    score_result.update({
        "score": acc - brier,
        "acc": bool(acc),
        "pred": pred_choice,
        "brier": brier,
    })
    return score_result


def normalize_lets_think_separator(messages, separator: str):
    """Return messages with a normalized separator before the standard math cue."""
    if separator == "keep":
        return messages

    replacement = {
        "space": " ",
        "double_newline": "\n\n",
    }[separator]

    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue

        message = dict(message)
        content = message.get("content")
        if isinstance(content, str):
            marker_idx = content.find(LETS_THINK_MARKER)
            if marker_idx > 0:
                message["content"] = content[:marker_idx].rstrip() + replacement + content[marker_idx:]
        normalized.append(message)

    return normalized


def normalize_data_prompt_separator(data: list[dict], separator: str) -> list[dict]:
    if separator == "keep":
        return data

    normalized_data = []
    for item in data:
        item = dict(item)
        item["messages"] = normalize_lets_think_separator(item["messages"], separator)
        normalized_data.append(item)
    return normalized_data

def load_validation_data(data_files: list[str]) -> list[dict]:
    """Load validation datasets and prepare prompts."""
    all_data = []

    def _resolve_data_file(path_str: str) -> str:
        raw = Path(path_str)
        candidates: list[Path] = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(REPO_ROOT / raw)
            candidates.append(raw)

        # Common aliases in this repo.
        alias_name = {
            "aime24.parquet": "aime24_mid.parquet",
            "amc23.parquet": "amc.parquet",
        }.get(raw.name)
        if alias_name:
            if raw.is_absolute():
                candidates.append(raw.with_name(alias_name))
            else:
                candidates.append(REPO_ROOT / raw.with_name(alias_name))

        # Generic fallback: foo.parquet -> foo_mid.parquet
        if raw.suffix == ".parquet" and not raw.stem.endswith("_mid"):
            mid_name = raw.with_name(f"{raw.stem}_mid{raw.suffix}")
            if raw.is_absolute():
                candidates.append(mid_name)
            else:
                candidates.append(REPO_ROOT / mid_name)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(f"Validation data file not found: {path_str}")
    
    for data_file in data_files:
        resolved_data_file = _resolve_data_file(data_file)
        if resolved_data_file != data_file:
            print(f"Resolved data file: {data_file} -> {resolved_data_file}")
        df = pd.read_parquet(resolved_data_file)
        
        for idx, row in df.iterrows():
            # Extract prompt from the format used in training
            prompt_messages = row["prompt"]
            if isinstance(prompt_messages, np.ndarray):
                prompt_messages = prompt_messages.tolist()
            if isinstance(prompt_messages, str):
                prompt_messages = json.loads(prompt_messages)
            
            # Get ground truth from reward_model field
            reward_model_info = row["reward_model"]
            if isinstance(reward_model_info, str):
                reward_model_info = json.loads(reward_model_info)
            ground_truth = reward_model_info.get("ground_truth", None)
            # Unwrap single-element list ground truths from parquet
            if isinstance(ground_truth, (list, np.ndarray)) and len(ground_truth) == 1:
                ground_truth = ground_truth[0]

            extra_info = row.get("extra_info", {})
            if isinstance(extra_info, str):
                extra_info = json.loads(extra_info)
            
            all_data.append({
                "messages": prompt_messages,
                "ground_truth": ground_truth,
                "data_source": row.get("data_source", "unknown"),
                "extra_info": extra_info,
            })
    
    return all_data


def apply_chat_template(tokenizer, data: list[dict]) -> list[str]:
    """Apply chat template to messages."""
    prompts = []
    for item in data:
        # Apply chat template with add_generation_prompt=True
        prompt = tokenizer.apply_chat_template(
            item["messages"],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(prompt)
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Reproduce validation rollouts")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--data_files", type=str, nargs="+",
                        # default=["deepscaler/data/aime_filtered.parquet",
                        #          "deepscaler/data/amc_filtered.parquet"],
                        default=["deepscaler/data/aime_filtered.parquet",],
                        help="Validation data files")
    parser.add_argument("--output_path", type=str, default="validation_rollouts.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--n_samples", type=int, default=16,
                        help="Number of samples per problem")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=8000,
                        help="Maximum tokens to generate")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization")
    parser.add_argument("--step", type=int, default=0,
                        help="Training step (for output metadata)")
    parser.add_argument("--use_rlcr", action="store_true",
                        help="Use RLCR scoring instead of standard boxed scoring")
    parser.add_argument("--prompt_lets_separator", choices=["keep", "space", "double_newline"],
                        default="keep",
                        help="Normalize whitespace immediately before the standard \"Let's think\" cue")
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load validation data
    print(f"Loading validation data from {args.data_files}")
    data = load_validation_data(args.data_files)
    data = normalize_data_prompt_separator(data, args.prompt_lets_separator)
    print(f"Loaded {len(data)} problems")
    
    # Apply chat template
    print("Applying chat template...")
    prompts = apply_chat_template(tokenizer, data)
    
    # Repeat prompts for n_samples
    repeated_prompts = []
    repeated_data = []
    for i, (prompt, item) in enumerate(zip(prompts, data)):
        for _ in range(args.n_samples):
            repeated_prompts.append(prompt)
            repeated_data.append(item)
    
    print(f"Total samples to generate: {len(repeated_prompts)}")
    
    # Initialize vLLM
    print(f"Initializing vLLM with {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=1,  # Already repeated prompts
    )
    
    print(f"Sampling params: temp={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")
    
    # Generate outputs
    print("Generating outputs...")
    outputs = llm.generate(repeated_prompts, sampling_params)
    
    # ---------------------------------------------------------------------------
    # Score all outputs – code tasks run in parallel (each spawns a subprocess
    # internally, so threading is safe and gives ~32× throughput vs sequential).
    # ---------------------------------------------------------------------------
    CODE_DATA_SOURCES = {"codecontests", "apps", "codeforces", "taco", "livecodebench"}
    # Check whether any output is a code task so we can choose the worker count.
    has_code = any(
        str(repeated_data[i].get("data_source", "unknown")) in CODE_DATA_SOURCES
        for i in range(len(outputs))
    )
    n_workers = 64 if has_code else 8

    def _score_one(idx):
        output = outputs[idx]
        item = repeated_data[idx]
        prompt = repeated_prompts[idx]
        generated_text = output.outputs[0].text
        data_source = str(item.get("data_source", "unknown"))
        raw_gt = item["ground_truth"]
        ground_truth = str(raw_gt) if isinstance(raw_gt, (int, float)) else raw_gt
        num_tokens = len(output.outputs[0].token_ids)

        if data_source in CODE_DATA_SOURCES:
            code_score = default_compute_score(
                data_source=data_source,
                solution_str=generated_text,
                ground_truth=ground_truth,
                extra_info=item.get("extra_info", {}),
            )
            if isinstance(code_score, dict):
                score_val = float(code_score.get("score", 0.0))
                acc_val = bool(code_score.get("acc", score_val > 0))
                pred_val = code_score.get("pred", None)
            else:
                score_val = float(code_score)
                acc_val = score_val > 0
                pred_val = None
            score_result = {"score": score_val, "acc": acc_val, "pred": pred_val}

        elif args.use_rlcr:
            rlcr_text = generated_text
            if "<answer>" not in rlcr_text and "</answer>" in rlcr_text:
                rlcr_text = "<answer>" + rlcr_text
            score_result = compute_rlcr_score(solution_str=rlcr_text, ground_truth=ground_truth)
            answer_valid = score_result.get("answer_valid", True)
            pred = score_result.get("pred")
            if (not answer_valid or pred in {None, ""}) and "\\boxed{" in generated_text:
                boxed_result = compute_score(
                    solution_str=generated_text, ground_truth=ground_truth, strict_box_verify=True
                )
                boxed_acc = 1 if boxed_result["acc"] else 0
                boxed_brier = (boxed_acc - 0.5) ** 2
                score_result = {
                    "score": boxed_acc - boxed_brier,
                    "acc": bool(boxed_acc),
                    "pred": boxed_result["pred"] if boxed_result["pred"] is not None else "",
                    "confidence": 0.5,
                    "confidence_valid": False,
                    "answer_valid": boxed_result["pred"] is not None,
                    "brier": boxed_brier,
                }
            if data_source.lower() in GPQA_DATA_SOURCES:
                score_result = apply_gpqa_choice_scoring(score_result, ground_truth)
        else:
            score_result = compute_score(
                solution_str=generated_text, ground_truth=ground_truth, strict_box_verify=True
            )

        result = {
            "input": tokenizer.decode(tokenizer.encode(prompt), skip_special_tokens=True),
            "output": generated_text,
            "gts": ground_truth,
            "score": score_result["score"],
            "step": args.step,
            "reward": score_result["score"],
            "acc": score_result["acc"],
            "pred": score_result["pred"],
            "data_source": data_source,
            "num_of_generated_tokens": num_tokens,
        }

        if "sara" in data_source.lower():
            try:
                float(str(ground_truth).replace('$', '').replace(',', ''))
                is_numerical = True
            except (ValueError, AttributeError):
                is_numerical = False
            if is_numerical:
                sara_acc = compute_sara_integer_part_accuracy(
                    solution_str=generated_text, ground_truth=ground_truth
                )
                if sara_acc is not None:
                    result["acc"] = bool(sara_acc)
                    result["score"] = 1.0 if sara_acc else -1.0
                    result["reward"] = result["score"]
                risk = compute_sara_numerical_risk(solution_str=generated_text, ground_truth=ground_truth)
                if risk is not None:
                    result["risk"] = risk

        return idx, result

    print(f"Processing outputs and computing scores (workers={n_workers})...")
    results_map: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_score_one, i): i for i in range(len(outputs))}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
            idx, result = future.result()
            results_map[idx] = result
    results = [results_map[i] for i in range(len(outputs))]

    # Save results
    print(f"Saving results to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    with open(args.output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o) + "\n")

    # Print summary
    correct = sum(1 for r in results if r["acc"])
    total = len(results)
    print(f"\n=== Summary ===")
    print(f"Total samples: {total}")
    print(f"Correct: {correct} ({100*correct/total:.2f}%)")
    print(f"No boxed: {sum(1 for r in results if r['pred'] is None)}")
    print(f"Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
