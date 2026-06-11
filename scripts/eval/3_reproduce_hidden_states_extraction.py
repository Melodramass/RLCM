#!/usr/bin/env python3
"""
Extract hidden states from a reproduced jsonl file.
"""

import os
import sys
import argparse
import json
import logging
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple


import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from tqdm import tqdm
from probe import HiddenStateHook

def get_available_gpus() -> List[int]:
    """Get list of available GPU indices."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))

def load_reproduced_data(input_file: str) -> List[Dict]:
    """Load reproduced jsonl file and extract relevant fields."""
    data = []
    with open(input_file, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    logging.info(f"Loaded {len(data)} rollouts from {input_file}")
    return data

def load_hf_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, HiddenStateHook]:

    logging.info(f"Loading HF model from {model_path} for hidden state extraction...")
    
    available_gpus = get_available_gpus()
    logging.info(f"Available GPUs: {len(available_gpus)} ({available_gpus})")
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        device_map="balanced",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    model.eval()
    
    # Attach hook to final layer norm to capture hidden states
    hook = HiddenStateHook()

    # Find the final layer norm (varies by model architecture)
    if hasattr(model.model, 'norm'):
        # Qwen2, Llama style
        model.model.norm.register_forward_hook(hook)
    elif hasattr(model.model, 'final_layernorm'):
        # Some other architectures
        model.model.final_layernorm.register_forward_hook(hook)
    else:
        raise ValueError(f"Cannot find final layer norm in model architecture")
    
    return model, hook

def build_sequence(item: Dict) -> str:
    """Build the full sequence to feed the model based on input format."""
    
    prompt = item.get("prompt", item.get("input", ""))
    truncated_cot = item.get("truncated_cot", "")
    output = item.get("output", "")
    return prompt + truncated_cot + output


@torch.no_grad()
def extract_hidden_state(
    model: AutoModelForCausalLM,
    hook: HiddenStateHook,
    tokenizer: AutoTokenizer,
    sequence: str,
    max_seq_len: Optional[int],
) -> torch.Tensor:
    """Extract last-token hidden state from the model."""
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    hook.clear()
    _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        use_cache=False,
    )

    seq_len = attention_mask.sum().item() - 1
    hidden = hook.hidden_states[0, seq_len, :].detach().cpu()
    return hidden

def sanitize_tag(text: str) -> str:
    """Sanitize strings for safe, simple file names."""
    if not text:
        return "unknown"
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "unknown"

def infer_dataset_name_from_input(input_file: str) -> str:
    """Infer dataset name from common input file naming patterns."""
    stem = os.path.basename(input_file)
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    for prefix in ("budget_eval_results_", "reproduced_", "outputs_"):
        if stem.startswith(prefix):
            rest = stem[len(prefix):]
            parts = rest.split("_")
            if parts and parts[0]:
                return parts[0]
    return "unknown"

def infer_run_name_from_input(input_file: str, dataset_name: str) -> Optional[str]:
    """Infer run name from input file by removing known prefixes and dataset token."""
    stem = os.path.basename(input_file)
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    for prefix in ("budget_eval_results_", "reproduced_", "outputs_"):
        if stem.startswith(prefix):
            rest = stem[len(prefix):]
            if dataset_name != "unknown" and rest.startswith(dataset_name + "_"):
                rest = rest[len(dataset_name) + 1:]
            return rest or None
    return None

def model_tag_from_path(model_path: str) -> str:
    """Build a compact model tag from a checkpoint path."""
    path = model_path.rstrip("/")
    parts = path.split(os.sep)
    base = parts[-1] if parts else ""

    if base in {"merge", "checkpoint", "final", "model"} and len(parts) >= 2:
        base = parts[-2]
        if re.match(r"global_step_\\d+", base) and len(parts) >= 3:
            model_name = parts[-3]
            step = base.replace("global_step_", "step")
            base = f"{model_name}-{step}"
    elif re.match(r"global_step_\\d+", base) and len(parts) >= 2:
        model_name = parts[-2]
        step = base.replace("global_step_", "step")
        base = f"{model_name}-{step}"

    return sanitize_tag(base)

def model_path_to_dir(model_path: str) -> str:
    """Convert a model path to a safe relative directory path."""
    normed = os.path.normpath(model_path).replace("\\", "/")
    normed = normed.lstrip("/")
    normed = normed.replace(":", "_")
    return normed or "unknown"

def make_budget_tag(budgets: List[int]) -> str:
    """Create a compact tag describing the budget split list."""
    if not budgets:
        return "bna"
    budgets = sorted(budgets)
    if len(budgets) == 1:
        return f"b{budgets[0]}"
    step = budgets[1] - budgets[0]
    if all(budgets[i] - budgets[i - 1] == step for i in range(1, len(budgets))):
        return f"b{budgets[0]}-{budgets[-1]}x{step}"
    if len(budgets) <= 6:
        return "b" + "-".join(str(b) for b in budgets)
    return f"b{budgets[0]}-{budgets[-1]}_n{len(budgets)}"

def resolve_output_paths(
    input_file: str,
    output_file: Optional[str],
    output_dir: Optional[str],
    model_path: str,
    dataset_name: str,
    run_name: Optional[str],
    model_tag: str,
    budget_tag: str,
) -> Tuple[str, str]:
    """Resolve output .pt and .json paths."""
    if output_file:
        pt_path = output_file
    else:
        base_dir = output_dir
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(input_file) or ".", "hidden_states")
        os.makedirs(base_dir, exist_ok=True)

        model_dir = os.path.join(base_dir, model_path_to_dir(model_path))
        os.makedirs(model_dir, exist_ok=True)

        pieces = ["hs", dataset_name]
        if run_name:
            pieces.append(run_name)
        if budget_tag != "bna":
            pieces.append(budget_tag)

        filename = sanitize_tag("_".join(pieces)) + ".pt"
        pt_path = os.path.join(model_dir, filename)

    json_path = os.path.splitext(pt_path)[0] + ".json"
    return pt_path, json_path

def compute_seq_quota(seq_len: int) -> float:
    """Compute quota for dynamic batching, mirroring eval_probe_anycal."""
    if seq_len > 4000:
        return (1 + 0.00025 * seq_len) * seq_len / 2
    return float(seq_len)

def create_dynamic_batches(
    sorted_indices: List[int],
    seq_lengths: List[int],
    max_quota: float,
    max_batch_size: int,
) -> List[List[int]]:
    """Create batches where total quota <= max_quota and size <= max_batch_size."""
    batches = []
    current_batch: List[int] = []
    current_quota = 0.0

    for idx in sorted_indices:
        seq_len = seq_lengths[idx]
        seq_quota = compute_seq_quota(seq_len)

        if current_batch and (current_quota + seq_quota > max_quota or len(current_batch) >= max_batch_size):
            batches.append(current_batch)
            current_batch = [idx]
            current_quota = seq_quota
        else:
            current_batch.append(idx)
            current_quota += seq_quota

    if current_batch:
        batches.append(current_batch)

    return batches

@torch.no_grad()
def extract_hidden_states_dynamic(
    model: AutoModelForCausalLM,
    hook: HiddenStateHook,
    tokenizer: AutoTokenizer,
    tokenized_sequences: List[List[int]],
    max_batch_quota: float,
    max_batch_size: int,
) -> torch.Tensor:
    """Extract last-token hidden states using dynamic batching."""
    num_samples = len(tokenized_sequences)
    hidden_size = model.config.hidden_size
    all_hidden_states = torch.zeros((num_samples, hidden_size), device="cpu")

    seq_lengths = [len(seq) for seq in tokenized_sequences]
    sorted_indices = sorted(range(num_samples), key=lambda x: seq_lengths[x], reverse=True)

    logging.info(f"Sorted {num_samples} sequences by length (descending) for efficient batching")
    if num_samples > 0:
        logging.info(f"  Longest sequence: {seq_lengths[sorted_indices[0]]} tokens")
        logging.info(f"  Shortest sequence: {seq_lengths[sorted_indices[-1]]} tokens")

    dynamic_batches = create_dynamic_batches(sorted_indices, seq_lengths, max_batch_quota, max_batch_size)
    batch_sizes = [len(b) for b in dynamic_batches]
    if batch_sizes:
        logging.info(
            "Created %d dynamic batches (max quota %.0f, max size %d). "
            "Batch sizes: min=%d, max=%d, avg=%.1f",
            len(dynamic_batches),
            max_batch_quota,
            max_batch_size,
            min(batch_sizes),
            max(batch_sizes),
            sum(batch_sizes) / len(batch_sizes),
        )

    model_device = next(model.parameters()).device

    for batch_indices in tqdm(dynamic_batches, desc="Extracting hidden states"):
        batch_seqs = [tokenized_sequences[idx] for idx in batch_indices]
        max_len = max(len(seq) for seq in batch_seqs)

        padded_seqs = []
        attention_masks = []
        for seq in batch_seqs:
            pad_len = max_len - len(seq)
            padded_seqs.append([tokenizer.pad_token_id] * pad_len + seq)
            attention_masks.append([0] * pad_len + [1] * len(seq))

        input_ids = torch.tensor(padded_seqs, dtype=torch.long, device=model_device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=model_device)

        hook.clear()
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )

        hidden_states = hook.hidden_states.detach().cpu()

        for batch_idx, sample_idx in enumerate(batch_indices):
            seq_len = len(batch_seqs[batch_idx])
            pad_len = max_len - seq_len
            last_pos = pad_len + seq_len - 1
            all_hidden_states[sample_idx] = hidden_states[batch_idx, last_pos, :]

        del hidden_states
        torch.cuda.empty_cache()

    return all_hidden_states

def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load a hidden state checkpoint from disk."""
    logging.info(f"Loading hidden states from {checkpoint_path}...")
    data = torch.load(checkpoint_path, map_location="cpu")
    return data
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract hidden states from a reproduced jsonl file"
    )
    parser.add_argument("--input_file",type=str,required=True,help="Path to reproduced jsonl file",)
    parser.add_argument("--model_path",type=str,required=True,help="Path to the model checkpoint",)
    parser.add_argument("--output_file", type=str, default=None, help="Path to output .pt file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for outputs when output_file is not set")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name for output naming")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for output naming")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Max sequence length (default: model max)")
    parser.add_argument("--max_batch_quota", type=float, default=12 * 10000, help="Max dynamic batch quota")
    parser.add_argument("--max_batch_size", type=int, default=32, help="Max dynamic batch size")
    parser.add_argument("--resume", action="store_true", help="Load existing checkpoint if present and exit")
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    data = load_reproduced_data(args.input_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    model, hook = load_hf_model(args.model_path)
    if args.max_seq_len is None:
        args.max_seq_len = getattr(model.config, "max_position_embeddings", None)

    budgets = sorted({item.get("budget") for item in data if item.get("budget") is not None})
    dataset_name = args.dataset_name or infer_dataset_name_from_input(args.input_file)
    run_name = args.run_name or infer_run_name_from_input(args.input_file, dataset_name)
    model_tag = model_tag_from_path(args.model_path)
    budget_tag = make_budget_tag(budgets)

    output_pt, output_json = resolve_output_paths(
        args.input_file,
        args.output_file,
        args.output_dir,
        args.model_path,
        dataset_name,
        sanitize_tag(run_name) if run_name else None,
        model_tag,
        budget_tag,
    )

    if args.resume and os.path.exists(output_pt):
        checkpoint = load_checkpoint(output_pt)
        hidden_states = checkpoint.get("hidden_states")
        logging.info(
            "Loaded hidden states from %s (shape=%s)",
            output_pt,
            tuple(hidden_states.shape) if hasattr(hidden_states, "shape") else "unknown",
        )
        return

    sequences = [build_sequence(item) for item in data]
    tokenized_sequences = []
    for seq in sequences:
        encoded = tokenizer(
            seq,
            truncation=True,
            max_length=args.max_seq_len,
        )
        input_ids = encoded.get("input_ids", [])
        if not input_ids:
            input_ids = [tokenizer.pad_token_id]
        tokenized_sequences.append(input_ids)

    metadata = []
    for idx, item in enumerate(data):
        prompt_text = item.get("prompt", item.get("input", ""))
        prompt_hash = None
        if isinstance(prompt_text, str) and prompt_text:
            prompt_hash = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()
        meta = {
            "index": idx,
            "rollout_idx": item.get("rollout_idx", None),
            "budget": item.get("budget", None),
            "mean_accuracy": item.get("mean_accuracy", None),
            "prompt_hash": prompt_hash,
            "data_source": item.get("data_source", "unknown"),
            "prompt": prompt_text,
        }
        metadata.append(meta)

    hidden_states = extract_hidden_states_dynamic(
        model,
        hook,
        tokenizer,
        tokenized_sequences,
        max_batch_quota=args.max_batch_quota,
        max_batch_size=args.max_batch_size,
    )

    checkpoint_payload = {
        "hidden_states": hidden_states,
        "metadata": metadata,
        "meta": {
            "input_file": args.input_file,
            "model_path": args.model_path,
            "dataset_name": dataset_name,
            "run_name": run_name,
            "model_tag": model_tag,
            "budget_tag": budget_tag,
            "budgets": budgets,
            "max_seq_len": args.max_seq_len,
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
    }

    os.makedirs(os.path.dirname(output_pt) or ".", exist_ok=True)
    torch.save(checkpoint_payload, output_pt)
    logging.info(f"Saved hidden states to {output_pt} (shape={tuple(hidden_states.shape)})")

    with open(output_json, "w") as f:
        json.dump(checkpoint_payload["meta"], f, indent=2)
    logging.info(f"Saved checkpoint metadata to {output_json}")

if __name__ == "__main__":
    main()
