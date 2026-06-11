import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import numpy as np
import itertools
import copy
import os
import argparse
import json
import logging
import re
from datetime import datetime

from probe import ConfProbe
from tqdm import tqdm
from metric_utils import _compute_ece, _compute_brier_score
from typing import Optional, Dict, List, Tuple, Any

INPUT_FILE = "hidden_states_aime25_step0_deepseek-r1-distill-qwen-7b.pt"
OUTPUT_DIR = "checkpoints/probe"
def parse_args():
    parser = argparse.ArgumentParser(description="Train COT probe with MC labels")

    # Data paths
    parser.add_argument("--input_file", type=str, default=",".join(INPUT_FILE),
                        help="Comma-separated paths to cached traces parquet files")

    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory for probe checkpoint")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="Optional labels file; if omitted, uses hidden-state metadata")
    parser.add_argument("--label_key", type=str, default="mean_accuracy",
                        help="Key to read labels from metadata or JSON/JSONL")
    parser.add_argument("--split_mode", type=str, default="prompt_hash",
                        choices=["prompt_hash", "prompt", "data_source", "random"],
                        help="How to split train/val: 'prompt_hash' (group by prompt hash), "
                             "'prompt' (group by prompt text), 'data_source' (group by dataset), "
                             "'random' (random split, no grouping)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of traces to use (None = all)")
    parser.add_argument("--sweep_tag", type=str, default=None,
                        help="Optional tag to include in saved checkpoint names")

    # Training settings
    parser.add_argument("--epochs", type=int, default=100,help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-2,help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1,help="Dropout probability")
    parser.add_argument("--val_split", type=float, default=0.2,help="Validation split ratio")
    parser.add_argument("--patience", type=int, default=30,help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,help="Random seed")
    parser.add_argument("--no_seed", action="store_true",
                        help="Do not set torch/numpy seeds or deterministic split generators")
    parser.add_argument("--device", type=str, default="cuda",help="Device for training")


    return parser.parse_args()

def build_group_split_indices(
    metadata: List[Dict[str, Any]],
    val_split: float,
    seed: Optional[int],
    group_key: str,
) -> Tuple[List[int], List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, meta in enumerate(metadata):
        group_val = meta.get(group_key)
        if group_val is None:
            available_keys = list(meta.keys()) if meta else []
            raise ValueError(
                f"Missing group key '{group_key}' at index {idx}. "
                f"Available keys: {available_keys}. "
                f"Try using --split_mode with one of: {available_keys} or 'random'"
            )
        group_val = str(group_val)
        groups.setdefault(group_val, []).append(idx)

    group_keys = list(groups.keys())
    if seed is None:
        group_keys = np.random.default_rng().permutation(group_keys).tolist()
    else:
        rng = np.random.RandomState(seed)
        rng.shuffle(group_keys)

    total = len(metadata)
    target_val = max(1, int(total * val_split))
    val_indices: List[int] = []
    train_indices: List[int] = []
    val_count = 0

    for key in group_keys:
        group_indices = groups[key]
        if val_count < target_val:
            val_indices.extend(group_indices)
            val_count += len(group_indices)
        else:
            train_indices.extend(group_indices)

    if not train_indices:
        # Ensure both splits are non-empty
        train_indices = val_indices
        val_indices = []

    logging.info(
        "Grouped split by %s: %d groups, train=%d, val=%d",
        group_key,
        len(group_keys),
        len(train_indices),
        len(val_indices),
    )
    return train_indices, val_indices

def train_single_run(
    X, y, 
    config, 
    device, 
    split_indices: Optional[Tuple[List[int], List[int]]] = None,
    verbose=False,
):
    """
    Trains one specific configuration and returns validation metrics.
    """
    # 1. Setup Data
    dataset = TensorDataset(X, y)
    if split_indices is not None:
        train_indices, val_indices = split_indices
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
    else:
        val_size = int(len(dataset) * config["val_split"])
        split_lengths = [len(dataset) - val_size, val_size]
        if config.get("seed") is None:
            train_ds, val_ds = random_split(dataset, split_lengths)
        else:
            train_ds, val_ds = random_split(
                dataset,
                split_lengths,
                generator=torch.Generator().manual_seed(config["seed"]),
            )
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    
    # 2. Setup Model
    model = ConfProbe(
        hidden_size=X.shape[1], 
        dropout=config['dropout']
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    # Standard BCE Loss
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience_limit = config["patience"]
    
    # 3. Training Loop
    for epoch in tqdm(range(config['epochs']), desc="Training"):
        # --- Train ---
        model.train()
        for h_states, targets in train_loader:
            h_states = h_states.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            # Model returns shape (batch,), view as (batch, 1)
            logits = model(h_states).view(-1, 1)
            loss = loss_fn(logits, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            
        # --- Validate ---
        model.eval()
        val_loss = 0.0
        preds_all = []
        targets_all = []
        
        with torch.no_grad():
            for h_states, targets in val_loader:
                h_states = h_states.to(device)
                targets_soft = targets.to(device) # Keep soft for evaluation metrics
                targets_hard = torch.round(targets_soft)
                
                logits = model(h_states).view(-1, 1)
                val_loss += loss_fn(logits, targets_hard.view(-1, 1)).item()
                
                preds_all.append(torch.sigmoid(logits).cpu())
                targets_all.append(targets_soft.cpu())
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # Calculate final metrics for the best epoch
            preds_tensor = torch.cat(preds_all).view(-1)
            targets_tensor = torch.cat(targets_all).view(-1)
            
            mae = (preds_tensor - targets_tensor).abs().mean().item()
            brier = ((preds_tensor - targets_tensor)**2).mean().item()
            corr = np.corrcoef(preds_tensor.numpy(), targets_tensor.numpy())[0, 1]
            ece = compute_ece(preds_tensor.numpy(), targets_tensor.numpy())
            
            metrics = {
                "loss": best_val_loss,
                "mae": mae,
                "brier": brier,
                "corr": corr,
                "ece": ece
            }
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                break
                
    if verbose:
        print(f"  Result: Loss {metrics['loss']:.4f} | Corr {metrics['corr']:.4f} | Brier {metrics['brier']:.4f} | ECE {metrics['ece']:.4f}")

    return metrics, best_state

def sanitize_tag(text: str) -> str:
    if not text:
        return "unknown"
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "unknown"

def load_hidden_states_checkpoint(path: str) -> Tuple[torch.Tensor, Optional[List[Dict]], Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "hidden_states" in payload:
        hidden_states = payload["hidden_states"]
        metadata = payload.get("metadata")
        meta = payload.get("meta", {})
    else:
        hidden_states = payload
        metadata = None
        meta = {}

    if isinstance(hidden_states, np.ndarray):
        hidden_states = torch.from_numpy(hidden_states)
    return hidden_states, metadata, meta

def load_labels_from_file(path: str, label_key: str) -> torch.Tensor:
    if path.endswith((".pt", ".pth", ".bin")):
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            labels = payload.get("labels")
            if labels is None:
                labels = payload.get(label_key)
        else:
            labels = payload
    elif path.endswith(".jsonl"):
        labels_list = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                if isinstance(item, dict):
                    labels_list.append(item.get(label_key))
                else:
                    labels_list.append(item)
        labels = labels_list
    elif path.endswith(".json"):
        with open(path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            labels = payload.get("labels", payload.get(label_key))
        else:
            labels = payload
    else:
        raise ValueError(f"Unsupported labels file: {path}")

    if labels is None:
        raise ValueError(f"Labels not found using key '{label_key}' in {path}")
    labels_tensor = torch.as_tensor(labels, dtype=torch.float32)
    return labels_tensor.view(-1)

def build_labels_from_metadata(metadata: List[Dict], label_key: str) -> torch.Tensor:
    labels = []
    for item in metadata:
        labels.append(item.get(label_key))
    if any(v is None for v in labels):
        raise ValueError(f"Missing '{label_key}' in hidden state metadata.")
    labels_tensor = torch.as_tensor(labels, dtype=torch.float32)
    return labels_tensor.view(-1)

def build_sweep_tag(meta: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return sanitize_tag(override)
    pieces = []
    for key in ("dataset_name", "run_name", "model_tag", "budget_tag"):
        value = meta.get(key)
        if value:
            pieces.append(str(value))
    if not pieces:
        pieces.append("probe")
    pieces.append(datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
    return sanitize_tag("-".join(pieces))

def run_probe_sweep(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str,
    base_config: Dict[str, Any],
    sweep_tag: str,
    split_indices: Optional[Tuple[List[int], List[int]]],
    meta: Optional[Dict[str, Any]] = None,
):
    """
    Main entry point: Sweeps hyperparameters and saves the best model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Input Shape: {hidden_states.shape}")
    
    # 1. Define Grid
    param_grid = {
        "lr": [1e-3, 5e-4, 1e-4],
        "dropout": [0.5, 0.8],
        "weight_decay": [1e-3, 1e-2],
    }
    
    # Generate all combinations
    keys, values = zip(*param_grid.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Sweeping {len(configs)} configurations...")
    print("-" * 60)
    
    best_overall_metric = float("inf") # Minimizing Loss
    best_overall_config = None
    best_overall_state = None
    sweep_results = []
    
    # 2. Sweep
    for i, config in enumerate(configs):
        full_config = {**base_config, **config}
        print(f"[{i+1}/{len(configs)}] cfg: {full_config} ...", end="", flush=True)
        metrics, state = train_single_run(
            hidden_states,
            labels,
            full_config,
            device,
            split_indices=split_indices,
            verbose=True,
        )
        sweep_results.append({"config": full_config, "metrics": metrics})
        
        if metrics['loss'] < best_overall_metric:
            best_overall_metric = metrics['loss']
            best_overall_config = full_config
            best_overall_state = state
            
    # 3. Finalize
    print("=" * 60)
    print(f"Best Config: {best_overall_config}")
    print(f"Best Val Loss: {best_overall_metric:.4f}")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    best_name = f"best_probe_{sweep_tag}.pt"
    save_path = os.path.join(output_dir, best_name)
    
    # Re-instantiate model structure to save standard state dict
    final_model = ConfProbe(
        hidden_size=hidden_states.shape[1], 
        dropout=best_overall_config['dropout']
    )
    final_model.load_state_dict(best_overall_state)
    torch.save(final_model.state_dict(), save_path)
    print(f"Saved best model to {save_path}")

    latest_path = os.path.join(output_dir, "best_probe.pt")
    torch.save(final_model.state_dict(), latest_path)
    print(f"Updated latest best model at {latest_path}")

    summary_path = os.path.join(output_dir, f"best_probe_{sweep_tag}.json")
    summary = {
        "best_config": best_overall_config,
        "best_metrics": {"loss": best_overall_metric},
        "sweep_tag": sweep_tag,
        "meta": meta or {},
        "num_configs": len(configs),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved best model summary to {summary_path}")

    sweep_path = os.path.join(output_dir, f"sweep_results_{sweep_tag}.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Saved sweep results to {sweep_path}")
    
    return final_model
def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, mask: Optional[np.ndarray] = None, bins: int = 10) -> Optional[float]:
    """Compute Expected Calibration Error (ECE)."""
    if len(confidences) == 0 or len(accuracies) == 0:
        return None
    
    if mask is None:
        valid_mask = np.ones_like(confidences, dtype=bool)
    else:
        valid_mask = mask > 0
    
    if not np.any(valid_mask):
        return None
    
    valid_conf = confidences[valid_mask]
    valid_acc = accuracies[valid_mask]
    
    conf_tensor = torch.from_numpy(valid_conf).float()
    acc_tensor = torch.from_numpy(valid_acc).float()
    mask_tensor = torch.ones_like(conf_tensor, dtype=torch.bool)
    
    ece = _compute_ece(conf_tensor, acc_tensor, mask_tensor, bins=bins)
    return ece


def compute_brier_score(confidences: np.ndarray, accuracies: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[float]:
    """Compute Brier score."""
    if len(confidences) == 0 or len(accuracies) == 0:
        return None
    
    if mask is None:
        valid_mask = np.ones_like(confidences, dtype=bool)
    else:
        valid_mask = mask > 0
    
    if not np.any(valid_mask):
        return None
    
    conf_tensor = torch.from_numpy(confidences).float()
    acc_tensor = torch.from_numpy(accuracies).float()
    mask_tensor = torch.from_numpy(valid_mask).to(torch.bool)

    brier_score = _compute_brier_score(conf_tensor, acc_tensor, mask_tensor)
    return brier_score
if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    hidden_states, metadata, meta = load_hidden_states_checkpoint(args.input_file)
    if args.max_samples is not None:
        hidden_states = hidden_states[:args.max_samples]
        if metadata is not None:
            metadata = metadata[:args.max_samples]

    if args.labels_file:
        labels = load_labels_from_file(args.labels_file, args.label_key)
    else:
        if metadata is None:
            raise ValueError("No labels_file provided and hidden state metadata is missing.")
        labels = build_labels_from_metadata(metadata, args.label_key)

    if args.max_samples is not None:
        labels = labels[:args.max_samples]

    if hidden_states.shape[0] != labels.shape[0]:
        raise ValueError(f"Hidden states ({hidden_states.shape[0]}) and labels ({labels.shape[0]}) size mismatch.")

    hidden_states = hidden_states.float()
    labels = labels.float()

    seed = None if args.no_seed else args.seed
    if seed is None:
        logging.info("No random seed set for probe training.")
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)

    base_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_split": args.val_split,
        "patience": args.patience,
        "seed": seed,
    }

    split_indices = None
    if args.split_mode == "random":
        # Random split - no grouping, will use random_split in train_single_run
        logging.info("Using random train/val split (no grouping)")
        split_indices = None
    elif metadata is not None:
        # Group-based split using the specified split_mode as the group key
        split_indices = build_group_split_indices(metadata, args.val_split, seed, args.split_mode)
    else:
        logging.warning("No metadata available for grouped split, falling back to random split")
        split_indices = None

    sweep_tag = build_sweep_tag(meta, args.sweep_tag)
    best_model = run_probe_sweep(hidden_states, labels, args.output_dir, base_config, sweep_tag, split_indices, meta)
