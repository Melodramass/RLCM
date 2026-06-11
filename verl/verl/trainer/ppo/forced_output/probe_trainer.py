"""
Budget Probe Trainer for Forced Output GRPO.

Trains a lightweight 2-layer MLP probe that predicts per-budget accuracy
from the model's hidden states.  The probe is entirely standalone — no
gradient flows back into the language model.

Architecture:
    Input  → Linear(hidden_size, hidden_size // 4)
           → ReLU
           → Dropout
           → Linear(hidden_size // 4, 1)
           → Sigmoid  (output: predicted accuracy ∈ [0, 1])

Training target:
    Monte-Carlo accuracy at each budget checkpoint, computed from
    ``num_forced_answers`` samples per budget.

Usage in the GRPO loop:
    1. After vLLM rollout, compute MC accuracy per budget from forced answers.
    2. During ``_compute_old_log_prob`` the actor model's forward pass runs
       with a hook to capture hidden states from the last ``probe_layer_idx``
       layer.
    3. For each budget, extract the mean of the last ``probe_avg_last_k``
       hidden states up to the budget position.
    4. Train the probe MLP on (hidden_state, mc_accuracy) pairs for
       ``probe_train_steps`` mini-batch gradient descent steps.
    5. Use the probe's predicted accuracy as an additional reward signal.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Probe MLP
# ---------------------------------------------------------------------------

class BudgetProbeMLP(nn.Module):
    """2-layer MLP: h → h//4 → ReLU → Dropout → 1."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        intermediate = hidden_size // 4
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, hidden_size)
        Returns:
            logits: (batch,)  — apply sigmoid to get probability.
        """
        return self.net(x).squeeze(-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probability."""
        return torch.sigmoid(self.forward(x))


# ---------------------------------------------------------------------------
# Probe Trainer
# ---------------------------------------------------------------------------

class BudgetProbeTrainer:
    """Manages the probe MLP lifecycle: init, train, infer, save/load.

    Config keys (all under ``forced_output.probe``):
        enable (bool):            Master switch.
        train_steps (int):        SGD steps per GRPO step (default 20).
        batch_size (int):         Mini-batch size for probe training (default 16).
        lr (float):               Learning rate (default 1e-3).
        dropout (float):          Dropout in MLP (default 0.1).
        avg_last_k (int):         # of hidden states to average before the budget
                                  position (default 1).
        last_layer_idx (int):     Which transformer layer to hook, counted from the
                                  end (1 = last layer, 2 = second-to-last, etc.).
                                  Default 1.
        reward_weight (float):    Weight for probe reward (default 0.0 = disabled).
        use_as_reward (bool):     Whether to add probe accuracy as reward term.
                margin_corner (bool):     In ``conf_diff`` / ``conf_diff_v1`` /
                      ``conf_margin_covar`` mode, whether to apply
                      corner handling when only one class exists:
                      - all correct -> ``conf_correct - 0``
                                            - all incorrect ->
                                                - ``conf_diff``: ``1 - conf_incorrect``
                                                - ``conf_diff_v1`` /
                                                  ``conf_margin_covar``:
                                                  ``0 - conf_incorrect``.
                      Default False (return 0 for one-class cases).
        weight_decay (float):     AdamW weight decay (default 0.01).
        device (str):             Device for probe training.  Use ``"auto"`` (default)
                                  to pick ``"cuda"`` when available and ``"cpu"``
                                  otherwise.  Explicit values like ``"cuda:0"`` or
                                  ``"cpu"`` are also accepted.
        ranking_reward_rt (float): Reward used by ``ranking_reward`` mode when
                                  the average probe confidence is on the correct
                                  side of the 0.5 decision boundary but not
                                  strong enough to receive the full +/-1 reward.
                                  Default 0.5.
        filter_natural_corner_groups (bool): Optional trainer-side filter flag
                      that drops prompt groups whose natural-trace
                      correctness is unanimous (all correct or all
                      incorrect). This filter is applied in the
                      forced-output trainer, not inside probe MLP
                      training itself.
    """

    VALID_REWARD_MODES = (
        "cross_entropy",
        "accuracy",
        "brier",
        "conf_diff",
        "conf_diff_v1",
        "conf_diff_v2",
        "conf_diff_v3",
        "conf_margin_covar",
        "ranking_reward",
    )
    RANKING_CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        hidden_size: int,
        *,
        train_steps: int = 20,
        batch_size: int = 16,
        lr: float = 1e-3,
        dropout: float = 0.1,
        avg_last_k: int = 1,
        last_layer_idx: int = 1,
        reward_weight: float = 0.0,
        use_as_reward: bool = False,
        margin_corner: bool = False,
        weight_decay: float = 0.01,
        device: str = "auto",
        reward_mode: str = "accuracy",
        ranking_reward_rt: float = 0.5,
        conf_diff_v2_tau: float = 0.5,
        conf_diff_v2_alpha: float = 1.0,
        conf_diff_v3_alpha_both: float = 1.0,
        conf_diff_v3_alpha_only_correct: float = 1.0,
        conf_diff_v3_alpha_only_incorrect: float = 1.0,
    ):
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.avg_last_k = avg_last_k
        self.last_layer_idx = last_layer_idx
        self.reward_weight = reward_weight
        self.use_as_reward = use_as_reward
        self.margin_corner = margin_corner
        self.ranking_reward_rt = float(ranking_reward_rt)
        self.conf_diff_v2_tau = float(conf_diff_v2_tau)
        self.conf_diff_v2_alpha = float(conf_diff_v2_alpha)
        self.conf_diff_v3_alpha_both = float(conf_diff_v3_alpha_both)
        self.conf_diff_v3_alpha_only_correct = float(conf_diff_v3_alpha_only_correct)
        self.conf_diff_v3_alpha_only_incorrect = float(conf_diff_v3_alpha_only_incorrect)
        if reward_mode not in self.VALID_REWARD_MODES:
            raise ValueError(
                f"Unknown probe reward_mode: {reward_mode!r}. "
                f"Supported: {self.VALID_REWARD_MODES}"
            )
        self.reward_mode = reward_mode
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        logger.info("BudgetProbeTrainer using device: %s", self.device)

        # Build MLP
        self.mlp = BudgetProbeMLP(hidden_size, dropout=dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=lr, weight_decay=weight_decay)

        # Running metrics
        self._step_count = 0

    # ------------------------------------------------------------------
    # Hidden-state extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_budget_hidden_states(
        hidden_states: torch.Tensor,
        prompt_length: int,
        budget_positions: list[int],
        avg_last_k: int = 1,
    ) -> torch.Tensor:
        """Extract per-budget hidden state vectors from a single sample.

        For each budget position, we take the mean of the last ``avg_last_k``
        hidden states up to (prompt_length + budget_pos - 1) inclusive.

        Args:
            hidden_states: (seq_len, hidden_size) — hidden states for one sample.
            prompt_length: Number of prompt tokens.
            budget_positions: List of budget truncation lengths (in response tokens).
            avg_last_k: Number of positions to average.

        Returns:
            (n_budgets, hidden_size) tensor.
        """
        result = []
        seq_len = hidden_states.shape[0]
        for bp in budget_positions:
            # Position of the last token at this budget (0-indexed into seq)
            end_pos = min(prompt_length + bp, seq_len)
            start_pos = max(end_pos - avg_last_k, 0)
            if start_pos >= end_pos:
                # Edge-case: zero-length range
                result.append(torch.zeros(hidden_states.shape[-1], device=hidden_states.device))
            else:
                result.append(hidden_states[start_pos:end_pos].mean(dim=0))
        return torch.stack(result, dim=0)  # (n_budgets, hidden_size)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(
        self,
        hidden_features: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """Train the probe for ``self.train_steps`` mini-batch SGD steps.

        Args:
            hidden_features: (N, hidden_size) — all budget hidden states for this
                GRPO step.  N = n_budgets × group_size × batch_size.
            targets: (N,) — Monte-Carlo accuracy for each budget point.  Values
                in [0, 1].

        Returns:
            Dict of metrics: loss, mean_pred, mean_target, accuracy.
        """
        self.mlp.train()
        N = hidden_features.shape[0]
        if N == 0:
            return {"probe/loss": 0.0, "probe/n_samples": 0}

        hidden_features = hidden_features.to(self.device, dtype=torch.float32)
        targets = targets.to(self.device).float()

        total_loss = 0.0
        n_batches = 0

        for _ in range(self.train_steps):
            # Shuffle and create mini-batches
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                idx = perm[start:end]
                batch_x = hidden_features[idx]
                batch_y = targets[idx]

                logits = self.mlp(batch_x)
                loss = F.binary_cross_entropy_with_logits(logits, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        self._step_count += 1

        # Compute final metrics
        self.mlp.eval()
        with torch.no_grad():
            all_logits = self.mlp(hidden_features)
            all_preds = torch.sigmoid(all_logits)
            final_loss = F.binary_cross_entropy_with_logits(all_logits, targets).item()
            # Binarize predictions at 0.5 and compare with rounded targets
            pred_binary = (all_preds > 0.5).float()
            target_binary = (targets > 0.5).float()
            accuracy = (pred_binary == target_binary).float().mean().item()

        metrics = {
            "probe/train_loss": total_loss / max(n_batches, 1),
            "probe/eval_loss": final_loss,
            "probe/mean_pred": all_preds.mean().item(),
            "probe/mean_target": targets.mean().item(),
            "probe/binary_accuracy": accuracy,
            "probe/n_samples": N,
            "probe/step_count": self._step_count,
        }
        # Always expose confidence bucket statistics when probe is active.
        # Buckets follow existing conf_diff semantics:
        # - correct   => target == 1.0
        # - incorrect => target == 0.0
        correct_preds = all_preds[targets == 1.0]
        incorrect_preds = all_preds[targets == 0.0]
        metrics.update(
            self._build_confidence_bucket_metrics(
                correct_preds=correct_preds.detach().cpu().tolist(),
                incorrect_preds=incorrect_preds.detach().cpu().tolist(),
            )
        )
        return metrics

    @staticmethod
    def _build_confidence_bucket_metrics(
        *,
        correct_preds: list[float],
        incorrect_preds: list[float],
    ) -> dict[str, float]:
        """Compute mean/std confidence per correctness bucket.

        Returns all four keys. If a bucket has no samples, both values are NaN.
        """

        def _summary(vals: list[float]) -> tuple[float, float]:
            if not vals:
                return float("nan"), float("nan")
            arr = np.asarray(vals, dtype=np.float32)
            return float(arr.mean()), float(arr.std(ddof=0))

        mean_correct, std_correct = _summary(correct_preds)
        mean_incorrect, std_incorrect = _summary(incorrect_preds)
        return {
            "probe/mean_conf_correct": mean_correct,
            "probe/mean_conf_incorrect": mean_incorrect,
            "probe/std_conf_correct": std_correct,
            "probe/std_conf_incorrect": std_incorrect,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, hidden_features: torch.Tensor) -> torch.Tensor:
        """Predict accuracy probabilities for each budget point.

        Args:
            hidden_features: (N, hidden_size)

        Returns:
            (N,) predictions in [0, 1].
        """
        self.mlp.eval()
        hidden_features = hidden_features.to(self.device, dtype=torch.float32)
        return self.mlp.predict(hidden_features).cpu()

    # ------------------------------------------------------------------
    # Batch processing helpers
    # ------------------------------------------------------------------

    def collect_probe_data(
        self,
        batch_hidden_states: torch.Tensor,
        batch_prompt_lengths: list[int],
        batch_budget_positions: list[list[int]],
        batch_mc_accuracies: list[list[float]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect (hidden_features, targets) from a full GRPO batch.

        Args:
            batch_hidden_states: (batch_size, seq_len, hidden_size) — on CPU.
            batch_prompt_lengths: List of prompt lengths per sample.
            batch_budget_positions: List of budget position lists per sample.
            batch_mc_accuracies: List of MC accuracy lists per sample (one per budget).

        Returns:
            hidden_features: (total_budgets, hidden_size)
            targets: (total_budgets,)
        """
        all_features = []
        all_targets = []

        for i in range(len(batch_prompt_lengths)):
            hs = batch_hidden_states[i]  # (seq_len, hidden_size)
            budgets = batch_budget_positions[i]
            mc_accs = batch_mc_accuracies[i]

            if not budgets or not mc_accs:
                continue

            features = self.extract_budget_hidden_states(
                hs, batch_prompt_lengths[i], budgets, self.avg_last_k
            )  # (n_budgets, hidden_size)
            targets = torch.tensor(mc_accs, dtype=torch.float32)

            all_features.append(features)
            all_targets.append(targets)

        if not all_features:
            hidden_size = batch_hidden_states.shape[-1] if batch_hidden_states.dim() == 3 else 1
            return torch.zeros(0, hidden_size), torch.zeros(0)

        return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)

    @staticmethod
    def _compute_reward_signal(
        preds: torch.Tensor,
        targets: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Convert probe predictions + MC targets into a reward signal.

        Args:
            preds: (N,) sigmoid probabilities in [0, 1].
            targets: (N,) MC accuracy values in [0, 1].
            mode: One of ``"cross_entropy"``, ``"accuracy"``, ``"brier"``.

        Returns:
            (N,) reward values.

        Modes:
            - **cross_entropy**: ``target * log(pred) + (1-target) * log(1-pred)``.
              Always ≤ 0; higher (closer to 0) is better.
            - **accuracy**: ``1.0`` if ``round(pred) == round(target)`` else ``0.0``.
            - **brier**: ``-(pred - target)^2``.  Always ≤ 0; closer to 0 is better.
        """
        eps = 1e-7
        if mode == "cross_entropy":
            p = preds.clamp(eps, 1.0 - eps)
            return targets * p.log() + (1.0 - targets) * (1.0 - p).log()
        elif mode == "accuracy":
            pred_bin = (preds > 0.5).float()
            target_bin = (targets > 0.5).float()
            return (pred_bin == target_bin).float()
        elif mode == "brier":
            return -((preds - targets) ** 2)
        elif mode in (
            "conf_diff",
            "conf_diff_v1",
            "conf_diff_v2",
            "conf_diff_v3",
            "conf_margin_covar",
            "ranking_reward",
        ):
            # These modes are per-sample aggregations, not per-budget signals.
            raise ValueError(
                "probe reward mode 'conf_diff'/'conf_diff_v1'/'conf_diff_v2'/'conf_diff_v3'/"
                "'conf_margin_covar'/"
                "'ranking_reward' cannot be computed via "
                "_compute_reward_signal; use the per-sample reward helpers instead."
            )
        else:
            raise ValueError(f"Unknown probe reward mode: {mode!r}")

    def compute_probe_rewards(
        self,
        batch_hidden_states: torch.Tensor,
        batch_prompt_lengths: list[int],
        batch_budget_positions: list[list[int]],
        batch_mc_accuracies: list[list[float]] | None = None,
    ) -> list[list[float]]:
        """Compute per-sample, per-budget probe reward.

        The reward signal depends on ``self.reward_mode``:

        - **cross_entropy**: ``target * log(pred) + (1-target) * log(1-pred)``
        - **accuracy**: ``1.0`` if ``round(pred) == round(target)`` else ``0.0``
        - **brier**: ``-(pred - target)^2``

        Args:
            batch_hidden_states: (batch_size, seq_len, hidden_size) — on CPU.
            batch_prompt_lengths: List of prompt lengths per sample.
            batch_budget_positions: List of budget position lists per sample.
            batch_mc_accuracies: List of MC accuracy lists per sample (one per
                budget).  Required for all reward modes.

        Returns:
            List of lists: probe_rewards[i][j] = reward for sample i, budget j.
        """
        all_features = []
        all_targets = []
        sample_budget_counts = []

        for i in range(len(batch_prompt_lengths)):
            hs = batch_hidden_states[i]
            budgets = batch_budget_positions[i]
            if not budgets:
                sample_budget_counts.append(0)
                continue

            features = self.extract_budget_hidden_states(
                hs, batch_prompt_lengths[i], budgets, self.avg_last_k
            )
            all_features.append(features)
            sample_budget_counts.append(len(budgets))

            # Collect targets for this sample
            if batch_mc_accuracies is not None and i < len(batch_mc_accuracies):
                all_targets.append(torch.tensor(batch_mc_accuracies[i], dtype=torch.float32))
            else:
                # Fallback: zeros (should not happen if caller passes targets)
                all_targets.append(torch.zeros(len(budgets), dtype=torch.float32))

        if not all_features:
            return [[] for _ in range(len(batch_prompt_lengths))]

        all_features_cat = torch.cat(all_features, dim=0)
        all_targets_cat = torch.cat(all_targets, dim=0)
        preds = self.predict(all_features_cat)  # (total_budgets,)

        rewards = self._compute_reward_signal(preds, all_targets_cat, self.reward_mode)

        # Split back per sample
        result = []
        offset = 0
        for count in sample_budget_counts:
            if count == 0:
                result.append([])
            else:
                result.append(rewards[offset:offset + count].tolist())
                offset += count

        return result

    def compute_ranking_rewards(
        self,
        batch_hidden_states: torch.Tensor,
        batch_prompt_lengths: list[int],
        batch_budget_positions: list[list[int]],
        natural_correct: list[bool] | np.ndarray | torch.Tensor | None = None,
    ) -> tuple[list[float], dict[str, float]]:
        """Compute per-sample ranking rewards from natural correctness and mean confidence.

        For each sample, let ``p`` be the mean probe confidence across all
        available budget checkpoints. Let ``c`` be the natural-trace
        correctness. The reward is then:

        - ``1`` if ``c`` is correct and ``p > 0.5``
        - ``rt`` if ``c`` is correct and ``p <= 0.5``
        - ``-rt`` if ``c`` is incorrect and ``p <= 0.5``
        - ``-1`` if ``c`` is incorrect and ``p > 0.5``

        Args:
            batch_hidden_states: (batch_size, seq_len, hidden_size) on CPU.
            batch_prompt_lengths: Prompt lengths for each sample.
            batch_budget_positions: Budget positions for each sample.
            natural_correct: Natural-trace correctness for each sample.

        Returns:
            Tuple of per-sample reward scalars and aggregate logging metrics.
        """
        if natural_correct is None:
            raise ValueError(
                "natural_correct is required for probe reward mode 'ranking_reward'"
            )

        try:
            natural_correct_np = np.asarray(natural_correct).reshape(-1).astype(bool)
        except Exception:
            natural_correct_np = np.array(list(natural_correct), dtype=bool)

        if natural_correct_np.shape[0] != len(batch_prompt_lengths):
            raise ValueError(
                "natural_correct must have the same length as batch_prompt_lengths, "
                f"got {natural_correct_np.shape[0]} and {len(batch_prompt_lengths)}"
            )

        all_features = []
        sample_budget_counts = []

        for i in range(len(batch_prompt_lengths)):
            hs = batch_hidden_states[i]
            budgets = batch_budget_positions[i]
            if not budgets:
                sample_budget_counts.append(0)
                continue

            features = self.extract_budget_hidden_states(
                hs, batch_prompt_lengths[i], budgets, self.avg_last_k
            )
            all_features.append(features)
            sample_budget_counts.append(len(budgets))

        def _build_metrics(
            avg_conf_all: list[float],
            avg_conf_correct: list[float],
            avg_conf_incorrect: list[float],
            high_conf_flags: list[float],
        ) -> dict[str, float]:
            return {
                "probe/ranking_reward_rt": float(self.ranking_reward_rt),
                "probe/ranking_high_conf_threshold": float(self.RANKING_CONFIDENCE_THRESHOLD),
                "probe/ranking_avg_conf_mean": float(np.mean(avg_conf_all)) if avg_conf_all else float("nan"),
                "probe/ranking_avg_conf_correct": (
                    float(np.mean(avg_conf_correct)) if avg_conf_correct else float("nan")
                ),
                "probe/ranking_avg_conf_incorrect": (
                    float(np.mean(avg_conf_incorrect)) if avg_conf_incorrect else float("nan")
                ),
                "probe/ranking_high_conf_rate": (
                    float(np.mean(high_conf_flags)) if high_conf_flags else float("nan")
                ),
            }

        if not all_features:
            return [0.0] * len(batch_prompt_lengths), _build_metrics([], [], [], [])

        all_features_cat = torch.cat(all_features, dim=0)
        preds = self.predict(all_features_cat)  # (total_budgets,)

        rewards: list[float] = []
        avg_conf_all: list[float] = []
        avg_conf_correct: list[float] = []
        avg_conf_incorrect: list[float] = []
        high_conf_flags: list[float] = []
        offset = 0

        for i, count in enumerate(sample_budget_counts):
            if count == 0:
                rewards.append(0.0)
                continue

            sample_preds = preds[offset : offset + count]
            offset += count

            avg_conf = float(sample_preds.mean().item())
            is_correct = bool(natural_correct_np[i])
            is_high_conf = avg_conf > self.RANKING_CONFIDENCE_THRESHOLD

            avg_conf_all.append(avg_conf)
            high_conf_flags.append(float(is_high_conf))
            if is_correct:
                avg_conf_correct.append(avg_conf)
                rewards.append(1.0 if is_high_conf else self.ranking_reward_rt)
            else:
                avg_conf_incorrect.append(avg_conf)
                rewards.append(-1.0 if is_high_conf else -self.ranking_reward_rt)

        return rewards, _build_metrics(
            avg_conf_all=avg_conf_all,
            avg_conf_correct=avg_conf_correct,
            avg_conf_incorrect=avg_conf_incorrect,
            high_conf_flags=high_conf_flags,
        )

    def compute_conf_diff_rewards(
        self,
        batch_hidden_states: torch.Tensor,
        batch_prompt_lengths: list[int],
        batch_budget_positions: list[list[int]],
        batch_mc_accuracies: list[list[float]] | None = None,
        margin_corner: Optional[bool] = None,
    ) -> tuple[list[float], dict[str, float]]:
        """Compute per-sample ``mu_conf_correct - mu_conf_incorrect`` probe reward.

        For each sample, budgets are split into two categories based on whether
        **all** forced answers were correct (``mc_accuracy == 1.0``) or **all**
        forced answers were incorrect (``mc_accuracy == 0.0``).  Budgets with
        partial accuracy (0 < mc_accuracy < 1) are ignored.

        ``mu_conf_correct`` is the mean probe-predicted confidence over all
        strictly-correct budgets; ``mu_conf_incorrect`` is the mean over all
        strictly-incorrect budgets.  Missing groups default to ``0.0``.

        Args:
            batch_hidden_states: (batch_size, seq_len, hidden_size) — on CPU.
            batch_prompt_lengths: List of prompt lengths per sample.
            batch_budget_positions: List of budget position lists per sample.
            batch_mc_accuracies: List of MC accuracy lists per sample (one per
                budget).  Required; each value should be in [0, 1].
            margin_corner: Whether to handle one-class corner cases. If None,
                uses trainer-level ``self.margin_corner``.

                Corner behavior for one-class traces when ``margin_corner=True``:
            - ``reward_mode=conf_diff``: all-incorrect uses ``1 - conf_incorrect``
            - ``reward_mode=conf_diff_v1`` /
              ``reward_mode=conf_margin_covar``:
              all-incorrect uses ``0 - conf_incorrect``
                        - ``reward_mode=conf_diff_v2``:
                            - only-correct: ``alpha * (mu_conf_correct - tau)``
                            - only-incorrect: ``alpha * (tau - mu_conf_incorrect)``
                        - ``reward_mode=conf_diff_v3``:
                            - both classes: ``alpha_both * (mu_conf_correct - mu_conf_incorrect)``
                            - only-correct: ``alpha_only_correct * (mu_conf_correct - tau)``
                            - only-incorrect: ``alpha_only_incorrect * (tau - mu_conf_incorrect)``

        Returns:
            Tuple of:
              - List of per-sample scalars: ``mu_conf_correct - mu_conf_incorrect``.
              - Dict of aggregate metrics: mean conf for correct/incorrect budgets.
        """
        # Restored behavior: when the caller does not pass `margin_corner`,
        # use the trainer-level config so older scripts (e.g. `probe.margin_corner=true`)
        # continue to work without call-site changes.
        if margin_corner is None:
            margin_corner = self.margin_corner

        all_features = []
        sample_budget_counts = []
        sample_mc_acc_lists: list[list[float]] = []

        for i in range(len(batch_prompt_lengths)):
            hs = batch_hidden_states[i]
            budgets = batch_budget_positions[i]
            if not budgets:
                sample_budget_counts.append(0)
                sample_mc_acc_lists.append([])
                continue

            features = self.extract_budget_hidden_states(
                hs, batch_prompt_lengths[i], budgets, self.avg_last_k
            )
            all_features.append(features)
            sample_budget_counts.append(len(budgets))

            mc_accs = (
                batch_mc_accuracies[i]
                if (batch_mc_accuracies is not None and i < len(batch_mc_accuracies))
                else [0.0] * len(budgets)
            )
            sample_mc_acc_lists.append(mc_accs)

        if not all_features:
            return [0.0] * len(batch_prompt_lengths), {}

        all_features_cat = torch.cat(all_features, dim=0)
        preds = self.predict(all_features_cat)  # (total_budgets,)

        # Split predictions back per sample and compute conf_diff
        result: list[float] = []
        all_correct_confs: list[float] = []
        all_incorrect_confs: list[float] = []
        offset = 0
        for i in range(len(batch_prompt_lengths)):
            count = sample_budget_counts[i]
            if count == 0:
                result.append(0.0)
                continue

            sample_preds = preds[offset: offset + count]  # (count,)
            mc_accs = sample_mc_acc_lists[i]
            offset += count

            correct_confs = [
                sample_preds[j].item()
                for j in range(count)
                if mc_accs[j] == 1.0
            ]
            incorrect_confs = [
                sample_preds[j].item()
                for j in range(count)
                if mc_accs[j] == 0.0
            ]

            all_correct_confs.extend(correct_confs)
            all_incorrect_confs.extend(incorrect_confs)

            mu_conf_correct = float(np.mean(correct_confs)) if correct_confs else None
            mu_conf_incorrect = float(np.mean(incorrect_confs)) if incorrect_confs else None
            if mu_conf_correct is None and mu_conf_incorrect is None:
                result.append(0.0)
            elif mu_conf_correct is None or mu_conf_incorrect is None:
                if margin_corner:
                    # Restored corner handling for one-class traces:
                    # - only correct class present: conf_correct - 0
                    # - only incorrect class present:
                    #   * conf_diff: 1 - conf_incorrect
                    #   * conf_diff_v1: 0 - conf_incorrect
                    if self.reward_mode == "conf_diff_v3":
                        tau = self.conf_diff_v2_tau
                        if mu_conf_correct is not None:
                            result.append(
                                self.conf_diff_v3_alpha_only_correct
                                * (mu_conf_correct - tau)
                            )
                        elif mu_conf_incorrect is not None:
                            result.append(
                                self.conf_diff_v3_alpha_only_incorrect
                                * (tau - mu_conf_incorrect)
                            )
                    elif self.reward_mode == "conf_diff_v2":
                        tau = self.conf_diff_v2_tau
                        alpha = self.conf_diff_v2_alpha
                        if mu_conf_correct is not None:
                            result.append(alpha * (mu_conf_correct - tau))
                        elif mu_conf_incorrect is not None:
                            result.append(alpha * (tau - mu_conf_incorrect))
                    elif mu_conf_correct is not None:
                        # All correct: reward confidence directly (conf - 0).
                        result.append(mu_conf_correct)
                    else:
                        if self.reward_mode in ("conf_diff_v1", "conf_margin_covar"):
                            # v1 variant: use negative confidence for
                            # incorrect-only traces.
                            result.append(-mu_conf_incorrect)
                        else:
                            # Default conf_diff: reward low confidence (1 - conf).
                            result.append(1.0 - mu_conf_incorrect)
                else:
                    # With margin_corner disabled, one-class traces intentionally
                    # do not contribute conf_diff signal.
                    result.append(0.0)
            else:
                if self.reward_mode == "conf_diff_v3":
                    result.append(
                        self.conf_diff_v3_alpha_both
                        * (mu_conf_correct - mu_conf_incorrect)
                    )
                else:
                    result.append(mu_conf_correct - mu_conf_incorrect)

        conf_metrics = self._build_confidence_bucket_metrics(
            correct_preds=all_correct_confs,
            incorrect_preds=all_incorrect_confs,
        )

        return result, conf_metrics

    @staticmethod
    def compute_group_covar_factors(
        uids,
        natural_correct,
    ) -> np.ndarray:
        """Compute per-sample group factors ``p * (1 - p)``.

        Here ``p`` is the fraction of natural traces marked correct within a
        prompt group, where groups are defined by identical ``uid`` values.

        Args:
            uids: 1D uid array-like, one uid per sample.
            natural_correct: 1D bool-like array, one natural correctness value
                per sample.

        Returns:
            1D float array with the same length as the inputs.
        """
        uids_np = np.asarray(uids)
        natural_np = np.asarray(natural_correct).reshape(-1).astype(bool)

        if uids_np.ndim != 1:
            raise ValueError(f"uids must be 1D, got shape={uids_np.shape}")
        if natural_np.ndim != 1:
            raise ValueError(f"natural_correct must be 1D, got shape={natural_np.shape}")
        if uids_np.shape[0] != natural_np.shape[0]:
            raise ValueError(
                "uids and natural_correct must have the same length, "
                f"got {uids_np.shape[0]} and {natural_np.shape[0]}"
            )

        factors = np.zeros(uids_np.shape[0], dtype=np.float32)
        unique_uids, inv = np.unique(uids_np, return_inverse=True)
        for g in range(len(unique_uids)):
            group_mask = inv == g
            p = float(natural_np[group_mask].mean())
            factors[group_mask] = p * (1.0 - p)
        return factors

    @classmethod
    def apply_conf_margin_covar(
        cls,
        base_rewards: list[float],
        uids,
        natural_correct,
    ) -> tuple[list[float], np.ndarray]:
        """Apply the ``p * (1 - p)`` group factor to margin-style rewards."""
        factors = cls.compute_group_covar_factors(
            uids=uids,
            natural_correct=natural_correct,
        )
        if len(base_rewards) != len(factors):
            raise ValueError(
                "base_rewards and factors must have the same length, "
                f"got {len(base_rewards)} and {len(factors)}"
            )
        scaled_rewards = [float(r) * float(f) for r, f in zip(base_rewards, factors)]
        return scaled_rewards, factors

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "mlp": self.mlp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: dict):
        self.mlp.load_state_dict(state["mlp"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._step_count = state.get("step_count", 0)
