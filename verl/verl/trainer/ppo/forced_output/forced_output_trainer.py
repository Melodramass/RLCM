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
"""
Forced Output GRPO Trainer.

Subclasses ``RayPPOTrainer`` to insert a forced-output evaluation stage
between rollout generation and advantage computation:

1. Rollout is generated with ``stop_token_ids`` set to ``</think>`` so the
   model produces only a thinking trace.
2. For each trace the model is queried again (short generation) at each
   preset token budget to force an answer (e.g. ``\\boxed{...}``).
3. The per-budget rewards are aggregated (mean / linear-weighted) and used
   as the trace's reward for GRPO advantage computation.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections import defaultdict
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.forced_output.forced_output_utils import (
    aggregate_multi_answer_rewards,
    aggregate_rewards_batch,
    compute_unanimous_group_keep_mask,
    create_token_level_reward_tensor,
    resolve_budget_checkpoint,
)
from verl.trainer.ppo.metric_utils import compute_rlcr_metrics
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics

logger = logging.getLogger(__name__)


class RayForcedOutputGRPOTrainer(RayPPOTrainer):
    """GRPO trainer with forced-output budget evaluation.

    All forced-output parameters come from ``config.forced_output.*`` and are
    set via the training shell script (Hydra overrides).

    Config keys consumed (all under ``forced_output``):
        enable (bool): Master switch.
        budget_checkpoints (list[int]): E.g. ``[2000, 4000, 8000]``.
        reward_aggregation (str): ``"mean"`` or ``"linear_weighted"``.
        stop_token (str): Default ``"</think>"``.
        suffix_prompt (str): Suffix appended after ``</think>``.
        forced_max_tokens (int): Default ``128``.
        forced_temperature (float): Default ``0.1``.
        include_full_trace (bool): Default ``True``.
        include_natural_trace (bool): Default ``False``. If true, also score
            the naturally generated response trace and include it in aggregation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fo_cfg = self.config.get("forced_output", {})
        self.forced_output_enabled = bool(fo_cfg.get("enable", False))
        if self.forced_output_enabled:
            self.fo_reward_aggregation = str(fo_cfg.get("reward_aggregation", "mean"))
            self.fo_num_forced_answers = int(fo_cfg.get("num_forced_answers", 1))
            self.fo_multi_answer_agg = str(fo_cfg.get("multi_answer_aggregation", "mean"))
            self.fo_include_natural_trace = bool(fo_cfg.get("include_natural_trace", False))
            # Weight factor for budget trace rewards. Natural trace always
            # gets the raw 1/-1 reward; budget trace rewards are multiplied
            # by this factor before aggregation. Default 1.0 (no scaling).
            self.fo_budget_reward_weight = float(fo_cfg.get("budget_reward_weight", 1.0))
            # Store configured budget checkpoints for metric labeling
            self.fo_budget_checkpoints = list(fo_cfg.get("budget_checkpoints", []))

            # ── Probe configuration ──
            probe_cfg = fo_cfg.get("probe", {})
            self.probe_enabled = bool(probe_cfg.get("enable", False))
            self.probe_use_as_reward = bool(probe_cfg.get("use_as_reward", False))
            self.probe_reward_weight = float(probe_cfg.get("reward_weight", 0.0))
            self.probe_reward_mode = str(probe_cfg.get("reward_mode", "accuracy"))
            # Restored knob for conf_diff one-class traces (compatible with
            # previous scripts that set `forced_output.probe.margin_corner=true`).
            self.probe_margin_corner = bool(probe_cfg.get("margin_corner", False))
            self.probe_filter_natural_corner_groups = bool(
                probe_cfg.get("filter_natural_corner_groups", False)
            )
            self.probe_include_budget_reward = bool(fo_cfg.get("include_budget_reward", True))
            self.probe_trainer = None  # Lazily initialized in fit() after model config is available

            if self.probe_filter_natural_corner_groups and not self.probe_enabled:
                raise ValueError(
                    "forced_output.probe.filter_natural_corner_groups=True requires "
                    "forced_output.probe.enable=True"
                )
            if self.probe_filter_natural_corner_groups and not self.fo_include_natural_trace:
                raise ValueError(
                    "forced_output.probe.filter_natural_corner_groups=True requires "
                    "forced_output.include_natural_trace=True"
                )
            if self.probe_reward_mode == "ranking_reward" and not self.fo_include_natural_trace:
                raise ValueError(
                    "forced_output.probe.reward_mode=ranking_reward requires "
                    "forced_output.include_natural_trace=True"
                )

            # If include_budget_reward is False, zero out the budget reward weight
            # so budget scores are still computed (for MC accuracy / probe training)
            # but don't contribute to the GRPO reward.
            if not self.probe_include_budget_reward:
                logger.info("include_budget_reward=False: setting fo_budget_reward_weight=0")
                self.fo_budget_reward_weight = 0.0

            if self.probe_enabled:
                self._probe_cfg = probe_cfg
                logger.info(
                    "Probe enabled | train_steps=%d | batch_size=%d | lr=%s | "
                    "avg_last_k=%d | last_layer_idx=%d | use_as_reward=%s | reward_weight=%s | "
                    "weight_decay=%s | reward_mode=%s | ranking_reward_rt=%s | "
                    "conf_diff_v2_tau=%s | conf_diff_v2_alpha=%s | "
                    "conf_diff_v3_alpha_both=%s | conf_diff_v3_alpha_only_correct=%s | "
                    "conf_diff_v3_alpha_only_incorrect=%s | "
                    "margin_corner=%s | include_budget_reward=%s | "
                    "filter_natural_corner_groups=%s",
                    int(probe_cfg.get("train_steps", 20)),
                    int(probe_cfg.get("batch_size", 16)),
                    probe_cfg.get("lr", 1e-3),
                    int(probe_cfg.get("avg_last_k", 1)),
                    int(probe_cfg.get("last_layer_idx", 1)),
                    self.probe_use_as_reward,
                    self.probe_reward_weight,
                    float(probe_cfg.get("weight_decay", 0.01)),
                    self.probe_reward_mode,
                    float(probe_cfg.get("ranking_reward_rt", 0.5)),
                    float(probe_cfg.get("conf_diff_v2_tau", 0.5)),
                    float(probe_cfg.get("conf_diff_v2_alpha", 1.0)),
                    float(probe_cfg.get("conf_diff_v3_alpha_both", 1.0)),
                    float(probe_cfg.get("conf_diff_v3_alpha_only_correct", 1.0)),
                    float(probe_cfg.get("conf_diff_v3_alpha_only_incorrect", 1.0)),
                    self.probe_margin_corner,
                    self.probe_include_budget_reward,
                    self.probe_filter_natural_corner_groups,
                )

            logger.info(
                "ForcedOutputGRPO enabled | budgets=%s | aggregation=%s | "
                "num_forced_answers=%d | multi_answer_agg=%s | include_natural_trace=%s | "
                "budget_reward_weight=%s",
                fo_cfg.get("budget_checkpoints", []),
                self.fo_reward_aggregation,
                self.fo_num_forced_answers,
                self.fo_multi_answer_agg,
                self.fo_include_natural_trace,
                self.fo_budget_reward_weight,
            )

    def _get_local_global_step_folder(self, global_step: Optional[int] = None) -> str:
        step = self.global_steps if global_step is None else global_step
        return os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

    def _get_probe_checkpoint_path(self, global_step: Optional[int] = None) -> str:
        return os.path.join(self._get_local_global_step_folder(global_step), "probe.pt")

    def _try_load_probe_checkpoint(self):
        if not (self.forced_output_enabled and self.probe_enabled and self.probe_trainer is not None):
            return

        probe_path = self._get_probe_checkpoint_path()
        if not os.path.exists(probe_path):
            logger.info("No probe checkpoint found at %s; starting probe from scratch.", probe_path)
            return

        probe_state = torch.load(probe_path, map_location="cpu")
        self.probe_trainer.load_state_dict(probe_state)
        logger.info("Loaded probe checkpoint from %s", probe_path)

    def _save_checkpoint(self):
        super()._save_checkpoint()

        if not (self.forced_output_enabled and self.probe_enabled and self.probe_trainer is not None):
            return

        probe_path = self._get_probe_checkpoint_path()
        os.makedirs(os.path.dirname(probe_path), exist_ok=True)
        torch.save(self.probe_trainer.state_dict(), probe_path)
        logger.info("Saved probe checkpoint to %s", probe_path)

    def _safe_compute_score(
        self,
        reward_fn,
        *,
        data_source: str,
        solution_str: str,
        ground_truth: Any,
        extra_info: dict[str, Any],
        sample_idx: int,
        reward_component: str,
    ) -> tuple[float, Optional[dict[str, Any]]]:
        """Run ``reward_fn.compute_score`` and normalize to a float score."""
        try:
            score_result = reward_fn.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            if isinstance(score_result, dict):
                return float(score_result.get("score", 0.0)), score_result
            return float(score_result), None
        except Exception as e:
            logger.warning(
                "Reward computation failed for sample %d (%s): %s",
                sample_idx,
                reward_component,
                e,
            )
            return 0.0, None

    def _compute_natural_trace_reward(self, item: Any, reward_fn, sample_idx: int) -> tuple[float, dict[str, Any]]:
        """Compute reward from the naturally generated trace in ``batch['responses']``."""
        ground_truth = item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
        data_source = item.non_tensor_batch.get("data_source", "unknown")
        extra_info = item.non_tensor_batch.get("extra_info", {})

        prompt_length = item.batch["prompts"].shape[-1]
        valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum().item())
        trace_length_from_loop = int(item.non_tensor_batch.get("trace_length", valid_response_length))
        natural_trace_length = max(min(trace_length_from_loop, valid_response_length), 0)

        if natural_trace_length <= 0:
            return 0.0, {
                "source": "natural_trace",
                "reward": 0.0,
                "trace_length": natural_trace_length,
            }

        response_ids = item.batch["responses"][:natural_trace_length]
        natural_solution_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        reward_val, score_result = self._safe_compute_score(
            reward_fn,
            data_source=data_source,
            solution_str=natural_solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            sample_idx=sample_idx,
            reward_component="natural_trace",
        )
        detail = {
            "source": "natural_trace",
            "reward": reward_val,
            "trace_length": natural_trace_length,
        }
        if isinstance(score_result, dict):
            if "acc" in score_result:
                detail["acc"] = bool(score_result["acc"])
            if "pred" in score_result:
                detail["pred"] = str(score_result["pred"])
        return reward_val, detail

    # ------------------------------------------------------------------
    # Forced-output reward computation
    # ------------------------------------------------------------------

    def _compute_forced_output_rewards(
        self,
        batch: DataProto,
        reward_fn,
    ) -> tuple[torch.Tensor, dict[str, list], dict[int, list[float]]]:
        """Compute rewards for each trace by evaluating forced outputs at each budget.

        For each sample in ``batch``, the ``ForcedOutputAgentLoop`` has already
        stored per-budget texts in ``non_tensor_batch["forced_full_texts"]``
        (single answer) and optionally ``non_tensor_batch["forced_full_texts_multi"]``
        (multiple answers per budget).

        When ``forced_full_texts_multi`` is present and ``num_forced_answers > 1``,
        we score all answers at each budget, aggregate them using
        ``fo_multi_answer_agg`` (mean / accuracy / max), then aggregate across
        budgets using ``fo_reward_aggregation`` (mean / linear_weighted).

        If ``forced_output.include_natural_trace=True``, we also score the
        naturally generated response trace and include it as an additional
        reward component in the same aggregation.

        Args:
            batch: The full training batch (after rollout + repeat).
            reward_fn: The reward manager (same one used for normal GRPO).

        Returns:
            reward_tensor: ``(batch_size, response_length)`` token-level reward.
            reward_extra_infos_dict: Extra info dict for logging.
            per_budget_reward_lists: Per-budget reward lists for metric logging.

        Side effects:
            When ``probe_enabled`` is True, stores per-sample Monte Carlo
            accuracy and budget position lists in ``batch.non_tensor_batch``
            under keys ``probe_mc_accuracies`` and ``probe_budget_positions``.
        """
        batch_size = batch.batch["responses"].shape[0]
        response_length = batch.batch["responses"].shape[1]
        response_mask = batch.batch.get("response_mask", None)

        rewards_matrix: list[list[float]] = []
        extra_infos: dict[str, list] = defaultdict(list)
        # Per-budget rewards tracked separately (not stored in non_tensor_batch)
        # to avoid batch-size mismatch when some samples lack forced outputs.
        # Keyed by configured budget checkpoint value (not clamped per-sample value).
        per_budget_reward_lists: dict[int, list[float]] = defaultdict(list)

        # Track per-sample MC accuracies and budget positions for probe training
        per_sample_mc_accuracies: list[list[float]] = []
        per_sample_budget_positions: list[list[int]] = []

        # Build mapping from clamped truncation length -> configured checkpoint
        # so metrics use fixed names like reward_budget_2000 instead of per-sample values.
        configured_budgets = self.fo_budget_checkpoints

        for i in range(batch_size):
            item = batch[i]

            # Per-budget forced full texts placed by ForcedOutputAgentLoop
            forced_full_texts = item.non_tensor_batch.get("forced_full_texts", None)
            forced_full_texts_multi = item.non_tensor_batch.get("forced_full_texts_multi", None)
            budget_checkpoints = item.non_tensor_batch.get("budget_checkpoints", None)
            has_forced_outputs = forced_full_texts is not None and budget_checkpoints is not None
            if not has_forced_outputs:
                logger.warning("Sample %d missing forced_full_texts; using available reward components only.", i)
                forced_full_texts = {}
                forced_full_texts_multi = None
                budget_checkpoints = []

            # Get ground truth and data source from the sample
            ground_truth = item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
            data_source = item.non_tensor_batch.get("data_source", "unknown")
            extra_info_item = item.non_tensor_batch.get("extra_info", {})

            per_budget_rewards = []
            per_budget_details = []
            # Map from clamped truncation length -> reward for later metric attribution
            clamped_to_reward: dict[int, float] = {}
            # MC accuracy per budget (for probe training)
            sample_mc_accs: list[float] = []
            sample_budget_pos: list[int] = list(budget_checkpoints)

            use_multi = (
                forced_full_texts_multi is not None
                and self.fo_num_forced_answers > 1
            )

            for budget in budget_checkpoints:
                if use_multi:
                    # Multi-answer: score all answers at this budget
                    answer_texts = forced_full_texts_multi.get(budget, [])
                    if not answer_texts:
                        answer_texts = [forced_full_texts.get(budget, "")]

                    per_answer_rewards = []
                    for ans_idx, solution_str in enumerate(answer_texts):
                        reward_val, _ = self._safe_compute_score(
                            reward_fn,
                            data_source=data_source,
                            solution_str=solution_str,
                            ground_truth=ground_truth,
                            extra_info=extra_info_item,
                            sample_idx=i,
                            reward_component=f"budget={budget}/answer={ans_idx}",
                        )
                        per_answer_rewards.append(reward_val)

                    # Aggregate across answers at this budget
                    budget_reward = aggregate_multi_answer_rewards(
                        per_answer_rewards, method=self.fo_multi_answer_agg
                    )
                    # MC accuracy = fraction of correct answers (reward > 0)
                    mc_acc = float(np.mean([1.0 if r > 0 else 0.0 for r in per_answer_rewards]))
                    sample_mc_accs.append(mc_acc)
                    # Scale budget trace reward by the configured weight
                    weighted_budget_reward = budget_reward * self.fo_budget_reward_weight
                    per_budget_rewards.append(weighted_budget_reward)
                    per_budget_details.append({
                        "budget": budget,
                        "reward": weighted_budget_reward,
                        "raw_reward": budget_reward,
                        "budget_reward_weight": self.fo_budget_reward_weight,
                        "per_answer_rewards": per_answer_rewards,
                        "num_answers": len(per_answer_rewards),
                        "multi_answer_agg": self.fo_multi_answer_agg,
                    })
                else:
                    # Single-answer (original path)
                    solution_str = forced_full_texts.get(budget, "")
                    reward_val, _ = self._safe_compute_score(
                        reward_fn,
                        data_source=data_source,
                        solution_str=solution_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info_item,
                        sample_idx=i,
                        reward_component=f"budget={budget}",
                    )

                    # Scale budget trace reward by the configured weight
                    weighted_reward_val = reward_val * self.fo_budget_reward_weight
                    # MC accuracy for single answer: 1 if correct, 0 otherwise
                    mc_acc = 1.0 if reward_val > 0 else 0.0
                    sample_mc_accs.append(mc_acc)
                    per_budget_rewards.append(weighted_reward_val)
                    per_budget_details.append(
                        {"source": "forced_budget", "budget": budget, "reward": weighted_reward_val,
                         "raw_reward": reward_val, "budget_reward_weight": self.fo_budget_reward_weight}
                    )

                # Record clamped budget -> reward for metric attribution below
                clamped_to_reward[budget] = weighted_budget_reward if use_multi else weighted_reward_val

            # Attribute each clamped budget's reward to the configured budget
            # checkpoints. Relative checkpoints (floats in (0, 1]) are first
            # resolved against the sample's trace length.
            trace_length = int(item.non_tensor_batch.get("trace_length", max(budget_checkpoints, default=0)))
            for cfg_budget in configured_budgets:
                resolved_budget = resolve_budget_checkpoint(trace_length, cfg_budget)
                if resolved_budget in clamped_to_reward:
                    per_budget_reward_lists[cfg_budget].append(clamped_to_reward[resolved_budget])

            natural_reward: Optional[float] = None
            if self.fo_include_natural_trace:
                natural_reward, natural_detail = self._compute_natural_trace_reward(item, reward_fn, sample_idx=i)
                # Natural reward is NOT appended to per_budget_rewards.
                # Instead it will be added on top of the aggregated budget
                # rewards after aggregation (natural_reward + agg(budget_rewards)).
                per_budget_details.append(natural_detail)
            extra_infos["forced_output_natural_reward"].append(
                float(natural_reward) if natural_reward is not None else np.nan
            )
            # Track whether the natural trace alone is correct (for solved_all/solved_none)
            extra_infos["forced_output_natural_correct"].append(
                bool(natural_reward is not None and natural_reward > 0)
            )

            if not per_budget_rewards:
                logger.warning("Sample %d produced no reward components; falling back to 0 reward.", i)
                per_budget_rewards = [0.0]
                per_budget_details.append({"source": "fallback_zero", "reward": 0.0})

            rewards_matrix.append(per_budget_rewards)
            extra_infos["forced_output_any_correct"].append(
                any(r > 0 for r in per_budget_rewards) or (natural_reward is not None and natural_reward > 0)
            )
            extra_infos["forced_output_fallback"].append(not has_forced_outputs)
            extra_infos["forced_output_num_components"].append(len(per_budget_rewards))
            extra_infos["forced_output_details"].append(json.dumps(per_budget_details))
            extra_infos["forced_output_per_budget_mean"].append(float(np.mean(per_budget_rewards)))

            # Store MC accuracies for probe training
            per_sample_mc_accuracies.append(sample_mc_accs)
            per_sample_budget_positions.append(sample_budget_pos)

        # Aggregate across budgets for each trace
        aggregated_rewards = aggregate_rewards_batch(
            rewards_matrix, self.fo_reward_aggregation
        )

        # Add natural reward on top of the aggregated budget rewards
        # (i.e. natural_reward + agg(budget_rewards)) instead of averaging
        # them together.
        if self.fo_include_natural_trace:
            natural_rewards_list = extra_infos["forced_output_natural_reward"]
            aggregated_rewards = [
                agg + (nat if not np.isnan(nat) else 0.0)
                for agg, nat in zip(aggregated_rewards, natural_rewards_list)
            ]

        extra_infos["forced_output_aggregated_reward"] = aggregated_rewards

        # Store MC accuracies and budget positions for probe training
        extra_infos["probe_mc_accuracies"] = per_sample_mc_accuracies
        extra_infos["probe_budget_positions"] = per_sample_budget_positions

        # Build token-level reward tensor (place reward on last valid token)
        reward_tensor = create_token_level_reward_tensor(
            aggregated_rewards=aggregated_rewards,
            response_length=response_length,
            response_mask=response_mask,
        )

        return reward_tensor, dict(extra_infos), per_budget_reward_lists

    # ------------------------------------------------------------------
    # Override: validation scoring prefers natural trace output
    # ------------------------------------------------------------------

    def _compute_or_extract_reward(self, batch, reward_fn=None, reward_for_val=False, sum_reward=False):
        """Override validation scoring to prefer natural traces.

        During training (``reward_for_val=False``), use the base implementation
        unchanged. During validation for forced-output experiments, score the
        natural trace text by default. Only when the natural trace is absent do
        we fall back to the last-budget forced output.
        """
        if reward_for_val and self.forced_output_enabled and len(batch) > 0:
            return self._compute_val_reward_preferring_natural_trace(batch, reward_fn, sum_reward)
        return super()._compute_or_extract_reward(batch, reward_fn, reward_for_val, sum_reward)

    def _select_validation_solution_text(self, item: Any, sample_idx: int) -> tuple[str, str]:
        """Pick text used for validation scoring.

        Returns:
            ``(solution_str, source)`` where source is one of:
            - ``"natural"``: natural trace text is present and used.
            - ``"forced_fallback"``: natural trace is absent, fallback to
              last-budget forced output.
            - ``"missing"``: neither natural nor forced text is available.
        """
        prompt_length = item.batch["prompts"].shape[-1]
        valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum().item())
        trace_length_from_loop = int(item.non_tensor_batch.get("trace_length", valid_response_length))
        natural_trace_length = max(min(trace_length_from_loop, valid_response_length), 0)

        if natural_trace_length > 0:
            response_ids = item.batch["responses"][:natural_trace_length]
            natural_solution_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            if natural_solution_str.strip():
                return natural_solution_str, "natural"

        forced_full_texts = item.non_tensor_batch.get("forced_full_texts", None)
        budget_checkpoints = item.non_tensor_batch.get("budget_checkpoints", None)
        if forced_full_texts is not None and budget_checkpoints:
            last_budget = max(budget_checkpoints)
            return forced_full_texts.get(last_budget, ""), "forced_fallback"

        logger.warning("Val sample %d missing both natural trace and forced_full_texts; scoring as 0.", sample_idx)
        return "", "missing"

    def _compute_val_reward_preferring_natural_trace(self, batch, reward_fn, sum_reward=False):
        """Score validation samples on natural traces, with forced fallback.

        Validation scoring source policy:
        1. Use natural trace output whenever it exists.
        2. If natural trace is absent, use last-budget forced output.
        """
        batch_size = batch.batch["responses"].shape[0]
        response_length = batch.batch["responses"].shape[1]
        reward_tensor = torch.zeros(batch_size, response_length, dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        for i in range(batch_size):
            item = batch[i]

            # Determine where to place the scalar reward (last valid response token)
            prompt_length = item.batch["prompts"].shape[-1]
            valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum().item())
            place_idx = max(min(valid_response_length - 1, response_length - 1), 0)

            solution_str, solution_source = self._select_validation_solution_text(item, i)
            if solution_source == "missing":
                reward_extra_info["score"].append(0.0)
                continue

            ground_truth = item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
            data_source = item.non_tensor_batch.get("data_source", "unknown")
            extra_info = item.non_tensor_batch.get("extra_info", {})

            score, score_result = self._safe_compute_score(
                reward_fn,
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                sample_idx=i,
                reward_component=f"validation_{solution_source}",
            )
            if isinstance(score_result, dict):
                for k, v in score_result.items():
                    reward_extra_info[k].append(v)
                if "score" not in score_result:
                    reward_extra_info["score"].append(score)
            else:
                reward_extra_info["score"].append(score)

            reward_tensor[i, place_idx] = score

        if sum_reward:
            reward_tensor = reward_tensor.sum(dim=-1)

        return reward_tensor, dict(reward_extra_info)

    # ------------------------------------------------------------------
    # Override: inject forced-output into the training loop
    # ------------------------------------------------------------------

    def fit(self):
        """Training loop identical to ``RayPPOTrainer.fit()`` except that when
        ``forced_output.enable`` is ``True`` the reward used for GRPO advantage
        is computed from forced-output budget evaluation instead of the normal
        end-of-sequence reward.
        """
        from omegaconf import OmegaConf
        from tqdm import tqdm

        from verl.trainer.ppo.core_algos import agg_loss
        from verl.trainer.ppo.ray_trainer import apply_kl_penalty
        from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
        from verl.utils.metric import reduce_metrics
        from verl.utils.rollout_skip import RolloutSkip
        from verl.utils.tracking import Tracking

        logger_tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # ── Initialize probe trainer if enabled ──
        if self.forced_output_enabled and self.probe_enabled and self.probe_trainer is None:
            from transformers import AutoConfig
            from verl.trainer.ppo.forced_output.probe_trainer import BudgetProbeTrainer

            model_config = AutoConfig.from_pretrained(
                self.config.actor_rollout_ref.model.path, trust_remote_code=True,
            )
            probe_cfg = self._probe_cfg
            self.probe_trainer = BudgetProbeTrainer(
                hidden_size=model_config.hidden_size,
                train_steps=int(probe_cfg.get("train_steps", 20)),
                batch_size=int(probe_cfg.get("batch_size", 16)),
                lr=float(probe_cfg.get("lr", 1e-3)),
                dropout=float(probe_cfg.get("dropout", 0.1)),
                avg_last_k=int(probe_cfg.get("avg_last_k", 1)),
                last_layer_idx=int(probe_cfg.get("last_layer_idx", 1)),
                reward_weight=self.probe_reward_weight,
                use_as_reward=self.probe_use_as_reward,
                margin_corner=self.probe_margin_corner,
                weight_decay=float(probe_cfg.get("weight_decay", 0.01)),
                device=str(probe_cfg.get("device", "auto")),
                reward_mode=self.probe_reward_mode,
                ranking_reward_rt=float(probe_cfg.get("ranking_reward_rt", 0.5)),
                conf_diff_v2_tau=float(probe_cfg.get("conf_diff_v2_tau", 0.5)),
                conf_diff_v2_alpha=float(probe_cfg.get("conf_diff_v2_alpha", 1.0)),
                conf_diff_v3_alpha_both=float(probe_cfg.get("conf_diff_v3_alpha_both", 1.0)),
                conf_diff_v3_alpha_only_correct=float(
                    probe_cfg.get("conf_diff_v3_alpha_only_correct", 1.0)
                ),
                conf_diff_v3_alpha_only_incorrect=float(
                    probe_cfg.get("conf_diff_v3_alpha_only_incorrect", 1.0)
                ),
            )
            logger.info(
                "Initialized BudgetProbeTrainer with hidden_size=%d",
                model_config.hidden_size,
            )
            self._try_load_probe_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger_tracker.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    natural_corner_keep_mask = None

                    # ── Phase 1: Rollout generation ──
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile(global_step=self.global_steps)
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                    # repeat prompt batch to align with rollout responses
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance batch across DP ranks
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all

                    # ── Phase 2: Reward computation ──
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # Compute reward model score if needed
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # --- Forced Output reward path ---
                        if self.forced_output_enabled and self.reward_fn is not None:
                            with marked_timer("forced_output_reward", timing_raw, color="magenta"):
                                reward_tensor, reward_extra_infos_dict, per_budget_reward_lists = (
                                    self._compute_forced_output_rewards(batch, self.reward_fn)
                                )
                                fo_mean = float(np.mean(
                                    reward_extra_infos_dict.get("forced_output_aggregated_reward", [0.0])
                                ))
                                metrics["forced_output/aggregated_reward_mean"] = fo_mean

                                # Log per-budget and natural trace reward means
                                for budget_val, vals in per_budget_reward_lists.items():
                                    if vals:
                                        metrics[f"forced_output/reward_budget_{budget_val}_mean"] = float(np.mean(vals))
                                natural_rewards = reward_extra_infos_dict.get("forced_output_natural_reward", [])
                                valid_natural = [r for r in natural_rewards if not np.isnan(r)]
                                if valid_natural:
                                    metrics["forced_output/natural_trace_reward_mean"] = float(np.mean(valid_natural))

                                if self.probe_filter_natural_corner_groups:
                                    if "uid" not in batch.non_tensor_batch:
                                        raise ValueError(
                                            "uid not found in batch.non_tensor_batch while "
                                            "forced_output.probe.filter_natural_corner_groups=True"
                                        )
                                    if "forced_output_natural_correct" not in reward_extra_infos_dict:
                                        raise ValueError(
                                            "forced_output_natural_correct not found while "
                                            "forced_output.probe.filter_natural_corner_groups=True"
                                        )

                                    natural_corner_keep_mask, filter_stats = compute_unanimous_group_keep_mask(
                                        uids=batch.non_tensor_batch["uid"],
                                        natural_correct=reward_extra_infos_dict["forced_output_natural_correct"],
                                    )

                                    metrics["probe_filter/groups_total"] = filter_stats["groups_total"]
                                    metrics["probe_filter/groups_kept_mixed"] = filter_stats["groups_kept_mixed"]
                                    metrics["probe_filter/groups_dropped_all_correct"] = (
                                        filter_stats["groups_dropped_all_correct"]
                                    )
                                    metrics["probe_filter/groups_dropped_all_incorrect"] = (
                                        filter_stats["groups_dropped_all_incorrect"]
                                    )
                                    metrics["probe_filter/samples_dropped"] = filter_stats["samples_dropped"]
                                    metrics["probe_filter/samples_kept"] = filter_stats["samples_kept"]
                                    if not natural_corner_keep_mask.any():
                                        metrics["probe_filter/fallback_unfiltered_step"] = 1
                                        natural_corner_keep_mask = None
                                        logger.warning(
                                            "All groups were unanimous under natural-trace filter; "
                                            "falling back to unfiltered rewards for this step."
                                        )
                                    else:
                                        metrics["probe_filter/fallback_unfiltered_step"] = 0
                        else:
                            # Standard reward computation
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, reward_for_val=False
                            )
                            metrics.update(compute_rlcr_metrics(reward_extra_infos_dict))

                    # ── Phase 3: Log probs, advantage, actor update ──
                    # Recompute old_log_probs
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = (
                        rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    )
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        # Request hidden states if probe is enabled
                        if self.forced_output_enabled and self.probe_enabled and self.probe_trainer is not None:
                            batch.meta_info["return_hidden_states"] = True
                            batch.meta_info["hidden_states_layer_idx"] = self.probe_trainer.last_layer_idx

                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                router_mode = getattr(
                                    self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                )
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "old_log_probs" in batch.batch and "response_mask" in batch.batch:
                                response_mask_f = batch.batch["response_mask"].to(dtype=batch.batch["old_log_probs"].dtype)
                                token_count = torch.clamp(response_mask_f.sum(dim=-1), min=1.0)
                                old_sequence_log_confidence = (
                                    (batch.batch["old_log_probs"] * response_mask_f).sum(dim=-1) / token_count
                                )
                                old_sequence_confidence = torch.exp(old_sequence_log_confidence)
                                batch.batch["old_sequence_log_confidence"] = old_sequence_log_confidence
                                batch.batch["old_sequence_confidence"] = old_sequence_confidence
                                metrics["c2/old_sequence_confidence_mean"] = (
                                    old_sequence_confidence.detach().mean().item()
                                )
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics
                                metrics.update(calculate_debug_metrics(batch))

                    # ── Probe: train & compute probe reward ──
                    if (
                        self.forced_output_enabled
                        and self.probe_enabled
                        and self.probe_trainer is not None
                        and "hidden_states" in batch.batch
                    ):
                        with marked_timer("probe", timing_raw, color="purple"):
                            probe_mc_accs = reward_extra_infos_dict.get("probe_mc_accuracies", [])
                            probe_budget_pos = reward_extra_infos_dict.get("probe_budget_positions", [])

                            # Compute prompt lengths for each sample
                            prompt_length_per_sample = []
                            batch_size_probe = batch.batch["responses"].shape[0]
                            total_seq_len = batch.batch["attention_mask"].shape[1]
                            response_len = batch.batch["responses"].shape[1]
                            prompt_len_global = total_seq_len - response_len
                            for idx in range(batch_size_probe):
                                prompt_length_per_sample.append(prompt_len_global)

                            # Collect probe training data
                            hidden_states_cpu = batch.batch["hidden_states"]  # (bs, seq, hidden)
                            probe_features, probe_targets = self.probe_trainer.collect_probe_data(
                                batch_hidden_states=hidden_states_cpu,
                                batch_prompt_lengths=prompt_length_per_sample,
                                batch_budget_positions=probe_budget_pos,
                                batch_mc_accuracies=probe_mc_accs,
                            )

                            # Train probe
                            if probe_features.shape[0] > 0:
                                probe_metrics = self.probe_trainer.train_step(
                                    probe_features, probe_targets,
                                )
                                metrics.update(probe_metrics)

                                # Log mean MC accuracy across all budgets
                                metrics["probe/mc_accuracy_mean"] = probe_targets.mean().item()

                            # Compute probe reward if enabled
                            if self.probe_use_as_reward and self.probe_reward_weight > 0:
                                if self.probe_reward_mode in (
                                    "conf_diff",
                                    "conf_diff_v1",
                                    "conf_diff_v2",
                                    "conf_diff_v3",
                                    "conf_margin_covar",
                                ):
                                    # conf_diff mode: per-sample probe contribution =
                                    #   reward_weight × (mu_conf_correct - mu_conf_incorrect)
                                    # r_natural is excluded; set include_natural_trace=False
                                    # so the base reward_tensor contains only budget rewards.
                                    conf_diff_list, conf_metrics = self.probe_trainer.compute_conf_diff_rewards(
                                        batch_hidden_states=hidden_states_cpu,
                                        batch_prompt_lengths=prompt_length_per_sample,
                                        batch_budget_positions=probe_budget_pos,
                                        batch_mc_accuracies=probe_mc_accs,
                                        margin_corner=self.probe_margin_corner,
                                    )
                                    metrics.update(conf_metrics)
                                    probe_reward_scalars = [float(cd) for cd in conf_diff_list]
                                    if self.probe_reward_mode == "conf_margin_covar":
                                        natural_correct = reward_extra_infos_dict.get("forced_output_natural_correct")
                                        if natural_correct is None:
                                            raise ValueError(
                                                "forced_output_natural_correct is required for "
                                                "probe reward mode 'conf_margin_covar'"
                                            )
                                        probe_reward_scalars, covar_factors = (
                                            self.probe_trainer.apply_conf_margin_covar(
                                                base_rewards=probe_reward_scalars,
                                                uids=batch.non_tensor_batch["uid"],
                                                natural_correct=natural_correct,
                                            )
                                        )
                                        metrics["probe/group_covar_mean"] = float(np.mean(covar_factors))
                                        metrics["probe/group_covar_max"] = float(np.max(covar_factors))
                                        metrics["probe/group_covar_min"] = float(np.min(covar_factors))
                                    if natural_corner_keep_mask is not None:
                                        # Interaction rule with dynamic filtering:
                                        # filtered unanimous natural groups must not receive
                                        # conf_diff auxiliary reward even when margin_corner=True.
                                        for idx, keep in enumerate(natural_corner_keep_mask):
                                            if not keep:
                                                probe_reward_scalars[idx] = 0.0
                                    metrics["probe/margin_reward_mean"] = float(np.mean(probe_reward_scalars))
                                elif self.probe_reward_mode == "ranking_reward":
                                    natural_correct = reward_extra_infos_dict.get("forced_output_natural_correct")
                                    if natural_correct is None:
                                        raise ValueError(
                                            "forced_output_natural_correct is required for "
                                            "probe reward mode 'ranking_reward'"
                                        )
                                    ranking_reward_list, ranking_metrics = (
                                        self.probe_trainer.compute_ranking_rewards(
                                            batch_hidden_states=hidden_states_cpu,
                                            batch_prompt_lengths=prompt_length_per_sample,
                                            batch_budget_positions=probe_budget_pos,
                                            natural_correct=natural_correct,
                                        )
                                    )
                                    metrics.update(ranking_metrics)
                                    probe_reward_scalars = [float(rr) for rr in ranking_reward_list]
                                    if natural_corner_keep_mask is not None:
                                        for idx, keep in enumerate(natural_corner_keep_mask):
                                            if not keep:
                                                probe_reward_scalars[idx] = 0.0
                                    metrics["probe/ranking_reward_mean"] = float(np.mean(probe_reward_scalars))
                                else:
                                    probe_rewards_per_sample = self.probe_trainer.compute_probe_rewards(
                                        batch_hidden_states=hidden_states_cpu,
                                        batch_prompt_lengths=prompt_length_per_sample,
                                        batch_budget_positions=probe_budget_pos,
                                        batch_mc_accuracies=probe_mc_accs,
                                    )
                                    # probe_rewards_per_sample[i] = list of per-budget predicted accs
                                    # Average across budgets to get per-sample probe reward
                                    probe_reward_scalars = []
                                    for pr in probe_rewards_per_sample:
                                        if pr:
                                            probe_reward_scalars.append(float(np.mean(pr)))
                                        else:
                                            probe_reward_scalars.append(0.0)

                                # Add weighted probe reward to the reward tensor
                                # reward_tensor shape: (batch_size, response_length)
                                probe_reward_contribution = torch.zeros_like(reward_tensor)
                                response_mask_tensor = batch.batch.get("response_mask", None)
                                for idx in range(batch_size_probe):
                                    pr_val = probe_reward_scalars[idx] * self.probe_reward_weight
                                    # Place on last valid response token
                                    if response_mask_tensor is not None:
                                        valid_len = int(response_mask_tensor[idx].sum().item())
                                        place_pos = max(min(valid_len - 1, response_len - 1), 0)
                                    else:
                                        place_pos = response_len - 1
                                    probe_reward_contribution[idx, place_pos] = pr_val

                                reward_tensor = reward_tensor + probe_reward_contribution
                                metrics["probe/reward_mean"] = float(np.mean(probe_reward_scalars))

                            # Free hidden states to save memory
                            batch.batch.pop("hidden_states", None)

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        with marked_timer(str("ref_policy"), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        if natural_corner_keep_mask is not None:
                            keep_mask_t = torch.as_tensor(
                                natural_corner_keep_mask,
                                device=reward_tensor.device,
                                dtype=torch.bool,
                            )
                            reward_tensor = reward_tensor.clone()
                            # Final guardrail for filtered groups: zero all token-level
                            # rewards so these samples produce zero GRPO advantage.
                            reward_tensor[~keep_mask_t] = 0.0

                        # Set the reward tensor on batch
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            # Remove probe-only keys before storing in non_tensor_batch
                            # (they are lists of lists which don't convert cleanly to numpy)
                            store_dict = {
                                k: np.array(v)
                                for k, v in reward_extra_infos_dict.items()
                                if k not in ("probe_mc_accuracies", "probe_budget_positions")
                            }
                            batch.non_tensor_batch.update(store_dict)

                        # Apply KL penalty if configured
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Rollout correction (decoupled mode)
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import (
                                compute_rollout_correction_and_add_to_batch,
                            )
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                batch, rollout_corr_config
                            )
                            metrics.update(is_metrics)

                        # Build C2GSPG binary sequence labels (0/1) when enabled.
                        effective_adv_estimator = self.config.algorithm.adv_estimator
                        if str(effective_adv_estimator) == "c2gspg":
                            c2_binary_reward = None
                            natural_correct = None
                            if reward_extra_infos_dict:
                                natural_correct = reward_extra_infos_dict.get("forced_output_natural_correct")

                            if natural_correct is not None:
                                natural_correct_np = np.asarray(natural_correct).reshape(-1)
                                if natural_correct_np.shape[0] == reward_tensor.shape[0]:
                                    c2_binary_reward = torch.as_tensor(
                                        natural_correct_np,
                                        dtype=torch.float32,
                                        device=reward_tensor.device,
                                    )

                            if c2_binary_reward is None and "forced_output_natural_correct" in batch.non_tensor_batch:
                                natural_correct_np = np.asarray(batch.non_tensor_batch["forced_output_natural_correct"]).reshape(-1)
                                if natural_correct_np.shape[0] == reward_tensor.shape[0]:
                                    c2_binary_reward = torch.as_tensor(
                                        natural_correct_np,
                                        dtype=torch.float32,
                                        device=reward_tensor.device,
                                    )

                            if c2_binary_reward is None:
                                seq_rewards = batch.batch["token_level_rewards"].sum(dim=-1)
                                c2_binary_reward = (seq_rewards > 0).to(dtype=torch.float32)
                                metrics["c2/label_fallback_from_scores"] = 1
                            else:
                                metrics["c2/label_fallback_from_scores"] = 0

                            batch.batch["c2_binary_reward"] = c2_binary_reward
                            metrics["c2/binary_reward_mean"] = c2_binary_reward.detach().mean().item()

                            missing_fields = [
                                key for key in ("old_sequence_confidence", "c2_binary_reward") if key not in batch.batch
                            ]
                            if missing_fields:
                                effective_adv_estimator = "grpo"
                                metrics["c2/fallback_to_grpo"] = 1
                                logger.warning(
                                    "C2GSPG missing fields %s. Falling back to GRPO for this step.",
                                    missing_fields,
                                )
                            else:
                                metrics["c2/fallback_to_grpo"] = 0

                            # Log confidence split by correct/incorrect
                            if "old_sequence_confidence" in batch.batch:
                                osc = batch.batch["old_sequence_confidence"].detach()
                                mask_correct = c2_binary_reward.bool()
                                mask_incorrect = ~mask_correct
                                if mask_correct.any():
                                    metrics["c2/old_sequence_confidence_correct"] = osc[mask_correct].mean().item()
                                if mask_incorrect.any():
                                    metrics["c2/old_sequence_confidence_incorrect"] = osc[mask_incorrect].mean().item()

                        # Compute advantages
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )
                        batch = compute_advantage(
                            batch,
                            adv_estimator=effective_adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # Update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # Update actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout data
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # Validation
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Checkpoint saving
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.save_freq == 0
                    or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self.checkpoint_manager.sleep_replicas()
                        self._save_checkpoint()
                        self.checkpoint_manager.update_weights()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # Metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                from verl.trainer.ppo.metric_utils import (
                    compute_data_metrics,
                    compute_solved_metrics,
                    compute_throughout_metrics,
                    compute_timing_metrics,
                    compute_variance_proxy_metrics,
                )

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_solved_metrics(batch=batch))

                # Compute solved_all / solved_none for the natural trace alone
                if (
                    self.forced_output_enabled
                    and "forced_output_natural_correct" in batch.non_tensor_batch
                    and "uid" in batch.non_tensor_batch
                ):
                    uids = batch.non_tensor_batch["uid"]
                    try:
                        uids_np = np.asarray(uids)
                    except Exception:
                        uids_np = np.array(list(uids), dtype=object)
                    nat_correct = batch.non_tensor_batch["forced_output_natural_correct"]
                    try:
                        nat_correct_np = np.asarray(nat_correct).reshape(-1).astype(bool)
                    except Exception:
                        nat_correct_np = np.array(list(nat_correct), dtype=bool)
                    if nat_correct_np.shape[0] == uids_np.shape[0]:
                        unique_uids, inv = np.unique(uids_np, return_inverse=True)
                        nat_solved_all = 0
                        nat_solved_none = 0
                        for g in range(len(unique_uids)):
                            mask = inv == g
                            if not mask.any():
                                continue
                            group = nat_correct_np[mask]
                            if np.all(group):
                                nat_solved_all += 1
                            elif not np.any(group):
                                nat_solved_none += 1
                        metrics["training/natural_trace_solved_all"] = int(nat_solved_all)
                        metrics["training/natural_trace_solved_none"] = int(nat_solved_none)
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
                )
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

                try:
                    from verl.experimental.dataset.sampler import AbstractCurriculumSampler
                    if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                        self.train_dataloader.sampler.update(batch=batch)
                except ImportError:
                    pass

                logger_tracker.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
