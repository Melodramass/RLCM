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
Forced Output Agent Loop.

A two-phase agent loop:
  1. Generate a thinking trace (stopping at </think>).
  2. For each budget checkpoint, truncate the trace, append a forced-output
     suffix, and do a short generation to produce a boxed answer.

The per-budget answers and their rewards are returned so that the trainer
can aggregate them into a single reward for GRPO advantage computation.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.trainer.ppo.forced_output.forced_output_utils import (
    build_forced_output_tokens,
    compute_budget_truncation_lengths,
)
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("forced_output_agent")
class ForcedOutputAgentLoop(AgentLoopBase):
    """Agent loop that generates a thinking trace then forces answer output at preset budgets.

    Configuration is read from ``config.forced_output`` (an OmegaConf dict) which
    must contain:
        stop_token (str): Token string to stop thinking generation (default "</think>").
        budget_checkpoints (list[int]): Token-budget positions (e.g. [2000, 4000, 8000]).
        suffix_prompt (str): Text appended after truncation + </think>.
        forced_max_tokens (int): Max tokens for the short forced generation.
        forced_temperature (float): Temperature for forced generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

        # Read forced-output config
        fo_config = self.config.forced_output
        self.stop_token = fo_config.get("stop_token", "</think>")
        self.budget_checkpoints = list(fo_config.budget_checkpoints)
        self.suffix_prompt = fo_config.get(
            "suffix_prompt",
            "\nIf I were to output the final answer now, it would be \\boxed{",
        )
        self.forced_max_tokens = int(fo_config.get("forced_max_tokens", 128))
        self.forced_temperature = float(fo_config.get("forced_temperature", 0.1))
        self.include_full_trace = bool(fo_config.get("include_full_trace", True))
        self.num_forced_answers = int(fo_config.get("num_forced_answers", 1))
        self.train_on_forced_output = bool(fo_config.get("train_on_forced_output", False))

        # Pre-tokenize stop token and suffix (without special tokens)
        self.stop_token_ids = self.tokenizer.encode(self.stop_token, add_special_tokens=False)
        self.think_end_token_id = self.stop_token_ids[-1] if self.stop_token_ids else self.tokenizer.eos_token_id
        self.suffix_token_ids = self.tokenizer.encode(self.suffix_prompt, add_special_tokens=False)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Execute the two-phase forced output generation.

        Phase 1: Generate a thinking trace, stopping at </think>.
        Phase 2: For each budget checkpoint, truncate + append suffix + generate short answer.

        The output contains the original thinking trace as response_ids, plus
        per-budget forced answers stored in extra_fields for downstream reward computation.
        """
        messages = list(kwargs["raw_prompt"])

        # 1. Extract images/videos
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. Apply chat template
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )

        # ── Phase 1: Generate thinking trace with stop at </think> ──
        metrics = {}
        phase1_params = dict(sampling_params)  # copy
        phase1_params["stop_token_ids"] = self.stop_token_ids

        with simple_timer("generate_sequences", metrics):
            thinking_output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=phase1_params,
                image_data=images,
                video_data=videos,
            )

        thinking_trace_ids = list(thinking_output.token_ids)
        trace_length = len(thinking_trace_ids)

        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = (
                thinking_output.num_preempted if thinking_output.num_preempted is not None else -1
            )

        # ── Phase 2: Forced output at each budget checkpoint ──
        truncation_lengths = compute_budget_truncation_lengths(trace_length, self.budget_checkpoints)

        # Optionally include full trace as its own budget
        if self.include_full_trace and trace_length not in truncation_lengths:
            truncation_lengths.append(trace_length)
            truncation_lengths.sort()

        forced_output_params = {
            "temperature": self.forced_temperature,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "logprobs": False,
            "max_tokens": self.forced_max_tokens,
            "stop_token_ids": self.tokenizer.encode("\n", add_special_tokens=False),

        }

        # Fire off all budget forced generations concurrently
        # When num_forced_answers > 1, generate multiple answers per budget
        num_answers = self.num_forced_answers
        forced_tasks = []
        budget_labels = []      # (budget, answer_idx) pairs to track each task
        for trunc_len in truncation_lengths:
            forced_prompt = prompt_ids + build_forced_output_tokens(
                thinking_trace_ids=thinking_trace_ids,
                truncation_length=trunc_len,
                think_end_token_id=self.think_end_token_id,
                suffix_token_ids=self.suffix_token_ids,
            )
            for ans_idx in range(num_answers):
                budget_labels.append((trunc_len, ans_idx))
                forced_tasks.append(
                    self.server_manager.generate(
                        request_id=uuid4().hex,
                        prompt_ids=forced_prompt,
                        sampling_params=dict(forced_output_params),  # copy each time
                        image_data=images,
                        video_data=videos,
                    )
                )

        with simple_timer("forced_output_gen", metrics):
            forced_outputs = await asyncio.gather(*forced_tasks)

        # Decode the forced answers for downstream reward evaluation
        # Single-answer dicts (backward compat): use the first answer per budget
        forced_answers = {}
        forced_answer_ids = {}
        # Multi-answer dicts: all answers per budget
        forced_answers_multi = {}       # {budget: [answer_text, ...]}
        forced_answer_ids_multi = {}    # {budget: [[token_ids], ...]}
        for (budget, ans_idx), fo in zip(budget_labels, forced_outputs):
            answer_ids = list(fo.token_ids)
            answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
            # Multi-answer storage
            forced_answers_multi.setdefault(budget, []).append(answer_text)
            forced_answer_ids_multi.setdefault(budget, []).append(answer_ids)
            # Keep first answer for backward compat single-answer dicts
            if ans_idx == 0:
                forced_answers[budget] = answer_text
                forced_answer_ids[budget] = answer_ids

        # Build full response for the trace (the thinking trace itself)
        response_mask = [1] * len(thinking_trace_ids)

        # Store per-budget info in extra_fields for the trainer to use
        extra_fields = {
            "forced_answers": forced_answers,              # {budget: answer_text}  (first answer)
            "forced_answer_ids": forced_answer_ids,        # {budget: [token_ids]}  (first answer)
            "forced_answers_multi": forced_answers_multi,  # {budget: [answer_text, ...]}
            "forced_answer_ids_multi": forced_answer_ids_multi,  # {budget: [[token_ids], ...]}
            "budget_checkpoints": truncation_lengths,      # [int]
            "num_forced_answers": num_answers,
            "trace_length": trace_length,
        }

        # Build full text for each budget (prompt + truncated_trace + </think> + suffix + answer)
        # This is needed for reward computation, which typically evaluates the full response.
        forced_full_texts = {}       # {budget: text}  (first answer, backward compat)
        forced_full_texts_multi = {} # {budget: [text, ...]}  (all answers)
        for budget in truncation_lengths:
            truncated_trace_text = self.tokenizer.decode(
                thinking_trace_ids[:budget], skip_special_tokens=True
            )
            # Only add </think> if the truncated trace doesn't already end with it.
            # The trace generated with stop_token_ids typically includes the stop token,
            # and skip_special_tokens=True doesn't strip it (it's not a special token).
            if truncated_trace_text.rstrip().endswith(self.stop_token):
                think_end_text = ""
            else:
                think_end_text = self.stop_token
            # Build full texts for all answers at this budget
            budget_full_texts = []
            for answer_text in forced_answers_multi[budget]:
                full_text = (
                    truncated_trace_text + think_end_text + self.suffix_prompt + answer_text
                )
                budget_full_texts.append(full_text)
            forced_full_texts_multi[budget] = budget_full_texts
            # First answer for backward compat
            forced_full_texts[budget] = budget_full_texts[0]
        extra_fields["forced_full_texts"] = forced_full_texts
        extra_fields["forced_full_texts_multi"] = forced_full_texts_multi

        # ── Train-on-forced-output: include forced output in response_ids ──
        # When enabled, and there are no budget checkpoints (only full trace),
        # append the forced output suffix + answer tokens to response_ids so
        # the model trains on them (forward + gradient flow).
        train_on_fo = (
            self.train_on_forced_output
            and len(self.budget_checkpoints) == 0
            and self.include_full_trace
        )
        extra_fields["train_on_forced_output"] = train_on_fo

        if train_on_fo:
            # Get the forced output suffix tokens that come after the thinking trace
            # build_forced_output_tokens returns: truncated_trace + [</think>] + suffix
            # We slice off the trace part to get just [</think>] + suffix (or just suffix
            # if the trace already ends with </think>)
            fo_suffix_ids = build_forced_output_tokens(
                thinking_trace_ids=thinking_trace_ids,
                truncation_length=trace_length,
                think_end_token_id=self.think_end_token_id,
                suffix_token_ids=self.suffix_token_ids,
            )[trace_length:]

            # Get the first forced answer's token IDs at the full trace budget
            full_trace_answer_ids = forced_answer_ids.get(trace_length, [])

            # Build full response: thinking_trace + suffix + answer
            full_response_ids = thinking_trace_ids + fo_suffix_ids + full_trace_answer_ids
            full_response_mask = [1] * len(full_response_ids)

            final_response_ids = full_response_ids[: self.response_length]
            final_response_mask = full_response_mask[: self.response_length]
            # Log_probs from thinking output don't cover forced output tokens;
            # set to None since old_log_probs are recomputed during training anyway.
            final_response_logprobs = None
            logger.info(
                "train_on_forced_output: response extended from %d to %d tokens "
                "(trace=%d, suffix=%d, answer=%d, capped at %d)",
                trace_length, len(full_response_ids),
                trace_length, len(fo_suffix_ids), len(full_trace_answer_ids),
                self.response_length,
            )
        else:
            final_response_ids = thinking_trace_ids[: self.response_length]
            final_response_mask = response_mask[: self.response_length]
            final_response_logprobs = (
                thinking_output.log_probs[: self.response_length]
                if thinking_output.log_probs else None
            )

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=final_response_ids,
            response_mask=final_response_mask,
            response_logprobs=final_response_logprobs,
            routed_experts=(
                thinking_output.routed_experts[: len(prompt_ids) + self.response_length]
                if thinking_output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )
        return output
