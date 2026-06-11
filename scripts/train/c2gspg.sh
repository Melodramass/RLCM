#!/bin/bash
# =============================================================================
# Experiment e25 — C2GSPG + Forced Output + Probe (ranking_reward)
# =============================================================================
#
# GOAL
#   Train an anytime-capable reasoning model whose hidden states are explicitly
#   calibrated to predict their own per-budget correctness (via a probe MLP).
#   The probe's confidence-difference signal acts as an auxiliary reward term
#   that encourages the model's internal representations to be well-calibrated
#   across multiple computation budgets.
#
# SETUP
#   • Dataset   : deepscaler/data/lead.parquet  (training)
#                 deepscaler/data/aime_filtered.parquet (validation)
#   • Base model: DeepSeek-R1-Distill-Qwen-7B  (default; override via --model)
#   • Hardware  : 1 node × 4 A100-80GB GPUs, TP=2
#   • Batch size: 32 prompts × 6 rollouts = 192 traces per step
#
# ROLLOUT (ForcedOutputAgentLoop)
#   Phase 1 — generate a free thinking trace (stop_token='', i.e. no early stop;
#             the model runs to max_response_length=8000 or EOS).
#   Phase 2 — at each of 4 RELATIVE budget checkpoints (25%, 50%, 75%, 100%
#             of the generated trace length), truncate the trace, append the
#             suffix prompt, and generate num_forced_answers=4 short boxed
#             answers (forced_max_tokens=32, forced_temperature=0.8).
#             All 4×4 = 16 forced generations run concurrently via asyncio.gather.
#   include_full_trace=False — no extra forced answer beyond the 1.0 checkpoint.
#
# REWARD COMPUTATION
#   Let B = {b1=0.25T, b2=0.50T, b3=0.75T, b4=1.00T} where T = trace length.
#   Let A_k = {a_{k,1}, ..., a_{k,4}} = 4 forced answers at budget bk.
#
#   ── Budget MC accuracy (for probe training only) ──
#     mc_acc(bk) = mean(correct(a_{k,j}) for j=1..4)   ∈ {0, 0.25, 0.5, 0.75, 1.0}
#     budget_reward(bk) = mean_reward(A_k) × budget_reward_weight
#                       = mean_reward(A_k) × 0.0   →   0   (zeroed out)
#     Budget scores are computed purely for probe training, not GRPO.
#
#   ── Natural trace reward ──
#     r_nat = reward_fn(natural thinking trace)   ∈ {0, 1}
#     (scored at the full generated trace, not a forced-output prompt)
#
#   ── Base GRPO reward ──
#     r_base = r_nat + agg(budget_rewards)  =  r_nat + 0  =  r_nat ∈ {0, 1}
#
#   ── Probe reward (ranking_reward mode) ──
#     After recomputing log-probs, capture last-layer hidden states.
#     For each budget bk, extract the mean hidden state over the last 4 tokens
#     before position bk (avg_last_k=4).
#     Train BudgetProbeMLP on mc_acc(bk), then infer probe confidence on the
#     same states and average across budgets:
#
#       p = mean(probe_conf(bk) for bk in B)
#       c = 1 if the natural trace is correct else 0
#
#       if c = 1 and p > 0.5:   reward =  1
#       if c = 1 and p <= 0.5:  reward =  rt
#       if c = 0 and p <= 0.5:  reward = -rt
#       if c = 0 and p > 0.5:   reward = -1
#
#     rt is tunable with --rt and defaults to 0.5.
#
#   ── FINAL GRPO reward ──
#     R = r_nat + ranking_reward
#
#   Intuition: the natural trace still optimizes correctness, while the probe
#   reward prefers high average probe confidence on correct traces and low
#   average probe confidence on incorrect traces.
#
# RUN
#   bash e25-C2GSPG.sh [--model PATH] [--rt 0.5]
# =============================================================================
set -x
ray stop

unset ROCR_VISIBLE_DEVICES

# Use verl instead of verl
# Add verl to PYTHONPATH so it takes precedence over the installed verl
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/verl:${PYTHONPATH}"

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
# export VLLM_ATTENTION_BACKEND=XFORMERS

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --rt)
            RANKING_REWARD_RT="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
fi
if [ -z "$RANKING_REWARD_RT" ]; then
    RANKING_REWARD_RT="0.5"
fi

echo "Using verl from: ${PROJECT_ROOT}/verl"
echo "Model path: $MODEL_PATH"
echo "Ranking reward rt: $RANKING_REWARD_RT"
echo "Extra args: ${@:1}"

run_name='c2gspg'

# Train over a single node, 4 a100-80GB GPUs.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

RAY_DEDUP_LOGS=0 PYTHONUNBUFFERED=1 python3 -m verl.trainer.ppo.forced_output.main_forced_output_grpo \
    algorithm.adv_estimator=c2gspg \
    +algorithm.c2gspg_confidence_clip_eps=1e-6 \
    +algorithm.c2gspg_denominator_eps=1e-6 \
    actor_rollout_ref.actor.policy_loss.loss_mode=c2gspg \
    +actor_rollout_ref.actor.policy_loss.c2gspg_bce_beta=0.1 \
    data.train_files=deepscaler/data/lead.parquet \
    data.val_files=[deepscaler/data/aime_filtered.parquet] \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.max_prompt_length=600 \
    data.max_response_length=8000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='anytime-reasoning' \
    trainer.experiment_name="$run_name" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.rollout_data_dir="rollouts/${run_name}/rollout_generations" \
    trainer.validation_data_dir="rollouts/${run_name}/validation_generations" \
    trainer.log_val_generations=999999 \
    trainer.total_epochs=3 \
    actor_rollout_ref.rollout.agent.default_agent_loop=forced_output_agent \
    +forced_output.enable=True \
    "+forced_output.stop_token=''" \
    '+forced_output.budget_checkpoints=[1.0]' \
    +forced_output.forced_max_tokens=32 \
    +forced_output.forced_temperature=0.8 \
    +forced_output.reward_aggregation=mean \
    +forced_output.include_full_trace=False \
    +forced_output.include_natural_trace=True \
    +forced_output.num_forced_answers=4 \
    +forced_output.train_on_forced_output=False \
    +forced_output.budget_reward_weight=0.0 \
    +forced_output.include_budget_reward=True \
    +forced_output.probe.enable=False \
    +forced_output.probe.filter_natural_corner_groups=False \
    +forced_output.probe.use_as_reward=False \
    +forced_output.probe.reward_weight=1.0 \
    +forced_output.probe.reward_mode=ranking_reward \
    +forced_output.probe.ranking_reward_rt=0.5 \
    +forced_output.probe.margin_corner=False \
    +forced_output.probe.train_steps=32 \
    +forced_output.probe.batch_size=16 \
    +forced_output.probe.lr=0.001 \
    +forced_output.probe.dropout=0.1 \
    +forced_output.probe.avg_last_k=4 \
    +forced_output.probe.last_layer_idx=1 \
    +forced_output.probe.weight_decay=0.01 \
    +forced_output.probe.device=auto \
    "$@"
