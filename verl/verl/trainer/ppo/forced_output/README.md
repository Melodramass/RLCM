# Forced Output GRPO

A variant of GRPO that evaluates reasoning quality at **preset token budgets** by forcing the model to produce a final answer at multiple points during its chain-of-thought, and optionally trains a lightweight **probe MLP** on the model's hidden states to predict per-budget correctness. The probe's calibration signal can be fed back as an auxiliary reward.

## Overview

Standard GRPO scores each rollout once, after the model finishes its entire response. Forced Output GRPO instead:

1. **Phase 1 — Thinking trace**: Generate a chain-of-thought, optionally stopping at `</think>`.
2. **Phase 2 — Forced answers**: At each budget checkpoint (absolute like `[2000, 4000, 8000]` or relative like `[0.25, 0.5, 0.75, 1.0]`), truncate the trace, append a suffix prompt, and run a short generation to force a boxed answer. Multiple answers per budget can be sampled (`num_forced_answers`).
3. **Phase 3 — Reward aggregation**: Score each forced answer; optionally score the natural trace; optionally run a probe MLP on hidden states to produce a calibration reward. Combine into one scalar for GRPO advantage.

---

## File Map

```
forced_output/
├── __init__.py                     # Package exports
├── forced_output_utils.py          # Budget computation, token construction, reward aggregation,
│                                   # group-filter helpers (compute_unanimous_group_keep_mask)
├── forced_output_agent_loop.py     # Two-phase agent loop (thinking → forced answers)
├── forced_output_trainer.py        # Trainer subclass with forced-output + probe reward path
├── probe_trainer.py                # BudgetProbeMLP + BudgetProbeTrainer (standalone MLP)
├── main_forced_output_grpo.py      # Hydra entry point & TaskRunner
└── README.md                       # This file
```

Training scripts in [anytime_reasoner-new/scripts/train_new/](../../../../../anytime_reasoner-new/scripts/train_new/):

| Script | Notes |
|---|---|
| `grpo_forced_output.sh` | Baseline forced-output, absolute budgets, no probe |
| `e19-grpo_forced_output_lead_bs32_conf_diff_no_corner.sh` | Probe + `conf_diff`, `margin_corner=False` |
| `e19-grpo_forced_output_lead_bs32_conf_diff_0.5_true.sh` | Same but `margin_corner=True` |

---

## Architecture

### 1. Agent Loop — `ForcedOutputAgentLoop`

**File**: [forced_output_agent_loop.py](forced_output_agent_loop.py)

Registered as [`"forced_output_agent"`](forced_output_agent_loop.py#L46) in the agent loop registry. Called by rollout workers during sequence generation.

#### Phase 1: Thinking trace generation

[Lines 111–126](forced_output_agent_loop.py#L111-L126) — Calls `server_manager.generate()` with `stop_token_ids` set to the tokenized `stop_token` (default `</think>`; set to `''` to run freely to EOS).

```python
phase1_params["stop_token_ids"] = self.stop_token_ids
thinking_output = await self.server_manager.generate(...)
```

#### Phase 2: Forced output at each budget

[Lines 133–177](forced_output_agent_loop.py#L133-L177) — For each budget checkpoint:
- Compute truncation lengths via [`compute_budget_truncation_lengths()`](forced_output_utils.py#L32-L70)
- Build forced prompt via [`build_forced_output_tokens()`](forced_output_utils.py#L90-L115)
- Fire `num_forced_answers` generations per budget concurrently via `asyncio.gather`

```python
for trunc_len in truncation_lengths:
    for ans_idx in range(num_answers):
        forced_tasks.append(server_manager.generate(...))
forced_outputs = await asyncio.gather(*forced_tasks)
```

#### Output packaging

[Lines 179–303](forced_output_agent_loop.py#L179-L303) — Per-budget data stored in `extra_fields`:

| Key | Type | Description |
|-----|------|-------------|
| `forced_answers` | `{int: str}` | First answer text per budget |
| `forced_answer_ids` | `{int: [int]}` | First answer token IDs per budget |
| `forced_answers_multi` | `{int: [str]}` | All answer texts per budget |
| `forced_answer_ids_multi` | `{int: [[int]]}` | All answer token IDs per budget |
| `forced_full_texts` | `{int: str}` | Full reconstructed text (first answer) per budget — used for reward |
| `forced_full_texts_multi` | `{int: [str]}` | Full reconstructed texts (all answers) per budget |
| `budget_checkpoints` | `[int]` | Actual clamped truncation lengths used |
| `trace_length` | `int` | Full thinking trace length |
| `num_forced_answers` | `int` | Number of answers generated per budget |

### 2. Utilities — `forced_output_utils.py`

**File**: [forced_output_utils.py](forced_output_utils.py)

#### Budget computation

[`compute_budget_truncation_lengths()`](forced_output_utils.py#L32-L70) — Resolves each configured budget into an absolute token count, clamps to `min(trace_length, budget)`, deduplicates, and sorts.

```python
>>> compute_budget_truncation_lengths(6000, [0.25, 0.5, 1.0])
[1500, 3000, 6000]
>>> compute_budget_truncation_lengths(6000, [4000, 8000])
[4000, 6000]
```

[`resolve_budget_checkpoint()`](forced_output_utils.py#L73-L87) — Converts a float `(0,1]` fraction to an absolute token count or passes through an integer.

#### Token construction

[`build_forced_output_tokens()`](forced_output_utils.py#L90-L115) — Builds `truncated_trace + [</think>] + suffix_tokens`, avoiding a double `</think>` if the trace already ends with it.

#### Reward aggregation

[`aggregate_rewards()`](forced_output_utils.py#L118-L155) / [`aggregate_rewards_batch()`](forced_output_utils.py#L158-L172) — Collapse per-budget rewards into one scalar:
- **`"mean"`**: Simple average.
- **`"linear_weighted"`**: Weights `[1, 2, ..., n]` normalized — later budgets weighted higher.

[`aggregate_multi_answer_rewards()`](forced_output_utils.py#L175-L210) — Collapse multiple answers at a single budget into one reward:
- **`"mean"`**: Average over all answer rewards.
- **`"max"`**: Optimistic max.

#### Token-level reward tensor

[`create_token_level_reward_tensor()`](forced_output_utils.py#L213-L248) — Places the aggregated scalar on the last valid token of each sequence.

#### Group filtering

[`compute_unanimous_group_keep_mask()`](forced_output_utils.py#L251-L314) — Drops prompt groups where all rollouts have the same natural-trace correctness (all correct or all incorrect), keeping only "mixed" groups. Used when `probe.filter_natural_corner_groups=True`.

### 3. Trainer — `RayForcedOutputGRPOTrainer`

**File**: [forced_output_trainer.py](forced_output_trainer.py)

Subclasses [`RayPPOTrainer`](../ray_trainer.py). Key additions:

#### Reward computation

[`_compute_forced_output_rewards()`](forced_output_trainer.py#L237-L459) — For each sample in the batch:
1. Read `forced_full_texts` / `forced_full_texts_multi` and `budget_checkpoints` from `non_tensor_batch`.
2. Score all forced answers (single or multi) via `reward_fn.compute_score()`.
3. Compute MC accuracy per budget = fraction of correct answers.
4. Scale budget rewards by `fo_budget_reward_weight` (set to `0.0` to use budget scores only for probe without GRPO reward contribution).
5. Optionally score the natural thinking trace and add it directly: `total = natural_reward + agg(budget_rewards)`.
6. Aggregate budget rewards via [`aggregate_rewards_batch()`](forced_output_utils.py#L158).
7. Return token-level reward tensor + extra info dict.

#### Validation override

[`_compute_or_extract_reward()`](forced_output_trainer.py#L465-L547) — During validation, scores the **last-budget forced answer** (which contains a boxed answer) instead of the raw thinking trace.

#### `fit()` loop

[`fit()`](forced_output_trainer.py#L553) — Identical to `RayPPOTrainer.fit()` with two inserted phases:

**Forced output reward phase** (after rollout, before log-probs):
```python
if self.forced_output_enabled and self.reward_fn is not None:
    reward_tensor, reward_extra_infos_dict, per_budget_reward_lists = (
        self._compute_forced_output_rewards(batch, self.reward_fn)
    )
```

**Probe phase** (after old log-probs, using captured hidden states):
```python
if self.forced_output_enabled and self.probe_enabled and "hidden_states" in batch.batch:
    probe_features, probe_targets = self.probe_trainer.collect_probe_data(...)
    self.probe_trainer.train_step(probe_features, probe_targets)
    if self.probe_use_as_reward and self.probe_reward_weight > 0:
        probe_reward_scalars = self.probe_trainer.compute_conf_diff_rewards(...)
        reward_tensor = reward_tensor + probe_reward_contribution
```

### 4. Probe Trainer — `BudgetProbeTrainer`

**File**: [probe_trainer.py](probe_trainer.py)

A standalone 2-layer MLP trained online during GRPO to predict per-budget correctness from the model's hidden states. No gradient flows back into the language model.

#### Architecture (`BudgetProbeMLP`)

```
hidden_state(hidden_size) → Linear(hidden_size, hidden_size//4) → ReLU → Dropout → Linear(→1) → Sigmoid
```
Output: predicted correctness probability ∈ [0, 1].

#### Training target

Monte-Carlo accuracy at each budget: `mc_acc(bk) = mean(correct(answer_j) for j=1..num_answers)`. Binary cross-entropy loss, trained for `train_steps` mini-batch SGD steps per GRPO step.

#### Hidden state extraction

[`extract_budget_hidden_states()`](probe_trainer.py#L173-L205) — For each budget position `bk`, takes the mean of the last `avg_last_k` hidden states at response positions up to `prompt_length + bk`.

#### Reward modes

[`_compute_reward_signal()`](probe_trainer.py#L385-L426) — Simple per-budget reward modes:
- **`accuracy`**: 1 if `round(pred) == round(mc_acc)` else 0.
- **`brier`**: `-(pred - mc_acc)²`  ≤ 0.
- **`cross_entropy`**: `mc_acc·log(pred) + (1−mc_acc)·log(1−pred)`  ≤ 0.

[`compute_conf_diff_rewards()`](probe_trainer.py#L498-L637) — Per-sample calibration reward (used by `conf_diff`, `conf_diff_v1`, `conf_margin_covar` modes):

```
conf_correct   = {probe_conf(bk) : mc_acc(bk) == 1.0}   # strictly-correct budgets
conf_incorrect = {probe_conf(bk) : mc_acc(bk) == 0.0}   # strictly-incorrect budgets
(partial budgets excluded)

conf_diff = mean(conf_correct) - mean(conf_incorrect)
```

Corner handling when only one class is present (`margin_corner=True`):
- All correct (no incorrect budgets): `conf_diff = mean(conf_correct)` (= `conf_correct - 0`)
- All incorrect (no correct budgets):
  - `conf_diff` mode: `1 - mean(conf_incorrect)`
  - `conf_diff_v1` / `conf_margin_covar`: `0 - mean(conf_incorrect)`

With `margin_corner=False` (no corner): one-class traces always return `conf_diff = 0`.

[`apply_conf_margin_covar()`](probe_trainer.py#L678-L696) — Multiplies `conf_diff` rewards by the group-level variance factor `p*(1-p)` (where `p` = fraction of natural traces correct in the prompt group), suppressing signal from trivially-easy or trivially-hard groups.

### 5. Entry Point — `main_forced_output_grpo.py`

**File**: [main_forced_output_grpo.py](main_forced_output_grpo.py)

- **Hydra entry**: [`main()`](main_forced_output_grpo.py#L42-L44) — loads config from `verl/trainer/ppo/config/ppo_trainer.yaml`.
- **Ray launch**: [`run_forced_output_grpo()`](main_forced_output_grpo.py#L47-L85) — initializes Ray, spawns a `ForcedOutputTaskRunner`.
- **Task runner**: [`ForcedOutputTaskRunner`](main_forced_output_grpo.py#L88) — mirrors `TaskRunner` but instantiates `RayForcedOutputGRPOTrainer`.

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  RayForcedOutputGRPOTrainer.fit()                                               │
│                                                                                 │
│  1. generate_sequences(batch)                                                   │
│     └─► ForcedOutputAgentLoop.run()                                             │
│         ├─ Phase 1: free thinking trace (stop at stop_token or EOS)             │
│         └─ Phase 2: for each budget × num_forced_answers → truncate+suffix+gen  │
│            └─ stores forced_full_texts[_multi] + budget_checkpoints in batch    │
│                                                                                 │
│  2. _compute_forced_output_rewards(batch, reward_fn)                            │
│     ├─ score forced_full_texts[_multi] → per-answer rewards                    │
│     ├─ MC accuracy per budget = mean(correct answers)                           │
│     ├─ scale budget rewards by budget_reward_weight                             │
│     ├─ [optional] score natural thinking trace; add to base reward              │
│     ├─ aggregate_rewards_batch() → one scalar per trace                         │
│     └─ create_token_level_reward_tensor() → (B, T) reward_tensor               │
│                                                                                 │
│  3. _compute_old_log_prob(batch)           ← requests hidden_states if probe   │
│                                                                                 │
│  4. [optional] Probe phase                                                      │
│     ├─ collect_probe_data() → (hidden_features, mc_accuracy targets)            │
│     ├─ probe_trainer.train_step(features, targets)  ← 16 SGD steps             │
│     ├─ compute_conf_diff_rewards() → per-sample conf_diff scalars               │
│     └─ reward_tensor += probe_reward_weight × conf_diff                         │
│                                                                                 │
│  5. compute_advantage() (standard GRPO)                                         │
│  6. update_actor() / update_critic()                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Code Trace & Cross-Dependencies

### File Tree with Roles

```
verl-new/verl/trainer/ppo/forced_output/
│
├── main_forced_output_grpo.py
│   ├── imports: RayForcedOutputGRPOTrainer  (forced_output_trainer.py)
│   └── mirrors: verl/trainer/ppo/main_ppo.py  (TaskRunner → ForcedOutputTaskRunner)
│
├── forced_output_trainer.py
│   ├── imports: RayPPOTrainer               (../ray_trainer.py)         [superclass]
│   ├── imports: forced_output_utils.py      [reward math + group filter]
│   ├── imports: probe_trainer.py            [BudgetProbeTrainer]        [lazy in fit()]
│   └── reads:   batch["hidden_states"]      ← populated by actor worker during old_log_prob
│
├── forced_output_agent_loop.py
│   ├── imports: forced_output_utils.py      [budget + token helpers]
│   ├── imports: AgentLoopBase               (verl/experimental/agent_loop/agent_loop.py)
│   └── writes:  batch.non_tensor_batch      → forced_full_texts, forced_full_texts_multi,
│                                               budget_checkpoints, trace_length,
│                                               num_forced_answers, ...
│
├── forced_output_utils.py
│   └── pure functions — no imports from this package; used by both agent loop and trainer
│
└── probe_trainer.py
    ├── BudgetProbeMLP    (nn.Module, 2-layer MLP → sigmoid)
    └── BudgetProbeTrainer
        ├── reads:  batch["hidden_states"]   (B, seq_len, hidden)  from actor forward pass
        ├── reads:  probe_mc_accuracies      from reward_extra_infos_dict
        └── reads:  probe_budget_positions   from reward_extra_infos_dict
```

### Call Graph (training step)

```
fit()  [forced_output_trainer.py]
 │
 ├─ actor_rollout_wg.generate_sequences()
 │   └─ ForcedOutputAgentLoop.run()   [forced_output_agent_loop.py]
 │       ├─ compute_budget_truncation_lengths()  [forced_output_utils.py]
 │       └─ build_forced_output_tokens()         [forced_output_utils.py]
 │
 ├─ _compute_forced_output_rewards()  [forced_output_trainer.py]
 │   ├─ reward_fn.compute_score()      (per forced answer + optional natural trace)
 │   ├─ aggregate_multi_answer_rewards()          [forced_output_utils.py]
 │   ├─ aggregate_rewards_batch()                 [forced_output_utils.py]
 │   └─ create_token_level_reward_tensor()        [forced_output_utils.py]
 │
 ├─ _compute_old_log_prob()   ← sets batch.meta_info["return_hidden_states"]=True
 │                              → actor worker populates batch["hidden_states"]
 │
 └─ [probe block]
     ├─ BudgetProbeTrainer.collect_probe_data()  [probe_trainer.py]
     ├─ BudgetProbeTrainer.train_step()          [probe_trainer.py]
     ├─ BudgetProbeTrainer.compute_conf_diff_rewards()  [probe_trainer.py]
     │   └─ BudgetProbeTrainer.predict()
     └─ [conf_margin_covar only]
         ├─ BudgetProbeTrainer.apply_conf_margin_covar()
         └─ BudgetProbeTrainer.compute_group_covar_factors()
```

### Key Data Handoffs

| From | Key | To |
|---|---|---|
| `ForcedOutputAgentLoop` | `forced_full_texts[_multi]`, `budget_checkpoints`, `trace_length` | `_compute_forced_output_rewards()` via `non_tensor_batch` |
| `_compute_forced_output_rewards()` | `probe_mc_accuracies`, `probe_budget_positions` | probe block in `fit()` via `reward_extra_infos_dict` |
| actor worker (`_compute_old_log_prob`) | `hidden_states` | `BudgetProbeTrainer.collect_probe_data()` via `batch.batch` |
| `BudgetProbeTrainer` | per-sample conf_diff scalars | `reward_tensor` addend in `fit()` |

---

## Configuration

All forced-output parameters are **Hydra overrides** (prefix `+` for new keys).

### Core forced-output params

| Config key | Type | Default | Description |
|---|---|---|---|
| `+forced_output.enable` | bool | `False` | Master switch |
| `+forced_output.stop_token` | str | `</think>` | Phase 1 stop token (`''` = no stop, free trace) |
| `+forced_output.budget_checkpoints` | list[int\|float] | — | Absolute (`[2000,4000,8000]`) or relative (`[0.25,0.5,1.0]`) |
| `+forced_output.suffix_prompt` | str | `\nIf I were to...` | Text appended after `</think>` at forced generation |
| `+forced_output.forced_max_tokens` | int | `128` | Max tokens for forced generation |
| `+forced_output.forced_temperature` | float | `0.1` | Temperature for forced generation |
| `+forced_output.num_forced_answers` | int | `1` | Answers generated per budget (MC accuracy uses all) |
| `+forced_output.multi_answer_aggregation` | str | `mean` | How to reduce multiple answers: `mean` or `max` |
| `+forced_output.reward_aggregation` | str | `mean` | How to aggregate across budgets: `mean` or `linear_weighted` |
| `+forced_output.budget_reward_weight` | float | `1.0` | Scale factor on budget rewards (set `0.0` to disable budget reward contribution) |
| `+forced_output.include_budget_reward` | bool | `True` | Shortcut: if `False` → forces `budget_reward_weight=0` |
| `+forced_output.include_full_trace` | bool | `True` | Add a forced answer at the natural trace's full length |
| `+forced_output.include_natural_trace` | bool | `False` | Score the natural thinking trace; add reward directly (not averaged into budget agg) |
| `+forced_output.train_on_forced_output` | bool | `False` | Append suffix+answer tokens to response so actor trains on them |

### Probe params

| Config key | Type | Default | Description |
|---|---|---|---|
| `+forced_output.probe.enable` | bool | `False` | Enable probe MLP |
| `+forced_output.probe.use_as_reward` | bool | `False` | Add probe signal to reward tensor |
| `+forced_output.probe.reward_weight` | float | `0.0` | Multiplier for probe reward |
| `+forced_output.probe.reward_mode` | str | `accuracy` | `accuracy`, `brier`, `cross_entropy`, `conf_diff`, `conf_diff_v1`, `conf_margin_covar` |
| `+forced_output.probe.margin_corner` | bool | `False` | Corner handling for one-class traces in `conf_diff*` modes |
| `+forced_output.probe.train_steps` | int | `20` | Probe SGD steps per GRPO step |
| `+forced_output.probe.batch_size` | int | `16` | Probe mini-batch size |
| `+forced_output.probe.lr` | float | `1e-3` | Probe AdamW learning rate |
| `+forced_output.probe.weight_decay` | float | `0.01` | Probe AdamW weight decay |
| `+forced_output.probe.dropout` | float | `0.1` | Probe MLP dropout |
| `+forced_output.probe.avg_last_k` | int | `1` | Tokens to average at each budget position |
| `+forced_output.probe.last_layer_idx` | int | `1` | Transformer layer to hook (1 = last) |
| `+forced_output.probe.device` | str | `auto` | Probe device (`auto`, `cuda`, `cpu`) |
| `+forced_output.probe.filter_natural_corner_groups` | bool | `False` | Drop prompt groups unanimous under natural trace (requires `include_natural_trace=True`) |

Plus agent loop selector:
```
actor_rollout_ref.rollout.agent.default_agent_loop=forced_output_agent
```

Config is read independently by the agent loop ([forced_output_agent_loop.py:68-80](forced_output_agent_loop.py#L68-L80)) and the trainer ([forced_output_trainer.py:81-147](forced_output_trainer.py#L81-L147)).

---

## Reward Formula Reference

### Without probe

```
budget_reward(bk) = mean(reward_fn(forced_answer_j at bk) for j) × budget_reward_weight
r_base = agg(budget_rewards)              # e.g. mean across budgets
if include_natural_trace:
    r_base += reward_fn(natural trace)    # added on top, not averaged

GRPO reward = r_base
```

### With probe (conf_diff mode, e.g. e19)

```
# Budget rewards as above (often budget_reward_weight=0.0, so r_base = r_nat)
mc_acc(bk) = mean(correct(answer_j at bk) for j)   # ∈ [0,1]

# Hidden states → probe MLP trained on mc_acc targets
conf_correct   = {probe_conf(bk) : mc_acc(bk) == 1.0}
conf_incorrect = {probe_conf(bk) : mc_acc(bk) == 0.0}
conf_diff      = mean(conf_correct) - mean(conf_incorrect)
  # margin_corner=False: → 0 if only one class present
  # margin_corner=True:  → corner reward (see probe_trainer.py)

GRPO reward = r_base + probe_reward_weight × conf_diff
```

### conf_margin_covar extension

```
p = fraction of natural-trace-correct rollouts in same prompt group
conf_diff_scaled = conf_diff × p × (1 - p)   # suppresses trivial groups
GRPO reward = r_base + probe_reward_weight × conf_diff_scaled
```

---

## Quick Start

```bash
# Baseline forced-output (no probe)
bash anytime_reasoner-new/scripts/train_new/grpo_forced_output.sh \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# e19: conf_diff probe, no corner handling
bash anytime_reasoner-new/scripts/train_new/e19-grpo_forced_output_lead_bs32_conf_diff_no_corner.sh

# Direct invocation
python3 -m verl.trainer.ppo.forced_output.main_forced_output_grpo \
    +forced_output.enable=True \
    '+forced_output.budget_checkpoints=[0.25,0.5,0.75,1.0]' \
    +forced_output.num_forced_answers=4 \
    +forced_output.budget_reward_weight=0.0 \
    +forced_output.include_natural_trace=True \
    +forced_output.probe.enable=True \
    +forced_output.probe.use_as_reward=True \
    +forced_output.probe.reward_weight=0.3 \
    +forced_output.probe.reward_mode=conf_diff \
    +forced_output.probe.margin_corner=False \
    actor_rollout_ref.rollout.agent.default_agent_loop=forced_output_agent \
    ...
```

---

## Key Design Decisions

- **Budget clamping**: Absolute budgets are clamped to `min(trace_len, budget)`. Relative float budgets in `(0, 1]` are resolved to `round(trace_len × fraction)` first, then clamped and deduplicated.
- **Natural trace added, not averaged**: When `include_natural_trace=True`, the natural trace reward is added directly to the aggregated budget reward (`r_nat + agg`), not folded into the budget average. This preserves a stable GRPO signal even when `budget_reward_weight=0`.
- **Budget-zeroing for probe-only signal**: Setting `budget_reward_weight=0.0` (or `include_budget_reward=False`) lets you compute MC accuracy for probe training without contributing budget rewards to GRPO advantage.
- **Probe is standalone**: No gradients flow from probe loss back into the LM. The probe trains on hidden states captured during the normal `old_log_prob` forward pass via `return_hidden_states=True`.
- **conf_diff excludes partial budgets**: Only strictly all-correct (`mc_acc==1`) or all-incorrect (`mc_acc==0`) budgets contribute to `conf_diff`. Mixed budgets (0 < mc_acc < 1) are excluded to keep the signal clean.
- **Concurrent forced generation**: All budget × answer generations run in parallel via `asyncio.gather` ([forced_output_agent_loop.py#L176-L177](forced_output_agent_loop.py#L176-L177)).
- **Reward replacement**: Forced-output reward fully replaces standard GRPO reward when enabled — no combining with normal outcome reward ([forced_output_trainer.py#L719-L779](forced_output_trainer.py#L719-L779)).
- **Subclass approach**: Clean separation via subclassed trainer — no modifications to base `RayPPOTrainer`.
