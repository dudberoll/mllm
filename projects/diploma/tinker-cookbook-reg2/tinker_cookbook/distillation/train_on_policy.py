"""
Implements on-policy distillation. For more details, see:
https://thinkingmachines.ai/blog/on-policy-distillation
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Sequence, cast

import chz
import numpy as np
import tinker
import torch
from scipy.stats import linregress

from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.display import colorize_example
from tinker_cookbook.distillation.datasets import (
    CompositeDataset,
    DistillationDatasetConfig,
)
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
from tinker_cookbook.rl.train import (
    compute_full_batch_metrics_and_get_sampling_client,
    do_group_rollout_and_filter_constant_reward,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import safezip, timed
from tinker_cookbook.utils.trace import scope, update_scope_context, trace_init

logger = logging.getLogger(__name__)


@scope
async def incorporate_kl_penalty(
    data_D: List[tinker.Datum],
    teacher_clients_D: List[tinker.SamplingClient],
    dataset_indices_D: List[int],
    current_step: int,
    kl_penalty_coef: float,
    kl_discount_factor: float,
    hurst_weight_mode: str,
    hurst_num_windows: int,
    hurst_n_active_min: int,
    hurst_n_active_max: int,
    hurst_weight_beta: float,
    hurst_weight_min: float,
    hurst_weight_max: float,
    hurst_weight_eps: float,
    hurst_weight_warmup_steps: int,
    hurst_weight_apply_to: str,
) -> tuple[Dict[str, float], List[dict]]:
    """
    Compute reverse KL between the student (log p) and the teacher model (log q), computed as
    log p - log q. We then adjust the advantages in-place as the negative reverse KL.

    Args:
        data_D: List of datums to compute KL for
        teacher_clients_D: List of teacher sampling clients, one per datum
        dataset_indices_D: List of dataset indices, one per datum
        current_step: Zero-based training step
        kl_penalty_coef: Coefficient for KL penalty
        kl_discount_factor: Discount factor for future KL
    """
    # Note: if your teacher has a different renderer than the student, you may want to modify
    #       the full_sequence_inputs_D to match the teacher's renderer.
    full_sequence_inputs_D = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data_D
    ]
    # Compute the teacher's logprobs for each element of the batch
    # Each datum uses its corresponding teacher sampling client
    teacher_logprobs_D = await asyncio.gather(
        *[
            teacher_client.compute_logprobs_async(sequence_input)
            for teacher_client, sequence_input in zip(teacher_clients_D, full_sequence_inputs_D)
        ]
    )
    # The reverse KL is computed as KL[p||q] = log p - log q, where
    #   - p: sampled_logprobs
    #   - q: teacher_logprobs
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
    reverse_kl = [
        (sampled_logprobs - torch.tensor(teacher_logprobs[1:])) * mask
        for teacher_logprobs, sampled_logprobs, mask in safezip(
            teacher_logprobs_D, sampled_logprobs_D, float_masks
        )
    ]
    if hurst_weight_mode not in {"none", "std_t", "std_s"}:
        raise ValueError(f"Unsupported hurst_weight_mode: {hurst_weight_mode}")
    if hurst_weight_apply_to != "kl_only":
        raise ValueError(f"Unsupported hurst_weight_apply_to: {hurst_weight_apply_to}")

    # Track per-dataset KL for logging
    # dataset_idx -> (sum of KL, sum of mask)
    per_dataset_kl: Dict[int, tuple[float, float]] = {}
    telemetry_data = []

    # Fractal metrics tracking
    h_student_list = []
    h_teacher_list = []
    std_student_list = []
    std_teacher_list = []
    num_hurst_windows_list = []
    n_active_tokens_list = []

    for i, _datum in enumerate(data_D):
        active = float_masks[i] > 0

        lp_s = sampled_logprobs_D[i][active]
        lp_t = torch.tensor(teacher_logprobs_D[i][1:])[active]
        n_active_tokens_list.append(int(active.sum().item()))

        hs = calculate_hurst_efficient(lp_s)
        ht = calculate_hurst_efficient(lp_t)
        h_student_list.append(hs)
        h_teacher_list.append(ht)

        _, std_s, num_windows_s = calculate_adaptive_hurst_stats(lp_s, hurst_num_windows)
        _, std_t, num_windows_t = calculate_adaptive_hurst_stats(lp_t, hurst_num_windows)
        std_student_list.append(std_s)
        std_teacher_list.append(std_t)
        num_hurst_windows_list.append(min(num_windows_s, num_windows_t))

    weight_signals = (
        std_teacher_list
        if hurst_weight_mode == "std_t"
        else std_student_list
        if hurst_weight_mode == "std_s"
        else [0.0] * len(data_D)
    )
    weighting_enabled = (
        hurst_weight_mode != "none" and current_step >= hurst_weight_warmup_steps
    )
    hurst_weights, signal_median, signal_mad, active_weight_count = compute_hurst_weights(
        weight_signals,
        n_active_tokens_list,
        enabled=weighting_enabled,
        n_active_min=hurst_n_active_min,
        n_active_max=hurst_n_active_max,
        beta=hurst_weight_beta,
        weight_min=hurst_weight_min,
        weight_max=hurst_weight_max,
        eps=hurst_weight_eps,
    )

    for i, datum in enumerate(data_D):
        mask = float_masks[i]

        # Resonance reward disabled for this experiment.
        # Keep the variable to avoid changing the downstream training flow.
        resonance_reward = 0.0

        # The advantage is the negative reverse KL, optionally reweighted by Hurst variability.
        kl_advantages = -kl_penalty_coef * mask * reverse_kl[i]
        if kl_discount_factor > 0:
            kl_advantages = torch.tensor(
                discounted_future_sum_vectorized(kl_advantages.numpy(), kl_discount_factor)
            )
        weighted_kl_advantages = kl_advantages * hurst_weights[i]
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch()
            + weighted_kl_advantages
            + resonance_reward * mask
        )

        telemetry_data.append({
            "logp_student": sampled_logprobs_D[i].detach().cpu().numpy().tolist(),
            "logp_teacher": teacher_logprobs_D[i][1:],  # [1:] for alignment with rollout
            "step_kl": reverse_kl[i].detach().cpu().numpy().tolist(),
            "advantages": datum.loss_fn_inputs["advantages"].to_torch().detach().cpu().numpy().tolist(),
            "h_student": h_student_list[i],
            "h_teacher": h_teacher_list[i],
            "std_s": std_student_list[i],
            "std_t": std_teacher_list[i],
            "num_hurst_windows": num_hurst_windows_list[i],
            "n_active_tokens": n_active_tokens_list[i],
            "hurst_weight": hurst_weights[i],
            "hurst_weight_signal": weight_signals[i],
        })

        # Accumulate per-dataset KL
        dataset_idx = dataset_indices_D[i]
        kl_sum = reverse_kl[i].sum().item()
        mask_sum = float_masks[i].sum().item()
        if dataset_idx not in per_dataset_kl:
            per_dataset_kl[dataset_idx] = (0.0, 0.0)
        prev_kl_sum, prev_mask_sum = per_dataset_kl[dataset_idx]
        per_dataset_kl[dataset_idx] = (prev_kl_sum + kl_sum, prev_mask_sum + mask_sum)

    # Compute average reverse KL over the batch for logging purposes
    avg_logp_diff = sum([diff.sum() for diff in reverse_kl]) / sum(
        [mask.sum() for mask in float_masks]
    )

    # Compute per-dataset metrics
    metrics = {
        "teacher_kl": float(avg_logp_diff),
        "optim/h_student": float(np.mean(h_student_list)),
        "optim/h_teacher": float(np.mean(h_teacher_list)),
        "optim/fractal_dissonance": float(np.mean([abs(s-t) for s, t in zip(h_student_list, h_teacher_list)])),
        "optim/std_s_mean": float(np.mean(std_student_list)),
        "optim/std_s_median": float(np.median(std_student_list)),
        "optim/std_t_mean": float(np.mean(std_teacher_list)),
        "optim/std_t_median": float(np.median(std_teacher_list)),
        "optim/hurst_num_windows_mean": float(np.mean(num_hurst_windows_list)),
        "optim/hurst_n_active_mean": float(np.mean(n_active_tokens_list)),
        "optim/hurst_weight_enabled": float(weighting_enabled),
        "optim/hurst_weight_mean": float(np.mean(hurst_weights)),
        "optim/hurst_weight_median": float(np.median(hurst_weights)),
        "optim/hurst_weight_std": float(np.std(hurst_weights)),
        "optim/hurst_weight_signal_median": signal_median,
        "optim/hurst_weight_signal_mad": signal_mad,
        "optim/hurst_weight_active_frac": float(active_weight_count / len(hurst_weights))
        if hurst_weights
        else 0.0,
    }
    for dataset_idx, (kl_sum, mask_sum) in per_dataset_kl.items():
        if mask_sum > 0:
            metrics[f"teacher_kl/dataset_{dataset_idx}"] = float(kl_sum / mask_sum)

    return metrics, telemetry_data


@chz.chz
class Config:
    learning_rate: float
    dataset_configs: List[DistillationDatasetConfig]
    model_name: str
    renderer_name: str | None = None
    max_tokens: int
    temperature: float = 1.0
    compute_post_kl: bool = False
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Loss function and configuration.
    # See https://tinker-docs.thinkingmachines.ai/losses
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    base_url: str | None = None
    enable_trace: bool = False

    eval_every: int = 20
    save_every: int = 20
    load_checkpoint_path: str | None = None

    # Maximum number of training steps. If None, train on the full dataset.
    max_step: int | None = None
    
    # Fractal RL parameters (FDRL)
    fractal_alpha: float = 0.0 # Coefficient for fractal regularization
    golden_zone_mode: str = "none"
    golden_zone_lambda: float = 0.0
    golden_zone_h_target: float = 0.70
    golden_zone_h_delta: float = 0.09
    hurst_weight_mode: str = "none"
    hurst_num_windows: int = 10
    hurst_n_active_min: int = 1024
    hurst_n_active_max: int = 1536
    hurst_weight_beta: float = 0.25
    hurst_weight_min: float = 0.9
    hurst_weight_max: float = 1.1
    hurst_weight_eps: float = 1e-6
    hurst_weight_warmup_steps: int = 5
    hurst_weight_apply_to: str = "kl_only"


def calculate_hurst_efficient(logprobs: torch.Tensor) -> float:
    """Fast estimation of Hurst exponent for online training."""
    if len(logprobs) < 20: return 0.5
    data = logprobs.detach().cpu().numpy()
    N = len(data)
    y = np.cumsum(data - np.mean(data))
    scales = np.unique(np.logspace(np.log10(4), np.log10(N // 4), num=8).astype(int))
    if len(scales) < 2: return 0.5
    fluctuations = []
    for s in scales:
        num_windows = N // s
        rms = np.sqrt(np.mean([np.var(y[i*s:(i+1)*s]) for i in range(num_windows)]))
        fluctuations.append(rms)
    H, _, _, _, _ = linregress(np.log(scales), np.log(fluctuations))
    return float(np.clip(H, 0.0, 1.0))


def calculate_golden_zone_h_penalty(h_value: float, h_target: float, h_delta: float) -> tuple[float, float]:
    """Return (penalty, excess) for the Hurst-only golden-zone baseline."""
    excess = max(0.0, abs(h_value - h_target) - h_delta)
    return excess * excess, excess


def calculate_adaptive_hurst_stats(
    logprobs: torch.Tensor, num_windows: int
) -> tuple[list[float], float, int]:
    """Compute Hurst values over a fixed number of equal-length windows."""
    if num_windows <= 0:
        raise ValueError(f"num_windows must be positive, got {num_windows}")
    if len(logprobs) < num_windows:
        return [], 0.0, 0

    boundaries = np.linspace(0, len(logprobs), num_windows + 1, dtype=int)
    hurst_profile = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            return [], 0.0, 0
        hurst_profile.append(calculate_hurst_efficient(logprobs[start:end]))
    return hurst_profile, float(np.std(hurst_profile)), num_windows


def compute_hurst_weights(
    signals: list[float],
    n_active_list: list[int],
    enabled: bool,
    n_active_min: int,
    n_active_max: int,
    beta: float,
    weight_min: float,
    weight_max: float,
    eps: float,
) -> tuple[list[float], float, float, int]:
    """Convert a batch of trajectory-level signals into bounded weighting coefficients."""
    weights = [1.0] * len(signals)
    if not enabled or not signals:
        return weights, 0.0, 1.0, 0

    eligible_indices = [
        i
        for i, (signal, n_active) in enumerate(zip(signals, n_active_list))
        if n_active_min <= n_active <= n_active_max and np.isfinite(signal)
    ]
    if not eligible_indices:
        return weights, 0.0, 1.0, 0

    eligible_values = np.array([signals[i] for i in eligible_indices], dtype=float)
    signal_median = float(np.median(eligible_values))
    signal_mad = float(np.median(np.abs(eligible_values - signal_median)))
    if (not np.isfinite(signal_mad)) or signal_mad < eps:
        signal_mad = 1.0

    for i in eligible_indices:
        z_score = (signals[i] - signal_median) / (signal_mad + eps)
        weights[i] = float(np.clip(1.0 + beta * np.tanh(z_score), weight_min, weight_max))

    return weights, signal_median, signal_mad, len(eligible_indices)


@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    dataset_indices_P: List[int],
    teacher_clients: List[tinker.SamplingClient],
    current_step: int,
    kl_penalty_coef: float,
    kl_discount_factor: float,
    fractal_alpha: float = 0.0,
    golden_zone_mode: str = "none",
    golden_zone_lambda: float = 0.0,
    golden_zone_h_target: float = 0.70,
    golden_zone_h_delta: float = 0.09,
    hurst_weight_mode: str = "none",
    hurst_num_windows: int = 10,
    hurst_n_active_min: int = 1024,
    hurst_n_active_max: int = 1536,
    hurst_weight_beta: float = 0.25,
    hurst_weight_min: float = 0.9,
    hurst_weight_max: float = 1.1,
    hurst_weight_eps: float = 1e-6,
    hurst_weight_warmup_steps: int = 5,
    hurst_weight_apply_to: str = "kl_only",
) -> tuple[list[tinker.Datum], dict[str, Any], list[dict]]:
    """Converts the trajectories into a minibatch, and provides metrics about the minibatch"""

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Print one datum per dataset
    printed_datasets = set()
    for datum, metadata in zip(data_D, metadata_D):
        dataset_idx = dataset_indices_P[metadata["group_idx"]]
        if dataset_idx not in printed_datasets:
            logger.info(colorize_example(datum, tokenizer, key="mask"))
            printed_datasets.add(dataset_idx)

    # FDRL: Apply Fractal Regularization
    if fractal_alpha != 0:
        hurst_values = []
        for i, datum in enumerate(data_D):
            logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
            mask = datum.loss_fn_inputs["mask"].to_torch()
            active_logprobs = logprobs[mask > 0]
            
            H = calculate_hurst_efficient(active_logprobs)
            hurst_values.append(H)
            
            # Add fractal reward to advantages
            fractal_reward = fractal_alpha * H
            datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
                datum.loss_fn_inputs["advantages"].to_torch() + fractal_reward
            )
        metrics["optim/avg_hurst_online"] = float(np.mean(hurst_values))

    if golden_zone_mode not in {"none", "h"}:
        raise ValueError(f"Unsupported golden_zone_mode: {golden_zone_mode}")

    gz_penalties_for_telemetry = [0.0] * len(data_D)
    gz_excesses_for_telemetry = [0.0] * len(data_D)
    if golden_zone_mode == "h" and golden_zone_lambda > 0:
        gz_penalties = []
        gz_excesses = []
        gz_active = 0
        h_values = []
        for i, datum in enumerate(data_D):
            logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
            mask = datum.loss_fn_inputs["mask"].to_torch()
            active_logprobs = logprobs[mask > 0]
            h_value = calculate_hurst_efficient(active_logprobs)
            penalty, excess = calculate_golden_zone_h_penalty(
                h_value, golden_zone_h_target, golden_zone_h_delta
            )
            datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
                datum.loss_fn_inputs["advantages"].to_torch() - golden_zone_lambda * penalty
            )
            h_values.append(h_value)
            gz_penalties.append(penalty)
            gz_excesses.append(excess)
            gz_penalties_for_telemetry[i] = penalty
            gz_excesses_for_telemetry[i] = excess
            if excess > 0:
                gz_active += 1
        metrics["optim/h_student"] = float(np.mean(h_values))
        metrics["optim/gz_penalty_mean"] = float(np.mean(gz_penalties))
        metrics["optim/gz_penalty_median"] = float(np.median(gz_penalties))
        metrics["optim/gz_active_frac"] = float(gz_active / len(gz_penalties)) if gz_penalties else 0.0
        metrics["optim/gz_h_excess"] = float(np.mean(gz_excesses))
        metrics["optim/gz_h_target"] = float(golden_zone_h_target)
        metrics["optim/gz_h_delta"] = float(golden_zone_h_delta)

    # Incorporate KL penalty if configured
    telemetry_list = []
    
    # Extract individual rewards for each datum in data_D
    individual_rewards_D = []
    # Pre-compute total rewards for each group
    total_rewards_P = [group.get_total_rewards() for group in trajectory_groups_P]
    
    for metadata in metadata_D:
        group_idx = metadata["group_idx"]
        traj_idx = metadata["traj_idx"]
        # Use the pre-computed total reward for this trajectory
        reward = total_rewards_P[group_idx][traj_idx]
        individual_rewards_D.append(float(reward))

    if kl_penalty_coef > 0:
        with timed("compute_kl_penalty", metrics):
            teacher_clients_D = [
                teacher_clients[dataset_indices_P[metadata["group_idx"]]] for metadata in metadata_D
            ]
            dataset_indices_D = [
                dataset_indices_P[metadata["group_idx"]] for metadata in metadata_D
            ]
            kl_penalty_metrics, telemetry_list = await incorporate_kl_penalty(
                data_D,
                teacher_clients_D,
                dataset_indices_D,
                current_step,
                kl_penalty_coef,
                kl_discount_factor,
                hurst_weight_mode,
                hurst_num_windows,
                hurst_n_active_min,
                hurst_n_active_max,
                hurst_weight_beta,
                hurst_weight_min,
                hurst_weight_max,
                hurst_weight_eps,
                hurst_weight_warmup_steps,
                hurst_weight_apply_to,
            )
        metrics.update(kl_penalty_metrics)
    else:
        # If no KL penalty, create empty/basic telemetry list
        for i, datum in enumerate(data_D):
            telemetry_list.append({
                "logp_student": datum.loss_fn_inputs["logprobs"].to_torch().detach().cpu().numpy().tolist(),
                "logp_teacher": [], # No teacher logprobs calculated
                "step_kl": [],
                "advantages": datum.loss_fn_inputs["advantages"].to_torch().detach().cpu().numpy().tolist(),
            })
        
    # Add individual rewards to telemetry_list
    for i, reward in enumerate(individual_rewards_D):
        if i < len(telemetry_list):
            telemetry_list[i]["individual_reward"] = reward
            telemetry_list[i]["gz_penalty"] = gz_penalties_for_telemetry[i]
            telemetry_list[i]["gz_h_excess"] = gz_excesses_for_telemetry[i]
        
    return data_D, metrics, telemetry_list


@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    dataset_indices_P: List[int],
    teacher_clients: List[tinker.SamplingClient],
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    update_scope_context({"step": i_batch})

    metrics = {}
    data_D, prepare_minibatch_metrics, telemetry_list = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        dataset_indices_P,
        teacher_clients,
        current_step=i_batch,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        fractal_alpha=cfg.fractal_alpha,
        golden_zone_mode=cfg.golden_zone_mode,
        golden_zone_lambda=cfg.golden_zone_lambda,
        golden_zone_h_target=cfg.golden_zone_h_target,
        golden_zone_h_delta=cfg.golden_zone_h_delta,
        hurst_weight_mode=cfg.hurst_weight_mode,
        hurst_num_windows=cfg.hurst_num_windows,
        hurst_n_active_min=cfg.hurst_n_active_min,
        hurst_n_active_max=cfg.hurst_n_active_max,
        hurst_weight_beta=cfg.hurst_weight_beta,
        hurst_weight_min=cfg.hurst_weight_min,
        hurst_weight_max=cfg.hurst_weight_max,
        hurst_weight_eps=cfg.hurst_weight_eps,
        hurst_weight_warmup_steps=cfg.hurst_weight_warmup_steps,
        hurst_weight_apply_to=cfg.hurst_weight_apply_to,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D=data_D,
            training_client=training_client,
            learning_rate=cfg.learning_rate,
            num_substeps=cfg.num_substeps,
            loss_fn=cfg.loss_fn,
            loss_fn_config=cfg.loss_fn_config,
            metrics=metrics,
        )

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        data_D,
        training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
    )
    metrics.update(full_batch_metrics)

    # Export telemetry
    if telemetry_list:
        import json
        log_file = os.path.join(cfg.log_path, "telemetry_rollouts.jsonl")
        
        # Get common metrics for this step
        step_entropy = metrics.get("optim/entropy", 0.0)
        step_reward = metrics.get("env/all/reward/total", 0.0)
        
        with open(log_file, "a") as f:
            for datum, tel in zip(data_D, telemetry_list):
                # datum.loss_fn_inputs values are typically tinker.TensorData
                # .data attribute might already be a list
                tokens = datum.loss_fn_inputs["target_tokens"].data
                if hasattr(tokens, "tolist"):
                    tokens = tokens.tolist()
                
                mask = datum.loss_fn_inputs["mask"].data
                if hasattr(mask, "tolist"):
                    mask = mask.tolist()

                record = {
                    "tokens": tokens,
                    "mask": mask,
                    "step_entropy": step_entropy,
                    "step_reward": step_reward,
                    **tel
                }
                f.write(json.dumps(record) + "\n")
    return sampling_client, metrics


@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list[SamplingClientEvaluator],
    dataset: CompositeDataset,
    teacher_clients: List[tinker.SamplingClient],
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """Implements fully synchronous on-policy training"""

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every
    )

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                for evaluator in evaluators:
                    eval_metrics = await evaluator(sampling_client)
                    metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        # Get batch and sample trajectories
        env_group_builders_P, dataset_indices_P = dataset.get_batch(i_batch)
        with timed("sample", metrics):
            trajectory_groups_P = await asyncio.gather(
                *[
                    asyncio.create_task(
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            temperature=cfg.temperature,
                            max_tokens=cfg.max_tokens,
                            do_remove_constant_reward_groups=False,
                        ),
                        name=f"sample_task_{i}",
                    )
                    for i, builder in enumerate(env_group_builders_P)
                ],
            )
        trajectory_groups_P = [
            trajectory_group
            for trajectory_group in trajectory_groups_P
            if trajectory_group is not None
        ]

        # Train step
        sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            service_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
            dataset_indices_P,
            teacher_clients,
        )

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


@scope
async def main(
    cfg: Config,
):
    """Main training loop for on-policy distillation."""
    if cfg.golden_zone_mode not in {"none", "h"}:
        raise ValueError(f"Unsupported golden_zone_mode: {cfg.golden_zone_mode}")
    if cfg.golden_zone_lambda < 0:
        raise ValueError(f"golden_zone_lambda must be >= 0, got {cfg.golden_zone_lambda}")
    if not 0.0 <= cfg.golden_zone_h_target <= 1.0:
        raise ValueError(f"golden_zone_h_target must be in [0, 1], got {cfg.golden_zone_h_target}")
    if cfg.golden_zone_h_delta < 0:
        raise ValueError(f"golden_zone_h_delta must be >= 0, got {cfg.golden_zone_h_delta}")
    if cfg.hurst_weight_mode not in {"none", "std_t", "std_s"}:
        raise ValueError(f"Unsupported hurst_weight_mode: {cfg.hurst_weight_mode}")
    if cfg.hurst_weight_apply_to != "kl_only":
        raise ValueError(f"Unsupported hurst_weight_apply_to: {cfg.hurst_weight_apply_to}")
    if cfg.hurst_num_windows < 2:
        raise ValueError(f"hurst_num_windows must be >= 2, got {cfg.hurst_num_windows}")
    if cfg.hurst_n_active_min < cfg.hurst_num_windows:
        raise ValueError(
            f"hurst_n_active_min must be >= hurst_num_windows, got {cfg.hurst_n_active_min} < {cfg.hurst_num_windows}"
        )
    if cfg.hurst_n_active_min < 20 * cfg.hurst_num_windows:
        raise ValueError(
            "hurst_n_active_min is too small for stable Hurst estimation: "
            f"{cfg.hurst_n_active_min} < {20 * cfg.hurst_num_windows}"
        )
    if cfg.hurst_n_active_max < cfg.hurst_n_active_min:
        raise ValueError(
            f"hurst_n_active_max must be >= hurst_n_active_min, got "
            f"{cfg.hurst_n_active_max} < {cfg.hurst_n_active_min}"
        )
    if cfg.hurst_weight_min <= 0 or cfg.hurst_weight_max <= 0:
        raise ValueError("hurst_weight_min and hurst_weight_max must be positive")
    if cfg.hurst_weight_min > cfg.hurst_weight_max:
        raise ValueError("hurst_weight_min must be <= hurst_weight_max")

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    if cfg.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, cfg.renderer_name)

    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank, user_metadata=user_metadata
    )

    load_state_path: str | None = (
        resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    )
    if load_state_path:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, load_state_path, cfg.renderer_name
        )
        future = await training_client.load_state_with_optimizer_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create datasets and teacher sampling clients from configs
    datasets = []
    teacher_clients = []
    groups_per_batch_list = []
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]

    for dataset_config in cfg.dataset_configs:
        # Create dataset
        dataset, maybe_test_dataset = await dataset_config.dataset_builder()
        datasets.append(dataset)
        groups_per_batch_list.append(dataset_config.groups_per_batch)

        # Add test dataset evaluator if present
        if maybe_test_dataset is not None:
            evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens))

        # Create teacher sampling client
        teacher_config = dataset_config.teacher_config
        teacher_client = service_client.create_sampling_client(base_model=teacher_config.base_model)
        # Load teacher checkpoint if specified
        if teacher_config.load_checkpoint_path is not None:
            teacher_client = service_client.create_sampling_client(
                base_model=teacher_config.base_model,
                model_path=teacher_config.load_checkpoint_path,
            )
        teacher_clients.append(teacher_client)
        logger.info(
            f"Created teacher sampling client for {teacher_config.base_model} "
            f"(checkpoint: {teacher_config.load_checkpoint_path})"
        )

    # Wrap datasets in CompositeDataset
    composite_dataset = CompositeDataset(datasets, groups_per_batch_list)
    num_batches = len(composite_dataset)
    num_batches = min(cfg.max_step, num_batches) if cfg.max_step is not None else num_batches
    logger.info(f"Will train on {num_batches} batches (dataset has {num_batches})")

    # Training loop
    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        service_client=service_client,
        evaluators=evaluators,
        dataset=composite_dataset,
        teacher_clients=teacher_clients,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
