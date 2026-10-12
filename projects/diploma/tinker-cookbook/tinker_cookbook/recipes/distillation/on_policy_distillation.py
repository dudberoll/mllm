"""
On-policy distillation for reasoning and chat tasks.

This script implements on-policy distillation where a student model learns from
a teacher model by minimizing KL divergence. No correctness or format rewards
are used - only KL penalty provides supervision.

Example usage:
    # For reasoning tasks (DeepMath)
    python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
        model_name=Qwen/Qwen3-8B-Base \
        dataset=deepmath \
        learning_rate=1e-4 \
        groups_per_batch=1024 \
        lora_rank=128 \
        wandb_project=cookbook_distillation

    # For chat tasks (Tulu3)
    python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
        model_name=Qwen/Qwen3-8B-Base \
        dataset=tulu3 \
        learning_rate=1e-4 \
        groups_per_batch=1024 \
        lora_rank=128 \
        wandb_project=cookbook_distillation
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

import chz
from tinker.types import LossFnType
from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDatasetBuilder,
    TeacherConfig,
)

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for on-policy distillation."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B-Base"  # Student model
    lora_rank: int = 128
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None  # Student checkpoint

    # Teacher configuration
    teacher_model: str = "Qwen/Qwen3-32B"
    teacher_checkpoint: str | None = None

    # Dataset configuration
    dataset: str = "deepmath"  # Options: deepmath, tulu3

    # Training hyperparameters
    group_size: int = 4  # Number of rollouts per prompt
    groups_per_batch: int = 1024
    learning_rate: float = 1e-4
    max_tokens: int = 4096
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Optimizer configuration
    num_substeps: int = 1

    # Loss function and configuration.
    # See https://tinker-docs.thinkingmachines.ai/losses
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Fractal RL parameters
    fractal_alpha: float = 0.0  # Coefficient for fractal regularization (FDRL)
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

    # Evaluation and checkpointing
    eval_every: int = 10
    save_every: int = 10
    max_step: int | None = None  # Maximum number of training steps

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get renderer name
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    # Create log path if not specified
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_name = cli_config.model_name.replace("/", "-")
        temperature_suffix = f"-t{cli_config.temperature}" if cli_config.temperature != 1.0 else ""
        hurst_suffix = ""
        if cli_config.hurst_weight_mode != "none":
            hurst_suffix = (
                f"-{cli_config.hurst_weight_mode}"
                f"-nw{cli_config.hurst_num_windows}"
                f"-b{cli_config.hurst_weight_beta}"
            )
        run_name = (
            f"distill-{cli_config.dataset}-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"{cli_config.groups_per_batch}batch{temperature_suffix}{hurst_suffix}-"
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/distillation/{run_name}")

    # Create wandb name if not specified
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = os.path.basename(log_path)

    # Create dataset builder
    if cli_config.dataset == "deepmath":
        from tinker_cookbook.recipes.math_rl.math_env import get_math_dataset_builder
        dataset_builder = get_math_dataset_builder(
            dataset_name="deepmath",
            batch_size=cli_config.groups_per_batch,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
        )
    else:
        dataset_builder = PromptOnlyDatasetBuilder(
            dataset_name=cli_config.dataset,
            groups_per_batch=cli_config.groups_per_batch,
            group_size=cli_config.group_size,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
        )

    # Create teacher config
    teacher_config = TeacherConfig(
        base_model=cli_config.teacher_model,
        load_checkpoint_path=cli_config.teacher_checkpoint,
    )

    # Create distillation dataset config
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=cli_config.groups_per_batch,
    )

    # Create full config
    config = train_on_policy.Config(
        learning_rate=cli_config.learning_rate,
        dataset_configs=[dataset_config],
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        num_substeps=cli_config.num_substeps,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        max_step=cli_config.max_step,
        fractal_alpha=cli_config.fractal_alpha,
        hurst_weight_mode=cli_config.hurst_weight_mode,
        hurst_num_windows=cli_config.hurst_num_windows,
        hurst_n_active_min=cli_config.hurst_n_active_min,
        hurst_n_active_max=cli_config.hurst_n_active_max,
        hurst_weight_beta=cli_config.hurst_weight_beta,
        hurst_weight_min=cli_config.hurst_weight_min,
        hurst_weight_max=cli_config.hurst_weight_max,
        hurst_weight_eps=cli_config.hurst_weight_eps,
        hurst_weight_warmup_steps=cli_config.hurst_weight_warmup_steps,
        hurst_weight_apply_to=cli_config.hurst_weight_apply_to,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await train_on_policy.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
