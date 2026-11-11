"""优化器与学习率调度。"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from ..config import TrainConfig

__all__ = ["build_optimizer", "build_scheduler", "describe_optimizer"]


def _split_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """返回 ``(需要 weight decay, 不需要 weight decay)`` 两组参数。

    经验规则：
    * 1 维参数（bias、LayerNorm/GroupNorm 权重）不做 weight decay；
    * ``time_state`` 是“初始记忆”，等价于偏置，也不做 decay；
    * 其余 >=2 维的权重做 decay。
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.dim() < 2 or name.endswith("time_state"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return decay, no_decay


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = _split_parameters(model)
    if not decay and not no_decay:
        raise RuntimeError("没有任何可训练参数：请检查 LoRA 注入或 State Tuning 是否成功。")
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": cfg.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})

    kwargs: dict[str, Any] = {
        "lr": cfg.lr,
        "betas": (cfg.beta1, cfg.beta2),
        "eps": 1e-8,
    }
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except (RuntimeError, TypeError):  # pragma: no cover - 个别平台不支持 fused
            pass
    return torch.optim.AdamW(groups, **kwargs)


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup = max(0, int(cfg.warmup_steps))
    ratio = max(0.0, min(1.0, cfg.min_lr / cfg.lr if cfg.lr > 0 else 0.0))
    schedule = cfg.lr_schedule

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        if schedule == "constant":
            return 1.0
        if schedule == "linear":
            return ratio + (1.0 - ratio) * (1.0 - progress)
        # cosine
        return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def describe_optimizer(model: nn.Module, cfg: TrainConfig) -> str:
    decay, no_decay = _split_parameters(model)
    n_decay = sum(p.numel() for p in decay)
    n_no_decay = sum(p.numel() for p in no_decay)
    return (
        f"AdamW(lr={cfg.lr:g}, betas=({cfg.beta1},{cfg.beta2}), wd={cfg.weight_decay:g}) "
        f"| decay {n_decay / 1e6:.2f}M + no-decay {n_no_decay / 1e3:.1f}K"
    )
