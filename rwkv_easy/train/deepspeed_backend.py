"""DeepSpeed 集成（可选依赖）。

不安装 deepspeed 时，本模块不会被导入；训练会走单卡或 DDP。

设计取舍
--------
* **梯度累积仍由主循环负责**：DeepSpeed 的 ``gradient_accumulation_steps`` 设为 1，
  这样单卡 / DDP / DeepSpeed 三条路径的语义完全一致。
* **只把可训练参数交给优化器**：LoRA / State 场景下基座是冻结的，ZeRO-1/2 足够；
  ZeRO-3 也能跑，但对 LoRA 性价比低（配置校验里会提醒）。
* **保存**：ZeRO-3 下用 ``GatheredParameters`` 在 rank 0 汇总，其余情况直接取。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..runtime import is_main_process
from .backends import DeepSpeedBackend, DistInfo

__all__ = ["build_deepspeed_config", "create_deepspeed_backend", "gather_trainable_state_dict"]


def build_deepspeed_config(cfg: Any, world_size: int) -> dict[str, Any]:
    """由统一的 YAML 配置生成 DeepSpeed JSON（纯函数，便于单测）。"""
    ds = cfg.dist.deepspeed
    micro = int(cfg.train.micro_batch_size)
    stage = int(ds.zero_stage)

    zero: dict[str, Any] = {"stage": stage}
    if stage >= 1:
        zero["reduce_bucket_size"] = float(ds.bucket_size)
    if stage >= 2:
        zero["allgather_bucket_size"] = float(ds.bucket_size)
        zero["contiguous_gradients"] = bool(ds.contiguous_gradients)
        zero["overlap_comm"] = bool(ds.overlap_comm)
    if ds.offload_optimizer and stage >= 1:
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    if ds.offload_param:
        if stage != 3:
            raise ValueError("dist.deepspeed.offload_param 需要 zero_stage: 3")
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}
    if stage == 3:
        zero["stage3_gather_16bit_weights_on_model_save"] = True

    precision = cfg.train.precision
    config: dict[str, Any] = {
        # 梯度累积由主循环手动完成（engine.step() 每 accum 个 micro-step 调一次），
        # 所以 DS 侧 gradient_accumulation_steps 固定为 1；train_batch_size 只用于
        # DS 的一致性校验，必须按 micro * world（不再乘 accum）填写。
        "train_batch_size": micro * max(1, world_size),
        "train_micro_batch_size_per_gpu": micro,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": float(cfg.train.grad_clip),
        "steps_per_print": 10**9,
        "wall_clock_breakdown": False,
        "zero_optimization": zero,
        "zero_allow_untested_optimizer": True,
        "fp16": {"enabled": precision == "fp16", "initial_scale_power": 12},
        "bf16": {"enabled": precision in ("bf16", "bf16-full")},
    }
    return config


def _init_single_process_group() -> None:
    """为单进程 DeepSpeed 建一个 world_size=1 的进程组。"""
    import os
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    # DeepSpeed 0.14 会直接读取 LOCAL_RANK。
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend, init_method="env://",
                                         world_size=1, rank=0)


def create_deepspeed_backend(
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    dist: DistInfo,
    cfg: Any,
    log: Any = None,
) -> DeepSpeedBackend:
    try:
        import deepspeed  # type: ignore
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise ImportError(
            "你设置了 `dist.mode: deepspeed`，但没有安装 DeepSpeed。\n"
            "  安装：pip install deepspeed\n"
            "  或者改成 `dist.mode: ddp`（多卡） / `dist.mode: single`（单卡），"
            "不需要 DeepSpeed 也能训练。"
        ) from exc

    if torch.cuda.is_available() is False:
        raise RuntimeError("DeepSpeed 训练需要 NVIDIA GPU / ROCm；当前没有可用 GPU。")

    ds_config = build_deepspeed_config(cfg, dist.world_size)
    if log is not None:
        log.info(
            "DeepSpeed 初始化：ZeRO-%d%s%s",
            cfg.dist.deepspeed.zero_stage,
            " + optimizer offload" if cfg.dist.deepspeed.offload_optimizer else "",
            " + param offload" if cfg.dist.deepspeed.offload_param else "",
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    # 单进程（没有 torchrun）时 DeepSpeed 没有进程组，会去探测 MPI 然后报
    # `No module named 'mpi4py'`。这里先建一个单进程 PG，再让 DS 复用。
    if not torch.distributed.is_initialized():
        _init_single_process_group()
    engine, ds_optimizer, _, ds_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=trainable,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
        dist_init_required=False,
    )
    return DeepSpeedBackend(
        engine=engine,
        optimizer=ds_optimizer,
        scheduler=ds_scheduler if ds_scheduler is not None else scheduler,
        dist=dist,
        raw_model=model,
    )


def gather_trainable_state_dict(backend: DeepSpeedBackend) -> dict[str, torch.Tensor] | None:
    """在 rank 0 汇总可训练参数（兼容 ZeRO-3 分片）。"""
    model = backend.raw_model
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named:
        return {} if is_main_process() else None

    if backend.is_zero3:
        import deepspeed.zero as ds_zero  # type: ignore

        with ds_zero.GatheredParameters([p for _, p in named], modifier_rank=0):
            if not is_main_process():
                return None
            return {name: p.detach().cpu().clone() for name, p in named}

    if not is_main_process():
        return None
    return {name: p.detach().cpu().clone() for name, p in named}
