"""训练后端：单卡 / DDP / DeepSpeed 的统一接口。

主训练循环只依赖这几个动作，因此不需要为每种并行方式写一份循环：

* ``sync_context(micro_step)``：梯度累积时避免每步 all-reduce
* ``backward(loss)`` / ``step()`` / ``zero_grad()``
* ``unwrap()``：拿到未包装的原始模型（保存权重用）
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..runtime import get_local_rank, get_rank, get_world_size

__all__ = [
    "DDPBackend",
    "DeepSpeedBackend",
    "DistInfo",
    "SingleBackend",
    "broadcast_object",
    "build_backend",
    "destroy_distributed",
    "init_distributed",
]


# --------------------------------------------------------------------------- #
# 进程组
# --------------------------------------------------------------------------- #


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def init_distributed(device: torch.device) -> DistInfo:
    """初始化 torch.distributed（torchrun 启动时环境变量已就绪）。"""
    if not torch.distributed.is_available():
        return DistInfo(0, 0, 1)
    if torch.distributed.is_initialized():
        return DistInfo(
            torch.distributed.get_rank(),
            int(os.environ.get("LOCAL_RANK", "0")),
            torch.distributed.get_world_size(),
        )
    world_size = get_world_size()
    if world_size <= 1:
        return DistInfo(0, 0, 1)

    if device.type == "cuda":
        torch.cuda.set_device(get_local_rank())
        backend = "nccl"
    else:
        backend = "gloo"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.distributed.init_process_group(backend=backend, init_method="env://")
    return DistInfo(get_rank(), get_local_rank(), world_size)


def destroy_distributed() -> None:
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
        finally:
            torch.distributed.destroy_process_group()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """把主进程算出来的字符串/字典广播到所有 rank（例如 run 目录、decay 模式）。"""
    if not torch.distributed.is_initialized():
        return obj
    payload = [obj]
    torch.distributed.broadcast_object_list(payload, src=src)
    return payload[0]


# --------------------------------------------------------------------------- #
# 后端实现
# --------------------------------------------------------------------------- #


class SingleBackend:
    kind = "single"

    def __init__(
        self, model: nn.Module, optimizer: Any, scheduler: Any, dist: DistInfo, **_: Any
    ) -> None:
        self.raw_model = model
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dist = dist
        self.scaler: Any = None
        self.grad_clip: float = 0.0

    # -- 训练动作 ------------------------------------------------------ #
    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _trainable(self):
        return [p for p in self.raw_model.parameters() if p.requires_grad]

    def step(self) -> bool:
        if self.grad_clip and self.grad_clip > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self._trainable(), self.grad_clip)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return True

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def sync_context(self, micro_step: int) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()

    def unwrap(self) -> nn.Module:
        return self.raw_model

    def barrier(self) -> None:
        return None


class DDPBackend(SingleBackend):
    kind = "ddp"

    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any,
        dist: DistInfo,
        grad_accum: int = 1,
        find_unused_parameters: bool = False,
        **_: Any,
    ) -> None:
        self.raw_model = model
        is_cuda = next(model.parameters()).is_cuda
        self.model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.local_rank] if is_cuda else None,
            output_device=dist.local_rank if is_cuda else None,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=True,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dist = dist
        self.scaler: Any = None
        self.grad_clip: float = 0.0
        self.grad_accum = max(1, int(grad_accum))

    def sync_context(self, micro_step: int) -> contextlib.AbstractContextManager:
        if self.grad_accum > 1 and (micro_step + 1) % self.grad_accum != 0:
            return self.model.no_sync()
        return contextlib.nullcontext()

    def unwrap(self) -> nn.Module:
        return self.raw_model


class DeepSpeedBackend:
    kind = "deepspeed"

    def __init__(
        self, engine: Any, optimizer: Any, scheduler: Any, dist: DistInfo, raw_model: nn.Module
    ) -> None:
        self.engine = engine
        self.model = engine
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dist = dist
        self.raw_model = raw_model
        self.scaler: Any = None
        self.grad_clip: float = 0.0  # 由 DeepSpeed 的 gradient_clipping 负责

    def backward(self, loss: torch.Tensor) -> None:
        self.engine.backward(loss)

    def step(self) -> bool:
        # DeepSpeed 的 engine.step() 在梯度累积边界上会自己调用 optimizer 和
        # lr_scheduler 的 step()，所以这里不能再重复调用调度器。
        self.engine.step()
        return True

    def zero_grad(self) -> None:
        self.engine.zero_grad()

    def sync_context(self, micro_step: int) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()

    def unwrap(self) -> nn.Module:
        return self.raw_model

    def barrier(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    @property
    def is_zero3(self) -> bool:
        try:
            return int(self.engine.zero_optimization_stage()) == 3
        except Exception:  # pragma: no cover
            return False


def build_backend(
    kind: str,
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    dist: DistInfo,
    cfg: Any,
    log: Any = None,
) -> SingleBackend | DDPBackend:
    if kind == "single":
        backend: SingleBackend | DDPBackend = SingleBackend(model, optimizer, scheduler, dist)
    elif kind == "ddp":
        if dist.world_size <= 1:
            backend = SingleBackend(model, optimizer, scheduler, dist)
        else:
            if log is not None:
                log.info("使用 DDP：%d 个进程", dist.world_size)
            backend = DDPBackend(
                model,
                optimizer,
                scheduler,
                dist,
                grad_accum=cfg.train.grad_accum,
                find_unused_parameters=cfg.dist.find_unused_parameters,
            )
    else:
        raise ValueError(
            f"build_backend 不支持 {kind!r}；DeepSpeed 请走 train/deepspeed_backend.py"
        )
    backend.grad_clip = float(cfg.train.grad_clip)
    return backend
