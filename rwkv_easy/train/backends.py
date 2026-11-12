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
from datetime import timedelta
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
        local_rank = get_local_rank()
        available = torch.cuda.device_count()
        if available and local_rank >= available:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} 超出可见 GPU 数量 {available}。"
                f"请减小 dist.num_gpus，或检查 CUDA_VISIBLE_DEVICES。"
            )
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    # 默认 10 分钟太长；初始化阶段不该有这么长的等待。
    torch.distributed.init_process_group(
        backend=backend, init_method="env://", timeout=timedelta(seconds=300)
    )
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


class _HiddenForwardAdapter(nn.Module):
    """把 ``forward_hidden`` / ``forward_chunk_hidden`` 暴露成 ``forward``。

    DDP / DeepSpeed 只有在调用包装后的 ``forward`` 时才会准备 reducer、
    参数 gather 与通信 hook；训练循环如果直接调用原始模型的
    ``forward_hidden``，backward 时梯度根本不会 all-reduce。这个适配层让
    包装层看到正常的 forward/backward 对，同时保持原始模型引用不变
    （保存、合并、推理仍然使用未包装的模型）。
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self._forward_hidden_fn = model.forward_hidden
        self._forward_chunk_hidden_fn = model.forward_chunk_hidden

    def forward(self, input_ids: torch.Tensor, *args: Any, chunk: bool = False, **kwargs: Any) -> Any:
        fn = self._forward_chunk_hidden_fn if chunk else self._forward_hidden_fn
        return fn(input_ids, *args, **kwargs)

    def compile_forward(self, mode: str) -> None:
        # 只编译真正的模型前向，不编译 DDP / DeepSpeed 的通信包装。
        hidden = torch.compile(
            self.model.forward_hidden, dynamic=False, mode=mode
        )
        chunk = torch.compile(
            self.model.forward_chunk_hidden, dynamic=None, mode=mode
        )
        self._forward_hidden_fn = hidden
        self._forward_chunk_hidden_fn = chunk


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
        self._forward_hidden_fn = model.forward_hidden
        self._forward_chunk_hidden_fn = model.forward_chunk_hidden

    # -- 前向 ---------------------------------------------------------- #
    def forward_hidden(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self._forward_hidden_fn(input_ids, *args, **kwargs)

    def forward_chunk_hidden(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self._forward_chunk_hidden_fn(input_ids, *args, **kwargs)

    def compile_forward(self, mode: str) -> None:
        hidden = torch.compile(
            self.raw_model.forward_hidden, dynamic=False, mode=mode
        )
        chunk = torch.compile(
            self.raw_model.forward_chunk_hidden, dynamic=None, mode=mode
        )
        self._forward_hidden_fn = hidden
        self._forward_chunk_hidden_fn = chunk

    def chunk_sync_context(self, is_last_chunk: bool) -> contextlib.AbstractContextManager:
        # 单卡没有跨进程通信，不需要 no_sync。
        return contextlib.nullcontext()

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
        self.adapter = _HiddenForwardAdapter(model)
        self.model = nn.parallel.DistributedDataParallel(
            self.adapter,
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
        self._manual_sync = False
        self._sync_now = True
        self._manual_sync_pending = False

    # -- 前向 ---------------------------------------------------------- #
    def forward_hidden(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        # 必须走 DDP 包装层：DDP.forward 会 prepare reducer，backward 时才 all-reduce。
        return self.model(input_ids, *args, chunk=False, **kwargs)

    def forward_chunk_hidden(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.model(input_ids, *args, chunk=True, **kwargs)

    def compile_forward(self, mode: str) -> None:
        self.adapter.compile_forward(mode)

    def chunk_sync_context(self, is_last_chunk: bool) -> contextlib.AbstractContextManager:
        """infctx 逐块反传：除最后一块外都跳过 all-reduce。

        外层 ``sync_context`` 负责梯度累积边界；这里只处理“同一次
        micro-batch 内部的多个 chunk”。最后一块同步后，DDP 会把前面所有
        chunk 累积的梯度一次性 all-reduce。
        """
        if is_last_chunk:
            return contextlib.nullcontext()
        return self.model.no_sync()

    def sync_context(self, micro_step: int) -> contextlib.AbstractContextManager:
        self._sync_now = self.grad_accum <= 1 or (micro_step + 1) % self.grad_accum == 0
        if not self._sync_now:
            return self.model.no_sync()
        return contextlib.nullcontext()

    # -- 延迟 infctx 路径的手动同步 ------------------------------------- #
    def enable_manual_grad_sync(self) -> None:
        """DDP + infctx 延迟反传：不经过 DDP.forward，需要手动 all-reduce。

        多块一次 backward 时，DDP 的 reducer 没有稳定的 forward/backward 配对；
        这条路径改为原始模型逐块前向、一次 backward，再在梯度累积边界手动
        all-reduce 所有可训练参数。
        """
        self._manual_sync = True

    def _all_reduce_grads(self) -> None:
        if not torch.distributed.is_initialized() or self.dist.world_size <= 1:
            return
        grads = [
            parameter.grad
            for parameter in self.raw_model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not grads:
            return
        for grad in grads:
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
            grad.div_(self.dist.world_size)

    def backward(self, loss: torch.Tensor) -> None:
        if self._manual_sync and self.scaler is not None:
            raise RuntimeError(
                "DDP 延迟 infctx 暂不支持 fp16 GradScaler：各 rank 的缩放因子可能不同，"
                "手动 all-reduce 会得到错误梯度。请改用 bf16，或开启 "
                "train.infctx_parallel_backward。"
            )
        super().backward(loss)
        if self._manual_sync:
            if self._sync_now:
                self._all_reduce_grads()
                self._manual_sync_pending = False
            else:
                self._manual_sync_pending = True

    def step(self) -> bool:
        # 最后一个 micro-batch 恰好没有可训练 token 时，不能让累积梯度未同步就 step。
        if self._manual_sync and self._manual_sync_pending:
            self._all_reduce_grads()
            self._manual_sync_pending = False
        return super().step()

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

    # -- 前向 ---------------------------------------------------------- #
    def forward_hidden(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        # 走 engine.forward，ZeRO-3 的参数 gather / activation checkpoint 才会生效。
        return self.engine(input_ids, *args, chunk=False, **kwargs)

    def forward_chunk_hidden(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return self.engine(input_ids, *args, chunk=True, **kwargs)

    def compile_forward(self, mode: str) -> None:
        compile_fn = getattr(self.engine.module, "compile_forward", None)
        if compile_fn is None:
            raise RuntimeError("DeepSpeed 包装层不支持 compile_forward")
        compile_fn(mode)

    def chunk_sync_context(self, is_last_chunk: bool) -> contextlib.AbstractContextManager:
        # DeepSpeed 暂不支持逐块反传；这个方法只用于统一接口。
        return contextlib.nullcontext()

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
