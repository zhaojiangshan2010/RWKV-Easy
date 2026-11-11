"""运行时工具：日志、设备、精度、随机种子、原子写盘。

这一层刻意不依赖项目内其它模块，方便在 ``config`` / ``cli`` 中早期调用。
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# 分布式环境
# --------------------------------------------------------------------------- #


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def is_distributed_launched() -> bool:
    """是否由 torchrun / deepspeed 启动（而不是用户直接 python train.py）。"""
    return get_world_size() > 1


# --------------------------------------------------------------------------- #
# 设备与精度
# --------------------------------------------------------------------------- #

_PRECISIONS = ("fp32", "fp16", "bf16", "bf16-full")


def resolve_device(name: str = "auto") -> torch.device:
    """把配置里的设备名解析成 ``torch.device``。

    ``auto`` 优先 CUDA（ROCm 在 PyTorch 里同样表现为 ``cuda``），其次 MPS，最后 CPU。
    """
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "配置要求使用 CUDA，但当前 PyTorch 检测不到可用 GPU。\n"
            "  1) 运行 `rwkv-easy doctor` 检查驱动与 torch 版本；\n"
            "  2) 或在配置里设置 `train.device: cpu` 先做小规模验证。"
        )
    return torch.device(name)


def device_name(device: torch.device) -> str:
    """给日志用的可读设备名。"""
    if device.type == "cuda":
        try:
            idx = device.index if device.index is not None else torch.cuda.current_device()
            return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
        except Exception:  # pragma: no cover - 驱动异常时不要影响训练
            return str(device)
    return str(device)


def device_description() -> str:
    """``doctor`` / 启动横幅用的硬件信息。"""
    parts = [f"torch {torch.__version__}"]
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        parts.append(f"CUDA {torch.version.cuda} / {n} 张 GPU")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            parts.append(f"  [{i}] {props.name}  {props.total_memory / 2**30:.1f} GiB")
        if torch.version.hip:
            parts.append(f"ROCm {torch.version.hip}")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        parts.append("Apple MPS")
    else:
        parts.append("CPU")
    return "\n".join(parts)


def validate_precision(precision: str, device: torch.device) -> None:
    precision = normalize_precision(precision)
    if precision == "fp16" and device.type == "cpu":
        raise RuntimeError(
            "CPU 上不支持 fp16 混合精度训练（会非常慢且容易 NaN）。\n"
            "请改用 `train.precision: fp32` 或 `bf16`。"
        )
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 bf16，请改用 `train.precision: fp16` 或 `fp32`。")


def normalize_precision(precision: str) -> str:
    precision = str(precision).lower().strip()
    aliases = {
        "float32": "fp32",
        "32": "fp32",
        "float16": "fp16",
        "half": "fp16",
        "16": "fp16",
        "bfloat16": "bf16",
        "bf16-true": "bf16-full",
        "bf16_true": "bf16-full",
    }
    precision = aliases.get(precision, precision)
    if precision not in _PRECISIONS:
        raise ValueError(
            f"未知精度 {precision!r}，可选：{', '.join(_PRECISIONS)}。\n"
            "  fp32 = 全精度；fp16 = 混合精度；bf16 = 混合精度（推荐）；"
            "bf16-full = 参数本身用 bf16（最省显存）。"
        )
    return precision


def storage_dtype(precision: str) -> torch.dtype:
    """模型参数在显存/内存中的 dtype。"""
    precision = normalize_precision(precision)
    if precision == "bf16-full":
        return torch.bfloat16
    if precision == "fp16":
        # fp16 采用混合精度：参数保持 fp32，matmul 用 autocast。
        return torch.float32
    return torch.float32


@contextlib.contextmanager
def autocast_context(precision: str, device: torch.device):
    """训练前向的 autocast 上下文。

    * ``fp32``     -> 不启用
    * ``fp16``     -> CUDA 上启用 autocast（配 GradScaler）
    * ``bf16``     -> 启用 autocast（参数仍是 fp32）
    * ``bf16-full``-> 不启用（参数本身就是 bf16）
    """
    precision = normalize_precision(precision)
    if precision == "fp16":
        if device.type == "cpu":
            yield
        else:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                yield
        return
    if precision == "bf16":
        if device.type in ("cuda", "cpu"):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                yield
        else:  # pragma: no cover - mps 等
            yield
        return
    yield


# --------------------------------------------------------------------------- #
# 随机种子
# --------------------------------------------------------------------------- #


def seed_everything(seed: int, deterministic: bool = False) -> None:
    seed = int(seed) & 0x7FFFFFFF
    # DataLoader 的 worker_init_fn 在子进程里读取这个环境变量。
    os.environ["RWKV_EASY_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    # TF32：A100/H100 上能明显加速 fp32 matmul，对训练精度影响可接受。
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def worker_init_fn(worker_id: int) -> None:
    """DataLoader 子进程的种子，保证各 rank 不重复。"""
    base = int(os.environ.get("RWKV_EASY_SEED", "42"))
    rank = get_rank()
    np.random.seed(base + rank * 1000 + worker_id)
    random.seed(base + rank * 1000 + worker_id)


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_LOG_DATE = "%m-%d %H:%M:%S"


class _RankFilter(logging.Filter):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        if self.rank == 0:
            return True
        record.msg = f"[rank{self.rank}] {record.msg}"
        return True


def setup_logger(
    name: str = "rwkv-easy",
    log_file: str | Path | None = None,
    rank: int = 0,
    level: str = "info",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    # 重复调用时先清空，避免日志翻倍。
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    if rank == 0:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATE))
        logger.addHandler(console)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATE))
        logger.addHandler(file_handler)
    logger.addFilter(_RankFilter(rank))
    return logger


def get_logger(name: str = "rwkv-easy") -> logging.Logger:
    return logging.getLogger(name)


class MetricsLogger:
    """把每步指标同时写到 JSONL（便于画曲线）和内存（便于计算平滑值）。"""

    def __init__(self, path: str | Path | None, enabled: bool = True) -> None:
        self.path = Path(path) if path else None
        self.enabled = enabled and self.path is not None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record = {**record, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class EMALoss:
    """指数滑动平均，日志里比逐步 loss 好看得多。"""

    def __init__(self, beta: float = 0.98) -> None:
        self.beta = beta
        self.value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.beta * self.value + (1 - self.beta) * float(x)
        return self.value


# --------------------------------------------------------------------------- #
# 杂项
# --------------------------------------------------------------------------- #


def format_count(n: float) -> str:
    n = float(n)
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{unit}"
    return f"{n:.0f}"


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PiB"


def atomic_save(obj: Any, path: str | Path) -> None:
    """先写临时文件再原子替换，避免训练中断留下半个 checkpoint。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(obj, tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp_name, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    params = module.parameters()
    return sum(p.numel() for p in params if (p.requires_grad or not trainable_only))


def iter_trainable(module: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Parameter]]:
    for name, param in module.named_parameters():
        if param.requires_grad:
            yield name, param


def make_grad_scaler(device: torch.device):
    """fp16 的 GradScaler，兼容 torch 2.2（没有 ``torch.amp.GradScaler``）。"""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type)
    return torch.cuda.amp.GradScaler(enabled=True)  # pragma: no cover - torch < 2.3


def peak_memory_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 2**30


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
