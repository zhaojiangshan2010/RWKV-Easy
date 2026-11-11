"""checkpoint 的保存 / 读取 / 续训 / 导出。"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..lora import lora_state_dict
from ..lora import merged_state_dict as lora_merged_state_dict
from ..runtime import atomic_save, atomic_write_text
from ..state_tuning import state_state_dict

__all__ = [
    "FORMAT_NAME",
    "AdapterInfo",
    "adapter_filename",
    "export_merged",
    "latest_pointer",
    "load_trainer_state",
    "prune_checkpoints",
    "read_adapter",
    "save_adapter",
    "save_trainer_state",
    "set_latest_pointer",
]

FORMAT_NAME = "rwkv-easy"
FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# 适配器 / 状态权重
# --------------------------------------------------------------------------- #


@dataclass
class AdapterInfo:
    method: str
    step: int
    epoch: int
    weights: dict[str, torch.Tensor]
    base_signature: str = ""
    model_config: dict[str, Any] = field(default_factory=dict)
    decay: str = "clampw"
    method_config: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"{self.method} @ step {self.step} | {len(self.weights)} 个张量 | "
            f"base={self.base_signature or '未知'}"
        )


def adapter_filename(method: str, step: int) -> str:
    prefix = "adapter" if method == "lora" else "state"
    return f"{prefix}-step{step:07d}.pth"


def save_adapter(
    path: str | Path,
    *,
    method: str,
    step: int,
    epoch: int,
    weights: dict[str, torch.Tensor],
    base_signature: str = "",
    model_config: dict[str, Any] | None = None,
    decay: str = "clampw",
    method_config: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "method": method,
        "step": int(step),
        "epoch": int(epoch),
        "base_signature": base_signature,
        "model_config": model_config or {},
        "decay": decay,
        "method_config": method_config or {},
        "weights": {k: v.detach().cpu() for k, v in weights.items()},
    }
    atomic_save(payload, path)
    return path


def read_adapter(path: str | Path) -> AdapterInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到权重文件：{path}")
    # weights_only=True：只允许张量/基本类型，防止加载来源不明的 checkpoint 时
    # 触发任意代码执行。本项目写出的文件都在这个白名单内。
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"{path} 不是有效的 RWKV-EASY 权重文件（安全反序列化失败：{exc}）。"
        ) from exc
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise ValueError(
            f"{path} 不是 RWKV-EASY 导出的适配器/状态权重。"
            "（如果你手里是完整模型权重，请用 --model 指定，而不是 --resume）"
        )
    if "weights" not in payload:
        raise ValueError(f"{path} 缺少 weights 字段")
    return AdapterInfo(
        method=payload.get("method", "lora"),
        step=int(payload.get("step", 0)),
        epoch=int(payload.get("epoch", 0)),
        weights=payload["weights"],
        base_signature=payload.get("base_signature", ""),
        model_config=payload.get("model_config", {}),
        decay=payload.get("decay", "clampw"),
        method_config=payload.get("method_config", {}),
    )


def collect_weights(model: nn.Module, method: str) -> dict[str, torch.Tensor]:
    if method == "lora":
        return lora_state_dict(model)
    if method == "state":
        return state_state_dict(model)
    raise ValueError(f"未知训练方法：{method!r}")


# --------------------------------------------------------------------------- #
# 训练过程状态（优化器 / 进度）
# --------------------------------------------------------------------------- #


def trainer_state_filename(step: int) -> str:
    return f"trainer-step{step:07d}.pt"


def save_trainer_state(
    run_dir: str | Path,
    *,
    step: int,
    epoch: int,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    best_loss: float,
) -> Path:
    path = Path(run_dir) / trainer_state_filename(step)
    payload = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "step": int(step),
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    atomic_save(payload, path)
    return path


def load_trainer_state(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


# --------------------------------------------------------------------------- #
# latest 指针与清理
# --------------------------------------------------------------------------- #


def latest_pointer(run_dir: str | Path) -> Path | None:
    pointer = Path(run_dir) / "latest.txt"
    if not pointer.exists():
        return None
    text = pointer.read_text(encoding="utf-8").strip()
    if not text:
        return None
    candidate = Path(run_dir) / text
    return candidate if candidate.exists() else None


def set_latest_pointer(run_dir: str | Path, filename: str) -> None:
    atomic_write_text(filename + "\n", Path(run_dir) / "latest.txt")


_STEP_RE = re.compile(r"step(\d+)\.(?:pth|pt)$")


def prune_checkpoints(run_dir: str | Path, keep_last: int) -> None:
    """只保留最近 ``keep_last`` 组 (adapter/state + trainer) 文件。"""
    if keep_last <= 0:
        return
    run_dir = Path(run_dir)
    steps: dict[int, list[Path]] = {}
    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        match = _STEP_RE.search(path.name)
        if match:
            steps.setdefault(int(match.group(1)), []).append(path)
    if len(steps) <= keep_last:
        return
    for step in sorted(steps)[: -keep_last]:
        for path in steps[step]:
            with contextlib.suppress(OSError):
                path.unlink()


# --------------------------------------------------------------------------- #
# 导出完整权重
# --------------------------------------------------------------------------- #


def export_merged(
    model: nn.Module,
    method: str,
    path: str | Path,
    dtype: torch.dtype | None = None,
) -> Path:
    """导出可直接给推理代码加载的完整权重。"""
    path = Path(path)
    if method == "lora":
        state_dict = lora_merged_state_dict(model, dtype=dtype)
    elif method == "state":
        state_dict = {
            key: value.detach().to("cpu", dtype=dtype or value.dtype)
            for key, value in model.state_dict().items()
        }
    else:
        raise ValueError(f"未知训练方法：{method!r}")
    atomic_save(state_dict, path)
    return path
