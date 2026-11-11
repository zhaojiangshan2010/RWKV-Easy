"""State Tuning：只训练每个 attention 层的初始循环状态。

RWKV-7 的每层 WKV 状态形状是 ``(n_head, head_size, head_size)``。State Tuning
把这块“初始记忆”当作可训练参数，而把整个基座冻结：

* 参数量极小（每层 ``H * K * K``，0.4B 模型通常 < 0.1%），训练非常稳；
* 与 infctx 组合时，参数作为 **第一块** 的初始状态，之后由真实状态接管；
* 导出的 checkpoint 只包含 ``*time_state``，需要用 :func:`merge_state_dict`
  合并进基座才能被普通推理代码加载。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "attach_state_tuning",
    "load_state_state_dict",
    "merge_state_dict",
    "state_parameter_summary",
    "state_state_dict",
]


def freeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def attach_state_tuning(model: nn.Module, init_std: float = 0.0, log: Any = None) -> list[nn.Parameter]:
    """注入 ``time_state`` 并冻结其余所有参数。"""
    if not hasattr(model, "attach_state_tuning"):
        raise TypeError("该模型不支持 State Tuning（缺少 attach_state_tuning）")

    freeze_all(model)
    params = model.attach_state_tuning(init_std=init_std)  # type: ignore[attr-defined]
    for parameter in params:
        parameter.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if log is not None:
        log.info(
            "State Tuning 就绪：%d 个 time_state，可训练 %.3fM / 总 %.2fM（%.4f%%）",
            len(params),
            trainable / 1e6,
            total / 1e6,
            100.0 * trainable / max(1, total),
        )
    return params


def state_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name.endswith("time_state"):
            result[name] = parameter.detach().cpu()
    if not result:
        raise RuntimeError("模型里没有 time_state 参数，无法导出 State Tuning 权重")
    return result


def load_state_state_dict(
    model: nn.Module, state: dict[str, torch.Tensor], strict: bool = True
) -> None:
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith("time_state")
    }
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    if strict and (missing or unexpected):
        raise RuntimeError(
            "State Tuning 权重与当前模型不匹配：\n"
            f"  缺少 {len(missing)} 个（例如 {missing[:4]}）\n"
            f"  多余 {len(unexpected)} 个（例如 {unexpected[:4]}）"
        )
    with torch.no_grad():
        for name, tensor in state.items():
            target = current.get(name)
            if target is None:
                continue
            target.copy_(tensor.to(target.device, target.dtype))


def merge_state_dict(
    base: dict[str, torch.Tensor], state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """把 ``time_state`` 覆盖进基座权重。"""
    unknown = [key for key in state if key not in base]
    if unknown:
        raise KeyError(
            "这些 State Tuning 参数在基座里不存在（模型结构不一致？）："
            + ", ".join(unknown[:5])
        )
    merged = dict(base)
    merged.update(state)
    return merged


def state_parameter_summary(model: nn.Module) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return (
        f"State 可训练参数 {trainable / 1e6:.3f}M / 总参数 {total / 1e6:.2f}M "
        f"({100.0 * trainable / max(1, total):.4f}%)"
    )
