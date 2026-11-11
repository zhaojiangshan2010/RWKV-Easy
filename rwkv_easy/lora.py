"""LoRA：零依赖实现（不依赖 HuggingFace peft）。

为什么不用 ``peft``？

* RWKV 的 infctx 需要把循环状态作为显式参数传进模型。HF PEFT 的
  ``PeftModelForCausalLM.forward`` 签名会把这些参数当成 ``attention_mask`` /
  ``inputs_embeds`` 吃掉，这正是参考实现里 ``--train_type infctx --peft lora``
  崩溃的原因。
* 自己实现后，模型的 ``forward`` 签名完全不变，LoRA 只是把 ``nn.Linear``
  换成 :class:`LoRALinear`，因此与 infctx、State Tuning、DDP、DeepSpeed 都能
  自然组合。

数学：``y = W x + (B A x) * alpha / r``，``B`` 零初始化，因此训练开始时输出与
基座完全一致。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "LoRALinear",
    "inject_lora",
    "lora_modules",
    "lora_parameter_summary",
    "lora_state_dict",
    "merge_and_unload",
    "merged_state_dict",
]


class LoRALinear(nn.Module):
    """把冻结的 ``nn.Linear`` 包一层低秩增量。"""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear 只能包装 nn.Linear，收到 {type(base).__name__}")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        in_features = base.in_features
        out_features = base.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))
        # 标准 LoRA 初始化：A 随机、B 为零。
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        A = self.lora_A.to(x.dtype)
        B = self.lora_B.to(x.dtype)
        delta = self.dropout(x) @ A.transpose(0, 1)
        delta = delta @ B.transpose(0, 1)
        return base_out + delta * self.scaling

    # ------------------------------------------------------------------ #
    # 合并
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        A = self.lora_A.to(self.base.weight.device, self.base.weight.dtype)
        B = self.lora_B.to(self.base.weight.device, self.base.weight.dtype)
        return self.base.weight + (B @ A) * self.scaling

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """返回一个权重已合并的普通 ``nn.Linear``。"""
        layer = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        layer.weight.copy_(self.merged_weight())
        if self.base.bias is not None:
            layer.bias.copy_(self.base.bias)
        layer.train(self.training)
        return layer

    def extra_repr(self) -> str:
        return (
            f"in={self.base.in_features}, out={self.base.out_features}, "
            f"r={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4g}"
        )


# --------------------------------------------------------------------------- #
# 注入
# --------------------------------------------------------------------------- #


@dataclass
class InjectionReport:
    targets: list[str]
    trainable_parameters: int
    total_parameters: int


def _matches(name: str, suffixes: list[str], regex: re.Pattern[str] | None) -> bool:
    if any(name.endswith(suffix) for suffix in suffixes):
        return True
    return bool(regex and regex.search(name))


def _parent_and_attr(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    target_regex: str | None = None,
    log: Any = None,
) -> InjectionReport:
    """按模块名后缀注入 LoRA，返回注入报告。"""
    if rank <= 0:
        raise ValueError("LoRA rank 必须 > 0")
    if alpha <= 0:
        raise ValueError("LoRA alpha 必须 > 0")
    regex = re.compile(target_regex) if target_regex else None

    candidates: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _matches(name, target_modules, regex):
            candidates.append(name)

    if not candidates:
        raise ValueError(
            "没有匹配到任何 nn.Linear，LoRA 无法注入。\n"
            f"target_modules={target_modules}, target_regex={target_regex!r}\n"
            "模型里可用的 Linear："
            + ", ".join(
                name for name, module in model.named_modules() if isinstance(module, nn.Linear)
            )[:800]
        )

    # LoRA 的定义就是“只训练适配器”：先把整个基座冻住（包括 emb / head / norm），
    # 再注入 LoRA 参数（nn.Parameter 默认 requires_grad=True）。
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for name in candidates:
        parent, attr = _parent_and_attr(model, name)
        module = getattr(parent, attr)
        if isinstance(module, LoRALinear):  # 幂等
            continue
        setattr(parent, attr, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = InjectionReport(candidates, trainable, total)
    if log is not None:
        log.info(
            "LoRA 注入完成：%d 个线性层，r=%d, alpha=%d, 可训练 %.2fM / 总 %.2fM（%.2f%%）",
            len(candidates),
            rank,
            alpha,
            trainable / 1e6,
            total / 1e6,
            100.0 * trainable / max(1, total),
        )
        log.info("注入位置：%s", ", ".join(candidates[:8]) + (" ..." if len(candidates) > 8 else ""))
    return report


def lora_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """只包含 LoRA 参数的 state_dict（key 形如 ``blocks.0.att.key.lora_A``）。"""
    result: dict[str, torch.Tensor] = {}
    for name, module in lora_modules(model):
        result[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        result[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return result


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor], strict: bool = True) -> None:
    current = dict(lora_modules(model))
    expected = set()
    for name in current:
        expected.add(f"{name}.lora_A")
        expected.add(f"{name}.lora_B")
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - expected)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "LoRA 权重与当前模型不匹配：\n"
            f"  缺少 {len(missing)} 个（例如 {missing[:4]}）\n"
            f"  多余 {len(unexpected)} 个（例如 {unexpected[:4]}）\n"
            "通常是 base 模型或 lora.target_modules 与训练时不一致。"
        )
    with torch.no_grad():
        for key, tensor in state.items():
            if not key.endswith((".lora_A", ".lora_B")):
                continue
            module_name, _, param = key.rpartition(".")
            module = current.get(module_name)
            if module is None:
                continue
            target = getattr(module, param)
            target.copy_(tensor.to(target.device, target.dtype))


@torch.no_grad()
def merge_and_unload(model: nn.Module) -> nn.Module:
    """把 LoRA 权重合并进基座，并替换成普通 ``nn.Linear``。

    会 **原地修改** 传入的模型；需要保留适配器时请先 ``copy.deepcopy``。
    """
    targets = [name for name, _ in lora_modules(model)]
    for name in targets:
        parent, attr = _parent_and_attr(model, name)
        module = getattr(parent, attr)
        assert isinstance(module, LoRALinear)
        setattr(parent, attr, module.merged())
    return model


@torch.no_grad()
def merged_state_dict(model: nn.Module, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """返回一个“已合并 LoRA”的普通 state_dict（key 与基座模型完全一致）。

    不修改模型本身，也不深拷整个模型，适合 7B 级别的导出。
    """
    result: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        result[name] = tensor.detach().to("cpu", dtype=dtype or tensor.dtype)

    for name, module in lora_modules(model):
        result[f"{name}.weight"] = module.merged_weight().detach().to(
            "cpu", dtype=dtype or module.base.weight.dtype
        )
        if module.base.bias is not None:
            result[f"{name}.bias"] = module.base.bias.detach().to(
                "cpu", dtype=dtype or module.base.bias.dtype
            )
        for stale in (f"{name}.base.weight", f"{name}.base.bias", f"{name}.lora_A", f"{name}.lora_B"):
            result.pop(stale, None)
    return result


def lora_parameter_summary(model: nn.Module) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return (
        f"LoRA 可训练参数 {trainable / 1e6:.2f}M / 总参数 {total / 1e6:.2f}M "
        f"({100.0 * trainable / max(1, total):.2f}%)"
    )
