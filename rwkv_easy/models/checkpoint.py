"""RWKV-7 checkpoint 读取、维度推断与模型构建。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..backends import create_backend
from .rwkv7 import RWKV7, RWKV7Config

__all__ = [
    "build_model",
    "checkpoint_signature",
    "infer_config",
    "load_model_from_checkpoint",
    "random_model",
    "read_checkpoint",
]

# torch.compile / DDP / FSDP / HF PEFT 包装常见的 key 前缀
_STRIP_PREFIXES = (
    "_forward_module.",
    "_orig_mod.",
    "_fsdp_wrapped_module.",
    "module.",
    "base_model.model.",
    "model.",
)

_STATE_DICT_KEYS = ("state_dict", "model", "module", "model_state_dict", "weights", "params")


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #


def _extract_state_dict(obj: Any, path: Path) -> dict[str, torch.Tensor]:
    if isinstance(obj, nn.Module):
        return dict(obj.state_dict())

    if isinstance(obj, dict):
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return dict(obj)
        for key in _STATE_DICT_KEYS:
            inner = obj.get(key)
            if isinstance(inner, dict) and inner and all(
                isinstance(v, torch.Tensor) for v in inner.values()
            ):
                return dict(inner)
        tensors = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        if tensors:
            return tensors
        raise TypeError(
            f"{path} 里没有找到 state_dict。顶层字段：{list(obj.keys())[:12]}"
        )

    raise TypeError(f"无法识别的 checkpoint 类型：{type(obj).__name__}（{path}）")


def _strip_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    changed = True
    while changed:
        changed = False
        for prefix in _STRIP_PREFIXES:
            if state_dict and all(key.startswith(prefix) for key in state_dict):
                state_dict = {key[len(prefix) :]: value for key, value in state_dict.items()}
                changed = True
    return state_dict


def read_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    """读取 ``.pth`` / ``.safetensors``，并清洗常见的 key 前缀。

    - 支持 ``{"state_dict": {...}}``、``{"model": {...}}`` 等包装；
    - 支持被 DDP / torch.compile / HF PEFT 加过前缀的权重；
    - 对 ``.pth`` 优先使用 ``mmap``，避免把整个模型先读进内存。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在：{path}")
    suffix = path.suffix.lower()

    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "读取 .safetensors 需要 safetensors：pip install safetensors"
            ) from exc
        state_dict = dict(load_file(str(path)))
    else:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:  # pragma: no cover - 老版本 torch 没有 mmap
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except RuntimeError:  # pragma: no cover - 非 zip 格式
            obj = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = _extract_state_dict(obj, path)

    state_dict = _strip_prefixes(state_dict)
    if not state_dict:
        raise ValueError(f"checkpoint 为空：{path}")
    return state_dict


# --------------------------------------------------------------------------- #
# 维度推断
# --------------------------------------------------------------------------- #


def infer_config(
    state_dict: dict[str, torch.Tensor],
    *,
    n_layer: int | None = None,
    n_embd: int | None = None,
    vocab_size: int | None = None,
    head_size: int | None = None,
    head_size_divisor: int = 8,
) -> RWKV7Config:
    """从 state_dict 推断模型维度；显式参数优先，并做一致性检查。"""
    emb = state_dict.get("emb.weight")
    if emb is None:
        raise KeyError(
            "checkpoint 缺少 `emb.weight`，不是 RWKV-7 权重。\n"
            f"实际前几个 key：{list(state_dict)[:8]}"
        )
    inferred_vocab, inferred_embd = int(emb.shape[0]), int(emb.shape[1])

    pattern = re.compile(r"^blocks\.(\d+)\.")
    indices = {int(match.group(1)) for key in state_dict if (match := pattern.match(key))}
    if not indices:
        raise KeyError("checkpoint 里没有 `blocks.N.*` 参数，不是 RWKV 模型。")
    inferred_layers = max(indices) + 1

    r_k = state_dict.get("blocks.0.att.r_k")
    inferred_head = int(r_k.shape[1]) if r_k is not None and r_k.dim() == 2 else None
    if (
        r_k is not None
        and r_k.dim() == 2
        and int(r_k.shape[0]) * int(r_k.shape[1]) != inferred_embd
    ):
        raise ValueError(
            "checkpoint 的 `blocks.0.att.r_k` 形状与 n_embd 不一致："
            f"{tuple(r_k.shape)} vs n_embd={inferred_embd}。不是 x070 权重？"
        )

    final_vocab = int(vocab_size) if vocab_size else inferred_vocab
    final_embd = int(n_embd) if n_embd else inferred_embd
    final_layers = int(n_layer) if n_layer else inferred_layers
    final_head = int(head_size) if head_size else (inferred_head or 64)

    if final_vocab != inferred_vocab:
        raise ValueError(
            f"配置的 vocab_size={final_vocab} 与 checkpoint 的 {inferred_vocab} 不一致"
        )
    if final_embd != inferred_embd:
        raise ValueError(f"配置的 n_embd={final_embd} 与 checkpoint 的 {inferred_embd} 不一致")
    if final_layers != inferred_layers:
        raise ValueError(
            f"配置的 n_layer={final_layers} 与 checkpoint 的 {inferred_layers} 不一致"
        )
    if inferred_head is not None and final_head != inferred_head:
        raise ValueError(
            f"配置的 head_size={final_head} 与 checkpoint 的 {inferred_head} 不一致"
        )

    ffn_key = state_dict.get("blocks.0.ffn.key.weight")
    if ffn_key is not None and int(ffn_key.shape[0]) != final_embd * 4:
        raise ValueError(
            "checkpoint 的 FFN 维度不是 4*n_embd，可能不是 x070 模型："
            f"ffn.key.weight={tuple(ffn_key.shape)}, n_embd={final_embd}"
        )

    return RWKV7Config(
        vocab_size=final_vocab,
        n_layer=final_layers,
        n_embd=final_embd,
        head_size=final_head,
        head_size_divisor=int(head_size_divisor),
    )


def checkpoint_signature(state_dict: dict[str, torch.Tensor]) -> str:
    """只对 key 名字做哈希：用于校验 adapter 与基座是否匹配（与数值无关）。"""
    joined = "|".join(sorted(state_dict.keys())).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #


def build_model(
    cfg: RWKV7Config,
    *,
    backend: str = "native",
    decay: str = "clampw",
    wkv_chunk: int = 64,
    grad_checkpoint: bool = False,
) -> RWKV7:
    if decay == "auto":
        # auto 由训练器在拿到 tokenizer 后校准；这里先给一个安全默认值。
        decay = "clampw"
    wkv_backend = create_backend(backend, chunk=wkv_chunk)
    return RWKV7(cfg, wkv_backend, decay_mode=decay, grad_checkpoint=grad_checkpoint)


def _materialize_meta(model: nn.Module) -> None:
    """把仍然停留在 meta 设备上的参数/缓冲区替换成零张量。"""
    for name, param in list(model.named_parameters()):
        if param.device.type == "meta":
            module = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            attr = name.rsplit(".", 1)[-1]
            replacement = nn.Parameter(
                torch.zeros(param.shape, dtype=torch.float32), requires_grad=param.requires_grad
            )
            module.register_parameter(attr, replacement)
    for name, buffer in list(model.named_buffers()):
        if buffer.device.type == "meta":
            module = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            attr = name.rsplit(".", 1)[-1]
            module.register_buffer(attr, torch.zeros(buffer.shape, dtype=torch.float32))


def load_model_from_checkpoint(
    path: str | Path,
    *,
    backend: str = "native",
    decay: str = "clampw",
    wkv_chunk: int = 64,
    grad_checkpoint: bool = False,
    dtype: torch.dtype | None = None,
    n_layer: int | None = None,
    n_embd: int | None = None,
    vocab_size: int | None = None,
    head_size: int | None = None,
    head_size_divisor: int = 8,
    allow_missing_time_state: bool = True,
    log: Any = None,
) -> tuple[RWKV7, RWKV7Config, str]:
    """加载 RWKV-7 权重。

    返回 ``(model, config, signature)``。模型先在 meta 设备上构建，再用
    ``assign=True`` 直接挂载 checkpoint 张量，CPU 峰值内存约等于一份模型。
    """
    state_dict = read_checkpoint(path)
    model_cfg = infer_config(
        state_dict,
        n_layer=n_layer,
        n_embd=n_embd,
        vocab_size=vocab_size,
        head_size=head_size,
        head_size_divisor=head_size_divisor,
    )
    signature = checkpoint_signature(state_dict)
    # 如果 checkpoint 里带 time_state，说明是 State Tuning 合并后的权重：
    # 构建模型时先把 State Tuning 的参数位建出来，否则加载会把它们当成“多余 key”丢掉。
    has_time_state = any(key.endswith("time_state") for key in state_dict)

    with torch.device("meta"):
        model = build_model(
            model_cfg,
            backend=backend,
            decay=decay,
            wkv_chunk=wkv_chunk,
            grad_checkpoint=grad_checkpoint,
        )
        if has_time_state:
            model.attach_state_tuning(init_std=0.0)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)

    allowed_missing = []
    fatal_missing = []
    for key in missing:
        if allow_missing_time_state and "time_state" in key:
            allowed_missing.append(key)
        else:
            fatal_missing.append(key)

    if fatal_missing:
        preview = "\n  ".join(fatal_missing[:12])
        raise RuntimeError(
            f"checkpoint 缺少 {len(fatal_missing)} 个必要参数，例如：\n  {preview}\n"
            "请确认这是 x070 (RWKV-7) 权重，且 n_layer/n_embd/head_size 正确。"
        )
    if unexpected and log is not None:
        preview = ", ".join(unexpected[:8])
        log.warning("checkpoint 里有 %d 个未使用参数（已忽略）：%s", len(unexpected), preview)

    _materialize_meta(model)

    if dtype is not None and dtype != torch.float32:
        model = model.to(dtype=dtype)

    if log is not None:
        log.info(
            "模型：%d 层 / %d 维 / %d 头 / head_size=%d / vocab=%d（%.2fM 参数）",
            model_cfg.n_layer,
            model_cfg.n_embd,
            model_cfg.n_head,
            model_cfg.head_size,
            model_cfg.vocab_size,
            model.num_parameters() / 1e6,
        )
    return model, model_cfg, signature


def random_model(
    cfg: RWKV7Config,
    *,
    backend: str = "native",
    decay: str = "clampw",
    wkv_chunk: int = 64,
    grad_checkpoint: bool = False,
    seed: int = 0,
) -> RWKV7:
    """构建随机初始化模型（自检、单元测试、小规模演示用）。"""
    torch.manual_seed(seed)
    model = build_model(
        cfg,
        backend=backend,
        decay=decay,
        wkv_chunk=wkv_chunk,
        grad_checkpoint=grad_checkpoint,
    )
    # 官方对 embedding / head 没有特殊初始化，默认分布即可；这里只把 GroupNorm
    # 与 LayerNorm 显式重置，避免随机种子的影响被掩盖。
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    return model
