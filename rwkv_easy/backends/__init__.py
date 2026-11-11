"""WKV 后端注册表与统一接口。

RWKV-7 的核心是一个带秩-1 修正的线性递推（per head）::

    S_t = diag(exp(w_t)) S_{t-1} + b_t (a_t · S_{t-1}) + k_t v_t^T
    o_t = r_t · S_t

其中 ``a_t = -kk_t``、``b_t = kk_t * alpha_t``。所有后端都实现同一个契约：

``wkv7(r, k, v, log_decay, a, b, state) -> (out, new_state)``

* ``r, k, v, a, b``: ``(B, T, H, D)``
* ``log_decay``: ``(B, T, H, D)``，且 ``<= 0``（衰减 = ``exp(log_decay)``）
* ``state``: ``(B, H, D, D)`` 或 ``None``（表示从零状态开始）
* 返回 ``out``: ``(B, T, H, D)``，``new_state``: ``(B, H, D, D)``

扩展点
------

新增后端只需要：

1. 写一个类，实现 ``name`` 属性与 ``wkv7(...)`` 方法；
2. 调用 :func:`register_backend` 注册；
3. （可选）在 ``create_backend`` 里处理构造参数。

例如接入自定义 CUDA 内核见 ``docs/EXTENDING.md``。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

__all__ = [
    "BACKEND_NAMES",
    "WKVBackend",
    "available_backends",
    "create_backend",
    "register_backend",
]


@runtime_checkable
class WKVBackend(Protocol):
    """WKV 算子的统一协议。"""

    name: str

    def wkv7(
        self,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_decay: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...


_REGISTRY: dict[str, Any] = {}


def register_backend(name: str, factory: Any) -> None:
    if not name or not isinstance(name, str):
        raise ValueError("后端名字必须是非空字符串")
    _REGISTRY[name.lower()] = factory


def available_backends() -> list[str]:
    names = ["native", "reference"]
    for name in ("fla",):
        try:
            __import__({"fla": "rwkvfla"}[name])
        except Exception:  # pragma: no cover - 取决于环境
            continue
        names.append(name)
    names.extend(name for name in _REGISTRY if name not in names)
    return names


BACKEND_NAMES = ("native", "reference", "fla")


def create_backend(name: str, **kwargs: Any) -> WKVBackend:
    """按名字创建后端实例。未知名字会给出可用列表。

    扩展后端在 import 时调用 :func:`register_backend` 注册即可，见
    ``docs/EXTENDING.md``。
    """
    key = (name or "native").lower()
    if key in _REGISTRY:
        return _REGISTRY[key](**kwargs)
    if key == "native":
        from .native import NativeBackend

        return NativeBackend(**kwargs)
    if key == "reference":
        from .reference import ReferenceBackend

        return ReferenceBackend(**kwargs)
    if key == "fla":
        from .fla import FLABackend

        return FLABackend(**kwargs)
    raise ValueError(
        f"未知的 WKV 后端 {name!r}。可用：{', '.join(available_backends())}。\n"
        "扩展方法见 docs/EXTENDING.md。"
    )
