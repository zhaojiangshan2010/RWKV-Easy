"""可选的 rwkv-fla / Triton 后端。

安装：``pip install rwkv-fla``（需要 CUDA 或 ROCm + Triton）。

注意
----
``chunk_rwkv7`` 要求 ``log_decay <= -0.5``，因此这个后端只与
``model.decay: softplus`` 兼容。配置成其它衰减模式时，``Config.resolve()`` 会
在启动阶段直接报错，而不是在训练中途给出错误的数值。

这个后端把计算留在输入 dtype（bf16 很常见），以便真正吃到 Triton 内核的加速。
"""

from __future__ import annotations

import torch

__all__ = ["FLABackend"]


class FLABackend:
    name = "fla"

    def __init__(self, chunk: int = 64) -> None:
        self.chunk = int(chunk)
        self._impl = None

    def _load(self):
        if self._impl is not None:
            return self._impl
        try:
            from rwkvfla.ops.rwkv7 import chunk_rwkv7
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise ImportError(
                "使用 `model.backend: fla` 需要安装 rwkv-fla：\n"
                "  pip install rwkv-fla\n"
                "如果暂时不想装，把配置改成 `model.backend: native` 即可（纯 PyTorch，"
                "CPU/CUDA/ROCm 都能跑）。"
            ) from exc
        self._impl = chunk_rwkv7
        return self._impl

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
        chunk_rwkv7 = self._load()

        if torch.cuda.is_available() and not r.is_cuda:
            raise RuntimeError("fla 后端需要 CUDA/ROCm 张量，当前输入在 CPU 上。")
        if float(log_decay.max()) > 0:
            raise ValueError("fla 后端要求 log_decay <= 0（衰减 <= 1）")

        batch, length, heads, head_dim = r.shape
        # FLA 的分块核要求长度是 64 的倍数；不足则用“零衰减 + 零 k/v”补齐，
        # 这样补出来的位置不会改变状态，也不会影响前面的输出。
        pad = (-length) % 64
        if pad:
            zeros = torch.zeros(batch, pad, heads, head_dim, dtype=r.dtype, device=r.device)
            r_p = torch.cat([r, zeros], dim=1)
            k_p = torch.cat([k, zeros], dim=1)
            v_p = torch.cat([v, zeros], dim=1)
            a_p = torch.cat([a, zeros], dim=1)
            b_p = torch.cat([b, zeros], dim=1)
            w_p = torch.cat([log_decay, torch.full_like(zeros, -60.0)], dim=1)
        else:
            r_p, k_p, v_p, a_p, b_p, w_p = r, k, v, a, b, log_decay

        out, new_state = chunk_rwkv7(
            r=r_p,
            w=w_p,
            k=k_p,
            v=v_p,
            a=a_p,
            b=b_p,
            scale=1.0,
            initial_state=state,
            output_final_state=True,
            head_first=False,
        )
        if pad:
            out = out[:, :length]
        if new_state is None:  # pragma: no cover - 防御式
            batch_, heads_, dim_ = batch, heads, head_dim
            new_state = r.new_zeros(batch_, heads_, dim_, dim_)
        return out, new_state
