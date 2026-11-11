"""可选的 rwkv-fla / Triton 后端。

安装：``pip install rwkv-fla``（需要 CUDA 或 ROCm + Triton ≥ 3.2）。

参数口径
--------
``chunk_rwkv7`` 的 ``w`` 是原始 decay 参数（内部会再算 ``-exp(w)``）；
本项目传的是已经变换好的 log-decay，因此走 ``log_w``。clampw / softplus /
exp 三种衰减模式都满足 ``log_decay <= 0``，都可以使用这个后端。

长度不是 64 的倍数时，尾部用 **log_decay = 0 + 全零 k/v/a/b** 补齐：
状态会原样穿过，不改变任何有效位置的结果。

这个后端把计算留在输入 dtype（bf16 很常见），以便真正吃到 Triton 内核的加速；
与 fp64 参考相比，bf16 下相对误差约 3e-3（内核内部按 bf16 累加）。
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

        # 混合精度下各输入的 dtype 可能不一致：例如 `kk` 由 fp32 参数
        # `k_k` 参与逐元素乘会被提升为 fp32，而 value 仍是 bf16。
        # Triton 内核要求 tl.dot 两侧 dtype 相同，这里统一到激活 dtype。
        dtype = r.dtype
        k = k.to(dtype)
        v = v.to(dtype)
        log_decay = log_decay.to(dtype)
        a = a.to(dtype)
        b = b.to(dtype)
        if state is not None:
            state = state.to(dtype)

        batch, length, heads, head_dim = r.shape
        # FLA 的分块核要求长度是 64 的倍数；不足则补齐。
        # 补齐位必须用 **log_decay = 0**（衰减因子 1）+ 全零 k/v/a/b：
        # 这样状态原样穿过，不改变前面任何位置的结果。
        # 不能用 -60 之类的“强衰减”——那会把携带的状态乘成 0。
        pad = (-length) % 64
        if pad:
            zeros = torch.zeros(batch, pad, heads, head_dim, dtype=r.dtype, device=r.device)
            r_p = torch.cat([r, zeros], dim=1)
            k_p = torch.cat([k, zeros], dim=1)
            v_p = torch.cat([v, zeros], dim=1)
            a_p = torch.cat([a, zeros], dim=1)
            b_p = torch.cat([b, zeros], dim=1)
            w_p = torch.cat([log_decay, torch.zeros_like(zeros)], dim=1)
        else:
            r_p, k_p, v_p, a_p, b_p, w_p = r, k, v, a, b, log_decay

        out, new_state = chunk_rwkv7(
            r=r_p,
            # 注意：fla 的 `w` 是原始 decay 参数，内部会再算 `-exp(w)`；
            # 我们这里已经是 log-decay，必须走 `log_w`。
            log_w=w_p,
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
