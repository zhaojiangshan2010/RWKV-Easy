"""逐 token 的参考实现（慢，但一眼能看懂，用于测试与数值对照）。

这个后端刻意不做任何分块优化：它就是把官方 CUDA 内核的标量循环逐行翻译成
PyTorch。任何“聪明”的实现都可以用它来验证。
"""

from __future__ import annotations

import torch

__all__ = ["ReferenceBackend"]


class ReferenceBackend:
    name = "reference"

    def __init__(self, double: bool = False) -> None:
        # double=True 时用 float64 计算，作为黄金参考。
        self.double = bool(double)

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
        out_dtype = r.dtype
        if self.double:
            r, k, v, log_decay, a, b = (x.double() for x in (r, k, v, log_decay, a, b))
            if state is not None:
                state = state.double()

        batch, length, heads, head_dim = r.shape
        if state is None:
            state = r.new_zeros(batch, heads, head_dim, head_dim)

        outputs: list[torch.Tensor] = []
        for t in range(length):
            d = torch.exp(log_decay[:, t])  # (B,H,K)
            at = a[:, t]  # (B,H,K)
            bt = b[:, t]  # (B,H,K)
            kt = k[:, t]  # (B,H,K)
            vt = v[:, t]  # (B,H,V)
            rt = r[:, t]  # (B,H,K)

            # sa = a_t · S_{t-1}
            sa = (at.unsqueeze(-1) * state).sum(dim=2)  # (B,H,V)
            state = (
                state * d.unsqueeze(-1)
                + bt.unsqueeze(-1) * sa.unsqueeze(-2)
                + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            )
            outputs.append((rt.unsqueeze(-1) * state).sum(dim=2))

        out = torch.stack(outputs, dim=1)
        return out.to(out_dtype), state.to(out_dtype)
