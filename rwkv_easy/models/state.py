"""循环状态容器（infctx / State Tuning 共用）。

状态用 3 个稠密张量表示，而不是每层一个 Python 对象：

* ``shift_att``: ``(n_layer, B, C)`` —— 每层 time-mix 的时间移位状态
* ``shift_ffn``: ``(n_layer, B, C)`` —— 每层 channel-mix 的时间移位状态
* ``wkv``:       ``(n_layer, B, H, K, V)`` —— 每层 WKV 的循环状态

这样 ``detach()`` / ``to()`` / 传给 ``torch.utils.checkpoint`` 都只是一个张量操作。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["BlockState", "ModelState"]


@dataclass
class BlockState:
    att_shift: torch.Tensor  # (B, C)
    ffn_shift: torch.Tensor  # (B, C)
    wkv: torch.Tensor | None  # (B, H, K, V)


@dataclass
class ModelState:
    shift_att: torch.Tensor  # (L, B, C)
    shift_ffn: torch.Tensor  # (L, B, C)
    wkv: torch.Tensor  # (L, B, H, K, V)

    # ------------------------------------------------------------------ #
    # 构造
    # ------------------------------------------------------------------ #

    @property
    def n_layer(self) -> int:
        return int(self.shift_att.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.shift_att.shape[1])

    @classmethod
    def zeros(
        cls,
        n_layer: int,
        batch_size: int,
        n_embd: int,
        n_head: int,
        head_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ModelState:
        return cls(
            shift_att=torch.zeros(n_layer, batch_size, n_embd, device=device, dtype=dtype),
            shift_ffn=torch.zeros(n_layer, batch_size, n_embd, device=device, dtype=dtype),
            wkv=torch.zeros(
                n_layer, batch_size, n_head, head_size, head_size, device=device, dtype=dtype
            ),
        )

    @classmethod
    def from_layers(
        cls,
        att_shifts: list[torch.Tensor],
        ffn_shifts: list[torch.Tensor],
        wkv_states: list[torch.Tensor],
    ) -> ModelState:
        return cls(
            shift_att=torch.stack(att_shifts, dim=0),
            shift_ffn=torch.stack(ffn_shifts, dim=0),
            wkv=torch.stack(wkv_states, dim=0),
        )

    # ------------------------------------------------------------------ #
    # 切片与变换
    # ------------------------------------------------------------------ #

    def layer(self, index: int) -> BlockState:
        return BlockState(
            att_shift=self.shift_att[index],
            ffn_shift=self.shift_ffn[index],
            wkv=self.wkv[index],
        )

    def detach(self) -> ModelState:
        return ModelState(
            shift_att=self.shift_att.detach(),
            shift_ffn=self.shift_ffn.detach(),
            wkv=self.wkv.detach(),
        )

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> ModelState:
        return ModelState(
            shift_att=self.shift_att.to(device=device, dtype=dtype),
            shift_ffn=self.shift_ffn.to(device=device, dtype=dtype),
            wkv=self.wkv.to(device=device, dtype=dtype),
        )

    def select_batch(self, indices: torch.Tensor) -> ModelState:
        return ModelState(
            shift_att=self.shift_att.index_select(1, indices),
            shift_ffn=self.shift_ffn.index_select(1, indices),
            wkv=self.wkv.index_select(1, indices),
        )

    def nbytes(self) -> int:
        return self.shift_att.numel() * self.shift_att.element_size() + self.shift_ffn.numel() * self.shift_ffn.element_size() + self.wkv.numel() * self.wkv.element_size()
