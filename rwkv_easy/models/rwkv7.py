"""RWKV-7 (x070) 模型实现。

严格对齐上游 RWKV-LM 的 x070 结构（参数名与官方 checkpoint 完全一致）：

* ``emb`` → ``blocks[i]``（``ln0`` 只在第 0 层）→ ``ln_out`` → ``head``
* time-mix: 6 路 time-shift 混合 + 3 组低秩矩阵（decay / a / value-residual / gate）
* value residual：第 0 层的 ``v`` 通过 ``v0 + (xv@v1)@v2`` 门控注入后续所有层
* channel-mix: ``relu(key(x))^2``

与参考实现相比，这里的差异只有工程性的：

* 训练模式不通过环境变量在 import 时决定，而是显式参数；
* 循环状态通过 :class:`ModelState` 显式传入传出，因此 LoRA / State Tuning /
  infctx 可以自由组合，不会再出现“PEFT 包装层吃掉状态参数”的问题；
* 时间移位用显式切片实现（``ZeroPad2d`` 在 3D 输入上的行为不够直观）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from ..backends import WKVBackend
from .state import BlockState, ModelState

__all__ = ["RWKV7", "Block", "ChannelMix", "RWKV7Config", "TimeMix"]


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RWKV7Config:
    """从 checkpoint 推断或手填的模型维度。"""

    vocab_size: int
    n_layer: int
    n_embd: int
    head_size: int = 64
    head_size_divisor: int = 8

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.n_layer <= 0 or self.n_embd <= 0:
            raise ValueError("模型维度必须为正数")
        if self.head_size <= 0 or self.n_embd % self.head_size != 0:
            raise ValueError(
                f"n_embd={self.n_embd} 必须能被 head_size={self.head_size} 整除"
            )

    @property
    def n_head(self) -> int:
        return self.n_embd // self.head_size

    @property
    def dim_att(self) -> int:
        return self.n_embd

    @property
    def dim_ffn(self) -> int:
        return self.n_embd * 4


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


def _ortho_init(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """官方对 w2 / a2 / v2 / g2 的正交初始化。"""
    with torch.no_grad():
        shape = tensor.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1.0
            nn.init.orthogonal_(tensor, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1.0
            for i in range(shape[0]):
                nn.init.orthogonal_(tensor[i], gain=gain * scale)
        else:
            raise ValueError(f"不支持的正交初始化形状：{shape}")
    return tensor


def _lora_dim(channels: int, coeff: float, *, exponent: float = 0.5, minimum: int = 32) -> int:
    """官方低秩维度公式：``max(32, round(coeff * C**exponent / 32) * 32)``。

    decay / aaa / value-residual 用 ``exponent=0.5``；输出门控在官方实现里是
    ``0.6 * C**0.8``（``exponent=0.8``）。两者不能混用，否则 checkpoint 里
    ``g1/g2`` 的形状会对不上。
    """
    return max(minimum, int(round(coeff * (channels ** exponent) / 32.0) * 32))


# --------------------------------------------------------------------------- #
# Time Mixing
# --------------------------------------------------------------------------- #


class TimeMix(nn.Module):
    """RWKV-7 的 time-mix（注意力）层。"""

    def __init__(
        self,
        cfg: RWKV7Config,
        layer_id: int,
        backend: WKVBackend,
        decay_mode: str = "clampw",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.backend = backend
        self.decay_mode = decay_mode
        self.head_size = cfg.head_size
        self.n_head = cfg.n_head
        heads, dim, channels = self.n_head, self.head_size, cfg.n_embd
        layers = cfg.n_layer

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (layers - 1) if layers > 1 else 0.0
            ratio_1_to_almost0 = 1.0 - (layer_id / layers)
            ddd = (torch.arange(channels, dtype=torch.float32) / channels).view(1, 1, channels)

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1)
            )
            self.x_v = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1)
            )
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            # decay 低秩分解：log_w = w0 + tanh(xw @ w1) @ w2
            d_decay = _lora_dim(channels, 1.8)
            d_aaa = _lora_dim(channels, 1.8)
            d_mv = _lora_dim(channels, 1.3)
            d_gate = _lora_dim(channels, 0.6, exponent=0.8)
            if layers == 61:  # 官方对 61 层模型的特例
                d_decay = 192
                d_aaa = 192
                d_mv = 128
                d_gate = 384
            if channels == 1024:
                d_gate = 128

            self.w1 = nn.Parameter(torch.zeros(channels, d_decay))
            self.w2 = nn.Parameter(_ortho_init(torch.zeros(d_decay, channels), 0.1))
            decay_speed = torch.empty(channels, dtype=torch.float32)
            for n in range(channels):
                ratio = n / (channels - 1) if channels > 1 else 0.0
                decay_speed[n] = -7 + 5 * ratio ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1, 1, channels) + 0.5)

            # in-context learning rate 的低秩分解
            self.a1 = nn.Parameter(torch.zeros(channels, d_aaa))
            self.a2 = nn.Parameter(_ortho_init(torch.zeros(d_aaa, channels), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, channels))

            # value residual 门控
            self.v1 = nn.Parameter(torch.zeros(channels, d_mv))
            self.v2 = nn.Parameter(_ortho_init(torch.zeros(d_mv, channels), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, channels) + 1.0)

            # 输出门控
            self.g1 = nn.Parameter(torch.zeros(channels, d_gate))
            self.g2 = nn.Parameter(_ortho_init(torch.zeros(d_gate, channels), 0.1))

            self.k_k = nn.Parameter(torch.ones(1, 1, channels) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, channels))
            self.r_k = nn.Parameter(torch.zeros(heads, dim))

        self.receptance = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)
        self.ln_x = nn.GroupNorm(
            heads,
            channels,
            eps=1e-5 * (cfg.head_size_divisor ** 2),
        )

        # State Tuning 时由 attach_state_tuning() 注入，默认 None。
        self.time_state: nn.Parameter | None = None

    # ------------------------------------------------------------------ #
    # 衰减
    # ------------------------------------------------------------------ #

    def set_decay_mode(self, mode: str) -> None:
        if mode not in ("clampw", "softplus", "exp"):
            raise ValueError(f"未知衰减模式：{mode}")
        self.decay_mode = mode

    def log_decay(self, xw: torch.Tensor) -> torch.Tensor:
        """把 decay 低秩输出映射成 log 衰减（<= 0）。

        三种口径来自不同版本的官方实现，必须与 checkpoint 对应：

        * ``clampw``  ：``-exp(-0.5) * sigmoid(raw)``，参考实现的 CUDA ``W_SCALE``
        * ``softplus``：``-softplus(-raw) - 0.5``，参考实现的 fla 路径
        * ``exp``     ：``raw``（旧版直接取 exp）
        """
        raw = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        if self.decay_mode == "clampw":
            return -math.exp(-0.5) * torch.sigmoid(raw)
        if self.decay_mode == "softplus":
            return -F.softplus(-raw) - 0.5
        if self.decay_mode == "exp":
            return raw.clamp(max=0.0)
        raise ValueError(f"未知衰减模式：{self.decay_mode}")

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #

    def _time_shift(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        state: BlockState | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        if state is not None:
            previous = state.att_shift.unsqueeze(1)  # (B,1,C)
            xx = torch.cat([previous, x[:, :-1]], dim=1) - x
        else:
            shifted = torch.zeros_like(x)
            shifted[:, 1:] = x[:, :-1]
            xx = shifted - x
        return x, xx

    def _addcmul(self, x: torch.Tensor, xx: torch.Tensor):
        return (
            x + xx * self.x_r,
            x + xx * self.x_w,
            x + xx * self.x_k,
            x + xx * self.x_v,
            x + xx * self.x_a,
            x + xx * self.x_g,
        )

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
        state: BlockState | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 ``(output, v_first, new_wkv_state, new_shift_state)``。"""
        batch, length, channels = x.shape
        heads, dim = self.n_head, self.head_size

        x_masked, xx = self._time_shift(x, attention_mask, state)
        xr, xw, xk, xv, xa, xg = self._addcmul(x_masked, xx)

        r = self.receptance(xr)
        log_decay = self.log_decay(xw)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first_out = v
        else:
            assert v_first is not None, "v_first 必须从第 0 层传下来"
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            v_first_out = v_first

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize((k * self.k_k).view(batch, length, heads, dim), dim=-1, p=2.0)
        k = k * (1 + (a - 1) * self.k_a)

        if attention_mask is not None:
            v = v * attention_mask.unsqueeze(-1).to(v.dtype)

        r_h = r.view(batch, length, heads, dim)
        k_h = k.view(batch, length, heads, dim)
        kk_h = kk.view(batch, length, heads, dim)
        v_h = v.view(batch, length, heads, dim)
        w_h = log_decay.view(batch, length, heads, dim)

        wkv_state = state.wkv if state is not None else None
        if wkv_state is None and self.time_state is not None:
            # State Tuning：每个样本第一块使用学习到的初始状态。
            wkv_state = self.time_state.to(v_h.dtype).unsqueeze(0).expand(batch, -1, -1, -1)

        mixed, new_wkv = self.backend.wkv7(
            r_h,
            k_h,
            v_h,
            w_h,
            -kk_h,
            kk_h * a.view(batch, length, heads, dim),
            wkv_state,
        )

        mixed = mixed.reshape(batch, length, channels)
        mixed = self.ln_x(mixed.reshape(batch * length, channels)).view(batch, length, channels)
        bonus = (r_h * k_h * self.r_k).sum(dim=-1, keepdim=True) * v_h
        mixed = mixed + bonus.reshape(batch, length, channels)
        out = self.output(mixed * g)
        # infctx 的 shift 状态：本块最后一个（已被 mask 过的）隐藏态。
        return out, v_first_out, new_wkv, x_masked[:, -1]


# --------------------------------------------------------------------------- #
# Channel Mixing
# --------------------------------------------------------------------------- #


class ChannelMix(nn.Module):
    """RWKV-7 的 channel-mix（FFN）层。"""

    def __init__(self, cfg: RWKV7Config, layer_id: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / cfg.n_layer)
            ddd = (torch.arange(cfg.n_embd, dtype=torch.float32) / cfg.n_embd).view(1, 1, cfg.n_embd)
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4))
        self.key = nn.Linear(cfg.n_embd, cfg.dim_ffn, bias=False)
        self.value = nn.Linear(cfg.dim_ffn, cfg.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: BlockState | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        if state is not None:
            xx = torch.cat([state.ffn_shift.unsqueeze(1), x[:, :-1]], dim=1) - x
        else:
            shifted = torch.zeros_like(x)
            shifted[:, 1:] = x[:, :-1]
            xx = shifted - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k), x[:, -1]


# --------------------------------------------------------------------------- #
# Block
# --------------------------------------------------------------------------- #


class Block(nn.Module):
    def __init__(self, cfg: RWKV7Config, layer_id: int, backend: WKVBackend, decay_mode: str) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(cfg.n_embd)
        self.att = TimeMix(cfg, layer_id, backend, decay_mode)
        self.ffn = ChannelMix(cfg, layer_id)

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor | None,
        state: BlockState | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, BlockState]:
        if self.layer_id == 0:
            x = self.ln0(x)

        att_out, v_first, new_wkv, new_att_shift = self.att(
            self.ln1(x), v_first, state, attention_mask
        )
        x = x + att_out

        ffn_out, new_ffn_shift = self.ffn(self.ln2(x), state, attention_mask)
        x = x + ffn_out

        new_state = BlockState(
            att_shift=new_att_shift,
            ffn_shift=new_ffn_shift,
            wkv=new_wkv,
        )
        return x, v_first, new_state


def _block_without_state(
    block: Block,
    x: torch.Tensor,
    v_first: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """给 ``torch.utils.checkpoint`` 用的纯张量版本（避免传 dataclass）。"""
    out, new_v_first, _ = block(x, v_first, None, attention_mask)
    return out, new_v_first


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class RWKV7(nn.Module):
    """RWKV-7 主体。"""

    def __init__(
        self,
        cfg: RWKV7Config,
        backend: WKVBackend,
        decay_mode: str = "clampw",
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.backend = backend
        self.grad_checkpoint = grad_checkpoint

        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(
            [Block(cfg, i, backend, decay_mode) for i in range(cfg.n_layer)]
        )
        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    # ------------------------------------------------------------------ #
    # 配置
    # ------------------------------------------------------------------ #

    @property
    def decay_mode(self) -> str:
        return self.blocks[0].att.decay_mode

    def set_decay_mode(self, mode: str) -> None:
        for block in self.blocks:
            block.att.set_decay_mode(mode)
        # native 后端在 clampw 下可以关掉动态二分（省掉每块一次 GPU 同步）。
        if hasattr(self.backend, "set_dynamic_split"):
            self.backend.set_dynamic_split(mode != "clampw")

    def attach_state_tuning(self, init_std: float = 0.0) -> list[nn.Parameter]:
        """给每层 attention 注入可训练的 ``time_state``。"""
        heads, dim = self.cfg.n_head, self.cfg.head_size
        params: list[nn.Parameter] = []
        with torch.no_grad():
            for block in self.blocks:
                if block.att.time_state is not None:
                    params.append(block.att.time_state)
                    continue
                value = torch.zeros(heads, dim, dim, dtype=torch.float32)
                if init_std > 0:
                    value.normal_(mean=0.0, std=init_std)
                param = nn.Parameter(value)
                block.att.time_state = param
                params.append(param)
        return params

    def state_parameters(self) -> list[nn.Parameter]:
        return [
            param
            for block in self.blocks
            if block.att.time_state is not None
            for param in [block.att.time_state]
        ]

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> ModelState:
        """构造 infctx 的初始状态。

        若启用了 State Tuning，每层的 ``time_state`` 会成为该层第一块的初始状态
        （broadcast 到 batch），因此梯度能正常回传到它。
        """
        if device is None:
            device = self.emb.weight.device
        heads, dim = self.cfg.n_head, self.cfg.head_size
        n_layer = self.cfg.n_layer
        zeros_att = torch.zeros(n_layer, batch_size, self.cfg.n_embd, device=device, dtype=dtype)
        zeros_ffn = torch.zeros_like(zeros_att)
        wkv_layers: list[torch.Tensor] = []
        for block in self.blocks:
            time_state = block.att.time_state
            if time_state is None:
                wkv_layers.append(
                    torch.zeros(batch_size, heads, dim, dim, device=device, dtype=dtype)
                )
            else:
                wkv_layers.append(
                    time_state.to(device=device, dtype=dtype)
                    .unsqueeze(0)
                    .expand(batch_size, -1, -1, -1)
                )
        return ModelState(
            shift_att=zeros_att,
            shift_ffn=zeros_ffn,
            wkv=torch.stack(wkv_layers, dim=0),
        )

    # ------------------------------------------------------------------ #
    # 前向
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        state: ModelState | None = None,
        need_state: bool = False,
    ) -> tuple[torch.Tensor, ModelState | None]:
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask 需要是 (B,T)")
            attention_mask = attention_mask.to(dtype=torch.float32)

        batch, length = input_ids.shape
        x = self.emb(input_ids)
        v_first: torch.Tensor | None = None

        att_shifts: list[torch.Tensor] = []
        ffn_shifts: list[torch.Tensor] = []
        wkv_states: list[torch.Tensor] = []

        for index, block in enumerate(self.blocks):
            block_state = state.layer(index) if state is not None else None
            use_checkpoint = (
                self.grad_checkpoint
                and self.training
                and not need_state
                and index > 0  # 第 0 层要生成 v_first，跳过更简单
            )
            if use_checkpoint:
                assert v_first is not None
                x, v_first = checkpoint_fn(
                    _block_without_state,
                    block,
                    x,
                    v_first,
                    attention_mask,
                    use_reentrant=False,
                )
                continue

            x, v_first, new_state = block(x, v_first, block_state, attention_mask)
            if need_state:
                att_shifts.append(new_state.att_shift)
                ffn_shifts.append(new_state.ffn_shift)
                wkv_states.append(new_state.wkv)

        x = self.ln_out(x)
        logits = self.head(x)

        new_model_state = None
        if need_state:
            new_model_state = ModelState.from_layers(att_shifts, ffn_shifts, wkv_states)
        return logits, new_model_state

    # ------------------------------------------------------------------ #
    # infctx：把状态拆成张量，便于 torch.utils.checkpoint
    # ------------------------------------------------------------------ #

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        shift_att: torch.Tensor | None = None,
        shift_ffn: torch.Tensor | None = None,
        wkv: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """infctx 单块前向。状态以张量形式传入/传出，方便 ``torch.utils.checkpoint``。"""
        state = None
        if shift_att is not None:
            assert shift_ffn is not None and wkv is not None
            state = ModelState(shift_att=shift_att, shift_ffn=shift_ffn, wkv=wkv)
        logits, new_state = self.forward(
            input_ids, attention_mask=attention_mask, state=state, need_state=True
        )
        assert new_state is not None
        return logits, new_state.shift_att, new_state.shift_ffn, new_state.wkv

    def forward_full(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, _ = self.forward(input_ids, attention_mask=attention_mask, need_state=False)
        return logits

    # ------------------------------------------------------------------ #
    # 杂项
    # ------------------------------------------------------------------ #

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)
        )
