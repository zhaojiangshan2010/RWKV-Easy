"""纯 PyTorch 的 WKV-7 分块实现（默认后端）。

算法
----

RWKV-7 的递推是“对角衰减 + 秩 1 修正”的线性系统::

    S_t = diag(d_t) S_{t-1} + b_t (a_t · S_{t-1}) + k_t v_t^T ,  d_t = exp(w_t)
    o_t = r_t · S_t

对长度为 L 的块做展开，令 ``D_t = prod_{i<=t} d_i``：

    S_t = D_t ⊙ S_{-1} + Σ_{j<=t} (D_t / D_j) ⊙ (k_j v_j^T + b_j u_j^T)
    u_j = a_j · S_{j-1}

把 ``u_j`` 收集成矩阵 U，可得到一个严格下三角系统::

    (I - A) U = RHS
    A[j, i] = Σ_c a[j,c] b[i,c] (D_{j-1}/D_i)[c]        (i < j)
    RHS[j]  = Σ_c a[j,c] D_{j-1}[c] S_{-1}[c,:] + Σ_{i<j} ... k_i

一次三角求解即可并行得到块内所有 token，块间只传 ``S``。这与 FLA / Triton 的
chunked delta-rule 同构，但完全用 PyTorch 写成，因此 CPU / CUDA / ROCm 都能跑，
不需要编译任何内核。

数值细节
--------

* 全部中间量用 **float32** 计算（三角求解也只支持 fp32/fp64）。
* 衰减因子的指数范围可能很大：``exp(-logD)`` 在 log-decay 很负时会溢出。
  这里先用一次 reduce 求出本 batch 的最强单 token 衰减，据此把块长压到
  安全范围内（``chunk * max_rate <= max_decay_span``），再顺序分块。
  分块在数学上是透明的（状态照常传递），结果完全不变。
* ``log_decay`` 下限截断到 -80，避免极端权重下 ``exp`` 溢出（此时衰减已经是
  ``e^-80 ≈ 0``，对 fp32 而言与 0 无异）。
"""

from __future__ import annotations

import contextlib

import torch

__all__ = ["NativeBackend"]

# float32 能表示到 3.4e38；留出足够余量，60 对应 1e26。
_DEFAULT_MAX_DECAY_SPAN = 60.0
_LOG_DECAY_FLOOR = -80.0


class NativeBackend:
    name = "native"

    def __init__(
        self,
        chunk: int = 64,
        max_decay_span: float = _DEFAULT_MAX_DECAY_SPAN,
        dynamic_split: bool = True,
    ) -> None:
        # 反归一化因子的指数上限约为 0.6*chunk；96 已经接近 fp32 的安全边界。
        self.chunk = max(1, min(int(chunk), 96))
        self.max_decay_span = float(max_decay_span)
        # clampw 模式的 log-decay 有固定上界（|w| < 0.6066），96*0.6066 < 60，
        # 永远不需要二分。关掉动态切分可以避免每个块一次 `.item()` 的 GPU 同步。
        self.dynamic_split = bool(dynamic_split)
        self._mask_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def set_dynamic_split(self, enabled: bool) -> None:
        self.dynamic_split = bool(enabled)

    # ------------------------------------------------------------------ #
    # 公共入口
    # ------------------------------------------------------------------ #

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
        for name, tensor in (("r", r), ("k", k), ("v", v), ("log_decay", log_decay), ("a", a), ("b", b)):
            if tensor.dim() != 4:
                raise ValueError(f"{name} 需要是 (B,T,H,D) 四维张量，收到 {tuple(tensor.shape)}")
        # 训练时外层可能开了 autocast；三角求解只支持 fp32，这里强制关闭 autocast，
        # 保证全部中间计算都在 fp32 完成（TF32 仍然生效，不影响速度太多）。
        if r.device.type in ("cuda", "cpu"):
            context: contextlib.AbstractContextManager = torch.autocast(
                device_type=r.device.type, enabled=False
            )
        else:  # pragma: no cover - mps 等
            context = contextlib.nullcontext()
        with context:
            return self._wkv7_fp32(r, k, v, log_decay, a, b, state)

    def _wkv7_fp32(
        self,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_decay: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, heads, head_dim = r.shape
        out_dtype = r.dtype

        # 统一升到 fp32；bf16/fp16 的输入在训练里很常见。
        r32 = r.float()
        k32 = k.float()
        v32 = v.float()
        a32 = a.float()
        b32 = b.float()
        w32 = log_decay.float().clamp(min=_LOG_DECAY_FLOOR)
        state32 = state.float() if state is not None else None

        if state32 is None:
            state32 = r32.new_zeros(batch, heads, head_dim, head_dim)

        # softplus / exp 的 log-decay 没有上界，块内累计衰减可能超过 fp32 的
        # 安全范围。这里只做 **一次** GPU 同步，用最负的 log-decay 估算本 batch
        # 允许的最大块长；之后所有块都用这个长度顺序处理。
        # （旧实现是每个块递归二分 + 每层一次 `.item()`，真实权重下会把 64
        #  切成 4/8，并产生上百次同步，实测慢 30 倍以上。）
        chunk = self.chunk
        if self.dynamic_split and length > 1:
            max_rate = float(-w32.min().item())  # 单个 token 的最大衰减（正数）
            if max_rate > 0:
                allowed = max(1, int(self.max_decay_span / max_rate))
                if allowed >= 8:
                    allowed = (allowed // 8) * 8
                chunk = min(chunk, allowed)

        outputs: list[torch.Tensor] = []
        start = 0
        while start < length:
            end = min(start + chunk, length)
            chunk_out, state32 = self._chunk(
                r32[:, start:end],
                k32[:, start:end],
                v32[:, start:end],
                a32[:, start:end],
                b32[:, start:end],
                w32[:, start:end],
                state32,
            )
            outputs.append(chunk_out)
            start = end

        out = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
        return out.to(out_dtype), state32.to(out_dtype)

    # ------------------------------------------------------------------ #
    # 单块
    # ------------------------------------------------------------------ #

    def _masks(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """缓存块内下三角掩码（full 与 strict），避免每个 chunk 重新分配。"""
        key = (length, str(device))
        masks = self._mask_cache.get(key)
        if masks is None:
            tril = torch.ones(length, length, dtype=torch.bool, device=device).tril()
            strict = tril.tril(diagonal=-1)
            if len(self._mask_cache) > 8:
                self._mask_cache.clear()
            self._mask_cache[key] = (tril, strict)
            masks = (tril, strict)
        return masks

    def _chunk(
        self,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, heads, head_dim = r.shape
        device = r.device
        bh = batch * heads

        # (B,T,H,D) -> (B,H,T,D) -> (BH,T,D)
        def to_bh(x: torch.Tensor) -> torch.Tensor:
            return x.permute(0, 2, 1, 3).reshape(bh, length, head_dim)

        r_bh = to_bh(r)
        k_bh = to_bh(k)
        v_bh = to_bh(v)
        a_bh = to_bh(a)
        b_bh = to_bh(b)
        w_bh = to_bh(w)

        log_d = torch.cumsum(w_bh, dim=1)  # D_t，包含第 t 个 token
        log_d_prev = log_d - w_bh  # D_{t-1}
        d = torch.exp(log_d)
        inv_d = torch.exp(-log_d)
        d_prev = torch.exp(log_d_prev)

        wq = r_bh * d
        kwi = k_bh * inv_d
        bwi = b_bh * inv_d
        wa = a_bh * d_prev

        # 块内 token 两两内积
        qk = wq @ kwi.transpose(1, 2)  # (BH,L,L)
        qb = wq @ bwi.transpose(1, 2)
        ak = wa @ kwi.transpose(1, 2)
        aa = wa @ bwi.transpose(1, 2)

        tril, strict = self._masks(length, device)
        qk = qk * tril
        qb = qb * tril
        ak = ak * strict
        aa = aa * strict

        state_bh = state.reshape(bh, head_dim, head_dim)
        rhs = wa @ state_bh + ak @ v_bh  # (BH,L,V)

        # (I - A) U = RHS。A 严格下三角、对角线为 0，因此 -A 是严格下三角，
        # 用 unitriangular=True 让求解器把对角线当成 1，省掉 eye 与对角减法。
        u = torch.linalg.solve_triangular(-aa, rhs, upper=False, unitriangular=True)

        out = wq @ state_bh + qk @ v_bh + qb @ u  # (BH,L,V)

        # 新状态：S_L-1 = D_last ⊙ S_0 + Σ_j (D_last/D_j) ⊙ (k_j v_j^T + b_j u_j^T)
        log_d_last = log_d[:, -1:, :]  # (BH,1,D)
        dec_last = torch.exp(log_d_last - log_d)  # <= 1
        kd = k_bh * dec_last
        bd = b_bh * dec_last
        new_state = (
            torch.exp(log_d_last).transpose(1, 2) * state_bh
            + kd.transpose(1, 2) @ v_bh
            + bd.transpose(1, 2) @ u
        )  # (BH,D,V)

        out = out.reshape(batch, heads, length, head_dim).permute(0, 2, 1, 3)
        new_state = new_state.reshape(batch, heads, head_dim, head_dim)
        return out, new_state
