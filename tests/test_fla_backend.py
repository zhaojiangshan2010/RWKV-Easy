"""rwkv-fla 后端对照测试（需要 CUDA + rwkv-fla）。

没有 GPU 或没装 rwkv-fla 时整组自动跳过；设置 ``RWKV_EASY_SKIP_FLA=1``
可以强制跳过。Triton 首次编译较慢，这个文件在 GPU 上通常要跑几十秒。
"""

from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn.functional as F

from rwkv_easy.backends import available_backends, create_backend
from rwkv_easy.backends.reference import ReferenceBackend

pytestmark = pytest.mark.skipif(
    os.environ.get("RWKV_EASY_SKIP_FLA") == "1"
    or not torch.cuda.is_available()
    or "fla" not in available_backends(),
    reason="需要 CUDA 与 rwkv-fla",
)


def _inputs(
    length: int,
    dtype: torch.dtype,
    batch: int = 2,
    heads: int = 4,
    dim: int = 64,
    decay: str = "softplus",
):
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(0)

    def randn() -> torch.Tensor:
        return torch.randn(batch, length, heads, dim, device=device, dtype=dtype, generator=generator)

    r, k, v, raw = randn(), randn(), randn(), randn()
    if decay == "clampw":
        log_decay = -math.exp(-0.5) * torch.sigmoid(raw.float())
    elif decay == "softplus":
        log_decay = -F.softplus(-raw.float()) - 0.5
    else:
        raise ValueError(decay)
    log_decay = log_decay.to(dtype)
    kk = F.normalize(k.float(), dim=-1, p=2.0).to(dtype)
    a = -kk
    b = kk * torch.sigmoid(randn())
    return r, k, v, log_decay, a, b


def _relative(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    return ((x - y).norm() / y.norm().clamp_min(1e-12)).item()


def test_fla_matches_fp64_reference():
    inputs = _inputs(128, torch.float32)
    out_ref, state_ref = ReferenceBackend(double=True).wkv7(*inputs)
    out, state = create_backend("fla").wkv7(*inputs)
    assert _relative(out, out_ref) < 5e-3
    assert _relative(state, state_ref) < 5e-3


def test_fla_accepts_clampw_decay():
    """官方权重用的是 clampw；fla 不应该要求 softplus。"""
    inputs = _inputs(128, torch.float32, decay="clampw")
    out_ref, state_ref = ReferenceBackend(double=True).wkv7(*inputs)
    out, state = create_backend("fla").wkv7(*inputs)
    assert _relative(out, out_ref) < 5e-3
    assert _relative(state, state_ref) < 5e-3


def test_fla_padding_is_neutral():
    """T 不是 64 的倍数时补齐位必须不改变状态（log_decay=0，而不是强衰减）。"""
    inputs = _inputs(100, torch.float32)
    out_nat, state_nat = create_backend("native", chunk=64).wkv7(*inputs)
    out_fla, state_fla = create_backend("fla").wkv7(*inputs)
    assert _relative(out_fla, out_nat) < 5e-3
    assert _relative(state_fla, state_nat) < 5e-3


def test_fla_chunked_state_matches_native():
    r, k, v, log_decay, a, b = _inputs(100, torch.float32)
    native = create_backend("native", chunk=64)
    fla = create_backend("fla")

    out_n1, state_n = native.wkv7(r[:, :64], k[:, :64], v[:, :64], log_decay[:, :64], a[:, :64], b[:, :64])
    out_n2, state_n2 = native.wkv7(
        r[:, 64:], k[:, 64:], v[:, 64:], log_decay[:, 64:], a[:, 64:], b[:, 64:], state_n
    )
    out_f1, state_f = fla.wkv7(r[:, :64], k[:, :64], v[:, :64], log_decay[:, :64], a[:, :64], b[:, :64])
    out_f2, state_f2 = fla.wkv7(
        r[:, 64:], k[:, 64:], v[:, 64:], log_decay[:, 64:], a[:, 64:], b[:, 64:], state_f
    )

    assert _relative(out_f1, out_n1) < 5e-3
    assert _relative(out_f2, out_n2) < 5e-3
    assert _relative(state_f2, state_n2) < 5e-3


def test_fla_bf16_backward():
    inputs = [x.clone().requires_grad_(True) for x in _inputs(128, torch.bfloat16)]
    out, state = create_backend("fla").wkv7(*inputs)
    assert out.dtype == torch.bfloat16
    (out.float().square().sum() + state.float().square().sum()).backward()
    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
