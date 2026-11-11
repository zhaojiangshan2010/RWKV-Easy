"""WKV 后端数值测试：native 分块实现 vs 逐 token fp64 参考。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rwkv_easy.backends.native import NativeBackend
from rwkv_easy.backends.reference import ReferenceBackend


def _inputs(batch=2, length=23, heads=3, dim=8, seed=0, decay_scale=0.6, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=dtype)
    r = rand(batch, length, heads, dim)
    k = rand(batch, length, heads, dim)
    v = rand(batch, length, heads, dim)
    a = -F.normalize(rand(batch, length, heads, dim), dim=-1)
    b = F.normalize(rand(batch, length, heads, dim), dim=-1) * 0.7
    w = -torch.rand(batch, length, heads, dim, generator=generator, dtype=dtype) * decay_scale
    return r, k, v, w, a, b


def test_forward_matches_reference():
    tensors = _inputs()
    with torch.no_grad():
        out_ref, state_ref = ReferenceBackend(double=True).wkv7(*tensors)
        out_nat, state_nat = NativeBackend(chunk=8).wkv7(*tensors)
    assert out_ref.shape == out_nat.shape
    assert state_ref.shape == state_nat.shape
    assert (out_ref.float() - out_nat).abs().max() < 1e-5
    assert (state_ref.float() - state_nat).abs().max() < 1e-5


def test_initial_state_matches():
    tensors = _inputs(length=11)
    state = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(3)) * 0.1
    with torch.no_grad():
        out_ref, state_ref = ReferenceBackend(double=True).wkv7(*tensors, state)
        out_nat, state_nat = NativeBackend(chunk=4).wkv7(*tensors, state)
    assert (out_ref.float() - out_nat).abs().max() < 1e-5
    assert (state_ref.float() - state_nat).abs().max() < 1e-5


def test_strong_decay_stays_finite_and_accurate():
    # log_decay ~ -4 会让块内累计衰减很大，必须自动缩小块长而不能溢出。
    tensors = _inputs(length=17, decay_scale=4.0)
    with torch.no_grad():
        out_ref, state_ref = ReferenceBackend(double=True).wkv7(*tensors)
        out_nat, state_nat = NativeBackend(chunk=16).wkv7(*tensors)
    assert torch.isfinite(out_nat).all()
    assert torch.isfinite(state_nat).all()
    assert (out_ref.float() - out_nat).abs().max() < 1e-5


def test_chunk_sizes_are_equivalent():
    tensors = _inputs(length=37, decay_scale=0.6)
    reference = None
    for chunk in (1, 3, 16, 64):
        out, _ = NativeBackend(chunk=chunk).wkv7(*tensors)
        if reference is None:
            reference = out
        else:
            # fp32 下不同分块的求和顺序不同，允许 1e-5 级别的舍入差异。
            assert (out - reference).abs().max() < 1e-5, f"chunk={chunk} 结果不一致"


def test_gradients_match_reference():
    base = _inputs(length=13, decay_scale=3.0, dtype=torch.float64)
    tensors_ref = [t.detach().clone().requires_grad_(True) for t in base]
    tensors_nat = [t.detach().clone().requires_grad_(True) for t in base]

    out_ref, state_ref = ReferenceBackend(double=True).wkv7(*tensors_ref)
    out_nat, state_nat = NativeBackend(chunk=6).wkv7(*tensors_nat)
    assert (out_ref - out_nat).abs().max() < 1e-5

    loss_ref = (out_ref * out_ref).mean() + state_ref.pow(2).mean()
    loss_nat = (out_nat * out_nat).mean() + state_nat.pow(2).mean()
    loss_ref.backward()
    loss_nat.backward()

    for name, ref, nat in zip("rkvwab", tensors_ref, tensors_nat, strict=True):
        scale = max(1e-9, float(ref.grad.abs().max()))
        error = float((ref.grad - nat.grad).abs().max())
        assert error / scale < 1e-5, f"{name} grad 误差 {error / scale:.2e}"


def test_bf16_input_output_dtype():
    tensors = _inputs(dtype=torch.float32)
    tensors = [t.to(torch.bfloat16) for t in tensors]
    out, state = NativeBackend(chunk=8).wkv7(*tensors)
    assert out.dtype == torch.bfloat16
    assert state.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
