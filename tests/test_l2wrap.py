"""L2Wrap 的两种实现应产生完全相同的梯度。"""

from __future__ import annotations

import torch

from rwkv_easy.train.loop import L2Wrap


def _grads(legacy: bool) -> torch.Tensor:
    torch.manual_seed(0)
    logits = torch.randn(4, 7, requires_grad=True)
    factor = 0.01
    base_loss = (logits * torch.arange(1, 8, dtype=torch.float32)).sum()
    if legacy:
        loss = L2Wrap.apply(base_loss, logits, factor)
    else:
        loss = base_loss

        def hook(grad: torch.Tensor) -> torch.Tensor:
            max_value, indices = torch.max(logits.detach(), dim=-1, keepdim=True)
            grad.scatter_add_(-1, indices, max_value * factor)
            return grad

        logits.register_hook(hook)
    loss.backward()
    assert logits.grad is not None
    return logits.grad.clone()


def test_l2wrap_hook_matches_legacy():
    torch.testing.assert_close(_grads(legacy=True), _grads(legacy=False))
