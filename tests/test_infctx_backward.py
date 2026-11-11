"""infctx 的 per-chunk backward 必须与旧的“最后统一 backward”梯度一致。"""

from __future__ import annotations

import pytest
import torch

from rwkv_easy.config import Config
from rwkv_easy.lora import inject_lora
from rwkv_easy.models.checkpoint import random_model
from rwkv_easy.models.rwkv7 import RWKV7Config
from rwkv_easy.state_tuning import attach_state_tuning
from rwkv_easy.train.loop import Trainer


class _Backend:
    kind = "single"

    @staticmethod
    def backward(loss: torch.Tensor) -> None:
        loss.backward()


def _build(method: str, grad_checkpoint: bool):
    cfg = Config.from_dict(
        {
            "train": {
                "infctx": True,
                "chunk_ctx": 16,
                "grad_accum": 1,
                "grad_checkpoint": grad_checkpoint,
                "l2wrap": 1.0e-4,
                "infctx_detach": True,
                "infctx_backward": False,
            }
        }
    )
    model = random_model(
        RWKV7Config(vocab_size=64, n_layer=2, n_embd=32, head_size=8),
        wkv_chunk=16,
        grad_checkpoint=grad_checkpoint,
        seed=0,
    )
    if method == "lora":
        inject_lora(
            model,
            target_modules=["att.key", "att.value", "ffn.key", "ffn.value"],
            rank=4,
            alpha=8,
        )
    else:
        attach_state_tuning(model, init_std=0.0)
    model.train()

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.model = model
    trainer.backend = _Backend()
    trainer.grad_accum = 1
    trainer.device = torch.device("cpu")
    return trainer


def _inputs():
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 64, (2, 64), generator=generator)
    keep = torch.zeros_like(ids, dtype=torch.bool)
    keep[:, ::3] = True
    labels = torch.where(keep, ids, torch.full_like(ids, -100))
    return ids, labels


def _trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


@pytest.mark.parametrize("grad_checkpoint", [False, True])
@pytest.mark.parametrize("method", ["lora", "state"])
def test_per_chunk_backward_matches_deferred(method: str, grad_checkpoint: bool):
    trainer = _build(method, grad_checkpoint)
    ids, labels = _inputs()
    params = _trainable(trainer.model)

    trainer.cfg.train.infctx_backward = False
    loss_old, count_old, done_old = trainer._infctx_loss(ids, labels, None)
    assert not done_old
    loss_old.backward()
    grads_old = [p.grad.detach().clone() for p in params]

    trainer.model.zero_grad(set_to_none=True)
    trainer.cfg.train.infctx_backward = True
    loss_new, count_new, done_new = trainer._infctx_loss(ids, labels, None)
    assert done_new
    grads_new = [p.grad.detach().clone() for p in params]

    assert count_old == count_new
    torch.testing.assert_close(loss_new, loss_old, rtol=1e-5, atol=1e-6)
    names = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    for name, old, new in zip(names, grads_old, grads_new, strict=True):
        torch.testing.assert_close(new, old, rtol=1e-4, atol=1e-5, msg=name)
