"""训练时只对可训练 token 过 head，必须与整段 logits + ignore_index 等价。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rwkv_easy.config import Config
from rwkv_easy.models.checkpoint import random_model
from rwkv_easy.models.rwkv7 import RWKV7Config
from rwkv_easy.train.loop import Trainer


def _setup():
    cfg = Config.from_dict({"train": {"l2wrap": 0.0}})
    model = random_model(
        RWKV7Config(vocab_size=64, n_layer=2, n_embd=32, head_size=8), seed=0
    )
    model.train()
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.model = model
    trainer.backend = None
    trainer.grad_accum = 1
    trainer.device = torch.device("cpu")
    return trainer, model


def test_forward_equals_head_of_hidden():
    _, model = _setup()
    ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        logits, _ = model(ids)
        hidden, _ = model.forward_hidden(ids)
        torch.testing.assert_close(logits, model.head(hidden))


def test_selected_head_matches_full_logits():
    trainer, model = _setup()
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(2, 12, 32, generator=generator, requires_grad=True)
    labels = torch.randint(0, 64, (2, 12), generator=generator)
    labels[:, ::2] = -100

    loss_new, count = trainer._cross_entropy(hidden, labels)
    loss_new.backward()
    assert hidden.grad is not None
    grad_new = hidden.grad.clone()

    hidden2 = hidden.detach().clone().requires_grad_(True)
    logits = model.head(hidden2)
    loss_old = (
        F.cross_entropy(
            logits.reshape(-1, 64),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        / count
    )
    loss_old.backward()
    assert hidden2.grad is not None

    torch.testing.assert_close(loss_new, loss_old, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(grad_new, hidden2.grad, rtol=1e-5, atol=1e-6)
