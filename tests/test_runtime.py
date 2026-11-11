"""运行时工具的单元测试。"""

from __future__ import annotations

import os

import pytest

from rwkv_easy.runtime import seed_everything


def test_seed_everything_exports_worker_seed(monkeypatch):
    monkeypatch.delenv("RWKV_EASY_SEED", raising=False)
    seed_everything(1234)
    assert os.environ["RWKV_EASY_SEED"] == "1234"


def test_seed_everything_masks_negative(monkeypatch):
    monkeypatch.delenv("RWKV_EASY_SEED", raising=False)
    seed_everything(-1)
    assert os.environ["RWKV_EASY_SEED"] == str((-1) & 0x7FFFFFFF)


@pytest.mark.parametrize("deterministic", [False, True])
def test_seed_everything_runs(deterministic):
    seed_everything(42, deterministic=deterministic)
