"""测试公共工具：小模型、小数据集、配置构造。"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv_easy.config import Config  # noqa: E402
from rwkv_easy.models.checkpoint import random_model  # noqa: E402
from rwkv_easy.models.rwkv7 import RWKV7Config  # noqa: E402

CONVERSATIONS = [
    {
        "conversations": [
            {"from": "human", "value": "你好，请介绍你自己。"},
            {"from": "gpt", "value": "我是一个用于测试的小助手。"},
        ]
    },
    {
        "conversations": [
            {"from": "system", "value": "你是一个简洁的助手。"},
            {"from": "human", "value": "一加一等于几？"},
            {"from": "gpt", "value": "等于二。"},
            {"from": "human", "value": "再加三呢？"},
            {"from": "gpt", "value": "等于五。"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "写一个打招呼的函数。"},
            {"role": "assistant", "content": "def hello():\n    return 'hi'"},
        ]
    },
    {"conversations": [{"from": "human", "value": "今天天气如何？"}, {"from": "gpt", "value": "今天晴朗。"}]},
    {"prompt": "地球是圆的吗？", "response": "是的。"},
    {"text": "这是一段纯文本，用来测试 text 记录。"},
] * 3


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@pytest.fixture
def tiny_model_cfg() -> RWKV7Config:
    return RWKV7Config(vocab_size=128, n_layer=2, n_embd=32, head_size=8)


@pytest.fixture
def tiny_checkpoint(tmp_path: Path):
    def _make(
        vocab_size: int = 65536,
        n_layer: int = 2,
        n_embd: int = 64,
        head_size: int = 16,
        seed: int = 0,
        name: str = "tiny-model.pth",
    ) -> Path:
        path = tmp_path / name
        model = random_model(
            RWKV7Config(
                vocab_size=vocab_size, n_layer=n_layer, n_embd=n_embd, head_size=head_size
            ),
            seed=seed,
        )
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
        return path

    return _make


@pytest.fixture
def tiny_sharegpt(tmp_path: Path) -> Path:
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(CONVERSATIONS, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_config(
    tmp_path: Path,
    model_path: Path,
    data_path: Path,
    overrides: dict[str, Any] | None = None,
    validate: bool = True,
) -> Config:
    base: dict[str, Any] = {
        "run": {"name": "test", "output_dir": str(tmp_path / "runs"), "seed": 42},
        "model": {"path": str(model_path)},
        "train": {
            "method": "lora",
            "ctx_len": 64,
            "chunk_ctx": 32,
            "micro_batch_size": 2,
            "grad_accum": 1,
            "epochs": 100,
            "max_steps": 6,
            "lr": 1.0e-3,
            "min_lr": 1.0e-4,
            "warmup_steps": 1,
            "precision": "fp32",
            "grad_checkpoint": False,
            "log_every": 2,
            "save_every_epochs": 0,
            "merge_at_end": False,
        },
        "lora": {"rank": 4, "alpha": 8, "dropout": 0.0},
        "state": {"init_std": 0.0},
        "data": {
            "path": str(data_path),
            "mode": "packed",
            "num_workers": 0,
            "num_proc": 0,
            "cache": False,
            "val_ratio": 0.0,
        },
        "dist": {"mode": "single"},
        "log": {"jsonl": True},
    }
    if overrides:
        base = deep_merge(base, overrides)
    cfg = Config.from_dict(base)
    if validate:
        cfg.validate()
    return cfg
