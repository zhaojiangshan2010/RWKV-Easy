"""DeepSpeed 配置生成测试（不需要安装 deepspeed）。"""

from __future__ import annotations

import pytest
from conftest import build_config

from rwkv_easy.train.deepspeed_backend import build_deepspeed_config


def _cfg(tmp_path, tiny_checkpoint, tiny_sharegpt, **overrides):
    return build_config(tmp_path, tiny_checkpoint(), tiny_sharegpt, overrides or None)


def test_zero2_defaults(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = _cfg(
        tmp_path,
        tiny_checkpoint,
        tiny_sharegpt,
        train={"micro_batch_size": 2, "grad_accum": 4, "precision": "bf16"},
        dist={"mode": "deepspeed", "deepspeed": {"zero_stage": 2}},
    )
    ds = build_deepspeed_config(cfg, world_size=2)
    assert ds["train_batch_size"] == 2 * 2  # 梯度累积由主循环负责，不乘 accum
    assert ds["train_micro_batch_size_per_gpu"] == 2
    assert ds["gradient_accumulation_steps"] == 1
    assert ds["zero_optimization"]["stage"] == 2
    assert ds["zero_optimization"]["allgather_bucket_size"] == 2.0e8
    assert ds["bf16"]["enabled"] is True
    assert ds["fp16"]["enabled"] is False
    assert ds["gradient_clipping"] == 1.0


def test_fp16(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = _cfg(
        tmp_path,
        tiny_checkpoint,
        tiny_sharegpt,
        train={"precision": "fp16"},
        dist={"mode": "deepspeed"},
    )
    ds = build_deepspeed_config(cfg, world_size=1)
    assert ds["fp16"]["enabled"] is True
    assert ds["bf16"]["enabled"] is False


def test_zero1_and_offload(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = _cfg(
        tmp_path,
        tiny_checkpoint,
        tiny_sharegpt,
        dist={
            "mode": "deepspeed",
            "deepspeed": {"zero_stage": 1, "offload_optimizer": True},
        },
    )
    ds = build_deepspeed_config(cfg, world_size=1)
    assert ds["zero_optimization"]["stage"] == 1
    assert ds["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert "allgather_bucket_size" not in ds["zero_optimization"]


def test_offload_param_requires_zero3(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = _cfg(
        tmp_path,
        tiny_checkpoint,
        tiny_sharegpt,
        dist={"mode": "deepspeed", "deepspeed": {"zero_stage": 2, "offload_param": True}},
    )
    with pytest.raises(ValueError):
        build_deepspeed_config(cfg, world_size=1)


def test_zero3(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = _cfg(
        tmp_path,
        tiny_checkpoint,
        tiny_sharegpt,
        dist={"mode": "deepspeed", "deepspeed": {"zero_stage": 3, "offload_param": True}},
    )
    ds = build_deepspeed_config(cfg, world_size=8)
    assert ds["zero_optimization"]["stage"] == 3
    assert ds["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] is True
    assert ds["zero_optimization"]["offload_param"]["device"] == "cpu"
