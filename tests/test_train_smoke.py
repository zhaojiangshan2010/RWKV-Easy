"""端到端训练冒烟测试（CPU，几秒级）。"""

from __future__ import annotations

import json

import torch
from conftest import build_config

from rwkv_easy.models.checkpoint import load_model_from_checkpoint
from rwkv_easy.train.checkpoint import read_adapter
from rwkv_easy.train.runner import run_training


def _losses(run_dir):
    values = []
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        loss = record.get("loss")
        if isinstance(loss, (int, float)) and loss == loss:
            values.append(float(loss))
    return values


def test_train_lora_packed(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"train": {"max_steps": 12, "lr": 3.0e-3, "merge_at_end": True}},
    )
    result = run_training(cfg, run_dir=tmp_path / "run")
    assert result.steps == 12
    assert result.last_adapter is not None and result.last_adapter.exists()
    assert result.merged is not None and result.merged.exists()

    losses = _losses(result.run_dir)
    assert len(losses) >= 4
    first = sum(losses[:2]) / 2
    last = sum(losses[-2:]) / 2
    assert last < first - 0.02, f"loss 没有下降：{first:.4f} -> {last:.4f}"

    # 合并后的权重可以被正常加载，且不含 lora key
    model, model_cfg, _ = load_model_from_checkpoint(result.merged)
    assert model_cfg.n_layer == 2
    assert not any("lora_" in key for key in model.state_dict())

    info = read_adapter(result.last_adapter)
    assert info.method == "lora"
    assert info.weights


def test_train_state_infctx(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {
            "train": {"method": "state", "infctx": True, "max_steps": 8, "lr": 5.0e-2, "merge_at_end": True},
            "state": {"init_std": 0.0},
        },
    )
    result = run_training(cfg, run_dir=tmp_path / "run")
    assert result.steps == 8
    adapters = sorted(result.run_dir.glob("state-step*.pth"))
    assert adapters
    info = read_adapter(adapters[-1])
    assert info.method == "state"
    assert all(name.endswith("time_state") for name in info.weights)
    assert any(float(tensor.abs().sum()) > 0 for tensor in info.weights.values())

    merged, _, _ = load_model_from_checkpoint(result.run_dir / "merged.pth")
    assert any("time_state" in key for key in merged.state_dict())
    assert not any("lora_" in key for key in merged.state_dict())


def test_train_sample_mode(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"data": {"mode": "sample"}, "train": {"max_steps": 4}},
    )
    result = run_training(cfg, run_dir=tmp_path / "run")
    assert result.steps == 4


def test_train_bf16_full(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"train": {"max_steps": 3, "precision": "bf16-full", "merge_at_end": True}},
    )
    result = run_training(cfg, run_dir=tmp_path / "run")
    assert result.steps == 3
    model, _, _ = load_model_from_checkpoint(result.merged)
    assert next(model.parameters()).dtype == torch.bfloat16


def test_train_bf16_autocast(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"train": {"max_steps": 3, "precision": "bf16", "infctx": True}},
    )
    result = run_training(cfg, run_dir=tmp_path / "run")
    assert result.steps == 3


def test_train_bf16_full_infctx(tmp_path, tiny_checkpoint, tiny_sharegpt):
    """bf16-full + infctx：状态张量必须与参数 dtype 一致，否则 Linear 会报 dtype 不匹配。"""
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"train": {"max_steps": 3, "precision": "bf16-full", "infctx": True}},
    )
    result = run_training(cfg, run_dir=tmp_path / "run")
    assert result.steps == 3


def test_resume(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"train": {"max_steps": 3, "save_every_epochs": 0, "lr": 1.0e-3}},
    )
    first = run_training(cfg, run_dir=tmp_path / "run")
    assert first.steps == 3
    assert (first.run_dir / "latest.txt").exists()

    cfg2 = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"train": {"max_steps": 6, "save_every_epochs": 0, "lr": 1.0e-3}},
    )
    second = run_training(cfg2, resume=first.run_dir, run_dir=first.run_dir)
    assert second.steps == 6
    assert (first.run_dir / "adapter-step0000006.pth").exists()


def test_validation_split(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {
            "data": {"val_ratio": 0.25},
            "train": {"max_steps": 4, "val_every_steps": 2, "ctx_len": 32},
        },
    )
    result = run_training(cfg, run_dir=tmp_path / "run")
    assert result.steps == 4
    text = (result.run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    assert "eval_loss" in text


def test_grad_accum_and_clip(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"train": {"max_steps": 4, "grad_accum": 3, "grad_clip": 0.5}},
    )
    result = run_training(cfg, run_dir=tmp_path / "run")
    assert result.steps == 4


def test_max_steps_overrides_epochs(tmp_path, tiny_checkpoint, tiny_sharegpt):
    """max_steps 应决定总步数，而不是被 epochs=1 截断。"""
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"train": {"max_steps": 10, "epochs": 1}, "data": {"num_workers": 0}},
    )
    result = run_training(cfg, run_dir=tmp_path / "run")
    assert result.steps == 10
