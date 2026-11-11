"""CLI 测试：init / merge / doctor / inspect-data / generate / train。"""

from __future__ import annotations

import pytest
import torch
from conftest import build_config

from rwkv_easy.cli import main
from rwkv_easy.lora import inject_lora, lora_state_dict
from rwkv_easy.models.checkpoint import load_model_from_checkpoint, random_model
from rwkv_easy.models.rwkv7 import RWKV7Config
from rwkv_easy.train.checkpoint import save_adapter


def test_init_non_interactive(tmp_path, tiny_checkpoint, tiny_sharegpt):
    model_path = tiny_checkpoint()
    output = tmp_path / "my.yaml"
    code = main(
        [
            "init",
            "--non-interactive",
            "--model",
            str(model_path),
            "--data",
            str(tiny_sharegpt),
            "--method",
            "lora",
            "--name",
            "demo",
            "-o",
            str(output),
        ]
    )
    assert code == 0
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert str(model_path) in text
    assert "method: lora" in text


def test_init_refuses_overwrite(tmp_path, tiny_checkpoint, tiny_sharegpt):
    output = tmp_path / "my.yaml"
    output.write_text("x", encoding="utf-8")
    code = main(
        [
            "init",
            "--non-interactive",
            "--model",
            str(tiny_checkpoint()),
            "--data",
            str(tiny_sharegpt),
            "-o",
            str(output),
        ]
    )
    assert code == 2


def _make_adapter(tmp_path, base_path):
    model = random_model(RWKV7Config(256, 2, 32, 8), seed=1)
    inject_lora(model, ["att.key", "ffn.value"], rank=4, alpha=8)
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "lora_B"):
                module.lora_A.normal_()
                module.lora_B.normal_()
    adapter = tmp_path / "adapter-step0000001.pth"
    save_adapter(
        adapter,
        method="lora",
        step=1,
        epoch=0,
        weights=lora_state_dict(model),
        base_signature="test",
        model_config={"vocab_size": 256, "n_layer": 2, "n_embd": 32, "head_size": 8, "head_size_divisor": 8},
        decay="clampw",
        method_config={"rank": 4, "alpha": 8, "target_modules": ["att.key", "ffn.value"]},
    )
    return adapter


def test_merge(tmp_path, tiny_checkpoint):
    base_path = tiny_checkpoint(vocab_size=256, n_layer=2, n_embd=32, head_size=8)
    adapter = _make_adapter(tmp_path, base_path)
    out = tmp_path / "merged.pth"
    code = main(["merge", "--base", str(base_path), "--adapter", str(adapter), "-o", str(out)])
    assert code == 0
    assert out.exists()
    model, cfg, _ = load_model_from_checkpoint(out)
    assert cfg.vocab_size == 256
    assert not any("lora_" in key for key in model.state_dict())


def test_doctor(capsys):
    code = main(["doctor"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "WKV 自检" in captured
    assert "Python" in captured


def test_inspect_data(tmp_path, tiny_checkpoint, tiny_sharegpt, capsys):
    cfg = build_config(tmp_path, tiny_checkpoint(), tiny_sharegpt)
    path = tmp_path / "cfg.yaml"
    cfg.save_yaml(path)
    code = main(["inspect-data", "-c", str(path), "-n", "1"])
    assert code == 0
    captured = capsys.readouterr().out
    assert "可训练 tokens" in captured
    assert "【" in captured


def test_generate(tmp_path, tiny_checkpoint):
    base_path = tiny_checkpoint(vocab_size=65536, n_layer=2, n_embd=64, head_size=16)
    code = main(
        [
            "generate",
            "-m",
            str(base_path),
            "-p",
            "你好",
            "--max-tokens",
            "2",
            "--template",
            "none",
        ]
    )
    assert code == 0


def test_train_cli(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"train": {"max_steps": 2, "merge_at_end": False}, "run": {"output_dir": str(tmp_path / "cli-runs")}},
    )
    path = tmp_path / "cfg.yaml"
    cfg.save_yaml(path)
    code = main(["train", "-c", str(path), "--single"])
    assert code == 0
    runs = list((tmp_path / "cli-runs").glob("**/adapter-step0000002.pth"))
    assert runs


def test_unknown_command():
    with pytest.raises(SystemExit):
        main(["nope"])


def test_merge_state(tmp_path, tiny_checkpoint):
    from rwkv_easy.state_tuning import attach_state_tuning, state_state_dict

    base_path = tiny_checkpoint(vocab_size=256, n_layer=2, n_embd=32, head_size=8)
    model = random_model(RWKV7Config(256, 2, 32, 8), seed=2)
    attach_state_tuning(model)
    with torch.no_grad():
        for block in model.blocks:
            block.att.time_state.normal_(std=0.05)
    adapter = tmp_path / "state-step0000003.pth"
    save_adapter(
        adapter,
        method="state",
        step=3,
        epoch=0,
        weights=state_state_dict(model),
        base_signature="test",
        model_config={"vocab_size": 256, "n_layer": 2, "n_embd": 32, "head_size": 8, "head_size_divisor": 8},
        decay="clampw",
    )
    out = tmp_path / "merged_state.pth"
    code = main(["merge", "--base", str(base_path), "--adapter", str(adapter), "-o", str(out)])
    assert code == 0
    merged, _, _ = load_model_from_checkpoint(out)
    assert any("time_state" in key for key in merged.state_dict())
    for key, value in state_state_dict(model).items():
        assert torch.allclose(merged.state_dict()[key].cpu(), value)


def test_check_model(tmp_path, tiny_checkpoint, capsys):
    base = tiny_checkpoint(vocab_size=65536, n_layer=2, n_embd=64, head_size=16)
    code = main(["check-model", "-m", str(base), "--max-tokens", "2"])
    assert code == 0
    output = capsys.readouterr().out
    assert "RWKV-EASY 模型验收" in output
    assert "decay 校准" in output
    assert "结构" in output
