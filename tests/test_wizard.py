"""交互式向导的纯逻辑测试（不启动 TUI）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rwkv_easy.config import Config
from rwkv_easy.wizard import (
    Console,
    Environment,
    FormField,
    WizardError,
    WizardState,
    apply_method_defaults,
    config_summary,
    cuda_tag_candidates,
    default_config,
    inspect_model,
    normalize_for_environment,
    validate_dataset_path,
    validate_model_path,
)


def _env(cuda: bool = False, fla: bool = False) -> Environment:
    return Environment(
        os_name="nt",
        python="python",
        python_version="3.11.9",
        device="cuda" if cuda else "cpu",
        torch_version="2.15.0",
        torch_cuda="13.0" if cuda else None,
        cuda_available=cuda,
        gpu_count=1 if cuda else 0,
        gpu_name="GPU" if cuda else None,
        gpu_memory_gib=24.0 if cuda else None,
        fla_available=fla,
        deepspeed_available=False,
    )


def test_cuda_tag_candidates():
    assert cuda_tag_candidates("12.4")[:2] == ["cu124", "cu121"]
    assert cuda_tag_candidates("13.0")[0] == "cu130"
    assert cuda_tag_candidates("11.8") == ["cu118"]
    assert cuda_tag_candidates(None)[0] == "cu130"


def test_default_config_cpu_and_gpu():
    cpu = default_config(_env())
    assert cpu.model.backend == "native"
    assert cpu.train.precision == "fp32"
    assert cpu.train.compile is False
    assert cpu.train.device == "cpu"

    gpu = default_config(_env(cuda=True, fla=True))
    assert gpu.model.backend == "fla"
    assert gpu.train.precision == "bf16-full"
    assert gpu.train.compile is True


def test_method_and_infctx_defaults():
    cfg = default_config(_env(cuda=True, fla=True))
    apply_method_defaults(cfg, "state", infctx=True)
    assert cfg.train.method == "state"
    assert cfg.train.infctx is True
    assert cfg.train.lr == pytest.approx(5e-3)
    assert cfg.train.weight_decay == 0.0
    assert cfg.train.chunk_ctx == 2048
    assert cfg.data.mode == "packed"
    apply_method_defaults(cfg, "lora", infctx=False)
    assert cfg.train.lr == pytest.approx(1e-4)
    assert cfg.train.chunk_ctx == 512


def test_normalize_cpu_forces_safe_values():
    cfg = default_config(_env(cuda=True, fla=True))
    cfg.model.backend = "fla"
    cfg.train.precision = "bf16-full"
    normalize_for_environment(cfg, _env())
    assert cfg.model.backend == "native"
    assert cfg.train.precision == "fp32"
    assert cfg.train.compile is False


def test_validate_paths(tmp_path, tiny_checkpoint, tiny_sharegpt):
    with pytest.raises(WizardError):
        validate_model_path(tmp_path / "missing.pth")
    info = inspect_model(str(tiny_checkpoint()))
    assert info.n_layer == 2
    assert info.n_embd == 64
    data = validate_dataset_path(str(tiny_sharegpt))
    assert "ShareGPT" in data.kind


def test_state_build_and_save(tmp_path, tiny_checkpoint, tiny_sharegpt):
    env = _env()
    cfg = default_config(env)
    state = WizardState(env=env, cfg=cfg)
    state.set_model(inspect_model(str(tiny_checkpoint())))
    state.set_dataset(validate_dataset_path(str(tiny_sharegpt)))
    state.set_method("lora")
    cfg.lora.target_layers = [0, 1]
    assert state.validate() == []
    assert "LoRA" in config_summary(state)

    path = tmp_path / "wizard.yaml"
    cfg.save_yaml(path)
    loaded = Config.load(path)
    assert loaded.train.method == "lora"
    assert loaded.lora.target_layers == [0, 1]
    assert loaded.data.path == str(tiny_sharegpt)


def test_form_field_shift():
    cfg = Config()
    field = FormField("ctx", "train.ctx_len", "int", step=512, minimum=64)
    assert field.get(cfg) == 2048
    field.shift(cfg, 1)
    assert field.get(cfg) == 2560
    field.shift(cfg, -10)
    assert field.get(cfg) == 64


def test_console_non_tty_is_line_mode():
    console = Console(non_interactive=True)
    assert console.tty is False
    assert console.color is False


def test_text_jsonl_dataset(tmp_path):
    path = tmp_path / "text.jsonl"
    path.write_text(json.dumps({"text": "hello"}, ensure_ascii=False) + "\n", encoding="utf-8")
    info = validate_dataset_path(str(path))
    assert "text" in info.kind or "纯文本" in info.kind


def test_quickstart_helpers():
    from quickstart import cuda_tags, parse_args, venv_python

    assert cuda_tags("12.4")[0] == "cu124"
    args = parse_args(["--cpu", "--non-interactive", "--no-train"])
    assert args.cpu is True
    assert args.non_interactive is True
    assert args.no_train is True
    assert venv_python(Path(".")).name in ("python", "python.exe")
