"""交互式向导的纯逻辑测试（不启动 TUI）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rwkv_easy.config import Config
from rwkv_easy.wizard import (
    PRESETS,
    Console,
    Environment,
    FormField,
    WizardError,
    WizardState,
    _data_fields,
    _find_field,
    _home_items,
    _lora_fields,
    _misc_fields,
    _model_fields,
    _parallel_backward_label,
    _training_fields,
    _truncate,
    _wrap_text,
    apply_method_defaults,
    apply_preset,
    config_summary,
    cuda_tag_candidates,
    default_config,
    inspect_model,
    normalize_for_environment,
    text_width,
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
    assert parse_args(["--no-infctx-parallel-backward"]).infctx_parallel_backward is False
    assert parse_args(["--infctx-parallel-backward"]).infctx_parallel_backward is True


def test_all_wizard_fields_have_comments():
    env = _env(cuda=True, fla=True)
    state = WizardState(env=env, cfg=default_config(env))
    groups = [
        _training_fields(state),
        _model_fields(state),
        _data_fields(state),
        _misc_fields(state),
        _lora_fields(state),
    ]
    paths = [field.path for group in groups for field in group]
    assert len(paths) == len(set(paths)), "同一个字段在高级设置里出现了两次"
    for group in groups:
        for field in group:
            assert field.comment.strip(), f"{field.path} 缺少行内注释"
            assert field.help.strip(), f"{field.path} 缺少详细说明"


def test_presets_cover_main_scenarios():
    keys = {preset.key for preset in PRESETS}
    assert {"recommended", "lora-infctx", "state", "state-infctx", "low-memory", "quick-test"} <= keys
    for preset in PRESETS:
        assert preset.title and preset.tag and preset.summary and preset.description


def test_apply_preset_updates_training_but_keeps_paths(tmp_path, tiny_checkpoint, tiny_sharegpt):
    env = _env(cuda=True, fla=True)
    cfg = default_config(env)
    state = WizardState(env=env, cfg=cfg)
    state.set_model(inspect_model(str(tiny_checkpoint())))
    state.set_dataset(validate_dataset_path(str(tiny_sharegpt)))
    model_path = cfg.model.path
    data_path = cfg.data.path
    output_dir = cfg.run.output_dir

    apply_preset(state, "lora-infctx")
    assert cfg.train.method == "lora"
    assert cfg.train.infctx is True
    assert cfg.train.infctx_parallel_backward is True
    assert cfg.train.chunk_ctx == 2048
    assert cfg.train.ctx_len >= 8192
    assert state.preset_key == "lora-infctx"

    apply_preset(state, "state-infctx")
    assert cfg.train.method == "state"
    assert cfg.train.infctx is True
    assert cfg.train.infctx_parallel_backward is False
    assert cfg.train.lr == pytest.approx(5e-3)

    apply_preset(state, "state")
    assert cfg.train.method == "state"
    assert cfg.train.infctx is False
    assert cfg.train.lr == pytest.approx(5e-3)

    apply_preset(state, "low-memory")
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.grad_checkpoint is True
    assert cfg.train.compile is False
    assert cfg.run.name.endswith("lowmem")

    apply_preset(state, "quick-test")
    assert cfg.train.max_steps == 20
    assert cfg.train.log_every == 1
    assert cfg.run.name == "quick-test"

    assert cfg.model.path == model_path
    assert cfg.data.path == data_path
    assert cfg.run.output_dir == output_dir


def test_home_items_have_help_and_presets():
    env = _env(cuda=True, fla=True)
    state = WizardState(env=env, cfg=default_config(env))
    items = _home_items(state)
    actions = [item.action for item in items]
    assert "preset" in actions and "guide" in actions
    assert actions[-2:] == ["start", "save"]
    for item in items:
        assert item.comment.strip(), f"{item.label} 缺少行内注释"
        assert item.help.strip(), f"{item.label} 缺少详细说明"


def test_parallel_backward_label_and_home_item():
    env = _env(cuda=True, fla=True)
    env.gpu_count = 2
    state = WizardState(env=env, cfg=default_config(env))
    cfg = state.cfg
    cfg.train.infctx = True
    cfg.dist.mode = "ddp"
    cfg.train.infctx_parallel_backward = True
    assert "逐块" in (_parallel_backward_label(state) or "")
    cfg.train.infctx_parallel_backward = False
    assert "最后统一" in (_parallel_backward_label(state) or "")
    cfg.train.method = "state"
    assert "State" in (_parallel_backward_label(state) or "")
    cfg.dist.mode = "deepspeed"
    assert "DeepSpeed" in (_parallel_backward_label(state) or "")
    cfg.dist.mode = "single"
    assert _parallel_backward_label(state) is None

    cfg.dist.mode = "ddp"
    cfg.train.method = "lora"
    cfg.train.infctx_parallel_backward = True
    items = _home_items(state)
    assert "parallel_backward" in [item.action for item in items]


def test_cjk_width_truncate_and_wrap():
    assert text_width("abc") == 3
    assert text_width("中文") == 4
    assert text_width("\033[31m红色\033[0m") == 4
    assert _truncate("这是一个很长的中文句子", 8).endswith("…")
    assert text_width(_truncate("这是一个很长的中文句子", 8)) <= 8
    lines = _wrap_text("第一段很长的中文内容需要折行显示", 10)
    assert len(lines) >= 2
    assert all(text_width(line) <= 10 for line in lines)
    # 英文单词不能被从中间折断。
    wrapped = _wrap_text("这里有一个 assistant 单词需要保持完整", 12)
    assert any("assistant" in line for line in wrapped)
    # 中文逗号/句号不能跑到下一行开头。
    wrapped = _wrap_text("这是一段很长的话，需要折行。", 8)
    assert all(not line.startswith(("，", "。")) for line in wrapped)


def test_form_field_search():
    env = _env(cuda=True, fla=True)
    fields = _training_fields(WizardState(env=env, cfg=default_config(env)))
    by_path = _find_field(fields, "ctx_len")
    assert by_path is not None and fields[by_path].path == "train.ctx_len"
    by_label = _find_field(fields, "学习率")
    assert by_label is not None and fields[by_label].path == "train.lr"
    assert _find_field(fields, "不存在的参数") is None
    assert _find_field(fields, "  ") is None


def test_compact_summary_stays_short(tmp_path, tiny_checkpoint, tiny_sharegpt):
    env = _env(cuda=True, fla=True)
    state = WizardState(env=env, cfg=default_config(env))
    state.set_model(inspect_model(str(tiny_checkpoint())))
    state.set_dataset(validate_dataset_path(str(tiny_sharegpt)))
    compact = config_summary(state, compact=True)
    full = config_summary(state, compact=False)
    assert "compile" not in compact
    assert "compile" in full
    assert len(compact.splitlines()) <= 9
