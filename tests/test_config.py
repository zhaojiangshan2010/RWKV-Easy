"""配置系统测试。"""

from __future__ import annotations

import pytest
import yaml
from conftest import build_config

from rwkv_easy.config import Config, ConfigError, template_config


def test_from_dict_roundtrip(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(tmp_path, tiny_checkpoint(), tiny_sharegpt)
    data = cfg.to_dict()
    again = Config.from_dict(data)
    assert again.train.method == "lora"
    assert again.train.micro_batch_size == 2
    out = tmp_path / "cfg.yaml"
    cfg.save_yaml(out)
    loaded = Config.load(out)
    assert loaded.run.seed == 42


def test_unknown_key_gives_hint():
    with pytest.raises(ConfigError) as exc:
        Config.from_dict({"train": {"learnig_rate": 1e-4}})
    message = str(exc.value)
    assert "learnig_rate" in message
    assert "可用的字段" in message


def test_unknown_key_typo_hint():
    with pytest.raises(ConfigError) as exc:
        Config.from_dict({"train": {"laming_rate": 1e-4}})
    assert "你是不是想写" in str(exc.value) or "可用的字段" in str(exc.value)


def test_set_override(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(tmp_path, tiny_checkpoint(), tiny_sharegpt)
    path = tmp_path / "cfg.yaml"
    cfg.save_yaml(path)
    loaded = Config.load(path, ["train.lr=2.5e-5", "lora.rank=8"])
    assert loaded.train.lr == 2.5e-5
    assert loaded.lora.rank == 8


def test_set_unknown_section(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(tmp_path, tiny_checkpoint(), tiny_sharegpt)
    path = tmp_path / "cfg.yaml"
    cfg.save_yaml(path)
    with pytest.raises(ConfigError):
        Config.load(path, ["nope.x=1"])


def test_validation_sample_with_infctx(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"data": {"mode": "sample"}, "train": {"infctx": True}},
        validate=False,
    )
    with pytest.raises(ConfigError) as exc:
        cfg.validate()
    assert "infctx" in str(exc.value)


def test_validation_chunk_larger_than_ctx(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"train": {"infctx": True, "ctx_len": 64, "chunk_ctx": 128}},
        validate=False,
    )
    with pytest.raises(ConfigError) as exc:
        cfg.validate()
    assert "chunk_ctx" in str(exc.value)


def test_validation_missing_paths(tmp_path):
    cfg = Config.from_dict({"data": {"path": str(tmp_path / "missing.json")}})
    with pytest.raises(ConfigError) as exc:
        cfg.validate()
    message = str(exc.value)
    assert "data.path" in message
    assert "model.path" in message


def test_fla_forces_softplus(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path, tiny_checkpoint(), tiny_sharegpt, {"model": {"backend": "fla", "decay": "auto"}}
    )
    cfg.resolve()
    assert cfg.model.decay == "softplus"

    cfg2 = build_config(
        tmp_path, tiny_checkpoint(), tiny_sharegpt, {"model": {"backend": "fla", "decay": "clampw"}}
    )
    with pytest.raises(ConfigError):
        cfg2.resolve()


def test_template_is_loadable(tmp_path):
    text = template_config("state", infctx=True, model_path="a.pth", data_path="b.json")
    path = tmp_path / "t.yaml"
    path.write_text(text, encoding="utf-8")
    raw = yaml.safe_load(text)
    assert raw["train"]["method"] == "state"
    assert raw["train"]["infctx"] is True
    assert raw["model"]["path"] == "a.pth"
    assert raw["data"]["path"] == "b.json"


def test_precision_normalization(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(tmp_path, tiny_checkpoint(), tiny_sharegpt, {"train": {"precision": "bf16-true"}})
    cfg.resolve()
    assert cfg.train.precision == "bf16-full"
