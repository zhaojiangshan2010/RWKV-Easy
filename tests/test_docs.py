"""文档完整性测试：新增配置字段不能漏掉参数手册。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from rwkv_easy.config import Config

ROOT = Path(__file__).resolve().parent.parent


def _all_paths(obj: object, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for field in dataclasses.fields(obj):  # type: ignore[arg-type]
        value = getattr(obj, field.name)
        path = f"{prefix}.{field.name}" if prefix else field.name
        if dataclasses.is_dataclass(value):
            paths.extend(_all_paths(value, path))
        else:
            paths.append(path)
    return paths


def test_parameter_manual_covers_every_config_field():
    text = (ROOT / "docs" / "TRAINING_PARAMETERS.md").read_text(encoding="utf-8")
    missing = [path for path in _all_paths(Config()) if f"`{path}`" not in text]
    assert not missing, "参数手册缺少字段：" + ", ".join(missing)
