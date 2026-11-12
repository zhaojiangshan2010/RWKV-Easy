"""quickstart 的依赖探测 / venv 复用测试（不实际安装任何包）。"""

from __future__ import annotations

import os
import venv
from pathlib import Path

from quickstart import (
    PROJECT_ROOT,
    main_site_packages,
    parse_args,
    plan_required_installs,
    probe_environment,
    venv_site_packages,
    venv_uses_system_site_packages,
    write_system_site_link,
)


def _venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def test_parse_args_clone_and_parallel_flags():
    args = parse_args(["--isolated-venv", "--no-infctx-parallel-backward"])
    assert args.isolated_venv is True
    assert args.infctx_parallel_backward is False
    assert parse_args([]).isolated_venv is False


def test_probe_environment_reports_required_packages():
    probe = probe_environment()
    assert probe["python"]["version"]
    assert probe["torch"]["available"]
    assert probe["torch"]["version"]
    assert probe["yaml"]["available"]
    assert probe["rwkv_easy"]["available"]
    assert Path(probe["rwkv_easy"]["file"]).name == "__init__.py"


def test_install_plan_skips_present_and_detects_missing():
    args = parse_args([])
    probe_ok = {
        "torch": {"available": True, "cuda_available": False},
        "yaml": {"available": True},
        "rwkv_easy": {
            "available": True,
            "file": str(PROJECT_ROOT / "rwkv_easy" / "__init__.py"),
        },
    }
    plan = plan_required_installs(probe_ok, {}, args)
    assert plan == {"torch": False, "pyyaml": False, "project": False}

    missing = {
        "torch": {"available": False},
        "yaml": {"available": False},
        "rwkv_easy": {"available": False},
    }
    plan = plan_required_installs(missing, {}, args)
    assert plan == {"torch": True, "pyyaml": True, "project": True}

    # 有 GPU 但当前 torch 是 CPU 版 -> 需要安装 CUDA 版。
    assert plan_required_installs(probe_ok, {"name": "GPU"}, args)["torch"] is True
    assert (
        plan_required_installs(probe_ok, {"name": "GPU"}, parse_args(["--cpu"]))["torch"]
        is False
    )

    # import 到的是别处的 rwkv_easy -> 需要在本 venv 里重装 editable。
    outside = dict(probe_ok)
    outside["rwkv_easy"] = {"available": True, "file": str(Path("/tmp/other/rwkv_easy.py"))}
    assert plan_required_installs(outside, {}, args)["project"] is True


def test_venv_inherits_main_site_packages_and_pth(tmp_path):
    inherited = tmp_path / "inherited"
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(inherited)
    assert venv_uses_system_site_packages(inherited) is True
    site = venv_site_packages(_venv_python(inherited))
    assert site is not None and site.is_dir()
    # 继承主环境后，venv 里直接能探测到 torch。
    probe = probe_environment(_venv_python(inherited))
    assert probe["torch"]["available"] is True

    isolated = tmp_path / "isolated"
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(isolated)
    assert venv_uses_system_site_packages(isolated) is False
    site = venv_site_packages(_venv_python(isolated))
    main_site = main_site_packages()
    assert site is not None and main_site is not None
    link = write_system_site_link(site, main_site)
    assert link is not None
    assert str(main_site) in link.read_text(encoding="utf-8")
    # 写入 .pth 后，isolated venv 也能 import 主环境库。
    probe = probe_environment(_venv_python(isolated))
    assert probe["torch"]["available"] is True
