#!/usr/bin/env python3
"""RWKV-EASY 一键安装 + 交互式训练向导。

用法（项目根目录）::

    python quickstart.py

它会：

1. 检测 Python / NVIDIA 驱动 / CUDA 版本；
2. 需要时创建 ``.venv``，默认复用主环境已装的库（system-site-packages），
   只安装缺失的依赖；
3. 逐个探测 torch / PyYAML / 本项目 / rwkv-fla，已装的跳过；
4. 进入交互式向导：选择 LoRA / State / infctx、模型、数据集、层与模块，
   保存 YAML 并直接开始训练。

常用参数::

    --no-venv        不创建 .venv，直接装到当前 Python 环境
    --isolated-venv  .venv 不复用主环境已装的库（默认复用，避免重装 torch）
    --skip-install   跳过安装，直接进向导（环境已就绪时）
    --install-only   只安装，不进向导
    --cpu            强制 CPU 版 PyTorch
    --cuda-tag TAG   指定 wheel 标签，如 cu124 / cu128
    --reinstall      已有 torch 也重新安装
    --yes            自动确认可选依赖（rwkv-fla）
    --non-interactive / --no-train / -o  透传给向导
    --(no-)infctx-parallel-backward      透传给向导：DDP + infctx 是否逐块反传
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
STAGE_ENV = "RWKV_EASY_QUICKSTART_STAGE"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
CYAN = "\033[0;36m"
NC = "\033[0m"


def color(text: str, code: str) -> str:
    if os.environ.get("NO_COLOR") is not None or not sys.stdout.isatty():
        return text
    return f"{code}{text}{NC}"


def say(text: str) -> None:
    print(color("[RWKV-EASY] ", GREEN) + text, flush=True)


def warn(text: str) -> None:
    print(color("[提示] ", YELLOW) + text, flush=True)


def die(text: str) -> int:
    print(color("[错误] ", RED) + text, file=sys.stderr, flush=True)
    return 1


def run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    say("$ " + shlex.join(cmd))
    try:
        return int(subprocess.call(cmd, env=env))
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        warn(f"命令启动失败：{exc}")
        return 1


# --------------------------------------------------------------------------- #
# 依赖探测：不安装，只报告哪些库已经可用
# --------------------------------------------------------------------------- #

_PROBE_MARKER = "<<<RWKV_EASY_PROBE>>>"
_PROBE_CODE = r"""
import importlib.util
import json
import sys

names = {
    "torch": "torch",
    "yaml": "yaml",
    "rwkv_easy": "rwkv_easy",
    "rwkvfla": "rwkvfla",
    "deepspeed": "deepspeed",
    "safetensors": "safetensors",
    "triton": "triton",
}
out = {"python": {"version": sys.version.split()[0], "executable": sys.executable}}
for key, module in names.items():
    try:
        out[key] = {"available": importlib.util.find_spec(module) is not None}
    except Exception as exc:
        out[key] = {"available": False, "error": str(exc)}
if out["torch"].get("available"):
    try:
        import torch
        out["torch"].update(
            version=torch.__version__,
            cuda=torch.version.cuda,
            cuda_available=bool(torch.cuda.is_available()),
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            out["torch"]["gpu_name"] = props.name
            out["torch"]["memory_gib"] = props.total_memory / 1024**3
            out["torch"]["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:
        out["torch"] = {"available": False, "error": str(exc)}
if out["yaml"].get("available"):
    try:
        import yaml
        out["yaml"]["version"] = getattr(yaml, "__version__", None)
    except Exception as exc:
        out["yaml"] = {"available": False, "error": str(exc)}
if out["rwkv_easy"].get("available"):
    try:
        import rwkv_easy
        out["rwkv_easy"]["version"] = getattr(rwkv_easy, "__version__", None)
        out["rwkv_easy"]["file"] = getattr(rwkv_easy, "__file__", None)
    except Exception as exc:
        out["rwkv_easy"] = {"available": False, "error": str(exc)}
if out["rwkvfla"].get("available"):
    try:
        import rwkvfla  # noqa: F401
        out["rwkvfla"]["version"] = getattr(rwkvfla, "__version__", None)
    except Exception as exc:
        out["rwkvfla"] = {"available": False, "error": str(exc)}
print("MARKER" + json.dumps(out, ensure_ascii=False))
""".replace("MARKER", _PROBE_MARKER)


def probe_environment(python: str | Path | None = None) -> dict[str, dict]:
    """在指定 Python 里探测依赖；返回 {包名: {available, version, ...}}。"""
    executable = str(python or sys.executable)
    try:
        result = subprocess.run(
            [executable, "-c", _PROBE_CODE],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    for line in (result.stdout or "").splitlines():
        if line.startswith(_PROBE_MARKER):
            try:
                payload = json.loads(line[len(_PROBE_MARKER) :])
                return payload if isinstance(payload, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def package_available(probe: dict[str, dict], name: str) -> bool:
    return bool(probe.get(name, {}).get("available"))


def _project_imports_from_root(probe: dict[str, dict]) -> bool:
    info = probe.get("rwkv_easy", {})
    if not info.get("available"):
        return False
    file = info.get("file")
    if not file:
        return False
    try:
        return Path(file).resolve().is_relative_to(PROJECT_ROOT.resolve())
    except (OSError, ValueError):
        return False


def plan_required_installs(
    probe: dict[str, dict],
    nvidia: dict[str, str | float | None],
    args: argparse.Namespace,
) -> dict[str, bool]:
    """计算必需依赖中哪些需要安装；已装的跳过。"""
    torch_info = probe.get("torch", {})
    gpu_present = bool(nvidia.get("name"))
    need_cuda = gpu_present and not bool(getattr(args, "cpu", False))
    torch_ready = bool(torch_info.get("available")) and (
        bool(torch_info.get("cuda_available")) or not need_cuda
    )
    return {
        "torch": not torch_ready,
        "pyyaml": not package_available(probe, "yaml"),
        "project": not (package_available(probe, "rwkv_easy") and _project_imports_from_root(probe)),
    }


def print_dependency_report(probe: dict[str, dict]) -> None:
    rows = [
        ("torch", "PyTorch"),
        ("yaml", "PyYAML"),
        ("rwkv_easy", "RWKV-EASY"),
        ("rwkvfla", "rwkv-fla"),
        ("deepspeed", "DeepSpeed"),
        ("safetensors", "safetensors"),
        ("triton", "Triton"),
    ]
    say("依赖检查（已安装的会跳过，不重复安装）")
    for key, label in rows:
        info = probe.get(key, {})
        if info.get("available"):
            version = info.get("version") or ""
            detail = ""
            if key == "torch":
                cuda = info.get("cuda")
                available = info.get("cuda_available")
                detail = (
                    f"CUDA {cuda} 可用" if available and cuda else
                    "CUDA 不可用" if cuda else "CPU 版"
                )
                if info.get("gpu_name"):
                    detail = f"{info['gpu_name']} / {detail}"
            elif key == "rwkv_easy":
                file = info.get("file")
                if file:
                    detail = _short_path(file)
            print(
                color(f"  ✓ {label:<12}", GREEN)
                + f"{version}"
                + (f"  ({detail})" if detail else ""),
                flush=True,
            )
        else:
            print(color(f"  ✗ {label:<12} 未安装", YELLOW), flush=True)


def print_install_plan(plan: dict[str, bool]) -> None:
    labels = {"torch": "PyTorch", "pyyaml": "PyYAML", "project": "RWKV-EASY（editable）"}
    missing = [label for key, label in labels.items() if plan.get(key)]
    if missing:
        say("需要安装：" + " / ".join(missing))
    else:
        say("必需依赖已齐全，跳过安装。")


def _short_path(path: str, limit: int = 64) -> str:
    return path if len(path) <= limit else "…" + path[-(limit - 1) :]


# --------------------------------------------------------------------------- #
# venv 复用主环境：不复制库，用 system-site-packages / .pth 让 venv 看见
# --------------------------------------------------------------------------- #


def venv_uses_system_site_packages(venv_root: Path) -> bool:
    cfg = venv_root / "pyvenv.cfg"
    if not cfg.exists():
        return False
    try:
        text = cfg.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip().lower() == "include-system-site-packages":
                return value.strip().lower() == "true"
    return False


def venv_site_packages(python: str | Path) -> Path | None:
    code = "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    try:
        result = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = (result.stdout or "").strip()
    return Path(value) if value else None


def main_site_packages() -> Path | None:
    try:
        return Path(sysconfig.get_paths()["purelib"])
    except KeyError:
        return None


def write_system_site_link(venv_site: Path, main_site: Path) -> Path | None:
    """给 venv 写一个 .pth，让它能 import 主环境已装的库。"""
    if not venv_site.is_dir():
        return None
    target = venv_site / "_rwkv_easy_system_site.pth"
    try:
        target.write_text(str(main_site) + "\n", encoding="utf-8")
    except OSError:
        return None
    return target


def ensure_venv_reuses_system(venv_python: Path, venv_root: Path) -> None:
    """已有 venv 缺库时，让它复用主环境 site-packages，而不是重装 torch。"""
    venv_probe = probe_environment(venv_python)
    if package_available(venv_probe, "torch"):
        return  # venv 自己已经有 torch，保持隔离
    main_probe = probe_environment(sys.executable)
    if not package_available(main_probe, "torch"):
        return
    venv_version = str(venv_probe.get("python", {}).get("version", ""))
    main_version = str(main_probe.get("python", {}).get("version", ""))
    if venv_version.split(".")[:2] != main_version.split(".")[:2]:
        warn(
            f"venv Python {venv_version or '?'} 与主环境 {main_version or '?'} "
            "主版本不同，跳过 site-packages 复用。"
        )
        return
    venv_site = venv_site_packages(venv_python)
    main_site = main_site_packages()
    if venv_site is None or main_site is None:
        return
    link = write_system_site_link(venv_site, main_site)
    if link is not None:
        say(f"已让 .venv 复用主环境 site-packages：{main_site}")
        say("不会复制库，也不会重装主环境已有的 torch / PyYAML。")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV-EASY 一键安装 + 交互式训练向导")
    parser.add_argument("--no-venv", action="store_true", help="不创建 .venv")
    parser.add_argument(
        "--isolated-venv",
        action="store_true",
        help=".venv 不复用主环境已装的库（默认复用，避免重装 torch）",
    )
    parser.add_argument("--skip-install", action="store_true", help="跳过安装，直接进向导")
    parser.add_argument("--install-only", action="store_true", help="只安装，不进向导")
    parser.add_argument("--cpu", action="store_true", help="强制 CPU 版 PyTorch")
    parser.add_argument("--cuda-tag", default=None, help="指定 PyTorch wheel 标签，如 cu124")
    parser.add_argument("--reinstall", action="store_true", help="已有 torch 也重新安装")
    parser.add_argument("--yes", action="store_true", help="自动确认可选依赖")
    parser.add_argument("--non-interactive", action="store_true", help="向导使用分行问答")
    parser.add_argument("--no-train", action="store_true", help="只生成配置，不训练")
    parser.add_argument(
        "--infctx-parallel-backward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="DDP + infctx 是否逐块反传（默认 true；仅 LoRA）",
    )
    parser.add_argument("-o", "--output", default=None, help="配置文件输出路径")
    return parser.parse_args(argv)


def venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def ensure_venv(args: argparse.Namespace) -> Path | None:
    """返回应该使用的 Python；None 表示直接用当前进程。"""
    if args.no_venv or in_virtualenv():
        return None
    target = PROJECT_ROOT / ".venv"
    python = venv_python(PROJECT_ROOT)
    if not python.exists():
        say(f"创建虚拟环境：{target}")
        try:
            # 默认让 venv 直接看见主环境的 site-packages：不复制、不重装 torch。
            venv.EnvBuilder(
                with_pip=True,
                clear=False,
                system_site_packages=not args.isolated_venv,
            ).create(target)
        except Exception as exc:
            warn(f"创建 .venv 失败（{exc}），改用当前 Python 环境。")
            return None
    if not python.exists():
        warn("没有找到 .venv 里的 Python，改用当前环境。")
        return None
    if args.isolated_venv:
        if venv_uses_system_site_packages(target):
            warn(
                "已有 .venv 启用了 system-site-packages；如需完全隔离，"
                "请删除 .venv 后重新运行。"
            )
    else:
        ensure_venv_reuses_system(python, target)
    if args.skip_install:
        # --skip-install 表示“环境已经就绪”。如果 .venv 是半成品（比如只建了目录、
        # 没有装 torch），继续用它只会在启动向导时崩，直接回退到当前解释器。
        probe = probe_environment(python)
        if not package_available(probe, "torch"):
            warn("检测到 .venv 不完整（没有可导入的 torch）；--skip-install 改用当前 Python。")
            return None
    return python


def nvidia_info() -> dict[str, str | float | None]:
    info: dict[str, str | float | None] = {
        "name": None,
        "memory_gib": None,
        "driver": None,
        "cuda": None,
    }
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return info
    if query:
        parts = [part.strip() for part in query.splitlines()[0].split(",")]
        if parts:
            info["name"] = parts[0]
        if len(parts) >= 2:
            match = re.search(r"([\d.]+)", parts[1])
            if match:
                info["memory_gib"] = float(match.group(1)) / 1024.0
        if len(parts) >= 3:
            info["driver"] = parts[2]
    try:
        text = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15, check=False
        ).stdout
        match = re.search(r"CUDA Version:\s*([\d.]+)", text)
        if match:
            info["cuda"] = match.group(1)
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def torch_status() -> tuple[str | None, str | None, bool]:
    try:
        import torch
    except Exception:
        return None, None, False
    return (
        getattr(torch, "__version__", None),
        getattr(torch.version, "cuda", None),
        bool(torch.cuda.is_available()),
    )


def torch_gpu_info() -> dict[str, str | float | None]:
    """torch 视角的 GPU 信息，用于 nvidia-smi 不可用的虚拟化环境。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "memory_gib": props.total_memory / 1024**3,
            "cuda": getattr(torch.version, "cuda", None),
        }
    except Exception:
        return {}


def cuda_tags(version: str | None) -> list[str]:
    supported = [
        ((13, 0), "cu130"),
        ((12, 8), "cu128"),
        ((12, 6), "cu126"),
        ((12, 4), "cu124"),
        ((12, 1), "cu121"),
        ((11, 8), "cu118"),
    ]
    if version:
        match = re.match(r"^(\d+)\.(\d+)", version.strip())
        if match:
            target = (int(match.group(1)), int(match.group(2)))
            return [tag for (ver, tag) in supported if ver <= target] or ["cu118"]
    return [tag for _, tag in supported]


def pip_install(args: list[str]) -> int:
    return run([sys.executable, "-m", "pip", "install", *args])


def install_torch(
    nvidia: dict[str, str | float | None],
    args: argparse.Namespace,
    force: bool,
) -> int:
    version, cuda, available = torch_status()
    has_gpu = nvidia.get("name") is not None
    if version and not args.reinstall and not force:
        if args.cpu or not has_gpu or available:
            say(f"已检测到 PyTorch {version}（CUDA {cuda or '—'}），跳过安装。")
            return 0
        warn(f"当前 PyTorch {version} 没有可用的 CUDA，将安装 CUDA 版。")
    if args.cpu or not has_gpu:
        say("安装 CPU 版 PyTorch ...")
        cmd = ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"]
        if version:
            cmd.append("--upgrade")
        return pip_install(cmd)

    tags = [args.cuda_tag] if args.cuda_tag else cuda_tags(str(nvidia.get("cuda") or ""))
    force_flag = ["--force-reinstall"] if version or args.reinstall else []
    for tag in dict.fromkeys(tags):
        say(f"安装 PyTorch（{tag}）...")
        code = pip_install(["torch", "--index-url", f"https://download.pytorch.org/whl/{tag}", *force_flag])
        if code == 0:
            return 0
        warn(f"{tag} 安装失败，尝试下一个兼容版本 ...")
    warn("CUDA 版 PyTorch 安装失败，回退到 CPU 版。")
    return pip_install(["torch", "--upgrade"])


def install_project() -> int:
    say("安装 RWKV-EASY（可编辑模式，不重装 torch 等依赖）...")
    # 依赖已在上一步逐个检查；--no-deps 防止 pip 把 torch 当成缺失依赖重装。
    return pip_install(["-e", str(PROJECT_ROOT), "--no-deps"])


def install_optional_fla(nvidia: dict[str, str | float | None], args: argparse.Namespace) -> None:
    if not nvidia.get("name") or not sys.platform.startswith("linux"):
        return
    version, _, _ = torch_status()
    if version:
        match = re.match(r"^(\d+)\.(\d+)", version)
        if match and (int(match.group(1)), int(match.group(2))) < (2, 4):
            warn(f"当前 torch {version} < 2.4，rwkv-fla 需要更新的 torch；跳过。")
            return
    try:
        import rwkvfla  # noqa: F401

        say("rwkv-fla 已安装。")
        return
    except Exception:
        pass
    if not args.yes:
        try:
            answer = input("检测到 CUDA，是否安装 rwkv-fla（Triton 加速，推荐）？[Y/n] ").strip().lower()
        except EOFError:
            answer = "n"
        if answer not in ("", "y", "yes"):
            return
    say("安装 rwkv-fla（--no-deps，不重装 torch/triton）...")
    code = pip_install(["rwkv-fla", "--no-deps"])
    if code != 0:
        warn("rwkv-fla 安装失败；向导里仍可选择 native 后端。")
    else:
        warn("如果 rwkv-fla 导入失败，通常是 triton 版本不匹配；可先跳过它用 native。")


def _wizard_command(args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-m", "rwkv_easy", "wizard"]
    if args.non_interactive:
        cmd.append("--non-interactive")
    if args.no_train:
        cmd.append("--no-train")
    if args.infctx_parallel_backward is not None:
        cmd.append(
            "--infctx-parallel-backward"
            if args.infctx_parallel_backward
            else "--no-infctx-parallel-backward"
        )
    if args.output:
        cmd.extend(["-o", args.output])
    return cmd


def run_ready(args: argparse.Namespace) -> int:
    print()
    print(color("=" * 68, CYAN))
    print(color(" RWKV-EASY 一键安装 / 训练向导", "1;36"))
    print(color("=" * 68, CYAN))
    print(" 这次会做 3 件事：")
    print("   1. 检测 Python / GPU / CUDA")
    print("   2. 安装 PyTorch 和本项目（已经装好会自动跳过）")
    print("   3. 打开配置向导：选模板 → 选模型/数据 → 开始训练")
    print()

    say("[1/3] 检测环境")
    say(f"Python：{sys.executable}（{sys.version.split()[0]}）")
    nvidia = nvidia_info()
    probe = probe_environment()
    torch_info = probe.get("torch", {})
    for key in ("name", "memory_gib", "cuda"):
        if not nvidia.get(key):
            value = torch_info.get("gpu_name" if key == "name" else key)
            if value:
                nvidia[key] = value
    if not probe:
        # 探测子进程异常时，退回当前进程的 torch 信息（nvidi-smi 不可用的虚拟化环境）。
        torch_gpu = torch_gpu_info()
        for key in ("name", "memory_gib", "cuda"):
            if not nvidia.get(key) and torch_gpu.get(key):
                nvidia[key] = torch_gpu[key]
    if nvidia.get("name"):
        driver = f" / 驱动 {nvidia.get('driver')}" if nvidia.get("driver") else ""
        say(
            f"GPU：{nvidia.get('name')}（{nvidia.get('memory_gib') or 0:.1f} GiB）"
            f"{driver} / CUDA {nvidia.get('cuda') or '?'}"
        )
    else:
        warn("未检测到 NVIDIA GPU；只能安装 CPU 版 PyTorch，训练会很慢。")
        if not args.skip_install and not args.cpu and not args.yes and sys.stdin.isatty():
            try:
                answer = input("仍要继续安装 CPU 版吗？[Y/n] ").strip().lower()
            except EOFError:
                answer = "n"
            if answer not in ("", "y", "yes"):
                warn("已取消。拿到 NVIDIA GPU 后重新运行 python quickstart.py 即可。")
                return 1

    print_dependency_report(probe)

    if args.skip_install:
        if not package_available(probe, "torch"):
            return die(
                "--skip-install 需要当前 Python 已经安装 torch；"
                "去掉这个参数让 quickstart 自动安装，或先手动 pip install torch。"
            )
        if not package_available(probe, "rwkv_easy"):
            return die(
                "--skip-install 需要当前环境能 import rwkv_easy；"
                "在项目根目录运行，或先 pip install -e . --no-deps。"
            )
        say("[2/3] --skip-install：跳过安装")
    else:
        plan = plan_required_installs(probe, nvidia, args)
        print_install_plan(plan)
        say("[2/3] 安装缺失依赖")
        if plan["torch"] or args.reinstall:
            code = install_torch(nvidia, args, force=bool(plan["torch"] or args.reinstall))
            if code != 0:
                die(
                    "PyTorch 安装失败。请检查网络，或手动执行：\n"
                    f"  {sys.executable} -m pip install torch --index-url "
                    "https://download.pytorch.org/whl/cu124\n"
                    f"  {sys.executable} -m pip install -e {PROJECT_ROOT} --no-deps"
                )
                return code
        if plan["pyyaml"]:
            code = pip_install(["pyyaml"])
            if code != 0:
                return die("PyYAML 安装失败；请检查网络或 pip 配置。")
        if plan["project"]:
            code = install_project()
            if code != 0:
                return die("项目安装失败；请检查上面的 pip 输出。")
        install_optional_fla(nvidia, args)

    if args.install_only:
        print()
        print(color("=" * 68, CYAN))
        print(color(" 环境已就绪" if args.skip_install else " 安装完成", "1;32"))
        print(color("=" * 68, CYAN))
        print(" 下一步：")
        print(f"   cd {PROJECT_ROOT}")
        print(f"   {sys.executable} -m rwkv_easy wizard")
        return 0

    print()
    say("[3/3] 启动交互式配置向导")
    say("向导里：←/→ 切换选项，空格勾选模块/层，Enter 进入下一步；q 退出。")
    try:
        return int(subprocess.call(_wizard_command(args)))
    except KeyboardInterrupt:
        return 130


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if os.environ.get(STAGE_ENV) == "ready":
        return run_ready(args)

    if sys.version_info < (3, 10):  # noqa: UP036 - 引导脚本可能被旧 Python 直接执行
        return die(f"需要 Python 3.10+，当前是 {sys.version.split()[0]}")

    python = ensure_venv(args)
    if python is None:
        return run_ready(args)

    env = dict(os.environ)
    env[STAGE_ENV] = "ready"
    say(f"切换到虚拟环境：{python}")
    try:
        return int(subprocess.call([str(python), str(Path(__file__).resolve()), *argv], env=env))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
