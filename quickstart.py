#!/usr/bin/env python3
"""RWKV-EASY 一键安装 + 交互式训练向导。

用法（项目根目录）::

    python quickstart.py

它会：

1. 检测 Python / NVIDIA 驱动 / CUDA 版本；
2. 需要时创建 ``.venv``，按 CUDA 版本安装对应 PyTorch；
3. 安装本项目与可选的 rwkv-fla；
4. 进入交互式向导：选择 LoRA / State / infctx、模型、数据集、层与模块，
   保存 YAML 并直接开始训练。

常用参数::

    --no-venv        不创建 .venv，直接装到当前 Python 环境
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
import os
import re
import shlex
import subprocess
import sys
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV-EASY 一键安装 + 交互式训练向导")
    parser.add_argument("--no-venv", action="store_true", help="不创建 .venv")
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
            venv.EnvBuilder(with_pip=True, clear=False).create(target)
        except Exception as exc:
            warn(f"创建 .venv 失败（{exc}），改用当前 Python 环境。")
            return None
    if not python.exists():
        warn("没有找到 .venv 里的 Python，改用当前环境。")
        return None
    if args.skip_install:
        # --skip-install 表示“环境已经就绪”。如果 .venv 是半成品（比如只建了目录、
        # 没有装 torch），继续用它只会在启动向导时崩，直接回退到当前解释器。
        try:
            probe = subprocess.run(
                [str(python), "-c", "import torch"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            probe = None
        if probe is None or probe.returncode != 0:
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
    say("安装 RWKV-EASY（可编辑模式）...")
    return pip_install(["-e", str(PROJECT_ROOT)])


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
    say("安装 rwkv-fla（失败不影响 native 后端）...")
    code = pip_install(["rwkv-fla"])
    if code != 0:
        warn("rwkv-fla 安装失败；向导里仍可选择 native 后端。")


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

    if args.skip_install:
        version, _, _ = torch_status()
        if version is None:
            return die(
                "--skip-install 需要当前 Python 已经安装 torch；"
                "去掉这个参数让 quickstart 自动安装，或先手动 pip install torch。"
            )

    if not args.skip_install:
        say("[2/3] 安装 PyTorch 和项目")
        version, cuda, available = torch_status()
        need_torch = version is None or (nvidia.get("name") and not available and not args.cpu)
        code = install_torch(nvidia, args, force=bool(need_torch or args.reinstall))
        if code != 0:
            die(
                "PyTorch 安装失败。请检查网络，或手动执行：\n"
                f"  {sys.executable} -m pip install torch --index-url "
                "https://download.pytorch.org/whl/cu124\n"
                f"  {sys.executable} -m pip install -e {PROJECT_ROOT}"
            )
            return code
        code = install_project()
        if code != 0:
            return die("项目安装失败；请检查上面的 pip 输出。")
        install_optional_fla(nvidia, args)
    else:
        say("[2/3] --skip-install：跳过安装")

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
