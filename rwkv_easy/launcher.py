"""多进程启动器。

小白用户只需要 ``rwkv-easy train -c config.yaml``；如果机器上有 N 张卡，
这里会自动拉起 N 个训练进程，并把 ``RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT``
设好，训练代码用标准 ``init_process_group("env://")`` 初始化。

为什么不用 ``torchrun``？
--------------------------
``torchrun`` 依赖 c10d rendezvous store，在部分 Windows / 精简版 PyTorch 上
会因为 ``libuv`` 缺失直接崩掉（本项目在 CPU-only Windows 上就遇到）。直接用
``subprocess`` 设置环境变量可以完全绕开这一层，行为在 Linux/Windows 上一致，
也不影响用户自己用 ``torchrun`` 启动。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import torch

from .config import Config

__all__ = ["is_launched_by_runner", "relaunch", "resolve_num_gpus"]


def is_launched_by_runner() -> bool:
    return os.environ.get("RWKV_EASY_LAUNCHED") == "1" or int(os.environ.get("WORLD_SIZE", "1")) > 1


def resolve_num_gpus(cfg: Config) -> int:
    """决定用几个进程。"""
    if cfg.dist.mode == "single":
        return 1
    if isinstance(cfg.dist.num_gpus, int):
        n = cfg.dist.num_gpus
    else:
        n = torch.cuda.device_count() if torch.cuda.is_available() else 1
    return max(1, int(n))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def relaunch(
    nproc: int,
    *,
    config_path: str | Path,
    overrides: list[str],
    resume: str | None,
    run_dir: str | None,
    device: str | None,
    backend: str | None,
    log_level: str | None,
    random_init: bool = False,
) -> int:
    """拉起 ``nproc`` 个训练进程，返回第一个非零退出码（全部成功则返回 0）。"""
    child_args: list[str] = ["-c", str(config_path)]
    for item in overrides or []:
        child_args += ["--set", item]
    if resume:
        child_args += ["--resume", str(resume)]
    if run_dir:
        child_args += ["--run-dir", str(run_dir)]
    if device:
        child_args += ["--device", device]
    if backend:
        child_args += ["--backend", backend]
    if log_level:
        child_args += ["--log-level", log_level]
    if random_init:
        child_args += ["--random-init"]

    master_port = _free_port()
    base_env = dict(os.environ)
    base_env.update(
        {
            "RWKV_EASY_LAUNCHED": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": str(nproc),
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "OMP_NUM_THREADS": "1",
        }
    )

    print(f"[rwkv-easy] 启动 {nproc} 个进程（MASTER_PORT={master_port}）……")
    processes: list[subprocess.Popen] = []
    try:
        for rank in range(nproc):
            env = dict(base_env)
            env["RANK"] = str(rank)
            env["LOCAL_RANK"] = str(rank)
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-m", "rwkv_easy.train_entry", *child_args],
                    env=env,
                )
            )
        codes = [process.wait() for process in processes]
    except KeyboardInterrupt:  # pragma: no cover
        for process in processes:
            if process.poll() is None:
                process.terminate()
        print("\n[rwkv-easy] 已中断")
        return 130

    for code in codes:
        if code != 0:
            return int(code)
    return 0
