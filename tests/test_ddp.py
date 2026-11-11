"""两进程 DDP 冒烟测试（CPU gloo）。

走项目自己的 ``launcher.relaunch``，因此也顺便验证了多卡启动逻辑。
设置 ``RWKV_EASY_SKIP_DDP=1`` 可跳过。
"""

from __future__ import annotations

import os

import pytest
import torch
from conftest import build_config

from rwkv_easy.launcher import relaunch


@pytest.mark.skipif(os.environ.get("RWKV_EASY_SKIP_DDP") == "1", reason="用户要求跳过 DDP 测试")
@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed 不可用")
def test_ddp_two_processes(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {
            "train": {"max_steps": 4, "merge_at_end": False},
            "dist": {"mode": "ddp", "num_gpus": 2},
            "data": {"num_workers": 0, "cache": False},
        },
    )
    config_path = tmp_path / "ddp.yaml"
    cfg.save_yaml(config_path)
    run_dir = tmp_path / "ddp-run"

    code = relaunch(
        2,
        config_path=config_path,
        overrides=[],
        resume=None,
        run_dir=str(run_dir),
        device="cpu",
        backend=None,
        log_level=None,
    )
    assert code == 0
    assert (run_dir / "adapter-step0000004.pth").exists()
    log_text = (run_dir / "train.log").read_text(encoding="utf-8")
    assert "使用 DDP" in log_text
