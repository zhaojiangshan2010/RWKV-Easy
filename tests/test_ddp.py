"""两进程 DDP 冒烟测试（CPU gloo）。

走项目自己的 ``launcher.relaunch``，因此也顺便验证了多卡启动逻辑。
设置 ``RWKV_EASY_SKIP_DDP=1`` 可跳过。
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from conftest import build_config

from rwkv_easy.launcher import relaunch
from rwkv_easy.train.backends import DistInfo, build_backend


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


class _DummyBackendModel(nn.Module):
    """最小模型：forward_hidden / forward_chunk_hidden 都只依赖一个参数。"""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(2))

    def forward_hidden(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None, need_state: bool = False
    ):
        return (x * self.w).sum(), None

    def forward_chunk_hidden(self, x: torch.Tensor, *args, **kwargs):
        return (x * self.w).sum(), None, None, None


def _assert_synced(rank: int, world: int, grad: torch.Tensor, expected: list[float]) -> None:
    gathered = [torch.zeros_like(grad) for _ in range(world)]
    dist.all_gather(gathered, grad.detach().clone())
    want = torch.tensor(expected)
    for got in gathered:
        assert torch.allclose(got.cpu(), want), f"rank {rank} 梯度未同步：{got.tolist()}"


def _ddp_backend_worker(rank: int, world: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        model = _DummyBackendModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        cfg = SimpleNamespace(
            train=SimpleNamespace(grad_accum=1, grad_clip=0.0),
            dist=SimpleNamespace(find_unused_parameters=False),
        )
        backend = build_backend("ddp", model, optimizer, None, DistInfo(rank, rank, world), cfg)
        x = torch.tensor([1.0, 0.0]) if rank == 0 else torch.tensor([0.0, 1.0])

        # 普通 DDP：两次不同输入，梯度应平均成 [0.5, 0.5]。
        loss, _ = backend.forward_hidden(x)
        backend.backward(loss)
        assert model.w.grad is not None
        _assert_synced(rank, world, model.w.grad, [0.5, 0.5])

        # infctx 逐块：两个 chunk，只有最后一块触发 all-reduce。
        backend.zero_grad()
        for index in range(2):
            with backend.chunk_sync_context(index == 1):
                loss = backend.forward_chunk_hidden(x)[0]
                backend.backward(loss)
        assert model.w.grad is not None
        _assert_synced(rank, world, model.w.grad, [1.0, 1.0])

        # 梯度累积 2 个 micro-batch × 2 个 chunk：外层 no_sync 与内层 chunk
        # no_sync 嵌套，只有最后一个 micro-batch 的最后一个 chunk 同步。
        backend.zero_grad()
        backend.grad_accum = 2
        for micro in range(2):
            with backend.sync_context(micro):
                for index in range(2):
                    with backend.chunk_sync_context(index == 1):
                        loss = backend.forward_chunk_hidden(x)[0]
                        backend.backward(loss)
        assert model.w.grad is not None
        _assert_synced(rank, world, model.w.grad, [2.0, 2.0])

        # 延迟路径：原始模型多次 forward、一次 backward，手动 all-reduce。
        backend.zero_grad()
        backend.enable_manual_grad_sync()
        outs = [model.forward_chunk_hidden(x)[0] for _ in range(2)]
        backend.backward(outs[0] + outs[1])
        assert model.w.grad is not None
        _assert_synced(rank, world, model.w.grad, [1.0, 1.0])
    finally:
        dist.barrier()
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.skipif(
    os.environ.get("RWKV_EASY_SKIP_DDP") == "1", reason="用户要求跳过 DDP 测试"
)
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed 不可用")
def test_ddp_backend_syncs_gradients_and_chunks():
    mp.spawn(_ddp_backend_worker, args=(2, _free_port()), nprocs=2, join=True)


@pytest.mark.parametrize(
    ("parallel", "expected"),
    [(True, "逐块反传"), (False, "最后统一反传")],
)
@pytest.mark.skipif(
    os.environ.get("RWKV_EASY_SKIP_DDP") == "1", reason="用户要求跳过 DDP 测试"
)
@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed 不可用")
def test_ddp_infctx_backward_path(tmp_path, tiny_checkpoint, tiny_sharegpt, parallel, expected):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {
            "train": {
                "infctx": True,
                "ctx_len": 64,
                "chunk_ctx": 32,
                "micro_batch_size": 2,
                "grad_accum": 1,
                "max_steps": 4,
                "merge_at_end": False,
                "grad_checkpoint": False,
                "infctx_parallel_backward": parallel,
            },
            "dist": {"mode": "ddp", "num_gpus": 2},
            "data": {"num_workers": 0, "cache": False},
        },
    )
    config_path = tmp_path / f"ddp-infctx-{parallel}.yaml"
    cfg.save_yaml(config_path)
    run_dir = tmp_path / f"ddp-infctx-run-{parallel}"

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
    assert expected in log_text
