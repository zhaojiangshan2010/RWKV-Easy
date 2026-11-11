"""训练编排：配置 -> 设备 -> 数据 -> 模型 -> 方法注入 -> Trainer。

这个模块负责“把所有零件装起来”，也是 ``rwkv-easy train`` 与 ``selftest`` 的共同入口。
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..config import Config
from ..data.dataset import build_datasets
from ..lora import inject_lora
from ..models.checkpoint import load_model_from_checkpoint, random_model
from ..models.decay import calibrate_decay
from ..models.rwkv7 import RWKV7Config
from ..runtime import (
    normalize_precision,
    now_stamp,
    resolve_device,
    seed_everything,
    setup_logger,
    storage_dtype,
    validate_precision,
)
from ..state_tuning import attach_state_tuning
from .backends import broadcast_object, destroy_distributed, init_distributed
from .loop import Trainer, TrainResult

__all__ = ["prepare_run_dir", "run_training"]


def prepare_run_dir(cfg: Config, resume: str | Path | None) -> Path:
    if resume:
        path = Path(resume)
        run_dir = path if path.is_dir() else path.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    suffix = cfg.train.method + ("-infctx" if cfg.train.infctx else "")
    run_dir = Path(cfg.run.output_dir) / f"{cfg.run.name}-{suffix}-{now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _model_dtype(cfg: Config) -> torch.dtype:
    precision = normalize_precision(cfg.train.precision)
    if cfg.model.dtype:
        override = cfg.model.dtype.lower()
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[override]
    return storage_dtype(precision)


def run_training(
    cfg: Config,
    *,
    resume: str | Path | None = None,
    random_init: bool = False,
    run_dir: str | Path | None = None,
    log_level: str | None = None,
) -> TrainResult:
    """执行一次完整训练。"""
    cfg.resolve()
    device = resolve_device(cfg.train.device)
    validate_precision(cfg.train.precision, device)
    seed_everything(cfg.run.seed, cfg.run.deterministic)

    dist = init_distributed(device)
    try:
        # ---- run 目录：由 rank0 决定并广播，保证多卡一致 ----
        if run_dir is None:
            run_dir_str = str(prepare_run_dir(cfg, resume)) if dist.is_main else ""
        else:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            run_dir_str = str(run_dir)
        run_dir_str = broadcast_object(run_dir_str, src=0)
        run_dir_path = Path(run_dir_str)

        log = setup_logger(
            log_file=(run_dir_path / "train.log") if dist.is_main else None,
            rank=dist.rank,
            level=log_level or cfg.log.level,
        )
        if dist.is_main:
            log.info("运行目录：%s", run_dir_path)
            cfg.save_yaml(run_dir_path / "config.resolved.yaml", header="# RWKV-EASY 解析后的完整配置")

        # ---- 数据 ----
        data = build_datasets(cfg, log if dist.is_main else None)

        # ---- 模型 ----
        torch.manual_seed(cfg.run.seed)
        if random_init or cfg.model.random_init or not cfg.model.path:
            if not (cfg.model.n_layer and cfg.model.n_embd and cfg.model.vocab_size):
                raise ValueError("随机初始化模型需要 model.n_layer / n_embd / vocab_size")
            model_cfg = RWKV7Config(
                vocab_size=int(cfg.model.vocab_size),
                n_layer=int(cfg.model.n_layer),
                n_embd=int(cfg.model.n_embd),
                head_size=int(cfg.model.head_size or 64),
                head_size_divisor=int(cfg.model.head_size_divisor),
            )
            model = random_model(
                model_cfg,
                backend=cfg.model.backend,
                decay="clampw",
                wkv_chunk=cfg.model.wkv_chunk,
                grad_checkpoint=cfg.train.grad_checkpoint,
                seed=cfg.run.seed,
            )
            base_signature = "random"
        else:
            model, model_cfg, base_signature = load_model_from_checkpoint(
                cfg.model.path,
                backend=cfg.model.backend,
                decay="clampw" if cfg.model.decay == "auto" else cfg.model.decay,
                wkv_chunk=cfg.model.wkv_chunk,
                grad_checkpoint=cfg.train.grad_checkpoint,
                n_layer=cfg.model.n_layer,
                n_embd=cfg.model.n_embd,
                vocab_size=cfg.model.vocab_size,
                head_size=cfg.model.head_size,
                head_size_divisor=cfg.model.head_size_divisor,
                log=log if dist.is_main else None,
            )

        # ---- decay 自动校准（需要 tokenizer + device） ----
        # 只在 rank0 上做，然后广播结果；校准期间保持 fp32 以获得干净的比较。
        if cfg.model.decay == "auto":
            model.to(device)
            if dist.is_main:
                calibration = calibrate_decay(
                    model,
                    data.tokenizer,
                    device,
                    cache_key=f"{base_signature}-{model_cfg.n_layer}x{model_cfg.n_embd}",
                    cache_dir=run_dir_path / ".cache",
                    log=log,
                )
                mode: str | None = calibration.mode
            else:
                mode = None
            mode = broadcast_object(mode, src=0) or "clampw"
            model.set_decay_mode(mode)
            if dist.is_main:
                log.info("decay 选择：%s", model.decay_mode)
        else:
            model.set_decay_mode(cfg.model.decay)

        # ---- 精度 ----
        dtype = _model_dtype(cfg)
        if dtype != torch.float32:
            model = model.to(dtype=dtype)
            if dist.is_main:
                log.info("模型参数 dtype -> %s", dtype)

        # ---- 方法：LoRA / State Tuning ----
        # 重新播种，保证多卡上 LoRA 的随机初始化完全一致。
        seed_everything(cfg.run.seed, cfg.run.deterministic)
        if cfg.train.method == "lora":
            inject_lora(
                model,
                target_modules=list(cfg.lora.target_modules),
                rank=int(cfg.lora.rank),
                alpha=int(cfg.lora.alpha),
                dropout=float(cfg.lora.dropout),
                target_regex=cfg.lora.target_regex,
                log=log if dist.is_main else None,
            )
        elif cfg.train.method == "state":
            attach_state_tuning(
                model, init_std=float(cfg.state.init_std), log=log if dist.is_main else None
            )
        else:  # pragma: no cover - 配置校验已拦截
            raise ValueError(f"未知训练方法：{cfg.train.method!r}")

        model = model.to(device)

        # ---- 训练 ----
        trainer = Trainer(cfg, model, model_cfg, data, dist, run_dir_path, base_signature)
        if dist.is_main:
            log.info("基座签名：%s", base_signature)
        return trainer.fit(resume=resume)
    finally:
        destroy_distributed()
