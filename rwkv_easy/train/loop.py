"""训练主循环：普通训练与 infctx 分块训练。

关键点
------

* **loss 归一化**：按“有效 token 数”求平均（忽略 ``-100``），infctx 下用
  token 数加权合并各块，结果与整段前向完全一致。
* **梯度累积**：DDP 下用 ``no_sync()`` 跳过中间 micro-step 的 all-reduce；
  DeepSpeed 下由主循环统一控制 step 时机。
* **infctx**：块间传递 :class:`ModelState`，默认 detach（截断 BPTT）；
  每块用 ``torch.utils.checkpoint`` 重算激活，显存与块长成正比，而不是与
  ``ctx_len`` 成正比。
* **保存**：只保存可训练参数（LoRA / time_state），文件很小；结束时可选
  自动导出合并后的完整权重。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_fn
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import BuiltData, make_dataloader
from ..lora import load_lora_state_dict
from ..models.rwkv7 import RWKV7, RWKV7Config
from ..models.state import ModelState
from ..runtime import (
    EMALoss,
    MetricsLogger,
    autocast_context,
    count_parameters,
    device_name,
    format_count,
    format_seconds,
    get_logger,
    is_main_process,
    make_grad_scaler,
    normalize_precision,
    peak_memory_gib,
    reset_peak_memory,
)
from ..state_tuning import load_state_state_dict
from .backends import (
    DeepSpeedBackend,
    DistInfo,
    build_backend,
)
from .checkpoint import (
    adapter_filename,
    collect_weights,
    export_merged,
    latest_pointer,
    load_trainer_state,
    prune_checkpoints,
    read_adapter,
    save_adapter,
    save_trainer_state,
    set_latest_pointer,
    trainer_state_filename,
)
from .optimizer import build_optimizer, build_scheduler, describe_optimizer

__all__ = ["L2Wrap", "TrainResult", "Trainer", "resolve_backend_kind"]

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- #
# L2Wrap：官方用于稳定 RWKV 训练的小技巧
# --------------------------------------------------------------------------- #


class L2Wrap(torch.autograd.Function):
    """把每行最大 logit 往 0 方向拉一点（只改梯度，不改 loss 值）。"""

    @staticmethod
    def forward(ctx, loss: torch.Tensor, logits: torch.Tensor, factor: float) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(logits)
        ctx.factor = float(factor)
        return loss

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        (logits,) = ctx.saved_tensors
        max_value, indices = torch.max(logits, dim=-1, keepdim=True)
        grad = torch.zeros_like(logits)
        grad.scatter_(-1, indices, max_value * ctx.factor * grad_output)
        return grad_output, grad, None


# --------------------------------------------------------------------------- #
# 结果
# --------------------------------------------------------------------------- #


@dataclass
class TrainResult:
    run_dir: Path
    steps: int
    best_loss: float | None
    last_adapter: Path | None
    merged: Path | None


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


def resolve_backend_kind(cfg: Config, dist: DistInfo, device: torch.device) -> str:
    mode = cfg.dist.mode
    if mode == "auto":
        return "ddp" if dist.world_size > 1 else "single"
    if mode == "deepspeed":
        return "deepspeed"
    if mode == "ddp":
        if dist.world_size <= 1:
            return "single"
        return "ddp"
    if mode == "single":
        if dist.world_size > 1:
            raise RuntimeError(
                "`dist.mode: single` 但检测到多个进程（WORLD_SIZE="
                f"{dist.world_size}）。请改成 auto/ddp，或只启动一个进程。"
            )
        return "single"
    raise ValueError(f"未知 dist.mode：{mode!r}")


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: RWKV7,
        model_cfg: RWKV7Config,
        data: BuiltData,
        dist: DistInfo,
        run_dir: str | Path,
        base_signature: str = "",
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.model_cfg = model_cfg
        self.data = data
        self.dist = dist
        self.run_dir = Path(run_dir)
        self.base_signature = base_signature
        self.log = get_logger("rwkv-easy")
        self.device = next(model.parameters()).device
        self.precision = normalize_precision(cfg.train.precision)
        self.grad_accum = max(1, int(cfg.train.grad_accum))

        # 数据
        self.train_loader = self._make_loader(data.train, shuffle=True)
        self.val_loader = (
            self._make_loader(data.val, shuffle=False) if data.val is not None else None
        )
        self.micro_per_epoch = len(self.train_loader)
        if self.micro_per_epoch == 0:
            raise RuntimeError(
                "训练集太小：一个 epoch 的 micro-batch 数为 0。\n"
                f"请减小 train.micro_batch_size（当前 {cfg.train.micro_batch_size}）"
                "或增加数据量。"
            )
        self.steps_per_epoch = math.ceil(self.micro_per_epoch / self.grad_accum)
        if cfg.train.max_steps is not None:
            # 与 HuggingFace Trainer 一致：给了 max_steps 就以它为准，epochs 只用来
            # 决定“最多循环几轮”；否则会出现“日志说 20 步、实际 1 个 epoch 就停”的困惑。
            self.total_steps = int(cfg.train.max_steps)
            self.max_epochs = max(1, math.ceil(self.total_steps / self.steps_per_epoch))
        else:
            self.total_steps = self.steps_per_epoch * int(cfg.train.epochs)
            self.max_epochs = int(cfg.train.epochs)

        # 优化器 / 调度器
        self.optimizer = build_optimizer(model, cfg.train)
        self.scheduler = build_scheduler(self.optimizer, cfg.train, self.total_steps)

        # 后端
        self.backend_kind = resolve_backend_kind(cfg, dist, self.device)
        train_model: nn.Module = model
        if cfg.train.compile:
            try:
                train_model = torch.compile(model)  # type: ignore[assignment]
                self.log.info("已启用 torch.compile（首次 step 可能较慢）")
            except Exception as exc:  # pragma: no cover
                self.log.warning("torch.compile 失败（%s），继续使用 eager。", exc)
        if self.backend_kind == "deepspeed":
            from .deepspeed_backend import create_deepspeed_backend

            self.backend = create_deepspeed_backend(
                train_model, self.optimizer, self.scheduler, dist, cfg, self.log
            )
        else:
            self.backend = build_backend(
                self.backend_kind, train_model, self.optimizer, self.scheduler, dist, cfg, self.log
            )

        # fp16 的 GradScaler（DDP / 单卡）
        self.scaler = None
        if (
            self.precision == "fp16"
            and self.device.type == "cuda"
            and not isinstance(self.backend, DeepSpeedBackend)
        ):
            self.scaler = make_grad_scaler(self.device)
            self.backend.scaler = self.scaler

        # 进度
        self.start_step = 0
        self.start_epoch = 0
        self.step = 0
        self.best_loss: float | None = None
        self.metrics = MetricsLogger(
            self.run_dir / "metrics.jsonl" if cfg.log.jsonl else None,
            enabled=cfg.log.jsonl and is_main_process(),
        )
        self.loss_ema = EMALoss(0.98)
        self._tokens_processed = 0
        self._last_log_time = time.time()
        self._last_log_tokens = 0

    # ------------------------------------------------------------------ #
    # 数据
    # ------------------------------------------------------------------ #

    def _make_loader(self, dataset: Any, shuffle: bool) -> DataLoader:
        return make_dataloader(
            dataset,
            batch_size=self.cfg.train.micro_batch_size,
            collate_fn=self.data.collate_fn,
            shuffle=shuffle,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            drop_last=self.dist.distributed,
            distributed=self.dist.distributed,
            rank=self.dist.rank,
            world_size=self.dist.world_size,
            seed=self.cfg.run.seed,
        )

    def _set_epoch(self, epoch: int) -> None:
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    # ------------------------------------------------------------------ #
    # 计算 loss
    # ------------------------------------------------------------------ #

    def _cross_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
        vocab = logits.size(-1)
        flat_labels = labels.reshape(-1)
        count = int((flat_labels != IGNORE_INDEX).sum().item())
        if count == 0:
            return logits.sum() * 0.0, 0
        loss = F.cross_entropy(
            logits.reshape(-1, vocab),
            flat_labels,
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        loss = loss / count
        if self.cfg.train.l2wrap > 0 and self.model.training and logits.requires_grad:
            # 与官方 L2Wrap 数学等价，但不再为 logits 新建一份全量梯度：
            # 在 CE 已经算出的 grad 上原地 scatter_add 每行的 max logit。
            # 注意闭包只能捕获小的 max_value/indices；捕获 logits 会形成
            # 引用环，让上一 步的 logits 无法及时释放（每步泄漏 2 GiB）。
            factor = self.cfg.train.l2wrap / max(1, flat_labels.numel())
            max_value, indices = torch.max(logits.detach(), dim=-1, keepdim=True)

            def _l2wrap_hook(grad: torch.Tensor) -> torch.Tensor:
                grad.scatter_add_(-1, indices, max_value.to(grad.dtype) * factor)
                return grad

            logits.register_hook(_l2wrap_hook)
        return loss, count

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        if self.cfg.train.infctx:
            return self._infctx_loss(input_ids, labels, attention_mask)
        logits, _ = self.model(input_ids, attention_mask=attention_mask, need_state=False)
        return self._cross_entropy(logits, labels)

    def _infctx_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        cfg = self.cfg.train
        chunk = int(cfg.chunk_ctx)
        batch_size, total = input_ids.shape
        state = self.model.initial_state(
            batch_size,
            device=input_ids.device,
            # 与模型参数保持同一 dtype：bf16-full 下隐藏态是 bf16，
            # 混入 fp32 状态会让 Linear 的 dtype 不匹配。
            dtype=self.model.emb.weight.dtype,
        )

        total_loss: torch.Tensor | None = None
        total_tokens = 0
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            chunk_ids = input_ids[:, start:end]
            chunk_labels = labels[:, start:end]
            chunk_mask = attention_mask[:, start:end] if attention_mask is not None else None

            use_checkpoint = cfg.grad_checkpoint and self.model.training
            if use_checkpoint:
                logits, shift_att, shift_ffn, wkv = checkpoint_fn(
                    self.model.forward_chunk,
                    chunk_ids,
                    state.shift_att,
                    state.shift_ffn,
                    state.wkv,
                    chunk_mask,
                    use_reentrant=False,
                )
            else:
                logits, shift_att, shift_ffn, wkv = self.model.forward_chunk(
                    chunk_ids, state.shift_att, state.shift_ffn, state.wkv, chunk_mask
                )

            chunk_loss, chunk_tokens = self._cross_entropy(logits, chunk_labels)
            if chunk_tokens > 0:
                total_loss = chunk_loss * chunk_tokens if total_loss is None else total_loss + chunk_loss * chunk_tokens
                total_tokens += chunk_tokens

            if cfg.infctx_detach:
                state = ModelState(shift_att.detach(), shift_ffn.detach(), wkv.detach())
            else:
                state = ModelState(shift_att, shift_ffn, wkv)

        if total_loss is None:
            # 整段都没有可训练 token：返回一个可反传的 0，避免训练崩溃。
            return input_ids.float().sum() * 0.0, 0
        return total_loss / total_tokens, total_tokens

    # ------------------------------------------------------------------ #
    # 训练
    # ------------------------------------------------------------------ #

    def fit(self, resume: str | Path | None = None) -> TrainResult:
        cfg = self.cfg
        if resume is not None:
            self._resume(resume)

        self.model.train()
        self.model.to(self.device)
        if is_main_process():
            self.log.info("开始训练：%s", cfg.summary())
            self.log.info("设备：%s", device_name(self.device))
            self.log.info("优化器：%s", describe_optimizer(self.model, cfg.train))
            self.log.info(
                "每 epoch：%d micro-batch / %d step；总 step：%d",
                self.micro_per_epoch,
                self.steps_per_epoch,
                self.total_steps,
            )
            self.log.info(
                "可训练参数：%.3fM / 总参数 %.2fM",
                count_parameters(self.model, trainable_only=True) / 1e6,
                count_parameters(self.model) / 1e6,
            )
        reset_peak_memory(self.device)
        self._train_start_time = time.time()
        self._eta_start_step = self.step

        last_adapter: Path | None = None
        stop = self.step >= self.total_steps
        for epoch in range(self.start_epoch, self.max_epochs):
            if stop or self.step >= self.total_steps:
                break
            self._set_epoch(epoch)
            micro_in_group = 0
            group_loss_sum = 0.0
            group_tokens = 0

            for batch in self.train_loader:
                batch = _move_batch(batch, self.device)
                with self.backend.sync_context(micro_in_group):
                    with autocast_context(self.precision, self.device):
                        loss, token_count = self.compute_loss(batch)
                    scaled = loss / self.grad_accum
                    self.backend.backward(scaled)

                micro_in_group += 1
                self._tokens_processed += int(batch["input_ids"].numel())
                if token_count > 0:
                    group_loss_sum += float(loss.detach()) * token_count
                    group_tokens += token_count

                if micro_in_group < self.grad_accum:
                    continue

                # ---- 一个优化器 step ----
                self._last_group_loss = (
                    group_loss_sum / group_tokens if group_tokens else float("nan")
                )
                if not math.isnan(self._last_group_loss):
                    self.loss_ema.update(self._last_group_loss)
                self.backend.step()
                self.step += 1
                micro_in_group = 0
                group_loss_sum = 0.0
                group_tokens = 0
                saved = self._on_step(epoch)
                if saved is not None:
                    last_adapter = saved

                if cfg.train.max_steps is not None and self.step >= cfg.train.max_steps:
                    stop = True
                    break

            # epoch 末尾的残余梯度
            if not stop and micro_in_group > 0:
                self._last_group_loss = (
                    group_loss_sum / group_tokens if group_tokens else float("nan")
                )
                if not math.isnan(self._last_group_loss):
                    self.loss_ema.update(self._last_group_loss)
                self.backend.step()
                self.step += 1
                saved = self._on_step(epoch)
                if saved is not None:
                    last_adapter = saved
                if (cfg.train.max_steps is not None and self.step >= cfg.train.max_steps) or self.step >= self.total_steps:
                    stop = True

            if not stop and is_main_process() and self._should_save_epoch(epoch):
                saved = self._save(epoch)
                if saved is not None:
                    last_adapter = saved

        # 最终保存
        if is_main_process():
            if self._last_saved_step != self.step:
                saved = self._save(self._eta_current_epoch)
                if saved is not None:
                    last_adapter = saved
            merged = self._export() if cfg.train.merge_at_end else None
            self.log.info("训练结束：%d step", self.step)
            if merged is not None:
                self.log.info("完整权重：%s", merged)
            elif last_adapter is not None:
                self.log.info("适配器/状态权重：%s", last_adapter)
            if self.best_loss is not None:
                self.log.info("最佳验证 loss：%.4f", self.best_loss)
            return TrainResult(
                run_dir=self.run_dir,
                steps=self.step,
                best_loss=self.best_loss,
                last_adapter=last_adapter,
                merged=merged,
            )

        return TrainResult(self.run_dir, self.step, self.best_loss, last_adapter, None)

    # ------------------------------------------------------------------ #
    # 单步回调
    # ------------------------------------------------------------------ #

    _last_saved_step = -1
    _eta_current_epoch = 0

    def _on_step(self, epoch: int) -> Path | None:
        self._eta_current_epoch = epoch
        lr = float(self.optimizer.param_groups[0]["lr"]) if self.optimizer else 0.0

        # 验证在所有 rank 上执行，避免某些 rank 卡在后续 all-reduce。
        if (
            self.val_loader is not None
            and self.cfg.train.val_every_steps > 0
            and self.step % self.cfg.train.val_every_steps == 0
        ):
            val_loss = self.evaluate()
            if is_main_process() and val_loss == val_loss:  # 排除 NaN
                self.metrics.log({"step": self.step, "eval_loss": val_loss})
            if val_loss == val_loss:
                self.best_loss = (
                    val_loss if self.best_loss is None else min(self.best_loss, val_loss)
                )

        if not is_main_process():
            return None

        if self.cfg.train.log_every > 0 and (
            self.step % self.cfg.train.log_every == 0 or self.step == 1
        ):
            self._log_progress(epoch, lr)
        if self.cfg.log.jsonl:
            self.metrics.log(
                {
                    "step": self.step,
                    "epoch": epoch,
                    "loss": getattr(self, "_last_group_loss", float("nan")),
                    "lr": lr,
                    "tokens": self._tokens_processed,
                }
            )

        if self.cfg.train.save_every_steps > 0 and self.step % self.cfg.train.save_every_steps == 0:
            return self._save(epoch)
        return None

    def _log_progress(self, epoch: int, lr: float) -> None:
        now = time.time()
        interval = max(1e-9, now - self._last_log_time)
        tokens = self._tokens_processed - self._last_log_tokens
        tok_per_sec = tokens / interval
        self._last_log_time = now
        self._last_log_tokens = self._tokens_processed

        done = self.step - getattr(self, "_eta_start_step", 0)
        elapsed = now - getattr(self, "_train_start_time", now)
        per_step = elapsed / max(1, done)
        remaining = per_step * max(0, self.total_steps - self.step)
        progress = self.step / max(1, self.total_steps)

        loss_text = (
            f"{self._last_group_loss:.4f}" if not math.isnan(getattr(self, "_last_group_loss", float("nan"))) else "n/a"
        )
        ema_text = f"{self.loss_ema.value:.4f}" if self.loss_ema.value is not None else "n/a"
        mem = peak_memory_gib(self.device)
        mem_text = f" {mem:.1f}GiB" if mem > 0 else ""
        self.log.info(
            "[step %d/%d %5.1f%%] loss %s (ema %s) lr %.2e %s tok/s%s ETA %s",
            self.step,
            self.total_steps,
            100.0 * progress,
            loss_text,
            ema_text,
            lr,
            format_count(tok_per_sec),
            mem_text,
            format_seconds(remaining),
        )

    def _should_save_epoch(self, epoch: int) -> bool:
        every = int(self.cfg.train.save_every_epochs)
        if every <= 0:
            return False
        return (epoch + 1) % every == 0

    # ------------------------------------------------------------------ #
    # 保存 / 导出
    # ------------------------------------------------------------------ #

    def _save(self, epoch: int) -> Path | None:
        if not is_main_process():
            return None
        method = self.cfg.train.method
        step = self.step

        if isinstance(self.backend, DeepSpeedBackend):
            from .deepspeed_backend import gather_trainable_state_dict

            weights = gather_trainable_state_dict(self.backend) or {}
        else:
            weights = collect_weights(self.model, method)

        path = self.run_dir / adapter_filename(method, step)
        save_adapter(
            path,
            method=method,
            step=step,
            epoch=epoch,
            weights=weights,
            base_signature=self.base_signature,
            model_config={
                "vocab_size": self.model_cfg.vocab_size,
                "n_layer": self.model_cfg.n_layer,
                "n_embd": self.model_cfg.n_embd,
                "head_size": self.model_cfg.head_size,
                "head_size_divisor": self.model_cfg.head_size_divisor,
            },
            decay=self.model.decay_mode,
            method_config=self._method_config(),
        )
        save_trainer_state(
            self.run_dir,
            step=step,
            epoch=epoch,
            optimizer=None if isinstance(self.backend, DeepSpeedBackend) else self.optimizer,
            scheduler=None if isinstance(self.backend, DeepSpeedBackend) else self.scheduler,
            scaler=self.scaler,
            best_loss=self.best_loss if self.best_loss is not None else float("nan"),
        )
        set_latest_pointer(self.run_dir, path.name)
        prune_checkpoints(self.run_dir, int(self.cfg.train.keep_last))
        self._last_saved_step = step
        if is_main_process():
            self.log.info("已保存：%s", path.name)
        return path

    def _method_config(self) -> dict[str, Any]:
        if self.cfg.train.method == "lora":
            return {
                "rank": self.cfg.lora.rank,
                "alpha": self.cfg.lora.alpha,
                "dropout": self.cfg.lora.dropout,
                "target_modules": list(self.cfg.lora.target_modules),
                "target_regex": self.cfg.lora.target_regex,
            }
        return {"init_std": self.cfg.state.init_std}

    def _export(self, use_last_epoch: int | None = None) -> Path | None:
        if not is_main_process():
            return None
        path = self.run_dir / "merged.pth"
        try:
            export_merged(self.model, self.cfg.train.method, path)
        except Exception as exc:  # pragma: no cover - 例如内存不足
            self.log.warning("导出合并权重失败：%s（适配器仍然可用）", exc)
            return None
        return path

    # ------------------------------------------------------------------ #
    # 续训
    # ------------------------------------------------------------------ #

    def _resolve_resume(self, resume: str | Path) -> Path:
        path = Path(resume)
        if path.is_dir():
            pointer = latest_pointer(path)
            if pointer is not None:
                return pointer
            candidates = sorted(path.glob("*-step*.pth"))
            if not candidates:
                raise FileNotFoundError(f"目录里没有可续训的权重：{path}")
            return candidates[-1]
        return path

    def _resume(self, resume: str | Path) -> None:
        path = self._resolve_resume(resume)
        info = read_adapter(path)
        if info.method != self.cfg.train.method:
            raise RuntimeError(
                f"续训失败：checkpoint 是 {info.method} 训练的，"
                f"但当前配置是 {self.cfg.train.method}。"
            )
        if info.base_signature and self.base_signature and info.base_signature != self.base_signature:
            self.log.warning(
                "基座模型签名不一致（checkpoint=%s, 当前=%s），请确认用的是同一个模型。",
                info.base_signature,
                self.base_signature,
            )
        if info.decay and info.decay != self.model.decay_mode:
            self.log.warning(
                "checkpoint 的 decay 模式是 %s，当前是 %s；已切换为 checkpoint 的值。",
                info.decay,
                self.model.decay_mode,
            )
            self.model.set_decay_mode(info.decay)

        if self.cfg.train.method == "lora":
            load_lora_state_dict(self.model, info.weights)
        else:
            load_state_state_dict(self.model, info.weights)

        self.start_step = info.step
        self.start_epoch = info.epoch + 1
        self.step = info.step
        self._last_saved_step = info.step

        trainer_state = self.run_dir / trainer_state_filename(info.step)
        if trainer_state.exists() and not isinstance(self.backend, DeepSpeedBackend):
            payload = load_trainer_state(trainer_state)
            if payload.get("optimizer") is not None:
                self.optimizer.load_state_dict(payload["optimizer"])
            if payload.get("scheduler") is not None:
                try:
                    self.scheduler.load_state_dict(payload["scheduler"])
                except Exception as exc:  # pragma: no cover
                    self.log.warning("恢复 LR 调度器失败（%s），将从头开始调度。", exc)
            if payload.get("scaler") is not None and self.scaler is not None:
                self.scaler.load_state_dict(payload["scaler"])
            best = payload.get("best_loss")
            if best is not None and not math.isnan(float(best)):
                self.best_loss = float(best)
            self.log.info(
                "续训：从 step %d / epoch %d 继续（已恢复优化器状态）",
                self.step,
                self.start_epoch,
            )
        else:
            self.log.info(
                "续训：从 step %d / epoch %d 继续（未找到优化器状态，仅恢复权重）",
                self.step,
                self.start_epoch,
            )

    # ------------------------------------------------------------------ #
    # 评估
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for batch in self.val_loader:
            batch = _move_batch(batch, self.device)
            with autocast_context(self.precision, self.device):
                loss, count = self.compute_loss(batch)
            if count > 0:
                total_loss += float(loss.detach()) * count
                total_tokens += count
        self.model.train()
        if total_tokens == 0:
            if is_main_process():
                self.log.warning("验证集没有可训练 token，跳过本次评估。")
            return float("nan")
        value = total_loss / total_tokens
        if is_main_process():
            self.log.info("验证 loss：%.4f（%d tokens）", value, total_tokens)
        return value
