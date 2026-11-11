"""ShareGPT 训练数据集：tokenize、assistant-only loss mask、packed / sample 两种组织方式。

核心概念
--------

* ``tokens``：``uint16`` 的 token 流（RWKV 词表 65529 < 65536，省一半内存）。
* ``mask``  ：``uint8``，1 表示该 token 是“可训练目标”（assistant 回复、结尾 EOS）。
* ``labels``：在取窗口时由 ``tokens``/``mask`` 即时算出：

  ``labels[t] = tokens[t+1] if mask[t+1] else -100``

  这样 ``labels[t]`` 表示“位置 t 需要预测的下一个 token”，也不会把不同文档的
  边界连起来学习。

两种组织方式
------------

* ``packed``：把全部对话拼成一条定长流，按 ``ctx_len`` 切窗。利用率接近 100%，
  且天然满足 infctx 对“定长、无 padding”的要求。
* ``sample``：一条对话一个样本，batch 内动态 padding（需要 attention_mask），
  不能与 infctx 同时使用。
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import pickle
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ..config import Config
from .sharegpt import Record, TextDocument, load_records
from .template import ChatTemplate, build_template
from .tokenizer import RWKVTokenizer, get_tokenizer

__all__ = [
    "DatasetStats",
    "PackedDataset",
    "SampleCollator",
    "SampleDataset",
    "build_datasets",
    "make_dataloader",
    "tokenize_record",
]

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #


@dataclass
class DatasetStats:
    documents: int = 0
    dropped: int = 0
    tokens: int = 0
    supervised_tokens: int = 0
    val_documents: int = 0
    val_tokens: int = 0
    val_supervised_tokens: int = 0
    windows: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "documents": self.documents,
            "dropped": self.dropped,
            "tokens": self.tokens,
            "supervised_tokens": self.supervised_tokens,
            "val_documents": self.val_documents,
            "val_tokens": self.val_tokens,
            "val_supervised_tokens": self.val_supervised_tokens,
            "windows": self.windows,
        }


# --------------------------------------------------------------------------- #
# 单条记录 -> (ids, mask)
# --------------------------------------------------------------------------- #


def _eos_text(tokenizer: RWKVTokenizer, eos_id: int | None) -> str:
    """把 EOS token 的字节表示安全地塞进 Python 字符串（surrogateescape 可往返）。"""
    if eos_id is None:
        eos_id = tokenizer.eot_id
    try:
        raw = tokenizer.token_text(int(eos_id))
    except KeyError as exc:
        raise ValueError(f"eos_token_id={eos_id} 不在词表中") from exc
    return raw.decode("utf-8", errors="surrogateescape")


def _encode_text(text: str) -> bytes:
    return text.encode("utf-8", errors="surrogateescape")


def tokenize_record(
    record: Record,
    tokenizer: RWKVTokenizer,
    template: ChatTemplate,
    *,
    train_on: str = "assistant",
    append_eos: str = "end",
    eos_token_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """把一条记录转成 ``(ids, mask)``。

    mask 的判定使用 **字节区间重叠**：只要 token 覆盖到任意一个可训练区间就算
    可训练。这样 "Assistant: " 与回复之间被合并成一个 token 时，回复的第一个
    token 不会被漏掉。
    """
    if isinstance(record, TextDocument):
        segments: list[tuple[str, bool]] = [(record.text, True)]
    else:
        segments = template.render(
            record,
            train_on=train_on,
            append_eos=append_eos,
            eos_text=_eos_text(tokenizer, eos_token_id),
        )
    if not segments:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.uint8)

    # 拼接字节并记录可训练区间
    chunks: list[bytes] = []
    spans: list[tuple[int, int]] = []
    offset = 0
    for text, trainable in segments:
        raw = _encode_text(text)
        if not raw:
            continue
        if trainable:
            spans.append((offset, offset + len(raw)))
        chunks.append(raw)
        offset += len(raw)
    if offset == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.uint8)
    payload = b"".join(chunks)

    byte_mask = np.zeros(offset, dtype=np.uint8)
    for start, end in spans:
        byte_mask[start:end] = 1

    encoded = tokenizer.encode_with_offsets(payload)
    ids = np.fromiter((item[0] for item in encoded), dtype=np.int32, count=len(encoded))
    mask = np.zeros(len(encoded), dtype=np.uint8)
    for i, (_, start, end) in enumerate(encoded):
        if byte_mask[start:end].any():
            mask[i] = 1

    # 第一个 token 没有“前一个位置”来预测它，不作为目标。
    if len(mask):
        mask[0] = 0
    return ids, mask


# --------------------------------------------------------------------------- #
# 多进程 tokenize（带缓存）
# --------------------------------------------------------------------------- #


_WORKER_STATE: dict[str, Any] = {}


def _init_worker(
    vocab_path: str | None,
    template_kwargs: dict[str, Any],
    tokenize_kwargs: dict[str, Any],
) -> None:
    _WORKER_STATE["tokenizer"] = get_tokenizer(vocab_path)
    _WORKER_STATE["template"] = build_template(**template_kwargs)
    _WORKER_STATE["kwargs"] = tokenize_kwargs


def _worker_tokenize(record: Record) -> tuple[np.ndarray, np.ndarray]:
    return tokenize_record(
        record,
        _WORKER_STATE["tokenizer"],
        _WORKER_STATE["template"],
        **_WORKER_STATE["kwargs"],
    )


def _data_files_signature(path: Path) -> list[tuple[str, int, int]]:
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in (".json", ".jsonl", ".txt", ".parquet")
        )
    else:
        files = [path]
    signature = []
    for file in files:
        stat = file.stat()
        signature.append((str(file.resolve()), stat.st_size, stat.st_mtime_ns))
    return signature


def _cache_key(cfg: Config, records_info: list[tuple[str, int, int]], tokenizer_path: str) -> str:
    payload = {
        "files": records_info,
        "tokenizer": tokenizer_path,
        "tokenizer_mtime": Path(tokenizer_path).stat().st_mtime if Path(tokenizer_path).exists() else 0,
        "format": cfg.data.format,
        "text_field": cfg.data.text_field,
        "template": [
            cfg.data.template,
            cfg.data.system_prefix,
            cfg.data.user_prefix,
            cfg.data.assistant_prefix,
            cfg.data.turn_separator,
        ],
        "train_on": cfg.data.train_on,
        "append_eos": cfg.data.append_eos,
        "eos_token_id": cfg.data.eos_token_id,
        "max_samples": cfg.data.max_samples,
        "val_ratio": cfg.data.val_ratio,
        "seed": cfg.run.seed,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _cache_path(cfg: Config) -> Path:
    if cfg.data.cache_dir:
        base = Path(cfg.data.cache_dir)
    else:
        data_path = Path(cfg.data.path)
        base = (data_path if data_path.is_dir() else data_path.parent) / ".rwkv_easy_cache"
    return base


def _tokenize_all(
    records: Sequence[Record],
    tokenizer: RWKVTokenizer,
    template: ChatTemplate,
    cfg: Config,
    log: Any = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    kwargs = {
        "train_on": cfg.data.train_on,
        "append_eos": cfg.data.append_eos,
        "eos_token_id": cfg.data.eos_token_id,
    }
    num_proc = int(cfg.data.num_proc or 0)
    # 多卡时每个 rank 都在 tokenize，按进程数摊薄，避免 8 卡起 32 个进程。
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        num_proc = max(1, num_proc // max(1, torch.distributed.get_world_size()))
    if num_proc <= 1 or len(records) < 256:
        results = [tokenize_record(record, tokenizer, template, **kwargs) for record in records]
        return results

    template_kwargs = {
        "name": cfg.data.template,
        "system_prefix": cfg.data.system_prefix,
        "user_prefix": cfg.data.user_prefix,
        "assistant_prefix": cfg.data.assistant_prefix,
        "turn_separator": cfg.data.turn_separator,
    }
    try:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_proc,
            mp_context=context,
            initializer=_init_worker,
            initargs=(cfg.data.vocab_path, template_kwargs, kwargs),
        ) as executor:
            results = []
            for index, item in enumerate(executor.map(_worker_tokenize, records, chunksize=64), 1):
                results.append(item)
                if log is not None and index % 2000 == 0:
                    log.info("tokenize %d/%d", index, len(records))
        return results
    except Exception as exc:  # pragma: no cover - 平台差异/权限问题
        if log is not None:
            log.warning("多进程 tokenize 失败（%s），回退到单进程。", exc)
        return [tokenize_record(record, tokenizer, template, **kwargs) for record in records]


def _load_or_tokenize(
    records: Sequence[Record],
    tokenizer: RWKVTokenizer,
    template: ChatTemplate,
    cfg: Config,
    log: Any = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not cfg.data.cache:
        return _tokenize_all(records, tokenizer, template, cfg, log)

    vocab_path = str(Path(cfg.data.vocab_path) if cfg.data.vocab_path else tokenizer.vocab_path)
    key = _cache_key(cfg, _data_files_signature(Path(cfg.data.path)), vocab_path)
    cache_dir = _cache_path(cfg)
    cache_file = cache_dir / f"tokens-{key}.pkl"

    if cache_file.exists():
        try:
            with cache_file.open("rb") as handle:
                payload = pickle.load(handle)
            if len(payload["ids"]) == len(records):
                if log is not None:
                    log.info("命中 tokenize 缓存：%s", cache_file)
                return list(zip(payload["ids"], payload["masks"], strict=True))
        except Exception as exc:  # pragma: no cover - 缓存损坏
            if log is not None:
                log.warning("缓存读取失败（%s），重新 tokenize。", exc)

    results = _tokenize_all(records, tokenizer, template, cfg, log)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # 先写临时文件再原子替换：多卡同时构建数据集时不会互相写坏缓存。
        import tempfile

        fd, tmp_name = tempfile.mkstemp(prefix=cache_file.name + ".", suffix=".tmp", dir=str(cache_dir))
        os.close(fd)
        try:
            with open(tmp_name, "wb") as handle:
                pickle.dump(
                    {"ids": [r[0] for r in results], "masks": [r[1] for r in results]},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(tmp_name, cache_file)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
    except OSError as exc:  # pragma: no cover - 只读目录等
        if log is not None:
            log.warning("写入 tokenize 缓存失败：%s", exc)
    return results


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #


class PackedDataset(Dataset):
    """定长 token 流数据集。"""

    def __init__(
        self, tokens: np.ndarray, mask: np.ndarray, ctx_len: int, pad_token_id: int = 0
    ) -> None:
        if len(tokens) != len(mask):
            raise ValueError("tokens 与 mask 长度不一致")
        self.ctx_len = int(ctx_len)
        self.pad_token_id = int(pad_token_id)
        self.num_windows = max(1, len(tokens) // self.ctx_len)
        # 保证每个窗口都能取到 t+1 位置
        needed = self.num_windows * self.ctx_len + 1
        if len(tokens) < needed:
            pad = needed - len(tokens)
            tokens = np.concatenate(
                [tokens, np.full(pad, self.pad_token_id, dtype=tokens.dtype)]
            )
            mask = np.concatenate([mask, np.zeros(pad, dtype=mask.dtype)])
        self.tokens = tokens
        self.mask = mask
        self.supervised_tokens = int(mask.sum())

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.ctx_len
        end = start + self.ctx_len
        ids = self.tokens[start:end].astype(np.int64)
        next_ids = self.tokens[start + 1 : end + 1].astype(np.int64)
        next_mask = self.mask[start + 1 : end + 1]
        labels = np.where(next_mask == 1, next_ids, IGNORE_INDEX).astype(np.int64)
        return {
            "input_ids": torch.from_numpy(ids),
            "labels": torch.from_numpy(labels),
        }

    def window_tokens(self, index: int) -> np.ndarray:  # pragma: no cover - 调试
        start = index * self.ctx_len
        return self.tokens[start : start + self.ctx_len]


class SampleDataset(Dataset):
    """一条对话一个样本（仅非 infctx）。"""

    def __init__(
        self,
        samples: Sequence[tuple[np.ndarray, np.ndarray]],
        ctx_len: int,
        truncation: str = "head",
        seed: int = 42,
    ) -> None:
        self.samples = list(samples)
        self.ctx_len = int(ctx_len)
        self.truncation = truncation
        self.rng = np.random.default_rng(seed)
        self.supervised_tokens = int(sum(int(mask.sum()) for _, mask in self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def _crop(self, ids: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(ids) <= self.ctx_len:
            return ids, mask
        if self.truncation == "tail":
            start = len(ids) - self.ctx_len
        elif self.truncation == "random":
            start = int(self.rng.integers(0, len(ids) - self.ctx_len + 1))
        else:  # head
            start = 0
        return ids[start : start + self.ctx_len], mask[start : start + self.ctx_len]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        ids, mask = self._crop(*self.samples[index])
        return {"ids": ids, "mask": mask}


class SampleCollator:
    """动态 padding + 构造 labels / attention_mask。"""

    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
        length = max(len(item["ids"]) for item in batch)
        size = len(batch)
        input_ids = np.full((size, length), self.pad_token_id, dtype=np.int64)
        labels = np.full((size, length), IGNORE_INDEX, dtype=np.int64)
        attention_mask = np.zeros((size, length), dtype=np.int64)
        for row, item in enumerate(batch):
            ids, mask = item["ids"], item["mask"]
            n = len(ids)
            input_ids[row, :n] = ids
            attention_mask[row, :n] = 1
            # labels[t] 预测 t+1
            if n > 1:
                nxt = ids[1:]
                nxt_mask = mask[1:]
                labels[row, : n - 1] = np.where(nxt_mask == 1, nxt, IGNORE_INDEX)
        return {
            "input_ids": torch.from_numpy(input_ids),
            "labels": torch.from_numpy(labels),
            "attention_mask": torch.from_numpy(attention_mask),
        }


# --------------------------------------------------------------------------- #
# 构建
# --------------------------------------------------------------------------- #


@dataclass
class BuiltData:
    train: Dataset
    val: Dataset | None
    collate_fn: Any
    stats: DatasetStats
    tokenizer: RWKVTokenizer
    template: ChatTemplate


def _concat_arrays(items: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if not items:
        return np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.uint8)
    total = sum(len(ids) for ids, _ in items)
    tokens = np.empty(total, dtype=np.uint16)
    mask = np.empty(total, dtype=np.uint8)
    offset = 0
    for ids, item_mask in items:
        n = len(ids)
        tokens[offset : offset + n] = ids
        mask[offset : offset + n] = item_mask
        offset += n
    return tokens, mask


def _split_records(
    records: Sequence[Record], val_ratio: float, seed: int
) -> tuple[list[Record], list[Record]]:
    if val_ratio <= 0 or len(records) < 10:
        return list(records), []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    val_size = max(1, int(len(records) * val_ratio))
    val_indices = {int(i) for i in order[:val_size]}
    train = [record for i, record in enumerate(records) if i not in val_indices]
    val = [record for i, record in enumerate(records) if i in val_indices]
    return train, val


def build_datasets(cfg: Config, log: Any = None) -> BuiltData:
    """按配置构建训练/验证数据集。"""
    tokenizer = get_tokenizer(cfg.data.vocab_path)
    template = build_template(
        name=cfg.data.template,
        system_prefix=cfg.data.system_prefix,
        user_prefix=cfg.data.user_prefix,
        assistant_prefix=cfg.data.assistant_prefix,
        turn_separator=cfg.data.turn_separator,
    )

    if log is not None:
        log.info("加载数据集：%s", cfg.data.path)
    records = load_records(
        cfg.data.path,
        fmt=cfg.data.format,
        text_field=cfg.data.text_field,
        max_samples=cfg.data.max_samples,
        log=log,
    )
    if log is not None:
        log.info("共 %d 条对话/文档", len(records))

    train_records, val_records = _split_records(records, cfg.data.val_ratio, cfg.run.seed)
    if val_records and log is not None:
        log.info(
            "训练/验证切分：%d / %d 条（val_ratio=%.3f）",
            len(train_records),
            len(val_records),
            cfg.data.val_ratio,
        )

    train_tokens = _load_or_tokenize(train_records, tokenizer, template, cfg, log)
    val_tokens = (
        _load_or_tokenize(val_records, tokenizer, template, cfg, log) if val_records else []
    )

    stats = DatasetStats(documents=len(train_records), val_documents=len(val_records))
    kept_train = [
        item for item in train_tokens if len(item[0]) > 0 and (not cfg.data.drop_empty or item[1].sum() > 0)
    ]
    stats.dropped += len(train_tokens) - len(kept_train)
    kept_val = [
        item for item in val_tokens if len(item[0]) > 0 and (not cfg.data.drop_empty or item[1].sum() > 0)
    ]

    if not kept_train:
        raise ValueError(
            "训练集里没有任何可训练的 assistant token。\n"
            "请检查 data.template / data.train_on / 数据集是否包含 assistant 回复。"
        )

    if cfg.data.mode == "packed":
        tokens, mask = _concat_arrays(kept_train)
        stats.tokens = len(tokens)
        stats.supervised_tokens = int(mask.sum())
        train_dataset: Dataset = PackedDataset(
            tokens, mask, cfg.train.ctx_len, pad_token_id=tokenizer.eot_id
        )
        stats.windows = len(train_dataset)  # type: ignore[arg-type]
        val_dataset: Dataset | None = None
        if kept_val:
            val_tokens_arr, val_mask = _concat_arrays(kept_val)
            stats.val_tokens = len(val_tokens_arr)
            stats.val_supervised_tokens = int(val_mask.sum())
            val_dataset = PackedDataset(
                val_tokens_arr, val_mask, cfg.train.ctx_len, pad_token_id=tokenizer.eot_id
            )
        collate_fn = None
    else:
        stats.tokens = int(sum(len(ids) for ids, _ in kept_train))
        stats.supervised_tokens = int(sum(int(m.sum()) for _, m in kept_train))
        train_dataset = SampleDataset(kept_train, cfg.train.ctx_len, cfg.data.truncation, cfg.run.seed)
        stats.windows = len(train_dataset)
        val_dataset = (
            SampleDataset(kept_val, cfg.train.ctx_len, cfg.data.truncation, cfg.run.seed + 1)
            if kept_val
            else None
        )
        stats.val_tokens = int(sum(len(ids) for ids, _ in kept_val))
        stats.val_supervised_tokens = int(sum(int(m.sum()) for _, m in kept_val))
        collate_fn = SampleCollator(pad_token_id=tokenizer.eot_id)

    if stats.supervised_tokens == 0:
        warnings.warn("训练集没有可训练的 token，请检查 data.train_on 与数据内容。", stacklevel=2)

    if log is not None:
        log.info(
            "数据集就绪：%d 窗口 / %d 文档，%d tokens（其中可训练 %d）",
            stats.windows,
            stats.documents,
            stats.tokens,
            stats.supervised_tokens,
        )

    return BuiltData(
        train=train_dataset,
        val=val_dataset,
        collate_fn=collate_fn,
        stats=stats,
        tokenizer=tokenizer,
        template=template,
    )


# --------------------------------------------------------------------------- #
# DataLoader
# --------------------------------------------------------------------------- #


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn: Any,
    *,
    shuffle: bool,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
        shuffle = False

    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=collate_fn,
        **kwargs,
    )
