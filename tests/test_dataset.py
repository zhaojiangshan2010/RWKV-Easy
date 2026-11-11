"""数据集与 loss mask 测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import build_config

from rwkv_easy.data.dataset import (
    PackedDataset,
    SampleCollator,
    build_datasets,
    tokenize_record,
)
from rwkv_easy.data.sharegpt import parse_record
from rwkv_easy.data.template import build_template
from rwkv_easy.data.tokenizer import get_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer()


@pytest.fixture(scope="module")
def template():
    return build_template("rwkv_world")


CONVERSATION = {
    "conversations": [
        {"from": "user", "value": "请只回答一个词"},
        {"from": "gpt", "value": "苹果"},
    ]
}


def test_mask_covers_only_assistant(tokenizer, template):
    record = parse_record(CONVERSATION)
    ids, mask = tokenize_record(record, tokenizer, template, train_on="assistant", append_eos="end")
    supervised = tokenizer.decode(ids[mask == 1].tolist())
    whole = tokenizer.decode(ids.tolist())
    assert "苹果" in supervised
    assert "User:" not in supervised
    assert "请只回答一个词" not in supervised
    assert whole.startswith("User: ")
    assert "\x00" in supervised  # EOS 也参与训练
    assert mask[0] == 0  # 第一个 token 永远不作为目标


def test_last_assistant_only(tokenizer, template):
    record = parse_record(
        {
            "conversations": [
                {"from": "user", "value": "第一问"},
                {"from": "gpt", "value": "第一答"},
                {"from": "user", "value": "第二问"},
                {"from": "gpt", "value": "第二答"},
            ]
        }
    )
    ids_all, mask_all = tokenize_record(record, tokenizer, template, train_on="assistant")
    ids_last, mask_last = tokenize_record(record, tokenizer, template, train_on="last_assistant")
    text_all = tokenizer.decode(ids_all[mask_all == 1].tolist())
    text_last = tokenizer.decode(ids_last[mask_last == 1].tolist())
    assert "第一答" in text_all and "第二答" in text_all
    assert "第一答" not in text_last and "第二答" in text_last


def test_append_eos_never(tokenizer, template):
    record = parse_record(CONVERSATION)
    ids, mask = tokenize_record(record, tokenizer, template, append_eos="never")
    assert "\x00" not in tokenizer.decode(ids[mask == 1].tolist())


def test_append_eos_each_assistant(tokenizer, template):
    record = parse_record(
        {
            "conversations": [
                {"from": "user", "value": "问一"},
                {"from": "gpt", "value": "答一"},
                {"from": "user", "value": "问二"},
                {"from": "gpt", "value": "答二"},
            ]
        }
    )
    ids, mask = tokenize_record(record, tokenizer, template, append_eos="each_assistant")
    text = tokenizer.decode(ids[mask == 1].tolist())
    assert text.count("\x00") == 2


def test_packed_dataset_labels(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(tmp_path, tiny_checkpoint(), tiny_sharegpt)
    built = build_datasets(cfg)
    dataset = built.train
    assert isinstance(dataset, PackedDataset)
    assert len(dataset) > 0
    for index in range(len(dataset)):
        item = dataset[index]
        ids = item["input_ids"].numpy()
        labels = item["labels"].numpy()
        start = index * dataset.ctx_len
        tokens = dataset.tokens[start : start + dataset.ctx_len].astype(np.int64)
        mask = dataset.mask[start + 1 : start + dataset.ctx_len + 1]
        next_tokens = dataset.tokens[start + 1 : start + dataset.ctx_len + 1].astype(np.int64)
        assert np.array_equal(ids, tokens)
        expected = np.where(mask == 1, next_tokens, -100)
        assert np.array_equal(labels, expected)
        assert (ids < 65530).all()


def test_packed_dataset_shapes(tmp_path):
    tokens = np.arange(200, dtype=np.uint16) % 100
    mask = np.ones_like(tokens, dtype=np.uint8)
    dataset = PackedDataset(tokens, mask, ctx_len=64, pad_token_id=1)
    assert len(dataset) == 3
    item = dataset[0]
    assert item["input_ids"].dtype == torch.int64
    assert item["labels"].shape == (64,)


def test_sample_collator():
    collator = SampleCollator(pad_token_id=1)
    batch = [
        {"ids": np.array([5, 6, 7, 8], dtype=np.int32), "mask": np.array([0, 0, 1, 1], dtype=np.uint8)},
        {"ids": np.array([9, 10], dtype=np.int32), "mask": np.array([0, 1], dtype=np.uint8)},
    ]
    out = collator(batch)
    assert out["input_ids"].shape == (2, 4)
    assert out["labels"].tolist() == [[-100, 7, 8, -100], [10, -100, -100, -100]]
    assert out["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    # padding 位置不参与 loss
    assert out["labels"][1, 2].item() == -100


def test_build_datasets_sample_mode(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(tmp_path, tiny_checkpoint(), tiny_sharegpt, {"data": {"mode": "sample"}})
    built = build_datasets(cfg)
    assert built.collate_fn is not None
    dataset = built.train
    assert len(dataset) > 0
    item = dataset[0]
    assert "ids" in item and "mask" in item
    assert len(item["ids"]) == len(item["mask"])


def test_build_datasets_cache(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"data": {"cache": True, "cache_dir": str(tmp_path / "cache")}},
    )
    first = build_datasets(cfg)
    cache_files = list((tmp_path / "cache").glob("tokens-*.pkl"))
    assert cache_files, "应该写出 tokenize 缓存"
    second = build_datasets(cfg)
    assert first.stats.tokens == second.stats.tokens
    assert first.stats.supervised_tokens == second.stats.supervised_tokens
    assert second.stats.supervised_tokens > 0


def test_val_split(tmp_path, tiny_checkpoint, tiny_sharegpt):
    cfg = build_config(
        tmp_path,
        tiny_checkpoint(),
        tiny_sharegpt,
        {"data": {"val_ratio": 0.2}, "train": {"ctx_len": 32}},
    )
    built = build_datasets(cfg)
    assert built.val is not None
    assert built.stats.val_documents > 0
