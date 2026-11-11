"""分词器与 ShareGPT 解析测试。"""

from __future__ import annotations

import json
from itertools import pairwise

import pytest

from rwkv_easy.data.sharegpt import (
    Conversation,
    TextDocument,
    content_to_text,
    load_records,
    normalize_role,
    parse_record,
)
from rwkv_easy.data.tokenizer import get_tokenizer

TEXTS = [
    "你好，世界！",
    "Hello, RWKV-EASY!",
    "混合 text 与中文 12345，还有标点：；、。",
    "emoji 😀🚀 and tabs\tand newlines\n\nend",
    "code: def f(x):\n    return x ** 2\n",
]


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer()


@pytest.mark.parametrize("text", TEXTS)
def test_roundtrip(tokenizer, text):
    ids = tokenizer.encode(text)
    assert ids
    assert tokenizer.decode(ids) == text


def test_offsets_are_contiguous(tokenizer):
    raw = "你好abc😀".encode()
    encoded = tokenizer.encode_with_offsets(raw)
    assert encoded[0][1] == 0
    for prev, cur in pairwise(encoded):
        assert prev[2] == cur[1]
    assert encoded[-1][2] == len(raw)


def test_eot_id(tokenizer):
    assert tokenizer.eot_id == tokenizer.token2idx[b"\x00"]
    assert tokenizer.decode([tokenizer.eot_id]) == "\x00"


def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size > 65000
    assert tokenizer.max_token_len > 1


# --------------------------------------------------------------------------- #
# ShareGPT 解析
# --------------------------------------------------------------------------- #


def test_normalize_role():
    assert normalize_role("human") == "user"
    assert normalize_role("GPT") == "assistant"
    assert normalize_role("system") == "system"
    assert normalize_role("whatever") == "user"


def test_content_to_text_list():
    value = [{"type": "text", "text": "hi "}, {"type": "image_url"}, "there"]
    assert content_to_text(value) == "hi there"


def test_parse_sharegpt():
    record = parse_record(
        {
            "conversations": [
                {"from": "human", "value": "a"},
                {"from": "gpt", "value": "b"},
            ]
        }
    )
    assert isinstance(record, Conversation)
    assert record.roles() == ["user", "assistant"]


def test_parse_messages_and_multimodal():
    record = parse_record(
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "看图"}]},
                {"role": "assistant", "content": "好的"},
            ]
        }
    )
    assert record.roles() == ["user", "assistant"]
    assert record.messages[0].content == "看图"


def test_parse_prompt_response():
    record = parse_record({"instruction": "做点什么", "output": "完成了"})
    assert record.roles() == ["user", "assistant"]


def test_parse_text():
    record = parse_record({"text": "一段纯文本"})
    assert isinstance(record, TextDocument)


def test_parse_unknown_returns_none():
    assert parse_record({"foo": 1}) is None


def test_merge_consecutive_same_role():
    record = parse_record(
        {
            "conversations": [
                {"from": "human", "value": "第一句"},
                {"from": "human", "value": "第二句"},
                {"from": "gpt", "value": "回答"},
            ]
        }
    )
    assert len(record.messages) == 2
    assert "第一句" in record.messages[0].content
    assert "第二句" in record.messages[0].content


def test_load_records_jsonl(tmp_path):
    path = tmp_path / "d.jsonl"
    rows = [
        {"conversations": [{"from": "human", "value": "a"}, {"from": "gpt", "value": "b"}]},
        {"messages": [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    records = load_records(path)
    assert len(records) == 2


def test_load_records_txt(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text("hello world", encoding="utf-8")
    records = load_records(path)
    assert len(records) == 1
    assert isinstance(records[0], TextDocument)


def test_load_records_wrapped_json(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"data": [{"text": "x"}]}), encoding="utf-8")
    records = load_records(path, text_field="text")
    assert len(records) == 1


def test_load_records_empty_raises(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_records(path)
