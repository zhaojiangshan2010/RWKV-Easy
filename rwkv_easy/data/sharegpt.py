"""对话数据结构与 ShareGPT 格式解析。

支持三种常见数据集写法，并统一成 :class:`Conversation`：

1. ShareGPT::

       {"conversations": [
           {"from": "human", "value": "你好"},
           {"from": "gpt",   "value": "你好！"}
       ]}

2. OpenAI Messages::

       {"messages": [
           {"role": "user",      "content": "你好"},
           {"role": "assistant", "content": "你好！"}
       ]}

3. 纯文本：``{"text": "..."}``、``.txt`` 文件、或 JSON 里就是字符串。

角色会被归一到 ``system / user / assistant``；``tool``、``function`` 等角色按
“上下文”处理（不计入 loss）。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Conversation",
    "Message",
    "Record",
    "TextDocument",
    "content_to_text",
    "load_records",
    "load_records_from_file",
    "normalize_role",
    "parse_record",
]


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    source: str | None = None

    def __bool__(self) -> bool:
        return bool(self.messages)

    def roles(self) -> list[str]:
        return [m.role for m in self.messages]


@dataclass
class TextDocument:
    """纯文本语料：整段都参与训练（第一 token 除外）。"""

    text: str
    source: str | None = None


Record = Conversation | TextDocument


# --------------------------------------------------------------------------- #
# 角色与内容归一化
# --------------------------------------------------------------------------- #

_ROLE_ALIASES = {
    "system": "system",
    "sys": "system",
    "developer": "system",
    "human": "user",
    "user": "user",
    "usr": "user",
    "prompt": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "ai": "assistant",
    "model": "assistant",
    "chatbot": "assistant",
    "tool": "user",
    "function": "user",
    "observation": "user",
    "tool_result": "user",
}


def normalize_role(role: Any) -> str:
    text = str(role or "").strip().lower()
    return _ROLE_ALIASES.get(text, "user")


def content_to_text(value: Any) -> str:
    """把 ``content`` 归一化成纯文本。

    OpenAI 的多模态格式是 ``[{"type": "text", "text": "..."}]``，这里只保留文本部分。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "content" in item:
                    parts.append(content_to_text(item["content"]))
        return "".join(parts)
    if isinstance(value, dict):
        if "text" in value:
            return content_to_text(value["text"])
        if "content" in value:
            return content_to_text(value["content"])
    return ""


def _merge_messages(messages: list[Message]) -> list[Message]:
    """合并连续同角色消息（ShareGPT 里偶尔会有多段 human）。"""
    merged: list[Message] = []
    for message in messages:
        content = message.content.strip()
        if not content:
            continue
        if merged and merged[-1].role == message.role:
            merged[-1] = Message(message.role, merged[-1].content + "\n\n" + content)
        else:
            merged.append(Message(message.role, content))
    return merged


# --------------------------------------------------------------------------- #
# 单条记录
# --------------------------------------------------------------------------- #


def _turns_from_conversations(items: Sequence[Any]) -> list[Message]:
    messages: list[Message] = []
    for item in items:
        if isinstance(item, str):
            messages.append(Message("user", item))
            continue
        if not isinstance(item, dict):
            continue
        role = item.get("from", item.get("role", item.get("speaker")))
        content = item.get(
            "value", item.get("content", item.get("text", item.get("message")))
        )
        text = content_to_text(content)
        if not text.strip():
            continue
        messages.append(Message(normalize_role(role), text))
    return messages


def _turns_from_messages(items: Sequence[Any]) -> list[Message]:
    messages: list[Message] = []
    for item in items:
        if isinstance(item, str):
            messages.append(Message("user", item))
            continue
        if not isinstance(item, dict):
            continue
        role = item.get("role", item.get("from", "user"))
        content = item.get("content", item.get("value", item.get("text")))
        text = content_to_text(content)
        if not text.strip():
            continue
        messages.append(Message(normalize_role(role), text))
    return messages


def _two_turn(prompt: Any, response: Any) -> list[Message]:
    messages = []
    prompt_text = content_to_text(prompt)
    response_text = content_to_text(response)
    if prompt_text.strip():
        messages.append(Message("user", prompt_text))
    if response_text.strip():
        messages.append(Message("assistant", response_text))
    return messages


def parse_record(
    obj: Any,
    fmt: str = "auto",
    text_field: str = "text",
    source: str | None = None,
) -> Record | None:
    """把一条原始记录解析成 :class:`Conversation` 或 :class:`TextDocument`。

    返回 ``None`` 表示无法识别（调用方负责统计/告警）。
    """
    if isinstance(obj, str):
        return TextDocument(obj, source=source) if obj.strip() else None
    if not isinstance(obj, dict):
        return None

    fmt = (fmt or "auto").lower()

    if fmt in ("auto", "text") and text_field in obj and isinstance(obj[text_field], str):
        return TextDocument(obj[text_field], source=source)

    if fmt in ("auto", "sharegpt", "messages"):
        if isinstance(obj.get("conversations"), list):
            if fmt == "messages":
                messages = _turns_from_messages(obj["conversations"])
            else:
                messages = _turns_from_conversations(obj["conversations"])
            messages = _merge_messages(messages)
            return Conversation(messages, source=source) if messages else None

        if isinstance(obj.get("messages"), list):
            messages = _turns_from_messages(obj["messages"])
            messages = _merge_messages(messages)
            return Conversation(messages, source=source) if messages else None

    # 兼容常见的两列指令数据
    for prompt_key, response_key in (
        ("prompt", "response"),
        ("instruction", "output"),
        ("question", "answer"),
        ("input", "output"),
    ):
        if prompt_key in obj and response_key in obj:
            messages = _merge_messages(_two_turn(obj[prompt_key], obj[response_key]))
            return Conversation(messages, source=source) if messages else None

    if fmt in ("auto", "text"):
        if "text" in obj and isinstance(obj["text"], str):
            return TextDocument(obj["text"], source=source)
        if "content" in obj and isinstance(obj["content"], str):
            return TextDocument(obj["content"], source=source)

    return None


# --------------------------------------------------------------------------- #
# 文件加载
# --------------------------------------------------------------------------- #

_SUPPORTED_SUFFIXES = {".json", ".jsonl", ".jsonl.gz", ".txt", ".parquet"}


def _flatten_json_payload(payload: Any) -> list[Any]:
    """把 ``{"data": [...]}`` / ``{"conversations": [...]}`` 之类的包装拆开。"""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "rows", "examples", "instances"):
            if isinstance(payload.get(key), list):
                return payload[key]
        # 单条对话（含 conversations/messages 字段）直接当作一条记录
        if "conversations" in payload or "messages" in payload:
            return [payload]
        return [payload]
    return [payload]


def _read_json(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    return _flatten_json_payload(payload)


def _read_jsonl(path: Path, errors: list[str]) -> list[Any]:
    rows: list[Any] = []
    opener = open
    if path.suffix == ".gz":  # pragma: no cover - 可选路径
        import gzip

        opener = gzip.open  # type: ignore[assignment]
    with opener(path, "rt", encoding="utf-8-sig") as handle:  # type: ignore[operator]
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{lineno} JSON 解析失败：{exc}")
    return rows


def _read_text(path: Path) -> list[Any]:
    return [path.read_text(encoding="utf-8-sig", errors="replace")]


def _read_parquet(path: Path) -> list[Any]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise RuntimeError(
            f"读取 parquet 需要 pyarrow：pip install pyarrow（文件：{path}）"
        ) from exc
    table = pq.read_table(path)
    return table.to_pylist()


def load_records_from_file(path: str | Path, errors: list[str] | None = None) -> list[Any]:
    path = Path(path)
    errors = errors if errors is not None else []
    suffix = path.suffix.lower()
    if path.name.endswith(".jsonl.gz"):
        return _read_jsonl(path, errors)
    if suffix == ".json":
        return _read_json(path)
    if suffix == ".jsonl":
        return _read_jsonl(path, errors)
    if suffix == ".txt":
        return _read_text(path)
    if suffix == ".parquet":
        return _read_parquet(path)
    raise ValueError(f"不支持的数据文件类型：{path.name}（支持 .json/.jsonl/.txt/.parquet）")


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_dir():
        files = []
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            if child.name.endswith(".jsonl.gz") or child.suffix.lower() in _SUPPORTED_SUFFIXES:
                files.append(child)
        if not files:
            raise FileNotFoundError(f"目录里没有找到可用的数据文件：{path}")
        return files
    if not path.exists():
        raise FileNotFoundError(f"数据路径不存在：{path}")
    return [path]


def load_records(
    path: str | Path,
    fmt: str = "auto",
    text_field: str = "text",
    max_samples: int | None = None,
    log: Any = None,
) -> list[Record]:
    """读取数据集中所有记录。

    ``log`` 可传一个 ``logging.Logger``；只在主进程打印进度，避免多卡刷屏。
    """
    path = Path(path)
    files = _iter_files(path)
    parse_errors: list[str] = []
    records: list[Record] = []
    skipped = 0

    for file_index, file in enumerate(files, 1):
        raw_rows = load_records_from_file(file, parse_errors)
        for row in raw_rows:
            record = parse_record(row, fmt=fmt, text_field=text_field, source=file.name)
            if record is None or not _is_usable(record):
                skipped += 1
            else:
                records.append(record)
            if max_samples is not None and len(records) >= max_samples:
                break
        if log is not None:
            log.info(
                "读取数据 %d/%d：%s（累计 %d 条）",
                file_index,
                len(files),
                file.name,
                len(records),
            )
        if max_samples is not None and len(records) >= max_samples:
            break

    if parse_errors:
        preview = "\n  ".join(parse_errors[:5])
        message = f"有 {len(parse_errors)} 行 JSON 解析失败，例如：\n  {preview}"
        if log is not None:
            log.warning(message)
        else:  # pragma: no cover
            print(message)
    if skipped and log is not None:
        log.warning("跳过 %d 条无法识别的记录（缺少 conversations/messages/text 字段）", skipped)
    if not records:
        raise ValueError(
            f"没有从 {path} 解析出任何可用样本。\n"
            "请确认格式为 ShareGPT：{\"conversations\": [{\"from\": \"human\", \"value\": ...}, ...]}"
        )
    return records


def _is_usable(record: Record) -> bool:
    if isinstance(record, TextDocument):
        return bool(record.text.strip())
    if not record.messages:
        return False
    return any(m.role == "assistant" and m.content.strip() for m in record.messages)
