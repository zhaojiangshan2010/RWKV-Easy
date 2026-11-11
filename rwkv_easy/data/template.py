"""对话模板：把 :class:`Conversation` 渲染成“文本片段 + 是否可训练”。

片段（segment）是关键设计：每个片段携带 ``trainable`` 标记，后续按 **字节区间**
映射到 token 上，从而精确实现 assistant-only loss mask。

注意：分词不是逐段可拼接的（"ab"+"cd" 与 "abcd" 可能切法不同），所以调用方必须
先把所有片段拼成一个完整字符串再做一次分词，而不是逐段分词后拼接。
"""

from __future__ import annotations

from dataclasses import dataclass

from .sharegpt import Conversation

__all__ = ["ROLE_DISPLAY", "ChatTemplate", "build_template"]

ROLE_DISPLAY = {"system": "System", "user": "User", "assistant": "Assistant"}

_TEMPLATES: dict[str, dict[str, str]] = {
    # RWKV 世界模型常用的对话格式
    "rwkv_world": {
        "system_prefix": "System: ",
        "user_prefix": "User: ",
        "assistant_prefix": "Assistant: ",
        "turn_separator": "\n\n",
    },
    # ChatML（部分社区 RWKV 模型使用）
    "chatml": {
        "system_prefix": "<|im_start|>system\n",
        "user_prefix": "<|im_start|>user\n",
        "assistant_prefix": "<|im_start|>assistant\n",
        "turn_separator": "<|im_end|>\n",
    },
}


@dataclass(frozen=True)
class ChatTemplate:
    system_prefix: str = "System: "
    user_prefix: str = "User: "
    assistant_prefix: str = "Assistant: "
    turn_separator: str = "\n\n"

    # ------------------------------------------------------------------ #
    # 渲染
    # ------------------------------------------------------------------ #

    def render(
        self,
        conversation: Conversation,
        train_on: str = "assistant",
        append_eos: str = "end",
        eos_text: str = "\x00",
    ) -> list[tuple[str, bool]]:
        """返回 ``[(text, trainable), ...]``，按顺序拼接即为完整训练文本。"""
        if not conversation.messages:
            return []

        messages = list(conversation.messages)
        last_assistant = -1
        for index, message in enumerate(messages):
            if message.role == "assistant":
                last_assistant = index

        segments: list[tuple[str, bool]] = []
        for index, message in enumerate(messages):
            content = message.content
            if message.role == "system":
                segments.append((self.system_prefix, False))
                segments.append((content + self.turn_separator, False))
                continue
            if message.role == "assistant":
                trainable = train_on != "last_assistant" or index == last_assistant
                segments.append((self.assistant_prefix, False))
                text = content + self.turn_separator
                if trainable and append_eos == "each_assistant":
                    text += eos_text
                segments.append((text, trainable))
                continue
            # user / tool / 其它：作为上下文
            segments.append((self.user_prefix, False))
            segments.append((content + self.turn_separator, False))

        # 只有对话以 assistant 结束时才补 EOS，否则等于让模型预测不存在的内容。
        if append_eos == "end" and last_assistant == len(messages) - 1:
            segments.append((eos_text, True))

        return [segment for segment in segments if segment[0]]

    def describe(self) -> str:
        return (
            f"system={self.system_prefix!r}, user={self.user_prefix!r}, "
            f"assistant={self.assistant_prefix!r}, sep={self.turn_separator!r}"
        )


def build_template(
    name: str = "rwkv_world",
    system_prefix: str | None = None,
    user_prefix: str | None = None,
    assistant_prefix: str | None = None,
    turn_separator: str | None = None,
) -> ChatTemplate:
    """按名字取内置模板，并允许逐字段覆盖。"""
    name = (name or "rwkv_world").lower()
    base = dict(_TEMPLATES.get(name, _TEMPLATES["rwkv_world"]))
    if system_prefix is not None:
        base["system_prefix"] = system_prefix
    if user_prefix is not None:
        base["user_prefix"] = user_prefix
    if assistant_prefix is not None:
        base["assistant_prefix"] = assistant_prefix
    if turn_separator is not None:
        base["turn_separator"] = turn_separator
    return ChatTemplate(**base)
