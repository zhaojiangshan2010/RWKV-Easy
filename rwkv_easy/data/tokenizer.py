"""RWKV World 词表分词器（零外部依赖）。

官方 RWKV-7 权重使用 ``rwkv_vocab_v20230424.txt``（65529 个 token，id 0 是 EOT）。
这里实现的是与官方一致的 **贪心最长匹配** 字节级分词：

1. 读取词表文件（每行 ``id <python 字节字面量> <字节长度>``）；
2. 按前两个字节分桶，桶内按长度降序排列；
3. 编码时优先匹配最长的 token，匹配不到则回退到单字节 token。

相比参考实现，这里用 :func:`ast.literal_eval` 代替 ``eval``（词表是可以被替换的
外部文件，不能当代码执行），并把桶内顺序显式按长度排序，保证“最长匹配”成立。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

__all__ = ["DEFAULT_VOCAB_PATH", "RWKVTokenizer", "get_tokenizer"]

DEFAULT_VOCAB_PATH = Path(__file__).resolve().parent.parent / "assets" / "rwkv_vocab_v20230424.txt"

# 官方世界词表从 id 1 开始（id 0 未使用），b"\x00" 的 id 由词表决定，通常是 1。
# 这里不写死，加载后通过 ``tokenizer.eot_id`` 获取。

_LINE_RE = re.compile(r"^\s*(\d+)\s+(.*?)\s+(\d+)\s*$")


class RWKVTokenizer:
    """RWKV World 词表分词器。"""

    def __init__(self, vocab_path: str | Path | None = None) -> None:
        self.vocab_path = Path(vocab_path) if vocab_path else DEFAULT_VOCAB_PATH
        if not self.vocab_path.exists():
            raise FileNotFoundError(
                f"找不到 RWKV 词表文件：{self.vocab_path}\n"
                "如果是源码运行，请确认 rwkv_easy/assets/rwkv_vocab_v20230424.txt 存在；"
                "也可以通过 data.vocab_path 指定。"
            )

        self.idx2token: dict[int, bytes] = {}
        self.token2idx: dict[bytes, int] = {}
        self._load()

        # 单字节 token
        self._single: dict[int, int] = {}
        for token, idx in self.token2idx.items():
            if len(token) == 1:
                self._single[token[0]] = idx

        # 双字节桶：bucket[(b0, b1)] = [token, ...]，按长度降序
        buckets: dict[tuple[int, int], list[bytes]] = {}
        for token in self.token2idx:
            if len(token) >= 2:
                buckets.setdefault((token[0], token[1]), []).append(token)
        for items in buckets.values():
            items.sort(key=len, reverse=True)
        self._buckets = {key: tuple(items) for key, items in buckets.items()}
        self._max_token_len = max((len(t) for t in self.token2idx), default=1)

        missing_single = [i for i in range(256) if i not in self._single]
        if missing_single:
            raise ValueError(
                "词表不完整：缺少单字节 token "
                f"{missing_single[:8]}...（共 {len(missing_single)} 个）。"
                "请确认使用的是官方 rwkv_vocab_v20230424.txt。"
            )

        if b"\x00" not in self.token2idx:
            raise ValueError("词表缺少 b'\\x00'（EOT token），请检查词表文件。")
        self.eot_id: int = int(self.token2idx[b"\x00"])

    # ------------------------------------------------------------------ #
    # 加载
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        with self.vocab_path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                match = _LINE_RE.match(line)
                if not match:
                    raise ValueError(f"词表第 {lineno} 行格式不正确：{line!r}")
                idx_text, literal, length_text = match.groups()
                try:
                    raw = ast.literal_eval(literal)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(f"词表第 {lineno} 行无法解析：{literal!r}") from exc
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                if not isinstance(raw, bytes):
                    raise ValueError(f"词表第 {lineno} 行不是字符串/字节串：{literal!r}")
                expected = int(length_text)
                if len(raw) != expected:
                    raise ValueError(
                        f"词表第 {lineno} 行长度不符：声明 {expected}，实际 {len(raw)}"
                    )
                idx = int(idx_text)
                self.idx2token[idx] = raw
                # 官方词表按 id 递增排列；若出现重复字节串，保留最小 id。
                self.token2idx.setdefault(raw, idx)

    # ------------------------------------------------------------------ #
    # 属性
    # ------------------------------------------------------------------ #

    @property
    def vocab_size(self) -> int:
        return max(self.idx2token) + 1

    @property
    def max_token_len(self) -> int:
        return self._max_token_len

    def token_text(self, token_id: int) -> bytes:
        return self.idx2token[token_id]

    # ------------------------------------------------------------------ #
    # 编码
    # ------------------------------------------------------------------ #

    def encode_bytes(self, src: bytes) -> list[int]:
        return [token for token, _, _ in self.encode_with_offsets(src)]

    def encode(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def encode_with_offsets(self, src: bytes) -> list[tuple[int, int, int]]:
        """返回 ``[(token_id, start_byte, end_byte), ...]``。

        偏移量用于把“哪些字符属于 assistant 回复”精确映射到 token 上——
        分词不是逐段可拼接的，先拼字符串再一次性分词才能保证 mask 正确。
        """
        out: list[tuple[int, int, int]] = []
        i = 0
        n = len(src)
        bucket_get = self._buckets.get
        token2idx = self.token2idx
        single = self._single
        while i < n:
            if i + 1 < n:
                bucket = bucket_get((src[i], src[i + 1]))
                if bucket is not None:
                    matched: bytes | None = None
                    # 桶内按长度降序 => 第一个 startswith 命中的就是最长匹配。
                    for token in bucket:
                        if src.startswith(token, i):
                            matched = token
                            break
                    if matched is not None:
                        out.append((token2idx[matched], i, i + len(matched)))
                        i += len(matched)
                        continue
            idx = single.get(src[i])
            if idx is None:  # pragma: no cover - 构造词表时已校验
                raise ValueError(f"字节 0x{src[i]:02x} 不在词表中（offset {i}）")
            out.append((idx, i, i + 1))
            i += 1
        return out

    # ------------------------------------------------------------------ #
    # 解码
    # ------------------------------------------------------------------ #

    def decode_bytes(self, tokens: list[int] | tuple[int, ...]) -> bytes:
        return b"".join(self.idx2token[int(t)] for t in tokens)

    def decode(self, tokens: list[int] | tuple[int, ...], errors: str = "replace") -> str:
        return self.decode_bytes(tokens).decode("utf-8", errors=errors)

    # ------------------------------------------------------------------ #
    # 便捷方法
    # ------------------------------------------------------------------ #

    def encode_document(self, text: str) -> list[int]:
        return self.encode(text)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"RWKVTokenizer(vocab_size={self.vocab_size}, path={self.vocab_path.name})"


_TOKENIZER_CACHE: dict[tuple[str, float], RWKVTokenizer] = {}


def get_tokenizer(vocab_path: str | Path | None = None) -> RWKVTokenizer:
    """按路径 + mtime 缓存，避免每次构建数据集都重新解析 65k 行词表。"""
    path = Path(vocab_path) if vocab_path else DEFAULT_VOCAB_PATH
    key = (str(path.resolve()), path.stat().st_mtime if path.exists() else 0.0)
    tokenizer = _TOKENIZER_CACHE.get(key)
    if tokenizer is None:
        tokenizer = RWKVTokenizer(path)
        _TOKENIZER_CACHE.clear()
        _TOKENIZER_CACHE[key] = tokenizer
    return tokenizer
