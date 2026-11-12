"""交互式一键配置向导。

设计目标：

* 小白只需要运行 ``rwkv-easy wizard``（或项目根目录的 ``quickstart``），
  用方向键选择、空格多选，就能生成完整 YAML 并开始训练；
* Windows / Linux / macOS 都不依赖第三方 TUI 库：
  Windows 用 ``msvcrt``，Unix 用 ``termios``；非 TTY 时退化成分行问答；
* 纯配置逻辑与终端 UI 分开，便于单元测试。
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import Config, ConfigError

__all__ = [
    "PRESETS",
    "DatasetInfo",
    "Environment",
    "ModelInfo",
    "Preset",
    "WizardState",
    "apply_preset",
    "cuda_tag_candidates",
    "default_config",
    "detect_environment",
    "run_wizard",
    "text_width",
]

DEFAULT_MODULES = [
    "att.receptance",
    "att.key",
    "att.value",
    "att.output",
    "ffn.key",
    "ffn.value",
]
MODULE_LABELS = {
    "att.receptance": "att.receptance  （时间混合接收门）",
    "att.key": "att.key         （时间混合 key）",
    "att.value": "att.value       （时间混合 value）",
    "att.output": "att.output      （时间混合输出）",
    "ffn.key": "ffn.key         （channel-mix key）",
    "ffn.value": "ffn.value       （channel-mix value）",
}
SUPPORTED_MODEL_SUFFIXES = (".pth", ".safetensors")
SUPPORTED_DATA_SUFFIXES = (".json", ".jsonl", ".jsonl.gz", ".txt", ".parquet")


class WizardCancelled(Exception):
    """用户按 q / Esc 取消。"""


class WizardError(RuntimeError):
    """可读的向导错误。"""


# --------------------------------------------------------------------------- #
# 文本宽度（中文按 2 列算，保证 TUI 对齐）
# --------------------------------------------------------------------------- #

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _char_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def text_width(text: str) -> int:
    """可见宽度：忽略 ANSI 颜色码，中日韩全角字符按 2 列。"""
    return sum(_char_width(char) for char in _ANSI_RE.sub("", text))


def _truncate(text: str, width: int, ellipsis: str = "…") -> str:
    if width <= 0:
        return ""
    if text_width(text) <= width:
        return text
    limit = max(0, width - text_width(ellipsis))
    used = 0
    result: list[str] = []
    for char in text:
        char_width = _char_width(char)
        if used + char_width > limit:
            break
        result.append(char)
        used += char_width
    return "".join(result) + ellipsis


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - text_width(text))


_CJK_PUNCTUATION = "，。！？；：、（）【】《》“”‘’「」『』…—"
_CLOSING_PUNCTUATION = "，。！？；：、）】》”’」』…"


def _wrap_tokens(paragraph: str) -> list[str]:
    """把段落切成“ASCII 单词 / 单个中文字 / 空格 / 中文标点”。"""
    tokens: list[str] = []
    buffer: list[str] = []
    for char in paragraph:
        if char.isspace():
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append(" ")
        elif _char_width(char) == 2 or char in _CJK_PUNCTUATION:
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append(char)
        else:
            buffer.append(char)
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def _wrap_text(text: str, width: int) -> list[str]:
    """按可见宽度折行；中文可断，英文单词和路径尽量保持完整。"""
    if width <= 1:
        return [text]
    lines: list[str] = []
    for paragraph in str(text).splitlines():
        if not paragraph:
            lines.append("")
            continue
        current = ""
        used = 0
        for token in _wrap_tokens(paragraph):
            if token == " ":
                if not current:
                    continue
                if used + 1 > width:
                    lines.append(current.rstrip())
                    current = ""
                    used = 0
                else:
                    current += " "
                    used += 1
                continue
            token_width = text_width(token)
            if token in _CLOSING_PUNCTUATION and current:
                # 避免“下一行以逗号/句号开头”这种难看的中文折行。
                current += token
                used += token_width
                continue
            if token_width > width:
                # 超长 ASCII 串（例如路径）只能硬断，避免整行溢出。
                if current:
                    lines.append(current.rstrip())
                    current = ""
                    used = 0
                for char in token:
                    char_width = _char_width(char)
                    if used + char_width > width and current:
                        lines.append(current.rstrip())
                        current = ""
                        used = 0
                    current += char
                    used += char_width
                continue
            if used + token_width > width and current:
                lines.append(current.rstrip())
                current = ""
                used = 0
            current += token
            used += token_width
        lines.append(current.rstrip())
    return lines or [""]


# --------------------------------------------------------------------------- #
# 环境检测
# --------------------------------------------------------------------------- #


@dataclass
class Environment:
    os_name: str
    python: str
    python_version: str
    device: str
    torch_version: str | None
    torch_cuda: str | None
    cuda_available: bool
    gpu_count: int = 0
    gpu_name: str | None = None
    gpu_memory_gib: float | None = None
    driver_version: str | None = None
    nvidia_cuda: str | None = None
    fla_available: bool = False
    deepspeed_available: bool = False

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def device_label(self) -> str:
        if self.is_cuda:
            memory = f"，{self.gpu_memory_gib:.1f} GiB" if self.gpu_memory_gib else ""
            count = f" × {self.gpu_count}" if self.gpu_count > 1 else ""
            return f"CUDA：{self.gpu_name or 'GPU'}{count}{memory}"
        return "CPU（只建议做自检；训练会很慢）"


def _run_capture(cmd: Sequence[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
        return result.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _nvidia_smi_info() -> dict[str, Any]:
    query = _run_capture(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    info: dict[str, Any] = {}
    first = query.strip().splitlines()[0] if query.strip() else ""
    if first:
        parts = [part.strip() for part in first.split(",")]
        if parts:
            info["gpu_name"] = parts[0]
        if len(parts) >= 2:
            match = re.search(r"([\d.]+)", parts[1])
            if match:
                info["gpu_memory_gib"] = float(match.group(1)) / 1024.0
        if len(parts) >= 3:
            info["driver_version"] = parts[2]
    text = _run_capture(["nvidia-smi"])
    match = re.search(r"CUDA Version:\s*([\d.]+)", text)
    if match:
        info["nvidia_cuda"] = match.group(1)
    return info


def detect_environment() -> Environment:
    """检测当前 Python / torch / GPU 状态（不安装任何东西）。"""
    smi = _nvidia_smi_info()
    torch_version = getattr(torch, "__version__", None)
    torch_cuda = getattr(torch.version, "cuda", None)
    cuda_available = bool(torch.cuda.is_available())
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpu_name = None
    gpu_memory = smi.get("gpu_memory_gib")
    if cuda_available:
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_memory = props.total_memory / 1024**3
        except Exception:  # pragma: no cover - 驱动异常
            gpu_name = smi.get("gpu_name")

    fla_available = False
    try:
        import rwkvfla  # noqa: F401
        from rwkvfla.ops.rwkv7 import chunk_rwkv7  # noqa: F401

        fla_available = True
    except Exception:
        fla_available = False

    try:
        import deepspeed  # noqa: F401

        deepspeed_available = True
    except Exception:
        deepspeed_available = False

    device = "cuda" if cuda_available else "cpu"
    return Environment(
        os_name=os.name,
        python=sys.executable,
        python_version=sys.version.split()[0],
        device=device,
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpu_name=gpu_name or smi.get("gpu_name"),
        gpu_memory_gib=gpu_memory,
        driver_version=smi.get("driver_version"),
        nvidia_cuda=smi.get("nvidia_cuda"),
        fla_available=fla_available,
        deepspeed_available=deepspeed_available,
    )


def cuda_tag_candidates(version: str | None) -> list[str]:
    """把驱动报告的 CUDA 版本映射成 PyTorch wheel 标签（新→旧）。"""
    supported = [
        ((13, 0), "cu130"),
        ((12, 8), "cu128"),
        ((12, 6), "cu126"),
        ((12, 4), "cu124"),
        ((12, 1), "cu121"),
        ((11, 8), "cu118"),
    ]
    if version:
        match = re.match(r"^(\d+)\.(\d+)", version.strip())
        if match:
            target = (int(match.group(1)), int(match.group(2)))
            tags = [tag for (ver, tag) in supported if ver <= target]
            return tags or ["cu118"]
    return [tag for _, tag in supported]


# --------------------------------------------------------------------------- #
# 路径 / 数据 / 模型
# --------------------------------------------------------------------------- #


def _expand(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def _human_size(num: int) -> str:
    size = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num} B"


def _short_path(path: str, limit: int = 56) -> str:
    """摘要页只保留路径尾部（文件名最有辨识度），完整路径仍在 YAML / 编辑页。"""
    if not path:
        return "未选择"
    if len(path) <= limit:
        return path
    return "…" + path[-(limit - 1):]


@dataclass
class ModelInfo:
    path: str
    n_layer: int
    n_embd: int
    n_head: int
    head_size: int
    vocab_size: int
    file_size: int

    @property
    def summary(self) -> str:
        return (
            f"{self.n_layer} 层 / {self.n_embd} 维 / {self.n_head} 头 / "
            f"vocab {self.vocab_size} / {_human_size(self.file_size)}"
        )


@dataclass
class DatasetInfo:
    path: str
    kind: str
    detail: str


def validate_model_path(path: str) -> Path:
    target = _expand(path)
    if not target.exists():
        raise WizardError(f"模型文件不存在：{target}")
    if not target.is_file():
        raise WizardError(f"模型路径不是文件：{target}")
    if target.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
        raise WizardError("模型只支持 .pth / .safetensors，收到：" + target.name)
    if target.stat().st_size <= 0:
        raise WizardError("模型文件是空的：" + str(target))
    return target


def inspect_model(path: str) -> ModelInfo:
    """读取权重结构（mmap，不复制全部张量）。失败时抛出 WizardError。"""
    from .models.checkpoint import infer_config, read_checkpoint

    target = validate_model_path(path)
    state: Any = None
    try:
        state = read_checkpoint(target)
        cfg = infer_config(state)
    except Exception as exc:
        raise WizardError(
            "这个文件不像 RWKV-7 权重，无法推断结构：\n"
            f"  {exc}\n"
            "请确认下载的是官方 state_dict（不是 LoRA adapter / HF 目录）。"
        ) from exc
    finally:
        state = None
    return ModelInfo(
        path=str(target),
        n_layer=cfg.n_layer,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        head_size=cfg.head_size,
        vocab_size=cfg.vocab_size,
        file_size=target.stat().st_size,
    )


def _peek_dataset(target: Path) -> str:
    """轻量探测数据集格式，不完整加载大文件。"""
    files = [target]
    if target.is_dir():
        files = sorted(
            child
            for child in target.rglob("*")
            if child.is_file()
            and (child.name.endswith(".jsonl.gz") or child.suffix.lower() in SUPPORTED_DATA_SUFFIXES)
        )
        if not files:
            raise WizardError("目录里没有找到 .json/.jsonl/.txt/.parquet 数据文件：" + str(target))
    first = files[0]
    suffix = first.suffix.lower()
    if first.name.endswith(".jsonl.gz"):
        return "text/jsonl（gzip）"
    if suffix == ".txt":
        with first.open("r", encoding="utf-8", errors="replace") as handle:
            sample = handle.read(200).strip()
        return "纯文本" if sample else "空文本"
    if suffix == ".jsonl":
        with first.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    return _describe_row(row)
        return "空 jsonl"
    if suffix == ".json" and first.stat().st_size <= 32 * 1024 * 1024:
        data = json.loads(first.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return _describe_row(data[0])
        if isinstance(data, dict):
            return _describe_row(data)
        return "JSON"
    if suffix == ".parquet":
        return "parquet（训练时读取）"
    return suffix.lstrip(".")


def _describe_row(row: Any) -> str:
    if not isinstance(row, dict):
        return "JSON 数组"
    keys = set(row)
    if "conversations" in keys:
        return "ShareGPT（conversations）"
    if "messages" in keys:
        return "OpenAI messages"
    if "text" in keys:
        return "纯文本（text 字段）"
    if ("prompt" in keys and "response" in keys) or ("instruction" in keys and "output" in keys):
        return "两列指令（prompt/response）"
    return "未知字段：" + ", ".join(sorted(keys)[:6])


def validate_dataset_path(path: str) -> DatasetInfo:
    target = _expand(path)
    if not target.exists():
        raise WizardError(f"数据集不存在：{target}")
    if target.is_file() and target.stat().st_size <= 0:
        raise WizardError("数据集文件是空的：" + str(target))
    if target.is_dir() and not any(target.iterdir()):
        raise WizardError("数据集目录是空的：" + str(target))
    try:
        kind = _peek_dataset(target)
    except WizardError:
        raise
    except Exception as exc:
        raise WizardError(f"无法识别数据集格式：{exc}") from exc
    detail = kind
    if target.is_file():
        detail += f" / {_human_size(target.stat().st_size)}"
    return DatasetInfo(path=str(target), kind=kind, detail=detail)


def find_model_candidates() -> list[Path]:
    roots = [Path("models"), Path("examples"), Path.cwd()]
    result: dict[Path, tuple[int, float]] = {}
    for priority, root in enumerate(roots):
        if not root.exists() or not root.is_dir():
            continue
        for path in root.glob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES:
                result[path.resolve()] = (priority, path.stat().st_mtime)
    return [
        path
        for path, _ in sorted(result.items(), key=lambda item: (item[1][0], -item[1][1]))
    ][:20]


def find_dataset_candidates() -> list[Path]:
    roots = [Path("data"), Path("examples"), Path.cwd()]
    result: dict[Path, tuple[int, float]] = {}
    for priority, root in enumerate(roots):
        if not root.exists() or not root.is_dir():
            continue
        for path in root.iterdir():
            if path.name.startswith("."):
                continue
            if path.is_dir():
                # 只把 data/、examples/ 下的目录当成数据集；cwd 下不扫描所有目录
                if root.name in ("data", "examples"):
                    result[path.resolve()] = (priority, path.stat().st_mtime)
            elif path.name.endswith(".jsonl.gz") or path.suffix.lower() in SUPPORTED_DATA_SUFFIXES:
                # cwd 下只自动识别 JSON/JSONL，避免把 requirements.txt 当成数据
                if root == Path.cwd() and path.suffix.lower() not in (".json", ".jsonl"):
                    continue
                rank = priority
                if root.name == "examples" and path.name != "sharegpt_tiny.json":
                    rank += 1  # 示例数据里优先推荐通用的 sharegpt_tiny
                result[path.resolve()] = (rank, path.stat().st_mtime)
    return [
        path
        for path, _ in sorted(result.items(), key=lambda item: (item[1][0], -item[1][1]))
    ][:20]


# --------------------------------------------------------------------------- #
# 配置默认值
# --------------------------------------------------------------------------- #


def _size_batch(file_size: int) -> tuple[int, bool]:
    """根据权重文件大小给出 batch / 梯度检查点建议。"""
    if file_size < 1_200_000_000:  # ~0.5B bf16 以下
        return 8, False
    if file_size < 3_000_000_000:
        return 4, True
    if file_size < 8_000_000_000:
        return 2, True
    return 1, True


def default_config(env: Environment) -> Config:
    cfg = Config()
    cfg.run.name = "quickstart-lora"
    cfg.model.decay = "auto"
    cfg.model.backend = "fla" if env.is_cuda and env.fla_available else "native"
    cfg.train.method = "lora"
    cfg.train.infctx = False
    cfg.train.ctx_len = 2048
    cfg.train.chunk_ctx = 512
    cfg.train.micro_batch_size = 4 if env.is_cuda else 1
    cfg.train.grad_accum = 1
    cfg.train.epochs = 1
    cfg.train.lr = 1.0e-4
    cfg.train.min_lr = 1.0e-5
    cfg.train.warmup_steps = 10
    cfg.train.precision = "bf16-full" if env.is_cuda else "fp32"
    cfg.train.grad_checkpoint = not env.is_cuda
    cfg.train.compile = bool(env.is_cuda)
    cfg.train.device = "auto" if env.is_cuda else "cpu"
    cfg.train.log_every = 10
    cfg.train.save_every_steps = 0
    cfg.train.save_every_epochs = 1
    cfg.train.keep_last = 2
    cfg.train.merge_at_end = True
    cfg.lora.target_modules = list(DEFAULT_MODULES)
    cfg.lora.target_layers = None
    cfg.data.mode = "packed"
    cfg.data.num_workers = 0
    cfg.data.num_proc = 2 if env.is_cuda else 0
    cfg.data.cache = True
    cfg.data.val_ratio = 0.0
    cfg.dist.mode = "auto" if env.gpu_count > 1 else "single"
    cfg.log.jsonl = True
    return cfg


def normalize_for_environment(cfg: Config, env: Environment) -> None:
    """把明显不兼容的组合收敛到可运行值。"""
    if not env.is_cuda:
        cfg.train.precision = "fp32"
        cfg.model.backend = "native"
        cfg.train.compile = False
        cfg.train.device = "cpu"
        cfg.dist.mode = "single"
        cfg.dist.num_gpus = 1
    else:
        if cfg.model.backend == "fla" and not env.fla_available:
            cfg.model.backend = "native"
        if cfg.train.precision == "fp32":
            pass  # 用户显式选择 fp32 时保留
    if cfg.train.infctx and cfg.data.mode == "sample":
        cfg.data.mode = "packed"
    if cfg.train.infctx and cfg.train.chunk_ctx > cfg.train.ctx_len:
        cfg.train.chunk_ctx = min(2048, cfg.train.ctx_len)
    # 不满足逐块反传条件的组合自动收敛，避免向导里出现无法启动的配置。
    if cfg.train.infctx and not cfg.train.infctx_detach:
        cfg.train.infctx_parallel_backward = False
    if cfg.dist.mode == "deepspeed":
        cfg.train.infctx_parallel_backward = False


def apply_method_defaults(cfg: Config, method: str, infctx: bool) -> None:
    method = "state" if method == "state" else "lora"
    cfg.train.method = method
    if method == "lora":
        cfg.train.lr = 1.0e-4
        cfg.train.min_lr = 1.0e-5
        cfg.train.weight_decay = 0.01
        if not cfg.lora.target_modules:
            cfg.lora.target_modules = list(DEFAULT_MODULES)
        cfg.run.name = "quickstart-lora" + ("-infctx" if infctx else "")
    else:
        cfg.train.lr = 5.0e-3
        cfg.train.min_lr = 5.0e-4
        cfg.train.weight_decay = 0.0
        cfg.run.name = "quickstart-state" + ("-infctx" if infctx else "")
    cfg.train.infctx = bool(infctx)
    if infctx:
        cfg.train.ctx_len = max(8192, cfg.train.ctx_len)
        cfg.train.chunk_ctx = 2048
        cfg.train.grad_checkpoint = False
        cfg.train.infctx_backward = True
        cfg.train.infctx_parallel_backward = method == "lora"
        cfg.data.mode = "packed"
        cfg.train.micro_batch_size = min(cfg.train.micro_batch_size, 2)
    else:
        cfg.train.ctx_len = 2048
        cfg.train.chunk_ctx = 512
        cfg.train.grad_checkpoint = not (cfg.train.micro_batch_size >= 4)


def apply_model_defaults(cfg: Config, info: ModelInfo, cuda: bool = True) -> None:
    if not cuda:
        # CPU 只建议自检：小 batch + 梯度检查点，避免一上来就卡住。
        cfg.train.micro_batch_size = 1
        cfg.train.grad_checkpoint = True
        return
    bsz, gc = _size_batch(info.file_size)
    if cfg.train.infctx:
        bsz = min(bsz, 2)
        gc = False
    cfg.train.micro_batch_size = bsz
    cfg.train.grad_checkpoint = gc


@dataclass(frozen=True)
class Preset:
    """新手模板：只描述“场景 → 训练配置”，不绑定具体模型/数据。"""

    key: str
    title: str
    tag: str
    summary: str
    description: str
    method: str = "lora"
    infctx: bool = False
    low_memory: bool = False
    quick_test: bool = False
    run_name: str | None = None


PRESETS: tuple[Preset, ...] = (
    Preset(
        key="recommended",
        title="LoRA 标准微调",
        tag="推荐",
        summary="大多数对话 / 指令微调",
        description=(
            "第一次用就选这个。LoRA 只训练低秩矩阵，效果稳定、显存友好；"
            "会根据显卡和权重大小自动选 bf16-full、compile 和 batch。"
        ),
        method="lora",
        infctx=False,
    ),
    Preset(
        key="lora-infctx",
        title="长上下文 LoRA",
        tag="长文",
        summary="8K+ 长对话、长文档、长记忆",
        description=(
            "开启 infctx 分块训练，可处理 8K–32K 上下文。显存只与 chunk_ctx 有关，"
            "不会因为上下文变长而暴涨；速度比整段训练略低。"
        ),
        method="lora",
        infctx=True,
    ),
    Preset(
        key="state",
        title="State Tuning 快速适配",
        tag="轻量",
        summary="小数据快速适配，参数 < 0.1%",
        description=(
            "只训练每层的初始循环状态 time_state，适配器只有几百 KB。"
            "适合风格/人设/小知识快速注入；数据量很大时效果不如 LoRA。"
        ),
        method="state",
        infctx=False,
    ),
    Preset(
        key="state-infctx",
        title="长上下文 State",
        tag="轻量+长文",
        summary="长上下文 + 极小可训练参数",
        description=(
            "State Tuning 叠加 infctx。只有第一块需要反传，训练很快、显存很省；"
            "适合长文本上的轻量适配。"
        ),
        method="state",
        infctx=True,
    ),
    Preset(
        key="low-memory",
        title="省显存模式",
        tag="低显存",
        summary="8–12GB 显卡 / 大模型也能跑",
        description=(
            "batch 降到 1、开启梯度检查点、关闭 compile，用更多梯度累积保持总 batch。"
            "速度会慢一些，但最不容易一上来就 OOM。"
        ),
        method="lora",
        infctx=False,
        low_memory=True,
    ),
    Preset(
        key="quick-test",
        title="先跑 20 步试试",
        tag="试跑",
        summary="验证模型、数据和训练链路",
        description=(
            "只训练 20 步并自动合并，几分钟内确认“数据 → loss → 保存 → 推理”是否正常。"
            "确认没问题后，再换回标准模板正式训练。"
        ),
        method="lora",
        infctx=False,
        quick_test=True,
        run_name="quick-test",
    ),
    Preset(
        key="custom",
        title="自定义",
        tag="高级",
        summary="保持当前参数，不套用模板",
        description="不修改任何训练参数；适合已经知道自己要什么的用户。",
    ),
)

PRESETS_BY_KEY = {preset.key: preset for preset in PRESETS}


def get_preset(key: str) -> Preset:
    return PRESETS_BY_KEY.get(key, PRESETS_BY_KEY["recommended"])


def apply_preset(state: WizardState, key: str) -> None:
    """套用模板，保留模型/数据/输出目录等用户上下文。"""
    preset = PRESETS_BY_KEY.get(key)
    if preset is None:
        raise WizardError(f"未知模板：{key}")
    if preset.key == "custom":
        state.preset_key = "custom"
        state.notice = "已切换为自定义：继续修改当前参数"
        return
    cfg = state.cfg
    apply_method_defaults(cfg, preset.method, preset.infctx)
    normalize_for_environment(cfg, state.env)
    if state.model_info is not None:
        apply_model_defaults(cfg, state.model_info, cuda=state.env.is_cuda)
    if preset.low_memory:
        cfg.train.micro_batch_size = 1
        cfg.train.grad_accum = max(8, cfg.train.grad_accum)
        cfg.train.grad_checkpoint = True
        cfg.train.compile = False
        if cfg.train.infctx:
            cfg.train.chunk_ctx = min(1024, cfg.train.ctx_len)
    if preset.quick_test:
        cfg.train.max_steps = 20
        cfg.train.log_every = 1
        cfg.train.save_every_steps = 0
        cfg.train.save_every_epochs = 0
        cfg.train.warmup_steps = 5
        cfg.train.merge_at_end = True
    else:
        cfg.train.max_steps = None
    if preset.run_name:
        cfg.run.name = preset.run_name
    elif preset.low_memory:
        cfg.run.name = f"{cfg.run.name}-lowmem"
    state.preset_key = preset.key
    state.notice = f"已套用模板：{preset.tag} {preset.title}"


@dataclass
class WizardState:
    env: Environment
    cfg: Config
    model_info: ModelInfo | None = None
    dataset_info: DatasetInfo | None = None
    preset_key: str = "recommended"
    notice: str = ""

    def preset_label(self) -> str:
        preset = get_preset(self.preset_key)
        return f"[{preset.tag}] {preset.title}"

    def mark_custom(self, notice: str = "当前参数已改为自定义") -> None:
        if self.preset_key != "custom":
            self.preset_key = "custom"
            self.notice = notice

    def set_method(self, method: str) -> None:
        apply_method_defaults(self.cfg, method, self.cfg.train.infctx)
        normalize_for_environment(self.cfg, self.env)
        self.mark_custom("训练方式已修改")

    def set_infctx(self, infctx: bool) -> None:
        apply_method_defaults(self.cfg, self.cfg.train.method, infctx)
        normalize_for_environment(self.cfg, self.env)
        self.mark_custom("长上下文设置已修改")

    def set_model(self, info: ModelInfo) -> None:
        self.model_info = info
        self.cfg.model.path = info.path
        apply_model_defaults(self.cfg, info, cuda=self.env.is_cuda)

    def set_dataset(self, info: DatasetInfo) -> None:
        self.dataset_info = info
        self.cfg.data.path = info.path

    def tokens_per_step(self) -> int:
        return int(
            self.cfg.train.micro_batch_size
            * max(1, int(self.cfg.train.grad_accum))
            * int(self.cfg.train.ctx_len)
        )

    def method_label(self) -> str:
        return "State Tuning" if self.cfg.train.method == "state" else "LoRA"

    def target_summary(self) -> str:
        if self.cfg.train.method != "lora":
            return "（State Tuning 不需要 LoRA 模块）"
        layers = self.cfg.lora.target_layers
        layer_text = "全部层" if layers is None else f"{len(layers)} 层"
        return f"{len(self.cfg.lora.target_modules)} 个模块 / {layer_text}"

    def modules_summary(self) -> str:
        if self.cfg.train.method != "lora":
            return "—"
        return f"{len(self.cfg.lora.target_modules)} 个模块"

    def layers_summary(self) -> str:
        if self.cfg.train.method != "lora":
            return "—"
        layers = self.cfg.lora.target_layers
        if layers is None:
            return "全部层"
        return f"{len(layers)} 层（{', '.join(map(str, layers[:6]))}{'…' if len(layers) > 6 else ''}）"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.cfg.model.path:
            errors.append("请选择模型权重（model.path）")
        else:
            try:
                validate_model_path(self.cfg.model.path)
            except WizardError as exc:
                errors.append(str(exc))
        if not self.cfg.data.path:
            errors.append("请选择数据集（data.path）")
        else:
            try:
                validate_dataset_path(self.cfg.data.path)
            except WizardError as exc:
                errors.append(str(exc))
        if self.cfg.train.infctx and self.cfg.data.mode == "sample":
            errors.append("infctx 需要 data.mode: packed")
        if self.cfg.train.infctx and self.cfg.train.chunk_ctx > self.cfg.train.ctx_len:
            errors.append("chunk_ctx 不能大于 ctx_len")
        if not self.env.is_cuda and self.cfg.train.precision != "fp32":
            errors.append("CPU 上必须使用 precision: fp32")
        if self.cfg.model.backend == "fla" and not self.env.fla_available:
            errors.append("选择了 fla 后端，但当前环境没有可导入的 rwkv-fla")
        if self.cfg.dist.mode == "deepspeed" and not self.env.deepspeed_available:
            errors.append("选择了 DeepSpeed，但当前环境没有安装 deepspeed")
        try:
            self.cfg.validate()
        except ConfigError as exc:
            errors.append(str(exc))
        return errors


def _parallel_backward_label(state: WizardState) -> str | None:
    cfg = state.cfg
    if not cfg.train.infctx:
        return None
    mode = cfg.dist.mode
    multi = mode == "ddp" or mode == "deepspeed" or (mode == "auto" and state.env.gpu_count > 1)
    if not multi:
        return None
    if mode == "deepspeed":
        return "最后统一（DeepSpeed）"
    if cfg.train.method == "state":
        return "最后统一（State）"
    return "逐块（最后 all-reduce）" if cfg.train.infctx_parallel_backward else "最后统一"


def config_summary(state: WizardState, compact: bool = False) -> str:
    cfg = state.cfg
    model_path = _short_path(cfg.model.path or "") if compact else (cfg.model.path or "未选择")
    data_path = _short_path(cfg.data.path) if compact else (cfg.data.path or "未选择")
    method = f"{state.method_label()}{' + infctx' if cfg.train.infctx else ''}"
    context = f"ctx {cfg.train.ctx_len}" + (
        f" / chunk {cfg.train.chunk_ctx}" if cfg.train.infctx else ""
    )
    steps = f"epochs {cfg.train.epochs}" + (
        f" / max_steps {cfg.train.max_steps}" if cfg.train.max_steps else ""
    )
    output = f"{cfg.run.output_dir}/{cfg.run.name}"
    if compact:
        lines = [
            f"训练方式    : {method}",
            f"模型        : {model_path}",
            f"数据集      : {data_path}",
            f"后端 / 精度 : {cfg.model.backend} / {cfg.train.precision}",
            f"batch / ctx : {cfg.train.micro_batch_size} × {cfg.train.grad_accum}"
            f" → {state.tokens_per_step()} token/step / {context}",
            f"学习率      : {cfg.train.lr:g} → {cfg.train.min_lr:g}（{cfg.train.lr_schedule}）",
            f"步数 / 输出 : {steps} / {output}",
        ]
        parallel = _parallel_backward_label(state)
        if parallel is not None:
            lines.append(f"多卡反传    : {parallel}")
        if cfg.train.method == "lora":
            lines.append(f"LoRA        : {state.target_summary()}")
        else:
            lines.append(f"State       : init_std={cfg.state.init_std}")
        return "\n".join(lines)

    lines = [
        f"训练方式    : {method}",
        f"模型        : {model_path}",
        f"数据集      : {data_path}",
        f"后端 / 精度 : {cfg.model.backend} / {cfg.train.precision}",
        f"batch       : micro {cfg.train.micro_batch_size} × accum {cfg.train.grad_accum}"
        f" → {state.tokens_per_step()} token/step",
        f"上下文      : {context}",
        f"学习率      : {cfg.train.lr:g} → {cfg.train.min_lr:g}（{cfg.train.lr_schedule}）",
        f"步数        : {steps}",
        f"compile     : {cfg.train.compile} ({cfg.train.compile_mode})",
        f"输出目录    : {output}",
    ]
    parallel = _parallel_backward_label(state)
    if parallel is not None:
        lines.append(f"多卡反传    : {parallel}")
    if cfg.train.method == "lora":
        lines.append(f"LoRA        : {state.target_summary()}")
        lines.append(f"LoRA 参数   : r={cfg.lora.rank}, alpha={cfg.lora.alpha}, dropout={cfg.lora.dropout}")
    else:
        lines.append(f"State       : init_std={cfg.state.init_std}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 终端 UI
# --------------------------------------------------------------------------- #


class Console:
    """最小跨平台 TUI：方向键 / 空格 / 回车；非 TTY 时退化成分行问答。"""

    def __init__(self, non_interactive: bool = False, stream: Any = None) -> None:
        self.stream = stream or sys.stdout
        self.tty = bool(
            not non_interactive
            and sys.stdin.isatty()
            and getattr(self.stream, "isatty", lambda: False)()
        )
        self.color = self.tty and os.environ.get("NO_COLOR") is None
        self._windows = os.name == "nt"
        self._vt = False
        self.width = 80
        self.height = 24
        self.unicode = False
        self._update_size()

    def __enter__(self) -> Console:
        if self.tty:
            self._enable_vt()
            self.write("\033[?25l")  # hide cursor
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.tty:
            self.write("\033[?25h\033[0m\n")

    def _update_size(self) -> None:
        size = shutil.get_terminal_size((80, 24))
        self.width = max(60, min(100, int(size.columns)))
        self.height = max(20, int(size.lines))
        encoding = (getattr(self.stream, "encoding", "") or "").lower()
        # Windows 传统控制台默认是 cp936/ascii，画框会乱码；检测到 UTF-8 才用制表符。
        self.unicode = bool(self.tty and ("utf" in encoding or os.name != "nt"))

    def _box_chars(self) -> tuple[str, ...]:
        if self.unicode:
            return "┌", "─", "┐", "│", "└", "┘"
        return "+", "-", "+", "|", "+", "+"

    def _enable_vt(self) -> None:
        if not self._windows or self._vt:
            return
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            self._vt = True
        except Exception:
            pass

    # -- 输出 ---------------------------------------------------------------
    def write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def print(self, text: str = "") -> None:
        self.write(text + "\n")

    def colorize(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def clear(self) -> None:
        if self.tty:
            self._update_size()
            self.write("\033[2J\033[H")

    def header(self, text: str, subtitle: str = "", step: str = "") -> None:
        """两行横线夹一行标题；右侧显示环境/位置，窄终端自动截断。"""
        self._update_size()
        rule = "─" * self.width if self.unicode else "=" * self.width
        left = " " + (f"{step}  " if step else "") + text
        right = subtitle.strip()
        if right:
            room = self.width - text_width(left) - 1
            right = _truncate(right, max(0, room))
            gap = max(1, self.width - text_width(left) - text_width(right) - 1)
            line = self.colorize(left, "1;36") + " " * gap + self.colorize(right, "90")
        else:
            line = self.colorize(_truncate(left, self.width - 1), "1;36")
        self.print(self.colorize(rule, "36"))
        self.print(line)
        self.print(self.colorize(rule, "36"))

    def title(self, text: str, subtitle: str = "") -> None:
        # 兼容旧调用：title 现在就是 header。
        self.header(text, subtitle)

    def box(self, lines: Sequence[str], title: str = "", tone: str = "36") -> None:
        """画一个全宽信息框；支持中文对齐与 ANSI 颜色。"""
        self._update_size()
        tl, horizontal, tr, vertical, bl, br = self._box_chars()
        inner = self.width - 2
        if title:
            label = f" {_truncate(title, max(2, inner - 6))} "
            remaining = max(0, inner - 1 - text_width(label))
            top = (
                self.colorize(tl + horizontal, tone)
                + self.colorize(label, "1;36")
                + self.colorize(horizontal * remaining + tr, tone)
            )
            self.print(top)
        else:
            self.print(self.colorize(tl + horizontal * inner + tr, tone))
        content_width = inner - 2
        for line in lines:
            self.print(
                self.colorize(vertical, tone)
                + " "
                + _pad(line, content_width)
                + " "
                + self.colorize(vertical, tone)
            )
        self.print(self.colorize(bl + horizontal * inner + br, tone))

    def note(self, text: str, title: str = "说明", tone: str = "90") -> None:
        """带标题的浅色说明块，用于参数解释与新手提示。"""
        self._update_size()
        if not self.tty:
            self.print(f"[{title}] {text}")
            return
        if not self.unicode:
            self.print(f"  [{title}] {text}")
            return
        content_width = self.width - 6
        label = _truncate(title, max(2, self.width - 10))
        dashes = max(0, self.width - 7 - text_width(label))
        self.print(
            self.colorize("  ┌─ ", tone)
            + self.colorize(label, "1;36")
            + self.colorize(" " + "─" * dashes + "┐", tone)
        )
        for line in _wrap_text(text, content_width):
            self.print(self.colorize("  │ ", tone) + _pad(line, content_width) + self.colorize(" │", tone))
        self.print(self.colorize("  └" + "─" * (self.width - 4) + "┘", tone))

    def footer(self, text: str) -> None:
        self.print(self.colorize(" " + _truncate(text, self.width - 2), "90"))

    def pause(self, text: str = "按回车继续…") -> None:
        if self.tty:
            self.write(text)
            with contextlib.suppress(EOFError):
                input()

    # -- 按键 ---------------------------------------------------------------
    def read_key(self) -> str:
        if not self.tty:
            raise RuntimeError("非交互终端没有按键")
        if self._windows:
            return self._read_key_windows()
        return self._read_key_unix()

    def _read_key_windows(self) -> str:
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            code = msvcrt.getwch()
            return {
                "H": "up",
                "P": "down",
                "K": "left",
                "M": "right",
                "G": "home",
                "O": "end",
                "S": "delete",
                "I": "pageup",
                "Q": "pagedown",
            }.get(code, "")
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        if ch in ("\x08", "\x7f"):
            return "backspace"
        if ch == "\x1b":
            return "esc"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch.lower()

    def _read_key_unix(self) -> str:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            # 不能用默认的 TCSAFLUSH：它会把已经缓冲的后续按键（例如连续方向键）
            # 清空，导致快速操作丢键。
            tty.setraw(fd, when=termios.TCSADRAIN)
            data = os.read(fd, 1)
            if data == b"\x1b":
                # 只消费一个转义序列；连续方向键在缓冲区里时要留给下一次调用。
                if not select.select([fd], [], [], 0.02)[0]:
                    return "esc"
                second = os.read(fd, 1)
                if second not in (b"[", b"O"):
                    return "esc"
                data += second
                third = os.read(fd, 1)
                data += third
                if third.isdigit():
                    while select.select([fd], [], [], 0.02)[0]:
                        ch = os.read(fd, 1)
                        data += ch
                        if ch == b"~":
                            break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

        if data == b"\x1b":
            return "esc"
        if data in (b"\x1b[A", b"\x1bOA"):
            return "up"
        if data in (b"\x1b[B", b"\x1bOB"):
            return "down"
        if data in (b"\x1b[C", b"\x1bOC"):
            return "right"
        if data in (b"\x1b[D", b"\x1bOD"):
            return "left"
        if data in (b"\x1b[H", b"\x1bOH", b"\x1b[1~"):
            return "home"
        if data in (b"\x1b[F", b"\x1bOF", b"\x1b[4~"):
            return "end"
        if data == b"\r" or data == b"\n":
            return "enter"
        if data == b" ":
            return "space"
        if data in (b"\x08", b"\x7f"):
            return "backspace"
        if data == b"\x03":
            raise KeyboardInterrupt
        try:
            return data.decode("utf-8", errors="ignore").lower()
        except Exception:
            return ""

    # -- 文本输入 -----------------------------------------------------------
    def read_line(
        self,
        prompt: str,
        default: str = "",
        validator: Callable[[str], tuple[bool, str]] | None = None,
    ) -> str:
        suffix = f"（默认 {default}）" if default else ""
        while True:
            try:
                raw = input(f"{prompt}{suffix}：").strip()
            except EOFError:
                raise WizardCancelled from None
            value = raw or default
            if validator is None:
                return value
            ok, message = validator(value)
            if ok:
                return value
            self.print(self.colorize("  " + message, "31"))

    # -- 菜单 ---------------------------------------------------------------
    def menu(
        self,
        title: str,
        items: Sequence[str],
        default: int = 0,
        subtitle: str = "",
        footer: str = "↑↓ 选择   Enter 确认   q 退出",
        descriptions: Sequence[str] | None = None,
        intro: Sequence[str] = (),
        extra_keys: dict[str, int] | None = None,
    ) -> int:
        if not items:
            raise WizardError("菜单是空的")
        descriptions = list(descriptions) if descriptions is not None else None
        if descriptions is not None:
            descriptions = (descriptions + [""] * len(items))[: len(items)]
        intro_lines: list[str] = []
        for paragraph in intro:
            intro_lines.extend(_wrap_text(paragraph, max(20, self.width - 4)))
        if not self.tty:
            return self._line_menu(title, items, default, subtitle, descriptions, intro_lines)
        index = max(0, min(default, len(items) - 1))
        # 给标题、说明、页脚留空间；终端很矮时仍然能滚动。
        window = max(4, self.height - 3 - len(intro_lines) - (4 if descriptions else 0) - 3)
        while True:
            self.clear()
            self.header(title, subtitle)
            if intro_lines:
                for line in intro_lines:
                    self.print(self.colorize("  " + line, "90"))
                self.print()
            start = max(0, min(index - window // 2, max(0, len(items) - window)))
            end = min(len(items), start + window)
            if start > 0:
                self.print(self.colorize(f"  ↑ 还有 {start} 项", "90"))
            for i in range(start, end):
                marker = "▶" if i == index else " "
                text = _truncate(items[i], self.width - 4)
                line = f" {marker} {text}"
                self.print(self.colorize(line, "1;32") if i == index else line)
            if end < len(items):
                self.print(self.colorize(f"  ↓ 还有 {len(items) - end} 项", "90"))
            if descriptions and descriptions[index]:
                self.print()
                for line in _wrap_text(descriptions[index], max(20, self.width - 6)):
                    self.print(self.colorize("  │ ", "90") + line)
            self.print()
            self.footer(footer)
            key = self.read_key()
            if key == "up":
                index = (index - 1) % len(items)
            elif key == "down":
                index = (index + 1) % len(items)
            elif key == "home":
                index = 0
            elif key == "end":
                index = len(items) - 1
            elif key in ("pageup", "pagedown"):
                step = max(1, window - 1)
                index = max(0, min(len(items) - 1, index + (step if key == "pagedown" else -step)))
            elif key == "enter":
                return index
            elif extra_keys and key in extra_keys:
                return extra_keys[key]
            elif key in ("q", "esc"):
                raise WizardCancelled

    def _line_menu(
        self,
        title: str,
        items: Sequence[str],
        default: int,
        subtitle: str,
        descriptions: Sequence[str] | None = None,
        intro_lines: Sequence[str] = (),
    ) -> int:
        self.print()
        self.print(title)
        if subtitle:
            self.print(subtitle)
        for line in intro_lines:
            self.print("  " + line)
        if intro_lines:
            self.print()
        for i, item in enumerate(items):
            self.print(f"  {i + 1}) {item}")
            if descriptions and descriptions[i]:
                self.print("       " + descriptions[i])
        while True:
            raw = input(f"请选择 [1-{len(items)}]（默认 {default + 1}）：").strip()
            if not raw:
                return default
            if raw.lower() in ("q", "quit", "exit"):
                raise WizardCancelled
            try:
                value = int(raw) - 1
            except ValueError:
                self.print("请输入编号，或 q 退出。")
                continue
            if 0 <= value < len(items):
                return value
            self.print("编号超出范围。")

    def multi_select(
        self,
        title: str,
        items: Sequence[str],
        selected: Sequence[int] | None = None,
        min_selected: int = 1,
        subtitle: str = "",
    ) -> list[int]:
        chosen = set(range(len(items))) if selected is None else set(selected)
        if not self.tty:
            return self._line_multi(title, items, chosen, min_selected, subtitle)
        index = 0
        window = max(4, self.height - 8)
        while True:
            self.clear()
            self.header(title, subtitle)
            start = max(0, min(index - window // 2, max(0, len(items) - window)))
            end = min(len(items), start + window)
            if start > 0:
                self.print(self.colorize(f"  ↑ 还有 {start} 项", "90"))
            for i in range(start, end):
                box = "[x]" if i in chosen else "[ ]"
                marker = "▶" if i == index else " "
                text = _truncate(items[i], self.width - 8)
                line = f" {marker} {box} {text}"
                self.print(self.colorize(line, "1;32") if i == index else line)
            if end < len(items):
                self.print(self.colorize(f"  ↓ 还有 {len(items) - end} 项", "90"))
            self.print()
            self.print(f" 已选 {len(chosen)} / {len(items)}（至少 {min_selected}）")
            self.footer("↑↓ 移动   空格 勾选   a 全选   n 清空   Enter 确认   q 取消")
            key = self.read_key()
            if key == "up":
                index = (index - 1) % len(items)
            elif key == "down":
                index = (index + 1) % len(items)
            elif key == "home":
                index = 0
            elif key == "end":
                index = len(items) - 1
            elif key == "space":
                chosen.symmetric_difference_update({index})
            elif key == "a":
                chosen = set(range(len(items)))
            elif key == "n":
                chosen.clear()
            elif key == "enter":
                if len(chosen) < min_selected:
                    self.print(self.colorize(f"  至少选择 {min_selected} 项。", "31"))
                    time.sleep(0.6)
                    continue
                return sorted(chosen)
            elif key in ("q", "esc"):
                raise WizardCancelled

    def _line_multi(
        self,
        title: str,
        items: Sequence[str],
        chosen: set[int],
        min_selected: int,
        subtitle: str,
    ) -> list[int]:
        self.print()
        self.print(title)
        if subtitle:
            self.print(subtitle)
        for i, item in enumerate(items):
            mark = "*" if i in chosen else " "
            self.print(f"  [{mark}] {i + 1}) {item}")
        self.print("输入编号（逗号分隔）、all、none；直接回车用当前选择。")
        while True:
            raw = input("选择：").strip().lower()
            if not raw:
                if len(chosen) < min_selected:
                    self.print(f"至少选择 {min_selected} 项。")
                    continue
                return sorted(chosen)
            if raw in ("q", "quit", "exit"):
                raise WizardCancelled
            if raw == "all":
                chosen = set(range(len(items)))
                return sorted(chosen)
            if raw == "none":
                chosen.clear()
                continue
            try:
                picks = set()
                for part in raw.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "-" in part:
                        lo, hi = part.split("-", 1)
                        picks.update(range(int(lo) - 1, int(hi)))
                    else:
                        picks.add(int(part) - 1)
            except ValueError:
                self.print("格式示例：1,3,5 或 2-6")
                continue
            if any(p < 0 or p >= len(items) for p in picks):
                self.print("编号超出范围。")
                continue
            chosen = picks

    def confirm(self, prompt: str, default: bool = True) -> bool:
        hint = "[Y/n]" if default else "[y/N]"
        if not self.tty:
            raw = input(f"{prompt} {hint}：").strip().lower()
            if not raw:
                return default
            return raw in ("y", "yes", "是")
        _ = hint
        self.write(f"{prompt} {'[Y/n]' if default else '[y/N]'} ")
        while True:
            key = self.read_key()
            if key == "enter":
                return default
            if key in ("y", "yes"):
                return True
            if key in ("n", "no"):
                return False
            if key in ("q", "esc"):
                raise WizardCancelled


# --------------------------------------------------------------------------- #
# 表单
# --------------------------------------------------------------------------- #


@dataclass
class FormField:
    label: str
    path: str
    kind: str  # choice | bool | int | float | text | optional_int | optional_float | optional_text
    choices: list[Any] | None = None
    step: float = 1.0
    minimum: float | None = None
    maximum: float | None = None
    comment: str = ""  # 行内一句话注释
    help: str = ""  # 选中时显示的详细说明
    labels: dict[Any, str] | None = None  # 把 lora/native 之类的值显示成中文
    transform: Callable[[Any], str] | None = None

    def get(self, root: Any) -> Any:
        value = root
        for part in self.path.split("."):
            value = getattr(value, part)
        return value

    def set(self, root: Any, value: Any) -> None:
        parts = self.path.split(".")
        target = root
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)

    def display(self, root: Any) -> str:
        value = self.get(root)
        if self.transform is not None:
            return self.transform(value)
        if value is None:
            return "自动"
        if isinstance(value, bool):
            return "开" if value else "关"
        if self.labels and value in self.labels:
            return self.labels[value]
        return str(value)

    def shift(self, root: Any, direction: int) -> None:
        value = self.get(root)
        if self.kind == "choice" and self.choices:
            try:
                index = self.choices.index(value)
            except ValueError:
                index = 0
            self.set(root, self.choices[(index + direction) % len(self.choices)])
        elif self.kind == "bool":
            self.set(root, not bool(value))
        elif self.kind in ("int", "float"):
            delta = self.step * direction
            new = (0 if value is None else value) + delta
            if self.minimum is not None:
                new = max(self.minimum, new)
            if self.maximum is not None:
                new = min(self.maximum, new)
            self.set(root, round(new) if self.kind == "int" else float(new))
        elif self.kind.startswith("optional_"):
            # 左右键在 None 和最小值之间切换，方便快速启用；精确值按 Enter。
            base = self.minimum if self.minimum is not None else 0
            self.set(root, None if value is not None else base)


def _find_field(fields: Sequence[FormField], query: str) -> int | None:
    """按字段路径/名称/注释/说明搜索，路径和名称优先于长说明。"""
    needle = query.strip().lower()
    if not needle:
        return None
    best: int | None = None
    best_score = 0
    for index, field in enumerate(fields):
        path = field.path.lower()
        if path == needle:
            score = 100
        elif needle in path:
            score = 80
        elif needle in field.label.lower():
            score = 60
        elif needle in field.comment.lower():
            score = 40
        elif needle in field.help.lower():
            score = 20
        else:
            score = 0
        if score > best_score:
            best, best_score = index, score
    return best


def _form(console: Console, title: str, fields: Sequence[FormField], root: Any, subtitle: str = "") -> None:
    if not fields:
        return
    if not console.tty:
        _line_form(console, title, fields, root, subtitle)
        return
    index = 0
    label_width = 20
    value_width = 18
    while True:
        console.clear()
        console.header(title, subtitle, step="高级设置")
        window = max(4, console.height - 12)
        start = max(0, min(index - window // 2, max(0, len(fields) - window)))
        end = min(len(fields), start + window)
        if start > 0:
            console.print(console.colorize(f"  ↑ 还有 {start} 项", "90"))
        for i in range(start, end):
            item = fields[i]
            marker = "▶" if i == index else " "
            label = _pad(item.label, label_width)
            raw_value = item.display(root)
            value = _pad(_truncate(raw_value, value_width), value_width)
            prefix = f" {marker} {label} {value}"
            room = console.width - 1 - text_width(prefix)
            comment = _truncate("  " + item.comment, max(0, room)) if item.comment else ""
            if i == index:
                console.print(
                    console.colorize(f" {marker} {label}", "1;32")
                    + console.colorize(f" {value}", "1;32")
                    + console.colorize(comment, "90")
                )
            else:
                console.print(f" {marker} {label} {value}" + console.colorize(comment, "90"))
        if end < len(fields):
            console.print(console.colorize(f"  ↓ 还有 {len(fields) - end} 项", "90"))
        current = fields[index]
        console.print()
        console.note(current.help or current.comment or current.label, title="参数说明")
        console.print()
        console.footer(
            "↑↓ 移动   ←→ 调整   Enter 编辑   / 搜索   PgUp/PgDn 翻页   q 返回"
        )
        key = console.read_key()
        if key == "up":
            index = (index - 1) % len(fields)
        elif key == "down":
            index = (index + 1) % len(fields)
        elif key == "home":
            index = 0
        elif key == "end":
            index = len(fields) - 1
        elif key == "/":
            query = console.read_line("  搜索参数（名称 / 字段 / 说明）", "")
            match = _find_field(fields, query)
            if match is None:
                console.print(console.colorize(f"  没有找到包含“{query}”的参数。", "33"))
                time.sleep(0.8)
            else:
                index = match
        elif key in ("pageup", "pagedown"):
            step = max(1, window - 1)
            index = max(0, min(len(fields) - 1, index + (step if key == "pagedown" else -step)))
        elif key == "left":
            current.shift(root, -1)
        elif key == "right" or (key == "space" and current.kind == "bool"):
            current.shift(root, 1)
        elif key == "enter":
            _edit_field(console, current, root)
        elif key in ("q", "esc"):
            return


def _line_form(
    console: Console, title: str, fields: Sequence[FormField], root: Any, subtitle: str
) -> None:
    console.print()
    console.print(title)
    if subtitle:
        console.print(subtitle)
    for i, item in enumerate(fields):
        comment = f"  # {item.comment}" if item.comment else ""
        console.print(f"  {i + 1:2d}) {item.label:<22} {item.display(root)}{comment}")
    console.print("输入编号修改，直接回车返回。")
    while True:
        raw = input("编号：").strip()
        if not raw:
            return
        try:
            index = int(raw) - 1
        except ValueError:
            console.print("请输入编号。")
            continue
        if 0 <= index < len(fields):
            _edit_field(console, fields[index], root)


def _edit_field(console: Console, field: FormField, root: Any) -> None:
    if field.kind in ("choice", "bool"):
        field.shift(root, 1)
        return
    if field.help:
        console.print()
        console.note(field.help, title=field.label)
    current = field.get(root)
    default = "" if current is None else str(current)
    if field.kind in ("int", "optional_int"):
        prompt = "输入整数（空 = None）" if field.kind == "optional_int" else "输入整数"
    elif field.kind in ("float", "optional_float"):
        prompt = "输入小数（空 = None）" if field.kind == "optional_float" else "输入小数"
    else:
        prompt = "输入文本（空 = None）" if field.kind == "optional_text" else "输入文本"

    def validator(text: str) -> tuple[bool, str]:
        if text == "" and field.kind.startswith("optional_"):
            return True, ""
        try:
            if field.kind in ("int", "optional_int"):
                value = int(text)
            elif field.kind in ("float", "optional_float"):
                value = float(text)
            else:
                value = text
        except ValueError:
            return False, "格式不对，请重新输入"
        if field.minimum is not None and float(value) < field.minimum:
            return False, f"不能小于 {field.minimum}"
        if field.maximum is not None and float(value) > field.maximum:
            return False, f"不能大于 {field.maximum}"
        return True, ""

    raw = console.read_line("  " + prompt, default, validator)
    if raw == "" and field.kind.startswith("optional_"):
        field.set(root, None)
        return
    if field.kind in ("int", "optional_int"):
        field.set(root, int(raw))
    elif field.kind in ("float", "optional_float"):
        field.set(root, float(raw))
    else:
        field.set(root, raw)


# --------------------------------------------------------------------------- #
# 向导流程
# --------------------------------------------------------------------------- #


def _path_candidates(
    candidates: Sequence[Path],
    manual_label: str,
    labeler: Callable[[Path], str] | None = None,
) -> list[str]:
    labels = [labeler(path) if labeler else str(path) for path in candidates]
    labels.append(manual_label)
    return labels


def _choose_path(
    console: Console,
    title: str,
    candidates: Sequence[Path],
    current: str,
    validator: Callable[[str], Any],
    manual_label: str = "手动输入路径…",
    labeler: Callable[[Path], str] | None = None,
    intro: Sequence[str] = (),
) -> Any:
    labels = _path_candidates(candidates, manual_label, labeler)
    default = 0
    matched = False
    if current:
        current_path = _expand(current)
        for i, path in enumerate(candidates):
            if path.resolve() == current_path:
                default = i
                matched = True
                break
        if not matched:
            default = len(labels) - 1
    descriptions = [str(path.resolve()) for path in candidates]
    descriptions.append("可以使用绝对路径、相对路径、~ 和环境变量；路径里有空格也可以。")
    choice = console.menu(
        title,
        labels,
        default,
        subtitle=f"当前：{current or '未选择'}",
        descriptions=descriptions,
        intro=intro,
    )
    if choice == len(labels) - 1:
        def _validate(text: str) -> tuple[bool, str]:
            try:
                validator(text)
                return True, ""
            except WizardError as exc:
                return False, str(exc)

        path = console.read_line("  路径", current, _validate)
    else:
        path = str(candidates[choice])
    return validator(path)


def _model_label(path: Path) -> str:
    try:
        size = _human_size(path.stat().st_size)
    except OSError:
        size = "?"
    return f"{path.name}  ({size})"


def _dataset_label(path: Path) -> str:
    if path.is_dir():
        return f"{path.name}/  (目录)"
    try:
        size = _human_size(path.stat().st_size)
    except OSError:
        size = "?"
    return f"{path.name}  ({size})"


def _edit_model(console: Console, state: WizardState) -> None:
    def _validator(text: str) -> ModelInfo:
        if not text:
            raise WizardError("路径为空")
        return inspect_model(text)

    candidates = find_model_candidates()
    intro = [
        "选择官方 RWKV-7 权重（.pth / .safetensors）。",
        "把文件放进 models/ 就会自动出现在列表里；也可以手动输入任意路径。",
    ]
    if not candidates:
        intro.append("没有自动找到权重：先把模型放到 models/，或选择最后一项手动输入。")
    try:
        info = _choose_path(
            console,
            "选择 RWKV-7 权重",
            candidates,
            state.cfg.model.path or "",
            _validator,
            intro=intro,
            labeler=_model_label,
        )
    except WizardCancelled:
        return
    except WizardError as exc:
        console.print(console.colorize(f"\n{exc}", "31"))
        console.pause()
        return
    state.set_model(info)
    console.print(console.colorize(f"\n已选择：{info.summary}", "32"))
    console.pause()


def _edit_dataset(console: Console, state: WizardState) -> None:
    def _validator(text: str) -> DatasetInfo:
        if not text:
            raise WizardError("路径为空")
        return validate_dataset_path(text)

    candidates = find_dataset_candidates()
    intro = [
        "支持 ShareGPT、OpenAI messages、两列指令和纯文本；文件或目录都可以。",
        "默认只有 assistant 回复参与 loss；选择后可以用 inspect-data 检查 mask。",
        "项目自带 examples/sharegpt_tiny.json，可以先用来跑通链路。",
    ]
    if not candidates:
        intro.append("没有自动找到数据：把 .json/.jsonl 放进 data/，或选择最后一项手动输入。")
    try:
        info = _choose_path(
            console,
            "选择训练数据",
            candidates,
            state.cfg.data.path or "",
            _validator,
            intro=intro,
            labeler=_dataset_label,
        )
    except WizardCancelled:
        return
    except WizardError as exc:
        console.print(console.colorize(f"\n{exc}", "31"))
        console.pause()
        return
    state.set_dataset(info)
    console.print(console.colorize(f"\n已选择：{info.detail}", "32"))
    console.pause()


def _select_modules(console: Console, state: WizardState) -> None:
    current = [i for i, name in enumerate(DEFAULT_MODULES) if name in state.cfg.lora.target_modules]
    labels = [MODULE_LABELS[name] for name in DEFAULT_MODULES]
    try:
        chosen = console.multi_select(
            "选择 LoRA 注入的模块（空格勾选）",
            labels,
            current,
            min_selected=1,
            subtitle="默认 6 个全部勾选；全选通常效果最好，显存紧张时可以取消 ffn。",
        )
    except WizardCancelled:
        return
    state.cfg.lora.target_modules = [DEFAULT_MODULES[i] for i in chosen]

def _select_layers(console: Console, state: WizardState) -> None:
    if state.model_info is None:
        console.print(console.colorize("请先选择模型，才能按层选择。", "31"))
        console.pause()
        return
    n_layer = state.model_info.n_layer
    current = state.cfg.lora.target_layers
    half = max(1, n_layer // 2)
    front = list(range(0, half))
    back = list(range(half, n_layer)) or list(range(n_layer))
    options = [
        f"全部 {n_layer} 层",
        f"前 {len(front)} 层（0–{front[-1]}）",
        f"后 {len(back)} 层（{back[0]}–{back[-1]}）",
        "自定义（空格多选）",
    ]
    if current is None:
        default = 0
    elif current == front:
        default = 1
    elif current == back and back != front:
        default = 2
    else:
        default = 3
    try:
        choice = console.menu(
            "选择 LoRA 目标 block",
            options,
            default,
            subtitle=f"模型共 {n_layer} 层",
            descriptions=[
                "效果最稳，参数量和显存也最多；大多数情况推荐。",
                "只调前一半层，参数更少，适合浅层风格适配。",
                "只调后一半层，常被认为更影响输出风格和知识。",
                "自己勾选任意层；层数多时可以用 a 全选、n 清空。",
            ],
            intro=["层数越少越省显存，但效果通常不如全部层；第一次建议保持全部。"],
        )
    except WizardCancelled:
        return
    if choice == 0:
        state.cfg.lora.target_layers = None
    elif choice == 1:
        state.cfg.lora.target_layers = front
    elif choice == 2:
        state.cfg.lora.target_layers = back
    else:
        selected = current if current is not None else list(range(n_layer))
        try:
            chosen = console.multi_select(
                f"选择 block（0–{n_layer - 1}，空格多选）",
                [f"block {i}" for i in range(n_layer)],
                selected,
                min_selected=1,
            )
        except WizardCancelled:
            return
        state.cfg.lora.target_layers = chosen


def _training_fields(state: WizardState) -> list[FormField]:
    choices = ["lora", "state"]
    precision = ["bf16-full", "bf16", "fp16", "fp32"] if state.env.is_cuda else ["fp32"]
    return [
        FormField(
            "训练方式", "train.method", "choice", choices,
            comment="LoRA 稳，State 省",
            help="LoRA 训练每个线性层的低秩矩阵，适合大多数指令/对话微调；"
                 "State Tuning 只训练每层初始循环状态，参数极少，适合小数据快速适配。",
            labels={"lora": "LoRA", "state": "State Tuning"},
        ),
        FormField(
            "infctx（长上下文）", "train.infctx", "bool",
            comment="长文本分块训练",
            help="开启后按 chunk_ctx 分块处理长文本，显存只与 chunk_ctx 有关，"
                 "不会因为 ctx_len 变长而暴涨；速度略低于整段训练。",
        ),
        FormField(
            "ctx_len", "train.ctx_len", "int", step=512, minimum=64,
            comment="单条样本 token 上限",
            help="每条训练样本的最大长度。对话一般 2048；长文档用 8192 或更大，"
                 "并配合 infctx。显存不够时先降它。",
        ),
        FormField(
            "chunk_ctx（infctx）", "train.chunk_ctx", "int", step=256, minimum=16,
            comment="每块 token 数",
            help="infctx 每次前向/反传的块长，必须 ≤ ctx_len。越大越快、越吃显存；"
                 "24GB 卡常用 2048，显存紧张降到 512–1024。",
        ),
        FormField(
            "micro_batch_size", "train.micro_batch_size", "int", minimum=1,
            comment="单次前向样本数",
            help="显存不够时先降它，再用 grad_accum 把总 batch 补回来。"
                 "0.1B / ctx512 在 24GB 卡上可以加到 32。",
        ),
        FormField(
            "grad_accum", "train.grad_accum", "int", minimum=1,
            comment="梯度累积步数",
            help="总 batch = micro_batch_size × grad_accum。显存不足时降 micro、加 accum，"
                 "保持等效 batch。",
        ),
        FormField(
            "epochs", "train.epochs", "int", minimum=1,
            comment="遍历数据集次数",
            help="小数据通常 1–3 个 epoch；如果设置了 max_steps，以步数优先。",
        ),
        FormField(
            "max_steps", "train.max_steps", "optional_int", minimum=1,
            comment="最多训练步数",
            help="留空（自动）表示按 epochs 跑完；填 20 可以做一次快速试跑。",
        ),
        FormField(
            "lr", "train.lr", "float", step=1.0e-5, minimum=0.0,
            comment="学习率",
            help="LoRA 从 1e-4 起，State 从 5e-3 起。loss 震荡就减半，几乎不降就翻倍试试。",
        ),
        FormField(
            "min_lr", "train.min_lr", "float", step=1.0e-6, minimum=0.0,
            comment="衰减终点",
            help="余弦/线性调度最后降到这个值，通常是 lr 的 1/10。",
        ),
        FormField(
            "warmup_steps", "train.warmup_steps", "int", minimum=0,
            comment="学习率预热步数",
            help="前 N 步学习率从 0 升到 lr，避免一开始梯度太大。",
        ),
        FormField(
            "lr_schedule", "train.lr_schedule", "choice",
            ["cosine", "linear", "constant"],
            comment="学习率调度",
            help="cosine 最稳；linear 简单；constant 全程不衰减。",
            labels={"cosine": "cosine（推荐）", "linear": "linear", "constant": "constant"},
        ),
        FormField(
            "weight_decay", "train.weight_decay", "float", step=1.0e-3, minimum=0.0,
            comment="权重衰减",
            help="LoRA 一般 0.01；State 建议 0。过拟合时略微增加。",
        ),
        FormField(
            "beta1", "train.beta1", "float", step=0.01, minimum=0.0, maximum=1.0,
            comment="AdamW 一阶动量",
            help="一般保持 0.9；不清楚时不要改。",
        ),
        FormField(
            "beta2", "train.beta2", "float", step=0.01, minimum=0.0, maximum=1.0,
            comment="AdamW 二阶动量",
            help="RWKV 常用 0.99；一般保持默认。",
        ),
        FormField(
            "grad_clip", "train.grad_clip", "float", step=0.1, minimum=0.0,
            comment="梯度裁剪阈值",
            help="loss 出现尖峰时降到 0.5–1.0；设为 0 表示不裁剪。",
        ),
        FormField(
            "precision", "train.precision", "choice", precision,
            comment="GPU 推荐 bf16-full",
            help="bf16-full 把冻结基座直接存 bf16，最快；bf16 用 autocast；"
                 "fp16 需要较新的卡；CPU 必须 fp32。LoRA/State 参数始终是 fp32。",
            labels={
                "bf16-full": "bf16-full（最快）",
                "bf16": "bf16（autocast）",
                "fp16": "fp16",
                "fp32": "fp32（CPU）",
            },
        ),
        FormField(
            "grad_checkpoint", "train.grad_checkpoint", "bool",
            comment="用时间换显存",
            help="长上下文/大模型显存不够时开启；infctx 已经把显存限制在 chunk 内，"
                 "通常可以关掉换速度。",
        ),
        FormField(
            "l2wrap", "train.l2wrap", "float", step=1.0e-5, minimum=0.0,
            comment="logits 稳定项",
            help="RWKV 的 L2Wrap 系数，抑制 logits 极端值；默认 1e-4，设 0 可关闭。",
        ),
        FormField(
            "infctx_detach", "train.infctx_detach", "bool",
            comment="块间截断梯度",
            help="开启后块与块之间不回传梯度，省显存；默认关闭以保留长程梯度。",
        ),
        FormField(
            "infctx_backward", "train.infctx_backward", "bool",
            comment="单卡每块立即反传",
            help="单卡 infctx 每个 chunk 立即 backward，显存 O(chunk_ctx)；"
                 "多卡是否逐块由下一个字段控制。",
        ),
        FormField(
            "多卡逐块反传", "train.infctx_parallel_backward", "bool",
            comment="DDP 长文本 O(chunk)",
            help="DDP + infctx 时每个 chunk 立即反传，只有最后一块触发一次梯度 all-reduce；"
                 "显存 O(chunk_ctx)，不随 ctx_len 增长。State Tuning 与 DeepSpeed "
                 "会自动回退到最后统一反传。",
        ),
        FormField(
            "log_every", "train.log_every", "int", minimum=1,
            comment="日志间隔步数",
            help="每多少步打印一次 loss / tok/s / 显存 / ETA。",
        ),
        FormField(
            "save_every_steps", "train.save_every_steps", "int", minimum=0,
            comment="按步保存，0=关",
            help="每隔多少步保存一次 checkpoint；0 表示不按步保存。",
        ),
        FormField(
            "save_every_epochs", "train.save_every_epochs", "int", minimum=0,
            comment="按 epoch 保存",
            help="每多少个 epoch 保存一次；0 表示不按 epoch 保存。",
        ),
        FormField(
            "keep_last", "train.keep_last", "int", minimum=1,
            comment="保留 checkpoint 数",
            help="最多保留几个最近的 checkpoint，避免磁盘被占满。",
        ),
        FormField(
            "merge_at_end", "train.merge_at_end", "bool",
            comment="结束自动合并",
            help="训练结束自动把适配器合并成 merged.pth，可以直接拿去推理。",
        ),
        FormField(
            "val_every_steps", "train.val_every_steps", "int", minimum=0,
            comment="验证间隔，0=关",
            help="每多少步在验证集上算一次 loss；需要 data.val_ratio > 0。",
        ),
        FormField(
            "compile", "train.compile", "bool",
            comment="torch.compile 加速",
            help="CUDA 上建议开启；fla 编译前向，native 只编译 WKV 单块。首次启动会慢几十秒。",
        ),
        FormField(
            "compile_mode", "train.compile_mode", "choice",
            ["default", "reduce-overhead", "max-autotune"],
            comment="编译模式",
            help="default 最稳；reduce-overhead 省显存（State 推荐）；"
                 "max-autotune 再快一点但首次编译约 8 分钟。",
            labels={
                "default": "default",
                "reduce-overhead": "reduce-overhead（省显存）",
                "max-autotune": "max-autotune（编译久）",
            },
        ),
        FormField(
            "device", "train.device", "choice", ["auto", "cuda", "cpu"],
            comment="训练设备",
            help="auto 自动选择；没有 GPU 时固定 cpu（只建议自检）。",
            labels={"auto": "auto（自动）", "cuda": "cuda", "cpu": "cpu"},
        ),
        FormField(
            "run.seed", "run.seed", "int", minimum=0,
            comment="随机种子",
            help="复现实验时固定；换一个种子可以看结果是否稳定。",
        ),
        FormField(
            "输出目录", "run.output_dir", "text",
            comment="输出根目录",
            help="checkpoint、日志、merged.pth 都写在这里；最终目录是 输出目录/实验名。",
        ),
        FormField(
            "实验名", "run.name", "text",
            comment="本次实验名",
            help="输出子目录名；也用于区分不同实验，建议带上方法/日期。",
        ),
    ]


def _model_fields(state: WizardState) -> list[FormField]:
    backends = ["native"]
    if state.env.is_cuda and state.env.fla_available:
        backends.insert(0, "fla")
    return [
        FormField(
            "权重路径", "model.path", "text",
            comment="官方 .pth/.safetensors",
            help="RWKV-7 官方 state_dict；维度会从权重自动推断。不要填 LoRA adapter 或 HF 目录。",
        ),
        FormField(
            "random_init", "model.random_init", "bool",
            comment="随机小模型，仅调试",
            help="不加载真实权重，随机初始化一个小模型跑通链路；只用于 CPU 冒烟测试。",
        ),
        FormField(
            "n_layer", "model.n_layer", "optional_int", minimum=1,
            comment="留空自动推断",
            help="层数；只有随机初始化或特殊 checkpoint 才需要手填。",
        ),
        FormField(
            "n_embd", "model.n_embd", "optional_int", minimum=1,
            comment="留空自动推断",
            help="隐藏维度；一般从权重推断。",
        ),
        FormField(
            "vocab_size", "model.vocab_size", "optional_int", minimum=1,
            comment="留空自动推断",
            help="词表大小；G1 系列通常是 65536。",
        ),
        FormField(
            "head_size", "model.head_size", "optional_int", minimum=1,
            comment="留空自动推断",
            help="每个 attention head 的维度；一般从权重推断。",
        ),
        FormField(
            "decay", "model.decay", "choice", ["auto", "clampw", "softplus", "exp"],
            comment="留 auto 最稳",
            help="衰减公式。不同 RWKV-7 实现公式不同，用错不报错但输出会明显变差；"
                 "auto 会用内置中英文本自动校准。",
            labels={
                "auto": "auto（自动校准）",
                "clampw": "clampw（官方权重）",
                "softplus": "softplus",
                "exp": "exp",
            },
        ),
        FormField(
            "WKV 后端", "model.backend", "choice", backends,
            comment="CUDA 优先 fla",
            help="fla 使用 Triton 融合内核，CUDA 上明显更快；native 是纯 PyTorch，"
                 "CPU/ROCm/没有 Triton 时使用。",
            labels={
                "fla": "fla（Triton 更快）",
                "native": "native（通用）",
            },
        ),
        FormField(
            "wkv_chunk（native）", "model.wkv_chunk", "int", step=8, minimum=8, maximum=96,
            comment="仅 native；64 最佳",
            help="native 后端内部的 WKV 分块长度，不影响数学结果；64 是甜点，"
                 "fla 后端会忽略这个字段。",
        ),
        FormField(
            "head_size_divisor", "model.head_size_divisor", "int", minimum=1,
            comment="GroupNorm eps；官方 8",
            help="GroupNorm 的 eps = 1e-5 × divisor²。官方权重保持 8，一般不要改。",
        ),
        FormField(
            "dtype 覆盖", "model.dtype", "optional_text",
            comment="一般留空",
            help="强制把权重转换成 fp32/bf16/fp16；留空跟随 train.precision。",
        ),
    ]


def _data_fields(state: WizardState) -> list[FormField]:
    return [
        FormField(
            "数据路径", "data.path", "text",
            comment="ShareGPT/JSONL/纯文本",
            help="可以是文件或目录，自动识别 ShareGPT、OpenAI messages、两列指令、纯文本。",
        ),
        FormField(
            "格式", "data.format", "choice", ["auto", "sharegpt", "messages", "text"],
            comment="auto 自动识别",
            help="自动识别失败时手动指定；纯文本数据用 text。",
            labels={
                "auto": "auto（自动）",
                "sharegpt": "sharegpt",
                "messages": "messages",
                "text": "text（纯文本）",
            },
        ),
        FormField(
            "mode", "data.mode", "choice", ["packed", "sample"],
            comment="packed 更省显存",
            help="packed 把多条样本拼满窗口，适合 infctx、吞吐更高；"
                 "sample 每条对话独立、动态 padding。",
            labels={"packed": "packed（拼满窗口）", "sample": "sample（单条样本）"},
        ),
        FormField(
            "模板", "data.template", "choice", ["rwkv_world", "chatml", "custom"],
            comment="对话格式",
            help="rwkv_world 是 RWKV G1 官方格式；chatml 适合通用模型；"
                 "custom 使用下面的前缀字段自定义。",
            labels={
                "rwkv_world": "rwkv_world（官方）",
                "chatml": "chatml",
                "custom": "custom（自定义）",
            },
        ),
        FormField(
            "train_on", "data.train_on", "choice", ["assistant", "last_assistant"],
            comment="训练哪些回复",
            help="assistant 训练每条 assistant 回复；last_assistant 只训练最后一条，"
                 "适合单轮问答或省显存。",
            labels={
                "assistant": "所有 assistant 回复",
                "last_assistant": "只训最后一条",
            },
        ),
        FormField(
            "append_eos", "data.append_eos", "choice", ["end", "each_assistant", "never"],
            comment="EOT 追加方式",
            help="end 只在对话末尾加 EOT；each_assistant 每条回复后都加；never 不加。",
            labels={
                "end": "仅末尾",
                "each_assistant": "每条回复后",
                "never": "不加",
            },
        ),
        FormField(
            "truncation（sample）", "data.truncation", "choice", ["head", "tail", "random"],
            comment="超长截断方向",
            help="sample 模式下样本超过 ctx_len 时，截掉头部/尾部或随机截断；"
                 "packed 模式忽略。",
            labels={"head": "截头", "tail": "截尾", "random": "随机"},
        ),
        FormField(
            "val_ratio", "data.val_ratio", "float", step=0.01, minimum=0.0, maximum=0.9,
            comment="验证集比例",
            help="从数据里划出多少做验证集；0 表示不划分。配合 val_every_steps 使用。",
        ),
        FormField(
            "cache", "data.cache", "bool",
            comment="缓存分词结果",
            help="强烈建议开启；第二次启动会快很多。修改模板/数据后会自动失效。",
        ),
        FormField(
            "cache_dir", "data.cache_dir", "optional_text",
            comment="缓存目录，留空默认",
            help="默认在项目下的 .rwkv_easy_cache；磁盘紧张时可以改到其他位置。",
        ),
        FormField(
            "vocab_path", "data.vocab_path", "optional_text",
            comment="自定义词表，留空内置",
            help="一般使用内置的 RWKV 词表；只有换词表/特殊模型时才需要填。",
        ),
        FormField(
            "eos_token_id", "data.eos_token_id", "optional_int", minimum=0,
            comment="EOT id，留空自动",
            help="结束符 token id；留空自动使用词表里的 EOT（通常是 0）。",
        ),
        FormField(
            "num_workers", "data.num_workers", "int", minimum=0,
            comment="DataLoader 进程数",
            help="Windows 建议 0；Linux 可以 2–4 提升数据加载速度。",
        ),
        FormField(
            "num_proc", "data.num_proc", "int", minimum=0,
            comment="分词并行进程数",
            help="首次分词时使用的进程数；0 表示主进程串行。",
        ),
        FormField(
            "pin_memory", "data.pin_memory", "bool",
            comment="锁页内存",
            help="GPU 训练时开启可以加快 H2D 拷贝；CPU 训练无影响。",
        ),
        FormField(
            "text_field", "data.text_field", "text",
            comment="纯文本字段名",
            help="纯文本 JSON/JSONL 里存放正文的字段名，默认 text。",
        ),
        FormField(
            "max_samples", "data.max_samples", "optional_int", minimum=1,
            comment="最多读取样本数",
            help="调试时只读前 N 条；留空（自动）表示全部。",
        ),
        FormField(
            "drop_empty", "data.drop_empty", "bool",
            comment="丢弃空样本",
            help="丢掉没有可训练 token 的样本，避免无效数据参与训练。",
        ),
        FormField(
            "system_prefix", "data.system_prefix", "optional_text",
            comment="自定义 System 前缀",
            help="template=custom 时使用；例如 \"[SYSTEM] \"。",
        ),
        FormField(
            "user_prefix", "data.user_prefix", "optional_text",
            comment="自定义 User 前缀",
            help="template=custom 时使用；例如 \"[USER] \"。",
        ),
        FormField(
            "assistant_prefix", "data.assistant_prefix", "optional_text",
            comment="自定义 Assistant 前缀",
            help="template=custom 时使用；例如 \"[ASSISTANT] \"。前缀本身不计 loss。",
        ),
        FormField(
            "turn_separator", "data.turn_separator", "optional_text",
            comment="轮次分隔符",
            help="自定义模板时每轮之间的分隔符；一般用空行。",
        ),
    ]


def _misc_fields(state: WizardState) -> list[FormField]:
    modes = ["auto", "single", "ddp"]
    if state.env.deepspeed_available:
        modes.append("deepspeed")
    return [
        FormField(
            "dist.mode", "dist.mode", "choice", modes,
            comment="多卡启动方式",
            help="auto 有 N 卡就自动 DDP；single 强制单进程；"
                 "ddp 手动多卡；deepspeed 需要安装 DeepSpeed。",
            labels={
                "auto": "auto（自动）",
                "single": "single（单卡）",
                "ddp": "ddp（多卡）",
                "deepspeed": "deepspeed（ZeRO）",
            },
        ),
        FormField(
            "num_gpus", "dist.num_gpus", "text",
            comment="auto 或 GPU 数量",
            help="auto 使用全部可见 GPU；也可以填 1/2/4 限制数量。",
        ),
        FormField(
            "master_port", "dist.master_port", "int", minimum=1, maximum=65535,
            comment="DDP 通信端口",
            help="多卡训练时主进程端口；撞端口时换一个（例如 29501）。",
        ),
        FormField(
            "find_unused_parameters", "dist.find_unused_parameters", "bool",
            comment="DDP 调试选项",
            help="有参数未参与反传时报错，可临时开启；会增加一点开销。",
        ),
        FormField(
            "zero_stage", "dist.deepspeed.zero_stage", "int", minimum=1, maximum=3,
            comment="ZeRO 分片级别",
            help="1/2 适合单机多卡；3 分片参数，适合 7B 级大模型，但更慢。",
        ),
        FormField(
            "offload_optimizer", "dist.deepspeed.offload_optimizer", "bool",
            comment="优化器卸载到 CPU",
            help="显存极紧张时开启，速度会明显变慢。",
        ),
        FormField(
            "offload_param", "dist.deepspeed.offload_param", "bool",
            comment="参数卸载到 CPU",
            help="ZeRO-3 下进一步省显存；只有大模型 + 小显存时才需要。",
        ),
        FormField(
            "bucket_size", "dist.deepspeed.bucket_size", "float", step=1.0e7, minimum=1.0,
            comment="通信桶大小",
            help="DeepSpeed 梯度通信桶；一般保持默认 2e8。",
        ),
        FormField(
            "contiguous_gradients", "dist.deepspeed.contiguous_gradients", "bool",
            comment="连续梯度内存",
            help="DeepSpeed 选项；一般保持开启。",
        ),
        FormField(
            "overlap_comm", "dist.deepspeed.overlap_comm", "bool",
            comment="通信与计算重叠",
            help="DeepSpeed 选项；一般保持开启。",
        ),
        FormField(
            "log.jsonl", "log.jsonl", "bool",
            comment="写 metrics.jsonl",
            help="记录每步 loss/tok/s 等指标，方便画曲线；建议开启。",
        ),
        FormField(
            "log.wandb", "log.wandb", "bool",
            comment="启用 wandb",
            help="需要安装 wandb 并登录；不上传时保持关闭。",
        ),
        FormField(
            "wandb_project", "log.wandb_project", "text",
            comment="wandb 项目名",
            help="启用 wandb 时的项目名。",
        ),
        FormField(
            "wandb_run", "log.wandb_run", "optional_text",
            comment="wandb run 名",
            help="留空自动生成；也可以指定一个名字方便对比。",
        ),
        FormField(
            "log.level", "log.level", "choice", ["info", "debug", "warning"],
            comment="日志级别",
            help="info 正常；debug 打更多细节；warning 只报问题。",
            labels={"info": "info", "debug": "debug", "warning": "warning"},
        ),
        FormField(
            "run.deterministic", "run.deterministic", "bool",
            comment="确定性模式",
            help="开启后尽量固定算子行为，复现性更好但可能略慢。",
        ),
    ]


def _lora_fields(state: WizardState) -> list[FormField]:
    return [
        FormField(
            "LoRA rank", "lora.rank", "int", minimum=1, maximum=512,
            comment="低秩维度 r",
            help="越大表达能力越强、参数越多；常用 16–64。小数据 8–16 就够。",
        ),
        FormField(
            "LoRA alpha", "lora.alpha", "int", minimum=1, maximum=1024,
            comment="缩放系数",
            help="实际缩放是 alpha/rank；常用 alpha=2×rank。",
        ),
        FormField(
            "LoRA dropout", "lora.dropout", "float", step=0.01, minimum=0.0, maximum=0.9,
            comment="丢弃率",
            help="小数据过拟合时加到 0.05–0.1；一般保持 0。",
        ),
        FormField(
            "target_regex", "lora.target_regex", "optional_text",
            comment="额外正则匹配",
            help="与上面的模块列表取并集；高级用法，一般留空。",
        ),
        FormField(
            "State init_std", "state.init_std", "float", step=0.001, minimum=0.0,
            comment="State 初始化标准差",
            help="0 表示零初始化（官方做法，推荐）；一般不需要改。",
        ),
    ]


def _advanced_menu(console: Console, state: WizardState) -> None:
    options = [
        "训练参数（lr / batch / 精度 / infctx / 保存…）",
        "模型与 WKV 后端（decay / backend / chunk…）",
        "数据（格式 / 模板 / mask / workers…）",
        "分布式 / 日志（DDP / DeepSpeed / wandb…）",
        "LoRA / State 参数（rank / alpha / dropout…）",
        "返回",
    ]
    descriptions = [
        "学习率、batch、上下文、精度、梯度检查点、保存与验证策略。",
        "权重路径、decay 公式、WKV 后端、分块长度、GroupNorm 参数。",
        "数据路径、格式、packed/sample、对话模板、loss mask、分词进程。",
        "DDP / DeepSpeed ZeRO、offload、日志级别与 wandb。",
        "LoRA 的 rank/alpha/dropout 与 State 的初始化标准差。",
        "回到一键配置首页。",
    ]
    while True:
        try:
            choice = console.menu(
                "高级设置",
                options,
                0,
                subtitle="所有配置字段都可以改，每一项都有说明",
                descriptions=descriptions,
                intro=["不确定的字段保持默认即可；改完会自动套用环境约束（例如 CPU 强制 fp32）。"],
            )
        except WizardCancelled:
            return
        if choice == 5:
            return
        if choice == 0:
            _form(console, "训练参数", _training_fields(state), state.cfg)
        elif choice == 1:
            _form(console, "模型与 WKV", _model_fields(state), state.cfg)
        elif choice == 2:
            _form(console, "数据", _data_fields(state), state.cfg)
        elif choice == 3:
            _form(console, "分布式 / 日志", _misc_fields(state), state.cfg)
        elif choice == 4:
            _form(console, "LoRA / State", _lora_fields(state), state.cfg)
        normalize_for_environment(state.cfg, state.env)


def _summary(console: Console, state: WizardState) -> str | None:
    """展示摘要并返回用户动作：train/save/back/quit。"""
    options = [
        "开始训练（自动保存配置）",
        "仅保存配置，不训练",
        "返回修改",
        "退出",
    ]
    descriptions = [
        "保存 YAML 并立即启动训练；会显示实时 loss、tok/s 和 ETA。",
        "只生成配置文件，之后再手动运行 rwkv-easy train。",
        "回到首页继续调整模板、参数或数据。",
        "不保存任何内容，直接退出。",
    ]
    while True:
        errors = state.validate()
        body: list[str] = [f"模板：{state.preset_label()}"]
        for line in config_summary(state, compact=True).splitlines():
            body.extend(_wrap_text(line, max(20, console.width - 8)))
        if not console.tty:
            console.print()
            console.print("训练配置")
            for line in body:
                console.print("  " + line)
            console.print()
            if errors:
                console.print("需要修正：")
                for error in errors:
                    console.print("  - " + error)
            else:
                console.print("配置检查通过")
            try:
                choice = console.menu("下一步", options, 0, descriptions=descriptions)
            except WizardCancelled:
                return "quit"
            if choice in (0, 1) and errors:
                console.print("还有错误，先修正配置。")
                continue
            return ("train", "save", "back", "quit")[choice]

        index = 0
        while True:
            console.clear()
            console.header("确认配置", "检查无误后开始训练", step="第 2 步")
            console.box(body, title="训练配置")
            if errors:
                console.print(console.colorize("  ✗ 需要修正：", "31"))
                for error in errors:
                    for line in _wrap_text(error, max(20, console.width - 8)):
                        console.print(console.colorize("    " + line, "31"))
            else:
                console.print(console.colorize("  ✓ 配置检查通过", "32"))
            if state.cfg.data.cache and console.height >= 28:
                console.print(
                    console.colorize(
                        "  提示：首次训练会先分词并写入缓存，可能多花 1–3 分钟。", "90"
                    )
                )
            if console.height >= 28:
                console.print()
            for i, option in enumerate(options[:3]):
                marker = "▶" if i == index else " "
                line = f" {marker} {option}"
                console.print(console.colorize(line, "1;32") if i == index else line)
            marker = "▶" if index == 3 else " "
            line = f" {marker} {options[3]}"
            console.print(console.colorize(line, "1;32") if index == 3 else line)
            if console.height >= 28:
                console.print()
            for line in _wrap_text(descriptions[index], max(20, console.width - 6)):
                console.print(console.colorize("  │ ", "90") + line)
            console.footer("↑↓ 选择   Enter 确认   q 退出")
            key = console.read_key()
            if key == "up":
                index = (index - 1) % len(options)
            elif key == "down":
                index = (index + 1) % len(options)
            elif key in ("q", "esc"):
                return "quit"
            elif key == "enter":
                if index in (0, 1) and errors:
                    console.print(console.colorize("\n还有错误，先修正配置。", "31"))
                    console.pause()
                    continue
                if index == 0:
                    return "train"
                if index == 1:
                    return "save"
                if index == 2:
                    return "back"
                return "quit"


def _save_config(console: Console, state: WizardState, output: str | None) -> Path | None:
    cfg = state.cfg
    name = (cfg.run.name or "quickstart").strip() or "quickstart"
    name = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", name)
    cfg.run.name = name
    if output:
        path = _expand(output)
        if path.is_dir():
            path = path / f"{name}.yaml"
    else:
        path = Path("configs") / f"{name}.yaml"
    if path.exists() and not console.confirm(f"{path} 已存在，覆盖？", default=False):
        path = path.with_name(f"{path.stem}-{time.strftime('%H%M%S')}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(
        path,
        header=(
            f"# 由 RWKV-EASY 交互式向导生成（模板：{state.preset_label()}）\n"
            "# 所有字段都可以手动修改；再次运行 rwkv-easy wizard 可以套用新模板"
        ),
    )
    console.print(console.colorize(f"\n配置已保存：{path}", "32"))
    return path


def _run_training(console: Console, state: WizardState, config_path: Path) -> int:
    state.cfg.save_yaml(config_path)
    console.print("\n启动训练：")
    console.print(f"  {sys.executable} -m rwkv_easy train -c {config_path}")
    console.print()
    console.__exit__(None, None, None)
    try:
        return int(subprocess.call([sys.executable, "-m", "rwkv_easy", "train", "-c", str(config_path)]))
    except KeyboardInterrupt:
        return 130


@dataclass
class HomeItem:
    kind: str  # choice | path | action
    label: str
    value: str
    action: str
    comment: str = ""
    help: str = ""


def _home_items(state: WizardState) -> list[HomeItem]:
    cfg = state.cfg
    method_value = "State Tuning" if cfg.train.method == "state" else "LoRA"
    infctx_value = "开启" if cfg.train.infctx else "关闭"
    items: list[HomeItem] = [
        HomeItem(
            "action", "配置模板", state.preset_label(), "preset",
            comment="按场景一键套用",
            help="模板会一次调整训练方式、infctx、上下文、batch、精度和步数；"
                 "模型/数据/输出目录保持不变。不知道自己该选什么时，先用推荐模板。",
        ),
        HomeItem(
            "choice", "训练方式", method_value, "method",
            comment="LoRA 稳，State 省",
            help="LoRA 训练低秩矩阵，适合大多数指令/对话微调；"
                 "State Tuning 只训练每层初始状态，参数 < 0.1%，适合小数据快速适配。",
        ),
        HomeItem(
            "choice", "长上下文（infctx）", infctx_value, "infctx",
            comment="长对话 / 长文档",
            help="开启后分块处理 8K+ 长文本，显存只与 chunk_ctx 有关；"
                 "处理长文档、长对话记忆时再开。",
        ),
        HomeItem(
            "path", "模型权重", cfg.model.path or "未选择", "model",
            comment=("未选择：把 .pth 放到 models/" if not cfg.model.path else
                     "官方 .pth/.safetensors"),
            help="RWKV-7 官方 state_dict，维度会自动识别；0.1B 约 365MB。"
                 "把文件放到 models/ 后会自动出现在候选里；没有模型时可以先看新手引导。",
        ),
        HomeItem(
            "path", "数据集", cfg.data.path or "未选择", "data",
            comment=("未选择：把数据放到 data/" if not cfg.data.path else
                     "ShareGPT / messages / 纯文本"),
            help="默认只有 assistant 回复参与 loss；支持 ShareGPT、OpenAI messages、"
                 "两列指令和纯文本。项目自带 examples/sharegpt_tiny.json 可以先试。",
        ),
    ]
    if (
        state.env.gpu_count > 1
        and cfg.train.infctx
        and cfg.train.method == "lora"
        and cfg.dist.mode != "deepspeed"
    ):
        items.append(
            HomeItem(
                "choice",
                "多卡逐块反传",
                "开启" if cfg.train.infctx_parallel_backward else "关闭",
                "parallel_backward",
                comment="DDP 长文本省显存",
                help="开启后 DDP infctx 每个 chunk 立即反传，只保留当前块的激活，"
                     "最后一块统一 all-reduce 梯度；显存 O(chunk_ctx)，不随 ctx_len 增长。"
                     "关闭则所有 chunk 最后统一反传，显存随 ctx_len 增长。",
            )
        )
    if cfg.train.method == "lora":
        items.append(
            HomeItem(
                "action", "LoRA 模块", state.modules_summary(), "modules",
                comment="默认 6 个线性层",
                help="控制 LoRA 注入哪些线性层。默认的 att/ffn 六个模块覆盖最常用位置；"
                     "全选通常效果最好。",
            )
        )
        items.append(
            HomeItem(
                "action", "LoRA 层", state.layers_summary(), "layers",
                comment="默认全部层",
                help="默认给所有 block 注入 LoRA；只调前几层/后几层可以省显存，"
                     "但通常全部层效果更稳。",
            )
        )
    items.extend(
        [
            HomeItem(
                "choice", "WKV 后端", cfg.model.backend, "backend",
                comment="CUDA 优先 fla",
                help="fla 是 Triton 融合内核，CUDA 上明显更快；native 是纯 PyTorch，"
                     "CPU/ROCm/没有 Triton 时使用。没有 rwkv-fla 时自动用 native。",
            ),
            HomeItem(
                "choice", "精度", cfg.train.precision, "precision",
                comment="GPU 推荐 bf16-full",
                help="bf16-full 让冻结基座直接以 bf16 存储，最快、显存最省；"
                     "LoRA/State 参数仍然是 fp32。CPU 固定 fp32。",
            ),
            HomeItem(
                "action", "训练参数", f"bsz {cfg.train.micro_batch_size}×{cfg.train.grad_accum}"
                f" / ctx {cfg.train.ctx_len}", "train",
                comment="batch / 上下文 / 学习率",
                help="打开训练参数表单，可改 batch、上下文、学习率、精度、保存策略等；"
                     "每一项都有行内注释和详细说明。",
            ),
            HomeItem(
                "action", "新手引导", "3 分钟了解", "guide",
                comment="第一次用先看这里",
                help="用 4 页说明这个工具做什么、训练前准备什么、模板怎么选、常见坑怎么避免。",
            ),
            HomeItem(
                "action", "高级设置", "全部字段", "advanced",
                comment="模型 / 数据 / 分布式",
                help="所有 81 个配置字段，分组显示；每一项都有注释，适合需要精细控制的用户。",
            ),
            HomeItem(
                "action", "开始训练", "保存配置并启动", "start",
                comment="检查摘要后开跑",
                help="先展示完整配置摘要和检查结果；确认后保存 YAML 并启动训练。",
            ),
            HomeItem(
                "action", "仅保存配置", "生成 YAML", "save",
                comment="不启动训练",
                help="只把当前配置写成 YAML，方便手动修改或提交到版本库。",
            ),
        ]
    )
    return items


def _preset_menu(console: Console, state: WizardState) -> None:
    keys = [preset.key for preset in PRESETS]
    try:
        current = keys.index(state.preset_key)
    except ValueError:
        current = 0
    labels = [f"[{preset.tag}] {preset.title}" for preset in PRESETS]
    descriptions = [f"{preset.summary}。{preset.description}" for preset in PRESETS]
    try:
        choice = console.menu(
            "选择配置模板",
            labels,
            current,
            subtitle=f"当前：{state.preset_label()}",
            descriptions=descriptions,
            intro=[
                "模板会调整训练方式、上下文、batch、精度和步数；"
                "模型、数据集、输出目录保持不变。选错了可以随时回来重选。"
            ],
        )
    except WizardCancelled:
        return
    apply_preset(state, keys[choice])


GUIDE_PAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "这个工具能做什么",
        (
            "RWKV-EASY 只做训练，不是推理产品。它把 RWKV-7 的 LoRA / State Tuning 打包成"
            "一条命令，并原生支持 ShareGPT 对话数据。",
            "两种方法：LoRA 训练每个线性层的低秩矩阵，效果稳定，适合大多数微调；"
            "State Tuning 只训练每层初始循环状态，适配器只有几百 KB，适合小数据和快速适配。",
            "两者都能叠加 infctx：长文本分块训练，显存只与 chunk_ctx 有关，"
            "0.1B 在 24GB 卡上可以训 32K 上下文。",
            "数据默认只有 assistant 回复参与 loss；System / User 只作为上下文。",
        ),
    ),
    (
        "训练前准备什么",
        (
            "1. 模型：官方 RWKV-7 state_dict（.pth / .safetensors），放到 models/。"
            "0.1B 文件约 365MB。不要放 LoRA adapter 或 HuggingFace 目录。",
            "2. 数据：ShareGPT JSON 放到 data/；也可以先用项目自带的"
            " examples/sharegpt_tiny.json 跑通链路。",
            "3. 验收：先用 rwkv-easy check-model 确认格式/结构/decay/出字，"
            "再用 inspect-data 确认只有 assistant 回复参与 loss。",
            "模型下载：HuggingFace 上的 RWKV 官方仓库（例如 RWKV/rwkv7-…）；"
            "下载前先看 docs/MODEL_CHECK.md 的格式说明。",
            "没有模型或数据也能进入向导，但开始训练前必须选择。",
        ),
    ),
    (
        "模板怎么选",
        (
            "[推荐] LoRA 标准微调：第一次用就选它，自动按显卡和权重大小选 batch/精度。",
            "长上下文 LoRA：长对话、长文档、长记忆，开启 infctx，8K–32K 上下文。",
            "State Tuning：小数据快速适配、人设/风格注入，参数 < 0.1%。",
            "省显存模式：8–12GB 显卡或大模型，batch=1 + 梯度检查点。",
            "先跑 20 步试试：只训练 20 步并自动合并，几分钟确认链路是否正常。",
        ),
    ),
    (
        "常见坑",
        (
            "CPU 只能做自检，正式训练请用 NVIDIA GPU；CPU 上必须 precision: fp32。",
            "CUDA out of memory：先降 micro_batch_size，再降 chunk_ctx / ctx_len，"
            "最后开 grad_checkpoint；用 grad_accum 补回总 batch。",
            "loss 不降：先 inspect-data 看 mask 是否正确，再调 lr，"
            "最后看启动日志里的 decay 校准（官方权重应选 clampw）。",
            "过拟合后所有回答都一样：推理 prompt 的 token 边界要和训练一致；"
            "用 rwkv-easy generate 测试，不要自己手拼前缀。",
            "更多性能与显存细节见 docs/PERFORMANCE.md，报错见 docs/TROUBLESHOOTING.md。",
        ),
    ),
)


def _guide(console: Console) -> None:
    if not console.tty:
        for title, paragraphs in GUIDE_PAGES:
            console.print()
            console.print(f"== {title} ==")
            for paragraph in paragraphs:
                console.print("  " + paragraph)
        console.pause("按回车返回…")
        return
    page = 0
    while True:
        title, paragraphs = GUIDE_PAGES[page]
        console.clear()
        console.header("新手引导", f"{page + 1} / {len(GUIDE_PAGES)}", step="指南")
        body: list[str] = []
        for paragraph in paragraphs:
            body.extend(_wrap_text(paragraph, max(20, console.width - 8)))
            body.append("")
        if body and body[-1] == "":
            body.pop()
        console.print()
        console.box(body, title=title)
        console.print()
        console.footer("←/→ 翻页   Enter 下一页   q 返回")
        key = console.read_key()
        if key in ("right", "enter", "down", "pagedown"):
            if page + 1 >= len(GUIDE_PAGES):
                return
            page += 1
        elif key in ("left", "up", "pageup"):
            page = max(0, page - 1)
        elif key in ("q", "esc"):
            return


def _welcome(console: Console, state: WizardState) -> str:
    intro = [
        f"设备：{state.env.device_label}",
        f"PyTorch：{state.env.torch_version or '未安装'}（CUDA {state.env.torch_cuda or '—'}）"
        f"   rwkv-fla：{'可用' if state.env.fla_available else '未安装'}",
        "接下来只需 3 步：① 选模板  ② 确认模型和数据集  ③ 开始训练。",
        "第一次用建议直接回车，采用推荐配置；想先了解选项可以看「新手引导」。",
    ]
    labels = ["使用推荐配置，下一步", "按场景选择模板", "查看新手引导（3 分钟）", "退出"]
    descriptions = [
        "按当前显卡和模型自动选择 LoRA + bf16-full + compile；大多数情况下这就是最佳起点。",
        "长上下文、State Tuning、省显存、只跑 20 步……按你的场景套用一套推荐参数。",
        "用 4 页说明工具能做什么、训练前准备什么、模板怎么选、常见坑。",
        "不保存任何配置，直接退出。",
    ]
    try:
        choice = console.menu(
            "欢迎使用 RWKV-EASY",
            labels,
            0,
            subtitle="RWKV-7 LoRA / State Tuning，支持 infctx",
            descriptions=descriptions,
            intro=intro,
            footer="↑↓ 选择   Enter 确认   g 新手引导   q 退出",
            extra_keys={"g": 2},
        )
    except WizardCancelled:
        return "quit"
    if choice == 0:
        apply_preset(state, "recommended")
        state.notice = "已使用推荐配置；可以直接开始，也可以继续修改"
        return "home"
    if choice == 1:
        _preset_menu(console, state)
        return "home"
    if choice == 2:
        _guide(console)
        return "home"
    return "quit"


def _keys_help(console: Console) -> None:
    lines = [
        "↑ / ↓        移动选项",
        "← / →        切换选项值",
        "Enter        编辑 / 进入 / 确认",
        "空格         勾选（多选界面）",
        "a / n        全选 / 清空（多选界面）",
        "PgUp / PgDn  翻页",
        "/            在高级表单里搜索参数",
        "g            打开新手引导",
        "p            打开配置模板",
        "?            显示这个帮助",
        "q / Esc      返回 / 退出",
    ]
    if not console.tty:
        console.print("\n".join(lines))
        return
    console.clear()
    console.header("按键说明", "所有界面通用", step="帮助")
    console.print()
    console.box(lines, title="快捷键")
    console.print()
    console.footer("按任意键返回")
    console.read_key()


def _run_action(
    console: Console,
    state: WizardState,
    action: str,
    no_train: bool,
    output: str | None,
) -> str | None:
    """执行首页的一项操作；返回 None 表示继续留在首页。"""
    cfg = state.cfg
    if action == "preset":
        _preset_menu(console, state)
        return None
    if action == "guide":
        _guide(console)
        return None
    if action == "method":
        if console.tty:
            _cycle_choice(state, "method", 1)
        else:
            current = 1 if cfg.train.method == "state" else 0
            pick = console.menu("训练方式", ["LoRA", "State Tuning"], current)
            state.set_method("state" if pick == 1 else "lora")
        return None
    if action == "infctx":
        if console.tty:
            _cycle_choice(state, "infctx", 1)
        else:
            state.set_infctx(console.confirm("启用 infctx 长上下文？", cfg.train.infctx))
        return None
    if action == "parallel_backward":
        cfg.train.infctx_parallel_backward = not cfg.train.infctx_parallel_backward
        state.mark_custom("多卡逐块反传已修改")
        return None
    if action == "backend":
        if console.tty:
            _cycle_choice(state, "backend", 1)
        else:
            choices = ["native"] + (["fla"] if state.env.fla_available else [])
            current = choices.index(cfg.model.backend) if cfg.model.backend in choices else 0
            cfg.model.backend = choices[console.menu("WKV 后端", choices, current)]
            state.mark_custom("WKV 后端已修改")
        return None
    if action == "precision":
        if console.tty:
            _cycle_choice(state, "precision", 1)
        else:
            choices = ["bf16-full", "bf16", "fp16", "fp32"] if state.env.is_cuda else ["fp32"]
            current = choices.index(cfg.train.precision) if cfg.train.precision in choices else 0
            cfg.train.precision = choices[console.menu("精度", choices, current)]
            state.mark_custom("精度已修改")
        return None
    if action == "model":
        _edit_model(console, state)
        return None
    if action == "data":
        _edit_dataset(console, state)
        return None
    if action == "modules":
        before = copy.deepcopy(cfg)
        _select_modules(console, state)
        if cfg != before:
            state.mark_custom("LoRA 模块已修改")
        return None
    if action == "layers":
        before = copy.deepcopy(cfg)
        _select_layers(console, state)
        if cfg != before:
            state.mark_custom("LoRA 层已修改")
        return None
    if action == "train":
        before = copy.deepcopy(cfg)
        _form(console, "训练参数", _training_fields(state), cfg)
        normalize_for_environment(cfg, state.env)
        if cfg != before:
            state.mark_custom("训练参数已修改")
        return None
    if action == "advanced":
        before = copy.deepcopy(cfg)
        _advanced_menu(console, state)
        if cfg != before:
            state.mark_custom("高级设置已修改")
        return None
    if action in ("start", "save"):
        result = _summary(console, state)
        if result == "quit":
            return "quit"
        if result == "back":
            return None
        path = _save_config(console, state, output)
        if path is None:
            return None
        if result == "save" or no_train:
            console.pause("配置已保存。按回车退出…")
            return "saved"
        return "train:" + str(path)
    return None


def _line_home(console: Console, state: WizardState, no_train: bool, output: str | None) -> str:
    """非 TTY 的分行问答版本（脚本 / CI / 管道输入）。"""
    while True:
        items = _home_items(state)
        labels = [f"{item.label}：{item.value}" for item in items]
        descriptions = [item.comment for item in items]
        try:
            choice = console.menu(
                "RWKV-EASY 一键配置",
                labels,
                0,
                subtitle="输入编号选择；q 退出",
                descriptions=descriptions,
            )
        except WizardCancelled:
            return "quit"
        result = _run_action(console, state, items[choice].action, no_train, output)
        if result is not None:
            return result


def _home(console: Console, state: WizardState, no_train: bool, output: str | None) -> str:
    index = 0
    label_width = 20
    while True:
        items = _home_items(state)
        value_width = max(14, min(30, console.width - 50))
        index = max(0, min(index, len(items) - 1))
        console.clear()
        console.header(
            "一键配置",
            f"{state.env.device_label} · torch {state.env.torch_version or '未安装'}",
            step="第 1 步",
        )
        if state.notice:
            console.print(console.colorize("  ✓ " + state.notice, "32"))
            state.notice = ""
        window = max(6, console.height - 13)
        start = max(0, min(index - window // 2, max(0, len(items) - window)))
        end = min(len(items), start + window)
        if start > 0:
            console.print(console.colorize(f"  ↑ 还有 {start} 项", "90"))
        for i in range(start, end):
            item = items[i]
            marker = "▶" if i == index else " "
            label = _pad(item.label, label_width)
            raw_value = f"◀ {item.value} ▶" if item.kind == "choice" else item.value
            value = _pad(_truncate(raw_value, value_width), value_width)
            prefix = f" {marker} {label} {value}"
            room = console.width - 1 - text_width(prefix)
            comment = _truncate("  " + item.comment, max(0, room)) if item.comment else ""
            if i == index:
                console.print(
                    console.colorize(f" {marker} {label}", "1;32")
                    + console.colorize(f" {value}", "1;32")
                    + console.colorize(comment, "90")
                )
            else:
                console.print(f" {marker} {label} {value}" + console.colorize(comment, "90"))
        if end < len(items):
            console.print(console.colorize(f"  ↓ 还有 {len(items) - end} 项", "90"))
        current = items[index]
        console.print()
        console.note(current.help or current.comment, title="说明")
        console.print()
        console.footer("↑↓ 移动   ←→ 切换   Enter 编辑/进入   g 引导   p 模板   ? 按键   q 退出")
        key = console.read_key()

        if key == "up":
            index = (index - 1) % len(items)
            continue
        if key == "down":
            index = (index + 1) % len(items)
            continue
        if key == "home":
            index = 0
            continue
        if key == "end":
            index = len(items) - 1
            continue
        if key in ("pageup", "pagedown"):
            step = max(1, window - 1)
            index = max(0, min(len(items) - 1, index + (step if key == "pagedown" else -step)))
            continue
        if key in ("q", "esc"):
            return "quit"
        if key == "g":
            _guide(console)
            continue
        if key == "p":
            _preset_menu(console, state)
            continue
        if key == "?":
            _keys_help(console)
            continue

        if key in ("left", "right"):
            if current.kind == "choice":
                _cycle_choice(state, current.action, 1 if key == "right" else -1)
            continue
        if key != "enter":
            continue
        result = _run_action(console, state, current.action, no_train, output)
        if result is not None:
            return result


def _cycle_choice(state: WizardState, action: str, direction: int) -> None:
    cfg = state.cfg
    if action == "method":
        methods = ["lora", "state"]
        current = methods.index(cfg.train.method if cfg.train.method in methods else "lora")
        state.set_method(methods[(current + direction) % len(methods)])
    elif action == "infctx":
        state.set_infctx(not cfg.train.infctx)
    elif action == "parallel_backward":
        cfg.train.infctx_parallel_backward = not cfg.train.infctx_parallel_backward
        state.mark_custom("多卡逐块反传已修改")
    elif action == "backend":
        choices = ["native"] + (["fla"] if state.env.fla_available else [])
        current = choices.index(cfg.model.backend) if cfg.model.backend in choices else 0
        cfg.model.backend = choices[(current + direction) % len(choices)]
        state.mark_custom("WKV 后端已修改")
    elif action == "precision":
        choices = ["bf16-full", "bf16", "fp16", "fp32"] if state.env.is_cuda else ["fp32"]
        current = choices.index(cfg.train.precision) if cfg.train.precision in choices else 0
        cfg.train.precision = choices[(current + direction) % len(choices)]
        state.mark_custom("精度已修改")
    normalize_for_environment(cfg, state.env)


def run_wizard(
    *,
    non_interactive: bool = False,
    no_train: bool = False,
    output: str | None = None,
    welcome: bool = True,
    infctx_parallel_backward: bool | None = None,
) -> int:
    """运行交互式向导；返回进程退出码。"""
    if not non_interactive and sys.stdin.isatty():
        print("正在检测环境并扫描模型（首次启动约几秒）…", flush=True)
    env = detect_environment()
    cfg = default_config(env)
    normalize_for_environment(cfg, env)
    state = WizardState(env=env, cfg=cfg)
    # 有候选时自动填入第一个，减少点击；用户仍可改。
    models = find_model_candidates()
    if models:
        with contextlib.suppress(WizardError):
            state.set_model(inspect_model(str(models[0])))
    datasets = find_dataset_candidates()
    if datasets:
        with contextlib.suppress(WizardError):
            state.set_dataset(validate_dataset_path(str(datasets[0])))
    # 推荐模板要在模型信息之后套用，才能按权重大小选 batch。
    apply_preset(state, "recommended")
    if infctx_parallel_backward is not None:
        state.cfg.train.infctx_parallel_backward = bool(infctx_parallel_backward)
    state.notice = ""

    with Console(non_interactive=non_interactive) as console:
        if console.tty:
            result = _welcome(console, state) if welcome else "home"
            if result != "quit":
                result = _home(console, state, no_train, output)
        else:
            console.print("RWKV-EASY 一键配置（非交互模式）")
            console.print(f"设备：{env.device_label}")
            console.print(
                f"PyTorch：{env.torch_version or '未安装'}（CUDA {env.torch_cuda or '—'}）"
            )
            console.print(f"rwkv-fla：{'可用' if env.fla_available else '未安装'}")
            if not env.is_cuda:
                console.print("提示：CPU 只能做自检，正式训练请用 NVIDIA GPU。")
            result = _line_home(console, state, no_train, output)
        if result.startswith("train:"):
            path = Path(result.split(":", 1)[1])
            return _run_training(console, state, path)
    if result == "saved":
        print("\n配置已保存，未开始训练。")
        return 0
    if result == "quit":
        print("\n已退出，未开始训练。")
        return 0
    return 0


# --------------------------------------------------------------------------- #
# CLI 入口（也便于测试）
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="RWKV-EASY 交互式配置向导")
    parser.add_argument("--non-interactive", action="store_true", help="分行问答（适合脚本/无 TTY）")
    parser.add_argument("--no-train", action="store_true", help="只生成配置，不启动训练")
    parser.add_argument("--no-welcome", action="store_true", help="跳过欢迎页，直接进入配置")
    parser.add_argument(
        "--infctx-parallel-backward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="DDP + infctx 是否逐块反传（默认 true；仅 LoRA）",
    )
    parser.add_argument("-o", "--output", default=None, help="配置文件输出路径")
    args = parser.parse_args(argv)
    try:
        return run_wizard(
            non_interactive=args.non_interactive,
            no_train=args.no_train,
            output=args.output,
            welcome=not args.no_welcome,
            infctx_parallel_backward=args.infctx_parallel_backward,
        )
    except (WizardCancelled, KeyboardInterrupt):
        print("\n已取消。")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
