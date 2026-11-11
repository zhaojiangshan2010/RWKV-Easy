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
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import Config, ConfigError

__all__ = [
    "DatasetInfo",
    "Environment",
    "ModelInfo",
    "WizardState",
    "cuda_tag_candidates",
    "default_config",
    "detect_environment",
    "run_wizard",
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
    result: dict[Path, float] = {}
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.glob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_MODEL_SUFFIXES:
                result[path.resolve()] = path.stat().st_mtime
    return [p for p, _ in sorted(result.items(), key=lambda item: item[1], reverse=True)][:20]


def find_dataset_candidates() -> list[Path]:
    roots = [Path("data"), Path("examples"), Path.cwd()]
    result: dict[Path, float] = {}
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.iterdir():
            if path.name.startswith("."):
                continue
            if path.is_dir():
                # 只把 data/、examples/ 下的目录当成数据集；cwd 下不扫描所有目录
                if root.name in ("data", "examples"):
                    result[path.resolve()] = path.stat().st_mtime
            elif path.name.endswith(".jsonl.gz") or path.suffix.lower() in SUPPORTED_DATA_SUFFIXES:
                # cwd 下只自动识别 JSON/JSONL，避免把 requirements.txt 当成数据
                if root == Path.cwd() and path.suffix.lower() not in (".json", ".jsonl"):
                    continue
                result[path.resolve()] = path.stat().st_mtime
    return [p for p, _ in sorted(result.items(), key=lambda item: item[1], reverse=True)][:20]


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


@dataclass
class WizardState:
    env: Environment
    cfg: Config
    model_info: ModelInfo | None = None
    dataset_info: DatasetInfo | None = None

    def set_method(self, method: str) -> None:
        apply_method_defaults(self.cfg, method, self.cfg.train.infctx)
        normalize_for_environment(self.cfg, self.env)

    def set_infctx(self, infctx: bool) -> None:
        apply_method_defaults(self.cfg, self.cfg.train.method, infctx)
        normalize_for_environment(self.cfg, self.env)

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


def config_summary(state: WizardState) -> str:
    cfg = state.cfg
    lines = [
        f"训练方式    : {state.method_label()}{' + infctx' if cfg.train.infctx else ''}",
        f"模型        : {cfg.model.path or '未选择'}",
        f"数据集      : {cfg.data.path or '未选择'}",
        f"后端 / 精度 : {cfg.model.backend} / {cfg.train.precision}",
        f"batch       : micro {cfg.train.micro_batch_size} × accum {cfg.train.grad_accum}"
        f" → {state.tokens_per_step()} token/step",
        f"上下文      : ctx {cfg.train.ctx_len}"
        + (f" / chunk {cfg.train.chunk_ctx}" if cfg.train.infctx else ""),
        f"学习率      : {cfg.train.lr:g} → {cfg.train.min_lr:g}（{cfg.train.lr_schedule}）",
        f"步数        : epochs {cfg.train.epochs}"
        + (f" / max_steps {cfg.train.max_steps}" if cfg.train.max_steps else ""),
        f"compile     : {cfg.train.compile} ({cfg.train.compile_mode})",
        f"输出目录    : {cfg.run.output_dir}/{cfg.run.name}",
    ]
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

    def __enter__(self) -> Console:
        if self.tty:
            self._enable_vt()
            self.write("\033[?25l")  # hide cursor
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.tty:
            self.write("\033[?25h\033[0m\n")

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
            self.write("\033[2J\033[H")

    def title(self, text: str, subtitle: str = "") -> None:
        line = "=" * 64
        self.print(self.colorize(line, "36"))
        self.print(self.colorize(f" {text}", "1;36"))
        if subtitle:
            self.print(f" {subtitle}")
        self.print(self.colorize(line, "36"))

    def footer(self, text: str) -> None:
        self.print(self.colorize(" " + text, "90"))

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
    ) -> int:
        if not items:
            raise WizardError("菜单是空的")
        if not self.tty:
            return self._line_menu(title, items, default, subtitle)
        index = max(0, min(default, len(items) - 1))
        while True:
            self.clear()
            self.title(title, subtitle)
            for i, item in enumerate(items):
                marker = "▶" if i == index else " "
                line = f" {marker} {item}"
                self.print(self.colorize(line, "1;32") if i == index else line)
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
            elif key == "enter":
                return index
            elif key in ("q", "esc"):
                raise WizardCancelled

    def _line_menu(self, title: str, items: Sequence[str], default: int, subtitle: str) -> int:
        self.print()
        self.print(title)
        if subtitle:
            self.print(subtitle)
        for i, item in enumerate(items):
            self.print(f"  {i + 1}) {item}")
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
        window = 12
        while True:
            self.clear()
            self.title(title, subtitle)
            start = max(0, min(index - window // 2, max(0, len(items) - window)))
            end = min(len(items), start + window)
            if start > 0:
                self.print("  …")
            for i in range(start, end):
                box = "[x]" if i in chosen else "[ ]"
                marker = "▶" if i == index else " "
                line = f" {marker} {box} {items[i]}"
                self.print(self.colorize(line, "1;32") if i == index else line)
            if end < len(items):
                self.print("  …")
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
    help: str = ""
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
            return "None"
        if isinstance(value, bool):
            return "开" if value else "关"
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


def _form(console: Console, title: str, fields: Sequence[FormField], root: Any, subtitle: str = "") -> None:
    if not fields:
        return
    if not console.tty:
        _line_form(console, title, fields, root, subtitle)
        return
    index = 0
    window = 12
    while True:
        console.clear()
        console.title(title, subtitle)
        start = max(0, min(index - window // 2, max(0, len(fields) - window)))
        end = min(len(fields), start + window)
        if start > 0:
            console.print("  …")
        for i in range(start, end):
            item = fields[i]
            marker = "▶" if i == index else " "
            line = f" {marker} {item.label:<22} {item.display(root)}"
            console.print(console.colorize(line, "1;32") if i == index else line)
        if end < len(fields):
            console.print("  …")
        current = fields[index]
        if current.help:
            console.print()
            console.print(console.colorize("  " + current.help, "90"))
        console.print()
        console.footer("↑↓ 移动   ←→ 调整   Enter 输入精确值/切换   q 返回")
        key = console.read_key()
        if key == "up":
            index = (index - 1) % len(fields)
        elif key == "down":
            index = (index + 1) % len(fields)
        elif key == "home":
            index = 0
        elif key == "end":
            index = len(fields) - 1
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
        console.print(f"  {i + 1:2d}) {item.label:<22} {item.display(root)}")
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


def _path_candidates(candidates: Sequence[Path], manual_label: str) -> list[str]:
    labels = [str(path) for path in candidates]
    labels.append(manual_label)
    return labels


def _choose_path(
    console: Console,
    title: str,
    candidates: Sequence[Path],
    current: str,
    validator: Callable[[str], Any],
    manual_label: str = "手动输入路径…",
) -> Any:
    labels = _path_candidates(candidates, manual_label)
    default = 0
    if current:
        current_path = _expand(current)
        for i, path in enumerate(candidates):
            if path.resolve() == current_path:
                default = i
                break
    choice = console.menu(title, labels, default, subtitle=f"当前：{current or '未选择'}")
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


def _edit_model(console: Console, state: WizardState) -> None:
    def _validator(text: str) -> ModelInfo:
        if not text:
            raise WizardError("路径为空")
        return inspect_model(text)

    try:
        info = _choose_path(
            console,
            "选择 RWKV-7 权重",
            find_model_candidates(),
            state.cfg.model.path or "",
            _validator,
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

    try:
        info = _choose_path(
            console,
            "选择 ShareGPT / messages / 纯文本数据",
            find_dataset_candidates(),
            state.cfg.data.path or "",
            _validator,
        )
    except WizardCancelled:
        return
    except WizardError as exc:
        console.print(console.colorize(f"\n{exc}", "31"))
        console.pause()
        return
    state.set_dataset(info)
    console.print(console.colorize(f"\n已选择：{info.kind}", "32"))
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
            subtitle="每层最多 6 个；默认全部。",
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
        choice = console.menu("选择 LoRA 目标 block", options, default)
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
        FormField("训练方式", "train.method", "choice", choices),
        FormField("infctx（长上下文）", "train.infctx", "bool"),
        FormField("ctx_len", "train.ctx_len", "int", step=512, minimum=64),
        FormField("chunk_ctx（infctx）", "train.chunk_ctx", "int", step=256, minimum=16),
        FormField("micro_batch_size", "train.micro_batch_size", "int", minimum=1),
        FormField("grad_accum", "train.grad_accum", "int", minimum=1),
        FormField("epochs", "train.epochs", "int", minimum=1),
        FormField("max_steps", "train.max_steps", "optional_int", minimum=1),
        FormField("lr", "train.lr", "float", step=1.0e-5, minimum=0.0),
        FormField("min_lr", "train.min_lr", "float", step=1.0e-6, minimum=0.0),
        FormField("warmup_steps", "train.warmup_steps", "int", minimum=0),
        FormField(
            "lr_schedule",
            "train.lr_schedule",
            "choice",
            ["cosine", "linear", "constant"],
        ),
        FormField("weight_decay", "train.weight_decay", "float", step=1.0e-3, minimum=0.0),
        FormField("beta1", "train.beta1", "float", step=0.01, minimum=0.0, maximum=1.0),
        FormField("beta2", "train.beta2", "float", step=0.01, minimum=0.0, maximum=1.0),
        FormField("grad_clip", "train.grad_clip", "float", step=0.1, minimum=0.0),
        FormField("precision", "train.precision", "choice", precision),
        FormField("grad_checkpoint", "train.grad_checkpoint", "bool"),
        FormField("l2wrap", "train.l2wrap", "float", step=1.0e-5, minimum=0.0),
        FormField("infctx_detach", "train.infctx_detach", "bool"),
        FormField("infctx_backward", "train.infctx_backward", "bool"),
        FormField("log_every", "train.log_every", "int", minimum=1),
        FormField("save_every_steps", "train.save_every_steps", "int", minimum=0),
        FormField("save_every_epochs", "train.save_every_epochs", "int", minimum=0),
        FormField("keep_last", "train.keep_last", "int", minimum=1),
        FormField("merge_at_end", "train.merge_at_end", "bool"),
        FormField("val_every_steps", "train.val_every_steps", "int", minimum=0),
        FormField("compile", "train.compile", "bool"),
        FormField(
            "compile_mode",
            "train.compile_mode",
            "choice",
            ["default", "reduce-overhead", "max-autotune"],
        ),
        FormField("device", "train.device", "choice", ["auto", "cuda", "cpu"]),
        FormField("run.seed", "run.seed", "int", minimum=0),
        FormField("输出目录", "run.output_dir", "text"),
        FormField("实验名", "run.name", "text"),
    ]


def _model_fields(state: WizardState) -> list[FormField]:
    backends = ["native"]
    if state.env.is_cuda and state.env.fla_available:
        backends.insert(0, "fla")
    return [
        FormField("权重路径", "model.path", "text"),
        FormField("random_init", "model.random_init", "bool"),
        FormField("n_layer", "model.n_layer", "optional_int", minimum=1),
        FormField("n_embd", "model.n_embd", "optional_int", minimum=1),
        FormField("vocab_size", "model.vocab_size", "optional_int", minimum=1),
        FormField("head_size", "model.head_size", "optional_int", minimum=1),
        FormField("decay", "model.decay", "choice", ["auto", "clampw", "softplus", "exp"]),
        FormField("WKV 后端", "model.backend", "choice", backends),
        FormField("wkv_chunk（native）", "model.wkv_chunk", "int", step=8, minimum=8, maximum=96),
        FormField("head_size_divisor", "model.head_size_divisor", "int", minimum=1),
        FormField("dtype 覆盖", "model.dtype", "optional_text"),
    ]


def _data_fields(state: WizardState) -> list[FormField]:
    return [
        FormField("数据路径", "data.path", "text"),
        FormField("格式", "data.format", "choice", ["auto", "sharegpt", "messages", "text"]),
        FormField("mode", "data.mode", "choice", ["packed", "sample"]),
        FormField("模板", "data.template", "choice", ["rwkv_world", "chatml", "custom"]),
        FormField("train_on", "data.train_on", "choice", ["assistant", "last_assistant"]),
        FormField("append_eos", "data.append_eos", "choice", ["end", "each_assistant", "never"]),
        FormField("truncation（sample）", "data.truncation", "choice", ["head", "tail", "random"]),
        FormField("val_ratio", "data.val_ratio", "float", step=0.01, minimum=0.0, maximum=0.9),
        FormField("cache", "data.cache", "bool"),
        FormField("cache_dir", "data.cache_dir", "optional_text"),
        FormField("vocab_path", "data.vocab_path", "optional_text"),
        FormField("eos_token_id", "data.eos_token_id", "optional_int", minimum=0),
        FormField("num_workers", "data.num_workers", "int", minimum=0),
        FormField("num_proc", "data.num_proc", "int", minimum=0),
        FormField("pin_memory", "data.pin_memory", "bool"),
        FormField("text_field", "data.text_field", "text"),
        FormField("max_samples", "data.max_samples", "optional_int", minimum=1),
        FormField("drop_empty", "data.drop_empty", "bool"),
        FormField("system_prefix", "data.system_prefix", "optional_text"),
        FormField("user_prefix", "data.user_prefix", "optional_text"),
        FormField("assistant_prefix", "data.assistant_prefix", "optional_text"),
        FormField("turn_separator", "data.turn_separator", "optional_text"),
    ]


def _misc_fields(state: WizardState) -> list[FormField]:
    modes = ["auto", "single", "ddp"]
    if state.env.deepspeed_available:
        modes.append("deepspeed")
    return [
        FormField("dist.mode", "dist.mode", "choice", modes),
        FormField("num_gpus", "dist.num_gpus", "text"),
        FormField("master_port", "dist.master_port", "int", minimum=1, maximum=65535),
        FormField("find_unused_parameters", "dist.find_unused_parameters", "bool"),
        FormField("zero_stage", "dist.deepspeed.zero_stage", "int", minimum=1, maximum=3),
        FormField("offload_optimizer", "dist.deepspeed.offload_optimizer", "bool"),
        FormField("offload_param", "dist.deepspeed.offload_param", "bool"),
        FormField("bucket_size", "dist.deepspeed.bucket_size", "float", step=1.0e7, minimum=1.0),
        FormField("contiguous_gradients", "dist.deepspeed.contiguous_gradients", "bool"),
        FormField("overlap_comm", "dist.deepspeed.overlap_comm", "bool"),
        FormField("log.jsonl", "log.jsonl", "bool"),
        FormField("log.wandb", "log.wandb", "bool"),
        FormField("wandb_project", "log.wandb_project", "text"),
        FormField("wandb_run", "log.wandb_run", "optional_text"),
        FormField("log.level", "log.level", "choice", ["info", "debug", "warning"]),
        FormField("run.deterministic", "run.deterministic", "bool"),
    ]


def _lora_fields(state: WizardState) -> list[FormField]:
    return [
        FormField("LoRA rank", "lora.rank", "int", minimum=1, maximum=512),
        FormField("LoRA alpha", "lora.alpha", "int", minimum=1, maximum=1024),
        FormField("LoRA dropout", "lora.dropout", "float", step=0.01, minimum=0.0, maximum=0.9),
        FormField("target_regex", "lora.target_regex", "optional_text"),
        FormField("State init_std", "state.init_std", "float", step=0.001, minimum=0.0),
    ]


def _advanced_menu(console: Console, state: WizardState) -> None:
    options = [
        "训练参数（lr / batch / 精度 / infctx / 保存…）",
        "模型与 WKV 后端（decay / backend / chunk…）",
        "数据（格式 / 模板 / mask / workers…）",
        "分布式 / 日志 / 输出（DDP / DeepSpeed / wandb…）",
        "LoRA / State 参数（rank / alpha / dropout…）",
        "返回",
    ]
    while True:
        try:
            choice = console.menu("高级设置（所有字段都可以改）", options, 0)
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
    while True:
        console.clear()
        console.title("确认配置", "检查无误后开始训练")
        for line in config_summary(state).splitlines():
            console.print("  " + line)
        errors = state.validate()
        console.print()
        if errors:
            console.print(console.colorize("  ⚠ 需要修正：", "31"))
            for error in errors:
                console.print(console.colorize("    - " + error, "31"))
        else:
            console.print(console.colorize("  ✓ 配置检查通过", "32"))
        console.print()
        options = [
            "开始训练（自动保存配置）",
            "仅保存配置，不训练",
            "返回修改",
            "退出",
        ]
        if not console.tty:
            print("\n".join(f"  {i + 1}) {item}" for i, item in enumerate(options)))
            raw = input("选择（默认 1）：").strip() or "1"
            try:
                choice = int(raw) - 1
            except ValueError:
                continue
        else:
            try:
                choice = console.menu("下一步", options, 0)
            except WizardCancelled:
                return "quit"
        if choice == 0:
            if errors:
                console.print(console.colorize("\n还有错误，先修正配置。", "31"))
                console.pause()
                return "back"
            return "train"
        if choice == 1:
            if errors:
                console.print(console.colorize("\n还有错误，先修正配置。", "31"))
                console.pause()
                return "back"
            return "save"
        if choice == 2:
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
    cfg.save_yaml(path, header="# 由 RWKV-EASY 交互式向导生成\n# 所有字段都可以手动修改")
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


def _home_items(state: WizardState) -> list[tuple[str, str, str, str]]:
    cfg = state.cfg
    method_value = "State Tuning" if cfg.train.method == "state" else "LoRA"
    infctx_value = "开启" if cfg.train.infctx else "关闭"
    items: list[tuple[str, str, str, str]] = [
        ("choice", "训练方式", method_value, "method"),
        ("choice", "长上下文（infctx）", infctx_value, "infctx"),
        ("path", "模型权重", cfg.model.path or "未选择", "model"),
        ("path", "数据集", cfg.data.path or "未选择", "data"),
    ]
    if cfg.train.method == "lora":
        items.append(("action", "LoRA 模块", state.modules_summary(), "modules"))
        items.append(("action", "LoRA 层", state.layers_summary(), "layers"))
    items.extend(
        [
            ("choice", "WKV 后端", cfg.model.backend, "backend"),
            ("choice", "精度", cfg.train.precision, "precision"),
            (
                "action",
                "训练参数摘要",
                f"bsz {cfg.train.micro_batch_size} × {cfg.train.grad_accum} / ctx {cfg.train.ctx_len}",
                "train",
            ),
            ("action", "高级设置", "所有配置字段…", "advanced"),
            ("action", "开始训练", "保存配置并启动", "start"),
            ("action", "仅保存配置", "生成 YAML", "save"),
        ]
    )
    return items


def _line_home(console: Console, state: WizardState, no_train: bool, output: str | None) -> str:
    """非 TTY 的分行问答版本（脚本 / CI / 管道输入）。"""
    while True:
        items = _home_items(state)
        labels = [f"{label}：{value}" for _kind, label, value, _action in items]
        try:
            choice = console.menu(
                "RWKV-EASY 一键配置",
                labels,
                0,
                subtitle="输入编号选择；q 退出",
            )
        except WizardCancelled:
            return "quit"
        action = items[choice][3]
        if action == "method":
            current = 1 if state.cfg.train.method == "state" else 0
            pick = console.menu("训练方式", ["LoRA", "State Tuning"], current)
            state.set_method("state" if pick == 1 else "lora")
        elif action == "infctx":
            state.set_infctx(console.confirm("启用 infctx 长上下文？", state.cfg.train.infctx))
        elif action == "model":
            _edit_model(console, state)
        elif action == "data":
            _edit_dataset(console, state)
        elif action == "modules":
            _select_modules(console, state)
        elif action == "layers":
            _select_layers(console, state)
        elif action == "backend":
            choices = ["native"] + (["fla"] if state.env.fla_available else [])
            current = choices.index(state.cfg.model.backend) if state.cfg.model.backend in choices else 0
            state.cfg.model.backend = choices[console.menu("WKV 后端", choices, current)]
        elif action == "precision":
            choices = ["bf16-full", "bf16", "fp16", "fp32"] if state.env.is_cuda else ["fp32"]
            current = (
                choices.index(state.cfg.train.precision)
                if state.cfg.train.precision in choices
                else 0
            )
            state.cfg.train.precision = choices[console.menu("精度", choices, current)]
        elif action == "train":
            _form(console, "训练参数", _training_fields(state), state.cfg)
            normalize_for_environment(state.cfg, state.env)
        elif action == "advanced":
            _advanced_menu(console, state)
        elif action in ("start", "save"):
            result = _summary(console, state)
            if result == "quit":
                return "quit"
            if result == "back":
                continue
            path = _save_config(console, state, output)
            if path is None:
                continue
            if result == "save" or no_train:
                console.pause("配置已保存。按回车退出…")
                return "saved"
            return "train:" + str(path)


def _home(console: Console, state: WizardState, no_train: bool, output: str | None) -> str:
    index = 0
    while True:
        items = _home_items(state)
        index = max(0, min(index, len(items) - 1))
        console.clear()
        console.title(
            "RWKV-EASY 一键配置",
            f"{state.env.device_label} | Python {state.env.python_version} | "
            f"torch {state.env.torch_version or '未安装'}",
        )
        for i, (kind, label, value, _key) in enumerate(items):
            marker = "▶" if i == index else " "
            if kind == "choice":
                line = f" {marker} {label:<20} ◀ {value} ▶"
            else:
                line = f" {marker} {label:<20} {value}"
            console.print(console.colorize(line, "1;32") if i == index else line)
        console.print()
        console.footer("↑↓ 移动   ←→ 修改选项   Enter 编辑/进入   q 退出")
        key = console.read_key()

        if key == "up":
            index = (index - 1) % len(items)
            continue
        if key == "down":
            index = (index + 1) % len(items)
            continue
        if key in ("q", "esc"):
            return "quit"

        kind, _label, _value, action = items[index]
        if key in ("left", "right"):
            if kind == "choice":
                _cycle_choice(state, action, 1 if key == "right" else -1)
            continue
        if key != "enter":
            continue
        if kind == "choice":
            _cycle_choice(state, action, 1)
            continue
        if action == "model":
            _edit_model(console, state)
        elif action == "data":
            _edit_dataset(console, state)
        elif action == "modules":
            _select_modules(console, state)
        elif action == "layers":
            _select_layers(console, state)
        elif action == "train":
            _form(console, "训练参数", _training_fields(state), state.cfg)
            normalize_for_environment(state.cfg, state.env)
        elif action == "advanced":
            _advanced_menu(console, state)
        elif action in ("start", "save"):
            result = _summary(console, state)
            if result == "quit":
                return "quit"
            if result == "back":
                continue
            path = _save_config(console, state, output)
            if path is None:
                continue
            if result == "save" or no_train:
                console.pause("配置已保存。按回车退出…")
                return "saved"
            return "train:" + str(path)

def _cycle_choice(state: WizardState, action: str, direction: int) -> None:
    cfg = state.cfg
    if action == "method":
        methods = ["lora", "state"]
        current = methods.index(cfg.train.method if cfg.train.method in methods else "lora")
        state.set_method(methods[(current + direction) % len(methods)])
    elif action == "infctx":
        state.set_infctx(not cfg.train.infctx)
    elif action == "backend":
        choices = ["native"] + (["fla"] if state.env.fla_available else [])
        current = choices.index(cfg.model.backend) if cfg.model.backend in choices else 0
        cfg.model.backend = choices[(current + direction) % len(choices)]
    elif action == "precision":
        choices = ["bf16-full", "bf16", "fp16", "fp32"] if state.env.is_cuda else ["fp32"]
        current = choices.index(cfg.train.precision) if cfg.train.precision in choices else 0
        cfg.train.precision = choices[(current + direction) % len(choices)]
    normalize_for_environment(cfg, state.env)


def run_wizard(
    *,
    non_interactive: bool = False,
    no_train: bool = False,
    output: str | None = None,
) -> int:
    """运行交互式向导；返回进程退出码。"""
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

    with Console(non_interactive=non_interactive) as console:
        console.clear()
        console.title(
            "RWKV-EASY 一键配置向导",
            "只做训练：LoRA / State Tuning，可叠加 infctx",
        )
        console.print()
        console.print(f"  设备：{env.device_label}")
        console.print(f"  PyTorch：{env.torch_version or '未安装'}（CUDA {env.torch_cuda or '—'}）")
        console.print(f"  rwkv-fla：{'可用' if env.fla_available else '未安装'}")
        console.print(f"  DeepSpeed：{'可用' if env.deepspeed_available else '未安装'}")
        if not env.is_cuda:
            console.print(console.colorize("  提示：CPU 只能做自检，正式训练请用 NVIDIA GPU。", "33"))
        console.print()
        console.pause("按回车进入配置…")
        if console.tty:
            result = _home(console, state, no_train, output)
        else:
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
    parser.add_argument("-o", "--output", default=None, help="配置文件输出路径")
    args = parser.parse_args(argv)
    try:
        return run_wizard(
            non_interactive=args.non_interactive,
            no_train=args.no_train,
            output=args.output,
        )
    except (WizardCancelled, KeyboardInterrupt):
        print("\n已取消。")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
