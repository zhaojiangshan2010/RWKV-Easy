"""配置系统：dataclass 定义 + YAML 读写 + ``--set`` 覆盖 + 严格校验。

设计目标：

* **小白友好**：所有选项都有中文注释模板，未知字段会给出“你是不是想写 xxx”的提示。
* **可复现**：训练开始时会把手写配置与解析后的配置一起写进输出目录。
* **不静默出错**：非法组合在启动前一次性报清楚，而不是训练到一半才崩。
"""

from __future__ import annotations

import dataclasses
import difflib
import re
import types
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from .runtime import normalize_precision

# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #


class ConfigError(ValueError):
    """配置错误（信息面向最终用户）。"""


# --------------------------------------------------------------------------- #
# 各段配置
# --------------------------------------------------------------------------- #


@dataclass
class RunConfig:
    """实验标识与输出位置。"""

    name: str = "rwkv-easy"
    output_dir: str = "runs"
    seed: int = 42
    deterministic: bool = False


@dataclass
class ModelConfig:
    """模型与 WKV 算子。"""

    # RWKV-7 权重路径（.pth / .safetensors）。留空则必须在配置里给全维度。
    path: str | None = None
    # 随机初始化一个小模型（不加载权重），用于 CPU 冒烟测试。
    random_init: bool = False
    # 以下维度默认从 checkpoint 自动推断；只有做随机初始化或特殊 checkpoint 才需要手填。
    n_layer: int | None = None
    n_embd: int | None = None
    vocab_size: int | None = None
    head_size: int | None = None
    # GroupNorm eps = 1e-5 * head_size_divisor^2，官方值 8。
    head_size_divisor: int = 8
    # 衰减公式：auto 会用内置校准文本自动判断，避免与权重不匹配。
    #   auto | clampw | softplus | exp
    decay: str = "auto"
    # WKV 后端：native（纯 PyTorch，通用）/ reference（逐 token，测试用）/ fla（Triton，可选）
    backend: str = "native"
    # 分块内部长度。只影响速度/显存，不影响数学结果。
    wkv_chunk: int = 64
    # 可选：加载后把权重转换成指定 dtype（fp32/bf16/fp16）。默认跟随 train.precision。
    dtype: str | None = None


@dataclass
class TrainConfig:
    """训练超参数。"""

    # lora | state
    method: str = "lora"
    # 是否启用 infinite context（分块训练）。
    infctx: bool = False
    # 单个样本的总长度。
    ctx_len: int = 2048
    # infctx 每次前向的块长；必须 <= ctx_len。
    chunk_ctx: int = 512
    micro_batch_size: int = 2
    grad_accum: int = 8
    epochs: int = 1
    max_steps: int | None = None
    lr: float = 1.0e-4
    min_lr: float = 1.0e-5
    warmup_steps: int = 20
    # cosine | linear | constant
    lr_schedule: str = "cosine"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0
    # fp32 | fp16 | bf16 | bf16-full
    precision: str = "bf16"
    # 激活重计算，省显存、略慢。
    grad_checkpoint: bool = True
    # RWKV 的 L2Wrap 稳定项；设 0 可关闭。
    l2wrap: float = 1.0e-4
    # infctx 块之间是否截断梯度（推荐 true；false 显存需求成倍上升）。
    infctx_detach: bool = True
    # infctx 每个 chunk 立即 backward（显存只与 chunk_ctx 有关）。
    # 单进程训练默认启用；多卡是否逐块由 infctx_parallel_backward 控制。
    infctx_backward: bool = True
    # DDP + infctx 时，每个 chunk 立即 backward，只有最后一块触发梯度 all-reduce。
    # 显存 O(chunk_ctx)，不再随 ctx_len 增长；DeepSpeed / State Tuning 暂不支持。
    infctx_parallel_backward: bool = True
    log_every: int = 10
    save_every_steps: int = 0
    save_every_epochs: int = 1
    keep_last: int = 2
    # 训练结束自动把适配器/状态合并成完整权重。
    merge_at_end: bool = True
    # 训练后做一次验证集评估（data.val_ratio > 0 时有效）。
    val_every_steps: int = 0
    # torch.compile（编译训练前向；首次启动慢）。
    compile: bool = False
    # default | reduce-overhead | max-autotune
    compile_mode: str = "default"
    # auto | cpu | cuda | cuda:0 ...
    device: str = "auto"


@dataclass
class LoRAConfig:
    """LoRA 超参数。"""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    # 按模块名后缀匹配。默认覆盖 attention 与 channel-mix 的线性层（每层 6 个）。
    target_modules: list[str] = field(
        default_factory=lambda: [
            "att.receptance",
            "att.key",
            "att.value",
            "att.output",
            "ffn.key",
            "ffn.value",
        ]
    )
    # 可选：额外用正则匹配（与 target_modules 取并集）。
    target_regex: str | None = None
    # 可选：只注入这些 block 下标（0-based）；None = 所有层。
    target_layers: list[int] | None = None


@dataclass
class StateConfig:
    """State Tuning 超参数。"""

    # time_state 的初始化标准差；0 表示零初始化（官方做法，推荐）。
    init_std: float = 0.0


@dataclass
class DataConfig:
    """数据与样本构造。"""

    # ShareGPT JSON / JSONL；也支持纯文本 .txt / 带 text 字段的 JSONL。
    path: str = ""
    # auto | sharegpt | messages | text
    format: str = "auto"
    # packed（拼成定长流，推荐，兼容 infctx）| sample（单条对话 + 动态 padding）
    mode: str = "packed"
    # 纯文本数据使用的字段名。
    text_field: str = "text"
    max_samples: int | None = None
    val_ratio: float = 0.0
    # 是否缓存 tokenize 结果（强烈建议开启）。
    cache: bool = True
    cache_dir: str | None = None
    # 词表路径；留空使用内置的 rwkv_vocab_v20230424.txt。
    vocab_path: str | None = None
    # tokenize 进程数；Windows 上建议 1-4。
    num_proc: int = 4
    num_workers: int = 2
    pin_memory: bool = True
    # 对话模板：rwkv_world | chatml | custom（custom 使用下面的前缀覆盖字段）
    template: str = "rwkv_world"
    # 覆盖模板里的角色前缀（留空则用模板默认值）。custom 模板至少需要 user/assistant 前缀。
    system_prefix: str | None = None
    user_prefix: str | None = None
    assistant_prefix: str | None = None
    turn_separator: str | None = None
    # assistant | last_assistant
    train_on: str = "assistant"
    # 每个对话末尾/每轮 assistant 之后追加的 token。留空自动使用词表里的 b"\x00"。
    append_eos: str = "end"
    eos_token_id: int | None = None
    # head | tail | random（仅 sample 模式）
    truncation: str = "head"
    # 丢弃没有任何可训练 token 的样本。
    drop_empty: bool = True


@dataclass
class DeepSpeedConfig:
    """DeepSpeed 参数；不装 deepspeed 时整段忽略。"""

    zero_stage: int = 2
    offload_optimizer: bool = False
    offload_param: bool = False
    bucket_size: float = 2.0e8
    contiguous_gradients: bool = True
    overlap_comm: bool = True


@dataclass
class DistConfig:
    """分布式启动方式。"""

    # auto | single | ddp | deepspeed
    mode: str = "auto"
    # auto 或 GPU 数量
    num_gpus: int | str = "auto"
    # 仅在使用非 standalone 启动时生效。
    master_port: int = 29500
    find_unused_parameters: bool = False
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)


@dataclass
class LogConfig:
    jsonl: bool = True
    wandb: bool = False
    wandb_project: str = "rwkv-easy"
    wandb_run: str | None = None
    level: str = "info"


@dataclass
class Config:
    """顶层配置。"""

    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    state: StateConfig = field(default_factory=StateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dist: DistConfig = field(default_factory=DistConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # ------------------------------------------------------------------ #
    # 读写
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        if not isinstance(data, dict):
            raise ConfigError(f"配置文件顶层必须是映射（YAML dict），收到 {type(data).__name__}。")
        return _build_dataclass(cls, data, where="config")

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> Config:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"找不到配置文件：{path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - 取决于用户输入
            raise ConfigError(f"解析 YAML 失败：{path}\n{exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"配置文件顶层必须是映射：{path}")
        raw = dict(raw)
        for item in overrides or []:
            _apply_override(raw, item)
        cfg = cls.from_dict(raw)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save_yaml(self, path: str | Path, header: str | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False, default_flow_style=False)
        if header:
            text = header.rstrip() + "\n\n" + text
        path.write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 校验与派生
    # ------------------------------------------------------------------ #

    def resolve(self) -> Config:
        """把依赖关系收敛成最终值（会原地修改少量字段并返回 self）。"""
        # fla 的 chunk 核接收 log-decay（走 `log_w`），clampw / softplus / exp
        # 三种口径都满足 `log_decay <= 0`，不需要强制成 softplus。
        self.train.precision = normalize_precision(self.train.precision)
        return self

    def validate(self) -> None:
        """一次性收集所有错误，避免用户改一个跑一次。"""
        errors: list[str] = []
        warns: list[str] = []

        # -- model ------------------------------------------------------
        if self.model.path is not None and self.model.path != "":
            if not Path(self.model.path).exists():
                errors.append(f"model.path 不存在：{self.model.path}")
        else:
            missing = [k for k in ("n_layer", "n_embd", "vocab_size") if getattr(self.model, k) is None]
            if missing:
                errors.append(
                    "没有配置 model.path 时，必须手动给出模型维度："
                    + ", ".join(f"model.{k}" for k in missing)
                )
        for key in ("n_layer", "n_embd", "vocab_size", "head_size"):
            value = getattr(self.model, key)
            if value is not None and int(value) <= 0:
                errors.append(f"model.{key} 必须为正整数，收到 {value}")
        if self.model.wkv_chunk < 8 or self.model.wkv_chunk > 256:
            errors.append("model.wkv_chunk 建议取自 8~256（默认 64）。")
        if self.model.decay not in ("auto", "clampw", "softplus", "exp"):
            errors.append(
                f"model.decay 只能是 auto/clampw/softplus/exp，收到 {self.model.decay!r}"
            )
        if not self.model.backend:
            errors.append("model.backend 不能为空（默认 native）")
        if self.model.dtype is not None and self.model.dtype not in ("fp32", "fp16", "bf16"):
            errors.append(f"model.dtype 只能是 fp32/fp16/bf16，收到 {self.model.dtype!r}")

        # -- train ------------------------------------------------------
        if self.train.method not in ("lora", "state"):
            errors.append(f"train.method 只能是 lora/state，收到 {self.train.method!r}")
        if self.train.ctx_len <= 0:
            errors.append("train.ctx_len 必须 > 0")
        if self.train.chunk_ctx <= 0:
            errors.append("train.chunk_ctx 必须 > 0")
        if self.train.infctx:
            if self.train.chunk_ctx > self.train.ctx_len:
                errors.append(
                    f"infctx 需要 chunk_ctx <= ctx_len，当前 {self.train.chunk_ctx} > {self.train.ctx_len}"
                )
            if self.train.chunk_ctx % 16 != 0:
                warns.append(
                    f"chunk_ctx={self.train.chunk_ctx} 不是 16 的倍数；纯 PyTorch 后端可以运行，"
                    "但未来接入 CUDA/Triton 内核时建议用 16 的倍数。"
                )
            if self.train.infctx_parallel_backward and not self.train.infctx_detach:
                errors.append(
                    "`train.infctx_parallel_backward: true` 需要 "
                    "`train.infctx_detach: true`：跨块反传无法在每块结束后立即释放激活。"
                )
            if self.train.infctx_parallel_backward and self.dist.mode == "deepspeed":
                errors.append(
                    "DeepSpeed 暂不支持多卡 infctx 逐块反传；"
                    "请设 `train.infctx_parallel_backward: false`，或改用 `dist.mode: ddp`。"
                )
            if (
                self.train.infctx_parallel_backward
                and self.train.method == "state"
                and self.dist.mode == "ddp"
            ):
                warns.append(
                    "State Tuning + infctx 多卡仍走最后统一反传（time_state 只在第一块参与计算）；"
                    "`train.infctx_parallel_backward` 只对 LoRA 生效。"
                )
            if (
                self.train.precision == "fp16"
                and self.dist.mode == "ddp"
                and not self.train.infctx_parallel_backward
            ):
                warns.append(
                    "DDP 延迟 infctx 暂不支持 fp16 GradScaler；请改用 bf16，"
                    "或开启 train.infctx_parallel_backward。"
                )
        if self.train.micro_batch_size <= 0:
            errors.append("train.micro_batch_size 必须 >= 1")
        if self.train.grad_accum <= 0:
            errors.append("train.grad_accum 必须 >= 1")
        if self.train.epochs <= 0 and self.train.max_steps is None:
            errors.append("train.epochs 与 train.max_steps 不能同时为空")
        if self.train.max_steps is not None and self.train.max_steps <= 0:
            errors.append("train.max_steps 必须 > 0")
        if self.train.lr <= 0:
            errors.append("train.lr 必须 > 0")
        if self.train.lr_schedule not in ("cosine", "linear", "constant"):
            errors.append(
                f"train.lr_schedule 只能是 cosine/linear/constant，收到 {self.train.lr_schedule!r}"
            )
        if self.train.compile_mode not in ("default", "reduce-overhead", "max-autotune"):
            errors.append(
                "train.compile_mode 只能是 default/reduce-overhead/max-autotune，"
                f"收到 {self.train.compile_mode!r}"
            )
        try:
            normalize_precision(self.train.precision)
        except ValueError as exc:
            errors.append(str(exc))
        if self.train.precision == "fp16" and self.train.grad_accum > 1:
            warns.append("fp16 + 梯度累积对 GradScaler 不友好，建议改用 bf16。")

        # -- lora -------------------------------------------------------
        if self.train.method == "lora":
            if self.lora.rank <= 0:
                errors.append("lora.rank 必须 > 0")
            if self.lora.alpha <= 0:
                errors.append("lora.alpha 必须 > 0")
            if not (0.0 <= self.lora.dropout < 1.0):
                errors.append("lora.dropout 必须在 [0, 1) 内")
            if not self.lora.target_modules and not self.lora.target_regex:
                errors.append("lora.target_modules 与 lora.target_regex 不能同时为空")
            if self.lora.target_layers is not None:
                if not isinstance(self.lora.target_layers, list) or not self.lora.target_layers:
                    errors.append("lora.target_layers 必须是非空列表或 null")
                else:
                    for value in self.lora.target_layers:
                        if not isinstance(value, int) or value < 0:
                            errors.append(
                                f"lora.target_layers 只能包含非负整数，收到 {value!r}"
                            )

        # -- data -------------------------------------------------------
        if not self.data.path:
            errors.append("必须配置 data.path（ShareGPT JSON/JSONL 数据集）")
        elif not Path(self.data.path).exists():
            errors.append(f"data.path 不存在：{self.data.path}")
        if self.data.mode not in ("packed", "sample"):
            errors.append(f"data.mode 只能是 packed/sample，收到 {self.data.mode!r}")
        if self.data.mode == "sample" and self.train.infctx:
            errors.append(
                "`data.mode: sample` 与 `train.infctx: true` 不兼容：infctx 需要定长无 padding 的样本。\n"
                "请改用 `data.mode: packed`，或关闭 infctx。"
            )
        if self.data.format not in ("auto", "sharegpt", "messages", "text"):
            errors.append(
                f"data.format 只能是 auto/sharegpt/messages/text，收到 {self.data.format!r}"
            )
        if self.data.template not in ("rwkv_world", "chatml", "custom"):
            errors.append(
                f"data.template 只能是 rwkv_world/chatml/custom，收到 {self.data.template!r}"
            )
        if self.data.template == "custom" and (
            not self.data.user_prefix or not self.data.assistant_prefix
        ):
            errors.append(
                "data.template=custom 时必须提供 data.user_prefix 与 data.assistant_prefix"
            )
        if not (0.0 <= self.data.val_ratio < 1.0):
            errors.append("data.val_ratio 必须在 [0, 1) 内")
        if self.data.train_on not in ("assistant", "last_assistant"):
            errors.append(
                f"data.train_on 只能是 assistant/last_assistant，收到 {self.data.train_on!r}"
            )
        if self.data.append_eos not in ("never", "end", "each_assistant"):
            errors.append(
                f"data.append_eos 只能是 never/end/each_assistant，收到 {self.data.append_eos!r}"
            )
        if self.data.truncation not in ("head", "tail", "random"):
            errors.append(
                f"data.truncation 只能是 head/tail/random，收到 {self.data.truncation!r}"
            )

        # -- dist -------------------------------------------------------
        if isinstance(self.dist.num_gpus, str) and (
            self.dist.num_gpus != "auto" and not self.dist.num_gpus.isdigit()
        ):
            errors.append("dist.num_gpus 只能是 auto 或正整数")
        if self.dist.mode not in ("auto", "single", "ddp", "deepspeed"):
            errors.append(f"dist.mode 只能是 auto/single/ddp/deepspeed，收到 {self.dist.mode!r}")
        if self.dist.deepspeed.zero_stage not in (0, 1, 2, 3):
            errors.append("dist.deepspeed.zero_stage 只能是 0/1/2/3")
        if self.dist.deepspeed.zero_stage == 3 and self.train.method in ("lora", "state"):
            warns.append(
                "ZeRO-3 会把冻结的基座参数也分片，LoRA/State 训练性价比低且容易踩坑；"
                "建议 dist.deepspeed.zero_stage: 2。"
            )

        for message in warns:
            # 用 warnings 模块输出，避免把“可接受”的问题混进错误列表。
            import warnings

            warnings.warn(f"[配置提醒] {message}", stacklevel=2)

        if errors:
            joined = "\n".join(f"  - {e}" for e in errors)
            raise ConfigError(f"配置校验失败：\n{joined}")

    # ------------------------------------------------------------------ #
    # 展示
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        method = self.train.method
        infctx = "infctx" if self.train.infctx else "full-ctx"
        if self.train.method == "lora":
            detail = f"r={self.lora.rank}, alpha={self.lora.alpha}"
        else:
            detail = f"init_std={self.state.init_std}"
        return (
            f"{method}({detail}) | {infctx} ctx={self.train.ctx_len}"
            + (f"/chunk={self.train.chunk_ctx}" if self.train.infctx else "")
            + (
                " | ddp-chunk-backward"
                if self.train.infctx and self.train.infctx_parallel_backward
                else ""
            )
            + f" | bsz={self.train.micro_batch_size}x{self.train.grad_accum}"
            f" | {self.train.precision} | data={Path(self.data.path).name}"
        )


# --------------------------------------------------------------------------- #
# YAML dict -> dataclass
# --------------------------------------------------------------------------- #


_HINTS_CACHE: dict[type, dict[str, Any]] = {}


def _resolved_hints(cls: type) -> dict[str, Any]:
    """``from __future__ import annotations`` 会让 field.type 变成字符串，这里统一解析。"""
    if cls not in _HINTS_CACHE:
        _HINTS_CACHE[cls] = get_type_hints(cls)
    return _HINTS_CACHE[cls]


def _build_dataclass(cls: type, data: Any, where: str):
    if not is_dataclass(cls):
        raise TypeError(f"{cls} 不是 dataclass")
    if not isinstance(data, dict):
        raise ConfigError(f"{where} 需要是一个映射，收到 {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    hints = _resolved_hints(cls)
    unknown = [k for k in data if k not in known]
    if unknown:
        hints_text = []
        for key in unknown:
            close = difflib.get_close_matches(key, list(known), n=1)
            hint = f"（你是不是想写 {close[0]!r}？）" if close else ""
            hints_text.append(f"{where}.{key}{hint}")
        available = ", ".join(sorted(known))
        raise ConfigError(
            "未知配置项：" + ", ".join(hints_text) + f"\n{where} 可用的字段：{available}"
        )
    kwargs = {}
    for name in known:
        if name not in data:
            continue
        kwargs[name] = _convert(data[name], hints[name], f"{where}.{name}")
    return cls(**kwargs)


def _convert(value: Any, tp: Any, where: str):
    origin = get_origin(tp)

    # Optional / Union
    if origin in (Union, types.UnionType):
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{where} 不能为空")
        errors = []
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _convert(value, candidate, where)
            except ConfigError as exc:  # 尝试下一个类型
                errors.append(str(exc))
        raise ConfigError(f"{where} 类型不匹配：收到 {value!r}（{type(value).__name__}）")

    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where} 需要列表，收到 {type(value).__name__}")
        (item_tp,) = get_args(tp) or (Any,)
        return [_convert(v, item_tp, f"{where}[{i}]") for i, v in enumerate(value)]

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where} 需要映射，收到 {type(value).__name__}")
        return dict(value)

    if is_dataclass(tp):
        return _build_dataclass(tp, value, where)

    if tp is Any or tp is object:
        return value

    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where} 需要 true/false，收到 {value!r}")

    if tp is int:
        if isinstance(value, bool):
            raise ConfigError(f"{where} 需要整数，收到布尔值")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ConfigError(f"{where} 需要整数，收到 {value!r}")

    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} 需要数字，收到 {value!r}")
        return float(value)

    if tp is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{where} 需要字符串，收到 {value!r}")

    return value


def _apply_override(raw: dict[str, Any], item: str) -> None:
    """把 ``train.lr=2e-4`` 这样的覆盖应用到原始 dict。"""
    if "=" not in item:
        raise ConfigError(f"--set 参数格式应为 key=value，收到 {item!r}")
    key, _, value_text = item.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"--set 参数缺少 key：{item!r}")
    try:
        value = yaml.safe_load(value_text)
    except yaml.YAMLError as exc:  # pragma: no cover
        raise ConfigError(f"--set {item!r} 的值无法解析：{exc}") from exc
    # YAML 1.1 会把 `2e-4` 解析成字符串（必须有小数点才算 float）。
    # 命令行里 `--set train.lr=2e-4` 太常见，这里兜底成浮点数。
    if isinstance(value, str) and re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+", value
    ):
        value = float(value)

    parts = key.split(".")
    node = raw
    for depth, part in enumerate(parts[:-1]):
        if part not in node:
            # YAML 里没写这一段（比如用户没写 dist.deepspeed）时自动补空 dict，
            # 交给 dataclass 转换去落实默认值；拼错字段仍会报“未知配置项”。
            node[part] = {}
        if not isinstance(node[part], dict):
            raise ConfigError(f"--set {key!r}：{'.'.join(parts[: depth + 1])} 不是配置段")
        node = node[part]
    if parts[-1] not in node:
        # 允许新增叶子：交给后续 dataclass 转换去报“未知配置项”。
        node[parts[-1]] = value
    else:
        node[parts[-1]] = value


# --------------------------------------------------------------------------- #
# 配置模板（给 `rwkv-easy init` 用）
# --------------------------------------------------------------------------- #

_TEMPLATE_HEADER = """\
# ============================================================
# RWKV-EASY 训练配置（YAML）
#
# 最常用的三个地方：
#   1. model.path   改成你的 RWKV-7 权重
#   2. data.path    改成你的 ShareGPT 数据集
#   3. train.method lora 或 state
#
# 其它参数保持默认即可。更多说明见 docs/CONFIG.md。
# GPU 建议：precision: bf16-full、compile: true；装了 rwkv-fla 可把
# model.backend 改成 fla。CPU / 调试必须用 precision: fp32。
# 命令行临时覆盖：rwkv-easy train -c 本文件 --set train.lr=2e-4
# ============================================================
"""


def template_config(
    method: str = "lora",
    infctx: bool = False,
    model_path: str | None = None,
    data_path: str | None = None,
    name: str | None = None,
) -> str:
    """生成带中文注释的配置模板。"""
    if method not in ("lora", "state"):
        raise ConfigError(f"method 只能是 lora/state，收到 {method!r}")
    model_value = model_path or "models/rwkv7-g1d-0.4b.pth"
    data_value = data_path or "data/sharegpt.json"
    run_name = name or ("my-state-run" if method == "state" else "my-lora-run")
    lines = [
        _TEMPLATE_HEADER.rstrip(),
        "",
        "run:",
        f"  name: {run_name}",
        "  output_dir: runs",
        "  seed: 42",
        "",
        "model:",
        f"  path: {model_value}     # ← 你的 RWKV-7 权重",
        "  decay: auto                        # auto 会自动判断衰减公式，推荐保持",
        "  backend: native                    # native 通用；装了 rwkv-fla 可改 fla",
        "  wkv_chunk: 64",
        "",
        "train:",
        f"  method: {method}",
        f"  infctx: {'true' if infctx else 'false'}",
        "  ctx_len: 2048",
        "  chunk_ctx: 2048                    # 仅 infctx 使用；显存 ∝ chunk_ctx",
        "  micro_batch_size: 2",
        "  grad_accum: 8",
        "  epochs: 1",
        "  lr: 1.0e-4",
        "  lr_schedule: cosine",
        "  warmup_steps: 20",
        "  precision: bf16-full               # GPU 推荐；CPU 必须改 fp32",
        "  grad_checkpoint: true              # infctx + per-chunk backward 后可关",
        "  infctx_backward: true              # 单卡：每块立即反传，显存 ∝ chunk_ctx",
        (
            "  infctx_parallel_backward: true     # DDP：每块反传，最后一块 all-reduce"
            if method == "lora"
            else "  infctx_parallel_backward: false    # State + 多卡仍走最后统一反传"
        ),
        "  compile: false                     # CUDA 上建议 true；首次启动会编译",
        "  merge_at_end: true                 # 训练完自动导出完整权重",
        "",
    ]
    if method == "lora":
        lines += [
            "lora:",
            "  rank: 16",
            "  alpha: 32",
            "  dropout: 0.0",
            "  target_modules:",
            "    - att.receptance",
            "    - att.key",
            "    - att.value",
            "    - att.output",
            "    - ffn.key",
            "    - ffn.value",
            "  target_layers: null                # null = 所有 block；或 [0,1,2]",
            "",
        ]
    else:
        lines += [
            "state:",
            "  init_std: 0.0                      # 0 = 零初始化（官方做法）",
            "",
        ]
    lines += [
        "data:",
        f"  path: {data_value}             # ← 你的 ShareGPT 数据集",
        "  mode: packed                       # packed 兼容 infctx；sample 仅非 infctx",
        "  template: rwkv_world",
        "  train_on: assistant",
        "  append_eos: end                    # 每个对话末尾加 EOT，避免模型不会停",
        "  val_ratio: 0.0",
        "  num_workers: 2",
        "",
        "dist:",
        "  mode: auto                         # auto: 多卡自动 DDP",
        "  num_gpus: auto",
        "  deepspeed:",
        "    zero_stage: 2                    # 装了 deepspeed 且 dist.mode=deepspeed 时生效",
        "",
        "log:",
        "  jsonl: true",
        "  wandb: false",
        "",
    ]
    return "\n".join(lines)
