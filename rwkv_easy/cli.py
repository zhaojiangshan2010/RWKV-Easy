"""RWKV-EASY 命令行入口。

命令一览::

    rwkv-easy init       生成一份带注释的配置
    rwkv-easy wizard     交互式一键配置并训练（新手推荐）
    rwkv-easy train      开始训练
    rwkv-easy merge      把 LoRA / State 权重合并成完整模型
    rwkv-easy inspect-data  检查数据集与 loss mask
    rwkv-easy doctor     检查环境
    rwkv-easy selftest   不下载任何权重，跑一遍完整流程（CPU 可跑）
    rwkv-easy generate   小规模生成，用于确认训练结果（诊断用）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

from . import __version__
from .config import Config, ConfigError, template_config
from .runtime import device_description, human_size, setup_logger

__all__ = ["build_parser", "main"]

ASSETS = Path(__file__).resolve().parent / "assets"


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rwkv-easy",
        description="RWKV-EASY：RWKV-7 LoRA / State Tuning 训练器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  rwkv-easy wizard     交互式一键配置并训练（新手推荐）
  rwkv-easy init --model models/rwkv7.pth --data data/sharegpt.json
  rwkv-easy train -c configs/my_lora.yaml
  rwkv-easy merge --base models/rwkv7.pth --adapter runs/xxx/adapter-step0000100.pth --out merged.pth
  rwkv-easy check-model -m models/rwkv7.pth
  rwkv-easy doctor
""",
    )
    parser.add_argument("--version", action="version", version=f"rwkv-easy {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # -- train ----------------------------------------------------------
    train = sub.add_parser("train", help="开始训练")
    train.add_argument("-c", "--config", required=True, help="YAML 配置文件")
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖配置项，可重复")
    train.add_argument("--resume", default=None, help="从 checkpoint 或 run 目录续训")
    train.add_argument("--run-dir", default=None, help="指定输出目录")
    train.add_argument("--num-gpus", type=int, default=None, help="覆盖 dist.num_gpus")
    train.add_argument("--device", default=None, help="覆盖 train.device")
    train.add_argument("--backend", default=None, choices=["native", "reference", "fla"], help="覆盖 model.backend")
    train.add_argument("--single", action="store_true", help="强制单进程（忽略多卡）")
    train.add_argument("--random-init", action="store_true", help="随机初始化小模型（不做真实训练）")
    train.add_argument("--log-level", default=None, help="debug/info/warning")

    # -- init -----------------------------------------------------------
    init = sub.add_parser("init", help="生成配置文件")
    init.add_argument("--model", default=None, help="RWKV-7 权重路径")
    init.add_argument("--data", default=None, help="ShareGPT 数据集路径")
    init.add_argument("--method", default=None, choices=["lora", "state"], help="训练方法")
    init.add_argument("--infctx", action="store_true", help="启用 infctx 分块训练")
    init.add_argument("--name", default=None, help="实验名")
    init.add_argument("-o", "--output", default=None, help="输出 YAML 路径（默认 configs/<name>.yaml）")
    init.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    init.add_argument("--non-interactive", action="store_true", help="不提问，缺失项用默认值")

    # -- merge ----------------------------------------------------------
    merge = sub.add_parser("merge", help="合并 LoRA / State 权重")
    merge.add_argument("--base", required=True, help="基座 RWKV-7 权重")
    merge.add_argument("--adapter", required=True, help="训练输出的 adapter/state 权重")
    merge.add_argument("-o", "--out", default=None, help="输出路径（默认最后一个 checkpoint 同目录 merged.pth）")
    merge.add_argument("--dtype", default=None, choices=["fp32", "bf16", "fp16"], help="输出 dtype")
    merge.add_argument("--backend", default="native", choices=["native", "reference"], help="构建模型时的 WKV 后端")

    # -- doctor ---------------------------------------------------------
    doctor = sub.add_parser("doctor", help="检查环境")
    doctor.add_argument("--json", action="store_true", help="以 JSON 输出")

    # -- selftest -------------------------------------------------------
    selftest = sub.add_parser("selftest", help="完整流程自检（不需要真实权重）")
    selftest.add_argument("--full", action="store_true", help="跑全部组合（较慢）")
    selftest.add_argument("--steps", type=int, default=None, help="每个场景的训练步数")
    selftest.add_argument("--device", default="auto", help="auto/cpu/cuda")
    selftest.add_argument("--keep", action="store_true", help="保留临时目录")
    selftest.add_argument("--workdir", default=None, help="指定工作目录")

    # -- inspect-data ---------------------------------------------------
    inspect = sub.add_parser("inspect-data", help="检查数据集与 loss mask")
    inspect.add_argument("-c", "--config", required=True, help="YAML 配置文件（只用到 data 段）")
    inspect.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    inspect.add_argument("-n", "--num", type=int, default=3, help="展示几个样本")
    inspect.add_argument("--window", type=int, default=0, help="从第几个窗口开始展示")

    # -- generate -------------------------------------------------------
    gen = sub.add_parser("generate", help="小规模生成（诊断用）")
    gen.add_argument("-m", "--model", required=True, help="完整模型权重（合并后的 .pth）")
    gen.add_argument("-p", "--prompt", required=True, help="提示词")
    gen.add_argument("--max-tokens", type=int, default=64)
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--sample", action="store_true", help="采样（默认贪心）")
    gen.add_argument("--decay", default="auto", choices=["auto", "clampw", "softplus", "exp"])
    gen.add_argument("--template", default="rwkv_world", choices=["rwkv_world", "chatml", "none"])
    gen.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "bf16"],
        help="auto = CPU 用 fp32（快很多）、GPU 保持权重 dtype",
    )

    # -- check-model ----------------------------------------------------
    check = sub.add_parser("check-model", help="一条命令验收 RWKV-7 权重（格式/架构/decay/出字）")
    check.add_argument("-m", "--model", required=True, help="RWKV-7 权重（.pth / .safetensors）")
    check.add_argument("--max-tokens", type=int, default=32)
    check.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    check.add_argument("--device", default="auto", help="auto / cpu / cuda")
    check.add_argument("--backend", default="native", choices=["native", "reference", "fla"])
    check.add_argument("--sample", action="store_true", help="采样（默认贪心）")

    # -- wizard ---------------------------------------------------------
    wiz = sub.add_parser("wizard", help="交互式一键配置并训练（新手推荐）")
    wiz.add_argument("--non-interactive", action="store_true", help="分行问答（无 TTY / 脚本）")
    wiz.add_argument("--no-train", action="store_true", help="只生成配置，不启动训练")
    wiz.add_argument("-o", "--output", default=None, help="配置文件输出路径")

    return parser


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "train": cmd_train,
        "wizard": cmd_wizard,
        "init": cmd_init,
        "merge": cmd_merge,
        "doctor": cmd_doctor,
        "selftest": cmd_selftest,
        "inspect-data": cmd_inspect_data,
        "generate": cmd_generate,
        "check-model": cmd_check_model,
    }
    try:
        return int(commands[args.command](args) or 0)
    except ConfigError as exc:
        print(f"\n配置错误：\n{exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130
    except Exception as exc:
        import os
        import traceback

        print(f"\n出错了：{exc}\n", file=sys.stderr)
        if os.environ.get("RWKV_EASY_DEBUG"):
            traceback.print_exc()
        else:
            print("设置环境变量 RWKV_EASY_DEBUG=1 可以看到完整堆栈。", file=sys.stderr)
        return 1


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #


def cmd_train(args: argparse.Namespace) -> int:
    from .launcher import is_launched_by_runner, relaunch, resolve_num_gpus
    from .train.runner import run_training

    cfg = Config.load(args.config, args.set)
    if args.num_gpus is not None:
        cfg.dist.num_gpus = int(args.num_gpus)
    if args.device:
        cfg.train.device = args.device
    if args.backend:
        cfg.model.backend = args.backend
        cfg.resolve()
    if args.single:
        cfg.dist.mode = "single"

    nproc = resolve_num_gpus(cfg)
    if nproc > 1 and not is_launched_by_runner():
        return relaunch(
            nproc,
            config_path=args.config,
            overrides=args.set,
            resume=args.resume,
            run_dir=args.run_dir,
            device=args.device,
            backend=args.backend,
            log_level=args.log_level,
            random_init=args.random_init,
        )

    result = run_training(
        cfg,
        resume=args.resume,
        run_dir=args.run_dir,
        random_init=args.random_init,
        log_level=args.log_level,
    )
    print(f"\n训练完成。输出目录：{result.run_dir}")
    if result.merged is not None:
        print(f"完整权重：{result.merged}")
    elif result.last_adapter is not None:
        print(f"适配器/状态权重：{result.last_adapter}")
        print(
            "合并成完整模型：\n"
            f"  rwkv-easy merge --base {cfg.model.path} --adapter {result.last_adapter} "
            f"-o {result.run_dir / 'merged.pth'}"
        )
    return 0


# --------------------------------------------------------------------------- #
# wizard
# --------------------------------------------------------------------------- #


def cmd_wizard(args: argparse.Namespace) -> int:
    from .wizard import WizardCancelled, run_wizard

    try:
        return run_wizard(
            non_interactive=bool(args.non_interactive),
            no_train=bool(args.no_train),
            output=args.output,
        )
    except (WizardCancelled, KeyboardInterrupt):
        print("\n已取消。")
        return 130


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f"（默认 {default}）" if default else ""
    try:
        value = input(f"{prompt}{suffix}：").strip()
    except EOFError:
        value = ""
    if not value and default is not None:
        return default
    return value


def cmd_init(args: argparse.Namespace) -> int:
    interactive = not args.non_interactive and sys.stdin.isatty()

    model = args.model
    data = args.data
    method = args.method
    if interactive:
        print("=" * 60)
        print("RWKV-EASY 配置向导（直接回车使用默认值）")
        print("=" * 60)
        if not model:
            model = _ask("RWKV-7 权重路径（.pth / .safetensors）", "models/rwkv7-g1d-0.4b.pth")
        if not data:
            data = _ask("ShareGPT 数据集路径（.json / .jsonl）", "data/sharegpt.json")
        if not method:
            method = _ask("训练方法 lora / state", "lora")
        if not args.infctx:
            answer = _ask("是否启用 infctx 长上下文分块训练？(y/N)", "n")
            args.infctx = answer.lower().startswith("y")

    model = model or "models/rwkv7-g1d-0.4b.pth"
    data = data or "data/sharegpt.json"
    method = method or "lora"
    if method not in ("lora", "state"):
        raise ConfigError(f"--method 只能是 lora/state，收到 {method!r}")

    name = args.name or ("my-state-run" if method == "state" else "my-lora-run")
    output = Path(args.output) if args.output else Path("configs") / f"{name}.yaml"
    if output.exists() and not args.force:
        raise ConfigError(f"{output} 已存在；加 --force 覆盖，或用 -o 指定别的路径。")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        template_config(method, args.infctx, model_path=model, data_path=data, name=name),
        encoding="utf-8",
    )

    print(f"已生成配置：{output}")
    if not Path(model).exists():
        print(f"提醒：模型文件还不存在：{model}")
    if not Path(data).exists():
        print(f"提醒：数据文件还不存在：{data}")
    print("\n下一步：")
    print(f"  rwkv-easy inspect-data -c {output}     # 先检查数据与 mask")
    print(f"  rwkv-easy train -c {output}            # 开始训练")
    return 0


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #


def cmd_merge(args: argparse.Namespace) -> int:
    from .lora import inject_lora, load_lora_state_dict
    from .models.checkpoint import load_model_from_checkpoint
    from .state_tuning import attach_state_tuning, load_state_state_dict
    from .train.checkpoint import export_merged, read_adapter

    info = read_adapter(args.adapter)
    print(f"读取 {args.adapter}：{info.describe()}")

    model_cfg = info.model_config or {}
    model, inferred, _ = load_model_from_checkpoint(
        args.base,
        backend=args.backend,
        decay=info.decay if info.decay in ("clampw", "softplus", "exp") else "clampw",
        n_layer=model_cfg.get("n_layer"),
        n_embd=model_cfg.get("n_embd"),
        vocab_size=model_cfg.get("vocab_size"),
        head_size=model_cfg.get("head_size"),
        head_size_divisor=model_cfg.get("head_size_divisor", 8),
    )

    if info.method == "lora":
        cfg = info.method_config or {}
        inject_lora(
            model,
            target_modules=cfg.get("target_modules", ["att.receptance", "att.key", "att.value", "att.output", "ffn.key", "ffn.value"]),
            rank=int(cfg.get("rank", 16)),
            alpha=int(cfg.get("alpha", 32)),
            dropout=float(cfg.get("dropout", 0.0)),
            target_regex=cfg.get("target_regex"),
        )
        load_lora_state_dict(model, info.weights)
    elif info.method == "state":
        attach_state_tuning(model, init_std=0.0)
        load_state_state_dict(model, info.weights)
    else:
        raise ConfigError(f"未知训练方法：{info.method!r}")

    out = Path(args.out) if args.out else Path(args.adapter).parent / "merged.pth"
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(
        args.dtype or "fp32"
    )
    export_merged(model, info.method, out, dtype=dtype)
    size = human_size(out.stat().st_size)
    print(f"\n已导出完整权重：{out}（{size}）")
    print(f"模型结构：{inferred.n_layer} 层 / {inferred.n_embd} 维 / vocab {inferred.vocab_size}")
    return 0


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def cmd_doctor(args: argparse.Namespace) -> int:
    import platform

    from .backends.native import NativeBackend
    from .backends.reference import ReferenceBackend
    from .data.tokenizer import DEFAULT_VOCAB_PATH, get_tokenizer

    info: dict[str, object] = {}
    info["rwkv-easy"] = __version__
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    info["rocm"] = getattr(torch.version, "hip", None)
    info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    info["gpus"] = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ] if torch.cuda.is_available() else []

    optional = {}
    for name, module in (
        ("deepspeed", "deepspeed"),
        ("rwkv-fla", "rwkvfla"),
        ("safetensors", "safetensors"),
        ("wandb", "wandb"),
        ("pyarrow", "pyarrow"),
    ):
        try:
            __import__(module)
            optional[name] = True
        except Exception:
            optional[name] = False
    info["optional"] = optional

    vocab_ok = DEFAULT_VOCAB_PATH.exists()
    info["vocab_file"] = str(DEFAULT_VOCAB_PATH) if vocab_ok else None

    free = shutil.disk_usage(".").free
    info["disk_free"] = human_size(free)

    # WKV 数值自检
    torch.manual_seed(0)
    r = torch.randn(1, 9, 2, 8)
    k = torch.randn(1, 9, 2, 8)
    v = torch.randn(1, 9, 2, 8)
    a = -torch.nn.functional.normalize(torch.randn(1, 9, 2, 8), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(1, 9, 2, 8), dim=-1)
    w = -torch.rand(1, 9, 2, 8) * 0.6
    with torch.no_grad():
        out_native, _ = NativeBackend(chunk=4).wkv7(r, k, v, w, a, b)
        out_ref, _ = ReferenceBackend(double=True).wkv7(r, k, v, w, a, b)
    error = float((out_native.double() - out_ref).abs().max())
    info["wkv_selfcheck_max_error"] = error

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    print("=" * 62)
    print(f"RWKV-EASY {__version__} 环境检查")
    print("=" * 62)
    print(device_description())
    print(f"Python      : {info['python']}  ({info['platform']})")
    print(f"PyTorch     : {info['torch']}")
    print(f"CUDA/ROCm   : {info['cuda'] or info['rocm'] or '无'}")
    print(f"GPU         : {info['gpu_count']} 张" + (f" -> {', '.join(info['gpus'])}" if info["gpus"] else ""))
    print(f"词表文件    : {'OK  ' + str(DEFAULT_VOCAB_PATH) if vocab_ok else '缺失！'}")
    print(f"磁盘剩余    : {info['disk_free']}")
    print("可选依赖    : " + ", ".join(f"{k}={'✓' if v else '×'}" for k, v in optional.items()))
    print(f"WKV 自检    : 最大误差 {error:.2e} " + ("✓" if error < 1e-4 else "✗ 异常"))

    if vocab_ok:
        try:
            tok = get_tokenizer()
            print(f"分词器      : vocab={tok.vocab_size}, eot={tok.eot_id}, 最长 token={tok.max_token_len} 字节")
        except Exception as exc:  # pragma: no cover
            print(f"分词器      : 加载失败（{exc}）")

    print()
    if torch.cuda.is_available():
        print("状态：可以使用 CUDA 训练。")
    else:
        print("状态：未检测到可用 GPU。可以先用 CPU 跑 `rwkv-easy selftest` 验证流程；")
        print("      正式训练请安装 CUDA 版 PyTorch（见 README 的安装章节）。")
    if not optional["deepspeed"]:
        print("提示：未安装 DeepSpeed。多卡会使用 PyTorch DDP，单卡训练完全不受影响。")
    return 0


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #


def _make_tiny_checkpoint(path: Path, seed: int = 0) -> None:
    from .models.checkpoint import random_model
    from .models.rwkv7 import RWKV7Config

    cfg = RWKV7Config(vocab_size=65536, n_layer=2, n_embd=64, head_size=16)
    model = random_model(cfg, backend="native", decay="clampw", seed=seed)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def _read_losses(metrics_path: Path) -> list[float]:
    losses: list[float] = []
    if not metrics_path.exists():
        return losses
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = record.get("loss")
            if isinstance(value, (int, float)) and value == value:  # 排除 NaN
                losses.append(float(value))
    return losses


def cmd_selftest(args: argparse.Namespace) -> int:
    from .models.checkpoint import load_model_from_checkpoint
    from .runtime import resolve_device
    from .train.runner import run_training

    device = resolve_device(args.device)
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="rwkv-easy-selftest-"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"工作目录：{workdir}")
    print(f"设备：{device}")

    checkpoint = workdir / "tiny-model.pth"
    if not checkpoint.exists():
        print("生成随机小模型（约 9M 参数，仅用于自检）……")
        _make_tiny_checkpoint(checkpoint, seed=0)
    data_path = ASSETS / "sharegpt_tiny.json"

    steps = args.steps or (20 if args.full else 40)
    scenarios: list[tuple[str, bool, str]] = [("lora", True, "fp32"), ("state", True, "fp32")]
    if args.full:
        scenarios = [
            ("lora", False, "fp32"),
            ("lora", True, "fp32"),
            ("lora", True, "bf16"),
            ("state", False, "fp32"),
            ("state", True, "fp32"),
            ("state", True, "bf16"),
            ("lora", False, "bf16"),
            ("state", False, "bf16"),
        ]

    failures: list[str] = []
    for method, infctx, precision in scenarios:
        label = f"{method}{'-infctx' if infctx else ''}-{precision}"
        print("\n" + "-" * 62)
        print(f"[{label}] {steps} 步")
        run_dir = workdir / label
        cfg = Config.from_dict(
            {
                "run": {"name": "selftest", "output_dir": str(workdir), "seed": 42},
                "model": {"path": str(checkpoint)},
                "train": {
                    "method": method,
                    "infctx": infctx,
                    "ctx_len": 64,
                    "chunk_ctx": 32,
                    "micro_batch_size": 2,
                    "grad_accum": 1,
                    "epochs": 100,
                    "max_steps": steps,
                    "lr": 2.0e-3 if method == "lora" else 5.0e-2,
                    "min_lr": 1.0e-4,
                    "warmup_steps": 2,
                    "precision": precision,
                    "grad_checkpoint": True,
                    "log_every": max(1, steps // 4),
                    "save_every_epochs": 0,
                    "merge_at_end": True,
                    "val_every_steps": 0,
                },
                "lora": {"rank": 8, "alpha": 16, "dropout": 0.0},
                "state": {"init_std": 0.0},
                "data": {
                    "path": str(data_path),
                    "mode": "packed",
                    "num_workers": 0,
                    "num_proc": 0,
                    "cache": False,
                    "val_ratio": 0.0,
                },
                "dist": {"mode": "single"},
                "log": {"jsonl": True},
            }
        )
        cfg.validate()
        try:
            result = run_training(cfg, run_dir=run_dir)
        except Exception as exc:
            failures.append(f"{label}: 训练失败（{exc}）")
            import traceback

            traceback.print_exc()
            continue

        losses = _read_losses(run_dir / "metrics.jsonl")
        if len(losses) < 2:
            failures.append(f"{label}: 没有记录到 loss")
        else:
            first = sum(losses[:3]) / min(3, len(losses))
            last = sum(losses[-3:]) / min(3, len(losses))
            print(f"loss: {first:.4f} -> {last:.4f}")
            if method == "lora" and not last < first - 0.02:
                failures.append(f"{label}: loss 没有下降（{first:.4f} -> {last:.4f}）")

        if result.merged is None or not result.merged.exists():
            failures.append(f"{label}: 没有导出 merged.pth")
        else:
            try:
                model, _, _ = load_model_from_checkpoint(result.merged, decay="clampw")
                keys = model.state_dict().keys()
                if any("lora_" in key for key in keys):
                    failures.append(f"{label}: 合并后的权重里仍有 lora 参数")
            except Exception as exc:
                failures.append(f"{label}: 合并后的权重无法加载（{exc}）")

        if method == "state":
            from .train.checkpoint import read_adapter

            adapters = sorted(run_dir.glob("state-step*.pth"))
            if not adapters:
                failures.append(f"{label}: 没有 state checkpoint")
            else:
                info = read_adapter(adapters[-1])
                nonzero = any(float(tensor.abs().sum()) > 0 for tensor in info.weights.values())
                if not nonzero:
                    failures.append(f"{label}: time_state 训练后仍然是全零（梯度没有传进去）")

    print("\n" + "=" * 62)
    if failures:
        print("自检失败：")
        for item in failures:
            print("  - " + item)
        return 1
    print("自检通过：训练、infctx、保存、合并、重新加载全部正常。")
    if args.keep or args.workdir:
        print(f"临时文件保留在：{workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


# --------------------------------------------------------------------------- #
# inspect-data
# --------------------------------------------------------------------------- #


def _highlight(ids: list[int], mask: list[int], tokenizer) -> str:
    """把 token 序列渲染成可读文本，并用【】标出参与 loss 的部分。

    中文常用 2~3 个 token 表示一个字，逐 token 解码会出现乱码，所以这里用
    增量解码器把跨 token 的 UTF-8 字节正确拼起来。
    """
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pieces: list[str] = []
    for token_id, flag in zip(ids, mask, strict=True):
        text = decoder.decode(tokenizer.token_text(token_id), final=False)
        if flag:
            pieces.append(f"【{text}】")
        else:
            pieces.append(text)
    pieces.append(decoder.decode(b"", final=True))
    return "".join(pieces).replace("\x00", "<EOT>")


def cmd_inspect_data(args: argparse.Namespace) -> int:
    from .data.dataset import PackedDataset, build_datasets

    cfg = Config.load(args.config, args.set)
    log = setup_logger(level="warning")
    built = build_datasets(cfg, log)
    stats = built.stats

    print("=" * 62)
    print("数据集统计")
    print("=" * 62)
    print(f"文档数        : {stats.documents}")
    print(f"token 总数    : {stats.tokens:,}")
    print(f"可训练 tokens : {stats.supervised_tokens:,} "
          f"({100.0 * stats.supervised_tokens / max(1, stats.tokens):.1f}%)")
    print(f"窗口数        : {stats.windows}")
    print(f"模式          : {cfg.data.mode} | 模板 {cfg.data.template} | ctx {cfg.train.ctx_len}")
    print("\n【】里的内容会参与 loss；其余只作为上下文。\n")

    dataset = built.train
    if isinstance(dataset, PackedDataset):
        for offset in range(args.num):
            index = args.window + offset
            if index >= len(dataset):
                break
            item = dataset[index]
            ids = item["input_ids"].tolist()
            labels = item["labels"].tolist()
            # mask[t] = labels[t-1] != -100
            mask = [0] + [1 if labels[i] != -100 else 0 for i in range(len(labels) - 1)]
            print(f"--- 窗口 {index} ---")
            print(_highlight(ids, mask, built.tokenizer))
            print()
            text = built.tokenizer.decode(ids)
            if len(text) < 40:
                print("（该窗口大部分是 padding/EOT，属于正常现象）\n")
    else:  # sample 模式
        for index in range(min(args.num, len(dataset))):
            item = dataset[index]
            ids = item["ids"].tolist()
            mask = item["mask"].tolist()
            print(f"--- 样本 {index}（{len(ids)} tokens）---")
            print(_highlight(ids, mask, built.tokenizer))
            print()
    return 0


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def _generate_text(
    model,
    tokenizer,
    prompt: str,
    device,
    *,
    max_tokens: int,
    sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    """贪心 / 采样生成（诊断用）。prompt 已经由调用方套好模板。"""
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    generated: list[int] = []
    with torch.no_grad():
        logits, state = model(input_ids, need_state=True)
        for _ in range(max_tokens):
            last = logits[:, -1, :].float()
            if sample:
                probs = torch.softmax(last / max(1e-5, temperature), dim=-1)
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cumulative = sorted_probs.cumsum(dim=-1)
                cutoff = cumulative - sorted_probs > top_p
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                choice = torch.multinomial(sorted_probs, 1)
                next_id = int(sorted_idx.gather(-1, choice).item())
            else:
                next_id = int(last.argmax(dim=-1).item())
            generated.append(next_id)
            if next_id == tokenizer.eot_id:
                break
            step_ids = torch.tensor([[next_id]], device=device)
            logits, state = model(step_ids, state=state, need_state=True)
    return tokenizer.decode(generated)


def cmd_generate(args: argparse.Namespace) -> int:
    from .data.template import build_template
    from .data.tokenizer import get_tokenizer
    from .models.checkpoint import load_model_from_checkpoint
    from .models.decay import calibrate_decay
    from .runtime import resolve_device

    device = resolve_device("auto")
    tokenizer = get_tokenizer()
    # CPU 上 bf16 的前向/反传都远慢于 fp32，所以 auto 在 CPU 上强制 fp32。
    if args.dtype == "fp32":
        load_dtype: torch.dtype | None = torch.float32
    elif args.dtype == "bf16":
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float32 if device.type == "cpu" else None
    model, model_cfg, signature = load_model_from_checkpoint(
        args.model,
        backend="native",
        decay="clampw" if args.decay == "auto" else args.decay,
        dtype=load_dtype,
    )
    model.to(device)
    model.eval()
    if args.decay == "auto":
        calibration = calibrate_decay(model, tokenizer, device, cache_key=signature, cache_dir=None)
        model.set_decay_mode(calibration.mode)
    print(f"模型：{model_cfg.n_layer} 层 / {model_cfg.n_embd} 维，decay={model.decay_mode}")

    prompt = args.prompt
    if args.template != "none":
        template = build_template(args.template)
        prompt = template.generation_prompt(args.prompt)

    completion = _generate_text(
        model,
        tokenizer,
        prompt,
        device,
        max_tokens=args.max_tokens,
        sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print("-" * 62)
    print(prompt + completion)
    print("-" * 62)
    return 0


# --------------------------------------------------------------------------- #
# check-model
# --------------------------------------------------------------------------- #


def cmd_check_model(args: argparse.Namespace) -> int:
    """一条命令完成新权重验收：格式 → 架构 → decay → 出字。"""
    import math

    from .data.tokenizer import get_tokenizer
    from .models.checkpoint import infer_config, load_model_from_checkpoint, read_checkpoint
    from .models.decay import calibrate_decay
    from .runtime import resolve_device

    path = Path(args.model)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在：{path}")

    print("=" * 62)
    print("RWKV-EASY 模型验收")
    print("=" * 62)
    print(f"文件    : {path}（{human_size(path.stat().st_size)}）")

    # ---- 1. 格式 ----
    state = read_checkpoint(path)
    cfg = infer_config(state)
    print(f"张量数  : {len(state)}")
    print(
        f"结构    : {cfg.n_layer} 层 / {cfg.n_embd} 维 / {cfg.n_head} 头 / "
        f"head_size={cfg.head_size} / vocab={cfg.vocab_size}"
    )
    print(f"emb     : {tuple(state['emb.weight'].shape)}   head: {tuple(state['head.weight'].shape)}")
    if "blocks.0.att.r_k" in state:
        print(f"r_k     : {tuple(state['blocks.0.att.r_k'].shape)}")

    tokenizer = get_tokenizer()
    if cfg.vocab_size < tokenizer.vocab_size:
        print(
            f"提醒    : 模型 vocab={cfg.vocab_size} 小于词表的 {tokenizer.vocab_size}，"
            "超出范围的 token 会越界；请确认词表匹配。"
        )

    # ---- 2. 加载 ----
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = resolve_device(args.device)
    model, model_cfg, signature = load_model_from_checkpoint(
        path, backend=args.backend, decay="clampw", dtype=dtype
    )
    model.to(device)
    model.eval()
    print(f"参数    : {model.num_parameters() / 1e6:.2f}M（{args.dtype}，{device}）")

    # ---- 3. decay 校准 ----
    calibration = calibrate_decay(
        model, tokenizer, device, cache_key=signature, cache_dir=None, log=None
    )
    baseline = math.log(max(2, cfg.vocab_size))
    print()
    print("decay 校准（256 token 的语言建模 loss，越低越匹配）")
    print(f"  均匀分布基线 : {baseline:.3f}")
    for mode, score in sorted(calibration.scores.items(), key=lambda item: item[1]):
        marker = "  <- 选中" if mode == calibration.mode else ""
        print(f"  {mode:<9}: {score:.4f}{marker}")
    model.set_decay_mode(calibration.mode)

    if calibration.scores:
        best = min(calibration.scores.values())
        spread = max(calibration.scores.values()) - best
        if best > baseline * 0.9:
            verdict = "可疑：loss 接近均匀分布，权重可能损坏、未训练好，或不是 x070"
        elif best > baseline * 0.7:
            verdict = "偏弱：能加载，但输出质量可能一般（多见于很小的模型或很短的校准文本）"
        else:
            verdict = "正常：loss 明显低于均匀分布"
        print(f"  结论        : {verdict}（候选差距 {spread:.4f}）")

    # ---- 4. 出字 ----
    prompt = "The quick brown fox"
    completion = _generate_text(
        model,
        tokenizer,
        prompt,
        device,
        max_tokens=args.max_tokens,
        sample=args.sample,
        temperature=0.8,
        top_p=0.9,
    )
    print()
    print(f"续写（贪心，{args.max_tokens} token）")
    print("-" * 62)
    print(prompt + completion.replace("\x00", "<EOT>"))
    print("-" * 62)
    print()
    print("下一步：")
    print("  rwkv-easy generate -m ... -p '你好' --template rwkv_world   # 对话格式再测一次")
    print("  rwkv-easy train -c configs/lora_sharegpt.yaml --set model.path=... --set train.max_steps=20")
    return 0
