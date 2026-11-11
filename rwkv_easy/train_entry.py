"""torchrun / DeepSpeed 启动时的训练入口。

单独拆出来是为了让 ``python -m torch.distributed.run -m rwkv_easy.train_entry``
拥有一个干净的 ``sys.argv``（不包含 ``train`` 子命令）。
"""

from __future__ import annotations

import argparse
import sys

from .config import Config, ConfigError
from .runtime import setup_logger
from .train.runner import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RWKV-EASY 训练入口（通常由 rwkv-easy train 自动调用）")
    parser.add_argument("-c", "--config", required=True, help="YAML 配置文件")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="覆盖配置项")
    parser.add_argument("--resume", default=None, help="从 checkpoint 或 run 目录续训")
    parser.add_argument("--run-dir", default=None, help="指定输出目录")
    parser.add_argument("--device", default=None, help="覆盖 train.device")
    parser.add_argument("--backend", default=None, help="覆盖 model.backend（native/reference/fla）")
    parser.add_argument("--random-init", action="store_true", help="随机初始化（自检用）")
    parser.add_argument("--log-level", default=None, help="日志级别")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = Config.load(args.config, args.set)
    except ConfigError as exc:
        print(f"配置错误：\n{exc}", file=sys.stderr)
        return 2

    if args.device:
        cfg.train.device = args.device
    if args.backend:
        cfg.model.backend = args.backend

    result = run_training(
        cfg,
        resume=args.resume,
        random_init=args.random_init,
        run_dir=args.run_dir,
        log_level=args.log_level,
    )
    log = setup_logger(rank=0, level=args.log_level or cfg.log.level)
    log.info("输出目录：%s", result.run_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
