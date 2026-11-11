#!/usr/bin/env bash
# ============================================================
# RWKV-EASY 一键安装（Linux / macOS）
#
#   bash install.sh
#
# 可选环境变量：
#   PYTHON=python3.11                 指定 Python 解释器
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124
#                                     指定 PyTorch 安装源（驱动较旧时使用）
#   SKIP_SELFTEST=1                   安装后不跑自检
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'
say()  { printf "${GREEN}[RWKV-EASY]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[提示]${NC} %s\n" "$*"; }
die()  { printf "${RED}[错误]${NC} %s\n" "$*" >&2; exit 1; }

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON=python3
  elif command -v python >/dev/null 2>&1; then PYTHON=python
  else die "没有找到 Python，请先安装 Python 3.10+（https://www.python.org/downloads/）"; fi
fi

say "使用解释器：$($PYTHON -c 'import sys; print(sys.executable)')"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || die "需要 Python 3.10 或更高版本，当前是 $($PYTHON -V 2>&1)"

if [ ! -d .venv ]; then
  say "创建虚拟环境 .venv ..."
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

say "升级 pip ..."
python -m pip install --upgrade pip wheel >/dev/null

if [ -n "${TORCH_INDEX:-}" ]; then
  say "安装 PyTorch（源：$TORCH_INDEX）..."
  python -m pip install torch --index-url "$TORCH_INDEX"
elif command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
  say "检测到 NVIDIA 驱动：$DRIVER"
  say "安装 PyTorch（Linux 默认轮子已包含 CUDA 运行时）..."
  python -m pip install torch
else
  warn "未检测到 NVIDIA GPU，安装 CPU 版 PyTorch（之后仍可换成 CUDA 版）。"
  python -m pip install torch
fi

say "安装 RWKV-EASY ..."
python -m pip install -e .

say "环境检查："
python -m rwkv_easy doctor

if [ "${SKIP_SELFTEST:-0}" != "1" ]; then
  say "运行 CPU 自检（不需要下载任何模型）..."
  python -m rwkv_easy selftest --steps 8 || warn "自检失败，请把上面的输出发给开发者。"
fi

echo
say "安装完成。下一步："
echo "  懒人一键（安装 + 配置 + 训练）："
echo "       python quickstart.py"
echo
echo "  或者手动："
echo "  1) 准备模型与数据"
echo "       mkdir -p models data"
echo "       # 把 RWKV-7 的 .pth 放到 models/，把 ShareGPT 数据集放到 data/"
echo "  2) 生成配置"
echo "       python -m rwkv_easy init --model models/你的模型.pth --data data/sharegpt.json"
echo "  3) 开始训练"
echo "       python -m rwkv_easy train -c configs/my-lora-run.yaml"
echo
echo "  之后每次运行前先激活环境：source .venv/bin/activate"
