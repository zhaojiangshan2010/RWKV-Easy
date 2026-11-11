#!/usr/bin/env bash
# ============================================================
# RWKV-EASY 一键安装 + 交互式训练向导（Linux / macOS）
#
#   bash quickstart.sh
#
# 参数会原样传给 quickstart.py，例如：
#   bash quickstart.sh --cpu
#   bash quickstart.sh --skip-install --no-train
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "[错误] 没有找到 Python，请先安装 Python 3.10+（https://www.python.org/downloads/）" >&2
  exit 1
fi

exec "$PYTHON" quickstart.py "$@"
