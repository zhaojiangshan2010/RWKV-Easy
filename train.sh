#!/usr/bin/env bash
# 一键训练：bash train.sh [配置文件] [额外参数...]
#
#   bash train.sh                                  使用 configs/lora_sharegpt.yaml
#   bash train.sh configs/state_sharegpt.yaml
#   bash train.sh configs/lora_sharegpt.yaml --set train.lr=2e-4
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-configs/lora_sharegpt.yaml}"
shift || true

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

exec python -m rwkv_easy train -c "$CONFIG" "$@"
