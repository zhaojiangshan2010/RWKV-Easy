# ============================================================
# RWKV-EASY 一键安装（Windows PowerShell）
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# 可选：
#   $env:TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
#   $env:SKIP_SELFTEST = "1"
# ============================================================

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Say($msg)  { Write-Host "[RWKV-EASY] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[提示] $msg" -ForegroundColor Yellow }
function Die($msg)  { Write-Host "[错误] $msg" -ForegroundColor Red; exit 1 }

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { Die "没有找到 Python，请先安装 Python 3.10+（安装时勾选 Add Python to PATH）" }

& python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if ($LASTEXITCODE -ne 0) { Die "需要 Python 3.10 或更高版本" }

if (-not (Test-Path ".venv")) {
    Say "创建虚拟环境 .venv ..."
    & python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1

Say "升级 pip ..."
& python -m pip install --upgrade pip wheel | Out-Null

if ($env:TORCH_INDEX) {
    Say "安装 PyTorch（源：$env:TORCH_INDEX）..."
    & python -m pip install torch --index-url $env:TORCH_INDEX
} elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Say "检测到 NVIDIA GPU，安装 CUDA 版 PyTorch ..."
    & python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) {
        Warn "cu128 安装失败，尝试默认源 ..."
        & python -m pip install torch
    }
} else {
    Warn "未检测到 NVIDIA GPU，安装 CPU 版 PyTorch。"
    & python -m pip install torch
}
if ($LASTEXITCODE -ne 0) { Die "PyTorch 安装失败" }

Say "安装 RWKV-EASY ..."
& python -m pip install -e .
if ($LASTEXITCODE -ne 0) { Die "安装失败" }

Say "环境检查："
& python -m rwkv_easy doctor

if ($env:SKIP_SELFTEST -ne "1") {
    Say "运行 CPU 自检 ..."
    & python -m rwkv_easy selftest --steps 8
}

Write-Host ""
Say "安装完成。下一步："
Write-Host "  懒人一键（安装 + 配置 + 训练）： python quickstart.py"
Write-Host "  或者手动："
Write-Host "  1) 把 RWKV-7 权重放到 models\，把 ShareGPT 数据放到 data\"
Write-Host "  2) python -m rwkv_easy init --model models\你的模型.pth --data data\sharegpt.json"
Write-Host "  3) python -m rwkv_easy train -c configs\my-lora-run.yaml"
Write-Host ""
Write-Host "  之后每次运行前先执行：.venv\Scripts\Activate.ps1"
