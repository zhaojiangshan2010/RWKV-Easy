@echo off
REM ============================================================
REM RWKV-EASY 一键安装（Windows）
REM
REM   install.bat
REM
REM 可选环境变量：
REM   TORCH_INDEX   指定 PyTorch 安装源，例如
REM                 set TORCH_INDEX=https://download.pytorch.org/whl/cu124
REM   SKIP_SELFTEST=1  安装后不跑自检
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 没有找到 Python，请先从 https://www.python.org/downloads/ 安装 Python 3.10+
  echo        安装时请勾选 "Add Python to PATH"
  exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
  echo [错误] 需要 Python 3.10 或更高版本
  python --version
  exit /b 1
)

if not exist .venv (
  echo [RWKV-EASY] 创建虚拟环境 .venv ...
  python -m venv .venv
)
call .venv\Scripts\activate.bat

echo [RWKV-EASY] 升级 pip ...
python -m pip install --upgrade pip wheel >nul

if defined TORCH_INDEX (
  echo [RWKV-EASY] 安装 PyTorch（源：%TORCH_INDEX%）...
  python -m pip install torch --index-url %TORCH_INDEX%
  if errorlevel 1 exit /b 1
  goto torch_done
)

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo [提示] 未检测到 NVIDIA GPU，安装 CPU 版 PyTorch。
  python -m pip install torch
) else (
  echo [RWKV-EASY] 检测到 NVIDIA GPU，安装 CUDA 版 PyTorch ...
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  if errorlevel 1 (
    echo [提示] cu128 安装失败，尝试默认源 ...
    python -m pip install torch
  )
)
if errorlevel 1 exit /b 1

:torch_done
echo [RWKV-EASY] 安装 RWKV-EASY ...
python -m pip install -e .
if errorlevel 1 exit /b 1

echo [RWKV-EASY] 环境检查：
python -m rwkv_easy doctor

if not "%SKIP_SELFTEST%"=="1" (
  echo [RWKV-EASY] 运行 CPU 自检 ...
  python -m rwkv_easy selftest --steps 8
)

echo.
echo [RWKV-EASY] 安装完成。下一步：
echo   1) 把 RWKV-7 权重放到 models\，把 ShareGPT 数据放到 data\
echo   2) python -m rwkv_easy init --model models\你的模型.pth --data data\sharegpt.json
echo   3) python -m rwkv_easy train -c configs\my-lora-run.yaml
echo.
echo   之后每次运行前先执行：.venv\Scripts\activate.bat
pause
