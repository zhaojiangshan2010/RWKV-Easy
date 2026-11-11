@echo off
REM ============================================================
REM RWKV-EASY 一键安装 + 交互式训练向导（Windows）
REM
REM   quickstart.bat
REM
REM 参数会原样传给 quickstart.py，例如：
REM   quickstart.bat --cpu
REM   quickstart.bat --skip-install --no-train
REM ============================================================
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 没有找到 Python，请先从 https://www.python.org/downloads/ 安装 Python 3.10+
  echo        安装时请勾选 "Add Python to PATH"
  pause
  exit /b 1
)

python quickstart.py %*
if errorlevel 1 pause
