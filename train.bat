@echo off
REM 一键训练：train.bat [配置文件]
REM
REM   train.bat
REM   train.bat configs\state_sharegpt.yaml
REM
REM 需要临时覆盖参数时，请直接使用完整命令，例如：
REM   python -m rwkv_easy train -c configs\lora_sharegpt.yaml --set train.lr=2e-4
setlocal
cd /d "%~dp0"

set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=configs\lora_sharegpt.yaml"

if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

python -m rwkv_easy train -c "%CONFIG%"
endlocal
