# 一键向导（新手 / 懒人）

`quickstart.py` 把**安装**和**训练配置**合成一条路径：

```bash
python quickstart.py
```

Windows 双击 `quickstart.bat`；Linux / macOS 也可以：

```bash
bash quickstart.sh
```

它不依赖任何第三方 TUI 库：Windows 用 `msvcrt`，Linux/macOS 用 `termios`；
在无 TTY 环境（CI、管道）会自动退化成“分行问答”。

---

## 1. 它做了什么

1. 检测 Python 版本、NVIDIA 驱动、CUDA 版本、显存；
2. 需要时创建项目内 `.venv`（不想用加 `--no-venv`）；
3. 根据 CUDA 版本选择 PyTorch wheel：
   `cu130 → cu128 → cu126 → cu124 → cu121 → cu118`，逐个尝试；
4. 安装本项目；CUDA + Linux 上可选安装 `rwkv-fla`；
5. 进入向导，生成完整 YAML；
6. 直接启动训练（也可以只保存配置）。

常用参数：

| 参数 | 作用 |
|---|---|
| `--cpu` | 强制 CPU 版 PyTorch |
| `--cuda-tag cu124` | 指定 wheel 标签，不自动探测 |
| `--reinstall` | 已有 torch 也重新安装 |
| `--skip-install` | 跳过安装，直接进向导 |
| `--install-only` | 只安装，不进向导 |
| `--no-venv` | 不创建 `.venv`，装到当前环境 |
| `--yes` | 自动确认可选依赖（rwkv-fla） |
| `--non-interactive` | 向导使用分行问答 |
| `--no-train` | 只生成配置，不训练 |
| `-o out.yaml` | 指定配置文件输出路径 |

> 安装失败时会打印手动命令，不会静默退出。
> 之后再次运行 `python quickstart.py` 会复用 `.venv`，不会重复装 torch。

---

## 2. 界面与按键

```
============================================================
 RWKV-EASY 一键配置
 CUDA：B1.gpu.large，23.3 GiB | Python 3.11.9 | torch 2.14.1
------------------------------------------------------------
 ▶ 训练方式              ◀ LoRA ▶
   长上下文（infctx）      ◀ 关闭 ▶
   模型权重               models/rwkv7-g1d-0.1b.pth
   数据集                 data/sharegpt.json
   LoRA 模块              6 个模块
   LoRA 层                全部层
   WKV 后端               ◀ fla ▶
   精度                   ◀ bf16-full ▶
   训练参数摘要           bsz 8 × 1 / ctx 2048
   高级设置               所有配置字段…
   开始训练               保存配置并启动
   仅保存配置             生成 YAML
------------------------------------------------------------
 ↑↓ 移动   ←→ 修改选项   Enter 编辑/进入   q 退出
```

| 按键 | 作用 |
|---|---|
| `↑` / `↓` | 上下移动 |
| `←` / `→` | 切换当前选项（训练方式 / infctx / 后端 / 精度） |
| `Enter` | 进入编辑、选择文件、打开子菜单 |
| `空格` | 在多选界面勾选/取消（LoRA 模块、block 层） |
| `a` / `n` | 多选界面全选 / 清空 |
| `q` / `Esc` | 返回或退出 |

### 选择模型 / 数据集

- 自动扫描 `models/`、`data/`、`examples/`、当前目录；
- 选中后会立即校验：
  - 模型：存在、`.pth/.safetensors`、能推断 RWKV-7 结构；
  - 数据：存在、扩展名、能识别 ShareGPT / messages / 纯文本；
- 也可以选“手动输入路径…”，支持 `~` 和环境变量。

### 选择 LoRA 模块 / 层

- “LoRA 模块”：6 个候选项，空格勾选；
- “LoRA 层”：全部 / 前一半 / 后一半 / 自定义；
- 自定义时用空格勾选任意 block（0-based），配置里写入
  `lora.target_layers: [0, 2, 5]`；`null` 表示全部层。

### 高级设置

“高级设置”分成 5 组，覆盖 `Config` 的所有字段：

| 组 | 内容 |
|---|---|
| 训练参数 | lr / batch / accum / ctx / precision / compile / infctx / 保存策略… |
| 模型与 WKV | decay / backend / wkv_chunk / head_size_divisor… |
| 数据 | format / mode / template / train_on / EOS / val_ratio / workers… |
| 分布式 / 日志 | DDP / DeepSpeed ZeRO / wandb / 日志级别… |
| LoRA / State | rank / alpha / dropout / target_regex / init_std |

表单里：

- `←/→` 按步长调整数值；
- `Enter` 输入精确值（输入空 = `None`）；
- 布尔项 `←/→` 或空格切换；
- `q` 返回上一级。

---

## 3. 默认值怎么来的

- 有 GPU：`bf16-full` + `fla`（已安装时）+ `compile: true`；
- 无 GPU：`fp32` + `native` + `compile: false`，模型选中后 batch 固定为 1 并开
  梯度检查点（CPU 只建议自检）；
- 根据权重文件大小估算 batch：
  - < 1.2 GB：bsz 8，关梯度检查点；
  - < 3 GB：bsz 4；
  - < 8 GB：bsz 2；
  - 更大：bsz 1；
- 选择 infctx：`ctx_len=8192`、`chunk_ctx=2048`、`infctx_backward=true`、
  `grad_checkpoint=false`（显存只与 chunk 有关）；
- LoRA：`r=16, alpha=32, lr=1e-4`；
- State：`lr=5e-3, weight_decay=0`。

所有默认值都可以在向导里改；生成的 YAML 也可以继续手改。

---

## 4. 非交互 / 脚本用法

没有 TTY 时自动使用分行问答：

```bash
python -m rwkv_easy wizard --non-interactive --no-train -o my.yaml
```

它会依次询问训练方式、infctx、模型、数据、模块、层、后端、精度，最后确认。
也可以只生成默认配置后退出。

---

## 5. 和 install.sh / train.sh 的关系

- `install.sh` / `install.bat`：只做安装，适合自动化；
- `train.sh` / `train.bat`：用已有 YAML 训练；
- `quickstart.py`：安装 + 交互式配置 + 训练，一条龙。

安装完成后，也可以随时运行：

```bash
rwkv-easy wizard
```

---

## 6. 常见问题

**Q：没有 GPU，向导还能用吗？**
可以，会自动选 CPU + fp32，只适合小模型自检。正式训练请用 NVIDIA GPU。

**Q：CUDA 13 没有对应 wheel？**
脚本按 `cu130 → cu128 → …` 逐个尝试；CUDA 13 驱动可以运行 cu128 的 wheel。

**Q：`rwkv-fla` 安装失败？**
不影响训练；向导会把 backend 回退到 `native`。也可以手动
`pip install rwkv-fla` 后重新进入向导。

**Q：想改已经生成的配置？**
直接编辑 `configs/xxx.yaml`，或用 `train --set` 临时覆盖。

**Q：路径有中文/空格？**
向导用 Python 的 `input` 接收路径，支持中文和空格；Windows 建议使用绝对路径。

**Q：按 q 会丢配置吗？**
只有在“开始训练/仅保存配置”确认后才会写 YAML；中途按 q 不会覆盖已有文件。
