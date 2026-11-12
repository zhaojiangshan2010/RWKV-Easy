# 一键向导（新手 / 懒人）

`quickstart.py` 把**安装**、**配置**和**训练**合成一条路径：

```bash
python quickstart.py
```

Windows 双击 `quickstart.bat`，Linux / macOS 也可以 `bash quickstart.sh`。

向导不依赖第三方 TUI 库：Windows 用 `msvcrt`，Linux/macOS 用 `termios`；
无 TTY 环境（CI、管道）会自动退化成“分行问答”。界面支持 UTF-8 制表符，
Windows 传统控制台会自动退回 ASCII 边框。

---

## 1. quickstart 做了什么

启动后会先看到 3 步说明，然后依次执行：

1. **检测环境**：Python 版本、NVIDIA 驱动、CUDA、显存；
2. **安装依赖**：需要时创建项目内 `.venv`，按 CUDA 版本选择 PyTorch wheel
   （`cu130 → cu128 → cu126 → cu124 → cu121 → cu118` 逐个尝试），
   安装本项目，CUDA + Linux 上可选安装 `rwkv-fla`；
3. **打开向导**：选模板 → 确认模型/数据 → 开始训练。

没有 NVIDIA GPU 时，quickstart 会先提示“只能做 CPU 自检”，避免误装后
等很久才发现不能正式训练。

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
| `--(no-)infctx-parallel-backward` | DDP + infctx 是否逐块反传（透传给向导） |
| `-o out.yaml` | 指定配置文件输出路径 |

> 安装失败时会打印手动命令，不会静默退出。
> 再次运行 `python quickstart.py` 会复用 `.venv`，不会重复装 torch。

---

## 2. 欢迎页

```
──────────────────────────────────────────────────────────────
 欢迎使用 RWKV-EASY                  RWKV-7 LoRA / State Tuning，支持 infctx
──────────────────────────────────────────────────────────────
  设备：CUDA：B1.gpu.large，23.3 GiB
  PyTorch：2.14.1+cu130（CUDA 13.0）   rwkv-fla：可用
  接下来只需 3 步：① 选模板  ② 确认模型和数据集  ③ 开始训练。
  第一次用建议直接回车，采用推荐配置；想先了解选项可以看「新手引导」。

 ▶ 使用推荐配置，下一步
   按场景选择模板
   查看新手引导（3 分钟）
   退出

  │ 按当前显卡和模型自动选择 LoRA + bf16-full + compile；大多数情况下这就是最佳起点。

 ↑↓ 选择   Enter 确认   g 新手引导   q 退出
```

- **直接回车**：套用推荐模板，进入首页；这是最省心的路径。
- **按场景选择模板**：长上下文、State、省显存、只跑 20 步。
- **新手引导**：4 页，解释工具做什么、训练前准备什么、模板怎么选、常见坑。
- 想跳过欢迎页：`rwkv-easy wizard --no-welcome`。

---

## 3. 首页

```
──────────────────────────────────────────────────────────────
 第 1 步  一键配置            CUDA：B1.gpu.large，23.3 GiB · torch 2.14.1+cu130
──────────────────────────────────────────────────────────────
  ✓ 已使用推荐配置；可以直接开始，也可以继续修改
 ▶ 配置模板             [推荐] LoRA 标准微调            按场景一键套用
   训练方式             ◀ LoRA ▶                        LoRA 稳，State 省
   长上下文（infctx）   ◀ 关闭 ▶                        长对话 / 长文档
   模型权重             /gemini/.../rwkv7-g1d-0.1b.pth  官方 .pth/.safetensors
   数据集               .../examples/sharegpt_tiny.json ShareGPT / messages / 纯文本
   LoRA 模块            6 个模块                        默认 6 个线性层
   LoRA 层              全部层                          默认全部层
   WKV 后端             ◀ fla ▶                         CUDA 优先 fla
   精度                 ◀ bf16-full ▶                   GPU 推荐 bf16-full
   训练参数             bsz 8×1 / ctx 2048              batch / 上下文 / 学习率
   新手引导             3 分钟了解                      第一次用先看这里
   高级设置             全部字段                        模型 / 数据 / 分布式
   开始训练             保存配置并启动                  检查摘要后开跑
   仅保存配置           生成 YAML                       不启动训练
──────────────────────────────────────────────────────────────
 ↑↓ 移动   ←→ 切换   Enter 编辑/进入   g 引导   p 模板   ? 按键   q 退出
```

每一行右侧是**一句话注释**；选中行下方还有一块「说明」，解释这个选项是什么、
什么时候该改。终端很矮时会自动滚动，并显示 `↑/↓ 还有 N 项`。

### 按键

| 按键 | 作用 |
|---|---|
| `↑` / `↓` | 上下移动 |
| `←` / `→` | 切换当前选项（训练方式 / infctx / 后端 / 精度） |
| `Enter` | 进入编辑、选择文件、打开子菜单 |
| `空格` | 在多选界面勾选/取消（LoRA 模块、block 层） |
| `a` / `n` | 多选界面全选 / 清空 |
| `PgUp` / `PgDn` | 长表单翻页 |
| `/` | 高级表单里按名称/字段/说明搜索参数 |
| `g` | 打开新手引导 |
| `p` | 打开配置模板 |
| `?` | 显示完整按键说明 |
| `q` / `Esc` | 返回或退出 |

### 选择模型 / 数据集

- 自动扫描 `models/`、`data/`、`examples/` 和当前目录；
- 列表显示文件名和大小，选中项下方显示完整路径；
- 选中后立即校验：
  - 模型：存在、`.pth/.safetensors`、能推断 RWKV-7 结构；
  - 数据：存在、扩展名、能识别 ShareGPT / messages / 纯文本；
- 没有候选时会直接提示“把文件放到 models/ 或 data/”，也可以手动输入；
- 手动输入支持 `~` 和环境变量，路径里有空格也可以。

示例数据 `examples/sharegpt_tiny.json` 会被优先推荐，适合第一次跑通链路。

### 选择 LoRA 模块 / 层

- “LoRA 模块”：6 个候选项，空格勾选；默认全选；
- “LoRA 层”：全部 / 前一半 / 后一半 / 自定义；
- 自定义时用空格勾选任意 block（0-based），配置里写入
  `lora.target_layers: [0, 2, 5]`；`null` 表示全部层。

---

## 4. 配置模板

首页第一项就是模板。也可以按 `p` 随时打开。

| 模板 | 适合场景 | 关键设置 |
|---|---|---|
| `[推荐] LoRA 标准微调` | 第一次用；大多数对话/指令微调 | LoRA，按显卡和权重大小自动选 bf16-full / compile / batch |
| `[长文] 长上下文 LoRA` | 长对话、长文档、长记忆 | infctx，ctx ≥ 8192，chunk 2048，bsz ≤ 2 |
| `[轻量] State Tuning 快速适配` | 小数据、人设/风格注入 | state，lr 5e-3，weight_decay 0，参数 < 0.1% |
| `[轻量+长文] 长上下文 State` | 长文本 + 极小参数 | state + infctx，只有第一块反传 |
| `[低显存] 省显存模式` | 8–12GB 显存或大模型 | bsz 1，grad_accum ≥ 8，梯度检查点，关 compile |
| `[试跑] 先跑 20 步试试` | 确认模型/数据/训练链路 | max_steps 20，每步打日志，结束自动合并 |
| `[高级] 自定义` | 已经知道自己要什么 | 不修改当前参数 |

套用模板只改训练相关配置，**模型、数据集、输出目录、seed 保持不变**。
手动改了训练方式/精度/参数后，首页模板会显示 `[高级] 自定义`，提醒你当前
配置不再是某个模板的默认值。

### 多卡逐块反传

当检测到 **多卡 + infctx + LoRA** 时，首页会出现「多卡逐块反传」开关：

- **开启（默认）**：每个 chunk 立即 backward，除最后一块外跳过 all-reduce；
  最后一块统一同步梯度。显存 O(chunk_ctx)，长上下文不再随 `ctx_len` 增长。
- **关闭**：所有 chunk 最后统一反传，显存随 `ctx_len` 增长，长文本更容易 OOM。

State Tuning + 多卡、以及 DeepSpeed 暂不支持逐块路径，向导会自动关闭这个
开关并回退到最后统一反传。高级设置里对应字段是
`train.infctx_parallel_backward`。

---

## 5. 新手引导

按 `g` 或从欢迎页进入，共 4 页，`←/→` 翻页，`q` 返回：

1. **这个工具能做什么**：只做训练；LoRA vs State；infctx；completion-only loss。
2. **训练前准备什么**：模型放 `models/`、数据放 `data/`、先 `check-model` /
   `inspect-data`；没有模型/数据也能进向导，但开始训练前必须选。
3. **模板怎么选**：按“第一次 / 长文 / 小数据 / 低显存 / 只试跑”给出对应模板。
4. **常见坑**：CPU 只能自检、OOM 的处理顺序、loss 不降先查 mask、推理
   prompt 边界、decay 校准。

---

## 6. 参数注释与高级设置

高级设置分成 5 组，覆盖 `Config` 的所有字段：

| 组 | 内容 |
|---|---|
| 训练参数 | lr / batch / accum / ctx / precision / compile / infctx / 多卡逐块反传 / 保存策略… |
| 模型与 WKV | decay / backend / wkv_chunk / head_size_divisor… |
| 数据 | format / mode / template / train_on / EOS / val_ratio / workers… |
| 分布式 / 日志 | DDP / DeepSpeed ZeRO / wandb / 日志级别… |
| LoRA / State | rank / alpha / dropout / target_regex / init_std |

表单长这样：

```
 ▶ 训练方式             LoRA                LoRA 稳，State 省
   infctx（长上下文）   关                  长文本分块训练
   ctx_len              2048                单条样本 token 上限
   chunk_ctx（infctx）  512                 每块 token 数
   micro_batch_size     8                   单次前向样本数
   ...

  ┌─ 参数说明 ──────────────────────────────────────────────┐
  │ 每条训练样本的最大长度。对话一般 2048；长文档用 8192    │
  │ 或更大，并配合 infctx。显存不够时先降它。               │
  └─────────────────────────────────────────────────────────┘
```

- 每行右侧是行内注释，选中行下方是完整说明；
- `←/→` 按步长调整数值；
- `Enter` 输入精确值（输入空 = `None`/自动）；
- 布尔项 `←/→` 或空格切换；
- `PgUp/PgDn` 翻页；`/` 可按名称/字段/说明搜索并跳到对应参数；
- `q` 返回上一级。

不确定的字段保持默认即可；改完会自动套用环境约束（例如 CPU 强制 fp32、
没有 rwkv-fla 自动回退 native）。

---

## 7. 默认值怎么来的

- 有 GPU：`bf16-full` + `fla`（已安装时）+ `compile: true`；
- 无 GPU：`fp32` + `native` + `compile: false`，模型选中后 batch 固定为 1 并开
  梯度检查点（CPU 只建议自检）；
- 根据权重文件大小估算 batch：
  - < 1.2 GB：bsz 8，关梯度检查点；
  - < 3 GB：bsz 4，开梯度检查点；
  - < 8 GB：bsz 2，开梯度检查点；
  - 更大：bsz 1，开梯度检查点；
- 选择 infctx：`ctx_len=8192`、`chunk_ctx=2048`、`infctx_backward=true`、
  `grad_checkpoint=false`（显存只与 chunk 有关）；
- LoRA：`r=16, alpha=32, lr=1e-4`；
- State：`lr=5e-3, weight_decay=0`。

欢迎页的推荐模板会根据当前模型信息重新计算一次 batch；之后手动改模型也会
重新套用与显存/文件大小相关的默认值。

---

## 8. 非交互 / 脚本用法

没有 TTY 时自动使用分行问答：

```bash
python -m rwkv_easy wizard --non-interactive --no-train -o my.yaml
```

它会逐项列出带注释的选项和编号，输入编号修改，直接回车保存。适合 CI、
远程无终端环境或想用脚本生成配置的场景。

跳过欢迎页：

```bash
python -m rwkv_easy wizard --no-welcome
```

自动化时可以显式指定多卡逐块反传：

```bash
python -m rwkv_easy wizard --non-interactive \
  --infctx-parallel-backward --no-train -o my.yaml
# 关闭：--no-infctx-parallel-backward
```

---

## 9. 和 install.sh / train.sh 的关系

- `install.sh` / `install.bat`：只做安装，适合自动化；
- `train.sh` / `train.bat`：用已有 YAML 训练；
- `quickstart.py`：安装 + 交互式配置 + 训练，一条龙。

安装完成后，也可以随时运行：

```bash
rwkv-easy wizard
```

---

## 10. 常见问题

**Q：没有 GPU，向导还能用吗？**
可以，会自动选 CPU + fp32，只适合小模型自检。正式训练请用 NVIDIA GPU。

**Q：CUDA 13 没有对应 wheel？**
脚本按 `cu130 → cu128 → …` 逐个尝试；CUDA 13 驱动可以运行 cu128 的 wheel。

**Q：`rwkv-fla` 安装失败？**
不影响训练；向导会把 backend 回退到 `native`。也可以手动
`pip install rwkv-fla` 后重新进入向导。

**Q：想改已经生成的配置？**
直接编辑 `configs/xxx.yaml`，或用 `train --set` 临时覆盖；也可以重新运行
`rwkv-easy wizard` 套模板。

**Q：路径有中文/空格？**
向导用 Python 的 `input` 接收路径，支持中文和空格；Windows 建议使用绝对路径。

**Q：按 q 会丢配置吗？**
只有在“开始训练/仅保存配置”确认后才会写 YAML；中途按 q 不会覆盖已有文件。
