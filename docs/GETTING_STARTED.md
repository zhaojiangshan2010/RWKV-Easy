# 快速上手（完整路径）

[README](../README.md) 只给最短路径；本页是从安装到训练完成的完整流程。
第一次使用建议按顺序读，已经装好环境的可以直接跳到第 3 节。

---

## 0. 环境要求

- Python **3.10+**
- 训练：NVIDIA GPU（CUDA 11.8+）或 AMD GPU（ROCm）；CPU 也能跑，但只建议做自检
- 磁盘：模型 + 数据集 + 输出，建议预留 20GB 以上
- 运行时只依赖 `torch` + `pyyaml`；DeepSpeed、rwkv-fla、safetensors 都是可选项

---

## 1. 安装

### 懒人一键（推荐）

```bash
python quickstart.py
```

Windows 双击 `quickstart.bat`，Linux/macOS 也可以 `bash quickstart.sh`。

它会：检测 NVIDIA 驱动与 CUDA → 创建 `.venv` → 按 CUDA 版本安装对应的
PyTorch → 安装本项目 →（可选）安装 rwkv-fla → 进入交互式向导 → 直接训练。

向导启动后先显示欢迎页：**直接回车使用推荐模板**，也可以按场景选长上下文 /
State / 省显存 / 20 步试跑，或按 `g` 查看 4 页新手引导。首页每个参数都有
行内注释，选中后还有详细说明。参数与界面见 [`WIZARD.md`](WIZARD.md)。

### 只安装，不配置

Linux / macOS：

```bash
bash install.sh
```

Windows（双击也可以）：

```bat
install.bat
```

PowerShell 用户可以用 `install.ps1`。

脚本会创建 `.venv`、安装 PyTorch 与本项目、运行 `doctor`，最后跑一遍 CPU 自检。
驱动比较旧、装不上自动选择的 CUDA 版时，用 `TORCH_INDEX` 指定：

```bash
TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash install.sh
```

### 手动安装

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch pyyaml
pip install -e .
```

可选加速：

```bash
pip install deepspeed    # 多卡 ZeRO（不装就用 DDP）
pip install rwkv-fla     # CUDA/ROCm 上的 Triton WKV 内核
pip install safetensors  # 读取 .safetensors 权重
```

> 之后每次运行前先激活 `.venv`（`quickstart.py` 与 `train.sh/.bat` 会自动处理）。

---

## 2. 准备模型与数据

```bash
mkdir -p models data
# 把 RWKV-7 的 .pth / .safetensors 放进 models/
# 把 ShareGPT 数据集放进 data/
```

本项目读取**官方 state_dict 格式**的 `.pth`，维度会自动从权重推断。
RWKV-7 权重可以从 HuggingFace 上的 RWKV 官方仓库获取（例如 `RWKV/rwkv7-...`）。
`models/` 已被 `.gitignore` 忽略；不确定该下哪个文件、或下载后想先确认格式，
见 [`MODEL_CHECK.md`](MODEL_CHECK.md)。

数据最小示例（ShareGPT）：

```json
[
  {
    "conversations": [
      {"from": "system", "value": "你是一个助手。"},
      {"from": "human", "value": "你好"},
      {"from": "gpt", "value": "你好！"}
    ]
  }
]
```

同时兼容 OpenAI `messages`、两列 `prompt`/`response` 和纯文本，支持
`.json` / `.jsonl` / `.txt` / 目录。默认**只有 assistant 回复参与 loss**，
System / User 只作为上下文。完整格式、模板与 mask 规则见 [`DATASET.md`](DATASET.md)。

想先试试？项目自带一份 [`examples/sharegpt_tiny.json`](../examples/sharegpt_tiny.json)。

---

## 3. 生成配置

```bash
rwkv-easy init --model models/你的模型.pth --data data/sharegpt.json --method lora
```

会生成 `configs/my-lora-run.yaml`（带中文注释，改路径即可）。加 `--infctx`
可以生成长上下文配置，`--method state` 生成 State Tuning 配置。也可以直接复制
`configs/` 里的模板。

---

## 4. 先检查数据

```bash
rwkv-easy inspect-data -c configs/my-lora-run.yaml -n 3
```

它会打印每一段样本，并用 `【】` 标出**会参与 loss** 的部分：

```
--- 窗口 0 ---
User: 请只回答一个词

Assistant: 【苹果】【
】
```

如果 `【】` 里的内容不对（比如把用户的问题也算进去了），检查
`data.template` / `data.train_on`。

---

## 5. 开始训练

```bash
rwkv-easy train -c configs/my-lora-run.yaml
```

- 单卡：直接训练
- 多卡：自动用 DDP 拉起多进程（`dist.mode: auto`）
- 想用 DeepSpeed：`--set dist.mode=deepspeed`
- CUDA 上想最快：用 `configs/lora_sharegpt_fla.yaml`（rwkv-fla + compile）
- 想先确认能跑：`rwkv-easy train -c configs/debug_cpu.yaml`（随机小模型）

训练过程中会打印：

```
[step 40/1000 4.0%] loss 2.3156 (ema 2.3401) lr 9.80e-05 12.3k tok/s 8.2GiB ETA 46m
```

输出目录（例如 `runs/rwkv7-lora-lora-20260101-120000/`）里有：

| 文件 | 说明 |
|---|---|
| `adapter-step0000100.pth` | LoRA / State 权重（很小，可断点续训） |
| `trainer-step0000100.pt` | 优化器 / 学习率 / 进度 |
| `merged.pth` | 训练结束自动合并好的完整模型 |
| `config.resolved.yaml` | 实际生效的全部配置 |
| `train.log` / `metrics.jsonl` | 日志与指标 |

---

## 6. 验证训练结果

```bash
rwkv-easy generate -m runs/xxx/merged.pth -p "你好，请介绍一下自己" --max-tokens 64
```

`generate` 只是诊断工具，不是推理产品。如果不想用自动合并的 `merged.pth`，
也可以手动合并：

```bash
rwkv-easy merge --base models/rwkv7.pth \
                --adapter runs/xxx/adapter-step0000100.pth \
                -o merged.pth --dtype bf16
```

---

## 7. 断点续训

```bash
rwkv-easy train -c configs/my-lora-run.yaml --resume runs/xxx
```

权重、优化器状态、学习率进度都会恢复。

---

## 8. 训练时内部发生了什么

```
读 YAML → 校验 → 建 run 目录
  → 读 ShareGPT、分词、构造 completion-only mask、缓存
  → 加载 RWKV-7（meta 设备 + mmap + assign，按精度定型）
  → decay 自动校准（三种公式各算一次 loss，选最低）
  → 注入 LoRA 或 time_state（冻结基座）
  → 可选 torch.compile（fla 编译前向 / native 编译 WKV 单块）
  → 循环：前向到 ln_out → 只对可训练 token 过 head → CE + L2Wrap
          （infctx 则分块、每块立即反传）→ 反传 → 裁剪 → step
  → 定期保存 adapter + 优化器；结束导出 merged.pth
```

启动日志里的每一行都能对应上面的步骤；出问题时先看最后一条成功的日志。
实现细节见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

---

## 9. 四种训练组合

| 组合 | 配置 | 适合场景 |
|---|---|---|
| **LoRA** | `train.method: lora`、`infctx: false` | 常规指令微调，效果与稳定性最好 |
| **State Tuning** | `train.method: state`、`infctx: false` | 小数据快速适配；参数量 < 0.1% |
| **LoRA + infctx** | `method: lora`、`infctx: true` | 长对话 / 长文档；显存只与 `chunk_ctx` 有关 |
| **State + infctx** | `method: state`、`infctx: true` | 长上下文 + 极小参数适配 |

对应模板：

```
configs/lora_sharegpt.yaml            # 通用 LoRA（native）
configs/lora_sharegpt_fla.yaml        # CUDA + rwkv-fla + compile，最快
configs/state_sharegpt.yaml
configs/lora_sharegpt_infctx.yaml     # 长上下文（默认 fla）
configs/state_sharegpt_infctx.yaml    # 长上下文 + State（默认 fla）
configs/lora_2gpu_deepspeed.yaml
configs/debug_cpu.yaml                # 随机小模型，CPU 冒烟
configs/self_identity_lora.yaml       # CPU 过拟合 demo
```

### LoRA 与 State 的区别

|  | LoRA | State Tuning |
|---|---|---|
| 训练什么 | 每个线性层旁的低秩矩阵 A/B | 每层 attention 的初始循环状态 `time_state` |
| 参数量（0.4B 模型） | 约 0.2–1% | 约 0.02% |
| 学习率 | 1e-4 ~ 5e-4 | 1e-3 ~ 1e-2 |
| 适配器文件 | 几 MB ~ 几十 MB | 几百 KB |
| 合并方式 | `W + (B@A) * alpha/r` | 覆盖 `time_state` 键 |

两种方法都只保存**可训练参数**，基座权重永远不会被改写。

---

## 10. 命令行速查

```bash
rwkv-easy init          生成配置（交互式向导）
rwkv-easy wizard        一键交互式配置并训练（新手推荐）
rwkv-easy inspect-data  检查数据与 loss mask
rwkv-easy check-model   验收一个 RWKV-7 权重（格式/架构/decay/出字）
rwkv-easy train         开始 / 继续训练
rwkv-easy merge         把适配器合并成完整模型
rwkv-easy generate      小规模生成（诊断）
rwkv-easy doctor        环境检查
rwkv-easy selftest      不下载权重、跑通完整流程
```

常用参数：

```bash
# 临时改一个配置项，不用编辑文件
rwkv-easy train -c cfg.yaml --set train.lr=2e-4 --set train.micro_batch_size=1

# 强制单进程
rwkv-easy train -c cfg.yaml --single

# 换 WKV 后端
rwkv-easy train -c cfg.yaml --backend fla

# 环境检查（JSON 输出，便于贴给他人）
rwkv-easy doctor --json

# 完整自检（8 种组合）
rwkv-easy selftest --full
```

所有命令等价于 `python -m rwkv_easy <command>`。字段含义见
[`TRAINING_PARAMETERS.md`](TRAINING_PARAMETERS.md)，快速调参见 [`CONFIG.md`](CONFIG.md)。

---

## 11. 验收一份新权重

建议在正式训练前花 5 分钟确认「格式对、架构对、decay 对、能出字」：

```bash
# 1) 一条命令验收：输出现成的结构、decay 三种候选 loss、一段续写
rwkv-easy check-model -m models/你的模型.pth

# 2) 环境 + WKV 数值自检
rwkv-easy doctor

# 3) 对话模型再按对话格式测一次
rwkv-easy generate -m models/你的模型.pth -p "你好" --max-tokens 64 --template rwkv_world

# 4) 20 步 LoRA 冒烟
rwkv-easy train -c configs/lora_sharegpt.yaml \
  --set model.path=models/你的模型.pth --set train.max_steps=20

# 5) infctx 冒烟（没有 rwkv-fla 时加 --set model.backend=native）
rwkv-easy train -c configs/lora_sharegpt_infctx.yaml \
  --set model.path=models/你的模型.pth --set train.max_steps=20
```

判断标准：

- `check-model` 会打印 `均匀分布基线` 与 `clampw / softplus / exp` 三个 loss，
  最低者被选中，且应明显低于基线；
- 续写应是通顺的文本，而不是乱码/重复；
- `doctor` 的 WKV 自检误差应 < 1e-4；
- 20 步训练里 loss 应从前 1–2 步的 `ln(vocab)` 附近开始下降。

完整异常对照表见 [`MODEL_CHECK.md`](MODEL_CHECK.md)。

---

## 12. 本机 CPU 自检（不需要 GPU）

WKV 算子是纯 PyTorch 写的，所以整条链路都能在 CPU 上验证。三条命令：

```bash
python -m pytest tests -q                            # 131 个（CPU 上 126 passed + 5 fla skipped）
rwkv-easy selftest --full                            # lora/state × infctx on/off × fp32/bf16
rwkv-easy train -c configs/debug_cpu.yaml             # 随机小模型，20 步
```

CPU 能验证：WKV 数学（与 float64 逐 token 参考对照）、completion-only mask、
infctx 状态传递、LoRA 合并、State 梯度、bf16/bf16-full、断点续训、2 进程 DDP。
CPU 不能验证：CUDA 性能/显存、fp16、DeepSpeed、fla/Triton、NCCL、7B 级训练速度。

完整矩阵、每项判据与实测数字见 [`CPU_VERIFICATION.md`](CPU_VERIFICATION.md)。

---

## 13. 10 分钟看到效果：教模型记住自己的名字

`examples/self_identity.json` 里有 58 条短对话，其中 50 条在问「你是什么模型」，
统一回答「RWKV 0.1B 大模型」。

```bash
rwkv-easy train -c configs/self_identity_lora.yaml
rwkv-easy generate -m runs/self-identity-*/merged.pth -p "你是谁？" --template rwkv_world
```

0.1B 模型在 CPU 上 400 步约 5 分钟：loss 3.75 → 0.087，
`你是什么模型？`、`你是谁？`、`你是哪个 AI 助理？`、`What model are you?`
全部回答 `RWKV 0.1B 大模型`（含训练中没见过的问法）。这是**故意过拟合**的演示，
用来确认「数据 → 训练 → 合并 → 推理」整条链路，不代表真实微调效果。

---

## 14. 显存不够怎么办

按顺序尝试：

1. 降 `micro_batch_size`，同时把 `grad_accum` 调大（总 batch 不变）
2. infctx 下先确认 `infctx_backward: true`，再把 `chunk_ctx` 从 2048 降到 1024/512
3. `grad_checkpoint: true`
4. `precision: bf16-full`（只有冻结基座用 bf16，LoRA/state 仍是 fp32）
5. 多卡 + DDP / DeepSpeed

每个旋钮的收益、显存模型与按模型规模的建议见 [`PERFORMANCE.md`](PERFORMANCE.md)。

---

## 15. 遇到问题

先查 [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)（安装、显存、性能、数值、数据、
模型加载、infctx、分布式）。报错时附上 `rwkv-easy doctor --json` 的输出和
`runs/.../train.log` 的最后几十行，最容易定位。

常见入口：

| 我想…… | 看 |
|---|---|
| 第一次跑通 / 懒人一键 | 本文、[`WIZARD.md`](WIZARD.md) |
| 调超参 / 查字段含义 | [`TRAINING_PARAMETERS.md`](TRAINING_PARAMETERS.md)、[`CONFIG.md`](CONFIG.md) |
| 改代码 / 加后端 | [`ARCHITECTURE.md`](ARCHITECTURE.md)、[`EXTENDING.md`](EXTENDING.md) |
| 参与开发 / 跑测试 | [`DEVELOPMENT.md`](DEVELOPMENT.md) |
