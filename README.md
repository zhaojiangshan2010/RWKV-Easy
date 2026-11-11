# RWKV-EASY

**RWKV-7 的一键式微调工具。** 只做训练，只做两件事：**LoRA** 和 **State Tuning**，
并且两者都能与 RWKV 的 **infinite context（infctx）** 组合。

- 数据：**ShareGPT** 格式（同时兼容 OpenAI messages / 纯文本）
- loss：**completion-only**，prompt 只作上下文，只有 assistant 回复参与训练
- 精度：**FP32 / FP16 / BF16 / BF16-full**
- 多卡：**DeepSpeed 可选**，不装也能用 PyTorch DDP
- 平台：CUDA 优先；纯 PyTorch 算子同时支持 **CPU / ROCm / MPS**
- 面向新手：一条命令安装、一条命令训练、出错信息讲人话

## 项目状态

| 范围 | 状态 |
|---|---|
| 数学正确性（WKV 前向/梯度、infctx 等价性、LoRA 合并） | ✅ CPU 上与 float64 参考对照，误差 < 1e-5 |
| 数据管线（ShareGPT、completion-only mask、packed/sample） | ✅ 98 个测试覆盖 |
| 端到端（训练、断点、合并、重载、bf16、2 进程 DDP） | ✅ CPU 上验证通过 |
| 真实 RWKV-7 权重 | ✅ `rwkv7-g1d-0.1b` 在 CPU 与 GPU 上都已验收：decay=clampw，NLL 2.54，续写连贯 |
| GPU 训练（LoRA / State / infctx / fp16 / DeepSpeed） | ✅ 24GB Ampere 级 GPU 全跑通，数字见 [`docs/GPU_VERIFICATION.md`](docs/GPU_VERIFICATION.md) |
| rwkv-fla（Triton 后端） | ⏳ 测试环境 torch 2.2 / triton 2.2 与 rwkv-fla 0.7 不兼容；需 torch ≥ 2.4 + triton ≥ 3.2 |

本机（Windows + CPU）已验证的内容与复现命令见
[`docs/CPU_VERIFICATION.md`](docs/CPU_VERIFICATION.md)。

---

## 目录

- [为什么需要它](#为什么需要它)
- [5 分钟上手](#5-分钟上手)
- [本机 CPU 验证](#本机-cpu-验证不需要-gpu)
- [拿到模型后的验收](#拿到模型后的验收)
- [四种训练组合](#四种训练组合)
- [命令行速查](#命令行速查)
- [数据集格式](#数据集格式)
- [显存与参数建议](#显存与参数建议)
- [常见问题](#常见问题)
- [项目结构](#项目结构)
- [进一步阅读](#进一步阅读)

---

## 为什么需要它

参考实现 [RWKV-PEFT](../example/RWKV-PEFT) 很强大，但有三个会直接卡住新手的坑：

| 问题 | RWKV-PEFT | RWKV-EASY |
|---|---|---|
| `infctx` + LoRA | ❌ HuggingFace PEFT 的 `forward` 会吃掉循环状态参数，运行即报 `forward_infctx() missing last_shift_states` | ✅ 自研 LoRA（只替换 `nn.Linear`，不改模型签名），与 infctx 天然兼容 |
| `infctx` + State Tuning | ❌ 两者在 `train.py` 里互斥 | ✅ 第一块用学习到的 `time_state`，之后由真实状态接管 |
| ShareGPT 数据 | ❌ 只有 binidx / jsonl / 两列 SFT，没有对话解析与 assistant-only mask | ✅ 原生 ShareGPT，自动构造只对 assistant 计 loss 的掩码 |
| 安装 | 需要 transformers / peft / lightning / rwkv / rwkv-fla / triton … | 只需要 `torch` + `pyyaml` |

另外还有两个“新手友好”但很实用的点：

- **衰减公式自动校准**：不同版本的 RWKV-7 实现使用不同的 decay 参数化，用错不会报错、
  但模型输出会明显变差。RWKV-EASY 会用一段内置中英文本自动比较三种公式，选 loss 最低的那个。
- **CPU 可完整自检**：`rwkv-easy selftest` 不需要下载任何权重，几分钟内跑完
  训练 → infctx → 保存 → 合并 → 重新加载 的完整链路。

---

## 5 分钟上手

### 0. 环境要求

- Python **3.10+**
- 训练：NVIDIA GPU（CUDA 11.8+）或 AMD GPU（ROCm）；CPU 也能跑，但只建议做自检
- 磁盘：模型 + 数据集 + 输出，建议预留 20GB 以上

### 1. 安装

Linux / macOS：

```bash
bash install.sh
```

Windows（双击也可以）：

```bat
install.bat
```

脚本会自动：创建 `.venv` → 安装 PyTorch（有 N 卡就装 CUDA 版）→ 安装本项目 →
运行 `doctor` 环境检查 → 跑一遍 CPU 自检。

> 驱动比较旧装不上 CUDA 版？用 `TORCH_INDEX` 指定：
> ```bash
> TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash install.sh
> ```

手动安装（如果你更喜欢自己管理环境）：

```bash
pip install torch pyyaml
pip install -e .
```

可选加速：

```bash
pip install deepspeed    # 多卡 ZeRO（不装就用 DDP）
pip install rwkv-fla     # CUDA/ROCm 上的 Triton WKV 内核
pip install safetensors  # 读取 .safetensors 权重
```

### 2. 准备模型与数据

```bash
mkdir -p models data
# 把 RWKV-7 的 .pth / .safetensors 放进 models/
# 把 ShareGPT 数据集放进 data/
```

RWKV-7 权重可以从 HuggingFace 上的 RWKV 官方仓库获取（例如 `RWKV/rwkv7-...`）。
`models/` 已被 `.gitignore` 忽略，不用担心把大文件提交进仓库；不确定该下哪个
文件、或者下载后想先确认格式，见 [`docs/MODEL_CHECK.md`](docs/MODEL_CHECK.md)。
本项目读取的是**官方 state_dict 格式**的 `.pth`，维度会自动从权重推断。

数据格式见 [数据集格式](#数据集格式)。想先试试？项目自带一份
[`examples/sharegpt_tiny.json`](examples/sharegpt_tiny.json)。

### 3. 生成配置

```bash
python -m rwkv_easy init --model models/你的模型.pth --data data/sharegpt.json --method lora
```

会生成 `configs/my-lora-run.yaml`（带中文注释，改路径即可）。
也可以直接复制 `configs/` 里的模板。

### 4. 先检查数据

```bash
python -m rwkv_easy inspect-data -c configs/my-lora-run.yaml -n 3
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

### 5. 开始训练

```bash
python -m rwkv_easy train -c configs/my-lora-run.yaml
```

- 单卡：直接训练
- 多卡：自动用 DDP 拉起多进程（`dist.mode: auto`）
- 想用 DeepSpeed：`--set dist.mode=deepspeed`
- 想先确认能跑：`python -m rwkv_easy train -c configs/debug_cpu.yaml`

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

### 6. 验证训练结果

```bash
python -m rwkv_easy generate -m runs/xxx/merged.pth -p "你好，请介绍一下自己" --max-tokens 64
```

`generate` 只是诊断工具，不是推理产品。

### 7. 断点续训

```bash
python -m rwkv_easy train -c configs/my-lora-run.yaml --resume runs/xxx
```

### 8. 训练时内部发生了什么

```
读 YAML → 校验 → 建 run 目录
  → 读 ShareGPT、分词、构造 completion-only mask、缓存
  → 加载 RWKV-7（meta 设备 + mmap + assign，按精度定型）
  → decay 自动校准（三种公式各算一次 loss，选最低）
  → 注入 LoRA 或 time_state（冻结基座）
  → 循环：前向（infctx 则分块传状态）→ CE + L2Wrap → 反传 → 裁剪 → step
  → 定期保存 adapter + 优化器；结束导出 merged.pth
```

启动日志里的每一行都能对应上面的步骤；出问题时先看最后一条成功的日志。

---

## 本机 CPU 验证（不需要 GPU）

WKV 算子是纯 PyTorch 写的，所以整条链路都能在 CPU 上验证。三条命令：

```bash
python -m pytest tests -q                      # 98 个测试，约 3-6 分钟
python -m rwkv_easy selftest --full            # lora/state × infctx on/off × fp32/bf16
python -m rwkv_easy train -c configs/debug_cpu.yaml   # 随机小模型，20 步
```

`debug_cpu.yaml` 不需要任何权重，会随机初始化一个 2 层小模型并跑通
「训练 → 保存 → 合并 → 重载」。预期输出：

```
[step 20/20 100.0%] loss 11.1975 (ema 11.2595) ...
已保存：adapter-step0000020.pth
完整权重：runs\debug-cpu-lora-infctx-...\merged.pth
```

CPU 能验证：WKV 数学（与 float64 逐 token 参考对照）、completion-only mask、
infctx 状态传递、LoRA 合并、State 梯度、bf16/bf16-full、断点续训、2 进程 DDP。
CPU 不能验证：CUDA 性能/显存、fp16、DeepSpeed、fla/Triton、NCCL、7B 级训练速度。

完整矩阵、每项判据与实测数字见
[`docs/CPU_VERIFICATION.md`](docs/CPU_VERIFICATION.md)。

### 10 分钟看到效果：教模型记住自己的名字

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

## 拿到模型后的验收

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

# 5) infctx 冒烟
rwkv-easy train -c configs/lora_sharegpt_infctx.yaml \
  --set model.path=models/你的模型.pth --set train.max_steps=20
```

判断标准：

- `check-model` 会打印 `均匀分布基线` 与 `clampw / softplus / exp` 三个 loss，
  最低者被选中，且应明显低于基线；
- 续写应是通顺的文本，而不是乱码/重复；
- `doctor` 的 WKV 自检误差应 < 1e-4；
- 20 步训练里 loss 应从前 1–2 步的 `ln(vocab)` 附近开始下降。

完整异常对照表见 [`docs/MODEL_CHECK.md`](docs/MODEL_CHECK.md)。

---

## 四种训练组合

| 组合 | 配置 | 适合场景 |
|---|---|---|
| **LoRA** | `train.method: lora`、`infctx: false` | 常规指令微调，效果与稳定性最好 |
| **State Tuning** | `train.method: state`、`infctx: false` | 小数据快速适配；参数量 < 0.1% |
| **LoRA + infctx** | `method: lora`、`infctx: true` | 长对话 / 长文档；显存只与 `chunk_ctx` 有关 |
| **State + infctx** | `method: state`、`infctx: true` | 长上下文 + 极小参数适配 |

对应的模板：

```
configs/lora_sharegpt.yaml
configs/state_sharegpt.yaml
configs/lora_sharegpt_infctx.yaml
configs/state_sharegpt_infctx.yaml
configs/lora_2gpu_deepspeed.yaml
configs/debug_cpu.yaml
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

## 命令行速查

```bash
rwkv-easy init          生成配置（交互式向导）
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

# 合并
rwkv-easy merge --base models/rwkv7.pth \
                --adapter runs/xxx/adapter-step0000100.pth \
                -o merged.pth --dtype bf16

# 环境检查（JSON 输出，便于贴给他人）
rwkv-easy doctor --json

# 完整自检（8 种组合）
rwkv-easy selftest --full
```

---

## 数据集格式

支持三种，自动识别：

### ShareGPT（推荐）

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

### OpenAI messages

```json
{"messages": [
  {"role": "user", "content": "你好"},
  {"role": "assistant", "content": "你好！"}
]}
```

### 纯文本

```json
{"text": "一整段用于继续预训练的文本……"}
```

也支持 `.jsonl`（每行一条）、`.txt`（整个文件一条）、目录（递归读取），
以及 `prompt`/`response`、`instruction`/`output` 这类两列格式。

### 对话模板

默认是 RWKV 世界模型常用的格式：

```
System: ...
User: ...
Assistant: ...
```

每轮之间空一行；`append_eos: end` 会在对话末尾追加 EOT（词表里的 `\x00`）。
也内置了 `chatml` 模板，或用 `data.system_prefix` 等字段自定义。

### completion-only loss

默认且唯一的行为：**只有 assistant 的回复参与 loss**，System / User / 工具返回
都只作为上下文。`Assistant: ` 前缀本身不计 loss，回复正文、它后面的分隔符、
以及可选的 EOT 计 loss；回复的第一个 token 由 prompt 最后一个 token 预测，
所以也会被监督。

```yaml
data:
  train_on: assistant      # 所有 assistant 轮次（默认）
  # train_on: last_assistant   # 只训练最后一条回复
  append_eos: end          # never / end / each_assistant
```

用 `rwkv-easy inspect-data` 可以直接看到哪些 token 参与 loss（`【】`）：

```
User: 请只回答一个词

Assistant: 【苹果】【

】
```

目前没有“连 prompt 一起训练”的开关（纯文本数据除外，`format: text` 时整段参与）。

更多细节见 [`docs/DATASET.md`](docs/DATASET.md)。

---

## 显存与参数建议

以下为 bf16 + 梯度检查点开启时的**粗略**估计（只训练 LoRA/State，基座不存梯度）：

| 模型 | 训练方式 | ctx 2048 / bsz 2 | ctx 8192 + infctx / bsz 1 |
|---|---|---|---|
| 0.4B | LoRA | ~8 GB | ~10 GB |
| 1.5B | LoRA | ~14 GB | ~16 GB |
| 3B | LoRA | ~22 GB | ~24 GB |
| 7B | LoRA | 需要 24GB+ 或 ZeRO-2/3 | 建议 `bf16-full` + `chunk_ctx 512` |

### 实测速度

CPU（Windows，8 线程）：

| 场景 | 配置 | 速度 |
|---|---|---|
| 0.1B `check-model` | fp32，3×256 token 校准 + 24 token 生成 | 14 秒 |
| 0.1B LoRA | ctx 64，bsz 1，r=16，fp32 | ≈70 tok/s |
| 0.1B LoRA + infctx | ctx 128 / chunk 64，fp32 | ≈62 tok/s |
| 0.1B 自我认知过拟合 | sample，ctx 64，400 步 | 约 4.5 分钟 |

CPU 上一定要用 `precision: fp32`：bf16 参数的反向传播会慢约 75 倍。

GPU（24GB Ampere 级，0.1B，bf16）：

| 场景 | 配置 | 速度 | 显存 |
|---|---|---|---|
| LoRA | ctx 512 / bsz 4 | ≈4.6–5.0K tok/s | 5.8 GiB |
| LoRA | ctx 1024 / bsz 2 | ≈2.5–2.8K tok/s | 5.8 GiB |
| LoRA + infctx | ctx 4096 / chunk 512 / bsz 1 | ≈700 tok/s | 3.7 GiB |
| State | ctx 512 / bsz 4 | ≈5.1–5.6K tok/s | 5.3 GiB |
| fp16 / DeepSpeed ZeRO-2 | ctx 512 / bsz 2 | ≈2.3–2.5K tok/s | 3.4 GiB |

0.1B 在 GPU 上是启动开销受限（bsz 1→8 单步耗时几乎不变）；模型越大，
计算占比越高。完整分析见 [`docs/GPU_VERIFICATION.md`](docs/GPU_VERIFICATION.md)。

显存不够时，按顺序尝试：

1. 降 `micro_batch_size`，同时把 `grad_accum` 调大（总 batch 不变）
2. 确认 `grad_checkpoint: true`
3. `chunk_ctx` 从 512 降到 256（infctx）
4. `precision: bf16-full`（参数本身用 bf16）
5. 多卡 + DDP / DeepSpeed

---

## 常见问题

**Q：提示 `CUDA out of memory`？**
见上面的显存表，先降 batch 或 chunk_ctx。

**Q：loss 不下降 / 一开始就是 ln(vocab)？**
- 先跑 `inspect-data`，确认 `【】` 只包住 assistant 回复；
- 学习率太大或太小：LoRA 从 1e-4 试起，State 从 5e-3 试起；
- 检查 `model.decay`：默认 `auto` 会在启动时打印选中的公式和三个候选的 loss。

**Q：过拟合后对所有问题都回答同一句？**
多半是推理 prompt 的 token 边界与训练不一致（最常见是 `"Assistant: "` 的尾空格）。
`rwkv-easy generate` 已经用 `ChatTemplate.generation_prompt()` 处理；自己写推理
代码时请调用同一个方法，并保持模板与训练时一致。

**Q：能不能不装 DeepSpeed？**
可以。单卡完全不需要；多卡用 DDP（`dist.mode: ddp` 或 `auto`）。
只有想要 ZeRO 分片 / CPU offload 时才需要 DeepSpeed。

**Q：能用在 AMD 显卡上吗？**
可以。纯 PyTorch WKV 后端不依赖 CUDA 专用代码，ROCm 版 PyTorch 会把设备识别为 `cuda`。
装了 `rwkv-fla` 也可以试 Triton 后端。

**Q：训练中断了怎么办？**
用 `--resume runs/xxx`；权重、优化器状态、学习率进度都会恢复。

**Q：合并后的模型怎么用？**
`merged.pth` 是标准 RWKV-7 state_dict，可以直接用官方推理代码 / `rwkv` 包加载。

更多问题见 [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)。

---

## 项目结构

```
RWKV-EASY/
├── rwkv_easy/
│   ├── cli.py              # 所有命令
│   ├── config.py           # YAML 配置 + 校验
│   ├── launcher.py         # 多进程启动
│   ├── lora.py             # 自研 LoRA
│   ├── state_tuning.py     # State Tuning
│   ├── backends/           # WKV 算子（native / reference / fla）
│   ├── models/             # RWKV-7 模型与 checkpoint 加载
│   ├── data/               # 分词器 / ShareGPT / 数据集
│   ├── train/              # 训练循环 / DDP / DeepSpeed / 保存
│   └── assets/             # 内置词表、校准文本、示例数据
├── configs/                # 配置模板（含自我认知过拟合 demo）
├── docs/                   # 详细文档
├── tests/                  # 纯 CPU 测试（含 2 进程 DDP）
├── examples/               # 示例数据（ShareGPT、自我认知实验数据）
├── CHANGELOG.md
├── install.sh / .bat / .ps1
└── train.sh / .bat
```

---

## 进一步阅读

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)：数据流、WKV 数学、infctx 与状态传递
- [`docs/CONFIG.md`](docs/CONFIG.md)：每一个配置项的含义
- [`docs/DATASET.md`](docs/DATASET.md)：ShareGPT、模板、completion-only mask、packed vs sample
- [`docs/CPU_VERIFICATION.md`](docs/CPU_VERIFICATION.md)：CPU 能验证什么、怎么复现、实测数字
- [`docs/MODEL_CHECK.md`](docs/MODEL_CHECK.md)：新权重 5 分钟验收清单与异常对照
- [`docs/GPU_VERIFICATION.md`](docs/GPU_VERIFICATION.md)：GPU 实测数字、瓶颈分析与 rwkv-fla 兼容性
- [`docs/EXTENDING.md`](docs/EXTENDING.md)：接入自定义 WKV 内核（CUDA/Triton/ROCm）、新模型版本、新数据格式
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)：报错对照表
- [`CHANGELOG.md`](CHANGELOG.md)：版本与本次验证/修复记录

---

## 开发与测试

```bash
pip install -e ".[dev]"
python -m pytest tests -q            # 全部纯 CPU
python -m rwkv_easy selftest --full  # 8 种组合
python -m ruff check rwkv_easy tests
```

---

## License

Apache-2.0。RWKV-7 模型结构与词表来自
[BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)（Apache-2.0），
详见 [`NOTICE`](NOTICE)。
