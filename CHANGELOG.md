# 变更记录

## 0.1.0 — 2026-11-11

首个可用版本。

### 功能

- RWKV-7（x070）训练：**LoRA** / **State Tuning**
- 两种方法都可以叠加 **infctx** 分块长上下文
- 数据：ShareGPT / OpenAI messages / 两列指令 / 纯文本，**completion-only loss**
- 样本组织：`packed`（定长流，兼容 infctx）/ `sample`（动态 padding），tokenize 缓存
- 精度：FP32 / FP16 / BF16 / BF16-full
- 并行：单卡 / PyTorch DDP / DeepSpeed（可选，不装也能多卡）
- WKV 后端：纯 PyTorch 分块实现（CPU/CUDA/ROCm 通用）+ fp64 reference 对照 +
  可选 `rwkv-fla`（Triton）
- **decay 公式自动校准**（`clampw` / `softplus` / `exp`）
- CLI：`init` / `train` / `merge` / `inspect-data` / `check-model` / `doctor` /
  `selftest` / `generate`
- 一键安装与训练脚本（Linux / Windows），中文文档与配置注释

### 修复（真实权重验收中发现）

- 输出门控低秩维度 `g1/g2` 的官方公式是 `0.6 * C^0.8`（此前误用 `0.6 * C^0.4`，
  会导致真实 checkpoint 形状不匹配）
- `precision: fp32` 未把 bf16 checkpoint 显式上转，CPU 反向传播慢约 75 倍
  （实测 42s vs 0.57s / step）
- `max_steps` 被 `epochs` 截断（日志写 20 步、实际 9 步）
- 生成 prompt 与训练文本的 token 边界不一致：`"Assistant: "` 的尾空格会被分词
  并入回答首 token，推理时必须用 `Assistant:` 结尾（`ChatTemplate.generation_prompt`）

### 验证

- 98 个纯 CPU 测试（含 fp64 数值对照、infctx 等价性、2 进程 DDP）
- `selftest --full`：lora/state × infctx on/off × fp32/bf16 共 8 组
- 真实权重 `rwkv7-g1d-0.1b`：decay=clampw（NLL 2.54），续写连贯；
  LoRA 20 步、infctx 8 步、合并等价性（logits 误差 7.8e-5）
- 自我认知过拟合 demo：400 步 EMA 3.75→0.087，10/10 命中（含未见问法）

### GPU 验证（2026-11-11，24GB Ampere 级）

- 真实 0.1B：`check-model`、LoRA、State、infctx、合并、fp16、DeepSpeed ZeRO-2 全部跑通
- LoRA ctx512/bsz4 ≈4.8K tok/s（5.8GiB）；infctx ctx4096/chunk512 ≈700 tok/s（3.7GiB）
- 0.1B 是启动开销受限：bsz 1→8 单步 365→378ms，吞吐 1.4K→10.8K tok/s
- 自我认知 demo 在 GPU 上 400 步 60 秒，8/8 命中

### 修复（GPU 验证中发现）

- `torch < 2.4` 的 `torch.load(mmap=True)` 只接受 `str`，传 `Path` 报错
- `torch 2.2` 没有 `torch.amp.GradScaler`，fp16 路径加兼容层
- 在 `model.to(device)` 之后注入 LoRA 时 A/B 建在 CPU，设备不匹配
- `--set train.lr=2e-4` 被 YAML 1.1 解析成字符串
- `--set dist.deepspeed.zero_stage` 在 YAML 缺该段时直接报错
- DeepSpeed 单进程缺进程组（去探测 MPI）、缺 `LOCAL_RANK`
- DeepSpeed `train_batch_size` 与手动梯度累积冲突
- clampw 模式下 WKV 逐块 `.item()` 同步，改为静态分块

### 尚未验证

- rwkv-fla / Triton 后端（测试环境 torch 2.2 + triton 2.2 与 rwkv-fla 0.7 不兼容）
- NCCL 多卡（当前只有单卡）
