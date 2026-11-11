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

- 96 个纯 CPU 测试（含 fp64 数值对照、infctx 等价性、2 进程 DDP）
- `selftest --full`：lora/state × infctx on/off × fp32/bf16 共 8 组
- 真实权重 `rwkv7-g1d-0.1b`：decay=clampw（NLL 2.54），续写连贯；
  LoRA 20 步、infctx 8 步、合并等价性（logits 误差 7.8e-5）
- 自我认知过拟合 demo：400 步 EMA 3.75→0.087，10/10 命中（含未见问法）

### 尚未验证

- CUDA 性能/显存、fp16 + GradScaler、DeepSpeed 实跑、fla/Triton、NCCL 多卡
