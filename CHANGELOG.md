# 变更记录

## 0.2.0 — 2026-11-12

性能与 infctx 架构升级：0.1B 在 24GB 卡上的整段 LoRA 从 13.7–16.1K 提升到
**37.0K tok/s**，infctx 真正做成 `O(chunk_ctx)` 显存。

### 新增

- **`train.infctx_backward`（默认 true）**：每个 chunk 立即 backward，
  只保留 detach 后的循环状态；8K ctx 显存 12.4 → 4.7 GiB，0.1B 可训
  32K 上下文。DDP/DeepSpeed/`infctx_detach=false` 自动回退旧路径。
- **`precision: bf16-full` 成为 GPU 模板默认**：基座 bf16、LoRA/time_state
  仍为 fp32；0.1B ctx512/bsz16 吞吐 +46%，显存 14.6 → 8.9 GiB。
- 新增 `tests/test_infctx_backward.py`、`tests/test_l2wrap.py`；
  测试总数 108。

### 性能

- fla LoRA：37.0K tok/s（ctx512/bsz32，15.4 GiB）；State：43.0K（12.8 GiB）。
- native LoRA：20.1–20.3K tok/s（bsz24–26，wkv_chunk 64），比最初 3.9K 提升 5 倍。
- infctx：ctx8192/chunk2048/bsz10 37.0K；chunk4096/bsz4 34.9K；
  ctx32768/chunk2048/bsz2 32.1K（4.7 GiB）。
- State + infctx：66.5K tok/s（4.0 GiB，只有第一块需要反传）。
- L2Wrap 改为在 CE 梯度上原地 `scatter_add`，省掉每步 2 GiB 全量梯度；
  闭包只捕获 max_value/indices，修复引用环泄漏。
- native 缓存下三角掩码，`solve_triangular` 用 `unitriangular=True`。

### 试过但没有保留

- `torch.compile`：整段训练没有稳定收益（GPU-bound）。
- bf16 交叉熵 / bf16 GroupNorm：没有收益。
- native 合并 4 个块内 matmul + 4D 布局：算子级略快，但显存更高、大 batch
  下 OOM，已回退。
- TF32、native chunk 128：没有收益或直接 OOM。

## 0.1.1 — 2026-11-11

第二次 GPU 验证（torch 2.14.1 + CUDA 13.0 + triton 3.8.0）。这一版的主题是
**rwkv-fla 后端正式可用**。

### 新增

- `model.backend: fla` 通过验证：支持 clampw / softplus / exp 三种 decay，
  不再强制 softplus；官方权重可以直接用 `clampw`。
- 新增 `configs/lora_sharegpt_fla.yaml`（CUDA + Triton 加速模板）。
- 新增 `tests/test_fla_backend.py`：与 fp64 参考对照、补齐路径、分块状态、
  clampw、bf16 反传；没有 CUDA/rwkv-fla 时自动跳过。
- launcher：请求进程数超过可见 GPU 数时提前报错；任一 rank 早退时立即
  终止其它进程，不再空等 10 分钟。

### 修复

- **fla 参数语义**：`chunk_rwkv7` 的 `w` 是原始 decay 参数（内部再算
  `-exp(w)`），必须把 log-decay 传给 `log_w`。此前误传 `w` 时 loss 仍会下降，
  但数值与参考不一致。
- **fla 补齐位**：长度不是 64 的倍数时用 `log_decay = 0`（不是 -60）+
  全零 k/v/a/b，否则携带状态会被衰减成 0。
- **fla dtype**：统一混合精度下各输入的 dtype，修复
  `Both operands must be same dtype. Got bf16 and fp32`。
- **native 动态分块**：softplus/exp 从递归二分改为一次 reduce 估算安全块长，
  去掉每层 `.item()` 的 GPU 同步；真实权重下仍受衰减限制，但不再指数级变慢。
- **DDP 测试**：显式 `--device cpu`，修复单卡 GPU 机器上 `num_gpus=2`
  导致 rank 1 在 `set_device(1)` 崩溃、rank 0 空等的问题。
- `init_process_group` 加 300 秒超时；`LOCAL_RANK` 超出可见 GPU 数时给出
  明确错误。

### 验证

- 103 个测试通过（GPU 上含 5 个 fla 对照；CPU 上跳过）。
- 0.1B / ctx512 / bsz4 / bf16：native clampw 约 3.9K tok/s（5.8 GiB），
  fla 约 13.7–16.1K tok/s（4.4 GiB），loss 曲线基本重合。
- 同配置下 fla 全面更快：State ctx512/bsz4 4.35K → 22.95K；
  infctx ctx1024/chunk256 470 → 1.37K。不同 `ctx_len`/`bsz` 的绝对 tok/s
  不可直接比较（每步 token 越少，固定开销占比越高）。
- fla + infctx、fla + State、fla + DeepSpeed ZeRO-2 均跑通。
- fla 训练出的 State 适配器合并后，`generate` 正确回答 `RWKV 0.1B 大模型`。

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
