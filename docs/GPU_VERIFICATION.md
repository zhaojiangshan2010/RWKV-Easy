# GPU 验证与性能记录

环境（2026-11-12）：虚拟化 `B1.gpu.large`（23.26 GiB，compute capability 8.6，
82 个 SM，Ampere 级）、Linux、Python 3.11.8、**torch 2.14.1 + cu130**、
CUDA 13.0、cuDNN 9.24、**triton 3.8.0**、DeepSpeed 0.19.8、
**rwkv-fla 0.7.202508221413**。

模型：`rwkv7-g1d-0.1b-20260129-ctx8192.pth`（12 层 / 768 维 / 191.08M，bf16）。

设备实测峰值（`torch.matmul`，30 次平均）：

| 尺寸 | bf16 | fp32（无 TF32） | fp32（TF32） |
|---|---|---|---|
| 2048³ | 62.1 TFLOPS | 23.9 TFLOPS | 28.3 TFLOPS |
| 4096³ | 64.3 TFLOPS | 23.8 TFLOPS | 33.2 TFLOPS |
| 8192³ | 73.7 TFLOPS | — | — |

---

## 1. 结论

- **fla 整段 LoRA：48.5K tok/s（ctx512/bsz32，compile，11.9 GiB）**；
  State：56.3K tok/s（reduce-overhead，9.2 GiB）。约 56–65 TFLOPS，
  为设备 bf16 峰值的 75–90%。
- **native 整段 LoRA：20.6K tok/s**（ctx512/bsz24）。比最初的 3.9K 提升 5 倍，
  主要来自 `bf16-full` + 大 batch + 选择性 head；与 fla 的差距约 2.3 倍。
- **infctx 真正做到显存 ∝ chunk_ctx**：每个 chunk 立即 backward，
  8K ctx 从 12.4 GiB 降到 **3.4 GiB**；0.1B 在 24GB 卡上可以训
  **32K 上下文（3.4 GiB，33.6K tok/s）**。
- infctx 大 batch 下接近整段训练：ctx8192 / chunk2048 / bsz10 / compile 达到
  **41.7K tok/s（15.2 GiB）**。
- State + infctx 由于只有第一块需要反传，达到 **72.6K tok/s（2.7 GiB）**。
- native + softplus 在真实权重上仍只有约 130 tok/s：log-decay 在 -2～-7，
  块长会被压到 8 左右。CUDA + softplus 请直接用 fla。
- 全量测试 **110 个**（GPU 上全跑，CPU 上 5 个 fla 测试跳过）。

---

## 2. 实测结果

`tok/s` 是日志里的吞吐；峰值显存为 `torch.cuda.max_memory_allocated`。

### 2.1 整段训练（ctx 512，bf16-full）

| 场景 | 后端 | 配置 | 速度 | 峰值显存 |
|---|---|---|---|---|
| LoRA | **fla** / clampw | bsz 32，compile | **48.5K tok/s** | 11.9 GiB |
| LoRA | fla / clampw | bsz 32，eager | 40.3K tok/s | 12.2 GiB |
| LoRA | fla / clampw | bsz 48，compile | 45.5K tok/s | 17.6 GiB |
| LoRA | native / clampw | bsz 24，wkv_chunk 64 | 20.6K tok/s | 17.6 GiB |
| LoRA | native / clampw | bsz 16，wkv_chunk 96 | 19.0K tok/s | 15.0 GiB |
| State | **fla** / clampw | bsz 32，reduce-overhead | **56.3K tok/s** | 9.2 GiB |
| State | fla / clampw | bsz 32，eager | 47.7K tok/s | 9.6 GiB |

### 2.2 infctx（bf16-full，fla，per-chunk backward，无梯度检查点）

| 场景 | 配置 | 速度 | 峰值显存 |
|---|---|---|---|
| LoRA + infctx | ctx 8192 / chunk 2048 / bsz 2 / compile | 37.0K tok/s | 3.4 GiB |
| LoRA + infctx | ctx 8192 / chunk 2048 / bsz 2 / eager | 34.6K tok/s | 3.4 GiB |
| LoRA + infctx | ctx 8192 / chunk 2048 / bsz 10 / compile | **41.7K tok/s** | 15.2 GiB |
| LoRA + infctx | ctx 8192 / chunk 2048 / bsz 10 / eager | 40.5K tok/s | 15.2 GiB |
| LoRA + infctx | ctx 32768 / chunk 2048 / bsz 2 | 33.6K tok/s | 3.4 GiB |
| LoRA + infctx | ctx 8192 / chunk 2048 / bsz 2，梯度检查点 | 23.7K tok/s | 3.9 GiB |
| State + infctx | ctx 8192 / chunk 2048 / bsz 2 | **72.6K tok/s** | 2.7 GiB |
| LoRA + infctx（旧路径） | ctx 8192 / chunk 2048 / bsz 2 | 31.8K tok/s | 12.4 GiB |
| LoRA + infctx（native） | ctx 8192 / chunk 2048 / bsz 2 | 2.3K tok/s | 7.5 GiB |

### 2.3 其它

| 场景 | 配置 | 结果 |
|---|---|---|
| `check-model` | native，bf16，3×256 token 校准 + 32 token 生成 | 22.8 秒 |
| fp16 混合精度 | native，ctx512/bsz2 | ≈2.5K tok/s，3.4 GiB |
| DeepSpeed ZeRO-2（单卡） | native，ctx512/bsz2，bf16 | ≈2.3K tok/s，3.4 GiB |
| DeepSpeed ZeRO-2（单卡） | fla，ctx512/bsz8，compile | 28.6K tok/s，3.7 GiB |
| DeepSpeed ZeRO-2（单卡） | fla，sample ctx64/bsz2 | 首次编译 ~42s，之后正常，1.5 GiB |
| 自我认知过拟合 | native，sample ctx64/bsz2，20 步 | 秒级，loss 5.25 → 0.001 |
| 全量测试（CPU + GPU） | 110 个测试 | 约 80–85 秒 |

> **只看同配置对比**：不同 `ctx_len` / `chunk_ctx` / `micro_batch_size` 的绝对
> tok/s 不可直接比较。早期表格里 State 的 fla 行用了 `sample ctx64/bsz2`
> （600 tok/s），native 行用了 `ctx512/bsz4`（5.3K），看起来像 fla 慢 9 倍；
> 同配置下 fla 实际更快。

---

## 3. 瓶颈与优化过程

### 3.1 基线

最初（`bf16` + fp32 参数 + autocast）：native ctx512/bsz4 约 3.9K tok/s，
fla 约 13.7–16.1K。`torch.profiler` 显示单步 GPU kernel 里：

```
elementwise        ~50%
aten::mm           ~26%
aten::copy_        ~17%
fla WKV 内核       ~9%
```

其中 `copy_` 很大一部分来自 **autocast 每次 Linear 都把 fp32 权重转成 bf16**，
以及 LoRA 的 A/B 每次 forward 都做 fp32→bf16 转换。

### 3.2 已采纳的优化

| 改动 | 效果（0.1B，ctx512） |
|---|---|
| `precision: bf16-full`（基座 bf16，适配器 fp32） | bsz16：23.8K → 34.8K tok/s；显存 14.6 → 8.9 GiB |
| fla 后端 + 大 batch | bsz4→32：17.4K → 40.3K tok/s（eager） |
| 只对可训练 token 过 lm_head | 40.3K 前后再 +9%，logits 显存明显下降 |
| `train.compile` 编译 `forward_hidden` | 40.3K → **48.5K tok/s** |
| `compile_mode: reduce-overhead`（State） | 47.7K → **56.3K tok/s**，9.2 GiB |
| infctx 每块即时 backward | 8K ctx：12.4 → 3.4 GiB，可训 32K |
| infctx 关闭梯度检查点 | 8K ctx：23.7K → 34.6K tok/s |
| L2Wrap 改成在 CE 已有梯度上原地 scatter_add | +2.5%，每步少 2 GiB 全量梯度 |
| native 缓存三角掩码 + `unitriangular=True` | chunk96：18.65K → 18.98K tok/s |
| native 大 batch（bsz24） | 约 20.6K tok/s |

### 3.3 试过但没有保留

- bf16 交叉熵 / bf16 GroupNorm：没有收益，GroupNorm 反而略慢。
- native 合并 4 个块内 matmul + 保持 4D 布局：算子级略快，但中间量更占显存，
  大 batch 下 OOM，真实训练反而不如旧内核，已回退。
- TF32：native 实测 18.65K vs 18.73K，差异在噪声内，没有启用。
- native chunk 128：bsz24 直接 OOM，没有收益。
- native + `torch.compile`：首次编译超过 9 分钟仍未完成（WKV 里大量小算子
  和三角求解），不适合作为默认路径。
- L2Wrap 直接关闭：能到 37.4K（当时未 compile），但会丢掉稳定性项；
  最终保留 L2Wrap，用原地梯度实现拿回大部分收益。

### 3.4 当前天花板

fla + compile 整段 48.5K tok/s ≈ 55.6 TFLOPS，State 56.3K ≈ 64.5 TFLOPS。
设备 bf16 峰值 62–74 TFLOPS，剩余差距主要来自：

- RWKV 结构本身大量 elementwise（sigmoid/tanh/门控），compile 已融合一部分；
- cross-entropy + logits 仍然是大头，已通过“只算可训练 token”削掉约一半；
- 0.1B 的每层矩阵较小（768×768），tensor core 利用率不如大模型。

继续增大 batch 在 bsz32–48 已经饱和；再往上需要更激进的 logits/CE 融合或
自定义 kernel，收益/风险比不高。

---

## 4. rwkv-fla 语义验证

### 4.1 三个必踩的坑

1. **`w` 不是 log-decay**：`chunk_rwkv7(w=...)` 内部会再算 `log_w = -exp(w)`。
   本项目已经算好 log-decay，必须传 `log_w=`。误传 `w` 时 loss 仍会下降，
   但输出与 fp64 参考差 O(10)。
2. **补齐位不能强衰减**：长度不是 64 的倍数时，尾部用 `log_decay = 0`
   + 全零 k/v/a/b。用 `-60` 会把携带的状态乘成 0。
3. **混合 dtype**：bf16 autocast 下 `k * k_k` 会被 fp32 参数提升成 fp32，
   而 value 是 bf16，Triton 的 `tl.dot` 会报
   `Both operands must be same dtype`；适配层已统一到激活 dtype。

### 4.2 数值精度（相对 float64 逐 token 参考）

| 输入 dtype | 后端 | 输出 rel-L2 | 状态 rel-L2 |
|---|---|---|---|
| fp32 | native | 2.8e-07 | 3.9e-07 |
| fp32 | fla | 7.9e-04 | 7.5e-04 |
| bf16 | native | 5.7e-05 | — |
| bf16 | fla | 3.3e-03 | 1.7e-03 |

梯度相对误差：native ~3e-07，fla 约 1–2e-3。fla 的 fp32 输入也会收到
`ChunkDeltaRuleFunction does not support float32` 警告，因为内核内部按 bf16
累加；用 fla 时保持 bf16 / bf16-full。

### 4.3 decay 口径

fla 源码里没有“log-decay 必须 ≤ -0.5”的硬限制。实测 `clampw`
（log-decay ∈ [-0.6066, 0]）、`softplus`、以及极弱衰减（w 接近 0）
都能与 fp64 参考对齐（rel-L2 ~7e-4）。因此配置不再强制
`backend: fla` → `decay: softplus`；官方权重的 `clampw` 可以直接用。

---

## 5. 本次修复/改进清单

- **fla**：`w` → `log_w`；补齐位 `-60` → `0`；统一混合精度 dtype。
- **infctx**：每个 chunk 立即 backward（`train.infctx_backward`），
  显存与 `ctx_len` 解耦；DDP/DeepSpeed/`infctx_detach=false` 自动回退旧路径。
- **只对可训练 token 过 lm_head**：SFT 约一半 prompt token 不再产生 65536 维
  logits，+9% 且显著省显存。
- **修正 `train.compile`**：之前编译的是整个 model，但训练前向一直没走编译
  路径；现在直接编译 `forward_hidden` / `forward_chunk_hidden`，fla 下 +20%。
- **L2Wrap**：改为在 CE 梯度上原地 `scatter_add`，闭包只捕获小的
  max_value/indices，避免每步 2 GiB 分配和引用环泄漏。
- **native**：缓存下三角掩码，`solve_triangular` 用 `unitriangular=True`
  省掉 eye 与对角减法。
- **launcher**：请求进程数超过可见 GPU 数时提前报错；rank 早退时终止其它进程。
- **进程组**：`init_process_group` 加 300 秒超时；`LOCAL_RANK` 越界给出明确错误。
- **DDP 测试**：显式 `--device cpu`，修复单卡 GPU 主机上的误失败。
- **torch < 2.4 兼容**：`torch.load(mmap=True)` 只接受 `str`；
  `torch.amp.GradScaler` 不存在，已有兼容层。

---

## 6. 复现命令

```bash
# 环境
python -m venv .venv && . .venv/bin/activate
pip install torch pyyaml && pip install -e .
pip install rwkv-fla deepspeed          # 可选
rwkv-easy doctor

# 模型验收
rwkv-easy check-model -m /path/to/rwkv7.pth --dtype bf16

# fla LoRA（CUDA，推荐）
rwkv-easy train -c configs/lora_sharegpt_fla.yaml \
  --set model.path=/path/to/model.pth --set data.path=/path/to/data.json \
  --set train.max_steps=100 --set train.micro_batch_size=32 --set train.ctx_len=512

# native LoRA
rwkv-easy train -c configs/lora_sharegpt.yaml \
  --set model.path=/path/to/model.pth --set data.path=/path/to/data.json \
  --set train.max_steps=100 --set train.micro_batch_size=24 --set train.ctx_len=512

# infctx（每块即时反传，32K 上下文）
rwkv-easy train -c configs/lora_sharegpt_infctx.yaml \
  --set model.path=/path/to/model.pth --set data.path=/path/to/data.json \
  --set train.ctx_len=32768 --set train.chunk_ctx=2048 --set train.max_steps=30

# DeepSpeed ZeRO-2（单卡/多卡都行）
rwkv-easy train -c configs/lora_sharegpt.yaml \
  --set dist.mode=deepspeed --set dist.deepspeed.zero_stage=2 ...

# 全量测试（GPU 上会额外跑 5 个 fla 对照）
python -m pytest tests -o addopts="" -q
```
