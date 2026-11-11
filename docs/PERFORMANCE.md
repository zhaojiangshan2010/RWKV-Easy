# 性能调优指南

这份文档回答三个问题：

1. 这台机器上应该开哪些开关？
2. 为什么某个设置更快/更慢？
3. 换模型规模或换卡之后，怎么自己找最佳点？

所有数字都在 **0.1B（`rwkv7-g1d-0.1b`）+ 24GB Ampere 级 GPU
（`B1.gpu.large`，82 SM，cc 8.6）** 上实测；设备 bf16 matmul 峰值
62–74 TFLOPS，fp32 无 TF32 约 24 TFLOPS。详细 profile 与实验记录见
[`GPU_VERIFICATION.md`](GPU_VERIFICATION.md)。

---

## 1. 一句话结论

| 场景 | 推荐设置 |
|---|---|
| CUDA，追求最快 | `model.backend: fla` + `precision: bf16-full` + `train.compile: true` |
| 没有 rwkv-fla | `model.backend: native` + `bf16-full` + `compile: true`（只编译 WKV 单块） |
| 长上下文 | `infctx: true` + `chunk_ctx: 2048` + `infctx_backward: true` + `grad_checkpoint: false` |
| CPU / 调试 | `precision: fp32`，`backend: native`，不要开 compile |

0.1B / ctx512 的实测：

| 配置 | tok/s | 峰值显存 |
|---|---|---|
| fla + bf16-full + compile（bsz32） | **48.5K** | 11.9 GiB |
| fla + bf16-full eager（bsz32） | 40.3K | 12.2 GiB |
| native + bf16-full + compile（bsz24） | **24.3K** | 16.9 GiB |
| native + bf16-full eager（bsz24） | 20.6K | 17.6 GiB |
| fla + State + bf16-full + reduce-overhead（bsz32） | **56.3K** | 9.2 GiB |
| fla + infctx 8K / chunk2048 / bsz2 + compile | **37.0K** | 3.4 GiB |
| fla + infctx 8K / chunk2048 / bsz10 + compile | **41.7K** | 15.2 GiB |
| fla + infctx 32K / chunk2048 / bsz2 | **33.6K** | 3.4 GiB |
| native + infctx 8K / chunk2048 / bsz2 + compile | 5.1K | 5.9 GiB |

> 不同 `ctx_len` / `chunk_ctx` / `bsz` 的绝对 tok/s 不可直接比较；
> 比较时至少固定其中两个。

---

## 2. 六个有效旋钮（按收益排序）

### 2.1 `precision: bf16-full`（最大单项收益）

基座权重直接以 bf16 加载，省掉 autocast 每次 Linear 的 fp32→bf16 权重转换。

**关键点：LoRA 的 A/B 和 State Tuning 的 `time_state` 是在模型 cast 之后创建的，
仍然是 fp32。** 所以 `bf16-full` 只影响冻结基座，训练精度不受影响。

0.1B / ctx512 / bsz16：23.8K → 34.8K tok/s（+46%），显存 14.6 → 8.9 GiB。

### 2.2 `model.backend: fla`（CUDA 上必开）

rwkv-fla 的融合内核把 WKV 的数百个小算子合成少量 Triton kernel。0.1B：

- 整段 LoRA：native 24.3K → fla 48.5K；
- infctx：native 5.1K → fla 41.7K（长上下文差距更大）。

fla 支持 `clampw` / `softplus` / `exp` 三种 decay，官方权重直接保持 `auto`
或 `clampw` 即可。建议 `precision: bf16` 或 `bf16-full`：它的内核内部按
bf16 累加，纯 fp32 输入会提示“内核不支持 fp32”。

没有 `rwkv-fla`（或版本不兼容）时用 native；native 在 CPU 上也能跑。

### 2.3 `train.compile: true`（按后端选择编译对象）

| 后端 | 编译什么 | 实测 |
|---|---|---|
| fla | `forward_hidden`（不含 lm_head） | 40.3K → 48.5K（+20%） |
| native | 只编译 WKV 的 `_chunk` 单块图 | 20.6K → 24.3K（+18%，首次约 25s） |

- 不要试图编译整个 native model：12 层 × N 块全部展开，首次编译可能 >9 分钟。
- `compile_mode: reduce-overhead` 可以进一步降显存（State 9.2 GiB）；
  `max-autotune` 再快约 2%，但首次编译约 8 分钟。
- Windows 上首次编译可能较慢；报错就关掉。

### 2.4 加大 `micro_batch_size`

0.1B 的每层矩阵只有 768×768，batch 越大，tensor core 利用率越高：

- fla ctx512：bsz4 → bsz32，17.4K → 40.3K（eager）；
- bsz32 之后开始饱和（bsz40/48 反而略慢）；
- 目标是让每步 **8K–20K token**（0.1B）；更大的模型按显存相应降低。

显存不够就降 batch、加 `grad_accum`，保持总 batch 不变。

### 2.5 infctx：`infctx_backward` + `chunk_ctx: 2048`

- `infctx_backward: true`（默认）：每个 chunk 立即 backward，显存真正
  `O(chunk_ctx)`；8K ctx 从 12.4 GiB 降到 **3.4 GiB**，0.1B 可训 32K。
- `chunk_ctx: 2048`：把串行块数从 32 降到 4，8K ctx 从 1.37K（chunk256）
  提升到 34.6K（eager）/ 37.0K（compile）。
- `grad_checkpoint: false`：per-chunk backward 已经限制了激活，
  0.1B 实测关掉后 23.7K → 34.6K；7B 级模型再按显存打开。
- DDP / DeepSpeed 暂不支持 per-chunk backward，会自动回退到“最后统一
  backward”，显存随 `ctx_len` 增长；长上下文请先用单卡验证。

### 2.6 只对可训练 token 过 lm_head

SFT 里约一半 token 是 prompt（`label=-100`），不需要 65536 维 logits。
`Trainer` 只 gather 可训练行过 `lm_head`：+9%，并显著降低 logits 显存。
这是默认行为，无需配置。

---

## 3. 显存模型

### 整段训练（`infctx: false`）

峰值 ≈ 基座权重 + LoRA/State 参数 + 当前 batch 的激活 + logits。

- `bf16-full`：基座 bf16；0.1B 约 0.38 GB；
- logits 只覆盖可训练 token（约 50%），bsz32×512 时约 1 GB；
- `grad_checkpoint: true` 用时间换显存，约省 2–3 倍激活。

### infctx（`infctx_backward: true`）

峰值只与 `chunk_ctx`、`micro_batch_size` 有关，与 `ctx_len` 无关：

```
显存 ≈ 基座 + 状态 + chunk_ctx × bsz 的激活 + 可训练 token 的 logits
```

0.1B / chunk2048：

| bsz | 峰值显存 |
|---|---|
| 2 | 3.4 GiB |
| 10 | 15.2 GiB |
| 12 | 18.2 GiB |

32K ctx、bsz2 仍然是 3.4 GiB。

---

## 4. 按模型规模的建议

以下是 0.1B 实测 + 线性外推的起点，**不是保证值**；先小步试显存。

| 模型 | 建议 |
|---|---|
| 0.1B / 0.4B | fla + bf16-full + compile；ctx512 / bsz 按显存加到 8K–16K token/step |
| 1.5B / 3B | fla + bf16-full + compile；ctx512–1024，bsz 2–8，按需 `grad_checkpoint` |
| 7B | bf16-full + fla；ctx512–1024，bsz1–2，`grad_checkpoint: true`，多卡 ZeRO-2/3 |
| 长上下文 | infctx chunk2048 + per-chunk backward；显存够就关 GC |
| CPU | native + fp32；只做自检，不追性能 |

模型越大，计算占比越高，0.1B 上的“启动开销”问题会自然消失；fla 与 compile
仍然建议开启。

---

## 5. 怎么自己 benchmark

用真实训练日志里的 tok/s（已排除启动时间），固定除一个变量外的所有配置：

```bash
# 基线
rwkv-easy train -c configs/lora_sharegpt_fla.yaml \
  --set model.path=/path/to/model.pth --set data.path=/path/to/data.json \
  --set train.max_steps=20 --set train.log_every=5 \
  --set train.micro_batch_size=32 --set train.ctx_len=512

# 单变量：关 compile
... --set train.compile=false

# 单变量：换 native
... --set model.backend=native --set train.compile=true

# infctx
rwkv-easy train -c configs/lora_sharegpt_infctx.yaml \
  --set train.ctx_len=8192 --set train.chunk_ctx=2048 \
  --set train.micro_batch_size=2 --set train.max_steps=6
```

看日志最后几个 step 的 tok/s，不要用 step 1（包含 compile/warmup）。
`peak_memory` 也在同一行。

### 常见误判

1. **拿不同 ctx/bsz 的 tok/s 比**：每步 token 数差几倍，结论会完全反过来。
2. **用 step 1 的 tok/s**：compile / Triton 首次编译会把它拉到几十 tok/s。
3. **只关 L2Wrap 追求速度**：收益约 2–3%，但少了一个稳定性项；现在 L2Wrap
   已经是原地梯度，默认保留即可。
4. **在 native 上开整模型 compile**：会编译十几分钟，不是卡死。
5. **以为 `bf16-full` 会把 LoRA 也变成 bf16**：不会；适配器仍是 fp32。

---

## 6. 当前天花板

0.1B 上 fla + compile 已经到 **55.6 TFLOPS（LoRA）/ 64.5 TFLOPS（State）**，
设备峰值 62–74 TFLOPS。剩余差距主要来自：

- RWKV 结构自带的 elementwise（sigmoid/tanh/门控）；
- lm_head + cross-entropy；
- 0.1B 的 768×768 小矩阵 tensor core 利用率。

继续提升需要自定义 CE/融合 kernel，收益/风险比不高。已试过并否决的方案
（native bf16 matmul、LoRA 权重合并、head+CE 单独编译、TF32、chunk128 等）
记录在 [`GPU_VERIFICATION.md`](GPU_VERIFICATION.md) 第 3.3 节和
[`../CHANGELOG.md`](../CHANGELOG.md)。

---

## 7. 相关文档

- 全部字段含义：[`CONFIG.md`](CONFIG.md)
- 实测记录、profile、复现命令：[`GPU_VERIFICATION.md`](GPU_VERIFICATION.md)
- CPU 验证矩阵：[`CPU_VERIFICATION.md`](CPU_VERIFICATION.md)
- 报错对照：[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
