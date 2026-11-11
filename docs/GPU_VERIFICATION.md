# GPU 验证记录

环境（2026-11-11，第二次 GPU 验证）：虚拟化 `B1.gpu.large`（23.3 GiB，
compute capability 8.6，Ampere 级）、Linux、Python 3.11.8、
**torch 2.14.1 + cu130**、CUDA 13.0、cuDNN 9.24、**triton 3.8.0**、
DeepSpeed 0.19.8、**rwkv-fla 0.7.202508221413**（这次可以正常导入）。

模型：`rwkv7-g1d-0.1b-20260129-ctx8192.pth`（12 层 / 768 维 / 191.08M，bf16）。

---

## 1. 结论

- 真实权重在 GPU 上：加载、decay 校准、生成、LoRA、State、infctx、合并、
  fp16、bf16-full、DeepSpeed ZeRO-2 **全部跑通**。
- **rwkv-fla 这次验证通过**：torch 2.14 / triton 3.8 不再有版本障碍。
  它支持官方权重的 `clampw`，不需要改成 softplus；同配置下全面快于 native：
  LoRA 3.9K → 13.7–16.1K、State 4.35K → 22.95K、infctx 470 → 1.37K tok/s，
  显存也更低。
- 速度对 0.1B 小模型仍是 **launch-overhead 限制**，但 fla 的融合内核把
  WKV 的数百个小算子消掉了，所以提升接近一个数量级；`sample ctx64/bsz2`
  这种每步只有几百 token 的场景，两边都被固定开销主导（~0.2s/step），
  不能拿它的 tok/s 去和 `ctx512/bsz4` 比。
- native + softplus 在真实权重上只有约 130 tok/s：这些权重的 log-decay
  在 -2～-7，块内累计衰减很大，native 必须把块长压到 8 左右。不是 bug，
  是 native 算法的保守边界；CUDA 上应改用 fla。

---

## 2. 实测结果

| 场景 | 后端 | 配置 | 速度 | 峰值显存 |
|---|---|---|---|---|
| `check-model` | native | bf16，3×256 token 校准 + 32 token 生成 | 22.8 秒 | — |
| LoRA（72 个线性层） | native / clampw | ctx 512，bsz 4，bf16，无梯度检查点 | ≈3.9–5.0K tok/s（≈0.5 s/step） | 5.8 GiB |
| LoRA | **fla** / clampw | ctx 512，bsz 4，bf16 | **≈13.7–16.1K tok/s** | 4.4 GiB |
| LoRA | native / clampw | ctx 1024，bsz 2，bf16 | ≈2.5–2.8K tok/s | 5.8 GiB |
| LoRA + infctx | native / clampw | ctx 1024 / chunk 256，bsz 1，梯度检查点 | ≈470 tok/s | 1.9 GiB |
| LoRA + infctx | **fla** / clampw | ctx 1024 / chunk 256，bsz 1，梯度检查点 | **≈1.37K tok/s** | 1.8 GiB |
| State Tuning | native / clampw | ctx 512，bsz 4，bf16 | ≈4.35K tok/s | 5.3 GiB |
| State Tuning | **fla** / clampw | ctx 512，bsz 4，bf16 | **≈22.95K tok/s** | 4.1 GiB |
| fp16 混合精度 | native | ctx 512，bsz 2 | ≈2.5K tok/s | 3.4 GiB |
| DeepSpeed ZeRO-2（单卡） | native | ctx 512，bsz 2，bf16 | ≈2.3K tok/s | 3.4 GiB |
| DeepSpeed ZeRO-2（单卡） | **fla** | sample ctx 64，bsz 2，bf16 | 首次编译 ~42s，之后正常 | 1.5 GiB |
| 自我认知过拟合 | native | sample ctx 64，bsz 2，20 步 | 秒级，loss 5.25 → 0.001 | 1.2 GiB |
| 全量测试（CPU + GPU） | — | 103 个测试 | 85 秒 | — |

`tok/s` 是日志里的含 padding token 吞吐；peak 为 `torch.cuda.max_memory_allocated`。

> **只看同配置对比**：上表的 native/fla 行已尽量使用相同 `ctx_len`、`chunk_ctx`、
> `micro_batch_size`。不同配置的绝对 tok/s 不可直接比较——每步 token 数越少，
> 固定开销占比越高，两边都会显得“慢”。同配置下 fla 全面更快：
> State ctx512/bsz4 4.35K → 22.95K；infctx ctx1024/chunk256 470 → 1.37K。
>
> 早期表格里 State fla 用了 `sample ctx64/bsz2`（约 600 tok/s），而 native
> 用了 `ctx512/bsz4`（约 5.3K），看起来像 fla 慢了 9 倍，实际是配置不同；
> 同为 `sample ctx64/bsz2` 时 native 约 500 tok/s，fla 约 600 tok/s。

0.1B / ctx512 / bsz4 的 loss 曲线在 native 与 fla 下几乎重合：

```
step 1   native clampw 0.5375   fla clampw 0.5374
step 10  native clampw 0.0580   fla clampw 0.0580
step 20  native clampw 0.0097   fla clampw 0.0098
```

---

## 3. 瓶颈分析（0.1B，native）

`torch.profiler` 单步（ctx 512，bsz 4）曾显示：

```
16.1 ms  elementwise (vectorized)
12.1 ms  ampere_sgemm_128x128_nn     ← fp32 matmul（WKV 后端）
 9.7 ms  Memcpy DtoD
 8.8 ms  ampere_sgemm_128x128_nt
 8.5 ms  ampere_sgemm_128x128_tn
 8.4 ms  FillFunc
 5.6 ms  bf16 s16816gemm             ← 模型主体 matmul
```

单步 365ms 里 GPU kernel 只占 ~100ms，其余是算子启动/CPU 调度空档。
fla 把这些小算子合并进 Triton 内核后，ctx512/bsz4 从 0.5s/step 降到约
0.15s/step（16K tok/s）。

已做的优化：

| 改动 | 效果 |
|---|---|
| `model.backend: fla` | 3.9K → 13.7–16.1K tok/s（0.1B，ctx512，bsz4） |
| `model.wkv_chunk: 128`（native 实际用 96） | 365 → 307 ms/step |
| `train.compile: true` | 365 → 277 ms/step |
| 上述两者叠加 | 262 ms/step，7.8K tok/s |
| clampw 下关闭动态分块检查 | 去掉每块 `.item()` 的 GPU 同步 |
| softplus/exp 改为一次 reduce 估算安全块长 | 不再递归二分；但真实权重仍受块长限制 |

> `wkv_chunk` 超过 96 会被 native 后端夹到 96；fla 固定按 64 分块，
> 这个字段对 fla 无效。bf16 下不同块长的 logits 可能差 2 个 ulp，属正常。

**建议**

| 场景 | 配置 |
|---|---|
| CUDA，任何规模 | 优先 `model.backend: fla` + `precision: bf16` |
| 0.1B / 0.4B（native） | `micro_batch_size` 尽量大（0.1B 可到 8+）+ `wkv_chunk: 96` + `compile: true` |
| 1.5B / 3B | 按显存定 batch（bsz 2–8），`wkv_chunk: 64`，`grad_checkpoint: true` |
| 7B | `bf16-full` + `chunk_ctx 256–512` + DDP/ZeRO-2，`compile` 可选 |
| 长上下文 | `infctx: true` + `chunk_ctx: 512`；显存充足时关 `grad_checkpoint` 可快约 1.7 倍 |
| 只有 CPU / ROCm 没装 fla | native + `clampw`；CPU 必须 `precision: fp32` |

---

## 4. rwkv-fla 验证

### 4.1 版本与导入

新环境是 torch 2.14.1 + triton 3.8.0，`import rwkvfla` 与
`from rwkvfla.ops.rwkv7 import chunk_rwkv7` 均正常。上一台机器
（torch 2.2 + triton 2.2）的两个阻塞点（`get_current_target` 与
`triton.language.extra.libdevice`）都不再存在。

### 4.2 修掉的两个语义问题

1. **`w` 不是 log-decay**：`chunk_rwkv7(w=...)` 内部会再算 `log_w = -exp(w)`。
   本项目已经算好 log-decay，必须传 `log_w=`。之前误传 `w` 时训练 loss
   也能下降，但数学上是错的，和 fp64 参考的输出差异 O(10)。
2. **补齐位不能强衰减**：长度不是 64 的倍数时，尾部用 `log_decay = 0`
   + 全零 k/v/a/b。这样状态原样穿过；用 `-60` 会把携带的状态乘成 0，
   最终状态直接变成 0。

另外统一了输入 dtype：bf16 autocast 下 `k * k_k` 会被 fp32 参数提升成 fp32，
而 value 仍是 bf16，Triton 的 `tl.dot` 会报
`Both operands must be same dtype. Got bf16 and fp32`。

### 4.3 数值精度（相对 float64 逐 token 参考）

| 输入 dtype | 后端 | 输出 rel-L2 | 状态 rel-L2 |
|---|---|---|---|
| fp32 | native | 2.8e-07 | 3.9e-07 |
| fp32 | fla | 7.9e-04 | 7.5e-04 |
| bf16 | native | 5.7e-05 | — |
| bf16 | fla | 3.3e-03 | 1.7e-03 |

梯度相对误差：native ~3e-07，fla 约 1–2e-3。fla 的 fp32 输入也会收到
`ChunkDeltaRuleFunction does not support float32 on some platforms` 警告，
因为内核内部按 bf16 累加；**用 fla 时保持 bf16 即可**。

### 4.4 decay 口径

fla 源码里没有“log-decay 必须 ≤ -0.5”的硬限制。实测 `clampw`
（log-decay ∈ [-0.6066, 0]）、`softplus`、以及极弱衰减（w 接近 0）
都能与 fp64 参考对齐（rel-L2 ~7e-4）。因此配置不再强制
`backend: fla` → `decay: softplus`；官方权重的 `clampw` 可以直接用。

### 4.5 组合验证

- fla + LoRA（clampw）：loss 曲线与 native 基本重合，见第 2 节。
- fla + infctx（clampw，ctx1024/chunk256）：跑通，loss 0.2121（native 0.2120）。
- fla + State Tuning（clampw）：跑通，20 步 loss 5.25 → 0.0009。
- fla + DeepSpeed ZeRO-2：跑通；首次 Triton 编译约 42 秒，之后正常。
- fla 训练出的 State 适配器合并成 `merged.pth` 后，`generate` 正确回答
  `RWKV 0.1B 大模型`，说明保存/合并/重载链路与后端无关。

---

## 5. 本次 GPU 验证中修掉的问题

- **fla**：`w` → `log_w`；补齐位 `-60` → `0`；统一混合精度 dtype。
- **native**：softplus/exp 的递归二分改为“一次 reduce 估算安全块长”，
  去掉每层一次 `.item()` 的同步；真实权重下仍受衰减限制，但不再指数级变慢。
- **launcher**：请求的进程数超过可见 GPU 数时提前报错，不再让 rank 0
  空等 10 分钟；子进程提前崩溃时立即终止其它 rank。
- **DDP 测试**：显式 `--device cpu`，否则在有 GPU 的机器上会自动选 CUDA，
  单卡机器上 rank 1 在 `set_device(1)` 处崩溃。
- **进程组初始化**：`init_process_group` 加 300 秒超时；`set_device` 前检查
  `LOCAL_RANK` 是否超出可见 GPU 数。
- **torch < 2.4 兼容**（上一台机器）：`torch.load(mmap=True)` 只接受 `str`；
  `torch.amp.GradScaler` 不存在，已有兼容层。

---

## 6. 复现命令

```bash
# 环境
python -m venv .venv && . .venv/bin/activate
pip install torch pyyaml && pip install -e .
pip install rwkv-fla deepspeed          # 可选
rwkv-easy doctor

# 模型
rwkv-easy check-model -m /path/to/rwkv7.pth --dtype bf16

# native LoRA（fp32 参数 + bf16 autocast）
rwkv-easy train -c configs/lora_sharegpt.yaml \
  --set model.path=/path/to/model.pth --set data.path=/path/to/data.json \
  --set train.max_steps=100 --set train.micro_batch_size=4 --set train.ctx_len=512 \
  --set train.precision=bf16 --set train.grad_checkpoint=false

# fla LoRA（CUDA，推荐）
rwkv-easy train -c configs/lora_sharegpt_fla.yaml \
  --set model.path=/path/to/model.pth --set data.path=/path/to/data.json \
  --set train.max_steps=100 --set train.micro_batch_size=4 --set train.ctx_len=512

# infctx
rwkv-easy train -c configs/lora_sharegpt_infctx.yaml \
  --set model.path=/path/to/model.pth --set data.path=/path/to/data.json \
  --set train.ctx_len=4096 --set train.chunk_ctx=512 --set train.max_steps=30

# DeepSpeed ZeRO-2（单卡/多卡都行）
rwkv-easy train -c configs/lora_sharegpt.yaml \
  --set dist.mode=deepspeed --set dist.deepspeed.zero_stage=2 ...

# 全量测试（GPU 上会额外跑 5 个 fla 对照）
python -m pytest tests -o addopts="" -q
```
