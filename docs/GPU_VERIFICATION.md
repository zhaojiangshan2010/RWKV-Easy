# GPU 验证记录

环境：虚拟化 GPU `B1.gpu.large`（23.3 GiB，compute capability 8.6，
Ampere 级）、Linux、Python 3.11.8、**torch 2.2.2 + cu121**、DeepSpeed 0.14.2、
rwkv-fla 0.7.20250822（已安装但与当前 torch/triton 不兼容，见第 4 节）。

模型：`rwkv7-g1d-0.1b-20260129-ctx8192.pth`（12 层 / 768 维 / 191.08M，bf16）。

---

## 1. 结论

- 真实权重在 GPU 上：加载、decay 校准、生成、LoRA、State、infctx、合并、
  fp16、DeepSpeed ZeRO-2 **全部跑通**。
- 速度对 0.1B 小模型是 **launch-overhead 限制**，不是算力限制：batch 从 1 加到 8
  单步耗时几乎不变（365ms → 378ms），吞吐从 1.4K 涨到 10.8K tok/s。
- 真实 1.5B/3B 模型上计算会占主导，固定开销占比会显著下降。

---

## 2. 实测结果

| 场景 | 配置 | 速度 | 峰值显存 |
|---|---|---|---|
| `check-model` | bf16，3×256 token 校准 + 24 token 生成 | 15.5 秒 | — |
| LoRA（72 个线性层） | ctx 512，bsz 4，bf16，无梯度检查点 | ≈4.6–5.0K tok/s（≈0.45 s/step） | 5.8 GiB |
| LoRA | ctx 1024，bsz 2，bf16 | ≈2.5–2.8K tok/s | 5.8 GiB |
| LoRA + infctx | ctx 4096 / chunk 512，bsz 1，梯度检查点 | ≈700 tok/s | 3.7 GiB |
| LoRA + infctx | 同上，关闭梯度检查点 | ≈1.2K tok/s | 11.4 GiB |
| State Tuning | ctx 512，bsz 4，bf16 | ≈5.1–5.6K tok/s | 5.3 GiB |
| State + infctx | ctx 1024 / chunk 256，bsz 1，梯度检查点 | ≈1.9K tok/s | 1.4 GiB |
| fp16 混合精度 | ctx 512，bsz 2 | ≈2.5K tok/s | 3.4 GiB |
| DeepSpeed ZeRO-2（单卡） | ctx 512，bsz 2，bf16 | ≈2.3K tok/s | 3.4 GiB |
| 自我认知过拟合 | sample ctx 64，bsz 2，400 步 | 60 秒，8/8 命中 | 1.2 GiB |

`tok/s` 是日志里的含 padding token 吞吐；peak 为 `torch.cuda.max_memory_allocated`。

---

## 3. 瓶颈分析（0.1B）

`torch.profiler` 单步（ctx 512，bsz 4）：

```
16.1 ms  elementwise (vectorized)
12.1 ms  ampere_sgemm_128x128_nn     ← fp32 matmul（WKV 后端）
 9.7 ms  Memcpy DtoD
 8.8 ms  ampere_sgemm_128x128_nt
 8.5 ms  ampere_sgemm_128x128_tn
 8.4 ms  FillFunc
 5.6 ms  bf16 s16816gemm             ← 模型主体 matmul
```

单步 365ms 里 GPU kernel 只占 ~100ms，其余是 **算子启动/CPU 调度空档**
（每个 WKV 分块有几十个小算子）。

已做的优化：

| 改动 | 效果 |
|---|---|
| `model.wkv_chunk: 128`（后端实际用 96） | 365 → 307 ms/step |
| `train.compile: true` | 365 → 277 ms/step |
| 上述两者叠加 | 262 ms/step，7.8K tok/s |
| clampw 下关闭动态二分（去掉每块一次 `.item()` GPU 同步） | infctx +4% |

> `wkv_chunk` 超过 96 会被 native 后端夹到 96；更大的块在 fp32 下反归一化因子
> 会接近溢出。bf16 下不同块长的 logits 可能差 2 个 ulp，属正常。

**建议**

| 场景 | 配置 |
|---|---|
| 0.1B / 0.4B | `micro_batch_size` 尽量大（0.1B 可到 8+）+ `wkv_chunk: 96` + `compile: true` |
| 1.5B / 3B | 按显存定 batch（bsz 2–8），`wkv_chunk: 64`，`grad_checkpoint: true` |
| 7B | `bf16-full` + `chunk_ctx 256–512` + DDP/ZeRO-2，`compile` 可选 |
| 长上下文 | `infctx: true` + `chunk_ctx: 512`；显存充足时关 `grad_checkpoint` 可快约 1.7 倍 |

0.1B 优先增大 batch（吞吐线性增长到 bsz 8 左右），其次 `wkv_chunk: 96` +
`compile: true`；1.5B 以上按显存正常调 batch 即可。

---

## 4. 关于 rwkv-fla（本次未能对比）

用户环境里装了 `rwkv-fla 0.7.202508221413`，但它与 `torch 2.2.2 / triton 2.2.0`
不兼容：

1. `rwkvfla/utils.py` 在 torch < 2.4 时断言 `device == 'cuda'`，而设备探测依赖
   Triton 新 API `triton.runtime.driver.active.get_current_target()`，2.2.0 没有，
   于是判定为 `cpu` → 导入失败；
2. 即使打补丁绕过，`rwkvfla/ops/utils/op.py` 需要
   `triton.language.extra.libdevice`（Triton ≥ 3.2），2.2.0 也没有。

**要用 fla 的正确组合**：

```bash
# 新环境（不要动现有环境）
python -m venv ~/venv-rwkvfla
~/venv-rwkvfla/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
~/venv-rwkvfla/bin/pip install rwkv-fla pyyaml
# 注意：fla 后端要求 model.decay: softplus
rwkv-easy train -c cfg.yaml --set model.backend=fla --set model.decay=softplus
```

> 但官方权重的正确 decay 是 **clampw**（NLL 2.54 vs softplus 6.75）。用 fla 会拿到
> Triton 的速度，但 decay 口径只能跟随其内核要求；如果追求质量，建议先保持
> `native`，或者在 fla 上做一次小规模 loss 对比再决定。

---

## 5. 本次 GPU 验证中修掉的问题

- `torch < 2.4` 的 `torch.load(mmap=True)` 只接受 `str`，传 `Path` 会报错。
- `torch 2.2` 没有 `torch.amp.GradScaler`（fp16 路径），已加兼容层。
- 在 `model.to(device)` 之后注入 LoRA 时，A/B 建在 CPU 会设备不匹配。
- `--set train.lr=2e-4` 被 YAML 1.1 解析成字符串。
- `--set dist.deepspeed.zero_stage=2` 在 YAML 没写该段时直接报错。
- DeepSpeed 单进程没有进程组会去探测 MPI（缺 `mpi4py`）；现在自动建
  world_size=1 的 PG，并设置 `LOCAL_RANK/RANK/WORLD_SIZE`。
- DeepSpeed 的 `train_batch_size` 与「主循环手动梯度累积」冲突；
  现在 `train_batch_size = micro * world`、`gradient_accumulation_steps = 1`。
- clampw 模式下 WKV 不再逐块做 `.item()` 同步。

---

## 6. 复现命令

```bash
# 环境
python -m venv .venv && . .venv/bin/activate
pip install torch pyyaml && pip install -e .
rwkv-easy doctor

# 模型
rwkv-easy check-model -m /gemini/pretrain/rwkv7-g1d-0.1b-20260129-ctx8192.pth --dtype bf16

# LoRA（fp32 参数 + bf16 autocast）
rwkv-easy train -c configs/lora_sharegpt.yaml \
  --set model.path=/path/to/model.pth --set data.path=/path/to/data.json \
  --set train.max_steps=100 --set train.micro_batch_size=4 --set train.ctx_len=512 \
  --set train.precision=bf16 --set train.grad_checkpoint=false

# infctx
rwkv-easy train -c configs/lora_sharegpt_infctx.yaml \
  --set model.path=/path/to/model.pth --set data.path=/path/to/data.json \
  --set train.ctx_len=4096 --set train.chunk_ctx=512 --set train.max_steps=30

# DeepSpeed ZeRO-2（单卡/多卡都行）
rwkv-easy train -c configs/lora_sharegpt.yaml \
  --set dist.mode=deepspeed --set dist.deepspeed.zero_stage=2 ...
```
