# 架构说明

目标读者：想改代码、加后端、或者判断实现是否正确的人。

## 1. 一张图

```
                      ┌─────────────────────────────┐
 configs/*.yaml ────► │ config.py  解析 + 严格校验   │
                      └──────────────┬──────────────┘
                                     ▼
                      ┌─────────────────────────────┐
                      │ launcher.py  （多卡时拉起 N  │
                      │ 个进程，设置 RANK/WORLD）    │
                      └──────────────┬──────────────┘
                                     ▼
        ┌────────────────────────────────────────────────────┐
        │ train/runner.py  （编排）                           │
        │                                                    │
        │  data/dataset.py ──► tokens/mask ──► DataLoader     │
        │  models/checkpoint.py ──► RWKV7 (meta→assign)       │
        │  models/decay.py ──► auto 校准衰减公式              │
        │  lora.py / state_tuning.py ──► 冻结基座 + 注入参数  │
        │  train/optimizer.py ──► AdamW + 调度器              │
        │  train/backends.py ──► single / DDP / DeepSpeed     │
        └────────────────────────────┬───────────────────────┘
                                     ▼
        ┌────────────────────────────────────────────────────┐
        │ train/loop.py  （训练循环）                         │
        │  普通前向 / infctx 分块前向                          │
        │  CE + L2Wrap + 梯度累积 + 裁剪 + 日志 + 保存         │
        └────────────────────────────┬───────────────────────┘
                                     ▼
        ┌────────────────────────────────────────────────────┐
        │ runs/<实验>/                                        │
        │   adapter-step*.pth / state-step*.pth  （可训练参数）│
        │   trainer-step*.pt              （优化器/进度）      │
        │   merged.pth                    （完整权重）        │
        │   config.resolved.yaml / train.log / metrics.jsonl │
        └────────────────────────────────────────────────────┘
```

## 2. 模型

`models/rwkv7.py` 严格复刻上游 x070 的参数名与计算：

```
emb → blocks[0..L-1] → ln_out → head
        │
        ├─ ln0（仅第 0 层）
        ├─ ln1 → TimeMix  ──┐
        ├─ 残差             │
        ├─ ln2 → ChannelMix ┤
        └─ 残差             │
                            ▼
                    下一层 / 输出
```

TimeMix 的关键量：

```
xx   = shift(x) - x
xr,xw,xk,xv,xa,xg = x + xx * {x_r,x_w,x_k,x_v,x_a,x_g}

r  = receptance(xr)
w  = decay(xw)                      # log 衰减，<= 0
k  = key(xk)
v  = value(xv)
v  = v + (v_first - v) * sigmoid(v0 + (xv@v1)@v2)     # 第 0 层存 v_first
a  = sigmoid(a0 + (xa@a1)@a2)                         # in-context learning rate
g  = sigmoid(xg@g1) @ g2

kk = normalize(k * k_k)                               # 每个 head 内 L2 归一化
k  = k * (1 + (a - 1) * k_a)

out = WKV(r, k, v, w, -kk, kk*a)                      # 核心递推
out = GroupNorm(out)
out = out + (r * k * r_k).sum(-1) * v                 # 额外的 token 级 shortcut
out = output(out * g)
```

GroupNorm 的 eps 是 `1e-5 * head_size_divisor^2`（官方 `head_size_divisor=8`）。

## 3. WKV-7 递推与分块算法

### 3.1 逐 token 形式

设每个 head 的状态为 `S ∈ R^{K×V}`，则对每个 token：

```
S_t = diag(exp(w_t)) S_{t-1} + b_t (a_t · S_{t-1}) + k_t v_t^T
o_t = r_t · S_t
```

其中 `a_t = -kk_t`、`b_t = kk_t * alpha_t`。注意 `a·S` 是对 K 维收缩，
得到 V 维向量，再与 `b`（K 维）做外积。

`backends/reference.py` 就是这个式子的逐行翻译（可选 float64），
是所有“聪明实现”的黄金参考。

### 3.2 块内并行

把长度 L 的块展开，记 `D_t = prod_{i<=t} exp(w_i)`：

```
S_t = D_t ⊙ S_{-1} + Σ_{j<=t} (D_t / D_j) ⊙ (k_j v_j^T + b_j u_j^T)
u_j = a_j · S_{j-1}
```

把 `u_j` 收集成矩阵 U，就得到严格下三角系统：

```
(I - A) U = RHS
A[j,i]  = Σ_c a[j,c] b[i,c] (D_{j-1}/D_i)[c]        (i < j)
RHS[j]  = Σ_c a[j,c] D_{j-1}[c] S_{-1}[c,:] + Σ_{i<j} ak[j,i] v_i
```

一次 `torch.linalg.solve_triangular` 就能解出整块，块间只传 `S`。
这就是 `backends/native.py`，与 FLA / Triton 的 chunked delta-rule 同构，
但完全用 PyTorch 写成。

### 3.3 数值稳定性

- 所有输入先转 **float32**；三角求解只支持 fp32/fp64。
- 指数一律用**差值**形式 `exp(logD_t - logD_j) ≤ 1`，避免 `1/cumprod` 溢出。
- 如果块内累计衰减可能超过 60（对应 `e^-60`，已经远低于 fp32 精度），
  先用一次 reduce 求出本 batch 最强的单 token 衰减，把块长压到
  `60 / max_rate`，再顺序分块。块边界在数学上是透明的，所以结果完全不变。
- `log_decay` 下限截断到 -80，防止极端权重下 `exp` 溢出。
- 外层的 autocast 会在后端内部被**关闭**，确保中间量始终是 fp32。

`tests/test_wkv.py` 用 float64 逐 token 参考验证前向、状态和**全部梯度**，
误差要求在 1e-5 以内。

### 3.4 fla 后端

`backends/fla.py` 适配 rwkv-fla 的 `chunk_rwkv7`，有两个容易踩的细节：

- `chunk_rwkv7(w=...)` 的 `w` 是**原始 decay 参数**，内核内部会再算
  `-exp(w)`；本项目传的已经是 log-decay，必须走 `log_w=`。
- 长度不是 64 的倍数时，尾部用 `log_decay = 0` + 全零 k/v/a/b 补齐。
  这表示“状态原样穿过”；用 -60 之类的强衰减会把携带的状态乘成 0。
- 混合精度下各输入 dtype 可能不一致（例如 `kk` 被 fp32 参数提升），
  适配层会统一到激活 dtype；Triton 的 `tl.dot` 要求两侧 dtype 相同。

fla 内核按输入 dtype 计算（bf16 下与 fp64 参考相对误差 ~3e-3），
clampw / softplus / exp 三种 decay 都支持。`tests/test_fla_backend.py`
在没有 GPU 时自动跳过。

## 4. infctx：状态怎么传

状态容器 `models/state.py`：

```
ModelState
├── shift_att : (L, B, C)          每层 time-mix 的时间移位
├── shift_ffn : (L, B, C)          每层 channel-mix 的时间移位
└── wkv       : (L, B, H, K, V)    每层 WKV 循环状态
```

训练循环（`train/loop.py::_infctx_loss`）：

```
state = model.initial_state(B)        # 若有 State Tuning，用 time_state 作为初值
total_tokens = count(labels != -100)
for chunk in split(ids, labels, chunk_ctx):
    logits, state = forward_chunk(chunk, state)
    loss_k = CE(logits, labels_k)

    if infctx_backward:               # 默认；单进程
        backward(loss_k * tokens_k / total_tokens / grad_accum)
        state = state.detach()        # 立刻释放本块激活
    else:                             # DDP / DeepSpeed / 显存充足
        total_loss += loss_k * tokens_k
        state = state.detach() if infctx_detach
backward(total_loss / total_tokens)   # 仅旧路径
```

几个容易搞错的点：

1. **样本长度仍是 `ctx_len`**，`chunk_ctx` 只控制前向粒度。
2. **块边界 token 也参与预测**：`labels` 是整段预先算好的，
   所以第 k 块最后一个 token 的 label 就是第 k+1 块的第一个 token。
3. **loss 按 token 加权合并**，最终值等价于整段 mean CE。
4. **默认每块立即 backward**（`train.infctx_backward: true`）：激活只保留当前块，
   显存真正是 `O(chunk_ctx)`。0.1B 可训 32K 上下文（3.4 GiB）。
   DDP / DeepSpeed 暂时回退到“所有块 loss 累加、最后统一 backward”，
   显存会随 `ctx_len` 增长。
5. **detach 是默认**：块间不反传，等价于截断 BPTT。设 `infctx_detach: false`
   会关闭 per-chunk backward（否则图已被释放），走全量 BPTT，显存成倍增长。
6. **梯度检查点可选**：per-chunk backward 已经限制了激活，通常不再需要
   `grad_checkpoint`（0.1B 实测关掉后 32.5K vs 23.7K tok/s）。

`tests/test_model.py` 断言：分块前向与整段前向的输出误差 < 1e-5；
`tests/test_infctx_backward.py` 断言新旧两条 backward 路径的 loss 与梯度一致。

## 5. 为什么自己写 LoRA

HuggingFace PEFT 的 `PeftModelForCausalLM.forward(input_ids, attention_mask, ...)`
会把额外的位置参数当成 `attention_mask` / `inputs_embeds`。参考实现
`--train_type infctx --peft lora` 崩溃就是这么来的。

本项目的做法：

```
TimeMix.key: nn.Linear(C, C)
        ↓ inject_lora
TimeMix.key: LoRALinear(base=nn.Linear(C, C), A, B)
```

模型 `forward(input_ids, attention_mask, state, need_state)` 的签名完全不变，
`state` 作为显式参数一路传到 WKV 后端。LoRA 只影响 `nn.Linear` 的计算：

```
y = base(x) + dropout(x) A^T B^T * (alpha/r)
```

`B` 零初始化 → 训练开始时与基座完全一致。

## 6. State Tuning

每层 attention 增加一个参数：

```
time_state : (H, K, K)      # K = head_size
```

- 零初始化；
- 训练时整个基座 `requires_grad_(False)`，只有 `time_state` 可训练；
- 前向时：
  - 若没有外部状态（普通训练 / infctx 的第一块）→ 用 `time_state` broadcast 到 batch；
  - 若有外部状态（infctx 后续块）→ 用真实状态；
- 因此 `state`、`state+infctx` 共用同一条路径，不需要两套类。

`model.initial_state()` 会显式把 `time_state` 堆叠成 `(L,B,H,K,K)`，
这样 `torch.utils.checkpoint` 能正确把梯度传回参数（`tests/test_train_smoke.py`
里有一条断言检查训练后 `time_state` 不为零）。

## 7. checkpoint 格式

### 适配器 / 状态（训练时每个 checkpoint）

```python
{
  "format": "rwkv-easy",
  "version": 1,
  "method": "lora" | "state",
  "step": 100,
  "epoch": 0,
  "base_signature": "b3c9373a9f73cdf7",   # 基座 key 集合的 sha1 前 16 位
  "model_config": {"n_layer": 24, ...},
  "decay": "clampw",
  "method_config": {"rank": 16, "alpha": 32, ...},
  "weights": {"blocks.0.att.key.lora_A": tensor, ...}
}
```

- LoRA 权重 key 形如 `blocks.0.att.key.lora_A` / `.lora_B`；
- State 权重 key 形如 `blocks.0.att.time_state`；
- `trainer-step*.pt` 另外保存优化器、调度器、GradScaler、step/epoch。

### `merged.pth`

普通 RWKV-7 state_dict，可以直接给官方推理代码用。

- LoRA：`W' = W + (B@A) * alpha/r`，并把 `LoRALinear` 替换成普通 `Linear`；
- State：把 `time_state` 覆盖到基座权重上。

## 8. 精度

| precision | 参数量化 | autocast | 缩放 |
|---|---|---|---|
| fp32 | 无 | 无 | 无 |
| fp16 | 无 | fp16 | GradScaler |
| bf16 | 无 | bf16 | 无 |
| bf16-full | bf16 | 无 | 无 |

WKV 内部始终 fp32。LoRA 的 A/B 在 `bf16-full` 下仍然保持 fp32，
前向时按输入 dtype 转换，梯度累积在 fp32 参数里。

**加载时会按目标精度显式转换**（包括 fp32 上转）。官方 checkpoint 通常是 bf16，
如果 `precision: fp32` 却保留 bf16 参数，CPU 上的反向传播会慢约 75 倍
（实测 42s vs 0.57s / step）；GPU 上则通常希望保持 bf16 省显存。因此
`runner.py` 在加载后立刻 `model.to(dtype=target)`，日志会打印
`模型参数 dtype -> ...`。

## 9. 分布式

| 方式 | 实现 | 备注 |
|---|---|---|
| 单卡 | 直接训练 | |
| DDP | `torch.nn.parallel.DistributedDataParallel` | 梯度累积时用 `no_sync()` 跳过中间 all-reduce |
| DeepSpeed | `deepspeed.initialize` | 只把可训练参数交给优化器；ZeRO-3 用 `GatheredParameters` 汇总保存 |

启动器（`launcher.py`）在检测到多卡时，用 `subprocess` 拉起 N 个进程并设置
`RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT`。不用 `torchrun` 是为了绕开
c10d rendezvous 在部分 Windows / 精简 PyTorch 上的 `libuv` 问题；用户自己用
`torchrun` 启动也完全兼容（代码读的是同一套环境变量）。

DeepSpeed 的两个额外处理：

- **单进程**（没有 torchrun）时 DS 没有进程组，会去探测 MPI 并报
  `No module named 'mpi4py'`。`deepspeed_backend.py` 会先建一个 world_size=1
  的 PG，并设置 `LOCAL_RANK/RANK/WORLD_SIZE`。
- **梯度累积由主循环负责**，所以 DS 侧 `gradient_accumulation_steps=1`，
  `train_batch_size = micro * world_size`（不乘 accum），否则 DS 的
  batch 一致性校验会失败。

## 10. 关键不变量（改代码时请保持）

1. `ModelState` 的三个张量形状固定：`(L,B,C)`、`(L,B,C)`、`(L,B,H,K,K)`。
2. 模型 forward **不得**依赖全局环境变量决定训练模式。
3. LoRA/State 只能通过 `requires_grad` 控制，不能改变模型签名。
4. infctx 的 loss 必须按有效 token 数加权，否则与非 infctx 不可比。
5. 所有保存都必须经过 `atomic_save`（先写临时文件再替换）。
6. 分块前向必须与整段前向数值等价；新增后端要过 `tests/test_wkv.py` 的
   float64 对照。
7. **生成 prompt 的分词结果必须是训练文本分词的前缀**。训练时
   `"Assistant: "` 的尾空格会被贪心分词并入回答首 token；推理必须用
   `ChatTemplate.generation_prompt()`（去掉尾空格、保留换行），否则过拟合模型
   会跑偏。`tests/test_dataset.py` 对 rwkv_world / chatml 都有断言。
8. 模型加载后必须按目标精度显式转换（含 fp32 上转），不要只在
   `dtype != fp32` 时转换。
9. `infctx_backward` 只在 `infctx_detach=true` 且单进程后端启用；
   它会在每个 chunk 结束后释放图，不能与跨块 BPTT 混用。

## 11. 性能画像

0.1B / ctx512 / bf16-full / fla + compile：bsz32 单步约 338ms（8192 token），
**48.5K tok/s ≈ 55.6 TFLOPS**；State + reduce-overhead 约 56.3K tok/s ≈ 64.5 TFLOPS。
设备 bf16 matmul 峰值 62–74 TFLOPS，已经用到 75–90%。

profile（bsz24）里 GPU 时间大致是：elementwise ~38%、`aten::mm` ~26%、
`aten::copy_` ~17%、fla WKV ~9%。因此：

- **`bf16-full` 是最大单项收益**：基座权重直接 bf16，消掉 autocast 每次
  Linear 的 fp32→bf16 权重转换；LoRA/time_state 仍是 fp32。
  bsz16：23.8K → 34.8K tok/s，显存 14.6 → 8.9 GiB；
- **只对可训练 token 过 lm_head**：SFT 里约一半 prompt token 不需要 logits，
  0.1B 再 +9%，显存也明显下降；
- **batch 要加大**：fla bsz4→32 从 17.4K 涨到 40.3K（eager）；
- **`torch.compile` 按后端选择对象**：fla 编译 `forward_hidden`（不含 head，
  +20%）；native 只编译 WKV `_chunk` 单块图（+18%），整模型展开会编译十几分钟；
  `reduce-overhead` 还能进一步降显存；
- `wkv_chunk` 只影响 native；clampw 下关闭动态分块检查，避免每块 `.item()`；
- L2Wrap 改为在原地梯度上 scatter_add，省掉每步 2 GiB 全量梯度；
- infctx 每个 chunk 立即 backward，显存从 O(ctx_len) 变成 O(chunk_ctx)：
  8K ctx 12.4 → 3.4 GiB，32K ctx 也能在 24GB 卡上训练。

native 在 clampw 下用 bsz24 可到 24.3K tok/s（compile WKV 单块）；softplus/exp
模式会先用一次 reduce 估算安全块长，真实权重的 softplus log-decay 约为 -2～-7，
块长会被压到 8 左右，吞吐降到约 130 tok/s——CUDA 上应直接用 fla。

详细数字与复现命令见 [`GPU_VERIFICATION.md`](GPU_VERIFICATION.md)。

## 12. 训练 loss 与前向编译

### 12.1 head 只作用于可训练 token

模型有两个前向入口：

```
forward(input_ids, ...)          -> logits, state       # 推理 / 诊断
forward_hidden(input_ids, ...)   -> hidden, state       # 训练
```

`forward` 就是 `head(forward_hidden(...))`。训练时 `Trainer._cross_entropy`
只取 `label != -100` 的行过 `lm_head`：

```
hidden: (B,T,C) -> flat[valid] -> (N_valid,C) -> head -> (N_valid,V)
loss = CE(logits, labels[valid], reduction="sum") / N_valid
```

数学上与「整段算 logits 再 `ignore_index`」完全等价，但 prompt token 不再
产生 65536 维 logits。`forward_chunk_hidden` 是 infctx 的对应版本，签名与
`forward_chunk` 相同，只是返回 hidden 而不是 logits。

L2Wrap 不再单独建一份全量梯度，而是在 CE 已算出的 logits 梯度上原地
`scatter_add_` 每行 max logit；缩放因子仍按总 token 数计算。

### 12.2 `train.compile` 的两条路径

`Trainer` 不再编译整个 model（之前编译结果也没被前向使用），而是按后端选择：

| 后端 | 编译对象 | 原因 |
|---|---|---|
| fla | `model.forward_hidden` / `forward_chunk_hidden` | 融合模型 elementwise，+20% |
| native | `backend._chunk` | 整模型会把 12 层×N 块展开，编译极慢；单块图 +18% |

`compile_mode` 支持 `default` / `reduce-overhead` / `max-autotune`。
`reduce-overhead` 会启用 CUDA graphs，State 场景还能降显存；`max-autotune`
首次编译约 8 分钟，适合长期训练。

### 12.3 infctx 的 backward 路径

`infctx_backward: true`（默认）时每个 chunk 立即 backward，显存 O(chunk_ctx)；
DDP / DeepSpeed / `infctx_detach=false` 会自动回退到「所有块 loss 累加、
最后统一 backward」的旧路径。两条路径的 loss 与梯度一致性由
`tests/test_infctx_backward.py` 覆盖。
