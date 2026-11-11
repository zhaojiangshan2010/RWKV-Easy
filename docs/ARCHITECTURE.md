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
- 如果某个块的累计衰减超过 60（对应 `e^-60`，已经远低于 fp32 精度），
  就把块对半切开递归处理。块边界在数学上是透明的，所以结果完全不变。
- `log_decay` 下限截断到 -80，防止极端权重下 `exp` 溢出。
- 外层的 autocast 会在后端内部被**关闭**，确保中间量始终是 fp32。

`tests/test_wkv.py` 用 float64 逐 token 参考验证前向、状态和**全部梯度**，
误差要求在 1e-5 以内。

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
for chunk in split(ids, labels, chunk_ctx):
    logits, state = checkpoint(model.forward_chunk, chunk, state)
    loss = 加权合并各块 CE（按有效 token 数）
    state = state.detach() if infctx_detach
```

几个容易搞错的点：

1. **样本长度仍是 `ctx_len`**，`chunk_ctx` 只控制前向粒度。
2. **块边界 token 也参与预测**：`labels` 是整段预先算好的，
   所以第 k 块最后一个 token 的 label 就是第 k+1 块的第一个 token。
3. **loss 按 token 加权合并**，最终值等价于整段 mean CE。
4. **默认 detach**：块间不反传，显存只与 `chunk_ctx` 有关。
   设 `infctx_detach: false` 可以全量反传，但显存会成倍增长。
5. **每块都用 `torch.utils.checkpoint`**，激活不保留，反向时重算。

`tests/test_model.py` 断言：分块前向与整段前向的输出误差 < 1e-5。

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
