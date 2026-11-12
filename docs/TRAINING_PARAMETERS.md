# 训练参数手册

本页是 `rwkv_easy/config.py` 中**全部配置字段**的权威说明。目标读者是以后维护、
扩展或排查训练问题的人。

- “默认值”指 dataclass 的默认值；`configs/*.yaml` 模板会覆盖一部分（例如 GPU
  模板常用 `bf16-full`、`compile: true`、`chunk_ctx: 2048`）。
- 配置优先级：**dataclass 默认值 < YAML < `--set`（按出现顺序）< 显式 CLI 参数**
  （`--device` / `--backend` / `--single` / `--num-gpus` / `--random-init` /
  `--run-dir` / `--resume` / `--log-level`）。
- 所有路径都相对于**启动命令的当前目录**；向导会保存绝对路径。
- 单位：`step` 是优化器更新次数；`epoch` 是遍历一遍数据；`token/step` 是
  一个优化器 step 实际消费的 token 数。

一个优化器 step 的有效 batch：

```
effective_batch = micro_batch_size × grad_accum × world_size
tokens_per_step = micro_batch_size × grad_accum × ctx_len × world_size
```

> 本页只讲字段语义。调优建议、实测数字见
> [`PERFORMANCE.md`](PERFORMANCE.md)，数据 mask 细节见 [`DATASET.md`](DATASET.md)。

---

## 1. `run` —— 实验标识

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `run.name` | str | `rwkv-easy` | 实验名；输出目录为 `<output_dir>/<name>-<method>[-infctx]-<时间戳>/` |
| `run.output_dir` | str | `runs` | 所有实验目录的父目录 |
| `run.seed` | int | `42` | Python / NumPy / PyTorch / CUDA / DataLoader worker / 验证集切分的随机种子 |
| `run.deterministic` | bool | `false` | 是否让 cuDNN 使用确定性算法（会关闭 benchmark） |

补充：

- `run.name` 会原样进入目录名，建议用简短英文/数字；不要带 `/`。
- 同一次实验的完整配置会写到 `run_dir/config.resolved.yaml`，复现时优先保留它。
- `run.seed` 现在也会写入环境变量 `RWKV_EASY_SEED`，DataLoader 子进程据此初始化
  NumPy/Python 随机数；修改 seed 会同时改变数据顺序、dropout、LoRA 初始化等。
- **TF32 不受 `run.deterministic` 控制**：`seed_everything()` 在 CUDA 上会始终打开
  `torch.backends.cuda.matmul.allow_tf32` 与 `cudnn.allow_tf32`（这是为了 fp32
  matmul 速度）。如果做严格数值复现，需要注意这一点。
- `run.deterministic=true` 只影响 cuDNN 的 deterministic/benchmark；分布式下
  数据顺序还取决于 `DistributedSampler` 的 seed（同样来自 `run.seed`）。

---

## 2. `model` —— 模型、权重与 WKV 后端

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `model.path` | str \| null | `null` | 官方 RWKV-7 state_dict（`.pth` / `.safetensors`） |
| `model.random_init` | bool | `false` | 不加载权重，随机初始化小模型（自检/调试） |
| `model.n_layer` | int \| null | `null` | 层数；默认从权重推断 |
| `model.n_embd` | int \| null | `null` | 隐藏维度；默认从权重推断 |
| `model.vocab_size` | int \| null | `null` | 词表大小；默认从权重推断 |
| `model.head_size` | int \| null | `null` | 每头维度；默认从 `att.r_k` 推断 |
| `model.head_size_divisor` | int | `8` | GroupNorm eps = `1e-5 × divisor²`，官方值 8 |
| `model.decay` | str | `auto` | 衰减公式：`auto` / `clampw` / `softplus` / `exp` |
| `model.backend` | str | `native` | WKV 后端：`native` / `reference` / `fla` / 自定义注册名 |
| `model.wkv_chunk` | int | `64` | native 的 WKV 内部块长（会被夹到 8–96）；fla 忽略 |
| `model.dtype` | str \| null | `null` | 强制权重存储 dtype：`fp32` / `bf16` / `fp16` |

补充：

- `model.path` 指向**基座权重**，不是 LoRA adapter。加载时支持 `module.` /
  `model.` 等前缀清洗；维度能自动推断。
- `model.random_init=true` 时必须提供 `n_layer` / `n_embd` / `vocab_size`
  （`head_size` 可省，默认 64）。它主要用于 `selftest` / `debug_cpu.yaml`。
- 手填 `n_layer` 等维度与权重不一致会直接报错；正常情况下留 `null`。
- `head_size_divisor` 决定 GroupNorm 的 eps，官方 x070 是 8；不要随意改，
  否则会影响数值。
- `model.dtype` 是**存储 dtype 覆盖**，优先级高于 `train.precision` 推导出的
  storage dtype。常见用法：`precision: bf16` + `model.dtype: bf16` 等价于
  `bf16-full`；一般不需要设置。

### 2.1 `model.decay`

| 值 | 公式 | 说明 |
|---|---|---|
| `clampw` | `-exp(-0.5) * sigmoid(raw)` | 官方权重（CUDA 内核口径），推荐 |
| `softplus` | `-softplus(-raw) - 0.5` | 参考实现 fla 路径；fla 也能用 |
| `exp` | `raw.clamp(max=0)` | 旧版 |
| `auto` | 用内置中英文本各跑一次，选 NLL 最低 | 默认；结果缓存到 `run_dir/.cache/decay_calibration.json` |

`auto` 需要在 rank0 上做 3 次 256-token 前向，首次启动会慢几秒到几十秒；
缓存 key 包含 checkpoint 签名与模型维度。真实权重通常选中 `clampw`。

### 2.2 `model.backend`

| 值 | 适用 | 说明 |
|---|---|---|
| `native` | CPU / CUDA / ROCm / MPS | 纯 PyTorch 分块 WKV，无额外依赖 |
| `fla` | CUDA / ROCm + Triton ≥ 3.2 | rwkv-fla 融合内核，CUDA 上通常最快 |
| `reference` | 测试 | 逐 token 参考实现，非常慢；测试里用 `double=True` 做 fp64 黄金参考 |
| 自定义 | 由 `register_backend()` 注册 | 见 `docs/EXTENDING.md` |

`fla` 支持 clampw / softplus / exp；不需要强制 softplus。用 fla 时建议
`precision: bf16` 或 `bf16-full`。

### 2.3 `model.wkv_chunk`

- **只影响 native**：fla 内核固定按 64 分块，这个字段对 fla 无效。
- native 用它控制每层 WKV 的内部分块数；太小会增加启动/三角求解次数，
  太大会增加 `O(L³)` 求解和显存。0.1B 实测 64–80 是甜点，32 明显变慢。
- 构造函数会把值夹到 **8–96**；softplus/exp 还会根据最强 log-decay 自动
  缩小实际块长（`model.wkv_chunk` 只是上限）。
- 不要和 `train.chunk_ctx` 混淆：后者是 **infctx 外层**前向/反传块。

---

## 3. `train` —— 训练超参

### 3.1 方法与上下文

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `train.method` | str | `lora` | `lora` 或 `state` |
| `train.infctx` | bool | `false` | 是否用分块长上下文训练 |
| `train.ctx_len` | int | `2048` | 每个样本/窗口的总长度 |
| `train.chunk_ctx` | int | `512` | infctx 每块长度；`<= ctx_len`，建议 16 的倍数 |

- `train.method=lora`：冻结基座，只训练 LoRA A/B。
- `train.method=state`：冻结基座，只训练每层 attention 的 `time_state`。
  此时 `lora.*` 字段不生效。
- `train.ctx_len` 是模型实际看到的长度，不是显存单位。
- `train.chunk_ctx` 是 infctx 的显存单位：开启 `infctx_backward` 后显存与
  `chunk_ctx` 成正比，与 `ctx_len` 无关。`data.mode` 必须是 `packed`。
- infctx 的块边界 token 也参与预测；具体 loss 加权见
  `docs/ARCHITECTURE.md` 第 4 节。

### 3.2 batch、步数与优化器

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `train.micro_batch_size` | int | `2` | 单卡单次前向的样本数 |
| `train.grad_accum` | int | `8` | 梯度累积 micro-batch 数 |
| `train.epochs` | int | `1` | 训练轮数 |
| `train.max_steps` | int \| null | `null` | 最大优化器 step；设置后以它为准 |
| `train.lr` | float | `1e-4` | 初始学习率 |
| `train.min_lr` | float | `1e-5` | 调度结束时的学习率 |
| `train.warmup_steps` | int | `20` | 线性 warmup 步数 |
| `train.lr_schedule` | str | `cosine` | `cosine` / `linear` / `constant` |
| `train.weight_decay` | float | `0.01` | AdamW weight decay（只作用于 ≥2 维参数） |
| `train.beta1` | float | `0.9` | AdamW beta1 |
| `train.beta2` | float | `0.99` | AdamW beta2 |
| `train.grad_clip` | float | `1.0` | 梯度范数裁剪；0 关闭 |

- `max_steps` 比 `epochs` 优先。设置后 `max_epochs = ceil(max_steps / steps_per_epoch)`，
  训练到 `max_steps` 停止；epoch 末尾的残余梯度也会做一次 optimizer step。
- `warmup_steps` 内 lr 从 `lr/warmup` 线性升到 `lr`；之后按
  `cosine/linear/constant` 降到 `min_lr`。
- `weight_decay` 只作用于 ≥2 维参数；1 维参数（bias/norm）与 `time_state`
  不衰减。LoRA 的 A/B 都是 2 维，会参与 decay。
- `grad_clip` 在 optimizer step 前按所有可训练参数的全局范数裁剪。
- DDP/DeepSpeed 下 `grad_accum` 由训练循环控制；DDP 使用 `no_sync()`，
  DeepSpeed 使用手动累积（`gradient_accumulation_steps=1`）。

### 3.3 精度

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `train.precision` | str | `bf16` | `fp32` / `fp16` / `bf16` / `bf16-full` |

| 值 | 参数存储 | 前向 autocast | GradScaler | 适用 |
|---|---|---|---|---|
| `fp32` | fp32 | 否 | 否 | 调试、CPU、数值验证 |
| `fp16` | fp32 | fp16 | 是 | 老 GPU；CPU 不支持 |
| `bf16` | fp32 | bf16 | 否 | 兼容性最好；每层 Linear 有 fp32→bf16 权重转换 |
| `bf16-full` | bf16 | 否 | 否 | **GPU 推荐**；基座 bf16，LoRA/time_state 仍 fp32 |

补充：

- `bf16-full` 需要 GPU 支持 bf16（Ampere 及以上）；CPU 上会很慢，不建议。
- `fp16` 只在 CUDA 可用；CPU 上 `validate_precision` 会直接拒绝。
- `fp16 + grad_accum > 1` 对 GradScaler 不友好，建议 bf16。
- WKV 的 native 后端内部始终用 fp32 计算；fla 按输入 dtype 交给 Triton。

### 3.4 显存与 infctx

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `train.grad_checkpoint` | bool | `true` | 激活重计算，省显存、略慢 |
| `train.infctx_detach` | bool | `true` | infctx 块间是否 detach 状态（截断 BPTT） |
| `train.infctx_backward` | bool | `true` | 单卡 infctx 每块立即 backward（显存 ∝ chunk_ctx） |
| `train.infctx_parallel_backward` | bool | `true` | DDP + infctx 每块立即 backward，最后一块 all-reduce |

- `grad_checkpoint`：对 block（以及 infctx 的 chunk）做 `torch.utils.checkpoint`；
  0.1B + per-chunk backward 时通常可以关掉，速度约 +40%。
- `infctx_detach=true`：块间不反传，显存稳定；`false` 会走全量 BPTT，
  显存成倍增加，并且 `infctx_backward` / `infctx_parallel_backward` 自动关闭。
- `infctx_backward=true`：单卡每个 chunk 完成 forward 后立即 backward，只保留
  detach 后的状态。
- `infctx_parallel_backward=true`（默认）：**DDP + LoRA** 下同样逐块反传；
  每个 chunk 是独立的 forward/backward 对，除最后一块外都包在 `no_sync()` 里，
  最后一块触发一次梯度 all-reduce。显存 O(chunk_ctx)，不随 `ctx_len` 增长。
- 设 `infctx_parallel_backward: false`：DDP 回退到“所有块 loss 累加、最后统一
  backward”，显存随 `ctx_len` 增长；长上下文容易 OOM。
- **DeepSpeed** 与 **State Tuning + DDP** 暂不支持逐块路径，会回退到最后统一
  反传；`dist.mode: deepspeed` + 该字段为 true 会在配置校验时直接报错。
- State Tuning 的后续块如果没有可训练输入，会跳过 backward（只有第一块
  对 `time_state` 有梯度）；这也是 State + DDP 暂不走逐块路径的原因。

### 3.5 loss 稳定项

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `train.l2wrap` | float | `1e-4` | RWKV L2Wrap 稳定项；0 关闭 |

L2Wrap 只改梯度，不改变报告的 loss：在每行 max logit 上叠加
`l2wrap / num_tokens` 的梯度。当前实现是在 CE 已算出的 logits 梯度上原地
`scatter_add_`，不额外分配全量梯度。SFT 里只作用于可训练 token 行。
如果显存非常紧或追求极限速度，可以设 0；一般建议保留。

### 3.6 日志、保存与验证

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `train.log_every` | int | `10` | 每多少 step 打印一次（step 1 总会打印） |
| `train.save_every_steps` | int | `0` | 每 N step 保存；0 关闭 |
| `train.save_every_epochs` | int | `1` | 每 N epoch 保存；0 关闭 |
| `train.keep_last` | int | `2` | 只保留最近 N 组 checkpoint |
| `train.merge_at_end` | bool | `true` | 结束自动导出 `merged.pth` |
| `train.val_every_steps` | int | `0` | 每 N step 跑验证集；0 关闭 |

- 训练结束一定会保存最后一个 checkpoint（即使 `save_every_*` 都是 0）。
- `keep_last` 只清理 checkpoint，不影响 `train.log` / `metrics.jsonl`。
- `merge_at_end` 会额外写一份完整权重；7B fp32 可能几十 GB，不需要就关。
- `val_every_steps` 需要 `data.val_ratio > 0`；当前实现只在
  `metrics.jsonl`/日志里记录 `eval_loss` 与 `best_loss` 指标，
  **不会单独保存 best checkpoint**。
- DeepSpeed ZeRO 的优化器状态是分片的，当前不保存/恢复；续训会恢复权重与
  step，但优化器状态从头开始。需要严格续训请用 DDP 或单卡。

### 3.7 compile 与 device

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `train.compile` | bool | `false` | 编译训练前向（fla）/ WKV 单块（native） |
| `train.compile_mode` | str | `default` | `default` / `reduce-overhead` / `max-autotune` |
| `train.device` | str | `auto` | `auto` / `cpu` / `cuda` / `cuda:0` |

- `compile=true` + `backend=fla`：编译 `forward_hidden` / `forward_chunk_hidden`，
  0.1B 实测 +20%；首次 step 会编译。
- `compile=true` + `backend=native`：只编译 `backend._chunk` 单块图（+18%，
  首次约 25 秒）；不要编译整个 native model，会展开 12 层×N 块，可能十几分钟。
- `reduce-overhead` 会启用 CUDA graphs，State 场景还能降显存；
  `max-autotune` 更快一点但首次编译约 8 分钟。
- `device=auto` 在 CUDA 机器上选当前 CUDA，否则 MPS，最后 CPU；
  多卡时每个 rank 的 `LOCAL_RANK` 由 launcher 设置。

---

## 4. `lora` —— LoRA 超参

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `lora.rank` | int | `16` | 低秩维度 r |
| `lora.alpha` | int | `32` | 缩放系数，实际 scale = `alpha / rank` |
| `lora.dropout` | float | `0.0` | LoRA 分支 dropout |
| `lora.target_modules` | list[str] | 6 个后缀 | 按模块名后缀匹配要注入的 Linear |
| `lora.target_regex` | str \| null | `null` | 额外正则匹配（与后缀取并集） |
| `lora.target_layers` | list[int] \| null | `null` | 只注入这些 block（0-based）；`null` = 全部 |

默认目标模块：

```yaml
target_modules:
  - att.receptance
  - att.key
  - att.value
  - att.output
  - ffn.key
  - ffn.value
```

- 只注入 attention：

  ```yaml
  target_modules: [att.receptance, att.key, att.value, att.output]
  ```

- 只训练前 6 个 block：

  ```yaml
  target_layers: [0, 1, 2, 3, 4, 5]
  ```

- 过滤顺序：先按 `blocks.<i>.` 判断 `target_layers`，再匹配后缀/regex。
- `lora.rank`/`alpha` 同时决定可训练参数量；0.1B 的 6 模块 r=8 约 1.33M。
- B 零初始化，训练开始时输出与基座完全一致；`merge` 使用
  `W + (B@A) * alpha/rank`。
- `target_regex` 用于官方命名之外的自定义模块；和 `target_modules` 是并集。
- `target_layers=[]` 是非法配置（会报错）；想训练全部请用 `null`。

---

## 5. `state` —— State Tuning

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `state.init_std` | float | `0.0` | `time_state` 初始化标准差；0 = 零初始化 |

- 每层 attention 增加 `time_state: (H, K, K)`；基座全部冻结。
- 训练时它作为每个样本第一块的初始循环状态；infctx 后续块用真实状态。
- 官方做法是 0；非 0 只用于实验。
- 合并时覆盖基座里的 `blocks.N.att.time_state`；推理代码需要认识这个 key，
  或者用 adapter 方式加载。

---

## 6. `data` —— 数据与样本

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `data.path` | str | 必填 | 数据文件或目录 |
| `data.format` | str | `auto` | `auto` / `sharegpt` / `messages` / `text` |
| `data.mode` | str | `packed` | `packed`（定长流）/ `sample`（单条对话） |
| `data.text_field` | str | `text` | 纯文本记录使用的字段名 |
| `data.max_samples` | int \| null | `null` | 最多读取多少条记录（调试/快速实验） |
| `data.val_ratio` | float | `0.0` | 验证集比例（0–1） |
| `data.cache` | bool | `true` | 是否缓存 tokenize 结果 |
| `data.cache_dir` | str \| null | `null` | 缓存目录；默认数据文件旁的 `.rwkv_easy_cache/` |
| `data.vocab_path` | str \| null | `null` | 自定义词表；默认内置 RWKV 世界词表 |
| `data.num_proc` | int | `4` | tokenize 进程数；0/1 为单进程 |
| `data.num_workers` | int | `2` | DataLoader worker 数 |
| `data.pin_memory` | bool | `true` | 是否 pin memory（仅 CUDA 生效） |
| `data.template` | str | `rwkv_world` | `rwkv_world` / `chatml` / `custom` |
| `data.system_prefix` | str \| null | `null` | 自定义模板的 system 前缀 |
| `data.user_prefix` | str \| null | `null` | 自定义模板的 user 前缀 |
| `data.assistant_prefix` | str \| null | `null` | 自定义模板的 assistant 前缀 |
| `data.turn_separator` | str \| null | `null` | 自定义模板的轮次分隔符 |
| `data.train_on` | str | `assistant` | `assistant` / `last_assistant` |
| `data.append_eos` | str | `end` | `never` / `end` / `each_assistant` |
| `data.eos_token_id` | int \| null | `null` | EOS token id；null 自动取词表里的 `b"\x00"` |
| `data.truncation` | str | `head` | sample 模式截断：`head` / `tail` / `random` |
| `data.drop_empty` | bool | `true` | 丢弃没有任何可训练 token 的样本 |

补充：

- `data.format=auto` 会根据记录字段自动识别 ShareGPT / messages / text /
  两列指令；`text` 会把整段内容都标为可训练。
- `data.mode=packed` 把所有记录拼成 token 流按 `ctx_len` 切窗，利用率高、
  兼容 infctx；`sample` 一条对话一个样本、动态 padding，不能与 infctx 同用。
- `data.train_on=last_assistant` 时，前面的 assistant 回复降级为上下文。
- `data.append_eos=end` 只在对话以 assistant 结尾时追加 EOT；
  `each_assistant` 每轮都追加；追加的 EOT 参与 loss。
- `data.eos_token_id` 自动从词表找 `b"\x00"`（通常是 1）；换词表时注意。
- `data.truncation` 只影响 `sample` 模式；`packed` 通过切窗处理长度。
- `data.max_samples` 在训练/验证切分前生效，也会进入 tokenize 缓存 key。
- `data.val_ratio` 至少 10 条记录才会切分；验证集 loss 只在
  `val_every_steps>0` 时计算。
- `data.num_proc` 会按 world_size 均分；记录数 <256 时自动单进程。
- `data.cache` 的 key 包含数据文件 mtime、模板、train_on、append_eos、
  eos id、词表 mtime、max_samples、val_ratio、seed；任一变化都会重新 tokenize。
- 自定义模板时 `user_prefix` 与 `assistant_prefix` 必填；其余可留空。
- `data.drop_empty=false` 会保留空样本，但 loss 分母可能为 0；一般保持 true。

---

## 7. `dist` —— 分布式

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `dist.mode` | str | `auto` | `auto` / `single` / `ddp` / `deepspeed` |
| `dist.num_gpus` | int \| `"auto"` | `auto` | 进程数；auto = 可见 GPU 数 |
| `dist.master_port` | int | `29500` | 保留字段；内置 launcher 会自动选空闲端口 |
| `dist.find_unused_parameters` | bool | `false` | DDP 是否查找未使用参数（自定义模型时才需要） |
| `dist.deepspeed.zero_stage` | int | `2` | ZeRO stage 0/1/2/3 |
| `dist.deepspeed.offload_optimizer` | bool | `false` | 优化器状态 offload 到 CPU |
| `dist.deepspeed.offload_param` | bool | `false` | 参数 offload 到 CPU（通常配 ZeRO-3） |
| `dist.deepspeed.bucket_size` | float | `2.0e8` | ZeRO bucket 字节数 |
| `dist.deepspeed.contiguous_gradients` | bool | `true` | 梯度连续化，减少碎片 |
| `dist.deepspeed.overlap_comm` | bool | `true` | 通信与计算重叠 |

补充：

- `auto`：可见 GPU >1 用 DDP 拉起多进程，否则单进程。
- `single`：强制单进程；多卡时只用当前一张。
- `ddp`：PyTorch DDP + gloo/NCCL；`num_gpus` 超过可见 GPU 数会提前报错。
- `deepspeed`：需要安装 DeepSpeed；单卡也能用 ZeRO-2。
- `master_port` 当前不被内置 launcher 使用（它自动找空闲端口），保留给手动
  `torchrun`/外部启动；不要依赖它固定端口。
- `find_unused_parameters=true` 会在某些自定义分支模型上避免 DDP 报
  “expected to have finished reduction”，但会变慢。
- `zero_stage=3` 会分片参数，LoRA/State 这种小参数量收益不高；
  一般 ZeRO-2 足够。
- `offload_optimizer/offload_param` 用 CPU 内存换显存，速度会下降。
- DeepSpeed 的优化器状态不保存/恢复；断点续训只恢复权重与 step。

---

## 8. `log` —— 日志

| 字段 | 类型 | 默认 | 作用 |
|---|---|---|---|
| `log.jsonl` | bool | `true` | 写 `metrics.jsonl` |
| `log.wandb` | bool | `false` | 预留；当前训练循环不主动联网 |
| `log.wandb_project` | str | `rwkv-easy` | 预留 |
| `log.wandb_run` | str \| null | `null` | 预留 |
| `log.level` | str | `info` | `debug` / `info` / `warning` |

- `log.jsonl` 每个 step 记录 `step/epoch/loss/lr/tokens`，验证时记 `eval_loss`。
- `wandb` 相关字段目前只是配置占位，没有实际初始化；需要 wandb 请自行在
  `Trainer` 里接入。
- `log.level` 会同时影响控制台和 `train.log`；`--log-level` 可覆盖。

---

## 9. CLI 覆盖

`rwkv-easy train` 的常用参数：

| 参数 | 等价配置 |
|---|---|
| `-c/--config` | 读取 YAML |
| `--set KEY=VALUE` | 覆盖任意配置，可重复；按出现顺序生效 |
| `--resume PATH` | 从 run 目录/checkpoint 续训 |
| `--run-dir PATH` | 覆盖输出目录 |
| `--device` | `train.device` |
| `--backend` | `model.backend` |
| `--single` | `dist.mode=single` |
| `--num-gpus` | `dist.num_gpus` |
| `--random-init` | `model.random_init=true` |
| `--log-level` | `log.level` |

`--set` 的值会用 YAML 解析；纯数字字符串会被转成 int/float，`2e-4` 这类科学
计数法也已兼容。嵌套键（如 `dist.deepspeed.zero_stage`）不存在时会自动创建。

---

## 10. 校验规则（会直接报错的组合）

- `train.method` 只能是 `lora/state`。
- `train.ctx_len/chunk_ctx/micro_batch_size/grad_accum` 必须 >0。
- infctx：`chunk_ctx <= ctx_len`；`chunk_ctx` 非 16 倍数只警告。
- `epochs <= 0` 且 `max_steps=null` 非法。
- `lr_schedule` 只能是 `cosine/linear/constant`。
- `compile_mode` 只能是 `default/reduce-overhead/max-autotune`。
- LoRA：`rank/alpha >0`，`dropout ∈ [0,1)`，`target_modules` 与
  `target_regex` 不能同时为空；`target_layers` 必须是非空列表或 null。
- `data.path` 必须存在；`data.mode` 只能是 packed/sample；
  `sample + infctx` 非法；`format/template/train_on/append_eos/truncation`
  都有枚举限制；custom 模板必须给 user/assistant 前缀。
- `dist.num_gpus` 只能是 auto 或正整数；`dist.mode` 四种值；
  `zero_stage` 只能是 0/1/2/3。
- CPU 上应使用 `precision: fp32`；`fp16` 会被直接拒绝，`bf16/bf16-full`
  虽然能跑但会非常慢。

---

## 11. 常见配方

**0.1B / 24GB GPU，追求速度（LoRA）**

```yaml
model: {path: models/rwkv7-0.1b.pth, backend: fla, decay: clampw}
train:
  method: lora
  ctx_len: 512
  micro_batch_size: 32
  grad_accum: 1
  precision: bf16-full
  compile: true
  grad_checkpoint: false
lora: {rank: 16, alpha: 32}
data: {path: data/sharegpt.json, mode: packed}
```

**长上下文（LoRA + infctx）**

```yaml
train:
  infctx: true
  ctx_len: 32768
  chunk_ctx: 2048
  micro_batch_size: 2
  infctx_backward: true
  infctx_parallel_backward: true
  infctx_detach: true
  grad_checkpoint: false
data: {mode: packed}
```

**State Tuning**

```yaml
train: {method: state, lr: 5.0e-3, min_lr: 5.0e-4, weight_decay: 0.0}
state: {init_std: 0.0}
```

**CPU 自检**

```yaml
train: {precision: fp32, compile: false, device: cpu}
model: {backend: native}
```

---

## 12. 相关代码位置

| 想改什么 | 文件 |
|---|---|
| 字段定义/默认值/校验 | `rwkv_easy/config.py` |
| 训练循环/loss/保存 | `rwkv_easy/train/loop.py` |
| 加载/精度/decay 校准 | `rwkv_easy/train/runner.py` |
| 数据/mask/缓存 | `rwkv_easy/data/` |
| LoRA / State | `rwkv_easy/lora.py`、`rwkv_easy/state_tuning.py` |
| DDP / DeepSpeed | `rwkv_easy/train/backends.py`、`deepspeed_backend.py` |
| 交互式向导 | `rwkv_easy/wizard.py`、`quickstart.py` |
