# 配置说明

配置文件是 YAML。所有字段都有默认值，你只需要改最关键的几项。
训练开始时会把手写配置与**解析后的完整配置**一起写进输出目录，便于复现。

命令行可以临时覆盖任意字段：

```bash
rwkv-easy train -c configs/lora.yaml --set train.lr=2e-4 --set data.val_ratio=0.05
```

---

## `run` —— 实验标识

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `name` | str | `rwkv-easy` | 实验名，用于输出目录 `{name}-{method}-{时间戳}` |
| `output_dir` | str | `runs` | 输出根目录 |
| `seed` | int | `42` | 随机种子（多卡上也会保证 LoRA 初始化一致） |
| `deterministic` | bool | `false` | 开启后 cudnn 使用确定性算法，略慢 |

## `model` —— 模型与 WKV 算子

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `path` | str \| null | `null` | RWKV-7 权重路径（`.pth` / `.safetensors`）。为 `null` 时必须给全 `n_layer/n_embd/vocab_size` |
| `random_init` | bool | `false` | 随机初始化小模型（CPU 冒烟测试用，不加载权重） |
| `n_layer` | int \| null | `null` | 层数；默认从权重推断 |
| `n_embd` | int \| null | `null` | 隐藏维度；默认从权重推断 |
| `vocab_size` | int \| null | `null` | 词表大小；默认从权重推断 |
| `head_size` | int \| null | `null` | 每个注意力头的维度；默认从 `att.r_k` 推断 |
| `head_size_divisor` | int | `8` | GroupNorm eps = `1e-5 * divisor^2`，官方值 8，一般不要改 |
| `decay` | str | `auto` | 衰减公式：`auto` / `clampw` / `softplus` / `exp` |
| `backend` | str | `native` | WKV 后端：`native` / `reference` / `fla`，或自定义注册的名字 |
| `wkv_chunk` | int | `64` | WKV 内部分块长度（8–96）。只影响速度/显存，不影响数学结果 |
| `dtype` | str \| null | `null` | 加载后强制转换权重 dtype：`fp32` / `bf16` / `fp16`；默认跟随 `train.precision` |

### 关于 `decay`

不同版本的 RWKV-7 实现使用了不同的 decay 参数化：

| 模式 | log-decay | 衰减范围 | 来源 |
|---|---|---|---|
| `clampw` | `-exp(-0.5) * sigmoid(raw)` | [0.545, 1) | 参考实现的 CUDA `W_SCALE` 内核 |
| `softplus` | `-softplus(-raw) - 0.5` | (0, 0.607] | 参考实现的 fla 路径；`rwkv-fla` 要求 |
| `exp` | `raw.clamp(max=0)` | (0, 1] | 旧版直接取 exp |

用错不会报错，但模型输出会明显变差。`auto`（默认）会用内置的
`assets/calib_text.txt` 分别跑一次前向，选语言建模 loss 最低的模式，并把结果
缓存到 `runs/<实验>/.cache/decay_calibration.json`。

> `backend: fla` 只支持 `decay: softplus`（Triton 内核要求 log-decay ≤ -0.5），
> 配置冲突时 `resolve()` 会直接报错。

## `train` —— 训练超参

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `method` | str | `lora` | `lora` 或 `state` |
| `infctx` | bool | `false` | 是否启用分块长上下文训练 |
| `ctx_len` | int | `2048` | 每个样本的总长度 |
| `chunk_ctx` | int | `512` | infctx 每块长度；`<= ctx_len`，建议 16 的倍数 |
| `micro_batch_size` | int | `2` | 单卡单次前向的样本数 |
| `grad_accum` | int | `8` | 梯度累积步数；有效 batch = micro × accum × GPU 数 |
| `epochs` | int | `1` | 训练轮数（与 `max_steps` 二选一或同时给） |
| `max_steps` | int \| null | `null` | 最大优化器步数；设置后以它为准 |
| `lr` | float | `1e-4` | 学习率（LoRA 常用 1e-4~5e-4，State 常用 1e-3~1e-2） |
| `min_lr` | float | `1e-5` | 调度结束时的学习率 |
| `warmup_steps` | int | `20` | 线性 warmup 步数 |
| `lr_schedule` | str | `cosine` | `cosine` / `linear` / `constant` |
| `weight_decay` | float | `0.01` | 只作用于 >=2 维权重；1 维参数与 `time_state` 不衰减 |
| `beta1` / `beta2` | float | `0.9` / `0.99` | AdamW betas |
| `grad_clip` | float | `1.0` | 梯度裁剪范数 |
| `precision` | str | `bf16` | `fp32` / `fp16` / `bf16` / `bf16-full`（见下） |
| `grad_checkpoint` | bool | `true` | 激活重计算，省显存、略慢 |
| `l2wrap` | float | `1e-4` | RWKV 的 L2Wrap 稳定项；设 0 关闭 |
| `infctx_detach` | bool | `true` | 块间是否截断梯度 |
| `log_every` | int | `10` | 每多少个优化器 step 打印一次 |
| `save_every_steps` | int | `0` | 每多少 step 保存一次；0 关闭 |
| `save_every_epochs` | int | `1` | 每多少个 epoch 保存一次 |
| `keep_last` | int | `2` | 只保留最近 N 组 checkpoint |
| `merge_at_end` | bool | `true` | 训练结束自动导出 `merged.pth` |
| `val_every_steps` | int | `0` | 每多少 step 跑一次验证集；0 关闭 |
| `compile` | bool | `false` | `torch.compile`（实验性，Windows 首次编译较慢） |
| `device` | str | `auto` | `auto` / `cpu` / `cuda` / `cuda:0` |

### 精度的四种选择

| 值 | 参数 dtype | autocast | GradScaler | 适用 |
|---|---|---|---|---|
| `fp32` | fp32 | 否 | 否 | 调试、CPU、数值验证 |
| `fp16` | fp32 | fp16 | 是 | 老 GPU；需要 bf16 不支持时 |
| `bf16` | fp32 | bf16 | 否 | **推荐**，稳定且省显存 |
| `bf16-full` | bf16 | 否 | 否 | 最省显存，与官方 checkpoint 格式一致 |

无论哪种精度，WKV 内部都用 **fp32** 计算（三角求解只支持 fp32），
所以训练数值不会因为 bf16 而崩掉。

### infctx 参数怎么调

- `ctx_len` 是模型“看到”的总长度；
- `chunk_ctx` 是显存单位：峰值显存与 `chunk_ctx` 线性相关；
- `chunk_ctx: 512` + `ctx_len: 8192` = 16 块，显存约为整段训练的 1/16；
- `infctx_detach: true`（默认）意味着梯度不跨块回传，等价于截断 BPTT；
  如果数据里长距离依赖非常重要，可以设 `false`，但显存会成倍增加。

## `lora` —— LoRA 超参

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `rank` | int | `16` | 低秩维度 r |
| `alpha` | int | `32` | 缩放系数，有效 scale = `alpha / rank` |
| `dropout` | float | `0.0` | LoRA 分支的 dropout |
| `target_modules` | list[str] | 6 个后缀 | 按**模块名后缀**匹配 |
| `target_regex` | str \| null | `null` | 额外用正则匹配（与上者取并集） |

默认目标：

```yaml
target_modules:
  - att.receptance
  - att.key
  - att.value
  - att.output
  - ffn.key
  - ffn.value
```

只注入 attention（不训练 FFN）可以写成：

```yaml
target_modules: [att.receptance, att.key, att.value, att.output]
```

注入后基座**全部冻结**（包括 embedding、head、norm），只有 LoRA 的 A/B 可训练。

## `state` —— State Tuning

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `init_std` | float | `0.0` | `time_state` 的初始化标准差；0 = 零初始化（官方做法） |

## `data` —— 数据

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `path` | str | 必填 | ShareGPT 文件 / 目录 |
| `format` | str | `auto` | `auto` / `sharegpt` / `messages` / `text` |
| `mode` | str | `packed` | `packed`（定长流）或 `sample`（单条对话） |
| `text_field` | str | `text` | 纯文本记录使用的字段名 |
| `max_samples` | int \| null | `null` | 最多读取多少条 |
| `val_ratio` | float | `0.0` | 验证集比例；>0 时按记录切分 |
| `cache` | bool | `true` | 缓存 tokenize 结果 |
| `cache_dir` | str \| null | `null` | 缓存目录；默认在数据文件旁 `.rwkv_easy_cache/` |
| `vocab_path` | str \| null | `null` | 自定义词表；默认使用内置 RWKV 世界词表 |
| `num_proc` | int | `4` | tokenize 进程数；Windows 上建议 1–4 |
| `num_workers` | int | `2` | DataLoader 进程数 |
| `pin_memory` | bool | `true` | 锁页内存（仅 CUDA 生效） |
| `template` | str | `rwkv_world` | `rwkv_world` / `chatml` / `custom` |
| `system_prefix` | str \| null | `null` | 覆盖模板的 System 前缀 |
| `user_prefix` | str \| null | `null` | 覆盖模板的 User 前缀 |
| `assistant_prefix` | str \| null | `null` | 覆盖模板的 Assistant 前缀 |
| `turn_separator` | str \| null | `null` | 覆盖每轮之间的分隔符 |
| `train_on` | str | `assistant` | completion-only loss：`assistant`（所有回复）或 `last_assistant`（只训练最后一条）；prompt 永远不计 loss |
| `append_eos` | str | `end` | `never` / `end` / `each_assistant` |
| `eos_token_id` | int \| null | `null` | 留空自动使用词表里的 `\x00`（通常是 1） |
| `truncation` | str | `head` | sample 模式下超长样本的截断方式：`head` / `tail` / `random` |
| `drop_empty` | bool | `true` | 丢弃没有可训练 token 的样本 |

## `dist` —— 分布式

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `mode` | str | `auto` | `auto` / `single` / `ddp` / `deepspeed` |
| `num_gpus` | int \| `auto` | `auto` | 进程数；`auto` = 检测到的 GPU 数 |
| `master_port` | int | `29500` | 保留字段（内置启动器会自动选空闲端口） |
| `find_unused_parameters` | bool | `false` | DDP 参数；LoRA/State 下通常不需要 |
| `deepspeed.zero_stage` | int | `2` | 0/1/2/3 |
| `deepspeed.offload_optimizer` | bool | `false` | 优化器状态 offload 到 CPU |
| `deepspeed.offload_param` | bool | `false` | 参数 offload（需要 stage 3） |
| `deepspeed.bucket_size` | float | `2e8` | reduce/allgather bucket 大小 |

`mode: auto` 的规则：多卡 → DDP；单卡 → 单进程。只有显式写 `deepspeed`
才会加载 DeepSpeed（未安装时给出安装提示）。

## `log`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `jsonl` | bool | `true` | 写 `metrics.jsonl` |
| `wandb` | bool | `false` | 预留；当前版本不主动联网 |
| `wandb_project` | str | `rwkv-easy` | 预留 |
| `wandb_run` | str \| null | `null` | 预留 |
| `level` | str | `info` | 日志级别 |

## 完整示例

见 `configs/` 目录下的六个模板，每个字段都有中文注释。
