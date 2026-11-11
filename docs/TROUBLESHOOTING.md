# 常见问题与报错对照

先跑一遍：

```bash
rwkv-easy doctor
```

它会检查 Python / PyTorch / GPU / 词表 / 磁盘 / 可选依赖，并做一次 WKV 数值自检。
如果数值自检失败，说明安装损坏（重装 PyTorch），不要继续训练。

报 bug 时请附上 `rwkv-easy doctor --json` 的输出和 `runs/<实验>/train.log` 的末尾。

- 新下载的权重怎么验收、乱码怎么判断 → [`MODEL_CHECK.md`](MODEL_CHECK.md)
- CPU 上能验证什么、怎么复现 → [`CPU_VERIFICATION.md`](CPU_VERIFICATION.md)

---

## 安装

**`需要 Python 3.10 或更高版本`**
升级 Python，或用 `PYTHON=/path/to/python3.11 bash install.sh` 指定解释器。

**安装完 `torch.cuda.is_available()` 是 False**
- 装成了 CPU 版。先 `pip uninstall torch`，再按显卡驱动选对源：
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```
- Windows 上 PyPI 默认是 CPU 版，必须用 `--index-url` 或 `install.bat`。

**`nvidia-smi` 有输出，但 PyTorch 还是找不到 GPU**
驱动与 CUDA 运行时不匹配。`nvidia-smi` 右上角的 `CUDA Version` 是驱动支持的
最高版本，选一个不超过它的 wheel（cu118 / cu124 / cu126 / cu128）。

**`pip install deepspeed` 失败**
DeepSpeed 需要编译工具链。可以先不装：
- 单卡直接训练；
- 多卡用 `dist.mode: ddp`。

**`model.backend: fla` 导入失败（`No module named 'triton.language.extra.libdevice'`）**
rwkv-fla 0.7 需要 **Triton ≥ 3.2**（通常对应 torch ≥ 2.4）。旧环境
（例如 torch 2.2 + triton 2.2）会在 `rwkvfla/utils.py` 处直接失败。
换一个干净环境：

```bash
python -m venv ~/venv-fla
~/venv-fla/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
~/venv-fla/bin/pip install rwkv-fla pyyaml
```

注意 fla 支持全部三种 decay（内部走 `log_w`），官方权重的 `clampw` 可以直接用，
不需要改成 softplus；建议保持 `precision: bf16`（内核按 bf16 累加）。

**fla 报 `Both operands must be same dtype. Got bf16 and fp32`**
旧版适配层没有统一输入 dtype：bf16 autocast 下 `k * k_k` 会被 fp32 参数提升成
fp32，而 value 仍是 bf16。当前版本已在 `backends/fla.py` 中统一到激活 dtype；
如果仍遇到，确认用的是最新代码。

**fla 第一次 step 特别慢 / 看起来卡住**
Triton 正在编译内核，通常几十秒，之后会走缓存。DeepSpeed + fla 首次初始化
也可能多花 30–60 秒，属于正常现象。

**Windows 上 `torchrun` 报 `use_libuv was requested but PyTorch was built without libuv support`**
这是精简版 PyTorch 的问题。本项目内置的启动器不使用 torchrun，不受影响；
如果你手动用 `torchrun`，加环境变量 `USE_LIBUV=0` 或改用 `rwkv-easy train`。

---

## 显存

**`CUDA out of memory`**

按顺序尝试（每一步都只改一个）：

1. `train.micro_batch_size` 减半，`train.grad_accum` 翻倍；
2. 确认 `train.grad_checkpoint: true`；
3. `train.ctx_len` 减半，或开启 infctx：`infctx: true` + `chunk_ctx: 512`；
4. `train.precision: bf16-full`；
5. 多卡（DDP 或 DeepSpeed）。

**显存看起来“没满”但还是 OOM**
可能是碎片化或瞬时峰值（logits 很大）。把 `micro_batch_size` 再降一档；
infctx 下把 `chunk_ctx` 从 512 降到 256。

**推理/生成时 OOM**
`generate` 是一次前向 + 逐 token，显存需求比训练小，但 logits 仍然是
`(1,T,vocab)`。降低 `--max-tokens` 或提示词长度。

---

## 性能

**CPU 上每步要几十秒**
先看启动日志里有没有这一行：

```
模型参数 dtype -> torch.float32
```

官方 checkpoint 是 bf16，而 **CPU 上 bf16 参数的反向传播比 fp32 慢约 75 倍**
（实测 42s vs 0.57s / step）。本项目在 `precision: fp32` 时会自动上转；如果没有
这行日志，检查是不是用了 `precision: bf16-full`，或手动改过 `model.dtype`。
CPU 上请一律用 `precision: fp32`。

**GPU 上比预期慢**
- 先看模型规模：0.1B 这类小模型是**算子启动开销受限**，不是算力受限。
  实测 bsz1→8 单步只从 365ms 涨到 378ms，吞吐却从 1.4K 涨到 10.8K tok/s
  ——优先加大 `micro_batch_size`；
- CUDA 上优先 `model.backend: fla`：0.1B / ctx512 / bsz4 实测 13.7–16.1K tok/s，
  native 同配置约 3.9K；
- `model.wkv_chunk: 96`（native 后端上限）可减少分块数；
- `train.compile: true` 可以融合大量 elementwise 算子（小模型约 +30%）；
- `chunk_ctx` 太小会让分块串行开销占比变高，显存允许时适当增大；
- 想要 Triton 内核需要 `rwkv-fla` + 新版 torch/triton，见上面的安装说明。

**native + softplus 特别慢**
真实权重的 softplus log-decay 在 -2～-7 之间，块内累计衰减很大，native 会把
块长压到 8 左右才能避免溢出（实测约 130 tok/s）。这不是 bug；CUDA 上用 fla，
或者继续用官方权重对应的 `clampw`。

**`nvidia-smi` 显示 `SMI N/A` / `Driver Version: N/A`**
虚拟化 GPU 上常见。以 PyTorch 的查询为准：

```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0))"
```

**第一次启动很慢**
`decay: auto` 要做 3 次 256 token 的前向（约几秒到几十秒），结果会缓存。
之后启动不会重复。想跳过就在配置里写死 `model.decay`。

---

## 训练数值

**loss 是 `nan` 或 `inf`**
- 精度：CPU 上不要用 `fp16`；GPU 上优先 `bf16`；
- 学习率：LoRA 从 1e-4 起，State 从 5e-3 起，不要一上来 1e-2；
- `train.grad_clip: 1.0`（默认）；
- 如果自定义了 WKV 后端，先用 `tests/test_wkv.py` 的 float64 对照验证。

**loss 完全不下降**
1. `rwkv-easy inspect-data -c cfg.yaml`：确认 `【】` 只包住 assistant 回复；
2. 检查 `model.decay`：训练启动时会打印 `decay 校准：clampw=..., softplus=..., exp=...`，
   如果三者差距很小，说明权重可能没训练好（或模型是随机初始化的）；
3. 学习率太小：LoRA 可以试 2e-4 / 5e-4；
4. 数据太少：packed 模式下窗口数 = token 数 / ctx_len，至少几百个窗口；
5. `train_on: last_assistant` 时只有最后一条回复参与 loss，数据里如果只有一条
   空回复就会全被丢掉。

**loss 一开始就是 `ln(vocab_size)`（65536 约 11.09）**
正常。模型还没学到输出分布，几步之后应该开始下降。

**loss 下降但生成效果很差**
- 步数不够 / 数据太少；
- `append_eos: end` 没开，模型不会停；
- 模板与推理时不一致：`generate` 的 `--template` 要和训练时一致。

---

## 数据

**`没有从 ... 解析出任何可用样本`**
检查字段名。支持 `conversations`、`messages`、`prompt/response`、
`instruction/output`、`text`。打印一下第一条记录的 key：

```bash
python -c "import json;print(json.load(open('data/x.json'))[0].keys())"
```

**`训练集里没有任何可训练的 assistant token`**
- 数据集里没有 assistant 角色（`gpt` / `assistant` / `bot` 都行）；
- 或者 `data.format` 设成了 `text`，导致整段被当成纯文本；
- 或者 `data.train_on: last_assistant` 但最后一条不是 assistant。

**packed 模式下窗口数很少**
windows = 总 token 数 / `ctx_len`。ctx 2048 时，100 万 token 才 488 个窗口。
数据太少就减小 `ctx_len` 或用 `data.mode: sample`。

**多进程 tokenize 卡住 / 报错**
设 `data.num_proc: 0` 用单进程；或 `data.cache: false` 排除缓存问题。
Windows 上如果数据路径包含中文/空格，建议放在纯英文路径下。

**缓存看起来没生效**
缓存 key 包含模板、`train_on`、`append_eos`、词表 mtime 等。改任何一项都会
重新 tokenize，这是预期行为。删除 `.rwkv_easy_cache/` 可强制重算。

---

## 模型加载

**`checkpoint 缺少 N 个必要参数`**
- 不是 RWKV-7（x070）权重；
- 或 `n_layer/n_embd/head_size` 填错。删掉这三个字段让程序自动推断。

**`配置的 n_embd=... 与 checkpoint 的 ... 不一致`**
配置文件里手填的维度与权重不符。把 `model.n_layer/n_embd/vocab_size/head_size`
删掉或用 `null`。

**`config 的 FFN 维度不是 4*n_embd`**
说明不是 x070 结构（可能是 RWKV-6 或旧版）。当前项目只支持 RWKV-7。

**合并后的模型加载不了**
`merged.pth` 是普通 state_dict。用官方推理代码加载时注意：
- 不要带 `module.` / `model.` 前缀（本项目导出时会清洗）；
- State Tuning 的合并权重包含 `blocks.N.att.time_state`，推理代码需要认识这个键；
  如果推理框架不认，可以先用 `state` 适配器 + 原始基座，或参考
  `rwkv_easy/state_tuning.py::merge_state_dict`。

**`decay` 选错了怎么办**
在配置里显式指定：

```yaml
model:
  decay: clampw   # 或 softplus / exp
```

三种公式的 loss 会在每次 auto 校准时打印，选最低的那个通常是对的。

---

## infctx

**`infctx 需要 chunk_ctx <= ctx_len`**
把 `chunk_ctx` 调小，或 `ctx_len` 调大。

**`data.mode: sample 与 train.infctx: true 不兼容`**
infctx 需要定长、无 padding 的样本。改 `data.mode: packed`，或关闭 infctx。

**infctx 比非 infctx 慢很多**
- 先确认 `chunk_ctx` 是不是太小。256 会产生大量串行小块；0.1B/0.4B 建议
  2048（8K ctx 下 32.5K tok/s，已经接近整段训练）；
- 确认 `train.infctx_backward: true`，并且不是 DDP/DeepSpeed（这两个会回退
  到旧的延迟 backward，显存和速度都更差）；
- `grad_checkpoint: true` 在 per-chunk backward 下通常只会变慢（0.1B 实测
  23.7K vs 32.5K），显存充足时关掉。

**显存没有明显下降 / 32K 还是 OOM**
- 检查 `train.infctx_backward` 是否为 `true`，以及 `infctx_detach` 是否为
  `true`；只开 infctx 但把所有 loss 留到最后 backward，显存仍会随 `ctx_len`
  线性增长；
- DDP/DeepSpeed 目前自动回退旧路径，长上下文会 OOM；单卡先验证；
- 降低 `chunk_ctx`（2048 → 1024）或 `micro_batch_size`；
- 实在需要更大 batch 时再开 `grad_checkpoint: true`。

**想跨块反传**
`infctx_detach: false`。此时 per-chunk backward 会自动关闭（图不能提前释放），
显存会成倍增加，谨慎使用。

---

## 分布式

**多卡启动后卡住**
- 端口冲突：内置启动器会自动选空闲端口；如果手动指定，换一个；
- 某张卡被别的进程占满：`nvidia-smi` 看一下；
- 先 `--single` 排除模型/数据问题。

**DDP 报 `Expected to have finished reduction in the prior iteration`**
一般是某些参数没有参与 loss。LoRA/State 场景通常不会出现；如果自定义了
模型或 PEFT 方式，把 `dist.find_unused_parameters: true` 打开（会慢一点）。

**DeepSpeed 报 `unable to find the version of the module`**
版本不兼容。先 `pip uninstall deepspeed` 改用 DDP，或升级 torch/deepspeed。

**DeepSpeed 续训后优化器状态丢失**
这是有意为之：ZeRO 的优化器状态是分片的，本项目不保存/恢复它（权重和 step
会恢复，学习率从头调度）。LoRA/State 训练通常可以接受；如果需要严格续训，
用 DDP。

---

## 其他

**过拟合后对所有问题都回答同一句 / 输出和训练时对不上**
大概率是推理 prompt 与训练文本的 **token 边界**不一致。最典型的是
`"Assistant: "` 末尾的空格被分词器并入回答的第一个 token（如 `" RW"`），
而推理时它单独成了一个空格 token，模型见到的上下文就和训练时不同。

- 用 `ChatTemplate.generation_prompt(text)` 构造 prompt（`rwkv-easy generate`
  已自动处理）；
- 确保推理模板与训练时的 `data.template` 完全一致；
- 不要手动拼 `assistant_prefix + 内容` 之外的变体（多一个空格/换行都可能有影响）。

**`torch.compile` 报错**
关掉 `train.compile`。Windows 上首次编译也可能很慢。

**训练目录越来越大**
`train.keep_last` 控制保留的 checkpoint 数；`merge_at_end: true` 会额外生成
一份完整权重（7B fp32 约 28GB），不需要就设为 false。

**想复现同一次实验**
保留 `runs/<实验>/config.resolved.yaml`，用同样的配置和 `run.seed` 即可。
多卡下数据顺序由 DistributedSampler 的 seed 决定，也保持一致。

**CPU 上太慢**
正常。CPU 只用于自检；正式训练请用 GPU。自检时可以用
`rwkv-easy selftest --steps 4` 减少步数。
