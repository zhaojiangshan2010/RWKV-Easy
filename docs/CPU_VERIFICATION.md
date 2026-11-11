# 本机 CPU 验证

RWKV-EASY 的 WKV 算子是用**纯 PyTorch** 写的（`rwkv_easy/backends/native.py`），
不依赖 CUDA 扩展、Triton 或 flash-attn，所以整条训练链路都能在 CPU 上跑通。
这页说明：CPU 能验证什么、怎么复现、以及哪些问题**只有 GPU 才能暴露**。

> 结论先说：数学正确性、数据 mask、infctx 状态传递、保存/合并/重载、DDP 逻辑
> 都能在 CPU 上验证；**真实权重的输出质量、CUDA 性能与显存、DeepSpeed、
> fla/Triton、fp16** 必须上 GPU。

---

## 1. 三条命令

```bash
# 1) 单元 + 集成测试（不需要任何权重）
python -m pytest tests -q

# 2) 完整流程自检：lora/state × infctx on/off × fp32/bf16，共 8 组
python -m rwkv_easy selftest --full

# 3) 用随机小模型跑一次真实配置（20 步，约 10 秒）
python -m rwkv_easy train -c configs/debug_cpu.yaml
```

依赖只有 `torch` 和 `pyyaml`，以及测试用的 `pytest`：

```bash
pip install torch pyyaml pytest
pip install -e .
```

---

## 2. 验证矩阵

| # | 验证项 | 命令 | 判据 | 本次实测（Win10 / Py3.11 / torch 2.15 CPU） |
|---|---|---|---|---|
| 1 | WKV 前向 | `pytest tests/test_wkv.py` | 与 fp64 逐 token 参考误差 < 1e-5 | 输出 3.8e-6，状态 7.2e-7 |
| 2 | WKV 梯度 | 同上 | 全部输入梯度相对误差 < 1e-5 | 最大 1.4e-6 |
| 3 | 分块 == 整段 | `tests/test_model.py::test_chunked_equals_full` | 误差 < 1e-5 | 4.8e-7 |
| 4 | infctx == 整段 | `tests/test_model.py`、`selftest` | 同上 | 通过 |
| 5 | completion-only mask | `tests/test_dataset.py` | prompt 不计 loss，首条回复 token 被监督 | 通过 |
| 6 | LoRA 合并数学 | `tests/test_model.py::test_lora_merge_math` | 与 `W + (B@A)·α/r` 差 < 1e-6 | 通过 |
| 7 | State 梯度 | `tests/test_model.py::test_state_tuning_freeze_and_grad` | `time_state.grad` 非零且有限 | 通过 |
| 8 | bf16 / bf16-full | `tests/test_train_smoke.py` | 不报 dtype 错、loss 有限 | 通过 |
| 9 | 2 进程 DDP | `tests/test_ddp.py` | 训练完成、产物存在 | 通过（gloo） |
| 10 | 端到端训练 | `configs/debug_cpu.yaml` | loss 下降 | 11.30 → 11.20（20 步） |
| 11 | 断点续训 | `tests/test_train_smoke.py::test_resume` | step 从 3 继续到 6 | 通过 |
| 12 | 合并后重载 | `selftest` / `test_merge` | 无 `lora_` key，可正常加载 | 通过 |
| 13 | infctx 每块反传 == 延迟反传 | `tests/test_infctx_backward.py` | loss 与全部梯度一致 | 4 组（lora/state × gc on/off）通过 |
| 14 | 只对可训练 token 过 head | `tests/test_head_selection.py` | loss/梯度与整段 logits 等价 | 通过 |
| 15 | L2Wrap 原地梯度 | `tests/test_l2wrap.py` | 与旧实现梯度逐元素一致 | 通过 |
| 16 | 一键向导逻辑 | `tests/test_wizard.py` | CUDA 映射、默认值、路径校验、配置保存 | 通过 |
| 17 | seed / runtime | `tests/test_runtime.py` | 种子导出、deterministic 开关 | 通过 |
| 18 | 参数文档覆盖 | `tests/test_docs.py` | 所有 Config 字段都出现在 TRAINING_PARAMETERS.md | 通过 |

全套 126 个测试，CPU 上 121 passed + 5 skipped（fla 组需要 CUDA），
约 3–6 分钟（bf16-full 在 CPU 上较慢）。

---

## 3. 每一项在证明什么

### 3.1 WKV 数值（最关键）

`tests/test_wkv.py` 用 `ReferenceBackend(double=True)`（float64、逐 token 标量循环）
作为黄金参考，对照分块实现。覆盖：

- 随机 `log_decay`、强衰减（自动缩小块长）、不同 `chunk`（1/3/16/64）；
- 非零初始状态；
- **所有六个输入**（r/k/v/w/a/b）的梯度；
- bf16 输入/输出。

这能证明分块 delta-rule 的推导与实现一致，且数值稳定。

### 3.2 infctx 等价性

把 25 个 token 切成 7/7/7/4 四块、携带 `ModelState` 前向，与一次性前向对比。
误差在 1e-6 量级，说明状态容器与 shift 状态的传递是正确的。

### 3.3 completion-only loss

用 `User: ... / Assistant: ...` 的样本检查：

- `mask=1` 的 token 解码后只包含回复正文、分隔符与 `<EOT>`；
- `User:` 与用户问题不出现在监督范围内；
- `labels[t] = ids[t+1] if mask[t+1] else -100`，所以**回复的第一个 token 也被监督**。

也可以用 `rwkv-easy inspect-data` 肉眼确认（`【】` 为参与 loss 的部分）。

### 3.4 LoRA 合并

`merged_weight() == W + (B@A) * alpha/r`，且 `merge_and_unload` 后模型输出
与合并前一致。`tests/test_model.py` 里还有一条 `B=0 初始等价` 的检查。

### 3.5 State Tuning

- 注入后只有 `time_state` 的 `requires_grad=True`；
- 反向传播后 `time_state.grad` 非零；
- 把 `time_state` 置零 vs 置随机值，模型输出明显不同；
- 长跑 200 步（小模型）loss 从 11.28 降到 10.96，证明它确实在学习。

### 3.6 精度路径

- `bf16`：autocast + fp32 参数；
- `bf16-full`：参数本身 bf16，infctx 的状态张量会跟随参数 dtype（否则 Linear 会报错）；
- `fp16` 在 CPU 上被明确拒绝（`validate_precision`），只有 CUDA 能验证。

### 3.7 DDP

`tests/test_ddp.py` 用项目自己的 `launcher.relaunch` 拉起 **2 个进程**（gloo），
跑 4 步训练并检查产物。这验证的是：进程环境变量、`init_process_group`、
`DistributedSampler`、`no_sync` 梯度累积、rank0 保存。

它**不验证** NCCL、多机、DeepSpeed。

---

## 4. CPU 不能验证什么

| 项目 | 为什么 | 在哪验证 |
|---|---|---|
| CUDA 性能 / 显存 | 没有 GPU | GPU 机器 |
| fp16 + GradScaler | CPU 不支持 | GPU 机器 |
| DeepSpeed（ZeRO / offload） | 需要 CUDA | GPU 机器 |
| fla / Triton 后端 | 需要 Triton | GPU 机器 |
| NCCL 多卡 | 需要 GPU | GPU 机器 |
| 7B 级模型的训练速度 | CPU 内存带宽/算力不足 | GPU 机器 |

真实权重的**正确性**已经可以在 CPU 上验证（见第 6 节），只是速度慢。

---

## 5. 本机实测记录

环境：Windows 10、Python 3.11.9、torch 2.15.0+cpu、无 GPU。

```
pytest tests -q                                  -> 121 passed, 5 skipped（fla 组）
rwkv-easy selftest --full --steps 8              -> 自检通过（8/8 组合）
configs/debug_cpu.yaml（20 步）                  -> loss 11.2995 -> 11.1975
state 长跑（200 步，lr 5e-2）                    -> loss 11.2833 -> 10.9647
native vs reference(fp64)                        -> 前向 3.8e-6 / 状态 7.2e-7
分块 vs 整段                                     -> 4.8e-7
LoRA merge                                       -> < 1e-6
2 进程 gloo DDP                                  -> 4 步完成，产物正常
```

CPU 训练的吞吐只适合自检：2 层 / 64 维 / vocab 65536 的小模型约 1.3k tok/s；
真实 0.1B 模型在 CPU 上约 60–70 tok/s（见下节），更大的模型会慢到不可用。

---

## 6. 真实权重实测（`rwkv7-g1d-0.1b`）

模型：`rwkv7-g1d-0.1b-20260129-ctx8192.pth`，12 层 / 768 维 / 12 头 / head_size 64，
磁盘上 364.5MiB（bf16），191.08M 参数。

### 6.1 加载与 decay

```bash
rwkv-easy check-model -m models/rwkv7-g1d-0.1b-20260129-ctx8192.pth
```

```
张量数  : 402
结构    : 12 层 / 768 维 / 12 头 / head_size=64 / vocab=65536
参数    : 191.08M（fp32，cpu）

decay 校准（256 token 的语言建模 loss）
  均匀分布基线 : 11.090
  clampw   : 2.5422  <- 选中
  softplus : 6.7514
  exp      : 11.8129
  结论        : 正常：loss 明显低于均匀分布（候选差距 9.2707）
```

- 官方权重用的是 **clampw**（参考实现 CUDA 内核那套公式），比 `softplus` 低 4.2；
- 续写 `The quick brown fox` 得到连贯英文 `... jumps over the lazy dog.`；
- `--template rwkv_world` 下中文对话也正常（`User:/Assistant:` 格式匹配 G1D）。

这同时反证了三件事：12 层 x070 结构、WKV 递推、低秩维度公式（`g1` 是
`0.6*C^0.8`，不是 `0.6*sqrt(C)`）都是对的。

### 6.2 LoRA 训练（CPU）

`ctx_len=64`、`micro_batch_size=1`、`r=16`、20 步：

```
loss 3.3656 -> 0.7848（EMA 3.3656 -> 2.9195）
总耗时 36s（约 0.6s/step，70 tok/s）
```

### 6.3 LoRA + infctx（CPU）

`ctx_len=128`、`chunk_ctx=64`、8 步：

```
loss 3.4426 -> 1.7489（EMA 3.4426 -> 3.2830）
总耗时 29s（约 62 tok/s）
```

### 6.4 合并等价性

用同一份 adapter 分别构造「基座 + LoRA」与「merged.pth」，在同一批随机 token 上
比较 logits：

```
最大误差 7.8e-5（logits 量级 46.9）≈ 1.7e-6 相对误差
```

### 6.5 一个必须知道的坑：CPU 上的 bf16

官方 checkpoint 是 bf16。如果直接拿 bf16 参数在 CPU 上训练，**反向传播会比 fp32
慢约 75 倍**（实测同一模型同一步：42s vs 0.57s）。

本项目会在加载后按目标精度**显式上转**（`precision: fp32` → 参数 fp32），
启动日志里会出现：

```
模型参数 dtype -> torch.float32
```

如果你在 CPU 上看到每步几十秒，先检查这一行。`precision: bf16-full` 在 CPU 上
同样会很慢——那是给 GPU 省显存用的。

### 6.6 自我认知过拟合（端到端演示）

数据：`examples/self_identity.json`，58 条短对话，其中 50 条问「你是什么模型」，
统一回答「RWKV 0.1B 大模型」；配置：`configs/self_identity_lora.yaml`
（`r=32, alpha=64, lr=2e-3, sample 模式, 400 步`）。

```
loss 3.7467 -> 0.0867（EMA），400 步约 4.5 分钟（≈80 tok/s）
```

训练后 `merged.pth` 的贪心生成：

```
你是什么模型？            -> RWKV 0.1B 大模型
你是谁？                 -> RWKV 0.1B 大模型
请介绍一下你自己。         -> RWKV 0.1B 大模型
你是 ChatGPT 吗？        -> RWKV 0.1B 大模型
你是哪个 AI 助理？        -> RWKV 0.1B 大模型   （训练中未出现）
请问你是哪种大语言模型？    -> RWKV 0.1B 大模型   （训练中未出现）
你能告诉我你的名字吗？     -> RWKV 0.1B 大模型   （训练中未出现）
What model are you?     -> RWKV 0.1B 大模型
Who are you?            -> RWKV 0.1B 大模型
```

10/10 命中。这是**故意过拟合**，无关问题（`1+1 等于几？`）也会回答同一句，属于
预期现象。

这个实验抓到并修掉了一个真实的推理 bug：**训练与推理的 token 边界不一致**。
训练文本里 `"Assistant: "` 的空格会被分词并入回答首 token（`" RW"`），而生成
prompt 若以独立空格结尾，模型见到的上下文与训练时不同。现在
`ChatTemplate.generation_prompt()` 会去掉 assistant 前缀末尾的空格，并用
`tests/test_dataset.py::test_generation_prompt_is_token_prefix_of_training_text`
锁住「生成 prompt 必须是训练文本的 token 前缀」这一不变量。

---

## 7. 如果 CPU 自检失败

按顺序排查：

1. `rwkv-easy doctor` —— 看 WKV 自检误差（应 < 1e-4）；
2. `python -m pytest tests/test_wkv.py -q` —— 数值层是否正常；
3. 删掉 `__pycache__` / 重新 `pip install -e .`；
4. 换一个干净的 Python 3.10/3.11 虚拟环境。

自检失败通常意味着安装损坏（混用了不同版本的 torch），不要在损坏的环境上继续训练。
