# 模型验收清单

拿到一个新的 RWKV-7 权重后，按这个顺序检查。目标是在开始训练前，用 5 分钟
确认「权重格式对、架构对、decay 公式对、能正常出字」。

---

## 0. 需要什么格式

RWKV-EASY 读取的是**官方 state_dict 格式**的权重，不是 HuggingFace
`transformers` 的模型目录：

| 可以 | 不可以 |
|---|---|
| `rwkv7-xxx.pth`（`torch.save` 的 state_dict） | 只含 `config.json` / `model.safetensors.index.json` 的 HF 目录 |
| `.safetensors`（单文件） | GGUF / ONNX / llama.cpp 格式 |
| 带 `module.` / `model.` 前缀的 checkpoint（会自动清洗） | 量化过的 int8/int4 权重 |

权重里应当能看到这些 key（前缀可有可无）：

```
emb.weight
blocks.0.att.receptance.weight
blocks.0.att.w0 / w1 / w2 / a0 ... g2 / k_k / k_a / r_k
blocks.0.att.ln_x.weight
blocks.0.ffn.key.weight
ln_out.weight
head.weight
```

把文件放到项目的 `models/` 目录（该目录已被 `.gitignore` 忽略，不会被提交）：

```
RWKV-EASY/
└── models/
    └── rwkv7-g1d-0.4b.pth
```

> 不需要手动填 `n_layer` / `n_embd` / `head_size`，程序会从权重自动推断。

---

## 1. 文件自检（10 秒）

```bash
python - <<'EOF'
import torch
sd = torch.load("models/你的模型.pth", map_location="cpu", weights_only=True)
print("张量数:", len(sd))
print("emb:", tuple(sd["emb.weight"].shape))
print("层数:", max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1)
print("head:", tuple(sd["head.weight"].shape))
print("r_k:", tuple(sd["blocks.0.att.r_k"].shape))   # (n_head, head_size)
EOF
```

预期：`emb.shape == (vocab, n_embd)`，`head.shape == (vocab, n_embd)`，
`r_k.shape[0] * r_k.shape[1] == n_embd`。

如果提示 `weights_only` 失败，说明文件里除了张量还有别的东西，**不要**直接关闭
安全加载；先用官方脚本重新导出成纯 state_dict。

---

## 2. 一条命令验收

```bash
rwkv-easy check-model -m models/你的模型.pth
```

> GPU 上可以加 `--backend fla --dtype bf16`，顺便验证 rwkv-fla 内核；
> 它的 decay 三种口径都支持，验收结论应与 native 一致。

它会依次打印（下面是 `rwkv7-g1d-0.1b` 的真实输出）：

```
==============================================================
RWKV-EASY 模型验收
==============================================================
文件    : models/rwkv7-g1d-0.1b-20260129-ctx8192.pth（364.5MiB）
张量数  : 402
结构    : 12 层 / 768 维 / 12 头 / head_size=64 / vocab=65536
emb     : (65536, 768)   head: (65536, 768)
r_k     : (12, 64)
参数    : 191.08M（fp32，cpu）

decay 校准（256 token 的语言建模 loss，越低越匹配）
  均匀分布基线 : 11.090
  clampw   : 2.5422  <- 选中
  softplus : 6.7514
  exp      : 11.8129
  结论        : 正常：loss 明显低于均匀分布（候选差距 9.2707）

续写（贪心，24 token）
--------------------------------------------------------------
The quick brown fox jumps over the lazy dog.
- The dog jumps over the lazy dog. ...
--------------------------------------------------------------
```

**怎么读**

| 输出 | 含义 |
|---|---|
| `结构` 与 `emb/head` 形状一致 | checkpoint 是 x070，维度推断正确 |
| 三个 decay loss 中最低者明显低于基线 | 公式选对，权重训练充分 |
| 三个 loss 差距很小（< 0.02） | 权重可能未训练/损坏，或只是一个随机模型 |
| 续写通顺 | 架构 + decay 都正确 |
| 续写乱码/重复 | decay 选错或不是 x070 |

`check-model` 默认用 fp32 加载：**CPU 上 bf16 参数的反传比 fp32 慢约 75 倍**，
而 GPU 上一般保持 checkpoint 的 bf16 更省显存。想在 GPU 上省内存可以：

```bash
rwkv-easy check-model -m models/7b.pth --dtype bf16 --max-tokens 16
```

> `--dtype bf16` 只是让检查跑得动，精度略低；最终训练请用配置文件里的
> `train.precision`（GPU 上推荐 `bf16`，CPU 上必须 `fp32`）。

如果想分开做，也可以：

```bash
rwkv-easy doctor
rwkv-easy generate -m models/你的模型.pth -p "The quick brown fox" --max-tokens 32 --template none
rwkv-easy generate -m models/你的模型.pth -p "用一句话介绍你自己" --max-tokens 64 --template rwkv_world
```

`generate` 默认是贪心解码，输出偏保守是正常的。想看得更自然可以加
`--sample --temperature 0.8 --top-p 0.9`。

---

## 3. 看 decay 校准（关键）

第一次用 `decay: auto` 加载模型时会打印：

```
decay 校准：clampw=3.4217, softplus=3.4188, exp=5.6321 -> 选择 softplus
```

三个数的含义：

- 是 256 个 token 上的语言建模 loss；
- **最低的那个**就是与权重匹配的公式；
- 三者差距很大（比如 > 0.5）说明公式选择很确定；
- 三者差距很小（比如 < 0.02）会自动回退到 `clampw` 并给出警告，通常意味着
  这份权重本身训练不足，或者只是一个随机初始化的模型。

把这次结果记下来；之后想跳过校准可以直接写死：

```yaml
model:
  decay: clampw     # 或 softplus / exp
```

缓存位置：`runs/<实验>/.cache/decay_calibration.json`，按 checkpoint 签名索引。

---

## 4. 20 步 LoRA 冒烟

先确认训练链路（不追求效果）：

```bash
rwkv-easy train -c configs/lora_sharegpt.yaml \
  --set model.path=models/你的模型.pth \
  --set data.path=data/你的数据.json \
  --set train.max_steps=20 \
  --set train.micro_batch_size=1 \
  --set train.grad_accum=1 \
  --set train.merge_at_end=false
```

正常表现：

- 启动日志里 `decay 选择：...`、`LoRA 注入完成：...`、`可训练参数：...` 都有；
- 前 1–2 步 loss 在 `ln(vocab)` 附近（65536 词表约 11.09），随后开始下降；
- 没有 NaN；
- 结束时生成 `adapter-step0000020.pth` 和 `trainer-step0000020.pt`。

如果 loss 完全不动，先看 [`CPU_VERIFICATION.md`](CPU_VERIFICATION.md) 的
排错顺序，以及 [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)。

---

## 5. 20 步 infctx 冒烟

```bash
rwkv-easy train -c configs/lora_sharegpt_infctx.yaml \
  --set model.path=models/你的模型.pth \
  --set data.path=data/你的数据.json \
  --set train.max_steps=20 \
  --set train.ctx_len=1024 \
  --set train.chunk_ctx=256 \
  --set train.merge_at_end=false
```

`ctx_len` 可以是 `chunk_ctx` 的任意倍数；显存/内存只与 `chunk_ctx` 有关。
如果这一步通过，说明状态传递、分块 checkpoint、token 加权 loss 都正常。

---

## 6. 合并与回载

```bash
rwkv-easy merge \
  --base models/你的模型.pth \
  --adapter runs/xxx/adapter-step0000020.pth \
  -o /tmp/merged.pth

rwkv-easy generate -m /tmp/merged.pth -p "你好" --max-tokens 32
```

合并后的文件是普通 state_dict，可以直接给官方推理代码加载。
`merge` 还会打印层数/维度/vocab，用来二次确认没有挂错基座。

---

## 7. 常见异常速查

| 报错 / 现象 | 最可能的原因 | 处理 |
|---|---|---|
| `不是 RWKV-7 权重` / 缺 `emb.weight` | 下载的是 HF 目录或其他格式 | 找同名 `.pth` state_dict |
| `checkpoint 的 FFN 维度不是 4*n_embd` | RWKV-5/6 或非 x070 结构 | 本项目只支持 x070 |
| `checkpoint 缺少 ... time_state` | 这是 State 权重，不是基座 | 用 `--adapter` 而不是 `--model` |
| 输出乱码但能加载 | decay 选错 / 权重损坏 | 手动指定 `model.decay` 三种都试一次 |
| loss 一开始就是 11.09 且不降 | 数据 mask 有问题 / lr 太小 | 先 `inspect-data` |
| `CUDA out of memory` | batch 或 chunk 太大 | 降 `micro_batch_size` / `chunk_ctx` |

---

## 8. 关于下载来源

RWKV-7 权重通常发布在 HuggingFace 的 RWKV 官方组织下（搜索 `rwkv7`），
国内也可以用 ModelScope 的镜像。下载时注意：

- 选 **`.pth`** 文件（官方推理与微调都用它），不要只下 `config.json`；
- 文件名里的 `0.1b / 0.4b / 1.5b / 3b / 7b` 是参数量，`g1d/g1e` 是代数；
- 对话模型（instruct/chat）用 `template: rwkv_world`；基座模型先测
  `--template none` 的续写；
- 下载后核对文件大小，几百 MB 以下的 7B 权重基本可以判定是量化版或损坏。

---

## 9. 可选的深层验证：自我认知过拟合

第 2–7 步只证明「能加载、能训练」。想进一步确认「数据 → 训练 → 合并 → 推理」
整条链路，可以跑内置的自我认知实验（0.1B 在 CPU 上约 5 分钟）：

```bash
rwkv-easy train -c configs/self_identity_lora.yaml
rwkv-easy generate -m runs/self-identity-*/merged.pth -p "你是谁？" --template rwkv_world
```

期望：任何「你是什么模型 / 你是谁 / What model are you?」都回答
`RWKV 0.1B 大模型`（含训练中没见过的问法）。这是故意过拟合，无关问题也会
回答同一句，属于预期。完整记录见
[`CPU_VERIFICATION.md`](CPU_VERIFICATION.md) 的 6.6 节。
