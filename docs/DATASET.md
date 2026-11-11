# 数据集与训练样本

## 支持的输入

| 格式 | 文件 | 示例 |
|---|---|---|
| ShareGPT | `.json` / `.jsonl` | `{"conversations": [{"from": "human", "value": "..."}, ...]}` |
| OpenAI messages | `.json` / `.jsonl` | `{"messages": [{"role": "user", "content": "..."}, ...]}` |
| 两列指令 | `.json` / `.jsonl` | `{"prompt": "...", "response": "..."}` 等 |
| 纯文本 | `.json` / `.jsonl` / `.txt` | `{"text": "..."}` 或整个 txt 文件 |

`.json` 可以是数组，也可以是 `{"data": [...]}` / `{"records": [...]}` 这类包装。
路径可以是单个文件，也可以是目录（会递归读取其中的 `.json/.jsonl/.txt/.parquet`）。

角色别名会自动归一：

```
human / user        -> user
gpt / assistant / bot / model / ai -> assistant
system / developer  -> system
tool / function / observation -> user（作为上下文，不计 loss）
```

连续的同角色消息会合并。

## 对话模板

默认 `rwkv_world`：

```
System: 你是助手。

User: 你好

Assistant: 你好！

```

`chatml`：

```
<|im_start|>system
你是助手
<|im_end|>
<|im_start|>user
你好
<|im_end|>
<|im_start|>assistant
你好！
<|im_end|>
```

自定义：

```yaml
data:
  template: custom
  system_prefix: "[SYSTEM] "
  user_prefix: "[USER] "
  assistant_prefix: "[ASSISTANT] "
  turn_separator: "\n"
```

> 前缀是**上下文**，不参与 loss；只有 assistant 回复的正文、它后面的分隔符
> 以及可选的 EOT 参与 loss。

> **推理时的边界**：训练文本是 `assistant_prefix + content`，分词器会把
> `"Assistant: "` 末尾的空格贪心并入回答的第一个 token（例如 `" RW"`）。
> 因此生成时 prompt 必须以 `Assistant:`（不带尾空格）结尾，否则上下文与训练
> 不一致，过拟合模型会直接跑偏。`ChatTemplate.generation_prompt(text)` 已经
> 处理了这一点，`rwkv-easy generate` 也用的是它；自己写推理代码时请调用同一
> 个方法。

## loss mask 是怎么算的

1. 按模板把对话渲染成若干「文本片段」，每段带 `trainable` 标记；
2. 把所有片段拼成**一个完整的字节串**（分词不是逐段可拼接的，必须先拼后分）；
3. 记录每个可训练片段在字节串中的区间；
4. 分词后，只要一个 token 的字节区间与任何可训练区间**有重叠**，就标记为可训练；
5. `labels[t] = ids[t+1] if trainable[t+1] else -100`。

第 4 步用「重叠」而不是「完全包含」，是为了保证 "Assistant: " 和回复正文
被合并成一个 token 时，回复的第一个 token 不会漏掉。

这就是标准的 **completion-only loss**：

- prompt / system / 工具返回只作为上下文，不产生任何 loss；
- `labels[t]` 由「下一个 token 是否可训练」决定，所以回复的第一个 token 由
  prompt 的最后一个 token 预测，同样被监督；
- loss 分母只统计有效 token（`ignore_index=-100`），不会被 prompt 稀释。

相关配置：

| 配置 | 效果 |
|---|---|
| `train_on: assistant`（默认） | 所有 assistant 轮次都计 loss |
| `train_on: last_assistant` | 只训练最后一条回复，之前的回复降级为上下文 |
| `append_eos: end`（默认） | 对话末尾的 EOT 计 loss，模型学会停止 |
| `append_eos: each_assistant` | 每轮回复后都训练一个 EOT |
| `append_eos: never` | 不训练 EOT |

目前没有「把 prompt 也计入 loss」的开关；纯文本数据（`format: text`）是唯一的
例外，它的整段内容都会参与训练。

### 用 `inspect-data` 肉眼检查

```bash
rwkv-easy inspect-data -c configs/lora.yaml -n 3
```

```
--- 窗口 0 ---
User: 请只回答一个词

Assistant: 【苹果】【

】
```

`【】` 里的内容会参与 loss。如果不对：

| 现象 | 调整 |
|---|---|
| 用户问题也被训练了 | `data.train_on` 应为 `assistant` |
| 完全没有可训练内容 | 数据集里 assistant 回复为空，或 `format` 设错 |
| 只训练了最后一条回复 | 你想训练所有回复，就把 `train_on` 改成 `assistant` |
| 模型不会停 | `append_eos: end`（默认） |

## packed vs sample

### `packed`（默认，推荐）

把所有对话的 token 拼成一条长流，按 `ctx_len` 切窗口：

```
[对话A 用户][对话A 助手][EOT][对话B 用户][对话B 助手][EOT] ...
|------------- 窗口 0 -------------|------------- 窗口 1 ------...
```

优点：

- 利用率接近 100%，短对话不会浪费；
- 定长、无 padding → **兼容 infctx**；
- token 流连续，适合长上下文训练。

注意：

- 一条对话可能被窗口切开（正常现象）；
- 文档边界处的 label 会被置为 `-100`，不会让模型学习“上一段结尾 → 下一段开头”。

### `sample`

一条对话一个样本，batch 内动态 padding：

```python
input_ids       # (B, 实际最长)
labels          # padding 位置为 -100
attention_mask  # padding 位置为 0
```

优点：语义干净，适合短对话 SFT。

限制：**不能与 infctx 同时使用**（配置校验会直接报错），因为 infctx 需要
定长、无 padding 的样本才能正确传递循环状态。

超长样本按 `data.truncation` 截断：`head`（默认，保留开头）、`tail`、`random`。

## EOT（结束符）

RWKV 世界词表里 `b"\x00"` 的 id 通常是 **1**（不是 0；词表从 1 开始编号）。
`data.eos_token_id` 留空时自动取词表里的值。

`append_eos` 的三种取值：

| 值 | 行为 |
|---|---|
| `end`（默认） | 对话以 assistant 结尾时，在最后追加 EOT |
| `each_assistant` | 每条 assistant 回复后都追加 EOT |
| `never` | 不追加 |

追加的 EOT 参与 loss，这样模型能学会“什么时候停”。

## 缓存

tokenize 结果会缓存为 `tokens-<hash>.pkl`，放在：

- `data.cache_dir` 指定的目录，或
- 数据文件旁边的 `.rwkv_easy_cache/`

缓存 key 包含：数据文件列表与 mtime、模板、`train_on`、`append_eos`、EOT id、
词表 mtime、`max_samples`、`val_ratio`、随机种子。改任何一项都会重新 tokenize。

首次 tokenize 之后，再次启动会打印「命中 tokenize 缓存」，几秒即可开始训练。

## 常见坑

**1. 数据里没有 assistant 回复**
`load_records` 会跳过没有 assistant 内容的记录，并打印跳过数量。

**2. 训练集太小**
packed 模式下窗口数 = 总 token 数 / ctx_len。如果不足一个窗口，会补一条并提示。
建议至少几十 MB 文本，或者用 `sample` 模式 + 小 `ctx_len`。

**3. 词表不匹配**
本项目内置的是 RWKV 世界词表（65529 个 token）。如果你用的是其它词表的模型
（例如 20B tokenizer），需要自己提供 `data.vocab_path`，并保证 token 文本
与模型训练时一致。

**4. 多进程 tokenize 在 Windows 上失败**
会自动回退到单进程并打印警告；也可以直接设 `data.num_proc: 0`。
