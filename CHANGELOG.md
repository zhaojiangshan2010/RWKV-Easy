# 变更记录

## 0.6.0 — 2026-11-13

quickstart 复用主环境已装的库，只补缺失依赖。

### 新增

- **venv 复用主环境**：新建 `.venv` 默认带 `system-site-packages`，直接看见
  主环境的 torch / PyYAML / rwkv-fla / DeepSpeed，不再从零安装。
- **已有半成品 .venv 兼容**：如果 `.venv` 是自己的隔离环境、里面没有 torch，
  会写一个 `_rwkv_easy_system_site.pth` 指向主环境 site-packages；不复制库、
  不删除 venv、不重装 torch。Python 主版本不同则跳过并提示。
- **依赖探测表**：启动时逐个报告 PyTorch / PyYAML / RWKV-EASY / rwkv-fla /
  DeepSpeed / safetensors / Triton 是否已安装，以及 torch 的 CUDA 状态。
- **只装缺失项**：`plan_required_installs()` 根据探测结果决定是否安装
  torch / pyyaml / 项目；已装的直接跳过。项目安装改为
  `pip install -e . --no-deps`，rwkv-fla 改为 `--no-deps`，避免 pip 把
  torch 当缺失依赖重装。
- `--isolated-venv`：需要完全隔离时，创建不带 system-site-packages 的 venv。
- GPU 探测优先使用子进程 probe 的 torch 信息，减少一次 torch 导入。

### 修复

- `venv_uses_system_site_packages()` 现在接受 venv 根目录，正确读取
  `pyvenv.cfg`（之前误拼成 `.venv/pyvenv.cfg`）。
- `--skip-install` 会先尝试复用主环境库，再做 torch 探测；半成品 `.venv`
  仍会自动回退到当前 Python。

### 测试

- **143 个测试**通过（CPU 138 passed + 5 fla skipped）。
- 新增 `tests/test_quickstart.py`：依赖探测、跳过/安装计划、GPU 与 --cpu
  分支、项目路径判断、system-site-packages venv、`.pth` 复用。
- 本地临时项目副本端到端验证两种场景：新建 venv 继承主环境；已有隔离 venv
  自动写 `.pth` 后复用。两次都输出“必需依赖已齐全，跳过安装”，没有触发 pip。
- 远程 GPU 实例已关机，本轮未做远程复测。

## 0.5.0 — 2026-11-12

多卡 DDP 修复 + infctx 逐块反传。

### 修复

- **DDP 梯度不同步（严重）**：Trainer 之前直接调用原始模型的
  `forward_hidden`，绕过了 `DDP.forward`，autograd hook 从未准备 reducer，
  backward 时梯度不会 all-reduce。现在 DDP / DeepSpeed 都通过
  `_HiddenForwardAdapter` 走包装层的 forward；两进程 gloo 实测普通 DDP
  梯度从各卡 `[1,0]/[0,1]` 修正为两边 `[0.5,0.5]`。
- **DeepSpeed 前向**：改为 `engine.forward`，ZeRO-2 单卡非 infctx / infctx
  均复测通过；ZeRO-3 未验证。
- **DDP 延迟 infctx 路径**：不再依赖“多次 forward 后一次 backward”的 DDP
  reducer；改为原始模型逐块前向、一次 backward，再在梯度累积边界手动
  all-reduce。fp16 GradScaler 下直接报错，避免各 rank 缩放因子不同导致错误。
- `torch.compile` 在 DDP/DeepSpeed 下改为编译适配层内部的模型前向，
  不编译通信包装。

### 新增

- **`train.infctx_parallel_backward`**（默认 `true`）：DDP + LoRA + infctx
  时每个 chunk 立即 backward，除最后一块外都包在 `no_sync()` 里；最后一块
  触发一次梯度 all-reduce。显存 O(chunk_ctx)，不再随 `ctx_len` 增长。
  设 `false` 可回退到“所有块最后统一反传”。
- 配置校验：逐块反传要求 `infctx_detach: true`；DeepSpeed + 该字段为 true
  直接报错；State Tuning + DDP 给出警告并自动回退。
- 向导：多卡 + infctx + LoRA 时首页出现「多卡逐块反传」开关；高级设置新增
  对应字段与完整说明；摘要显示“逐块（最后 all-reduce）/ 最后统一”；
  `长上下文 LoRA` 模板默认开启，`State` 模板默认关闭。
- `lora_sharegpt_infctx.yaml` / `state_sharegpt_infctx.yaml` /
  `template_config()` 补上该字段。
- 新增两进程梯度一致性测试：普通 DDP、infctx 逐块、梯度累积 × 逐块嵌套；
  新增 DDP infctx 两种路径的端到端测试与配置校验测试。

### 测试

- **139 个测试**通过（CPU 134 passed + 5 fla skipped），本地与远程 GPU 均通过。
- 远程另跑了 DeepSpeed 单卡非 infctx（2 步，1.0 GiB）与 infctx 最后统一反传
  （2 步，0.8 GiB）冒烟；**NCCL 多卡与 DeepSpeed 多卡仍未实测**。

## 0.4.0 — 2026-11-12

一键向导 TUI 重做：配置模板、参数注释与新手引导。

### 新增

- **欢迎页**：启动后先说明“3 步开始”，可直接使用推荐配置、按场景选模板、
  查看新手引导或退出。
- **7 个配置模板**：推荐 LoRA、长上下文 LoRA、State、长上下文 State、
  省显存、先跑 20 步、自定义。套用模板只改训练相关配置，保留模型/数据/输出目录。
- **参数注释**：首页每一项右侧都有一句话注释；高级设置的全部 81 个字段都有
  行内注释 + 选中后的详细说明，并给常用枚举值加了中文标签。
- **新手引导**：4 页（工具能力 / 训练前准备 / 模板怎么选 / 常见坑），
  可从欢迎页或首页按 `g` 打开。
- **按键帮助**：首页按 `?` 查看完整快捷键；`p` 随时打开模板菜单；
  高级表单按 `/` 可按名称/字段/说明搜索并跳到对应参数；欢迎页按 `g` 直接看引导。
- **摘要页重做**：配置摘要与下一步选项同屏显示，不再被菜单覆盖；小终端自动
  收起次要提示。
- **路径选择器**：列表显示文件名+大小，选中项显示完整路径；没有候选时直接
  提示该把文件放到哪个目录。
- **中文 TUI 细节**：按终端宽度自适应，中文按 2 列对齐，英文单词不被从中间
  折断，说明框/边框在 80×24 到 100×32 都能闭合对齐；Windows 非 UTF-8
  控制台自动退回 ASCII 边框。
- **quickstart**：增加 3 步说明与 `[1/3]` 进度；没有 NVIDIA GPU 时先确认是否
  继续安装 CPU 版，避免误装后才发现不能正式训练。
- `rwkv-easy wizard --no-welcome`：跳过欢迎页，直接进入配置。

### 测试

- 133 个测试通过（CPU 上 128 passed + 5 fla skipped）；新增模板、字段注释、
  中文宽度/折行、首页说明覆盖测试。

### 文档

- 重写 **`docs/WIZARD.md`**：欢迎页、首页、模板表、参数注释、新手引导、
  路径选择、非交互模式与常见问题。
- README 精简为「性能展示 + 上手入口 + 文档索引」，不再承载完整教程。
- 新增 **`docs/GETTING_STARTED.md`**：安装、模型/数据准备、配置生成、数据检查、
  训练与输出、续训、内部流程、四种组合、CLI 速查、权重验收、CPU 自检、
  自我认知 demo、显存处理顺序。
- 新增 **`docs/DEVELOPMENT.md`**：项目结构、测试分组、配置字段改动清单、
  调试开关、版本与提交约定。

## 0.3.1 — 2026-11-12

文档与可复现性收尾。

### 新增

- **`docs/TRAINING_PARAMETERS.md`**：覆盖 `Config` 全部字段（run / model /
  train / lora / state / data / dist / log），包含类型、默认值、真实语义、
  交互关系、校验规则、常用配方与代码位置。
- `tests/test_docs.py`：自动检查参数手册是否覆盖所有配置字段，新增字段漏文档
  会直接测试失败。
- `tests/test_runtime.py`：验证 seed 导出与 deterministic 开关。

### 修复

- `run.seed` 现在写入 `RWKV_EASY_SEED`，DataLoader worker 的 NumPy/Python
  随机种子会跟随配置，而不是固定 42。

### 测试

- 126 个测试通过（CPU 上 121 passed + 5 fla skipped）。

## 0.3.0 — 2026-11-12

一键安装 + 交互式配置向导，并支持按 block 选择 LoRA 层。

### 新增

- **`quickstart.py` / `quickstart.sh` / `quickstart.bat`**：安装与训练合一。
  检测 NVIDIA 驱动 / CUDA 版本，自动创建 `.venv`、安装对应 PyTorch、
  安装项目与可选 rwkv-fla，然后进入向导并直接训练。
- **`rwkv-easy wizard`**：跨平台方向键 TUI（Windows `msvcrt`、Unix `termios`，
  无第三方依赖）。支持：
  - `←/→` 切换 LoRA / State、infctx、后端、精度；
  - 自动扫描并校验模型 / 数据集路径，也可手动输入；
  - `空格` 多选 LoRA 模块与 block 层；
  - 高级设置覆盖全部 Config 字段；
  - 摘要检查、保存 YAML、直接启动训练；
  - 无 TTY 时自动退化为分行问答（`--non-interactive`）。
- **LoRA `target_layers`**：只注入指定 block（`blocks.<i>.`），向导可按空格
  勾选；`null` 表示全部层。
- 新增 `tests/test_wizard.py`，测试总数 **121**。

### 文档

- 新增 [`docs/WIZARD.md`](docs/WIZARD.md)：按键、流程、默认值、非交互用法与排错。
- README 增加「一键向导」入口，配置/结构/命令表同步；
- `CONFIG.md` 增加 `lora.target_layers`。

## 0.2.0 — 2026-11-12

性能与 infctx 架构升级：0.1B 在 24GB 卡上的整段 LoRA 从 13.7–16.1K 提升到
**48.5K tok/s**，infctx 真正做成 `O(chunk_ctx)` 显存。

### 新增

- **`train.infctx_backward`（默认 true）**：每个 chunk 立即 backward，
  只保留 detach 后的循环状态；8K ctx 显存 12.4 → 3.4 GiB，0.1B 可训
  32K 上下文。DDP/DeepSpeed/`infctx_detach=false` 自动回退旧路径。
- **`precision: bf16-full` 成为 GPU 模板默认**：基座 bf16、LoRA/time_state
  仍为 fp32；0.1B ctx512/bsz16 吞吐 +46%，显存 14.6 → 8.9 GiB。
- **只对可训练 token 过 lm_head**：SFT 约一半 prompt token 不再产生
  65536 维 logits，+9% 并显著省显存。
- **修正 `train.compile`**：之前编译整个 model 但前向没走编译路径；
  现在 fla 编译 `forward_hidden`（+20%），native 只编译 WKV `_chunk`
  （+18%，首次约 25s）。新增 `train.compile_mode`
  （default / reduce-overhead / max-autotune）。
- 新增 `tests/test_infctx_backward.py`、`tests/test_l2wrap.py`、
  `tests/test_head_selection.py`；测试总数 110。

### 性能

- fla LoRA：48.5K tok/s（ctx512/bsz32，compile，11.9 GiB）；
  State：56.3K（reduce-overhead，9.2 GiB）。
- native LoRA：24.3K tok/s（bsz24，wkv_chunk 64，compile 单块），
  比最初 3.9K 提升 6 倍；native infctx 2.3K → 5.1K。
- infctx：ctx8192/chunk2048/bsz2 37.0K（3.4 GiB）；
  bsz10 41.7K（15.2 GiB）；ctx32768/chunk2048/bsz2 33.6K（3.4 GiB）。
- State + infctx：72.6K tok/s（2.7 GiB，只有第一块需要反传）。
- L2Wrap 改为在 CE 梯度上原地 `scatter_add`，省掉每步 2 GiB 全量梯度；
  闭包只捕获 max_value/indices，修复引用环泄漏。
- native 缓存下三角掩码，`solve_triangular` 用 `unitriangular=True`。

### 文档

- 新增 [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)：按收益排序的调优旋钮、
  显存模型、按模型规模的建议、benchmark 方法与常见误判。
- README 重写「显存与参数建议 / 性能调优」章节，统一为 0.2.0 的实测数字；
  新增与 RWKV-PEFT 的同配置 LoRA 对比（表格 + `docs/assets/peft_vs_easy.svg`）。
- `CONFIG.md` 增加 `compile_mode` / `infctx_backward` / bf16-full 说明；
  `ARCHITECTURE.md` 增加「训练 loss 与前向编译」；`TROUBLESHOOTING.md`
  更新 compile / infctx / GPU 慢的排查。

### 试过但没有保留

- bf16 交叉熵 / bf16 GroupNorm：没有收益。
- native 合并 4 个块内 matmul + 4D 布局：算子级略快，但显存更高、大 batch
  下 OOM，已回退。
- TF32、native chunk 128：没有收益或直接 OOM。
- native 整模型 `torch.compile`：12 层 × N 块全部展开，首次编译 >9 分钟；
  改为只编译 WKV `_chunk` 后 +18%。
- native bf16 快速路径（块内 matmul 用 bf16、solve/状态仍 fp32）：
  相对 fp64 误差 4e-3（可接受），但速度 20.17K vs fp32 20.76K，没有收益，已回退。
- LoRA forward 合并 B@A 进权重：50.44K vs 50.34K，在噪声内，未采用。
- head+CE 单独编译：34.2 → 26.6 ms、显存 3.1 → 1.1 GiB，但只占整步约 2%，
  集成 L2Wrap 的复杂度不值。

## 0.1.1 — 2026-11-11

第二次 GPU 验证（torch 2.14.1 + CUDA 13.0 + triton 3.8.0）。这一版的主题是
**rwkv-fla 后端正式可用**。

### 新增

- `model.backend: fla` 通过验证：支持 clampw / softplus / exp 三种 decay，
  不再强制 softplus；官方权重可以直接用 `clampw`。
- 新增 `configs/lora_sharegpt_fla.yaml`（CUDA + Triton 加速模板）。
- 新增 `tests/test_fla_backend.py`：与 fp64 参考对照、补齐路径、分块状态、
  clampw、bf16 反传；没有 CUDA/rwkv-fla 时自动跳过。
- launcher：请求进程数超过可见 GPU 数时提前报错；任一 rank 早退时立即
  终止其它进程，不再空等 10 分钟。

### 修复

- **fla 参数语义**：`chunk_rwkv7` 的 `w` 是原始 decay 参数（内部再算
  `-exp(w)`），必须把 log-decay 传给 `log_w`。此前误传 `w` 时 loss 仍会下降，
  但数值与参考不一致。
- **fla 补齐位**：长度不是 64 的倍数时用 `log_decay = 0`（不是 -60）+
  全零 k/v/a/b，否则携带状态会被衰减成 0。
- **fla dtype**：统一混合精度下各输入的 dtype，修复
  `Both operands must be same dtype. Got bf16 and fp32`。
- **native 动态分块**：softplus/exp 从递归二分改为一次 reduce 估算安全块长，
  去掉每层 `.item()` 的 GPU 同步；真实权重下仍受衰减限制，但不再指数级变慢。
- **DDP 测试**：显式 `--device cpu`，修复单卡 GPU 机器上 `num_gpus=2`
  导致 rank 1 在 `set_device(1)` 崩溃、rank 0 空等的问题。
- `init_process_group` 加 300 秒超时；`LOCAL_RANK` 超出可见 GPU 数时给出
  明确错误。

### 验证

- 103 个测试通过（GPU 上含 5 个 fla 对照；CPU 上跳过）。
- 0.1B / ctx512 / bsz4 / bf16：native clampw 约 3.9K tok/s（5.8 GiB），
  fla 约 13.7–16.1K tok/s（4.4 GiB），loss 曲线基本重合。
- 同配置下 fla 全面更快：State ctx512/bsz4 4.35K → 22.95K；
  infctx ctx1024/chunk256 470 → 1.37K。不同 `ctx_len`/`bsz` 的绝对 tok/s
  不可直接比较（每步 token 越少，固定开销占比越高）。
- fla + infctx、fla + State、fla + DeepSpeed ZeRO-2 均跑通。
- fla 训练出的 State 适配器合并后，`generate` 正确回答 `RWKV 0.1B 大模型`。

## 0.1.0 — 2026-11-11

首个可用版本。

### 功能

- RWKV-7（x070）训练：**LoRA** / **State Tuning**
- 两种方法都可以叠加 **infctx** 分块长上下文
- 数据：ShareGPT / OpenAI messages / 两列指令 / 纯文本，**completion-only loss**
- 样本组织：`packed`（定长流，兼容 infctx）/ `sample`（动态 padding），tokenize 缓存
- 精度：FP32 / FP16 / BF16 / BF16-full
- 并行：单卡 / PyTorch DDP / DeepSpeed（可选，不装也能多卡）
- WKV 后端：纯 PyTorch 分块实现（CPU/CUDA/ROCm 通用）+ fp64 reference 对照 +
  可选 `rwkv-fla`（Triton）
- **decay 公式自动校准**（`clampw` / `softplus` / `exp`）
- CLI：`init` / `train` / `merge` / `inspect-data` / `check-model` / `doctor` /
  `selftest` / `generate`
- 一键安装与训练脚本（Linux / Windows），中文文档与配置注释

### 修复（真实权重验收中发现）

- 输出门控低秩维度 `g1/g2` 的官方公式是 `0.6 * C^0.8`（此前误用 `0.6 * C^0.4`，
  会导致真实 checkpoint 形状不匹配）
- `precision: fp32` 未把 bf16 checkpoint 显式上转，CPU 反向传播慢约 75 倍
  （实测 42s vs 0.57s / step）
- `max_steps` 被 `epochs` 截断（日志写 20 步、实际 9 步）
- 生成 prompt 与训练文本的 token 边界不一致：`"Assistant: "` 的尾空格会被分词
  并入回答首 token，推理时必须用 `Assistant:` 结尾（`ChatTemplate.generation_prompt`）

### 验证

- 98 个纯 CPU 测试（含 fp64 数值对照、infctx 等价性、2 进程 DDP）
- `selftest --full`：lora/state × infctx on/off × fp32/bf16 共 8 组
- 真实权重 `rwkv7-g1d-0.1b`：decay=clampw（NLL 2.54），续写连贯；
  LoRA 20 步、infctx 8 步、合并等价性（logits 误差 7.8e-5）
- 自我认知过拟合 demo：400 步 EMA 3.75→0.087，10/10 命中（含未见问法）

### GPU 验证（2026-11-11，24GB Ampere 级）

- 真实 0.1B：`check-model`、LoRA、State、infctx、合并、fp16、DeepSpeed ZeRO-2 全部跑通
- LoRA ctx512/bsz4 ≈4.8K tok/s（5.8GiB）；infctx ctx4096/chunk512 ≈700 tok/s（3.7GiB）
- 0.1B 是启动开销受限：bsz 1→8 单步 365→378ms，吞吐 1.4K→10.8K tok/s
- 自我认知 demo 在 GPU 上 400 步 60 秒，8/8 命中

### 修复（GPU 验证中发现）

- `torch < 2.4` 的 `torch.load(mmap=True)` 只接受 `str`，传 `Path` 报错
- `torch 2.2` 没有 `torch.amp.GradScaler`，fp16 路径加兼容层
- 在 `model.to(device)` 之后注入 LoRA 时 A/B 建在 CPU，设备不匹配
- `--set train.lr=2e-4` 被 YAML 1.1 解析成字符串
- `--set dist.deepspeed.zero_stage` 在 YAML 缺该段时直接报错
- DeepSpeed 单进程缺进程组（去探测 MPI）、缺 `LOCAL_RANK`
- DeepSpeed `train_batch_size` 与手动梯度累积冲突
- clampw 模式下 WKV 逐块 `.item()` 同步，改为静态分块

### 尚未验证

- rwkv-fla / Triton 后端（测试环境 torch 2.2 + triton 2.2 与 rwkv-fla 0.7 不兼容）
- NCCL 多卡（当前只有单卡）
