# 开发与测试

面向想改代码、跑测试或发版的人。只想训练请看
[`GETTING_STARTED.md`](GETTING_STARTED.md)；实现原理见
[`ARCHITECTURE.md`](ARCHITECTURE.md)。

---

## 1. 本地开发环境

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

`dev` 额外安装 `pytest` 与 `ruff`。GPU 上再加 `pip install rwkv-fla`（可选）。

---

## 2. 项目结构

```
RWKV-EASY/
├── quickstart.py / .sh / .bat   # 安装 + 交互式向导（新手入口）
├── rwkv_easy/
│   ├── cli.py              # 所有命令
│   ├── wizard.py           # 跨平台交互式配置向导
│   ├── config.py           # YAML 配置 + 严格校验
│   ├── launcher.py         # 多进程启动
│   ├── lora.py             # 自研 LoRA（支持按模块/按层）
│   ├── state_tuning.py     # State Tuning
│   ├── backends/           # WKV 算子（native / reference / fla）
│   ├── models/             # RWKV-7 模型与 checkpoint 加载
│   ├── data/               # 分词器 / ShareGPT / 数据集
│   ├── train/              # 训练循环 / DDP / DeepSpeed / 保存
│   └── assets/             # 内置词表、校准文本、示例数据
├── configs/                # 配置模板（含自我认知过拟合 demo）
├── docs/                   # 详细文档
├── tests/                  # 143 个测试（CPU 上跳过 5 个 fla）
├── examples/               # 示例数据（ShareGPT、自我认知实验数据）
├── CHANGELOG.md
├── install.sh / .bat / .ps1
└── train.sh / .bat
```

---

## 3. 测试

```bash
python -m pytest tests -q            # 全量：CPU 上 143 个，138 passed + 5 fla skipped
python -m ruff check rwkv_easy tests
```

测试分组与用途：

| 文件 | 覆盖 |
|---|---|
| `test_wkv.py` / `test_model.py` | WKV 前向/状态/梯度 vs float64 参考、分块等价、模型结构 |
| `test_data.py` / `test_dataset.py` | ShareGPT 解析、模板、completion-only mask、packed/sample |
| `test_train_smoke.py` | 训练 → 保存 → 合并 → 重载全链路 |
| `test_infctx_backward.py` / `test_head_selection.py` / `test_l2wrap.py` | infctx 每块反传、可训练 token 过 head、L2Wrap |
| `test_ddp.py` | 2 进程 gloo DDP：梯度 all-reduce 一致性、infctx 逐块反传、梯度累积 × 逐块嵌套、两种 DDP infctx 端到端（`RWKV_EASY_SKIP_DDP=1` 可跳过） |
| `test_fla_backend.py` | fla 数值对照（无 GPU / 无 fla 时自动跳过；`RWKV_EASY_SKIP_FLA=1` 强制跳过） |
| `test_config.py` / `test_cli.py` / `test_runtime.py` | 配置校验、CLI、种子与运行时 |
| `test_wizard.py` / `test_docs.py` | 向导逻辑、文档字段覆盖 |
| `test_quickstart.py` | quickstart 依赖探测、跳过/安装计划、system-site venv 与 `.pth` 复用 |

CPU 上跑完整自检：

```bash
python -m rwkv_easy selftest --full
```

---

## 4. 改配置字段的检查清单

新增或重命名 `Config` 字段时，至少同步这几处，否则用户文档会失真：

1. `rwkv_easy/config.py`：字段、默认值、校验规则；
2. `docs/TRAINING_PARAMETERS.md`：必须出现 `` `字段路径` ``，否则
   `tests/test_docs.py::test_parameter_manual_covers_every_config_field` 会失败；
3. `docs/CONFIG.md`：常用字段与配方；
4. `rwkv_easy/cli.py` 的 `init` / `wizard` 默认值与帮助文本；
5. `configs/*.yaml` 模板与 `CHANGELOG.md`。

```bash
python -m pytest tests/test_docs.py -q
```

---

## 5. 调试

- `RWKV_EASY_DEBUG=1 rwkv-easy train ...`：打印完整异常堆栈（默认只给一行）；
- `rwkv-easy doctor --json`：环境快照，适合贴进 issue；
- `rwkv-easy train ... --log-level debug`：更详细的训练日志；
- `rwkv-easy inspect-data`：确认 loss mask；
- `RWKV_EASY_SEED`：DataLoader worker 的种子来源，默认由 `run.seed` 设置。

GPU 数值问题先固定变量：同 `ctx_len` / `chunk_ctx` / `micro_batch_size` 再比较；
不要用 step 1 的 tok/s（包含 compile / Triton 首次编译）。

---

## 6. 版本与提交

- 版本号在 `rwkv_easy/__init__.py` 的 `__version__` 与 `pyproject.toml`；
- 行为变化、修复、性能数字同步写入 [`CHANGELOG.md`](../CHANGELOG.md)；
- 文档改动也建议跑一遍 `pytest tests/test_docs.py` 与链接检查；
- 提交信息用 `feat:` / `fix:` / `docs:` / `perf:` / `test:` 前缀，一次提交只做一件事。

---

## 7. 相关文档

| 主题 | 文档 |
|---|---|
| 实现、不变量、数据流 | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| 加 WKV 内核 / 模型版本 / 数据格式 | [`EXTENDING.md`](EXTENDING.md) |
| 性能画像与调优 | [`PERFORMANCE.md`](PERFORMANCE.md) |
| CPU / GPU 验证记录 | [`CPU_VERIFICATION.md`](CPU_VERIFICATION.md)、[`GPU_VERIFICATION.md`](GPU_VERIFICATION.md) |
