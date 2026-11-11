# 扩展指南

三个最常被问到的扩展方向：**换 WKV 内核**、**加模型版本**、**加数据格式**。

---

## 1. 接入自定义 WKV 内核（CUDA / Triton / ROCm）

### 1.1 接口契约

所有后端实现同一个方法：

```python
def wkv7(
    r, k, v, log_decay, a, b, state=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    r, k, v, a, b : (B, T, H, D)          输入 dtype 可能是 fp32 / bf16 / fp16
    log_decay     : (B, T, H, D)          已经算好的 log 衰减，<= 0
    state         : (B, H, D, D) 或 None  初始循环状态（None 表示全零）

    return:
      out         : (B, T, H, D)          与输入同 dtype
      new_state   : (B, H, D, D)          块结束时的状态（即使调用方暂时不用也要返回）
    """
```

语义（逐 token）：

```
S_t = diag(exp(log_decay_t)) S_{t-1} + b_t (a_t · S_{t-1}) + k_t v_t^T
o_t = r_t · S_t
```

注意 `a` 传进来时已经是 `-kk`，`b` 已经是 `kk * alpha`，不要重复处理。

### 1.2 最小示例

```python
# my_backend.py
from __future__ import annotations

import torch

from rwkv_easy.backends import register_backend


class MyKernelBackend:
    name = "mykernel"

    def __init__(self, chunk: int = 64) -> None:
        self.chunk = chunk

    def wkv7(self, r, k, v, log_decay, a, b, state=None):
        # 这里用一个“逐 token 循环”占位；真实实现应换成你的 kernel。
        # 关键点：计算必须可微，且 state 要按 (B,H,D,D) 返回。
        B, T, H, D = r.shape
        S = state if state is not None else r.new_zeros(B, H, D, D)
        outs = []
        for t in range(T):
            d = log_decay[:, t].exp()
            sa = (a[:, t].unsqueeze(-1) * S).sum(dim=2)          # (B,H,V)
            S = (S * d.unsqueeze(-1)
                 + b[:, t].unsqueeze(-1) * sa.unsqueeze(-2)
                 + k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2))
            outs.append((r[:, t].unsqueeze(-1) * S).sum(dim=2))
        return torch.stack(outs, dim=1), S


register_backend("mykernel", MyKernelBackend)
```

用一个小脚本先注册再走 CLI：

```python
# train_my.py
import my_backend                          # noqa: F401  注册副作用
from rwkv_easy.cli import main

raise SystemExit(main([
    "train", "-c", "configs/lora_sharegpt.yaml",
    "--set", "model.backend=mykernel",
]))
```

```bash
python train_my.py
```

### 1.3 用 torch.utils.cpp_extension 接入 CUDA 源码

```python
from pathlib import Path
from torch.utils.cpp_extension import load

_SRC = Path(__file__).resolve().parent / "cuda"      # 用绝对路径！
_ext = load(
    name="my_wkv7",
    sources=[str(_SRC / "wkv7.cu"), str(_SRC / "wkv7.cpp")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    is_python_module=False,
    verbose=False,
)
```

> 官方 RWKV-LM / RWKV-PEFT 用的是**相对路径**（`sources=['cuda/...']`），
> 因此必须从项目根目录启动。本项目建议一律用绝对路径，避免这个隐式依赖。

然后用 `torch.autograd.Function` 包一层：

```python
class WKV7Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, k, v, w, a, b, s0):
        ...
        return y

    @staticmethod
    def backward(ctx, dy):
        ...
        return dr, dk, dv, dw, da, db, ds0
```

### 1.4 ROCm 与 Triton

- **ROCm**：PyTorch 在 ROCm 上把设备暴露成 `cuda`，`native` 后端**直接可用**，
  不需要任何额外代码。
- **Triton**：`rwkv-fla` 同时支持 CUDA/ROCm。装好之后 `model.backend: fla`
  即可；它要求 `model.decay: softplus`。
- 想自己写 Triton：按上面的契约实现 `wkv7`，注意 Triton 内核通常要求
  `T` 是块长的倍数，可以在你的后端里自行 pad（参考 `backends/fla.py`）。

### 1.5 必须通过的测试

新增后端请至少通过：

```python
from rwkv_easy.backends.native import NativeBackend
from rwkv_easy.backends.reference import ReferenceBackend

ref = ReferenceBackend(double=True)   # float64 黄金参考
mine = MyKernelBackend()
# 前向、初始状态、以及全部输入的梯度都应在 1e-5 相对误差内
```

可以直接把 `tests/test_wkv.py` 里的用例参数化到你的后端。

---

## 2. 增加模型版本（例如 RWKV-6 / x060）

当前只实现了 RWKV-7。要加 v6，需要提供同样的模型接口。

### 2.1 必须实现的成员

```python
class RWKV6(nn.Module):
    def __init__(self, cfg, backend, decay_mode="clampw", grad_checkpoint=False): ...

    def forward(
        self,
        input_ids: Tensor,                 # (B,T)
        attention_mask: Tensor | None,     # (B,T)，0/1
        state: ModelState | None,          # 三个稠密张量
        need_state: bool,
    ) -> tuple[Tensor, ModelState | None]:  # logits, new_state

    def forward_chunk(
        self, input_ids,
        shift_att, shift_ffn, wkv,         # 可为 None
        attention_mask=None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """infctx 单块前向，返回 (logits, 新 shift_att, 新 shift_ffn, 新 wkv)"""

    def attach_state_tuning(self, init_std: float = 0.0) -> list[nn.Parameter]: ...
    def state_parameters(self) -> list[nn.Parameter]: ...
    def initial_state(self, batch_size, device=None, dtype=torch.float32) -> ModelState: ...
    def set_decay_mode(self, mode: str) -> None: ...
    @property
    def decay_mode(self) -> str: ...
```

### 2.2 需要改的地方

| 文件 | 改什么 |
|---|---|
| `models/rwkv6.py` | 新增模型实现（可以复制 `rwkv7.py` 再改层结构） |
| `models/checkpoint.py` | `build_model()` 里按 `cfg.version` 分支 |
| `models/rwkv7.py::RWKV7Config` | 如需不同维度关系，新增一个 config 类 |
| `data/` | 不需要改（数据层与模型无关） |

建议在 `ModelConfig` 里加一个 `version: str = "rwkv7"` 字段，然后：

```python
def build_model(cfg, *, version="rwkv7", ...):
    if version == "rwkv7":
        return RWKV7(...)
    if version == "rwkv6":
        from .rwkv6 import RWKV6
        return RWKV6(...)
    raise ValueError(version)
```

### 2.3 不要做的事

- 不要用环境变量在 import 时决定训练模式（参考实现这么干，导致测试和组合
  非常难写）；
- 不要把循环状态塞进 `**kwargs` 或位置参数的“空位”；
- 不要让 LoRA 改变 forward 的签名。

---

## 3. 增加数据格式

在 `data/sharegpt.py::parse_record` 里加一个分支，返回
`Conversation`（有角色的多轮）或 `TextDocument`（整段可训练文本）：

```python
if "my_format" in obj:
    messages = [
        Message(normalize_role(item["speaker"]), content_to_text(item["utterance"]))
        for item in obj["my_format"]
    ]
    messages = _merge_messages(messages)
    return Conversation(messages, source=source) if messages else None
```

其余部分（模板、mask、packed/sample、缓存）都会自动工作。

如果要支持新的文件后缀，在 `load_records_from_file()` 里加读取分支即可。

---

## 4. 调试技巧

**只跑一个 batch**

```yaml
train:
  max_steps: 1
  micro_batch_size: 1
  ctx_len: 128
data:
  max_samples: 32
  num_workers: 0
```

**看数据长什么样**

```bash
rwkv-easy inspect-data -c cfg.yaml -n 5 --window 3
```

**对照参考实现验证 WKV**

```python
python -m pytest tests/test_wkv.py -q
```

**多卡问题先退回单卡**

```bash
rwkv-easy train -c cfg.yaml --single
```

**完整堆栈**

```bash
RWKV_EASY_DEBUG=1 rwkv-easy train -c cfg.yaml
```
