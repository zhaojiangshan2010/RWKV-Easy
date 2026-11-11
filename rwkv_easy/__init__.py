"""RWKV-EASY：简单、可靠、对新手友好的 RWKV-7 训练器。

只做训练，聚焦两件事：

* **LoRA** 微调
* **State Tuning** 微调

两者都可以叠加 RWKV 的 **infinite context (infctx)** 分块训练。

典型用法::

    python quickstart.py                 # 安装 + 交互式向导
    rwkv-easy wizard                     # 已安装后直接进向导
    rwkv-easy train -c configs/my_lora.yaml

参见 ``README.md`` 与 ``docs/``。
"""

from __future__ import annotations

__version__ = "0.3.1"

__all__ = ["__version__"]
