"""衰减公式自动校准。

不同版本的 RWKV-7 实现使用不同的 decay 参数化（见 ``TimeMix.log_decay``）。
用错公式不会报错，但模型输出会明显变差。这里用一段内置的中英校准文本，分别
在三种公式下计算语言建模 loss，取最低者。

校准只在启动时跑一次（3 次 256 token 的前向），结果会缓存到
``<cache_dir>/decay_calibration.json``，按 checkpoint 签名 + 模型维度索引。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..data.tokenizer import RWKVTokenizer

__all__ = ["CALIB_TEXT_PATH", "DecayCalibration", "calibrate_decay"]

CALIB_TEXT_PATH = Path(__file__).resolve().parent.parent / "assets" / "calib_text.txt"
_MODES = ("clampw", "softplus", "exp")


@dataclass
class DecayCalibration:
    mode: str
    scores: dict[str, float]
    source: str  # "cache" | "calibrated" | "default"

    def describe(self) -> str:
        parts = ", ".join(f"{k}={v:.4f}" for k, v in self.scores.items())
        return f"decay={self.mode}（{self.source}；{parts}）"


def _read_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover
        return {}


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:  # pragma: no cover
        pass


@torch.no_grad()
def calibrate_decay(
    model: torch.nn.Module,
    tokenizer: RWKVTokenizer,
    device: torch.device,
    cache_key: str,
    cache_dir: str | Path | None = None,
    max_tokens: int = 256,
    log: Any = None,
) -> DecayCalibration:
    """返回校准后的衰减模式。"""
    cache_path = Path(cache_dir) / "decay_calibration.json" if cache_dir else None
    if cache_path is not None:
        cached = _read_cache(cache_path).get(cache_key)
        if isinstance(cached, dict) and cached.get("mode") in _MODES:
            return DecayCalibration(
                mode=cached["mode"],
                scores={k: float(v) for k, v in cached.get("scores", {}).items()},
                source="cache",
            )

    text = CALIB_TEXT_PATH.read_text(encoding="utf-8")
    ids = tokenizer.encode(text)[:max_tokens]
    if len(ids) < 16:
        if log is not None:
            log.warning("校准文本过短，跳过 decay 校准，使用 clampw。")
        return DecayCalibration("clampw", {}, "default")

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    was_training = model.training
    original_mode = getattr(model, "decay_mode", "clampw")
    model.eval()

    scores: dict[str, float] = {}
    try:
        for mode in _MODES:
            model.set_decay_mode(mode)  # type: ignore[attr-defined]
            logits, _ = model(input_ids, need_state=False)  # type: ignore[operator]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                input_ids[:, 1:].reshape(-1),
            )
            scores[mode] = float(loss.item())
    finally:
        model.set_decay_mode(original_mode)  # type: ignore[attr-defined]
        model.train(was_training)

    best = min(scores, key=scores.get)
    # 差距太小说明 checkpoint 可能没训练好（或随机初始化），保留原默认值更安全。
    ordered = sorted(scores.values())
    if len(ordered) >= 2 and (ordered[1] - ordered[0]) < 0.02:
        if log is not None:
            log.warning(
                "decay 校准结果区分度很低（%s），回退到 clampw。",
                ", ".join(f"{k}={v:.3f}" for k, v in scores.items()),
            )
        best = "clampw"

    if log is not None:
        log.info(
            "decay 校准：%s -> 选择 %s",
            ", ".join(f"{k}={v:.4f}" for k, v in scores.items()),
            best,
        )

    if cache_path is not None:
        payload = _read_cache(cache_path)
        payload[cache_key] = {"mode": best, "scores": scores}
        _write_cache(cache_path, payload)

    return DecayCalibration(best, scores, "calibrated")
