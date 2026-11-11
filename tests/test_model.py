"""RWKV-7 模型、LoRA、State Tuning、checkpoint 加载测试。"""

from __future__ import annotations

import copy

import pytest
import torch

from rwkv_easy.backends.native import NativeBackend
from rwkv_easy.lora import (
    LoRALinear,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    merge_and_unload,
    merged_state_dict,
)
from rwkv_easy.models.checkpoint import (
    infer_config,
    load_model_from_checkpoint,
    read_checkpoint,
)
from rwkv_easy.models.rwkv7 import RWKV7, RWKV7Config
from rwkv_easy.state_tuning import (
    attach_state_tuning,
    freeze_all,
    state_state_dict,
)


@pytest.fixture
def model(tiny_model_cfg):
    torch.manual_seed(0)
    return RWKV7(tiny_model_cfg, NativeBackend(chunk=8), decay_mode="clampw")


def test_forward_shape(model, tiny_model_cfg):
    model.eval()
    ids = torch.randint(0, tiny_model_cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits, state = model(ids, need_state=False)
    assert logits.shape == (2, 16, tiny_model_cfg.vocab_size)
    assert state is None


def test_chunked_equals_full(model):
    model.eval()
    ids = torch.randint(0, model.cfg.vocab_size, (2, 25))
    with torch.no_grad():
        full, _ = model(ids, need_state=False)
        outputs = []
        state = None
        for start in range(0, 25, 7):
            logits, state = model(ids[:, start : start + 7], state=state, need_state=True)
            outputs.append(logits)
        chunked = torch.cat(outputs, dim=1)
    assert (full - chunked).abs().max() < 1e-5


def test_attention_mask_right_padding(model):
    """右 padding 不应该影响有效位置（infctx 的 shift 状态也一致）。"""
    model.eval()
    ids = torch.randint(0, model.cfg.vocab_size, (2, 12))
    mask = torch.ones(2, 12)
    mask[0, 8:] = 0
    ids[0, 8:] = 1
    with torch.no_grad():
        out_a, _ = model(ids, attention_mask=mask)
        out_b, _ = model(ids[:, :8], state=None, need_state=False)
    assert (out_a[0, :8] - out_b[0]).abs().max() < 1e-4


def test_all_decay_modes_run(model):
    model.eval()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    values = {}
    for mode in ("clampw", "softplus", "exp"):
        model.set_decay_mode(mode)
        with torch.no_grad():
            values[mode] = model(ids)[0].clone()
        assert torch.isfinite(values[mode]).all()
    assert model.decay_mode == "exp"


def test_inject_lora_freezes_base(model):
    report = inject_lora(
        model,
        target_modules=["att.receptance", "att.key", "att.value", "att.output", "ffn.key", "ffn.value"],
        rank=4,
        alpha=8,
    )
    assert len(report.targets) == 6 * model.cfg.n_layer
    # 只有 LoRA 参数可训练
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            assert parameter.requires_grad
        else:
            assert not parameter.requires_grad
    expected = sum(
        module.lora_A.numel() + module.lora_B.numel()
        for module in model.modules()
        if isinstance(module, LoRALinear)
    )
    assert report.trainable_parameters == expected
    assert report.trainable_parameters < report.total_parameters


def test_lora_zero_init_is_identity(model):
    inject_lora(model, ["att.key"], rank=4, alpha=8)
    model.eval()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 6))
    with torch.no_grad():
        out = model(ids)[0]
    assert torch.isfinite(out).all()
    # B 为零 => 初始 LoRA 不改变输出
    module = model.blocks[0].att.key
    assert isinstance(module, LoRALinear)
    assert torch.count_nonzero(module.lora_B) == 0


def test_lora_merge_math(model):
    inject_lora(model, ["att.key"], rank=4, alpha=8)
    module = model.blocks[0].att.key
    assert isinstance(module, LoRALinear)
    with torch.no_grad():
        module.lora_A.normal_()
        module.lora_B.normal_()
    expected = module.base.weight + (module.lora_B @ module.lora_A) * (8 / 4)
    assert (module.merged_weight() - expected).abs().max() < 1e-6

    x = torch.randn(3, module.base.in_features)
    with torch.no_grad():
        out = module(x)
    manual = module.base(x) + (x @ module.lora_A.T @ module.lora_B.T) * (8 / 4)
    assert (out - manual).abs().max() < 1e-5


def test_lora_merge_and_unload(model, tiny_model_cfg):
    inject_lora(model, ["att.key", "ffn.value"], rank=4, alpha=8)
    with torch.no_grad():
        for _, module in ((n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)):
            module.lora_A.normal_()
            module.lora_B.normal_()
    ids = torch.randint(0, tiny_model_cfg.vocab_size, (1, 7))
    model.eval()
    with torch.no_grad():
        before = model(ids)[0].clone()
    merged = copy.deepcopy(model)
    merge_and_unload(merged)
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    with torch.no_grad():
        after = merged(ids)[0]
    assert (before - after).abs().max() < 1e-4


def test_merged_state_dict_keys(model):
    inject_lora(model, ["att.key"], rank=4, alpha=8)
    state = merged_state_dict(model)
    assert not any("lora_" in key for key in state)
    assert not any(".base." in key for key in state)
    assert state["blocks.0.att.key.weight"].shape == model.blocks[0].att.key.base.weight.shape
    assert "emb.weight" in state and "head.weight" in state


def test_lora_state_dict_roundtrip(model):
    inject_lora(model, ["att.key"], rank=4, alpha=8)
    with torch.no_grad():
        module = model.blocks[0].att.key
        module.lora_A.normal_()
        module.lora_B.normal_()
    saved = lora_state_dict(model)
    other = copy.deepcopy(model)
    with torch.no_grad():
        for p in other.parameters():
            if p.requires_grad:
                p.zero_()
    load_lora_state_dict(other, saved)
    for key, value in saved.items():
        module_name, _, param = key.rpartition(".")
        target = getattr(dict(other.named_modules())[module_name], param)
        assert torch.equal(target.cpu(), value)


def test_state_tuning_freeze_and_grad(model):
    attach_state_tuning(model)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(name.endswith("time_state") for name in trainable)
    assert len(trainable) == model.cfg.n_layer

    model.train()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 8))
    logits = model(ids)[0]
    loss = logits.float().pow(2).mean()
    loss.backward()
    for block in model.blocks:
        assert block.att.time_state.grad is not None
        assert torch.isfinite(block.att.time_state.grad).all()


def test_state_tuning_changes_output(model):
    ids = torch.randint(0, model.cfg.vocab_size, (1, 5))
    model.eval()
    with torch.no_grad():
        before = model(ids)[0].clone()
    attach_state_tuning(model)
    with torch.no_grad():
        for block in model.blocks:
            block.att.time_state.normal_()
        after = model(ids)[0]
    assert (before - after).abs().max() > 1e-3


def test_state_state_dict_roundtrip(model):
    attach_state_tuning(model)
    with torch.no_grad():
        for block in model.blocks:
            block.att.time_state.normal_()
    saved = state_state_dict(model)
    assert len(saved) == model.cfg.n_layer
    fresh = copy.deepcopy(model)
    freeze_all(fresh)
    attach_state_tuning(fresh)
    with torch.no_grad():
        for block in fresh.blocks:
            block.att.time_state.zero_()
    from rwkv_easy.state_tuning import load_state_state_dict

    load_state_state_dict(fresh, saved)
    for key, value in saved.items():
        assert torch.equal(dict(fresh.named_parameters())[key].cpu(), value)


def test_infer_config_and_reload(tmp_path, tiny_checkpoint):
    path = tiny_checkpoint(vocab_size=256, n_layer=3, n_embd=48, head_size=12)
    state = read_checkpoint(path)
    cfg = infer_config(state)
    assert cfg.vocab_size == 256 and cfg.n_layer == 3 and cfg.n_embd == 48 and cfg.head_size == 12
    assert cfg.n_head == 4

    model, cfg2, signature = load_model_from_checkpoint(path)
    assert cfg2 == cfg
    assert len(signature) == 16
    ids = torch.randint(0, 256, (1, 8))
    model.eval()
    with torch.no_grad():
        logits = model(ids)[0]
    assert logits.shape == (1, 8, 256)


def test_infer_config_rejects_mismatch(tiny_checkpoint):
    path = tiny_checkpoint(vocab_size=256, n_layer=2, n_embd=32, head_size=8)
    state = read_checkpoint(path)
    with pytest.raises(ValueError):
        infer_config(state, n_layer=9)


def test_load_missing_key_raises(tmp_path, tiny_checkpoint):
    path = tiny_checkpoint(vocab_size=256, n_layer=2, n_embd=32, head_size=8)
    state = read_checkpoint(path)
    state.pop("blocks.0.att.key.weight")
    broken = tmp_path / "broken.pth"
    torch.save(state, broken)
    with pytest.raises(RuntimeError) as exc:
        load_model_from_checkpoint(broken)
    assert "缺少" in str(exc.value)


def test_initial_state_matches_learned_state(model):
    attach_state_tuning(model)
    with torch.no_grad():
        for block in model.blocks:
            block.att.time_state.normal_(std=0.1)
    state = model.initial_state(batch_size=3)
    for index, block in enumerate(model.blocks):
        assert torch.allclose(state.wkv[index], block.att.time_state.unsqueeze(0).expand(3, -1, -1, -1))
    ids = torch.randint(0, model.cfg.vocab_size, (3, 4))
    with torch.no_grad():
        with_state, _ = model(ids, state=state, need_state=True)
        without_state, _ = model(ids, need_state=True)
    assert (with_state - without_state).abs().max() < 1e-6


def test_checkpoint_prefix_stripping(tmp_path):
    from rwkv_easy.models.checkpoint import _strip_prefixes

    state = {"module.emb.weight": torch.zeros(4, 4), "module.head.weight": torch.zeros(4, 4)}
    assert set(_strip_prefixes(state)) == {"emb.weight", "head.weight"}
    wrapped = {"model.transformer.emb.weight": torch.zeros(4, 4)}
    assert set(_strip_prefixes(wrapped)) == {"transformer.emb.weight"}


def test_official_lowrank_dims():
    """低秩维度必须与官方 x070 公式一致，否则真实 checkpoint 的 g1/g2 形状对不上。"""
    from rwkv_easy.models.rwkv7 import TimeMix

    cfg = RWKV7Config(vocab_size=64, n_layer=2, n_embd=2048, head_size=64)
    layer = TimeMix(cfg, 0, NativeBackend(), decay_mode="clampw")
    assert layer.w1.shape == (2048, 96)   # 1.8 * sqrt(2048) = 81.5 -> 96
    assert layer.a1.shape == (2048, 96)
    assert layer.v1.shape == (2048, 64)   # 1.3 * sqrt(2048) = 58.8 -> 64
    assert layer.g1.shape == (2048, 256)  # 0.6 * 2048**0.8 = 262 -> 256
    assert layer.g2.shape == (256, 2048)
    assert layer.k_k.shape == (1, 1, 2048)
    assert layer.r_k.shape == (32, 64)    # n_head=32, head_size=64


def test_official_lowrank_dims_61_layers():
    from rwkv_easy.models.rwkv7 import TimeMix

    cfg = RWKV7Config(vocab_size=64, n_layer=61, n_embd=4096, head_size=64)
    layer = TimeMix(cfg, 0, NativeBackend(), decay_mode="clampw")
    assert layer.w1.shape == (4096, 192)
    assert layer.a1.shape == (4096, 192)
    assert layer.v1.shape == (4096, 128)
    assert layer.g1.shape == (4096, 384)
