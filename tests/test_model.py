"""模型层测试：形状、因果性、KV-Cache 等价性、损失掩码、优化器分组、采样器。

这些测试是「从零实现 Transformer」正确性的主要保证，尤其是
``test_kv_cache_matches_full_forward``：它同时验证了 RoPE 位置对齐、
GQA 扩展和缓存拼接三件事，任何一处写错都会在这里暴露。
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from codemind.config import ModelConfig  # noqa: E402
from codemind.model import (  # noqa: E402
    GPT,
    SamplingParams,
    build_rope_cache,
    causal_mask,
    rotate_half,
    sample_next_token,
)


@pytest.fixture()
def cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=128, n_layer=2, n_head=4, n_embd=32, n_kv_head=2, block_size=32
    )


@pytest.fixture()
def model(cfg: ModelConfig) -> GPT:
    torch.manual_seed(0)
    m = GPT(cfg)
    m.eval()
    return m


# ---------------------------------------------------------------------------- 形状与配置


def test_config_validates_divisibility():
    with pytest.raises(ValueError):
        ModelConfig(n_embd=30, n_head=4)
    with pytest.raises(ValueError):
        ModelConfig(n_head=4, n_kv_head=3)


def test_derived_dimensions(cfg: ModelConfig):
    assert cfg.head_dim == 8
    assert cfg.hidden_dim % 64 == 0
    assert cfg.hidden_dim >= cfg.n_embd
    assert cfg.n_kv_groups == 2


def test_preset_loading():
    cfg = ModelConfig.from_preset("nano", n_layer=1)
    assert cfg.n_layer == 1
    with pytest.raises(KeyError):
        ModelConfig.from_preset("does-not-exist")


def test_param_estimate_is_close_to_actual(cfg: ModelConfig, model: GPT):
    actual = model.num_params()
    estimate = cfg.estimated_params()
    assert abs(actual - estimate) / actual < 0.05, f"估计 {estimate} vs 实际 {actual}"


def test_forward_shapes(model: GPT, cfg: ModelConfig):
    x = torch.randint(0, cfg.vocab_size, (2, 7))
    logits, loss, aux = model(x)
    assert logits.shape == (2, 7, cfg.vocab_size)
    assert loss is None
    assert aux == {}


def test_forward_with_targets_returns_loss(model: GPT, cfg: ModelConfig):
    x = torch.randint(0, cfg.vocab_size, (2, 9))
    logits, loss, _ = model(x, x)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_loss_accepts_float32_mask(model: GPT, cfg: ModelConfig):
    """回归测试：loss_mask 可能是非 0/1 的浮点权重，不能因此报错。"""
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    mask = torch.full((2, 8), 2.0)
    _, loss, _ = model(x, x, loss_mask=mask)
    assert torch.isfinite(loss)


def test_block_size_overrun_raises(model: GPT, cfg: ModelConfig):
    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(ValueError):
        model(x)


def test_empty_input_raises(model: GPT):
    with pytest.raises(ValueError):
        model(torch.zeros((1, 0), dtype=torch.long))


# ---------------------------------------------------------------------------- 数值正确性


def test_rotate_half_is_negated_involution():
    """rotate_half 连用两次等于取负：它是 90° 旋转矩阵的"半操作"。

    第一次: [x0..x3 | x4..x7] -> [-x4..-x7 | x0..x3]
    第二次: 再交换并取负 -> [-x0..-x3 | -x4..-x7] = -x
    这正是 RoPE 的 ``x*cos + rotate_half(x)*sin`` 能构成正交旋转的原因。
    """
    x = torch.randn(2, 3, 4, 8)
    assert torch.allclose(rotate_half(rotate_half(x)), -x)
    # 单独验证一次交换+取负的语义
    half = x.shape[-1] // 2
    once = rotate_half(x)
    assert torch.allclose(once[..., :half], -x[..., half:])
    assert torch.allclose(once[..., half:], x[..., :half])


def test_rope_cache_shape_and_range():
    cos, sin = build_rope_cache(8, 16, 10000.0, torch.device("cpu"), torch.float32)
    assert cos.shape == (16, 8) and sin.shape == (16, 8)
    assert torch.allclose(cos[:, 0], cos[:, 1]), "配对维度必须相同"
    assert (cos**2 + sin**2).allclose(torch.ones_like(cos), atol=1e-5)


def test_causal_mask_blocks_future():
    device = torch.device("cpu")
    mask = causal_mask(3, 3, device)
    assert mask.shape == (1, 1, 3, 3)
    assert mask.dtype == torch.bool
    assert mask[0, 0].tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]
    # 增量解码：单查询对齐到最末尾（绝对位置 4），因此能看见全部 5 个 key。
    # 修复前这里会退化成"任何 t_q 行都全可见"，即因果性完全失效。
    inc = causal_mask(1, 5, device)
    assert inc[0, 0].tolist() == [[True, True, True, True, True]]
    # 两查询对五个 key：倒数第二行（绝对位置 3）看不到最后一个 key
    two = causal_mask(2, 5, device)
    assert two[0, 0].tolist() == [
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]
    # 一般规律：第 i 行恰有 (t_k - t_q + i + 1) 个可见 key
    for t_q in range(1, 6):
        for t_k in range(t_q, 8):
            m = causal_mask(t_q, t_k, device)[0, 0]
            for i in range(t_q):
                expected = t_k - t_q + i + 1
                assert int(m[i].sum()) == expected, f"t_q={t_q} t_k={t_k} row={i}"
                # 且可见的必须是**前** expected 个，不能是别的
                assert m[i].tolist() == [True] * expected + [False] * (t_k - expected)


def test_attention_is_causal(model: GPT, cfg: ModelConfig):
    """改动后面的 token 不能影响前面的输出。"""
    torch.manual_seed(1)
    base = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        logits_a, _, _ = model(base)
        modified = base.clone()
        modified[0, 5:] = torch.randint(0, cfg.vocab_size, (5,))
        logits_b, _, _ = model(modified)
    assert torch.allclose(logits_a[0, :5], logits_b[0, :5], atol=1e-5)


def test_kv_cache_matches_full_forward(model: GPT, cfg: ModelConfig):
    """KV-Cache 增量解码必须与一次性全量前向逐位一致（贪心路径）。"""
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        full_a, _, _ = model(ids, use_cache=True)
        caches = None
        pieces = []
        for t in range(ids.shape[1]):
            step = ids[:, t : t + 1]
            logits, _, aux = model(step, caches=caches, use_cache=True)
            caches = aux["caches"]
            pieces.append(logits)
        incremental = torch.cat(pieces, dim=1)
    assert incremental.shape == full_a.shape
    assert torch.allclose(full_a, incremental, atol=1e-4), "KV-Cache 路径与全量前向不一致"


def test_cache_lengths_grow(model: GPT, cfg: ModelConfig):
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    logits, _, aux = model(ids, use_cache=True)
    caches = aux["caches"]
    assert len(caches) == cfg.n_layer
    k, v = caches[0]
    assert k.shape == (1, cfg.n_kv_head, 4, cfg.head_dim)
    assert v.shape == (1, cfg.n_kv_head, 4, cfg.head_dim)


def test_gqa_reduces_cache_size(model: GPT, cfg: ModelConfig):
    """GQA 下 KV 头数必须小于 Q 头数，否则缓存不会缩小。"""
    assert cfg.n_kv_head < cfg.n_head
    ids = torch.randint(0, cfg.vocab_size, (1, 3))
    _, _, aux = model(ids, use_cache=True)
    k, _ = aux["caches"][0]
    assert k.shape[1] == cfg.n_kv_head


def test_loss_with_mask_ignores_masked_region(model: GPT, cfg: ModelConfig):
    """把某些位置从损失中排除，结果应等于只用剩余位置计算的交叉熵。"""
    torch.manual_seed(3)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = torch.randint(0, cfg.vocab_size, (1, 8))
    mask = torch.zeros((1, 8))
    mask[0, 2:5] = 1.0
    with torch.no_grad():
        _, masked_loss, _ = model(x, y, loss_mask=mask)
        logits, _, _ = model(x)
        expected = torch.nn.functional.cross_entropy(logits[0, 2:5].float(), y[0, 2:5])
    assert torch.allclose(masked_loss, expected, atol=1e-4)


def test_all_masked_returns_zero_loss(model: GPT, cfg: ModelConfig):
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    logits, loss, _ = model(x, x, loss_mask=torch.zeros((2, 6)))
    assert logits is None, "掩码路径不计算全量 logits"
    assert loss is not None
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_masked_loss_gradient_survives_with_tiny_reference(model: GPT, cfg: ModelConfig):
    """掩码全零时 loss 必须是可反传的零（用 x.sum()*0 构造），否则训练会崩。"""
    model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    _, loss, _ = model(x, x, loss_mask=torch.zeros((2, 6)))
    loss.backward()  # 不应抛异常
    assert any(p.grad is not None for p in model.parameters())


# ---------------------------------------------------------------------------- 训练相关


def test_backward_produces_finite_gradients(model: GPT, cfg: ModelConfig):
    model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss, _ = model(x, x)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "没有任何梯度"
    assert all(torch.isfinite(g).all() for g in grads)


def test_weight_tying_shares_storage(cfg: ModelConfig):
    m = GPT(cfg)
    assert m.lm_head.weight is m.tok_emb.weight


def test_untied_embeddings_have_own_params():
    cfg = ModelConfig(
        vocab_size=64, n_layer=1, n_head=2, n_embd=16, n_kv_head=1, block_size=16,
        tie_embeddings=False,
    )
    m = GPT(cfg)
    assert m.lm_head.weight is not m.tok_emb.weight
    assert m.num_params() == cfg.estimated_params()


def test_optimizer_groups_split_by_dimension(model: GPT):
    opt = model.configure_optimizer(0.1, 1e-3, (0.9, 0.95), "cpu")
    assert len(opt.param_groups) == 2
    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() == 1 for p in no_decay_group["params"])


def test_optimizer_does_not_duplicate_tied_weights(model: GPT):
    opt = model.configure_optimizer(0.1, 1e-3, (0.9, 0.95), "cpu")
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "共享权重被重复加入优化器"


def test_rmsnorm_normalizes():
    from codemind.model import RMSNorm

    norm = RMSNorm(16)
    x = torch.randn(4, 16) * 100
    out = norm(x)
    assert torch.allclose(out.pow(2).mean(dim=-1), torch.ones(4), atol=1e-4)


def test_attention_maps_shape(model: GPT, cfg: ModelConfig):
    x = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        _, _, aux = model(x, return_attn=True)
    maps = aux["attn"]
    assert len(maps) == cfg.n_layer
    assert maps[0].shape == (1, cfg.n_head, 5, 5)


# ---------------------------------------------------------------------------- 采样器


def test_greedy_picks_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    out = sample_next_token(logits, SamplingParams(greedy=True))
    assert int(out[0, 0]) == 1
    out = sample_next_token(logits, SamplingParams(temperature=0.0))
    assert int(out[0, 0]) == 1


def test_banned_ids_never_sampled():
    logits = torch.zeros(1, 5)
    logits[0, 3] = 100.0
    params = SamplingParams(temperature=1.0, top_k=0, top_p=1.0, banned_ids=(3,))
    for _ in range(50):
        out = sample_next_token(logits, params)
        assert int(out[0, 0]) != 3


def test_top_k_restricts_support():
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, 0.0, 0.0, 10.0, 9.0]])
    params = SamplingParams(temperature=1.0, top_k=2, top_p=1.0)
    seen = {int(sample_next_token(logits, params)[0, 0]) for _ in range(200)}
    assert seen <= {3, 4}


def test_temperature_zero_equivalent_to_greedy():
    logits = torch.tensor([[0.1, 0.2, 0.7]])
    a = sample_next_token(logits, SamplingParams(temperature=0.0))
    b = sample_next_token(logits, SamplingParams(greedy=True))
    assert int(a[0, 0]) == int(b[0, 0]) == 2


def test_repetition_penalty_downweights_seen_tokens():
    logits = torch.tensor([[5.0, 4.0]])
    prev = torch.tensor([[0]])
    penalized = sample_next_token(
        logits, SamplingParams(temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=4.0), prev
    )
    assert int(penalized[0, 0]) == 1, "已出现过的 token 应被压低"


def test_sampling_params_validation():
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.0)
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1.0)
    with pytest.raises(ValueError):
        SamplingParams(repetition_penalty=0.0)


def test_infill_default_repetition_penalty_is_1_2():
    """回归：``Generator.infill`` 的默认重复惩罚是 1.2，不是 1.05。

    这是实测调出来的（step-2000 检查点、300 个留出文档）：相对 1.05，
    ``rep8`` 60.68% → 46.30%（z=−7.36）、``rep_line`` 21.76% → 15.01%（z=−3.23），
    而 ``edit_sim``/``syntax_ok`` 都没有付出代价。
    **把它调小会退回"复读"退化模式**——见 AGENTS.md 杠杆排序 ①(b)。
    """
    import inspect

    from codemind.generate import INFILL_DEFAULT_REPETITION_PENALTY, Generator

    assert INFILL_DEFAULT_REPETITION_PENALTY == 1.2
    assert "INFILL_DEFAULT_REPETITION_PENALTY" in inspect.getsource(Generator.infill), (
        "infill 必须用具名常量作默认值，否则这个实测结论会被无声改掉"
    )


def test_seed_makes_sampling_reproducible():
    logits = torch.randn(1, 20)
    params = SamplingParams(temperature=1.0, top_k=0, top_p=1.0, seed=42)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = sample_next_token(logits, params, generator=g1)
    b = sample_next_token(logits, params, generator=g2)
    assert int(a[0, 0]) == int(b[0, 0])


# ---------------------------------------------------------------------------- 生成


def test_generate_respects_max_tokens(model: GPT, cfg: ModelConfig):
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    out = model.generate(prompt, 5, SamplingParams(greedy=True))
    assert out.shape == (1, 8)


def test_generate_streams_tokens(model: GPT, cfg: ModelConfig):
    prompt = torch.randint(0, cfg.vocab_size, (1, 2))
    collected = []
    model.generate(
        prompt, 4, SamplingParams(greedy=True), stream_callback=lambda tid, piece: collected.append(tid)
    )
    assert len(collected) == 4


def test_generate_stops_on_stop_token(model: GPT, cfg: ModelConfig):
    prompt = torch.randint(0, cfg.vocab_size, (1, 2))
    with torch.no_grad():
        first = model.generate(prompt, 1, SamplingParams(greedy=True))
    stop = int(first[0, -1])
    out = model.generate(prompt, 10, SamplingParams(greedy=True), stop_ids=(stop,))
    assert out.shape[1] == 3, "遇到停止词元应立即结束"


def test_generate_handles_context_overflow(model: GPT, cfg: ModelConfig):
    """超出 block_size 时自动裁剪上下文，而不是抛异常。"""
    prompt = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 5))
    out = model.generate(prompt, 3, SamplingParams(greedy=True))
    assert out.shape[1] > cfg.block_size


def test_generate_without_cache_matches_with_cache(model: GPT, cfg: ModelConfig):
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    params = SamplingParams(greedy=True)
    a = model.generate(prompt, 6, params, use_cache=True)
    b = model.generate(prompt, 6, params, use_cache=False)
    assert torch.equal(a, b), "缓存与非缓存路径的贪心结果必须一致"


# ---------------------------------------------------------------------------- 持久化


def test_save_and_load_roundtrip(tmp_path: Path, model: GPT):
    path = tmp_path / "m.pt"
    model.save(path, {"iteration": 7})
    loaded, payload = GPT.from_pretrained(path)
    assert payload["iteration"] == 7
    assert loaded.cfg.vocab_size == model.cfg.vocab_size
    x = torch.randint(0, model.cfg.vocab_size, (1, 5))
    with torch.no_grad():
        a, _, _ = model(x)
        b, _, _ = loaded(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_from_pretrained_rejects_foreign_file(tmp_path: Path):
    path = tmp_path / "foreign.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError):
        GPT.from_pretrained(path)


def test_standalone_fim_payload_runs_through_model():
    """回归测试：独立（非梯度累积）构造的 FIM 批次必须能直接喂给模型。

    当 loss_mask 生效时，lm_head 只会收到 2D 输入；这条路径必须不依赖 batch 维。
    """
    import random

    from codemind.data import fim_transform_ids
    from codemind.tokenizer import BPETokenizer

    tok = BPETokenizer()
    tok.train_from_texts(
        ["def f(x):\n    return x + 1\n\n\nprint(f(2))\n"] * 4, vocab_size=300, verbose=False
    )
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, n_layer=1, n_head=2, n_embd=16, n_kv_head=1, block_size=64
    )
    torch.manual_seed(0)
    m = GPT(cfg)
    rows = []
    masks = []
    for seed in range(3):
        ids = tok.encode("def f(x):\n    return x + 1\n\n\nprint(f(2))\n")
        out, mask = fim_transform_ids(ids, tok, random.Random(seed), spm_rate=0.0)
        payload = out[:64]
        mrow = mask[:64]
        while len(payload) < 64:
            payload.append(0)
            mrow.append(0.0)
        rows.append(payload)
        masks.append(mrow)

    x = torch.tensor([r[:-1] for r in rows], dtype=torch.long)
    y = torch.tensor([r[1:] for r in rows], dtype=torch.long)
    loss_mask = torch.tensor([m[1:] for m in masks], dtype=torch.float32)
    _, loss, _ = m(x, y, loss_mask=loss_mask)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_dropout_active_in_train_mode(cfg: ModelConfig):
    cfg = ModelConfig(**{**cfg.to_dict(), "dropout": 0.5})
    torch.manual_seed(0)
    m = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 6))
    m.train()
    with torch.no_grad():
        a, _, _ = m(x)
        b, _, _ = m(x)
    assert not torch.allclose(a, b), "训练模式下 dropout 应当生效"


# ═══════════════ 梯度检查点（grad_checkpoint）═══════════════
#
# 这个开关的作用是「用计算换显存」，判据有两条：
#   1. **数值必须等价** —— 开与不开，loss 和梯度必须一致（否则训练语义变了）
#   2. **只在训练模式生效** —— 生成/评估路径绝不能用（拿不到 cache，且白慢）
#
# 2026-11 实测背景：1B + ctx 1024 + batch 16 的激活约 50 GB，开检查点后约 12 GB。
# 这是「32 GB 双 T4 上能否装下 1B 训练」的决定性开关。


def test_grad_checkpoint_matches_plain_forward(cfg: ModelConfig) -> None:
    """开/不开梯度检查点，loss 与梯度必须一致。"""
    torch.manual_seed(1234)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    tgt = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))

    model.train()
    model.zero_grad(set_to_none=True)
    _, loss_plain, _ = model(idx, tgt, logits_for_all=False, grad_checkpoint=False)
    loss_plain.backward()
    grads_plain = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }

    model.zero_grad(set_to_none=True)
    _, loss_ckpt, _ = model(idx, tgt, logits_for_all=False, grad_checkpoint=True)
    loss_ckpt.backward()
    grads_ckpt = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }

    assert float(loss_ckpt.detach()) == pytest.approx(float(loss_plain.detach()), rel=0, abs=1e-6)
    assert set(grads_plain) == set(grads_ckpt)
    assert grads_plain, "没有任何梯度，测试无意义"
    for name in grads_plain:
        assert torch.allclose(grads_plain[name], grads_ckpt[name], atol=1e-6), name


def test_grad_checkpoint_disabled_in_eval_mode(cfg: ModelConfig) -> None:
    """eval 模式下即使传 grad_checkpoint=True 也不能走检查点路径。

    关闭 dropout 后，训练模式与 eval 模式的前向应当一致。
    """
    torch.manual_seed(99)
    cfg0 = ModelConfig(
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        n_kv_head=cfg.n_kv_head,
        block_size=cfg.block_size,
        dropout=0.0,
    )
    model = GPT(cfg0)
    idx = torch.randint(0, cfg0.vocab_size, (2, cfg0.block_size))
    tgt = torch.randint(0, cfg0.vocab_size, (2, cfg0.block_size))

    model.train()
    _, loss_train, _ = model(idx, tgt, logits_for_all=False, grad_checkpoint=True)

    model.eval()
    with torch.no_grad():
        _, loss_eval, _ = model(idx, tgt, logits_for_all=False, grad_checkpoint=True)

    assert float(loss_eval) == pytest.approx(float(loss_train), rel=0, abs=1e-6)


def test_grad_checkpoint_skipped_when_cache_used(cfg: ModelConfig) -> None:
    """use_cache=True 时必须退回普通路径 —— 检查点拿不到 KV cache。"""
    model = GPT(cfg)
    model.train()
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    tgt = torch.randint(0, cfg.vocab_size, (1, 4))
    _, loss, aux = model(
        idx, tgt, use_cache=True, logits_for_all=False, grad_checkpoint=True
    )
    assert loss is not None
    # KW-Cache 必须真的产出了（n_layer 个），说明没走检查点分支
    caches = aux.get("caches") if isinstance(aux, dict) else None
    assert caches is not None and len(caches) == cfg.n_layer
