"""训练与生成集成测试：用极小模型跑通「分片 -> 训练 -> 续训 -> 生成」全链路。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import pytest

torch = pytest.importorskip("torch")

from codemind.config import DataConfig, ModelConfig, TrainConfig, pick_dtype  # noqa: E402
from codemind.data import TokenShards, build_shards, collect_code_files  # noqa: E402
from codemind.generate import Generator, describe_checkpoint, load_runner  # noqa: E402
from codemind.model import SamplingParams  # noqa: E402
from codemind.tokenizer import BPETokenizer  # noqa: E402
from codemind.train import TrainReport, estimate_training_time, load_checkpoint, train  # noqa: E402

CODE = (
    "def add(a, b):\n    return a + b\n\n\n"
    "def sub(a, b):\n    return a - b\n\n\n"
    "def mul(a, b):\n    return a * b\n\n\n"
    "class Calc:\n    def __init__(self):\n        self.value = 0\n\n"
    "    def reset(self):\n        self.value = 0\n        return self\n"
)


# ---------------------------------------------------------------------------- 夹具


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> Tuple[BPETokenizer, TokenShards, Path]:
    """建立一个可复用的小型语料 + 分词器 + 分片。"""
    root = tmp_path_factory.mktemp("ws")
    corpus = root / "corpus"
    corpus.mkdir()
    for i in range(30):
        (corpus / f"m{i}.py").write_text(
            CODE + f"\n\ndef variant_{i}(x):\n    return x + {i}\n", encoding="utf-8"
        )
    tokenizer = BPETokenizer()
    tokenizer.train_from_texts(
        [p.read_text(encoding="utf-8") for p in corpus.glob("*.py")],
        vocab_size=300,
        verbose=False,
    )
    cfg = DataConfig(source_dirs=[str(corpus)], out_dir=str(root / "data"), name="t", val_ratio=0.2)
    shards = build_shards(collect_code_files(cfg, verbose=False), tokenizer, cfg, verbose=False)
    return tokenizer, shards, root


@pytest.fixture(scope="module")
def trained(workspace) -> Tuple[Generator, TrainReport, Path]:
    tokenizer, shards, root = workspace
    tokenizer.save(root / "tokenizer.json")
    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size, n_layer=2, n_head=2, n_embd=32, n_kv_head=1, block_size=32
    )
    train_cfg = TrainConfig(
        out_dir=str(root / "run"),
        batch_size=4,
        max_iters=120,
        lr=5e-3,
        eval_interval=40,
        eval_batches=3,
        log_interval=0,
        ckpt_interval=60,
        fim_rate=0.5,
        resume=False,
        device="cpu",
        num_threads=2,
        seed=0,
    )
    report = train(model_cfg, train_cfg, shards=shards, tokenizer=tokenizer, verbose=False)
    gen = load_runner(root / "run", checkpoint="best", device="cpu")
    return gen, report, root


# ---------------------------------------------------------------------------- 学习信号


def test_training_reduces_loss(trained):
    """核心断言：训练后必须显著优于随机初始化。"""
    _, report, _ = trained
    assert report.iterations == 120
    baseline = math.log(300)  # 均匀分布的交叉熵 ≈ 5.7
    assert report.final_train < baseline * 0.9, (
        f"训练损失 {report.final_train:.4f} 未明显低于随机基线 {baseline:.4f}"
    )
    assert report.best_val < baseline


def test_report_fields(trained):
    _, report, root = trained
    assert report.tokens_seen > 0
    assert report.seconds > 0
    assert report.tokens_per_second > 0
    assert report.out_dir.exists()
    assert report.best_ckpt is not None and Path(report.best_ckpt).exists()
    assert "训练完成" in report.summary()


def test_artifacts_written(trained):
    _, _, root = trained
    run = root / "run"
    for name in ("best.pt", "latest.pt", "tokenizer.json", "log.json"):
        assert (run / name).exists(), f"缺少产物 {name}"


def test_log_json_shape(trained):
    import json

    _, _, root = trained
    payload = json.loads((root / "run" / "log.json").read_text(encoding="utf-8"))
    assert payload["final"]["iteration"] == 120
    assert payload["final"]["tokens_seen"] > 0
    assert payload["model_config"]["n_layer"] == 2
    assert any("val_loss" in entry for entry in payload["history"])


# ---------------------------------------------------------------------------- 检查点


def test_checkpoint_metadata(trained):
    _, _, root = trained
    info = describe_checkpoint(root / "run" / "best.pt")
    assert info["format"] == "codemind-checkpoint-v1"
    assert info["iteration"] > 0
    assert info["model_config"]["n_layer"] == 2
    assert info["size_mb"] > 0


def test_checkpoint_roundtrip(trained):
    _, _, root = trained
    # latest.pt 是续训入口，必须带优化器状态
    payload = load_checkpoint(root / "run" / "latest.pt")
    assert "optimizer" in payload
    assert payload["has_optimizer"] is True
    assert payload["train_config"]["batch_size"] == 4
    assert payload["iteration"] == 120


def test_best_checkpoint_is_weights_only(trained):
    """best.pt 只存权重：它是推理入口，省掉一倍的优化器动量空间。"""
    _, _, root = trained
    best = root / "run" / "best.pt"
    latest = root / "run" / "latest.pt"
    payload = load_checkpoint(best)
    assert payload.get("has_optimizer") is False
    assert "optimizer" not in payload
    assert payload["val_loss"] is not None
    assert best.stat().st_size < latest.stat().st_size, "best.pt 应显著小于 latest.pt"
    # 权重本身必须完整可载入
    gen = load_runner(root / "run", checkpoint="best", device="cpu")
    assert gen.model.cfg.n_layer == 2


def test_resume_continues_from_step(workspace, tmp_path: Path):
    """从 latest.pt 续训时，不应从头开始，且步数要累加到新目标。"""
    tokenizer, _, _ = workspace
    tokenizer.save(tmp_path / "tokenizer.json")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for i in range(10):
        (corpus / f"m{i}.py").write_text(
            CODE + f"\n\ndef v{i}(x):\n    return x + {i}\n", encoding="utf-8"
        )
    data_cfg = DataConfig(source_dirs=[str(corpus)], out_dir=str(tmp_path / "data"), name="r", val_ratio=0.0)
    shards = build_shards(collect_code_files(data_cfg, verbose=False), tokenizer, data_cfg, verbose=False)
    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size, n_layer=1, n_head=2, n_embd=16, n_kv_head=1, block_size=32
    )
    run = tmp_path / "run"

    def _cfg(iters: int, resume: bool) -> TrainConfig:
        return TrainConfig(
            out_dir=str(run),
            batch_size=4,
            max_iters=iters,
            eval_interval=0,
            eval_batches=1,
            log_interval=0,
            ckpt_interval=iters,
            fim_rate=0.0,
            resume=resume,
            device="cpu",
            num_threads=2,
            seed=0,
        )

    train(model_cfg, _cfg(20, False), shards=shards, tokenizer=tokenizer, verbose=False)
    first = load_checkpoint(run / "latest.pt")
    assert first["iteration"] == 20

    train(model_cfg, _cfg(30, True), shards=shards, tokenizer=tokenizer, verbose=False)
    resumed = load_checkpoint(run / "latest.pt")
    assert resumed["iteration"] == 30, "续训必须累加步数而不是从头开始"
    assert resumed["best_val"] == first["best_val"], "续训应保留历史最优记录"


def test_load_missing_run_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_runner(tmp_path / "nope")


def test_vocab_mismatch_is_rejected(trained, tmp_path: Path):
    """分词器与模型词表必须匹配，否则必须明确报错而不是静默产出垃圾。"""
    _, _, root = trained
    payload = torch.load(root / "run" / "best.pt", map_location="cpu", weights_only=False)
    other_tok = BPETokenizer()
    other_tok.train_from_texts([CODE * 6], vocab_size=280, verbose=False)
    other_tok.save(root / "other_tokenizer.json")
    assert other_tok.vocab_size != payload["model_config"]["vocab_size"]
    with pytest.raises(ValueError):
        load_runner(root / "run", checkpoint="best", tokenizer_path=root / "other_tokenizer.json")


# ---------------------------------------------------------------------------- 生成


def test_complete_returns_new_text_only(trained):
    gen, _, _ = trained
    prompt = "def add(a, b):\n"
    out = gen.complete(prompt, max_new_tokens=12, params=SamplingParams(greedy=True, seed=0))
    assert isinstance(out, str)
    assert not out.startswith(prompt), "返回值应是新增内容，不应包含提示"


def test_generation_is_reproducible_with_seed(trained):
    gen, _, _ = trained
    params = SamplingParams(greedy=True, seed=1234)
    a = gen.complete("def add(a, b):\n", max_new_tokens=16, params=params)
    b = gen.complete("def add(a, b):\n", max_new_tokens=16, params=params)
    assert a == b


def test_infill_produces_something_usable(trained):
    gen, _, _ = trained
    prefix = "def add(a, b):\n"
    suffix = "\n\ndef sub(a, b):\n"
    filled = gen.infill(prefix, suffix, max_new_tokens=16, params=SamplingParams(greedy=True))
    assert isinstance(filled, str)
    assert "<|fim" not in filled, "特殊词元不应泄漏到结果里"


def test_streaming_callback_fires(trained):
    gen, _, _ = trained
    pieces = []
    gen.complete(
        "def add(a, b):\n",
        max_new_tokens=8,
        params=SamplingParams(greedy=True),
        on_token=lambda tid, piece: pieces.append(piece),
    )
    assert len(pieces) == 8


def test_generator_exposes_block_size(trained):
    gen, _, _ = trained
    assert gen.block_size == 32


def test_empty_prompt_still_generates(trained):
    gen, _, _ = trained
    out = gen.complete("", max_new_tokens=4, params=SamplingParams(greedy=True))
    assert isinstance(out, str)


def test_long_prompt_is_truncated_not_crashing(trained):
    gen, _, _ = trained
    out = gen.complete("x = 1\n" * 200, max_new_tokens=4, params=SamplingParams(greedy=True))
    assert isinstance(out, str)


def test_banned_special_tokens_never_appear(trained):
    gen, _, _ = trained
    for _ in range(3):
        out = gen.complete("def f():\n", max_new_tokens=24, params=SamplingParams(temperature=1.0))
        assert "<|pad|>" not in out
        assert "<|fim_begin|>" not in out


def test_complete_batch(trained):
    gen, _, _ = trained
    outs = gen.complete_batch(["def add(a, b):\n", "x = "], max_new_tokens=4)
    assert len(outs) == 2 and all(isinstance(o, str) for o in outs)


def test_attention_map_shape(trained):
    gen, _, _ = trained
    attention = gen.attention_map("def add(a, b):\n    return a + b\n")
    assert attention.dim() == 4
    assert attention.shape[0] == 2  # n_layer
    assert attention.shape[1] == 2  # n_head


# ---------------------------------------------------------------------------- 配置与工具


def test_lr_schedule_shape():
    cfg = TrainConfig(lr=1.0, max_iters=100, warmup_iters=10, min_lr_ratio=0.1)
    assert cfg.lr_at(0) < cfg.lr_at(9), "warmup 阶段应递增"
    assert cfg.lr_at(9) == pytest.approx(1.0)
    assert cfg.lr_at(100) == pytest.approx(0.1, abs=1e-6)
    assert cfg.lr_at(50) < cfg.lr_at(10), "退火阶段应递减"
    assert cfg.lr_at(50) > cfg.lr_at(100)


def test_lr_schedule_auto_warmup():
    """自动 warmup = **3%**，不是 10%。

    ⚠️ 这条断言被改过一次，原因是**实测推翻了自己**：
    云端 78.66M 模型跑 20,000 步时，10% 的 warmup 是 2,000 步 ——
    模型在 LR 还在爬升的区间就被推到了 3e-3，损失在 step 1001（lr=1.5e-3）
    触底 2.8683 后**一路反弹到 3.67**，再没回到最低点。

    所以 warmup 缩短到 3%（20,000 步 → 600 步），峰值 LR 也降到 1e-3。
    **这条测试锁的是"当前认为正确的值"，而那个值来自一次真实的发散事故。**
    """
    cfg = TrainConfig(lr=1.0, max_iters=1000, warmup_iters=0)
    assert cfg.resolved_warmup() == 30


def test_default_lr_is_conservative():
    """默认峰值 LR 必须是 1e-3 —— 3e-3 在 78M 上实测发散过。"""
    assert TrainConfig().lr == 1e-3
    # step 1000 时的 LR 必须显著低于旧的 3e-3 —— 那正是发散区间
    cfg = TrainConfig(max_iters=20000)
    assert cfg.lr_at(1000) < 1.2e-3


def test_tokens_per_iter():
    cfg = TrainConfig(batch_size=8, grad_accum=4)
    assert cfg.tokens_per_iter(256) == 8 * 4 * 256


def test_pick_dtype_rules():
    assert pick_dtype("off", torch.device("cpu")) == (None, False)
    assert pick_dtype("bf16", torch.device("cpu"))[0] is torch.bfloat16
    with pytest.raises(ValueError):
        pick_dtype("fp16", torch.device("cpu"))
    with pytest.raises(ValueError):
        pick_dtype("nonsense", torch.device("cpu"))


def test_estimate_training_time():
    model_cfg = ModelConfig(vocab_size=300, n_layer=1, n_head=2, n_embd=16, n_kv_head=1, block_size=32)
    train_cfg = TrainConfig(batch_size=8, max_iters=100)
    text = estimate_training_time(model_cfg, train_cfg, tokens_per_second=1000.0)
    assert "分钟" in text


def test_train_config_json_roundtrip():
    cfg = TrainConfig(batch_size=7, lr=0.01)
    restored = TrainConfig.from_dict(cfg.to_dict())
    assert restored.batch_size == 7
    assert restored.lr == 0.01


def test_unknown_config_fields_are_ignored(capsys):
    payload = TrainConfig().to_dict()
    payload["field_from_the_future"] = 1
    cfg = TrainConfig.from_dict(payload)
    assert cfg.batch_size == TrainConfig().batch_size
