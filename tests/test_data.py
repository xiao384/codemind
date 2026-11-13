"""数据流水线测试：语料采集、分片、随机切窗、FIM 变换。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

import numpy as np
import pytest

from codemind.config import DataConfig
from codemind.data import (
    RandomWindowSampler,
    TokenShards,
    apply_fim,
    build_shards,
    collect_code_files,
    fim_transform_ids,
    iter_documents,
    read_file_text,
)
from codemind.tokenizer import FIM_BEGIN, FIM_END, FIM_HOLE, BPETokenizer

CODE = (
    "def add(a, b):\n    return a + b\n\n\n"
    "def sub(a, b):\n    return a - b\n\n\n"
    "class Calc:\n    def __init__(self):\n        self.v = 0\n"
)


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    t = BPETokenizer()
    t.train_from_texts([CODE * 5, CODE.replace("add", "mul") * 5], vocab_size=420, verbose=False)
    return t


def _make_corpus(root: Path, count: int = 6) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (root / f"m{i}.py").write_text(CODE + f"\n\ndef f{i}(x):\n    return x + {i}\n", encoding="utf-8")
    (root / "note.md").write_text("# docs\n\nsome text\n" * 3, encoding="utf-8")
    (root / "binary.bin").write_bytes(b"\x00\x01\x02" * 30)
    (root / "tiny.py").write_text("x=1\n", encoding="utf-8")
    (root / "generated.py").write_text("# @generated\nx = 1\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------- 采集


def test_collect_filters_by_extension_and_size(tmp_path: Path):
    corpus = _make_corpus(tmp_path / "src")
    cfg = DataConfig(source_dirs=[str(corpus)], min_file_bytes=32)
    files = collect_code_files(cfg, verbose=False)
    names = {f.name for f in files}
    assert "m0.py" in names
    assert "note.md" in names
    assert "binary.bin" not in names, "二进制文件应被拒绝"
    assert "tiny.py" not in names, "过小文件应被拒绝"
    assert "generated.py" not in names, "自动生成标记文件应被跳过"


def test_collect_dedup(tmp_path: Path):
    root = tmp_path / "dup"
    root.mkdir()
    (root / "a.py").write_text(CODE, encoding="utf-8")
    (root / "b.py").write_text(CODE, encoding="utf-8")
    (root / "c.py").write_text(CODE + "\n# different\n", encoding="utf-8")
    cfg = DataConfig(source_dirs=[str(root)], dedup=True)
    assert len(collect_code_files(cfg, verbose=False)) == 2
    cfg_nodedup = DataConfig(source_dirs=[str(root)], dedup=False)
    assert len(collect_code_files(cfg_nodedup, verbose=False)) == 3


def test_collect_respects_max_files(tmp_path: Path):
    corpus = _make_corpus(tmp_path / "src", count=10)
    cfg = DataConfig(source_dirs=[str(corpus)], max_files=3)
    assert len(collect_code_files(cfg, verbose=False)) == 3


def test_collect_missing_dir_is_not_fatal(tmp_path: Path):
    cfg = DataConfig(source_dirs=[str(tmp_path / "nope")])
    assert collect_code_files(cfg, verbose=False) == []


def test_read_file_text_handles_binary_and_broken_utf8(tmp_path: Path):
    binary = tmp_path / "b.bin"
    binary.write_bytes(b"\x00\x01\x02")
    assert read_file_text(binary) is None
    broken = tmp_path / "bad.py"
    broken.write_bytes(b"x = '\xff\xfe'\n")
    text = read_file_text(broken)
    assert text is not None and "x = " in text


def test_exclude_dirs_default_hits_venv(tmp_path: Path):
    root = tmp_path / "proj"
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "dep.py").write_text(CODE, encoding="utf-8")
    (root / "app.py").write_text(CODE, encoding="utf-8")
    cfg = DataConfig(source_dirs=[str(root)])
    names = {f.name for f in collect_code_files(cfg, verbose=False)}
    assert names == {"app.py"}


# ---------------------------------------------------------------------------- 分片


def test_build_shards_and_reload(tmp_path: Path, tok: BPETokenizer):
    corpus = _make_corpus(tmp_path / "src", count=12)
    cfg = DataConfig(source_dirs=[str(corpus)], out_dir=str(tmp_path / "data"), name="t")
    files = collect_code_files(cfg, verbose=False)
    shards = build_shards(files, tok, cfg, verbose=False)
    assert shards.total_tokens > 0
    assert shards.n_docs > 0
    assert shards.vocab_size == tok.vocab_size
    assert shards.train_paths, "必须产生训练分片"
    for p in shards.train_paths + shards.val_paths:
        assert p.exists() and p.stat().st_size > 0
    reloaded = TokenShards.load(shards.meta_path)
    assert reloaded.total_tokens == shards.total_tokens
    assert reloaded.train_files == shards.train_files
    assert TokenShards.load(shards.dir).name == "t"


def test_shards_use_uint16_for_small_vocab(tmp_path: Path, tok: BPETokenizer):
    corpus = _make_corpus(tmp_path / "src")
    cfg = DataConfig(source_dirs=[str(corpus)], out_dir=str(tmp_path / "d"), name="v")
    shards = build_shards(collect_code_files(cfg, verbose=False), tok, cfg, verbose=False)
    assert shards.dtype == "uint16"


def test_shards_contain_eot_separators(tmp_path: Path, tok: BPETokenizer):
    corpus = _make_corpus(tmp_path / "src", count=4)
    cfg = DataConfig(source_dirs=[str(corpus)], out_dir=str(tmp_path / "d"), name="e", val_ratio=0.0)
    shards = build_shards(collect_code_files(cfg, verbose=False), tok, cfg, verbose=False)
    arr = np.fromfile(shards.train_paths[0], dtype=np.uint16)
    assert (arr == tok.eot_id).sum() >= 4, "每个文档后应有 EOT 分隔符"


def test_single_shard_splits_at_token_level(tmp_path: Path, tok: BPETokenizer):
    """只有一个分片时也要能切出验证集，且训练/验证不重叠。"""
    corpus = _make_corpus(tmp_path / "src", count=8)
    cfg = DataConfig(source_dirs=[str(corpus)], out_dir=str(tmp_path / "d"), name="s", val_ratio=0.2)
    shards = build_shards(
        collect_code_files(cfg, verbose=False), tok, cfg, verbose=False, block_size=32
    )
    assert shards.val_paths, "必须切出验证分片"
    train = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in shards.train_paths])
    val = np.concatenate([np.fromfile(p, dtype=np.uint16) for p in shards.val_paths])
    assert train.size > 0 and val.size > 0
    assert train.size + val.size == shards.total_tokens


def test_few_files_still_get_validation_split(tmp_path: Path, tok: BPETokenizer):
    """回归测试：文件数很少时也必须切出验证集。

    早期版本用「有多少文件被分到验证集（n_val）」作为是否切分的判据。
    文件数少时 ``int(n_files * val_ratio)`` 恒为 0，导致单分片 token 级切分
    这条路径永远执行不到 —— 小语料永远拿不到验证集，训练只能拿训练损失自欺。
    """
    corpus = _make_corpus(tmp_path / "src", count=4)
    cfg = DataConfig(
        source_dirs=[str(corpus)], out_dir=str(tmp_path / "d"), name="few", val_ratio=0.01
    )
    shards = build_shards(
        collect_code_files(cfg, verbose=False), tok, cfg, verbose=False, block_size=32
    )
    assert shards.val_paths, "文件数少 + 小 val_ratio 时仍必须切出验证集"
    assert shards.train_paths
    # 验证集应当明显小于训练集（1% 切分），但不能为空
    val_tokens = shards.tokens("val")
    train_tokens = shards.tokens("train")
    assert 0 < val_tokens < train_tokens
    assert val_tokens + train_tokens == shards.total_tokens


def test_val_ratio_respected_with_multiple_shards(tmp_path: Path, tok: BPETokenizer):
    """多分片时 ``val_ratio`` 必须真正被遵守，而不是「至少切走一个整分片」。

    回归测试：旧实现是 ``min(n_shards-1, max(1, round(n_shards*val_ratio)))``，
    下限 1 会盖过比例 —— 实测 4 个分片 + ``val_ratio=0.01`` 也被切走 12.6% 的 token。

    语料要够大：切分下限是 ``max(64, 2*block_size) = 64`` 个 token，
    如果目标验证集比这还小，切分会被正当地拒绝并退回"整个末分片做验证"。
    """
    corpus = _make_corpus(tmp_path / "src", count=20)
    cfg = DataConfig(
        source_dirs=[str(corpus)], out_dir=str(tmp_path / "d"), name="ratio", val_ratio=0.05
    )
    shards = build_shards(
        collect_code_files(cfg, verbose=False), tok, cfg,
        verbose=False, shard_tokens=400, block_size=32,
    )
    assert len(shards.train_files) > 1, "前置条件：本用例需要多个分片"
    val_tokens = shards.tokens("val")
    train_tokens = shards.tokens("train")
    assert val_tokens + train_tokens == shards.total_tokens
    assert val_tokens >= 64, "前置条件：目标验证集必须高于切分下限，否则本用例没有意义"
    ratio = val_tokens / shards.total_tokens
    assert 0.02 <= ratio <= 0.10, (
        f"请求 val_ratio=0.05，实际 {ratio:.3f} —— 偏离过大说明又被整分片下限盖过了"
    )


def test_tiny_corpus_skips_split_instead_of_producing_broken_shard(tmp_path: Path, tok: BPETokenizer):
    """语料小到切不出一个窗口时，应整体作为训练集，而不是产出坏分片。"""
    corpus = tmp_path / "tiny"
    corpus.mkdir()
    (corpus / "a.py").write_text("x = 1\n", encoding="utf-8")
    cfg = DataConfig(
        source_dirs=[str(corpus)], out_dir=str(tmp_path / "d"), name="t", val_ratio=0.5,
        min_file_bytes=1,
    )
    shards = build_shards(
        collect_code_files(cfg, verbose=False), tok, cfg, verbose=False, block_size=512
    )
    assert shards.train_paths
    assert not shards.val_paths
    for p in shards.train_paths:
        assert p.exists() and p.stat().st_size > 0


def test_build_shards_without_files_raises(tmp_path: Path, tok: BPETokenizer):
    cfg = DataConfig(source_dirs=[str(tmp_path)], out_dir=str(tmp_path / "d"), name="x")
    with pytest.raises(RuntimeError):
        build_shards([], tok, cfg, verbose=False)


# ---------------------------------------------------------------------------- 采样器


def test_sampler_shapes(tmp_path: Path, tok: BPETokenizer):
    corpus = _make_corpus(tmp_path / "src", count=10)
    cfg = DataConfig(source_dirs=[str(corpus)], out_dir=str(tmp_path / "d"), name="s")
    shards = build_shards(collect_code_files(cfg, verbose=False), tok, cfg, verbose=False)
    sampler = RandomWindowSampler(shards.train_paths, block_size=32, batch_size=4, seed=0)
    x, y, mask = sampler.sample_batch()
    assert x.shape == (4, 32) and y.shape == (4, 32)
    assert mask is None
    assert torch_equal_shift(x, y)


def torch_equal_shift(x, y) -> bool:
    """y 必须是 x 右移一位（同一窗口的下一 token）。"""
    import torch  # noqa: F401

    return bool((x[:, 1:] == y[:, :-1]).all())


def test_sampler_is_reproducible(tmp_path: Path, tok: BPETokenizer):
    corpus = _make_corpus(tmp_path / "src", count=10)
    cfg = DataConfig(source_dirs=[str(corpus)], out_dir=str(tmp_path / "d"), name="r")
    shards = build_shards(collect_code_files(cfg, verbose=False), tok, cfg, verbose=False)
    a = RandomWindowSampler(shards.train_paths, 32, 4, seed=7)
    b = RandomWindowSampler(shards.train_paths, 32, 4, seed=7)
    xa, ya, _ = a.sample_batch()
    xb, yb, _ = b.sample_batch()
    import torch

    assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_sampler_rejects_tiny_shards(tmp_path: Path):
    small = tmp_path / "tiny.bin"
    np.asarray([1, 2, 3], dtype=np.uint16).tofile(small)
    with pytest.raises(ValueError):
        RandomWindowSampler([small], block_size=64, batch_size=1)


def test_sampler_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        RandomWindowSampler([tmp_path / "nope.bin"], 8, 1)


def test_sampler_with_fim_produces_mask(tmp_path: Path, tok: BPETokenizer):
    corpus = _make_corpus(tmp_path / "src", count=12)
    cfg = DataConfig(source_dirs=[str(corpus)], out_dir=str(tmp_path / "d"), name="f")
    shards = build_shards(collect_code_files(cfg, verbose=False), tok, cfg, verbose=False)
    sampler = RandomWindowSampler(
        shards.train_paths, 64, 8, seed=1, fim_rate=1.0, tokenizer=tok
    )
    x, y, mask = sampler.sample_batch()
    assert mask is not None
    assert mask.shape == (8, 64)
    assert mask.sum() > 0, "FIM 掩码不应全为 0"
    assert (mask > 0).any(), "至少要有部分位置参与损失"


def test_sampler_without_fim_has_no_mask(tmp_path: Path, tok: BPETokenizer):
    corpus = _make_corpus(tmp_path / "src", count=8)
    cfg = DataConfig(source_dirs=[str(corpus)], out_dir=str(tmp_path / "d"), name="nf")
    shards = build_shards(collect_code_files(cfg, verbose=False), tok, cfg, verbose=False)
    sampler = RandomWindowSampler(shards.train_paths, 32, 2, seed=0, fim_rate=0.0, tokenizer=tok)
    _, _, mask = sampler.sample_batch()
    assert mask is None


# ---------------------------------------------------------------------------- FIM v2
#
# 背景（都有实测证据，见 AGENTS.md 缺陷清单）：
#   v1 格式每次丢掉「中间段」尾部 3 个 token（fim_transform_ids 输出 n+3，
#   调用方只取前 T+1）；且中间段之后没有任何终止符，模型无法学会「填完收尾」，
#   实测主动收尾率只有 0~4%。
#   v2 格式（fim_terminator=True）两件事一起修，默认关闭以保证历史行为不变。


def test_fim_v1_is_unchanged_by_default(tok):
    """默认（v1）行为必须逐位不变：输出长度 = 输入 + 3，终止符不参与。"""
    ids = list(range(300, 500))
    out, mask = fim_transform_ids(ids, tok, random.Random(0))
    assert len(out) == len(ids) + 3
    assert len(mask) == len(out)
    assert tok.eot_id not in out
    # v1：计损恰好是 <|fim_end|> 本身 + 其后的整段中间内容，前面一个都不算
    fe = len(out) - 1 - out[::-1].index(tok.fim_end_id)
    assert mask[fe] == 1.0
    assert all(v == 0.0 for v in mask[:fe]), "中间段之前的 token 不应计损"
    assert all(v == 1.0 for v in mask[fe:]), "中间段应全部计损"
    assert len(out) - fe - 1 > 0, "中间段不应为空"


def test_fim_v2_appends_terminator_and_masks_it(tok):
    """v2：中间段之后有终止符，且终止符计入损失。"""
    ids = list(range(300, 500))
    out, mask = fim_transform_ids(ids, tok, random.Random(0), terminator_id=tok.eot_id)
    assert len(out) == len(ids) + 4          # 比 v1 多一个终止符
    assert out[-1] == tok.eot_id             # 终止符在最后
    assert mask[-1] == 1.0                   # 且计入损失
    assert len(mask) == len(out)


def test_fim_v2_output_fits_window_exactly(tok):
    """v2 的核心：给了 max_len 就正好填满，不再丢中间段尾部。"""
    T = 64
    window = np.arange(1000, 1000 + T + 1, dtype=np.int64)
    out, mask = apply_fim(
        window, None, tok, random.Random(1), 1.0, 0.5,
        terminator_id=tok.eot_id, max_len=T + 1,
    )
    assert len(out) == T + 1, "FIM 输出必须正好等于窗口长度，否则会被调用方截断"
    assert len(mask) == T + 1
    assert out[-1] == tok.eot_id

    # 对照：v1 的输出比窗口长 3，调用方按 [:T+1] 截断就会吃掉尾部
    out_v1, _ = apply_fim(window, None, tok, random.Random(1), 1.0, 0.5)
    assert len(out_v1) == T + 4


def test_sampler_v2_terminator_lands_in_targets(tmp_path, tok):
    """端到端：v2 采样出的 batch 里，最后一个 target 就是终止符且计损。"""
    shard = tmp_path / "s.bin"
    data = np.array([(i * 37) % 256 for i in range(4000)], dtype=np.uint16)
    data.tofile(shard)

    T = 64
    s = RandomWindowSampler(
        [shard], block_size=T, batch_size=4, dtype="uint16", seed=7,
        fim_rate=1.0, fim_spm_rate=0.5, tokenizer=tok, fim_terminator=True,
    )
    x, y, m = s.sample_batch()
    assert m is not None
    yn, mn = y.numpy(), m.numpy()
    for b in range(4):
        assert yn[b, T - 1] == tok.eot_id, "最后一个 target 应当是终止符"
        assert mn[b, T - 1] > 0, "终止符必须计入损失，否则学不会收尾"
    assert np.array_equal(yn[:, :-1], x.numpy()[:, 1:])


def test_sampler_v1_default_has_no_terminator(tmp_path, tok):
    """对照：默认采样器（v1）最后一位不是终止符。"""
    shard = tmp_path / "s.bin"
    data = np.array([(i * 37) % 256 for i in range(4000)], dtype=np.uint16)
    data.tofile(shard)
    T = 64
    s = RandomWindowSampler(
        [shard], block_size=T, batch_size=4, dtype="uint16", seed=7,
        fim_rate=1.0, fim_spm_rate=0.5, tokenizer=tok,
    )
    _x, y, m = s.sample_batch()
    assert m is not None
    assert int((y.numpy()[:, T - 1] == tok.eot_id).sum()) == 0


def test_fim_transform_structure(tok: BPETokenizer):
    ids = tok.encode(CODE)
    out, mask = fim_transform_ids(ids, tok, random.Random(0), spm_rate=0.0)
    assert len(out) == len(mask) == len(ids) + 3
    assert out[0] == tok.fim_begin_id
    assert tok.fim_hole_id in out
    assert tok.fim_end_id in out
    assert 0.0 in mask and 1.0 in mask
    # 三个特殊词元各出现一次，且顺序固定
    assert out.index(tok.fim_begin_id) < out.index(tok.fim_hole_id) < out.index(tok.fim_end_id)


def test_fim_psm_ordering(tok: BPETokenizer):
    """PSM：<begin> prefix <hole> suffix <end> middle，且三段拼回原序列。"""
    ids = list(range(100, 120))
    out, _ = fim_transform_ids(ids, tok, random.Random(1), spm_rate=0.0, max_middle_ratio=0.5)
    hole = out.index(tok.fim_hole_id)
    end = out.index(tok.fim_end_id)
    prefix = out[1:hole]
    suffix = out[hole + 1 : end]
    middle = out[end + 1 :]
    assert prefix + middle + suffix == ids, "prefix/middle/suffix 必须无损重排原序列"
    assert middle and prefix and suffix


def test_fim_mask_marks_middle_and_end_marker(tok: BPETokenizer):
    ids = list(range(50, 80))
    rng = random.Random(2)
    out, mask = fim_transform_ids(ids, tok, rng, spm_rate=0.0, max_middle_ratio=0.4)
    end = out.index(tok.fim_end_id)
    assert mask[end] == 1.0, "<fim_end> 位置应计损（学会收尾）"
    assert all(m == 1.0 for m in mask[end + 1 :]), "中间内容应全部计损"
    assert mask[0] == 0.0, "<fim_begin> 位置不应计损"


def test_fim_spm_ordering(tok: BPETokenizer):
    """SPM：<begin> suffix <hole> prefix <end> middle，同样必须无损重排。"""
    ids = list(range(200, 230))
    out, _ = fim_transform_ids(ids, tok, random.Random(3), spm_rate=1.0, max_middle_ratio=0.4)
    hole = out.index(tok.fim_hole_id)
    end = out.index(tok.fim_end_id)
    suffix = out[1:hole]
    prefix = out[hole + 1 : end]
    middle = out[end + 1 :]
    assert prefix + middle + suffix == ids
    assert middle and prefix and suffix


def test_fim_short_sequence_is_passthrough(tok: BPETokenizer):
    ids = [1, 2, 3]
    out, mask = fim_transform_ids(ids, tok, random.Random(0))
    assert out == ids
    assert mask == [1.0, 1.0, 1.0]


def test_fim_transform_preserves_length_and_has_context(tok: BPETokenizer):
    """回归测试：FIM 变换必须保持序列长度（线性层对形状很敏感），且前后缀都不能为空。"""
    ids = list(range(10, 90))
    for seed in range(12):
        out, mask = fim_transform_ids(
            ids, tok, random.Random(seed), spm_rate=seed / 12.0, max_middle_ratio=0.5
        )
        assert len(out) == len(ids) + 3, f"seed={seed} 长度发生了变化"
        assert len(mask) == len(out), f"seed={seed} 掩码长度不匹配"
        hole = out.index(tok.fim_hole_id)
        end = out.index(tok.fim_end_id)
        assert hole >= 2, "hole 之前必须同时有 <fim_begin> 和实质内容"
        assert end - hole >= 2, "hole 与 end 之间必须有实质内容"


def test_apply_fim_respects_rate(tok: BPETokenizer):
    arr = np.asarray(tok.encode(CODE), dtype=np.uint16)
    same, _ = apply_fim(arr, None, tok, random.Random(0), fim_rate=0.0, spm_rate=0.5)
    assert np.array_equal(same, arr)
    changed, mask = apply_fim(arr, None, tok, random.Random(0), fim_rate=1.0, spm_rate=0.5)
    assert mask is not None
    assert len(changed) == arr.size + 3


def test_iter_documents_splits_on_eot():
    arr = np.asarray([1, 2, 3, 99, 4, 5, 99, 6], dtype=np.uint16)
    docs = list(iter_documents(arr, eot_id=99))
    assert [d.tolist() for d in docs] == [[1, 2, 3], [4, 5], [6]]
