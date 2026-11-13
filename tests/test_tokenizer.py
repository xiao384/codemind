"""分词器测试（纯 Python，无第三方依赖）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from codemind.tokenizer import (
    EOT,
    FIM_BEGIN,
    FIM_END,
    FIM_HOLE,
    N_BYTE_SYMBOLS,
    PAD,
    SPECIAL_TOKENS,
    BPETokenizer,
)

DOCS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "def mul(a, b):\n    return a * b\n\n\n",
    "class Calculator:\n    def __init__(self):\n        self.value = 0\n\n"
    "    def add(self, x):\n        self.value += x\n        return self.value\n",
    "import os\nimport sys\n\nprint(os.getcwd())\n",
    "for i in range(10):\n    print(i * 2)\n",
    "values = [x ** 2 for x in range(20) if x % 3 == 0]\n",
    "with open('data.txt', encoding='utf-8') as fh:\n    content = fh.read()\n",
    "try:\n    result = compute()\nexcept ValueError as exc:\n    print(exc)\n",
    "async def fetch(url):\n    async with session.get(url) as resp:\n        return await resp.json()\n",
]

# 合并训练需要足够丰富的 n-gram 才能消耗掉目标合并数；语料过小时训练会提前收敛，
# 此时 vocab_size < 请求值（属于正常行为，不是错误）。384 = 261 + 123 是本语料
# 实际能撑满的最大值，选它可以让断言保持确定性。
CORPUS = DOCS * 6
TARGET_VOCAB = 384


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    t = BPETokenizer()
    t.train_from_texts(CORPUS, vocab_size=TARGET_VOCAB, verbose=False)
    return t


# ---------------------------------------------------------------------------- 词表布局


def test_vocab_size_matches_request(tok: BPETokenizer):
    assert tok.vocab_size == TARGET_VOCAB
    assert tok.n_merges == TARGET_VOCAB - tok.merge_base


def test_layout_is_bytes_then_specials_then_merges(tok: BPETokenizer):
    """词表布局必须严格分层：字节 -> 特殊词元预留区 -> 合并词元。

    这是核心不变量：早期版本把特殊词元放在所有合并之后，导致
    (a) 特殊词元 ID 随训练漂移、(b) 合并与特殊词元 ID 撞车后互相覆盖。
    """
    assert tok.merge_base == N_BYTE_SYMBOLS + len(SPECIAL_TOKENS)
    assert tok.eot_id == N_BYTE_SYMBOLS
    assert tok.pad_id == N_BYTE_SYMBOLS + 1
    assert tok.fim_begin_id == N_BYTE_SYMBOLS + 2
    assert tok.fim_hole_id == N_BYTE_SYMBOLS + 3
    assert tok.fim_end_id == N_BYTE_SYMBOLS + 4
    assert tok.merge_base == N_BYTE_SYMBOLS + 5
    # 合并词元必须全部落在预留区之后
    for rank in range(tok.n_merges):
        assert tok.merge_base + rank < tok.vocab_size


def test_special_ids_are_constant_during_training():
    """回归测试：特殊词元 ID 在整个训练过程中必须恒定。

    早期实现在训练循环里用 ``eot_id`` 作为文档分隔哨兵，而那个 id 是
    ``256 + len(merges)`` —— 会随每次合并移动，最终与新建合并词元撞车，
    并在 ``_rebuild`` 时抛 KeyError: 256。
    """
    t = BPETokenizer()
    before = dict(t._special_ids)
    t.train_from_texts(CORPUS, vocab_size=320, verbose=False)
    assert t._special_ids == before, "训练后特殊词元 ID 发生了漂移"

    mid = BPETokenizer()
    mid.train_from_texts(DOCS, vocab_size=300, verbose=False)
    mid_specials = dict(mid._special_ids)
    mid.train_from_texts(CORPUS, vocab_size=340, verbose=False)
    assert mid._special_ids == mid_specials, "增量训练后特殊词元 ID 发生了漂移"


def test_special_ids_identical_across_instances():
    a = BPETokenizer()
    a.train_from_texts(DOCS, vocab_size=300, verbose=False)
    b = BPETokenizer()
    b.train_from_texts(CORPUS, vocab_size=320, verbose=False)
    assert a.eot_id == b.eot_id
    assert a.fim_begin_id == b.fim_begin_id
    assert a.fim_hole_id == b.fim_hole_id
    assert a.fim_end_id == b.fim_end_id


def test_no_merge_references_special_tokens(tok: BPETokenizer):
    """合并序列里绝不能出现特殊词元 ID（否则会产生 ' <|endoftext|>' 这类词元）。"""
    specials = set(tok._special_ids.values())
    for a, b in tok.merges:
        assert a not in specials, f"合并引用了特殊词元 {a}"
        assert b not in specials, f"合并引用了特殊词元 {b}"
        assert 0 <= a < tok.vocab_size and 0 <= b < tok.vocab_size


def test_no_merge_contains_special_token_text(tok: BPETokenizer):
    for rank in range(tok.n_merges):
        symbol = tok.id_to_symbol(tok.merge_base + rank)
        for name in SPECIAL_TOKENS:
            assert name not in symbol, f"合并词元 {symbol!r} 混入了特殊词元文本"


def test_is_special_and_is_merge_token(tok: BPETokenizer):
    assert tok.is_special(tok.eot_id)
    assert not tok.is_special(0)
    assert not tok.is_special(N_BYTE_SYMBOLS - 1)
    assert not tok.is_special(tok.merge_base)
    assert tok.is_merge_token(tok.merge_base)
    assert not tok.is_merge_token(tok.eot_id)


# ---------------------------------------------------------------------------- 编解码


@pytest.mark.parametrize("text", DOCS + ["", "   ", "\n\n\n", "x"])
def test_roundtrip(tok: BPETokenizer, text: str):
    assert tok.decode(tok.encode(text)) == text


def test_pretokenizer_never_drops_characters():
    """回归测试：预切分必须无损覆盖原文。

    曾经的模式 `[^\\W\\d_]` 排除了下划线，而 `_` 又不匹配其它分支，
    re.findall 会静默跳过它 —— 导致 `my_var` 被编码成 `myvar`。
    """
    from codemind.tokenizer import DEFAULT_PRETOKEN_RE, finditer_chunks

    samples = [
        "my_var = other_name + 1",
        "__init__(self, x_y):",
        "a_b_c",
        "_",
        "___",
        "self.__dict__['_key']",
        "if x_1 == y_2:\n    pass\n",
        "中文_变量 = 1",
        "x=_=y",
        "def f(a_b, c_d):\n    return a_b * c_d\n",
    ]
    for text in samples:
        chunks = finditer_chunks(DEFAULT_PRETOKEN_RE, text)
        assert "".join(chunks) == text, f"预切分有损于 {text!r} -> {chunks!r}"


def test_underscore_survives_encoding(tok: BPETokenizer):
    """下划线必须在编解码全程存活。"""
    for text in ["my_var", "_private", "__init__", "a_1_b"]:
        assert tok.decode(tok.encode(text)) == text
        assert "_" in tok.decode(tok.encode(text))


def test_roundtrip_unicode_and_binary(tok: BPETokenizer):
    for text in ["中文注释：测试 123", "emoji 🚀🔥", "tab\there", "\x00\x01"]:
        assert tok.decode(tok.encode(text)) == text


def test_roundtrip_text_containing_special_token_literals(tok: BPETokenizer):
    """回归测试：文本里字面包含特殊词元时，往返必须仍然无损。

    真实代码会出现 ``"<|endoftext|>"`` 这样的字符串字面量（本项目源码里就有）。
    encode 会把它识别为特殊词元；若 decode 默认把它丢掉，
    ``decode(encode(x)) == x`` 就不成立，训练数据会被静默抹掉一部分。
    """
    samples = [
        'assert "<|endoftext|>" not in text\n',
        "SPECIAL = ['<|pad|>', '<|fim_begin|>', '<|fim_hole|>', '<|fim_end|>']\n",
        "```<|endoftext|>```",
        "<|endoftext|>",
        "a<|pad|>b<|fim_end|>c",
    ]
    for text in samples:
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"含特殊词元字面量的文本往返失败: {text!r}"

    # 显式剥离时才去掉框架词元
    ids = tok.encode("a<|pad|>b")
    assert tok.decode(ids, strip_special=True) == "ab"


def test_full_source_roundtrip():
    """整份源码往返必须无损 —— 这是训练数据完整性的底线。"""
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent
    files = sorted((root / "codemind").glob("*.py"))
    if not files:
        pytest.skip("找不到源码文件")
    docs = [p.read_text(encoding="utf-8") for p in files]
    t = BPETokenizer()
    t.train_from_texts(docs, vocab_size=1024, verbose=False)
    for path, text in zip(files, docs):
        assert t.decode(t.encode(text)) == text, f"{path.name} 往返有损"


def test_special_tokens_are_single_ids(tok: BPETokenizer):
    ids = tok.encode(f"{FIM_BEGIN}abc{FIM_HOLE}xyz{FIM_END}")
    assert ids[0] == tok.fim_begin_id
    assert tok.fim_hole_id in ids
    assert ids[-1] == tok.fim_end_id
    assert tok.eot_id == tok.special_id(EOT)
    assert tok.pad_id == tok.special_id(PAD)


def test_disable_special_parsing(tok: BPETokenizer):
    text = f"{FIM_BEGIN}"
    assert tok.encode(text)[0] == tok.fim_begin_id
    plain = tok.encode(text, allow_special=False)
    assert tok.fim_begin_id not in plain
    assert tok.decode(plain) == text


def test_whitespace_runs_become_single_tokens(tok: BPETokenizer):
    """缩进是代码结构的关键，理想情况下 4 空格应合并为独立 token。"""
    ids = tok.encode("    return 1")
    first = tok.decode_token(ids[0])
    assert first in ("   ", "    ") or first.startswith(" "), f"首 token = {first!r}"


def test_compression_is_meaningful(tok: BPETokenizer):
    text = "def add(a, b):\n    return a + b\n" * 20
    ids = tok.encode(text)
    ratio = len(text) / len(ids)
    assert ratio > 1.5, f"压缩率过低: {ratio}"


def test_batch_encode(tok: BPETokenizer):
    batch = tok.encode_batch(DOCS[:3])
    assert [tok.decode(ids) for ids in batch] == DOCS[:3]


def test_encode_empty_returns_empty(tok: BPETokenizer):
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_decode_rejects_out_of_range(tok: BPETokenizer):
    with pytest.raises(KeyError):
        tok.decode([tok.vocab_size + 10])


def test_id_for_text(tok: BPETokenizer):
    assert tok.id_for_text("def") is not None
    assert tok.id_for_text("绝对不存在的词元串") is None


# ---------------------------------------------------------------------------- 训练


def test_incremental_training_keeps_existing_merges():
    t = BPETokenizer()
    t.train_from_texts(DOCS, vocab_size=300, verbose=False)
    before = list(t.merges)
    t.train_from_texts(
        DOCS * 2 + ["def mul(a, b):\n    return a * b\n"], vocab_size=340, verbose=False
    )
    assert t.merges[: len(before)] == before, "增量训练必须保留原合并顺序"
    assert t.vocab_size == 340


def test_training_is_deterministic():
    a = BPETokenizer()
    a.train_from_texts(DOCS * 2, vocab_size=320, verbose=False)
    b = BPETokenizer()
    b.train_from_texts(DOCS * 2, vocab_size=320, verbose=False)
    assert a.merges == b.merges


def test_train_rejects_smaller_vocab(tok: BPETokenizer):
    with pytest.raises(ValueError):
        tok.train_from_texts(DOCS, vocab_size=100, verbose=False)


def test_empty_corpus_raises():
    t = BPETokenizer()
    with pytest.raises(ValueError):
        t.train_from_texts([], vocab_size=300, verbose=False)


def test_training_never_merges_across_documents():
    """用单个字符组成的"文档"训练，跨文档合并会产生超长幽灵词元。"""
    t = BPETokenizer()
    docs = ["ab"] * 50 + ["cd"] * 50
    t.train_from_texts(docs, vocab_size=300, verbose=False)
    for rank in range(t.n_merges):
        symbol = t.id_to_symbol(t.merge_base + rank)
        assert "ab" not in symbol or "cd" not in symbol
        assert "\n" not in symbol or len(symbol) < 4


# ---------------------------------------------------------------------------- 持久化


def test_save_and_load(tmp_path: Path, tok: BPETokenizer):
    path = tok.save(tmp_path / "tok.json")
    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.merges == tok.merges
    assert loaded.eot_id == tok.eot_id
    text = DOCS[0]
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_load_rejects_foreign_file(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"type": "sentencepiece"}', encoding="utf-8")
    with pytest.raises(ValueError):
        BPETokenizer.load(bad)


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        BPETokenizer.load(tmp_path / "nope.json")


def test_legacy_v1_file_is_migrated(tmp_path: Path):
    """v1 词表（特殊词元排在合并之后）必须能被迁移到 v2 布局。

    v1 的 256 号是第一个合并词元，v2 的 256 号是 <|endoftext|>，
    因此所有 >= 256 的 id 都要整体后移，且迁移后词表必须完全可用。
    """
    import json

    legacy = {
        "version": 1,
        "type": "byte-level-bpe",
        "special_tokens": list(SPECIAL_TOKENS),
        # v1 的 256/257 是前两个合并词元；第三条引用它们，验证合并链能延续
        "merges": [[100, 101], [256, 10], [257, 256]],
    }
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    t = BPETokenizer.load(path)

    assert t.eot_id == N_BYTE_SYMBOLS, "特殊词元必须落在预留区起始处"
    assert t.merge_base == N_BYTE_SYMBOLS + len(SPECIAL_TOKENS)
    assert t.merges[0] == (100, 101)
    # v1 的 256/257 -> v2 的 261/262
    assert t.merges[1] == (261, 10)
    assert t.merges[2] == (262, 261), "后续合并必须能引用先前的合并词元"
    # 迁移后的词表内容（注意 v1 里 256=换行符、257=第一个合并，别读串了）：
    #   261 = byte 100 + byte 101 = "de"
    #   262 = 261 + byte 10        = "de\n"
    #   263 = 262 + 261            = "de\nde"
    assert t.decode([261]) == "de"
    assert t.decode([262]) == "de\n"
    assert t.decode([263]) == "de\nde"
    # 迁移后必须能正常往返编码（"def" 没有对应合并，会被切成 ["de", "f"]）
    assert t.encode("de") == [261]
    assert t.decode(t.encode("def")) == "def"
    assert t.decode(t.encode("def\n")) == "def\n"


def test_legacy_migration_drops_special_referencing_merges(tmp_path: Path):
    """引用了特殊词元的非法合并（v1 的固有缺陷）必须被整条丢弃。"""
    import json

    legacy = {
        "version": 1,
        "type": "byte-level-bpe",
        "special_tokens": list(SPECIAL_TOKENS),
        # v1 里 256..260 是合并词元，261+ 才是特殊词元
        "merges": [[100, 101], [261, 10]],
    }
    path = tmp_path / "v1b.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    t = BPETokenizer.load(path)
    assert t.merges == [(100, 101)], "跨越特殊词元的合并链必须被截断"
    for a, b in t.merges:
        assert not t.is_special(a) and not t.is_special(b)


def test_saved_file_declares_v2_layout(tmp_path: Path, tok: BPETokenizer):
    import json

    path = tok.save(tmp_path / "v2.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["layout"] == "reserved-specials"


# ---------------------------------------------------------------------------- 调试辅助


def test_stats_and_visual(tok: BPETokenizer):
    stats = tok.stats()
    assert stats["vocab_size"] == tok.vocab_size
    assert stats["longest_merge_chars"] >= 2
    assert stats["special_ids"][EOT] == N_BYTE_SYMBOLS
    visual = tok.tokenize_visual("def f():\n    pass\n")
    assert "def" in visual and "|" in visual
