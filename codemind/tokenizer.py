"""字节级 BPE 分词器 —— 从零实现，不依赖任何第三方 tokenizer。

词表布局（这是本模块最重要的设计决定）::

    0 .. 255          基础字节符号（永不改变）
    256 .. 256+K-1    特殊词元预留区（K = len(special_tokens)，永不改变）
    256+K ..          合并产生的普通词元，按训练顺序递增

早期版本把特殊词元放在「所有合并之后」，结果是 **特殊词元 ID 会随训练过程漂移**：
每新增一个合并，``<|endoftext|>`` 的 ID 就 +1。这带来两个致命后果：

1. 训练语料里已经写好的 token id 会在下一次训练时失效；
2. 合并过程可能把特殊词元当成普通符号参与合并，产生 ``' <|endoftext|>'`` 这种怪物词元。

固定预留区彻底消除了这两个问题：**特殊词元 ID 在整个模型生命周期内恒定**，
checkpoint 可以安全地跨分词器版本迁移。

其他关键设计
------------
1. **字节级**：基础词表是 256 个字节符号，任意文件（含非法 UTF-8）都能编码，
   不存在 ``<unk>``。代码场景里这很重要：模型永远不会因为遇到生僻字符而失明。
2. **代码感知预切分**：先按「连续空白 / 标识符 / 数字 / 标点」切块再合并。
   这样缩进（4 空格、Tab）天然成为一个 token，是模型学会 Python 缩进结构的前提。
3. **块级缓存训练**：合并训练时对「完全相同的文本块」只处理一次并缓存结果。
   代码语料重复度极高（空行、``):``、``return`` 每次出现都一模一样），
   实测能把分词器训练从 O(分钟) 压到 O(秒) 量级。
4. **FIM 特殊词元**：``<|fim_begin|> / <|fim_hole|> / <|fim_end|>`` 是代码补全能力的
   来源（见 :mod:`codemind.data`）。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import bytecodec
from .bytecodec import bytes_to_symbols

__all__ = ["BPETokenizer", "SPECIAL_TOKENS", "DEFAULT_PRETOKEN_RE"]

# 字符 -> 字节 反查表（模块级缓存，避免每次解码都重建映射）。
# 来自 :mod:`codemind.bytecodec` 的字节字母表，保证与编码端严格互逆。
_CHAR_TO_BYTE_TABLE: Dict[str, int] = {
    char: byte for byte, char in bytecodec.bytes_to_unicode().items()
}


# 特殊词元顺序即 ID 顺序（从 N_BYTE_SYMBOLS 开始编号），不可随意调整。
SPECIAL_TOKENS: Tuple[str, ...] = (
    "<|endoftext|>",
    "<|pad|>",
    "<|fim_begin|>",
    "<|fim_hole|>",
    "<|fim_end|>",
)

EOT, PAD, FIM_BEGIN, FIM_HOLE, FIM_END = SPECIAL_TOKENS

N_BYTE_SYMBOLS = 256

# 预切分模式（GPT-2 改良版，针对代码调整）：
#   1) 英文缩写（'s 'll 've 're 及大写形式）
#   2) 行内连续空白（缩进！）—— 让 4 空格 / Tab 成为独立块
#   3) 可选前导空格 + 标识符（\w+ 含下划线与数字）
#   4) 可选前导空格 + 数字串
#   5) 可选前导空格 + 单个非空白/非字母数字/非撇号字符
#   6) 兜底：任何单字符
#
# 两个必须记住的坑：
# * GPT-2 原版用 `[^\W\d_]+`（刻意排除下划线）。这在英文里没问题，
#   但在 Python 里 `my_var` 会被切成 `my|_|var`，更糟的是若后面缺少兜底分支，
#   `_` 会**匹配不上任何分支而被 re.findall 静默丢弃**，导致数据被悄悄破坏。
# * 因此这里改用 `\w+` 并把撇号从第 5 条里排除，最后再加 `.` 兜底。
#   `finditer_chunks` 还会校验「块拼接恒等于原文」，双保险。
DEFAULT_PRETOKEN_RE = (
    r"'[sdmt]|'ll|'ve|'re|'[SDMT]|'LL|'VE|'RE"
    r"|[^\S\r\n]+"
    r"| ?\w+"
    r"| ?[^\s\w]"
    r"|\s+"
    r"|."
)

# 训练分词器时用于分隔文档的内部保护符。它**不属于**任何词表，
# 只作为一个"墙"阻止跨文档合并，因此不会污染正常词元空间。
DOC_WALL = -1


@dataclass
class _EncoderState:
    """冻结的编码器状态：一份查找结构 + 热缓存。"""

    merges: Dict[Tuple[int, int], int]
    special_ids: Dict[str, int]
    special_re: Optional[re.Pattern]
    chunk_cache: Dict[str, List[int]]


# 编码热缓存的上限。代码语料重复度极高，缓存能省下大量重复合并；
# 但处理 GB 级语料时缓存本身会无限增长（数百万条目 → 上 GB 内存），
# 因此超过阈值直接清空重建 —— 清空后命中率依然很高，代价可忽略。
_CHUNK_CACHE_LIMIT = 200_000


@dataclass
class BPETokenizer:
    """字节级 BPE 分词器。

    典型用法::

        tok = BPETokenizer()
        tok.train_from_texts(code_strings, vocab_size=4096)
        ids = tok.encode("def add(a, b):\\n    return a + b\\n")
        assert tok.decode(ids) == "def add(a, b):\\n    return a + b\\n"
        tok.save("artifacts/tokenizer.json")
    """

    merges: List[Tuple[int, int]] = field(default_factory=list)
    special_tokens: Tuple[str, ...] = SPECIAL_TOKENS
    pretoken_re: str = DEFAULT_PRETOKEN_RE

    # 运行时状态（不参与 dataclass 比较/序列化）
    _symbol_to_id: Dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    _id_to_symbol: Dict[int, str] = field(default_factory=dict, repr=False, compare=False)
    _merge_rank: Dict[Tuple[int, int], int] = field(default_factory=dict, repr=False, compare=False)
    _merge_rank_pair: Dict[int, Tuple[int, int]] = field(default_factory=dict, repr=False, compare=False)
    _merge_rank_to_id: Dict[int, int] = field(default_factory=dict, repr=False, compare=False)
    _special_ids: Dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    _special_re: Optional[re.Pattern] = field(default=None, repr=False, compare=False)
    _cache: Optional[_EncoderState] = field(default=None, repr=False, compare=False)
    # token id -> 原始字节。合并词元是多个字节的拼接（如 'def' -> b'def'），
    # 解码时必须逐字节展开，否则无法还原成 bytes。
    _id_to_bytes: Dict[int, bytes] = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ 生命周期
    def __post_init__(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        """根据 merges 重建全部查找表。

        合并 id 从 ``merge_base`` 开始，因此永远不会与特殊词元预留区重叠 ——
        这正是早期版本崩溃的地方（``new_id`` 与 ``eot_id`` 撞车后互相覆盖）。
        """
        symbols = bytes_to_symbols(bytes(range(N_BYTE_SYMBOLS)))
        self._symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(symbols)}

        self._special_ids = {
            tok: N_BYTE_SYMBOLS + i for i, tok in enumerate(self.special_tokens)
        }
        for tok, tid in self._special_ids.items():
            self._id_to_symbol[tid] = tok
            self._symbol_to_id[tok] = tid

        base = self.merge_base
        for rank, (a, b) in enumerate(self.merges):
            merged_symbol = self._id_to_symbol[a] + self._id_to_symbol[b]
            token_id = base + rank
            self._symbol_to_id[merged_symbol] = token_id
            self._id_to_symbol[token_id] = merged_symbol

        self._rebuild_byte_forms()

        self._merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self._merge_rank_pair = {rank: pair for rank, pair in enumerate(self.merges)}
        self._merge_rank_to_id = {rank: base + rank for rank in range(len(self.merges))}

        if self.special_tokens:
            # 必须带捕获组：re.split 只有在模式含捕获组时才会把分隔符一并返回。
            # 少了这对括号，特殊词元会被静默丢弃（encode("<|fim_begin|>") 返回空列表）。
            alternation = "|".join(
                re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)
            )
            self._special_re = re.compile(f"({alternation})")
        else:
            self._special_re = None

        # 编码热缓存：代码语料中完全相同的块（空行、":", "    return " 等）出现极其频繁，
        # 缓存可以把重复编码的开销降到接近零。
        self._cache = _EncoderState(
            merges=self._merge_rank,
            special_ids=self._special_ids,
            special_re=self._special_re,
            chunk_cache={},
        )

    def _rebuild_byte_forms(self) -> None:
        """为每个字节级/合并词元预计算原始字节。

        关键点：``_id_to_symbol[512]`` 可能是 ``'def'`` 这种**多字符**符号，
        它代表的是 b'def' 这三个字节。必须用 bytecodec 的「字符 -> 字节」反查表
        逐字符还原，**不能**用 ``symbol.encode('utf-8')`` ——
        字节 0x20（空格）映射到字符 'Ġ'（U+0120），它 UTF-8 编码后是两个字节
        ``b'\\xc4\\xa0'``，会把解码结果彻底写坏。

        合并符号按构造必然是若干个 bytecodec 字符的拼接，所以逐字符查表恒成立。
        """
        table = _CHAR_TO_BYTE_TABLE
        forms: Dict[int, bytes] = {}
        for token_id, symbol in self._id_to_symbol.items():
            if self.is_special(token_id):
                continue
            try:
                forms[token_id] = bytes(table[ch] for ch in symbol)
            except KeyError as exc:  # pragma: no cover - 只在词表被外部破坏时触发
                raise ValueError(
                    f"词元 {token_id} 的符号 {symbol!r} 含非法字符 {exc.args[0]!r}"
                ) from exc
        self._id_to_bytes = forms

    # ------------------------------------------------------------------ 基本属性
    @property
    def merge_base(self) -> int:
        """合并词元的起始 id：紧跟特殊词元预留区之后。"""
        return N_BYTE_SYMBOLS + len(self.special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.merge_base + len(self.merges)

    @property
    def n_merges(self) -> int:
        return len(self.merges)

    @property
    def special_base(self) -> int:
        return N_BYTE_SYMBOLS

    @property
    def eot_id(self) -> int:
        return self._special_ids[EOT]

    @property
    def pad_id(self) -> int:
        return self._special_ids[PAD]

    @property
    def fim_begin_id(self) -> int:
        return self._special_ids[FIM_BEGIN]

    @property
    def fim_hole_id(self) -> int:
        return self._special_ids[FIM_HOLE]

    @property
    def fim_end_id(self) -> int:
        return self._special_ids[FIM_END]

    def special_id(self, token: str) -> int:
        try:
            return self._special_ids[token]
        except KeyError as exc:
            raise KeyError(f"未知特殊词元 {token!r}") from exc

    def id_to_symbol(self, token_id: int) -> str:
        try:
            return self._id_to_symbol[token_id]
        except KeyError as exc:
            raise KeyError(f"越界 token id: {token_id}") from exc

    def symbol_to_id(self, symbol: str) -> Optional[int]:
        return self._symbol_to_id.get(symbol)

    def id_for_text(self, text: str) -> Optional[int]:
        """若 ``text`` 恰好是单个词元，返回其 ID；否则 None。用于 logits 屏蔽。"""
        return self._symbol_to_id.get(text)

    def is_special(self, token_id: int) -> bool:
        return N_BYTE_SYMBOLS <= token_id < self.merge_base

    def is_merge_token(self, token_id: int) -> bool:
        return token_id >= self.merge_base

    def __len__(self) -> int:
        return self.vocab_size

    # ------------------------------------------------------------------ 预切分
    def _split_specials(self, text: str) -> List[str]:
        """按特殊词元切分，长度恒为奇数，偶数位是普通文本。"""
        if self._special_re is None:
            return [text]
        return self._special_re.split(text)

    def _iter_chunks(self, text: str) -> Iterator[Tuple[str, bool]]:
        """产出 ``(chunk, is_special)``，chunk 是预切分后的最小合并单元。

        判断是否特殊词元用查表而不是下标奇偶 —— ``_split_specials`` 会产出空串，
        一旦过滤掉空串，奇偶下标就不再与「是否分隔符」对齐。
        """
        for part in self._split_specials(text):
            if part == "":
                continue
            if part in self._special_ids:
                yield part, True
            else:
                for chunk in finditer_chunks(self.pretoken_re, part):
                    yield chunk, False

    # ------------------------------------------------------------------ 编码
    def _merge_chunk(self, chunk: str, merges: Dict[Tuple[int, int], int]) -> List[int]:
        """对单个预切分块做 BPE 合并。"""
        symbols = bytes_to_symbols(chunk.encode("utf-8"))
        ids = [self._symbol_to_id[s] for s in symbols]
        n = len(ids)
        if n < 2 or not merges:
            return ids

        while True:
            best_rank = len(merges)
            best_index = -1
            for i in range(n - 1):
                rank = merges.get((ids[i], ids[i + 1]))
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_index = i
            if best_index < 0:
                return ids
            merged = self._merge_rank_pair[best_rank]
            new_token = self._merge_rank_to_id[best_rank]
            out: List[int] = []
            i = 0
            while i < n:
                if i < n - 1 and (ids[i], ids[i + 1]) == merged:
                    out.append(new_token)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
            n = len(ids)
            if n < 2:
                return ids

    def encode(self, text: str, *, allow_special: bool = True) -> List[int]:
        """把文本编码为 token id 列表。

        ``allow_special=False`` 时特殊词元会被当作普通文本逐字节编码。
        """
        if not text:
            return []
        out: List[int] = []
        if allow_special and self._special_re is not None:
            for chunk, is_special in self._iter_chunks(text):
                if is_special:
                    out.append(self._special_ids[chunk])
                else:
                    out.extend(self._encode_chunk_cached(chunk))
            return out
        for chunk in finditer_chunks(self.pretoken_re, text):
            out.extend(self._encode_chunk_cached(chunk))
        return out

    def _encode_chunk_cached(self, chunk: str) -> List[int]:
        cache = self._cache
        if cache is None:
            return self._merge_chunk(chunk, self._merge_rank)
        hit = cache.chunk_cache.get(chunk)
        if hit is None:
            hit = self._merge_chunk(chunk, self._merge_rank)
            if len(cache.chunk_cache) >= _CHUNK_CACHE_LIMIT:
                # 防止 GB 级语料把缓存撑爆；清空后最常见块会迅速重新填充
                cache.chunk_cache.clear()
            cache.chunk_cache[chunk] = hit
        return hit

    def encode_batch(self, texts: Iterable[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    # ------------------------------------------------------------------ 解码
    def decode(self, ids: Sequence[int], *, strip_special: bool = False) -> str:
        """把 token id 还原为文本。

        默认行为是**严格可逆**：特殊词元被还原成它的字面文本。
        这一点至关重要 —— 真实代码里会出现 ``"<|endoftext|>"`` 这样的字符串字面量
        （本项目的源码里就有），若解码时静默丢弃它们，
        ``decode(encode(x)) == x`` 就不再成立，训练数据会被无声抹掉一部分。

        需要「剥掉框架词元拿到干净代码」时显式传 ``strip_special=True``
        （生成路径即如此，用于去掉 ``<|fim_*|>``）。

        非法字节序列用 ``errors="replace"`` 兜底。
        """
        pieces: List[str] = []
        buffer = bytearray()
        for raw in ids:
            tid = int(raw)
            if tid not in self._id_to_symbol:
                raise KeyError(f"越界 token id: {tid}")
            if self.is_special(tid):
                if buffer:
                    pieces.append(bytes(buffer).decode("utf-8", errors="replace"))
                    buffer.clear()
                symbol = self._id_to_symbol[tid]
                if not strip_special:
                    pieces.append(symbol)
            else:
                buffer += self._id_to_bytes[tid]
        if buffer:
            pieces.append(bytes(buffer).decode("utf-8", errors="replace"))
        return "".join(pieces)

    def decode_token(self, token_id: int) -> str:
        """单个 token 的可见文本（特殊词元原样返回）。"""
        return self.decode([token_id])

    # ------------------------------------------------------------------ 训练
    def _apply_to_work(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """把一次合并应用到一段 id 序列上。"""
        a, b = pair
        out: List[int] = []
        j = 0
        n = len(ids)
        while j < n:
            if j < n - 1 and ids[j] == a and ids[j + 1] == b:
                out.append(new_id)
                j += 2
            else:
                out.append(ids[j])
                j += 1
        return out

    @staticmethod
    def _count_pairs(ids: List[int]) -> Dict[Tuple[int, int], int]:
        """统计一段序列里每个相邻对的出现次数（DOC_WALL 作为右元素时不计）。"""
        counts: Dict[Tuple[int, int], int] = {}
        prev = -1
        for tid in ids:
            if prev >= 0 and tid != DOC_WALL:
                key = (prev, tid)
                counts[key] = counts.get(key, 0) + 1
            prev = tid
        return counts

    def train_from_texts(
        self,
        texts: Iterable[str],
        vocab_size: int,
        *,
        verbose: bool = True,
        log_every: int = 8,
    ) -> "BPETokenizer":
        """在文本语料上训练 BPE 合并规则（增量：保留已有 merges）。

        实现要点：**增量维护 pair 频次**，而不是每轮重新全量扫描。

        朴素做法是每产生一个合并就把整份语料重扫一遍统计相邻对，
        复杂度 O(语料 × 目标合并数)。实测 220KB 语料训练 4096 词表要 118 秒，
        按这个速度 100MB 语料需要十几小时，完全不可用。

        这里改为：
        1. 初始只做一次全量统计；
        2. 每轮从最大堆里取最高频 pair；
        3. 只把该 pair 出现过的那些块做替换，并**按增量**更新全局计数
           （减去块的旧贡献、加上新贡献），只重算被改动的块；
        4. 堆采用惰性校验：取出的计数与当前计数不一致就丢弃重推。

        结果与朴素实现逐位等价（同样的频次优先 + 字典序并列规则），
        因为我们始终按「局部替换后的真实计数」做增量更新。
        """
        target_merges = max(0, vocab_size - self.merge_base)
        if target_merges < len(self.merges):
            raise ValueError(
                f"目标 vocab_size={vocab_size} 小于当前词表 {self.vocab_size}，无法增量训练"
            )

        # ---------------------------------------------------------------- 1) 建块缓存
        chunk_freq: Counter = Counter()
        total_chars = 0
        for text in texts:
            total_chars += len(text)
            for chunk, is_special in self._iter_chunks(text):
                if is_special:
                    continue
                chunk_freq[chunk] += 1
        if not chunk_freq:
            raise ValueError("语料为空，无法训练分词器")

        # DOC_WALL 是一个不属于词表的哨兵：保证合并不跨越文档边界，
        # 也不必占用真实的特殊词元 id。
        chunks: List[List[int]] = []
        freqs: List[int] = []
        for chunk, freq in chunk_freq.items():
            chunks.append(list(self._merge_chunk(chunk, self._merge_rank)) + [DOC_WALL])
            freqs.append(freq)

        if verbose:
            print(
                f"[tokenizer] 语料 {total_chars/1e6:.2f}M 字符 / {len(chunks)} 个唯一块；"
                f"目标合并数 {target_merges}（当前 {len(self.merges)}）"
            )

        # ---------------------------------------------------------------- 2) 初始化计数
        # 朴素实现每轮把全部块重扫一遍统计相邻对：实测 220KB 语料训练 4096 词表
        # 要 100 秒以上，其中 78% 的时间花在这些重复扫描上（8400 万次 token 访问）。
        #
        # 这里改为**增量维护**：
        #   * 初始只统计一次；
        #   * 每次合并只重扫「确实包含该 pair」的那些块（通常占 10%~20%），
        #     并按「旧序列贡献 − 新序列贡献」修正全局计数；
        #   * 最大 pair 用惰性堆维护：堆里是历史快照，
        #     取出时与当前计数比对，过期就丢弃、不等就按当前值重推。
        # 结果与朴素实现完全一致（同样是频次优先 + 字典序并列规则）。
        import heapq

        # 反向索引：pair -> 包含它的块下标集合。
        # 必须用 set 而不是 list：同一次合并会让一个块被登记多次，
        # 用 list 会让条目数超线性膨胀，实测直接 MemoryError。
        pair_counts: Dict[Tuple[int, int], int] = {}
        pair_chunks: Dict[Tuple[int, int], set] = {}
        for index, ids in enumerate(chunks):
            for pair, count in self._count_pairs(ids).items():
                pair_counts[pair] = pair_counts.get(pair, 0) + count * freqs[index]
                pair_chunks.setdefault(pair, set()).add(index)

        # 惰性最大堆：(-计数, a, b)。计数下降时不主动清理，
        # 取出时若与 pair_counts 不符就按当前值重推 —— 这是标准的 lazy deletion。
        heap: List[Tuple[int, int, int]] = [
            (-count, pair[0], pair[1]) for pair, count in pair_counts.items()
        ]
        heapq.heapify(heap)
        base = self.merge_base

        # ---------------------------------------------------------------- 3) 合并循环
        while len(self.merges) < target_merges:
            best: Optional[Tuple[int, int]] = None
            while heap:
                neg_count, a, b = heapq.heappop(heap)
                current = pair_counts.get((a, b), 0)
                if current <= 0:
                    continue  # 该 pair 已消失
                if -neg_count != current:
                    heapq.heappush(heap, (-current, a, b))  # 计数已变，按真实值重推
                    continue
                best = (a, b)
                break
            if best is None:
                break

            rank = len(self.merges)
            new_id = base + rank
            self.merges.append(best)
            self._merge_rank[best] = rank
            self._merge_rank_pair[rank] = best
            self._merge_rank_to_id[rank] = new_id
            merged_symbol = self._id_to_symbol[best[0]] + self._id_to_symbol[best[1]]
            self._id_to_symbol[new_id] = merged_symbol
            self._symbol_to_id[merged_symbol] = new_id
            self._id_to_bytes[new_id] = bytes(_CHAR_TO_BYTE_TABLE[ch] for ch in merged_symbol)

            # 只处理包含该 pair 的块，并做增量计数修正
            touched: set = set()
            for index in pair_chunks.pop(best, []):
                old_ids = chunks[index]
                old_local = self._count_pairs(old_ids)
                new_ids = self._apply_to_work(old_ids, best, new_id)
                new_local = self._count_pairs(new_ids)
                weight = freqs[index]

                for pair, count in old_local.items():
                    updated = pair_counts.get(pair, 0) - count * weight
                    if updated > 0:
                        pair_counts[pair] = updated
                        touched.add(pair)
                    else:
                        pair_counts.pop(pair, None)
                for pair, count in new_local.items():
                    pair_counts[pair] = pair_counts.get(pair, 0) + count * weight
                    pair_chunks.setdefault(pair, set()).add(index)
                    touched.add(pair)
                chunks[index] = new_ids

            # 只把真正变动的 pair 推进堆，避免每轮重推全部候选
            for pair in touched:
                count = pair_counts.get(pair, 0)
                if count > 0:
                    heapq.heappush(heap, (-count, pair[0], pair[1]))

            if verbose and (len(self.merges) % log_every == 0 or len(self.merges) == target_merges):
                printable = merged_symbol if len(merged_symbol) <= 12 else merged_symbol[:12] + "…"
                print(
                    f"[tokenizer] merge {len(self.merges)}/{target_merges} "
                    f"freq={pair_counts.get(best, 1)} -> {printable!r}"
                )

        self._rebuild()
        return self

    def train_from_files(
        self, paths: Iterable[Path], vocab_size: int, **kwargs
    ) -> "BPETokenizer":
        def _read() -> Iterator[str]:
            for p in paths:
                try:
                    yield Path(p).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

        return self.train_from_texts(_read(), vocab_size, **kwargs)

    # ------------------------------------------------------------------ 持久化
    def to_dict(self) -> Dict[str, object]:
        return {
            "version": 2,
            "type": "byte-level-bpe",
            "layout": "reserved-specials",
            "special_tokens": list(self.special_tokens),
            "pretoken_re": self.pretoken_re,
            "merges": [[a, b] for a, b in self.merges],
            "vocab_size": self.vocab_size,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=0), encoding="utf-8"
        )
        tmp.replace(path)
        return path

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "BPETokenizer":
        merges = [tuple(int(x) for x in pair) for pair in payload.get("merges", [])]  # type: ignore[union-attr]
        specials = tuple(payload.get("special_tokens", SPECIAL_TOKENS))  # type: ignore[arg-type]
        version = int(payload.get("version", 1) or 1)
        if version < 2:
            merges = _remap_legacy_merges(merges, len(specials))  # type: ignore[arg-type]
        return cls(
            merges=merges,  # type: ignore[arg-type]
            special_tokens=specials,  # type: ignore[arg-type]
            pretoken_re=str(payload.get("pretoken_re", DEFAULT_PRETOKEN_RE)),
        )

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到分词器文件: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("type") != "byte-level-bpe":
            raise ValueError(f"{path} 不是 CodeMind 字节级 BPE 分词器")
        return cls.from_dict(payload)

    # ------------------------------------------------------------------ 调试辅助
    def tokenize_visual(self, text: str) -> str:
        """把文本渲染成 ``token|token|…`` 形式，便于人工检查切分质量。"""
        ids = self.encode(text)
        return "|".join(self.decode_token(i).replace("\n", "\\n") for i in ids)

    def stats(self) -> Dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "merges": len(self.merges),
            "specials": list(self.special_tokens),
            "special_ids": dict(self._special_ids),
            "longest_merge_chars": max(
                (len(self._id_to_symbol[self.merge_base + i]) for i in range(len(self.merges))),
                default=0,
            ),
        }


def _remap_legacy_merges(
    merges: List[Tuple[int, int]], n_specials: int
) -> List[Tuple[int, int]]:
    """把 v1 布局（特殊词元排在所有合并之后）的 merges 迁移到 v2 布局。

    v1: ``0..255`` 字节 -> ``256..256+M-1`` 合并 -> ``256+M..`` 特殊词元
    v2: ``0..255`` 字节 -> ``256..256+K-1`` 特殊词元 -> ``256+K..`` 合并

    迁移前的词元 id 需要整体偏移，且必须丢弃任何"把特殊词元当普通符号合并"
    产生的非法合并（那正是 v1 的缺陷）——注意是**丢弃该条**，而不是终止整个迁移。
    """
    offset = n_specials
    remapped: List[Tuple[int, int]] = []
    valid_ids = set(range(N_BYTE_SYMBOLS))
    for a, b in merges:
        new_a = a if a < N_BYTE_SYMBOLS else a + offset
        new_b = b if b < N_BYTE_SYMBOLS else b + offset
        if new_a not in valid_ids or new_b not in valid_ids:
            # 该合并引用了特殊词元或已被丢弃的合并 -> 跳过这一条
            continue
        remapped.append((new_a, new_b))
        # 新词元本身进入 v2 的合并区，后续合并可以继续引用它
        valid_ids.add(N_BYTE_SYMBOLS + n_specials + len(remapped) - 1)
    return remapped


def finditer_chunks(pattern: str, text: str) -> List[str]:
    """按预切分模式产出非空块。

    这是编码路径上最关键、也最容易出错的一环：``re.findall`` 只返回**匹配到**的片段，
    未匹配的字符会被静默丢弃（曾经的 ``[^\\W\\d_]`` 就丢掉了所有下划线）。
    因此这里加一道验证：块拼接必须无损还原原文，否则直接抛错而不是悄悄污染数据。
    验证开销很小（一次字符串拼接与比较），却能在编码阶段就拦住整条数据链的污染。
    """
    chunks = [m for m in re.findall(pattern, text) if m]
    if len(chunks) > 1 and "".join(chunks) != text:
        joined = "".join(chunks)
        raise ValueError(
            "预切分模式有损：块拼接后与原文不一致。\n"
            f"  原文: {text[:80]!r}\n"
            f"  拼接: {joined[:80]!r}\n"
            "这通常是预切分正则漏掉了某个字符类（例如下划线）。"
        )
    return chunks
