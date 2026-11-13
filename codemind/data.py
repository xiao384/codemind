"""数据流水线：代码语料采集 -> 二进制分片 -> memmap 随机切窗 -> FIM 变换。

三段式设计的原因：

1. **采集阶段**只做文本清洗与去重，输出到内存里的 token id 流。
2. **分片阶段**把 token 流写成 ``uint16`` 的 ``.bin``（4096 词表下 1 个 token 仅 2 字节），
   原始代码 1MB 大约压缩到 200-300KB，训练时直接 mmap，不占内存。
3. **采样阶段**用 NumPy 随机取窗口，绕过 ``DataLoader`` 的多进程开销 ——
   在 CPU 训练场景下，worker 的启动成本比取数本身还高。

FIM（Fill-In-The-Middle）是代码补全能力的来源：随机把文档切成
``前缀 / 中间 / 后缀``，重排后让模型学会「给定上下文填中间」，
这正是 IDE 里最常发生的事（光标在文件中间）。两种模式：

* **PSM**：``<fim_begin> 前缀 <fim_hole> 中间 <fim_end> 后缀``
* **SPM**：``<fim_begin> 后缀 <fim_hole> 前缀 <fim_end> 中间``（额外训练「后缀在前」的能力）

损失只在「中间」区间计算，前缀与后缀区间被 mask 掉。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import DataConfig
from .tokenizer import BPETokenizer

__all__ = [
    "TokenShards",
    "collect_code_files",
    "build_shards",
    "RandomWindowSampler",
    "apply_fim",
    "fim_transform_ids",
    "read_file_text",
]

README_NAMES = {"readme.md", "readme.rst", "readme.txt", "readme"}


# ======================================================================================
# 1) 语料采集
# ======================================================================================


def read_file_text(path: Path) -> Optional[str]:
    """尽最大努力把文件读成文本；失败返回 None（不抛异常）。"""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:  # 大概率是二进制文件
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover
            return None


def collect_code_files(cfg: DataConfig, verbose: bool = True) -> List[Path]:
    """遍历 ``source_dirs``，按后缀/体积/去重规则筛出可用代码文件。"""
    exts = set(cfg.resolved_extensions())
    exclude = set(cfg.exclude_dirs)
    files: List[Path] = []
    # 去重只保留 16 字节摘要，不保留原文。
    # 早期版本把 text 也存进 seen_hashes 的值里，处理 GB 级语料时会白白占住整份语料的内存。
    seen_digests: set = set()
    total_scanned = 0
    skipped_big = 0

    for source in cfg.source_dirs:
        root = Path(source).expanduser().resolve()
        if not root.exists():
            print(f"[data] 跳过不存在的目录: {root}")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in exclude and not d.startswith(".")]
            for fname in filenames:
                path = Path(dirpath) / fname
                total_scanned += 1
                if path.suffix.lower() not in exts and fname.lower() not in README_NAMES:
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size < cfg.min_file_bytes:
                    continue
                if size > cfg.max_file_bytes:
                    skipped_big += 1
                    continue

                if cfg.dedup or cfg.skip_markers:
                    # 只有确实需要检查内容时才读文件；读完即释放
                    text = read_file_text(path)
                    if text is None:
                        continue
                    if any(marker in text for marker in cfg.skip_markers):
                        del text
                        continue
                    if cfg.dedup:
                        digest = hashlib.blake2b(
                            text.encode("utf-8", "replace"), digest_size=16
                        ).digest()
                        del text
                        if digest in seen_digests:
                            continue
                        seen_digests.add(digest)

                files.append(path)
                if cfg.max_files and len(files) >= cfg.max_files:
                    break
            if cfg.max_files and len(files) >= cfg.max_files:
                break

    files.sort()
    if verbose:
        print(
            f"[data] 扫描 {total_scanned} 个文件 -> 命中 {len(files)} 个"
            f"（去重后；跳过超大文件 {skipped_big} 个）"
        )
    return files


# ======================================================================================
# 2) 分片
# ======================================================================================


@dataclass
class TokenShards:
    """分片元信息，负责解释 ``.bin`` 文件里的内容。"""

    name: str
    dir: Path
    train_files: List[str]
    val_files: List[str]
    dtype: str = "uint16"
    total_tokens: int = 0
    vocab_size: int = 0
    n_docs: int = 0
    created_at: float = 0.0
    sources: List[str] = field(default_factory=list)

    @property
    def train_paths(self) -> List[Path]:
        return [self.dir / f for f in self.train_files]

    @property
    def val_paths(self) -> List[Path]:
        return [self.dir / f for f in self.val_files]

    @property
    def meta_path(self) -> Path:
        return self.dir / f"{self.name}.meta.json"

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "train_files": self.train_files,
            "val_files": self.val_files,
            "total_tokens": self.total_tokens,
            "vocab_size": self.vocab_size,
            "n_docs": self.n_docs,
            "created_at": self.created_at,
            "sources": self.sources or [],
        }

    def save(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return self.meta_path

    @classmethod
    def load(cls, path: Path) -> "TokenShards":
        """``path`` 可以是 meta.json，也可以是分片目录。"""
        path = Path(path)
        if path.is_dir():
            candidates = sorted(path.glob("*.meta.json"))
            if not candidates:
                raise FileNotFoundError(f"{path} 下没有 *.meta.json")
            path = candidates[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            name=payload["name"],
            dir=path.parent,
            train_files=list(payload.get("train_files", [])),
            val_files=list(payload.get("val_files", [])),
            dtype=payload.get("dtype", "uint16"),
            total_tokens=int(payload.get("total_tokens", 0)),
            vocab_size=int(payload.get("vocab_size", 0)),
            n_docs=int(payload.get("n_docs", 0)),
            created_at=float(payload.get("created_at", 0.0)),
            sources=list(payload.get("sources", [])),
        )

    def tokens(self, split: str = "train") -> int:
        paths = self.train_paths if split == "train" else self.val_paths
        return sum(p.stat().st_size // np.dtype(self.dtype).itemsize for p in paths if p.exists())

    # ------------------------------------------------------------------ 清单
    @property
    def manifest_path(self) -> Path:
        return self.dir / f"{self.name}.files.json"

    def save_manifest(self, files: Sequence[Path]) -> Path:
        """记录参与训练的原始文件清单。

        为什么要落盘：GB 级语料必须分两步走（先用样本训分词器，再流式编码全量），
        第二步不该重新跑一遍目录扫描与内容去重 —— 那个过程要读一遍全部文件。
        """
        import json

        path = self.manifest_path
        payload = {
            "name": self.name,
            "count": len(files),
            "files": [str(Path(f).resolve()) for f in files],
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load_manifest(cls, path: Path) -> List[Path]:
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return [Path(p) for p in payload.get("files", [])]


def _dtype_for_vocab(vocab_size: int) -> str:
    if vocab_size <= 65535:
        return "uint16"
    return "uint32"


def build_shards(
    files: Sequence[Path],
    tokenizer: BPETokenizer,
    cfg: DataConfig,
    *,
    verbose: bool = True,
    shard_tokens: int = 20_000_000,
    block_size: int = 512,
) -> TokenShards:
    """把代码文件编码为 token 分片。

    每个文件之间插入 ``<|endoftext|>`` 作为文档分隔符，使模型学到「文档边界」概念，
    也为采样器提供了「不得跨越文档」的切窗依据。

    ``block_size`` 只用于判断「单分片能否切成训练/验证两半」：切完的每一半
    至少要能凑出一个完整窗口，否则训练会直接报错。它与模型的 block_size
    保持一致即可（CLI 会自动传入）。
    """
    out_dir = Path(cfg.out_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = _dtype_for_vocab(tokenizer.vocab_size)
    np_dtype = np.dtype(dtype)

    eot = tokenizer.eot_id
    order = list(range(len(files)))
    rng = random.Random(cfg.seed)
    rng.shuffle(order)
    n_val = int(len(order) * cfg.val_ratio)
    val_idx = set(order[:n_val])

    shard_buffers: List[List[int]] = []
    shard_meta: List[Dict[str, object]] = []
    # 用 array('I') 而不是 list 缓冲：Python list 里每个 int 对象占 28 字节，
    # 缓冲 2000 万 token 就是 560 MB 常驻内存；array 只占 4 字节/元素（80 MB）。
    buffer = array("I")
    n_docs = 0
    total_tokens = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        idx = len(shard_buffers)
        fname = f"{cfg.name}.{idx:05d}.bin"
        count = len(buffer)
        # array('I') -> uint16/uint32 二进制，避免中间再生成巨大的 Python 列表
        arr = np.frombuffer(buffer, dtype=np.uint32).astype(np_dtype, copy=False)
        arr.tofile(out_dir / fname)
        shard_meta.append({"file": fname, "tokens": count, "split": "pending"})
        shard_buffers.append([])
        buffer = array("I")

    for position, file_index in enumerate(order):
        text = read_file_text(files[file_index])
        if not text:
            continue
        if any(marker in text for marker in cfg.skip_markers):
            continue
        ids = tokenizer.encode(text)
        if not ids:
            continue
        buffer.extend(ids)
        buffer.append(eot)
        total_tokens += len(ids) + 1
        n_docs += 1
        if len(buffer) >= shard_tokens:
            flush()
        if verbose and n_docs % 200 == 0:
            print(f"[data] 已编码 {n_docs} 个文件 / {total_tokens/1e6:.2f}M tokens")

    flush()
    if not shard_meta:
        raise RuntimeError("没有编码出任何 token；请检查 source_dirs 与后缀白名单")

    # 按分片分配 split。
    # 注意：判断依据必须是 ``cfg.val_ratio``，**不能**用「有多少文件被分到验证集」——
    # 文件数很少时 n_val 恒为 0，那会让下面的「单分片 token 级切分」永远执行不到，
    # 结果是小语料永远拿不到验证集（曾是真实 bug）。
    for meta in shard_meta:
        meta["split"] = "pending"
    val_files: List[str] = []
    train_files: List[str] = []
    n_shards = len(shard_meta)
    # 切完的每一半至少要能凑出两个窗口，否则采样器无法工作
    min_split_tokens = max(64, 2 * int(block_size))

    if cfg.val_ratio <= 0:
        train_files = [str(m["file"]) for m in shard_meta]
    elif n_shards > 1:
        # 目标：验证集 token 数尽量贴近 val_ratio × 总量。
        # 旧实现是 ``min(n_shards-1, max(1, round(n_shards * val_ratio)))``：
        # 那个「至少 1 个整分片」的下限会盖过比例 —— 实测 4 个分片 + val_ratio=0.01
        # 也没拿到 1%，而是被切走整整 12.6% 的 token。
        n_val_whole = min(n_shards - 1, int(round(n_shards * cfg.val_ratio)))
        if n_val_whole:
            for meta in shard_meta[-n_val_whole:]:
                meta["split"] = "val"
                val_files.append(str(meta["file"]))
        for meta in shard_meta[: n_shards - n_val_whole]:
            meta["split"] = "train"
            train_files.append(str(meta["file"]))

        # 整分片粒度太粗时，对最后一个训练分片再做一次 token 级切分来补足差额
        acc = (
            sum(int(m.get("tokens", 0)) for m in shard_meta[-n_val_whole:])
            if n_val_whole
            else 0
        )
        leftover = int(round(total_tokens * cfg.val_ratio - acc))
        boundary = n_shards - n_val_whole - 1
        if leftover > 0 and boundary >= 0:
            meta = shard_meta[boundary]
            path = out_dir / str(meta["file"])
            arr = np.fromfile(path, dtype=np_dtype)
            cut = int(arr.size) - leftover
            if cut >= min_split_tokens and leftover >= min_split_tokens:
                train_name = f"{cfg.name}.p{boundary:02d}.train.bin"
                val_name = f"{cfg.name}.p{boundary:02d}.val.bin"
                arr[:cut].tofile(out_dir / train_name)
                arr[cut:].tofile(out_dir / val_name)
                path.unlink()
                train_files = [f for f in train_files if f != str(meta["file"])]
                train_files.append(train_name)
                val_files.append(val_name)
                meta["split"] = "split"
                if verbose:
                    print(
                        f"[data] 为凑够 {cfg.val_ratio:.2%} 的验证集，把 {meta['file']} "
                        f"按 token 切成 {cut}(训练) / {leftover}(验证)"
                    )

        if not val_files:
            # 兜底：切不动也必须留出验证集（否则训练只能拿训练损失自欺）。
            # 走到这里说明目标验证集小于切分下限 max(64, 2*block_size)，
            # 此时"验证集偏大"好过"没有验证集"。
            last = shard_meta[-1]
            last["split"] = "val"
            val_files.append(str(last["file"]))
            train_files = [f for f in train_files if f != str(last["file"])]
    else:
        # 单分片：在 token 级别切一刀，否则小语料永远没有验证集
        only = shard_meta[0]
        path = out_dir / str(only["file"])
        arr = np.fromfile(path, dtype=np_dtype)
        cut = int(arr.size * (1.0 - cfg.val_ratio))
        if arr.size < min_split_tokens or cut <= 0 or arr.size - cut <= 0:
            # 数据太少，切出来的两半都不够一个窗口：整体作为训练集
            train_files = [str(only["file"])]
        else:
            train_name = f"{cfg.name}.train.bin"
            val_name = f"{cfg.name}.val.bin"
            arr[:cut].tofile(out_dir / train_name)
            arr[cut:].tofile(out_dir / val_name)
            path.unlink()
            train_files = [train_name]
            val_files = [val_name]
            shard_meta = []

    shards = TokenShards(
        name=cfg.name,
        dir=out_dir,
        train_files=train_files,
        val_files=val_files,
        dtype=dtype,
        total_tokens=total_tokens,
        vocab_size=tokenizer.vocab_size,
        n_docs=n_docs,
        created_at=time.time(),
        sources=[str(Path(s).resolve()) for s in cfg.source_dirs],
    )
    shards.save()
    if verbose:
        print(
            f"[data] 完成：{total_tokens/1e6:.2f}M tokens / {n_docs} 文档 / "
            f"{len(train_files)} 训练分片 + {len(val_files)} 验证分片，"
            f"耗时 {time.time()-t0:.1f}s -> {out_dir}"
        )
    return shards


# ======================================================================================
# 3) FIM 变换
# ======================================================================================


def fim_transform_ids(
    ids: Sequence[int],
    tokenizer: BPETokenizer,
    rng: random.Random,
    *,
    spm_rate: float = 0.5,
    min_middle: int = 1,
    max_middle_ratio: float = 0.5,
    terminator_id: Optional[int] = None,
) -> Tuple[List[int], List[float]]:
    """把一段 token 序列变成 FIM 训练样本。

    返回 ``(tokens, loss_mask)``，``loss_mask`` 长度与 ``tokens`` 相同，
    1 表示该位置参与损失（即那段被挖空的「中间」内容）。

    ``terminator_id`` 不为 None 时，会在**中间段之后**再补一个终止符并纳入损失。
    这是 v2 格式的关键：没有它，模型即使学会了「该填多长」，推理时也**没有任何
    token 可以表达「我填完了」**，只能一直填到 ``max_new_tokens``。
    v1 格式（默认）把 ``<|fim_end|>`` 放在中间段**之前**，README 声称那能让模型
    「主动收尾」——与实现相反，实测主动收尾率 0~4%。
    """
    n = len(ids)
    if n < 5:
        return list(ids), [1.0] * n

    # 中间段长度：1 ~ min(n/2, ...)，保证前后缀都有内容
    max_middle = max(min_middle, min(int(n * max_middle_ratio), n - 2))
    middle_len = rng.randint(min_middle, max_middle)
    start = rng.randint(1, n - middle_len - 1)
    end = start + middle_len

    prefix = list(ids[:start])
    middle = list(ids[start:end])
    suffix = list(ids[end:])

    fb, fh, fe = tokenizer.fim_begin_id, tokenizer.fim_hole_id, tokenizer.fim_end_id
    if rng.random() < spm_rate:
        # SPM: <begin> suffix <hole> prefix <end> middle [terminator]
        tokens = [fb] + suffix + [fh] + prefix + [fe] + middle
    else:
        # PSM: <begin> prefix <hole> suffix <end> middle [terminator]
        tokens = [fb] + prefix + [fh] + suffix + [fe] + middle

    has_term = terminator_id is not None
    if has_term:
        tokens.append(int(terminator_id))

    # 掩码与 tokens 同长，1 表示该位置计入损失。
    # v1：只有「中间」内容 + 紧邻中间之前的 <|fim_end|> 参与损失。
    # v2：再包含中间段之后的终止符，让模型学会「填完就收尾」。
    # 注意：掩码会由调用方 shift 一位对齐 targets（位置 t 预测 t+1），
    # 最后一个位置没有 target，shift 后自然被丢弃。
    mask = [0.0] * len(tokens)
    middle_start = len(tokens) - len(middle) - (1 if has_term else 0)
    for i in range(middle_start, len(tokens)):
        mask[i] = 1.0
    if middle_start - 1 >= 0:
        mask[middle_start - 1] = 1.0
    return tokens, mask


def apply_fim(
    tokens: np.ndarray,
    mask: Optional[np.ndarray],
    tokenizer: BPETokenizer,
    rng: random.Random,
    fim_rate: float,
    spm_rate: float,
    *,
    terminator_id: Optional[int] = None,
    max_len: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """概率性地对窗口施加 FIM；``fim_rate=0`` 时原样返回。

    ``terminator_id`` 不为 None 时启用 v2 格式：中间段之后补终止符。
    此时若给了 ``max_len``，会先把输入缩短，使 **FIM 输出正好等于 ``max_len``** ——
    否则调用方按 ``[:max_len]`` 截断时会把中间段尾部吃掉（v1 固定丢 3 个 token）。
    """
    if fim_rate <= 0 or rng.random() >= fim_rate:
        return tokens, mask
    ids = tokens.tolist()
    if terminator_id is not None and max_len:
        # 输出长度 = 输入 + 3 个结构词元 + 1 个终止符；要正好等于 max_len
        keep = max(8, int(max_len) - 4)
        ids = ids[:keep]
    new_ids, new_mask = fim_transform_ids(
        ids,
        tokenizer,
        rng,
        spm_rate=spm_rate,
        terminator_id=terminator_id,
    )
    return (
        np.asarray(new_ids, dtype=tokens.dtype),
        np.asarray(new_mask, dtype=np.float32),
    )


# ======================================================================================
# 4) 采样器
# ======================================================================================


class RandomWindowSampler:
    """从分片文件里随机切定长窗口，可选在线 FIM。

    这是训练循环的唯一数据来源，刻意做成「零依赖 + 可复现」：
    同一个 seed 一定产出同一批窗口。
    """

    def __init__(
        self,
        paths: Sequence[Path],
        block_size: int,
        batch_size: int,
        *,
        dtype: str = "uint16",
        device: str = "cpu",
        seed: int = 0,
        fim_rate: float = 0.0,
        fim_spm_rate: float = 0.5,
        tokenizer: Optional[BPETokenizer] = None,
        pin_memory: bool = False,
        strict_epoch: bool = False,
        fim_terminator: bool = False,
    ) -> None:
        self.paths = [Path(p) for p in paths if Path(p).exists()]
        if not self.paths:
            raise FileNotFoundError("没有任何可用的分片文件")
        self.block_size = int(block_size)
        self.batch_size = int(batch_size)
        self.np_dtype = np.dtype(dtype)
        self.device = device
        self.fim_rate = float(fim_rate)
        self.fim_spm_rate = float(fim_spm_rate)
        self.tokenizer = tokenizer
        self.pin_memory = pin_memory
        self.strict_epoch = strict_epoch
        # v2 FIM 格式：中间段之后补终止符并纳入损失，且让 FIM 输出正好填满窗口。
        # 默认 False，保持与历史数据逐位一致。
        self.fim_terminator = bool(fim_terminator)
        self.rng = random.Random(seed)
        self._starts_cache: Optional[Tuple[int, np.ndarray]] = None
        self._mm: List[np.memmap] = []
        for p in self.paths:
            arr = np.memmap(p, dtype=self.np_dtype, mode="r")
            if arr.size <= self.block_size + 1:
                continue
            self._mm.append(arr)
        if not self._mm:
            raise ValueError(
                f"分片数据太少：每个分片需至少 {self.block_size + 2} 个 token，"
                "请准备更多语料或减小 block_size"
            )
        self.total_tokens = int(sum(a.size for a in self._mm))

    def __len__(self) -> int:
        return self.total_tokens

    def _window_starts(self, length: int) -> np.ndarray:
        """所有合法窗口起点。

        按「迄今见过的最长分片」建一次 ``arange``，短分片取前缀切片 ——
        ``arange`` 的前缀就是更短的 ``arange``，因此与逐长度各建一份逐位等价，
        但只占用一份内存。

        早先的实现只缓存单一长度，而各分片的 token 数几乎不会恰好相等
        （分片按 token 数切、文档边界会让末尾取整），于是随机抽分片时缓存
        每次都失效，**每个窗口都要重算一个 8B/token 的大数组**：在 20M token
        的分片上实测占掉训练步时间的 25%（2439ms -> 3ms）。
        """
        cached = self._starts_cache
        if cached is None or cached[0] < length:
            cached = (length, np.arange(0, length - self.block_size, dtype=np.int64))
            self._starts_cache = cached
        return cached[1][: length - self.block_size]

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """返回 ``(x, y, loss_mask)``，形状分别为 ``(B, T)``、``(B, T)``、``(B, T)`` 或 None。

        ``y[b, t] = x[b, t+1]``，因此训练时无需再做 shift。
        """
        B, T = self.batch_size, self.block_size
        xs = np.empty((B, T), dtype=np.int64)
        ys = np.empty((B, T), dtype=np.int64)
        masks: List[np.ndarray] = []
        use_mask = self.fim_rate > 0 and self.tokenizer is not None
        for b in range(B):
            arr = self._mm[self.rng.randrange(len(self._mm))]
            starts = self._window_starts(arr.size)
            start = int(starts[self.rng.randrange(starts.size)])
            window = np.asarray(arr[start : start + T + 1], dtype=np.int64)
            mask: Optional[np.ndarray] = None
            if use_mask:
                assert self.tokenizer is not None
                window, mask = apply_fim(
                    window,
                    None,
                    self.tokenizer,
                    self.rng,
                    self.fim_rate,
                    self.fim_spm_rate,
                    terminator_id=(
                        self.tokenizer.eot_id if self.fim_terminator else None
                    ),
                    max_len=T + 1 if self.fim_terminator else None,
                )
                window = np.asarray(window, dtype=np.int64)
                if mask is not None:
                    mask = np.asarray(mask, dtype=np.float32)
            if window.size < T + 1:
                pad = T + 1 - window.size
                window = np.pad(window, (0, pad), constant_values=0)
                if mask is not None:
                    mask = np.pad(mask, (0, pad), constant_values=0.0)
            xs[b] = window[:T]
            ys[b] = window[1 : T + 1]
            if use_mask:
                if mask is None:
                    masks.append(np.ones(T, dtype=np.float32))
                else:
                    # mask 对齐 targets：位置 t 的 target 是否计损
                    aligned = mask[: T + 1]
                    masks.append(aligned[1 : T + 1].astype(np.float32))

        x = torch.from_numpy(xs)
        y = torch.from_numpy(ys)
        m = torch.from_numpy(np.stack(masks)) if use_mask else None
        if self.device != "cpu":
            x = x.pin_memory() if self.pin_memory else x
            y = y.pin_memory() if self.pin_memory else y
            if m is not None:
                m = m.pin_memory() if self.pin_memory else m
        return x, y, m

    def load_all(self, path: Path, max_tokens: int = 0) -> np.ndarray:
        """把某个分片整体读入内存（小数据集上更快）。"""
        arr = np.fromfile(Path(path), dtype=self.np_dtype)
        if max_tokens and arr.size > max_tokens:
            arr = arr[:max_tokens]
        return arr

    def close(self) -> None:
        self._mm = []


def is_document_start(ids: np.ndarray, index: int, eot_id: int) -> bool:
    return index == 0 or int(ids[index - 1]) == eot_id


def iter_documents(arr: np.ndarray, eot_id: int) -> Iterator[np.ndarray]:
    """按 ``<|endoftext|>`` 切分 token 流为独立文档。"""
    start = 0
    for i, tid in enumerate(arr):
        if int(tid) == eot_id:
            if i > start:
                yield arr[start:i]
            start = i + 1
    if start < arr.size:
        yield arr[start:]
