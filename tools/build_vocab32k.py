"""重建 tokenizer：vocab 4,096 → **32,768**，让中文别被切碎。

## 为什么必须换

2026-11-08 实测（同一段文本分别用两套分词器编码）：

| 样本 | 我们(4096) | Qwen(151,936) | 倍数 |
|---|---|---|---|
| 中文短句 | 54 | 10 | **5.40×** |
| 中文段落 | 102 | 20 | **5.10×** |
| Python 代码 | 39 | 21 | 1.86× |
| 英文长文 | 135 | 51 | 2.65× |

ctx 512 下，中文实际只能装 **约 95 个汉字** —— 一轮问答就爆。这是聊天能力为零的
直接原因（不是训练问题）。

## 训练策略

**不扫全量语料** —— `train_from_texts` 的 docstring 记着"220KB 训练 4096 词表要 118 秒"，
按这个速度 100MB 要十几小时。实际上 **BPE 只看高频模式，50~200 MB 样本足够**。

按 5000 字为一个采样块，从代码和对话里各取一部分，保证：
- 中文对话能学到中文子词
- 代码能学到常见 token（`def `、`self`、`return`、缩进等）

用法::

    python tools/build_vocab32k.py --sample-mb 60 --vocab 32768
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\AI")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from codemind.tokenizer import BPETokenizer  # noqa: E402


def sample_texts(corpus_dir: Path, old_tok: BPETokenizer, mb: float,
                 block: int = 5000, seed: int = 20261108) -> list[str]:
    """从语料分片里按块采样，解码成文本。"""
    files = sorted(corpus_dir.glob("*.bin"))
    if not files:
        print(f"  ⚠️ {corpus_dir} 没有 .bin")
        return []
    want_chars = int(mb * 1024 * 1024)
    got = 0
    out: list[str] = []
    rng = random.Random(seed)
    rng.shuffle(files)

    for f in files:
        if got >= want_chars:
            break
        try:
            arr = np.fromfile(f, dtype=np.uint16)
        except Exception:
            continue
        if len(arr) < block * 2:
            continue
        # 每片随机取 3 段
        for _ in range(3):
            if got >= want_chars:
                break
            i = rng.randrange(0, len(arr) - block)
            txt = old_tok.decode(arr[i:i + block].tolist(), strip_special=True)
            if txt:
                out.append(txt)
                got += len(txt)
    print(f"  {corpus_dir.name}: 取 {len(out)} 块 / {got/1e6:.1f} M 字符")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--sample-mb", type=float, default=40,
                    help="代码侧采样量（MB 字符）；对话侧取一半")
    ap.add_argument("--out", default="data/tok32k/tokenizer.json")
    args = ap.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 旧分词器只用来把 uint16 分片解码回文本
    old_tok = BPETokenizer.load(ROOT / "data/corpus4/tokenizer.json")
    print(f"旧分词器 vocab = {old_tok.vocab_size}")

    t0 = time.time()
    texts = sample_texts(ROOT / "data/corpus4", old_tok, args.sample_mb)
    texts += sample_texts(ROOT / "data/chat_full", old_tok, args.sample_mb * 0.5)
    if not texts:
        print("✗ 没采到任何文本")
        return 2
    total_chars = sum(len(t) for t in texts)
    print(f"合计 {len(texts)} 块 / {total_chars/1e6:.1f} M 字符  "
          f"（采样 {time.time()-t0:.0f}s）")

    print(f"\n开始训练 BPE vocab={args.vocab} …")
    t1 = time.time()
    tok = BPETokenizer()
    tok.train_from_texts(iter(texts), args.vocab, verbose=True)
    tok.save(out_path)
    print(f"训练完成 {time.time()-t1:.0f}s  → {out_path}")
    print(f"实际 vocab = {tok.vocab_size}")

    # 立刻对比效果
    samples = {
        "中文短句": "你好，你是谁？我是一个人工智能助手。",
        "中文段落": "保持健康的三个提示：第一，均衡饮食；第二，规律运动；第三，充足睡眠。",
        "Python代码": 'def add(a, b):\n    """Return the sum of a and b."""\n    return a + b',
        "英文长文": "The quick brown fox jumps over the lazy dog. " * 5,
    }
    print(f"\n{'样本':10} {'旧(4096)':>9} {'新(32k)':>9} {'倍数':>7}")
    print("-" * 40)
    for k, v in samples.items():
        a = len(old_tok.encode(v))
        b = len(tok.encode(v))
        print(f"{k:10} {a:>9} {b:>9} {a/b:>6.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
