"""把指令语料转成 CodeMind 能吃的对话分片。

## 为什么标记不能用 `#` 开头（这是聊天能力为零的**根因**）

现有 `Generator.chat()` 的提示词是 `## User` / `## Assistant`。实测：

* `## User` 以 `#` 开头 → **在 Python 里就是注释**，模型把整段对话读成注释块；
* 这两个标记在 6800 万 token 的代码语料里**出现 0 次**（实测），模型没有任何理由尊重它们。

所以本脚本改用 `[用户]` / `[助手]`：
**行首 `[` 不是注释、不是常见代码模式，而且会在 BPE 里成为稳定的 token 序列。**

## 格式

每条一个文档，用 `<|endoftext|>`(256) 分隔::

    [用户] 保持健康的三个提示。
    [助手] 1. 保持身体活动……

训练目标仍是**普通的下一个 token 预测**——不需要改模型，只需要换数据。
`[用户]` 之后的 token 也计入损失（不像 FIM 那样掩码），
因为我们要它学会**在有 `[助手]` 上下文时开始生成回答**。

用法::

    python tools/make_chat_corpus.py --csv corpora/data/train.csv --out data/chat
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EOT_ID = 256
#: ?? 从 codemind.generate 导入，**不在这里重复定义** ——
#: 训练语料与推理提示词的标记必须逐字一致，两处各写一份迟早会漂移。
from codemind.generate import BOT_TAG, USER_TAG  # noqa: E402


#: firefly-train-1.1M 的任务类型过滤。
#:
#: ⚠️ **必须过滤，不能全倒进去。** 实测该数据集**答案长度中位数只有 44 字符** ——
#: 因为里面大量是**判别式任务**（情感分析、文本匹配），答案就一两个词。
#: 把这类样本混进对话语料，只会教模型"回答要短"，
#: 而那正是当前聊天模型的问题之一。
#:
#: 保留**生成式**（答案长、需要组织语言），丢弃**判别式**（输出一个标签）。
DROP_KINDS = {
    "SentimentAnalyze", "TextMatching", "NLI", "KeywordsExtraction",
    "QuestionGeneration", "Rank", "Classify", "Score",
    # ?? MRC（机器阅读理解）大量是**选择题**（A/B/C/D 选项）。
    # 实测：网页端出现过模型输出「A. 我 / B. 我 / C. 我 / D. 我」——
    # 就是这类数据教出来的。**答案是一个字母的样本对会说话没有帮助。**
    "MRC",
}


def _clean(s: str) -> str:
    return " ".join((s or "").split())


def build_texts(path: Path, limit: int, max_chars: int,
                drop_kinds: bool = True) -> tuple[list[str], dict]:
    """同时支持两种格式：

    * **CSV**：`instruction, input, output`（alpaca-gpt4-data-zh）
    * **JSONL**：`kind, input, target`（firefly-train-1.1M）

    ⚠️ 写解析器之前**先读了两份文件的实际字段**，而不是假设格式 ——
    本项目在"假设格式"上栽过（手写 meta.json 漏字段、把 oracle_body 当函数源码）。
    """
    stats: dict = {"total": 0, "empty": 0, "dropped_kind": 0, "too_long": 0}
    texts: list[str] = []
    kinds: dict = {}

    def add(q: str, a: str) -> None:
        stats["total"] += 1
        q, a = _clean(q), _clean(a)
        if not q or not a:
            stats["empty"] += 1
            return
        if len(q) > max_chars or len(a) > max_chars * 4:
            stats["too_long"] += 1
            return
        texts.append(f"{USER_TAG}{q}\n{BOT_TAG}{a}")

    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = str(o.get("kind") or "")
                kinds[kind] = kinds.get(kind, 0) + 1
                if drop_kinds and kind in DROP_KINDS:
                    stats["dropped_kind"] += 1
                    continue
                add(o.get("input") or o.get("instruction") or "",
                    o.get("target") or o.get("output") or "")
                if limit and len(texts) >= limit:
                    break
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                q = row.get("instruction") or ""
                if (row.get("input") or "").strip():
                    q = f"{q}\n{row['input'].strip()}"
                add(q, row.get("output") or "")
                if limit and len(texts) >= limit:
                    break

    if kinds:
        stats["kinds_top"] = dict(sorted(kinds.items(), key=lambda kv: -kv[1])[:10])
    return texts, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="corpora/data/train.csv",
                    help="支持 .csv（alpaca）与 .jsonl（firefly）")
    ap.add_argument("--keep-all-kinds", action="store_true",
                    help="不过滤判别式任务（默认会丢掉 SentimentAnalyze 等短答案类型）")
    ap.add_argument("--out", default="data/chat")
    ap.add_argument("--name", default="chat")
    ap.add_argument("--tokenizer", default="data/corpus2/tokenizer.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--shard-tokens", type=int, default=8_000_000)
    ap.add_argument("--seed", type=int, default=20261006)
    args = ap.parse_args()

    from codemind.tokenizer import BPETokenizer
    tok_path = ROOT / args.tokenizer
    tokenizer = BPETokenizer.load(tok_path)
    print(f"[chat] 分词器 {tok_path}  vocab={tokenizer.vocab_size}")

    texts, bstats = build_texts(ROOT / args.csv, args.limit, args.max_chars,
                                drop_kinds=not args.keep_all_kinds)
    print(f"[chat] 解析: 总 {bstats['total']:,}  空 {bstats['empty']:,}  "
          f"按 kind 丢弃 {bstats['dropped_kind']:,}  过长 {bstats['too_long']:,}")
    if bstats.get('kinds_top'):
        print(f"[chat] kind 前几名: {bstats['kinds_top']}")
    print(f"[chat] 对话样本 {len(texts):,} 条")
    if not texts:
        print("[chat] 没有样本，检查 CSV")
        return 2

    rng = random.Random(args.seed)
    rng.shuffle(texts)
    n_val = max(1, int(len(texts) * args.val_ratio))
    val_texts, train_texts = texts[:n_val], texts[n_val:]
    print(f"[chat] 训练 {len(train_texts):,} / 验证 {len(val_texts):,}")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    def encode_all(items: list[str]) -> np.ndarray:
        ids: list[int] = []
        for t in items:
            ids.extend(tokenizer.encode(t, allow_special=False))
            ids.append(EOT_ID)          # 文档分隔，训练循环靠它切文档
        return np.asarray(ids, dtype=np.uint16)

    for split, items in (("train", train_texts), ("val", val_texts)):
        arr = encode_all(items)
        n_sh = max(1, (len(arr) + args.shard_tokens - 1) // args.shard_tokens)
        for i in range(n_sh):
            chunk = arr[i * args.shard_tokens : (i + 1) * args.shard_tokens]
            # ⚠️ 训练与验证**必须用不同前缀**。
            # 实测踩过：两者都叫 `{name}.0000N.bin` 时，验证分片会**先覆盖训练分片 0
            # 再改名走**，于是训练集凭空少一个分片（31.65M 变 23.65M，正好差 8M）。
            p = out / f"{args.name}.{split}{i:05d}.bin"
            chunk.tofile(p)
            print(f"  {split:5} {p.name}  {len(chunk):,} token")

    total = sum(p.stat().st_size // 2 for p in out.glob("*.bin"))
    # ⚠️ meta.json 的字段必须和 `prepare` 生成的**完全一致** —— 训练器按
    # `train_files` / `val_files` 两个**文件名列表**去找分片。
    # 实测踩过：我手写了 name/vocab_size/total_tokens，但漏了这两个字段和 dtype，
    # 训练器直接报「没有任何可用的分片文件」——**文件都在，是索引不对**。
    # 教训：手写一个已有格式的元数据之前，**先读一份现成的**对照。
    import time as _t
    meta = {
        "name": args.name,
        "dtype": "uint16",
        "train_files": sorted(p.name for p in out.glob(f"{args.name}.train*.bin")),
        "val_files": sorted(p.name for p in out.glob(f"{args.name}.val*.bin")),
        "total_tokens": total,
        "vocab_size": tokenizer.vocab_size,
        "n_docs": len(texts),
        "created_at": _t.time(),
        "sources": [str(args.csv)],
        "chat_format": {"user": USER_TAG, "assistant": BOT_TAG},
    }
    (out / f"{args.name}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "tokenizer.json").write_bytes(tok_path.read_bytes())
    print(f"[chat] 合计 {total:,} token -> {out}")
    print(f"[chat] train_files={len(meta['train_files'])}  val_files={len(meta['val_files'])}")
    print(f"[chat] 示例:\n  " + texts[0][:200].replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
