"""检查 firefly-train-1.1M 的**真实可用量**。

## 为什么必须查

本项目已经被同一类问题坑过两次：

* JS/TS：全机扫出 112.6 MB，严格过滤后只剩 **3.3 MB**（97% 是打包产物）
* 图片：3,310 个 / 126 MB，中位 96×96，**75.8% 是 UI 图标**，且有内容的没配文

**"文件大"和"可用"是两件事。** 所以下载完先量：记录数、字段、长度分布、
重复率、以及**真正能用于训练的比例**。
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

SRC = Path(r"E:\AI\corpora\data\firefly-train-1.1M.jsonl")
MAX_LINES = 200_000          # 抽样上限，避免全文解析太慢


def main() -> int:
    print(f"文件 {SRC.stat().st_size/1e6:.1f} MB")
    n = 0
    keys: Counter = Counter()
    kinds: Counter = Counter()
    inst_lens: list[int] = []
    out_lens: list[int] = []
    seen: set[str] = set()
    dup = 0
    empty = 0
    with SRC.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if n >= MAX_LINES:
                break
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                kinds["JSON 解析失败"] += 1
                continue
            keys.update(o.keys())
            kinds[str(o.get("kind", o.get("type", "?")))] += 1
            i = (o.get("input") or o.get("instruction") or "").strip()
            t = (o.get("output") or o.get("target") or "").strip()
            if not i or not t:
                empty += 1
                continue
            inst_lens.append(len(i))
            out_lens.append(len(t))
            h = hashlib.md5((i + "\x00" + t).encode("utf-8")).hexdigest()
            if h in seen:
                dup += 1
            else:
                seen.add(h)

    print(f"抽样 {n:,} 行")
    print(f"字段分布: {dict(keys.most_common(8))}")
    print(f"kind 分布: {dict(kinds.most_common(8))}")
    print(f"空/缺字段: {empty:,}")
    print(f"重复(按 输入+输出 去重): {dup:,}  ({dup/max(1,len(seen)+dup)*100:.1f}%)")
    print(f"可用样本: {len(inst_lens):,}")

    if inst_lens:
        def pct(a, p):
            a = sorted(a)
            return a[min(len(a) - 1, int(len(a) * p))]
        print(f"\n问长度  中位 {pct(inst_lens,0.5)}  90% {pct(inst_lens,0.9)}  99% {pct(inst_lens,0.99)}  最大 {max(inst_lens):,}")
        print(f"答长度  中位 {pct(out_lens,0.5)}  90% {pct(out_lens,0.9)}  99% {pct(out_lens,0.99)}  最大 {max(out_lens):,}")
        print(f"答总字符 {sum(out_lens):,}")

        # 按中位长度估算：全量 1.1M 条能产出多少 token
        full = 1_100_000
        est_chars = sum(out_lens) / len(out_lens) * full + sum(inst_lens) / len(inst_lens) * full
        print(f"\n【外推全量 {full:,} 条】")
        print(f"  估算字符 {est_chars/1e6:.1f} M  →  按 2.4 字符/token ≈ **{est_chars/2.4/1e6:.1f}M token**")
        print(f"  对比现有聊天语料 31.65M token")

        # 过滤后的可用量（去掉过长/过短）
        ok = [i for i, t in zip(inst_lens, out_lens) if 5 <= i <= 2000 and 5 <= t <= 4000]
        print(f"  长度过滤后抽样可用率 {len(ok)/len(inst_lens)*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
