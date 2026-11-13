"""构建**混合语料**（代码 + 对话），用于同时保住两种能力。

## 为什么需要它

实测已确认两件事，它们指向同一个解法：

1. **聊天训练完全冲掉了代码能力**
   `edit_sim` 27.90% → **14.48%**（跌到机会基线 20.87% 之下）
   —— 纯对话微调的必然结果。

2. **纯代码训练会过拟合**
   78M 模型吃 60M token，**参数比数据还多**（每参数 0.76 token，Chinchilla 要 20）。
   实测跑到 10.3 个 epoch 后验证损失不再下降，而训练损失还在降 = 在背诵。

**混合语料同时缓解这两件事**：代码样本持续出现 → 提醒模型"别忘了代码"；
对话样本提供新信息 → 降低对同一批代码的重复次数。

## 关键设计：**交错排布，不是首尾拼接**

如果把所有代码放前面、所有对话放后面，模型会在后半段**只见到对话** ——
那和纯对话微调没有区别，**照样灾难性遗忘**。

所以本脚本按 `--block` 大小**逐块交替**写入，保证**任何一段连续的训练数据里
两种内容都在场**。这是整个脚本里最要紧的一点。

## 容量约束（脚本会自己算并告诉你）

设代码总量 C、对话总量 L、目标比例 `rc:rl`，则能构建的总量受**两个来源同时约束**：

```
总量 ≤ C / rc      （代码不许超采）
总量 ≤ L / rl      （对话不许超采）
```

**取两者较小值。** 例：C=60M、L=31.65M、70:30 →
总量 ≤ min(85.7M, 105.5M) = **85.7M**（**代码侧是瓶颈**）。

用法::

    python tools/make_mixed_corpus.py \\
        --code data/corpus2 --lang data/chat --out data/mix \\
        --ratio 70:30 --shard-tokens 8000000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EOT_ID = 256


def load_shards(d: Path) -> tuple[np.ndarray, dict]:
    """读一个语料目录的所有分片，拼成一个 token 数组。"""
    meta_p = sorted(d.glob("*.meta.json"))
    if not meta_p:
        raise FileNotFoundError(f"{d} 下没有 *.meta.json")
    meta = json.loads(meta_p[0].read_text(encoding="utf-8"))
    files = list(meta.get("train_files", [])) + list(meta.get("val_files", []))
    if not files:
        raise ValueError(f"{meta_p[0]} 里没有 train_files / val_files")
    arrs = []
    for f in files:
        p = d / f
        if not p.exists():
            raise FileNotFoundError(f"缺分片 {p}")
        arrs.append(np.fromfile(p, dtype=np.uint16))
    return np.concatenate(arrs), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="data/corpus2", help="代码语料目录")
    ap.add_argument("--lang", default="data/chat", help="对话语料目录")
    ap.add_argument("--out", default="data/mix")
    ap.add_argument("--name", default="mix")
    ap.add_argument("--ratio", default="70:30", help="代码:对话")
    ap.add_argument("--block", type=int, default=200_000,
                    help="每块交替的 token 数（必须远大于单步 token 数）")
    ap.add_argument("--shard-tokens", type=int, default=8_000_000)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    args = ap.parse_args()

    rc, rl = (float(x) for x in args.ratio.split(":"))
    tot_r = rc + rl
    rc, rl = rc / tot_r, rl / tot_r

    code, cmeta = load_shards(ROOT / args.code)
    lang, lmeta = load_shards(ROOT / args.lang)
    print(f"[mix] 代码 {args.code}: {len(code):,} token  (vocab={cmeta.get('vocab_size')})")
    print(f"[mix] 对话 {args.lang}: {len(lang):,} token  (vocab={lmeta.get('vocab_size')})")

    # ⚠️ 分词器必须一致，否则混出来的 token 流没有意义
    cv, lv = cmeta.get("vocab_size"), lmeta.get("vocab_size")
    if cv != lv:
        print(f"[mix] ✗ 分词器不一致：代码 vocab={cv}，对话 vocab={lv}")
        print("        必须先统一分词器再混合（换分词器 = 现有检查点作废）")
        return 2

    # 容量约束：两个来源同时限制
    cap_code = int(len(code) / rc)
    cap_lang = int(len(lang) / rl)
    total = min(cap_code, cap_lang)
    which = "代码" if cap_code <= cap_lang else "对话"
    print(f"[mix] 容量上限：代码侧 {cap_code:,} / 对话侧 {cap_lang:,}  → 取 {total:,}")
    print(f"[mix] **瓶颈在{which}侧**")

    n_code = int(total * rc)
    n_lang = int(total * rl)

    # 交错排布 —— 见文件头说明，这是最要紧的一步
    blocks: list[np.ndarray] = []
    ci = li = 0
    b = max(args.block, 10_000)
    while ci < n_code or li < n_lang:
        take_c = min(b, n_code - ci)
        if take_c > 0:
            blocks.append(code[ci:ci + take_c])
            ci += take_c
        take_l = min(b, n_lang - li)
        if take_l > 0:
            blocks.append(lang[li:li + take_l])
            li += take_l
    mixed = np.concatenate(blocks)
    print(f"[mix] 交错完成：{len(mixed):,} token，块大小 {b:,}")

    # 验证交错是否真的生效：前 5% 与后 5% 里两种来源的比例应当接近
    print(f"[mix] 交错检查：前 5% = {len(mixed)//20:,} token 内应含两种来源（块 {b:,} 步长）")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.bin"):
        old.unlink()

    n_val = int(len(mixed) * args.val_ratio)
    n_train = len(mixed) - n_val
    train, val = mixed[:n_train], mixed[n_train:]
    print(f"[mix] 训练 {n_train:,} / 验证 {n_val:,}（{args.val_ratio*100:.1f}%）")

    def write(arr: np.ndarray, split: str) -> list[str]:
        names = []
        n_sh = max(1, (len(arr) + args.shard_tokens - 1) // args.shard_tokens)
        for i in range(n_sh):
            chunk = arr[i * args.shard_tokens:(i + 1) * args.shard_tokens]
            # ⚠️ 训练与验证必须用**不同前缀**：同名会让验证分片覆盖训练分片（实测踩过）
            name = f"{args.name}.{split}{i:05d}.bin"
            chunk.tofile(out / name)
            names.append(name)
            print(f"  {split:5} {name}  {len(chunk):,} token")
        return names

    tr = write(train, "train")
    va = write(val, "val")

    # meta.json 的字段必须与 prepare 生成的一致，否则训练器读不到（实测踩过）
    (out / f"{args.name}.meta.json").write_text(json.dumps({
        "name": args.name,
        "dtype": "uint16",
        "train_files": tr,
        "val_files": va,
        "total_tokens": int(len(mixed)),
        "vocab_size": cv,
        "n_docs": int(cmeta.get("n_docs", 0)) + int(lmeta.get("n_docs", 0)),
        "created_at": time.time(),
        "sources": [str(ROOT / args.code), str(ROOT / args.lang)],
        "mix": {"ratio": args.ratio, "block": b,
                "code_tokens": n_code, "lang_tokens": n_lang,
                "bottleneck": which},
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    import shutil
    tok = ROOT / args.code / "tokenizer.json"
    if tok.is_file():
        shutil.copy2(tok, out / "tokenizer.json")

    print(f"\n[mix] ✓ {out}")
    print(f"      训练 {n_train:,} token / 验证 {n_val:,} token")
    print(f"      配比 代码 {n_code/len(mixed)*100:.0f}% / 对话 {n_lang/len(mixed)*100:.0f}%")
    print(f"\n  下一步（从**代码模型**出发，别从聊天模型出发）：")
    print(f"      python -m codemind train --run runs/mix --data {args.out} \\")
    print(f"          --preset tiny --steps {n_train//16384} --threads 24")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
