"""合并多个 token 分片目录 → 一个交错的语料。

## 为什么交错

各批来自不同的随机包集合，代码风格有差异。首尾拼接会让训练后半段只见一批的分布，
loss 出现人为台阶。按块交错能让每个训练窗口里各批都在场。

## 用法

    python tools/merge_dirs.py --dirs data/corpus4 data/pypi_tok3 --out data/corpus5 --name corpus5
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\AI")


def load_dir(d: Path) -> np.ndarray:
    metas = sorted(d.glob("*.meta.json"))
    if not metas:
        # 目录里可能有子目录（pypi_tok3/b000 ...），逐个收
        arrs = []
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                arrs.append(load_dir(sub))
        if not arrs:
            raise FileNotFoundError(f"{d} 下没有 *.meta.json 也没有子目录")
        return np.concatenate(arrs)
    m = json.loads(metas[0].read_text(encoding="utf-8"))
    fs = list(m.get("train_files", [])) + list(m.get("val_files", []))
    if not fs:
        raise ValueError(f"{metas[0]} 里没有分片清单")
    return np.concatenate([np.fromfile(d / f, dtype=np.uint16) for f in fs])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="corpus5")
    ap.add_argument("--block", type=int, default=2_000_000)
    ap.add_argument("--shard-tokens", type=int, default=8_000_000)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--tokenizer", default="data/corpus3/tokenizer.json")
    args = ap.parse_args()

    parts = []
    for rel in args.dirs:
        d = ROOT / rel
        a = load_dir(d)
        print(f"[merge] {rel:24} {len(a):>15,} token")
        parts.append(a)

    # 按块交错
    b = args.block
    mixed = []
    idx = [0] * len(parts)
    while any(idx[k] < len(parts[k]) for k in range(len(parts))):
        for k, a in enumerate(parts):
            if idx[k] < len(a):
                mixed.append(a[idx[k]:idx[k] + b])
                idx[k] += b
    arr = np.concatenate(mixed)
    print(f"[merge] 交错后 {len(arr):,} token")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.bin"):
        f.unlink()

    n_val = max(1, int(len(arr) * args.val_ratio))
    train, val = arr[:-n_val], arr[-n_val:]

    def write(a: np.ndarray, split: str) -> list[str]:
        names, S = [], args.shard_tokens
        for k in range(max(1, (len(a) + S - 1) // S)):
            nm = f"{args.name}.{split}{k:05d}.bin"
            a[k * S:(k + 1) * S].tofile(out / nm)
            names.append(nm)
        return names

    tr, va = write(train, "train"), write(val, "val")
    (out / f"{args.name}.meta.json").write_text(json.dumps({
        "name": args.name, "dtype": "uint16",
        "train_files": tr, "val_files": va,
        "total_tokens": int(len(arr)), "vocab_size": 4096,
        "n_docs": 0, "created_at": time.time(),
        "sources": args.dirs,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    tok = ROOT / args.tokenizer
    if tok.is_file():
        shutil.copy2(tok, out / "tokenizer.json")

    print(f"\n[merge] ✓ {out}")
    print(f"        训练 {len(train):,} / 验证 {len(val):,} token")
    print(f"        分片 {len(tr)} 训练 + {len(va)} 验证")
    n = 1e9
    print(f"\n  1B 模型 Chinchilla = {n*20/1e9:,.0f}B token")
    print(f"  corpus5 提供 = {len(train)/1e9:.2f}B → {len(train)/(n*20):.3f} × Chinchilla")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
