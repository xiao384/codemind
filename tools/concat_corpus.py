"""合并语料：**只复制分片 + 写 meta**，不加载到内存。

## 为什么可以简单拼接

训练用 `RandomWindowSampler`，它用 `rng.randrange()` **随机选分片、随机选起点**
（见 `codemind/data.py:630-632`）。**所以分片在磁盘上的顺序完全不影响训练。**

而各分片大小基本一致（8M token），随机选分片 ≈ 按 token 数比例采样。

## 为什么不能像之前那样交错

15B token × 2 字节 = 30 GB。本机 32 GB 内存，
`np.concatenate` 直接报 `Unable to allocate 23.9 GiB`（实测）。
**流式复制分片不占内存。**
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(r"E:\AI")


def collect(d: Path) -> tuple[list[Path], int]:
    """收集目录下所有训练/验证分片，支持一层子目录（pypi_tok3/b000 ...）。"""
    metas = sorted(d.glob("*.meta.json"))
    if metas:
        m = json.loads(metas[0].read_text(encoding="utf-8"))
        tr = [d / f for f in m.get("train_files", [])]
        va = [d / f for f in m.get("val_files", [])]
        return tr, va
    tr, va = [], []
    for sub in sorted(d.iterdir()):
        if sub.is_dir():
            t, v = collect(sub)
            tr += t
            va += v
    return tr, va


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="corpus5")
    ap.add_argument("--tokenizer", default="data/corpus3/tokenizer.json")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.bin"):
        f.unlink()

    tr_names, va_names = [], []
    total = 0
    t0 = time.time()
    for si, rel in enumerate(args.dirs):
        d = ROOT / rel
        tr, va = collect(d)
        print(f"[concat] {rel:22} 训练 {len(tr):>5} 分片 / 验证 {len(va):>3} 分片")
        for kind, files, names in (("train", tr, tr_names), ("val", va, va_names)):
            for k, f in enumerate(files):
                nm = f"{args.name}.{kind}{len(names):05d}.bin"
                shutil.copy2(f, out / nm)
                names.append(nm)
                total += f.stat().st_size
        print(f"[concat]   累计 {total/2/1e9:.2f}B token  "
              f"({(time.time()-t0)/60:.1f} 分钟)", flush=True)

    (out / f"{args.name}.meta.json").write_text(json.dumps({
        "name": args.name, "dtype": "uint16",
        "train_files": tr_names, "val_files": va_names,
        "total_tokens": total // 2, "vocab_size": 4096,
        "n_docs": 0, "created_at": time.time(),
        "sources": args.dirs,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    tok = ROOT / args.tokenizer
    if tok.is_file():
        shutil.copy2(tok, out / "tokenizer.json")

    print(f"\n[concat] ✓ {out}")
    print(f"         训练 {len(tr_names):,} 分片 / 验证 {len(va_names)} 分片")
    print(f"         **合计 {total/2/1e9:.2f}B token**")
    print(f"         1B Chinchilla 比例 = {total/2/20e9:.3f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
