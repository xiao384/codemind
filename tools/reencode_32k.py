"""用新 tokenizer（vocab 32,768）重新编码全部语料。

## 为什么需要重新编码

旧分片是 uint16 存的 token id（vocab 4,096）。新 tokenizer 词表 32,768，
**同一段文本的 token 序列完全不同**，必须解码回文本再重新编码。

## 为什么仍用 uint16

vocab = 32,768，token id 范围 0~32,767，**远小于 uint16 上限 65,535** ✓
所以数据格式不变，`codemind/data.py` 不需要改。

## 并行策略

纯 Python 的 BPE 编码是 CPU 密集的，用多进程按分片并行。
实测 1.0M token 分片约需 X 秒（见输出），8 进程线性加速。

用法::

    python tools/reencode_32k.py --old-tok data/corpus4/tokenizer.json \\
        --new-tok data/tok32k/tokenizer.json \\
        --src data/corpus4 --out data/corpus32k --procs 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(r"E:\AI")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from codemind.tokenizer import BPETokenizer  # noqa: E402

# 工作进程里的全局（避免每个文件重新加载 tokenizer）
_OLD: BPETokenizer | None = None
_NEW: BPETokenizer | None = None
_OLD_PATH = ""
_NEW_PATH = ""


def _init(old_path: str, new_path: str) -> None:
    global _OLD, _NEW, _OLD_PATH, _NEW_PATH
    _OLD_PATH, _NEW_PATH = old_path, new_path
    _OLD = BPETokenizer.load(Path(old_path))
    _NEW = BPETokenizer.load(Path(new_path))


def _one(args: tuple[str, str]) -> tuple[str, int, int, float]:
    """按 EOT 切分文档 → 逐文档解码 → 重编码 → 补回 EOT → 写盘。

    ## 为什么必须按文档走，而不是整片 decode/encode

    第一版写的是::

        text = old.decode(arr.tolist(), strip_special=True)   # 去掉 EOT
        ids  = new.encode(text)                               # 新流无边界

    **这把文档边界全剥掉了**，后果有两个，第二个是致命的：

    1. `cb1.py` 靠 ``arr == EOT_ID`` 切文档（`load_documents`），
       剥掉 EOT 后它找到 0 个文档 → 直接 `StatisticsError: fmean requires
       at least one data point`（实测踩到）。
    2. **训练数据里再也没有 EOT** → 模型根本没机会学「什么时候该停」。
       而「主动收尾」正是我们要测的能力之一。

    正确做法：保留旧流里的 EOT 作为边界，**每个文档单独重编码后再补一个 EOT**。
    新旧 tokenizer 的 `eot_id` 都是 256（实测），所以边界可以直接沿用数值。
    """
    src, dst = args
    t0 = time.time()
    arr = np.fromfile(src, dtype=np.uint16)
    eot = _OLD.eot_id if _OLD is not None else 256

    # 按 EOT 切出文档区间
    bounds = np.flatnonzero(arr == eot)
    out: list[int] = []
    prev = 0
    for pos in bounds:
        if pos > prev:
            seg = arr[prev:pos].tolist()
            txt = _OLD.decode(seg, strip_special=True)
            if txt:
                out.extend(_NEW.encode(txt))
            out.append(eot)             # ← 关键：补回文档边界
        prev = int(pos) + 1
    if prev < arr.size:                 # 末尾没有 EOT 的残段
        txt = _OLD.decode(arr[prev:].tolist(), strip_special=True)
        if txt:
            out.extend(_NEW.encode(txt))
        out.append(eot)

    if out and max(out) > 65535:
        raise ValueError(f"token id 超出 uint16: max={max(out)}")
    np.asarray(out, dtype=np.uint16).tofile(dst)
    return Path(src).name, len(arr), len(out), time.time() - t0


def collect(d: Path) -> list[Path]:
    files = sorted(d.glob("*.bin"))
    if not files:
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                files += collect(sub)
    return files


def run(src_dir: Path, out_dir: Path, old_path: str, new_path: str,
        procs: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.bin"):
        f.unlink()

    files = collect(src_dir)
    print(f"\n[{src_dir.name}] {len(files)} 个分片 → {out_dir}")
    jobs = []
    for f in files:
        # 保持原名但加前缀，避免碰撞
        rel = f.relative_to(src_dir)
        name = str(rel).replace("\\", "__").replace("/", "__")
        jobs.append((str(f), str(out_dir / name)))

    t0 = time.time()
    old_total = new_total = 0
    done = 0
    with Pool(procs, initializer=_init,
              initargs=(old_path, new_path)) as pool:
        for name, no, nn, dt in pool.imap_unordered(_one, jobs):
            old_total += no
            new_total += nn
            done += 1
            if done % 20 == 0 or done == len(jobs):
                el = time.time() - t0
                eta = el / done * (len(jobs) - done)
                print(f"  {done}/{len(jobs)}  旧 {old_total/1e6:.0f}M → "
                      f"新 {new_total/1e6:.0f}M token  "
                      f"用时 {el/60:.1f} 分  ETA {eta/60:.1f} 分", flush=True)

    return {"files": len(files), "old_tokens": old_total,
            "new_tokens": new_total, "seconds": time.time() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-tok", default="data/corpus4/tokenizer.json")
    ap.add_argument("--new-tok", default="data/tok32k/tokenizer.json")
    ap.add_argument("--src", action="append", default=[])
    ap.add_argument("--out", action="append", default=[])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--meta", default="data/corpus32k.meta.json")
    args = ap.parse_args()

    srcs = args.src or ["data/corpus4", "data/chat_full"]
    outs = args.out or ["data/corpus32k", "data/chat32k"]
    if len(srcs) != len(outs):
        print("✗ --src 与 --out 数量不一致")
        return 2

    old_path = str(ROOT / args.old_tok)
    new_path = str(ROOT / args.new_tok)
    if not Path(new_path).is_file():
        print(f"✗ 新 tokenizer 不存在: {new_path}")
        return 2

    results = {}
    for s, o in zip(srcs, outs):
        r = run(ROOT / s, ROOT / o, old_path, new_path, args.procs)
        results[Path(o).name] = r
        print(f"  ✓ {o}: {r['old_tokens']/1e6:.0f}M → "
              f"**{r['new_tokens']/1e6:.0f}M** token  "
              f"（{r['new_tokens']/max(r['old_tokens'],1)*100:.1f}%）"
              f"  {r['seconds']/60:.1f} 分钟")

    meta = ROOT / args.meta
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({
        "new_tokenizer": args.new_tok,
        "vocab_size": 32768,
        "dtype": "uint16",
        "sources": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✓ meta → {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
