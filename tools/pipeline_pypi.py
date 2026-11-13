"""解包 + 多进程编码，串成一条（避免手动等）。

## 为什么要串

解包 16,000+ 个 sdist 要 40 分钟左右，编码只要十几分钟。
分两步手动跑容易忘记接第二步，所以写成一个脚本一路做完。

## ⚠️ 解包为什么慢

sdist 是小文件密集（平均每包 76 个文件）。**Windows 上小文件创建开销高**，
所以解包比下载还慢 —— 这是文件系统特性，不是代码问题。
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(r"E:\AI")
PY = r"E:\AI\.venv\Scripts\python.exe"


def extract(src: Path, dst: Path) -> tuple[int, int, float]:
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = bad = 0
    files = sorted(src.glob("*.tar.gz"))
    for i, f in enumerate(files, 1):
        try:
            with tarfile.open(f, "r:gz") as tf:
                tf.extractall(dst, filter="data")
            n += 1
        except Exception:
            bad += 1
        if i % 2000 == 0:
            print(f"    解包 {i:,}/{len(files):,}  成功 {n:,}  失败 {bad}  "
                  f"{(time.time()-t0)/60:.1f} 分钟", flush=True)
    return n, bad, time.time() - t0


def main() -> int:
    src = ROOT / sys.argv[1] if len(sys.argv) > 1 else ROOT / "corpora/pypi_bulk2"
    dst = ROOT / sys.argv[2] if len(sys.argv) > 2 else ROOT / "corpora/pypi_bulk2_src"
    out = sys.argv[3] if len(sys.argv) > 3 else "data/pypi_tok2"

    print(f"[pipeline] 解包 {src.name} → {dst.name}")
    n, bad, el = extract(src, dst)
    cnt = sum(1 for p in dst.rglob("*") if p.is_file())
    sz = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(f"[pipeline] ✓ 解包 {n:,} 成功 / {bad} 失败 / {el/60:.1f} 分钟")
    print(f"[pipeline]   {cnt:,} 个文件 / {sz/1e6:.0f} MB")

    print(f"[pipeline] 多进程编码 → {out}")
    r = subprocess.run([PY, "-u", str(ROOT / "tools/encode_pypi_mp.py"),
                        "--src", str(dst.relative_to(ROOT)).replace("\\", "/"),
                        "--out", out, "--procs", "8",
                        "--tokenizer", "data/corpus3/tokenizer.json"])
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
