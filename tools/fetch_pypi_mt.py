"""多线程批量下载 pypi sdist。

## 为什么下载用线程、编码用进程

| 任务 | 瓶颈 | 正确工具 | 实测 |
|---|---|---|---|
| **下载** | **网络 IO** | **多线程** | 单线程 1.2 MB/s |
| **编码** | CPU（纯 Python BPE） | **多进程**（GIL 卡线程） | 单进程 11.7 MB/分钟 → 8 进程 60× |

**用错工具会白费力气**：下载用多进程只会增加内存占用而不提速（甚至更慢）；
编码用多线程则完全无效（GIL）。

## 每包产量很低，需要量大

实测（这是必须记住的数）：

* 精选 169 包 → 158.2M token → **0.94M token/包**
* 随机 3,000 包 → 287.0M token → **0.096M token/包**

**随机包产量只有精选的 1/10** —— 大量是 `0`、`0-0-1` 这类空壳包。
所以"下 3,000 个包能得 2.8B token"的外推乐观了 **9.8 倍**。

用法::

    python tools/fetch_pypi_mt.py --want 20000 --workers 24 --out corpora/pypi_bulk2
"""
from __future__ import annotations

import argparse
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

INDEX_ROOT = "https://pypi.tuna.tsinghua.edu.cn/simple/"
UA = {"User-Agent": "python-urllib"}

_lock = threading.Lock()
_stat = {"ok": 0, "fail": 0, "bytes": 0, "skip": 0}


def all_package_names() -> list[str]:
    with urllib.request.urlopen(
            urllib.request.Request(INDEX_ROOT, headers=UA), timeout=120) as r:
        html = r.read().decode("utf-8", "replace")
    # ⚠️ href 是纯相对路径（`pkgname/`），不能强制要求前导斜杠
    names = re.findall(r'<a[^>]+href="([^"/]+)/"[^>]*>', html)
    return sorted({n for n in names if n and not n.startswith((".", "_"))})


def sdist_url(pkg: str) -> str | None:
    base = INDEX_ROOT + pkg + "/"
    try:
        with urllib.request.urlopen(
                urllib.request.Request(base, headers=UA), timeout=15) as r:
            html = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    raw = re.findall(r'href="([^"#]+\.tar\.gz)', html)
    if not raw:
        return None
    links = [urljoin(base, l) for l in raw]

    def ver(u: str) -> tuple:
        m = re.search(r"-(\d[^-]*?)\.tar\.gz$", u.rsplit("/", 1)[-1])
        if not m:
            return (0,)
        return tuple(int(x) if x.isdigit() else 0
                     for x in re.split(r"[.\-]", m.group(1)))[:4]

    return max(links, key=ver)


def fetch_one(pkg: str, out: Path, max_mb: float) -> str:
    dest = out / f"{pkg}.tar.gz"
    if dest.exists() and dest.stat().st_size > 1024:
        with _lock:
            _stat["skip"] += 1
        return "skip"
    url = sdist_url(pkg)
    if not url:
        with _lock:
            _stat["fail"] += 1
        return "fail"
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=45) as r:
            data = r.read()
        if len(data) > max_mb * 1e6 or len(data) < 2048:
            with _lock:
                _stat["fail"] += 1
            return "fail"
        dest.write_bytes(data)
        with _lock:
            _stat["ok"] += 1
            _stat["bytes"] += len(data)
        return "ok"
    except Exception:
        with _lock:
            _stat["fail"] += 1
        return "fail"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpora/pypi_bulk2")
    ap.add_argument("--want", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-mb", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=20261016)
    args = ap.parse_args()

    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    names = all_package_names()
    print(f"[mt] 索引 {len(names):,} 个包名")
    rng = random.Random(args.seed)
    rng.shuffle(names)
    todo = names[: args.want * 2]      # 留余量，失败的不补
    todo = [n for n in todo if not (out / f"{n}.tar.gz").exists()][: args.want]
    print(f"[mt] 目标 {len(todo):,} 个包  worker {args.workers}  → {out}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(fetch_one, p, out, args.max_mb) for p in todo]
        done = 0
        for _ in as_completed(futs):
            done += 1
            if done % 500 == 0:
                el = time.time() - t0
                mb = _stat["bytes"] / 1e6
                eta = (len(todo) - done) * el / max(1, done)
                print(f"  {done:,}/{len(todo):,}  成功 {_stat['ok']:,}  "
                      f"跳过 {_stat['skip']:,}  失败 {_stat['fail']:,}  "
                      f"{mb:.0f} MB  {mb/el:.2f} MB/s  "
                      f"已用 {el/60:.1f} 分钟  还需 {eta/60:.0f} 分钟", flush=True)

    el = time.time() - t0
    files = list(out.glob("*.tar.gz"))
    sz = sum(f.stat().st_size for f in files)
    print(f"\n[mt] ✓ {len(files):,} 个包 / {sz/1e6:.1f} MB / {el/60:.1f} 分钟")
    print(f"[mt] 平均 {sz/1e6/el*60:.2f} MB/s（单线程实测是 1.2 MB/s）")
    print(f"[mt] 按 0.096M token/包 估算 → 约 {len(files)*0.096:.1f}M token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
