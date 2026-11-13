"""从 pypi 镜像批量拉 sdist —— 规模化的那一版。

## 与 `fetch_pypi_http.py` 的区别

那个用的是**手写的 180 个包名**（挑常见的）。这个从**镜像的完整索引**里取名字，
所以能拉到几千甚至几万个包 —— **这是唯一能把代码语料做到 B 级的路径**。

实测换算（169 个包的真实数据）：

    1 个包 ≈ 1.7 MB 原始 ≈ 0.94M token
    下载 2.8 MB/s

## 为什么先下再筛

sdist 里约一半内容（JS/CSS/文档/生成代码）会在编码阶段被过滤掉，
但**下载前无法判断**。所以策略是「广下 + 严筛」：
下载便宜（2.8 MB/s），编码贵（12 MB/分钟）—— **筛选放在编码阶段做**。

## ⚠️ 索引页的大小

完整索引有几十万行。**一次性读进内存没问题（约 20 MB）**，但不要反复拉。

用法::

    python tools/fetch_pypi_bulk.py --want 2000 --out corpora/pypi_bulk
"""
from __future__ import annotations

import argparse
import random
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

INDEX_ROOT = "https://pypi.tuna.tsinghua.edu.cn/simple/"
UA = {"User-Agent": "python-urllib"}


def all_package_names() -> list[str]:
    """拉完整索引，返回包名列表。

    ⚠️ 索引页的格式是 `<a href="pkgname/">pkgname</a>` ——
    **href 是纯相对路径（没有前导斜杠）**。
    我第一版正则写的是 `href="[^"]*?/([^/"]+)/"`，强制要求名字前面有斜杠，
    于是 46.6 MB 的索引解析出 **0 个包名**。
    **凭想象写正则的代价** —— 与"凭记忆拼文件名"是同一类错误。
    """
    t0 = time.time()
    with urllib.request.urlopen(
            urllib.request.Request(INDEX_ROOT, headers=UA), timeout=120) as r:
        html = r.read().decode("utf-8", "replace")
    names = re.findall(r'<a[^>]+href="([^"/]+)/"[^>]*>', html)
    names = sorted({n for n in names if n and not n.startswith((".", "_"))})
    print(f"[bulk] 索引 {len(html)/1e6:.1f} MB / {len(names):,} 个包名 "
          f"/ {time.time()-t0:.0f}s")
    return names


def sdist_url(pkg: str) -> str | None:
    base = INDEX_ROOT + pkg + "/"
    try:
        with urllib.request.urlopen(
                urllib.request.Request(base, headers=UA), timeout=20) as r:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpora/pypi_bulk")
    ap.add_argument("--want", type=int, default=2000, help="下载多少个包")
    ap.add_argument("--max-mb", type=float, default=30.0, help="单包上限")
    ap.add_argument("--seed", type=int, default=20261015, help="抽样种子（可复现）")
    args = ap.parse_args()

    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    names = all_package_names()
    if not names:
        print("[bulk] ✗ 索引解析失败")
        return 1
    rng = random.Random(args.seed)
    rng.shuffle(names)
    print(f"[bulk] 随机抽样（seed={args.seed}，可复现）→ 目标 {args.want} 个")

    t0 = time.time()
    ok = skip = fail = 0
    total = 0
    for pkg in names:
        if ok >= args.want:
            break
        dest = out / f"{pkg}.tar.gz"
        if dest.exists() and dest.stat().st_size > 1024:
            skip += 1
            ok += 1
            continue        # 已下的算已达成
        url = sdist_url(pkg)
        if not url:
            fail += 1
            continue
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
            if len(data) > args.max_mb * 1e6 or len(data) < 2048:
                fail += 1
                continue
            dest.write_bytes(data)
            total += len(data)
            ok += 1
        except Exception:
            fail += 1
        if (ok + fail) % 100 == 0:
            el = time.time() - t0
            print(f"  {ok}/{args.want} 已下  {fail} 失败  {total/1e6:.0f} MB  "
                  f"{el/60:.1f} 分钟  "
                  f"（预计还需 {(args.want-ok)*el/max(1,ok)/60:.0f} 分钟）", flush=True)

    el = time.time() - t0
    files = list(out.glob("*.tar.gz"))
    sz = sum(f.stat().st_size for f in files)
    print(f"\n[bulk] ✓ {len(files)} 个包 / {sz/1e6:.1f} MB / {el/60:.1f} 分钟")
    print(f"[bulk] 本次新增 {ok-skip}，跳过 {skip}，失败 {fail}")
    print(f"[bulk] 换算: 约 {len(files)/169*158.2:.0f}M token（按 169 包=158.2M 的比例）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
