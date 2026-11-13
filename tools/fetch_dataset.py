"""从 ModelScope 拉数据集（**在云端跑，不要在本机传**）。

## 为什么需要它

训练用的指令语料本来就在 ModelScope 上：

* `AI-ModelScope/firefly-train-1.1M`   1,171 MB（多任务指令）
* `AI-ModelScope/alpaca-gpt4-data-zh`    32 MB（中文指令）

**在平台上直接下（实测 11 MB/s）比从本机上传 1.17 GB 快得多，也省流量。**

## ⚠️ 一个踩过的坑

**数据集用 `/repo/tree`，模型用 `/repo/files`** —— 两个端点不一样。
混用会**全部 404**，而 404 长得像"数据集不存在"，很容易误判成"平台没有这个数据"。
本项目已经在这上面浪费过一轮探测。

用法::

    python tools/fetch_dataset.py --list AI-ModelScope/firefly-train-1.1M
    python tools/fetch_dataset.py --get AI-ModelScope/firefly-train-1.1M \\
        --file firefly-train-1.1M.jsonl --out corpora/data
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.modelscope.cn/api/v1/datasets/{ns}/{name}"


def list_files(ns: str, name: str) -> list[tuple[str, int]]:
    url = f"{BASE.format(ns=ns, name=name)}/repo/tree?Revision=master"
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.loads(r.read())
    return [(f["Path"], f.get("Size", 0)) for f in d["Data"]["Files"]]


def download(ns: str, name: str, rel: str, out: Path) -> int:
    q = urllib.parse.urlencode({"Revision": "master", "FilePath": rel})
    url = f"{BASE.format(ns=ns, name=name)}/repo?{q}"
    dst = out / Path(rel).name
    out.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    t0, got, last = time.time(), 0, 0
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
            if got - last >= 100 << 20:
                last = got
                pct = f"{got/total*100:.0f}%" if total else "?"
                print(f"    {got/1e6:7.0f} MB {pct}  {got/1024/(time.time()-t0):6.0f} KB/s",
                      flush=True)
    tmp.replace(dst)
    print(f"  ✓ {dst.name}  {got/1e6:.1f} MB / {time.time()-t0:.0f}s "
          f"= {got/1024/(time.time()-t0):.0f} KB/s")
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", metavar="NS/NAME", help="只列文件")
    ap.add_argument("--get", metavar="NS/NAME", help="下载")
    ap.add_argument("--file", default=None, help="要下的文件名（默认取最大的那个）")
    ap.add_argument("--out", default="corpora/data")
    args = ap.parse_args()

    spec = args.list or args.get
    if not spec:
        ap.error("需要 --list 或 --get")
    ns, _, name = spec.partition("/")

    try:
        files = list_files(ns, name)
    except Exception as e:
        print(f"✗ 读不到 {spec}：{type(e).__name__} {e}")
        print("  可能原因：① 名字不对 ② 用了 /repo/files（数据集要用 /repo/tree）")
        return 2

    total = sum(s for _, s in files)
    print(f"[dataset] {spec}: {len(files)} 个文件 / {total/1e6:.1f} MB")
    for p, s in sorted(files, key=lambda x: -x[1])[:10]:
        print(f"    {s/1e6:10.1f} MB  {p}")
    if args.list:
        return 0

    rel = args.file
    if rel is None:
        rel = max(files, key=lambda x: x[1])[0]
        print(f"[dataset] 未指定 --file，取最大的：{rel}")
    download(ns, name, rel, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
