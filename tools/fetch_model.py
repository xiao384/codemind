"""从 ModelScope 拉取公开模型权重，供 cb-1 的外部队照使用。

**为什么不用 HuggingFace**：实测 `huggingface.co` ConnectionRefused、
`cdn-lfs.huggingface.co` Timeout——**权重根本下不来**。
而 pypi 可用，ModelScope 实测 **6.4 MB/s**（是 hf-mirror 的 30 倍）。

用法::

    python tools/fetch_model.py --model AI-ModelScope/gpt2 --out runs/_ref/gpt2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.modelscope.cn/api/v1/models/{ns}/{name}/repo"

#: 跑 transformers 的因果 LM 所需的最小文件集
WANTED = ("config.json", "model.safetensors", "pytorch_model.bin",
          "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "generation_config.json")


def list_files(ns: str, name: str) -> list[tuple[str, int]]:
    url = f"{BASE.format(ns=ns, name=name)}/files?Revision=master&Root="
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read())
    return [(f["Path"], f.get("Size", 0)) for f in data["Data"]["Files"]]


def download(ns: str, name: str, rel: str, out: Path) -> int:
    q = urllib.parse.urlencode({"Revision": "master", "FilePath": rel})
    url = f"{BASE.format(ns=ns, name=name)}?{q}"
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst.stat().st_size
    tmp = dst.with_suffix(dst.suffix + ".part")
    t0, got = time.time(), 0
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total and got % (20 << 20) < (1 << 20):
                pct = got / total * 100
                print(f"    {rel}: {got/1e6:.0f}/{total/1e6:.0f} MB ({pct:.0f}%) "
                      f"{got/1024/(time.time()-t0):.0f} KB/s", flush=True)
    tmp.replace(dst)
    print(f"    ✓ {rel}  {got/1e6:.1f} MB  {got/1024/(time.time()-t0):.0f} KB/s", flush=True)
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="如 AI-ModelScope/gpt2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ns, _, name = args.model.partition("/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = dict(list_files(ns, name))
    print(f"[fetch] {args.model}: {len(files)} 个文件")
    have = [f for f in WANTED if f in files]
    if not have:
        print("[fetch] 没有找到所需文件，实际清单：")
        for p, s in sorted(files.items(), key=lambda kv: -kv[1])[:15]:
            print(f"    {s/1e6:8.1f} MB  {p}")
        return 2

    # 权重二选一：优先 safetensors
    if "model.safetensors" in have:
        have = [f for f in have if f != "pytorch_model.bin"]
    elif "pytorch_model.bin" in have:
        pass

    total = 0
    for rel in have:
        print(f"  下载 {rel} ({files[rel]/1e6:.1f} MB)")
        total += download(ns, name, rel, out)
    print(f"[fetch] 完成，共 {total/1e6:.1f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
