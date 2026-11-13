"""在 ModelScope 上探测可用的**指令语料**与**代码语料**。

用已跑通的 `/api/v1/datasets/{ns}/{name}/repo/tree` 端点
（⚠️ 数据集用 `repo/tree`，**模型**用 `repo/files` —— 混用会全部 404，
本项目已经在这上面踩过一次）。

用法::  python tools/search_datasets.py
"""
from __future__ import annotations

import json
import urllib.request

CANDIDATES = [
    # ---- 中文指令 ----
    ("AI-ModelScope", "alpaca-gpt4-data-zh", "指令"),
    ("AI-ModelScope", "firefly-train-1.1M", "指令"),
    ("AI-ModelScope", "belle-3.5M", "指令"),
    ("AI-ModelScope", "Chinese-LLaMA-Alpaca", "指令"),
    ("AI-ModelScope", "sharegpt-chinese", "对话"),
    ("AI-ModelScope", "ShareGPT-Chinese", "对话"),
    ("modelscope", "alpaca-chinese", "指令"),
    ("AI-ModelScope", "test_instruction_zh", "指令"),
    ("swift", "alpaca-gpt4-data-zh", "指令"),
    ("modelscope", "Chinese-Instruction", "指令"),
    # ---- 代码 ----
    ("AI-ModelScope", "code_search_net", "代码"),
    ("modelscope", "code_search_net", "代码"),
    ("AI-ModelScope", "codeparrot-clean", "代码"),
    ("AI-ModelScope", "the-stack", "代码"),
    ("AI-ModelScope", "CodeAlpaca", "代码指令"),
    ("AI-ModelScope", "codealpaca", "代码指令"),
    ("AI-ModelScope", "python-code-3M", "代码"),
    ("AI-ModelScope", "starcoderdata", "代码"),
    ("AI-ModelScope", "github-code", "代码"),
    ("beyond", "code_search_net", "代码"),
]


def tree(ns: str, name: str) -> list[tuple[str, int]] | None:
    url = (f"https://www.modelscope.cn/api/v1/datasets/{ns}/{name}"
           f"/repo/tree?Revision=master")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.loads(r.read())
    except Exception:
        return None
    files = d.get("Data", {}).get("Files") or []
    return [(f.get("Path", "?"), f.get("Size", 0)) for f in files]


def main() -> int:
    hits = []
    print(f"{'数据集':46}{'类型':8}{'文件':>5}{'大小':>12}")
    for ns, name, kind in CANDIDATES:
        files = tree(ns, name)
        full = f"{ns}/{name}"
        if files is None:
            print(f"  {full:44}{kind:8}{'—':>5}{'不存在':>12}")
            continue
        total = sum(s for _, s in files)
        print(f"  {full:44}{kind:8}{len(files):>5}{total/1e6:>10.1f} MB")
        for p, s in sorted(files, key=lambda x: -x[1])[:3]:
            print(f"      {s/1e6:9.1f} MB  {p}")
        if total > 5e6:
            hits.append((full, kind, total))
    print()
    print(f"=== 可用（>5 MB）共 {len(hits)} 个 ===")
    for full, kind, total in sorted(hits, key=lambda x: -x[2]):
        print(f"  {total/1e6:9.1f} MB  [{kind}]  {full}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
