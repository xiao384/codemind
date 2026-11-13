"""清理未上传的旧文件与已编码完的中间产物。

## 纪律

删除前**逐个核对路径与大小**，并把每个目标的完整路径打印出来 ——
AGENTS.md 明确要求"删除前验证解析出的绝对路径就是要删的那个"。

## 什么算"可以删"

1. **已被取代的包**（新包已生成，旧包没上传过）
2. **已经上传过的包**（内容在 ModelScope 上，本地可重建）
3. **已编码完的中间产物**（源码 tar.gz 与解包目录 —— token 已经落盘，原始文件不再需要）

## 什么不删

* `codemind-models.7z` —— 用户明确要求的模型包
* `data/` 下的语料分片 —— 那是训练数据本身
* `runs/` 下的检查点 —— 训练成果
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(r"E:\AI")

# (相对路径, 是否为目录, 下限MB, 理由)
TARGETS = [
    ("ailyun_fixed.zip", False, 10, "旧结构包，从未上传（用 mv core/* . 绕过了）"),
    ("chat_for_cloud.zip", False, 5, "已上传到 ModelScope，本地可重建"),
    ("codemind-transfer.zip", False, 0, "更早的临时包"),
    ("corpora/pypi_test", True, 0, "下载通道验证用的 5 个测试包"),
    ("corpora/pypi_bulk", True, 500, "源码 tar.gz —— **token 已编码进 pypi_tok**"),
    ("corpora/pypi_bulk_src", True, 500, "解包目录 —— 同上"),
    ("corpora/pypi_bulk2", True, 1000, "源码 tar.gz —— **已编码进 pypi_tok2**"),
    ("corpora/pypi_bulk2_src", True, 1000, "解包目录 —— 同上"),
    ("corpora/pypi_extra", True, 100, "第一批 sdist —— 已编码进 corpus3"),
    ("corpora/pypi_src", True, 100, "第一批解包 —— 同上"),
]


def size_of(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    print("=" * 78)
    print(" 清理前核对（逐个打印完整路径与大小）")
    print("=" * 78)
    ok = []
    for rel, is_dir, min_mb, why in TARGETS:
        p = (ROOT / rel).resolve()
        if not str(p).startswith(str(ROOT)):
            print(f"  ✗ 拒绝（越出工作区）: {p}")
            continue
        if not p.exists():
            print(f"  · 不存在: {rel}")
            continue
        if p.is_dir() != is_dir:
            print(f"  ✗ 拒绝（类型不符，期望{'目录' if is_dir else '文件'}）: {p}")
            continue
        mb = size_of(p) / 1e6
        if mb < min_mb:
            print(f"  ✗ 拒绝（{mb:.1f}MB < 预期 {min_mb}MB，可能不是我认得的那个）: {p}")
            continue
        print(f"  ✓ {mb:9.1f} MB  {p}")
        print(f"      理由: {why}")
        ok.append((p, mb))

    total = sum(m for _, m in ok)
    print()
    print(f" 即将删除 {len(ok)} 项，共 {total/1000:,.1f} GB")
    for p, mb in ok:
        print(f"   {mb:10.1f} MB  {p}")

    freed = 0
    for p, mb in ok:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                os.remove(p)
            freed += mb
        except Exception as e:
            print(f"  ✗ 删除失败 {p}: {type(e).__name__}: {e}")

    print()
    print(f"  ✓ 已释放 {freed/1000:,.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
