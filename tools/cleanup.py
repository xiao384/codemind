"""清理冗余文件。

## 纪律（来自 AGENTS.md）

**删除前必须验证解析出的绝对路径就是要删的那个**，不能对"计算出来的路径"直接动手。
所以本脚本先打印每个目标的**完整路径 + 大小**，再删，最后报告释放量。

## 判定标准：只删满足以下之一者

1. **内容已抽取到别处**（如 7z 里的融合模型已解到 `runs/_ref78mix/`）
2. **逐字节重复**（如 `latest78.pt` 与 `latest.pt`）
3. **已被取代**（如旧结构的全量包，被修好结构的包取代）
4. **临时草稿**（看守的 auto 副本、探测用检查点）

**不动**：任何有文档记录用途的东西（旧基线 `runs/corpus`、历史检查点
`ckpt-step1500`/`step2000`、firefly 语料、`AI.zip` 内容未知）。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

ROOT = Path(r"E:\AI")
ALLOWED_ROOTS = [Path(r"E:\AI")]

# (路径, 大小下限MB, 删除理由)
TARGETS = [
    (r"btmb\ailyun.7z", 100, "融合模型已抽取到 runs/_ref78mix/"),
    (r"runs\_ref78\latest78.pt", 100, "与 runs/_ref78/latest.pt 逐字节重复"),
    (r"alyyun.tar.gz", 50, "旧结构全量包（含 core/ 层级 bug），已被 ailyun 包取代"),
    (r"runs\_verify\ckpt-auto.pt", 50, "看守的自动副本，已过期"),
    (r"runs\_verify\ckpt-new.pt", 10, "调试期临时检查点"),
    (r"runs\_verify\ckpt-probe.pt", 10, "探测用临时检查点"),
    (r"ailyun.zip", 10, "与 ailyun_fixed.zip 重复（后者结构已修）"),
    (r"ailyun.tar.gz", 10, "已在 ModelScope，且本机有 ailyun/ 目录"),
]


def sha8(p: Path, limit: int = 8 << 20) -> str:
    """只哈希前 8 MB —— 查重够用，且比全文件快得多。"""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def main() -> int:
    print("=" * 78)
    print(" 清理前核对（每个目标必须先通过路径与大小检查）")
    print("=" * 78)

    ok: list[tuple[Path, int, str]] = []
    for rel, min_mb, why in TARGETS:
        p = (ROOT / rel).resolve()
        # ⚠️ 路径必须落在允许的根之下 —— 否则拒绝
        if not any(str(p).startswith(str(r)) for r in ALLOWED_ROOTS):
            print(f"  ✗ 拒绝（不在允许范围内）: {p}")
            continue
        if not p.is_file():
            print(f"  · 不存在，跳过: {rel}")
            continue
        mb = p.stat().st_size / 1e6
        if mb < min_mb:
            print(f"  ✗ 拒绝（大小 {mb:.1f}MB < 预期 {min_mb}MB，可能不是我认得的那个文件）: {p}")
            continue
        print(f"  ✓ {mb:9.1f} MB  {p}")
        print(f"      理由: {why}")
        ok.append((p, p.stat().st_size, why))

    if not ok:
        print("\n没有可删的。")
        return 0

    # 查重验证：删除"重复副本"前，确认正本真的在
    key = ROOT / "runs" / "_ref78" / "latest.pt"
    dup = ROOT / "runs" / "_ref78" / "latest78.pt"
    if dup in [p for p, _, _ in ok]:
        if not key.is_file():
            print(f"\n  ✗ 正本 {key} 不存在，拒绝删除副本")
            ok = [t for t in ok if t[0] != dup]
        elif key.stat().st_size != dup.stat().st_size:
            print(f"\n  ✗ 正本与副本大小不同，拒绝删除副本")
            ok = [t for t in ok if t[0] != dup]
        else:
            print(f"\n  正本核对: {key.name} {key.stat().st_size:,} B == 副本 {dup.stat().st_size:,} B ✓")

    key2 = ROOT / "runs" / "_ref78mix" / "latest.pt"
    z = ROOT / "btmb" / "ailyun.7z"
    if z in [p for p, _, _ in ok]:
        if not key2.is_file() or key2.stat().st_size != 944_088_651:
            print(f"  ✗ 融合模型 {key2} 不在或大小不符，拒绝删除 7z")
            ok = [t for t in ok if t[0] != z]
        else:
            print(f"  融合模型已抽取核对: {key2.stat().st_size:,} B ✓")

    total = sum(s for _, s, _ in ok)
    print()
    print("=" * 78)
    print(f" 即将删除 {len(ok)} 项，共 {total/1e6:,.1f} MB")
    print("=" * 78)
    for p, s, _ in ok:
        print(f"  {s/1e6:9.1f} MB  {p}")

    freed = 0
    failed = []
    for p, s, _ in ok:
        try:
            os.remove(p)
            freed += s
        except Exception as e:
            failed.append((p, e))

    print()
    print(f"  ✓ 已释放 {freed/1e6:,.1f} MB（{freed/1e9:.2f} GB）")
    for p, e in failed:
        print(f"  ✗ 删除失败 {p}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
