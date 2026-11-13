"""把训练所需的文件打包成**一个 tar.gz**，并校验二进制完整性。

## 为什么需要它

1. **分片是 `uint16` 裸数组，不是文本。** 用 FTP 文本模式、某些网盘同步、
   或任何"按行处理"的工具传输，**都会静默毁掉数据**（前两字节被当 BOM 就全错位）。
   本脚本**逐个分片比对 sha256**，并**在解包后重新校验**。

2. **上传 320 MB 到云端，一次成功比反复重试重要。** 单包 + 清单 + 校验值，
   出问题能立刻定位是哪个文件坏了。

3. **排除掉不该传的东西**（`runs/_ref/` 有 2 GB 的对照模型，练不了也用不上）。

用法::

    # 打包（默认不含对照模型与旧检查点）
    python tools/pack_for_cloud.py --out cloud/codemind_$(date +%Y%m%d).tar.gz

    # 解包后校验（在云上跑）
    python tools/pack_for_cloud.py --verify cloud/codemind_20261006.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 一定要带上的
INCLUDE = [
    "codemind", "tools", "tests",
    "data/corpus2", "data/chat",
    "CLOUD_TRAIN.md", "AGENTS.md", "REPORT.md",
    "pyproject.toml", "requirements.txt",
]
#: 明确排除的大件（对照模型 2 GB、历史运行）
EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "runs/_ref", "runs/_verify",
                "corpora", ".pytest_cache"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".part", ".tmp"}


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect() -> list[Path]:
    files: list[Path] = []
    for rel in INCLUDE:
        p = ROOT / rel
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                r = f.relative_to(ROOT).as_posix()
                if any(r.startswith(x + "/") or r == x for x in EXCLUDE_DIRS):
                    continue
                if f.suffix in EXCLUDE_SUFFIX:
                    continue
                files.append(f)
    return files


def pack(out: Path, include_runs: bool) -> int:
    files = collect()
    if include_runs:
        for f in sorted((ROOT / "runs").rglob("*.pt")):
            r = f.relative_to(ROOT).as_posix()
            if not any(r.startswith(x) for x in EXCLUDE_DIRS):
                files.append(f)

    out.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    total = 0
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        size = f.stat().st_size
        total += size
        manifest[rel] = {"size": size, "sha256": sha256(f)}

    print(f"[pack] {len(files)} 个文件 / {total/1e6:.1f} MB")
    print(f"[pack] 校验大文件（>1 MB）…")
    big = [(r, m) for r, m in manifest.items() if m["size"] > 1_000_000]
    for r, m in sorted(big, key=lambda kv: -kv[1]["size"])[:12]:
        print(f"    {m['size']/1e6:8.1f} MB  {m['sha256'][:16]}…  {r}")

    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "root": str(ROOT),
            "n_files": len(files), "total_bytes": total, "files": manifest}

    with tarfile.open(out, "w:gz") as tf:
        for f in files:
            tf.add(f, arcname=f.relative_to(ROOT).as_posix())
        # 清单也放进包里，云上解包后可以直接校验
        mf = out.parent / "MANIFEST.json"
        mf.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        tf.add(mf, arcname="MANIFEST.json")

    print(f"[pack] ✓ {out}  {out.stat().st_size/1e6:.1f} MB "
          f"(压缩率 {out.stat().st_size/total*100:.0f}%)")
    print(f"[pack] 校验清单：{out.parent/'MANIFEST.json'}")
    return 0


def verify(pkg: Path) -> int:
    """解包后校验 —— 用来证明**二进制没有被传输毁掉**。"""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="verify_") as td:
        td = Path(td)
        with tarfile.open(pkg) as tf:
            tf.extractall(td)
        mf = td / "MANIFEST.json"
        if not mf.exists():
            print("[verify] 包里没有 MANIFEST.json，无法校验")
            return 2
        meta = json.loads(mf.read_text(encoding="utf-8"))

        bad, missing, ok = [], [], 0
        for rel, m in meta["files"].items():
            p = td / rel
            if not p.exists():
                missing.append(rel)
                continue
            if p.stat().st_size != m["size"]:
                bad.append((rel, f"大小 {p.stat().st_size} != {m['size']}"))
                continue
            if m["size"] > 1_000_000 and sha256(p) != m["sha256"]:
                bad.append((rel, "sha256 不符"))
                continue
            ok += 1

        print(f"[verify] {ok}/{len(meta['files'])} 通过（大文件比对 sha256）")
        for rel, why in bad[:10]:
            print(f"  ✗ {rel}: {why}")
        for rel in missing[:10]:
            print(f"  ✗ 缺失 {rel}")
        if not bad and not missing:
            print("[verify] ✓ 完整，可以开始训练")
            return 0
        print(f"[verify] **失败**：{len(bad)} 个损坏 / {len(missing)} 个缺失")
        return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="输出 tar.gz 路径")
    ap.add_argument("--verify", default=None, help="改为校验已有包")
    ap.add_argument("--include-runs", action="store_true", help="连 runs/ 下的 .pt 一起打包")
    args = ap.parse_args()
    if args.verify:
        return verify(Path(args.verify))
    out = Path(args.out) if args.out else ROOT / "cloud" / (
        "codemind_" + time.strftime("%Y%m%d") + ".tar.gz")
    return pack(out, args.include_runs)


if __name__ == "__main__":
    raise SystemExit(main())
