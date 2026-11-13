"""7-Zip 版的 pypi 流水线：解包 → 编码 → 删源。

## 为什么换 7-Zip

实测对比（20 个包 / 4.9 MB / 1,270 个文件）：

| 工具 | 速度 |
|---|---|
| Python `tarfile` | 约 **103 文件/秒** |
| **7-Zip（两趟）** | **373 文件/秒** ≈ **3.6 倍** |

解包是整个流水线的瓶颈（编码只要 17 分钟/批，解包要 50+ 分钟）。
换 7z 后 5 批从 9.5 小时压到约 3.7 小时。

## 为什么"两趟"

7-Zip 不能直接从 `.tar.gz` 解到目录 —— 它先解出 `.tar`，再解 `.tar`。
所以每批：
  ① 所有 .tar.gz → .tar（8 个进程并行）
  ② 所有 .tar → 文件（8 个进程并行，**这步慢**）
  ③ 删掉 .tar
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(r"E:\AI")
PY = r"E:\AI\.venv\Scripts\python.exe"
SEVENZ = r"C:\Program Files\7-Zip\7z.exe"
SRC = ROOT / "corpora/pypi_bulk3"
TARDIR = ROOT / "corpora/_tars"
DST = ROOT / "corpora/pypi_bulk3_src"
OUT = ROOT / "data/pypi_tok3"
BATCH = 15000
WORKERS = 8
START_IDX = 5          # 从 b005 开始（b000~b004 已完成）


def run7z(args: list[str]) -> bool:
    try:
        r = subprocess.run([SEVENZ] + args + ["-y", "-bso0", "-bsp0"],
                           capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


def extract_batch(files: list[Path]) -> tuple[int, int]:
    """两趟解包。返回 (成功, 失败) 计数。"""
    TARDIR.mkdir(parents=True, exist_ok=True)
    DST.mkdir(parents=True, exist_ok=True)

    # ① .tar.gz → .tar（并行）
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(lambda f: run7z(["x", str(f), f"-o{TARDIR}"]), files))
    tars = list(TARDIR.glob("*.tar"))
    print(f"    第一趟 {len(tars):,}/{len(files):,} → .tar  "
          f"{time.time()-t0:.0f}s", flush=True)

    # ② .tar → 文件（并行）
    t1 = time.time()
    ok = [0]
    lock_ok = []

    def one(t: Path) -> None:
        if run7z(["x", str(t), f"-o{DST}"]):
            lock_ok.append(1)
        try:
            t.unlink()
        except OSError:
            pass

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(one, tars))
    ok[0] = len(lock_ok)
    print(f"    第二趟 {ok[0]:,} 成功  {time.time()-t1:.0f}s", flush=True)
    return ok[0], len(files) - ok[0]


def main() -> int:
    allf = sorted(SRC.glob("*.tar.gz"))
    print(f"[7z] 剩 {len(allf):,} 个包，从批次 {START_IDX} 开始")
    nb = (len(allf) + BATCH - 1) // BATCH
    total = 0

    for k in range(nb):
        idx = START_IDX + k
        files = allf[k * BATCH:(k + 1) * BATCH]
        print(f"\n[7z] 批次 {idx}: {len(files):,} 个包", flush=True)
        t0 = time.time()

        ok, bad = extract_batch(files)
        print(f"[7z] 批次 {idx}: 解包 {ok:,} 成功 / {bad} 失败  "
              f"{(time.time()-t0)/60:.1f} 分钟", flush=True)

        out = OUT / f"b{idx:03d}"
        out.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        subprocess.run([PY, "-u", str(ROOT / "tools/encode_pypi_mp.py"),
                        "--src", str(DST.relative_to(ROOT)).replace("\\", "/"),
                        "--out", str(out.relative_to(ROOT)).replace("\\", "/"),
                        "--tokenizer", "data/corpus3/tokenizer.json",
                        "--procs", "8", "--name", f"t3b{idx:03d}"])
        import json
        tok = 0
        for m in out.glob("*.meta.json"):
            tok += int(json.loads(m.read_text(encoding="utf-8")).get("total_tokens", 0))
        total += tok
        print(f"[7z] 批次 {idx}: 编码 {tok:,} token / "
              f"{(time.time()-t1)/60:.1f} 分钟", flush=True)
        print(f"[7z] 累计 {total:,} token（{total/1e9:.2f}B）", flush=True)

        # 删源：解包目录 + tar.gz（tar 已在第二趟删掉）
        print(f"[7z] 批次 {idx}: 删源…", flush=True)
        shutil.rmtree(DST, ignore_errors=True)
        if DST.exists():                     # 长路径兜底
            empty = ROOT / "_empty"
            empty.mkdir(exist_ok=True)
            subprocess.run(["robocopy", str(empty), str(DST), "/MIR", "/NFL",
                            "/NDL", "/NJH", "/NJS", "/NC", "/NS"], capture_output=True)
            shutil.rmtree(DST, ignore_errors=True)
            shutil.rmtree(empty, ignore_errors=True)
        shutil.rmtree(TARDIR, ignore_errors=True)
        for f in files:
            try:
                f.unlink()
            except OSError:
                pass
        print(f"[7z] 批次 {idx} 完成，用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)

    print(f"\n[7z] ✓ 全部完成：新增 {total:,} token = {total/1e9:.2f}B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
