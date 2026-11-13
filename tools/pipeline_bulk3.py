"""流水线：解包 → 多进程编码 → 删源，逐批推进，最后合并成 corpus5。

## 为什么不一次全解包

175,000 个包解包后约 130 GB，加上 67 GB 的 sdist，磁盘峰值 197 GB。
**分批处理可以把峰值降到约 20 GB** —— 每批解包、编码、立刻删掉。

## 判据：下载完成的两个信号（不只看进程）

1. 进程消失 **且** 文件数不再增长（连续 3 分钟不变）
2. 这才算完成 —— 单看进程退出可能撞上脚本提前崩溃

## 每批的失败隔离

单批的某个文件坏了不该拖垮全局 —— 解包和编码都各自 try/except，
失败的计数后继续。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\AI")
PY = r"E:\AI\.venv\Scripts\python.exe"
SRC = ROOT / "corpora/pypi_bulk3"
TMP = ROOT / "corpora/pypi_bulk3_src"
OUT = ROOT / "data/pypi_tok3"
BATCH = 15000          # 每批处理的包数


def wait_download() -> int:
    """等下载结束：进程消失 + 文件数连续 3 分钟不变。"""
    print("[pipe] 等下载完成…")
    t0 = time.time()
    last_n, stable = -1, 0
    while True:
        n = len(list(SRC.glob("*.tar.gz")))
        running = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'fetch_pypi_mt' }).Count"],
            capture_output=True, text=True)
        try:
            alive = int(running.stdout.strip() or 0) > 0
        except ValueError:
            alive = False
        if n == last_n:
            stable += 1
        else:
            stable = 0
            last_n = n
        el = time.time() - t0
        if el > 60 and not alive and stable >= 6:      # 6 × 30s = 3 分钟
            print(f"[pipe] 下载结束：{n:,} 个文件 / {el/60:.0f} 分钟")
            return n
        if el > 4 * 3600:
            print(f"[pipe] 下载等待超时（已有 {n:,} 个）")
            return n
        if int(el) % 180 < 30:
            print(f"[pipe]   {n:,} 个 / {el/60:.0f} 分钟 / "
                  f"进程{'在' if alive else '已退'}", flush=True)
        time.sleep(30)


def process_batch(files: list[Path], idx: int) -> int:
    """解包 + 编码 + 删源。返回 token 数。"""
    TMP.mkdir(parents=True, exist_ok=True)

    print(f"[pipe] 批次 {idx}: 解包 {len(files):,} 个包…", flush=True)
    t0 = time.time()
    ok = bad = 0
    import tarfile
    for i, f in enumerate(files, 1):
        try:
            with tarfile.open(f, "r:gz") as tf:
                tf.extractall(TMP, filter="data")
            ok += 1
        except Exception:
            bad += 1
        if i % 3000 == 0:
            print(f"    解包 {i:,}/{len(files):,}  成功 {ok:,}  "
                  f"{(time.time()-t0)/60:.1f} 分钟", flush=True)
    print(f"[pipe] 批次 {idx}: 解包完成 {ok:,}/{bad} 失败 "
          f"{(time.time()-t0)/60:.1f} 分钟", flush=True)

    out = OUT / f"b{idx:03d}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[pipe] 批次 {idx}: 多进程编码…", flush=True)
    t1 = time.time()
    subprocess.run([PY, "-u", str(ROOT / "tools/encode_pypi_mp.py"),
                    "--src", str(TMP.relative_to(ROOT)).replace("\\", "/"),
                    "--out", str(out.relative_to(ROOT)).replace("\\", "/"),
                    "--tokenizer", "data/corpus3/tokenizer.json",
                    "--procs", "8", "--name", f"t3b{idx:03d}"])
    print(f"[pipe] 批次 {idx}: 编码完成 {(time.time()-t1)/60:.1f} 分钟", flush=True)

    # 算这批的 token 数
    import json
    tok = 0
    for m in out.glob("*.meta.json"):
        tok += int(json.loads(m.read_text(encoding="utf-8")).get("total_tokens", 0))

    # 删源（tar.gz + 解包目录）—— 磁盘立省
    print(f"[pipe] 批次 {idx}: 删除源文件…", flush=True)
    for f in files:
        try:
            f.unlink()
        except OSError:
            pass
    shutil.rmtree(TMP, ignore_errors=True)
    # ⚠️ 长路径目录 shutil 删不干净，用 robocopy 兜底
    if TMP.exists():
        empty = ROOT / "_empty"
        empty.mkdir(exist_ok=True)
        subprocess.run(["robocopy", str(empty), str(TMP), "/MIR", "/NFL", "/NDL",
                        "/NJH", "/NJS", "/NC", "/NS"], capture_output=True)
        shutil.rmtree(TMP, ignore_errors=True)
        shutil.rmtree(empty, ignore_errors=True)
    return tok


def main() -> int:
    n = wait_download()
    allf = sorted(SRC.glob("*.tar.gz"))
    print(f"[pipe] 共 {len(allf):,} 个包，分 {(len(allf)+BATCH-1)//BATCH} 批")
    total = 0
    for idx in range((len(allf) + BATCH - 1) // BATCH):
        files = allf[idx * BATCH:(idx + 1) * BATCH]
        tok = process_batch(files, idx)
        total += tok
        print(f"[pipe] 累计 token {total:,}（{total/1e9:.2f}B）", flush=True)
        getattr(sys, "stdout").flush()

    print(f"[pipe] ✓ 全部完成：{total:,} token = {total/1e9:.2f}B")
    print(f"[pipe] 输出 {OUT}（合并成 corpus5 请再跑 merge 脚本）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
