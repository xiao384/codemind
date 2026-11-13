"""等 pypi_tok2 编码完成后，自动合并出 corpus4。

## 合并策略：按块交错

三个来源（corpus3 / pypi_tok / pypi_tok2）的代码风格不同 ——
标准库偏严谨、随机包偏杂乱。**首尾拼接会让训练后半段只见一种分布**，
loss 会出现人为台阶。按块交错能让每个训练窗口里三种都在场。

## 判据：**看产出物，不看日志**

本项目在看守上栽过三次（盯错文件名、子进程被杀、日志缓冲）。
所以这里判定"编码完成"用**两个独立信号**：

1. `pypi.meta.json` 存在（编码器只在全部写完后才写它）
2. 该 meta 里的 `total_tokens` 与 `.bin` 文件实际大小一致

**两者都满足才算完成。**
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\AI")
EOT = 256


def load_dir(d: Path) -> np.ndarray:
    metas = sorted(d.glob("*.meta.json"))
    if not metas:
        raise FileNotFoundError(f"{d} 下没有 *.meta.json")
    m = json.loads(metas[0].read_text(encoding="utf-8"))
    fs = list(m.get("train_files", [])) + list(m.get("val_files", []))
    if not fs:
        raise ValueError(f"{metas[0]} 里没有分片清单")
    return np.concatenate([np.fromfile(d / f, dtype=np.uint16) for f in fs])


def ready(d: Path) -> tuple[bool, int]:
    """检查编码是否真的写完（两个信号）。"""
    metas = list(d.glob("*.meta.json"))
    if not metas:
        return False, 0
    try:
        m = json.loads(metas[0].read_text(encoding="utf-8"))
    except Exception:
        return False, 0
    fs = list(m.get("train_files", [])) + list(m.get("val_files", []))
    if not fs or not all((d / f).is_file() for f in fs):
        return False, 0
    on_disk = sum((d / f).stat().st_size // 2 for f in fs)
    claimed = int(m.get("total_tokens", 0))
    if on_disk != claimed:
        return False, on_disk
    return True, claimed


def main() -> int:
    srcs = [ROOT / "data/corpus3", ROOT / "data/pypi_tok"]
    wait_for = ROOT / "data/pypi_tok2"
    out = ROOT / "data/corpus4"

    print("[merge] 等 data/pypi_tok2 编码完成…")
    t0 = time.time()
    last = 0.0
    while True:
        ok, n = ready(wait_for)
        if ok:
            print(f"[merge] pypi_tok2 就绪：{n:,} token（{time.time()-t0:.0f}s）")
            break
        el = time.time() - t0
        if el - last > 300:
            last = el
            print(f"[merge]   等待中 {el/60:.0f} 分钟（磁盘上 {n:,} token）", flush=True)
        if el > 3 * 3600:
            print("[merge] 超时 3 小时，退出")
            return 2
        time.sleep(30)

    parts = []
    for d in srcs + [wait_for]:
        try:
            a = load_dir(d)
            print(f"[merge] {d.name:12} {len(a):>13,} token")
            parts.append(a)
        except Exception as e:
            print(f"[merge] ✗ {d}: {type(e).__name__}: {e}")
            return 1

    # 按块交错（见文件头说明）
    b = 2_000_000
    mixed = []
    i = [0] * len(parts)
    while any(i[k] < len(parts[k]) for k in range(len(parts))):
        for k, a in enumerate(parts):
            if i[k] < len(a):
                mixed.append(a[i[k]:i[k] + b])
                i[k] += b
    arr = np.concatenate(mixed)
    print(f"[merge] 交错后 {len(arr):,} token")

    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.bin"):
        f.unlink()

    n_val = max(1, int(len(arr) * 0.01))
    train, val = arr[:len(arr) - n_val], arr[len(arr) - n_val:]

    def write(a: np.ndarray, split: str) -> list[str]:
        names, S = [], 8_000_000
        for k in range(max(1, (len(a) + S - 1) // S)):
            nm = f"corpus4.{split}{k:05d}.bin"
            a[k * S:(k + 1) * S].tofile(out / nm)
            names.append(nm)
        return names

    tr, va = write(train, "train"), write(val, "val")
    km = json.loads(sorted((ROOT / "data/corpus3").glob("*.meta.json"))[0]
                    .read_text(encoding="utf-8"))
    (out / "corpus4.meta.json").write_text(json.dumps({
        "name": "corpus4", "dtype": "uint16",
        "train_files": tr, "val_files": va,
        "total_tokens": int(len(arr)), "vocab_size": km.get("vocab_size", 4096),
        "n_docs": 0, "created_at": time.time(),
        "sources": [str(d) for d in srcs + [wait_for]],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    import shutil
    shutil.copy2(ROOT / "data/corpus3/tokenizer.json", out / "tokenizer.json")

    print(f"\n[merge] ✓ {out}")
    print(f"        训练 {len(train):,} / 验证 {len(val):,} token")
    print(f"        分片 {len(tr)} 训练 + {len(va)} 验证")
    n = 20.46e6
    print(f"\n  20.46M 模型 Chinchilla = {n*20/1e6:,.0f}M token"
          f" = {n*20/len(train):.1f} epoch")
    for m in (50e6, 100e6, 200e6):
        print(f"  {m/1e6:5.0f}M 模型 Chinchilla = {m*20/1e6:,.0f}M"
              f" = {m*20/len(train):.1f} epoch of corpus4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
