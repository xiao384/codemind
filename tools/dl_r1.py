"""从 ModelScope 下载 DeepSeek-R1 蒸馏模型。

## 为什么选这几个规模

| 模型 | 参数 | 下载体积(fp16) | 和本项目的关系 |
|---|---|---|---|
| `DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | **约 3.5 GB** | **和我们要训的 1B 最接近** |
| `DeepSeek-R1-Distill-Qwen-7B` | 7B | 约 15 GB | 中档参考 |
| `DeepSeek-R1-Distill-Llama-8B` | 8B | 约 16 GB | Llama 底座 |
| `DeepSeek-R1-Distill-Qwen-14B` | 14B | 约 28 GB | 高档 |
| `DeepSeek-R1-Distill-Qwen-32B` | 32B | 约 64 GB | 本机跑不动 |

## 关键：缓存目录必须指到 E:\\AI 里

DSH 沙箱只允许写 `E:\\AI` 之内，默认缓存 `C:\\Users\\...\\.cache` 会报 WinError 5。
所以设 `MODELSCOPE_CACHE=E:\\AI\\models`。

用法::

    python tools/dl_r1.py --model 1.5B
    python tools/dl_r1.py --model 7B
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

ROOT = Path(r"E:\AI")
CACHE = ROOT / "models"
# ⚠️ 三个 env 都要设：modelscope 会往 `~/.modelscope` 建 SDK 目录，
# 沙箱禁止写 C:\Users\... 实测报 `[E1022] Failed to create SDK directories: WinError 5`。
os.environ["MODELSCOPE_CACHE"] = str(CACHE)
os.environ["MODELSCOPE_HOME"] = str(ROOT / ".modelscope")
os.environ["MODELSCOPE_CACHE_DIR"] = str(CACHE)
os.environ["HOME"] = str(ROOT / ".home")
for _d in (CACHE, ROOT / ".modelscope", ROOT / ".home"):
    _d.mkdir(parents=True, exist_ok=True)

MODELS = {
    "1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5B", choices=list(MODELS))
    args = ap.parse_args()

    mid = MODELS[args.model]
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"模型  : {mid}")
    print(f"缓存  : {CACHE}")
    print(f"开始下载 {time.strftime('%H:%M:%S')}", flush=True)

    from modelscope import snapshot_download
    t0 = time.time()
    p = snapshot_download(mid, cache_dir=str(CACHE))
    dt = time.time() - t0

    total = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
    print(f"\n✓ 完成 {time.strftime('%H:%M:%S')}  用时 {dt/60:.1f} 分钟")
    print(f"  目录: {p}")
    print(f"  体积: {total/1e9:.2f} GB")
    print(f"  速度: {total/1e6/max(dt,1):.1f} MB/s")
    print("\n  文件:")
    for f in sorted(Path(p).rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(p)}  {f.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
