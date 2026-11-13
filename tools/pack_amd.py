"""打包 AMD 云端要用的全部内容。

## 为什么带 `corpus3` + `chat_ff` 而不是直接带 `mix3`

`mix3` 可以由这两者用 `make_mixed_corpus.py` 现场构建（约 1 分钟）。
**带原始语料更灵活** —— 想换配比（70:30 / 50:50 / 30:70）随时能重建，
而带一个固定的 mix 就锁死了配比。

## 带什么 / 不带什么

| 内容 | 带? | 原因 |
|---|---|---|
| `codemind/` `tools/` `tests/` | ✅ | 代码 |
| `data/corpus3/` | ✅ | 代码语料 226.8M token |
| `data/chat_ff/` | ✅ | 对话语料 226.3M token |
| `data/mix3/` | ❌ | 可由上两者重建 |
| `runs/` 检查点 | ❌ | 上百 GB，云端从头训 |
| `corpora/` 原始包 | ❌ | 已编码成 token，原始包无用 |
"""
from __future__ import annotations

import shutil
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(r"E:\AI")
STAGE = ROOT / "pkg_amd"
OUT = ROOT / "codemind-amd.zip"

DIRS = ["codemind", "tools", "tests"]
FILES = ["pyproject.toml", "AGENTS.md", "REPORT.md", "DATA_REQUIREMENTS.md"]
DATA = ["data/corpus4", "data/chat_full"]
SKIP = {"__pycache__", ".pytest_cache", ".git"}
SKIP_EXT = {".pyc", ".pyo", ".part"}

README = """# CodeMind — AMD GPU 云端训练包

## 里面有什么

```
codemind/  tools/  tests/  pyproject.toml    代码（纯 PyTorch，不加载预训练权重）
data/corpus4/    代码语料 2,195M token（275 分片）
data/chat_full/  对话语料 819M token（104 分片）
docs/            项目约定、报告、数据需求
```

## 环境自检（AMD ROCm）

```bash
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 镜像预装 torch 2.12.0+ROCm7.2.3，应直接 True
```

## ⚠️ 第一件事：量真实吞吐（不要信外推）

```bash
python -u -m codemind train --run runs/probe --data data/corpus4 \\
    --preset tiny --layers 12 --embd 384 --heads 6 --kv-heads 2 --ctx 512 \\
    --steps 100 --batch-size 32 --threads 8 --amp bf16 --device cuda 2>&1 | tail -8
```

记下最后一行那个 `tok/s`。**所有时长估算都基于它，而它现在是外推的。**

## 为什么先训这个

我们已有的三次测量都说「参数不换能力」（11M / 78M / GPT-2 124M、355M），
**但最大只测到 78M**。现在有 192 GB 显存，可以把曲线拉到 1B —— 跨度 50 倍。

| 规模 | 步数（batch32×ctx512） | 预计时长 |
|---|---|---|
| 20M | 27,400 | 约 0.5 h |
| 50M | 61,000 | 约 1.3 h |
| 100M | 122,000 | 约 2.5 h |
| 300M | 366,000 | 约 7.5 h |
| 1B | 1,220,000 | 约 25 h |

⚠️ **这些都用「数据量 × epoch」反推，必须先用探针校正。**

## 构建混合语料（要训对话能力时）

```bash
python tools/make_mixed_corpus.py --code data/corpus4 --lang data/chat_full \\
    --out data/mix3 --ratio 70:30
```

**70:30 是经过验证的配比**：78M 用它在代码零损失的前提下获得了对话能力；
而 11M/20M 从零直接学混合语料都失败了 —— 所以要先纯代码预训练，再混合微调。

## 数据校验（分片是 uint16 裸数组，传坏了训练不报错）

```bash
python -c "
import json;from pathlib import Path
for d in ('data/corpus4','data/chat_full'):
    p=Path(d); m=json.loads(sorted(p.glob('*.meta.json'))[0].read_text(encoding='utf-8'))
    fs=m['train_files']+m['val_files']
    miss=[f for f in fs if not (p/f).exists()]
    print(f'{d}: {len(fs)} 分片, {sum((p/f).stat().st_size//2 for f in fs):,} token, 缺失 {miss or 0}')
"
```

## ⚠️ 别忘了

**训练中的 `latest.pt` 每 500 步被覆盖写**，加载时撞上写入会读到半写状态的文件。
要测就用 `best.pt`，或者先把文件快照一份。
"""


def main() -> int:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    total = n = 0
    for rel in DIRS + DATA:
        src = ROOT / rel
        if not src.is_dir():
            print(f"  ✗ 缺目录 {src}")
            return 1
        t0 = time.time()
        c = b = 0
        for f in sorted(src.rglob("*")):
            if not f.is_file() or any(x in SKIP for x in f.parts) or f.suffix in SKIP_EXT:
                continue
            out = STAGE / f.relative_to(ROOT)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
            c += 1
            b += f.stat().st_size
        total += b
        n += c
        print(f"  ✓ {rel:16} {c:>5} 个 {b/1e6:>8.1f} MB  ({time.time()-t0:.0f}s)")

    (STAGE / "docs").mkdir(parents=True, exist_ok=True)
    for rel in FILES:
        f = ROOT / rel
        if f.is_file():
            shutil.copy2(f, STAGE / "docs" / f.name)
            total += f.stat().st_size
    (STAGE / "README.md").write_text(README, encoding="utf-8")
    print(f"\n  合计 {n} 个文件 / {total/1e6:.1f} MB")

    if OUT.exists():
        OUT.unlink()
    t0 = time.time()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(STAGE).as_posix())
    print(f"  ✓ {OUT.name}  {OUT.stat().st_size/1e6:.1f} MB  "
          f"({time.time()-t0:.0f}s, 压缩率 {OUT.stat().st_size/total*100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
