"""把 78M 与 20M 模型打成一个 7z 包。

## 为什么要先"快照"

`runs/big20mix/latest.pt` **每 500 步就被训练覆盖一次**（约 2.4 小时一轮）。
直接从原路径打包，可能压到"半写状态"的文件 —— **加载时会报张量不匹配或直接崩**。

所以先 `shutil.copy2` 到暂存目录（拷贝是原子的读，不受后续写入影响），再压。

## 包结构

    codemind-models/
      README.md
      78M-pure-code/   latest.pt + tokenizer.json
      78M-mixed/       latest.pt + tokenizer.json
      20M-mixed/       latest.pt  best.pt + tokenizer.json
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(r"E:\AI")
STAGE = ROOT / "pkg_models"
OUT = ROOT / "codemind-models.7z"
SEVENZ = r"C:\Program Files\7-Zip\7z.exe"

ITEMS = [
    ("78M-pure-code", ROOT / "runs/_ref78/latest.pt", ROOT / "runs/_ref78/tokenizer.json"),
    ("78M-mixed", ROOT / "runs/_ref78mix/latest.pt", ROOT / "runs/_ref78mix/tokenizer.json"),
    ("20M-mixed", ROOT / "runs/big20mix/latest.pt", ROOT / "runs/big20mix/tokenizer.json"),
]

README = """# CodeMind 模型包

从零实现的代码 + 对话模型（纯 PyTorch，**不加载任何外部预训练权重**）。

## 里面有什么

### 78M 系列（云端 A10 训练）

| 目录 | 参数 | 能力 | 实测 |
|---|---|---|---|
| `78M-pure-code/` | 78.66M | 纯代码 | `edit_sim` 余量 **+7.23pp**；`rep8` 32.21% |
| `78M-mixed/` ⭐ | 78.66M | **代码 + 对话** | `edit_sim` 余量 **+8.55pp**；对话可出**编号列表** |

**`78M-mixed` 是本项目最好的模型** —— 用「70% 代码 + 30% 对话」**交错混合**语料训成，
代码能力没有损失，同时获得了对话能力（纯对话微调会把代码能力冲掉 13.42pp）。

### 20M 系列（本机 CPU 训练）

| 目录 | 参数 | 说明 |
|---|---|---|
| `20M-mixed/` | 20.46M | 12 层 / d=384 / ctx=512，混合语料训练中 |

- `latest.pt` —— 含优化器状态（**续训必须用这个**）
- `best.pt` —— 只有权重（验证损失最低那一次）

⚠️ **20M 模型仍在训练中**，这个包是某个时间点的快照。

## 怎么用

```bash
# 需要项目代码（codemind/ 包）
python -m codemind chat --run <模型目录> --checkpoint latest -m "你好"
```

或直接用 Python API：

```python
import sys; sys.path.insert(0, "<项目根>")
from pathlib import Path
from codemind.generate import load_runner
from codemind.model import SamplingParams

g = load_runner(Path("78M-mixed/latest.pt"), device="cpu",
                tokenizer_path=Path("78M-mixed/tokenizer.json"))
reply, _ = g.chat("保持健康的三个提示。", max_new_tokens=110,
                  params=SamplingParams(temperature=0.7, top_k=40,
                                        repetition_penalty=1.15, seed=7))
print(reply)
```

## ⚠️ 三点必须知道

**1. 两个 78M 是同一份权重演化的两个状态，不是两个独立模型。**
先训纯代码，再用交错混合语料续训得到融合版。

**2. `latest.pt` 含优化器状态，比 `best.pt` 大 3 倍。**
续训要用 `latest.pt`；只做推理可以用 `best.pt`（更小）。

**3. 词表是 4096（byte-level BPE）。**
换分词器会让这些检查点全部作废。每个目录都附了对应的 `tokenizer.json`。

## 实测能力一览

| 指标 | 11M 纯代码 | 78M 纯代码 | **78M 混合** |
|---|---|---|---|
| `edit_sim` 余量 | +7.42pp | +7.23pp | **+8.55pp** |
| `rep8`（低好） | 51.99% | 32.21% | **28.26%** |
| 主动收尾率 | 0.38% | 13.50% | 9.75% |
| cb-2 执行通过 | 0/119 | 0/119 | 未测 |

**「参数不换能力」已被三次独立测量确认**（`edit_sim` / cb-2 / 外部对标 GPT-2 124M、355M）。
真正的瓶颈是**数据量**，不是参数规模。
"""


def main() -> int:
    if not Path(SEVENZ).is_file():
        print(f"✗ 找不到 7-Zip: {SEVENZ}")
        return 1
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    total = 0
    for name, pt, tk in ITEMS:
        if not pt.is_file():
            print(f"  ✗ 缺 {pt}")
            return 1
        d = STAGE / name
        d.mkdir(parents=True)
        t0 = time.time()
        shutil.copy2(pt, d / pt.name)
        n = (d / pt.name).stat().st_size
        total += n
        print(f"  ✓ {name}/{pt.name}  {n/1e6:8.1f} MB  ({time.time()-t0:.0f}s)")
        if tk.is_file():
            shutil.copy2(tk, d / "tokenizer.json")
            total += tk.stat().st_size
        else:
            print(f"    ⚠ 缺 tokenizer: {tk}")

    # 20M 额外带 best.pt（只有权重，小 3 倍）
    b = ROOT / "runs/big20mix/best.pt"
    if b.is_file():
        shutil.copy2(b, STAGE / "20M-mixed" / "best.pt")
        total += b.stat().st_size
        print(f"  ✓ 20M-mixed/best.pt  {b.stat().st_size/1e6:8.1f} MB")

    (STAGE / "README.md").write_text(README, encoding="utf-8")
    print(f"\n  暂存合计 {total/1e6:.0f} MB")

    if OUT.exists():
        OUT.unlink()
    t0 = time.time()
    r = subprocess.run([SEVENZ, "a", "-mx=3", "-bso0", "-bsp0",
                        str(OUT), str(STAGE / "*")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("✗ 压缩失败:", r.stderr[:400])
        return 1
    print(f"  ✓ {OUT.name}  {OUT.stat().st_size/1e6:.1f} MB  "
          f"({time.time()-t0:.0f}s, 压缩率 {OUT.stat().st_size/total*100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
