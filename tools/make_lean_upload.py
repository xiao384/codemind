"""整理**本次云训练真正需要**的文件到 `ailyun/`。

## 与 `make_upload_dir.py` 的区别

那个是"全量包"（334 MB），含 11M 检查点与对话语料 —— **本次训练用不上**：

| 内容 | 全量包 | 本包 | 为什么 |
|---|---|---|---|
| 代码 | ✅ | ✅ | 必需 |
| `data/corpus2/`（68.6M token） | ✅ | ✅ | **训练数据，必需** |
| 11M 检查点（132 MB） | ✅ | ❌ | **从头训 `small`，不需要起点** |
| `data/chat/`（63 MB） | ✅ | ❌ | 聊天实验才用 |
| 文档 | ✅ | ✅ | 体积极小，带着方便 |

**334 MB → 约 140 MB。**

用法::

    python tools/make_lean_upload.py                 # 精简包（默认）
    python tools/make_lean_upload.py --with-chat     # 连对话语料一起
    python tools/make_lean_upload.py --with-ckpt     # 连 11M 检查点一起
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "ailyun"

CORE = ["codemind", "tools", "tests", "pyproject.toml"]
DOCS = ["AGENTS.md", "REPORT.md", "CLOUD_STEPS.md", "CLOUD_TRAIN.md"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}
SKIP_SUFFIX = {".pyc", ".pyo"}


def copy_tree(src: Path, dst: Path) -> tuple[int, int]:
    n = b = 0
    for f in sorted(src.rglob("*")):
        if not f.is_file() or any(p in SKIP_DIRS for p in f.parts):
            continue
        if f.suffix in SKIP_SUFFIX:
            continue
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        n += 1
        b += f.stat().st_size
    return n, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-chat", action="store_true")
    ap.add_argument("--with-ckpt", action="store_true")
    args = ap.parse_args()

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    report: list[tuple[str, int, int]] = []

    # ⚠️ 代码**必须放在根目录**，不能套一层 `core/`。
    # 实测踩过：`python -m codemind` 要求 `codemind/` 在**当前工作目录**下，
    # 套一层就会报 `No module named codemind` —— 而数据在 `data/`，
    # 所以又不能靠 `cd core` 绕开（那样 `--data data/corpus2` 就找不到）。
    # **这两条约束同时成立，只有"代码与数据同在根目录"一个解。**
    for rel in CORE:
        src = ROOT / rel
        if src.is_file():
            shutil.copy2(src, DST / rel)
            report.append((rel, 1, src.stat().st_size))
        elif src.is_dir():
            n, b = copy_tree(src, DST / rel)
            report.append((rel, n, b))

    src = ROOT / "data" / "corpus2"
    if src.is_dir():
        n, b = copy_tree(src, DST / "data" / "corpus2")
        report.append(("data/corpus2", n, b))

    if args.with_chat:
        src = ROOT / "data" / "chat"
        if src.is_dir():
            n, b = copy_tree(src, DST / "data" / "chat")
            report.append(("data/chat", n, b))

    if args.with_ckpt:
        ck = ROOT / "runs" / "_verify" / "ckpt-step13428.pt"
        if ck.is_file():
            (DST / "ckpt").mkdir(parents=True, exist_ok=True)
            shutil.copy2(ck, DST / "ckpt" / ck.name)
            report.append(("ckpt/" + ck.name, 1, ck.stat().st_size))

    (DST / "docs").mkdir(parents=True, exist_ok=True)
    for rel in DOCS:
        p = ROOT / rel
        if p.is_file():
            shutil.copy2(p, DST / "docs" / p.name)
            report.append(("docs/" + p.name, 1, p.stat().st_size))

    tn = sum(r[1] for r in report)
    tb = sum(r[2] for r in report)
    print(f"[ailyun] -> {DST}")
    for name, n, b in report:
        print(f"  {name:34}{n:>6} 个{b/1e6:>10.1f} MB")
    print(f"  {'合计':34}{tn:>6} 个{tb/1e6:>10.1f} MB")

    (DST / "README.md").write_text(f"""# 云训练上传包（精简）

生成：{time.strftime('%Y-%m-%d %H:%M:%S')}　共 {tn} 个文件 / **{tb/1e6:.1f} MB**

## 里面有什么

```
core/          代码（codemind / tools / tests / pyproject.toml）
data/corpus2/  ✅ 训练数据 68.6M token（这是包的主体）
docs/          项目约定、报告、上云步骤
{"data/chat/     对话语料（本次可选）" if args.with_chat else ""}
{"ckpt/          11M 起点检查点（本次可选）" if args.with_ckpt else ""}
```

## ⚠️ 上传必须用**二进制模式**

`data/corpus2/*.bin` 是 **uint16 裸数组，不是文本**。
用 FTP ASCII 模式或"按行处理"的同步工具会**静默毁掉数据**，
而且**训练不会报错** —— 只是 loss 降不下去。

上传后先核对大小：

```
corpus2.00000.bin   40,006,066
corpus2.00001.bin   40,004,130
corpus2.00002.bin   40,000,358
corpus2.00003.bin   17,250,932
```

**对不上就重传。**

## 云上第一步

```bash
# 0. 确认 GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 1. 起训练（默认 LR 已修正为 1e-3、warmup 3%，不需要额外参数）
nohup python -m codemind train --run runs/small_v2 --data data/corpus2 \\
    --preset small --steps 20000 --batch-size 32 --threads 8 \\
    --amp bf16 --device cuda --ckpt-interval 500 \\
    > train_v2.log 2>&1 &
echo "PID=$!"

# 2. 三分钟后看 loss
tail -20 train_v2.log
```

## 判据（**这次的关键**）

上次在同一位置**发散**过（LR 3e-3 太高）。新默认 1e-3。

```
step    1   loss ≈ 8.4     ← ln(4096)，随机基线
step  100   loss < 5
step  500   loss < 3.5
step 1000   loss < 2.9     ← 上次这里是 2.8683（最低点）
step 2000   loss < 2.7     ← ⚠️ 上次这里反弹到 3.3370
step 5000   loss < 2.4
```

**盯 step 1000→3000 这一段**：
- ✅ **继续下降** → 修好了
- ❌ **触底反弹** → 停，告诉我，再降 LR

## 这次**没有**包含什么（以及为什么）

| 排除 | 大小 | 原因 |
|---|---|---|
| `runs/_ref/`（GPT-2 对照） | 2 GB | 训练用不上 |
| `ckpt-step13428.pt` | 132 MB | 从头训 `small`，不需要起点 |
| `data/chat/` | 63 MB | 聊天实验才用 |
| `corpora/python-packages` | 164 MB | 只有跑 cb-2 评测才需要 |

需要用的时候加 `--with-chat` / `--with-ckpt` 重新生成。
""", encoding="utf-8")
    print(f"\n[ailyun] README 已写")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
