"""把要上传到阿里云的文件整理到 `alyyun/`。

**为什么要单独整理而不是直接压项目根目录**：

1. 项目根有 2 GB 的 `runs/_ref/`（GPT-2 对照模型）—— **云端训练用不上**，
   传上去纯浪费带宽和 OSS 存储。
2. `corpora/python-packages`（164 MB）只有**跑 cb-2 评测**才需要，
   训练本身不用 —— 分开放，让用户按需选。
3. 根目录还有 `.venv`、`__pycache__`、正在写入的检查点，
   直接打包会把**不一致的中间状态**带上去。

产出结构::

    alyyun/
      README.md              上传说明 + 云上第一步命令
      core/                  ① 必需：代码
        codemind/ tools/ tests/ pyproject.toml
      data/                  ② 必需：训练数据
        corpus2/ chat/
      ckpt/                  ③ 可选：起点检查点
      docs/                  ④ 参考：文档
      eval/                  ⑤ 可选：cb-2 的源码包（跑执行评测才需要）

用法::

    python tools/make_upload_dir.py            # 默认不含 eval/
    python tools/make_upload_dir.py --with-eval
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "alyyun"

CORE = ["codemind", "tools", "tests"]
DATA = ["data/corpus2", "data/chat"]
DOCS = ["AGENTS.md", "REPORT.md", "CLOUD_STEPS.md", "CLOUD_TRAIN.md"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}
SKIP_SUFFIX = {".pyc", ".pyo"}


def copy_tree(src: Path, dst: Path) -> tuple[int, int]:
    """拷贝目录，跳过缓存。返回 (文件数, 字节数)。"""
    n = b = 0
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        if any(part in SKIP_DIRS for part in f.parts):
            continue
        if f.suffix in SKIP_SUFFIX:
            continue
        rel = f.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
        n += 1
        b += f.stat().st_size
    return n, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-eval", action="store_true",
                    help="连 cb-2 需要的 python-packages 一起复制（约 164 MB）")
    ap.add_argument("--with-ckpt", action="store_true", default=True)
    args = ap.parse_args()

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    report: list[tuple[str, int, int]] = []

    # ① 核心代码
    for rel in CORE:
        src = ROOT / rel
        if src.is_dir():
            n, b = copy_tree(src, DST / "core" / rel)
            report.append((f"core/{rel}", n, b))

    # pyproject / requirements
    for rel in ("pyproject.toml", "requirements.txt"):
        p = ROOT / rel
        if p.is_file():
            (DST / "core").mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, DST / "core" / rel)
            report.append((f"core/{rel}", 1, p.stat().st_size))

    # ② 数据
    for rel in DATA:
        src = ROOT / rel
        if src.is_dir():
            n, b = copy_tree(src, DST / rel)
            report.append((rel, n, b))

    # ③ 检查点（只拿冻结的那个，不碰正在写的）
    if args.with_ckpt:
        ck = ROOT / "runs" / "_verify" / "ckpt-step13428.pt"
        if ck.is_file():
            (DST / "ckpt").mkdir(parents=True, exist_ok=True)
            shutil.copy2(ck, DST / "ckpt" / ck.name)
            report.append(("ckpt/" + ck.name, 1, ck.stat().st_size))

    # ④ 文档
    (DST / "docs").mkdir(parents=True, exist_ok=True)
    for rel in DOCS:
        p = ROOT / rel
        if p.is_file():
            shutil.copy2(p, DST / "docs" / p.name)
            report.append(("docs/" + p.name, 1, p.stat().st_size))

    # ⑤ 可选：cb-2 的源码包
    if args.with_eval:
        pk = ROOT / "corpora" / "python-packages"
        if pk.is_dir():
            n, b = copy_tree(pk, DST / "eval" / "python-packages")
            report.append(("eval/python-packages", n, b))

    total_n = sum(r[1] for r in report)
    total_b = sum(r[2] for r in report)
    print(f"[alyyun] 已整理到 {DST}")
    print(f"{'目录':40}{'文件':>8}{'大小':>12}")
    for name, n, b in report:
        print(f"  {name:38}{n:>8}{b/1e6:>10.1f} MB")
    print(f"  {'合计':38}{total_n:>8}{total_b/1e6:>10.1f} MB")

    (DST / "README.md").write_text(f"""# 上传到阿里云 PAI-DLC 的文件

生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}　共 {total_n} 文件 / {total_b/1e6:.1f} MB

## 目录说明

| 目录 | 必需 | 说明 |
|---|---|---|
| `core/` | ✅ | 代码（codemind / tools / tests） |
| `data/corpus2/` | ✅ | 代码语料 68.6M token |
| `data/chat/` | 可选 | 对话语料 31.6M token（聊天实验用） |
| `ckpt/` | 可选 | 11M 起点检查点（从它继续训，能省时间） |
| `docs/` | 参考 | 项目约定、报告、上云步骤 |
| `eval/` | 可选 | cb-2 执行评测要用的纯 Python 包 |

## ⚠️ 上传时必须注意

**分片是 `uint16` 裸数组，不是文本。**
用 FTP 的 **ASCII 模式**、或任何"按行处理"的同步工具，都会**静默毁掉数据**，
而且**训练不会报错**——只是 loss 降不下去。

**用二进制模式（SCP / rsync / OSS 控制台上传均可）。**

## 云上第一步（照抄）

```bash
# 1. 校验数据完整性（必须 55/55 之类全过）
python tools/pack_for_cloud.py --verify <包名>.tar.gz
#    若没用 tar 包，直接比对大小：
ls -l data/corpus2/*.bin      # 应为 40/40/40/17 MB

# 2. 确认 GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#    必须是 True

# 3. **先量 MFU，再决定训什么**（约 10 分钟，不可跳过）
python -m codemind bench --presets tiny small --device cuda

# 4. 用 100 步量真实吞吐
python -m codemind train --run runs/small --data data/corpus2 \\
    --preset small --steps 100 --threads 8 --amp bf16 --device cuda
```

## 为什么第 3 步不能跳

本项目已实测：**同类外推会乐观 2.4~3.2 倍**
（MIGRATION.md 预估 tiny 10 小时，实测 32 小时）。

| 实测 MFU | 决策 |
|---|---|
| ≥ 25% | 按计划训 `small`；余量大可考虑 medium |
| 15~25% | 只做 `small`，不碰 medium |
| **< 15%** | ⛔ **停下查数据管线**（本机曾因一个 bug 丢 36% 吞吐） |

详见 `docs/CLOUD_STEPS.md`。
""", encoding="utf-8")
    print(f"\n[alyyun] README 已写：{DST/'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
