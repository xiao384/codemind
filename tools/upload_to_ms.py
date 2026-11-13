"""把代码与语料上传到 ModelScope 数据集仓库。

## 为什么传上去

训练环境是**会话型**的：到点被杀、空闲休眠、临时盘会丢。
把代码和语料放到 ModelScope 仓库，**环境重建后能内网快速拉齐**，
只需要保住检查点即可 —— 这是"可断点续训"的前提。

## 传什么 / 不传什么

| 内容 | 大小 | 传？ | 原因 |
|---|---|---|---|
| `codemind/` `tools/` `tests/` | ~1 MB | ✅ | 代码 |
| `data/corpus2/` | 138 MB | ✅ | **不在 ModelScope 上，必须传** |
| firefly / alpaca 指令语料 | 1.2 GB | ❌ | **平台内直接拉，不必传** |
| `runs/` `corpora/` `*.pt` | GB 级 | ❌ | 训练产物，不该进仓库 |

用法::

    python tools/upload_to_ms.py --repo xiaoxiao384/codemind-corpus --dry-run
    python tools/upload_to_ms.py --repo xiaoxiao384/codemind-corpus
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 上传的目录（相对项目根）
UPLOAD = ["codemind", "tools", "tests", "data/corpus2"]
#: 上传的单文件
UPLOAD_FILES = ["pyproject.toml", "AGENTS.md", "REPORT.md",
                "CLOUD_STEPS.md", "CLOUD_TRAIN.md"]

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git"}
SKIP_SUFFIX = {".pyc", ".pyo", ".part"}


def token() -> str:
    t = os.environ.get("MODELSCOPE_API_TOKEN", "").strip()
    if t:
        return t
    f = ROOT / ".secrets" / "ms_token.txt"
    if f.is_file():
        return f.read_text(encoding="ascii").strip()
    sys.exit("✗ 找不到令牌：设 MODELSCOPE_API_TOKEN 或放 .secrets/ms_token.txt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="如 xiaoxiao384/codemind-corpus")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # SDK 的配置目录默认在 C:\Users\...，被沙箱拒绝 —— 必须指到工作区内
    os.environ.setdefault("MODELSCOPE_HOME", str(ROOT / ".secrets" / "ms_home"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / ".secrets" / "ms_cache"))
    os.environ["MODELSCOPE_API_TOKEN"] = token()
    for k in ("MODELSCOPE_HOME", "MODELSCOPE_CACHE"):
        Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

    import warnings
    warnings.filterwarnings("ignore")
    from modelscope.hub.api import HubApi

    files: list[Path] = []
    for rel in UPLOAD:
        p = ROOT / rel
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not any(x in SKIP_DIRS for x in f.parts) \
                        and f.suffix not in SKIP_SUFFIX:
                    files.append(f)
        elif p.is_file():
            files.append(p)
    for rel in UPLOAD_FILES:
        if (ROOT / rel).is_file():
            files.append(ROOT / rel)

    total = sum(f.stat().st_size for f in files)
    print(f"[upload] 目标仓库 {args.repo}")
    print(f"[upload] {len(files)} 个文件 / {total/1e6:.1f} MB")
    big = sorted(files, key=lambda f: -f.stat().st_size)[:8]
    for f in big:
        print(f"    {f.stat().st_size/1e6:9.1f} MB  {f.relative_to(ROOT).as_posix()}")

    if args.dry_run:
        print("[upload] --dry-run，未上传")
        return 0

    api = HubApi()
    name = args.repo.split("/")[-1]
    try:
        api.create_dataset(args.repo, name=name, license="Apache License 2.0")
        print(f"[upload] ✓ 数据集仓库已创建（或已存在）：{args.repo}")
    except Exception as e:
        print(f"[upload] 创建仓库返回：{type(e).__name__}: {str(e)[:120]}")
        print("[upload] 若提示已存在，属正常，继续上传")

    for i, f in enumerate(files, 1):
        rel = f.relative_to(ROOT).as_posix()
        try:
            api.upload_file(
                repo_id=args.repo,
                path_or_fileobj=str(f),
                path_in_repo=rel,
                repo_type="dataset",
                commit_message=f"add {rel}",
            )
            if i % 10 == 0 or f.stat().st_size > 5_000_000:
                print(f"  [{i}/{len(files)}] {rel} ({f.stat().st_size/1e6:.1f} MB)", flush=True)
        except Exception as e:
            print(f"  ✗ {rel}: {type(e).__name__}: {str(e)[:100]}")

    print(f"[upload] 完成。云端拉取：")
    print(f"    python tools/fetch_dataset.py --get {args.repo} --list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
