"""等训练跑到指定步数，然后自动做一次**高功率**的关键检查点评估。

为什么需要它：
  * step 4188（1.14 epoch）是本次训练第一个有意义的判据点，
    而 `cb1_watch.py` 只用 n=400（分辨力约 2.3pp）；关键点值得用 n=1000（约 1.5pp）。
  * 会话可能在此之前结束。**判据点的数据一旦错过去就补不回来**
    （latest.pt 会被后续检查点覆盖）。

做法：轮询训练 stdout 日志里的步数，到了就冻结检查点、跑 `key_eval.py`。
不反复 load 检查点（132 MB），只在确认到达后才读一次校验。

用法：
    python tools/wait_key_eval.py --target 4188 --samples 1000
    python tools/wait_key_eval.py --target 13428 --samples 2000 --ckpt best
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = ROOT / "runs" / "_verify" / "train-corpus2.out.log"
VERIFY = ROOT / "runs" / "_verify"


def current_step() -> int:
    if not LOG.exists():
        return 0
    txt = LOG.read_text(encoding="utf-8", errors="replace")
    hits = re.findall(r"^\[train\] step (\d+)/\d+", txt, re.M)
    return int(hits[-1]) if hits else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, required=True, help="目标步数")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ckpt", default="latest", choices=["latest", "best"])
    ap.add_argument("--poll", type=int, default=120, help="轮询间隔（秒）")
    args = ap.parse_args()

    tag = f"step{args.target}"
    marker = VERIFY / f"key-eval-{tag}.txt"
    if marker.exists():
        print(f"{marker.name} 已存在，说明该点已评估过，退出。")
        return 0

    print(f"[wait_key_eval] 目标 step {args.target}，样本量 {args.samples}，"
          f"每 {args.poll}s 轮询", flush=True)
    while True:
        s = current_step()
        if s >= args.target:
            print(f"[wait_key_eval] 训练已到 step {s}（>= {args.target}）", flush=True)
            break
        time.sleep(args.poll)

    # 等 latest.pt 落盘稳定（保存需要几秒）
    ckpt_src = ROOT / "runs" / "corpus2" / f"{args.ckpt}.pt"
    last = -1
    for _ in range(30):
        size = ckpt_src.stat().st_size if ckpt_src.exists() else 0
        if size and size == last:
            break
        last = size
        time.sleep(10)

    frozen = VERIFY / f"ckpt-{tag}.pt"
    if not frozen.exists():
        import shutil

        # ⚠️ 关键：训练每 500 步才落一次盘。日志里出现 "step 4188" 时，
        # latest.pt 很可能还是 **step 4000** 的检查点。
        # 无条件冻结就会把 4000 的检查点命名成 ckpt-step4188.pt 并当作 4188 评估
        # —— 数据会失真且看不出来。所以必须校验 iteration，不匹配就丢弃重等。
        for attempt in range(60):
            shutil.copy2(ckpt_src, frozen)
            try:
                import torch

                ck = torch.load(frozen, map_location="cpu", weights_only=False)
                it = int(ck.get("iteration") or 0)
            except Exception as exc:  # noqa: BLE001
                print(f"[wait_key_eval] 读取检查点失败：{exc}", flush=True)
                it = -1

            if abs(it - args.target) <= 10:
                print(f"[wait_key_eval] 已冻结 {frozen.name}"
                      f"（{frozen.stat().st_size/1e6:.0f} MB，iteration={it}，"
                      f"best_val={ck.get('best_val')}）", flush=True)
                break

            print(f"[wait_key_eval] latest.pt 仍是 step {it}（目标是 {args.target}），"
                  f"等下一次落盘…", flush=True)
            frozen.unlink(missing_ok=True)
            # 等 latest.pt 变化
            last_m = ckpt_src.stat().st_mtime
            for _ in range(120):
                time.sleep(15)
                if ckpt_src.stat().st_mtime != last_m:
                    break
        else:
            print(f"[wait_key_eval] 等了 60 轮仍没拿到 step {args.target} 附近的检查点，"
                  f"放弃以免写入错误命名的数据。", flush=True)
            return 1

    cmd = [str(PY), "tools/key_eval.py", "--steps", str(args.target),
           "--samples", str(args.samples), "--threads", str(args.threads)]
    print(f"[wait_key_eval] 开始评估：{' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    print(f"[wait_key_eval] 评估结束，退出码 {rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
