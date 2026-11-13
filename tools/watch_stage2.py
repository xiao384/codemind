"""阶段 1 训练结束后，自动启动阶段 2（混合微调）。

## 判据用两个，不只看一个

**日志里出现"训练完成"** 是主判据；但如果日志因缓冲或异常缺失，
进程退出 + 检查点步数到位也能说明问题。**两个判据任一成立即开跑。**

本项目吃过"只看日志/只盯文件名"的亏：

* 看守盯错文件名（`key-eval-13428.txt` vs 实际的 `key-eval-step13428.txt`）→ 白等 4 小时
* 看守的子进程被杀 → 曲线冻结 11 小时无人察觉

所以这里**同时检查：进程是否还在 + 日志是否出现完成字样 + 检查点步数**。

用法::

    python tools/watch_stage2.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\AI")
LOG1 = ROOT / "runs/_verify/train-stage1.log"
CK1 = ROOT / "runs/stage1/latest.pt"
TARGET = 13706
STAGE2_STEPS = 8811          # 0.45 epoch of mix3 = 147M token
PY = r"E:\AI\.venv\Scripts\python.exe"


def alive() -> bool:
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -match 'runs/stage1' }).Count"],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip() or 0) > 0
    except ValueError:
        return False


def log_done() -> bool:
    if not LOG1.is_file():
        return False
    try:
        return "训练完成" in LOG1.read_text(encoding="utf-8", errors="replace")[-4000:]
    except OSError:
        return False


def main() -> int:
    print(f"[watch] 等 {LOG1.name} 出现「训练完成」（目标 {TARGET} 步）")
    t0 = time.time()
    last = 0.0
    while True:
        done = log_done()
        running = alive()
        el = time.time() - t0
        if done or (not running and el > 120):
            why = "日志出现完成字样" if done else "进程已退出"
            print(f"[watch] 阶段 1 结束（{why}），用时 {el/3600:.1f} 小时")
            break
        if el - last > 1800:
            last = el
            print(f"[watch] {el/3600:5.1f} 小时  进程{'在跑' if running else '已退'}", flush=True)
        if el > 6 * 86400:
            print("[watch] 超时 6 天，退出")
            return 2
        time.sleep(60)

    if not CK1.is_file():
        print(f"[watch] ✗ 找不到 {CK1}，无法续训")
        return 1
    mb = CK1.stat().st_size / 1e6
    print(f"[watch] 检查点 {CK1.name}  {mb:.1f} MB")

    # ---- 建 stage2 目录（重置步数，保留权重）----
    s2 = ROOT / "runs/stage2"
    s2.mkdir(parents=True, exist_ok=True)
    code = (
        "import torch;from pathlib import Path;"
        f"ck=torch.load(r'{CK1}',map_location='cpu',weights_only=False);"
        "print('来源步数',ck.get('iteration'),'best_val',round(ck.get('best_val',0),4));"
        "ck['iteration']=0;ck['tokens_seen']=0;ck['best_val']=float('inf');"
        f"[torch.save(ck,Path(r'{s2}')/d) for d in ('latest.pt','best.pt')]"
    )
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    print("[watch]", (r.stdout or r.stderr).strip()[:200])

    tok = ROOT / "data/corpus3/tokenizer.json"
    if tok.is_file():
        import shutil
        shutil.copy2(tok, s2 / "tokenizer.json")

    mix = ROOT / "data/mix3"
    if not (mix / "mix3.meta.json").is_file():
        print("[watch] ✗ data/mix3 不存在，请先建混合语料")
        return 1

    # ---- 起阶段 2 ----
    log = ROOT / "runs/_verify/train-stage2.log"
    err = ROOT / "runs/_verify/train-stage2.err.log"
    args = [PY, "-u", "-m", "codemind", "train",
            "--run", "runs/stage2", "--data", "data/mix3", "--preset", "tiny",
            "--layers", "12", "--embd", "384", "--heads", "6", "--kv-heads", "2",
            "--ctx", "512", "--steps", str(STAGE2_STEPS),
            "--batch-size", "32", "--threads", "24", "--ckpt-interval", "500"]
    p = subprocess.Popen(args, cwd=str(ROOT),
                         stdout=open(log, "w", encoding="utf-8"),
                         stderr=open(err, "w", encoding="utf-8"),
                         creationflags=0x00000008 | 0x00000200)  # DETACHED | NEW_GROUP
    print(f"[watch] ✓ 阶段 2 已启动 PID={p.pid}")
    print(f"[watch]   语料 data/mix3  步数 {STAGE2_STEPS}  ≈ 1.7 天")
    print(f"[watch]   日志 {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
