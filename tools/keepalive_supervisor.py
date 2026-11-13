"""看护 keepalive.py 本身：挂了就自动重启。

## 为什么需要这一层

2026-11-07 实测：`keepalive.py` 跑到第 2 轮就**无声退出**（`keep.err.log` 是空的，
进程消失），导致 1.5 小时没有浏览器交互 —— 差点触发平台「1 小时无交互挂起实例」。

**单层循环不够，必须双层**：外层守着内层，内层挂了外层重新拉起。

用法::

    python tools/keepalive_supervisor.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"E:\AI")
PY = str(ROOT / ".venv/Scripts/python.exe")
OUT = ROOT / "runs/_verify/monitor"


def log(msg: str) -> None:
    line = f"[{datetime.now():%m%d-%H%M}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "supervisor.log").open("a", encoding="utf-8").write(line + "\n")


def main() -> int:
    log("=== 看护启动 ===")
    restarts = 0
    while True:
        t0 = time.time()
        try:
            r = subprocess.run(
                [PY, "-u", str(ROOT / "tools/keepalive.py"),
                 "--interval", "2100", "--rounds", "6"],
                cwd=str(ROOT), timeout=6 * 3600 + 1800)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        except Exception as e:
            code = f"{type(e).__name__}: {e}"
        dur = time.time() - t0
        restarts += 1
        log(f"keepalive 退出 code={code} 运行 {dur/60:.1f} 分钟，第 {restarts} 次重启")
        if dur < 60:
            time.sleep(30)          # 秒退 → 等 30 秒再起，避免疯狂重启
        else:
            time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
