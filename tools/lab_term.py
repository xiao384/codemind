"""通过 JupyterLab 终端执行云端命令（**目前唯一可靠的方式**）。

## 为什么这个能work，之前那些不行

| 界面 | 结果 |
|---|---|
| ModelScope Web IDE（魔改 VS Code） | ✗ 终端标签点不动，打字进筛选框 |
| **阿里云 DSW JupyterLab** | ✓ **标准 xterm.js，`.xterm-helper-textarea` 可直接输入** |

进入方式：先登录 `account.aliyun.com`，再开
`https://dsw-gateway-cn-hangzhou.data.aliyun.com/dsw-<实例ID>/lab`，
点 Launcher 里的 `Terminal` 卡片（用 JS `click()` 最稳）。

用法::

    python tools/lab_term.py --script tools/cloud_check.txt
    python tools/lab_term.py --cmd "tail -3 runs/s1b.log"
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"E:\AI")
PROFILE = ROOT / ".browser_profile"
SHOT = ROOT / "runs/_verify"
LAB = "https://dsw-gateway-cn-hangzhou.data.aliyun.com/dsw-2346511/lab"


def open_terminal(pg, lab_url: str) -> None:
    pg.goto(lab_url, wait_until="domcontentloaded", timeout=90000)
    pg.wait_for_timeout(15000)
    if pg.locator(".xterm-helper-textarea").count():
        print("  已有终端标签", flush=True)
        return
    r = pg.evaluate("""() => {
      const c = document.querySelector('.jp-LauncherCard[title="Start a new terminal session"]');
      if (!c) return 'no card';
      c.click(); return 'clicked';
    }""")
    print("  开终端:", r, flush=True)
    pg.wait_for_timeout(9000)


def run_cmd(pg, cmd: str, wait: float) -> None:
    n = pg.locator(".xterm-helper-textarea").count()
    if not n:
        raise RuntimeError("找不到 xterm textarea")
    ta = pg.locator(".xterm-helper-textarea").last
    ta.click()
    pg.wait_for_timeout(500)
    pg.keyboard.insert_text(cmd)
    pg.wait_for_timeout(400)
    pg.keyboard.press("Enter")
    print(f"    ▸ {cmd[:110]}", flush=True)
    pg.wait_for_timeout(int(wait * 1000))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", action="append", default=[])
    ap.add_argument("--script")
    ap.add_argument("--wait", type=float, default=8)
    ap.add_argument("--url", default=LAB)
    args = ap.parse_args()

    cmds = list(args.cmd)
    if args.script:
        cmds += [l.strip() for l in Path(args.script).read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]

    SHOT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE), headless=False, channel="msedge",
            viewport={"width": 1600, "height": 1000}, locale="zh-CN",
            args=["--disable-blink-features=AutomationControlled"])
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            open_terminal(pg, args.url)
            for i, c in enumerate(cmds):
                run_cmd(pg, c, args.wait)
            ts = datetime.now().strftime("%H%M%S")
            f = SHOT / f"term-{ts}.png"
            pg.screenshot(path=str(f))
            print("  截图:", f, flush=True)
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
