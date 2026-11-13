"""通过命令面板新建终端并执行命令（比点【终端】标签可靠）。

## 为什么用命令面板

实测这个魔搭 Web IDE：
- 点【终端】标签坐标 → **无效**（面板仍停在【问题】页，打字进筛选框）
- `Ctrl+`` → 只是开关面板，不保证焦点
- **`Ctrl+Shift+P` → 输入「新建终端」→ 回车** → 新终端**自带焦点**，实测可行

关键：新建终端后**立刻**打字，中间不要点别的地方。
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"E:\AI")
PROFILE = ROOT / ".browser_profile"
SHOT = ROOT / "runs/_verify"
NOTEBOOK_URL = "https://www.modelscope.cn/my/mynotebook"


def shot(page, name):
    try:
        p = SHOT / f"run-{name}.png"
        page.screenshot(path=str(p))
        print(f"    图: {p.name}")
    except Exception:
        pass


def new_terminal(ide):
    """用命令面板开一个新终端（自带焦点）。"""
    ide.keyboard.press("Control+Shift+p")
    ide.wait_for_timeout(1800)
    ide.keyboard.insert_text("新建终端")
    ide.wait_for_timeout(1800)
    ide.keyboard.press("Enter")
    ide.wait_for_timeout(4000)
    print("    已新建终端")


def type_run(ide, cmd, wait):
    ide.keyboard.insert_text(cmd)
    ide.wait_for_timeout(400)
    ide.keyboard.press("Enter")
    print(f"    ▸ {cmd[:100]}")
    ide.wait_for_timeout(int(wait * 1000))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--wait", type=float, default=10)
    ap.add_argument("--hold", type=int, default=0)
    args = ap.parse_args()

    cmds = [l.strip() for l in Path(args.script).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")]

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE), headless=False, channel="msedge",
            viewport={"width": 1600, "height": 1000}, locale="zh-CN",
            args=["--disable-blink-features=AutomationControlled"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.goto(NOTEBOOK_URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(8000)
            body = page.inner_text("body")
            print("运行中:", "运行中" in body)
            for txt in ("查看", "打开", "进入"):
                loc = page.locator(f"text={txt}")
                if loc.count():
                    try:
                        loc.first.click(timeout=5000)
                        break
                    except Exception:
                        continue
            page.wait_for_timeout(15000)
            ide = ctx.pages[-1]
            try:
                ide.bring_to_front()
            except Exception:
                pass
            ide.wait_for_timeout(5000)
            shot(ide, "ide")

            for i, c in enumerate(cmds):
                # 每条命令前都新建一个干净终端 —— 避免焦点被抢
                new_terminal(ide)
                shot(ide, f"term{i}")
                type_run(ide, c, args.wait)
                shot(ide, f"after{i}")

            print("全部命令已发送")
            if args.hold:
                time.sleep(args.hold)
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
