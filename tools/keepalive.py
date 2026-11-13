"""保活 + 监控：每轮**重建浏览器**，只做最小的有效交互。

## 两个必须满足的需求

1. **防实例回收**：平台规则是「1 小时无浏览器交互 → 挂起实例」（GPU 满载不算）。
   所以每 35 分钟要真实访问一次 modelscope.cn。
2. **监控训练**：进 IDE 打开 runs/s1b.log 截图。

## 为什么每轮重建 context

实测复用同一个 persistent context 时，第二轮会报
`TargetClosedError: Target context or browser has been closed`
（IDE 打开的标签页被 IDE 自己关掉，连带把 context 弄挂）。
**每轮 launch + close 最稳**，代价是每轮多花 15 秒启动。

用法::

    python tools/keepalive.py --interval 2100 --rounds 10
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"E:\AI")
PROFILE = ROOT / ".browser_profile"
OUT = ROOT / "runs/_verify/monitor"
NOTEBOOK_URL = "https://www.modelscope.cn/my/mynotebook"


def one_round() -> str:
    ts = datetime.now().strftime("%m%d-%H%M")
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE), headless=False, channel="msedge",
            viewport={"width": 1600, "height": 1000}, locale="zh-CN",
            args=["--disable-blink-features=AutomationControlled"])
        try:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(NOTEBOOK_URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(8000)
            body = page.inner_text("body")
            if "运行中" not in body:
                page.screenshot(path=str(OUT / f"{ts}-notrunning.png"))
                return f"[{ts}] 实例**不在运行中**"
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
            ide.wait_for_timeout(4000)
            ide.keyboard.press("Control+p")
            ide.wait_for_timeout(1500)
            ide.keyboard.insert_text("runs/s1b.log")
            ide.wait_for_timeout(2500)
            ide.keyboard.press("Enter")
            ide.wait_for_timeout(6000)
            f = OUT / f"{ts}-log.png"
            ide.screenshot(path=str(f))
            return f"[{ts}] ✓ 保活+截图 {f.name}"
        finally:
            try:
                ctx.close()
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=2100)
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for k in range(args.rounds):
        try:
            msg = one_round()
        except Exception as e:
            msg = f"[{datetime.now():%m%d-%H%M}] 出错 {type(e).__name__}: {str(e)[:90]}"
        print(msg, flush=True)
        (OUT / "keepalive.log").open("a", encoding="utf-8").write(msg + "\n")
        if k < args.rounds - 1:
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
