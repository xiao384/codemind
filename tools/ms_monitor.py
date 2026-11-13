"""定时截图 ModelScope Web IDE，用于监控云端训练。

## 为什么不读日志文本

Web IDE 的终端/编辑器不是标准 DOM，`inner_text` 拿不到内容；
而**打字又极不稳定**（焦点被【问题】面板抢走，本项目反复踩过）。

**最可靠的办法是截图** —— 我（视觉模型）直接看截图里的终端输出，
和用户在屏幕上看到的一样。

用法::

    python tools/ms_monitor.py --interval 1800 --shots 12
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"E:\AI")
PROFILE = ROOT / ".browser_profile"
SHOT_DIR = ROOT / "runs/_verify/monitor"
NOTEBOOK_URL = "https://www.modelscope.cn/my/mynotebook"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=1800, help="每多少秒截一次")
    ap.add_argument("--shots", type=int, default=12)
    args = ap.parse_args()

    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE), headless=False, channel="msedge",
            viewport={"width": 1600, "height": 1000}, locale="zh-CN",
            args=["--disable-blink-features=AutomationControlled"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            for k in range(args.shots):
                ts = datetime.now().strftime("%m%d-%H%M")
                try:
                    page.goto(NOTEBOOK_URL, wait_until="domcontentloaded", timeout=60000)
                    page.wait_for_timeout(8000)
                    body = page.inner_text("body")
                    running = "运行中" in body

                    if not running:
                        msg = f"[{ts}] 实例不在运行中（可能已停止/额度用完）"
                        print(msg, flush=True)
                        (SHOT_DIR / "status.log").open("a", encoding="utf-8").write(msg + "\n")
                        page.screenshot(path=str(SHOT_DIR / f"{ts}-notrunning.png"))
                        break

                    # 进入 IDE
                    clicked = False
                    for txt in ("查看", "打开", "进入"):
                        loc = page.locator(f"text={txt}")
                        if loc.count():
                            try:
                                loc.first.click(timeout=5000)
                                clicked = True
                                break
                            except Exception:
                                continue
                    if not clicked:
                        print(f"[{ts}] 找不到进入按钮", flush=True)
                        page.screenshot(path=str(SHOT_DIR / f"{ts}-nobtn.png"))
                    else:
                        page.wait_for_timeout(14000)
                        ide = ctx.pages[-1]
                        try:
                            ide.bring_to_front()
                        except Exception:
                            pass
                        ide.wait_for_timeout(4000)
                        # 切到【终端】标签（坐标法，实测文本选择器无效）
                        vp = ide.viewport_size or {"width": 1600, "height": 1000}
                        try:
                            ide.keyboard.press("Escape")
                            ide.wait_for_timeout(300)
                            ide.mouse.click(int(vp["width"] * 0.3425),
                                            int(vp["height"] * 0.634))
                            ide.wait_for_timeout(1500)
                        except Exception:
                            pass
                        f = SHOT_DIR / f"{ts}-ide.png"
                        ide.screenshot(path=str(f))
                        print(f"[{ts}] 截图 {f.name}", flush=True)
                        (SHOT_DIR / "status.log").open("a", encoding="utf-8").write(
                            f"{ts} running, shot {f.name}\n")
                        # 关掉 IDE 标签，避免越开越多
                        if len(ctx.pages) > 1:
                            try:
                                ide.close()
                            except Exception:
                                pass
                except Exception as e:
                    msg = f"[{ts}] 出错 {type(e).__name__}: {str(e)[:90]}"
                    print(msg, flush=True)
                    (SHOT_DIR / "status.log").open("a", encoding="utf-8").write(msg + "\n")

                if k < args.shots - 1:
                    time.sleep(args.interval)
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
