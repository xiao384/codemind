"""通过 Web IDE 的**文件树**打开 runs/s1b.log，跳到文件末尾并截图。

## 为什么走文件树而不是终端

实测这个魔搭 Web IDE 的【终端】标签**点不动**（坐标点击无效，文本选择器无效），
而文件树是标准 DOM 元素，`locator.click()` 可以稳定展开/打开文件。
打开日志后按 `Ctrl+End` 跳到末尾，截图即可看到最新几行训练输出。
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
    ap.add_argument("--log", default="runs/s1b.log")
    ap.add_argument("--hold", type=int, default=0)
    args = ap.parse_args()
    SHOT_DIR.mkdir(parents=True, exist_ok=True)

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
            print("运行中:", "运行中" in body, "| 已登录:", "xiaoxiao384" in body)
            if "运行中" not in body:
                raise SystemExit("实例不在运行中")

            for txt in ("查看", "打开", "进入"):
                loc = page.locator(f"text={txt}")
                if loc.count():
                    try:
                        loc.first.click(timeout=5000)
                        print(f"点击「{txt}」")
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

            # 用快速打开 Ctrl+P 直接打开文件（比点文件树更稳）
            ide.keyboard.press("Control+p")
            ide.wait_for_timeout(1500)
            ide.keyboard.insert_text(args.log)
            ide.wait_for_timeout(2500)
            ide.keyboard.press("Enter")
            print("已用 Ctrl+P 打开:", args.log)
            ide.wait_for_timeout(6000)

            # 跳到文件末尾
            ide.keyboard.press("Control+End")
            ide.wait_for_timeout(2500)

            ts = datetime.now().strftime("%m%d-%H%M")
            f = SHOT_DIR / f"{ts}-log.png"
            ide.screenshot(path=str(f))
            print("截图:", f)

            # 尝试读编辑器文本（VS Code Web 的 Monaco 有 .view-lines）
            try:
                txt = ide.inner_text(".view-lines")
                lines = [l for l in txt.splitlines() if l.strip()][-12:]
                print("\n=== 日志末尾 ===")
                for l in lines:
                    print(" ", l[:150])
            except Exception as e:
                print("读文本失败:", type(e).__name__)

            if args.hold:
                time.sleep(args.hold)
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
