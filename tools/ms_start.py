"""打开 ModelScope Notebook 页，点击指定实例的「启动」，
然后**保持浏览器窗口打开 3 分钟**，让用户手动拖验证码滑块。

## 为什么这么做

阿里滑块验证码校验**轨迹时序**：
- DSH 的 `browser_mouse` 每次调用延迟 1~3 秒 → 拖 10 秒 → **实测"验证失败"**
- JS 合成 MouseEvent → `isTrusted=false` → **阿里直接忽略**（实测无任何反应）
- Playwright 的 `page.mouse` 能产生 trusted 事件，但求解需要**准确知道缺口位置**，
  而 AIGC 验证码的遮罩规律不稳定

**浏览器窗口是可见的（headless=False）**，所以最可靠的办法是：
脚本把窗口开到验证码那一步，用户用自己的鼠标拖一下。

用法::

    python tools/ms_start.py --which 2 --hold 180
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"E:\AI")
PROFILE = ROOT / ".browser_profile"
NOTEBOOK_URL = "https://www.modelscope.cn/my/mynotebook"
SHOT = ROOT / "runs/_verify/ms_start.png"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=int, default=2,
                    help="DOM 顺序：0=CPU 1=GPU(24G) 2=AMD(192G)")
    ap.add_argument("--hold", type=int, default=180,
                    help="点完启动后保持窗口打开的秒数")
    args = ap.parse_args()

    with sync_playwright() as p:
        # ⚠️ 复用同一个 profile 目录 —— 登录态（cookie）在里面
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE),
            headless=False,                    # 必须可见，用户要手动拖
            channel="msedge",
            viewport={"width": 1440, "height": 900},
            locale="zh-CN",
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.goto(NOTEBOOK_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(4000)
            print("  标题:", page.title())

            body = page.inner_text("body")
            if "登录" in body[:400] and "xiaoxiao384" not in body:
                print("  ⚠️ 可能未登录 —— 请先在窗口里手动登录")
            else:
                print("  ✓ 已登录（页面能看到账号信息）")

            btns = page.locator("text=启动")
            n = btns.count()
            print(f"  找到 {n} 个「启动」按钮")
            if n == 0:
                print("  ✗ 没找到启动按钮，请看窗口")
                time.sleep(args.hold)
                return 1
            if args.which >= n:
                print(f"  ✗ --which {args.which} 超出范围（共 {n} 个）")
                return 2

            print(f"  点击第 {args.which} 个启动（0=CPU 1=GPU 2=AMD）")
            btns.nth(args.which).click()
            print(f"\n>>> 窗口已打开，请**手动拖动滑块**完成验证（{args.hold} 秒后脚本退出）")

            t0 = time.time()
            while time.time() - t0 < args.hold:
                time.sleep(3)
                try:
                    gone = page.evaluate(
                        "() => !document.querySelector('#aliyunCaptcha-mask.mask-show')")
                except Exception:
                    gone = True
                if gone and time.time() - t0 > 20:
                    print(f"  验证码已消失（{int(time.time()-t0)} 秒），可能已通过 ✓")
                    time.sleep(15)          # 给实例启动留时间
                    break

            SHOT.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(SHOT))
            print("  截图:", SHOT)
            print("  当前 URL:", page.url[:100])
            print("  页面摘要:", page.inner_text("body")[:200].replace("\n", " | "))
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
