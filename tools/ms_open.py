"""打开一个可见的 Edge 窗口停在指定页面，保持 N 秒让用户手动操作（登录等）。

用法::

    # 打开登录页，保持 10 分钟让用户登录
    python tools/ms_open.py --url https://www.modelscope.cn/login --hold 600

    # 或直接开到 Notebook 页
    python tools/ms_open.py --url https://www.modelscope.cn/my/mynotebook --hold 600

登录态保存在 `.browser_profile`（和 `ms_start.py` 共用），
所以这里登录一次之后，`ms_start.py` 就能直接用。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"E:\AI")
PROFILE = ROOT / ".browser_profile"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://www.modelscope.cn/login")
    ap.add_argument("--hold", type=int, default=600)
    args = ap.parse_args()

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE),
            headless=False,
            channel="msedge",
            viewport={"width": 1440, "height": 900},
            locale="zh-CN",
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.goto(args.url, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)
            print(f"窗口已打开：{args.url}")
            print(f"标题：{page.title()}")
            print(f">>> 请在这个窗口里手动操作（登录等）。保持 {args.hold} 秒。")
            print(">>> 完成后不用关窗口，脚本会自己退出（或按 Ctrl+C 让我提前知道）。")

            t0 = time.time()
            last = ""
            while time.time() - t0 < args.hold:
                time.sleep(5)
                try:
                    cur = page.url
                    if cur != last:
                        last = cur
                        print(f"  [{int(time.time()-t0):4d}s] URL -> {cur[:90]}")
                    body = page.inner_text("body")[:300]
                    logged = "xiaoxiao384" in body or "我的Notebook" in page.title()
                    if logged:
                        print(f"  ✓ 检测到已登录（{int(time.time()-t0)}s）")
                        ctx.storage_state(path=str(ROOT / ".ms_state.json"))
                        print("  登录态已存 .ms_state.json")
                        time.sleep(5)
                        break
                except Exception as e:
                    print(f"  轮询异常 {type(e).__name__}: {str(e)[:60]}")

            print("  当前 URL:", page.url[:100])
            print("  页面摘要:", page.inner_text("body")[:180].replace("\n", " | "))
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
