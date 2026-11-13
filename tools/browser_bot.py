"""浏览器自动化模板：① 自己的网页游戏 ② 用自己的账号浏览视频。

## 为什么用 Playwright 而不是 Selenium

| | Selenium | **Playwright** |
|---|---|---|
| 速度 | 慢（每次要等 WebDriver） | **快** |
| 自动等待 | 要手写 `WebDriverWait` | **内置**（元素出现就点，不用 sleep） |
| 隐身 | 默认很容易被识别 | 默认更像真人（但**仍不是隐身**） |
| 多标签/iframes | 麻烦 | 原生支持 |

## ⚠️ 三条必须先知道的现实

1. **持久化登录**：第一次手动登录，之后 profile 目录会保存 cookie，不用重复登录。
2. **"像真人"不能靠加睡眠**：平台风控检测的是**行为指纹**（鼠标轨迹、间隔分布、
   滚动加速度、无头浏览器特征）。本脚本做了基础的随机化，但**不等于不会被封**。
3. **自己的游戏随便玩**；**用自己账号浏览内容也 OK**，但只要被判定"异常自动化"，
   账号仍可能被限流或封禁 —— 这个风险你自己承担。

## 安装

    uv pip install --python .venv\\Scripts\\python.exe playwright
    .venv\\Scripts\\python.exe -m playwright install chromium

用法::

    # 第一次：手动登录（脚本会打开浏览器，你登录完按回车）
    python tools/browser_bot.py --mode login --url https://example.com

    # 之后：对自己的游戏做自动化
    python tools/browser_bot.py --mode game --url https://yourgame.example.com

    # 或者浏览视频
    python tools/browser_bot.py --mode video --url https://www.bilibili.com
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

#: 持久化 profile 目录 —— 登录一次就记住
PROFILE = Path(r"E:\AI\.browser_profile")


def human_pause(a: float = 0.6, b: float = 2.2) -> None:
    """随机停顿（**固定间隔是风控最容易识别的东西**）。"""
    time.sleep(random.uniform(a, b))


def human_move(page, x: int, y: int) -> None:
    """分几步移动鼠标，而不是瞬移。"""
    page.mouse.move(x, y, steps=random.randint(8, 25))
    human_pause(0.15, 0.5)


def human_scroll(page, times: int = 3) -> None:
    """带随机量的滚动 —— 匀速滚动是机器特征。"""
    for _ in range(times):
        page.mouse.wheel(0, random.randint(200, 700))
        human_pause(0.6, 2.5)


def launch(p, headless: bool):
    """打开一个使用持久化 profile 的浏览器。

    ## ⚠️ 为什么用系统 Edge 而不是 playwright 的 chromium

    标准的 `p.chromium.launch()` 会去找 `ms-playwright` 目录下的 chromium，
    **没装就会报错**；而 `playwright install chromium` 要从海外 CDN 下 150 MB ——
    本项目实测**到海外 CDN 的连接基本不可用**（huggingface 直接超时）。

    **Windows 自带 Edge**（Chromium 内核），指定 `channel="msedge"` 就能直接用，
    **零下载**。这是本项目"网络受限时优先用本地已有资源"这条经验的应用。
    """
    PROFILE.mkdir(parents=True, exist_ok=True)
    ctx = p.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE),
        headless=headless,
        channel="msedge",              # ← 用系统 Edge，不下载 chromium
        viewport={"width": 1440, "height": 900},
        locale="zh-CN",
        # ⚠️ 隐藏最明显的无头特征（不是万能的，但少一条少一条）
        args=["--disable-blink-features=AutomationControlled"],
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    return ctx, page


def mode_login(page, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded")
    print("\n>>> 浏览器已打开，请**手动登录**。登录完成后回到这里按回车。")
    input()
    print(">>> 已保存登录状态到", PROFILE)


def mode_game(page, url: str) -> None:
    """自己的网页游戏：自动化框架。

    ⚠️ 这里只能给**框架** —— 你的游戏里有哪些按钮/元素，我不知道。
    用 F12 → 选中元素 → 右键 Copy → Copy selector，把选择器填到下面。
    """
    page.goto(url, wait_until="domcontentloaded")
    print(">>> 已进入游戏页，5 秒后开始（可按 Ctrl+C 停）")
    time.sleep(5)

    # ══════════ 在这里填你自己的选择器 ══════════
    START_BTN = "button.start"          # 开始/开局按钮
    ACTION_BTN = "button.attack"        # 主要操作按钮
    # ══════════════════════════════════════════

    try:
        if page.locator(START_BTN).count():
            human_move(page, 700, 500)
            page.locator(START_BTN).first.click()
            print("  已点开始")
            human_pause(1.0, 3.0)
    except Exception as e:
        print(f"  ⚠️ 点开始失败: {type(e).__name__}: {e}")

    rounds = 0
    while True:
        rounds += 1
        try:
            btn = page.locator(ACTION_BTN)
            if btn.count():
                box = btn.first.bounding_box()
                if box:
                    # 点到按钮内的**随机位置**，不是永远点正中心
                    human_move(page,
                               int(box["x"] + box["width"] * random.uniform(0.3, 0.7)),
                               int(box["y"] + box["height"] * random.uniform(0.3, 0.7)))
                btn.first.click(timeout=5000)
            if rounds % 20 == 0:
                print(f"  第 {rounds} 轮")
                human_pause(2.0, 5.0)       # 偶尔歇一下
            human_pause(0.8, 2.0)           # ⚠️ 别用固定间隔
        except KeyboardInterrupt:
            print(f"\n>>> 已停止，共 {rounds} 轮")
            return
        except Exception as e:
            print(f"  ⚠️ 第 {rounds} 轮出错: {type(e).__name__}: {str(e)[:80]}")
            human_pause(2.0, 4.0)


def mode_video(page, url: str) -> None:
    """用自己账号浏览视频。

    ⚠️ **这是"看"，不是"刷量"** —— 不循环播放同一视频、不伪造完播率。
    只做：进首页 → 随机点几个视频 → 看一会儿 → 返回 → 重复。
    """
    page.goto(url, wait_until="domcontentloaded")
    human_pause(2.0, 4.0)
    rounds = 0
    while True:
        rounds += 1
        try:
            # 找页面上的视频链接（各站结构不同，这里是通用做法）
            links = page.locator('a[href*="/video/"], a[href*="/watch"]').all()
            if not links:
                print(f"  第 {rounds} 轮：没找到视频链接，滚动看看")
                human_scroll(page, 3)
                human_pause(1.5, 3.0)
                continue

            pick = random.choice(links[:40])    # 从可见的里随机挑
            pick.click()
            print(f"  第 {rounds} 轮：进入一个视频")
            human_pause(3.0, 6.0)
            human_scroll(page, random.randint(0, 3))   # 有时不滚，有时滚几下
            watch = random.uniform(20, 90)             # 看 20~90 秒
            print(f"    看 {watch:.0f} 秒…")
            time.sleep(watch)
            page.go_back()
            human_pause(2.0, 5.0)
        except KeyboardInterrupt:
            print(f"\n>>> 已停止，共看了 {rounds} 个")
            return
        except Exception as e:
            print(f"  ⚠️ 第 {rounds} 轮出错: {type(e).__name__}: {str(e)[:80]}")
            human_pause(2.0, 4.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["login", "game", "video"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--headless", action="store_true",
                    help="无头模式（更快但**更容易被识别**，建议不开）")
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("✗ 没装 playwright：\n"
              "  uv pip install --python .venv\\Scripts\\python.exe playwright\n"
              "  .venv\\Scripts\\python.exe -m playwright install chromium")
        return 2

    with sync_playwright() as p:
        ctx, page = launch(p, args.headless)
        try:
            if args.mode == "login":
                mode_login(page, args.url)
            elif args.mode == "game":
                mode_game(page, args.url)
            else:
                mode_video(page, args.url)
        except KeyboardInterrupt:
            print("\n已中止")
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
