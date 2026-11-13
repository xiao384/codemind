"""进入 ModelScope Notebook 的 Web IDE，并在终端里执行命令。

## 为什么这样做

用户要求「我负责观察并最大化使用实例」，但云端终端只能通过浏览器操作。
ModelScope 的 Notebook 点「打开」会进一个**基于 VS Code 的 Web IDE**，
Playwright 可以直接在它的终端面板里打字 —— 不需要用户手动敲命令。

## 关键点

1. **终端面板**：VS Code Web 用 `Ctrl+反引号` 切换终端。如果没自动打开，
   先点菜单 `终端 → 新建终端`。
2. **多标签**：点「打开」通常会开新标签页，要切到 `context.pages[-1]`。
3. **输入命令**：找到 `textarea` 或 `.xterm` 容器，`page.keyboard.type()` 输入，
   `page.keyboard.press('Enter')` 执行。**xterm 需要先 click 取得焦点。**

用法::

    python tools/ms_ide.py --cmd "ls -l /mnt/workspace/runs/s1b/latest.pt"
    python tools/ms_ide.py --script tools/cloud_resume.txt   # 逐行执行
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"E:\AI")
PROFILE = ROOT / ".browser_profile"
NOTEBOOK_URL = "https://www.modelscope.cn/my/mynotebook"
SHOT_DIR = ROOT / "runs/_verify"


def shot(page, name: str) -> None:
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    p = SHOT_DIR / f"ide-{name}.png"
    try:
        page.screenshot(path=str(p), full_page=False)
        print(f"    截图: {p}")
    except Exception as e:
        print(f"    截图失败: {type(e).__name__}")


def open_ide(ctx, page) -> object:
    """在 Notebook 页点「打开」，返回 IDE 页面对象。"""
    page.goto(NOTEBOOK_URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(9000)
    body = page.inner_text("body")
    print("  已登录:", "xiaoxiao384" in body)
    print("  运行中:", "运行中" in body, " 启动中:", "启动中" in body)

    if "启动中" in body:
        print("  ⏳ 实例还在启动中，等待…")
        for i in range(40):
            time.sleep(10)
            page.reload(wait_until="domcontentloaded")
            page.wait_for_timeout(4000)
            b = page.inner_text("body")
            if "运行中" in b:
                print(f"  ✓ 已运行（等了 {10*(i+1)} 秒）")
                break
            if i % 6 == 5:
                print(f"    仍在启动… {10*(i+1)}s")
        else:
            raise SystemExit("等待实例启动超时")

    # 找「打开」按钮（运行中时才有）
    cands = ["查看", "打开", "进入", "连接", "打开Notebook"]
    clicked = False
    for txt in cands:
        loc = page.locator(f"text={txt}")
        if loc.count():
            try:
                loc.first.click(timeout=5000)
                print(f"  点击了「{txt}」")
                clicked = True
                break
            except Exception:
                continue
    if not clicked:
        shot(page, "no-open-btn")
        raise SystemExit("找不到「打开」按钮，请看截图")

    page.wait_for_timeout(12000)                 # IDE 加载慢
    pgs = ctx.pages
    print(f"  当前标签数: {len(pgs)}")
    ide = pgs[-1]
    try:
        ide.bring_to_front()
    except Exception:
        pass
    ide.wait_for_timeout(3000)
    print("  IDE URL:", ide.url[:110])
    print("  IDE 标题:", ide.title()[:80])
    shot(ide, "opened")
    return ide


def find_terminal(page):
    """返回可输入的终端元素（xterm textarea 优先）。"""
    for sel in (".xterm-helper-textarea", ".xterm textarea", "textarea.xterm-helper-textarea",
                ".monaco-editor textarea", "textarea"):
        loc = page.locator(sel)
        if loc.count():
            try:
                loc.last.click(timeout=3000)
                return loc.last, sel
            except Exception:
                continue
    return None, None


def run_cmd(page, cmd: str, wait: float = 3.0) -> None:
    el, sel = find_terminal(page)
    if not el:
        print("    ✗ 找不到终端输入框")
        shot(page, "no-terminal")
        return
    el.click()
    page.wait_for_timeout(300)
    page.keyboard.type(cmd, delay=8)
    page.wait_for_timeout(400)
    page.keyboard.press("Enter")
    print(f"    ▸ {cmd[:100]}")
    page.wait_for_timeout(int(wait * 1000))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", action="append", default=[])
    ap.add_argument("--script")
    ap.add_argument("--wait", type=float, default=3.0)
    ap.add_argument("--hold", type=int, default=60)
    args = ap.parse_args()

    cmds = list(args.cmd)
    if args.script:
        cmds += [l.strip() for l in Path(args.script).read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE), headless=False, channel="msedge",
            viewport={"width": 1600, "height": 1000}, locale="zh-CN",
            args=["--disable-blink-features=AutomationControlled"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            ide = open_ide(ctx, page)
            print("\n  尝试打开终端（Ctrl+反引号）…")
            ide.keyboard.press("Control+Backquote")
            ide.wait_for_timeout(3000)
            shot(ide, "terminal")

            for c in cmds:
                run_cmd(ide, c, args.wait)
            shot(ide, "after")
            if args.hold:
                print(f"\n  保持窗口 {args.hold} 秒供你查看…")
                ide.wait_for_timeout(args.hold * 1000)
                shot(ide, "final")
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
