"""驱动 ModelScope Web IDE 的终端执行命令。

## 和 ms_ide.py 的区别

`ms_ide.py` 用选择器找终端输入框 —— **找不到**（"代码工作区"的终端不是标准 xterm
`textarea`，可能是 canvas 渲染或 shadow DOM）。

**本脚本改用「点坐标 + keyboard.type」**：终端固定在页面底部面板，
点一下面板区域取得焦点，然后直接打字。实测可行。

## 用法

    python tools/ms_term.py --script tools/cloud_resume.txt
    python tools/ms_term.py --cmd "ls -l /mnt/workspace/runs/s1b/latest.pt"
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"E:\AI")
PROFILE = ROOT / ".browser_profile"
SHOT_DIR = ROOT / "runs/_verify"
NOTEBOOK_URL = "https://www.modelscope.cn/my/mynotebook"


def shot(page, name: str) -> Path:
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    p = SHOT_DIR / f"term-{name}.png"
    try:
        page.screenshot(path=str(p))
    except Exception:
        pass
    return p


def open_ide(ctx, page):
    page.goto(NOTEBOOK_URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(9000)
    body = page.inner_text("body")
    print("  已登录:", "xiaoxiao384" in body, "| 运行中:", "运行中" in body)
    if "运行中" not in body:
        raise SystemExit("实例不在运行中")

    clicked = False
    for txt in ("查看", "打开", "进入"):
        loc = page.locator(f"text={txt}")
        if loc.count():
            try:
                loc.first.click(timeout=5000)
                print(f"  点击「{txt}」")
                clicked = True
                break
            except Exception:
                continue
    if not clicked:
        raise SystemExit("找不到进入按钮")

    page.wait_for_timeout(14000)
    ide = ctx.pages[-1]
    try:
        ide.bring_to_front()
    except Exception:
        pass
    ide.wait_for_timeout(4000)
    print("  IDE:", ide.url[:90])
    return ide


def focus_terminal(page) -> bool:
    """点击底部终端面板取得键盘焦点。

    ⚠️ 踩过两次的坑：
    1. 面板默认停在【问题】页 —— 必须先切到【终端】，否则打字进**右上角筛选框**。
    2. 文本选择器 `text=终端` 找不到/点不动 —— 实测无效，改**按坐标点标签**。
    """
    vp = page.viewport_size or {"width": 1600, "height": 1000}
    w, h = vp["width"], vp["height"]

    # ① Esc 清掉当前焦点（筛选框会抢焦点）
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)
    except Exception:
        pass

    # ② 优先用 VS Code 快捷键 Ctrl+` 聚焦终端（比点坐标可靠得多）
    try:
        page.keyboard.press("Control+Backquote")
        print("    按 Ctrl+` 聚焦终端")
        page.wait_for_timeout(1500)
    except Exception as e:
        print(f"    快捷键失败: {type(e).__name__}")

    # ③ 兜底：再按坐标点一次【终端】标签 + 面板正文
    tab_xy = (int(w * 0.3425), int(h * 0.634))
    try:
        page.mouse.click(*tab_xy)
        page.wait_for_timeout(1200)
    except Exception:
        pass

    cands = [
        (int(w * 0.30), int(h * 0.73)),
        (int(w * 0.45), int(h * 0.74)),
        (int(w * 0.30), int(h * 0.76)),
    ]
    for (x, y) in cands:
        try:
            page.mouse.click(x, y)
            page.wait_for_timeout(500)
        except Exception:
            continue
    return True


def type_cmd(page, cmd: str, wait: float) -> None:
    """输入命令。

    ⚠️ 用 `keyboard.insert_text` 而不是 `keyboard.type`：
    `type()` 会逐字符发键盘事件，长命令中途容易被 VS Code 的快捷键/面板抢焦点
    （实测命令尾巴跑进【问题】面板的筛选框）。`insert_text` 是直接插入文本，
    不触发快捷键，长命令更稳。
    """
    try:
        page.keyboard.insert_text(cmd)
    except Exception as e:
        print(f"    insert_text 失败({type(e).__name__})，退回 type()")
        page.keyboard.type(cmd, delay=5)
    page.wait_for_timeout(500)
    page.keyboard.press("Enter")
    print(f"    ▸ {cmd[:110]}")
    page.wait_for_timeout(int(wait * 1000))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", action="append", default=[])
    ap.add_argument("--script")
    ap.add_argument("--wait", type=float, default=6.0)
    ap.add_argument("--hold", type=int, default=0)
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
            print("  聚焦终端…")
            focus_terminal(ide)
            shot(ide, "focused")

            # ⚠️ 不要在这里发探针命令 —— 实测探针会先打在筛选框里，
            # 切到终端时残余字符会污染第一条真实命令（"echo" 变成 "cho"）。
            # 焦点是否正确，看 step 截图即可。
            p1 = shot(ide, "focused")
            print(f"    焦点截图: {p1}")

            for c in cmds:
                # ⚠️ 每条命令前都重新聚焦 —— 实测焦点会在长命令中途被
                # 【问题】面板/筛选框抢走（命令尾巴跑进筛选框）。
                focus_terminal(ide)
                type_cmd(ide, c, args.wait)
                shot(ide, "step")

            shot(ide, "done")
            if args.hold:
                print(f"  保持 {args.hold}s")
                ide.wait_for_timeout(args.hold * 1000)
                shot(ide, "final")
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
