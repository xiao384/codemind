"""阿里云 DSW Lab 全自动：登录 → 打开 JupyterLab → 终端执行命令。

## 账号密码

从 `E:\\AI\\.aliyun_cred`（两行：账号 / 密码）读取，**不硬编码在脚本里**，
避免脚本被复制或提交时泄露。文件不存在时回退到环境变量
`ALIYUN_USER` / `ALIYUN_PASS`。

## 为什么必须走这条路

ModelScope 的 Web IDE 是魔改 VS Code，终端**点不动也打不进字**（试过 8 种方法）。
阿里云 DSW 的 JupyterLab 是**标准 xterm.js**，`.xterm-helper-textarea` 可直接输入 ——
实测 `pwd && ls -l runs/s1b/latest.pt && echo TERM_OK` 一次成功。

用法::

    python tools/lab_auto.py --cmd "tail -3 runs/s1b.log"
    python tools/lab_auto.py --script tools/cloud_check.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"E:\AI")
PROFILE = ROOT / ".browser_profile"
CRED = ROOT / ".aliyun_cred"
SHOT = ROOT / "runs/_verify"
LAB = ("https://dsw-gateway-cn-hangzhou.data.aliyun.com/dsw-2346511/lab"
       "?appId=MAAS&accounttraceid=c03f1047e3d749bc979e22a2bbe2793chdjj&features=EnableSubDomainProxy")


def creds() -> tuple[str, str]:
    if CRED.is_file():
        lines = [l.strip() for l in CRED.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) >= 2:
            return lines[0], lines[1]
    return os.environ.get("ALIYUN_USER", ""), os.environ.get("ALIYUN_PASS", "")


def is_logged_in(pg) -> bool:
    try:
        pg.goto(LAB, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_timeout(8000)
        t = pg.title()
        return "JupyterLab" in t
    except Exception:
        return False


def login(pg, user: str, pw: str) -> bool:
    pg.goto("https://account.aliyun.com/login/login.htm", wait_until="domcontentloaded",
            timeout=60000)
    pg.wait_for_timeout(4000)
    print("  当前:", pg.title()[:40], flush=True)

    # 切到「账密登录」
    for txt in ("账密登录", "账号密码登录", "密码登录"):
        try:
            loc = pg.locator(f"text={txt}")
            if loc.count():
                loc.first.click(timeout=4000)
                print(f"  切换【{txt}】", flush=True)
                pg.wait_for_timeout(2000)
                break
        except Exception:
            continue

    filled = False
    # ⚠️ 阿里云登录表单在 **跨域 iframe**（passport.aliyun.com/havanaone）里，
    # 直接 `pg.locator("input")` 永远是 0 个（实测）。必须用 frame_locator。
    FRAME_SEL = 'iframe[src*="passport.aliyun.com/havanaone"]'
    scope = None
    try:
        fl = pg.frame_locator(FRAME_SEL)
        if fl.locator('input[name="fm-login-id"]').count():
            scope = fl
            print("  定位到登录 iframe", flush=True)
    except Exception as e:
        print("  frame_locator 失败:", type(e).__name__, flush=True)

    if scope is None:
        for f in pg.frames:
            try:
                if f.locator('input[name="fm-login-id"]').count():
                    f.locator('input[name="fm-login-id"]').first.fill(user, timeout=6000)
                    f.locator('input[name="fm-login-password"]').first.fill(pw, timeout=6000)
                    print("  账号+密码已填（frames 遍历）", flush=True)
                    filled = True
                    break
            except Exception:
                continue
    else:
        try:
            scope.locator('input[name="fm-login-id"]').first.fill(user, timeout=6000)
            scope.locator('input[name="fm-login-password"]').first.fill(pw, timeout=6000)
            print("  账号+密码已填（iframe）", flush=True)
            filled = True
        except Exception as e:
            print("  填充失败:", type(e).__name__, flush=True)

    # 图片验证码：如果有就要人工输入
    try:
        cap = scope.locator('input[placeholder="图片验证码"]') if scope else None
        if cap and cap.count():
            print("  ⚠️ 出现图片验证码 —— 请在窗口里手动输入", flush=True)
    except Exception:
        pass

    if not filled:
        pg.screenshot(path=str(SHOT / "login-nouser.png"))
        print("  ✗ 找不到账号框", flush=True)
        return False

    for sel in ("#fm-login-password", 'input[name="password"]', 'input[type=password]'):
        try:
            if pg.locator(sel).count():
                pg.locator(sel).first.fill(pw, timeout=6000)
                print(f"  密码填入 {sel}", flush=True)
                break
        except Exception:
            continue

    pg.wait_for_timeout(800)
    for sel in ("button[type=submit]", ".fm-button", "#login-submit", "text=登录"):
        try:
            loc = pg.locator(sel)
            if loc.count():
                loc.first.click(timeout=6000)
                print(f"  提交 {sel}", flush=True)
                break
        except Exception:
            continue

    pg.wait_for_timeout(12000)
    print("  登录后 URL:", pg.url[:90], flush=True)
    pg.screenshot(path=str(SHOT / "after-login.png"))
    return True


def open_terminal(pg) -> None:
    if pg.locator(".xterm-helper-textarea").count():
        print("  已有终端", flush=True)
        return
    r = pg.evaluate("""() => {
      const c = document.querySelector('.jp-LauncherCard[title="Start a new terminal session"]');
      if (!c) return 'no card';
      c.click(); return 'clicked';
    }""")
    print("  开终端:", r, flush=True)
    pg.wait_for_timeout(9000)


def run_cmd(pg, cmd: str, wait: float) -> None:
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
    args = ap.parse_args()

    cmds = list(args.cmd)
    if args.script:
        cmds += [l.strip() for l in Path(args.script).read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]

    user, pw = creds()
    if not user:
        print("✗ 缺少账号：请写 E:\\AI\\.aliyun_cred（第一行账号，第二行密码）", flush=True)
        return 2

    SHOT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE), headless=False, channel="msedge",
            viewport={"width": 1600, "height": 1000}, locale="zh-CN",
            args=["--disable-blink-features=AutomationControlled"])
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            if not is_logged_in(pg):
                print("  需要登录…", flush=True)
                login(pg, user, pw)
                if not is_logged_in(pg):
                    print("  ⚠️ 仍未进入 Lab（可能需要滑块验证）—— 截图见 runs/_verify", flush=True)
                    pg.screenshot(path=str(SHOT / "not-in-lab.png"))
                    return 3
            print("  ✓ 已在 Lab", flush=True)
            open_terminal(pg)
            for c in cmds:
                run_cmd(pg, c, args.wait)
            ts = datetime.now().strftime("%H%M%S")
            f = SHOT / f"lab-{ts}.png"
            pg.screenshot(path=str(f))
            print("  截图:", f, flush=True)
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
