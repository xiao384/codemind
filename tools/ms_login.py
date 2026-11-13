"""用 Playwright 登录 ModelScope 并尝试启动 Notebook 实例（含验证码求解）。

## 为什么要用这个脚本，而不是 DSH 的浏览器工具

**DSH 的 `browser_mouse` 每次调用有 1~3 秒延迟**。阿里滑块验证码会校验
**拖动轨迹的时序**（真人 0.3~1 秒完成），分 7 次调用拖 10 秒必然失败
（实测：第一次尝试被明确判「验证失败，请重试」）。

**JS 合成 MouseEvent 也不行**：`isTrusted=false`，阿里直接忽略（实测：拖完无任何反应）。

**只有 Playwright 的 `page.mouse` 能产生 trusted 事件，且可以精确控制 ~20ms 的步进。**

## 用法

    # 第一步：登录（会把登录态存到 .ms_state.json）
    python tools/ms_login.py --login

    # 之后：启动实例（自动解验证码）
    python tools/ms_login.py --start
"""
from __future__ import annotations

import argparse
import base64
import io
import random
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"E:\AI")
STATE = ROOT / ".ms_state.json"
PROFILE = ROOT / ".browser_profile"
CODE_FILE = ROOT / "runs" / "_verify" / "sms_code.txt"

#: --which 的全局值（避免在后台运行时用 input() 卡死）
_WHICH = [1]
NOTEBOOK_URL = "https://www.modelscope.cn/my/mynotebook"


def human_drag(page, box: dict, dx: float) -> None:
    """按人类轨迹拖动滑块：缓出曲线 + 轻微抖动 + 约 20ms 步进。"""
    x0 = box["x"] + box["width"] / 2
    y0 = box["y"] + box["height"] / 2
    page.mouse.move(x0, y0)
    page.wait_for_timeout(random.randint(60, 140))
    page.mouse.down()
    page.wait_for_timeout(random.randint(40, 90))

    n = 18
    for i in range(1, n + 1):
        t = i / n
        e = 1 - (1 - t) ** 2.5                       # 缓出（先快后慢，像真人）
        x = x0 + dx * e + random.uniform(-1.2, 1.2)
        y = y0 + random.uniform(-1.2, 1.2)
        page.mouse.move(x, y)
        page.wait_for_timeout(random.randint(10, 22))
    page.wait_for_timeout(random.randint(50, 120))
    page.mouse.up()


def analyze_blank(page) -> tuple[int, str]:
    """用 canvas 分析验证码图：返回 (空白区宽度, 图片URL)。"""
    js = """
    (async () => {
      const p = document.querySelector('img.puzzle');
      if (!p) return null;
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = p.src;
      await img.decode();
      const c = document.createElement('canvas');
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      const ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const d = ctx.getImageData(0, 0, c.width, c.height).data;
      const W = c.width, H = c.height;
      const stds = [];
      for (let x = 0; x < W; x++) {
        let s = 0, s2 = 0, nn = 0;
        for (let y = 0; y < H; y += 2) {
          const i = (y * W + x) * 4;
          const v = (d[i] + d[i+1] + d[i+2]) / 3;
          s += v; s2 += v * v; nn++;
        }
        const m = s / nn;
        stds.push(Math.sqrt(Math.max(0, s2 / nn - m * m)));
      }
      const mx = Math.max(...stds);
      const thr = mx * 0.15;
      const blank = stds.map(v => v < thr);
      // 找最长的一段连续空白
      let best = { len: 0, start: 0 }, cur = { len: 0, start: 0 };
      for (let i = 0; i < W; i++) {
        if (blank[i]) {
          if (cur.len === 0) cur.start = i;
          cur.len++;
          if (cur.len > best.len) best = { len: cur.len, start: cur.start };
        } else cur = { len: 0, start: 0 };
      }
      return JSON.stringify({ width: best.len, start: best.start, W, url: p.src });
    })()
    """
    r = page.evaluate(js)
    if not r:
        return 0, ""
    import json
    o = json.loads(r)
    return int(o["width"]), o["url"]


def solve_captcha(page, max_try: int = 4) -> bool:
    """解滑块验证码：拖动距离 = 被遮白区的宽度。"""
    for k in range(max_try):
        page.wait_for_timeout(1200)
        w, url = analyze_blank(page)
        print(f"    第 {k+1} 次：遮白区宽度 = {w}px  {url[-40:] if url else ''}")
        if w <= 0:
            print("    读不到验证码图，重试")
            page.wait_for_timeout(1500)
            continue

        sb = page.locator("#aliyunCaptcha-sliding-slider").bounding_box()
        if not sb:
            print("    找不到滑块")
            return False
        human_drag(page, sb, w)
        page.wait_for_timeout(2500)

        # 判定：遮罩层消失 = 通过
        ok = page.evaluate(
            "() => !document.querySelector('#aliyunCaptcha-mask.mask-show')")
        print(f"    结果：{'✓ 通过' if ok else '✗ 未通过'}")
        if ok:
            return True
        if k < max_try - 1:
            page.wait_for_timeout(1500)      # 失败后界面会自动换图
    return False


def do_login(page, phone: str) -> None:
    page.goto("https://www.modelscope.cn/login", wait_until="domcontentloaded")
    page.wait_for_timeout(3000)
    # 登录表单在跨域 iframe 里，直接打开 iframe 的地址
    src = page.evaluate(
        "() => { const f=document.querySelector('iframe'); return f?f.src:''; }")
    print("  iframe:", src[:80])
    page.goto(src, wait_until="domcontentloaded")
    page.wait_for_timeout(2000)

    page.click("text=短信登录")
    page.wait_for_timeout(800)
    page.fill('input[placeholder="请输入手机号"]', phone)
    page.click("text=获取验证码")
    print("\n>>> 验证码已发送，请把收到的 6 位数字写入该文件：")
    print(f"    {CODE_FILE}")
    CODE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CODE_FILE.exists():
        CODE_FILE.unlink()

    for _ in range(120):
        if CODE_FILE.exists():
            code = CODE_FILE.read_text(encoding="utf-8").strip()
            if code:
                break
        time.sleep(1)
    else:
        raise SystemExit("等待验证码超时")

    print("  收到验证码:", code)
    page.fill('input[placeholder="请输入验证码"]', code)
    # 勾选协议
    try:
        page.click("input.fm-agreement-checkbox", timeout=3000)
    except Exception:
        pass
    for sel in ("button.fm-button.sms-login", "button:has-text('登录')"):
        try:
            page.click(sel, timeout=3000)
            break
        except Exception:
            continue
    page.wait_for_timeout(4000)
    print("  当前 URL:", page.url[:90])

    page.goto(NOTEBOOK_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)
    text = page.inner_text("body")[:300]
    ok = "xiaoxiao384" in text or "我的Notebook" in page.title()
    print(f"  登录{'成功' if ok else '可能失败'}")
    page.context.storage_state(path=str(STATE))
    print("  登录态已存:", STATE)


def do_start(page) -> None:
    page.goto(NOTEBOOK_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)
    print("  页面标题:", page.title())

    # 三个「启动」按 DOM 顺序：CPU / GPU / AMD
    btns = page.locator("text=启动")
    n = btns.count()
    print("  找到启动按钮:", n)
    for i in range(n):
        try:
            print(f"  按钮 {i}: " + (btns.nth(i).inner_text(timeout=1000) or "").strip())
        except Exception:
            pass

    # ⚠️ 不在脚本里用 input() —— 后台运行时 stdin 是空的会卡死。
    # 默认选 GPU 环境（DOM 顺序 0=CPU 1=GPU 2=AMD GPU），用 --which 覆盖。
    which = _WHICH[0]
    print(f"  选择第 {which} 个启动")
    btns.nth(which).click()
    print("  已点击，等待验证码…")
    page.wait_for_timeout(2500)

    if page.locator("#aliyunCaptcha-sliding-slider").count():
        if solve_captcha(page):
            print("\n>>> 验证码通过")
        else:
            print("\n>>> 验证码未通过")
    else:
        print("  没有验证码，直接继续")

    page.wait_for_timeout(4000)
    page.screenshot(path=str(ROOT / "runs/_verify/ms_start.png"))
    print("  截图:", ROOT / "runs/_verify/ms_start.png")
    print("  最终 URL:", page.url[:100])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--login", action="store_true")
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--phone", default="15014046886")
    ap.add_argument("--which", type=int, default=1)
    args = ap.parse_args()
    _WHICH[0] = args.which

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE),
            headless=False,                 # ⚠️ 非无头，反检测更好
            channel="msedge",
            viewport={"width": 1440, "height": 900},
            locale="zh-CN",
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            if args.login:
                do_login(page, args.phone)
            if args.start:
                do_start(page)
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
