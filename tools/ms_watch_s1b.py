"""监控云端 1B 训练：打开 runs/s1b.log，用 Monaco 的 API 读出文本并解析。

## 方法演进（每一步都是实测踩出来的）

| 尝试 | 结果 |
|---|---|
| 点【终端】标签 + keyboard.type | ✗ 焦点被【问题】面板筛选框抢走，命令尾巴跑进筛选框 |
| Ctrl+` | ✗ 只是开关面板，不保证聚焦 |
| 截图 + 视觉识别 | ✓ 可行但每轮要人看 |
| **Ctrl+P 打开日志 + Monaco API 读文本** | ✓ **本脚本采用** |

Monaco（VS Code 的编辑器内核）在页面里暴露 `monaco.editor.getModels()`，
直接 `getValue()` 就能拿到整个文件内容 —— **不需要截图、不需要 OCR**。

用法::

    python tools/ms_watch_s1b.py --interval 1200 --rounds 18
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"E:\AI")
PROFILE = ROOT / ".browser_profile"
OUT = ROOT / "runs/_verify/monitor"
NOTEBOOK_URL = "https://www.modelscope.cn/my/mynotebook"
LOG = "runs/s1b.log"

READ_JS = """
() => {
  try {
    if (typeof monaco === 'undefined' || !monaco.editor) return null;
    const models = monaco.editor.getModels();
    if (!models.length) return null;
    // 找名字里带 s1b 的，否则取最后一个
    let m = models.find(x => (x.uri.path || '').includes('s1b')) || models[models.length - 1];
    return m.getValue();
  } catch (e) { return null; }
}
"""


def parse_tail(text: str, n: int = 6) -> dict:
    """从日志文本里抽出最后几步和验证损失。"""
    if not text:
        return {}
    lines = [l for l in text.splitlines() if l.strip()]
    steps = []
    for l in lines:
        m = re.search(r"step (\d+)/(\d+)\s+loss ([\d.]+)\s+lr ([\deE.+-]+).*?\|\s*([\d,]+) tok/s.*?ETA ([\d.]+)min", l)
        if m:
            steps.append({
                "step": int(m.group(1)), "total": int(m.group(2)),
                "loss": float(m.group(3)), "lr": m.group(4),
                "tps": int(m.group(5).replace(",", "")), "eta_min": float(m.group(6)),
            })
    vals = []
    for l in lines:
        m = re.search(r"验证损失 ([\d.]+).*?@ step (\d+)", l)
        if m:
            vals.append({"val": float(m.group(1)), "step": int(m.group(2))})
    return {
        "last": steps[-1] if steps else None,
        "recent": steps[-n:],
        "last_val": vals[-1] if vals else None,
        "n_steps_logged": len(steps),
    }


def once(ctx, page) -> dict | None:
    page.goto(NOTEBOOK_URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(8000)
    body = page.inner_text("body")
    if "运行中" not in body:
        return {"error": "实例不在运行中"}
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
    ide.wait_for_timeout(5000)

    ide.keyboard.press("Control+p")
    ide.wait_for_timeout(1500)
    ide.keyboard.insert_text(LOG)
    ide.wait_for_timeout(2500)
    ide.keyboard.press("Enter")
    ide.wait_for_timeout(7000)

    text = None
    for _ in range(3):
        try:
            text = ide.evaluate(READ_JS)
        except Exception:
            text = None
        if text:
            break
        ide.wait_for_timeout(2000)

    # Monaco API 读不到时退回截图（裁编辑器区域，只看日志正文）
    ts = datetime.now().strftime("%m%d-%H%M%S")
    shot = OUT / f"{ts}-log.png"
    try:
        ide.screenshot(path=str(shot), clip={"x": 360, "y": 70, "width": 1210, "height": 520})
    except Exception:
        try:
            ide.screenshot(path=str(shot))
        except Exception:
            shot = None

    if text:
        d = parse_tail(text)
        d["shot"] = str(shot) if shot else None
        return d
    return {"error": "Monaco API 读不到，已存截图", "shot": str(shot) if shot else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=1200)
    ap.add_argument("--rounds", type=int, default=18)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    hist = OUT / "s1b-curve.jsonl"

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE), headless=False, channel="msedge",
            viewport={"width": 1600, "height": 1000}, locale="zh-CN",
            args=["--disable-blink-features=AutomationControlled"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            for k in range(args.rounds):
                ts = datetime.now().strftime("%m%d-%H%M")
                try:
                    info = once(ctx, page) or {}
                except Exception as e:
                    info = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
                info["ts"] = ts
                print(f"[{ts}] {json.dumps(info, ensure_ascii=False)[:260]}", flush=True)
                hist.open("a", encoding="utf-8").write(json.dumps(info, ensure_ascii=False) + "\n")
                if len(ctx.pages) > 1:
                    try:
                        ctx.pages[-1].close()
                    except Exception:
                        pass
                if k < args.rounds - 1:
                    time.sleep(args.interval)
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
