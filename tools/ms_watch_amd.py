"""轮询 ModelScope Notebook 页，检测 AMD GPU 实例是否从「已失败」恢复可用。

## 背景

2026-10-31 实测：点 AMD 的「启动」时报

    创建notebook失败，信息：当前免费资源已经用满，请等待，或者点击创建实例按钮进行购买

**注意区分两件事**（容易混）：

| 概念 | 实测值 | 含义 |
|---|---|---|
| **剩余额度** | AMD **71 小时 57 分** | 你还能用多少小时 |
| **当前可用性** | **已失败** | 现在有没有空闲机器给你 |

额度够但机器满了 —— 所以要**等资源释放**。本脚本就是盯这个的。

## 用法

    # 后台跑，每 5 分钟查一次，状态变化时写日志（不弹窗）
    python tools/ms_watch_amd.py --interval 300

日志：`runs/_verify/ms-amd-status.log`（每行一条时间戳记录）
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(r"E:\AI")
PROFILE = ROOT / ".browser_profile"
LOG = ROOT / "runs/_verify/ms-amd-status.log"
NOTEBOOK_URL = "https://www.modelscope.cn/my/mynotebook"


def read_status(page) -> dict:
    """从页面读出三个实例的状态与额度。"""
    js = """
    () => {
      // 找包含「CPU环境」「GPU环境」「AMD GPU环境」的行
      const rows = Array.from(document.querySelectorAll('tr, [class*=row], [class*=table] > *'))
        .filter(e => (e.innerText || '').includes('环境'));
      const out = [];
      for (const r of rows) {
        const t = (r.innerText || '').replace(/\\s+/g, ' ').trim();
        if (t.length > 300) continue;
        let name = '';
        if (t.includes('AMD')) name = 'AMD';
        else if (t.includes('GPU环境')) name = 'GPU';
        else if (t.includes('CPU环境')) name = 'CPU';
        if (!name) continue;
        const failed = t.includes('已失败') || t.includes('启动失败');
        const stopped = t.includes('已停止') || t.includes('已停');
        const running = t.includes('运行中');
        const m = t.match(/(\\d+)\\s*小时\\s*(\\d+)\\s*分钟/);
        out.push({
          name, failed, stopped, running,
          quota: m ? `${m[1]}h${m[2]}m` : (t.includes('长期有效') ? '长期' : ''),
          raw: t.slice(0, 160),
        });
      }
      return JSON.stringify(out);
    }
    """
    try:
        return json.loads(page.evaluate(js) or "[]")
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--max-hours", type=float, default=24.0)
    args = ap.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    t_end = time.time() + args.max_hours * 3600
    last_sig = ""

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE), headless=True, channel="msedge",
            viewport={"width": 1440, "height": 900}, locale="zh-CN",
            args=["--disable-blink-features=AutomationControlled"])
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            while time.time() < t_end:
                ts = datetime.now().strftime("%m-%d %H:%M:%S")
                try:
                    page.goto(NOTEBOOK_URL, wait_until="domcontentloaded", timeout=45000)
                    page.wait_for_timeout(6000)
                    st = read_status(page)
                    amd = next((x for x in st if x["name"] == "AMD"), None)
                    if not amd:
                        line = f"[{ts}] 读不到 AMD 状态（可能页面结构变了或未登录）"
                        print(line, flush=True)
                        LOG.open("a", encoding="utf-8").write(line + "\n")
                    else:
                        sig = f"{amd['failed']}|{amd['stopped']}|{amd['running']}|{amd['quota']}"
                        line = (f"[{ts}] AMD: failed={amd['failed']} "
                                f"stopped={amd['stopped']} running={amd['running']} "
                                f"额度={amd['quota']}")
                        print(line, flush=True)
                        if sig != last_sig:
                            LOG.open("a", encoding="utf-8").write(line + "\n")
                            if last_sig and not amd["failed"]:
                                note = f"[{ts}] *** AMD 状态变了！可以尝试启动 ***"
                                print(note, flush=True)
                                LOG.open("a", encoding="utf-8").write(note + "\n")
                            last_sig = sig
                except Exception as e:
                    msg = f"[{ts}] 出错 {type(e).__name__}: {str(e)[:80]}"
                    print(msg, flush=True)
                    LOG.open("a", encoding="utf-8").write(msg + "\n")
                time.sleep(args.interval)
        finally:
            ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
