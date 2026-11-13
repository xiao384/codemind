"""本地看护脚本：定时往云端终端发命令，每 50 分钟报状态，7 小时前自动安全停止训练。

## 修掉的三个 bug（原版）

| # | 原版问题 | 后果 | 修法 |
|---|---|---|---|
| 1 | `kill -INT 1202` **PID 硬编码** | 新会话 PID 变了 → kill 失败或杀错 | 改用 `pkill -INT -f "..."` 按命令行匹配 |
| 2 | `time_time >= 288`（4.8h） | 不是 7 小时 | 改成按**真实经过时间**判断，停止点可配 |
| 3 | `time_time += 50` 是"假时间" | 每次命令耗时没算进去，会漂移 | 用 `time.time()` 算真实 elapsed |

## 为什么用 `pkill -f` 而不是 PID

本项目 AGENTS.md 明确记着：**不要用排除法/硬编码 PID 杀进程** ——
曾因漏掉子进程导致看守静默死亡 11 小时。**按命令行全匹配是最稳的。**

## 用前须知

1. **必须把云端终端窗口放在最前面并点一下**（取得焦点）
2. 运行后 **5 秒开始**，这 5 秒内别动鼠标键盘
3. **pyautogui 的急停**：把鼠标猛甩到屏幕左上角会抛异常中止
4. 终端粘贴是 `Ctrl+Shift+V`（不是 Ctrl+V）

用法::

    .\\.venv\\Scripts\\python.exe tools\\cloud_watch.py
"""
from __future__ import annotations

import time
from datetime import datetime

import pyautogui
import pyperclip

# ═══════════════ 配置 ═══════════════
TOTAL_HOURS = 7.0          # 会话总时长
STOP_MARGIN_MIN = 15       # 提前多少分钟停（给保存 12 GB 检查点留时间）
REPORT_EVERY_MIN = 50      # 每多少分钟上报一次
RUN = "s1b"                # 训练名（用于 pkill 匹配）
# ═══════════════════════════════════

STOP_AT_MIN = TOTAL_HOURS * 60 - STOP_MARGIN_MIN
PAUSE = 0.8                # 每条命令之间的间隔（秒）


def send(cmd: str, wait: float = PAUSE) -> None:
    """把命令粘贴进当前焦点窗口并回车。"""
    pyperclip.copy(cmd)
    time.sleep(0.15)                   # 给剪贴板一点时间
    pyautogui.hotkey("ctrl", "shift", "v")
    time.sleep(0.15)
    pyautogui.press("enter")
    time.sleep(wait)


def main() -> int:
    print("=" * 60)
    print(f" 云端看护  总时长 {TOTAL_HOURS} 小时，"
          f"{STOP_AT_MIN/60:.2f} 小时时停止训练")
    print(f" 每 {REPORT_EVERY_MIN} 分钟上报：tail / latest.pt / ps")
    print("=" * 60)
    print("⚠️ 请现在把云端终端窗口放到最前面并点一下（5 秒后开始）")
    time.sleep(5)

    start = time.time()
    next_report = 0.0
    n = 0

    while True:
        el_min = (time.time() - start) / 60.0

        # ── 到点：安全停止 ──
        if el_min >= STOP_AT_MIN:
            print(f"\n[{datetime.now():%H:%M:%S}] 到 {el_min:.0f} 分钟"
                  f"（{el_min/60:.2f}h），发送停止命令")
            # ⚠️ 用 pkill 按命令行匹配，不用硬编码 PID
            send(f'pkill -INT -f "codemind train --run runs/{RUN}"', 2.0)
            send(f'sleep 15 && ls -l runs/{RUN}/latest.pt', 2.0)
            send(f'tail -5 runs/{RUN}.log', 2.0)
            send('echo "=== 已停止，可以关闭实例 ==="', 1.0)
            print("已发送停止序列。请回云端确认 latest.pt 已更新。")
            return 0

        # ── 定时上报 ──
        if el_min >= next_report:
            send(f'tail -3 runs/{RUN}.log')
            send(f'ls -l runs/{RUN}/latest.pt')
            send('ps aux | grep codemind')
            print(f"[{datetime.now():%H:%M:%S}] 第 {n+1} 次上报  "
                  f"已运行 {el_min:.0f} 分钟 / {STOP_AT_MIN:.0f} 分钟停止")
            n += 1
            next_report += REPORT_EVERY_MIN

        time.sleep(5)     # 每 5 秒检查一次，到点精度 ±5 秒


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n手动中止（Ctrl+C）")
    except pyautogui.FailSafeException:
        print("\n触发 pyautogui 急停（鼠标到左上角）—— 已中止")
