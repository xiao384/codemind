"""用**确定性哈希种子**重新校验任务池，丢掉不合格的任务。

## 为什么必须重验

建池时的判据依赖 `PYTHONHASHSEED`，而它是随机的 —— 于是
**依赖 set/dict 遍历顺序的 doctest 会时过时不过**，恰好蒙混过关。

实测：`sympy/solvers.py:denoms` 固定种子后**空体稳定通过**，
也就是说它**根本没有判别力**，却留在了 155 个任务里。

重验比重新建池快得多：**155 × 4 次子进程 ≈ 15 分钟**，
而重建池要 2.7 小时。

用法::

    python tools/revalidate_tasks.py            # 重验并原地更新
    python tools/revalidate_tasks.py --dry-run  # 只看会丢哪些
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from cb2 import _build_module, run_doctest  # noqa: E402

TASKS = ROOT / "runs" / "_verify" / "cb2-tasks.json"
TRIVIAL = ("pass", "return 0", "return None")


def _oracle_body(t: dict) -> str:
    """取任务的原函数体。

    🔴 **绝对不要在这里用 `_body_of`。** `oracle_body` **已经是函数体**，
    而 `_body_of` 期望的是**完整函数源码**（它以 `tree.body[0]` 是不是 `FunctionDef` 做断言）。
    喂错了会走到 `assert` 失败分支、**静默返回 `"pass"`** ——
    于是"原函数体"全变成空体、全部判定失败、**把整个任务池清空**。

    实测踩过：155 个任务被这个错误砍到 **1 个**，而且脚本**已经把结果写回磁盘**。
    这个错误在本项目已经犯过 **3 次**（cb2 建池、FIM 组装验证、本次），
    所以在这里留一条显式警告。
    """
    return t["oracle_body"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=float, default=6.0)
    ap.add_argument("--repeats", type=int, default=2,
                    help="重复几次才算稳定（默认 2，抓间歇性失败）")
    args = ap.parse_args()

    payload = json.loads(TASKS.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    print(f"[revalidate] 池内 {len(tasks)} 个任务，每个判据跑 {args.repeats} 次")

    good, dropped = [], []
    for i, t in enumerate(tasks, 1):
        label = f"{Path(t['file']).name}:{t['name']}"

        # ① 原函数体必须**每次都过**
        r_ok = all(run_doctest(_build_module(t, _oracle_body(t)),
                               t["imports"], args.timeout)["ok"]
                   for _ in range(args.repeats))
        if not r_ok:
            dropped.append((label, "原函数体不稳定/不过"))
            print(f"  ✗ {label}: 原函数体不过", flush=True)
            continue

        # ② 三个平凡体必须**每次都不过**
        bad = []
        for b in TRIVIAL:
            if any(run_doctest(_build_module(t, b), t["imports"], args.timeout)["ok"]
                   for _ in range(args.repeats)):
                bad.append(b)
        if bad:
            dropped.append((label, f"无判别力（{bad[0]} 能过）"))
            print(f"  ✗ {label}: 无判别力，{bad} 能过", flush=True)
            continue

        good.append(t)
        if i % 25 == 0:
            print(f"  … {i}/{len(tasks)}  留 {len(good)}  丢 {len(dropped)}", flush=True)

    print(f"\n[revalidate] 保留 {len(good)} / 丢弃 {len(dropped)}")
    for label, why in dropped:
        print(f"    丢弃 {label}: {why}")

    if not args.dry_run and good:
        payload["tasks"] = good
        payload["n_tasks"] = len(good)
        payload["revalidated"] = True
        TASKS.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        print(f"[revalidate] 已写回 {TASKS}（{len(good)} 个任务）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
