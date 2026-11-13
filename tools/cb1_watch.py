#!/usr/bin/env python
"""看守 runs/corpus2/latest.pt：每落一次盘就跑一遍 cb-1，把结果追加成能力曲线。

为什么需要它
------------
训练要跑 28 小时，而 goal 轮次额度覆盖不到。手动在每个检查点跑基准必然会漏。
这个看守进程脱离会话独立运行，保证能力曲线是完整的——而能力曲线才是判断
"这批数据到底有没有教会模型写代码"的唯一依据。

用法（脱离会话）：
    Start-Process -FilePath .venv\\Scripts\\python.exe `
        -ArgumentList 'tools\\cb1_watch.py' -WorkingDirectory E:\\AI `
        -RedirectStandardOutput runs\\_verify\\cb1-watch.log ...
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs" / "corpus2"
LATEST = RUN_DIR / "latest.pt"
TMP_CKPT = ROOT / "runs" / "_verify" / "ckpt-auto.pt"
CURVE = ROOT / "runs" / "_verify" / "cb1-curve.jsonl"
CTX_CURVE = ROOT / "runs" / "_verify" / "context-curve.jsonl"
CB1 = ROOT / "tools" / "cb1.py"
DIAG = ROOT / "tools" / "diag_context.py"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

POLL_SECONDS = 120
# 样本量由「最小可检出差异」反推，不是拍脑袋定的：
#   n=100 时（实测标准误）两检查点比较的最小可检出差异约
#     edit_sim 4.7pp / syntax_ok 10.5pp / rep8 8.7pp
#   而我们真正关心的变化量级是 2~5pp —— n=100 根本读不出来（曾据此误报过
#   "syntax_ok 下滑"，实际 z=-1.04 不显著）。se ∝ 1/sqrt(n)，提到 400 后
#   edit_sim 的分辨力约 2.3pp，且耗时（约 28 分钟）仍远小于 500 步的间隔（约 71 分钟）。
SAMPLES = 400
THREADS = 4
CTX_BATCHES = 25


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_steps(path: Path):
    """从检查点里读训练步数（只读元数据，避免整份载入）。"""
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload.get("iteration")
    except Exception as exc:  # 检查点可能正在被替换
        log(f"  读取步数失败（可能正在落盘）：{exc}")
        return None


def main() -> int:
    if not PY.exists():
        log(f"找不到解释器 {PY}")
        return 1
    log(f"看守启动，监视 {LATEST}，每 {POLL_SECONDS}s 轮询")
    last_mtime = None
    while True:
        try:
            if not LATEST.exists():
                time.sleep(POLL_SECONDS)
                continue
            mtime = LATEST.stat().st_mtime
            if last_mtime != mtime:
                log("检测到新的检查点，开始评估…")
                # 先拷贝，避免与训练落盘互相干扰
                TMP_CKPT.write_bytes(LATEST.read_bytes())
                steps = read_steps(TMP_CKPT)
                tag = f"auto-step{steps}"
                out_json = ROOT / "runs" / "_verify" / f"cb1-{tag}.json"
                cmd = [
                    str(PY), str(CB1),
                    "--run", str(TMP_CKPT),
                    "--tokenizer", "runs/corpus2/tokenizer.json",
                    "--samples", str(SAMPLES),
                    "--threads", str(THREADS),
                    "--tag", tag,
                    "--out", str(out_json),
                    "--quiet",
                ]
                r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
                if r.returncode == 0 and out_json.exists():
                    res = json.loads(out_json.read_text(encoding="utf-8"))
                    m = res["metrics"]
                    b = res["baselines"]
                    rec = {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "steps": res.get("train_steps"),
                        "samples": res["params"]["samples"],
                        "chance_edit_sim": b["chance_edit_sim"],
                        "chance_syntax_ok": b["chance_syntax_ok"],
                    }
                    for arm in res["params"]["arms"]:
                        rec[f"{arm}_edit_sim"] = m[arm]["edit_sim"]["mean"]
                        rec[f"{arm}_edit_sim_se"] = m[arm]["edit_sim"]["stderr"]
                        rec[f"{arm}_syntax_ok"] = m[arm]["syntax_ok"]["mean"]
                        rec[f"{arm}_rep8"] = m[arm]["rep8"]["mean"]
                        rec[f"{arm}_emitted_end"] = m[arm]["emitted_end"]["mean"]
                    with CURVE.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    log("  " + json.dumps(rec, ensure_ascii=False))
                    # ---- 上下文消融：看"在不在学"（聚合指标看不见的东西）----
                    ctx_json = ROOT / "runs" / "_verify" / f"ctx-{tag}.json"
                    cmd2 = [
                        str(PY), str(DIAG),
                        "--run", str(TMP_CKPT),
                        "--batches", str(CTX_BATCHES),
                        "--threads", str(THREADS),
                        "--tag", tag,
                        "--out", str(ctx_json),
                    ]
                    r2 = subprocess.run(cmd2, cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
                    if r2.returncode == 0 and ctx_json.exists():
                        c = json.loads(ctx_json.read_text(encoding="utf-8"))
                        crec = {
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "steps": c.get("train_steps"),
                            "hole_ppl": c.get("hole_ppl"),
                            "fim_end_loss": c.get("fim_end_loss"),
                            "context_gain_nats": c.get("context_gain_nats"),
                            "context_gain_ratio": c.get("context_gain_ratio"),
                        }
                        with CTX_CURVE.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(crec, ensure_ascii=False) + "\n")
                        log("  ctx " + json.dumps(crec, ensure_ascii=False))
                    else:
                        log(f"  context 诊断失败 rc={r2.returncode}: {r2.stderr[-300:]}")
                    last_mtime = mtime
                else:
                    log(f"  cb-1 失败 rc={r.returncode}: {r.stderr[-400:]}")
                    # 不更新 last_mtime，下轮重试
        except Exception as exc:
            log(f"轮询异常：{exc}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
