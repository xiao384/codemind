"""关键检查点的完整评估协议 —— 一条命令产出可交付的证据包。

为什么需要它：
  * 单点 cb-1 的 n=100 分辨力只有 4.7pp（edit_sim）/10.5pp（syntax_ok），
    读不出小变化。关键判据点必须用更大的 n。
  * 「能不能用」（cb-1）、「在不在学」（上下文消融）、「horizon 有多长」（难度扫描）
    是三件不同的事，只跑其中一个会得出片面结论。
  * 训练落盘会覆盖 latest.pt，且看守会覆盖 ckpt-auto.pt —— 必须先冻结检查点。

用法：
    python tools/key_eval.py --steps 4188 --samples 1000
    python tools/key_eval.py --steps 13428 --samples 2000 --ckpt runs/corpus2/best.pt

产出（全部落在 runs/_verify/）：
    ckpt-stepNNNN.pt            冻结的检查点（不会被后续覆盖）
    cb1-key-stepNNNN.json       cb-1 主表（双解码臂 + 机会基线）
    cb1-sweep-stepNNNN.json     难度扫描 holes=8,24,48,96
    ctx-stepNNNN.json           上下文消融
    key-eval-stepNNNN.txt       人可读的汇总
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
VERIFY = ROOT / "runs" / "_verify"
TOK = ROOT / "runs" / "corpus2" / "tokenizer.json"


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'=' * 70}\n[{label}] {' '.join(str(c) for c in cmd)}\n{'=' * 70}", flush=True)
    t0 = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    print(f"[{label}] 耗时 {time.time() - t0:.0f}s，退出码 {proc.returncode}", flush=True)
    return proc.returncode


def pct(x, nd=2):
    return f"{x * 100:.{nd}f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True, help="检查点步数（用于命名）")
    ap.add_argument("--ckpt", default="runs/corpus2/latest.pt", help="要评估的检查点")
    ap.add_argument("--samples", type=int, default=1000, help="cb-1 主表样本量")
    ap.add_argument("--sweep-samples", type=int, default=200, help="难度扫描样本量")
    ap.add_argument("--batches", type=int, default=60, help="上下文消融 batch 数")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    tag = f"step{args.steps}"
    frozen = VERIFY / f"ckpt-{tag}.pt"
    src = ROOT / args.ckpt
    if not src.exists():
        print(f"检查点不存在: {src}", file=sys.stderr)
        return 2

    VERIFY.mkdir(parents=True, exist_ok=True)
    if not frozen.exists():
        shutil.copy2(src, frozen)
        print(f"已冻结检查点 -> {frozen.relative_to(ROOT)}")
    else:
        print(f"复用已冻结的检查点 {frozen.relative_to(ROOT)}"
              f"（想重取请先删除它）")

    t_start = time.time()
    main_json = VERIFY / f"cb1-key-{tag}.json"
    rc = run([PY, "tools/cb1.py", "--run", frozen, "--tokenizer", TOK,
              "--samples", args.samples, "--threads", args.threads,
              "--tag", tag, "--out", main_json, "--quiet"], "cb1 主表")
    if rc != 0:
        return rc

    sweep_json = VERIFY / f"cb1-sweep-{tag}.json"
    if not args.skip_sweep:
        run([PY, "tools/cb1.py", "--run", frozen, "--tokenizer", TOK,
             "--samples", args.sweep_samples, "--holes", "8,24,48,96",
             "--threads", args.threads, "--tag", f"{tag}-sweep",
             "--out", sweep_json, "--quiet"], "难度扫描")

    ctx_json = VERIFY / f"ctx-{tag}.json"
    run([PY, "tools/diag_context.py", "--run", frozen, "--batches", args.batches,
         "--threads", args.threads, "--tag", tag, "--out", ctx_json], "上下文消融")

    # ---------------------------------------------------------------- 汇总
    out = [f"关键检查点评估 @ {tag}",
           f"检查点: {frozen.relative_to(ROOT)}  样本量: {args.samples}",
           f"总耗时: {time.time() - t_start:.0f}s", ""]
    if main_json.exists():
        d = json.loads(main_json.read_text(encoding="utf-8"))
        b = d.get("baselines", {})
        out.append("--- cb-1 主表（product 臂 = codemind fim 实际行为）---")
        out.append(f"{'指标':<14}{'模型':>10}{'机会基线':>12}{'余量':>10}")
        for k in ("edit_sim", "syntax_ok", "first_tok", "rep8", "rep_line", "emitted_end"):
            m = d["metrics"]["product"].get(k)
            if not m:
                continue
            mean = m["mean"]
            base = b.get(f"chance_{k}")
            if base is None:
                out.append(f"{k:<14}{pct(mean):>10}{'—':>12}{'—':>10}")
            else:
                out.append(f"{k:<14}{pct(mean):>10}{pct(base):>12}"
                           f"{f'{(mean - base) * 100:+.2f}pp':>10}")
        out.append("")
        out.append(f"生成长度 {d['metrics']['product'].get('mean_gen_chars', 0):.1f} 字符"
                   f"（目标 96 token ≈ 234 字符）")
        out.append(f"验证损失见训练日志（本脚本不读）")
    if ctx_json.exists():
        c = json.loads(ctx_json.read_text(encoding="utf-8"))
        out.append("")
        out.append("--- 上下文消融（看「在不在学」）---")
        out.append(f"  hole_ppl        {c['hole_ppl']:.2f}")
        out.append(f"  fim_end_loss    {c['fim_end_loss']:.4f}")
        out.append(f"  context_gain    {c['context_gain_nats']:+.4f} nats"
                   f"（×{c['context_gain_ratio']:.3f}）")
    if sweep_json.exists():
        s = json.loads(sweep_json.read_text(encoding="utf-8"))
        out.append("")
        out.append("--- 难度扫描（看 horizon 有多长）---")
        for hole, blk in s.get("by_hole", {}).items():
            a, bb = blk["arms"]["product"], blk["baselines"]
            out.append(f"  hole {hole:>3}: first_tok {pct(a['first_tok']['mean'])}"
                       f"  edit_sim {pct(a['edit_sim']['mean'])}"
                       f"（机会 {pct(bb['chance_edit_sim'])}）"
                       f"  rep8 {pct(a['rep8']['mean'], 1)}")

    text = "\n".join(out)
    (VERIFY / f"key-eval-{tag}.txt").write_text(text, encoding="utf-8")
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
