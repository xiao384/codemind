#!/usr/bin/env python
"""上下文消融诊断：验证损失下降，降的到底是"看懂了上下文"还是"背熟了边缘分布"？

做法
----
在验证分片上按训练时的 FIM 格式取样，把**上下文换成另一篇文档的真实代码**
（同分布、但内容错误），再看损失。四种条件：

  normal        原样
  suffix_wrong  后缀内容换成另一行的（长度位置都不变，仍是真实代码）
  prefix_wrong  前缀内容换成另一行的
  no_context    前后缀都换掉，模型只剩 FIM 结构词元

判读
----
* 若 normal 明显低于三个消融条件 -> 模型确实在用上下文，损失下降含真实能力。
* 若四者几乎相同 -> 模型**完全没用上下文**，它只是在背"这个位置的 token 边缘分布"
  （比如"洞的开头通常是换行/缩进"），这正好解释"损失降而能力不动"。

同时把损失拆成两段，看降的是哪一段：
  end_loss     预测 <|fim_end|> 的损失（结构词元，几乎送分）
  middle_loss  预测洞内内容的损失（真正要学的）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from codemind.data import RandomWindowSampler  # noqa: E402
from codemind.generate import load_runner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="上下文消融诊断")
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--val", default="data/corpus2/corpus2.00003.bin")
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    run_path = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    ckpt = run_path / f"{args.checkpoint}.pt" if run_path.is_dir() else run_path
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    steps = payload.get("iteration")
    del payload

    gen = load_runner(run_path, checkpoint=args.checkpoint, device="cpu")
    model, tok = gen.model, gen.tokenizer
    model.eval()
    T = gen.block_size
    val_path = Path(args.val) if Path(args.val).is_absolute() else ROOT / args.val

    sampler = RandomWindowSampler(
        [val_path], block_size=T, batch_size=args.batch_size, dtype="uint16",
        seed=99, fim_rate=1.0, fim_spm_rate=0.5, tokenizer=tok,
    )

    COND = ("normal", "suffix_wrong", "prefix_wrong", "no_context")
    acc = {c: {"end": [], "mid": [], "all": []} for c in COND}

    with torch.no_grad():
        for _ in range(args.batches):
            x, y, m = sampler.sample_batch()
            if m is None:
                print("掩码为空，说明 FIM 没生效", file=sys.stderr)
                return 2
            B = x.shape[0]
            # 定位每行的 FIM 结构：结构词元位置在同一行内保持一致，替换时不动它们
            spans = []
            for b in range(B):
                ids = x[b].tolist()
                fh = ids.index(tok.fim_hole_id) if tok.fim_hole_id in ids else -1
                fe = -1
                for i in range(len(ids) - 1, -1, -1):
                    if ids[i] == tok.fim_end_id:
                        fe = i
                        break
                spans.append((fh, fe))

            for cond in COND:
                xx = x.clone()
                for b in range(B):
                    fh, fe = spans[b]
                    if fh < 0 or fe < 0:
                        continue
                    src = x[(b + 1) % B]          # 另一篇文档的真实 token
                    if cond in ("prefix_wrong", "no_context"):
                        xx[b, 1:fh] = src[1:fh]   # 前缀区（fim_begin 之后到 fim_hole）
                    if cond in ("suffix_wrong", "no_context"):
                        xx[b, fh + 1 : fe] = src[fh + 1 : fe]   # 后缀区
                logits, _, _ = model(xx, logits_for_all=True)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    reduction="none",
                ).reshape(B, T)
                mm = m > 0
                end_mask = torch.zeros_like(mm)
                mid_mask = torch.zeros_like(mm)
                for b in range(B):
                    fh, fe = spans[b]
                    if fe > 0:
                        end_mask[b, fe - 1] = mm[b, fe - 1]      # 预测 <|fim_end|> 的位置
                    mid_mask[b, fe:] = mm[b, fe:]                # 预测洞内内容的位置
                for key, mask in (("end", end_mask), ("mid", mid_mask), ("all", mm)):
                    sel = ce[mask]
                    if sel.numel():
                        acc[cond][key].append(float(sel.mean()))

    print()
    print("=" * 76)
    print(f"上下文消融诊断  |  {run_path.name}  |  step {steps}  |  {args.tag}")
    print("=" * 76)
    print(f"  验证分片: {val_path.name}   batch {args.batch_size} × {args.batches} 批")
    print()
    print(f"  {'条件':<16}{'总损失':>12}{'<|fim_end|>段':>16}{'洞内容段':>14}")
    base = None
    for c in COND:
        a = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc[c].items()}
        if c == "normal":
            base = a
        print(f"  {c:<16}{a['all']:12.4f}{a['end']:16.4f}{a['mid']:14.4f}")
    print()
    result = {
        "tool": "diag_context",
        "run": run_path.name,
        "train_steps": steps,
        "tag": args.tag,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "val_shard": val_path.name,
        "conditions": {
            c: {k: (float(sum(v) / len(v)) if v else None) for k, v in acc[c].items()}
            for c in COND
        },
    }
    if base:
        nc = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc["no_context"].items()}
        result["hole_ppl"] = float(np.exp(base["mid"]))
        result["fim_end_loss"] = float(base["end"])
        result["context_gain_nats"] = float(nc["mid"] - base["mid"])
        result["context_gain_ratio"] = float(np.exp(nc["mid"] - base["mid"]))
        print("  与 normal 的差值（正数 = 换掉上下文后损失变大 = 模型在用上下文）：")
        for c in COND[1:]:
            a = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc[c].items()}
            print(
                f"    {c:<16} 总 {a['all']-base['all']:+.4f}   "
                f"洞内容 {a['mid']-base['mid']:+.4f}   困惑度比 {np.exp(a['mid']-base['mid']):.3f}×"
            )
        print()
        print(f"  normal 洞内容损失 {base['mid']:.4f} nats = 每 token 困惑度 {np.exp(base['mid']):.2f}")
        # ⚠️ 别把这个位置的损失当成"难度指标"。v1 格式是
        # `[fb] prefix [fh] suffix [fe] middle`，<|fim_end|> 永远紧跟后缀，
        # 取值由格式完全决定 —— 真实条件熵是 0，是个送分位置。
        # 早期这里写过"上界 5.500 nats"（那其实是 middle_len 的熵，张冠李戴），
        # 实测 step 2000 已降到 0.3084，证明上界就是 0。
        end_loss = base["end"]
        if end_loss < 1.0:
            verdict = "已学会（该位置真值就是 0，是格式决定的送分位置）"
        elif end_loss < 4.0:
            verdict = "正在学（目标 0，尚未收敛）"
        else:
            verdict = "**还没学会这条格式规则**（真值 0），属训练不足"
        print(f"  <|fim_end|> 段损失 {end_loss:.4f} nats —— {verdict}")
        print()
        print("  判读：")
        gap = nc["mid"] - base["mid"]
        if gap < 0.05:
            print("    ⚠️ 换掉全部上下文后洞内容损失几乎不变 —— 模型基本没用上下文，")
            print("       损失下降来自背边缘分布。这解释了损失与能力脱钩。")
        elif gap < 0.3:
            print("    上下文作用很弱（<0.3 nats），与能力停滞吻合。")
        else:
            print(f"    模型确实在用上下文（{gap:.3f} nats）。")

        out = (
            Path(args.out)
            if args.out
            else ROOT / "runs" / "_verify" / f"ctx-{args.tag or 'run'}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        import json as _json

        out.write_text(_json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  结果已存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
