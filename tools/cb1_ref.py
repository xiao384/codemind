"""外部模型对照片：把 `cb-1` 的协议跑在公开模型上，量出**真实差距**。

## 为什么需要它

`cb-1` 只能在**本项目内部**做纵向比较（同一模型不同步数）。
"我的模型相当于什么水平"这个问题，**必须有一个外部锚点**才有根据。
本项目已多次栽在"感觉差不多"上，所以这里只给同协议下的实测数字。

## 两个协议，回答两个不同问题

* **协议 A（native）**：各自用它被设计的方式。
  本模型用 FIM（给前缀+后缀），GPT-2 用纯前缀续写。
  → 回答「**部署出去，用户实际能拿到什么**」。

* **协议 B（matched）**：两边都只给前缀，同条件。
  → 回答「**抛开格式优势，模型本身的建模质量差多少**」。这是真正的"对标"。

⚠️ **两个协议必须分开报，不能混**：协议 A 里本模型多拿了后缀，
直接拿来比会高估自己；协议 B 里本模型的 FIM 能力用不上，会低估自己。
**只有一个数字的话，读者无从判断差异来自"模型"还是"格式"。**

## 关键设计

* **同一批文档、同一个洞位置**（复用 `cb1.py` 的 `build_samples`，保证可比）；
* **上下文都截到 512 token**（本模型的上限），避免拿 GPT-2 的 1024 窗口占便宜；
* 判据是**文本空间**的 `similarity`（`cb1.py` 同款），因此跨分词器可比；
* 同时报 `mean_gen_chars`——两个分词器产出的字符数不同，不报就没法解读。

用法::

    python tools/cb1_ref.py --ref runs/_ref/gpt2 --samples 40 --hole 96
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from difflib import SequenceMatcher  # noqa: E402

from cb1 import (  # noqa: E402
    EOT_ID, MAX_DOC, MIN_DOC, MIN_SIDE,
    build_samples, line_repetition, load_documents, parseable, repetition, similarity,
)

VERIFY = ROOT / "runs" / "_verify"
MAX_NEW = 128          # 两个模型都最多生成这么多**自己的** token
CONTEXT = 512          # 两边都截到这个上下文预算


class MyEngine:
    """本项目的模型。"""
    name = "CodeMind tiny 11M"

    def __init__(self, ckpt: str, tokenizer: str, threads: int = 4) -> None:
        import torch
        torch.set_num_threads(threads)
        from codemind.generate import load_runner
        self.gen = load_runner(Path(ckpt), device="cpu", tokenizer_path=Path(tokenizer))
        self.tok = self.gen.tokenizer

    def chars_per_token(self, text: str) -> float:
        ids = self.tok.encode(text, allow_special=False)
        return max(1.0, len(text) / max(1, len(ids)))

    def complete(self, prefix: str, n: int) -> str:
        ids = self.tok.encode(prefix, allow_special=False)[-CONTEXT:]
        text = self.tok.decode(ids, strip_special=True)
        return self.gen.complete(text, max_new_tokens=n)

    def infill(self, prefix: str, suffix: str, n: int) -> str:
        p = self.tok.encode(prefix, allow_special=False)[-CONTEXT // 2:]
        s = self.tok.encode(suffix, allow_special=False)[: CONTEXT // 2]
        return self.gen.infill(self.tok.decode(p, strip_special=True),
                               self.tok.decode(s, strip_special=True),
                               max_new_tokens=n)


class HFEngine:
    """公开的 HuggingFace 因果 LM。"""
    def __init__(self, path: str, threads: int = 4) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        self.model.eval()
        n = sum(p.numel() for p in self.model.parameters())
        self.name = f"{Path(path).name} {n/1e6:.0f}M"

    def chars_per_token(self, text: str) -> float:
        n = len(self.tok(text)["input_ids"])
        return max(1.0, len(text) / max(1, n))

    def complete(self, prefix: str, n: int) -> str:
        ids = self.tok(prefix, return_tensors="pt")["input_ids"][:, -CONTEXT:]
        with self.torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=n, do_sample=True, temperature=0.2,
                top_k=20, repetition_penalty=1.05, pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    infill = None       # 没有 FIM 训练，不能做协议 A


def metrics(gen_text: str, target: str) -> dict:
    """把 `edit_sim` 拆成 **召回 / 精度**，因为单一 ratio 会被长度污染。

    `SequenceMatcher.ratio() = 2M / (len(gen) + len(target))` —— **长度在分母**。
    实测踩过：GPT-2 输出 227 字符、本模型 73 字符，于是
    "GPT-2 绝对匹配 35.2 字符 > 本模型 27.7 字符"，但 ratio 反而是本模型高
    （15.75% vs 18.88%）。**只看 ratio 会把"啰嗦"读成"更差"、把"话少"读成"更好"。**

    拆开之后：

    * `cover` = M / len(target) —— **召回**：目标有多少被复现（不在乎输出多长）
    * `prec`  = M / len(gen)    —— **精度**：输出有多少是对的
    * `edit_sim` 保留原定义，**以便与 `cb-1` 的历史曲线对齐**

    三个一起报，缺任何一个都会被误读。
    """
    sm = SequenceMatcher(None, gen_text, target, autojunk=False)
    m = sum(b.size for b in sm.get_matching_blocks())
    return {
        "edit_sim": similarity(gen_text, target),
        "cover": (m / len(target)) if target else 0.0,
        "prec": (m / len(gen_text)) if gen_text else 0.0,
        "m_chars": float(m),
        "tgt_chars": float(len(target)),
        "syntax_ok": 1.0 if parseable(gen_text) else 0.0,
        "rep8": repetition(gen_text),
        "rep_line": line_repetition(gen_text),
        "gen_chars": float(len(gen_text)),
    }


def agg(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="HF 模型目录")
    ap.add_argument("--run", default="runs/_verify/ckpt-auto.pt", help="本模型检查点")
    ap.add_argument("--tokenizer", default="data/corpus2/tokenizer.json")
    ap.add_argument("--val", default="data/corpus2/corpus2.00003.bin")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--hole", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hole-chars", type=int, default=220,
                    help="长度匹配的字符预算（默认取 96 token 洞的典型字符数）")
    ap.add_argument("--tag", default="ref")
    args = ap.parse_args()

    from codemind.tokenizer import BPETokenizer
    model_tok = BPETokenizer.load(ROOT / args.tokenizer)
    shard_tok = BPETokenizer.load(ROOT / args.tokenizer)
    arr, bounds = load_documents(ROOT / args.val, args.seed)
    samples, scanned = build_samples(arr, bounds, shard_tok, model_tok,
                                     args.samples, args.seed, args.hole)
    print(f"[ref] 扫描 {scanned} 篇，取 {len(samples)} 篇（挖空 {args.hole}）")

    mine = MyEngine(str(ROOT / args.run), str(ROOT / args.tokenizer), args.threads)
    ref = HFEngine(args.ref, args.threads)
    print(f"[ref] 本模型 : {mine.name}")
    print(f"[ref] 对照   : {ref.name}")
    print("-" * 78)

    rng = random.Random(args.seed)
    rows: dict[str, list[dict]] = {"mine_A": [], "mine_B": [], "ref_B": []}
    # 长度匹配：判据 `SequenceMatcher.ratio() = 2M/(len(a)+len(b))` **把长度放进分母**，
    # 所以"生成得多"会被直接惩罚。实测 GPT-2 在同样 128 token 下产出 645 字符、
    # 本模型只有 194 —— 不匹配长度的话，比的是分词器而不是模型。
    # 做法：按各自实测的「字符/token」把 max_new_tokens 折算成**相同的字符预算**。
    cpt_mine = mine.chars_per_token(samples[0]["prefix_text"]) if samples else 2.4
    cpt_ref = ref.chars_per_token(samples[0]["prefix_text"]) if samples else 4.0
    budget_chars = args.hole_chars
    n_mine = max(8, round(budget_chars / cpt_mine))
    n_ref = max(8, round(budget_chars / cpt_ref))
    print(f"[ref] 长度匹配: 目标 {budget_chars} 字符  →  本模型 {n_mine} token "
          f"({cpt_mine:.2f} 字符/token) / 对照 {n_ref} token ({cpt_ref:.2f} 字符/token)")

    t0 = time.perf_counter()
    for i, s in enumerate(samples, 1):
        doc, hs = s["doc"], s["hole_start"]
        prefix_text = s["prefix_text"]
        target = model_tok.decode(doc[hs : hs + args.hole], strip_special=True)
        suffix_text = model_tok.decode(doc[hs + args.hole :], strip_special=True)
        if not target.strip():
            continue
        # 协议 A —— 各自最佳形态
        rows["mine_A"].append(metrics(mine.infill(prefix_text, suffix_text, n_mine), target))
        # 协议 B —— 同条件：只给前缀 + 同长度预算
        rows["mine_B"].append(metrics(mine.complete(prefix_text, n_mine), target))
        rows["ref_B"].append(metrics(ref.complete(prefix_text, n_ref), target))
        if i % 5 == 0 or i == len(samples):
            print(f"  … {i}/{len(samples)}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    print("-" * 78)
    print(f"{'协议/模型':30}{'edit_sim':>10}{'cover':>9}{'prec':>8}{'匹配字':>8}{'生成字':>8}{'目标字':>8}{'rep8':>8}")
    label = {
        "mine_A": f"A 各自最佳  {mine.name}",
        "mine_B": f"B 同条件    {mine.name}",
        "ref_B":  f"B 同条件    {ref.name}",
    }
    out = {}
    for key in ("mine_A", "mine_B", "ref_B"):
        a = agg(rows[key])
        out[key] = {"label": label[key], "n": len(rows[key]), **a}
        print(f"{label[key]:30}{a['edit_sim']*100:9.2f}%{a['cover']*100:8.2f}%"
              f"{a['prec']*100:7.2f}%{a['m_chars']:8.0f}{a['gen_chars']:8.0f}"
              f"{a['tgt_chars']:8.0f}{a['rep8']*100:7.1f}%")

    print()
    for key in ('edit_sim', 'cover'):
        d = out['mine_B'][key] - out['ref_B'][key]
        va = float(np.var([r[key] for r in rows['mine_B']], ddof=1)) / len(rows['mine_B'])
        vb = float(np.var([r[key] for r in rows['ref_B']], ddof=1)) / len(rows['ref_B'])
        se = float(np.sqrt(va + vb))
        verdict = '显著' if abs(d/se) >= 1.96 else '不显著'
        who = '本模型更好' if d > 0 else '外部模型更好'
        note = '（长度稳健）' if key == 'cover' else '（受长度影响）'
        print(f"  协议B {key:9}{note}: 差距 {d*100:+6.2f}pp  se {se*100:5.2f}pp  "
              f"z={d/se:+5.2f}  {verdict}  [{who}]")

    (VERIFY / f"cb1-{args.tag}.json").write_text(
        json.dumps({"n": len(samples), "hole": args.hole, "seed": args.seed,
                    "results": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[ref] 已写 runs/_verify/cb1-{args.tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
