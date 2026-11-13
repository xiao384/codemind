#!/usr/bin/env python
"""cb-1：CodeMind 代码能力基准（Fill-in-the-Middle 中间填空）。

为什么需要它
------------
在此之前，项目的"能力"只有两个证据：验证损失，和肉眼看生成结果。
损失不等于能力，肉眼看不出趋势。要追求"极致的代码能力"，必须先有一个
能报出数字、能跨版本比较的指标。

任务定义
--------
从**留出的验证分片**（训练从未见过）里取出**完整的源文件文档**
（分片用 ``<|endoftext|>`` 分隔文档），在文档内部挖掉一段连续 token，
按训练时同样的 PSM 格式给出
``<|fim_begin|> prefix <|fim_hole|> suffix <|fim_end|>``，让模型补回来。
这正是 IDE 里"光标在文件中间"的真实场景。

挖空长度扫描（``--holes``）
--------------------------
单个 96-token 的洞对 11M 模型是很高的要求，可能掩盖早期进步。
``--holes 8,24,96`` 会在**同一批文档、同一个起始位置**上挖不同长度的洞
（嵌套关系），于是"能力随挖空长度如何衰减"就成了一条曲线。
**区分"没学到"和"任务太难"，靠的就是它。**

两个解码臂（都要看）
--------------------
* ``greedy``  温度 0、无重复惩罚。确定性，暴露模型的**众数**。
  实测：小模型在贪心下一律退化成重复循环（``aten.aten.aten…``、几十个空行），
  所以这个臂的数字**不代表产品能力**，但它是可复现的下界。
* ``product`` 温度 0.2 / top_k 20 / 重复惩罚 1.05 —— 与 ``Generator.infill``
  默认值一致，也就是 ``codemind fim`` 真正跑的东西。**判断产品能力看这一列。**

指标
----
* ``first_tok``  补出的**第一个 token** 是否命中原文。最敏感的指标——
                 "知不知道接下来该写什么"就体现在这里。
* ``tok_acc``    逐 token 准确率（按较短者对齐）。比字符相似度敏感得多。
* ``edit_sim``   片段级字符相似度，连续可分，与历史数据可比。
* ``exact``      片段逐字符完全相同。
* ``first_line`` 片段第一行非空内容相同。
* ``syntax_ok``  把补出的片段拼回**完整文档**后 ``ast.parse`` 通过。
* ``rep8``       重复度 = 1 − 唯一 8-gram 占比，趋近 1 = 复读机。
* ``chance``     机会基线：拿另一个样本的目标片段来比对。

用法
----
    # 单点（看守进程用这个）
    .venv\\Scripts\\python.exe tools\\cb1.py --run runs/corpus2 --tag new

    # 难度扫描
    .venv\\Scripts\\python.exe tools\\cb1.py --run runs/corpus2 \\
        --holes 8,24,48,96 --samples 50 --tag sweep --show 2
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import statistics
import sys
import time
import warnings
from difflib import SequenceMatcher
from pathlib import Path

# 被解析的文档里常有 `\d` 之类的无效转义，ast.parse 会吐 SyntaxWarning 污染 stderr，
# 让调用方误以为基准失败了。这里只静音这一类告警。
warnings.simplefilter("ignore", SyntaxWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from codemind.generate import load_runner  # noqa: E402
from codemind.model import SamplingParams  # noqa: E402
from codemind.tokenizer import BPETokenizer  # noqa: E402

EOT_ID = 256
DEFAULT_HOLE = 96       # 默认挖空长度，与历史数据一致
MIN_SIDE = 32           # 前后缀各自至少保留的 token 数
MIN_DOC = 200
MAX_DOC = 12000

ARMS = {
    "greedy": dict(temperature=0.0, greedy=True, repetition_penalty=1.0, seed=20261004),
    "product": dict(temperature=0.2, top_k=20, repetition_penalty=1.05, seed=20261004),
}


def parse_arm_specs(spec: str) -> dict[str, dict]:
    """把 ``--arm-spec`` 解析成额外的解码臂。

    格式：``名字:temperature=0.1,top_k=5,rep=1.0``，多个臂用 ``;`` 分隔。
    ``rep`` 是 ``repetition_penalty`` 的简写。

    为什么需要它：原先 ``ARMS`` 是写死的两个臂，**解码参数根本无法被测量**。
    而 ``codemind fim`` 的默认值（= product 臂）是否有别的最优选择，
    是个直接影响产品可用性的问题，必须能被实验。
    """
    extra: dict[str, dict] = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        name, kv = chunk.split(":", 1)
        name = name.strip()
        params = dict(ARMS["product"])      # 以 product 为底，只覆盖给出的项
        for pair in kv.split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            k, v = (t.strip() for t in pair.split("=", 1))
            k = "repetition_penalty" if k == "rep" else k
            params[k] = float(v) if "." in v or "e" in v.lower() else int(v)
        if params.get("temperature", 1.0) == 0.0:
            params["greedy"] = True
        extra[name] = params
    return extra
METRIC_KEYS = (
    "first_tok",
    "tok_acc",
    "edit_sim",
    "exact",
    "first_line",
    "syntax_ok",
    "rep8",
    "rep_line",
)


def load_documents(val_path: Path, seed: int):
    arr = np.memmap(val_path, dtype=np.uint16, mode="r")
    eot = np.flatnonzero(arr == EOT_ID)
    bounds = []
    prev = 0
    for pos in eot:
        if pos > prev:
            bounds.append((prev, int(pos)))
        prev = int(pos) + 1
    if prev < arr.size:
        bounds.append((prev, int(arr.size)))
    rng = random.Random(seed)
    rng.shuffle(bounds)
    return arr, bounds


def decode_ids(tok, ids) -> str:
    return tok.decode([int(i) for i in ids], strip_special=True)


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def parseable(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError, MemoryError):
        return False


def repetition(text: str, n: int = 8) -> float:
    """1 - 唯一 n-gram 占比（字符级）。趋近 1 = 复读机。

    **必须带语料基线读**：代码里的缩进空白让 8-gram 天然高频，
    实测 corpus2 的文档自身 rep8 均值就有 **44.2%**（行级仅 13.7%）。
    所以"rep8 = 55%"只比数据自身高 12pp，不等于"一半输出在复读"。
    """
    if len(text) < n:
        return 0.0
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def line_repetition(text: str) -> float:
    """1 - 唯一非空行占比。语料基线 13.7%。

    与 rep8 互补：rep8 抓得到"几十个连续空行"这类空白退化，
    rep_line 抓得到"同一行代码复读"。两个都要看。
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


# 语料自身的重复度基线（实测 200 个文档，见 AGENTS.md）
CORPUS_REP8 = 43.6
CORPUS_REP_LINE = 12.6


def build_samples(arr, bounds, shard_tok, model_tok, n, seed, hole_max):
    """挑出 n 个「原文可解析」的文档，并在其中挖一个**嵌套**的洞。

    洞的起点对所有长度都一样，所以 holes=8/24/96 得到的是同一位置、
    逐层变长的洞 —— 这样不同长度之间的结果才可比。
    """
    rng = random.Random(seed + 1)
    samples = []
    scanned = 0
    for start, end in bounds:
        if len(samples) >= n:
            break
        scanned += 1
        length = end - start
        if length < MIN_DOC or length > MAX_DOC:
            continue
        doc_text = shard_tok.decode([int(v) for v in arr[start:end]], strip_special=True)
        if not parseable(doc_text):
            continue
        doc = model_tok.encode(doc_text, allow_special=False)
        if len(doc) < hole_max + 2 * MIN_SIDE:
            continue
        hole_start = rng.randint(MIN_SIDE, len(doc) - hole_max - MIN_SIDE)
        samples.append(
            {
                "doc": doc,
                "hole_start": hole_start,
                "prefix_text": model_tok.decode(doc[:hole_start], strip_special=True),
            }
        )
    return samples, scanned


def make_view(sample, hole, block_size):
    """按挖空长度切出喂给模型的上下文窗口。"""
    doc, hs = sample["doc"], sample["hole_start"]
    budget = block_size - 3 - hole
    prefix = doc[:hs]
    suffix = doc[hs + hole :]
    if len(prefix) + len(suffix) > budget:
        half = max(1, budget // 2)
        pv = prefix[-half:]
        sv = suffix[: budget - half]
    else:
        pv, sv = prefix, suffix
    return pv, sv, doc[hs : hs + hole], suffix


def run_arm(gen, tok, sample, hole, params, block_size, suffix_text, want):
    pv, sv, target, _ = make_view(sample, hole, block_size)
    prompt_ids = [tok.fim_begin_id] + pv + [tok.fim_hole_id] + sv + [tok.fim_end_id]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=gen.device)
    with torch.no_grad():
        out = gen.model.generate(
            idx,
            max_new_tokens=hole,
            params=params,
            stop_ids=(tok.fim_end_id, tok.eot_id),
            use_cache=True,
        )
    new_ids = [int(v) for v in out[0].tolist()[len(prompt_ids):]]
    emitted_end = bool(new_ids) and new_ids[-1] in (tok.fim_end_id, tok.eot_id)
    if emitted_end:
        new_ids = new_ids[:-1]
    got = decode_ids(tok, new_ids)
    first_tok = float(bool(new_ids) and bool(target) and new_ids[0] == target[0])
    k = min(len(new_ids), len(target))
    tok_acc = float(sum(1 for i in range(k) if new_ids[i] == target[i]) / k) if k else 0.0
    return {
        "first_tok": first_tok,
        "tok_acc": tok_acc,
        "edit_sim": similarity(got, want),
        "exact": float(got == want),
        "first_line": float(
            bool(first_nonempty_line(got))
            and first_nonempty_line(got) == first_nonempty_line(want)
        ),
        "syntax_ok": float(parseable(sample["prefix_text"] + got + suffix_text)),
        "rep8": repetition(got),
        "rep_line": line_repetition(got),
        "emitted_end": float(emitted_end),
        "gen_len": len(got),
    }, got, want, list(target)


def main() -> int:
    ap = argparse.ArgumentParser(description="cb-1 代码能力基准（中间填空）")
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--val", default="data/corpus2/corpus2.00003.bin")
    ap.add_argument("--val-tokenizer", default="data/corpus2/tokenizer.json")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--holes", default=str(DEFAULT_HOLE), help="挖空长度，逗号分隔，如 8,24,96")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu / cuda（1B 在 CPU 上极慢，务必用 cuda）")
    ap.add_argument("--arms", default="greedy,product", help="要跑的解码臂，逗号分隔")
    ap.add_argument(
        "--arm-spec", default="",
        help="自定义解码臂：'名字:temperature=0.1,top_k=5,rep=1.0'，多个用 ';' 分隔。"
             "给定时默认只跑这些臂（除非同时显式给了 --arms）。",
    )
    ap.add_argument(
        "--nbest",
        type=int,
        default=1,
        help="对 product 臂采样 N 个候选，再按验证器择优（推理期脚手架实验）",
    )
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    holes = sorted({int(h) for h in args.holes.split(",") if h.strip()})
    holes = [h for h in holes if h >= 4]
    if not holes:
        print("--holes 至少要有一个 >= 4 的值", file=sys.stderr)
        return 2

    torch.set_num_threads(max(1, args.threads))
    val_path = Path(args.val) if Path(args.val).is_absolute() else ROOT / args.val
    if not val_path.exists():
        print(f"找不到验证分片: {val_path}", file=sys.stderr)
        return 2
    run_path = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    ckpt = run_path / f"{args.checkpoint}.pt" if run_path.is_dir() else run_path
    if not ckpt.exists():
        print(f"找不到检查点: {ckpt}", file=sys.stderr)
        return 2
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    steps = payload.get("iteration")
    del payload

    gen = load_runner(
        run_path,
        checkpoint=args.checkpoint,
        # ?? 原来硬编码 "cpu" —— 1B 模型在 8 核 CPU 上跑 50 个样本要 40+ 分钟。
        # 改成可配置，GPU 上快 20~50 倍。
        device=args.device,
        tokenizer_path=Path(args.tokenizer) if args.tokenizer else None,
    )
    tok = gen.tokenizer
    block_size = gen.block_size

    arr, bounds = load_documents(val_path, args.seed)
    val_tok_path = (
        Path(args.val_tokenizer)
        if Path(args.val_tokenizer).is_absolute()
        else ROOT / args.val_tokenizer
    )
    shard_tok = BPETokenizer.load(val_tok_path)
    samples, scanned = build_samples(
        arr, bounds, shard_tok, tok, args.samples, args.seed, max(holes)
    )
    if len(samples) < args.samples:
        print(f"提示：只凑到 {len(samples)} 个可解析文档（扫描 {scanned} 个）")

    extra_arms = parse_arm_specs(args.arm_spec)
    if extra_arms:
        ARMS.update(extra_arms)
        if not args.arms or args.arms == "greedy,product":
            args.arms = ",".join(extra_arms)
    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]
    nbest = max(1, args.nbest)
    use_nbest = nbest > 1 and "product" in arms
    STRATS = ("first", "syntax", "oracle")
    nbest_rows: dict[str, list] = {s: [] for s in STRATS} if use_nbest else {}
    t0 = time.perf_counter()
    by_hole: dict[str, dict[str, list]] = {}
    targets_by_hole: dict[str, list] = {}
    examples: list[dict] = []
    for h in holes:
        by_hole[str(h)] = {a: [] for a in arms}
        targets_by_hole[str(h)] = []
    for i, s in enumerate(samples):
        for h in holes:
            key = str(h)
            _pv, _sv, target_ids, full_suffix = make_view(s, h, block_size)
            want = decode_ids(tok, target_ids)
            suffix_text = decode_ids(tok, full_suffix)
            arm_out = {}
            for a in arms:
                m, got, _w, _t = run_arm(
                    gen, tok, s, h, SamplingParams(**ARMS[a]), block_size,
                    suffix_text, want,
                )
                m["i"] = i
                by_hole[key][a].append(m)
                arm_out[a] = got
            targets_by_hole[key].append((want, list(target_ids)))

            # ---- best-of-N + 验证器择优（推理期脚手架，只作用于 product 臂）----
            if use_nbest:
                cands = []
                for k in range(nbest):
                    p = dict(ARMS["product"])
                    p["seed"] = int(p["seed"]) + k
                    m, got, _w, _t = run_arm(
                        gen, tok, s, h, SamplingParams(**p), block_size, suffix_text, want
                    )
                    cands.append((got, m))
                picks = {
                    # 基线：直接取第一个候选
                    "first": cands[0],
                    # 可部署的验证器：能保持整份文档可解析就选它，都不行则退回第一个
                    "syntax": next((c for c in cands if c[1]["syntax_ok"] > 0), cands[0]),
                    # 上界（不可部署，需要答案）：按 edit_sim 挑最好的
                    "oracle": max(cands, key=lambda c: c[1]["edit_sim"]),
                }
                for strat, (_got, m) in picks.items():
                    nbest_rows[strat].append(m)
            if len(examples) < 8 and h == max(holes):
                ex = {
                    "i": i,
                    "hole": h,
                    "prefix_tail": s["prefix_text"][-200:],
                    "target": want,
                }
                ex.update(arm_out)
                examples.append(ex)
        if not args.quiet and (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{len(samples)}", flush=True)
    elapsed = time.perf_counter() - t0

    def mean(rows, key):
        vals = [r[key] for r in rows]
        if not vals:
            return float("nan"), float("nan")
        se = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        return float(statistics.fmean(vals)), float(se)

    def baselines(targets):
        """机会基线：拿**另一个样本**的目标片段当答案。

        没有它，first_tok / tok_acc 这两个数无法解读——"第一个 token 蒙对 8%"
        到底算好还是算差，取决于随便挑一段代码能蒙对多少。
        """
        n = len(targets)
        if n < 2:
            return {"chance_edit_sim": float("nan")}
        # ⚠️ 配对必须用**随机置换**，不能用 j=(i*k+c)%n 这种确定性映射：
        # k 与 n 不互素时（如 5 与 100）映射只在少数伙伴间循环，机会基线会
        # **随样本量漂移**（实测 syntax 侧 5.0%(n=100) → 9.25%(n=400)），
        # 而且不给标准误 —— 那样"模型 vs 机会"的比较分母本身就不稳、不可比。
        rng = random.Random(20261004)
        perm = list(range(n))
        rng.shuffle(perm)
        for i in range(n):
            if perm[i] == i:
                j = (i + 1) % n
                perm[i], perm[j] = perm[j], perm[i]
        sims, firsts, accs = [], [], []
        for i in range(n):
            j = perm[i]
            t_i, ids_i = targets[i]
            t_j, ids_j = targets[j]
            sims.append(similarity(t_i, t_j))
            firsts.append(float(bool(ids_i) and bool(ids_j) and ids_i[0] == ids_j[0]))
            k = min(len(ids_i), len(ids_j))
            accs.append(
                float(sum(1 for x in range(k) if ids_i[x] == ids_j[x]) / k) if k else 0.0
            )
        return {
            "chance_edit_sim": float(statistics.fmean(sims)),
            "chance_edit_sim_se": float(statistics.stdev(sims) / math.sqrt(len(sims))),
            "chance_first_tok": float(statistics.fmean(firsts)),
            "chance_tok_acc": float(statistics.fmean(accs)),
        }

    per_hole = {}
    for h in holes:
        key = str(h)
        bl = baselines(targets_by_hole[key])
        # 语法侧机会基线：把**另一个样本**的目标片段插进本文档，看能否保持可解析。
        # 同样必须用随机置换（理由见上）。
        if len(samples) > 1:
            rng = random.Random(20261005)
            perm = list(range(len(samples)))
            rng.shuffle(perm)
            for i in range(len(samples)):
                if perm[i] == i:
                    j = (i + 1) % len(samples)
                    perm[i], perm[j] = perm[j], perm[i]
            syn = []
            for i, s in enumerate(samples):
                _pv, _sv, _t, full_suffix = make_view(s, h, block_size)
                other = decode_ids(tok, targets_by_hole[key][perm[i]][1])
                syn.append(
                    float(parseable(s["prefix_text"] + other + decode_ids(tok, full_suffix)))
                )
            bl["chance_syntax_ok"] = float(statistics.fmean(syn))
            bl["chance_syntax_ok_se"] = float(
                statistics.stdev(syn) / math.sqrt(len(syn))
            ) if len(syn) > 1 else 0.0
        per_hole[key] = {"baselines": bl, "arms": {}}
        for a in arms:
            rows = by_hole[key][a]
            entry = {}
            for k in METRIC_KEYS:
                m, se = mean(rows, k)
                entry[k] = {"mean": m, "stderr": se}
            entry["emitted_end"] = {"mean": mean(rows, "emitted_end")[0]}
            entry["mean_gen_chars"] = float(statistics.fmean([r["gen_len"] for r in rows]))
            per_hole[key]["arms"][a] = entry

    primary = str(DEFAULT_HOLE) if str(DEFAULT_HOLE) in per_hole else str(holes[-1])
    metrics = per_hole[primary]["arms"]          # 向后兼容：看守进程读这一层

    nbest_metrics = None
    if use_nbest:
        nbest_metrics = {}
        for strat, rows in nbest_rows.items():
            nbest_metrics[strat] = {}
            for k in METRIC_KEYS:
                mm, se = mean(rows, k)
                nbest_metrics[strat][k] = {"mean": mm, "stderr": se}
            nbest_metrics[strat]["emitted_end"] = {"mean": mean(rows, "emitted_end")[0]}

    result = {
        "bench": "cb-1",
        "run": str(args.run),
        "checkpoint": args.checkpoint,
        "train_steps": steps,
        "params": {
            "samples": len(samples),
            "seed": args.seed,
            "holes": holes,
            "primary_hole": int(primary),
            "block_size": block_size,
            "val_shard": str(val_path),
            "arms": arms,
            "arm_params": {a: ARMS[a] for a in arms},
            "nbest": nbest if use_nbest else 1,
            "threads": args.threads,
        },
        "baselines": per_hole[primary]["baselines"],
        "metrics": metrics,
        "by_hole": per_hole,
        "nbest": nbest_metrics,
        "elapsed_sec": round(elapsed, 1),
    }

    print()
    print("=" * 92)
    print(f"cb-1 代码能力基准  |  {args.run}  |  {args.tag or '未命名'}")
    print("=" * 92)
    print(f"  检查点训练步数 : {steps}    样本数 : {len(samples)} 个完整文档")
    print(f"  挖空长度       : {holes} token")
    print(f"  语料重复度基线 : rep8 {CORPUS_REP8}%  rep_line {CORPUS_REP_LINE}%"
          f"   <- 模型输出必须和它比，不是和 0 比")
    if len(holes) > 1:
        print("-" * 92)
        print("  能力随挖空长度的衰减（每格：模型 / 机会基线）")
        print(f"  {'挖空':>5} {'解码臂':<8}{'first_tok':>17}{'tok_acc':>17}{'edit_sim':>17}{'rep8':>8}")
        for h in holes:
            key = str(h)
            b = per_hole[key]["baselines"]
            for a in arms:
                e = per_hole[key]["arms"][a]
                print(
                    f"  {h:>5} {a:<8}"
                    f"{e['first_tok']['mean']*100:9.2f}%/{b.get('chance_first_tok', float('nan'))*100:6.2f}%"
                    f"{e['tok_acc']['mean']*100:9.2f}%/{b.get('chance_tok_acc', float('nan'))*100:6.2f}%"
                    f"{e['edit_sim']['mean']*100:9.2f}%/{b['chance_edit_sim']*100:6.2f}%"
                    f"{e['rep8']['mean']*100:7.1f}%"
                )
    print("-" * 92)
    key = primary
    b = per_hole[key]["baselines"]
    print(f"  主挖空长度 {key} token 的完整指标：")
    print(f"  {'指标':<26}" + "".join(f"{a:>14}" for a in arms) + f"{'机会基线':>14}")
    labels = {
        "first_tok": "first_tok 首token命中",
        "tok_acc": "tok_acc token准确率",
        "edit_sim": "edit_sim 编辑相似度",
        "exact": "exact 逐字符相同",
        "first_line": "first_line 首行相同",
        "syntax_ok": "syntax_ok 可解析",
        "rep8": "rep8 重复度(低好)",
        "rep_line": "rep_line 行重复(低好)",
        "emitted_end": "主动收尾率",
        "mean_gen_chars": "输出平均字符数",
    }
    base_map = {
        "first_tok": b.get("chance_first_tok"),
        "tok_acc": b.get("chance_tok_acc"),
        "edit_sim": b["chance_edit_sim"],
        "syntax_ok": b.get("chance_syntax_ok"),
    }
    # 机会基线自身的标准误也要落盘 —— 「模型 - 机会」这个差值的误差是两者之和，
    # 只报模型那一侧的 se 会系统性低估不确定性。见 AGENTS.md「统计分辨力」。
    base_se_map = {
        "edit_sim": b.get("chance_edit_sim_se"),
        "syntax_ok": b.get("chance_syntax_ok_se"),
    }
    for k, label in labels.items():
        cells = ""
        for a in arms:
            entry = per_hole[key]["arms"][a][k]
            v = entry["mean"] if isinstance(entry, dict) else entry
            cells += f"{v:14.1f}" if k == "mean_gen_chars" else f"{v*100:13.2f}%"
        bv = base_map.get(k)
        bse = base_se_map.get(k)
        bstr = f"{bv*100:13.2f}%" if isinstance(bv, float) else ""
        if isinstance(bv, float) and isinstance(bse, float):
            bstr += f" ±{bse*100:.2f}"
        print(f"  {label:<26}{cells}{bstr}")
    if nbest_metrics:
        print("-" * 92)
        print(f"  推理期脚手架：product 臂采样 {nbest} 个候选后择优"
              f"（{len(nbest_rows['first'])} 次选择）")
        print(f"  {'策略':<28}{'first_tok':>12}{'edit_sim':>12}{'syntax_ok':>12}{'rep8':>10}")
        names = {
            "first": "first 取第一个候选",
            "syntax": "syntax 可解析优先(可部署)",
            "oracle": "oracle 取最优(上界)",
        }
        for strat in STRATS:
            if strat not in nbest_metrics:
                continue
            e = nbest_metrics[strat]
            print(
                f"  {names[strat]:<28}{e['first_tok']['mean']*100:11.2f}%"
                f"{e['edit_sim']['mean']*100:11.2f}%{e['syntax_ok']['mean']*100:11.2f}%"
                f"{e['rep8']['mean']*100:9.1f}%"
            )
        base = nbest_metrics["first"]
        syn = nbest_metrics["syntax"]
        orc = nbest_metrics["oracle"]
        print(
            f"  可部署的语法验证器净收益：syntax_ok {base['syntax_ok']['mean']*100:.1f}% -> "
            f"{syn['syntax_ok']['mean']*100:.1f}%（{(syn['syntax_ok']['mean']-base['syntax_ok']['mean'])*100:+.1f}pp）"
            f" | edit_sim {(syn['edit_sim']['mean']-base['edit_sim']['mean'])*100:+.2f}pp"
        )
        print(
            f"  采样 {nbest} 倍算力的理论上界：edit_sim "
            f"{orc['edit_sim']['mean']*100:.2f}%（比单次 {(orc['edit_sim']['mean']-base['edit_sim']['mean'])*100:+.2f}pp）"
        )
    gens = len(samples) * len(holes) * (len(arms) + (nbest if use_nbest else 0))
    print("-" * 92)
    print(f"  耗时 {elapsed:.1f}s（共 {gens} 次生成，{elapsed/max(1,gens):.2f}s/次，{args.threads} 线程）")

    out = Path(args.out) if args.out else ROOT / "runs" / "_verify" / f"cb1-{args.tag or 'run'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  结果已存: {out}")

    if args.show:
        print()
        print("=" * 92)
        print(f"前 {args.show} 个样本的原始输出（挖空 {max(holes)} token）")
        print("=" * 92)
        for ex in examples[: args.show]:
            print(f"\n--- 样本 {ex['i']} ---")
            print("  [上下文末尾] " + repr(ex["prefix_tail"][-120:]))
            print("  [原文实际]   " + repr(ex["target"][:240]))
            for a in arms:
                txt = ex.get(a) or ""
                print(f"  [{a:<7}]    " + repr(txt[:240]))
                print(f"              相似度 {similarity(txt, ex['target'])*100:5.1f}%"
                      f"  重复度 {repetition(txt)*100:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
