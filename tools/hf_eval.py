"""用 HF transformers 测 DeepSeek-R1-Distill-1.5B 的代码补全与聊天能力。

## 为什么单独写一个，不复用 cb-1

`cb-1` 是围绕**我们自己的 tokenizer（vocab 4096）**和 PSM/FIM 格式设计的，
换个模型（Qwen tokenizer 151k）就没法用同一套 prompt。

**要跨模型可比，必须在「文本层面」定义任务**：

    给一段代码前缀（用人能读的文本），让模型续写，再和真实后续比较。

本文具就做这件事，指标仍是 `edit_sim`（环境无关）。

用法::

    python tools/hf_eval.py --n 20 --max-new 60
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(r"E:\AI")
os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))
MODEL_DIR = next(
    (p for p in (ROOT / "models").rglob("model.safetensors")), None)
if MODEL_DIR is None:
    raise SystemExit("找不到已下载的模型")


def pick_docs(n: int) -> list[str]:
    """从 corpus4 验证分片里取若干段可读文本。"""
    import numpy as np
    from codemind.tokenizer import BPETokenizer

    tok = BPETokenizer.load(ROOT / "data/corpus4/tokenizer.json")
    files = sorted((ROOT / "data/corpus4").glob("*.val*.bin"))
    if not files:
        raise SystemExit("找不到验证分片")
    arr = np.concatenate([np.fromfile(f, dtype=np.uint16) for f in files])
    print(f"  验证集 {len(arr):,} token")

    docs, i = [], 0
    step = max(1, len(arr) // (n * 3))
    while i < len(arr) - 600 and len(docs) < n:
        chunk = arr[i:i + 600].tolist()
        txt = tok.decode(chunk, strip_special=True)
        # 只保留像 Python 的、且以换行结尾的片段
        if len(txt) > 400 and "\n" in txt and ("def " in txt or "=" in txt):
            docs.append(txt)
        i += step
    print(f"  取到 {len(docs)} 个文档片段")
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--prefix-chars", type=int, default=600)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(24)
    path = MODEL_DIR.parent
    print(f"模型目录: {path}")
    t0 = time.time()
    tk = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    md = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
    md.eval()
    print(f"加载完成 {time.time()-t0:.1f}s  参数 {sum(p.numel() for p in md.parameters())/1e9:.2f}B")

    docs = pick_docs(args.n)
    results = []
    t0 = time.time()
    tok_count = 0

    for k, txt in enumerate(docs):
        p = min(args.prefix_chars, len(txt) // 2)
        prefix, target = txt[:p], txt[p:p + args.max_new * 4]

        ids = tk(prefix, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            out = md.generate(ids, max_new_tokens=args.max_new,
                              do_sample=False, pad_token_id=tk.eos_token_id)
        gen = tk.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        tok_count += args.max_new

        sim = difflib.SequenceMatcher(None, gen[:400], target[:400]).ratio()
        try:
            compile(gen, "<gen>", "exec")
            syn = 1
        except SyntaxError:
            syn = 0
        results.append({"k": k, "sim": sim, "syntax": syn,
                        "gen_head": gen[:120].replace("\n", "\\n")})
        if (k + 1) % 5 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(docs)}  {el:.0f}s  {el/5/(k+1)*len(docs):.0f}s 预计剩余")
    dt = time.time() - t0
    n = len(results)
    print(f"\n=== 结果（{n} 个片段，{dt/60:.1f} 分钟）===")
    print(f"  edit_sim 均值 : {sum(r['sim'] for r in results)/n*100:.2f}%")
    print(f"  syntax_ok     : {sum(r['syntax'] for r in results)/n*100:.1f}%")
    print(f"  速度          : {tok_count/max(dt,1):.2f} tok/s")
    print("\n  样例:")
    for r in results[:3]:
        print(f"    [{r['k']}] sim={r['sim']*100:.1f}% syn={r['syntax']}  {r['gen_head'][:90]}")

    out = ROOT / "runs/_verify/hf-eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": str(path), "results": results,
                               "edit_sim": sum(r['sim'] for r in results)/n,
                               "syntax_ok": sum(r['syntax'] for r in results)/n},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  结果已存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
