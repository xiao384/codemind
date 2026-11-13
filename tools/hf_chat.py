"""用 HF 模型跑我们一直在用的那 4 个聊天问题，便于和自训模型直接对比。

之前测过的模型（同一套问题）：
    11M chat2 / 78M 融合 / 20M 阶段2 / 1B（24% 进度）
对零输出「回答形状」或「复读问题」；这里看 1.5B 蒸馏模型的表现作为上界参考。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\AI")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = next((p.parent for p in (ROOT / "models").rglob("model.safetensors")), None)
QS = ("你好，你是谁？", "保持健康的三个提示。", "什么是递归？", "介绍一下中国的长城")

torch.set_num_threads(24)
tk = AutoTokenizer.from_pretrained(str(PATH), trust_remote_code=True)
md = AutoModelForCausalLM.from_pretrained(
    str(PATH), dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
md.eval()
print(f"模型: {PATH.name}  参数 {sum(p.numel() for p in md.parameters())/1e9:.2f}B\n", flush=True)

t0 = time.time()
for q in QS:
    msgs = [{"role": "user", "content": q}]
    text = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tk(text, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        out = md.generate(ids, max_new_tokens=160, do_sample=False,
                          pad_token_id=tk.eos_token_id)
    ans = tk.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"问：{q}")
    print(f"答：{ans[:300]}")
    print(flush=True)

print(f"\n耗时 {time.time()-t0:.0f}s")
