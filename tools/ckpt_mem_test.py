"""在 CPU 上验证梯度检查点确实省内存（没有 GPU 时的替代验证）。

## 为什么必须用子进程 + 独立峰值采样

第一版在同一进程里先后测「不开/开」，结果被三件事污染：
1. **warmup**：第一次 forward 要初始化大量内部缓冲，24 秒 vs 0.6 秒
2. **RSS 不回落**：PyTorch 的 CPU 分配器释放后不还给操作系统
3. **懒初始化**：某些算子第一次调用才分配 workspace

所以改成：**每个模式一个子进程**，进程内开一个采样线程记录 RSS 峰值，
父进程只负责对比两个 JSON。

## 说明：为什么 CPU 的绝对数字不能直接搬到 GPU

CUDA 的 `max_memory_allocated` 精确统计张量；CPU 只能看进程 RSS（混着 Python 堆、
分配器 arena、碎片）。**但"开与不开的相对比值"有意义** —— 检查点省的是同一批激活张量。

用法::

    python tools/ckpt_mem_test.py --compare
    python tools/ckpt_mem_test.py --compare --layers 24 --embd 768 --ctx 1024 --batch 4
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(r"E:\AI")
sys.path.insert(0, str(ROOT))

PY = str(ROOT / ".venv/Scripts/python.exe")


class RssPeak:
    """后台线程持续采样 RSS，记录峰值。"""

    def __init__(self, interval: float = 0.02) -> None:
        import psutil

        self._proc = psutil.Process()
        self._peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, args=(interval,), daemon=True)
        self._t.start()

    def _run(self, interval: float) -> None:
        while not self._stop.is_set():
            try:
                self._peak = max(self._peak, self._proc.memory_info().rss)
            except Exception:
                break
            self._stop.wait(interval)

    def stop(self) -> float:
        self._stop.set()
        self._t.join(timeout=1.0)
        return self._peak / 2**30


def one(mode: str, cfg_args: argparse.Namespace) -> dict:
    import psutil
    import torch

    from codemind.config import ModelConfig
    from codemind.model import GPT

    gc.collect()
    before = psutil.Process().memory_info().rss / 2**30

    cfg = ModelConfig(
        vocab_size=cfg_args.vocab, n_layer=cfg_args.layers, n_head=cfg_args.heads,
        n_embd=cfg_args.embd, n_kv_head=cfg_args.kv_heads,
        block_size=cfg_args.ctx, dropout=0.0,
    )
    model = GPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    # 先跑一步暖机（不计入峰值统计）
    g = torch.Generator().manual_seed(3)
    x = torch.randint(0, cfg.vocab_size, (cfg_args.batch, cfg.block_size), generator=g)
    y = torch.randint(0, cfg.vocab_size, (cfg_args.batch, cfg.block_size), generator=g)
    model.train()
    _, loss, _ = model(x, y, logits_for_all=False, grad_checkpoint=(mode == "ckpt"))
    loss.backward()
    model.zero_grad(set_to_none=True)
    del loss
    gc.collect()

    opt = model.configure_optimizer(0.1, 2e-4, (0.9, 0.95), "cpu")
    del opt
    gc.collect()

    # 正式测量：从干净的 grads 状态开始，开峰值采样
    peak = RssPeak()
    t0 = time.time()
    model.train()
    _, loss, _ = model(x, y, logits_for_all=False, grad_checkpoint=(mode == "ckpt"))
    loss.backward()
    dt = time.time() - t0
    peak_gb = peak.stop()

    return {
        "mode": mode,
        "params_b": n_params / 1e9,
        "rss_before_gb": before,
        "rss_peak_gb": peak_gb,
        "delta_gb": peak_gb - before,
        "seconds": dt,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plain", "ckpt"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--embd", type=int, default=384)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    if args.mode:
        print(json.dumps(one(args.mode, args)))
        return 0
    if not args.compare:
        print("用 --compare 或 --mode plain|ckpt")
        return 2

    # 串行跑两个子进程（避免同时占内存）
    common = ["--layers", str(args.layers), "--embd", str(args.embd),
              "--heads", str(args.heads), "--kv-heads", str(args.kv_heads),
              "--ctx", str(args.ctx), "--vocab", str(args.vocab),
              "--batch", str(args.batch)]
    results = {}
    for mode in ("plain", "ckpt"):
        print(f"  测量 {mode} …", flush=True)
        r = subprocess.run([PY, __file__, "--mode", mode] + common,
                           capture_output=True, text=True, timeout=3600)
        line = [l for l in r.stdout.splitlines() if l.strip().startswith("{")]
        if not line:
            print("✗ 子进程失败:", r.stderr[-400:])
            return 1
        results[mode] = json.loads(line[-1])

    p, c = results["plain"], results["ckpt"]
    print()
    print(f"配置 {args.layers}层 d={args.embd} ctx={args.ctx} batch={args.batch} "
          f"| 参数 {p['params_b']:.3f}B")
    print(f"  不开检查点  峰值增量 **{p['delta_gb']:6.2f} GB**  {p['seconds']:5.1f}s")
    print(f"  开检查点    峰值增量 **{c['delta_gb']:6.2f} GB**  {c['seconds']:5.1f}s")
    if p["delta_gb"] > 0.02:
        print(f"  激活节省    **{(1-c['delta_gb']/p['delta_gb'])*100:.0f}%**"
              f"   速度 {(c['seconds']/p['seconds']-1)*100:+.0f}%")
        scale = 1.0 / p["params_b"]
        print()
        print("  按激活∝参数量线性外推到 1B（状态 fp32 = 16 GB）:")
        for name, d in (("不开", p["delta_gb"]), ("开", c["delta_gb"])):
            act = d * scale
            print(f"    {name:4}: 激活 {act:6.1f} GB + 状态 16 GB = **{act+16:6.1f} GB**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
