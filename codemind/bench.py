"""性能基准：在目标机器上实测训练/推理吞吐，并给出可行性判断。

为什么需要它：CPU 训练的吞吐受核心数、内存带宽、BLAS 实现、指令集
（AVX2 vs AVX-512）影响极大，纸面估算经常差 3-5 倍。决定「该训多大的模型、
要跑多久」之前，必须在**目标机器**上量一次。

用法::

    python -m codemind bench                       # 默认扫 nano/tiny/small
    python -m codemind bench --threads 48 --presets tiny small
    python -m codemind bench --presets small --batch-size 8 --steps 12
"""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .config import MODEL_PRESETS, ModelConfig, summarize_config
from .model import GPT, SamplingParams

__all__ = ["run_benchmark", "BenchResult"]


def _env_info() -> List[str]:
    """收集影响 CPU 训练速度的关键环境信息。"""
    import os
    import platform

    lines = [
        f"平台        : {platform.platform()}",
        f"CPU         : {platform.processor() or 'unknown'}",
        f"逻辑核心    : {os.cpu_count()}",
        f"PyTorch     : {torch.__version__}",
        f"torch 线程  : {torch.get_num_threads()}",
    ]
    # 指令集：AVX2 与 AVX-512 对 bf16/fp32 吞吐影响很大
    try:
        from torch.backends import cpu as cpu_backend

        caps = getattr(cpu_backend, "get_cpu_capability", lambda: "unknown")()
        lines.append(f"CPU 能力    : {caps}")
    except Exception:
        pass
    # bf16 在无 AVX512-BF16 的 CPU 上通常比 fp32 更慢，需要实测确认
    try:
        lines.append(f"bf16 支持   : {torch.backends.cpu.has_bf16 if hasattr(torch.backends.cpu, 'has_bf16') else 'n/a'}")
    except Exception:
        pass
    try:
        import numpy as np

        lines.append(f"NumPy       : {np.__version__}")
    except Exception:
        pass
    return lines


class BenchResult:
    """单个配置的基准结果。"""

    def __init__(
        self,
        preset: str,
        cfg: ModelConfig,
        threads: int,
        batch_size: int,
        *,
        train_tps: float = 0.0,
        infer_tps: float = 0.0,
        fwd_ms: float = 0.0,
        bwd_ms: float = 0.0,
        peak_mb: float = 0.0,
        error: str = "",
        effective_batch: int = 0,
    ) -> None:
        self.preset = preset
        self.cfg = cfg
        self.threads = threads
        self.batch_size = batch_size
        self.effective_batch = effective_batch
        self.train_tps = train_tps
        self.infer_tps = infer_tps
        self.fwd_ms = fwd_ms
        self.bwd_ms = bwd_ms
        self.peak_mb = peak_mb
        self.error = error

    @property
    def ok(self) -> bool:
        return not self.error and self.train_tps > 0

    def hours_for(self, tokens: float) -> float:
        if self.train_tps <= 0:
            return float("inf")
        return tokens / self.train_tps / 3600.0


def _peak_memory_mb() -> float:
    """当前进程的峰值内存（MB）。跨平台，取不到时返回 0。

    调 ``--batch-size`` 时这是最关键的参考值 —— 峰值逼近物理内存就会
    触发「无回溯的静默崩溃」，而那种崩溃很难定位，所以必须能提前看到数字。
    """
    import sys

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)

            # 必须显式声明原型：ctypes.windll 默认把返回值当 32 位 int，
            # 64 位下 BOOL 的高位被丢弃，调用明明成功也会读成 0。
            get_current = kernel32.GetCurrentProcess
            get_current.restype = wintypes.HANDLE
            get_current.argtypes = []

            get_info = psapi.GetProcessMemoryInfo
            get_info.restype = wintypes.BOOL
            get_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            if get_info(get_current(), ctypes.byref(counters), counters.cb):
                return counters.PeakWorkingSetSize / 1024 / 1024
        except Exception:
            return 0.0
        return 0.0

    try:
        import resource

        # Linux: KB；macOS: 字节
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return usage / 1024.0 if sys.platform != "darwin" else usage / 1024 / 1024
    except Exception:
        return 0.0


def _measure(
    cfg: ModelConfig,
    *,
    threads: int,
    batch_size: int,
    steps: int,
    device: torch.device,
    block_size: Optional[int] = None,
) -> BenchResult:
    """测量单个配置的前向/反向/优化器吞吐。"""
    from .train import setup_threads

    measured = ModelConfig(**{**cfg.to_dict(), "block_size": block_size or cfg.block_size})
    setup_threads(threads)
    result = BenchResult("", measured, threads, batch_size, effective_batch=batch_size)

    model = None
    try:
        torch.manual_seed(0)
        model = GPT(measured).to(device)
        model.train()
        optimizer = model.configure_optimizer(0.1, 1e-3, (0.9, 0.95), device.type)

        x = torch.randint(0, measured.vocab_size, (batch_size, measured.block_size), device=device)
        y = torch.randint(0, measured.vocab_size, (batch_size, measured.block_size), device=device)

        # 用满掩码走「只算计损位置 logits」的训练路径 —— 这是 train() 实际使用的最省内存路径，
        # 基准必须贴着它测，否则会高估内存需求、低估吞吐。
        full_mask = torch.ones_like(y, dtype=torch.float32)

        # 预热（首次前向会触发 BLAS 初始化与内存分配）
        _, loss, _ = model(x, y, loss_mask=full_mask, logits_for_all=False)
        assert loss is not None
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        fwd_times: List[float] = []
        bwd_times: List[float] = []

        for _ in range(max(1, steps)):
            t0 = time.perf_counter()
            _, loss, _ = model(x, y, loss_mask=full_mask, logits_for_all=False)
            assert loss is not None
            t1 = time.perf_counter()
            loss.backward()
            t2 = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            fwd_times.append(t1 - t0)
            bwd_times.append(t2 - t1)

        tokens = batch_size * measured.block_size
        train_tps = tokens / (sum(fwd_times) + sum(bwd_times)) * len(fwd_times)
        result.fwd_ms = sum(fwd_times) / len(fwd_times) * 1000
        result.bwd_ms = sum(bwd_times) / len(bwd_times) * 1000
        result.train_tps = train_tps

        # 推理吞吐（KV-Cache 路径，逐 token 解码 —— 这是实际补全时的形态）
        model.eval()
        prompt = torch.randint(0, measured.vocab_size, (1, 8), device=device)
        params = SamplingParams(greedy=True)
        n_gen = 32
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(prompt, n_gen, params)
        elapsed = time.perf_counter() - t0
        result.infer_tps = n_gen / elapsed if elapsed > 0 else 0.0

        # 峰值内存：调 batch_size 时这是最关键的参考数字
        result.peak_mb = _peak_memory_mb()
    except (RuntimeError, MemoryError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        del model
        gc.collect()
    return result


def _shrink_until_fits(
    cfg: ModelConfig,
    *,
    threads: int,
    batch_size: int,
    steps: int,
    device: torch.device,
    block_size: Optional[int],
) -> BenchResult:
    """内存不足时先减半 batch，再减半上下文，直到跑通或触底。

    真实训练里 batch 与 block_size 都可以调，所以两个维度都退让，
    才能回答「这个预设在这台机器上到底能不能跑」。
    """
    base_block = block_size or cfg.block_size
    batch = batch_size
    block = base_block
    noted: List[str] = []
    last: Optional[BenchResult] = None

    while batch >= 1 and block >= 64:
        res = _measure(
            cfg,
            threads=threads,
            batch_size=batch,
            steps=steps,
            device=device,
            block_size=block,
        )
        if res.ok:
            used_block = ModelConfig(
                **{**cfg.to_dict(), "block_size": block}
            )
            res.cfg = used_block
            res.batch_size = batch
            res.effective_batch = batch
            res.error = ""
            if batch != batch_size:
                noted.append(f"batch {batch_size}→{batch}")
            if block != base_block:
                noted.append(f"ctx {base_block}→{block}")
            res.preset = res.preset + (f" [{', '.join(noted)}]" if noted else "")
            return res
        last = res
        if "memory" not in res.error.lower() and "MemoryError" not in res.error:
            return res  # 不是内存问题，不必再试
        if batch > 1:
            batch //= 2
        else:
            block //= 2

    assert last is not None
    return last


def run_benchmark(
    presets: Sequence[str] = ("nano", "tiny", "small"),
    *,
    threads: Optional[int] = None,
    batch_size: int = 8,
    steps: int = 8,
    block_size: Optional[int] = None,
    token_budgets: Sequence[float] = (100e6, 300e6, 1e9, 2e9),
    device_spec: str = "auto",
    verbose: bool = True,
) -> List[BenchResult]:
    """对若干预设跑基准，返回结果列表（同时打印可读报告）。"""
    from .config import resolve_device
    from .train import setup_threads

    device = resolve_device(device_spec)
    import os

    thread_candidates = [threads] if threads else sorted(
        {max(1, (os.cpu_count() or 8)), max(1, (os.cpu_count() or 8) // 2), 4}
    )

    if verbose:
        print("=" * 74)
        print("CodeMind 性能基准")
        print("=" * 74)
        for line in _env_info():
            print(line)
        print(f"设备        : {device}")
        print(f"基准参数    : batch={batch_size} steps={steps}")
        print("-" * 74)

    results: List[BenchResult] = []
    for preset in presets:
        if preset not in MODEL_PRESETS:
            print(f"[bench] 跳过未知预设 {preset!r}")
            continue
        base_cfg = ModelConfig.from_preset(preset)
        for n_threads in thread_candidates:
            if verbose:
                print(
                    f"\n>>> {preset}: {summarize_config(base_cfg)} | 线程={n_threads}"
                )
            res = _shrink_until_fits(
                base_cfg,
                threads=n_threads,
                batch_size=batch_size,
                steps=steps,
                device=device,
                block_size=block_size,
            )
            if not res.preset:
                res.preset = preset
            results.append(res)
            if not res.ok:
                print(f"    失败: {res.error}")
                continue
            print(
                f"    前向 {res.fwd_ms:7.1f} ms | 反向 {res.bwd_ms:7.1f} ms | "
                f"训练 {res.train_tps:9,.0f} tok/s | 推理 {res.infer_tps:6.1f} tok/s"
            )
            if res.peak_mb:
                print(f"    进程峰值内存 {res.peak_mb:,.0f} MB")

    if verbose:
        _print_report(results, token_budgets)
    return results


def _print_report(results: Sequence[BenchResult], token_budgets: Sequence[float]) -> None:
    ok = [r for r in results if r.ok]
    print("\n" + "=" * 74)
    print("训练时长估算（按实测吞吐线性外推）")
    print("=" * 74)
    if not ok:
        print("没有任何配置跑通，无法估算。")
        return

    header = f"{'预设':<8}{'线程':>5}{'训练 tok/s':>13}" + "".join(
        f"{_fmt_tokens(b):>12}" for b in token_budgets
    )
    print(header)
    print("-" * len(header))
    for r in ok:
        row = f"{r.preset:<8}{r.threads:>5}{r.train_tps:>13,.0f}"
        for budget in token_budgets:
            row += f"{_fmt_hours(r.hours_for(budget)):>12}"
        print(row)

    best = max(ok, key=lambda r: r.train_tps)
    print("-" * len(header))
    print(f"最快配置: {best.preset} @ {best.threads} 线程 ({best.train_tps:,.0f} tok/s)")

    # 给出按 Chinchilla 比例（约 20 token/参数）所需的训练量
    print("\n按 Chinchilla 比例（≈20 token/参数）达到较充分训练所需时间：")
    for r in ok:
        params = r.cfg.estimated_params()
        need = params * 20
        print(
            f"  {r.preset:<8} {params/1e6:6.2f}M 参数 -> 需 {need/1e6:7.1f}M token "
            f"-> {_fmt_hours(r.hours_for(need))}"
        )

    print("\n提示：")
    print("  * 多线程超过物理核心数后吞吐通常不再提升，甚至下降；")
    print("    请对比不同线程数的结果，选最快的那个并固定下来。")
    print("  * 训练吞吐为 0 或报错时，先用 --batch-size 1 确认能否跑起来。")
    print("  * 估算不含数据加载与评估开销，实际总时长通常再多 10%~20%。")


def _fmt_tokens(value: float) -> str:
    if value >= 1e9:
        return f"{value/1e9:.1f}B"
    if value >= 1e6:
        return f"{value/1e6:.0f}M"
    return f"{value/1e3:.0f}K"


def _fmt_hours(hours: float) -> str:
    if not math.isfinite(hours):
        return "n/a"
    if hours < 1:
        return f"{hours*60:.0f}分钟"
    if hours < 48:
        return f"{hours:.1f}小时"
    return f"{hours/24:.1f}天"
