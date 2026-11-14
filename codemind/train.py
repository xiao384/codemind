"""训练循环 —— 从零实现，不依赖任何训练框架。

包含现代 LLM 训练的必要组件：

* AdamW + 分组权重衰减（norm/bias 不衰减）
* 线性 warmup + 余弦退火
* 梯度累积（小显存/CPU 上模拟大 batch）
* 梯度裁剪、可选混合精度（CPU 上默认关闭）
* 定期评估、最优权重保留、断点续训（optimizer + 进度完整落盘）
* FIM 掩码损失（只对「被挖空的中间段」计算损失）

用法::

    from codemind.train import train
    train(model_cfg, train_cfg, shards_dir=Path("data/corpus"), tokenizer=tok)
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .config import ModelConfig, TrainConfig, dump_json, pick_dtype, resolve_device
from .data import RandomWindowSampler, TokenShards
from .model import GPT
from .tokenizer import BPETokenizer

__all__ = ["train", "TrainReport", "setup_threads", "save_checkpoint", "load_checkpoint"]


# ======================================================================================
# 环境准备
# ======================================================================================


def setup_threads(num_threads: int = 0) -> int:
    """配置 CPU 线程数。CPU 训练时这是最影响速度的单一开关。"""
    if num_threads <= 0:
        num_threads = max(1, (os.cpu_count() or 4))
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, num_threads // 4 or 1)))
    except RuntimeError:
        # 只能在并行工作启动前设置一次，失败无害
        pass
    return num_threads


def restore_rng(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================================
# Checkpoint
# ======================================================================================


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    iteration: int,
    best_val: float,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    tokenizer_path: Optional[str] = None,
    extra: Optional[Dict[str, object]] = None,
) -> Path:
    """保存检查点。

    ``optimizer`` 为 None 时只存模型权重（用于 ``best.pt``）：一个 11M 参数模型
    的权重是 44MB，而 AdamW 的两组动量会再翻一倍多。
    小机器上同时留两份全量检查点很容易把内存/磁盘打满，因此 ``best.pt`` 刻意只存权重
    —— 它本来就是给推理和 ``load_runner`` 用的，续训走 ``latest.pt``。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "format": "codemind-checkpoint-v1",
        "iteration": iteration,
        "best_val": best_val,
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        # state_dict 内部按张量去重，权重共享不会导致重复存储
        "model": model.state_dict(),
        "tokenizer_path": tokenizer_path,
        "has_optimizer": optimizer is not None,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, object]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format") != "codemind-checkpoint-v1":
        raise ValueError(f"{path} 不是 CodeMind checkpoint（format={payload.get('format')!r}）")
    return payload


# ======================================================================================
# 训练报告
# ======================================================================================


@dataclass
class TrainReport:
    iterations: int
    best_val: float
    final_train: float
    tokens_seen: int
    seconds: float
    out_dir: Path
    best_ckpt: Optional[Path]

    @property
    def tokens_per_second(self) -> float:
        return self.tokens_seen / self.seconds if self.seconds > 0 else float("inf")

    def summary(self) -> str:
        return (
            f"训练完成：{self.iterations} 步 / {self.tokens_seen/1e6:.2f}M tokens / "
            f"{self.seconds/60:.1f} 分钟 ({self.tokens_per_second:,.0f} tok/s)\n"
            f"  最优验证损失 {self.best_val:.4f}（困惑度 {math.exp(min(self.best_val, 20)):.2f}）"
            f" | 末尾训练损失 {self.final_train:.4f}\n"
            f"  输出目录 {self.out_dir}"
        )


# ======================================================================================
# 主训练函数
# ======================================================================================


def train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    *,
    shards: TokenShards,
    tokenizer: BPETokenizer,
    resume_from: Optional[Path] = None,
    verbose: bool = True,
) -> TrainReport:
    """执行训练并返回报告。中途可通过 ``runs/<name>/latest.pt`` 续训。"""
    device = resolve_device(train_cfg.device)
    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = setup_threads(train_cfg.num_threads)
    restore_rng(train_cfg.seed)

    amp_dtype, use_scaler = pick_dtype(train_cfg.amp, device)

    # ---------------------------------------------------------------- 数据
    sampler = RandomWindowSampler(
        shards.train_paths,
        block_size=model_cfg.block_size,
        batch_size=train_cfg.batch_size,
        dtype=shards.dtype,
        device=device.type,
        seed=train_cfg.seed,
        fim_rate=train_cfg.fim_rate,
        fim_spm_rate=train_cfg.fim_spm_rate,
        tokenizer=tokenizer,
        pin_memory=device.type == "cuda",
        fim_terminator=train_cfg.fim_terminator,
    )
    val_paths = shards.val_paths or shards.train_paths
    val_sampler = RandomWindowSampler(
        val_paths,
        block_size=model_cfg.block_size,
        batch_size=train_cfg.batch_size,
        dtype=shards.dtype,
        device=device.type,
        seed=train_cfg.seed + 991,
        # 验证集也走 FIM，这样验证损失直接反映「填空能力」而不是普通语言建模
        fim_rate=train_cfg.fim_rate,
        fim_spm_rate=train_cfg.fim_spm_rate,
        tokenizer=tokenizer,
        pin_memory=device.type == "cuda",
        fim_terminator=train_cfg.fim_terminator,
    )

    # ---------------------------------------------------------------- 模型
    model = GPT(model_cfg).to(device)
    optimizer = model.configure_optimizer(
        weight_decay=train_cfg.weight_decay,
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
        device_type=device.type,
    )
    scaler = _make_grad_scaler(device, use_scaler)

    start_iter = 0
    best_val = float("inf")
    resume_path = resume_from
    if resume_path is None and train_cfg.resume:
        candidate = out_dir / "latest.pt"
        resume_path = candidate if candidate.exists() else None
    if resume_path is not None and Path(resume_path).exists():
        payload = load_checkpoint(Path(resume_path), map_location=str(device))
        model.load_state_dict(payload["model"])  # type: ignore[arg-type]
        if payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])  # type: ignore[arg-type]
        elif verbose:
            print(
                f"[train] {resume_path} 不含优化器状态（best.pt 只存权重），"
                "本次从模型权重继续，优化器动量重新累积"
            )
        start_iter = int(payload.get("iteration", 0))
        best_val = float(payload.get("best_val", float("inf")))
        if verbose:
            print(f"[train] 从 {resume_path} 续训，起始步 {start_iter}，历史最优 {best_val:.4f}")

    if verbose:
        print(f"[train] 设备={device} 线程={threads} amp={train_cfg.amp}")
        print(f"[train] 模型 {model_cfg.estimated_params()/1e6:.2f}M 参数 | {model.num_params()/1e6:.2f}M 实参")
        plan_tokens = train_cfg.tokens_per_iter(model_cfg.block_size) * (
            train_cfg.max_iters - start_iter
        )
        print(
            f"[train] 计划 {train_cfg.max_iters - start_iter} 步 × "
            f"{train_cfg.tokens_per_iter(model_cfg.block_size):,} tok/步 ≈ {plan_tokens/1e6:.1f}M tokens"
        )
        # 用绝对数值而不是 MB 显示：小验证集（几百 token）在 MB 下会显示成 0.00M，
        # 容易被误读成"没有验证集"。
        print(
            f"[train] 数据 训练 {sampler.total_tokens:,} tokens / "
            f"验证 {val_sampler.total_tokens:,} tokens"
        )

    if train_cfg.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            if verbose:
                print("[train] 已启用 torch.compile")
        except Exception as exc:  # pragma: no cover
            print(f"[train] torch.compile 不可用，跳过：{exc}")

    # ---------------------------------------------------------------- 训练循环
    tokens_seen = 0
    t0 = time.time()
    running_loss = 0.0
    running_count = 0
    last_train_loss = float("nan")
    best_ckpt: Optional[Path] = None
    log_lines: List[Dict[str, object]] = []

    model.train()
    for step in range(start_iter, train_cfg.max_iters):
        lr = train_cfg.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(train_cfg.grad_accum):
            x, y, mask = sampler.sample_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            with _autocast(device, amp_dtype):
                # logits_for_all=False：训练只要 loss，不算全量 (B,T,V) logits。
                # 这是 CPU 训练最关键的省内存开关（见 GPT.forward 的说明）。
                _, loss, _ = model(x, y, loss_mask=mask, logits_for_all=False,
                                  grad_checkpoint=bool(train_cfg.grad_checkpoint))
                assert loss is not None
                loss = loss / train_cfg.grad_accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_loss += float(loss.detach()) * train_cfg.grad_accum
            tokens_seen += int(x.numel())

        if train_cfg.grad_clip > 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        avg_loss = step_loss / train_cfg.grad_accum
        running_loss += avg_loss
        running_count += 1
        last_train_loss = avg_loss

        if verbose and train_cfg.log_interval > 0 and (
            step % train_cfg.log_interval == 0 or step == train_cfg.max_iters - 1
        ):
            elapsed = time.time() - t0
            tps = tokens_seen / elapsed if elapsed > 0 else 0.0
            eta = (train_cfg.max_iters - step - 1) * (elapsed / max(1, step - start_iter + 1))
            print(
                f"[train] step {step+1}/{train_cfg.max_iters} "
                f"loss {running_loss/running_count:.4f} lr {lr:.2e} "
                f"| {tps:,.0f} tok/s ETA {eta/60:.1f}min"
            )
            log_lines.append(
                {
                    "step": step + 1,
                    "loss": running_loss / running_count,
                    "lr": lr,
                    "tokens": tokens_seen,
                }
            )
            running_loss = 0.0
            running_count = 0

        # ------------------------------------------------------------ 评估
        do_eval = train_cfg.eval_interval > 0 and (
            (step + 1) % train_cfg.eval_interval == 0 or step == train_cfg.max_iters - 1
        )
        if do_eval:
            val_loss = evaluate(model, val_sampler, train_cfg.eval_batches, device, amp_dtype)
            if verbose:
                ppl = math.exp(min(val_loss, 20))
                print(f"[train] --- 验证集 loss {val_loss:.4f} (ppl {ppl:.2f}) @ step {step+1}")
            log_lines.append({"step": step + 1, "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                # best.pt 只存权重（不存优化器动量）：它是推理入口，且小机器上省一倍空间
                best_ckpt = save_checkpoint(
                    out_dir / "best.pt",
                    model,
                    None,
                    iteration=step + 1,
                    best_val=best_val,
                    train_cfg=train_cfg,
                    model_cfg=model_cfg,
                    tokenizer_path=str(out_dir / "tokenizer.json"),
                    extra={"val_loss": val_loss},
                )
                if verbose:
                    print(f"[train] +++ 新的最优权重 -> {best_ckpt.name}")

        # ------------------------------------------------------------ 存档
        if train_cfg.ckpt_interval > 0 and (step + 1) % train_cfg.ckpt_interval == 0:
            ckpt = save_checkpoint(
                out_dir / "latest.pt",
                model,
                optimizer,
                iteration=step + 1,
                best_val=best_val,
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                tokenizer_path=str(out_dir / "tokenizer.json"),
            )
            if verbose:
                print(f"[train] 已保存 {ckpt}")

    # ---------------------------------------------------------------- 收尾
    seconds = time.time() - t0
    final_ckpt = save_checkpoint(
        out_dir / "latest.pt",
        model,
        optimizer,
        iteration=train_cfg.max_iters,
        best_val=best_val,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        tokenizer_path=str(out_dir / "tokenizer.json"),
    )
    if best_ckpt is None:
        best_ckpt = final_ckpt
    tokenizer.save(out_dir / "tokenizer.json")
    dump_json(
        out_dir / "log.json",
        {
            "history": log_lines,
            "final": {
                "iteration": train_cfg.max_iters,
                "best_val": best_val if math.isfinite(best_val) else None,
                "train_loss": last_train_loss,
                "tokens_seen": tokens_seen,
                "seconds": seconds,
                "tokens_per_second": tokens_seen / seconds if seconds > 0 else None,
            },
            "model_config": model_cfg.to_dict(),
            "train_config": train_cfg.to_dict(),
        },
    )

    report = TrainReport(
        iterations=train_cfg.max_iters,
        best_val=best_val,
        final_train=last_train_loss,
        tokens_seen=tokens_seen,
        seconds=seconds,
        out_dir=out_dir,
        best_ckpt=best_ckpt,
    )
    if verbose:
        print(report.summary())
    return report


# ======================================================================================
# 辅助
# ======================================================================================


def _autocast(device, amp_dtype):
    """构造（或跳过）autocast 上下文，避免在 amp=off 时引入无谓开销。"""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _make_grad_scaler(device, enabled: bool):
    """兼容 torch 2.2（``GradScaler(enabled)``）与 2.3+（``GradScaler(device, enabled)``）。"""
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)  # type: ignore[call-arg]
    except TypeError:
        return torch.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate(
    model: GPT,
    sampler: RandomWindowSampler,
    batches: int,
    device,
    amp_dtype,
) -> float:
    """在固定批数上评估平均损失。评估用同一批采样器状态，因此波动可控。"""
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for _ in range(max(1, batches)):
        x, y, mask = sampler.sample_batch()
        x = x.to(device)
        y = y.to(device)
        if mask is not None:
            mask = mask.to(device)
        with _autocast(device, amp_dtype):
            _, loss, _ = model(x, y, loss_mask=mask, logits_for_all=False,
                                  grad_checkpoint=False)
        if loss is not None:
            total += float(loss)
            count += 1
    if was_training:
        model.train()
    return total / max(1, count)


def estimate_training_time(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    tokens_per_second: float,
) -> str:
    """根据实测吞吐估算总时长，用于训练前的可行性判断。"""
    tokens = train_cfg.tokens_per_iter(model_cfg.block_size) * train_cfg.max_iters
    if tokens_per_second <= 0:
        return "无法估算（吞吐为 0）"
    seconds = tokens / tokens_per_second
    return (
        f"{tokens/1e6:.1f}M tokens @ {tokens_per_second:,.0f} tok/s "
        f"≈ {seconds/60:.1f} 分钟 ({seconds/3600:.2f} 小时)"
    )
