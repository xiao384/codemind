"""CodeMind 命令行入口。

    codemind doctor                     环境自检（先跑这个）
    codemind presets                    查看模型规模预设
    codemind prepare --source .         采集代码 -> 训练分词器 -> 切分片
    codemind train --run runs/tiny      训练（CPU 可跑）
    codemind gen --run runs/tiny ...    续写补全
    codemind fim --run runs/tiny -f a.py --line 20   文件中间填空
    codemind repl --run runs/tiny       交互式补全
    codemind agent --run runs/tiny -f a.py -i "..."  改代码 + 测试驱动自纠错
    codemind info --run runs/tiny       检查点信息
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from .config import (
    MODEL_PRESETS,
    DataConfig,
    ModelConfig,
    TrainConfig,
    describe_presets,
    resolve_device,
    summarize_config,
)
from .editing import CodeEditor, build_edit_prompt, extract_code_block, unified_diff

__all__ = ["main", "build_parser"]


# ======================================================================================
# 参数解析
# ======================================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codemind",
        description="CodeMind：从零实现的代码大模型与编程 CLI（纯 Python + PyTorch）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "典型流程:\n"
            "  codemind doctor\n"
            "  codemind prepare --source . --name corpus\n"
            "  codemind train --run runs/tiny --data data/corpus --preset tiny\n"
            "  codemind gen --run runs/tiny --prompt 'def add(a, b):'\n"
            "  codemind fim --run runs/tiny -f codemind/config.py --line 40\n"
            "  codemind repl --run runs/tiny\n"
            "  codemind agent --run runs/tiny -f main.py -i '加上 __main__ 入口' --write\n"
        ),
    )
    parser.add_argument("--version", action="version", version="codemind 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    _add_doctor(sub)
    _add_presets(sub)
    _add_prepare(sub)
    _add_train(sub)
    _add_gen(sub)
    _add_fim(sub)
    _add_repl(sub)
    _add_agent(sub)
    _add_chat(sub)
    _add_zip(sub)
    _add_info(sub)
    _add_bench(sub)
    _add_selftest(sub)
    return parser


def _add_sampling_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--temperature", type=float, default=0.8, help="采样温度（0 = 贪心）")
    p.add_argument("--top-k", type=int, default=40, help="top-k 截断（0 = 关闭）")
    p.add_argument("--top-p", type=float, default=0.95, help="核采样阈值")
    p.add_argument("--repetition-penalty", type=float, default=1.1, help="重复惩罚")
    p.add_argument("--seed", type=int, default=None, help="随机种子（便于复现）")
    p.add_argument("--greedy", action="store_true", help="强制贪心解码")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    p.add_argument("--run", default="runs/tiny", help="训练输出目录（含 best.pt 与 tokenizer.json）")
    p.add_argument("--checkpoint", default="best", help="best / latest / 具体 .pt 文件名")


def _add_doctor(sub) -> None:
    p = sub.add_parser("doctor", help="检查运行环境是否满足要求")
    p.add_argument("--device", default="auto")


def _add_presets(sub) -> None:
    sub.add_parser("presets", help="列出模型规模预设")


def _add_prepare(sub) -> None:
    p = sub.add_parser("prepare", help="采集代码语料、训练分词器、生成训练分片")
    p.add_argument(
        "--source",
        action="append",
        default=None,
        help="代码目录，可重复指定。默认使用 ./codemind 与 ./tests",
    )
    p.add_argument("--out", default="data", help="分片输出根目录")
    p.add_argument("--name", default="corpus", help="数据集名称（决定子目录名）")
    p.add_argument("--vocab", type=int, default=4096, help="BPE 词表大小")
    p.add_argument("--preset", default="tiny", choices=sorted(MODEL_PRESETS), help="按预设推断 vocab")
    p.add_argument("--extensions", nargs="*", default=None, help="只收录这些后缀（默认内置白名单）")
    p.add_argument("--min-bytes", type=int, default=32)
    p.add_argument("--max-bytes", type=int, default=512 * 1024)
    p.add_argument("--max-files", type=int, default=0, help="最多收录多少文件（0 = 不限）")
    p.add_argument("--val-ratio", type=float, default=0.01)
    p.add_argument(
        "--shard-tokens",
        type=int,
        default=20_000_000,
        help="单个分片的最大 token 数（大语料可调小以降低单文件体积）",
    )
    p.add_argument(
        "--tokenizer-chars",
        type=int,
        default=50_000_000,
        help="训练分词器时最多读取的字符数；语料越大越该用样本，避免 OOM",
    )
    p.add_argument("--no-dedup", action="store_true", help="关闭内容去重")
    p.add_argument("--seed", type=int, default=1337)


def _add_train(sub) -> None:
    p = sub.add_parser("train", help="训练模型")
    p.add_argument("--run", default="runs/tiny", help="输出目录")
    p.add_argument("--data", default="data/corpus", help="分片目录或 *.meta.json")
    p.add_argument("--preset", default="tiny", choices=sorted(MODEL_PRESETS))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="梯度检查点：用计算换显存（1B 在 32GB 双 T4 上必需）")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    # ⚠️ help 里的 `%` 必须写成 `%%`：argparse 会对 help 字符串做 `%` 插值，
    # 单个 `%` 会让 `train --help` 直接抛 `ValueError: incomplete format`。
    # 实测踩过：整个 train 子命令的帮助页崩掉，而参数本身完全正常。
    p.add_argument("--warmup", type=int, default=0, help="0 = 自动取 3%%")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ctx", type=int, default=None, help="覆盖 block_size")
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--embd", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--ckpt-interval", type=int, default=200)
    p.add_argument("--fim-rate", type=float, default=0.5, help="FIM 训练比例（代码补全能力的关键）")
    p.add_argument("--fim-spm-rate", type=float, default=0.5)
    p.add_argument(
        "--fim-terminator",
        action="store_true",
        help="v2 FIM 格式：中间段后补终止符并纳入损失（让模型学会「填完收尾」），"
             "同时不再丢失中间段尾部 3 个 token。默认关闭以保持历史行为。",
    )
    p.add_argument("--amp", default="off", choices=["off", "bf16", "fp16"])
    p.add_argument("--threads", type=int, default=0, help="CPU 线程数（0 = 自动）")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-resume", action="store_true", help="忽略已有 latest.pt，从头训练")
    p.add_argument("--compile", action="store_true", help="尝试 torch.compile")


def _add_gen(sub) -> None:
    p = sub.add_parser("gen", help="续写补全（打印新增内容）")
    _add_model_args(p)
    _add_sampling_args(p)
    p.add_argument("--prompt", default=None, help="提示文本")
    p.add_argument("-f", "--file", default=None, help="从文件读取提示（可配合 --tail）")
    p.add_argument("--tail", type=int, default=0, help="只取文件末尾 N 个字符作为提示")
    p.add_argument("--max-tokens", type=int, default=128, help="最多生成多少 token")
    p.add_argument("--stop", action="append", default=None, help="停止字符串，可重复")
    p.add_argument("--show-prompt", action="store_true")
    p.add_argument("--no-stream", action="store_true", help="关闭逐字流式输出")


def _add_fim(sub) -> None:
    p = sub.add_parser("fim", help="填空补全：在文件的指定行处生成代码")
    _add_model_args(p)
    _add_sampling_args(p)
    p.add_argument("-f", "--file", required=True, help="目标文件")
    p.add_argument("--line", type=int, default=None, help="光标所在行号（1 基）；默认文件末尾")
    p.add_argument("--before", type=int, default=0, help="光标前保留的字符数（默认按行）")
    p.add_argument("--after", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--write", action="store_true", help="把结果写回文件")
    p.add_argument("--dry-run", action="store_true", help="只校验不写入")


def _add_repl(sub) -> None:
    p = sub.add_parser("repl", help="交互式补全（逐行输入，模型接着写）")
    _add_model_args(p)
    _add_sampling_args(p)
    p.add_argument("--max-tokens", type=int, default=96)


def _add_chat(sub) -> None:
    """`chat` 子命令：把 `Generator.chat()` 接到命令行上。

    ⚠️ 这条命令补的是一个**记录了很久的空缺**：`Generator.chat()` 一直存在，
    但没有任何子命令调用它 —— 等于一段死 API。能力再弱，也得有入口才能被看见、被测量。
    """
    p = sub.add_parser("chat", help="对话（标记为 [用户]/[助手]，需在对话语料上微调过）")
    _add_model_args(p)
    _add_sampling_args(p)
    p.add_argument("-m", "--message", default=None, help="单轮提问；不给则进交互模式")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--system", default=None, help="可选的系统提示（拼在第一轮前）")


def cmd_chat(args: argparse.Namespace) -> int:
    gen = _load_generator(args)
    params = _sampling_from_args(args)
    history: list = []

    def one(text: str) -> str:
        # `Generator.chat()` 返回 **(reply, history) 元组** —— 不是字符串。
        # 直接把返回值当字符串用会在下游炸掉（Web 界面就这么踩过一次）。
        reply, hist = gen.chat(text, history=history, max_new_tokens=args.max_tokens,
                               params=params)
        history.clear()
        history.extend(hist)
        return reply

    if args.message:
        print(one(args.message))
        return 0

    print("CodeMind 对话（空行或 Ctrl+C 退出）")
    print("提示：标记是 [用户]/[助手]。若模型没在对话语料上微调过，"
          "它会接着写代码，而不是回答问题——那是已知状态，不是 bug。")
    while True:
        try:
            line = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        try:
            print("AI > " + one(line))
        except KeyboardInterrupt:
            print()
    return 0


def _add_zip(sub) -> None:
    """`zip` 子命令：把压缩包当成工作区。

    **读压缩包是工程问题，不是模型能力问题** —— 模型不需要"懂" ZIP，
    CLI 解开、映射路径、改完装回去即可。所以这个能力对任何规模的模型都立刻可用。
    """
    p = sub.add_parser("zip", help="把 ZIP 当工作区：列内容 / 解压 / 在其中补全或改代码")
    p.add_argument("archive", help="要处理的 .zip")
    p.add_argument("--list", action="store_true", help="列出包内可处理的代码文件")
    p.add_argument("--extract", default=None, metavar="DIR", help="解压到目录")
    p.add_argument("--all", action="store_true", help="连非代码文件一起处理（默认只处理代码）")
    p.add_argument("--path", default=None, help="包内目标文件（配合 --fim / --agent）")
    p.add_argument("--line", type=int, default=None, help="--fim 的光标行号（1 基）")
    p.add_argument("--instruction", default=None, help="--agent 的自然语言指令")
    p.add_argument("--rounds", type=int, default=3, help="--agent 自纠错轮数")
    p.add_argument("--max-tokens", type=int, default=128, help="--fim 最多生成多少 token")
    p.add_argument("--dry-run", action="store_true", help="只演示、不写回压缩包")
    _add_model_args(p)
    _add_sampling_args(p)


def _add_agent(sub) -> None:
    p = sub.add_parser("agent", help="按指令改代码，支持测试驱动的多轮自纠错")
    _add_model_args(p)
    _add_sampling_args(p)
    p.add_argument("-f", "--file", required=True, help="要修改的文件")
    p.add_argument("--zip", default=None, help="改为操作该 ZIP 内的文件（配合 --zip-path）")
    p.add_argument("--zip-path", default=None, help="ZIP 内的相对路径")
    p.add_argument("-i", "--instruction", required=True, help="自然语言修改指令")
    p.add_argument("--rounds", type=int, default=3, help="自纠错最大轮数")
    p.add_argument(
        "--check",
        action="append",
        default=None,
        help="校验命令，例如 --check 'python -m pytest -q'；失败则回滚并重试",
    )
    p.add_argument("--write", action="store_true", help="通过校验后写入文件（默认 dry-run）")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--show-diff", action="store_true", help="打印 diff")


def _add_info(sub) -> None:
    p = sub.add_parser("info", help="查看检查点信息")
    p.add_argument("--run", default="runs/tiny")
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--data", default=None, help="同时查看分片信息")


def _add_bench(sub) -> None:
    p = sub.add_parser(
        "bench",
        help="在目标机器上实测训练吞吐，并估算各预设的训练时长（换机器后先跑这个）",
    )
    p.add_argument(
        "--presets",
        nargs="*",
        default=["nano", "tiny", "small"],
        choices=sorted(MODEL_PRESETS),
        help="要测的模型预设",
    )
    p.add_argument("--threads", type=int, default=None, help="固定线程数；默认自动对比几档")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=8, help="每个配置的计时步数")
    p.add_argument("--ctx", type=int, default=None, help="覆盖上下文长度")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--budgets",
        nargs="*",
        type=float,
        default=[100e6, 300e6, 1e9, 2e9],
        help="要估算的训练 token 预算（可写 300e6 或 1e9）",
    )


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_benchmark

    run_benchmark(
        presets=args.presets,
        threads=args.threads,
        batch_size=args.batch_size,
        steps=args.steps,
        block_size=args.ctx,
        token_budgets=args.budgets,
        device_spec=args.device,
    )
    return 0


def _add_selftest(sub) -> None:
    p = sub.add_parser("selftest", help="端到端冒烟测试（几分钟内验证全链路可用）")
    p.add_argument("--dir", default=".codemind_selftest", help="临时工作目录")
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--keep", action="store_true", help="保留临时目录")


# ======================================================================================
# 各子命令实现
# ======================================================================================


def cmd_doctor(args: argparse.Namespace) -> int:
    print("=" * 68)
    print("CodeMind 环境自检")
    print("=" * 68)
    ok = True

    print(f"Python      : {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 10):
        print("  [X] 需要 Python >= 3.10")
        ok = False

    try:
        import torch

        print(f"PyTorch     : {torch.__version__}")
        print(f"CUDA 可用   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU         : {torch.cuda.get_device_name(0)}")
        mps = getattr(torch.backends, "mps", None)
        print(f"MPS 可用    : {bool(mps and mps.is_available())}")
    except ImportError:
        print("PyTorch     : 未安装  [X]")
        print("  安装方法: uv pip install torch  或  pip install torch")
        ok = False

    try:
        import numpy

        print(f"NumPy       : {numpy.__version__}")
    except ImportError:
        print("NumPy       : 未安装  [X]   (pip install numpy)")
        ok = False

    try:
        device = resolve_device(args.device)
        print(f"选定设备    : {device}")
    except ImportError:
        device = None
        ok = False

    if device is not None:
        try:
            import torch

            n_threads = torch.get_num_threads()
            print(f"CPU 线程数  : {n_threads} / 逻辑核 {os.cpu_count()}")
            if device.type == "cpu" and n_threads < 2:
                print("  [!] 线程数偏少，CPU 训练会很慢；用 --threads 显式指定")
        except Exception:
            pass

        if ok:
            try:
                from .model import GPT

                cfg = ModelConfig.from_preset("tiny")
                model = GPT(cfg).to(device)
                import torch

                x = torch.randint(0, cfg.vocab_size, (2, min(64, cfg.block_size)), device=device)
                t0 = time.time()
                _, loss, _ = model(x, x)
                fwd = time.time() - t0
                t0 = time.time()
                loss.backward()
                bwd = time.time() - t0
                print(f"冒烟前向    : {fwd*1000:.0f} ms / 反向 {bwd*1000:.0f} ms  (tiny 预设)")
                if device.type == "cpu" and fwd > 2.0:
                    print("  [!] 前向很慢：检查是否装到了无优化的 torch，或线程数设置过低")
            except Exception as exc:
                print(f"冒烟测试失败: {exc}")
                ok = False

    print("-" * 68)
    if ok:
        print("结论：环境可用。下一步执行 ->  codemind prepare --source .")
    else:
        print("结论：环境尚不可用，请先解决上面标 [X] 的问题。")
    print(describe_presets())
    return 0 if ok else 1


def cmd_presets(_args: argparse.Namespace) -> int:
    print(describe_presets())
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from .data import build_shards, collect_code_files
    from .tokenizer import BPETokenizer

    sources = args.source
    if not sources:
        default_sources = [p for p in ("codemind", "tests") if Path(p).exists()]
        if not default_sources:
            default_sources = ["."]
        sources = default_sources
        print(f"[prepare] 未指定 --source，使用默认目录: {', '.join(sources)}")

    vocab = args.vocab
    if args.preset:
        preset_vocab = MODEL_PRESETS[args.preset]["vocab_size"]
        if vocab == 4096 and preset_vocab != 4096:
            vocab = int(preset_vocab)
    vocab = max(280, vocab)
    # 用预设的 block_size 决定「单分片能否切成训练/验证两半」
    model_block_size = int(MODEL_PRESETS.get(args.preset, {}).get("block_size", 512))

    cfg = DataConfig(
        source_dirs=list(sources),
        out_dir=args.out,
        name=args.name,
        extensions=list(args.extensions or []),
        min_file_bytes=args.min_bytes,
        max_file_bytes=args.max_bytes,
        dedup=not args.no_dedup,
        max_files=args.max_files,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    files = collect_code_files(cfg)
    if not files:
        print("[prepare] 没有找到任何代码文件，请检查 --source 与后缀白名单")
        return 1

    total_bytes = 0
    for path in files:
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass
    print(
        f"[prepare] 原始语料 {total_bytes/1e6:.1f} MB / {len(files)} 个文件；"
        f"分词器样本上限 {args.tokenizer_chars/1e6:.1f} MB"
    )

    print(f"[prepare] 训练 BPE 分词器（目标词表 {vocab}）…")
    tokenizer = BPETokenizer()
    budget = args.tokenizer_chars
    rng = random.Random(args.seed)
    sample_files = list(files)
    rng.shuffle(sample_files)

    def _texts():
        """按需读取样本文件 —— 绝不要把整份语料读进内存。

        分词器训练只需要有代表性的样本（几十 MB 足够覆盖常见 n-gram），
        而 GB 级语料全量读入会直接 OOM。这里用生成器保证内存恒定。
        """
        used = 0
        for path in sample_files:
            text = path.read_text(encoding="utf-8", errors="replace")
            used += len(text)
            yield text
            if used >= budget:
                return

    tokenizer.train_from_texts(_texts(), vocab_size=vocab)
    print(f"[prepare] 分词器完成：{tokenizer.stats()}")

    shards = build_shards(
        files,
        tokenizer,
        cfg,
        block_size=model_block_size,
        shard_tokens=args.shard_tokens,
    )
    tok_path = Path(args.out) / args.name / "tokenizer.json"
    tokenizer.save(tok_path)
    manifest = shards.save_manifest(files)
    print(f"[prepare] 分词器已保存到 {tok_path}")
    print(f"[prepare] 文件清单已保存到 {manifest}（供后续流式重编码使用）")
    print(
        f"[prepare] 数据集 {shards.name}: {shards.total_tokens/1e6:.3f}M tokens, "
        f"{shards.n_docs} 文档，训练分片 {len(shards.train_files)} / 验证分片 {len(shards.val_files)}"
    )

    # 规模报告：直接回答「这份数据够训多大的模型、要多久」
    print()
    print("[prepare] 规模评估（按 Chinchilla 比例 ≈20 token/参数）：")
    for preset in ("nano", "tiny", "small", "medium"):
        cfg_p = ModelConfig.from_preset(preset)
        need = cfg_p.estimated_params() * 20
        status = "OK 数据充足" if shards.total_tokens >= need else f"需约 {need/1e6:.0f}M token"
        print(
            f"  {preset:<8} {cfg_p.estimated_params()/1e6:7.2f}M 参数 "
            f"需 {need/1e6:8.1f}M token  {status}"
        )
    print(
        f"\n  当前语料 {shards.total_tokens/1e6:.1f}M token。"
        f"不确定能训多久请先跑:  codemind bench --presets nano tiny small"
    )
    print(f"[prepare] 下一步 ->  codemind train --run runs/{args.name} --data {Path(args.out)/args.name}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .data import TokenShards
    from .tokenizer import BPETokenizer
    from .train import train

    overrides = {}
    if args.ctx is not None:
        overrides["block_size"] = args.ctx
    if args.layers is not None:
        overrides["n_layer"] = args.layers
    if args.heads is not None:
        overrides["n_head"] = args.heads
    if args.kv_heads is not None:
        overrides["n_kv_head"] = args.kv_heads
    if args.embd is not None:
        overrides["n_embd"] = args.embd
    if args.dropout:
        overrides["dropout"] = args.dropout

    model_cfg = ModelConfig.from_preset(args.preset, **overrides)

    data_dir = Path(args.data)
    shards = TokenShards.load(data_dir)
    if shards.vocab_size and shards.vocab_size != model_cfg.vocab_size:
        print(
            f"[train] 词表不匹配：数据是 {shards.vocab_size}，模型预设是 {model_cfg.vocab_size}。"
            f"已自动对齐到 {shards.vocab_size}"
        )
        model_cfg.vocab_size = int(shards.vocab_size)

    tok_path = Path(args.run) / "tokenizer.json"
    if not tok_path.exists():
        source_tok = shards.dir / "tokenizer.json"
        if source_tok.exists():
            tok_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy2(source_tok, tok_path)
        else:
            print(f"[train] 找不到分词器 {tok_path}，请先运行 codemind prepare")
            return 1
    tokenizer = BPETokenizer.load(tok_path)
    if tokenizer.vocab_size != model_cfg.vocab_size:
        print(
            f"[train] 分词器词表 {tokenizer.vocab_size} != 模型词表 {model_cfg.vocab_size}，"
            f"已对齐到分词器"
        )
        model_cfg.vocab_size = tokenizer.vocab_size

    print(f"[train] 模型配置: {summarize_config(model_cfg)}")
    train_cfg = TrainConfig(
        out_dir=args.run,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        grad_checkpoint=bool(args.grad_checkpoint),
        max_iters=args.steps,
        warmup_iters=args.warmup,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        log_interval=args.log_interval,
        ckpt_interval=args.ckpt_interval,
        amp=args.amp,
        compile=args.compile,
        seed=args.seed,
        num_threads=args.threads,
        fim_rate=args.fim_rate,
        fim_spm_rate=args.fim_spm_rate,
        fim_terminator=args.fim_terminator,
        resume=not args.no_resume,
        device=args.device,
    )
    report = train(model_cfg, train_cfg, shards=shards, tokenizer=tokenizer)
    print(
        f"\n[train] 试着补全一下：\n"
        f"  codemind gen --run {args.run} --prompt 'def add(a, b):' --max-tokens 64"
    )
    return 0 if report.iterations > 0 else 1


def _load_generator(args: argparse.Namespace):
    from .generate import load_runner

    return load_runner(Path(args.run), checkpoint=args.checkpoint, device=args.device)


def _sampling_from_args(args: argparse.Namespace):
    from .model import SamplingParams

    return SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        greedy=args.greedy,
        seed=args.seed,
    )


def _resolve_prompt(args: argparse.Namespace) -> Optional[str]:
    if args.prompt is not None:
        return args.prompt
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
        if args.tail and args.tail > 0:
            text = text[-args.tail :]
        return text
    return None


def _stream_writer():
    """返回一个把 token 片段立刻写到 stdout 的回调（用于交互场景）。"""

    def on_token(_tid: int, piece: str) -> None:
        if piece:
            sys.stdout.write(piece)
            sys.stdout.flush()

    return on_token


def cmd_gen(args: argparse.Namespace) -> int:
    prompt = _resolve_prompt(args)
    if prompt is None:
        print("[gen] 需要 --prompt 或 -f/--file")
        return 2
    gen = _load_generator(args)
    params = _sampling_from_args(args)

    if args.show_prompt:
        sys.stderr.write(prompt)
        if not prompt.endswith("\n"):
            sys.stderr.write("\n")
        sys.stderr.flush()

    # 只有真实终端才逐字流式输出；管道/重定向下一次性输出，避免内容重复。
    use_stream = not args.no_stream and sys.stdout.isatty()

    text = gen.complete(
        prompt,
        max_new_tokens=args.max_tokens,
        params=params,
        stop_sequences=tuple(args.stop or ()),
        on_token=_stream_writer() if use_stream else None,
    )
    if not use_stream:
        sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


def cmd_fim(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"[fim] 文件不存在: {path}")
        return 2
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines(keepends=True)
    if args.line is None:
        cut_line = len(lines)
    else:
        cut_line = max(0, min(args.line - 1, len(lines)))
    prefix = "".join(lines[:cut_line])
    suffix = "".join(lines[cut_line:])
    if args.before:
        prefix = prefix[-args.before :]
    if args.after:
        suffix = suffix[: args.after]

    gen = _load_generator(args)
    params = _sampling_from_args(args)
    print(f"[fim] 光标位置: 第 {cut_line + 1} 行 / 共 {len(lines)} 行", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    print(prefix[-400:], file=sys.stderr, end="")
    print("<<<< 光标 >>>>", file=sys.stderr)
    print(suffix[:400], file=sys.stderr, end="")
    print("-" * 60, file=sys.stderr)

    addition = gen.infill(prefix, suffix, max_new_tokens=args.max_tokens, params=params)
    # 补全结果走 stdout（保持可被管道消费），上下文预览与校验报告走 stderr
    sys.stdout.write(addition)
    if not addition.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()

    if args.write or args.dry_run:
        editor = CodeEditor(root=path.parent if str(path.parent) else Path("."))
        new_content = prefix + addition + suffix
        result = editor.propose(path, new_content, dry_run=args.dry_run)
        print(result.report(), file=sys.stderr)
        if not result.success:
            return 1
    return 0


def cmd_repl(args: argparse.Namespace) -> int:
    gen = _load_generator(args)
    params = _sampling_from_args(args)
    print(f"CodeMind 交互式补全 | 上下文 {gen.block_size} tokens | Ctrl-C / 空行结束")
    print("输入一段代码，模型会接着往下写；以 ':' 开头的行是命令（:quit 退出，:temp 0.5 调温）")
    buffer: List[str] = []
    while True:
        try:
            line = input(">>> " if not buffer else "... ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.startswith(":"):
            cmd = line[1:].strip()
            if cmd in ("quit", "q", "exit"):
                break
            if cmd.startswith("temp"):
                try:
                    params.temperature = float(cmd.split()[1])
                    print(f"温度已设为 {params.temperature}")
                except (IndexError, ValueError):
                    print("用法: :temp 0.5")
            elif cmd == "clear":
                buffer.clear()
                print("已清空缓冲区")
            else:
                print("可用命令: :quit :clear :temp <值>")
            continue
        if not line.strip() and not buffer:
            break
        buffer.append(line)
        prompt = "\n".join(buffer) + "\n"
        # REPL 一定逐字流式：等待整段生成完再显示会非常难用
        addition = gen.complete(
            prompt,
            max_new_tokens=args.max_tokens,
            params=params,
            on_token=_stream_writer(),
        )
        if not addition.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        buffer.append(addition.rstrip("\n"))
    print("再见。")
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f"[agent] 文件不存在: {path}")
        return 2
    gen = _load_generator(args)
    params = _sampling_from_args(args)
    editor = CodeEditor(root=Path.cwd())
    original = path.read_text(encoding="utf-8", errors="replace")
    checks = [c.split() for c in (args.check or [])]
    dry_run = not args.write

    print(f"[agent] 目标文件: {path}")
    print(f"[agent] 指令: {args.instruction}")
    print(f"[agent] 模式: {'dry-run（不写入）' if dry_run else '写入'} | 自纠错轮数 {args.rounds}")
    if checks:
        print(f"[agent] 校验命令: {'; '.join(' '.join(c) for c in checks)}")

    def propose(round_index: int, feedback: Optional[str]) -> Optional[str]:
        print(f"\n[agent] --- 第 {round_index} 轮生成 ---")
        prompt = build_edit_prompt(
            args.instruction, path, original, feedback=feedback
        )
        text = gen.complete(prompt, max_new_tokens=args.max_tokens, params=params)
        candidate = extract_code_block(text)
        if candidate is None:
            return None
        if not candidate.endswith("\n"):
            candidate += "\n"
        return candidate

    outcome = editor.self_repair(
        path, propose, max_rounds=args.rounds, checks=checks, dry_run=dry_run
    )
    print()
    print(outcome.report())
    if args.show_diff and outcome.success:
        print("\n" + unified_diff(original, outcome.final_content, str(path)))
    if outcome.success and dry_run:
        print("\n[dry-run] 未写入文件。确认无误后加 --write 提交。")
    if not outcome.success:
        print(
            "\n提示：从零训练的小模型产出可用代码的概率有限。可尝试：\n"
            "  1) 更多训练步数与语料（--steps / prepare --source 增加目录）\n"
            "  2) 降低温度（--temperature 0.2 --greedy）\n"
            "  3) 增加自纠错轮数并挂上真实测试（--rounds 5 --check 'python -m pytest -q'）"
        )
        return 1
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .generate import describe_checkpoint

    run_dir = Path(args.run)
    name = args.checkpoint if args.checkpoint.endswith(".pt") else f"{args.checkpoint}.pt"
    ckpt = run_dir / name
    if not ckpt.exists():
        candidates = sorted(run_dir.glob("*.pt")) if run_dir.exists() else []
        if not candidates:
            print(f"[info] {run_dir} 下没有检查点")
            return 1
        ckpt = candidates[-1]
    info = describe_checkpoint(ckpt)
    print(f"检查点      : {info['path']}")
    print(f"格式        : {info['format']}")
    print(f"训练步数    : {info['iteration']}")
    print(f"最优验证损失: {info['best_val']}")
    print(f"文件大小    : {info['size_mb']:.2f} MB")
    print(f"分词器      : {info['tokenizer_path']}")
    cfg = ModelConfig.from_dict(dict(info["model_config"]))  # type: ignore[arg-type]
    print(f"模型配置    : {summarize_config(cfg)}")
    log_file = Path(info["path"]).parent / "log.json"
    if log_file.exists():
        import json

        payload = json.loads(log_file.read_text(encoding="utf-8"))
        final = payload.get("final", {})
        print(
            f"训练日志    : {log_file} (tokens/秒 {final.get('tokens_per_second')}, "
            f"总 tokens {final.get('tokens_seen')})"
        )
    if args.data:
        from .data import TokenShards

        shards = TokenShards.load(Path(args.data))
        print(
            f"数据集      : {shards.name} | {shards.total_tokens/1e6:.3f}M tokens | "
            f"{shards.n_docs} 文档 | 词表 {shards.vocab_size}"
        )
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """端到端冒烟：小词表 + 小模型 + 少量步数，验证全链路真的能跑通。"""
    import shutil

    from .data import build_shards, collect_code_files
    from .tokenizer import BPETokenizer
    from .train import train

    work = Path(args.dir)
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    print(f"[selftest] 工作目录: {work.resolve()}")

    # 1) 用两个极小语料文件保证可复现，不依赖仓库内容
    corpus = work / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    sample_code = (
        "def add(a, b):\n    return a + b\n\n\n"
        "def sub(a, b):\n    return a - b\n\n\n"
        "class Calculator:\n"
        "    def __init__(self):\n        self.value = 0\n\n"
        "    def add(self, x):\n        self.value += x\n        return self.value\n\n"
        "    def reset(self):\n        self.value = 0\n        return self\n"
    )
    for i in range(40):
        (corpus / f"sample_{i}.py").write_text(
            sample_code + f"\n\ndef variant_{i}(x):\n    return x * {i}\n", encoding="utf-8"
        )

    import random as _random

    print("[selftest] 1/5 训练分词器…")
    tokenizer = BPETokenizer()
    tokenizer.train_from_texts(
        [p.read_text(encoding="utf-8") for p in corpus.glob("*.py")],
        vocab_size=args.vocab,
        verbose=False,
    )
    probe = "def add(a, b):\n    return a + b\n"
    ids = tokenizer.encode(probe)
    assert tokenizer.decode(ids) == probe, "分词器往返编码失败"
    print(f"  词表 {tokenizer.vocab_size}，往返编码 OK，压缩率 {len(probe)/len(ids):.2f} 字符/token")
    tok_path = work / "tokenizer.json"
    tokenizer.save(tok_path)
    assert BPETokenizer.load(tok_path).vocab_size == tokenizer.vocab_size
    print("  分词器落盘/加载 OK")

    print("[selftest] 2/5 构建训练分片…")
    data_cfg = DataConfig(
        source_dirs=[str(corpus)], out_dir=str(work / "data"), name="self", val_ratio=0.2
    )
    files = collect_code_files(data_cfg, verbose=False)
    shards = build_shards(files, tokenizer, data_cfg, verbose=False)
    print(f"  {shards.total_tokens} tokens / {shards.n_docs} 文档")

    print("[selftest] 3/5 训练（FIM 模式）…")
    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_kv_head=1,
        block_size=64,
    )
    train_cfg = TrainConfig(
        out_dir=str(work / "run"),
        batch_size=8,
        max_iters=args.steps,
        lr=3e-3,
        eval_interval=max(10, args.steps // 3),
        eval_batches=4,
        log_interval=max(10, args.steps // 6),
        ckpt_interval=args.steps,
        fim_rate=0.5,
        resume=False,
        device="cpu",
        seed=0,
    )
    report = train(model_cfg, train_cfg, shards=shards, tokenizer=tokenizer, verbose=True)
    assert report.best_val < 12.0, f"验证损失异常: {report.best_val}"
    print(f"  损失 {report.final_train:.4f}（随机基线约 {__import__('math').log(tokenizer.vocab_size):.4f}）")

    print("[selftest] 4/5 载入并生成…")
    from .generate import load_runner

    gen = load_runner(work / "run", checkpoint="best", device="cpu")
    out = gen.complete("def add(a, b):\n", max_new_tokens=16)
    print(f"  续写输出: {out!r}")
    filled = gen.infill("def add(a, b):\n", "\n\ndef sub(a, b):\n", max_new_tokens=16)
    print(f"  填空输出: {filled!r}")

    print("[selftest] 5/5 编辑器的校验与回滚…")
    # 注意：CodeEditor.resolve() 会给相对路径拼上 root，所以这里统一用绝对路径，
    # 避免出现 ".codemind_selftest/.codemind_selftest/..." 这种双重拼接。
    editor = CodeEditor(root=work)
    target = (work / "target.py").resolve()
    target.write_text("x = 1\n", encoding="utf-8")

    bad = editor.propose(target, "def broken(:\n    pass\n")
    after_bad = target.read_text(encoding="utf-8")
    assert not bad.success, "非法代码不应被接受"
    assert after_bad == "x = 1\n", f"坏代码未被回滚，文件变成 {after_bad!r}"
    print(f"  语法拦截 OK（{bad.error}）")

    good = editor.propose(target, "x = 2\ny = 3\n")
    after_good = target.read_text(encoding="utf-8")
    assert good.success, f"合法改动被拒绝：error={good.error!r}"
    assert after_good == "x = 2\ny = 3\n", f"写入内容不符：{after_good!r}"
    print("  语法拦截 + 回滚 OK；合法改动写入 OK")

    # 编辑区的路径归一化必须自洽：绝对路径不能被再次拼接 root；
    # resolve() 返回的是已归一化的绝对路径，比较时也要先归一化。
    assert editor.resolve(target) == target
    assert editor.resolve(Path("sub/a.py")) == (work / "sub/a.py").resolve()

    print("\n" + "=" * 68)
    print("✅ 全链路自检通过：分词 -> 分片 -> 训练 -> 生成 -> 编辑校验")
    print("=" * 68)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
        print("（临时目录已清理，加 --keep 可保留）")
    return 0


# ======================================================================================
# 入口
# ======================================================================================

def cmd_zip(args: argparse.Namespace) -> int:
    """把 ZIP 当工作区：列内容 / 解压 / 在其中跑 `fim` 或 `agent`。

    **这是工程能力，不是模型能力** —— 所以无论模型多小都立刻可用。
    模型只需要在解出来的普通文件上工作。
    """
    from .zipfs import ZipWorkspace
    from .generate import load_runner

    archive = Path(args.archive)
    with ZipWorkspace(archive, include_all=args.all) as ws:
        files = list(ws.iter_code_files())
        print(f"[zip] {archive.name}: {len(files)} 个可处理文件 "
              f"（{ws.stats['bytes']/1e6:.2f} MB，跳过 {ws.stats['skipped']} 个非代码文件）")

        if args.list or (not args.extract and not args.path):
            for f in files[:200]:
                size = len(ws.read_bytes(f))
                print(f"  {size:>9,}  {f}")
            if len(files) > 200:
                print(f"  … 另有 {len(files)-200} 个")
            if not args.extract and not args.path:
                return 0

        if args.extract:
            dest = ws.materialize_to(args.extract)
            print(f"[zip] 已解压到 {dest}")
            return 0

        if not args.path:
            print("[zip] 需要 --path 指定包内文件（或用 --list / --extract）")
            return 2
        if not ws.exists(args.path):
            print(f"[zip] 包内没有 {args.path}")
            return 2

        before = ws.read(args.path)
        tokenizer = getattr(args, "tokenizer", None)

        if args.instruction:
            # ⚠️ 不猜 agent 的内部 API —— 直接对**解出来的真实文件**调用同一个
            # `agent` 命令（子进程），复用整条「改写 + 自纠错 + 校验 + 回滚」链路。
            # 踩过的坑：我先前按想象中的函数名 `run_agent_edit` 去调，那个函数不存在。
            gen = load_runner(
                Path(args.run), checkpoint=args.checkpoint, device=args.device,
                tokenizer_path=Path(tokenizer) if tokenizer else None,
            )
            del gen  # 只是为了尽早暴露检查点问题；真正的生成在子进程里做
            tmp_file = ws.root / args.path
            print(f"[zip] 对 {args.path} 执行指令：{args.instruction}")
            cmd = [
                sys.executable, "-m", "codemind", "agent",
                "--run", str(args.run), "--checkpoint", str(args.checkpoint),
                "--device", str(args.device),
                "-f", str(tmp_file), "-i", args.instruction,
                "--rounds", str(args.rounds), "--write",
            ]
            if tokenizer:
                cmd += ["--tokenizer", str(tokenizer)]
            proc = subprocess.run(cmd, cwd=str(ws.root), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace")
            print(proc.stdout.strip()[-1500:] if proc.stdout else "")
            if proc.returncode != 0:
                print(f"[zip] agent 退出码 {proc.returncode}，压缩包**未改动**")
                if proc.stderr:
                    print(proc.stderr.strip()[-600:])
                return 1
            after = tmp_file.read_text(encoding="utf-8")
        elif args.line is not None or args.max_tokens:
            gen = load_runner(
                Path(args.run), checkpoint=args.checkpoint, device=args.device,
                tokenizer_path=Path(tokenizer) if tokenizer else None,
            )
            lines = before.splitlines(keepends=True)
            line = args.line or len(lines)
            prefix = "".join(lines[: line - 1])
            suffix = "".join(lines[line - 1:])
            addition = gen.infill(prefix, suffix, max_new_tokens=args.max_tokens)
            after = prefix + addition + suffix
            print(f"[zip] 在 {args.path}:{line} 补全 {len(addition)} 字符")
        else:
            print("[zip] 需要 --instruction（改代码）或 --line/--max-tokens（补全）")
            return 2

        if args.dry_run:
            print("[zip] --dry-run：不写回。生成结果如下：")
            print("-" * 60)
            print(after[:2000])
            print("-" * 60)
            return 0

        ws.write(args.path, after)
        changed = ws.commit()
        print(f"[zip] 已写回 {archive.name}，改动文件：{changed}")
    return 0


COMMANDS = {
    "doctor": cmd_doctor,
    "presets": cmd_presets,
    "prepare": cmd_prepare,
    "train": cmd_train,
    "gen": cmd_gen,
    "fim": cmd_fim,
    "repl": cmd_repl,
    "agent": cmd_agent,
    "chat": cmd_chat,
    "zip": cmd_zip,
    "info": cmd_info,
    "bench": cmd_bench,
    "selftest": cmd_selftest,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMANDS[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
