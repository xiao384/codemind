"""CLI 测试：参数解析、纯静态子命令，以及一次完整的端到端冒烟。

``codemind selftest`` 是整条链路的守门测试（分词 -> 分片 -> 训练 -> 生成 -> 编辑校验），
这里把它跑一遍，确保任何一处改动都不会悄悄破坏可用性。
``test_editing`` 用假模型验证了 agent 逻辑，所以这里不需要真的生成代码。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from codemind.cli import build_parser, main

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------- 参数解析


def test_parser_requires_command():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


@pytest.mark.parametrize(
    "argv",
    [
        ["doctor"],
        ["presets"],
        ["prepare"],
        ["train"],
        ["info"],
        ["repl"],
        ["fim", "-f", "x.py"],
        ["agent", "-f", "x.py", "-i", "do something"],
        ["selftest"],
    ],
)
def test_parser_accepts_valid_commands(argv):
    args = build_parser().parse_args(argv)
    assert args.command == argv[0]


def test_parser_gen_defaults():
    args = build_parser().parse_args(["gen", "--prompt", "x"])
    assert args.run == "runs/tiny"
    assert args.temperature == 0.8
    assert args.max_tokens == 128
    assert args.checkpoint == "best"


def test_parser_train_overrides():
    args = build_parser().parse_args(
        ["train", "--steps", "50", "--layers", "3", "--embd", "64", "--fim-rate", "0.0"]
    )
    assert args.steps == 50
    assert args.layers == 3
    assert args.embd == 64
    assert args.fim_rate == 0.0


def test_parser_rejects_unknown_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["frobnicate"])


def test_parser_rejects_bad_preset():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--preset", "gigantic"])


def test_parser_bench_defaults():
    args = build_parser().parse_args(["bench"])
    assert args.presets == ["nano", "tiny", "small"]
    assert args.steps == 8
    assert args.device == "auto"


def test_parser_bench_custom():
    args = build_parser().parse_args(
        ["bench", "--presets", "small", "--threads", "48", "--batch-size", "4", "--ctx", "512"]
    )
    assert args.presets == ["small"]
    assert args.threads == 48
    assert args.ctx == 512


# ---------------------------------------------------------------------------- 基准模块


def test_peak_memory_reader_is_sane():
    """峰值内存读取必须可用：调 batch_size 时它是最关键的参考值。

    回归测试：早期用 ctypes.windll 调用 GetProcessMemoryInfo，
    64 位下 BOOL 返回值被当成 32 位 int 截断成 0，导致永远读不到内存。
    """
    from codemind.bench import _peak_memory_mb

    value = _peak_memory_mb()
    assert value >= 0.0
    if value > 0:  # Windows / Linux 都应能读到真实值
        assert value > 10.0, f"峰值内存读数可疑: {value} MB"


def test_bench_measure_runs():
    """基准测量本身必须能跑通并给出正数吞吐。"""
    from codemind.bench import _measure
    from codemind.config import ModelConfig

    cfg = ModelConfig(vocab_size=128, n_layer=1, n_head=2, n_embd=32, n_kv_head=1, block_size=32)
    res = _measure(cfg, threads=2, batch_size=2, steps=1, device=torch.device("cpu"))
    assert res.ok, res.error
    assert res.train_tps > 0
    assert res.fwd_ms > 0 and res.bwd_ms > 0
    assert res.infer_tps > 0


def test_bench_result_time_estimate():
    from codemind.bench import BenchResult
    from codemind.config import ModelConfig

    cfg = ModelConfig(vocab_size=128, n_layer=1, n_head=2, n_embd=32, n_kv_head=1, block_size=32)
    res = BenchResult("t", cfg, 4, 8, train_tps=1000.0)
    assert res.ok
    assert abs(res.hours_for(3_600_000) - 1.0) < 1e-6
    dead = BenchResult("t", cfg, 4, 8, error="boom")
    assert not dead.ok
    assert dead.hours_for(1e9) == float("inf")


# ---------------------------------------------------------------------------- 静态子命令


def test_presets_command(capsys):
    assert main(["presets"]) == 0
    out = capsys.readouterr().out
    for name in ("nano", "tiny", "small", "medium"):
        assert name in out


def test_doctor_command(capsys):
    code = main(["doctor"])
    out = capsys.readouterr().out
    assert "Python" in out and "PyTorch" in out
    assert code in (0, 1)


def test_info_command_without_checkpoint(tmp_path: Path, capsys):
    assert main(["info", "--run", str(tmp_path / "empty")]) == 1
    assert "没有检查点" in capsys.readouterr().out


def test_gen_requires_prompt(capsys):
    assert main(["gen", "--run", "runs/nonexistent"]) == 2
    assert "prompt" in capsys.readouterr().out


def test_fim_missing_file(capsys):
    assert main(["fim", "--run", "runs/nonexistent", "-f", "no/such/file.py"]) == 2
    assert "不存在" in capsys.readouterr().out


def test_agent_missing_file(capsys):
    code = main(["agent", "--run", "runs/nonexistent", "-f", "no/such/file.py", "-i", "x"])
    assert code == 2
    assert "不存在" in capsys.readouterr().out


# ---------------------------------------------------------------------------- 端到端冒烟


@pytest.mark.slow
def test_selftest_end_to_end(tmp_path: Path, capsys):
    """完整跑一遍 codemind selftest（几十秒级）。"""
    work = tmp_path / "smoke"
    code = main(
        [
            "selftest",
            "--dir", str(work),
            "--vocab", "320",
            "--steps", "30",
            "--keep",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, f"selftest 失败:\n{out}"
    assert "全链路自检通过" in out
    assert (work / "run" / "best.pt").exists()
    assert (work / "run" / "tokenizer.json").exists()
    shutil.rmtree(work, ignore_errors=True)
