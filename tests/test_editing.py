"""编辑层测试：脚本注入式自纠错循环、语法拦截、回滚、命令校验。

这组测试用「脚本化的假模型」替代真模型，因此完全确定性、不需要训练。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pytest

from codemind.editing import (
    CodeEditor,
    build_edit_prompt,
    check_python_syntax,
    extract_code_block,
    read_code_file,
    run_command,
    unified_diff,
)


# ---------------------------------------------------------------------------- 语法校验


@pytest.mark.parametrize(
    "source",
    ["x = 1\n", "def f():\n    return 1\n", "", "# 只有注释\n", "class A:\n    pass\n"],
)
def test_valid_python(source: str):
    assert check_python_syntax(source)


@pytest.mark.parametrize(
    "source",
    ["def f(:\n    pass\n", "x = \n", "if True\n    pass\n", "return 1\n", "def f():\npass\n"],
)
def test_invalid_python(source: str):
    result = check_python_syntax(source)
    assert not result
    assert result.message


def test_syntax_error_reports_line():
    """未闭合括号由 Python 报在**开括号所在行**（不是文件末尾）。"""
    result = check_python_syntax("x = 1\ny = (\n", "t.py")
    assert not result
    assert result.line == 2

    result = check_python_syntax("x = 1\ndef f(:\n    pass\n", "t.py")
    assert not result
    assert result.line == 2
    assert "第 2 行" in result.message


# ---------------------------------------------------------------------------- 命令执行


def test_run_command_success():
    result = run_command([sys.executable, "-c", "print('hello')"])
    assert result.ok
    assert "hello" in result.stdout


def test_run_command_failure_captures_stderr():
    result = run_command([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"])
    assert not result.ok
    assert result.exit_code == 3
    assert "boom" in result.tail()


def test_run_command_missing_binary():
    result = run_command(["definitely-not-a-real-binary-xyz"])
    assert not result.ok
    assert result.exit_code == 127


def test_run_command_timeout():
    result = run_command([sys.executable, "-c", "import time; time.sleep(5)"], timeout=1)
    assert result.exit_code == 124


# ---------------------------------------------------------------------------- 输出解析


def test_extract_fenced_block():
    text = "解释一下\n```python\nx = 1\n```\n完事"
    assert extract_code_block(text) == "x = 1\n"


def test_extract_unclosed_fence():
    assert extract_code_block("```python\nx = 1\n") == "x = 1\n"


def test_extract_plain_text():
    assert extract_code_block("x = 1\n") == "x = 1"


def test_extract_empty():
    assert extract_code_block("") is None


def test_unified_diff_marks_changes():
    diff = unified_diff("a\n", "b\n", "f.py")
    assert "-a" in diff and "+b" in diff


# ---------------------------------------------------------------------------- 编辑器


def test_propose_writes_valid_file(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    outcome = editor.propose(target, "x = 2\ny = 3\n")
    assert outcome.success
    assert target.read_text(encoding="utf-8") == "x = 2\ny = 3\n"


def test_propose_rejects_syntax_error_without_touching_file(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    outcome = editor.propose(target, "def broken(:\n")
    assert not outcome.success
    assert target.read_text(encoding="utf-8") == "x = 1\n", "非法改动绝不能落盘"


def test_propose_dry_run_does_not_write(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    outcome = editor.propose(target, "x = 99\n", dry_run=True)
    assert outcome.success
    assert target.read_text(encoding="utf-8") == "x = 1\n"


def test_propose_requires_actual_change(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    outcome = editor.propose(target, "x = 1\n")
    assert not outcome.success
    assert "没有产出" in outcome.error or "一致" in outcome.attempts[0].note


def test_propose_rolls_back_on_failing_check(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    failing = [sys.executable, "-c", "import sys; sys.exit(1)"]
    outcome = editor.propose(target, "x = 2\n", checks=[failing])
    assert not outcome.success
    assert target.read_text(encoding="utf-8") == "x = 1\n", "检查失败必须回滚"


def test_propose_accepts_when_check_passes(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    passing = [sys.executable, "-c", "print('ok')"]
    outcome = editor.propose(target, "x = 2\n", checks=[passing])
    assert outcome.success
    assert target.read_text(encoding="utf-8") == "x = 2\n"


def test_backup_created(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, backup_dir=tmp_path / "bak", verbose=False)
    editor.propose(target, "x = 2\n")
    backups = list((tmp_path / "bak").glob("*.bak"))
    assert backups, "写入前应留备份"
    assert backups[0].read_text(encoding="utf-8") == "x = 1\n"


def test_json_validation(tmp_path: Path):
    target = tmp_path / "c.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    assert not editor.propose(target, "{not json}").success
    assert editor.propose(target, '{"a": 2}\n').success


# ---------------------------------------------------------------- 行尾（CRLF）保真
#
# 实测背景（不是推断）：
#   read_code_file 走通用换行，内容里的 \r 被归一掉；write_code_file 用 newline=""
#   落盘，于是 **CRLF 文件会被静默改成 LF**。
#   后果一：一次单行改动在 git 里表现为整个文件被改。
#   后果二（更严重）：propose 的文档承诺「success=False 时文件保证未被修改」，
#   而回滚路径 FileSnapshot.restore() 走的是同一个写入函数 —— 回滚也会改行尾，
#   等于"没改成，但文件被动过了"。

CRLF_SRC = "def add(a, b):\r\n    return a + b\r\n"


def test_crlf_rollback_is_byte_exact(tmp_path: Path):
    """检查失败回滚后，文件字节必须与原文完全一致（含行尾）。"""
    target = tmp_path / "a.py"
    target.write_bytes(CRLF_SRC.encode("utf-8"))
    before = target.read_bytes()

    editor = CodeEditor(root=tmp_path, verbose=False)
    failing = [sys.executable, "-c", "import sys; sys.exit(1)"]
    outcome = editor.propose(target, "def add(a, b):\r\n    return a + b + 1\r\n",
                             checks=[failing])
    assert not outcome.success
    assert target.read_bytes() == before, "回滚必须逐字节还原，不能顺手改掉行尾"


def test_crlf_successful_edit_preserves_crlf(tmp_path: Path):
    """改成功的文件应当保留原有行尾风格。

    必须走**真实流程**：read_code_file（会把 CRLF 归一成 LF）-> 改动 -> 写回。
    直接把 CRLF 内容传给 propose 是不真实的，那样 newline="" 会原样保留 CRLF，
    反而掩盖问题。
    """
    target = tmp_path / "a.py"
    target.write_bytes(CRLF_SRC.encode("utf-8"))

    editor = CodeEditor(root=tmp_path, verbose=False)
    edited = read_code_file(target).replace("return a + b", "return a + b + 1")
    assert "\r" not in edited, "前置条件：读取后内容已归一成 LF"
    outcome = editor.propose(target, edited)
    assert outcome.success
    raw = target.read_bytes()
    assert b"+ 1" in raw
    assert raw.count(b"\r\n") >= 2, "改动后仍应是 CRLF 行尾"
    assert raw.count(b"\n") == raw.count(b"\r\n"), "不应混入裸 LF"


def test_lf_file_stays_lf(tmp_path: Path):
    """对照：LF 文件不应被改成 CRLF。"""
    target = tmp_path / "b.py"
    target.write_bytes("x = 1\ny = 2\n".encode("utf-8"))
    editor = CodeEditor(root=tmp_path, verbose=False)
    assert editor.propose(target, "x = 1\ny = 3\n").success
    raw = target.read_bytes()
    assert b"\r" not in raw
    assert raw == "x = 1\ny = 3\n".encode("utf-8")


def test_new_file_gets_lf(tmp_path: Path):
    """新文件没有历史行尾可继承，应当是 LF。"""
    target = tmp_path / "new.py"
    editor = CodeEditor(root=tmp_path, verbose=False)
    assert editor.propose(target, "z = 9\n").success
    assert target.read_bytes() == b"z = 9\n"


def test_non_python_files_skip_syntax_check(tmp_path: Path):
    target = tmp_path / "notes.txt"
    target.write_text("hello\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    assert editor.propose(target, "anything ((( \n").success


def test_new_file_creation(tmp_path: Path):
    target = tmp_path / "pkg" / "new.py"
    editor = CodeEditor(root=tmp_path, verbose=False)
    outcome = editor.propose(target, "value = 42\n")
    assert outcome.success
    assert target.read_text(encoding="utf-8") == "value = 42\n"


# ---------------------------------------------------------------------------- 自纠错循环


def _scripted(candidates: List[Optional[str]]):
    """构造 propose_fn：按轮次返回预设候选，记录每轮收到的 feedback。"""
    seen_feedback: List[Optional[str]] = []

    def propose(round_index: int, feedback: Optional[str]) -> Optional[str]:
        seen_feedback.append(feedback)
        if round_index - 1 < len(candidates):
            return candidates[round_index - 1]
        return None

    return propose, seen_feedback


def test_self_repair_succeeds_after_bad_round(tmp_path: Path):
    """第 1 轮语法错误 -> 第 2 轮修正 -> 通过。这正是不接大模型也能验证的核心逻辑。"""
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    propose, feedback = _scripted(["def broken(:\n    pass\n", "def fixed():\n    return 1\n"])
    outcome = editor.self_repair(target, propose, max_rounds=3)
    assert outcome.success, outcome.report()
    assert outcome.rounds_used == 2
    assert target.read_text(encoding="utf-8") == "def fixed():\n    return 1\n"
    assert feedback[0] is None, "第一轮不应有 feedback"
    assert feedback[1] is not None and "语法" in feedback[1], "第二轮必须收到语法错误反馈"


def test_self_repair_gives_up_and_rolls_back(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    propose, _ = _scripted(["bad(\n", "worse(\n", "still bad(\n"])
    outcome = editor.self_repair(target, propose, max_rounds=3)
    assert not outcome.success
    assert outcome.rounds_used == 3
    assert target.read_text(encoding="utf-8") == "x = 1\n", "全部失败必须回滚"


def test_self_repair_handles_none_candidate(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    propose, feedback = _scripted([None, "x = 5\n"])
    outcome = editor.self_repair(target, propose, max_rounds=3)
    assert outcome.success
    assert feedback[1] is not None and "没有产出" in feedback[1]


def test_self_repair_rejects_identical_output(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    propose, feedback = _scripted(["x = 1\n", "x = 1\n", "x = 99\n"])
    outcome = editor.self_repair(target, propose, max_rounds=3)
    assert outcome.success
    assert outcome.rounds_used == 3
    assert feedback[1] is not None and "相同" in feedback[1]


def test_self_repair_uses_check_output_as_feedback(tmp_path: Path):
    """测试失败时，必须把真实的测试输出回灌给下一轮。"""
    target = tmp_path / "a.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    check = [
        sys.executable,
        "-c",
        f"import runpy,sys; g=runpy.run_path(r'{target}'); "
        "sys.stderr.write('AssertionError: VALUE should be 7'); "
        "sys.exit(1 if g['VALUE'] != 7 else 0)",
    ]
    propose, feedback = _scripted(["VALUE = 2\n", "VALUE = 7\n"])
    outcome = editor.self_repair(target, propose, max_rounds=3, checks=[check])
    assert outcome.success, outcome.report()
    assert outcome.rounds_used == 2
    assert target.read_text(encoding="utf-8") == "VALUE = 7\n"
    assert feedback[1] is not None and "VALUE should be 7" in feedback[1], (
        "真实测试报错必须回灌给模型"
    )


def test_self_repair_dry_run_never_writes(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    propose, _ = _scripted(["x = 2\n"])
    outcome = editor.self_repair(target, propose, max_rounds=2, dry_run=True)
    assert outcome.success
    assert target.read_text(encoding="utf-8") == "x = 1\n"


def test_outcome_report_is_readable(tmp_path: Path):
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    editor = CodeEditor(root=tmp_path, verbose=False)
    propose, _ = _scripted(["bad(\n", "x = 3\n"])
    outcome = editor.self_repair(target, propose, max_rounds=2)
    report = outcome.report()
    assert "成功" in report
    assert "尝试轮数: 2" in report


# ---------------------------------------------------------------------------- 提示构造


def test_build_edit_prompt_contains_essentials():
    prompt = build_edit_prompt("添加类型注解", Path("a.py"), "def f(x):\n    return x\n")
    assert "a.py" in prompt
    assert "添加类型注解" in prompt
    assert "def f(x):" in prompt
    assert "```python" in prompt, "Python 文件必须用 python 语言标签而不是 py"


def test_language_tag_mapping():
    from codemind.editing import language_tag

    assert language_tag(Path("a.py")) == "python"
    assert language_tag(Path("a.ts")) == "typescript"
    assert language_tag(Path("a.cpp")) == "cpp"
    assert language_tag(Path("a.unknownext")) == "unknownext"
    assert language_tag(Path("Makefile")) == "text"


def test_build_edit_prompt_includes_feedback_and_errors():
    prompt = build_edit_prompt(
        "fix", Path("a.py"), "x = 1\n", feedback="上一轮语法错误", error_context="Traceback..."
    )
    assert "上一轮语法错误" in prompt
    assert "Traceback" in prompt


def test_build_edit_prompt_truncates_long_error():
    prompt = build_edit_prompt("fix", Path("a.py"), "x = 1\n", error_context="E" * 5000)
    assert len(prompt) < 5000
