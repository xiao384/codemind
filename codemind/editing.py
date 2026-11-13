"""文件级代码编辑：补丁应用、语法校验、回滚，以及测试驱动的自纠错循环。

核心思想
--------
语言模型的输出天生不可靠，所以「改代码」这件事必须被工程化地兜住：

1. 模型只负责**提出**改动（整文件重写或最小片段替换）；
2. 程序负责**校验**改动（语法树能否编译、测试是否通过）；
3. 校验失败时把**真实错误信息**回灌给模型重试，最多 N 轮；
4. 任何一轮失败都**完整回滚**到原始内容，绝不留下半成品烂文件。

这一层是「能不能安心用」的分界线：没有它，模型输出只是幻觉文本；
有了它，最坏情况也只是「什么都没改」。
"""

from __future__ import annotations

import difflib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "FileSnapshot",
    "SyntaxCheckResult",
    "CommandResult",
    "RepairAttempt",
    "RepairOutcome",
    "check_python_syntax",
    "run_command",
    "read_code_file",
    "write_code_file",
    "unified_diff",
    "extract_code_block",
    "CodeEditor",
]

# --------------------------------------------------------------------------------------
# 基础校验
# --------------------------------------------------------------------------------------


@dataclass
class SyntaxCheckResult:
    ok: bool
    message: str = ""
    line: Optional[int] = None

    def __bool__(self) -> bool:  # 允许 `if check: ...`
        return self.ok


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    seconds: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def tail(self, limit: int = 2000) -> str:
        text = (self.stdout + ("\n" + self.stderr if self.stderr else "")).strip()
        if len(text) <= limit:
            return text
        return "…" + text[-limit:]


def check_python_syntax(source: str, filename: str = "<string>") -> SyntaxCheckResult:
    """检查 Python 源码能否编译；返回可读的错误定位。"""
    try:
        compile(source, filename, "exec", dont_inherit=True)
        return SyntaxCheckResult(True)
    except SyntaxError as exc:
        location = f"第 {exc.lineno} 行" if exc.lineno else "未知行"
        if exc.offset:
            location += f" 第 {exc.offset} 列"
        return SyntaxCheckResult(False, f"{exc.msg}（{location}）", exc.lineno)
    except ValueError as exc:  # 例如源码里含空字节
        return SyntaxCheckResult(False, str(exc))


def run_command(
    command: Sequence[str],
    cwd: Optional[Path] = None,
    timeout: int = 120,
    env: Optional[Dict[str, str]] = None,
) -> CommandResult:
    """执行外部命令并收集输出（用于跑测试/静态检查）。"""
    t0 = time.time()
    display = " ".join(command)
    try:
        proc = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return CommandResult(display, proc.returncode, proc.stdout or "", proc.stderr or "", time.time() - t0)
    except FileNotFoundError as exc:
        return CommandResult(display, 127, "", f"命令不存在: {exc}", time.time() - t0)
    except subprocess.TimeoutExpired:
        return CommandResult(display, 124, "", f"命令超时（{timeout}s）", time.time() - t0)


def read_code_file(path: Path) -> str:
    path = Path(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def detect_newline(path: Path) -> str:
    """检测文件当前的行尾风格，返回 ``"\\n"`` 或 ``"\\r\\n"``；读不到时按 LF。"""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return "\n"
    idx = data.find(b"\n")
    if idx > 0 and data[idx - 1 : idx] == b"\r":
        return "\r\n"
    return "\n"


def write_code_file(path: Path, content: str, newline: Optional[str] = None) -> None:
    """把内容写回文件，**保留文件原有的行尾风格**。

    ``content`` 内部统一按 LF 处理（``read_code_file`` 也会归一成 LF），
    落盘时再按目标风格展开。否则一次单行改动会把整个 CRLF 文件变成 LF ——
    在 git 里表现为"整个文件都被改了"，而用户只是想改一行。

    ``newline`` 显式给出时按它落盘（``"\\n"`` / ``"\\r\\n"``）；
    文件不存在时默认 LF。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if newline is None:
        newline = detect_newline(path)
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    if newline == "\r\n":
        text = text.replace("\n", "\r\n")
    path.write_text(text, encoding="utf-8", newline="")


def unified_diff(old: str, new: str, path: str = "file") -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3,
    )
    return "".join(diff)


# --------------------------------------------------------------------------------------
# 模型输出解析
# --------------------------------------------------------------------------------------


def extract_code_block(text: str) -> Optional[str]:
    """从模型输出里抽出代码块。

    依次尝试：```lang 围栏 -> 裸 ``` 围栏 -> 整体文本（若看起来像代码）。
    小模型经常吐出不完整的围栏，所以这里要尽量宽容。
    """
    if not text:
        return None
    fence_pos = text.find("```")
    if fence_pos >= 0:
        rest = text[fence_pos + 3 :]
        newline = rest.find("\n")
        if newline >= 0:
            body = rest[newline + 1 :]
            end = body.find("```")
            if end >= 0:
                return body[:end]
            return body  # 截断的输出：把剩下的都当代码
    stripped = text.strip()
    return stripped or None


# --------------------------------------------------------------------------------------
# 编辑会话
# --------------------------------------------------------------------------------------


@dataclass
class FileSnapshot:
    """文件原始内容快照，用于回滚与 diff。

    ``raw`` 保留**原始字节**，``restore()`` 按字节写回。这一点很重要：
    ``propose()`` 承诺「success=False 时文件保证未被修改」，而按内容重写
    会顺手把行尾（CRLF -> LF）或 BOM 改掉 —— 那就等于"没改成，但文件被动过了"。
    """

    path: Path
    content: str
    existed: bool
    raw: bytes = b""

    @classmethod
    def capture(cls, path: Path) -> "FileSnapshot":
        path = Path(path)
        if path.exists():
            raw = path.read_bytes()
            return cls(path, raw.decode("utf-8", "replace"), True, raw)
        return cls(path, "", False, b"")

    def restore(self) -> None:
        if self.existed:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(self.raw)          # 逐字节还原，含行尾与 BOM
        elif self.path.exists():
            self.path.unlink()


@dataclass
class RepairAttempt:
    round_index: int
    candidate: str
    syntax: Optional[SyntaxCheckResult] = None
    command: Optional[CommandResult] = None
    accepted: bool = False
    note: str = ""


@dataclass
class RepairOutcome:
    path: Path
    success: bool
    attempts: List[RepairAttempt] = field(default_factory=list)
    final_content: str = ""
    error: str = ""

    @property
    def rounds_used(self) -> int:
        return len(self.attempts)

    def report(self) -> str:
        lines = [
            f"目标文件: {self.path}",
            f"结果: {'✅ 成功（已写入）' if self.success else '❌ 失败（已回滚，文件保持原样）'}",
            f"尝试轮数: {self.rounds_used}",
        ]
        for a in self.attempts:
            status = "通过" if a.accepted else "否决"
            detail = a.note
            if a.syntax is not None and not a.syntax.ok:
                detail = f"语法错误：{a.syntax.message}"
            elif a.command is not None and not a.command.ok:
                detail = f"命令失败（exit {a.command.exit_code}）：{a.command.tail(400)}"
            lines.append(f"  第 {a.round_index} 轮 [{status}] {detail}")
        if self.error:
            lines.append(f"  说明: {self.error}")
        return "\n".join(lines)


class CodeEditor:
    """带校验与回滚的文件编辑器。

    ``editor = CodeEditor(".")``
    ``outcome = editor.propose(Path("a.py"), new_content)``
    """

    def __init__(
        self,
        root: Path = Path("."),
        *,
        backup_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.backup_dir = Path(backup_dir) if backup_dir else self.root / ".codemind_backup"
        self.verbose = verbose
        self._snapshots: Dict[Path, FileSnapshot] = {}

    # ------------------------------------------------------------------ 路径
    def resolve(self, path: Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else (self.root / path)

    def is_inside_root(self, path: Path) -> bool:
        try:
            self.resolve(path).resolve().relative_to(self.root)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------ 校验
    def validate(self, path: Path, content: str) -> SyntaxCheckResult:
        """按文件扩展名做语法校验；暂只覆盖 Python（其余语言直接放行）。"""
        suffix = self.resolve(path).suffix.lower()
        if suffix == ".py":
            return check_python_syntax(content, str(path))
        if suffix in (".json",):
            try:
                json.loads(content)
                return SyntaxCheckResult(True)
            except json.JSONDecodeError as exc:
                return SyntaxCheckResult(False, f"JSON 解析失败：{exc.msg}（第 {exc.lineno} 行）", exc.lineno)
        return SyntaxCheckResult(True)

    def run_checks(
        self,
        commands: Sequence[Sequence[str]],
        cwd: Optional[Path] = None,
        timeout: int = 120,
    ) -> List[CommandResult]:
        return [run_command(cmd, cwd=cwd or self.root, timeout=timeout) for cmd in commands]

    # ------------------------------------------------------------------ 写入
    def snapshot(self, path: Path) -> FileSnapshot:
        real = self.resolve(path)
        snap = FileSnapshot.capture(real)
        self._snapshots[real] = snap
        return snap

    def rollback(self, path: Optional[Path] = None) -> None:
        targets = [self.resolve(path)] if path is not None else list(self._snapshots)
        for real in targets:
            snap = self._snapshots.get(real)
            if snap is not None:
                snap.restore()

    def backup(self, path: Path) -> Optional[Path]:
        """把当前文件复制一份到备份目录（带时间戳）。"""
        real = self.resolve(path)
        if not real.exists():
            return None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = self.backup_dir / f"{real.name}.{stamp}.bak"
        shutil.copy2(real, target)
        return target

    def propose(
        self,
        path: Path,
        new_content: str,
        *,
        dry_run: bool = False,
        checks: Sequence[Sequence[str]] = (),
        require_change: bool = True,
    ) -> RepairOutcome:
        """提交一份改动：校验 -> （可选）跑检查 -> 写入或回滚。

        返回 ``RepairOutcome``；``success=False`` 时文件**保证**未被修改。
        """
        real = self.resolve(path)
        snap = self._snapshots.get(real) or self.snapshot(real)
        attempt = RepairAttempt(round_index=1, candidate=new_content, note="")

        if require_change and new_content.strip() == snap.content.strip():
            attempt.note = "内容与原文一致，无可提交的改动"
            return RepairOutcome(real, False, [attempt], snap.content, "模型没有产出有效改动")

        syntax = self.validate(real, new_content)
        attempt.syntax = syntax
        if not syntax.ok:
            return RepairOutcome(real, False, [attempt], snap.content, "语法校验未通过")

        if dry_run:
            attempt.accepted = True
            attempt.note = "dry-run：仅校验，未写入"
            return RepairOutcome(real, True, [attempt], new_content, "")

        result = self._write(real, new_content)
        if not result.ok:
            return RepairOutcome(real, False, [attempt], snap.content, "写入失败")

        for cmd in checks:
            command_result = run_command(cmd, cwd=self.root)
            attempt.command = command_result
            if not command_result.ok:
                snap.restore()
                attempt.note = "检查命令失败，已回滚"
                return RepairOutcome(
                    real,
                    False,
                    [attempt],
                    snap.content,
                    f"检查命令失败: {' '.join(cmd)}",
                )

        attempt.accepted = True
        attempt.note = "校验通过并已写入"
        return RepairOutcome(real, True, [attempt], new_content, "")

    def _write(self, path: Path, content: str) -> CommandResult:
        try:
            self.backup(path)
            write_code_file(path, content)
            return CommandResult(f"write {path}", 0, "", "", 0.0)
        except OSError as exc:
            return CommandResult(f"write {path}", 1, "", str(exc), 0.0)

    # ------------------------------------------------------------------ 自纠错循环
    def self_repair(
        self,
        path: Path,
        propose_fn: Callable[[int, Optional[str]], Optional[str]],
        *,
        max_rounds: int = 3,
        checks: Sequence[Sequence[str]] = (),
        dry_run: bool = False,
    ) -> RepairOutcome:
        """测试驱动的自纠错循环。

        ``propose_fn(round_index, feedback)`` 负责产出候选内容：
        第一轮 ``feedback=None``，之后传入上一轮的真实报错文本。
        返回的 ``None`` 表示放弃。
        """
        real = self.resolve(path)
        snap = self.snapshot(real)
        attempts: List[RepairAttempt] = []
        feedback: Optional[str] = None

        for round_index in range(1, max_rounds + 1):
            candidate = propose_fn(round_index, feedback)
            if candidate is None:
                attempts.append(
                    RepairAttempt(round_index, "", note="模型未产出候选内容")
                )
                feedback = "上一轮没有产出任何代码，请直接输出完整的文件内容。"
                continue

            attempt = RepairAttempt(round_index=round_index, candidate=candidate)
            syntax = self.validate(real, candidate)
            attempt.syntax = syntax
            if not syntax.ok:
                attempt.note = "语法错误"
                attempts.append(attempt)
                feedback = (
                    f"上一轮的代码无法通过语法检查：{syntax.message}\n"
                    "请修正后重新输出完整文件内容。"
                )
                continue

            if candidate.strip() == snap.content.strip():
                attempt.note = "与原文一致"
                attempts.append(attempt)
                feedback = "上一轮输出与原文完全相同，请输出真正做了修改的完整文件内容。"
                continue

            if dry_run:
                attempt.accepted = True
                attempt.note = "dry-run 通过（未写入）"
                attempts.append(attempt)
                return RepairOutcome(real, True, attempts, candidate, "")

            write_result = self._write(real, candidate)
            if not write_result.ok:
                attempt.note = "写入失败"
                attempts.append(attempt)
                snap.restore()
                return RepairOutcome(real, False, attempts, snap.content, write_result.stderr)

            failed_command: Optional[CommandResult] = None
            for cmd in checks:
                command_result = run_command(cmd, cwd=self.root)
                attempt.command = command_result
                if not command_result.ok:
                    failed_command = command_result
                    break
            if failed_command is None:
                attempt.accepted = True
                attempt.note = "语法与检查全部通过"
                attempts.append(attempt)
                return RepairOutcome(real, True, attempts, candidate, "")

            snap.restore()
            attempt.note = f"检查失败并回滚：{failed_command.command}"
            attempts.append(attempt)
            feedback = (
                f"上一轮代码在运行检查时失败：\n{failed_command.tail(1500)}\n"
                "请修正后重新输出完整文件内容。"
            )

        snap.restore()
        return RepairOutcome(
            real, False, attempts, snap.content, f"{max_rounds} 轮内未通过全部校验，已回滚"
        )


# --------------------------------------------------------------------------------------
# 提示构造
# --------------------------------------------------------------------------------------


# 文件后缀 -> 代码块语言标签。模型在预训练里见到的围栏就是这些名字，
# 直接用后缀（如 "py"）会与训练分布不一致。
_LANGUAGE_TAGS: Dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php", ".java": "java",
    ".kt": "kotlin", ".swift": "swift", ".cs": "csharp", ".cpp": "cpp",
    ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".c": "c", ".h": "c",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".ps1": "powershell",
    ".sql": "sql", ".html": "html", ".css": "css", ".scss": "scss",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".md": "markdown", ".xml": "xml", ".lua": "lua", ".r": "r",
}


def language_tag(path: Path) -> str:
    """把文件路径映射为代码块语言标签（未知后缀则退化为去掉点后缀）。"""
    suffix = Path(path).suffix.lower()
    return _LANGUAGE_TAGS.get(suffix, suffix.lstrip(".") or "text")


def build_edit_prompt(
    instruction: str,
    path: Path,
    content: str,
    *,
    feedback: Optional[str] = None,
    error_context: Optional[str] = None,
) -> str:
    """构造「让模型重写整个文件」的提示。

    选择整文件重写而不是 diff，是因为小模型的 diff 行号对齐能力极差；
    整文件重写虽然费 token，但配合语法校验 + 回滚，实际可用性高得多。
    """
    tag = language_tag(path)
    parts = [
        "# Task\n",
        f"Rewrite the file `{Path(path).name}` completely according to the instruction.\n",
        f"Instruction: {instruction}\n\n",
        "# Rules\n",
        f"- Output the ENTIRE new content of the file in one ```{tag} code block.\n",
        "- Do not explain. Do not add commentary outside the code block.\n",
        "- Keep the original code style and indentation.\n",
        "- The result must be syntactically valid.\n\n",
        "# Current file content\n",
        f"```{tag}\n{content}\n```\n",
    ]
    if error_context:
        parts.append(f"\n# Previous failure output\n```\n{error_context[:2000]}\n```\n")
    if feedback:
        parts.append(f"\n# Reviewer feedback\n{feedback}\n")
    parts.append("\n# New file content\n")
    return "".join(parts)
