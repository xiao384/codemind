"""把 ZIP 当成工作区用 —— 让 `fim` / `agent` / `gen` 直接操作压缩包里的文件。

**为什么放在这里而不是让模型去"理解" ZIP**：
读压缩包是**工程问题**，不是模型能力问题。模型不需要知道 ZIP 格式，
CLI 把它解开、把文件路径映射好、改完再装回去就行。
把它做进工具层，任何规模的模型都能立刻用上。

设计要点：

1. **透明**：用完和普通目录一样，`read/write/exists/iter_code_files` 的接口对齐 `pathlib` 的常用子集。
2. **只在真改过时才回写**：逐文件比对，没变的原样复用原压缩数据，避免整包重压。
3. **原子替换**：先写临时包，成功后再替换原文件，中途失败不会毁掉用户的压缩包。
4. **防目录穿越（zip-slip）**：归档里带 `../` 的条目一律拒绝。
5. **默认不解压整包**：只解出需要的条目，并对总大小设上限。

用法::

    with ZipWorkspace("proj.zip") as ws:
        src = ws.read("pkg/mod.py")
        ws.write("pkg/mod.py", src.replace("foo", "bar"))
    # 退出时自动回写

单独列内容::

    python -m codemind zip --list proj.zip
"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator, Optional

#: 可被当作代码处理的扩展名（与 prepare 的采集范围保持一致）
CODE_SUFFIXES = {
    ".py", ".pyi", ".c", ".h", ".cc", ".cpp", ".hpp", ".cxx",
    ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb",
    ".sh", ".ps1", ".sql", ".md", ".txt", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml",
}

#: 单次解压总字节上限（防止误开巨型归档）
DEFAULT_MAX_BYTES = 512 * 1024 * 1024


class ZipSlipError(ValueError):
    """归档条目试图逃出目标目录。"""


def _safe_member(name: str) -> bool:
    """拒绝绝对路径与 `..` 穿越。"""
    p = name.replace("\\", "/")
    if p.startswith("/") or (len(p) > 1 and p[1] == ":"):
        return False
    parts = [x for x in p.split("/") if x not in ("", ".")]
    return ".." not in parts


class ZipWorkspace:
    """把 ZIP 解到临时目录，改完再装回去。

    只有**内容真的变了**的文件才会被重写；其余条目原样复用。
    """

    def __init__(self, archive: "str | Path", *, max_bytes: int = DEFAULT_MAX_BYTES,
                 include_all: bool = False) -> None:
        self.archive = Path(archive)
        self.max_bytes = max_bytes
        self.include_all = include_all
        self._tmp: Optional[tempfile.TemporaryDirectory] = None
        self._root: Optional[Path] = None
        self._orig: dict[str, bytes] = {}   # 相对路径 -> 原始字节
        self._meta: dict[str, zipfile.ZipInfo] = {}
        self.stats = {"files": 0, "bytes": 0, "skipped": 0}

    # ---- 生命周期 ----------------------------------------------------
    def __enter__(self) -> "ZipWorkspace":
        if not self.archive.is_file():
            raise FileNotFoundError(f"找不到压缩包：{self.archive}")
        if not zipfile.is_zipfile(self.archive):
            raise ValueError(f"不是有效的 ZIP：{self.archive}")
        self._tmp = tempfile.TemporaryDirectory(prefix="codemind_zip_")
        self._root = Path(self._tmp.name)
        total = 0
        with zipfile.ZipFile(self.archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if not _safe_member(info.filename):
                    raise ZipSlipError(f"归档条目试图逃出目录：{info.filename!r}")
                if not self.include_all:
                    ext = Path(info.filename).suffix.lower()
                    if ext not in CODE_SUFFIXES:
                        self.stats["skipped"] += 1
                        continue
                total += info.file_size
                if total > self.max_bytes:
                    raise ValueError(
                        f"解压体积超过上限 {self.max_bytes/1e6:.0f} MB，"
                        f"如确需处理请调大 max_bytes"
                    )
                data = zf.read(info)
                dst = self._root / info.filename
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                self._orig[info.filename] = data
                self._meta[info.filename] = info
                self.stats["files"] += 1
                self.stats["bytes"] += len(data)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.commit()
        finally:
            if self._tmp is not None:
                self._tmp.cleanup()
                self._tmp = None

    # ---- 文件接口 ----------------------------------------------------
    @property
    def root(self) -> Path:
        if self._root is None:
            raise RuntimeError("工作区尚未打开，请用 with 语句")
        return self._root

    def exists(self, rel: str) -> bool:
        return (self.root / rel).is_file()

    def read(self, rel: str, encoding: str = "utf-8") -> str:
        return (self.root / rel).read_text(encoding=encoding)

    def read_bytes(self, rel: str) -> bytes:
        return (self.root / rel).read_bytes()

    def write(self, rel: str, content: "str | bytes", encoding: str = "utf-8") -> None:
        """写入包内文件。

        ⚠️ **必须 `newline=""`**：`Path.write_text` 在 Windows 上默认把 `\\n` 翻成 `\\r\\n`，
        会把压缩包里每个被改写的文件都悄悄改成 CRLF。实测踩过——
        写入 `def f():\\n    ...` 读回变成 `def f():\\r\\n    ...`。
        这与 `editing.py` 里修过的行尾缺陷是**同一类**，只是走了新路径。
        """
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            dst.write_bytes(content)
        else:
            with open(dst, "w", encoding=encoding, newline="") as f:
                f.write(content)

    def iter_code_files(self) -> Iterator[str]:
        """按相对路径遍历包内的代码文件（已按扩展名过滤）。"""
        return iter(sorted(self._orig))

    def changed(self) -> list[str]:
        out = []
        for rel, before in self._orig.items():
            now = (self.root / rel).read_bytes()
            if now != before:
                out.append(rel)
        return out

    # ---- 回写 --------------------------------------------------------
    def commit(self) -> list[str]:
        """把改动写回压缩包（原子替换）。返回被改动的相对路径列表。"""
        changed = self.changed()
        if not changed:
            return []
        # 新增的文件（不在 _orig 里）也要算上
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.root).as_posix()
                if rel not in self._orig:
                    changed.append(rel)
        tmp_out = self.archive.with_suffix(self.archive.suffix + ".tmp")
        try:
            with zipfile.ZipFile(self.archive) as zin, \
                    zipfile.ZipFile(tmp_out, "w", zipfile.ZIP_DEFLATED) as zout:
                written = set()
                for info in zin.infolist():
                    if info.is_dir():
                        zout.writestr(info, b"")
                        continue
                    if info.filename in self._orig and info.filename in changed:
                        zout.writestr(info, (self.root / info.filename).read_bytes())
                    else:
                        zout.writestr(info, zin.read(info))
                    written.add(info.filename)
                # 新增文件
                for rel in changed:
                    if rel not in written:
                        zout.writestr(rel, (self.root / rel).read_bytes())
            tmp_out.replace(self.archive)   # 原子替换
        except Exception:
            tmp_out.unlink(missing_ok=True)
            raise
        return changed

    # ---- 便捷构造 ----------------------------------------------------
    def materialize_to(self, dest: "str | Path") -> Path:
        """把整个工作区拷到一个持久目录（用于 `--extract`）。"""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.root)
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out)
        return dest
