"""`zipfs` 的回归测试。

每条都对应一个**实测踩过或差点踩到**的行为，不是凑覆盖率：

* 行尾保真 —— 第一版忘了 `newline=""`，回写后 `\\n` 全变 `\\r\\n`（Windows 默认翻译）；
* 只改有变动的文件 —— 否则每次都要整包重压；
* 非代码文件**不列出但要保留** —— 否则压缩包会被"清理"掉二进制资源；
* zip-slip —— 归档条目带 `../` 必须拒绝，这是安全底线。
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from codemind.zipfs import ZipSlipError, ZipWorkspace


def _make_zip(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return path


def test_read_write_roundtrip(tmp_path: Path) -> None:
    zp = _make_zip(tmp_path / "a.zip", {"pkg/m.py": b"def f():\n    return 1\n"})
    with ZipWorkspace(zp) as ws:
        assert ws.read("pkg/m.py") == "def f():\n    return 1\n"
        ws.write("pkg/m.py", "def f():\n    return 2\n")
    with zipfile.ZipFile(zp) as z:
        assert z.read("pkg/m.py") == b"def f():\n    return 2\n"


def test_no_crlf_translation(tmp_path: Path) -> None:
    """⚠️ Windows 上 `Path.write_text` 会把 \\n 翻成 \\r\\n，必须用 newline=""。"""
    zp = _make_zip(tmp_path / "b.zip", {"m.py": b"a\nb\n"})
    with ZipWorkspace(zp) as ws:
        ws.write("m.py", "x\ny\n")
    with zipfile.ZipFile(zp) as z:
        raw = z.read("m.py")
    assert b"\r" not in raw, f"回写引入了 CRLF：{raw!r}"
    assert raw == b"x\ny\n"


def test_only_changed_entries_rewritten(tmp_path: Path) -> None:
    zp = _make_zip(tmp_path / "c.zip", {"a.py": b"A\n", "b.py": b"B\n"})
    with ZipWorkspace(zp) as ws:
        ws.write("a.py", "A2\n")
        changed = ws.commit()
    assert changed == ["a.py"]
    with zipfile.ZipFile(zp) as z:
        assert z.read("b.py") == b"B\n"      # 未变动的原样
        assert z.read("a.py") == b"A2\n"


def test_non_code_listed_but_preserved(tmp_path: Path) -> None:
    zp = _make_zip(tmp_path / "d.zip",
                   {"m.py": b"x=1\n", "blob.bin": b"\x00\x01\x02", "big.dat": b"zz"})
    with ZipWorkspace(zp) as ws:
        assert list(ws.iter_code_files()) == ["m.py"]        # 只列代码
        assert ws.stats["skipped"] == 2
        ws.write("m.py", "x=2\n")
    with zipfile.ZipFile(zp) as z:
        assert z.read("blob.bin") == b"\x00\x01\x02"         # 二进制逐字节保留
        assert z.read("big.dat") == b"zz"


def test_zip_slip_rejected(tmp_path: Path) -> None:
    zp = _make_zip(tmp_path / "e.zip", {"../../pwned.py": b"bad"})
    with pytest.raises(ZipSlipError):
        with ZipWorkspace(zp):
            pass


def test_absolute_path_rejected(tmp_path: Path) -> None:
    zp = _make_zip(tmp_path / "f.zip", {"/etc/passwd": b"bad"})
    with pytest.raises(ZipSlipError):
        with ZipWorkspace(zp):
            pass


def test_not_a_zip_raises(tmp_path: Path) -> None:
    p = tmp_path / "notzip.zip"
    p.write_bytes(b"this is not a zip")
    with pytest.raises(ValueError):
        with ZipWorkspace(p):
            pass


def test_exception_leaves_archive_untouched(tmp_path: Path) -> None:
    """在 `with` 里抛异常时**不能回写**——避免半成品写进用户的压缩包。"""
    zp = _make_zip(tmp_path / "g.zip", {"a.py": b"orig\n"})
    before = zp.read_bytes()
    with pytest.raises(RuntimeError):
        with ZipWorkspace(zp) as ws:
            ws.write("a.py", "半成品\n")
            raise RuntimeError("模拟中途失败")
    assert zp.read_bytes() == before


def test_materialize_to(tmp_path: Path) -> None:
    zp = _make_zip(tmp_path / "h.zip", {"p/q.py": b"Q\n", "r.md": b"# r\n"})
    out = tmp_path / "extracted"
    with ZipWorkspace(zp) as ws:
        ws.materialize_to(out)
    assert (out / "p" / "q.py").read_bytes() == b"Q\n"
    assert (out / "r.md").read_bytes() == b"# r\n"
