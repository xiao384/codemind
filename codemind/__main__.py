"""``python -m codemind`` 入口。

顺带处理 Windows 控制台编码问题：PowerShell 5.1 默认按 GBK 解码子进程输出，
而 Python 3 默认写 UTF-8，中文会变成乱码。这里显式把标准流切到 UTF-8。
"""

from __future__ import annotations

import sys


def _force_utf8_stdio() -> None:
    """尽力把 stdout/stderr 切到 UTF-8；失败时保持原样（不影响功能）。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def main() -> int:
    from .cli import main as cli_main

    _force_utf8_stdio()
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
