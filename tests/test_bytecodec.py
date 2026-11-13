"""字节编解码层的测试（纯 Python，无第三方依赖）。"""

from __future__ import annotations

import pytest

from codemind.bytecodec import (
    bytes_to_symbols,
    bytes_to_unicode,
    char_to_byte,
    byte_to_char,
    symbols_to_bytes,
)


def test_mapping_is_bijective():
    table = bytes_to_unicode()
    assert len(table) == 256
    assert len(set(table.values())) == 256, "映射必须是一一对应"
    for byte, char in table.items():
        assert char_to_byte(char) == byte
        assert byte_to_char(byte) == char


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"def add(a, b):\n    return a + b\n",
        b"\x00\x01\x02\xff\xfe",
        "中文注释：测试".encode("utf-8"),
        bytes(range(256)),
    ],
)
def test_roundtrip_arbitrary_bytes(data: bytes):
    assert symbols_to_bytes(bytes_to_symbols(data)) == data


def test_symbols_are_single_characters():
    symbols = bytes_to_symbols(b"abc\xff")
    assert len(symbols) == 4
    assert all(len(s) == 1 for s in symbols)


def test_invalid_symbol_raises():
    with pytest.raises(ValueError):
        symbols_to_bytes(["中"])
