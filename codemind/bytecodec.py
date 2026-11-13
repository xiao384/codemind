"""字节 ↔ 可见字符的可逆映射（GPT-2 风格 byte-level 编码）。

为什么需要这一层：BPE 的合并操作建立在「符号序列」上。如果直接用原始字节，
换行、制表符等控制字符会以不可见形式进入词表，调试和落盘都很痛苦。
本模块把 256 个字节双射到 256 个可打印字符，保证：

* 任意二进制/非法 UTF-8 输入都能被无损编码，编码器永不抛异常；
* 词表文件（tokenizer.json）保持人类可读；
* ``decode(encode(x)) == x`` 对任意字节串成立。
"""

from __future__ import annotations

from typing import Dict, List

__all__ = ["bytes_to_unicode", "byte_to_char", "char_to_byte", "bytes_to_symbols", "symbols_to_bytes"]


def bytes_to_unicode() -> Dict[int, str]:
    """构造 256 个字节到可见字符的双射表。"""
    # 可见区间原样保留：'!'..'~'、'¡'..'¬'、'®'..'ÿ'
    printable = list(range(ord("!"), ord("~") + 1))
    printable += list(range(ord("\xa1"), ord("\xac") + 1))
    printable += list(range(ord("\xae"), ord("\xff") + 1))
    mapped: List[int] = list(printable)

    # 其余字节顺序映射到 256 之后的连续码位
    n = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + n)
            n += 1

    return {byte: chr(cp) for byte, cp in zip(printable, mapped)}


_BYTE_TO_CHAR: Dict[int, str] = bytes_to_unicode()
_CHAR_TO_BYTE: Dict[str, int] = {ch: byte for byte, ch in _BYTE_TO_CHAR.items()}


def byte_to_char(byte: int) -> str:
    return _BYTE_TO_CHAR[byte]


def char_to_byte(char: str) -> int:
    return _CHAR_TO_BYTE[char]


def bytes_to_symbols(data: bytes) -> List[str]:
    """``b"def f(x):"`` -> ``["d", "e", "f", ...]``（每个元素是 1 个字符的符号）。"""
    return [_BYTE_TO_CHAR[b] for b in data]


def symbols_to_bytes(symbols: List[str]) -> bytes:
    table = _CHAR_TO_BYTE
    try:
        return bytes(table[s] for s in symbols)
    except KeyError as exc:  # pragma: no cover - 只在词表被外部破坏时触发
        raise ValueError(f"符号 {exc.args[0]!r} 不属于字节字母表") from exc


def is_byte_alphabet(text: str) -> bool:
    """校验整串字符是否都落在字节字母表内。"""
    missing = [c for c in set(text) if c not in _CHAR_TO_BYTE]
    return not missing
