"""对话格式的回归测试。

**这组测试守的是一条已经踩过的坑**：训练语料与推理提示词的格式必须逐字一致。
不一致的症状是"训完还是不会聊天"，而且**没有任何报错**——最难查的一类。

同时守住一个更根深蒂固的坑：**标记绝不能用 `#` 开头**。
旧实现用 `## User` / `## Assistant`，而 `#` 在 Python 里是注释，
模型把整段对话读成注释块，然后接着写代码。实测问「你好，你是谁？」
得到的是 `# Remove the pre-process registry...`。
"""
from __future__ import annotations

from typing import List, Tuple

from codemind.generate import BOT_TAG, USER_TAG, Generator


def _mk() -> Tuple[Generator, List[str]]:
    """造一个不加载权重的 Generator，并把 `complete` 换成记录器。"""
    gen = Generator.__new__(Generator)
    seen: List[str] = []

    def fake_complete(prompt, *, max_new_tokens=128, params=None,
                      stop_sequences=(), on_token=None, use_cache=True):
        seen.append(prompt)
        return "这是回答。"

    gen.complete = fake_complete        # type: ignore[method-assign]
    return gen, seen


def test_tags_do_not_start_with_hash() -> None:
    """⚠️ 核心约束：`#` 在 Python 里是注释，会让模型把对话当注释块。"""
    assert not USER_TAG.startswith("#"), f"USER_TAG 不能以 # 开头：{USER_TAG!r}"
    assert not BOT_TAG.startswith("#"), f"BOT_TAG 不能以 # 开头：{BOT_TAG!r}"


def test_markers_are_exactly_the_training_format() -> None:
    """字面量锁死 —— 改了这里就必须同步改 `tools/make_chat_corpus.py` 生成的语料。"""
    assert USER_TAG == "[用户] "
    assert BOT_TAG == "[助手] "


def test_chat_prompt_uses_new_markers() -> None:
    gen, seen = _mk()
    gen.chat("你好", max_new_tokens=8)
    assert seen, "chat() 没有调用 complete()"
    prompt = seen[0]
    assert USER_TAG in prompt and BOT_TAG in prompt
    # 旧标记必须彻底消失
    assert "## User" not in prompt, "旧标记 ## User 仍在提示词里"
    assert "## Assistant" not in prompt, "旧标记 ## Assistant 仍在提示词里"
    assert prompt.rstrip().endswith(BOT_TAG.rstrip()), "提示词应以 [助手] 结尾"


def test_chat_prompt_shape_matches_corpus() -> None:
    """语料里是 `[用户] 问题\\n[助手] 回答`，提示词必须是它的前缀。"""
    gen, seen = _mk()
    gen.chat("保持健康的三个提示。", max_new_tokens=8)
    assert seen[0] == f"{USER_TAG}保持健康的三个提示。\n{BOT_TAG}"


def test_chat_returns_tuple_and_appends_history() -> None:
    """⚠️ `chat()` 返回 (reply, history) 元组，不是字符串 —— 按字符串用会在序列化时炸。"""
    gen, _ = _mk()
    out = gen.chat("第一句", max_new_tokens=8)
    assert isinstance(out, tuple) and len(out) == 2
    reply, hist = out
    assert isinstance(reply, str) and isinstance(hist, list)
    assert hist[-2] == ("user", "第一句")
    assert hist[-1][0] == "assistant"


def test_multi_turn_history_is_formatted() -> None:
    gen, seen = _mk()
    hist = [("user", "A"), ("assistant", "B")]
    gen.chat("C", history=hist, max_new_tokens=8)
    prompt = seen[0]
    assert f"{USER_TAG}A" in prompt
    assert f"{BOT_TAG}B" in prompt
    assert prompt.index(f"{USER_TAG}A") < prompt.index(f"{BOT_TAG}B") < prompt.index("C")


def test_stop_sequence_is_the_next_turn_marker() -> None:
    """模型写完一轮后应停在下一轮的标记处，而不是继续编对话。

    ⚠️ 测**行为**，不测源码文本：`stop_sequences=(f"\\n{USER_TAG}", …)` 里
    是 f-string 表达式，拿 `inspect.getsource` 去匹配解析后的字面量必然失败
    （我第一版就这么写的，测试自己错了）。
    """
    gen = Generator.__new__(Generator)
    got: dict = {}

    def fake_complete(prompt, *, max_new_tokens=128, params=None,
                      stop_sequences=(), on_token=None, use_cache=True):
        got["stops"] = tuple(stop_sequences)
        return "回答"

    gen.complete = fake_complete        # type: ignore[method-assign]
    gen.chat("你好", max_new_tokens=8)
    stops = got["stops"]
    assert f"\n{USER_TAG}" in stops, f"停止序列缺下一轮用户标记：{stops}"
    assert f"\n{BOT_TAG}" in stops, f"停止序列缺助手标记：{stops}"


def test_corpus_builder_reuses_the_same_tags() -> None:
    """语料脚本必须**导入**标记，不能在本地各写一份（会漂移）。"""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "tools" / "make_chat_corpus.py"
           ).read_text(encoding="utf-8")
    assert "from codemind.generate import" in src, "语料脚本应导入标记而非重复定义"
    assert 'USER_TAG = "[' not in src, "语料脚本不应再本地定义 USER_TAG"
