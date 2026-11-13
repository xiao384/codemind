"""生成与推理：把模型 + 分词器封装成可直接给 CLI 使用的接口。

三个核心能力，对应 IDE 里真实发生的三种场景：

* ``complete``  **续写**：给定光标前的内容，继续往下写。
* ``infill``    **填空**：给定光标前的 ``prefix`` 与光标后的 ``suffix``，
  用 FIM 特殊词元构造提示，让模型把中间补上。这是代码补全最有价值的形态 ——
  用户往往在文件中间写代码，模型必须同时「看见」下文。
* ``chat``      **对话**：REPL 交互，带系统提示与多轮上下文裁剪。

所有生成都走增量 KV-Cache，并通过 ``on_token`` 回调实现逐字流式输出。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from .config import ModelConfig
from .model import GPT, SamplingParams
from .tokenizer import EOT, FIM_BEGIN, FIM_END, FIM_HOLE, PAD, BPETokenizer

__all__ = ["Generator", "load_runner", "CODE_SYSTEM_PROMPT",
           "INFILL_DEFAULT_REPETITION_PENALTY", "USER_TAG", "BOT_TAG"]

#: ``Generator.infill`` 的默认重复惩罚。**1.2 是实测调出来的，不是拍脑袋**：
#: 相对 1.05，300 个留出文档上 ``rep8`` 60.68% → 46.30%（z=−7.36）、


#: 对话标记 —— **必须与 	ools/make_chat_corpus.py 逐字一致**。
#: 单一来源：改这里就等于同时改了训练语料与推理提示词。
#: 绝不能用 # 开头（Python 注释），那正是旧实现聊天为零的根因。
USER_TAG = "[用户] "
BOT_TAG = "[助手] "
#: ``rep_line`` 21.76% → 15.01%（z=−3.23），而 ``edit_sim``/``syntax_ok`` 无损失。
#: 复现：``tools/cb1.py --arm-spec "strongrep:repetition_penalty=1.2"``。
#: ⚠️ 该参数是在同一批验证文档上调出的，效应量远超分辨力但严格说仍需独立样本复核。
INFILL_DEFAULT_REPETITION_PENALTY = 1.2


CODE_SYSTEM_PROMPT = (
    "You are CodeMind, a precise code completion engine trained from scratch. "
    "Continue or fill in the code exactly where the cursor is. "
    "Output code only. Never explain. Never use markdown fences. "
    "Match the surrounding indentation and naming style."
)


#: 对话的默认采样参数 —— **greedy（温度 0）**。
#:
#: ?? **不要改成温度 >0。** 实测：同一个模型、同一个问题，
#: 温度 0.7 给「定期课程: 鼓励学生快速准确，你需要记得购买的方式以便进行使用」，
#: 温度 0 给「作为一个人工智能助手，我无法为您提供关于这个信息的信息」。
#: 小模型的分布很平，温度 >0 等于在平分布上随机抽样。
#: 详见 `Generator.chat` 的 docstring。
CHAT_DEFAULT_PARAMS = SamplingParams(temperature=0.0, top_k=1, repetition_penalty=1.1)

class Generator:
    """模型 + 分词器的采样门面。"""

    def __init__(
        self,
        model: GPT,
        tokenizer: BPETokenizer,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ 内部
    @property
    def block_size(self) -> int:
        return self.model.cfg.block_size

    def _encode(self, text: str, *, allow_special: bool = True) -> List[int]:
        return self.tokenizer.encode(text, allow_special=allow_special)

    def _prepare_input(self, ids: List[int]) -> torch.Tensor:
        """把提示截断到上下文窗口内（保留最靠近光标的部分）。"""
        if not ids:
            ids = [self.tokenizer.eot_id]
        keep = self.block_size - 8  # 给生成留一点余量
        if len(ids) > keep:
            ids = ids[-keep:]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _generate_ids(
        self,
        prompt_ids: List[int],
        params: SamplingParams,
        max_new_tokens: int,
        stop_ids: Sequence[int],
        on_token: Optional[Callable[[int, str], None]],
        use_cache: bool,
    ) -> List[int]:
        idx = self._prepare_input(prompt_ids)
        out = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            params=params,
            stop_ids=tuple(stop_ids),
            tokenizer=self.tokenizer,
            stream_callback=on_token,
            use_cache=use_cache,
        )
        return out[0].tolist()[idx.shape[1] :]

    @staticmethod
    def _cut_stop_sequences(text: str, stops: Sequence[str]) -> str:
        cut = len(text)
        for s in stops:
            if not s:
                continue
            pos = text.find(s)
            if pos >= 0:
                cut = min(cut, pos)
        return text[:cut]

    # ------------------------------------------------------------------ 续写
    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        params: Optional[SamplingParams] = None,
        stop_sequences: Sequence[str] = (),
        on_token: Optional[Callable[[int, str], None]] = None,
        use_cache: bool = True,
    ) -> str:
        """续写：返回**新增**的文本（不含提示本身）。"""
        params = params or SamplingParams()
        # 续写任务里禁止模型吐出结构词元，否则普通代码里会混进 <|fim_*|> 之类的标记
        params = params.with_(
            banned_ids=(
                self.tokenizer.pad_id,
                self.tokenizer.special_id(FIM_BEGIN),
                self.tokenizer.special_id(FIM_HOLE),
                self.tokenizer.special_id(FIM_END),
            )
        )
        prompt_ids = self._encode(prompt)
        new_ids = self._generate_ids(
            prompt_ids,
            params,
            max_new_tokens,
            stop_ids=(self.tokenizer.eot_id,),
            on_token=on_token,
            use_cache=use_cache,
        )
        text = self.tokenizer.decode(new_ids)
        return self._cut_stop_sequences(text, stop_sequences)

    # ------------------------------------------------------------------ 填空
    def build_fim_prompt(self, prefix: str, suffix: str) -> str:
        """构造 FIM 提示。默认使用 PSM 顺序（prefix 在前），与训练分布一致。"""
        return f"{FIM_BEGIN}{prefix}{FIM_HOLE}{suffix}{FIM_END}"

    def infill(
        self,
        prefix: str,
        suffix: str,
        *,
        max_new_tokens: int = 128,
        params: Optional[SamplingParams] = None,
        stop_sequences: Sequence[str] = (),
        on_token: Optional[Callable[[int, str], None]] = None,
        use_cache: bool = True,
    ) -> str:
        """填空：返回补在 ``prefix`` 与 ``suffix`` 之间的代码。

        ``repetition_penalty`` 默认 **1.2**，不是 1.05 —— 这是实测调出来的，不是拍脑袋：

        | 指标 | rep 1.05 | rep 1.2 | Δ | z |
        |---|---|---|---|---|
        | `rep8` 重复度 | 60.68% | **46.30%** | −14.38pp | −7.36 |
        | `rep_line` 行重复 | 21.76% | **15.01%** | −6.75pp | −3.23 |
        | `edit_sim` | 24.32% | 24.11% | −0.21pp | −0.21 |
        | `syntax_ok` | 11.67% | 14.67% | +3.00pp | +1.09 |

        （step-2000 检查点、300 个留出文档，`tools/cb1.py --arm-spec`。）
        **重复度砍掉 24~31%，而相似度与语法没有付出代价。**
        ⚠️ 这组参数是在同一批 300 个验证文档上调出的；效应量远超该样本量的分辨力
        （约 2.6pp），且两个互补的重复指标同向显著，但严格来说仍需独立样本复核。
        """
        params = params or SamplingParams(
            temperature=0.2, top_k=20, repetition_penalty=INFILL_DEFAULT_REPETITION_PENALTY
        )
        prompt = self.build_fim_prompt(prefix, suffix)
        prompt_ids = self._encode(prompt)
        new_ids = self._generate_ids(
            prompt_ids,
            params,
            max_new_tokens,
            stop_ids=(self.tokenizer.fim_end_id, self.tokenizer.eot_id),
            on_token=on_token,
            use_cache=use_cache,
        )
        text = self.tokenizer.decode(new_ids, strip_special=True)
        text = self._cut_stop_sequences(text, stop_sequences)
        return _clean_infill(text, prefix)

    # ------------------------------------------------------------------ 对话
    def chat(
        self,
        user_message: str,
        *,
        history: Optional[List[Tuple[str, str]]] = None,
        max_new_tokens: int = 256,
        params: Optional[SamplingParams] = None,
        on_token: Optional[Callable[[int, str], None]] = None,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """对话：把多轮历史拼成 ``[用户]`` / ``[助手]`` 格式的文本提示。

        ⚠️ **标记绝不能用 `#` 开头。** 这是被实测打出来的：
        旧实现用 ``## User`` / ``## Assistant``，而 ``#`` 在 Python 里是注释 ——
        模型把整段对话读成注释块，然后"接着写代码"。实测问「你好，你是谁？」
        得到的是 ``# Remove the pre-process registry...`` 这样的 Python 注释。

        更根本的问题：这两个标记在 6800 万 token 的**代码语料里出现 0 次**，
        模型没有任何理由尊重它们。所以改用 ``[用户]`` / ``[助手]``，
        并**在对话语料上做了微调**（见 ``tools/make_chat_corpus.py``）。

        ⚠️ 标记必须与训练语料**逐字一致**（``USER_TAG`` / ``BOT_TAG``），
        否则训完也不会 work —— 训练格式与推理格式不一致是最容易犯、也最难查的错。
        """
        # ⚠️ 默认用 **greedy（温度 0）**，不是 0.4。
        #
        # 这是被实测打出来的。同一个 78M 融合模型、同一个问题
        # 「保持健康的三个提示。」，只换采样温度：
        #
        #   temperature=0.7  →  「定期课程: 鼓励学生快速准确，你需要记得购买的方式以便进行使用。」
        #   temperature=0.0  →  **「减少压力和焦虑：保持健康的三个提示可以帮助你更好地了解压力和焦虑」**
        #
        # 换几个问题再测（温度 0）：
        #   「你好，你是谁？」    → **「作为一个人工智能助手，我无法为您提供关于这个信息的信息。」**
        #   「介绍一下中国的长城」→ **「中国的长城是一个大型范围，它是一个关于地理世界的国家。」**
        #   「1+1等于几？」       → **「1+1等于几，可以考虑使用两个元素：」**
        #
        # **原因**：小模型的输出分布很平（每个 token 概率都差不多）。
        # 温度 >0 在平分布上采样 = **随机挑一个**；greedy 挑最高概率 = **挑最像话的**。
        #
        # 这与 AGENTS.md 记的代码侧杠杆是同一类问题（那里是 `rep 1.05→1.2`，
        # 砍掉 24% 重复度）。**只改先验的解码参数不需要额外算力，是本项目
        # 已验证有效、且唯一零成本的改进手段。**
        #
        # ⚠️ 抽成具名常量，避免后人看到"温度 0"以为是笔误改回去，让这个结论无声丢失。
        params = params or CHAT_DEFAULT_PARAMS
        history = list(history or [])
        parts: List[str] = []
        for role, content in history[-6:]:
            tag = USER_TAG if role == "user" else BOT_TAG
            parts.append(f"{tag}{content}\n")
        parts.append(f"{USER_TAG}{user_message}\n{BOT_TAG}")
        prompt = "".join(parts)
        reply = self.complete(
            prompt,
            max_new_tokens=max_new_tokens,
            params=params,
            # 一轮结束后模型应当开始写下一轮的用户输入 —— 在那里停
            stop_sequences=(f"\n{USER_TAG}", f"\n{BOT_TAG}"),
            on_token=on_token,
        )
        reply = reply.strip()
        history.append(("user", user_message))
        history.append(("assistant", reply))
        return reply, history

    # ------------------------------------------------------------------ 批量
    def complete_batch(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 128,
        params: Optional[SamplingParams] = None,
    ) -> List[str]:
        """批量续写（逐条处理；CPU 上批内 padding 收益有限，保持简单）。"""
        return [
            self.complete(p, max_new_tokens=max_new_tokens, params=params) for p in prompts
        ]

    def stream_complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        params: Optional[SamplingParams] = None,
        stop_sequences: Sequence[str] = (),
    ) -> Iterator[str]:
        """生成器形式：逐段吐出文本，便于 CLI 实时打印。"""
        buffer: List[str] = []

        def _collect(token_id: int, piece: str) -> None:
            buffer.append(piece)

        # 简化实现：先跑完再按块 yield（保留接口，便于以后接真流式）
        text = self.complete(
            prompt,
            max_new_tokens=max_new_tokens,
            params=params,
            stop_sequences=stop_sequences,
        )
        for line in _split_keepends(text):
            yield line

    # ------------------------------------------------------------------ 注意力可视化
    def attention_map(self, text: str) -> torch.Tensor:
        """返回 ``(n_layer, n_head, T, T)`` 的注意力权重，用于理解模型在看哪里。"""
        ids = self._encode(text)
        if len(ids) > self.block_size:
            ids = ids[-self.block_size :]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            _, _, aux = self.model(x, return_attn=True)
        maps = aux.get("attn") or []
        return torch.stack([m[0] for m in maps], dim=0)


# ======================================================================================
# 工具函数
# ======================================================================================


def _split_keepends(text: str) -> List[str]:
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    return lines or [text]


def _clean_infill(text: str, prefix: str) -> str:
    """清理填空结果：去掉与 prefix 重复的开头、以及悬空的结尾。"""
    if not text:
        return ""
    text = text.replace(FIM_END, "").replace(FIM_BEGIN, "").replace(FIM_HOLE, "")
    text = text.replace(EOT, "").replace(PAD, "")
    # 模型有时会把光标前的最后一个标识符重写一遍，这里做最长重叠裁剪
    max_overlap = min(len(prefix), len(text), 64)
    for size in range(max_overlap, 2, -1):
        if prefix.endswith(text[:size]):
            text = text[size:]
            break
    return text


def load_runner(
    run_dir: Path,
    *,
    checkpoint: str = "best",
    device: str = "auto",
    tokenizer_path: Optional[Path] = None,
) -> Generator:
    """从训练输出目录载入模型与分词器。

    ``checkpoint`` 可为 ``best`` / ``latest`` / 具体文件名。
    """
    from .config import resolve_device

    run_dir = Path(run_dir)
    if run_dir.is_file():
        ckpt_path = run_dir
        run_dir = run_dir.parent
    else:
        name = checkpoint if checkpoint.endswith(".pt") else f"{checkpoint}.pt"
        ckpt_path = run_dir / name
        if not ckpt_path.exists():
            # 退回到目录下最新的 .pt
            candidates = sorted(run_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError(f"{run_dir} 下找不到任何 .pt 检查点")
            ckpt_path = candidates[-1]

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "codemind-checkpoint-v1":
        raise ValueError(f"{ckpt_path} 不是 CodeMind 检查点")
    cfg = ModelConfig.from_dict(payload["model_config"])
    model = GPT(cfg)
    model.load_state_dict(payload["model"])

    tok_file = tokenizer_path or (
        Path(payload["tokenizer_path"]) if payload.get("tokenizer_path") else run_dir / "tokenizer.json"
    )
    if not Path(tok_file).exists():
        alt = run_dir / "tokenizer.json"
        if alt.exists():
            tok_file = alt
        else:
            raise FileNotFoundError(
                f"找不到分词器：{tok_file}（请确保训练时保存过 tokenizer.json）"
            )
    tokenizer = BPETokenizer.load(Path(tok_file))
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError(
            f"词表不匹配：分词器 {tokenizer.vocab_size} vs 模型 {cfg.vocab_size}。"
            "请使用与该模型配套训练的分词器。"
        )
    dev = resolve_device(device)
    model.eval()
    return Generator(model, tokenizer, device=dev)


def describe_checkpoint(path: Path) -> Dict[str, object]:
    """读取检查点元信息（不加载权重到模型，用于 ``codemind info``）。"""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(payload.get("model_config", {}))
    info: Dict[str, object] = {
        "path": str(path),
        "format": payload.get("format"),
        "iteration": payload.get("iteration"),
        "best_val": payload.get("best_val"),
        "val_loss": payload.get("val_loss"),
        "model_config": cfg.to_dict(),
        "params_estimate": cfg.estimated_params(),
        "tokenizer_path": payload.get("tokenizer_path"),
        "size_mb": Path(path).stat().st_size / 1024 / 1024,
    }
    return info
