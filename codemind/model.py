"""从零实现的 decoder-only Transformer，专为代码建模调优。

与通用 LLM 的差异点（都是为「代码补全」服务的）：

* **RoPE 旋转位置编码**：外推到更长上下文时表现稳定，且无位置嵌入参数。
* **RMSNorm + Pre-Norm 残差**：训练稳定，无需 LayerNorm 的均值中心化开销。
* **SwiGLU 前馈**：同等算力下优于 GELU MLP（LLaMA 配方）。
* **GQA 分组查询注意力**：KV-Cache 缩小 ``n_head / n_kv_head`` 倍，
  在 CPU 上直接决定长文件补全能不能跑得动。
* **权重共享**：嵌入与输出投影共用参数，小模型省下约 1/4 参数。
* **KV-Cache 增量解码**：逐 token 生成时只算 1 个位置，复杂度从 O(T²) 降到 O(T)。

前向/反向数值路径完全由本项目实现，只使用 PyTorch 的基础算子。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

__all__ = ["GPT", "SamplingParams", "sample_next_token", "causal_mask"]


# ======================================================================================
# 基础组件
# ======================================================================================


class RMSNorm(nn.Module):
    """均方根归一化：``x / sqrt(mean(x²) + eps) * weight``。"""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # 在 float32 下做归一化，避免 bf16 下平方溢出/精度塌陷
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32.to(dtype)) * self.weight


def build_rope_cache(
    head_dim: int, max_pos: int, theta: float, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos/sin 表。

    返回形状 ``(max_pos, head_dim)``：为了让 ``q * cos`` 直接逐元素相乘，
    把每一对维度展开成重复两次的形式（即 rotate_half 的等价写法）。
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # (max_pos, head_dim/2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # 交错展开: [c0, c0, c1, c1, ...]，与 rotate_half 的配对方式一致
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    return cos.to(dtype), sin.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把最后一维前后两半交换并给前半取负（标准 RoPE 旋转）。"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x``: (B, H, T, Dh)；``cos/sin``: (T, Dh)。"""
    return x * cos + rotate_half(x) * sin


def causal_mask(t_q: int, t_k: int, device: torch.device) -> torch.Tensor:
    """标准因果掩码，``True`` 表示「可见」，形状 ``(1, 1, t_q, t_k)``，布尔类型。

    查询行对齐到 key 序列的**末尾**：第 ``i`` 个查询对应绝对位置 ``t_k - t_q + i``，
    因此增量解码（``t_q=1``）时它只能看见最后一个 key。

    广播方向必须是 ``q_idx (t_q, 1)`` 对 ``k_idx (1, t_k)`` ——
    反过来的话会得到 ``(t_k, t_k)``，单查询场景下掩码会退化成"全可见"，
    直接造成因果性泄漏（未来 token 影响过去）。

    刻意不接收 dtype：返回布尔掩码，配合 ``masked_fill(~mask, -inf)`` 使用，
    省掉一次类型转换，也避免调用方误以为它是加性浮点掩码。
    """
    q_idx = torch.arange(t_k - t_q, t_k, device=device).unsqueeze(1)  # (t_q, 1)
    k_idx = torch.arange(t_k, device=device).unsqueeze(0)  # (1, t_k)
    mask = k_idx <= q_idx  # (t_q, t_k)
    return mask.unsqueeze(0).unsqueeze(0)


class CausalSelfAttention(nn.Module):
    """多头因果自注意力，支持 GQA 与增量 KV-Cache。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.attn_dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        return_attn: bool = False,
    ):
        B, T, _ = x.shape
        cfg = self.cfg

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        new_cache = None
        if use_cache:
            if cache is not None:
                k_prev, v_prev = cache
                k = torch.cat((k_prev, k), dim=2)
                v = torch.cat((v_prev, v), dim=2)
            new_cache = (k, v)

        t_k = k.shape[2]
        # 扩展 KV 头以匹配 Q 头（GQA）：先在 head 维复制，避免 materialize 整个 KV
        if self.n_rep > 1:
            k_full = k.repeat_interleave(self.n_rep, dim=1)
            v_full = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_full, v_full = k, v

        # SDPA 的融合实现不会返回注意力权重，因此仅在「无缓存且不需要权重」时使用它。
        # 其余情况走手写路径：长度可不等（增量解码），权重可导出（可解释性/调试）。
        if cache is None and not return_attn:
            ctx = F.scaled_dot_product_attention(
                q, k_full, v_full,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
            )
            attn_weights = None
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (q @ k_full.transpose(-2, -1)) * scale
            mask = causal_mask(T, t_k, x.device)
            scores = scores.masked_fill(~mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            if self.training and self.attn_dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.attn_dropout)
            ctx = attn_weights @ v_full

        out = ctx.transpose(1, 2).contiguous().view(B, T, cfg.n_embd)
        out = self.resid_dropout(self.o_proj(out))
        # 恒定返回三元组：缓存关闭时 new_cache 为 None，注意力未被显式计算时为 None。
        # 调用方因此永远可以安全地做三元解包。
        return out, attn_weights, new_cache


class SwiGLU(nn.Module):
    """SwiGLU 前馈：``down(silu(gate(x)) * up(x))``。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.hidden_dim
        self.gate_proj = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.up_proj = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.down_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    """一个 Transformer 层：Pre-Norm 注意力 + Pre-Norm 前馈。

    ``forward`` 恒定返回三元组 ``(x, attn_weights, new_cache)`` ——
    两个辅助值在不需要时为 ``None``。**不要**根据标志位切换返回结构：
    早期版本在 ``return_attn=False`` 时返回 ``(out, (None, cache))``，
    调用方取 ``aux[1]`` 会拿到张量而不是 ``(k, v)`` 元组，
    到 KV-Cache 拼接时才以 "not enough values to unpack" 的形式爆炸。
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, use_cache=False, return_attn=False):
        attn_out, attn_weights, new_cache = self.attn(
            self.norm1(x), cos, sin, cache, use_cache, return_attn
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights, new_cache


# ======================================================================================
# 完整模型
# ======================================================================================


def _block_core(
    block: "Block",
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """只返回 hidden state 的 block 调用，供 ``torch.utils.checkpoint`` 使用。

    ``Block.forward`` 返回 ``(x, attn_weights, cache)`` 三元组，而梯度检查点要求
    函数返回值参与反向图。这里包一层只取第一个元素 —— 检查点模式下不需要
    注意力图和 KV cache（训练路径两个都是 None/False）。
    """
    out, _attn, _cache = block(x, cos, sin, None, False, False)
    return out


class GPT(nn.Module):
    """从零实现的代码语言模型。

    ``forward`` 返回 ``(logits, loss, aux)``；``loss`` 仅在提供 ``targets`` 时非 None。
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # 残差输出投影按层数缩放初始化（GPT-2 配方），避免深层残差方差爆炸
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        self._rope_cache: Dict[Tuple[int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------ 初始化
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ RoPE 缓存
    def _rope(self, t_total: int, device: torch.device, dtype: torch.dtype):
        key = (t_total, device, dtype)
        cached = self._rope_cache.get(key)
        if cached is None:
            cached = build_rope_cache(
                self.cfg.head_dim, t_total, self.cfg.rope_theta, device, dtype
            )
            self._rope_cache[key] = cached
        return cached

    def crop_rope_cache(self, max_pos: int) -> None:
        """裁剪 RoPE 缓存（例如换用短上下文时释放显存/内存）。"""
        for key in list(self._rope_cache):
            if key[0] > max_pos:
                self._rope_cache.pop(key, None)

    # ------------------------------------------------------------------ 前向
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        loss_mask: Optional[torch.Tensor] = None,
        caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_attn: bool = False,
        logits_for_all: bool = True,
        ignore_index: int = -100,
        grad_checkpoint: bool = False,
    ):
        """前向计算。

        契约（调用方务必遵守）：
        * ``targets`` 已做过 shift，即 ``targets[b, t]`` 是 ``idx[b, t]`` 的**下一个** token。
        * ``loss_mask`` 与 ``targets`` 同形状，非零表示该位置计入损失。
          FIM 预训练正是靠它把「前缀+后缀」区间的损失屏蔽掉。
        * ``caches`` 非空时必须配合 ``use_cache=True``，且 ``idx`` 只包含新增 token。

        ``logits_for_all=False``（训练时用）：只返回 ``None`` 作为 logits。
        这是 CPU 训练最重要的省内存开关 —— 全量 logits 的形状是
        ``(batch, seq, vocab)``，在 ``small`` 预设 + 1024 上下文 + batch 8 下，
        单是这一块 fp32 就占 8×1024×16384×4 ≈ 512 MB，
        连同反向所需的中间量会轻松吃掉数 GB。训练只依赖 loss，
        因此这一整块可以完全不算。
        """
        B, T = idx.shape
        if T == 0:
            raise ValueError("输入序列长度为 0")
        past = 0 if caches is None else caches[0][0].shape[2]
        if past + T > self.cfg.block_size:
            raise ValueError(
                f"位置 {past + T} 超过 block_size={self.cfg.block_size}；"
                "生成时应截断上下文，训练时应检查数据切窗长度"
            )

        cos, sin = self._rope(past + T, idx.device, self.tok_emb.weight.dtype)
        cos = cos[past : past + T]
        sin = sin[past : past + T]

        x = self.drop(self.tok_emb(idx))
        new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        attn_maps: List[torch.Tensor] = []
        # 梯度检查点：用计算换显存。只在**训练 + 无 cache + 不取注意力图**时启用
        # （生成/评估路径绝不能用，否则既慢又拿不到 cache）。
        # 实测意义：1B + ctx 1024 + batch 16 的激活约 50 GB，开检查点后降到约 12 GB，
        # 这是「32 GB 双 T4 上能否装下 1B 训练」的关键。
        use_ckpt = (
            grad_checkpoint
            and self.training
            and not use_cache
            and not return_attn
            and past == 0
        )
        for i, block in enumerate(self.blocks):
            layer_cache = None if caches is None else caches[i]
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    _block_core, block, x, cos, sin, use_reentrant=False
                )
                attn_weights, layer_new_cache = None, None
            else:
                x, attn_weights, layer_new_cache = block(
                    x, cos, sin, layer_cache, use_cache, return_attn
                )
            if use_cache:
                new_caches.append(layer_new_cache)
            if return_attn and attn_weights is not None:
                attn_maps.append(attn_weights)
        x = self.norm_f(x)

        if targets is None:
            logits = self.lm_head(x)
            return logits, None, self._pack_aux(use_cache, new_caches, return_attn, attn_maps)

        if loss_mask is None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
            return logits, loss, self._pack_aux(use_cache, new_caches, return_attn, attn_maps)

        # 掩码路径：只对需要计损的位置做 lm_head —— 这是训练时最省内存的一条路。
        # FIM 训练里通常只有约 1/4 的位置计损，因此显存/内存占用直接降到 1/4 量级。
        flat_mask = loss_mask.reshape(-1) > 0
        keep = flat_mask.nonzero(as_tuple=True)[0]
        if keep.numel() == 0:
            # 极端情况（整批都被屏蔽）：返回零损失但保持计算图连通
            return None, x.sum() * 0.0, self._pack_aux(use_cache, new_caches, return_attn, attn_maps)
        x_flat = x.reshape(-1, x.shape[-1])
        tgt_flat = targets.reshape(-1)
        logits_sel = self.lm_head(x_flat.index_select(0, keep))
        tgt_sel = tgt_flat.index_select(0, keep)
        loss = F.cross_entropy(logits_sel.float(), tgt_sel, ignore_index=ignore_index)
        # 训练不需要 logits 时（默认）直接返回 None，彻底避免再算一遍全量投影
        if logits_for_all:
            return self.lm_head(x), loss, self._pack_aux(use_cache, new_caches, return_attn, attn_maps)
        return None, loss, self._pack_aux(use_cache, new_caches, return_attn, attn_maps)

    @staticmethod
    def _pack_aux(use_cache: bool, caches, return_attn: bool, attn_maps) -> Dict[str, object]:
        aux: Dict[str, object] = {}
        if use_cache:
            aux["caches"] = caches
        if return_attn:
            aux["attn"] = attn_maps
        return aux

    # ------------------------------------------------------------------ 生成
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        params: "SamplingParams",
        *,
        stop_ids: Optional[Tuple[int, ...]] = None,
        tokenizer=None,
        stream_callback=None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """自回归生成，默认走增量 KV-Cache（每步只前向 1 个位置）。

        ``stream_callback(token_id, text_piece)`` 每产生一个 token 就被调用一次，
        CLI 因此可以逐字输出而不是等全部生成完。
        """
        self.eval()
        if max_new_tokens <= 0:
            return idx
        stop_ids = tuple(stop_ids or ())
        generated = idx
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=idx.device).manual_seed(int(params.seed))

        caches = None
        for _ in range(max_new_tokens):
            # 上下文裁剪：只保留最后 block_size 个 token，缓存随之重建
            if generated.shape[1] > self.cfg.block_size:
                generated = generated[:, -self.cfg.block_size :]
                caches = None

            if use_cache:
                if caches is None:
                    logits, _, aux = self.forward(generated, use_cache=True)
                else:
                    logits, _, aux = self.forward(
                        generated[:, -1:], caches=caches, use_cache=True
                    )
                caches = aux.get("caches")
                next_logits = logits[:, -1, :]
            else:
                logits, _, _ = self.forward(generated, use_cache=False)
                next_logits = logits[:, -1, :]

            next_id = sample_next_token(
                next_logits, params, generated, generator=generator
            )
            generated = torch.cat((generated, next_id), dim=1)
            tid = int(next_id[0, 0].item())

            if stream_callback is not None:
                piece = "" if tokenizer is None else tokenizer.decode([tid])
                stream_callback(tid, piece)
            if tid in stop_ids:
                break
        return generated

    # ------------------------------------------------------------------ 工具
    def num_params(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                total -= self.lm_head.weight.numel()
        return total

    def configure_optimizer(
        self,
        weight_decay: float,
        lr: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """把参数分成「需要 weight decay」和「不需要」两组：所有 1D 参数（norm/bias）不衰减。"""
        decay, no_decay = [], []
        seen = set()
        for _, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        try:
            return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
        except (TypeError, RuntimeError):
            return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @classmethod
    def from_pretrained(cls, path, map_location="cpu") -> Tuple["GPT", Dict[str, object]]:
        """从本项目的 checkpoint 载入模型（只认自家格式，绝不加载外部权重）。"""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        cfg_payload = payload.get("model_config") or payload.get("config")
        if cfg_payload is None:
            raise ValueError(f"{path} 缺少 model_config 字段，不是有效的 CodeMind checkpoint")
        cfg = ModelConfig.from_dict(cfg_payload)
        model = cls(cfg)
        state = payload.get("model") or payload.get("state_dict")
        if state is None:
            raise ValueError(f"{path} 缺少 model 权重字段")
        model.load_state_dict(state)
        model.eval()
        return model, payload

    def save(self, path, payload: Optional[Dict[str, object]] = None) -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: Dict[str, object] = {"model_config": self.cfg.to_dict(), "model": self.state_dict()}
        if payload:
            data.update(payload)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(data, tmp)
        tmp.replace(path)


# ======================================================================================
# 采样
# ======================================================================================


@dataclass
class SamplingParams:
    """采样超参。默认偏保守：代码补全最怕「胡说八道」。"""

    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    greedy: bool = False
    seed: Optional[int] = None
    # 允许出现的 token id 白名单之外的（结构性特殊词元）在这里屏蔽；
    # 例如续写任务里禁掉 <|fim_*|>，避免模型在普通代码里插入填空标记。
    banned_ids: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature 不能为负")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p 需落在 (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty 必须为正")

    def with_(self, **overrides) -> "SamplingParams":
        from dataclasses import replace

        return replace(self, **overrides)


def sample_next_token(
    logits: torch.Tensor,
    params: SamplingParams,
    prev_tokens: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    banned_ids: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """从 ``(B, V)`` logits 中采样下一个 token，返回 ``(B, 1)``。

    采样顺序：温度 -> 重复惩罚 -> 屏蔽 -> top-k -> top-p -> 多项分布。
    ``temperature <= 0`` 或 ``params.greedy`` 时退化为贪心。
    """
    logits = logits.float()
    banned = banned_ids if banned_ids is not None else params.banned_ids
    if banned:
        logits = logits.clone()
        logits[:, list(banned)] = float("-inf")

    if params.repetition_penalty != 1.0 and prev_tokens is not None and prev_tokens.numel() > 0:
        logits = _apply_repetition_penalty(logits, prev_tokens, params.repetition_penalty)

    if params.greedy or params.temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / params.temperature

    if params.top_k and params.top_k > 0 and params.top_k < logits.size(-1):
        kth = torch.topk(logits, params.top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # 保留累计概率首次超过 top_p 的那个 token
        remove = cumulative - probs > params.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or probs.sum(dim=-1).min() <= 0:
        # 全部被屏蔽时兜底为原 logits 的 argmax
        return logits.argmax(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def _apply_repetition_penalty(
    logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float
) -> torch.Tensor:
    """对已出现过的 token 施加惩罚（CTRL 论文的做法，逐 batch 处理）。"""
    vocab = logits.size(-1)
    for b in range(logits.size(0)):
        ids = torch.unique(prev_tokens[b])
        ids = ids[(ids >= 0) & (ids < vocab)]
        if ids.numel() == 0:
            continue
        row = logits[b, ids]
        logits[b, ids] = torch.where(row > 0, row / penalty, row * penalty)
    return logits
