"""配置系统：模型、训练、数据三类超参，全部可序列化为 JSON 并随 checkpoint 保存。

设计原则：
1. 所有默认值必须能在**只有 CPU** 的机器上跑起来（这是本项目的基准环境）。
2. 配置对象一律 dataclass，便于 ``json.dumps(asdict(cfg))`` 落盘与复现。
3. 预设（preset）只改规模，不改逻辑：``tiny`` 用于冒烟验证，``small`` 用于真实训练。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------------------
# 预设模型规模
# --------------------------------------------------------------------------------------

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    # 冒烟测试用：毫秒级前向，可在 CPU 上几分钟内完成可见的学习。
    "nano": dict(
        vocab_size=4096, n_layer=2, n_head=2, n_embd=128, n_kv_head=2, block_size=128
    ),
    # 默认：单机 CPU 可训练，能学会缩进、括号配对、常见函数骨架。
    "tiny": dict(
        vocab_size=4096, n_layer=6, n_head=6, n_embd=384, n_kv_head=2, block_size=512
    ),
    # 需要 GPU（>=8GB）才有实际意义。
    "small": dict(
        vocab_size=16384, n_layer=12, n_head=12, n_embd=768, n_kv_head=4, block_size=1024
    ),
    "medium": dict(
        vocab_size=32768, n_layer=24, n_head=16, n_embd=1024, n_kv_head=8, block_size=2048
    ),
}


@dataclass
class ModelConfig:
    """Transformer 结构超参。

    ``n_kv_head`` 小于 ``n_head`` 时启用 GQA（分组查询注意力），可显著缩小 KV-Cache。
    """

    vocab_size: int = 4096
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    n_kv_head: int = 2
    block_size: int = 512
    dropout: float = 0.0
    # SwiGLU 隐层倍率。8/3 是 LLaMA 的选择，经 64 对齐后即为经典 4x 的等价算力。
    mlp_ratio: float = 8.0 / 3.0
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    # 权重共享：嵌入矩阵与输出投影共用参数，小模型下省掉约 1/3 参数量。
    tie_embeddings: bool = True
    # 若不共享，可单独指定输出层偏置（一般不需要）。
    bias: bool = False

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd({self.n_embd}) 必须能被 n_head({self.n_head}) 整除"
            )
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head({self.n_head}) 必须能被 n_kv_head({self.n_kv_head}) 整除"
            )
        if self.vocab_size <= 0 or self.block_size <= 0 or self.n_layer <= 0:
            raise ValueError("vocab_size / block_size / n_layer 必须为正数")

    # -- 派生属性 -----------------------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def hidden_dim(self) -> int:
        """SwiGLU 隐层维度，向上对齐到 64 的倍数以贴合硬件。"""
        raw = int(self.mlp_ratio * self.n_embd)
        return 64 * math.ceil(raw / 64)

    @property
    def n_kv_groups(self) -> int:
        return self.n_head // self.n_kv_head

    def estimated_params(self) -> int:
        """粗略估计参数量（含/不含权重共享两种情况）。"""
        emb = self.vocab_size * self.n_embd
        per_block = (
            self.n_embd * self.n_head * self.head_dim  # q
            + self.n_embd * self.n_kv_head * self.head_dim * 2  # k, v
            + self.n_head * self.head_dim * self.n_embd  # o
            + 3 * self.n_embd * self.hidden_dim  # SwiGLU: gate/up/down
            + 2 * self.n_embd  # 两个 RMSNorm 的缩放
        )
        total = emb + self.n_layer * per_block + self.n_embd  # 末层 norm
        if not self.tie_embeddings:
            total += emb
        return total

    @classmethod
    def from_preset(cls, preset: str, **overrides: Any) -> "ModelConfig":
        if preset not in MODEL_PRESETS:
            raise KeyError(
                f"未知预设 {preset!r}，可选：{', '.join(sorted(MODEL_PRESETS))}"
            )
        merged = dict(MODEL_PRESETS[preset])
        merged.update(overrides)
        return cls(**merged)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelConfig":
        return cls(**_known_fields(cls, payload))


@dataclass
class TrainConfig:
    """训练超参。默认值针对「CPU + 小语料」调优，不是论文配方。"""

    out_dir: str = "runs/tiny"
    batch_size: int = 16
    grad_accum: int = 1
    max_iters: int = 2000
    # 0 表示由 max_iters 推导（取 3% 作为 warmup）。
    #
    # ⚠️ 曾是 10%，**实测证明太长**：云端 78.66M 模型跑 20,000 步时，
    # warmup 达 2,000 步 —— 模型在 LR 还在一路爬升的区间里就被推到了 3e-3，
    # 于是损失在 step 1001（lr=1.50e-03）触底 2.8683 后**一路反弹到 3.67**，
    # 再也没回到过最低点。**warmup 太长 = 让模型在高 LR 区待太久。**
    warmup_iters: int = 0
    #: 峰值学习率。
    #
    # ⚠️ 曾是 3e-3，**那个值只对 11M 验证过**。云端 78.66M 用它直接发散：
    # 损失在 lr=1.50e-03 时触底，lr 升到 3.00e-03 后单调变差。
    # **参数量大 7 倍，同样的峰值 LR 就过头了** —— 这是标准规律，不是偶然。
    #
    # 取 1e-3 的依据是那次实测的损失曲线（触底点在 1.5e-3），留三分之一余量。
    # ⚠️ 更大的预设（medium 312M）应当再降，建议 6e-4。
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    ckpt_interval: int = 500
    # 混合精度：CPU 上 bf16 反而可能更慢，默认关闭。
    amp: str = "off"  # off | bf16 | fp16
    compile: bool = False
    seed: int = 1337
    num_threads: int = 0  # 0 = 自动
    # FIM（fill-in-the-middle）预训练：代码补全能力的来源。
    fim_rate: float = 0.5
    fim_spm_rate: float = 0.5  # 其中 SPM（suffix-prefix-middle）模式占比
    # v2 FIM 格式：中间段之后补终止符并纳入损失，同时让 FIM 输出正好填满窗口
    # （v1 每次固定丢掉中间段尾部 3 个 token）。默认 False 保持历史行为。
    fim_terminator: bool = False
    resume: bool = True
    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainConfig":
        return cls(**_known_fields(cls, payload))

    def resolved_warmup(self) -> int:
        if self.warmup_iters > 0:
            return min(self.warmup_iters, max(1, self.max_iters))
        # ⚠️ 3%，不是 10%。实测：20,000 步时 10% = 2,000 步 warmup，
        # 模型在高 LR 爬升区待太久，损失触底后反弹（见 `warmup_iters` 的注释）。
        # 3% 让 LR 尽早到达平台，又不至于一上来就冲太高。
        return max(1, int(0.03 * self.max_iters))

    def lr_at(self, it: int) -> float:
        """余弦退火 + 线性 warmup。"""
        warmup = self.resolved_warmup()
        if it < warmup:
            return self.lr * (it + 1) / warmup
        if it >= self.max_iters:
            return self.lr * self.min_lr_ratio
        ratio = (it - warmup) / max(1, self.max_iters - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * coeff)

    def tokens_per_iter(self, block_size: int) -> int:
        return self.batch_size * self.grad_accum * block_size


@dataclass
class DataConfig:
    """代码语料采集与分片配置。"""

    source_dirs: List[str] = field(default_factory=list)
    out_dir: str = "data"
    name: str = "corpus"
    # 只收录这些后缀；空列表表示使用内置的代码后缀白名单。
    extensions: List[str] = field(default_factory=list)
    exclude_dirs: List[str] = field(
        default_factory=lambda: [
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            "node_modules",
            "dist",
            "build",
            ".idea",
            ".vscode",
            "site-packages",
            ".tox",
            ".ruff_cache",
            "target",
        ]
    )
    min_file_bytes: int = 32
    max_file_bytes: int = 512 * 1024
    # 语料清洗：丢掉含这些标记的文件（多半是自动生成或二进制垃圾）。
    skip_markers: List[str] = field(
        default_factory=lambda: ["\x00", "@generated", "DO NOT EDIT", "auto-generated"]
    )
    dedup: bool = True
    max_files: int = 0  # 0 = 不限
    val_ratio: float = 0.01
    seed: int = 1337

    def resolved_extensions(self) -> List[str]:
        if self.extensions:
            return [e.lower() if e.startswith(".") else "." + e.lower() for e in self.extensions]
        return list(DEFAULT_CODE_EXTENSIONS)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DataConfig":
        return cls(**_known_fields(cls, payload))


DEFAULT_CODE_EXTENSIONS: Tuple[str, ...] = (
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".java", ".kt", ".kts", ".scala", ".go", ".rs", ".rb", ".php",
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx",
    ".cs", ".swift", ".m", ".mm", ".lua", ".pl", ".r", ".jl", ".dart",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".proto", ".tf", ".vue", ".svelte", ".astro",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    ".xml", ".md", ".rst", ".tex", ".txt",
    ".dockerfile", ".cmake", ".gradle", ".properties",
)


# --------------------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------------------


def _known_fields(cls: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """过滤掉未知键，使旧 checkpoint 能兼容新版本配置（多余字段直接忽略）。"""
    allowed = {f.name for f in fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        # 只警告不报错：保证向前兼容。
        print(f"[config] 忽略 {cls.__name__} 中的未知字段: {sorted(unknown)}")
    return {k: v for k, v in payload.items() if k in allowed}


def resolve_device(spec: str = "auto"):
    """按 auto / cpu / cuda / cuda:0 / mps 解析出 torch.device。"""
    import torch

    if spec and spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(spec: str, device: Any):
    """把 amp 配置解析为 (autocast_dtype, grad_scaler_enabled)。"""
    import torch

    spec = (spec or "off").lower()
    is_cuda = getattr(device, "type", str(device)) == "cuda"
    if spec in ("bf16", "bfloat16"):
        return torch.bfloat16, False
    if spec in ("fp16", "float16", "half"):
        # fp16 只有在 CUDA 上才配 GradScaler；CPU 上 fp16 极慢，直接拒绝。
        if not is_cuda:
            raise ValueError("fp16 混合精度仅在 CUDA 设备上有意义，CPU 请用 bf16 或 off")
        return torch.float16, True
    if spec == "off":
        return None, False
    raise ValueError(f"未知 amp 模式: {spec!r}（可选 off/bf16/fp16）")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_config(cfg: ModelConfig) -> str:
    params = cfg.estimated_params()
    kv_bytes = (
        2
        * cfg.n_layer
        * cfg.n_kv_head
        * cfg.head_dim
        * cfg.block_size
        * 4  # float32
    )
    return (
        f"vocab={cfg.vocab_size} layers={cfg.n_layer} heads={cfg.n_head}"
        f"(kv={cfg.n_kv_head}) d_model={cfg.n_embd} ctx={cfg.block_size} "
        f"params≈{params/1e6:.2f}M kv_cache≈{kv_bytes/1024/1024:.2f}MB"
    )


def iter_presets() -> Iterable[Tuple[str, ModelConfig]]:
    for name in MODEL_PRESETS:
        yield name, ModelConfig.from_preset(name)


def describe_presets() -> str:
    lines = ["可用模型预设："]
    for name, cfg in iter_presets():
        lines.append(f"  {name:<8} {summarize_config(cfg)}")
    return "\n".join(lines)


__all__ = [
    "ModelConfig",
    "TrainConfig",
    "DataConfig",
    "MODEL_PRESETS",
    "DEFAULT_CODE_EXTENSIONS",
    "resolve_device",
    "pick_dtype",
    "dump_json",
    "load_json",
    "summarize_config",
    "describe_presets",
]
