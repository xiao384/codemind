# CodeMind

**从零实现的代码大模型 + 原生编程 CLI。**
分词器、Transformer、训练循环、采样器、补全引擎、自纠错编辑器 —— 全部原创实现，
只依赖 PyTorch 与 NumPy。不加载任何外部预训练权重，不调用任何模型 API。

```text
codemind/
├── bytecodec.py   字节 ↔ 可见字符双射（byte-level 编码的基础）
├── tokenizer.py   字节级 BPE：代码感知预切分、增量合并训练、FIM 特殊词元
├── model.py       Transformer：RoPE / RMSNorm / SwiGLU / GQA / KV-Cache / 权重共享
├── data.py        语料采集 -> uint16 二进制分片 -> memmap 随机切窗 -> FIM 变换
├── train.py       AdamW + warmup/余弦退火 + 梯度累积 + 断点续训 + 掩码损失
├── generate.py    采样器 + 增量 KV-Cache 流式解码 + 填空/续写/对话
├── editing.py     补丁校验、语法拦截、测试驱动自纠错、失败回滚
├── bench.py       实测吞吐与训练时长估算（换机器后先跑它）
└── cli.py         命令行入口
tools/
├── pack.ps1       打成干净的迁移包（排除 .venv / 产物 / 缓存）
└── run_server.ps1 服务器长跑：数据 -> 基准 -> 训练（内存崩溃自愈）-> 抽样
```

---

## 先说清楚能力边界（很重要）

这是一个**从零训练**的小模型系统。它的竞争力来自**工程闭环**，不是参数规模：

| 能做什么 | 不能做什么 |
|---|---|
| 在你的代码库上学会缩进、括号配对、常见函数骨架、命名风格 | 不会写复杂算法，不能替代 Copilot / Claude |
| 学习「给定上下文填空」——IDE 里最真实的补全场景 | 没有指令微调，`repl` 对话能力很弱 |
| 完整的训练→推理→编辑→校验→回滚闭环，全部可复现、可审计 | 不会推理、不懂业务逻辑 |
| 单机 CPU 就能跑完全流程，几十秒看到损失下降 | 达到生产可用的补全质量需要 GPU + 数十亿 token |

**一句话**：这是一台可以完全掌控、可拆开看每个零件的代码模型引擎；
在它之上接多少数据、多少算力，上限由你决定。

---

## 实测数据与已知上限

以下数字全部来自本机的真实运行（Python 3.12.13 / torch 2.14.1+cpu / 12 逻辑核 / **7.3 GB 物理内存**）。

**分词器**（220 KB 真实源码，19 个文件）：

| 词表 | 合并数 | 训练耗时 | 压缩率 | 编码吞吐 |
|---|---|---|---|---|
| 1024 | 763 | 1.07 s | 1.92 字符/token | ~5.0 M字符/s |
| 4096 | 3835 | 1.57 s | 2.50 字符/token | ~4.9 M字符/s |
| 8192 | 7931 | 1.94 s | 2.65 字符/token | ~3.3 M字符/s |

增量训练（2048 → 4096）只需 **0.28 s**，可以在既有词表上无痛扩容。
整份源码往返编码 **100% 无损**（222,435 字符逐字一致）。

**训练**（`tiny` 预设 11.02M 参数，`block_size=512`）：

| 预设 | 参数 | 4 线程吞吐 | 进程峰值内存 | 说明 |
|---|---|---|---|---|
| `nano` | 0.95M | 10,609 tok/s | — | 冒烟 / 极小语料 |
| `tiny` | 11.02M | 1,316 tok/s | 915 MB | 本机推荐，24 小时能充分训练 |
| `small` | 88.10M | 147 tok/s | 4,006 MB | 本机需 3 周+，建议上 GPU |

> 换到别的机器后**先跑 `python -m codemind bench`** —— 它会实测这几栏数字
> 并据此估算各预设的训练时长。纸面估算与实际常差 3-5 倍，必须实测。

本机 7.3 GB 内存下的真实训练曲线（78k token 语料，batch=6）：

```text
step 1   loss 7.95   随机基线 ln(4096) = 8.32
step 101 loss 4.46
step 201 loss 3.33   ← 训练损失持续下降，说明模型确实在学代码结构
step 600 loss 1.82
验证损失 4.37 → 3.98  ← 小语料上正常过拟合，验证集正确发挥作用
```

**检查点大小**（11.02M 参数）：`best.pt` **42 MB**（只存权重，供推理/`load_runner`），
`latest.pt` **126 MB**（含 AdamW 两组动量，供续训）。
两者分开是有意为之 —— 小内存机器上同时留两份全量检查点很容易撑爆磁盘。

**训练出来到底能写出什么？**（600 步模型的真实输出，greedy 解码）：

```text
提示: "class TokenShards:"
输出: 空行 + 三引号 docstring 骨架 + 正确缩进的函数体
      （结构对，语义还是浅的）

提示: "from dataclasses import dataclass\n\n@dataclass\nclass "
输出: "Dict, List, Optional, Tuple, Sequence, E402: int, Optional, Optional, ..."
      （学到了类型注解的写法，但陷入重复 —— 语料太小 + 训练步数不足以收敛）
```

**结论**：它学到了**缩进、括号配对、docstring 形态、类型注解风格**这些结构规律，
这正是 11M 参数 / 78k token 应得的结果。要产出可读的代码，需要 100MB+ 语料与 GPU。

**已知上限（诚实说明）**：

* **内存是主要瓶颈。** 本机 7.3 GB 物理内存，`tiny` 预设 + `batch_size=16` 会触及上限；
  训练时请勿并行跑其它重任务。建议 `--batch-size 6 --threads 4`。
* **`small`（88M）与 `medium`（312M）预设在本机无法训练**，需要 16 GB+ 内存或 GPU。
  代码里已备好配置，换机器即可用。
* **78k token 的语料太小。** 600 步约等于 24 个 epoch，验证损失必然反弹。
  要得到可用的补全质量，语料至少需要 100MB 量级（`prepare --source` 多指几个目录）。
* **CPU 训练适合验证流程，不适合追求质量。** 要真正可用请上 GPU
  （`--amp bf16 --preset small`）。

---

## 快速开始

> **要把项目搬到另一台机器（服务器）上训练？** 见 [MIGRATION.md](MIGRATION.md)：
> 打包、装依赖、执行策略、中文乱码、内存静默崩溃、训练时长预期，
> 以及让服务器上的 DSH 接手的具体说法。换机器前先读它。
>
> **到了新机器，第一件事是跑 `codemind bench`** —— 它实测吞吐并告诉你这台机器
> 适合训多大的模型、要多久，比任何纸面估算都准。

```bash
# 0) 安装（推荐 uv；用 pip 也可以）
uv venv
uv pip install torch numpy pytest
# 如果只想先看看能不能跑，torch 的 CPU 版足够：uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# 1) 环境自检（先跑这个，会做一次真实的前向/反向冒烟）
python -m codemind doctor

# 2) 一条命令验证全链路（几十秒，不需要任何外部数据）
python -m codemind selftest

# 3) 用本项目自己的代码训练分词器 + 生成分片
python -m codemind prepare --source . --name corpus

# 4) 训练（CPU 可跑；tiny 预设约 1500 万参数）
python -m codemind train --run runs/corpus --data data/corpus --preset tiny --steps 2000

# 5) 续写补全
python -m codemind gen --run runs/corpus --prompt "def add(a, b):"

# 6) 文件中间填空（这才是代码补全的主场景）
python -m codemind fim --run runs/corpus -f codemind/config.py --line 40

# 7) 交互式补全
python -m codemind repl --run runs/corpus

# 8) 按指令改代码（默认 dry-run，通过校验才写）
python -m codemind agent --run runs/corpus -f main.py -i "加上 __main__ 入口" \
    --check "python -m pytest -q" --rounds 3 --show-diff
```

安装为可执行命令后可以直接用 `codemind` 替代 `python -m codemind`：

```bash
uv pip install -e .
codemind doctor
```

---

## 技术要点

### 1. 字节级 BPE 分词器（`tokenizer.py`）

* 基础词表是 **256 个字节**，任意文件（含非法 UTF-8）都能编码，**永远没有 `<unk>`**。
* 预切分模式把「连续空白」切成独立块，所以 **4 空格缩进天然是一个 token** ——
  这是模型能学会 Python 结构的前提；标识符分支用 `\w+` 包含下划线，
  `my_var` / `__init__` 不会被切碎。预切分带**无损校验**：块拼接必须等于原文，
  否则直接报错，绝不让数据被静默破坏。
* **词表布局**：`0..255` 字节 → `256..260` 特殊词元预留区 → `261+` 合并词元。
  特殊词元 ID **在整个模型生命周期内恒定**，checkpoint 可安全迁移。
* **缓存 + 增量计数**：相同文本块只处理一次；合并训练用惰性堆增量维护相邻对频次，
  只重扫真正被改动的块。实测 220KB 语料训练 4096 词表 **1.6 秒**
  （朴素全量重扫实现需要 118 秒，慢 75 倍）。
* **解码严格可逆**：`decode(encode(x)) == x` 对任意文本成立，
  包括字面包含 `<|endoftext|>` 字符串的代码（本项目源码里就有）。
  生成路径需要去掉框架词元时显式传 `strip_special=True`。

### 2. FIM：代码补全能力的真正来源（`data.py`）

普通语言建模只学「往后写」，但 IDE 里用户**在文件中间**写代码。所以训练时随机把文档切三段：

```text
PSM:  <|fim_begin|> 前缀  <|fim_hole|> 后缀  <|fim_end|> 中间
SPM:  <|fim_begin|> 后缀  <|fim_hole|> 前缀  <|fim_end|> 中间
```

**损失只在「中间」区间计算**（外加紧邻的 `<|fim_end|>`，让模型学会主动收尾）。
前缀与后缀区间的损失被掩码屏蔽 —— 这是 `model.forward(loss_mask=...)` 存在的唯一原因，
也是掩码路径只为选中位置做 `lm_head` 投影（省下大量算力）的原因。

### 3. Transformer（`model.py`）

| 组件 | 选择 | 理由 |
|---|---|---|
| 位置编码 | RoPE | 无位置嵌入参数，外推稳定 |
| 归一化 | RMSNorm (Pre-Norm) | 比 LayerNorm 省算力，fp32 内部计算保证 bf16 稳定 |
| 前馈 | SwiGLU | 同算力下优于 GELU MLP |
| 注意力 | GQA（`n_kv_head < n_head`） | KV-Cache 直接缩小 `n_head/n_kv_head` 倍，CPU 上决定生死 |
| 嵌入 | 权重共享 | 小模型省约 1/4 参数 |
| 解码 | 增量 KV-Cache | 每步只前向 1 个位置，O(T²) → O(T) |

`tests/test_model.py::test_kv_cache_matches_full_forward` 逐位比对缓存路径与全量前向，
一次覆盖 RoPE 位置对齐、GQA 扩展、缓存拼接三处最容易写错的地方。

### 4. 数据流水线（`data.py`）

```
代码目录 ──采集/去重──> 文本 ──BPE──> token 流 ──> uint16 .bin 分片
                                                      │
                                    训练时 mmap 随机切窗 ┘（不占内存、无 DataLoader 多进程开销）
```

4096 词表下 1 个 token 只占 2 字节，1MB 源码约压到 200–300KB。

### 5. 自纠错编辑器（`editing.py`）—— 唯一能真正「用起来」的部分

模型输出天生不可靠，所以改代码这件事必须被工程化兜住：

```
指令 ──> 模型产出候选 ──> 语法校验 ──┐
                 ↑                  ├─ 通过 ──> 跑测试 ──> 写入文件
                 │                  │
                 └── 真实报错回灌 <──┴─ 不通过 ──> 回滚 + 重试（最多 N 轮）
```

* 校验失败 → **完整回滚**，文件保证保持原样，绝不留下半成品烂文件；
* 失败时把**真实错误输出**（语法错误定位 / 测试 traceback）回灌给下一轮；
* 默认 dry-run，`--write` 才落盘，写入前自动留 `.bak`。

`tests/test_editing.py` 用「脚本化假模型」确定性地验证了这套循环：
坏候选 → 语法反馈 → 修正 → 通过，以及全部失败后的回滚。

---

## 放大规模

CPU 适合把流程跑通，真实质量需要 GPU。项目已经预留好路径：

```bash
# GPU 训练（8GB+ 显存）：更大词表 + 更长上下文 + 混合精度
python -m codemind prepare --source . --vocab 16384 --name big
python -m codemind train --run runs/big --data data/big --preset small \
    --ctx 1024 --batch-size 16 --grad-accum 8 --amp bf16 --steps 20000

# 多目录语料
python -m codemind prepare --source ./src --source D:/datasets/python --source D:/datasets/ts
```

预设一览（`codemind presets`）：

| 预设 | 参数量级 | 建议环境 |
|---|---|---|
| `nano` | ~0.5M | 单元测试 / 冒烟 |
| `tiny` | ~15M | 单机 CPU |
| `small` | ~120M | 单卡 GPU 8GB+ |
| `medium` | ~350M | 单卡 GPU 24GB+ 或多卡 |

---

## 测试

```bash
python -m pytest -q                 # 全部（含几十秒的端到端冒烟）
python -m pytest -q -m "not slow"   # 跳过慢测
python -m pytest tests/test_editing.py -q   # 不需要 torch 的纯逻辑测试
```

测试覆盖：

* `test_bytecodec.py` 256 字节双射、任意字节串往返
* `test_tokenizer.py` 往返编码（含**字面含特殊词元的代码**）、特殊词元 ID 恒定性、
  预切分无损校验、增量训练、v1→v2 词表迁移、落盘加载
* `test_model.py` 因果性、KV-Cache 与全量前向**逐位等价**、损失掩码、GQA、
  优化器分组、采样器、FIM 载荷直通
* `test_data.py` 采集过滤/去重、分片、验证集切分、FIM 无损重排、mask 正确性、采样可复现
* `test_train.py` 损失下降、产物完整性、续训累加步数、best/latest 体积分离、词表不匹配拒绝
* `test_editing.py` 语法拦截、回滚、命令校验、自纠错循环（假模型，确定性）
* `test_cli.py` 参数解析 + 端到端 `selftest`

**217 个测试全部通过**（源码 4,832 行 + 测试 2,125 行）。
---

## 常见问题

**训练很慢？**
`doctor` 会打印前向/反向耗时。CPU 上先确认线程数：`--threads 8`（默认取逻辑核数）。
`block_size` 与 `batch_size` 对速度影响是线性的，冒烟阶段先把它们调小。

**损失降到多少算正常？**
随机基线是 `ln(vocab_size)`。4096 词表约 8.3；降到 3 以下说明模型已经学到真实的代码结构。

**`agent` 产出的代码不能用？**
这是预期内的 —— 小模型的重写能力有限。三个方向：加大训练规模、
降低温度（`--temperature 0.2 --greedy`）、挂上真实测试并加轮数
（`--rounds 5 --check "python -m pytest -q"`）。**回滚机制保证最坏情况是「什么都没改」。**

**为什么不用 HuggingFace / sentencepiece？**
本项目的目标就是「完全自研」，因此刻意不引入任何现成模型或分词器实现。
如果你想接更大的模型，`generate.py` 的 `Generator` 是干净的替换点。

**Windows 上路径怎么写？**
全部走 `pathlib`，正反斜杠都可以：`--source D:/datasets/python` 或 `--source .\src`。

**中文输出是乱码？**
PowerShell 5.1 默认按 GBK 解码子进程输出，而 Python 写 UTF-8。
`python -m codemind`（即 `__main__.py`）已自动把标准流切到 UTF-8；
若你直接用 `codemind` 可执行入口且仍乱码，加上
`$env:PYTHONIOENCODING="utf-8"` 和 `[Console]::OutputEncoding=[Text.Encoding]::UTF8`。

**训练进程崩溃、没有回溯、退出码是 `0xC0000005` 或 `0xC0000142`？**
这是 **Windows 内存耗尽**导致的 native 崩溃，不是代码 bug。
本项目在 7.3 GB 内存的机器上验证过：`tiny` 预设 + `batch_size=16` 会触及上限，
与其它内存占用大的程序并行运行时尤其容易崩。对策按优先级：

```powershell
# 1) 减小 batch（最有效）
codemind train ... --batch-size 4 --grad-accum 4   # 等效 batch 16，但峰值内存低得多

# 2) 限制线程数（每个线程都有栈和临时缓冲开销）
codemind train ... --threads 4

# 3) 训练期间不要并行跑其它重任务（含 pytest、浏览器、IDE 索引）
```
