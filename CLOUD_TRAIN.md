# 阿里云训练部署清单（PAI-DLC / PAI-EAS）

> 目的：把 CodeMind 放到云端 GPU 上训练，**突破 11M 的参数容量瓶颈**
> （这一条已被实测确认：11M 能学到"主题关联"，学不到"回答问题"）。
>
> ⚠️ **本文件里的所有 GPU 耗时都是外推，不是实测。** 依据是本机 CPU 实测吞吐
> + A10/V100 的公开规格。**本项目已证明同类外推会乐观 2.4~3.2 倍**
> （`MIGRATION.md` 预估 tiny 10 小时，实测 32 小时）。
> **所以第一步的目标不是"训完"，而是"量出真实 MFU"。**

---

## 一、额度与选型

| 产品 | 额度 | 用途 |
|---|---|---|
| **PAI-DLC**（模型训练） | **100 CU\*H / 3 个月** | 训练，支持 A10 / V100 / G6 |
| **PAI-EAS**（在线服务） | 500 元 / 1 个月 | 推理服务（Web 界面） |

**按 `FLOPs = 6 × N × D`、A10 保守 30% MFU ≈ 37.5 TFLOPS 估算：**

| 预设 | 参数 | Chinchilla token | 算力 | 预计耗时 | 预算占比 |
|---|---|---|---|---|---|
| tiny（现状） | 11M | 220M | 1.45e16 | 0.1 h | 0.1% |
| **small** | **88.1M** | 1,762M | 9.31e17 | **7 h** | 7% |
| medium | 312M | 6,240M | 1.17e19 | 86 h | 86% ⚠️ |

**外推偏 2.4~3.2 倍时的现实区间：**

| | 乐观 | 现实 | 悲观 |
|---|---|---|---|
| small | 7 h | **17 h** | **22 h** |
| medium | 86 h | **207 h ❌** | **276 h ❌** |

### 🎯 结论：**先 small，不要直接上 medium**

- **small**：即使悲观也只占 22% 预算，**且能把"参数容量"这个假设验证掉**（11M→88M 是 8 倍）
- **medium**：只有 MFU ≥ 25% 才可能塞进 100 CU\*H，**风险太高**

---

## 二、第一步的执行顺序（不要跳步）

### ① 先跑基准，**量真实 MFU**（约 10 分钟，占预算 0.2%）

```bash
python -m codemind bench --presets tiny small --device cuda --threads 8
```

**判据**（这是**唯一**该看的东西）：

```
MFU = 实测 TFLOPS / A10 峰值 31.2 TFLOPS(fp32) 或 125 TFLOPS(fp16)
实测 TFLOPS = 6 × N × tokens_per_sec
```

| MFU | 结论 |
|---|---|
| ≥ 25% | `medium` 可行，可以规划 |
| 15%~25% | 只做 `small`，把额度用在多轮实验 |
| < 15% | **先查数据管线**，不要开始长跑（本机就踩过：bug 吃掉 36% 吞吐） |

### ② small 的完整训练

```bash
python -m codemind train --run runs/small --data data/corpus2 \
    --preset small --steps 21500 --threads 8 --amp bf16 --device cuda \
    --ckpt-interval 500
```

> `--steps` 算法：`88.1M × 20 / (batch × ctx)`。以 batch 32、ctx 512 计，
> 每步 16,384 token → `1,762M / 16,384 = 107,544` 步。
> ⚠️ **不是 21,500** —— 上面那行是占位。**上云后先跑 100 步量实际吞吐，再算总步数。**

### ③ 训练中途就能出能力数字

```bash
# 每 2 小时看一眼，不要等跑完
python tools/cb1.py --run runs/small/latest.pt --samples 400 --threads 4 --tag small-mid
```

**判据**：`edit_sim` 余量 vs 机会基线。**11M 的终局是 +7.42pp**，
88M 若显著超过它，说明"参数容量"假设成立。

---

## 三、软件就绪情况（**已实测确认，不用改代码**）

```
config.py::resolve_device    支持 auto/cpu/cuda/cuda:0/mps，自动检测
CLI                          各子命令都有 --device
train                        --amp off/bf16/fp16，pick_dtype 正确处理
                             （bf16 不缩放、fp16 用 grad scaler）
```

**唯一要做的**：装 CUDA 版 torch。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

⚠️ **如果 PAI-DLC 不通公网**，就要把 wheel 一起传上去（约 2.5 GB）。

---

## 四、要上传的东西

| 内容 | 大小 | 必需 |
|---|---|---|
| `codemind/` `tools/` `tests/` | ~2 MB | ✅ |
| `data/corpus2/`（68.6M token） | 132 MB | ✅ |
| `data/chat/`（对话语料） | 63 MB | 可选（聊天实验用） |
| `runs/_verify/ckpt-step13428.pt` | 126 MB | 可选（续训起点） |
| `runs/_ref/`（GPT-2 对照） | 2 GB | ❌ 不必传 |
| **合计** | **~320 MB** | |

---

## 五、三个必须先确认的坑

1. **PAI-DLC 能否连公网装 pip 包？**
   不能 → 把 CUDA torch wheel + 依赖一起打包上传。

2. **数据挂载方式**：DLC 用 OSS 还是 NAS？
   `--data` 必须能读到 `corpus2.meta.json` 与 `corpus2.0000N.bin`。
   ⚠️ **分片是 `uint16` 小端裸数组**，不是文本 —— 传输时**必须二进制模式**，
   任何"按文本复制"都会毁掉数据（前 2 字节被当 BOM 就全错位）。

3. **`run_server.ps1` 是 Windows 脚本，云上是 Linux。**
   直接调 `python -m codemind train ...`，不要用那个脚本。

---

## 六、本机已具备、可直接搬过去的能力

| 工具 | 作用 |
|---|---|
| `tools/cb1.py` | 代码能力基准（**云端也要跑，否则不知道训得好不好**） |
| `tools/cb2.py` | 执行通过率基准（400 任务，含仪器自检） |
| `tools/key_eval.py` | 关键检查点一键评估 |
| `tools/cb1_watch.py` | 能力曲线看守（**注意：杀进程时不要用排除法**） |
| `tools/webchat.py` | Web 界面（可部署到 PAI-EAS） |
| `tools/make_chat_corpus.py` | 对话语料构建 |

⚠️ **看守必须能被发现已死**：本项目发生过一次"看守静默死亡 11 小时"。
判据用**产出物**（`cb1-curve.jsonl` 的 mtime），**不要用日志文件**。

---

## 七、预算分配建议（100 CU\*H）

| 步骤 | 预算 | 说明 |
|---|---|---|
| ① 基准量 MFU | 0.2 h | **必须先做** |
| ② small 完整训练 | 7~22 h | 核心目标 |
| ③ 中途评估 | 1 h | cb1 + cb2 |
| ④ 余量 | **76~91 h** | 看 ② 的真实 MFU 再决定：上 medium，或多轮 small |
| ⑤ PAI-EAS 部署 | 独立额度 | Web 界面，500 元/月 |

**关键原则**：**先花 0.2 小时把不确定性消掉，再花 20 小时。**
本项目的教训是"外推会乐观 2.4~3.2 倍"——那就**先量，再赌**。
