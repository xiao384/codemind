# PAI-DLC 上云具体步骤

> 目标：**在云 GPU 上把模型从 11M 提升到 88M（small）**，突破已实测确认的参数容量瓶颈。
>
> **总预算**：100 CU\*H / 3 个月。**本方案预计用 8~23 CU\*H（8%~23%）**，留足余量。

---

## 阶段 0：本机打包（已完成 ✅）

```powershell
cd E:\AI
.\.venv\Scripts\python.exe tools\pack_for_cloud.py --out cloud\codemind.tar.gz
```

**产出**：`cloud/codemind.tar.gz`（**53 MB**，含 55 个文件）+ `cloud/MANIFEST.json`

```powershell
# 上传前自检（证明二进制没坏）
.\.venv\Scripts\python.exe tools\pack_for_cloud.py --verify cloud\codemind.tar.gz
# → [verify] 55/55 通过（大文件比对 sha256）
```

---

## 阶段 1：创建 PAI-DLC 任务（控制台操作）

### 1.1 上传数据

| 项 | 值 |
|---|---|
| 上传文件 | `cloud/codemind.tar.gz` |
| 目标 | **OSS Bucket**（DLC 要能挂载） |
| ⚠️ **必须二进制上传** | 分片是 `uint16` 裸数组。**任何文本模式/FTP ASCII 传输都会静默毁数据** |

### 1.2 创建工作空间

| 配置项 | 建议值 | 理由 |
|---|---|---|
| **镜像** | `pytorch:2.x-cuda12.1` 官方镜像 | 自带 CUDA torch，**省掉 2.5 GB wheel 上传** |
| **规格** | **A10 单卡**（24 GB） | `small` 88M 只需 ~7.5 GB 显存，**A10 绰绰有余**；V100 更贵但本项目用不上 |
| **数据集挂载** | OSS → `/mnt/data` | |
| **代码目录** | `/mnt/data/codemind` | |
| **公网** | **开启** | 若不能开，见「阶段 1 备选」 |

### 1.3 启动命令

```bash
cd /mnt/data
tar xzf codemind.tar.gz
cd .        # 解包后就是项目根目录

# ✅ 第一件事：校验二进制完整性
python tools/pack_for_cloud.py --verify ../codemind.tar.gz
# 必须看到「55/55 通过」，否则不要往下走

# ✅ 第二件事：确认 GPU 可用
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 阶段 2：**先量 MFU，再决定训什么**（约 10 分钟，0.2 CU\*H）

```bash
python -m codemind bench --presets tiny small --device cuda
```

**这一步不能跳。** 本项目已验证过：同类外推**乐观 2.4~3.2 倍**
（MIGRATION 预估 tiny 10 小时，实测 32 小时）。

**算法**：

```
实测 TFLOPS = 6 × N × (tokens_per_sec)
MFU = 实测 TFLOPS / 31.2      # A10 fp32 峰值
```

| 实测 MFU | 决策 |
|---|---|
| **≥ 25%** | ✅ 按计划训 `small`；余量大可考虑 `medium` |
| **15% ~ 25%** | ✅ 训 `small`，但**不要碰 medium** |
| **< 15%** | ⛔ **停下**，先查数据管线（本机曾因一个 bug 丢掉 36% 吞吐） |

---

## 阶段 3：训练 `small`（88.1M）

### 3.1 先用 100 步量真实吞吐

```bash
python -m codemind train --run runs/small --data data/corpus2 \
    --preset small --steps 100 --threads 8 --amp bf16 --device cuda
```

**读最后一行**：`训练完成：100 步 / X.XXM tokens / Y 分钟`

```
真实吞吐 = X.XXM tokens / Y 分钟
```

### 3.2 算总步数（**用实测值，不用我的估算**）

```
总步数 = 1,762,000,000 ÷ (每步 token 数)
每步 token 数 = batch_size × ctx = 32 × 512 = 16,384
→ 总步数 ≈ 107,500
```

### 3.3 正式训练

```bash
nohup python -m codemind train --run runs/small --data data/corpus2 \
    --preset small --steps 107500 --threads 8 --amp bf16 --device cuda \
    --ckpt-interval 1000 > train.log 2>&1 &
```

**用 `nohup` + `&`**：DLC 任务断开 SSH 会杀掉前台进程。

### 3.4 看训练是否正常

```bash
tail -f train.log
```

**判据**：loss 应从前 5 步的 **~8.3（=ln(4096)）** 快速下降。
若 100 步后还在 8.0 以上 → 有问题。

---

## 阶段 4：训练中途出能力数字（**不要等跑完**）

每 2 小时跑一次：

```bash
cp runs/small/latest.pt runs/_verify/ckpt-mid.pt
python tools/cb1.py --run runs/_verify/ckpt-mid.pt \
    --tokenizer data/corpus2/tokenizer.json --samples 400 --threads 4 --tag small-mid
```

**判据表**（11M 的终局是已知的）：

| 指标 | 11M 终局（实测） | 88M 达到多少算成功 |
|---|---|---|
| `edit_sim` 余量 | **+7.42pp** | **> +10pp** |
| `syntax_ok` 余量 | +6.38pp | > +8pp |
| `hole_ppl` | 3.90 | **< 3.0** |

**若余量显著超过 11M → "参数容量"假设成立，继续训完。**
**若持平 → 假设被推翻**，应立刻停下来重新想（省下几十 CU\*H）。

---

## 阶段 5：训练完成后

```bash
python tools/key_eval.py --steps 107500 --samples 1000
python tools/cb2.py --selftest --samples 20      # ⚠️ 先验证仪器，再测模型
python tools/cb2.py --run runs/small/latest.pt --mode fim --samples 400
```

产出：`runs/_verify/key-eval-step107500.txt`（终局汇总）

---

## 阶段 1 备选：如果 PAI-DLC **不能连公网**

需要自己准备 CUDA torch wheel（**约 2.5 GB，单独打包，不要塞进 codemind.tar.gz**）：

```powershell
# 本机下载
uv pip download torch --index-url https://download.pytorch.org/whl/cu121 -d E:\AI\cloud\wheels

# 单独压一个包（wheel 已经压缩过，不用再压）
tar -czf cloud\wheels.tar.gz -C E:\AI\cloud wheels
```

云上：

```bash
tar xzf wheels.tar.gz
pip install --no-index --find-links wheels/ torch
python -c "import torch; print(torch.cuda.is_available())"   # 必须是 True
```

---

## 预算分配（100 CU\*H）

| 阶段 | 预估 | 累计 |
|---|---|---|
| 2. 量 MFU | 0.2 h | 0.2 |
| 3.1 量吞吐（100 步） | 0.2 h | 0.4 |
| 3.3 `small` 完整训练 | **7~22 h** | **7~22** |
| 4. 中途评估（×5 次） | 1 h | 8~23 |
| **余量** | **77~92 h** | |

**决策点**：阶段 2/3.1 的真实 MFU 出来后，
- MFU ≥ 25% 且余量充足 → 可规划 `medium`（312M，约 86 h）
- 否则 → 把余量用在**多轮 `small` 实验**（换数据、开 `--fim-terminator`、对话微调）

---

## 三个最容易出事的地方

1. **数据二进制完整性** —— 上传后**必须先跑 `--verify`**。分片是 `uint16` 裸数组，
   文本模式传输会静默毁掉，而且**训练不会报错**（只是 loss 降不下去）。
2. **`best.pt` 只存权重，`latest.pt` 存优化器状态** —— 续训要用 `latest.pt`。
3. **看产不看日志** —— 判断训练是否在推进，看 `runs/small/latest.pt` 的**修改时间**，
   不要只看 `train.log`（本机踩过：`Tee-Object` 缓冲导致日志为空，但训练其实正常）。
