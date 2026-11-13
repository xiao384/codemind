# 迁移到服务器（32GB / E5-2680 v3 ×2 / 无 GPU）

本文档只讲一件事：**把这套代码搬到服务器上跑起来**。
踩过的坑都写在「避坑清单」里，照着做能省掉几小时排查。

---

## 一、在你当前机器上打包

```powershell
cd D:\AI
powershell -ExecutionPolicy Bypass -File .\tools\pack.ps1 -Out D:\codemind-transfer.zip
```

产出约 **110 KB**（只含源码与文档）。刻意**不打包** `.venv`：
虚拟环境里有硬编码的绝对路径和平台相关二进制，跨机复制必然出问题，
而服务器上重装依赖只要一两分钟。

包里会自带一份 `SERVER_QUICKSTART.md`。

---

## 二、服务器上三步

### 第 1 步：解压 + 装依赖

```powershell
Expand-Archive -Path D:\codemind-transfer.zip -DestinationPath D:\codemind -Force
cd D:\codemind

# 没有 uv 就先 pip install uv
uv venv
uv pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
uv pip install pytest
```

CPU 版 torch 约 **200 MB**；如果用默认源会拉到 2.5 GB 的 CUDA 版——你没有 GPU，纯属浪费。

### 第 2 步：确认能跑 + 量出真实吞吐

```powershell
python -m codemind doctor     # 环境自检
python -m codemind selftest   # 全链路冒烟（几十秒）
python -m codemind bench --presets nano tiny small
```

**`bench` 是这一步的重点**，它会打印每个预设的实测吞吐与训练时长估算。
拿它来决定训哪个尺寸，别拍脑袋。

### 第 3 步：准备数据 + 长跑训练

```powershell
# 准备数据（把 D:\corpora\... 换成你的代码目录）
python -m codemind prepare `
    --source D:\corpora\github `
    --source D:\corpora\internal `
    --name corpus --vocab 16384 --preset small --tokenizer-chars 50000000

# 长跑训练（崩溃自愈 + 自动续训）
.\tools\run_server.ps1 -Mode train -Run runs\small -Preset small -TargetSteps 20000
```

`prepare` 结束时会打印「这份数据够训多大的模型」，照着它选预设。

---

## 三、避坑清单（都是实测踩出来的）

### 1. 脚本执行策略

Windows 默认禁止运行未签名脚本：

```powershell
# 方式 A：单次绕过（推荐，不动系统设置）
powershell -ExecutionPolicy Bypass -File .\tools\run_server.ps1 -Mode bench

# 方式 B：只对当前会话放开
Set-ExecutionPolicy -Scope Process Bypass
```

### 2. 中文乱码

PowerShell 5.1 按 GBK 解码子进程输出，Python 写的是 UTF-8：

```powershell
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

`python -m codemind` 已自动处理标准流，但 `.ps1` 脚本本身必须是 **UTF-8 with BOM**——
否则 PS 5.1 会把中文字符串读成乱码并报一堆语法错误。`pack.ps1` 已按此保存。

### 3. 内存不足会「静默崩溃」

这是最容易浪费一晚上的坑：Windows 内存枯竭时进程以 `0xC0000005`（访问冲突）
或 `0xC0000142`（DLL 初始化失败）退出，**不打印任何 Python 回溯**。

对策（按优先级）：

```powershell
--batch-size 4 --grad-accum 4    # 等效 batch 16，峰值内存低得多
--threads 24                     # 别超过物理核心数
--ckpt-interval 200              # 落盘更勤，崩溃损失更小
```

`run_server.ps1 -Mode train` **已经内置自愈**：识别这几种崩溃码，
自动把 batch 减半并从 `latest.pt` 续训；batch 降到 1 后改为提高梯度累积。

### 4. 32GB 内存的实际分配

| 用途 | 占用 |
|---|---|
| `small` 预设（88M 参数）+ AdamW 动量 | ~1.4 GB |
| 激活值 + logits（batch 8 / ctx 1024） | ~3-5 GB |
| 语料分片 memmap（1GB 代码 → 约 250MB 分片） | 按需分页，几乎不占常驻 |
| 分词器训练（流式，只读样本） | < 1 GB |

**结论：32GB 够用，不要同时跑两个训练进程。**

### 5. 磁盘

1GB 源码 → 约 250MB 分片（uint16）。训练产物每个检查点：
`best.pt`（纯权重，88M → ~350MB）+ `latest.pt`（含优化器，~1.1GB）。
跑之前确认至少 **10 GB 空闲**。

---

## 四、训练时长预期

这台机器（12 线程 / 7.3GB）实测：`nano` 10,609 tok/s、`tiny` 1,342 tok/s、`small` 190 tok/s。
服务器有 **24 物理核心 + 4 倍内存带宽**，预计提升 4-6 倍：

| 预设 | 参数量 | 服务器预计吞吐 | Chinchilla 所需 token | 预计时长 |
|---|---|---|---|---|
| `nano` | 0.95M | ~50k tok/s | 19M | **约 10 分钟** |
| `tiny` | 11.0M | ~6k tok/s | 220M | **约 10 小时** |
| `small` | 88.1M | ~900 tok/s | 1.76B | **约 22 天** |

**这是你必须先接受的现实：**

- `nano` 能真正「训练充分」，但模型太小，补全质量有限；
- `tiny` 是**性价比拐点**——一个晚上到一天能跑完，能学到像样的代码结构；
- `small` 在纯 CPU 上要 **3 周以上不间断**。除非你确定要长期挂机，
  否则建议先用 `tiny` 跑通全流程、验证数据质量，再决定是否上 `small`。

> 想突破这个上限，唯一的办法是 GPU。哪怕一张二手 P40/P100，
> `small` 也能从 22 天压到 **几小时**。

---

## 五、让服务器上的 DSH 接手

服务器上有 DSH 的话，最省事的方式是把任务直接交给它。
建议这样开场（把实际路径替换进去）：

```
我在 D:\codemind 有一个从零实现的代码大模型项目（Python + PyTorch，纯 CPU）。
请帮我：
1. 读 SERVER_QUICKSTART.md 和 README.md，确认环境依赖已装好
2. 跑 python -m codemind doctor 和 python -m codemind selftest，把结果贴给我
3. 跑 python -m codemind bench --presets nano tiny small，根据实测吞吐
   告诉我这台机器最适合训哪个预设、预计多久
4. 我的代码语料在 D:\corpora\github（约 XXX MB），
   帮我跑 prepare 并在结束后报告「这份数据够训多大的模型」
先只做 1-3 步，等我确认后再动数据。
```

要点：

- **让它先读 `README.md`**，里面有完整的架构说明与实测数据，它不用重新摸索；
- **明确要求先做 1-3 步**，避免它直接开一个几天的训练；
- 训练阶段用 `.\tools\run_server.ps1 -Mode train`，它自带崩溃自愈，
  比让 DSH 手写循环更稳。

### 长期挂机建议

```powershell
# 用 Start-Process 后台跑，输出重定向到日志
Start-Process -FilePath "powershell" `
  -ArgumentList "-ExecutionPolicy","Bypass","-File","D:\codemind\tools\run_server.ps1",
                "-Mode","train","-Preset","tiny","-TargetSteps","40000" `
  -RedirectStandardOutput "D:\codemind\train.log" `
  -RedirectStandardError  "D:\codemind\train.err" `
  -WindowStyle Hidden
  
# 看进度
Get-Content D:\codemind\train.log -Tail 20 -Wait
```

`latest.pt` 每 `--ckpt-interval` 步落盘，所以随时 Ctrl-C 或断电，
重跑同一条命令即可从断点续训（`--no-resume` 才会从头开始）。

---

## 六、验收标准

训练跑起来后，用这三条判断它是否真的在工作：

```powershell
# 1. 损失必须明显低于随机基线 ln(vocab_size)
#    16384 词表的基线是 9.70；降到 4 以下说明学到了真实结构
python -m codemind info --run runs\tiny --data data\corpus

# 2. 补全出来的东西应该有正确的缩进与括号配对
python -m codemind gen --run runs\tiny --prompt "def process(items):`n" --max-tokens 64

# 3. 验证损失若持续上升而训练损失下降，说明过拟合了 —— 该加数据或减步数
```

**不要用「生成的代码能不能跑」来判断**——小模型产出可运行代码的概率很低，
这是规模问题不是 bug。`agent` 命令的语法校验与回滚会保证最坏情况是「什么都没改」。
