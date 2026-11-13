#!/usr/bin/env bash
# 会话型 GPU 环境（ModelScope Notebook / DLC 等）的训练启动脚本。
#
# ## 为什么需要它
#
# 这类环境**会话有时间限制**：到点被强制停止、空闲自动休眠、非持久化目录会丢。
# 所以不能"起一个 38 小时的任务就不管了"，必须做成**可断点续训、反复重启**。
#
# 本脚本做的事：
#   1. 自检 GPU 与数据（数据坏了训练不报错，只是 loss 降不下去——本项目踩过）
#   2. **自动判断是首跑还是续训**（看 runs/<name>/latest.pt 在不在）
#   3. 频繁存检查点（默认每 200 步），会话被杀也不会丢太多
#   4. 打印当前进度与真实吞吐，便于判断是否达到预期
#
# 用法::
#
#     bash tools/run_session.sh small_v2 data/corpus2 20000
#     # 会话被杀后，**原样再跑一次**即可续训
#
# ⚠️ 输出目录必须放在**持久化**路径上，否则会话重置后检查点全丢。

set -euo pipefail

RUN_NAME="${1:-small_v2}"
DATA="${2:-data/corpus2}"
STEPS="${3:-20000}"
PRESET="${4:-small}"
BATCH="${5:-32}"
THREADS="${6:-8}"
CKPT_EVERY="${CKPT_EVERY:-200}"

RUN_DIR="runs/${RUN_NAME}"

echo "=============================================================="
echo " CodeMind 会话式训练   $(date '+%F %T')"
echo "=============================================================="

# ---------- 1. 环境自检 ----------
echo "[1/4] 环境自检"
python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("✗ 没装 torch。ModelScope 镜像一般自带；否则 pip install torch")
print(f"  torch   {torch.__version__}")
if not torch.cuda.is_available():
    sys.exit("✗ CUDA 不可用。这个脚本是给 GPU 会话用的；纯 CPU 请直接跑 train")
p = torch.cuda.get_device_properties(0)
print(f"  GPU     {p.name}  {p.total_memory/1e9:.1f} GB")
print(f"  bf16    {'支持' if torch.cuda.is_bf16_supported() else '不支持（改用 --amp fp16）'}")
PY

# ---------- 2. 数据自检 ----------
echo "[2/4] 数据自检（分片是 uint16 裸数组，坏了训练不报错，只会 loss 降不下去）"
python - <<PY
import json, sys
from pathlib import Path
d = Path("${DATA}")
metas = sorted(d.glob("*.meta.json"))
if not metas:
    sys.exit(f"✗ {d} 下没有 *.meta.json —— 数据没传对")
m = json.loads(metas[0].read_text(encoding="utf-8"))
files = m.get("train_files", []) + m.get("val_files", [])
miss = [f for f in files if not (d / f).exists()]
if miss:
    sys.exit(f"✗ 缺失分片: {miss}")
total = sum((d / f).stat().st_size // 2 for f in files)
print(f"  {m['name']}: {len(m['train_files'])} 训练 + {len(m['val_files'])} 验证 = {total:,} token")
PY

# ---------- 3. 判断首跑还是续训 ----------
echo "[3/4] 检查续训点"
RESUME=""
if [ -f "${RUN_DIR}/latest.pt" ]; then
  ITER=$(python -c "import torch;print(torch.load('${RUN_DIR}/latest.pt',map_location='cpu',weights_only=False).get('iteration',0))" 2>/dev/null || echo "?")
  echo "  ✓ 发现 ${RUN_DIR}/latest.pt（step ${ITER}）→ **续训**"
  echo "    ⚠️ 续训必须用 latest.pt（含优化器状态）；best.pt 只有权重，续训会掉精度"
else
  echo "  未发现检查点 → **首跑**"
fi

# ---------- 4. 起训练 ----------
echo "[4/4] 启动训练"
echo "  输出   ${RUN_DIR}   （⚠️ 必须在持久化路径上）"
echo "  检查点 每 ${CKPT_EVERY} 步"
echo

mkdir -p "${RUN_DIR}"

# `--ckpt-interval` 设小：会话随时可能被杀
python -m codemind train \
    --run "${RUN_DIR}" \
    --data "${DATA}" \
    --preset "${PRESET}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH}" \
    --threads "${THREADS}" \
    --amp bf16 --device cuda \
    --ckpt-interval "${CKPT_EVERY}" \
    2>&1 | tee "${RUN_DIR}/train.log"

echo
echo "=============================================================="
echo " 结束 $(date '+%F %T')"
echo " 若会话被中途杀掉：**原样再跑一次这个脚本**即可从 latest.pt 续训"
echo " 判据：loss 应持续下降；若触底反弹，说明 LR 过高，调 --lr 再降"
echo "=============================================================="
