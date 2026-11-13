"""pytest 共享配置。"""

from __future__ import annotations

import sys
from pathlib import Path

# 保证在未安装包的情况下（直接 pytest tests/）也能导入 codemind
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
