"""批量下载 pypi 源码包（sdist），用于扩充代码语料。

## 为什么用 sdist 而不是 wheel

```
wheel（.whl）  只含安装时要的东西：.py + 编译好的 .so    → 拿不到 C 源码
sdist（.tar.gz）含完整源码：.py + .c + .h + 测试 + 文档   → **内容多得多**
```

本项目现有语料里 12% 是 C/C++ 头文件 —— 那些只能从 sdist 里拿到。

## 为什么从清华镜像下

实测可达性（本会话量过）：

| 源 | 速度 |
|---|---|
| **pypi 清华镜像** | **195~658 KB/s** ✅ |
| github / raw.githubusercontent | 6~16 KB/s ❌ |
| huggingface | API 超时 ❌ |

用法::

    python tools/fetch_pypi_corpus.py --out corpora/pypi_extra --limit 300
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

#: 只挑**纯 Python 或源码可读**的包 —— 避开只有二进制 wheel 的
PACKAGES = """
requests urllib3 certifi charset-normalizer idna chardet
click flask jinja2 werkzeug itsdangerous markupsafe blinker
pyyaml toml tomli tomli-w configobj json5
pytest pluggy iniconfig packaging attrs more-itertools
black isort flake8 pycodestyle pyflakes mccabe pylint astroid
rich colorama termcolor tabulate prettytable tqdm
httpx httpcore h11 anyio sniffio
aiohttp yarl multidict frozenlist aiosignal async-timeout
sqlalchemy alembic
pydantic annotated-types typing-extensions
fastapi starlette uvicorn
jinja2 markdown docutils restructuredtext-lint
beautifulsoup4 soupsieve lxml html5lib
pillow imageio
numpy scipy sympy mpmath networkx
pandas pytz python-dateutil
scikit-learn joblib threadpoolctl
matplotlib contourpy cycler fonttools kiwisolver pyparsing
seaborn statsmodels patsy
pytest-cov coverage
virtualenv distlib filelock platformdirs
setuptools wheel build pyproject-hooks
pip-tools pipreqs
psutil pywin32
paramiko cryptography cffi pycparser bcrypt nacl pynacl
boto3 botocore s3transfer jmespath
redis pymongo
celery kombu billiard amqp vine
gunicorn waitress
email-validator dnspython
python-dotenv
marshmallow
cerberus jsonschema
tablib openpyxl xlrd xlwt
reportlab
docx python-docx
pypdf pdfminer.six
chardet
regex
text-unidecode unidecode
faker
arrow pendulum
humanize
inflection
boltons toolz funcy
sortedcontainers
cachetools
diskcache
joblib
dill cloudpickle
msgpack
protobuf
grpcio
""".split()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpora/pypi_extra")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    pkgs = list(dict.fromkeys(PACKAGES))[: args.limit]
    print(f"[pypi] 目标 {len(pkgs)} 个包 → {out}")
    print(f"[pypi] 镜像 {MIRROR}")

    t0 = time.time()
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--no-deps",              # 不递归依赖，避免下载量爆炸
        "--no-binary", ":all:",   # ⚠️ 关键：拿 sdist 而非 wheel
        "--dest", str(out),
        "--index-url", MIRROR,
        "--trusted-host", "pypi.tuna.tsinghua.edu.cn",
        "--timeout", str(args.timeout),
        "--retries", "2",
        "--quiet",
    ]
    # 分批下：一个包失败不影响其他
    B = 25
    ok = fail = 0
    for i in range(0, len(pkgs), B):
        batch = pkgs[i:i + B]
        r = subprocess.run(cmd + batch, capture_output=True, text=True)
        got = len(list(out.glob("*.tar.gz"))) + len(list(out.glob("*.zip")))
        if r.returncode == 0:
            ok += len(batch)
        else:
            fail += len(batch)
        print(f"  批次 {i//B+1}: {len(batch)} 个 → 累计文件 {got}  "
              f"({(time.time()-t0)/60:.1f} 分钟)", flush=True)

    files = sorted(list(out.glob("*.tar.gz")) + list(out.glob("*.zip")))
    total = sum(f.stat().st_size for f in files)
    print(f"\n[pypi] ✓ {len(files)} 个源码包 / {total/1e6:.1f} MB "
          f"/ {(time.time()-t0)/60:.1f} 分钟")
    print(f"[pypi] 失败批次 {fail} 个包（可重跑，已下的会跳过）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
