"""从 pypi 镜像直接拉 sdist（**不依赖 pip**）。

## 为什么不用 pip

本项目的 `.venv` 是 **uv 管理**的，**默认不装 pip** ——
`python -m pip` 直接报 `No module named pip`（实测踩过）。

改用 HTTP 直取：读镜像的 simple 索引页 → 解析出 `.tar.gz` 链接 → 下载。
**少一层依赖，也就少一个失败点。**

## 镜像索引页的格式

    https://pypi.tuna.tsinghua.edu.cn/simple/<pkg>/
      <a href=".../click-8.1.7.tar.gz#sha256=..." >click-8.1.7.tar.gz</a>

取**版本号最大的那个**（简单的数字比较，够用）。

用法::

    python tools/fetch_pypi_http.py --out corpora/pypi_extra --limit 300
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.request
from urllib.parse import urljoin
from pathlib import Path

INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple/{pkg}/"
UA = {"User-Agent": "python-urllib"}

PACKAGES = """
requests urllib3 certifi charset-normalizer idna chardet
click flask jinja2 werkzeug itsdangerous markupsafe blinker
pyyaml toml tomli configobj json5
pytest pluggy iniconfig packaging attrs more-itertools
black isort flake8 pycodestyle pyflakes mccabe pylint astroid
rich colorama termcolor tabulate prettytable tqdm
httpx httpcore h11 anyio sniffio
aiohttp yarl multidict frozenlist aiosignal
sqlalchemy alembic
pydantic annotated-types typing-extensions
fastapi starlette uvicorn
markdown docutils
beautifulsoup4 soupsieve html5lib
pillow imageio
numpy scipy sympy mpmath networkx
pandas pytz python-dateutil
scikit-learn joblib threadpoolctl
matplotlib contourpy cycler fonttools kiwisolver pyparsing
seaborn statsmodels patsy
pytest-cov coverage
virtualenv distlib filelock platformdirs
setuptools wheel build pyproject-hooks
psutil
paramiko cryptography cffi pycparser bcrypt pynacl
boto3 botocore s3transfer jmespath
redis pymongo
celery kombu billiard amqp vine
gunicorn waitress
email-validator dnspython
python-dotenv marshmallow cerberus jsonschema
tablib openpyxl xlrd reportlab python-docx
pypdf
regex unidecode faker arrow pendulum humanize inflection
boltons toolz funcy sortedcontainers cachetools diskcache
dill cloudpickle msgpack protobuf
ruamel.yaml
platformdirs
exceptiongroup
sniffio
six
python-multipart
websockets
uvloop
watchfiles
orjson
ujson
simplejson
pygments
mako
alembic
greenlet
nose
mock
freezegun
hypothesis
pytest-asyncio
pytest-mock
responses
betamax
factory-boy
sqlparse
tabulate
arrow
dateparser
pytzdata
tzlocal
babel
langcodes
""".split()


def sdist_url(pkg: str) -> str | None:
    """读简单索引，返回版本号最大的 .tar.gz 链接。

    ⚠️ **索引页里的链接是相对路径**（形如 `../../packages/f8/5c/.../Click-7.1.2.tar.gz`），
    必须用 `urljoin` 相对索引页解析。直接当绝对 URL 用会全部失败 —— 实测踩过，
    而且报的是"下载失败"，看不出根因。
    """
    base = INDEX.format(pkg=pkg)
    try:
        with urllib.request.urlopen(
                urllib.request.Request(base, headers=UA), timeout=25) as r:
            html = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    raw = re.findall(r'href="([^"#]+\.tar\.gz)', html)
    if not raw:
        return None
    links = [urljoin(base, l) for l in raw]

    def ver(u: str) -> tuple:
        m = re.search(r'-(\d[^-]*?)\.tar\.gz$', u.rsplit('/', 1)[-1])
        if not m:
            return (0,)
        return tuple(int(x) if x.isdigit() else 0
                     for x in re.split(r'[.\-]', m.group(1)))[:4]

    return max(links, key=ver)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpora/pypi_extra")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-mb", type=float, default=40.0,
                    help="单个包超过这个大小就跳过（避免偶然的超大包）")
    args = ap.parse_args()

    out = Path(__file__).resolve().parents[1] / args.out
    out.mkdir(parents=True, exist_ok=True)

    pkgs = list(dict.fromkeys(PACKAGES))[: args.limit]
    print(f"[pypi] {len(pkgs)} 个包 → {out}")

    t0 = time.time()
    ok = skip = fail = 0
    total = 0
    for i, pkg in enumerate(pkgs, 1):
        dest = out / f"{pkg}.tar.gz"
        if dest.exists() and dest.stat().st_size > 1024:
            skip += 1
            continue
        url = sdist_url(pkg)
        if not url:
            fail += 1
            continue
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=90) as r:
                data = r.read()
            if len(data) > args.max_mb * 1e6:
                fail += 1
                continue
            dest.write_bytes(data)
            total += len(data)
            ok += 1
        except Exception:
            fail += 1
        if i % 20 == 0:
            print(f"  {i}/{len(pkgs)}  已下 {ok}  跳过 {skip}  失败 {fail}  "
                  f"{total/1e6:.0f} MB  {(time.time()-t0)/60:.1f} 分钟", flush=True)

    files = list(out.glob("*.tar.gz"))
    sz = sum(f.stat().st_size for f in files)
    print(f"\n[pypi] ✓ {len(files)} 个包 / {sz/1e6:.1f} MB / {(time.time()-t0)/60:.1f} 分钟")
    print(f"[pypi] 本次新增 {ok}，跳过 {skip}，失败 {fail}（可重跑，已下的会跳过）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
