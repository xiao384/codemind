"""cb-2：**执行评测环** —— 用"跑不跑得通"而不是"像不像"来量代码能力。

与 `cb-1` 的区别（两者互补，不是替代）：

| | `cb-1` | `cb-2`（本工具） |
|---|---|---|
| 判据 | 文本相似度（`edit_sim`） | **真的执行，看 doctest 过不过** |
| 能回答 | "补出来像不像目标" | **"补出来的代码能不能跑对"** |
| 弱点 | 像 ≠ 对 | 覆盖面窄（只测有 doctest 的纯函数） |

设计要点：

1. **任务来自真实标准库/第三方库**，不是我自己编的题。
2. **判据是库作者亲手写的 doctest** —— 天然可抽取、不需要我造输入。
3. **门槛：doctest 必须在原函数上通过**。不通过的一律丢掉——
   否则判据本身不可信，测出来的数字没有意义。
4. **执行在独立子进程里，带超时**。⚠️ 这不是真正的沙箱（见 `--help` 末尾的说明）。

用法：

    # 1) 只建任务集并校验判据（不跑模型）
    python tools/cb2.py --build --out runs/_verify/cb2-tasks.json

    # 2) 跑模型评测
    python tools/cb2.py --run runs/_verify/ckpt-step2500.pt --tasks runs/_verify/cb2-tasks.json \
        --samples 100 --timeout 8
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT / "runs" / "_verify"
PY = sys.executable

SRC_ROOTS = [
    # ⚠️ **只用标准库**。踩过的坑：把 `E:\AI\corpora\python-packages` 也放进来后，
    # 候选从 2,571 → 大量来自 sympy/numpy/torch 的模块，而它们的依赖没装，
    # 于是失败原因全是 `ModuleNotFoundError` / `NameError`（导入的同名模块不存在），
    # 产出率反而掉到 0，而且每次失败都要等超时，慢了 4 倍。
    # 标准库是**自包含**的：上下文里的 import 一定能解析。这是任务池质量的关键。
    Path(r"C:\Program Files\python\Lib"),
    # 第三方包里**纯 Python、依赖封闭**的模块也是好来源（如 more_itertools）。
    # 实测：只有标准库时候选 46 个、入池 2 个；加上这个目录后候选 225 个、入池 18 个。
    # 配合下面的 SAFE_MODULES 筛子，缺依赖的模块会被自动滤掉。
    Path(r"E:\AI\corpora\python-packages"),
]
SKIP_PARTS = {"site-packages", ".venv", "test", "tests", "idlelib", "lib2to3", "tkinter"}

# 允许的 stdlib 模块。
#
# ⚠️ 设计说明（想过一轮才想清楚）：**这份名单不提供安全性**。
# 真正要防的是"模型生成的代码"，而模型可以在函数体里写 `import os; ...` ——
# 名单管不到它。实际的隔离手段是**独立子进程 + 超时 + 干净环境变量 + 临时目录**
# （见 run_doctest 与 --help 末尾的诚实说明）。
# 名单在这里的作用是**限制任务来源在纯 stdlib 的纯函数上**，让判据可靠、任务可复现，
# 而不是当沙箱用。所以它应该**宽**，不该窄——
# 之前按"整个文件的 import 必须是安全子集"过滤，产出率只有 12%，是个错误的严格。
SAFE_MODULES = {
    "math", "cmath", "string", "textwrap", "re", "json", "itertools", "functools",
    "operator", "collections", "heapq", "bisect", "copy", "decimal", "fractions",
    "statistics", "random", "datetime", "difflib", "unicodedata", "binascii",
    "base64", "hashlib", "hmac", "secrets", "uuid", "zlib", "csv", "io", "pathlib",
    "numbers", "typing", "dataclasses", "enum", "abc", "contextlib", "warnings",
    "types", "urllib", "html", "os", "sys", "posixpath", "ntpath", "glob",
    "fnmatch", "shutil", "tempfile", "time", "calendar", "locale", "codecs",
    "struct", "array", "weakref", "gc", "inspect", "traceback", "pprint",
    "reprlib", "keyword", "token", "tokenize", "ast", "dis", "graphlib",
    "queue", "threading", "subprocess_placeholder_removed", "socket_placeholder_removed",
    "email", "http", "mimetypes", "quopri", "uu", "gzip", "bz2", "lzma",
    "tarfile", "zipfile", "configparser", "argparse", "getopt", "logging",
    "unittest", "doctest", "pdb", "profile", "timeit", "trace", "tracemalloc",
}
# 明确排除：会真的联网/起进程的模块名（原文件 import 它们时整份跳过）
RISKY_MODULES = {"socket", "subprocess", "ctypes", "multiprocessing", "asyncio",
                 "urllib.request", "http.client", "ftplib", "smtplib", "telnetlib",
                 "ssl", "importlib", "pickle", "shelve", "dbm", "sqlite3"}


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 区间 —— n 小时比正态近似可靠得多。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


#: `_importable` 的缓存 —— 每个模块名只探一次，避免几万次 find_spec
_IMPORTABLE_CACHE: dict[str, bool] = {}


def _importable(name: str) -> bool:
    """这个顶层模块名在当前源集合下能否解析？

    **取代静态白名单。** 白名单的问题是"新加的源永远进不来" ——
    依赖名只要不在名单上就被拒，于是候选池怎么加包都不涨（实测卡在 225）。
    判据应该是**能不能真的导入**，而不是"名字在不在我列的清单里"。
    """
    if name in _IMPORTABLE_CACHE:
        return _IMPORTABLE_CACHE[name]
    ok = name in sys.stdlib_module_names
    if not ok:
        for root in SRC_ROOTS:
            r = Path(root)
            if (r / name).is_dir() or (r / f"{name}.py").is_file():
                ok = True
                break
    if not ok:      # 装进 venv 的（torch / numpy / transformers 等）也算
        try:
            import importlib.util
            ok = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError, ModuleNotFoundError, AttributeError):
            ok = False
    _IMPORTABLE_CACHE[name] = ok
    return ok


def _imports_of(tree: ast.Module) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def extract_candidates(limit_files: int = 0) -> list[dict[str, Any]]:
    """扫源文件，抽出「顶层函数 + doctest」的候选任务。"""
    out: list[dict[str, Any]] = []
    files = []
    for r in SRC_ROOTS:
        if not r.is_dir():
            continue
        for p in r.rglob("*.py"):
            if set(p.parts) & SKIP_PARTS:
                continue
            try:
                if p.stat().st_size > 300_000:
                    continue
            except OSError:
                continue
            files.append(p)
    files.sort()
    if limit_files:
        files = files[:limit_files]

    for p in files:
        try:
            src = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if ">>>" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        mods = _imports_of(tree)
        if mods & RISKY_MODULES:
            continue
        # ⚠️ 曾经这里写的是 `if not (mods <= SAFE_MODULES): continue` ——
        # 一个**静态白名单**。它的本意是"只留依赖封闭的模块，standalone doctest 才可能通过"，
        # 但副作用是：**新下载的包因为依赖不在名单里，一个都进不来**
        # （池子卡在 225 个候选，怎么加源都不涨）。
        #
        # 改成**动态可导入性检查**：子进程已经把 `corpora/python-packages` 追加到
        # `sys.path`，所以判据应该是"这些 import 到底能不能解析"，而不是"名字在不在名单上"。
        # 名单只保留给"明确危险"的模块（RISKY_MODULES）。
        if not all(_importable(m) for m in mods):
            continue   # 附加筛子：选出依赖封闭的模块，standalone doctest 才可能通过
        # 注意：不再要求 `mods <= SAFE_MODULES`。
        # 那个"白名单子集"判据把产出率压到了 3.5%（6,517 个带 doctest 的函数只剩 225 个候选），
        # 而它换来的安全性是**虚假的**——真正的威胁是模型生成的代码，白名单管不到。
        # 现在只排除明确的危险模块，安全性靠 run_doctest 的进程隔离。

        # 模块上下文：同一文件里**所有**顶层定义（含目标函数本身）。
        #
        # 为什么含目标函数也没关系：`build_task` 把目标函数放在**最后**，
        # Python 里后定义覆盖前定义，所以模型填的那份会生效。
        # 这样就不必按节点做差集，少一类出错的可能。
        #
        # 为什么需要上下文：绝大多数 doctest 会引用同模块的兄弟函数/常量，
        # 只把目标函数单独拿出来跑它们必然失败——
        # 这是任务池小的**主因**，不是任务本身不合格。
        ctx_parts = []
        for other in tree.body:
            if isinstance(other, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                 ast.Assign, ast.AnnAssign)):
                seg = ast.get_source_segment(src, other)
                if seg and len(seg) < 4000:
                    ctx_parts.append(seg)
        context = "\n\n".join(ctx_parts)

        for node in tree.body:  # 只要顶层函数，不要方法
            if not isinstance(node, ast.FunctionDef):
                continue
            doc = ast.get_docstring(node, clean=False)
            if not doc or ">>>" not in doc:
                continue
            if node.decorator_list:
                continue  # 装饰器会改变语义，先跳过
            try:
                sig_src = ast.unparse(node.args)
            except Exception:
                continue
            out.append({
                "file": str(p),
                "name": node.name,
                "lineno": node.lineno,
                "args": sig_src,
                "doc": doc,
                "imports": sorted(mods),
                "context": context,
                "src": ast.get_source_segment(src, node) or "",
            })
    return out


def docstring_block(doc: str) -> str:
    """渲染成**真实 Python 源码的样子**——**只用于给模型看的提示词**。

    ⚠️ 不能用 `repr(doc)` 做提示词：repr 产生一整行带 `\\n` 转义的字符串，
    真实源码不长这样。11M 代码 LM 见到那种模式后只会重复 `''' ''' '''`，
    **3/3 任务全部 syntax_error**。提示词必须落在训练分布内。
    """
    q = '"""' if '"""' not in doc else "'''"
    if not doc.endswith("\n"):
        doc += "\n"
    return f"    {q}{doc}    {q}\n"


def _build_module(cand: dict[str, Any], body: str) -> str:
    """判据模块：**docstring 用 repr**，必须精确保真。

    ⚠️ 这里和提示词用**不同格式**，是刻意的，而且是被实测逼出来的：
    doctest 会**去掉源行尾空白**，而有些任务的例子恰好依赖尾空格
    （`difflib.IS_LINE_JUNK` 的 `'  #   '` 就是），换成字面量后判据直接坏掉——
    自检从 20/20 掉到 **17/20**。而 repr 写法能逐字保存 docstring，doctest 才不会误判。
    **提示词求"在分布内"，判据求"精确保真"，两个目标不同。**
    """
    b = textwrap.indent(body.rstrip() or "pass", "    ")
    return (f"def {cand['name']}({cand['args']}):\n"
            f"    {cand['doc']!r}\n{b}\n")


def build_task(cand: dict[str, Any], model_body: str) -> str:
    """构造可执行模块：签名 + 真实 docstring + 函数体（供判据校验与判别力门槛）。"""
    ctx = cand.get("context", "")
    head = textwrap.dedent(ctx).strip()
    return (head + "\n\n\n" if head else "") + _build_module(cand, model_body)


def run_doctest(module_src: str, imports: list[str], timeout: float) -> dict[str, Any]:
    """在独立子进程里跑 doctest。返回 {ok, reason, executed}。"""
    # ⚠️ 任务源码来自 `corpora/python-packages`，而那是 `--target` 装的，
    # **不在 venv 的 site-packages 里**。子进程用 `-I`（隔离模式，忽略 PYTHONPATH 与环境变量），
    # 所以必须**在代码里显式把源目录插进 sys.path**，否则 `import sympy` 一律 ModuleNotFoundError
    # —— 文件明明在磁盘上，却一个都跑不通。
    # 实测踩过：把包下到 corpora 后候选池从 225 涨到几千，入池却仍是 0，就是这个原因。
    # ⚠️ **必须 `append`，不能 `insert(0, …)`**。
    # 插到最前面会让 `corpora/python-packages` 里的模块**遮蔽标准库** ——
    # 实测报 `AssertionError: SRE module mismatch`（`re` 被从两个位置各导入一次）。
    # 追加到末尾：标准库优先，只有标准库里没有的模块才从 corpora 取，正是我们要的。
    roots = "".join(
        f"_sys.path.append({str(r)!r})\n"
        for r in SRC_ROOTS if Path(r).is_dir()
    )
    prelude = (
        "import sys as _sys\n"
        + roots
        + "".join(f"import {m}\n" for m in imports)
    )
    harness = (
        prelude
        + module_src
        + "\n\nimport doctest as _dt\n"
        "_r = _dt.testmod(_sys.modules[__name__], verbose=False)\n"
        "_sys.exit(0 if _r.failed == 0 and _r.attempted > 0 else 1)\n"
    )
    with tempfile.TemporaryDirectory(prefix="cb2_") as td:
        f = Path(td) / "candidate.py"
        f.write_text(harness, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            # 🔴 **固定哈希种子**。默认下 `PYTHONHASHSEED` 是随机的，
            # 于是**任何依赖 set/dict 遍历顺序的 doctest 跨进程结果就不确定**。
            # 实测：`sympy/solvers.py:denoms` 连跑三次得到「正确体 过不过不过」、
            # 「空体 不过过不过」—— **两个方向都在翻转**，这种任务既不能用来测模型，
            # 也解释了为什么自检会报「正确体 39/40、空体 1/40」。
            "PYTHONHASHSEED": "0",
        }
        t0 = time.perf_counter()
        try:
            # ⚠️ 用 `-s`（不加载用户 site）而**不是 `-I`**：
            # `-I` 隐含 `-E`，会**忽略所有 PYTHON* 环境变量** —— 那恰好把上面
            # `PYTHONHASHSEED=0` 屏蔽掉，不确定性就修不掉。
            # 隔离性由「显式构造的干净 env」保证，不靠 `-E`。
            #
            # **也不要加 `-S`** —— 那会连主 site-packages 一起关掉，
            # 于是 `import numpy` 全部 ModuleNotFoundError。
            # 实测踩过：加 `-S` 时候选 2,571 个、入池 0 个，全是这个原因。
            pr = subprocess.run(
                [PY, "-s", str(f)],
                capture_output=True, text=True, timeout=timeout,
                cwd=td, env=env, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "reason": "timeout", "executed": True,
                    "secs": time.perf_counter() - t0}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"spawn:{type(exc).__name__}", "executed": False,
                    "secs": time.perf_counter() - t0}
    secs = time.perf_counter() - t0
    if pr.returncode == 0:
        return {"ok": True, "reason": "pass", "executed": True, "secs": secs}
    err = (pr.stderr or "").strip().splitlines()
    tail = err[-1][:200] if err else "fail"
    kind = "fail"
    if "SyntaxError" in (pr.stderr or "") or "IndentationError" in (pr.stderr or ""):
        kind = "syntax_error"
    elif "NameError" in (pr.stderr or ""):
        kind = "name_error"
    elif "Error" in (pr.stderr or ""):
        kind = "runtime_error"
    return {"ok": False, "reason": kind, "detail": tail, "executed": True, "secs": secs}


def cmd_build(args: argparse.Namespace) -> int:
    print("[cb2] 扫描源码…", flush=True)
    cands = extract_candidates(args.limit_files)
    # 默认**不带**模块上下文（实测带上下文产出率反而低：同样 225 个候选，
    # 带上下文入池 4 个、不带 18 个——上下文引入的导入期报错多于它修复的引用问题）
    if not args.context:
        for c in cands:
            c["context"] = ""
    print(f"[cb2] 候选（顶层函数 + doctest + 依赖封闭）: {len(cands):,}", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(cands)
    if args.scan:
        cands = cands[: args.scan]

    good: list[dict[str, Any]] = []
    per_file: dict[str, int] = {}
    t0 = time.perf_counter()
    debug_shown = 0
    for i, c in enumerate(cands, 1):
        # 关键门槛：用**原函数体 + 模块上下文**跑它自己的 doctest，不通过就丢掉。
        # 必须走 build_task 同一条路径（不能图省事直接丢 c["src"] 进去）——
        # 否则验证的和评测用的不是同一个东西，判据就不可信。
        # ⚠️ **带上下文与不带上下文各有胜负，所以两个都试、取通过的那个**：
        # 实测同样 225 个候选，带上下文入池 4 个、不带 18 个 ——
        # 但"带"能救回那些引用兄弟函数的任务，"不带"能躲开上下文自身的导入期报错。
        # 二选一必然丢掉一半，**兜底重试才是正确策略**。
        c = dict(c)
        r = run_doctest(build_task(c, _body_of(c["src"])), c["imports"], args.timeout)
        if not r["ok"] and c.get("context"):
            c_nc = dict(c)
            c_nc["context"] = ""
            r2 = run_doctest(build_task(c_nc, _body_of(c["src"])), c["imports"],
                             args.timeout)
            if r2["ok"]:
                c, r = c_nc, r2
        elif r["ok"] and not c.get("context"):
            # 不带上下文就过了 —— 顺手记下"它其实不需要上下文"，
            # 这样后面统计产出率时能分清是哪条路径救回来的。
            r["no_context_ok"] = True
        fname = Path(c["file"]).name
        if r["ok"]:
            # 🔴 第二道门槛：**判别力**。只有"正确体过"远远不够——
            # 实测发现有些 doctest 只写 `>>> f(x)` 而**没有期望输出**，
            # 这种 doctest 只检查"不抛异常"，于是 `pass` 也能过
            # （`utils.py:reverse_dict` 就是）。这类任务测不出能力，只会把分数灌水。
            # 判据：拿**平凡体**跑一遍，**必须不过**。
            # ⚠️ 必须测多个平凡体，不能只测一个：
            # `return 0` 和 `pass` 返回的值不同（后者是 None），
            # 实测有任务的 doctest 恰好期望 None，只测 `return 0` 会漏掉它。
            TRIVIAL = ("return 0", "pass", "return None")
            triv_ok = [b for b in TRIVIAL
                       if run_doctest(build_task(c, b), c["imports"], args.timeout)["ok"]]
            if triv_ok:
                if args.debug and debug_shown < 6:
                    debug_shown += 1
                    print(f"  [debug] 丢弃（无判别力，{triv_ok[0]!r} 也能过）: "
                          f"{fname}:{c['name']}", flush=True)
                continue
            # 每个模块设上限：否则任务集会过度集中在某一个包上
            # （实测第一版 30 个任务里 21 个来自 more_itertools，对基准是缺陷）
            if per_file.get(fname, 0) >= args.per_file:
                continue
            per_file[fname] = per_file.get(fname, 0) + 1
            good.append({
                "file": c["file"], "name": c["name"], "args": c["args"],
                "doc": c["doc"], "imports": c["imports"],
                "context": c.get("context", ""),
                "oracle_body": _body_of(c["src"]),
            })
        elif args.debug and debug_shown < 3:
            debug_shown += 1
            print(f"  [debug] {Path(c['file']).name}:{c['name']} -> {r['reason']}  "
                  f"{(r.get('detail') or '')[:120]}", flush=True)
        if i % 25 == 0:
            print(f"  … {i}/{len(cands)}  已入池 {len(good)}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        if len(good) >= args.want:
            break

    print(f"[cb2] 判据校验通过（原函数跑通自己的 doctest）: **{len(good)}** 个任务", flush=True)
    if good:
        files = {}
        for g in good:
            files[Path(g["file"]).name] = files.get(Path(g["file"]).name, 0) + 1
        top = sorted(files.items(), key=lambda kv: -kv[1])[:8]
        print("  来源分布: " + ", ".join(f"{k}×{v}" for k, v in top))
    out = VERIFY / (args.out or "cb2-tasks.json")
    out.write_text(json.dumps({
        "seed": args.seed, "timeout": args.timeout, "n_tasks": len(good), "tasks": good,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[cb2] 已写 {out.relative_to(ROOT)}")
    return 0 if good else 1


def _body_of(func_src: str) -> str:
    """从函数源码里取出**函数体**，并**跳过 docstring**。

    ⚠️ 必须跳过 docstring：`_build_module` 会自己渲染一遍 docstring，
    若这里再带一份就会重复。实测踩过——自检从 20/20 掉到 **17/20**，
    正是这一步把 3 个任务的判据弄坏了。**自检就是为抓这个而存在的。**
    """
    try:
        src = textwrap.dedent(func_src)
        tree = ast.parse(src)
        fn = tree.body[0]
        assert isinstance(fn, ast.FunctionDef)
        body = fn.body
        # 第一个语句是裸字符串表达式 => 它是 docstring，跳过
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            body = body[1:]
        if not body:
            return "pass"
        return textwrap.dedent("\n".join(src.splitlines()[body[0].lineno - 1:]))
    except Exception:
        return "pass"


def cmd_selftest(args: argparse.Namespace) -> int:
    """验证仪器本身：**同一套 build_task + run_doctest 路径**下，
    正确函数体必须通过、错误函数体必须不通过。

    这一步不能省。一个"什么都过"或"什么都不"的基准，数字再漂亮也没有意义——
    本项目已经因为"没验证仪器"栽过好几次（机会基线漂移、检查点步数错标）。
    """
    p = ROOT / args.tasks
    if not p.exists():
        print(f"[selftest] 找不到任务集 {p}，先跑 --build")
        return 2
    tasks = json.loads(p.read_text(encoding="utf-8"))["tasks"]
    if not tasks:
        print("[selftest] 任务集为空")
        return 2
    rng = random.Random(args.seed)
    pick = rng.sample(tasks, min(args.samples, len(tasks)))
    print(f"[selftest] 在 {len(pick)} 个任务上验证仪器（同一路径 build_task + run_doctest）")
    print(f"{'任务':<44}{'正确体':>8}{'空体':>8}{'乱写体':>9}")
    ok_right = ok_empty = ok_junk = 0
    for t in pick:
        label = f"{Path(t['file']).name}:{t['name']}"[:43]
        r_right = run_doctest(build_task(t, t["oracle_body"]), t["imports"], args.timeout)
        r_empty = run_doctest(build_task(t, "pass"), t["imports"], args.timeout)
        r_junk = run_doctest(build_task(t, "return 0"), t["imports"], args.timeout)
        ok_right += r_right["ok"]
        ok_empty += r_empty["ok"]
        ok_junk += r_junk["ok"]
        print(f"{label:<44}{'过' if r_right['ok'] else '不过':>8}"
              f"{'过' if r_empty['ok'] else '不过':>8}"
              f"{'过' if r_junk['ok'] else '不过':>9}")
    n = len(pick)
    print()
    print(f"  正确体通过 {ok_right}/{n}   <- **必须 = n**，否则仪器坏了（会低估模型）")
    print(f"  空体通过   {ok_empty}/{n}   <- **必须 = 0**，否则判据太松（会高估模型）")
    print(f"  乱写体通过 {ok_junk}/{n}   <- 同上")
    verdict = (ok_right == n and ok_empty == 0 and ok_junk == 0)
    print()
    print(f"  => 仪器{'**可用**' if verdict else '**有问题，不能用来测模型**'}")
    return 0 if verdict else 1


def _fit_prompt(gen: Any, prefix: str, suffix: str, budget: int) -> Optional[tuple[str, str]]:
    """把 FIM 提示词压进模型的上下文预算内。

    ⚠️ 实测踩过：FIM 模式把「签名+docstring」和「函数之后的代码」拼在一起，
    **7/20 的任务合计超过 block_size=512**。超出时模型输入被裁，输出退化成
    `', 'g', '` 这种几个字符的垃圾，然后全部 syntax_error——
    **看起来像"模型不会"，其实是喂超了。**

    压缩顺序（尽量保住判据所需的信息）：
      1. 原样；
      2. 丢掉后缀（后缀只是上下文，判据不需要它）；
      3. 逐步截短 docstring 的后半（doctest 通常在开头，先保住）；
      4. 仍放不下 → 返回 None，调用方跳过并计入报告。
    """
    enc = getattr(gen, "_encode", None)
    if enc is None:                       # 拿不到分词器就按字符粗略估
        return (prefix, suffix) if len(prefix) + len(suffix) <= budget * 3 else None

    if len(enc(prefix + suffix)) <= budget:
        return prefix, suffix

    head, _, body = prefix.partition("\n")
    lines = body.splitlines(keepends=True)

    # 先截**提示词里的** docstring 给后缀腾地方。
    # ⚠️ 这里能放心截，是因为**判据用的是 `_build_module` 里的完整 docstring**，
    # 提示词只影响模型看到什么。实测踩过：先砍后缀，FIM 就退化成纯续写
    # （模型输出 `:\n        raise TypeError(...)` 这种不守缩进的片段），
    # 而**后缀正是让"填空"区别于"续写"的东西**。
    for keep in (10, 6, 3, 1):
        cand_prefix = head + "\n" + "".join(lines[:keep])
        if len(enc(cand_prefix + suffix)) <= budget:
            return cand_prefix, suffix

    if len(enc(prefix)) <= budget:
        return prefix, ""                 # 实在放不下后缀，退回无后缀
    for keep in (10, 6, 3, 1):
        cand_prefix = head + "\n" + "".join(lines[:keep])
        if len(enc(cand_prefix)) <= budget:
            return cand_prefix, ""
    return None


def _tail_after(cand: dict[str, Any], limit: int = 400) -> str:
    """取目标函数**之后**的源码，作为 FIM 的后缀上下文。

    加了后缀，任务才真正落在训练分布内：训练时 PSM 格式
    `[fb] prefix [fh] suffix [fe] middle` 里 suffix 是有内容的，
    只给前缀会让模型按"续写"而非"填空"来处理。
    """
    try:
        src = Path(cand["file"]).read_text(encoding="utf-8")
        tree = ast.parse(src)
        lines = src.splitlines(keepends=True)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == cand["name"] \
                    and node.lineno == cand.get("lineno"):
                return "".join(lines[node.end_lineno:])[:limit]
        # 退而求其次：按名字找
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == cand["name"]:
                return "".join(lines[node.end_lineno:])[:limit]
    except Exception:
        pass
    return ""


def cmd_run(args: argparse.Namespace) -> int:
    """把模型接到任务集上：**第一个"真的能跑对多少"的数字**。"""
    import torch

    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(ROOT))
    from codemind.generate import load_runner
    from codemind.model import SamplingParams

    p = ROOT / args.tasks
    if not p.exists():
        print(f"[cb2] 找不到任务集 {p}，先跑 --build")
        return 2
    tasks = json.loads(p.read_text(encoding="utf-8"))["tasks"]
    rng = random.Random(args.seed)
    pick = rng.sample(tasks, min(args.samples, len(tasks)))

    run_path = Path(args.run)
    if not run_path.is_absolute():
        run_path = ROOT / run_path
    gen = load_runner(run_path, device="cpu",
                      tokenizer_path=Path(args.tokenizer) if args.tokenizer else None)
    params = SamplingParams(temperature=args.temperature, top_k=args.top_k,
                            repetition_penalty=args.rep, seed=args.seed)

    print(f"[cb2] 检查点 : {run_path.name}")
    print(f"[cb2] 任务   : {len(pick)} 个（池 {len(tasks)}）  超时 {args.timeout}s")
    print(f"[cb2] 解码   : temp {args.temperature} / top_k {args.top_k} / rep {args.rep}")
    print("-" * 78)

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, t in enumerate(pick, 1):
        # 提示词有两种形态，**用途不同、别混**：
        #  * spec —— 只给签名 + docstring，让模型"照规格实现"。**不在训练分布内**，
        #            实测 0/20，因为它同时测了"指令跟随"，而模型没学过这个。
        #  * fim  —— 用 FIM 专用标记挖洞（`<|fim_begin|>…<|fim_hole|>…<|fim_end|>`），
        #            并给出函数之后的代码作为后缀。**这才是训练时的形状**，
        #            所以它测的是"能不能按规格把这段填对"，不被指令跟随能力污染。
        prefix = f"def {t['name']}({t['args']}):\n{docstring_block(t['doc'])}"
        if args.mode == "fim":
            fitted = _fit_prompt(gen, prefix, _tail_after(t), args.max_prompt_tokens)
            if fitted is None:
                rows.append({"file": t["file"], "name": t["name"], "ok": False,
                             "reason": "prompt_too_long", "gen_chars": 0})
                if not args.quiet:
                    print(f"  - {i:>3}/{len(pick)}  {Path(t['file']).name}:{t['name']}"
                          f"  [跳过：提示词超出预算]", flush=True)
                continue
            prefix, suffix = fitted
            try:
                out = gen.infill(prefix, suffix, max_new_tokens=args.max_new_tokens,
                                 params=params)
            except Exception as exc:  # noqa: BLE001
                out, err = "", f"gen:{type(exc).__name__}"
            else:
                err = ""
            # ⚠️ 模型输出必须先**去掉公共缩进**再当作函数体。
            # FIM 返回的是原始文本续写，缩进层级不受控——
            # 实测它给出 `:\n        raise TypeError(...)`（0 格起、8 格续），
            # 直接再缩 4 格就变 12 格 → syntax_error。
            # 只做 dedent，**不丢任何内容**：缩进是拼接产物，不是模型要答的东西。
            body_text = textwrap.dedent(out).strip("\n")
            # ⚠️ **判据必须重新用 `_build_module` 组装，不能复用给模型看的 prefix**。
            # 提示词里的 docstring 是字面量（为落在训练分布内），
            # 而 doctest 需要 repr 才逐字保真（见 `_build_module` 的说明）。
            module_src = _build_module(t, body_text)
        else:
            try:
                out = gen.complete(prefix, max_new_tokens=args.max_new_tokens, params=params)
            except Exception as exc:  # noqa: BLE001
                out, err = "", f"gen:{type(exc).__name__}"
            else:
                err = ""
            module_src = prefix + out
        r = run_doctest(module_src, t["imports"], args.timeout) if not err else {
            "ok": False, "reason": err}
        rows.append({"file": t["file"], "name": t["name"], "ok": bool(r["ok"]),
                     "reason": r.get("reason", ""), "gen_chars": len(out)})
        if not args.quiet:
            mark = "✓" if r["ok"] else "✗"
            print(f"  {mark} {i:>3}/{len(pick)}  {Path(t['file']).name}:{t['name']}"
                  f"  [{r.get('reason','')}]", flush=True)

    n = len(rows)
    k = sum(r["ok"] for r in rows)
    lo, hi = _wilson(k, n)
    secs = time.perf_counter() - t0

    print("-" * 78)
    print(f"  **执行通过率（pass@1）: {k}/{n} = {k/n*100:.1f}%**"
          f"   95% Wilson 区间 [{lo*100:.1f}%, {hi*100:.1f}%]")
    print(f"  耗时 {secs:.0f}s（{secs/max(n,1):.1f}s/任务）")
    print()
    print("  失败原因分布:")
    reasons: dict[str, int] = {}
    for r in rows:
        if not r["ok"]:
            reasons[r["reason"] or "?"] = reasons.get(r["reason"] or "?", 0) + 1
    for reason, cnt in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:<18} {cnt:>3}")
    print()
    print("  对照：**机会基线尚未测**。不测机会基线的通过率无法解读——")
    print("        一个乱写函数体也能蒙对若干任务，必须量出那个底。")

    out_json = VERIFY / (args.out or f"cb2-{args.tag}.json")
    out_json.write_text(json.dumps({
        "checkpoint": str(run_path), "samples": n, "passed": k,
        "pass_at_1": k / n if n else 0.0, "wilson": [lo, hi],
        "params": {"temperature": args.temperature, "top_k": args.top_k,
                   "repetition_penalty": args.rep, "seed": args.seed},
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[cb2] 已写 {out_json.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="cb-2 执行评测环：用 doctest 真的跑一遍模型生成的代码",
        epilog=textwrap.dedent("""
        ⚠️ 执行安全的诚实说明：
          本工具在**独立子进程 + 超时 + 干净环境变量 + 临时目录**里跑模型生成的代码，
          并且只从少数纯 stdlib 模块里取任务。这**不是真正的沙箱**：
          Windows 上没有 seccomp/namespace，恶意代码理论上仍可触碰文件系统。
          对本项目（自训 11M 模型，无联网、无对抗动机）这个风险可接受；
          **若要测第三方模型或联网模型，必须先换成真正的隔离环境。**
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--build", action="store_true", help="只建任务集并校验判据")
    ap.add_argument("--run", help="要评测的检查点路径")
    ap.add_argument("--tasks", default="runs/_verify/cb2-tasks.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--want", type=int, default=200, help="最多收集多少个任务")
    ap.add_argument("--scan", type=int, default=0, help="最多扫多少个候选（0=全部）")
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--samples", type=int, default=100, help="评测多少个任务")
    ap.add_argument("--timeout", type=float, default=8.0, help="单个任务的执行超时（秒）")
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-prompt-tokens", type=int, default=480,
                    help="提示词 token 预算；超出则先砍后缀、再截 docstring")
    ap.add_argument("--mode", default="fim", choices=["fim","spec"],
                    help="fim=用 FIM 标记挖洞（训练分布内，默认）；spec=只给规格（测指令跟随）")
    ap.add_argument("--tag", default="cb2")
    ap.add_argument("--tokenizer", default="runs/corpus2/tokenizer.json")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--rep", type=float, default=1.2)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--debug", action="store_true", help="打印前几个失败任务的原因")
    ap.add_argument("--per-file", type=int, default=6, help="每个模块最多取几个任务（保证多样性）")
    ap.add_argument("--context", action="store_true",
                    help="把同模块其他顶层定义也带进执行环境。**默认关**——实测量过："
                         "同样 225 个候选，带上下文入池 4 个、不带 18 个。"
                         "上下文引入的新失败（导入期报错）多于它修复的（引用兄弟函数）。")
    ap.add_argument("--selftest", action="store_true",
                    help="验证仪器本身：正确函数体必须过、错误函数体必须不过")
    args = ap.parse_args()

    VERIFY.mkdir(parents=True, exist_ok=True)
    if args.run:
        return cmd_run(args)
    if args.selftest:
        return cmd_selftest(args)
    if args.build:
        return cmd_build(args)
    print("目前只实现了 --build（任务集构建）。评测部分见下一步。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
