"""把 REPORT.md 里的关键数字与原始证据文件对账（可重复执行）。

结构：metrics[arm][metric] = {"mean":…, "stderr":…}；基线在 baselines；难度扫描在 by_hole。
"""
import json
import sys
from pathlib import Path

V = Path(r"E:\AI\runs\_verify")
REPORT = Path(r"E:\AI\REPORT.md").read_text(encoding="utf-8")

fails: list[str] = []
checks = 0


def load(name):
    return json.loads((V / name).read_text(encoding="utf-8"))


ARMED = {"greedy", "product", "first", "syntax", "oracle"}


def mean(d, arm, metric):
    """兼容两种格式：早期单臂 metrics[metric]，和双臂 metrics[arm][metric]。"""
    m = d["metrics"]
    if arm in m:                       # 双臂格式
        return m[arm][metric]["mean"]
    return m[metric]["mean"]           # 单臂格式，忽略 arm


def check(label, rendered):
    global checks
    checks += 1
    if rendered not in REPORT:
        fails.append(f"{label}: 报告里找不到 {rendered!r}")


def check_pct(label, x, nds=(1, 2)):
    """百分比：报告可能用 1 位或 2 位小数，两者都算通过。空格不算差异。"""
    global checks
    checks += 1
    flat = REPORT.replace(" ", "")
    for nd in nds:
        if f"{x*100:.{nd}f}%" in flat:
            return
    fails.append(f"{label}: 报告里找不到 {x*100:.1f}% 或 {x*100:.2f}%（证据值 {x}）")


def check_delta(label, pp, nds=(1, 2)):
    """增量：报告用 Unicode 减号 U+2212，且小数位数可能不同。"""
    global checks
    checks += 1
    flat = REPORT.replace(" ", "").replace("\u2212", "-")
    for nd in nds:
        if f"{pp:+.{nd}f}pp" in flat:
            return
    fails.append(f"{label}: 报告里找不到 {pp:+.1f}pp 或 {pp:+.2f}pp（证据值 {pp}）")


def pct(x, nd=2):
    return f"{x*100:.{nd}f}%"


# ------------------------------------------------------------------ 主表
old = load("cb1-old-corpus.json")            # 旧模型：单臂（产物是 greedy 还是 product？看结构）
n5 = load("cb1-step500-2arm.json")
n10 = load("cb1-step1000-2arm.json")

print("=== 证据文件里的臂 ===")
for name, d in (("old-corpus", old), ("step500", n5), ("step1000", n10)):
    print(f"  {name}: arms={list(d['metrics'].keys())}  baselines={ {k: round(v,4) for k,v in d.get('baselines',{}).items()} }")

old_arm = "product" if "product" in old["metrics"] else "-"
print(f"\n  旧模型取臂 = {old_arm}")

print("\n=== cb-1 主表 ===")
rows = [
    ("旧模型 edit_sim", old, old_arm, "edit_sim"),
    ("旧模型 syntax_ok", old, old_arm, "syntax_ok"),
    ("新500 edit_sim", n5, "product", "edit_sim"),
    ("新500 syntax_ok", n5, "product", "syntax_ok"),
    ("新1000 edit_sim", n10, "product", "edit_sim"),
    ("新1000 syntax_ok", n10, "product", "syntax_ok"),
]
for label, d, arm, metric in rows:
    print(f"  {label:<20} {pct(mean(d, arm, metric))}")

check_pct("旧 edit_sim", mean(old, old_arm, "edit_sim"))
check_pct("旧 syntax_ok", mean(old, old_arm, "syntax_ok"))
check_pct("新500 edit_sim", mean(n5, "product", "edit_sim"))
check_pct("新1000 edit_sim", mean(n10, "product", "edit_sim"))

print("\n  机会基线：")
for label, d in (("旧模型那批", old), ("新模型那批", n10)):
    b = d.get("baselines") or {"chance_edit_sim": d["metrics"]["chance_control"]["mean"], "chance_syntax_ok": d["metrics"]["syntax_control"]["mean"]}
    print(f"    {label}: edit_sim={pct(b['chance_edit_sim'])} syntax_ok={pct(b['chance_syntax_ok'])}")
    check_pct(f"{label} chance_edit_sim", b["chance_edit_sim"])
    check_pct(f"{label} chance_syntax_ok", b["chance_syntax_ok"])

print("\n  exact / first_line（报告称都是 0%）:")
for label, d in (("旧", old), ("新1000", n10)):
    for m in ("exact", "first_line"):
        v = mean(d, "product", m)
        print(f"    {label} {m} = {v:.4f}")
        if v == 0:
            check(f"{label} {m} 为 0", "0%")

# ------------------------------------------------------------------ 上下文消融
print("\n=== 上下文消融 ===")
c500, c1500 = load("ctx-step500.json"), load("ctx-step1500.json")
for tag, d in (("500", c500), ("1500", c1500)):
    line = (f"  step {tag}: hole_ppl={d['hole_ppl']:.2f} fim_end={d['fim_end_loss']:.4f} "
            f"gain={d['context_gain_nats']:+.4f} ratio=x{d['context_gain_ratio']:.3f}")
    print(line)
    check(f"{tag} hole_ppl", f"{d['hole_ppl']:.2f}")
    check(f"{tag} fim_end", f"{d['fim_end_loss']:.4f}")
    check(f"{tag} gain", f"{d['context_gain_nats']:+.4f}")
    check(f"{tag} ratio", f"{d['context_gain_ratio']:.3f}")

# ------------------------------------------------------------------ 难度扫描
print("\n=== 难度扫描（by_hole）===")
sw = load("cb1-sweep-step1000.json")
report_flat = REPORT.replace(" ", "")
for hole_key, block in sw["by_hole"].items():
    hole = int(hole_key)
    a, b = block["arms"]["product"], block["baselines"]
    first = a["first_tok"]["mean"]
    es, ce = a["edit_sim"]["mean"], b["chance_edit_sim"]
    rep8 = a["rep8"]["mean"]
    margin = (es - ce) * 100
    print(f"  hole {hole:>3}: first_tok={pct(first)} (机会 {pct(b['chance_first_tok'])})  "
          f"edit_sim={pct(es)}/{pct(ce)} 余量={margin:+.1f}pp  rep8={pct(rep8)}")
    check_pct(f"hole {hole} first_tok", first)
    check_pct(f"hole {hole} edit_sim", es)
    check_pct(f"hole {hole} chance", ce)
    check_pct(f"hole {hole} rep8", rep8)
    check(f"hole {hole} 机会 first_tok", pct(b["chance_first_tok"]))
    checks += 1
    if f"{margin:+.1f}pp" not in report_flat:
        fails.append(f"hole {hole} 余量: 报告里找不到 {margin:+.1f}pp")

# ------------------------------------------------------------------ 脚手架
print("\n=== 脚手架（nbest=4 三种选择策略）===")
sc = load("cb1-scaffolding.json")
nb = sc["nbest"]
for arm in ("first", "syntax", "oracle"):
    a = nb[arm]
    print(f"  {arm:<8} edit_sim={pct(a['edit_sim']['mean'])}  syntax_ok={pct(a['syntax_ok']['mean'])}  "
          f"rep8={pct(a['rep8']['mean'], 1)}")
    check_pct(f"nbest {arm} edit_sim", a["edit_sim"]["mean"])
    check_pct(f"nbest {arm} syntax_ok", a["syntax_ok"]["mean"])

d_es = (nb["syntax"]["edit_sim"]["mean"] - nb["first"]["edit_sim"]["mean"]) * 100
d_sy = (nb["syntax"]["syntax_ok"]["mean"] - nb["first"]["syntax_ok"]["mean"]) * 100
d_or = (nb["oracle"]["edit_sim"]["mean"] - nb["first"]["edit_sim"]["mean"]) * 100
print(f"  syntax 相对 first: edit_sim {d_es:+.2f}pp  syntax_ok {d_sy:+.2f}pp")
print(f"  oracle 相对 first: edit_sim {d_or:+.2f}pp")
check_delta("syntax Δedit_sim", d_es)
check_delta("syntax Δsyntax_ok", d_sy)
check_delta("oracle Δedit_sim", d_or)

# ------------------------------------------------------------------ 曲线
print("\n=== 曲线文件 ===")
for line in (V / "cb1-curve.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    print(f"  cb1      step {r['steps']:>5}  product {pct(r['product_edit_sim'])}")
for line in (V / "context-curve.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    print(f"  context  step {r['steps']:>5}  hole_ppl {r['hole_ppl']:.2f}")

print(f"\n=== 对账：{checks - len(fails)}/{checks} 项通过 ===")
for f in fails:
    print("  ✗", f)
if not fails:
    print("  报告中的关键数字全部可追溯到 runs/_verify/ 下的证据文件")
sys.exit(1 if fails else 0)
