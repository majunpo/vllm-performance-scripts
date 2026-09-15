#!/usr/bin/env python3
"""Build a cross-platform comparison report from two single-step analyses.

Consumes the text output of `analyze_trace.py` (Intel XPU / unitrace) and
`analyze_nv_trace.py` (NVIDIA / torch profiler) and emits the markdown tables of
a `xpu-vs-nv-comparison.md`: overall summary, per-category prefill and decode
tables with kernel names, and a decomposition of where the gap comes from.

Only the "single step" sections are used. The aggregate tables divide by the
step count and yield fractional counts because the first and last step are
clipped, which makes them useless for a row-by-row comparison.
"""

import argparse
import re
import sys
from collections import OrderedDict

# the two platforms tag the same work differently; fold them onto one label
ALIASES = {
    "Attention(FMHA)": "Attention",
    "Attention(FA)": "Attention",
    "Quantize(fp8/fp4)": "Quantize",
    "Quantize(fp8)": "Quantize",
}

RE_SECTION = re.compile(
    r"^--- single (PREFILL|DECODE) step(?: #(\d+))? \(exact counts, "
    r"([\d.]+) ms")
RE_ROW = re.compile(r"^(?P<cat>\S.*?)\s+(?P<cnt>\d+)\s+(?P<ms>[\d.]+)\s+"
                    r"(?P<pct>[\d.]+)%\s*$")
RE_KERNEL = re.compile(r"^\s+(?P<cnt>\d+)\s+(?P<ms>[\d.]+)\s+"
                       r"(?P<pct>[\d.]+)%\s+(?P<name>.+?)\s*$")

# per-layer kernel counts that must agree between prefill and decode
PER_LAYER = ("Dense-GEMM", "Quantize", "Norm/RoPE", "Activation(SiLU)",
             "KVCache-Write")
# decode-only helpers that legitimately inflate a category's count
RE_AUX = re.compile(r"split_k_reduce|ReduceSplitK|splitkv_combine")


def core_count(entry):
    """Kernel count with decode-only split-K helpers removed."""
    if not entry:
        return None
    aux = sum(c for c, _, n in entry["kernels"] if RE_AUX.search(n))
    return entry["cnt"] - aux


def parse(path):
    """-> {'PREFILL'|'DECODE': {'total': ms, 'cats': OrderedDict}}"""
    out = {}
    cur = None
    cat = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = RE_SECTION.match(line)
            if m:
                cur = {"total": float(m.group(3)), "cats": OrderedDict(),
                       "step": m.group(2)}
                out[m.group(1)] = cur
                cat = None
                continue
            if cur is None or not line.strip():
                continue
            if line.startswith("---") or line.startswith("category"):
                continue
            k = RE_KERNEL.match(line)
            if k and cat:
                cur["cats"][cat]["kernels"].append(
                    (int(k.group("cnt")), float(k.group("ms")),
                     k.group("name")))
                continue
            r = RE_ROW.match(line)
            if r:
                name = r.group("cat").strip()
                if name == "TOTAL":
                    cur = None
                    cat = None
                    continue
                cat = ALIASES.get(name, name)
                cur["cats"][cat] = {"cnt": int(r.group("cnt")),
                                    "ms": float(r.group("ms")),
                                    "pct": float(r.group("pct")),
                                    "kernels": []}
    if not out:
        sys.exit(f"no 'single ... step' section found in {path} -- rerun the "
                 f"analyzer with --batch set")
    return out


def check(name, data):
    """Warn when a step window looks clipped."""
    msgs = []
    if "PREFILL" in data and "DECODE" in data:
        for c in PER_LAYER:
            p = core_count(data["PREFILL"]["cats"].get(c))
            d = core_count(data["DECODE"]["cats"].get(c))
            if p is not None and d is not None and p != d:
                msgs.append(f"  {name}: {c} prefill={p} decode={d} (should match)")
    for ph, v in data.items():
        s = sum(c["ms"] for c in v["cats"].values())
        if abs(s - v["total"]) > 0.01 * max(v["total"], 1e-9):
            msgs.append(f"  {name}/{ph}: category sum {s:.3f} != total {v['total']:.3f}")
    return msgs


def cell(entry, top=3, width=60):
    if not entry or not entry["kernels"]:
        return "—"
    ks = sorted(entry["kernels"], key=lambda x: -x[1])[:top]
    parts = [f"`{n[:width]}` ×{c}" for c, _, n in ks]
    extra = len(entry["kernels"]) - len(ks)
    return " + ".join(parts) + (f" + {extra} more" if extra > 0 else "")


def ratio(a, b):
    if not b:
        return "∞" if a else "—"
    return f"{a / b:.2f}×"


def table(fa, fb, na, nb, phase, unit, top, width):
    a = fa.get(phase)
    b = fb.get(phase)
    if not a or not b:
        return []
    cats = list(a["cats"]) + [c for c in b["cats"] if c not in a["cats"]]
    cats.sort(key=lambda c: -a["cats"].get(c, {}).get("ms", 0.0))
    out = [
        f"| 类别 | {na} cnt | {na} {unit} | {na} % | {na} kernel | "
        f"{nb} cnt | {nb} {unit} | {nb} % | {nb} kernel | {nb} 加速 |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for c in cats:
        x = a["cats"].get(c)
        y = b["cats"].get(c)
        xm = x["ms"] if x else 0.0
        ym = y["ms"] if y else 0.0
        out.append(
            f"| **{c}** | {x['cnt'] if x else 0} | "
            f"{xm:.3f} | {x['pct'] if x else 0:.2f} | {cell(x, top, width)} | "
            f"{y['cnt'] if y else 0} | {ym:.3f} | {y['pct'] if y else 0:.2f} | "
            f"{cell(y, top, width)} | {ratio(xm, ym)} |")
    ta, tb = a["total"], b["total"]
    ca = sum(v["cnt"] for v in a["cats"].values())
    cb = sum(v["cnt"] for v in b["cats"].values())
    out.append(f"| **TOTAL** | **{ca}** | **{ta:.3f}** | **100.00** | | "
               f"**{cb}** | **{tb:.3f}** | **100.00** | | **{ratio(ta, tb)}** |")
    return out


def gap_table(fa, fb, na, nb, phase):
    a, b = fa.get(phase), fb.get(phase)
    if not a or not b:
        return []
    delta = a["total"] - b["total"]
    if abs(delta) < 1e-9:
        return []
    rows = []
    for c in set(a["cats"]) | set(b["cats"]):
        d = a["cats"].get(c, {}).get("ms", 0.0) - b["cats"].get(c, {}).get("ms", 0.0)
        rows.append((c, d, d / delta * 100))
    rows.sort(key=lambda r: -abs(r[1]))
    lead = na if delta < 0 else nb
    out = [f"{phase} 总差距 **{abs(delta):.3f} ms**（{lead} 更快），来源分解：", "",
           f"> 差值 = {na} − {nb}。正值表示 {nb} 在该类别更快，负值表示 {na} 更快。",
           f"> 单项占比可超过 100%，说明它被反向的类别抵消了一部分。", "",
           "| 类别 | 差值 ms | 占总差距 % |", "|---|---:|---:|"]
    for c, d, p in rows:
        out.append(f"| {c} | {d:+.3f} | {p:+.1f} % |")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xpu", required=True, help="analyze_trace.py output (.txt)")
    p.add_argument("--nv", required=True, help="analyze_nv_trace.py output (.txt)")
    p.add_argument("--xpu-name", default="XPU")
    p.add_argument("--nv-name", default="NV")
    p.add_argument("--prompt-len", type=int, default=0, help="for prefill tok/s")
    p.add_argument("--batch", type=int, default=1, help="for decode tok/s")
    p.add_argument("--top-kernels", type=int, default=2,
                   help="kernels listed per category cell")
    p.add_argument("--kernel-width", type=int, default=52)
    p.add_argument("-o", "--output", default="xpu-vs-nv-comparison.md")
    args = p.parse_args()

    fx, fn = parse(args.xpu), parse(args.nv)
    na, nb = args.xpu_name, args.nv_name

    warn = check(na, fx) + check(nb, fn)
    if warn:
        print("WARNING: step windows look clipped, numbers are not comparable:")
        print("\n".join(warn))
        print("Fix the analyzer run before trusting this report.\n")

    L = [f"# 推理性能对比：{na} vs {nb}", "",
         "> 数据取自两边的**单步精确统计**窗口（以 sampler kernel 为边界，恰好一次",
         "> 完整 forward），计数为整数。聚合表的均值除法会产生小数，不能用于逐行对比。",
         "", "## 1. 总体", "",
         f"| 指标 | {na} | {nb} | {nb} 加速 |", "|---|---:|---:|---:|"]

    for phase, label in (("PREFILL", "prefill"), ("DECODE", "decode 每步")):
        if phase in fx and phase in fn:
            x, y = fx[phase]["total"], fn[phase]["total"]
            L.append(f"| {label} | {x:.3f} ms | {y:.3f} ms | {ratio(x, y)} |")
    if "PREFILL" in fx and "PREFILL" in fn and args.prompt_len:
        x = args.prompt_len / fx["PREFILL"]["total"] * 1000
        y = args.prompt_len / fn["PREFILL"]["total"] * 1000
        L.append(f"| prefill 吞吐 | {x:.0f} tok/s | {y:.0f} tok/s | {ratio(y, x)} |")
    if "DECODE" in fx and "DECODE" in fn:
        x = args.batch / fx["DECODE"]["total"] * 1000
        y = args.batch / fn["DECODE"]["total"] * 1000
        L.append(f"| decode 吞吐 | {x:.2f} tok/s | {y:.2f} tok/s | {ratio(y, x)} |")

    for phase, title, unit in (("PREFILL", "2. PREFILL 单步分类对照", "ms"),
                               ("DECODE", "3. DECODE 单步分类对照", "ms/step")):
        rows = table(fx, fn, na, nb, phase, unit, args.top_kernels,
                     args.kernel_width)
        if rows:
            L += ["", f"## {title}", "", *rows]

    L += ["", "## 4. gap 分解", ""]
    for phase in ("PREFILL", "DECODE"):
        g = gap_table(fx, fn, na, nb, phase)
        if g:
            L += [*g, ""]

    L += ["## 5. 待补充（需人工分析）", "",
          "- [ ] §0 对比基准：设备规格、量化格式、GEMM 后端、图执行模式、负载",
          "- [ ] 逐 shape GEMM 对照（prefill TFLOPS / decode GB/s）"
          " —— 来自 analyze_gemm_shapes.py 与 analyze_nv_trace.py 的 GEMM 表",
          "- [ ] attention 对照（TFLOPS、有效带宽、reduce/combine 占比）",
          "- [ ] 每个 gap 的根因判断：软件问题 vs 硬件规格差异",
          "- [ ] 优化优先级表，每条带量化收益估算",
          "- [ ] 计量注意事项（见 qwen3-perf-analysis/references/pitfalls.md）", ""]

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote {args.output} ({len(L)} lines)")
    if not warn:
        print("sanity checks passed: per-layer counts match across phases")


if __name__ == "__main__":
    main()
