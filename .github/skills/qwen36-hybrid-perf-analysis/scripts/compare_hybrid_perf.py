#!/usr/bin/env python3
"""Build an XPU-vs-NV comparison report from two hybrid-model analyses.

`xpu-nv-perf-comparison/scripts/compare_perf.py` only parses the *dense*
`analyze_trace.py` format, so it cannot read the hybrid analyses.  This script
consumes the text output this skill already produces:

    XPU : analyze_hybrid_trace.txt  (+ analyze_hybrid_gemm.txt for the GEMM tables)
    NV  : analyze_nv_hybrid_trace.txt   (categories and GEMM are in one file)

and emits the mechanical half of the report -- overview, per-category
prefill/decode tables with a gap column, per-linear GEMM tables, per-layer-type
tables and the gap decomposition -- plus a checklist of the parts that need a
human.  It reads only the .txt files, so it needs no dependencies and runs
anywhere in under a second.

Nothing here is model- or device-specific: category and linear names are taken
from the inputs, so a new model or a new GPU needs no code change.
"""

import argparse
import os
import re
import sys
from datetime import date

# Categories that mean the same thing but are named differently by the two
# analyzers.  Extend this when a new backend shows up; everything not listed
# is compared under its own name and simply shows 0 on the side that lacks it.
ALIASES = {
    "FullAttn-SplitK-Reduce": "FullAttn-Reduce/Merge",
    "FullAttn-MergeStates": "FullAttn-Reduce/Merge",
}

# Rows worth calling out separately in the gap table even when they are small.
NOISE_MS_PREFILL = 1.0
NOISE_MS_DECODE = 0.02


def canon(cat):
    return ALIASES.get(cat, cat)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def sections(text):
    """Split an analyzer output into {section header: [body lines]}."""
    out, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^---\s+(.*?)\s+---\s*$", line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, [])
        elif cur is not None:
            out[cur].append(line)
    return out


def find_section(secs, *patterns):
    for pat in patterns:
        for k in secs:
            if re.search(pat, k):
                return secs[k], k
    return None, None


def num(s):
    try:
        return float(s.replace("%", ""))
    except (ValueError, AttributeError):
        return None


def parse_categories(lines, per_step=False):
    """{category: (cnt, ms)} from a '--- ... categories ...' section.

    Handles both layouts: the XPU decode table carries ms/step in a later
    column, the NV one is already per step.
    """
    rows, total = {}, None
    for ln in lines:
        if not ln.strip() or ln.lstrip().startswith(("category", "-", "!")):
            continue
        m = re.match(r"^(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)%"
                     r"(?:\s+([\d.]+)\s+([\d.]+))?\s*$", ln)
        if not m:
            continue
        cat, cnt, ms, _pct, ms_step, cnt_step = m.groups()
        cat = cat.strip()
        if per_step and ms_step is not None:
            cnt, ms = cnt_step, ms_step
        rec = (float(cnt), float(ms))
        if cat == "TOTAL":
            total = rec
        else:
            rows[canon(cat)] = tuple(a + b for a, b in
                                     zip(rows.get(canon(cat), (0.0, 0.0)), rec))
    return rows, total


def parse_layer_types(lines):
    rows = {}
    for ln in lines:
        m = re.match(r"^(gdn|full|head|TOTAL)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+"
                     r"([\d.]+)%\s+([\d.]+)", ln)
        if m:
            k, layers, kern, ms, _pct, per = m.groups()
            rows[k] = (int(layers), int(kern), float(ms), float(per))
    return rows


def parse_gemm_xpu(lines, decode):
    """XPU analyze_hybrid_gemm.py per-shape table -> {linear: dict}."""
    rows = {}
    for ln in lines:
        col = ln.split()
        if len(col) < 8 or not re.match(r"^[a-z_]+$", col[0]):
            continue
        try:
            if decode:                 # linear K N calls med min MB_w GB/s GB/s@min ms
                rows[col[0]] = dict(K=int(col[1]), N=int(col[2]), calls=int(col[3]),
                                    us=float(col[4]), metric=float(col[7]),
                                    ms=float(col[9]))
            else:                      # linear M K N calls med GFLOP MB TFLOPS GB/s ms
                rows[col[0]] = dict(K=int(col[2]), N=int(col[3]), calls=int(col[4]),
                                    us=float(col[5]), metric=float(col[8]),
                                    ms=float(col[10]))
        except (ValueError, IndexError):
            continue
    return rows


def parse_gemm_nv(lines, decode):
    """NV per-linear table -> {linear: dict}. Column 2 is the backend."""
    rows = {}
    for ln in lines:
        col = ln.split()
        if len(col) < 11 or not re.match(r"^[a-z_]+$", col[0]):
            continue
        try:
            rows[col[0]] = dict(backend=col[1], K=int(col[2]), N=int(col[3]),
                                calls=int(col[4]), us=float(col[5]),
                                metric=float(col[9] if decode else col[8]),
                                ms=float(col[10]))
        except (ValueError, IndexError):
            continue
    return rows


def parse_backends(lines):
    rows = {}
    for ln in lines:
        m = re.match(r"^(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%", ln)
        if m:
            rows[m.group(1)] = (int(m.group(2)), float(m.group(3)),
                                float(m.group(4)))
    return rows


def parse_xpu(trace_txt, gemm_txt):
    t = open(trace_txt, encoding="utf-8", errors="replace").read()
    d = {"categories": {}, "layers": {}, "gemm": {}, "aggregate": {}}

    m = re.search(r"^prefill\s+:\s+([\d.]+) ms(?:\s+\((\d+) tokens\))?", t, re.M)
    d["prefill_ms"] = float(m.group(1)) if m else None
    d["tokens"] = int(m.group(2)) if m and m.group(2) else None
    m = re.search(r"^decode\s+:\s+[\d.]+ ms over (\d+) steps -> ([\d.]+) ms/step"
                  r"\s+\(([\d.]+) tok/s at bs=(\S+?)\)", t, re.M)
    if m:
        d["decode_steps"] = int(m.group(1))
        d["decode_ms"] = float(m.group(2))
        d["decode_toks"] = float(m.group(3))
        d["batch"] = m.group(4)
    m = re.search(r"^workload: prompt=(\d+) tokens, batch=(\d+)", t, re.M)
    if m:
        d["tokens"], d["batch"] = int(m.group(1)), m.group(2)

    secs = sections(t)
    body, _ = find_section(secs, r"^PREFILL categories")
    d["categories"]["prefill"], d["total_prefill"] = parse_categories(body or [])
    body, _ = find_section(secs, r"^DECODE categories")
    d["categories"]["decode"], d["total_decode"] = parse_categories(body or [],
                                                                    per_step=True)
    body, _ = find_section(secs, r"^PREFILL: time by layer type")
    d["layers"]["prefill"] = parse_layer_types(body or [])
    body, _ = find_section(secs, r"^DECODE step #\d+: time by layer type")
    d["layers"]["decode"] = parse_layer_types(body or [])

    if gemm_txt and os.path.exists(gemm_txt):
        g = open(gemm_txt, encoding="utf-8", errors="replace").read()
        gs = sections(g)
        body, _ = find_section(gs, r"^PREFILL Dense-GEMM per shape")
        d["gemm"]["prefill"] = parse_gemm_xpu(body or [], decode=False)
        body, _ = find_section(gs, r"^DECODE Dense-GEMM per shape")
        d["gemm"]["decode"] = parse_gemm_xpu(body or [], decode=True)
        # the aggregate line is inflated when the walker merged in_proj*; the
        # script prints a corrected line right below it -- always prefer it
        m = re.search(r"^!! corrected\s+:.*?([\d.]+) TFLOPS", g, re.M)
        d["aggregate"]["prefill_tflops"] = float(m.group(1)) if m else None
        d["aggregate"]["prefill_corrected"] = bool(m)
        if not m:
            m = re.search(r"aggregate.*?\n\s+([\d.]+) TFLOPS", g)
            d["aggregate"]["prefill_tflops"] = float(m.group(1)) if m else None
        m = re.search(r"aggregate\s+: [\d.]+ ms.*?\n\s+([\d.]+) GB/s", g)
        d["aggregate"]["decode_gbs"] = float(m.group(1)) if m else None
        m = re.search(r"^MXFP4 linears\s+: ([\d.]+) ms, ([\d.]+) GiB -> ([\d.]+) GB/s",
                      g, re.M)
        if m:
            d["aggregate"]["decode_quant_ms"] = float(m.group(1))
            d["aggregate"]["decode_quant_gib"] = float(m.group(2))
            d["aggregate"]["decode_quant_gbs"] = float(m.group(3))
    return d


def parse_nv(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    d = {"categories": {}, "layers": {}, "gemm": {}, "aggregate": {}}

    m = re.search(r"^device\s*:\s*(.+?)\s*$", t, re.M)
    d["device"] = m.group(1) if m else None
    m = re.search(r"^load\s*:\s*prompt=(\d+), batch=(\d+)", t, re.M)
    if m:
        d["tokens"], d["batch"] = int(m.group(1)), m.group(2)
    m = re.search(r"^prefill\s+: window\s+([\d.]+) ms\s+kernels\s+([\d.]+) ms"
                  r"\s+\(([\d.]+)% busy\)\s+->\s+(\d+) tok/s", t, re.M)
    if m:
        d["prefill_wall"] = float(m.group(1))
        d["prefill_ms"] = float(m.group(2))
        d["prefill_busy"] = float(m.group(3))
    m = re.search(r"^decode / step\s+: window\s+([\d.]+) ms\s+kernels\s+([\d.]+) ms"
                  r"\s+\(([\d.]+)% busy\)\s+->\s+([\d.]+) tok/s", t, re.M)
    if m:
        d["decode_wall"] = float(m.group(1))
        d["decode_ms"] = float(m.group(2))
        d["decode_busy"] = float(m.group(3))
        d["decode_toks"] = float(m.group(4))
    m = re.search(r"^end-to-end\s+: ([\d.]+) ms", t, re.M)
    d["e2e_ms"] = float(m.group(1)) if m else None

    secs = sections(t)
    body, _ = find_section(secs, r"^PREFILL categories")
    d["categories"]["prefill"], d["total_prefill"] = parse_categories(body or [])
    body, _ = find_section(secs, r"^DECODE categories")
    d["categories"]["decode"], d["total_decode"] = parse_categories(body or [])
    body, _ = find_section(secs, r"^PREFILL: time by layer type")
    d["layers"]["prefill"] = parse_layer_types(body or [])
    body, _ = find_section(secs, r"^DECODE step #\d+: time by layer type")
    d["layers"]["decode"] = parse_layer_types(body or [])
    body, _ = find_section(secs, r"^PREFILL Dense-GEMM per linear")
    d["gemm"]["prefill"] = parse_gemm_nv(body or [], decode=False)
    d["backends_prefill"] = parse_backends(body or [])
    body, _ = find_section(secs, r"^DECODE Dense-GEMM per linear")
    d["gemm"]["decode"] = parse_gemm_nv(body or [], decode=True)
    d["backends_decode"] = parse_backends(body or [])
    for phase, pat in (("prefill_tflops", r"aggregate.*?\n\s+([\d.]+) TFLOPS"),
                       ("decode_gbs", r"aggregate\s+: [\d.]+ ms.*?\n\s+([\d.]+) GB/s")):
        m = re.search(pat, t)
        d["aggregate"][phase] = float(m.group(1)) if m else None
    return d


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def f(v, spec=".3f", dash="—"):
    return dash if v is None else format(v, spec)


def ratio(a, b):
    return None if not a or not b else a / b


def pct(a, b):
    """a/b in %, or None when the denominator is missing or zero."""
    return None if not b else a / b * 100


def category_table(xa, nb, xtot, ntot, noise, out):
    """Side-by-side categories ordered by |difference|.

    Carries both a ratio and a difference column on purpose: the ratio says
    *which operator is implemented worst*, the difference says *where the time
    actually is*.  Ranking optimisations on the ratio alone is a classic
    mistake -- the worst ratio is usually a category too small to matter.
    """
    gap = (xtot[1] if xtot else sum(v[1] for v in xa.values())) - \
          (ntot[1] if ntot else sum(v[1] for v in nb.values()))
    keys = sorted(set(xa) | set(nb),
                  key=lambda k: -abs(xa.get(k, (0, 0))[1] - nb.get(k, (0, 0))[1]))
    out.append("| 类别 | A cnt | A ms | A % | B cnt | B ms | B % | "
               "**B 领先** | **差值 ms** | 占 gap |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    xsum = xtot[1] if xtot else 1.0
    nsum = ntot[1] if ntot else 1.0
    small_x = small_n = small_d = 0.0
    small_cx = small_cn = 0

    def rat(mx, mn):
        if not mn:
            return "**B 无此开销**" if mx else "—"
        if not mx:
            return "**A 无此开销**"
        r = mx / mn
        return f"{r:.2f}×" + (f"（A 快 {1/r:.2f}×）" if r < 1 else "")

    for k in keys:
        cx, mx = xa.get(k, (0.0, 0.0))
        cn, mn = nb.get(k, (0.0, 0.0))
        dd = mx - mn
        if abs(dd) < noise:
            small_x += mx; small_n += mn; small_d += dd
            small_cx += cx; small_cn += cn
            continue
        bold = "**" if abs(dd) > 0.25 * abs(gap) else ""
        out.append(f"| {bold}{k}{bold} | {cx:.0f} | {bold}{mx:.3f}{bold} | "
                   f"{f(pct(mx, xsum), '.2f')} % | {cn:.0f} | "
                   f"{bold}{mn:.3f}{bold} | {f(pct(mn, nsum), '.2f')} % | "
                   f"{bold}{rat(mx, mn)}{bold} | "
                   f"{bold}{dd:+.3f}{bold} | {f(pct(dd, gap), '.1f')} % |")
    if small_cx or small_cn:
        out.append(f"| 其余小项合计 | {small_cx:.0f} | {small_x:.3f} | "
                   f"{f(pct(small_x, xsum), '.2f')} % | {small_cn:.0f} | "
                   f"{small_n:.3f} | {f(pct(small_n, nsum), '.2f')} % | "
                   f"{rat(small_x, small_n)} | "
                   f"{small_d:+.3f} | {f(pct(small_d, gap), '.1f')} % |")
    if xtot and ntot:
        out.append(f"| **TOTAL** | **{xtot[0]:.0f}** | **{xtot[1]:.3f}** | 100 % | "
                   f"**{ntot[0]:.0f}** | **{ntot[1]:.3f}** | 100 % | "
                   f"**{rat(xtot[1], ntot[1])}** | "
                   f"**{gap:+.3f}** | 100 % |")
    out += ["",
            "> **比值和差值要一起看**：比值指向\"**哪个算子实现最差**\"，"
            "差值指向\"**时间实际花在哪**\"。比值最差的那一项往往小到不值得优化，"
            "而占 gap 最大的那一项比值可能只是中游 —— **只按比值排优先级会排错**。"]
    return gap


def gemm_table(xg, ng, unit, out):
    """Per-linear side by side. Both TFLOPS and GB/s are higher-is-better.

    Flags the row the XPU layer-walker inflates: when `in_proj_ba` is present
    on one side only, the other side folded it into `in_proj_qkvz`, doubling
    that row's metric (the aggregate is corrected, the row is not).
    """
    merged = "in_proj_ba" in ng and "in_proj_ba" not in xg
    order = sorted(set(xg) | set(ng),
                   key=lambda k: -(xg.get(k, {}).get("ms", 0.0)))
    out.append(f"| linear | K | N | A µs/层 | **A {unit}** | B µs/层 | "
               f"**B {unit}** | B 后端 | B 领先 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for k in order:
        x, n = xg.get(k, {}), ng.get(k, {})
        r = ratio(n.get("metric"), x.get("metric"))
        flag = " ⚠" if merged and k == "in_proj_qkvz" else ""
        out.append(
            f"| `{k}`{flag} | {x.get('K') or n.get('K') or '—'} | "
            f"{x.get('N') or n.get('N') or '—'} | {f(x.get('us'), '.1f')} | "
            f"{f(x.get('metric'), '.1f')} | {f(n.get('us'), '.1f')} | "
            f"**{f(n.get('metric'), '.1f')}** | {n.get('backend', '—')} | "
            f"{f(r, '.2f')}× |")
    if merged:
        out += ["",
                "> ⚠ A 侧的 layer-walker 把 `in_proj_qkvz` 和 `in_proj_ba` 合并成了"
                "一个桶，该行的 A 指标**虚高一倍**，`in_proj_ba` 行缺失。"
                "正确的逐 linear 拆分见 `dump_kernels.py` 的 duration-clustering 段；"
                "下面的**合计已是修正值**。"]
    only_a, only_b = sorted(set(xg) - set(ng)), sorted(set(ng) - set(xg))
    if (only_a or only_b) and not merged:
        out += ["",
                f"> 两侧的 linear 拆分不同（仅 A：{', '.join(only_a) or '无'}；"
                f"仅 B：{', '.join(only_b) or '无'}）—— checkpoint 把同一组权重存成了"
                "不同数量的 linear。这些行不能逐行比，只能看合计。"]


def layer_table(x, n, out):
    out.append("| 层类型 | A 层数 | A ms | A ms/层 | B ms | B ms/层 | B 领先 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for k in ("gdn", "full", "head"):
        if k not in x and k not in n:
            continue
        xl = x.get(k, (0, 0, None, None))
        nl = n.get(k, (0, 0, None, None))
        out.append(f"| {k} | {xl[0]} | {f(xl[2])} | {f(xl[3], '.4f')} | "
                   f"{f(nl[2])} | {f(nl[3], '.4f')} | "
                   f"{f(ratio(xl[3], nl[3]), '.2f')}× |")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xpu", required=True, help="analyze_hybrid_trace.txt")
    p.add_argument("--xpu-gemm", help="analyze_hybrid_gemm.txt (per-shape tables)")
    p.add_argument("--nv", required=True, help="analyze_nv_hybrid_trace.txt")
    p.add_argument("--xpu-name", default="Intel XPU")
    p.add_argument("--nv-name", default="NVIDIA")
    p.add_argument("--title", default="Qwen3.6-27B 推理性能对比")
    p.add_argument("-o", "--output", required=True,
                   help="suffix it with the XPU trace timestamp; never overwrite "
                        "an existing comparison")
    a = p.parse_args()

    if os.path.exists(a.output):
        sys.exit(f"{a.output} already exists -- comparisons are kept so two runs "
                 f"can be diffed; pick a new -o (suffix it with the XPU trace "
                 f"timestamp)")
    if not a.xpu_gemm:
        cand = os.path.join(os.path.dirname(a.xpu), "analyze_hybrid_gemm.txt")
        a.xpu_gemm = cand if os.path.exists(cand) else None

    x = parse_xpu(a.xpu, a.xpu_gemm)
    n = parse_nv(a.nv)

    if x.get("tokens") and n.get("tokens") and x["tokens"] != n["tokens"]:
        print(f"WARNING: prompt length differs ({x['tokens']} vs {n['tokens']}) "
              f"-- attention and prefill numbers are NOT comparable", file=sys.stderr)
    if x.get("batch") and n.get("batch") and x["batch"] != n["batch"]:
        print(f"WARNING: batch differs ({x['batch']} vs {n['batch']}) "
              f"-- per-step numbers are NOT comparable", file=sys.stderr)
    if not x["gemm"]:
        print("WARNING: no XPU GEMM tables (pass --xpu-gemm); "
              "the per-shape sections will be empty", file=sys.stderr)
    if x["aggregate"].get("prefill_corrected"):
        print("NOTE: using the corrected XPU prefill TFLOPS (the walker had "
              "merged in_proj_qkvz + in_proj_ba)", file=sys.stderr)

    A, B = a.xpu_name, a.nv_name
    o = [f"# {a.title}：{A} vs {B}", "",
         f"**日期**：{date.today().isoformat()}",
         f"**生成方式**：`qwen36-hybrid-perf-analysis/scripts/compare_hybrid_perf.py`"
         f"（§1–§4 自动生成，§0/§5/§6 需人工补充）", "",
         f"- A = **{A}** ← `{os.path.relpath(a.xpu)}`",
         f"- B = **{B}** ← `{os.path.relpath(a.nv)}`"
         + (f"（{n['device']}）" if n.get("device") else ""), "", "---", ""]

    o += ["## 0. 对比基准（人工填写）", "",
          "| | " + A + " | " + B + " |", "|---|---|---|",
          "| 设备 | <型号 / 架构 / 计算单元数 / 显存> | "
          + (n.get("device") or "<型号>") + " |",
          "| Trace | <文件名>（unitrace） | <文件名>（PyTorch Profiler） |",
          "| 模型 | <layers / hidden / inter / vocab / GDN 与 full 的拆分> | 同左 |",
          "| 权重+激活量化 | <格式，group size，W?A?> | <格式，是否 MIXED_PRECISION> |",
          "| GEMM 后端 | <库 + kernel 家族> | "
          + (" / ".join(n.get("backends_decode", {})) or "<后端>") + " |",
          "| **Full attention** | <库 + kernel（prefill / decode / reduce）> | "
          "<库 + kernel> |",
          "| **GDN（linear attention）** | <库 + kernel；prefill 的 chunk 路径与 decode 的 "
          "recurrent 路径是**两套不同的 kernel**，要分开写> | <库 + kernel> |",
          "| KV cache 精度 | <dtype> | <dtype> |",
          "| 图执行 | <XPU-Graph / CUDA Graph / eager> | <...> |",
          f"| 负载 | in{x.get('tokens','?')} / bs{x.get('batch','?')} / tp? / out? | "
          f"in{n.get('tokens','?')} / bs{n.get('batch','?')} / tp? / out? |", "",
          "**可比性**：<逐条说明哪些条件相同、哪些不同、不同项的影响>", "",
          "**数据来源**：两边都取单步窗口（sampler kernel / annotation 为边界，"
          "恰好一次完整 forward），计数为整数。", ""]

    # ---- 1 overview -----------------------------------------------------
    pr = ratio(x.get("prefill_ms"), n.get("prefill_ms"))
    dr = ratio(x.get("decode_ms"), n.get("decode_ms"))
    o += ["## 1. 总体", "",
          f"| 指标 | **{A}** | **{B}** | B 领先 |", "|---|---:|---:|---:|",
          f"| prefill kernel 时间 | **{f(x.get('prefill_ms'))} ms** | "
          f"**{f(n.get('prefill_ms'))} ms** | **{f(pr, '.2f')}×** |"]
    if x.get("tokens"):
        o.append(f"| prefill 吞吐（kernel 口径） | "
                 f"{f(x['tokens']/x['prefill_ms']*1000, '.0f')} tok/s | "
                 f"{f(n['tokens']/n['prefill_ms']*1000, '.0f')} tok/s | "
                 f"{f(pr, '.2f')}× |")
    o += [f"| decode kernel 时间/step | **{f(x.get('decode_ms'))} ms** | "
          f"**{f(n.get('decode_ms'))} ms** | **{f(dr, '.2f')}×** |",
          f"| decode 吞吐（bs={x.get('batch','?')}） | "
          f"{f(x.get('decode_toks'), '.2f')} tok/s | "
          f"{f(n.get('decode_toks'), '.2f')} tok/s | {f(dr, '.2f')}× |",
          f"| GPU busy（decode） | <reflow 后填写> | "
          f"{f(n.get('decode_busy'), '.1f')} % | |",
          f"| 每步 kernel 数（decode） | {f(x['total_decode'][0] if x.get('total_decode') else None, '.0f')} | "
          f"{f(n['total_decode'][0] if n.get('total_decode') else None, '.0f')} | |", "",
          "> **一句话结论**：<谁赢在哪个阶段，主因是什么>", ""]

    # ---- 2 prefill ------------------------------------------------------
    o += ["## 2. PREFILL 单步分类对照", "",
          f"A = {A}，B = {B}。按 |差值| 排序。", ""]
    pgap = category_table(x["categories"]["prefill"], n["categories"]["prefill"],
                          x.get("total_prefill"), n.get("total_prefill"),
                          NOISE_MS_PREFILL, o)
    o += ["", "### 2.1 prefill GEMM 逐 shape（TFLOPS）", ""]
    if x["gemm"].get("prefill"):
        gemm_table(x["gemm"]["prefill"], n["gemm"]["prefill"], "TFLOPS", o)
        o += ["", f"**合计**：A **{f(x['aggregate'].get('prefill_tflops'), '.1f')} TFLOPS**"
              + ("（已按时长聚类修正 `in_proj_qkvz`/`in_proj_ba` 的合并）"
                 if x["aggregate"].get("prefill_corrected") else "")
              + f" vs B **{f(n['aggregate'].get('prefill_tflops'), '.1f')} TFLOPS**"
              + f" → **{f(ratio(n['aggregate'].get('prefill_tflops'), x['aggregate'].get('prefill_tflops')), '.2f')}×**", ""]
    else:
        o += ["<未提供 --xpu-gemm，无法生成>", ""]
    if n.get("backends_prefill"):
        o += [f"**{B} 的 GEMM 后端分布（prefill）**：", "",
              "| 后端 | launches | ms | 占 GEMM |", "|---|---:|---:|---:|"]
        for k, (c, ms, share) in n["backends_prefill"].items():
            o.append(f"| `{k}` | {c} | {ms:.3f} | {share:.2f} % |")
        o += ["", "> 后端不同 = **MAC 精度不同**。逐 shape 的差距大小通常由这张表解释，"
              "对比前务必确认两边到底在比什么精度。", ""]
    o += ["### 2.2 prefill attention（人工填写）", "",
          "| | A | B |", "|---|---|---|", "| kernel | | |",
          "| calls / µs per call | | |", "| TFLOPS | | |",
          "| **相对本平台自身 GEMM 算力** | | |", "",
          "### 2.3 GDN chunk prefill（人工填写）", "",
          "先写两边的**实现来源**（库 + 源码路径）—— 混合模型两侧的 GDN 往往是"
          "**两套独立实现**（例如手写 SYCL vs Triton FLA），kernel 无法一一对应，"
          "只能按**功能**分组（UT-transform 求逆 / W、U 投影 / chunk 输出 / "
          "causal conv1d / 状态递推）对照，并标出比值最大的一项。", ""]

    # ---- 3 decode -------------------------------------------------------
    o += ["## 3. DECODE 单步分类对照", ""]
    dgap = category_table(x["categories"]["decode"], n["categories"]["decode"],
                          x.get("total_decode"), n.get("total_decode"),
                          NOISE_MS_DECODE, o)
    o += ["", "### 3.1 decode GEMM 逐 shape（GB/s）", ""]
    if x["gemm"].get("decode"):
        gemm_table(x["gemm"]["decode"], n["gemm"]["decode"], "GB/s", o)
        xq = x["aggregate"]
        o += ["", f"**合计**：A **{f(xq.get('decode_gbs'), '.1f')} GB/s**"
              + (f"（仅量化 linear **{f(xq.get('decode_quant_gbs'), '.1f')} GB/s**）"
                 if xq.get("decode_quant_gbs") else "")
              + f" vs B **{f(n['aggregate'].get('decode_gbs'), '.1f')} GB/s**"
              + f" → **{f(ratio(n['aggregate'].get('decode_gbs'), xq.get('decode_gbs')), '.2f')}×**", ""]
        # self-reference: each platform against its own best shape
        o += ["**各平台的自我参照**（判断是硬件上限还是软件问题的关键）：", "",
              "| | A | B |", "|---|---:|---:|"]
        best_x = max((v["metric"] for v in x["gemm"]["decode"].values()), default=None)
        best_n = max((v["metric"] for v in n["gemm"]["decode"].values()), default=None)
        # a BF16 XPU trace has no separate quantised-linear line
        xq_gbs = x["aggregate"].get("decode_quant_gbs") or x["aggregate"].get("decode_gbs")
        nq_gbs = n["aggregate"].get("decode_gbs")
        o += [f"| 本平台最佳 shape 带宽 | {f(best_x, '.1f')} GB/s | {f(best_n, '.1f')} GB/s |",
              f"| 主干 linear 聚合实测（有量化则取量化 linear） | {f(xq_gbs, '.1f')} GB/s | "
              f"{f(nq_gbs, '.1f')} GB/s |",
              f"| **达成率** | **{f(pct(xq_gbs, best_x), '.1f')} %** | "
              f"**{f(pct(nq_gbs, best_n), '.1f')} %** |", "",
              "> 达成率低的一方，其量化 GEMM 是**软件问题**（同一张卡的其他 shape 已经证明"
              "取数通路没问题）；两边都高则是硬件规格差异。", ""]
    else:
        o += ["<未提供 --xpu-gemm，无法生成>", ""]
    o += ["### 3.2 decode attention（人工填写：注意 KV 精度可能不同，先按字节归一）", "",
          "### 3.2.1 decode 的 GDN 递推路径（人工填写）", "",
          "decode 的 GDN 走 recurrent 路径，与 prefill 的 chunk 路径是**两套不同的 "
          "kernel**，不要把 §2.3 的结论套过来；这条路径的达成率往往反而是健康的，"
          "可以用来佐证 GEMM 的问题是软件而非硬件。", ""]

    # ---- 3.3 layers -----------------------------------------------------
    o += ["### 3.3 按层类型对照", "", "**PREFILL**", ""]
    layer_table(x["layers"].get("prefill", {}), n["layers"].get("prefill", {}), o)
    o += ["", "**DECODE**", ""]
    layer_table(x["layers"].get("decode", {}), n["layers"].get("decode", {}), o)
    o += [""]

    # ---- 4 gap ----------------------------------------------------------
    o += ["## 4. gap 分解汇总", "",
          f"prefill 总差 **{f(pgap, '+.3f')} ms**（**{f(ratio(x.get('prefill_ms'), n.get('prefill_ms')), '.2f')}×**），"
          f"decode 总差 **{f(dgap, '+.3f')} ms**（**{f(dr, '.2f')}×**）。", "",
          "| 类别 | prefill B 领先 | prefill 差值 ms | 占 prefill gap | "
          "decode B 领先 | decode 差值 ms | 占 decode gap |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    allk = set(x["categories"]["prefill"]) | set(n["categories"]["prefill"]) | \
        set(x["categories"]["decode"]) | set(n["categories"]["decode"])

    def diff(phase, k):
        return (x["categories"][phase].get(k, (0, 0))[1]
                - n["categories"][phase].get(k, (0, 0))[1])

    def rat2(phase, k):
        a = x["categories"][phase].get(k, (0, 0))[1]
        b = n["categories"][phase].get(k, (0, 0))[1]
        if not a and not b:
            return "—"
        if not b:
            return "B 无"
        if not a:
            return "A 无"
        return f"{a/b:.2f}×"
    for k in sorted(allk, key=lambda k: -(abs(diff("prefill", k)) / max(abs(pgap), 1e-9)
                                          + abs(diff("decode", k)) / max(abs(dgap), 1e-9))):
        dp, dd = diff("prefill", k), diff("decode", k)
        if abs(dp) < NOISE_MS_PREFILL and abs(dd) < NOISE_MS_DECODE:
            continue
        o.append(f"| {k} | {rat2('prefill', k)} | {dp:+.3f} | "
                 f"{f(pct(dp, pgap), '.1f')} % | {rat2('decode', k)} | "
                 f"{dd:+.3f} | {f(pct(dd, dgap), '.1f')} % |")
    o += ["",
          "> \u6bd4\u503c < 1 \u7684\u9879 = **A \u66f4\u5feb**，要在报告里明确指出，不要只报差距。",
          "> 同一个类别在 prefill 和 decode 的比值常常差很多（计算受限 vs 访存受限），"
          "两列要分开解读。", "",
          "### 每个 gap 的根因判定（人工填写）", "",
          "| gap | 根因（软件问题 / 硬件规格 / 实现成熟度） | 证据 |",
          "|---|---|---|", "| <最大项> | | |", "", ""]

    # ---- 5 checklist ----------------------------------------------------
    o += ["## 5. 仍需人工补充的部分", "",
          "- [ ] §0 对比基准：设备、量化格式（**务必写清 W?A? 和 MAC 精度**）、"
          "GEMM 后端、KV cache 精度、图执行模式、可比性声明",
          "- [ ] §0 的 **Full attention 与 GDN 两行后端要分开写** —— 混合模型两边往往用"
          "完全不同的库（例如手写 SYCL vs Triton FLA），漏写会让读者误以为是同一套"
          "算法的快慢之分；若实现不同，§2.3 要按**功能**分组对照而非 kernel 一一对应",
          "- [ ] §1 的一句话结论，以及 XPU 侧 reflow 后的 wall/busy",
          "- [ ] §2.2 / §2.3 / §3.2：full attention、GDN chunk（prefill）、"
          "GDN recurrent（decode）的逐 kernel 对照 —— **三者是三套不同的 kernel**",
          "- [ ] §4 每个 gap 的根因判定 —— **先用各平台自身最佳实测值归一，"
          "再判断是软件问题还是硬件规格差异**",
          "- [ ] §6 优化优先级表：每行都要有从实测值推导的量化收益，"
          "并**标注哪些项之间互相覆盖、不能叠加**",
          "- [ ] 计量注意事项：时间戳塌缩、误入 prefill 的图重放、"
          "`cat=overhead`、两边完整 decode 窗口数不同等", "",
          "## 6. 优化优先级（人工填写）", "",
          "| # | 优化项 | 依据（实测值） | 预期收益 | 难度 |", "|---|---|---|---|---|",
          "| 1 | | | | |", "",
          "### 6.1 差距收窄路径", "",
          "| 阶段 | 现状 | 完成 #1 | #1+#2 | 全部完成 |", "|---|---:|---:|---:|---:|",
          f"| prefill | {f(x.get('prefill_ms'))} ms（{f(pr, '.2f')}×） | | | |",
          f"| decode/step | {f(x.get('decode_ms'))} ms（{f(dr, '.2f')}×） | | | |", ""]

    with open(a.output, "w", encoding="utf-8") as fh:
        fh.write("\n".join(o) + "\n")
    print(f"wrote {a.output}")
    print(f"  prefill {f(x.get('prefill_ms'))} vs {f(n.get('prefill_ms'))} ms "
          f"-> {f(pr, '.2f')}x")
    print(f"  decode  {f(x.get('decode_ms'))} vs {f(n.get('decode_ms'))} ms/step "
          f"-> {f(dr, '.2f')}x")
    print(f"  gap: prefill {f(pgap, '+.1f')} ms, decode {f(dgap, '+.2f')} ms/step")


if __name__ == "__main__":
    main()
