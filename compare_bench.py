#!/usr/bin/env python3
"""
多机结果对比可视化
=================

读取多个 `bench_*.json`（来自 test_dgx.py），生成一张并排对比图。

用法：
    python compare_bench.py results/bench_mac*.json results/bench_dgx*.json
    python compare_bench.py results/*.json                    # 全部
    python compare_bench.py results/*.json -o compare.png     # 指定输出
    python compare_bench.py results/*.json --metric ttft      # 只画某个指标

输出默认 `compare_<timestamp>.png`。
"""

import argparse
import glob
import json
import math
import sys
from datetime import datetime
from pathlib import Path


# ============ 综合评分（与 test_dgx.py 同源，复制以保持本脚本自包含）============
# 权重必须与 test_dgx.py SCORE_WEIGHTS 一致，否则单机分和对比分会不一致。
SCORE_WEIGHTS = {"quality": 0.5, "throughput": 0.3, "ttft": 0.1, "stability": 0.1}


def _quality_to_01(quality):
    """quality dict → 0-1"""
    if quality is None:
        return None
    if "recall" in quality:
        return quality["recall"]
    if "accuracy" in quality:
        return quality["accuracy"]
    if "scored" in quality and "max" in quality:
        m = quality["max"] or 1
        return quality["scored"] / m
    return None


def _tps_to_score(tps):
    if tps is None or tps <= 0:
        return 0
    return max(0.0, min(100.0, (math.log10(tps) - 0.7) / (2.0 - 0.7) * 100))


def _ttft_to_score(ttft_s):
    if ttft_s is None or ttft_s < 0:
        return 0
    if ttft_s <= 0.1:
        return 100.0
    return max(0.0, min(100.0, (1.3 - math.log10(ttft_s)) / (1.3 - (-1.0)) * 100))


def _stability_to_score(cv_pct):
    if cv_pct is None or cv_pct < 0:
        return 0
    return max(0.0, min(100.0, 100.0 * (10.0 - cv_pct) / (10.0 - 1.0)))


def compute_overall_score(data):
    """从 bench JSON dict 计算综合评分。需要 TASKS 顺序但容错（缺失任务跳过）。"""
    tasks = data.get("tasks", {})

    quality_scores = []
    for td in tasks.values():
        q01 = _quality_to_01(td.get("quality"))
        if q01 is not None:
            quality_scores.append(q01)
    quality_score = (sum(quality_scores) / len(quality_scores) * 100) if quality_scores else None

    tps_values = [td["tok_per_s"]["mean"] for td in tasks.values()
                  if isinstance(td.get("tok_per_s"), dict)
                  and td["tok_per_s"]["mean"] is not None]
    if tps_values:
        tps_sorted = sorted(tps_values)
        throughput_score = _tps_to_score(tps_sorted[len(tps_sorted) // 2])
    else:
        throughput_score = None

    ttft_values = [td["ttft_s"]["mean"] for td in tasks.values()
                   if isinstance(td.get("ttft_s"), dict)
                   and td["ttft_s"]["mean"] is not None]
    if ttft_values:
        ttft_sorted = sorted(ttft_values)
        ttft_score = _ttft_to_score(ttft_sorted[len(ttft_sorted) // 2])
    else:
        ttft_score = None

    stability_score = None
    lg = tasks.get("long_generation", {})
    totals = [s.get("total_s") for s in (lg.get("samples") or []) if "total_s" in s]
    if len(totals) >= 2:
        mean_t = sum(totals) / len(totals)
        if mean_t > 0:
            var = sum((t - mean_t) ** 2 for t in totals) / len(totals)
            cv = (var ** 0.5) / mean_t * 100
            stability_score = _stability_to_score(cv)

    sub_scores = {
        "quality":    quality_score,
        "throughput": throughput_score,
        "ttft":       ttft_score,
        "stability":  stability_score,
    }
    # 优先用 JSON 里已经算好的 overall（保证与 test_dgx.py 完全一致）；
    # 旧 JSON 没有则本地重算
    overall = data.get("overall_score", {}).get("overall")
    if overall is None:
        overall = 0.0
        for key, weight in SCORE_WEIGHTS.items():
            s = sub_scores[key]
            overall += weight * (s if s is not None else 0.0)

    return {
        "overall":    round(overall, 1),
        "sub_scores": {k: (round(v, 1) if v is not None else None) for k, v in sub_scores.items()},
    }


def load_results(paths):
    """从一组文件路径加载 JSON 结果。返回 [(path, dict), ...]。"""
    results = []
    for p in paths:
        try:
            with open(p) as f:
                results.append((p, json.load(f)))
        except Exception as e:
            print(f"⚠️  跳过 {p}: {e}", file=sys.stderr)
    return results


def task_metric_mean(task_data, key):
    """从 task_data 取某个指标的 mean。失败/缺失返回 None。"""
    if "ttft_s" not in task_data:
        return None
    if key not in task_data:
        return None
    val = task_data[key]
    if isinstance(val, dict) and "mean" in val:
        return val["mean"]
    return None


def task_quality_01(task_data):
    """把不同任务的 quality 归一化到 0-1。"""
    q = task_data.get("quality")
    if q is None:
        return None
    if "recall" in q:
        return q.get("recall")
    if "accuracy" in q:
        return q.get("accuracy")
    if "scored" in q and "max" in q:
        m = q.get("max") or 1
        return q.get("scored") / m
    return None


def collect_metric_matrix(results, task_order, getter):
    """返回 rows=机器数 × cols=任务数 的二维列表，缺失填 None。"""
    matrix = []
    for path, data in results:
        row = []
        tasks = data.get("tasks", {})
        for t in task_order:
            td = tasks.get(t, {})
            row.append(getter(td))
        matrix.append(row)
    return matrix


def main():
    ap = argparse.ArgumentParser(description="多机 bench JSON 对比可视化")
    ap.add_argument("paths", nargs="+", help="bench_*.json 路径或通配符")
    ap.add_argument("-o", "--output", default=None, help="输出 PNG 路径")
    ap.add_argument("--metric",
                    choices=["all", "ttft", "tps", "quality"],
                    default="all",
                    help="只画某个指标（默认 all = 全部三个）")
    ap.add_argument("--no-table", action="store_true",
                    help="不画环境对比表")
    args = ap.parse_args()

    # 展开通配符
    files = []
    for p in args.paths:
        if any(c in p for c in "*?[]"):
            files.extend(sorted(glob.glob(p)))
        else:
            files.append(p)
    if not files:
        sys.exit("❌ 没有匹配到任何 JSON 文件")

    results = load_results(files)
    if len(results) < 1:
        sys.exit("❌ 没有成功加载任何 JSON")
    n = len(results)
    print(f"📦 加载 {n} 份结果:")
    for p, d in results:
        env = d.get("environment", {})
        print(f"   - {d.get('label', '?'):<20} {env.get('chip', '?'):<14}"
              f"  {env.get('llm_engine', '?'):<12}  ({Path(p).name})")

    # 任务集合：所有结果出现的任务（按 TASKS 顺序优先）——必须与 test_dgx.py 的 9 个任务一致
    TASK_ORDER = ["mutation_call", "expression_genes", "expression_matrix",
                  "expression_code", "protein_function",
                  "pubmedqa", "medmcqa", "json_output", "long_generation"]
    seen = set()
    for _, d in results:
        seen.update(d.get("tasks", {}).keys())
    task_order = [t for t in TASK_ORDER if t in seen] + sorted(seen - set(TASK_ORDER))

    # 三个指标的矩阵
    ttft_m = collect_metric_matrix(results, task_order,
                                   lambda td: task_metric_mean(td, "ttft_s"))
    tps_m = collect_metric_matrix(results, task_order,
                                  lambda td: task_metric_mean(td, "tok_per_s"))
    quality_m = collect_metric_matrix(results, task_order, task_quality_01)

    # ---- 画图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np

    labels = [d.get("label", Path(p).stem) for p, d in results]
    envs = [d.get("environment", {}) for _, d in results]

    # 给每台机器一个稳定颜色（tab10）
    palette = plt.get_cmap("tab10").colors
    colors = [palette[i % len(palette)] for i in range(n)]

    # 子图布局：根据 --metric 决定
    show_ttft = args.metric in ("all", "ttft")
    show_tps = args.metric in ("all", "tps")
    show_quality = args.metric in ("all", "quality")
    show_table = not args.no_table

    panels = sum([show_ttft, show_tps, show_quality, show_table])
    if panels == 0:
        sys.exit("❌ 没有要画的指标")

    # 选布局：1 个 -> 1x1，2 个 -> 1x2，3-4 个 -> 2x2
    if panels <= 2:
        nrows, ncols = 1, panels
        fig_h, fig_w = 8, 7 * panels
    else:
        nrows, ncols = 2, 2
        fig_h, fig_w = 11, 15

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=110)
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.5, wspace=0.3,
                  left=0.07, right=0.97, top=0.85, bottom=0.1)
    if panels == 3:
        # 让最后一个子图（表格）占满下方
        gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.3,
                      left=0.07, right=0.97, top=0.85, bottom=0.1,
                      height_ratios=[1, 1])

    def grouped_bars(ax, matrix, ylabel, title, higher_better):
        """画分组柱状图：每组 N 根柱子（N=机器数）。"""
        x = np.arange(len(task_order))
        width = 0.8 / max(n, 1)
        for i, (row, lbl, c) in enumerate(zip(matrix, labels, colors)):
            offset = (i - n / 2 + 0.5) * width
            vals = [v if v is not None else 0 for v in row]
            mask = [v is not None for v in row]
            bars = ax.bar(x + offset, vals, width, label=lbl, color=c,
                          alpha=0.85, edgecolor="black", linewidth=0.4)
            # 标数值（小字）
            for b, v, ok in zip(bars, vals, mask):
                if ok and v > 0:
                    ax.text(b.get_x() + b.get_width() / 2, v,
                            f"{v:.1f}" if v >= 1 else f"{v:.2f}",
                            ha="center", va="bottom", fontsize=6, rotation=0)
        ax.set_xticks(x)
        ax.set_xticklabels(task_order, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        arrow = "↑" if higher_better else "↓"
        ax.set_title(f"{title}  {arrow}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    panel_idx = 0
    if show_ttft:
        ax = fig.add_subplot(gs[panel_idx // ncols, panel_idx % ncols])
        grouped_bars(ax, ttft_m, "TTFT (s)", "Time to First Token", higher_better=False)
        panel_idx += 1
    if show_tps:
        ax = fig.add_subplot(gs[panel_idx // ncols, panel_idx % ncols])
        grouped_bars(ax, tps_m, "tokens/s", "Generation Throughput", higher_better=True)
        panel_idx += 1
    if show_quality:
        ax = fig.add_subplot(gs[panel_idx // ncols, panel_idx % ncols])
        grouped_bars(ax, quality_m, "Quality (0-1)", "Output Quality", higher_better=True)
        # 基线参考线
        ax.axhline(y=0.25, color="#94a3b8", linestyle="--", linewidth=0.8)
        ax.axhline(y=0.5, color="#94a3b8", linestyle=":", linewidth=0.8)
        ax.set_ylim(0, 1.05)
        panel_idx += 1

    if show_table:
        ax = fig.add_subplot(gs[panel_idx // ncols, panel_idx % ncols]
                             if panels > 1 else gs[0, 0])
        ax.axis("off")
        ax.set_title("Environment Comparison", fontsize=12, fontweight="bold", loc="left")
        # 表头：字段 + 每台机器一列
        fields = ["label", "chip", "memory_gb_str", "memory_gb",
                  "gpu", "llm_engine", "llm_model", "llm_context"]
        headers = ["label", "chip", "memory", "memory(GB)", "gpu", "engine", "model", "ctx"]
        # 每行一个字段，每列一台机器
        col_labels = [l[:18] for l in labels]
        cell_text = []
        for hdr, f in zip(headers, fields):
            row = []
            for env in envs:
                v = env.get(f)
                if v is None and f == "memory_gb_str":
                    v = env.get("memory_gb")
                row.append(str(v) if v is not None else "-")
            cell_text.append(row)
        tbl = ax.table(
            cellText=cell_text,
            rowLabels=headers,
            colLabels=col_labels,
            loc="upper center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.35)
        # 给表头上色（按机器颜色）
        for j, c in enumerate(colors):
            tbl[(0, j)].set_facecolor((*c[:3], 0.4) if len(c) >= 3 else c)
            tbl[(0, j)].set_text_props(weight="bold")

    # 标题（含综合评分排行）
    machines_str = " vs ".join(labels)
    # 计算每台机器的综合分
    scores = [compute_overall_score(d) for _, d in results]
    # 按总分排序（保持 colors 和 labels 的对应关系），分数相同保留原顺序（稳定排序）
    order = sorted(range(len(labels)), key=lambda i: -scores[i]["overall"])
    sorted_labels = [labels[i] for i in order]
    sorted_scores = [scores[i] for i in order]
    sorted_colors = [colors[i] for i in order]

    fig.suptitle(f"Local LLM Benchmark — {machines_str}",
                 fontsize=15, fontweight="bold", y=0.97)
    sub = " · ".join(f"{l}: {e.get('chip', '?')} / {e.get('llm_engine', '?')}"
                     for l, e in zip(labels, envs))
    fig.text(0.5, 0.935, sub, ha="center", fontsize=9, color="#4b5563")

    # 分数排行行：用纯文字 1st/2nd/3rd 避免跨平台 emoji 字体问题
    # （DejaVu Sans 默认没有 🥈🥉，Windows/Linux 同事会看到方框）
    medals_short = ["#1", "#2", "#3"] + [f"#{i+1}" for i in range(3, len(labels))]
    score_line = "    ".join(
        f"[{medals_short[i]}] {l} {s['overall']:.0f}"
        for i, (l, s) in enumerate(zip(sorted_labels, sorted_scores))
    )
    fig.text(0.5, 0.905, score_line, ha="center", fontsize=11,
             color="#374151", weight="bold")

    # 输出
    out = args.output
    if out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"compare_{ts}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 终端打印综合分排行表
    print(f"\n{'=' * 78}")
    print(f"⭐ Overall score ranking (weights: {SCORE_WEIGHTS})")
    print(f"{'=' * 78}")
    print(f"{'rank':<6}{'label':<24}{'overall':<10}{'quality':<10}{'tput':<8}{'ttft':<8}{'stab':<8}")
    print("-" * 78)
    medals = ["1st", "2nd", "3rd"] + [f"{i+1}th" for i in range(3, len(labels))]
    for rank, (lbl, sc) in enumerate(zip(sorted_labels, sorted_scores)):
        sub = sc["sub_scores"]
        fmt = lambda x: "-" if x is None else f"{x:.0f}"
        print(f"{medals[rank]:<6}{lbl:<24}{sc['overall']:<10.1f}"
              f"{fmt(sub['quality']):<10}{fmt(sub['throughput']):<8}"
              f"{fmt(sub['ttft']):<8}{fmt(sub['stability']):<8}")
    print(f"\n📊 对比图已保存: {out}")


if __name__ == "__main__":
    main()
