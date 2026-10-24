#!/usr/bin/env python
"""Generate thesis figures from benchmark results (styled like the
inspiration paper — see figstyle.py).

With no arguments, every dataset in benchmark/results/ is plotted (the
newest run per dataset) and all figures are regenerated:

  benchmark_<dataset>_quality.pdf   — (a) nDCG@k, (b) Recall@k over cutoff k,
                                      (c) nDCG@10 per method as bars
  benchmark_<dataset>_tradeoff.pdf  — quality vs median query latency (log x)
  benchmark_all_datasets.pdf        — nDCG@10 grouped bars, methods x datasets

Usage:
    python -m benchmark.plot_results
    # or plot specific runs:
    ... plot_results benchmark/results/scifact_*.json

A .png preview is written next to each .pdf.
"""
import argparse
import glob
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch

from benchmark.figstyle import (
    SERIES, DATASET_HUES, GRID, INK, INK_2, INK_3, MONO, apply_style,
    card_panels, panel_letter, bottom_legend, rounded_bar, series_line,
    method_colors, mono_numeric_ticks, style_axes, tint,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_OUT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures"))

DATASET_NAMES = {
    "scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna",
    "scidocs": "SciDocs", "fiqa": "FiQA", "trec-covid": "TREC-COVID",
    "quora": "Quora", "nq": "NQ", "msmarco": "MS MARCO",
    "juristcu": "JurisTCU",
}


def _methods_in(results):
    """Canonical series order, restricted to methods present in the run."""
    return [m for m in SERIES if m in results["methods"]]


def _metric_over_k(results, method, metric, k_values):
    q = results["methods"][method]["quality"]
    return [q[f"{metric}@{k}"] for k in k_values]


def _save(fig, out_dir, name):
    mono_numeric_ticks(fig)
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(pdf)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200)
    plt.close(fig)
    return pdf


def _style_k_axis(ax, k_values):
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.minorticks_off()
    ax.set_xlim(k_values[0] * 0.85, k_values[-1] * 1.18)


def _clamp_bar_width(ax, width, max_px=24):
    """Cap a bar's data-unit width so it never exceeds max_px on screen."""
    origin = ax.transData.transform((0, 0))
    per_px = abs(ax.transData.inverted().transform(origin + [1, 0])[0])
    return min(width, per_px * max_px)


def plot_quality(results, out_dir, hue_idx=0):
    methods = _methods_in(results)
    # One distinct hue per method (the SERIES palette), so the four lines can
    # be told apart; a single faded hue made them too similar.
    colors = {m: SERIES[m][1] for m in methods}
    k_values = results["config"]["k_values"]

    fig = plt.figure(figsize=(6.6, 2.55))
    axes, cards = card_panels(fig, 3, pad_l=0.084)

    # (a) nDCG@k, (b) Recall@k — identity over cutoff
    for ax, metric, label in ((axes[0], "nDCG", "nDCG@k"),
                              (axes[1], "Recall", "Recall@k")):
        for m in methods:
            series_line(ax, k_values, _metric_over_k(results, m, metric, k_values),
                        colors[m])
        _style_k_axis(ax, k_values)
        ax.set_xlabel("Cutoff k")
        ax.set_ylabel(label)

    # (c) MAP@k — same treatment as the other two panels
    ax = axes[2]
    for m in methods:
        series_line(ax, k_values, _metric_over_k(results, m, "MAP", k_values),
                    colors[m])
    _style_k_axis(ax, k_values)
    ax.set_xlabel("Cutoff k")
    ax.set_ylabel("MAP@k")

    for card, letter in zip(cards, "abc"):
        panel_letter(fig, card, letter)
    bottom_legend(fig, methods, y=0.02, colors=colors)
    return _save(fig, out_dir, f"benchmark_{results['dataset']}_quality")


def plot_tradeoff(results, out_dir, hue_idx=0):
    methods = _methods_in(results)
    colors = method_colors(hue_idx, methods)

    fig = plt.figure(figsize=(4.8, 3.1))
    axes, _ = card_panels(fig, 1, card_bottom=0.03, pad_b=0.16, pad_l=0.105,
                          pad_t=0.055, pad_r=0.035)
    ax = axes[0]

    lats = [results["methods"][m]["latency"]["p50_ms"] for m in methods]
    ndcgs = [results["methods"][m]["quality"]["nDCG@10"] for m in methods]

    ax.set_xscale("log")
    ax.set_xlim(min(lats) * 0.4, max(lats) * 4)
    ticks = [t for t in (0.1, 1, 10, 100, 1000, 10000)
             if min(lats) * 0.4 <= t <= max(lats) * 4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    span = max(ndcgs) - min(ndcgs) or 0.05
    ax.set_ylim(min(ndcgs) - span * 0.35, max(ndcgs) + span * 0.45)

    for m, x, y in zip(methods, lats, ndcgs):
        ax.plot([x], [y], marker="o", markersize=9,
                markerfacecolor=colors[m], markeredgecolor="#f5f5f7",
                markeredgewidth=2, zorder=4)
        ax.annotate(SERIES[m][0], (x, y), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    fontsize=8, color=INK)

    ax.set_xlabel("Median query latency (ms, log scale)")
    ax.set_ylabel("nDCG@10")
    return _save(fig, out_dir, f"benchmark_{results['dataset']}_tradeoff")


def plot_metric_matrix(results, out_dir, hue_idx=0):
    """Methods x cutoff-k matrix for nDCG and Recall: printed values, cell
    shade = magnitude within its panel (pale = low, saturated = high)."""
    methods = _methods_in(results)
    k_values = results["config"]["k_values"]
    base = DATASET_HUES[hue_idx % len(DATASET_HUES)]
    short = {"bm25": "Sparse", "dense": "Dense", "hybrid": "Hybrid",
             "hybrid_rerank": "+Rerank"}

    fig = plt.figure(figsize=(6.6, 2.35))
    axes, cards = card_panels(fig, 2, card_bottom=0.03, pad_l=0.075,
                              pad_r=0.012, pad_t=0.115, pad_b=0.155)

    for ax, metric in ((axes[0], "nDCG"), (axes[1], "Recall")):
        vals = [[results["methods"][m]["quality"][f"{metric}@{k}"]
                 for k in k_values] for m in methods]
        vmin = min(min(row) for row in vals)
        vmax = max(max(row) for row in vals)
        ax.grid(False)
        ax.set_xlim(0, len(k_values))
        ax.set_ylim(len(methods), 0)  # first method on top
        for i, m in enumerate(methods):
            for j, v in enumerate(vals[i]):
                frac = (v - vmin) / ((vmax - vmin) or 1)
                fade = 0.88 * (1 - frac)  # high value -> saturated
                ax.add_patch(FancyBboxPatch(
                    (j + 0.05, i + 0.08), 0.90, 0.84,
                    boxstyle="round,pad=0,rounding_size=0.07",
                    facecolor=tint(base, fade), edgecolor="none"))
                ax.text(j + 0.5, i + 0.5, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, fontfamily=MONO,
                        color="#ffffff" if fade < 0.30 else INK)
        ax.set_xticks([j + 0.5 for j in range(len(k_values))])
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_yticks([i + 0.5 for i in range(len(methods))])
        ax.set_yticklabels([short.get(m, m) for m in methods], fontsize=8)
        ax.set_xlabel("Cutoff k")
        ax.set_title(f"{metric}@k", fontsize=9, color=INK_2, pad=5)

    for card, letter in zip(cards, "ab"):
        panel_letter(fig, card, letter)
    return _save(fig, out_dir, f"benchmark_{results['dataset']}_matrix")


def plot_sweep(sweep, out_dir):
    """RRF_K x candidate-pool heatmap (diverging vs the config baseline:
    blue = better, red = worse; amber ring = best cell, gray ring = baseline)."""
    rrf_ks = sweep["rrf_k_values"]
    cands = sweep["candidate_values"]
    cells = sweep["cells"]
    base_k = sweep["baseline"]["rrf_k"]
    base_c = sweep["baseline"]["candidates"]

    if base_k in rrf_ks and base_c in cands:
        base_v = cells[rrf_ks.index(base_k)][cands.index(base_c)]
    else:
        flat = sorted(v for row in cells for v in row)
        base_v = flat[len(flat) // 2]
    scale = max(abs(v - base_v) for row in cells for v in row) or 1.0

    fig_w = 1.15 + 0.60 * len(cands)
    fig_h = 1.15 + 0.335 * len(rrf_ks)
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes, _ = card_panels(fig, 1, card_bottom=0.02, pad_l=0.55 / fig_w,
                          pad_r=0.10 / fig_w, pad_b=0.52 / fig_h,
                          pad_t=0.34 / fig_h)
    ax = axes[0]
    ax.grid(False)
    ax.set_xlim(0, len(cands))
    ax.set_ylim(len(rrf_ks), 0)  # smallest RRF k on top

    best = sweep["best"]
    for i, K in enumerate(rrf_ks):
        for j, c in enumerate(cands):
            v = cells[i][j]
            frac = (v - base_v) / scale  # -1 .. 1
            pole = "#3d6eb5" if frac >= 0 else "#bf4e2c"
            color = tint(pole, 0.94 * (1 - abs(frac)))
            edge, lw = "none", 0.0
            if K == best["rrf_k"] and c == best["candidates"]:
                edge, lw = "#b48507", 1.6
            elif K == base_k and c == base_c:
                edge, lw = INK_2, 1.2
            ax.add_patch(FancyBboxPatch(
                (j + 0.05, i + 0.075), 0.90, 0.85,
                boxstyle="round,pad=0,rounding_size=0.07",
                facecolor=color, edgecolor=edge, linewidth=lw))
            ax.text(j + 0.5, i + 0.5, f"{v * 100:.1f}", ha="center", va="center",
                    fontsize=6.8, fontfamily=MONO,
                    color="#ffffff" if abs(frac) > 0.72 else INK)

    ax.set_xticks([j + 0.5 for j in range(len(cands))])
    ax.set_xticklabels([str(c) for c in cands], fontsize=7.5)
    ax.set_yticks([i + 0.5 for i in range(len(rrf_ks))])
    ax.set_yticklabels([str(K) for K in rrf_ks], fontsize=7.5)
    ax.set_xlabel("Candidate pool size")
    ax.set_ylabel("RRF k")
    ax.set_title(f"{sweep['metric']} ×100 — hybrid RRF", fontsize=9,
                 color=INK_2, pad=5)
    return _save(fig, out_dir, f"benchmark_{sweep['dataset']}_sweep")


def plot_all_datasets(runs, out_dir, metric="nDCG@100", suffix="", exclude=(),
                      fig_h=3.4, unit_scale=True, bar_frac=0.92):
    """Grouped bars of `metric`: one hue per dataset, fading saturation across
    the methods within each group (method identity via per-bar labels)."""
    datasets = sorted(runs)
    # One-letter labels under the bars: S sparse, D dense, H hybrid,
    # R hybrid with reranking.
    short = {"bm25": "S", "dense": "D", "hybrid": "H", "hybrid_rerank": "R"}

    # No card here — plain page background. Sized for a full-text-width figure
    # in the thesis (the two metric panels are stacked vertically there), so
    # everything is larger than in the card-style panels: figure, bars, text.
    fig_w = max(4.5, 1.6 + 1.55 * len(datasets))
    fig = plt.figure(figsize=(fig_w, fig_h))
    # Fixed margins in inches (0.68 in below for the labels, 0.18 in above), so a
    # shorter figure only shortens the bars and keeps every label the same size.
    bottom, top = 0.68 / fig_h, 0.18 / fig_h
    ax = fig.add_axes([0.72 / fig_w, bottom, 1 - 0.85 / fig_w, 1 - bottom - top])
    style_axes(ax)
    ax.tick_params(axis="y", labelsize=10)

    all_vals = [runs[d]["methods"][m]["quality"][metric]
                for d in datasets for m in runs[d]["methods"]]
    ax.set_xlim(-0.52, len(datasets) - 0.48)
    # Every metric plotted here lies in [0, 1]: show the full 0-1 scale, with
    # headroom above 1 for the rotated value labels
    if unit_scale:
        ax.set_ylim(0, max(1.0, max(all_vals)) * 1.32)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    else:  # metrics far below 1 (P@100) would be invisible on a 0-1 scale
        ax.set_ylim(0, max(all_vals) * 1.32)
        ticks = [t for t in ax.get_yticks() if t <= max(all_vals) * 1.05]
        ax.set_yticks(ticks)

    group_span = 0.92
    tick_pos, tick_labels = [], []
    for i, d in enumerate(datasets):
        methods_d = [m for m in SERIES
                     if m in runs[d]["methods"] and m not in exclude]
        colors = method_colors(i, methods_d)  # saturated -> pale, one hue
        slot = group_span / len(methods_d)
        # Wide bars that nearly touch within a group
        width = _clamp_bar_width(ax, slot * bar_frac, max_px=60)
        for j, m in enumerate(methods_d):
            v = runs[d]["methods"][m]["quality"][metric]
            x = i - group_span / 2 + slot * (j + 0.5)
            rounded_bar(ax, x, v, width, colors[m], radius_px=3)
            ax.annotate(f"{v:.3f}", (x, v), xytext=(1.5, 4),
                        textcoords="offset points", ha="left", rotation=75,
                        rotation_mode="anchor", fontsize=9, color=INK,
                        fontfamily=MONO)
            tick_pos.append(x)
            tick_labels.append(short.get(m, m))
        ax.annotate(DATASET_NAMES.get(d, d), (i, 0),
                    xycoords=("data", "axes fraction"), xytext=(0, -34),
                    textcoords="offset points", ha="center",
                    fontsize=12, color=INK_2)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=9.5)
    ax.set_ylabel(metric, fontsize=12)
    # Centre the label on the 0-1 tick range, not on the headroom above it
    ax.yaxis.label.set_y(0.5 * ax.get_yticks()[-1] / ax.get_ylim()[1])
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, f"benchmark_all_datasets{suffix}")


# Headline columns of the rendered results table: (label, getter, better)
# Reported at @10: that is the cutoff BEIR papers use (so these numbers are
# comparable with published results) and the number of chunks the pipeline
# actually hands to the LLM (RERANKER_TOP_N). Recall@100 is kept as the
# candidate-pool ceiling — it is what shows that reranking only reorders.
TABLE_COLUMNS = [
    ("nDCG@10", lambda r: r["quality"]["nDCG@10"], "higher"),
    ("MAP@10", lambda r: r["quality"]["MAP@10"], "higher"),
    ("Recall@10", lambda r: r["quality"]["Recall@10"], "higher"),
    ("Recall@100", lambda r: r["quality"]["Recall@100"], "higher"),
]


def plot_results_table(runs, out_dir):
    """Headline results as a rendered table (inspiration-paper style):
    one column group per dataset, one row per method, plain values with
    the best value per metric column in bold."""
    datasets = sorted(runs)
    methods = [m for m in SERIES
               if any(m in runs[d]["methods"] for d in datasets)]
    n_cols = len(TABLE_COLUMNS)

    label_w, cell_w, group_gap = 1.30, 0.68, 0.20
    head1_h, head2_h, row_h = 0.56, 0.30, 0.36
    pad = 0.16
    W = pad + label_w + len(datasets) * n_cols * cell_w \
        + (len(datasets) - 1) * group_gap + pad
    H = pad + head1_h + head2_h + row_h * len(methods) + pad

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def group_x(g):
        return pad + label_w + g * (n_cols * cell_w + group_gap)

    # Headers
    for g, d in enumerate(datasets):
        x0 = group_x(g)
        xc = x0 + n_cols * cell_w / 2
        ax.text(xc, pad + 0.15, DATASET_NAMES.get(d, d), ha="center",
                fontsize=14, color=INK, weight="bold")
        ax.text(xc, pad + 0.37,
                f"{runs[d]['indexing']['corpus_docs']:,} docs · "
                f"{runs[d]['methods'][methods[0]]['latency']['queries']} queries",
                ha="center", fontsize=10, color=INK_3)
        for c, (name, _, _) in enumerate(TABLE_COLUMNS):
            ax.text(x0 + (c + 0.5) * cell_w, pad + head1_h + 0.16,
                    name, ha="center", fontsize=9, color=INK_3)
        if g > 0:  # vertical separator between dataset groups
            xs = x0 - group_gap / 2
            ax.plot([xs, xs], [pad, H - pad], color=GRID, lw=1)
    y_rule = pad + head1_h + head2_h
    ax.plot([pad, W - pad], [y_rule, y_rule], color=GRID, lw=1)

    # Reranking only reorders the hybrid candidate pool, so its Recall@100
    # is hybrid's by construction — shown as a dash, excluded from "best".
    skip_cells = {("hybrid_rerank", "Recall@100")}

    # Best value per (dataset, metric) column across methods
    best = {
        (d, c): max(get(runs[d]["methods"][m])
                    for m in methods if m in runs[d]["methods"]
                    and (m, name) not in skip_cells)
        for d in datasets
        for c, (name, get, _) in enumerate(TABLE_COLUMNS)
    }

    y = y_rule
    for r, m in enumerate(methods):
        yc = y + row_h / 2
        ax.text(pad + 0.04, yc, SERIES[m][0], va="center", fontsize=11.5,
                color=INK)
        for g, d in enumerate(datasets):
            if m not in runs[d]["methods"]:
                continue
            res = runs[d]["methods"][m]
            x0 = group_x(g)
            for c, (name, get, _) in enumerate(TABLE_COLUMNS):
                if (m, name) in skip_cells:
                    ax.text(x0 + (c + 0.5) * cell_w, yc, "–", ha="center",
                            va="center", fontsize=10.5, color=INK_3)
                    continue
                v = get(res)
                is_best = v == best[(d, c)]
                ax.text(x0 + (c + 0.5) * cell_w, yc, f"{v:.3f}",
                        ha="center", va="center", fontsize=10.5,
                        fontfamily=MONO,
                        weight="bold" if is_best else "normal",
                        color=INK if is_best else INK_2)
        y += row_h
        if r < len(methods) - 1:
            ax.plot([pad, W - pad], [y, y], color=GRID, lw=0.8)

    return _save(fig, out_dir, "benchmark_results_table")


def _latex_escape(s):
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def write_latex_tables(runs, tables_dir):
    """Complete metric dump as booktabs LaTeX tables (appendix material):
    one file per dataset (all quality metrics at every k + latency stats)
    plus one indexing-statistics table across datasets. Best value per row
    is bolded. Requires \\usepackage{booktabs}."""
    os.makedirs(tables_dir, exist_ok=True)
    written = []
    heads = {"bm25": "Sparse", "dense": "Dense", "hybrid": "Hybrid",
             "hybrid_rerank": "Hybrid+RR"}

    for d in sorted(runs):
        results = runs[d]
        methods = [m for m in SERIES if m in results["methods"]]
        k_values = results["config"]["k_values"]
        name = DATASET_NAMES.get(d, d)
        cols = " ".join(["l l"] + ["r"] * len(methods))
        header = " & ".join([r"Metric", r"$k$"] + [heads[m] for m in methods])

        lines = [
            "% Auto-generated by benchmark/plot_results.py — do not edit.",
            r"\begin{table}[htbp]", r"\centering", r"\small",
            rf"\caption{{Retrieval quality on {name} "
            rf"({results['indexing']['corpus_docs']:,} documents, "
            rf"{results['methods'][methods[0]]['latency']['queries']} queries). "
            rf"Best value per row in bold.}}",
            rf"\label{{tab:benchmark-{d}}}",
            rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
            header + r" \\", r"\midrule",
        ]
        for metric in ("nDCG", "MAP", "Recall", "P", "MRR"):
            for i, k in enumerate(k_values):
                vals = [results["methods"][m]["quality"][f"{metric}@{k}"]
                        for m in methods]
                best = max(vals)
                cells = [rf"\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}"
                         for v in vals]
                row_name = rf"{metric}@$k$" if i == 0 else ""
                lines.append(" & ".join([row_name, str(k)] + cells) + r" \\")
            lines.append(r"\addlinespace")
        lines += [r"\midrule"]
        for stat, label in (("mean_ms", "mean (ms)"), ("p50_ms", "p50 (ms)"),
                            ("p95_ms", "p95 (ms)"), ("p99_ms", "p99 (ms)"),
                            ("qps", "queries/s")):
            vals = [results["methods"][m]["latency"][stat] for m in methods]
            best = min(vals) if stat != "qps" else max(vals)
            cells = [rf"\textbf{{{v:g}}}" if v == best else f"{v:g}"
                     for v in vals]
            row_name = "Latency" if stat == "mean_ms" else ""
            lines.append(" & ".join([row_name, label] + cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

        path = os.path.join(tables_dir, f"benchmark_{d}.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        written.append(path)

    # Indexing statistics across datasets
    lines = [
        "% Auto-generated by benchmark/plot_results.py — do not edit.",
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\caption{Indexing statistics per dataset, on the Apple M4 used for every experiment, with the embeddings computed on its GPU.}",
        r"\label{tab:benchmark-indexing}",
        r"\begin{tabular}{l r r r r l r r}", r"\toprule",
        r"Dataset & Docs & BM25 (s) & Embed (s) & Docs/s & FAISS type & "
        r"Index (MB) & Peak RSS (MB) \\",
        r"\midrule",
    ]
    for d in sorted(runs):
        idx = runs[d]["indexing"]
        total_mb = idx["faiss"]["size_on_disk_mb"] + idx["bm25"]["size_on_disk_mb"]
        lines.append(" & ".join([
            DATASET_NAMES.get(d, d), f"{idx['corpus_docs']:,}",
            f"{idx['bm25']['build_seconds']:g}",
            f"{idx['embeddings']['build_seconds']:g}",
            f"{idx['embeddings']['docs_per_second']:g}",
            _latex_escape(idx["faiss"]["resolved_type"]),
            f"{total_mb:g}", f"{runs[d].get('peak_rss_mb', 0):g}",
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path = os.path.join(tables_dir, "benchmark_indexing.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    written.append(path)
    return written


def load_runs(paths):
    """Load result JSONs, keeping only the newest run per dataset."""
    runs, sources = {}, {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
        dataset = results["dataset"]
        stamp = results.get("timestamp", "")
        if dataset not in runs or stamp > runs[dataset].get("timestamp", ""):
            if dataset in runs:
                print(f"  {dataset}: using newer run, skipping {sources[dataset]}")
            runs[dataset] = results
            sources[dataset] = os.path.basename(path)
    for dataset, src in sorted(sources.items()):
        print(f"  {dataset}: {src}")
    return runs


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results for the thesis.")
    parser.add_argument("results", nargs="*",
                        help="results JSON path(s) or glob(s); default: every "
                             f"dataset in {RESULTS_DIR} (newest run each)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT,
                        help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--tables-dir", default=None,
                        help="LaTeX tables output (default: <out-dir>/../tables)")
    args = parser.parse_args()

    if args.results:
        paths = [p for pattern in args.results
                 for p in (sorted(glob.glob(pattern)) or [pattern])]
    else:
        paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")))
    if not paths:
        sys.exit(f"No results JSON found in {RESULTS_DIR}. Run the benchmark first.")

    print("Runs used:")
    runs = load_runs(paths)

    apply_style()
    # hue_idx follows sorted dataset order, matching plot_all_datasets
    for hue_idx, (dataset, results) in enumerate(sorted(runs.items())):
        print(f"  wrote {plot_quality(results, args.out_dir, hue_idx)}")
        print(f"  wrote {plot_tradeoff(results, args.out_dir, hue_idx)}")
        print(f"  wrote {plot_metric_matrix(results, args.out_dir, hue_idx)}")
    # Headline figures at @10 (comparable with BEIR; equals what reaches the LLM)
    # The two figures shown in the thesis (Figure benchmark-retrieval) are drawn
    # shorter, so the stacked pair does not fill a page on its own.
    print(f"  wrote {plot_all_datasets(runs, args.out_dir, metric='nDCG@10', bar_frac=0.8)}")
    print(f"  wrote {plot_all_datasets(runs, args.out_dir, metric='Recall@10', suffix='_recall10')}")
    # @100 kept: Recall@100 is the candidate-pool ceiling, identical for hybrid
    # and hybrid+rerank by construction, which is the "reranking only reorders"
    # evidence. hybrid_rerank is excluded there to avoid duplicating hybrid.
    print(f"  wrote {plot_all_datasets(runs, args.out_dir, metric='nDCG@100', suffix='_ndcg100')}")
    print(f"  wrote {plot_all_datasets(runs, args.out_dir, metric='Recall@100', suffix='_recall100', exclude=('hybrid_rerank',), bar_frac=0.8)}")
    print(f"  wrote {plot_all_datasets(runs, args.out_dir, metric='MAP@100', suffix='_map100')}")
    print(f"  wrote {plot_all_datasets(runs, args.out_dir, metric='P@100', suffix='_p100', unit_scale=False)}")
    print(f"  wrote {plot_all_datasets(runs, args.out_dir, metric='MRR@100', suffix='_mrr100')}")
    # Taller copies for the appendix, where two share a row on a landscape page
    for metric, suffix, kw in (("nDCG@10", "", {}), ("Recall@10", "_recall10", {}),
                               ("nDCG@100", "_ndcg100", {}),
                               ("Recall@100", "_recall100", {"exclude": ("hybrid_rerank",)}),
                               ("MAP@100", "_map100", {}), ("P@100", "_p100", {"unit_scale": False}),
                               ("MRR@100", "_mrr100", {})):
        print(f"  wrote {plot_all_datasets(runs, args.out_dir, metric=metric, suffix=suffix + '_appendix', fig_h=4.8, **kw)}")
    print(f"  wrote {plot_results_table(runs, args.out_dir)}")
    tables_dir = args.tables_dir or os.path.join(
        os.path.dirname(os.path.normpath(args.out_dir)), "tables")
    for path in write_latex_tables(runs, tables_dir):
        print(f"  wrote {path}")

    # Regenerate sweep heatmaps too (newest sweep per dataset)
    if not args.results:
        sweeps = {}
        for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "sweeps", "*.json"))):
            with open(path, encoding="utf-8") as f:
                sweep = json.load(f)
            if sweep.get("timestamp", "") >= sweeps.get(sweep["dataset"], {}).get("timestamp", ""):
                sweeps[sweep["dataset"]] = sweep
        for sweep in sweeps.values():
            print(f"  wrote {plot_sweep(sweep, args.out_dir)}")


if __name__ == "__main__":
    main()
