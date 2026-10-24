"""Benchmark results visualizer.

Reads all benchmark JSON files from evaluation/results/, groups them by
dataset, and produces grouped bar charts (one figure per dataset) comparing
runs across git commits. Error bars show per-sample standard deviation.

Usage (from the repository root):
    python evaluation/Visualizations/plot_benchmarks.py

Output:
    One PNG per dataset saved next to this script.
"""

import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUTPUT_DIR = os.path.dirname(__file__)

# Metrics to display, in order. Tuple: (key, display label)
METRICS = [
    ("faithfulness",        "Faithfulness"),
    ("answer_relevancy",    "Answer\nRelevancy"),
    ("context_recall",      "Context\nRecall"),
    ("context_precision",   "Context\nPrecision"),
    ("factual_correctness", "Factual\nCorrectness"),
    ("rouge1",              "ROUGE-1"),
    ("rougeL",              "ROUGE-L"),
    ("semantic_similarity", "Semantic\nSimilarity"),
    ("answer_groundedness", "Answer\nGroundedness"),
    ("context_relevance",   "Context\nRelevance"),
]

# Colour palette — one per run (cycles if more runs than colours)
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def _load_results(folder: str) -> list[dict]:
    """Load all benchmark JSON files from *folder*."""
    runs = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".json") or fname.startswith("_"):
            continue
        path = os.path.join(folder, fname)
        with open(path) as f:
            data = json.load(f)
        # Only handle benchmark-mode files
        if data.get("metadata", {}).get("mode") != "benchmark":
            print(f"Skipping {fname} (not a benchmark file)")
            continue
        runs.append(data)
    return runs


def _per_sample_std(samples: list[dict], metric_key: str) -> float:
    """Compute std dev of *metric_key* across per-sample results."""
    values = []
    for s in samples:
        v = s.get("scores", {}).get(metric_key)
        if v is not None and isinstance(v, (int, float)) and not math.isnan(v):
            values.append(v)
    if len(values) < 2:
        return 0.0
    arr = np.array(values)
    return float(np.std(arr, ddof=1))


def _run_label(meta: dict) -> str:
    sha = meta.get("git_commit", "unknown")
    dirty = " *" if meta.get("git_dirty") else ""
    return f"{sha}{dirty}"


def plot_dataset(dataset_name: str, runs: list[dict], output_dir: str) -> None:
    """Create and save a grouped bar chart for all runs of *dataset_name*."""
    n_metrics = len(METRICS)
    n_runs = len(runs)

    x = np.arange(n_metrics)
    group_width = 0.75
    bar_width = group_width / n_runs
    offsets = np.linspace(-(group_width - bar_width) / 2,
                           (group_width - bar_width) / 2,
                           n_runs)

    fig, ax = plt.subplots(figsize=(max(14, n_metrics * 1.4), 6))

    bars_handles = []
    for i, run in enumerate(runs):
        meta = run["metadata"]
        agg = run["aggregate_scores"]
        samples = run["per_sample_results"]
        colour = PALETTE[i % len(PALETTE)]
        label = _run_label(meta)

        means = []
        stds = []
        for key, _ in METRICS:
            v = agg.get(key)
            means.append(v if v is not None else 0.0)
            stds.append(_per_sample_std(samples, key))

        bars = ax.bar(
            x + offsets[i],
            means,
            width=bar_width * 0.92,
            color=colour,
            alpha=0.88,
            label=label,
            yerr=stds,
            error_kw=dict(elinewidth=1.2, capsize=3, capthick=1.2, ecolor="#333333"),
            zorder=3,
        )
        bars_handles.append(bars)

        # Value labels on top of each bar
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + std + 0.012,
                    f"{mean:.2f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="#222222",
                    rotation=90 if n_runs > 3 else 0,
                )

    # Axes formatting
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRICS], fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Title
    num_q = runs[0]["metadata"].get("num_samples", "?")
    ax.set_title(
        f"Benchmark — {dataset_name}  ·  {num_q} questions",
        fontsize=13, fontweight="bold", pad=14,
    )

    # Legend
    ax.legend(
        title="Git commit",
        title_fontsize=8,
        fontsize=8,
        loc="upper right",
        framealpha=0.9,
    )

    # Git / dataset watermark in bottom-right corner
    all_shas = ", ".join(_run_label(r["metadata"]) for r in runs)
    fig.text(
        0.99, 0.01,
        f"dataset: {dataset_name}  |  commits: {all_shas}  |  n={num_q}",
        ha="right", va="bottom",
        fontsize=7, color="#888888",
        style="italic",
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"benchmark_{dataset_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"Results folder not found: {RESULTS_DIR}")
        return

    runs = _load_results(RESULTS_DIR)
    if not runs:
        print("No benchmark JSON files found in results/")
        return

    # Group by dataset
    by_dataset: dict[str, list] = defaultdict(list)
    for run in runs:
        ds = run["metadata"].get("dataset", "unknown")
        by_dataset[ds].append(run)

    for dataset_name, dataset_runs in by_dataset.items():
        # Sort runs by timestamp so legend order is chronological
        dataset_runs.sort(key=lambda r: r["metadata"].get("timestamp", ""))
        print(f"\nPlotting {dataset_name} ({len(dataset_runs)} run(s))...")
        plot_dataset(dataset_name, dataset_runs, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
