"""Render scheduler controls and Engine timelines from reviewed public JSON.

Optional plotting dependency: matplotlib==3.10.8. Runtime inference does not need it.
"""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_svg(fig, path):
    fig.savefig(path, metadata={"Date": None})
    # Matplotlib path attributes contain trailing spaces; normalize source text.
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip()
                    + "\n", encoding="utf-8")


def scheduler_plot(directory, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for ax, name, title, limit in zip(axes, ("main", "batch1"),
            ("Batch limit 8: compatible grouping helps", "Batch limit 1: scheduling cost dominates"),
            (5, 1.2)):
        summary = read(directory / name / "summary.json")
        if summary["status"] != "passed":
            raise ValueError("only complete passed matrices can be plotted")
        rows = summary["comparisons"]
        values = [r["metrics"]["throughput_speedup"] for r in rows]
        y = [v["median"] for v in values]
        ax.bar(range(len(rows)), y, color="#2563eb" if name == "main" else "#b45309", width=.58)
        ax.errorbar(range(len(rows)), y,
                    yerr=[[v["median"]-v["min"] for v in values],
                          [v["max"]-v["median"] for v in values]],
                    fmt="none", ecolor="#111827", capsize=4)
        for i, v in enumerate(values):
            ax.text(i, v["max"] + limit*.025, f'{v["median"]:.3f}x', ha="center", fontsize=10)
        ax.axhline(1, color="#6b7280", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(rows)), ["no reuse" if r["scenario"] == "no_reuse" else
                                        r["scenario"].replace("_reuse", "").replace("_", " ")
                                        for r in rows])
        ax.set_ylim(0, limit)
        ax.set_ylabel("Cache-Aware / FCFS throughput")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("RTX 5090 | same 1,000-request burst | 30 s deadline | Graph off\n"
                 "Paired medians; whiskers show min-max across 3 pairs (not confidence intervals)",
                 fontsize=12)
    save_svg(fig, directory / "scheduler_comparison.svg")
    plt.close(fig)


def engine_plot(directory, plt):
    names = ("eager_single", "eager_double", "graph_single", "graph_double")
    fig, axes = plt.subplots(4, 1, figsize=(12, 8.2), sharex=True, layout="constrained")
    phases = [("feno.cache_identity", 3, "#2563eb"),
              ("feno.scheduler_select", 2, "#b45309"),
              ("feno.batch_build", 2, "#fbbf24"),
              ("feno.runner_execute", 1, "#7c3aed")]
    for ax, name in zip(axes, names):
        trace = read(directory / name / "trace.json")
        gpu = read(directory / name / "gpu_timeline.json")
        events = trace["traceEvents"]
        origin = next(e["ts"] for e in events if e["name"] == "feno.engine_window")
        for phase, lane, color in phases:
            ranges = [((e["ts"]-origin)/1000, e["dur"]/1000)
                      for e in events if e["name"] == phase]
            ax.broken_barh(ranges, (lane-.32, .64), facecolors=color)
        for kind in ("KERNEL", "MEMCPY", "MEMSET", "GRAPH_TRACE"):
            ranges = [(e["start_ns"]/1e6, (e["end_ns"]-e["start_ns"])/1e6)
                      for e in gpu["gpu_intervals"] if e["kind"] == kind]
            ax.broken_barh(ranges, (-.32, .64),
                           facecolors="#0f766e" if kind == "GRAPH_TRACE" else "#16a34a")
        ax.set_yticks([0, 1, 2, 3], ["GPU records", "Runner", "Select / build", "Cache identity"])
        ax.set_ylim(-.6, 3.6)
        ax.set_title(f'{name.replace("_", " ")} | window {gpu["window_ms"]:.2f} ms | '
                     f'first GPU record {gpu["first_gpu_activity_ms"]:.2f} ms', loc="left", fontsize=10)
        ax.axvline(gpu["window_ms"], color="#6b7280", linestyle=":", linewidth=1)
        ax.grid(axis="x", alpha=.2)
    axes[-1].set_xlabel("Milliseconds from engine_window start (common scale)")
    axes[-1].set_xlim(0, 200)
    fig.suptitle("Diagnostic only: 64-request burst, 8 batches per profile\n"
                 "GPU lane includes Graph envelopes; blank host regions are not classified CPU idle time",
                 fontsize=12)
    save_svg(fig, directory / "engine_timeline.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if args.scheduler_dir is None and args.profile_dir is None:
        parser.error("provide at least one result directory")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"svg.hashsalt": "feno-rt", "font.family": "DejaVu Sans"})
    import matplotlib.pyplot as plt
    if args.scheduler_dir:
        scheduler_plot(args.scheduler_dir, plt)
    if args.profile_dir:
        engine_plot(args.profile_dir, plt)


if __name__ == "__main__":
    main()
