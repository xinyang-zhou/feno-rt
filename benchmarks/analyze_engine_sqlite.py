"""Extract anonymous, window-relative GPU evidence from an Nsight SQLite export.

Graph trace spans include internal gaps. Their union with kernels and copies is
recorded activity/envelope coverage, not GPU Active, SM utilization, or occupancy.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


def union_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def extract(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        windows = db.execute(
            "SELECT n.start,n.end FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId=s.id "
            "WHERE coalesce(n.text,s.value)='feno.engine_window'"
        ).fetchall()
        if len(windows) != 1 or windows[0][1] is None or windows[0][1] <= windows[0][0]:
            raise ValueError("expected one complete, positive engine_window")
        start, end = windows[0]
        records = []
        counts = Counter()
        for kind in ("KERNEL", "MEMCPY", "MEMSET", "GRAPH_TRACE"):
            table = "CUPTI_ACTIVITY_KIND_" + kind
            if table not in tables:
                continue
            for begin, finish in db.execute(
                f"SELECT start,end FROM {table} WHERE start < ? AND end > ?", (end, start)
            ):
                records.append({"kind": kind, "start_ns": max(start, begin) - start,
                                "end_ns": min(end, finish) - start})
                counts[kind] += 1
        if not records:
            raise ValueError("no GPU activity inside engine_window")
        api_totals = []
        for name, count, total, longest in db.execute(
            "SELECT s.value,count(*),sum(min(r.end,?)-max(r.start,?)),"
            "max(min(r.end,?)-max(r.start,?)) "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId "
            "WHERE r.start < ? AND r.end > ? GROUP BY s.value ORDER BY s.value",
            (end, start, end, start, end, start),
        ):
            api_totals.append(dict(name=name, count=count, total_ms=total / 1e6,
                                   max_ms=longest / 1e6))
    intervals = union_intervals([(r["start_ns"], r["end_ns"]) for r in records])
    covered = sum(b - a for a, b in intervals)
    return dict(
        schema_version="1.0.0", result_class="diagnostic",
        source_sqlite_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        window_ms=(end - start) / 1e6,
        first_gpu_activity_ms=intervals[0][0] / 1e6,
        last_gpu_activity_ms=intervals[-1][1] / 1e6,
        recorded_gpu_coverage_ms=covered / 1e6,
        recorded_gpu_coverage_percent=covered / (end - start) * 100,
        record_counts=dict(counts), cuda_api=api_totals,
        gpu_intervals=sorted(records, key=lambda r: (r["start_ns"], r["end_ns"], r["kind"])),
        notes=[
            "Intervals are clipped to feno.engine_window and expressed relative to its start.",
            "Coverage merges overlapping kernel/copy/memset intervals and Graph trace envelopes.",
            "Graph envelopes may include internal gaps; coverage is not GPU Active or SM utilization.",
            "No process/thread identifiers, paths, kernel argument values, or binary report are exported.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a new output path")
    result = extract(args.sqlite)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Window: {result['window_ms']:.3f} ms; coverage: "
          f"{result['recorded_gpu_coverage_ms']:.3f} ms")


if __name__ == "__main__":
    main()
