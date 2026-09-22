"""Summarize host phases without adding overlapping or nested durations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(directory):
    result = json.loads((directory / "diagnostic.json").read_text())
    trace = json.loads((directory / "trace.json").read_text())
    if result.get("result_class") != "diagnostic":
        raise ValueError("expected an engine diagnostic result")
    phases = defaultdict(list)
    for event in trace["traceEvents"]:
        if event.get("ph") == "X":
            phases[event["name"]].append(event["dur"] / 1000)
    lines = ["# Async Engine diagnostic report", "",
             "Diagnostic only: instrumentation changes host timing. No formal speedup claim.", "",
             f"- Source: `{result['source']}`",
             f"- Environment: `{result['environment']}`",
             f"- Workload: `{result['workload']}`",
             f"- Execution: `{result['execution']}`",
             f"- Correctness: `{result['correctness']}`",
             f"- Trace health: `{trace['metadata']}`", "",
             "## Host phase durations", "",
             "| Phase | Count | Sum ms | Mean ms | Max ms |",
             "|:---|---:|---:|---:|---:|"]
    for name, samples in sorted(phases.items()):
        lines.append(f"| {name} | {len(samples)} | {sum(samples):.3f} | "
                     f"{statistics.mean(samples):.3f} | {max(samples):.3f} |")
    lines += ["", "Phases overlap across threads and requests and include nested scopes. "
              "Do not sum this table into an end-to-end total or GPU execution time.", "",
              "## Request and scheduling metrics", "", "```json",
              json.dumps({k: v for k, v in result["stats"].items() if k != "raw_samples"}, indent=2),
              "```", "", "## Interpretation", "",
              "- queue_wait ends at execution activation, including prefetched waiting time.",
              "- runner_execute includes host dispatch, output ownership and the configured CUDA synchronization.",
              "- output_ownership isolates the engine's output clone/split from the runner's own output stage.",
              "- completion measures result delivery on the event loop; request_e2e ends when the future is resolved.",
              "- GPU kernel activity, API launch gaps and CPU scheduling stalls require the matching Nsight report.",
              "- Compare the same trace with single/double buffering and Graph off/on before attributing a bottleneck.",
              "- Truncated traces or active requests invalidate phase-count analysis.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    path = args.directory / "report.md"
    path.write_text(summarize(args.directory), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
