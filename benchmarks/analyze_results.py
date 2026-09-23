#!/usr/bin/env python3
"""Validate AddRmsNormBias benchmark CSV and summarize bottlenecks."""

import argparse
import csv
import math
import pathlib

from benchmark_schema import validate_result_rows


def _number(row, field):
    value = row.get(field, "")
    return float(value) if value not in (None, "") else 0.0


def load_results(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_result_rows(rows)
    return rows


def compute_speedups(rows):
    pairs = {}
    for item in rows:
        if item.get("suite") != "operator" or item.get("status") != "PASS":
            continue
        key = (item.get("dtype", ""), item.get("rows", ""),
               item.get("hidden", ""))
        pairs.setdefault(key, {})[item.get("variant", "")] = item

    speedups = []
    for key, variants in sorted(pairs.items()):
        if not {"baseline", "optimized"} <= variants.keys():
            continue
        baseline = _number(variants["baseline"], "latency_us_p50")
        optimized = _number(variants["optimized"], "latency_us_p50")
        if baseline <= 0 or optimized <= 0:
            continue
        speedups.append({
            "dtype": key[0],
            "rows": key[1],
            "hidden": key[2],
            "baseline_us": baseline,
            "optimized_us": optimized,
            "speedup": baseline / optimized,
        })
    return speedups


def _best_micro(rows, suite):
    candidates = [item for item in rows
                  if item.get("suite") == suite and item.get("status") == "PASS"]
    return max(candidates, key=lambda item: int(item.get("elements", "0") or 0),
               default=None)


def classify_case(operator_row, micro_rows):
    operator_us = _number(operator_row, "latency_us_p50")
    p50 = operator_us
    p90 = _number(operator_row, "latency_us_p90") or p50
    launch = _best_micro(micro_rows, "launch")
    reduction = _best_micro(micro_rows, "reduction")
    vector = _best_micro(micro_rows, "vector_mul")
    gm = _best_micro(micro_rows, "gm")

    launch_us = _number(launch, "latency_us_p50") if launch else 0.0
    reduction_rate = _number(reduction, "gelements_per_s") if reduction else 0.0
    vector_rate = _number(vector, "gelements_per_s") if vector else 0.0
    gm_gbps = _number(gm, "effective_gbps") if gm else 0.0
    operator_gbps = _number(operator_row, "effective_gbps")

    if launch_us and operator_us <= launch_us * 1.5:
        return ("launch_control",
                f"operator={operator_us:.3f}us launch={launch_us:.3f}us")
    if reduction_rate and vector_rate and reduction_rate < vector_rate * 0.35:
        return ("reduction",
                f"reduction={reduction_rate:.3f} vs vector={vector_rate:.3f} Gelem/s")
    if gm_gbps and operator_gbps and operator_gbps < gm_gbps * 0.45:
        return ("memory_or_pipeline",
                f"operator={operator_gbps:.3f} vs gm={gm_gbps:.3f} GB/s")
    if p50 and p90 > p50 * 1.20:
        return "runtime_variance", f"p90/p50={p90 / p50:.3f}"
    return "mixed", "no single microbenchmark dominates"


def summarize_rows(rows):
    passing = [item for item in rows if item.get("status") == "PASS"]
    operator_rows = [item for item in passing
                     if item.get("suite") == "operator"
                     and item.get("variant") == "optimized"]
    if not passing:
        return "# AddRmsNormBias Benchmark Report\n\nNo passing measurements.\n"

    lines = ["# AddRmsNormBias Benchmark Report", "",
             f"Passing rows: {len(passing)}", ""]
    speedups = compute_speedups(rows)
    if speedups:
        lines.extend([
            "## Operator A/B",
            "",
            "| dtype | rows | hidden | baseline us | optimized us | speedup |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for item in speedups:
            lines.append(
                f"| {item['dtype']} | {item['rows']} | {item['hidden']} | "
                f"{item['baseline_us']:.3f} | {item['optimized_us']:.3f} | "
                f"{item['speedup']:.3f}x |"
            )
        lines.append("")

    if operator_rows:
        lines.extend(["## Bottleneck Classification", "",
                      "| dtype | rows | hidden | category | evidence |",
                      "| --- | ---: | ---: | --- | --- |"])
        micro_rows = [item for item in passing if item.get("suite") != "operator"]
        for item in operator_rows:
            category, evidence = classify_case(item, micro_rows)
            lines.append(f"| {item['dtype']} | {item['rows']} | {item['hidden']} | "
                         f"{category} | {evidence} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("benchmark_report.md"))
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_results(args.input)
    report = summarize_rows(rows)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
