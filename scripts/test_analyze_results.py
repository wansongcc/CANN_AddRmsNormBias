import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from analyze_results import classify_case, compute_speedups, summarize_rows


def row(suite, variant, latency, rows="128", hidden="4096", status="PASS"):
    return {"suite": suite, "variant": variant, "dtype": "fp16",
            "rows": rows, "hidden": hidden,
            "latency_us_p50": str(latency), "latency_us_p90": str(latency),
            "status": status, "effective_gbps": "100",
            "gelements_per_s": "10"}


class AnalyzeResultsTest(unittest.TestCase):
    def test_speedup_pairs_only_passing_variants(self):
        values = compute_speedups([row("operator", "baseline", 20),
                                   row("operator", "optimized", 10)])
        self.assertEqual(2.0, values[0]["speedup"])
        self.assertEqual([], compute_speedups([row("operator", "baseline", 20),
                                               row("operator", "optimized", 10,
                                                   status="FAIL")]))

    def test_launch_limited_classification(self):
        category, evidence = classify_case(
            row("operator", "optimized", 4, rows="1", hidden="64"),
            [row("launch", "micro", 3.5, rows="1", hidden="1")])
        self.assertEqual("launch_control", category)
        self.assertIn("3.5", evidence)

    def test_reduction_limited_classification(self):
        micro = [row("launch", "micro", 1),
                 dict(row("vector_mul", "micro", 2), gelements_per_s="20"),
                 dict(row("reduction", "micro", 8), gelements_per_s="4")]
        self.assertEqual("reduction", classify_case(
            row("operator", "optimized", 20), micro)[0])

    def test_nonfinite_or_measure_only_rows_are_not_summarized(self):
        text = summarize_rows([row("operator", "optimized", 10,
                                   status="MEASURE_ONLY")])
        self.assertNotIn("speedup", text.lower())


if __name__ == "__main__":
    unittest.main()
