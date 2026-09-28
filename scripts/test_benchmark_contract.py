import math
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_schema import RESULT_FIELDS, load_operator_cases, validate_result_rows


class BenchmarkContractTest(unittest.TestCase):
    def test_default_matrix_contains_boundary_and_small_row_cases(self):
        cases = load_operator_cases(ROOT / "benchmarks" / "operator_cases.csv")
        keys = {(c.rows, c.hidden) for c in cases}
        self.assertTrue({(1, 64), (32768, 64), (128, 4095),
                         (128, 4096), (128, 4097), (8, 32768)} <= keys)
        self.assertEqual({"fp16", "bf16", "fp32"},
                         {dtype for c in cases for dtype in c.dtypes})

    def test_invalid_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "cases.csv"
            path.write_text("rows,hidden,dtypes,purpose\n0,63,int8,bad\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rows.*positive|hidden.*64|dtype"):
                load_operator_cases(path)

    def test_result_contract_rejects_missing_nonfinite_and_duplicate_rows(self):
        valid = {field: "" for field in RESULT_FIELDS}
        valid.update({"suite": "launch", "variant": "micro", "dtype": "",
                      "rows": "1", "hidden": "1", "elements": "1",
                      "device": "0", "soc": "dav-2201", "vector_cores": "40",
                      "warmup": "100", "iterations": "1000",
                      "latency_us_min": "1.0", "latency_us_p50": "1.1",
                      "latency_us_p90": "1.2", "logical_bytes": "4",
                      "effective_gbps": "", "gelements_per_s": "",
                      "checksum": "1", "status": "PASS"})
        validate_result_rows([valid])
        invalid = dict(valid, latency_us_p50=str(math.inf))
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_result_rows([invalid])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_result_rows([valid, dict(valid)])

    def test_kernel_entry_points_and_fp32_reduction_contract(self):
        optimized = (ROOT / "kernel.asc").read_text(encoding="utf-8")
        baseline = (ROOT / "benchmarks" / "kernel_baseline.asc").read_text(
            encoding="utf-8"
        )
        self.assertIn('extern "C" void run_kernel(', optimized)
        self.assertIn('extern "C" void run_baseline_kernel(', baseline)
        self.assertIn("LocalTensor<float>", optimized)
        self.assertNotRegex(optimized, r"ReduceSum\s*<\s*(half|bfloat16_t)")
        self.assertIn("DataCopyPad", optimized)
        self.assertIn("tileElements = tileLimit", optimized)
        self.assertIn("maxBatchRows = tileElements_ / hiddenSize_", optimized)

    def test_row_batch_covers_medium_hidden_sizes_and_two_rows_per_core(self):
        optimized = (ROOT / "kernel.asc").read_text(encoding="utf-8")
        self.assertIn("const bool useRowBatch =", optimized)
        self.assertIn("hiddenSize <= tileLimit / 2", optimized)
        self.assertIn("rowCount >= 2ULL * blockNum", optimized)
        self.assertIn("tileElements_ / hiddenSize_ >= 2", optimized)
        self.assertIn("rowsForCore_ >= 2", optimized)

    def test_row_batch_squares_the_batch_before_reducing_row_views(self):
        optimized = (ROOT / "kernel.asc").read_text(encoding="utf-8")
        batch_mul = "AscendC::Mul(scratchLocal, yLocal, yLocal, batchElements);"
        row_view = "auto squareRow = scratchLocal[rowOffset];"
        self.assertIn(batch_mul, optimized)
        self.assertIn(row_view, optimized)
        self.assertRegex(
            optimized,
            r"AscendC::ReduceSum<float>\(squareRow, squareRow,\s*"
            r"workLocal, hiddenSize_\);",
        )
        self.assertLess(optimized.index(batch_mul), optimized.index(row_view))

    def test_row_batch_converts_gamma_and_bias_once_per_batch(self):
        optimized = (ROOT / "kernel.asc").read_text(encoding="utf-8")
        self.assertIn("inline void ApplyGammaBiasBatch", optimized)
        batch_helper = optimized[
            optimized.index("inline void ApplyGammaBiasBatch"):
            optimized.index("class KernelAddRmsNormBias")
        ]
        row_batch = optimized[
            optimized.index("inline void ProcessSmallRows"):
            optimized.index("inline void WriteSingleTile")
        ]
        self.assertEqual(1, batch_helper.count("ConvertToFloat(tmp, gamma"))
        self.assertEqual(1, batch_helper.count("ConvertToFloat(tmp, bias"))
        self.assertIn("ApplyGammaBiasBatch(yLocal", row_batch)
        self.assertNotIn("ApplyGammaBias(yRow", row_batch)

    def test_row_batch_reuses_parameter_segments_across_rows(self):
        optimized = (ROOT / "kernel.asc").read_text(encoding="utf-8")
        helper = optimized[
            optimized.index("inline void ApplyParameterBatch"):
            optimized.index("class KernelAddRmsNormBias")
        ]
        self.assertIn("BinaryRepeatParams", helper)
        self.assertIn("parameterRepStride = 0", helper)
        self.assertIn("static_cast<uint8_t>(batchRows)", helper)
        self.assertIn("ApplyParameterBatch<true>", optimized)
        self.assertIn("ApplyParameterBatch<false>", optimized)

    def test_row_batch_reduces_and_computes_scales_as_a_batch(self):
        optimized = (ROOT / "kernel.asc").read_text(encoding="utf-8")
        self.assertIn("inline void ComputeBatchScales", optimized)
        batch_scale = optimized[
            optimized.index("inline void ComputeBatchScales"):
            optimized.index("inline void ProcessSmallRows")
        ]
        row_batch = optimized[
            optimized.index("inline void ProcessSmallRows"):
            optimized.index("inline void WriteSingleTile")
        ]
        self.assertIn(
            "workLocal.SetValue(row, squareRow.GetValue(0))", batch_scale
        )
        for operation in ("Muls", "Adds", "Sqrt", "Div"):
            self.assertRegex(
                batch_scale,
                rf"AscendC::{operation}\([^;]*batchRows\);",
            )
        self.assertIn("ComputeBatchScales(scratchLocal, workLocal, batchRows)",
                      row_batch)
        self.assertNotIn("ComputeScale(squareRow.GetValue(0))", row_batch)

    def test_non_batch_tiles_use_hierarchical_hardware_reduction(self):
        optimized = (ROOT / "kernel.asc").read_text(encoding="utf-8")
        reduction = optimized[
            optimized.index("inline float ReduceTileChunk"):
            optimized.index("inline float ReduceTile(")
        ]
        row_tile = optimized[
            optimized.index("inline float ReduceTile("):
            optimized.index("inline float ComputeScale")
        ]
        row_batch = optimized[
            optimized.index("inline void ProcessSmallRows"):
            optimized.index("inline void WriteSingleTile")
        ]
        self.assertIn("BlockReduceSum<float, false>", reduction)
        self.assertIn("WholeReduceSum<float, false>", reduction)
        self.assertIn(
            "workLocal, sourceLocal, 1, AscendC::MASK_PLACEHOLDER",
            reduction,
        )
        self.assertIn(
            "sourceLocal, workLocal, AscendC::MASK_PLACEHOLDER,\n"
            "            1, 1, 1, 8",
            reduction,
        )
        self.assertIn("PipeBarrier<PIPE_V>", reduction)
        self.assertNotIn("ReduceSum<float>", row_tile)
        self.assertIn("ReduceTileChunk(squareChunk, workLocal, count)",
                      row_tile)
        self.assertIn("ReduceSum<float>", row_batch)

    def test_microbenchmark_entry_points_are_complete(self):
        source = (ROOT / "benchmarks" / "micro_kernels.asc").read_text(
            encoding="utf-8"
        )
        for name in ("launch_noop_benchmark", "launch_gm_copy_benchmark",
                     "launch_vector_benchmark", "launch_reduction_benchmark"):
            self.assertIn(name, source)
        self.assertIn("DataCopyPad", source)
        self.assertIn("LocalTensor<float>", source)

        runner = (ROOT / "benchmarks" / "benchmark_main.asc").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("reinterpret_cast<GM_ADDR>", runner)
        self.assertIn("(GM_ADDR)input.get()", runner)

    def test_runner_is_release_safe_and_checks_environment(self):
        source = (ROOT / "benchmarks" / "run_benchmarks.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("set -euo pipefail", source)
        self.assertIn("CMAKE_BUILD_TYPE=Release", source)
        self.assertIn("ASCEND_HOME_PATH", source)
        self.assertIn("operator_cases.csv", source)
        self.assertIn("verify_result.py", source)
        self.assertIn("analyze_results.py", source)
        self.assertIn('compgen -G "/dev/davinci[0-9]*"', source)
        self.assertNotIn('[[ -e "/dev/davinci${device}" ]]', source)

        runner = (ROOT / "benchmarks" / "benchmark_main.asc").read_text(
            encoding="utf-8"
        )
        self.assertIn("aclrtGetDeviceCount", runner)
        self.assertIn("logical device index", runner)

        focus = (ROOT / "benchmarks" / "run_focus.sh").read_text(
            encoding="utf-8"
        )
        for shape in ("32768 64", "1024 192", "1024 576", "128 4096",
                      "8 32768"):
            self.assertIn(shape, focus)
        self.assertIn("verify_result.py", focus)
        self.assertIn("baseline optimized", focus)

    def test_readme_documents_hardware_run_and_submission_artifact(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("./run.sh", readme)
        self.assertIn("./benchmarks/run_benchmarks.sh", readme)
        self.assertIn("--warmup 100 --iterations 1000", readme)
        self.assertIn("benchmark_results.csv", readme)
        self.assertIn("kernel.asc", readme)


if __name__ == "__main__":
    unittest.main()
