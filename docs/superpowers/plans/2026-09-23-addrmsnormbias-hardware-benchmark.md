# AddRmsNormBias Hardware Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在空工程模板中实现正确的 AddRmsNormBias、四类算子微基准、可复现的设备计时和数据分析，并产出根目录可单独提交的优化版 `kernel.asc`。

**Architecture:** 根目录保留比赛直接调用工程和最终 `kernel.asc`。`benchmarks/` 同时编译冻结基线、优化内核与四类微内核，由一个 ACL 主程序在同一套事件计时规则下运行；Python 工具负责生成 golden、验证结果、检查 CSV 契约并分析瓶颈。首轮优化只采用不改变 FP32 累加及归约顺序的改动，后续优化由硬件 CSV 决定。

**Tech Stack:** Ascend C、CANN ACL Runtime、CMake 3.16+、C++14、Bash、Python 3、NumPy、ml_dtypes、unittest。

**Spec:** `docs/superpowers/specs/2026-09-23-addrmsnormbias-hardware-benchmark-design.md`

## Global Constraints

- 支持 FP16、BF16、FP32，输入、残差、gamma、bias 和输出类型一致。
- 输入 rank 为 2、3 或 4；前置维度展平为 rows；最后一维 `D` 在 `[64, 32768]`。
- FP16/BF16 必须转 FP32 完成平方、归约、归一化和仿射运算；写回时才转回输入类型。
- FP32 的相对误差和绝对误差均小于 `1e-4`；FP16/BF16 均小于 `1e-3`。
- 非 32 字节对齐的尾块必须只读写逻辑范围，不能改动哨兵区。
- 设备计时必须使用同一流上的 ACL Event；分配、初始化、H2D、D2H、文件 I/O 和校验不进入计时区间。
- 默认预热 100 次、测量 1000 次、采集 10 个批次；长用例每个批次至少 20 次。
- `kernel.asc` 必须保留模板要求的 `run_kernel` 签名，并可作为唯一比赛提交文件。
- 近似倒平方根和 FP16/BF16 累加不进入首轮实现。

## Review Focus

1. `D=4095/4097` 的最后一次传输不能越界或污染哨兵；Task 2 的随机 golden 测试和 Task 7 的设备哨兵检查覆盖该问题。
2. 只有 1 行、`D=64` 时计时不能被零持续时间或整数除法破坏；Task 6 的计时统计测试覆盖该问题。
3. CSV 缺列、重复身份行或非有限时延必须被拒绝；Task 4 的失败输入测试覆盖该问题。
4. 无 CANN 环境、无设备或 BF16 Python 依赖缺失时必须给出明确错误；Task 2 和 Task 7 的命令行测试覆盖该问题。
5. NaN、Inf 和全零行必须保持数学传播且不崩溃，同时不能进入吞吐汇总；Task 2 的 reference 测试、Task 4 的过滤测试和 Task 7 的设备验证覆盖该问题。

---

### Task 1: 固化基准用例与 CSV 契约

**Files:**
- Create: `benchmarks/operator_cases.csv`
- Create: `benchmarks/benchmark_schema.py`
- Create: `scripts/test_benchmark_contract.py`

**Interfaces:**
- Consumes: 设计规格中的用例矩阵和 CSV 字段。
- Produces: `BenchmarkCase`、`load_operator_cases(path)`、`RESULT_FIELDS`、`validate_result_rows(rows)`；后续生成器、运行脚本和分析器共用这些名字。

- [ ] **Step 1: 写入失败的契约测试**

在 `scripts/test_benchmark_contract.py` 中创建标准库 `unittest` 测试：

```python
import csv
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认契约模块尚不存在**

Run: `python -m unittest scripts/test_benchmark_contract.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark_schema'`.

- [ ] **Step 3: 实现用例解析和结果校验**

`benchmarks/operator_cases.csv` 使用以下内容：

```csv
rows,hidden,dtypes,purpose
1,64,fp16|bf16|fp32,launch_floor
32768,64,fp16|bf16|fp32,small_row_throughput
1024,192,fp16|bf16|fp32,unaligned_model_width
1024,576,fp16|bf16|fp32,unaligned_large_width
128,4095,fp16|bf16|fp32,left_reduction_boundary
128,4096,fp16|bf16|fp32,aligned_llm_width
128,4097,fp16|bf16|fp32,right_reduction_boundary
32,8192,fp16|bf16|fp32,large_row
8,32768,fp16|bf16|fp32,max_hidden
```

在 `benchmarks/benchmark_schema.py` 定义：

```python
from dataclasses import dataclass
import csv
import math

DTYPES = {"fp16", "bf16", "fp32"}
RESULT_FIELDS = (
    "suite", "variant", "dtype", "rows", "hidden", "elements",
    "device", "soc", "vector_cores", "warmup", "iterations",
    "latency_us_min", "latency_us_p50", "latency_us_p90",
    "logical_bytes", "effective_gbps", "gelements_per_s",
    "checksum", "status",
)

@dataclass(frozen=True)
class BenchmarkCase:
    rows: int
    hidden: int
    dtypes: tuple[str, ...]
    purpose: str

def load_operator_cases(path):
    cases = []
    with open(path, newline="", encoding="utf-8") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            rows, hidden = int(row["rows"]), int(row["hidden"])
            dtypes = tuple(row["dtypes"].split("|"))
            problems = []
            if rows <= 0:
                problems.append("rows must be positive")
            if not 64 <= hidden <= 32768:
                problems.append("hidden must be in [64, 32768]")
            if not dtypes or not set(dtypes) <= DTYPES:
                problems.append("dtype must be fp16, bf16, or fp32")
            if problems:
                raise ValueError(f"line {line}: {'; '.join(problems)}")
            cases.append(BenchmarkCase(rows, hidden, dtypes, row["purpose"]))
    return cases

def validate_result_rows(rows):
    identities = set()
    for index, row in enumerate(rows, start=1):
        missing = [field for field in RESULT_FIELDS if field not in row]
        if missing:
            raise ValueError(f"row {index}: missing columns {missing}")
        if int(row["iterations"]) <= 0:
            raise ValueError(f"row {index}: iterations must be positive")
        for field in ("latency_us_min", "latency_us_p50", "latency_us_p90"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"row {index}: {field} must be finite")
        identity = tuple(row[field] for field in
                         ("suite", "variant", "dtype", "rows", "hidden"))
        if identity in identities:
            raise ValueError(f"row {index}: duplicate identity {identity}")
        identities.add(identity)
```

- [ ] **Step 4: 运行契约测试**

Run: `python -m unittest scripts/test_benchmark_contract.py -v`

Expected: 3 tests PASS.

- [ ] **Step 5: 提交契约**

```bash
git add benchmarks/operator_cases.csv benchmarks/benchmark_schema.py scripts/test_benchmark_contract.py
git commit -m "test: define benchmark data contracts"
```

### Task 2: 扩展 golden、数据生成和误差验证

**Files:**
- Modify: `scripts/AddRmsNormBias.py`
- Modify: `scripts/gen_data.py`
- Modify: `scripts/verify_result.py`
- Create: `scripts/test_reference.py`

**Interfaces:**
- Consumes: `BenchmarkCase` 和 dtype 名称 `fp16|bf16|fp32`。
- Produces: `resolve_dtype(name)`、`generate_case(rows, hidden, dtype_name, epsilon, seed, directory, pattern="random")`、`compare_arrays(actual, golden, dtype_name)`；运行脚本通过这些 CLI 生成并验证每个硬件用例。

- [ ] **Step 1: 写入 reference 与 CLI 的失败测试**

`scripts/test_reference.py` 覆盖数学公式、特殊值和依赖错误：

```python
import pathlib
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from AddRmsNormBias import impl
from gen_data import generate_case, resolve_dtype
from verify_result import compare_arrays

class ReferenceTest(unittest.TestCase):
    def test_fp32_matches_direct_formula(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]], np.float32)
        r = np.array([[0.1, 0.2, 0.3, 0.4]], np.float32)
        g = np.ones(4, np.float32)
        b = np.full(4, 0.5, np.float32)
        y = x + r
        expected = y / np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + 1e-5) + b
        np.testing.assert_allclose(impl(x, r, g, b), expected, rtol=1e-6, atol=1e-6)

    def test_zero_nan_and_inf_do_not_raise(self):
        for row in (np.zeros((1, 64), np.float32),
                    np.full((1, 64), np.nan, np.float32),
                    np.full((1, 64), np.inf, np.float32)):
            out = impl(row, np.zeros_like(row), np.ones(64, np.float32),
                       np.zeros(64, np.float32))
            self.assertEqual(row.shape, out.shape)

    def test_generator_writes_boundary_case_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = generate_case(2, 4097, "fp16", 1e-5, 42, pathlib.Path(tmp))
            self.assertEqual(8194, np.fromfile(pathlib.Path(tmp) / "x.bin",
                                              dtype=np.float16).size)
            self.assertEqual(4097, meta["hidden"])
            self.assertTrue((pathlib.Path(tmp) / "golden_output.bin").is_file())

    def test_compare_arrays_uses_competition_thresholds(self):
        golden = np.array([1.0, 2.0], np.float32)
        self.assertTrue(compare_arrays(golden + 5e-5, golden, "fp32")[0])
        self.assertFalse(compare_arrays(golden + 2e-3, golden, "fp32")[0])

    def test_unknown_dtype_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "fp16, bf16, or fp32"):
            resolve_dtype("int8")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认新接口缺失**

Run: `python -m unittest scripts/test_reference.py -v`

Expected: FAIL because `generate_case`, `resolve_dtype`, and `compare_arrays` are not defined.

- [ ] **Step 3: 实现确定性数据生成与动态验证**

保留 `impl` 的 FP32 内部计算，并使用 `np.errstate(invalid="ignore", divide="ignore")` 允许 NaN/Inf 自然传播。`gen_data.py` 增加 `argparse` 参数：

```text
--rows INT --hidden INT --dtype {fp16,bf16,fp32}
--epsilon FLOAT --seed INT --output-dir PATH
--pattern {random,zero,nan,inf}
```

`generate_case` 用 `numpy.random.default_rng(seed)` 生成：`x/residual` 范围 `[-2,2]`、`gamma` 范围 `[0.8,1.2]`、`bias` 范围 `[-0.3,0.3]`，写入五个二进制文件和 `metadata.json`。`zero` 使用全零 x/residual、全一 gamma 和全零 bias；`nan`/`inf` 将 x 填为对应特殊值，其他张量与 zero 相同。当 dtype 为 BF16 且 `ml_dtypes` 不存在时抛出：

```python
RuntimeError("bf16 requires the Python package ml_dtypes")
```

`compare_arrays` 返回 `(passed, metrics)`，其中 `metrics` 包含 `max_abs`、`max_rel` 和 `mismatch_count`；FP32 使用 `1e-4`，FP16/BF16 使用 `1e-3`，并以 `equal_nan=True` 比较。

- [ ] **Step 4: 运行 Python 测试及 CLI 冒烟测试**

Run:

```bash
python -m unittest scripts/test_reference.py scripts/test_benchmark_contract.py -v
python scripts/gen_data.py --rows 2 --hidden 4097 --dtype fp16 --epsilon 1e-5 --seed 42 --output-dir build/smoke_case
python scripts/verify_result.py --actual build/smoke_case/golden_output.bin --golden build/smoke_case/golden_output.bin --dtype fp16
```

Expected: all unit tests PASS; verifier prints `PASS`, zero mismatches and zero maximum error.

- [ ] **Step 5: 提交 reference 工具**

```bash
git add scripts/AddRmsNormBias.py scripts/gen_data.py scripts/verify_result.py scripts/test_reference.py
git commit -m "feat: add reproducible operator test data"
```

### Task 3: 实现冻结基线和首轮优化内核

**Files:**
- Create: `benchmarks/kernel_baseline.asc`
- Modify: `kernel.asc`
- Modify: `scripts/test_benchmark_contract.py`

**Interfaces:**
- Consumes: 模板定义的 `TensorInfo`、`TensorGroupInfo`、`GM_ADDR` 和 ACL stream。
- Produces: `run_baseline_kernel(...)` 和模板原有 `run_kernel(...)`；两者接受完全相同的输入并分别启动 `add_rms_norm_bias_baseline` 与 `add_rms_norm_bias_custom`。

- [ ] **Step 1: 添加源代码接口失败测试**

在 `scripts/test_benchmark_contract.py` 增加：

```python
    def test_kernel_entry_points_and_fp32_reduction_contract(self):
        optimized = (ROOT / "kernel.asc").read_text(encoding="utf-8")
        baseline = (ROOT / "benchmarks" / "kernel_baseline.asc").read_text(encoding="utf-8")
        self.assertIn('extern "C" void run_kernel(', optimized)
        self.assertIn('extern "C" void run_baseline_kernel(', baseline)
        self.assertIn("LocalTensor<float>", optimized)
        self.assertNotRegex(optimized, r"ReduceSum\s*<\s*(half|bfloat16_t)")
        self.assertIn("DataCopyPad", optimized)
```

- [ ] **Step 2: 运行测试并确认基线文件缺失**

Run: `python -m unittest scripts/test_benchmark_contract.py -v`

Expected: FAIL with `FileNotFoundError` for `benchmarks/kernel_baseline.asc`.

- [ ] **Step 3: 写入正确基线**

`kernel_baseline.asc` 使用 `KernelAddRmsNormBias<T>`：`Init` 按 `rowCount / blockNum` 加余数将连续行分给各 block，绑定五个 GlobalTensor，并为 x、residual、gamma、bias、output 各分配一个 `4096 * sizeof(T)` TBuf，另分配三个 4096 元素 FP32 TBuf。`ProcessRow` 第一遍按 4096 切片，完成 x/residual 搬入、FP32 转换、Add、平方和 `ReduceSum<float>`，按列顺序累加每个 tile 的 scalar sum；随后计算 `1 / sqrt(sum / D + epsilon)`。第二遍重新搬入 x/residual/gamma/bias，完成 FP32 Add、scale、Mul、bias Add 和 dtype 转换后写回。`D <= 4096` 时保留第一次计算得到的 y，并在 scale 得到后直接完成仿射和写回。所有非对齐 tile 用零填充 `DataCopyPad`，归约和 vector 指令使用对齐元素数。

作以下符号隔离：

```cpp
namespace AddRmsNormBiasBaseline { /* baseline templates */ }

extern "C" __global__ __vector__ void add_rms_norm_bias_baseline(
    GM_ADDR x, GM_ADDR residual, GM_ADDR gamma, GM_ADDR bias, GM_ADDR output,
    uint64_t rowCount, uint32_t hiddenSize, uint32_t tileElements,
    float epsilon, float reciprocalD, int32_t dtype);

extern "C" void run_baseline_kernel(
    GM_ADDR x, const TensorGroupInfo& info_x,
    GM_ADDR residual, const TensorGroupInfo& info_residual,
    GM_ADDR gamma, const TensorGroupInfo& info_gamma,
    GM_ADDR bias, const TensorGroupInfo& info_bias,
    GM_ADDR output, const TensorGroupInfo& info_output,
    int64_t availableCoreNum, aclrtStream stream, float epsilon);
```

基线固定采用 4096 元素 tile、FP32 平方/归约、每行顺序处理和 padded tail，不再随优化版变动。

- [ ] **Step 4: 实现首轮优化版 `kernel.asc`**

从同一正确实现起步，落实以下可审计差异：

```cpp
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t REDUCE_CHUNK_ELEMENTS = 4096;
constexpr uint32_t HALF_IO_TILE_ELEMENTS = 8192;
constexpr uint32_t FLOAT_IO_TILE_ELEMENTS = 4096;

__aicore__ inline uint32_t ReductionWorkspaceElements(uint32_t count) {
    const uint32_t partials = (count + 63U) / 64U;
    return (partials + 7U) & ~7U;
}
```

UB 预算按同一时刻存活对象计算：五个输入/输出 `T` tile、两个 FP32 计算 tile、一个 `ReductionWorkspaceElements(tile)` 工作区。若静态预算超过目标 SoC 可用 UB，host 端将 half/bfloat16 tile 回退到 4096。FP16/BF16 可以一次搬入 8192 元素，但平方和仍按两个最多 4096 元素的连续子视图依次归约，确保 scalar 累加顺序不变。

数据路径满足：

```cpp
if (valid == aligned) {
    AscendC::DataCopy(local, global[offset], aligned);
} else {
    AscendC::DataCopyPad(local, global[offset], copyParams, padParams);
}
```

FP32 分支直接 `Add(y, x, residual, count)` 和 `Mul/Add` gamma/bias，不执行乘 1 的表示转换。单 tile 行保留 `x + residual` 的 FP32 local tensor，在归约后直接归一化并写回；多 tile 行维持 4096 元素归约顺序和第二次 GM 读取。

小 `D` 路径仅在 `D <= 256 && rows >= 4 * blockNum` 时启用。每个 block 将 `batchRows = min(remainingRows, tileElements / D)` 个连续行一次搬入 UB，并对整个 batch 执行 x/residual Add；随后通过 `LocalTensor` 子视图逐行做 FP32 平方归约、scale、gamma/bias 和写回。gamma/bias 每个 block 只搬入一次。任何归约都不能跨越行边界。

- [ ] **Step 5: 运行静态契约并在 CANN 主机编译两个入口**

Run locally: `python -m unittest scripts/test_benchmark_contract.py -v`

Run on Ascend host after sourcing CANN:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH=dav-2201
cmake --build build -j
```

Expected: local contract tests PASS; the Ascend compiler accepts FP16, BF16, FP32 templates and both host entry signatures without UB allocation errors.

- [ ] **Step 6: 提交内核基线和首轮优化**

```bash
git add kernel.asc benchmarks/kernel_baseline.asc scripts/test_benchmark_contract.py
git commit -m "feat: add baseline and optimized AddRmsNormBias kernels"
```

### Task 4: 实现结果分析器

**Files:**
- Create: `benchmarks/analyze_results.py`
- Create: `scripts/test_analyze_results.py`

**Interfaces:**
- Consumes: `benchmark_results.csv` 和 `benchmark_schema.validate_result_rows`。
- Produces: `load_results(path)`、`compute_speedups(rows)`、`classify_case(operator_row, micro_rows)`、命令行 Markdown 摘要。

- [ ] **Step 1: 写入分析逻辑失败测试**

`scripts/test_analyze_results.py` 构造内存行，不依赖 NPU：

```python
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from analyze_results import classify_case, compute_speedups, summarize_rows

def row(suite, variant, latency, rows="128", hidden="4096", status="PASS"):
    return {"suite": suite, "variant": variant, "dtype": "fp16",
            "rows": rows, "hidden": hidden,
            "latency_us_p50": str(latency), "status": status,
            "effective_gbps": "100", "gelements_per_s": "10"}

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
```

- [ ] **Step 2: 运行测试并确认分析器尚不存在**

Run: `python -m unittest scripts/test_analyze_results.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'analyze_results'`.

- [ ] **Step 3: 实现速度比、瓶颈分类和 Markdown 输出**

分类顺序固定为：

```python
if operator_us <= launch_us * 1.5:
    return "launch_control", f"operator={operator_us:.3f}us launch={launch_us:.3f}us"
if reduction_rate and vector_rate and reduction_rate < vector_rate * 0.35:
    return "reduction", f"reduction={reduction_rate:.3f} vs vector={vector_rate:.3f} Gelem/s"
if gm_gbps and operator_gbps and operator_gbps < gm_gbps * 0.45:
    return "memory_or_pipeline", f"operator={operator_gbps:.3f} vs gm={gm_gbps:.3f} GB/s"
if p90 > p50 * 1.20:
    return "runtime_variance", f"p90/p50={p90 / p50:.3f}"
return "mixed", "no single microbenchmark dominates"
```

NaN/Inf 专用用例和 `status != PASS` 的行保留在校验报告中，但不参与 speedup 或吞吐聚合。CLI 用法：

```bash
python benchmarks/analyze_results.py benchmark_results.csv --output benchmark_report.md
```

- [ ] **Step 4: 运行分析器测试**

Run: `python -m unittest scripts/test_analyze_results.py scripts/test_benchmark_contract.py -v`

Expected: all tests PASS.

- [ ] **Step 5: 提交分析器**

```bash
git add benchmarks/analyze_results.py scripts/test_analyze_results.py
git commit -m "feat: analyze AddRmsNormBias bottlenecks"
```

### Task 5: 实现四类 Ascend C 微内核

**Files:**
- Create: `benchmarks/micro_kernels.asc`
- Modify: `scripts/test_benchmark_contract.py`

**Interfaces:**
- Consumes: 设备 GM 地址、元素数、行数、hidden、重复次数。
- Produces: `launch_noop_benchmark`、`launch_gm_copy_benchmark`、`launch_vector_benchmark`、`launch_reduction_benchmark` 四个 host wrapper；Task 6 的 runner 只调用这些 wrapper。

- [ ] **Step 1: 写入微内核入口失败测试**

在契约测试增加：

```python
    def test_microbenchmark_entry_points_are_complete(self):
        source = (ROOT / "benchmarks" / "micro_kernels.asc").read_text(encoding="utf-8")
        for name in ("launch_noop_benchmark", "launch_gm_copy_benchmark",
                     "launch_vector_benchmark", "launch_reduction_benchmark"):
            self.assertIn(name, source)
        self.assertIn("DataCopyPad", source)
        self.assertIn("LocalTensor<float>", source)
```

- [ ] **Step 2: 运行测试并确认文件缺失**

Run: `python -m unittest scripts/test_benchmark_contract.py -v`

Expected: FAIL with `FileNotFoundError` for `benchmarks/micro_kernels.asc`.

- [ ] **Step 3: 实现 no-op 与 GM copy 内核**

入口签名固定为：

```cpp
extern "C" void launch_noop_benchmark(GM_ADDR checksum,
                                       aclrtStream stream);
extern "C" void launch_gm_copy_benchmark(GM_ADDR input, GM_ADDR output,
                                          uint64_t elements, int32_t dtype,
                                          int64_t availableCoreNum,
                                          aclrtStream stream);
```

No-op 仅由 block 0 写入 `0x3f800000`。GM copy 按 block 连续分片，32 字节对齐主体使用 `DataCopy`，每个分片最多一个尾块使用 `DataCopyPad`，输出字节数严格等于逻辑元素数。

- [ ] **Step 4: 实现 FP16/FP32 Vector Add/Mul 内核**

固定接口：

```cpp
enum class VectorBenchOp : int32_t { ADD = 0, MUL = 1 };
extern "C" void launch_vector_benchmark(
    GM_ADDR lhs, GM_ADDR rhs, GM_ADDR output, uint64_t elements,
    int32_t dtype, VectorBenchOp op, uint32_t innerRepeats,
    int64_t availableCoreNum, aclrtStream stream);
```

输入在计时 kernel 内只搬入 UB 一次；两个输出 local buffer 交替作为源和目的，Add 或 Mul 形成数据依赖并重复 `innerRepeats` 次；最后一次结果写回 GM。FP16 与 FP32 分开实例化，默认 `innerRepeats=256`。Add 的 rhs 使用 `1e-3`，Mul 的 rhs 使用可表示且接近 1 的有限值，确保 256 次后 checksum 仍有限。每元素操作数按 `elements * innerRepeats` 计入 `gelements_per_s`。

- [ ] **Step 5: 实现 FP32 平方归约微内核**

固定接口：

```cpp
extern "C" void launch_reduction_benchmark(
    GM_ADDR input, GM_ADDR rowSums, uint64_t rows, uint32_t hidden,
    int32_t inputDtype, int64_t availableCoreNum, aclrtStream stream);
```

每个 block 处理若干完整行；每行按 4096 元素顺序转换为 FP32、平方、`ReduceSum<float>`，将一个 FP32 sum 写入 `rowSums[row]`。尾块用零填充，保证 padding 不进入平方和。

- [ ] **Step 6: 运行静态测试并在 Ascend 主机编译**

Run locally: `python -m unittest scripts/test_benchmark_contract.py -v`

Run on Ascend host: `cmake --build build -j`

Expected: static tests PASS; Ascend compiler accepts all dtype branches and host wrappers.

- [ ] **Step 7: 提交微内核**

```bash
git add benchmarks/micro_kernels.asc scripts/test_benchmark_contract.py
git commit -m "feat: add Ascend operator microbenchmarks"
```

### Task 6: 实现统一 ACL Event 计时主程序

**Files:**
- Create: `benchmarks/benchmark_stats.h`
- Create: `benchmarks/benchmark_main.asc`
- Create: `scripts/test_benchmark_stats.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: 两个 operator wrapper、四个 micro wrapper、Task 1 CSV schema、Task 2 输入文件。
- Produces: `add_rms_norm_bias_benchmark` 可执行文件；每次运行向指定 CSV 追加一行标准结果。

- [ ] **Step 1: 写入不依赖 ACL 的统计测试**

`scripts/test_benchmark_stats.cpp`：

```cpp
#include <cassert>
#include <cmath>
#include <vector>
#include "../benchmarks/benchmark_stats.h"

int main() {
    const auto stats = ComputeLatencyStats({4.0, 1.0, 3.0, 2.0, 100.0});
    assert(std::fabs(stats.minimum - 1.0) < 1e-12);
    assert(std::fabs(stats.p50 - 3.0) < 1e-12);
    assert(std::fabs(stats.p90 - 100.0) < 1e-12);
    assert(std::fabs(ToPerLaunchUs(0.001, 1000) - 0.001) < 1e-12);
    bool threw = false;
    try { (void)ToPerLaunchUs(1.0, 0); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}
```

- [ ] **Step 2: 运行测试并确认统计头文件缺失**

Run on any C++14 host:

```bash
c++ -std=c++14 -I. scripts/test_benchmark_stats.cpp -o build/test_benchmark_stats
```

Expected: FAIL because `benchmarks/benchmark_stats.h` is absent.

- [ ] **Step 3: 实现统计函数**

`benchmark_stats.h` 仅包含标准库，并定义：

```cpp
struct LatencyStats { double minimum; double p50; double p90; };
LatencyStats ComputeLatencyStats(std::vector<double> samples);
double ToPerLaunchUs(double elapsedMilliseconds, uint64_t launches);
```

分位数使用排序后的 nearest-rank：`ceil(q * n) - 1` 并夹在合法下标内。`ToPerLaunchUs` 计算 `elapsedMilliseconds * 1000.0 / launches`，launches 为 0 时抛出 `std::invalid_argument`。

- [ ] **Step 4: 实现 CLI、资源管理和事件计时**

`benchmark_main.asc` 支持：

```text
--suite {launch,gm,vector_add,vector_mul,reduction,operator}
--variant {micro,baseline,optimized}
--dtype {fp16,bf16,fp32}
--rows INT --hidden INT --device INT
--shape D0,D1,...,D --validate-only
--warmup INT --iterations INT --input-dir PATH --output PATH
```

用 RAII 类管理 `aclrtStream`、`aclrtEvent`、device/host memory。核心计时函数接口：

```cpp
using LaunchFn = std::function<void()>;
LatencyStats TimeKernel(aclrtStream stream, uint32_t warmup,
                        uint32_t iterations, const LaunchFn& launch) {
    for (uint32_t i = 0; i < warmup; ++i) launch();
    ACL_THROW(aclrtSynchronizeStream(stream));
    std::vector<double> samples;
    for (uint32_t batch = 0; batch < 10; ++batch) {
        ACL_THROW(aclrtRecordEvent(start, stream));
        for (uint32_t i = 0; i < iterations; ++i) launch();
        ACL_THROW(aclrtRecordEvent(end, stream));
        ACL_THROW(aclrtSynchronizeEvent(end));
        float elapsedMs = 0.0f;
        ACL_THROW(aclrtEventElapsedTime(&elapsedMs, start, end));
        samples.push_back(ToPerLaunchUs(elapsedMs, iterations));
        ACL_THROW(aclrtResetEvent(start, stream));
        ACL_THROW(aclrtResetEvent(end, stream));
    }
    return ComputeLatencyStats(samples);
}
```

如果当前 CANN 的 `aclrtResetEvent` 只有单参数重载，则用该版本；不得改变事件记录、同步和耗时换算语义。

Operator suite 从 `input-dir` 读取二进制、在计时前 H2D，并按 variant 调用 `run_baseline_kernel` 或 `run_kernel`。缺少 `--shape` 时使用 `[rows, hidden]`；提供 `--shape` 时要求 2 到 4 个正整数、最后一维等于 hidden、所有前置维度乘积等于 rows，并将该 shape 原样写入 `TensorInfo`。计时结束后 D2H 写 `actual_output.bin`。`--validate-only` 只 launch 一次并同步、写 actual，不创建结果 CSV；它用于 rank、特殊值、确定性和哨兵检查。微基准计时结束后单独读取输出，计算有限 checksum；checksum 不计时。

CSV 仅由主程序写 header 和一行数据；`logical_bytes`、GB/s 和 Gelem/s 使用 `double` 计算，避免小用例整数截断。`p50 <= 0`、checksum 非有限或 ACL 错误时返回非零退出码。

- [ ] **Step 5: 在 CMake 中加入 host 统计测试和 Ascend benchmark target**

```cmake
add_executable(benchmark_stats_test scripts/test_benchmark_stats.cpp)
target_include_directories(benchmark_stats_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(add_rms_norm_bias_benchmark benchmarks/benchmark_main.asc)
target_include_directories(add_rms_norm_bias_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(add_rms_norm_bias_benchmark PRIVATE
    tiling_api register platform unified_dlog dl m graph_base)
target_compile_options(add_rms_norm_bias_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>)
```

benchmark source按顺序 include `../kernel.asc`、`kernel_baseline.asc` 和 `micro_kernels.asc`；基线符号必须已隔离，不能重定义优化版静态函数。

- [ ] **Step 6: 运行 host 统计测试和 Ascend 构建**

Run on any C++14 host:

```bash
c++ -std=c++14 -I. scripts/test_benchmark_stats.cpp -o build/test_benchmark_stats
./build/test_benchmark_stats
```

Run on Ascend host:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH=dav-2201
cmake --build build -j
```

Expected: stats executable exits 0; both `add_rms_norm_bias_custom` and `add_rms_norm_bias_benchmark` build.

- [ ] **Step 7: 提交计时 runner**

```bash
git add CMakeLists.txt benchmarks/benchmark_stats.h benchmarks/benchmark_main.asc scripts/test_benchmark_stats.cpp
git commit -m "feat: add ACL event benchmark runner"
```

### Task 7: 编排 Release 构建、A/B 校验和微基准

**Files:**
- Create: `benchmarks/run_benchmarks.sh`
- Modify: `run.sh`
- Modify: `scripts/test_benchmark_contract.py`

**Interfaces:**
- Consumes: `operator_cases.csv`、数据生成 CLI、benchmark executable 和 verifier。
- Produces: `benchmark_results.csv`、`benchmark_report.md`、每个 operator case 的 actual/golden 文件。

- [ ] **Step 1: 写入 shell 契约失败测试**

在 `scripts/test_benchmark_contract.py` 增加：

```python
    def test_runner_is_release_safe_and_checks_environment(self):
        source = (ROOT / "benchmarks" / "run_benchmarks.sh").read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", source)
        self.assertIn("CMAKE_BUILD_TYPE=Release", source)
        self.assertIn("ASCEND_HOME_PATH", source)
        self.assertIn("operator_cases.csv", source)
        self.assertIn("verify_result.py", source)
        self.assertIn("analyze_results.py", source)
```

- [ ] **Step 2: 运行测试并确认 runner 缺失**

Run: `python -m unittest scripts/test_benchmark_contract.py -v`

Expected: FAIL with `FileNotFoundError` for `benchmarks/run_benchmarks.sh`.

- [ ] **Step 3: 实现完整 benchmark orchestration**

脚本开头固定检查并报告：

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is not set; source the CANN set_env.sh first}"
source "${ASCEND_HOME_PATH}/set_env.sh"
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
```

接受 `--device`、`--warmup`、`--iterations`、`--soc` 和 `--output`。参数解析后执行 `test -e "/dev/davinci${device}"`；不存在时打印所选设备节点并返回非零。构建命令必须是：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH="${soc}"
cmake --build build -j"$(nproc)"
```

执行顺序：

1. 清空目标 CSV。
2. 运行 launch 一次。
3. 对 FP16/FP32 的 64、4095、4096、4097、8192、32768 元素运行 GM 与 vector Add/Mul。
4. 对三种 dtype 和 operator matrix 对应的 rows/hidden 运行 reduction。
5. 对每个 operator case 生成一次数据，先运行 baseline，验证 actual，再运行 optimized，再次验证 actual。
6. 对预计单轮大于 1 ms 的 operator case 使用 `max(20, iterations / 20)`；其他 case 使用原 iterations。
7. 每次 verifier 失败立即退出，失败 case 不写作 `PASS`。
8. 对 `[6,64]` 数据分别用 `--shape 6,64`、`--shape 2,3,64`、`--shape 1,2,3,64` 做 `--validate-only`，验证 rank 2、3、4 的相同展平语义。
9. 对 FP32 `[1,64]` 的 zero、nan、inf pattern 做 `--validate-only` 并与 golden 比较，不向 benchmark CSV 写吞吐行。
10. 每个 optimized operator case 完成计时后再用相同输入执行两次 `--validate-only`，以 `cmp` 比较两个 actual 文件，验证逐字节确定性。
11. 调用 analyzer 生成 Markdown 报告。

设备输出数组多分配 64 字节并在逻辑范围后填充固定哨兵；每次 operator 与 GM case 完成后验证这 64 字节未改变。runner 将哨兵失败写入 stderr 并返回非零。

- [ ] **Step 4: 更新根目录快速正确性脚本**

`run.sh` 使用 Release build，调用新的动态 `gen_data.py` 和 `verify_result.py`，仍保留默认 `rows=1, hidden=64, dtype=fp16`，保证模板最短验证路径继续工作。

- [ ] **Step 5: 运行无 NPU 契约测试和 shell 语法检查**

Run:

```bash
python -m unittest scripts/test_benchmark_contract.py scripts/test_reference.py scripts/test_analyze_results.py -v
bash -n run.sh benchmarks/run_benchmarks.sh
```

Expected: all Python tests PASS and both shell scripts parse successfully. 在没有 `/dev/davinci0` 的 Linux 环境运行 benchmark script 时，应打印明确设备错误并以非零状态退出。

- [ ] **Step 6: 提交运行编排**

```bash
git add run.sh benchmarks/run_benchmarks.sh scripts/test_benchmark_contract.py
git commit -m "feat: orchestrate hardware benchmark matrix"
```

### Task 8: 硬件验收、报告和仓库交付

**Files:**
- Create: `README.md`
- Modify: `kernel.asc` only if measured evidence selects one follow-up change
- Create: `docs/hardware-results.md`

**Interfaces:**
- Consumes: Ascend 设备、完整测试套件和 `benchmark_results.csv`。
- Produces: 可复现实验说明、硬件特征报告、最终已验证的根目录 `kernel.asc`。

- [ ] **Step 1: 编写硬件运行说明**

README 必须给出以下可复制命令：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install --user numpy ml_dtypes
./run.sh
./benchmarks/run_benchmarks.sh --device 0 --soc dav-2201 \
  --warmup 100 --iterations 1000 --output benchmark_results.csv
python3 benchmarks/analyze_results.py benchmark_results.csv \
  --output benchmark_report.md
```

同时说明比赛提交仅需根目录 `kernel.asc`，基准文件用于本地硬件分析。

- [ ] **Step 2: 在硬件上运行完整正确性和 benchmark**

Run:

```bash
./run.sh
./benchmarks/run_benchmarks.sh --device 0 --soc dav-2201 --warmup 100 --iterations 1000
```

Expected: 所有 operator baseline/optimized 行为 `PASS`；rank 2/3/4、zero、NaN、Inf 和逐字节确定性验证通过；微基准 checksum 有限；输出包含 launch、gm、vector_add、vector_mul、reduction、operator 六个 suite；每个默认 operator case 同时具有 baseline 和 optimized 行。

- [ ] **Step 3: 根据报告只选择一个二阶段改动**

按 analyzer 类别执行一项，并以独立提交实现：

- `launch_control`: 扩大小 `D` 每 block 行数，保持每行独立归约。
- `reduction`: 用相同 FP32 输入和顺序对比 `WholeReduceSum`，仅在全部精度 case 通过时保留。
- `memory_or_pipeline`: 为 CopyIn/Compute/CopyOut 引入双缓冲 `TQue`，UB 静态预算必须通过编译。
- `runtime_variance`: 增加单批次 launches 并用 profiler 确认 event/stream 间隙，不直接改数学路径。
- `mixed`: 保持首轮内核，不为不确定收益增加复杂度。

改动后重跑全部 operator 正确性和受影响的 micro suite。只有优化版相关 case 的 p50 改善且任何 case 不超过基线 p50 的 1.02 倍时才保留；否则回退该提交。

- [ ] **Step 4: 记录硬件证据**

`docs/hardware-results.md` 记录：设备型号、SoC、CANN/编译器版本、vector core 数、执行命令、结果 CSV 的 commit、六类 suite 摘要、各 operator speedup、最终保留或回退的二阶段改动及原因。数值直接由 `benchmark_report.md` 转录，不手工重算。

- [ ] **Step 5: 运行最终验证**

Run:

```bash
python3 -m unittest discover -s scripts -p 'test_*.py' -v
bash -n run.sh benchmarks/run_benchmarks.sh
git diff --check
git status --short
```

Run on Ascend host:

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH=dav-2201
cmake --build build -j"$(nproc)"
./run.sh
./benchmarks/run_benchmarks.sh --device 0 --soc dav-2201 --warmup 100 --iterations 1000
```

Expected: local tests and syntax checks PASS; clean Release build succeeds; all correctness rows PASS; analyzer produces a report without schema errors.

- [ ] **Step 6: 提交文档与最终实测内核并推送**

```bash
git add README.md docs/hardware-results.md kernel.asc
git commit -m "docs: record AddRmsNormBias hardware results"
git push origin main
```
