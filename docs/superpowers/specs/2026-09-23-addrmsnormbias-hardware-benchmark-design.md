# AddRmsNormBias Hardware Benchmark Design

Date: 2026-09-23

## 1. Goal

This project will turn the supplied empty Ascend C template into a reproducible AddRmsNormBias optimization workspace for Ascend hardware. It will provide:

- a correct baseline implementation for comparison;
- an optimized submission kernel at the repository root as `kernel.asc`;
- operator-level latency and accuracy measurements;
- four focused microbenchmarks that isolate launch, global-memory movement, vector arithmetic, and reduction behavior;
- machine-readable results and an analyzer that maps measurements to optimization choices.

The first optimization pass prioritizes numerical stability and low-risk memory-path improvements. More aggressive scheduling changes will be selected from hardware evidence instead of assumed hardware characteristics.

## 2. Scope and Constraints

The operator computes, for each row whose final dimension is `D`:

1. `y = x + residual`
2. `scale = 1 / sqrt(mean(y * y) + epsilon)`
3. `output = y * scale * gamma + bias`

The supported input data types are FP16, BF16, and FP32. FP16 and BF16 inputs are converted to FP32 for squaring, reduction, normalization, and affine arithmetic. The result is cast to the input type only when written to global memory. FP32 accumulation and the established reduction order are retained unless hardware measurements and accuracy tests justify a different path.

Input ranks from two through four are flattened into `rows = numel / D`; normalization remains independent per row. The implementation must support `D` in `[64, 32768]`, including final dimensions that do not end on a 32-byte boundary.

The microbenchmarks describe the tested device and compiler configuration. They are diagnostics rather than a synthetic replacement for the full operator benchmark.

## 3. Repository Layout

The supplied template remains the project root so the competition submission path stays obvious.

| Path | Purpose |
| --- | --- |
| `kernel.asc` | Optimized competition kernel and required `run_kernel` entry point |
| `main.asc` | Existing single-case correctness runner, updated only as required by the template interface |
| `scripts/` | Existing golden-data and result-verification utilities |
| `benchmarks/kernel_baseline.asc` | Frozen correct baseline used by A/B tests |
| `benchmarks/micro_kernels.asc` | Launch, GM copy, vector, and reduction kernels |
| `benchmarks/benchmark_main.asc` | ACL runtime host runner and device-event timing |
| `benchmarks/operator_cases.csv` | Default operator test matrix |
| `benchmarks/run_benchmarks.sh` | Release build and benchmark orchestration |
| `benchmarks/analyze_results.py` | Result validation, summaries, and bottleneck classification |
| `scripts/test_benchmark_contract.py` | Hardware-independent checks for cases, CSV schema, and source contracts |
| `benchmark_results.csv` | Generated raw measurements; ignored by Git |

The baseline and optimized kernels will be separate build targets so both binaries can be measured in one build without copying files between runs. The competition artifact remains only the root `kernel.asc`.

## 4. Measurement Method

### 4.1 Build and Environment

Benchmarks use a Release build for the selected SoC version. Each result records the SoC name, compiler version when available, device identifier, vector-core count reported by the runtime, data type, shape, warmup count, iteration count, and kernel variant.

The runner accepts:

- `--device`, default `0`;
- `--warmup`, default `100`;
- `--iterations`, default `1000`;
- `--output`, default `benchmark_results.csv`.

Long operator cases may lower the number of measured iterations to keep a run practical, but the effective count is always written to the CSV. The default script keeps at least 20 measured launches for every case.

### 4.2 Device Timing

All reported kernel latency comes from ACL device events on one stream. Inputs and persistent workspaces are allocated and initialized before warmup. Host-to-device setup, device-to-host validation copies, file I/O, allocation, and first-use initialization stay outside the timed region.

For a sample batch, the runner records a start event, launches the same kernel repeatedly, records an end event, synchronizes the end event, and divides elapsed device time by the launch count. Ten sample batches are collected after warmup. CSV output reports minimum, median (`p50`), and 90th-percentile (`p90`) per-launch latency. Performance comparisons use `p50`; `p90` exposes instability.

The no-op benchmark estimates the fixed launch floor with the same event and batching procedure. It is not subtracted from operator latency because launch and execution can overlap differently across kernels.

### 4.3 Result Schema

Every CSV row uses this schema:

```text
suite,variant,dtype,rows,hidden,elements,device,soc,vector_cores,warmup,iterations,latency_us_min,latency_us_p50,latency_us_p90,logical_bytes,effective_gbps,gelements_per_s,checksum,status
```

Unavailable fields are empty. `status` is `PASS`, `FAIL`, or `MEASURE_ONLY`. The analyzer rejects missing columns, non-finite nonempty timing values, nonpositive iteration counts, and duplicate identity rows.

## 5. Microbenchmarks

### 5.1 Launch Latency

The launch kernel performs the smallest legal observable operation: one selected block writes a fixed scalar to a device result buffer. The checksum verifies that the launch was executed. The primary metric is device-event latency in microseconds per launch.

This measurement identifies shapes for which fixed dispatch cost dominates and provides the evidence for batching multiple rows in one block.

### 5.2 Global-Memory Movement

The GM kernel copies an input buffer through local memory to a distinct output buffer. It uses the same aligned fast path and padded tail handling intended for the operator. The result is protected by a device-side checksum sampled after timing.

Effective bandwidth is calculated as:

```text
effective_gbps = (input_bytes + output_bytes) / latency_seconds / 1e9
```

The benchmark sweeps sizes that fit one row and sizes large enough to occupy all vector cores. It includes aligned byte counts and the FP16, BF16, and FP32 tails produced by `D = 4095`, `4096`, and `4097`.

### 5.3 Vector Arithmetic

The vector benchmark measures FP16 and FP32 Add and Mul separately. Input is copied to local memory before the timed arithmetic loop, the operation is repeated enough times to dominate fixed overhead, and one result is written out after the loop. The runner divides total operation count by elapsed time and reports giga-elements per second.

The benchmark uses finite deterministic inputs and emits a checksum so the compiler cannot remove the arithmetic. BF16 operator behavior is represented by FP32 compute after BF16 conversion; conversion cost remains visible in the full operator benchmark.

### 5.4 Reduction

The reduction kernel loads rows, squares in FP32, and reduces each row to one FP32 sum. It uses the same row lengths and reduction chunking as the operator. Results report microseconds per row and giga-elements per second, with a checksum over row sums.

This benchmark compares the current block-reduction path against any later `WholeReduceSum` or hierarchical alternative. An alternative is accepted only when it passes the operator accuracy thresholds on all supported data types and tested row lengths.

## 6. Operator Benchmark Matrix

The checked-in default matrix contains these logical shapes for FP16, BF16, and FP32 when the toolchain supports the type:

| Rows | D | Purpose |
| ---: | ---: | --- |
| 1 | 64 | Launch and scalar-control floor |
| 32768 | 64 | Small-row throughput and multi-row scheduling |
| 1024 | 192 | Common non-32-element hidden size |
| 1024 | 576 | Larger non-32-element hidden size |
| 128 | 4095 | Left tail around a major reduction boundary |
| 128 | 4096 | Common LLM hidden size and aligned reference |
| 128 | 4097 | Right tail around a major reduction boundary |
| 32 | 8192 | Large-row bandwidth and reduction pressure |
| 8 | 32768 | Maximum hidden dimension and UB tiling |

The runner may add competition cases without changing the checked-in defaults. Every operator row records both baseline and optimized latency. Speedup is computed as `baseline_p50 / optimized_p50` only when both variants pass correctness.

Logical operator bandwidth uses the minimum tensor traffic visible at the interface:

```text
logical_bytes = bytes(x) + bytes(residual) + bytes(gamma) + bytes(bias) + bytes(output)
```

It is labeled logical bandwidth because an implementation can reread tensors or spill intermediates.

## 7. Correctness and Determinism

The existing PyTorch golden implementation remains the numerical reference. Operator checks cover every benchmark shape before its timing is accepted. The limits are:

- FP32: relative error below `1e-4` and absolute error below `1e-4`;
- FP16/BF16: relative error below `1e-3` and absolute error below `1e-3`.

Each case is run repeatedly with identical inputs to verify deterministic output. Dedicated correctness cases cover nonaligned tails, zero-valued rows, NaN propagation, and Inf execution without crashes. NaN and Inf cases are excluded from throughput reporting.

Microbenchmarks use deterministic inputs and checksums to catch skipped execution, out-of-bounds tails, or compiler elimination. Checksums are validation guards and are not timed.

## 8. Initial Optimized Kernel

The first optimized `kernel.asc` will retain FP32 numerical work and the proven reduction ordering while reducing avoidable data movement and synchronization:

1. Size the FP32 reduction workspace from the number of 64-element partial sums, rounded only to the required local-memory alignment, instead of reserving a full-tile-sized buffer.
2. Use ordinary `DataCopy` for transfer lengths that satisfy the Ascend C alignment requirements. Use `DataCopyPad` only for a final partial transfer and mask stores so no tail writes escape the logical row.
3. Reuse the local FP32 representation of `x + residual` for square reduction and final normalization, avoiding a second GM read or intermediate GM tensor.
4. Remove FP32 identity casts and copies that do not change representation or ownership.
5. Increase FP16/BF16 I/O tile size only after the static UB budget accounts for every simultaneously live local tensor and queue buffer.
6. Preserve the existing 4096-element reduction chunk order in the initial version.
7. Add a guarded small-`D` multi-row path when rows are numerous enough to amortize launch and scalar control. Each block processes consecutive independent rows; the generic one-row path remains the fallback.

The initial kernel will not use approximate reciprocal square root and will not accumulate squares in FP16 or BF16.

## 9. Evidence-Driven Follow-Up

The analyzer compares each operator case with the four microbenchmarks and produces a suggested next action:

| Evidence | Interpretation | Candidate change |
| --- | --- | --- |
| Operator latency is close to no-op latency for small `D` | Launch/control limited | Process multiple rows per block and reduce per-row scalar setup |
| Logical bandwidth is low while GM copy reaches a high fraction of expected device bandwidth | Extra traffic or pipeline stalls | Inspect repeated loads, cache gamma/bias in UB when capacity permits, and add queue overlap |
| Reduction throughput is far below vector Add/Mul throughput | Reduction limited | Benchmark `WholeReduceSum` or a hierarchical reduction with the same FP32 order constraints |
| Large aligned cases improve but tail cases regress | Tail handling overhead | Split aligned main loop from one padded tail without per-tile branching |
| `p90` is much larger than `p50` | Runtime or synchronization instability | Increase work per sample batch and profile stream/event gaps |
| Vector microbenchmark is strong but the full operator remains slow | Scheduling or synchronization limited | Introduce `TQue` double buffering and verify CopyIn/Compute/CopyOut overlap |

Follow-up kernel changes are made one at a time and retained only when the affected correctness suite passes and representative A/B cases improve. Hardware profiler traces are used when available to distinguish vector-core occupancy, GM stalls, and synchronization gaps.

## 10. Two-Stage Workflow

Stage one delivers the template-based repository, frozen baseline, microbenchmark suite, result analyzer, and the conservative optimized kernel. The hardware owner runs `benchmarks/run_benchmarks.sh` and retains the generated CSV together with compiler and device metadata.

Stage two uses the CSV and profiler evidence to choose one bottleneck-specific change. The same cases are rerun against the unchanged baseline, and the root `kernel.asc` is updated only after accuracy and performance acceptance.

## 11. Risks and Mitigations

- **Compiler/API differences:** Ascend C intrinsic signatures and ACL event APIs vary by CANN version. The build keeps platform-specific code localized and records the compiler version with results.
- **Misleading timing:** Allocation, transfers, and validation are outside event timing; repeated event batches reduce timer quantization error.
- **Dead-code elimination in microbenchmarks:** Every microkernel produces an observable checksum checked after timing.
- **UB overflow:** Tile changes require a static byte budget and a build for the selected SoC before hardware execution.
- **Tail corruption:** The contract tests inspect tail-path presence, and device correctness cases use sentinel padding around logical buffers.
- **Fast but inaccurate reductions:** FP32 accumulation and the baseline reduction order remain the default; alternatives must pass every numerical threshold before comparison.

## 12. Acceptance Criteria

The implementation is accepted when all of the following hold:

1. The project configures and builds Release targets for baseline, optimized operator, and microbenchmarks on the selected Ascend toolchain and SoC.
2. The root `kernel.asc` exports the competition-required `run_kernel` signature.
3. Baseline and optimized variants pass the PyTorch accuracy limits for every supported type and default matrix shape.
4. Aligned and nonaligned tail cases complete without out-of-bounds sentinel changes.
5. The launch, GM copy, vector Add/Mul, and reduction suites each emit valid CSV rows and passing checksums.
6. `analyze_results.py` validates the schema, reports A/B speedups, and assigns a bottleneck category with the supporting measurements.
7. `test_benchmark_contract.py` passes on a machine without an NPU.
8. Benchmark assets remain outside the competition submission requirement; submitting only the root `kernel.asc` remains possible.
