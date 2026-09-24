# CANN AddRmsNormBias

这是一个基于 Ascend C kernel-direct 空模板开发的 AddRmsNormBias 融合算子工程。算子对每一行执行：

```text
y = x + residual
scale = 1 / sqrt(mean(y * y) + epsilon)
output = y * scale * gamma + bias
```

支持 FP16、BF16 和 FP32，输入 rank 为 2–4，最后一维 `D` 支持 64–32768，包括非 32 字节对齐的尾块。FP16/BF16 输入使用 FP32 完成平方、归约、归一化和仿射计算。

## 目录

- `kernel.asc`：优化版算子，也是比赛提交文件。
- `benchmarks/kernel_baseline.asc`：冻结的正确性基线，用于 A/B 对比。
- `benchmarks/micro_kernels.asc`：launch、GM copy、Vector Add/Mul 和 reduction 微内核。
- `benchmarks/benchmark_main.asc`：统一 ACL Event 计时程序。
- `benchmarks/operator_cases.csv`：默认 operator 测试矩阵。
- `benchmarks/analyze_results.py`：CSV 校验、speedup 和瓶颈分类。
- `benchmarks/run_benchmarks.sh`：完整 Release 构建、精度验证和性能测试入口。

## 环境要求

- Linux AArch64 Ascend 运行环境
- CANN Toolkit，且 `ASCEND_HOME_PATH` 指向包含 `set_env.sh` 的目录
- 可见的 `/dev/davinciN` 设备节点
- CMake、Python 3、NumPy 和 ml_dtypes

```bash
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install --user numpy ml_dtypes
```

如果 CANN 安装在其他目录，请改为 source 对应的 `set_env.sh`。

## 快速正确性检查

```bash
chmod +x run.sh benchmarks/run_benchmarks.sh
./run.sh
```

该命令进行 Release 构建，生成 `[1,64]` FP16 数据，运行优化版 `kernel.asc`，再与 NumPy FP32 内部计算的 golden 比较。

## 完整硬件基准

```bash
./benchmarks/run_benchmarks.sh --device 0 --soc dav-2201 \
  --warmup 100 --iterations 1000 --output benchmark_results.csv
```

`--device` 使用 ACL Runtime 的逻辑编号，取值范围是
`0..aclrtGetDeviceCount()-1`。它不是容器内 `/dev/davinciN` 的 `N`；例如容器
只挂载 `/dev/davinci8` 和 `/dev/davinci9` 时，两个可见设备的逻辑编号通常为
`0` 和 `1`。

脚本会执行：

1. launch 固定开销测试；
2. FP16/FP32 GM 搬运；
3. FP16/FP32 Vector Add 和 Mul；
4. FP16/BF16/FP32 平方归约；
5. baseline 与 optimized 的完整 AddRmsNormBias A/B；
6. rank 2/3/4、4095/4096/4097 尾块、zero/NaN/Inf、输出哨兵和逐字节确定性检查。

计时使用同一 stream 上的 ACL Event。默认先预热 100 次，然后采集 10 批、每批 1000 次，输出 `min/p50/p90`。内存分配、H2D、D2H、文件 I/O 和结果验证不计入设备时延。

只验证当前 kernel 优化时，运行缩减后的 FP16/FP32 聚焦测试：

```bash
./benchmarks/run_focus.sh --device 0 --warmup 30 --iterations 200
```

该脚本只测试 `32768x64`、`128x4096` 和 `8x32768` 的
baseline/optimized，输出 `focus_results.csv` 和 `focus_report.md`。

原始数据保存在 `benchmark_results.csv`。再次生成报告：

```bash
python3 benchmarks/analyze_results.py benchmark_results.csv \
  --output benchmark_report.md
```

报告只使用精度验证后标记为 `PASS` 的 operator 数据计算 speedup，并依据微基准将用例归类为 launch/control、reduction、memory/pipeline、runtime variance 或 mixed。

## 主要优化

- FP16/BF16 保持 FP32 平方、归约和归一化。
- 归约顺序以 4096 元素为一段，减少性能修改对数值结果的影响。
- 按归约 partial 数量分配工作区，降低 UB 占用。
- 对齐地址和长度使用普通 `DataCopy`，非对齐尾块使用 `DataCopyPad`。
- FP16/BF16 使用更大的 I/O tile；单 tile 行复用 `x + residual` 的 FP32 数据。
- FP32 写回直接使用计算结果，避免乘 1 的表示转换。
- 对满足对齐条件的小 `D` 多行进行 UB 批处理，并缓存 gamma/bias。

## 比赛提交

比赛提交界面只需要提交根目录的 `kernel.asc`。`benchmarks/`、`scripts/`、`main.asc` 和 CMake 文件用于本地正确性与硬件性能分析，不需要上传到仅接收 kernel 的提交入口。
