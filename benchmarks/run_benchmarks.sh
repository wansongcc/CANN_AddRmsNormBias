#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASES_FILE="${SCRIPT_DIR}/operator_cases.csv"

device=0
warmup=100
iterations=1000
soc="dav-2201"
output="${ROOT_DIR}/benchmark_results.csv"

usage() {
    cat <<'EOF'
Usage: benchmarks/run_benchmarks.sh [options]
  --device INT       Ascend device index (default: 0)
  --warmup INT       Warmup launches per case (default: 100)
  --iterations INT   Measured launches per batch (default: 1000)
  --soc NAME         NPU architecture passed to CMake (default: dav-2201)
  --output PATH      Result CSV path (default: benchmark_results.csv)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) device="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --soc) soc="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${device}" =~ ^[0-9]+$ ]] || { echo "--device must be nonnegative" >&2; exit 2; }
[[ "${warmup}" =~ ^[1-9][0-9]*$ ]] || { echo "--warmup must be positive" >&2; exit 2; }
[[ "${iterations}" =~ ^[1-9][0-9]*$ ]] || { echo "--iterations must be positive" >&2; exit 2; }

: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is not set; source the CANN set_env.sh first}"
[[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]] || {
    echo "CANN environment script not found: ${ASCEND_HOME_PATH}/set_env.sh" >&2
    exit 1
}
source "${ASCEND_HOME_PATH}/set_env.sh"
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
command -v cmake >/dev/null || { echo "cmake is required" >&2; exit 1; }
[[ -e "/dev/davinci${device}" ]] || {
    echo "Ascend device node not found: /dev/davinci${device}" >&2
    exit 1
}

cd "${ROOT_DIR}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH="${soc}"
cmake --build build -j"$(nproc)"

benchmark="${ROOT_DIR}/build/add_rms_norm_bias_benchmark"
[[ -x "${benchmark}" ]] || { echo "benchmark executable not found: ${benchmark}" >&2; exit 1; }
mkdir -p "$(dirname "${output}")" "${ROOT_DIR}/build/benchmark_cases"
rm -f "${output}" "${ROOT_DIR}/benchmark_report.md"

common=(--device "${device}" --warmup "${warmup}" --iterations "${iterations}" --output "${output}")

run_micro() {
    local suite="$1" dtype="$2" rows="$3" hidden="$4"
    "${benchmark}" --suite "${suite}" --variant micro --dtype "${dtype}" \
        --rows "${rows}" --hidden "${hidden}" "${common[@]}"
}

promote_last_operator_row() {
    python3 - "${output}" <<'PY'
import csv
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fields = reader.fieldnames
if not rows or rows[-1]["suite"] != "operator" or rows[-1]["status"] != "MEASURE_ONLY":
    raise SystemExit("last result is not an unverified operator row")
rows[-1]["status"] = "PASS"
with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY
}

verify_case() {
    local case_dir="$1" dtype="$2"
    python3 scripts/verify_result.py \
        --actual "${case_dir}/actual_output.bin" \
        --golden "${case_dir}/golden_output.bin" --dtype "${dtype}"
}

echo "[1/5] Launch and focused microbenchmarks"
run_micro launch fp32 1 1
for dtype in fp16 fp32; do
    for hidden in 64 4095 4096 4097 8192 32768; do
        run_micro gm "${dtype}" 1 "${hidden}"
        run_micro vector_add "${dtype}" 1 "${hidden}"
        run_micro vector_mul "${dtype}" 1 "${hidden}"
    done
done

echo "[2/5] Reduction matrix"
while IFS=, read -r rows hidden dtypes purpose; do
    [[ "${rows}" == "rows" ]] && continue
    IFS='|' read -r -a dtype_list <<<"${dtypes}"
    for dtype in "${dtype_list[@]}"; do
        run_micro reduction "${dtype}" "${rows}" "${hidden}"
    done
done < "${CASES_FILE}"

echo "[3/5] Baseline/optimized operator A/B"
case_id=0
while IFS=, read -r rows hidden dtypes purpose; do
    [[ "${rows}" == "rows" ]] && continue
    IFS='|' read -r -a dtype_list <<<"${dtypes}"
    for dtype in "${dtype_list[@]}"; do
        case_dir="${ROOT_DIR}/build/benchmark_cases/case_${case_id}_${dtype}_${rows}x${hidden}"
        python3 scripts/gen_data.py --rows "${rows}" --hidden "${hidden}" \
            --dtype "${dtype}" --epsilon 1e-5 --seed "$((42 + case_id))" \
            --output-dir "${case_dir}"

        case_iterations="${iterations}"
        if (( rows * hidden >= 4194304 )); then
            case_iterations=$((iterations / 20))
            (( case_iterations < 20 )) && case_iterations=20
        fi

        for variant in baseline optimized; do
            "${benchmark}" --suite operator --variant "${variant}" \
                --dtype "${dtype}" --rows "${rows}" --hidden "${hidden}" \
                --device "${device}" --warmup "${warmup}" \
                --iterations "${case_iterations}" --input-dir "${case_dir}" \
                --output "${output}"
            verify_case "${case_dir}" "${dtype}"
            promote_last_operator_row
        done

        "${benchmark}" --suite operator --variant optimized --dtype "${dtype}" \
            --rows "${rows}" --hidden "${hidden}" --device "${device}" \
            --input-dir "${case_dir}" --validate-only
        cp "${case_dir}/actual_output.bin" "${case_dir}/deterministic_first.bin"
        "${benchmark}" --suite operator --variant optimized --dtype "${dtype}" \
            --rows "${rows}" --hidden "${hidden}" --device "${device}" \
            --input-dir "${case_dir}" --validate-only
        cmp "${case_dir}/deterministic_first.bin" "${case_dir}/actual_output.bin"
        case_id=$((case_id + 1))
    done
done < "${CASES_FILE}"

echo "[4/5] Rank and special-value correctness"
rank_dir="${ROOT_DIR}/build/benchmark_cases/rank"
python3 scripts/gen_data.py --rows 6 --hidden 64 --dtype fp32 \
    --epsilon 1e-5 --seed 777 --output-dir "${rank_dir}"
for shape in 6,64 2,3,64 1,2,3,64; do
    "${benchmark}" --suite operator --variant optimized --dtype fp32 \
        --rows 6 --hidden 64 --shape "${shape}" --device "${device}" \
        --input-dir "${rank_dir}" --validate-only
    verify_case "${rank_dir}" fp32
done

for pattern in zero nan inf; do
    special_dir="${ROOT_DIR}/build/benchmark_cases/${pattern}"
    python3 scripts/gen_data.py --rows 1 --hidden 64 --dtype fp32 \
        --epsilon 1e-5 --seed 888 --pattern "${pattern}" \
        --output-dir "${special_dir}"
    "${benchmark}" --suite operator --variant optimized --dtype fp32 \
        --rows 1 --hidden 64 --device "${device}" \
        --input-dir "${special_dir}" --validate-only
    verify_case "${special_dir}" fp32
done

echo "[5/5] Validate and analyze results"
python3 - "${output}" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[0]
sys.path.insert(0, str(pathlib.Path.cwd() / "benchmarks"))
from benchmark_schema import validate_result_rows

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    validate_result_rows(list(csv.DictReader(handle)))
PY
python3 benchmarks/analyze_results.py "${output}" \
    --output "${ROOT_DIR}/benchmark_report.md"
echo "Results: ${output}"
echo "Report:  ${ROOT_DIR}/benchmark_report.md"
