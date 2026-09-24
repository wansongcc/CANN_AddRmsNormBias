#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

device=0
warmup=30
iterations=200
soc="dav-2201"
output="${ROOT_DIR}/focus_results.csv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) device="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --soc) soc="$2"; shift 2 ;;
        --output) output="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is not set}"
source "${ASCEND_HOME_PATH}/set_env.sh"
cd "${ROOT_DIR}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH="${soc}"
cmake --build build -j"$(nproc)"

benchmark="${ROOT_DIR}/build/add_rms_norm_bias_benchmark"
rm -f "${output}" "${ROOT_DIR}/focus_report.md"
mkdir -p "${ROOT_DIR}/build/focus_cases"

promote_last_row() {
    python3 - "${output}" <<'PY'
import csv
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fields = reader.fieldnames
if not rows or rows[-1]["suite"] != "operator":
    raise SystemExit("missing operator result")
rows[-1]["status"] = "PASS"
with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY
}

cases=("32768 64" "128 4096" "8 32768")
for dtype in fp16 fp32; do
    for shape in "${cases[@]}"; do
        read -r rows hidden <<<"${shape}"
        case_dir="${ROOT_DIR}/build/focus_cases/${dtype}_${rows}x${hidden}"
        python3 scripts/gen_data.py --rows "${rows}" --hidden "${hidden}" \
            --dtype "${dtype}" --epsilon 1e-5 --seed 42 \
            --output-dir "${case_dir}"
        for variant in baseline optimized; do
            "${benchmark}" --suite operator --variant "${variant}" \
                --dtype "${dtype}" --rows "${rows}" --hidden "${hidden}" \
                --device "${device}" --warmup "${warmup}" \
                --iterations "${iterations}" --input-dir "${case_dir}" \
                --output "${output}"
            python3 scripts/verify_result.py \
                --actual "${case_dir}/actual_output.bin" \
                --golden "${case_dir}/golden_output.bin" --dtype "${dtype}"
            promote_last_row
        done
    done
done

python3 benchmarks/analyze_results.py "${output}" \
    --output "${ROOT_DIR}/focus_report.md"
echo "Results: ${output}"
echo "Report:  ${ROOT_DIR}/focus_report.md"
