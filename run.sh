#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is not set; source the CANN set_env.sh first}"
[[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]] || {
    echo "CANN environment script not found: ${ASCEND_HOME_PATH}/set_env.sh" >&2
    exit 1
}
source "${ASCEND_HOME_PATH}/set_env.sh"

device="${DEVICE_ID:-0}"
soc="${NPU_ARCH:-dav-2201}"
case_dir="${ROOT_DIR}/build/quick_case"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNPU_ARCH="${soc}"
cmake --build build -j"$(nproc)"

python3 scripts/gen_data.py --rows 1 --hidden 64 --dtype fp16 \
    --epsilon 1e-5 --seed 42 --output-dir "${case_dir}"

build/add_rms_norm_bias_benchmark --suite operator --variant optimized \
    --dtype fp16 --rows 1 --hidden 64 --device "${device}" \
    --input-dir "${case_dir}" --validate-only

python3 scripts/verify_result.py \
    --actual "${case_dir}/actual_output.bin" \
    --golden "${case_dir}/golden_output.bin" --dtype fp16

echo "=== PASSED: AddRmsNormBias quick correctness check ==="
