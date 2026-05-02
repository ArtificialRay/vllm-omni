#!/usr/bin/env bash
# Run check_modelopt_fp8_export.py against the 6 ModelOpt FP8 checkpoints
# produced by quant_videogen.sh and aggregate the disk-size numbers into a
# single markdown table.
#
#   0  Wan-AI/Wan2.2-I2V-A14B-Diffusers              per-tensor
#   1  Wan-AI/Wan2.2-I2V-A14B-Diffusers              per-block
#   2  HunyuanVideo-1.5-Diffusers-720p_t2v           per-tensor
#   3  HunyuanVideo-1.5-Diffusers-720p_t2v           per-block
#   4  Wan-AI/Wan2.1-VACE-14B-diffusers              per-tensor
#   5  Wan-AI/Wan2.1-VACE-14B-diffusers              per-block
#
# For each config:
#   1. Run check_modelopt_fp8_export.py with --output and --baseline
#   2. Capture full report to ${LOG_DIR}/<label>.log
#   3. Parse the [C] lines for FP8/BF16 transformer GiB, transformer
#      reduction%, FP8/BF16 whole-repo GiB, whole-repo reduction%
#   4. Append a row to ${OUTPUT_ROOT}/sizes.md
#
# Pure I/O (no GPU); checks run sequentially. Warnings (e.g., baseline not in
# the local HF cache) are non-fatal — the script keeps going and records "-"
# for any field it could not parse.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/quant_checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/quant_size_results}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

CHECK_SCRIPT="${REPO_ROOT}/examples/quantization/check_modelopt_fp8_export.py"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# <fp8-checkpoint-label>|<bf16-baseline-hf-id>
# Per-tensor and per-block share the same BF16 baseline.
CONFIGS=(
    "wan22-i2v-a14b-fp8-per-tensor|Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    "wan22-i2v-a14b-fp8-per-block|Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    "hv15-720p-t2v-fp8-per-tensor|hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"
    "hv15-720p-t2v-fp8-per-block|hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"
    "wan21-vace-14b-r2v-fp8-per-tensor|Wan-AI/Wan2.1-VACE-14B-diffusers"
    "wan21-vace-14b-r2v-fp8-per-block|Wan-AI/Wan2.1-VACE-14B-diffusers"
)

# ============================================================
# Pre-flight
# ============================================================
preflight() {
    local missing=0
    [[ -f "${CHECK_SCRIPT}" ]] || { echo "[preflight] FAIL — check script not found: ${CHECK_SCRIPT}" >&2; missing=1; }
    [[ -d "${CKPT_DIR}" ]] || { echo "[preflight] FAIL — checkpoint dir not found: ${CKPT_DIR}" >&2; missing=1; }
    command -v python >/dev/null 2>&1 || { echo "[preflight] FAIL — python not on PATH" >&2; missing=1; }
    [[ "${missing}" -eq 0 ]]
}
preflight || exit 1

ts() { date +%H:%M:%S; }

# Parse the check script's stdout for size fields. Output (pipe-separated):
#   fp8_xfm | bf16_xfm | xfm_red | fp8_tot | bf16_tot | tot_red
# Any field that could not be parsed is emitted as "-".
#
# Lines we look for in the report:
#   [C] FP8 transformer disk size (transformer + transformer_2): 12.34 GiB
#   [C] BF16 baseline transformer disk size (transformer + transformer_2): 56.78 GiB (...)
#   [C] Disk reduction: 78.3%  (FP8 is 22% of BF16)
#   [C] Whole-repo: FP8 12.34 GiB / BF16 56.78 GiB (reduction 78.3%, deployment footprint)
parse_log() {
    local log="$1"
    local fp8_xfm bf16_xfm xfm_red fp8_tot bf16_tot tot_red
    fp8_xfm=$(grep -oP '\[C\] FP8 transformer disk size \([^)]+\): \K[0-9.]+(?= GiB)' "${log}" | head -1 || true)
    bf16_xfm=$(grep -oP '\[C\] BF16 baseline transformer disk size \([^)]+\): \K[0-9.]+(?= GiB)' "${log}" | head -1 || true)
    xfm_red=$(grep -oP '\[C\] Disk reduction: \K[0-9.]+(?=%)' "${log}" | head -1 || true)
    fp8_tot=$(grep -oP '\[C\] Whole-repo: FP8 \K[0-9.]+(?= GiB)' "${log}" | head -1 || true)
    bf16_tot=$(grep -oP '\[C\] Whole-repo: FP8 [0-9.]+ GiB / BF16 \K[0-9.]+(?= GiB)' "${log}" | head -1 || true)
    tot_red=$(grep -oP '\[C\] Whole-repo: FP8 [0-9.]+ GiB / BF16 [0-9.]+ GiB \(reduction \K[0-9.]+(?=%)' "${log}" | head -1 || true)
    echo "${fp8_xfm:--}|${bf16_xfm:--}|${xfm_red:--}|${fp8_tot:--}|${bf16_tot:--}|${tot_red:--}"
}

# Returns:
#   0 = check ran (sizes likely parseable)
#   1 = check ran but exited non-zero (sizes still parsed if printed before exit)
#   2 = checkpoint dir missing (no run)
run_check() {
    local label="$1" baseline="$2"
    local out_dir="${CKPT_DIR}/${label}"
    local logfile="${LOG_DIR}/${label}.log"

    if [[ ! -d "${out_dir}" ]]; then
        echo "[$(ts)] [${label}] SKIP — checkpoint dir not found: ${out_dir}"
        return 2
    fi

    echo "[$(ts)] [${label}] CHECK — log=${logfile}"
    local rc=0
    python "${CHECK_SCRIPT}" --output "${out_dir}" --baseline "${baseline}" \
        > "${logfile}" 2>&1 || rc=$?
    if [[ "${rc}" -eq 0 ]]; then
        echo "[$(ts)] [${label}] DONE"
    else
        echo "[$(ts)] [${label}] WARN/FAIL (rc=${rc}) — see ${logfile}" >&2
    fi
    return ${rc}
}

# ============================================================
# Main
# ============================================================
echo "[$(ts)] Checkpoint dir: ${CKPT_DIR}"
echo "[$(ts)] Output dir:     ${OUTPUT_ROOT}"
echo "[$(ts)] Log dir:        ${LOG_DIR}"
echo

RESULTS_MD="${OUTPUT_ROOT}/sizes.md"
{
    echo "# FP8 vs BF16 disk size comparison"
    echo
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo
    echo "Sizes parsed from \`check_modelopt_fp8_export.py\` output. \"transformer\""
    echo "covers \`transformer/\` (+ \`transformer_2/\` for Wan2.2 MoE A14B);"
    echo "\"whole-repo\" includes VAE / text-encoder / tokenizer / scheduler /"
    echo "metadata — i.e. the deployment footprint."
    echo
    echo "| Label | FP8 transformer (GiB) | BF16 transformer (GiB) | Transformer reduction (%) | FP8 whole-repo (GiB) | BF16 whole-repo (GiB) | Whole-repo reduction (%) |"
    echo "|-------|-----------------------|------------------------|---------------------------|----------------------|-----------------------|--------------------------|"
} > "${RESULTS_MD}"

overall_rc=0
for entry in "${CONFIGS[@]}"; do
    label="${entry%%|*}"
    baseline="${entry##*|}"
    rc=0
    run_check "${label}" "${baseline}" || rc=$?

    if [[ "${rc}" -eq 2 ]]; then
        printf '| %s | _missing_ | - | - | - | - | - |\n' "${label}" >> "${RESULTS_MD}"
        overall_rc=1
        continue
    fi

    IFS='|' read -r fp8_xfm bf16_xfm xfm_red fp8_tot bf16_tot tot_red \
        <<< "$(parse_log "${LOG_DIR}/${label}.log")"
    printf '| %s | %s | %s | %s | %s | %s | %s |\n' \
        "${label}" "${fp8_xfm}" "${bf16_xfm}" "${xfm_red}" "${fp8_tot}" "${bf16_tot}" "${tot_red}" \
        >> "${RESULTS_MD}"

    [[ "${rc}" -ne 0 ]] && overall_rc=1
done

echo
echo "================================================================"
echo "Aggregated results: ${RESULTS_MD}"
echo "================================================================"
cat "${RESULTS_MD}"

exit ${overall_rc}
