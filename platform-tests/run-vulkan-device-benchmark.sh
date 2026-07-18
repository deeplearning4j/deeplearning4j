#!/bin/bash
# run-vulkan-device-benchmark.sh — Real-device Vulkan benchmark via adb
#
# Detects a connected Android device, pushes and runs the Vulkan benchmark binary,
# pulls the JSON results, and appends a CSV row to vulkan-device-perf.csv.
#
# CSV format: timestamp,device_model,lateSteady_tok_s,status
#
# Usage:
#   ./run-vulkan-device-benchmark.sh [OPTIONS]
#
# Options:
#   --tokens N          Decode tokens (default: 250 — ALWAYS use 250 for perf, fewer for debug only)
#   --config NAME       Benchmark config name (default: AUTO)
#   --binary PATH       Path to benchmark binary on host (default: ./vulkan-benchmark-android-arm64)
#   --output-dir DIR    Directory for local results (default: .)
#   --no-pull           Skip adb pull (use device-side JSON only; for Firebase Test Lab)
#   --dry-run           Print adb commands without executing them
#   -h, --help          Show this help
#
# Exit codes:
#   0  — device not found (graceful skip) OR benchmark succeeded
#   1  — benchmark binary missing, adb push failed, or benchmark returned an error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ───────────────────────────────────────────────────────────────────
TOKENS=250
CONFIG="AUTO"
BENCHMARK_BIN="${SCRIPT_DIR}/vulkan-benchmark-android-arm64"
OUTPUT_DIR="${SCRIPT_DIR}"
DEVICE_TMP="/data/local/tmp"
BENCH_JSON_DEVICE="${DEVICE_TMP}/vulkan-bench.json"
NO_PULL=false
DRY_RUN=false
CSV_FILE="${OUTPUT_DIR}/vulkan-device-perf.csv"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tokens)
      TOKENS="$2"; shift 2 ;;
    --config)
      CONFIG="$2"; shift 2 ;;
    --binary)
      BENCHMARK_BIN="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; CSV_FILE="${OUTPUT_DIR}/vulkan-device-perf.csv"; shift 2 ;;
    --no-pull)
      NO_PULL=true; shift ;;
    --dry-run)
      DRY_RUN=true; shift ;;
    -h|--help)
      sed -n '/^# Usage:/,/^[^#]/{ /^#/{ s/^# \{0,1\}//; p }; /^[^#]/q }' "$0"
      exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

# ── Helpers ────────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
warn() { echo "[WARN]  $*" >&2; }
err()  { echo "[ERROR] $*" >&2; exit 1; }

run_adb() {
  # Wrapper that respects --dry-run
  if $DRY_RUN; then
    echo "[dry-run] adb $*"
    return 0
  fi
  adb "$@"
}

# ── Check adb availability ─────────────────────────────────────────────────────
if ! command -v adb &>/dev/null; then
  warn "adb not found. Install Android SDK Platform Tools and add to PATH."
  warn "Skipping Vulkan device benchmark."
  exit 0
fi

# ── Detect connected non-offline device ───────────────────────────────────────
# "adb devices | grep -v offline" per task spec; also filter the header line.
DEVICE_LINE=$(adb devices | grep -v offline | grep -v "^List of" | grep -v "^$" | head -1 || true)

if [[ -z "$DEVICE_LINE" ]]; then
  log "No Android device found, skipping."
  exit 0
fi

DEVICE_ID=$(echo "$DEVICE_LINE" | awk '{print $1}')
log "Found device: ${DEVICE_ID}"

# ── Verify device is reachable ─────────────────────────────────────────────────
if ! adb -s "$DEVICE_ID" shell "getprop ro.product.model" >/dev/null 2>&1; then
  warn "Device ${DEVICE_ID} is not fully reachable; skipping."
  exit 0
fi

DEVICE_MODEL=$(adb -s "$DEVICE_ID" shell "getprop ro.product.model" 2>/dev/null | tr -d '\r' || echo "unknown")
log "Device model: ${DEVICE_MODEL}"

# ── Check benchmark binary exists on host ─────────────────────────────────────
if [[ ! -f "$BENCHMARK_BIN" ]] && ! $DRY_RUN; then
  err "Benchmark binary not found: ${BENCHMARK_BIN}
Build it first with: cd platform-tests && ./build-vulkan.sh --android --skip-tests"
fi

# ── Push binary to device ─────────────────────────────────────────────────────
log "Pushing benchmark binary to device..."
run_adb -s "$DEVICE_ID" push "$BENCHMARK_BIN" "${DEVICE_TMP}/" >/dev/null

# Ensure the binary is executable
run_adb -s "$DEVICE_ID" shell "chmod 755 ${DEVICE_TMP}/vulkan-benchmark-android-arm64"

# ── Run benchmark on device ───────────────────────────────────────────────────
log "Running Vulkan benchmark (tokens=${TOKENS}, config=${CONFIG})..."

BENCH_CMD="LD_LIBRARY_PATH=${DEVICE_TMP} timeout 600 ${DEVICE_TMP}/vulkan-benchmark-android-arm64 --tokens ${TOKENS} --config ${CONFIG} --output ${BENCH_JSON_DEVICE}"

BENCH_STATUS=0
if $DRY_RUN; then
  log "[dry-run] adb -s ${DEVICE_ID} shell \"${BENCH_CMD}\""
else
  adb -s "$DEVICE_ID" shell "$BENCH_CMD" || BENCH_STATUS=$?
fi

if [[ $BENCH_STATUS -ne 0 ]]; then
  warn "Benchmark exited with status ${BENCH_STATUS}."
fi

# ── Pull results ──────────────────────────────────────────────────────────────
LOCAL_JSON="${OUTPUT_DIR}/vulkan-bench.json"

if ! $NO_PULL; then
  log "Pulling results from device..."
  if ! run_adb -s "$DEVICE_ID" pull "${BENCH_JSON_DEVICE}" "${LOCAL_JSON}" 2>/dev/null; then
    warn "Could not pull ${BENCH_JSON_DEVICE} from device."
    LOCAL_JSON=""
  else
    log "Results pulled to: ${LOCAL_JSON}"
  fi
fi

# ── Parse lateSteady_tok_per_sec from JSON ────────────────────────────────────
LATE_STEADY=""
STATUS="success"

if [[ -n "$LOCAL_JSON" && -f "$LOCAL_JSON" ]]; then
  if command -v jq &>/dev/null; then
    LATE_STEADY=$(jq -r '.lateSteady_tok_per_sec // .lateSteady // empty' "$LOCAL_JSON" 2>/dev/null || true)
  else
    # Fallback: grep-based extraction (no jq required)
    LATE_STEADY=$(grep -oE '"lateSteady_tok_per_sec"\s*:\s*[0-9]+(\.[0-9]+)?' "$LOCAL_JSON" 2>/dev/null \
                  | grep -oE '[0-9]+(\.[0-9]+)?$' | head -1 || true)
    if [[ -z "$LATE_STEADY" ]]; then
      LATE_STEADY=$(grep -oE '"lateSteady"\s*:\s*[0-9]+(\.[0-9]+)?' "$LOCAL_JSON" 2>/dev/null \
                    | grep -oE '[0-9]+(\.[0-9]+)?$' | head -1 || true)
    fi
  fi
fi

if [[ -z "$LATE_STEADY" ]]; then
  STATUS="no_results"
  if [[ $BENCH_STATUS -ne 0 ]]; then
    STATUS="benchmark_error"
  fi
fi

# ── Append to CSV ─────────────────────────────────────────────────────────────
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Write CSV header if file does not yet exist (idempotent: header only on first run)
if [[ ! -f "$CSV_FILE" ]]; then
  mkdir -p "$(dirname "$CSV_FILE")"
  echo "timestamp,device_model,lateSteady_tok_s,status" > "$CSV_FILE"
fi

LATE_STEADY_VAL="${LATE_STEADY:--}"
printf '%s,%s,%s,%s\n' "$TIMESTAMP" "$DEVICE_MODEL" "$LATE_STEADY_VAL" "$STATUS" >> "$CSV_FILE"

log "CSV appended: ${CSV_FILE}"
log "  timestamp:      ${TIMESTAMP}"
log "  device_model:   ${DEVICE_MODEL}"
log "  lateSteady_tok_s: ${LATE_STEADY_VAL}"
log "  status:         ${STATUS}"

# ── Clean up device tmp files (idempotent) ────────────────────────────────────
run_adb -s "$DEVICE_ID" shell \
  "rm -f ${DEVICE_TMP}/vulkan-benchmark-android-arm64 ${BENCH_JSON_DEVICE}" \
  >/dev/null 2>&1 || true

log "Done."
exit 0
