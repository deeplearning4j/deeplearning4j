#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_ID="${1:-}"
OUTPUT_DIR="${2:-.}"
BENCHMARK_BIN="${SCRIPT_DIR}/vulkan-benchmark-android-arm64"
RESULTS_DIR="/tmp/vulkan-device-results"
DEVICE_TMP="/data/local/tmp"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
  echo "[ERROR] $*" >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR"

log "Vulkan Device Benchmark Harness"
log "Output directory: $OUTPUT_DIR"

if ! command -v adb &> /dev/null; then
  error "adb not found. Install Android SDK Platform Tools."
fi

log "Detecting connected device..."
DEVICES=$(adb devices -l 2>/dev/null | grep -v "^List\|^$" | awk '{print $1}' | sort | uniq)
DEVICE_COUNT=$(echo "$DEVICES" | grep -c . || true)

if [ "$DEVICE_COUNT" -eq 0 ]; then
  error "No device connected. Connect an Android device via USB."
fi

if [ -z "$DEVICE_ID" ]; then
  if [ "$DEVICE_COUNT" -eq 1 ]; then
    DEVICE_ID=$(echo "$DEVICES" | head -1)
    log "Auto-selected device: $DEVICE_ID"
  else
    log "Multiple devices detected:"
    echo "$DEVICES" | nl
    error "Specify device ID: $0 <device-id>"
  fi
fi

log "Verifying device: $DEVICE_ID"
if ! adb -s "$DEVICE_ID" shell "getprop ro.build.version.release" > /dev/null 2>&1; then
  error "Device $DEVICE_ID not accessible."
fi

API_LEVEL=$(adb -s "$DEVICE_ID" shell "getprop ro.build.version.sdk" | tr -d '\r')
BUILD_FINGERPRINT=$(adb -s "$DEVICE_ID" shell "getprop ro.build.fingerprint" | tr -d '\r')
log "Device API level: $API_LEVEL"
log "Device fingerprint: $BUILD_FINGERPRINT"

if [ "$API_LEVEL" -lt 26 ]; then
  error "Device API level $API_LEVEL < 26. Vulkan requires Android 8.0 (API 26)."
fi

log "Checking for Vulkan support..."
VULKAN_VERSION=$(adb -s "$DEVICE_ID" shell "getprop ro.opengles.version" 2>/dev/null | tr -d '\r' || echo "0")
if [ -z "$VULKAN_VERSION" ]; then
  log "WARNING: Could not verify Vulkan version. Proceeding anyway."
fi

if [ ! -f "$BENCHMARK_BIN" ]; then
  error "Benchmark binary not found: $BENCHMARK_BIN"
fi

log "Pushing benchmark binary to device..."
adb -s "$DEVICE_ID" push "$BENCHMARK_BIN" "$DEVICE_TMP/" > /dev/null 2>&1

log "Running benchmark on device..."
BENCHMARK_CMD="cd $DEVICE_TMP && LD_LIBRARY_PATH=$DEVICE_TMP ./vulkan-benchmark-android-arm64 --tokens 250 --config AUTO"
BENCH_OUTPUT=$(adb -s "$DEVICE_ID" shell "$BENCHMARK_CMD" 2>&1 || true)

if [ -z "$BENCH_OUTPUT" ]; then
  error "Benchmark execution failed or produced no output."
fi

log "Pulling results from device..."
mkdir -p "$RESULTS_DIR"

BENCH_JSON="$RESULTS_DIR/benchmark.json"
if ! adb -s "$DEVICE_ID" pull "$DEVICE_TMP/benchmark.json" "$BENCH_JSON" > /dev/null 2>&1; then
  log "WARNING: benchmark.json not found on device. Checking for alternate locations..."
  if ! adb -s "$DEVICE_ID" pull "/sdcard/benchmark.json" "$BENCH_JSON" > /dev/null 2>&1; then
    log "WARNING: Could not pull benchmark results JSON."
    BENCH_JSON=""
  fi
fi

BENCH_LOG="$RESULTS_DIR/vulkan-bench.log"
if ! adb -s "$DEVICE_ID" pull "$DEVICE_TMP/vulkan-bench.log" "$BENCH_LOG" > /dev/null 2>&1; then
  log "WARNING: vulkan-bench.log not found on device."
  BENCH_LOG=""
fi

log "Cleaning up device temporary files..."
adb -s "$DEVICE_ID" shell "rm -f $DEVICE_TMP/vulkan-benchmark-android-arm64 $DEVICE_TMP/benchmark.json $DEVICE_TMP/vulkan-bench.log" > /dev/null 2>&1 || true

CSV_FILE="$OUTPUT_DIR/vulkan-device-results.csv"
JSON_OUTPUT="$OUTPUT_DIR/benchmark.json"

log "Processing results..."

if [ -n "$BENCH_JSON" ] && [ -f "$BENCH_JSON" ]; then
  cp "$BENCH_JSON" "$JSON_OUTPUT"
  log "Benchmark JSON copied to: $JSON_OUTPUT"
else
  log "WARNING: No benchmark JSON available. Creating minimal output."
  JSON_OUTPUT=""
fi

if [ -n "$BENCH_LOG" ] && [ -f "$BENCH_LOG" ]; then
  cp "$BENCH_LOG" "$OUTPUT_DIR/vulkan-bench.log"
  log "Benchmark log copied to: $OUTPUT_DIR/vulkan-bench.log"
fi

{
  echo "device_id,api_level,fingerprint,timestamp,late_steady_tok_s,steady_tok_s,decode_tok_s,prefill_tok_s,status"

  LATE_STEADY=""
  STEADY=""
  DECODE=""
  PREFILL=""
  STATUS="success"

  if [ -n "$JSON_OUTPUT" ] && [ -f "$JSON_OUTPUT" ]; then
    if command -v jq &> /dev/null; then
      LATE_STEADY=$(jq -r '.lateSteady // empty' "$JSON_OUTPUT" 2>/dev/null || echo "")
      STEADY=$(jq -r '.steady // empty' "$JSON_OUTPUT" 2>/dev/null || echo "")
      DECODE=$(jq -r '.decode // empty' "$JSON_OUTPUT" 2>/dev/null || echo "")
      PREFILL=$(jq -r '.prefill // empty' "$JSON_OUTPUT" 2>/dev/null || echo "")
    else
      log "WARNING: jq not found. Attempting grep-based parsing."
      LATE_STEADY=$(grep -oP '"lateSteady":\K[0-9.]+' "$JSON_OUTPUT" 2>/dev/null | head -1 || echo "")
      STEADY=$(grep -oP '"steady":\K[0-9.]+' "$JSON_OUTPUT" 2>/dev/null | head -1 || echo "")
      DECODE=$(grep -oP '"decode":\K[0-9.]+' "$JSON_OUTPUT" 2>/dev/null | head -1 || echo "")
      PREFILL=$(grep -oP '"prefill":\K[0-9.]+' "$JSON_OUTPUT" 2>/dev/null | head -1 || echo "")
    fi
  fi

  if [ -z "$LATE_STEADY" ]; then
    STATUS="no_results"
  fi

  TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$DEVICE_ID" "$API_LEVEL" "$BUILD_FINGERPRINT" "$TIMESTAMP" \
    "${LATE_STEADY:--}" "${STEADY:--}" "${DECODE:--}" "${PREFILL:--}" "$STATUS"
} > "$CSV_FILE"

log "Results written to: $CSV_FILE"
log "CSV preview:"
head -3 "$CSV_FILE"

if [ -f "$JSON_OUTPUT" ]; then
  log "Metrics extracted:"
  if [ -n "$LATE_STEADY" ]; then
    log "  lateSteady tok/s: $LATE_STEADY"
  fi
  if [ -n "$STEADY" ]; then
    log "  steady tok/s: $STEADY"
  fi
  if [ -n "$DECODE" ]; then
    log "  decode tok/s: $DECODE"
  fi
else
  log "No metrics available in JSON output."
fi

log "Benchmark harness complete."
exit 0
