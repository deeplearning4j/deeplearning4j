#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_ID="${1:?Usage: $0 <device-id> [model-name]}"
MODEL_NAME="${2:-vulkan-android-arm64}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
JUNIT_OUTPUT="$OUTPUT_DIR/vulkan-device-junit.xml"
CSV_OUTPUT="$OUTPUT_DIR/vulkan-device-results.csv"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
  echo "[ERROR] $*" >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR"

log "Vulkan Device CI Harness"
log "Device ID: $DEVICE_ID"
log "Model: $MODEL_NAME"
log "Output directory: $OUTPUT_DIR"

TEST_NAME="vulkan_device_benchmark"
TEST_CLASS="VulkanDeviceBenchmark"
TEST_TIME_START=$(date +%s%N | cut -b1-13)

if ! command -v adb &> /dev/null; then
  error "adb not found. Install Android SDK Platform Tools."
fi

log "Waiting for device..."
WAIT_TIME=0
MAX_WAIT=30
while ! adb -s "$DEVICE_ID" shell "getprop ro.build.version.release" > /dev/null 2>&1; do
  if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    error "Device $DEVICE_ID not available after ${MAX_WAIT}s."
  fi
  sleep 1
  WAIT_TIME=$((WAIT_TIME + 1))
done

log "Device $DEVICE_ID is ready."

TEST_STATUS="passed"
TEST_FAILURE_MSG=""
TEST_OUTPUT=""

if "$SCRIPT_DIR/test-vulkan-device.sh" "$DEVICE_ID" "$OUTPUT_DIR" > "$OUTPUT_DIR/test-output.log" 2>&1; then
  log "Benchmark completed successfully."
  TEST_OUTPUT=$(cat "$OUTPUT_DIR/test-output.log" || echo "")
else
  TEST_STATUS="failed"
  TEST_FAILURE_MSG="Benchmark execution failed."
  TEST_OUTPUT=$(cat "$OUTPUT_DIR/test-output.log" 2>/dev/null || echo "No output captured.")
  log "WARNING: $TEST_FAILURE_MSG"
fi

TEST_TIME_END=$(date +%s%N | cut -b1-13)
TEST_TIME_SEC=$(echo "scale=3; ($TEST_TIME_END - $TEST_TIME_START) / 1000" | bc)

DEVICE_NAME=$(adb -s "$DEVICE_ID" shell "getprop ro.product.model" 2>/dev/null | tr -d '\r' || echo "unknown")
DEVICE_BRAND=$(adb -s "$DEVICE_ID" shell "getprop ro.product.brand" 2>/dev/null | tr -d '\r' || echo "unknown")
API_LEVEL=$(adb -s "$DEVICE_ID" shell "getprop ro.build.version.sdk" 2>/dev/null | tr -d '\r' || echo "0")

METRICS_OUTPUT=""
if [ -f "$CSV_OUTPUT" ]; then
  METRICS_OUTPUT=$(tail -n 1 "$CSV_OUTPUT" || echo "")
fi

{
  cat <<'XMLEOF'
<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="1" failures="0" time="TESTTIME">
XMLEOF

  if [ "$TEST_STATUS" = "passed" ]; then
    cat <<XMLEOF
  <testsuite name="$TEST_CLASS" tests="1" failures="0" errors="0" time="$TEST_TIME_SEC">
    <testcase classname="$TEST_CLASS" name="$TEST_NAME" time="$TEST_TIME_SEC">
      <properties>
        <property name="device.id" value="$DEVICE_ID"/>
        <property name="device.model" value="$DEVICE_BRAND $DEVICE_NAME"/>
        <property name="device.api_level" value="$API_LEVEL"/>
        <property name="test.model" value="$MODEL_NAME"/>
        <property name="test.timestamp" value="$(date -u +%Y-%m-%dT%H:%M:%SZ)"/>
      </properties>
    </testcase>
  </testsuite>
XMLEOF
  else
    cat <<XMLEOF
  <testsuite name="$TEST_CLASS" tests="1" failures="1" errors="0" time="$TEST_TIME_SEC">
    <testcase classname="$TEST_CLASS" name="$TEST_NAME" time="$TEST_TIME_SEC">
      <failure message="$TEST_FAILURE_MSG">
        <![CDATA[
$TEST_OUTPUT
        ]]>
      </failure>
      <properties>
        <property name="device.id" value="$DEVICE_ID"/>
        <property name="device.model" value="$DEVICE_BRAND $DEVICE_NAME"/>
        <property name="device.api_level" value="$API_LEVEL"/>
        <property name="test.model" value="$MODEL_NAME"/>
        <property name="test.timestamp" value="$(date -u +%Y-%m-%dT%H:%M:%SZ)"/>
      </properties>
    </testcase>
  </testsuite>
XMLEOF
  fi

  cat <<'XMLEOF'
</testsuites>
XMLEOF
} > "$JUNIT_OUTPUT"

log "JUnit XML written to: $JUNIT_OUTPUT"

{
  echo "device,model,api_level,build,test_status,metrics,timestamp"

  DEVICE_BUILD=$(adb -s "$DEVICE_ID" shell "getprop ro.build.id" 2>/dev/null | tr -d '\r' || echo "unknown")
  echo "$DEVICE_ID,$DEVICE_BRAND $DEVICE_NAME,$API_LEVEL,$DEVICE_BUILD,$TEST_STATUS,$METRICS_OUTPUT,$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUTPUT_DIR/vulkan-device-ci-summary.csv"

log "CI summary written to: $OUTPUT_DIR/vulkan-device-ci-summary.csv"

if [ "$TEST_STATUS" = "failed" ]; then
  error "Test failed. See $JUNIT_OUTPUT for details."
fi

log "CI harness complete."
exit 0
