#!/bin/bash
# Wrapper script to run Java under nvprof for CUDA profiling
# Usage: Set JAVA_HOME and this script will profile the Java execution

NVPROF_OUTPUT="${NVPROF_OUTPUT:-/tmp/nvprof_output.log}"
NVPROF_OPTS="${NVPROF_OPTS:---print-gpu-trace --print-api-trace}"

echo "=== Running Java under nvprof ===" >&2
echo "Output: $NVPROF_OUTPUT" >&2
echo "Options: $NVPROF_OPTS" >&2

# Run nvprof with Java
nvprof $NVPROF_OPTS --log-file "$NVPROF_OUTPUT" "$@"
EXIT_CODE=$?

echo "=== nvprof completed. Output saved to: $NVPROF_OUTPUT ===" >&2
echo "=== Summary of GPU activity: ===" >&2
tail -100 "$NVPROF_OUTPUT" 2>/dev/null | grep -E "Time|Calls|Avg|memcpy|kernel" | head -50

exit $EXIT_CODE
