#!/usr/bin/env bash
#
# Run the ROCm-PJRT smoke tier: ND4J on AMD GPUs via the ROCm PJRT plugin
# (xla_rocm_plugin.so), loaded by the same native PjrtClientManager as TPU.
#
# The plugin is fetched WITHOUT python (curl+jq+unzip of the jax-rocm7-pjrt wheel).
# GetPjrtApi/existence checks run on any host; actual in-process load needs a
# ROCm 7 install (libamdhip64.so.7, librocblas.so.5, libMIOpen.so.1, …), i.e. an
# AMD GPU box — so on a non-AMD host the load test skips cleanly.
#
# Requirements:
#   - nd4j-tpu installed once (the PJRT carrier module):
#       mvn install -DskipTests -Ptpu -pl :nd4j-tpu,:nd4j-tpu-preset   (from repo root)
#
# Usage:
#   ./run-rocm-pjrt-smoke-tests.sh                       # auto-fetch plugin python-free
#   PJRT_PLUGIN_LIBRARY_PATH=/path/to/xla_rocm_plugin.so ./run-rocm-pjrt-smoke-tests.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -z "${PJRT_PLUGIN_LIBRARY_PATH:-}" ] && [ -z "${ROCM_PJRT_PATH:-}" ]; then
    echo "PJRT_PLUGIN_LIBRARY_PATH not set — fetching xla_rocm_plugin.so python-free..."
    PLUGIN=$(bash "$REPO_ROOT/libnd4j/scripts/fetch-pjrt-plugin.sh" rocm) || {
        echo "ERROR: could not fetch the ROCm PJRT plugin"; exit 1; }
    export PJRT_PLUGIN_LIBRARY_PATH="$PLUGIN"
    echo "Using ROCm PJRT plugin: $PJRT_PLUGIN_LIBRARY_PATH"
fi

EXTRA_ARGS="$@"

# CPU backend so this runs on machines without an ND4J GPU backend; the ROCm plugin
# is exercised at the native PJRT layer, independent of the ND4J compute backend.
/home/agibsonccc/dev-apps/mvn/bin/mvn test \
  -Ptest-rocm \
  -Dbackend.artifactId=nd4j-native \
  -Drocm.pjrt.path="$PJRT_PLUGIN_LIBRARY_PATH" \
  -Dtest=RocmPjrtSmokeTest \
  ${EXTRA_ARGS} \
  2>&1 | tee rocm-pjrt-smoke-tests.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ROCm PJRT SMOKE TESTS: PASSED"
else
    echo "  ROCm PJRT SMOKE TESTS: FAILED (exit code $EXIT_CODE)"
    echo "  See rocm-pjrt-smoke-tests.log for details"
fi
echo "========================================="

exit $EXIT_CODE
