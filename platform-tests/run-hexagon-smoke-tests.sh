#!/usr/bin/env bash
#
# Run the Hexagon NPU wiring smoke tier: classpath/SPI registration of nd4j-hexagon,
# optional in-process dlopen of a hexagon-mlir runtime library, and the
# HexagonEnvironment device-info contract.
#
# No NPU hardware is required — the backend is a stub (canRun()=false) until the
# hexagon-mlir bindings land; this tier locks those contracts. hexagon-mlir is
# BSD-3 open source (Qualcomm, Dec 2025), so a runtime .so can be built from
# source and pointed at via HEXAGON_MLIR_PATH for the dlopen check.
#
# Requirements:
#   - nd4j-hexagon installed once:
#       mvn install -DskipTests -Phexagon -pl :nd4j-hexagon-preset,:nd4j-hexagon   (from the repo root)
#
# Usage:
#   ./run-hexagon-smoke-tests.sh
#   HEXAGON_MLIR_PATH=/path/to/libhexagon_mlir_runtime.so ./run-hexagon-smoke-tests.sh
#   ./run-hexagon-smoke-tests.sh -Dbackend.artifactId=nd4j-cuda-12.9
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

EXTRA_ARGS="$@"

# Default to the CPU backend so this runs on machines without GPUs; override via args.
/home/agibsonccc/dev-apps/mvn/bin/mvn test \
  -Ptest-hexagon \
  -Dbackend.artifactId=nd4j-native \
  -Dtest=HexagonBackendSmokeTest \
  ${EXTRA_ARGS} \
  2>&1 | tee hexagon-smoke-tests.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  HEXAGON SMOKE TESTS: PASSED"
else
    echo "  HEXAGON SMOKE TESTS: FAILED (exit code $EXIT_CODE)"
    echo "  See hexagon-smoke-tests.log for details"
fi
echo "========================================="

exit $EXIT_CODE
