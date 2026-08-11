#!/usr/bin/env bash
#
# Run the ZLUDA smoke tier: the CUDA backend executing on AMD (ROCm) or Intel
# (Level Zero) GPUs through the ZLUDA transpiler.
#
# Requirements:
#   - ZLUDA_PATH env var pointing at a ZLUDA install
#     (directory containing libcuda.so, or with a lib/ subdirectory)
#   - ROCm installed for AMD, or oneAPI Level Zero for Intel
#   - the CUDA-versioned ZLUDA backend installed once:
#       mvn install -DskipTests -Pzluda -pl :nd4j-zluda-12.9   (from the repo root)
#   - CUDA backend jars available (default backend.artifactId=nd4j-cuda-12.9)
#
# Scope: core driver/runtime, PTX JIT and cuBLAS GEMM paths only. cuDNN and
# CUDA graph capture/replay are NOT expected to work under ZLUDA v6 — do not
# run DSP CUDA_GRAPHS configs through this script.
#
# Usage:
#   ZLUDA_PATH=/opt/zluda ./run-zluda-smoke-tests.sh
#   ZLUDA_PATH=/opt/zluda ./run-zluda-smoke-tests.sh -Dzluda.target=AMD
#   ZLUDA_PATH=/opt/zluda ./run-zluda-smoke-tests.sh -Dzluda.test.groups=smoke   # broaden to smoke tier
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -z "$ZLUDA_PATH" ]; then
    echo "ERROR: ZLUDA_PATH is not set. Point it at your ZLUDA installation, e.g.:"
    echo "  ZLUDA_PATH=/opt/zluda ./run-zluda-smoke-tests.sh"
    exit 1
fi

if [ ! -f "$ZLUDA_PATH/libcuda.so" ] && [ ! -f "$ZLUDA_PATH/lib/libcuda.so" ]; then
    echo "ERROR: no libcuda.so under $ZLUDA_PATH (checked root and lib/)."
    exit 1
fi

EXTRA_ARGS="$@"

# -Ptest-zluda also auto-activates via the ZLUDA_PATH env var; kept explicit for clarity.
/home/agibsonccc/dev-apps/mvn/bin/mvn test \
  -Ptest-zluda \
  -Dtest=ZludaSmokeTest \
  ${EXTRA_ARGS} \
  2>&1 | tee zluda-smoke-tests.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ZLUDA SMOKE TESTS: PASSED"
else
    echo "  ZLUDA SMOKE TESTS: FAILED (exit code $EXIT_CODE)"
    echo "  See zluda-smoke-tests.log for details"
fi
echo "========================================="

exit $EXIT_CODE
