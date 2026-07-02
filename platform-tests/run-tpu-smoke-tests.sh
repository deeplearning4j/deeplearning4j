#!/usr/bin/env bash
#
# Run the TPU wiring smoke tier: classpath/SPI registration of nd4j-tpu, in-process
# dlopen of a PJRT/libtpu library, and the Nd4jTpuHelper availability contract.
#
# No TPU hardware is required: `pip install libtpu` provides a libtpu.so that loads
# on any linux-x86_64 host (TPU VMs are x86_64), which is enough to smoke-test the
# native-library wiring. On a real Cloud TPU VM the same script runs against the
# system libtpu.
#
# Requirements:
#   - nd4j-tpu installed once:
#       mvn install -DskipTests -Ptpu -pl :nd4j-tpu,:nd4j-tpu-preset   (from the repo root)
#
# Usage:
#   ./run-tpu-smoke-tests.sh                                   # auto-discovers libtpu via python3
#   PJRT_PATH=/path/to/libtpu.so ./run-tpu-smoke-tests.sh      # explicit library
#   ./run-tpu-smoke-tests.sh -Dbackend.artifactId=nd4j-cuda-12.9
#
# NOTE: only one process can hold a real TPU at a time — keep surefire.forks=1
# (the default) when running on a TPU VM.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ -z "$PJRT_PATH" ] && [ -z "$TPU_LIBRARY_PATH" ]; then
    echo "PJRT_PATH not set - attempting to discover libtpu.so from the python 'libtpu' package..."
    DISCOVERED=$(python3 - <<'PY' 2>/dev/null
import glob, os
try:
    import libtpu
    cands = glob.glob(os.path.join(os.path.dirname(libtpu.__file__), '**', 'libtpu.so'), recursive=True)
    print(cands[0] if cands else '')
except Exception:
    print('')
PY
)
    if [ -n "$DISCOVERED" ]; then
        export PJRT_PATH="$DISCOVERED"
        echo "Discovered libtpu: $PJRT_PATH"
    else
        echo "WARNING: no libtpu found (pip install libtpu). The dlopen test will skip;"
        echo "         classpath/SPI checks still run."
    fi
fi

EXTRA_ARGS="$@"

# Default to the CPU backend so this runs on machines without GPUs; override via args.
/home/agibsonccc/dev-apps/mvn/bin/mvn test \
  -Ptest-tpu \
  -Dbackend.artifactId=nd4j-native \
  -Dtest=TpuBackendSmokeTest \
  ${EXTRA_ARGS} \
  2>&1 | tee tpu-smoke-tests.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  TPU SMOKE TESTS: PASSED"
else
    echo "  TPU SMOKE TESTS: FAILED (exit code $EXIT_CODE)"
    echo "  See tpu-smoke-tests.log for details"
fi
echo "========================================="

exit $EXIT_CODE
