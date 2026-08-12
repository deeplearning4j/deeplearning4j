#!/usr/bin/env bash
#
# Run the self-contained ZLUDA smoke tier on an AMD GPU.
#
# Requirements:
#   - a compatible AMD kernel driver and accessible GPU device nodes
#   - nd4j-zluda-12.9-platform available from the configured Maven repository
#
# ZLUDA, HIP/ROCm user-space libraries, and DL4J native binaries are extracted
# from the Maven classifier. No CUDA toolkit, ZLUDA_PATH, or LD_LIBRARY_PATH is
# part of the consumer contract.
#
# Usage:
#   ./run-zluda-smoke-tests.sh
#   ./run-zluda-smoke-tests.sh -Dzluda.test.groups=smoke
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

/home/agibsonccc/dev-apps/mvn/bin/mvn test \
  -Ptest-zluda \
  -Dtest=ZludaSmokeTest \
  "$@" \
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
