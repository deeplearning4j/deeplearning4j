#!/usr/bin/env bash
#
# Run smoke-tier tests only (~30s).
# These are quick sanity checks that verify core functionality:
# backend init, NDArray creation, basic math, SameDiff graph construction,
# DSP data model, import pipeline wiring.
#
# Safe for low-spec CI machines. No GPU required. No model downloads.
#
# Usage:
#   ./run-smoke-tests.sh                          # CPU backend (default)
#   ./run-smoke-tests.sh -Dbackend.artifactId=nd4j-cuda-12.9  # CUDA backend
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Allow -Dtest= override to run a specific smoke test
EXTRA_ARGS="$@"

/home/agibsonccc/dev-apps/mvn/bin/mvn test \
  -Psmoke \
  ${EXTRA_ARGS} \
  2>&1 | tee smoke-tests.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  SMOKE TESTS: PASSED"
else
    echo "  SMOKE TESTS: FAILED (exit code $EXIT_CODE)"
    echo "  See smoke-tests.log for details"
fi
echo "========================================="

exit $EXIT_CODE
