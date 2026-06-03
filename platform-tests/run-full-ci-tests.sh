#!/usr/bin/env bash
#
# Run full-ci tier tests (~5 min).
# Includes all smoke tests PLUS: DSP lifecycle/concurrency, evaluation metrics,
# op validation basics, Keras/GGML model config, core ND4J shape/indexing.
#
# Suitable for standard CI machines. May require GPU for CUDA-specific tests.
#
# Usage:
#   ./run-full-ci-tests.sh                          # CPU backend (default)
#   ./run-full-ci-tests.sh -Dbackend.artifactId=nd4j-cuda-12.9  # CUDA backend
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Allow extra Maven args
EXTRA_ARGS="$@"

/home/agibsonccc/dev-apps/mvn/bin/mvn test \
  -Pfull-ci \
  ${EXTRA_ARGS} \
  2>&1 | tee full-ci-tests.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  FULL CI TESTS: PASSED"
else
    echo "  FULL CI TESTS: FAILED (exit code $EXIT_CODE)"
    echo "  See full-ci-tests.log for details"
fi
echo "========================================="

exit $EXIT_CODE
