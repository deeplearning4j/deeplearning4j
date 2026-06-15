#!/bin/bash
# ===========================================================================
# Run ALL platform-tests by category
# ===========================================================================
# Runs every test category sequentially. Each category produces its own
# log file (*-test-output.log). A summary is printed at the end.
#
# Usage:
#   ./run-all-tests.sh                                  # All categories
#   ./run-all-tests.sh -Dbackend.artifactId=nd4j-native # CPU backend
#
# To run a single category, use the individual script instead:
#   ./run-datavec-tests.sh
#   ./run-nd4j-tests.sh
#   ./run-samediff-tests.sh
#   ./run-dl4jcore-tests.sh
#   ./run-keras-tests.sh
#   ./run-libnd4j-tests.sh
# ===========================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CATEGORIES=(
    "datavec"
    "nd4j"
    "samediff"
    "dl4jcore"
    "keras"
    "libnd4j"
    "onnx"
    "tensorflow"
    "ggml"
    "integration"
    "llm"
    "vlm"
    "zoo"
    "longrunning"
)

PASS=()
FAIL=()

for cat in "${CATEGORIES[@]}"; do
    script="./run-${cat}-tests.sh"
    echo "========================================"
    echo "  Running: ${cat}"
    echo "========================================"
    if "$script" "$@"; then
        PASS+=("$cat")
    else
        FAIL+=("$cat")
    fi
    echo ""
done

echo "========================================"
echo "  SUMMARY"
echo "========================================"
echo "  Passed: ${PASS[*]:-none}"
echo "  Failed: ${FAIL[*]:-none}"
echo "========================================"

if [ ${#FAIL[@]} -gt 0 ]; then
    exit 1
fi
