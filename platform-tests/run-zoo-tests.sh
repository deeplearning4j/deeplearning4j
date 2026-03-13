#!/bin/bash
# ===========================================================================
# Zoo Tests — Model zoo instantiation and validation
# ===========================================================================
# Covers: org.eclipse.deeplearning4j.zoo.**
#   - Model instantiation tests
#   - Miscellaneous zoo tests
#
# Usage:
#   ./run-zoo-tests.sh                                # Run all zoo tests
#   ./run-zoo-tests.sh -Dtest=TestInstantiation        # Run a specific test
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-zoo-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee zoo-test-output.log
else
    mvn test \
        '-Dtest=org.eclipse.deeplearning4j.zoo.**' \
        "$@" 2>&1 | tee zoo-test-output.log
fi
