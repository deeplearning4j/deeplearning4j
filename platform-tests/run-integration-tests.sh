#!/bin/bash
# ===========================================================================
# Integration Tests — End-to-end DL4J + SameDiff integration
# ===========================================================================
# Covers: org.eclipse.deeplearning4j.integration.**
#   - DL4J integration (CNN 1D/2D/3D, MLP, RNN, unsupervised)
#   - SameDiff integration (CNN, MLP, RNN)
#   - Baseline generation and validation
#
# Usage:
#   ./run-integration-tests.sh                            # Run all
#   ./run-integration-tests.sh -Dtest=IntegrationTestsDL4J # Run a specific test
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-integration-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee integration-test-output.log
else
    mvn test \
        '-Dtest=org.eclipse.deeplearning4j.integration.**' \
        "$@" 2>&1 | tee integration-test-output.log
fi
