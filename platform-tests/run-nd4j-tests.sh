#!/bin/bash
# ===========================================================================
# ND4J Tests — N-dimensional array operations and linear algebra
# ===========================================================================
# Covers: org.eclipse.deeplearning4j.nd4j.linalg.** + org.eclipse.deeplearning4j.nd4j.evaluation.**
#         + org.nd4j.**
#   - NDArray operations (creation, indexing, broadcasting, slicing)
#   - BLAS operations
#   - Data buffers and memory management
#   - Workspaces
#   - Random number generation
#   - Shape operations
#   - Serialization
#   - Loss functions, activations, learning rate schedules
#   - CUDA/GPU-specific tests
#   - Arrow, TFLite, Python4J interop
#
# Usage:
#   ./run-nd4j-tests.sh                       # Run all nd4j tests
#   ./run-nd4j-tests.sh -Dtest=Nd4jTestsC     # Run a specific test
#   ./run-nd4j-tests.sh -Dtest='*Workspace*'  # Run tests matching pattern
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-nd4j-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee nd4j-test-output.log
else
    mvn test \
        '-Dtest=org.eclipse.deeplearning4j.nd4j.linalg.**,org.eclipse.deeplearning4j.nd4j.evaluation.**,org.nd4j.**,org.eclipse.deeplearning4j.frameworkimport.nd4j.**' \
        "$@" 2>&1 | tee nd4j-test-output.log
fi
