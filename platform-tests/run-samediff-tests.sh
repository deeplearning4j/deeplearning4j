#!/bin/bash
# ===========================================================================
# SameDiff Tests — Automatic differentiation and graph computation
# ===========================================================================
# Covers: org.eclipse.deeplearning4j.nd4j.autodiff.**
#   - SameDiff graph construction and execution
#   - Op validation (transforms, reductions, shape ops, random, etc.)
#   - Gradient verification
#   - Serialization
#   - Listeners (checkpoint, profiling, debugging)
#   - Pipeline import (ONNX, GGML, SafeTensors)
#   - Control flow, mixed precision, transfer learning
#   - Optimization passes
#
# Usage:
#   ./run-samediff-tests.sh                        # Run all samediff tests
#   ./run-samediff-tests.sh -Dtest=SameDiffTests   # Run a specific test
#   ./run-samediff-tests.sh -Dtest='*OpValidation*' # Run tests matching pattern
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-samediff-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee samediff-test-output.log
else
    mvn test \
        '-Dtest=org.eclipse.deeplearning4j.nd4j.autodiff.**' \
        "$@" 2>&1 | tee samediff-test-output.log
fi
