#!/bin/bash
# ===========================================================================
# ONNX Import Tests — ONNX model import and op conversion
# ===========================================================================
# Covers: org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx.**
#   - ONNX model converter tests
#   - Multi-head attention import
#
# Usage:
#   ./run-onnx-tests.sh                               # Run all ONNX tests
#   ./run-onnx-tests.sh -Dtest=TestOnnxConverter       # Run a specific test
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-onnx-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee onnx-test-output.log
else
    mvn test \
        '-Dtest=org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx.**' \
        "$@" 2>&1 | tee onnx-test-output.log
fi
