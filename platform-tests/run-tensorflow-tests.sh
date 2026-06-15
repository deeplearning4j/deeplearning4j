#!/bin/bash
# ===========================================================================
# TensorFlow Import Tests — TensorFlow model import and graph conversion
# ===========================================================================
# Covers: org.eclipse.deeplearning4j.frameworkimport.tensorflow.**
#         + org.eclipse.deeplearning4j.longrunning.frameworkimport.tensorflow.**
#   - TF graph import tests
#   - TF model zoo tests
#   - TF listener tests
#
# Usage:
#   ./run-tensorflow-tests.sh                     # Run all TF tests
#   ./run-tensorflow-tests.sh -Dtest=FooTest      # Run a specific test
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-tensorflow-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee tensorflow-test-output.log
else
    mvn test \
        '-Dtest=org.eclipse.deeplearning4j.frameworkimport.tensorflow.**,org.eclipse.deeplearning4j.longrunning.frameworkimport.tensorflow.**' \
        "$@" 2>&1 | tee tensorflow-test-output.log
fi
