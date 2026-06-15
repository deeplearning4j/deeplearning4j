#!/bin/bash
# ===========================================================================
# VLM Tests — Vision Language Model pipeline and preprocessing
# ===========================================================================
# Covers: org.eclipse.deeplearning4j.vlm.**
#   - SmolDocling optimized pipeline
#   - VLM decode quality
#   - ONNX reference comparison
#   - Model import pipeline
#   - PDF/image loading
#   - Image preprocessing, tiling, prompt building
#   - Embedding merger, vision encoder
#
# Usage:
#   ./run-vlm-tests.sh                                       # Run all VLM tests
#   ./run-vlm-tests.sh -Dtest=TestSmolDoclingOptimizedPipeline # Run a specific test
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-vlm-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee vlm-test-output.log
else
    mvn test \
        '-Dtest=org.eclipse.deeplearning4j.vlm.**' \
        "$@" 2>&1 | tee vlm-test-output.log
fi
