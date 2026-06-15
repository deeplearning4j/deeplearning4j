#!/bin/bash
# ===========================================================================
# GGML Tests — GGML/GGUF model format import, export, and quantization
# ===========================================================================
# Covers: org.eclipse.deeplearning4j.ggml.**
#   - GGUF reader/writer
#   - Model import/export
#   - Architecture detection and export
#   - Quantization and dequantization
#   - Round-trip tests
#   - Format detection
#
# Usage:
#   ./run-ggml-tests.sh                              # Run all GGML tests
#   ./run-ggml-tests.sh -Dtest=GGUFReaderTest         # Run a specific test
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-ggml-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee ggml-test-output.log
else
    mvn test \
        '-Dtest=org.eclipse.deeplearning4j.ggml.**' \
        "$@" 2>&1 | tee ggml-test-output.log
fi
