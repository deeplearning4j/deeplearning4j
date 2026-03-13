#!/bin/bash
# ===========================================================================
# Keras Import Tests — Keras model import and layer conversion
# ===========================================================================
# Covers: org.eclipse.deeplearning4j.frameworkimport.keras.**
#   - Layer tests (activations, convolution, core, embeddings, normalization, pooling, recurrent)
#   - Configuration tests (model serialization, JSON/HDF5 parsing)
#   - End-to-end model import tests
#   - Preprocessing (sequence, text)
#   - Weight loading, optimizer mapping
#
# Usage:
#   ./run-keras-tests.sh                             # Run all keras tests
#   ./run-keras-tests.sh -Dtest=KerasModelImportTest  # Run a specific test
#   ./run-keras-tests.sh -Dtest='*Conv*'              # Run tests matching pattern
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-keras-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee keras-test-output.log
else
    mvn test \
        '-Dtest=org.eclipse.deeplearning4j.frameworkimport.keras.**' \
        "$@" 2>&1 | tee keras-test-output.log
fi
