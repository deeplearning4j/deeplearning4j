#!/bin/bash
# ===========================================================================
# Long-Running Tests — Tests that download large datasets or take a long time
# ===========================================================================
# Covers: org.eclipse.deeplearning4j.longrunning.**
#   - Dataset downloads (MNIST, SVHN, EMNIST, Strumpf)
#   - Concurrent downloader tests
#   - TensorFlow zoo model tests
#   - Random/stress tests
#
# NOTE: These tests may download large files and take significant time.
#
# Usage:
#   ./run-longrunning-tests.sh                               # Run all
#   ./run-longrunning-tests.sh -Dtest=MnistFetcherTest        # Run a specific test
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-longrunning-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee longrunning-test-output.log
else
    mvn test \
        '-Dtest=org.eclipse.deeplearning4j.longrunning.**' \
        "$@" 2>&1 | tee longrunning-test-output.log
fi
