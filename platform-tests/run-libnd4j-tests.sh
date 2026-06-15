#!/bin/bash
# ===========================================================================
# Libnd4j Native Tests — C++ GTest suite run via Java harness
# ===========================================================================
# Covers: org.eclipse.deeplearning4j.nd4j.libnd4j.**
#   - LibNd4jNativeTestRunner — executes the native GTest binary
#   - LibNd4jCategoryTests — categorized native test suites
#
# Prerequisites:
#   The native test binary must be built first:
#     mvn -Pcpu -Dlibnd4j.buildthreads=12 -Dlibnd4j.tests=--tests \
#       -pl libnd4j,:nd4j-cpu-backend-common,:nd4j-native \
#       clean install -DskipTests
#
# Usage:
#   ./run-libnd4j-tests.sh                                          # All native tests
#   ./run-libnd4j-tests.sh -Dtest=LibNd4jNativeTestRunner           # Via runner
#   ./run-libnd4j-tests.sh -Dlibnd4j.test.filter='ArrayOptionsTests.*'  # Filter
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-libnd4j-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee libnd4j-test-output.log
else
    mvn test \
        '-Dtest=org.eclipse.deeplearning4j.nd4j.libnd4j.**' \
        "$@" 2>&1 | tee libnd4j-test-output.log
fi
