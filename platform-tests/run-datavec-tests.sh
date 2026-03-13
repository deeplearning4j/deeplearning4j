#!/bin/bash
# ===========================================================================
# DataVec Tests — Data ingestion, transformation, and ETL
# ===========================================================================
# Covers: org.datavec.**
#   - Record readers (CSV, JSON, LibSVM, Regex, etc.)
#   - Record writers
#   - Data transforms (conditions, filters, joins, reduce, schema)
#   - Image loading and transforms
#   - Arrow, Excel, JDBC connectors
#   - Local transforms (analysis, functions, sequences)
#
# Usage:
#   ./run-datavec-tests.sh                    # Run all datavec tests
#   ./run-datavec-tests.sh -Dtest=FooTest     # Run a specific test
#   ./run-datavec-tests.sh -Dtest='*CSV*'     # Run tests matching pattern
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-datavec-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee datavec-test-output.log
else
    mvn test \
        '-Dtest=org.datavec.**' \
        "$@" 2>&1 | tee datavec-test-output.log
fi
