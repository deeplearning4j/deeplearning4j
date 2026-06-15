#!/bin/bash
# ===========================================================================
# DL4J Core Tests — Old DL4J API (neural networks, layers, graph, training)
# ===========================================================================
# Covers: org.eclipse.deeplearning4j.dl4jcore.** + org.deeplearning4j.**
#   - Neural network layers (feedforward, conv, recurrent, normalization, pooling)
#   - ComputationGraph, MultiLayerNetwork
#   - Transfer learning, gradient checking, early stopping
#   - Datasets, iterators, evaluation
#   - Word2Vec, ParagraphVectors, text processing
#   - UI, serialization, regression tests
#
# Usage:
#   ./run-dl4jcore-tests.sh                    # Run all dl4jcore tests
#   ./run-dl4jcore-tests.sh -Dtest=FooTest     # Run a specific test
#   ./run-dl4jcore-tests.sh -Dtest='*Conv*'    # Run tests matching pattern
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-dl4jcore-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# If user passed -Dtest=..., use that. Otherwise run the full category.
if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee dl4jcore-test-output.log
else
    mvn test \
        '-Dtest=org.eclipse.deeplearning4j.dl4jcore.**,org.deeplearning4j.**' \
        "$@" 2>&1 | tee dl4jcore-test-output.log
fi
