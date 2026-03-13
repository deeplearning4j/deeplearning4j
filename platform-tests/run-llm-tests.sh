#!/bin/bash
# ===========================================================================
# LLM Tests — Large Language Model generation and decoding
# ===========================================================================
# Covers: org.eclipse.deeplearning4j.llm.**
#   - Batch compaction and generation state
#   - Decoder utilities
#   - Sampling and sampler utilities
#   - N-gram speculative decoding
#   - SameDiff memory utilities
#   - Qwen 3.5 pipeline test
#
# Usage:
#   ./run-llm-tests.sh                              # Run all LLM tests
#   ./run-llm-tests.sh -Dtest=SamplerTest            # Run a specific test
#
# Backend selection (default: nd4j-cuda-12.9):
#   ./run-llm-tests.sh -Dbackend.artifactId=nd4j-native
# ===========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if echo "$@" | grep -q '\-Dtest='; then
    mvn test "$@" 2>&1 | tee llm-test-output.log
else
    mvn test \
        '-Dtest=org.eclipse.deeplearning4j.llm.**' \
        "$@" 2>&1 | tee llm-test-output.log
fi
