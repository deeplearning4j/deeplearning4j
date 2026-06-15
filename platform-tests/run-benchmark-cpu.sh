#!/bin/bash
# SmolDocling VLM Decode Benchmark — CPU backend (nd4j-native)
#
# Thin wrapper around run-benchmark.sh with --backend cpu.
# All options are forwarded — see run-benchmark.sh --help for details.
#
# Usage: ./run-benchmark-cpu.sh [OPTIONS]
#
# Examples:
#   ./run-benchmark-cpu.sh                           # Default: 250 tokens, FP16 ON
#   ./run-benchmark-cpu.sh --tokens 50               # Quick 50-token run
#   ./run-benchmark-cpu.sh --debug                   # Full DSP diagnostics
#   ./run-benchmark-cpu.sh --no-fp16                 # FP32 weights baseline
set -euo pipefail

exec "$(dirname "$0")/run-benchmark.sh" --backend cpu "$@"
