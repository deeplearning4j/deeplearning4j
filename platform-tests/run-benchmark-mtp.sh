#!/bin/bash
# Qwen3.5 bundled-MTP benchmark.
#
# This is deliberately separate from run-benchmark.sh, which measures the
# SmolDocling VLM workload and does not contain a Qwen3.5 NextN predictor.
#
# The JUnit oracle uses the MTP-enabled Qwen3.5-0.8B Q4_K_M GGUF, one fixed
# buffer pipeline, K=4 (W=5), two warmup generations, then:
#   1. a measured native-MTP generation;
#   2. a measured greedy generation on the same frozen plan and buffers;
#   3. exact token-by-token comparison.
#
# JUnit owns all validation and metrics. This launcher intentionally contains
# no Python log parser: Maven succeeds only when MTP proposed and accepted
# tokens and remained exactly lossless.
#
# Usage:
#   ./run-benchmark-mtp.sh
#   ./run-benchmark-mtp.sh --backend cuda --tokens 250
#   ./run-benchmark-mtp.sh --backend cpu --tokens 250
#   ./run-benchmark-mtp.sh --tokens 60 --log /tmp/mtp-debug.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MVN="/home/agibsonccc/dev-apps/mvn/bin/mvn"
TEST_SELECTOR="TestQwen35MtpDecode#testBundledMtpIsLosslessAndEngaged"

BACKEND="cuda"
TOKENS=250
LOG_FILE=""

usage() {
    printf '%s\n' \
        "Qwen3.5 fixed-buffer bundled-MTP benchmark" \
        "" \
        "Usage: ./run-benchmark-mtp.sh [OPTIONS]" \
        "" \
        "  --backend cuda|cpu   Backend (default: cuda)" \
        "  --tokens N           Generated tokens (default: 250)" \
        "  --log FILE           Tee log (default: /tmp/mtp-qwen-<backend>-<tokens>-<timestamp>.log)" \
        "  -h, --help           Show this help" \
        "" \
        "The curated workload is fixed to the MTP-enabled Qwen3.5-0.8B Q4_K_M" \
        "artifact with K=4 (W=5). The JUnit run fails if native MTP does not" \
        "propose and accept tokens or if its output differs from greedy."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            [[ $# -ge 2 ]] || { echo "ERROR: --backend requires a value" >&2; exit 2; }
            BACKEND="$2"
            shift 2
            ;;
        --tokens)
            [[ $# -ge 2 ]] || { echo "ERROR: --tokens requires a value" >&2; exit 2; }
            TOKENS="$2"
            shift 2
            ;;
        --log)
            [[ $# -ge 2 ]] || { echo "ERROR: --log requires a value" >&2; exit 2; }
            LOG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "$TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --tokens must be a positive integer, got '$TOKENS'" >&2
    exit 2
fi

TRITON_FLAG=""
case "$BACKEND" in
    cuda)
        BACKEND_ARTIFACT="nd4j-cuda-12.9"
        TRITON_FLAG="-Dlibnd4j.triton=ON"
        ;;
    cpu)
        BACKEND_ARTIFACT="nd4j-native"
        ;;
    *)
        echo "ERROR: --backend must be cuda or cpu, got '$BACKEND'" >&2
        exit 2
        ;;
esac

if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="/tmp/mtp-qwen-$BACKEND-$TOKENS-$(date +%Y%m%d-%H%M%S).log"
fi

if (( TOKENS < 250 )); then
    RUN_CLASS="DEBUG/CORRECTNESS (not performance evidence)"
else
    RUN_CLASS="PERFORMANCE"
fi

printf '%s\n' \
    "═══════════════════════════════════════════════════════════" \
    "  QWEN3.5 BUNDLED-MTP BENCHMARK" \
    "═══════════════════════════════════════════════════════════" \
    "  Workload: MTP-enabled Qwen3.5-0.8B Q4_K_M" \
    "  Policy:   native NextN MTP K=4 (W=5) vs greedy" \
    "  Buffers:  fixed prefill/KV; same frozen plan for both" \
    "  Backend:  $BACKEND ($BACKEND_ARTIFACT)" \
    "  Tokens:   $TOKENS" \
    "  Class:    $RUN_CLASS" \
    "  Log:      $LOG_FILE" \
    "═══════════════════════════════════════════════════════════"

set +e
"$MVN" test \
  -Dtest="$TEST_SELECTOR" \
  -Dbench.max.tokens="$TOKENS" \
  -Dbackend.artifactId="$BACKEND_ARTIFACT" \
  $TRITON_FLAG \
  -Dnd4j.optimizer.enabled=true \
  -Dtest.maxphysicalbytes=48g \
  2>&1 | tee "$LOG_FILE"
BUILD_RESULT=${PIPESTATUS[0]}
set -e

if (( BUILD_RESULT != 0 )); then
    printf '%s\n' \
        "MTP benchmark FAILED (Maven exit $BUILD_RESULT)." \
        "Full log: $LOG_FILE"
    exit "$BUILD_RESULT"
fi

printf '%s\n' \
    "MTP benchmark PASSED: native MTP engaged and remained token-exact." \
    "Metrics and full output: $LOG_FILE"
