#!/bin/bash
# Video VLM Benchmark Suite
# Runs video vision-language model benchmarks across models and execution modes.
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ Benchmarks: GGUF import, video frame extraction, vision encoding,  │
# │             embedding merge, end-to-end video generation            │
# │                                                                     │
# │ Default models: smolvlm2-256m                                      │
# │ Also available: smolvlm2-2.2b, qwen3-vl-2b, minicpm-v-4.5        │
# │ Default backend: CUDA                                              │
# │ Default tokens: 20                                                  │
# └─────────────────────────────────────────────────────────────────────┘
#
# Usage: ./run-video-vlm-benchmarks.sh [OPTIONS]
#
# Test selection:
#   --test TEST           Run specific test (default: all)
#                         Valid: registration, preprocessing, pipeline, generation, all
#
# Model selection:
#   --models MODELS       Comma-separated model keys (default: smolvlm2-256m)
#                         Valid: smolvlm2-256m, smolvlm2-2.2b, qwen3-vl-2b,
#                                minicpm-v-4.5, all
#
# Configuration:
#   --tokens N            Max decode tokens (default: 20)
#   --frames N            Max video frames (default: 16)
#
# Backend:
#   --backend cuda|cpu    Select backend (default: cuda)
#
# Options:
#   --debug               Enable DSP diagnostics at FULL level
#   --op-timing           Enable native op timing
#   --verbose             Verbose maven output
#
# Examples:
#   ./run-video-vlm-benchmarks.sh                                       # All tests, SmolVLM2-256M
#   ./run-video-vlm-benchmarks.sh --test registration                   # Registration tests only
#   ./run-video-vlm-benchmarks.sh --test preprocessing --frames 32      # Preprocessing with 32 frames
#   ./run-video-vlm-benchmarks.sh --test pipeline --models qwen3-vl-2b  # Qwen3-VL pipeline
#   ./run-video-vlm-benchmarks.sh --backend cpu --test registration     # CPU-only registration
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MVN="/home/agibsonccc/dev-apps/mvn/bin/mvn"
SMOLVLM2_TEST="TestSmolVLM2VideoGeneration"
QWEN3VL_TEST="TestQwen3VLVideoGeneration"
TEST_PKG="org.eclipse.deeplearning4j.nd4j.autodiff.samediff.vlm"
LOG_FILE="$SCRIPT_DIR/video-vlm-bench-output.log"

# Defaults
BACKEND="cuda"
TEST_NAME="all"
MODELS="smolvlm2-256m"
MAX_TOKENS=20
MAX_FRAMES=16
DEBUG_MODE=false
OP_TIMING=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --test)
            TEST_NAME="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --op-timing)
            OP_TIMING=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            head -40 "$0" | tail -n +2 | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# ─── Map test name to JUnit test classes/methods ──────────────────────

build_test_selector() {
    local test_name="$1"
    local models="$2"

    # Determine which test classes to run based on model selection
    local smolvlm2=false
    local qwen3vl=false

    IFS=',' read -ra MODEL_ARRAY <<< "$models"
    for model in "${MODEL_ARRAY[@]}"; do
        case "$model" in
            smolvlm2*) smolvlm2=true ;;
            qwen3*) qwen3vl=true ;;
            minicpm*) qwen3vl=true ;; # MiniCPM-V uses similar pipeline to Qwen
            all) smolvlm2=true; qwen3vl=true ;;
        esac
    done

    local selectors=""

    if [ "$test_name" = "all" ]; then
        if [ "$smolvlm2" = true ]; then
            selectors="$SMOLVLM2_TEST"
        fi
        if [ "$qwen3vl" = true ]; then
            if [ -n "$selectors" ]; then
                selectors="$selectors,$QWEN3VL_TEST"
            else
                selectors="$QWEN3VL_TEST"
            fi
        fi
    elif [ "$test_name" = "registration" ]; then
        if [ "$smolvlm2" = true ]; then
            selectors="${SMOLVLM2_TEST}#testSmolVLM2ArchitectureRegistration+testSmolVLM2SafeTensorsRegistration"
        fi
        if [ "$qwen3vl" = true ]; then
            local qsel="${QWEN3VL_TEST}#testQwen3VLArchitectureRegistration+testQwen3VLSafeTensorsRegistration"
            if [ -n "$selectors" ]; then
                selectors="$selectors,$qsel"
            else
                selectors="$qsel"
            fi
        fi
    elif [ "$test_name" = "preprocessing" ]; then
        if [ "$smolvlm2" = true ]; then
            selectors="${SMOLVLM2_TEST}#testVideoPreprocessorForSmolVLM2+testVideoFrameSamplerDefaults+testPixelShuffleProjector"
        fi
        if [ "$qwen3vl" = true ]; then
            local qsel="${QWEN3VL_TEST}#testVideoPreprocessorQwen3VL+testVideoFrameSamplerQwen3VL+testTemporalPatchEmbedCreation"
            if [ -n "$selectors" ]; then
                selectors="$selectors,$qsel"
            else
                selectors="$qsel"
            fi
        fi
    elif [ "$test_name" = "pipeline" ]; then
        if [ "$smolvlm2" = true ]; then
            selectors="${SMOLVLM2_TEST}#testEmbeddingMergerVideoTokens+testImagePromptBuilderVideoToken"
        fi
        if [ "$qwen3vl" = true ]; then
            local qsel="${QWEN3VL_TEST}#testFusedMRoPEOpCreation+testDeepStackMergerCreation"
            if [ -n "$selectors" ]; then
                selectors="$selectors,$qsel"
            else
                selectors="$qsel"
            fi
        fi
    elif [ "$test_name" = "generation" ]; then
        # End-to-end generation tests (require model download)
        if [ "$smolvlm2" = true ]; then
            selectors="${SMOLVLM2_TEST}"
        fi
        if [ "$qwen3vl" = true ]; then
            if [ -n "$selectors" ]; then
                selectors="$selectors,$QWEN3VL_TEST"
            else
                selectors="$QWEN3VL_TEST"
            fi
        fi
    else
        echo "ERROR: Unknown test '$test_name'"
        echo "Valid tests: registration, preprocessing, pipeline, generation, all"
        exit 1
    fi

    echo "$selectors"
}

TEST_SELECTOR=$(build_test_selector "$TEST_NAME" "$MODELS")

if [ -z "$TEST_SELECTOR" ]; then
    echo "ERROR: No tests selected for models '$MODELS'"
    exit 1
fi

# ─── Backend configuration ───────────────────────────────────────────

TRITON_FLAG=""
if [ "$BACKEND" = "cpu" ]; then
    BACKEND_ARTIFACT="nd4j-native"
    echo "[backend] CPU mode"
elif [ "$BACKEND" = "cuda" ]; then
    BACKEND_ARTIFACT="nd4j-cuda-12.9"
    TRITON_FLAG="-Dlibnd4j.triton=ON"
    echo "[backend] CUDA mode"
else
    echo "ERROR: Unknown backend '$BACKEND'. Use 'cuda' or 'cpu'."
    exit 1
fi

# ─── Build Maven args ────────────────────────────────────────────────

EXTRA_ARGS=""

# Video-specific settings
EXTRA_ARGS="$EXTRA_ARGS -Dvideo.max.frames=$MAX_FRAMES"
echo "[frames] $MAX_FRAMES"
echo "[models] $MODELS"

# Debug mode
if [ "$DEBUG_MODE" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full"
    echo "[debug] DSP diagnostics enabled at FULL level"
fi

# Op timing
if [ "$OP_TIMING" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.execution.timing=true"
    echo "[timing] Op timing enabled"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  VIDEO VLM BENCHMARK SUITE"
echo "═══════════════════════════════════════════════════════════"
echo "  Test:     $TEST_NAME"
echo "  Selector: $TEST_SELECTOR"
echo "  Backend:  $BACKEND ($BACKEND_ARTIFACT)"
echo "  Tokens:   $MAX_TOKENS"
echo "  Frames:   $MAX_FRAMES"
echo "  Log:      $LOG_FILE"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ─── Run benchmark ───────────────────────────────────────────────────

set +e
$MVN test \
  -Dtest="$TEST_SELECTOR" \
  -Dbench.max.tokens="$MAX_TOKENS" \
  $TRITON_FLAG \
  -Dbackend.artifactId=$BACKEND_ARTIFACT \
  $EXTRA_ARGS \
  2>&1 | tee "$LOG_FILE"
BUILD_RESULT=${PIPESTATUS[0]}
set -e

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  RESULTS"
echo "═══════════════════════════════════════════════════════════"

if [ $BUILD_RESULT -eq 0 ]; then
    echo "  STATUS: PASSED"
else
    echo "  STATUS: FAILED (exit code $BUILD_RESULT)"
fi

# ─── Parse test results from log ────────────────────────────────────

if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
    echo ""
    echo "  Test Summary:"
    grep -E "Tests run:|BUILD (SUCCESS|FAILURE)" "$LOG_FILE" | tail -5 || true
    echo ""

    # Extract any benchmark timing lines
    echo "  Video VLM Metrics:"
    grep -iE "(tok/s|frames?/s|vision tokens|preprocessing|total time)" "$LOG_FILE" | tail -20 || true
fi

echo ""
echo "  Full log: $LOG_FILE"
echo "═══════════════════════════════════════════════════════════"

exit $BUILD_RESULT
