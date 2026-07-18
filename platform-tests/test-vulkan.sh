#!/bin/bash
# Vulkan Backend Testing & Validation Script
#
# Usage: ./test-vulkan.sh [OPTIONS]
#
# Test types:
#   --accuracy      Run output accuracy validation (token-level correctness)
#   --decode-step   Validate decode step-by-step correctness
#   --benchmark     Run 250-token VLM decode benchmark
#   --matrix        Run DSP configuration matrix (8 configs)
#   --full-dsp-gate Run complete DSP regression gate
#   --all           Run all validations and benchmarks (comprehensive)
#
# Common options:
#   --tokens N      Override decode token count (default: per-test)
#   --config NAME   Override DSP config (SLOT_BY_SLOT, OPTIMAL, TRITON, CUDA_GRAPHS, AUTO)
#   --diag-json FILE  Write diagnostic report to JSON file
#   --no-cleanup    Don't clear DSP cache between runs

set -euo pipefail

# Configuration
TEST_TYPE="all"
TOKENS=""
CONFIG=""
DIAG_JSON=""
CLEANUP="true"
PLATFORM_TESTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MVN="${MVN:-mvn}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --accuracy)
      TEST_TYPE="accuracy"
      shift
      ;;
    --decode-step)
      TEST_TYPE="decode-step"
      shift
      ;;
    --benchmark)
      TEST_TYPE="benchmark"
      shift
      ;;
    --matrix)
      TEST_TYPE="matrix"
      shift
      ;;
    --full-dsp-gate)
      TEST_TYPE="full-dsp-gate"
      shift
      ;;
    --all)
      TEST_TYPE="all"
      shift
      ;;
    --tokens)
      TOKENS="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --diag-json)
      DIAG_JSON="$2"
      shift 2
      ;;
    --no-cleanup)
      CLEANUP="false"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: ./test-vulkan.sh [--accuracy|--decode-step|--benchmark|--matrix|--full-dsp-gate|--all] [OPTIONS]"
      exit 1
      ;;
  esac
done

# Change to platform-tests directory
cd "$PLATFORM_TESTS_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Vulkan Backend Testing — $TEST_TYPE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Backends:     CUDA + Vulkan"
echo "Test type:    $TEST_TYPE"
echo "Cleanup:      $CLEANUP"
[ -n "$TOKENS" ] && echo "Tokens:       $TOKENS"
[ -n "$CONFIG" ] && echo "Config:       $CONFIG"
[ -n "$DIAG_JSON" ] && echo "Diag JSON:    $DIAG_JSON"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Keep CUDA and Vulkan on the same classpath and exercise the universal
# Triton/DSP replay stack used by both device backends.
MVN_PROPS="-Ptest-vulkan,test-cuda-vulkan-coexistence -Dlibnd4j.triton=ON"
[ -n "$CONFIG" ] && MVN_PROPS="$MVN_PROPS -Dnd4j.dsp.graphExecutionMode=$CONFIG"
[ -n "$DIAG_JSON" ] && MVN_PROPS="$MVN_PROPS -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full -Dnd4j.dsp.diagnostics.file=$DIAG_JSON"

# Run tests based on type
case $TEST_TYPE in
  accuracy)
    echo "Running output accuracy validation..."
    $MVN test $MVN_PROPS \
      -Dtest=TestDspValidation#outputAccuracy \
      -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full \
      2>&1 | tee /tmp/vulkan-accuracy-test.log
    ;;

  decode-step)
    echo "Running decode step validation..."
    $MVN test $MVN_PROPS \
      -Dtest=TestDspValidation#decodeStep \
      -Dnd4j.dsp.diagnostics=EXECUTE,TIMING \
      2>&1 | tee /tmp/vulkan-decode-step-test.log
    ;;

  benchmark)
    echo "Running VLM decode benchmark (250 tokens)..."
    ./run-benchmark.sh --backend vulkan \
      --tokens ${TOKENS:-250} \
      --op-timing \
      ${CONFIG:+--config "$CONFIG"} \
      ${DIAG_JSON:+--diag-json "$DIAG_JSON"} \
      2>&1 | tee /tmp/vulkan-benchmark.log
    ;;

  matrix)
    echo "Running DSP configuration matrix (8 configs)..."
    ./run-dsp-matrix.sh \
      --backend vulkan \
      ${DIAG_JSON:+--diag-json "$DIAG_JSON"} \
      2>&1 | tee /tmp/vulkan-dsp-matrix.log
    ;;

  full-dsp-gate)
    echo "Running full DSP regression gate..."
    echo "This will run comprehensive DSP lifecycle tests..."
    $MVN test $MVN_PROPS \
      -Dtest=DspHandleDataModelTest,DspBufferAliasAccuracyTest,DspHandleTest,\
DspLifecycleExhaustiveTest,DspLifecycleValidationTest,DspFrozenConstantInvariantTest,\
DspExtInputStalenessTest,DspSlotLifecycleAuditTest,DspConcurrentPlanSharingTest,\
DspCompositeReplayTest,TestDspShapePrePass \
      -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full \
      2>&1 | tee /tmp/vulkan-dsp-gate.log
    ;;

  all)
    echo "Running comprehensive test suite..."
    echo ""
    echo "1. Output accuracy validation..."
    $MVN test $MVN_PROPS \
      -Dtest=TestDspValidation#outputAccuracy \
      -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full \
      2>&1 | tee /tmp/vulkan-all-1-accuracy.log
    
    echo ""
    echo "2. Decode step validation..."
    $MVN test $MVN_PROPS \
      -Dtest=TestDspValidation#decodeStep \
      -Dnd4j.dsp.diagnostics=EXECUTE,TIMING \
      2>&1 | tee /tmp/vulkan-all-2-decode-step.log
    
    echo ""
    echo "3. VLM decode benchmark (250 tokens)..."
    ./run-benchmark.sh --backend vulkan \
      --tokens 250 \
      --op-timing \
      ${CONFIG:+--config "$CONFIG"} \
      ${DIAG_JSON:+--diag-json "/tmp/vulkan-all-3-benchmark.json"} \
      2>&1 | tee /tmp/vulkan-all-3-benchmark.log
    
    echo ""
    echo "4. DSP configuration matrix..."
    ./run-dsp-matrix.sh \
      --backend vulkan \
      ${DIAG_JSON:+--diag-json "/tmp/vulkan-all-4-matrix.json"} \
      2>&1 | tee /tmp/vulkan-all-4-matrix.log
    ;;
esac

# Check results
if [ $? -eq 0 ]; then
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "✓ Vulkan tests PASSED"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "Log files:"
  ls -lh /tmp/vulkan-*.log 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
  [ -f "$DIAG_JSON" ] && echo "  Diagnostics: $DIAG_JSON"
  echo ""
  exit 0
else
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "✗ Vulkan tests FAILED"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "Debug steps:"
  echo "  1. Check log files in /tmp/vulkan-*.log"
  echo "  2. Enable diagnostics: --diag-json /tmp/diag.json"
  echo "  3. Check DSP diagnostics: grep 'DSP_DIAG' /tmp/vulkan-*.log"
  echo ""
  exit 1
fi
