#!/bin/bash
# SmolDocling DSP Accuracy Validation — compare execution modes for correctness
#
# Validates that different DSP execution modes (SLOT_BY_SLOT, TRITON, OPTIMAL)
# produce identical or near-identical output. Uses the DSP Validation Framework
# with slot output interceptors and per-step token comparison.
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ VALIDATION TESTS                                                    │
# │                                                                     │
# │  outputAccuracy:    Compare generated tokens across configs         │
# │  perOpSlot:         Interceptor captures during OPTIMAL decode      │
# │  decodeStep:        SLOT_BY_SLOT vs OPTIMAL step-by-step compare    │
# │  tf32Isolation:     Same config with/without TF32                   │
# │                                                                     │
# │  Default: run ALL tests                                             │
# └─────────────────────────────────────────────────────────────────────┘
#
# Usage: ./run-validation.sh [OPTIONS]
#
# Test selection:
#   --test NAME         Run specific test: outputAccuracy, perOpSlot, decodeStep,
#                       tf32Isolation, or ALL (default: ALL)
#
# Configuration:
#   --tokens N          Override max decode tokens per test (default: per-test)
#   --configs LIST      Comma-separated configs for outputAccuracy
#                       (e.g., SLOT_BY_SLOT,OPTIMAL — default: all available)
#   --tolerance NAME    Tolerance preset: standard, strict, tf32 (default: standard)
#   --match-rate N      Minimum token match rate percent (default: 90)
#   --verbose           Enable per-step token logging
#
# Optimizer options:
#   --fp16              Enable FP16 weight pre-casting (DEFAULT: ON)
#   --no-fp16           Disable FP16 pre-casting
#   --no-optimizer      Disable the GraphOptimizer entirely
#
# Debug:
#   --debug             Enable DSP diagnostics and verbose tracing
#
# Examples:
#   ./run-validation.sh                               # Run all validation tests
#   ./run-validation.sh --test outputAccuracy          # Only output accuracy tests
#   ./run-validation.sh --tokens 20 --verbose          # 20 tokens with per-step log
#   ./run-validation.sh --configs OPTIMAL              # Only test OPTIMAL config
#   ./run-validation.sh --tolerance strict             # Strict tolerance thresholds
#   ./run-validation.sh --test tf32Isolation --tokens 10  # TF32 test, 10 tokens
#   ./run-validation.sh --test decodeStep --match-rate 95 # 95% match required
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MVN="/home/agibsonccc/dev-apps/mvn/bin/mvn"
TEST_CLASS="TestDspValidation"
LOG_FILE="$SCRIPT_DIR/validation-output.log"
SUREFIRE_OUT="$SCRIPT_DIR/target/surefire-reports/org.eclipse.deeplearning4j.llm.generation.${TEST_CLASS}-output.txt"

# Defaults
TEST_NAME="ALL"
MAX_TOKENS=""
CONFIGS=""
TOLERANCE="standard"
MATCH_RATE=""
VERBOSE=false
DEBUG_MODE=false
FP16=true
NO_OPTIMIZER=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            TEST_NAME="$2"
            shift 2
            ;;
        --tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --configs)
            CONFIGS="$2"
            shift 2
            ;;
        --tolerance)
            TOLERANCE="$2"
            shift 2
            ;;
        --match-rate)
            MATCH_RATE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --fp16)
            FP16=true
            shift
            ;;
        --no-fp16)
            FP16=false
            shift
            ;;
        --no-optimizer)
            NO_OPTIMIZER=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run-validation.sh [--test NAME] [--tokens N] [--configs LIST]"
            echo "       [--tolerance NAME] [--match-rate N] [--verbose] [--debug]"
            echo "       [--fp16] [--no-fp16] [--no-optimizer]"
            exit 1
            ;;
    esac
done

# Map test name to JUnit test method(s)
case "$TEST_NAME" in
    ALL|all)
        TEST_SELECTOR="${TEST_CLASS}"
        ;;
    outputAccuracy|output)
        TEST_SELECTOR="${TEST_CLASS}#testOutputAccuracy"
        ;;
    perOpSlot|perop)
        TEST_SELECTOR="${TEST_CLASS}#testPerOpSlotValidation"
        ;;
    decodeStep|decode)
        TEST_SELECTOR="${TEST_CLASS}#testDecodeStepValidation"
        ;;
    tf32Isolation|tf32)
        TEST_SELECTOR="${TEST_CLASS}#testTf32ImpactIsolation"
        ;;
    *)
        echo "Unknown test: $TEST_NAME"
        echo "Available: ALL, outputAccuracy, perOpSlot, decodeStep, tf32Isolation"
        exit 1
        ;;
esac

echo "═══════════════════════════════════════════════════════════"
echo "  SmolDocling DSP Accuracy Validation"
echo "  Test:      $TEST_NAME"
[ -n "$MAX_TOKENS" ] && echo "  Tokens:    $MAX_TOKENS"
[ -n "$CONFIGS" ]    && echo "  Configs:   $CONFIGS"
echo "  Tolerance: $TOLERANCE"
[ -n "$MATCH_RATE" ] && echo "  Match rate: ${MATCH_RATE}%"
$VERBOSE              && echo "  Verbose:   ON"
$DEBUG_MODE           && echo "  Debug:     ON"
$NO_OPTIMIZER         && echo "  Optimizer: DISABLED"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Build Maven args
EXTRA_ARGS=""

# Validation properties
[ -n "$MAX_TOKENS" ] && EXTRA_ARGS="$EXTRA_ARGS -Dvlm.validation.tokens=$MAX_TOKENS"
[ -n "$CONFIGS" ]    && EXTRA_ARGS="$EXTRA_ARGS -Dvlm.validation.configs=$CONFIGS"
[ -n "$MATCH_RATE" ] && EXTRA_ARGS="$EXTRA_ARGS -Dvlm.validation.matchRate=$MATCH_RATE"
EXTRA_ARGS="$EXTRA_ARGS -Dvlm.validation.tolerance=$TOLERANCE"
$VERBOSE && EXTRA_ARGS="$EXTRA_ARGS -Dvlm.validation.verbose=true"

# Optimizer flags
if $NO_OPTIMIZER; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.optimizer.enabled=false"
fi
if ! $FP16; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.optimizer.fp16=false"
fi

# Debug mode
if $DEBUG_MODE; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics=ALL"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.native.dumpOutputs=true"
fi

$MVN test \
  -Dtest="$TEST_SELECTOR" \
  -Dlibnd4j.triton=ON \
  -Dbackend.artifactId=nd4j-cuda-12.9 \
  $EXTRA_ARGS \
  2>&1 | tee "$LOG_FILE"

BUILD_RESULT=${PIPESTATUS[0]}

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  VALIDATION RESULTS"
echo "═══════════════════════════════════════════════════════════"

if [ $BUILD_RESULT -ne 0 ]; then
    echo "  STATUS: FAILED (exit code $BUILD_RESULT)"
    echo ""
    # Extract test failures
    if [ -f "$SUREFIRE_OUT" ]; then
        echo "  Surefire output (last 30 lines):"
        tail -30 "$SUREFIRE_OUT"
    else
        grep -E "FAILED|ERROR|Exception|assert|Token match rate" "$LOG_FILE" | tail -20
    fi
    exit 1
fi

# Parse results from surefire output
if [ -f "$SUREFIRE_OUT" ] && [ -s "$SUREFIRE_OUT" ]; then
    python3 - "$SUREFIRE_OUT" <<'PYEOF'
import sys, re

with open(sys.argv[1]) as f:
    lines = f.read().split('\n')

print("  ─────────────────────────────────")

# Extract token match rates
match_results = []
for l in lines:
    m = re.search(r'\[(\w+)\] Token match rate: (\d+)/(\d+) \(([\d.]+)%\)', l)
    if m:
        config = m.group(1)
        matched = int(m.group(2))
        total = int(m.group(3))
        pct = float(m.group(4))
        match_results.append((config, matched, total, pct))
        status = "PASS" if pct >= 90 else "WARN" if pct >= 50 else "FAIL"
        print(f"  [{status}] {config}: {matched}/{total} tokens matched ({pct:.1f}%)")

    # Also catch non-prefixed match rates (decodeStep, tf32)
    m2 = re.search(r'Token match rate: (\d+)/(\d+) \(([\d.]+)%\)', l)
    if m2 and not re.search(r'\[\w+\]', l):
        matched = int(m2.group(1))
        total = int(m2.group(2))
        pct = float(m2.group(3))
        match_results.append(("decode", matched, total, pct))
        status = "PASS" if pct >= 90 else "WARN" if pct >= 50 else "FAIL"
        print(f"  [{status}] decode: {matched}/{total} tokens matched ({pct:.1f}%)")

    m3 = re.search(r'TF32 token match rate: (\d+)/(\d+) \(([\d.]+)%\)', l)
    if m3:
        matched = int(m3.group(1))
        total = int(m3.group(2))
        pct = float(m3.group(3))
        match_results.append(("tf32", matched, total, pct))
        print(f"  [INFO] TF32 impact: {matched}/{total} tokens matched ({pct:.1f}%)")

# Interceptor capture info
for l in lines:
    m = re.search(r'Interceptor captured (\d+) variables across (\d+) steps', l)
    if m:
        print(f"  [INFO] Interceptor: {m.group(1)} variables, {m.group(2)} steps")

# Show captured variable shapes if verbose
captured = [l for l in lines if "Captured '" in l]
if captured:
    print(f"  [INFO] Captured variables: {len(captured)}")
    for c in captured[:5]:
        m = re.search(r"Captured '(.+?)': shape=(.+?) dtype=(.+)", c)
        if m:
            print(f"    {m.group(1)}: {m.group(2)} ({m.group(3)})")
    if len(captured) > 5:
        print(f"    ... and {len(captured) - 5} more")

# Reference/test text comparison
ref_texts = []
test_texts = []
for l in lines:
    m = re.search(r'Reference text: (.+)', l)
    if m: ref_texts.append(m.group(1))
    m = re.search(r'Test text:\s+(.+)', l)
    if m: test_texts.append(m.group(1))

if ref_texts:
    print("")
    print(f"  Reference text: {ref_texts[0][:80]}")
    if test_texts:
        print(f"  Test text:      {test_texts[0][:80]}")

# Overall status
print("")
if not match_results:
    print("  OVERALL: No token comparison results found")
elif all(pct >= 90 for _, _, _, pct in match_results):
    print("  OVERALL: PASS — all configs match >=90%")
else:
    failing = [(c, p) for c, _, _, p in match_results if p < 90]
    print(f"  OVERALL: WARN — {len(failing)} config(s) below 90% match rate")
    for c, p in failing:
        print(f"    {c}: {p:.1f}%")

PYEOF
else
    echo "  No surefire report found. Grep from log:"
    grep -E "Token match rate|Interceptor captured|FAIL|PASS" "$LOG_FILE" | tail -15
fi

echo "═══════════════════════════════════════════════════════════"

if $DEBUG_MODE; then
    echo ""
    echo "  Debug files:"
    echo "    Validation log:  $LOG_FILE"
    echo "    Surefire report: $SUREFIRE_OUT"
fi
