#!/bin/bash
# DSP Configuration Matrix — runs TestDspConfigurationMatrix, which sweeps all
# meaningful combinations of GraphExecutionMode × Triton flags × frozen-constants
# × batched-GEMM against a golden SLOT_BY_SLOT baseline and asserts the DSP phase
# contract holds for each configuration.
#
# Each matrix entry catches a different class of regression:
#   - Backend selection logic (AUTO defaults → {TRITON, CUDA_GRAPHS} resolution)
#   - Triton compile pipeline (compileAll, section fusion, include types)
#   - Frozen-constants fast path (shapes frozen → CUDA graph capture)
#   - Batched GEMM integration with the rest of the replay path
#
# The matrix uses a tiny 2-layer MLP graph so it runs in seconds, but every
# assertion uses the same DspPlanAssertions helper that the SmolDocling test
# uses — failures name the phase that broke (e.g., POINTERS_STABLE, REPLAYING).
#
# Usage: ./run-dsp-matrix.sh [OPTIONS]
#
# Selection:
#   --config NAME       Run only one matrix entry (e.g., CUDA_GRAPHS_frozen).
#                       Without this flag, the full matrix runs.
#   --list              Print the list of configs and exit.
#
# Diagnostics:
#   --diag-replay       Enable GRAPH_REPLAY diagnostics
#   --diag-segment      Enable SEGMENT + BACKEND diagnostics (backend selection, boundaries)
#   --diag-phase        Enable phase-transition diagnostics (plan/segment phase events)
#   --diag-all          Enable ALL categories at FULL level with JSON report
#   --diag-json FILE    Write structured JSON diagnostic report to FILE
#
# Backend:
#   --backend NAME      Run on cuda (default), vulkan, or cpu.
#   --cpu               Compatibility alias for --backend cpu.
#   --no-triton         Pass -Dnd4j.triton.skipKernels=true so Triton-dependent
#                       configs fall through to slot-by-slot. Useful on CPU-only
#                       machines or when triaging non-Triton regressions.
#
# Examples:
#   ./run-dsp-matrix.sh                             # Full matrix on CUDA
#   ./run-dsp-matrix.sh --config CUDA_GRAPHS_frozen # Single config
#   ./run-dsp-matrix.sh --diag-all                  # Full diagnostics
#   ./run-dsp-matrix.sh --cpu                       # Run on CPU backend
#   ./run-dsp-matrix.sh --list                      # Show available configs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MVN="${MVN:-mvn}"
TEST_CLASS="TestDspConfigurationMatrix"
TEST_METHOD="testConfiguration"
LOG_FILE="$SCRIPT_DIR/dsp-matrix-output.log"
SUREFIRE_OUT="$SCRIPT_DIR/target/surefire-reports/org.eclipse.deeplearning4j.nd4j.autodiff.samediff.${TEST_CLASS}-output.txt"
SUREFIRE_XML="$SCRIPT_DIR/target/surefire-reports/TEST-org.eclipse.deeplearning4j.nd4j.autodiff.samediff.${TEST_CLASS}.xml"

# Keep this in sync with dspConfigurations() in TestDspConfigurationMatrix.java
CONFIGS=(
    "SLOT_BY_SLOT_baseline"
    "SLOT_BY_SLOT_batchedGemm"
    "AUTO_defaults"
    "AUTO_frozen"
    "TRITON_sectionFusion"
    "TRITON_compileAll"
    "TRITON_frozen_batchedGemm"
    "CUDA_GRAPHS_frozen"
)

# Defaults
SELECTED_CONFIG=""
BACKEND_NAME="cuda"
BACKEND_ARTIFACT="nd4j-cuda-12.9"
BACKEND_PROFILE=""
TRITON_OPTION="ON"
NO_TRITON=false
DIAG_REPLAY=false
DIAG_SEGMENT=false
DIAG_PHASE=false
DIAG_ALL=false
DIAG_JSON=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            SELECTED_CONFIG="$2"
            shift 2
            ;;
        --list)
            echo "Available DSP matrix configurations:"
            for c in "${CONFIGS[@]}"; do
                echo "  - $c"
            done
            exit 0
            ;;
        --backend)
            case "$2" in
                cuda)
                    BACKEND_NAME="cuda"
                    BACKEND_ARTIFACT="nd4j-cuda-12.9"
                    BACKEND_PROFILE=""
                    TRITON_OPTION="ON"
                    ;;
                vulkan)
                    BACKEND_NAME="vulkan"
                    BACKEND_ARTIFACT="nd4j-cuda-12.9"
                    BACKEND_PROFILE="-Ptest-vulkan"
                    TRITON_OPTION="OFF"
                    NO_TRITON=true
                    ;;
                cpu)
                    BACKEND_NAME="cpu"
                    BACKEND_ARTIFACT="nd4j-native"
                    BACKEND_PROFILE=""
                    TRITON_OPTION="OFF"
                    ;;
                *)
                    echo "Unknown backend: $2 (expected cuda, vulkan, or cpu)"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --cpu)
            BACKEND_NAME="cpu"
            BACKEND_ARTIFACT="nd4j-native"
            BACKEND_PROFILE=""
            TRITON_OPTION="OFF"
            shift
            ;;
        --no-triton)
            NO_TRITON=true
            shift
            ;;
        --diag-replay)
            DIAG_REPLAY=true
            shift
            ;;
        --diag-segment)
            DIAG_SEGMENT=true
            shift
            ;;
        --diag-phase)
            DIAG_PHASE=true
            shift
            ;;
        --diag-all)
            DIAG_ALL=true
            shift
            ;;
        --diag-json)
            DIAG_JSON="$2"
            shift 2
            ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -60
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run-dsp-matrix.sh [--config NAME] [--list] [--backend NAME|--cpu] [--no-triton]"
            echo "       [--diag-replay] [--diag-segment] [--diag-phase] [--diag-all] [--diag-json FILE]"
            exit 1
            ;;
    esac
done

# Validate selected config against the known list
if [ -n "$SELECTED_CONFIG" ]; then
    FOUND=false
    for c in "${CONFIGS[@]}"; do
        if [ "$c" = "$SELECTED_CONFIG" ]; then
            FOUND=true
            break
        fi
    done
    if ! $FOUND; then
        echo "Unknown config: $SELECTED_CONFIG"
        echo "Available configs:"
        for c in "${CONFIGS[@]}"; do
            echo "  - $c"
        done
        exit 1
    fi
fi

echo "═══════════════════════════════════════════════════════════"
echo "  DSP Configuration Matrix"
echo "  Test:   ${TEST_CLASS}#${TEST_METHOD}"
echo "  Backend: $BACKEND_NAME ($BACKEND_ARTIFACT)"
if [ -n "$SELECTED_CONFIG" ]; then
    echo "  Config: $SELECTED_CONFIG (single)"
else
    echo "  Configs: ${#CONFIGS[@]} matrix entries"
fi
$NO_TRITON   && echo "  Triton:  DISABLED (skipKernels=true)"
$DIAG_REPLAY && echo "  Diag:    GRAPH_REPLAY"
$DIAG_SEGMENT && echo "  Diag:    SEGMENT + BACKEND"
$DIAG_PHASE  && echo "  Diag:    phase transitions"
$DIAG_ALL    && echo "  Diag:    ALL categories at FULL level"
[ -n "$DIAG_JSON" ] && echo "  Diag JSON: $DIAG_JSON"
echo "  Log:     $LOG_FILE"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Build extra Maven/JVM args
EXTRA_ARGS=""

if $NO_TRITON; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.triton.skipKernels=true"
fi

# Targeted diagnostic modes — can be combined
if $DIAG_ALL; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics=ALL"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics.level=full"
    if [ -z "$DIAG_JSON" ]; then
        DIAG_JSON="$SCRIPT_DIR/dsp-matrix-diagnostics.json"
    fi
else
    DIAG_CATS=""
    if $DIAG_REPLAY; then
        DIAG_CATS="${DIAG_CATS:+$DIAG_CATS,}GRAPH_REPLAY,EXECUTE"
    fi
    if $DIAG_SEGMENT; then
        DIAG_CATS="${DIAG_CATS:+$DIAG_CATS,}SEGMENT,BACKEND,COMPILE"
    fi
    if $DIAG_PHASE; then
        DIAG_CATS="${DIAG_CATS:+$DIAG_CATS,}EXECUTE,SEGMENT,COMPILE,FALLBACK"
    fi
    if [ -n "$DIAG_CATS" ]; then
        EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics=$DIAG_CATS"
        EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics.level=full"
    fi
fi
if [ -n "$DIAG_JSON" ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics.file=$DIAG_JSON"
fi

# Build the -Dtest selector. JUnit 5 parameterized tests accept the
# `Class#method[displayName]` form for individual runs.
if [ -n "$SELECTED_CONFIG" ]; then
    TEST_SELECTOR="${TEST_CLASS}#${TEST_METHOD}[${SELECTED_CONFIG}]"
else
    TEST_SELECTOR="${TEST_CLASS}#${TEST_METHOD}"
fi

echo "Running: $TEST_SELECTOR"
echo ""

set +e
$MVN test \
  $BACKEND_PROFILE \
  -Dtest="$TEST_SELECTOR" \
  -Dlibnd4j.triton="$TRITON_OPTION" \
  -Dbackend.artifactId="$BACKEND_ARTIFACT" \
  $EXTRA_ARGS \
  2>&1 | tee "$LOG_FILE"
BUILD_RESULT=${PIPESTATUS[0]}
set -e

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  RESULTS"
echo "═══════════════════════════════════════════════════════════"

# Pick up a surefire report if one exists — XML preferred for per-testcase status.
REPORT=""
if [ -f "$SUREFIRE_XML" ]; then
    REPORT="$SUREFIRE_XML"
elif [ -f "$SUREFIRE_OUT" ] && [ -s "$SUREFIRE_OUT" ]; then
    REPORT="$SUREFIRE_OUT"
fi

if [ -n "$REPORT" ]; then
    python3 - "$REPORT" "$LOG_FILE" <<'PYEOF'
import sys, re, os, xml.etree.ElementTree as ET

report_path = sys.argv[1]
log_path = sys.argv[2]

pass_rows = []
fail_rows = []
skip_rows = []

if report_path.endswith('.xml'):
    tree = ET.parse(report_path)
    for tc in tree.findall('.//testcase'):
        name = tc.attrib.get('name', '?')
        time = tc.attrib.get('time', '?')
        failure = tc.find('failure')
        error = tc.find('error')
        skipped = tc.find('skipped')
        if failure is not None or error is not None:
            node = failure if failure is not None else error
            msg = (node.attrib.get('message') or '').strip().split('\n')[0]
            fail_rows.append((name, time, msg[:140]))
        elif skipped is not None:
            msg = (skipped.attrib.get('message') or '').strip().split('\n')[0]
            skip_rows.append((name, msg[:140]))
        else:
            pass_rows.append((name, time))
else:
    with open(report_path) as f:
        txt = f.read()
    for m in re.finditer(r'testConfiguration\[([^\]]+)\]', txt):
        pass_rows.append((m.group(1), '?'))

print("")
print(f"  Passed:  {len(pass_rows)}")
print(f"  Failed:  {len(fail_rows)}")
print(f"  Skipped: {len(skip_rows)}")
print("")

if pass_rows:
    print("  ── PASS ─────────────────────────────────────────────")
    for name, time in pass_rows:
        config = re.sub(r'^testConfiguration\[', '', name).rstrip(']')
        time_str = f"{time}s" if time != '?' else '--'
        print(f"    [PASS] {config:<40} {time_str}")
    print("")

if fail_rows:
    print("  ── FAIL ─────────────────────────────────────────────")
    for name, time, msg in fail_rows:
        config = re.sub(r'^testConfiguration\[', '', name).rstrip(']')
        print(f"    [FAIL] {config}")
        print(f"           {msg}")
    print("")

if skip_rows:
    print("  ── SKIP ─────────────────────────────────────────────")
    for name, msg in skip_rows:
        config = re.sub(r'^testConfiguration\[', '', name).rstrip(']')
        print(f"    [SKIP] {config:<40} {msg}")
    print("")

# Pull key plan-health log lines from the tee file — these are emitted by
# assertDspHealth() in TestDspConfigurationMatrix after each successful run.
with open(log_path) as f:
    log_lines = f.readlines()

health_lines = [l.rstrip() for l in log_lines if 'DSP health OK' in l]
if health_lines:
    print("  ── DSP Plan Health ─────────────────────────────────")
    for l in health_lines:
        stripped = re.sub(r'^.*?\[', '[', l).strip()
        print(f"    {stripped}")
    print("")

# Phase-contract violations (from DspPlanAssertions failure output)
viol = [l.rstrip() for l in log_lines if 'phase contract violation' in l.lower()
        or 'captureFailed=true' in l or 'stuck in WARMUP' in l]
if viol:
    print("  ── Phase-Contract Violations ───────────────────────")
    for l in viol[:20]:
        stripped = re.sub(r'^\s+', '    ', l)
        print(stripped)
    print("")

sys.exit(0 if not fail_rows else 1)
PYEOF
    PARSE_RESULT=$?
else
    echo "  No surefire report found. Grep from log:"
    grep -E "testConfiguration|PASS|FAIL|DspPlanAssertions" "$LOG_FILE" | tail -20
    PARSE_RESULT=0
fi

if [ $BUILD_RESULT -ne 0 ] || [ $PARSE_RESULT -ne 0 ]; then
    echo "  STATUS: FAILED"
    echo ""
    echo "  To debug a specific config:"
    echo "    ./run-dsp-matrix.sh --config <NAME> --diag-all"
    echo ""
    echo "  Log file: $LOG_FILE"
    if [ -n "$DIAG_JSON" ] && [ -f "$DIAG_JSON" ]; then
        echo "  JSON diagnostics: $DIAG_JSON"
    fi
    exit 1
fi

echo "  STATUS: PASSED"
echo ""
echo "  Log file: $LOG_FILE"
if [ -n "$DIAG_JSON" ] && [ -f "$DIAG_JSON" ]; then
    echo "  JSON diagnostics: $DIAG_JSON"
fi
echo "═══════════════════════════════════════════════════════════"
