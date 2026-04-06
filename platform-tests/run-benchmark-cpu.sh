#!/bin/bash
# SmolDocling VLM Decode Benchmark — CPU backend (nd4j-native)
# Runs the same SmolDocling benchmark as run-benchmark.sh but on CPU via OpenVINO/OneDNN Graph backends.
#
# Usage: ./run-benchmark-cpu.sh [OPTIONS]
#
# Options (same as run-benchmark.sh except CUDA-specific ones):
#   --debug           Enable DSP diagnostics, verbose tracing
#   --diag-replay     Enable GRAPH_REPLAY diagnostics (capture/instantiate/launch/address validation)
#   --diag-stream     Enable STREAM_SYNC diagnostics (stream ordering, event waits, sync points)
#   --diag-device     Enable MULTI_DEVICE diagnostics (device selection, P2P, migrations)
#   --diag-all        Enable ALL diagnostic categories at FULL level with JSON report
#   --diag-json FILE  Write structured JSON diagnostic report to FILE
#   --tokens N        Override max decode tokens (default: 250)
#   --config NAME     Override benchmark config name (default: OPTIMAL)
#   --fp16            Enable FP16 weight pre-casting via GraphOptimizer (DEFAULT: ON)
#   --no-fp16         Disable FP16 pre-casting (run with FP32 weights)
#   --no-optimizer    Disable the GraphOptimizer entirely
#   --optimizer-log   Log which constants the optimizer transforms
#   --clear-cache     Delete cached .sdz model files and re-import from ONNX
#   --clear-decoder   Delete only the decoder .sdz cache (DEFAULT: ON)
#   --no-clear-decoder Keep decoder .sdz cache (skip re-import)
#   --op-timing       Enable decode-only native op timing and export CSV per config
#   --op-timing-detailed Enable per-phase op timing breakdown data
#   --op-breakdown OPS Print per-op timing breakdowns for comma-separated op names
#   --op-histogram OPS Print per-op timing histograms for comma-separated op names
#   --no-freeze       Disable shape freezing
#
# Examples:
#   ./run-benchmark-cpu.sh                           # Default: 250 tokens, FP16 ON
#   ./run-benchmark-cpu.sh --tokens 50               # Quick 50-token run
#   ./run-benchmark-cpu.sh --debug                   # Full DSP diagnostics
#   ./run-benchmark-cpu.sh --no-fp16                 # FP32 weights baseline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MVN="/home/agibsonccc/dev-apps/mvn/bin/mvn"
TEST_CLASS="TestSmolDoclingOptimizedPipeline"
TEST_METHOD="testOptimizedDoclingPipeline"
LOG_FILE="$SCRIPT_DIR/bench-cpu-output.log"
SUREFIRE_OUT="$SCRIPT_DIR/target/surefire-reports/org.eclipse.deeplearning4j.vlm.${TEST_CLASS}-output.txt"
SUREFIRE_XML="$SCRIPT_DIR/target/surefire-reports/TEST-org.eclipse.deeplearning4j.vlm.${TEST_CLASS}.xml"
MODEL_CACHE="$HOME/.cache/dl4j-vlm-models"

# Defaults
DEBUG_MODE=false
OP_TIMING=false
OP_TIMING_DETAILED=false
OP_BREAKDOWN_OPS=""
OP_HISTOGRAM_OPS=""
MAX_TOKENS=250
CONFIG="OPTIMAL"
FP16=true
NO_OPTIMIZER=false
OPTIMIZER_LOG=false
CLEAR_CACHE=false
CLEAR_DECODER=true
NO_FREEZE=false
NO_ATTN_OVERRIDE=false
NO_DIRECT=false
DIAG_REPLAY=false
DIAG_STREAM=false
DIAG_DEVICE=false
DIAG_ALL=false
DIAG_JSON=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --op-timing)
            OP_TIMING=true
            shift
            ;;
        --op-timing-detailed)
            OP_TIMING=true
            OP_TIMING_DETAILED=true
            shift
            ;;
        --op-breakdown)
            OP_TIMING=true
            OP_BREAKDOWN_OPS="$2"
            shift 2
            ;;
        --op-histogram)
            OP_TIMING=true
            OP_HISTOGRAM_OPS="$2"
            shift 2
            ;;
        --tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
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
        --optimizer-log)
            OPTIMIZER_LOG=true
            shift
            ;;
        --clear-cache)
            CLEAR_CACHE=true
            shift
            ;;
        --clear-decoder)
            CLEAR_DECODER=true
            shift
            ;;
        --no-clear-decoder)
            CLEAR_DECODER=false
            shift
            ;;
        --no-freeze)
            NO_FREEZE=true
            shift
            ;;
        --no-attn-override)
            NO_ATTN_OVERRIDE=true
            shift
            ;;
        --no-direct)
            NO_DIRECT=true
            shift
            ;;
        --diag-replay)
            DIAG_REPLAY=true
            shift
            ;;
        --diag-stream)
            DIAG_STREAM=true
            shift
            ;;
        --diag-device)
            DIAG_DEVICE=true
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
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run-benchmark-cpu.sh [--debug] [--op-timing] [--tokens N] [--config NAME]"
            echo "       [--fp16] [--no-fp16] [--no-optimizer] [--optimizer-log]"
            echo "       [--clear-cache] [--clear-decoder] [--no-clear-decoder]"
            echo "       [--no-freeze] [--no-attn-override] [--no-direct]"
            echo "       [--diag-replay] [--diag-stream] [--diag-device] [--diag-all] [--diag-json FILE]"
            exit 1
            ;;
    esac
done

# Handle cache clearing
if $CLEAR_CACHE; then
    echo "Clearing all cached SDZ models in $MODEL_CACHE..."
    rm -f "$MODEL_CACHE"/*.sdz 2>/dev/null || true
elif $CLEAR_DECODER; then
    echo "Clearing decoder cache (base + optimized)..."
    rm -f "$MODEL_CACHE/smoldocling-decoder.sdz" 2>/dev/null || true
    rm -f "$MODEL_CACHE/smoldocling-decoder.opt.sdz" 2>/dev/null || true
fi

echo "═══════════════════════════════════════════════════════════"
echo "  SmolDocling VLM Decode Benchmark (CPU)"
echo "  Backend: nd4j-native (OpenVINO + OneDNN Graph)"
echo "  PDF:     pathfinder-mythic.pdf page 10"
echo "  Tokens:  $MAX_TOKENS"
echo "  Config:  $CONFIG"
$FP16         && echo "  FP16:    ON  (weight pre-cast via optimizer)"
$NO_OPTIMIZER && echo "  Optimizer: DISABLED"
$OPTIMIZER_LOG && echo "  Optimizer: logging applied transforms"
$DEBUG_MODE   && echo "  Mode:    DEBUG (DSP diagnostics)"
$NO_FREEZE    && echo "  Freeze:  DISABLED"
$DIAG_REPLAY  && echo "  Diag:    GRAPH_REPLAY (capture/instantiate/launch phases)"
$DIAG_STREAM  && echo "  Diag:    STREAM_SYNC (stream ordering, event waits)"
$DIAG_DEVICE  && echo "  Diag:    MULTI_DEVICE (device selection, P2P, migrations)"
$DIAG_ALL     && echo "  Diag:    ALL categories at FULL level"
[ -n "$DIAG_JSON" ] && echo "  Diag JSON: $DIAG_JSON"
$OP_TIMING    && echo "  OpTiming: ON"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Build extra Maven/JVM args
EXTRA_ARGS=""

# Optimizer flags
if $NO_OPTIMIZER; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.optimizer.enabled=false"
fi

if ! $FP16; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.optimizer.fp16=false"
fi

if $OPTIMIZER_LOG; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.optimizer.logApplied=true"
fi

if $NO_FREEZE; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.nofreeze=true"
fi

if $NO_ATTN_OVERRIDE; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.noAttnOverride=true"
fi

if $NO_DIRECT; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.noDirect=true"
fi

# Debug mode flags
if $DEBUG_MODE; then
    DSP_LOG="$SCRIPT_DIR/dsp-diagnostics-cpu.log"
    echo "  DSP diagnostics: ALL"
    echo ""
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics=ALL"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics.level=full"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.native.dumpOutputs=true"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.frozen.summary=true"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.frozen.debug=true"
fi

# Targeted diagnostic modes — can be combined
if $DIAG_ALL; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics=ALL"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics.level=full"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.frozen.summary=true"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.frozen.debug=true"
    if [ -z "$DIAG_JSON" ]; then
        DIAG_JSON="$SCRIPT_DIR/dsp-diagnostics-cpu.json"
    fi
else
    DIAG_CATS=""
    if $DIAG_REPLAY; then
        DIAG_CATS="${DIAG_CATS:+$DIAG_CATS,}GRAPH_REPLAY,SEGMENT,EXECUTE"
    fi
    if $DIAG_STREAM; then
        DIAG_CATS="${DIAG_CATS:+$DIAG_CATS,}STREAM_SYNC,EXECUTE,TIMING"
    fi
    if $DIAG_DEVICE; then
        DIAG_CATS="${DIAG_CATS:+$DIAG_CATS,}MULTI_DEVICE,TRANSFER,BACKEND,MEMORY"
    fi
    if [ -n "$DIAG_CATS" ]; then
        EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics=$DIAG_CATS"
        EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics.level=full"
    fi
fi
if [ -n "$DIAG_JSON" ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics.file=$DIAG_JSON"
fi

if $OP_TIMING; then
    OP_TIMING_DIR="$SCRIPT_DIR/op-timing-cpu"
    mkdir -p "$OP_TIMING_DIR"
    EXTRA_ARGS="$EXTRA_ARGS -Dvlm.benchmark.opTiming=true"
    EXTRA_ARGS="$EXTRA_ARGS -Dvlm.benchmark.opTimingTopN=16"
    EXTRA_ARGS="$EXTRA_ARGS -Dvlm.benchmark.opTimingCsvDir=$OP_TIMING_DIR"
    if $OP_TIMING_DETAILED; then
        EXTRA_ARGS="$EXTRA_ARGS -Dvlm.benchmark.opTimingDetailed=true"
    fi
    if [ -n "$OP_BREAKDOWN_OPS" ]; then
        EXTRA_ARGS="$EXTRA_ARGS -Dvlm.benchmark.opTimingBreakdownOps=$OP_BREAKDOWN_OPS"
    fi
    if [ -n "$OP_HISTOGRAM_OPS" ]; then
        EXTRA_ARGS="$EXTRA_ARGS -Dvlm.benchmark.opTimingHistogramOps=$OP_HISTOGRAM_OPS"
    fi
fi

# Use all available cores for OpenBLAS/MKL BLAS operations.
# Default pom.xml sets OMP_NUM_THREADS=1 for test reproducibility;
# override here for benchmark performance.
OMP_THREADS=$(nproc)

$MVN test \
  -Dtest="${TEST_CLASS}#${TEST_METHOD}" \
  -Dvlm.test.maxTokens="$MAX_TOKENS" \
  -Dvlm.test.pdf.path=pathfinder-mythic.pdf \
  -Dvlm.test.pdf.page=10 \
  -Dvlm.test.configs="$CONFIG" \
  -Dbackend.artifactId=nd4j-native \
  -Domp.num.threads="$OMP_THREADS" \
  $EXTRA_ARGS \
  2>&1 | tee "$LOG_FILE"

BUILD_RESULT=${PIPESTATUS[0]}

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  RESULTS (CPU)"
echo "═══════════════════════════════════════════════════════════"

if [ $BUILD_RESULT -ne 0 ]; then
    echo "  STATUS: FAILED (exit code $BUILD_RESULT)"
    echo ""
    grep -E "FAILED|ERROR|Exception|assert" "$LOG_FILE" | tail -20
    exit 1
fi

if grep -q "Filtered to 0 configs via vlm.test.configs" "$LOG_FILE"; then
    echo "  STATUS: FAILED (requested config resolved to 0 benchmark configs)"
    grep "Filtered to 0 configs via vlm.test.configs" "$LOG_FILE"
    exit 1
fi

# Extract metrics from surefire output or XML
REPORT=""
if [ -f "$SUREFIRE_OUT" ] && [ -s "$SUREFIRE_OUT" ]; then
    REPORT="$SUREFIRE_OUT"
elif [ -f "$SUREFIRE_XML" ]; then
    REPORT="$SUREFIRE_XML"
fi

if [ -n "$REPORT" ]; then
    python3 - "$REPORT" <<'PYEOF'
import sys, re, xml.etree.ElementTree as ET

report_path = sys.argv[1]

# Get lines from either plain text or XML
if report_path.endswith('.xml'):
    tree = ET.parse(report_path)
    lines = []
    for tc in tree.findall('.//testcase'):
        for tag in ['system-out', 'system-err']:
            el = tc.find(tag)
            if el is not None and el.text:
                lines.extend(el.text.split('\n'))
else:
    with open(report_path) as f:
        lines = f.read().split('\n')

# Decode steps
steps = []
for l in lines:
    if 'Step' in l and 'id=' in l:
        steps.append(l.strip())
    elif 'throughput' in l.lower() and 'tok/s' in l.lower():
        steps.append(l.strip())

# Show first 10 + last 5 steps
print("  Decode Steps:")
show = steps[:10] + (['  ...'] if len(steps) > 15 else []) + steps[-5:] if len(steps) > 15 else steps
for s in show:
    s = re.sub(r'^.*?Step', '    Step', s)
    s = re.sub(r'^.*?[Dd]ecode throughput', '    Decode throughput', s)
    print(s)

# Key metrics
print("")
print("  ─────────────────────────────────")
for l in lines:
    m = re.search(r'[Dd]ecode throughput:\s*([\d.]+)\s*tok/s', l)
    if m:
        tps = float(m.group(1))
        print(f"  DECODE THROUGHPUT: {m.group(1)} tok/s")

# Correctness check
all_text = '\n'.join(lines)
has_doctag = 'doctag' in all_text
has_english = any(w in all_text.lower() for w in ['mythic', 'hero', 'path', 'ability', 'tier'])
garbage = all_text.count('UserT') > 3

if garbage:
    print("  CORRECTNESS:      FAIL — repeating garbage")
elif has_doctag and has_english:
    print("  CORRECTNESS:      PASS — doctag + coherent English")
elif has_doctag:
    print("  CORRECTNESS:      PARTIAL — doctag present, check text quality")
else:
    print("  CORRECTNESS:      UNKNOWN — no doctag found")

PYEOF
else
    echo "  No surefire report found. Grep from log:"
    grep -E "tok/s|throughput|Performance" "$LOG_FILE" | tail -10
fi

echo "═══════════════════════════════════════════════════════════"

if $DEBUG_MODE; then
    echo ""
    echo "  Debug files:"
    echo "    Benchmark log:   $LOG_FILE"
    echo "    Surefire report: $SUREFIRE_OUT"
fi

# Show diagnostic summary if any diag mode was enabled
if $DIAG_REPLAY || $DIAG_STREAM || $DIAG_DEVICE || $DIAG_ALL; then
    echo ""
    echo "  ─── DSP Diagnostics Summary (CPU) ───────────────────"

    if $DIAG_REPLAY || $DIAG_ALL; then
        echo "  Graph Replay:"
        CAPTURE_FAIL=$(grep -c "CAPTURE_FAILED\|capture failed\|captureFailed=true" "$LOG_FILE" 2>/dev/null || true)
        REPLAY_OK=$(grep -c "REPLAYING\|graph replay active" "$LOG_FILE" 2>/dev/null || true)
        echo "    Capture failures: $CAPTURE_FAIL"
        echo "    Replay active:    $REPLAY_OK events"
        grep -E "GRAPH_REPLAY.*phase|capture.*fail|replay.*error" "$LOG_FILE" 2>/dev/null | tail -10
    fi

    if $DIAG_STREAM || $DIAG_ALL; then
        echo "  Stream Sync:"
        SYNC_EVENTS=$(grep -c "STREAM_SYNC\|stream.*sync" "$LOG_FILE" 2>/dev/null || true)
        echo "    Sync events: $SYNC_EVENTS"
        grep -E "STREAM_SYNC|stream.*stall|sync.*miss" "$LOG_FILE" 2>/dev/null | tail -10
    fi

    if $DIAG_DEVICE || $DIAG_ALL; then
        echo "  Multi-Device:"
        TRANSFERS=$(grep -c "TRANSFER\|D2D\|H2D\|D2H" "$LOG_FILE" 2>/dev/null || true)
        echo "    Transfer events: $TRANSFERS"
        grep -E "MULTI_DEVICE|memory.*pressure|reroute" "$LOG_FILE" 2>/dev/null | tail -10
    fi

    if [ -n "$DIAG_JSON" ] && [ -f "$DIAG_JSON" ]; then
        echo ""
        echo "  JSON diagnostic report: $DIAG_JSON"
    fi
    echo "  ─────────────────────────────────────────────────────"
fi
