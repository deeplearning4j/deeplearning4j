#!/bin/bash
# SmolDocling VLM Decode Benchmark — pathfinder-mythic.pdf page 10, 250 tokens
# Target: 100+ tok/s with correct SmolDocling layout tags + coherent English
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ CURRENT OPTIMAL DEFAULTS (update this when perf improves)          │
# │                                                                     │
# │  FP16 weight pre-cast:  ON  (default, disable: --no-fp16)         │
# │  GraphOptimizer:        ON  (default, disable: --no-optimizer)     │
# │  Clear decoder cache:   ON  (default, disable: --no-clear-decoder) │
# │  Config:                OPTIMAL (warps4/stages1/tf32/batchedGemm)   │
# │  Max tokens:            250                                         │
# │                                                                     │
# │  Best measured:         ~87-92 tok/s late steady-state decode       │
# │  Speculation (K=5):    ~37 tok/s (n-gram 0% acceptance, overhead)  │
# │  Expected with FP16:    mythic passage + section header present     │
# └─────────────────────────────────────────────────────────────────────┘
#
# Usage: ./run-benchmark.sh [OPTIONS]
#
# Execution options:
#   --debug           Enable DSP diagnostics, CUDA driver log, verbose tracing
#   --diag-replay     Enable GRAPH_REPLAY diagnostics (capture/instantiate/launch/address validation)
#   --diag-stream     Enable STREAM_SYNC diagnostics (stream ordering, event waits, sync points)
#   --diag-device     Enable MULTI_DEVICE diagnostics (device selection, P2P, migrations)
#   --diag-all        Enable ALL diagnostic categories at FULL level with JSON report
#   --diag-json FILE  Write structured JSON diagnostic report to FILE
#   --op-timing       Enable decode-only native op timing and export CSV per config
#   --op-timing-detailed
#                     Enable per-phase op timing breakdown data
#   --op-breakdown OPS
#                     Print per-op timing breakdowns for comma-separated op names
#                     (requires --op-timing)
#   --op-histogram OPS
#                     Print per-op timing histograms for comma-separated op names
#                     (requires --op-timing)
#   --tokens N        Override max decode tokens (default: 250)
#   --config NAME     Override benchmark config name (default: OPTIMAL)
#
# Optimizer options:
#   --fp16            Enable FP16 weight pre-casting via GraphOptimizer (DEFAULT: ON)
#                     (halves weight memory bandwidth; MmulHelper mixed-type path
#                     casts only the FP32 activation — 1 cast vs 2 for dspFp16Compute)
#   --no-fp16         Disable FP16 pre-casting (run with FP32 weights)
#   --no-optimizer    Disable the GraphOptimizer entirely
#   --optimizer-log   Log which constants the optimizer transforms
#
# Precision options:
#   --triton-tf32       Enable TF32 precision for Triton-compiled DotOps (10-bit mantissa)
#   --no-triton-tf32    Disable TF32 for Triton (DEFAULT: OFF — use IEEE for accuracy)
#
# Cache options:
#   --clear-cache       Delete cached .sdz model files and re-import from ONNX
#   --clear-decoder     Delete only the decoder .sdz cache (DEFAULT: ON)
#   --no-clear-decoder  Keep decoder .sdz cache (skip re-import)
#
# Examples:
#   ./run-benchmark.sh                           # Default: FP16 pre-cast, optimizer ON, clear decoder cache, 250 tokens
#   ./run-benchmark.sh --tokens 100              # Quick 100-token run
#   ./run-benchmark.sh --no-clear-decoder         # Keep cached decoder (skip re-import)
#   ./run-benchmark.sh --debug                    # Full DSP diagnostics + CUDA driver log
#   ./run-benchmark.sh --op-timing                # Decode-only op timing CSV + hotspot table
#   ./run-benchmark.sh --no-fp16                  # FP32 weights (baseline comparison)
#   ./run-benchmark.sh --no-optimizer             # No optimization at all
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MVN="/home/agibsonccc/dev-apps/mvn/bin/mvn"
TEST_CLASS="TestSmolDoclingOptimizedPipeline"
TEST_METHOD="testOptimizedDoclingPipeline"
LOG_FILE="$SCRIPT_DIR/bench-output.log"
SUREFIRE_OUT="$SCRIPT_DIR/target/surefire-reports/org.eclipse.deeplearning4j.vlm.${TEST_CLASS}-output.txt"
SUREFIRE_XML="$SCRIPT_DIR/target/surefire-reports/TEST-org.eclipse.deeplearning4j.vlm.${TEST_CLASS}.xml"
MODEL_CACHE="$HOME/.cache/dl4j-vlm-models"

# Defaults
DEBUG_MODE=false
NSYS_MODE=false
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
SPECULATIVE_K=0
DRAFT_MODEL=false
NO_NATIVE_DECODE=false
NO_CUBLAS_WORKSPACE=false
NO_FREEZE=false
TRITON_TF32=false
NO_ATTN_OVERRIDE=false
NO_DIRECT=false
NO_TRITON=false
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
        --nsys)
            NSYS_MODE=true
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
        --speculative)
            SPECULATIVE_K="$2"
            shift 2
            ;;
        --draft)
            DRAFT_MODEL=true
            if [ "$SPECULATIVE_K" -eq 0 ]; then
                SPECULATIVE_K=5
            fi
            shift
            ;;
        --no-native-decode)
            NO_NATIVE_DECODE=true
            shift
            ;;
        --no-cublas-workspace)
            NO_CUBLAS_WORKSPACE=true
            shift
            ;;
        --no-freeze)
            NO_FREEZE=true
            shift
            ;;
        --triton-tf32)
            TRITON_TF32=true
            shift
            ;;
        --no-triton-tf32)
            TRITON_TF32=false
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
        --no-triton)
            NO_TRITON=true
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
            echo "Usage: ./run-benchmark.sh [--debug] [--nsys] [--op-timing] [--op-timing-detailed]"
            echo "       [--op-breakdown OPS] [--op-histogram OPS] [--tokens N] [--config NAME]"
            echo "       [--fp16] [--no-fp16] [--no-optimizer] [--optimizer-log]"
            echo "       [--clear-cache] [--clear-decoder] [--no-clear-decoder]"
            echo "       [--draft] [--speculative K] [--no-native-decode] [--no-cublas-workspace] [--no-freeze]"
            echo "       [--triton-tf32] [--no-triton-tf32]"
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
echo "  SmolDocling VLM Decode Benchmark"
echo "  PDF:    pathfinder-mythic.pdf page 10"
echo "  Tokens: $MAX_TOKENS"
echo "  Config: $CONFIG"
echo "  Target: 100+ tok/s"
$FP16         && echo "  FP16:   ON  (weight pre-cast via optimizer)"
$NO_OPTIMIZER && echo "  Optimizer: DISABLED"
$OPTIMIZER_LOG && echo "  Optimizer: logging applied transforms"
$DEBUG_MODE   && echo "  Mode:   DEBUG (DSP diagnostics + CUDA driver log)"
$NSYS_MODE    && echo "  Mode:   NSYS (NVIDIA Nsight Systems profiler)"
$NO_NATIVE_DECODE && echo "  NativeDecodeInputs: DISABLED (Java feedDict path)"
$NO_CUBLAS_WORKSPACE && echo "  cuBLAS workspace:   DISABLED (no explicit workspace during capture)"
$NO_FREEZE           && echo "  Freeze:             DISABLED (no shape freezing, no CUDA graph)"
$NO_ATTN_OVERRIDE    && echo "  AttnOverride:       DISABLED (use model's attn_mask_reformat subgraph)"
$NO_DIRECT           && echo "  Direct exec:        DISABLED (use output() instead of outputDirect())"
$NO_TRITON           && echo "  Triton:             DISABLED (native CUDA ops only)"
$TRITON_TF32         && echo "  Triton TF32:        ON  (10-bit mantissa for Triton DotOps)"
$DIAG_REPLAY  && echo "  Diag:     GRAPH_REPLAY (capture/instantiate/launch phases)"
$DIAG_STREAM  && echo "  Diag:     STREAM_SYNC (stream ordering, event waits)"
$DIAG_DEVICE  && echo "  Diag:     MULTI_DEVICE (device selection, P2P, migrations)"
$DIAG_ALL     && echo "  Diag:     ALL categories at FULL level"
[ -n "$DIAG_JSON" ] && echo "  Diag JSON: $DIAG_JSON"
$OP_TIMING    && echo "  OpTiming: ON  (decode-only native op timing)"
$OP_TIMING_DETAILED && echo "  OpTiming: detailed phase breakdown ON"
[ -n "$OP_BREAKDOWN_OPS" ] && echo "  Op breakdowns: $OP_BREAKDOWN_OPS"
[ -n "$OP_HISTOGRAM_OPS" ] && echo "  Op histograms: $OP_HISTOGRAM_OPS"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Build extra Maven/JVM args
EXTRA_ARGS=""

# Nsys mode - enables NVIDIA Nsight Systems profiling
if $NSYS_MODE; then
    EXTRA_ARGS="-Dtest.prefix=nsys"
fi

# Optimizer flags (optimizer is ON by default in OnnxModelCache)
if $NO_OPTIMIZER; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.optimizer.enabled=false"
fi

# FP16 is ON by default in QuantizationOptimizations; only pass when disabling
if ! $FP16; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.optimizer.fp16=false"
fi

if $OPTIMIZER_LOG; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.optimizer.logApplied=true"
fi

if [ "$SPECULATIVE_K" -gt 0 ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dvlm.speculative.tokens=$SPECULATIVE_K"
    echo "  Speculation:  K=$SPECULATIVE_K (seqLen=$((SPECULATIVE_K + 1)))"
fi

if $NO_NATIVE_DECODE; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.noNativeDecodeInputs=true"
fi

if $NO_CUBLAS_WORKSPACE; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.cublas.captureWorkspace=0"
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

if $NO_TRITON; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.triton.skipKernels=true"
fi

if $TRITON_TF32; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.triton.tf32=1"
fi

if $DRAFT_MODEL; then
    EXTRA_ARGS="$EXTRA_ARGS -Dvlm.speculative.draft=true"
    echo "  Draft model:  SmolLM2-135M (K=$SPECULATIVE_K)"
fi

# Debug mode flags
if $DEBUG_MODE; then
    CUDA_LOG="$SCRIPT_DIR/cuda-driver.log"
    DSP_LOG="$SCRIPT_DIR/dsp-diagnostics.log"
    echo "  CUDA driver log: $CUDA_LOG"
    echo "  DSP diagnostics: ALL"
    echo ""
    # DSP diagnostics: ALL enables MEMORY, EXECUTION, COMPILATION tracing
    # CUDA_LOG_FILE captures CUDA driver/graph capture errors
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics=ALL"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.native.dumpOutputs=true"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.frozen.summary=true"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.frozen.debug=true"
    EXTRA_ARGS="$EXTRA_ARGS -Dcuda.log.file=$CUDA_LOG"
    # Export CUDA_LOG_FILE for the CUDA driver (picked up by surefire env)
    export CUDA_LOG_FILE="$CUDA_LOG"
fi

# Targeted diagnostic modes — can be combined
if $DIAG_ALL; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics=ALL"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics.level=full"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.frozen.summary=true"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.frozen.debug=true"
    if [ -z "$DIAG_JSON" ]; then
        DIAG_JSON="$SCRIPT_DIR/dsp-diagnostics.json"
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
    OP_TIMING_DIR="$SCRIPT_DIR/op-timing"
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

$MVN test \
  -Dtest="${TEST_CLASS}#${TEST_METHOD}" \
  -Dvlm.test.maxTokens="$MAX_TOKENS" \
  -Dvlm.test.pdf.path=pathfinder-mythic.pdf \
  -Dvlm.test.pdf.page=10 \
  -Dvlm.test.configs="$CONFIG" \
  -Dlibnd4j.triton=ON \
  -Dbackend.artifactId=nd4j-cuda-12.9 \
  $EXTRA_ARGS \
  2>&1 | tee "$LOG_FILE"

BUILD_RESULT=${PIPESTATUS[0]}

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  RESULTS"
echo "═══════════════════════════════════════════════════════════"

if [ $BUILD_RESULT -ne 0 ]; then
    echo "  STATUS: FAILED (exit code $BUILD_RESULT)"
    echo ""
    grep -E "FAILED|ERROR|Exception|assert" "$LOG_FILE" | tail -20
    if $DEBUG_MODE && [ -f "$CUDA_LOG" ]; then
        echo ""
        echo "  CUDA driver log (last 20 lines):"
        tail -20 "$CUDA_LOG"
    fi
    exit 1
fi

if grep -q "Filtered to 0 configs via vlm.test.configs" "$LOG_FILE"; then
    echo "  STATUS: FAILED (requested config resolved to 0 benchmark configs)"
    grep "Filtered to 0 configs via vlm.test.configs" "$LOG_FILE"
    exit 1
fi

if [ "$MAX_TOKENS" -ge 200 ]; then
    if ! grep -q "CREATING A MYTHIC CHARACTER" "$LOG_FILE" && \
       ! grep -q "mythic heroes" "$LOG_FILE" && \
       ! grep -q "hytic heroes" "$LOG_FILE"; then
        echo "  STATUS: FAILED (expected mythic passage not found in benchmark output)"
        exit 1
    fi
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

# Capture buffer summary
cb_lines = [l for l in lines if 'CAPTURE_BUFFER' in l or 'CROSS_SEG' in l]
if cb_lines:
    print("")
    print("  Capture Buffers:")
    seen = set()
    for c in cb_lines:
        key = re.sub(r'execCount=\d+', 'execCount=N', c)
        if key not in seen:
            seen.add(key)
            print("    " + re.sub(r'^.*?\[EXECUTE\]\s*', '', c.strip()))

# Key metrics
print("")
print("  ─────────────────────────────────")
for l in lines:
    m = re.search(r'[Dd]ecode throughput:\s*([\d.]+)\s*tok/s', l)
    if m:
        tps = float(m.group(1))
        status = "OK" if tps >= 100 else "BELOW TARGET"
        print(f"  DECODE THROUGHPUT: {m.group(1)} tok/s  [{status}]")

# Correctness check
all_text = '\n'.join(lines)
has_doctag = 'doctag' in all_text
has_english = any(w in all_text.lower() for w in ['mythic', 'hero', 'path', 'ability', 'tier'])
garbage = all_text.count('UserT') > 3
repeat_lt = sum(1 for l in lines if "'<' (id=44)" in l) > 10

if garbage:
    print("  CORRECTNESS:      FAIL — repeating garbage (UserT)")
elif repeat_lt:
    print("  CORRECTNESS:      FAIL — repeating '<' tokens (stale replay)")
elif has_doctag and has_english:
    print("  CORRECTNESS:      PASS — doctag + coherent English")
elif has_doctag:
    print("  CORRECTNESS:      PARTIAL — doctag present, check text quality")
else:
    print("  CORRECTNESS:      UNKNOWN — no doctag found")

PYEOF
else
    echo "  No surefire report found. Grep from log:"
    grep -E "tok/s|throughput|Performance|CAPTURE_BUFFER" "$LOG_FILE" | tail -10
fi

echo "═══════════════════════════════════════════════════════════"

if $DEBUG_MODE; then
    echo ""
    echo "  Debug files:"
    [ -f "$SCRIPT_DIR/cuda-driver.log" ] && echo "    CUDA driver:     $SCRIPT_DIR/cuda-driver.log"
    echo "    Benchmark log:   $LOG_FILE"
    echo "    Surefire report: $SUREFIRE_OUT"
    echo ""
    # Show lineage tracking summary if present
    if grep -q "LINEAGE\|FROZEN\|EXT_INPUT" "$LOG_FILE" 2>/dev/null; then
        echo "  Lineage/Frozen/ExtInput tracking (last 30 lines):"
        grep -E "LINEAGE|FROZEN|EXT_INPUT_DISCOVER" "$LOG_FILE" | tail -30
    fi
fi

# Show diagnostic summary if any diag mode was enabled
if $DIAG_REPLAY || $DIAG_STREAM || $DIAG_DEVICE || $DIAG_ALL; then
    echo ""
    echo "  ─── DSP Diagnostics Summary ─────────────────────────"

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
        SYNC_EVENTS=$(grep -c "STREAM_SYNC\|stream.*sync\|cudaStreamSynchronize" "$LOG_FILE" 2>/dev/null || true)
        echo "    Sync events: $SYNC_EVENTS"
        grep -E "STREAM_SYNC|stream.*stall|sync.*miss" "$LOG_FILE" 2>/dev/null | tail -10
    fi

    if $DIAG_DEVICE || $DIAG_ALL; then
        echo "  Multi-Device:"
        TRANSFERS=$(grep -c "TRANSFER\|D2D\|H2D\|D2H" "$LOG_FILE" 2>/dev/null || true)
        DEVICE_SWITCH=$(grep -c "MULTI_DEVICE\|device.*switch\|switchDevice" "$LOG_FILE" 2>/dev/null || true)
        echo "    Transfer events:  $TRANSFERS"
        echo "    Device switches:  $DEVICE_SWITCH"
        grep -E "MULTI_DEVICE|memory.*pressure|reroute" "$LOG_FILE" 2>/dev/null | tail -10
    fi

    if [ -n "$DIAG_JSON" ] && [ -f "$DIAG_JSON" ]; then
        echo ""
        echo "  JSON diagnostic report: $DIAG_JSON"
    fi
    echo "  ─────────────────────────────────────────────────────"
fi
