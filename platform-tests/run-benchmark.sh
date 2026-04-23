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
VALIDATION_CLASS="TestDspValidation"
VALIDATION_METHOD="testOutputAccuracy+testDecodeStepValidation"
TEST_CLASS="TestSmolDoclingOptimizedPipeline"
TEST_METHOD="testOptimizedDoclingPipeline"
VALIDATION_LOG="$SCRIPT_DIR/dsp-validation.log"
LOG_FILE="$SCRIPT_DIR/bench-output.log"
SUREFIRE_OUT="$SCRIPT_DIR/target/surefire-reports/org.eclipse.deeplearning4j.vlm.${TEST_CLASS}.txt"
SUREFIRE_XML="$SCRIPT_DIR/target/surefire-reports/TEST-org.eclipse.deeplearning4j.vlm.${TEST_CLASS}.xml"
MODEL_CACHE="$HOME/.cache/dl4j-vlm-models"

# Backend: cuda (default) or cpu
BACKEND="cuda"

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
NO_CUBLAS_WORKSPACE=false
NO_FREEZE=false
TRITON_TF32=false
NO_ATTN_OVERRIDE=false
NO_DIRECT=false
NO_TRITON=false
DISABLE_VIEW_FASTPATH=false
DISABLE_CAST_HWM=false
DISABLE_WS_SKIP=false
DSP_TIMING=false
DIAG_REPLAY=false
DIAG_STREAM=false
DIAG_DEVICE=false
DIAG_ALL=false
DIAG_JSON=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"
            shift 2
            ;;
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
        --disable-view-fastpath)
            DISABLE_VIEW_FASTPATH=true
            shift
            ;;
        --disable-cast-hwm)
            DISABLE_CAST_HWM=true
            shift
            ;;
        --disable-ws-skip)
            DISABLE_WS_SKIP=true
            shift
            ;;
        --dsp-timing)
            DSP_TIMING=true
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
            echo "       [--draft] [--speculative K] [--no-cublas-workspace] [--no-freeze]"
            echo "       [--triton-tf32] [--no-triton-tf32]"
echo "       [--disable-view-fastpath] [--disable-cast-hwm] [--disable-ws-skip]"
echo "       [--dsp-timing]"
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
$NO_CUBLAS_WORKSPACE && echo "  cuBLAS workspace:   DISABLED (no explicit workspace during capture)"
$NO_FREEZE           && echo "  Freeze:             DISABLED (no shape freezing, no CUDA graph)"
$NO_ATTN_OVERRIDE    && echo "  AttnOverride:       DISABLED (use model's attn_mask_reformat subgraph)"
$NO_DIRECT           && echo "  Direct exec:        DISABLED (use output() instead of outputDirect())"
$NO_TRITON           && echo "  Triton:             DISABLED (native CUDA ops only)"
$TRITON_TF32         && echo "  Triton TF32:        ON  (10-bit mantissa for Triton DotOps)"
$DISABLE_VIEW_FASTPATH && echo "  ISOLATION: view-op fast path DISABLED"
$DISABLE_CAST_HWM     && echo "  ISOLATION: cast cache HWM DISABLED (reset to 0)"
$DISABLE_WS_SKIP      && echo "  ISOLATION: workspace skip DISABLED (live gaps use workspace)"
$DIAG_REPLAY  && echo "  Diag:     GRAPH_REPLAY (capture/instantiate/launch phases)"
$DIAG_STREAM  && echo "  Diag:     STREAM_SYNC (stream ordering, event waits)"
$DIAG_DEVICE  && echo "  Diag:     MULTI_DEVICE (device selection, P2P, migrations)"
$DIAG_ALL     && echo "  Diag:     ALL categories at FULL level"
$DSP_TIMING   && echo "  Diag:     DSP_TIMING (COMPOSITE_REPLAY breakdown)"
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

# Isolation flags — disable individual replay mechanisms for debugging
if $DISABLE_VIEW_FASTPATH; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.disableViewFastpath=1"
fi
if $DISABLE_CAST_HWM; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.disableCastHwm=1"
fi
if $DISABLE_WS_SKIP; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.disableWsSkip=1"
fi

if $DSP_TIMING; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.timing=1"
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

VALIDATION_TOKENS="$MAX_TOKENS"
if [ "$VALIDATION_TOKENS" -gt 10 ]; then
    VALIDATION_TOKENS=10
fi

echo "Running DSP validation preflight (${VALIDATION_CLASS}#${VALIDATION_METHOD}, tokens=${VALIDATION_TOKENS})..."
# Validation runs WITHOUT diagnostic flags — diagnostics add overhead and
# are only meaningful on the benchmark run. Only pass optimizer/precision flags.
VALIDATION_ARGS=""
if $NO_OPTIMIZER; then
    VALIDATION_ARGS="$VALIDATION_ARGS -Dnd4j.optimizer.enabled=false"
fi
if ! $FP16; then
    VALIDATION_ARGS="$VALIDATION_ARGS -Dnd4j.optimizer.fp16=false"
fi
if $NO_FREEZE; then
    VALIDATION_ARGS="$VALIDATION_ARGS -Dnd4j.dsp.nofreeze=true"
fi
if $NO_TRITON; then
    VALIDATION_ARGS="$VALIDATION_ARGS -Dnd4j.triton.skipKernels=true"
fi
if $TRITON_TF32; then
    VALIDATION_ARGS="$VALIDATION_ARGS -Dnd4j.triton.tf32=1"
fi

# ─── Backend resolution ──────────────────────────────────────────────
if [ "$BACKEND" = "cpu" ]; then
    BACKEND_ARTIFACT="nd4j-native"
    TRITON_FLAG=""
    # CPU-specific: add OMP thread configuration
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.omp.numthreads=${OMP_NUM_THREADS:-$(nproc)}"
    # Skip CUDA-only flags
    NSYS_MODE=false
    DRAFT_MODEL=false
    SPECULATIVE_K=0
    NO_CUBLAS_WORKSPACE=false
    TRITON_TF32=false
    NO_TRITON=false
    echo "[backend] CPU mode: artifact=$BACKEND_ARTIFACT"
elif [ "$BACKEND" = "cuda" ]; then
    BACKEND_ARTIFACT="nd4j-cuda-12.9"
    TRITON_FLAG="-Dlibnd4j.triton=ON"
    echo "[backend] CUDA mode: artifact=$BACKEND_ARTIFACT"
else
    echo "ERROR: Unknown backend '$BACKEND'. Use 'cuda' or 'cpu'."
    exit 1
fi

set +e
$MVN test \
  -Dtest="${VALIDATION_CLASS}#${VALIDATION_METHOD}" \
  -Dvlm.validation.tokens="$VALIDATION_TOKENS" \
  -Dvlm.validation.configs="$CONFIG" \
  $TRITON_FLAG \
  -Dbackend.artifactId=$BACKEND_ARTIFACT \
  $VALIDATION_ARGS \
  2>&1 | tee "$VALIDATION_LOG"
VALIDATION_RESULT=${PIPESTATUS[0]}
set -e

if [ $VALIDATION_RESULT -ne 0 ]; then
    echo ""
    echo "Validation failed before benchmark execution."
    echo "Validation log: $VALIDATION_LOG"
    exit $VALIDATION_RESULT
fi

echo ""
echo "Validation passed. Starting benchmark..."

set +e
$MVN test \
  -Dtest="${TEST_CLASS}#${TEST_METHOD}" \
  -Dvlm.test.maxTokens="$MAX_TOKENS" \
  -Dvlm.test.pdf.path=pathfinder-mythic.pdf \
  -Dvlm.test.pdf.page=10 \
  -Dvlm.test.configs="$CONFIG" \
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

if [ $BUILD_RESULT -ne 0 ]; then
    echo "  STATUS: FAILED (exit code $BUILD_RESULT)"
    echo ""
    # Still run the Python parser to show metrics + DSP health — a perf assertion
    # failure means the test ran but didn't meet the target. We still want to see
    # the actual throughput and DSP health to diagnose why.
fi

if grep -q "Filtered to 0 configs via vlm.test.configs" "$LOG_FILE"; then
    echo "  STATUS: FAILED (requested config resolved to 0 benchmark configs)"
    grep "Filtered to 0 configs via vlm.test.configs" "$LOG_FILE"
    exit 1
fi

# Content check uses surefire report (benchmark only), NOT the tee log
# which mixes validation + benchmark output
if [ "$MAX_TOKENS" -ge 200 ] && [ -f "$SUREFIRE_OUT" ]; then
    if ! grep -q "CREATING A MYTHIC CHARACTER" "$SUREFIRE_OUT" && \
       ! grep -q "mythic heroes" "$SUREFIRE_OUT" && \
       ! grep -q "hytic heroes" "$SUREFIRE_OUT"; then
        echo "  STATUS: FAILED (expected mythic passage not found in benchmark output)"
        echo "  (checked $SUREFIRE_OUT, not the mixed tee log)"
        exit 1
    fi
fi

# Extract metrics from surefire XML (has full stdout) or text fallback.
# The .txt file is just a summary line — actual test output is in the XML.
REPORT=""
if [ -f "$SUREFIRE_XML" ]; then
    REPORT="$SUREFIRE_XML"
elif [ -f "$SUREFIRE_OUT" ] && [ -s "$SUREFIRE_OUT" ]; then
    REPORT="$SUREFIRE_OUT"
fi

if [ -n "$REPORT" ]; then
    python3 - "$REPORT" "$MAX_TOKENS" <<'PYEOF'
import sys, re, xml.etree.ElementTree as ET

report_path = sys.argv[1]
max_tokens = int(sys.argv[2])

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

# ── FIND THE BENCHMARK RUN ──
# The surefire output contains ONLY the benchmark test class output.
# Look for the BenchmarkRunner summary line — this is the authoritative result.
# Find the [PASS] or [FAIL] summary from BenchmarkRunner.
# Two formats exist:
#   [PASS] CONFIG: N tokens, overall=X tok/s, decode=Y tok/s, steady=Z tok/s, ...
#   [FAIL] CONFIG: IllegalStateException: ... result{...,throughput=Z,...}
# Prefer [PASS] (has full metrics). Fall back to [FAIL] for throughput extraction.
benchmark_summary = None
pass_summary = None
fail_summary = None
for l in lines:
    stripped = l.strip()
    if '[PASS]' in stripped and ('tok/s' in stripped or 'throughput=' in stripped):
        pass_summary = stripped
    if '[FAIL]' in stripped and ('tok/s' in stripped or 'throughput=' in stripped):
        fail_summary = stripped
# Prefer [PASS] — it has steady=X tok/s format. [FAIL] only has result{throughput=X}.
benchmark_summary = pass_summary or fail_summary

# ── EXTRACT THE ONE TRUE METRIC ──
# Steady-state is the ONLY number that matters for decode performance.
steady_tps = None
decode_tps = None
overall_tps = None
first_token_ms = None
config_name = None
token_count = None
warmup_steps = None

if benchmark_summary:
    # Try steady=X tok/s format first ([PASS] line)
    m = re.search(r'steady=([\d.]+)\s*tok/s', benchmark_summary)
    if m: steady_tps = float(m.group(1))
    # Fallback: result{...,throughput=X,...} format ([FAIL] line)
    if steady_tps is None:
        m = re.search(r'throughput=([\d.]+)', benchmark_summary)
        if m: steady_tps = float(m.group(1))
    m = re.search(r'decode=([\d.]+)\s*tok/s', benchmark_summary)
    if m: decode_tps = float(m.group(1))
    m = re.search(r'overall=([\d.]+)\s*tok/s', benchmark_summary)
    if m: overall_tps = float(m.group(1))
    m = re.search(r'firstToken=([\d.]+)ms', benchmark_summary)
    if m: first_token_ms = float(m.group(1))
    m = re.search(r'\]\s+(\w+):', benchmark_summary)
    if m: config_name = m.group(1)
    m = re.search(r'(\d+)\s+tokens', benchmark_summary)
    if m: token_count = int(m.group(1))
    m = re.search(r'warmup:\s*(\d+)\s*steps', benchmark_summary)
    if m: warmup_steps = int(m.group(1))

# ── DSP HEALTH CHECK ──
# Extract from the plan stats line (always present in benchmark output).
# Format: plan: decoder{N/M cap,R replay,H host,valid,captured=C(Sslots),...}
# Also from the DSP state log: segments=S, captured=C, replays=R
errors = []

# Parse the plan stats from the [PASS]/[FAIL] summary line
plan_line = benchmark_summary or ''
plan_stats_line = None
for l in lines:
    if 'segments=' in l and 'captured=' in l and 'replays=' in l:
        plan_stats_line = l.strip()

# Segments
seg_count = 0
m = re.search(r'segments=(\d+)', plan_stats_line or '')
if m: seg_count = int(m.group(1))

# Captured segments (from "N/M cap" or "captured=N")
segs_captured = 0
m = re.search(r'(\d+)/(\d+)\s+cap', plan_line)
if m: segs_captured = int(m.group(1))
if segs_captured == 0:
    m = re.search(r'(?<!\w)captured=(\d+)', plan_stats_line or '')
    if m: segs_captured = int(m.group(1))

# Replays
replay_count = 0
m = re.search(r'replays=(\d+)', plan_stats_line or '')
if m: replay_count = int(m.group(1))
if replay_count == 0:
    m = re.search(r'(\d+)\s+replay', plan_line)
    if m: replay_count = int(m.group(1))

# Captured slots (from stats=captured=C(Sslots))
cap_stats_count = 0
cap_stats_slots = 0
m = re.search(r'stats=captured=(\d+)\((\d+)slots\)', plan_stats_line or plan_line)
if m:
    cap_stats_count = int(m.group(1))
    cap_stats_slots = int(m.group(2))

# captureValid
capture_valid = 'captureValid=true' in (plan_stats_line or plan_line)

# PermFailed / OOM
perm_failed = 0
m = re.search(r'permFailed=(\d+)', plan_line)
if m: perm_failed = int(m.group(1))

oom_retrying = 0
m = re.search(r'oomRetrying=(\d+)', plan_line)
if m: oom_retrying = int(m.group(1))

# Triton launch count
triton_launches = 0
m = re.search(r'triton:\s*launches=(\d+)', plan_line)
if m: triton_launches = int(m.group(1))

# DSP_DIAG lines (only present with --diag-all, but check anyway)
composite_replay_count = sum(1 for l in lines if 'COMPOSITE_REPLAY_ENTER' in l)
composite_capture_ok = sum(1 for l in lines if 'COMPOSITE_CAPTURE_COMPLETE' in l)
island_launches = sum(1 for l in lines if 'COMPOSITE_REPLAY: island' in l and 'launching' in l)
capture_failures = sum(1 for l in lines if 'CAPTURE_FAIL' in l or 'launch FAILED' in l)
arg_stable_count = sum(1 for l in lines if 'ARG_TABLE_STABLE' in l)

# ── BUILD ERROR LIST ──
if segs_captured == 0 and seg_count > 0:
    errors.append(f"NO SEGMENTS CAPTURED: {seg_count} segments but 0 captured — CUDA graph capture not working")
if not capture_valid and seg_count > 0:
    errors.append("CAPTURE INVALID: captureValid=false — graph capture failed or was invalidated")
if replay_count == 0 and segs_captured > 0:
    errors.append(f"NO REPLAYS: {segs_captured} captured but 0 replayed — graph replay not firing")
if capture_failures > 0:
    errors.append(f"CAPTURE FAILURES: {capture_failures} — graphs failing during capture/launch")
if perm_failed > 0:
    errors.append(f"PERMANENT FAILURES: {perm_failed} segments permanently failed capture")
if oom_retrying > 0:
    errors.append(f"OOM RETRYING: {oom_retrying} segments hit OOM during graph instantiation")

# ── DECODE STEPS ──
# Show ONLY the benchmark run steps (after the last "[PREFILL] Step 0")
all_steps = []
last_prefill_idx = -1
for i, l in enumerate(lines):
    if '[PREFILL] Step 0' in l:
        last_prefill_idx = i

# Collect steps from the last (benchmark) decode loop only
if last_prefill_idx >= 0:
    for l in lines[last_prefill_idx:]:
        if ('Step' in l and 'id=' in l) or ('Steady-state' in l) or ('Decode throughput' in l):
            all_steps.append(l.strip())

# ── PRINT RESULTS ──
print("")
print(f"  Config:  {config_name or 'UNKNOWN'}")
print(f"  Tokens:  {token_count or max_tokens}")
if warmup_steps is not None:
    print(f"  Warmup:  {warmup_steps} steps")
print("")

# The one number
print("  ┌─────────────────────────────────────────────┐")
if steady_tps is not None:
    status = "OK" if steady_tps >= 100 else "BELOW TARGET"
    bar = "█" * min(int(steady_tps), 50) + "░" * max(0, 50 - min(int(steady_tps), 50))
    print(f"  │  STEADY-STATE: {steady_tps:>7.2f} tok/s  [{status:>12s}] │")
else:
    print(f"  │  STEADY-STATE: ???     tok/s  [NO DATA]       │")
print("  └─────────────────────────────────────────────┘")
print("")

# Supporting numbers
if decode_tps is not None:
    print(f"  Decode (incl warmup): {decode_tps:.2f} tok/s")
if first_token_ms is not None:
    print(f"  First token latency:  {first_token_ms:.0f} ms")

# Decode steps (last run only)
if all_steps:
    print("")
    print("  Steps (benchmark run only):")
    show = all_steps[:8] + (['    ...'] if len(all_steps) > 13 else []) + all_steps[-5:] if len(all_steps) > 13 else all_steps
    for s in show:
        s = re.sub(r'^.*?(\[(?:PREFILL|WARMUP|STEADY)\])', r'    \1', s)
        s = re.sub(r'^.*?[Dd]ecode throughput', '    Decode throughput', s)
        s = re.sub(r'^.*?[Ss]teady-state', '    Steady-state', s)
        print(s)

# Correctness
print("")
all_text = '\n'.join(lines[last_prefill_idx:] if last_prefill_idx >= 0 else lines)
has_doctag = 'doctag' in all_text
has_english = any(w in all_text.lower() for w in ['mythic', 'hero', 'path', 'ability', 'tier'])
garbage = all_text.count('UserT') > 3
repeat_lt = sum(1 for l in lines if "'<' (id=44)" in l) > 10

if garbage:
    print("  CORRECTNESS: FAIL — repeating garbage (UserT)")
elif repeat_lt:
    print("  CORRECTNESS: FAIL — repeating '<' tokens (stale replay)")
elif has_doctag and has_english:
    print("  CORRECTNESS: PASS")
elif has_doctag:
    print("  CORRECTNESS: PARTIAL — doctag present, check text quality")
else:
    print("  CORRECTNESS: UNKNOWN — no doctag found (may need more tokens)")

# DSP Health — always extracted from plan stats (no --diag-all needed)
print("")
print("  ─── DSP Health ──────────────────────────────────")
print(f"  Segments:          {seg_count}")
print(f"  Captured:          {segs_captured}/{seg_count}")
print(f"  Replays:           {replay_count}")
print(f"  Capture valid:     {capture_valid}")
print(f"  Triton launches:   {triton_launches}")
print(f"  Perm failures:     {perm_failed}")
print(f"  OOM retrying:      {oom_retrying}")
# DSP_DIAG detail (only with --diag-all)
if composite_replay_count > 0 or island_launches > 0 or arg_stable_count > 0:
    print(f"  [diag] Composite replay enters: {composite_replay_count}")
    print(f"  [diag] Island launches:         {island_launches}")
    print(f"  [diag] Arg table stable:        {arg_stable_count}")
    print(f"  [diag] Capture failures:        {capture_failures}")

if errors:
    print("")
    print("  *** ERRORS — fix these before trusting perf numbers ***")
    for e in errors:
        print(f"  *** {e}")
else:
    print("")
    print("  Health: OK")

PYEOF
else
    echo "  No surefire report found at: $SUREFIRE_OUT"
    echo "  Check benchmark log: $LOG_FILE"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Logs:"
echo "    Benchmark:  $LOG_FILE"
echo "    Validation: $VALIDATION_LOG"
echo "    Surefire:   $SUREFIRE_OUT"
echo "═══════════════════════════════════════════════════════════"

# Exit with build result — 0 for pass, non-zero for fail
exit $BUILD_RESULT
