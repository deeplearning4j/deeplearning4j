#!/bin/bash
# SmolDocling VLM Decode Benchmark — pathfinder-mythic.pdf page 10
# Target: 100+ tok/s late steady-state with coherent SmolDocling DocTags.
# Current coherent baseline is ~64-67 tok/s; correctness is judged from FULL_TEXT.
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ CURRENT OPTIMAL DEFAULTS (update this when perf improves)          │
# │                                                                     │
# │  FP16 weight pre-cast:  ON  (default, disable: --no-fp16)         │
# │  GraphOptimizer:        ON  (default, disable: --no-optimizer)     │
# │  Clear decoder cache:   OFF (enable: --clear-decoder)              │
# │  Config:                SMOLDOC_IDEAL (OPTIMAL + text proof)        │
# │  Max tokens:            250                                         │
# │                                                                     │
# │  Ideal command:         ./run-benchmark.sh                          │
# │  Audit command:         ./run-benchmark.sh --audit                  │
# │                                                                     │
# │  Best measured:         ~64-67 tok/s late steady-state decode       │
# │  FP16 note:             upcast HALF→FLOAT32 (no downcast)          │
# │  Expected content:      ythic heroes / mythic characters / paths    │
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
#   --config NAME     Override benchmark config name (default: SMOLDOC_IDEAL)
#   --ideal           Synonym for the owner-facing default golden path:
#                     SMOLDOC_IDEAL, 250 tokens, --skip-audit,
#                     --no-clear-decoder, FULL_TEXT mythic passage proof
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
#   --clear-decoder     Delete only the decoder .sdz cache (default: OFF)
#   --no-clear-decoder  Keep decoder .sdz cache (skip re-import)
#
# Examples:
#   ./run-benchmark.sh                           # Default: SMOLDOC_IDEAL text-proof config
#   ./run-benchmark.sh --ideal                   # Quick cached perf check for the same config
#   ./run-benchmark.sh --tokens 100              # Quick 100-token run
#   ./run-benchmark.sh --no-clear-decoder         # Keep cached decoder (skip re-import)
#   ./run-benchmark.sh --debug                    # Full DSP diagnostics + CUDA driver log
#   ./run-benchmark.sh --op-timing                # Decode-only op timing CSV + hotspot table
#   ./run-benchmark.sh --no-fp16                  # FP32 weights (baseline comparison)
#   ./run-benchmark.sh --no-optimizer             # No optimization at all
#
# DSP audit options:
#   --audit             Run the DSP test audit after the benchmark
#   --skip-audit        Skip the DSP test audit entirely (fast benchmark only)
#   --audit-only        Run ONLY the DSP audit, skip the benchmark
#   --audit-suite SUITE Which audit suites to run. Comma-separated list from:
#                         lifecycle   — DspLifecycleValidationTest, DspLifecycleExhaustiveTest,
#                                      DspSlotLifecycleAuditTest, DspLifecycleGates
#                         frozen      — DspFrozenConstantInvariantTest, FrozenPhaseDriftDetection,
#                                      DspViewOpFrozenReplayTest, ValueDependentShapeClassification
#                         replay      — DspCompositeReplayTest, DspDeepIsolationTest,
#                                      DspPipelineIsolationTest, DspRepeatedOutputFreshness,
#                                      DspMergedSegmentReplay
#                         regression  — DspRegressionHarness tests: SegmentOutputZeroInvariant,
#                                      CrossStreamEventOrdering, GapExecutionSlotInvariants,
#                                      ArgTableStablePerfFloor
#                         capture     — DspCaptureConfigMatrix, DspHandleTest, DspHandleDataModelTest
#                         training    — DspTrainingE2ETest, DspOptimizedSlotBySlotTest
#                         ext-input   — DspExtInputStalenessTest, DspValueKeySegmentTest
#                         pooling     — DynamicShapePlanPoolingTest, DspBatchedModulePreloadTest,
#                                      DspLruModuleResidencyTest, DspCompilationSealTest
#                         validation  — TestDspValidation (outputAccuracy + decodeStep)
#                         all         — Run ALL suites (DEFAULT)
#   --audit-timeout N   Timeout in seconds for the audit phase (default: 600)
#
# Examples with audit:
#   ./run-benchmark.sh --audit                     # Benchmark + full DSP audit
#   ./run-benchmark.sh --audit-only                # Audit only, no benchmark
#   ./run-benchmark.sh --audit-suite lifecycle,frozen  # Benchmark + selected suites
#   ./run-benchmark.sh --audit-suite validation    # Benchmark + validation only
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
CONFIG="SMOLDOC_IDEAL"
IDEAL_MODE=true
FP16=true
FP16_EXPLICIT=false  # set to true when user passes --fp16 or --no-fp16 explicitly
NO_OPTIMIZER=false
OPTIMIZER_LOG=false
CLEAR_CACHE=false
CLEAR_DECODER=false
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
DIAG_DETAILED=false
DIAG_JSON=""
DIAG_STEP=false
DIAG_D2D=false
DIAG_CAPTURE=false
DSP_ASSERT=false
SKIP_AUDIT=true
AUDIT_ONLY=false
SKIP_FINAL_VALIDATE=false
# Gap capture tuning (empty = use C++ defaults)
GAP_MAX_SLOTS=""
GAP_BLOCK_EXT_WS=""
GAP_TC=""
GAP_TC_WARMUP=""
AUDIT_SUITE="all"
AUDIT_TIMEOUT=600

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
            if [ "$CONFIG" = "SMOLDOC_IDEAL" ]; then
                IDEAL_MODE=true
            else
                IDEAL_MODE=false
            fi
            shift 2
            ;;
        --ideal|--golden)
            IDEAL_MODE=true
            CONFIG="SMOLDOC_IDEAL"
            MAX_TOKENS=250
            SKIP_AUDIT=true
            CLEAR_DECODER=false
            shift
            ;;
        --fp16)
            FP16=true
            FP16_EXPLICIT=true
            shift
            ;;
        --no-fp16)
            FP16=false
            FP16_EXPLICIT=true
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
        --diag-detailed)
            DIAG_DETAILED=true
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
        --diag-step)
            DIAG_STEP=true
            shift
            ;;
        --diag-d2d)
            DIAG_D2D=true
            shift
            ;;
        --diag-capture)
            DIAG_CAPTURE=true
            shift
            ;;
        --diag-json)
            DIAG_JSON="$2"
            shift 2
            ;;
        --dsp-assert)
            DSP_ASSERT=true
            shift
            ;;
        --skip-audit)
            SKIP_AUDIT=true
            shift
            ;;
        --audit)
            SKIP_AUDIT=false
            shift
            ;;
        --skip-final-validate|--skip-validation)
            SKIP_FINAL_VALIDATE=true
            shift
            ;;
        --audit-only)
            AUDIT_ONLY=true
            SKIP_AUDIT=false
            shift
            ;;
        --audit-suite)
            AUDIT_SUITE="$2"
            SKIP_AUDIT=false
            shift 2
            ;;
        --audit-timeout)
            AUDIT_TIMEOUT="$2"
            shift 2
            ;;
        --max-gap-slots)
            GAP_MAX_SLOTS="$2"
            shift 2
            ;;
        --no-gap-block-ext-ws)
            GAP_BLOCK_EXT_WS="0"
            shift
            ;;
        --gap-block-ext-ws)
            GAP_BLOCK_EXT_WS="1"
            shift
            ;;
        --gap-tensor-cores)
            GAP_TC="1"
            shift
            ;;
        --no-gap-tensor-cores)
            GAP_TC="0"
            shift
            ;;
        --gap-tc-warmup)
            GAP_TC_WARMUP="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run-benchmark.sh [--debug] [--nsys] [--op-timing] [--op-timing-detailed]"
            echo "       [--op-breakdown OPS] [--op-histogram OPS] [--tokens N] [--config NAME]"
            echo "       [--ideal]  # SMOLDOC_IDEAL, 250 tokens, skip audit, keep decoder cache"
            echo "       [--fp16] [--no-fp16] [--no-optimizer] [--optimizer-log]"
            echo "       [--clear-cache] [--clear-decoder] [--no-clear-decoder]"
            echo "       [--draft] [--speculative K] [--no-cublas-workspace] [--no-freeze]"
            echo "       [--triton-tf32] [--no-triton-tf32]"
echo "       [--disable-view-fastpath] [--disable-cast-hwm] [--disable-ws-skip]"
            echo "       [--dsp-timing] [--skip-final-validate]"
            echo "       [--diag-replay] [--diag-stream] [--diag-device] [--diag-all] [--diag-json FILE]"
echo "       [--diag-step] [--diag-d2d] [--diag-capture]"
echo "       [--dsp-assert]"
echo "       [--max-gap-slots N] [--gap-block-ext-ws] [--no-gap-block-ext-ws]"
echo "       [--gap-tensor-cores] [--no-gap-tensor-cores] [--gap-tc-warmup N]"
echo "       [--audit] [--skip-audit] [--audit-only] [--audit-suite SUITE] [--audit-timeout N]"
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
    rm -f "$MODEL_CACHE/smoldocling-decoder.nofp16.opt.sdz" 2>/dev/null || true
fi

# CPU backend: auto-disable FP16 unless user explicitly passed --fp16.
# On CPU, HALF weights use software FP16 emulation (~13 seconds per token).
# The GraphOptimizer FP16 pre-cast is only beneficial with CUDA Tensor Cores.
if [ "$BACKEND" = "cpu" ] && ! $FP16_EXPLICIT; then
    FP16=false
    echo "  [CPU] Auto-disabled FP16 weight pre-cast (use --fp16 to override)"
fi

echo "═══════════════════════════════════════════════════════════"
echo "  SmolDocling VLM Decode Benchmark"
echo "  PDF:    pathfinder-mythic.pdf page 10"
echo "  Tokens: $MAX_TOKENS"
echo "  Config: $CONFIG"
echo "  Target: 100+ tok/s, coherent mythic-heroes FULL_TEXT"
$IDEAL_MODE   && echo "  Ideal:  SMOLDOC_IDEAL text-proof config"
$IDEAL_MODE   && echo "  Proof:  FULL_TEXT contains ythic heroes, mythic characters, mythic paths"
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
$DIAG_STEP    && echo "  Diag:     Per-step StepSnapshot introspection"
$DIAG_D2D     && echo "  Diag:     D2D copy status per step"
$DIAG_CAPTURE && echo "  Diag:     Capture quality audit"
$DSP_TIMING   && echo "  Diag:     DSP_TIMING (COMPOSITE_REPLAY breakdown)"
[ -n "$DIAG_JSON" ] && echo "  Diag JSON: $DIAG_JSON"
$OP_TIMING    && echo "  OpTiming: ON  (decode-only native op timing)"
$OP_TIMING_DETAILED && echo "  OpTiming: detailed phase breakdown ON"
[ -n "$OP_BREAKDOWN_OPS" ] && echo "  Op breakdowns: $OP_BREAKDOWN_OPS"
[ -n "$OP_HISTOGRAM_OPS" ] && echo "  Op histograms: $OP_HISTOGRAM_OPS"
if $AUDIT_ONLY; then
    echo "  Audit:    ONLY (no benchmark)"
elif $SKIP_AUDIT; then
    echo "  Audit:    SKIPPED"
else
    echo "  Audit:    ON (suite: $AUDIT_SUITE, timeout: ${AUDIT_TIMEOUT}s)"
fi
$SKIP_FINAL_VALIDATE && echo "  Final validation: SKIPPED"
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

# QuantizationOptimizations' FP16 default is OFF (since fd89cb724f), so the old
# "only pass when disabling" relied on a default that no longer exists and silently
# ran the benchmark with FP16 OFF. Pass the flag EXPLICITLY so the benchmark's FP16
# setting (default ON) is honored regardless of the code default.
if $FP16; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.optimizer.fp16=true"
else
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
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.noFreeze=true"
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
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.executionTiming=true"
    # COMPOSITE_REPLAY_TIMING emits via DSP_DIAG(TIMING,...) — executionTiming alone
    # ticks the timers but records nothing unless the TIMING category is enabled.
    # Use sync-free 'detailed' level (ring + JSON dump; 'full' echoes/syncs and
    # distorts the numbers being measured). If combined with --diag-* flags their
    # later -Dnd4j.dsp.diagnostics properties win (Maven last-wins), as before.
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics=TIMING"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics.level=detailed"
    if [ -z "$DIAG_JSON" ]; then
        DIAG_JSON="$SCRIPT_DIR/dsp-timing.json"
    fi
fi

if $DRAFT_MODEL; then
    EXTRA_ARGS="$EXTRA_ARGS -Dvlm.speculative.draft=true"
    echo "  Draft model:  SmolLM2-135M (K=$SPECULATIVE_K)"
fi

if $SKIP_FINAL_VALIDATE; then
    EXTRA_ARGS="$EXTRA_ARGS -Dvlm.test.skipFinalValidate=true"
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
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.env.verbose=true"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.env.debug=true"
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
# Detailed sync-free DSP_DIAG: EXECUTE+LIFECYCLE+MEMORY at level=detailed. The ring
# captures WITHOUT syncToHost, so it does NOT force slot-by-slot (unlike level=full),
# and is dumped to a JSON ring file post-run. Use to OBSERVE the consolidated
# GraphSegmentExec transitions (markArgsStale/resetCaptureKeys/resetForWarmup/
# slotAddrDrifted) during decode without corrupting what's being measured.
if $DIAG_DETAILED; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics=EXECUTE,LIFECYCLE,MEMORY"
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics.level=detailed"
    if [ -z "$DIAG_JSON" ]; then
        DIAG_JSON="$SCRIPT_DIR/dsp-ring-detailed.json"
    fi
fi
if [ -n "$DIAG_JSON" ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.diagnostics.file=$DIAG_JSON"
fi

# Pipeline introspection flags (DspHandle StepSnapshot, D2D, capture audit)
if $DIAG_STEP; then
    EXTRA_ARGS="$EXTRA_ARGS -Dvlm.diag.stepSnapshot=true"
fi
if $DIAG_D2D; then
    EXTRA_ARGS="$EXTRA_ARGS -Dvlm.diag.d2dCheck=true"
fi
if $DIAG_CAPTURE; then
    EXTRA_ARGS="$EXTRA_ARGS -Dvlm.diag.captureAudit=true"
fi
if $DSP_ASSERT; then
    EXTRA_ARGS="$EXTRA_ARGS -Dvlm.benchmark.dspAssert=true"
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

# Gap capture tuning knobs
if [ -n "$GAP_MAX_SLOTS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.maxCapturableGapSlots=$GAP_MAX_SLOTS"
    echo "  Gap max slots: $GAP_MAX_SLOTS"
fi
if [ -n "$GAP_BLOCK_EXT_WS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.gapCaptureBlockExternalWorkspace=$GAP_BLOCK_EXT_WS"
    echo "  Gap block ext ws: $GAP_BLOCK_EXT_WS"
fi
if [ -n "$GAP_TC" ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.gapTensorCores=$GAP_TC"
    echo "  Gap tensor cores: $GAP_TC"
fi
if [ -n "$GAP_TC_WARMUP" ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dnd4j.dsp.gapTensorCoreWarmup=$GAP_TC_WARMUP"
    echo "  Gap TC warmup: $GAP_TC_WARMUP"
fi

VALIDATION_TOKENS="$MAX_TOKENS"
if [ "$VALIDATION_TOKENS" -gt 10 ]; then
    VALIDATION_TOKENS=10
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

BUILD_RESULT=0

if ! $AUDIT_ONLY; then
echo ""
echo "Running benchmark..."

set +e
$MVN test \
  -Dtest="${TEST_CLASS}#${TEST_METHOD}" \
  -Dvlm.test.maxTokens="$MAX_TOKENS" \
  -Dvlm.test.pdf.path=pathfinder-mythic.pdf \
  -Dvlm.test.pdf.page=10 \
  -Dvlm.test.configs="$CONFIG" \
  -Dsd.test.skip.large=false \
  -Dsd.test.skip.downloads=false \
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

# Detect JVM crash (SIGABRT=134, SIGSEGV=139, SIGKILL=137, OOM=137)
JVM_CRASHED=false
if [ $BUILD_RESULT -ne 0 ]; then
    if grep -q "SIGABRT\|SIGSEGV\|SIGKILL\|OutOfMemoryError\|JVM killed" "$LOG_FILE"; then
        JVM_CRASHED=true
        echo "  STATUS: CRASHED (JVM killed, exit code $BUILD_RESULT)"
        # Show what signal killed it
        grep "SIGABRT\|SIGSEGV\|SIGKILL\|JVM killed\|OutOfMemoryError" "$LOG_FILE" | head -3
        echo ""
    else
        echo "  STATUS: FAILED (exit code $BUILD_RESULT)"
        echo ""
    fi
fi

if grep -q "Filtered to 0 configs via vlm.test.configs" "$LOG_FILE"; then
    echo "  STATUS: FAILED (requested config resolved to 0 benchmark configs)"
    grep "Filtered to 0 configs via vlm.test.configs" "$LOG_FILE"
    exit 1
fi

# Early abort on crash — no point checking content if JVM died
if $JVM_CRASHED; then
    echo "  Skipping content check — JVM crashed before generating text"
    echo ""
fi

# Content check: look for mythic content in GENERATED text only (decode step lines).
# Check for mythic passage content in the generated output.
# Multiple detection methods since different pipelines log differently:
# 1. Token-by-token output lines (contain 'id=' markers)
# 2. "generated text" summary blocks
# 3. FULL_TEXT log line (from optimized pipeline)
# 4. Surefire XML (has full stdout with all generated text)
# Do NOT grep the full surefire .txt — it contains "pathfinder-mythic" in config lines.
MYTHIC_FOUND=false
if ! $JVM_CRASHED && [ "$MAX_TOKENS" -ge 200 ] && [ -f "$SUREFIRE_OUT" ]; then
    # Method 1: Token-by-token output lines
    if grep "id=" "$SUREFIRE_OUT" | grep -qi "mythic\|hero\|character\|creating\|ability\|tier"; then
        MYTHIC_FOUND=true
    fi
    # Method 2: Generated text summary blocks
    if ! $MYTHIC_FOUND && grep -A5 -i "generated text" "$SUREFIRE_OUT" | grep -qi "mythic\|hero\|creating"; then
        MYTHIC_FOUND=true
    fi
    # Method 3: FULL_TEXT log line (optimized pipeline logs full generated text)
    if ! $MYTHIC_FOUND && grep "FULL_TEXT" "$SUREFIRE_OUT" | grep -qi "mythic\|hero\|creating\|champion\|ascension"; then
        MYTHIC_FOUND=true
    fi
    # Method 4: Surefire XML has full stdout — search for mythic content in generated text
    # Exclude lines with 'pathfinder-mythic.pdf' to avoid false positives from config lines
    if ! $MYTHIC_FOUND && [ -f "$SUREFIRE_XML" ]; then
        if grep -v "pathfinder-mythic" "$SUREFIRE_XML" | grep -qi "mythic heroes\|mythic character\|mythic champion\|creating a mythic\|moment of ascension"; then
            MYTHIC_FOUND=true
        fi
    fi
    if ! $MYTHIC_FOUND; then
        echo "  WARNING: mythic passage not found in generated tokens"
        echo "  (checked decode step lines in $SUREFIRE_OUT)"
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

CRASH_FLAG=""
if $JVM_CRASHED; then CRASH_FLAG="CRASHED"; fi

if [ -n "$REPORT" ]; then
    python3 - "$REPORT" "$MAX_TOKENS" "$CRASH_FLAG" <<'PYEOF'
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
    # Prefer lateSteady (excludes warmup/compilation), then steady, then throughput
    m = re.search(r'lateSteady=([\d.]+)\s*tok/s', benchmark_summary)
    if m: steady_tps = float(m.group(1))
    # Fall back to steady=X tok/s (includes warmup overhead)
    if steady_tps is None:
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

# Correctness — ONLY check actual generated token text, NOT config/setup output.
# Generated text appears in decode step lines like: "Step N ... 'tokenText' (id=NNNN)"
# or in the final generated text summary line.
print("")

# Detect JVM crash from build result passed as argv[3] if available
jvm_crashed = len(sys.argv) > 3 and sys.argv[3] == 'CRASHED'

# Extract ONLY the generated text from decode step lines, generated text blocks,
# and the optimized pipeline's [FULL_TEXT] proof line.
# Format: Step N ... 'TOKEN_TEXT' (id=NNN) or text='...'
import re as _re2
generated_tokens = []
generated_text_block = []
full_text_block = []
in_generated_block = False
in_full_text_block = False
for l in lines:
    # Match decode step token output: 'tokenText' (id=NNN)
    m = _re2.search(r"'(.+?)'\s+\(id=(\d+)\)", l)
    if m and 'Step' in l:
        generated_tokens.append(m.group(1))
    # Match optimized pipeline proof output:
    #   [FULL_TEXT] 250 tokens: <doctag>...
    if '[FULL_TEXT]' in l:
        in_full_text_block = True
        m_full = _re2.search(r'\[FULL_TEXT\]\s+\d+\s+tokens:\s*(.*)', l)
        if m_full and m_full.group(1).strip():
            full_text_block.append(m_full.group(1).strip())
        continue
    if in_full_text_block:
        stripped = l.strip()
        if not stripped:
            in_full_text_block = False
        elif stripped.startswith('o.') or stripped.startswith('[INFO]') or 'Token diversity:' in stripped:
            in_full_text_block = False
        else:
            full_text_block.append(stripped)
            continue
    # Match generated text summary blocks
    if 'Generated text:' in l or 'generated text:' in l:
        in_generated_block = True
        # Extract inline text if present
        m2 = _re2.search(r'[Gg]enerated text:\s*(.*)', l)
        if m2 and m2.group(1).strip():
            generated_text_block.append(m2.group(1).strip())
        continue
    if in_generated_block:
        stripped = l.strip()
        if stripped and not stripped.startswith('[') and not stripped.startswith('o.'):
            generated_text_block.append(stripped)
        elif not stripped or stripped.startswith('['):
            in_generated_block = False

# Combine all actual generated text
actual_generated = ' '.join(generated_tokens) + ' ' + ' '.join(generated_text_block) + ' ' + ' '.join(full_text_block)
actual_generated_lower = actual_generated.lower()

# Count actual generated tokens (from step lines)
token_count_actual = len(generated_tokens)

has_doctag = 'doctag' in actual_generated_lower
has_mythic_content = any(w in actual_generated_lower for w in [
    'mythic', 'hero', 'creating a mythic', 'ability', 'tier',
    'character', 'ascend', 'path'])
garbage = actual_generated.count('UserT') > 3
repeat_lt = sum(1 for t in generated_tokens if t == '<') > 10

# Token count sanity: if we asked for 250 and got < 10, something went wrong
expected_min_tokens = max(1, max_tokens // 5)  # at least 20% of requested

if jvm_crashed:
    print(f"  CORRECTNESS: CRASH — JVM killed (only {token_count_actual} tokens generated)")
elif token_count_actual == 0 and not generated_text_block:
    if full_text_block and has_mythic_content:
        print("  CORRECTNESS: PASS (FULL_TEXT mythic content confirmed)")
        print(f"    FULL_TEXT: {actual_generated.strip()[:220]}...")
    elif full_text_block:
        print("  CORRECTNESS: PARTIAL — FULL_TEXT found, but mythic keywords missing")
        print(f"    FULL_TEXT: {actual_generated.strip()[:220]}...")
    else:
        print(f"  CORRECTNESS: UNKNOWN — no generated text found in output")
        print(f"    (looked for decode step lines, generated text blocks, and [FULL_TEXT])")
elif garbage:
    print("  CORRECTNESS: FAIL — repeating garbage (UserT)")
elif repeat_lt:
    print("  CORRECTNESS: FAIL — repeating '<' tokens (stale replay)")
elif token_count_actual > 0 and token_count_actual < expected_min_tokens:
    print(f"  CORRECTNESS: FAIL — only {token_count_actual}/{max_tokens} tokens generated (early EOS or crash)")
elif has_mythic_content:
    source = "FULL_TEXT" if full_text_block else "generated tokens"
    print(f"  CORRECTNESS: PASS ({source}, mythic content confirmed)")
elif has_doctag and token_count_actual >= expected_min_tokens:
    print(f"  CORRECTNESS: PARTIAL — {token_count_actual} tokens, doctag present but no mythic keywords")
    print(f"    Generated: {actual_generated[:200]}...")
elif has_doctag:
    print(f"  CORRECTNESS: PARTIAL — doctag present, only {token_count_actual} tokens")
else:
    print(f"  CORRECTNESS: UNKNOWN — {token_count_actual} tokens, no doctag or mythic content found")
    if actual_generated.strip():
        print(f"    Generated: {actual_generated[:200]}...")

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

# Pipeline introspection (only with --diag-step/--diag-d2d/--diag-capture)
d2d_fired = sum(1 for l in lines if 'D2D:' in l and 'fired' in l)
d2d_drift = sum(1 for l in lines if 'POINTER DRIFT' in l or 'address drift' in l.lower())
stale_count = sum(1 for l in lines if 'stale output' in l.lower() or 'STALE_OUTPUT' in l)
capture_complete = any('testCaptureCompleteness PASSED' in l for l in lines)
d2d_integrity = any('testD2DCopyIntegrity PASSED' in l for l in lines)

if d2d_fired > 0 or d2d_drift > 0 or stale_count > 0 or capture_complete or d2d_integrity:
    print("")
    print("  ─── Pipeline Introspection ───────────────────────")
    if d2d_fired > 0:
        print(f"  D2D copies fired:  {d2d_fired} report(s)")
    if d2d_drift > 0:
        print(f"  Pointer drift:     {d2d_drift} detection(s) *** WARNING ***")
    if stale_count > 0:
        print(f"  Stale outputs:     {stale_count} detection(s) *** WARNING ***")
    if d2d_integrity:
        print(f"  D2D integrity:     PASS")
    if capture_complete:
        print(f"  Capture complete:  PASS")

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
    if $JVM_CRASHED; then
        echo "  CORRECTNESS: CRASH — JVM died before producing surefire report"
    else
        echo "  No surefire report found at: $SUREFIRE_OUT"
        echo "  CORRECTNESS: UNKNOWN — cannot verify (no report file)"
    fi
    echo "  Check benchmark log: $LOG_FILE"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Logs:"
echo "    Benchmark:  $LOG_FILE"
echo "    Validation: $VALIDATION_LOG"
echo "    Surefire:   $SUREFIRE_OUT"
echo "═══════════════════════════════════════════════════════════"

fi  # end of if ! $AUDIT_ONLY

# ═══════════════════════════════════════════════════════════════════════
# DSP AUDIT — runs ALL DSP test suites to catch validation issues
# ═══════════════════════════════════════════════════════════════════════
#
# Suite → test class mapping. Each suite groups related DSP tests.
# Tests are run with failsafe-style error collection: failures in one
# suite do NOT abort subsequent suites.

if ! $SKIP_AUDIT; then

AUDIT_LOG="$SCRIPT_DIR/dsp-audit.log"
AUDIT_RESULT=0
AUDIT_PASS=0
AUDIT_FAIL=0
AUDIT_SKIP=0
AUDIT_SUITES_RUN=0
AUDIT_SUITES_FAILED=0
AUDIT_FAILED_SUITES=""
AUDIT_FAILED_TESTS=""

# ─── Suite definitions ──────────────────────────────────────────────
# Each suite is: SUITE_NAME|TEST_SELECTOR
# TEST_SELECTOR is the -Dtest= value (comma-separated class names, optional #method)
declare -a AUDIT_ENTRIES=()

add_suite() {
    local suite_name="$1"
    local test_selector="$2"
    # Check if this suite is selected
    if [ "$AUDIT_SUITE" = "all" ]; then
        AUDIT_ENTRIES+=("${suite_name}|${test_selector}")
    else
        # Check if suite_name appears in the comma-separated AUDIT_SUITE
        IFS=',' read -ra SELECTED <<< "$AUDIT_SUITE"
        for s in "${SELECTED[@]}"; do
            if [ "$s" = "$suite_name" ]; then
                AUDIT_ENTRIES+=("${suite_name}|${test_selector}")
                return
            fi
        done
    fi
}

# lifecycle — DSP phase progression, lifecycle gates, slot lifecycle, shape drift
add_suite "lifecycle" \
    "DspLifecycleValidationTest,DspLifecycleExhaustiveTest,DspSlotLifecycleAuditTest,TestDspLifecycleGates,TestDspLifecycleMultiExecuteShapeDrift"

# frozen — frozen constants, phase drift, view-op replay, value-dependent shapes, op-level frozen shape
add_suite "frozen" \
    "DspFrozenConstantInvariantTest,TestFrozenPhaseDriftDetection,DspViewOpFrozenReplayTest,TestValueDependentShapeClassification,AllRegisteredOpsFrozenShapeTest,OpCategoryFrozenShapeTest"

# replay — composite replay, deep isolation, pipeline isolation, output freshness, merged segments, device analytics
add_suite "replay" \
    "DspCompositeReplayTest,DspDeepIsolationTest,DspPipelineIsolationTest,TestDspRepeatedOutputFreshness,TestDspMergedSegmentReplay,DspReplayDeviceAnalyticsTest"

# regression — harness-based regression tests: segment output, cross-stream, gap execution, arg-table, native decode
add_suite "regression" \
    "TestSegmentOutputZeroInvariant,TestCrossStreamEventOrdering,TestGapExecutionSlotInvariants,TestArgTableStablePerfFloor,TestNativeDecodeLoopRegression,TestNativeDecodeInputsRegression"

# capture — capture config matrix, DspHandle tests, handle data model, view capture correctness
add_suite "capture" \
    "TestDspCaptureConfigMatrix,DspHandleTest,DspHandleDataModelTest,TestDspViewCaptureCorrectness"

# training — end-to-end training, optimized slot-by-slot parity
add_suite "training" \
    "DspTrainingE2ETest,DspOptimizedSlotBySlotTest"

# ext-input — external input staleness, value-key segment invalidation
add_suite "ext-input" \
    "DspExtInputStalenessTest,DspValueKeySegmentTest"

# pooling — buffer pooling, batched module preload, LRU residency, compilation seal
add_suite "pooling" \
    "DynamicShapePlanPoolingTest,DspBatchedModulePreloadTest,DspLruModuleResidencyTest,DspCompilationSealTest"

# precision — mixed precision replay, FP16 constant lifecycle, weight chain NaN detection
add_suite "precision" \
    "DspMixedPrecisionReplayTest"

# pipeline — pipeline facets, shape pre-pass, config enumeration, working tree changes, decode perf floor
add_suite "pipeline" \
    "TestDspPipelineFacets,TestDspShapePrePass,TestDspConfigEnumeration,TestDspWorkingTreeChanges,TestDspDecodePerfFloor"

# device — multi-device memory leak simulation, multi-device analytics
add_suite "device" \
    "DspMultiDeviceMemoryLeakSimulationTest"

# openvino — OpenVINO accuracy tests
add_suite "openvino" \
    "DspOpenVinoAccuracyTest"

# validation — SmolDocling DSP accuracy validation (ALL test methods)
add_suite "validation" \
    "TestDspValidation"

if [ ${#AUDIT_ENTRIES[@]} -eq 0 ]; then
    echo ""
    echo "WARNING: No audit suites matched '$AUDIT_SUITE'"
    echo "Available suites: lifecycle, frozen, replay, regression, capture, training, ext-input, pooling, validation, all"
else

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  DSP AUDIT"
echo "  Suites:  ${#AUDIT_ENTRIES[@]}"
echo "  Timeout: ${AUDIT_TIMEOUT}s per suite"
echo "  Backend: $BACKEND_ARTIFACT"
echo "  Log:     $AUDIT_LOG"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Truncate the audit log
> "$AUDIT_LOG"

for entry in "${AUDIT_ENTRIES[@]}"; do
    SUITE_NAME="${entry%%|*}"
    TEST_SELECTOR="${entry#*|}"
    AUDIT_SUITES_RUN=$((AUDIT_SUITES_RUN + 1))

    echo "─── [$AUDIT_SUITES_RUN/${#AUDIT_ENTRIES[@]}] Suite: $SUITE_NAME ───"
    echo "    Tests: $TEST_SELECTOR"

    SUITE_LOG="$SCRIPT_DIR/dsp-audit-${SUITE_NAME}.log"

    set +e
    timeout "${AUDIT_TIMEOUT}s" \
    $MVN test \
      -Dtest="$TEST_SELECTOR" \
      $TRITON_FLAG \
      -Dbackend.artifactId=$BACKEND_ARTIFACT \
      -Dnd4j.dsp.diagnostics=EXECUTE,SEGMENT,FALLBACK \
      -Dnd4j.dsp.diagnostics.level=summary \
      2>&1 | tee "$SUITE_LOG"
    SUITE_RESULT=${PIPESTATUS[0]}
    set -e

    # Parse results from the tee log
    SUITE_TESTS_RUN=$(grep -c 'Tests run:' "$SUITE_LOG" 2>/dev/null || echo 0)
    SUITE_PASS=$(grep -oP 'Tests run: \d+, Failures: \d+, Errors: \d+, Skipped: \d+' "$SUITE_LOG" | tail -1 || echo "")

    if [ $SUITE_RESULT -eq 0 ]; then
        echo "    Result: PASS"
        AUDIT_PASS=$((AUDIT_PASS + 1))
    elif [ $SUITE_RESULT -eq 124 ]; then
        echo "    Result: TIMEOUT (>${AUDIT_TIMEOUT}s)"
        AUDIT_FAIL=$((AUDIT_FAIL + 1))
        AUDIT_SUITES_FAILED=$((AUDIT_SUITES_FAILED + 1))
        AUDIT_FAILED_SUITES="${AUDIT_FAILED_SUITES}  [TIMEOUT] ${SUITE_NAME}\n"
    else
        echo "    Result: FAIL (exit $SUITE_RESULT)"
        AUDIT_FAIL=$((AUDIT_FAIL + 1))
        AUDIT_SUITES_FAILED=$((AUDIT_SUITES_FAILED + 1))
        AUDIT_FAILED_SUITES="${AUDIT_FAILED_SUITES}  [FAIL]    ${SUITE_NAME}\n"
        # Extract failed test names from the log
        FAILED_NAMES=$(grep -oP '(?<=FAILED: )\S+|(?<=<<< FAILURE!)\s*\S+|(?<=<<< ERROR!)\s*\S+' "$SUITE_LOG" 2>/dev/null | head -10 || true)
        if [ -n "$FAILED_NAMES" ]; then
            AUDIT_FAILED_TESTS="${AUDIT_FAILED_TESTS}  ${SUITE_NAME}:\n"
            while IFS= read -r fname; do
                AUDIT_FAILED_TESTS="${AUDIT_FAILED_TESTS}    - ${fname}\n"
            done <<< "$FAILED_NAMES"
        fi
    fi

    if [ -n "$SUITE_PASS" ]; then
        echo "    $SUITE_PASS"
    fi

    # Append to combined audit log
    echo "═══ SUITE: $SUITE_NAME (exit=$SUITE_RESULT) ═══" >> "$AUDIT_LOG"
    cat "$SUITE_LOG" >> "$AUDIT_LOG"
    echo "" >> "$AUDIT_LOG"
    echo ""
done

# ─── Audit summary ──────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  DSP AUDIT SUMMARY"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Suites run:    $AUDIT_SUITES_RUN"
echo "  Suites passed: $AUDIT_PASS"
echo "  Suites failed: $AUDIT_SUITES_FAILED"
echo ""

if [ $AUDIT_SUITES_FAILED -gt 0 ]; then
    echo "  ── Failed Suites ──────────────────────────────────────"
    echo -e "$AUDIT_FAILED_SUITES"
    if [ -n "$AUDIT_FAILED_TESTS" ]; then
        echo "  ── Failed Tests ───────────────────────────────────────"
        echo -e "$AUDIT_FAILED_TESTS"
    fi
    echo "  To debug a specific suite:"
    echo "    ./run-benchmark.sh --audit-only --audit-suite <SUITE>"
    echo ""
    echo "  Per-suite logs:"
    for entry in "${AUDIT_ENTRIES[@]}"; do
        SUITE_NAME="${entry%%|*}"
        echo "    $SUITE_NAME: $SCRIPT_DIR/dsp-audit-${SUITE_NAME}.log"
    done
    AUDIT_RESULT=1
else
    echo "  ALL SUITES PASSED"
fi
echo ""
echo "  Combined audit log: $AUDIT_LOG"
echo "═══════════════════════════════════════════════════════════"

fi  # end of AUDIT_ENTRIES check
fi  # end of ! SKIP_AUDIT

# ─── Final exit ──────────────────────────────────────────────────────
# Non-zero if either benchmark OR audit failed
if [ $BUILD_RESULT -ne 0 ]; then
    exit $BUILD_RESULT
elif [ "${AUDIT_RESULT:-0}" -ne 0 ]; then
    exit 1
fi
exit 0
