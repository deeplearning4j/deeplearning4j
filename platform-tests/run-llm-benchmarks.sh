#!/bin/bash
# LLM Multi-Model Benchmark Suite
# Runs TestLLMBenchmarkSuite across model families and execution modes.
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ Benchmarks: GGUF import, SLOT_BY_SLOT, CUDA_GRAPHS, Triton,       │
# │             kernel fusion, graph optimizer, perplexity, quants     │
# │                                                                     │
# │ Default models: qwen (0.8B), gemma (1B)                            │
# │ Also available: phi, mistral, lfm2-extract (350M)                  │
# │ Default backend: CUDA                                              │
# │ Default tokens: 20                                                  │
# └─────────────────────────────────────────────────────────────────────┘
#
# Usage: ./run-llm-benchmarks.sh [OPTIONS]
#
# Test selection:
#   --test TEST           Run specific benchmark test (default: all)
#                         Valid: import, baseline, cuda-graphs, triton, fusion,
#                                optimizer, matrix, perplexity, quant, prompts, device
#   --test all            Run all benchmark tests
#
# Model selection:
#   --models MODELS       Comma-separated model keys (default: qwen,gemma)
#                         Valid: qwen, gemma, phi, mistral, lfm2-extract, all
#
# Configuration:
#   --tokens N            Max decode tokens (default: 20)
#   --config CONFIGS      Comma-separated config filter (supports * wildcard)
#   --quant TYPE          Quantization type (default: Q4_K_M)
#   --prompt "TEXT"       Custom test prompt
#
# Backend:
#   --backend cuda|cpu    Select backend (default: cuda)
#
# Options:
#   --debug               Enable DSP diagnostics at FULL level
#   --op-timing           Enable native op timing
#   --skip-generation     Run import benchmarks only (no tokenizer needed)
#   --verbose             Verbose maven output
#
# Examples:
#   ./run-llm-benchmarks.sh                                # All benchmarks, default models
#   ./run-llm-benchmarks.sh --test baseline --models qwen  # Quick Qwen baseline
#   ./run-llm-benchmarks.sh --test import --models all     # Import timing for all models
#   ./run-llm-benchmarks.sh --test triton --tokens 50      # Triton with 50 tokens
#   ./run-llm-benchmarks.sh --backend cpu --test baseline  # CPU-only baseline
#   ./run-llm-benchmarks.sh --test matrix --models qwen,phi --tokens 30  # Full matrix
#   ./run-llm-benchmarks.sh --test prompts --models lfm2-extract        # LFM2 extraction accuracy
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MVN="/home/agibsonccc/dev-apps/mvn/bin/mvn"
TEST_CLASS="TestLLMBenchmarkSuite"
TEST_PKG="org.eclipse.deeplearning4j.llm.benchmark"
LOG_FILE="$SCRIPT_DIR/llm-bench-output.log"
SUREFIRE_OUT="$SCRIPT_DIR/target/surefire-reports/${TEST_PKG}.${TEST_CLASS}-output.txt"
SUREFIRE_XML="$SCRIPT_DIR/target/surefire-reports/TEST-${TEST_PKG}.${TEST_CLASS}.xml"

# Defaults
BACKEND="cuda"
TEST_NAME="all"
MODELS=""
MAX_TOKENS=20
CONFIGS=""
QUANT=""
PROMPT=""
DEBUG_MODE=false
OP_TIMING=false
SKIP_GENERATION=false
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
        --config)
            CONFIGS="$2"
            shift 2
            ;;
        --quant)
            QUANT="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
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
        --skip-generation)
            SKIP_GENERATION=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            head -45 "$0" | tail -n +2 | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# ─── Map test name to JUnit method ──────────────────────────────────

declare -A TEST_MAP
TEST_MAP[import]="testModelImportBenchmark"
TEST_MAP[baseline]="testOptimalBaseline"
TEST_MAP[cuda-graphs]="testCudaGraphsBenchmark"
TEST_MAP[triton]="testTritonCompileBenchmark"
TEST_MAP[fusion]="testKernelFusionImpact"
TEST_MAP[optimizer]="testGraphOptimizerImpact"
TEST_MAP[matrix]="testFullConfigMatrix"
TEST_MAP[perplexity]="testPerplexityComparison"
TEST_MAP[quant]="testQuantizationComparison"
TEST_MAP[prompts]="testReferencePromptAccuracy"
TEST_MAP[device]="testDeviceSpecificBenchmark"

if [ "$TEST_NAME" = "all" ]; then
    TEST_SELECTOR="$TEST_CLASS"
else
    METHOD="${TEST_MAP[$TEST_NAME]:-}"
    if [ -z "$METHOD" ]; then
        echo "ERROR: Unknown test '$TEST_NAME'"
        echo "Valid tests: ${!TEST_MAP[*]} all"
        exit 1
    fi
    TEST_SELECTOR="${TEST_CLASS}#${METHOD}"
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

# Model selection
if [ -n "$MODELS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dbench.models=$MODELS"
    echo "[models] $MODELS"
else
    echo "[models] default (qwen, gemma)"
fi

# Config filter
if [ -n "$CONFIGS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dbench.configs=$CONFIGS"
    echo "[configs] $CONFIGS"
fi

# Quantization
if [ -n "$QUANT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dbench.quant=$QUANT"
    echo "[quant] $QUANT"
fi

# Prompt
if [ -n "$PROMPT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dbench.prompt=$PROMPT"
fi

# Skip generation
if [ "$SKIP_GENERATION" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS -Dbench.skip.generation=true"
    echo "[mode] import-only (generation skipped)"
fi

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
echo "  LLM BENCHMARK SUITE"
echo "═══════════════════════════════════════════════════════════"
echo "  Test:     $TEST_NAME ($TEST_SELECTOR)"
echo "  Backend:  $BACKEND ($BACKEND_ARTIFACT)"
echo "  Tokens:   $MAX_TOKENS"
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
  -Dnd4j.optimizer.enabled=true \
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
fi

# ─── Parse results from log ──────────────────────────────────────────

REPORT=""
if [ -f "$SUREFIRE_XML" ]; then
    REPORT="$SUREFIRE_XML"
elif [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
    REPORT="$LOG_FILE"
fi

if [ -n "$REPORT" ]; then
    python3 - "$REPORT" "$TEST_NAME" <<'PYEOF'
import sys, re

report_path = sys.argv[1]
test_name = sys.argv[2]

# Read all lines
if report_path.endswith('.xml'):
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(report_path)
        lines = []
        for tc in tree.findall('.//testcase'):
            for tag in ['system-out', 'system-err']:
                el = tc.find(tag)
                if el is not None and el.text:
                    lines.extend(el.text.split('\n'))
    except Exception:
        with open(report_path) as f:
            lines = f.read().split('\n')
else:
    with open(report_path) as f:
        lines = f.read().split('\n')

# ── Extract benchmark results ──
pass_results = []
fail_results = []
import_results = []

for l in lines:
    stripped = l.strip()
    if '[PASS]' in stripped:
        if 'tok/s' in stripped:
            pass_results.append(stripped)
        elif 'import=' in stripped:
            import_results.append(stripped)
    elif '[FAIL]' in stripped:
        fail_results.append(stripped)

# ── Print summary ──
print("")
if import_results:
    print("  ┌─── Import Results ───────────────────────────────┐")
    for r in import_results:
        # Clean up log prefix
        m = re.search(r'\[PASS\].*', r)
        if m:
            print(f"  │  {m.group()}")
    print("  └─────────────────────────────────────────────────┘")
    print("")

if pass_results:
    print("  ┌─── Generation Results ──────────────────────────┐")
    # Extract tok/s values for summary
    all_tps = []
    for r in pass_results:
        m = re.search(r'\[PASS\].*', r)
        if m:
            print(f"  │  {m.group()}")
        tps = re.search(r'([\d.]+)\s*tok/s', r)
        if tps:
            all_tps.append(float(tps.group(1)))
    print("  └─────────────────────────────────────────────────┘")
    print("")

    if all_tps:
        print("  ┌─── Throughput Summary ──────────────────────────┐")
        print(f"  │  Min:  {min(all_tps):>8.2f} tok/s                       │")
        print(f"  │  Avg:  {sum(all_tps)/len(all_tps):>8.2f} tok/s                       │")
        print(f"  │  Max:  {max(all_tps):>8.2f} tok/s                       │")
        print(f"  │  Runs: {len(all_tps):>4d}                                   │")
        print("  └─────────────────────────────────────────────────┘")
        print("")

if fail_results:
    print("  ┌─── Failures ─────────────────────────────────────┐")
    for r in fail_results:
        m = re.search(r'\[FAIL\].*', r)
        if m:
            line = m.group()
            # Truncate long failure messages
            if len(line) > 100:
                line = line[:97] + "..."
            print(f"  │  {line}")
    print("  └─────────────────────────────────────────────────┘")
    print("")

# Final status
total = len(pass_results) + len(import_results) + len(fail_results)
passed = len(pass_results) + len(import_results)
print(f"  Total: {total} benchmarks, {passed} passed, {len(fail_results)} failed")
PYEOF
fi

echo ""
echo "  Log: $LOG_FILE"
echo "═══════════════════════════════════════════════════════════"

exit $BUILD_RESULT
