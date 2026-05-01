---
name: cpu-dsp-testing-two-benchmarks
description: "CPU DSP has 2 benchmarks: TestLLMBenchmarkSuite (Qwen 0.8B, run FIRST) and run-benchmark.sh (SmolDocling VLM, run SECOND). SLOT_BY_SLOT removed — only OPTIMAL/AUTO."
type: project
---

# CPU DSP Testing: Two Benchmarks

**Updated**: 2026-04-27 — ALL SLOT_BY_SLOT references removed. Only OPTIMAL (AUTO mode) matters.

## Benchmark 1: TestLLMBenchmarkSuite (RUN FIRST — smaller, faster)

**Model**: Qwen3.5-0.8B (Q4_K_M GGUF), smallest dense model
**Test class**: `org.eclipse.deeplearning4j.llm.benchmark.TestLLMBenchmarkSuite`
**Script**: `platform-tests/run-llm-benchmarks.sh`
**Default tokens**: 20

### Run commands

```bash
# Quick Qwen-only baseline (OPTIMAL = AUTO mode)
cd platform-tests && ./run-llm-benchmarks.sh --backend cpu --test baseline --models qwen --tokens 20

# All tests
cd platform-tests && ./run-llm-benchmarks.sh --backend cpu --test all --models qwen

# Direct mvn
cd platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test \
  -Dtest=TestLLMBenchmarkSuite#testOptimalBaseline \
  -Dbench.models=qwen -Dbench.max.tokens=20 \
  -Dbackend.artifactId=nd4j-native \
  -Dnd4j.optimizer.enabled=true \
  2>&1 | tee /tmp/llm-cpu-bench.log
```

### Available tests (updated method names)
- `import` → testModelImportBenchmark
- `baseline` → testOptimalBaseline (was testSlotBySlotBaseline — RENAMED)
- `cuda-graphs` → testCudaGraphsBenchmark (CUDA only)
- `triton` → testTritonCompileBenchmark (CUDA only)
- `fusion` → testKernelFusionImpact
- `optimizer` → testGraphOptimizerImpact
- `matrix` → testFullConfigMatrix
- `perplexity` → testPerplexityComparison
- `quant` → testQuantizationComparison
- `prompts` → testReferencePromptAccuracy
- `device` → testDeviceSpecificBenchmark

### SLOT_BY_SLOT is REMOVED
- All benchmark configs use `GraphExecutionMode.AUTO` (OPTIMAL)
- Only exception: device-specific test has SLOT_BY_SLOT as intentional comparison baseline
- `getBaselineConfigs()` returns single "OPTIMAL" config with AUTO mode
- run-llm-benchmarks.sh `TEST_MAP[baseline]="testOptimalBaseline"`

### GGUF model execution path
- GGUF models have NO KV cache outputs → `generateSimpleNoKvCache()` (concat path)
- DSP auto-compile is disabled for this path (shapes change every step, plan reuse impossible)
- Without this fix, each token triggers plan recompilation + OpenVINO compilation (~10-30s overhead)

---

## Benchmark 2: SmolDocling VLM (RUN SECOND — bigger, slower)

**Model**: SmolDocling-256M (30-layer VLM decoder), PDF page decode
**Script**: `platform-tests/run-benchmark.sh` (or `run-benchmark-cpu.sh`)
**Default tokens**: 250
**CPU baseline**: 3 tok/s (as of 2026-04-27)

---

## Order of Operations

**Why:** TestLLMBenchmarkSuite uses Qwen 0.8B which is much smaller and faster to iterate on.
**How to apply:** Always run TestLLMBenchmarkSuite first when testing CPU DSP changes. Only proceed to SmolDocling after the smaller benchmark shows expected results.
