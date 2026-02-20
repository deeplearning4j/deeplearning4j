# VLM Decode Performance Optimization

## Current Baseline (SmolDocling, RTX 4090 24GB, batch=2 pages)

| Metric | Value |
|--------|-------|
| Vision encoding | 32,352ms (20 frames, 1,617ms/frame) |
| Decode (512 steps) | 344,705ms |
| ms/token (batch amortized) | 673ms |
| Throughput | 3.0 tok/sec |
| Decoder ops per step | 4,441 slots |
| Outputs per step | 61 (1 logits + 60 KV cache tensors) |
| Model | 30-layer LLaMA-style, FP32, ~250MB |

## Architecture Overview

Each decode step:
1. Java builds inputs (embeddings, attention_mask, causal_mask, 60 past_key_values)
2. Java calls `DynamicShapePlanExecutor.executeNative()` (single JNI call)
3. C++ `NativeDynamicShapePlan` executes 4,441 ops slot-by-slot
4. C++ copies 61 output NDArrays back to Java (`copyBuffer` per output)
5. Java extracts logits + 60 present KV tensors, stores for next step
6. Java does argmax, token decode, builds next step inputs

## Bottleneck Analysis

### 1. Native Executor KV Cache Round-Trip (~530-555ms/step)

**The single biggest bottleneck.** Each step:
- 60 KV tensors copied C++ → Java via `copyBuffer` (creates new Java INDArrays)
- 60 KV tensors passed Java → C++ as external inputs next step (different GPU addresses)
- C++ plan sees new addresses each step → CUDA graph capture permanently fails
- Shape caches cleared every step (`clearDynamicShapePlanCaches`) forcing re-inference

**The native executor produces wrong output** (`<doctag>` repeated) because KV cache
round-trip through Java doesn't preserve tensor state correctly. The Java DSP executor
works correctly but is slower due to per-op JNI overhead.

**Optimization: Keep KV cache in C++.** Instead of copying 60 KV tensors back to Java
and re-passing them, let the C++ plan manage KV cache internally:
- Map `present.N.key` output slots directly to `past_key_values.N.key` input slots
- Only copy logits back to Java (1 output instead of 61)
- KV tensors stay at fixed GPU addresses → enables CUDA graph capture
- Eliminates `clearDynamicShapePlanCaches` overhead (shapes don't change for KV)

**Files:**
- `DynamicShapePlanExecutor.java:3584` - `executeNative()` copies all outputs
- `NativeDynamicShapePlan.cpp:1020` - slot input resolution from external arrays
- `NativeDynamicShapePlan.h:249` - `clearShapeCaches()` called every step
- `TestVLMModelImportPipeline.java:849` - all 61 outputs requested

### 2. CUDA Graph Capture Fails (~10-15% overhead from kernel launch)

CUDA graphs record a sequence of GPU operations and replay them with a single launch,
eliminating per-kernel launch overhead (~5-15μs × 4,441 ops = ~22-66ms).

**Current failures:**
- Segment [549-563]: "cudaMemsetAsync failed! Error code: [901]" during capture
- Input address changes (KV cache buffers reallocated) → permanent fallback
- Shape changes each step (seqLen grows) → warmup pass without capture

**Fix depends on #1.** Once KV cache stays in C++ at fixed addresses:
- Input addresses stable → graphs can be captured
- Only `position_ids` and `attention_mask` change (can use graph update API)
- The 4,441-op plan has ~61 capturable segments (fusion candidates already detected)

**Files:**
- `NativeDynamicShapePlan.cpp:674` - `executeSegmentWithGraph()`
- `CudaGraphHandle.cu` - CUDA graph capture/replay wrapper
- `CudaGraphScheduler.h` - segment-level graph management

### 3. N-gram Speculative Decoding Never Activates (potential 2-4x speedup)

The n-gram speculator (`NgramSpeculator` + `SpeculativeDecodeLoop`) is already
implemented and wired into the decode loop. It generates K candidate tokens from
n-gram pattern matching on previously generated DocTag tokens, then validates all K
in a single multi-token forward pass.

**Why it shows 0 speculative attempts:**

The gate at `TestVLMModelImportPipeline.java:3889`:
```java
if (activeBatchSize == 1 && step > 10) {
    // speculative decode only for single-sequence
```

With 2 pages, `activeBatchSize` stays at 2 for the entire run (neither page hits EOS
before `maxTokens`), so speculation **never triggers**. Additionally, when it does
trigger (single page), the model may fail multi-token KV-cache decode (ONNX Expand
broadcast failure), causing `SpeculativeDecodeLoop` to permanently self-disable.

**Fix options:**
- Enable speculation for batch > 1: run speculative multi-token decode per-sequence
  with independent KV caches, then merge accepted tokens back into the batch
- Fix the ONNX Expand broadcast failure for multi-token decode (likely attention mask
  shape mismatch when `currentSeqLen > 1`)
- Lower the gate: remove the `activeBatchSize == 1` check and handle batch speculation
- For structured DocTag output, n-gram hit rates should be high (repeated `<loc_`,
  `<text>`, `</text>` patterns) — potential 3-5 tokens accepted per speculation

**Files:**
- `TestVLMModelImportPipeline.java:3889` - `activeBatchSize == 1` gate
- `SpeculativeDecodeLoop.java:184` - permanent disable on model failure
- `NgramSpeculator.java` - n-gram pattern matching logic
- `DecoderUtils.java:72` - `buildCausalMask` for multi-token decode (currentSeqLen > 1)

### 4. FP32 → FP16/BF16 (potential 2x speedup)

All computation is FP32. The RTX 4090 has:
- FP32: 82.6 TFLOPS
- FP16 (with tensor cores): 330.3 TFLOPS (4x)
- BF16 (with tensor cores): 330.3 TFLOPS (4x)

**Approach:**
- Cast model weights to FP16 at import time (or load FP16 ONNX model)
- Run attention in FP16, accumulate in FP32 (mixed precision)
- KV cache in FP16 (halves memory bandwidth for 60 tensors per step)
- Keep logits in FP32 for numerical stability

**Files:**
- `OnnxFrameworkImporter.kt` - model import (add dtype conversion)
- `SDZSerializer.java` - model serialization (preserve FP16)
- `NativeDynamicShapePlan.cpp` - execution (dtype-aware allocation)

### 5. Shape Cache Optimization (minor, ~1-5ms/step)

Currently `clearDynamicShapePlanCaches()` invalidates ALL 4,441 slots every step.
Most slots have shapes that DON'T change between steps (weight multiplies, layer norms).
Only ~30 slots change (attention mask, KV cache concat, position-dependent ops).

**Fix:** Mark slots as "shape-static" vs "shape-dynamic" at compile time.
Only clear dynamic slots each step. Static slots keep their cached shapes.

**Files:**
- `NativeDynamicShapePlan.h:104` - `shapeCacheValid` per slot
- `NativeDynamicShapePlan.cpp:1430` - `clearShapeCaches()` clears everything
- `DynamicShapePlanExecutor.java:3696` - calls clear before every execute

### 6. Vision Encoding (~32s for 20 frames)

Currently ~1,617ms/frame. The vision encoder (SigLIP) processes frames one-at-a-time
because the ONNX model only supports batch_size=1.

**Optimizations:**
- Export vision encoder with dynamic batch dimension
- Process 2-4 frames per forward pass
- Use FP16 for vision encoder (image data is inherently low-precision)
- Pipeline more aggressively: start decode while later pages still encoding

## Priority Order

| Priority | Optimization | Expected Speedup | Effort |
|----------|-------------|-------------------|--------|
| 1 | Fix native KV cache (keep in C++) | 1.5-2x | High |
| 2 | Enable CUDA graphs (depends on #1) | 1.1-1.3x | Medium |
| 3 | FP16 mixed precision | 1.5-2x | High |
| 4 | Fix speculative decoding | 2-4x | Medium |
| 5 | Selective shape cache clearing | 1.02-1.05x | Low |
| 6 | Batch vision encoding | 1.3-1.5x (vision only) | Medium |

**Combined theoretical ceiling: 6-20x → 6-20 tok/sec → 45-150ms/token**

## How to Measure

Enable timing breakdown:
```bash
-Dnd4j.inference.timing=true
```

Enable DSP diagnostic output:
```bash
-Dnd4j.dsp.native.dumpOutputs=true
-Dnd4j.dsp.java.dumpOutputs=true
```

Run test:
```bash
cd platform-tests && mvn test \
  -Dtest=TestVLMModelImportPipeline#testOptimizedPipeline \
  -Dvlm.test.pdf.path=pathfinder-mythic.pdf \
  -Dvlm.test.pdf.startPage=10 \
  -Dvlm.test.pdf.maxPages=2 \
  -Dvlm.test.maxTokens=512 \
  -Dsurefire.timeout=7200 -DforkedProcessTimeoutInSeconds=7200
```

Monitor GPU utilization during decode:
```bash
watch -n 0.5 nvidia-smi
```

Target: **10+ tok/sec** (sub-100ms/token batch-amortized)
