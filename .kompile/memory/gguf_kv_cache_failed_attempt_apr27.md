---
name: gguf-kv-cache-failed-attempt-apr27
description: Failed attempt to add KV cache to GGUF LLaMAArchitecture — detailed analysis of what was tried, what broke, and the unsolved design problems
type: project
---

# GGUF KV Cache Implementation: Failed Attempt (2026-04-27)

## Goal
Add KV cache support to GGUF-imported models so they use `generateSimpleWithKvCache()` instead of the O(n²) concat path. This would let DSP compile once after prefill and reuse the plan for all decode steps.

## What Was Tried (all reverted)

### 1. LLaMAArchitecture.java — Added KV cache graph structure
- Added `position_ids` placeholder (INT64 `[-1,-1]`)
- Added `_causal_mask` placeholder (FLOAT `[-1,-1,-1,-1]`) for additive attention bias
- Added `past_key_values.{layer}.key/value` placeholders (dtype `[-1,-1,-1,-1]`) for each non-GDN layer
- Registered `present.{layer}.key/value` as graph outputs via `sd.setOutputs()`
- In `buildSeparateQKVAttention()` and `buildGatedAttention()`:
  - Changed FusedRoPE to use dynamic position via `positionIds` SDVariable input
  - Permuted past KV from ONNX `[B,H,S,D]` to internal `[B,S,H,D]`
  - Concat past KV + new K/V along seq dim (axis=1)
  - Permuted back to `[B,H,S,D]` for present output
  - Wired `_causal_mask` as attention bias into `dotProductAttentionV2`

### 2. FusedRoPE.java — Dynamic position offset constructor
- Added constructor taking `SDVariable positionOffsetVar` as second input
- C++ side (`fused_llm_ops.cpp`): when `block.width() == 2`, reads input[1] as scalar INT tensor for position

### 3. KVCacheUpdate.java — Dynamic position tensor constructor
- Added constructor taking `SDVariable positionTensor` as input[4]
- C++ side (`llm_ops.cpp`): when `block.width() > 4`, reads position from input tensor

### 4. OpenVinoGraphBackend.cpp — Zero-length output buffer fix
- Added dummy buffer for zero-length output arrays (same pattern as zero-length inputs)
- Needed because initial KV cache `[B,H,0,D]` → permute → NULL output buffer

### 5. DecoderInputBuilder.java — Mask fix for in-graph concat
- Attempted to unmask last position (`totalSeqLen-1`) in padded causal mask for GGUF concat

### 6. GenerationPipeline.java — Shape freezing after first decode step
- Added `executor.setShapesFrozen(true)` after step 1 in `generateSimpleWithKvCache()`

## What Broke and Why

### Problem 1: OpenVINO NULL buffer crash (fixable)
- Initial KV `[1,H,0,D]` → permute in graph → output has zero elements → NULL buffer
- OpenVINO `executeSegment` returns KERNEL_FAILURE at output tensor binding
- **Diagnostic**: `/tmp/llm-cpu-qwen-kv-diag.log` line 4546: "output slot 75 has NULL buffer"
- **Fix was straightforward**: dummy buffer for zero-length outputs, same as input side

### Problem 2: Shapes still change every step with concat (fundamental)
- Even with KV cache, in-graph `concat(pastKV, newKV)` produces output that grows by 1 each step
- WITHOUT padded static KV: past shape changes `[B,H,cachePos,D]` → different every step → recompile
- WITH padded static KV (`usePadded=true`): past shape is FIXED `[B,H,maxKvLen,D]` → concat produces FIXED `[B,maxKvLen+1,H,D]` — this WOULD give stable shapes

### Problem 3: Attention masking with padded concat (fundamental, unsolved)
- With padded KV + in-graph concat: concat result is `[B, maxKvLen+1, H, D]`
  - Positions 0..cachePos-1: valid past KV
  - Positions cachePos..maxKvLen-1: ZERO PADDING
  - Position maxKvLen: current step's new K/V (appended by concat)
- The causal mask `[1,1,1,maxKvLen+1]` unmasks positions 0..cachePos, masks cachePos+1..maxKvLen
- **BUG**: Position maxKvLen (current token's K/V) is MASKED because it's after the padding region
- Unmasking the last position breaks ONNX models that share the same mask builder
- Without proper masking, zero-padded positions corrupt softmax normalization

### Problem 4: Chicken-and-egg with static KV (fundamental, unsolved)
- For attention to work, Q must see current step's K/V
- Options explored:
  - **In-graph concat**: shapes change OR padding mask problem (above)
  - **kv_cache_update scatter**: needs dynamic position (added but) empty initial cache `[B,0,H,D]` can't be scattered into
  - **Java pre-scatter**: can't scatter K/V that hasn't been computed yet
  - **Separate prefill/decode paths**: SameDiff graphs are static, can't have conditional paths

## Key Architectural Insights

### How ONNX models avoid this problem
ONNX models export `present_key_values` outputs natively. The ONNX graph internally handles K/V management. The Java side just manages the static buffer + mask. There's no in-graph concat added by us.

### How production inference engines solve this
- **TensorRT-LLM / vLLM**: Use paged attention kernels that operate on pre-allocated KV blocks. The kernel internally handles the position-based indexing. No graph-level concat.
- **ONNX Runtime**: The model's own ops handle KV concatenation. Shapes change but ORT has optimized execution without full recompilation.
- **llama.cpp**: Direct C++ implementation with manual KV cache management, no graph abstraction.

### The fundamental tension
SameDiff's DSP framework optimizes by compiling shape-specific execution plans. KV cache decode fundamentally involves growing sequences. These are incompatible UNLESS:
1. The graph uses fixed-size buffers with position-based scatter (requires a new attention op or in-graph scatter)
2. Or the graph separates K/V computation from attention (requires Java to manage the full KV and feed it back)

## Recommended Next Approach (NOT yet attempted)

**Option A — Fixed-size KV buffer with in-graph scatter**:
- Graph accepts `[B,H,maxKvLen,D]` static buffer as input
- Graph computes new K/V, uses `kv_cache_update` to scatter at position
- Attention uses the updated buffer with explicit mask
- Present output = the updated buffer (same shape always)
- Requires: (1) dynamic position for kv_cache_update, (2) handling prefill where initial buffer is all zeros, (3) attention mask that masks beyond current position
- Challenge: prefill doesn't fit this pattern (seq_len > 1 into fixed buffer)

**Option B — Two-phase execution**:
- Phase 1 (prefill): Use current concat graph, let DSP compile for prefill shapes
- Phase 2 (decode): Switch to a DIFFERENT graph that takes full KV as input (no concat), computes only Q + attention
- Requires building two separate SameDiff graphs for one model

**Option C — Accept recompilation, optimize compile time**:
- Keep in-graph concat (shapes change each step)
- Focus on making OpenVINO/oneDNN compilation fast (sub-100ms) or caching better
- OpenVINO topology cache already helps across layers, could be extended

## Files Involved
- `nd4j/nd4j-ggml/src/main/java/org/nd4j/ggml/architecture/LLaMAArchitecture.java` — graph builder
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/GenerationPipeline.java` — decode loop
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/ModelIOConfig.java` — KV cache discovery, createEmptyKvCache (`[B,H,0,D]`)
- `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/DecoderInputBuilder.java` — mask/KV building
- `libnd4j/include/graph/cpu/OpenVinoGraphBackend.cpp` — zero-length buffer issue
- `libnd4j/include/ops/declarable/generic/nn/fused_llm_ops.cpp` — FusedRoPE dynamic position
- `libnd4j/include/ops/declarable/generic/nn/llm_ops.cpp` — kv_cache_update dynamic position

## Current State (2026-04-27)
- ALL changes reverted — LLaMAArchitecture.java is back to original (no KV cache support)
- GGUF models still use `generateSimpleNoKvCache()` (concat path, DSP disabled)
- CPU benchmark: ~0.03 tok/s with DSP enabled (recompilation per step), ~0.03 tok/s with DSP disabled (raw SameDiff)
- The baseline is terrible either way because the model is fundamentally O(n²) without KV cache
