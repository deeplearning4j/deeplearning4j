---
name: cpu-dsp-concat-path-leak
description: GGUF models use concat-based decode (no KV cache outputs) — DSP plan recompilation every token causes growing exec time. Fixed by disabling DSP auto-compile for this path.
type: project
---

# CPU DSP Concat-Path Leak: Plan Recompilation Every Token (fixed 2026-04-27)

**Symptom**: Execution time grows linearly per token (28s→40s over 20 tokens). 20 plan evictions for 20 tokens. `shapesFrozen=false` throughout. `contentHash` different each step.

**Root cause chain**:
1. GGUF-imported models (Qwen, Gemma, Phi, etc.) have NO KV cache outputs — `findKVCacheOutputNames()` returns empty lists because GGUF architectures only output `logits` (no `present.*.key`/`present.*.value` variables)
2. `GenerationPipeline.generateSimple()` routes to `generateSimpleNoKvCache()` (concat path)
3. Concat path grows `input_ids` each step: `[prompt]` → `[prompt, tok1]` → `[prompt, tok1, tok2]`...
4. Different shapes every step → DSP `redispatchForCurrentShapes` creates new plan each token
5. Plan cache max 2 for CPU (`planCacheMaxPlansCpu`) → evicts previous plan immediately
6. Each new plan triggers OpenVINO + oneDNN compilation → ~10-30s overhead per step
7. Net effect: DSP compilation adds massive overhead with ZERO reuse benefit

**Fix** (`GenerationPipeline.java:generateSimpleNoKvCache()`):
- Save `dspAutoCompileEnabled` and `dspNativeAutoCompileEnabled` state
- Disable both before decode loop (shapes change every step → plan reuse impossible)
- Restore in `finally` block
- This is NOT a fallback to slot-by-slot — it's correct: don't compile plans that can never be reused

**Why:** DSP plan compilation (including OpenVINO model compilation at ~50-200MB each) is only valuable when shapes stabilize and plans are reused across steps. The concat-based decode path has shapes that NEVER stabilize, making every compilation wasted work.

**How to apply:**
- GGUF models without KV cache outputs → `generateSimpleNoKvCache()` → DSP disabled (correct)
- GGUF models WITH KV cache outputs (future) → `generateSimpleWithKvCache()` → DSP enabled + shapes frozen after step 1 (correct)
- ONNX models with explicit present_key/present_value outputs → `generateSimpleWithKvCache()` → DSP enabled (correct)
- The long-term fix for GGUF models is to add KV cache output support to the GGUF architecture builders (LLaMAArchitecture, etc.) so they can use the static-KV path

**Files**: `GenerationPipeline.java` (fix), `ModelIOConfig.java` (KV cache detection), GGUF architecture files in `nd4j/nd4j-ggml/src/main/java/org/nd4j/ggml/architecture/` (root cause — no present_key/value outputs)
