---
name: vlm-decode-native-path-restored
description: VLM decode routed through generateNative (C++ loop) after fixing causal mask, segment splitting, GEMV bugs
type: project
---

## VLM Decode — Native Path Restored (May 4 2026)

VLM generation (`generate(prefillEmbeddings, ...)`) now routes through `generateNative` (C++ AutoregressiveDecode op).

**Why:** The Java-side loop (`generateSimpleWithKvCacheVlm`) was a TEMPORARY workaround when generateNative had bugs. Those bugs are now fixed:
1. Causal mask pre-unmask (token can attend to its own KV position)
2. Segment splitting (value-key ops isolated into non-capturable segments)
3. Mixed-type GEMV (HALF weight × FLOAT32 activation alignment)
4. Phase check relaxation (vision encoder argTableStable not required)

**How to apply:** The native path does prefill → warmup → freeze → C++ decode loop with zero JNI round-trips. Target: 100+ tok/s. The Java-side loop still exists as reference implementation but is NOT called.

**Status:** Code changes committed but binary NOT YET BUILT. Need CUDA build to activate.

**File:** `nd4j/samediff-llm/src/main/java/org/eclipse/deeplearning4j/llm/generation/GenerationPipeline.java`
- Line 1487-1497: `generate()` calls `generateNative()`
