---
name: cpu-qwen-iargs-parsing-fix
description: "CPU Qwen fix: iArgs parsing skips kvOutputIndices when hasInGraphKvCache=true (GGUF models)"
type: project
---

## CPU Qwen iArgs Parsing Fix (2026-05-02)

**Root cause:** GGUF models (Qwen3.5) use in-graph KV cache where attention writes KV in-place. Java sends 0 kvOutputIndices for this mode, but the C++ iArgs parser in `autoregressive_decode.cpp` unconditionally skipped `2*numKvPairs` entries for kvOutputIndices. This shifted the stopTokenStartIdx by 56 positions (28 KV pairs * 2), causing `stopTokenCount` to be 62 instead of 2, and the decode loop to stop after just 1 token.

**Fix locations:**
1. `libnd4j/include/ops/declarable/generic/nn/autoregressive_decode.cpp` lines 230-251: Skip kvOutputIndices when `hasInGraphKvCache=true` (bit 4 of optionalMask). Calculate `nextIdx` conditionally.
2. `nd4j/.../AutoregressiveDecode.java` ONNX constructor: Guard kvOutputIndices packing with `(optionalMask & 16) == 0` to match GGUF constructor behavior.
3. `libnd4j/include/ops/declarable/helpers/cpu/autoregressive_decode.cpp`: Gated debug printfs behind `env_isVerbose()` using `#include <system/env_functions.h>`.

**Why:** The GGUF constructor in Java correctly sends 0 kvOutputIndices, but the C++ parser assumed they were always present. The offset error cascaded through GDN indices, conv state indices, and stop token parsing.

**How to apply:** Any future iArgs packing/unpacking changes must account for the `hasInGraphKvCache` flag (bit 4) which changes the layout of kvOutputIndices in the packed integer array.
