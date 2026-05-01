---
name: gguf-kv-cache-status-apr27-working
description: GGUF in-graph KV cache working end-to-end on Qwen 0.8B — 1.83 tok/s, dtype fix committed, quality issues remain
type: project
---

# GGUF In-Graph KV Cache: Working (2026-04-27)

## Status: WORKING — 1.83 tok/s, quality needs fixing

The in-graph KV cache path (`generateSimpleWithInGraphKvCache`) runs end-to-end on Qwen 0.8B GGUF without crashes. DSP compiles and shapes freeze correctly. Commit `694c89f631`.

## What works
- Prefill extracts per-layer K/V from graph outputs (`k_rope_{L}`, `v_heads_{L}`)
- Static KV buffers initialized as `[B, maxKvLen, numKVHeads, headDim]` BSHD
- Warmup decode step compiles DSP plan, shapes freeze
- Native AutoregressiveDecode op runs decode loop with plan.execute()
- All segments compile with Triton GPU backend
- No buffer overruns, no crashes

## Key fix: causal mask dtype mismatch
- `buildInGraphDecodeMask`/`buildInGraphCausalMask` hardcoded FLOAT32
- Qwen GGUF model uses FLOAT16 for `_causal_mask` placeholder
- Plan compiled with FLOAT16-sized buffers but received FLOAT32 data → buffer overrun at slot 1599
- Fix: resolve dtype from `decoder.getVariable(causalMaskName).dataType()` and cast

## Remaining issues
1. **Quality is poor**: diversity=0.10, coherence=0.25 — suggests masking or RoPE position wiring may be incorrect
2. **Throughput only 1.83 tok/s** with DSP frozen — should be much faster with replay
3. Plan has ~1683 slots across many segments — may be fragmented

## Key files
- `GenerationPipeline.java` — `generateSimpleWithInGraphKvCache()` method
- `DecoderInputBuilder.java` — `buildInGraphDecodeMask(cachePos, maxKvLen, dtype)`, `buildInGraphCausalMask(prefillLen, maxKvLen, dtype)`
- `autoregressive_decode.cpp` (CPU helper) — native decode loop with in-graph KV (optionalMask bit 4)
- `dot_product_attention_v2.cpp` — rank-0 guard for empty KV inputs during prefill
- `ModelIOConfig.java` — `findKVCacheInputNames()` uses `.endsWith(".key")`/`.endsWith(".value")`

## Architecture
- optionalMask bit 4 = in-graph KV cache (GGUF pattern)
- `planOwnsKvScatter=true` — attention op writes KV in-place, no external scatter
- `positionOffsetExtIdx` / `cachePositionExtIdx` — scalar ext inputs updated per decode step
- C++ `updateCausalMaskCpu` uses `BUILD_SINGLE_SELECTOR(causalMask->dataType(), ...)` — handles FLOAT16
