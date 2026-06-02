---
name: dsp_cache_position_inplace_candidate_rejected_20260528
description: Synthesized cache_position in-place KV candidate built but regressed VLM 250-token throughput
type: project
---

**Task:** Attempt one DSP/VLM optimization candidate for in-place KV/cache_position semantics.
**Change tried:** GenerationPipeline synthesized a cache_position placeholder only for onnx_multi_head_attention ops whose past K/V inputs were real KV cache inputs; DecoderInputBuilder created scalar position/cache inputs; decode requested logits only when in-place KV was active.
**Benchmark:** platform-tests/run-benchmark.sh --tokens 250 --skip-audit, log /tmp/dsp-benchmark-cache-position-inplace-250-20260528.log.
**Result:** steady=55.84 tok/s, lateSteady=62.04 tok/s, native decode 55.8 tok/s. Baseline commit 4a37770da0 was steady=57.48, lateSteady=64.96.
**Decision:** Rejected and reverted exact source edits. Restore Java-only build passed with /home/agibsonccc/dev-apps/mvn/bin/mvn install -DskipTests -pl :samediff-llm.
**Why:** Current SmolDocling/VLM graph does not feed real KV cache buffers directly into onnx_multi_head_attention; MHA optional past inputs were scalar constants and cache concat/repeat happens upstream, so the candidate did not activate for the measured benchmark.
**How to apply:** Do not retry this candidate for the current VLM benchmark unless the graph import/lowering is changed so MHA consumes real static KV buffers.
