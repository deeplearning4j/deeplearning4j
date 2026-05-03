---
name: test-results-may2-all-failed
description: "May 2 test results: root causes found for both CPU and CUDA. Last test still failed. Rebuilds in progress with fixes."
type: project
---

## Test Results May 2 2026 — ROOT CAUSES FOUND, REBUILDING

### CPU Qwen3.5 (TestQwen35Pipeline, nd4j-native) — LAST RUN FAILED
- SLOT_BY_SLOT: `' of of.'` — first token=314 (` of`), coherence=1.00 (trivial pass)
- DSP configs: random multilingual garbage, coherence=0.05, DIFFERENT prefill tokens per config
- With optimizer+DSP disabled: STILL token 314 — proves op-level bug
- **ROOT CAUSE:** MKL SDPA causal mask never applied (input[8] not read)
- **FIX APPLIED:** sdpa.cpp three-way bias detection. CPU build in progress.

### CUDA VLM SmolDocling (run-benchmark.sh --tokens 250) — LAST RUN FAILED
- Token IDs: [216, 49229, 0, 0, 0, ...] — all zeros after first 2
- DSP validation: ref=12015 test=0 at step 2
- 50.65 tok/s throughput (perf fine, accuracy garbage)
- **ROOT CAUSE:** Frozen fast-path skips staging buffer sync (ensureAndSyncStagingBuffers)
- **FIX APPLIED:** NativeDynamicShapePlan_cuda.cu. CUDA build waiting on CPU.

### Key Observation About Config Variation
Different DSP configs produced DIFFERENT prefill tokens (12990, 4094, 222376, 37089, 13010) — all garbage. This is because without causal masking, attention weights are non-deterministic (attending to future tokens gives unstable representations). The ssm_a sign fix changed SOME configs' tokens but not SLOT_BY_SLOT, which confirmed the root cause is deeper than GDN.

### Next Verification
After builds complete, run both tests. Expected:
- CPU: First token should NOT be 314 if causal mask fix works
- CUDA: Tokens after first 2 should NOT be zeros if staging fix works
