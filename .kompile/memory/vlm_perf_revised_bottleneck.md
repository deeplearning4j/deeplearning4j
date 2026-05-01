---
name: vlm-perf-revised-bottleneck
description: "VLM decode perf: bottleneck is ~12ms GPU compute inside single merged CUDA graph. GPU argmax done, KV scatter efficient, scheduling overhead negligible."
type: project
---

## VLM Decode Performance — Revised Understanding (updated 2026-04-29)

### Current State: ~51 tok/s (~19ms/step)

### What's Actually Happening
Composite CUDA graph replay merges ALL Triton islands AND all capture-safe gap ops into
a **single merged CUDA graph** (`mergedGroups=1, gapExec=0us`). One `cudaGraphLaunch`
per step. The `cudaStreamSynchronize` blocks ~12ms waiting for GPU compute to finish.

### CORRECTED misconceptions (from prior analysis):
1. **GPU argmax IS implemented** — argmaxLauncher runs on GPU, only 8B D2H. The "3MB D2H logits" claim was WRONG (only applies to Java fallback javaArgmax path).
2. **KV scatter IS efficient** — kvScatterBatched copies ~135KB/step. The "3MB D2D" claim was WRONG (that's the Java scatterKvToStatic during 2 warmup steps only).
3. **22 graph replays in nsys were from capture phase (exec=1)** — steady state uses 1 merged launch.
4. **CPU scheduling is NOT the bottleneck** — moving KV scatter pre-sync produced 0% improvement (2026-04-29). GPU compute dominates.

### True Optimization Targets
1. **Kernel fusion / op count reduction** — reduce GPU work inside the merged graph
2. **reduce cast/reshape/expand_dims count** — 743 trivial ops add graph nodes
3. **Op-level optimization** — matmul, attention are the heaviest; need fresh --op-timing to identify current hotspots after recent fusions (skip_rms_norm, rms_norm_linear, fused softmax)

### What NOT to target
- CPU scheduling overhead (negligible, proven 2026-04-29)
- GPU argmax (already done)
- KV scatter (already efficient 135KB batched kernel)
- reshape_no_copy copies (load-bearing for cuBLAS, -29% regression when bypassed)
