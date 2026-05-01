---
name: stream-sync-reordering-negative
description: Reordering GPU work around cudaStreamSynchronize on a single stream does NOT help — FIFO serialization prevents overlap
type: feedback
---

Single-stream reordering of GPU work around cudaStreamSynchronize is a dead end for decode loop optimization.

**Why:** CUDA stream FIFO means all operations on one stream execute sequentially. Moving KV scatter, embed lookup, or input_ids update before/after the sync changes nothing — the sync still drains the entire pipeline. D2H and compute do NOT overlap on the same stream.

**Tested 2026-04-28:**
- Moving KV scatter + embed + inputIds BEFORE sync: 48.75 tok/s (regression from 53.28)
- Issuing D2H first, then GPU work, then sync: 50.02 tok/s (still regression)
- Device-pointer kernels (read token from GPU instead of host value) with original ordering: 52.56 tok/s (neutral)
- Baseline (original ordering): 53.28 tok/s

**How to apply:** To actually overlap D2H with GPU compute, you need a SECOND stream + CUDA events. Single-stream tricks are FIFO-bound. Don't attempt single-stream reordering again. The ~13-14ms per-step sync cost IS the graph replay execution time — it's not wasted wait, it's the GPU actually doing the forward pass.
