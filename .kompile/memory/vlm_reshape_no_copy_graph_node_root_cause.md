---
name: VLM reshape_no_copy graph node root cause
description: "604 reshape_no_copy: ~half produce graph nodes due to permute→reshape non-contiguous pattern, ARRAY_NEEDS_COPY flag"
type: project
---

## reshape_no_copy Graph Nodes Root Cause (May 4 2026)

### The Problem
604 reshape_no_copy calls during VLM decode. ~Half are zero-copy (P50=0.5µs). ~Half do actual memcpy (P99=3072µs, mean=228µs). The copy ones produce CUDA graph nodes.

### Root Cause Chain
1. Upstream `permute`/`transpose` produces non-C-contiguous strides (shuffled)
2. `DECLARE_SHAPE_FN(reshape_no_copy)` calls `reshapeNoAlloc()` which fails for non-contiguous strides
3. When reshapeNoAlloc fails, `ARRAY_NEEDS_COPY` flag is set on output shapeInfo
4. DSP's `tryCreateViewForSlot()` returns VIEW_NOT_POSSIBLE (contiguity gate at slotexec.cpp:724)
5. Fresh buffer allocated for output (different from input)
6. At execute time: `arrayNeedsCopy()` true OR `output->dataBuffer() != input->dataBuffer()` → calls `assign()`
7. `assign()` launches CUDA memcpy kernel → captured into graph as a node

### The Pattern
`permute(batch, heads, seq, dim) → reshape_no_copy(batch*heads, seq, dim)` — merging batch and heads after permute always fails contiguity check because permuted strides are [seq*heads*dim, dim, heads*dim, 1].

### Fix Options (from investigation)

**Option A: Make permute produce contiguous output** — shift copy to permute op. Same node count.

**Option B (minimum code change): Mark assign() reshapes as gap ops** — during composite replay schedule construction in `NativeDynamicShapePlan_gpubackend.cu`, check each reshape_no_copy slot's cached shapeInfo for ARRAY_NEEDS_COPY. If set, mark as gap unit (REPLAY_UNIT_GAP). This prevents the assign() from being captured into the graph. Gap execution overhead is minimal for decode-phase seq_len=1 data.

**Option C: Graph transformation pass** — detect permute→reshape_no_copy chains and replace with single `contiguous_reshape` op.

### Key Fact
The concat DATADEP issue (552 nodes) + the reshape_no_copy copy issue (~300 nodes) together account for ~850 graph nodes. Eliminating these from the captured graph would reduce 2742 → ~1890, saving ~2.5ms/step. Combined with other optimizations, this gets us closer to 100 tok/s.

### Key Code Locations
- reshapeNoAlloc: `libnd4j/include/helpers/impl/reshapeNoCopy.cpp:82-94`
- ARRAY_NEEDS_COPY set: `reshape_no_copy.cpp:312`
- tryCreateViewForSlot contiguity gate: `NativeDynamicShapePlan_slotexec.cpp:724`
- Capture loop: `NativeDynamicShapePlan_gpubackend.cu:3113`

**Why:** ~300 unnecessary graph nodes from forced copies after permute.
**How to apply:** Option B (gap-op classification) is cheapest fix. Long-term: fuse permute+reshape into contiguous_reshape.
