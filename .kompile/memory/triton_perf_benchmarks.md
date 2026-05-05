---
name: Triton perf benchmarks
description: VLM decode performance — current 60 tok/s monolithic CUDA graph, target 100+, key findings May 4
type: project
---

## SmolDocling VLM Decode, RTX 4090 (Updated May 4 2026)

**Current**: ~60 tok/s with OPTIMAL config (monolithic CUDA graph, all 2742 ops captured)
**Previous composite (Apr 27)**: ~50 tok/s (25 Triton islands + 26 gap units)
**Previous inaccurate**: ~100+ tok/s (user confirmed achievable)
**Accuracy**: CORRECT — coherent doctag + mythic heroes text
**Target**: 100+ tok/s WITH correct output

### Architecture (current)
- 1 segment, ~2742 slots captured into single monolithic CUDA graph
- executeSteadyState() → platformTryFrozenFastPath → seg.exec.replayHandle->replay(stream)
- KV scatter runs POST-graph (autoregressive_decode.cu:817)
- Primary overhead: NVIDIA driver scheduling at ~2-4µs/node × 2742 = 5.5-11ms/replay

### Graph Node Breakdown (2742 total captured nodes)
- ~552 concat (shape assembly, DATADEP prevents freeze — all execute during capture)
- ~300 reshape_no_copy with memcpy (permute→reshape forces ARRAY_NEEDS_COPY)
- ~604 matmul (cuBLAS GEMM, real compute)
- ~90 onnx_multi_head_attention (multi-kernel, probably 5-10 nodes each = 450-900)
- ~526 cast (type conversion kernels)
- ~248 broadcast_to (kernel launches)
- ~120 misc (add, multiply, sigmoid, etc.)
- View ops (permute, expand_dims, reshape) do NOT produce nodes

### Key Rules (STILL VALID)
- Monolithic capture now works WITH correct output (fixes from Apr 28-May 3)
- NEVER compile MATMUL via Triton for M=1 decode (cuBLAS faster)
- ONE change at a time, benchmark + accuracy after EACH

### Optimization Priority (May 4)
1. **Eliminate concat graph nodes** — 552 nodes from shape-assembly. These run only during capture but inflate the graph. Options: make them gap ops, or freeze them pre-capture.
2. **Eliminate reshape_no_copy memcpy nodes** — ~300 nodes from permute→reshape pattern. Option: classify as gap ops during composite schedule.
3. **Reduce attention kernel count** — 90 onnx_multi_head_attention calls produce 450-900 nodes internally. Might be irreducible.
4. **Consider hybrid: compute-only graph** — capture ONLY matmul+attention into islands, everything else as gaps. This is essentially the composite replay approach with better island boundaries.

### Performance Math
- Current 2742 nodes × 3µs/node = 8.2ms scheduling overhead
- If reduced to 1000 nodes: 3ms overhead → 10ms/token → 100 tok/s ✓
- If reduced to 500 nodes: 1.5ms overhead → 8.2ms/token → 122 tok/s ✓
- Goal: eliminate ~1742 non-compute nodes from graph

### Process
Each trial: research → implement → BUILD → BENCHMARK → CHECK ACCURACY → commit-or-revert.
One change at a time. Always test against clean HEAD baseline.
